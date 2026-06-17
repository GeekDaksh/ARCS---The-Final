"""
Component 5 — Estimation layer: two SEPARATE estimators.

The meteorological message is never perfectly accurate. The frozen physics core
faithfully computes where a shell lands GIVEN the MET it is told, but if the MET
is wrong the shell misses. This module observes the fall of shot, notices it
disagrees with the MET-based prediction, and estimates the corrections — split
into two estimators that must stay separate:

  Estimator A  — GunBiasEstimator: the weapon's fixed mechanical bias (a
                 consistent muzzle-velocity / aim offset). The gun's unchanging
                 personality. Reuses the Phase 1 forgetting-factor scalar-RLS
                 approach (a scalar Kalman filter for a constant parameter).

  Estimator B  — AtmosphericStateEstimator: a Kalman filter whose STATE is the
                 residual effective wind correction the MET got wrong
                 [delta_wind_downrange, delta_wind_cross] (m/s). The day's
                 weather error, which changes engagement to engagement.

They are deliberately kept as separate objects: the gun's personality and the
day's weather are different things and must not bleed into one another.

The estimators call the frozen physics integrator only as a TOOL (to measure
sensitivity and simulate fall of shot) — they never modify it. numpy + stdlib.
"""

import numpy as np

from physics.trajectory import integrate_trajectory


# ===========================================================================
# Sensitivity wiring — obtained empirically from the frozen integrator.
# "If my wind estimate is off by 1 m/s, how many metres does the impact move?"
# Run the integrator at the baseline wind, then again with +1 m/s in each
# horizontal component; the difference in impact is the sensitivity (a 2x2
# Jacobian d(impact_x, impact_y) / d(wind_dr, wind_cross)).
# ===========================================================================
def wind_sensitivity(baseline_wind=(0.0, 0.0, 0.0), delta=1.0, **shot_kwargs):
    """
    Empirical Jacobian H (metres per m/s) of impact position w.r.t. horizontal
    wind, evaluated at baseline_wind, using the frozen trajectory integrator.

    Returns a 2x2 array:
        H[:,0] = d(impact_x, impact_y) / d(wind_downrange)
        H[:,1] = d(impact_x, impact_y) / d(wind_cross)
    """
    w0 = np.asarray(baseline_wind, dtype=float)

    def impact(w):
        r = integrate_trajectory(wind=tuple(w), **shot_kwargs)
        return np.array([r["impact_x"], r["impact_y"]])

    base = impact(w0)
    d_dr = (impact(w0 + np.array([delta, 0.0, 0.0])) - base) / delta
    d_cr = (impact(w0 + np.array([0.0, delta, 0.0])) - base) / delta
    return np.column_stack([d_dr, d_cr])


# ===========================================================================
# Estimator A — Gun bias (Phase 1 forgetting-factor scalar RLS reused).
# ===========================================================================
class _ScalarRLS:
    """
    Single-parameter forgetting-factor RLS == scalar Kalman filter with
    observation matrix H=1. Reused from the Phase 1 StructuredBiasEstimator.

        K = P / (P + R)
        x = x + K (obs - x)        # convex combination -> can never diverge
        P = (1 - K) P / lam        # capped to prevent runaway

    Proven MVUE for a constant parameter from noisy scalar observations
    (Gauss-Markov). The convex-combination update is the Phase 1 stability
    lesson: K in (0,1) means the estimate stays bounded between its prior and
    the observations.
    """

    def __init__(self, init_val=0.0, init_P=100.0, lam=1.0, R=1.0):
        self.x = float(init_val)
        self.P = float(init_P)
        self.lam = float(lam)
        self.R = float(R)
        self.n = 0

    def update(self, observation):
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (observation - self.x)
        self.P = (1.0 - K) * self.P / self.lam
        self.P = min(self.P, 1.0e6)
        self.n += 1
        return self.x

    def get(self):
        return self.x


class GunBiasEstimator:
    """
    Estimator A — the gun's fixed mechanical bias as a CONSTANT impact offset
    (downrange_m, cross_m), the same every shot regardless of weather. Two
    independent forgetting-RLS scalars (the Phase 1 approach). This is the
    weapon's unchanging personality; it does not scale with the flight.
    """

    def __init__(self, init=(0.0, 0.0), init_P=1.0e4, lam=1.0, R=400.0):
        self._dr = _ScalarRLS(init[0], init_P=init_P, lam=lam, R=R)
        self._cr = _ScalarRLS(init[1], init_P=init_P, lam=lam, R=R)

    def update(self, observed_offset):
        """observed_offset: the constant (downrange, cross) impact offset in m
        attributable to the gun (after removing the wind-explained part)."""
        o = np.asarray(observed_offset, dtype=float)
        self._dr.update(o[0])
        self._cr.update(o[1])
        return self.state()

    def state(self):
        return np.array([self._dr.get(), self._cr.get()])

    def uncertainty(self):
        return np.array([self._dr.P, self._cr.P])


# ===========================================================================
# Estimator B — Atmospheric correction (multivariate Kalman filter).
# ===========================================================================
class AtmosphericStateEstimator:
    """
    Kalman filter estimating the residual atmospheric correction the MET
    message got wrong, from observed fall of shot. State: effective
    [delta_wind_downrange, delta_wind_cross] (m/s) (and optionally a
    range-scale term). Fuses the MET-based prediction (imperfect) with the
    observed miss (noisy) to converge on the true effective wind.

    Measurement model. With the MET claiming wind W_met and the true wind being
    W_met + true_delta, a shell aimed using the current estimate x lands with a
    residual miss z (metres) given by the linearisation

        z = H (true_delta - x) + noise

    where H is the empirical wind sensitivity (metres per m/s) from
    wind_sensitivity(). Because the aim already used x, the innovation is z
    itself, and the standard Kalman update drives x -> true_delta:

        S = H P H^T + R
        K = P H^T S^-1
        x = x + K z
        P = (I - K H) P
    """

    def __init__(self, initial_state, initial_covariance,
                 process_noise, measurement_noise):
        self.x = np.array(initial_state, dtype=float).reshape(-1)
        n = self.x.size
        self.P = np.array(initial_covariance, dtype=float).reshape(n, n)
        self.Q = np.array(process_noise, dtype=float).reshape(n, n)
        self.R = np.array(measurement_noise, dtype=float)
        self.n = n
        self._updates = 0

    def predict(self):
        """Engagement/time update: the effective wind correction persists, its
        uncertainty grows slightly (process noise) so the filter stays able to
        track a genuinely changing day."""
        # State transition is identity (correction persists between shots).
        self.P = self.P + self.Q
        return self.x

    def update(self, observed_miss, sensitivity):
        """
        Measurement update. observed_miss: (downrange, lateral) miss in metres
        of a shell aimed with the current estimate. sensitivity: 2x2 Jacobian H
        (metres per m/s) from wind_sensitivity(). Corrects the state estimate
        and shrinks the covariance via the Kalman gain.
        """
        z = np.asarray(observed_miss, dtype=float).reshape(-1)
        H = np.asarray(sensitivity, dtype=float).reshape(z.size, self.n)
        R = self.R
        if np.isscalar(R) or np.ndim(R) == 0:
            R = np.eye(z.size) * float(R)

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        # Innovation = observed miss (the aim used the current estimate, so the
        # predicted miss under a perfect estimate is zero).
        self.x = self.x + K @ z
        # Joseph-free standard covariance update; symmetrise for stability.
        self.P = (np.eye(self.n) - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        self._updates += 1
        return self.x

    def state(self):
        return self.x.copy()

    def uncertainty(self):
        return self.P.copy()
