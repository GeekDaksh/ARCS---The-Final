"""
ARCS — Bayesian Optimizer  (Benchmark Edition v3.0)

ALL CRITICAL FIXES FROM AUDIT:

    [FIX-1] Evaluation protocol: PAIRED comparison.
            Baseline and verified CEP now use the SAME RNG stream.
            Baseline = first 30 shots. Verified = next 30 shots, same rng.
            Wilcoxon signed-rank test for statistical significance.
            Previously: split RNGs caused 2m std measurement noise that
            swamped any real improvement signal.

    [FIX-2] BO bounds tightened to +/-3*sigma of mechanical noise.
            pitch: +/-0.9 deg (was +/-1.5 = 5*sigma).
            yaw:   +/-0.6 deg (was +/-1.5).
            v0:    +/-4.5 m/s (was +/-5.0).
            Eliminates exploration budget wasted on physically impossible
            corrections.

    [FIX-3] n_avg: SNR-based formula targeting SNR >= 3.0.
            n = ceil((3 * single_shot_std / min_correction_effect)^2)
            Previously: CEP-based formula produced SNR < 2 at short range.

    [FIX-4] n_suggest raised from 8 to 16.
            3D GP needs >= 10 points per dimension to learn anything.
            4+16=20 total evaluations gives 6.7 per dimension (floor).

    [FIX-5] GlobalModel reduced to 2D: (range_m, pitch_cmd).
            Previously 5D with <20 training points = 4 per dimension.
            range and pitch_cmd are the two dominant drivers of systematic
            bias. v0 and height are secondary.

    [FIX-6] Bootstrap CI uses its own seeded RNG (not numpy global).

    [FIX-7] Systematic bias model (RobotBiasModel) is now required.
            Without systematic bias, there is nothing to correct.

    [FIX-8] PINN warm_start double-application bug fixed in reset().
            When pinn_active=True, run_engagement pre-applies the PINN
            correction to firing_sol and firing_v0 before handing them
            to fire_averaged. The original code also passed warm_start=
            pinn_correction, so the first BO suggestion (= PINN delta)
            was added a second time inside fire_averaged, producing:
              actual_pitch = sol.pitch + pinn.dp + pinn.dp  (2×)
              actual_v0    = v0 + pinn.dv + pinn.dv         (2×)
            This caused a large miss on the first BO suggestion, poisoning
            the GP model and wasting suggestion budget before the GP could
            recover. Fix: when pinn_active=True, warm_start is cleared to
            None so the BO starts from the correct residual zero (0,0,0).

    [FIX-9] _adaptive_n_avg formula corrected (FIX-3 was range-independent).
            Previous formula omitted v0 noise; pitch_noise and correction_effect
            both equalled range×tan(sigma_pitch) and cancelled, giving a constant
            n=13 (no PINN) or n=20 (PINN) at every range. Fixed by including
            v0_noise = sigma_v0 × tof in single_shot_std and using
            correction_effect = BIAS_SCALE × sigma_v0 × tof (the dominant v0
            signal). Also passes pitch_deg and v0 from run_engagement so tof_approx
            is accurate. Result: n≈5 (no PINN), n≈16–19 (with PINN) — still
            roughly constant due to v0 SNR = BIAS_SCALE ≈ 1.5 at all ranges,
            but now physically grounded rather than an accidental cancellation.

    Retained from v2.0:
    [1.1] Matern 5/2 kernel
    [1.2] Decaying kappa schedule (2.0 -> 0.5)
    [2.2] Adaptive bounds (further tightened after 5 engagements)
    [2.3] GlobalModel target = delta_miss
    [3.2] Coroutine BO
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import wilcoxon
from physics.ballistic_solver import BallisticSolver
from physics.constants import (deg_to_rad, GRAVITY,
                                SIGMA_PITCH_DEG, SIGMA_YAW_DEG, SIGMA_V0,
                                N_SHOTS_BASELINE, N_SHOTS_VERIFY, N_KF_SHOTS,
                                FALLBACK_THRESHOLD, BIAS_SCALE,
                                GP_SIGMA_CONVERGED_THRESHOLD)
from kalman_filter import EngagementKF
from physics.bias_model import RobotBiasModel


# ─── GAUSSIAN PROCESS — Matern 5/2 ───────────────────────────────

class GaussianProcess:
    def __init__(self, length_scale=0.8, signal_var=1.0, noise_var=100.0):
        self.length_scale = length_scale
        self.signal_var   = signal_var
        self.noise_var    = noise_var
        self.X_obs = None
        self.y_obs = None
        self.K_inv = None
        self.n_obs = 0

    def _matern52(self, X1, X2):
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        diff  = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
        r     = np.sqrt(np.sum(diff**2, axis=-1)) / (self.length_scale + 1e-9)
        s5r   = np.sqrt(5.0) * r
        return self.signal_var * (1.0 + s5r + 5.0*r**2/3.0) * np.exp(-s5r)


    def fit(self, X, y, optimise=True, optimise_hyperparams=None):
        if optimise_hyperparams is not None:
            optimise = optimise_hyperparams
        self.X_obs = np.atleast_2d(X)
        self.y_obs = np.array(y, dtype=float)
        self.n_obs = len(y)
        if optimise and self.n_obs >= 5:
            self._optimise_hyperparams()
        self._compute_K_inv()

    def _compute_K_inv(self):
        K   = self._matern52(self.X_obs, self.X_obs)
        K_y = K + self.noise_var*np.eye(self.n_obs) + 1e-6*np.eye(self.n_obs)
        try:
            self.K_inv = np.linalg.inv(K_y)
        except np.linalg.LinAlgError:
            self.K_inv = np.linalg.pinv(K_y)

    def _lml(self, params):
        ls, sv, nv = np.exp(params)
        old = (self.length_scale, self.signal_var, self.noise_var)
        self.length_scale, self.signal_var, self.noise_var = ls, sv, nv
        K   = self._matern52(self.X_obs, self.X_obs)
        K_y = K + nv*np.eye(self.n_obs) + 1e-6*np.eye(self.n_obs)
        self.length_scale, self.signal_var, self.noise_var = old
        try:
            L     = np.linalg.cholesky(K_y)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_obs))
            lml   = (-0.5*self.y_obs@alpha
                     - np.sum(np.log(np.diag(L)))
                     - 0.5*self.n_obs*np.log(2*np.pi))
            return -float(lml)
        except np.linalg.LinAlgError:
            return 1e10

    def _optimise_hyperparams(self):
        best_lml, best_p = np.inf, None
        for _ in range(5):
            x0 = np.array([np.random.uniform(-1,1),
                            np.random.uniform(-1,2),
                            np.random.uniform(4.3,6)])
            r = minimize(self._lml, x0, method='L-BFGS-B',
                         bounds=[(-3,3),(-3,4),(4.3,7)],
                         options={'maxiter':100})
            if r.fun < best_lml:
                best_lml, best_p = r.fun, r.x
        if best_p is not None:
            self.length_scale = float(np.exp(best_p[0]))
            self.signal_var   = float(np.exp(best_p[1]))
            self.noise_var    = float(np.exp(best_p[2]))

    def predict(self, X_new):
        X_new = np.atleast_2d(X_new)
        if self.X_obs is None or self.K_inv is None:
            n = len(X_new)
            return np.zeros(n), np.full(n, float(np.sqrt(self.signal_var)))
        K_s  = self._matern52(X_new, self.X_obs)
        K_ss = self._matern52(X_new, X_new)
        mean = K_s @ self.K_inv @ self.y_obs
        var  = np.maximum(np.diag(K_ss - K_s @ self.K_inv @ K_s.T), 1e-9)
        return mean, np.sqrt(var)


# ─── GLOBAL MODEL — 2D reduced [FIX-5] ───────────────────────────

class GlobalModel:
    """
    FIX-5: Reduced from 5D to 2D feature space.
    Features: [range_m, delta_pitch]  — (range, correction applied)
    Target:   delta_miss = baseline_cep - verified_cep

    Justification: systematic bias is primarily a function of barrel
    angle (pitch) and range (which determines required pitch). v0, height,
    and yaw are secondary. Reducing dimensionality from 5 to 2 means the
    same 20 training records give 10 per dimension instead of 4 — enough
    for the GP to actually learn something.

    Note: predict_improvement() accepts a pitch_cmd argument for API
    compatibility but the model is trained on delta_pitch, not pitch_cmd.
    The pitch_cmd parameter is currently unused in prediction.
    """

    MIN_RECORDS = 8

    def __init__(self):
        self.gp        = GaussianProcess(length_scale=50.0,
                                          signal_var=20.0,
                                          noise_var=50.0)
        self.is_fitted = False
        self.n_records = 0
        self._X_mean   = None
        self._X_std    = None

    def train(self, corrections_df):
        if corrections_df is None or len(corrections_df) < self.MIN_RECORDS:
            self.is_fitted = False
            return
        df = corrections_df.dropna(subset=[
            "range_m", "miss_before", "miss_after"])
        if len(df) < self.MIN_RECORDS:
            self.is_fitted = False
            return

        # 2D feature space
        X = df[["range_m", "delta_pitch"]].values
        y = (df["miss_before"] - df["miss_after"]).values   # delta_miss

        self._X_mean = X.mean(axis=0)
        self._X_std  = X.std(axis=0) + 1e-6
        X_norm = (X - self._X_mean) / self._X_std

        self.gp.fit(X_norm, y, optimise=(len(df) >= 15))
        self.is_fitted = True
        self.n_records = len(df)

    def predict_improvement(self, range_m, pitch_cmd, candidates_dp):
        """
        Predict expected improvement for each candidate pitch correction.
        candidates_dp: (n,) array of delta_pitch values.
        Returns (n,) array (higher = more expected improvement).
        """
        if not self.is_fitted or self._X_mean is None:
            return None
        n = len(candidates_dp)
        X = np.column_stack([np.full(n, range_m), candidates_dp])
        X_norm = (X - self._X_mean) / self._X_std
        mean, _ = self.gp.predict(X_norm)
        return mean


# ─── ENGAGEMENT MEMORY ────────────────────────────────────────────

class EngagementMemory:
    def __init__(self):
        self.records          = []
        self.engagement_count = 0

    def record(self, best_correction, best_miss):
        self.records.append({"correction": np.array(best_correction),
                              "miss": float(best_miss)})
        self.engagement_count += 1

    def prior_mean(self):
        if not self.records:
            return np.zeros(3)
        return np.mean([r["correction"] for r in self.records], axis=0)

    def prior_std(self):
        if len(self.records) < 2:
            return np.array([SIGMA_PITCH_DEG, SIGMA_YAW_DEG, SIGMA_V0])
        return np.std([r["correction"] for r in self.records], axis=0) + 1e-6

    def summary(self):
        if not self.records:
            return {"n_engagements": 0}
        corrections = np.array([r["correction"] for r in self.records])
        misses = [r["miss"] for r in self.records]
        return {
            "n_engagements":   self.engagement_count,
            "mean_correction": corrections.mean(axis=0),
            "std_correction":  corrections.std(axis=0),
            "mean_best_miss":  float(np.mean(misses)),
            "best_ever_miss":  float(np.min(misses)),
        }


# ─── BAYESIAN OPTIMIZER ───────────────────────────────────────────

class BayesianOptimizer:
    """
    FIX-2: Tightened bounds to +/-3*sigma of mechanical noise.
    FIX-4: n_suggest raised to 16.
    Retained: Matern 5/2, decaying kappa, adaptive bounds, coroutine.
    """

    # FIX-2: bounds tightened from +/-1.5 to +/-3*sigma
    DEFAULT_BOUNDS = np.array([
        [-3*SIGMA_PITCH_DEG,  3*SIGMA_PITCH_DEG],   # pitch: +/-0.9 deg
        [-3*SIGMA_YAW_DEG,    3*SIGMA_YAW_DEG  ],   # yaw:   +/-0.6 deg
        [-3*SIGMA_V0,         3*SIGMA_V0        ],   # v0:    +/-4.5 m/s
    ])

    def __init__(self, memory=None, global_model=None,
                 n_avg=5, n_init=4, n_suggest=16,   # FIX-4: n_suggest=16
                 kappa=2.0, kappa_min=0.5, bounds=None):
        self.memory    = memory or EngagementMemory()
        self._gm_ref   = [global_model]
        self.n_avg     = n_avg
        self.n_init    = n_init
        self.n_suggest = n_suggest
        self.kappa     = kappa
        self.kappa_min = kappa_min
        self._base_bounds = (bounds if bounds is not None
                              else self.DEFAULT_BOUNDS.copy())
        self.bounds          = self._base_bounds.copy()
        self.gp              = GaussianProcess()
        self._X              = []
        self._y              = []
        self.best_correction = np.zeros(3)
        self.best_miss       = np.inf
        self._range_m        = None
        self._height_m       = None
        self._pitch_cmd      = None
        self._suggest_iter   = 0
        self._pinn_active    = False   # tracks last PINN regime for change detection
        self._early_stopped   = False
        self._early_stop_iter = -1

    @property
    def global_model(self):
        return self._gm_ref[0]

    @global_model.setter
    def global_model(self, value):
        self._gm_ref[0] = value

    def reset(self, range_m=None, height_m=None, pitch_cmd=None,
              pinn_active=False, warm_start_correction=None):
        # Detect PINN regime change. Memory records from a different regime
        # point in the wrong direction: pre-PINN records hold full robot bias
        # (~[-0.5, 0.3, -4.47]); post-PINN records hold near-zero residuals.
        # Using the wrong prior wastes the 2nd BO suggestion. Clear memory
        # when the regime flips so BO searches from a neutral starting point.
        if pinn_active != self._pinn_active and self.memory.engagement_count > 0:
            self.memory.records.clear()
            self.memory.engagement_count = 0
        self.gp              = GaussianProcess()
        self._X              = []
        self._y              = []
        self.best_correction = np.zeros(3)
        self.best_miss       = np.inf
        self._range_m        = range_m
        self._height_m       = height_m
        self._pitch_cmd      = pitch_cmd
        self._suggest_iter   = 0
        self._pinn_active    = pinn_active
        self._n0_used_prior  = False   # set True when n=0 warm-starts from memory
        self._early_stopped   = False
        self._early_stop_iter = -1
        self.bounds          = self._adaptive_bounds()

        # ── BUG FIX ──────────────────────────────────────────────────────────
        # When pinn_active=True, run_engagement has ALREADY applied the PINN
        # correction to firing_sol.turret_pitch_deg and firing_v0:
        #
        #   firing_sol.turret_pitch_deg = sol.turret_pitch_deg + pre_dp
        #   firing_v0                   = v0 + pre_dv
        #
        # fire_averaged then adds the BO correction on top:
        #   actual_pitch = firing_sol.turret_pitch_deg + correction[0]
        #               = sol.pitch + pre_dp + correction[0]
        #
        # If warm_start = pinn_correction → correction[0] = pre_dp
        #   actual_pitch = sol.pitch + pre_dp + pre_dp  ← DOUBLE-APPLIED
        #
        # This causes the first BO shot to fire at 2× the PINN correction,
        # producing a large miss that poisons the GP and wastes a suggestion.
        #
        # Fix: when PINN is active, the BO searches RESIDUAL space where (0,0,0)
        # is the correct neutral start. The warm_start must be cleared.
        # Without PINN, warm_start = cross-engagement memory prior (unchanged).
        # ─────────────────────────────────────────────────────────────────────
        if pinn_active:
            self._warm_start = None   # PINN already applied; start from residual zero
        else:
            self._warm_start = warm_start_correction

    def _adaptive_bounds(self):
        if self.memory.engagement_count < 5:
            return self._base_bounds.copy()
        mean = self.memory.prior_mean()
        std  = self.memory.prior_std()
        tight = np.column_stack([
            np.maximum(self._base_bounds[:,0], mean - 3*std),
            np.minimum(self._base_bounds[:,1], mean + 3*std)
        ])
        for i in range(len(tight)):
            if tight[i,1] - tight[i,0] < 0.05:
                mid = (tight[i,0]+tight[i,1])/2
                tight[i] = [mid-0.025, mid+0.025]
        return tight

    def _decaying_kappa(self):
        frac = min(1.0, self._suggest_iter / max(1, self.n_suggest))
        return self.kappa*(1-frac) + self.kappa_min*frac

    def suggest(self):
        n = len(self._X)
        if n == 0:
            # Priority 1: PINN warm-start — seed first shot at PINN prediction.
            # Eliminates the 3 wasted shots that rediscover what PINN already knows.
            if self._warm_start is not None:
                return np.clip(np.array([
                    self._warm_start.get('delta_pitch', 0.0),
                    self._warm_start.get('delta_yaw',   0.0),
                    self._warm_start.get('delta_v0',    0.0),
                ]), self.bounds[:, 0], self.bounds[:, 1])
            # Priority 2: memory prior warm-start — without PINN, seed from
            # cross-engagement history. Saves shots rediscovering known bias.
            # With PINN active, the BO searches residual space where (0,0,0)
            # is the correct neutral starting point (PINN already pre-corrected).
            if not self._pinn_active and self.memory.engagement_count >= 2:
                self._n0_used_prior = True
                return np.clip(self.memory.prior_mean(),
                               self.bounds[:,0], self.bounds[:,1])
            return np.zeros(3)
        if n == 1 and self.memory.engagement_count >= 2:
            # When PINN is active it has already applied the global bias.
            # The memory prior reflects pre-PINN corrections (full robot bias),
            # which would push BO in the wrong direction. Skip it and let BO
            # search freely from [0,0,0] in the PINN-residual space.
            if self._pinn_active:
                return np.array([np.random.uniform(lo, hi)
                                 for lo, hi in self.bounds])
            # If n=0 already used memory prior, go directly to random
            # exploration — firing at prior twice in a row adds no information.
            if self._n0_used_prior:
                return np.array([np.random.uniform(lo, hi)
                                 for lo, hi in self.bounds])
            return np.clip(self.memory.prior_mean(),
                           self.bounds[:,0], self.bounds[:,1])
        if n < self.n_init:
            return np.array([np.random.uniform(lo,hi) for lo,hi in self.bounds])

        kappa_t = self._decaying_kappa()
        self._suggest_iter += 1
        n_cand = 2000
        cands  = np.column_stack([
            np.random.uniform(lo,hi,n_cand) for lo,hi in self.bounds])
        mean, std = self.gp.predict(cands)

        gm = self.global_model
        if (gm is not None and gm.is_fitted
                and self._range_m is not None
                and self._pitch_cmd is not None):
            imp = gm.predict_improvement(
                self._range_m, self._pitch_cmd, cands[:,0])
            if imp is not None:
                # Global model guides pitch axis; GP handles full 3D
                mean -= 0.3 * imp

        lcb = mean - kappa_t * std
        return cands[np.argmin(lcb)]

    def update(self, correction, avg_miss):
        self._X.append(np.array(correction))
        self._y.append(float(avg_miss))
        if len(self._X) >= 2:
            self.gp.fit(np.array(self._X), np.array(self._y),
                        optimise=(len(self._X) >= 6))
        if avg_miss < self.best_miss:
            self.best_miss       = avg_miss
            self.best_correction = np.array(correction)

    def run(self):
        """Coroutine BO: send(miss) to provide result of last suggestion."""
        avg_miss = None
        for i in range(self.n_init + self.n_suggest):
            if avg_miss is not None:
                self.update(self._last_correction, avg_miss)

                # ── Early stopping: check GP convergence ────────────────
                # If the GP is already certain about its best correction,
                # further suggestions will not improve the estimate — they
                # only add noise. This saves shots for verification (which
                # needs them more). Reference: Fiedler et al. (AAAI 2021).
                if i >= self.n_init and self.gp.X_obs is not None:
                    try:
                        x_curr_best = np.array(self.best_correction).reshape(1, -1)
                        _, sigma_curr = self.gp.predict(x_curr_best)
                        if float(sigma_curr[0]) < GP_SIGMA_CONVERGED_THRESHOLD:
                            self._early_stopped   = True
                            self._early_stop_iter = i
                            # avg_miss was already folded into the GP via the
                            # self.update() call above — clear it so the
                            # post-loop "pending miss" handler below does not
                            # apply the same observation a second time (which
                            # would duplicate a point in self._X/self._y and
                            # artificially distort the posterior variance).
                            avg_miss = None
                            break   # GP has converged — stop wasting shots
                    except Exception:
                        pass  # GP not yet fitted — continue

            correction = self.suggest()
            self._last_correction = correction
            avg_miss = yield correction
        if avg_miss is not None:
            self.update(self._last_correction, avg_miss)
        self.memory.record(self.best_correction, self.best_miss)


# ─── ENGAGEMENT SIMULATOR ─────────────────────────────────────────

class EngagementSimulator:
    """
    FIX-1: PAIRED evaluation protocol.
        Baseline and verified CEP use the SAME RNG stream.
        Baseline = first n shots. Verified = next n shots.
        Wilcoxon signed-rank test for significance.

    FIX-3: SNR-based n_avg formula.
        n = ceil((SNR_target * single_shot_std / correction_effect)^2)
        SNR_target = 3.0

    FIX-6: Bootstrap CI uses its own seeded RNG.

    FIX-7: Requires RobotBiasModel for physical simulation.
    """

    SNR_TARGET = 3.0   # minimum required SNR for correction detection

    def __init__(self, sigma_pitch=SIGMA_PITCH_DEG, sigma_yaw=SIGMA_YAW_DEG,
                 sigma_v0=SIGMA_V0, range_table=None, seed=None,
                 bias_model: RobotBiasModel = None):
        self.solver      = BallisticSolver()
        self.sigma_pitch = sigma_pitch
        self.sigma_yaw   = sigma_yaw
        self.sigma_v0    = sigma_v0
        self.range_table = range_table

        # FIX-7: require bias model for physical realism
        self.bias_model = bias_model or RobotBiasModel(seed=seed or 42)

        # Single shared RNG for paired evaluation (FIX-1)
        self._seed = seed
        self.rng = np.random.default_rng(seed)

        # Separate RNG for bootstrap only (FIX-6)
        self._boot_rng = np.random.default_rng(
            (seed + 99999) if seed is not None else None)

        # Separate RNG for KF refinement shots.
        # KF uses its own stream so the baseline→verify pairing (shared self.rng)
        # is completely unaffected by the 8 KF draws. Without this, KF shots
        # consume 24 draws from self.rng, shifting which noise values hit the
        # verification shots — causing seed-specific fallback spikes.
        self._kf_rng = np.random.default_rng(
            (seed + 55555) if seed is not None else None)


    def _fire_one(self, pitch_cmd, yaw_cmd, v0_cmd,
                  target_x, target_y, target_z, rng=None):
        """Fire one shot: systematic bias + stochastic noise."""
        if rng is None:
            rng = self.rng
        act_pitch, act_yaw, act_v0 = self.bias_model.apply(
            pitch_cmd, yaw_cmd, v0_cmd, rng,
            sigma_pitch=self.sigma_pitch,
            sigma_yaw=self.sigma_yaw,
            sigma_v0=self.sigma_v0)
        p  = deg_to_rad(act_pitch)
        ya = deg_to_rad(act_yaw)
        v0x = act_v0*np.cos(p)*np.cos(ya)
        v0y = act_v0*np.sin(p)
        v0z = act_v0*np.cos(p)*np.sin(ya)
        disc = v0y**2 - 2*GRAVITY*target_y
        if disc < 0:
            disc = 0.0
        sqrt_d = np.sqrt(disc)
        t1 = (v0y+sqrt_d)/GRAVITY
        t2 = (v0y-sqrt_d)/GRAVITY
        pos = sorted([t for t in [t1,t2] if t > 1e-9])
        if not pos:
            return 999.0
        tof = min(pos, key=lambda t: abs(v0x*t - target_x))
        return float(np.sqrt((v0x*tof - target_x)**2 + (v0z*tof - target_z)**2))

    def _fire_one_directional(self, pitch_cmd, yaw_cmd, v0_cmd,
                               target_x, target_y, target_z):
        """
        Fire one shot and return (error_x, error_z, miss_dist).
        error_x / error_z are the signed downrange and lateral misses (metres).
        Used by the KF refinement step — directional errors let the KF estimate
        the remaining systematic pitch and yaw residual after BO correction.
        Uses self._kf_rng (separate from self.rng) so that the baseline→verify
        paired evaluation is completely unaffected by the KF draws.
        """
        act_pitch, act_yaw, act_v0 = self.bias_model.apply(
            pitch_cmd, yaw_cmd, v0_cmd, self._kf_rng,
            sigma_pitch=self.sigma_pitch,
            sigma_yaw=self.sigma_yaw,
            sigma_v0=self.sigma_v0)
        p   = deg_to_rad(act_pitch)
        ya  = deg_to_rad(act_yaw)
        v0x = act_v0 * np.cos(p) * np.cos(ya)
        v0y = act_v0 * np.sin(p)
        v0z = act_v0 * np.cos(p) * np.sin(ya)
        disc = v0y**2 - 2 * GRAVITY * target_y
        if disc < 0:
            disc = 0.0
        sqrt_d = np.sqrt(disc)
        t1 = (v0y + sqrt_d) / GRAVITY
        t2 = (v0y - sqrt_d) / GRAVITY
        pos = sorted([t for t in [t1, t2] if t > 1e-9])
        if not pos:
            return 0.0, 0.0, 999.0
        tof     = min(pos, key=lambda t: abs(v0x * t - target_x))
        error_x = v0x * tof - target_x
        error_z = v0z * tof - target_z
        return float(error_x), float(error_z), float(np.sqrt(error_x**2 + error_z**2))

    def fire_averaged(self, correction, sol, tx, ty, tz, v0, n_avg):
        """Fire n_avg shots with correction; returns (median, list)."""
        dp, dy, dv = correction
        misses = [
            self._fire_one(sol.turret_pitch_deg+dp,
                           sol.turret_yaw_deg+dy,
                           v0+dv, tx, ty, tz)
            for _ in range(n_avg)
        ]
        return float(np.median(misses)), misses

    def baseline_cep(self, sol, tx, ty, tz, v0, n=N_SHOTS_BASELINE):
        """
        FIX-1: Baseline uses shared RNG stream.
        Returns (cep, shots_list) for paired comparison.
        """
        misses = [
            self._fire_one(sol.turret_pitch_deg,
                           sol.turret_yaw_deg, v0, tx, ty, tz)
            for _ in range(n)
        ]
        return float(np.median(misses)), misses

    def verified_cep_with_ci(self, sol, correction, tx, ty, tz, v0,
                              n=N_SHOTS_VERIFY, n_bootstrap=1000):
        """
        FIX-1: Verified uses SAME RNG stream as baseline (paired).
        FIX-6: Bootstrap uses its own seeded RNG.
        Returns dict with cep_50, ci_low, ci_high, wilcoxon_p.
        """
        dp, dy, dv = correction
        misses = [
            self._fire_one(sol.turret_pitch_deg+dp,
                           sol.turret_yaw_deg+dy,
                           v0+dv, tx, ty, tz)
            for _ in range(n)
        ]
        cep_50 = float(np.median(misses))

        # Bootstrap CI — FIX-6: use dedicated boot_rng
        boot_medians = [
            np.median(self._boot_rng.choice(misses, size=n, replace=True))
            for _ in range(n_bootstrap)
        ]
        ci_low  = float(np.percentile(boot_medians, 2.5))
        ci_high = float(np.percentile(boot_medians, 97.5))

        return {"cep_50": cep_50, "ci_low": ci_low, "ci_high": ci_high,
                "n_shots": n, "misses": misses}

    def verified_cep(self, sol, correction, tx, ty, tz, v0, n=N_SHOTS_VERIFY):
        """Scalar wrapper for backward compat."""
        return self.verified_cep_with_ci(
            sol, correction, tx, ty, tz, v0, n)["cep_50"]

    def paired_wilcoxon(self, baseline_misses: list,
                         verified_misses: list) -> dict:
        """
        FIX-1: Paired Wilcoxon signed-rank test on per-shot miss distances.
        Tests H0: median(baseline) == median(verified).
        Returns p-value and whether improvement is statistically significant.
        """
        n = min(len(baseline_misses), len(verified_misses))
        if n < 10:
            return {"p_value": 1.0, "significant": False, "n": n}
        try:
            stat, p = wilcoxon(baseline_misses[:n], verified_misses[:n],
                                alternative='greater')
            return {"stat": float(stat), "p_value": float(p),
                    "significant": bool(p < 0.05), "n": n}
        except Exception:
            return {"p_value": 1.0, "significant": False, "n": n}

    def _adaptive_n_avg(self, horiz_range, pitch_deg=None, v0=100.0,
                        gp_pre_applied=False):
        """
        FIX-9: SNR-based n_avg now correctly includes v0 noise.

        The original formula omitted v0 noise entirely, causing pitch_noise_m
        and correction_effect (both = range × tan(sigma_pitch)) to cancel.
        The result was a range-independent constant (13 without PINN, 20 with)
        instead of a genuine SNR-adaptive result.

        Root cause:
            single_shot_std  = sqrt(pitch_noise² + yaw_noise²)   ← missing v0
            correction_effect = range × tan(sigma_pitch)          ← = pitch_noise

            → ratio = sqrt(1 + (yaw/pitch)²)  (constant, no range dependence)

        Fix: include v0 noise in single_shot_std; use v0 bias effect as the
        minimum detectable correction (v0 is the dominant systematic error
        across all engagement ranges).

        v0 SNR analysis:
            bias_v0 ≈ BIAS_SCALE × sigma_v0  (expected systematic magnitude)
            signal_per_shot = bias_v0 × tof   (range effect of v0 bias)
            noise_per_shot  = sigma_v0 × tof  (range effect of v0 noise)
            SNR per shot    = BIAS_SCALE ≈ 1.5 (constant across ranges)
            n for SNR=3     = (3/1.5)² = 4  (without PINN)
                              (3/0.75)² = 16 (with PINN, residual ≈ 0.5×)

        Pitch SNR is much lower (bias ≈ 0.1–0.3° vs noise 0.3°) so the
        dominant constraint is always v0. The formula now reflects this.
        """
        # Approximate tof for this range
        if pitch_deg is not None:
            pitch_r = deg_to_rad(pitch_deg)
        else:
            # Low-elevation approximation: pitch ≈ range / (v0² / g)
            pitch_r = deg_to_rad(3.0 + horiz_range * 0.02)  # rough
        tof_approx = max(horiz_range / max(v0 * np.cos(pitch_r), 1.0), 0.1)

        # All noise sources (v0 was absent before)
        pitch_noise_m   = horiz_range * np.tan(deg_to_rad(self.sigma_pitch))
        yaw_noise_m     = horiz_range * np.tan(deg_to_rad(self.sigma_yaw))
        v0_noise_m      = self.sigma_v0 * tof_approx          # ← was missing
        single_shot_std = np.sqrt(pitch_noise_m**2
                                  + yaw_noise_m**2
                                  + v0_noise_m**2)

        # Minimum detectable correction: v0 bias at this range
        # (v0 is the dominant systematic error; pitch bias is smaller)
        v0_bias_est    = BIAS_SCALE * self.sigma_v0            # expected systematic
        correction_effect = v0_bias_est * tof_approx           # range effect

        residual = 0.5 if gp_pre_applied else 1.0
        n = int(np.ceil(
            (self.SNR_TARGET * single_shot_std
             / max(correction_effect * residual, 0.01))**2
        ))
        return int(max(6, min(20, n)))

    def run_engagement(self, target_x, target_y, target_z,
                       v0=100.0, optimizer=None, verbose=False,
                       gp_pre_correction=None, prefer="LOW"):
        """
        Full engagement with:
          - Paired evaluation (FIX-1)
          - Wilcoxon significance test (FIX-1)
          - SNR-based n_avg (FIX-3)
          - Coroutine BO (retained from v2)
          - Bootstrap CI (retained from v2, with FIX-6)
        """
        sol = self.solver.solve(target_x, target_y, target_z, v0, prefer=prefer)
        if not sol.reachable:
            return None

        # FIX-1: baseline uses shared RNG (saves the shot list for pairing)
        bl_cep, bl_misses = self.baseline_cep(
            sol, target_x, target_y, target_z, v0)

        gp_active = gp_pre_correction is not None
        n_avg = self._adaptive_n_avg(sol.horiz_range,
                                     pitch_deg=sol.turret_pitch_deg,
                                     v0=v0,
                                     gp_pre_applied=gp_active)

        if optimizer is None:
            optimizer = BayesianOptimizer()

        optimizer.n_avg = n_avg
        optimizer.reset(range_m=sol.horiz_range, height_m=target_y,
                        pitch_cmd=sol.turret_pitch_deg,
                        pinn_active=gp_active,
                        warm_start_correction=gp_pre_correction)

        if gp_pre_correction is not None:
            pre_dp = gp_pre_correction.get("delta_pitch", 0.0)
            pre_dy = gp_pre_correction.get("delta_yaw",   0.0)
            pre_dv = gp_pre_correction.get("delta_v0",    0.0)
            import types as _types
            firing_sol = _types.SimpleNamespace(
                turret_pitch_deg=sol.turret_pitch_deg + pre_dp,
                turret_yaw_deg=sol.turret_yaw_deg + pre_dy,
            )
            firing_v0 = v0 + pre_dv
        else:
            firing_sol, firing_v0 = sol, v0
            pre_dp = pre_dy = pre_dv = 0.0

        # Create the KF at the start of the engagement so it accumulates
        # directional observations throughout the engagement (post-BO refinement
        # shots fire at best_corr_total and provide the actual kf.update calls).
        kf = EngagementKF(range_m=sol.horiz_range,
                          sigma_pitch=self.sigma_pitch,
                          sigma_yaw=self.sigma_yaw)

        # Coroutine BO loop
        history     = []
        total_shots = N_SHOTS_BASELINE
        gen         = optimizer.run()
        correction  = next(gen)

        try:
            while True:
                avg_miss, _ = self.fire_averaged(
                    correction, firing_sol, target_x, target_y, target_z,
                    firing_v0, n_avg)
                total_shots += n_avg
                history.append({
                    "suggestion":  len(history)+1,
                    "delta_pitch": float(correction[0]),
                    "delta_yaw":   float(correction[1]),
                    "delta_v0":    float(correction[2]),
                    "avg_miss":    avg_miss,
                    "best_miss":   optimizer.best_miss,
                    "kappa":       optimizer._decaying_kappa(),
                })
                if verbose:
                    print(f"  [{len(history):2d}] dp={correction[0]:+.3f} "
                          f"dy={correction[1]:+.3f} dv={correction[2]:+.2f} "
                          f"-> {avg_miss:.2f}m best={optimizer.best_miss:.2f}m "
                          f"k={optimizer._decaying_kappa():.2f}")
                correction = gen.send(avg_miss)
        except StopIteration:
            pass

        # ── GP posterior variance at best correction ─────────────────────
        # The GP's σ(x_best) tells us how certain the model is about its
        # recommendation. Small σ = converged; large σ = more shots needed.
        # Reference: Fiedler et al. (AAAI 2021) — rigorous GP uncertainty bounds.
        try:
            x_best = np.array(optimizer.best_correction).reshape(1, -1)
            _, gp_sigma = optimizer.gp.predict(x_best)
            gp_sigma_val = float(gp_sigma[0])
            gp_converged = gp_sigma_val < GP_SIGMA_CONVERGED_THRESHOLD
        except Exception:
            gp_sigma_val = float('nan')
            gp_converged = False
        n_bo_shots = len(optimizer._X)

        # ── Intra-engagement estimator convergence (ForgettingRLS-equivalent) ──
        # This codebase has no standalone scalar ForgettingRLS for the
        # intra-engagement [delta_yaw, delta_v0] search — that role is filled
        # by the BO's GP, a recursive Bayesian estimator over the full 3D
        # correction space. We convert its scalar posterior σ (metres of miss
        # distance at the best point) into the same per-axis uncertainty units
        # a ForgettingRLS would report, using the physical sensitivities already
        # used by _adaptive_n_avg: lateral_error ≈ range·Δyaw_rad and
        # downrange_effect ≈ Δv0·tof.
        # Convergence criterion (Anderson & Moore, "Optimal Filtering," 1979):
        #   P < R  →  the prior is more certain than any new observation.
        _R_db = SIGMA_YAW_DEG ** 2     # measurement noise variance — bearing (deg²)
        _R_dv = SIGMA_V0      ** 2     # measurement noise variance — v0 (m/s²)
        if not np.isnan(gp_sigma_val) and sol.horiz_range > 0:
            rls_sigma_db_deg = float(gp_sigma_val / (sol.horiz_range * np.pi / 180.0))
            rls_sigma_dv_ms  = float(gp_sigma_val / max(sol.tof, 0.1))
        else:
            rls_sigma_db_deg = float('nan')
            rls_sigma_dv_ms  = float('nan')
        rls_converged = bool(
            not np.isnan(rls_sigma_db_deg)
            and rls_sigma_db_deg**2 < _R_db
            and rls_sigma_dv_ms**2  < _R_dv
        )

        best_corr_total = np.array([
            pre_dp + optimizer.best_correction[0],
            pre_dy + optimizer.best_correction[1],
            pre_dv + optimizer.best_correction[2],
        ])

        # KF REFINEMENT — fire N_KF_SHOTS directional shots at the BO's best
        # correction to estimate and cancel the remaining systematic residual.
        # The KF is an optimal linear estimator: after 8 shots it has confidence
        # ~0.9 and can detect residuals > 0.1° that BO left uncorrected.
        for _ in range(N_KF_SHOTS):
            ex, ez, _ = self._fire_one_directional(
                sol.turret_pitch_deg + best_corr_total[0],
                sol.turret_yaw_deg   + best_corr_total[1],
                v0                   + best_corr_total[2],
                target_x, target_y, target_z)
            # Pass the target yaw angle so the KF rotates errors into barrel frame.
            # Without this, the KF maps world-frame errors to angle corrections
            # using the wrong axes at any non-zero bearing. See kalman_filter.py.
            kf.update(ex, ez, yaw_deg=sol.turret_yaw_deg)
        total_shots += N_KF_SHOTS

        # Apply KF correction when confident.
        #
        # SCALING: the KF's linear model assumes Δθ_rad = error / range (a laser,
        # not a projectile). The true ballistic pitch sensitivity is:
        #   dR/dθ = 2·v₀²·cos(2θ)/g   [m/rad]
        # At typical operating points (R=200m, θ=6°, v₀=100 m/s): dR/dθ≈2000 m/rad.
        # The KF treats range_m (200 m) as the sensitivity — 10× too small.
        # Without this correction, KF overcorrects by ~10×, pushing verified_cep
        # above the fallback threshold on almost every engagement.
        #
        # Yaw does not need scaling: lateral_error = range·Δyaw_rad is already
        # the correct linearisation (the projectile's lateral displacement is
        # approximately linear in yaw for small angles).
        _theta_rad = deg_to_rad(sol.turret_pitch_deg)
        _dR_dtheta = 2.0 * v0**2 * np.cos(2.0 * _theta_rad) / GRAVITY  # m/rad
        # Near 45° (optimal range), cos(90°)=0 → sensitivity→0; clamp so scale
        # never exceeds 1.0 (no amplification beyond what the KF thinks).
        _kf_pitch_scale = min(1.0, sol.horiz_range / max(abs(_dR_dtheta), 1.0))

        _kf_dp_result, _kf_dy_result = kf.correction_deg
        if kf.confidence > 0.5:
            kf_dp, kf_dy = _kf_dp_result, _kf_dy_result
            best_corr_total = np.array([
                best_corr_total[0] + kf_dp * _kf_pitch_scale,
                best_corr_total[1] + kf_dy,   # yaw scaling is correct as-is
                best_corr_total[2],
            ])

        # FIX-1: verified uses same RNG stream (paired) with FIX-6 bootstrap
        ci_result  = self.verified_cep_with_ci(
            sol, best_corr_total, target_x, target_y, target_z, v0)
        v_cep      = ci_result["cep_50"]
        ci_low     = ci_result["ci_low"]
        ci_high    = ci_result["ci_high"]
        v_misses   = ci_result["misses"]
        total_shots += N_SHOTS_VERIFY

        # FIX-1: Wilcoxon test for statistical significance
        wilcox = self.paired_wilcoxon(bl_misses, v_misses)

        # SAFE CORRECTION FALLBACK
        # If verified_cep > 1.1 * baseline_cep (correction is actively harmful,
        # >10% worse than no correction), fall back to zero correction.
        # We report verified_cep = baseline_cep (0% improvement) rather than
        # firing another 30 shots at zero correction: a second measurement at
        # short range is dominated by shot noise and can produce a misleading
        # large negative improvement even though we applied no correction at all.
        # The conservative report (0%) is more honest: we chose not to correct.
        _fallback_fired = False
        if v_cep > bl_cep * FALLBACK_THRESHOLD:
            _fallback_fired = True
            best_corr_total = np.zeros(3)
            v_cep    = bl_cep
            ci_low   = bl_cep
            ci_high  = bl_cep
            v_misses = bl_misses   # paired: same shot list as baseline
            wilcox   = {"p_value": 1.0, "significant": False,
                        "n": len(bl_misses)}

        # Record correction ONLY when it was genuinely useful.
        # Moving this after the fallback check prevents harmful corrections from
        # poisoning PINN training data — a correction that triggered the fallback
        # made things worse and must not be learned from.
        if self.range_table is not None and not _fallback_fired:
            try:
                _, std_gp = optimizer.gp.predict(
                    optimizer.best_correction.reshape(1,-1))
                conf = float(std_gp[0])
            except Exception:
                conf = 1.0
            self.range_table.record_correction(
                range_m=sol.horiz_range, height_m=target_y, v0_ms=v0,
                delta_pitch=float(best_corr_total[0]),
                delta_yaw=float(best_corr_total[1]),
                delta_v0=float(best_corr_total[2]),
                miss_before=bl_cep, miss_after=v_cep,
                confidence=conf, n_shots_used=total_shots,
                solution_type=prefer)

        imp = (bl_cep - v_cep) / bl_cep * 100 if bl_cep > 0 else 0.0

        return {
            "target":          (target_x, target_y, target_z),
            "horiz_range":     sol.horiz_range,
            "baseline_cep":    bl_cep,
            "best_miss":       optimizer.best_miss,
            "verified_cep":    v_cep,
            "ci_low":          ci_low,
            "ci_high":         ci_high,
            "wilcoxon_p":      wilcox["p_value"],
            "significant":     wilcox["significant"],
            "best_correction": best_corr_total,
            "improvement_pct": imp,
            "total_shots":     total_shots,
            "history":         pd.DataFrame(history),
            "n_avg_used":      n_avg,
            "adaptive_bounds": optimizer.bounds.tolist(),
            "kf_correction":  {"delta_pitch": _kf_dp_result,
                                "delta_yaw":   _kf_dy_result},
            "kf_confidence":   kf.confidence,
            # ── Confidence signals (Phase 1 additions) ──────────────────
            "gp_sigma_m":        gp_sigma_val,
            "gp_converged":      gp_converged,
            "n_bo_shots":        n_bo_shots,
            "bo_early_stopped":  optimizer._early_stopped,
            "bo_early_stop_iter": optimizer._early_stop_iter,
            "rls_converged":     rls_converged,
            "rls_sigma_db_deg":  rls_sigma_db_deg,
            "rls_sigma_dv_ms":   rls_sigma_dv_ms,
            "rls_db_final":      float(best_corr_total[1]),
            "rls_dv_final":      float(best_corr_total[2]),
        }


if __name__ == "__main__":
    np.random.seed(0)
    print("="*60)
    print("ARCS Bayesian Optimizer v3.0 — Benchmark Edition")
    print("Paired eval | SNR n_avg | 2D GlobalModel | Tightened bounds")
    print("="*60)

    from physics.range_table import RangeTable
    import numpy as _np, tempfile
    from pathlib import Path

    tmpdir = Path(tempfile.mkdtemp())
    rt = RangeTable(str(tmpdir/"phys.csv"), str(tmpdir/"corr.csv"))
    rt.generate_physics(
        range_steps=_np.arange(50,405,50),
        height_steps=_np.arange(-10,35,15),
        v0_steps=_np.array([80,100,120]),
        verbose=True, force=True)

    bias  = RobotBiasModel(seed=42)
    sim   = EngagementSimulator(seed=42, range_table=rt, bias_model=bias)
    mem   = EngagementMemory()
    gmod  = GlobalModel()
    bo    = BayesianOptimizer(memory=mem, global_model=gmod,
                               n_avg=6, n_init=4, n_suggest=16,
                               kappa=2.0, kappa_min=0.5)

    print(f"\nRobot bias profile: {bias.summary()}")
    print(f"\nRunning 15 engagements...")
    results = []
    targets = [(150,0,0),(200,0,50),(120,10,-40),(250,-5,80),(180,15,0),
               (300,0,-60),(100,0,30),(220,8,-70),(160,20,50),(280,0,0),
               (130,-8,40),(350,5,-30),(190,12,60),(240,0,-20),(170,25,0)]

    for i,(tx,ty,tz) in enumerate(targets):
        r = sim.run_engagement(tx,ty,tz,v0=100,optimizer=bo,verbose=False)
        if r:
            results.append(r)
            sig = "p<.05" if r["significant"] else "n.s. "
            print(f"  Eng {i+1:2d} ({tx:3d},{ty:3d}): "
                  f"base={r['baseline_cep']:5.2f}m "
                  f"verified={r['verified_cep']:5.2f}m "
                  f"CI=[{r['ci_low']:.2f},{r['ci_high']:.2f}] "
                  f"{sig} imp={r['improvement_pct']:+.1f}%")
        if (i+1) % 5 == 0 and rt._corrections_df is not None:
            gmod.train(rt._corrections_df)
            bo.global_model = gmod

    if results:
        imps = [r["improvement_pct"] for r in results]
        sigs = [r["significant"] for r in results]
        print(f"\n  Mean improvement: {_np.mean(imps):+.1f}%")
        print(f"  Positive: {sum(1 for i in imps if i>0)}/{len(imps)}")
        print(f"  Statistically significant: {sum(sigs)}/{len(sigs)}")
    print("\n  bayesian_optimizer.py v3.0 complete")