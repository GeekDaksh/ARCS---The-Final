"""
Validation suite for Component 5 — the estimation layer (two SEPARATE
estimators). Ground truth = convergence to a KNOWN HIDDEN truth: we set a true
effective wind the estimator is not told, simulate fall of shot through the
frozen physics, and verify the estimator converges to it. The gun-bias and
atmospheric estimators are kept separate and must each capture their own
component without bleeding into the other.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.trajectory import integrate_trajectory
from physics.state_estimator import (AtmosphericStateEstimator, GunBiasEstimator,
                                     wind_sensitivity)

V0 = 827.0


def impact(wind, e=45.0):
    r = integrate_trajectory(v0=V0, elevation_deg=e, wind=tuple(wind))
    return np.array([r["impact_x"], r["impact_y"]])


def make_estB(P=25.0, Q=0.05, R=2500.0):
    return AtmosphericStateEstimator(
        initial_state=[0.0, 0.0],
        initial_covariance=np.diag([P, P]),
        process_noise=np.diag([Q, Q]),
        measurement_noise=np.diag([R, R]),
    )


def run_wind_convergence(true_wind, met_wind=(20.0, 0.0, 0.0), e=45.0,
                         n_shots=6, est=None):
    """Closed-loop: aim with the current estimate, observe the residual miss
    (which IS the Kalman innovation), update. Returns (estimates, misses)."""
    met = np.asarray(met_wind, dtype=float)
    truth = np.asarray(true_wind, dtype=float)
    H = wind_sensitivity(baseline_wind=met, v0=V0, elevation_deg=e)
    if est is None:
        est = make_estB()
    estimates, misses = [], []
    for _ in range(n_shots):
        aim = met + np.array([est.state()[0], est.state()[1], 0.0])
        miss = impact(truth, e) - impact(aim, e)
        est.predict()
        est.update(miss, H)
        estimates.append(est.state().copy())
        misses.append(np.hypot(*miss))
    return np.array(estimates), np.array(misses), est


# ---------------------------------------------------------------------------
# Layer 1 — Convergence to a hidden truth (the core proof)
# ---------------------------------------------------------------------------
def test_layer1_converges_to_hidden_downrange_wind():
    # MET says 20 m/s downrange; the TRUE wind is 25. Estimator only sees misses.
    est_trace, misses, _ = run_wind_convergence(true_wind=(25.0, 0.0, 0.0))
    true_delta = 5.0  # 25 - 20
    assert abs(est_trace[4][0] - true_delta) < 0.6, est_trace[:, 0]  # within 5 shots
    assert abs(est_trace[-1][0] - true_delta) < 0.3


def test_layer1_converges_to_hidden_crosswind():
    est_trace, misses, _ = run_wind_convergence(true_wind=(20.0, 6.0, 0.0))
    assert abs(est_trace[-1][1] - 6.0) < 0.5, est_trace[:, 1]


def test_layer1_correction_reduces_miss():
    # Later shots land closer than the first (the estimate is correcting aim).
    _, misses, _ = run_wind_convergence(true_wind=(25.0, 0.0, 0.0))
    assert misses[-1] < misses[0]
    assert misses[-1] < 0.25 * misses[0]


# ---------------------------------------------------------------------------
# Layer 2 — Boundary behaviour
# ---------------------------------------------------------------------------
def test_layer2_no_drift_when_met_is_correct():
    # True wind == MET wind: there is nothing to correct. Estimator must NOT
    # invent a correction.
    est_trace, _, _ = run_wind_convergence(true_wind=(20.0, 0.0, 0.0))
    assert np.linalg.norm(est_trace[-1]) < 0.3, est_trace[-1]


def test_layer2_zero_miss_no_state_change():
    est = make_estB()
    H = wind_sensitivity(baseline_wind=(20.0, 0.0, 0.0), v0=V0, elevation_deg=45.0)
    est.predict()
    before = est.state().copy()
    est.update(np.array([0.0, 0.0]), H)
    assert np.allclose(est.state(), before, atol=1e-12)


# ---------------------------------------------------------------------------
# Layer 3 — Known-statistics relationships
# ---------------------------------------------------------------------------
def test_layer3_uncertainty_shrinks_with_observations():
    est = make_estB()
    H = wind_sensitivity(baseline_wind=(20.0, 0.0, 0.0), v0=V0, elevation_deg=45.0)
    est.predict(); est.update(impact((25, 0, 0)) - impact((20, 0, 0)), H)
    unc_after_1 = np.trace(est.uncertainty())
    for _ in range(5):
        est.predict(); est.update(impact((25, 0, 0)) - impact((20, 0, 0)), H)
    unc_after_n = np.trace(est.uncertainty())
    assert unc_after_n < unc_after_1


def test_layer3_higher_measurement_noise_converges_slower():
    # Same hidden truth; a filter that trusts measurements less (higher R) moves
    # more cautiously and is further from the truth after the same few shots.
    trace_lowR, _, _ = run_wind_convergence((25, 0, 0), est=make_estB(R=400.0), n_shots=3)
    trace_hiR, _, _ = run_wind_convergence((25, 0, 0), est=make_estB(R=40000.0), n_shots=3)
    err_low = abs(trace_lowR[-1][0] - 5.0)
    err_hi = abs(trace_hiR[-1][0] - 5.0)
    assert err_low < err_hi, (err_low, err_hi)


def test_layer3_higher_process_noise_tracks_change_faster():
    # Converge both to a first truth, then STEP-change the true wind. The filter
    # with more process noise (more willing to move) adapts faster.
    met = (20.0, 0.0, 0.0)
    H = wind_sensitivity(baseline_wind=met, v0=V0, elevation_deg=45.0)

    def converge_then_step(Q):
        est = make_estB(Q=Q)
        # converge to 25
        for _ in range(8):
            aim = np.array(met) + np.array([est.state()[0], est.state()[1], 0.0])
            est.predict(); est.update(impact((25, 0, 0)) - impact(aim), H)
        # step change: true wind jumps to 32
        for _ in range(3):
            aim = np.array(met) + np.array([est.state()[0], est.state()[1], 0.0])
            est.predict(); est.update(impact((32, 0, 0)) - impact(aim), H)
        return est.state()[0]

    slow = converge_then_step(Q=0.001)
    fast = converge_then_step(Q=0.5)
    # After the step to 32 (true delta 12), the high-Q filter is closer.
    assert abs(fast - 12.0) < abs(slow - 12.0), (fast, slow)


# ---------------------------------------------------------------------------
# Layer 4 — Separation & sanity
# ---------------------------------------------------------------------------
def run_separation(gun_bias=(150.0, 60.0), true_wind=(25.0, 0.0, 0.0),
                   met_wind=(20.0, 0.0, 0.0), e=45.0):
    """Registration-then-operation, the doctrinal separation. Returns the two
    estimators after a known-correct registration (isolates gun bias) followed
    by operation under a wrong MET (isolates wind)."""
    B = np.asarray(gun_bias, dtype=float)
    met = np.asarray(met_wind, dtype=float)
    H = wind_sensitivity(baseline_wind=met, v0=V0, elevation_deg=e)
    estA = GunBiasEstimator(R=400.0)
    estB = make_estB()

    # REGISTRATION: MET is correct, so every miss is PURE gun bias -> A.
    for _ in range(5):
        raw = (impact(met, e) + B) - impact(met, e)  # == B
        estA.update(raw)
        estB.predict(); estB.update(raw - estA.state(), H)  # B sees ~0

    # OPERATION: MET now wrong. Remove the KNOWN gun bias -> residual is wind -> B.
    truth = np.asarray(true_wind, dtype=float)
    for _ in range(8):
        aim = met + np.array([estB.state()[0], estB.state()[1], 0.0])
        observed = impact(truth, e) + B
        predicted = impact(aim, e) + estA.state()   # apply both corrections
        estB.predict(); estB.update(observed - predicted, H)
    return estA, estB


def test_layer4_gun_and_wind_separated():
    estA, estB = run_separation(gun_bias=(150.0, 60.0), true_wind=(25.0, 0.0, 0.0))
    gunA = estA.state()
    windB = estB.state()
    # Estimator A captured the gun's constant offset (not the wind).
    assert np.allclose(gunA, [150.0, 60.0], atol=12.0), gunA
    # Estimator B captured the wind error (not the gun bias).
    assert abs(windB[0] - 5.0) < 0.6 and abs(windB[1]) < 0.6, windB
    # No bleed: the wind estimate is m/s-scale (~5), the gun estimate is
    # metres-scale (~150) — neither absorbed the other's magnitude.
    assert np.linalg.norm(windB) < 20.0      # B did not swallow the 150 m gun bias
    assert np.linalg.norm(gunA) > 100.0      # A did not collapse to the wind


def test_layer4_no_false_gun_bias_when_only_wind():
    # Pure wind, zero gun bias: A must stay ~0, B captures the wind.
    estA, estB = run_separation(gun_bias=(0.0, 0.0), true_wind=(25.0, 0.0, 0.0))
    assert np.linalg.norm(estA.state()) < 12.0, estA.state()
    assert abs(estB.state()[0] - 5.0) < 0.6, estB.state()


def test_layer4_estimate_stays_bounded_many_shots():
    # No oscillation blow-up over many engagements (the Phase 1 Kalman-gain
    # stability lesson).
    est = make_estB()
    H = wind_sensitivity(baseline_wind=(20.0, 0.0, 0.0), v0=V0, elevation_deg=45.0)
    met = np.array([20.0, 0.0, 0.0])
    for _ in range(60):
        aim = met + np.array([est.state()[0], est.state()[1], 0.0])
        est.predict(); est.update(impact((25, 0, 0)) - impact(aim), H)
        assert np.all(np.isfinite(est.state()))
        assert np.linalg.norm(est.state()) < 50.0  # bounded, never diverges
    assert abs(est.state()[0] - 5.0) < 0.3


if __name__ == "__main__":
    print("\n=== Component 5 — estimator validation summary ===\n")

    print("(1) Convergence to hidden truth (MET says 20 m/s, TRUE is 25):")
    trace, misses, _ = run_wind_convergence((25, 0, 0))
    print(f"    {'shot':>4} {'est wind (m/s)':>16} {'miss (m)':>10}")
    for i, (st, ms) in enumerate(zip(trace, misses), 1):
        print(f"    {i:>4} {st[0]:>15.3f}  {ms:>9.1f}")
    print(f"    -> converged to {trace[-1][0]:.3f} m/s  (true 5.000),  "
          f"miss {misses[0]:.0f} m -> {misses[-1]:.1f} m")

    print("\n(2) Uncertainty shrinks with consistent observations:")
    est = make_estB()
    H = wind_sensitivity(baseline_wind=(20, 0, 0), v0=V0, elevation_deg=45.0)
    est.predict(); est.update(impact((25, 0, 0)) - impact((20, 0, 0)), H)
    u1 = np.trace(est.uncertainty())
    for _ in range(5):
        est.predict(); est.update(impact((25, 0, 0)) - impact((20, 0, 0)), H)
    print(f"    trace(P) after 1 shot = {u1:.3f}  ->  after 6 shots = "
          f"{np.trace(est.uncertainty()):.3f}  (more confident)")

    print("\n(3) Separation — gun personality vs weather (both present):")
    estA, estB = run_separation(gun_bias=(150.0, 60.0), true_wind=(25.0, 0.0, 0.0))
    print(f"    Estimator A (gun bias, m) = {np.round(estA.state(), 1)}   "
          f"(true [150.0, 60.0])")
    print(f"    Estimator B (wind,  m/s)  = {np.round(estB.state(), 2)}   "
          f"(true [5.0, 0.0])")
    print("    -> each estimator captured its own component; no bleed.")

    print("\nRun 'python -m pytest tests/test_state_estimator.py -v'.")
