"""
ARCS — Kalman Filter Bearing Rotation Test
Verifies that EngagementKF applies the bearing rotation correctly.

Two sub-tests:
  A) Deterministic (zero noise): convergence must be exact at all 4 bearings.
  B) Noisy (realistic noise): all 4 bearings must give the SAME estimate,
     proving the rotation doesn't swap pitch/yaw at large bearing angles.

The bearing-rotation bug manifested at bearing=90°: without the fix, pitch
and yaw corrections were completely swapped, causing divergence in the wrong
physical direction.  With the fix, all bearings converge identically.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from kalman_filter import EngagementKF

TOTAL = PASSED = FAILED = 0
_failures = []

def record(name: str, ok: bool, detail: str = ""):
    global TOTAL, PASSED, FAILED
    TOTAL += 1
    if ok:
        PASSED += 1
        print(f"  ✓ PASS  {name}")
    else:
        FAILED += 1
        _failures.append(name)
        print(f"  ✗ FAIL  {name}{': ' + detail if detail else ''}")


def simulate_shots(bearing_deg: float,
                   true_bias_pitch: float, true_bias_yaw: float,
                   n: int = 30, noise_pitch: float = 0.0,
                   noise_yaw: float = 0.0, seed: int = 42) -> tuple:
    """
    Simulate n shots at the given bearing angle.
    Returns (Δpitch_correction_deg, Δyaw_correction_deg) from the KF.

    The barrel-frame errors are generated from the known biases, then
    rotated to world frame before being fed to the KF (which rotates
    them back).  The two rotations must cancel exactly.
    """
    RANGE = 300.0
    kf    = EngagementKF(range_m=RANGE)
    rng   = np.random.default_rng(seed)
    c     = np.cos(np.deg2rad(bearing_deg))
    s     = np.sin(np.deg2rad(bearing_deg))

    for _ in range(n):
        np_deg = rng.normal(0, noise_pitch) if noise_pitch > 0 else 0.0
        ny_deg = rng.normal(0, noise_yaw)   if noise_yaw   > 0 else 0.0

        # Barrel-aligned miss (range direction, lateral direction)
        err_along = RANGE * np.sin(np.deg2rad(true_bias_pitch + np_deg))
        err_lat   = RANGE * np.sin(np.deg2rad(true_bias_yaw   + ny_deg))

        # Rotate to world frame (ARCS X=forward, Z=lateral)
        error_x = c * err_along - s * err_lat
        error_z = s * err_along + c * err_lat

        kf.update(error_x, error_z, yaw_deg=bearing_deg)

    return kf.correction_deg


# ─────────────────────────────────────────────────────────────────────────────
# Constants from CLAUDE.md Section 6.5 (seed=42 robot)
# ─────────────────────────────────────────────────────────────────────────────
BEARINGS        = [0, 45, 90, 135]
TRUE_PITCH_BIAS = -0.075    # deg — gravity sag at 8.6° pitch
TRUE_YAW_BIAS   = -0.122    # deg — IMU yaw offset (seed=42 robot)
EXPECTED_DP     = -TRUE_PITCH_BIAS   # +0.075° — the needed correction
EXPECTED_DY     = -TRUE_YAW_BIAS     # +0.122°

print("=" * 64)
print("ARCS — EngagementKF Bearing Rotation Test")
print(f"  True pitch bias : {TRUE_PITCH_BIAS:+.4f}°  (correction needed: {EXPECTED_DP:+.4f}°)")
print(f"  True yaw bias   : {TRUE_YAW_BIAS:+.4f}°  (correction needed: {EXPECTED_DY:+.4f}°)")
print("=" * 64)

# ─────────────────────────────────────────────────────────────────────────────
# Part A — Deterministic test (no noise): tight convergence required
# After 30 shots with zero noise, the KF must converge within 0.005°.
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Part A — Deterministic convergence (noise=0, n=30, tol=0.005°):")
TOLS_DET = (0.005, 0.005)

det_results = {}
for bearing in BEARINGS:
    dp, dy = simulate_shots(bearing, TRUE_PITCH_BIAS, TRUE_YAW_BIAS,
                             n=30, noise_pitch=0.0, noise_yaw=0.0)
    dp_err = abs(dp - EXPECTED_DP)
    dy_err = abs(dy - EXPECTED_DY)
    det_results[bearing] = (dp, dy)
    print(f"\n    Bearing {bearing:>3}°:  "
          f"dp={dp:+.5f}° (err={dp_err:.5f}°)  "
          f"dy={dy:+.5f}° (err={dy_err:.5f}°)")
    record(f"A: bearing {bearing:>3}° — pitch (det, zero noise)",
           dp_err < TOLS_DET[0], f"err={dp_err:.5f}° > {TOLS_DET[0]}")
    record(f"A: bearing {bearing:>3}° — yaw   (det, zero noise)",
           dy_err < TOLS_DET[1], f"err={dy_err:.5f}° > {TOLS_DET[1]}")

# ─────────────────────────────────────────────────────────────────────────────
# Part B — Noisy test: all 4 bearings must give the SAME correction ±0.01°
# (same noise seed → same barrel-frame noise at every bearing, so after
# rotating forward then back the estimates must be identical regardless of θ).
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n  Part B — Bearing rotation consistency (noise_p=0.3°, noise_y=0.2°, n=30):")
print(f"           All bearings must agree within ±0.01°:")
noisy_results = {}
for bearing in BEARINGS:
    dp, dy = simulate_shots(bearing, TRUE_PITCH_BIAS, TRUE_YAW_BIAS,
                             n=30, noise_pitch=0.3, noise_yaw=0.2, seed=42)
    noisy_results[bearing] = (dp, dy)

ref_dp, ref_dy = noisy_results[0]
print(f"\n    Reference (bearing=0°): dp={ref_dp:+.5f}°  dy={ref_dy:+.5f}°")
for bearing in BEARINGS[1:]:
    dp, dy = noisy_results[bearing]
    dp_diff = abs(dp - ref_dp)
    dy_diff = abs(dy - ref_dy)
    print(f"    Bearing {bearing:>3}°: dp={dp:+.5f}° (diff={dp_diff:.5f}°)  "
          f"dy={dy:+.5f}° (diff={dy_diff:.5f}°)")
    record(f"B: bearing {bearing:>3}° vs 0° — pitch consistent",
           dp_diff < 0.01, f"diff={dp_diff:.5f}° > 0.01°")
    record(f"B: bearing {bearing:>3}° vs 0° — yaw consistent",
           dy_diff < 0.01, f"diff={dy_diff:.5f}° > 0.01°")

# ─────────────────────────────────────────────────────────────────────────────
# Part C — Axis-swap check at 90°: pitch and yaw must NOT be swapped
# Without the bearing-rotation fix, at 90° pitch≈+0.122° and yaw≈+0.075°
# (the axes are swapped).  With the fix both bearings give the same values.
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n  Part C — Axis-swap guard at bearing=90° (det, n=30):")
dp_0,  dy_0  = det_results[0]
dp_90, dy_90 = det_results[90]

# These must be the SAME sign and approximately SAME magnitude
pitch_not_swapped = abs(dp_90 - dp_0) < 0.01
yaw_not_swapped   = abs(dy_90 - dy_0) < 0.01

record("C: 90° pitch correction is NOT the yaw correction", pitch_not_swapped,
       f"dp(90°)={dp_90:+.5f}° dp(0°)={dp_0:+.5f}° diff={abs(dp_90-dp_0):.5f}°")
record("C: 90° yaw correction is NOT the pitch correction", yaw_not_swapped,
       f"dy(90°)={dy_90:+.5f}° dy(0°)={dy_0:+.5f}° diff={abs(dy_90-dy_0):.5f}°")

# ─────────────────────────────────────────────────────────────────────────────
# Part D — KF confidence increases with shots
# ─────────────────────────────────────────────────────────────────────────────
kf_c = EngagementKF(range_m=300.0)
conf_before = kf_c.confidence
simulate_shots(0, TRUE_PITCH_BIAS, TRUE_YAW_BIAS, n=8, seed=42)
# (separate KF for confidence check)
kf_c2 = EngagementKF(range_m=300.0)
for _ in range(8):
    kf_c2.update(0.0, -0.393, yaw_deg=0.0)
conf_after = kf_c2.confidence
record("D: KF confidence increases after 8 shots",
       conf_after > conf_before, f"{conf_before:.3f} → {conf_after:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*64}")
print(f"TOTAL: {TOTAL} | PASSED: {PASSED} | FAILED: {FAILED}")
if _failures:
    print("  Failed tests:")
    for f in _failures:
        print(f"  ✗ {f}")
print("=" * 64)

if __name__ == "__main__":
    sys.exit(0 if FAILED == 0 else 1)
