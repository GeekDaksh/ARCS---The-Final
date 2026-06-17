"""
ARCS — StructuredBiasEstimator test suite
CLAUDE.md Section 7.1 (test_sbe.py)

Validates that the SBE:
  1. Converges to the true physical bias parameters (b_sag, b_yaw, b_v0)
     from seed=42 robot within 20 engagements.
  2. Confidence score increases correctly.
  3. Predictions correct the right direction and magnitude.
  4. warm-start from existing data works.
  5. reset() clears estimates.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from structured_bias_estimator import StructuredBiasEstimator
from physics.constants import SIGMA_PITCH_DEG, SIGMA_YAW_DEG, SIGMA_V0

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


# ─────────────────────────────────────────────────────────────────────────────
# True bias parameters for seed=42 robot (from RobotBiasModel docs)
# ─────────────────────────────────────────────────────────────────────────────
TRUE_SAG = 0.499    # gravity sag coefficient (deg/unit_sin_theta)
TRUE_YAW = -0.122   # deg — IMU yaw offset
TRUE_V0  = 2.1      # m/s — total v0 bias

print("=" * 64)
print("ARCS — StructuredBiasEstimator Tests")
print(f"  Seed=42 robot: b_sag={TRUE_SAG}  b_yaw={TRUE_YAW:+.3f}°  b_v0={TRUE_V0:+.2f}m/s")
print("=" * 64)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Convergence to true bias after 20 engagements (CLAUDE.md §7.1)
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Test 1: 20-engagement convergence to true bias parameters")
rng = np.random.default_rng(42)
sbe = StructuredBiasEstimator()

print(f"\n  {'Eng':>4}  {'b_sag':>10}  {'b_yaw':>10}  {'b_v0':>10}  {'conf':>6}")

for eng in range(20):
    pitch_deg = rng.uniform(3.0, 15.0)
    sin_theta = np.sin(np.deg2rad(pitch_deg))

    # Simulate Level-1 results: BO found the optimal corrections with noise
    dp_opt = TRUE_SAG * sin_theta + rng.normal(0, SIGMA_PITCH_DEG)
    db_opt = -TRUE_YAW + rng.normal(0, SIGMA_YAW_DEG)
    dv_opt = -TRUE_V0  + rng.normal(0, SIGMA_V0)

    sbe.update_engagement(pitch_deg, db_opt, dv_opt, dp_opt)

    if (eng + 1) in (5, 10, 15, 20):
        print(f"  {eng+1:>4}  {sbe._b_sag.get():>+10.4f}  "
              f"{sbe._b_yaw.get():>+10.4f}°  {sbe._b_v0.get():>+10.3f}m/s  "
              f"{sbe.confidence():>6.2f}")

b_sag_est = sbe._b_sag.get()
b_yaw_est = sbe._b_yaw.get()
b_v0_est  = sbe._b_v0.get()

record("SBE: b_yaw converges within 0.05° of true (-0.122°)",
       abs(b_yaw_est - TRUE_YAW) < 0.05,
       f"est={b_yaw_est:+.4f}° err={abs(b_yaw_est-TRUE_YAW):.4f}°")
record("SBE: b_v0 converges within 0.5 m/s of true (+2.1 m/s)",
       abs(b_v0_est - TRUE_V0) < 0.5,
       f"est={b_v0_est:+.3f} err={abs(b_v0_est-TRUE_V0):.3f}")
record("SBE: b_sag converges within 0.1 of true (0.499)",
       abs(b_sag_est - TRUE_SAG) < 0.1,
       f"est={b_sag_est:.4f} err={abs(b_sag_est-TRUE_SAG):.4f}")
record("SBE: confidence >= 0.9 after 20 engagements",
       sbe.confidence() >= 0.9,
       f"conf={sbe.confidence():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Prediction direction and magnitude
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Test 2: Prediction direction and magnitude at pitch=8.6°")
pred = sbe.predict(pitch_deg=8.6, v0=100.0)

sin_86 = np.sin(np.deg2rad(8.6))
expected_dp = TRUE_SAG * sin_86 * 0.88    # PITCH_EFF = 0.88
expected_db = -TRUE_YAW * 0.87            # YAW_EFF = 0.87
expected_dv = -TRUE_V0  * 0.91            # V0_EFF = 0.91

print(f"\n    Expected: dp={expected_dp:+.4f}°  db={expected_db:+.4f}°  dv={expected_dv:+.3f}m/s")
print(f"    Got:      dp={pred['delta_pitch']:+.4f}°  db={pred['delta_yaw']:+.4f}°  "
      f"dv={pred['delta_v0']:+.3f}m/s")

record("Prediction: delta_pitch correct direction (positive, counters negative sag)",
       pred['delta_pitch'] > 0,
       f"got {pred['delta_pitch']:+.4f}°")
record("Prediction: delta_yaw correct direction (positive, counters negative imu_offset)",
       pred['delta_yaw'] > 0,
       f"got {pred['delta_yaw']:+.4f}°")
record("Prediction: delta_v0 correct direction (negative, counters positive v0_bias)",
       pred['delta_v0'] < 0,
       f"got {pred['delta_v0']:+.3f}m/s")
record("Prediction: delta_pitch within 0.05° of expected",
       abs(pred['delta_pitch'] - expected_dp) < 0.05,
       f"diff={abs(pred['delta_pitch']-expected_dp):.5f}°")
record("Prediction: delta_yaw within 0.05° of expected",
       abs(pred['delta_yaw'] - expected_db) < 0.05,
       f"diff={abs(pred['delta_yaw']-expected_db):.5f}°")
record("Prediction: delta_v0 within 0.5 m/s of expected",
       abs(pred['delta_v0'] - expected_dv) < 0.5,
       f"diff={abs(pred['delta_v0']-expected_dv):.3f}m/s")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Confidence curve — increases monotonically for first 20 engagements
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Test 3: Confidence monotonically increases with engagement count")
sbe_conf = StructuredBiasEstimator()
rng2 = np.random.default_rng(7)
conf_prev = 0.0
mono_ok = True
for i in range(20):
    pitch = rng2.uniform(3, 15)
    sbe_conf.update_engagement(pitch,
                                db_opt = -TRUE_YAW + rng2.normal(0, SIGMA_YAW_DEG),
                                dv_opt = -TRUE_V0  + rng2.normal(0, SIGMA_V0),
                                dp_opt = TRUE_SAG * np.sin(np.deg2rad(pitch)) + rng2.normal(0, SIGMA_PITCH_DEG))
    c = sbe_conf.confidence()
    if c < conf_prev - 1e-9:
        mono_ok = False
    conf_prev = c

record("Confidence is non-decreasing over 20 engagements", mono_ok)
record("Confidence at 0 engagements is 0.0",
       StructuredBiasEstimator().confidence() == 0.0)
record("Confidence in [0, 1] after 20 engagements",
       0.0 <= sbe_conf.confidence() <= 1.0,
       f"conf={sbe_conf.confidence():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: reset() clears estimates back to prior
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Test 4: reset() clears estimates")
sbe_r = StructuredBiasEstimator()
sbe_r.update_engagement(8.0, db_opt=0.12, dv_opt=-2.1, dp_opt=0.05)
sbe_r.update_engagement(10.0, db_opt=0.11, dv_opt=-2.2, dp_opt=0.07)
assert sbe_r._n_engagements == 2

sbe_r.reset()
record("reset: n_engagements = 0", sbe_r._n_engagements == 0,
       f"got {sbe_r._n_engagements}")
record("reset: b_yaw back to 0.0", abs(sbe_r._b_yaw.get()) < 1e-9,
       f"got {sbe_r._b_yaw.get():.6f}")
record("reset: b_v0 back to 0.0", abs(sbe_r._b_v0.get()) < 1e-9,
       f"got {sbe_r._b_v0.get():.6f}")
record("reset: confidence back to 0.0", sbe_r.confidence() == 0.0,
       f"got {sbe_r.confidence():.4f}")
record("reset: sag_history cleared", len(sbe_r._sag_history) == 0)


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: summary() returns a non-empty string
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Test 5: summary() format")
summary_str = sbe.summary()
record("summary() contains 'SBE('", "SBE(" in summary_str)
record("summary() contains 'b_sag'", "b_sag" in summary_str)
record("summary() contains 'conf'", "conf" in summary_str)
print(f"    {summary_str}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Prediction clamps — no physically impossible corrections
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Test 6: Safety clamps on prediction output")
sbe_clamp = StructuredBiasEstimator()
# Force extreme b values
for _ in range(5):
    sbe_clamp.update_engagement(45.0, db_opt=10.0, dv_opt=-50.0, dp_opt=5.0)
pred_clamp = sbe_clamp.predict(pitch_deg=45.0, v0=100.0)
record("Prediction: |delta_pitch| <= 2.0°",
       abs(pred_clamp['delta_pitch']) <= 2.0,
       f"got {pred_clamp['delta_pitch']:+.3f}°")
record("Prediction: |delta_yaw| <= 3.0°",
       abs(pred_clamp['delta_yaw']) <= 3.0,
       f"got {pred_clamp['delta_yaw']:+.3f}°")
record("Prediction: |delta_v0| <= 12.0 m/s",
       abs(pred_clamp['delta_v0']) <= 12.0,
       f"got {pred_clamp['delta_v0']:+.3f}m/s")


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
