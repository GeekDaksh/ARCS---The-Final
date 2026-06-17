"""
ARCS — SBE Inter-Robot Transfer Learning test suite
CLAUDE_PHASE1_ADDITIONS.md Step E (test_sbe_transfer.py)

Validates StructuredBiasEstimator.export_parameters()/load_parameters():
warm-starting a new robot unit from a calibrated robot's exported parameters
should reach high confidence in far fewer engagements than a cold start.

Research basis: Abpeikar, Kasmarik & Garratt (Frontiers in Robotics, 2023) —
iterative transfer learning across robot platforms reduces calibration data
requirements ~4×.
"""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from structured_bias_estimator import StructuredBiasEstimator

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


print("=" * 64)
print("ARCS — SBE Transfer Learning Tests")
print("=" * 64)

TRUE_SAG, TRUE_YAW, TRUE_V0 = 0.499, -0.122, 2.1


def calibrate(sbe, rng, n):
    for _ in range(n):
        pitch = float(rng.uniform(3, 15))
        sin_t = np.sin(np.deg2rad(pitch))
        sbe.update_engagement(
            pitch_deg=pitch,
            db_opt=-TRUE_YAW + float(rng.normal(0, 0.2)),
            dv_opt=-TRUE_V0  + float(rng.normal(0, 1.5)),
            dp_opt=TRUE_SAG * sin_t + float(rng.normal(0, 0.3)))


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: export_parameters() round-trips through JSON correctly
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Test 1: export_parameters() / load_parameters() round-trip")
rng_a = np.random.default_rng(0)
sbe_a = StructuredBiasEstimator()
calibrate(sbe_a, rng_a, 20)

with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
    path = f.name
exported = sbe_a.export_parameters(path)

record("export_parameters() returns a dict", isinstance(exported, dict))
record("exported file exists on disk", os.path.exists(path))
record("export contains 'confidence'", 'confidence' in exported)
record("export contains 'n_engagements'", 'n_engagements' in exported)
print(f"    Robot A calibrated: {sbe_a.summary()}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: cold-start robot needs many more engagements to reach confidence
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Test 2: cold start vs. transfer warm start — confidence after 5 engagements")
rng_cold = np.random.default_rng(1)
sbe_cold = StructuredBiasEstimator()
calibrate(sbe_cold, rng_cold, 5)
cold_conf = sbe_cold.confidence()

rng_b = np.random.default_rng(0)
# Re-derive the same 20 calibration draws Robot A used so Robot B's
# subsequent 5 engagements start from an equivalent draw stream
calibrate(StructuredBiasEstimator(), rng_b, 20)   # advance rng_b past Robot A's draws

sbe_b = StructuredBiasEstimator()
sbe_b.load_parameters(path, uncertainty_scale=1.2)
warm_conf_before = sbe_b.confidence()
calibrate(sbe_b, rng_b, 5)
warm_conf_after = sbe_b.confidence()

print(f"    Cold start  (5 eng):            confidence = {cold_conf:.2f}")
print(f"    Transfer warm-start (pre-eng):  confidence = {warm_conf_before:.2f}")
print(f"    Transfer warm-start (+5 eng):   confidence = {warm_conf_after:.2f}")

record("transfer warm-start confidence (pre-engagement) > cold start after 5 eng",
       warm_conf_before > cold_conf,
       f"{warm_conf_before:.2f} <= {cold_conf:.2f}")
record("transfer warm-start reaches >= 0.90 confidence within 5 engagements",
       warm_conf_after >= 0.90, f"confidence={warm_conf_after:.2f}")
record("cold start has NOT reached 0.90 confidence after only 5 engagements",
       cold_conf < 0.90, f"confidence={cold_conf:.2f}")

os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: uncertainty_scale inflates posterior variance (manufacturing tolerance)
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Test 3: uncertainty_scale inflates the loaded posterior variance")
rng_c = np.random.default_rng(0)
sbe_c = StructuredBiasEstimator()
calibrate(sbe_c, rng_c, 20)
with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
    path2 = f.name
sbe_c.export_parameters(path2)

sbe_tight = StructuredBiasEstimator()
sbe_tight.load_parameters(path2, uncertainty_scale=1.0)
sbe_loose = StructuredBiasEstimator()
sbe_loose.load_parameters(path2, uncertainty_scale=3.0)
os.unlink(path2)

record("uncertainty_scale=3.0 yields larger b_v0 posterior variance than scale=1.0",
       sbe_loose._b_v0.P > sbe_tight._b_v0.P,
       f"{sbe_loose._b_v0.P:.4f} <= {sbe_tight._b_v0.P:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print(f"TOTAL: {TOTAL} | PASSED: {PASSED} | FAILED: {FAILED}")
print("=" * 64)

if _failures:
    print("\nFailed tests:")
    for f in _failures:
        print(f"  - {f}")

if __name__ == "__main__":
    sys.exit(0 if FAILED == 0 else 1)
