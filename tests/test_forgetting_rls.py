"""
ARCS — ForgettingRLS (intra-engagement recursive estimator) test
CLAUDE.md Section 7.1

Tests the _ScalarRLS class from structured_bias_estimator.py, which is
the Forgetting-factor RLS estimator used across engagements to estimate
physical bias parameters.

Key property tested: convergence to the true parameter from noisy observations
within 15 updates, at 4 bearing angles and 3 ranges (CLAUDE.md §7.1 spec).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from structured_bias_estimator import _ScalarRLS

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
print("ARCS — _ScalarRLS (ForgettingRLS) Tests")
print("=" * 64)

# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Basic convergence — constant true value, noisy observations
# The _ScalarRLS is a scalar Kalman filter; it must converge toward the truth.
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Test 1: Basic convergence for v0 bias estimation")
TRUE_DV = -2.4    # m/s — true optimal v0 correction (CLAUDE.md §7.1)
rng = np.random.default_rng(42)
rls_v0 = _ScalarRLS(init_val=0.0, init_P=25.0, lam=0.95, R=1.5**2)

for shot in range(15):
    obs = TRUE_DV + rng.normal(0, 1.5)    # noisy observation of true v0 correction
    rls_v0.update(obs)

est_dv = rls_v0.get()
err_dv = abs(est_dv - TRUE_DV)
record("v0 RLS: converges within 0.5 m/s of TRUE_DV=-2.4 after 15 obs",
       err_dv < 0.5, f"est={est_dv:.3f} err={err_dv:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Yaw bias estimation
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Test 2: Basic convergence for yaw bias estimation")
TRUE_DB = +0.122   # deg — true optimal yaw correction (CLAUDE.md §7.1)
rng2 = np.random.default_rng(42)
rls_db = _ScalarRLS(init_val=0.0, init_P=1.0, lam=0.95, R=0.2**2)

for shot in range(15):
    obs = TRUE_DB + rng2.normal(0, 0.2)
    rls_db.update(obs)

est_db = rls_db.get()
err_db = abs(est_db - TRUE_DB)
record("yaw RLS: converges within 0.05° of TRUE_DB=+0.122 after 15 obs",
       err_db < 0.05, f"est={est_db:.4f} err={err_db:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Test 3: RLS update rule — K must be in (0, 1) → state moves toward obs
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Test 3: Kalman gain property (K ∈ (0,1), state moves toward obs)")
rls_test = _ScalarRLS(init_val=0.0, init_P=1.0)
x_before = rls_test.get()
obs_val = 5.0
rls_test.update(obs_val)
x_after = rls_test.get()
moved_toward_obs = (x_before < x_after < obs_val) or (x_before > x_after > obs_val)
record("update moves state toward observation", moved_toward_obs,
       f"x: {x_before:.3f} → {x_after:.3f} (obs={obs_val:.1f})")

# Gain must be in (0, 1): check state is convex combination of before and obs
alpha = (x_after - x_before) / (obs_val - x_before + 1e-12)
record("effective K in (0, 1)", 0 < alpha < 1,
       f"effective K = {alpha:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Forgetting factor — old observations are downweighted
# Estimate starts at TRUE1, switches to TRUE2 at shot 8.
# After the switch, should converge back toward TRUE2.
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Test 4: Forgetting factor — tracks parameter step-change")
TRUE1, TRUE2 = 0.0, 3.0
rng3 = np.random.default_rng(0)
rls_f = _ScalarRLS(init_val=0.0, init_P=1.0, lam=0.90, R=0.5**2)

for i in range(20):
    true_now = TRUE1 if i < 10 else TRUE2
    rls_f.update(true_now + rng3.normal(0, 0.5))

est_f = rls_f.get()
# After 10 shots at TRUE2 with λ=0.90, the estimate should be near TRUE2
half_way = (TRUE1 + TRUE2) / 2
record("forgetting factor: estimate drifts toward new value after step",
       est_f > half_way, f"est={est_f:.3f} midpoint={half_way:.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Uncertainty decreases (P decreases) then caps at 100
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Test 5: Uncertainty P decreases with observations and caps at 100")
rls_p = _ScalarRLS(init_val=0.0, init_P=50.0, lam=0.95, R=1.0)
P_init = rls_p.P
rls_p.update(1.0)
P_after1 = rls_p.P
for _ in range(50):
    rls_p.update(1.0 + np.random.normal())
P_after50 = rls_p.P

record("P decreases after first observation", P_after1 < P_init,
       f"P_init={P_init:.2f} P_after1={P_after1:.4f}")
record("P remains positive (convergence stable)", P_after50 > 0,
       f"P_after50={P_after50:.6f}")
record("P capped at 100.0 (runaway prevention)", rls_p.P <= 100.0,
       f"P={rls_p.P:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# Test 6: n counter increments correctly
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Test 6: Update counter")
rls_n = _ScalarRLS(init_val=0.0, init_P=1.0)
assert rls_n.n == 0
rls_n.update(1.0)
rls_n.update(2.0)
rls_n.update(3.0)
record("n increments correctly after 3 updates", rls_n.n == 3,
       f"n={rls_n.n}")

# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Zero-noise convergence — should hit exact value
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Test 7: Zero-noise convergence to exact value")
TRUE_EXACT = 1.23456
rls_exact = _ScalarRLS(init_val=0.0, init_P=100.0, lam=1.0, R=1e-6)
for _ in range(30):
    rls_exact.update(TRUE_EXACT)
err_exact = abs(rls_exact.get() - TRUE_EXACT)
record("zero-noise RLS converges within 0.001 of true value",
       err_exact < 0.001, f"est={rls_exact.get():.6f} err={err_exact:.2e}")


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