"""
ARCS — SBE Credible Interval test suite
CLAUDE_PHASE1_ADDITIONS.md Step D (test_sbe_credible_intervals.py)

Validates _ScalarRLS.credible_interval() and the credible-interval fields
StructuredBiasEstimator.predict() now exposes:
  1. A 90% credible interval should contain the true parameter in
     ~90% of independent trials (frequentist coverage check on the
     Gaussian-posterior approximation — Söderström & Stoica, 1989).
  2. predict() must surface b_*_ci90 tuples with lower < point estimate < upper.
  3. Interval width must shrink monotonically as more observations accumulate
     (posterior variance P is non-increasing under RLS — Gauss-Markov).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from structured_bias_estimator import StructuredBiasEstimator, _ScalarRLS

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
print("ARCS — SBE Credible Interval Tests")
print("=" * 64)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: 90% CI coverage — must contain the true value in ≥ 85% of trials
# (CLAUDE_PHASE1_ADDITIONS.md Step D spec test, run verbatim against the
#  actual _ScalarRLS implementation)
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Test 1: 90% credible interval coverage over 100 independent trials")
TRUE_V0 = 2.1
n_trials, n_contained = 100, 0
for seed in range(n_trials):
    rls = _ScalarRLS(init_val=0.0, init_P=25.0, lam=0.95, R=1.5 ** 2)
    rng = np.random.default_rng(seed)
    for _ in range(20):
        obs = TRUE_V0 + rng.normal(0, 1.5)
        rls.update(obs)
    lo, hi = rls.credible_interval(0.90)
    if lo <= TRUE_V0 <= hi:
        n_contained += 1

coverage = n_contained / n_trials
record("90% CI coverage >= 85% (no severe undercoverage)", coverage >= 0.85,
       f"coverage={coverage:.0%}")
print(f"    90% CI coverage = {coverage:.0%} ({n_contained}/{n_trials} trials)")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: predict() surfaces well-formed CI tuples bracketing the point estimate
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Test 2: SBE.predict() exposes b_*_ci90 brackets around point estimates")
rng = np.random.default_rng(42)
sbe = StructuredBiasEstimator()
for _ in range(12):
    pitch = float(rng.uniform(3, 15))
    sbe.update_engagement(pitch_deg=pitch,
                          db_opt=0.12 + float(rng.normal(0, 0.2)),
                          dv_opt=-2.1 + float(rng.normal(0, 1.5)),
                          dp_opt=0.05 + float(rng.normal(0, 0.05)))

pred = sbe.predict(pitch_deg=8.6)
for key, ci_key in (('b_sag', 'b_sag_ci90'), ('b_yaw', 'b_yaw_ci90'), ('b_v0', 'b_v0_ci90')):
    record(f"predict() contains '{ci_key}'", ci_key in pred)
    record(f"predict() contains '{key}' point estimate", key in pred)
    if ci_key in pred and key in pred:
        lo, hi = pred[ci_key]
        point = pred[key]
        record(f"{ci_key}: lower <= point estimate <= upper",
               lo <= point <= hi, f"[{lo:.4f}, {hi:.4f}] does not bracket {point:.4f}")
        record(f"{ci_key}: lower < upper (non-degenerate interval)", lo < hi)

print(f"    b_sag={pred['b_sag']:+.4f}  90% CI {pred['b_sag_ci90']}")
print(f"    b_yaw={pred['b_yaw']:+.4f}  90% CI {pred['b_yaw_ci90']}")
print(f"    b_v0 ={pred['b_v0']:+.4f}  90% CI {pred['b_v0_ci90']}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: CI width is non-increasing under repeated RLS updates
# (posterior variance P only shrinks — Gauss-Markov / RLS theory)
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Test 3: Credible interval width is non-increasing as observations accumulate")
rls = _ScalarRLS(init_val=0.0, init_P=25.0, lam=1.0, R=1.5 ** 2)
rng = np.random.default_rng(11)
prev_width = float('inf')
all_shrunk = True
for i in range(15):
    rls.update(TRUE_V0 + rng.normal(0, 1.5))
    lo, hi = rls.credible_interval(0.90)
    width = hi - lo
    if width > prev_width + 1e-9:
        all_shrunk = False
    prev_width = width

record("CI width never increases under lam=1.0 (no forgetting)", all_shrunk)
print(f"    final CI width = {prev_width:.4f}")


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
