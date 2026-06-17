"""
ARCS — BO Early Stopping test suite
CLAUDE_PHASE1_ADDITIONS.md Step B (test_bo_early_stopping.py)

Validates that the BayesianOptimizer halts its suggestion loop early once
the GP posterior sigma at the current best correction drops below
GP_SIGMA_CONVERGED_THRESHOLD — saving shots for verification, which needs
them more (Fiedler, Scherer & Trimpe, AAAI 2021 — rigorous GP uncertainty
bounds as a stopping criterion).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from bayesian_optimizer import BayesianOptimizer, EngagementMemory
from physics.constants import GP_SIGMA_CONVERGED_THRESHOLD

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
print("ARCS — BO Early Stopping Tests")
print(f"  GP_SIGMA_CONVERGED_THRESHOLD = {GP_SIGMA_CONVERGED_THRESHOLD} m")
print("=" * 64)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: drive the BO coroutine with a synthetic, smooth miss-distance
# function of the correction. Low sigma_noise ⇒ the GP becomes confident
# quickly ⇒ early stopping should trigger well before n_suggest iterations.
# ─────────────────────────────────────────────────────────────────────────────
TRUE_OPT = np.array([0.10, -0.05, -1.0])

def run_with_synthetic_miss(seed, sigma_noise, n_init=4, n_suggest=16):
    rng = np.random.default_rng(seed)
    bo  = BayesianOptimizer(memory=EngagementMemory(), n_init=n_init, n_suggest=n_suggest)
    bo.reset(range_m=300.0, height_m=0.0, pitch_cmd=8.0)

    gen = bo.run()
    correction = next(gen)
    try:
        while True:
            miss = float(np.linalg.norm(np.asarray(correction) - TRUE_OPT))
            miss = max(miss + rng.normal(0, sigma_noise), 0.01)
            correction = gen.send(miss)
    except StopIteration:
        pass
    return bo


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: _early_stopped / _early_stop_iter initialise correctly
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Test 1: Early-stop attributes initialise correctly")
bo0 = BayesianOptimizer(memory=EngagementMemory())
record("_early_stopped is False on construction", bo0._early_stopped is False)
record("_early_stop_iter is -1 on construction", bo0._early_stop_iter == -1)

bo0._early_stopped, bo0._early_stop_iter = True, 7
bo0.reset(range_m=200.0, height_m=0.0, pitch_cmd=6.0)
record("_early_stopped reset to False by reset()", bo0._early_stopped is False)
record("_early_stop_iter reset to -1 by reset()", bo0._early_stop_iter == -1)


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Low-noise scenario ⇒ BO converges and stops early
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Test 2: BO early-stops on a low-noise, easily-modelled miss landscape")
N_INIT, N_SUGGEST = 4, 16
bo_lo = run_with_synthetic_miss(seed=1, sigma_noise=0.05, n_init=N_INIT, n_suggest=N_SUGGEST)

record("BO sets _early_stopped=True under low noise", bo_lo._early_stopped,
       f"_early_stopped={bo_lo._early_stopped}")
record("BO early-stops before exhausting the suggestion budget",
       0 <= bo_lo._early_stop_iter < (N_INIT + N_SUGGEST),
       f"_early_stop_iter={bo_lo._early_stop_iter}")
record("BO early-stops at/after n_init (no early stop during exploration phase)",
       bo_lo._early_stop_iter >= N_INIT,
       f"_early_stop_iter={bo_lo._early_stop_iter} < n_init={N_INIT}")
print(f"    Early stop at iteration {bo_lo._early_stop_iter} "
      f"(budget was {N_INIT + N_SUGGEST}) — best_miss={bo_lo.best_miss:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: GP sigma at the reported stop point is indeed below threshold
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Test 3: GP posterior sigma at stop point is below the convergence threshold")
x_best = np.array(bo_lo.best_correction).reshape(1, -1)
_, sigma_at_stop = bo_lo.gp.predict(x_best)
sigma_at_stop = float(sigma_at_stop[0])
record("sigma at best correction < GP_SIGMA_CONVERGED_THRESHOLD",
       sigma_at_stop < GP_SIGMA_CONVERGED_THRESHOLD,
       f"sigma={sigma_at_stop:.4f} >= threshold={GP_SIGMA_CONVERGED_THRESHOLD}")
print(f"    sigma at best correction = {sigma_at_stop:.4f} m "
      f"(threshold = {GP_SIGMA_CONVERGED_THRESHOLD} m)")


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
