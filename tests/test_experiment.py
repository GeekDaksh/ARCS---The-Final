"""
ARCS — Experiment Module Test Suite
Covers the research benchmark functions used for ablation studies,
learning curves, and baseline comparisons.

Sections (6 total, ~35 tests):
    [1] fixed_target_set — reproducible and bounded
    [2] physics_only_cep — returns float, non-zero for reachable
    [3] RandomSearchBaseline — result dict completeness
    [4] GridSearchBaseline — result dict completeness
    [5] has_converged — correct logic at boundary conditions
    [6] run_ablation — all 4 conditions run, DataFrame complete
"""

import numpy as np
import pandas as pd
import sys, os, time, datetime, tempfile
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiment import (fixed_target_set, physics_only_cep,
                        RandomSearchBaseline, GridSearchBaseline,
                        has_converged, run_ablation)

RNG  = np.random.default_rng(int(time.time()))
PASS = 0
FAIL = 0
LOG  = []

def check(name, cond, detail=""):
    global PASS, FAIL
    status = "PASS" if cond else "FAIL"
    if cond: PASS += 1
    else:    FAIL += 1
    print(f"  {'✓' if cond else '✗'} {status:4s}  {name:60s} {detail}")
    LOG.append({"test": name, "status": status,
                "detail": detail, "time": datetime.datetime.now()})

print("=" * 70)
print("ARCS — Experiment Module Test Suite")
print(f"Seed: {int(time.time())}  (new random values every run)")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] FIXED_TARGET_SET — reproducible and bounded")

targets_a = fixed_target_set(seed=99, n=10)
targets_b = fixed_target_set(seed=99, n=10)   # same seed → identical
targets_c = fixed_target_set(seed=42, n=10)   # different seed → different

check("Returns correct number of targets",
      len(targets_a) == 10, f"got {len(targets_a)}")
check("Each target is a 3-tuple",
      all(len(t) == 3 for t in targets_a))
check("Same seed → identical target sets",
      all(abs(a[0]-b[0]) < 1e-9 for a,b in zip(targets_a, targets_b)))
check("Different seeds → different target sets",
      not all(abs(a[0]-b[0]) < 1e-9 for a,b in zip(targets_a, targets_c)))
check("All x coordinates in [80, 420]",
      all(80 <= t[0] <= 420 for t in targets_a),
      f"x range=[{min(t[0] for t in targets_a):.0f},{max(t[0] for t in targets_a):.0f}]")
check("All y coordinates in [-10, 30]",
      all(-10 <= t[1] <= 30 for t in targets_a))
check("All z coordinates in [-150, 150]",
      all(-150 <= t[2] <= 150 for t in targets_a))

# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] PHYSICS_ONLY_CEP — returns float, non-zero for reachable")

# A definitely reachable mid-range target
cep_mid = physics_only_cep(200.0, 0.0, 0.0, v0=100.0, seed=42)
check("physics_only_cep returns a float (not tuple)",
      isinstance(cep_mid, float),
      f"type={type(cep_mid).__name__}  val={cep_mid:.3f}")
check("physics_only_cep > 0 for reachable target",
      cep_mid > 0.0,
      f"cep={cep_mid:.3f}m")
check("physics_only_cep < 50m (not explosive for mid-range)",
      cep_mid < 50.0,
      f"cep={cep_mid:.3f}m")

# Unreachable target (too far)
cep_far = physics_only_cep(600.0, 0.0, 0.0, v0=50.0, seed=42)
check("physics_only_cep returns None for unreachable target",
      cep_far is None,
      f"got {cep_far!r}")

# Multiple seeds give different results (stochastic noise)
cep1 = physics_only_cep(200.0, 0.0, 0.0, v0=100.0, seed=1)
cep2 = physics_only_cep(200.0, 0.0, 0.0, v0=100.0, seed=2)
check("Different seeds give different CEP values (stochastic)",
      abs(cep1 - cep2) > 0.01 if (cep1 and cep2) else True,
      f"seed1={cep1:.3f}m  seed2={cep2:.3f}m" if (cep1 and cep2) else "skip")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] RANDOMSEARCHBASELINE — result dict completeness")

rs = RandomSearchBaseline(n_candidates=8, n_avg=3, seed=42)
_tx = float(RNG.uniform(120, 300))
_ty = float(RNG.uniform(-5, 15))
_tz = float(RNG.uniform(-80, 80))
r_rs = rs.run_engagement(_tx, _ty, _tz, v0=100.0, seed=42)

check("RandomSearch returns a dict for reachable target",
      r_rs is not None and isinstance(r_rs, dict),
      f"got {type(r_rs).__name__}")
if r_rs is not None:
    check("RandomSearch result has 'baseline_cep'",
          "baseline_cep" in r_rs and isinstance(r_rs["baseline_cep"], float),
          f"baseline={r_rs['baseline_cep']:.3f}m")
    check("RandomSearch result has 'verified_cep'",
          "verified_cep" in r_rs and isinstance(r_rs["verified_cep"], float),
          f"verified={r_rs['verified_cep']:.3f}m")
    check("RandomSearch result has 'improvement_pct'",
          "improvement_pct" in r_rs,
          f"imp={r_rs['improvement_pct']:+.1f}%")
    check("RandomSearch result has 'method'=='random_search'",
          r_rs.get("method") == "random_search")
    check("RandomSearch baseline_cep > 0",
          r_rs["baseline_cep"] > 0,
          f"baseline={r_rs['baseline_cep']:.3f}m")

# Unreachable target returns None
r_rs_none = rs.run_engagement(600.0, 0.0, 0.0, v0=50.0)
check("RandomSearch returns None for unreachable target",
      r_rs_none is None)

# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] GRIDSEARCHBASELINE — result dict completeness")

gs = GridSearchBaseline(n_grid=2, n_avg=3, seed=42)
r_gs = gs.run_engagement(_tx, _ty, _tz, v0=100.0, seed=42)

check("GridSearch returns a dict for reachable target",
      r_gs is not None and isinstance(r_gs, dict))
if r_gs is not None:
    check("GridSearch result has 'baseline_cep'",
          "baseline_cep" in r_gs and isinstance(r_gs["baseline_cep"], float),
          f"baseline={r_gs['baseline_cep']:.3f}m")
    check("GridSearch result has 'verified_cep'",
          "verified_cep" in r_gs and r_gs["verified_cep"] > 0,
          f"verified={r_gs['verified_cep']:.3f}m")
    check("GridSearch result has 'method'=='grid_search'",
          r_gs.get("method") == "grid_search")
    check("GridSearch n_grid=2 → 2³=8 grid points evaluated",
          r_gs.get("total_shots", 0) >= 2**3 * gs.n_avg,
          f"shots={r_gs.get('total_shots', 0)}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] HAS_CONVERGED — boundary conditions")

# Empty records → not converged
check("has_converged([]) → False (not enough data)",
      not has_converged([]))
check("has_converged(4 records) → False (window=5)",
      not has_converged([{"improvement_pct": 0.5}] * 4))

# 5 records below threshold → converged
_conv_recs = [{"improvement_pct": 0.5}] * 5
check("has_converged(5 records at 0.5%) → True (threshold=2.0)",
      has_converged(_conv_recs))

# 5 records above threshold → not converged
_not_conv = [{"improvement_pct": 15.0}] * 5
check("has_converged(5 records at 15%) → False",
      not has_converged(_not_conv))

# Convergence on last 5 even with non-converged earlier records
_mixed = [{"improvement_pct": 30.0}] * 10 + [{"improvement_pct": 0.3}] * 5
check("has_converged with mixed: converges on last 5",
      has_converged(_mixed))

# Custom window and threshold
check("has_converged(window=3, threshold=5.0) with 3 records at 2%",
      has_converged([{"improvement_pct": 2.0}] * 3, window=3, threshold_pct=5.0))
check("has_converged(window=3, threshold=1.0) with 3 records at 2%",
      not has_converged([{"improvement_pct": 2.0}] * 3, window=3, threshold_pct=1.0))

# Negative improvement (harmful corrections) — mean is still near zero → converged
_neg_recs = [{"improvement_pct": -0.5}] * 5
check("has_converged: near-zero negative improvement also converges",
      has_converged(_neg_recs))

# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] RUN_ABLATION — all 4 conditions run, DataFrame complete")
# Use a small target set and few shots so the test completes quickly
_ablation_targets = fixed_target_set(seed=42, n=4)
_df_abl = run_ablation(_ablation_targets, v0=100.0, seed=42, verbose=False)

check("run_ablation returns a DataFrame",
      isinstance(_df_abl, pd.DataFrame),
      f"type={type(_df_abl).__name__}")
check("run_ablation has rows",
      len(_df_abl) > 0,
      f"rows={len(_df_abl)}")

_REQUIRED_COLS = ["condition", "label", "engagement",
                  "baseline_cep_m", "verified_cep_m", "improvement_pct"]
check("run_ablation DataFrame has all required columns",
      all(c in _df_abl.columns for c in _REQUIRED_COLS),
      f"missing={[c for c in _REQUIRED_COLS if c not in _df_abl.columns]}")

_conditions = _df_abl["condition"].unique().tolist()
check("run_ablation produces all 4 conditions (A,B,C,D)",
      set(_conditions) == {"A", "B", "C", "D"},
      f"got {sorted(_conditions)}")
check("Condition A: improvement_pct == 0 (no correction)",
      (_df_abl[_df_abl["condition"] == "A"]["improvement_pct"] == 0.0).all(),
      f"A_imps={_df_abl[_df_abl['condition']=='A']['improvement_pct'].tolist()}")
check("All baseline_cep_m > 0",
      (_df_abl["baseline_cep_m"] > 0).all(),
      f"min={_df_abl['baseline_cep_m'].min():.3f}m")
check("All verified_cep_m > 0",
      (_df_abl["verified_cep_m"] > 0).all(),
      f"min={_df_abl['verified_cep_m'].min():.3f}m")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  TOTAL: {PASS+FAIL}  |  PASSED: {PASS}  |  FAILED: {FAIL}")
print(f"{'='*70}")

import os as _os
_os.makedirs(_os.path.join(_os.path.dirname(__file__), '..', 'data'), exist_ok=True)
_out = _os.path.join(_os.path.dirname(__file__), '..', 'data',
                     'test_results_experiment.csv')
_df_log = pd.DataFrame(LOG)
if _os.path.exists(_out):
    _df_log = pd.concat([pd.read_csv(_out), _df_log], ignore_index=True)
_df_log.to_csv(_out, index=False)
print(f"\n  Results saved → data/test_results_experiment.csv")
print(f"  ({len(_df_log)} total records across all runs)")
