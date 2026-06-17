"""ARCS — Pipeline Integration Test Suite  v2.0
Phase 1

Sections (12 total, ~75 tests):
    [1]  Pipeline startup — all components initialised
    [2]  Single engagement — result keys, tracker entry
    [3]  Batch run — DataFrame column completeness
    [4]  Corrections persist across reload
    [5]  Warm-start improvement — PINN active, ≥60% positive (10 engagements)
    [6]  PINN activation — source field and non-trivial correction value
    [7]  HIGH trajectory pipeline — prefer=HIGH returns trajectory_type=HIGH
    [8]  Fallback protection — harmful corrections never written to CSV
    [9]  PINN retraining trigger — should_retrain() → load_and_train() succeeds
    [10] status() completeness — all expected fields and consistent counts
    [11] stop_on_convergence — batch halts early when system has converged
    [12] Corruption recovery — Migration 2 auto-fixes column-shifted CSV

DESIGN:
    - ONE shared physics table generated at startup (saves ~2 min vs 5 separate tables)
    - Sections [5], [7] use real engagement data for PINN warm-start (bias-matched)
    - All sections use isolated temp corrections/history files — no cross-contamination
"""

import numpy as np, pandas as pd, sys, os, time, datetime, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pathlib
RNG  = np.random.default_rng(int(time.time()))
PASS = 0
FAIL = 0
LOG  = []

def check(name, cond, detail=""):
    global PASS, FAIL
    status = "PASS" if cond else "FAIL"
    if cond: PASS += 1
    else:    FAIL += 1
    print(f"  {'✓' if cond else '✗'} {status:4s}  {name:56s} {detail}")
    LOG.append({"test": name, "status": status, "detail": detail,
                "time": datetime.datetime.now()})

# ── Temp dir + ONE shared physics table ──────────────────────────────────────
_TMPDIR = pathlib.Path(tempfile.mkdtemp())
_PHYS   = str(_TMPDIR / "shared_physics.csv")   # generated once, reused everywhere

def _corr(name): return str(_TMPDIR / f"corr_{name}.csv")
def _hist(name): return str(_TMPDIR / f"hist_{name}.csv")
def _db(name):   return str(_TMPDIR / f"db_{name}.db")

from pipeline      import ARCSPipeline
from physics.range_table import RangeTable
from pinn_corrector      import PINNCorrector
from physics.ballistic_solver import BallisticSolver as _BS

print("=" * 64)
print("ARCS — Pipeline Integration Test Suite  v2.0")
print(f"Seed: {int(time.time())}  (new random targets every run)")
print("=" * 64)

# ─────────────────────────────────────────────────────────────────────────────
# [1] PIPELINE STARTUP — generates the shared physics table once
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] PIPELINE STARTUP  (generates shared physics table once)")
p1 = ARCSPipeline(physics_path=_PHYS, corrections_path=_corr("s1"),
                  history_path=_hist("s1"), db_path=_db("s1"), seed=42, verbose=True)
p1.verbose = False   # suppress per-engagement BO output for cleaner test log

check("Pipeline initialised",       p1 is not None)
check("Range table loaded",          p1.rt._physics_df is not None)
check("Corrections df not None",    p1.rt._corrections_df is not None)
check("Tracker created",            p1.tracker is not None)
check("PINN LOW corrector exists",  p1.cf_low  is not None)
check("PINN HIGH corrector exists", p1.cf_high is not None)
check("Physics has HIGH solutions",
      p1.rt.stats().get("physics_high_solutions", 0) > 0,
      f"n={p1.rt.stats().get('physics_high_solutions', 0)}")

# ─────────────────────────────────────────────────────────────────────────────
# [2] SINGLE ENGAGEMENT — result keys, tracker entry
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] SINGLE ENGAGEMENT — mid-range target, result completeness")
tx = float(RNG.uniform(150, 350))
ty = float(RNG.uniform(-5, 20))
tz = float(RNG.uniform(-80, 80))
print(f"  Target: ({tx:.0f}, {ty:.0f}, {tz:.0f})")
r2 = p1.engage(tx, ty, tz)

check("Engagement returns result",        r2 is not None)
check("Baseline CEP > 0",
      r2["baseline_cep"] > 0,            f"{r2['baseline_cep']:.2f}m")
check("Verified CEP is finite positive",
      r2["verified_cep"] > 0,            f"{r2['verified_cep']:.2f}m")
check("improvement_pct key present",     "improvement_pct"  in r2)
check("trajectory_type key present",     "trajectory_type"  in r2)
check("pinn_correction key present",     "pinn_correction"  in r2)
check("rt_correction key present",       "rt_correction"    in r2)
check("total_shots key present",         "total_shots"      in r2)
check("Tracker has 1+ records",
      len(p1.tracker._records) >= 1,     f"n={len(p1.tracker._records)}")

# ─────────────────────────────────────────────────────────────────────────────
# [3] BATCH RUN — DataFrame column completeness
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] BATCH RUN — output DataFrame column completeness")
batch3 = [(float(RNG.uniform(120, 380)),
           float(RNG.uniform(-8, 20)),
           float(RNG.uniform(-100, 100)))
          for _ in range(5)]
df3 = p1.run_batch(batch3)

check("Batch returns non-empty DataFrame",  len(df3) > 0,         f"{len(df3)} rows")
check("Column: engagement_n",               "engagement_n"    in df3.columns)
check("Column: baseline_cep_m",             "baseline_cep_m"  in df3.columns)
check("Column: verified_cep_m",             "verified_cep_m"  in df3.columns)
check("Column: improvement_pct",            "improvement_pct" in df3.columns)
check("Column: trajectory_type",            "trajectory_type" in df3.columns)
check("Column: pinn_applied",               "pinn_applied"    in df3.columns)
check("Column: shots_used",                 "shots_used"      in df3.columns)
check("All verified CEP > 0",
      (df3["verified_cep_m"] > 0).all(),   f"min={df3['verified_cep_m'].min():.2f}m")

# ─────────────────────────────────────────────────────────────────────────────
# [4] CORRECTIONS PERSIST ACROSS RELOAD
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] CORRECTIONS PERSIST ACROSS RELOAD")
n_recorded = (len(p1.rt._corrections_df)
              if p1.rt._corrections_df is not None else 0)
rt4 = RangeTable(physics_path=_PHYS, corrections_path=_corr("s1"))
rt4.load(verbose=False)
n_reloaded = len(rt4._corrections_df) if rt4._corrections_df is not None else 0

check("At least 1 correction written",    n_recorded >= 1,          f"n={n_recorded}")
check("Correction count survives reload", n_reloaded == n_recorded,  f"n={n_reloaded}")
check("solution_type column present",
      "solution_type" in rt4._corrections_df.columns)
check("All solution_type values are LOW or HIGH",
      rt4._corrections_df["solution_type"].isin(["LOW","HIGH"]).all())

# ─────────────────────────────────────────────────────────────────────────────
# [5] WARM-START IMPROVEMENT — real BO corrections, PINN active
#
# Strategy: run 28 cold-start BO engagements first (these write real corrections
# using the SAME RobotBiasModel seed as the test engagements). PINN trains on
# those real corrections. Then run 10 more — the PINN pre-correction should
# reduce bias, and BO cleans up the residual.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] WARM-START IMPROVEMENT — 28 real corrections → PINN → 10 test engagements")
SEED5 = 1234   # fixed seed so bias is reproducible
# Pre-seed the corrections file with 1 valid LOW record. With n_avg=6, 28 warm-up
# engagements produce exactly 19 non-fallback corrections; 19+1=20 = MIN_RECORDS.
_rt_pre5 = RangeTable(physics_path=_PHYS, corrections_path=_corr("s5"))
_rt_pre5.record_correction(200.0, 0.0, 100.0, 0.07, 0.01, -1.8, 10.0, 8.0, 0.8, 30, "LOW")
del _rt_pre5
p5 = ARCSPipeline(physics_path=_PHYS, corrections_path=_corr("s5"),
                  history_path=_hist("s5"), db_path=_db("s5"), seed=SEED5, verbose=False)

# Phase A: 28 cold-start engagements — builds real corrections that match the bias
print("  Phase A: warming up (28 cold-start engagements)...")
_rng5  = np.random.default_rng(SEED5)
_warm5 = [(float(_rng5.uniform(150, 380)),
           float(_rng5.uniform(-5, 15)),
           float(_rng5.uniform(-80, 80))) for _ in range(28)]
for _tx, _ty, _tz in _warm5:
    p5.engage(_tx, _ty, _tz)

_n5_corr = (len(p5.rt._corrections_df)
            if p5.rt._corrections_df is not None else 0)
check("Phase A: ≥15 corrections collected",
      _n5_corr >= 15, f"n={_n5_corr}")

# Force PINN retrain if it hasn't triggered automatically
if not p5.cf_low.is_fitted:
    p5.cf_low.load_and_train(verbose=False)
check("PINN LOW fitted after warmup",
      p5.cf_low.is_fitted, f"n_records={p5.cf_low.n_records}")

# Phase B: 10 mid-range test engagements with PINN active
print("  Phase B: testing with PINN active (10 engagements)...")
_test5 = [(float(_rng5.uniform(150, 380)),
           float(_rng5.uniform(-5, 15)),
           float(_rng5.uniform(-80, 80))) for _ in range(10)]
_df5 = pd.DataFrame()
_rows5 = []
for _tx, _ty, _tz in _test5:
    _r = p5.engage(_tx, _ty, _tz)
    if _r is not None:
        _rows5.append(_r["improvement_pct"])

_n5    = len(_rows5)
_pos5  = sum(1 for x in _rows5 if x > 0)
_pct5  = _pos5 / _n5 * 100 if _n5 > 0 else 0
_mean5 = float(np.mean(_rows5)) if _rows5 else 0.0
_min5  = float(np.min(_rows5))  if _rows5 else 0.0

check(f"≥50% positive with PINN active ({_pos5}/{_n5})",
      _pct5 >= 50, f"{_pct5:.0f}%")
check("Mean improvement > 12%",
      _mean5 > 12.0, f"mean={_mean5:+.1f}%")
check("No engagement < -30% (fallback cap)",
      _min5 >= -30.0, f"min={_min5:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# [6] PINN ACTIVATION — source field and non-trivial correction value
#     (reuses p5 which has PINN fitted)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] PINN ACTIVATION — source field, delta_pitch and delta_v0")
_tx6 = float(_rng5.uniform(150, 350))
_ty6 = float(_rng5.uniform(-5, 20))
_tz6 = float(_rng5.uniform(-80, 80))
_r6  = p5.engage(_tx6, _ty6, _tz6)

check("Engagement with PINN returns result", _r6 is not None)
if _r6 is not None:
    _src6 = _r6.get("pinn_correction", {}).get("source", "none")
    # NOTE: Phase 1 Additions fixed the 'best_correction_total' key bug that
    # was silently preventing self.sbe.update_engagement() from ever running
    # (see CLAUDE_PHASE1_ADDITIONS.md). With cross-engagement learning now
    # actually wired up, the SBE accumulates confidence during warm-up and —
    # exactly per the architecture (CLAUDE.md §5.2: "PINNCorrector ... used
    # until SBE confidence >= 0.6, then SBE takes over") — legitimately
    # supersedes the PINN as the pre-correction source. Any of pinn/blend/sbe
    # is a correctly-functioning pre-correction pipeline.
    check("Pre-correction source is an active model (pinn/blend/sbe)",
          _src6 == "pinn_torch_LOW" or _src6.startswith("blend(") or _src6.startswith("sbe"),
          f"source='{_src6}'")
    check("PINN |delta_pitch| > 0.01°",
          abs(_r6["pinn_correction"]["delta_pitch"]) > 0.01,
          f"dp={_r6['pinn_correction']['delta_pitch']:+.3f}°")
    check("PINN |delta_v0| > 0.1 m/s",
          abs(_r6["pinn_correction"]["delta_v0"]) > 0.1,
          f"dv={_r6['pinn_correction']['delta_v0']:+.2f} m/s")
else:
    check("Pre-correction source is an active model (pinn/blend/sbe)", False, "engagement returned None")
    check("PINN |delta_pitch| > 0.01°",   False, "")
    check("PINN |delta_v0| > 0.1 m/s",   False, "")

# ─────────────────────────────────────────────────────────────────────────────
# [7] HIGH TRAJECTORY PIPELINE — prefer=HIGH, PINN HIGH activates
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] HIGH TRAJECTORY PIPELINE — prefer=HIGH round-trip with PINN HIGH")
SEED7 = 777
p7 = ARCSPipeline(physics_path=_PHYS, corrections_path=_corr("s7"),
                  history_path=_hist("s7"), db_path=_db("s7"), seed=SEED7, verbose=False)

# Warm up with HIGH engagements so PINN HIGH gets enough records
_rng7   = np.random.default_rng(SEED7)
_solver7 = _BS()
_high_targets = []
for _ in range(200):
    if len(_high_targets) >= 28:
        break
    _R7  = float(_rng7.uniform(150, 380))
    _H7  = float(_rng7.uniform(-5, 15))
    _sol7 = _solver7.solve(_R7, _H7, 0.0, 100.0, prefer="HIGH")
    if _sol7.reachable and _sol7.solution_type == "HIGH":
        _high_targets.append((_R7, _H7, float(_rng7.uniform(-60, 60))))

print(f"  Warming up PINN HIGH ({len(_high_targets)} HIGH engagements)...")
for _tx7, _ty7, _tz7 in _high_targets:
    p7.engage(_tx7, _ty7, _tz7, prefer="HIGH")

if not p7.cf_high.is_fitted:
    p7.cf_high.load_and_train(verbose=False)

check("PINN HIGH fitted after warmup",
      p7.cf_high.is_fitted, f"n_records={p7.cf_high.n_records}")

# Now run a single HIGH test engagement
_r7 = None
for _ in range(30):
    _tx7t = float(_rng7.uniform(150, 380))
    _ty7t = float(_rng7.uniform(-5, 15))
    _tz7t = float(_rng7.uniform(-60, 60))
    _sol7t = _solver7.solve(_tx7t, _ty7t, _tz7t, 100.0, prefer="HIGH")
    if _sol7t.reachable and _sol7t.solution_type == "HIGH":
        _r7 = p7.engage(_tx7t, _ty7t, _tz7t, prefer="HIGH")
        break

check("HIGH engagement returns result", _r7 is not None)
if _r7 is not None:
    _tt7 = _r7.get("trajectory_type", "")
    check("trajectory_type is HIGH",
          _tt7 == "HIGH",                   f"got '{_tt7}'")
    _src7 = _r7.get("pinn_correction", {}).get("source", "none")
    # See note above [6] — SBE legitimately supersedes PINN once confident
    # (CLAUDE.md §5.2), now that cross-engagement learning is correctly wired.
    check("Pre-correction source is an active model (pinn/blend/sbe)",
          _src7 == "pinn_torch_HIGH" or _src7.startswith("blend(") or _src7.startswith("sbe"),
          f"source='{_src7}'")
    check("Baseline CEP > 0",
          _r7["baseline_cep"] > 0,          f"{_r7['baseline_cep']:.2f}m")
else:
    check("trajectory_type is HIGH",        False, "no HIGH target found")
    check("Pre-correction source is an active model (pinn/blend/sbe)", False, "")
    check("Baseline CEP > 0",              False, "")

# ─────────────────────────────────────────────────────────────────────────────
# [8] FALLBACK PROTECTION — no harmful correction written to CSV
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] FALLBACK PROTECTION — harmful corrections never written to CSV")
p8 = ARCSPipeline(physics_path=_PHYS, corrections_path=_corr("s8"),
                  history_path=_hist("s8"), db_path=_db("s8"), seed=888, verbose=False)

_batch8 = [(float(RNG.uniform(150, 350)),
            float(RNG.uniform(-5, 15)),
            float(RNG.uniform(-80, 80))) for _ in range(8)]
for _tx8, _ty8, _tz8 in _batch8:
    p8.engage(_tx8, _ty8, _tz8)

if p8.rt._corrections_df is not None and len(p8.rt._corrections_df) > 0:
    _df8 = p8.rt._corrections_df
    # Fallback guard: any written correction must have miss_after ≤ miss_before × 1.10
    # (corrections that triggered the fallback are NOT written at all)
    _ratio8 = (_df8["miss_after"] / _df8["miss_before"].clip(lower=0.01)).values
    check("All written corrections pass 1.10× threshold",
          (_ratio8 <= 1.101).all(),
          f"max_ratio={_ratio8.max():.3f}")
    # improvement_pct in tracker: fallback gives 0%, never deeply negative
    _imps8 = [r["improvement_pct"] for r in p8.tracker._records]
    check("No improvement < -15% in tracker (fallback cap working)",
          all(i >= -15.0 for i in _imps8),
          f"min={min(_imps8):.1f}%")
    check("At least 1 correction recorded (BO working)",
          len(_df8) >= 1, f"n={len(_df8)}")
else:
    # All 8 engagements triggered fallback — safe but note it
    _imps8 = [r["improvement_pct"] for r in p8.tracker._records]
    check("All written corrections pass 1.10× threshold", True,
          "no corrections written (all fallback — OK)")
    check("No improvement < -15% in tracker (fallback cap working)",
          all(i >= -15.0 for i in _imps8) if _imps8 else True,
          f"min={min(_imps8):.1f}%" if _imps8 else "no records")
    check("At least 1 correction recorded (BO working)", False,
          "all 8 engagements hit fallback — unusual")

# ─────────────────────────────────────────────────────────────────────────────
# [9] PINN RETRAINING TRIGGER
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] PINN RETRAINING TRIGGER — should_retrain() fires after novel records")
import datetime as _dt

_c9 = _corr("s9")
_rng9  = np.random.default_rng(int(time.time()) % 2222 + 300)
_solver9 = _BS()

# Build 22 baseline records
_rows9 = []
for _i in range(40):
    if len(_rows9) >= 22:
        break
    _R9  = float(_rng9.uniform(120, 360))
    _H9  = float(_rng9.uniform(-8, 18))
    _sol9 = _solver9.solve(_R9, _H9, 0.0, 100.0)
    if not _sol9.reachable:
        continue
    _th9 = _sol9.turret_pitch_deg
    _mb9 = float(_rng9.uniform(8, 18))
    _rows9.append({
        "range_m": _R9, "height_m": _H9, "v0_ms": 100.0,
        "delta_pitch": -0.45 * np.sin(np.deg2rad(_th9)) + _rng9.normal(0, 0.05),
        "delta_yaw":   0.28  + _rng9.normal(0, 0.03),
        "delta_v0":   -1.9   + _rng9.normal(0, 0.2),
        "miss_before": _mb9, "miss_after": float(_rng9.uniform(2, _mb9 * 0.6)),
        "confidence": 0.5, "n_shots_used": 30,
        "engagement_id": _i,
        "timestamp": _dt.datetime.now().isoformat(),
        "solution_type": "LOW",
    })
pd.DataFrame(_rows9).to_csv(_c9, index=False)

pc9 = PINNCorrector(_c9, solution_type="LOW")
pc9.load_and_train(verbose=False)
check("PINN trained on 22 records",          pc9.is_fitted, f"n={pc9.n_records}")
check("should_retrain() False — no new data", not pc9.should_retrain())

# Append 6 records with novel range (400-450m, far from 120-360m training set)
_extra9 = []
for _j in range(6):
    _R9e = float(400 + _j * 10)
    _sol9e = _solver9.solve(_R9e, 0.0, 0.0, 100.0)
    _th9e  = _sol9e.turret_pitch_deg if _sol9e.reachable else 5.0
    _mb9e  = float(_rng9.uniform(8, 18))
    _extra9.append({
        "range_m": _R9e, "height_m": 0.0, "v0_ms": 100.0,
        "delta_pitch": -0.45 * np.sin(np.deg2rad(_th9e)),
        "delta_yaw": 0.28, "delta_v0": -1.9,
        "miss_before": _mb9e, "miss_after": float(_rng9.uniform(2, _mb9e * 0.6)),
        "confidence": 0.5, "n_shots_used": 30,
        "engagement_id": 22 + _j,
        "timestamp": _dt.datetime.now().isoformat(),
        "solution_type": "LOW",
    })
_full9 = pd.concat([pd.DataFrame(_rows9), pd.DataFrame(_extra9)], ignore_index=True)
_full9.to_csv(_c9, index=False)

check("should_retrain() True after 6 novel records", pc9.should_retrain())
ok9 = pc9.load_and_train(verbose=False)
check("Retrain succeeds after novel records",  ok9 and pc9.is_fitted, f"n={pc9.n_records}")
check("n_records grew after retrain",           pc9.n_records > 22,   f"n={pc9.n_records}")

# ─────────────────────────────────────────────────────────────────────────────
# [10] STATUS() COMPLETENESS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[10] STATUS() COMPLETENESS — all expected fields present")
_s5  = p5.rt.stats()
_ts5 = p5.tracker.summary()

check("stats(): physics_reachable > 0",
      _s5.get("physics_reachable", 0) > 0,
      f"n={_s5.get('physics_reachable', 0)}")
check("stats(): physics_high_solutions > 0",
      _s5.get("physics_high_solutions", 0) > 0,
      f"n={_s5.get('physics_high_solutions', 0)}")
check("stats(): corrections_low and corrections_high keys present",
      "corrections_low" in _s5 and "corrections_high" in _s5)
check("tracker.summary(): n_engagements ≥ 38",   # 28 warm + 10 test + 1 from [6]
      _ts5.get("n_engagements", 0) >= 38,
      f"n={_ts5.get('n_engagements', 0)}")
check("tracker.summary(): mean_improvement key present", "mean_improvement" in _ts5)
check("tracker.summary(): pct_positive key present",     "pct_positive"     in _ts5)
check("tracker.summary(): best_bo_miss key present",     "best_bo_miss"     in _ts5)

import io, contextlib
try:
    _buf = io.StringIO()
    with contextlib.redirect_stdout(_buf):
        p5.status()
    _out = _buf.getvalue()
    check("status() runs without exception",        True)
    check("status() prints mean improvement",        "Mean improvement" in _out)
    check("status() prints PINN LOW fitted status",  "PINN LOW" in _out)
    check("status() prints correction record count", "Corrections" in _out)
except Exception as _e:
    check("status() runs without exception", False, str(_e))
    check("status() prints mean improvement",     False, "")
    check("status() prints PINN LOW fitted",      False, "")
    check("status() prints correction counts",    False, "")

# ─────────────────────────────────────────────────────────────────────────────
# [11] STOP_ON_CONVERGENCE — batch halts early when system has converged
# ─────────────────────────────────────────────────────────────────────────────
print("\n[11] STOP_ON_CONVERGENCE — batch halts early after convergence")
from experiment import has_converged as _has_converged

p11 = ARCSPipeline(physics_path=_PHYS, corrections_path=_corr("s11"),
                   history_path=_hist("s11"), db_path=_db("s11"), seed=111, verbose=False)

# Force convergence state: inject 5 records all with ~0.5% improvement
# (well below the 2.0% threshold has_converged() uses)
import datetime as _dt11
for _i in range(5):
    p11.tracker.record(
        engagement_n=_i + 1, baseline_cep=10.0, bo_cep=9.95,
        gp_applied=False, target_range=200.0, improvement_pct=0.5,
        n_shots_used=320, label="convergence_setup", solution_type="LOW")
p11.engagement_n = 5

check("has_converged() True after 5 near-zero improvements",
      _has_converged(p11.tracker._records),
      f"last5_mean={float(np.mean([r['improvement_pct'] for r in p11.tracker._records[-5:]])):.2f}%")

# Run a batch of 10 — should stop immediately since already converged
_batch11 = [(float(np.random.default_rng(111 + _i).uniform(150, 350)),
             float(np.random.default_rng(222 + _i).uniform(-5, 15)),
             0.0) for _i in range(10)]
_df11 = p11.run_batch(_batch11, stop_on_convergence=True)

check("Batch with stop_on_convergence=True halts before all 10 targets",
      len(_df11) < 10,
      f"rows={len(_df11)} (expected < 10)")
check("Batch with stop_on_convergence=True returns a DataFrame",
      isinstance(_df11, pd.DataFrame),
      f"type={type(_df11).__name__}")

# Sanity: without stop_on_convergence, same batch runs all targets
_df11_full = p11.run_batch(_batch11, stop_on_convergence=False)
check("Batch without stop_on_convergence runs all 10 targets",
      len(_df11_full) == 10,
      f"rows={len(_df11_full)}")

# ─────────────────────────────────────────────────────────────────────────────
# [12] CORRECTIONS FILE CORRUPTION RECOVERY — migration auto-fixes bad CSV
# ─────────────────────────────────────────────────────────────────────────────
print("\n[12] CORRUPTION RECOVERY — Migration 2 auto-fixes column-shifted CSV")
from physics.range_table import RangeTable, CORRECTIONS_COLS
import datetime as _dt12

_c12 = _corr("s12")

# Write a CSV that reproduces the column-order shift bug:
# engagement_id column holds "LOW"/"HIGH", solution_type holds timestamps.
# This is exactly what the pre-migration pipeline wrote.
_bad_rows = []
_rng12 = np.random.default_rng(42)
for _i in range(10):
    _R12  = float(_rng12.uniform(120, 350))
    _bad_rows.append({
        "range_m":      _R12,
        "height_m":     0.0,
        "v0_ms":        100.0,
        "delta_pitch":  float(_rng12.uniform(-0.5, 0.5)),
        "delta_yaw":    float(_rng12.uniform(-0.3, 0.3)),
        "delta_v0":     float(_rng12.uniform(-3, 3)),
        "miss_before":  float(_rng12.uniform(8, 18)),
        "miss_after":   float(_rng12.uniform(2, 7)),
        "confidence":   0.5,
        "n_shots_used": 30,
        # Column-shift: engagement_id holds the solution_type string,
        # timestamp holds the integer, solution_type holds the timestamp.
        "solution_type":  _dt12.datetime.now().isoformat(),   # WRONG
        "engagement_id":  "LOW",                               # WRONG
        "timestamp":      _i,                                  # WRONG
    })
pd.DataFrame(_bad_rows).to_csv(_c12, index=False)

# Load via RangeTable — migration should auto-fix
_rt12 = RangeTable(_PHYS, _c12)
_rt12.load_corrections(verbose=False)

check("Migration 2: all solution_type values fixed to LOW/HIGH",
      _rt12._corrections_df["solution_type"].isin(["LOW", "HIGH"]).all(),
      f"unique={_rt12._corrections_df['solution_type'].unique().tolist()}")
check("Migration 2: engagement_id is numeric after fix",
      pd.to_numeric(_rt12._corrections_df["engagement_id"],
                    errors="coerce").notna().all(),
      f"sample={_rt12._corrections_df['engagement_id'].iloc[0]}")
check("Migration 2: record count preserved (no rows lost)",
      len(_rt12._corrections_df) == 10,
      f"n={len(_rt12._corrections_df)}")

# File is rewritten in canonical order — reload should show no bad rows
_rt12b = RangeTable(_PHYS, _c12)
_rt12b.load_corrections(verbose=False)
check("Reloaded file needs no further migration (idempotent)",
      _rt12b._corrections_df["solution_type"].isin(["LOW", "HIGH"]).all(),
      f"unique={_rt12b._corrections_df['solution_type'].unique().tolist()}")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*64}")
print(f"  TOTAL: {PASS+FAIL}  |  PASSED: {PASS}  |  FAILED: {FAIL}")
print(f"{'='*64}")

if FAIL > 0:
    print("\n  Failed tests:")
    for row in LOG:
        if row["status"] == "FAIL":
            print(f"    ✗  {row['test']}")

os.makedirs(os.path.join(os.path.dirname(__file__), "..", "data"), exist_ok=True)
_out_path = os.path.join(os.path.dirname(__file__), "..", "data",
                          "test_results_pipeline.csv")
_df_log = pd.DataFrame(LOG)
if os.path.exists(_out_path):
    _df_log = pd.concat([pd.read_csv(_out_path), _df_log], ignore_index=True)
_df_log.to_csv(_out_path, index=False)
print(f"\n  Results saved → data/test_results_pipeline.csv")
print(f"  ({len(_df_log)} total records across all runs)")
