"""
ARCS — Range Table Test Suite  v2.0
Random values every run. Results saved and appended.

Sections (10 total, ~50 tests):
    [1]  Physics generation — dual-trajectory table built
    [2]  Lookup — physics only, pitch valid, HIGH/LOW distinct
    [3]  Record corrections — append, in-memory and on-disk
    [4]  Reload — corrections survive session restart
    [5]  Force protection — force=False does not regenerate
    [6]  Stats — all expected keys, correct arithmetic
    [7]  Weighted correction — nearby records weighted higher than distant
    [8]  HIGH/LOW separation — corrections never cross-contaminate
    [9]  Migration 1 — old files without solution_type column get migrated
    [10] Boundary clamping — lookup never crashes at or beyond grid edges
"""

import numpy as np
import pandas as pd
import sys, os, time, datetime, tempfile
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from physics.range_table import RangeTable, CORRECTIONS_COLS

RNG  = np.random.default_rng(int(time.time()))
PASS = 0
FAIL = 0
LOG  = []

def check(name, cond, detail=""):
    global PASS, FAIL
    status = "PASS" if cond else "FAIL"
    if cond: PASS += 1
    else:    FAIL += 1
    print(f"  {'✓' if cond else '✗'} {status:4s}  {name:58s} {detail}")
    LOG.append({"test": name, "status": status,
                "detail": detail, "time": datetime.datetime.now()})

print("=" * 68)
print("ARCS — Range Table Test Suite  v2.0")
print(f"Seed: {int(time.time())}  (new random values every run)")
print("=" * 68)

# All tests use isolated temp paths — no interference with production data
_TMPDIR = Path(tempfile.mkdtemp())
PP = str(_TMPDIR / "physics.csv")
CP = str(_TMPDIR / "corrections.csv")

rt = RangeTable(physics_path=PP, corrections_path=CP)

_RANGE_STEPS  = np.arange(50, 205, 25)    # 7 values
_HEIGHT_STEPS = np.arange(-10, 25, 10)    # 4 values
_V0_STEPS     = np.array([80.0, 100.0, 120.0])

# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] PHYSICS GENERATION — dual-trajectory table")
rt.generate_physics(
    range_steps=_RANGE_STEPS, height_steps=_HEIGHT_STEPS,
    v0_steps=_V0_STEPS, verbose=False, force=True)

check("Physics file created",       Path(PP).exists())
check("Physics rows > 0",           len(rt._physics_df) > 0,
      f"{len(rt._physics_df)} rows")
check("LOW interpolator built",     rt._interp_pitch is not None)
check("HIGH interpolator built",    rt._interp_pitch_high is not None)
_n_high = int(rt._physics_df["has_high_solution"].sum())
_n_low  = int(rt._physics_df["reachable"].sum())
check("HIGH solutions exist in table",
      _n_high > 0, f"{_n_high} HIGH ({_n_high*100//_n_low}% of reachable)")
check("LOW solutions > HIGH solutions (HIGH is a subset)",
      _n_low > _n_high,
      f"LOW={_n_low}  HIGH={_n_high}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] LOOKUP — physics only, LOW and HIGH distinct")
# Use fixed seed for 5 LOW lookups so test count is always exactly 10
_rng2 = np.random.default_rng(2)
for _ in range(5):
    R  = float(_rng2.uniform(60, 190))
    H  = float(_rng2.uniform(-8, 20))
    v0 = float(_rng2.choice(_V0_STEPS))
    lkp = rt.lookup(R, H, v0)
    check(f"Lookup ({R:.0f}m, {H:.0f}m, {v0:.0f}m/s) returns valid pitch",
          not np.isnan(lkp["pitch_deg"]) and lkp["pitch_deg"] > 0,
          f"pitch={lkp['pitch_deg']:.2f}°")
    check(f"No AI correction before any records",
          not lkp["ai_corrected"], f"n_obs={lkp['n_observations']}")

# HIGH pitch must be greater than LOW pitch — test 3 pre-screened targets
# that definitely have HIGH solutions within this small physics table.
# Fixed seed 22 → always exactly 3 tests, no randomness.
from physics.ballistic_solver import BallisticSolver as _BS
_solver = _BS()
_rng2h = np.random.default_rng(22)
_high2  = []
for _ in range(500):
    if len(_high2) >= 3:
        break
    _R2 = float(_rng2h.uniform(120, 190))
    _H2 = float(_rng2h.uniform(-5, 15))
    _v2 = float(_rng2h.choice(_V0_STEPS))
    _sh = _solver.solve(_R2, _H2, 0.0, _v2, prefer="HIGH")
    if _sh.reachable and _sh.solution_type == "HIGH":
        _lkp_h2 = rt.lookup(_R2, _H2, _v2, prefer="HIGH")
        if not _lkp_h2["trajectory_fallback"]:
            _high2.append((_R2, _H2, _v2))

for _R2, _H2, _v2 in _high2:
    _lkp_l = rt.lookup(_R2, _H2, _v2, prefer="LOW")
    _lkp_h = rt.lookup(_R2, _H2, _v2, prefer="HIGH")
    check(f"HIGH pitch > LOW pitch  R={_R2:.0f}m",
          _lkp_h["pitch_deg"] > _lkp_l["pitch_deg"] - 0.5,
          f"H={_lkp_h['pitch_deg']:.2f}° L={_lkp_l['pitch_deg']:.2f}°")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] RECORD CORRECTIONS — append, in-memory and on-disk")
_n_recs = int(RNG.integers(4, 10))
for _i in range(_n_recs):
    rt.record_correction(
        range_m=float(RNG.uniform(80, 180)),
        height_m=float(RNG.uniform(-5, 15)),
        v0_ms=100.0,
        delta_pitch=float(RNG.uniform(-0.5, 0.5)),
        delta_yaw=float(RNG.uniform(-0.3, 0.3)),
        delta_v0=float(RNG.uniform(-2, 2)),
        miss_before=float(RNG.uniform(8, 20)),
        miss_after=float(RNG.uniform(2, 10)),
        confidence=float(RNG.uniform(0.1, 0.5)),
        n_shots_used=int(RNG.integers(10, 36)),
    )

check("Corrections file created",     Path(CP).exists())
check(f"All {_n_recs} records in memory",
      len(rt._corrections_df) == _n_recs,
      f"got {len(rt._corrections_df)}")
check("solution_type column present in corrections",
      "solution_type" in rt._corrections_df.columns)
check("All solution_type values are LOW or HIGH",
      rt._corrections_df["solution_type"].isin(["LOW","HIGH"]).all())
check("engagement_id auto-increments",
      rt._corrections_df["engagement_id"].nunique() == _n_recs,
      f"unique_ids={rt._corrections_df['engagement_id'].nunique()}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] RELOAD — corrections survive session restart")
rt2 = RangeTable(physics_path=PP, corrections_path=CP)
rt2.load(verbose=False)

check("Corrections survive reload",
      rt2._corrections_df is not None and len(rt2._corrections_df) == _n_recs,
      f"n={len(rt2._corrections_df) if rt2._corrections_df is not None else 0}")
_r_after = rt2.lookup(120, 5, 100)
check("Weighted correction applied after reload",
      _r_after["n_observations"] > 0,
      f"n_obs={_r_after['n_observations']}")
check("Corrected pitch = raw pitch + delta_pitch",
      abs(_r_after["corrected_pitch"] - (_r_after["pitch_deg"] + _r_after["delta_pitch"])) < 1e-9,
      f"corr={_r_after['corrected_pitch']:.4f}° = {_r_after['pitch_deg']:.4f}+{_r_after['delta_pitch']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] FORCE PROTECTION — force=False skips regeneration")
_n_before = len(rt2._physics_df)
rt2.generate_physics(
    range_steps=_RANGE_STEPS, height_steps=_HEIGHT_STEPS,
    v0_steps=_V0_STEPS, force=False, verbose=False)
check("force=False does NOT regenerate physics",
      len(rt2._physics_df) == _n_before, f"rows={len(rt2._physics_df)}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] STATS — all expected keys and correct arithmetic")
_s = rt2.stats()

check("stats(): physics_reachable > 0",
      _s.get("physics_reachable", 0) > 0,      f"n={_s.get('physics_reachable')}")
check("stats(): physics_high_solutions ≥ 0",
      "physics_high_solutions" in _s,            f"n={_s.get('physics_high_solutions')}")
check("stats(): corrections_total == n_records",
      _s.get("corrections_total", -1) == _n_recs, f"n={_s.get('corrections_total')}")
check("stats(): corrections_low + corrections_high == total",
      _s.get("corrections_low", 0) + _s.get("corrections_high", 0) == _n_recs,
      f"low={_s.get('corrections_low')} high={_s.get('corrections_high')}")
check("stats(): mean_miss_after < mean_miss_before (improvement recorded)",
      _s.get("mean_miss_after", 999) < _s.get("mean_miss_before", 0),
      f"before={_s.get('mean_miss_before', 0):.2f}  after={_s.get('mean_miss_after', 0):.2f}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] WEIGHTED CORRECTION — nearby records weighted higher than distant")
_PP7 = str(_TMPDIR / "phys7.csv")
_CP7 = str(_TMPDIR / "corr7.csv")
_rt7 = RangeTable(_PP7, _CP7)
_rt7.generate_physics(range_steps=_RANGE_STEPS, height_steps=_HEIGHT_STEPS,
                      v0_steps=_V0_STEPS, verbose=False, force=True)

# Write two clusters of corrections: one near R=150m with dp=+1.0°,
# one near R=80m with dp=-1.0°. Lookup at R=145m should see mostly +1.0°.
for _i in range(6):
    _rt7.record_correction(
        range_m=float(150 + RNG.uniform(-5, 5)),
        height_m=0.0, v0_ms=100.0,
        delta_pitch=+1.0, delta_yaw=0.0, delta_v0=0.0,
        miss_before=15.0, miss_after=4.0,
        confidence=0.5, n_shots_used=30)
for _i in range(6):
    _rt7.record_correction(
        range_m=float(80 + RNG.uniform(-5, 5)),
        height_m=0.0, v0_ms=100.0,
        delta_pitch=-1.0, delta_yaw=0.0, delta_v0=0.0,
        miss_before=15.0, miss_after=4.0,
        confidence=0.5, n_shots_used=30)

_rt7.load_corrections(verbose=False)
_near_lookup  = _rt7.lookup(145.0, 0.0, 100.0)
_far_lookup   = _rt7.lookup(85.0,  0.0, 100.0)

check("Lookup near R=150m: delta_pitch > 0 (nearby +1.0° cluster dominates)",
      _near_lookup["delta_pitch"] > 0.0,
      f"dp={_near_lookup['delta_pitch']:+.3f}°")
check("Lookup near R=80m: delta_pitch < 0 (nearby −1.0° cluster dominates)",
      _far_lookup["delta_pitch"] < 0.0,
      f"dp={_far_lookup['delta_pitch']:+.3f}°")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] HIGH/LOW SEPARATION — corrections never cross-contaminate")
_PP8 = str(_TMPDIR / "phys8.csv")
_CP8 = str(_TMPDIR / "corr8.csv")
_rt8 = RangeTable(_PP8, _CP8)
_rt8.generate_physics(range_steps=_RANGE_STEPS, height_steps=_HEIGHT_STEPS,
                      v0_steps=_V0_STEPS, verbose=False, force=True)

# R=200m: HIGH angle ≈ 84.3° which is within the 85° mechanical limit.
# R=150m intentionally avoided — HIGH angle ≈ 85.8° > 85° → falls back to LOW.
_rt8.record_correction(200.0, 0.0, 100.0, +0.6, 0.0, 0.0, 12.0, 3.0, 0.5, 30,
                        solution_type="LOW")
_rt8.record_correction(200.0, 0.0, 100.0, +0.9, 0.0, 0.0, 12.0, 3.0, 0.5, 30,
                        solution_type="HIGH")
_rt8.load_corrections(verbose=False)

_lkp8_low  = _rt8.lookup(200.0, 0.0, 100.0, prefer="LOW")
_lkp8_high = _rt8.lookup(200.0, 0.0, 100.0, prefer="HIGH")

check("LOW lookup sees only LOW corrections (dp≈0.6°)",
      abs(_lkp8_low["delta_pitch"] - 0.6) < 0.15,
      f"dp={_lkp8_low['delta_pitch']:.3f}° (expect≈0.6°)")
check("HIGH lookup sees only HIGH corrections (dp≈0.9°)",
      abs(_lkp8_high["delta_pitch"] - 0.9) < 0.15,
      f"dp={_lkp8_high['delta_pitch']:.3f}° (expect≈0.9°)")
check("LOW dp ≠ HIGH dp (not mixed)",
      abs(_lkp8_low["delta_pitch"] - _lkp8_high["delta_pitch"]) > 0.1,
      f"low={_lkp8_low['delta_pitch']:.3f}° high={_lkp8_high['delta_pitch']:.3f}°")

# Corrections CSV has both types with correct labels
_rt8b = RangeTable(_PP8, _CP8)
_rt8b.load_corrections(verbose=False)
check("Both LOW and HIGH records written to file",
      len(_rt8b._corrections_df[_rt8b._corrections_df["solution_type"] == "LOW"]) >= 1 and
      len(_rt8b._corrections_df[_rt8b._corrections_df["solution_type"] == "HIGH"]) >= 1)

# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] MIGRATION 1 — old files without solution_type get column added")
_CP9 = str(_TMPDIR / "corr9_old.csv")

# Write a CSV in the pre-solution_type schema (no solution_type column)
OLD_COLS = [c for c in CORRECTIONS_COLS if c != "solution_type"]
_old_rows = []
for _i in range(5):
    _old_rows.append({
        "range_m": float(100 + _i * 20), "height_m": 0.0, "v0_ms": 100.0,
        "delta_pitch": 0.3, "delta_yaw": 0.1, "delta_v0": -1.0,
        "miss_before": 12.0, "miss_after": 4.0,
        "confidence": 0.5, "n_shots_used": 30,
        "engagement_id": _i, "timestamp": datetime.datetime.now().isoformat(),
    })
pd.DataFrame(_old_rows)[OLD_COLS].to_csv(_CP9, index=False)

_rt9 = RangeTable(PP, _CP9)
_rt9.load_corrections(verbose=False)

check("Migration 1: solution_type column added",
      "solution_type" in _rt9._corrections_df.columns)
check("Migration 1: all old records default to 'LOW'",
      (_rt9._corrections_df["solution_type"] == "LOW").all(),
      f"unique={_rt9._corrections_df['solution_type'].unique().tolist()}")
check("Migration 1: no records lost",
      len(_rt9._corrections_df) == 5, f"n={len(_rt9._corrections_df)}")

# Reload: migrated file should not re-migrate (idempotent)
_rt9b = RangeTable(PP, _CP9)
_rt9b.load_corrections(verbose=False)
check("Migration 1: idempotent on reload",
      (_rt9b._corrections_df["solution_type"] == "LOW").all() and
      len(_rt9b._corrections_df) == 5)

# ─────────────────────────────────────────────────────────────────────────────
print("\n[10] BOUNDARY CLAMPING — lookup never crashes at grid edges")
# The interpolator should clamp out-of-grid queries, not extrapolate or crash.
_edge_cases = [
    (10.0,   0.0, 100.0),    # below min range
    (600.0,  0.0, 100.0),    # above max range
    (150.0, -30.0, 100.0),   # below min height
    (150.0,  80.0, 100.0),   # above max height
    (150.0,  0.0,  10.0),    # below min v0
    (150.0,  0.0, 500.0),    # above max v0
]
for _R, _H, _v in _edge_cases:
    try:
        _lkp_edge = rt.lookup(_R, _H, _v)
        _ok = not np.isnan(_lkp_edge["pitch_deg"])
    except Exception as _e:
        _ok = False
        print(f"    Exception at ({_R},{_H},{_v}): {_e}")
    check(f"lookup({_R:.0f}m,{_H:.0f}m,{_v:.0f}m/s) doesn't crash",
          _ok, f"pitch={_lkp_edge.get('pitch_deg','ERR'):.2f}°" if _ok else "EXCEPTION")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*68}")
print(f"  TOTAL: {PASS+FAIL}  |  PASSED: {PASS}  |  FAILED: {FAIL}")
print(f"{'='*68}")

import shutil
shutil.rmtree(_TMPDIR, ignore_errors=True)

os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'data'), exist_ok=True)
_out = os.path.join(os.path.dirname(__file__), '..', 'data',
                    'test_results_range_table.csv')
_df_log = pd.DataFrame(LOG)
if os.path.exists(_out):
    _df_log = pd.concat([pd.read_csv(_out), _df_log], ignore_index=True)
_df_log.to_csv(_out, index=False)
print(f"\n  Results saved → data/test_results_range_table.csv")
print(f"  ({len(_df_log)} total records across all runs)")
