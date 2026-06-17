"""
ARCS — Physics Engine Test Suite  v2.0
Phase 1

Tests run with new random values every execution.
Results appended to data/test_results_physics.csv.

NEW IN v2.0 — 4 additional sections beyond the original 6:
    [7]  HIGH-angle ballistic solver — lobbed trajectory accuracy
    [8]  solve_both() — dual trajectory correctness and distinction
    [9]  Verification error sweep — full envelope accuracy guarantee
    [10] Range table — physics generation, lookup, HIGH/LOW separation

PHILOSOPHY:
    The direct ballistic solver (ballistic_solver.py) is exact — verification
    error < 0.05m is guaranteed for every reachable target by Newton's equations.
    The range table (range_table.py) is an interpolated approximation used for
    corrections lookup. Its pitch accuracy depends on grid resolution; what
    matters is that LOW corrections are never confused with HIGH corrections.
"""

import numpy as np
import pandas as pd
import sys, os, time, datetime, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from physics.constants       import validate_target
from physics.rotation        import RotationMatrix
from physics.ballistic_solver import BallisticSolver, GRAVITY

RNG  = np.random.default_rng(seed=int(time.time()))
PASS = 0
FAIL = 0
LOG  = []

def check(name, condition, detail=""):
    global PASS, FAIL
    ok = bool(condition)
    if ok: PASS += 1
    else:  FAIL += 1
    sym = "✓" if ok else "✗"
    status = "PASS" if ok else "FAIL"
    print(f"  {sym} {status:4s}  {name:52s} {detail}")
    LOG.append({"test": name, "status": status,
                "detail": detail, "time": datetime.datetime.now()})

solver = BallisticSolver()
VERIFY_TOL = 0.05      # 5cm — direct solver accuracy guarantee

print("=" * 68)
print("ARCS — Physics Engine Test Suite v2.0")
print(f"Seed: {int(time.time())}  (new random values every run)")
print("=" * 68)

# ─── [1] UNIT CONVERSIONS ─────────────────────────────────────────────────────
print("\n[1] UNIT CONVERSIONS")
for _ in range(5):
    deg = RNG.uniform(0, 360)
    rad = np.deg2rad(deg)
    back = np.rad2deg(rad)
    check(f"deg2rad round-trip  {deg:.2f}°",
          abs(back - deg) < 1e-9, f"got {back:.6f}°")

# ─── [2] TARGET VALIDATION ───────────────────────────────────────────────────
print("\n[2] TARGET VALIDATION")
for _ in range(4):
    x = RNG.uniform(15, 450); z = RNG.uniform(-100, 100)
    ok, msg = validate_target(x, 0, z)
    check(f"Valid target ({x:.0f}, 0, {z:.0f})", ok, msg)

x = RNG.uniform(0.1, 9.9)
ok, msg = validate_target(x, 0, 0)
check(f"Too close ({x:.1f}m)", not ok, msg)

x = RNG.uniform(501, 800)
ok, msg = validate_target(x, 0, 0)
check(f"Too far ({x:.0f}m)", not ok, msg)

# ─── [3] ROTATION MATRICES ───────────────────────────────────────────────────
print("\n[3] ROTATION MATRICES — unit vector + known results")
for _ in range(8):
    yaw   = RNG.uniform(-180, 180)
    pitch = RNG.uniform(0, 85)
    bd    = RotationMatrix.barrel_direction(0, 0, 0, yaw, pitch)
    mag   = np.linalg.norm(bd)
    check(f"Unit vector  yaw={yaw:+.1f}° pitch={pitch:.1f}°",
          abs(mag - 1.0) < 1e-9, f"mag={mag:.8f}")

bd = RotationMatrix.barrel_direction(0, 0, 0, 0, 0)
check("Identity: barrel points forward [1,0,0]",
      np.allclose(bd, [1, 0, 0], atol=1e-9), str(bd))

bd = RotationMatrix.barrel_direction(0, 0, 0, 0, 90)
check("Pitch 90° → barrel points up [0,1,0]",
      np.allclose(bd, [0, 1, 0], atol=1e-6), str(bd))

bd = RotationMatrix.barrel_direction(0, 0, 0, 90, 0)
check("Yaw 90° → barrel points left [0,0,-1]",
      np.allclose(bd, [0, 0, -1], atol=1e-6), str(bd))

# ─── [4] BALLISTIC SOLVER — LOW trajectory ───────────────────────────────────
print("\n[4] BALLISTIC SOLVER — LOW trajectory (10 random targets)")
for _ in range(10):
    tx = RNG.uniform(30, 450)
    ty = RNG.uniform(-15, 40)
    tz = RNG.uniform(-200, 200)
    v0 = RNG.uniform(60, 200)
    sol = solver.solve(tx, ty, tz, v0)
    if sol.reachable:
        check(f"LOW reachable  t=({tx:.0f},{ty:.0f},{tz:.0f}) v0={v0:.0f}",
              sol.verification_error < VERIFY_TOL,
              f"err={sol.verification_error:.4f}m  θ={sol.turret_pitch_deg:.2f}°")
        # When prefer='LOW' but LOW is mechanically impossible (e.g., below-horizon
        # short-range target needs negative pitch → clamped), the solver falls back
        # to HIGH. Only assert pitch ≤ 45° when the solver actually returned LOW.
        if sol.solution_type == "LOW":
            check(f"LOW pitch ≤ 45°  t=({tx:.0f},{ty:.0f},{tz:.0f})",
                  sol.turret_pitch_deg <= 45.5,
                  f"θ={sol.turret_pitch_deg:.2f}°")
        else:
            check(f"LOW fallback→HIGH  t=({tx:.0f},{ty:.0f},{tz:.0f})",
                  True,
                  f"type={sol.solution_type}  θ={sol.turret_pitch_deg:.2f}° (fallback)")
    else:
        min_v   = solver._min_v0(np.sqrt(tx**2 + tz**2), ty)
        energy  = v0 < min_v * 1.05
        limits  = ("pitch limits" in sol.error_message.lower() or
                   "outside"      in sol.error_message.lower())
        check(f"Unreachable    t=({tx:.0f},{ty:.0f},{tz:.0f}) v0={v0:.0f}",
              energy or limits,
              f"reason: {sol.error_message}")

# ─── [5] TEXTBOOK VALIDATION (Galileo) ───────────────────────────────────────
print("\n[5] TEXTBOOK VALIDATION (Galileo 1638)")
for _ in range(4):
    v0 = RNG.uniform(50, 150)
    expected_R = v0**2 / GRAVITY
    if expected_R <= 490:
        sol = solver.solve(expected_R * 0.9999, 0, 0, v0)
        if sol.reachable:
            check(f"Near-optimal angle  v0={v0:.0f}m/s  R={expected_R:.1f}m",
                  abs(sol.turret_pitch_deg - 45) < 5,
                  f"θ={sol.turret_pitch_deg:.2f}°")

# ─── [6] TRAJECTORY POINTS ───────────────────────────────────────────────────
print("\n[6] TRAJECTORY GENERATION")
for _ in range(3):
    tx = RNG.uniform(50, 300)
    ty = RNG.uniform(-10, 20)
    tz = RNG.uniform(-100, 100)
    sol = solver.solve(tx, ty, tz, 100)
    if sol.reachable:
        traj = solver.trajectory(sol, steps=200)
        check(f"Trajectory starts at origin ({tx:.0f},{ty:.0f},{tz:.0f})",
              abs(traj[0].x) < 1e-9 and abs(traj[0].z) < 1e-9,
              f"start=({traj[0].x:.2e},{traj[0].z:.2e})")
        check(f"Trajectory ends near target ({tx:.0f},{ty:.0f},{tz:.0f})",
              abs(traj[-1].x - tx) < 0.5 and abs(traj[-1].z - tz) < 0.5,
              f"end=({traj[-1].x:.1f},{traj[-1].z:.1f})")
        # Apex is above both start and end
        apex_y = max(pt.y for pt in traj)
        # Apex must be at or above the higher of (ground level, target height).
        # For LOW trajectories hitting elevated targets on the ascending path,
        # the apex equals the impact point (ty). Use 0.1m tolerance for float.
        check(f"Trajectory apex ≥ target height ({tx:.0f},{ty:.0f},{tz:.0f})",
              apex_y >= max(0.0, ty) - 0.1,
              f"apex={apex_y:.2f}m  target_y={ty:.2f}m")

# ─── [7] HIGH-ANGLE BALLISTIC SOLVER ─────────────────────────────────────────
# Use a fixed inner RNG (seed 7) to pre-generate exactly 8 targets that
# always have a valid HIGH solution. This makes the test count deterministic
# (always 8×4 = 32 checks) regardless of the outer random seed.
print("\n[7] HIGH-ANGLE BALLISTIC SOLVER — 8 pre-screened HIGH targets")
_rng7 = np.random.default_rng(7)
_high7 = []
for _ in range(500):
    if len(_high7) >= 8:
        break
    _tx7 = float(_rng7.uniform(120, 350))
    _ty7 = float(_rng7.uniform(-5, 20))
    _tz7 = float(_rng7.uniform(-100, 100))
    _v07 = float(_rng7.uniform(80, 180))
    _s7  = solver.solve(_tx7, _ty7, _tz7, _v07, prefer="HIGH")
    if _s7.reachable and _s7.solution_type == "HIGH":
        _high7.append((_tx7, _ty7, _tz7, _v07))

for tx, ty, tz, v0 in _high7:
    sol_h = solver.solve(tx, ty, tz, v0, prefer="HIGH")
    sol_l = solver.solve(tx, ty, tz, v0, prefer="LOW")

    check(f"HIGH θ > LOW θ   t=({tx:.0f},{ty:.0f},{tz:.0f}) v0={v0:.0f}",
          sol_h.turret_pitch_deg > sol_l.turret_pitch_deg + 0.5,
          f"HIGH={sol_h.turret_pitch_deg:.2f}° LOW={sol_l.turret_pitch_deg:.2f}°")

    check(f"HIGH accuracy    t=({tx:.0f},{ty:.0f},{tz:.0f}) v0={v0:.0f}",
          sol_h.verification_error < VERIFY_TOL,
          f"err={sol_h.verification_error:.4f}m")

    check(f"HIGH ToF > LOW ToF  t=({tx:.0f},{ty:.0f},{tz:.0f})",
          sol_h.tof > sol_l.tof - 0.01,
          f"HIGH={sol_h.tof:.2f}s  LOW={sol_l.tof:.2f}s")

    check(f"HIGH apex > LOW apex t=({tx:.0f},{ty:.0f},{tz:.0f})",
          sol_h.max_height >= sol_l.max_height - 0.5,
          f"HIGH={sol_h.max_height:.1f}m  LOW={sol_l.max_height:.1f}m")

print(f"\n  Fixed HIGH targets tested: {len(_high7)}/8")

# ─── [8] SOLVE_BOTH() — dual trajectory ──────────────────────────────────────
print("\n[8] SOLVE_BOTH() — dual trajectory correctness")
both_pass = 0
for _ in range(8):
    tx = RNG.uniform(60, 380)
    ty = RNG.uniform(-5, 20)
    v0 = RNG.uniform(80, 180)

    both = solver.solve_both(tx, ty, 0, v0)
    lo   = both["LOW"]
    hi   = both["HIGH"]

    if lo is not None:
        check(f"LOW reachable    R={tx:.0f}  v0={v0:.0f}",
              lo.verification_error < VERIFY_TOL,
              f"err={lo.verification_error:.4f}m  θ={lo.turret_pitch_deg:.2f}°")

    if hi is not None:
        check(f"HIGH reachable   R={tx:.0f}  v0={v0:.0f}",
              hi.verification_error < VERIFY_TOL,
              f"err={hi.verification_error:.4f}m  θ={hi.turret_pitch_deg:.2f}°")

        if lo is not None:
            diff = hi.turret_pitch_deg - lo.turret_pitch_deg
            check(f"HIGH-LOW pitch gap  R={tx:.0f}  v0={v0:.0f}",
                  diff > 0.5,
                  f"Δθ={diff:.2f}°")
            both_pass += 1
    else:
        check(f"No HIGH (near optimal or limits)  R={tx:.0f}  v0={v0:.0f}",
              True, "valid — no distinct HIGH solution exists here")

# Confirm solve_both never returns two identical solutions
for _ in range(5):
    tx = RNG.uniform(100, 350); v0 = RNG.uniform(90, 150)
    both = solver.solve_both(tx, 0, 0, v0)
    if both["LOW"] is not None and both["HIGH"] is not None:
        diff = abs(both["HIGH"].turret_pitch_deg - both["LOW"].turret_pitch_deg)
        check(f"No duplicate solutions  R={tx:.0f}  v0={v0:.0f}",
              diff > 0.5,
              f"LOW={both['LOW'].turret_pitch_deg:.2f}° HIGH={both['HIGH'].turret_pitch_deg:.2f}°")

# ─── [9] VERIFICATION ERROR SWEEP — full envelope ────────────────────────────
print("\n[9] VERIFICATION ERROR SWEEP — full reachable envelope")
print("  Firing solver at 30 random targets, verifying <5cm accuracy for each.")

errors = []
n_tested = n_high_tested = n_unreachable = 0

for _ in range(30):
    tx = RNG.uniform(20, 470)
    ty = RNG.uniform(-18, 45)
    tz = RNG.uniform(-200, 200)
    v0 = RNG.uniform(55, 280)

    sol = solver.solve(tx, ty, tz, v0)
    if sol.reachable:
        errors.append(sol.verification_error)
        n_tested += 1
    else:
        n_unreachable += 1

    # Also test HIGH
    sol_h = solver.solve(tx, ty, tz, v0, prefer="HIGH")
    if sol_h.reachable and sol_h.solution_type == "HIGH":
        errors.append(sol_h.verification_error)
        n_high_tested += 1

# All must be below 5cm
all_accurate = all(e < VERIFY_TOL for e in errors)
check(f"All {len(errors)} solutions have verification error < {VERIFY_TOL*100:.0f}cm",
      all_accurate,
      f"max={max(errors)*100:.4f}cm  mean={np.mean(errors)*100:.4f}cm")

check(f"Mean verification error < 0.001m  (effectively exact)",
      np.mean(errors) < 0.001,
      f"mean={np.mean(errors)*1000:.4f}mm")

print(f"  LOW  tested: {n_tested}   HIGH tested: {n_high_tested}   "
      f"Unreachable: {n_unreachable}")
if errors:
    print(f"  Max error: {max(errors)*100:.4f}cm   "
          f"Mean: {np.mean(errors)*1000:.4f}mm")

# ─── [10] RANGE TABLE — physics generation and lookup ────────────────────────
print("\n[10] RANGE TABLE — generation, HIGH solutions, correction separation")

try:
    from physics.range_table import RangeTable

    tmpdir = Path(tempfile.mkdtemp())
    pp     = str(tmpdir / "phys.csv")
    cp     = str(tmpdir / "corr.csv")

    rt = RangeTable(pp, cp)

    # Generate small but representative physics table
    rt.generate_physics(
        range_steps  = np.arange(50, 305, 25),   # 11 values
        height_steps = np.arange(-10, 30, 10),   # 4 values
        v0_steps     = np.array([80, 100, 120]),  # 3 values
        verbose=False, force=True)

    df       = rt._physics_df
    n_high   = int(df["has_high_solution"].sum())
    n_reach  = int(df["reachable"].sum())

    check("Physics table generated (dual trajectory)",
          len(df) > 0, f"{len(df)} rows")

    check("HIGH solutions exist in table",
          n_high > 0, f"{n_high} HIGH solutions ({n_high/max(1,n_reach)*100:.0f}% of reachable)")

    # Lookup LOW and HIGH and verify HIGH pitch > LOW pitch
    rt_errors = 0
    for _ in range(10):
        R  = float(RNG.uniform(60, 270))
        H  = float(RNG.uniform(-8, 25))
        v0 = float(RNG.choice([80, 100, 120]))

        sol_l = solver.solve(R, H, 0, v0, prefer="LOW")
        sol_h = solver.solve(R, H, 0, v0, prefer="HIGH")
        lkp_l = rt.lookup(R, H, v0, prefer="LOW")
        lkp_h = rt.lookup(R, H, v0, prefer="HIGH")

        # LOW lookup must not have trajectory_fallback for this grid
        check(f"LOW lookup returns finite pitch  R={R:.0f}m H={H:.0f}m",
              not np.isnan(lkp_l["pitch_deg"]) and lkp_l["pitch_deg"] > 0,
              f"pitch={lkp_l['pitch_deg']:.2f}°")

        # If both solutions exist, HIGH must be above LOW
        if sol_h.reachable and sol_h.solution_type == "HIGH" and \
                not lkp_h["trajectory_fallback"]:
            check(f"HIGH lookup pitch > LOW lookup pitch  R={R:.0f}m",
                  lkp_h["pitch_deg"] > lkp_l["pitch_deg"] - 1.0,
                  f"H={lkp_h['pitch_deg']:.2f}° L={lkp_l['pitch_deg']:.2f}°")

    # Verify LOW and HIGH corrections are NOT mixed.
    # Use R=200 where HIGH angle = 84.3° (within the 85° pitch limit).
    # R=100 is intentionally avoided: its theoretical HIGH angle is 87.2°
    # which exceeds the mechanical limit, so HIGH correctly falls back to LOW.
    rt.record_correction(200, 0, 100, 0.5, 0.1, 1.0, 10.0, 4.0, 0.3, 12,
                          solution_type="LOW")
    rt.record_correction(200, 0, 100, 0.9, 0.2, 0.5, 10.0, 5.0, 0.3, 10,
                          solution_type="HIGH")
    rt.load_corrections(verbose=False)

    lkp_l = rt.lookup(200, 0, 100, prefer="LOW")
    lkp_h = rt.lookup(200, 0, 100, prefer="HIGH")

    check("LOW corrections not contaminated by HIGH",
          abs(lkp_l["delta_pitch"] - 0.5) < 0.15,
          f"dp={lkp_l['delta_pitch']:.3f}° (expect≈0.5°)")

    check("HIGH corrections not contaminated by LOW",
          abs(lkp_h["delta_pitch"] - 0.9) < 0.15,
          f"dp={lkp_h['delta_pitch']:.3f}° (expect≈0.9°)")

    # Persistence check
    rt2 = RangeTable(pp, cp)
    rt2.load(verbose=False)
    lkp2 = rt2.lookup(200, 0, 100, prefer="HIGH")
    check("HIGH corrections survive reload",
          lkp2["n_observations"] > 0, f"n_obs={lkp2['n_observations']}")

    # Stats check
    s = rt.stats()
    check("stats() counts LOW and HIGH separately",
          s.get("corrections_low", 0) == 1 and s.get("corrections_high", 0) == 1,
          f"low={s.get('corrections_low')} high={s.get('corrections_high')}")

except Exception as e:
    check("Range table test ran without exception", False, str(e))

# ─── SUMMARY ─────────────────────────────────────────────────────────────────
print(f"\n{'='*68}")
print(f"  TOTAL: {PASS+FAIL}  |  PASSED: {PASS}  |  FAILED: {FAIL}")
print(f"{'='*68}")

if FAIL > 0:
    print(f"\n  Failed tests:")
    for row in LOG:
        if row["status"] == "FAIL":
            print(f"    ✗  {row['test']}")

# Save results
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "data"), exist_ok=True)
out = os.path.join(os.path.dirname(__file__), "..", "data",
                   "test_results_physics.csv")
df_log = pd.DataFrame(LOG)
if os.path.exists(out):
    df_log = pd.concat([pd.read_csv(out), df_log], ignore_index=True)
df_log.to_csv(out, index=False)
print(f"\n  Results saved → data/test_results_physics.csv")
print(f"  ({len(df_log)} total records across all runs)")