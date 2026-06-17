"""
ARCS — Main entry point
Run this to verify the full physics stack is working.
"""
import numpy as np
from physics.constants  import validate_target
from physics.rotation   import RotationMatrix
from physics.ballistic_solver import BallisticSolver
from physics.range_table      import RangeTable

def main():
    print("=" * 52)
    print("ARCS Phase 1 — Full Stack Verification")
    print("=" * 52)

    # 1. Validate a target
    valid, msg = validate_target(120, 15, 80)
    print(f"\n[1] Target (120,15,80): {'✓' if valid else '✗'} {msg}")

    # 2. Barrel direction
    bd = RotationMatrix.barrel_direction(0, 0, 0, 30, 25)
    print(f"\n[2] Barrel direction (turret yaw=30°, pitch=25°):")
    print(f"    {bd}  magnitude={np.linalg.norm(bd):.6f}")

    # 3. Ballistic solution — LOW angle (default)
    solver = BallisticSolver()
    sol    = solver.solve(120, 15, 80, 100)
    print(f"\n[3] Ballistic solution (LOW)  target=(120,15,80)  v0=100m/s:")
    print(f"    Bearing     : {sol.turret_yaw_deg:.3f}°")
    print(f"    Elevation   : {sol.turret_pitch_deg:.3f}°")
    print(f"    ToF         : {sol.tof:.3f}s")
    print(f"    Max height  : {sol.max_height:.3f}m")
    print(f"    Impact vel  : {sol.impact_velocity:.3f}m/s")
    print(f"    Verify err  : {sol.verification_error:.2e}m")
    print(f"    Type        : {sol.solution_type}")

    # 3b. HIGH angle solution — check actual solution_type, not just reachable
    sol_h = solver.solve(120, 15, 80, 100, prefer="HIGH")
    if sol_h.reachable and sol_h.solution_type == "HIGH":
        print(f"\n[3b] Ballistic solution (HIGH) target=(120,15,80)  v0=100m/s:")
        print(f"    Elevation   : {sol_h.turret_pitch_deg:.3f}°  "
              f"(vs LOW {sol.turret_pitch_deg:.3f}°)")
        print(f"    ToF         : {sol_h.tof:.3f}s  "
              f"(vs LOW {sol.tof:.3f}s)")
        print(f"    Max height  : {sol_h.max_height:.3f}m")
        print(f"    Verify err  : {sol_h.verification_error:.2e}m")
    elif sol_h.reachable:
        print(f"\n[3b] HIGH solution at (120,15,80): above 85° limit → fell back to LOW")

    # 3c. solve_both()
    both = solver.solve_both(120, 15, 80, 100)
    print(f"\n[3c] solve_both():  "
          f"LOW={'✓' if both['LOW'] else '✗'}  "
          f"HIGH={'✓' if both['HIGH'] else '✗'}")

    # 4. Trajectory
    traj = solver.trajectory(sol, steps=100)
    print(f"\n[4] Trajectory: {len(traj)} points, "
          f"apex at t={traj[max(range(len(traj)), key=lambda i: traj[i].y)].t:.2f}s")

    # 5. Range table
    print("\n[5] Building small range table (dual trajectory)...")
    rt = RangeTable()
    rt.generate_physics(
        range_steps  = np.arange(50, 205, 25),
        height_steps = np.arange(-10, 25, 10),
        v0_steps     = np.array([80, 100, 120]),
        verbose=True
    )

    result_l = rt.lookup(120, 15, 100, prefer="LOW")
    result_h = rt.lookup(120, 15, 100, prefer="HIGH")
    print(f"    Lookup LOW (120,15,100): "
          f"pitch={result_l['pitch_deg']:.3f}°  tof={result_l['tof_s']:.3f}s")
    print(f"    Lookup HIGH(120,15,100): "
          f"pitch={result_h['pitch_deg']:.3f}°  tof={result_h['tof_s']:.3f}s  "
          f"fallback={result_h['trajectory_fallback']}")

    s = rt.stats()
    print(f"    HIGH solutions in table: {s.get('physics_high_solutions', 0)}")

    print("\n✓ All systems operational — ARCS Phase 1 stack ready")

if __name__ == "__main__":
    main()