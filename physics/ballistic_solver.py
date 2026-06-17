"""
ARCS — Ballistic Solver
Phase 1: Pure vacuum ballistics. No drag, no wind, gravity only.

Given:  target (x, y, z) and muzzle velocity v0
Solve:  turret_yaw, turret_pitch, time_of_flight

The projectile is treated as a point mass under constant gravity.
Equations of motion:
    x(t) = v0x · t
    y(t) = v0y · t  -  ½ · g · t²
    z(t) = v0z · t

Where:
    v0x = v0 · cos(pitch) · cos(yaw)
    v0y = v0 · sin(pitch)
    v0z = v0 · cos(pitch) · sin(yaw)

Two solutions always exist for any reachable target:
    LOW  ANGLE : flatter trajectory, shorter time of flight
    HIGH ANGLE : lobbed trajectory, longer time of flight
    OPTIMAL    : 45° pitch gives maximum range at given v0

TRAJECTORY SELECTION:
    Time of flight is always determined by the positive root of the
    quadratic whose x(t) matches the target range — correct for both
    LOW and HIGH angle solutions and for elevated/below-horizon targets.
    Do NOT use max() or min() on the two roots naively.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from .constants import (GRAVITY, validate_target,
                        PITCH_MIN_DEG, PITCH_MAX_DEG,
                        V0_MIN, V0_MAX)


@dataclass
class BallisticSolution:
    """
    Complete solution for one firing engagement.
    All angles in degrees for readability.
    """
    # ── Inputs ──────────────────────────────────────────
    target_x:       float = 0.0
    target_y:       float = 0.0
    target_z:       float = 0.0
    v0:             float = 0.0

    # ── Primary solution ────────────────────────────────
    turret_yaw_deg:   float = 0.0   # bearing  φ — left/right
    turret_pitch_deg: float = 0.0   # elevation θ — up/down
    tof:              float = 0.0   # time of flight (seconds)
    max_height:       float = 0.0   # apex of trajectory (meters)
    impact_velocity:  float = 0.0   # speed at impact (m/s)
    horiz_range:      float = 0.0   # ground distance (meters)

    # ── Velocity components at launch ───────────────────
    v0x: float = 0.0
    v0y: float = 0.0
    v0z: float = 0.0

    # ── Alternate solution ───────────────────────────────
    alt_pitch_deg:    Optional[float] = None
    alt_tof:          Optional[float] = None
    alt_max_height:   Optional[float] = None
    alt_impact_vel:   Optional[float] = None

    # ── Solution metadata ────────────────────────────────
    solution_type:    str   = "LOW"     # LOW | HIGH | OPTIMAL | UNREACHABLE
    reachable:        bool  = False     # False until solve() completes successfully
    error_message:    str   = ""

    # ── Verification ────────────────────────────────────
    verified_impact:  Optional[np.ndarray] = field(default=None, repr=False)
    verification_error: float = 0.0        # should be near 0


@dataclass
class TrajectoryPoint:
    """One point along the flight path."""
    t:   float   # time (s)
    x:   float   # position (m)
    y:   float   # position (m)
    z:   float   # position (m)
    vx:  float   # velocity (m/s)
    vy:  float   # velocity (m/s)
    vz:  float   # velocity (m/s)
    speed: float # total speed (m/s)


class BallisticSolver:
    """
    Core ballistic solver for ARCS Phase 1.
    All computation assumes:
        - Vacuum (no drag)
        - Constant gravity (9.81 m/s² downward)
        - Flat reference plane (robot at y=0)
        - Robot at world origin (0,0,0)

    Both LOW-angle and HIGH-angle solutions are fully computed,
    including accurate time-of-flight for the alternate solution.
    Use solve(prefer="HIGH") to obtain the lobbed trajectory.
    Use solve_both() to obtain a dict with both solutions.
    """

    def __init__(self, g: float = GRAVITY):
        self.g = g

    # ─── TOF SELECTION ───────────────────────────────────────────

    def _select_tof(self, v0y: float, v0x: float,
                    target_x: float, target_y: float) -> Optional[float]:
        """
        Pick the positive root of 0.5·g·t² - v0y·t + target_y = 0 whose
        x(t) = v0x·t is closest to target_x. Works for LOW and HIGH angle
        solutions and for elevated, level, or below-horizon targets.
        Returns None if no valid positive root exists.
        """
        disc = v0y**2 - 2 * self.g * target_y
        if disc < 0:
            disc = 0.0
        sqrt_d = np.sqrt(disc)
        t1 = (v0y + sqrt_d) / self.g
        t2 = (v0y - sqrt_d) / self.g
        positives = sorted([t for t in [t1, t2] if t > 1e-9])
        if not positives:
            return None
        return min(positives, key=lambda t: abs(v0x * t - target_x))

    # ─── MAIN SOLVE ──────────────────────────────────────────────

    def solve(self,
              target_x: float,
              target_y: float,
              target_z: float,
              v0:       float,
              prefer:   str = "LOW") -> BallisticSolution:
        """
        Compute the complete firing solution.

        Args:
            target_x : downrange distance (m)
            target_y : height of target above robot (m)
            target_z : lateral offset (m)
            v0       : muzzle velocity (m/s)
            prefer   : "LOW" for flat trajectory (default),
                       "HIGH" for lobbed trajectory

        Returns:
            BallisticSolution dataclass with all parameters.
            Both primary and alternate solutions are fully populated
            (pitch, TOF, max_height, impact_vel) when valid.
        """
        if prefer not in ("LOW", "HIGH"):
            raise ValueError(f"prefer must be 'LOW' or 'HIGH', got {prefer!r}")

        sol = BallisticSolution(
            target_x=target_x, target_y=target_y,
            target_z=target_z, v0=v0
        )

        # ── Step 1: validate inputs ───────────────────────────────
        valid, reason = validate_target(target_x, target_y, target_z)
        if not valid:
            sol.reachable      = False
            sol.solution_type  = "UNREACHABLE"
            sol.error_message  = reason
            return sol

        if not (V0_MIN <= v0 <= V0_MAX):
            sol.reachable     = False
            sol.solution_type = "UNREACHABLE"
            sol.error_message = f"v0={v0} outside [{V0_MIN},{V0_MAX}] m/s"
            return sol

        # ── Step 2: bearing (yaw) ─────────────────────────────────
        bearing_rad = np.arctan2(target_z, target_x)
        bearing_deg = np.rad2deg(bearing_rad)
        sol.turret_yaw_deg = bearing_deg

        # ── Step 3: horizontal range ──────────────────────────────
        R = np.sqrt(target_x**2 + target_z**2)
        sol.horiz_range = R

        # ── Step 4: elevation angle ───────────────────────────────
        # General ballistic equation for target at height Δy:
        #   tan θ = (v² ± √(v⁴ - g(g·R² + 2·Δy·v²))) / (g·R)
        v2   = v0 * v0
        disc = v2**2 - self.g * (self.g * R**2 + 2 * target_y * v2)

        if disc < 0:
            sol.reachable     = False
            sol.solution_type = "UNREACHABLE"
            sol.error_message = (f"Target unreachable: need v0 ≥ "
                                 f"{self._min_v0(R, target_y):.1f} m/s")
            return sol

        theta_low_rad  = np.arctan2(v2 - np.sqrt(disc), self.g * R)
        theta_high_rad = np.arctan2(v2 + np.sqrt(disc), self.g * R)
        theta_low_deg  = np.rad2deg(theta_low_rad)
        theta_high_deg = np.rad2deg(theta_high_rad)

        # Choose preferred solution
        if prefer == "HIGH":
            primary_rad = theta_high_rad
            primary_deg = theta_high_deg
            alt_rad     = theta_low_rad
            alt_deg     = theta_low_deg
            sol.solution_type = "HIGH"
        else:
            primary_rad = theta_low_rad
            primary_deg = theta_low_deg
            alt_rad     = theta_high_rad
            alt_deg     = theta_high_deg
            sol.solution_type = "LOW"

        # Check mechanical limits
        if not (PITCH_MIN_DEG <= primary_deg <= PITCH_MAX_DEG):
            if PITCH_MIN_DEG <= alt_deg <= PITCH_MAX_DEG:
                primary_rad, primary_deg = alt_rad, alt_deg
                alt_rad  = theta_low_rad  if prefer == "LOW" else theta_high_rad
                alt_deg  = theta_low_deg  if prefer == "LOW" else theta_high_deg
                sol.solution_type = "HIGH" if prefer == "LOW" else "LOW"
            else:
                sol.reachable     = False
                sol.solution_type = "UNREACHABLE"
                sol.error_message = (f"Both solutions outside pitch limits: "
                                     f"{theta_low_deg:.1f}° / {theta_high_deg:.1f}°")
                return sol

        if abs(primary_deg - 45.0) < 0.5:
            sol.solution_type = "OPTIMAL"

        sol.turret_pitch_deg = primary_deg

        # ── Step 5: velocity components ───────────────────────────
        cos_pitch = np.cos(primary_rad)
        sin_pitch = np.sin(primary_rad)
        cos_bear  = np.cos(bearing_rad)
        sin_bear  = np.sin(bearing_rad)

        sol.v0x = v0 * cos_pitch * cos_bear
        sol.v0y = v0 * sin_pitch
        sol.v0z = v0 * cos_pitch * sin_bear

        # ── Step 6: time of flight (primary) ─────────────────────
        sol.tof = self._select_tof(sol.v0y, sol.v0x, target_x, target_y)
        if sol.tof is None:
            sol.reachable     = False
            sol.solution_type = "UNREACHABLE"
            sol.error_message = "No positive time of flight"
            return sol

        # ── Step 7: max height ────────────────────────────────────
        t_apex = sol.v0y / self.g
        if 0 <= t_apex <= sol.tof:
            sol.max_height = (sol.v0y * t_apex
                              - 0.5 * self.g * t_apex**2)
        else:
            sol.max_height = target_y   # monotone trajectory

        # ── Step 8: impact velocity ───────────────────────────────
        vx_impact = sol.v0x
        vy_impact = sol.v0y - self.g * sol.tof
        vz_impact = sol.v0z
        sol.impact_velocity = np.sqrt(
            vx_impact**2 + vy_impact**2 + vz_impact**2)

        # ── Step 9: alternate solution metadata ───────────────────
        if PITCH_MIN_DEG <= alt_deg <= PITCH_MAX_DEG:
            cos_a = np.cos(alt_rad)
            sin_a = np.sin(alt_rad)
            v0x_a = v0 * cos_a * cos_bear
            v0y_a = v0 * sin_a
            v0z_a = v0 * cos_a * sin_bear

            tof_a = self._select_tof(v0y_a, v0x_a, target_x, target_y)
            if tof_a is not None:
                t_ap = v0y_a / self.g
                mh_a = (v0y_a * t_ap - 0.5 * self.g * t_ap**2
                        if 0 <= t_ap <= tof_a else target_y)
                vy_a_impact = v0y_a - self.g * tof_a
                iv_a = np.sqrt(v0x_a**2 + vy_a_impact**2 + v0z_a**2)

                sol.alt_pitch_deg  = alt_deg
                sol.alt_tof        = tof_a
                sol.alt_max_height = mh_a
                sol.alt_impact_vel = iv_a

        # ── Step 10: verification ─────────────────────────────────
        t = sol.tof
        xi = sol.v0x * t
        yi = sol.v0y * t - 0.5 * self.g * t**2
        zi = sol.v0z * t
        sol.verified_impact = np.array([xi, yi, zi])
        sol.verification_error = np.sqrt(
            (xi - target_x)**2 +
            (yi - target_y)**2 +
            (zi - target_z)**2
        )

        sol.reachable = True
        return sol

    # ─── SOLVE BOTH TRAJECTORIES ─────────────────────────────────

    def solve_both(self,
                   target_x: float,
                   target_y: float,
                   target_z: float,
                   v0:       float) -> dict:
        """
        Compute and return BOTH the LOW-angle and HIGH-angle solutions.

        Returns a dict:
            {
                "LOW":  BallisticSolution or None,
                "HIGH": BallisticSolution or None,
            }

        Either value is None if that trajectory is unreachable or
        outside mechanical pitch limits [PITCH_MIN_DEG, PITCH_MAX_DEG].

        Use this when the range table needs to populate BOTH trajectory
        columns per grid cell.
        """
        low  = self.solve(target_x, target_y, target_z, v0, prefer="LOW")
        high = self.solve(target_x, target_y, target_z, v0, prefer="HIGH")

        low_pitch  = low.turret_pitch_deg  if (low  and low.reachable)  else 0.0
        high_pitch = high.turret_pitch_deg if (high and high.reachable) else 0.0

        return {
            "LOW":  low  if (low  and low.reachable  and low.solution_type  != "UNREACHABLE") else None,
            # At the optimal 45° range both solutions converge — treat as single solution.
            "HIGH": high if (high and high.reachable and high.solution_type != "UNREACHABLE"
                             and abs(high_pitch - low_pitch) > 0.5) else None,
        }

    # ─── TRAJECTORY GENERATION ───────────────────────────────────

    def trajectory(self,
                   sol: BallisticSolution,
                   steps: int = 500) -> list[TrajectoryPoint]:
        """
        Generate full flight path as a list of TrajectoryPoints.
        Works for both LOW and HIGH angle solutions.

        Args:
            sol   : a solved BallisticSolution (any trajectory type)
            steps : number of points along the path

        Returns:
            List of TrajectoryPoint from t=0 to t=tof
        """
        if not sol.reachable:
            return []

        t      = np.linspace(0.0, sol.tof, steps + 1)
        x      = sol.v0x * t
        y      = sol.v0y * t - 0.5 * self.g * t**2
        z      = sol.v0z * t
        vy     = sol.v0y - self.g * t
        speeds = np.sqrt(sol.v0x**2 + vy**2 + sol.v0z**2)
        return [
            TrajectoryPoint(t=t[i], x=x[i], y=y[i], z=z[i],
                            vx=sol.v0x, vy=vy[i], vz=sol.v0z,
                            speed=speeds[i])
            for i in range(steps + 1)
        ]

    # ─── MINIMUM V0 ───────────────────────────────────────────────

    def _min_v0(self, R: float, delta_y: float) -> float:
        """
        Minimum muzzle velocity needed to reach a target
        at horizontal range R and height delta_y.
        """
        return np.sqrt(self.g * (delta_y + np.sqrt(R**2 + delta_y**2)))

    def max_range(self, v0: float) -> float:
        """Maximum horizontal range achievable with given v0 (flat ground)."""
        return v0**2 / self.g


# ─── SELF TEST ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("ARCS — ballistic_solver.py smoke test")
    solver = BallisticSolver()
    tol = 0.01

    sol = solver.solve(100, 0, 0, 50)
    assert sol.reachable and sol.verification_error < tol
    print(f"  LOW  (100,0,0)   : θ={sol.turret_pitch_deg:.2f}°  ToF={sol.tof:.2f}s  err={sol.verification_error:.2e}m ✓")

    sol = solver.solve(200, 0, 0, 80, prefer="HIGH")
    assert sol.reachable and sol.turret_pitch_deg > 45 and sol.verification_error < tol
    print(f"  HIGH (200,0,0)   : θ={sol.turret_pitch_deg:.2f}°  ToF={sol.tof:.2f}s  err={sol.verification_error:.2e}m ✓")

    sol = solver.solve(400, 0, 0, 30)
    assert not sol.reachable
    print(f"  UNREACHABLE      : {sol.error_message} ✓")

    both = solver.solve_both(200, 0, 0, 80)
    assert both["LOW"] and both["HIGH"]
    assert both["LOW"].turret_pitch_deg < 45 < both["HIGH"].turret_pitch_deg
    print(f"  solve_both()     : LOW={both['LOW'].turret_pitch_deg:.1f}°  HIGH={both['HIGH'].turret_pitch_deg:.1f}° ✓")

    sol  = solver.solve(120, 15, 80, 100)
    traj = solver.trajectory(sol, steps=200)
    assert len(traj) == 201 and abs(traj[-1].x - 120) < 0.5
    print(f"  trajectory(201)  : final=({traj[-1].x:.1f},{traj[-1].y:.1f},{traj[-1].z:.1f}) ✓")

    print("  ballistic_solver.py ✓ all checks passed")