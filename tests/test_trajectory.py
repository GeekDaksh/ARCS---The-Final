"""
Validation suite for the point-mass trajectory integrator (Component 2).

External ground truth = vacuum projectile physics, which has exact closed-form
answers. We validate the RK4 integrator against those to tight tolerance, then
sanity-check the atmospheric (drag) case.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.trajectory import integrate_trajectory, G0


# --- Analytical vacuum formulas ---
def vac_range(v0, theta_deg):
    th = math.radians(theta_deg)
    return v0 ** 2 * math.sin(2 * th) / G0


def vac_tof(v0, theta_deg):
    th = math.radians(theta_deg)
    return 2 * v0 * math.sin(th) / G0


def vac_apex(v0, theta_deg):
    th = math.radians(theta_deg)
    return v0 ** 2 * math.sin(th) ** 2 / (2 * G0)


VAC_CASES = [
    (827.0, 30.0),
    (827.0, 45.0),
    (827.0, 60.0),
    (500.0, 45.0),
    (300.0, 35.0),
]


# ---------------------------------------------------------------------------
# Layer 1 — Ground-truth match (vacuum, analytical), tolerance 0.1%
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("v0,theta", VAC_CASES)
def test_layer1_vacuum_matches_analytical(v0, theta):
    res = integrate_trajectory(v0=v0, elevation_deg=theta, vacuum=True)
    R = vac_range(v0, theta)
    T = vac_tof(v0, theta)
    H = vac_apex(v0, theta)
    assert abs(res["range_m"] - R) <= 1e-3 * R, f"range {res['range_m']:.2f} vs {R:.2f}"
    assert abs(res["tof_s"] - T) <= 1e-3 * T, f"tof {res['tof_s']:.3f} vs {T:.3f}"
    assert abs(res["apex_m"] - H) <= 1e-3 * H, f"apex {res['apex_m']:.2f} vs {H:.2f}"


def test_layer1_reference_45deg():
    # Spec's stated reference: v0=827, theta=45 -> R~69757, apex~17439, TOF~119.3
    res = integrate_trajectory(v0=827.0, elevation_deg=45.0, vacuum=True)
    assert abs(res["range_m"] - 69757) <= 0.001 * 69757
    assert abs(res["apex_m"] - 17439) <= 0.001 * 17439
    assert abs(res["tof_s"] - 119.3) <= 0.001 * 119.3


# ---------------------------------------------------------------------------
# Layer 2 — Boundary behaviour
# ---------------------------------------------------------------------------
def test_layer2_straight_up():
    # theta=90 in vacuum: range ~ 0, apex = v0^2/(2g), comes straight back.
    v0 = 827.0
    res = integrate_trajectory(v0=v0, elevation_deg=90.0, vacuum=True)
    assert res["range_m"] <= 1.0  # essentially zero downrange
    assert abs(res["apex_m"] - v0 ** 2 / (2 * G0)) <= 0.001 * (v0 ** 2 / (2 * G0))


def test_layer2_zero_elevation():
    # theta=0: starts at ground heading level -> immediate impact.
    res = integrate_trajectory(v0=827.0, elevation_deg=0.0, vacuum=True)
    assert res["range_m"] <= 1e-6
    assert res["tof_s"] <= 1e-6


def test_layer2_symmetry_complementary_angles():
    # In vacuum, theta and (90 - theta) give the same range.
    a = integrate_trajectory(v0=827.0, elevation_deg=30.0, vacuum=True)
    b = integrate_trajectory(v0=827.0, elevation_deg=60.0, vacuum=True)
    assert abs(a["range_m"] - b["range_m"]) <= 1e-3 * a["range_m"]


# ---------------------------------------------------------------------------
# Layer 3 — Known-physics relationship (energy conservation in vacuum)
# ---------------------------------------------------------------------------
def test_layer3_energy_conservation_vacuum():
    # Total mechanical energy at launch vs at impact must match within 0.1%
    # (z=0 at both ends, so this reduces to speed conservation, which is the
    # strongest statement RK4 can make here). We also check mid-flight via the
    # apex: at apex all vertical KE has become PE.
    v0 = 827.0
    res = integrate_trajectory(v0=v0, elevation_deg=45.0, vacuum=True)
    # Launch speed == impact speed (energy conserved, returns to z=0).
    assert abs(res["impact_speed"] - v0) <= 1e-3 * v0
    # Energy at apex: 0.5*m*v_horiz^2 + m*g*H == 0.5*m*v0^2
    m = 43.2
    v_horiz = v0 * math.cos(math.radians(45.0))
    E_apex = 0.5 * m * v_horiz ** 2 + m * G0 * res["apex_m"]
    E_launch = 0.5 * m * v0 ** 2
    assert abs(E_apex - E_launch) <= 1e-3 * E_launch


def test_layer3_drag_removes_energy():
    # With drag ON, impact speed must be strictly LESS than launch speed.
    v0 = 827.0
    res = integrate_trajectory(v0=v0, elevation_deg=45.0, vacuum=False)
    assert res["impact_speed"] < v0


# ---------------------------------------------------------------------------
# Layer 4 — Monotonic / physical-sense (atmospheric case)
# ---------------------------------------------------------------------------
def test_layer4_drag_range_less_than_vacuum():
    drag = integrate_trajectory(v0=827.0, elevation_deg=45.0, vacuum=False)
    vac = integrate_trajectory(v0=827.0, elevation_deg=45.0, vacuum=True)
    assert drag["range_m"] > 0.0
    assert drag["range_m"] < vac["range_m"]
    # A real 155mm with this simple drag lands in the tens of km, not 70 km.
    assert 5000.0 < drag["range_m"] < 50000.0


def test_layer4_apex_within_atmosphere_ceiling():
    # For a normal firing angle (45 deg) the drag-case apex should stay within
    # the atmosphere model's valid range (<= 20000 m). Flag if it exceeds.
    res = integrate_trajectory(v0=827.0, elevation_deg=45.0, vacuum=False)
    assert res["apex_m"] <= 20000.0, (
        f"apex {res['apex_m']:.0f} m exceeds atmosphere ceiling — extend the model"
    )


def test_layer4_higher_velocity_longer_range():
    # Drag case: higher muzzle velocity -> longer range (monotonic).
    speeds = [400.0, 600.0, 827.0, 1000.0]
    ranges = [integrate_trajectory(v0=v, elevation_deg=45.0, vacuum=False)["range_m"]
              for v in speeds]
    assert all(ranges[i] < ranges[i + 1] for i in range(len(ranges) - 1)), ranges


if __name__ == "__main__":
    print("\n=== Vacuum: integrator vs analytical (v0=827) ===\n")
    header = (f"{'theta':>6} | {'R calc':>10} {'R formula':>10} | "
              f"{'apex calc':>10} {'apex form':>10} | {'tof calc':>9} {'tof form':>9}")
    print(header)
    print("-" * len(header))
    for theta in (30.0, 45.0, 60.0):
        r = integrate_trajectory(v0=827.0, elevation_deg=theta, vacuum=True)
        print(f"{theta:>6.0f} | {r['range_m']:>10.1f} {vac_range(827,theta):>10.1f} | "
              f"{r['apex_m']:>10.1f} {vac_apex(827,theta):>10.1f} | "
              f"{r['tof_s']:>9.3f} {vac_tof(827,theta):>9.3f}")

    print("\n=== Energy conservation (vacuum, 45 deg) ===\n")
    rv = integrate_trajectory(v0=827.0, elevation_deg=45.0, vacuum=True)
    print(f"  launch speed = 827.000 m/s   impact speed = {rv['impact_speed']:.3f} m/s")
    print(f"  relative error = {abs(rv['impact_speed']-827)/827*100:.4f} %")

    print("\n=== Drag vs vacuum range (v0=827, 45 deg) ===\n")
    rd = integrate_trajectory(v0=827.0, elevation_deg=45.0, vacuum=False)
    print(f"  vacuum range = {rv['range_m']:>10.1f} m   apex = {rv['apex_m']:.0f} m")
    print(f"  drag   range = {rd['range_m']:>10.1f} m   apex = {rd['apex_m']:.0f} m")
    print(f"  impact speed (drag) = {rd['impact_speed']:.1f} m/s   steps = {rd['steps']}")

    print("\nRun 'python -m pytest tests/test_trajectory.py -v' for full PASS/FAIL.")
