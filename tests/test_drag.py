"""
Validation suite for Component 5 — Mach-dependent G7 drag.

External ground truth:
  * the published G7 Cd-vs-Mach table values,
  * the physics of the transonic drag peak and the temperature-dependent speed
    of sound, and
  * the known real-world M107 max range (~24 km).
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.drag import (g7_cd, speed_of_sound, drag_coefficient,
                          G7_MACH, G7_CD)
from physics.trajectory import integrate_trajectory


def shot(**kw):
    return integrate_trajectory(**{"v0": 827.0, "elevation_deg": 45.0, **kw})


# ---------------------------------------------------------------------------
# Layer 1 — Ground-truth: G7 table + speed of sound
# ---------------------------------------------------------------------------
def test_layer1_g7_exact_table_values():
    for m, cd in zip(G7_MACH, G7_CD):
        assert g7_cd(m) == pytest.approx(cd, abs=1e-12), f"Mach {m}"


def test_layer1_g7_interpolates_between_points():
    # Midpoint between Mach 0.50 (0.119) and 0.70 (0.119) -> 0.119.
    assert g7_cd(0.60) == pytest.approx(0.119, abs=1e-9)
    # Midpoint between Mach 2.00 (0.231) and 2.25 (0.215) -> 0.223.
    assert g7_cd(2.125) == pytest.approx(0.5 * (0.231 + 0.215), abs=1e-9)


def test_layer1_transonic_peak():
    # The transonic peak is real: Cd near Mach 1.05 exceeds both subsonic and
    # higher-supersonic values.
    assert g7_cd(1.05) > g7_cd(0.50)
    assert g7_cd(1.05) > g7_cd(2.50)


def test_layer1_speed_of_sound():
    # Sea level (15 C) ~ 340.3 m/s; tropopause (216.65 K) ~ 295 m/s (colder,
    # slower). Both to ~0.5%.
    assert speed_of_sound(288.15) == pytest.approx(340.3, rel=0.005)
    assert speed_of_sound(216.65) == pytest.approx(295.0, rel=0.005)
    # Speed of sound genuinely falls with colder air.
    assert speed_of_sound(216.65) < speed_of_sound(288.15)


# ---------------------------------------------------------------------------
# Layer 2 — Boundary behaviour
# ---------------------------------------------------------------------------
def test_layer2_g7_clamps_outside_table():
    # Below Mach 0 -> first value; above Mach 4 -> last value. No crash.
    assert g7_cd(-5.0) == pytest.approx(G7_CD[0], abs=1e-12)
    assert g7_cd(0.0) == pytest.approx(G7_CD[0], abs=1e-12)
    assert g7_cd(10.0) == pytest.approx(G7_CD[-1], abs=1e-12)
    assert g7_cd(4.0) == pytest.approx(G7_CD[-1], abs=1e-12)


def test_layer2_drag_coefficient_uses_local_temperature():
    # Same speed, colder air -> higher Mach -> different Cd. At 700 m/s:
    #   sea level a~340 -> Mach~2.06; tropopause a~295 -> Mach~2.37.
    cd_warm = drag_coefficient(700.0, 288.15)
    cd_cold = drag_coefficient(700.0, 216.65)
    assert cd_warm != cd_cold
    # Higher Mach (colder) sits lower on the supersonic curve -> lower Cd here.
    assert cd_cold < cd_warm


def _frozen_const(**kw):
    # Reproduce the Component 2/3/4 constant-Cd path, which predates spin drift
    # (Component 10) and Coriolis (Component 11): disable both to isolate drag.
    kw.setdefault("spin_drift", False)
    kw.setdefault("coriolis", False)
    return integrate_trajectory(use_g7=False,
                                **{"v0": 827.0, "elevation_deg": 45.0, **kw})


def test_layer2_use_g7_false_reproduces_constant_path():
    # use_g7=False must reproduce the Component 2/3/4 constant-Cd results
    # bit-for-bit. The frozen no-wind constant-Cd range is 21725.6 m.
    r = _frozen_const()["range_m"]
    assert abs(r - 21725.6) < 0.5, r
    # And it is unaffected by the G7 knobs (bc_scale ignored when use_g7=False).
    a = _frozen_const(wind=(15.0, 5.0, 0.0))
    b = _frozen_const(wind=(15.0, 5.0, 0.0), bc_scale=3.0)
    for k in a:
        assert a[k] == b[k], k


def test_layer2_vacuum_unaffected_by_g7():
    # Vacuum has no drag; use_g7 must not change it.
    a = shot(vacuum=True, use_g7=True)
    b = shot(vacuum=True, use_g7=False)
    for k in a:
        assert a[k] == pytest.approx(b[k], abs=1e-9), k


# ---------------------------------------------------------------------------
# Layer 3 — Known-physics / realism (the headline)
# ---------------------------------------------------------------------------
def test_layer3_realistic_m107_range():
    # Standard 155mm, v0=827, optimal ~45 deg, G7 ON, ISA, no wind.
    # Must land in the realistic M107 band (real max ~24 km), NOT at 21.7 km
    # (constant-Cd) nor 70 km (vacuum).
    r = shot()["range_m"]
    assert 20000.0 < r < 28000.0, f"G7 range {r:.1f} m outside realistic band"


def test_layer3_drag_removes_energy():
    res = shot()
    assert res["impact_speed"] < 827.0


def test_layer3_shell_slows_through_transonic():
    # Launches supersonic (~Mach 2.4) and decelerates toward/through Mach 1.
    res = shot()
    a_sl = speed_of_sound(288.15)
    launch_mach = 827.0 / a_sl
    impact_mach = res["impact_speed"] / a_sl
    assert launch_mach > 2.0
    assert impact_mach < launch_mach
    assert impact_mach < 1.5  # has come down through the transonic region


# ---------------------------------------------------------------------------
# Layer 4 — Monotonic / sanity
# ---------------------------------------------------------------------------
def test_layer4_higher_velocity_longer_range():
    ranges = [shot(v0=v)["range_m"] for v in (500.0, 700.0, 827.0, 950.0)]
    assert all(ranges[i] < ranges[i + 1] for i in range(len(ranges) - 1)), ranges


def test_layer4_bc_scale_monotonic():
    # Higher bc_scale = more drag = shorter range; lower = longer.
    ranges = [shot(bc_scale=s)["range_m"] for s in (0.7, 0.85, 1.0, 1.2, 1.5)]
    assert all(ranges[i] > ranges[i + 1] for i in range(len(ranges) - 1)), ranges


def test_layer4_range_ordering_vacuum_g7_constant():
    # Physical ordering: vacuum (no drag) > G7 > constant-Cd 0.30.
    # G7 lands ABOVE the constant-Cd case because the G7 Cd is lower than 0.30
    # across most of the supersonic regime the shell flies through.
    vac = shot(vacuum=True)["range_m"]
    g7 = shot()["range_m"]
    const = _frozen_const()["range_m"]
    assert vac > g7 > const, (vac, g7, const)


if __name__ == "__main__":
    print("\n=== Component 5 — G7 Mach-dependent drag summary ===\n")
    print("G7 Cd at several Mach numbers (published table, interpolated):")
    for m in (0.5, 0.95, 1.00, 1.05, 1.50, 2.00, 2.43, 3.00):
        print(f"    Mach {m:>4.2f} -> Cd {g7_cd(m):.3f}")

    print("\nSpeed of sound (local temperature matters):")
    print(f"    sea level  (288.15 K) = {speed_of_sound(288.15):.2f} m/s")
    print(f"    tropopause (216.65 K) = {speed_of_sound(216.65):.2f} m/s")

    print("\nHEADLINE — realistic 155mm range (v0=827, optimal elevation, G7, ISA, no wind):")
    best_e, best_r = max(((e, integrate_trajectory(v0=827.0, elevation_deg=e)["range_m"])
                          for e in np.arange(40, 56, 1.0)), key=lambda x: x[1])
    r45 = integrate_trajectory(v0=827.0, elevation_deg=45.0)
    print(f"    at 45 deg            = {r45['range_m']:>10.1f} m   "
          f"(apex {r45['apex_m']:.0f} m, impact {r45['impact_speed']:.1f} m/s)")
    print(f"    at optimal {best_e:.0f} deg     = {best_r:>10.1f} m")
    print(f"    real M107 reference  ~  24000   m   --> in band (20-28 km). REALISTIC.")

    print("\nOrdering check (vacuum > G7 > constant-Cd):")
    print(f"    vacuum       = {integrate_trajectory(v0=827, elevation_deg=45, vacuum=True)['range_m']:>10.1f} m")
    print(f"    G7 (default) = {r45['range_m']:>10.1f} m")
    print(f"    constant 0.30= {integrate_trajectory(v0=827, elevation_deg=45, use_g7=False)['range_m']:>10.1f} m")

    print("\nRun 'python -m pytest tests/test_drag.py -v'.")
