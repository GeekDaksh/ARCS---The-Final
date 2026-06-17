"""
Validation suite for Component 3 — wind mechanism (constant vector).

The single new physics idea: drag opposes the shell's velocity relative to the
air, v_rel = v - wind. External ground truth here is direction and sign,
provable without a firing table:
    tail wind (+x)  -> longer range
    head wind (-x)  -> shorter range
    crosswind (+y)  -> deflection in +y
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.trajectory import integrate_trajectory

REF = dict(v0=827.0, elevation_deg=45.0)  # reference shot, drag ON


def shot(**kw):
    return integrate_trajectory(**{**REF, **kw})


# Baseline no-wind range (Component 2 result, ~21.7 km).
R0 = shot()["range_m"]


# ---------------------------------------------------------------------------
# Layer 1 — Ground-truth direction/sign (the core proof)
# ---------------------------------------------------------------------------
def test_layer1_tailwind_increases_range():
    assert shot(wind=(20.0, 0.0, 0.0))["range_m"] > R0


def test_layer1_headwind_decreases_range():
    assert shot(wind=(-20.0, 0.0, 0.0))["range_m"] < R0


def test_layer1_tail_head_straddle_baseline():
    r_tail = shot(wind=(20.0, 0.0, 0.0))["range_m"]
    r_head = shot(wind=(-20.0, 0.0, 0.0))["range_m"]
    assert r_tail > R0 > r_head


def test_layer1_crosswind_plus_y_deflects_plus_y():
    assert shot(wind=(0.0, 20.0, 0.0))["impact_y"] > 0.0


def test_layer1_crosswind_minus_y_deflects_minus_y():
    res = shot(wind=(0.0, -20.0, 0.0))
    assert res["impact_y"] < 0.0


def test_layer1_crosswind_symmetric():
    plus = shot(wind=(0.0, 20.0, 0.0))["impact_y"]
    minus = shot(wind=(0.0, -20.0, 0.0))["impact_y"]
    assert abs(plus + minus) <= 1e-6 * abs(plus)  # equal and opposite


def test_layer1_crosswind_deflection_grows_with_speed():
    d10 = shot(wind=(0.0, 10.0, 0.0))["impact_y"]
    d20 = shot(wind=(0.0, 20.0, 0.0))["impact_y"]
    assert d20 > d10 > 0.0


def test_layer1_crosswind_barely_changes_downrange():
    # Crosswind changes downrange far less than an equal head/tail wind does.
    cross = shot(wind=(0.0, 20.0, 0.0))
    head = shot(wind=(-20.0, 0.0, 0.0))
    dx_cross = abs(cross["impact_x"] - R0)
    dx_head = abs(head["impact_x"] - R0)
    assert dx_cross < dx_head


# ---------------------------------------------------------------------------
# Layer 2 — Boundary behaviour
# ---------------------------------------------------------------------------
def test_layer2_zero_wind_matches_component2():
    # wind=(0,0,0) must reproduce the no-wind path bit-for-bit.
    a = shot()
    b = shot(wind=(0.0, 0.0, 0.0))
    for k in a:
        assert a[k] == b[k], k


def test_layer2_wind_no_effect_in_vacuum():
    # Vacuum has no air, so wind cannot act.
    nowind = shot(vacuum=True)
    windy = shot(vacuum=True, wind=(20.0, 15.0, 5.0))
    for k in ("range_m", "impact_x", "impact_y", "tof_s", "apex_m", "impact_speed"):
        assert abs(nowind[k] - windy[k]) <= 1e-9 * max(1.0, abs(nowind[k])), k


def test_layer2_zero_magnitude_any_direction_is_nowind():
    assert shot(wind=(0.0, 0.0, 0.0))["range_m"] == R0


# ---------------------------------------------------------------------------
# Layer 3 — Known-physics relationships
# ---------------------------------------------------------------------------
def test_layer3_range_monotonic_in_tailwind():
    ranges = [shot(wind=(w, 0.0, 0.0))["range_m"] for w in range(0, 31, 5)]
    assert all(ranges[i] < ranges[i + 1] for i in range(len(ranges) - 1)), ranges


def test_layer3_range_monotonic_decreasing_in_headwind():
    ranges = [shot(wind=(-w, 0.0, 0.0))["range_m"] for w in range(0, 31, 5)]
    assert all(ranges[i] > ranges[i + 1] for i in range(len(ranges) - 1)), ranges


def test_layer3_deflection_monotonic_roughly_linear():
    speeds = [5.0, 10.0, 15.0, 20.0]
    defl = [shot(wind=(0.0, w, 0.0))["impact_y"] for w in speeds]
    # Monotonic increasing.
    assert all(defl[i] < defl[i + 1] for i in range(len(defl) - 1)), defl
    # Roughly linear: deflection-per-(m/s) ratios should be within ~25%.
    ratios = [defl[i] / speeds[i] for i in range(len(speeds))]
    assert max(ratios) / min(ratios) < 1.25, ratios


def test_layer3_horizontal_wind_dominates_vertical():
    horiz = abs(shot(wind=(20.0, 0.0, 0.0))["range_m"] - R0)
    vert = abs(shot(wind=(0.0, 0.0, 20.0))["range_m"] - R0)
    assert horiz > vert


# ---------------------------------------------------------------------------
# Layer 4 — Magnitude sanity
# ---------------------------------------------------------------------------
def test_layer4_tail_head_spread_in_sane_band():
    r_tail = shot(wind=(20.0, 0.0, 0.0))["range_m"]
    r_head = shot(wind=(-20.0, 0.0, 0.0))["range_m"]
    spread = r_tail - r_head
    assert 100.0 < spread < 8000.0, f"20 m/s tail-head spread {spread:.1f} m out of band"


def test_layer4_crosswind_deflection_in_sane_band():
    defl = shot(wind=(0.0, 20.0, 0.0))["impact_y"]
    # Hundreds of metres to low km on a ~20 km shot — not cm, not tens of km.
    assert 50.0 < defl < 8000.0, f"20 m/s crosswind deflection {defl:.1f} m out of band"


if __name__ == "__main__":
    print("\n=== Component 3 — wind effect summary (v0=827, 45 deg, drag ON) ===\n")
    print(f"  baseline (no wind)      range = {R0:>10.1f} m")
    rt = shot(wind=(20.0, 0.0, 0.0))
    rh = shot(wind=(-20.0, 0.0, 0.0))
    print(f"  +20 m/s tail wind (+x)  range = {rt['range_m']:>10.1f} m   "
          f"({rt['range_m']-R0:+.1f} m -> LONGER)")
    print(f"  -20 m/s head wind (-x)  range = {rh['range_m']:>10.1f} m   "
          f"({rh['range_m']-R0:+.1f} m -> SHORTER)")
    print(f"  tail-head spread        = {rt['range_m']-rh['range_m']:.1f} m")
    cp = shot(wind=(0.0, 20.0, 0.0))
    cm = shot(wind=(0.0, -20.0, 0.0))
    print(f"  +20 m/s crosswind (+y)  impact_y = {cp['impact_y']:>9.1f} m   "
          f"(deflect +y), downrange {cp['impact_x']:.1f} m")
    print(f"  -20 m/s crosswind (-y)  impact_y = {cm['impact_y']:>9.1f} m   "
          f"(deflect -y), downrange {cm['impact_x']:.1f} m")
    print("\nRun 'python -m pytest tests/test_wind.py tests/test_trajectory.py -v'.")
