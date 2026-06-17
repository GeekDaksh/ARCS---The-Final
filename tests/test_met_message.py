"""
Validation suite for Component 4 — standard artillery MET message.

External ground truth:
  * a standard-ISA MET message must reproduce the already-frozen Components
    (1/2/3) within the zone-discretisation tolerance, and
  * known wind profiles must produce the correct directional effects.

The MET feeds Component 3's relative-airspeed drag mechanism (frozen) and the
Component 1 ISA density relation (frozen) with altitude-varying values.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.met_message import (MetMessage, wind_vector_from_dir_speed,
                                 STANDARD_ZONE_TOPS)
from physics.trajectory import integrate_trajectory

REF = dict(v0=827.0, elevation_deg=45.0)


def shot(**kw):
    return integrate_trajectory(**{**REF, **kw})


def make_uniform_met(wind_dir_deg=0.0, wind_speed_ms=0.0, max_alt_m=16000):
    return MetMessage.standard_isa(max_alt_m=max_alt_m,
                                   surface_wind=(wind_dir_deg, wind_speed_ms))


# ---------------------------------------------------------------------------
# Layer 1 — Ground-truth: reproduce the frozen components
# ---------------------------------------------------------------------------
def test_layer1_standard_isa_reproduces_component2_nowind():
    met = make_uniform_met()  # zero wind
    r_met = shot(met=met)["range_m"]
    r_c2 = shot()["range_m"]  # Component 2 no-wind atmospheric trajectory
    assert abs(r_met - r_c2) <= 0.005 * r_c2, f"met {r_met:.1f} vs C2 {r_c2:.1f}"


def test_layer1_uniform_wind_met_matches_component3():
    # Tail wind from 180 deg at 20 m/s == Component 3 wind vector (20,0,0).
    met = make_uniform_met(wind_dir_deg=180.0, wind_speed_ms=20.0)
    r_met = shot(met=met)["range_m"]
    r_c3 = shot(wind=(20.0, 0.0, 0.0))["range_m"]
    assert abs(r_met - r_c3) <= 0.005 * r_c3, f"met {r_met:.1f} vs C3 {r_c3:.1f}"


def test_layer1_sample_returns_correct_zone():
    met = make_uniform_met()
    # 250 m -> zone with top 500; 0 -> surface; 7000 -> zone with top 8000.
    # Compare each sampled density to the zone's own stored density.
    def zone_top_for(alt):
        for t in STANDARD_ZONE_TOPS:
            if alt <= t:
                return t
        return STANDARD_ZONE_TOPS[-1]
    for alt, expected_top in [(250, 500), (0, 0), (7000, 8000), (500, 500)]:
        assert zone_top_for(alt) == expected_top
        # The sampled density must equal the zone line whose top is expected_top.
        line = next(l for l in met.lines if l["zone_top_m"] == expected_top)
        assert met.sample(alt)["density_kgm3"] == line["density_kgm3"]


# ---------------------------------------------------------------------------
# Layer 2 — Boundary behaviour
# ---------------------------------------------------------------------------
def test_layer2_boundary_belongs_to_that_zone():
    # Rule: altitude equal to a zone top belongs to THAT zone (the lower one).
    met = make_uniform_met()
    line_500 = next(l for l in met.lines if l["zone_top_m"] == 500)
    assert met.sample(500.0)["density_kgm3"] == line_500["density_kgm3"]
    # Just above 500 belongs to the next zone (top 1000), lower density.
    assert met.sample(500.001)["density_kgm3"] < met.sample(500.0)["density_kgm3"]


def test_layer2_surface_and_above_top():
    met = make_uniform_met(max_alt_m=2000)  # deliberately low ceiling
    surface_line = met.lines[0]
    assert met.sample(0.0)["density_kgm3"] == surface_line["density_kgm3"]
    # Above the top zone clamps to the top zone and flags it.
    assert met.covers(2000.0) and not met.covers(2000.1)
    before = met.clamp_count
    top = met.lines[-1]
    assert met.sample(9999.0)["density_kgm3"] == top["density_kgm3"]
    assert met.clamp_count == before + 1


def test_layer2_wind_from_direction_convention():
    s = 20.0
    # from 180 -> +x tail
    assert np.allclose(wind_vector_from_dir_speed(180, s), [s, 0, 0], atol=1e-9)
    # from 0 -> -x head
    assert np.allclose(wind_vector_from_dir_speed(0, s), [-s, 0, 0], atol=1e-9)
    # from 270 -> -y crosswind (documented direction)
    assert np.allclose(wind_vector_from_dir_speed(270, s), [0, -s, 0], atol=1e-9)
    # from 90 -> +y crosswind
    assert np.allclose(wind_vector_from_dir_speed(90, s), [0, s, 0], atol=1e-9)


def test_layer2_met_wind_drives_integrator_direction():
    # from 180 (tail) lengthens; from 0 (head) shortens; from 270 (-y) deflects -y.
    base = shot(met=make_uniform_met())["range_m"]
    tail = shot(met=make_uniform_met(180, 20))["range_m"]
    head = shot(met=make_uniform_met(0, 20))["range_m"]
    cross = shot(met=make_uniform_met(270, 20))
    assert tail > base > head
    assert cross["impact_y"] < 0.0


# ---------------------------------------------------------------------------
# Layer 3 — Known-physics / directional correctness
# ---------------------------------------------------------------------------
def test_layer3_uniform_tail_head_cross_directions():
    base = shot(met=make_uniform_met())["range_m"]
    assert shot(met=make_uniform_met(180, 25))["range_m"] > base   # tail longer
    assert shot(met=make_uniform_met(0, 25))["range_m"] < base     # head shorter
    assert shot(met=make_uniform_met(90, 25))["impact_y"] > 0.0    # +y deflect


def _layered_met(surface_speed, upper_speed, upper_dir=180.0, surface_dir=180.0):
    """ISA-temp/pressure zones, but wind that varies by altitude: surface vs
    upper zones get different wind. Upper zones = tops > 2000 m."""
    base = MetMessage.standard_isa()
    lines = []
    for ln in base.lines:
        upper = ln["zone_top_m"] > 2000
        lines.append({
            "zone_top_m": ln["zone_top_m"],
            "wind_dir_deg": upper_dir if upper else surface_dir,
            "wind_speed_ms": upper_speed if upper else surface_speed,
            "temp_C": ln["temp_C"],
            "pressure_Pa": ln["pressure_Pa"],
        })
    return MetMessage(lines)


def test_layer3_altitude_varying_wind_is_distinct_and_signed():
    # Calm at the gun, strong tail wind aloft. The shell spends most of its
    # flight aloft, so it should GAIN range vs no-wind, but differ from a
    # uniform tail wind of the same upper speed.
    nowind = shot(met=make_uniform_met())["range_m"]
    layered = shot(met=_layered_met(surface_speed=0.0, upper_speed=30.0))["range_m"]
    uniform = shot(met=make_uniform_met(180, 30))["range_m"]
    assert layered > nowind          # high-altitude tail wind gains range
    assert abs(layered - uniform) > 1.0  # layered != uniform (varies w/ height)
    assert layered < uniform         # less wind low down -> less gain than uniform


def test_layer3_headline_layered_scenario_responds_to_layers():
    # "Calm at the gun, head wind low, tail wind high" — a genuinely layered
    # column. Result must differ from both the no-wind and any single-average.
    base = MetMessage.standard_isa()
    lines = []
    for ln in base.lines:
        top = ln["zone_top_m"]
        if top <= 1000:
            d, s = 0.0, 10.0      # head wind low (from 0 = -x)
        elif top <= 4000:
            d, s = 180.0, 0.0     # calm mid
        else:
            d, s = 180.0, 25.0    # tail wind high (from 180 = +x)
        lines.append({"zone_top_m": top, "wind_dir_deg": d, "wind_speed_ms": s,
                      "temp_C": ln["temp_C"], "pressure_Pa": ln["pressure_Pa"]})
    layered = shot(met=MetMessage(lines))["range_m"]
    nowind = shot(met=make_uniform_met())["range_m"]
    assert layered != nowind
    # Net effect dominated by the strong high tail wind -> longer than no-wind.
    assert layered > nowind


# ---------------------------------------------------------------------------
# Layer 4 — Monotonic / sanity
# ---------------------------------------------------------------------------
def test_layer4_upper_tailwind_monotonic_range():
    nowind = shot(met=make_uniform_met())["range_m"]
    ranges = [shot(met=_layered_met(0.0, s))["range_m"] for s in (0, 10, 20, 30, 40)]
    assert ranges[0] == pytest.approx(nowind, rel=1e-9)
    assert all(ranges[i] < ranges[i + 1] for i in range(len(ranges) - 1)), ranges


def test_layer4_uniform_met_equals_constant_wind():
    # Layered model with identical zones == single constant wind (consistency).
    met = make_uniform_met(180, 20)
    r_met = shot(met=met)["range_m"]
    r_const = shot(wind=(20.0, 0.0, 0.0))["range_m"]
    assert abs(r_met - r_const) <= 0.005 * r_const


def test_layer4_zone_density_strictly_decreasing():
    met = make_uniform_met()
    densities = [l["density_kgm3"] for l in met.lines]
    assert all(densities[i] > densities[i + 1] for i in range(len(densities) - 1)), densities


def test_layer4_met_none_reproduces_component3_exactly():
    # met=None must reproduce Component 2/3 bit-for-bit.
    a = shot(wind=(15.0, 5.0, 0.0))
    b = shot(wind=(15.0, 5.0, 0.0), met=None)
    for k in a:
        assert a[k] == b[k], k


if __name__ == "__main__":
    print("\n=== Component 4 — MET message summary (v0=827, 45 deg, drag ON) ===\n")

    r_c2 = shot()["range_m"]
    r_isa = shot(met=make_uniform_met())["range_m"]
    print("(a) standard-ISA MET reproduces no-wind range:")
    print(f"    Component 2 (continuous ISA) = {r_c2:>10.1f} m")
    print(f"    standard-ISA MET (zoned)     = {r_isa:>10.1f} m   "
          f"({(r_isa-r_c2)/r_c2*100:+.3f} %)")

    r_c3 = shot(wind=(20.0, 0.0, 0.0))["range_m"]
    r_umet = shot(met=make_uniform_met(180, 20))["range_m"]
    print("\n(b) uniform-wind MET matches Component 3 constant wind (20 m/s tail):")
    print(f"    Component 3 wind=(20,0,0)    = {r_c3:>10.1f} m")
    print(f"    MET uniform from 180 @20 m/s = {r_umet:>10.1f} m   "
          f"({(r_umet-r_c3)/r_c3*100:+.3f} %)")

    r_nowind = shot(met=make_uniform_met())["range_m"]
    r_layer = shot(met=_layered_met(0.0, 30.0))["range_m"]
    r_uni = shot(met=make_uniform_met(180, 30))["range_m"]
    print("\n(c) altitude-varying wind (calm at gun, 30 m/s tail aloft):")
    print(f"    no wind                      = {r_nowind:>10.1f} m")
    print(f"    layered (calm low/tail high) = {r_layer:>10.1f} m   "
          f"({r_layer-r_nowind:+.1f} m -> gains range aloft)")
    print(f"    uniform 30 m/s tail all zones= {r_uni:>10.1f} m   "
          f"(layered < uniform, as expected)")

    print("\n(d) layered-reduces-to-constant consistency:")
    print(f"    MET uniform from 180 @20 m/s = {r_umet:>10.1f} m")
    print(f"    constant wind (20,0,0)       = {r_c3:>10.1f} m   "
          f"({(r_umet-r_c3)/r_c3*100:+.3f} %)")

    print("\nRun 'python -m pytest tests/test_met_message.py -v'.")
