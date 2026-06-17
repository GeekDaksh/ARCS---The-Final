"""
Validation suite for the ISA standard atmosphere model — all four layers.

Principle: test against PUBLISHED ISA values, not against our own assumptions.
The ground-truth rows below are external ISA-table values.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.atmosphere import atmosphere, R, MAX_ALT

# Published ISA ground-truth table: (altitude_m, T_K, P_Pa, rho_kgm3)
ISA_TABLE = [
    (0,     288.15, 101325.0, 1.2250),
    (2000,  275.15,  79495.0, 1.0066),
    (5000,  255.65,  54020.0, 0.7364),
    (8000,  236.15,  35600.0, 0.5252),
    (11000, 216.65,  22632.0, 0.3639),
    (15000, 216.65,  12045.0, 0.1937),
]


# ---------------------------------------------------------------------------
# Layer 1 — Ground-truth match (tight tolerance)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("alt,T_pub,P_pub,rho_pub", ISA_TABLE)
def test_layer1_ground_truth(alt, T_pub, P_pub, rho_pub):
    res = atmosphere(alt)
    # temperature within +/- 0.1 K
    assert abs(res["temperature_K"] - T_pub) <= 0.1, (
        f"T at {alt} m: {res['temperature_K']:.3f} vs {T_pub}"
    )
    # density within +/- 0.5%
    assert abs(res["density_kgm3"] - rho_pub) <= 0.005 * rho_pub, (
        f"rho at {alt} m: {res['density_kgm3']:.5f} vs {rho_pub}"
    )


# ---------------------------------------------------------------------------
# Layer 2 — Boundary behaviour
# ---------------------------------------------------------------------------
def test_layer2_sea_level_datum():
    res = atmosphere(0)
    assert abs(res["temperature_K"] - 288.15) <= 1e-9
    assert abs(res["pressure_Pa"] - 101325.0) <= 1e-6
    assert abs(res["density_kgm3"] - 1.225) <= 0.005 * 1.225


def test_layer2_tropopause_temperature_continuity():
    # Temperature must equal 216.65 K at 11000 m. The troposphere formula
    # (T0 - L*h) and the stratosphere definition (constant 216.65) must agree.
    res = atmosphere(11000)
    assert abs(res["temperature_K"] - 216.65) <= 1e-9
    # Troposphere formula evaluated directly at the boundary:
    from physics.atmosphere import T0, L
    assert abs((T0 - L * 11000) - 216.65) <= 1e-9


def test_layer2_tropopause_density_continuity():
    # Density must be continuous across the boundary to within 0.5%.
    eps = 1e-6
    below = atmosphere(11000 - eps)
    above = atmosphere(11000 + eps)
    assert abs(above["density_kgm3"] - below["density_kgm3"]) <= 0.005 * below["density_kgm3"]


def test_layer2_below_range_raises():
    with pytest.raises(ValueError):
        atmosphere(-1)


def test_layer2_above_range_raises():
    with pytest.raises(ValueError):
        atmosphere(20001)


def test_layer2_boundaries_inclusive():
    # 0 and 20000 are valid; just outside is not.
    atmosphere(0)
    atmosphere(20000)
    with pytest.raises(ValueError):
        atmosphere(-0.0001)
    with pytest.raises(ValueError):
        atmosphere(20000.0001)


# ---------------------------------------------------------------------------
# Layer 3 — Known-physics relationship (ideal gas law)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("alt", [0, 1000, 5000, 8000, 11000, 11000.0001, 15000, 20000])
def test_layer3_ideal_gas_law(alt):
    res = atmosphere(alt)
    rho_from_pt = res["pressure_Pa"] / (R * res["temperature_K"])
    assert abs(res["density_kgm3"] - rho_from_pt) <= 1e-12 * max(1.0, rho_from_pt)


# ---------------------------------------------------------------------------
# Layer 4 — Monotonic / conservation
# ---------------------------------------------------------------------------
def test_layer4_pressure_strictly_decreasing():
    alts = np.arange(0, 20000 + 1, 100)
    pressures = [atmosphere(float(h))["pressure_Pa"] for h in alts]
    diffs = np.diff(pressures)
    assert np.all(diffs < 0), "pressure must strictly decrease at every step"


def test_layer4_density_strictly_decreasing():
    alts = np.arange(0, 20000 + 1, 100)
    densities = [atmosphere(float(h))["density_kgm3"] for h in alts]
    diffs = np.diff(densities)
    assert np.all(diffs < 0), "density must strictly decrease at every step"


def test_layer4_temperature_never_rises():
    alts = np.arange(0, 20000 + 1, 100)
    temps = [atmosphere(float(h))["temperature_K"] for h in alts]
    diffs = np.diff(temps)
    # Falls through troposphere, constant in stratosphere; never rises.
    assert np.all(diffs <= 1e-9), "temperature must never rise below 20 km"
    # And it genuinely decreases somewhere (troposphere) — not a flat line.
    assert np.any(diffs < 0)


if __name__ == "__main__":
    # Pretty summary: ground-truth rows side-by-side + per-layer PASS/FAIL.
    print("\n=== ISA Standard Atmosphere — Ground-truth comparison ===\n")
    header = f"{'Alt(m)':>7} | {'T_K pub':>8} {'T_K calc':>9} | {'P_Pa pub':>9} {'P_Pa calc':>10} | {'rho pub':>8} {'rho calc':>9}"
    print(header)
    print("-" * len(header))
    for alt, T_pub, P_pub, rho_pub in ISA_TABLE:
        r = atmosphere(alt)
        print(
            f"{alt:>7} | {T_pub:>8.2f} {r['temperature_K']:>9.3f} | "
            f"{P_pub:>9.1f} {r['pressure_Pa']:>10.2f} | "
            f"{rho_pub:>8.4f} {r['density_kgm3']:>9.5f}"
        )

    print("\n=== Per-layer validation ===\n")
    layers = {
        "Layer 1 (ground-truth match)": [
            "test_layer1_ground_truth",
        ],
        "Layer 2 (boundary behaviour)": [
            "test_layer2_sea_level_datum",
            "test_layer2_tropopause_temperature_continuity",
            "test_layer2_tropopause_density_continuity",
            "test_layer2_below_range_raises",
            "test_layer2_above_range_raises",
            "test_layer2_boundaries_inclusive",
        ],
        "Layer 3 (ideal gas law)": ["test_layer3_ideal_gas_law"],
        "Layer 4 (monotonic/conservation)": [
            "test_layer4_pressure_strictly_decreasing",
            "test_layer4_density_strictly_decreasing",
            "test_layer4_temperature_never_rises",
        ],
    }

    import traceback

    g = globals()
    overall_ok = True
    for layer_name, fns in layers.items():
        ok = True
        err = None
        for fn in fns:
            f = g[fn]
            try:
                # Handle parametrized functions by invoking their cases.
                marks = getattr(f, "pytestmark", [])
                params = None
                for m in marks:
                    if m.name == "parametrize":
                        params = m.args[1]
                if params is not None:
                    for case in params:
                        args = case if isinstance(case, tuple) else (case,)
                        f(*args)
                else:
                    f()
            except Exception:
                ok = False
                err = traceback.format_exc()
                break
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {layer_name}")
        if err:
            print(err)
        overall_ok = overall_ok and ok

    print("\n" + ("ALL LAYERS PASS" if overall_ok else "SOME LAYERS FAILED"))
    sys.exit(0 if overall_ok else 1)
