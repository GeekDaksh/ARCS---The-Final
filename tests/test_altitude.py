"""
Validation suite for Component 8 — target altitude difference.

External ground truth: provable direction and sign of the gun-target height
effect, plus exact reproduction at height 0. The frame is GUN-RELATIVE (gun at
0): positive target_height_m = target above the gun (ridge), negative = below
(valley). Impact is detected at the target's altitude on the descending branch.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.trajectory import integrate_trajectory
from physics.atmosphere import MAX_ALT

REF = dict(v0=827.0, elevation_deg=45.0)        # realistic G7 reference shot
VAC = dict(v0=827.0, elevation_deg=45.0, vacuum=True)


def rng(**kw):
    return integrate_trajectory(**{**REF, **kw})


# ---------------------------------------------------------------------------
# Layer 1 — Ground-truth direction / sign (the core proof)
# ---------------------------------------------------------------------------
def test_layer1_height_ordering_below_level_above():
    r_below = rng(target_height_m=-300)["range_m"]
    r_level = rng(target_height_m=0)["range_m"]
    r_above = rng(target_height_m=+300)["range_m"]
    # target below the gun reaches further; above the gun, shorter.
    assert r_below > r_level > r_above, (r_below, r_level, r_above)


def test_layer1_impact_on_descending_branch():
    # A target above the gun must be met on the way DOWN, not during ascent.
    # The descending crossing happens late in flight, so the horizontal range is
    # almost the full level range — not the tiny range of an ascending crossing.
    r_level = rng(target_height_m=0)["range_m"]
    r_above = rng(target_height_m=+300)["range_m"]
    assert r_above > 0.9 * r_level     # near the full arc => descending branch
    assert r_above < r_level           # but shorter than level (met earlier in descent)


# ---------------------------------------------------------------------------
# Layer 2 — Boundary / reproduction
# ---------------------------------------------------------------------------
def test_layer2_height_zero_reproduces_existing_bitforbit():
    a = integrate_trajectory(**REF)                       # no target_height arg
    b = integrate_trajectory(**REF, target_height_m=0.0)  # explicit default
    for k in a:
        assert a[k] == b[k], k


def test_layer2_unreachable_target_above_apex():
    # Apex of the reference shot is ~8.6 km; a target at 15 km is within the
    # atmosphere's valid range but physically unreachable -> clear "not reached".
    res = rng(target_height_m=15000)
    assert res.get("reached") is False
    assert res["range_m"] is None
    assert res["apex_m"] < 15000
    assert "not reached" in res["reason"]


def test_layer2_outside_atmosphere_range_raises():
    with pytest.raises(ValueError):
        rng(target_height_m=MAX_ALT + 5000)      # above the 20 km ceiling
    with pytest.raises(ValueError):
        rng(target_height_m=-(MAX_ALT + 5000))   # absurdly far below
    # The boundary itself (== ceiling) is allowed (no artificial margin).
    rng(target_height_m=MAX_ALT)   # reachable check returns a result, no raise


# ---------------------------------------------------------------------------
# Layer 3 — Known-physics relationship
# ---------------------------------------------------------------------------
def test_layer3_range_monotonic_over_wide_span():
    # Wide span proves there is no artificial threshold: range decreases
    # monotonically as the target rises from a deep valley to a high ridge.
    heights = [-1000, -500, -200, 0, 200, 500, 1000, 2000, 3000]
    ranges = [rng(target_height_m=h)["range_m"] for h in heights]
    assert all(ranges[i] > ranges[i + 1] for i in range(len(ranges) - 1)), ranges


def test_layer3_lower_target_higher_impact_speed():
    # In vacuum (drag does not dominate), a lower target means the shell has
    # fallen further and is faster; a higher target, slower.
    s_below = integrate_trajectory(**VAC, target_height_m=-300)["impact_speed"]
    s_level = integrate_trajectory(**VAC, target_height_m=0)["impact_speed"]
    s_above = integrate_trajectory(**VAC, target_height_m=+300)["impact_speed"]
    assert s_below > s_level > s_above
    # matches energy conservation v = sqrt(v0^2 - 2 g h)
    g = 9.80665
    assert abs(s_below - math.sqrt(827**2 - 2 * g * (-300))) < 0.5


# ---------------------------------------------------------------------------
# Layer 4 — Adaptability / no artificial cap
# ---------------------------------------------------------------------------
def test_layer4_handles_small_and_large_heights():
    # Both tiny (+/-50 m) and large (+/-2000 m) differences are accepted and
    # behave correctly — proving the design is adaptable, not tuned to one case.
    for h in (-2000, -50, 50, 2000):
        r = rng(target_height_m=h)
        assert r.get("reached", True) and r["range_m"] > 0

    assert rng(target_height_m=-2000)["range_m"] > rng(target_height_m=-50)["range_m"]
    assert rng(target_height_m=+50)["range_m"]   > rng(target_height_m=+2000)["range_m"]
    # large valley reaches further than large ridge — the full unbounded effect
    assert rng(target_height_m=-2000)["range_m"] > rng(target_height_m=+2000)["range_m"]


if __name__ == "__main__":
    print("\n=== Component 8 — target altitude difference (v0=827, 45 deg, G7) ===\n")
    r0 = rng(target_height_m=0)
    for th in (-300, 0, 300):
        r = rng(target_height_m=th)
        tag = "valley (below gun)" if th < 0 else ("ridge (above gun)" if th > 0 else "level")
        print(f"  target_height {th:+5d} m  ->  range {r['range_m']:>9.1f} m   {tag}")
    rb, rl, ra = (rng(target_height_m=t)["range_m"] for t in (-300, 0, 300))
    print(f"\n  ordering: range(-300)={rb:.0f} > range(0)={rl:.0f} > range(+300)={ra:.0f}  "
          f"{'OK' if rb>rl>ra else 'FAIL'}")
    print(f"  descending branch: range(+300)/range(0) = {ra/rl:.3f} (near 1 => met on the way down)")

    unreach = rng(target_height_m=15000)
    print(f"\n  unreachable target at 15 km (apex {unreach['apex_m']:.0f} m): "
          f"reached={unreach.get('reached')}  -> {unreach['reason']}")
    try:
        rng(target_height_m=25000)
    except ValueError as e:
        print(f"  target at 25 km (> {MAX_ALT:.0f} m ceiling): ValueError raised -> {e}")
    print("\nRun 'python -m pytest tests/test_altitude.py -v'.")
