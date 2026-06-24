"""
Validation suite for the honest unreachable-target guard (Option A).

The integrator will compute a trajectory for any inputs, and the engagement loop
would otherwise keep firing at a target the projectile cannot physically reach —
falsely "destroying" it. This guard computes the weapon's maximum range from its
REAL parameters (v0/mass/drag, never hardcoded) and refuses an impossible target:
reachable=False, zero rounds, no claim of destruction. A reachable target is
completely unaffected.

External ground truth: the known 155mm M107 maximum range (~25.2 km at v0=827),
and the fact that a higher muzzle velocity reaches farther (so the bound is
clearly computed from parameters, not a constant).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.met_message import MetMessage
from physics.engagement import max_achievable_range, run_engagement_until_destroyed
from arcs_server import build_mission

STILL = MetMessage.standard_isa(surface_wind=(0.0, 0.0))   # still air
TRUE = MetMessage.standard_isa(surface_wind=(0.0, 0.0))
GUN = (200.0, 80.0)


def tc(true_met=TRUE):
    return {"true_met": true_met, "gun_bias": GUN}


def fire(target_range, told=STILL, **kw):
    return run_engagement_until_destroyed("W", target_range, 0.0, told, tc(),
                                          lethal_radius_m=8.0, max_rounds=15, **kw)


# ---------------------------------------------------------------------------
# max_achievable_range — the computed bound
# ---------------------------------------------------------------------------
def test_max_range_matches_known_m107_value():
    # Still air, v0=827: the published M107 maximum is ~25.2 km, reached in the
    # realistic 43-50 deg quadrant band (drag pulls the optimum off 45 deg).
    mr, elev = max_achievable_range({"v0": 827.0}, met=None, return_elev=True)
    assert 25000.0 <= mr <= 25500.0, mr
    assert 43.0 <= elev <= 50.0, elev


def test_max_range_scales_with_muzzle_velocity():
    # Computed from parameters, not hardcoded: a faster shot reaches farther.
    lo = max_achievable_range({"v0": 600.0})
    mid = max_achievable_range({"v0": 827.0})
    hi = max_achievable_range({"v0": 900.0})
    assert lo < mid < hi, (lo, mid, hi)
    assert lo < 25000.0 and hi > 25500.0           # genuinely different bounds


def test_max_range_tailwind_extends_it():
    # A tail wind genuinely extends the reachable distance (the guard is honest
    # about the conditions it is given), so the told-MET bound exceeds still air.
    still = max_achievable_range({"v0": 827.0}, met=None)
    tail = max_achievable_range({"v0": 827.0},
                                met=MetMessage.standard_isa(surface_wind=(180.0, 20.0)))
    assert tail > still


# ---------------------------------------------------------------------------
# Reachable target — existing behaviour preserved
# ---------------------------------------------------------------------------
def test_reachable_target_fires_and_can_destroy():
    r = fire(20000.0)
    assert r.get("reachable", True) is True          # not rejected
    assert r["rounds_fired"] > 0
    assert r["destroyed"] is True                     # within range -> normal kill
    assert "reason" not in r                           # no unreachable payload


# ---------------------------------------------------------------------------
# Unreachable target — honest rejection
# ---------------------------------------------------------------------------
def test_unreachable_target_is_rejected():
    r = fire(100000.0)
    assert r["reachable"] is False
    assert r["rounds_fired"] == 0
    assert r["destroyed"] is False
    assert r["destroying_round"] is None
    assert r["history"] == []
    assert r["target_range_m"] == 100000.0
    assert 24000.0 < r["max_range_m"] < 27000.0
    assert "exceeds weapon maximum range" in r["reason"]


def test_unreachable_never_claims_destruction_even_huge():
    for rng in (40000.0, 100000.0, 1_000_000.0):
        r = fire(rng)
        assert r["reachable"] is False and r["destroyed"] is False and r["rounds_fired"] == 0


# ---------------------------------------------------------------------------
# Boundary — just inside reachable, just outside not
# ---------------------------------------------------------------------------
def test_boundary_behaviour():
    mr = max_achievable_range({"v0": 827.0}, met=STILL)
    inside = fire(mr * 0.97)
    outside = fire(mr * 1.03)
    assert inside.get("reachable", True) is True and inside["rounds_fired"] > 0
    assert outside["reachable"] is False and outside["rounds_fired"] == 0


# ---------------------------------------------------------------------------
# Mission endpoint surfaces it cleanly for the UI
# ---------------------------------------------------------------------------
BASE = {
    "weapon_id": "HOW-1", "target_bearing": 0, "target_height_m": 0,
    "told_wind_dir": 180, "told_wind_speed": 20,
    "true_wind_dir": 180, "true_wind_speed": 23,
    "gun_bias_dr": 200, "gun_bias_cr": 80,
    "lethal_radius_m": 8.0, "max_rounds": 15, "warm_start": False,
}


def test_mission_endpoint_unreachable():
    m = build_mission({**BASE, "target_range": 100000})
    assert m["reachable"] is False
    assert m["rounds"] == []
    assert m["mission"]["destroyed"] is False
    assert m["mission"]["rounds_fired"] == 0
    assert m["learned"] is None
    assert m["target_range_m"] == 100000
    assert m["max_range_m"] > 24000


def test_mission_endpoint_reachable_unchanged():
    # A normal mission keeps EXACTLY the existing package shape (no new keys),
    # so valid missions are completely unaffected.
    m = build_mission({**BASE, "target_range": 22000})
    assert set(m) == {"mission", "rounds", "learned"}
    assert len(m["rounds"]) >= 1
    assert m["mission"]["rounds_fired"] == len(m["rounds"])


if __name__ == "__main__":
    print("\n=== Honest unreachable-target guard (Option A) ===\n")
    mr, elev = max_achievable_range({"v0": 827.0}, met=None, return_elev=True)
    print(f"max_achievable_range (still air, v0=827) = {mr:.0f} m at {elev:.1f} deg "
          f"(known M107 ~25.2 km)")
    print(f"  v0=600 -> {max_achievable_range({'v0':600.0}):.0f} m   "
          f"v0=900 -> {max_achievable_range({'v0':900.0}):.0f} m  (scales with v0)")
    print("\nReachable 20 km target:")
    r = fire(20000.0)
    print(f"  reachable, fired {r['rounds_fired']} rounds, destroyed={r['destroyed']}")
    print("\nUnreachable 100 km target:")
    r = fire(100000.0)
    print(f"  reachable={r['reachable']}, rounds={r['rounds_fired']}, destroyed={r['destroyed']}")
    print(f"  {r['reason']}")
    print("\nRun 'python -m pytest tests/test_reachability.py -v'.")
