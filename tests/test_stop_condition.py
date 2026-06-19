"""
Validation suite for Component 9 — fire until destroyed, then stop.

The mission fires, learns from each miss, and ends the instant a round lands
within an ADAPTABLE lethal radius (the "destroyed" criterion is a parameter),
with a safety cap so it can never fire forever. run_engagement is untouched.
"""

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.met_message import MetMessage
from physics.engagement import run_engagement_until_destroyed
from engagement_database import EngagementDatabase

TOLD = MetMessage.standard_isa(surface_wind=(180.0, 20.0))   # imperfect report
TRUE = MetMessage.standard_isa(surface_wind=(180.0, 23.0))   # real weather
GUN = (200.0, 80.0)


def fire(lethal_radius_m=8.0, db=None, weapon_id="HOW-K", **kw):
    tc = {"true_met": TRUE, "gun_bias": GUN}
    return run_engagement_until_destroyed(
        weapon_id, 22000.0, 0.0, TOLD, tc, db=db,
        lethal_radius_m=lethal_radius_m, **kw)


@pytest.fixture
def db():
    path = tempfile.mktemp(suffix=".db")
    d = EngagementDatabase(path)
    yield d
    d.close()
    os.remove(path)


# ---------------------------------------------------------------------------
# Layer 1 — Stops when destroyed (core proof)
# ---------------------------------------------------------------------------
def test_layer1_stops_on_destruction():
    r = fire(lethal_radius_m=8.0, max_rounds=15, min_learning_rounds=2)
    assert r["destroyed"] is True
    assert r["rounds_fired"] < r["max_rounds"]            # stopped early on success
    assert r["rounds_fired"] > r["min_learning_rounds"]   # had to learn first
    # The mission ended ON the destroying round (no extra rounds fired).
    assert r["destroying_round"] == r["rounds_fired"]
    kill = r["history"][-1]
    assert kill["destroyed_target"] is True
    assert kill["radial"] <= 8.0
    assert r["final_miss"] <= 8.0


def test_layer1_only_destroying_round_is_flagged():
    r = fire(lethal_radius_m=8.0)
    flags = [h["destroyed_target"] for h in r["history"]]
    assert flags.count(True) == 1 and flags[-1] is True


# ---------------------------------------------------------------------------
# Layer 2 — Adaptable threshold
# ---------------------------------------------------------------------------
def test_layer2_smaller_radius_needs_more_rounds():
    tight = fire(lethal_radius_m=2.0, max_rounds=20)
    loose = fire(lethal_radius_m=30.0, max_rounds=20)
    assert tight["destroyed"] and loose["destroyed"]
    assert tight["rounds_fired"] >= loose["rounds_fired"]


def test_layer2_large_radius_destroys_almost_immediately():
    # A radius larger than the first ranging miss destroys right after the
    # mandatory learning rounds.
    r = fire(lethal_radius_m=300.0, min_learning_rounds=2)
    assert r["destroyed"] is True
    assert r["destroying_round"] <= r["min_learning_rounds"] + 2
    # ...and clearly sooner than a tight radius on the same scenario.
    assert r["rounds_fired"] < fire(lethal_radius_m=2.0, max_rounds=20)["rounds_fired"]


# ---------------------------------------------------------------------------
# Layer 3 — Safety cap
# ---------------------------------------------------------------------------
def test_layer3_impossible_radius_hits_safety_cap():
    r = fire(lethal_radius_m=0.001, max_rounds=15)
    assert r["destroyed"] is False
    assert r["rounds_fired"] == 15
    assert r["destroying_round"] is None


def test_layer3_never_exceeds_max_rounds():
    for lethal, cap in [(0.001, 5), (0.001, 15), (8.0, 3), (2.0, 8)]:
        r = fire(lethal_radius_m=lethal, max_rounds=cap)
        assert r["rounds_fired"] <= cap


def test_layer3_rejects_nonpositive_radius():
    with pytest.raises(ValueError):
        fire(lethal_radius_m=0.0)


# ---------------------------------------------------------------------------
# Layer 4 — Fleet memory + sanity
# ---------------------------------------------------------------------------
def test_layer4_warm_start_destroys_in_fewer_rounds(db):
    cold = fire(lethal_radius_m=8.0, db=db, weapon_id="HOW-FLEET")
    warm = fire(lethal_radius_m=8.0, db=db, weapon_id="HOW-FLEET")
    assert cold["warm_started"] is False and warm["warm_started"] is True
    assert cold["destroyed"] and warm["destroyed"]
    assert warm["rounds_fired"] < cold["rounds_fired"]


def test_layer4_result_and_trace_fields():
    r = fire(lethal_radius_m=8.0)
    for key in ("destroyed", "rounds_fired", "destroying_round", "final_miss"):
        assert key in r
    assert isinstance(r["destroyed"], bool)
    for h in r["history"]:
        assert {"round", "phase", "miss", "radial", "destroyed_target"} <= set(h)
    assert r["rounds_fired"] == len(r["history"])


if __name__ == "__main__":
    print("\n=== Component 9 — fire until destroyed (v0=827, 22 km, gun (200,80)) ===\n")

    r = fire(lethal_radius_m=8.0)
    print(f"Default lethal radius 8.0 m  (target destroyed at round {r['destroying_round']}):")
    for h in r["history"]:
        mark = "  <== TARGET DESTROYED, end of mission" if h["destroyed_target"] else ""
        print(f"    round {h['round']:>2}  {h['phase']:<16} miss {h['radial']:>7.1f} m{mark}")
    print(f"  -> fired {r['rounds_fired']} of max {r['max_rounds']} rounds; "
          f"stopped the instant it was lethal.")

    print("\nAdaptable threshold (more rounds for a tighter kill criterion):")
    for L in (30.0, 8.0, 2.0):
        rr = fire(lethal_radius_m=L, max_rounds=20)
        print(f"    lethal {L:>5.1f} m -> {rr['rounds_fired']:>2} rounds "
              f"(destroyed at round {rr['destroying_round']})")

    print("\nSafety cap (impossibly tight radius never fires forever):")
    cap = fire(lethal_radius_m=0.001, max_rounds=15)
    print(f"    lethal 0.001 m -> destroyed={cap['destroyed']}, "
          f"fired {cap['rounds_fired']} (= max_rounds), final miss {cap['final_miss']:.2f} m")

    print("\nFleet memory (a remembered gun is destroyed sooner):")
    path = tempfile.mktemp(suffix=".db"); d = EngagementDatabase(path)
    cold = fire(db=d, weapon_id="DEMO"); warm = fire(db=d, weapon_id="DEMO")
    print(f"    engagement 1 (cold): {cold['rounds_fired']} rounds, kill at {cold['destroying_round']}")
    print(f"    engagement 2 (warm): {warm['rounds_fired']} rounds, kill at {warm['destroying_round']}"
          f"  -> {cold['rounds_fired']-warm['rounds_fired']} fewer rounds")
    d.close(); os.remove(path)
    print("\nRun 'python -m pytest tests/test_stop_condition.py -v'.")
