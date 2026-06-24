"""
Validation for the sim->dashboard live link (Round 4): the backend remembers the
most recently RUN mission so the dashboard can fetch and show the exact same
mission. This only stores-and-serves the already-computed package — no change to
the mission computation or physics.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arcs_server

BASE = {
    "weapon_id": "HOW-LINK", "target_range": 22000, "target_bearing": 0,
    "target_height_m": 0, "told_wind_dir": 180, "told_wind_speed": 20,
    "true_wind_dir": 180, "true_wind_speed": 23,
    "gun_bias_dr": 200, "gun_bias_cr": 80,
    "lethal_radius_m": 8.0, "max_rounds": 15, "warm_start": False,
}


def client():
    return arcs_server.app.test_client()


def test_no_mission_before_any_run():
    # A clean "nothing yet" state — never fabricated data.
    arcs_server._LAST_MISSION = None
    r = client().get("/api/p2/last_mission")
    assert r.status_code == 200
    assert r.get_json() == {"available": False}


def test_stores_exact_package_after_run():
    arcs_server._LAST_MISSION = None
    c = client()
    run = c.post("/api/p2/run_mission", json=BASE).get_json()
    last = c.get("/api/p2/last_mission").get_json()
    assert last["available"] is True
    pkg = last["mission_package"]
    # the stored package is byte-for-byte the mission the run returned
    assert pkg == run
    assert pkg["mission"]["rounds_fired"] == run["mission"]["rounds_fired"]
    assert pkg["learned"]["gun_bias_est"] == run["learned"]["gun_bias_est"]
    assert pkg["rounds"] == run["rounds"]


def test_latest_run_overwrites_previous():
    arcs_server._LAST_MISSION = None
    c = client()
    c.post("/api/p2/run_mission", json=BASE)
    second = c.post("/api/p2/run_mission",
                    json={**BASE, "target_range": 18000}).get_json()
    last = c.get("/api/p2/last_mission").get_json()["mission_package"]
    assert last["mission"]["target_range"] == 18000
    assert last == second


def test_unreachable_mission_is_stored_too():
    # An unreachable run is still the "last mission" — the dashboard shows the
    # honest unreachable state, not stale analytics.
    arcs_server._LAST_MISSION = None
    c = client()
    c.post("/api/p2/run_mission", json={**BASE, "target_range": 100000})
    last = c.get("/api/p2/last_mission").get_json()
    assert last["available"] is True
    assert last["mission_package"]["reachable"] is False


def test_store_does_not_change_computation():
    # Running through the route returns the same package build_mission produces
    # directly (storing is a side effect only).
    arcs_server._LAST_MISSION = None
    direct = arcs_server.build_mission(dict(BASE))
    routed = client().post("/api/p2/run_mission", json=BASE).get_json()
    assert routed["mission"]["rounds_fired"] == direct["mission"]["rounds_fired"]
    assert routed["learned"] == direct["learned"]


if __name__ == "__main__":
    arcs_server._LAST_MISSION = None
    c = client()
    print("\n=== Round 4 — sim->dashboard last-mission link ===\n")
    print("before any run:", c.get("/api/p2/last_mission").get_json())
    run = c.post("/api/p2/run_mission", json=BASE).get_json()
    last = c.get("/api/p2/last_mission").get_json()
    print("after a run: available =", last["available"],
          "| rounds_fired =", last["mission_package"]["mission"]["rounds_fired"],
          "| gun_est =", [round(x, 1) for x in last["mission_package"]["learned"]["gun_bias_est"]])
    print("matches the run:", last["mission_package"] == run)
    print("\nRun 'python -m pytest tests/test_last_mission.py -v'.")
