"""
Validation for the complete-mission endpoint (/api/p2/run_mission).

The endpoint returns the ENTIRE finished mission in one response so the future
visualizer holds no truth and computes no physics. These tests prove the
package is complete, consistent, REAL (matches the engagement loop directly),
reflects the stop condition, applies target altitude, and is valid JSON.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arcs_server
from arcs_server import build_mission
from physics.met_message import MetMessage
from physics.engagement import run_engagement_until_destroyed

BASE = {
    "weapon_id": "HOW-1", "target_range": 22000, "target_bearing": 0,
    "target_height_m": 0,
    "told_wind_dir": 180, "told_wind_speed": 20,
    "true_wind_dir": 180, "true_wind_speed": 23,
    "gun_bias_dr": 200, "gun_bias_cr": 80,
    "lethal_radius_m": 8.0, "max_rounds": 15, "warm_start": False,
}


def mission(**over):
    return build_mission({**BASE, **over})


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------
def test_completeness():
    m = mission()
    assert set(m) == {"mission", "rounds", "learned"}
    for k in ("destroyed", "destroying_round", "rounds_fired", "lethal_radius_m"):
        assert k in m["mission"]
    assert len(m["rounds"]) >= 1
    for r in m["rounds"]:
        assert len(r["trajectory"]) > 1                 # non-empty arc
        assert "x" in r["impact"] and "y" in r["impact"]
        assert isinstance(r["miss_m"], float)
        assert len(r["trajectory"][0]) == 3             # [x, y, z]
    lr = m["learned"]
    assert len(lr["gun_bias_est"]) == 2 and len(lr["gun_bias_true"]) == 2
    assert len(lr["atmo_correction_est"]) == 2 and len(lr["atmo_correction_true"]) == 2


# ---------------------------------------------------------------------------
# Consistency with the engagement
# ---------------------------------------------------------------------------
def test_consistency_rounds_and_kill():
    m = mission()
    assert m["mission"]["rounds_fired"] == len(m["rounds"])
    # round numbers are 1..N in order
    assert [r["round"] for r in m["rounds"]] == list(range(1, len(m["rounds"]) + 1))
    if m["mission"]["destroyed"]:
        kills = [r for r in m["rounds"] if r["destroyed_target"]]
        assert len(kills) == 1
        assert kills[0]["round"] == m["mission"]["destroying_round"]
        assert kills[0]["miss_m"] <= m["mission"]["lethal_radius_m"]


# ---------------------------------------------------------------------------
# Real data, not fabricated — matches the engagement loop called directly
# ---------------------------------------------------------------------------
def test_matches_engagement_directly():
    told = MetMessage.standard_isa(surface_wind=(BASE["told_wind_dir"], BASE["told_wind_speed"]))
    true = MetMessage.standard_isa(surface_wind=(BASE["true_wind_dir"], BASE["true_wind_speed"]))
    tc = {"true_met": true, "gun_bias": (BASE["gun_bias_dr"], BASE["gun_bias_cr"])}
    direct = run_engagement_until_destroyed(
        BASE["weapon_id"], BASE["target_range"], BASE["target_bearing"], told, tc,
        db=None, lethal_radius_m=BASE["lethal_radius_m"], max_rounds=BASE["max_rounds"],
        target_height_m=0.0)

    m = mission()
    assert m["mission"]["rounds_fired"] == direct["rounds_fired"]
    assert m["mission"]["destroyed"] == direct["destroyed"]
    assert m["mission"]["destroying_round"] == direct["destroying_round"]
    assert abs(m["mission"]["final_miss_m"] - direct["final_miss"]) < 1e-9
    assert np.allclose(m["learned"]["gun_bias_est"], direct["gun_bias_est"], atol=1e-9)
    assert np.allclose(m["learned"]["atmo_correction_est"], direct["atmo_correction_est"], atol=1e-9)
    # per-round misses match the engagement's history exactly
    assert np.allclose([r["miss_m"] for r in m["rounds"]],
                       [h["radial"] for h in direct["history"]], atol=1e-9)


# ---------------------------------------------------------------------------
# Stop condition faithfully reflected
# ---------------------------------------------------------------------------
def test_stop_condition_destroyed_early():
    m = mission(lethal_radius_m=8.0)
    assert m["mission"]["destroyed"] is True
    assert m["mission"]["rounds_fired"] < m["mission"]["max_rounds"]


def test_stop_condition_safety_cap():
    m = mission(lethal_radius_m=0.001, max_rounds=10)
    assert m["mission"]["destroyed"] is False
    assert m["mission"]["rounds_fired"] == 10
    assert m["mission"]["destroying_round"] is None


# ---------------------------------------------------------------------------
# Altitude passthrough (Component 8 actually applied)
# ---------------------------------------------------------------------------
def test_altitude_passthrough():
    m0 = mission(target_height_m=0)
    mhi = mission(target_height_m=500)
    # the arc ends exactly at the target altitude -> target_height reached the integrator
    assert abs(m0["rounds"][0]["trajectory"][-1][2] - 0.0) < 1e-6
    assert abs(mhi["rounds"][0]["trajectory"][-1][2] - 500.0) < 1e-6
    # a higher target changes the firing solution (different, correct direction:
    # more quadrant elevation to loft onto the raised target at the same range)
    assert mhi["mission"]["elevation_deg"] != m0["mission"]["elevation_deg"]
    assert mhi["mission"]["elevation_deg"] > m0["mission"]["elevation_deg"]


# ---------------------------------------------------------------------------
# numpy cast -> valid JSON, via the actual Flask route
# ---------------------------------------------------------------------------
def test_valid_json_no_numpy_leak():
    # build_mission output must be plain-Python JSON-serializable
    json.dumps(mission())
    # and the real route returns valid JSON 200
    client = arcs_server.app.test_client()
    resp = client.post("/api/p2/run_mission", json=BASE)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["mission"]["rounds_fired"] == len(data["rounds"])


def test_warm_start_reduces_rounds():
    cold = mission(warm_start=False)
    warm = mission(warm_start=True)
    assert warm["mission"]["warm_started"] is True
    assert warm["mission"]["rounds_fired"] <= cold["mission"]["rounds_fired"]


if __name__ == "__main__":
    m = mission()
    print("\n=== /api/p2/run_mission — one mission's returned structure ===\n")
    M = m["mission"]
    print(f"mission: destroyed={M['destroyed']} at round {M['destroying_round']} "
          f"of {M['rounds_fired']} fired (cap {M['max_rounds']}), "
          f"elevation {M['elevation_deg']:.1f} deg, final miss {M['final_miss_m']:.1f} m")
    print(f"\n{'rd':>3} {'phase':<16}{'arc pts':>8}{'apex m':>8}{'miss m':>9}  kill")
    for r in m["rounds"]:
        print(f"{r['round']:>3} {r['phase']:<16}{len(r['trajectory']):>8}"
              f"{r['apex_m']:>8.0f}{r['miss_m']:>9.1f}  {'<== DESTROYED' if r['destroyed_target'] else ''}")
    L = m["learned"]
    print(f"\nlearned (from misses alone):")
    print(f"  gun bias  est {np.round(L['gun_bias_est'],1)}  true {L['gun_bias_true']}")
    print(f"  wind error est {np.round(L['atmo_correction_est'],2)}  true {np.round(L['atmo_correction_true'],2)}")
    print(f"\npayload: {len(m['rounds'])} rounds, "
          f"{len(m['rounds'][0]['trajectory'])} points/arc — complete & self-contained.")
    print("\nRun 'python -m pytest tests/test_mission_endpoint.py -v'.")
