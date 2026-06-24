"""
Validation for the read-only weather-along-path endpoint (Round 5 UI).

It exposes the existing physics function weather_profile_along_path so the sim
can honestly visualize Component 12: wind that varies from the gun (measured) to
the target (estimated) plus a confidence that decreases toward the target. It
changes NO physics or mission computation — it only reads the function.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arcs_server


def client():
    return arcs_server.app.test_client()


BASE = {"target_range": 22000, "told_wind_dir": 180, "told_wind_speed": 10,
        "true_wind_dir": 180, "true_wind_speed": 30, "n_points": 12}


def test_returns_profile_of_requested_length():
    r = client().post("/api/p2/weather_profile", json=BASE)
    assert r.status_code == 200
    d = r.get_json()
    assert d["n_points"] == 12 and len(d["profile"]) == 12
    p0, pN = d["profile"][0], d["profile"][-1]
    assert p0["downrange_m"] == 0.0
    assert abs(pN["downrange_m"] - 22000) < 1e-6


def test_confidence_decreases_toward_target():
    prof = client().post("/api/p2/weather_profile", json=BASE).get_json()["profile"]
    confs = [p["confidence"] for p in prof]
    assert abs(confs[0] - 1.0) < 1e-9                      # certain at the gun
    assert confs[-1] < confs[0]                            # less certain at the target
    assert all(confs[i] >= confs[i+1] - 1e-12 for i in range(len(confs)-1))


def test_wind_varies_gun_to_target():
    # gun wind 10 m/s, target wind 30 m/s -> speed grows along the path (real data).
    prof = client().post("/api/p2/weather_profile", json=BASE).get_json()["profile"]
    speeds = [p["wind_speed_ms"] for p in prof]
    assert abs(speeds[0] - 10.0) < 1e-6
    assert abs(speeds[-1] - 30.0) < 1e-6
    assert speeds[-1] > speeds[0]


def test_reads_only_no_side_effects():
    # The weather endpoint must not disturb the stored last mission.
    arcs_server._LAST_MISSION = None
    client().post("/api/p2/weather_profile", json=BASE)
    assert arcs_server._LAST_MISSION is None


if __name__ == "__main__":
    prof = client().post("/api/p2/weather_profile", json=BASE).get_json()["profile"]
    print("\n=== Round 5 — weather-along-path endpoint ===\n")
    for p in prof[::3]:
        print(f"  downrange {p['downrange_m']:7.0f} m   wind {p['wind_speed_ms']:5.1f} m/s "
              f"from {p['wind_dir_deg']:5.0f}°   confidence {p['confidence']:.2f}")
    print("\nRun 'python -m pytest tests/test_weather_endpoint.py -v'.")
