"""
ARCS Simulation Server
Serves arcs_simulation.html and provides live API endpoints so the
3-D visualisation uses real physics, real robot bias, and real PINN corrections.

Run:
    python arcs_server.py
Then open: http://localhost:5000
"""

import os
import tempfile

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask import make_response

from physics.ballistic_solver import BallisticSolver
from physics.bias_model import RobotBiasModel
from pinn_corrector import PINNCorrector

# ── Phase 2: frozen physics + estimation, called only as tools ───────────────
from physics import trajectory as _p2traj
from physics.trajectory import integrate_trajectory
from physics.met_message import MetMessage, wind_vector_from_dir_speed
from physics.engagement import (run_engagement, run_engagement_until_destroyed,
                                _save_gun_bias, _ensure_phase2_columns)
from physics.horizontal_met import HorizontalMetField, weather_profile_along_path
from engagement_database import EngagementDatabase

app = Flask(__name__, static_folder=".")

# ── Initialise once at startup ───────────────────────────────────────────────
_solver  = BallisticSolver()
_bias    = RobotBiasModel(seed=42)   # must match EngagementSimulator seed in pipeline
_cf_low  = PINNCorrector("data/range_table_corrections.csv", solution_type="LOW")
_cf_high = PINNCorrector("data/range_table_corrections.csv", solution_type="HIGH")
_cf_low.load_and_train(verbose=False)
_cf_high.load_and_train(verbose=False)

print(f"  PINN LOW  fitted : {_cf_low.is_fitted}  ({_cf_low.n_records} records)")
print(f"  PINN HIGH fitted : {_cf_high.is_fitted}  ({_cf_high.n_records} records)")
print(f"  Robot bias       : {_bias.summary()}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    resp = make_response(send_from_directory(".", "arcs_simulation.html"))
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.route("/api/solve", methods=["POST"])
def solve():
    """
    Body: { x, y, z, v0 (opt, default 100), prefer (opt, default 'LOW') }
    Returns physics solution + robot bias + PINN correction.
    """
    d = request.get_json(force=True)
    x      = float(d.get("x", 0))
    y      = float(d.get("y", 0))
    z      = float(d.get("z", 0))
    v0     = float(d.get("v0", 100.0))
    prefer = str(d.get("prefer", "LOW"))

    sol = _solver.solve(x, y, z, v0, prefer=prefer)

    if not sol.reachable:
        return jsonify({"reachable": False, "error": sol.error_message})

    # Real systematic bias at this pitch/yaw
    bias_vals = _bias.expected_bias(sol.turret_pitch_deg, sol.turret_yaw_deg, v0)

    # PINN correction from the fitted network
    cf = _cf_high if sol.solution_type == "HIGH" else _cf_low
    pinn = {"delta_pitch": 0.0, "delta_yaw": 0.0, "delta_v0": 0.0, "source": "none"}
    if cf.is_fitted:
        raw = cf.predict(sol.horiz_range, y, v0)
        pinn = {
            "delta_pitch": float(raw.get("delta_pitch", 0.0)),
            "delta_yaw":   float(raw.get("delta_yaw",   0.0)),
            "delta_v0":    float(raw.get("delta_v0",    0.0)),
            "source":      str(raw.get("source", "none")),
        }

    return jsonify({
        "reachable":     True,
        "pitch_deg":     sol.turret_pitch_deg,
        "yaw_deg":       sol.turret_yaw_deg,
        "v0":            v0,
        "tof":           sol.tof,
        "max_height":    sol.max_height,
        "horiz_range":   sol.horiz_range,
        "solution_type": sol.solution_type,
        "bias": {
            "pitch": float(bias_vals["pitch_bias"]),
            "yaw":   float(bias_vals["yaw_bias"]),
            "v0":    float(bias_vals["v0_bias"]),
        },
        "pinn": pinn,
    })


@app.route("/api/status")
def status():
    """Quick health-check / PINN status for the HUD badge."""
    return jsonify({
        "pinn_low_fitted":   _cf_low.is_fitted,
        "pinn_high_fitted":  _cf_high.is_fitted,
        "pinn_low_records":  _cf_low.n_records,
        "pinn_high_records": _cf_high.n_records,
        "robot_bias":        _bias.summary(),
    })


# ════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Backend-driven analysis view (thin wiring; no physics changes).
# Every numeric result is cast to a plain Python float before jsonify because
# the frozen code returns numpy types.
# ════════════════════════════════════════════════════════════════════════════

def _met_from_wind(direction_deg, speed_ms):
    """Build a standard-ISA MET with a uniform surface wind, exactly as
    tests/test_engagement.py does."""
    return MetMessage.standard_isa(surface_wind=(float(direction_deg), float(speed_ms)))


def _trajectory_path(v0, elevation_deg, azimuth_deg, met, dt=0.01, sample_every=40,
                     target_height_m=0.0):
    """Real flight path as a list of (x, y, z), produced by stepping the FROZEN
    integrator internals (_rk4_step -> _acceleration). The physics lives in the
    frozen module untouched; this only collects intermediate states and detects
    the impact crossing, mirroring integrate_trajectory's own loop.

    target_height_m: impact altitude (Component 8). 0.0 (default) reproduces the
    former z=0 ground-crossing behaviour exactly."""
    theta = np.radians(elevation_deg)
    phi = np.radians(azimuth_deg)
    vx = v0 * np.cos(theta) * np.cos(phi)
    vy = v0 * np.cos(theta) * np.sin(phi)
    vz = v0 * np.sin(theta)
    state = np.array([0.0, 0.0, 0.0, vx, vy, vz])

    mass, area, Cd = 43.2, 0.018869, _p2traj.CD_PLACEHOLDER
    # Component 10 spin-drift params (match integrate_trajectory's defaults), so
    # the drawn arc is consistent with the real validated physics.
    diameter = 2.0 * np.sqrt(area / np.pi)
    spin_p = 2.0 * np.pi * v0 / (20.0 * diameter)   # standard ~20 cal/turn, right-hand
    spin_dir, ux, uy = 1.0, np.cos(phi), np.sin(phi)
    # Coriolis (Component 11) at the default mid-latitude, matching the integrator.
    lat = np.radians(20.0)
    cor_on = True
    cor_omega = np.array([0.0, _p2traj.OMEGA_EARTH * np.cos(lat),
                          _p2traj.OMEGA_EARTH * np.sin(lat)])
    pts = [[0.0, 0.0, 0.0]]
    speeds = [float(np.linalg.norm(state[3:6]))]   # real speed (m/s) per point
    step = 0
    while step < 10_000_000:
        new = _p2traj._rk4_step(state, dt, mass, area, Cd, False,
                                (0.0, 0.0, 0.0), met, True, 1.0,
                                spin_p, spin_dir, ux, uy, cor_on, cor_omega)
        step += 1
        # Impact at the target altitude on the descending branch (z = 0 default).
        if new[2] < target_height_m <= state[2]:
            frac = (state[2] - target_height_m) / (state[2] - new[2])
            impact = state + frac * (new - state)
            pts.append([float(impact[0]), float(impact[1]), float(target_height_m)])
            speeds.append(float(np.linalg.norm(impact[3:6])))
            break
        state = new
        if step % sample_every == 0:
            pts.append([float(state[0]), float(state[1]), float(state[2])])
            speeds.append(float(np.linalg.norm(state[3:6])))
    return pts, speeds


@app.route("/p2")
def p2_index():
    resp = make_response(send_from_directory(".", "arcs_p2.html"))
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.route("/p2/sim")
def p2_sim_index():
    resp = make_response(send_from_directory(".", "arcs_p2_sim.html"))
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.route("/api/p2/trajectory", methods=["POST"])
def p2_trajectory():
    """Body: { v0, elevation, azimuth (opt), wind_dir (opt), wind_speed (opt) }.
    Returns the real arc points + apex/range/tof from the frozen integrator."""
    d = request.get_json(force=True)
    v0 = float(d.get("v0", 827.0))
    elevation = float(d.get("elevation", 45.0))
    azimuth = float(d.get("azimuth", 0.0))
    met = None
    if d.get("wind_dir") is not None or d.get("wind_speed") is not None:
        met = _met_from_wind(d.get("wind_dir", 0.0), d.get("wind_speed", 0.0))

    summary = integrate_trajectory(v0=v0, elevation_deg=elevation,
                                   azimuth_deg=azimuth, met=met, use_g7=True)
    points, speeds = _trajectory_path(v0, elevation, azimuth, met)
    return jsonify({
        "points": points,
        "speeds": speeds,
        "apex_m": float(summary["apex_m"]),
        "range_m": float(summary["range_m"]),
        "tof_s": float(summary["tof_s"]),
        "impact_speed": float(summary["impact_speed"]),
    })


@app.route("/api/p2/engagement", methods=["POST"])
def p2_engagement():
    """Body: weapon_id, target_range, target_bearing, told/true wind dir+speed,
    gun_bias_dr, gun_bias_cr, optional n_register/n_adjust/n_ffe.
    Builds told/true MET exactly as the tests do, calls the frozen
    run_engagement, and returns the full result as plain-float JSON."""
    d = request.get_json(force=True)
    weapon_id = str(d.get("weapon_id", "HOW-1"))
    target_range = float(d.get("target_range", 22000.0))
    target_bearing = float(d.get("target_bearing", 0.0))

    told_dir = float(d.get("told_wind_dir", 180.0))
    told_spd = float(d.get("told_wind_speed", 20.0))
    true_dir = float(d.get("true_wind_dir", 180.0))
    true_spd = float(d.get("true_wind_speed", 23.0))
    gun_dr = float(d.get("gun_bias_dr", 200.0))
    gun_cr = float(d.get("gun_bias_cr", 80.0))

    told_met = _met_from_wind(told_dir, told_spd)
    true_met = _met_from_wind(true_dir, true_spd)
    true_conditions = {"true_met": true_met, "gun_bias": (gun_dr, gun_cr)}

    kwargs = {}
    for k in ("n_register", "n_adjust", "n_ffe"):
        if d.get(k) is not None:
            kwargs[k] = int(d[k])

    res = run_engagement(weapon_id, target_range, target_bearing,
                         told_met, true_conditions, **kwargs)

    # True effective wind error, rotated into the firing range/cross frame so
    # the learned-vs-true panel can compare like-for-like.
    delta = (wind_vector_from_dir_speed(true_dir, true_spd)
             - wind_vector_from_dir_speed(told_dir, told_spd))
    phi = np.radians(target_bearing)
    u_dr = np.array([np.cos(phi), np.sin(phi)])
    u_cr = np.array([-np.sin(phi), np.cos(phi)])
    wind_true = [float(np.dot(delta[:2], u_dr)), float(np.dot(delta[:2], u_cr))]

    return jsonify({
        "weapon_id": res["weapon_id"],
        "target_range": float(res["target_range"]),
        "target_bearing": float(res["target_bearing"]),
        "elevation_deg": float(res["elevation_deg"]),
        "warm_started": bool(res["warm_started"]),
        "final_cep": (float(res["final_cep"]) if res["final_cep"] is not None else None),
        "rounds_to_converge": int(res["rounds_to_converge"]),
        "total_shots": int(res["total_shots"]),
        "gun_bias_est": [float(x) for x in res["gun_bias_est"]],
        "gun_bias_true": [gun_dr, gun_cr],
        "atmo_correction_est": [float(x) for x in res["atmo_correction_est"]],
        "wind_true": wind_true,
        "phase_misses": {k: [float(x) for x in v]
                         for k, v in res["phase_misses"].items()},
        "history": [{"round": int(h["round"]), "phase": h["phase"],
                     "radial": float(h["radial"]),
                     "miss": [float(h["miss"][0]), float(h["miss"][1])]}
                    for h in res["history"]],
    })


# ════════════════════════════════════════════════════════════════════════════
# Complete-mission endpoint — one call returns the ENTIRE finished mission so
# the (future) visualizer holds no truth and computes no physics. It runs the
# REAL Component 9 loop and, per round, the frozen integrator for the arc. Every
# numpy value is cast to a plain float before jsonify.
# ════════════════════════════════════════════════════════════════════════════
def _rc_axes(bearing_deg):
    phi = np.radians(bearing_deg)
    return (np.array([np.cos(phi), np.sin(phi)]),      # downrange unit
            np.array([-np.sin(phi), np.cos(phi)]))     # cross-range unit


def build_mission(d):
    """Run a full mission and assemble the complete, self-contained package.
    Pure-Python/float output; safe to jsonify. Factored out of the route so the
    tests can drive it directly."""
    weapon_id = str(d.get("weapon_id", "HOW-1"))
    target_range = float(d.get("target_range", 22000.0))
    target_bearing = float(d.get("target_bearing", 0.0))
    target_height_m = float(d.get("target_height_m", 0.0))
    told_dir = float(d.get("told_wind_dir", 180.0))
    told_spd = float(d.get("told_wind_speed", 20.0))
    true_dir = float(d.get("true_wind_dir", 180.0))
    true_spd = float(d.get("true_wind_speed", 23.0))
    gun_dr = float(d.get("gun_bias_dr", 200.0))
    gun_cr = float(d.get("gun_bias_cr", 80.0))
    lethal_radius_m = float(d.get("lethal_radius_m", 8.0))
    max_rounds = int(d.get("max_rounds", 15))
    warm_start = bool(d.get("warm_start", False))

    told_met = _met_from_wind(told_dir, told_spd)
    true_met = _met_from_wind(true_dir, true_spd)
    true_conditions = {"true_met": true_met, "gun_bias": (gun_dr, gun_cr)}

    # warm_start: hand the loop a DB that already remembers this gun's bias, so
    # Component 9 skips registration (a remembered weapon). Temp DB, cleaned up.
    db = None
    db_path = None
    if warm_start:
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db = EngagementDatabase(db_path)
        _ensure_phase2_columns(db)
        _save_gun_bias(db, weapon_id, "155mm-M107", (gun_dr, gun_cr), 1, 0)

    try:
        res = run_engagement_until_destroyed(
            weapon_id, target_range, target_bearing, told_met, true_conditions,
            db=db, lethal_radius_m=lethal_radius_m, max_rounds=max_rounds,
            target_height_m=target_height_m)
    finally:
        if db is not None:
            db.close()
            os.remove(db_path)

    # Honest reachability guard (Option A): an unreachable target fired zero
    # rounds and was never "destroyed". Return a clean, distinct package the UI
    # can show as UNREACHABLE — no arc, no impacts, no fabricated solution.
    if not res.get("reachable", True):
        return {
            "reachable": False,
            "reason": res["reason"],
            "max_range_m": res["max_range_m"],
            "target_range_m": res["target_range_m"],
            "mission": {
                "weapon_id": weapon_id,
                "target_range": target_range,
                "target_bearing": target_bearing,
                "target_height_m": target_height_m,
                "lethal_radius_m": lethal_radius_m,
                "max_rounds": max_rounds,
                "destroyed": False,
                "destroying_round": None,
                "rounds_fired": 0,
                "warm_started": bool(warm_start),
                "elevation_deg": None,
                "final_miss_m": None,
            },
            "rounds": [],
            "learned": None,
        }

    elevation = float(res["elevation_deg"])

    # Real shell arc — fired at the firing elevation through the true atmosphere
    # to the target altitude. Same geometry each round; the per-round impact
    # below carries the learning. (~70-100 sampled points for smooth drawing.)
    pts, speeds = _trajectory_path(827.0, elevation, 0.0, true_met,
                                   sample_every=60, target_height_m=target_height_m)
    summ = integrate_trajectory(v0=827.0, elevation_deg=elevation, azimuth_deg=0.0,
                                met=true_met, use_g7=True,
                                target_height_m=target_height_m)
    arc = [[float(p[0]), float(p[1]), float(p[2])] for p in pts]
    apex_m = float(summ["apex_m"])
    tof_s = float(summ["tof_s"])
    impact_speed = float(summ["impact_speed"])

    rounds = []
    for h in res["history"]:
        miss = h["miss"]
        rounds.append({
            "round": int(h["round"]),
            "phase": h["phase"],
            "trajectory": arc,
            "apex_m": apex_m,
            "tof_s": tof_s,
            "impact_speed": impact_speed,
            # where this round landed, downrange/cross (target + miss):
            "impact": {"x": float(target_range + miss[0]), "y": float(miss[1])},
            "miss_m": float(h["radial"]),
            "destroyed_target": bool(h["destroyed_target"]),
        })

    # True effective wind error rotated into the firing range/cross frame.
    delta = (wind_vector_from_dir_speed(true_dir, true_spd)
             - wind_vector_from_dir_speed(told_dir, told_spd))
    u_dr, u_cr = _rc_axes(target_bearing)
    atmo_true = [float(np.dot(delta[:2], u_dr)), float(np.dot(delta[:2], u_cr))]

    return {
        "mission": {
            "weapon_id": weapon_id,
            "target_range": target_range,
            "target_bearing": target_bearing,
            "target_height_m": target_height_m,
            "lethal_radius_m": lethal_radius_m,
            "max_rounds": max_rounds,
            "destroyed": bool(res["destroyed"]),
            "destroying_round": (int(res["destroying_round"])
                                 if res["destroying_round"] is not None else None),
            "rounds_fired": int(res["rounds_fired"]),
            "warm_started": bool(res["warm_started"]),
            "elevation_deg": elevation,
            "final_miss_m": (float(res["final_miss"])
                             if res["final_miss"] is not None else None),
        },
        "rounds": rounds,
        "learned": {
            "gun_bias_est": [float(x) for x in res["gun_bias_est"]],
            "gun_bias_true": [gun_dr, gun_cr],
            "atmo_correction_est": [float(x) for x in res["atmo_correction_est"]],
            "atmo_correction_true": atmo_true,
        },
    }


# In-memory "last mission" store (Round 4 sim->dashboard live link). The backend
# remembers the most recently RUN mission package so the dashboard can fetch and
# display the exact same mission the sim just ran — a single source of truth, no
# brittle browser-to-browser messaging. It holds the already-computed result; it
# changes no physics or mission computation.
_LAST_MISSION = None


@app.route("/api/p2/run_mission", methods=["POST"])
def p2_run_mission():
    """One call -> the COMPLETE finished mission (rounds, trajectories, impacts,
    learning, stop). The browser animates this; it computes no physics. The
    result is also cached as the 'last mission' for the dashboard live link."""
    global _LAST_MISSION
    try:
        package = build_mission(request.get_json(force=True))
        _LAST_MISSION = package          # remember it for /api/p2/last_mission
        return jsonify(package)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/p2/weather_profile", methods=["POST"])
def p2_weather_profile():
    """Read-only Component-12 weather-along-path data for honest visualization.
    Builds a HorizontalMetField whose GUN profile is the told/measured wind and
    whose TARGET profile is the true wind, then returns weather_profile_along_path:
    the wind that varies from gun to target plus a confidence that decreases
    toward the target (known near the gun, estimated downrange). This only reads
    the existing physics function — it changes no mission computation."""
    d = request.get_json(force=True)
    target_range = float(d.get("target_range", 22000.0))
    told_dir = float(d.get("told_wind_dir", 180.0))
    told_spd = float(d.get("told_wind_speed", 20.0))
    true_dir = float(d.get("true_wind_dir", 180.0))
    true_spd = float(d.get("true_wind_speed", 23.0))
    n_points = int(d.get("n_points", 16))
    field = HorizontalMetField(_met_from_wind(told_dir, told_spd),     # measured at gun
                               _met_from_wind(true_dir, true_spd),     # real toward target
                               target_range)
    profile = weather_profile_along_path(field, n_points=n_points)
    return jsonify({"target_range_m": target_range,
                    "n_points": len(profile), "profile": profile})


@app.route("/api/p2/last_mission")
def p2_last_mission():
    """The most recently run mission (for the dashboard 'live from sim' mode), or
    a clean no-mission state before anything has been run. Never fabricates data."""
    if _LAST_MISSION is None:
        return jsonify({"available": False})
    return jsonify({"available": True, "mission_package": _LAST_MISSION})


if __name__ == "__main__":
    print("\n" + "=" * 52)
    print("  ARCS Simulation Server")
    print("  Phase 1 sim : http://localhost:8766/")
    print("  Phase 2 view: http://localhost:8766/p2")
    print("=" * 52 + "\n")
    app.run(host="0.0.0.0", port=8766, debug=False)
