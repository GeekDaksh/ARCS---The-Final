"""
ARCS Simulation Server
Serves arcs_simulation.html and provides live API endpoints so the
3-D visualisation uses real physics, real robot bias, and real PINN corrections.

Run:
    python arcs_server.py
Then open: http://localhost:5000
"""

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
from physics.engagement import run_engagement

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


def _trajectory_path(v0, elevation_deg, azimuth_deg, met, dt=0.01, sample_every=40):
    """Real flight path as a list of (x, y, z), produced by stepping the FROZEN
    integrator internals (_rk4_step -> _acceleration). The physics lives in the
    frozen module untouched; this only collects intermediate states and detects
    the ground crossing, mirroring integrate_trajectory's own loop."""
    theta = np.radians(elevation_deg)
    phi = np.radians(azimuth_deg)
    vx = v0 * np.cos(theta) * np.cos(phi)
    vy = v0 * np.cos(theta) * np.sin(phi)
    vz = v0 * np.sin(theta)
    state = np.array([0.0, 0.0, 0.0, vx, vy, vz])

    mass, area, Cd = 43.2, 0.018869, _p2traj.CD_PLACEHOLDER
    pts = [[0.0, 0.0, 0.0]]
    step = 0
    while step < 10_000_000:
        new = _p2traj._rk4_step(state, dt, mass, area, Cd, False,
                                (0.0, 0.0, 0.0), met, True, 1.0)
        step += 1
        if new[2] < 0.0:  # ground crossing on the way down: interpolate impact
            frac = state[2] / (state[2] - new[2])
            impact = state + frac * (new - state)
            pts.append([float(impact[0]), float(impact[1]), 0.0])
            break
        state = new
        if step % sample_every == 0:
            pts.append([float(state[0]), float(state[1]), float(state[2])])
    return pts


@app.route("/p2")
def p2_index():
    resp = make_response(send_from_directory(".", "arcs_p2.html"))
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
    points = _trajectory_path(v0, elevation, azimuth, met)
    return jsonify({
        "points": points,
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


if __name__ == "__main__":
    print("\n" + "=" * 52)
    print("  ARCS Simulation Server")
    print("  Phase 1 sim : http://localhost:8766/")
    print("  Phase 2 view: http://localhost:8766/p2")
    print("=" * 52 + "\n")
    app.run(host="0.0.0.0", port=8766, debug=False)
