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


if __name__ == "__main__":
    print("\n" + "=" * 52)
    print("  ARCS Simulation Server")
    print("  Open: http://localhost:8765")
    print("=" * 52 + "\n")
    app.run(host="0.0.0.0", port=8765, debug=False)
