"""
ARCS — Kalman Filter Test Suite
Phase 1 (EngagementKF) + Phase 2 foundation (TargetTrackingKF)

Sections (6 total, ~35 tests):
    [1] EngagementKF initialisation — correct prior, dimensions
    [2] EngagementKF update — posterior converges toward true bias
    [3] EngagementKF properties — confidence rises, uncertainty falls
    [4] EngagementKF update_scalar — degraded update still converges
    [5] TargetTrackingKF — predict, update, predict_at
    [6] Integration — KF estimate feeds into corrected firing solution
"""

import numpy as np
import sys, os, time, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kalman_filter import EngagementKF, TargetTrackingKF

RNG  = np.random.default_rng(int(time.time()))
PASS = 0
FAIL = 0
LOG  = []

def check(name, cond, detail=""):
    global PASS, FAIL
    status = "PASS" if cond else "FAIL"
    if cond: PASS += 1
    else:    FAIL += 1
    print(f"  {'✓' if cond else '✗'} {status:4s}  {name:58s} {detail}")
    LOG.append({"test": name, "status": status,
                "detail": detail, "time": datetime.datetime.now()})

print("=" * 68)
print("ARCS — Kalman Filter Test Suite")
print(f"Seed: {int(time.time())}  (new random values every run)")
print("=" * 68)

# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] ENGAGEMENTKF INITIALISATION — correct prior and dimensions")

kf0 = EngagementKF(range_m=200.0, sigma_pitch=0.3, sigma_yaw=0.2)

check("Initial state is [0, 0] (no correction known)",
      np.allclose(kf0.x, [0.0, 0.0]),
      f"x={kf0.x}")
check("Initial covariance P is 2×2",
      kf0.P.shape == (2, 2))
check("Initial P diagonal is positive (uncertainty > 0)",
      kf0.P[0, 0] > 0 and kf0.P[1, 1] > 0,
      f"P_diag=[{kf0.P[0,0]:.4f},{kf0.P[1,1]:.4f}]")
check("Initial correction_deg is (0.0, 0.0)",
      kf0.correction_deg == (0.0, 0.0),
      f"got {kf0.correction_deg}")
check("Initial uncertainty_deg is positive",
      kf0.uncertainty_deg[0] > 0 and kf0.uncertainty_deg[1] > 0,
      f"unc={kf0.uncertainty_deg}")
check("Initial confidence ≈ 0 (no information yet, < 0.001)",
      kf0.confidence < 0.001,
      f"conf={kf0.confidence:.2e}")
check("n_updates starts at 0",
      kf0.n_updates == 0)

# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] ENGAGEMENTKF UPDATE — posterior converges toward true bias")
# Simulate a robot with a known systematic bias:
# pitch bias = -0.5°, yaw bias = +0.3°  (what the robot underaims / sidelists)
# At R=200m: error_x = +200*tan(0.5°) ≈ +1.745m (long by 1.745m)
#            error_z = -200*tan(0.3°) ≈ -1.047m (left by 1.047m)
# After correction by (-dp, -dy), shots should hit center.

TRUE_PITCH_BIAS_DEG = -0.50   # robot undershoots
TRUE_YAW_BIAS_DEG   = +0.30   # robot lists right

def simulate_shot(rng, range_m=200.0, sigma_p=0.3, sigma_y=0.2,
                  dp_bias=TRUE_PITCH_BIAS_DEG, dy_bias=TRUE_YAW_BIAS_DEG):
    """Simulate miss due to systematic bias + stochastic noise."""
    actual_dp = dp_bias + rng.normal(0, sigma_p)
    actual_dy = dy_bias + rng.normal(0, sigma_y)
    error_x = range_m * np.tan(np.deg2rad(actual_dp))
    error_z = range_m * np.tan(np.deg2rad(actual_dy))
    return float(error_x), float(error_z)

kf2 = EngagementKF(range_m=200.0, sigma_pitch=0.3, sigma_yaw=0.2)
for _ in range(20):
    ex, ez = simulate_shot(RNG)
    kf2.update(ex, ez)

dp_est, dy_est = kf2.correction_deg
# The KF should estimate a correction that COMPENSATES the bias:
# dp_est ≈ -TRUE_PITCH_BIAS_DEG, dy_est ≈ -TRUE_YAW_BIAS_DEG
check("After 20 shots: pitch correction converges toward -bias (±0.3°)",
      abs(dp_est - (-TRUE_PITCH_BIAS_DEG)) < 0.3,
      f"est={dp_est:+.3f}°  truth={-TRUE_PITCH_BIAS_DEG:+.3f}°")
check("After 20 shots: yaw correction converges toward -bias (±0.2°)",
      abs(dy_est - (-TRUE_YAW_BIAS_DEG)) < 0.2,
      f"est={dy_est:+.3f}°  truth={-TRUE_YAW_BIAS_DEG:+.3f}°")
check("n_updates == 20",
      kf2.n_updates == 20, f"got {kf2.n_updates}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] ENGAGEMENTKF PROPERTIES — confidence rises, uncertainty falls")

kf3 = EngagementKF(range_m=200.0, sigma_pitch=0.3, sigma_yaw=0.2)
unc0 = kf3.uncertainty_deg
conf0 = kf3.confidence

confidences = [conf0]
uncertainties_p = [unc0[0]]
for _ in range(10):
    ex, ez = simulate_shot(RNG)
    kf3.update(ex, ez)
    confidences.append(kf3.confidence)
    uncertainties_p.append(kf3.uncertainty_deg[0])

check("Confidence monotonically non-decreasing over 10 shots",
      all(confidences[i] <= confidences[i+1] + 1e-9
          for i in range(len(confidences)-1)),
      f"final conf={confidences[-1]:.3f}")
check("Confidence > 0 after 10 shots",
      confidences[-1] > 0.0,
      f"conf={confidences[-1]:.3f}")
check("Confidence ≤ 1.0 at all times",
      all(0.0 <= c <= 1.0 for c in confidences),
      f"max={max(confidences):.3f}")
check("Uncertainty decreases after 10 shots",
      uncertainties_p[-1] < uncertainties_p[0],
      f"before={uncertainties_p[0]:.4f}°  after={uncertainties_p[-1]:.4f}°")
check("summary() returns a non-empty string",
      len(kf3.summary()) > 0,
      kf3.summary()[:40])

# More shots at different range
kf3b = EngagementKF(range_m=float(RNG.uniform(80, 400)))
for _ in range(5):
    kf3b.update(float(RNG.normal(2, 1)), float(RNG.normal(-1, 0.5)))
check("Works at random range",
      np.all(np.isfinite(kf3b.x)),
      f"x={kf3b.x}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] ENGAGEMENTKF UPDATE_SCALAR — scalar miss still converges")

kf4 = EngagementKF(range_m=200.0, sigma_pitch=0.3, sigma_yaw=0.2)
# Fire 10 shots with a consistent miss of ~5m
for _ in range(10):
    miss = abs(float(RNG.normal(5.0, 1.0)))
    kf4.update_scalar(miss)

dp4, dy4 = kf4.correction_deg
check("update_scalar: correction moves from zero",
      abs(dp4) > 0 or abs(dy4) > 0,
      f"dp={dp4:+.3f}° dy={dy4:+.3f}°")
check("update_scalar: state remains finite",
      np.all(np.isfinite([dp4, dy4])),
      f"dp={dp4:.4f} dy={dy4:.4f}")
check("update_scalar: n_updates increments",
      kf4.n_updates == 10, f"got {kf4.n_updates}")
check("update_scalar: confidence > 0 after 10 shots",
      kf4.confidence > 0.0, f"conf={kf4.confidence:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] TARGETTRACKINGKF — predict, update, predict_at")

# Simulate target at (150, 0, 50) moving at vx=8 m/s
kf5 = TargetTrackingKF(sensor_noise_m=3.0, dt=0.1)

TRUE_X0, TRUE_Y0, TRUE_Z0  = 150.0, 0.0, 50.0
TRUE_VX, TRUE_VY, TRUE_VZ  = 8.0,   0.0,  0.0
N_OBS = 30    # 3s of observations — enough for velocity to converge with 3m sensor noise
DT    = 0.1

for tick in range(N_OBS):
    t = tick * DT
    x_true = TRUE_X0 + TRUE_VX * t
    y_true = TRUE_Y0 + TRUE_VY * t
    z_true = TRUE_Z0 + TRUE_VZ * t
    # Noisy sensor reading
    x_obs = x_true + float(RNG.normal(0, 3.0))
    y_obs = y_true + float(RNG.normal(0, 3.0))
    z_obs = z_true + float(RNG.normal(0, 3.0))
    kf5.predict(dt=DT)
    kf5.update(x_obs, y_obs, z_obs)

pos = kf5.position
vel = kf5.velocity

check("TargetKF: position estimate is 3-element tuple",
      len(pos) == 3,
      f"pos={pos}")
check("TargetKF: x position within 10m of truth after 10 obs",
      abs(pos[0] - (TRUE_X0 + TRUE_VX * N_OBS * DT)) < 10.0,
      f"est={pos[0]:.1f}m  truth={TRUE_X0 + TRUE_VX*N_OBS*DT:.1f}m")
check("TargetKF: velocity estimate is 3-element tuple",
      len(vel) == 3,
      f"vel={vel}")
check("TargetKF: vx estimate within 3 m/s of truth",
      abs(vel[0] - TRUE_VX) < 3.0,
      f"est={vel[0]:.2f}m/s  truth={TRUE_VX:.2f}m/s")

# predict_at: where will target be after TOF=2.0s?
tof = 2.0
px, py, pz = kf5.predict_at(tof)
x_true_at_tof = TRUE_X0 + TRUE_VX * (N_OBS * DT + tof)
check("predict_at(2.0s): x position within 15m of truth",
      abs(px - x_true_at_tof) < 15.0,
      f"predicted={px:.1f}m  truth≈{x_true_at_tof:.1f}m")
check("predict_at: all three coordinates are finite",
      all(np.isfinite([px, py, pz])),
      f"({px:.1f},{py:.1f},{pz:.1f})")

# summary string
check("TargetKF summary() is non-empty",
      len(kf5.summary()) > 0,
      kf5.summary()[:50])

# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] INTEGRATION — KF estimate feeds into firing solution")
from physics.ballistic_solver import BallisticSolver

# Fire 8 shots, feed miss into KF, then apply KF correction to firing solution
solver = BallisticSolver()
TARGET = (200.0, 0.0, 0.0)
V0     = 100.0
sol    = solver.solve(*TARGET, V0)

check("Ballistic solution is reachable for integration test",
      sol.reachable, f"type={sol.solution_type}")

if sol.reachable:
    kf6 = EngagementKF(range_m=sol.horiz_range,
                       sigma_pitch=0.3, sigma_yaw=0.2)
    # Simulate shots with known bias
    for _ in range(8):
        ex, ez = simulate_shot(RNG, range_m=sol.horiz_range)
        kf6.update(ex, ez)

    dp6, dy6 = kf6.correction_deg
    corrected_pitch = sol.turret_pitch_deg + dp6
    corrected_yaw   = sol.turret_yaw_deg   + dy6

    check("KF-corrected pitch is within valid mechanical range",
          0.0 <= corrected_pitch <= 85.0,
          f"pitch={corrected_pitch:.3f}°")
    check("KF-corrected yaw is within ±180°",
          -180.0 <= corrected_yaw <= 180.0,
          f"yaw={corrected_yaw:.3f}°")
    check("KF correction moves pitch in the right direction (toward −bias)",
          np.sign(dp6) == np.sign(-TRUE_PITCH_BIAS_DEG)
          or abs(dp6) < 0.05,  # near-zero is also valid (noise-dominated)
          f"dp6={dp6:+.3f}°  expected sign {np.sign(-TRUE_PITCH_BIAS_DEG)}")
    check("KF confidence after 8 shots > 0.5",
          kf6.confidence > 0.5,
          f"conf={kf6.confidence:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*68}")
print(f"  TOTAL: {PASS+FAIL}  |  PASSED: {PASS}  |  FAILED: {FAIL}")
print(f"{'='*68}")

import os as _os, pandas as _pd
_os.makedirs(_os.path.join(_os.path.dirname(__file__), '..', 'data'), exist_ok=True)
_out = _os.path.join(_os.path.dirname(__file__), '..', 'data',
                     'test_results_kalman_filter.csv')
_df = _pd.DataFrame(LOG)
if _os.path.exists(_out):
    _df = _pd.concat([_pd.read_csv(_out), _df], ignore_index=True)
_df.to_csv(_out, index=False)
print(f"\n  Results saved → data/test_results_kalman_filter.csv")
print(f"  ({len(_df)} total records across all runs)")
