"""
ARCS — Kalman Filter Module
Phase 1 ready  /  Phase 2 foundation

WHAT A KALMAN FILTER DOES — IN PLAIN ENGLISH
─────────────────────────────────────────────
Imagine you're trying to estimate the true position of a target, but your
sensor gives noisy readings. Two sources of information are available:

  1. Physics model:   "Based on how targets usually move, I PREDICT it's here."
  2. Sensor reading:  "My sensor MEASURED it at this position (with some noise)."

The Kalman Filter combines both optimally:
  - If the sensor is very accurate → trust the measurement more
  - If the sensor is very noisy   → trust the physics model more
  - The ratio is computed mathematically using the covariance matrices

Result: the best possible estimate of the true state from all available data.

This is what GPS uses. It's what Apollo used. It's been the gold standard
for state estimation since 1960 (Kalman, R.E., 1960).

═══════════════════════════════════════════════════════════════════════════════
PHASE 1 — EngagementKF (active now)
═══════════════════════════════════════════════════════════════════════════════

PURPOSE: Better shot averaging within each engagement.

The Bayesian Optimizer fires N shots and takes the median miss distance.
The KF is smarter: it maintains a PROBABILISTIC ESTIMATE of the systematic
correction needed, updating after each shot.

WHY THIS IS BETTER THAN MEDIAN:
  • Weights recent shots more if systematic bias is drifting
  • Provides uncertainty estimates (we know HOW CONFIDENT we are)
  • Propagates information across shots rather than treating each independently

STATE: [Δpitch_needed, Δyaw_needed]  — the correction the robot needs right now
MEASUREMENT: directional error (error_x, error_z) observed after each shot,
             ROTATED into the barrel-aligned frame via target yaw angle θ.

MEASUREMENT MODEL (bearing-rotation-corrected):
  err_along = cos(θ)·error_x + sin(θ)·error_z   (range direction → Δpitch)
  err_lat   = −sin(θ)·error_x + cos(θ)·error_z  (lateral direction → Δyaw)

  err_along ≈ range_m · Δpitch_rad  →  Δpitch_deg = −err_along / range_m · (180/π)
  err_lat   ≈ range_m · Δyaw_rad    →  Δyaw_deg   = −err_lat   / range_m · (180/π)

Previously (bug): error_x→Δpitch, error_z→Δyaw — only correct at θ=0° (target straight ahead).
At θ=90°, the axes are completely swapped and corrections point in the wrong direction.

═══════════════════════════════════════════════════════════════════════════════
PHASE 2 — TargetTrackingKF (foundation, activate when Phase 2 starts)
═══════════════════════════════════════════════════════════════════════════════

PURPOSE: Predict where a MOVING target will be when the projectile arrives.

A static target is Phase 1. Phase 2 involves targets that move.
The projectile takes 2-5 seconds to reach the target.
We must aim at where the target WILL BE, not where it IS NOW.

STATE:  [x, y, z, vx, vy, vz]  — 6D: position + velocity
PROCESS: constant velocity model  →  x_{t+1} = x_t + vx_t · Δt
MEASUREMENT: noisy position from radar/sensor  →  z = [x_obs, y_obs, z_obs]

WORKFLOW:
  1. Observe target position (noisy, from sensor)
  2. KF update: refine estimate of [x, y, z, vx, vy, vz]
  3. KF predict: propagate state forward by TOF (time-of-flight)
  4. Fire at predicted position

The predicted position at TOF is the aim point.
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Engagement Kalman Filter
# ═══════════════════════════════════════════════════════════════════════════════

class EngagementKF:
    """
    Linear Kalman Filter for optimal shot averaging within one engagement.

    Estimates the systematic correction [Δpitch, Δyaw] from shot observations.

    EQUATIONS (standard KF, see Kalman 1960):
    ─────────────────────────────────────────
    State: x = [Δpitch_deg, Δyaw_deg]

    Predict step:
        x_pred = F · x              (F = I, bias is approximately constant)
        P_pred = F · P · Fᵀ + Q    (Q = process noise, small)

    Update step (after each shot):
        y  = z − H · x_pred        (innovation: observed − predicted)
        S  = H · P_pred · Hᵀ + R   (innovation covariance)
        K  = P_pred · Hᵀ · S⁻¹    (Kalman gain)
        x  = x_pred + K · y        (updated state)
        P  = (I − K · H) · P_pred  (updated covariance)

    Where:
        z = [error_x / range_m · (180/π),  error_z / range_m · (180/π)]
            (observed errors converted to approximate angle corrections)
        H = I₂ (identity — the state IS the measurement in this linear model)
        R = diag(σ_pitch², σ_yaw²)  (measurement noise from motor sigmas)

    UNCERTAINTY:
        P (2×2 covariance matrix) tracks how certain we are about the correction.
        After many shots: P → 0 (high confidence).
        After 1 shot: P is large (low confidence, dominated by prior).

    USAGE (in Bayesian Optimizer or Pipeline):
    ──────────────────────────────────────────
        kf = EngagementKF(range_m=200, sigma_pitch=0.3, sigma_yaw=0.2)
        kf.update(error_x=-3.5, error_z=1.2)  # after shot 1
        kf.update(error_x=-3.1, error_z=0.9)  # after shot 2
        dp, dy = kf.correction_deg             # best estimate of correction needed
        conf   = kf.confidence                 # scalar in [0,1], higher = more certain
    """

    def __init__(self,
                 range_m:      float = 200.0,
                 sigma_pitch:  float = 0.3,    # motor noise (degrees)
                 sigma_yaw:    float = 0.2,    # motor noise (degrees)
                 process_noise: float = 0.01): # how fast bias drifts between shots
        """
        Args:
            range_m:       Horizontal distance to target (metres).
                           Used to convert miss distance → angle correction.
            sigma_pitch:   Pitch motor noise std dev (degrees).
            sigma_yaw:     Yaw motor noise std dev (degrees).
            process_noise: Q diagonal — how much the systematic bias can shift
                           between individual shots. Small = assumes bias is stable.
        """
        self.range_m = max(range_m, 1.0)

        # ── Initial state: no correction known yet ─────────────────────
        self.x = np.zeros(2)                         # [Δpitch_deg, Δyaw_deg]
        self.P = np.eye(2) * (2.0 * sigma_pitch)**2  # initial uncertainty (large)

        # ── System matrices ───────────────────────────────────────────
        self.F = np.eye(2)           # process: bias stays constant
        self.H = np.eye(2)           # measurement: observe the correction directly
        self.Q = np.eye(2) * process_noise            # process noise
        self.R = np.diag([sigma_pitch**2, sigma_yaw**2])  # measurement noise

        # ── Diagnostics ───────────────────────────────────────────────
        self.n_updates   = 0
        self._innovations = []   # track how much each shot changes the estimate

    def update(self, error_x: float, error_z: float,
               yaw_deg: float = 0.0) -> None:
        """
        Incorporate a new shot observation.

        Args:
            error_x: ARCS X-axis miss (metres, positive = forward of target).
            error_z: ARCS Z-axis miss (metres, positive = right of target).
            yaw_deg: Turret yaw angle to the target (degrees). REQUIRED for
                     non-zero bearing targets. Without this, the KF maps world-
                     frame errors to angle corrections using the wrong axes.

        COORDINATE ROTATION (the structural fix):
        ─────────────────────────────────────────
        The ARCS frame has X=forward, Z=lateral. For a target at yaw angle θ:

            Barrel direction unit vector: (cos θ, 0, sin θ)

        The observed (error_x, error_z) are in the WORLD frame. They must be
        rotated into the BARREL frame before mapping to angle corrections:

            err_along = cos(θ)·error_x + sin(θ)·error_z   ← range error → Δpitch
            err_lat   = −sin(θ)·error_x + cos(θ)·error_z  ← lateral error → Δyaw

        Without this rotation (the original bug):
          • At θ=0°:  works correctly (forward=along-barrel, lateral=cross-barrel)
          • At θ=90°: completely wrong — error_x is lateral, error_z is range,
                      but the KF treats them as pitch and yaw respectively.
                      The corrections point in the opposite physical direction,
                      causing shots to diverge rather than converge.

        HOW BARREL-ALIGNED ERRORS MAP TO CORRECTIONS:
            err_along ≈ range_m · Δpitch_rad  →  Δpitch_deg = −err_along/range · (180/π)
            err_lat   ≈ range_m · Δyaw_rad    →  Δyaw_deg   = −err_lat/range · (180/π)
        """
        # ── Rotate world-frame errors into barrel-aligned frame ───────────
        yaw_rad = np.deg2rad(yaw_deg)
        cos_t   = np.cos(yaw_rad)
        sin_t   = np.sin(yaw_rad)

        err_along = cos_t * error_x + sin_t * error_z   # range direction → Δpitch
        err_lat   = -sin_t * error_x + cos_t * error_z  # lateral direction → Δyaw

        # ── Convert barrel-aligned errors to required angle corrections ───
        rad_per_m = 1.0 / self.range_m
        deg_per_m = rad_per_m * (180.0 / np.pi)
        z = np.array([
            -err_along * deg_per_m,   # Δpitch needed (negate: overshoot → reduce pitch)
            -err_lat   * deg_per_m,   # Δyaw needed
        ])

        # ── Predict ───────────────────────────────────────────────────────
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # ── Update ────────────────────────────────────────────────────────
        innovation = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        self.x = x_pred + K @ innovation
        self.P = (np.eye(2) - K @ self.H) @ P_pred

        self.n_updates += 1
        self._innovations.append(float(np.linalg.norm(innovation)))

    def update_scalar(self, miss_dist: float) -> None:
        """
        Simplified update using only the scalar miss distance (no direction).
        Used when error_x / error_z are not available.

        Assumes the miss is equally likely in both directions.
        Less informative than update() — prefer update() when available.
        """
        # Distribute miss equally as a prior push toward the observed magnitude
        deg_correction = miss_dist / self.range_m * (180.0 / np.pi) * 0.5
        z = np.array([deg_correction, 0.0])   # pitch dominant guess
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R * 4.0   # higher uncertainty
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(2) - K @ self.H) @ self.P
        self.n_updates += 1

    @property
    def correction_deg(self) -> tuple:
        """Best current estimate of (Δpitch_deg, Δyaw_deg)."""
        return float(self.x[0]), float(self.x[1])

    @property
    def uncertainty_deg(self) -> tuple:
        """1-sigma uncertainty on (Δpitch_deg, Δyaw_deg)."""
        return float(np.sqrt(self.P[0, 0])), float(np.sqrt(self.P[1, 1]))

    @property
    def confidence(self) -> float:
        """
        Scalar confidence in [0, 1].
        0 = no information (prior only), 1 = very high confidence.
        Computed from the trace of P relative to the initial uncertainty.
        """
        P_trace_init = (2.0 * 0.3)**2 * 2       # approx initial trace
        P_trace_now  = float(np.trace(self.P))
        return float(np.clip(1.0 - P_trace_now / (P_trace_init + 1e-9), 0.0, 1.0))

    def summary(self) -> str:
        dp, dy = self.correction_deg
        sp, sy = self.uncertainty_deg
        return (f"KF(shots={self.n_updates}  "
                f"Δpitch={dp:+.3f}±{sp:.3f}°  "
                f"Δyaw={dy:+.3f}±{sy:.3f}°  "
                f"conf={self.confidence:.2f})")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 FOUNDATION — Target Tracking KF (uncomment to activate in Phase 2)
# ═══════════════════════════════════════════════════════════════════════════════

class TargetTrackingKF:
    """
    Linear Kalman Filter for MOVING target position estimation.
    Phase 2 — activate when working with non-static targets.

    STATE:    [x, y, z, vx, vy, vz]  — 6D: position (m) + velocity (m/s)
    PROCESS:  constant velocity model:  x_{k+1} = x_k + v_k · Δt
    MEASUREMENT: noisy position from sensor:  z = [x_obs, y_obs, z_obs]

    WORKFLOW:
        # Initialise once
        tracker = TargetTrackingKF(sensor_noise_m=5.0, dt=0.1)

        # At each sensor tick (every ~0.1s)
        tracker.predict(dt=0.1)                  # propagate physics forward
        tracker.update(x_obs, y_obs, z_obs)      # fuse sensor measurement

        # When ready to fire:
        tof = compute_tof(...)                   # from ballistic solver
        px, py, pz = tracker.predict_at(tof)    # where will target be?
        # Aim at (px, py, pz)

    CONSTANT VELOCITY MODEL:
        x_{k+1} = x_k + vx_k · Δt + noise
        vx_{k+1} = vx_k + noise

    This works well for targets moving at roughly constant velocity over
    the time scales relevant to ballistic engagement (a few seconds).
    For accelerating targets, an EKF with a higher-order model is needed
    (see Phase 2 extension notes at the bottom of this file).
    """

    def __init__(self,
                 sensor_noise_m:   float = 5.0,    # sensor position noise (m)
                 process_noise_pos: float = 0.5,   # position process noise (m)
                 process_noise_vel: float = 1.0,   # velocity process noise (m/s)
                 dt:               float = 0.1):   # default prediction step (s)
        """
        Args:
            sensor_noise_m:    Std dev of sensor position measurement (metres).
                               Radar: ~2-5m. Optical: ~0.5-2m.
            process_noise_pos: How much position can change unpredictably per step.
            process_noise_vel: How much velocity can change (acceleration noise).
            dt:                Default prediction timestep (seconds).
        """
        self.dt = dt

        # ── State: [x, y, z, vx, vy, vz] ─────────────────────────────
        self.x = np.zeros(6)             # position + velocity estimate
        self.P = np.eye(6) * 1000.0     # large initial uncertainty

        # ── Process model: constant velocity ──────────────────────────
        # x_{k+1} = F · x_k + noise
        # F = [[I₃  Δt·I₃],
        #      [0₃   I₃  ]]
        self.F = self._build_F(dt)

        # Process noise matrix Q
        # Accounts for unknown accelerations (target manoeuvring)
        q_p = process_noise_pos**2
        q_v = process_noise_vel**2
        self.Q = np.diag([q_p, q_p, q_p, q_v, q_v, q_v])

        # ── Measurement model: observe position only ───────────────────
        # z = H · x  →  z = [x, y, z]  (not velocity)
        self.H = np.zeros((3, 6))
        self.H[:3, :3] = np.eye(3)

        # Measurement noise R
        self.R = np.eye(3) * sensor_noise_m**2

        # ── State ──────────────────────────────────────────────────────
        self.is_initialised = False
        self.n_updates      = 0

    @staticmethod
    def _build_F(dt: float) -> np.ndarray:
        """Constant-velocity state transition matrix for timestep dt."""
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt    # position += velocity · dt
        return F

    def initialise(self, x: float, y: float, z: float,
                   vx: float = 0.0, vy: float = 0.0, vz: float = 0.0) -> None:
        """
        Initialise state from a known (or guessed) position and velocity.
        Call this before the first predict/update cycle.
        """
        self.x = np.array([x, y, z, vx, vy, vz], dtype=float)
        self.P = np.diag([25.0, 25.0, 25.0,   # position uncertainty: ±5m
                          16.0, 16.0, 16.0])   # velocity uncertainty: ±4m/s
        self.is_initialised = True

    def predict(self, dt: float | None = None) -> None:
        """
        Propagate the state estimate forward by dt seconds.
        Call this at each time step BEFORE update().

        Physics model: position += velocity · dt  (constant velocity)
        """
        if not self.is_initialised:
            return
        F = self._build_F(dt if dt is not None else self.dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, x_obs: float, y_obs: float, z_obs: float) -> None:
        """
        Incorporate a noisy sensor position measurement.
        Call this AFTER predict() at each sensor tick.

        z = [x_obs, y_obs, z_obs]  (metres, in world frame)
        """
        if not self.is_initialised:
            self.initialise(x_obs, y_obs, z_obs)
            return

        z          = np.array([x_obs, y_obs, z_obs])
        innovation = z - self.H @ self.x
        S          = self.H @ self.P @ self.H.T + self.R
        K          = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ innovation
        self.P = (np.eye(6) - K @ self.H) @ self.P
        self.n_updates += 1

    def predict_at(self, t_ahead: float) -> tuple:
        """
        Predict where the target will be in t_ahead seconds.

        Args:
            t_ahead: Time until projectile impact (seconds), from ballistic solver.

        Returns:
            (px, py, pz): Predicted target position at impact time (metres).

        This is the AIM POINT — fire here, not at the current position.
        """
        F_ahead = self._build_F(t_ahead)
        x_pred  = F_ahead @ self.x
        return float(x_pred[0]), float(x_pred[1]), float(x_pred[2])

    @property
    def position(self) -> tuple:
        """Current estimated position (x, y, z) in metres."""
        return float(self.x[0]), float(self.x[1]), float(self.x[2])

    @property
    def velocity(self) -> tuple:
        """Current estimated velocity (vx, vy, vz) in m/s."""
        return float(self.x[3]), float(self.x[4]), float(self.x[5])

    @property
    def position_uncertainty_m(self) -> float:
        """1-sigma position uncertainty (metres), averaged over x, y, z."""
        return float(np.sqrt(np.trace(self.P[:3, :3]) / 3.0))

    def summary(self) -> str:
        px, py, pz = self.position
        vx, vy, vz = self.velocity
        return (f"TargetKF(updates={self.n_updates}  "
                f"pos=({px:.1f},{py:.1f},{pz:.1f})m  "
                f"vel=({vx:.1f},{vy:.1f},{vz:.1f})m/s  "
                f"unc=±{self.position_uncertainty_m:.1f}m)")


# ─── Self test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("ARCS — kalman_filter.py  self-test")
    print("=" * 60)

    # ── Phase 1: EngagementKF ──────────────────────────────────────
    print("\n[Phase 1] EngagementKF — shot averaging")
    print("  Simulated scenario: robot has +0.4° pitch bias, +0.2° yaw bias")
    print("  Each shot has ±0.3° random noise on pitch, ±0.2° on yaw")

    rng        = np.random.default_rng(42)
    range_m    = 200.0
    true_bias_p = 0.4    # degrees
    true_bias_y = 0.2    # degrees
    sigma_p    = 0.3
    sigma_y    = 0.2

    kf = EngagementKF(range_m=range_m, sigma_pitch=sigma_p, sigma_yaw=sigma_y)

    print(f"\n  {'Shot':>4}  {'error_x':>9}  {'error_z':>9}  "
          f"{'est_Δpitch':>11}  {'est_Δyaw':>9}  {'conf':>6}")
    print(f"  {'─'*4}  {'─'*9}  {'─'*9}  {'─'*11}  {'─'*9}  {'─'*6}")

    for shot in range(8):
        # Simulate: robot fires with bias, we observe error
        actual_p_error_deg = true_bias_p + rng.normal(0, sigma_p)
        actual_y_error_deg = true_bias_y + rng.normal(0, sigma_y)

        # Convert angle errors to miss distances at range_m
        error_x = range_m * np.tan(np.deg2rad(actual_p_error_deg))
        error_z = range_m * np.tan(np.deg2rad(actual_y_error_deg))

        kf.update(error_x, error_z)
        dp, dy = kf.correction_deg
        print(f"  {shot+1:>4}  {error_x:>+8.2f}m  {error_z:>+8.2f}m  "
              f"{dp:>+10.3f}°  {dy:>+8.3f}°  {kf.confidence:>5.2f}")

    dp_final, dy_final = kf.correction_deg
    sp_final, sy_final = kf.uncertainty_deg
    print(f"\n  True bias:       Δpitch={true_bias_p:+.3f}°   Δyaw={true_bias_y:+.3f}°")
    print(f"  KF estimate:     Δpitch={dp_final:+.3f}°   Δyaw={dy_final:+.3f}°")
    print(f"  Uncertainty:     ±{sp_final:.3f}°  ±{sy_final:.3f}°")
    print(f"  {kf.summary()}")

    # ── Phase 2: TargetTrackingKF ──────────────────────────────────
    print("\n[Phase 2] TargetTrackingKF — moving target prediction")
    print("  Simulated scenario: target at (150, 0, 50)m moving at vx=8m/s")

    tracker = TargetTrackingKF(sensor_noise_m=3.0, dt=0.1)
    tracker.initialise(150, 0, 50, vx=8, vy=0, vz=0)

    # Simulate 5 sensor observations (every 0.1s) with noise
    print(f"\n  {'Tick':>4}  {'x_obs':>8}  {'x_est':>8}  {'vx_est':>8}  {'unc':>6}")
    true_x = 150.0
    for tick in range(5):
        true_x += 8.0 * 0.1        # target moves 0.8m per tick
        x_obs   = true_x + rng.normal(0, 3.0)   # noisy sensor
        tracker.predict(dt=0.1)
        tracker.update(x_obs, 0, 50)
        px, _, _ = tracker.position
        vx, _, _ = tracker.velocity
        print(f"  {tick+1:>4}  {x_obs:>7.2f}m  {px:>7.2f}m  {vx:>7.2f}m/s  "
              f"±{tracker.position_uncertainty_m:.2f}m")

    # Predict aim point 2 seconds ahead (typical TOF)
    tof = 2.0
    ax, ay, az = tracker.predict_at(tof)
    print(f"\n  TOF = {tof}s — aim at ({ax:.1f}, {ay:.1f}, {az:.1f})m")
    print(f"  Target will be at ≈ ({true_x + 8*tof:.1f}, 0, 50)m  (ground truth)")
    print(f"\n  {tracker.summary()}")

    print("\n  kalman_filter.py ✓")


# ─── Phase 2 Extension Notes ─────────────────────────────────────────────────
#
# When Phase 2 begins, extend TargetTrackingKF with:
#
# 1. EXTENDED KALMAN FILTER (for nonlinear ballistic measurement):
#    Instead of measuring position directly (H = [I|0]),
#    measure bearing angle + range from the robot's sensors.
#    z = [azimuth_rad, elevation_rad, range_m]
#    h(x) = [atan2(z,x), atan2(y, sqrt(x²+z²)), sqrt(x²+y²+z²)]
#    → Jacobian Hk = ∂h/∂x evaluated at current estimate
#
# 2. IMM FILTER (for manoeuvring targets):
#    Multiple motion models in parallel (constant velocity + constant turn).
#    Each model has its own KF. The IMM weights them by likelihood.
#    Best for targets that can suddenly change direction.
#
# 3. UNSCENTED KF (UKF):
#    Better than EKF for highly nonlinear systems.
#    Uses sigma points instead of Jacobians.
#    See: Julier & Uhlmann (1997).
#
# Reference: Thrun, Burgard, Fox — "Probabilistic Robotics" (2005), Chapter 3-4.