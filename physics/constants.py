"""
ARCS — Autonomous Range Control System
Phase 1: Constants and Coordinate System

COORDINATE SYSTEM (Right-handed):
    Origin : Robot base center (0, 0, 0)
    X-axis : Forward (downrange)
    Y-axis : Upward  (against gravity)
    Z-axis : Lateral (right of robot)

ROTATION CONVENTION:
    Intrinsic Euler angles, ZYX order
    Yaw   (psi)   : Y-axis → turret left/right
    Pitch (theta) : Z-axis → barrel up/down
    Roll  (phi)   : X-axis → chassis tilt

UNITS:
    Distance  : meters
    Velocity  : m/s
    Angles    : degrees (external) | radians (internal)
    Time      : seconds
"""

import numpy as np

# ─── PHYSICAL CONSTANTS ───────────────────────────────
GRAVITY        = 9.81
GRAVITY_VECTOR = np.array([0.0, -GRAVITY, 0.0])

# ─── ROBOT MECHANICAL LIMITS ─────────────────────────
YAW_MIN_DEG   = -180.0
YAW_MAX_DEG   =  180.0
PITCH_MIN_DEG =    0.0
PITCH_MAX_DEG =   85.0
ROLL_MIN_DEG  =  -30.0
ROLL_MAX_DEG  =   30.0
V0_MIN        =   20.0   # m/s
V0_MAX        =  300.0   # m/s
RANGE_MIN     =   10.0   # m
RANGE_MAX     =  500.0   # m

# ─── NOISE MODEL ─────────────────────────────────────
# Mechanical precision limits — what the actuators
# actually achieve vs what they are commanded to do
SIGMA_PITCH_DEG = 0.3    # elevation servo std dev (degrees)
SIGMA_YAW_DEG   = 0.2    # bearing motor std dev   (degrees)
SIGMA_V0        = 1.5    # propulsion std dev       (m/s)

# Bias scale: robot bias = BIAS_SCALE × sigma. Max improvable CEP ≈ 44%.
BIAS_SCALE      = 1.5

# ─── ENGAGEMENT PROTOCOL CONSTANTS ───────────────────────────────
# Named so changes propagate everywhere without grep-and-edit.
N_SHOTS_BASELINE      = 30    # shots fired at zero correction to measure baseline CEP
N_SHOTS_VERIFY        = 30    # shots fired at best correction to verify improvement
N_KF_SHOTS            = 8     # directional shots fired for KF refinement after BO
FALLBACK_THRESHOLD    = 1.10  # verified_cep / baseline_cep above which correction is rejected
QUALITY_FILTER_RATIO  = 1.20  # miss_after / miss_before above which record is excluded from PINN

# GP convergence threshold — BO considered converged when posterior σ
# at best point drops below this fraction of the initial σ.
# Set to 0.3 × initial noise level (empirically robust).
GP_SIGMA_CONVERGED_THRESHOLD = 0.45   # metres — at this σ, GP is tight

# ─── RANGE TABLE SWEEP PARAMETERS ────────────────────
RT_RANGE_STEPS  = np.arange(10,  505,  5)    # 10–500m   every 5m
RT_HEIGHT_STEPS = np.arange(-20,  55,  5)    # -20–+50m  every 5m
RT_V0_STEPS     = np.arange(50,  305,  10)   # 50–300m/s every 10

# ─── UNIT HELPERS (used by downstream modules) ───────
def deg_to_rad(d): return np.deg2rad(d)
def rad_to_deg(r): return np.rad2deg(r)

# ─── TARGET VALIDATION ───────────────────────────────
def validate_target(x, y, z):
    """Returns (bool, reason_str)"""
    R = np.sqrt(x**2 + z**2)
    if R < RANGE_MIN:
        return False, f"Too close: {R:.1f}m < {RANGE_MIN}m"
    if R > RANGE_MAX:
        return False, f"Too far:   {R:.1f}m > {RANGE_MAX}m"
    if x <= 0 and abs(z) < 0.1:
        return False, "Target behind robot"
    return True, "Within engagement envelope"

# ─── SELF TEST ───────────────────────────────────────
if __name__ == "__main__":
    print("=" * 52)
    print("ARCS Phase 1 — constants.py self-test")
    print("=" * 52)
    print(f"  Gravity          : {GRAVITY} m/s²")
    print(f"  Gravity vector   : {GRAVITY_VECTOR}")
    print(f"  V0 range         : {V0_MIN}–{V0_MAX} m/s")
    print(f"  Range envelope   : {RANGE_MIN}–{RANGE_MAX} m")
    print(f"  Pitch limits     : {PITCH_MIN_DEG}°–{PITCH_MAX_DEG}°")
    print()

    assert abs(np.deg2rad(180) - np.pi) < 1e-9
    assert abs(np.rad2deg(np.pi) - 180) < 1e-9
    print("  Unit conversions : ✓")

    total = len(RT_RANGE_STEPS)*len(RT_HEIGHT_STEPS)*len(RT_V0_STEPS)
    print(f"  Range table size : {total:,} entries")
    print()

    cases = [(120,0,80,"valid"),(5,0,0,"close"),(600,0,0,"far"),(-50,0,0,"behind")]
    for x,y,z,label in cases:
        ok, msg = validate_target(x,y,z)
        print(f"  ({x:4d},{y:2d},{z:2d}) [{label:6s}] → {'✓' if ok else '✗'} {msg}")

    print("\n  constants.py ✓ all checks passed")