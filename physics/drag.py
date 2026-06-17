"""
Component 5 — Mach-dependent drag via the published G7 standard drag model.

A projectile's drag coefficient is NOT constant: it changes sharply with Mach
number (speed / local speed of sound), rising steeply through the transonic
region and peaking near Mach 1, then falling at higher Mach. A standard 155mm
M107 leaves the muzzle near Mach 2.4 and decelerates down through Mach 1 during
flight, so a single fixed Cd cannot capture the real drag.

The G7 standard is the published US-standard drag shape for long boat-tailed
projectiles such as the 155mm M107.
Reference: McCoy, "Modern Exterior Ballistics" (G7 standard projectile).

Standard library + numpy only.
"""

import numpy as np

# G7 standard drag function: Mach number -> drag coefficient Cd.
# Source: McCoy, Modern Exterior Ballistics (G7 standard projectile).
# Note the sharp peak at Mach 1 — that is the physically correct transonic
# drag rise. Use these published values as given; do not invent drag values.
G7_MACH = [0.00, 0.50, 0.70, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20,
           1.35, 1.50, 1.75, 2.00, 2.25, 2.50, 3.00, 3.50, 4.00]
G7_CD   = [0.120, 0.119, 0.119, 0.121, 0.128, 0.140, 0.300, 0.339, 0.338, 0.320,
           0.295, 0.275, 0.250, 0.231, 0.215, 0.200, 0.178, 0.162, 0.150]

GAMMA = 1.4              # ratio of specific heats for dry air
R_AIR = 287.05287        # specific gas constant for dry air, J/(kg.K) (frozen)
SPEED_OF_SOUND_SEA_LEVEL = 340.29  # m/s at 15 C (reference; computed locally)


def speed_of_sound(temperature_K):
    """
    Local speed of sound a = sqrt(gamma * R * T), gamma=1.4, R=287.05287.
    Speed of sound falls with altitude as air gets colder — this matters: the
    Mach number must be computed against the LOCAL temperature, not a fixed
    sea-level value, or the transonic drag rise is placed at the wrong speed.
    """
    return float(np.sqrt(GAMMA * R_AIR * temperature_K))


def g7_cd(mach):
    """
    Linear interpolation into the G7 table. Clamps below Mach 0 and above
    Mach 4.0 (np.interp returns the table end values outside the range).
    """
    return float(np.interp(mach, G7_MACH, G7_CD))


def drag_coefficient(speed, temperature_K, bc_scale=1.0):
    """
    speed: shell speed (m/s). temperature_K: local air temp (for speed of sound).
    Returns the effective Cd for this speed at this altitude's temperature.
    bc_scale: a single tuning multiplier (ballistic-coefficient form factor)
      so the model can be matched to a specific real shell. Default 1.0.
      Higher bc_scale -> more drag -> shorter range.
    """
    mach = speed / speed_of_sound(temperature_K)
    return g7_cd(mach) * bc_scale
