"""
ISA / ISO 2533 Standard Atmosphere model.

This is the bedrock atmosphere component for ARCS Phase 2. Given an altitude
above sea level it returns air temperature, pressure, and density using the
International Standard Atmosphere piecewise equations. The trajectory integrator
will call ``atmosphere`` at every step of a shell's flight to obtain the air
density used for drag.

Coverage: sea level (0 m) to 20,000 m, spanning the full troposphere and the
start of the isothermal lower stratosphere — sufficient for a 20-40 km howitzer
whose apex reaches up to ~12 km.

Standard library + numpy only.
"""

import numpy as np

# --- ISA standard constants (sea level datum) ---
T0 = 288.15            # K   (15.0 degrees C)
P0 = 101325.0          # Pa
RHO0 = 1.225           # kg/m^3
G0 = 9.80665           # m/s^2
R = 287.05287          # J/(kg.K)  specific gas constant for dry air
L = 0.0065             # K/m  troposphere lapse rate (temp falls 6.5 C per km)

# --- Layer boundaries ---
TROPOPAUSE_ALT = 11000.0   # m   top of troposphere / base of stratosphere
T_STRAT = 216.65           # K   constant temperature in lower stratosphere
MAX_ALT = 20000.0          # m   model validity ceiling

# Pressure at the tropopause (11,000 m) from the troposphere formula, using the
# ISA standard baseline. Computed once so the stratosphere layer is continuous
# with the troposphere layer at the boundary.
_P11_STD = P0 * ((T0 - L * TROPOPAUSE_ALT) / T0) ** (G0 / (R * L))


def atmosphere(altitude_m, sea_level_temp_C=15.0, sea_level_pressure_Pa=101325.0):
    """
    Returns air properties at a given altitude using the ISA model.

    altitude_m: height above sea level in metres (0 to 20000).
    Optional overrides let a non-standard (real-weather) day be modelled
    by shifting the sea-level baseline; the lapse structure is unchanged.
    Returns a dict: {temperature_K, temperature_C, pressure_Pa, density_kgm3}.
    Raises ValueError if altitude < 0 or > 20000.
    """
    if altitude_m < 0.0 or altitude_m > MAX_ALT:
        raise ValueError(
            f"altitude_m must be between 0 and {MAX_ALT:.0f} m, got {altitude_m}"
        )

    # Sea-level baseline. Defaults reproduce the exact ISA standard datum.
    t_sl = sea_level_temp_C + 273.15      # K
    p_sl = sea_level_pressure_Pa          # Pa

    # Pressure at the tropopause for this baseline (continuity anchor).
    t_trop = t_sl - L * TROPOPAUSE_ALT
    p11 = p_sl * (t_trop / t_sl) ** (G0 / (R * L))

    if altitude_m <= TROPOPAUSE_ALT:
        # Troposphere: temperature decreases linearly.
        temperature_K = t_sl - L * altitude_m
        pressure_Pa = p_sl * (temperature_K / t_sl) ** (G0 / (R * L))
    else:
        # Lower stratosphere: isothermal, exponential pressure decay.
        temperature_K = t_trop
        pressure_Pa = p11 * np.exp(-G0 * (altitude_m - TROPOPAUSE_ALT) / (R * t_trop))

    # Density always follows directly from the ideal gas law.
    density_kgm3 = pressure_Pa / (R * temperature_K)

    return {
        "temperature_K": float(temperature_K),
        "temperature_C": float(temperature_K - 273.15),
        "pressure_Pa": float(pressure_Pa),
        "density_kgm3": float(density_kgm3),
    }
