"""
Component 12 — Horizontal weather variation (the finale).

Until now a MET message varied conditions by ALTITUDE only (Component 4): the
shell sampled the zone for its current height. Real weather also varies by
horizontal POSITION along the flight path, and — this is the honest part — you
cannot measure the weather at the target, because the target is in enemy
territory. So:

  * Near the gun you have REAL soundings: the gun profile is reliable.
  * Toward the target you have only a remote satellite/recon ESTIMATE: the
    target profile is a guess, not a measurement, and the uncertainty grows with
    distance from the gun.
  * The TRUE downrange weather differs from your told estimate, most of all near
    the target where you never measured it.

This module models the told (and, separately, the true) weather as conditions
that vary continuously with downrange position, interpolated between a
gun-location profile and a target-location profile. That is exactly how real
fire control handles MET between known locations (STANAG 4082 / FM 6-40): MET is
interpolated between reporting points. The novelty here is being explicit that
the target end is a remote estimate, and letting the fall-of-shot estimators
LEARN the discrepancy — the shell is the sensor we cannot place at the target.

It changes NO validated physics. The integrator already samples `met` once per
step; a HorizontalMetField simply answers that sample using BOTH the shell's
downrange distance and its altitude. When the gun and target profiles are equal
(no horizontal variation) it reproduces the altitude-only behaviour bit-for-bit.

numpy + standard library only.
"""

import numpy as np


class HorizontalMetField:
    """
    Weather that varies by downrange position AND altitude, built from a
    gun-location MET profile and a target-location MET profile. Conditions at any
    (downrange, altitude) are interpolated between the two profiles along the
    downrange axis, then sampled at the altitude. Models "reliable near the gun,
    a remote estimate toward the target".

    Reference: real fire control interpolates the computer MET message between
    known reporting locations (STANAG 4082 / FM 6-40). Here the target profile is
    explicitly a remote estimate (satellite/recon), NOT a measurement — there is
    no sensor in enemy territory.

    Interpolation is linear in the downrange fraction f = downrange / target_range
    (clamped to [0, 1]): f = 0 at the gun returns the gun profile exactly, f = 1
    at the target returns the target profile exactly. The interpolation is written
    as  a + f*(b - a)  so that when the two profiles are identical (b - a == 0)
    the result is the gun profile BIT-FOR-BIT, for any f — that is what makes the
    no-variation path reproduce the altitude-only model exactly.

    confidence_floor: the confidence in the weather at the target end (the remote
        estimate), on a 0..1 scale, with 1.0 at the gun (real soundings). It is a
        parameter, not hardcoded; default 0.2 represents a usable-but-uncertain
        remote estimate. Used only by weather_profile_along_path() for the
        honest "uncertainty grows toward the target" overlay — it does not affect
        the physics.
    """

    is_horizontal = True   # duck-type flag the integrator checks (see trajectory.py)

    def __init__(self, gun_met, target_met, target_range_m, confidence_floor=0.2):
        if target_range_m <= 0.0:
            raise ValueError("target_range_m must be positive")
        if not 0.0 <= confidence_floor <= 1.0:
            raise ValueError("confidence_floor must be in [0, 1]")
        self.gun_met = gun_met
        self.target_met = target_met
        self.target_range_m = float(target_range_m)
        self.confidence_floor = float(confidence_floor)

    def downrange_fraction(self, downrange_m):
        """Clamped downrange fraction f in [0, 1] (0 at gun, 1 at target)."""
        return min(1.0, max(0.0, float(downrange_m) / self.target_range_m))

    def sample_at(self, downrange_m, altitude_m):
        """
        Conditions at (downrange_m, altitude_m): the gun and target profiles are
        each sampled at the altitude (their existing altitude-zone logic), then
        interpolated by the downrange fraction. Returns the same dict shape as
        MetMessage.sample so the integrator consumes it identically:
            {wind_vector, temp_C, pressure_Pa, density_kgm3}
        """
        f = self.downrange_fraction(downrange_m)
        g = self.gun_met.sample(altitude_m)
        t = self.target_met.sample(altitude_m)
        # a + f*(b - a): exact reproduction of the gun profile when g == t.
        return {
            "wind_vector": g["wind_vector"] + f * (t["wind_vector"] - g["wind_vector"]),
            "temp_C":      g["temp_C"]      + f * (t["temp_C"]      - g["temp_C"]),
            "pressure_Pa": g["pressure_Pa"] + f * (t["pressure_Pa"] - g["pressure_Pa"]),
            "density_kgm3": g["density_kgm3"] + f * (t["density_kgm3"] - g["density_kgm3"]),
        }

    def sample(self, altitude_m):
        """Convenience MetMessage-compatible sample: at the gun (downrange 0) the
        field IS the gun profile. The integrator uses sample_at() with the live
        downrange distance; this exists only so a HorizontalMetField can stand in
        wherever a plain altitude-only sample is wanted."""
        return self.sample_at(0.0, altitude_m)

    def confidence_at(self, downrange_m):
        """Confidence in the weather estimate at this downrange, 1.0 at the gun
        (measured) decreasing linearly to confidence_floor at the target (remote
        estimate). Monotonically non-increasing with distance from the gun. This
        is the epistemic-uncertainty model for honest visualization, parameterised
        by confidence_floor — it is NOT a physical effect."""
        f = self.downrange_fraction(downrange_m)
        return 1.0 - (1.0 - self.confidence_floor) * f


def weather_profile_along_path(field, target_range_m=None, n_points=20,
                               altitude_m=0.0):
    """
    Sample the position-varying weather at n_points downrange positions from the
    gun to the target, as REAL computed data for honest visualization (wind that
    varies along the path + a confidence that decreases toward the target). This
    is not a weather cartoon — every value is the field's actual interpolated
    wind and the documented confidence model.

    Returns a list of n_points dicts (gun -> target), each:
        {downrange_m, wind_speed_ms, wind_dir_deg, confidence}

    wind_dir_deg is reported back in the standard meteorological FROM convention
    (the inverse of wind_vector_from_dir_speed). confidence is 1.0 at the gun and
    decreases monotonically to the field's confidence_floor at the target.

    altitude_m: the altitude at which to read the along-path wind (default 0 =
    surface, the natural ground-track overlay). n_points must be >= 2.
    """
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    R = field.target_range_m if target_range_m is None else float(target_range_m)

    out = []
    for i in range(n_points):
        dr = R * i / (n_points - 1)
        zone = field.sample_at(dr, altitude_m)
        wv = zone["wind_vector"]
        speed = float(np.hypot(wv[0], wv[1]))
        # Invert wind_vector_from_dir_speed: vec = speed*(-cos a, sin a).
        wdir = float(np.degrees(np.arctan2(wv[1], -wv[0])) % 360.0) if speed > 1e-12 else 0.0
        out.append({
            "downrange_m": float(dr),
            "wind_speed_ms": speed,
            "wind_dir_deg": wdir,
            "confidence": float(field.confidence_at(dr)),
        })
    return out
