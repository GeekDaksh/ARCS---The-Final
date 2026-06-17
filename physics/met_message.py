"""
Component 4 — Standard artillery meteorological message ("MET message").

A real artillery met message reports atmospheric conditions in standard
altitude zones, from the surface up past the shell's maximum ordinate. Each
zone line gives wind direction, wind speed, air temperature, and pressure for
that layer. During flight the fire-control computer samples the zone the shell
is currently passing through — this is what makes "clear at the gun, raining at
the target" real: the message describes the whole air column and the shell
experiences each layer in turn.

Zone structure: STANAG 4082 / FM 6-40 Ch. 11 computer met message. The standard
zones are NOT evenly spaced — they are denser near the ground (where conditions
change fastest) and sparser aloft.

This module feeds the already-validated wind mechanism (Component 3, relative
airspeed drag) and the ISA density relation (Component 1). Standard library +
numpy only.
"""

import numpy as np

# Specific gas constant for dry air — same value as Component 1 (frozen).
R = 287.05287  # J/(kg.K)

# Standard met-line zone TOP altitudes (m), STANAG 4082 / FM 6-40 Ch. 11.
# Line 0 = surface. Each line covers from the previous line's top up to its own
# top. (The optional 3500 m "line 7" is omitted; the major standard boundaries
# below are sufficient to cover a 20-40 km howitzer's maximum ordinate.)
STANDARD_ZONE_TOPS = [
    0,      # 0  surface
    200,    # 1
    500,    # 2
    1000,   # 3
    1500,   # 4
    2000,   # 5
    3000,   # 6
    4000,   # 8
    5000,   # 9
    6000,   # 10
    8000,   # 11
    10000,  # 12
    12000,  # 13
    14000,  # 14
    16000,  # 15
]


def wind_vector_from_dir_speed(wind_dir_deg, wind_speed_ms):
    """
    Convert a meteorological (direction, speed) into a trajectory-frame velocity
    vector (wx, wy, wz) with wz = 0.

    CONVENTION (document carefully — a sign error here is the classic met bug):
    Meteorological wind direction is the direction the wind blows *FROM*,
    measured CLOCKWISE from downrange +x. The air therefore moves TOWARD the
    direction (dir + 180 deg). Working it out:

        wind_vector = speed * (-cos(dir), sin(dir), 0)

    Sanity check of the mapping (used in the tests):
        from 180 deg -> ( +speed, 0, 0)  = TAIL wind  (blows toward +x)
        from   0 deg -> ( -speed, 0, 0)  = HEAD wind  (blows toward -x)
        from  90 deg -> ( 0, +speed, 0)  = crosswind toward +y
        from 270 deg -> ( 0, -speed, 0)  = crosswind toward -y
    (Clockwise FROM-direction: +x rotates toward -y, so "from 270" pushes -y.)
    """
    a = np.radians(wind_dir_deg)
    return np.array([-wind_speed_ms * np.cos(a),
                     wind_speed_ms * np.sin(a),
                     0.0])


class MetMessage:
    """
    A standard artillery meteorological message: per-zone wind, temperature,
    and pressure, indexed by the standard met-line altitude structure.
    Reference: STANAG 4082 / FM 6-40 computer met message.
    """

    def __init__(self, lines):
        # lines: list of dicts, one per zone, each:
        #   {zone_top_m, wind_dir_deg, wind_speed_ms, temp_C, pressure_Pa}
        # wind_dir_deg: meteorological convention — direction wind blows FROM,
        #   measured clockwise from downrange +x (see wind_vector_from_dir_speed).
        if not lines:
            raise ValueError("MetMessage requires at least one zone line")

        # Sort zones by top altitude ascending so sampling can scan upward.
        self.lines = sorted((dict(ln) for ln in lines),
                            key=lambda ln: ln["zone_top_m"])

        # Precompute the air-density and wind vector for each zone once.
        for ln in self.lines:
            temp_K = ln["temp_C"] + 273.15
            ln["density_kgm3"] = ln["pressure_Pa"] / (R * temp_K)
            ln["wind_vector"] = wind_vector_from_dir_speed(
                ln["wind_dir_deg"], ln["wind_speed_ms"])

        # Count how often a sample falls above the highest zone (apex not
        # covered by the message). The fire mission should report enough zones
        # to cover the maximum ordinate; this flags when it does not.
        self.clamp_count = 0

    @property
    def max_zone_top(self):
        return self.lines[-1]["zone_top_m"]

    def covers(self, altitude_m):
        """True if altitude_m falls within the message's zone coverage."""
        return altitude_m <= self.max_zone_top

    def sample(self, altitude_m):
        """
        Return the conditions for the zone containing altitude_m:
            {wind_vector (wx,wy,wz), temp_C, pressure_Pa, density_kgm3}

        Boundary rule (documented and tested): each zone covers
        (previous_top, this_top], i.e. an altitude exactly equal to a zone top
        belongs to THAT zone (the lower one). The zone for altitude h is the
        first zone whose top is >= h.

        Above the highest zone the value clamps to the top zone and increments
        clamp_count (the message did not cover the maximum ordinate).
        """
        for ln in self.lines:
            if altitude_m <= ln["zone_top_m"]:
                return {
                    "wind_vector": ln["wind_vector"],
                    "temp_C": ln["temp_C"],
                    "pressure_Pa": ln["pressure_Pa"],
                    "density_kgm3": ln["density_kgm3"],
                }
        # Above coverage: clamp to the top zone and flag it.
        self.clamp_count += 1
        top = self.lines[-1]
        return {
            "wind_vector": top["wind_vector"],
            "temp_C": top["temp_C"],
            "pressure_Pa": top["pressure_Pa"],
            "density_kgm3": top["density_kgm3"],
        }

    @classmethod
    def standard_isa(cls, max_alt_m=16000, surface_wind=(0.0, 0.0)):
        """
        Build a MET message that reproduces the ISA atmosphere (Component 1)
        per zone, with an optional uniform wind applied to every zone.

        surface_wind = (wind_dir_deg, wind_speed_ms) in the standard FROM
        convention. The same wind is applied to all zones, so the layered model
        reduces to a single constant wind — the bridge for validating against
        Components 2 and 3.

        Each zone's temperature and pressure are sampled from the continuous ISA
        model at the zone MIDPOINT (the representative altitude of the layer),
        which keeps the zone-discretisation error small.
        """
        from physics.atmosphere import atmosphere

        wind_dir, wind_speed = surface_wind
        tops = [t for t in STANDARD_ZONE_TOPS if t <= max_alt_m]
        if tops[-1] < max_alt_m:
            tops.append(max_alt_m)

        lines = []
        prev_top = 0.0
        for top in tops:
            # Representative altitude of the zone: its midpoint (surface = 0).
            mid = 0.0 if top == 0 else 0.5 * (prev_top + top)
            air = atmosphere(mid)
            lines.append({
                "zone_top_m": float(top),
                "wind_dir_deg": float(wind_dir),
                "wind_speed_ms": float(wind_speed),
                "temp_C": air["temperature_C"],
                "pressure_Pa": air["pressure_Pa"],
            })
            prev_top = top
        return cls(lines)
