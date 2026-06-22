"""
Validation suite for Component 11 — the Coriolis effect (Earth's rotation).

The Earth turns under the shell during its flight, so a long-range round lands a
DETERMINISTIC, predictable offset from where it would on a non-rotating Earth
(not a random error). External ground truth: the deflection is to the right in
the Northern hemisphere, flips in the Southern, grows with time of flight,
depends on firing azimuth, vanishes (cross-range) at the equator, and is zero
when disabled. Reference: McCoy, "Modern Exterior Ballistics" (artillery
Coriolis), a = -2*Omega x v. Frame: local ENU (x=East, y=North, z=Up).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.trajectory import integrate_trajectory

V0 = 827.0


def _perp_right(az_deg):
    """Unit vector to the RIGHT of the firing line in the (East, North) plane."""
    a = np.radians(az_deg)
    return np.array([np.sin(a), -np.cos(a)])   # right of (cos a, sin a) is (sin a, -cos a)


def cor_right(elev, az=0.0, lat=20.0, v0=V0):
    """Rightward Coriolis deflection (m): the impact shift vs the coriolis-off
    trajectory, projected onto the right-of-firing direction."""
    on = integrate_trajectory(v0=v0, elevation_deg=elev, azimuth_deg=az, latitude_deg=lat)
    off = integrate_trajectory(v0=v0, elevation_deg=elev, azimuth_deg=az, coriolis=False)
    d = np.array([on["impact_x"] - off["impact_x"], on["impact_y"] - off["impact_y"]])
    return float(d @ _perp_right(az))


# ---------------------------------------------------------------------------
# Layer 1 — Direction & dependence (core proof)
# ---------------------------------------------------------------------------
def test_layer1_northern_hemisphere_deflects_right():
    # Standard textbook result: NH projectile is deflected to the right of fire.
    assert cor_right(45.0, az=0.0, lat=20.0) > 0.0


def test_layer1_southern_hemisphere_deflects_left():
    assert cor_right(45.0, az=0.0, lat=-20.0) < 0.0


def test_layer1_grows_with_time_of_flight():
    short = cor_right(30.0)   # ~60 s tof
    long = cor_right(55.0)    # ~97 s tof
    assert long > short > 0.0, (short, long)


# ---------------------------------------------------------------------------
# Layer 2 — Reproduction / boundary
# ---------------------------------------------------------------------------
def test_layer2_disabled_reproduces_nonrotating_bitforbit():
    a = integrate_trajectory(v0=V0, elevation_deg=45.0, coriolis=False)
    # latitude is irrelevant when off -> identical result
    b = integrate_trajectory(v0=V0, elevation_deg=45.0, coriolis=False, latitude_deg=70.0)
    for k in a:
        assert a[k] == b[k], k
    # and turning it ON actually changes the impact (the effect is present)
    c = integrate_trajectory(v0=V0, elevation_deg=45.0, coriolis=True, latitude_deg=45.0)
    assert (c["impact_x"], c["impact_y"]) != (a["impact_x"], a["impact_y"])


def test_layer2_vacuum_has_no_coriolis():
    on = integrate_trajectory(v0=V0, elevation_deg=45.0, vacuum=True, coriolis=True)
    off = integrate_trajectory(v0=V0, elevation_deg=45.0, vacuum=True, coriolis=False)
    for k in on:
        assert on[k] == off[k], k


def test_layer2_equator_cross_deflection_vanishes():
    # At the equator sin(latitude) = 0, so the vertical-rotation component that
    # drives the cross-range (right) deflection vanishes. For an East shot the
    # rightward deflection collapses to ~0 (a non-zero latitude does deflect it).
    assert abs(cor_right(45.0, az=0.0, lat=0.0)) < 2.0
    assert abs(cor_right(45.0, az=0.0, lat=30.0)) > 20.0


def test_layer2_short_flight_near_zero():
    # Near-zero time of flight -> almost no Earth rotation -> ~0 deflection.
    assert abs(cor_right(3.0)) < 2.0


# ---------------------------------------------------------------------------
# Layer 3 — Known-physics magnitude & azimuth dependence
# ---------------------------------------------------------------------------
def test_layer3_magnitude_in_published_band():
    # Published artillery Coriolis corrections at 25-30 km are order tens of
    # metres: not centimetres, not kilometres.
    for elev in (40.0, 45.0, 50.0):
        d = abs(cor_right(elev, lat=20.0))
        assert 10.0 < d < 250.0, (elev, d)


def test_layer3_depends_on_firing_azimuth():
    # Different compass directions deflect differently. Compare the full impact
    # shift vector for an East shot vs a North shot — they must differ.
    def shift(az):
        on = integrate_trajectory(v0=V0, elevation_deg=45.0, azimuth_deg=az, latitude_deg=20.0)
        off = integrate_trajectory(v0=V0, elevation_deg=45.0, azimuth_deg=az, coriolis=False)
        return np.array([on["impact_x"] - off["impact_x"], on["impact_y"] - off["impact_y"]])
    east, north = shift(0.0), shift(90.0)
    assert np.linalg.norm(east - north) > 5.0          # azimuth genuinely matters
    # both still deflect to the right in the NH
    assert cor_right(45.0, az=0.0) > 0.0 and cor_right(45.0, az=90.0) > 0.0


# ---------------------------------------------------------------------------
# Layer 4 — Adaptability
# ---------------------------------------------------------------------------
def test_layer4_magnitude_scales_with_latitude():
    # The cross-range (right) deflection scales with sin(latitude): stronger
    # toward the poles, ~0 at the equator.
    d0 = abs(cor_right(45.0, lat=0.0))
    d20 = abs(cor_right(45.0, lat=20.0))
    d60 = abs(cor_right(45.0, lat=60.0))
    assert d0 < d20 < d60, (d0, d20, d60)


def test_layer4_hemisphere_sign_flip_symmetric():
    north = cor_right(45.0, lat=35.0)
    south = cor_right(45.0, lat=-35.0)
    assert north > 0.0 > south
    assert abs(north + south) < 0.1 * north        # near-symmetric magnitude


def test_layer4_no_hardcoded_constant_scales_with_velocity():
    # Faster shot -> longer flight -> more Earth rotation -> more deflection.
    lo = cor_right(45.0, v0=600.0)
    hi = cor_right(45.0, v0=950.0)
    assert hi > lo > 0.0, (lo, hi)


if __name__ == "__main__":
    print("\n=== Component 11 — Coriolis effect (Earth's rotation) summary ===\n")
    print("Hemisphere (45 deg, East shot, ~25 km):")
    print(f"    Northern (+20 lat): {cor_right(45.0, lat=20.0):+6.1f} m  (deflects RIGHT)")
    print(f"    Southern (-20 lat): {cor_right(45.0, lat=-20.0):+6.1f} m  (deflects LEFT)")

    print("\nGrowth with range / time of flight (NH +20, East):")
    for e in (30.0, 45.0, 55.0):
        r = integrate_trajectory(v0=V0, elevation_deg=e)
        print(f"    range {r['range_m']:>7.0f} m  tof {r['tof_s']:>5.1f} s"
              f"  -> right {cor_right(e):>6.1f} m")

    print("\nAzimuth dependence (lat +20, 45 deg):")
    print(f"    fired East  (az 0):  right {cor_right(45.0, az=0.0):>6.1f} m")
    print(f"    fired North (az 90): right {cor_right(45.0, az=90.0):>6.1f} m")

    print("\nLatitude scaling (cross-range, ~0 at equator, stronger toward poles):")
    for la in (0.0, 20.0, 45.0, 60.0):
        print(f"    lat {la:>4.0f}: {abs(cor_right(45.0, lat=la)):>6.1f} m")

    print("\nMagnitude sanity: ~{:.0f} m at ~25 km (published artillery Coriolis: "
          "order tens of m).".format(abs(cor_right(45.0, lat=20.0))))
    off = integrate_trajectory(v0=V0, elevation_deg=45.0, coriolis=False)
    off2 = integrate_trajectory(v0=V0, elevation_deg=45.0, coriolis=False, latitude_deg=70.0)
    print("Reproduction with effect OFF (bit-for-bit): {}".format(
        all(off[k] == off2[k] for k in off)))
    print("\nRun 'python -m pytest tests/test_coriolis.py -v'.")
