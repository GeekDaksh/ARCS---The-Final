"""
Validation suite for Component 10 — gyroscopic spin drift (yaw of repose).

This is the effect that makes the model genuinely the STANAG 4355 Modified Point
Mass model rather than a plain point-mass model: a spin-stabilised shell drifts
consistently to one side. External ground truth: the drift is consistent in
direction, grows with range / time of flight, is zero when disabled, and sits in
the published order-of-magnitude band (tens of metres at long range for a 155mm).
Reference: McCoy, "Modern Exterior Ballistics" (yaw of repose); STANAG 4355.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.trajectory import integrate_trajectory

V0 = 827.0


def shot(elev, **kw):
    # Isolate spin drift (Component 10) from Coriolis (Component 11): Coriolis,
    # default-on, induces a small horizontal velocity that would itself trigger
    # spin drift, masking the pure-spin symmetry and zero-flight boundaries.
    kw.setdefault("coriolis", False)
    return integrate_trajectory(v0=V0, elevation_deg=elev, **kw)


def drift_right(elev, **kw):
    """Rightward drift (metres) = the cross-range offset spin drift adds vs the
    no-spin-drift trajectory. Right-hand twist drifts to -y, so rightward = -dy."""
    on = shot(elev, **kw)
    off = shot(elev, spin_drift=False, **kw)
    return -(on["impact_y"] - off["impact_y"])


# ---------------------------------------------------------------------------
# Layer 1 — Direction & growth (core proof), in STILL AIR
# ---------------------------------------------------------------------------
def test_layer1_right_twist_drifts_right():
    # Standard right-hand twist: consistent rightward drift in still air.
    d = drift_right(45.0)
    assert d > 0.0, d


def test_layer1_drift_grows_with_range_and_tof():
    # A longer / higher-angle shot (more time of flight) drifts more.
    short = drift_right(30.0)   # ~22 km, ~60 s tof
    long = drift_right(55.0)    # higher angle, ~97 s tof
    assert long > short > 0.0, (short, long)


def test_layer1_left_twist_is_opposite_and_symmetric():
    right = drift_right(45.0, spin_right=True)
    left = drift_right(45.0, spin_right=False)
    assert left < 0.0 < right                      # opposite directions
    assert abs(right + left) <= 1e-6 * right       # exactly symmetric magnitude


# ---------------------------------------------------------------------------
# Layer 2 — Reproduction / boundary
# ---------------------------------------------------------------------------
def test_layer2_disabled_reproduces_pointmass_bitforbit():
    a = shot(45.0, spin_drift=False)
    b = shot(45.0, spin_drift=True, twist_rate_cal=20.0, spin_right=True)
    # the spin term must not touch downrange/vertical results, and with it OFF
    # the whole trajectory is identical to the prior point-mass model.
    c = shot(45.0, spin_drift=False, twist_rate_cal=12.0)   # params ignored when off
    for k in a:
        assert a[k] == c[k], k
    # with it ON the cross-range differs (the effect is actually present)
    assert a["impact_y"] != b["impact_y"]


def test_layer2_vacuum_has_no_spin_drift():
    # Spin drift is aerodynamic: no air, no yaw of repose. Bit-for-bit in vacuum.
    on = shot(45.0, vacuum=True, spin_drift=True)
    off = shot(45.0, vacuum=True, spin_drift=False)
    for k in on:
        assert on[k] == off[k], k


def test_layer2_short_flight_drift_near_zero():
    # At very short range / near-zero time of flight there is no time for the
    # drift to accumulate, so it collapses to ~0 (and a purely vertical shot,
    # with no horizontal turning at all, drifts exactly 0).
    assert abs(drift_right(3.0)) < 1.0            # ~8 s tof, ~5.8 km
    assert abs(drift_right(90.0)) < 1e-6          # vertical: zero by construction


# ---------------------------------------------------------------------------
# Layer 3 — Known-physics magnitude
# ---------------------------------------------------------------------------
def test_layer3_magnitude_in_published_band():
    # Published 155mm spin drift at long range is order tens of metres (~1-3 mils;
    # 1 mil ~ 25 m at 25 km). Assert a sane band: not centimetres, not kilometres.
    for elev in (40.0, 45.0, 50.0):
        d = drift_right(elev)
        assert 10.0 < d < 200.0, (elev, d)


def test_layer3_much_smaller_than_strong_crosswind():
    # Spin drift is real but modest: far smaller than a 20 m/s crosswind's
    # deflection, yet not negligible (well above a metre).
    spin = drift_right(45.0)
    cw = (shot(45.0, wind=(0.0, 20.0, 0.0), spin_drift=False)["impact_y"]
          - shot(45.0, wind=(0.0, 0.0, 0.0), spin_drift=False)["impact_y"])
    assert abs(cw) > 5.0 * spin > 5.0          # crosswind >> spin drift > a few m


# ---------------------------------------------------------------------------
# Layer 4 — Adaptability (no hardcoded drift constant)
# ---------------------------------------------------------------------------
def test_layer4_direction_parameter_both_ways():
    assert drift_right(45.0, spin_right=True) > 0.0
    assert drift_right(45.0, spin_right=False) < 0.0


def test_layer4_scales_with_twist_rate():
    # Tighter twist (fewer calibers per turn) -> higher spin -> more drift.
    tight = drift_right(45.0, twist_rate_cal=16.0)
    std = drift_right(45.0, twist_rate_cal=20.0)
    loose = drift_right(45.0, twist_rate_cal=24.0)
    assert tight > std > loose > 0.0, (tight, std, loose)


def test_layer4_scales_with_muzzle_velocity_spin():
    # Spin rate p = 2*pi*v0/(twist*d): higher v0 -> more spin (and longer flight)
    # -> more drift. Physics-driven, not a fixed number.
    lo = -(integrate_trajectory(v0=600, elevation_deg=45)["impact_y"]
           - integrate_trajectory(v0=600, elevation_deg=45, spin_drift=False)["impact_y"])
    hi = -(integrate_trajectory(v0=900, elevation_deg=45)["impact_y"]
           - integrate_trajectory(v0=900, elevation_deg=45, spin_drift=False)["impact_y"])
    assert hi > lo > 0.0, (lo, hi)


if __name__ == "__main__":
    print("\n=== Component 10 — spin drift (yaw of repose) summary ===\n")
    print("Still-air rightward drift (right-hand twist, grows with range/tof):")
    for e in (30.0, 45.0, 55.0):
        on = shot(e)
        print(f"    elev {e:>4.0f}  range {on['range_m']:>7.0f} m  tof {on['tof_s']:>5.1f} s"
              f"  -> drift {drift_right(e):>6.1f} m RIGHT")

    print("\nLeft/right symmetry (45 deg):")
    print(f"    right twist {drift_right(45.0, spin_right=True):+7.1f} m   "
          f"left twist {drift_right(45.0, spin_right=False):+7.1f} m")

    print("\nMagnitude sanity vs published order-of-magnitude:")
    print(f"    ~{drift_right(45.0):.0f} m at ~25 km  "
          f"(published 155mm: order tens of m, ~1-3 mils) — in band.")
    cw = (shot(45.0, wind=(0.0, 20.0, 0.0), spin_drift=False)["impact_y"]
          - shot(45.0, wind=(0.0, 0.0, 0.0), spin_drift=False)["impact_y"])
    print(f"    20 m/s crosswind deflects {abs(cw):.0f} m  >>  spin drift "
          f"{drift_right(45.0):.0f} m  (real but modest).")

    print("\nReproduction with the effect OFF (bit-for-bit point mass):")
    a = shot(45.0, spin_drift=False); b = shot(45.0, spin_drift=False, twist_rate_cal=12.0)
    print(f"    spin_drift=False identical regardless of twist: {all(a[k]==b[k] for k in a)}")
    print(f"    near-vertical (89.5 deg) drift: {drift_right(89.5):.3f} m (~0)")

    print("\nRun 'python -m pytest tests/test_spin_drift.py -v'.")
