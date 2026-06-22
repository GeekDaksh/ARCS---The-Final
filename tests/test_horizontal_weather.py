"""
Validation suite for Component 12 — horizontal weather variation (the finale).

Weather varies by downrange POSITION, not just altitude: reliable near the gun
(real soundings), a remote estimate toward the target (enemy territory, no
sensor). A HorizontalMetField interpolates conditions between a gun profile and a
target profile along the downrange axis; the integrator samples it by both
downrange and altitude. The true downrange weather differs from the told
estimate — and the fall-of-shot estimators LEARN the discrepancy, because the
shell is the sensor we cannot place at the target.

External ground truth: direction/sign of the wind effect (provable without a
firing table), exact reproduction when the two profiles match, the known
learning behaviour of the closed loop (it converges to whatever effective
conditions explain the misses), and a monotonic confidence that decreases toward
the target. Reference: STANAG 4082 / FM 6-40 (MET interpolated between known
locations; the target profile is a remote estimate, not a measurement).

numpy + stdlib only.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.met_message import MetMessage
from physics.horizontal_met import HorizontalMetField, weather_profile_along_path
from physics.trajectory import integrate_trajectory
from physics.engagement import run_engagement

V0, R = 827.0, 22000.0
CALM = MetMessage.standard_isa(surface_wind=(0.0, 0.0))
TAIL = MetMessage.standard_isa(surface_wind=(180.0, 30.0))   # 30 m/s tail (+x)
CROSS = MetMessage.standard_isa(surface_wind=(90.0, 30.0))   # 30 m/s cross (+y)


def rng_m(met):
    return integrate_trajectory(v0=V0, elevation_deg=45.0, met=met)["range_m"]


def shot(met):
    return integrate_trajectory(v0=V0, elevation_deg=45.0, met=met)


# ---------------------------------------------------------------------------
# Layer 1 — Horizontal variation produces the correct effect (core proof)
# ---------------------------------------------------------------------------
def test_layer1_ramp_is_between_uniform_calm_and_uniform_windy():
    # Calm at the gun ramping to a strong tail wind at the target extends range
    # (tail wind), but LESS than a tail wind blowing along the whole path: the
    # shell only meets the wind on the back half. The ramp sits strictly between
    # all-calm and all-windy, and differs from both (the variation is real).
    calm, tail = rng_m(CALM), rng_m(TAIL)
    ramp = rng_m(HorizontalMetField(CALM, TAIL, R))
    assert calm < ramp < tail, (calm, ramp, tail)
    assert abs(ramp - calm) > 1.0 and abs(ramp - tail) > 1.0


def test_layer1_downrange_distribution_matters():
    # Same average wind, different downrange placement -> different impact. A tail
    # wind concentrated near the target is not the same as one near the gun, even
    # though a uniform model would treat them identically.
    near_target = rng_m(HorizontalMetField(CALM, TAIL, R))   # calm gun -> windy target
    near_gun = rng_m(HorizontalMetField(TAIL, CALM, R))      # windy gun -> calm target
    assert abs(near_target - near_gun) > 10.0, (near_target, near_gun)
    # both still between the uniform extremes
    calm, tail = rng_m(CALM), rng_m(TAIL)
    assert calm < near_target < tail and calm < near_gun < tail


def test_layer1_crosswind_near_target_deflects_partially():
    # A crosswind present only toward the target deflects the shell the same way
    # a uniform crosswind does (+y), but by less (it acts on only part of the
    # path). The partial deflection is strictly between zero and the uniform one.
    uniform = shot(CROSS)["impact_y"]
    ramp = shot(HorizontalMetField(CALM, CROSS, R))["impact_y"]
    assert uniform > 0.0
    assert 0.0 < ramp < uniform, (ramp, uniform)


# ---------------------------------------------------------------------------
# Layer 2 — Reproduction / boundary
# ---------------------------------------------------------------------------
def test_layer2_equal_profiles_reproduce_altitude_only_bitforbit():
    # gun_met == target_met (no horizontal variation) must reproduce the plain
    # altitude-only MET trajectory bit-for-bit — for the SAME object and for a
    # distinct-but-equal profile (the a + f*(b-a) interpolation makes b-a == 0).
    plain = shot(TAIL)
    same = shot(HorizontalMetField(TAIL, TAIL, R))
    distinct_equal = shot(HorizontalMetField(
        TAIL, MetMessage.standard_isa(surface_wind=(180.0, 30.0)), R))
    for k in plain:
        assert plain[k] == same[k], k
        assert plain[k] == distinct_equal[k], k


def test_layer2_sample_endpoints_are_exact():
    field = HorizontalMetField(CALM, TAIL, R)
    for alt in (0.0, 100.0, 1500.0, 7000.0):
        g, t = CALM.sample(alt), TAIL.sample(alt)
        sg, st = field.sample_at(0.0, alt), field.sample_at(R, alt)
        for key in ("temp_C", "pressure_Pa", "density_kgm3"):
            assert sg[key] == g[key], (key, alt)
            assert st[key] == t[key], (key, alt)
        assert np.array_equal(sg["wind_vector"], g["wind_vector"])
        assert np.array_equal(st["wind_vector"], t["wind_vector"])


def test_layer2_interior_sample_is_between_the_two_profiles():
    # A wind component at an interior downrange is strictly between the gun and
    # target values (linear interpolation), and at the midpoint is their mean.
    field = HorizontalMetField(CALM, TAIL, R)
    mid = field.sample_at(R / 2.0, 0.0)["wind_vector"][0]
    g = CALM.sample(0.0)["wind_vector"][0]
    t = TAIL.sample(0.0)["wind_vector"][0]
    assert min(g, t) < mid < max(g, t)
    assert abs(mid - 0.5 * (g + t)) < 1e-9


# ---------------------------------------------------------------------------
# Layer 3 — Learning the unmeasured target weather (the headline)
# ---------------------------------------------------------------------------
# Gun-area weather is MEASURED (same in told and true). The target-area profile
# is a remote GUESS in the told field and actually DIFFERENT in the true field.
GUN_MET = MetMessage.standard_isa(surface_wind=(180.0, 8.0))
TARGET_TOLD = MetMessage.standard_isa(surface_wind=(180.0, 10.0))
TARGET_TRUE = MetMessage.standard_isa(surface_wind=(200.0, 22.0))
TOLD_FIELD = HorizontalMetField(GUN_MET, TARGET_TOLD, R)
TRUE_FIELD = HorizontalMetField(GUN_MET, TARGET_TRUE, R)
GUN_BIAS = (200.0, 80.0)


def test_layer3_learns_unmeasured_target_weather_and_converges():
    res = run_engagement("WH", R, 0.0, TOLD_FIELD,
                         {"true_met": TRUE_FIELD, "gun_bias": GUN_BIAS})
    adj0 = res["phase_misses"]["ADJUSTMENT"][0]
    # The first aimed shot misses badly because the target-area weather was
    # wrong and unmeasured...
    assert adj0 > 100.0, adj0
    # ...then the system learns the effective downrange conditions from fall of
    # shot and converges to a tight group — without ever measuring the target.
    assert res["final_cep"] < 15.0, res["final_cep"]
    assert res["final_cep"] < 0.15 * adj0
    # The gun bias is still recovered (the separate estimator is not confused by
    # the horizontal weather error).
    assert np.allclose(res["gun_bias_est"], GUN_BIAS, atol=12.0), res["gun_bias_est"]


def test_layer3_converges_with_observation_noise():
    # The learning survives BOTH unmeasured target weather AND noisy observation
    # of the fall of shot (Component 13) together.
    res = run_engagement("WH", R, 0.0, TOLD_FIELD,
                         {"true_met": TRUE_FIELD, "gun_bias": GUN_BIAS},
                         observation_noise_m=15.0, noise_seed=1)
    adj0 = res["phase_misses"]["ADJUSTMENT"][0]
    assert res["final_cep"] < 25.0, res["final_cep"]
    assert res["final_cep"] < 0.25 * adj0
    assert np.allclose(res["gun_bias_est"], GUN_BIAS, atol=20.0), res["gun_bias_est"]


def test_layer3_no_discrepancy_needs_no_correction():
    # When the told field equals the true field there is nothing to learn: the
    # aimed shots are already on target and no spurious correction is invented.
    res = run_engagement("WH", R, 0.0, TOLD_FIELD,
                         {"true_met": TOLD_FIELD, "gun_bias": (0.0, 0.0)})
    assert res["phase_misses"]["ADJUSTMENT"][0] < 1.0
    assert res["final_cep"] < 1.0
    assert np.linalg.norm(res["atmo_correction_est"]) < 0.5


# ---------------------------------------------------------------------------
# Layer 4 — Visualizable data + adaptability
# ---------------------------------------------------------------------------
def test_layer4_profile_has_points_varying_wind_and_decreasing_confidence():
    prof = weather_profile_along_path(HorizontalMetField(CALM, TAIL, R), n_points=20)
    assert len(prof) == 20
    # gun end is the gun profile (calm), target end is the target profile (windy).
    assert prof[0]["downrange_m"] == 0.0
    assert abs(prof[-1]["downrange_m"] - R) < 1e-6
    assert prof[0]["wind_speed_ms"] < 1e-9
    assert abs(prof[-1]["wind_speed_ms"] - 30.0) < 1e-6
    # wind grows monotonically from gun to target (real varying data).
    speeds = [p["wind_speed_ms"] for p in prof]
    assert all(speeds[i] <= speeds[i + 1] + 1e-9 for i in range(len(speeds) - 1))
    # confidence is 1.0 at the gun and decreases monotonically toward the target.
    confs = [p["confidence"] for p in prof]
    assert abs(confs[0] - 1.0) < 1e-9
    assert confs[-1] < confs[0]
    assert all(confs[i] >= confs[i + 1] - 1e-12 for i in range(len(confs) - 1)), confs


def test_layer4_confidence_floor_is_adaptable():
    # The uncertainty model is a parameter, not hardcoded: the target-end
    # confidence equals the chosen floor, and a different floor changes it.
    for floor in (0.05, 0.2, 0.5):
        field = HorizontalMetField(CALM, TAIL, R, confidence_floor=floor)
        prof = weather_profile_along_path(field, n_points=10)
        assert abs(prof[-1]["confidence"] - floor) < 1e-9
        assert abs(prof[0]["confidence"] - 1.0) < 1e-9


def test_layer4_works_for_arbitrary_profile_pairs_and_npoints():
    # Any gun/target pair and any n_points >= 2: interior wind is between the two
    # endpoints, confidence still monotonic.
    a = MetMessage.standard_isa(surface_wind=(45.0, 12.0))
    b = MetMessage.standard_isa(surface_wind=(300.0, 25.0))
    field = HorizontalMetField(a, b, 30000.0, confidence_floor=0.1)
    for n in (2, 5, 50):
        prof = weather_profile_along_path(field, n_points=n)
        assert len(prof) == n
        confs = [p["confidence"] for p in prof]
        assert all(confs[i] >= confs[i + 1] - 1e-12 for i in range(len(confs) - 1))


def test_layer4_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        HorizontalMetField(CALM, TAIL, 0.0)               # non-positive range
    with pytest.raises(ValueError):
        HorizontalMetField(CALM, TAIL, R, confidence_floor=1.5)  # out of [0,1]
    with pytest.raises(ValueError):
        weather_profile_along_path(HorizontalMetField(CALM, TAIL, R), n_points=1)


if __name__ == "__main__":
    print("\n=== Component 12 — horizontal weather variation (the finale) ===\n")
    calm, tail = rng_m(CALM), rng_m(TAIL)
    ramp = rng_m(HorizontalMetField(CALM, TAIL, R))
    rev = rng_m(HorizontalMetField(TAIL, CALM, R))
    print("Horizontal variation vs uniform (45 deg, 30 m/s tail):")
    print(f"    uniform calm      range = {calm:9.1f} m")
    print(f"    calm gun -> windy target = {ramp:9.1f} m  (between; wind on the back half)")
    print(f"    windy gun -> calm target = {rev:9.1f} m  (same wind, different place -> {abs(ramp-rev):.0f} m apart)")
    print(f"    uniform windy     range = {tail:9.1f} m")

    plain = shot(TAIL); same = shot(HorizontalMetField(TAIL, TAIL, R))
    print(f"\nBit-for-bit when gun profile == target profile: "
          f"{all(plain[k] == same[k] for k in plain)}")

    print("\nHEADLINE — learning the UNMEASURED target-area weather from fall of shot:")
    res = run_engagement("WH", R, 0.0, TOLD_FIELD,
                         {"true_met": TRUE_FIELD, "gun_bias": GUN_BIAS})
    print(f"    told target wind is a guess; the TRUE target wind is different and never measured.")
    print(f"    opening aimed miss = {res['phase_misses']['ADJUSTMENT'][0]:.0f} m  ->  "
          f"FFE CEP = {res['final_cep']:.1f} m   (the shell sensed it)")
    print(f"    learned gun bias = {np.round(res['gun_bias_est'], 1)}  (true {GUN_BIAS})")
    print(f"    learned effective downrange wind correction = "
          f"{np.round(res['atmo_correction_est'], 2)} m/s")

    print("\nWeather along the path (real data; confidence DECREASES toward target):")
    for p in weather_profile_along_path(HorizontalMetField(CALM, TAIL, R, confidence_floor=0.2), n_points=6):
        print(f"    downrange {p['downrange_m']:7.0f} m   wind {p['wind_speed_ms']:4.1f} m/s "
              f"from {p['wind_dir_deg']:5.0f} deg   confidence {p['confidence']:.2f}")
    print("\nRun 'python -m pytest tests/test_horizontal_weather.py -v'.")
