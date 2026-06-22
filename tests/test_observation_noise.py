"""
Validation suite for Component 13 — noisy observation of the fall of shot.

Until now the system saw each round's miss exactly. In reality the fall of shot
is reported by a forward observer / drone / radar, with measurement error. This
component feeds the ESTIMATORS the TRUE miss plus Gaussian observation noise (a
parameter, std dev in metres) and raises the Kalman/RLS measurement-noise term to
match — while leaving the TRUE physics and the kill assessment exact.

External ground truth: the known behaviour of Kalman estimation under measurement
noise — it still converges (bounded, no divergence), with a wider floor and
possibly more rounds, and it does NOT chase the noise. The converged estimate is
better than any single noisy reading because the filter averages.

numpy + stdlib only; everything here is deterministic via noise_seed.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.met_message import MetMessage
from physics.engagement import run_engagement, run_engagement_until_destroyed

# Same hidden-truth scenario as Component 6/9: an imperfect told MET vs the real
# atmosphere (3 m/s effective wind error) and a gun with a constant bias.
TOLD = MetMessage.standard_isa(surface_wind=(180.0, 20.0))
TRUE = MetMessage.standard_isa(surface_wind=(180.0, 23.0))
GUN = (200.0, 80.0)
TRUE_DELTA = (3.0, 0.0)
RANGE, BEARING = 22000.0, 0.0


def true_conditions():
    return {"true_met": TRUE, "gun_bias": GUN}


def engage(noise, seed, **kw):
    return run_engagement("W", RANGE, BEARING, TOLD, true_conditions(),
                          observation_noise_m=noise, noise_seed=seed, **kw)


# Each engagement runs a 60-step elevation solve over full trajectories, so it is
# expensive. The default-config runs are memoized so the several mean-based tests
# (which all sweep the same noise levels and seeds) share one batch rather than
# recomputing. Special-config runs (forced rounds, reproducibility) bypass this.
_CACHE = {}


def engage_cached(noise, seed):
    key = (noise, seed)
    if key not in _CACHE:
        _CACHE[key] = engage(noise, seed)
    return _CACHE[key]


def gun_err(res):
    return float(np.linalg.norm(np.asarray(res["gun_bias_est"]) - np.array(GUN)))


def atmo_err(res):
    return float(np.linalg.norm(np.asarray(res["atmo_correction_est"]) - np.array(TRUE_DELTA)))


def final_cep(res):
    return res["final_cep"]


def mean_over_seeds(noise, fn, n=5):
    """Average a per-engagement metric over many noise realizations (cached)."""
    return float(np.mean([fn(engage_cached(noise, s)) for s in range(n)]))


def single_reading_error(noise, n=5000):
    """Expected magnitude of using ONE noisy 2D observation as the estimate."""
    return float(np.mean([np.linalg.norm(np.random.default_rng(s).normal(0.0, noise, 2))
                          for s in range(n)]))


# ---------------------------------------------------------------------------
# Layer 1 — Converges despite noise (core proof)
# ---------------------------------------------------------------------------
def test_layer1_converges_with_realistic_noise():
    # One realistic-noise engagement (15 m std dev) still converges: the learned
    # gun bias and atmospheric error land near the hidden truths (tolerance
    # widened for the noise), and the FFE grouping is far better than the
    # opening miss.
    res = engage(15.0, seed=7)
    reg0 = res["phase_misses"]["REGISTRATION"][0]
    assert reg0 > 150.0                                        # opening miss really is large
    assert gun_err(res) < 40.0, res["gun_bias_est"]           # near true gun
    assert atmo_err(res) < 2.5, res["atmo_correction_est"]    # near true wind error
    assert res["final_cep"] < 0.5 * reg0                      # far better than opening


def test_layer1_estimator_stays_bounded_no_divergence():
    # The Kalman stability property: under noise the estimate never diverges or
    # oscillates unboundedly. Force many rounds (no early stop) at high noise and
    # check every logged miss and the learned state stay finite and bounded.
    res = engage(40.0, seed=3, n_register=6, n_adjust=8, n_ffe=4,
                 converge_threshold_m=0.0)
    radials = [h["radial"] for h in res["history"]]
    assert all(np.isfinite(r) for r in radials)
    assert max(radials) < 1000.0                              # bounded, no blow-up
    assert np.all(np.isfinite(res["gun_bias_est"]))
    assert np.all(np.isfinite(res["atmo_correction_est"]))
    assert gun_err(res) < 120.0                               # near truth, not chasing noise
    assert atmo_err(res) < 8.0


def test_layer1_converges_across_many_seeds():
    # Convergence is not a lucky-seed artifact: averaged over many realizations
    # at realistic noise, the learned values are close to the hidden truths.
    assert mean_over_seeds(15.0, gun_err) < 25.0
    assert mean_over_seeds(15.0, atmo_err) < 1.5


# ---------------------------------------------------------------------------
# Layer 2 — Reproduction / boundary (zero noise == perfect observation)
# ---------------------------------------------------------------------------
def test_layer2_zero_noise_reproduces_perfect_observation_bitforbit():
    # observation_noise_m=0.0 must reproduce the existing perfect-observation
    # results exactly, regardless of seed (no RNG is touched when noise is 0).
    base = run_engagement("W", RANGE, BEARING, TOLD, true_conditions())
    for seed in (None, 12345):
        r = engage(0.0, seed=seed)
        assert np.array_equal(np.asarray(r["gun_bias_est"]),
                              np.asarray(base["gun_bias_est"]))
        assert np.array_equal(np.asarray(r["atmo_correction_est"]),
                              np.asarray(base["atmo_correction_est"]))
        assert r["final_cep"] == base["final_cep"]
        assert [h["radial"] for h in r["history"]] == \
               [h["radial"] for h in base["history"]]


def test_layer2_until_destroyed_zero_noise_bitforbit():
    def fire(**kw):
        return run_engagement_until_destroyed(
            "W", RANGE, BEARING, TOLD, true_conditions(),
            lethal_radius_m=8.0, max_rounds=20, **kw)
    base = fire()
    z = fire(observation_noise_m=0.0, noise_seed=999)
    assert z["destroyed"] == base["destroyed"]
    assert z["rounds_fired"] == base["rounds_fired"]
    assert z["final_miss"] == base["final_miss"]
    assert [h["radial"] for h in z["history"]] == [h["radial"] for h in base["history"]]


def test_layer2_seed_is_reproducible():
    # Same seed -> identical realization; different seeds -> genuinely different.
    a = engage(20.0, seed=42)
    b = engage(20.0, seed=42)
    c = engage(20.0, seed=43)
    assert a["final_cep"] == b["final_cep"]
    assert np.array_equal(np.asarray(a["gun_bias_est"]), np.asarray(b["gun_bias_est"]))
    assert a["final_cep"] != c["final_cep"]


# ---------------------------------------------------------------------------
# Layer 3 — Known-statistics relationship
# ---------------------------------------------------------------------------
def test_layer3_more_noise_widens_the_floor():
    # You cannot learn the truth more precisely than you can observe it: more
    # observation noise -> noisier converged estimates and a wider final spread.
    assert mean_over_seeds(40.0, gun_err) > mean_over_seeds(3.0, gun_err)
    assert mean_over_seeds(40.0, final_cep) > mean_over_seeds(3.0, final_cep)


def test_layer3_floor_scales_monotonically_with_noise():
    errs = [mean_over_seeds(nz, gun_err) for nz in (3.0, 15.0, 40.0)]
    assert errs[0] < errs[1] < errs[2], errs


def test_layer3_averaging_beats_a_single_noisy_reading():
    # The filter AVERAGES: over many rounds the noise partially cancels, so the
    # converged gun-bias estimate is better than any single noisy reading and
    # tightens as more observations accrue. (Disable early-stop so every
    # registration round actually fires; estA depends only on registration, so
    # skip adjustment/FFE to keep this cheap.)
    noise, n = 15.0, 6
    one = float(np.mean([gun_err(engage(noise, s, n_register=1, n_adjust=0, n_ffe=0,
                                        converge_threshold_m=0.0)) for s in range(n)]))
    many = float(np.mean([gun_err(engage(noise, s, n_register=12, n_adjust=0, n_ffe=0,
                                         converge_threshold_m=0.0)) for s in range(n)]))
    ref = single_reading_error(noise)
    assert many < one                       # more observations -> tighter estimate
    assert many < 0.75 * ref                # clearly better than a single reading


# ---------------------------------------------------------------------------
# Layer 4 — Adaptability / robustness
# ---------------------------------------------------------------------------
def test_layer4_parameter_works_across_the_range():
    # Precise sensor (3 m) ... degraded observer (40 m): convergence degrades
    # gracefully and monotonically rather than breaking, and is already tight at
    # the precise end.
    errs = [mean_over_seeds(nz, gun_err) for nz in (3.0, 15.0, 40.0)]
    assert all(errs[i] < errs[i + 1] for i in range(len(errs) - 1)), errs
    assert all(np.isfinite(e) for e in errs)
    assert errs[0] < 12.0                     # a precise sensor learns the gun tightly


def test_layer4_extreme_noise_still_bounded():
    # Even with extreme observation noise the system does not diverge — it just
    # converges less tightly. Every learned value stays finite and bounded.
    for seed in range(5):
        res = engage(120.0, seed=seed)
        assert np.all(np.isfinite(res["gun_bias_est"]))
        assert np.all(np.isfinite(res["atmo_correction_est"]))
        assert gun_err(res) < 300.0           # bounded near the truth's scale
        assert np.isfinite(res["final_cep"])


def test_layer4_extreme_noise_stops_short_of_a_tight_lethal_radius():
    # With heavy observation noise the closed loop can't reliably place a round
    # inside a very tight lethal radius — it converges less tightly (or hits the
    # safety cap) rather than diverging. A tight radius under heavy noise is
    # harder than a loose one under light noise, and it never fires forever.
    def kills(noise, lethal, n=3):
        k = 0
        for s in range(n):
            r = run_engagement_until_destroyed(
                "W", RANGE, BEARING, TOLD, true_conditions(),
                lethal_radius_m=lethal, max_rounds=20,
                observation_noise_m=noise, noise_seed=s)
            assert r["rounds_fired"] <= 20            # never fires forever
            k += r["destroyed"]
        return k
    easy = kills(5.0, 30.0)
    hard = kills(60.0, 3.0)
    assert easy >= hard                                # graceful degradation


if __name__ == "__main__":
    print("\n=== Component 13 — noisy observation of fall of shot ===\n")
    base = run_engagement("W", RANGE, BEARING, TOLD, true_conditions())
    z = engage(0.0, seed=7)
    print("Bit-for-bit at zero noise:")
    print(f"    perfect-obs CEP {base['final_cep']:.4f} m  ==  "
          f"noise=0 CEP {z['final_cep']:.4f} m  -> "
          f"{z['final_cep'] == base['final_cep']}")

    print("\nConverges despite realistic noise (15 m std dev, seed 7):")
    r = engage(15.0, seed=7)
    print(f"    learned gun bias {np.round(r['gun_bias_est'], 1)}  (true {GUN}, "
          f"err {gun_err(r):.1f} m)")
    print(f"    learned atmo     {np.round(r['atmo_correction_est'], 2)}  "
          f"(true {TRUE_DELTA}, err {atmo_err(r):.2f})")
    print(f"    opening miss {r['phase_misses']['REGISTRATION'][0]:.0f} m  ->  "
          f"FFE CEP {r['final_cep']:.1f} m")

    print("\nFloor widens with observation noise (mean over seeds):")
    for nz in (0.0, 5.0, 15.0, 40.0):
        g = mean_over_seeds(nz, gun_err) if nz > 0 else gun_err(base)
        c = mean_over_seeds(nz, lambda r: r["final_cep"]) if nz > 0 else base["final_cep"]
        print(f"    noise {nz:>5.1f} m std -> mean gun err {g:6.2f} m   "
              f"mean FFE CEP {c:6.2f} m")

    print(f"\nAveraging beats a single reading (15 m std dev): a single 2D "
          f"observation errs ~{single_reading_error(15.0):.1f} m; the converged\n"
          f"    estimate over many rounds is far tighter (no divergence under "
          f"extreme 120 m noise either).")
    print("\nRun 'python -m pytest tests/test_observation_noise.py -v'.")
