"""
Validation suite for Component 6 — full engagement loop + persistence.

Ground truth: the complete loop must converge to the HIDDEN true conditions
(real gun bias + real atmospheric error), persist what it learned, and a
remembered weapon must converge faster next time. Integration only — the frozen
physics and the Component 5 estimators are called as tools, never modified.
"""

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.met_message import MetMessage
from physics.engagement import run_engagement, _load_gun_bias
from engagement_database import EngagementDatabase

# Told MET (imperfect) vs the real atmosphere (hidden): a 3 m/s effective wind
# error, and a gun whose constant bias dominates that error (so the doctrinal
# phase ordering register > adjust > FFE holds).
TOLD = MetMessage.standard_isa(surface_wind=(180.0, 20.0))   # 20 m/s tail (told)
TRUE = MetMessage.standard_isa(surface_wind=(180.0, 23.0))   # 23 m/s tail (real)
GUN = (200.0, 80.0)
TRUE_DELTA = (3.0, 0.0)   # effective wind error, bearing-0 range/cross frame


@pytest.fixture
def db():
    path = tempfile.mktemp(suffix=".db")
    d = EngagementDatabase(path)
    yield d
    d.close()
    os.remove(path)


def true_conditions(gun=GUN, true_met=TRUE):
    return {"true_met": true_met, "gun_bias": gun}


# ---------------------------------------------------------------------------
# Layer 1 — Full-loop convergence (the core proof)
# ---------------------------------------------------------------------------
def test_layer1_converges_to_hidden_truth(db):
    res = run_engagement("HOW-1", 22000.0, 0.0, TOLD, true_conditions(), db=db)
    reg0 = res["phase_misses"]["REGISTRATION"][0]
    # FFE CEP much smaller than the opening registration miss.
    assert res["final_cep"] < 15.0
    assert res["final_cep"] < 0.1 * reg0
    # Converged gun bias matches the hidden truth.
    assert np.allclose(res["gun_bias_est"], GUN, atol=12.0), res["gun_bias_est"]
    # Converged atmospheric correction matches the hidden truth.
    assert abs(res["atmo_correction_est"][0] - TRUE_DELTA[0]) < 0.6
    assert abs(res["atmo_correction_est"][1] - TRUE_DELTA[1]) < 0.6


def test_layer1_miss_shrinks_across_phases(db):
    res = run_engagement("HOW-1", 22000.0, 0.0, TOLD, true_conditions(), db=db)
    pm = res["phase_misses"]
    reg_first = pm["REGISTRATION"][0]
    adj_first = pm["ADJUSTMENT"][0]
    ffe = res["final_cep"]
    assert reg_first > adj_first > ffe, (reg_first, adj_first, ffe)


# ---------------------------------------------------------------------------
# Layer 2 — Persistence boundary
# ---------------------------------------------------------------------------
def test_layer2_learned_bias_round_trips(db):
    res = run_engagement("HOW-2", 22000.0, 0.0, TOLD, true_conditions(), db=db)
    saved = _load_gun_bias(db, "HOW-2")
    assert np.allclose(saved, res["gun_bias_est"], atol=1e-9)
    # And it survives a fresh DB connection to the same file.
    db._conn.commit()
    row = db._conn.execute(
        "SELECT gun_bias_dr, gun_bias_cr FROM weapon_profiles WHERE weapon_id=?",
        ("HOW-2",)).fetchone()
    assert np.allclose([row[0], row[1]], res["gun_bias_est"], atol=1e-9)


def test_layer2_fresh_estimator_starts_from_remembered_bias(db):
    run_engagement("HOW-2", 22000.0, 0.0, TOLD, true_conditions(), db=db)
    # A new weapon has no profile -> cold; this one does -> warm.
    assert _load_gun_bias(db, "HOW-2") is not None
    assert _load_gun_bias(db, "NEVER-FIRED") is None
    res2 = run_engagement("HOW-2", 22000.0, 0.0, TOLD, true_conditions(), db=db)
    assert res2["warm_started"] is True
    # Warm start: the very first registration shot is already tight.
    assert res2["phase_misses"]["REGISTRATION"][0] < 30.0


# ---------------------------------------------------------------------------
# Layer 3 — Fleet memory (the persistence payoff)
# ---------------------------------------------------------------------------
def test_layer3_warm_start_converges_faster(db):
    cold = run_engagement("HOW-3", 22000.0, 0.0, TOLD, true_conditions(), db=db)
    warm = run_engagement("HOW-3", 22000.0, 0.0, TOLD, true_conditions(), db=db)
    assert cold["warm_started"] is False and warm["warm_started"] is True
    # Fewer rounds to converge, and tighter early shots.
    assert warm["rounds_to_converge"] < cold["rounds_to_converge"]
    assert (warm["phase_misses"]["REGISTRATION"][0]
            < cold["phase_misses"]["REGISTRATION"][0])


def test_layer3_separate_weapons_no_cross_contamination(db):
    run_engagement("GUN-A", 22000.0, 0.0, TOLD, true_conditions(gun=(200.0, 80.0)), db=db)
    bias_a_before = _load_gun_bias(db, "GUN-A").copy()
    # A different weapon with a very different bias.
    run_engagement("GUN-B", 22000.0, 0.0, TOLD, true_conditions(gun=(-120.0, 40.0)), db=db)
    bias_a_after = _load_gun_bias(db, "GUN-A")
    bias_b = _load_gun_bias(db, "GUN-B")
    # GUN-A's profile is untouched by GUN-B's engagement.
    assert np.allclose(bias_a_before, bias_a_after, atol=1e-9)
    # Each weapon learned its own distinct personality.
    assert np.allclose(bias_a_after, [200.0, 80.0], atol=12.0)
    assert np.allclose(bias_b, [-120.0, 40.0], atol=12.0)


# ---------------------------------------------------------------------------
# Layer 4 — End-to-end sanity
# ---------------------------------------------------------------------------
def test_layer4_long_engagement_never_diverges(db):
    # Disable early-stop (threshold 0) to force many rounds; all must stay
    # finite and bounded (the Phase 1 Kalman-gain stability lesson).
    res = run_engagement("HOW-4", 22000.0, 0.0, TOLD, true_conditions(), db=db,
                         n_register=6, n_adjust=14, n_ffe=12,
                         converge_threshold_m=0.0)
    radials = [h["radial"] for h in res["history"]]
    assert all(np.isfinite(r) for r in radials)
    assert max(radials) < 1000.0          # bounded, never blows up
    assert res["final_cep"] < 15.0        # still converges


def test_layer4_no_false_correction_when_met_correct(db):
    # True conditions == told MET and zero gun bias: nothing to learn. The
    # system must fire accurately from the start without inventing corrections.
    res = run_engagement("HOW-5", 22000.0, 0.0, TOLD,
                         {"true_met": TOLD, "gun_bias": (0.0, 0.0)}, db=db)
    assert res["phase_misses"]["REGISTRATION"][0] < 1.0
    assert res["final_cep"] < 1.0
    assert np.linalg.norm(res["gun_bias_est"]) < 5.0
    assert np.linalg.norm(res["atmo_correction_est"]) < 0.5


def test_layer4_engagement_record_has_expected_fields(db):
    res = run_engagement("HOW-6", 22000.0, 45.0, TOLD, true_conditions(), db=db)
    row = db._conn.execute("""
        SELECT weapon_id, target_range, target_bearing, corrected_cep,
               rounds_to_converge, total_shots, atmo_dwind_dr, atmo_dwind_cr,
               final_cep, warm_started
        FROM engagements WHERE engagement_id=?""",
        (res["engagement_id"],)).fetchone()
    assert row["weapon_id"] == "HOW-6"
    assert row["target_range"] == 22000.0
    assert row["target_bearing"] == 45.0
    assert row["corrected_cep"] is not None
    assert row["rounds_to_converge"] == res["rounds_to_converge"]
    assert row["total_shots"] == res["total_shots"]
    assert row["final_cep"] is not None
    assert abs(row["atmo_dwind_dr"] - res["atmo_correction_est"][0]) < 1e-9


if __name__ == "__main__":
    path = tempfile.mktemp(suffix=".db")
    d = EngagementDatabase(path)
    print("\n=== Component 6 — full engagement summary ===\n")

    r1 = run_engagement("DEMO-HOW", 22000.0, 0.0, TOLD, true_conditions(), db=d)
    print("Engagement 1 (NEW weapon, cold start):")
    for ph in ("REGISTRATION", "ADJUSTMENT", "FFE"):
        ms = r1["phase_misses"][ph]
        print(f"    {ph:<13} misses (m): {[round(x, 1) for x in ms]}")
    print(f"    -> FFE CEP = {r1['final_cep']:.1f} m   "
          f"(opening miss {r1['phase_misses']['REGISTRATION'][0]:.0f} m)")
    print(f"    learned gun bias   = {np.round(r1['gun_bias_est'], 1)}  (true {GUN})")
    print(f"    learned atmo corr  = {np.round(r1['atmo_correction_est'], 2)}  "
          f"(true {TRUE_DELTA})")
    print(f"    rounds to converge = {r1['rounds_to_converge']}")

    saved = _load_gun_bias(d, "DEMO-HOW")
    print(f"\nPersistence round-trip: DB has gun bias {np.round(saved, 1)} "
          f"== learned {np.round(r1['gun_bias_est'], 1)}")

    r2 = run_engagement("DEMO-HOW", 22000.0, 0.0, TOLD, true_conditions(), db=d)
    print("\nEngagement 2 (SAME weapon, warm start from DB):")
    print(f"    first registration miss = {r2['phase_misses']['REGISTRATION'][0]:.1f} m "
          f"(was {r1['phase_misses']['REGISTRATION'][0]:.0f} m cold)")
    print(f"    rounds to converge = {r2['rounds_to_converge']}  "
          f"(was {r1['rounds_to_converge']} cold)  -> FLEET MEMORY PAYOFF")

    d.close(); os.remove(path)
    print("\nRun 'python -m pytest tests/test_engagement.py -v'.")
