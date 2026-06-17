#!/usr/bin/env python3
"""
ARCS — Fleet Persistence Proof

Demonstrates that:
  1. Two weapons with DIFFERENT bias fingerprints (seed=101, seed=202) each
     learn their OWN bias and store it independently in the database.
  2. A second engagement on VAJRA-07 loads that weapon's profile from the DB
     (warm_started=1), NOT VAJRA-08's profile — memories are independent.
  3. The warm-started engagement converges in fewer rounds than the cold start.
  4. The two stored bias triples differ (different robots = different biases).

Run:
    python demo_persistence.py
"""

import sys
from pathlib import Path

DB_PATH = Path("arcs_fleet_test.db")
if DB_PATH.exists():
    DB_PATH.unlink()
    print(f"  Deleted stale {DB_PATH}")

import os
sys.path.insert(0, os.path.dirname(__file__))

from engagement_database       import EngagementDatabase, run_persistent_engagement
from structured_bias_estimator  import StructuredBiasEstimator
from bayesian_optimizer         import EngagementSimulator
from physics.bias_model         import RobotBiasModel

WEAPON_A    = "VAJRA-07"
WEAPON_B    = "VAJRA-08"
TYPE_A      = "40mm_autocannon"
TYPE_B      = "40mm_autocannon"
SEED_A      = 101    # different robot fingerprint from VAJRA-08
SEED_B      = 202
TARGET      = (300.0, 0.0, 0.0)   # 300 m straight ahead, flat ground
V0          = 100.0

print()
print("=" * 68)
print("  ARCS — Fleet Persistence Proof")
print("  Two weapons, independent memory, faster second engagement")
print("=" * 68)

# Shared database for the whole fleet
db = EngagementDatabase(db_path=str(DB_PATH))

# Print robot fingerprints so we can verify they differ
bias_a = RobotBiasModel(seed=SEED_A)
bias_b = RobotBiasModel(seed=SEED_B)
print(f"\n  Robot fingerprints:")
print(f"    {WEAPON_A}: {bias_a.summary()}")
print(f"    {WEAPON_B}: {bias_b.summary()}")

# ═════════════════════════════════════════════════════════════════════════════
#  ENGAGEMENT 1 on VAJRA-07 — cold start
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n  [{WEAPON_A}] Engagement 1 — COLD START (no prior profile)")

sim_a1  = EngagementSimulator(seed=SEED_A, bias_model=RobotBiasModel(seed=SEED_A))
sbe_a1  = StructuredBiasEstimator()

result_a1 = run_persistent_engagement(
    db=db, sim=sim_a1, sbe=sbe_a1,
    weapon_id=WEAPON_A, weapon_type=TYPE_A,
    target_x=TARGET[0], target_y=TARGET[1], target_z=TARGET[2],
    v0=V0,
)

if result_a1 is None:
    print(f"  ERROR: {WEAPON_A} engagement 1 returned None (target unreachable?)")
    sys.exit(1)

print(f"    warm_started  : {result_a1['warm_started']}")
print(f"    baseline CEP  : {result_a1['baseline_cep']:.3f} m")
print(f"    corrected CEP : {result_a1['verified_cep']:.3f} m")
print(f"    improvement   : {result_a1['improvement_pct']:+.1f}%")
print(f"    BO shots      : {result_a1['n_bo_shots']}")
bc_a1 = result_a1['best_correction']
print(f"    correction    : dp={bc_a1[0]:+.4f}°  dy={bc_a1[1]:+.4f}°  dv={bc_a1[2]:+.3f}m/s")
print(f"    SBE state     : {sbe_a1.summary()}")

# ═════════════════════════════════════════════════════════════════════════════
#  ENGAGEMENT 1 on VAJRA-08 — cold start, DIFFERENT bias
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n  [{WEAPON_B}] Engagement 1 — COLD START (no prior profile)")

sim_b1  = EngagementSimulator(seed=SEED_B, bias_model=RobotBiasModel(seed=SEED_B))
sbe_b1  = StructuredBiasEstimator()

result_b1 = run_persistent_engagement(
    db=db, sim=sim_b1, sbe=sbe_b1,
    weapon_id=WEAPON_B, weapon_type=TYPE_B,
    target_x=TARGET[0], target_y=TARGET[1], target_z=TARGET[2],
    v0=V0,
)

if result_b1 is None:
    print(f"  ERROR: {WEAPON_B} engagement 1 returned None (target unreachable?)")
    sys.exit(1)

print(f"    warm_started  : {result_b1['warm_started']}")
print(f"    baseline CEP  : {result_b1['baseline_cep']:.3f} m")
print(f"    corrected CEP : {result_b1['verified_cep']:.3f} m")
print(f"    improvement   : {result_b1['improvement_pct']:+.1f}%")
print(f"    BO shots      : {result_b1['n_bo_shots']}")
bc_b1 = result_b1['best_correction']
print(f"    correction    : dp={bc_b1[0]:+.4f}°  dy={bc_b1[1]:+.4f}°  dv={bc_b1[2]:+.3f}m/s")
print(f"    SBE state     : {sbe_b1.summary()}")

# Check bias triples differ between weapons
print(f"\n  Learned bias triple comparison (after 1 engagement each):")
profile_a = db.get_weapon_profile(WEAPON_A)
profile_b = db.get_weapon_profile(WEAPON_B)
print(f"    {WEAPON_A}: b_sag={profile_a['b_sag']:.4f}  "
      f"b_yaw={profile_a['b_yaw']:+.4f}°  b_v0={profile_a['b_v0']:+.3f}m/s")
print(f"    {WEAPON_B}: b_sag={profile_b['b_sag']:.4f}  "
      f"b_yaw={profile_b['b_yaw']:+.4f}°  b_v0={profile_b['b_v0']:+.3f}m/s")

# ═════════════════════════════════════════════════════════════════════════════
#  ENGAGEMENT 2 on VAJRA-07 — warm start from DB
#  Proves memory survives object death: fresh SBE instance loads from DB.
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n  [{WEAPON_A}] Engagement 2 — WARM START (loading remembered bias from DB)")
print(f"    (fresh EngagementSimulator + StructuredBiasEstimator instances — "
      f"proves memory survives object death)")

sim_a2 = EngagementSimulator(seed=SEED_A, bias_model=RobotBiasModel(seed=SEED_A))
sbe_a2 = StructuredBiasEstimator()   # cold object — load_state() fills it from DB

result_a2 = run_persistent_engagement(
    db=db, sim=sim_a2, sbe=sbe_a2,
    weapon_id=WEAPON_A, weapon_type=TYPE_A,
    target_x=TARGET[0], target_y=TARGET[1], target_z=TARGET[2],
    v0=V0,
)

if result_a2 is None:
    print(f"  ERROR: {WEAPON_A} engagement 2 returned None")
    sys.exit(1)

print(f"    warm_started  : {result_a2['warm_started']}")
print(f"    baseline CEP  : {result_a2['baseline_cep']:.3f} m")
print(f"    corrected CEP : {result_a2['verified_cep']:.3f} m")
print(f"    improvement   : {result_a2['improvement_pct']:+.1f}%")
print(f"    BO shots      : {result_a2['n_bo_shots']}")
bc_a2 = result_a2['best_correction']
print(f"    correction    : dp={bc_a2[0]:+.4f}°  dy={bc_a2[1]:+.4f}°  dv={bc_a2[2]:+.3f}m/s")
print(f"    SBE state     : {sbe_a2.summary()}")

# ═════════════════════════════════════════════════════════════════════════════
#  Cross-check: VAJRA-07's loaded profile != VAJRA-08's stored profile
# ═════════════════════════════════════════════════════════════════════════════
profile_a_final = db.get_weapon_profile(WEAPON_A)
profile_b_final = db.get_weapon_profile(WEAPON_B)

print(f"\n  Fleet weapon summaries:")
print(f"    {db.weapon_summary(WEAPON_A)}")
print(f"    {db.weapon_summary(WEAPON_B)}")

# ═════════════════════════════════════════════════════════════════════════════
#  Assertions
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n  Assertions:")
failures = []

def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"    PASS  {name}")
    else:
        print(f"    FAIL  {name}{': ' + detail if detail else ''}")
        failures.append(name)

# DB file created
check("arcs_fleet_test.db created",
      DB_PATH.exists(), f"expected {DB_PATH}")

# Engagement 1 on VAJRA-07 is cold
check(f"{WEAPON_A} engagement 1: warm_started=0",
      result_a1['warm_started'] == 0,
      f"got {result_a1['warm_started']}")

# Engagement 1 on VAJRA-08 is cold
check(f"{WEAPON_B} engagement 1: warm_started=0",
      result_b1['warm_started'] == 0,
      f"got {result_b1['warm_started']}")

# Engagement 2 on VAJRA-07 loaded from DB
check(f"{WEAPON_A} engagement 2: warm_started=1",
      result_a2['warm_started'] == 1,
      f"got {result_a2['warm_started']}")

# VAJRA-07 has 2 engagements in history
hist_a = db.get_engagement_history(WEAPON_A)
check(f"{WEAPON_A} has 2 engagements in history",
      len(hist_a) == 2,
      f"got {len(hist_a)}")

# VAJRA-08 has 1 engagement in history
hist_b = db.get_engagement_history(WEAPON_B)
check(f"{WEAPON_B} has 1 engagement in history",
      len(hist_b) == 1,
      f"got {len(hist_b)}")

# weapon_profiles exist for both
check(f"{WEAPON_A} profile exists", profile_a_final is not None)
check(f"{WEAPON_B} profile exists", profile_b_final is not None)

# VAJRA-07 has n_engagements=2
if profile_a_final:
    check(f"{WEAPON_A} n_engagements=2 in weapon_profile",
          profile_a_final['n_engagements'] == 2,
          f"got {profile_a_final['n_engagements']}")

# Bias triples differ between weapons (different seeds → different robots)
if profile_a_final and profile_b_final:
    b_yaw_diff = abs(profile_a_final['b_yaw'] - profile_b_final['b_yaw'])
    b_v0_diff  = abs(profile_a_final['b_v0']  - profile_b_final['b_v0'])
    biases_differ = b_yaw_diff > 0.001 or b_v0_diff > 0.01
    check("Stored bias triples differ between weapons "
          f"(Δb_yaw={b_yaw_diff:.4f}° Δb_v0={b_v0_diff:.3f}m/s)",
          biases_differ,
          "different robot seeds should produce different learned biases")

# Rounds logged for engagement 2
if result_a2.get('engagement_id', -1) > 0:
    round_count = db._conn.execute(
        "SELECT COUNT(*) FROM rounds WHERE engagement_id = ?",
        (result_a2['engagement_id'],)
    ).fetchone()[0]
    check(f"Round history logged for {WEAPON_A} engagement 2",
          round_count > 0, f"got {round_count} round rows")

# Warm start should not make CEP much worse (noise tolerance ×1.25)
cep_a1 = result_a1['verified_cep']
cep_a2 = result_a2['verified_cep']
check(f"{WEAPON_A} engagement 2 CEP ≤ 1.25× engagement 1 CEP "
      f"({cep_a2:.3f}m vs {cep_a1:.3f}m)",
      cep_a2 <= cep_a1 * 1.25,
      f"warm start CEP {cep_a2:.3f} > 1.25 × cold start {cep_a1:.3f}")

# Warm start should converge in fewer BO shots (or equal)
bo1 = result_a1.get('n_bo_shots', 0)
bo2 = result_a2.get('n_bo_shots', 0)
check(f"{WEAPON_A} engagement 2 BO shots ≤ engagement 1 "
      f"({bo2} vs {bo1})",
      bo2 <= bo1,
      f"warm start used {bo2} shots vs cold start {bo1}")

# VAJRA-07 loaded profile must differ from VAJRA-08's profile
# (proves warm-start loaded the correct weapon's memory, not a neighbour's)
if profile_a_final and profile_b_final:
    # After 2 engagements VAJRA-07's profile should show n_engagements=2
    # while VAJRA-08 shows n_engagements=1 — if they were the same object
    # this would be 2 for both.
    check("Weapons have independent engagement counts "
          f"({WEAPON_A}: n={profile_a_final['n_engagements']}, "
          f"{WEAPON_B}: n={profile_b_final['n_engagements']})",
          profile_a_final['n_engagements'] != profile_b_final['n_engagements'],
          "memories must be stored separately per weapon_id")

# ── Final verdict ──────────────────────────────────────────────────────────
print()
print("=" * 68)
passed = len(failures) == 0
if passed:
    print("  FLEET PERSISTENCE PROOF: PASS")
    print(f"  {WEAPON_A} engaged twice; warm start loaded from DB on 2nd engagement.")
    print(f"  {WEAPON_B} and {WEAPON_A} have independent learned biases.")
else:
    print(f"  FLEET PERSISTENCE PROOF: FAIL — {len(failures)} assertion(s) failed:")
    for f_name in failures:
        print(f"    FAIL  {f_name}")
print("=" * 68)

db.close()
sys.exit(0 if passed else 1)
