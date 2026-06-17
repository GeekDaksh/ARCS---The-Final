"""Live demo: run a full Phase 2 engagement and print the result."""
import os
import tempfile

from physics.met_message import MetMessage
from physics.engagement import run_engagement
from engagement_database import EngagementDatabase

# Told MET (the imperfect weather report the computer uses): 20 m/s tail wind
TOLD = MetMessage.standard_isa(surface_wind=(180.0, 20.0))
# The REAL atmosphere (hidden from the system): actually 23 m/s tail
TRUE = MetMessage.standard_isa(surface_wind=(180.0, 23.0))
# The gun's hidden mechanical bias: shoots 200 m long, 80 m right, every shot
GUN = (200.0, 80.0)

true_conditions = {"true_met": TRUE, "gun_bias": GUN}

print("=" * 64)
print("ARCS PHASE 2 — LIVE ENGAGEMENT")
print("Gun: HOW-1 (155mm M107)   Target: 22 km, bearing 0")
print("Told weather:  20 m/s tail wind")
print("REAL weather:  23 m/s tail wind   (the report is 3 m/s off)")
print("Hidden gun bias: shoots 200 m long, 80 m right -- every shot")
print("The system knows NONE of the hidden truths. It learns from misses.")
print("=" * 64)

res = run_engagement("HOW-1", 22000.0, 0.0, TOLD, true_conditions)

# Per-phase miss trace
print(f"\n{'Phase':<18}{'Misses (m)':>40}")
print("-" * 60)
for phase, misses in res["phase_misses"].items():
    trace = "  ".join(f"{m:6.1f}" for m in misses)
    print(f"{phase:<18}{trace:>40}")

print("-" * 60)
print(f"\nFinal CEP (fire for effect): {res['final_cep']:.2f} m")
print(f"Rounds to converge:          {res.get('rounds_to_converge')}")
print("\n--- WHAT THE SYSTEM LEARNED (from misses alone) ---")
print(f"  Gun bias estimate:  {tuple(round(float(x),1) for x in res['gun_bias_est'])}  "
      f"(hidden truth: {GUN})")
print(f"  Wind error estimate: {tuple(round(float(x),2) for x in res['atmo_correction_est'])}  "
      f"(hidden truth: (3.0, 0.0))")
print("=" * 64)


# ============================================================================
# FLEET MEMORY — the "living range table" payoff.
# The same gun is engaged twice. The first time it is a stranger (cold start);
# the learned gun bias is persisted to the database. The second time it is
# remembered (warm start), so only the new day's weather is left to learn and
# the very first shot lands far closer.
# ============================================================================
print("\n\n" + "=" * 64)
print("FLEET MEMORY — same gun, second engagement (warm start from DB)")
print("=" * 64)

_fd, db_path = tempfile.mkstemp(suffix=".db")
os.close(_fd)
db = EngagementDatabase(db_path)
try:
    # Engagement #1: a NEW gun the database has never seen -> cold start.
    e1 = run_engagement("HOW-FLEET", 22000.0, 0.0, TOLD, true_conditions, db=db)
    # Engagement #2: the SAME gun, SAME database -> its bias is remembered.
    e2 = run_engagement("HOW-FLEET", 22000.0, 0.0, TOLD, true_conditions, db=db)

    cold_first = float(e1["phase_misses"]["REGISTRATION"][0])
    warm_first = float(e2["phase_misses"]["REGISTRATION"][0])

    print(f"\n{'':<26}{'1st-shot miss':>16}{'rounds to converge':>22}")
    print("-" * 64)
    print(f"{'Engagement 1 (cold)':<26}{cold_first:>13.1f} m"
          f"{e1['rounds_to_converge']:>22}")
    print(f"{'Engagement 2 (warm)':<26}{warm_first:>13.1f} m"
          f"{e2['rounds_to_converge']:>22}")
    print("-" * 64)

    tighter = cold_first / warm_first if warm_first > 0 else float("inf")
    print(f"\nHEADLINE: the warm first shot is {tighter:.0f}x tighter "
          f"({cold_first:.0f} m -> {warm_first:.1f} m).")
    print(f"It remembered this gun's bias "
          f"{tuple(round(float(x),1) for x in e2['gun_bias_est'])} "
          f"(hidden truth: {GUN}),")
    print("so only the new day's weather was left to learn. Living range table.")
    print("=" * 64)
finally:
    db.close()
    os.remove(db_path)