"""Live demo: run a full Phase 2 engagement and print the result."""
from physics.met_message import MetMessage
from physics.engagement import run_engagement

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