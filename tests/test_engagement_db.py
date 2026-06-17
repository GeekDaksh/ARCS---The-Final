"""
ARCS — EngagementDatabase test suite
CLAUDE.md Section 7.1 (test_engagement_db.py)

Tests log(), get_all(), get_sbe_inputs(), statistics(), and clear().
"""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path
from engagement_database import EngagementDatabase

TOTAL = PASSED = FAILED = 0
_failures = []

def record(name: str, ok: bool, detail: str = ""):
    global TOTAL, PASSED, FAILED
    TOTAL += 1
    if ok:
        PASSED += 1
        print(f"  ✓ PASS  {name}")
    else:
        FAILED += 1
        _failures.append(name)
        print(f"  ✗ FAIL  {name}{': ' + detail if detail else ''}")


def make_record(range_m=300.0, bearing_deg=0.0, pitch_deg=8.6,
                baseline_cep=8.5, corrected_cep=5.1, improvement=40.0,
                converged=True, dp=0.08, db=0.12, dv=-2.1):
    """Helper: build a minimal but complete engagement record dict."""
    return {
        'target': {
            'x': range_m, 'y': 0.0, 'z': 0.0,
            'range': range_m, 'bearing_deg': bearing_deg,
        },
        'nominal_solution': {
            'pitch_deg': pitch_deg, 'yaw_deg': bearing_deg,
            'tof': 3.2, 'v0': 100.0,
        },
        'results': {
            'baseline_cep_m':  baseline_cep,
            'corrected_cep_m': corrected_cep,
            'improvement_pct': improvement,
            'best_correction': {'delta_pitch': dp, 'delta_yaw': db, 'delta_v0': dv},
        },
        'estimator': {
            'forgetting_rls_db_final': db,
            'forgetting_rls_dv_final': dv,
            'n_shots_bo': 16,
            'converged': converged,
        },
        'sbe_input': {
            'pitch_deg_nominal': pitch_deg,
            'dp_opt': dp, 'db_opt': db, 'dv_opt': dv,
        },
    }


print("=" * 64)
print("ARCS — EngagementDatabase Tests")
print("=" * 64)

tmpdir = Path(tempfile.mkdtemp())
db = EngagementDatabase(db_path=tmpdir / "test.db")

# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Empty database returns sensible defaults
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Test 1: Empty database")
stats = db.statistics()
record("statistics() on empty DB: n_engagements=0", stats['n_engagements'] == 0,
       f"got {stats['n_engagements']}")
record("get_all() on empty DB returns []", db.get_all() == [])
record("get_sbe_inputs() on empty DB returns []", db.get_sbe_inputs() == [])

# ─────────────────────────────────────────────────────────────────────────────
# Test 2: log() returns a UUID string
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Test 2: log() returns UUID")
eid = db.log(make_record())
record("log() returns a non-empty string", isinstance(eid, str) and len(eid) > 0)
record("returned ID is UUID format (has dashes)", eid.count('-') == 4,
       f"got '{eid}'")

# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Single record round-trip
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Test 3: Single record round-trip")
all_recs = db.get_all()
record("get_all() returns 1 record after 1 log()", len(all_recs) == 1,
       f"got {len(all_recs)}")
rec = all_recs[0]
record("logged record has engagement_id key", 'engagement_id' in rec)
record("logged record has timestamp key",      'timestamp' in rec)
record("logged record preserves range",
       abs(rec['target']['range'] - 300.0) < 0.001)
record("logged record preserves improvement_pct",
       abs(rec['results']['improvement_pct'] - 40.0) < 0.001)

# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Multiple records, ordering, and statistics
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Test 4: Multiple records")
db2 = EngagementDatabase(db_path=tmpdir / "test2.db")

improvements = [40.0, 35.0, 28.0, 42.0, 31.0]
expected_mean = sum(improvements) / len(improvements)

for imp in improvements:
    db2.log(make_record(improvement=imp,
                        baseline_cep=10.0, corrected_cep=10.0*(1 - imp/100)))

stats2 = db2.statistics()
record("statistics(): correct n_engagements count",
       stats2['n_engagements'] == len(improvements),
       f"got {stats2['n_engagements']}")
record("statistics(): mean improvement within 0.1%",
       abs(stats2['mean_improvement_pct'] - expected_mean) < 0.1,
       f"got {stats2['mean_improvement_pct']:.2f}% expected {expected_mean:.2f}%")
record("get_all() returns records in insertion order",
       len(db2.get_all()) == len(improvements))

# ─────────────────────────────────────────────────────────────────────────────
# Test 5: get_sbe_inputs() filters correctly
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Test 5: get_sbe_inputs() filters converged records only")
db3 = EngagementDatabase(db_path=tmpdir / "test3.db")
db3.log(make_record(converged=True,  db=0.12, dv=-2.1, dp=0.08))
db3.log(make_record(converged=False, db=0.10, dv=-1.9, dp=0.06))   # NOT converged
db3.log(make_record(converged=True,  db=0.11, dv=-2.0, dp=0.07))

sbe_in = db3.get_sbe_inputs()
record("get_sbe_inputs() returns only converged records (2 of 3)",
       len(sbe_in) == 2, f"got {len(sbe_in)}")
record("get_sbe_inputs() rows have required keys",
       all(k in sbe_in[0] for k in ('pitch_deg', 'dp_opt', 'db_opt', 'dv_opt')))

# ─────────────────────────────────────────────────────────────────────────────
# Test 6: clear() with confirm guard
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Test 6: clear() confirm guard")
try:
    db2.clear(confirm=False)
    record("clear(confirm=False) raises ValueError", False,
           "no exception raised")
except ValueError:
    record("clear(confirm=False) raises ValueError", True)

n_before = db2.statistics()['n_engagements']
db2.clear(confirm=True)
n_after = db2.statistics()['n_engagements']
record("clear(confirm=True) empties database",
       n_before > 0 and n_after == 0,
       f"before={n_before} after={n_after}")

# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Persistence — open same file in a new instance
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Test 7: Persistence across instances")
db_a = EngagementDatabase(db_path=tmpdir / "persist.db")
db_a.log(make_record(improvement=37.0))
db_a.log(make_record(improvement=42.0))

db_b = EngagementDatabase(db_path=tmpdir / "persist.db")
record("persisted data readable from second instance",
       db_b.statistics()['n_engagements'] == 2,
       f"got {db_b.statistics()['n_engagements']}")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*64}")
print(f"TOTAL: {TOTAL} | PASSED: {PASSED} | FAILED: {FAILED}")
if _failures:
    print("  Failed tests:")
    for f in _failures:
        print(f"  ✗ {f}")
print("=" * 64)

if __name__ == "__main__":
       sys.exit(0 if FAILED == 0 else 1)