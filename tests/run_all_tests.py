"""ARCS — Full Test Suite Runner  v2.0

Improvements over v1.0:
  - Per-suite elapsed time
  - Per-suite pass/fail counts parsed from output
  - Grand total at end (tests + time)
  - Failure detail lines reprinted in summary
  - Exit code 1 if any suite fails
"""

import subprocess, sys, os, re, datetime, time as _time
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

TESTS = [
    ("Physics",              "tests/test_physics.py"),
    ("Data Generator",       "tests/test_data_generator.py"),
    ("Range Table",          "tests/test_range_table.py"),
    ("Bayesian Optimizer",   "tests/test_bayesian_optimizer.py"),
    ("Pipeline",             "tests/test_pipeline.py"),
    ("PINN Corrector",       "tests/test_pinn.py"),
    ("HIGH Trajectory",      "tests/test_high_trajectory.py"),
    ("Kalman Filter",        "tests/test_kalman_filter.py"),
    ("Experiment",           "tests/test_experiment.py"),
    # Phase 1 new component tests
    ("KF Bearing Rotation",  "tests/test_kf_bearing_rotation.py"),
    ("Forgetting RLS",       "tests/test_forgetting_rls.py"),
    ("Struct Bias Estimator","tests/test_sbe.py"),
    ("Engagement Database",  "tests/test_engagement_db.py"),
    # Phase 1 Additions — confidence/uncertainty quantification layer
    ("BO Early Stopping",    "tests/test_bo_early_stopping.py"),
    ("SBE Credible Intervals","tests/test_sbe_credible_intervals.py"),
    ("SBE Transfer Learning","tests/test_sbe_transfer.py"),
    ("Confidence Signals",   "tests/test_confidence_signals.py"),
]

print("=" * 64)
print("ARCS — Full Test Suite  v2.0")
print(f"Run at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 64)

_TOTAL_RE = re.compile(r"TOTAL:\s*(\d+)\s*\|\s*PASSED:\s*(\d+)\s*\|\s*FAILED:\s*(\d+)")

summary        = []
grand_total    = 0
grand_passed   = 0
grand_failed   = 0
grand_elapsed  = 0.0
failure_lines  = []   # collect "✗ FAIL" lines for reprinting at end

for name, path in TESTS:
    print(f"\n{'─'*64}")
    print(f"  Running: {name}")
    print(f"{'─'*64}")

    t0 = _time.monotonic()
    r  = subprocess.run([sys.executable, path], capture_output=True, text=True)
    elapsed = _time.monotonic() - t0

    if r.stdout:
        print(r.stdout, end="")
    if r.stderr:
        print(r.stderr, end="")

    # Parse TOTAL/PASSED/FAILED from output
    suite_total = suite_passed = suite_failed = None
    for line in r.stdout.splitlines():
        m = _TOTAL_RE.search(line)
        if m:
            suite_total  = int(m.group(1))
            suite_passed = int(m.group(2))
            suite_failed = int(m.group(3))

    # Collect failure detail lines
    for line in r.stdout.splitlines():
        if "✗ FAIL" in line:
            failure_lines.append(f"  [{name}]  {line.strip()}")

    # Determine pass/fail: scan for explicit fail markers AND exit code
    has_fail = (
        r.returncode != 0
        or (suite_failed is not None and suite_failed > 0)
        or "✗ FAIL" in r.stdout
        or "Traceback" in r.stdout
        or "Traceback" in r.stderr
        or "Error:" in r.stderr
    )

    status = "ERRORS" if has_fail else "OK"
    summary.append((name, status, suite_total, suite_passed, suite_failed, elapsed))

    if suite_total is not None:
        grand_total   += suite_total
        grand_passed  += suite_passed
        grand_failed  += suite_failed
    grand_elapsed += elapsed

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*64}")
print("  SUITE SUMMARY")
print(f"{'='*64}")

all_ok = True
for name, status, total, passed, failed, elapsed in summary:
    ok  = status == "OK"
    if not ok:
        all_ok = False
    sym = "✓" if ok else "✗"
    cnt = f"{passed}/{total}" if total is not None else "?"
    print(f"  {sym}  {name:28s}  {status:6s}  "
          f"{cnt:>8} tests  {elapsed:5.1f}s")

print(f"{'─'*64}")
print(f"     {'TOTAL':28s}  {'':6s}  "
      f"{grand_passed}/{grand_total:>4} tests  {grand_elapsed:5.1f}s")
print(f"{'='*64}")

if failure_lines:
    print(f"\n  Failed tests:")
    for fl in failure_lines:
        print(fl)

if all_ok:
    print(f"\n  All {grand_total} tests passed across {len(TESTS)} suites ✓")
else:
    n_fail_suites = sum(1 for _, s, *_ in summary if s != "OK")
    print(f"\n  {n_fail_suites} suite(s) FAILED — "
          f"{grand_failed} test(s) failed of {grand_total} total ✗")

sys.exit(0 if all_ok else 1)
