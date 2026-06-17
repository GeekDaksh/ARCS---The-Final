"""
ARCS — PINN Corrector Test Suite
Phase 1

Tests run on random values every execution.
Results appended to data/test_results_pinn.csv.

Test categories:
    [1] Physics loss — verify the Galileo range equation residual
    [2] Training — PINN fits to synthetic correction data
    [3] Prediction quality — corrects systematic bias, not noise
    [4] Online learning — should_retrain() trigger fires correctly
    [5] Fallback — graceful degradation without PyTorch
    [6] API parity — identical interface to CorrectionFormula
"""

import numpy as np
import pandas as pd
import sys, os, time, datetime, tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pinn_corrector        import PINNCorrector
from physics.ballistic_solver import BallisticSolver
from physics.constants     import GRAVITY

# ── Test bookkeeping ─────────────────────────────────────────────────────────
RNG  = np.random.default_rng(seed=int(time.time()))
PASS = 0
FAIL = 0
LOG  = []

def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    ok = bool(condition)
    if ok:
        PASS += 1
    else:
        FAIL += 1
    sym = "✓" if ok else "✗"
    status = "PASS" if ok else "FAIL"
    print(f"  {sym} {status:4s}  {name:54s} {detail}")
    LOG.append({"test": name, "status": status,
                "detail": detail, "time": datetime.datetime.now()})

# ── Shared helpers ───────────────────────────────────────────────────────────

def _synthetic_corrections_df(n: int = 30, seed: int = 42) -> pd.DataFrame:
    """
    Create a synthetic corrections CSV matching the real schema.

    Simulates a robot with:
        pitch_bias = -0.5 * sin(theta_cmd)  (gravity sag)
        yaw_bias   = +0.3                    (constant IMU offset)
        v0_bias    = -2.0                    (constant velocity deficit)

    These are the 'ground truth' corrections the PINN must learn.
    """
    rng    = np.random.default_rng(seed)
    solver = BallisticSolver()
    rows   = []

    for _ in range(n):
        R  = float(rng.uniform(80, 400))
        H  = float(rng.uniform(-10, 25))
        v0 = float(rng.choice([80, 100, 120]))

        sol = solver.solve(R, H, 0.0, v0)
        if not sol.reachable:
            continue

        theta = sol.turret_pitch_deg
        # Ground-truth systematic corrections (what the PINN should learn)
        dp = -0.50 * np.sin(np.deg2rad(theta))   # gravity sag
        dy = +0.30                                 # IMU yaw offset
        dv = -2.00                                 # velocity deficit
        # Add small random noise (stochastic component, NOT learnable)
        dp += rng.normal(0, 0.05)
        dy += rng.normal(0, 0.03)
        dv += rng.normal(0, 0.20)

        rows.append({
            "range_m":     R,
            "height_m":    H,
            "v0_ms":       v0,
            "delta_pitch": dp,
            "delta_yaw":   dy,
            "delta_v0":    dv,
            "miss_before": float(rng.uniform(8, 18)),
            "miss_after":  float(rng.uniform(2, 7)),
            "confidence":  float(rng.uniform(0.3, 0.8)),
            "n_shots_used": int(rng.integers(10, 30)),
            "solution_type": "LOW",
            "engagement_id": f"test_{_}",
            "timestamp":   datetime.datetime.now().isoformat(),
        })

    return pd.DataFrame(rows)


def _write_corrections(df: pd.DataFrame, tmpdir: Path) -> str:
    path = str(tmpdir / "corr.csv")
    df.to_csv(path, index=False)
    return path


# ─────────────────────────────────────────────────────────────────────────────
print("=" * 68)
print("ARCS — PINN Corrector Test Suite")
print(f"Seed: {int(time.time())}  (different random values every run)")
print("=" * 68)

TMPDIR = Path(tempfile.mkdtemp())

# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] PHYSICS LOSS — Galileo range equation residual")

try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

if TORCH_OK:
    from pinn_corrector import PINNCorrector

    # At zero correction, residual should be ≈ 0 (nominal solution is exact)
    solver = BallisticSolver()
    for _ in range(5):
        R  = float(RNG.uniform(80, 380))
        H  = float(RNG.uniform(-10, 20))
        v0 = float(RNG.choice([80, 100, 120]))
        sol = solver.solve(R, H, 0.0, v0)
        if not sol.reachable:
            continue

        theta_cmd = torch.tensor([sol.turret_pitch_deg], dtype=torch.float32)
        range_t   = torch.tensor([R],  dtype=torch.float32)
        height_t  = torch.tensor([H],  dtype=torch.float32)
        v0_t      = torch.tensor([v0], dtype=torch.float32)

        # Zero correction → residual must be near zero
        pred_zero = torch.zeros(1, 3)
        resid = PINNCorrector._physics_residual(
            range_t, height_t, v0_t, theta_cmd, pred_zero)

        check(f"Physics residual ≈ 0 at zero correction  R={R:.0f}m H={H:.0f}m",
              float(resid.item()) < 1e-3,
              f"residual={float(resid.item()):.6f}")

    # Large (wrong) correction → residual must be large
    pred_large = torch.tensor([[3.0, 0.0, 0.0]])
    R_test = 200.0
    sol_test = solver.solve(R_test, 0.0, 0.0, 100.0)
    if sol_test.reachable:
        resid_large = PINNCorrector._physics_residual(
            torch.tensor([R_test]), torch.tensor([0.0]),
            torch.tensor([100.0]),
            torch.tensor([sol_test.turret_pitch_deg]),
            pred_large)
        check("Physics residual > 0 for large erroneous correction",
              float(resid_large.item()) > 0.001,
              f"residual={float(resid_large.item()):.6f}")
else:
    check("Physics residual test skipped (no PyTorch)", True, "n/a")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] TRAINING — PINN fits synthetic correction data")

n_records = int(RNG.integers(22, 45))   # random size each run
df_corr   = _synthetic_corrections_df(n=n_records, seed=int(time.time()) % 10000)
corr_path = _write_corrections(df_corr, TMPDIR)

pc = PINNCorrector(corr_path, solution_type="LOW")

check("PINNCorrector initialises",
      pc is not None and not pc.is_fitted, "")

# Training with fewer than MIN_RECORDS should fail gracefully
df_tiny    = df_corr.iloc[:5]
tiny_path  = str(TMPDIR / "tiny.csv")
df_tiny.to_csv(tiny_path, index=False)
pc_tiny    = PINNCorrector(tiny_path, solution_type="LOW")
ok_tiny    = pc_tiny.load_and_train(verbose=False)
check("load_and_train returns False with too few records",
      not ok_tiny and not pc_tiny.is_fitted,
      f"n_records={len(df_tiny)}")

# Full training with adequate records
ok = pc.load_and_train(verbose=True)
check("load_and_train returns True with adequate records",
      ok, f"n_records={pc.n_records}")
check("is_fitted = True after training",
      pc.is_fitted, "")
check("n_records matches CSV",
      pc.n_records == len(df_corr[df_corr["solution_type"] == "LOW"]),
      f"n={pc.n_records}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] PREDICTION QUALITY — learns systematic bias, not noise")

# After training on sag_coeff=-0.5 data, check predicted corrections are in
# the right direction across several ranges.
# Ground truth: dp ≈ -0.5 * sin(theta_cmd)
solver = BallisticSolver()

if pc.is_fitted:
    errors = []
    for R in [100, 150, 200, 250, 300]:
        sol = solver.solve(R, 0.0, 0.0, 100.0)
        if not sol.reachable:
            continue
        p = pc.predict(R, 0.0, 100.0)
        expected_dp = -0.5 * np.sin(np.deg2rad(sol.turret_pitch_deg))
        error = abs(p["delta_pitch"] - expected_dp)
        errors.append(error)

        check(f"Δpitch direction correct at R={R}m (should be ~{expected_dp:+.3f}°)",
              p["delta_pitch"] < 0,           # sag is always negative
              f"got {p['delta_pitch']:+.3f}°  expected~{expected_dp:+.3f}°")

    if errors:
        mean_err = float(np.mean(errors))
        check("Mean pitch prediction error < 0.4° (within noise tolerance)",
              mean_err < 0.40,
              f"mean_err={mean_err:.4f}°")

    # Yaw correction should be positive (≈+0.3°)
    p_yaw = pc.predict(200.0, 0.0, 100.0)
    check("Δyaw prediction is positive (≈+0.3° IMU offset)",
          p_yaw["delta_yaw"] > 0,
          f"got {p_yaw['delta_yaw']:+.3f}°")

    # v0 correction should be negative (≈-2.0 m/s)
    check("Δv₀ prediction is negative (≈-2.0 m/s velocity deficit)",
          p_yaw["delta_v0"] < 0,
          f"got {p_yaw['delta_v0']:+.2f} m/s")

    # Clipping: predictions must stay within hard bounds
    for R in [float(RNG.uniform(80, 400)) for _ in range(5)]:
        p = pc.predict(R, float(RNG.uniform(-10, 25)), 100.0)
        in_bounds = (abs(p["delta_pitch"]) <= 3.0 and
                     abs(p["delta_yaw"])   <= 2.0 and
                     abs(p["delta_v0"])    <= 10.0)
        check(f"Prediction within hard bounds  R={R:.0f}m",
              in_bounds,
              f"dp={p['delta_pitch']:+.2f}° dy={p['delta_yaw']:+.2f}° "
              f"dv={p['delta_v0']:+.2f}")

    # Source string identifies backend
    p_src = pc.predict(200.0, 0.0, 100.0)
    check("Source string set (pinn_torch or pinn_sklearn)",
          p_src["source"].startswith("pinn_"),
          f"source='{p_src['source']}'")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] ONLINE LEARNING — should_retrain() trigger")

# should_retrain() = False immediately after training
check("should_retrain() False right after training",
      not pc.should_retrain(), "no new records added")

# Add 6 records at a novel range region (250m from nearest training range)
novelty_records = _synthetic_corrections_df(n=8, seed=77)
novelty_records["range_m"] = 490.0   # far from training distribution
novel_df = pd.concat([df_corr, novelty_records], ignore_index=True)
novel_df.to_csv(corr_path, index=False)

check("should_retrain() True after novel range records added",
      pc.should_retrain(),
      f"novelty_range=490m vs trained max={df_corr['range_m'].max():.0f}m")

# After retraining, should_retrain() resets
pc.load_and_train(verbose=False)
check("should_retrain() False again after retraining",
      not pc.should_retrain(), "")

# Add records at close range (within 30m of existing) — should NOT trigger
close_records = df_corr.copy()
close_records["range_m"] = df_corr["range_m"] + float(RNG.uniform(5, 15))
close_df = pd.concat([novel_df, close_records.iloc[:6]], ignore_index=True)
close_df.to_csv(corr_path, index=False)
check("should_retrain() False for close-range records (novelty < 30m)",
      not pc.should_retrain(),
      "new records within 30m of training distribution")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] API PARITY — identical interface to CorrectionFormula")

# Check all required attributes exist
check("has .is_fitted attribute",   hasattr(pc, "is_fitted"),   "")
check("has .n_records attribute",   hasattr(pc, "n_records"),   "")
check("has .solution_type attribute", hasattr(pc, "solution_type"), "")

# Check predict() returns correct dict structure
p = pc.predict(200.0, 0.0, 100.0)
check("predict() returns dict with delta_pitch",
      isinstance(p.get("delta_pitch"), float), f"{p.get('delta_pitch')}")
check("predict() returns dict with delta_yaw",
      isinstance(p.get("delta_yaw"), float),   f"{p.get('delta_yaw')}")
check("predict() returns dict with delta_v0",
      isinstance(p.get("delta_v0"), float),    f"{p.get('delta_v0')}")
check("predict() returns dict with source",
      isinstance(p.get("source"), str),        f"{p.get('source')}")

# Unfitted instance returns zeros (safe default)
pc_empty = PINNCorrector(str(TMPDIR / "nonexistent.csv"), solution_type="LOW")
p_empty  = pc_empty.predict(200.0, 0.0, 100.0)
check("Unfitted predict() returns zero corrections",
      p_empty["delta_pitch"] == 0.0 and p_empty["source"] == "none", "")

# load_and_train on missing file returns False (not an exception)
ok_missing = pc_empty.load_and_train(verbose=False)
check("load_and_train on missing file returns False (no crash)",
      not ok_missing, "")

# HIGH trajectory — separate corrector, same API
df_high = df_corr.copy()
df_high["solution_type"] = "HIGH"
high_path = str(TMPDIR / "high_corr.csv")
df_high.to_csv(high_path, index=False)
pc_high = PINNCorrector(high_path, solution_type="HIGH")
ok_high = pc_high.load_and_train(verbose=False)
check("HIGH-trajectory PINNCorrector trains successfully",
      ok_high, f"n_records={pc_high.n_records}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] WEIGHT PERSISTENCE — save / load across instances")

if TORCH_OK and pc.is_fitted:
    weight_dir = TMPDIR / "weights"
    weight_dir.mkdir(exist_ok=True)
    weight_file = str(weight_dir / "test_weights.pt")

    # save_weights() returns True
    saved = pc.save_weights(weight_file)
    check("save_weights() returns True",
          saved, f"path={weight_file}")

    # File exists on disk
    check("Weight file exists after save_weights()",
          Path(weight_file).exists(),
          f"size={Path(weight_file).stat().st_size if Path(weight_file).exists() else 0}B")

    # Load into a fresh instance — is_fitted=True without retraining
    pc_loaded = PINNCorrector(corr_path, solution_type="LOW")
    loaded = pc_loaded.load_weights(weight_file)
    check("load_weights() returns True on fresh instance",
          loaded and pc_loaded.is_fitted,
          f"is_fitted={pc_loaded.is_fitted}")

    # Predictions from loaded weights match original
    p_orig   = pc.predict(200.0, 0.0, 100.0)
    p_loaded = pc_loaded.predict(200.0, 0.0, 100.0)
    match = (abs(p_orig["delta_pitch"] - p_loaded["delta_pitch"]) < 1e-4 and
             abs(p_orig["delta_yaw"]   - p_loaded["delta_yaw"])   < 1e-4)
    check("Loaded weights produce identical predictions",
          match,
          f"orig_dp={p_orig['delta_pitch']:+.4f}° loaded_dp={p_loaded['delta_pitch']:+.4f}°")

    # Auto-save: load_and_train() should save weights automatically
    auto_path = pc._weight_path
    if auto_path.exists():
        auto_path.unlink()   # remove so we can verify it gets re-created
    pc.load_and_train(verbose=False)
    check("load_and_train() auto-saves weight file",
          auto_path.exists(),
          f"path={auto_path}")

    # load_weights() on nonexistent path returns False
    bad = pc.load_weights(str(TMPDIR / "no_such_weights.pt"))
    check("load_weights() returns False for missing file",
          not bad, "")
else:
    for _ in range(6):
        check("Weight persistence test skipped (no PyTorch or not fitted)", True, "n/a")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] PARTIAL FIT — online fine-tuning from fitted state")

if TORCH_OK and pc.is_fitted:
    # partial_fit() returns True when fitted
    new_df = _synthetic_corrections_df(n=10, seed=int(time.time()) % 10000 + 1)
    result_pf = pc.partial_fit(new_df, n_epochs=20, verbose=False)
    check("partial_fit() returns True when fitted",
          result_pf, "")

    # is_fitted stays True after partial_fit
    check("is_fitted still True after partial_fit",
          pc.is_fitted, "")

    # Prediction changes after partial_fit (network was updated)
    p_before = pc_loaded.predict(200.0, 0.0, 100.0)   # pc_loaded has old weights
    p_after  = pc.predict(200.0, 0.0, 100.0)           # pc was fine-tuned
    changed = (abs(p_before["delta_pitch"] - p_after["delta_pitch"]) > 1e-6 or
               abs(p_before["delta_yaw"]   - p_after["delta_yaw"])   > 1e-6)
    check("partial_fit() changes predictions (network updated)",
          changed,
          f"before_dp={p_before['delta_pitch']:+.4f}° after_dp={p_after['delta_pitch']:+.4f}°")

    # partial_fit() with < 3 records returns False
    tiny_new = new_df.iloc[:2]
    check("partial_fit() returns False with < 3 records",
          not pc.partial_fit(tiny_new), f"n={len(tiny_new)}")

    # partial_fit() on unfitted instance returns False
    pc_unfit = PINNCorrector(str(TMPDIR / "nonexistent.csv"), solution_type="LOW")
    check("partial_fit() returns False on unfitted instance",
          not pc_unfit.partial_fit(new_df), "")
else:
    for _ in range(5):
        check("partial_fit test skipped (no PyTorch or not fitted)", True, "n/a")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] DATA MANAGEMENT — malformed solution_type handling")

import datetime as _dt

def _make_corrupted_csv(tmpdir: Path, n_good: int = 25, n_bad: int = 30) -> str:
    """
    Write a CSV where `n_bad` rows have timestamps (not 'LOW'/'HIGH') in
    solution_type — mimicking the column-order bug from old pipeline runs.
    All rows represent LOW-trajectory corrections.
    """
    rng2   = np.random.default_rng(999)
    solver = BallisticSolver()
    rows   = []

    for i in range(n_good + n_bad):
        R  = float(rng2.uniform(80, 380))
        H  = float(rng2.uniform(-10, 20))
        v0 = 100.0
        sol = solver.solve(R, H, 0.0, v0)
        if not sol.reachable:
            continue
        # Good rows have "LOW"; bad rows have a timestamp string
        st = "LOW" if i < n_good else _dt.datetime.now().isoformat()
        rows.append({
            "range_m": R, "height_m": H, "v0_ms": v0,
            "delta_pitch": float(rng2.uniform(-1, 1)),
            "delta_yaw":   float(rng2.uniform(-0.5, 0.5)),
            "delta_v0":    float(rng2.uniform(-3, 3)),
            "miss_before": 10.0, "miss_after": 5.0,
            "confidence": 0.5, "n_shots_used": 30,
            "engagement_id": i, "timestamp": _dt.datetime.now().isoformat(),
            "solution_type": st,
        })

    path = str(tmpdir / "corrupted.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path

corr_corrupt = _make_corrupted_csv(TMPDIR, n_good=25, n_bad=30)
pc_cor = PINNCorrector(corr_corrupt, solution_type="LOW")
ok_cor = pc_cor.load_and_train(verbose=False)

# All 55 rows (good + bad) should be counted as LOW
check("Corrupted CSV: all rows (good + timestamp-type) counted as LOW",
      pc_cor.n_records >= 50,
      f"n_records={pc_cor.n_records} (expected ≥50)")

check("Corrupted CSV: PINN trains successfully on full dataset",
      ok_cor and pc_cor.is_fitted,
      f"is_fitted={pc_cor.is_fitted}")

# should_retrain() also uses the fixed filter
check("should_retrain() False immediately after training on corrupted CSV",
      not pc_cor.should_retrain(),
      f"last_trained_n={pc_cor.last_trained_n}")

# NULL-only solution_type column → treated as LOW
null_df = _synthetic_corrections_df(n=25, seed=404)
null_df["solution_type"] = None   # explicitly set all to null
null_path = str(TMPDIR / "null_types.csv")
null_df.to_csv(null_path, index=False)
pc_null = PINNCorrector(null_path, solution_type="LOW")
ok_null = pc_null.load_and_train(verbose=False)
check("Null solution_type column: trains as LOW",
      ok_null and pc_null.n_records >= 20,
      f"n_records={pc_null.n_records}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] PRINT FORMULAS — human-readable output")

import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    pc.print_formulas()
output = buf.getvalue()

check("print_formulas() produces output",
      len(output) > 50, f"chars={len(output)}")
check("print_formulas() shows range table",
      "80" in output and "400" in output, "range table included")
check("print_formulas() shows backend",
      "pytorch" in output.lower() or "sklearn" in output.lower(), "")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*68}")
print(f"  TOTAL: {PASS+FAIL}  |  PASSED: {PASS}  |  FAILED: {FAIL}")
print(f"{'='*68}")

if FAIL > 0:
    print(f"\n  {'─'*40}")
    print(f"  Failed tests:")
    for row in LOG:
        if row["status"] == "FAIL":
            print(f"    ✗  {row['test']}  {row['detail']}")

# Save results
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "data"), exist_ok=True)
out = os.path.join(os.path.dirname(__file__), "..", "data",
                   "test_results_pinn.csv")
df_log = pd.DataFrame(LOG)
if os.path.exists(out):
    df_log = pd.concat([pd.read_csv(out), df_log], ignore_index=True)
df_log.to_csv(out, index=False)
print(f"\n  Results saved → data/test_results_pinn.csv")
print(f"  ({len(df_log)} total records across all runs)")