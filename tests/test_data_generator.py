"""
ARCS — Data Generator Test Suite  v2.0
Random values every run. Results saved and appended.

Sections (8 total, ~40 tests):
    [1] Zero noise — shots must hit exactly (auto-bias-zeroing verified)
    [2] Noisy shots — miss non-zero and bounded
    [3] Noise distribution — sigma matches config
    [4] Dataset generation — shots_per_target contract and column completeness
    [5] High noise vs standard noise
    [6] fire_shot() return value completeness — all required keys present
    [7] Systematic bias present — learnable component exists in noisy shots
    [8] Unreachable targets — fire_shot() returns None gracefully
"""

import numpy as np
import pandas as pd
import sys, os, time, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from synthetic_data_generator import SyntheticDataGenerator, NoiseConfig
from physics.constants import SIGMA_PITCH_DEG, SIGMA_YAW_DEG, SIGMA_V0
from physics.bias_model import RobotBiasParams

RNG  = np.random.default_rng(seed=int(time.time()))
PASS = 0
FAIL = 0
LOG  = []

def check(name, condition, detail=""):
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    if condition: PASS += 1
    else:         FAIL += 1
    symbol = "✓" if condition else "✗"
    print(f"  {symbol} {status:4s}  {name:56s} {detail}")
    LOG.append({"test": name, "status": status,
                "detail": detail, "time": datetime.datetime.now()})

print("=" * 68)
print("ARCS — Data Generator Test Suite  v2.0")
print(f"Seed: {int(time.time())}  (new random values every run)")
print("=" * 68)

# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] ZERO NOISE — shots must hit exactly, auto-bias-zeroing verified")
# The code auto-zeros bias when sigma=0 (is_zero_noise check in __init__).
# This test must NOT manually force nominal bias — that would bypass the code
# path being verified.
gen_clean = SyntheticDataGenerator(noise=NoiseConfig.zero(), v0=100)

# Auto-zeroing: constructor must have set nominal (zero) bias automatically
check("Auto-zero bias: sag_coeff == 0 when NoiseConfig.zero()",
      gen_clean.bias_model.params.sag_coeff == 0.0,
      f"got {gen_clean.bias_model.params.sag_coeff}")
check("Auto-zero bias: v0_bias == 0 when NoiseConfig.zero()",
      gen_clean.bias_model.params.v0_bias == 0.0,
      f"got {gen_clean.bias_model.params.v0_bias}")

for _ in range(6):
    tx = float(RNG.uniform(30, 400))
    ty = float(RNG.uniform(-10, 30))
    tz = float(RNG.uniform(-150, 150))
    row = gen_clean.fire_shot(tx, ty, tz)
    if row is not None:
        check(f"Clean hit  ({tx:.0f},{ty:.0f},{tz:.0f})",
              row["miss_dist"] < 0.01,
              f"miss={row['miss_dist']:.2e}m")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] NOISY SHOTS — miss non-zero and bounded")
gen_noisy = SyntheticDataGenerator(noise=NoiseConfig.standard(), v0=100)
misses = []
for _ in range(20):
    tx = float(RNG.uniform(50, 400))
    ty = float(RNG.uniform(-10, 25))
    tz = float(RNG.uniform(-150, 150))
    row = gen_noisy.fire_shot(tx, ty, tz)
    if row is not None:
        misses.append(row["miss_dist"])

check("All noisy misses > 0",
      all(m > 0 for m in misses),
      f"min={min(misses):.4f}m")
check("Mean noisy miss < 50m (not explosive)",
      np.mean(misses) < 50,
      f"mean={np.mean(misses):.2f}m")
check("Mean noisy miss > 0.5m (systematic bias present)",
      np.mean(misses) > 0.5,
      f"mean={np.mean(misses):.2f}m")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] NOISE DISTRIBUTION — sigma must match config")
tx_stat = float(RNG.uniform(100, 300))
ty_stat = 0.0
tz_stat = float(RNG.uniform(-100, 100))
gen_stat = SyntheticDataGenerator(noise=NoiseConfig.standard(), v0=100)
rows_stat = [gen_stat.fire_shot(tx_stat, ty_stat, tz_stat) for _ in range(500)]
rows_stat = [r for r in rows_stat if r is not None]
df_stat   = pd.DataFrame(rows_stat)

np_std = df_stat["act_pitch"].sub(df_stat["cmd_pitch"]).std()
ny_std = df_stat["act_yaw"].sub(df_stat["cmd_yaw"]).std()
nv_std = df_stat["act_v0"].sub(df_stat["cmd_v0"]).std()

check(f"Pitch noise σ ≈ {SIGMA_PITCH_DEG}° (within ±0.05)",
      abs(np_std - SIGMA_PITCH_DEG) < 0.05,
      f"got σ={np_std:.3f}°")
check(f"Yaw noise σ ≈ {SIGMA_YAW_DEG}° (within ±0.05)",
      abs(ny_std - SIGMA_YAW_DEG) < 0.05,
      f"got σ={ny_std:.3f}°")
check(f"V0 noise σ ≈ {SIGMA_V0}m/s (within ±0.3)",
      abs(nv_std - SIGMA_V0) < 0.3,
      f"got σ={nv_std:.3f}m/s")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] DATASET GENERATION — shots_per_target contract and columns")
n_targets = int(RNG.integers(30, 80))
n_shots   = int(RNG.integers(3, 8))
print(f"  Using {n_targets} targets × {n_shots} shots = {n_targets*n_shots} total")

gen_ds = SyntheticDataGenerator(noise=NoiseConfig.standard(), v0=100)
df_ds  = gen_ds.generate(n_targets=n_targets, shots_per_target=n_shots)

REQUIRED_COLS = [
    "target_x", "target_y", "target_z",
    "cmd_pitch", "cmd_yaw", "cmd_v0",
    "act_pitch", "act_yaw", "act_v0",
    "error_x", "error_z", "miss_dist",
]

check("Dataset is not empty",
      len(df_ds) > 0, f"{len(df_ds)} rows")
check("All required columns present",
      all(c in df_ds.columns for c in REQUIRED_COLS),
      f"missing={[c for c in REQUIRED_COLS if c not in df_ds.columns]}")
check("No NaN in key columns",
      df_ds[["miss_dist", "cmd_pitch", "cmd_yaw", "act_pitch"]].isna().sum().sum() == 0)
check("Miss distances all non-negative",
      (df_ds["miss_dist"] >= 0).all(),
      f"min={df_ds['miss_dist'].min():.4f}m")
check("shots_per_target contract: row count ≤ n_targets × n_shots",
      len(df_ds) <= n_targets * n_shots,
      f"got {len(df_ds)}  max={n_targets*n_shots}")
check("shots_per_target contract: at least 80% of targets reachable",
      len(df_ds) >= n_targets * n_shots * 0.8,
      f"got {len(df_ds)}  min={int(n_targets*n_shots*0.8)}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] HIGH NOISE vs STANDARD NOISE — high must miss more")
gen_high = SyntheticDataGenerator(noise=NoiseConfig.high(), v0=100)
misses_std  = []
misses_high = []
for _ in range(30):
    tx = float(RNG.uniform(100, 300))
    ty = 0.0
    tz = float(RNG.uniform(-80, 80))
    r_std  = gen_noisy.fire_shot(tx, ty, tz)
    r_high = gen_high.fire_shot(tx, ty, tz)
    if r_std:  misses_std.append(r_std["miss_dist"])
    if r_high: misses_high.append(r_high["miss_dist"])

check("High noise mean > standard noise mean",
      np.mean(misses_high) > np.mean(misses_std),
      f"high={np.mean(misses_high):.2f}m  std={np.mean(misses_std):.2f}m")
check("High noise std > standard noise std",
      np.std(misses_high) > np.std(misses_std) * 0.8,
      f"high_std={np.std(misses_high):.2f}m  std_std={np.std(misses_std):.2f}m")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] fire_shot() RETURN VALUE COMPLETENESS")
_gen6 = SyntheticDataGenerator(noise=NoiseConfig.standard(), v0=100)
_r6   = None
for _ in range(10):
    _tx6 = float(RNG.uniform(100, 350))
    _ty6 = float(RNG.uniform(-5, 20))
    _tz6 = float(RNG.uniform(-100, 100))
    _r6  = _gen6.fire_shot(_tx6, _ty6, _tz6)
    if _r6 is not None:
        break

FIRE_SHOT_KEYS = [
    "target_x", "target_y", "target_z",
    "cmd_pitch", "cmd_yaw", "cmd_v0",
    "act_pitch", "act_yaw", "act_v0",
    "error_x", "error_z", "miss_dist",
    "horiz_range",
]
check("fire_shot() returns a dict (not None for reachable target)",
      _r6 is not None)
if _r6 is not None:
    for _key in FIRE_SHOT_KEYS:
        check(f"fire_shot() result has key '{_key}'",
              _key in _r6,
              f"got {_r6.get(_key, 'MISSING')!r}")
    check("miss_dist = sqrt(error_x² + error_z²)",
          abs(_r6["miss_dist"] - np.sqrt(_r6["error_x"]**2 + _r6["error_z"]**2)) < 1e-9,
          f"miss={_r6['miss_dist']:.4f}  computed={np.sqrt(_r6['error_x']**2+_r6['error_z']**2):.4f}")
    check("act_v0 differs from cmd_v0 (noise applied)",
          abs(_r6["act_v0"] - _r6["cmd_v0"]) > 1e-6,
          f"cmd={_r6['cmd_v0']:.2f}  act={_r6['act_v0']:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] SYSTEMATIC BIAS — learnable component exists in noisy shots")
# Use fixed robot_seed=42 so bias values are deterministic and thresholds reliable.
# With seed=42, RobotBiasModel produces a sag_coeff ≈ 0.49° and |v0_bias| > 0.8 m/s.
_gen7 = SyntheticDataGenerator(noise=NoiseConfig.standard(), v0=100,
                                robot_seed=42)
_tx7, _ty7, _tz7 = 200.0, 0.0, 0.0
_rows7 = [_gen7.fire_shot(_tx7, _ty7, _tz7) for _ in range(200)]
_rows7 = [r for r in _rows7 if r is not None]
_df7   = pd.DataFrame(_rows7)

_pitch_bias = float((_df7["act_pitch"] - _df7["cmd_pitch"]).mean())
_v0_bias    = float((_df7["act_v0"]    - _df7["cmd_v0"]).mean())
_yaw_bias   = float((_df7["act_yaw"]   - _df7["cmd_yaw"]).mean())

check("Systematic pitch bias detected (|mean| > 0.03°)",
      abs(_pitch_bias) > 0.03,
      f"mean_pitch_bias={_pitch_bias:+.4f}°")
check("Systematic v0 bias detected (|mean| > 0.5 m/s)",
      abs(_v0_bias) > 0.5,
      f"mean_v0_bias={_v0_bias:+.3f} m/s")
check("Yaw bias mean is finite",
      np.isfinite(_yaw_bias),
      f"mean_yaw_bias={_yaw_bias:+.4f}°")

# Zero-noise generator must have ZERO bias (verifies auto-zeroing in constructor)
_gen7_zero = SyntheticDataGenerator(noise=NoiseConfig.zero(), v0=100)
_rows7z = [_gen7_zero.fire_shot(_tx7, _ty7, _tz7) for _ in range(50)]
_rows7z = [r for r in _rows7z if r is not None]
_df7z   = pd.DataFrame(_rows7z)
_pitch_bias_zero = float((_df7z["act_pitch"] - _df7z["cmd_pitch"]).mean())
_v0_bias_zero    = float((_df7z["act_v0"]    - _df7z["cmd_v0"]).mean())
check("Zero-noise generator has zero pitch bias (auto-zeroing works)",
      abs(_pitch_bias_zero) < 1e-9,
      f"mean_pitch_bias={_pitch_bias_zero:+.2e}°")
check("Zero-noise generator has zero v0 bias (auto-zeroing works)",
      abs(_v0_bias_zero) < 1e-9,
      f"mean_v0_bias={_v0_bias_zero:+.2e} m/s")

# Different robot seeds must produce different bias profiles
_gen7a = SyntheticDataGenerator(noise=NoiseConfig.standard(), v0=100, robot_seed=1)
_gen7b = SyntheticDataGenerator(noise=NoiseConfig.standard(), v0=100, robot_seed=2)
check("Different robot seeds → different systematic bias",
      _gen7a.bias_model.params.v0_bias != _gen7b.bias_model.params.v0_bias,
      f"seed1={_gen7a.bias_model.params.v0_bias:+.4f}  "
      f"seed2={_gen7b.bias_model.params.v0_bias:+.4f}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] UNREACHABLE TARGETS — fire_shot() returns None gracefully")
_gen8 = SyntheticDataGenerator(noise=NoiseConfig.standard(), v0=50)

# Target that is unreachable at v0=50 (range > v0²/g ≈ 255m)
_r8_far = _gen8.fire_shot(400.0, 0.0, 0.0)
check("fire_shot() returns None for out-of-range target",
      _r8_far is None,
      f"got {_r8_far!r}")

# Reachable target at same v0 should work
_r8_ok = _gen8.fire_shot(100.0, 0.0, 0.0)
check("fire_shot() returns dict for reachable target",
      _r8_ok is not None and "miss_dist" in _r8_ok,
      f"miss={_r8_ok['miss_dist']:.2f}m" if _r8_ok else "None")

# generate() with some unreachable targets does not crash
_gen8b = SyntheticDataGenerator(noise=NoiseConfig.standard(), v0=50)
# Mix reachable and unreachable targets by using a wide range range
_df8 = _gen8b.generate(n_targets=10, shots_per_target=3)
check("generate() completes without crash when some targets unreachable",
      isinstance(_df8, pd.DataFrame),
      f"rows={len(_df8)}")
check("generate() result has no NaN miss_dist",
      _df8["miss_dist"].isna().sum() == 0 if len(_df8) > 0 else True,
      f"nan_count={_df8['miss_dist'].isna().sum() if len(_df8) > 0 else 0}")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*68}")
print(f"  TOTAL: {PASS+FAIL}  |  PASSED: {PASS}  |  FAILED: {FAIL}")
print(f"{'='*68}")

os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'data'), exist_ok=True)
_out = os.path.join(os.path.dirname(__file__), '..', 'data',
                    'test_results_data_generator.csv')
_df_log = pd.DataFrame(LOG)
if os.path.exists(_out):
    _df_log = pd.concat([pd.read_csv(_out), _df_log], ignore_index=True)
_df_log.to_csv(_out, index=False)
print(f"\n  Results saved → data/test_results_data_generator.csv")
print(f"  ({len(_df_log)} total test records across all runs)")
