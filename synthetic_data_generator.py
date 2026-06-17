"""
ARCS — Synthetic Data Generator  (Benchmark Edition v3.0)

CRITICAL FIX vs v2.0:
    The original simulator used pure zero-mean Gaussian noise:
        act_pitch = cmd_pitch + N(0, sigma)
    This has E[error] = 0 — no systematic bias exists.
    The GP formula therefore has NOTHING to learn and fits noise,
    making corrections harmful rather than helpful.

    v3.0 injects physically realistic SYSTEMATIC bias via RobotBiasModel:
        act_pitch = cmd_pitch + BIAS(pitch, robot_params) + N(0, sigma)
    where BIAS is deterministic and learnable by the GP formula.

    This is the difference between a toy simulation and a research result.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from physics.ballistic_solver import BallisticSolver
from physics.constants import (SIGMA_PITCH_DEG, SIGMA_YAW_DEG, SIGMA_V0,
                                 deg_to_rad, GRAVITY)
from physics.bias_model import RobotBiasModel, RobotBiasParams


class NoiseConfig:
    def __init__(self, sigma_pitch, sigma_yaw, sigma_v0, seed=42):
        self.sigma_pitch = sigma_pitch
        self.sigma_yaw   = sigma_yaw
        self.sigma_v0    = sigma_v0
        self.seed        = seed

    @classmethod
    def zero(cls):
        return cls(0.0, 0.0, 0.0, seed=42)

    @classmethod
    def standard(cls):
        return cls(SIGMA_PITCH_DEG, SIGMA_YAW_DEG, SIGMA_V0, seed=42)

    @classmethod
    def high(cls):
        return cls(0.8, 0.6, 4.0, seed=42)

    def __repr__(self):
        return (f"NoiseConfig(pitch={self.sigma_pitch}°, "
                f"yaw={self.sigma_yaw}°, v0={self.sigma_v0}m/s)")


class SyntheticDataGenerator:

    def __init__(self, noise: NoiseConfig = None, v0: float = 100.0,
                 robot_seed: int = 42):
        self.noise      = noise or NoiseConfig.standard()
        self.v0         = v0
        self.solver     = BallisticSolver()
        self.rng        = np.random.default_rng(self.noise.seed)

        # KEY FIX: when sigma=0 (zero-noise physics verification), the bias
        # model MUST also be nominal (all zeros). Otherwise systematic bias
        # causes non-zero miss distances even with perfectly noise-free shots.
        # The zero-noise test verifies physics accuracy only — both noise
        # AND bias must be zero for the miss_dist < 0.01m assertion to hold.
        is_zero_noise = (self.noise.sigma_pitch == 0.0 and
                         self.noise.sigma_yaw   == 0.0 and
                         self.noise.sigma_v0    == 0.0)
        if is_zero_noise:
            self.bias_model = RobotBiasModel(params=RobotBiasParams.nominal())
        else:
            # Each robot instance has its own systematic bias profile
            self.bias_model = RobotBiasModel(seed=robot_seed)

    def _forward_simulate(self, pitch_deg, yaw_deg, v0, target_y,
                           target_x_hint=0.0):
        p  = deg_to_rad(pitch_deg)
        ya = deg_to_rad(yaw_deg)
        v0x = v0 * np.cos(p) * np.cos(ya)
        v0y = v0 * np.sin(p)
        v0z = v0 * np.cos(p) * np.sin(ya)
        disc = v0y**2 - 2 * GRAVITY * target_y
        if disc < 0:
            disc = 0.0
        sqrt_d = np.sqrt(disc)
        t1 = (v0y + sqrt_d) / GRAVITY
        t2 = (v0y - sqrt_d) / GRAVITY
        positives = sorted([t for t in [t1, t2] if t > 1e-9])
        if not positives:
            return (0.0, target_y, 0.0)
        tof = min(positives, key=lambda t: abs(v0x * t - target_x_hint))
        return v0x*tof, v0y*tof - 0.5*GRAVITY*tof**2, v0z*tof

    def fire_shot(self, target_x, target_y, target_z):
        sol = self.solver.solve(target_x, target_y, target_z, self.v0)
        if not sol.reachable:
            return None

        cmd_pitch = sol.turret_pitch_deg
        cmd_yaw   = sol.turret_yaw_deg
        cmd_v0    = self.v0

        # Apply systematic bias + stochastic noise (v3.0 fix)
        act_pitch, act_yaw, act_v0 = self.bias_model.apply(
            cmd_pitch, cmd_yaw, cmd_v0, self.rng,
            sigma_pitch=self.noise.sigma_pitch,
            sigma_yaw=self.noise.sigma_yaw,
            sigma_v0=self.noise.sigma_v0)

        ix, iy, iz = self._forward_simulate(act_pitch, act_yaw, act_v0,
                                             target_y, target_x_hint=target_x)
        ex = ix - target_x
        ey = iy - target_y
        ez = iz - target_z
        miss = np.sqrt(ex**2 + ez**2)

        # Retrieve systematic component for analysis
        sys_bias = self.bias_model.expected_bias(cmd_pitch, cmd_yaw, cmd_v0)

        return {
            "target_x": target_x, "target_y": target_y, "target_z": target_z,
            "cmd_pitch": cmd_pitch, "cmd_yaw": cmd_yaw, "cmd_v0": cmd_v0,
            "error_x": ex, "error_y": ey, "error_z": ez,
            "miss_dist": miss,
            "act_pitch": act_pitch, "act_yaw": act_yaw, "act_v0": act_v0,
            "impact_x": ix, "impact_y": iy, "impact_z": iz,
            "horiz_range": sol.horiz_range,
            "tof": sol.tof,
            # New v3.0: expose systematic component
            "sys_pitch_bias": sys_bias["pitch_bias"],
            "sys_yaw_bias":   sys_bias["yaw_bias"],
            "sys_v0_bias":    sys_bias["v0_bias"],
        }

    def generate(self, n_targets=500, shots_per_target=10,
                  x_range=(30, 450), y_range=(-15, 40), z_range=(-200, 200)):
        rows = []
        shot_id = target_id = skipped = 0
        print(f"  Generating {n_targets} targets x {shots_per_target} shots "
              f"[robot: {self.bias_model.summary()}]")
        for _ in range(n_targets):
            tx = self.rng.uniform(*x_range)
            ty = self.rng.uniform(*y_range)
            tz = self.rng.uniform(*z_range)
            for _ in range(shots_per_target):
                row = self.fire_shot(tx, ty, tz)
                if row is not None:
                    row["shot_id"] = shot_id
                    row["target_id"] = target_id
                    rows.append(row)
                    shot_id += 1
                else:
                    skipped += 1
            target_id += 1
        df = pd.DataFrame(rows)
        print(f"  Done: {len(df):,} shots | {skipped} skipped")
        if len(df):
            # v3.0: verify systematic bias exists and is measurable
            mean_sys_pitch = df["sys_pitch_bias"].mean()
            mean_err_x = df["error_x"].mean()
            print(f"  Mean systematic pitch bias: {mean_sys_pitch:+.4f} deg")
            print(f"  Mean error_x (should be nonzero): {mean_err_x:+.3f}m")
            print(f"  Miss dist: mean={df['miss_dist'].mean():.3f}m "
                  f"std={df['miss_dist'].std():.3f}m")
        return df

    @staticmethod
    def save(df, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"  Saved -> {path} ({len(df):,} rows)")


if __name__ == "__main__":
    print("=" * 56)
    print("ARCS — Synthetic Data Generator v3.0")
    print("With physically realistic systematic bias")
    print("=" * 56)
    Path("data").mkdir(exist_ok=True)

    print("\n[1] Clean dataset (zero noise, zero bias):")
    gen = SyntheticDataGenerator(noise=NoiseConfig.zero(),
                                  v0=100, robot_seed=42)
    gen.bias_model.params = RobotBiasParams.nominal()
    df_clean = gen.generate(n_targets=200, shots_per_target=1)
    SyntheticDataGenerator.save(df_clean, "data/dataset_clean.csv")

    print("\n[2] Noisy dataset (standard noise + robot bias, robot A):")
    gen = SyntheticDataGenerator(noise=NoiseConfig.standard(),
                                  v0=100, robot_seed=42)
    df_noisy = gen.generate(n_targets=500, shots_per_target=10)
    SyntheticDataGenerator.save(df_noisy, "data/dataset_noisy.csv")

    print("\n[3] BO training dataset:")
    gen = SyntheticDataGenerator(noise=NoiseConfig.standard(),
                                  v0=100, robot_seed=42)
    df_bo = gen.generate(n_targets=300, shots_per_target=15)
    SyntheticDataGenerator.save(df_bo, "data/dataset_bo_training.csv")

    print("\n[4] Robot B dataset (different bias profile):")
    gen_b = SyntheticDataGenerator(noise=NoiseConfig.standard(),
                                    v0=100, robot_seed=999)
    df_b = gen_b.generate(n_targets=200, shots_per_target=10)
    SyntheticDataGenerator.save(df_b, "data/dataset_robot_b.csv")

    print("\nAll datasets ready.")