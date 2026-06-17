"""
ARCS — Benchmark Experiment Runner
Phase 1 — Research-Grade Evaluation

Implements the full set of experiments required for top-tier publication:

    [1.4] Multi-trial learning curves: run N_TRIALS times with different
          seeds, plot mean ± std of verified CEP vs engagement number.
          Shows improvement is real, not noise.

    [1.5] Ablation study: measures contribution of each component:
          (A) Physics only (no BO, no PINN)
          (B) Physics + BO only (no PINN)
          (C) Physics + PINN only (no BO)
          (D) Full system (physics + BO + PINN)

    [4.1] Baseline comparisons:
          - Random search: same shot budget as BO, random corrections
          - Grid search: systematic grid over correction space
          (PID controller requires real hardware; omitted here)

    [4.2] Jensen bias quantification: measure systematic miss vs range
          from 1000 zero-noise shots at 5 benchmark ranges.

    [4.3] Shot efficiency: CEP vs shots fired (convergence speed).

    [4.4] Transfer learning: train on robot A (seed A), test robot B (seed B).
          Does prior from A help B converge faster?

    [3.5] Convergence detection: _has_converged() to detect when to stop.

OUTPUT FILES:
    data/experiment_ablation.csv      — ablation results
    data/experiment_learning_curves.csv — multi-trial curves
    data/experiment_shot_efficiency.csv — CEP vs shots
    data/experiment_jensen_bias.csv    — Jensen bias by range
    data/experiment_transfer.csv       — transfer learning
    data/experiment_summary.json       — top-level metrics (for paper)
"""

import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from copy import deepcopy

from physics.ballistic_solver import BallisticSolver
from physics.range_table import RangeTable
from bayesian_optimizer import (BayesianOptimizer, EngagementMemory,
                                  EngagementSimulator, GlobalModel)
from pinn_corrector import PINNCorrector
from metrics import ConvergenceTracker

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────

N_TRIALS     = 5    # learning-curve trials (use 10 for camera-ready paper)
N_ENGAGE     = 20   # engagements per trial
SHOT_BUDGET  = 14   # n_init + n_suggest (BO shot budget)
V0           = 100.0
DATA_DIR     = Path("data")


def fixed_target_set(seed: int = 99, n: int = 20) -> list:
    """Generate a fixed set of target (x,y,z) tuples from a seed."""
    rng = np.random.default_rng(seed)
    return [(float(rng.uniform(80, 420)),
             float(rng.uniform(-10, 30)),
             float(rng.uniform(-150, 150)))
            for _ in range(n)]


# ─────────────────────────────────────────────────────────────────
# PHYSICS-ONLY BASELINE
# ─────────────────────────────────────────────────────────────────

def physics_only_cep(target_x, target_y, target_z, v0=V0, seed=0, n=30):
    """
    CEP achievable with pure physics solver, no corrections.
    This is the baseline all AI methods must beat.
    """
    sim = EngagementSimulator(seed=seed)
    sol = BallisticSolver().solve(target_x, target_y, target_z, v0)
    if not sol.reachable:
        return None
    cep, _ = sim.baseline_cep(sol, target_x, target_y, target_z, v0, n=n)
    return cep


# ─────────────────────────────────────────────────────────────────
# RANDOM SEARCH BASELINE  [Improvement 4.1]
# ─────────────────────────────────────────────────────────────────

class RandomSearchBaseline:
    """
    Random search over correction space with same shot budget as BO.
    Shows BO is better than luck.
    """

    def __init__(self, n_candidates=SHOT_BUDGET, n_avg=4, seed=None):
        self.n_candidates = n_candidates
        self.n_avg        = n_avg
        self.rng          = np.random.default_rng(seed)
        self.bounds       = np.array([[-1.5, 1.5], [-1.5, 1.5], [-5.0, 5.0]])

    def run_engagement(self, target_x, target_y, target_z,
                       v0=V0, seed=None) -> dict:
        sim = EngagementSimulator(seed=seed)
        sol = BallisticSolver().solve(target_x, target_y, target_z, v0)
        if not sol.reachable:
            return None

        bl, _ = sim.baseline_cep(sol, target_x, target_y, target_z, v0)

        best_miss = np.inf
        best_corr = np.zeros(3)
        total_shots = 0

        for _ in range(self.n_candidates):
            corr = np.array([self.rng.uniform(lo, hi)
                              for lo, hi in self.bounds])
            avg_miss, _ = sim.fire_averaged(corr, sol,
                                             target_x, target_y, target_z,
                                             v0, self.n_avg)
            total_shots += self.n_avg
            if avg_miss < best_miss:
                best_miss = avg_miss
                best_corr = corr

        v_cep = sim.verified_cep(sol, best_corr,
                                  target_x, target_y, target_z, v0)
        total_shots += 30
        imp = (bl - v_cep) / bl * 100 if bl > 0 else 0.0

        return {"baseline_cep": bl, "verified_cep": v_cep,
                "improvement_pct": imp, "total_shots": total_shots,
                "method": "random_search"}


# ─────────────────────────────────────────────────────────────────
# GRID SEARCH BASELINE  [Improvement 4.1]
# ─────────────────────────────────────────────────────────────────

class GridSearchBaseline:
    """
    Systematic grid over correction space.
    Shows BO's efficiency advantage over exhaustive search.
    """

    def __init__(self, n_grid=3, n_avg=4, seed=None):
        """n_grid=3 gives 3^3=27 grid points."""
        self.n_grid = n_grid
        self.n_avg  = n_avg
        # Build grid
        pitch_vals = np.linspace(-1.5, 1.5, n_grid)
        yaw_vals   = np.linspace(-1.5, 1.5, n_grid)
        dv_vals    = np.linspace(-5.0, 5.0, n_grid)
        self.grid  = np.array([[dp, dy, dv]
                                for dp in pitch_vals
                                for dy in yaw_vals
                                for dv in dv_vals])

    def run_engagement(self, target_x, target_y, target_z,
                       v0=V0, seed=None) -> dict:
        sim = EngagementSimulator(seed=seed)
        sol = BallisticSolver().solve(target_x, target_y, target_z, v0)
        if not sol.reachable:
            return None

        bl, _ = sim.baseline_cep(sol, target_x, target_y, target_z, v0)

        best_miss = np.inf
        best_corr = np.zeros(3)
        total_shots = 0

        for corr in self.grid:
            avg_miss, _ = sim.fire_averaged(corr, sol,
                                             target_x, target_y, target_z,
                                             v0, self.n_avg)
            total_shots += self.n_avg
            if avg_miss < best_miss:
                best_miss = avg_miss
                best_corr = corr

        v_cep = sim.verified_cep(sol, best_corr,
                                  target_x, target_y, target_z, v0)
        total_shots += 30
        imp = (bl - v_cep) / bl * 100 if bl > 0 else 0.0

        return {"baseline_cep": bl, "verified_cep": v_cep,
                "improvement_pct": imp, "total_shots": total_shots,
                "method": "grid_search"}


# ─────────────────────────────────────────────────────────────────
# ABLATION STUDY  [Improvement 1.5]
# ─────────────────────────────────────────────────────────────────

def run_ablation(targets: list, v0: float = V0,
                 seed: int = 42, verbose: bool = True) -> pd.DataFrame:
    """
    Measure contribution of each ARCS component.

    Conditions:
        (A) Physics only     — no BO, no GP formula
        (B) Physics + BO     — no GP formula
        (C) Physics + PINN    — no BO (PINN pre-correction only)
        (D) Full ARCS         — BO + PINN

    Without this ablation a claim that "ARCS works" cannot be made —
    you can only claim the full system achieves X CEP.
    This separates the contribution of each layer.
    """

    import tempfile
    tmpdir = Path(tempfile.mkdtemp())

    def make_rt(tag):
        rt = RangeTable(
            physics_path=str(tmpdir / f"phys_{tag}.csv"),
            corrections_path=str(tmpdir / f"corr_{tag}.csv"))
        rt.generate_physics(
            range_steps=np.arange(50, 455, 50),
            height_steps=np.arange(-10, 35, 15),
            v0_steps=np.array([80, 100, 120]),
            verbose=False, force=True)
        return rt

    rows = []

    for condition, label in [("A","Physics_Only"), ("B","Physics+BO"),
                               ("C","Physics+GP"), ("D","Full_ARCS")]:
        if verbose:
            print(f"\n  [{condition}] {label}")

        rt  = make_rt(condition)
        sim = EngagementSimulator(seed=seed, range_table=rt)

        # PINN corrector (only used in C and D)
        cf = PINNCorrector(str(tmpdir / f"corr_{condition}.csv"),
                                solution_type="LOW")

        condition_results = []
        for i, (tx, ty, tz) in enumerate(targets):
            sol = BallisticSolver().solve(tx, ty, tz, v0)
            if not sol.reachable:
                continue

            bl, _ = sim.baseline_cep(sol, tx, ty, tz, v0)

            if condition == "A":
                # Physics only: no correction
                v_cep = bl
                imp   = 0.0
                shots = 30

            elif condition == "B":
                # Physics + BO: no PINN
                bo = BayesianOptimizer(
                    memory=EngagementMemory(), global_model=GlobalModel(),
                    n_init=4, n_suggest=8, kappa=2.0, kappa_min=0.5)
                r = sim.run_engagement(tx, ty, tz, v0=v0, optimizer=bo,
                                        gp_pre_correction=None)
                if r is None:
                    continue
                v_cep = r["verified_cep"]
                imp   = r["improvement_pct"]
                shots = r["total_shots"]

            elif condition == "C":
                # Physics + PINN: no BO (single shot with PINN pre-correction)
                gp_corr = {"delta_pitch": 0.0, "delta_yaw": 0.0, "delta_v0": 0.0}
                if cf.is_fitted:
                    gp_corr = cf.predict(sol.horiz_range, ty, v0)
                corr = np.array([gp_corr["delta_pitch"],
                                  gp_corr["delta_yaw"],
                                  gp_corr["delta_v0"]])
                v_cep = sim.verified_cep(sol, corr, tx, ty, tz, v0)
                imp   = (bl - v_cep) / bl * 100 if bl > 0 else 0.0
                shots = 30

                # Record correction for GP retraining
                rt.record_correction(
                    range_m=sol.horiz_range, height_m=ty, v0_ms=v0,
                    delta_pitch=corr[0], delta_yaw=corr[1], delta_v0=corr[2],
                    miss_before=bl, miss_after=v_cep,
                    confidence=0.5, n_shots_used=shots)
                if i >= PINNCorrector.MIN_RECORDS - 1:
                    cf.load_and_train(verbose=False)

            else:  # D: Full ARCS
                bo = BayesianOptimizer(
                    memory=EngagementMemory(), global_model=GlobalModel(),
                    n_init=4, n_suggest=8, kappa=2.0, kappa_min=0.5)
                gp_corr = None
                if cf.is_fitted:
                    gp_corr = cf.predict(sol.horiz_range, ty, v0)
                r = sim.run_engagement(tx, ty, tz, v0=v0, optimizer=bo,
                                        gp_pre_correction=gp_corr)
                if r is None:
                    continue
                v_cep = r["verified_cep"]
                imp   = r["improvement_pct"]
                shots = r["total_shots"]
                if i >= PINNCorrector.MIN_RECORDS - 1:
                    cf.load_and_train(verbose=False)

            row = {
                "condition":      condition,
                "label":          label,
                "engagement":     i+1,
                "target":         f"({tx:.0f},{ty:.0f},{tz:.0f})",
                "horiz_range":    sol.horiz_range,
                "baseline_cep_m": bl,
                "verified_cep_m": v_cep,
                "improvement_pct": imp,
                "shots_used":     shots,
            }
            condition_results.append(row)

            if verbose and (i+1) % 5 == 0:
                mean_imp = np.mean([r["improvement_pct"] for r in condition_results])
                print(f"    eng={i+1}  mean_improve={mean_imp:+.1f}%")

        rows.extend(condition_results)

    df = pd.DataFrame(rows)
    return df


# ─────────────────────────────────────────────────────────────────
# MULTI-TRIAL LEARNING CURVES  [Improvement 1.4]
# ─────────────────────────────────────────────────────────────────

def run_learning_curves(targets: list, n_trials: int = N_TRIALS,
                         v0: float = V0, verbose: bool = True) -> pd.DataFrame:
    """
    Run the full pipeline N_TRIALS times with different seeds.
    Returns mean ± std of verified CEP vs engagement number.

    This proves improvement is systematic, not a lucky run.
    Required for any credible research publication.
    """
    import tempfile
    all_rows = []

    for trial in range(n_trials):
        seed = 42 + trial * 100
        if verbose:
            print(f"  Trial {trial+1}/{n_trials}  (seed={seed})")

        tmpdir = Path(tempfile.mkdtemp())
        rt = RangeTable(
            physics_path=str(tmpdir / "phys.csv"),
            corrections_path=str(tmpdir / "corr.csv"))
        rt.generate_physics(
            range_steps=np.arange(50, 455, 50),
            height_steps=np.arange(-10, 35, 15),
            v0_steps=np.array([80, 100, 120]),
            verbose=False, force=True)

        sim  = EngagementSimulator(seed=seed, range_table=rt)
        mem  = EngagementMemory()
        gmod = GlobalModel()
        bo   = BayesianOptimizer(memory=mem, global_model=gmod,
                                  n_init=4, n_suggest=8,
                                  kappa=2.0, kappa_min=0.5)

        for i, (tx, ty, tz) in enumerate(targets):
            r = sim.run_engagement(tx, ty, tz, v0=v0, optimizer=bo)
            if r is None:
                continue

            all_rows.append({
                "trial":           trial,
                "seed":            seed,
                "engagement":      i+1,
                "baseline_cep_m":  r["baseline_cep"],
                "verified_cep_m":  r["verified_cep"],
                "ci_low":          r.get("ci_low", r["verified_cep"]),
                "ci_high":         r.get("ci_high", r["verified_cep"]),
                "improvement_pct": r["improvement_pct"],
                "shots_used":      r["total_shots"],
            })

            # Retrain global model every 5 engagements
            if (i+1) % 5 == 0 and rt._corrections_df is not None:
                gmod.train(rt._corrections_df)
                bo.global_model = gmod

    df = pd.DataFrame(all_rows)
    return df


# ─────────────────────────────────────────────────────────────────
# SHOT EFFICIENCY CURVE  [Improvement 4.3]
# ─────────────────────────────────────────────────────────────────

def run_shot_efficiency(target_x: float = 200.0, target_y: float = 0.0,
                         target_z: float = 0.0, v0: float = V0,
                         max_shots: int = 120, seed: int = 42) -> pd.DataFrame:
    """
    Track CEP as shots accumulate within one engagement.
    Shows how quickly BO reaches near-optimal vs random search.
    'How many shots to achieve X CEP?' — the key industrial metric.
    """
    sim = EngagementSimulator(seed=seed)
    sol = BallisticSolver().solve(target_x, target_y, target_z, v0)
    if not sol.reachable:
        return pd.DataFrame()

    rows = []
    bl, _ = sim.baseline_cep(sol, target_x, target_y, target_z, v0)

    # BO efficiency
    bo       = BayesianOptimizer(n_init=4, n_suggest=20,
                                   kappa=2.0, kappa_min=0.5)
    bo_sim   = EngagementSimulator(seed=seed)
    bo.reset(range_m=sol.horiz_range, height_m=target_y)
    gen      = bo.run()
    correction = next(gen)
    shot_count = 0
    best_corr  = np.zeros(3)

    try:
        while shot_count < max_shots:
            avg_miss, _ = bo_sim.fire_averaged(correction, sol,
                                                target_x, target_y, target_z,
                                                v0, 4)
            shot_count += 4
            v_cep = bo_sim.verified_cep(sol, bo.best_correction,
                                         target_x, target_y, target_z, v0,
                                         n=20)
            rows.append({"method": "BO_Matern52",
                          "shots": shot_count,
                          "verified_cep_m": v_cep,
                          "baseline_m": bl})
            correction = gen.send(avg_miss)
    except StopIteration:
        pass

    # Random search efficiency
    rs_sim = EngagementSimulator(seed=seed+1)
    rng    = np.random.default_rng(seed)
    bounds = np.array([[-1.5,1.5],[-1.5,1.5],[-5.0,5.0]])
    best_rs_miss = bl
    best_rs_corr = np.zeros(3)
    for shot_i in range(0, max_shots, 4):
        corr = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
        miss, _ = rs_sim.fire_averaged(corr, sol, target_x, target_y, target_z, v0, 4)
        if miss < best_rs_miss:
            best_rs_miss = miss
            best_rs_corr = corr
        v_cep = rs_sim.verified_cep(sol, best_rs_corr,
                                     target_x, target_y, target_z, v0, n=20)
        rows.append({"method": "RandomSearch",
                      "shots": shot_i+4,
                      "verified_cep_m": v_cep,
                      "baseline_m": bl})

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# JENSEN BIAS QUANTIFICATION  [Improvement 4.2]
# ─────────────────────────────────────────────────────────────────

def quantify_jensen_bias(ranges: list = None, v0: float = V0,
                          n_shots: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Measure systematic (Jensen's inequality) bias vs range.

    Fire 1000 zero-noise shots at each range and measure mean impact
    position. The nonlinear projection of angles causes a range-dependent
    systematic offset even with perfect noise. Characterising this turns
    a footnote into a publishable finding.
    """
    from physics.constants import SIGMA_PITCH_DEG
    if ranges is None:
        ranges = [80, 150, 200, 300, 400]

    solver = BallisticSolver()
    rng    = np.random.default_rng(seed)
    rows   = []

    for R in ranges:
        sol = solver.solve(R, 0, 0, v0)
        if not sol.reachable:
            continue

        # Zero noise: cmd_pitch = act_pitch exactly
        impacts_x, impacts_z = [], []
        for _ in range(n_shots):
            # Add tiny epsilon noise to simulate numerical reality
            pitch = sol.turret_pitch_deg
            yaw   = sol.turret_yaw_deg
            from physics.constants import deg_to_rad, GRAVITY
            p  = deg_to_rad(pitch)
            ya = deg_to_rad(yaw)
            v0x = v0*np.cos(p)*np.cos(ya)
            v0y = v0*np.sin(p)
            v0z = v0*np.cos(p)*np.sin(ya)
            disc = v0y**2
            sqrt_d = np.sqrt(disc)
            t1 = (v0y+sqrt_d)/GRAVITY
            t2 = (v0y-sqrt_d)/GRAVITY
            pos = sorted([t for t in [t1,t2] if t > 1e-9])
            if not pos:
                continue
            tof = min(pos, key=lambda t: abs(v0x*t - R))
            impacts_x.append(v0x*tof)
            impacts_z.append(v0z*tof)

        mean_x    = float(np.mean(impacts_x))
        mean_z    = float(np.mean(impacts_z))
        bias_x    = mean_x - R          # systematic x-bias
        bias_z    = mean_z - 0.0        # systematic z-bias
        bias_mag  = float(np.sqrt(bias_x**2 + bias_z**2))

        rows.append({
            "range_m":     R,
            "n_shots":     n_shots,
            "mean_impact_x": mean_x,
            "mean_impact_z": mean_z,
            "bias_x_m":    bias_x,
            "bias_z_m":    bias_z,
            "bias_magnitude_m": bias_mag,
            "bias_pct_range":   bias_mag / R * 100,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# TRANSFER LEARNING  [Improvement 4.4]
# ─────────────────────────────────────────────────────────────────

def run_transfer_learning(targets_train: list, targets_test: list,
                           v0: float = V0, seed_a: int = 42,
                           seed_b: int = 999, verbose: bool = True) -> dict:
    """
    Train memory on robot A (seed_a). Test robot B (seed_b) with and
    without the prior from A. Shows whether ARCS generalises across
    robot instances — key for fleet deployment.

    Returns dict with convergence speed comparison.
    """
    import tempfile

    def run_robot(targets, seed, memory=None, label=""):
        tmpdir = Path(tempfile.mkdtemp())
        rt = RangeTable(
            physics_path=str(tmpdir/"phys.csv"),
            corrections_path=str(tmpdir/"corr.csv"))
        rt.generate_physics(
            range_steps=np.arange(50,455,50),
            height_steps=np.arange(-10,35,15),
            v0_steps=np.array([80,100,120]),
            verbose=False, force=True)
        sim = EngagementSimulator(seed=seed, range_table=rt)
        mem = memory if memory is not None else EngagementMemory()
        gmod = GlobalModel()
        bo  = BayesianOptimizer(memory=mem, global_model=gmod,
                                 n_init=4, n_suggest=8,
                                 kappa=2.0, kappa_min=0.5)
        results = []
        for i, (tx, ty, tz) in enumerate(targets):
            r = sim.run_engagement(tx, ty, tz, v0=v0, optimizer=bo)
            if r:
                results.append({
                    "engagement": i+1,
                    "label": label,
                    "verified_cep_m": r["verified_cep"],
                    "improvement_pct": r["improvement_pct"],
                    "baseline_cep_m": r["baseline_cep"],
                })
            if (i+1) % 5 == 0 and rt._corrections_df is not None:
                gmod.train(rt._corrections_df)
                bo.global_model = gmod
        return results, mem

    if verbose:
        print("  Training robot A...")
    results_a, memory_a = run_robot(targets_train, seed_a, label="robot_A_train")

    if verbose:
        print("  Testing robot B (no prior)...")
    results_b_cold, _ = run_robot(targets_test, seed_b, memory=None,
                                    label="robot_B_cold")

    if verbose:
        print("  Testing robot B (with prior from A)...")
    results_b_warm, _ = run_robot(targets_test, seed_b, memory=memory_a,
                                    label="robot_B_warm")

    df_a    = pd.DataFrame(results_a)
    df_cold = pd.DataFrame(results_b_cold)
    df_warm = pd.DataFrame(results_b_warm)

    # Convergence speed: engagement number where CEP first drops below threshold
    def convergence_at(df, threshold=5.0):
        for _, row in df.iterrows():
            if row["verified_cep_m"] < threshold:
                return int(row["engagement"])
        return len(df) + 1  # never converged

    return {
        "df_robot_a":    df_a,
        "df_cold":       df_cold,
        "df_warm":       df_warm,
        "cold_converge": convergence_at(df_cold),
        "warm_converge": convergence_at(df_warm),
        "speedup_factor": convergence_at(df_cold) / max(1, convergence_at(df_warm)),
        "cold_final_cep": float(df_cold["verified_cep_m"].mean()),
        "warm_final_cep": float(df_warm["verified_cep_m"].mean()),
    }


# ─────────────────────────────────────────────────────────────────
# CONVERGENCE DETECTION  [Improvement 3.5]
# ─────────────────────────────────────────────────────────────────

def has_converged(records: list, window: int = 5,
                  threshold_pct: float = 2.0) -> bool:
    """
    Returns True if last `window` engagements improved by < threshold_pct.
    Used to stop firing correction shots in deployment (saves ammunition).
    """
    if len(records) < window:
        return False
    recent = [r["improvement_pct"] for r in records[-window:]]
    return abs(float(np.mean(recent))) < threshold_pct


# ─────────────────────────────────────────────────────────────────
# FULL EXPERIMENT RUNNER
# ─────────────────────────────────────────────────────────────────

def run_all_experiments(n_trials: int = N_TRIALS, verbose: bool = True,
                        quick: bool = False):
    """
    Run the complete benchmark experiment suite.

    Args:
        n_trials: Number of independent learning-curve trials (default 5).
        verbose:  Print progress to stdout.
        quick:    When True, reduce targets and trials for a fast smoke-test
                  (~2 minutes instead of ~10). Suitable for CI / development.
                  quick=True uses n_trials=2, 8 targets, 500 Jensen shots.
    """
    if quick:
        n_trials  = min(n_trials, 2)
        n_engage  = 8
        n_jensen  = 200
    else:
        n_engage  = N_ENGAGE
        n_jensen  = 500

    DATA_DIR.mkdir(exist_ok=True)
    targets_fixed = fixed_target_set(seed=99, n=n_engage)
    results = {}

    mode_tag = "QUICK" if quick else "FULL"
    print("=" * 60)
    print(f"ARCS BENCHMARK EXPERIMENT SUITE  [{mode_tag}]")
    print(f"  Targets: {n_engage}  |  Trials: {n_trials}")
    print("=" * 60)

    # ── 1. Jensen bias quantification ──────────────────────────────
    print(f"\n[1/5] Jensen Bias Quantification...")
    t0 = time.time()
    df_jensen = quantify_jensen_bias(ranges=[80,150,200,300,400],
                                      n_shots=n_jensen, seed=42)
    df_jensen.to_csv(DATA_DIR / "experiment_jensen_bias.csv", index=False)
    results["jensen_bias"] = df_jensen.to_dict("records")
    print(f"  Done in {time.time()-t0:.1f}s")
    print(df_jensen[["range_m","bias_x_m","bias_magnitude_m","bias_pct_range"]]
          .to_string(index=False))

    # ── 2. Shot efficiency curve ────────────────────────────────────
    print(f"\n[2/5] Shot Efficiency Curve...")
    t0 = time.time()
    df_shot = run_shot_efficiency(200, 0, 0, v0=V0, max_shots=80, seed=42)
    df_shot.to_csv(DATA_DIR / "experiment_shot_efficiency.csv", index=False)
    results["shot_efficiency_rows"] = len(df_shot)
    print(f"  Done in {time.time()-t0:.1f}s  ({len(df_shot)} data points)")

    # ── 3. Ablation study ──────────────────────────────────────────
    print(f"\n[3/5] Ablation Study...")
    t0 = time.time()
    df_abl = run_ablation(targets_fixed[:min(n_engage, 12)], v0=V0, seed=42, verbose=verbose)
    df_abl.to_csv(DATA_DIR / "experiment_ablation.csv", index=False)
    summary_abl = df_abl.groupby("label")["improvement_pct"].agg(
        ["mean","std","count"]).round(2)
    results["ablation"] = summary_abl.to_dict()
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"\n  Ablation results:")
    print(summary_abl.to_string())

    # ── 4. Multi-trial learning curves ─────────────────────────────
    print(f"\n[4/5] Multi-Trial Learning Curves ({n_trials} trials)...")
    t0 = time.time()
    df_curves = run_learning_curves(targets_fixed, n_trials=n_trials,
                                     v0=V0, verbose=verbose)
    df_curves.to_csv(DATA_DIR / "experiment_learning_curves.csv", index=False)
    # Compute mean ± std per engagement
    summary_curves = df_curves.groupby("engagement").agg(
        mean_cep=("verified_cep_m", "mean"),
        std_cep=("verified_cep_m", "std"),
        mean_imp=("improvement_pct", "mean"),
        std_imp=("improvement_pct", "std"),
    ).round(3)
    results["learning_curves"] = summary_curves.to_dict()
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"\n  Learning curve (mean ± std):")
    for eng, row in list(summary_curves.iterrows())[:5]:
        print(f"    Eng {eng:2d}: CEP={row['mean_cep']:.2f}±{row['std_cep']:.2f}m  "
              f"Imp={row['mean_imp']:+.1f}±{row['std_imp']:.1f}%")
    print(f"    ...")

    # ── 5. Transfer learning ────────────────────────────────────────
    print(f"\n[5/5] Transfer Learning...")
    t0 = time.time()
    targets_train = fixed_target_set(seed=99,  n=15)
    targets_test  = fixed_target_set(seed=200, n=10)
    tr = run_transfer_learning(targets_train, targets_test,
                                 seed_a=42, seed_b=999, verbose=verbose)
    df_tr = pd.concat([tr["df_robot_a"], tr["df_cold"], tr["df_warm"]])
    df_tr.to_csv(DATA_DIR / "experiment_transfer.csv", index=False)
    results["transfer"] = {
        "cold_converge_eng":   tr["cold_converge"],
        "warm_converge_eng":   tr["warm_converge"],
        "speedup_factor":      round(tr["speedup_factor"], 2),
        "cold_final_cep_m":    round(tr["cold_final_cep"], 3),
        "warm_final_cep_m":    round(tr["warm_final_cep"], 3),
    }
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Cold start convergence: eng {tr['cold_converge']}")
    print(f"  Warm start convergence: eng {tr['warm_converge']}")
    print(f"  Transfer speedup: {tr['speedup_factor']:.1f}x")

    # ── Summary JSON ──────────────────────────────────────────────
    summary_json = {
        "experiment_date": time.strftime("%Y-%m-%d %H:%M"),
        "n_targets":       N_ENGAGE,
        "n_trials":        n_trials,
        "v0_ms":           V0,
        "improvements_summary": results.get("ablation", {}),
        "transfer_speedup": results.get("transfer", {}).get("speedup_factor"),
        "jensen_bias_at_200m_pct": next(
            (r["bias_pct_range"] for r in results.get("jensen_bias",[])
             if r.get("range_m") == 200), None),
    }
    with open(DATA_DIR / "experiment_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"  Outputs saved to data/experiment_*.csv")
    print(f"  Summary: data/experiment_summary.json")
    print(f"{'='*60}")

    return results


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, argparse as _ap
    _parser = _ap.ArgumentParser(description="ARCS Benchmark Experiment Suite")
    _parser.add_argument("--trials", type=int, default=N_TRIALS,
                         help=f"Number of learning-curve trials (default {N_TRIALS})")
    _parser.add_argument("--quick",  action="store_true",
                         help="Fast smoke-test mode: 2 trials, 8 targets (~2 min)")
    _args = _parser.parse_args()
    run_all_experiments(n_trials=_args.trials, verbose=True, quick=_args.quick)