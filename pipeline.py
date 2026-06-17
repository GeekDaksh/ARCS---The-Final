"""
ARCS — Full Pipeline  (PINN Edition v3.0)
Phase 1

CHANGES FROM v2.0 (Benchmark Edition):
    [PINN] PINNCorrector replaces the old correction formula.
           All pipeline logic is unchanged.
           Physics-informed loss now constrains learned corrections.
    [PINN] MIN_RECORDS reduced 30 → 20 (physics constraint compensates).
    [PINN] _maybe_retrain_gp() renamed _maybe_retrain_pinn() (cosmetic).

RETAINED FROM v2.0:
    [1.2] Decaying kappa propagated through from BayesianOptimizer v2.
    [1.3] Bootstrap CI recorded in tracker and results dict.
    [2.1] Info-gain retraining trigger (should_retrain() on PINNCorrector).
    [2.2] Adaptive bounds via BayesianOptimizer v2.
    [2.4] Adaptive n_avg accounts for PINN pre-correction activity.
    [3.5] _has_converged() halts engagement loop on improvement plateau.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from physics.ballistic_solver import BallisticSolver
from physics.range_table      import RangeTable
from bayesian_optimizer       import (BayesianOptimizer, EngagementMemory,
                                       EngagementSimulator, GlobalModel)
from pinn_corrector           import PINNCorrector
from structured_bias_estimator import StructuredBiasEstimator
from engagement_database      import EngagementDatabase
from metrics                  import ConvergenceTracker
from experiment               import has_converged


class ARCSPipeline:
    """
    Full ARCS Phase 1 pipeline — PINN Edition.

    Intelligence stack per engagement:
        1. Physics Engine   → exact Newton solution (starting point)
        2. PINN Corrector   → global systematic-bias pre-correction
        3. Bayesian Optimiser → per-engagement fine-tuning (fewer shots needed)
        4. Range Table       → weighted correction lookup from past engagements

    The PINN replaces symbolic/polynomial regression. It trains a small neural
    network (6,496 parameters) constrained by the Galileo range equation so
    that its predicted corrections are always physically valid.
    """

    def __init__(self,
                 physics_path:     str   = "data/range_table_physics.csv",
                 corrections_path: str   = "data/range_table_corrections.csv",
                 history_path:     str   = "data/metrics_history.csv",
                 db_path:          str   = None,
                 v0:               float = 100.0,
                 seed:             int   = None,
                 verbose:          bool  = True):

        self.v0      = v0
        self.verbose = verbose

        if seed is not None:
            np.random.seed(seed)
        self._seed = seed

        # ── Range table (physics + corrections) ───────────────────────
        self.rt = RangeTable(physics_path, corrections_path)
        if not Path(physics_path).exists():
            self._generate_physics_table()
        else:
            self.rt.load(verbose=verbose)

        # ── Bayesian Optimizer (per-engagement search) ─────────────────
        self.memory = EngagementMemory()
        self.gmodel = GlobalModel()
        self.bo     = BayesianOptimizer(
            memory       = self.memory,
            global_model = self.gmodel,
            n_avg=3, n_init=4, n_suggest=16,   # 16 matches standalone benchmark
            kappa=2.0, kappa_min=0.5,          # decaying GP-UCB schedule
        )

        # ── GlobalModel warm-start ────────────────────────────────────
        if self.rt._corrections_df is not None and len(self.rt._corrections_df) > 0:
            self.gmodel.train(self.rt._corrections_df)
            self.bo.global_model = self.gmodel

        # ── Engagement simulator ───────────────────────────────────────
        self.sim = EngagementSimulator(seed=seed, range_table=self.rt)

        # ── PINN Correctors (one per trajectory type) ──────────────────
        # Two separate correctors because LOW and HIGH trajectories have
        # different physics (different pitch angles → different gravity sag).
        self.cf_low  = PINNCorrector(corrections_path, solution_type="LOW")
        self.cf_high = PINNCorrector(corrections_path, solution_type="HIGH")
        self.cf_low.load_and_train(verbose=verbose)
        self.cf_high.load_and_train(verbose=verbose)

        # ── Structured Bias Estimator + Engagement Database ───────────
        self.sbe = StructuredBiasEstimator()
        self.db  = (EngagementDatabase(db_path=Path(db_path))
                    if db_path is not None else EngagementDatabase())

        # Warm-start SBE from existing database on restart
        for sbe_input in self.db.get_sbe_inputs():
            self.sbe.update_engagement(**sbe_input)
        n_eng_db = self.db.statistics()['n_engagements']
        if n_eng_db > 0 and verbose:
            print(f"  SBE warm-started from {n_eng_db} DB records: {self.sbe.summary()}")

        # ── Convergence tracker ────────────────────────────────────────
        self.tracker      = ConvergenceTracker(history_path)
        self.engagement_n = len(self.tracker._records)

        if verbose:
            s = self.rt.stats()
            print(f"\n  Pipeline ready (PINN Edition v3.0 + SBE).")
            print(f"  Physics entries     : {s.get('physics_reachable',0):,}")
            print(f"  HIGH-angle entries  : {s.get('physics_high_solutions',0):,}")
            if s.get('physics_high_solutions', 0) == 0:
                print(f"  ⚠  HIGH table empty — run: python rebuild_physics.py")
            print(f"  Correction records  : {s.get('corrections_total',0)}")
            print(f"  Engagements history : {len(self.tracker._records)}")
            print(f"  DB engagements      : {self.db.statistics()['n_engagements']}")
            print(f"  PINN LOW fitted     : {self.cf_low.is_fitted}")
            print(f"  PINN HIGH fitted    : {self.cf_high.is_fitted}")
            print(f"  SBE confidence      : {self.sbe.confidence():.2f}")
            print(f"  BO kappa schedule   : 2.0 → 0.5 (GP-UCB decaying)")

    # ─── Physics table generation ─────────────────────────────────────────────

    def _generate_physics_table(self):
        """Generate the physics lookup table on first run."""
        print("  Generating physics table (first run — takes ~30s)...")
        self.rt.generate_physics(
            range_steps  = np.arange(10, 505, 5),
            height_steps = np.arange(-20, 55, 5),
            v0_steps     = np.arange(50, 305, 10),
            verbose      = self.verbose,
            force        = True,
        )
        # Ensure _corrections_df is initialized (empty but typed) so it is
        # never None even if every engagement in the first batch triggers
        # the safe-fallback and record_correction() is never called.
        self.rt.load_corrections(verbose=False)

    # ─── PINN retraining (information-gain trigger) ───────────────────────────

    def _maybe_retrain_pinn(self):
        """
        Retrain PINN correctors when the info-gain trigger fires.

        Trigger: ≥5 new records exist AND they cover a novel range region
        (more than 30m from anything seen during the last training run).

        Renamed from _maybe_retrain_gp() — identical logic, different corrector.
        Also refreshes the GlobalModel used by the Bayesian Optimizer.
        """
        low_retrain  = self.cf_low.should_retrain()
        high_retrain = self.cf_high.should_retrain()

        if low_retrain or high_retrain:
            if self.verbose:
                print(f"  [PINN Retrain] LOW={low_retrain} HIGH={high_retrain} "
                      f"(info-gain trigger)")
            if low_retrain:
                self.cf_low.load_and_train(verbose=False)
            if high_retrain:
                self.cf_high.load_and_train(verbose=False)

            # Also refresh the GlobalModel (used by BO for cross-engagement priors)
            if self.rt._corrections_df is not None:
                self.gmodel.train(self.rt._corrections_df)
                self.bo.global_model = self.gmodel

    # ─── Single engagement ────────────────────────────────────────────────────

    def engage(self, target_x: float, target_y: float,
               target_z: float, label: str = "",
               prefer: str = "LOW") -> dict:
        """
        Run one full engagement at a target.

        Flow:
          1. Physics solver → exact aim angles (θ, φ, v₀)
          2. Range table lookup → historical weighted correction
          3. PINN → pre-correction (if fitted, ≥20 records seen)
          4. Bayesian Optimizer → fire N shots, search for best correction
          5. Record result → update correction table + PINN training data

        Returns result dict including:
            verified_cep, ci_low, ci_high
            trajectory_type, pinn_correction, rt_correction
        """
        self.engagement_n += 1

        if self.verbose:
            print(f"\n{'─'*60}")
            print(f"  Engagement {self.engagement_n:3d}  "
                  f"target=({target_x:.0f},{target_y:.0f},{target_z:.0f})  "
                  f"prefer={prefer}")

        # Step 1: Physics solution
        solver = BallisticSolver()
        sol    = solver.solve(target_x, target_y, target_z, self.v0, prefer=prefer)
        if not sol.reachable:
            print(f"  [WARNING] Engagement {self.engagement_n} skipped: "
                  f"unreachable — {sol.error_message}")
            return None

        # Step 2: Range table lookup (historical corrections)
        rt_lookup        = self.rt.lookup(sol.horiz_range, target_y, self.v0,
                                          prefer=prefer)
        effective_prefer = rt_lookup["solution_type"]

        # Step 3: Pre-correction — SBE when confident, PINN otherwise
        cf   = self.cf_high if effective_prefer == "HIGH" else self.cf_low
        sbe_conf = self.sbe.confidence()

        if sbe_conf >= 0.6:
            # SBE has enough history — use it directly
            pinn_correction = self.sbe.predict(sol.turret_pitch_deg, self.v0)
        elif sbe_conf > 0.2 and cf.is_fitted:
            # Blend SBE and PINN by SBE confidence
            pinn_pred = cf.predict(sol.horiz_range, target_y, self.v0)
            sbe_pred  = self.sbe.predict(sol.turret_pitch_deg, self.v0)
            alpha = sbe_conf
            pinn_correction = {
                "delta_pitch": alpha * sbe_pred["delta_pitch"] + (1 - alpha) * pinn_pred["delta_pitch"],
                "delta_yaw":   alpha * sbe_pred["delta_yaw"]   + (1 - alpha) * pinn_pred["delta_yaw"],
                "delta_v0":    alpha * sbe_pred["delta_v0"]    + (1 - alpha) * pinn_pred["delta_v0"],
                "source":      f"blend(sbe={sbe_conf:.2f})",
            }
        elif cf.is_fitted:
            pinn_correction = cf.predict(sol.horiz_range, target_y, self.v0)
        else:
            pinn_correction = {"delta_pitch": 0.0, "delta_yaw": 0.0,
                               "delta_v0": 0.0, "source": "none"}

        if self.verbose:
            print(f"  Range={sol.horiz_range:.0f}m  "
                  f"Physics ({effective_prefer}): "
                  f"θ={sol.turret_pitch_deg:.2f}°  φ={sol.turret_yaw_deg:.2f}°")
            print(f"  RT correction  : "
                  f"dp={rt_lookup['delta_pitch']:+.3f}°  "
                  f"n_obs={rt_lookup['n_observations']}  "
                  f"type={effective_prefer}")
            print(f"  Pre-correction : "
                  f"dp={pinn_correction['delta_pitch']:+.3f}°  "
                  f"source={pinn_correction['source']}  "
                  f"SBE_conf={sbe_conf:.2f}")

        # Step 4: Bayesian Optimizer engagement
        result = self.sim.run_engagement(
            target_x, target_y, target_z,
            v0              = self.v0,
            optimizer       = self.bo,
            verbose         = self.verbose,
            gp_pre_correction = pinn_correction if cf.is_fitted else None,
            prefer          = effective_prefer,
        )
        if result is None:
            return None

        # Step 5: Record result
        self.tracker.record(
            engagement_n    = self.engagement_n,
            baseline_cep    = result["baseline_cep"],
            bo_cep          = result["verified_cep"],
            gp_applied      = cf.is_fitted,
            target_range    = sol.horiz_range,
            improvement_pct = result["improvement_pct"],
            n_shots_used    = result["total_shots"],
            label           = label,
            solution_type   = effective_prefer,
        )

        # Trigger PINN retraining if info-gain threshold reached
        self._maybe_retrain_pinn()

        # Log engagement to persistent database
        # NOTE: run_engagement() returns 'best_correction' as a numpy array
        # [delta_pitch, delta_yaw, delta_v0] — NOT a dict keyed
        # 'best_correction_total'. Convert it to a dict here so downstream
        # .get('delta_pitch'/...) calls (SBE update, DB sbe_input) actually
        # see real values instead of always None.
        best_corr_arr = result.get('best_correction')
        if best_corr_arr is not None:
            best_corr = {
                'delta_pitch': float(best_corr_arr[0]),
                'delta_yaw':   float(best_corr_arr[1]),
                'delta_v0':    float(best_corr_arr[2]),
            }
        else:
            best_corr = {}
        rls_converged = bool(result.get('rls_converged', False))

        sbe_pred = self.sbe.predict(sol.turret_pitch_deg, self.v0)

        self.db.log({
            'target': {
                'x': target_x, 'y': target_y, 'z': target_z,
                'range': sol.horiz_range, 'bearing_deg': sol.turret_yaw_deg,
                'height_m': target_y, 'v0_mps': self.v0,
            },
            'nominal_solution': {
                'pitch_deg': sol.turret_pitch_deg, 'yaw_deg': sol.turret_yaw_deg,
                'tof': sol.tof, 'v0': self.v0,
            },
            'results': {
                'baseline_cep_m':  result.get('baseline_cep'),
                'corrected_cep_m': result.get('verified_cep'),
                'improvement_pct': result.get('improvement_pct'),
                'best_correction': best_corr,
            },
            'estimator': {
                'forgetting_rls_db_final': result.get('rls_db'),
                'forgetting_rls_dv_final': result.get('rls_dv'),
                'n_shots_bo':              result.get('total_shots'),
                'converged':               rls_converged,
            },
            'sbe_input': {
                'pitch_deg_nominal': sol.turret_pitch_deg,
                'dp_opt': best_corr.get('delta_pitch'),
                'db_opt': best_corr.get('delta_yaw'),
                'dv_opt': best_corr.get('delta_v0'),
            },
            'confidence': {
                'gp_sigma_m':          result.get('gp_sigma_m'),
                'gp_converged':        result.get('gp_converged'),
                'n_bo_shots':          result.get('n_bo_shots'),
                'bo_early_stopped':    result.get('bo_early_stopped'),
                'bo_early_stop_iter':  result.get('bo_early_stop_iter'),
                'rls_sigma_db_deg':    result.get('rls_sigma_db_deg'),
                'rls_sigma_dv_ms':     result.get('rls_sigma_dv_ms'),
                'rls_db_final':        result.get('rls_db_final'),
                'rls_dv_final':        result.get('rls_dv_final'),
            },
            'sbe_output': {
                'b_sag':       sbe_pred.get('b_sag'),
                'b_yaw':       sbe_pred.get('b_yaw'),
                'b_v0':        sbe_pred.get('b_v0'),
                'b_sag_ci90':  sbe_pred.get('b_sag_ci90'),
                'b_yaw_ci90':  sbe_pred.get('b_yaw_ci90'),
                'b_v0_ci90':   sbe_pred.get('b_v0_ci90'),
                'confidence':  sbe_pred.get('confidence'),
            },
        })

        # Update SBE with this engagement's outcome
        if best_corr:
            self.sbe.update_engagement(
                pitch_deg = sol.turret_pitch_deg,
                db_opt    = best_corr.get('delta_yaw', 0.0),
                dv_opt    = best_corr.get('delta_v0', 0.0),
                dp_opt    = best_corr.get('delta_pitch', None),
            )

        if self.verbose:
            imp   = result["improvement_pct"]
            arrow = "↑" if imp > 0 else "↓"
            ci    = (f"CI=[{result.get('ci_low', 0):.2f},"
                     f"{result.get('ci_high', 0):.2f}]")
            print(f"  Result : base={result['baseline_cep']:.2f}m  "
                  f"verified={result['verified_cep']:.2f}m  {ci}  "
                  f"{arrow}{abs(imp):.1f}%  shots={result['total_shots']}")
            print(f"  SBE    : {self.sbe.summary()}")
            self._print_engagement_summary(self.engagement_n, result, sbe_pred)

        result["pinn_correction"]  = pinn_correction
        result["rt_correction"]    = rt_lookup
        result["engagement_n"]     = self.engagement_n
        result["trajectory_type"]  = effective_prefer
        return result

    # ─── Batch engagements ────────────────────────────────────────────────────

    def run_batch(self, targets: list,
                  label:              str  = "",
                  prefer:             str  = "LOW",
                  stop_on_convergence: bool = False) -> pd.DataFrame:
        """
        Run engagements on a list of (x, y, z) targets.

        stop_on_convergence: halt the batch early when the improvement plateau
        is detected — saves ammunition in real deployment (Improvement 3.5).
        """
        rows = []
        for i, (tx, ty, tz) in enumerate(targets):
            if stop_on_convergence and has_converged(self.tracker._records):
                if self.verbose:
                    print(f"  [Convergence] System converged after "
                          f"{i} engagements — halting batch.")
                break

            r = self.engage(tx, ty, tz, label=label, prefer=prefer)
            if r:
                rows.append({
                    "engagement_n":    r["engagement_n"],
                    "target":          f"({tx:.0f},{ty:.0f},{tz:.0f})",
                    "range_m":         r["horiz_range"],
                    "baseline_cep_m":  r["baseline_cep"],
                    "verified_cep_m":  r["verified_cep"],
                    "ci_low":          r.get("ci_low", r["verified_cep"]),
                    "ci_high":         r.get("ci_high", r["verified_cep"]),
                    "improvement_pct": r["improvement_pct"],
                    "shots_used":      r["total_shots"],
                    "pinn_applied":    self.cf_low.is_fitted or self.cf_high.is_fitted,
                    "trajectory_type": r.get("trajectory_type", "LOW"),
                })
        return pd.DataFrame(rows)

    # ─── Confidence reporting ─────────────────────────────────────────────────

    def _print_engagement_summary(self, eng_n: int, result: dict,
                                   sbe_pred: dict) -> None:
        """Print the full engagement result including all confidence signals."""
        imp   = result.get('improvement_pct', float('nan'))
        gp_s  = result.get('gp_sigma_m', float('nan'))
        rls_c = result.get('rls_converged', False)
        gp_c  = result.get('gp_converged', False)
        sbe_c = sbe_pred.get('confidence', 0.0)

        conv_flags = []
        if gp_c:         conv_flags.append('BO✓')
        if rls_c:        conv_flags.append('RLS✓')
        if sbe_c >= 0.9: conv_flags.append('SBE✓')
        conv_str = ' '.join(conv_flags) if conv_flags else 'converging'

        print(f"\n  Engagement {eng_n:>3d} │ "
              f"Improvement: {imp:>+6.1f}% │ "
              f"GP σ={gp_s:.2f}m │ "
              f"SBE conf={sbe_c:.2f} │ {conv_str}")
        print(f"            │ "
              f"b_sag={sbe_pred.get('b_sag', 0):.4f} "
              f"b_yaw={sbe_pred.get('b_yaw', 0):+.4f}° "
              f"b_v0={sbe_pred.get('b_v0', 0):+.3f}m/s")
        if 'b_v0_ci90' in sbe_pred:
            ci = sbe_pred['b_v0_ci90']
            print(f"            │ "
                  f"v0 bias 90% CI: [{ci[0]:+.3f}, {ci[1]:+.3f}] m/s")

    def _print_final_report(self, all_results: list = None) -> None:
        """Print the aggregate confidence/CEP report — what an industry
        evaluator will read first when judging Phase 1 completion."""
        stats = self.db.statistics()
        print("\n" + "═" * 60)
        print("ARCS Phase 1 — Final Report")
        print("═" * 60)
        print(f"  Engagements:          {stats['n_engagements']}")
        if stats.get('mean_improvement_pct') is not None:
            print(f"  Mean improvement:     {stats['mean_improvement_pct']:+.1f}%")
            print(f"  Mean baseline CEP:    {stats['mean_baseline_cep']:.2f} m")
            print(f"  Mean corrected CEP:   {stats['mean_corrected_cep']:.2f} m")
        print(f"  SBE confidence:       {self.sbe.confidence():.2f}")
        pred = self.sbe.predict(pitch_deg=8.6)
        print(f"  b_sag:  {pred['b_sag']:+.4f}  90% CI {pred['b_sag_ci90']}")
        print(f"  b_yaw:  {pred['b_yaw']:+.4f}°  90% CI {pred['b_yaw_ci90']}")
        print(f"  b_v0:   {pred['b_v0']:+.3f} m/s  90% CI {pred['b_v0_ci90']}")
        print("═" * 60)

    # ─── Status report ────────────────────────────────────────────────────────

    def status(self):
        s  = self.rt.stats()
        ts = self.tracker.summary()
        print(f"\n{'='*60}")
        print(f"  ARCS PIPELINE STATUS (PINN Edition v3.0)")
        print(f"{'='*60}")
        print(f"  Physics entries    : {s.get('physics_reachable', 0):,}")
        print(f"  HIGH-angle entries : {s.get('physics_high_solutions', 0):,}")
        if s.get('physics_high_solutions', 0) == 0:
            print(f"  ⚠  HIGH table empty — run: python rebuild_physics.py")
        print(f"  Corrections stored : {s.get('corrections_total', 0)}")
        print(f"    LOW-angle corr   : {s.get('corrections_low', 0)}")
        print(f"    HIGH-angle corr  : {s.get('corrections_high', 0)}")
        print(f"  Engagements total  : {ts.get('n_engagements', 0)}")
        print(f"  PINN LOW  fitted   : {self.cf_low.is_fitted} "
              f"({self.cf_low.n_records} records)")
        print(f"  PINN HIGH fitted   : {self.cf_high.is_fitted} "
              f"({self.cf_high.n_records} records)")
        print(f"  Global model fitted: {self.gmodel.is_fitted}")
        print(f"  SBE                : {self.sbe.summary()}")
        print(f"  DB engagements     : {self.db.statistics()['n_engagements']}")
        print(f"  BO kappa schedule  : 2.0 → 0.5 (GP-UCB decaying)")
        if ts.get("n_engagements", 0) > 0:
            print(f"  Mean improvement   : {ts['mean_improvement']:+.1f}%")
            print(f"  Best verified CEP  : {ts['best_bo_miss']:.2f}m")
            print(f"  Positive eng       : {ts['pct_positive']:.0f}%")
            print(f"  Converged?         : "
                  f"{'YES' if has_converged(self.tracker._records) else 'NO'}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SEED = 42
    print("=" * 60)
    print("ARCS Phase 1 — Full Pipeline (PINN Edition v3.0)")
    print("Physics-Informed NN | Decaying kappa | Bootstrap CI")
    print("=" * 60)

    pipeline = ARCSPipeline(seed=SEED, verbose=True)
    rng      = np.random.default_rng(SEED)

    # Batch 1 — LOW trajectory
    batch1 = [(float(rng.uniform(80, 420)), float(rng.uniform(-10, 30)),
               float(rng.uniform(-150, 150))) for _ in range(20)]
    print(f"\n  Running batch 1 (LOW angle, {len(batch1)} engagements)...")
    df1 = pipeline.run_batch(batch1, label="batch1_low",
                              prefer="LOW", stop_on_convergence=False)

    # Batch 2 — HIGH trajectory
    batch2 = [(float(rng.uniform(80, 420)), float(rng.uniform(-10, 30)),
               float(rng.uniform(-150, 150))) for _ in range(10)]
    print(f"\n  Running batch 2 (HIGH angle, {len(batch2)} engagements)...")
    df2 = pipeline.run_batch(batch2, label="batch2_high", prefer="HIGH")

    # Summary
    for lbl, df in [("LOW", df1), ("HIGH", df2)]:
        if len(df):
            print(f"\n  Batch ({lbl}) summary:")
            print(f"    Mean baseline   : {df['baseline_cep_m'].mean():.2f}m")
            print(f"    Mean verified   : {df['verified_cep_m'].mean():.2f}m")
            print(f"    Mean CI width   : {(df['ci_high']-df['ci_low']).mean():.2f}m")
            print(f"    Mean improve    : {df['improvement_pct'].mean():+.1f}%")

    pipeline.status()
    pipeline.tracker.print_learning_curve()
    pipeline._print_final_report()

    Path("data").mkdir(exist_ok=True)
    pd.concat([df1, df2]).to_csv("data/pipeline_results_v3.csv", index=False)
    print(f"\n  Saved → data/pipeline_results_v3.csv")
    print(f"\n  pipeline.py v3.0 complete")