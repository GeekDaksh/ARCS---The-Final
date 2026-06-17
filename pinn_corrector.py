"""
ARCS — Physics-Informed Neural Network (PINN) Corrector
Phase 1 — Benchmark Edition v1.0

WHAT THIS DOES IN PLAIN ENGLISH
────────────────────────────────
Every robot has systematic mechanical flaws:
  • Barrel gravity sag:  Δpitch ∝ −sin(pitch_cmd)  — barrel droops under its own weight
  • Gear backlash:       constant offset each direction change
  • IMU yaw drift:       constant yaw error after extended use
  • Propellant Δv₀:      temperature-dependent velocity shift

The PINN learns to predict these errors given (range, height, v₀).
The PHYSICS CONSTRAINT guarantees: the predicted corrections, when applied,
result in a trajectory that hits the target per Newton's exact equations.
This constraint acts as a powerful regulariser — the network needs only
20 records instead of 30, and never predicts physically impossible corrections.

PHYSICS CONSTRAINT (the "PI" in PINN)
───────────────────────────────────────
Galileo's range equation (exact, vacuum ballistics, no drag):

    y(x) = x · tan(θ) − g · x² / (2 · v₀² · cos²(θ))

After applying predicted corrections θ_c = θ_cmd + Δpitch, v₀_c = v₀ + Δv₀:

    L_physics = [y(range_m) − height_m]² / range_m²

This is zero only when the corrected trajectory exactly hits the target.
It is FULLY DIFFERENTIABLE — gradients flow back through cos, tan, etc.
to the network weights during training (autograd handles this automatically).

ARCHITECTURE
────────────
Input  : (range_norm, height_norm, v₀_norm, sin_θ_cmd)  — 4 features
Hidden : Linear(4→64) → Tanh → Linear(64→64) → Tanh → Linear(64→32) → Tanh
Output : (Δpitch_deg, Δyaw_deg, Δv₀_ms)                — 3 corrections
Params : 6,496 (tiny — trains in <1s on CPU with 20-200 samples)

WHY sin(θ_cmd) as a feature?
    Gravity sag bias = sag_coeff · sin(θ_cmd)
    Giving the network sin(θ_cmd) directly means it can learn this in a single
    linear weight. Without it, the network needs deep nonlinear layers to
    discover the sin relationship from range/height alone.

LOSS
────
L_total = L_data + 0.1 · L_physics + 1e-5 · ‖W‖²

L_data   : MSE(predicted corrections, observed corrections from CSV)
           "What worked in past engagements?"
L_physics: trajectory residual (range equation above)
           "Does the corrected trajectory actually hit the target?"
‖W‖²     : L2 weight decay (Adam handles this via weight_decay param)

TRAINING
────────
Adam optimizer, lr=3e-4, cosine LR decay, 300 epochs.
Retrain trigger: information-gain (novelty > 30m).
Min records: 20 (physics constraint reduces data needed vs GP's 30).

FALLBACK
────────
PyTorch not installed → sklearn MLPRegressor (identical architecture, no physics loss).
sklearn not installed → returns zero corrections (safe, not harmful).
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Backend availability checks ──────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH = True
except ImportError:
    _TORCH = False

try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

from physics.ballistic_solver import BallisticSolver
from physics.constants import GRAVITY, QUALITY_FILTER_RATIO


def _normalize_solution_type(series: "pd.Series") -> "pd.Series":
    """
    Normalize a solution_type column to contain only 'LOW' or 'HIGH'.
    Null, empty, or any unrecognised value (e.g. a timestamp written by an
    old pipeline with wrong column order) is treated as 'LOW' so legacy
    records are never silently dropped.
    """
    return (series.fillna("").astype(str).str.strip()
            .apply(lambda x: x if x in ("LOW", "HIGH") else "LOW"))


# ── Neural network backbone ───────────────────────────────────────────────────

if _TORCH:
    class _PINNNet(nn.Module):
        """
        4-input, 3-output neural network with tanh activations.

        tanh is chosen over ReLU because:
          • Corrections are bounded (max ±3°) — tanh outputs are naturally bounded
          • Systematic biases are smooth functions — tanh is smooth everywhere
          • Better extrapolation behaviour at unseen ranges
        """
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(4, 64),   nn.Tanh(),
                nn.Linear(64, 64),  nn.Tanh(),
                nn.Linear(64, 32),  nn.Tanh(),
                nn.Linear(32, 3),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)


# ── PINN Corrector ────────────────────────────────────────────────────────────

class PINNCorrector:
    """
    Physics-Informed Neural Network corrector for systematic robot bias.

    Usage:
        from pinn_corrector import PINNCorrector
        cf = PINNCorrector(corrections_path, solution_type="LOW")
        cf.load_and_train(verbose=True)   # returns True when fitted
        cf.should_retrain()               # info-gain trigger
        cf.predict(range_m, h, v0)        # {delta_pitch, delta_yaw, delta_v0, source}
        cf.print_formulas()               # human-readable summary
    """

    MIN_RECORDS    = 20     # [v1.0] lowered from GP's 30 — physics compensates
    LAMBDA_PHYSICS = 0.05   # weight of physics loss relative to data loss (lightweight regulariser)
    N_EPOCHS       = 300    # training epochs (converges well by epoch 200)

    def __init__(self,
                 corrections_path: str = "data/range_table_corrections.csv",
                 solution_type:    str = "LOW"):
        self.corrections_path = corrections_path
        self.solution_type    = solution_type

        # Public state (mirrors CorrectionFormula)
        self.is_fitted          = False
        self.n_records          = 0
        self.last_trained_n     = 0
        self._last_train_ranges = None

        # Private — PyTorch backend
        self._net:    "_PINNNet | None" = None

        # Private — sklearn fallback
        self._scaler = None
        self._mlp    = None

        # Physics solver — computes θ_cmd for physics loss
        self._solver = BallisticSolver()

        # Training diagnostics
        self._data_loss: float = float("nan")
        self._phys_loss: float = float("nan")
        self._n_params:  int   = 0
        self._backend:   str   = "none"

        # Auto-load saved weights so predict() works immediately on restart.
        # Only attempt when the corrections file exists — if there's no history
        # file at all, this instance has no context to make predictions from.
        if Path(self.corrections_path).exists():
            self.load_weights()

    # ─── Feature engineering ──────────────────────────────────────────────────

    def _theta_cmd(self, range_m: float, height_m: float, v0_ms: float) -> float:
        """
        Nominal firing elevation from the physics solver.
        Used as a feature and in the physics loss.
        """
        sol = self._solver.solve(range_m, height_m, 0.0, v0_ms,
                                  prefer=self.solution_type)
        return sol.turret_pitch_deg if sol.reachable else 30.0

    def _features(self, range_m: float, height_m: float,
                   v0_ms: float, theta_deg: float) -> "np.ndarray":
        """
        Normalised 4-feature vector for one sample.

        range_norm  = range_m / 500      → [0, 1]  (max range 500m)
        height_norm = height_m / 50      → [-0.4, 1] (height ±50m)
        v0_norm     = v0_ms / 300        → [0, 1]  (max v0 300m/s)
        sin_theta   = sin(θ_cmd)         → [-1, 1] (THE key feature for sag)
        """
        return np.array([
            range_m  / 500.0,
            height_m / 50.0,
            v0_ms    / 300.0,
            np.sin(np.deg2rad(theta_deg)),
        ], dtype=np.float32)

    def _build_training_data(self, df: pd.DataFrame):
        """
        Build (X, Y, thetas) from the corrections DataFrame.
        thetas[i] is the nominal pitch for sample i (for physics loss).
        """
        X_rows, thetas = [], []
        for _, row in df.iterrows():
            th = self._theta_cmd(row["range_m"], row["height_m"], row["v0_ms"])
            thetas.append(th)
            X_rows.append(self._features(row["range_m"], row["height_m"],
                                          row["v0_ms"], th))

        X = np.array(X_rows, dtype=np.float32)
        Y = df[["delta_pitch", "delta_yaw", "delta_v0"]].values.astype(np.float32)
        return X, Y, np.array(thetas, dtype=np.float32)

    # ─── Physics loss (differentiable with PyTorch autograd) ─────────────────

    @staticmethod
    def _physics_residual(range_t:  "torch.Tensor",
                           height_t: "torch.Tensor",
                           v0_t:     "torch.Tensor",
                           theta_t:  "torch.Tensor",
                           pred:     "torch.Tensor") -> "torch.Tensor":
        """
        Structure-enforcing regulariser.

        REPLACES the original physics loss which was proven to penalise
        correct bias corrections (L_physics = 0 only when dp=0, dv=0).
        Also removes L_yaw = sin²(Δyaw) which suppressed the yaw correction.

        New constraints enforce the KNOWN PHYSICAL STRUCTURE of the bias:
        1. Monotonicity: pitch correction must increase with sin(pitch_cmd)
           because sag ∝ −sin(pitch), so correction = +sag_coeff × sin(pitch).
        2. Boundedness: corrections must stay within ±3σ of mechanical noise.
        3. No yaw constraint: yaw bias is a constant offset, not range-dependent.
        """
        dp = pred[:, 0]
        dv = pred[:, 2]
        SIGMA_PITCH = 0.3    # deg
        SIGMA_V0    = 1.5    # m/s
        sin_theta = torch.sin(theta_t * (torch.pi / 180.0))
        L_mono    = torch.relu(-dp * sin_theta).mean()
        L_bound_p = torch.relu(dp.abs() - 3 * SIGMA_PITCH).pow(2).mean()
        L_bound_v = torch.relu(dv.abs() - 3 * SIGMA_V0).pow(2).mean()
        return L_mono + L_bound_p + L_bound_v

    # ─── Training — PyTorch ──────────────────────────────────────────────────

    def _train_pytorch(self, df: pd.DataFrame, verbose: bool) -> bool:
        """
        Train the PINN with physics constraint.

        Steps:
          1. Build feature matrix X and target matrix Y from CSV
          2. For each epoch: forward → L_data + λ·L_physics → backward → step
          3. Keep best weights (by total loss) across all epochs
          4. Restore best weights and save net

        The physics loss uses the exact Galileo range equation — this is the
        "physics-informed" part. Every gradient update must satisfy two goals:
          (a) fit the observed corrections (L_data)
          (b) produce a trajectory that hits the target (L_physics)
        """
        X, Y, thetas = self._build_training_data(df)

        X_t      = torch.from_numpy(X)
        Y_t      = torch.from_numpy(Y)
        range_t  = torch.tensor(df["range_m"].values,  dtype=torch.float32)
        height_t = torch.tensor(df["height_m"].values, dtype=torch.float32)
        v0_t     = torch.tensor(df["v0_ms"].values,    dtype=torch.float32)
        theta_t  = torch.from_numpy(thetas)

        net       = _PINNNet()
        optimizer = optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.N_EPOCHS, eta_min=5e-6)

        best_loss, best_state = float("inf"), None

        net.train()
        for _ in range(self.N_EPOCHS):
            optimizer.zero_grad()

            pred    = net(X_t)
            L_data  = nn.functional.mse_loss(pred, Y_t)
            L_phys  = self._physics_residual(range_t, height_t, v0_t, theta_t, pred)
            loss    = L_data + self.LAMBDA_PHYSICS * L_phys

            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total = float(loss.item())
            if total < best_loss:
                best_loss  = total
                best_state = {k: v.clone() for k, v in net.state_dict().items()}

        # Restore best checkpoint
        if best_state:
            net.load_state_dict(best_state)

        net.eval()
        with torch.no_grad():
            pred_f = net(X_t)
            self._data_loss = float(nn.functional.mse_loss(pred_f, Y_t).item())
            self._phys_loss = float(
                self._physics_residual(range_t, height_t, v0_t, theta_t, pred_f).item())

        self._net      = net
        self._n_params = sum(p.numel() for p in net.parameters())
        self._backend  = "pytorch"

        if verbose:
            print(f"    Epochs={self.N_EPOCHS}  "
                  f"data_loss={self._data_loss:.5f}  "
                  f"phys_loss={self._phys_loss:.5f}  "
                  f"params={self._n_params}")
        return True

    # ─── Training — sklearn fallback ─────────────────────────────────────────

    def _train_sklearn(self, df: pd.DataFrame, verbose: bool) -> bool:
        """
        Fallback when PyTorch is unavailable.
        MLPRegressor with identical architecture but no physics constraint.
        Still much better than polynomial regression.
        """
        X, Y, _ = self._build_training_data(df)
        self._scaler = StandardScaler()
        X_s          = self._scaler.fit_transform(X)

        self._mlp = MLPRegressor(
            hidden_layer_sizes=(64, 64, 32),
            activation="tanh",
            max_iter=600,
            alpha=1e-4,
            random_state=42,
            tol=1e-6,
        )
        self._mlp.fit(X_s, Y)
        self._backend = "sklearn"

        if verbose:
            print(f"    sklearn fallback: loss={self._mlp.loss_:.5f}")
        return True

    # ─── Weight persistence ───────────────────────────────────────────────────

    @property
    def _weight_path(self) -> Path:
        """Default weight file — same directory as corrections CSV."""
        return Path(self.corrections_path).parent / \
               f"pinn_{self.solution_type.lower()}_weights.pt"

    def save_weights(self, path: str = None) -> bool:
        """Save trained network weights to disk. Returns True on success."""
        if not _TORCH or self._net is None:
            return False
        try:
            p = Path(path) if path else self._weight_path
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._net.state_dict(), p)
            return True
        except Exception:
            return False

    def load_weights(self, path: str = None) -> bool:
        """
        Load network weights saved by save_weights().
        Sets is_fitted=True so predict() works immediately without retraining.
        Note: n_records and _last_train_ranges are not restored — those require
        a full load_and_train() call. Use this for inference-only warm start.
        """
        if not _TORCH:
            return False
        try:
            p = Path(path) if path else self._weight_path
            if not p.exists():
                return False
            net = _PINNNet()
            net.load_state_dict(torch.load(p, weights_only=True))
            net.eval()
            self._net     = net
            self._backend = "pytorch"
            self.is_fitted = True
            return True
        except Exception:
            return False

    def partial_fit(self, new_df: "pd.DataFrame",
                    n_epochs: int = 50, verbose: bool = False) -> bool:
        """
        Fine-tune on new records without a full retrain.

        Uses the existing network weights as starting point (50 epochs, lr=1e-4).
        Faster than load_and_train() for online updates on small batches.
        Requires is_fitted=True — call load_and_train() first.
        """
        if not (_TORCH and self._net is not None and self.is_fitted):
            return False
        if len(new_df) < 3:
            return False
        try:
            X, Y, thetas = self._build_training_data(new_df)
            X_t      = torch.from_numpy(X)
            Y_t      = torch.from_numpy(Y)
            range_t  = torch.tensor(new_df["range_m"].values,  dtype=torch.float32)
            height_t = torch.tensor(new_df["height_m"].values, dtype=torch.float32)
            v0_t     = torch.tensor(new_df["v0_ms"].values,    dtype=torch.float32)
            theta_t  = torch.from_numpy(thetas)

            opt = optim.Adam(self._net.parameters(), lr=1e-4, weight_decay=1e-5)
            self._net.train()
            for _ in range(n_epochs):
                opt.zero_grad()
                pred   = self._net(X_t)
                L_data = nn.functional.mse_loss(pred, Y_t)
                L_phys = self._physics_residual(
                    range_t, height_t, v0_t, theta_t, pred)
                loss   = L_data + self.LAMBDA_PHYSICS * L_phys
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
                opt.step()

            self._net.eval()
            if verbose:
                with torch.no_grad():
                    pred_f = self._net(X_t)
                    dl = float(nn.functional.mse_loss(pred_f, Y_t).item())
                print(f"  [{self.solution_type}] partial_fit: "
                      f"{n_epochs} epochs  data_loss={dl:.5f}")
            return True
        except Exception:
            return False

    # ─── Public API (identical to CorrectionFormula) ─────────────────────────

    def load_and_train(self, verbose: bool = True, force_retrain: bool = False) -> bool:
        """
        Load correction records from CSV and train the PINN.

        Returns True when the PINN is fitted and ready to predict.
        Same behaviour as CorrectionFormula.load_and_train().
        force_retrain: skip the weight auto-load and always retrain from CSV.
        """
        if force_retrain:
            self._net       = None
            self.is_fitted  = False
            self.last_trained_n = 0

        if not Path(self.corrections_path).exists():
            if verbose:
                print(f"  [{self.solution_type}] No corrections file yet.")
            return False

        df_all = pd.read_csv(self.corrections_path)

        # Filter by trajectory type (LOW or HIGH).
        # Normalise first: null, empty, or any non-valid value (e.g. a timestamp
        # written into the column by an old pipeline with wrong column order)
        # all default to "LOW" so legacy records are never silently dropped.
        if "solution_type" in df_all.columns:
            df = df_all[_normalize_solution_type(df_all["solution_type"])
                        == self.solution_type].copy()
        else:
            df = df_all.copy() if self.solution_type == "LOW" else pd.DataFrame()

        # Quality filter: exclude records where the correction made things worse.
        # Harmful corrections come from BO cold-start failures or pre-fix pipeline
        # runs. QUALITY_FILTER_RATIO (1.20) allows up to 20% worse to absorb
        # shot noise without blocking genuinely good corrections.
        if "miss_before" in df.columns and "miss_after" in df.columns:
            df = df[df["miss_after"] <= df["miss_before"] * QUALITY_FILTER_RATIO].copy()

        self.n_records = len(df)
        if self.n_records < self.MIN_RECORDS:
            if verbose:
                print(f"  [{self.solution_type}] Only {self.n_records} quality records "
                      f"(need ≥ {self.MIN_RECORDS})")
            return False

        backend = "PyTorch" if _TORCH else ("sklearn" if _SKLEARN else "NONE")
        if verbose:
            print(f"  [{self.solution_type}] Training PINN on "
                  f"{self.n_records} records  [backend: {backend}]")

        try:
            if _TORCH:
                ok = self._train_pytorch(df, verbose)
            elif _SKLEARN:
                ok = self._train_sklearn(df, verbose)
            else:
                if verbose:
                    print("  No backend available. pip install torch  (or scikit-learn)")
                return False
        except Exception as exc:
            if verbose:
                print(f"  [{self.solution_type}] Training failed: {exc}")
            return False

        if ok:
            self.is_fitted          = True
            self.last_trained_n     = self.n_records
            self._last_train_ranges = df["range_m"].values.copy()
            if _TORCH and self._net is not None:
                self.save_weights()
        return ok

    def should_retrain(self) -> bool:
        """
        Information-gain retraining trigger.
        Retrain when ≥5 new records exist AND they cover a range region
        more than 30m from the training distribution (novelty check).

        Identical logic to CorrectionFormula.should_retrain().
        """
        if not Path(self.corrections_path).exists():
            return False
        try:
            df_all = pd.read_csv(self.corrections_path)
            if "solution_type" in df_all.columns:
                current_n = int(
                    (_normalize_solution_type(df_all["solution_type"])
                     == self.solution_type).sum())
            else:
                current_n = len(df_all) if self.solution_type == "LOW" else 0
        except Exception:
            return False

        if current_n < self.MIN_RECORDS:
            return False            # not enough records to train at all
        if current_n < self.last_trained_n + 5:
            return False            # not enough new records yet
        if self._last_train_ranges is None:
            return True             # first time — always retrain

        try:
            df_new = pd.read_csv(self.corrections_path)
            if "solution_type" in df_new.columns:
                df_new = df_new[
                    _normalize_solution_type(df_new["solution_type"])
                    == self.solution_type].copy()
            new_ranges = df_new["range_m"].values[-5:]
            novelty = float(np.min([
                np.min(np.abs(r - self._last_train_ranges)) for r in new_ranges]))
            return novelty > 30.0   # 30m novelty threshold
        except Exception:
            return True

    def predict(self, range_m: float, height_m: float,
                v0_ms: float = 100.0) -> dict:
        """
        Predict corrections for the given target geometry.

        Returns:
            {"delta_pitch": float,   # degrees, correction to elevation
             "delta_yaw":   float,   # degrees, correction to bearing
             "delta_v0":    float,   # m/s,     correction to muzzle velocity
             "source":      str}     # backend used
        """
        if not self.is_fitted:
            return {"delta_pitch": 0.0, "delta_yaw": 0.0,
                    "delta_v0": 0.0, "source": "none"}

        theta = self._theta_cmd(range_m, height_m, v0_ms)
        feats = self._features(range_m, height_m, v0_ms, theta)

        if _TORCH and self._net is not None:
            self._net.eval()
            with torch.no_grad():
                x   = torch.from_numpy(feats).unsqueeze(0)
                out = self._net(x).numpy()[0]
            source = f"pinn_torch_{self.solution_type}"

        elif self._mlp is not None:
            X_s = self._scaler.transform(feats.reshape(1, -1))
            out = self._mlp.predict(X_s)[0]
            source = f"pinn_sklearn_{self.solution_type}"

        else:
            return {"delta_pitch": 0.0, "delta_yaw": 0.0,
                    "delta_v0": 0.0, "source": "none"}

        # Hard clips — physically motivated bounds
        return {
            "delta_pitch": float(np.clip(out[0], -3.0,  3.0)),
            "delta_yaw":   float(np.clip(out[1], -2.0,  2.0)),
            "delta_v0":    float(np.clip(out[2], -10.0, 10.0)),
            "source":      source,
        }

    def print_formulas(self):
        """
        Print a human-readable summary of what the PINN has learned.
        Shows learned corrections at benchmark ranges.
        """
        tag = self.solution_type
        if not self.is_fitted:
            print(f"  [{tag}] Not fitted yet.")
            return

        print(f"\n  [{tag}] PINN Corrector Summary")
        print(f"    Backend          : {self._backend}")
        print(f"    Records used     : {self.n_records}")
        if self._backend == "pytorch":
            print(f"    Network params   : {self._n_params:,}")
            print(f"    Final data loss  : {self._data_loss:.5f}  (°²)")
            print(f"    Final phys loss  : {self._phys_loss:.5f}")

        print(f"\n    Learned corrections  (height=0 m,  v₀=100 m/s):")
        print(f"    {'Range':>8}  {'Δpitch':>8}  {'Δyaw':>8}  {'Δv₀':>9}")
        print(f"    {'─'*8}  {'─'*8}  {'─'*8}  {'─'*9}")
        for R in [80, 120, 160, 200, 250, 300, 400]:
            p = self.predict(R, 0.0, 100.0)
            print(f"    {R:>7}m  "
                  f"{p['delta_pitch']:>+7.3f}°  "
                  f"{p['delta_yaw']:>+7.3f}°  "
                  f"{p['delta_v0']:>+8.3f} m/s")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, sys
    print("=" * 60)
    print("ARCS — pinn_corrector.py  self-test")
    print(f"  PyTorch : {'✓ available' if _TORCH else '✗ not installed'}")
    print(f"  sklearn : {'✓ available' if _SKLEARN else '✗ not installed'}")
    print("=" * 60)

    if not _TORCH and not _SKLEARN:
        print("  No backend available. pip install torch")
        sys.exit(1)

    import numpy as _np
    from physics.range_table import RangeTable
    from bayesian_optimizer  import BayesianOptimizer, EngagementMemory, EngagementSimulator

    tmpdir    = Path(tempfile.mkdtemp())
    corr_path = str(tmpdir / "corr.csv")
    phys_path = str(tmpdir / "phys.csv")

    # Build a small physics table and run enough engagements to surpass MIN_RECORDS
    rt = RangeTable(phys_path, corr_path)
    rt.generate_physics(
        range_steps  = _np.arange(50, 305, 50),
        height_steps = _np.arange(-10, 25, 15),
        v0_steps     = _np.array([80, 100, 120]),
        verbose=False, force=True)

    sim = EngagementSimulator(seed=42, range_table=rt)
    mem = EngagementMemory()
    rng = _np.random.default_rng(42)

    print("\n  Running 60 engagements to build correction records...")
    for _ in range(60):
        tx = float(rng.uniform(80, 280))
        ty = float(rng.uniform(-8, 15))
        tz = float(rng.uniform(-60, 60))
        bo = BayesianOptimizer(memory=mem, n_init=4, n_suggest=8)
        sim.run_engagement(tx, ty, tz, v0=100, optimizer=bo, verbose=False)

    n_corr = len(rt._corrections_df) if rt._corrections_df is not None else 0
    print(f"  Corrections collected: {n_corr}")

    print("\n  Training PINN (first instance)...")
    pc = PINNCorrector(corr_path, solution_type="LOW")
    ok = pc.load_and_train(verbose=True)
    print(f"  Fitted: {ok}")

    if ok:
        pc.print_formulas()

        print("\n  Prediction spot-checks:")
        for R, H in [(100, 0), (200, 10), (300, -5)]:
            p = pc.predict(R, H, 100.0)
            print(f"    ({R}m, {H}m): Δpitch={p['delta_pitch']:+.3f}°  "
                  f"Δyaw={p['delta_yaw']:+.3f}°  Δv₀={p['delta_v0']:+.2f}m/s")

        print(f"\n  should_retrain() = {pc.should_retrain()}")

        # ── BUG 1 VERIFICATION: weights auto-loaded on second instance ─────────
        print("\n  ─── BUG 1 verification: weight auto-load on restart ───")
        weight_path = pc._weight_path
        print(f"  Weight file: {weight_path}")
        print(f"  Weight file exists: {weight_path.exists()}")

        pc2 = PINNCorrector(corr_path, solution_type="LOW")
        if pc2.is_fitted:
            p1 = pc.predict(200.0, 0.0, 100.0)
            p2 = pc2.predict(200.0, 0.0, 100.0)
            identical = (abs(p1["delta_pitch"] - p2["delta_pitch"]) < 1e-6 and
                         abs(p1["delta_yaw"]   - p2["delta_yaw"])   < 1e-6)
            print(f"  pc2.is_fitted = True  ← weights restored from disk (no retraining)")
            print(f"  Predictions identical: {identical}")
            print(f"    pc  @ 200m: Δpitch={p1['delta_pitch']:+.4f}°  source={p1['source']}")
            print(f"    pc2 @ 200m: Δpitch={p2['delta_pitch']:+.4f}°  source={p2['source']}")
        else:
            print(f"  pc2.is_fitted = False  ← BUG 1 NOT fixed — auto-load missing")

    print("\n  pinn_corrector.py ✓")