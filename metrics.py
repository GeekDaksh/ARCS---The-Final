"""
ARCS — Metrics & Convergence Tracker
Phase 1

Measures accuracy and tracks improvement over time.
Every measurement is appended to history files — never overwritten.

METRICS USED:
    CEP 50%  — half of all shots land within this radius (median miss)
    CEP 90%  — 90% of shots land within this radius
    Mean miss — average miss distance
    Bias      — systematic offset (x and z direction separately)

    CEP is the official NATO/DoD accuracy metric for ballistic systems.
    Source: US DoD MIL-STD-1316, Grubbs (1964)

CONVERGENCE TRACKING:
    After each engagement, record the current system CEP.
    Over time this shows a learning curve — how fast the system improves.
    This is your core research result: the AI makes the system get better.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import datetime

HISTORY_PATH = "data/metrics_history.csv"

# Canonical column order for the history file.
# Extend this list if new columns are added — _load_existing() auto-migrates.
HISTORY_COLS = [
    "engagement_n", "timestamp",
    "baseline_cep_m", "bo_cep_m", "improvement_pct",
    "gp_applied", "target_range_m", "n_shots_used",
    "solution_type",   # added Flaw 28 — absent in files written by old code
    "label",
]


# ─────────────────────────────────────────────────────────────────
# CORE METRICS
# ─────────────────────────────────────────────────────────────────

def cep_50(df: pd.DataFrame) -> float:
    """Median miss distance — 50% of shots within this radius."""
    return float(df["miss_dist"].median())

def cep_90(df: pd.DataFrame) -> float:
    """90th percentile miss distance."""
    return float(df["miss_dist"].quantile(0.90))

def mean_miss(df: pd.DataFrame) -> float:
    return float(df["miss_dist"].mean())

def std_miss(df: pd.DataFrame) -> float:
    return float(df["miss_dist"].std())

def max_miss(df: pd.DataFrame) -> float:
    return float(df["miss_dist"].max())

def bias(df: pd.DataFrame) -> dict:
    """
    Systematic directional bias.
    If error_x mean is consistently negative, the system always
    undershoots in X — a systematic error GP can correct.
    Random errors cannot be corrected. Systematic ones can.
    """
    bx  = float(df["error_x"].mean())
    bz  = float(df["error_z"].mean())
    mag = float(np.sqrt(bx**2 + bz**2))
    return {
        "bias_x":         bx,
        "bias_z":         bz,
        "bias_magnitude": mag,
        "is_systematic":  mag > 0.5,
    }

def by_range_band(df: pd.DataFrame, band_size: int = 50) -> pd.DataFrame:
    """Accuracy broken down by target distance."""
    d = df.copy()
    d["range_band"] = (d["horiz_range"] // band_size) * band_size
    rows = []
    for start, grp in d.groupby("range_band"):
        rows.append({
            "range_label": f"{int(start)}-{int(start+band_size)}m",
            "n_shots":     len(grp),
            "cep_50_m":    cep_50(grp),
            "mean_miss_m": mean_miss(grp),
            "max_miss_m":  max_miss(grp),
            "bias_x_m":    float(grp["error_x"].mean()),
            "bias_z_m":    float(grp["error_z"].mean()),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# BASELINE REPORT
# ─────────────────────────────────────────────────────────────────

def baseline_report(df: pd.DataFrame, label: str = "Dataset") -> dict:
    """Full accuracy report — printed and returned as dict."""
    b = bias(df)
    print(f"\n{'='*56}")
    print(f"  ACCURACY REPORT — {label}")
    print(f"{'='*56}")
    print(f"  Shots          : {len(df):,}")
    print(f"  Targets        : {df['target_id'].nunique():,}")
    print(f"  CEP 50%        : {cep_50(df):.3f} m")
    print(f"  CEP 90%        : {cep_90(df):.3f} m")
    print(f"  Mean miss      : {mean_miss(df):.3f} m")
    print(f"  Std miss       : {std_miss(df):.3f} m")
    print(f"  Max miss       : {max_miss(df):.3f} m")
    print(f"  Bias X         : {b['bias_x']:+.3f} m")
    print(f"  Bias Z         : {b['bias_z']:+.3f} m")
    print(f"  Systematic bias: {'YES — GP can correct' if b['is_systematic'] else 'NO'}")
    print()
    bands = by_range_band(df)
    print(f"  {'Range':15s} {'Shots':>6} {'CEP':>8} {'Mean':>8} {'Max':>8}")
    print(f"  {'-'*15} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
    for _, row in bands.iterrows():
        print(f"  {row['range_label']:15s} "
              f"{row['n_shots']:>6} "
              f"{row['cep_50_m']:>7.2f}m "
              f"{row['mean_miss_m']:>7.2f}m "
              f"{row['max_miss_m']:>7.2f}m")
    return {
        "cep_50":   cep_50(df),
        "cep_90":   cep_90(df),
        "mean":     mean_miss(df),
        "std":      std_miss(df),
        "max":      max_miss(df),
        "bias":     b,
        "by_range": bands,
    }


# ─────────────────────────────────────────────────────────────────
# CONVERGENCE TRACKER
# ─────────────────────────────────────────────────────────────────

class ConvergenceTracker:
    """
    Records system accuracy after each engagement.
    Builds a learning curve: CEP vs engagement number.

    Every record is appended — never overwritten.
    Old history files (missing solution_type) are migrated on load.
    """

    def __init__(self, path: str = HISTORY_PATH):
        self.path = path
        self._records = []
        self._load_existing()

    def _load_existing(self):
        if not Path(self.path).exists():
            return
        # Use on_bad_lines='skip' so a column-count mismatch caused by old
        # rows (9 cols) mixed with new rows (10 cols, added solution_type)
        # does not raise a ParserError and lose all history.
        try:
            df = pd.read_csv(self.path, on_bad_lines="skip")
        except TypeError:
            # pandas < 1.3 used the error_bad_lines kwarg
            try:
                df = pd.read_csv(self.path, error_bad_lines=False)
            except Exception:
                df = pd.DataFrame(columns=HISTORY_COLS)

        # Migrate old files that lack solution_type (written before Flaw 28).
        # Rewrite in canonical HISTORY_COLS order so future appends land in
        # the correct columns — same root-cause fix applied to corrections.
        needs_migration = "solution_type" not in df.columns
        if needs_migration:
            df["solution_type"] = "LOW"

        # Ensure every canonical column exists (forward-compat)
        for col in HISTORY_COLS:
            if col not in df.columns:
                df[col] = ""

        df = df.reindex(columns=HISTORY_COLS)

        if needs_migration:
            df.to_csv(self.path, index=False)

        self._records = df.to_dict("records")

    def record(self,
               engagement_n:  int,
               baseline_cep:  float,
               bo_cep:        float,
               gp_applied:    bool,
               target_range:  float,
               improvement_pct: float,
               n_shots_used:  int,
               label:         str = "",
               solution_type: str = "LOW"):
        """
        Append one engagement result to history.

        Flaw 28 fix: solution_type stored for per-trajectory learning curves.
        Column-order fix: always written in HISTORY_COLS order so appending
        to a migrated file never causes a field-count ParserError on reload.
        """
        row = {
            "engagement_n":    engagement_n,
            "timestamp":       datetime.datetime.now().isoformat(),
            "baseline_cep_m":  baseline_cep,
            "bo_cep_m":        bo_cep,
            "improvement_pct": improvement_pct,
            "gp_applied":      gp_applied,
            "target_range_m":  target_range,
            "n_shots_used":    n_shots_used,
            "solution_type":   solution_type,
            "label":           label,
        }
        self._records.append(row)

        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        header = not Path(self.path).exists()
        pd.DataFrame([row]).reindex(columns=HISTORY_COLS).to_csv(
            self.path, mode='a', header=header, index=False)

    def summary(self) -> dict:
        """Summary stats across all recorded engagements."""
        if not self._records:
            return {"n_engagements": 0}
        df  = pd.DataFrame(self._records)
        pos = df[df["improvement_pct"] > 0]
        result = {
            "n_engagements":      len(df),
            "mean_baseline_cep":  df["baseline_cep_m"].mean(),
            "mean_bo_cep":        df["bo_cep_m"].mean(),
            "mean_improvement":   df["improvement_pct"].mean(),
            "pct_positive":       len(pos) / len(df) * 100,
            "best_improvement":   df["improvement_pct"].max(),
            "best_bo_miss":       df["bo_cep_m"].min(),
        }
        # Flaw 28 fix: per-type improvement breakdown
        if "solution_type" in df.columns:
            for st in ["LOW", "HIGH"]:
                sub = df[df["solution_type"] == st]
                if len(sub):
                    result[f"mean_improvement_{st.lower()}"] = sub["improvement_pct"].mean()
                    result[f"n_engagements_{st.lower()}"]    = len(sub)
        return result

    def print_learning_curve(self):
        """Print convergence curve — CEP over engagement number."""
        if not self._records:
            print("  No engagement history yet.")
            return
        df = pd.DataFrame(self._records)
        # Flaw 28 fix: show solution_type column in learning curve
        has_type = "solution_type" in df.columns
        print(f"\n  {'Eng':>5}  {'Baseline':>10}  {'BO CEP':>10}  "
              f"{'Improve':>10}  {'Range':>8}"
              + ("  Type" if has_type else ""))
        print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}"
              + ("  ----" if has_type else ""))
        for _, row in df.iterrows():
            arrow = "↑" if row["improvement_pct"] > 0 else "↓"
            ttype = f"  {row['solution_type']}" if has_type else ""
            print(f"  {row['engagement_n']:>5}  "
                  f"{row['baseline_cep_m']:>9.2f}m  "
                  f"{row['bo_cep_m']:>9.2f}m  "
                  f"{arrow}{abs(row['improvement_pct']):>8.1f}%  "
                  f"{row['target_range_m']:>7.0f}m{ttype}")

        s = self.summary()
        print(f"\n  Mean improvement : {s['mean_improvement']:+.1f}%")
        print(f"  Positive engagements : "
              f"{s['pct_positive']:.0f}%")
        print(f"  Best single improvement : "
              f"{s['best_improvement']:.1f}%")
        print(f"  Best miss ever : {s['best_bo_miss']:.2f}m")


# ─────────────────────────────────────────────────────────────────
# COMPARE BEFORE / AFTER
# ─────────────────────────────────────────────────────────────────

def compare(before: dict, after: dict, label: str = "AI"):
    """Print before/after comparison table."""
    keys = [("CEP 50%","cep_50"),("CEP 90%","cep_90"),
            ("Mean miss","mean"),("Std miss","std"),("Max miss","max")]
    print(f"\n{'='*56}")
    print(f"  IMPROVEMENT — Baseline vs {label}")
    print(f"{'='*56}")
    print(f"  {'Metric':12s}  {'Before':>10}  {'After':>10}  {'Δ':>10}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")
    for label_m, key in keys:
        b   = before[key]
        a   = after[key]
        pct = (b-a)/b*100 if b > 0 else 0
        arrow = "↓" if pct > 0 else "↑"
        print(f"  {label_m:12s}  "
              f"{b:>9.3f}m  "
              f"{a:>9.3f}m  "
              f"  {arrow}{abs(pct):.1f}%")


# ─────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*56)
    print("ARCS Phase 1 — Metrics & Convergence")
    print("="*56)

    df_clean = pd.read_csv("data/dataset_clean.csv")
    df_noisy = pd.read_csv("data/dataset_noisy.csv")

    r_clean = baseline_report(df_clean, "CLEAN (zero noise)")
    r_noisy = baseline_report(df_noisy, "NOISY — AI BASELINE")

    print(f"\n{'='*56}")
    print(f"  BASELINE SUMMARY")
    print(f"{'='*56}")
    print(f"  AI must beat CEP : {r_noisy['cep_50']:.2f}m")
    print(f"  Systematic bias  : "
          f"{'YES' if r_noisy['bias']['is_systematic'] else 'NO'}")

    # Print convergence history if available
    tracker = ConvergenceTracker()
    if tracker._records:
        print(f"\n  LEARNING CURVE ({len(tracker._records)} engagements):")
        tracker.print_learning_curve()
    else:
        print(f"\n  No engagement history yet.")
        print(f"  Run pipeline.py to start accumulating history.")