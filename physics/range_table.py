"""
ARCS — Range Table
Phase 1 — Production Quality

TWO-FILE DESIGN:
    range_table_physics.csv     — generated once from pure physics, never modified
    range_table_corrections.csv — append-only, grows with every engagement

    Physics values are deterministic: same range/height/v0 always gives the same
    pitch/tof. Corrections are learned: regenerating the physics table never erases
    accumulated engagement knowledge.

DUAL-TRAJECTORY:
    Each grid cell stores both LOW-angle (flat) and HIGH-angle (lobbed) solutions.
    lookup() accepts prefer="LOW" (default) or prefer="HIGH". Corrections are tagged
    with solution_type so LOW and HIGH data are never mixed.

CORRECTION LOOKUP:
    weight_i = n_obs_i / (distance_i + 1.0)  — trust scales with observation count.
"""

import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, TypedDict
from scipy.interpolate import RegularGridInterpolator

from .constants import (RT_RANGE_STEPS, RT_HEIGHT_STEPS, RT_V0_STEPS,
                        RANGE_MIN, RANGE_MAX)
from .ballistic_solver import BallisticSolver

# ── Default file paths (absolute, relative to this file's package root) ──────
_DATA_DIR        = Path(__file__).parent.parent / "data"
PHYSICS_PATH     = str(_DATA_DIR / "range_table_physics.csv")
CORRECTIONS_PATH = str(_DATA_DIR / "range_table_corrections.csv")


class LookupResult(TypedDict):
    range_m:             float
    height_m:            float
    v0_ms:               float
    pitch_deg:           float
    tof_s:               float
    max_height_m:        float
    solution_type:       str
    trajectory_fallback: bool
    delta_pitch:         float
    delta_yaw:           float
    delta_v0:            float
    corrected_pitch:     float
    corrected_yaw:       float
    corrected_v0:        float
    confidence:          Optional[float]
    n_observations:      int
    ai_corrected:        bool


class StatsResult(TypedDict, total=False):
    """Return type of RangeTable.stats(). All keys optional — presence depends on what is loaded."""
    physics_total:          int
    physics_reachable:      int
    physics_high_solutions: int
    pitch_range:            tuple
    corrections_total:      int
    engagements_recorded:   int
    mean_miss_before:       float
    mean_miss_after:        float
    mean_improvement_pct:   float
    range_covered:          tuple
    corrections_low:        int
    corrections_high:       int


# ── Corrections schema ────────────────────────────────────────────
# solution_type added to distinguish LOW vs HIGH corrections.
# Old files without this column are handled gracefully (defaults to "LOW").
CORRECTIONS_COLS = [
    "range_m", "height_m", "v0_ms",
    "delta_pitch", "delta_yaw", "delta_v0",
    "miss_before", "miss_after",
    "confidence", "n_shots_used",
    "solution_type",          # "LOW" or "HIGH"
    "engagement_id", "timestamp"
]


class RangeTable:
    """
    Physics range table with persistent learned corrections.

    Stores BOTH LOW-angle and HIGH-angle ballistic solutions per grid
    cell, enabling the system to engage targets with either trajectory.

    Usage:
        rt = RangeTable()
        rt.generate_physics()          # once only
        rt.load()                      # every startup

        sol = rt.lookup(150, 10, 100)           # LOW angle (default)
        sol = rt.lookup(150, 10, 100, "HIGH")   # HIGH angle
        rt.record_correction(...)      # after each engagement
    """

    def __init__(self,
                 physics_path:     str = PHYSICS_PATH,
                 corrections_path: str = CORRECTIONS_PATH):
        self.physics_path     = physics_path
        self.corrections_path = corrections_path
        self.solver           = BallisticSolver()

        self._physics_df:     Optional[pd.DataFrame] = None
        self._corrections_df: Optional[pd.DataFrame] = None

        # Interpolators for LOW-angle physics columns (built once on load)
        self._interp_pitch: Optional[RegularGridInterpolator] = None
        self._interp_tof:   Optional[RegularGridInterpolator] = None
        self._interp_maxh:  Optional[RegularGridInterpolator] = None

        # Interpolators for HIGH-angle physics columns
        self._interp_pitch_high: Optional[RegularGridInterpolator] = None
        self._interp_tof_high:   Optional[RegularGridInterpolator] = None
        self._interp_maxh_high:  Optional[RegularGridInterpolator] = None

        self._engagement_counter = 0

    # ─── GENERATE PHYSICS TABLE ──────────────────────────────────

    def generate_physics(self,
                         range_steps:  np.ndarray = RT_RANGE_STEPS,
                         height_steps: np.ndarray = RT_HEIGHT_STEPS,
                         v0_steps:     np.ndarray = RT_V0_STEPS,
                         verbose: bool = True,
                         force:   bool = False) -> pd.DataFrame:
        """
        Generate physics table from ballistic equations and save to physics_path.

        For each (range, height, v0) grid cell both LOW and HIGH solutions are
        computed and stored as separate columns. LOW columns also exist as
        backward-compatible aliases (pitch_deg, tof_s, max_height_m, impact_vel_ms).
        HIGH columns are NaN when the solution is invalid or outside pitch limits.

        force=False: skip generation if file already exists.
        """
        if not force and Path(self.physics_path).exists():
            print(f"  Physics table already exists: {self.physics_path}")
            print(f"  Use force=True to regenerate. Loading existing.")
            return self.load_physics()

        rows  = []
        total = len(range_steps) * len(height_steps) * len(v0_steps)
        count = 0

        if verbose:
            print(f"  Generating physics table (DUAL trajectory): "
                  f"{total:,} entries...")

        for R in range_steps:
            for H in height_steps:
                for v0 in v0_steps:
                    # Compute BOTH trajectories
                    both = self.solver.solve_both(
                        target_x=float(R),
                        target_y=float(H),
                        target_z=0.0,
                        v0=float(v0)
                    )
                    low  = both["LOW"]
                    high = both["HIGH"]

                    # LOW solution
                    has_low = low is not None
                    p_l   = low.turret_pitch_deg if has_low else np.nan
                    t_l   = low.tof              if has_low else np.nan
                    mh_l  = low.max_height       if has_low else np.nan
                    iv_l  = low.impact_velocity  if has_low else np.nan
                    st_l  = low.solution_type    if has_low else "UNREACHABLE"

                    # HIGH solution
                    has_high = high is not None
                    p_h   = high.turret_pitch_deg if has_high else np.nan
                    t_h   = high.tof              if has_high else np.nan
                    mh_h  = high.max_height       if has_high else np.nan
                    iv_h  = high.impact_velocity  if has_high else np.nan

                    rows.append({
                        # Backward-compatible aliases (= LOW)
                        "range_m":          float(R),
                        "height_m":         float(H),
                        "v0_ms":            float(v0),
                        "pitch_deg":        p_l,
                        "tof_s":            t_l,
                        "max_height_m":     mh_l,
                        "impact_vel_ms":    iv_l,
                        "solution_type":    st_l,
                        "reachable":        has_low,
                        # Explicit LOW
                        "pitch_deg_low":    p_l,
                        "tof_s_low":        t_l,
                        "max_height_m_low": mh_l,
                        "impact_vel_ms_low":iv_l,
                        # HIGH
                        "pitch_deg_high":   p_h,
                        "tof_s_high":       t_h,
                        "max_height_m_high":mh_h,
                        "impact_vel_ms_high":iv_h,
                        "has_high_solution": has_high,
                    })
                    count += 1
                    if verbose and count % max(1, total // 10) == 0:
                        pct = int(count / total * 100)
                        print(f"    {pct}%")

        self._physics_df = pd.DataFrame(rows)
        Path(self.physics_path).parent.mkdir(parents=True, exist_ok=True)
        self._physics_df.to_csv(self.physics_path, index=False)

        reachable   = self._physics_df["reachable"].sum()
        n_high      = self._physics_df["has_high_solution"].sum()
        if verbose:
            print(f"  Saved: {self.physics_path}  "
                  f"({len(self._physics_df):,} rows, "
                  f"{reachable:,} LOW reachable, "
                  f"{n_high:,} HIGH solutions)")

        self._build_interpolators()
        return self._physics_df

    # ─── LOAD ────────────────────────────────────────────────────

    def load(self, verbose: bool = True) -> "RangeTable":
        """
        Load physics table and corrections.
        Corrections file may not exist yet — that is fine.
        """
        self.load_physics(verbose=verbose)
        self.load_corrections(verbose=verbose)
        return self

    def load_physics(self, verbose: bool = True) -> pd.DataFrame:
        if not Path(self.physics_path).exists():
            raise FileNotFoundError(
                f"Physics table not found: {self.physics_path}\n"
                f"Run rt.generate_physics() first."
            )
        self._physics_df = pd.read_csv(self.physics_path)

        # Migrate old single-trajectory tables that lack HIGH columns.
        # Old tables have only pitch_deg, tof_s, max_height_m, impact_vel_ms.
        # We add the new columns and trigger a one-time regeneration warning.
        if "pitch_deg_high" not in self._physics_df.columns:
            if verbose:
                print(f"  [MIGRATION] Old physics table detected (single-trajectory).")
                print(f"  [MIGRATION] Adding empty HIGH-angle columns.")
                print(f"  [MIGRATION] Run generate_physics(force=True) to rebuild "
                      f"with full dual-trajectory data.")
            self._physics_df["pitch_deg_low"]    = self._physics_df["pitch_deg"]
            self._physics_df["tof_s_low"]        = self._physics_df["tof_s"]
            self._physics_df["max_height_m_low"] = self._physics_df["max_height_m"]
            self._physics_df["impact_vel_ms_low"]= self._physics_df["impact_vel_ms"]
            self._physics_df["pitch_deg_high"]   = np.nan
            self._physics_df["tof_s_high"]       = np.nan
            self._physics_df["max_height_m_high"]= np.nan
            self._physics_df["impact_vel_ms_high"]= np.nan
            self._physics_df["has_high_solution"] = False

        if verbose:
            n_high = int(self._physics_df["has_high_solution"].sum()) \
                     if "has_high_solution" in self._physics_df.columns else 0
            print(f"  Loaded physics: {self.physics_path}  "
                  f"({len(self._physics_df):,} rows, "
                  f"{n_high:,} with HIGH solution)")
        self._build_interpolators()
        return self._physics_df

    def load_corrections(self, verbose: bool = True) -> pd.DataFrame:
        if Path(self.corrections_path).exists():
            self._corrections_df = pd.read_csv(self.corrections_path)
            # Migration 1: files that predate the solution_type column entirely.
            if "solution_type" not in self._corrections_df.columns:
                self._corrections_df["solution_type"] = "LOW"
                write_cols = [c for c in CORRECTIONS_COLS
                              if c in self._corrections_df.columns]
                self._corrections_df = self._corrections_df.reindex(
                    columns=write_cols)
                self._corrections_df.to_csv(self.corrections_path, index=False)
                if verbose:
                    print(f"  [MIGRATION] Added solution_type='LOW' to all "
                          f"existing records in {self.corrections_path}")

            # Migration 2: column-order shift bug.
            # When CORRECTIONS_COLS moved solution_type before engagement_id/timestamp
            # (to group logical fields together), new rows were appended in the new
            # order while the file header still described the old order. Detect by
            # finding rows where solution_type is not "LOW"/"HIGH" (it holds a
            # timestamp string instead) and swap the three affected columns back.
            bad = ~self._corrections_df["solution_type"].isin(["LOW", "HIGH"])
            if bad.any():
                n_bad = int(bad.sum())
                real_st  = self._corrections_df.loc[bad, "engagement_id"].values.copy()
                real_eid = self._corrections_df.loc[bad, "timestamp"].values.copy()
                real_ts  = self._corrections_df.loc[bad, "solution_type"].values.copy()
                # Cast the three swapped columns to object dtype before
                # reassignment. Pandas 2.x uses StringDtype for string columns
                # which rejects integer values (engagement counters). Object
                # dtype accepts any Python type. The file is rewritten in
                # canonical column order immediately after, so the temporary
                # mixed-type state is never persisted.
                for _col in ["solution_type", "engagement_id", "timestamp"]:
                    self._corrections_df[_col] = \
                        self._corrections_df[_col].astype(object)
                self._corrections_df.loc[bad, "solution_type"] = real_st
                self._corrections_df.loc[bad, "engagement_id"] = real_eid
                self._corrections_df.loc[bad, "timestamp"]     = real_ts
                # Rewrite file in canonical column order so future appends align.
                self._corrections_df.reindex(columns=CORRECTIONS_COLS).to_csv(
                    self.corrections_path, index=False)
                if verbose:
                    print(f"  [MIGRATION] Fixed {n_bad} column-shifted records "
                          f"in {self.corrections_path}")

            if verbose:
                n   = len(self._corrections_df)
                eng = self._corrections_df["engagement_id"].nunique()
                print(f"  Loaded corrections: {self.corrections_path}  "
                      f"({n} records, {eng} engagements)")
            if len(self._corrections_df):
                # pd.read_csv can parse engagement_id as object/string dtype
                # depending on how the file was written (append mode, concat,
                # or migration rewrite). Force numeric before calling max()
                # to avoid "can only concatenate str to str" TypeError.
                eng_ids = pd.to_numeric(
                    self._corrections_df["engagement_id"], errors="coerce"
                ).fillna(0)
                self._corrections_df["engagement_id"] = eng_ids.astype(int)
                self._engagement_counter = int(eng_ids.max() + 1)
        else:
            # Build empty DataFrame with proper dtypes so numeric operations
            # (distance calcs, np.sqrt) never fail on object-typed columns.
            _FLOAT = {"range_m", "height_m", "v0_ms", "delta_pitch",
                      "delta_yaw", "delta_v0", "miss_before", "miss_after",
                      "confidence"}
            _INT   = {"n_shots_used", "engagement_id"}
            self._corrections_df = pd.DataFrame(
                {c: pd.array([], dtype="float64" if c in _FLOAT
                             else "int64" if c in _INT else "object")
                 for c in CORRECTIONS_COLS})
            if verbose:
                print(f"  No corrections file yet — starting fresh.")
        return self._corrections_df

    # ─── INTERPOLATORS ───────────────────────────────────────────

    def _build_interpolators(self):
        """Build 3D physics interpolators for both LOW and HIGH trajectories."""
        if self._physics_df is None:
            return

        reach   = self._physics_df[self._physics_df["reachable"]].copy()
        ranges  = np.sort(reach["range_m"].unique())
        heights = np.sort(reach["height_m"].unique())
        v0s     = np.sort(reach["v0_ms"].unique())

        # Build index maps for vectorized grid fill
        r_map = {r: i for i, r in enumerate(ranges)}
        h_map = {h: i for i, h in enumerate(heights)}
        v_map = {v: i for i, v in enumerate(v0s)}
        r_idx_arr = np.array([r_map.get(v, -1) for v in reach["range_m"].values])
        h_idx_arr = np.array([h_map.get(v, -1) for v in reach["height_m"].values])
        v_idx_arr = np.array([v_map.get(v, -1) for v in reach["v0_ms"].values])
        valid_mask = (r_idx_arr >= 0) & (h_idx_arr >= 0) & (v_idx_arr >= 0)

        def make_grid(col, fill_nan_with_median: bool = True):
            """
            Build a 3-D RegularGridInterpolator for col.

            LOW columns (fill_nan_with_median=True): missing cells are filled
            with the column median so the interpolator is always defined.
            HIGH columns (fill_nan_with_median=False): NaN is preserved so
            lookup() can detect "no HIGH solution here" and fall back to LOW.
            HIGH uses nearest-neighbour to avoid NaN propagation from adjacent
            empty cells that trilinear stencil would otherwise include.
            """
            grid = np.full((len(ranges), len(heights), len(v0s)), np.nan)
            col_vals = pd.to_numeric(reach[col], errors="coerce").values
            ri = r_idx_arr[valid_mask]
            hi = h_idx_arr[valid_mask]
            vi = v_idx_arr[valid_mask]
            grid[ri, hi, vi] = col_vals[valid_mask]
            if fill_nan_with_median and np.isnan(grid).any():
                med = np.nanmedian(grid)
                grid = np.where(np.isnan(grid),
                                med if not np.isnan(med) else 0.0, grid)
            fill_value = None if fill_nan_with_median else np.nan
            # LOW uses linear (dense, no NaN → smooth between grid points).
            # HIGH uses nearest-neighbour: sparse grid — NaN propagation from
            # the 8-corner trilinear stencil would corrupt valid adjacent cells.
            method = "linear" if fill_nan_with_median else "nearest"
            return RegularGridInterpolator(
                (ranges, heights, v0s), grid,
                method=method, bounds_error=False, fill_value=fill_value
            )

        # LOW-angle interpolators — dense; NaN filled with median
        self._interp_pitch = make_grid("pitch_deg",    fill_nan_with_median=True)
        self._interp_tof   = make_grid("tof_s",        fill_nan_with_median=True)
        self._interp_maxh  = make_grid("max_height_m", fill_nan_with_median=True)

        # HIGH-angle interpolators — sparse; NaN kept so lookup() detects
        # "no HIGH solution here" cleanly, without a misleading median fill.
        if "pitch_deg_high" in reach.columns:
            self._interp_pitch_high = make_grid("pitch_deg_high",    fill_nan_with_median=False)
            self._interp_tof_high   = make_grid("tof_s_high",        fill_nan_with_median=False)
            self._interp_maxh_high  = make_grid("max_height_m_high", fill_nan_with_median=False)

        self._ranges  = ranges
        self._heights = heights
        self._v0s     = v0s

    # ─── LOOKUP ──────────────────────────────────────────────────

    def lookup(self, range_m: float, height_m: float, v0_ms: float,
               prefer: str = "LOW") -> LookupResult:
        """
        Return physics solution + weighted correction from learned data.

        Args:
            range_m  : horizontal distance to target (m)
            height_m : target height above robot (m)
            v0_ms    : muzzle velocity (m/s)
            prefer   : "LOW" (flat, default) or "HIGH" (lobbed)

        Returns LookupResult TypedDict with keys: pitch_deg, tof_s,
        max_height_m, solution_type, trajectory_fallback (True if HIGH
        requested but unavailable), delta_pitch/yaw/v0, corrected_pitch,
        corrected_yaw, corrected_v0, confidence, n_observations, ai_corrected.
        """
        if prefer not in ("LOW", "HIGH"):
            raise ValueError(f"prefer must be 'LOW' or 'HIGH', got {prefer!r}")
        if self._physics_df is None:
            raise RuntimeError("Call load() or generate_physics() first.")

        resolved_type       = prefer
        trajectory_fallback = False

        # Direct Newton solution — exact ballistics, zero interpolation error at
        # any point in the envelope (including grid boundaries and beyond).
        sol = self.solver.solve(range_m, height_m, 0.0, v0_ms, prefer=prefer)

        if not sol.reachable and prefer == "HIGH":
            # HIGH unreachable (mechanical limits) — try LOW as fallback.
            sol = self.solver.solve(range_m, height_m, 0.0, v0_ms, prefer="LOW")
            resolved_type       = "LOW"
            trajectory_fallback = True

        if not sol.reachable:
            # Both trajectories mechanically unreachable (e.g. LOW angle < 0° and
            # HIGH angle > 85° at very close range with high v0). Fall back to
            # grid interpolation so lookup() never raises — the result is
            # approximate and flagged with solution_type="UNREACHABLE".
            r_q = float(np.clip(range_m,  self._ranges[0],  self._ranges[-1]))
            h_q = float(np.clip(height_m, self._heights[0], self._heights[-1]))
            v_q = float(np.clip(v0_ms,    self._v0s[0],     self._v0s[-1]))
            pt  = np.array([[r_q, h_q, v_q]])
            pitch = float(self._interp_pitch(pt)[0])
            tof   = float(self._interp_tof(pt)[0])
            maxh  = float(self._interp_maxh(pt)[0])
            dp, dy, dv, conf, n_obs = self._weighted_correction(
                range_m, height_m, v0_ms, solution_type="LOW")
            return LookupResult(
                range_m=range_m, height_m=height_m, v0_ms=v0_ms,
                pitch_deg=pitch, tof_s=tof, max_height_m=maxh,
                solution_type="UNREACHABLE", trajectory_fallback=True,
                delta_pitch=dp, delta_yaw=dy, delta_v0=dv,
                corrected_pitch=pitch + dp, corrected_yaw=dy,
                corrected_v0=v0_ms + dv,
                confidence=conf, n_observations=n_obs, ai_corrected=n_obs > 0)

        # Mechanical limits inside solve() may have swapped trajectory type.
        resolved_type = sol.solution_type
        if prefer == "HIGH" and sol.solution_type not in ("HIGH", "OPTIMAL"):
            trajectory_fallback = True

        pitch = sol.turret_pitch_deg
        tof   = sol.tof
        maxh  = sol.max_height

        # Weighted correction from learned engagement history
        dp, dy, dv, conf, n_obs = self._weighted_correction(
            range_m, height_m, v0_ms, solution_type=resolved_type)

        return LookupResult(
            range_m=range_m,
            height_m=height_m,
            v0_ms=v0_ms,
            pitch_deg=pitch,
            tof_s=tof,
            max_height_m=maxh,
            solution_type=resolved_type,
            trajectory_fallback=trajectory_fallback,
            delta_pitch=dp,
            delta_yaw=dy,
            delta_v0=dv,
            corrected_pitch=pitch + dp,
            corrected_yaw=dy,
            corrected_v0=v0_ms + dv,
            confidence=conf,
            n_observations=n_obs,
            ai_corrected=n_obs > 0,
        )

    def _weighted_correction(self, range_m, height_m, v0_ms,
                              solution_type: str = "LOW"):
        """
        Compute a weighted average correction from nearby records.

        Weighting: w_i = n_obs_i / (dist_i + 1.0). Corrections are filtered
        by solution_type so LOW and HIGH data are never mixed.

        Returns (delta_pitch, delta_yaw, delta_v0, confidence, n_total)
        """
        if (self._corrections_df is None or
                len(self._corrections_df) == 0):
            return 0.0, 0.0, 0.0, None, 0

        df = self._corrections_df

        if "solution_type" in df.columns:
            df = df[df["solution_type"] == solution_type]
        if len(df) == 0:
            return 0.0, 0.0, 0.0, None, 0

        # Normalised distance: scale factors (50m range, 10m height, 50 m/s v0).
        # radius_norm converts the adaptive meter radius into the same space so
        # the threshold is dimensionally consistent across all three axes.
        dist = np.sqrt(
            ((df["range_m"]  - range_m)  / 50.0) ** 2 +
            ((df["height_m"] - height_m) / 10.0) ** 2 +
            ((df["v0_ms"]    - v0_ms)    / 50.0) ** 2
        )
        radius_norm = max(20.0, range_m * 0.20) / 50.0

        mask = dist < radius_norm
        if not mask.any():
            nearest = dist.idxmin()
            mask = pd.Series(False, index=df.index)
            mask[nearest] = True

        nearby      = df[mask].copy()
        nearby_dist = dist[mask].values

        weights = nearby["n_shots_used"].values / (nearby_dist + 1.0)
        weights = weights / weights.sum()

        dp   = float((nearby["delta_pitch"].values * weights).sum())
        dy   = float((nearby["delta_yaw"].values   * weights).sum())
        dv   = float((nearby["delta_v0"].values    * weights).sum())
        conf = float((nearby["confidence"].values  * weights).sum())
        n    = int(nearby["n_shots_used"].sum())

        return dp, dy, dv, conf, n

    # ─── RECORD CORRECTION ───────────────────────────────────────

    def record_correction(self,
                          range_m:      float,
                          height_m:     float,
                          v0_ms:        float,
                          delta_pitch:  float,
                          delta_yaw:    float,
                          delta_v0:     float,
                          miss_before:  float,
                          miss_after:   float,
                          confidence:   float,
                          n_shots_used: int,
                          solution_type: str = "LOW"):
        """
        Append one correction record to the corrections file.
        solution_type ("LOW" or "HIGH") ensures corrections are later
        filtered by trajectory type during weighted lookup.
        NEVER overwrites existing data — only appends.
        """
        if solution_type not in ("LOW", "HIGH"):
            raise ValueError(
                f"solution_type must be 'LOW' or 'HIGH', got {solution_type!r}")
        row = {
            "range_m":      range_m,
            "height_m":     height_m,
            "v0_ms":        v0_ms,
            "delta_pitch":  delta_pitch,
            "delta_yaw":    delta_yaw,
            "delta_v0":     delta_v0,
            "miss_before":  miss_before,
            "miss_after":   miss_after,
            "confidence":   confidence,
            "n_shots_used": n_shots_used,
            "solution_type": solution_type,
            "engagement_id": self._engagement_counter,
            "timestamp":    datetime.datetime.now().isoformat(),
        }
        self._engagement_counter += 1

        new_row = pd.DataFrame([row])
        if self._corrections_df is None:
            self._corrections_df = new_row
        else:
            self._corrections_df = pd.concat(
                [self._corrections_df, new_row], ignore_index=True)

        Path(self.corrections_path).parent.mkdir(parents=True, exist_ok=True)
        header = not Path(self.corrections_path).exists()
        # Always write columns in canonical CORRECTIONS_COLS order to prevent
        # column misalignment on migration-rewritten files.
        new_row.reindex(columns=CORRECTIONS_COLS).to_csv(
            self.corrections_path, mode='a', header=header, index=False)

    # ─── STATISTICS ──────────────────────────────────────────────

    def stats(self) -> StatsResult:
        s = {}
        if self._physics_df is not None:
            reach  = self._physics_df[self._physics_df["reachable"]]
            n_high = int(self._physics_df.get("has_high_solution",
                         pd.Series([False])).sum())
            s["physics_total"]       = len(self._physics_df)
            s["physics_reachable"]   = len(reach)
            s["physics_high_solutions"] = n_high
            s["pitch_range"]         = (reach["pitch_deg"].min(),
                                        reach["pitch_deg"].max())

        if (self._corrections_df is not None and
                len(self._corrections_df) > 0):
            c = self._corrections_df
            s["corrections_total"]    = len(c)
            s["engagements_recorded"] = c["engagement_id"].nunique()
            s["mean_miss_before"]     = c["miss_before"].mean()
            s["mean_miss_after"]      = c["miss_after"].mean()
            s["mean_improvement_pct"] = (
                (c["miss_before"] - c["miss_after"]) /
                c["miss_before"].clip(lower=0.01) * 100
            ).mean()
            s["range_covered"]        = (c["range_m"].min(),
                                          c["range_m"].max())
            # Count corrections per trajectory type
            if "solution_type" in c.columns:
                s["corrections_low"]  = int((c["solution_type"] == "LOW").sum())
                s["corrections_high"] = int((c["solution_type"] == "HIGH").sum())
        else:
            s["corrections_total"]    = 0
            s["engagements_recorded"] = 0

        return s


# ─── SELF TEST ───────────────────────────────────────────────────
if __name__ == "__main__":
    import time, tempfile, shutil
    print("ARCS — range_table.py smoke test")
    _tmpdir      = Path(tempfile.mkdtemp())
    test_physics = str(_tmpdir / "rt_test_physics.csv")
    test_corr    = str(_tmpdir / "rt_test_corrections.csv")

    rt = RangeTable(physics_path=test_physics, corrections_path=test_corr)
    t0 = time.time()
    rt.generate_physics(
        range_steps  = np.arange(50, 205, 25),
        height_steps = np.arange(-10, 25, 10),
        v0_steps     = np.array([80, 100, 120]),
        verbose=False, force=True
    )
    n_high = rt._physics_df["has_high_solution"].sum()
    print(f"  Generated in {time.time()-t0:.2f}s  HIGH solutions: {n_high} ✓")

    res_l = rt.lookup(200, 0, 100, prefer="LOW")
    res_h = rt.lookup(200, 0, 100, prefer="HIGH")
    assert not res_h["trajectory_fallback"], "HIGH solution missing at (200,0,100)"
    assert res_h["pitch_deg"] > res_l["pitch_deg"], "HIGH pitch must exceed LOW pitch"
    print(f"  LOW={res_l['pitch_deg']:.2f}°  HIGH={res_h['pitch_deg']:.2f}°  HIGH>LOW ✓")

    rt.record_correction(200, 0, 100, 0.4, -0.2, 1.0, 12.5, 4.3, 0.15, 12, "LOW")
    rt.record_correction(200, 0, 100, 0.7, -0.1, 0.5, 11.0, 5.0, 0.12, 10, "HIGH")
    res_l2 = rt.lookup(200, 0, 100, prefer="LOW")
    res_h2 = rt.lookup(200, 0, 100, prefer="HIGH")
    assert abs(res_l2["delta_pitch"] - 0.4) < 0.1
    assert abs(res_h2["delta_pitch"] - 0.7) < 0.1
    print(f"  LOW Δpitch={res_l2['delta_pitch']:.3f}°  HIGH Δpitch={res_h2['delta_pitch']:.3f}°  separated ✓")

    rt2 = RangeTable(physics_path=test_physics, corrections_path=test_corr)
    rt2.load(verbose=False)
    assert rt2.lookup(200, 0, 100, prefer="HIGH")["n_observations"] > 0
    print(f"  Corrections survive reload ✓")

    shutil.rmtree(_tmpdir, ignore_errors=True)
    print("  range_table.py ✓ all checks passed")