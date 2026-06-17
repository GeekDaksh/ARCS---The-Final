"""
ARCS — Engagement Database v2.0
Persistent SQLite storage with per-weapon bias profiles.

Three tables:
  weapon_profiles — one row per weapon, holds the living learned-bias triple.
  engagements     — one row per fire mission, linked to a weapon.
  rounds          — one row per shell fired, linked to an engagement.

The key reuse path:
  1. run_persistent_engagement() looks up the weapon in weapon_profiles.
  2. If found, it loads the stored [b_sag, b_yaw, b_v0] into the SBE
     as a warm prior so engagement #2 converges faster than engagement #1.
  3. After the engagement the SBE's updated bias is written back to
     weapon_profiles and the full engagement is recorded for audit.

Backward compatibility:
  The legacy log() / get_all() / get_sbe_inputs() / statistics() / clear()
  interface is fully preserved so pipeline.py needs no changes.

Standard-library only: sqlite3, json, datetime, pathlib, logging.
"""

import json
import logging
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

_DEFAULT_DB_PATH = "arcs_engagements.db"


# ══════════════════════════════════════════════════════════════════════════════
# EngagementDatabase
# ══════════════════════════════════════════════════════════════════════════════

class EngagementDatabase:
    """
    SQLite-backed store for weapon bias profiles, fire missions, and round data.

    Uses a single persistent connection with WAL journalling for concurrent
    read safety. DB errors are caught and logged; they never crash the caller.

    Usage:
        db = EngagementDatabase()                   # opens arcs_engagements.db
        db = EngagementDatabase("arcs_test.db")     # explicit path

        # New per-weapon interface (Steps 2–3):
        profile = db.get_weapon_profile("VAJRA-07") # None on first use
        eid = db.record_engagement("VAJRA-07", ...)
        db.upsert_weapon_profile("VAJRA-07", ...)

        # Legacy interface (backward compat with pipeline.py):
        eid_uuid = db.log({...})
        stats    = db.statistics()
        inputs   = db.get_sbe_inputs()
    """

    def __init__(self, db_path: str = _DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._init_schema()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        c = self._conn
        c.execute("""
            CREATE TABLE IF NOT EXISTS weapon_profiles (
                weapon_id            TEXT PRIMARY KEY,
                weapon_type          TEXT,
                b_sag                REAL NOT NULL DEFAULT 0.499,
                b_yaw                REAL NOT NULL DEFAULT 0.0,
                b_v0                 REAL NOT NULL DEFAULT 0.0,
                n_engagements        INTEGER DEFAULT 0,
                total_rounds_fired   INTEGER DEFAULT 0,
                confidence           REAL DEFAULT 0.0,
                created_at           TEXT,
                updated_at           TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS engagements (
                engagement_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                legacy_id            TEXT,
                weapon_id            TEXT,
                target_range         REAL,
                target_bearing       REAL,
                target_height        REAL DEFAULT 0.0,
                pitch_deg            REAL,
                temperature_c        REAL DEFAULT 15.0,
                altitude_m           REAL DEFAULT 0.0,
                uncorrected_cep      REAL,
                corrected_cep        REAL,
                improvement_pct      REAL,
                rounds_to_converge   INTEGER,
                total_shots          INTEGER,
                warm_started         INTEGER DEFAULT 0,
                delta_pitch          REAL,
                delta_yaw            REAL,
                delta_v0             REAL,
                converged            INTEGER DEFAULT 0,
                timestamp            TEXT,
                full_record          TEXT,
                FOREIGN KEY(weapon_id) REFERENCES weapon_profiles(weapon_id)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS rounds (
                round_id             INTEGER PRIMARY KEY AUTOINCREMENT,
                engagement_id        INTEGER NOT NULL,
                round_number         INTEGER,
                phase                TEXT,
                elevation_deg        REAL,
                azimuth_deg          REAL,
                muzzle_v             REAL,
                result_range         REAL,
                miss_distance        REAL,
                correction_pitch     REAL,
                correction_yaw       REAL,
                correction_v0        REAL,
                FOREIGN KEY(engagement_id) REFERENCES engagements(engagement_id)
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_wp_weapon   ON weapon_profiles(weapon_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_eng_weapon  ON engagements(weapon_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_eng_range   ON engagements(target_range)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_eng_bearing ON engagements(target_bearing)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_rnd_eid     ON rounds(engagement_id)")
        c.commit()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ══════════════════════════════════════════════════════════════════════════
    # Per-weapon profile methods (Step 2)
    # ══════════════════════════════════════════════════════════════════════════

    def get_weapon_profile(self, weapon_id: str) -> Optional[Dict]:
        """
        Return the stored bias profile for weapon_id, or None if this weapon
        has never fired before. This is the warm-start lookup: a non-None return
        means the engagement can skip cold-start BO exploration.

        Returns dict with: weapon_id, weapon_type, b_sag, b_yaw, b_v0,
                           n_engagements, total_rounds_fired, confidence.
        """
        try:
            row = self._conn.execute(
                "SELECT weapon_id, weapon_type, b_sag, b_yaw, b_v0, "
                "n_engagements, total_rounds_fired, confidence "
                "FROM weapon_profiles WHERE weapon_id = ?",
                (weapon_id,)
            ).fetchone()
            if row is None:
                return None
            return dict(row)
        except Exception as exc:
            log.error("get_weapon_profile(%s) failed: %s", weapon_id, exc)
            return None

    def upsert_weapon_profile(self, weapon_id: str, weapon_type: str,
                               b_sag: float, b_yaw: float, b_v0: float,
                               n_engagements: int, total_rounds_fired: int,
                               confidence: float) -> None:
        """
        Insert a new weapon profile or update an existing one.

        Uses INSERT ... ON CONFLICT DO UPDATE so callers never need to check
        whether the weapon already exists.

        Sets updated_at to now; sets created_at only on first insert.
        """
        now = self._now()
        try:
            self._conn.execute("""
                INSERT INTO weapon_profiles
                    (weapon_id, weapon_type, b_sag, b_yaw, b_v0,
                     n_engagements, total_rounds_fired, confidence,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(weapon_id) DO UPDATE SET
                    weapon_type        = excluded.weapon_type,
                    b_sag              = excluded.b_sag,
                    b_yaw              = excluded.b_yaw,
                    b_v0               = excluded.b_v0,
                    n_engagements      = excluded.n_engagements,
                    total_rounds_fired = excluded.total_rounds_fired,
                    confidence         = excluded.confidence,
                    updated_at         = excluded.updated_at
            """, (weapon_id, weapon_type,
                  float(b_sag), float(b_yaw), float(b_v0),
                  int(n_engagements), int(total_rounds_fired), float(confidence),
                  now, now))
            self._conn.commit()
        except Exception as exc:
            log.error("upsert_weapon_profile(%s) failed: %s", weapon_id, exc)

    def record_engagement(self, weapon_id: str,
                           target_range: float, target_bearing: float,
                           target_height: float, uncorrected_cep: float,
                           corrected_cep: float, improvement_pct: float,
                           rounds_to_converge: int, total_shots: int,
                           warm_started: int, delta_pitch: float,
                           delta_yaw: float, delta_v0: float,
                           temperature_c: float = 15.0,
                           altitude_m: float = 0.0) -> int:
        """
        Insert one engagement row.

        Returns the new engagement_id (integer) to link round records.
        Returns -1 on error (DB errors must not crash an engagement).
        """
        try:
            cur = self._conn.execute("""
                INSERT INTO engagements
                    (weapon_id, target_range, target_bearing, target_height,
                     temperature_c, altitude_m,
                     uncorrected_cep, corrected_cep, improvement_pct,
                     rounds_to_converge, total_shots, warm_started,
                     delta_pitch, delta_yaw, delta_v0, converged, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (weapon_id,
                  float(target_range), float(target_bearing), float(target_height),
                  float(temperature_c), float(altitude_m),
                  float(uncorrected_cep) if uncorrected_cep is not None else None,
                  float(corrected_cep)   if corrected_cep   is not None else None,
                  float(improvement_pct) if improvement_pct is not None else None,
                  int(rounds_to_converge) if rounds_to_converge is not None else None,
                  int(total_shots)        if total_shots        is not None else None,
                  int(warm_started),
                  float(delta_pitch) if delta_pitch is not None else None,
                  float(delta_yaw)   if delta_yaw   is not None else None,
                  float(delta_v0)    if delta_v0    is not None else None,
                  1,
                  self._now()))
            self._conn.commit()
            return cur.lastrowid
        except Exception as exc:
            log.error("record_engagement(%s) failed: %s", weapon_id, exc)
            return -1

    def record_round(self, engagement_id: int, round_number: int,
                      phase: str,
                      elevation_deg: Optional[float],
                      azimuth_deg:   Optional[float],
                      muzzle_v:      Optional[float],
                      result_range:  Optional[float],
                      miss_distance: Optional[float],
                      correction_pitch: Optional[float],
                      correction_yaw:   Optional[float],
                      correction_v0:    Optional[float]) -> None:
        """
        Insert one round row. Any parameter may be None — missing telemetry
        is stored as NULL rather than raising an exception.
        """
        try:
            self._conn.execute("""
                INSERT INTO rounds
                    (engagement_id, round_number, phase,
                     elevation_deg, azimuth_deg, muzzle_v,
                     result_range, miss_distance,
                     correction_pitch, correction_yaw, correction_v0)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (int(engagement_id), round_number, phase,
                  elevation_deg, azimuth_deg, muzzle_v,
                  result_range, miss_distance,
                  correction_pitch, correction_yaw, correction_v0))
            self._conn.commit()
        except Exception as exc:
            log.error("record_round(eid=%s) failed: %s", engagement_id, exc)

    def record_rounds_from_history(self, engagement_id: int, history_df) -> None:
        """
        Write each row of the BO history DataFrame to the rounds table.

        Inspects the DataFrame's actual columns at runtime; any column that is
        absent is stored as NULL rather than raising an exception. The history
        DataFrame returned by EngagementSimulator.run_engagement() has:
            suggestion, delta_pitch, delta_yaw, delta_v0, avg_miss, best_miss, kappa
        """
        if history_df is None or len(history_df) == 0:
            return
        cols = set(history_df.columns)
        try:
            rows = []
            for _, row in history_df.iterrows():
                def _get(col):
                    return float(row[col]) if col in cols else None
                def _geti(col):
                    return int(row[col]) if col in cols else None
                rows.append((
                    int(engagement_id),
                    _geti('suggestion'),
                    'ADJUSTMENT',
                    None,         # elevation_deg — not in BO history
                    None,         # azimuth_deg
                    None,         # muzzle_v
                    None,         # result_range
                    _get('avg_miss'),
                    _get('delta_pitch'),
                    _get('delta_yaw'),
                    _get('delta_v0'),
                ))
            self._conn.executemany("""
                INSERT INTO rounds
                    (engagement_id, round_number, phase,
                     elevation_deg, azimuth_deg, muzzle_v,
                     result_range, miss_distance,
                     correction_pitch, correction_yaw, correction_v0)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            self._conn.commit()
        except Exception as exc:
            log.error("record_rounds_from_history(eid=%s) failed: %s",
                      engagement_id, exc)

    def get_engagement_history(self, weapon_id: str,
                                limit: int = 50) -> List[Dict]:
        """
        Return past engagement records for weapon_id, newest first.

        Returns a list of dicts with all engagement columns, suitable for
        computing convergence trends or confirming engagement #2 beat #1.
        """
        try:
            rows = self._conn.execute("""
                SELECT engagement_id, weapon_id,
                       target_range, target_bearing, target_height,
                       uncorrected_cep, corrected_cep, improvement_pct,
                       rounds_to_converge, total_shots, warm_started,
                       delta_pitch, delta_yaw, delta_v0, timestamp
                FROM engagements
                WHERE weapon_id = ?
                ORDER BY engagement_id DESC
                LIMIT ?
            """, (weapon_id, int(limit))).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            log.error("get_engagement_history(%s) failed: %s", weapon_id, exc)
            return []

    def weapon_summary(self, weapon_id: str) -> str:
        """
        Human-readable summary of a weapon's learned state.

        Format: WEAPON_ID | type=... | n_eng=N | b_sag=... b_yaw=... b_v0=...
                | conf=0.XX | mean_imp=+XX.X%
        """
        try:
            profile = self.get_weapon_profile(weapon_id)
            if profile is None:
                return f"{weapon_id}: no profile found"
            row = self._conn.execute("""
                SELECT COUNT(*), AVG(improvement_pct)
                FROM engagements
                WHERE weapon_id = ? AND improvement_pct IS NOT NULL
            """, (weapon_id,)).fetchone()
            n_in_db  = row[0] or 0
            mean_imp = row[1]
            imp_str  = f"{mean_imp:+.1f}%" if mean_imp is not None else "n/a"
            return (
                f"{weapon_id} | type={profile['weapon_type']} | "
                f"n_eng={profile['n_engagements']} | "
                f"b_sag={profile['b_sag']:.4f} "
                f"b_yaw={profile['b_yaw']:+.4f}° "
                f"b_v0={profile['b_v0']:+.3f}m/s | "
                f"conf={profile['confidence']:.2f} | "
                f"mean_imp={imp_str} (from {n_in_db} missions)"
            )
        except Exception as exc:
            log.error("weapon_summary(%s) failed: %s", weapon_id, exc)
            return f"{weapon_id}: error — {exc}"

    def close(self) -> None:
        """Commit any pending writes and close the connection."""
        try:
            if self._conn:
                self._conn.commit()
                self._conn.close()
                self._conn = None
        except Exception as exc:
            log.error("close() failed: %s", exc)

    def __del__(self):
        try:
            if getattr(self, '_conn', None):
                self._conn.commit()
                self._conn.close()
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════════
    # Legacy interface — preserved for pipeline.py and existing tests
    # ══════════════════════════════════════════════════════════════════════════

    def log(self, record: Dict) -> str:
        """
        Legacy: persist a full engagement record dict.

        Assigns a UUID engagement_id and a Unix timestamp, stores the full JSON
        blob in full_record (for get_all()), and extracts scalar columns for
        get_sbe_inputs() and statistics().

        Returns: UUID string (backward compat with pipeline.py).
        """
        eid = str(uuid.uuid4())
        record = dict(record)
        record['engagement_id'] = eid
        record['timestamp']     = time.time()

        r    = record.get('results', {}) or {}
        sol  = record.get('nominal_solution', {}) or {}
        tgt  = record.get('target', {}) or {}
        est  = record.get('estimator', {}) or {}
        sbe  = record.get('sbe_input', {}) or {}

        converged_val = bool(est.get('converged', est.get('rls_converged', False)))

        dp = sbe.get('dp_opt')
        if dp is None:
            dp = (r.get('best_correction') or {}).get('delta_pitch')
        db_val = sbe.get('db_opt')
        if db_val is None:
            db_val = (r.get('best_correction') or {}).get('delta_yaw')
        dv = sbe.get('dv_opt')
        if dv is None:
            dv = (r.get('best_correction') or {}).get('delta_v0')

        pitch = sol.get('pitch_deg', sbe.get('pitch_deg_nominal', 0.0))
        height = tgt.get('height_m', tgt.get('y', 0.0))

        try:
            self._conn.execute("""
                INSERT INTO engagements
                    (legacy_id, weapon_id,
                     target_range, target_bearing, target_height, pitch_deg,
                     uncorrected_cep, corrected_cep, improvement_pct,
                     delta_pitch, delta_yaw, delta_v0,
                     converged, timestamp, full_record)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                eid, None,
                float(tgt.get('range', 0.0)),
                float(tgt.get('bearing_deg', 0.0)),
                float(height),
                float(pitch),
                r.get('baseline_cep_m'),
                r.get('corrected_cep_m'),
                r.get('improvement_pct'),
                dp, db_val, dv,
                int(converged_val),
                str(time.time()),
                json.dumps(record),
            ))
            self._conn.commit()
        except Exception as exc:
            log.error("log() failed: %s", exc)
        return eid

    def get_all(self) -> List[Dict]:
        """Legacy: return all engagement records as list of dicts, in insertion order."""
        try:
            rows = self._conn.execute(
                "SELECT full_record FROM engagements "
                "WHERE full_record IS NOT NULL ORDER BY engagement_id"
            ).fetchall()
            return [json.loads(r[0]) for r in rows]
        except Exception as exc:
            log.error("get_all() failed: %s", exc)
            return []

    def get_sbe_inputs(self) -> List[Dict]:
        """
        Legacy: return {pitch_deg, dp_opt, db_opt, dv_opt} for converged
        engagements. Used by pipeline.py to warm-start the SBE on restart.
        """
        try:
            rows = self._conn.execute("""
                SELECT pitch_deg, delta_pitch, delta_yaw, delta_v0
                FROM engagements
                WHERE delta_yaw IS NOT NULL AND converged = 1
                ORDER BY engagement_id
            """).fetchall()
            return [
                {'pitch_deg': r[0], 'dp_opt': r[1], 'db_opt': r[2], 'dv_opt': r[3]}
                for r in rows
            ]
        except Exception as exc:
            log.error("get_sbe_inputs() failed: %s", exc)
            return []

    def statistics(self) -> Dict:
        """Legacy: aggregate metrics across all logged engagements."""
        try:
            row = self._conn.execute("""
                SELECT COUNT(*), AVG(improvement_pct),
                       AVG(uncorrected_cep), AVG(corrected_cep)
                FROM engagements WHERE improvement_pct IS NOT NULL
            """).fetchone()
            return {
                'n_engagements':        row[0] or 0,
                'mean_improvement_pct': row[1],
                'mean_baseline_cep':    row[2],
                'mean_corrected_cep':   row[3],
            }
        except Exception as exc:
            log.error("statistics() failed: %s", exc)
            return {'n_engagements': 0, 'mean_improvement_pct': None,
                    'mean_baseline_cep': None, 'mean_corrected_cep': None}

    def clear(self, confirm: bool = False) -> None:
        """Delete all rows from all tables. Pass confirm=True to execute."""
        if not confirm:
            raise ValueError("Pass confirm=True to clear the database.")
        try:
            self._conn.execute("DELETE FROM rounds")
            self._conn.execute("DELETE FROM engagements")
            self._conn.execute("DELETE FROM weapon_profiles")
            self._conn.commit()
        except Exception as exc:
            log.error("clear() failed: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# run_persistent_engagement — orchestration (Step 3)
# ══════════════════════════════════════════════════════════════════════════════

def run_persistent_engagement(db: EngagementDatabase,
                               sim,
                               sbe,
                               weapon_id: str,
                               weapon_type: str,
                               target_x: float,
                               target_y: float,
                               target_z: float,
                               v0: float = 100.0,
                               prefer: str = "LOW",
                               verbose: bool = False) -> Optional[Dict]:
    """
    Run one engagement with full memory: warm-start lookup, bias persistence,
    and round logging.

    Steps:
      1. Pre-solve the target to get the firing pitch angle for SBE.predict().
      2. Look up weapon_id in db.
         - Found  → load bias into sbe via sbe.load_state(); warm_started=1.
         - Not found → cold start; warm_started=0.
      3. Get pre-correction from sbe.predict() (or None for cold start).
      4. Run sim.run_engagement() with the pre-correction injected.
      5. Update sbe with the engagement result.
      6. Persist: upsert_weapon_profile, record_engagement, record_rounds.
      7. Return result dict enriched with engagement_id and warm_started.

    Returns None if the target is unreachable.

    Args:
        db:          EngagementDatabase instance.
        sim:         EngagementSimulator instance.
        sbe:         StructuredBiasEstimator instance (holds cross-eng learning).
        weapon_id:   Unique weapon identifier (e.g. "VAJRA-07").
        weapon_type: Human-readable weapon type string.
        target_x/y/z: Target coordinates (m) in ARCS frame.
        v0:          Muzzle velocity (m/s).
        prefer:      Ballistic solution branch ("LOW" or "HIGH").
        verbose:     If True, print per-shot BO progress.
    """
    # Step 1 — pre-solve to get pitch angle for SBE prediction
    pre_sol = sim.solver.solve(target_x, target_y, target_z, v0, prefer=prefer)
    if not pre_sol.reachable:
        return None

    # Step 2 — warm-start lookup
    profile     = db.get_weapon_profile(weapon_id)
    warm_started = 0
    pre_correction = None

    if profile is not None:
        sbe.load_state(
            b_sag=profile['b_sag'],
            b_yaw=profile['b_yaw'],
            b_v0=profile['b_v0'],
            n_engagements=profile['n_engagements'],
        )
        warm_started = 1
        # Step 3 — derive pre-correction from warm-started SBE
        pre_correction = sbe.predict(pre_sol.turret_pitch_deg, v0)
        pre_correction['source'] = f"sbe_warm(n={profile['n_engagements']})"

    # Step 4 — run the engagement
    result = sim.run_engagement(
        target_x, target_y, target_z,
        v0=v0, gp_pre_correction=pre_correction,
        prefer=prefer, verbose=verbose,
    )
    if result is None:
        return None

    best_corr = result['best_correction']  # numpy array [dp, dy, dv]

    # Step 5 — update SBE with engagement outcome
    sbe.update_engagement(
        pitch_deg=float(pre_sol.turret_pitch_deg),
        db_opt=float(best_corr[1]),
        dv_opt=float(best_corr[2]),
        dp_opt=float(best_corr[0]),
    )

    # Step 6a — persist weapon profile (upsert with updated bias + round count)
    prev_rounds = profile['total_rounds_fired'] if profile else 0
    db.upsert_weapon_profile(
        weapon_id=weapon_id,
        weapon_type=weapon_type,
        b_sag=float(sbe._b_sag.get()),
        b_yaw=float(sbe._b_yaw.get()),
        b_v0=float(sbe._b_v0.get()),
        n_engagements=int(sbe._n_engagements),
        total_rounds_fired=int(prev_rounds) + int(result['total_shots']),
        confidence=float(sbe.confidence()),
    )

    # Step 6b — record engagement
    eid = db.record_engagement(
        weapon_id=weapon_id,
        target_range=float(result['horiz_range']),
        target_bearing=float(pre_sol.turret_yaw_deg),
        target_height=float(target_y),
        uncorrected_cep=float(result['baseline_cep']),
        corrected_cep=float(result['verified_cep']),
        improvement_pct=float(result['improvement_pct']),
        rounds_to_converge=int(result.get('n_bo_shots', 0)),
        total_shots=int(result['total_shots']),
        warm_started=int(warm_started),
        delta_pitch=float(best_corr[0]),
        delta_yaw=float(best_corr[1]),
        delta_v0=float(best_corr[2]),
    )

    # Step 6c — record per-round BO history
    history = result.get('history')
    if history is not None and len(history) > 0:
        db.record_rounds_from_history(eid, history)

    # Step 7 — return enriched result
    return {**result, 'engagement_id': eid, 'warm_started': warm_started}


# ══════════════════════════════════════════════════════════════════════════════
# Self-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, tempfile

    print("=" * 64)
    print("ARCS — EngagementDatabase v2.0 self-test")
    print("=" * 64)

    tmpdir = Path(tempfile.mkdtemp())
    db = EngagementDatabase(db_path=str(tmpdir / "selftest.db"))

    # ── Legacy interface tests ────────────────────────────────────────────────
    print("\n  [1] Legacy log() / get_all() / statistics() / clear()")

    r1 = {
        'target':           {'range': 300.0, 'bearing_deg': 0.0, 'y': 0.0},
        'nominal_solution': {'pitch_deg': 8.6, 'v0': 100.0},
        'results':          {'baseline_cep_m': 8.5, 'corrected_cep_m': 5.1,
                             'improvement_pct': 40.0,
                             'best_correction': {'delta_pitch': 0.08,
                                                 'delta_yaw': 0.12,
                                                 'delta_v0': -2.1}},
        'estimator':        {'converged': True},
        'sbe_input':        {'pitch_deg_nominal': 8.6,
                             'dp_opt': 0.08, 'db_opt': 0.12, 'dv_opt': -2.1},
    }
    eid1 = db.log(r1)
    eid2 = db.log({**r1, 'results': {**r1['results'], 'improvement_pct': 30.0}})

    assert isinstance(eid1, str) and eid1.count('-') == 4, "log() must return UUID"
    print(f"    log() UUID: {eid1[:8]}... ✓")

    all_recs = db.get_all()
    assert len(all_recs) == 2, f"get_all(): expected 2, got {len(all_recs)}"
    assert 'engagement_id' in all_recs[0], "engagement_id missing from record"
    print(f"    get_all(): {len(all_recs)} records ✓")

    stats = db.statistics()
    assert stats['n_engagements'] == 2
    assert abs(stats['mean_improvement_pct'] - 35.0) < 0.1, \
        f"mean_imp={stats['mean_improvement_pct']:.2f}"
    print(f"    statistics(): n={stats['n_engagements']}, "
          f"mean_imp={stats['mean_improvement_pct']:.1f}% ✓")

    sbe_in = db.get_sbe_inputs()
    assert len(sbe_in) == 2
    assert 'pitch_deg' in sbe_in[0] and 'db_opt' in sbe_in[0]
    print(f"    get_sbe_inputs(): {len(sbe_in)} rows, keys={list(sbe_in[0])} ✓")

    # ── New weapon-profile interface ──────────────────────────────────────────
    print("\n  [2] Weapon profile: upsert / get / summary")

    assert db.get_weapon_profile("TEST-01") is None, "new weapon must return None"
    print("    get_weapon_profile(unknown) → None ✓")

    db.upsert_weapon_profile("TEST-01", "demo_cannon",
                              b_sag=0.49, b_yaw=-0.12, b_v0=2.1,
                              n_engagements=5, total_rounds_fired=50,
                              confidence=0.63)
    p = db.get_weapon_profile("TEST-01")
    assert p is not None and abs(p['b_v0'] - 2.1) < 1e-6
    print(f"    upsert + get_weapon_profile ✓  b_v0={p['b_v0']:.3f}")

    db.upsert_weapon_profile("TEST-01", "demo_cannon",
                              b_sag=0.49, b_yaw=-0.12, b_v0=2.05,
                              n_engagements=6, total_rounds_fired=60,
                              confidence=0.70)
    p2 = db.get_weapon_profile("TEST-01")
    assert abs(p2['b_v0'] - 2.05) < 1e-6, "upsert should update existing"
    assert p2['n_engagements'] == 6
    print(f"    upsert (update) ✓  b_v0={p2['b_v0']:.3f}, n_eng={p2['n_engagements']}")

    # ── record_engagement / rounds / history ──────────────────────────────────
    print("\n  [3] record_engagement / record_round / get_engagement_history")

    eid_int = db.record_engagement(
        weapon_id="TEST-01", target_range=300.0, target_bearing=0.0,
        target_height=0.0, uncorrected_cep=8.5, corrected_cep=5.1,
        improvement_pct=40.0, rounds_to_converge=12, total_shots=45,
        warm_started=0, delta_pitch=0.08, delta_yaw=0.12, delta_v0=-2.1,
    )
    assert isinstance(eid_int, int) and eid_int > 0, \
        f"record_engagement must return int > 0, got {eid_int}"
    print(f"    record_engagement → id={eid_int} ✓")

    db.record_round(eid_int, 1, "ADJUSTMENT",
                    elevation_deg=8.6, azimuth_deg=0.0, muzzle_v=100.0,
                    result_range=298.5, miss_distance=3.2,
                    correction_pitch=0.05, correction_yaw=0.10, correction_v0=-1.8)
    db.record_round(eid_int, 2, "FIRE_FOR_EFFECT",
                    elevation_deg=8.68, azimuth_deg=0.12, muzzle_v=97.9,
                    result_range=300.1, miss_distance=0.8,
                    correction_pitch=0.08, correction_yaw=0.12, correction_v0=-2.1)

    rounds = db._conn.execute(
        "SELECT COUNT(*) FROM rounds WHERE engagement_id = ?", (eid_int,)
    ).fetchone()[0]
    assert rounds == 2, f"Expected 2 rounds, got {rounds}"
    print(f"    2 rounds stored ✓")

    history = db.get_engagement_history("TEST-01")
    assert len(history) == 1 and history[0]['engagement_id'] == eid_int
    print(f"    get_engagement_history: {len(history)} records ✓")

    # ── weapon_summary ────────────────────────────────────────────────────────
    summary = db.weapon_summary("TEST-01")
    assert "TEST-01" in summary and "b_v0" in summary
    print(f"\n  [4] weapon_summary:\n      {summary}")

    # ── clear ─────────────────────────────────────────────────────────────────
    print("\n  [5] clear()")
    try:
        db.clear(confirm=False)
        print("    ERROR: should have raised ValueError")
        sys.exit(1)
    except ValueError:
        print("    clear(confirm=False) → ValueError ✓")
    db.clear(confirm=True)
    assert db.statistics()['n_engagements'] == 0
    assert db.get_weapon_profile("TEST-01") is None
    print("    clear(confirm=True) → empty ✓")

    db.close()
    print("\n  engagement_database.py v2.0 ✓")
