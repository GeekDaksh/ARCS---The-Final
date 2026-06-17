"""
Component 6 — Full engagement loop + persistence.

Wires the proven Phase 2 pieces into one complete fire mission:
  * physics core (Components 1-4.5): the frozen integrate_trajectory, called
    with met= and use_g7=True for every shot;
  * the two SEPARATE estimators (Component 5): GunBiasEstimator (Estimator A)
    and AtmosphericStateEstimator (Estimator B), used unchanged;
  * the three-phase doctrine (register -> adjust -> fire for effect);
  * the Phase 1 engagement database (weapon_profiles / engagements / rounds),
    extended with a few added columns for the Phase 2 gun bias + atmospheric
    correction (columns added via ALTER TABLE; the schema is never rewritten).

This is INTEGRATION, not new physics. Nothing in the frozen physics files or
the validated estimators is modified — they are called as tools. The hidden
true conditions (real wind error + real gun bias) drive the fall of shot; the
system only ever sees the miss. numpy + stdlib + sqlite3 only.
"""

import numpy as np

from physics.trajectory import integrate_trajectory
from physics.met_message import MetMessage
from physics.state_estimator import (GunBiasEstimator, AtmosphericStateEstimator,
                                     wind_sensitivity)

V0 = 827.0  # standard 155mm muzzle velocity (charge)


# ===========================================================================
# Range/cross <-> world rotation about the firing azimuth.
# ===========================================================================
def _frame(bearing_deg):
    phi = np.radians(bearing_deg)
    u_dr = np.array([np.cos(phi), np.sin(phi)])    # downrange unit (world xy)
    u_cr = np.array([-np.sin(phi), np.cos(phi)])   # cross-range unit
    return u_dr, u_cr


def _rc_to_world(vec_rc, bearing_deg):
    u_dr, u_cr = _frame(bearing_deg)
    return vec_rc[0] * u_dr + vec_rc[1] * u_cr


def _world_to_rc(vec_world, bearing_deg):
    u_dr, u_cr = _frame(bearing_deg)
    return np.array([np.dot(vec_world, u_dr), np.dot(vec_world, u_cr)])


# ===========================================================================
# MET with an added uniform wind (the effective atmospheric correction).
# ===========================================================================
def _met_plus_uniform_wind(base_met, add_world_xy):
    """Return a new MetMessage equal to base_met but with a uniform wind vector
    (world xy) added to every zone. Used to bake the estimated/true effective
    wind correction into the MET that the frozen integrator consumes."""
    ax, ay = float(add_world_xy[0]), float(add_world_xy[1])
    lines = []
    for ln in base_met.lines:
        vec = ln["wind_vector"]
        vx, vy = vec[0] + ax, vec[1] + ay
        speed = float(np.hypot(vx, vy))
        # Invert wind_vector_from_dir_speed: vec = speed*(-cos a, sin a).
        wdir = float(np.degrees(np.arctan2(vy, -vx)) % 360.0) if speed > 1e-12 else 0.0
        lines.append({
            "zone_top_m": ln["zone_top_m"],
            "wind_dir_deg": wdir,
            "wind_speed_ms": speed,
            "temp_C": ln["temp_C"],
            "pressure_Pa": ln["pressure_Pa"],
        })
    return MetMessage(lines)


def _impact_rc(met, elevation_deg, bearing_deg):
    """Impact of one shot through the frozen integrator, in the range/cross
    frame relative to the firing azimuth."""
    r = integrate_trajectory(v0=V0, elevation_deg=elevation_deg,
                             azimuth_deg=bearing_deg, met=met, use_g7=True)
    return _world_to_rc(np.array([r["impact_x"], r["impact_y"]]), bearing_deg)


def _solve_elevation(met, target_range, bearing_deg, lo=15.0, hi=48.0):
    """Fire-control solution: elevation (ascending branch) whose downrange under
    `met` equals target_range. Simple bisection on the monotonic branch."""
    def dr(e):
        return _impact_rc(met, e, bearing_deg)[0] - target_range
    flo, fhi = dr(lo), dr(hi)
    if flo > 0:  # target shorter than even the minimum elevation reaches
        return lo
    if fhi < 0:  # target beyond max range at hi; clamp
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fm = dr(mid)
        if abs(fm) < 1e-3:
            return mid
        if fm > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ===========================================================================
# Database schema extension (added columns only — never a rewrite).
# ===========================================================================
def _ensure_phase2_columns(db):
    conn = db._conn
    def cols(table):
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    wp = cols("weapon_profiles")
    for c in ("gun_bias_dr", "gun_bias_cr"):
        if c not in wp:
            conn.execute(f"ALTER TABLE weapon_profiles ADD COLUMN {c} REAL DEFAULT 0.0")
    eng = cols("engagements")
    for c in ("atmo_dwind_dr", "atmo_dwind_cr", "final_cep"):
        if c not in eng:
            conn.execute(f"ALTER TABLE engagements ADD COLUMN {c} REAL")
    conn.commit()


def _load_gun_bias(db, weapon_id):
    """Return remembered (dr, cr) gun bias for a weapon, or None if new."""
    row = db._conn.execute(
        "SELECT gun_bias_dr, gun_bias_cr FROM weapon_profiles WHERE weapon_id=?",
        (weapon_id,)).fetchone()
    if row is None:
        return None
    dr = row[0] if not hasattr(row, "keys") else row["gun_bias_dr"]
    cr = row[1] if not hasattr(row, "keys") else row["gun_bias_cr"]
    if dr is None and cr is None:
        return None
    return np.array([dr or 0.0, cr or 0.0])


def _save_gun_bias(db, weapon_id, weapon_type, gun_bias, n_eng, total_rounds):
    conn = db._conn
    now = db._now()
    conn.execute("""
        INSERT INTO weapon_profiles (weapon_id, weapon_type, gun_bias_dr,
            gun_bias_cr, n_engagements, total_rounds_fired, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(weapon_id) DO UPDATE SET
            weapon_type=excluded.weapon_type,
            gun_bias_dr=excluded.gun_bias_dr,
            gun_bias_cr=excluded.gun_bias_cr,
            n_engagements=excluded.n_engagements,
            total_rounds_fired=excluded.total_rounds_fired,
            updated_at=excluded.updated_at
    """, (weapon_id, weapon_type, float(gun_bias[0]), float(gun_bias[1]),
          int(n_eng), int(total_rounds), now, now))
    conn.commit()


# ===========================================================================
# The engagement loop.
# ===========================================================================
def run_engagement(weapon_id, target_range, target_bearing,
                   met_message, true_conditions, db=None,
                   n_register=2, n_adjust=4, n_ffe=4,
                   weapon_type="155mm-M107", converge_threshold_m=40.0):
    """
    Runs a complete Phase 2 fire mission against a target.

    true_conditions: dict carrying the HIDDEN truth the system is not told:
        {"true_met": MetMessage,            # the real atmosphere (weather)
         "gun_bias": (dr_m, cr_m)}          # the gun's real constant offset
      The real fall of shot = integrate(true_met) + gun_bias. The told
      `met_message` is the imperfect MET the fire-control computer uses.

    Doctrine:
      1. REGISTRATION — fire under a known-good MET (== true_met) so every miss
         isolates the gun's mechanical bias -> GunBiasEstimator (A). A remembered
         bias from the DB warm-starts this phase.
      2. ADJUSTMENT  — fire under the imperfect told MET; remove the known gun
         bias; the residual miss is atmospheric -> AtmosphericStateEstimator (B).
      3. FIRE FOR EFFECT — lock both corrections, fire the mission rounds.
      4. PERSIST     — write the learned gun bias + engagement record to the DB.
    """
    true_met = true_conditions["true_met"]
    true_gun = np.asarray(true_conditions["gun_bias"], dtype=float)

    # Fire-control elevation from the TOLD MET (the nominal solution).
    elevation = _solve_elevation(met_message, target_range, target_bearing)

    # Wind sensitivity H (m per m/s), measured from the frozen integrator by
    # perturbing the told MET by +1 m/s uniform in each horizontal axis.
    base_imp = _impact_rc(met_message, elevation, target_bearing)
    h_dr = _impact_rc(_met_plus_uniform_wind(met_message, _rc_to_world([1.0, 0.0], target_bearing)),
                      elevation, target_bearing) - base_imp
    h_cr = _impact_rc(_met_plus_uniform_wind(met_message, _rc_to_world([0.0, 1.0], target_bearing)),
                      elevation, target_bearing) - base_imp
    H = np.column_stack([h_dr, h_cr])

    # --- Estimators (Component 5, unchanged). Warm-start A from the DB. ---
    warm = None
    if db is not None:
        _ensure_phase2_columns(db)
        warm = _load_gun_bias(db, weapon_id)
    warm_started = warm is not None
    estA = GunBiasEstimator(init=tuple(warm) if warm_started else (0.0, 0.0),
                            init_P=(100.0 if warm_started else 1.0e4), R=400.0)
    estB = AtmosphericStateEstimator([0.0, 0.0], np.diag([25.0, 25.0]),
                                     np.diag([0.05, 0.05]), np.diag([2500.0, 2500.0]))

    history = []          # list of (phase, miss_vector)
    round_no = 0
    rounds_to_converge = None

    def observed_rc():
        return _impact_rc(true_met, elevation, target_bearing) + true_gun

    def log_miss(phase, miss):
        nonlocal round_no
        round_no += 1
        radial = float(np.hypot(*miss))
        history.append({"phase": phase, "round": round_no,
                        "miss": miss.copy(), "radial": radial})

    # --- Phase 1: REGISTRATION (known-good MET == true_met; isolate gun) -----
    # Known-good MET means predicted and observed share the same atmosphere, so
    # the residual miss is PURE gun bias. Predicted includes the current gun
    # estimate, so the registration miss shrinks as A learns (and is small from
    # the first shot when warm-started). The estimator observes the ABSOLUTE
    # gun offset = miss + current estimate.
    for _ in range(n_register):
        predicted = _impact_rc(true_met, elevation, target_bearing) + estA.state()
        miss = observed_rc() - predicted               # = true_gun - estA
        log_miss("REGISTRATION", miss)
        estA.update(miss + estA.state())               # absolute gun observation
        if float(np.hypot(*miss)) < converge_threshold_m and round_no >= 1:
            break
    round_after_reg = round_no

    # --- Phase 2: ADJUSTMENT (imperfect MET; remove gun; isolate wind) -------
    for _ in range(n_adjust):
        corrected_met = _met_plus_uniform_wind(
            met_message, _rc_to_world(estB.state(), target_bearing))
        predicted = _impact_rc(corrected_met, elevation, target_bearing) + estA.state()
        miss = observed_rc() - predicted
        log_miss("ADJUSTMENT", miss)
        estB.predict()
        estB.update(miss, H)
        if float(np.hypot(*miss)) < 0.5 * converge_threshold_m and round_no >= round_after_reg + 1:
            break

    # Rounds-to-converge = the register+adjust rounds spent dialling in before
    # FFE. Warm-starting the gun bias from the DB cuts the registration rounds,
    # so a remembered weapon converges in fewer rounds.
    rounds_to_converge = round_no

    # --- Phase 3: FIRE FOR EFFECT (lock both corrections) -------------------
    corrected_met = _met_plus_uniform_wind(
        met_message, _rc_to_world(estB.state(), target_bearing))
    ffe_radials = []
    for _ in range(n_ffe):
        predicted = _impact_rc(corrected_met, elevation, target_bearing) + estA.state()
        miss = observed_rc() - predicted
        log_miss("FFE", miss)
        ffe_radials.append(float(np.hypot(*miss)))

    final_cep = float(np.median(ffe_radials)) if ffe_radials else None

    result = {
        "weapon_id": weapon_id,
        "target_range": target_range,
        "target_bearing": target_bearing,
        "elevation_deg": elevation,
        "warm_started": warm_started,
        "history": history,
        "phase_misses": {p: [h["radial"] for h in history if h["phase"] == p]
                         for p in ("REGISTRATION", "ADJUSTMENT", "FFE")},
        "final_cep": final_cep,
        "gun_bias_est": estA.state(),
        "atmo_correction_est": estB.state(),
        "rounds_to_converge": rounds_to_converge,
        "total_shots": round_no,
    }

    # --- Phase 4: PERSIST ---------------------------------------------------
    if db is not None:
        prof = db.get_weapon_profile(weapon_id)
        n_eng = (prof["n_engagements"] if prof else 0) + 1
        total_rounds = (prof["total_rounds_fired"] if prof else 0) + round_no
        _save_gun_bias(db, weapon_id, weapon_type, estA.state(), n_eng, total_rounds)

        reg0 = result["phase_misses"]["REGISTRATION"][0] if result["phase_misses"]["REGISTRATION"] else None
        eid = db.record_engagement(
            weapon_id=weapon_id, target_range=target_range,
            target_bearing=target_bearing, target_height=0.0,
            uncorrected_cep=reg0, corrected_cep=final_cep,
            improvement_pct=(100.0 * (1 - final_cep / reg0) if reg0 and final_cep is not None else None),
            rounds_to_converge=rounds_to_converge, total_shots=round_no,
            warm_started=1 if warm_started else 0,
            delta_pitch=None, delta_yaw=None, delta_v0=None)
        db._conn.execute(
            "UPDATE engagements SET atmo_dwind_dr=?, atmo_dwind_cr=?, final_cep=? "
            "WHERE engagement_id=?",
            (float(estB.state()[0]), float(estB.state()[1]),
             final_cep, int(eid)))
        db._conn.commit()
        for h in history:
            db.record_round(eid, h["round"], h["phase"], elevation,
                            target_bearing, V0, None, h["radial"], None, None, None)
        result["engagement_id"] = eid

    return result
