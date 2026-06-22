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
from physics.horizontal_met import HorizontalMetField
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
    """Return a new MET equal to base_met but with a uniform wind vector
    (world xy) added to every zone. Used to bake the estimated/true effective
    wind correction into the MET that the frozen integrator consumes.

    Component 12: a HorizontalMetField is handled by adding the uniform wind to
    BOTH its gun and target profiles and rebuilding the field (same target range
    and confidence floor), so the estimator's effective-wind correction applies
    to a horizontally-varying field exactly as it does to a plain MetMessage.
    The engagement loop is therefore unchanged whether it is given a MetMessage
    or a HorizontalMetField."""
    if getattr(base_met, "is_horizontal", False):
        return HorizontalMetField(
            _met_plus_uniform_wind(base_met.gun_met, add_world_xy),
            _met_plus_uniform_wind(base_met.target_met, add_world_xy),
            base_met.target_range_m,
            confidence_floor=base_met.confidence_floor)
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
                   weapon_type="155mm-M107", converge_threshold_m=40.0,
                   observation_noise_m=0.0, noise_seed=None):
    """
    Runs a complete Phase 2 fire mission against a target.

    true_conditions: dict carrying the HIDDEN truth the system is not told:
        {"true_met": MetMessage,            # the real atmosphere (weather)
         "gun_bias": (dr_m, cr_m)}          # the gun's real constant offset
      The real fall of shot = integrate(true_met) + gun_bias. The told
      `met_message` is the imperfect MET the fire-control computer uses.

    observation_noise_m: std dev (m, per axis) of the measurement error in the
      REPORTED fall of shot (Component 13). Nobody measures the impact with a
      ruler: a forward observer / drone / radar estimates the miss with error.
      The true physics is untouched — the shell really lands where the frozen
      integrator says — but the miss the ESTIMATORS learn from is the true miss
      plus N(0, observation_noise_m) per axis, and the Kalman/RLS measurement-
      noise term is raised to match. Default 0.0 reproduces perfect observation
      bit-for-bit. ~10-20 m is realistic for a competent optical/laser observer
      at multi-km range; a precise drone/radar is smaller, a degraded visual
      observer larger. noise_seed makes the realization reproducible.

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

    # --- Observation noise (Component 13): a seedable measurement error on the
    # REPORTED fall of shot. Applied only where the miss is handed to an
    # estimator, never to the true physics. obs_var feeds the filters' R term.
    obs_noise = float(observation_noise_m)
    if obs_noise < 0.0:
        raise ValueError("observation_noise_m must be >= 0")
    obs_var = obs_noise ** 2
    _rng = np.random.default_rng(noise_seed) if obs_noise > 0.0 else None

    def observe(true_miss):
        """The fall of shot the estimators LEARN from: the true miss as reported
        by a noisy sensor. With observation_noise_m=0 this is the true miss
        unchanged (bit-for-bit); otherwise true_miss + N(0, obs_noise) per axis."""
        if _rng is None:
            return true_miss
        return true_miss + _rng.normal(0.0, obs_noise, size=2)

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
    # The measurement-noise term is the filters' assumed model floor PLUS the
    # observation noise (the correct place to tell a Kalman filter how noisy its
    # measurements are). obs_var=0 leaves the established R values untouched.
    estA = GunBiasEstimator(init=tuple(warm) if warm_started else (0.0, 0.0),
                            init_P=(100.0 if warm_started else 1.0e4),
                            R=400.0 + obs_var)
    estB = AtmosphericStateEstimator([0.0, 0.0], np.diag([25.0, 25.0]),
                                     np.diag([0.05, 0.05]),
                                     np.diag([2500.0 + obs_var, 2500.0 + obs_var]))

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
        miss = observed_rc() - predicted               # = true_gun - estA (true)
        log_miss("REGISTRATION", miss)
        # The estimator learns from the NOISY reported miss; the true fall of
        # shot (logged above) is what really happened.
        estA.update(observe(miss) + estA.state())      # absolute gun observation
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
        estB.update(observe(miss), H)                  # learn from noisy report
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


# ===========================================================================
# Component 9 — fire until destroyed, then stop (adaptable lethal radius).
# A NEW function; run_engagement above is untouched. It reuses the same proven
# estimators and frozen integrator as tools.
# ===========================================================================
def run_engagement_until_destroyed(weapon_id, target_range, target_bearing,
                                   met_message, true_conditions, db=None,
                                   lethal_radius_m=8.0, max_rounds=15,
                                   min_learning_rounds=2,
                                   weapon_type="155mm-M107",
                                   target_height_m=0.0,
                                   observation_noise_m=0.0, noise_seed=None):
    """
    Fires, learns from each miss, and STOPS the instant a round lands within
    lethal_radius_m of the target (target destroyed) — or when max_rounds is
    reached (safety cap). Returns the shot-by-shot trace, whether the target
    was destroyed, the round number that destroyed it, and the final miss.

    lethal_radius_m: distance within which a round destroys the target.
      Adaptable — any positive value; default 8.0 (155mm standard). A precision
      use case might set 2 m, an area weapon 30 m; the model fits the case.
    max_rounds: safety cap so the mission can't fire forever.
    min_learning_rounds: fire at least this many ranging/adjustment rounds
      before it's allowed to declare a hit (the first cold shot at long range
      rarely lands within lethal radius; the system needs a couple of
      observations to learn the gun bias and atmospheric error).
    target_height_m: passed through to the integrator (Component 8).
    observation_noise_m: std dev (m, per axis) of measurement error on the
      REPORTED fall of shot (Component 13). The estimators learn from the noisy
      report; the KILL assessment uses the TRUE fall of shot (a round physically
      within lethal_radius is effect-achieved — a BDA system would confirm it,
      see DATA_SOURCES_AND_SCOPE.md). 0.0 reproduces perfect observation
      bit-for-bit. noise_seed makes the realization reproducible.

    Doctrine: a remembered weapon (gun bias in the DB) skips registration and
    fires for effect immediately; an unknown weapon fires a brief registration
    (known-good MET) to learn — and persist — its mechanical bias first. Either
    way the operation rounds aim at the target under the imperfect told MET, the
    atmospheric Kalman learns the residual from each miss, and the mission ends
    the instant a round falls within the lethal radius.
    """
    if lethal_radius_m <= 0.0:
        raise ValueError("lethal_radius_m must be positive")

    true_met = true_conditions["true_met"]
    true_gun = np.asarray(true_conditions["gun_bias"], dtype=float)
    told_met = met_message

    # Observation noise (Component 13) — applied only to the miss the estimators
    # learn from, never to the true physics or the kill assessment.
    obs_noise = float(observation_noise_m)
    if obs_noise < 0.0:
        raise ValueError("observation_noise_m must be >= 0")
    obs_var = obs_noise ** 2
    _rng = np.random.default_rng(noise_seed) if obs_noise > 0.0 else None

    def observe(true_miss):
        """Noisy sensor report of the true miss for the estimators. With
        observation_noise_m=0 returns the true miss unchanged (bit-for-bit)."""
        if _rng is None:
            return true_miss
        return true_miss + _rng.normal(0.0, obs_noise, size=2)

    # Height-aware impact in the range/cross frame (None if unreachable).
    def imp(met, elev):
        r = integrate_trajectory(v0=V0, elevation_deg=elev,
                                 azimuth_deg=target_bearing, met=met,
                                 use_g7=True, target_height_m=target_height_m)
        if r.get("reached") is False:
            return None
        return _world_to_rc(np.array([r["impact_x"], r["impact_y"]]), target_bearing)

    # Fire-control elevation (height-aware) putting the told-MET impact on target.
    def solve_elev(lo=15.0, hi=48.0):
        def dr(e):
            p = imp(told_met, e)
            return -1e9 if p is None else p[0] - target_range
        if dr(lo) > 0:
            return lo
        if dr(hi) < 0:
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

    elevation = solve_elev()

    # Wind sensitivity H (metres per m/s) at the told MET, height-aware.
    base = imp(told_met, elevation)
    h_dr = imp(_met_plus_uniform_wind(told_met, _rc_to_world([1.0, 0.0], target_bearing)),
               elevation) - base
    h_cr = imp(_met_plus_uniform_wind(told_met, _rc_to_world([0.0, 1.0], target_bearing)),
               elevation) - base
    H = np.column_stack([h_dr, h_cr])

    # Warm-start the gun bias (Estimator A) from the DB if the weapon is known.
    warm = None
    if db is not None:
        _ensure_phase2_columns(db)
        warm = _load_gun_bias(db, weapon_id)
    warm_started = warm is not None
    # Raise the filters' measurement-noise term to reflect the observation noise
    # (obs_var=0 leaves the established R values untouched -> bit-for-bit).
    estA = GunBiasEstimator(init=tuple(warm) if warm_started else (0.0, 0.0),
                            init_P=(100.0 if warm_started else 1.0e4),
                            R=400.0 + obs_var)
    estB = AtmosphericStateEstimator([0.0, 0.0], np.diag([25.0, 25.0]),
                                     np.diag([0.05, 0.05]),
                                     np.diag([2500.0 + obs_var, 2500.0 + obs_var]))

    obs_true = imp(true_met, elevation)        # constant realized impact (hidden)
    if obs_true is None:
        raise ValueError("target altitude unreachable under true conditions")

    history = []
    round_no = 0
    destroyed = False
    destroying_round = None
    final_miss = None

    def log(phase, miss, destroy):
        nonlocal round_no
        round_no += 1
        history.append({"round": round_no, "phase": phase,
                        "miss": [float(miss[0]), float(miss[1])],
                        "radial": float(np.hypot(*miss)),
                        "destroyed_target": bool(destroy)})
        return round_no

    # --- COLD start only: brief registration to learn + persist the gun bias.
    # Known-good MET isolates the gun, so the miss is purely the residual gun
    # error. (Calibration under an idealised MET, so these rounds do not declare
    # a kill; the real assessment is the operation rounds below.)
    if not warm_started:
        for _ in range(max(2, min_learning_rounds)):
            if round_no >= max_rounds:
                break
            miss = true_gun - estA.state()
            log("REGISTRATION", miss, False)
            estA.update(observe(miss) + estA.state())   # learn from noisy report
            if float(np.hypot(*miss)) < 10.0:
                break

    # --- OPERATION: fire at the target under the imperfect told MET, assess,
    # learn the atmospheric residual, and STOP the instant a round is lethal.
    while round_no < max_rounds and not destroyed:
        corrected_met = _met_plus_uniform_wind(
            told_met, _rc_to_world(estB.state(), target_bearing))
        pred = imp(corrected_met, elevation)
        # TRUE accuracy miss vs the target (the aim correction cancels out): as
        # the estimators converge this collapses toward zero. The kill is judged
        # on this real fall of shot, NOT on the noisy observation.
        miss = (obs_true + true_gun) - (pred + estA.state())
        radial = float(np.hypot(*miss))

        this_round = round_no + 1
        if this_round >= min_learning_rounds and radial <= lethal_radius_m:
            log("FIRE_FOR_EFFECT", miss, True)        # target destroyed -> stop
            destroyed = True
            destroying_round = this_round
            final_miss = radial
            break

        log("ADJUSTMENT", miss, False)
        estB.predict()
        estB.update(observe(miss), H)                 # learn from noisy report

    if final_miss is None and history:
        final_miss = history[-1]["radial"]

    # Persist the learned gun bias so the next mission on this weapon warm-starts.
    if db is not None:
        prof = db.get_weapon_profile(weapon_id)
        n_eng = (prof["n_engagements"] if prof else 0) + 1
        total = (prof["total_rounds_fired"] if prof else 0) + round_no
        _save_gun_bias(db, weapon_id, weapon_type, estA.state(), n_eng, total)

    return {
        "weapon_id": weapon_id,
        "destroyed": destroyed,
        "rounds_fired": round_no,
        "destroying_round": destroying_round,
        "final_miss": final_miss,
        "lethal_radius_m": float(lethal_radius_m),
        "max_rounds": max_rounds,
        "min_learning_rounds": min_learning_rounds,
        "warm_started": warm_started,
        "gun_bias_est": estA.state(),
        "atmo_correction_est": estB.state(),
        "elevation_deg": elevation,
        "history": history,
    }
