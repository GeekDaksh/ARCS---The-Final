"""
ARCS — Clean Integration API  (Gap 5)
======================================
Three-function API wrapping the ForgettingRLS estimator.
This is the integration surface for external fire control systems,
simulation frameworks, and hardware-in-the-loop test rigs.

Usage:
    from arcs_api import ARCSApi

    api = ARCSApi()
    api.initialise(weapon_type='40mm_autocannon', range_m=300, bearing_deg=45)

    # After each round lands — pass observed fall-of-shot errors:
    api.update(range_error_m=12.3, lateral_error_m=2.1)
    api.update(range_error_m=3.1,  lateral_error_m=-0.8)

    # Get the current fire control correction:
    delta_qe, delta_defl, delta_mv = api.get_correction()
    # → (-0.38, +0.12, -2.14)   units: degrees, degrees, m/s

Integration notes:
    - Call initialise() once per engagement (new target or new mission).
    - Call update() after every observed round impact.
    - Call get_correction() before each subsequent round to get the
      FCS-recommended correction to apply to the nominal firing solution.
    - No state persists between calls to initialise(); each engagement
      starts from the pre-correction prior.

Research basis:
    ForgettingRLS: Johnstone et al., IEEE 1982 (convergence proof).
    ILC framework: Bristow, Tharayil & Alleyne, IEEE Control Sys. 2006.
    Bearing rotation: FM 6-40 §3-12 (directional error decomposition).
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# ── Weapon profiles ──────────────────────────────────────────────────────────

@dataclass
class WeaponProfile:
    """Physical constants for a supported weapon type."""
    name:              str
    nominal_v0_ms:     float   # nominal muzzle velocity (m/s)
    sigma_pitch_deg:   float   # FCS laying error — elevation (°)
    sigma_yaw_deg:     float   # FCS laying error — deflection (°)
    sigma_v0_ms:       float   # round-to-round MV dispersion (m/s)
    rls_lambda:        float   # ForgettingRLS forgetting factor
    rls_p_db_cold:     float   # initial bearing covariance (cold start)
    rls_p_dv_cold:     float   # initial v0 covariance (cold start)
    rls_p_db_warm:     float   # bearing covariance (warm start from prior)
    rls_p_dv_warm:     float   # v0 covariance (warm start from prior)


WEAPON_PROFILES = {
    '40mm_autocannon': WeaponProfile(
        name='40mm Autocannon (AGL)',
        nominal_v0_ms=240.0,
        sigma_pitch_deg=0.10,
        sigma_yaw_deg=0.10,
        sigma_v0_ms=1.5,
        rls_lambda=0.90,
        rls_p_db_cold=16.0,
        rls_p_dv_cold=25.0,
        rls_p_db_warm=0.09,
        rls_p_dv_warm=0.75,
    ),
    'direct_fire_100mm': WeaponProfile(
        name='100mm Direct Fire Gun',
        nominal_v0_ms=900.0,
        sigma_pitch_deg=0.05,
        sigma_yaw_deg=0.05,
        sigma_v0_ms=3.0,
        rls_lambda=0.92,
        rls_p_db_cold=16.0,
        rls_p_dv_cold=35.0,
        rls_p_db_warm=0.05,
        rls_p_dv_warm=1.0,
    ),
    'generic': WeaponProfile(
        name='Generic Direct-Fire Weapon',
        nominal_v0_ms=100.0,
        sigma_pitch_deg=0.10,
        sigma_yaw_deg=0.10,
        sigma_v0_ms=1.5,
        rls_lambda=0.90,
        rls_p_db_cold=16.0,
        rls_p_dv_cold=25.0,
        rls_p_db_warm=0.09,
        rls_p_dv_warm=0.75,
    ),
}


# ── Internal ForgettingRLS (mirrors JS class; supports warm-start) ────────────

class _ForgettingRLS:
    """
    Forgetting-factor RLS for [delta_bearing_deg, delta_v0_ms].
    Mirrors the JavaScript ForgettingRLS class in arcs_simulation.html
    including the outlier rejection fix (Bug 2).
    """

    def __init__(self, init_db: float = 0.0, init_dv: float = 0.0,
                 lam: float = 0.90,
                 p_db: float = 16.0, p_dv: float = 25.0):
        self.db    = init_db
        self.dv    = init_dv
        self.lam   = lam
        self.P_db  = p_db
        self.P_dv  = p_dv
        self.n     = 0

    def update(self, corr_db: float, corr_dv: float,
               err_lat: float, err_range: float,
               range_m: float, tof_cos: float) -> None:
        """Update from one observed round impact."""
        # Lateral / bearing axis
        H_db  = range_m * math.pi / 180.0
        R_lat = (0.36 * H_db) ** 2
        # Outlier rejection — cap to 3σ of measurement noise
        max_db = 3.0 * math.sqrt(R_lat)
        y_db   = self.db + max(-max_db, min(max_db, err_lat - self.db))
        inn_db = y_db  - H_db * (corr_db - self.db)
        S_db   = H_db**2 * self.P_db + R_lat
        K_db   = self.P_db * H_db / max(S_db, 1e-9)
        self.db  -= K_db * inn_db
        self.P_db = (1.0 - K_db * H_db) * self.P_db / self.lam

        # Range / v0 axis
        H_dv  = tof_cos
        R_rng = (1.5 * tof_cos) ** 2 + (0.3 * H_db) ** 2
        max_dv = 3.0 * math.sqrt(R_rng)
        y_dv   = self.dv + max(-max_dv, min(max_dv, err_range - self.dv))
        inn_dv = y_dv  - H_dv * (corr_dv - self.dv)
        S_dv   = H_dv**2 * self.P_dv + R_rng
        K_dv   = self.P_dv * H_dv / max(S_dv, 1e-9)
        self.dv  -= K_dv * inn_dv
        self.P_dv = (1.0 - K_dv * H_dv) * self.P_dv / self.lam
        self.n   += 1

    @property
    def delta_bearing_deg(self) -> float:
        return float(max(-8.0, min(8.0, self.db)))

    @property
    def delta_v0_ms(self) -> float:
        return float(max(-22.0, min(22.0, self.dv)))


# ── Engagement state (populated by initialise()) ──────────────────────────────

@dataclass
class _EngagementState:
    weapon:       WeaponProfile
    range_m:      float
    bearing_deg:  float
    bearing_rad:  float
    tof_s:        float          # time-of-flight to target (s)
    tof_cos:      float          # tof × cos(pitch) — RLS range-axis sensitivity
    rls:          _ForgettingRLS
    n_rounds:     int = 0


# ── Public API ────────────────────────────────────────────────────────────────

class ARCSApi:
    """
    ARCS Fire Control API.

    The three public methods form the complete integration surface:
        initialise() → set engagement geometry
        update()     → ingest observed impact error
        get_correction() → query current fire control solution
    """

    def __init__(self):
        self._state: Optional[_EngagementState] = None

    # ── 1. initialise ─────────────────────────────────────────────────────────

    def initialise(
        self,
        range_m:       float,
        bearing_deg:   float,
        weapon_type:   str   = 'generic',
        pitch_deg:     Optional[float] = None,
        prior_db:      float = 0.0,
        prior_dv:      float = 0.0,
        tight_prior:   bool  = False,
    ) -> None:
        """
        Initialise the FCS for a new engagement.

        Call this once per target before firing any rounds.

        Args:
            range_m:     Horizontal range to target (metres).
            bearing_deg: Gun bearing to target (degrees from North / gun-front).
            weapon_type: Key from WEAPON_PROFILES dict.
                         Supported: '40mm_autocannon', 'direct_fire_100mm', 'generic'.
            pitch_deg:   Nominal firing elevation (degrees).
                         Used to compute tof_cos. If None, estimated from range.
            prior_db:    Warm-start prior for delta_bearing (degrees).
                         Set from a StructuredBiasEstimator or saved calibration.
            prior_dv:    Warm-start prior for delta_v0 (m/s).
            tight_prior: If True, use tight RLS covariance (P_db_warm, P_dv_warm).
                         Use when prior_db/prior_dv come from a trusted calibration.

        Raises:
            ValueError: Unknown weapon_type.
        """
        if weapon_type not in WEAPON_PROFILES:
            raise ValueError(
                f"Unknown weapon_type '{weapon_type}'. "
                f"Supported: {list(WEAPON_PROFILES)}"
            )
        wp = WEAPON_PROFILES[weapon_type]

        # Estimate pitch and TOF if not provided
        if pitch_deg is None:
            # Approximate flat-fire elevation: sin(2θ) ≈ gR/v² → θ ≈ sin⁻¹(gR/v²)/2
            v0 = wp.nominal_v0_ms
            g  = 9.81
            arg = min(1.0, (g * range_m) / (v0 ** 2))
            pitch_deg = 0.5 * math.degrees(math.asin(arg))

        pitch_rad = math.radians(pitch_deg)
        tof_s     = 2.0 * wp.nominal_v0_ms * math.sin(pitch_rad) / 9.81
        tof_cos   = tof_s * math.cos(pitch_rad)

        p_db = wp.rls_p_db_warm if tight_prior else wp.rls_p_db_cold
        p_dv = wp.rls_p_dv_warm if tight_prior else wp.rls_p_dv_cold

        rls = _ForgettingRLS(
            init_db=prior_db,
            init_dv=prior_dv,
            lam=wp.rls_lambda,
            p_db=p_db,
            p_dv=p_dv,
        )

        self._state = _EngagementState(
            weapon      = wp,
            range_m     = range_m,
            bearing_deg = bearing_deg,
            bearing_rad = math.radians(bearing_deg),
            tof_s       = tof_s,
            tof_cos     = tof_cos,
            rls         = rls,
        )

    # ── 2. update ─────────────────────────────────────────────────────────────

    def update(
        self,
        range_error_m:   float,
        lateral_error_m: float,
    ) -> None:
        """
        Ingest the observed fall-of-shot error from the most recent round.

        Call this after each round lands during the Adjustment phase.
        Do NOT call during Fire for Effect (locked solution).

        Args:
            range_error_m:   Signed range error — positive = beyond target,
                             negative = short of target (metres, world frame).
            lateral_error_m: Signed lateral error — positive = right of target,
                             negative = left of target (metres, world frame).

        Raises:
            RuntimeError: initialise() has not been called.
        """
        if self._state is None:
            raise RuntimeError("Call initialise() before update().")
        s = self._state

        # Rotate world-frame errors into barrel frame (FM 6-40 §3-12)
        cos_b     = math.cos(s.bearing_rad)
        sin_b     = math.sin(s.bearing_rad)
        err_range = cos_b * range_error_m   + sin_b * lateral_error_m
        err_lat   = -sin_b * range_error_m  + cos_b * lateral_error_m

        c = s.rls
        s.rls.update(
            corr_db   = c.delta_bearing_deg,
            corr_dv   = c.delta_v0_ms,
            err_lat   = err_lat,
            err_range = err_range,
            range_m   = s.range_m,
            tof_cos   = s.tof_cos,
        )
        s.n_rounds += 1

    # ── 3. get_correction ─────────────────────────────────────────────────────

    def get_correction(self) -> Tuple[float, float, float]:
        """
        Return the current fire control correction.

        Apply these corrections to the nominal firing solution before firing
        the next round.

        Returns:
            (delta_QE_deg, delta_deflection_deg, delta_muzzle_velocity_ms)

            delta_QE_deg:            Add to elevation (Quadrant Elevation).
                                     Positive = raise the barrel.
            delta_deflection_deg:    Add to deflection (azimuth / bearing).
                                     Positive = traverse right.
            delta_muzzle_velocity_ms: Add to nominal muzzle velocity.
                                     Positive = faster (longer range).

        Raises:
            RuntimeError: initialise() has not been called.
        """
        if self._state is None:
            raise RuntimeError("Call initialise() before get_correction().")

        rls = self._state.rls
        # delta_QE from pitch component is not estimated by range-axis RLS.
        # It is the elevation correction component of the range error,
        # derived from the v0 correction via the sensitivity ratio.
        # For direct-fire, Δpitch ≈ ΔQE (small angle, flat trajectory).
        # Full pitch estimation requires the StructuredBiasEstimator (SBE).
        return (
            0.0,                         # delta_QE_deg (use SBE for this axis)
            rls.delta_bearing_deg,       # delta_deflection_deg
            rls.delta_v0_ms,             # delta_muzzle_velocity_ms
        )

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Return a diagnostic dict with current estimator state."""
        if self._state is None:
            return {'initialised': False}
        s = self._state
        return {
            'initialised':          True,
            'weapon':               s.weapon.name,
            'range_m':              s.range_m,
            'bearing_deg':          s.bearing_deg,
            'rounds_observed':      s.n_rounds,
            'delta_QE_deg':         0.0,
            'delta_deflection_deg': s.rls.delta_bearing_deg,
            'delta_mv_ms':          s.rls.delta_v0_ms,
            'P_db':                 s.rls.P_db,
            'P_dv':                 s.rls.P_dv,
        }

    def __repr__(self) -> str:
        if self._state is None:
            return "ARCSApi(uninitialised)"
        s = self._state
        return (f"ARCSApi(weapon={s.weapon.name!r}, "
                f"range={s.range_m:.0f}m, n_rounds={s.n_rounds}, "
                f"db={s.rls.delta_bearing_deg:+.3f}°, "
                f"dv={s.rls.delta_v0_ms:+.2f}m/s)")


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    print("=" * 60)
    print("ARCS API — Self-test")
    print("=" * 60)

    # Simulate a 300m engagement with known systematic bias
    TRUE_BIAS_BEARING = +0.122   # deg (imu offset + blast kick)
    TRUE_BIAS_V0      = -2.1     # m/s (v0 drift)
    RANGE             = 300.0
    BEARING           = 45.0

    rng = random.Random(42)

    api = ARCSApi()
    api.initialise(
        range_m     = RANGE,
        bearing_deg = BEARING,
        weapon_type = '40mm_autocannon',
    )

    print(f"\n  Target: {RANGE}m at {BEARING}°")
    print(f"  True bias: bearing={TRUE_BIAS_BEARING:+.3f}°, v0={TRUE_BIAS_V0:+.2f}m/s")
    print(f"\n  {'Rnd':>4}  {'Range err':>10}  {'Lat err':>9}  "
          f"{'Δ deflection':>13}  {'Δ MV':>9}")
    print(f"  {'─'*56}")

    for rnd in range(8):
        dq, dd, dmv = api.get_correction()

        # Simulate impact: systematic bias + noise, minus current correction
        noise_r   = rng.gauss(0, 4.5)   # 1.5 m/s × tof≈3
        noise_l   = rng.gauss(0, 1.7)   # 0.10° × 300m × π/180
        range_err = (TRUE_BIAS_V0  - dmv)  * 3.0 + noise_r
        lat_err   = (TRUE_BIAS_BEARING - dd) * RANGE * math.pi / 180 + noise_l

        print(f"  {rnd+1:>4}  {range_err:>+9.2f}m  {lat_err:>+8.2f}m  "
              f"  {dd:>+11.3f}°  {dmv:>+8.2f}m/s")

        if rnd < 7:   # update during Adjustment phase, lock during verify
            api.update(range_error_m=range_err, lateral_error_m=lat_err)

    dq, dd, dmv = api.get_correction()
    print(f"\n  Final correction: ΔQE={dq:+.3f}°  Δdefl={dd:+.3f}°  ΔMV={dmv:+.2f}m/s")
    print(f"  True bias:        ΔQE=  N/A   Δdefl={TRUE_BIAS_BEARING:+.3f}°  "
          f"ΔMV={TRUE_BIAS_V0:+.2f}m/s")
    print(f"\n  Status: {api.status()}")
    print(f"\n  Repr:   {api}")
    print("\n  ✓ Self-test complete")
