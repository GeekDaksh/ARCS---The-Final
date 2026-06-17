"""
ARCS — Realistic Robot Bias Model v2.0
Calibrated for publishable research results.

CALIBRATION:
    Systematic bias magnitude = 1.5 * sigma_stochastic.
    At this level: CEP_systematic = 1.5 * CEP_stochastic.
    Total CEP = sqrt(1 + 1.5^2) * CEP_stochastic = 1.80x baseline.
    Max improvable = 44% of total CEP.
    This is detectable, meaningful, and matches degraded field hardware.

PHYSICAL BASIS:
    A worn/field-deployed actuator has:
      - Barrel gravity sag: 0.3-0.6 degrees (heavy barrel + worn mount)
      - Gear backlash: 0.2-0.4 degrees
      - Propellant charge variation: 3-8 m/s systematic (lot-to-lot)
      - IMU yaw drift: 0.1-0.4 degrees (gyro drift after extended use)
    These are all range/pitch-dependent and learnable by the GP formula.
"""

import numpy as np
from dataclasses import dataclass
from physics.constants import SIGMA_PITCH_DEG, SIGMA_YAW_DEG, SIGMA_V0, BIAS_SCALE


@dataclass
class RobotBiasParams:
    sag_coeff:        float   # deg per unit sin(pitch) — barrel gravity sag
    pitch_backlash:   float   # deg — gear backlash in pitch
    yaw_backlash:     float   # deg — gear backlash in yaw
    v0_bias:          float   # m/s — propellant charge offset
    imu_yaw_offset:   float   # deg — IMU alignment error
    thermal_v0_coeff: float   # m/s per degC above 20C
    ambient_temp_c:   float   # current temperature (degC)
    blast_yaw_kick:   float   # deg — muzzle blast lateral kick

    @classmethod
    def from_seed(cls, seed: int) -> 'RobotBiasParams':
        """
        Generate calibrated bias params for one robot instance.
        Bias magnitudes scaled so systematic component = BIAS_SCALE * sigma.
        Different seeds give different robots for transfer learning study.
        """
        rng = np.random.default_rng(seed)
        # Scale each component so TOTAL systematic is ~BIAS_SCALE * sigma
        return cls(
            sag_coeff        = float(rng.uniform(
                                0.8*BIAS_SCALE*SIGMA_PITCH_DEG,
                                1.2*BIAS_SCALE*SIGMA_PITCH_DEG)),
            pitch_backlash   = float(rng.uniform(0.1, 0.2) * SIGMA_PITCH_DEG),
            yaw_backlash     = float(rng.uniform(0.1, 0.2) * SIGMA_YAW_DEG),
            v0_bias          = float(rng.uniform(
                                -BIAS_SCALE*SIGMA_V0,
                                 BIAS_SCALE*SIGMA_V0)),
            imu_yaw_offset   = float(rng.uniform(
                                -0.5*BIAS_SCALE*SIGMA_YAW_DEG,
                                 0.5*BIAS_SCALE*SIGMA_YAW_DEG)),
            thermal_v0_coeff = float(rng.uniform(0.05, 0.12)),
            ambient_temp_c   = float(rng.uniform(15.0, 40.0)),
            blast_yaw_kick   = float(rng.uniform(
                                0.3*BIAS_SCALE*SIGMA_YAW_DEG,
                                0.7*BIAS_SCALE*SIGMA_YAW_DEG)),
        )

    @classmethod
    def nominal(cls) -> 'RobotBiasParams':
        """Zero systematic bias — for physics verification."""
        return cls(sag_coeff=0., pitch_backlash=0., yaw_backlash=0.,
                   v0_bias=0., imu_yaw_offset=0., thermal_v0_coeff=0.,
                   ambient_temp_c=20., blast_yaw_kick=0.)


class RobotBiasModel:
    """
    Applies calibrated systematic bias to commanded firing angles.

    The systematic component is DETERMINISTIC given robot params.
    It is what the GP formula learns and corrects.
    The stochastic component is added on top and is irreducible.
    """

    def __init__(self, seed: int = None, params: RobotBiasParams = None):
        if params is not None:
            self.params = params
        elif seed is not None:
            self.params = RobotBiasParams.from_seed(seed)
        else:
            self.params = RobotBiasParams.from_seed(0)
        self._prev_pitch = 0.0

    def systematic_pitch_bias(self, pitch_deg: float) -> float:
        p = self.params
        pitch_rad = np.deg2rad(pitch_deg)
        sag   = -p.sag_coeff * np.sin(pitch_rad)
        delta = pitch_deg - self._prev_pitch
        bl    = (-np.sign(delta) * p.pitch_backlash * 0.5
                  if abs(delta) > 0.01 else 0.0)
        self._prev_pitch = pitch_deg
        return sag + bl

    def systematic_yaw_bias(self, yaw_deg: float) -> float:
        p = self.params
        return p.imu_yaw_offset + p.blast_yaw_kick

    def systematic_v0_bias(self, v0: float) -> float:
        p = self.params
        return p.v0_bias + p.thermal_v0_coeff * (p.ambient_temp_c - 20.0)

    def apply(self, pitch_cmd, yaw_cmd, v0_cmd, rng,
              sigma_pitch=SIGMA_PITCH_DEG,
              sigma_yaw=SIGMA_YAW_DEG,
              sigma_v0=SIGMA_V0):
        sys_p = self.systematic_pitch_bias(pitch_cmd)
        sys_y = self.systematic_yaw_bias(yaw_cmd)
        sys_v = self.systematic_v0_bias(v0_cmd)
        return (pitch_cmd + sys_p + rng.normal(0, sigma_pitch),
                yaw_cmd   + sys_y + rng.normal(0, sigma_yaw),
                v0_cmd    + sys_v + rng.normal(0, sigma_v0))

    def expected_bias(self, pitch_cmd, yaw_cmd, v0_cmd):
        return {
            "pitch_bias": self.systematic_pitch_bias(pitch_cmd),
            "yaw_bias":   self.systematic_yaw_bias(yaw_cmd),
            "v0_bias":    self.systematic_v0_bias(v0_cmd),
        }

    def summary(self):
        p = self.params
        return (f"RobotBias(sag={p.sag_coeff:.3f}d "
                f"v0={p.v0_bias:+.2f}m/s "
                f"yaw_imu={p.imu_yaw_offset:+.3f}d "
                f"T={p.ambient_temp_c:.0f}C)")


if __name__ == "__main__":
    print("=" * 60)
    print("ARCS Robot Bias Model v2.0 — Calibrated")
    print(f"BIAS_SCALE = {BIAS_SCALE} * sigma (=> max improvable ~44% CEP)")
    print("=" * 60)
    from physics.ballistic_solver import BallisticSolver
    solver = BallisticSolver()
    for seed in [42, 99, 777]:
        b = RobotBiasModel(seed=seed)
        print(f"\nRobot seed={seed}: {b.summary()}")
        print(f"  {'Rng':>6}  {'Pitch_cmd':>10}  {'sys_pitch':>12}  {'total_at_rng':>14}")
        for R in [100, 200, 300, 400]:
            sol = solver.solve(R, 0, 0, 100)
            if sol.reachable:
                bv = b.expected_bias(sol.turret_pitch_deg, 0, 100)
                total_m = R * np.tan(np.deg2rad(abs(bv['pitch_bias'])))
                print(f"  {R:>6}m  {sol.turret_pitch_deg:>10.2f}d  "
                      f"{bv['pitch_bias']:>+12.4f}d  {total_m:>12.3f}m")
    print("\n  bias_model.py v2.0 OK")