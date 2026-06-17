"""
ARCS — Rotation Module
Handles all orientation math for the robot.

Two bodies:
    1. CHASSIS  — the base with caterpillar treads
                  Yaw   : heading direction on ground
                  Pitch : forward/backward tilt (terrain slope)
                  Roll  : sideways tilt (terrain slope)

    2. TURRET   — the arm mounted on the chassis
                  Yaw   : 360° spin relative to chassis
                  Pitch : barrel elevation 0°–85°
                  Roll  : always 0 (no roll actuator on arm)

Full rotation stack (world frame):
    R_world = R_chassis(yaw,pitch,roll) · R_turret(yaw,pitch)
    barrel_direction = R_world · [1, 0, 0]ᵀ

Why order matters:
    Think of it like a robotic arm. First the base rotates (chassis),
    THEN the arm rotates on top of it (turret). The turret's angles
    are always relative to whatever the chassis is doing.
    Getting this order wrong makes the barrel point at the wrong place.
"""

import numpy as np
from .constants import (YAW_MIN_DEG, YAW_MAX_DEG,
                        PITCH_MIN_DEG, PITCH_MAX_DEG,
                        ROLL_MIN_DEG, ROLL_MAX_DEG)


class RotationMatrix:
    """
    Pure static rotation matrix builders + combined robot kinematics.
    All input angles in DEGREES. All matrices are 3×3 numpy arrays.
    """

    # ─── ELEMENTARY ROTATION MATRICES ────────────────────────────

    @staticmethod
    def yaw(angle_deg: float) -> np.ndarray:
        """
        Rotation around the Y-axis (vertical).
        Positive angle = rotate LEFT (counter-clockwise from above).

        Physical meaning: turret spinning left/right,
        or chassis turning on the ground.

              Y
              │   ↺ positive yaw
              │
        ──────┼──────  X (forward)
              │
              Z

        Matrix:
            | cos ψ   0   sin ψ |
            |   0     1     0   |
            |-sin ψ   0   cos ψ |
        """
        psi = np.deg2rad(angle_deg)
        c, s = np.cos(psi), np.sin(psi)
        return np.array([
            [ c,  0,  s],
            [ 0,  1,  0],
            [-s,  0,  c]
        ])

    @staticmethod
    def pitch(angle_deg: float) -> np.ndarray:
        """
        Rotation around the Z-axis.
        Positive angle = barrel tilts UP.

        Physical meaning: barrel elevation.
        0° = horizontal, 90° = straight up.

        Matrix:
            | cos θ  -sin θ   0 |
            | sin θ   cos θ   0 |
            |   0       0     1 |
        """
        theta = np.deg2rad(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [ c, -s,  0],
            [ s,  c,  0],
            [ 0,  0,  1]
        ])

    @staticmethod
    def roll(angle_deg: float) -> np.ndarray:
        """
        Rotation around the X-axis (forward axis).
        Positive angle = right side of chassis tilts DOWN.

        Physical meaning: chassis tilting sideways on uneven terrain.
        This rotates the barrel sideways even if turret commands roll=0.
        That's the roll compensation problem.

        Matrix:
            | 1     0       0   |
            | 0   cos φ  -sin φ |
            | 0   sin φ   cos φ |
        """
        phi = np.deg2rad(angle_deg)
        c, s = np.cos(phi), np.sin(phi)
        return np.array([
            [1,  0,  0],
            [0,  c, -s],
            [0,  s,  c]
        ])

    # ─── COMBINED ROTATION MATRICES ──────────────────────────────

    @staticmethod
    def chassis(yaw_deg: float,
                pitch_deg: float = 0.0,
                roll_deg: float  = 0.0) -> np.ndarray:
        """
        Full chassis orientation in world frame.
        Order: Yaw first, then Pitch, then Roll (ZYX intrinsic).

        In Phase 1: pitch_deg=0, roll_deg=0 (flat terrain).
        In Phase 2: these come from the IMU sensor on the chassis.

        Returns 3×3 rotation matrix.
        """
        Ry = RotationMatrix.yaw(yaw_deg)
        Rp = RotationMatrix.pitch(pitch_deg)
        Rr = RotationMatrix.roll(roll_deg)
        return Ry @ Rp @ Rr

    @staticmethod
    def turret(yaw_deg: float, pitch_deg: float) -> np.ndarray:
        """
        Turret rotation RELATIVE to chassis.
        Yaw = spin left/right, Pitch = barrel up/down.

        Returns 3×3 rotation matrix.
        """
        Ry = RotationMatrix.yaw(yaw_deg)
        Rp = RotationMatrix.pitch(pitch_deg)
        return Ry @ Rp

    @staticmethod
    def full_stack(chassis_yaw:   float,
                   chassis_pitch: float,
                   chassis_roll:  float,
                   turret_yaw:    float,
                   turret_pitch:  float) -> np.ndarray:
        """
        Complete robot rotation matrix.
        Combines chassis orientation AND turret orientation.

        R_world = R_chassis · R_turret

        This is the matrix that tells us where the barrel
        actually points in the real world.
        """
        R_chassis = RotationMatrix.chassis(chassis_yaw,
                                           chassis_pitch,
                                           chassis_roll)
        R_turret  = RotationMatrix.turret(turret_yaw,
                                          turret_pitch)
        return R_chassis @ R_turret

    # ─── BARREL DIRECTION ────────────────────────────────────────

    @staticmethod
    def barrel_direction(chassis_yaw:   float,
                         chassis_pitch: float,
                         chassis_roll:  float,
                         turret_yaw:    float,
                         turret_pitch:  float) -> np.ndarray:
        """
        Returns a unit vector pointing in the direction the barrel faces.

        Barrel starts pointing along +X axis: [1, 0, 0]
        After all rotations it points somewhere in 3D space.

        This vector IS the initial direction of the projectile.
        Multiply by muzzle velocity v0 to get the velocity vector.

        Returns: np.array([vx, vy, vz]) — unit vector
        """
        R = RotationMatrix.full_stack(chassis_yaw,
                                      chassis_pitch,
                                      chassis_roll,
                                      turret_yaw,
                                      turret_pitch)
        barrel = R @ np.array([1.0, 0.0, 0.0])
        # Normalise (should already be unit but floating point safety)
        return barrel / np.linalg.norm(barrel)

    # ─── EULER ANGLES FROM ROTATION MATRIX ───────────────────────

    @staticmethod
    def matrix_to_euler_deg(R: np.ndarray) -> dict:
        """
        Extract yaw, pitch, roll (degrees) from a rotation matrix.
        Used for debugging and display.
        """
        pitch = np.rad2deg(np.arcsin(-R[2, 0]))
        roll  = np.rad2deg(np.arctan2(R[2, 1], R[2, 2]))
        yaw   = np.rad2deg(np.arctan2(R[1, 0], R[0, 0]))
        return {"yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll}

    # ─── ROLL COMPENSATION ───────────────────────────────────────

    @staticmethod
    def compute_roll_compensation(chassis_roll_deg: float,
                                  turret_pitch_deg: float) -> float:
        """
        When the chassis is tilted (roll), the barrel drifts sideways
        even though the turret didn't command any lateral movement.

        This function computes the yaw correction to apply to the turret
        to counteract the chassis roll effect.

        Derivation:
            A rolled chassis rotates the elevation axis.
            The lateral drift ≈ sin(roll) × sin(pitch)
            Correction yaw ≈ arctan(sin(roll) × sin(pitch) / cos(pitch))

        Returns: yaw_correction_deg to add to turret yaw command
        """
        roll_r  = np.deg2rad(chassis_roll_deg)
        pitch_r = np.deg2rad(turret_pitch_deg)
        correction = np.arctan2(
            np.sin(roll_r) * np.sin(pitch_r),
            np.cos(pitch_r)
        )
        return np.rad2deg(correction)


# ─── SELF TEST ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 52)
    print("ARCS Phase 1 — rotation.py self-test")
    print("=" * 52)

    tol = 1e-6

    # Test 1: Identity — no rotation → barrel points forward
    bd = RotationMatrix.barrel_direction(0, 0, 0, 0, 0)
    assert np.allclose(bd, [1,0,0], atol=tol), f"Identity failed: {bd}"
    print("  Test 1 — Identity (no rotation)         : ✓")

    # Test 2: turret yaw +90° → barrel points in -Z (left), yaw -90° → +Z (right)
    bd = RotationMatrix.barrel_direction(0, 0, 0, 90, 0)
    assert np.allclose(bd, [0,0,-1], atol=tol), f"Yaw +90 failed: {bd}"
    bd2 = RotationMatrix.barrel_direction(0, 0, 0, -90, 0)
    assert np.allclose(bd2, [0,0,1], atol=tol), f"Yaw -90 failed: {bd2}"
    print("  Test 2 — Turret yaw ±90° (barrel left/right): ✓")

    # Test 3: 90° turret pitch → barrel points straight up (+Y)
    bd = RotationMatrix.barrel_direction(0, 0, 0, 0, 90)
    assert np.allclose(bd, [0,1,0], atol=tol), f"Pitch 90 failed: {bd}"
    print("  Test 3 — Turret pitch 90° (barrel up)   : ✓")

    # Test 4: 45° pitch → barrel at 45° between X and Y
    bd = RotationMatrix.barrel_direction(0, 0, 0, 0, 45)
    expected = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0])
    assert np.allclose(bd, expected, atol=tol), f"Pitch 45 failed: {bd}"
    print("  Test 4 — Turret pitch 45° (diagonal)    : ✓")

    # Test 5: chassis yaw 90° + no turret → barrel points in -Z
    bd = RotationMatrix.barrel_direction(90, 0, 0, 0, 0)
    assert np.allclose(bd, [0,0,-1], atol=tol), f"Chassis yaw 90 failed: {bd}"
    print("  Test 5 — Chassis yaw 90°                : ✓")

    # Test 6: roll compensation non-zero when roll and pitch both active
    comp = RotationMatrix.compute_roll_compensation(10, 45)
    assert abs(comp) > 0, "Roll compensation should be non-zero"
    print(f"  Test 6 — Roll compensation(10°,45°)     : ✓  ({comp:.4f}°)")

    # Test 7: unit vector check
    for angles in [(30,0,0,45,30),(0,5,10,120,60),(-45,0,15,0,20)]:
        bd = RotationMatrix.barrel_direction(*angles)
        assert abs(np.linalg.norm(bd) - 1.0) < tol, f"Not unit vector: {bd}"
    print("  Test 7 — All outputs are unit vectors   : ✓")

    # Display matrix for a real firing scenario
    print()
    print("  Sample: chassis(0,0,0) + turret(yaw=30°, pitch=25°)")
    R = RotationMatrix.full_stack(0, 0, 0, 30, 25)
    bd = RotationMatrix.barrel_direction(0, 0, 0, 30, 25)
    euler = RotationMatrix.matrix_to_euler_deg(R)
    print(f"    Barrel direction : {bd}")
    print(f"    Extracted euler  : {euler}")
    print()
    print("  rotation.py ✓ all checks passed")