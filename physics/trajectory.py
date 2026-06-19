"""
Point-mass trajectory integrator (Component 2) — no wind.

Flies a shell from the muzzle through the ISA atmosphere (Component 1) under
gravity and air drag, stepping forward with classic 4th-order Runge-Kutta at a
fixed time step until the shell returns to ground. Returns the impact point,
time of flight, and apex height.

This is the first component that consumes Component 1: at every step the air
density that drives drag comes from physics.atmosphere.atmosphere().

Coordinate frame:
    x = downrange horizontal
    y = lateral (cross-range)
    z = vertical (up)
Launch from origin (0, 0, 0). Elevation theta from horizontal, azimuth phi
from the x-axis.

Deliberately NOT here (later components): wind, MET, spin/yaw-of-repose drift,
Mach-dependent drag, Coriolis. Standard library + numpy only.
"""

import numpy as np

from physics.atmosphere import atmosphere, MAX_ALT
from physics.drag import drag_coefficient

G0 = 9.80665  # m/s^2, gravitational acceleration (matches ISA datum)

# NOTE: Cd = 0.30 is the legacy PLACEHOLDER constant drag coefficient from
# Component 2. The realistic default is now the Mach-dependent G7 model
# (Component 5, use_g7=True); the constant is retained only for the
# use_g7=False path that reproduces the Component 2/3/4 validation results.
CD_PLACEHOLDER = 0.30


def _acceleration(state, mass, area, Cd, vacuum, wind, met, use_g7, bc_scale):
    """
    Acceleration vector a = (F_gravity + F_drag) / mass at the given state.

    state: length-6 array [x, y, z, vx, vy, vz].
    wind: constant air-velocity vector (length-3 array). Drag opposes the
    air-relative velocity: F_d = -0.5 * rho * |v_rel| * v_rel * Cd_eff * A,
    where v_rel = v - wind.
    rho is the ISA/MET air density at the shell's current altitude z (when not
    vacuum). Below z=0 the atmosphere model is out of range, but integration
    stops at ground so that is never evaluated for a valid impact.

    Cd_eff: when use_g7 is True it is computed per-step from the G7 standard
    drag model against the AIRSPEED and the LOCAL temperature (Mach number =
    speed_rel / speed_of_sound(local_temp)); when False the constant Cd is used,
    which reproduces the Component 2/3/4 path exactly.
    """
    z = state[2]
    vel = state[3:6]

    # Gravity (always present).
    acc = np.array([0.0, 0.0, -G0])

    if not vacuum:
        # Drag acts on the shell's velocity RELATIVE TO THE AIR, not its ground
        # velocity. With a moving air mass, v_rel = v - v_wind. A tail wind
        # (+x) reduces airspeed and shifts the drag-loss balance so range
        # increases; a head wind (-x) does the opposite; a crosswind (+/-y)
        # produces lateral drag that deflects the impact sideways. Gravity is
        # untouched. (FM 6-40.) With wind=(0,0,0), v_rel == v and this reduces
        # exactly to the Component 2 no-wind drag term.
        # RK4 sub-steps can probe a hair below ground (z slightly < 0) near
        # impact; clamp to the sea-level datum for both lookups. Integration
        # still stops at the true z=0 crossing, so this does not affect the
        # reported impact point.
        z_lookup = max(z, 0.0)
        if met is not None:
            # Component 4: wind, density AND local temperature come from the MET
            # zone at this altitude, overriding the constant wind argument and
            # the plain atmosphere() lookup. This delivers altitude-varying
            # conditions into the exact same relative-airspeed drag mechanism.
            zone = met.sample(z_lookup)
            wind_here = zone["wind_vector"]
            rho = zone["density_kgm3"]
            local_temp_K = zone["temp_C"] + 273.15
        else:
            wind_here = wind
            air = atmosphere(z_lookup)
            rho = air["density_kgm3"]
            local_temp_K = air["temperature_K"]

        v_rel = vel - wind_here
        speed_rel = np.linalg.norm(v_rel)
        if speed_rel > 0.0:
            # Component 5: Mach-dependent G7 drag, evaluated against airspeed
            # and the LOCAL temperature (speed of sound falls with altitude).
            if use_g7:
                cd_eff = drag_coefficient(speed_rel, local_temp_K, bc_scale)
            else:
                cd_eff = Cd
            drag_force = -0.5 * rho * speed_rel * cd_eff * area * v_rel
            acc = acc + drag_force / mass

    return np.concatenate(([vel[0], vel[1], vel[2]], acc))


def _rk4_step(state, dt, mass, area, Cd, vacuum, wind, met, use_g7, bc_scale):
    """One classic RK4 step. Returns the new length-6 state."""
    args = (mass, area, Cd, vacuum, wind, met, use_g7, bc_scale)
    k1 = _acceleration(state, *args)
    k2 = _acceleration(state + 0.5 * dt * k1, *args)
    k3 = _acceleration(state + 0.5 * dt * k2, *args)
    k4 = _acceleration(state + dt * k3, *args)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_trajectory(v0=827.0, elevation_deg=45.0, azimuth_deg=0.0,
                         mass=43.2, area=0.018869, Cd=CD_PLACEHOLDER, dt=0.01,
                         vacuum=False, wind=(0.0, 0.0, 0.0), met=None,
                         use_g7=True, bc_scale=1.0, target_height_m=0.0):
    """
    Flies a point-mass shell through the ISA atmosphere until it reaches the
    target's altitude.
    vacuum=True disables drag (for validation against analytical formulas).
    Calls physics.atmosphere.atmosphere() each step for air density (when not vacuum).

    target_height_m: altitude of the target relative to the gun (metres).
        0.0 (default) = same level as the gun, reproduces the previous
        same-level behaviour bit-for-bit. Positive = target above the gun (a
        ridge); negative = target below the gun (a valley). Impact is detected
        when the shell crosses this altitude on its DESCENDING branch only, so a
        target above the gun is not falsely triggered while the shell climbs
        through that altitude on the way up.

        The frame is GUN-RELATIVE (gun at z = 0). For the air-density lookup the
        gun is treated as sea level and any sub-gun altitude (z < 0, a valley)
        uses clamped sea-level density. There is NO artificial limit on the
        magnitude of the height difference — only genuine physical bounds:
          * |target_height_m| must stay within the atmosphere model's valid
            span (<= 20,000 m); outside that a clear ValueError is raised.
          * if the target altitude is above the shell's apex it is physically
            unreachable: the result dict has {"reached": False, ...} rather than
            a crash or a fabricated number.

    use_g7: when True (the realistic default for real shots) the drag
        coefficient is computed per step from the published G7 Mach-dependent
        drag model (physics.drag), using the airspeed and the LOCAL temperature
        from the atmosphere/MET lookup. When False, the constant Cd is used,
        which reproduces the Component 2/3/4 validation results bit-for-bit.
    bc_scale: ballistic-coefficient form-factor multiplier on the G7 Cd (single
        knob to match a specific real shell). Higher -> more drag -> shorter
        range. Only used when use_g7 is True.

    wind: constant air-velocity vector (wind_x, wind_y, wind_z) in m/s, same
          frame as the trajectory. Default (0,0,0) reproduces Component 2
          exactly. Drag is computed against v_rel = v - wind, so:
            * tail wind (+x) -> INCREASES range
            * head wind (-x) -> DECREASES range
            * crosswind (+/-y) -> deflects impact in +/-y
          Wind acts only through drag, so it has NO effect when vacuum=True
          (no air to move the shell). Component 4 (MET) will replace this
          single constant with a per-altitude wind lookup feeding this exact
          mechanism.

    met: optional MetMessage (Component 4). If given, the wind vector AND the
         air density come from met.sample(current_altitude) each step, over-
         riding both the constant wind argument and the plain atmosphere()
         density. If None, the Component 2/3 behaviour is reproduced exactly.

    Returns a dict:
      range_m        (horizontal distance to impact)
      impact_x, impact_y
      tof_s          (time of flight)
      apex_m         (maximum altitude)
      impact_speed   (speed at impact)
      steps          (number of integration steps)
    """
    wind = np.asarray(wind, dtype=float)

    # Genuine physical bound only (no artificial threshold): the impact altitude
    # must stay within the atmosphere model's validated span. Beyond +/-20 km the
    # density model is not defined, so refuse it loudly.
    if target_height_m > MAX_ALT or target_height_m < -MAX_ALT:
        raise ValueError(
            f"target_height_m must be within +/-{MAX_ALT:.0f} m of the gun "
            f"(atmosphere model's valid range), got {target_height_m}")

    theta = np.radians(elevation_deg)
    phi = np.radians(azimuth_deg)

    # Initial velocity components. Horizontal speed splits by azimuth.
    v_horiz = v0 * np.cos(theta)
    vx = v_horiz * np.cos(phi)
    vy = v_horiz * np.sin(phi)
    vz = v0 * np.sin(theta)

    state = np.array([0.0, 0.0, 0.0, vx, vy, vz])
    t = 0.0
    apex = 0.0
    steps = 0

    # Degenerate launch: no upward motion (theta <= 0) means the shell is
    # already at the ground heading level/down — it impacts immediately.
    if vz <= 0.0:
        speed = np.linalg.norm(state[3:6])
        return {
            "range_m": 0.0,
            "impact_x": 0.0,
            "impact_y": 0.0,
            "tof_s": 0.0,
            "apex_m": 0.0,
            "impact_speed": float(speed),
            "steps": 0,
        }

    while True:
        new_state = _rk4_step(state, dt, mass, area, Cd, vacuum, wind, met,
                              use_g7, bc_scale)
        steps += 1
        apex = max(apex, new_state[2])

        # Unreachable target: the shell is past apex (descending) and never
        # climbed to the target altitude. Report it clearly rather than crash or
        # fabricate a number.
        if new_state[5] < 0.0 and apex < target_height_m:
            return {
                "reached": False,
                "range_m": None,
                "impact_x": None,
                "impact_y": None,
                "tof_s": None,
                "apex_m": float(apex),
                "impact_speed": None,
                "steps": steps,
                "reason": (f"target altitude {target_height_m:.1f} m not reached "
                           f"(apex {apex:.1f} m)"),
            }

        # Impact when the shell crosses the target altitude on the DESCENDING
        # branch: z goes from >= target_height_m down to < target_height_m. The
        # "state[2] >= target_height_m" guard excludes the upward crossing while
        # climbing. With target_height_m = 0 this reduces exactly to the former
        # "new_state[2] < 0.0" ground test (state[2] >= 0 always holds there).
        if new_state[2] < target_height_m <= state[2]:
            # Linear interpolation on z to the exact target altitude.
            z0 = state[2]
            z1 = new_state[2]
            frac = (z0 - target_height_m) / (z0 - z1)  # in [0, 1]
            impact = state + frac * (new_state - state)
            t_impact = t + frac * dt
            speed = float(np.linalg.norm(impact[3:6]))
            return {
                "range_m": float(np.hypot(impact[0], impact[1])),
                "impact_x": float(impact[0]),
                "impact_y": float(impact[1]),
                "tof_s": float(t_impact),
                "apex_m": float(apex),
                "impact_speed": speed,
                "steps": steps,
            }

        state = new_state
        t += dt

        # Safety ceiling: a sane shot reaches its target altitude well within
        # this many steps.
        if steps > 10_000_000:
            raise RuntimeError("trajectory did not reach the target altitude")
