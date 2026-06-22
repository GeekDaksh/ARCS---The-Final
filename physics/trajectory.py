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

# Component 11 — Coriolis effect from the Earth's rotation. The ground turns
# under the shell during its (up to ~100 s) flight, so a long-range round lands a
# deterministic, predictable offset from where it would on a non-rotating Earth.
# Standard treatment (McCoy, "Modern Exterior Ballistics", artillery Coriolis):
# the Coriolis acceleration is a = -2 * Omega x v, with Omega the Earth's angular
# velocity vector. Earth's rotation rate:
OMEGA_EARTH = 7.292e-5  # rad/s
#
# FRAME CONVENTION (documented and consistent with Component 10): the integrator
# works in a right-handed local ENU frame -- x = East, y = North, z = Up -- so
# x_hat x y_hat = z_hat. The firing azimuth_deg is the angle of the shot in this
# frame (the velocity is fired along (cos az, sin az) in (East, North)). At
# latitude phi the Earth's angular velocity has NO east component, a north
# component Omega*cos(phi), and an up component Omega*sin(phi):
#     Omega_ENU = (0, Omega*cos(phi), Omega*sin(phi))
# With a = -2 Omega x v this reproduces the textbook result that, in the Northern
# hemisphere, a projectile is deflected to the RIGHT of its line of fire; the
# Southern hemisphere (phi < 0) flips it. The effect is a known correctable
# shift, NOT a random error. Like spin drift it is treated as absent in vacuum
# mode (the idealized non-rotating analytical reference used for validation).

# NOTE: Cd = 0.30 is the legacy PLACEHOLDER constant drag coefficient from
# Component 2. The realistic default is now the Mach-dependent G7 model
# (Component 5, use_g7=True); the constant is retained only for the
# use_g7=False path that reproduces the Component 2/3/4 validation results.
CD_PLACEHOLDER = 0.30

# Component 10 — gyroscopic spin drift (yaw of repose), the defining feature of
# the STANAG 4355 Modified Point Mass model. A spin-stabilised shell sits at a
# tiny steady "yaw of repose" angle to its flight path; the resulting side force
# drifts it consistently to one side (to the shooter's right for the standard
# right-hand-twist barrel). Reference: McCoy, "Modern Exterior Ballistics",
# ch. 9-12 (yaw of repose) and STANAG 4355 (Modified Point Mass).
#
# Form used (McCoy's practical yaw-of-repose scaling): the lateral acceleration
# is proportional to the spin rate divided by the airspeed, times the component
# of gravity perpendicular to the velocity (the rate at which gravity turns the
# velocity vector — the driver of the yaw of repose):
#     a_lat = sign * K_SD * (p / V) * g * cos(gamma)
# where p is the spin rate, V the speed, and cos(gamma) = V_horizontal / V (so
# the term vanishes for a purely vertical shot and peaks near apex). K_SD is a
# single dimensionless coefficient that lumps the MPM aerodynamic/inertia
# constants (C_Lalpha, C_Malpha, axial inertia); it is calibrated ONCE to the
# published 155mm M107 spin-drift magnitude (order tens of metres at ~20-30 km),
# not tuned per scenario. Spin rate and direction come from the barrel twist and
# muzzle velocity, so the drift scales with the physics rather than a fixed
# number. Spin decay over the flight is slow for a stable shell and is neglected
# (p held at its muzzle value), a standard MPM simplification.
SPIN_DRIFT_COEFF = 0.0005


def _acceleration(state, mass, area, Cd, vacuum, wind, met, use_g7, bc_scale,
                  spin_p, spin_dir, ux, uy, cor_on, cor_omega):
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
            #
            # Component 12: if the MET is a horizontally-varying field it also
            # depends on the shell's current DOWNRANGE distance, so it is sampled
            # by both downrange and altitude. The plain MetMessage path (else)
            # is untouched, so the altitude-only behaviour is bit-for-bit
            # preserved; a field whose gun and target profiles are equal also
            # reproduces it exactly (the field interpolates a + f*(b - a)).
            if getattr(met, "is_horizontal", False):
                downrange = float(np.hypot(state[0], state[1]))
                zone = met.sample_at(downrange, z_lookup)
            else:
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

        # Component 10: gyroscopic spin drift (yaw of repose). Aerodynamic, so it
        # only acts when there is air (inside this `not vacuum` block) and only
        # when enabled (spin_p > 0). The lateral acceleration is applied in the
        # horizontal direction PERPENDICULAR to the firing line, perp = (uy,-ux),
        # which is the shooter's right for a right-hand twist. cos_gamma is built
        # from the ALONG-firing horizontal speed (v . firing_unit), not the total
        # horizontal speed, so the (small) cross drift it produces does not feed
        # back into its own magnitude — a purely vertical shot has v_along = 0
        # and therefore zero drift. With spin_p == 0 the term is absent and the
        # trajectory is bit-for-bit the pre-spin-drift result.
        if spin_p > 0.0:
            speed = np.linalg.norm(vel)
            if speed > 0.0:
                v_along = vel[0] * ux + vel[1] * uy        # horizontal speed along firing
                cos_gamma = abs(v_along) / speed
                a_mag = spin_dir * SPIN_DRIFT_COEFF * (spin_p / speed) * G0 * cos_gamma
                acc[0] += a_mag * uy
                acc[1] += a_mag * (-ux)

        # Component 11: Coriolis acceleration from the Earth's rotation,
        # a = -2 * Omega x v (McCoy). cor_omega is the Earth's angular velocity
        # expressed in the local ENU frame (set from the latitude). With cor_on
        # False the term is absent and the trajectory is bit-for-bit unchanged.
        # It is grouped with the aerodynamic terms here so vacuum mode (the
        # analytical reference) excludes it too.
        if cor_on:
            acc = acc - 2.0 * np.cross(cor_omega, vel)

    return np.concatenate(([vel[0], vel[1], vel[2]], acc))


def _rk4_step(state, dt, mass, area, Cd, vacuum, wind, met, use_g7, bc_scale,
              spin_p, spin_dir, ux, uy, cor_on, cor_omega):
    """One classic RK4 step. Returns the new length-6 state."""
    args = (mass, area, Cd, vacuum, wind, met, use_g7, bc_scale,
            spin_p, spin_dir, ux, uy, cor_on, cor_omega)
    k1 = _acceleration(state, *args)
    k2 = _acceleration(state + 0.5 * dt * k1, *args)
    k3 = _acceleration(state + 0.5 * dt * k2, *args)
    k4 = _acceleration(state + dt * k3, *args)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_trajectory(v0=827.0, elevation_deg=45.0, azimuth_deg=0.0,
                         mass=43.2, area=0.018869, Cd=CD_PLACEHOLDER, dt=0.01,
                         vacuum=False, wind=(0.0, 0.0, 0.0), met=None,
                         use_g7=True, bc_scale=1.0, target_height_m=0.0,
                         spin_drift=True, twist_rate_cal=20.0, spin_right=True,
                         coriolis=True, latitude_deg=20.0):
    """
    Flies a point-mass shell through the ISA atmosphere until it reaches the
    target's altitude.

    coriolis: include the Coriolis deflection from the Earth's rotation, a
        -2*Omega x v acceleration (McCoy). Default True (the realistic model).
        It is a DETERMINISTIC, correctable shift (not a random error). False
        reproduces the prior non-rotating-Earth result bit-for-bit; it is also
        absent in vacuum mode (the idealized analytical reference). See
        OMEGA_EARTH for the formulation and ENU frame convention.
    latitude_deg: gun latitude in degrees. Positive = Northern hemisphere
        (deflects right), negative = Southern (deflects left). Default 20.0
        (a mid-latitude, India-relevant choice). With the firing azimuth (the
        existing azimuth_deg parameter) this sets the Coriolis direction and
        magnitude; the effect is strongest toward the poles and the cross-range
        part vanishes at the equator where sin(latitude) = 0.

    spin_drift: include gyroscopic spin drift (yaw of repose), the defining
        feature of the STANAG 4355 Modified Point Mass model. Default True (the
        realistic model). False removes the term entirely and reproduces the
        prior point-mass result BIT-FOR-BIT (used where exact reproduction is
        asserted). The effect is aerodynamic, so it is also absent when
        vacuum=True. See SPIN_DRIFT_COEFF for the formulation and citation.
    twist_rate_cal: barrel rifling twist in calibers per turn. The 155mm
        standard is ~20 cal/turn (documented default); this and v0 set the
        muzzle spin rate p = 2*pi*v0 / (twist_rate_cal * d), so the drift scales
        with the physics, not a fixed number.
    spin_right: True = right-hand twist -> drifts to the shooter's RIGHT (the
        standard for 155mm), which is -y in this x-forward, z-up right-handed
        frame. False = left-hand twist -> drifts left (symmetric).
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

    # Spin drift setup. The muzzle spin rate comes from the barrel twist and v0;
    # the direction comes from the twist hand and is applied perpendicular to the
    # firing line (ux, uy) = (cos phi, sin phi). Disabled (spin_p = 0) when
    # spin_drift is off or in vacuum, which leaves the integration bit-for-bit
    # unchanged.
    ux, uy = np.cos(phi), np.sin(phi)
    if spin_drift and not vacuum:
        diameter = 2.0 * np.sqrt(area / np.pi)          # 0.155 m for the M107 area
        spin_p = 2.0 * np.pi * v0 / (twist_rate_cal * diameter)   # rad/s
        spin_dir = 1.0 if spin_right else -1.0           # right-hand twist -> right
    else:
        spin_p = 0.0
        spin_dir = 0.0

    # Coriolis setup: Earth's angular velocity in the local ENU frame
    # (East, North, Up) at the gun latitude. Disabled (cor_on False) when
    # coriolis is off or in vacuum, leaving the integration bit-for-bit unchanged.
    cor_on = bool(coriolis and not vacuum)
    lat = np.radians(latitude_deg)
    cor_omega = np.array([0.0, OMEGA_EARTH * np.cos(lat), OMEGA_EARTH * np.sin(lat)])

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
                              use_g7, bc_scale, spin_p, spin_dir, ux, uy,
                              cor_on, cor_omega)
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
