# ARCS Phase 2 — Data Sources & Scope Boundary

This document states, in engineering terms, where every battlefield input to the
fire-control model comes from, how it enters the model, and — precisely — what is
in scope versus what is an adjacent sensing/effects problem the model consumes as
an interface rather than solves. It is the honest boundary: it defines
interfaces, it does not assume problems away.

The fire-control model proper is the frozen physics core (`physics/trajectory.py`
et al.) plus the two estimators (`physics/state_estimator.py`) wired by the
engagement loop (`physics/engagement.py`). Everything below is described relative
to that model.

---

## 1. Fall of shot (where the round landed) — IN SCOPE, modeled with noise

**Source.** A forward observer, drone, or counter-battery radar reports where the
round landed relative to the target — a miss in range and deflection. Nobody
measures the impact with a ruler; every such report carries measurement error
(range estimation error, optical/angular error, radar resolution, observer-to-
target geometry).

**How it enters the model.** The true fall of shot is real: the shell lands where
the frozen integrator + the hidden true conditions say it does. The model never
gets to see that true value directly — it sees an *observation* of it. Component
13 models this explicitly:

```
observed_miss = true_miss + N(0, observation_noise_m)   # per axis (downrange, cross)
```

The estimators (gun-bias RLS and atmospheric Kalman filter) learn from
`observed_miss`. The Kalman/RLS measurement-noise term `R` is raised to reflect
`observation_noise_m` — telling the filter how noisy its measurements are, which
is exactly what that term is for. `observation_noise_m` is a **parameter** (std
dev in metres): a precise drone/radar is small (~5 m), a competent optical/laser
observer at multi-km range is ~10–20 m, a degraded visual observer is large
(40 m+). `observation_noise_m = 0.0` reproduces perfect observation bit-for-bit.

**In scope because:** handling noisy measurements is the core purpose of the
Kalman filter. The model's job is to converge on the truth *despite* observation
error, and to do so provably (bounded, no divergence, floor widening gracefully
with noise). That robustness is a property of the fire-control model and is
validated here (`tests/test_observation_noise.py`).

---

## 2. Target location & altitude — INPUT from the sensing layer (out of scope)

**Source.** An observer, satellite, map with grid reference, or
target-acquisition system provides the target's location and altitude. These
carry their own geolocation uncertainty (sensor fusion, datum/registration error,
DTED resolution).

**Interface.** The fire-control model **consumes** target range, bearing, and
height as *given inputs* (`target_range`, `target_bearing`, `target_height_m`).
It computes the firing solution that puts the round on those coordinates.

**Out of scope because:** modeling geolocation/sensor-fusion uncertainty is a
sensing problem upstream of ballistics — it determines *where we are told the
target is*, not *how the shell flies*. The model's contract is: "given a target
location from the sensing layer, hit it." If the supplied coordinates are wrong,
the model faithfully hits the wrong coordinates; quantifying that input error
belongs to the target-acquisition system, not the ballistics/fire-control model.

---

## 3. Battle damage assessment (target destroyed) — INTERFACE, out of scope

**Source.** An observer or satellite confirms whether the target was actually
neutralized (BDA).

**Interface.** The model treats a round landing within `lethal_radius_m` of the
target as **effect-achieved** and reports that (`destroyed`, `destroying_round`).
This kill assessment is made on the **true** fall of shot, not the noisy
observation: a round that is physically within lethal radius achieved effect
regardless of how precisely the observer measured it (Component 13 keeps this
distinction explicit — observation noise affects *learning*, never the
ground-truth kill).

**Out of scope because:** whether a target within lethal radius is genuinely
destroyed — fuzing, terminal effects, target hardening, cover — is a
weapons-effects / ISR problem. The model reports lethal-radius achievement; a BDA
system confirms destruction. `lethal_radius_m` is an adaptable input
parameter, not a weapons-effects model.

---

## 4. Weather / atmosphere — IN SCOPE, modeled imperfect and learned

**Source.** Soundings measured near the gun (radiosonde / met station) plus a
remote estimate toward the target (satellite / recon), assembled into the MET
message the fire-control computer is told. This told MET is imperfect — it
differs from the real atmosphere along the trajectory.

The told weather is **horizontally varying** (Component 12,
`HorizontalMetField`): conditions are interpolated along the downrange axis
between a gun-location profile and a target-location profile. The gun profile is
measured (real soundings); the **target profile is explicitly a remote estimate**
(satellite/recon), not a measurement — there is no sensor in enemy territory.
This is the standard fire-control MET interpolation between known locations
(STANAG 4082 / FM 6-40). `weather_profile_along_path()` returns the real
position-varying wind plus a confidence value that decreases from the gun
(measured, confidence 1.0) toward the target (remote estimate, `confidence_floor`).

**How it enters the model.** The told field drives the firing solution; the
*hidden true* field drives the real fall of shot, differing most toward the
target where nothing was measured. The residual — the effective downrange wind
error the MET got wrong — is learned from the fall of shot by the atmospheric
Kalman filter (`AtmosphericStateEstimator`, Estimator B), separately from the
gun's fixed mechanical bias (`GunBiasEstimator`, Estimator A). **The shell is the
sensor we cannot place at the target.**

**In scope because:** closing the loop on atmospheric error from observed misses
is central to the model. The system converges on the effective conditions that
explain the misses even though the target-area weather was never measured —
validated in `tests/test_horizontal_weather.py`, including with observation noise
(Component 13) on.

---

## Scope summary

| Battlefield input | Source (sensing layer) | Model treatment | In/out of scope |
|---|---|---|---|
| Fall of shot | FO / drone / radar | Observed **with noise**; estimators learn from it (Component 13) | **In scope** |
| Target location & altitude | Observer / satellite / map | **Consumed as input**; model hits the given coordinates | Out (sensing) |
| Battle damage assessment | Observer / satellite | Model reports lethal-radius achievement; BDA confirms kill | Out (weapons-effects/ISR) |
| Weather / atmosphere | Soundings (gun) + remote estimate (target) | Horizontally-varying told field; residual **learned** from fall of shot | **In scope** |

**Deliberately not modeled here:** target-location uncertainty and
BDA/weapons-effects uncertainty (both defined above as out-of-scope
sensing/effects problems with clean interfaces), and any UI (next phase — though
`weather_profile_along_path()` already returns honest position-varying weather +
confidence data for it to plot). With Component 12 (horizontal weather) and
Component 13 (observation noise) complete, the model now handles every major
real-world factor: full STANAG 4355 physics (drag, wind, altitude atmosphere,
spin drift, Coriolis), target altitude difference, fire-until-destroyed, noisy
observation, and horizontally-varying weather that is reliable near the gun and
learned toward the target.
