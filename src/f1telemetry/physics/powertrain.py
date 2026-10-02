"""P1-T4/P1-T6: the synthesised ICE torque curve, and the 2026 limits that bound the whole unit.

``PHASES.md`` P1-T4 asks for an ICE torque curve that is **synthesised rather than sourced**, with
a turbo-lag multiplier collapsing below about 4 000 rpm. The eight knots and both lag numbers are
``car_spec.yaml`` data reached through
:class:`~f1telemetry.contracts.car_spec.KernelConfig` like every other coefficient, and
``docs/calibration.md`` section 3 carries their provenance. What lives here is the two decisions
that turn a table into a torque any engine speed can ask for, and - since the regulations turned
out to bound the engine through fuel energy flow rather than through a power - the limits that
decide how much of that torque the car is allowed to use.

**Interpolation is inherited, not reinvented.** :func:`ice_torque_nm` reads the curve through
:func:`~f1telemetry.physics.forces.speed_curve` - piecewise linear between knots, held flat at
both ends - because that is the rule Task 3 froze for ``Cl(v)`` and ``Cd(v)`` against the same
reasoning: a table of points is a table, not a fit. One interpolator rather than two that agree
today.

**The lag is a threshold, and the threshold is configured data.** :func:`turbo_lag_factor`
returns the multiplier strictly below ``collapse_below_rpm`` and exactly ``1.0`` at and above it,
which is the literal reading of ``PLAN.md`` section 6's "falls off sharply below ~4 000 rpm" and
of the threshold ``car_spec.yaml`` records. ``PHASES.md`` P1-T4 asks for a multiplier "collapsing
below ~4 000 rpm" and names no ramp back, so a step is what was asked for. Torque is therefore
discontinuous at that rpm, and that is a property of the synthesised data rather than of this
code: the file names where the lag stops and not where full boost returns, so there is nothing to
recover over, and a recovery ramp invented here would put a number in Python that no file
supplies. ``docs/calibration.md`` section 3 states the same shape to a reader checking the file
against this code.

The committed lag threshold and ``idle_rpm`` are both 4 000 rpm, so the committed curve can
only be reduced below idle. Its lowest knot is 2 000 rpm, so the multiplier is live over
2 000-4 000 rpm and held flat below that, on the same clamped-end rule the curve follows.

**The 2026 power-unit limits live here too, and they are the *only* thing that bounds the
engine.** Section C Issue 20 never states an ICE power in kW: C5.2.3 (page 64) caps fuel energy
flow at 3 000 MJ/h, C5.2.4 replaces that below 10 500 rpm with ``0.27*N + 165``, and C5.2.5 adds a
two-arm limit in *engine power*. All three read "must not exceed", so
:func:`ice_fuel_energy_flow_limit_mj_h` returns the smallest of the applicable arms and
:func:`step_ice_torque` cuts the curve back to fit - which is why the published 400 kW shorthand
never reaches the physics at all. Two rules make that check meaningful rather than nominal:

* **The power is the one the engine is making, not the one it was asked for.** ``throttle`` is a
  parameter of :func:`step_ice_torque` rather than something the caller applies afterwards, because
  C5.2.5's arm is a function of delivered power and the arm *rises* with it. Measuring at the full
  request would clip torque the driver is legally allowed to ask for.
* **The cap is not applied at or below zero delivered power.** There is no fuel request to bound
  there, and C5.2.4's linear arm goes negative below about 610 rpm, which is not a physical fuel
  flow. Engine braking and a stalled engine therefore get the curve uncut, as they do in the P1-T4
  tests.

The per-cylinder arm of C5.2.3 is **not** applied. It is cited in ``car_spec.yaml`` and named as
unimplemented there, because a per-cylinder cap needs a cylinder count no block of that file
carries; an invented count would be an unsourced number sitting inside a regulatory check.

**The MGU-K is the same crankshaft's other source of torque**, geared to it at a fixed ratio
(C5.18.2, page 78), so its limits are applied before it joins the drivetrain rather than after.
:func:`step_mgu_k` holds the whole of C5.2.7/.8 (power), C5.2.11 (torque), C5.18.5 (part speed),
C5.2.12 (standing start) and C5.2.9/.10 (energy) in one place, because they are five ways of
saying the same thing - how much of one 4 MJ store may be moved, and in which direction. Three
decisions in it are worth naming here:

* **C5.2.11's 500 Nm is referenced to the crankshaft, and C5.18.4's 520 Nm is not read at all.**
  With the motor shaft three times faster in this model, its shaft torque is capped at 500/3 Nm;
  the 520 Nm figure is only the threshold above which an *optional* limiter may act.
* **C5.2.12 is scoped to a declared grid standing start, and its exception is a declaration.** Both
  halves are caller flags. Whether the FIA Standard ECU mandates minimum acceleration is a fact
  about the mandated launch, not a constant any car file can hold - and enforcing the 50 km/h rule
  outside a standing start would break every mid-corner deployment under it.
* **The per-lap recharge limit is chosen by the event, not fixed at the baseline.** C5.2.10 states
  8.5 MJ and then reduces it; :class:`RechargeEvent` is how a run says which of the three figures
  applies, and the conditional 0.5 MJ allowance is a separate flag on top of whichever is selected.

**Two ways in, as in :mod:`f1telemetry.physics.forces`.** The compiled primitives take flat
scalars and ``float64`` arrays and are what a kernel calls, unvalidated because a loop cannot
afford Python in it; :func:`step_ice_torque` and :func:`step_mgu_k` are the Python-facing
compositions that read them off the configuration and are the one place those values are checked
before any arithmetic happens. ``KernelConfig`` is public and replaceable, so a loader's guarantees
are a property of the loader rather than a guarantee about the object. Nothing here has state of
its own, allocates, or reads a clock, so a run's determinism does not have to be re-earned on this
path; the MGU-K's own state is a caller-owned buffer it writes in place, like the gearbox's.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import TYPE_CHECKING, Final

import numpy as np
from numba import njit

from f1telemetry.physics.forces import speed_curve  # noqa: TID251 -- same-layer kernel helper

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import KernelConfig

__all__ = [
    "LAP_RECHARGE_INDEX",
    "MGU_K_STATE_SIZE",
    "SOC_INDEX",
    "RechargeEvent",
    "begin_lap",
    "ice_fuel_energy_flow_limit_mj_h",
    "ice_fuel_energy_flow_mj_h",
    "ice_power_kw",
    "ice_torque_nm",
    "mgu_k_initial_state",
    "mgu_k_power_limit_kw",
    "mgu_k_rpm",
    "mgu_k_shaft_torque_nm",
    "step_ice_torque",
    "step_mgu_k",
    "turbo_lag_factor",
]

# C5.2.5 is stated in MJ/h and C5.2.7 in kW, so every conversion between an energy flow, a power
# and a torque in this module goes through one of these two constants rather than through a
# literal. 3.6 is MJ/h per kW (1 kW = 1 kJ/s), 1e6 is joules per MJ.
_MJ_H_PER_KW: Final[float] = 3.6
_J_PER_MJ: Final[float] = 1.0e6

# Every message this module raises names the entry point that raised it. Two functions here share
# the scalar helpers, and `request_nm must be a real number` is a worse error than one that says
# which of them refused it.
_ICE: Final[str] = "step_ice_torque"
_MGU_K: Final[str] = "step_mgu_k"


class RechargeEvent(IntEnum):
    """Which of C5.2.10's per-lap recharge figures applies to this run.

    **C5.2.10 states a baseline and then reduces it, so there is no single number to read.** It
    gives 8.5 MJ/lap and then lists conditions that lower it - 7 MJ at some competitions, and a
    4 MJ floor in Qualifying - with a conditional 0.5 MJ allowance on top of whichever applies.
    Treating 8.5 as a universal cap would enforce a rule the article does not state; treating the
    4 MJ floor as universal would understate what a race allows. The choice is therefore a fact
    about the event the run is at, and a caller declares it.

    The members are an :class:`~enum.IntEnum` so the compiled step can take the integer code
    without a second conversion, and so a scenario that already holds an integer does not have to
    translate it. An unknown code is refused rather than coerced, because every comparison would
    silently fail and the run would fall back to the *last* arm - a stricter limit than the event
    asked for, with nothing in the output to say so.
    """

    RACE = 0
    REDUCED = 1
    QUALIFYING = 2


# The configuration field each event selects, resolved per step rather than once: a caller may
# change the event between laps, and a table of offsets would be a second place for the ordering
# C5.2.10 states to be wrong.
_RECHARGE_LIMITS: Final[dict[RechargeEvent, str]] = {
    RechargeEvent.RACE: "recharge_limit_mj_per_lap",
    RechargeEvent.REDUCED: "recharge_limit_reduced_mj_per_lap",
    RechargeEvent.QUALIFYING: "recharge_limit_qualifying_floor_mj_per_lap",
}
_RECHARGE_CODES: Final[frozenset[int]] = frozenset(int(member) for member in RechargeEvent)

# The caller owns both slots. The state of charge is the 4 MJ usable window of C5.2.9, advanced
# by every deployment and every recovery, and the lap accumulator is C5.2.10's per-lap budget.
# Both are written in place, exactly as the gearbox step writes its own state.
MGU_K_STATE_SIZE: Final[int] = 2
SOC_INDEX: Final[int] = 0
LAP_RECHARGE_INDEX: Final[int] = 1

# rad/s per rpm, the conversion every power in this module is built from. Written once so the
# ICE and the MGU-K cannot disagree about it by a rounding difference.
_RAD_PER_S_PER_RPM: Final[float] = math.tau / 60.0


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def turbo_lag_factor(
    rpm: float,
    collapse_below_rpm: float,
    multiplier_at_collapse: float,
) -> float:
    """``multiplier_at_collapse`` strictly below ``collapse_below_rpm``.

    Exactly ``1.0`` at or above.

    The turbo-lag reduction of ``PLAN.md`` section 6, expressed as the one decision it leaves
    open: a threshold at a configured engine speed rather than a spool characteristic, which
    would need a boost curve the regulations do not publish. ``1.0`` at the threshold rather than
    a hair under it, so ``collapse_below_rpm`` means what it says - and a caller can see the
    reduction in this return value rather than inferring it from the torque.
    """
    if rpm < collapse_below_rpm:
        return multiplier_at_collapse
    return 1.0


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def ice_torque_nm(
    rpm: float,
    torque_rpm: np.ndarray,
    torque_nm: np.ndarray,
    collapse_below_rpm: float,
    multiplier_at_collapse: float,
) -> float:
    """ICE torque in newton-metres at ``rpm``, with the turbo lag applied.

    The synthesised curve interpolated between its knots and held flat outside them, multiplied by
    :func:`turbo_lag_factor`. **Throttle is not applied here**: this is the engine's delivered
    torque at full request, and scaling it by throttle position is the drivetrain's call in
    P1-T5.

    Negative and zero ``rpm`` are not refused, because the ends are held flat and a stalled or
    backfiring engine should get a finite number rather than an index error. ``torque_rpm`` must
    be strictly increasing and share a length with ``torque_nm``, which the ``car_spec.yaml``
    loader checks before it builds a ``KernelConfig`` and :func:`step_ice_torque` checks again
    before it hands over a config that may have been replaced. The one rule no loader applies is
    the sign of ``torque_nm``, so it is :func:`step_ice_torque`'s alone.
    """
    return speed_curve(rpm, torque_rpm, torque_nm) * turbo_lag_factor(
        rpm, collapse_below_rpm, multiplier_at_collapse
    )


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def ice_power_kw(torque_nm: float, rpm: float) -> float:
    """Crankshaft power in kW, which is what both fuel-energy-flow clauses are stated against.

    Negative below zero by construction: a torque at the crank and a negative engine speed mean
    the engine is absorbing work, not delivering it. Callers that need a magnitude take the
    absolute value themselves rather than having it applied here, because the sign is what tells
    a deployment from a braking overrun apart.
    """
    return torque_nm * rpm * _RAD_PER_S_PER_RPM / 1_000.0


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def ice_fuel_energy_flow_mj_h(power_kw: float, efficiency: float) -> float:
    """The fuel energy flow that ``power_kw`` of shaft power costs, in MJ/h.

    The inverse of an efficiency the regulations do not publish, which is why ``efficiency`` is
    ``car_spec.yaml`` data declared synthetic rather than a constant here: C5.2.3, C5.2.4 and
    C5.2.5 all bound the engine in MJ/h, and every number the model has at the crankshaft is in
    kW, so without this conversion none of the three clauses could be checked at all.
    """
    return power_kw * _MJ_H_PER_KW / efficiency


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def ice_fuel_energy_flow_limit_mj_h(
    rpm: float,
    power_kw: float,
    max_mj_h: float,
    low_rpm_limit_rpm: float,
    low_rpm_gain: float,
    low_rpm_offset_mj_h: float,
    partial_threshold_kw: float,
    partial_gain: float,
    partial_offset_mj_h: float,
    partial_min_mj_h: float,
) -> float:
    """The smallest fuel energy flow any of C5.2.3, C5.2.4 and C5.2.5 permits, in MJ/h.

    **Three articles, each "must not exceed", so the binding one is the minimum.** C5.2.3 caps
    the total at 3 000 MJ/h; C5.2.4 replaces that *below* 10 500 rpm with ``0.27*N + 165``; and
    C5.2.5 adds a limit in engine power - a flat 380 MJ/h at or below -50 kW and ``9.78*P + 869``
    above it. Taking the minimum is the only reading under which all three hold at once, and it is
    what keeps the published 400 kW shorthand out of the physics: that figure is not a clause, and
    a model that checked against it instead would be enforcing a number the FIA never wrote.

    **Both comparisons are on the side the clauses state.** C5.2.4 reads "below 10500 rpm", so the
    rpm arm applies strictly below its crossover; C5.2.5's flat arm reads "at or below", so it
    applies inclusively at the threshold. One of them being inclusive and the other exclusive is
    not an inconsistency - it is the difference between the article's two wordings - and a shared
    comparison would halve one arm exactly at the speed a standing start sits on.
    """
    limit = max_mj_h
    if rpm < low_rpm_limit_rpm:
        rpm_limit = low_rpm_gain * rpm + low_rpm_offset_mj_h
        if rpm_limit < limit:
            limit = rpm_limit
    if power_kw <= partial_threshold_kw:
        if partial_min_mj_h < limit:
            limit = partial_min_mj_h
    else:
        power_limit = partial_gain * power_kw + partial_offset_mj_h
        if power_limit < limit:
            limit = power_limit
    return limit


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def mgu_k_rpm(ice_rpm: float, crankshaft_ratio: float) -> float:
    """MGU-K part speed, the product C5.18.2's fixed gearing implies and C5.18.5 caps.

    The clause requires the coupling to be permanent and fixed but states no value, so the ratio
    is this project's; the part's speed is whatever that ratio makes of the engine's. Both
    clauses are read here and nowhere else, which is what keeps the speed-dependent power cap and
    the relative-speed ceiling talking about the same number.
    """
    return ice_rpm * crankshaft_ratio


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def mgu_k_power_limit_kw(
    speed_km_h: float,
    ers_speed_km_h: np.ndarray,
    ers_limit_kw: np.ndarray,
) -> float:
    """The effective ERS-K power limit at ``speed_km_h``, in kW.

    **The curve in ``KernelConfig`` is already ``min(C5.2.7, C5.2.8)``**, so this is a lookup and
    not a formula: C5.2.8's own ``1800 - 5v`` permits 1 800 kW at rest, and C5.2.7's absolute
    350 kW cap does not. Reading the clause's formula instead of the stored curve would hand the
    motor five times the permitted power below 290 km/h, which is why the curves are clamped in
    ``car_spec.yaml``, checked by ``build_kernel_config``, and read here.

    The interpolation and the clamping at both ends are inherited from
    :func:`~f1telemetry.physics.forces.speed_curve` for the reason the aero curves gave - a table
    of points is a table, not a fit. Both arrays must be the same length and strictly increasing,
    which the loader checks; ``ers_speed_km_h`` and ``ers_overtake_speed_km_h`` have different
    lengths (4 and 3 breakpoints), so each is read with its own partner and never against a
    shared bound.
    """
    return speed_curve(speed_km_h, ers_speed_km_h, ers_limit_kw)


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def mgu_k_shaft_torque_nm(
    request_nm: float,
    ice_rpm: float,
    dc_limit_kw: float,
    crankshaft_ratio: float,
    motor_inverter_efficiency: float,
    torque_limit_nm: float,
) -> float:
    """The requested MGU-K torque after C5.2.7/.8's power cap and C5.2.11's torque cap.

    **Both caps are magnitudes, so the sign survives them and only the magnitude is reduced.**
    Regeneration is not exempt from C5.2.7: the clause caps the *absolute* electrical DC power, and
    a model that let recovery through uncapped would move energy the article does not allow.

    **C5.2.11 is referenced to the crankshaft, so the cap is divided by the ratio.** The MGU-K
    spins at ``crankshaft_ratio`` times the engine, so the crankshaft-equivalent torque is its
    shaft torque times that ratio. C5.18.4's 520 Nm is *not* used
    here: it is the threshold above which an optional torque-limiting device may act, so it is not
    a second cap and ``KernelConfig`` does not carry it at all.

    **At a standstill the power cap cannot bind**, because torque at zero speed costs no power.
    The arithmetic divides by the part's angular velocity, which is zero there, so the power arm is
    skipped rather than guarded with a large constant - the crank torque cap is still applied, so a
    stationary motor cannot be asked for unlimited torque.
    """
    if request_nm == 0.0:
        return 0.0
    magnitude = -request_nm if request_nm < 0.0 else request_nm

    # C5.2.11 is stated at the crankshaft. With the MGU-K spinning `ratio` times
    # faster, its shaft torque must be lower by that ratio to deliver the same power.
    crank_cap = torque_limit_nm / crankshaft_ratio
    if magnitude > crank_cap:
        magnitude = crank_cap

    part_omega = mgu_k_rpm(ice_rpm, crankshaft_ratio)
    if part_omega < 0.0:
        part_omega = -part_omega
    part_omega *= _RAD_PER_S_PER_RPM
    if part_omega > 0.0:
        dc_to_mechanical = (
            motor_inverter_efficiency if request_nm > 0.0 else 1.0 / motor_inverter_efficiency
        )
        power_cap = dc_limit_kw * dc_to_mechanical * 1_000.0 / part_omega
        if power_cap < magnitude:
            magnitude = power_cap

    if request_nm < 0.0:
        return -magnitude
    return magnitude


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def _step_mgu_k(
    state: np.ndarray,
    request_nm: float,
    ice_rpm: float,
    speed_km_h: float,
    dt_s: float,
    dc_limit_kw: float,
    crankshaft_ratio: float,
    motor_inverter_efficiency: float,
    torque_limit_nm: float,
    relative_speed_limit_rpm: float,
    launch_speed_kmh: float,
    store_energy_mj: float,
    grid_standing_start: bool,
    ecu_mandates_min_acceleration: bool,
    recharge_limit_mj_per_lap: float,
) -> float:
    """One step of the MGU-K, writing ``state`` in place and returning crankshaft torque.

    Nothing here is checked - see the module docstring - so it must only ever be reached through
    :func:`step_mgu_k`.

    **The order of the four gates is the model.** C5.18.5's part speed comes first because it is a
    hardware ceiling: past it the motor transmits nothing in either direction, and no amount of
    request or store changes that. C5.2.12's standing-start rule comes next, and it only ever
    removes *positive* torque - regeneration below 50 km/h is what gets the car there. The
    mechanical caps come next, then the energy is settled against the store and the lap budget and
    the state is advanced.

    **The energy is settled by scaling the torque, not by refusing it.** A deployment larger than
    the store holds is cut back to what the store can supply over ``dt_s``, and a recovery larger
    than the room left or the lap budget allows is cut back to fit. Both leave the invariant true
    after any number of steps, which a clamp applied to the state *after* the fact could not: the
    state would read 0 for a step that had already been integrated as a full deployment.
    """
    relative_rpm = mgu_k_rpm(ice_rpm, crankshaft_ratio)
    if relative_rpm > relative_speed_limit_rpm or relative_rpm < -relative_speed_limit_rpm:
        return 0.0

    if (
        grid_standing_start
        and not ecu_mandates_min_acceleration
        and speed_km_h < launch_speed_kmh
        and request_nm > 0.0
    ):
        request_nm = 0.0

    torque_nm = mgu_k_shaft_torque_nm(
        request_nm,
        ice_rpm,
        dc_limit_kw,
        crankshaft_ratio,
        motor_inverter_efficiency,
        torque_limit_nm,
    )

    part_omega = mgu_k_rpm(ice_rpm, crankshaft_ratio)
    if part_omega < 0.0:
        part_omega = -part_omega
    part_omega *= _RAD_PER_S_PER_RPM
    if part_omega <= 0.0:
        return torque_nm * crankshaft_ratio

    joules_per_nm = part_omega * dt_s
    if torque_nm > 0.0:
        # State of charge is electrical energy: motoring draws more DC than shaft energy.
        wanted_mj = torque_nm * joules_per_nm / motor_inverter_efficiency / _J_PER_MJ
        available_mj = state[SOC_INDEX]
        if wanted_mj > available_mj:
            torque_nm = torque_nm * available_mj / wanted_mj
        state[SOC_INDEX] -= torque_nm * joules_per_nm / motor_inverter_efficiency / _J_PER_MJ
    elif torque_nm < 0.0:
        # Regeneration stores only the fraction that survives the motor/inverter.
        wanted_mj = -torque_nm * joules_per_nm * motor_inverter_efficiency / _J_PER_MJ
        room_mj = store_energy_mj - state[SOC_INDEX]
        budget_mj = recharge_limit_mj_per_lap - state[LAP_RECHARGE_INDEX]
        if budget_mj < room_mj:
            room_mj = budget_mj
        if room_mj < 0.0:
            room_mj = 0.0
        if wanted_mj > room_mj:
            torque_nm = torque_nm * room_mj / wanted_mj
        stored_mj = -torque_nm * joules_per_nm * motor_inverter_efficiency / _J_PER_MJ
        state[SOC_INDEX] += stored_mj
        state[LAP_RECHARGE_INDEX] += stored_mj
    # The public drivetrain joins sources at the crankshaft. The motor shaft is faster by
    # `crankshaft_ratio`, so its torque is converted back to the equivalent crank torque here.
    return torque_nm * crankshaft_ratio


def mgu_k_initial_state(config: KernelConfig, soc_mj: float | None = None) -> np.ndarray:
    """Caller-owned float64 state buffer, ``[state_of_charge_mj, lap_recharge_mj]``.

    **A full store is the launch condition**, so ``soc_mj`` defaults to the whole of C5.2.9's
    window rather than to zero. C5.2.9 bounds the *swing* between the maximum and minimum state
    of charge, so the model carries the usable window as ``[0, store_energy_mj]`` and the absolute
    state of charge of a real 2026 car - which the regulations never state - never appears.

    ``lap_recharge_mj`` starts at zero because it is a per-lap budget; :func:`begin_lap` is what
    restarts one.

    Nothing is checked here, exactly as in
    :func:`~f1telemetry.physics.gearbox.initial_state`: this only manufactures a buffer, and
    :func:`step_mgu_k` is the boundary that decides whether the numbers in it are usable.
    """
    charge = (
        float(config.store_energy_mj)
        if soc_mj is None
        else _checked_float(_MGU_K, "soc_mj", soc_mj)
    )
    return np.array([charge, 0.0], dtype=np.float64)


def begin_lap(state: np.ndarray) -> None:
    """Zero the lap recharge accumulator, leaving the state of charge alone.

    A lap boundary is not a store reset. C5.2.10's budget is per lap and C5.2.9's window is not
    tied to a lap at all, so resetting both here would hand every car a full 4 MJ of free
    deployment at the line - which is the mistake that comes from reading a per-lap limit as a
    whole-race one in the other direction.
    """
    _check_state(state)
    state[LAP_RECHARGE_INDEX] = 0.0


def step_ice_torque(config: KernelConfig, rpm: float, throttle: float = 1.0) -> float:
    """ICE torque in newton-metres at ``rpm``, read off a validated configuration.

    The Python-facing entry point, and the reason it exists: a caller inside a kernel reads the
    curve and the two lag scalars out of the config once, outside its loop, and calls
    :func:`ice_torque_nm` directly - the dataclass is not something a compiled loop can hold.
    That a caller outside a kernel can ask for the torque in one call is the whole of the job.

    ``KernelConfig`` is public and replaceable, so the values are checked here rather than
    assumed to have been checked by whoever built it. ``ice_torque_nm`` is compiled with
    ``boundscheck=False`` and no finiteness checks, so an empty curve reads uninitialised memory,
    a mismatched length reads past the end of ``torque_nm``, a reversed axis finds a segment that
    is not there, and a NaN ``rpm`` weighs nothing - each returning a plausible torque rather than
    an error, and a wrong torque is exactly what this model is asked to produce. A negative knot
    reads just as quietly and for the other reason: it is not a lookup failure but an engine that
    pushes the wrong way. The primitives stay unvalidated because a kernel calls them once per
    step and cannot afford Python.

    **``throttle`` is part of this call rather than a multiplication the caller does afterwards**,
    because the fuel-energy-flow limit below is a function of delivered power and C5.2.5's arm
    rises with it. Scaling the pedal in the drivetrain instead would measure the cap at the
    full-request power and cut torque the driver is legally allowed to ask for - silently, and by
    an amount that grows as the pedal comes off.

    **The returned torque is what C5.2.3, C5.2.4 and C5.2.5 permit at the delivered power**, so
    it is the curve cut back to fit rather than the curve itself. The published 400 kW peak plays
    no part: it is not in Section C, and with the committed curve and efficiency the cap does not
    bite at all, which is the property ``tests/test_powertrain.py`` asserts so that a later edit
    cannot turn a bound into a hidden calibration of the table.

    ``rpm``, ``throttle`` and every scalar it reads are narrowed to ``float`` so equivalent ``int``
    and ``float`` callers share one compiled specialisation rather than compiling a second
    signature for each.
    """
    rpm = _checked_float(_ICE, "rpm", rpm)
    throttle = _checked_pedal("throttle", throttle)
    collapse_rpm = _checked_float(
        _ICE, "config.turbo_lag_collapse_rpm", config.turbo_lag_collapse_rpm
    )
    multiplier = _checked_float(_ICE, "config.turbo_lag_multiplier", config.turbo_lag_multiplier)
    if collapse_rpm <= 0.0:
        raise ValueError(
            "step_ice_torque: config.turbo_lag_collapse_rpm must be finite and > 0, "
            f"got {collapse_rpm!r}"
        )
    if not 0.0 < multiplier <= 1.0:
        raise ValueError(
            f"step_ice_torque: config.turbo_lag_multiplier must be in (0, 1], got {multiplier!r}"
        )
    _check_torque_arrays(config)
    torque_nm = throttle * ice_torque_nm(
        rpm,
        config.torque_rpm,
        config.torque_nm,
        collapse_rpm,
        multiplier,
    )
    return _within_fuel_energy_flow(config, rpm, torque_nm)


def _within_fuel_energy_flow(config: KernelConfig, rpm: float, torque_nm: float) -> float:
    """Cut ``torque_nm`` back to the fuel energy flow C5.2.3/.4/.5 permit at its own power.

    The check is on the power the engine is *making*, so it is only applied above zero. Below
    that there is no fuel request to bound - the engine is braking or not turning - and C5.2.4's
    linear arm goes negative at about 610 rpm, which would otherwise read as "this engine may burn
    no fuel at all" and silently zero the whole curve below idle.

    Scaling by the ratio of the two powers rather than recomputing a torque from the limit keeps
    the throttle and the turbo-lag multiplier exact: both already scaled the torque, and dividing
    by the same arithmetic would be a second chance to disagree with them.
    """
    power_kw = ice_power_kw(torque_nm, rpm)
    if power_kw <= 0.0:
        return torque_nm
    efficiency = _checked_fraction(
        _ICE, "config.fuel_to_shaft_efficiency", config.fuel_to_shaft_efficiency
    )
    limit_mj_h = ice_fuel_energy_flow_limit_mj_h(
        rpm,
        power_kw,
        config.fuel_energy_flow_max_mj_h,
        config.fuel_energy_flow_low_rpm_limit_rpm,
        config.fuel_energy_flow_low_rpm_gain,
        config.fuel_energy_flow_low_rpm_offset_mj_h,
        config.fuel_energy_flow_partial_load_threshold_kw,
        config.fuel_energy_flow_partial_load_gain,
        config.fuel_energy_flow_partial_load_offset_mj_h,
        config.fuel_energy_flow_partial_load_min_mj_h,
    )
    allowed_kw = limit_mj_h * efficiency / _MJ_H_PER_KW
    if power_kw <= allowed_kw:
        return torque_nm
    return torque_nm * allowed_kw / power_kw


def step_mgu_k(
    config: KernelConfig,
    state: np.ndarray,
    request_nm: float,
    ice_rpm: float,
    speed_m_s: float,
    dt_s: float | None = None,
    *,
    overtake: bool = False,
    grid_standing_start: bool = False,
    ecu_mandates_minimum_acceleration: bool = False,
    recharge_event: RechargeEvent | int = RechargeEvent.RACE,
    recharge_allowance_applies: bool = False,
) -> float:
    """The MGU-K torque this step delivers, in newton-metres at the motor's own shaft.

    **Every limit C5.2.7 to C5.2.12 and C5.18.2 to C5.18.5 put on the motor is applied here**, in
    one place, because they are five statements about one 4 MJ store and how fast it may be moved.
    :func:`step_ice_torque` is the ICE's equivalent, and the two are joined at the crankshaft by
    :func:`~f1telemetry.physics.gearbox.step_drivetrain` - upstream of the clutch and gearbox, as
    C5.18.2's fixed gearing requires.

    **The request is the MGU-K's own actuator command in newton-metres at its shaft**, positive
    for deployment and negative for regeneration. The return value is equivalent crankshaft torque
    so it can be summed directly with ICE torque. It is deliberately *not* a throttle or a 0..1
    deployment fraction: the caps are stated in torque and power, so a caller that asked for a
    percentage would have to convert with a peak torque the regulations do not state. A caller with
    a deployment strategy passes the torque it wants and learns what it got.

    **Five of the arguments are declarations, not quantities**, and each is one clause:

    * ``grid_standing_start`` - C5.2.12's 50 km/h rule applies *only* from a grid standing start.
      Enforcing it unconditionally would break every mid-corner deployment under 50 km/h.
    * ``ecu_mandates_minimum_acceleration`` - the article's exception. Whether the FIA Standard
      ECU mandates minimum acceleration is a fact about the mandated launch, not a constant a car
      file could carry, so it cannot be a number in ``car_spec.yaml`` and is not one.
    * ``overtake`` - which of the two C5.2.8 profiles applies. ``KernelConfig`` carries both
      curves and they have different lengths, so the caller picks and each is read with its own.
    * ``recharge_event`` - which of C5.2.10's three per-lap figures applies.
    * ``recharge_allowance_applies`` - the article's conditional 0.5 MJ allowance, added to
      whichever figure the event selected.

    **``dt_s`` defaults to the configured step** so a caller running at the file's rate does not
    have to pass it, and the energy settled per step cannot then disagree with the step the
    integrator takes.

    Returns crankshaft-equivalent torque actually applied, which may be smaller than the request
    for any of the five reasons above and may be exactly ``0.0``. ``state`` is written in place, so
    a run steps one buffer over and over with no allocation.
    """
    _check_state(state)
    request = _checked_float(_MGU_K, "request_nm", request_nm)
    ice = _checked_float(_MGU_K, "ice_rpm", ice_rpm)
    speed = _checked_float(_MGU_K, "speed_m_s", speed_m_s)
    step = config.dt_s if dt_s is None else _checked_float(_MGU_K, "dt_s", dt_s)
    if step <= 0.0:
        raise ValueError(
            f"{_MGU_K}: dt_s must be finite and > 0, got {step!r}. It is the interval the store "
            "energy is settled over, so a zero step would divide by nothing"
        )
    ratio = _checked_float(_MGU_K, "config.mgu_k_crankshaft_ratio", config.mgu_k_crankshaft_ratio)
    if ratio <= 0.0:
        raise ValueError(
            f"{_MGU_K}: config.mgu_k_crankshaft_ratio must be finite and > 0, got {ratio!r}. "
            "C5.18.2's fixed coupling has to have a direction and a magnitude; zero would leave "
            "the motor with no coupling to join at and a negative one would drive the "
            "crankshaft backwards"
        )
    efficiency = _checked_fraction(
        _MGU_K, "config.mgu_k_motor_inverter_efficiency", config.mgu_k_motor_inverter_efficiency
    )
    torque_limit = _checked_float(
        _MGU_K, "config.mgu_k_torque_limit_nm", config.mgu_k_torque_limit_nm
    )
    relative_limit = _checked_float(
        _MGU_K, "config.mgu_k_relative_speed_limit_rpm", config.mgu_k_relative_speed_limit_rpm
    )
    store = _checked_float(_MGU_K, "config.store_energy_mj", config.store_energy_mj)
    launch = _checked_float(_MGU_K, "config.launch_speed_kmh", config.launch_speed_kmh)
    event = _checked_event(recharge_event)
    standing_start = _checked_flag("grid_standing_start", grid_standing_start)
    ecu_mandates = _checked_flag(
        "ecu_mandates_minimum_acceleration", ecu_mandates_minimum_acceleration
    )
    use_overtake = _checked_flag("overtake", overtake)
    allowance = _checked_flag("recharge_allowance_applies", recharge_allowance_applies)

    limit_field = _RECHARGE_LIMITS[event]
    base = _checked_float(_MGU_K, f"config.{limit_field}", getattr(config, limit_field))
    recharge_limit = base
    if allowance:
        recharge_limit += _checked_float(
            _MGU_K, "config.recharge_allowance_mj_per_lap", config.recharge_allowance_mj_per_lap
        )

    speed_km_h = speed * 3.6
    if speed_km_h < 0.0:
        speed_km_h = -speed_km_h
    if use_overtake:
        speeds, limits = _curve(config.ers_overtake_speed_km_h, config.ers_overtake_limit_kw, True)
    else:
        speeds, limits = _curve(config.ers_speed_km_h, config.ers_limit_kw, False)
    dc_limit_kw = mgu_k_power_limit_kw(speed_km_h, speeds, limits)

    return float(
        _step_mgu_k(
            state,
            request,
            ice,
            speed_km_h,
            step,
            dc_limit_kw,
            ratio,
            efficiency,
            torque_limit,
            relative_limit,
            launch,
            store,
            standing_start,
            ecu_mandates,
            recharge_limit,
        )
    )


def _curve(speeds: np.ndarray, limits: np.ndarray, overtake: bool) -> tuple[np.ndarray, np.ndarray]:
    """One C5.2.8 profile's two arrays, checked as a pair before the lookup indexes them.

    The two profiles have different lengths - four breakpoints without Overtake and three with it -
    so they can never be read against one shared bound. That hazard is why this returns the pair
    together rather than taking both curves' arrays as separate arguments somewhere higher up: the
    arrays that go to :func:`mgu_k_power_limit_kw` are always the two that belong together.
    """
    name = "ers_overtake" if overtake else "ers"
    for label, array in ((f"{name}_speed_km_h", speeds), (f"{name}_limit_kw", limits)):
        if (
            not isinstance(array, np.ndarray)
            or array.dtype != np.float64
            or array.ndim != 1
            or array.size == 0
            or not array.flags.c_contiguous
        ):
            raise ValueError(
                f"step_mgu_k: config.{label} must be a non-empty C-contiguous float64 vector"
            )
    if speeds.shape != limits.shape:
        raise ValueError(
            f"step_mgu_k: config.{name}_speed_km_h and config.{name}_limit_kw must share a "
            f"length, got {speeds.shape} and {limits.shape}"
        )
    if not np.isfinite(speeds).all() or not np.isfinite(limits).all():
        raise ValueError(
            f"step_mgu_k: config.{name}_speed_km_h and config.{name}_limit_kw must be finite"
        )
    if not np.all(np.diff(speeds) > 0.0):
        raise ValueError(f"step_mgu_k: config.{name}_speed_km_h must be strictly increasing")
    return speeds, limits


def _check_state(state: np.ndarray) -> None:
    """The caller's MGU-K buffer, before a ``boundscheck=False`` step writes it.

    A writable C-contiguous float64 vector of exactly :data:`MGU_K_STATE_SIZE` entries, because
    the compiled step advances both of them in place. The *values* are not range-checked here
    beyond finiteness: a state of charge outside the window is a caller bug rather than a
    un-lookupable input, and refusing it here would hide the window's own bound - which is
    enforced by the step, on purpose.
    """
    if (
        not isinstance(state, np.ndarray)
        or state.dtype != np.float64
        or state.shape != (MGU_K_STATE_SIZE,)
        or not state.flags.c_contiguous
    ):
        raise ValueError(
            f"step_mgu_k: state must be a C-contiguous float64 vector of length "
            f"{MGU_K_STATE_SIZE} (state of charge, lap recharge), got {type(state).__name__} "
            f"{getattr(state, 'shape', None)} of {getattr(state, 'dtype', None)}"
        )
    if not state.flags.writeable:
        raise ValueError(
            "step_mgu_k: state is read-only, and it is where this step writes the state of charge "
            "and the lap accumulator. A read-only buffer is an out-of-bounds write"
        )
    if not np.isfinite(state).all():
        raise ValueError("step_mgu_k: state must be finite")


def _checked_event(value: object) -> RechargeEvent:
    """The recharge event as the member the compiled step's limit table was built from.

    Refused rather than coerced, and the reason is what an unknown code would do: the lookup below
    would raise a bare ``KeyError`` from a dict that reads as configuration rather than as a
    model, which is a worse error than a message that names what was expected.
    """
    if isinstance(value, bool) or not isinstance(value, (RechargeEvent, int)):
        raise ValueError(
            f"step_mgu_k: recharge_event must be a RechargeEvent or its integer code, got {value!r}"
        )
    if int(value) not in _RECHARGE_CODES:
        raise ValueError(
            f"step_mgu_k: recharge_event must be one of "
            f"{', '.join(member.name for member in RechargeEvent)}, got {int(value)!r}"
        )
    return RechargeEvent(int(value))


def _checked_flag(label: str, value: object) -> bool:
    """A declaration, and therefore a real ``bool``.

    Refused for ``int`` the way a gear request is: these five arguments say *which situation the
    car is in*, and one arriving as ``1`` is a coincidence rather than the answer. Passing one
    straight into the compiled step would also compile a second specialisation against the
    ``boolean`` one, and every later call would have to be checked against both.
    """
    if not isinstance(value, bool):
        raise ValueError(f"step_mgu_k: {label} must be True or False, got {value!r}")
    return value


def _checked_pedal(label: str, value: object) -> float:
    """A pedal position: a finite number in ``[0, 1]``."""
    number = _checked_float(_ICE, label, value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{_ICE}: {label} must be in [0, 1], got {number!r}")
    return number


def _checked_float(prefix: str, label: str, value: object) -> float:
    """Return a finite numeric scalar as float, or fail before it reaches Numba.

    The entry point is named in every message rather than left to the reader of the traceback:
    two functions in this module share these helpers, and ``request_nm must be a real number`` is
    a worse error than one that says which function refused it.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{prefix}: {label} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{prefix}: {label} must be finite, got {value!r}")
    return number


def _checked_fraction(prefix: str, label: str, value: object) -> float:
    """A value in ``(0, 1]`` - a share of something, not a magnitude.

    Every fraction in this file is one a zero would make undefined rather than merely unhelpful:
    both efficiencies divide the fuel energy flow or the electrical power by themselves, and the
    crankshaft ratio is the divisor of the C5.2.11 conversion. An above-one is refused for the
    other reason - it would report more shaft power than the fuel burned, and every
    fuel-energy-flow limit would become unreachable.
    """
    number = _checked_float(prefix, label, value)
    if not 0.0 < number <= 1.0:
        raise ValueError(f"{prefix}: {label} must be in (0, 1], got {number!r}")
    return number


def _check_torque_arrays(config: KernelConfig) -> None:
    """Validate the torque curve before it reaches the compiled lookup.

    Every rule here is one ``speed_curve`` cannot make for itself once the bounds check is off.
    The sign rule is the one that is not about reading the table at all: ``ice_torque_nm``
    returns whatever the table says, so a negative knot is a backwards push at full throttle
    request, and a drivetrain that cannot tell engine braking from drive torque is exactly the
    drivetrain P1-T5 builds. Zero is allowed, because an engine that makes no torque is a real
    operating point rather than a malformed table - unlike ``Cl`` and ``Cd``, which have no zero.
    """
    axis, values = config.torque_rpm, config.torque_nm
    arrays = (("torque_rpm", axis), ("torque_nm", values))
    for name, array in arrays:
        if (
            not isinstance(array, np.ndarray)
            or array.dtype != np.float64
            or array.ndim != 1
            or array.size == 0
            or not array.flags.c_contiguous
        ):
            raise ValueError(
                f"step_ice_torque: config.{name} must be a non-empty C-contiguous float64 vector"
            )
    if axis.shape != values.shape:
        raise ValueError(
            "step_ice_torque: config.torque_rpm and config.torque_nm must share a length, "
            f"got {axis.shape} and {values.shape}"
        )
    for name, array in arrays:
        if not np.isfinite(array).all():
            raise ValueError(f"step_ice_torque: config.{name} must be finite")
    if not np.all(np.diff(axis) > 0.0):
        raise ValueError("step_ice_torque: config.torque_rpm must be strictly increasing")
    if np.any(values < 0.0):
        raise ValueError("step_ice_torque: config.torque_nm must be non-negative")
