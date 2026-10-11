"""P1-T5/P1-T6 driver-requested shifts, clutch demand, and the MGU-K at the crankshaft.

Explicit requests control shifts; rpm never changes gear automatically. The caller owns gear,
shift timer, and clutch engagement. Neutral transmits zero, reverse uses a synthetic ratio with
negative torque, and C9.2.5 clutch demand limits torque after gear reduction.

**The MGU-K joins at the crankshaft, before the gearbox and the clutch.** C5.18.2 (page 78) requires
the motor's rotating parts to be permanently geared to the ICE at a *fixed* ratio to the
crankshaft, and that coupling is this model's inferred upstream boundary rather than a second path
into the differential: the driveline torque is
``(throttle * ice + mgu_k) * ratio * final_drive`` limited by one C9.2.5 clutch demand. The ratio
itself is synthesised - the clause fixes its property and states no value - so it is
``car_spec.yaml`` data, and :func:`~f1telemetry.physics.powertrain.step_mgu_k` is what turns the
motor's request into the torque that joins here.

**The throttle is the engine's alone.** The MGU-K is its own actuator with its own request, and
scaling its torque by the ICE pedal would make deployment a function of a control the driver does
not have for it.

Wheel speed and force assembly remain P1-T6/T7.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import TYPE_CHECKING, Final

import numpy as np
from numba import njit

from f1telemetry.physics.powertrain import (  # noqa: TID251 -- same-layer composition
    RechargeEvent,
    step_ice_torque,
    step_mgu_k,
)

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import KernelConfig

__all__ = [
    "CLUTCH_INDEX",
    "GEAR_INDEX",
    "NEUTRAL_GEAR",
    "REVERSE_GEAR",
    "SHIFT_TIMER_INDEX",
    "STATE_SIZE",
    "GearRequest",
    "clutch_demand_nm",
    "clutch_output_torque_nm",
    "initial_state",
    "shift_remaining_after",
    "step_drivetrain",
    "step_gearbox",
    "step_requested_gear",
]


class GearRequest(IntEnum):
    """What the driver asked for this step. The only thing that moves the gearbox.

    **A request is a command, not a level.** There is no latch: a request that arrives while a shift
    is running is dropped, because C9.8.3 allows one change at a time and the driver has to ask
    again afterwards. :attr:`HOLD` is what "asked for nothing" looks like, and it is the default, so
    a caller that never thinks about gear requests drives a gearbox that holds its gear.

    :attr:`UP` and :attr:`DOWN` are the two paddles, and they move one *forward* gear. Neither is a
    way into neutral or reverse: those are gear selections rather than steps along the box, so they
    are separate requests. :attr:`UP` from neutral or reverse re-enters at first, which is what
    makes
    every request meaningful from every state.

    The members are :class:`~enum.IntEnum` rather than strings so the compiled step can take the
    integer code, and so a channel reader that already holds an integer does not have to convert.
    """

    HOLD = 0
    UP = 1
    DOWN = 2
    NEUTRAL = 3
    REVERSE = 4


# The codes the compiled step compares against. Module-level integers rather than the enum members
# because the compiled function takes a plain `int64`, and deriving them from the enum means the two
# cannot drift: a renumbered member would silently change what the kernel acts on.
_REQUEST_HOLD: Final[int] = int(GearRequest.HOLD)
_REQUEST_UP: Final[int] = int(GearRequest.UP)
_REQUEST_DOWN: Final[int] = int(GearRequest.DOWN)
_REQUEST_NEUTRAL: Final[int] = int(GearRequest.NEUTRAL)
_REQUEST_REVERSE: Final[int] = int(GearRequest.REVERSE)

# The caller owns all three slots. The gear and the shift timer are advanced by the step; the clutch
# engagement is supplied by the caller and read, never written - see the module docstring for why a
# read-only input slot is still caller-owned rather than simulation state.
STATE_SIZE: Final[int] = 3
GEAR_INDEX: Final[int] = 0
SHIFT_TIMER_INDEX: Final[int] = 1
CLUTCH_INDEX: Final[int] = 2

# The two gears outside the forward box, in the domain `channels.yaml` publishes. They are states
# rather than errors: C9.7 (page 104) requires the car to be drivable in reverse by the driver
# at any
# time, and neutral is an existing model state. Neither of them indexes `gear_ratios`.
REVERSE_GEAR: Final[float] = -1.0
NEUTRAL_GEAR: Final[float] = 0.0
_FIRST_GEAR: Final[float] = 1.0

# The configuration scalars `step_gearbox` reads and the sign each one has to have, read as data
# rather than as a list of names because a name that is not checked here is a name nobody notices is
# unchecked. `shift_time_s` is absent because it is not merely a sign rule: a zero shift time
# removes
# the freeze that stops a held request from ratcheting the box a gear per step, so `step_gearbox`
# requires it to be strictly positive and checks it beside the rest. `shift_up_rpm` and
# `shift_down_rpm` are here as a pair only - they are the driver's shift-point schedule,
# validated so
# that a replaced config still has to describe a coherent schedule, and read by no line of this
# module.
_POSITIVE_CONFIG_SCALARS: Final[tuple[str, ...]] = (
    "clutch_demand_torque_nm",
    "final_drive",
    "reverse_ratio",
    "shift_down_rpm",
    "shift_time_s",
    "shift_up_rpm",
)


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def step_requested_gear(gear: float, request: int, gear_count: float) -> float:
    """The gear this step settles on, given where the box is and what was asked for.

    **One request, at most one gear, and only ever a forward neighbour.** ``UP`` adds one and stops
    at ``gear_count``; ``DOWN`` subtracts one and stops at first. Those stops are not defensive
    programming: they are what a one-gear-per-request rule looks like at the ends, and a box that
    counted past them would be indexing a ratio that is not there. A down request from first gear is
    dropped rather than stepping into neutral or reverse, because those are selections a driver
    makes
    deliberately and this slice models the forward box as adjacent-only.

    **Neutral and reverse are absolute.** Either request names a state rather than a step, so it
    applies from anywhere - C9.7 wants reverse available at any time - and re-requesting the state
    the box is already in returns the same gear, which is what stops a held request from arming a
    shift on every step. ``UP`` from either of them re-enters the forward box at first, so no
    request
    is ever a dead end.

    The gear is returned rather than written, so the caller decides whether the step that owns the
    buffer accepts a change made while a shift is already running.
    """
    if request == _REQUEST_HOLD:
        return gear
    if request == _REQUEST_UP:
        if gear < _FIRST_GEAR:
            return _FIRST_GEAR
        if gear < gear_count:
            return gear + 1.0
        return gear
    if request == _REQUEST_DOWN:
        if gear > _FIRST_GEAR:
            return gear - 1.0
        return gear
    if request == _REQUEST_NEUTRAL:
        return NEUTRAL_GEAR
    if request == _REQUEST_REVERSE:
        return REVERSE_GEAR
    return gear


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def clutch_demand_nm(engagement: float, demand_torque_nm: float, travel_fraction: float) -> float:
    """C9.2.5's mapping from a clutch paddle position to a torque demand at the rear axle.

    **The gain is applied to the travel, not to the whole stroke.** The regulation expresses the
    driver's demand as rear-axle torque "by applying a gain of 5200Nm / 90%" (page 101), and the
    90 %
    is 90 % of the engagement travel - so the demand rises from nothing to the full gain across the
    middle of the stroke, with a dead band around a disengaged clutch at each end.

    **The band is derived, not written down.** The two ends are ``(1 - travel_fraction) / 2`` and
    that
    plus ``travel_fraction``: the configured span, centred on the middle of the stroke. With the
    committed 90 % that is 5 % and 95 %, which are the figures C9.2.5's own sentence describes, but
    nothing here has to be told them separately - a file edit that changes the travel moves both
    ends
    together and there is no second copy of the number to be wrong.

    **Fully disengaged asks for nothing**, which is C9.2.3's "incapable of transmitting any useable
    torque" as a demand rather than as a capacity, and it is why the step below returns exactly
    ``0.0`` rather than a fraction of the gain. Full travel asks for the whole gain.

    Monotone and inside ``[0, demand_torque_nm]`` for an engagement in ``[0, 1]``, which is what
    makes
    it usable as the ceiling a launch runs into.
    """
    low = (1.0 - travel_fraction) / 2.0
    if engagement <= low:
        return 0.0
    high = low + travel_fraction
    if engagement >= high:
        return demand_torque_nm
    return demand_torque_nm * (engagement - low) / travel_fraction


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def clutch_output_torque_nm(gearbox_torque_nm: float, demand_nm: float) -> float:
    """``min(what the gear train offers, what the clutch is asked for)``, with the sign of the
    offer.

    **The clutch cannot transmit more than the gearbox has.** A demand is a request from the driver,
    not a supply from the engine, so past the point where the engine can hold the demand the plates
    slip and what arrives is the engine's own torque reduced by the gear - which is a launch, and
    the case C9.2.5 exempts from its tracking band for the first 85 ms of a launch step. Reversing
    the order, and scaling the engine torque by the engagement instead, would be a torque limiter
    the driver cannot ask for and C9.1.2 forbids.

    **The sign follows the gear, not the demand.** A rear-axle demand is a magnitude - the
    engagement says how hard, and the gear says which way - so reverse transmits negative torque
    at a
    positive demand and a neutral transmits nothing at all. Carrying the sign this way is what keeps
    the magnitude comparison free of an absolute value on both operands.
    """
    ceiling = gearbox_torque_nm
    if ceiling < 0.0:
        ceiling = -ceiling
    if ceiling > demand_nm:
        ceiling = demand_nm
    if gearbox_torque_nm < 0.0:
        return -ceiling
    return ceiling


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def shift_remaining_after(shift_remaining_s: float, dt_s: float) -> float:
    """The shift countdown one step later, landing on exactly ``0.0`` rather than below it.

    A deadline would let the timer go negative and read as "not shifting" for every later step while
    still being a negative duration if anything ever added to it, so the last step clamps. The clamp
    is what makes the cut end at a step boundary rather than partway through one.
    """
    remaining = shift_remaining_s - dt_s
    if remaining < 0.0:
        return 0.0
    return remaining


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def _step_gearbox(
    state: np.ndarray,
    engine_torque_nm: float,
    dt_s: float,
    gear_ratios: np.ndarray,
    reverse_ratio: float,
    final_drive: float,
    shift_time_s: float,
    clutch_demand_torque_nm: float,
    clutch_demand_travel_fraction: float,
    request: int,
) -> float:
    """One step of the gearbox, writing ``state`` in place and returning the driveline torque.

    Nothing here is checked - see the module docstring - so it must only ever be reached through
    :func:`step_gearbox`, which is the only thing that validates the buffer and the numbers in it.

    The order of the four steps is the model. The timer is integrated first, so a shift armed by the
    previous step is already running here. The gear is chosen second from the request alone, and a
    request that arrives while the shift is running is *dropped* - the gear stays where it is rather
    than moving at the end of the cut, which is C9.8.3's one change at a time. Neutral then returns
    before any ratio is touched, because it has none: the forward table is ``1..8`` and indexing it
    with 0 would be exactly the bug the gear domain exists to prevent. The clutch is forced open
    while a shift runs, which is the boost cut, and the demand and the transmitted torque follow
    from
    the engagement rather than from a constant.

    ``state[CLUTCH_INDEX]`` is read and not written. The boost cut below assigns to a local
    ``engagement``, never to the slot, so the caller's value survives the shift untouched and it is
    the caller's to change between steps. Writing zero back would be a different model - one where
    the
    step owns the pedal - and would silently discard the caller's launch or trailing-throttle input.
    """
    shifting = state[SHIFT_TIMER_INDEX] > 0.0
    state[SHIFT_TIMER_INDEX] = shift_remaining_after(state[SHIFT_TIMER_INDEX], dt_s)

    gear = state[GEAR_INDEX]
    requested = step_requested_gear(gear, request, float(gear_ratios.size))
    if shifting:
        # Dropped, not deferred: C9.8.3 allows one change at a time, and after the shift ends the
        # driver has to ask again. Queuing it would move the box without a fresh request.
        requested = gear
    elif requested != gear:
        # Apply the requested gear immediately and keep torque cut until the timer expires.
        shifting = True
        state[SHIFT_TIMER_INDEX] = shift_time_s
    state[GEAR_INDEX] = requested

    if requested == NEUTRAL_GEAR:
        return 0.0

    # Read from state and assigned to a local. The boost cut is this step's decision about what to
    # transmit, so the caller's engagement has to survive it untouched and stay theirs to change
    # between steps; writing zero back would silently discard a launch or trailing-throttle input.
    engagement = state[CLUTCH_INDEX]
    if shifting:
        engagement = 0.0
    demand_nm = clutch_demand_nm(engagement, clutch_demand_torque_nm, clutch_demand_travel_fraction)

    # Ratio first, demand second: `PLAN.md` section 6 draws `torque_curve -> gearbox -> clutch ->
    # differential -> wheels`, so the clutch sees the torque after the reduction and C9.2.5's demand
    # is a rear-axle figure. `engine_torque_nm` arrives here already assembled at the crankshaft -
    # the ICE's contribution through its own throttle, plus the MGU-K's - which is what puts the
    # motor upstream of this gear rather than beside the clutch.
    #
    # The 1-based gear indexes a 0-based table, which is the one indexing rule in this module. It is
    # inlined rather than factored out because a public helper would be callable with an arbitrary
    # gear: compiled with `boundscheck=False`, it would read past the end of a short table
    # and return
    # a plausible ratio rather than raising. `step_gearbox` validates the table and the gear
    # together,
    # and that pairing is the only safe way in.
    if requested >= _FIRST_GEAR:
        ratio = gear_ratios[int(requested) - 1] * final_drive
    else:
        # Reverse negates its own synthetic magnitude rather than borrowing the first or the last
        # forward ratio, so the reverse layout does not depend on how many ratios the box has.
        ratio = -reverse_ratio * final_drive
    gearbox_torque_nm = engine_torque_nm * ratio
    return clutch_output_torque_nm(gearbox_torque_nm, demand_nm)


def initial_state(gear: float = 1.0, clutch_engagement: float = 1.0) -> np.ndarray:
    """Caller-owned float64 state buffer, ``[gear, shift_remaining_s, clutch_engagement]``.

    **All three slots are the caller's, and they are not the same kind of thing.** The gear and the
    shift timer are advanced by :func:`step_gearbox`; the clutch engagement is supplied by the
    caller,
    read on every step and **never written**, so a caller may change it between steps to model a
    launch or a trailing-throttle lift. That asymmetry is the contract, not an oversight - see the
    module docstring.

    ``gear`` accepts the whole telemetry domain ``{-1, 0, 1..n}``: reverse and neutral are states
    the
    driver selects (C9.7, page 104), not errors. The launch default is first gear, no shift in
    progress, fully closed clutch, which is the state a standing start is in.

    Nothing is checked here, exactly as in
    :func:`~f1telemetry.kernels.longitudinal.initial_state`: this only manufactures a buffer, and
    :func:`step_gearbox` is the boundary that decides whether the numbers in it are usable. A gear
    outside the domain, or an engagement outside ``[0, 1]``, therefore fails on the first step
    rather
    than at construction.
    """
    return np.array([gear, 0.0, clutch_engagement], dtype=np.float64)


def step_gearbox(
    config: KernelConfig,
    state: np.ndarray,
    ice_rpm: float,
    throttle: float,
    request: GearRequest | int = GearRequest.HOLD,
    *,
    mgu_k_torque_nm: float = 0.0,
) -> float:
    """The torque the driveline receives this step, in newton-metres at the differential.

    The Python-facing entry point, and the only one: it reads the configuration off
    :class:`KernelConfig`, checks the caller's state buffer and every number it hands over, and
    calls
    the compiled step with flat scalars. The engine torque comes from
    :func:`~f1telemetry.physics.powertrain.step_ice_torque`, so the ICE curve, the turbo lag and
    the fuel-energy-flow limit are validated by the composition that already owns them rather than
    by a second copy of its rules.

    **The gear request is the only thing that moves the gearbox.** ``request`` is one of
    :class:`GearRequest` - or the integer code behind one - and defaults to
    :attr:`GearRequest.HOLD`, so a caller that never asks for a change drives a gearbox that holds
    its gear no matter what the engine speed does. C9.8.1 (page 104) is why: an automatic gear
    change
    is a driver aid and is not permitted. ``ice_rpm`` still arrives here, but only because it
    selects
    a point on the ICE curve; it decides nothing about the gear.

    **The caller owns all three state slots, but the step only advances two of them.** ``state`` is
    ``[gear, shift_remaining_s, clutch_engagement]`` and is written in place, so a run steps one
    buffer over and over with no allocation. The gear and the shift timer are *advanced* here. The
    clutch engagement is *supplied* by the caller and read, never written, so the caller may
    change it
    between steps to model a launch or a trailing-throttle lift. A shift's boost cut therefore
    overrides a *local* copy: the returned torque is ``0.0`` through a shift while
    ``state[CLUTCH_INDEX]`` still holds the engagement the caller set. The caller reads the gear out
    with :data:`GEAR_INDEX`.

    **The returned torque is differential-side, not engine-side.** It is
    ``(throttle * engine torque + mgu_k_torque_nm) * ratio * final_drive`` limited by the clutch
    demand the engagement asks for, which is what ``PHASES.md`` P1-T5 means by naming eight ratios
    and a final drive and what ``PLAN.md`` section 6 draws as ``torque_curve -> gearbox(8-speed)
    -> clutch -> differential -> wheels``. The ratios are not decoration and ``final_drive`` is not
    configuration no model reads. It stops at a torque because that is still what this is: P1-T7 is
    what divides by the rolling radius to get ``Fx``, and P1-T6 is what owns the wheel speed the
    caller uses to arrive at ``ice_rpm``.

    **``mgu_k_torque_nm`` is added, not geared separately**, because C5.18.2 fixes the MGU-K to the
    crankshaft: whatever torque survives
    :func:`~f1telemetry.physics.powertrain.step_mgu_k` is already at the crankshaft and takes the
    same reduction as the engine's. It defaults to zero so a caller with no hybrid in the picture
    never has to name it, and it is *not* scaled by ``throttle`` - the motor has its own actuator
    and its own request. A caller that wants the MGU-K's own limits applied should call
    :func:`step_drivetrain`, which is the composition that applies them; this parameter exists for
    the caller that has already done so.

    **``shift_time_s`` must be positive.** A zero shift time is refused rather than supported,
    because the timer is the only thing that freezes the gear while a shift runs. With it at zero a
    held request would advance the box a gear per step, and the one-at-a-time property the timer
    exists to provide would be gone.

    ``throttle`` is a pedal position in ``[0, 1]``, ``state[CLUTCH_INDEX]`` is another in the same
    units as ``throttle_pct`` and ``clutch_pct`` in ``channels.yaml`` divided by 100 - the
    percentages ``GroundTruthStep`` publishes are the scaled forms of those two numbers - and
    ``request`` is a driver command rather than a quantity, so it is checked against the codes this
    model knows how to act on rather than against a range.

    ``KernelConfig`` is public and replaceable, so the values are checked here rather than assumed
    to
    have been checked by whoever built it. The compiled step is ``boundscheck=False`` and holds no
    finiteness checks, so an unusable buffer writes out of bounds, a gear outside the ratio table
    indexes past its end, and a NaN comes back as a NaN torque that reads as a finished run. Scalars
    are narrowed to ``float`` and the request to ``int`` so equivalent ``int`` and ``float`` callers
    share one compiled specialisation.
    """
    _check_ratios(config)
    _check_state(state, float(config.gear_ratios.size))
    ice_rpm = _checked_float("ice_rpm", ice_rpm)
    throttle = _checked_pedal("throttle", throttle)
    mgu_k_torque = _checked_float("mgu_k_torque_nm", mgu_k_torque_nm)
    code = _checked_request(request)
    dt_s = _checked_float("config.dt_s", config.dt_s)
    if dt_s <= 0.0:
        raise ValueError(
            f"step_gearbox: config.dt_s must be finite and > 0, got {dt_s!r}. The shift timer is "
            "integrated over it, so a zero step would never finish a shift"
        )
    shift_time = _checked_float("config.shift_time_s", config.shift_time_s)
    if shift_time <= 0.0:
        raise ValueError(
            f"step_gearbox: config.shift_time_s must be finite and > 0, got {shift_time!r}. The "
            "timer is what freezes the gear while a shift runs, so a zero shift time removes the "
            "freeze and a held request walks the box a gear per step"
        )
    values = {
        name: _checked_float(f"config.{name}", getattr(config, name))
        for name in _POSITIVE_CONFIG_SCALARS
    }
    for name, value in values.items():
        if value <= 0.0:
            raise ValueError(f"step_gearbox: config.{name} must be finite and > 0, got {value!r}")
    travel = _checked_travel_fraction(config.clutch_demand_travel_fraction)
    shift_up = values["shift_up_rpm"]
    shift_down = values["shift_down_rpm"]
    if shift_down >= shift_up:
        raise ValueError(
            f"step_gearbox: config.shift_down_rpm must be below config.shift_up_rpm, got "
            f"{shift_down!r} and {shift_up!r}. They are the driver's shift-point schedule, so "
            "overlapping points leave a single rpm that could satisfy both"
        )

    # The throttle goes in here rather than into `_step_gearbox`, because the ICE's fuel-energy-flow
    # limit is a function of the power the engine is actually making and its arm rises with power.
    engine_torque_nm = step_ice_torque(config, ice_rpm, throttle)
    return float(
        _step_gearbox(
            state,
            engine_torque_nm + mgu_k_torque,
            dt_s,
            config.gear_ratios,
            values["reverse_ratio"],
            values["final_drive"],
            shift_time,
            values["clutch_demand_torque_nm"],
            travel,
            code,
        )
    )


def step_drivetrain(
    config: KernelConfig,
    state: np.ndarray,
    mgu_k_state: np.ndarray,
    ice_rpm: float,
    throttle: float,
    speed_m_s: float,
    mgu_k_request_nm: float = 0.0,
    *,
    request: GearRequest | int = GearRequest.HOLD,
    overtake: bool = False,
    grid_standing_start: bool = False,
    ecu_mandates_minimum_acceleration: bool = False,
    recharge_event: RechargeEvent | int = RechargeEvent.RACE,
    recharge_allowance_applies: bool = False,
) -> float:
    """Capped MGU-K torque joined at the crankshaft, then one gearbox step - differential-side.

    **This is the composition the drivetrain actually runs.** It exists because the two halves have
    to happen in that order and for a reason a caller should not have to remember: C5.18.2 fixes
    the MGU-K to the crankshaft, so its torque is *upstream* of the gear reduction and has to be
    limited before it is added, or a capped-at-the-axle motor would arrive at the axle already
    multiplied by the first gear. Calling :func:`step_mgu_k` and handing its result to
    :func:`step_gearbox` as ``mgu_k_torque_nm`` does the same thing in two calls; this does it in
    one and makes the ordering the only available one.

    Two caller-owned buffers are advanced in place - the gearbox's ``[gear, shift_remaining_s,
    clutch_engagement]`` and the MGU-K's ``[state_of_charge_mj, lap_recharge_mj]`` - so a run steps
    both buffers over and over with no allocation and no object per step.

    The MGU-K keyword arguments are exactly
    :func:`~f1telemetry.physics.powertrain.step_mgu_k`'s and mean exactly what they mean there:
    they are five declarations about the event the run is at, not tuning. ``mgu_k_request_nm``
    defaults to zero, so a car with an empty store and no deployment strategy still produces a
    driveline torque and still advances the gearbox identically.
    """
    mgu_k_torque_nm = step_mgu_k(
        config,
        mgu_k_state,
        mgu_k_request_nm,
        ice_rpm,
        speed_m_s,
        overtake=overtake,
        grid_standing_start=grid_standing_start,
        ecu_mandates_minimum_acceleration=ecu_mandates_minimum_acceleration,
        recharge_event=recharge_event,
        recharge_allowance_applies=recharge_allowance_applies,
    )
    return step_gearbox(config, state, ice_rpm, throttle, request, mgu_k_torque_nm=mgu_k_torque_nm)


def _checked_float(label: str, value: object) -> float:
    """Return a finite numeric scalar as float, or fail before it reaches Numba."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"step_gearbox: {label} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"step_gearbox: {label} must be finite, got {value!r}")
    return number


def _checked_pedal(label: str, value: object) -> float:
    """A pedal position: a finite number in ``[0, 1]``."""
    number = _checked_float(label, value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"step_gearbox: {label} must be in [0, 1], got {number!r}")
    return number


def _checked_travel_fraction(value: object) -> float:
    """``clutch_demand_travel_fraction``: the share of the paddle's stroke the C9.2.5 gain spans.

    A fraction, so at most ``1``, and strictly positive because the demand divides by it - a zero
    would make the mapping undefined rather than merely unhelpful. The upper bound is what stops a
    config asking for more travel than the pedal has, which would put the gain's far end outside the
    stroke entirely.
    """
    number = _checked_float("config.clutch_demand_travel_fraction", value)
    if not 0.0 < number <= 1.0:
        raise ValueError(
            f"step_gearbox: config.clutch_demand_travel_fraction must be in (0, 1], got {number!r}"
        )
    return number


def _checked_request(value: object) -> int:
    """The gear request as the integer code the compiled step acts on.

    Refused rather than coerced, and the reason is what an unknown code would do. Every comparison
    in
    :func:`step_requested_gear` would fail, the function would fall through to its last return,
    and a
    typo in a scenario would read as a *reverse* selection rather than as an error - a wrong model
    rather than no model. A float is refused for the same reason as a pedal out of range: it would
    compile a second Numba specialisation beside the integer one.

    ``bool`` is refused even though it is an ``int``: ``True`` is ``GearRequest.UP`` by value, and a
    driver command arriving as a truth value is a bug rather than a request.
    """
    if isinstance(value, bool) or not isinstance(value, (GearRequest, int)):
        raise ValueError(
            f"step_gearbox: request must be a GearRequest or its integer code, got {value!r}"
        )
    code = int(value)
    if code not in _REQUEST_CODES:
        raise ValueError(
            f"step_gearbox: request must be one of "
            f"{', '.join(member.name for member in GearRequest)}, got {code!r}"
        )
    return code


_REQUEST_CODES: Final[frozenset[int]] = frozenset(int(member) for member in GearRequest)


def _check_ratios(config: KernelConfig) -> None:
    """The ratio table, which this step indexes - so the checks the loader's do not cover.

    The compiled step reads ``gear_ratios[gear - 1]`` with ``boundscheck=False``, so the rules that
    protect that read are here: a non-empty contiguous float64 vector, and finite. A gear outside it
    would read past the end and return a plausible ratio rather than failing, and a strided or
    ``float32`` buffer would compile as a different kernel. The indexing is deliberately inlined in
    :func:`_step_gearbox` rather than factored into a helper for the same reason - a public callable
    would take an arbitrary gear and none of these checks with it.

    Positivity is re-checked even though ``car_spec.yaml`` states it: the ratios multiply the
    transmitted torque directly, so a zero ratio would silently return zero driveline torque and a
    negative one would drive the car forwards in reverse, and ``KernelConfig`` is replaceable.
    Monotonicity is not: strictly decreasing is the loader's rule, nothing here depends on it, and a
    second place for the same rule to be wrong helps nobody.
    """
    ratios = config.gear_ratios
    if (
        not isinstance(ratios, np.ndarray)
        or ratios.dtype != np.float64
        or ratios.ndim != 1
        or ratios.size == 0
        or not ratios.flags.c_contiguous
    ):
        raise ValueError(
            "step_gearbox: config.gear_ratios must be a non-empty C-contiguous float64 vector"
        )
    if not np.isfinite(ratios).all():
        raise ValueError("step_gearbox: config.gear_ratios must be finite")
    if not np.all(ratios > 0.0):
        raise ValueError("step_gearbox: config.gear_ratios must be positive")


def _check_state(state: np.ndarray, gear_count: float) -> None:
    """The caller's buffer, before a ``boundscheck=False`` step writes it and indexes with it.

    Two kinds of rule, and both are needed. The buffer has to be a writable C-contiguous float64
    vector of exactly :data:`STATE_SIZE` entries, because the compiled step writes two of them and a
    short or strided buffer is an out-of-bounds write or a differently-typed kernel. The *values*
    have to be usable, because ``boundscheck=False`` says nothing about arithmetic and the gear is
    about to index the ratio table with: a gear outside ``-1..gear_count``, or a fractional one,
    would read the wrong entry of the table rather than failing.

    **The domain is the one the telemetry contract publishes, ``{-1, 0, 1..n}``.** Reverse and
    neutral are inside it - C9.7 requires the car to be drivable in reverse and the channels already
    carry both states - so this checks the two *ends* of the domain rather than refusing zero and
    negative gears. Checking the gear against ``gear_count`` is what makes the index safe, so this
    and :func:`_check_ratios` are two halves of one rule.
    """
    if (
        not isinstance(state, np.ndarray)
        or state.dtype != np.float64
        or state.shape != (STATE_SIZE,)
        or not state.flags.c_contiguous
    ):
        raise ValueError(
            f"step_gearbox: state must be a C-contiguous float64 vector of length {STATE_SIZE} "
            f"(gear, shift timer), got {type(state).__name__} "
            f"{getattr(state, 'shape', None)} of {getattr(state, 'dtype', None)}"
        )
    if not state.flags.writeable:
        raise ValueError(
            "step_gearbox: state is read-only, and it is where this step writes the gear and the "
            "shift timer. A read-only buffer is an out-of-bounds write"
        )
    gear = float(state[GEAR_INDEX])
    if (
        not math.isfinite(gear)
        or not REVERSE_GEAR <= gear <= gear_count
        or gear != math.floor(gear)
    ):
        raise ValueError(
            f"step_gearbox: state gear must be in "
            f"{int(REVERSE_GEAR)}..{int(gear_count)} and be a whole number"
            f" - reverse, neutral or a forward gear - got {gear!r}. The forward part is the domain "
            "the channels publish and the gear indexes the ratio table"
        )
    _checked_pedal("state clutch engagement", float(state[CLUTCH_INDEX]))
    remaining = _checked_float("state shift timer", float(state[SHIFT_TIMER_INDEX]))
    if remaining < 0.0:
        raise ValueError(
            f"step_gearbox: state shift timer must be >= 0, got {remaining!r}. A shift is a "
            "duration, not a deadline"
        )
