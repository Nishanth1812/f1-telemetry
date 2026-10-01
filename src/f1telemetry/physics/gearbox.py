"""P1-T5: the gearbox - gear selection, the shift timer, and the clutch between the two.

``PHASES.md`` P1-T5 asks for eight ratios, a final drive, shift logic and clutch state, with
trailing throttle and the boost cut both routed through the clutch. The ratios, the final drive,
both shift points, the shift duration and the clutch capacity are all ``car_spec.yaml`` data
reached through :class:`~f1telemetry.contracts.car_spec.KernelConfig`, so what lives here is the
decisions that turn those numbers into a torque at the differential.

**The ratios and the final drive are used, not merely counted.** The transmitted torque is
``throttle * engine torque * gear_ratios[gear - 1] * final_drive``, which is what ``PHASES.md``
P1-T5 means by naming ratios and a final drive and what ``PLAN.md`` section 6 draws as
``torque_curve -> gearbox(8-speed) -> clutch -> differential -> wheels``. A gearbox that selected a
gear and then returned the same engine-side torque for all eight would make both configured
quantities decorative, and ``final_drive`` would be configuration no model reads. The result is
still a **torque**, so it stops here: P1-T7 is what divides by the rolling radius to get ``Fx``.

**The gear is chosen from engine speed, and it cannot go backwards.** :func:`step_gear` compares
``ice_rpm`` against the configured upshift and downshift points and moves by exactly one, so a
threshold crossed once moves the box one gear rather than walking it. There is no reverse and no
neutral in this model: :func:`step_gearbox` refuses a gear outside ``1..n_gears``, which is what
makes ``PLAN.md`` section 11's invariant 7 - monotonic progression, no reverse under positive
throttle - a property of the model rather than of a fixture that happens to be monotone.

**The shift timer is state, and it freezes the gear.** :func:`shift_remaining_after` is a countdown
the caller integrates over its own ``dt_s``; nothing here reads a clock, so a shift is over when
the timer reaches zero and a run is as reproducible as its inputs. While the timer runs the gear is
held, which is the half of the boost cut that makes it a *shift* rather than a gap: without it, the
rpm that triggered the upshift would trigger the next one on the step the timer expired and the
torque would never be delivered at all.

**The boost cut is the clutch, and there is only one path.** During a shift the engagement is
forced to zero, so the transmitted torque is exactly ``0.0`` - not reduced, and not scaled by a
leftover engagement. Trailing throttle goes through the same path from the other end: the engine
makes a fraction of its torque and, with the clutch closed, that fraction is what arrives. Two
branches here would be two places for ``PHASES.md``'s claim to be true of one of them.
**The clutch capacity is what a launch runs into, and it is differential-side.**
:func:`clutch_output_torque_nm` is ``min(reduced torque, capacity * engagement)``, so the
capacity is a ceiling that a partly engaged clutch reaches and a fully closed one does not on the
committed data. That is the launch: the box is asked for its whole torque and the driveline still
only gets what the clutch can hold at that pedal position.

**The clamp is downstream of the ratio, which is what fixes the units.** ``PLAN.md`` section 6 puts
the clutch *after* the gearbox, so the capacity is compared against the already-reduced torque and
is a differential-side figure. Clamping the engine torque first and multiplying by the ratio
afterwards is the same formula in the wrong order, and on this data it does not inflate the
torque - it makes the clamp **dead**: ``min(330, 600) * 11.649`` is still 3 844 Nm, because the
engine peak is below the capacity and the minimum never selects the capacity. A capacity that can
never be selected is configuration no model reads, which is the failure the ordering fixes.

**Why the committed 3 000 Nm.** It sits inside the range the committed box produces rather than
above it, so the capacity is reachable in the gears where launch traction limiting belongs and
invisible above them. At the 330 Nm engine peak the box offers 3 844 Nm in first and 1 618 Nm in
eighth, so 3 000 Nm clamps **gears 1-3 at full engagement** and passes 4-8 through unmodulated.
That is the shape the model wants: the low gears - the ones a launch happens in - are
clutch-limited, and the ratios stay observable in the top half. A capacity above 3 844 Nm would be
a ceiling no gear reaches; one below 1 618 Nm would flatten every gear to the same number.

**Caller-owned state, and its three slots are not the same kind of thing.** The gear and the shift
timer are *advanced* by this step. Clutch engagement is *supplied* by the caller and read, never
written: it is a pedal position the caller holds, exactly like the throttle, and a launch or a
trailing-throttle lift is the caller changing it between steps. That is why the boost cut overrides
a *local* copy and leaves the slot alone - the cut is this step's decision about what to transmit,
not a change to where the driver has the pedal, and writing zero back would silently discard the
caller's input. Modelling how the pedal moves over time would need a clutch ramp rate and a closing
curve; no such dynamics is asked for here, and inventing one would put numbers in Python that no
file supplies.
**Engine speed is an input here, not state.** Turning ``ice_rpm`` into a wheel speed is P1-T6's
wheel rotational state - deriving it here would need the gear this step has not chosen yet - and
turning the differential-side torque into ``Fx`` is P1-T7's force assembly. This step is therefore
the last part of the drive path that is a pure function of numbers the caller hands over.

**Two ways in, as in :mod:`f1telemetry.physics.forces` and
:mod:`f1telemetry.physics.powertrain`.** :func:`step_gearbox` is the Python-facing composition and
the only validating entry: it reads the configuration, checks it and the caller's state buffer, and
hands the compiled step nothing but numbers. :func:`_step_gearbox` is compiled with
``boundscheck=False`` and is private for the reason ``kernels.longitudinal._integrate`` is - it
*writes* into a caller-owned buffer and indexes the ratio table with it, so there has to be exactly
one way in and that way validates. The same is why the ratio lookup is inlined there rather than
offered as a helper: an exported primitive taking a gear and a table would be callable with either
one unchecked.

The engine torque is not recomputed here. P1-T5 has no new torque model, so the step reads the
curve through :func:`~f1telemetry.physics.powertrain.step_ice_torque`, which is where the curve and
the turbo lag are already validated; a second copy of that lookup would be a second place for the
engine's torque to be wrong.

There is no MGU-K contribution to the transmitted torque in this task. The motor's limit and its
deployment curves are in ``car_spec.yaml`` and in :class:`KernelConfig`, and adding a deployment
arithmetic here would be P1-T7's decision about which wheels receive it.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import numpy as np
from numba import njit

from f1telemetry.physics.powertrain import step_ice_torque  # noqa: TID251 -- same-layer composition

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import KernelConfig

__all__ = [
    "CLUTCH_INDEX",
    "GEAR_INDEX",
    "SHIFT_TIMER_INDEX",
    "STATE_SIZE",
    "clutch_output_torque_nm",
    "initial_state",
    "shift_remaining_after",
    "step_gear",
    "step_gearbox",
]

# The caller owns all three slots. The gear and the shift timer are advanced by the step; the clutch
# engagement is supplied by the caller and read, never written - see the module docstring for why a
# read-only input slot is still caller-owned rather than simulation state.
STATE_SIZE: Final[int] = 3
GEAR_INDEX: Final[int] = 0
SHIFT_TIMER_INDEX: Final[int] = 1
CLUTCH_INDEX: Final[int] = 2

# The configuration scalars `step_gearbox` reads and the sign each one has to have, read as data
# rather than as a list of names because a name that is not checked here is a name nobody notices
# is unchecked. `shift_time_s` is absent because it is not merely a sign rule: a zero shift time
# removes the freeze that stops a held rpm threshold from ratcheting the box a gear per step, so
# `step_gearbox` requires it to be strictly positive and checks it beside the rest.
_POSITIVE_CONFIG_SCALARS: Final[tuple[str, ...]] = (
    "clutch_torque_capacity_nm",
    "final_drive",
    "shift_down_rpm",
    "shift_time_s",
    "shift_up_rpm",
)


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def step_gear(
    gear: float,
    ice_rpm: float,
    gear_count: float,
    shift_up_rpm: float,
    shift_down_rpm: float,
) -> float:
    """The gear for this step: one up at the upshift point, one down at the downshift point.

    The two comparisons are ``>=`` and ``<=``, so each threshold means what it says, and both are
    clamped to the box: ``gear_count`` stops the upshift at top gear and ``1`` stops the downshift
    at first. Neither clamp is defensive programming - they are what makes a progression terminate,
    and a gearbox that counted past the end of the ratios would be indexing a table that is not
    there.

    **Moving one gear, never several.** The caller freezes the gear while a shift is in progress,
    so a threshold is sampled once per shift rather than once per step, and an rpm that sits on the
    upshift point cannot walk the box a gear per step.
    """
    if gear < gear_count and ice_rpm >= shift_up_rpm:
        return gear + 1.0
    if gear > 1.0 and ice_rpm <= shift_down_rpm:
        return gear - 1.0
    return gear


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def clutch_output_torque_nm(
    gearbox_torque_nm: float,
    clutch_capacity_nm: float,
    engagement: float,
) -> float:
    """``min(gearbox torque, capacity * engagement)``: what the clutch passes to the differential.

    **The capacity is scaled by the engagement before the minimum, not after it.** A clutch that is
    partly closed cannot carry its full capacity - the plates are only partly pressed together - so
    the ceiling a partly engaged clutch presents falls with the engagement. Scaling after the
    minimum would instead make the capacity a constant multiplier that a slipping clutch cannot
    reach, and on the committed curve that is exactly the dead branch Task 3's grip-limit ruling
    refused to write.

    **This clamp is downstream of the ratio, which is what fixes the units.** ``PLAN.md`` section 6
    draws ``torque_curve -> gearbox(8-speed) -> clutch -> differential -> wheels``, so the clutch
    sees the torque *after* the gear reduction and the capacity is a differential-side figure.
    Clamping the engine torque instead - the order this function had before the P1-T5 boundary
    review - did **not** inflate the output; it made the clamp unreachable. On the committed data
    the engine peak is 330 Nm, so ``min(330, 600)`` is the engine and the minimum never selects
    the capacity, and the result is the same 3 844 Nm the correct order produces at full
    engagement. A capacity that can never be selected is configuration no model reads, which is
    the same failure as the ratio table being decorative.

    With the ceiling inside the minimum, both ends mean something. At full engagement the ceiling is
    the capacity, which the committed 3 000 Nm places above what gears 4-8 produce, so those pass
    the gear train's own torque and the clutch is out of the way; in gears 1-3 it is below what the
    box offers and the clutch is the limit. At lower engagement the ceiling falls with it and every
    gear becomes clutch-limited - which is the launch, where the engine offers its whole torque and
    the driveline still takes only what the plates can hold at that pedal position.

    Never negative for a non-negative gearbox torque and an engagement in ``[0, 1]``, which is what
    keeps the sign of the drive torque the same as the sign of the throttle.
    """
    ceiling = clutch_capacity_nm * engagement
    if gearbox_torque_nm < ceiling:
        return gearbox_torque_nm
    return ceiling


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def shift_remaining_after(shift_remaining_s: float, dt_s: float) -> float:
    """The shift countdown one step later, landing on exactly ``0.0`` rather than below it.

    A deadline would let the timer go negative and read as "not shifting" for every later step
    while still being a negative duration if anything ever added to it, so the last step clamps.
    The clamp is what makes the cut end at a step boundary rather than partway through one.
    """
    remaining = shift_remaining_s - dt_s
    if remaining < 0.0:
        return 0.0
    return remaining


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def _step_gearbox(
    state: np.ndarray,
    ice_rpm: float,
    throttle: float,
    engine_torque_nm: float,
    dt_s: float,
    gear_ratios: np.ndarray,
    final_drive: float,
    shift_up_rpm: float,
    shift_down_rpm: float,
    shift_time_s: float,
    clutch_capacity_nm: float,
) -> float:
    """One step of the gearbox, writing ``state`` in place and returning the driveline torque.

    Nothing here is checked - see the module docstring - so it must only ever be reached through
    :func:`step_gearbox`, which is the only thing that validates the buffer and the numbers in it.

    The order of the five steps is the model. The timer is integrated first, so a shift armed by
    the previous step is already running here; the gear is chosen second, and a change on a step
    that is not already shifting arms the timer *for the step after this one*; the clutch is forced
    open if a shift is in progress, which is the boost cut; the throttle is applied to the engine
    torque and the gear reduction next, because ``PLAN.md`` section 6 puts the clutch downstream of
    the gearbox and the capacity is compared against the reduced torque; and the clutch ceiling is
    applied last. The ratio uses the gear this step settled on rather than the one it started in,
    since the gear changes on this step. The engine torque arrives as a scalar because it was read
    through :func:`~f1telemetry.physics.powertrain.step_ice_torque` on the Python side of the
    boundary.

    **``state[CLUTCH_INDEX]`` is read and not written.** The boost cut below assigns to a local
    ``engagement``, never to the slot, so the caller's value survives the shift untouched and it is
    the caller's to change between steps. Writing zero back would be a different model - one where
    the step owns the pedal - and would silently discard the caller's launch or trailing-throttle
    input.
    """
    shifting = state[SHIFT_TIMER_INDEX] > 0.0
    state[SHIFT_TIMER_INDEX] = shift_remaining_after(state[SHIFT_TIMER_INDEX], dt_s)

    gear = state[GEAR_INDEX]
    next_gear_value = step_gear(
        gear, ice_rpm, float(gear_ratios.size), shift_up_rpm, shift_down_rpm
    )
    if next_gear_value != gear and not shifting:
        # Armed rather than applied: the gear changes on this step, and the cut starts with it.
        shifting = True
        state[SHIFT_TIMER_INDEX] = shift_time_s
    state[GEAR_INDEX] = next_gear_value

    # Read from state and assigned to a local. The boost cut is this step's decision about what to
    # transmit, so the caller's engagement has to survive it untouched and stay theirs to change
    # between steps; writing zero back would silently discard a launch or trailing-throttle input.
    engagement = state[CLUTCH_INDEX]
    if shifting:
        engagement = 0.0
    # Ratio first, clamp second: the clutch is downstream of the gearbox in `PLAN.md` section 6's
    # chain, so its capacity is a differential-side figure and has to be compared against the
    # reduced torque. Clamping first and multiplying afterwards leaves the clamp dead on this data
    # rather than inflating the torque - `min(330, 600) * 11.649` is still 3 844 Nm - so the wrong
    # order silently loses the capacity's effect instead of misreporting it.
    #
    # The 1-based gear indexes a 0-based table, which is the one indexing rule in this module. It is
    # inlined rather than factored out because a public helper would be callable with an arbitrary
    # gear: compiled with `boundscheck=False`, it would read past the end of a short table and
    # return a plausible ratio rather than raising. `step_gearbox` validates the table and the gear
    # together, and that pairing is the only safe way in.
    ratio = gear_ratios[int(next_gear_value) - 1] * final_drive
    gearbox_torque_nm = throttle * engine_torque_nm * ratio
    return clutch_output_torque_nm(gearbox_torque_nm, clutch_capacity_nm, engagement)


def initial_state(gear: float = 1.0, clutch_engagement: float = 1.0) -> np.ndarray:
    """Caller-owned float64 state buffer, ``[gear, shift_remaining_s, clutch_engagement]``.

    **All three slots are the caller's, and they are not the same kind of thing.** The gear and the
    shift timer are advanced by :func:`step_gearbox`; the clutch engagement is supplied by the
    caller, read on every step and **never written**, so a caller may change it between steps to
    model a launch or a trailing-throttle lift. That asymmetry is the contract, not an oversight -
    see the module docstring.

    The launch default is first gear, no shift in progress, fully closed clutch, which is the state
    a standing start is in. Nothing is checked here, exactly as in
    :func:`~f1telemetry.kernels.longitudinal.initial_state`: this only manufactures a buffer, and
    :func:`step_gearbox` is the boundary that decides whether the numbers in it are usable. A gear
    outside the box, or an engagement outside ``[0, 1]``, therefore fails on the first step rather
    than at construction.
    """
    return np.array([gear, 0.0, clutch_engagement], dtype=np.float64)


def step_gearbox(
    config: KernelConfig,
    state: np.ndarray,
    ice_rpm: float,
    throttle: float,
) -> float:
    """The torque the driveline receives this step, in newton-metres at the differential.

    The Python-facing entry point, and the only one: it reads the configuration off
    ``KernelConfig``, checks the caller's state buffer and every number it hands over, and calls the
    compiled step with flat scalars. The engine torque comes from
    :func:`~f1telemetry.physics.powertrain.step_ice_torque`, so the ICE curve and the turbo lag are
    validated by the composition that already owns them rather than by a second copy of its rules.

    **The caller owns all three state slots, but the step only advances two of them.** ``state`` is
    ``[gear, shift_remaining_s, clutch_engagement]`` and is written in place, so a run steps one
    buffer over and over with no allocation. The gear and the shift timer are *advanced* here. The
    clutch engagement is *supplied* by the caller and read, never written, so the caller may change
    it between steps to model a launch or a trailing-throttle lift. The shift's boost cut therefore
    overrides a *local* copy: the returned torque is ``0.0`` through a shift while
    ``state[CLUTCH_INDEX]`` still holds the engagement the caller set. Writing zero back would be a
    different model - one where the step owns the pedal - and would discard the caller's input. The
    caller reads the gear out with :data:`GEAR_INDEX`.

    **The returned torque is differential-side, not engine-side.** It is
    ``throttle * engine torque * gear_ratios[gear - 1] * final_drive`` passed through the clutch
    ceiling, which is what ``PHASES.md`` P1-T5 means by naming eight ratios and a final drive and
    what ``PLAN.md`` section 6 draws as ``torque_curve -> gearbox(8-speed) -> clutch -> differential
    -> wheels``. The ratios are not decoration and ``final_drive`` is not configuration no model
    reads. It stops at a torque because that is still what this is: P1-T7 is what divides by the
    rolling radius to get ``Fx``, and P1-T6 is what owns the wheel speed the caller uses to arrive
    at ``ice_rpm``.

    **``shift_time_s`` must be positive.** A zero shift time is refused rather than supported,
    because the timer is the only thing that freezes the gear while a shift runs. With it at zero an
    rpm sitting exactly on the upshift point would advance the box one gear per step, and the
    no-ratchet property the shift timer exists to provide would be gone.

    ``ice_rpm`` is an input rather than something this step computes: the wheel speed that would
    produce it is P1-T6's state, and deriving it here would need the gear this step has not chosen
    yet. ``throttle`` is a pedal position in ``[0, 1]`` and ``state[CLUTCH_INDEX]`` is another, in
    the same units as ``throttle_pct`` and ``clutch_pct`` in ``channels.yaml`` divided by 100 - the
    percentages ``GroundTruthStep`` publishes are the scaled forms of those two numbers.

    ``KernelConfig`` is public and replaceable, so the values are checked here rather than assumed
    to have been checked by whoever built it. The compiled step is ``boundscheck=False`` and holds
    no finiteness checks, so an unusable buffer writes out of bounds, a gear outside the ratio
    table indexes past its end, and a NaN comes back as a NaN torque that reads as a finished run.
    Scalars are narrowed to ``float`` so equivalent ``int`` and ``float`` callers share one
    compiled specialisation.
    """
    _check_ratios(config)
    _check_state(state, float(config.gear_ratios.size))
    ice_rpm = _checked_float("ice_rpm", ice_rpm)
    throttle = _checked_pedal("throttle", throttle)
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
            "freeze and an rpm sitting on the upshift point walks the box a gear per step"
        )
    values = {
        name: _checked_float(f"config.{name}", getattr(config, name))
        for name in _POSITIVE_CONFIG_SCALARS
    }
    for name, value in values.items():
        if value <= 0.0:
            raise ValueError(f"step_gearbox: config.{name} must be finite and > 0, got {value!r}")
    shift_up = values["shift_up_rpm"]
    shift_down = values["shift_down_rpm"]
    if shift_down >= shift_up:
        raise ValueError(
            f"step_gearbox: config.shift_down_rpm must be below config.shift_up_rpm, got "
            f"{shift_down!r} and {shift_up!r}. Overlapping thresholds let one rpm satisfy both"
        )

    engine_torque_nm = step_ice_torque(config, ice_rpm)
    return float(
        _step_gearbox(
            state,
            ice_rpm,
            throttle,
            engine_torque_nm,
            dt_s,
            config.gear_ratios,
            values["final_drive"],
            shift_up,
            shift_down,
            shift_time,
            values["clutch_torque_capacity_nm"],
        )
    )


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
    negative one would drive the car backwards, and ``KernelConfig`` is replaceable. Monotonicity is
    not: strictly decreasing is the loader's rule, nothing here depends on it, and a second place
    for the same rule to be wrong helps nobody.
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
    vector of exactly :data:`STATE_SIZE` entries, because the compiled step writes both and a short
    or strided buffer is an out-of-bounds write or a differently-typed kernel. The *values* have to
    be usable, because ``boundscheck=False`` says nothing about arithmetic and the gear is about to
    index the ratio table with: a gear of zero is neutral and a negative one is reverse, both of
    which ``PLAN.md`` section 11's invariant 7 has an opinion about and neither of which this model
    can produce from a launch. Checking the gear against ``gear_count`` is what makes that index
    safe, so this and :func:`_check_ratios` are two halves of one rule.
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
    if not math.isfinite(gear) or not 1.0 <= gear <= gear_count or gear != math.floor(gear):
        raise ValueError(
            f"step_gearbox: state gear must be a whole number in 1..{int(gear_count)}, got "
            f"{gear!r}. This gearbox has no neutral and no reverse, and the gear indexes the "
            "ratio table"
        )
    _checked_pedal("state clutch engagement", float(state[CLUTCH_INDEX]))
    remaining = _checked_float("state shift timer", float(state[SHIFT_TIMER_INDEX]))
    if remaining < 0.0:
        raise ValueError(
            f"step_gearbox: state shift timer must be >= 0, got {remaining!r}. A shift is a "
            "duration, not a deadline"
        )
