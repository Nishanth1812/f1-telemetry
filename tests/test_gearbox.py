"""P1-T5 under the 2026 rule set: driver-requested gears and a driver-demanded clutch.

The approved spec (``docs/superpowers/specs/2026-10-01-f1-phase-1-fidelity-design.md``) closes
three gaps in this model, and this file is the arithmetic that closes them:

* **G1 - no automatic gear changes.** C9.8.1 (page 104) makes an automatic gear change a driver aid
  and forbids it, so :func:`~f1telemetry.physics.gearbox.step_gearbox` no longer looks at
  ``ice_rpm`` to decide anything. The rpm thresholds in ``car_spec.yaml`` stay as the *driver's*
  shift-point schedule; they are data a caller judges a request against, never a trigger. The
  second test here is the one that would fail if any rpm-driven trigger survived.
* **G2 - neutral and reverse are real states.** C9.7 (page 104) requires the car to be drivable in
  reverse by the driver at any time, and the channel contract already publishes ``-1/0/1..8``. So
  the state domain is ``{-1, 0, 1..8}``, gear 0 transmits exactly nothing, gear -1 transmits
  negative torque through the synthetic ``reverse_ratio``, and the forward table is never indexed
  by 0 or -1 - it is the only indexing in the module, and ``boundscheck=False`` would return a
  plausible ratio rather than fail.
* **G3 - the clutch is a driver demand, not a synthetic clamp.** C9.2.5 (page 101) expresses the
  driver's clutch demand as torque at the rear axle with a 5 200 Nm gain over 90 % of engagement
  travel. That is what the caller-owned engagement maps to, and the old
  ``clutch_torque_capacity_nm`` of 3 000 Nm is gone from control behaviour: one test here replaces
  it with several values and requires identical output.

The model is otherwise the one P1-T5 committed and the rest of this file still pins: a shift timer
the caller integrates over its own ``dt_s``, a boost cut that is the clutch rather than a second
multiplier, and ``throttle * ice torque * ratio * final_drive`` reaching the differential.
``shift_time_s`` of 40 ms is validated against C9.8.4's 200 ms and 300 ms limits by the loader, so
the one shared duration is inside both directions.

Every physical number comes from the loaded ``car_spec.yaml`` through
:class:`~f1telemetry.contracts.car_spec.KernelConfig`, so this file tunes nothing.

**Wheel speed is not here, on purpose.** ``ice_rpm`` is an input to this step - it selects a point
on the ICE curve - because turning an engine speed into a wheel speed is P1-T6's wheel rotational
state and assembling either into a force is P1-T7's.
"""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import yaml

from f1telemetry.contracts.car_spec import load_car_spec
from f1telemetry.physics import gearbox, powertrain
from f1telemetry.testing.invariants import check_gearbox_progression
from f1telemetry.testing.records import GroundTruthStep, SampleRecord

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import CarSpec, KernelConfig

pytestmark = pytest.mark.gearbox

# Read back off the dispatchers rather than duplicated here: an indirection between the reader
# and the options is one more place for them to be wrong, and these are the options the
# determinism claim rests on. `cache` is absent because numba consumes it at decoration time.
EXPECTED_OPTIONS: dict[str, Any] = {
    "fastmath": False,
    "nopython": True,
    "nogil": True,
    "boundscheck": False,
    "error_model": "numpy",
}

COMPILED_STEP = gearbox._step_gearbox  # pyright: ignore[reportPrivateUsage]

# The channel contract's state domain (``channels.yaml``'s ``gear``), restated as the two states
# outside the forward box. Every test below that needs one of them spells it as a named value rather
# than as a bare number, so a reader never has to remember which way round the domain runs.
NEUTRAL = gearbox.NEUTRAL_GEAR
REVERSE = gearbox.REVERSE_GEAR


@pytest.fixture(scope="module")
def config(spec: CarSpec) -> KernelConfig:
    """The validated, flat configuration the gearbox model is allowed to see."""
    return spec.kernel_config()


def _state(gear: float = 1.0, clutch: float = 1.0) -> np.ndarray:
    return gearbox.initial_state(gear=gear, clutch_engagement=clutch)


def _driveline_torque_nm(
    config: KernelConfig,
    gear: float,
    ice_rpm: float,
    throttle: float,
    clutch: float = 1.0,
    request: Any = None,
) -> float:
    """``step_gearbox`` from a fresh caller-owned state at ``gear``, for tests with no history.

    Most of this file cares about the *value* the step produces rather than where the state ended
    up, so it needs a one-shot call at a chosen gear. A helper is the honest way to say that: a
    fresh buffer, one step, no shared history. Tests that are about the state write it themselves.
    ``request`` defaults to no request, so a test about a torque rather than about a shift does not
    have to say so twice.
    """
    if request is None:
        return gearbox.step_gearbox(config, _state(gear, clutch), ice_rpm, throttle)
    return gearbox.step_gearbox(config, _state(gear, clutch), ice_rpm, throttle, request)


def _ratio(config: KernelConfig, gear: float) -> float:
    """The configured total reduction of ``gear``, read off the table in the test's own arithmetic.

    Deliberately *not* a call into the model: there is no public ratio helper to call, because a
    compiled one would take an arbitrary gear with ``boundscheck=False`` and read past the end of a
    short table. Writing the lookup here keeps the reference independent of the implementation and
    documents the 1-based-gear/0-based-table rule in the one place a reader looks for it. Gear 0
    has no ratio at all, which is why the caller below guards on ``gear >= 1`` rather than indexing.
    """
    return float(config.gear_ratios[int(gear) - 1]) * config.final_drive


def _peak_rpm(config: KernelConfig) -> float:
    return float(config.torque_rpm[int(np.argmax(config.torque_nm))])


def _offered_nm(config: KernelConfig, gear: float, rpm: float, throttle: float = 1.0) -> float:
    """What the gear train offers the clutch: throttle, engine torque, then the reduction.

    The engine torque is read through :func:`~f1telemetry.physics.powertrain.step_ice_torque` - the
    same composition the model uses - and the reduction is the test's own lookup, so this is the
    documented arithmetic rather than the gearbox's output restated. Reverse negates the configured
    reverse ratio instead of indexing the forward table, which is the construction under test.
    """
    ratio = _ratio(config, gear) if gear >= 1.0 else -config.reverse_ratio * config.final_drive
    return throttle * powertrain.step_ice_torque(config, rpm) * ratio


def _travel_band(config: KernelConfig) -> tuple[float, float]:
    """The engagement fractions the C9.2.5 gain spans, read off the configured travel fraction.

    The 5 % and 95 % endpoints are not written down in the model; they are the ends of the travel
    fraction C9.2.5 states, centred on the middle of the stroke. Deriving them here is what keeps
    the expectations independent of the arithmetic they check.
    """
    low = (1.0 - config.clutch_demand_travel_fraction) / 2.0
    return low, low + config.clutch_demand_travel_fraction


def _demand_nm(config: KernelConfig, engagement: float) -> float:
    """The rear-axle torque the driver's engagement asks for, written out here.

    Nothing in the project states this, so the reference is the regulation's own sentence: the gain
    applies to the engagement travel, the demand is zero with the clutch out and the full gain at
    full travel, and it rises linearly between them.
    """
    low, high = _travel_band(config)
    if engagement <= low:
        return 0.0
    if engagement >= high:
        return config.clutch_demand_torque_nm
    span = config.clutch_demand_travel_fraction
    return config.clutch_demand_torque_nm * (engagement - low) / span


def _sweep(
    config: KernelConfig,
    rpms: list[float],
    *,
    gear: float = 1.0,
    throttle: float = 1.0,
) -> list[tuple[int, float]]:
    """One step per rpm from one caller-owned state, holding the clutch closed and making no
    request.

    Used only where the caller holds rpm steady across steps, which after this slice is the only
    way rpm is allowed to reach the gearbox at all.
    """
    state = _state(gear, 1.0)
    seen: list[tuple[int, float]] = []
    for rpm in rpms:
        torque = gearbox.step_gearbox(config, state, rpm, throttle)
        seen.append((int(state[gearbox.GEAR_INDEX]), torque))
    return seen


def _shifted(config: KernelConfig, gear: float, rpm: float, request: Any) -> np.ndarray:
    """Step one caller-owned buffer once with the requested gear command."""
    state = _state(gear, 1.0)
    gearbox.step_gearbox(config, state, rpm, 1.0, request)
    return state


def _selected(config: KernelConfig, gear: float, request: Any) -> np.ndarray:
    """A buffer that has finished settling on the gear ``request`` selects from ``gear``.

    Selecting a gear goes through a shift, so a test that wants to measure the torque of a selected
    state has to wait the cut out first - otherwise it is measuring the cut. The gear the request
    moves to and the torque it then transmits are two different claims and this separates them.
    """
    state = _shifted(config, gear, 10_000.0, request)
    for _ in range(_cut_steps(config, state) + 2):
        gearbox.step_gearbox(config, state, 10_000.0, 1.0)
    return state


def _cut_steps(config: KernelConfig, state: np.ndarray, request: Any = None) -> int:
    """Hold ``request`` until the shift ends, and return how many steps the cut actually lasted.

    Driven by the state rather than by a step count, because the point of every caller of this is to
    measure the cut against the configured shift time instead of against the count it hoped for.
    ``request`` is optional so a caller that means "keep asking" and a caller that means "stop
    asking"
    read differently at the call site.
    """
    steps = 0
    while state[gearbox.SHIFT_TIMER_INDEX] > 0.0:
        if request is None:
            torque = gearbox.step_gearbox(config, state, 10_000.0, 1.0)
        else:
            torque = gearbox.step_gearbox(config, state, 10_000.0, 1.0, request)
        if torque != 0.0:
            break
        steps += 1
    return steps


def _record(steps: list[tuple[int, float]], throttle: float) -> SampleRecord:
    """A truth stream carrying only what invariant 7 reads, so the checker can be run on it.

    Every other field of :class:`GroundTruthStep` is about forces and is P1-T6/P1-T7's, and the
    record gets its defaults for all of them. What invariant 7 looks at is ``gear`` and
    ``throttle_pct``, and those are the two columns this file can honestly fill.
    """
    truth = tuple(
        GroundTruthStep(
            t_s=index * 1.0,
            vx_m_s=0.0,
            vy_m_s=0.0,
            ax_m_s2=0.0,
            ay_m_s2=0.0,
            az_m_s2=0.0,
            gear=gear,
            clutch=1.0,
            throttle_pct=100.0 * throttle,
            ice_power_w=0.0,
            mgu_k_power_w=0.0,
            drag_w=0.0,
            downforce_n=0.0,
            steer_rad=0.0,
        )
        for index, (gear, _) in enumerate(steps)
    )
    return SampleRecord(
        name="gearbox-progression",
        dt_s=0.0,
        description="gear progression driven by the P1-T5 model",
        ground_truth=truth,
        frames=(),
    )


def _document(repo: Path) -> dict[str, Any]:
    document = yaml.safe_load((repo / "car_spec.yaml").read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _at(root: dict[str, Any], path: tuple[Any, ...]) -> Any:
    node: Any = root
    for part in path:
        node = node[part]
    return node


def _write(root: dict[str, Any], tmp_path: Path) -> Path:
    path = tmp_path / "car_spec.yaml"
    path.write_text(yaml.safe_dump(root, sort_keys=False), encoding="utf-8")
    return path


def test_the_gearbox_primitives_are_compiled_kernels_with_the_project_options(
    config: KernelConfig,
) -> None:
    """A Python fallback here would be a silent step backwards, so the dispatchers are asked.

    What this pins is that each primitive is a compiled dispatcher carrying the project's options
    and that a call registered a signature rather than falling back to Python. ``cache=True`` is
    not asserted here: numba consumes it at decoration time and cannot read it back off
    ``targetoptions``; ``tests/test_longitudinal_kernel.py`` proves it once, from a fresh
    interpreter and an empty cache directory, for the convention these functions follow.
    """
    state = _state()
    gearbox.step_gearbox(config, state, 6_000.0, 1.0, gearbox.GearRequest.UP)
    gearbox.clutch_demand_nm(
        0.5, config.clutch_demand_torque_nm, config.clutch_demand_travel_fraction
    )
    gearbox.clutch_output_torque_nm(300.0, 5_200.0)
    gearbox.step_requested_gear(1, int(gearbox.GearRequest.UP), 8)
    for function in (
        COMPILED_STEP,
        gearbox.clutch_demand_nm,
        gearbox.clutch_output_torque_nm,
        gearbox.step_requested_gear,
    ):
        name = function.py_func.__name__
        assert len(function.signatures) >= 1, f"{name} never compiled a signature"
        options = dict(function.targetoptions)
        for option, expected in EXPECTED_OPTIONS.items():
            assert options[option] == expected, (name, option)


def test_engine_speed_alone_never_changes_the_gear(config: KernelConfig) -> None:
    """C9.8.1: an automatic gear change is a driver aid, so rpm may not move the box.

    This is the test the whole slice exists for. The sweep crosses ``shift_up_rpm`` several times
    over a rev-limited range, so a gearbox with any rpm comparison left in it - even one that
    required several steps above the threshold - would move here. Every step is otherwise a
    full-throttle, closed-clutch step with the engine on its torque curve, which is exactly the
    input
    the previous model used as its trigger.
    """
    rpms = [
        config.idle_rpm + 250.0 * index
        for index in range(int((config.rev_limit_rpm - config.idle_rpm) / 250.0) + 1)
    ]
    assert max(rpms) > config.shift_up_rpm, "the sweep has to cross the driver's upshift point"

    rising = _sweep(config, rpms, gear=1.0)
    assert {gear for gear, _ in rising} == {1}, "engine speed upshifted the gearbox"
    assert max(torque for _, torque in rising) > 0.0, "and the torque really was live throughout"

    # The same sweep from top gear has to be just as inert: a downshift trigger would show here,
    # where the rpm falls back below `shift_down_rpm` on the way down.
    falling = _sweep(config, list(reversed(rpms)), gear=float(config.gear_ratios.size))
    assert {gear for gear, _ in falling} == {int(config.gear_ratios.size)}


def test_an_up_request_advances_exactly_one_gear_and_a_down_request_exactly_one_back(
    config: KernelConfig,
) -> None:
    """The request is the trigger, and one request moves one gear.

    C9.8.3 requires each change to be separately driver-initiated and only one at a time, so the
    move has to be a single gear rather than "whatever the rpm says". The rpm is held between the
    two configured shift points on purpose, so the assertion is about the request rather than about
    a threshold that would also have fired.
    """
    rpm = (config.shift_up_rpm + config.shift_down_rpm) / 2.0
    assert _shifted(config, 1.0, rpm, gearbox.GearRequest.UP)[gearbox.GEAR_INDEX] == 2.0
    assert _shifted(config, 3.0, rpm, gearbox.GearRequest.DOWN)[gearbox.GEAR_INDEX] == 2.0

    # A hold request is a request too, and it has to be inert: if HOLD were not handled the box
    # would fall through to some other command.
    assert _shifted(config, 3.0, rpm, gearbox.GearRequest.HOLD)[gearbox.GEAR_INDEX] == 3.0


def test_a_request_walks_the_box_one_gear_at_a_time_and_stops_at_the_top(
    config: KernelConfig,
) -> None:
    """Every step up the box is one gear, and the last gear refuses to go anywhere.

    A model that read the rpm and jumped to whichever ratio fitted would skip; one that treated UP
    as "top gear" would jump too. The refusals at the end are asserted on the shift timer as well as
    the gear, because a refusal that armed a cut would look identical on the gear alone.
    """
    top = int(config.gear_ratios.size)
    budget = int(config.shift_time_s / config.dt_s)
    state = _state(1.0, 1.0)
    seen: list[int] = []
    for _ in range(top + 2):
        gearbox.step_gearbox(config, state, 10_000.0, 1.0, gearbox.GearRequest.UP)
        for _ in range(budget + 2):
            gearbox.step_gearbox(config, state, 10_000.0, 1.0, gearbox.GearRequest.HOLD)
        seen.append(int(state[gearbox.GEAR_INDEX]))

    assert seen[: top - 1] == list(range(2, top + 1)), f"the box walked {seen} rather than one gear"
    assert set(seen[top - 1 :]) == {top}, "the top gear kept shifting"
    assert state[gearbox.SHIFT_TIMER_INDEX] == 0.0, "a refused upshift still cut the torque"
    assert gearbox.step_gearbox(config, state, 10_000.0, 1.0, gearbox.GearRequest.UP) > 0.0


def test_first_gear_refuses_a_down_request_into_neutral_or_reverse(config: KernelConfig) -> None:
    """A down request moves one *forward* gear, and there is no forward gear below first.

    C9.8.3 fixes a minimum gear while the car is moving, and this slice models the forward box as
    adjacent-only: a down request from first gear is dropped rather than walking the box into
    neutral or reverse, because those are states the driver selects deliberately and not a step
    further down the ratios.
    """
    rpm = 10_000.0
    refused = _shifted(config, 1.0, rpm, gearbox.GearRequest.DOWN)
    assert refused[gearbox.GEAR_INDEX] == 1.0
    assert refused[gearbox.SHIFT_TIMER_INDEX] == 0.0, "a refused request still cut the torque"
    assert gearbox.step_gearbox(config, refused, rpm, 1.0) > 0.0, "the box stopped driving"


def test_a_request_is_refused_until_the_shift_ends_and_nothing_is_queued(
    config: KernelConfig,
) -> None:
    """C9.8.3's "one at a time", and the spec's "a fresh request is required" after it.

    A request that arrives during a shift is dropped, not deferred. If it were deferred the second
    half would move to third gear without the driver asking again, and the third half would not
    need a fresh request at all - so all three are here.
    """
    budget = int(config.shift_time_s / config.dt_s)
    state = _state(1.0, 1.0)

    gearbox.step_gearbox(config, state, 10_000.0, 1.0, gearbox.GearRequest.UP)
    assert state[gearbox.GEAR_INDEX] == 2.0
    assert state[gearbox.SHIFT_TIMER_INDEX] > 0.0

    cut = _cut_steps(config, state, gearbox.GearRequest.UP)
    assert 1 <= cut <= budget + 2, f"the cut lasted {cut} steps against a budget of {budget}"
    assert state[gearbox.GEAR_INDEX] == 2.0, "a request during the cut was accepted"
    assert state[gearbox.SHIFT_TIMER_INDEX] == 0.0

    gearbox.step_gearbox(config, state, 10_000.0, 1.0)
    assert state[gearbox.GEAR_INDEX] == 2.0, "the dropped request was queued and fired later"

    gearbox.step_gearbox(config, state, 10_000.0, 1.0, gearbox.GearRequest.UP)
    assert state[gearbox.GEAR_INDEX] == 3.0, "a fresh request was not accepted after the shift"


def test_a_shift_cuts_the_torque_for_exactly_the_configured_shift_time(
    config: KernelConfig,
) -> None:
    """The cut is a hard zero for ``shift_time_s`` and the gear is frozen throughout it.

    Two claims, and the second is what makes the first mean something. A hard zero is only the
    boost cut if the gear cannot move while the cut is running: otherwise the driver could request
    the next gear on the step the timer expired and the torque would never be delivered at all.

    The count is allowed one step either side of ``shift_time_s / dt_s`` because the countdown is
    accumulated in floating point - ``0.04`` is not representable - so the final subtraction can
    land
    a step early or late. What is pinned is that the cut is neither one step nor a hundred.
    """
    dt_s = config.dt_s
    budget = config.shift_time_s / dt_s
    assert budget == pytest.approx(400.0), "the committed shift time or step changed"

    state = _state(1.0, 1.0)
    gearbox.step_gearbox(config, state, 10_000.0, 1.0, gearbox.GearRequest.UP)
    assert state[gearbox.GEAR_INDEX] == 2.0

    cut = _cut_steps(config, state)
    assert cut in (int(budget), int(budget) + 1)
    assert state[gearbox.GEAR_INDEX] == 2.0, "the gear moved before its shift finished"

    # A closed clutch at full throttle delivers torque again as soon as the cut is over; without the
    # cut this is the step that would have produced it, so the assertion above is not vacuous.
    resumed = gearbox.step_gearbox(config, state, 10_000.0, 1.0)
    assert resumed > 0.0
    assert resumed == pytest.approx(_offered_nm(config, 2.0, 10_000.0), rel=1e-12)


def test_the_original_gear_is_disengaged_within_the_c9_8_4_window(config: KernelConfig) -> None:
    """Request to disengagement stays inside the 80 ms C9.8.4 allows.

    C9.8.4 bounds two quantities separately: the whole change at 200 ms up and 300 ms down, and the
    time from the request to the original gear being disengaged at 80 ms. This model disengages by
    cutting the clutch on the request step itself, so the second is one step - and the first is the
    shift time, which the loader already checks against the smaller limit. Both are asserted against
    the configured limits rather than against numbers written into the test.
    """
    assert config.shift_time_s <= min(config.shift_time_max_up_s, config.shift_time_max_down_s)

    state = _state(1.0, 1.0)
    assert gearbox.step_gearbox(config, state, 10_000.0, 1.0, gearbox.GearRequest.UP) == 0.0, (
        "the request step still delivered first gear's torque, so nothing was disengaged"
    )
    assert state[gearbox.GEAR_INDEX] == 2.0

    assert config.dt_s <= config.shift_disengage_max_s
    assert state[gearbox.SHIFT_TIMER_INDEX] > 0.0

    _cut_steps(config, state)
    assert gearbox.step_gearbox(config, state, 10_000.0, 1.0) == pytest.approx(
        _offered_nm(config, 2.0, 10_000.0), rel=1e-12
    )


def test_neutral_transmits_exactly_zero_and_is_reached_only_by_request(
    config: KernelConfig,
) -> None:
    """Gear 0 is neutral: it transmits nothing, at any throttle, in either direction.

    C9.7 makes reverse drivability a requirement and neutral an existing model state, so both are
    requested rather than inferred. The torque assertions are ``== 0.0`` and not approximations
    because a neutral that returned ``-0.0``, or a ratio of zero multiplying a large torque, would
    slip past a tolerance and still be a neutral that is not nothing.
    """
    rpm = _peak_rpm(config)
    for throttle in (0.0, 0.5, 1.0):
        for clutch in (0.0, 0.5, 1.0):
            state = _selected(config, 2.0, gearbox.GearRequest.NEUTRAL)
            assert state[gearbox.GEAR_INDEX] == NEUTRAL
            state[gearbox.CLUTCH_INDEX] = clutch
            assert gearbox.step_gearbox(config, state, rpm, throttle) == 0.0, (
                f"neutral transmitted at throttle {throttle}, clutch {clutch}"
            )

    # Reached only by request: an up or down request never selects it, and a neutral box asked for
    # first gear goes back to first.
    assert _shifted(config, 1.0, rpm, gearbox.GearRequest.UP)[gearbox.GEAR_INDEX] == 2.0
    assert _shifted(config, 1.0, rpm, gearbox.GearRequest.DOWN)[gearbox.GEAR_INDEX] == 1.0
    assert _shifted(config, NEUTRAL, rpm, gearbox.GearRequest.UP)[gearbox.GEAR_INDEX] == 1.0
    assert (
        _shifted(config, NEUTRAL, rpm, gearbox.GearRequest.NEUTRAL)[gearbox.GEAR_INDEX] == NEUTRAL
    )


def test_reverse_transmits_negative_torque_from_the_configured_reverse_ratio(
    config: KernelConfig,
) -> None:
    """C9.7: the car has to be drivable in reverse, and reverse negates rather than indexing.

    The discriminator is between two ways to build reverse. Index the forward table with ``-1`` and
    the last ratio drives backwards; negate a separate synthetic ratio and the magnitude is the one
    ``car_spec.yaml`` records. Sign, magnitude and the fact that a neutral request out of reverse
    still transmits nothing are all asserted, because the first two distinguish the two
    constructions and the third stops ``-1`` being read as "some forward gear".
    """
    rpm = 10_000.0
    engine = powertrain.step_ice_torque(config, rpm)
    assert engine > 0.0
    assert config.reverse_ratio > 0.0, "a negative reverse ratio would be the ratio counted twice"

    # Selecting reverse goes through a shift, so the cut is waited out before the torque is read:
    # what is under test is what reverse transmits, not what a shift transmits.
    state = _selected(config, 1.0, gearbox.GearRequest.REVERSE)
    assert state[gearbox.GEAR_INDEX] == REVERSE

    torque = gearbox.step_gearbox(config, state, rpm, 1.0)
    assert torque == pytest.approx(-engine * config.reverse_ratio * config.final_drive, rel=1e-12)
    assert torque < 0.0, "reverse under positive throttle has to drive the car backwards"

    # Reverse is a state the driver selects at any time rather than a step on the way down the box,
    # so it is reachable from top gear in one request. Its committed magnitude is deliberately equal
    # to the first ratio - ``car_spec.yaml`` calls that the conventional reverse layout - so
    # it is the
    # *end* of the table it must not be borrowing: indexing gear -1 would give the last ratio.
    from_top = _selected(config, float(config.gear_ratios.size), gearbox.GearRequest.REVERSE)
    assert from_top[gearbox.GEAR_INDEX] == REVERSE
    assert config.reverse_ratio == pytest.approx(float(config.gear_ratios[0]))
    assert not math.isclose(config.reverse_ratio, float(config.gear_ratios[-1]), rel_tol=1e-9)

    # And the only way out is a request, which is what makes the sign a consequence of the gear
    # rather than of the torque.
    neutral = _selected(config, REVERSE, gearbox.GearRequest.NEUTRAL)
    assert neutral[gearbox.GEAR_INDEX] == NEUTRAL
    assert gearbox.step_gearbox(config, neutral, rpm, 1.0) == 0.0, (
        "leaving reverse for neutral has to stop the torque, not just change the label"
    )
    assert _selected(config, REVERSE, gearbox.GearRequest.UP)[gearbox.GEAR_INDEX] == 1.0
    assert (
        _shifted(config, REVERSE, rpm, gearbox.GearRequest.DOWN)[gearbox.GEAR_INDEX] == REVERSE
    ), "a down request in reverse has to be a no-op rather than a step further down"


def test_the_clutch_demand_maps_the_travel_range_onto_the_c9_2_5_gain(config: KernelConfig) -> None:
    """The engagement is a paddle position and the demand is the rear-axle torque it asks for.

    C9.2.5 (page 101) states the demand as a 5 200 Nm gain over 90 % of engagement travel, mapped
    from 5 % to 95 %. Three claims, and the endpoints are two of them: fully disengaged asks for
    nothing, full travel asks for the whole gain, and the span between is linear and monotone. The
    endpoints are derived from the configured travel fraction rather than written down, so a change
    to the fraction moves this test's expectations with it.

    **The demand has to be reachable, or the gearbox could not be launched.** With the committed
    gear set the first gear offers about 3 844 Nm at the engine's peak, so a demand topping out
    below that could never hold the clutch against a launch in first, and the engagement would be a
    control with no effect on the phase that needs it.
    """
    low, high = _travel_band(config)
    assert (low, high) == pytest.approx((0.05, 0.95)), "the committed travel fraction moved"
    gain = config.clutch_demand_torque_nm
    assert gain == pytest.approx(5_200.0), "the committed C9.2.5 gain moved"

    offered = _offered_nm(config, 1.0, _peak_rpm(config))
    assert 0.0 < offered < gain, (
        f"first gear offers {offered:.0f} Nm, below the {gain:.0f} Nm demand: "
        "and the paddle would have no effect on a launch"
    )

    # The endpoints, the closed intervals, and the exact midpoint of the span.
    for engagement, expected in (
        (0.0, 0.0),
        (low, 0.0),
        (high, gain),
        (1.0, gain),
        ((low + high) / 2.0, gain / 2.0),
    ):
        assert gearbox.clutch_demand_nm(
            engagement, gain, config.clutch_demand_travel_fraction
        ) == pytest.approx(expected, rel=1e-12), (
            f"engagement {engagement} did not map to {expected}"
        )

    # Linear in between, and monotone across the whole stroke.
    strokes = [index / 200.0 for index in range(201)]
    demands = [
        gearbox.clutch_demand_nm(engagement, gain, config.clutch_demand_travel_fraction)
        for engagement in strokes
    ]
    assert all(later >= earlier for earlier, later in pairwise(demands))
    for engagement in (0.2, 0.4, 0.6, 0.8):
        assert demands[int(engagement * 200)] == pytest.approx(
            _demand_nm(config, engagement), rel=1e-12
        )

    # Zero demand is a zero output whatever the gearbox would have offered, which is the
    # fully-disengaged end of C9.2.3: disengaged, it cannot transmit any usable torque.
    assert _driveline_torque_nm(config, 1.0, _peak_rpm(config), 1.0, clutch=0.0) == 0.0
    assert _driveline_torque_nm(config, 1.0, _peak_rpm(config), 1.0, clutch=low) == 0.0


def test_the_transmitted_torque_tracks_the_demand_within_the_c9_2_5_band(
    config: KernelConfig,
) -> None:
    """The clutch output follows the demand, and the C9.2.5 tracking band is checked, not assumed.

    C9.2.5 requires the control system to hold the transmitted rear-axle torque within +/-150 Nm of
    the demand. This model has no controller between the two - the demand *is* the transmitted
    torque
    wherever the gear train can supply it - so the band is checked rather than assumed: every
    engagement whose demand the first gear can carry has to land inside it. An implementation that
    clamped to the old 3 000 Nm capacity, or that scaled the engine torque by the engagement, would
    be outside the band exactly where the demand is 3 000-5 200 Nm.

    The launch exception is the other half: where the gear train cannot supply the demand the clutch
    slips and transmits what the engine makes, which is the case C9.2.5 exempts for the first 85 ms
    of a launch step, and the reason a synthetic ceiling above the engine's offer was wrong.
    """
    rpm = _peak_rpm(config)
    offered = _offered_nm(config, 1.0, rpm)
    band = config.clutch_control_error_max_nm
    assert band == pytest.approx(150.0), "the committed C9.2.5 band moved"
    assert config.clutch_launch_exception_s == pytest.approx(0.085)

    for index in range(1, 101):
        engagement = index / 100.0
        demand = _demand_nm(config, engagement)
        torque = _driveline_torque_nm(config, 1.0, rpm, 1.0, clutch=engagement)
        assert torque == pytest.approx(min(demand, offered), rel=1e-12), f"engagement {engagement}"
        if demand <= offered:
            assert abs(torque - demand) <= band, (
                f"engagement {engagement} is inside the gear train's capability, so the "
                f"transmitted {torque:.1f} Nm is outside the +/-{band:.0f} Nm band around the "
                f"{demand:.1f} Nm demand"
            )


def test_a_launch_ramp_tracks_the_demand_until_the_gear_train_becomes_the_limit(
    config: KernelConfig,
) -> None:
    """Feathering the clutch up a standing start is the demand being followed, not a clamp.

    A launch is a ramp: the driver lets the clutch out over a few tens of milliseconds and the
    driveline torque follows the paddle until the engine can hold it. The ramp is walked in
    configured steps so the phase boundary is a real crossing rather than a hand-picked engagement,
    and the three phases are the ones the model has to have: nothing while the clutch is out, the
    demand through the middle where the gear train can supply it, and then the engine's own torque
    where the demand asks for more than first gear can deliver - a slipping clutch at a launch, not
    a ceiling.
    """
    rpm = _peak_rpm(config)
    offered = _offered_nm(config, 1.0, rpm)
    ramp_s = 0.05
    assert ramp_s <= config.clutch_launch_exception_s, (
        "the whole launch exception has to cover the ramp, or this test is asserting the band "
        "outside the window C9.2.5 exempts"
    )
    ramp_steps = int(ramp_s / config.dt_s)
    engagements = [index / ramp_steps for index in range(ramp_steps + 1)]
    assert engagements[-1] == 1.0

    state = _state(1.0, 0.0)
    torques: list[float] = []
    for engagement in engagements:
        state[gearbox.CLUTCH_INDEX] = engagement
        torques.append(gearbox.step_gearbox(config, state, rpm, 1.0))

    demands = [_demand_nm(config, engagement) for engagement in engagements]
    for engagement, demand, torque in zip(engagements, demands, torques, strict=True):
        assert torque == pytest.approx(min(demand, offered), rel=1e-9), (
            f"at engagement {engagement:.3f} the ramp transmitted {torque:.1f} Nm for a demand of "
            f"{demand:.1f} Nm against an offer of {offered:.1f} Nm"
        )
    assert all(later >= earlier for earlier, later in pairwise(torques))
    assert torques[0] == 0.0
    assert max(torques) == pytest.approx(offered, rel=1e-12), (
        "a fully engaged clutch has to deliver the engine's torque, so the ramp has to end at the "
        "offer and not at the demand"
    )


def test_the_synthetic_clutch_capacity_clamp_is_gone_from_control_behavior(
    config: KernelConfig,
) -> None:
    """The 3 000 Nm ceiling must not reach the output, at any engagement or gear.

    G3 in the spec: ``clutch_torque_capacity_nm`` is a synthetic clamp from the previous model, not
    a mechanical capacity and not the C9.2.5 demand, and keeping it would contradict the
    torque-demand model while it binds. The committed first gear offers more than it at the engine
    peak, so replacing the config field with several values - including the committed number and
    numbers either side of the offer - has to leave every answer identical. A field the step still
    read would move the output on at least one of them.
    """
    peak = _peak_rpm(config)
    offered = _offered_nm(config, 1.0, peak)
    assert offered > config.clutch_torque_capacity_nm, (
        "this test needs a gear whose offer is above the old clamp, or nothing would differ"
    )

    for capacity in (0.0, 1_000.0, config.clutch_torque_capacity_nm, 10.0 * offered):
        replaced = replace(config, clutch_torque_capacity_nm=capacity)
        for gear in (1.0, 2.0, 8.0):
            for engagement in (0.0, 0.3, 0.7, 1.0):
                assert _driveline_torque_nm(replaced, gear, peak, 1.0, clutch=engagement) == (
                    pytest.approx(
                        _driveline_torque_nm(config, gear, peak, 1.0, clutch=engagement), rel=1e-12
                    )
                ), f"a {capacity} Nm capacity still moved gear {int(gear)} at {engagement}"


def test_the_boost_cut_is_the_clutch_and_not_a_second_multiplier(config: KernelConfig) -> None:
    """Zero torque through the cut whatever the engagement was, and zero only through the cut.

    ``PHASES.md`` P1-T5 asks for trailing throttle and the boost cut to route through the clutch,
    which is a claim about there being *one* path. It shows up as: with the clutch shut, a fully
    open
    throttle at high rpm delivers exactly nothing, and with the clutch open by any amount the
    delivery is that same nothing - not a fraction of it.
    """
    state = _state(1.0, 1.0)
    gearbox.step_gearbox(config, state, 10_000.0, 1.0, gearbox.GearRequest.UP)
    assert state[gearbox.GEAR_INDEX] == 2.0
    assert state[gearbox.SHIFT_TIMER_INDEX] > 0.0

    for engagement in (0.0, 0.25, 0.5, 1.0):
        fresh = _state(2.0, engagement)
        fresh[gearbox.SHIFT_TIMER_INDEX] = state[gearbox.SHIFT_TIMER_INDEX]
        assert gearbox.step_gearbox(config, fresh, 10_000.0, 1.0) == 0.0


def test_gear_progression_satisfies_invariant_7(config: KernelConfig, spec: CarSpec) -> None:
    """The real checker, over a driven trace: in range, and no reverse under power.

    Invariant 7 is the condition this task has to satisfy, so it is run here on the gear the model
    actually produces rather than on a supplied fixture. The trace is driven by requests only - a
    rising rpm is the situation the invariant describes - because invariant 7 reads *any* gear
    decrease as a violation and a requested downshift is legal behaviour that a later task turns
    into
    an assertion of its own.
    """
    top = int(config.gear_ratios.size)
    state = _state(1.0, 1.0)
    trace: list[tuple[int, float]] = []
    for _ in range(top):
        trace.append(
            (int(state[gearbox.GEAR_INDEX]), gearbox.step_gearbox(config, state, 10_000.0, 1.0))
        )
        gearbox.step_gearbox(config, state, 10_000.0, 1.0, gearbox.GearRequest.UP)
        for _ in range(_cut_steps(config, state) + 2):
            trace.append(
                (int(state[gearbox.GEAR_INDEX]), gearbox.step_gearbox(config, state, 10_000.0, 1.0))
            )

    gears = [gear for gear, _ in trace]
    assert gears == sorted(gears), "the gearbox went backwards under a rising rpm"
    assert min(gears) >= 1, "the gearbox answered with neutral or reverse"
    assert max(gears) == top, (
        "the trace has to reach the top gear for the range check to mean anything"
    )
    assert all(torque >= 0.0 for _, torque in trace), "drive torque must not reverse a launch"
    assert max(torque for _, torque in trace) > 0.0, "the trace never delivered torque"

    violations = check_gearbox_progression(_record(trace, throttle=1.0), spec)
    assert not violations, f"invariant 7 rejected a gearbox the model drove: {violations}"


def test_each_gear_multiplies_the_transmitted_torque_by_its_own_ratio(config: KernelConfig) -> None:
    """Each gear reduces the engine torque by its own ``ratio * final_drive``.

    This is the test that a gearbox which selected a gear and then returned the same engine torque
    for all eight cannot pass: every other test in this file looks at *when* the gear changes or at
    whether the torque is zero, and all of those are satisfied by a model that ignores the ratio
    table entirely. Here every gear is checked against its own row of the table, so a model reading
    only ``len(gear_ratios)`` - using the table as a gear count and nothing else - fails on gear 1.

    Full throttle is deliberate now. With the C9.2.5 demand and no ceiling above it, nothing is
    demand-limited in any gear, so the ratio is the only thing in the output.
    """
    rpm = 10_000.0
    for gear in range(1, int(config.gear_ratios.size) + 1):
        assert _driveline_torque_nm(config, float(gear), rpm, 1.0) == pytest.approx(
            _offered_nm(config, float(gear), rpm), rel=1e-12
        ), f"gear {gear} did not reduce the torque by its own ratio"

    # The ratios fall across the box, so the output has to fall too: a model returning a
    # gear-independent torque cannot satisfy this even if the per-gear arithmetic above is faked.
    outputs = [_driveline_torque_nm(config, float(gear), rpm, 1.0) for gear in range(1, 9)]
    assert outputs == sorted(outputs, reverse=True)
    assert outputs[0] / outputs[-1] == pytest.approx(
        (float(config.gear_ratios[0]) * config.final_drive)
        / (float(config.gear_ratios[-1]) * config.final_drive),
        rel=1e-12,
    )


def test_trailing_throttle_passes_through_the_clutch_unmodulated(config: KernelConfig) -> None:
    """Off the throttle the engine decides the torque, not the clutch.

    Trailing throttle is the reason the model scales the engine torque by a pedal before the clutch
    sees it: a driver part-way down the throttle at 12 000 rpm wants that fraction of the engine's
    torque through a closed clutch, and a model that limited the clutch would hand back the demand
    instead. The expectation is the documented arithmetic - throttle fraction, engine torque, then
    the
    gear - written out here rather than taken from the model.
    """
    rpm = 12_000.0
    gear = 3.0
    engine = powertrain.step_ice_torque(config, rpm)
    assert engine > 0.0

    for throttle in (0.0, 0.05, 0.2, 0.5):
        torque = _driveline_torque_nm(config, gear, rpm, throttle)
        assert torque == pytest.approx(throttle * engine * _ratio(config, gear), rel=1e-12), (
            f"throttle {throttle} is not reaching the transmission unmodulated"
        )
    assert _driveline_torque_nm(config, gear, rpm, 0.0) == 0.0


def test_the_state_is_the_callers_and_the_step_writes_the_gear_and_the_timer_only(
    config: KernelConfig,
) -> None:
    """One caller-owned buffer, and the step writes two of its three slots.

    Caller-owned is the whole point: a run steps one array over and over with no allocation and no
    second copy to keep in step, which is the rule ``kernels.longitudinal`` already follows for
    ``out``. What the step *writes* is the gear and the shift timer; the clutch engagement it reads.
    The engagement check is the one that matters here - if the boost cut ever wrote zero back, a
    caller ramping a launch would have its input silently destroyed on its first upshift.
    """
    state = gearbox.initial_state(gear=1.0, clutch_engagement=0.6)
    assert state.dtype == np.float64
    assert state.shape == (gearbox.STATE_SIZE,)
    assert state[gearbox.GEAR_INDEX] == 1.0
    assert state[gearbox.CLUTCH_INDEX] == 0.6
    assert state[gearbox.SHIFT_TIMER_INDEX] == 0.0

    torque = gearbox.step_gearbox(config, state, 6_000.0, 1.0)
    expected = min(_demand_nm(config, 0.6), _offered_nm(config, 1.0, 6_000.0))
    assert torque == pytest.approx(expected, rel=1e-12)
    assert state[gearbox.GEAR_INDEX] == 1.0
    assert state[gearbox.CLUTCH_INDEX] == 0.6, "the step read the engagement and left it alone"
    assert np.all(state >= 0.0)


def test_the_caller_owned_engagement_survives_a_shift_and_the_caller_may_change_it(
    config: KernelConfig,
) -> None:
    """The boost cut is this step's decision; the engagement in the buffer is the caller's.

    Caller-owned means the caller supplies and retains the engagement, and the step may change it
    between calls. This drives that in both directions: a part-engaged launch takes a requested
    upshift, and through the whole shift - including the step the gear changes on - the buffer keeps
    the value the caller put there and the torque is exactly zero. When the shift ends the same
    engagement is still in place and torque resumes immediately, with no re-arming.

    Writing the cut back into the slot would fail the second half: a launch would be silently reset
    to a shut clutch on its first upshift and could not move again without the caller noticing why.
    """
    engagement = 0.45
    state = gearbox.initial_state(gear=1.0, clutch_engagement=engagement)

    # The launch itself transmits through a partly closed clutch.
    launch = gearbox.step_gearbox(config, state, 10_000.0, 1.0)
    assert launch == pytest.approx(
        min(_demand_nm(config, engagement), _offered_nm(config, 1.0, 10_000.0)), rel=1e-12
    )
    assert launch > 0.0

    # The requested upshift cuts the torque to nothing, and the engagement is untouched throughout.
    assert gearbox.step_gearbox(config, state, 10_000.0, 1.0, gearbox.GearRequest.UP) == 0.0
    while state[gearbox.SHIFT_TIMER_INDEX] > 0.0:
        if gearbox.step_gearbox(config, state, 10_000.0, 1.0) == 0.0:
            assert state[gearbox.CLUTCH_INDEX] == engagement, (
                "the boost cut wrote the engagement back into the caller's state"
            )
        else:
            break
    assert state[gearbox.GEAR_INDEX] == 2.0
    assert state[gearbox.CLUTCH_INDEX] == engagement, "the shift consumed the caller's engagement"

    # And the same engagement is still in force the moment the shift is over.
    assert gearbox.step_gearbox(config, state, 10_000.0, 1.0) == pytest.approx(
        min(_demand_nm(config, engagement), _offered_nm(config, 2.0, 10_000.0)), rel=1e-12
    )

    # The caller may then change it, which is what a driver feathering mid-shift looks like.
    state[gearbox.CLUTCH_INDEX] = 0.9
    assert gearbox.step_gearbox(config, state, 10_000.0, 1.0) == pytest.approx(
        min(_demand_nm(config, 0.9), _offered_nm(config, 2.0, 10_000.0)), rel=1e-12
    )


def test_the_ratios_reverse_ratio_and_demand_gain_are_read_off_the_kernel_config(
    config: KernelConfig,
) -> None:
    """Replacing a config field moves the answer that field names, and nothing else.

    A ratio, a reverse ratio or the C9.2.5 gain written into the module would pass every other test
    here and fail this one, which is what lets the file claim it tunes nothing. The gear count is
    read from the ratios, so replacing them with a shorter set has to shorten the gearbox too.
    """
    assert float(config.gear_ratios.size) == 8.0
    assert config.clutch_demand_torque_nm > 0.0
    idle = config.idle_rpm
    # Throttle is held down so that a doubled final drive cannot push any of these gears past the
    # demand: the point of each edit below is that the ratio reaches the output, and a demand in the
    # way would turn a 2x expectation into two identical numbers.
    throttle = 0.25

    halved = replace(config, gear_ratios=config.gear_ratios / 2.0)
    assert _driveline_torque_nm(halved, 1.0, idle, throttle) == pytest.approx(
        0.5 * _driveline_torque_nm(config, 1.0, idle, throttle), rel=1e-12
    ), "the ratio table has to reach the driveline torque"

    longer = replace(config, final_drive=config.final_drive * 2.0)
    for gear in (1.0, 4.0, 8.0):
        assert _driveline_torque_nm(longer, gear, idle, throttle) == pytest.approx(
            2.0 * _driveline_torque_nm(config, gear, idle, throttle), rel=1e-12
        ), f"final_drive is not reaching the driveline torque in gear {int(gear)}"

    # A ratio edit is local to its own gear: changing one ratio must not move the other seven.
    retaller = config.gear_ratios.copy()
    retaller[2] *= 1.5
    third_heavier = replace(config, gear_ratios=retaller)
    assert _driveline_torque_nm(third_heavier, 3.0, idle, throttle) == pytest.approx(
        1.5 * _driveline_torque_nm(config, 3.0, idle, throttle), rel=1e-12
    )
    for other in (1.0, 2.0, 4.0, 8.0):
        assert _driveline_torque_nm(third_heavier, other, idle, throttle) == pytest.approx(
            _driveline_torque_nm(config, other, idle, throttle), rel=1e-12
        ), f"editing the third ratio moved gear {int(other)}"

    # The C9.2.5 gain reaches the demand, so a smaller one limits a launch the committed one does
    # not: half of 5 200 Nm is below what first gear offers at the engine peak, which is what makes
    # this the right edit to distinguish a demand from a multiplier of the output.
    peak = _peak_rpm(config)
    weaker = replace(config, clutch_demand_torque_nm=config.clutch_demand_torque_nm / 2.0)
    assert _driveline_torque_nm(weaker, 1.0, peak, 1.0) == pytest.approx(
        0.5 * config.clutch_demand_torque_nm, rel=1e-12
    ), "the demand gain is not reaching the output"

    # The travel fraction is the span the gain is applied over, so widening it lowers the
    # demand at a
    # fixed engagement. Derived here rather than asserted against a written constant.
    wide = replace(config, clutch_demand_travel_fraction=0.5)
    engagement = 0.75
    assert gearbox.clutch_demand_nm(
        engagement, config.clutch_demand_torque_nm, wide.clutch_demand_travel_fraction
    ) == pytest.approx(config.clutch_demand_torque_nm, rel=1e-12)

    # And the reverse ratio reaches the reverse output, which nothing else in this file touches.
    slower = replace(config, reverse_ratio=config.reverse_ratio * 0.5)
    assert _driveline_torque_nm(slower, REVERSE, 10_000.0, 1.0) == pytest.approx(
        0.5 * _driveline_torque_nm(config, REVERSE, 10_000.0, 1.0), rel=1e-12
    )

    # A shorter table is a shorter box, so the top gear has to be read off the array rather than
    # assumed to be eight: the shift is driven through the same shortened config, because a gear of
    # three is out of the domain a two-ratio box can hold.
    shorter = replace(config, gear_ratios=config.gear_ratios[:2].copy())
    short_state = _shifted(shorter, 2.0, 10_000.0, gearbox.GearRequest.UP)
    assert short_state[gearbox.GEAR_INDEX] == 2.0, "the last gear must not upshift into nothing"
    assert gearbox.step_gearbox(shorter, short_state, 10_000.0, 1.0) > 0.0


def test_editing_the_car_spec_changes_the_gearbox_with_no_code_edit(
    tmp_path: Path,
    repo: Path,
    config: KernelConfig,
) -> None:
    """P1-T1b for the gearbox: the file is the only place a coefficient lives.

    One edit per quantity the model consumes - a shift time and the C9.2.5 demand gain - and each
    has to move the answer it is supposed to move. A constant written into the module would pass
    every other test in this file.
    """
    idle = config.idle_rpm
    closed = gearbox.step_gearbox(config, _state(1.0, 1.0), idle, 1.0)
    assert closed > 0.0

    root = _document(repo)
    _at(root, ("gearbox",))["shift_time_s"] = config.shift_time_s * 5.0
    _at(root, ("gearbox",))["clutch_demand_torque_nm"] = config.clutch_demand_torque_nm * 0.5
    edited = load_car_spec(_write(root, tmp_path)).kernel_config()
    assert edited.shift_time_s == config.shift_time_s * 5.0
    assert edited.clutch_demand_torque_nm == config.clutch_demand_torque_nm * 0.5

    # The launch is demand-bound at half the gain, so the edit has to show up in the torque as well
    # as in the config - the second assertion above is not enough on its own.
    peak_rpm = _peak_rpm(config)
    before = gearbox.step_gearbox(config, _state(1.0, 1.0), peak_rpm, 1.0)
    after = gearbox.step_gearbox(edited, _state(1.0, 1.0), peak_rpm, 1.0)
    assert before > 0.5 * config.clutch_demand_torque_nm, (
        "the committed case must not be demand-bound, or this edit proves nothing"
    )
    assert after == pytest.approx(0.5 * config.clutch_demand_torque_nm, rel=1e-12), (
        "half the C9.2.5 gain has to halve the launch torque"
    )

    # A longer shift keeps the clutch open for longer, and the timer is armed at the full
    # duration on
    # the step the gear changes rather than starting from a partial step.
    long_shift = _state()
    gearbox.step_gearbox(edited, long_shift, 10_000.0, 1.0, gearbox.GearRequest.UP)
    assert long_shift[gearbox.GEAR_INDEX] == 2.0
    assert long_shift[gearbox.SHIFT_TIMER_INDEX] == pytest.approx(edited.shift_time_s, rel=1e-12)
    assert _cut_steps(edited, long_shift) == pytest.approx(
        edited.shift_time_s / edited.dt_s, rel=0.005
    )


def test_step_gearbox_refuses_a_state_it_could_not_write_to(config: KernelConfig) -> None:
    """A state buffer that is not a writable float64 vector of the model's size is refused.

    The compiled step is ``boundscheck=False``, so a short or strided buffer writes past its end or
    compiles as a different signature, and a read-only one is an out-of-bounds write. The same
    argument ``kernels.longitudinal.simulate`` makes for ``state`` and ``out``.
    """
    good = _state()
    read_only = _state()
    read_only.flags.writeable = False
    strided = np.empty(gearbox.STATE_SIZE * 2, dtype=np.float64)
    strided[::2] = good
    malformed = (
        [1.0, 1.0, 0.0],
        good.astype(np.float32),
        good.astype(np.int64),
        strided[::2],
        read_only,
        np.zeros((gearbox.STATE_SIZE, 1), dtype=np.float64),
        good[:-1],
    )
    for broken in malformed:
        with pytest.raises(ValueError, match="state"):
            gearbox.step_gearbox(config, broken, 6_000.0, 1.0)  # pyright: ignore[reportArgumentType]


def test_step_gearbox_refuses_a_state_value_it_could_not_use(config: KernelConfig) -> None:
    """The gear is ``{-1, 0, 1..n}`` and integral; the engagement and the timer are ranges.

    ``boundscheck=False`` is about *arrays*; a state value is arithmetic that returns a plausible
    number instead of an error, and the gear is what indexes the ratio table. Neutral and reverse
    are
    *inside* the domain now - C9.7 requires the car to be drivable in reverse and ``channels.yaml``
    already publishes the three states - so the assertions here are about the two ends, the
    integrality, and the two read-only slots, none of which the channel contract can produce.
    """
    top = float(config.gear_ratios.size)
    for gear in (REVERSE, NEUTRAL, 1.0, top):
        state = _state()
        state[gearbox.GEAR_INDEX] = gear
        gearbox.step_gearbox(config, state, 6_000.0, 1.0)

    out_of_range = (
        (gearbox.GEAR_INDEX, top + 1.0),
        (gearbox.GEAR_INDEX, REVERSE - 1.0),
        (gearbox.GEAR_INDEX, 1.5),
        (gearbox.GEAR_INDEX, math.nan),
        (gearbox.GEAR_INDEX, math.inf),
        (gearbox.CLUTCH_INDEX, -0.01),
        (gearbox.CLUTCH_INDEX, 1.01),
        (gearbox.CLUTCH_INDEX, math.inf),
        (gearbox.SHIFT_TIMER_INDEX, -1.0e-9),
        (gearbox.SHIFT_TIMER_INDEX, math.nan),
    )
    for index, value in out_of_range:
        state = _state()
        state[index] = value
        with pytest.raises(ValueError, match="state"):
            gearbox.step_gearbox(config, state, 6_000.0, 1.0)


@pytest.mark.parametrize(
    "command",
    [None, "UP", 5, -1, 1.0, True, math.nan],
    ids=["none", "text", "too-high", "negative", "float", "bool", "nan"],
)
def test_step_gearbox_refuses_a_request_it_could_not_narrow_to_a_code(
    config: KernelConfig,
    command: Any,
) -> None:
    """The request is a driver command, so it has to be one this model knows how to act on.

    A float, a bool and an unknown code are all refused rather than coerced. An unknown code
    reaching
    the compiled step would fall through every comparison; the public boundary prevents that.
    A typo in a
    scenario would silently stop the gearbox changing gear instead of failing - and a float would
    compile a second Numba specialisation alongside the integer one. The enum members and their
    plain integer equivalents are both accepted, because a channel reader hands over an int.
    """
    for good in (*gearbox.GearRequest, *(int(member) for member in gearbox.GearRequest)):
        gearbox.step_gearbox(config, _state(), 6_000.0, 1.0, good)
    with pytest.raises((TypeError, ValueError)):
        gearbox.step_gearbox(config, _state(), 6_000.0, 1.0, command)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("rpm", "throttle"),
    [
        (math.nan, 1.0),
        (math.inf, 1.0),
        (-math.inf, 1.0),
        (np.float32(6_000.0), 1.0),
        (6_000.0, math.nan),
        (6_000.0, 1.5),
        (6_000.0, -0.01),
        (6_000.0, np.float32(1.0)),
    ],
)
def test_step_gearbox_refuses_a_demand_it_could_not_narrow_to_a_float(
    config: KernelConfig,
    rpm: float,
    throttle: float,
) -> None:
    """Two ways a demand fails, neither of which is the torque: not finite, or not a pedal.

    A NaN reaches the compiled step and comes back as a NaN torque, which reads as a finished run. A
    throttle outside ``[0, 1]`` is a value the channel contract cannot produce either, and a
    ``float32`` would compile a second Numba specialisation alongside the ``float`` one.
    """
    state = _state()
    with pytest.raises(ValueError, match=r"ice_rpm|throttle"):
        gearbox.step_gearbox(config, state, rpm, throttle)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("final_drive", 0.0),
        ("final_drive", -1.0),
        ("reverse_ratio", 0.0),
        ("reverse_ratio", -3.883),
        ("reverse_ratio", math.nan),
        ("shift_time_s", 0.0),
        ("shift_time_s", -0.01),
        ("shift_time_s", math.inf),
        ("clutch_demand_torque_nm", 0.0),
        ("clutch_demand_torque_nm", -5_200.0),
        ("clutch_demand_torque_nm", math.nan),
        ("clutch_demand_travel_fraction", 0.0),
        ("clutch_demand_travel_fraction", 1.01),
        ("clutch_demand_travel_fraction", -0.5),
        ("shift_up_rpm", 0.0),
        ("shift_down_rpm", 13_000.0),
    ],
)
def test_step_gearbox_refuses_a_gearbox_configuration_it_could_not_use(
    config: KernelConfig,
    name: str,
    value: float,
) -> None:
    """The gearbox scalars carry the ranges ``car_spec.yaml`` checks them against.

    ``clutch_demand_travel_fraction`` is the span of engagement travel the C9.2.5 gain is applied
    over, so it is a fraction and strictly positive: zero would make the demand undefined, and a
    value above one would ask for more travel than the paddle has. The downshift point has to sit
    below the upshift one, or the driver's two rules overlap and a single rpm could satisfy both -
    and that pair is validated even though it is no longer a trigger, because ``car_spec.yaml``
    still records the shift points as the driver's schedule and a replaced config should not be able
    to invert it silently.
    """
    state = _state()
    with pytest.raises(ValueError, match=name):
        gearbox.step_gearbox(replace(config, **{name: value}), state, 6_000.0, 1.0)


def test_step_gearbox_refuses_a_replaced_ratio_table_it_could_not_index(
    config: KernelConfig,
) -> None:
    """The gear count comes off the ratios, so the array is checked before it is read.

    ``boundscheck=False`` is the reason for the shape, dtype and contiguity rules: the array is
    handed
    to the same caller as the step, and a strided or ``float32`` buffer compiles as a differently
    typed kernel. The ratio values matter for a different reason - a table of zeroes would transmit
    nothing however hard the driver asked.
    """
    ratios = config.gear_ratios
    empty = np.array([], dtype=np.float64)
    malformed = (
        replace(config, gear_ratios=ratios.astype(np.float32)),
        replace(config, gear_ratios=ratios.astype(np.int64)),
        replace(config, gear_ratios=ratios.tolist()),
        replace(config, gear_ratios=ratios.reshape(2, -1)),
        replace(config, gear_ratios=empty),
        replace(config, gear_ratios=np.zeros_like(ratios)),
        replace(config, gear_ratios=-ratios),
        replace(config, gear_ratios=np.full_like(ratios, np.nan)),
    )
    for broken in malformed:
        with pytest.raises(ValueError, match="gear_ratios"):
            gearbox.step_gearbox(broken, _state(), 6_000.0, 1.0)


def test_step_gearbox_refuses_a_replaced_torque_curve_it_could_not_read(
    config: KernelConfig,
) -> None:
    """The engine torque still comes through the engine's own validated composition.

    ``PHASES.md`` P1-T5 has no new torque model, so the gearbox reaches the ICE curve through
    :func:`~f1telemetry.physics.powertrain.step_ice_torque` rather than around it. These are the
    same
    malformed curves that composition refuses, and a gearbox that read past the end of one would
    return a torque nothing downstream could question.
    """
    values = config.torque_nm
    infinite = values.copy()
    infinite[2] = np.inf
    malformed = (
        replace(config, torque_nm=values.astype(np.int64)),
        replace(config, torque_rpm=config.torque_rpm.tolist()),
        replace(config, torque_nm=values.reshape(2, -1)),
        replace(config, torque_nm=np.array([], dtype=np.float64)),
        replace(config, torque_nm=infinite),
        replace(config, turbo_lag_multiplier=0.0),
        replace(config, turbo_lag_multiplier=1.5),
    )
    for broken in malformed:
        with pytest.raises(ValueError, match="step_ice_torque"):
            gearbox.step_gearbox(broken, _state(), 6_000.0, 1.0)


def test_step_gearbox_narrows_numeric_scalars_before_numba(config: KernelConfig) -> None:
    """Equivalent int and float inputs use the same compiled specialisation.

    Without the narrowing, an int read off a config and a float read off one compile two signatures
    of the same step and every later demand has to be checked against a second copy of the gearbox
    table. The request is included because a channel reader hands over an int and a scenario an enum
    member, and those have to reach the kernel as the same type.
    """
    gearbox.step_gearbox(config, _state(), 6_000.0, 1.0)
    compiled = len(COMPILED_STEP.nopython_signatures)
    assert compiled > 0
    integer_config = replace(config, clutch_demand_torque_nm=5_200, shift_time_s=0.04)
    gearbox.step_gearbox(integer_config, _state(), 6_000, 1, int(gearbox.GearRequest.UP))
    gearbox.step_gearbox(config, _state(), 6_000.0, 1.0, gearbox.GearRequest.UP)
    assert len(COMPILED_STEP.nopython_signatures) == compiled


def test_every_request_code_reaches_the_state_it_names(config: KernelConfig) -> None:
    """The enumeration and the codes the compiled step compares against are the same numbers.

    ``step_requested_gear`` is compiled, so it takes an integer code rather than the enum object,
    and
    the only thing keeping those two in step is that this module assigns the values once. If a
    member
    were renumbered the compiled step would act on a different command than the caller asked for - a
    request for neutral selecting a gear, say - and nothing else in the file would see it.
    """
    assert {member.name: int(member) for member in gearbox.GearRequest} == {
        "HOLD": 0,
        "UP": 1,
        "DOWN": 2,
        "NEUTRAL": 3,
        "REVERSE": 4,
    }

    from_gear = 3.0
    expected = {
        gearbox.GearRequest.HOLD: 3.0,
        gearbox.GearRequest.UP: 4.0,
        gearbox.GearRequest.DOWN: 2.0,
        gearbox.GearRequest.NEUTRAL: NEUTRAL,
        gearbox.GearRequest.REVERSE: REVERSE,
    }
    for request, gear in expected.items():
        assert _shifted(config, from_gear, 10_000.0, request)[gearbox.GEAR_INDEX] == gear
