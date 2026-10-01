"""P1-T5: gear progression, shift timing and clutch engagement, as a fixed-step model.

``PHASES.md`` P1-T5 asks for the gearbox - 8 ratios, final drive, shift logic, clutch state -
with trailing throttle and the boost cut both routed through the clutch. This file covers the
four decisions that make those words into arithmetic, and the boundary they are read through:

* **Gear progression.** A launch starts in first, the gear advances when the engine reaches the
  configured ``shift_up_rpm`` and falls when it reaches ``shift_down_rpm``, and the model never
  answers with neutral or reverse. That last part is invariant 7's condition - no reverse under
  positive throttle - and it is checked here by running the real checker over a driven trace,
  not by restating the rule.
* **The shift timer** is a countdown the caller integrates, never a clock the model reads. It is
  armed on a gear change and it freezes the gear while it runs, so one rpm threshold cannot
  ratchet the gearbox up a gear per step.
* **The boost cut is the clutch.** During a shift the engagement is forced to zero and the
  transmitted torque is exactly ``0.0`` - not reduced, not scaled by a leftover engagement - so
  trailing throttle and the cut share one path rather than two.
* **The clutch capacity** is what makes the launch work and what makes trailing throttle pass
  straight through: ``engagement * min(engine torque, capacity)``.

**Wheel speed is not here, on purpose.** ``ice_rpm`` is an input to this step rather than
something it computes, because turning an engine speed into a wheel speed is P1-T6's wheel
rotational state and assembling either into a force is P1-T7's. This step is the last part of the
drive path that is a pure function of numbers the caller hands over.

Every physical number comes from the loaded ``car_spec.yaml``, so this file tunes nothing.
"""

from __future__ import annotations

import math
from dataclasses import replace
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
) -> float:
    """``step_gearbox`` from a fresh caller-owned state at ``gear``, for tests with no history.

    Most of this file cares about the *value* the step produces rather than where the state ended
    up, so it needs a one-shot call at a chosen gear. A helper is the honest way to say that: a
    fresh buffer, one step, no shared history. Tests that are about the state write it themselves.
    """
    return gearbox.step_gearbox(config, _state(gear, clutch), ice_rpm, throttle)


def _ratio(config: KernelConfig, gear: float) -> float:
    """The configured total reduction of ``gear``, read off the table in the test's own arithmetic.

    Deliberately *not* a call into the model: there is no public ratio helper to call, because a
    compiled one would take an arbitrary gear with ``boundscheck=False`` and read past the end of a
    short table. Writing the lookup here keeps the reference independent of the implementation and
    documents the 1-based-gear/0-based-table rule in the one place a reader looks for it.
    """
    return float(config.gear_ratios[int(gear) - 1]) * config.final_drive


def _gearbox_output_nm(
    config: KernelConfig,
    gear: float,
    engine_torque_nm: float,
    throttle: float,
    clutch: float = 1.0,
) -> float:
    """The expected output: **ratio first, then the clutch ceiling**.

    ``PLAN.md`` section 6 draws ``torque_curve -> gearbox -> clutch -> differential``, so the gear
    reduction happens before the clutch sees the torque and the capacity is differential-side
    figure. Written out here rather than reached through the model, so a test comparing against it
    is comparing against the documented arithmetic and not the implementation restated - and so that
    reversing the order in the module does not silently move the reference with it.
    """
    ratio = _ratio(config, gear)
    reduced_nm = throttle * engine_torque_nm * ratio
    return gearbox.clutch_output_torque_nm(reduced_nm, config.clutch_torque_capacity_nm, clutch)


def _sweep(
    config: KernelConfig,
    rpms: list[float],
    *,
    gear: float = 1.0,
    throttle: float = 1.0,
) -> list[tuple[int, float]]:
    """One step per rpm from one caller-owned state, holding the clutch closed.

    Used only where the caller holds rpm steady across steps. The shift tests step the rpm *down*
    after an upshift, which is what the drivetrain would actually see and is why rpm is an argument
    rather than something this file reads out of a gear.
    """
    state = _state(gear, 1.0)
    seen: list[tuple[int, float]] = []
    for rpm in rpms:
        torque = gearbox.step_gearbox(config, state, rpm, throttle)
        seen.append((int(state[gearbox.GEAR_INDEX]), torque))
    return seen


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
        description="gearbox progression driven by the P1-T5 model",
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
    gearbox.step_gearbox(config, state, 6_000.0, 1.0)
    gearbox.clutch_output_torque_nm(300.0, config.clutch_torque_capacity_nm, 1.0)
    gearbox.step_gear(1, 6_000.0, 8, config.shift_up_rpm, config.shift_down_rpm)
    for function in (COMPILED_STEP, gearbox.clutch_output_torque_nm, gearbox.step_gear):
        name = function.py_func.__name__
        assert len(function.signatures) >= 1, f"{name} never compiled a signature"
        options = dict(function.targetoptions)
        for option, expected in EXPECTED_OPTIONS.items():
            assert options[option] == expected, (name, option)


def test_the_state_is_the_callers_and_the_step_writes_the_gear_and_the_timer_only(
    config: KernelConfig,
) -> None:
    """One caller-owned buffer, and the step writes two of its three slots.

    Caller-owned is the whole point: a run steps one array over and over with no allocation and no
    second copy to keep in step, which is the rule ``kernels.longitudinal`` already follows for
    ``out``. What the step *writes* is the gear and the shift timer; the clutch engagement it reads.
    The engagement check is the one that matters here - if the boost cut ever wrote zero back, a
    caller ramping a launch would have its input silently destroyed on the first shift.
    """
    state = gearbox.initial_state(gear=1.0, clutch_engagement=0.6)
    assert state.dtype == np.float64
    assert state.shape == (gearbox.STATE_SIZE,)
    assert state[gearbox.GEAR_INDEX] == 1.0
    assert state[gearbox.CLUTCH_INDEX] == 0.6
    assert state[gearbox.SHIFT_TIMER_INDEX] == 0.0

    torque = gearbox.step_gearbox(config, state, 6_000.0, 1.0)
    assert torque == pytest.approx(
        _gearbox_output_nm(config, 1.0, powertrain.step_ice_torque(config, 6_000.0), 1.0, 0.6),
        rel=1e-12,
    )
    assert state[gearbox.GEAR_INDEX] == 1.0
    assert state[gearbox.CLUTCH_INDEX] == 0.6, "the step read the engagement and left it alone"
    assert np.all(state >= 0.0)


def test_the_gear_advances_only_at_the_configured_upshift_threshold(config: KernelConfig) -> None:
    """One rpm below the threshold the gear holds; at the threshold it advances by one.

    Both sides of the comparison are checked against the same state, so the assertion is about
    the threshold and not about where the previous test happened to leave the gearbox. The
    threshold itself is pinned from the file, which is what makes this a test of the model rather
    than a restatement of it.
    """
    assert config.shift_up_rpm == pytest.approx(12_500.0), "the committed shift point moved"

    just_below = _sweep(config, [config.shift_up_rpm - 1.0] * 4)
    assert [gear for gear, _ in just_below] == [1, 1, 1, 1]

    # At the threshold the gear advances, and the shift cut takes the torque to zero on the step
    # the change happens - the clutch opens at the moment of the change, not one step later.
    at_threshold = _sweep(config, [config.shift_up_rpm])
    assert at_threshold[0][0] == 2
    assert at_threshold[0][1] == 0.0

    # And it settles there while the engine sits between the two thresholds, which is what a
    # drivetrain actually sees: the rpm falls as the taller gear takes over. A gearbox that kept
    # climbing would be ratcheting, and running past the whole shift time is what rules that out -
    # a short tail would only ever see the cut and could not tell a held gear from a ratcheting
    # one.
    steps = int(config.shift_time_s / config.dt_s) + 4
    settled = _sweep(config, [config.shift_up_rpm] + [10_000.0] * steps)
    assert [gear for gear, _ in settled] == [2] * (steps + 1)
    assert settled[0][1] == 0.0, "the step the gear changes is itself the cut"
    assert max(torque for _, torque in settled) > 0.0, "the shift has to end and deliver torque"


def test_the_gear_falls_only_at_the_configured_downshift_threshold(config: KernelConfig) -> None:
    """The downshift threshold is a different number from the upshift one, and it is honoured.

    A drop is what a real gearbox does when the engine is dragged below the downshift point, and
    it is the other half of the same comparison: a model that only knows about upshifts would
    sit in a gear whose ratio is too short and never recover.
    """
    assert config.shift_down_rpm == pytest.approx(8_000.0), "the committed shift point moved"
    assert config.shift_down_rpm < config.shift_up_rpm

    held = _sweep(config, [config.shift_down_rpm + 1.0] * 3, gear=3.0)
    assert [gear for gear, _ in held] == [3, 3, 3]

    # From the same starting gear, at the threshold itself, one gear down and no further: a single
    # step can only move one gear, so this cannot be confused with a drop past the threshold.
    dropped = _sweep(config, [config.shift_down_rpm], gear=3.0)
    assert dropped[0][0] == 2
    assert dropped[0][1] == 0.0, "a downshift cuts torque the same way an upshift does"


def test_a_shift_cuts_the_torque_for_exactly_the_configured_shift_time(
    config: KernelConfig,
) -> None:
    """The cut is a hard zero for ``shift_time_s`` and the gear is frozen throughout it.

    Two claims, and the second is what makes the first mean something. A hard zero is only the
    boost cut if the gear cannot move while the cut is running: otherwise the rpm that triggered
    the shift triggers the next one and the torque is never delivered at all.

    The count is allowed one step either side of ``shift_time_s / dt_s`` because the countdown is
    accumulated in floating point - ``0.04`` is not representable - so the final subtraction can
    land a step early or late. What is pinned is that the cut is neither one step nor a hundred.
    """
    dt_s = config.dt_s
    budget = config.shift_time_s / dt_s
    assert budget == pytest.approx(400.0), "the committed shift time or step changed"

    state = _state(1.0, 1.0)
    cut_steps = 0
    for _ in range(int(budget) + 4):
        rpm = config.shift_up_rpm if cut_steps == 0 else 10_000.0
        torque = gearbox.step_gearbox(config, state, rpm, 1.0)
        if torque == 0.0:
            cut_steps += 1
        else:
            break
        assert state[gearbox.GEAR_INDEX] == 2.0, "the gear moved before its shift finished"

    assert cut_steps in (int(budget), int(budget) + 1)
    # A closed clutch at full throttle between the two thresholds delivers torque; without the cut
    # this is the step that would have produced it, so the assertion above is not vacuous.
    resumed = gearbox.step_gearbox(config, state, 10_000.0, 1.0)
    assert resumed > 0.0
    assert resumed == pytest.approx(
        _gearbox_output_nm(config, 2.0, powertrain.step_ice_torque(config, 10_000.0), 1.0),
        rel=1e-12,
    ), "a fully engaged clutch must not scale the torque below what the engine makes"


def test_the_boost_cut_is_the_clutch_and_not_a_second_multiplier(config: KernelConfig) -> None:
    """Zero torque through the cut whatever the engagement was, and zero only through the cut.

    ``PHASES.md`` P1-T5 asks for trailing throttle and the boost cut to route through the clutch,
    which is a claim about there being *one* path. It shows up as: with the clutch shut, a fully
    open throttle at high rpm delivers exactly nothing, and with the clutch open by any amount the
    delivery is that same nothing - not a fraction of it.
    """
    state = _state(1.0, 1.0)
    gearbox.step_gearbox(config, state, config.shift_up_rpm, 1.0)
    assert state[gearbox.GEAR_INDEX] == 2.0
    assert state[gearbox.SHIFT_TIMER_INDEX] > 0.0

    for engagement in (0.0, 0.25, 0.5, 1.0):
        fresh = _state(2.0, engagement)
        fresh[gearbox.SHIFT_TIMER_INDEX] = state[gearbox.SHIFT_TIMER_INDEX]
        assert gearbox.step_gearbox(config, fresh, 10_000.0, 1.0) == 0.0


def test_gear_progression_satisfies_invariant_7(config: KernelConfig, spec: CarSpec) -> None:
    """The real checker, over a driven trace: monotonic, in range, and no reverse.

    Invariant 7 is the condition this task has to satisfy, so it is run here on the gear the
    model actually produces rather than on a supplied fixture. The rpm sweep only ever rises,
    which is the situation the invariant describes - a gearbox that drops a gear under
    acceleration is correct behaviour and is not what ``monotonic progression`` is about.
    """
    top = len(spec.gear_ratios)
    rpms = [
        config.idle_rpm + 250.0 * index
        for index in range(int((config.rev_limit_rpm - config.idle_rpm) / 250.0) + 1)
    ]
    trace = _sweep(config, rpms)
    gears = [gear for gear, _ in trace]

    assert gears == sorted(gears), "the gearbox went backwards under a rising rpm"
    assert min(gears) >= 1, "the gearbox answered with neutral or reverse"
    assert max(gears) <= top
    assert all(torque >= 0.0 for _, torque in trace), "drive torque must not reverse a launch"

    violations = check_gearbox_progression(_record(trace, throttle=1.0), spec)
    assert not violations, f"invariant 7 rejected a gearbox the model drove: {violations}"


def test_the_gearbox_walks_to_top_gear_and_stays_there(config: KernelConfig) -> None:
    """A constant rpm above the threshold has to reach the last gear and then hold it.

    This is the shift-timer's other job. A ratcheting gearbox - one that upshifts again on the
    step the timer expires - would reach top gear and then oscillate, and no single-step test
    would notice.
    """
    top = int(config.gear_ratios.size)
    budget = int(config.shift_time_s / config.dt_s)
    state = _state(1.0, 1.0)
    gears = [int(state[gearbox.GEAR_INDEX])]
    torques: list[float] = []
    for _ in range(budget * (top + 2)):
        torques.append(gearbox.step_gearbox(config, state, config.shift_up_rpm, 1.0))
        gears.append(int(state[gearbox.GEAR_INDEX]))

    assert gears[-1] == top
    assert gears == sorted(gears)
    # Every shift costs a cut, so a gearbox that reached top by ratcheting would show more cuts
    # than shifts: exactly one per gear, plus a tail of torque once the timer is exhausted.
    assert torques.count(0.0) <= top * (budget + 1)
    assert max(torques[-budget - 2 :]) > 0.0


def test_the_clutch_capacity_limits_the_launch_and_the_engagement_scales_what_gets_through(
    config: KernelConfig,
) -> None:
    """``min(ratio-reduced torque, capacity * engagement)``, in both regimes it has to have.

    The launch is the case the capacity exists for: the gear train offers its whole torque and the
    driveline still takes only what the plates can hold at that pedal position, so a partly closed
    clutch - not the engine - is what limits the car off the line.

    **The committed capacity has to be reachable somewhere, and to leave ratios visible elsewhere.**
    At the 330 Nm engine peak the box offers 3 844 Nm in first down to 1 618 Nm in eighth, and the
    committed 3 000 Nm sits inside that band: gears 1-3 are clutch-limited at full engagement and
    gears 4-8 pass the gear train's own torque through. Both halves are asserted below, because a
    capacity above the first-gear maximum would be a ceiling no gear can select - the dead-clamp
    failure the ordering correction exists to fix - and one below the eighth-gear figure would
    flatten all eight gears to the same number.
    """
    capacity = config.clutch_torque_capacity_nm
    idle = config.idle_rpm
    peak_rpm = float(config.torque_rpm[int(np.argmax(config.torque_nm))])
    idle_torque = powertrain.step_ice_torque(config, idle)
    peak_torque = powertrain.step_ice_torque(config, peak_rpm)
    first = _ratio(config, 1.0)
    smallest = _ratio(config, float(config.gear_ratios.size))
    largest = peak_torque * first
    smallest_output = peak_torque * smallest

    assert idle_torque > 0.0, "a launch needs an engine that makes torque at idle"
    assert smallest_output < capacity < largest, (
        f"the capacity of {capacity} Nm must sit between what eighth gear offers "
        f"({smallest_output:.0f} Nm) and what first offers ({largest:.0f} Nm), so that it limits "
        "the low gears and leaves the high gears observable"
    )

    # Gears the capacity can reach are limited by it, and the ones it cannot are passed through.
    clamped, transparent = [], []
    for gear in range(1, int(config.gear_ratios.size) + 1):
        offered = peak_torque * _ratio(config, float(gear))
        transmitted = _driveline_torque_nm(config, float(gear), peak_rpm, 1.0)
        assert transmitted == pytest.approx(min(offered, capacity), rel=1e-12), (
            f"gear {gear} transmitted {transmitted} for an offer of {offered:.1f} Nm"
        )
        (clamped if offered > capacity else transparent).append(gear)
    assert clamped == [1, 2, 3], "the committed capacity should limit the three lowest gears"
    assert transparent == [4, 5, 6, 7, 8], (
        "the committed capacity should leave the top half of the box observable"
    )

    # A shut clutch transmits nothing at all, whatever the gear would have multiplied.
    assert _driveline_torque_nm(config, 1.0, idle, 1.0, clutch=0.0) == 0.0

    # Part engagement lowers the ceiling the plates present, so the launch torque is that ceiling -
    # in differential-side Nm, with no gear left to multiply afterwards. This is the discriminator
    # between the two orders: clamping after the ratio gives exactly engagement*capacity, whereas
    # clamping before it would give engagement*capacity*ratio, 11.65 times larger in first gear.
    for engagement in (0.05, 0.1, 0.25):
        ceiling = engagement * capacity
        assert ceiling < largest, "this case must be ceiling-bound to be a test"
        launch = _driveline_torque_nm(config, 1.0, peak_rpm, 1.0, clutch=engagement)
        assert launch == pytest.approx(ceiling, rel=1e-12)
        assert launch != pytest.approx(ceiling * first, rel=1e-6), (
            "the ceiling was multiplied by the gear ratio, so the capacity was applied before the "
            "reduction"
        )

    # The primitive is the same rule stated once, with nothing hidden behind the step. The committed
    # capacity is below what first gear offers, so even a closed clutch is limited there.
    assert gearbox.clutch_output_torque_nm(largest, capacity, 1.0) == pytest.approx(
        capacity, rel=1e-12
    )
    assert gearbox.clutch_output_torque_nm(smallest_output, capacity, 1.0) == pytest.approx(
        smallest_output, rel=1e-12
    ), "a gear the capacity can carry must pass through unmodulated"
    assert gearbox.clutch_output_torque_nm(largest, capacity, 0.0) == 0.0
    assert gearbox.clutch_output_torque_nm(0.0, capacity, 1.0) == 0.0
    assert gearbox.clutch_output_torque_nm(10.0, capacity, 1.0) == 10.0


def test_a_lower_clutch_capacity_limits_the_output_in_every_gear(config: KernelConfig) -> None:
    """A capacity below what a gear delivers caps that gear, and the cap does not scale with it.

    This is the order made falsifiable. ``PLAN.md`` section 6 puts the clutch *after* the gearbox,
    so the capacity is differential-side and the clamp happens on the already-reduced torque. Clamp
    first and multiply afterwards - the order the boundary review corrected - and a 1000 Nm capacity
    would pass 11 649 Nm in first gear, because the 1000 was applied to the engine's 330 Nm and
    then multiplied by 11.65. The error is a constant factor that grows with the ratio, so a test
    that only checks one gear cannot see it; this checks the whole box against one ceiling.

    The capacity used here is a replaced config, not the committed one, because the committed
    number deliberately sits above every gear's output and is therefore never the limit on track.
    """
    rpm = 10_000.0
    engine = powertrain.step_ice_torque(config, rpm)
    reduced = replace(config, clutch_torque_capacity_nm=1_000.0)
    first = _ratio(config, 1.0)

    # First gear wants more than the clutch will take, so the clutch is the limit and the ratio
    # does not make it pass more.
    assert engine * first > reduced.clutch_torque_capacity_nm, "this case must be ceiling-bound"
    eighth = _ratio(config, 8.0)
    assert _driveline_torque_nm(reduced, 1.0, rpm, 1.0) == pytest.approx(1_000.0, rel=1e-12)
    assert _driveline_torque_nm(reduced, 1.0, rpm, 1.0) == pytest.approx(
        gearbox.clutch_output_torque_nm(engine * first, reduced.clutch_torque_capacity_nm, 1.0),
        rel=1e-12,
    )

    # The same capacity caps a mid gear at the same number: the ceiling is the capacity, not the
    # capacity times something.
    for gear in range(1, int(config.gear_ratios.size) + 1):
        ratio = _ratio(config, float(gear))
        torque = _driveline_torque_nm(reduced, float(gear), rpm, 1.0)
        assert torque == pytest.approx(min(engine * ratio, 1_000.0), rel=1e-12), (
            f"gear {gear} did not clamp against the clutch capacity independently of the ratio"
        )

    # And the gear still matters where the clutch is not the limit. Eighth gear's 4.90 ratio makes
    # 1618 Nm, which is above the 1000 Nm ceiling, so a *lower* capacity is what leaves the ratio
    # visible: at 2000 Nm first gear clamps and eighth does not.
    partial = replace(config, clutch_torque_capacity_nm=2_000.0)
    assert _driveline_torque_nm(partial, 1.0, rpm, 1.0) == pytest.approx(2_000.0, rel=1e-12)
    assert _driveline_torque_nm(partial, 8.0, rpm, 1.0) == pytest.approx(
        engine * eighth, rel=1e-12
    ), "the ratio stopped reaching the output where the clutch is not the limit"

    # A capacity the box cannot reach changes nothing at all: every gear passes its own torque.
    generous = replace(config, clutch_torque_capacity_nm=10.0 * first * engine)
    for gear in (1.0, 5.0, 8.0):
        assert _driveline_torque_nm(generous, gear, rpm, 1.0) == pytest.approx(
            engine * _ratio(config, gear), rel=1e-12
        ), f"a capacity above every gear changed the output in gear {int(gear)}"


def test_trailing_throttle_passes_through_the_clutch_unmodulated(config: KernelConfig) -> None:
    """Off the throttle the engine decides the torque, not the clutch.

    Trailing throttle is the reason the model needs the engine torque scaled by a pedal before
    the clutch sees it: a driver part-way down the throttle at 12 000 rpm wants that fraction of
    the engine's torque through a closed clutch, and a model that clamped first would hand back
    the capacity instead. The expectation is the documented arithmetic - throttle fraction, engine
    torque, then the gear - written out here rather than taken from the model.
    """
    rpm = 12_000.0
    gear = 3.0
    engine = powertrain.step_ice_torque(config, rpm)
    ratio = _ratio(config, gear)
    assert engine > 0.0

    for throttle in (0.0, 0.05, 0.2, 0.5):
        torque = _driveline_torque_nm(config, gear, rpm, throttle)
        assert torque == pytest.approx(throttle * engine * ratio, rel=1e-12), (
            f"throttle {throttle} is not reaching the transmission unmodulated"
        )
    assert _driveline_torque_nm(config, gear, rpm, 0.0) == 0.0

    # And the gear still follows the engine speed, because a trailing throttle does not stop a
    # downshift from happening when the rpm drops far enough.
    dropped = _sweep(config, [config.shift_down_rpm], gear=3.0)
    assert dropped[0][0] == 2


def test_each_gear_multiplies_the_transmitted_torque_by_its_own_ratio(config: KernelConfig) -> None:
    """Each gear reduces the engine torque by its own ``ratio * final_drive``.

    This is the test that a gearbox which selected a gear and then returned the same engine torque
    for all eight cannot pass: every other test in this file looks at *when* the gear changes or at
    whether the torque is zero, and all of those are satisfied by a model that ignores the ratio
    table entirely. Here every gear is checked against its own row of the table, so a model reading
    only ``len(gear_ratios)`` - using the table as a gear count and nothing else - fails on gear 1.

    The reference is the documented arithmetic, not a call into the model: there is no public ratio
    helper to call, because a compiled one would take an arbitrary gear with ``boundscheck=False``
    and read past the end of a short table, so the lookup is written out here. Throttle is held at
    half, which keeps every gear's output under the committed 3 000 Nm capacity - the clamp has its
    own test, and letting it bind here would blur which of the two effects is being measured.
    """
    rpm = 10_000.0
    throttle = 0.5
    engine = powertrain.step_ice_torque(config, rpm)
    for gear in range(1, int(config.gear_ratios.size) + 1):
        ratio = _ratio(config, float(gear))
        assert engine * ratio * throttle < config.clutch_torque_capacity_nm, (
            f"gear {gear} is clutch-limited at this throttle, so this test would be measuring the "
            "clamp rather than the ratio"
        )
        assert _driveline_torque_nm(config, float(gear), rpm, throttle) == pytest.approx(
            engine * ratio * throttle, rel=1e-12
        ), f"gear {gear} did not reduce the torque by its own ratio"

    # The ratios fall across the box, so the output has to fall too: a model returning a
    # gear-independent torque cannot satisfy this even if the per-gear arithmetic above is faked.
    # At the same half throttle, where no gear is clamped.
    outputs = [_driveline_torque_nm(config, float(gear), rpm, throttle) for gear in range(1, 9)]
    assert outputs == sorted(outputs, reverse=True)
    assert outputs[0] / outputs[-1] == pytest.approx(
        (float(config.gear_ratios[0]) * config.final_drive)
        / (float(config.gear_ratios[-1]) * config.final_drive),
        rel=1e-12,
    )


def test_the_caller_owned_engagement_survives_a_shift_and_the_caller_may_change_it(
    config: KernelConfig,
) -> None:
    """The boost cut is this step's decision; the engagement in the buffer is the caller's.

    Caller-owned means the caller supplies and retains the engagement, and the step may change it
    between calls. This drives that in both directions: a part-engaged launch crosses the upshift
    point, and through the whole shift - including the step the gear changes on - the buffer keeps
    the value the caller put there and the torque is exactly zero. When the shift ends the same
    engagement is still in place and torque resumes immediately, with no re-arming.

    Writing the cut back into the slot would fail the second half: a launch would be silently reset
    to a shut clutch on its first upshift and could not move again without the caller noticing why.
    """
    engagement = 0.45
    state = gearbox.initial_state(gear=1.0, clutch_engagement=engagement)
    budget = int(config.shift_time_s / config.dt_s)

    # The launch itself transmits through a partly closed clutch, below what the engine makes.
    launch = gearbox.step_gearbox(config, state, config.shift_down_rpm, 1.0)
    assert launch > 0.0
    assert launch == pytest.approx(
        _gearbox_output_nm(
            config, 1.0, powertrain.step_ice_torque(config, config.shift_down_rpm), 1.0, engagement
        ),
        rel=1e-12,
    )

    # The upshift cuts the torque to nothing, and the engagement is untouched throughout.
    assert gearbox.step_gearbox(config, state, config.shift_up_rpm, 1.0) == 0.0
    for _ in range(budget + 2):
        torque = gearbox.step_gearbox(config, state, 10_000.0, 1.0)
        if torque == 0.0:
            assert state[gearbox.CLUTCH_INDEX] == engagement, (
                "the boost cut wrote the engagement back into the caller's state"
            )
        else:
            break
    assert state[gearbox.GEAR_INDEX] == 2.0
    assert state[gearbox.CLUTCH_INDEX] == engagement, "the shift consumed the caller's engagement"

    # And the same engagement is still in force the moment the shift is over.
    resumed = gearbox.step_gearbox(config, state, 10_000.0, 1.0)
    assert resumed == pytest.approx(
        _gearbox_output_nm(
            config, 2.0, powertrain.step_ice_torque(config, 10_000.0), 1.0, engagement
        ),
        rel=1e-12,
    )

    # The caller may then change it, which is what a driver lifting off mid-shift looks like.
    state[gearbox.CLUTCH_INDEX] = 0.9
    assert gearbox.step_gearbox(config, state, 10_000.0, 1.0) == pytest.approx(
        _gearbox_output_nm(config, 2.0, powertrain.step_ice_torque(config, 10_000.0), 1.0, 0.9),
        rel=1e-12,
    )


def test_the_gear_ratios_and_the_clutch_engagement_are_read_off_the_kernel_config(
    config: KernelConfig,
) -> None:
    """Replacing a config field moves the answer that field names, and nothing else.

    A ratio or a capacity written into the module would pass every test above and fail here, which
    is what lets this file claim it tunes nothing. The gear count is read from the ratios, so
    replacing them with a shorter set has to shorten the gearbox too.
    """
    assert config.clutch_torque_capacity_nm > 0.0
    assert float(config.gear_ratios.size) == 8.0
    idle = config.idle_rpm
    # Throttle is held down so that a doubled final drive cannot push any of these gears into the
    # clutch ceiling: the point of each edit below is that the ratio reaches the output, and a clamp
    # in the way would turn a 2x expectation into two identical clamped numbers.
    throttle = 0.25

    # Neither a ratio nor the final drive written into the module would be caught by any shift test
    # above, because none of those looks at the output torque per gear. These do.
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

    # A shift point moved in the file moves the shift. A threshold written into the module would
    # keep shifting at 12 500 rpm no matter what the config said.
    earlier_up = replace(config, shift_up_rpm=config.shift_up_rpm - 2_000.0)
    held = _state()
    gearbox.step_gearbox(earlier_up, held, config.shift_up_rpm - 2_000.0 - 1.0, 1.0)
    assert held[gearbox.GEAR_INDEX] == 1.0
    shifted = _state()
    gearbox.step_gearbox(earlier_up, shifted, config.shift_up_rpm - 2_000.0, 1.0)
    assert shifted[gearbox.GEAR_INDEX] == 2.0

    shorter = replace(config, gear_ratios=config.gear_ratios[:2].copy())
    short_state = _state(2.0, 1.0)
    assert gearbox.step_gearbox(shorter, short_state, config.shift_up_rpm, 1.0) > 0.0
    assert short_state[gearbox.GEAR_INDEX] == 2.0, "the last gear must not upshift into nothing"


def test_editing_the_car_spec_changes_the_gearbox_with_no_code_edit(
    tmp_path: Path,
    repo: Path,
    config: KernelConfig,
) -> None:
    """P1-T1b for the gearbox: the file is the only place a coefficient lives.

    One edit per quantity the model consumes - an upshift point, a downshift point, a shift time
    and the clutch capacity - and each has to move the answer it is supposed to move. A constant
    written into the module would pass every other test in this file.
    """
    idle = config.idle_rpm
    closed = gearbox.step_gearbox(config, _state(1.0, 1.0), idle, 1.0)
    assert closed > 0.0

    root = _document(repo)
    _at(root, ("gearbox",))["shift_up_rpm"] = config.shift_up_rpm - 1_000.0
    _at(root, ("gearbox",))["shift_down_rpm"] = config.shift_down_rpm - 500.0
    _at(root, ("gearbox",))["shift_time_s"] = config.shift_time_s * 5.0
    _at(root, ("gearbox",))["clutch_torque_capacity_nm"] = config.clutch_torque_capacity_nm * 0.5
    edited = load_car_spec(_write(root, tmp_path)).kernel_config()
    assert edited.shift_up_rpm == config.shift_up_rpm - 1_000.0
    assert edited.shift_time_s == config.shift_time_s * 5.0
    assert edited.clutch_torque_capacity_nm == config.clutch_torque_capacity_nm * 0.5

    # The launch is capacity-bound at half the original capacity, so the edit has to show up in
    # the torque as well as in the config - the second assertion above is not enough on its own.
    peak_rpm = float(config.torque_rpm[int(np.argmax(config.torque_nm))])
    engagement = 0.1
    before = gearbox.step_gearbox(config, _state(1.0, engagement), peak_rpm, 1.0)
    after = gearbox.step_gearbox(edited, _state(1.0, engagement), peak_rpm, 1.0)
    assert after == pytest.approx(0.5 * before, rel=1e-12), "less capacity, less launch torque"

    # A longer shift keeps the clutch open for longer, and the timer is armed at the full duration
    # on the step the gear changes rather than starting from a partial step.
    long_shift = _state()
    gearbox.step_gearbox(edited, long_shift, edited.shift_up_rpm, 1.0)
    assert long_shift[gearbox.GEAR_INDEX] == 2.0
    assert long_shift[gearbox.SHIFT_TIMER_INDEX] == pytest.approx(edited.shift_time_s, rel=1e-12)
    assert closed > 0.0


def test_step_gearbox_refuses_a_state_it_could_not_write_to(config: KernelConfig) -> None:
    """A state buffer that is not a writable float64 vector of the model's size is refused.

    The compiled step is ``boundscheck=False``, so a short or strided buffer writes past its end
    or compiles as a different signature, and a read-only one is an out-of-bounds write. The same
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
    """Gear, engagement and shift timer are all ranges, and the buffer is not trusted for them.

    ``boundscheck=False`` is about *arrays*; a state value is arithmetic that returns a plausible
    number instead of an error. A gear of zero is neutral, a negative engagement would drive the
    transmission backwards through the clutch, and a negative shift timer is a deadline rather
    than a duration.
    """
    out_of_range = (
        (gearbox.GEAR_INDEX, 0.0),
        (gearbox.GEAR_INDEX, -1.0),
        (gearbox.GEAR_INDEX, float(config.gear_ratios.size) + 1.0),
        (gearbox.GEAR_INDEX, math.nan),
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

    A NaN reaches the compiled step and comes back as a NaN torque, which reads as a finished
    run. A throttle outside ``[0, 1]`` is a value the channel contract cannot produce either, and
    a ``float32`` would compile a second Numba specialisation alongside the ``float`` one.
    """
    state = _state()
    with pytest.raises(ValueError, match=r"ice_rpm|throttle"):
        gearbox.step_gearbox(config, state, rpm, throttle)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("clutch_torque_capacity_nm", 0.0),
        ("clutch_torque_capacity_nm", -1.0),
        ("clutch_torque_capacity_nm", math.nan),
        ("shift_up_rpm", 0.0),
        ("shift_up_rpm", -1.0),
        ("shift_down_rpm", 0.0),
        ("shift_down_rpm", math.nan),
        ("shift_down_rpm", 13_000.0),
        ("shift_time_s", 0.0),
        ("shift_time_s", -0.01),
        ("shift_time_s", math.inf),
    ],
)
def test_step_gearbox_refuses_a_gearbox_configuration_it_could_not_use(
    config: KernelConfig,
    name: str,
    value: float,
) -> None:
    """The gearbox scalars carry the ranges ``car_spec.yaml`` checks them against.

    The downshift point has to sit below the upshift one, or the two rules overlap and a single
    rpm can satisfy both. ``shift_time_s`` carries strict positivity rather than a sign rule, which
    the rest of the model depends on: zero would remove the freeze that stops an rpm sitting on the
    upshift point from advancing the box a gear per step, so the no-ratchet guarantee the timer
    exists to provide would be gone. ``car_spec.yaml`` refuses it too, so a file edit and a replaced
    config agree.
    """
    state = _state()
    with pytest.raises(ValueError, match=name):
        gearbox.step_gearbox(replace(config, **{name: value}), state, 6_000.0, 1.0)


def test_step_gearbox_refuses_a_replaced_ratio_table_it_could_not_index(
    config: KernelConfig,
) -> None:
    """The gear count comes off the ratios, so the array is checked before it is read.

    ``boundscheck=False`` is the reason for the shape, dtype and contiguity rules: the array is
    handed to the same caller as the step, and a strided or ``float32`` buffer compiles as a
    differently-typed kernel. The ratio values matter for a different reason - the model's
    progression assumes a shorter gear is a higher number, which a table of zeroes cannot be.
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
    same malformed curves that composition refuses, and a gearbox that read past the end of one
    would return a torque nothing downstream could question.
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

    Without the narrowing, an int read off a config and a float read off one compile two
    signatures of the same step and every later demand has to be checked against a second copy of
    the gearbox table.
    """
    gearbox.step_gearbox(config, _state(), 6_000.0, 1.0)
    compiled = len(COMPILED_STEP.nopython_signatures)
    assert compiled > 0
    integer_config = replace(config, clutch_torque_capacity_nm=600, shift_up_rpm=12_500)
    gearbox.step_gearbox(integer_config, _state(), 6_000, 1)
    gearbox.step_gearbox(config, _state(), 6_000.0, 1.0)
    assert len(COMPILED_STEP.nopython_signatures) == compiled
