"""The headless fast path: same bytes as the reference runner, faster to get them.

``docs/latency.md`` profiles :func:`~f1telemetry.testing.scenarios.run_scenario` at ~0.28x real time
and puts 98.9% of that in runner overhead rather than in the kernel.
:mod:`f1telemetry.testing.headless` is the response: the same kernel, the same drivetrain and the
same record assembler, scheduled so the per-step Python validation is done once per segment
instead of once per step.

That claim is only worth anything if it is checked against the runner it replaces, so this file is
mostly equality:

* **Byte-identical output on a straight-line scenario and on a steering one.**
  ``accelerate_to_speed`` is the profiled case: a clutch-slip launch, a declared engine speed, a
  driver-requested gear ladder and a top-gear tail. ``steady_state_circle`` is neutral and steers,
  so it exercises the Ackermann split, lateral load transfer, bump steer and the suspension
  diagnostics that a straight-line run never reaches. The assertion is on ``trace.tobytes()``, and
  then on every other array the run carries and on the assembled record, because a fast path that
  matched on the trace while quietly disagreeing about a tyre temperature would be a worse bug than
  a slow one.
* **The manifest attaches, and attaching it changes nothing.** Same contract as the reference
  runner: carried through untouched, and no effect on a byte of the run it describes.
* **The speedup is measured, and the measurement is the point.** Both paths are warmed past
  Numba's JIT first, then each is timed several times and the *fastest* run of each is compared -
  the minimum is the estimator least contaminated by whatever else the machine was doing, which on
  a shared box is the difference between a real 5x and a meaningless 1.2x. The factor is asserted
  only to exceed 1.0, deliberately: the honest number is reported, and it is short of the P6 50x
  real-time gate that ``docs/latency.md`` says so in as many words.
* **Refusals and determinism hold.** The headless path makes the reference runner's refusals through
  the same helpers, so one refused plan is checked here rather than restated; and two headless runs
  of one plan are byte-identical, which is the reproducibility claim a faster schedule has to keep.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import pytest

from f1telemetry.testing.headless import run_headless
from f1telemetry.testing.run_manifest import RunManifest
from f1telemetry.testing.scenarios import (
    DrivetrainTrace,
    Scenario,
    ScenarioSegment,
    build_scenarios,
    run_scenario,
)

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import CarSpec, KernelConfig
    from f1telemetry.testing.scenarios import ScenarioRun

pytestmark = [pytest.mark.kernel]

# The two scenarios the equality claim is made on. The first is the one `docs/latency.md` profiles
# and the only long one here; the second is short, steers, and is the reason this file's wall time
# is dominated by the first. There is no tolerance on either: they are byte comparisons, not
# performance bands.
STRAIGHT_SCENARIO: Final[str] = "accelerate_to_speed"
STEERING_SCENARIO: Final[str] = "steady_state_circle"

# Timed repetitions per path, after one warm-up of each. Three keeps the test's wall time sane on a
# 7 s run while still taking the minimum of more than one sample, which is what makes the factor
# below meaningful on a machine that is not otherwise idle.
TIMED_REPEATS: Final[int] = 3

# The threshold is 1.0 and not the 50x the P6 gate names, because the honest measured factor is a
# few times over and writing a larger bound here would be a test that fails for the wrong reason.
# `docs/latency.md` carries the measured numbers and the shortfall.
MINIMUM_SPEEDUP: Final[float] = 1.0

GIT_SHA = "3f1c0a9e2b7d5468af0c1d3e5b7a9f2046813c5d"
SETUP_HASH = "cd" * 32

# Either runner's signature, as the timing loop needs it. Typed rather than ``object`` so that
# swapping the two entry points is a type-checked edit and not a runtime surprise.
if TYPE_CHECKING:
    Runner = Callable[..., ScenarioRun]


@pytest.fixture(scope="module")
def config(spec: CarSpec) -> KernelConfig:
    """The validated, flat configuration both runners are given."""
    return spec.kernel_config()


@pytest.fixture(scope="module")
def suite(config: KernelConfig) -> dict[str, Scenario]:
    """The scenario set, so both tests name a scenario that actually exists."""
    return dict(build_scenarios(config))


@dataclass(frozen=True, slots=True)
class Comparison:
    """One scenario run twice, once per path, with both wall times measured.

    The runs are produced *by* the timing loop rather than by a separate equality loop, so the
    comparison is not two extra 7 s runs and the bytes under test are the bytes that were timed.
    """

    plan: Scenario
    reference: ScenarioRun
    headless: ScenarioRun
    reference_s: float
    headless_s: float

    @property
    def speedup(self) -> float:
        """Fastest reference run over fastest headless run."""
        return self.reference_s / self.headless_s


def _fastest(run: Runner, plan: Scenario, config: KernelConfig) -> tuple[ScenarioRun, float]:
    """Run ``run`` ``TIMED_REPEATS`` times and keep the fastest run and its wall time.

    The minimum, not the median: the runs share a machine with the rest of the suite and with
    whatever else is on it, and a slow sample is noise while a fast one is close to the true cost.
    """
    fastest_run: ScenarioRun | None = None
    fastest_s = float("inf")
    for _ in range(TIMED_REPEATS):
        start = time.perf_counter()
        result = run(config, plan)
        elapsed = time.perf_counter() - start
        if elapsed < fastest_s:
            fastest_s = elapsed
            fastest_run = result
    if fastest_run is None:  # pragma: no cover -- TIMED_REPEATS is a module constant above
        msg = "the timing loop produced no run"
        raise AssertionError(msg)
    return fastest_run, fastest_s


def _compare(config: KernelConfig, plan: Scenario) -> Comparison:
    """Both runners on one plan, each warmed past the JIT and then timed.

    The warm-up is what makes this a comparison of the two schedules rather than of the first call
    to a Numba function: without it the first path measured pays for compiling every kernel the
    other one then reuses.
    """
    run_scenario(config, plan)
    run_headless(config, plan)
    reference, reference_s = _fastest(run_scenario, plan, config)
    headless, headless_s = _fastest(run_headless, plan, config)
    return Comparison(
        plan=plan,
        reference=reference,
        headless=headless,
        reference_s=reference_s,
        headless_s=headless_s,
    )


@pytest.fixture(scope="module")
def straight(config: KernelConfig, suite: dict[str, Scenario]) -> Comparison:
    """``accelerate_to_speed`` through both paths: launch, gear ladder, full-throttle tail."""
    return _compare(config, suite[STRAIGHT_SCENARIO])


@pytest.fixture(scope="module")
def steering(config: KernelConfig, suite: dict[str, Scenario]) -> Comparison:
    """``steady_state_circle`` through both paths: neutral, steering, lateral load transfer."""
    return _compare(config, suite[STEERING_SCENARIO])


def _drivetrain_fields() -> tuple[str, ...]:
    """``DrivetrainTrace``'s own field names, so a column added there is compared here too.

    Read off the dataclass rather than listed, because a hand-written list is a list that goes
    stale: a new published column would then be silently unchecked, which is the failure this
    helper exists to prevent.
    """
    return tuple(DrivetrainTrace.__dataclass_fields__)


def _step_output_fields(headless: ScenarioRun) -> tuple[str, ...]:
    """The run's own ``StepOutputs`` field names - it is a ``NamedTuple``, so ``_fields`` is it.

    Taken from the run rather than imported from the kernel module so that this file does not have
    to import a Numba kernel to learn a buffer layout; the layout is whatever the run carries.
    """
    return tuple(headless.step_outputs._fields)


def _assert_run_matches(headless: ScenarioRun, reference: ScenarioRun, label: str) -> None:
    """Every array and the whole record, compared as bytes rather than as numbers.

    Byte comparison rather than ``allclose`` on purpose. A fast path earns the right to be faster
    by doing the same arithmetic in a different order of *scheduling*; if a single rounding
    difference crept in, a tolerance-based assertion would hide it and the run would no longer be
    the run it claims to reproduce.
    """
    assert headless.trace.shape == reference.trace.shape, label
    assert headless.trace.tobytes() == reference.trace.tobytes(), f"{label}: trace bytes differ"
    for name in ("drive_torque_nm", "brake_torque_nm", "steer_wheel_deg"):
        headless_array = np.ascontiguousarray(getattr(headless, name))
        reference_array = np.ascontiguousarray(getattr(reference, name))
        assert headless_array.tobytes() == reference_array.tobytes(), f"{label}: {name} differs"
    for field in _drivetrain_fields():
        headless_column = np.ascontiguousarray(getattr(headless.drivetrain, field))
        reference_column = np.ascontiguousarray(getattr(reference.drivetrain, field))
        assert headless_column.tobytes() == reference_column.tobytes(), (
            f"{label}: drivetrain.{field} differs"
        )
    for field in _step_output_fields(headless):
        headless_column = np.ascontiguousarray(getattr(headless.step_outputs, field))
        reference_column = np.ascontiguousarray(getattr(reference.step_outputs, field))
        assert headless_column.tobytes() == reference_column.tobytes(), (
            f"{label}: step_outputs.{field} differs"
        )
    assert headless.steps == reference.steps, label
    assert headless.control_steps == reference.control_steps, label
    assert headless.dt_s == reference.dt_s, label
    assert headless.record_dt_s == reference.record_dt_s, label
    assert headless.gears == reference.gears, label
    assert headless.record == reference.record, f"{label}: the assembled record differs"


def test_straight_line_trace_is_byte_identical(straight: Comparison) -> None:
    """``accelerate_to_speed``: the profiled scenario reproduces byte for byte."""
    _assert_run_matches(straight.headless, straight.reference, STRAIGHT_SCENARIO)


def test_steering_trace_is_byte_identical(steering: Comparison) -> None:
    """``steady_state_circle``: a steering run reproduces byte for byte.

    The case worth having separately. A straight-line run never reaches the lateral load transfer,
    the Ackermann split, bump steer or the per-corner slip work the energy residual is built from,
    so an equality claim that only covered a straight line would leave most of the kernel's own
    output unverified on the path that now produces it.
    """
    _assert_run_matches(steering.headless, steering.reference, STEERING_SCENARIO)


def test_manifest_attaches_and_changes_nothing(
    config: KernelConfig, suite: dict[str, Scenario]
) -> None:
    """A manifest rides on a headless run and moves no byte of it.

    Same contract as the reference runner: the five cited facts are the caller's to supply, the
    runner infers nothing, and attaching one needs no seed source, no car-spec or scenario
    resolution, no setup hashing, no git call and no clock.
    """
    plan = suite[STEERING_SCENARIO]
    manifest = RunManifest(
        seed=7,
        car_spec_version="fia-2026-c-issue-20",
        scenario_version="headless-2026-10-09",
        setup_hash=SETUP_HASH,
        git_sha=GIT_SHA,
    )
    bare = run_headless(config, plan)
    cited = run_headless(config, plan, manifest=manifest)
    assert cited.manifest == manifest
    assert bare.manifest is None
    assert cited.trace.tobytes() == bare.trace.tobytes()
    assert cited.record == bare.record


def test_headless_is_faster(straight: Comparison) -> None:
    """The speedup is measured here and asserted only to be a speedup.

    The bound is 1.0 rather than the P6 gate's 50x. The measured factor is a few times over on
    this machine, and the remaining cost is the record post-pass and the kernel rather than
    avoidable Python, so a larger bound would be a test asserting a number this implementation does
    not reach. ``docs/latency.md`` reports what was measured, including the shortfall.
    """
    speedup = straight.speedup
    assert speedup > MINIMUM_SPEEDUP, (
        f"{STRAIGHT_SCENARIO}: run_headless was not faster than run_scenario - "
        f"reference {straight.reference_s:.3f} s, headless {straight.headless_s:.3f} s, "
        f"factor {speedup:.2f}x"
    )


def test_refusals_match_the_reference_runner(
    config: KernelConfig, suite: dict[str, Scenario]
) -> None:
    """A plan the reference runner refuses, the headless runner refuses before stepping.

    One case rather than a table, because the point is structural and not per-input: the headless
    path reaches its validation through the reference runner's own helpers, so it inherits the whole
    refusal set rather than a copy of it. A segment that is not a whole number of control
    intervals is the cheapest one that fails before any stepping, which is exactly what has to keep
    holding when the per-step checks move to per-segment.
    """
    plan = suite[STEERING_SCENARIO]
    segment = plan.segments[0]
    # One kernel step longer than the declared segment: still a whole number of kernel steps, so the
    # fixed-step rule is satisfied, but no longer a whole number of 100-step control intervals.
    unusable = Scenario(
        name=plan.name,
        initial_speed_m_s=plan.initial_speed_m_s,
        description=plan.description,
        initial_gear=plan.initial_gear,
        segments=(
            ScenarioSegment(
                segment.duration_s + config.dt_s,
                steer_wheel_deg=segment.steer_wheel_deg,
            ),
            *plan.segments[1:],
        ),
    )
    with pytest.raises(ValueError, match="whole number"):
        run_scenario(config, unusable)
    with pytest.raises(ValueError, match="whole number"):
        run_headless(config, unusable)


def test_repeat_headless_runs_are_byte_identical(
    config: KernelConfig, suite: dict[str, Scenario]
) -> None:
    """Two headless runs of one plan give the same bytes.

    Reproducibility is a property of the *schedule* as much as of the kernel: a path that read a
    clock, reused a buffer it should have zeroed or carried state across runs could be faster and
    still wrong. No clock and no random source are read here, and this is the check that says so.
    """
    plan = suite[STEERING_SCENARIO]
    first = run_headless(config, plan)
    second = run_headless(config, plan)
    assert first.trace.tobytes() == second.trace.tobytes()
    assert first.record == second.record
