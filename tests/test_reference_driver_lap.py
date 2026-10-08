"""P4 closed-loop integration: the reference driver laps both fixtures.

The reference driver is the missing end-to-end evidence the demo record
asks for: both fictional fixtures driven, end to end, by the pure-pursuit
law over the solved minimum-curvature line and a curvature-bound speed
profile whose lateral-acceleration target is the *measured* P2 capability
(~7.5 m/s^2 from the steady-state circle), not the 25-30 m/s^2 the
unit tests use on an untuned model. Every run starts at the start/finish
pose on the solved line, so the first crossing stream is a true lap.

Runs twice per track and pins byte-identical traces and identical
lap/sector times: the runner is deterministic, and so is this closed
loop.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from f1telemetry.contracts.car_spec import CarSpec, KernelConfig, load_car_spec
from f1telemetry.laps import TrackEventKind
from f1telemetry.racing_line import (
    LateralOffsetSolution,
    SpeedProfile,
    minimum_curvature_offsets,
    speed_profile,
)
from f1telemetry.testing.reference_lap import assess_scenario_lap, make_reference_driver
from f1telemetry.testing.scenarios import (
    TRACE_X_INDEX,
    TRACE_Y_INDEX,
    Scenario,
    ScenarioRun,
    ScenarioSegment,
    run_scenario,
)
from f1telemetry.tracks import Track, load_track

pytestmark = pytest.mark.kernel

# Measured P2 steady-state-circle lateral capability, ~7.5 m/s^2. The
# untuned fixture model cannot hold the 25-30 m/s^2 the unit tests
# assume, so the profile is matched to what the car actually reaches.
_LATERAL_ACCEL_M_S2 = 7.5
_MAX_SPEED_M_S = 60.0
_MAX_ACCEL_M_S2 = 8.0
_MAX_BRAKE_M_S2 = 25.0
# The solved line is clipped half a car inside the painted edge only
# with the default margin, which a following CG overshoots into DNF
# territory; the wider margin keeps the CG's every-wheel envelope
# inside the local half-width with no change to the physics.
_LINE_MARGIN_M = 2.0
_GRIP_MARGIN = 0.9
# Technical ring's chicane halves its lap pace (~108 s per lap), so
# the window closes both fixtures' first lap with room for one more
# sector crossing to confirm the sector stream.
_DURATION_S = 115.0

_AXLE_TRACK_M = 2.0

_TRACKS = ("coastal_loop", "technical_ring")


@pytest.fixture(scope="module")
def spec() -> CarSpec:
    return load_car_spec()


@pytest.fixture(scope="module")
def config(spec: CarSpec) -> KernelConfig:
    return spec.kernel_config()


def _line_and_profile(track: Track) -> tuple[LateralOffsetSolution, SpeedProfile]:
    line = minimum_curvature_offsets(track, margin_m=_LINE_MARGIN_M)
    profile = speed_profile(
        track,
        line,
        max_speed_m_s=_MAX_SPEED_M_S,
        max_accel_m_s2=_MAX_ACCEL_M_S2,
        max_brake_m_s2=_MAX_BRAKE_M_S2,
        lateral_accel_m_s2=_LATERAL_ACCEL_M_S2,
    )
    return line, profile


def _closed_loop_run(track: Track, config: KernelConfig) -> ScenarioRun:
    line, profile = _line_and_profile(track)
    assert line.converged and profile.converged
    v0 = float(np.interp(0.0, profile.s_m, profile.speed_m_s, period=track.length_m))
    x0, y0 = track.point_at(0.0, float(line.lateral_m[0]))
    tangent_x, tangent_y = track.tangent_at(0.0)
    law = make_reference_driver(
        track,
        line,
        profile,
        wheelbase_m=config.wheelbase_m,
        steering_ratio=config.steering_ratio,
        grip_margin=_GRIP_MARGIN,
    )
    plan = Scenario(
        name=f"reference_driver_{track.name}",
        initial_speed_m_s=v0,
        initial_gear=4,
        upshift_at_shift_point=True,
        initial_x_m=x0,
        initial_y_m=y0,
        initial_heading_rad=math.atan2(tangent_y, tangent_x),
        description="P4 reference driver lapping the fixture",
        segments=(ScenarioSegment(_DURATION_S),),
    )
    return run_scenario(config, plan, control_law=law, max_brake_torque_nm=9000.0)


def _lap_and_sector_times(
    run: ScenarioRun, track: Track, config: KernelConfig
) -> tuple[float, ...]:
    assessment = assess_scenario_lap(
        track,
        run,
        wheelbase_m=config.wheelbase_m,
        axle_track_m=_AXLE_TRACK_M,
    )
    assert assessment.validity.valid, assessment.validity.failures

    lap_index = 0
    lap_times: list[float] = []
    sector_times: list[float] = []
    boundary_t = 0.0
    for event in assessment.events:
        if event.kind is TrackEventKind.LAP:
            lap_index += 1
            lap_times.append(event.time_s - boundary_t)
            boundary_t = event.time_s
        else:
            sector_times.append(event.time_s - boundary_t)
            boundary_t = event.time_s
    assert lap_index >= 1
    assert lap_times and sector_times
    for value in (*lap_times, *sector_times):
        assert math.isfinite(value) and value > 0.0
    return tuple(lap_times + sector_times)


@pytest.fixture(scope="module")
def reference_runs(config: KernelConfig) -> dict[str, tuple[Track, ScenarioRun, ScenarioRun]]:
    """Both fixtures' closed-loop run, twice each, for reuse across tests."""
    runs: dict[str, tuple[Track, ScenarioRun, ScenarioRun]] = {}
    for fixture_name in _TRACKS:
        track = load_track(Path(__file__).parents[1] / "tracks" / f"{fixture_name}.yaml")
        runs[fixture_name] = (
            track,
            _closed_loop_run(track, config),
            _closed_loop_run(track, config),
        )
    return runs


@pytest.mark.parametrize("fixture_name", _TRACKS)
def test_reference_driver_laps_the_fixture_valid_and_repeatable(
    fixture_name: str,
    config: KernelConfig,
    reference_runs: dict[str, tuple[Track, ScenarioRun, ScenarioRun]],
) -> None:
    track, first, second = reference_runs[fixture_name]
    assessment = assess_scenario_lap(
        track,
        first,
        wheelbase_m=config.wheelbase_m,
        axle_track_m=_AXLE_TRACK_M,
    )
    assert assessment.validity.valid
    assert any(event.kind is TrackEventKind.LAP for event in assessment.events)
    times_first = _lap_and_sector_times(first, track, config)

    assert second.trace.tobytes() == first.trace.tobytes()
    times_second = _lap_and_sector_times(second, track, config)
    assert times_second == times_first


@pytest.mark.parametrize("fixture_name", _TRACKS)
def test_invalid_companion_run_is_rejected(
    fixture_name: str,
    config: KernelConfig,
    reference_runs: dict[str, tuple[Track, ScenarioRun, ScenarioRun]],
) -> None:
    track, run, _ = reference_runs[fixture_name]

    excursion = run.trace.copy()
    mid = run.steps // 2
    x_m, y_m = track.point_at(track.length_m / 2.0, lateral_m=20.0)
    excursion[mid, TRACE_X_INDEX] = x_m
    excursion[mid, TRACE_Y_INDEX] = y_m
    off_track_run = dataclasses.replace(run, trace=excursion)
    excursion_assessment = assess_scenario_lap(
        track,
        off_track_run,
        wheelbase_m=config.wheelbase_m,
        axle_track_m=_AXLE_TRACK_M,
    )
    assert not excursion_assessment.validity.valid
    assert "off_track" in excursion_assessment.validity.failures

    dnf_assessment = assess_scenario_lap(
        track,
        run,
        wheelbase_m=config.wheelbase_m,
        axle_track_m=_AXLE_TRACK_M,
        dnf=True,
    )
    assert not dnf_assessment.validity.valid
    assert "dnf" in dnf_assessment.validity.failures
