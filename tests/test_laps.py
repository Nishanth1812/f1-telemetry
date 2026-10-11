"""P4-T6: lap timing plus the delta against a reference lap.

A lap and its sectors are timed from the forward start/finish and
sector crossings; the delta answers "how far ahead of or behind the
reference lap is the car here", i.e. the signed time gap at the same
arc length on the lap.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from f1telemetry.laps import (
    ReferenceLap,
    TrackSample,
    assess_lap,
    crossing_events,
    evaluate_lap_validity,
    reference_lap_delta,
    reference_lap_from_samples,
)
from f1telemetry.tracks import Track, load_track

_SQUARE = """\
version: 1
name: square
description: synthetic
centerline:
  - {x_m: 0.0, y_m: 0.0}
  - {x_m: 400.0, y_m: 0.0}
  - {x_m: 400.0, y_m: 400.0}
  - {x_m: 0.0, y_m: 400.0}
widths_m: [12.0, 12.0, 12.0, 12.0]
sectors:
  - {s_m: 400.0}
  - {s_m: 800.0}
"""

_SPEED_M_S = 40.0


def _constant_speed_lap(
    tmp_path: Path, *, speed_m_s: float = _SPEED_M_S
) -> tuple[Track, list[TrackSample]]:
    track_path = tmp_path / "square.yaml"
    track_path.write_text(_SQUARE, encoding="utf-8")
    track = load_track(track_path)
    # Walk the centreline at fixed steps, and interpolating heading
    # through the corners is unnecessary: pure pursuit is out of scope
    # here, a position stream is enough.
    samples = []
    step_m = 8.0
    steps = int(track.length_m / step_m)
    for index in range(steps + 1):
        s = index * step_m
        x, y = track.centerline_at(s)
        tangent = track.tangent_at(s)
        heading = math.atan2(tangent[1], tangent[0])
        samples.append(TrackSample(s / speed_m_s, x, y, heading))
    finish_x, finish_y = track.centerline_at(track.length_m)
    finish_tangent = track.tangent_at(track.length_m)
    samples.append(
        TrackSample(
            track.length_m / speed_m_s,
            finish_x,
            finish_y,
            math.atan2(finish_tangent[1], finish_tangent[0]),
        )
    )
    return track, samples


def _sample_at(track: Track, s_m: float, time_s: float) -> TrackSample:
    x_m, y_m = track.centerline_at(s_m)
    tangent_x, tangent_y = track.tangent_at(s_m)
    return TrackSample(time_s, x_m, y_m, math.atan2(tangent_y, tangent_x))


def test_crossing_events_time_the_lap_and_sectors(tmp_path: Path) -> None:
    track, samples = _constant_speed_lap(tmp_path)
    events = crossing_events(track, samples)
    laps = [event for event in events if event.kind.value == "lap"]
    sectors = [event for event in events if event.kind.value == "sector"]
    assert len(laps) == 1
    expected = track.length_m / _SPEED_M_S
    assert math.isclose(laps[0].time_s, expected, rel_tol=2e-2)
    assert [event.index for event in sectors] == [1, 2]
    assert math.isclose(sectors[0].time_s, 400.0 / _SPEED_M_S, rel_tol=5e-2)


def test_assess_lap_rejects_a_track_limits_excursion(tmp_path: Path) -> None:
    track, samples = _constant_speed_lap(tmp_path)
    index = len(samples) // 2
    progress = track.length_m / 2.0
    x_m, y_m = track.point_at(progress, lateral_m=20.0)
    tangent_x, tangent_y = track.tangent_at(progress)
    samples[index] = replace(
        samples[index],
        x_m=x_m,
        y_m=y_m,
        heading_rad=math.atan2(tangent_y, tangent_x),
    )

    result = assess_lap(track, samples, wheel_base_m=3.0, axle_track_m=2.0)

    assert result.validity.completed_lap
    assert not result.validity.valid
    assert not result.validity.wheels_within_limits
    assert "off_track" in result.validity.failures


def test_lap_and_sector_timing_repeat_for_identical_samples(tmp_path: Path) -> None:
    track, samples = _constant_speed_lap(tmp_path)

    first = assess_lap(track, samples, wheel_base_m=3.0, axle_track_m=2.0)
    second = assess_lap(track, samples, wheel_base_m=3.0, axle_track_m=2.0)

    assert first.events == second.events
    assert first.validity == second.validity


def test_reference_lap_replays_the_sample_times(tmp_path: Path) -> None:
    track, samples = _constant_speed_lap(tmp_path)
    reference = reference_lap_from_samples(track, samples)
    assert isinstance(reference, ReferenceLap)
    assert math.isclose(reference.lap_time_s, track.length_m / _SPEED_M_S, rel_tol=2e-2)
    assert math.isclose(reference.time_at(0.0), 0.0, abs_tol=1e-9)
    assert math.isclose(reference.time_at(400.0), 400.0 / _SPEED_M_S, rel_tol=2e-2)


def test_delta_is_zero_for_an_identical_lap(tmp_path: Path) -> None:
    track, samples = _constant_speed_lap(tmp_path)
    reference = reference_lap_from_samples(track, samples)
    for s, t in ((0.0, 0.0), (400.0, 10.0), (800.0, 20.0), (1200.0, 30.0)):
        delta = reference_lap_delta(reference, s_m=s, lap_time_s=t)
        assert math.isclose(delta, 0.0, abs_tol=0.3)


def test_delta_sign_behind_and_ahead_of_reference(tmp_path: Path) -> None:
    track, samples = _constant_speed_lap(tmp_path, speed_m_s=_SPEED_M_S)
    reference = reference_lap_from_samples(track, samples)
    # Half a second behind the reference at the same place.
    behind = reference_lap_delta(reference, s_m=400.0, lap_time_s=10.5)
    assert behind > 0.0
    assert math.isclose(behind, 0.5, abs_tol=0.3)
    # Half a second ahead.
    ahead = reference_lap_delta(reference, s_m=400.0, lap_time_s=9.5)
    assert ahead < 0.0
    assert math.isclose(ahead, -0.5, abs_tol=0.3)


def test_delta_wraps_arc_length_around_the_lap(tmp_path: Path) -> None:
    track, samples = _constant_speed_lap(tmp_path)
    reference = reference_lap_from_samples(track, samples)
    direct = reference_lap_delta(reference, s_m=400.0, lap_time_s=10.25)
    wrapped = reference_lap_delta(reference, s_m=400.0 + track.length_m, lap_time_s=10.25)
    assert math.isclose(direct, wrapped, abs_tol=1e-9)


def test_reference_lap_requires_a_completed_lap(tmp_path: Path) -> None:
    track, samples = _constant_speed_lap(tmp_path)
    quarter = samples[: max(2, len(samples) // 4)]
    with pytest.raises(ValueError):
        reference_lap_from_samples(track, quarter)


def test_reference_lap_requires_monotone_time(tmp_path: Path) -> None:
    track, samples = _constant_speed_lap(tmp_path)
    scrambled = list(samples)
    scrambled[5], scrambled[6] = scrambled[6], scrambled[5]
    with pytest.raises(ValueError):
        reference_lap_from_samples(track, scrambled)


def test_assess_lap_rejects_a_lap_flagged_invalid(tmp_path: Path) -> None:
    track, samples = _constant_speed_lap(tmp_path)

    result = assess_lap(track, samples, wheel_base_m=3.0, axle_track_m=2.0, invalid=True)

    assert result.validity.completed_lap
    assert result.validity.wheels_within_limits
    assert result.validity.invalid
    assert not result.validity.valid
    assert result.validity.failures == ("invalid",)


def test_assess_lap_reports_incomplete_lap_on_a_short_stream(tmp_path: Path) -> None:
    track, samples = _constant_speed_lap(tmp_path)
    # A quarter of the lap cannot contain a forward start/finish
    # crossing, so the stream completes no lap.
    short = samples[: max(2, len(samples) // 4)]

    result = assess_lap(track, short, wheel_base_m=3.0, axle_track_m=2.0)

    assert not result.validity.completed_lap
    assert result.validity.wheels_within_limits
    assert not result.validity.valid
    assert result.validity.failures == ("incomplete_lap",)


def test_reversing_over_a_boundary_emits_no_events(tmp_path: Path) -> None:
    track, _ = _constant_speed_lap(tmp_path)
    boundary = track.sector_boundaries_m[0]
    # The car reaches 8 m past the boundary, then reverses back
    # across it: 16 m apart, far inside the half-lap unwrap
    # contract, so the backward move is a reversal, not a wrap.
    stream = (
        _sample_at(track, boundary + 8.0, 10.0),
        _sample_at(track, boundary - 8.0, 11.0),
    )

    assert crossing_events(track, stream) == ()


def test_every_failure_reason_is_listed_at_once() -> None:
    validity = evaluate_lap_validity(
        completed_lap=False,
        wheels_within_limits=False,
        dnf=True,
        invalid=True,
    )

    assert not validity.valid
    assert validity.failures == ("dnf", "invalid", "incomplete_lap", "off_track")


def test_tolerance_m_widens_the_track_limits(tmp_path: Path) -> None:
    track, samples = _constant_speed_lap(tmp_path)
    index = len(samples) // 2
    progress = track.length_m / 2.0
    # The CG rides 5.5 m left of the centreline, so with the
    # 1.0 m half-track the left wheels sit at 6.5 m against the
    # 6.0 m half-width: 0.5 m out, and 0.5 m inside once
    # tolerance_m = 1.0 widens the limit to 7.0 m.
    x_m, y_m = track.point_at(progress, lateral_m=5.5)
    tangent_x, tangent_y = track.tangent_at(progress)
    samples[index] = replace(
        samples[index],
        x_m=x_m,
        y_m=y_m,
        heading_rad=math.atan2(tangent_y, tangent_x),
    )

    strict = assess_lap(track, samples, wheel_base_m=3.0, axle_track_m=2.0)
    widened = assess_lap(track, samples, wheel_base_m=3.0, axle_track_m=2.0, tolerance_m=1.0)

    assert strict.validity.failures == ("off_track",)
    assert not strict.validity.wheels_within_limits
    assert widened.validity.valid
    assert widened.validity.failures == ()


def test_assess_lap_rejects_a_single_wheel_excursion(tmp_path: Path) -> None:
    track, samples = _constant_speed_lap(tmp_path)
    index = len(samples) // 2
    progress = track.length_m / 2.0
    # A CG 4.5 m left with a 45-degree heading puts the
    # front-left wheel at 4.5 + 1.5*sin(pi/4) + 1.0*cos(pi/4)
    # = 6.27 m, past the 6.0 m half-width; the other three
    # wheels land at 4.85 m, 4.14 m and 2.73 m, inside it.
    x_m, y_m = track.point_at(progress, lateral_m=4.5)
    tangent_x, tangent_y = track.tangent_at(progress)
    heading = math.atan2(tangent_y, tangent_x) + math.pi / 4.0
    samples[index] = replace(samples[index], x_m=x_m, y_m=y_m, heading_rad=heading)

    result = assess_lap(track, samples, wheel_base_m=3.0, axle_track_m=2.0)

    assert result.validity.completed_lap
    assert not result.validity.wheels_within_limits
    assert not result.validity.valid
    assert result.validity.failures == ("off_track",)
