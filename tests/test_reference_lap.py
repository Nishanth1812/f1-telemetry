"""Scenario-to-telemetry publication (P4/P5 integration): lap/sector session channels.

Pins :func:`f1telemetry.testing.reference_lap.publish_session_channels` - the
bounded handoff the demo checkpoint names as missing ("Scenario records do
not yet publish P4 lap/sector event channels into that path"). A published
record carries the five contract session channels on every frame, so the
existing sensor/Parquet/replay path and the dashboard's channel-change event
log carry lap timing with no special casing.

Owns: this new file only. Uses invented track/run geometry on the same
synthetic square as the lap-timing tests; no car coefficients, no wall clock.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from f1telemetry.laps import TrackSample, reference_lap_from_samples
from f1telemetry.telemetry.pipeline import run_bridge
from f1telemetry.testing.parquet_io import read_frames, serialise_frames
from f1telemetry.testing.records import SampleRecord, SensorFrame
from f1telemetry.testing.reference_lap import SESSION_CHANNELS, publish_session_channels
from f1telemetry.testing.replay import ReplaySource
from f1telemetry.testing.scenarios import (
    TRACE_PSI_INDEX,
    TRACE_STATE_SIZE,
    TRACE_X_INDEX,
    TRACE_Y_INDEX,
    DrivetrainTrace,
    ScenarioRun,
    allocate_step_outputs,
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
_STEP_M = 8.0
_DT_S = _STEP_M / _SPEED_M_S
_WHEELBASE_M = 3.0
_AXLE_TRACK_M = 2.0


def _load_square(tmp_path: Path) -> Track:
    track_path = tmp_path / "square.yaml"
    track_path.write_text(_SQUARE, encoding="utf-8")
    return load_track(track_path)


def _centreline_walk(track: Track, count: int) -> tuple[tuple[float, float, float], ...]:
    """``count`` centreline poses ``(x_m, y_m, heading_rad)`` at fixed spacing."""
    poses: list[tuple[float, float, float]] = []
    for index in range(count):
        s_m = index * _STEP_M
        x_m, y_m = track.centerline_at(s_m)
        tangent_x, tangent_y = track.tangent_at(s_m)
        poses.append((x_m, y_m, math.atan2(tangent_y, tangent_x)))
    return tuple(poses)


def _crossing_steps(track: Track) -> int:
    """``_STEP_M`` steps to reach the lap line: ceil, never int.

    The centreline spline is longer than the waypoint polygon, so a lap is
    1752.2299 m while 219 whole steps cover only 1752 m. Truncating would
    leave the final sample short of the line and no start/finish crossing
    would ever be emitted; rounding up gives one sample past it, which is
    what makes the published ``lap_index``/``lap_time`` observable.
    """
    return math.ceil(track.length_m / _STEP_M)


def _square_lap_run(track: Track, count: int) -> ScenarioRun:
    """A fabricated constant-speed run: ``count`` record rows from the line."""
    poses = _centreline_walk(track, count)
    trace = np.zeros((count, TRACE_STATE_SIZE), dtype=np.float64)
    for index, (x_m, y_m, heading_rad) in enumerate(poses):
        trace[index, TRACE_X_INDEX] = x_m
        trace[index, TRACE_Y_INDEX] = y_m
        trace[index, TRACE_PSI_INDEX] = heading_rad
    ones = np.ones(count, dtype=np.float64)
    zeros = np.zeros(count, dtype=np.float64)
    drivetrain = DrivetrainTrace(
        gear=4 * ones,
        clutch=ones,
        throttle=ones,
        ice_rpm=8300.0 * ones,
        ice_torque_nm=300.0 * ones,
        ice_power_w=zeros,
        mgu_k_power_w=zeros,
        drive_torque_nm=zeros,
        soc_mj=zeros,
        lap_recharge_mj=zeros,
        accel_m_s2=zeros,
    )
    record = SampleRecord(
        name="square_lap",
        dt_s=_DT_S,
        description="fabricated constant-speed square lap",
        ground_truth=(),
        frames=tuple(
            SensorFrame(t_s=index * _DT_S, values={"vx": _SPEED_M_S}) for index in range(count)
        ),
    )
    return ScenarioRun(
        name="square_lap",
        description="fabricated constant-speed square lap",
        dt_s=_DT_S,
        control_steps=1,
        steps=count - 1,
        trace=trace,
        drive_torque_nm=np.zeros(count - 1, dtype=np.float64),
        brake_torque_nm=np.zeros((count - 1, 4), dtype=np.float64),
        drivetrain=drivetrain,
        record=record,
        step_outputs=allocate_step_outputs(count - 1),
        steer_wheel_deg=np.zeros(count - 1, dtype=np.float64),
    )


def _channel_values(record: SampleRecord, channel: str) -> tuple[float, ...]:
    return tuple(frame.values[channel] for frame in record.frames)


class TestSessionChannelPublication:
    def test_short_stream_publishes_uniform_zeroed_channels(self, tmp_path: Path) -> None:
        track = _load_square(tmp_path)
        run = _square_lap_run(track, 5)
        published = publish_session_channels(
            run, track, wheelbase_m=_WHEELBASE_M, axle_track_m=_AXLE_TRACK_M
        )
        assert published.name == run.record.name
        assert published.dt_s == run.record.dt_s
        assert published.ground_truth == run.record.ground_truth
        assert len(published.frames) == len(run.record.frames)
        for original, frame in zip(run.record.frames, published.frames, strict=True):
            assert frame.t_s == original.t_s
            assert frame.values["vx"] == _SPEED_M_S
            for channel in SESSION_CHANNELS:
                assert channel in frame.values
            assert frame.values["lap_index"] == 0.0
            assert frame.values["sector_index"] == 0.0
            assert frame.values["lap_time"] == 0.0
            assert frame.values["sector_time"] == 0.0
            assert frame.values["delta"] == 0.0

    def test_full_lap_publishes_crossings_and_valid_times(self, tmp_path: Path) -> None:
        track = _load_square(tmp_path)
        steps = _crossing_steps(track)
        run = _square_lap_run(track, steps + 1)
        published = publish_session_channels(
            run, track, wheelbase_m=_WHEELBASE_M, axle_track_m=_AXLE_TRACK_M
        )
        lap_index = _channel_values(published, "lap_index")
        sector_index = _channel_values(published, "sector_index")
        lap_time = _channel_values(published, "lap_time")
        sector_time = _channel_values(published, "sector_time")
        # Mid-lap: nothing completed yet.
        assert lap_index[25] == 0.0
        assert sector_index[25] == 0.0
        assert lap_time[25] == 0.0
        assert sector_time[25] == 0.0
        # First sector boundary at 400 m / 40 m/s.
        assert sector_index[51] == 1.0
        assert sector_time[51] == pytest.approx(10.0, abs=1.0)
        # Second boundary at 800 m: sectors are 10 s each so far.
        assert sector_index[101] == 2.0
        assert sector_time[101] == pytest.approx(10.0, abs=1.0)
        # The lap line: indices advance, the valid 40 s lap publishes.
        assert lap_index[-1] == 1.0
        assert sector_index[-1] == 0.0
        assert lap_time[-1] == pytest.approx(track.length_m / _SPEED_M_S, abs=1.0)
        assert sector_time[-1] == pytest.approx(10.0, abs=1.0)

    def test_off_track_lap_advances_indices_without_publishing_lap_time(
        self, tmp_path: Path
    ) -> None:
        track = _load_square(tmp_path)
        steps = _crossing_steps(track)
        run = _square_lap_run(track, steps + 1)
        excursion = run.trace.copy()
        x_m, y_m = track.point_at(track.length_m / 2.0, lateral_m=20.0)
        excursion[steps // 2, TRACE_X_INDEX] = x_m
        excursion[steps // 2, TRACE_Y_INDEX] = y_m
        off_track = dataclasses.replace(run, trace=excursion)
        published = publish_session_channels(
            off_track, track, wheelbase_m=_WHEELBASE_M, axle_track_m=_AXLE_TRACK_M
        )
        # The crossings still happened, but the lap is not a measurement.
        assert published.frames[-1].values["lap_index"] == 1.0
        assert published.frames[-1].values["lap_time"] == 0.0
        # The opening clean sector still publishes its crossing time.
        assert published.frames[51].values["sector_time"] == pytest.approx(10.0, abs=1.0)

    def test_dnf_withholds_times_but_keeps_crossings(self, tmp_path: Path) -> None:
        track = _load_square(tmp_path)
        steps = _crossing_steps(track)
        run = _square_lap_run(track, steps + 1)
        published = publish_session_channels(
            run, track, wheelbase_m=_WHEELBASE_M, axle_track_m=_AXLE_TRACK_M, dnf=True
        )
        assert published.frames[-1].values["lap_index"] == 1.0
        assert published.frames[-1].values["lap_time"] == 0.0
        assert published.frames[-1].values["sector_time"] == 0.0

    def test_delta_tracks_a_matching_reference(self, tmp_path: Path) -> None:
        track = _load_square(tmp_path)
        steps = _crossing_steps(track)
        samples = tuple(
            TrackSample(time_s=index * _DT_S, x_m=x_m, y_m=y_m, heading_rad=heading)
            for index, (x_m, y_m, heading) in enumerate(_centreline_walk(track, steps + 1))
        )
        reference = reference_lap_from_samples(track, samples)
        run = _square_lap_run(track, steps + 1)
        published = publish_session_channels(
            run,
            track,
            wheelbase_m=_WHEELBASE_M,
            axle_track_m=_AXLE_TRACK_M,
            reference=reference,
        )
        for frame in published.frames:
            assert frame.values["delta"] == pytest.approx(0.0, abs=0.5)


class TestPublishedRunsFlowThroughTelemetry:
    def test_bridge_and_parquet_preserve_session_channels(self, tmp_path: Path) -> None:
        track = _load_square(tmp_path)
        steps = _crossing_steps(track)
        published = publish_session_channels(
            _square_lap_run(track, steps + 1),
            track,
            wheelbase_m=_WHEELBASE_M,
            axle_track_m=_AXLE_TRACK_M,
        )
        result = run_bridge(published, seed=7)
        assert _channel_values(published, "lap_index") == pytest.approx(
            [frame.values["lap_index"] for frame in result.frames]
        )
        expected_lap_time = track.length_m / _SPEED_M_S
        assert result.frames[-1].values["lap_time"] == pytest.approx(expected_lap_time, abs=1.0)
        reread = read_frames(result.parquet)
        assert reread[-1].values["lap_index"] == 1.0
        assert reread[-1].values["sector_index"] == 0.0
        assert reread[-1].values["lap_time"] == pytest.approx(expected_lap_time, abs=1.0)

    def test_replay_carries_session_channels_as_ordinary_channels(self, tmp_path: Path) -> None:
        track = _load_square(tmp_path)
        steps = _crossing_steps(track)
        published = publish_session_channels(
            _square_lap_run(track, steps + 1),
            track,
            wheelbase_m=_WHEELBASE_M,
            axle_track_m=_AXLE_TRACK_M,
        )
        path = tmp_path / "lap_run.parquet"
        path.write_bytes(serialise_frames(published))
        source = ReplaySource(path)
        frames = list(source.iter_frames(duration_us=60_000_000, max_hz=10_000.0))
        assert len(frames) == len(published.frames)
        assert frames[51].channels["sector_index"] == 1.0
        assert frames[-1].channels["lap_index"] == 1.0
        assert frames[-1].channels["lap_time"] == pytest.approx(
            track.length_m / _SPEED_M_S, abs=1.0
        )
        for frame in frames:
            assert frame.events == ()

    def test_publication_is_deterministic_and_idempotent(self, tmp_path: Path) -> None:
        track = _load_square(tmp_path)
        steps = _crossing_steps(track)
        run = _square_lap_run(track, steps + 1)
        first = publish_session_channels(
            run, track, wheelbase_m=_WHEELBASE_M, axle_track_m=_AXLE_TRACK_M
        )
        second = publish_session_channels(
            run, track, wheelbase_m=_WHEELBASE_M, axle_track_m=_AXLE_TRACK_M
        )
        assert [frame.values for frame in first.frames] == [frame.values for frame in second.frames]
        republished = publish_session_channels(
            dataclasses.replace(run, record=first),
            track,
            wheelbase_m=_WHEELBASE_M,
            axle_track_m=_AXLE_TRACK_M,
        )
        assert [frame.values for frame in republished.frames] == [
            frame.values for frame in first.frames
        ]
