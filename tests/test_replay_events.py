"""Replay-to-dashboard event path (P5-T9): saved fault annotations on the wire.

Owns: this new file only. Pins the optional ``events`` key on
:class:`f1telemetry.synthetic.Frame` - present only when a frame carries events,
so live frames stay byte-identical - and :class:`f1telemetry.testing.replay.ReplaySource`'s
emission of attached :class:`f1telemetry.telemetry.sensors.FaultAnnotation` records:
exactly once, in onset order, never rate-limited away, re-armed by ``reset``.
"""

from __future__ import annotations

import json
from pathlib import Path

from f1telemetry.server import replay_source
from f1telemetry.synthetic import Frame, FrameEvent
from f1telemetry.telemetry.pipeline import read_annotations, run_bridge
from f1telemetry.telemetry.sensors import FaultAnnotation, FaultParameters
from f1telemetry.testing.fixtures import straight_line_record
from f1telemetry.testing.parquet_io import serialise_frames
from f1telemetry.testing.records import SampleRecord, SensorFrame
from f1telemetry.testing.replay import ReplaySource


class TestFrameEventContract:
    """The frame contract gains an optional ``events`` key and nothing else."""

    def test_frame_without_events_serialises_without_events_key(self) -> None:
        frame = Frame(123_000, {"speed": 180.0})
        assert frame.to_wire() == {"time_us": 123_000, "channels": {"speed": 180.0}}

    def test_frame_event_omits_none_fields_and_keeps_false(self) -> None:
        event = FrameEvent(
            kind="fault",
            time_us=10_000,
            fault_type="step",
            channel="vx",
            severity=0.5,
            label=False,
        )
        assert event.to_wire() == {
            "kind": "fault",
            "time_us": 10_000,
            "fault_type": "step",
            "channel": "vx",
            "severity": 0.5,
            "label": False,
        }

    def test_frame_with_events_round_trips_through_json(self) -> None:
        frame = Frame(
            20_000,
            {"vx": 50.0},
            (
                FrameEvent(
                    kind="fault",
                    time_us=10_000,
                    fault_type="freeze",
                    channel="gear",
                    severity=1.0,
                    duration_samples=3,
                    label=True,
                ),
            ),
        )
        # The server's allow_nan=False guard applies to event payloads too.
        decoded = json.loads(json.dumps(frame.to_wire(), separators=(",", ":"), allow_nan=False))
        assert decoded == {
            "time_us": 20_000,
            "channels": {"vx": 50.0},
            "events": [
                {
                    "kind": "fault",
                    "time_us": 10_000,
                    "fault_type": "freeze",
                    "channel": "gear",
                    "severity": 1.0,
                    "duration_samples": 3,
                    "label": True,
                }
            ],
        }


class TestReplayAnnotationEmission:
    """Saved fault annotations surface as frame events, exactly once, in order."""

    def test_replay_without_annotations_keeps_legacy_wire_shape(self, tmp_path: Path) -> None:
        path = tmp_path / "run.parquet"
        path.write_bytes(serialise_frames(straight_line_record()))
        source = ReplaySource(path)
        frames = list(source.iter_frames(duration_us=1_000_000, max_hz=10_000.0))
        assert len(frames) == 4
        for frame in frames:
            assert frame.events == ()
            assert "events" not in frame.to_wire()

    def test_annotations_emit_exactly_once_in_onset_order(self, tmp_path: Path) -> None:
        path = tmp_path / "run.parquet"
        path.write_bytes(serialise_frames(straight_line_record()))
        annotations = (
            # Handed in out of order; the source sorts by onset.
            FaultAnnotation(
                fault_type="spike",
                severity=0.75,
                onset_t_s=0.02,
                ground_truth_t_s=0.02,
                channel_name="vx",
                seed=7,
            ),
            FaultAnnotation(
                fault_type="step",
                severity=1.0,
                onset_t_s=0.01,
                ground_truth_t_s=0.01,
                channel_name="gear",
                duration_samples=2,
                seed=7,
            ),
        )
        source = ReplaySource(path, annotations=annotations)
        frames = list(source.iter_frames(duration_us=1_000_000, max_hz=10_000.0))
        carried = [(event.fault_type, event.time_us) for frame in frames for event in frame.events]
        assert carried == [("step", 10_000), ("spike", 20_000)]
        # Every event rides a frame at or after its onset, and the wire times
        # stay non-decreasing across the whole stream.
        for frame in frames:
            for event in frame.events:
                assert frame.time_us >= event.time_us
                assert event.kind == "fault"
        times = [frame.time_us for frame in frames]
        assert times == sorted(times)

    def test_trailing_annotations_arrive_as_event_only_frames(self, tmp_path: Path) -> None:
        path = tmp_path / "run.parquet"
        path.write_bytes(serialise_frames(straight_line_record()))
        # Onset past the last stored frame (t_s = 0.03): nothing can carry it.
        annotations = (
            FaultAnnotation(
                fault_type="saturate",
                severity=0.9,
                onset_t_s=0.055,
                ground_truth_t_s=0.055,
                channel_name="vx",
                duration_samples=4,
                seed=3,
            ),
        )
        source = ReplaySource(path, annotations=annotations)
        frames = list(source.iter_frames(duration_us=1_000_000, max_hz=10_000.0))
        assert source.exhausted
        last = frames[-1]
        assert last.time_us == 55_000
        assert last.channels == {}
        assert len(last.events) == 1
        event = last.events[0]
        assert event.kind == "fault"
        assert event.fault_type == "saturate"
        assert event.channel == "vx"
        assert event.severity == 0.9
        assert event.duration_samples == 4
        times = [frame.time_us for frame in frames]
        assert times == sorted(times)

    def test_events_survive_rate_limiting(self, tmp_path: Path) -> None:
        path = tmp_path / "run.parquet"
        path.write_bytes(serialise_frames(straight_line_record()))
        annotations = (
            FaultAnnotation(
                fault_type="step",
                severity=1.0,
                onset_t_s=0.01,
                ground_truth_t_s=0.01,
                channel_name="vx",
                seed=1,
            ),
        )
        source = ReplaySource(path, annotations=annotations)
        # 30 Hz tick-limits the 100 Hz stored frames to the first one; the
        # annotation must still arrive, as an event-only frame at its onset.
        frames = list(source.iter_frames(duration_us=1_000_000, max_hz=30.0))
        carried = [event for frame in frames for event in frame.events]
        assert len(carried) == 1
        assert carried[0].time_us == 10_000

    def test_reset_rearms_annotations(self, tmp_path: Path) -> None:
        path = tmp_path / "run.parquet"
        path.write_bytes(serialise_frames(straight_line_record()))
        annotations = (
            FaultAnnotation(
                fault_type="freeze",
                severity=0.5,
                onset_t_s=0.02,
                ground_truth_t_s=0.02,
                channel_name="gear",
                seed=11,
            ),
        )
        source = ReplaySource(path, annotations=annotations)
        first = [
            frame.to_wire() for frame in source.iter_frames(duration_us=1_000_000, max_hz=10_000.0)
        ]
        assert source.exhausted
        assert any("events" in payload for payload in first)
        source.reset()
        second = [
            frame.to_wire() for frame in source.iter_frames(duration_us=1_000_000, max_hz=10_000.0)
        ]
        assert second == first

    def test_annotations_property_exposes_attached_ground_truth(self, tmp_path: Path) -> None:
        path = tmp_path / "run.parquet"
        path.write_bytes(serialise_frames(straight_line_record()))
        source = ReplaySource(path)
        assert source.annotations == ()


class TestSessionEventChannelsPassThrough:
    """Lap/sector event channels stay ordinary channels on the wire."""

    def test_event_channels_replay_as_ordinary_channels(self, tmp_path: Path) -> None:
        record = SampleRecord(
            name="session_events",
            dt_s=0.01,
            description="frames carrying the session event channels",
            ground_truth=(),
            frames=(
                # serialise_frames derives one column schema from the first
                # frame, so lap_time rides every frame - held at 0.0 until the
                # completed lap publishes its time.
                SensorFrame(
                    t_s=0.0,
                    values={"vx": 50.0, "lap_index": 0.0, "sector_index": 0.0, "lap_time": 0.0},
                ),
                SensorFrame(
                    t_s=0.01,
                    values={"vx": 50.1, "lap_index": 0.0, "sector_index": 1.0, "lap_time": 0.0},
                ),
                SensorFrame(
                    t_s=0.02,
                    values={
                        "vx": 50.2,
                        "lap_index": 1.0,
                        "sector_index": 0.0,
                        "lap_time": 91.37,
                    },
                ),
            ),
        )
        path = tmp_path / "run.parquet"
        path.write_bytes(serialise_frames(record))
        source = ReplaySource(path)
        frames = list(source.iter_frames(duration_us=1_000_000, max_hz=10_000.0))
        assert len(frames) == 3
        assert frames[1].channels["sector_index"] == 1.0
        assert frames[2].channels["lap_index"] == 1.0
        assert frames[2].channels["lap_time"] == 91.37
        # Session channels never produce wire events: only fault annotations do.
        for frame in frames:
            assert frame.events == ()


class TestServerReplayWiring:
    """The server's replay path attaches the run's saved fault annotations."""

    def test_replay_source_carries_saved_annotations(self, tmp_path: Path) -> None:
        record = straight_line_record()
        params = FaultParameters(severity=1.0, onset_index=1, duration_samples=2)
        result = run_bridge(record, seed=2, active_faults=[("step", params)])
        assert result.annotations, "expected at least one step annotation"
        path = tmp_path / "run.parquet"
        path.write_bytes(result.parquet)
        source = replay_source(path)
        assert source.annotations == tuple(result.annotations)
        assert read_annotations(path) == source.annotations

    def test_replay_source_on_legacy_run_has_no_annotations(self, tmp_path: Path) -> None:
        path = tmp_path / "run.parquet"
        path.write_bytes(serialise_frames(straight_line_record()))
        source = replay_source(path)
        assert source.annotations == ()
