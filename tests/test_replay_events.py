"""Replay-to-dashboard event path (P5-T9): saved fault annotations on the wire.

Owns: this new file only. Pins the optional ``events`` key on
:class:`f1telemetry.synthetic.Frame` - present only when a frame carries events,
so live frames stay byte-identical - and :class:`f1telemetry.testing.replay.ReplaySource`'s
emission of attached :class:`f1telemetry.telemetry.sensors.FaultAnnotation` records:
exactly once, in onset order, never rate-limited away, re-armed by ``reset``.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from f1telemetry.contracts.channels import Channel, ChannelContract
from f1telemetry.server import replay_source
from f1telemetry.synthetic import Frame, FrameEvent, SampleFn, SyntheticSource
from f1telemetry.telemetry.pipeline import read_annotations, run_bridge
from f1telemetry.telemetry.sensors import FaultAnnotation, FaultParameters
from f1telemetry.testing.fixtures import straight_line_record
from f1telemetry.testing.parquet_io import serialise_frames
from f1telemetry.testing.records import SampleRecord, SensorFrame
from f1telemetry.testing.replay import ReplaySource

_ONE_SECOND_US = 1_000_000


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

    def test_trailing_annotations_ride_the_grid_past_the_stored_frames(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "run.parquet"
        path.write_bytes(serialise_frames(straight_line_record()))
        # Onset past the last stored frame (t_s = 0.03): no stored frame can carry it.
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
        # It rides the first grid tick at or after its onset, carrying the newest
        # stored values - held, exactly as the live feed holds them between samples.
        assert last.time_us == 55_000
        assert last.time_us % 100 == 0
        assert last.samples == ()
        assert last.channels == dict(straight_line_record().frames[-1].values)
        assert len(last.events) == 1
        event = last.events[0]
        assert event.kind == "fault"
        assert event.fault_type == "saturate"
        assert event.channel == "vx"
        assert event.severity == 0.9
        assert event.duration_samples == 4
        times = [frame.time_us for frame in frames]
        assert times == sorted(times)
        # Ticks with nothing new and no event due are not delivered: replay is finite.
        assert times == [0, 10_000, 20_000, 30_000, 55_000]

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


def _live_contract_for(record: SampleRecord, contract: ChannelContract) -> ChannelContract:
    """The record's own channels, declared at the record's own cadence.

    Everything else about the real contract is kept - the names, the dtypes, the
    ranges - so the live side publishes exactly the columns the saved run carries.
    Only the rates are replaced by the record's own frame rate, because the question
    this answers is whether the *delivery* of a record looks live, not whether two
    different sampling rates would.
    """
    published = tuple(
        replace(contract.by_name(name), rate_hz=1.0 / record.dt_s)
        for name in record.frames[0].values
        if not contract.by_name(name).event
    )
    return replace(contract, channels=published)


def _replaying_samples(record: SampleRecord) -> SampleFn:
    """A :data:`SampleFn` that hands back the saved run's own values, channel by channel.

    The cursor is per channel, and the live source samples each channel on its own
    declared clock in stream order, so each channel replays its saved series exactly -
    which is what makes the two deliveries comparable frame for frame.
    """
    cursors = dict.fromkeys(record.frames[0].values, 0)

    def sample(channel: Channel, rng: np.random.Generator) -> float:
        index = cursors[channel.name]
        cursors[channel.name] = index + 1
        return float(record.frames[index].values[channel.name])

    return sample


class TestReplaySharesTheLiveDeliveryGrid:
    """Replay and live delivery are the same wire for the same record.

    One record, two sources: a live :class:`SyntheticSource` declaring that record's
    channels at the record's own rate, and a :class:`ReplaySource` reading the record
    back from Parquet. At the same ``max_hz`` the timestamps, the wire keys, the
    newest-value snapshot and the full-rate batch have to match frame for frame, so
    nothing downstream can tell which one it is watching.
    """

    def test_replay_and_live_delivery_share_the_grid_and_the_wire_keys(
        self, tmp_path: Path, contract: ChannelContract
    ) -> None:
        record = straight_line_record()
        path = tmp_path / "run.parquet"
        path.write_bytes(serialise_frames(record))
        live_contract = _live_contract_for(record, contract)
        # One window over the record's own span, so neither source is asked for
        # samples the record does not have.
        window_us = round(len(record.frames) * record.dt_s * _ONE_SECOND_US)

        # 100 Hz delivery puts every stored instant on its own tick; 30 Hz coalesces
        # three of them into one. Both are the same grid the live source would use.
        for max_hz in (100.0, 30.0):
            replay = list(ReplaySource(path).iter_frames(window_us, max_hz=max_hz))
            live = list(
                SyntheticSource(live_contract, sample_fn=_replaying_samples(record)).iter_frames(
                    window_us, max_hz=max_hz
                )
            )

            # Same grid, same keys, same newest-value snapshot, same batch:
            # byte-identical wire payloads. Replay is the finite one, so it covers
            # exactly the run's span.
            assert replay
            assert [frame.to_wire() for frame in replay] == [frame.to_wire() for frame in live]
            tick_us = math.ceil(_ONE_SECOND_US / max_hz)
            for frame in replay:
                assert frame.time_us % tick_us == 0
                assert set(frame.channels) == set(record.frames[0].values)

    def test_a_coarse_grid_coalesces_the_samples_behind_one_delivery_frame(
        self, tmp_path: Path
    ) -> None:
        record = straight_line_record()
        path = tmp_path / "run.parquet"
        path.write_bytes(serialise_frames(record))
        tick_us = math.ceil(_ONE_SECOND_US / 30.0)
        frames = list(ReplaySource(path).iter_frames(_ONE_SECOND_US, max_hz=30.0))

        # 30 Hz delivery over a 100 Hz record: the t=0 tick carries one stored frame,
        # the next tick carries the three it aggregated, and the samples keep every
        # stored instant countable even though delivery is decimated.
        assert [frame.time_us for frame in frames] == [0, tick_us]
        assert [[sample.time_us for sample in frame.samples] for frame in frames] == [
            [0],
            [10_000, 20_000, 30_000],
        ]
        assert frames[1].channels == dict(record.frames[-1].values)
        instants = [sample.time_us for frame in frames for sample in frame.samples]
        assert instants == [round(frame.t_s * _ONE_SECOND_US) for frame in record.frames]
        # The batch rides the wire, so the delivered frames are not the legacy shape.
        assert "samples" in frames[0].to_wire()

    def test_chunk_seams_neither_repeat_nor_skip_a_tick(self, tmp_path: Path) -> None:
        path = tmp_path / "run.parquet"
        path.write_bytes(serialise_frames(straight_line_record()))
        source = ReplaySource(path)
        chunks = [list(source.iter_frames(20_000, max_hz=100.0)) for _ in range(2)]
        times = [frame.time_us for chunk in chunks for frame in chunk]
        assert times == [0, 10_000, 20_000, 30_000]
        instants = [
            sample.time_us for chunk in chunks for frame in chunk for sample in frame.samples
        ]
        # Every stored instant is delivered exactly once, across the seam.
        assert instants == sorted(set(instants))
        assert len(instants) == 4


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
