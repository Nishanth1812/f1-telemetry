"""Unified P5 bridge: record -> sensors -> CAN-FD -> Parquet -> replay."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from f1telemetry.contracts.channels import ChannelContract
from f1telemetry.telemetry.bus import (
    bus_utilisation,
    message_specs_from_contract,
    schedule,
)
from f1telemetry.telemetry.pipeline import (
    bus_plan_for_record,
    decode_canfd_values,
    read_annotations,
    replay_bytes,
    run_bridge,
)
from f1telemetry.telemetry.sensors import FaultParameters
from f1telemetry.testing.fixtures import cornering_record, straight_line_record
from f1telemetry.testing.parquet_io import read_frames


class TestCanFdRoundTrip:
    def test_wire_decodes_to_processed_values(self) -> None:
        record = straight_line_record()
        result = run_bridge(record, seed=7)
        decoded = decode_canfd_values(result.canfd, result.canfd_manifest)
        # Same number of physical frames, every value recovered.
        assert len(decoded) >= len(record.frames)
        by_time: dict[float, dict[str, float]] = {}
        for t_s, values in decoded:
            by_time.setdefault(t_s, {}).update(values)
        assert len(by_time) == len(record.frames)
        for original in result.frames:
            recovered = by_time[original.t_s]
            assert set(recovered) == set(original.values)
            for name, value in original.values.items():
                assert recovered[name] == pytest.approx(value, abs=1e-6)

    def test_wire_is_deterministic_for_same_seed(self) -> None:
        record = cornering_record()
        first = run_bridge(record, seed=3)
        second = run_bridge(record, seed=3)
        assert first.canfd == second.canfd
        assert first.parquet == second.parquet


class TestUnifiedPath:
    def test_parquet_round_trips_processed_frames(self) -> None:
        record = cornering_record()
        result = run_bridge(record, seed=11)
        replayed = read_frames(result.parquet)
        assert len(replayed) == len(result.frames)
        for original, saved in zip(result.frames, replayed, strict=True):
            assert saved.t_s == original.t_s
            assert set(saved.values) == set(original.values)
            for name, value in original.values.items():
                assert saved.values[name] == pytest.approx(value, abs=1e-6)

    def test_replay_source_matches_processed_frames(self, tmp_path: Path) -> None:
        record = straight_line_record()
        result = run_bridge(record, seed=5)
        source = result.replay_source(tmp_path / "run.parquet")
        frames = list(source.iter_frames(duration_us=10_000_000, max_hz=10_000.0))
        times = [frame.time_us for frame in frames]
        assert all(b > a for a, b in itertools.pairwise(times))
        assert len(frames) == len(result.frames)
        for processed, frame in zip(result.frames, frames, strict=True):
            assert frame.time_us == round(processed.t_s * 1_000_000)
            for name, value in processed.values.items():
                assert frame.channels[name] == pytest.approx(value, abs=1e-6)
        carried = [event for frame in frames for event in frame.events]
        assert [(event.fault_type, event.time_us) for event in carried] == [
            (annotation.fault_type, round(annotation.onset_t_s * 1_000_000))
            for annotation in result.annotations
        ]

    def test_fault_injection_is_annotated_and_persisted(self) -> None:
        record = straight_line_record()
        params = FaultParameters(severity=1.0, onset_index=1, duration_samples=2)
        result = run_bridge(record, seed=2, active_faults=[("step", params)])
        steps = [a for a in result.annotations if a.fault_type == "step"]
        assert steps, "expected at least one step annotation"
        assert all(a.seed == 2 for a in steps)
        # The corrupted stream differs from ground truth; replay keeps it.
        assert result.frames != result.ground_truth
        replayed = replay_bytes(result.parquet)
        assert [f.values for f in replayed] == [f.values for f in result.frames]

    def test_fault_annotations_survive_parquet_round_trip(self) -> None:
        record = straight_line_record()
        params = FaultParameters(severity=0.6, onset_index=2, duration_samples=3)
        result = run_bridge(record, seed=4, active_faults=[("step", params)])
        assert result.annotations, "expected step annotations"
        restored = read_annotations(result.parquet)
        assert restored == tuple(result.annotations)
        assert restored == result.saved_annotations()
        for annotation in restored:
            assert annotation.seed == 4
            assert annotation.duration_samples == 3
            assert annotation.channel_name

    def test_fault_annotations_round_trip_through_saved_file(self, tmp_path: Path) -> None:
        record = cornering_record()
        params = FaultParameters(severity=1.0, onset_index=0)
        result = run_bridge(record, seed=6, active_faults=[("freeze", params)])
        path = tmp_path / "run.parquet"
        source = result.replay_source(path)
        assert read_annotations(path) == tuple(result.annotations)
        # Replay path is unchanged: logical frames still come from the same bytes.
        frames = list(source.iter_frames(duration_us=10_000_000, max_hz=10_000.0))
        assert len(frames) == len(result.frames)
        events = [event for frame in frames for event in frame.events]
        assert [(event.fault_type, event.time_us) for event in events] == [
            (annotation.fault_type, round(annotation.onset_t_s * 1_000_000))
            for annotation in result.annotations
        ]

    def test_empty_annotations_round_trip(self) -> None:
        record = straight_line_record()
        result = run_bridge(record, seed=0)
        assert result.annotations == ()
        assert read_annotations(result.parquet) == ()
        assert result.saved_annotations() == ()

    def test_legacy_parquet_without_annotation_key_reads_empty(self) -> None:
        from f1telemetry.testing.fixtures import straight_line_record as fixture
        from f1telemetry.testing.parquet_io import serialise_frames

        record = fixture()
        legacy = serialise_frames(record)
        assert read_annotations(legacy) == ()

    def test_annotation_bytes_are_deterministic(self) -> None:
        record = cornering_record()
        params = FaultParameters(severity=0.9, onset_index=3, duration_samples=4)
        first = run_bridge(record, seed=9, active_faults=[("step", params)])
        second = run_bridge(record, seed=9, active_faults=[("step", params)])
        assert first.parquet == second.parquet
        assert read_annotations(first.parquet) == read_annotations(second.parquet)
        assert first.annotations == second.annotations

    def test_ground_truth_is_untouched(self) -> None:
        record = straight_line_record()
        before = [dict(f.values) for f in record.frames]
        result = run_bridge(
            record,
            seed=2,
            active_faults=[("step", FaultParameters(severity=1.0, onset_index=0))],
        )
        assert [dict(f.values) for f in record.frames] == before
        assert result.ground_truth == tuple(record.frames)


class TestContractBusPlan:
    def test_bus_plan_derives_from_contract_rates_and_sim_time(
        self, contract: ChannelContract
    ) -> None:
        record = straight_line_record()
        result = run_bridge(record, seed=1)
        plan = result.bus_plan
        assert plan.specs == message_specs_from_contract(contract)
        assert plan.bitrate_bps == 2_000_000
        expected_us = round(
            (record.frames[-1].t_s - record.frames[0].t_s + record.dt_s) * 1_000_000
        )
        assert plan.duration_us == expected_us
        assert plan.utilisation == bus_utilisation(plan.specs, plan.bitrate_bps)
        assert plan.utilisation < 0.70
        assert plan.schedule == schedule(plan.specs, plan.duration_us, plan.bitrate_bps)
        assert all(0 <= message.release_us < plan.duration_us for message in plan.schedule)
        assert all(message.start_us >= message.release_us for message in plan.schedule)

    def test_bus_plan_matches_direct_helper_and_is_deterministic(
        self, contract: ChannelContract
    ) -> None:
        record = cornering_record()
        first = run_bridge(record, seed=3)
        second = run_bridge(record, seed=3)
        assert first.bus_plan == second.bus_plan
        assert first.bus_plan == bus_plan_for_record(record, contract)
