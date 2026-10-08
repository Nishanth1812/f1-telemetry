"""Unified P5 bridge: record -> sensors -> CAN-FD -> Parquet -> replay."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from f1telemetry.contracts.channels import ChannelContract
from f1telemetry.telemetry.bus import (
    bus_utilisation,
    channel_groups_from_contract,
    message_specs_from_contract,
    schedule,
)
from f1telemetry.telemetry.canfd import DLC_PAYLOAD_BYTES, decode_frame
from f1telemetry.telemetry.pipeline import (
    bus_plan_for_record,
    decode_canfd_values,
    read_annotations,
    replay_bytes,
    run_bridge,
)
from f1telemetry.telemetry.sensors import FaultParameters
from f1telemetry.testing import scenarios
from f1telemetry.testing.fixtures import cornering_record, straight_line_record
from f1telemetry.testing.parquet_io import (
    METADATA,
    metadata_with_manifest,
    read_frames,
    serialise_frames,
)
from f1telemetry.testing.records import SampleRecord, SensorFrame
from f1telemetry.testing.run_manifest import (
    MANIFEST_METADATA_KEY,
    RunManifest,
    from_metadata,
)

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import CarSpec, KernelConfig

_ONE_SECOND_US: Final = 1_000_000
# The fastest rate the contract declares, and so the shortest period any scheduled
# message has. A record sampled this fast decimates nothing, which is what "full
# sensor rate" means for the bus model.
_FASTEST_HZ: Final[float] = 200.0


def _full_rate_record(contract: ChannelContract, *, seconds: float = 1.0) -> SampleRecord:
    """One frame per fastest declared period, carrying every periodic channel.

    Supplied data, not simulation: each channel ramps across its own declared range
    at the contract's own rate. The span is ``seconds`` plus one period, so the run's
    simulated window is a whole second plus a frame and the releases inside the first
    simulated second can be counted on their own.
    """
    names = tuple(channel.name for channel in contract.channels if channel.rate_hz is not None)
    steps = round(seconds * _FASTEST_HZ) + 1
    dt_s = 1.0 / _FASTEST_HZ
    return SampleRecord(
        name="full_rate",
        dt_s=dt_s,
        description=(
            f"Every periodic contract channel sampled once per {_FASTEST_HZ:g} Hz period, so the "
            "wire is the schedule with nothing decimated ahead of it."
        ),
        ground_truth=(),
        frames=tuple(
            SensorFrame(
                t_s=index * dt_s,
                values={
                    name: contract.by_name(name).range_min
                    + (contract.by_name(name).range_max - contract.by_name(name).range_min)
                    * (index / steps)
                    for name in names
                },
            )
            for index in range(steps)
        ),
    )


class TestCanFdRoundTrip:
    def test_wire_carries_the_processed_values_it_releases(self) -> None:
        record = straight_line_record()
        result = run_bridge(record, seed=7)
        decoded = decode_canfd_values(result.canfd, result.canfd_manifest)
        by_time = {frame.t_s: frame.values for frame in result.frames}
        # Every frame the run processed reaches the wire, and every wire frame decodes
        # to the values of the frame its manifest names - never to an invented one.
        assert {t_s for t_s, _values in decoded} == set(by_time)
        carried: dict[str, list[float]] = {}
        for t_s, values in decoded:
            assert set(values) <= set(by_time[t_s])
            for name, value in values.items():
                assert value == pytest.approx(by_time[t_s][name], abs=1e-6)
                carried.setdefault(name, []).append(t_s)
        # Nothing the contract schedules is missing: every channel the record
        # published appears on the wire, at the releases its message is scheduled
        # for - a channel slower than this run simply releases once and is held by
        # the decoder in between, which is what the schedule says happens.
        assert set(carried) == set(record.frames[0].values)
        for name, times in carried.items():
            assert times == sorted(times), f"{name} went backwards on the wire"

    def test_a_message_faster_than_the_run_holds_the_newest_value(self) -> None:
        record = straight_line_record()  # 100 Hz frames
        result = run_bridge(record, seed=1)
        plan = result.bus_plan
        fastest = min(plan.specs, key=lambda spec: spec.period_us)
        frame_us = round(record.dt_s * _ONE_SECOND_US)
        assert fastest.period_us < frame_us
        # Inside the first frame interval the 200 Hz message releases twice, and both
        # releases carry the one sample that exists - held, not resampled.
        first_interval = [
            decode_frame(blob).identifier
            for blob, chunk in zip(result.canfd, result.canfd_manifest, strict=True)
            if chunk.t_s == record.frames[0].t_s
        ]
        assert first_interval.count(fastest.identifier) == frame_us // fastest.period_us

    def test_wire_is_deterministic_for_same_seed(self) -> None:
        record = cornering_record()
        first = run_bridge(record, seed=3)
        second = run_bridge(record, seed=3)
        assert first.canfd == second.canfd
        assert first.parquet == second.parquet


class TestFullRateBusLoad:
    """PLAN.md section 8.2: full-rate traffic fits the virtual bus, and it is the plan."""

    def test_wire_realizes_the_scheduled_releases_at_full_rate(
        self, contract: ChannelContract
    ) -> None:
        result = run_bridge(_full_rate_record(contract), seed=4)
        plan = result.bus_plan
        assert plan.specs == message_specs_from_contract(contract)
        assert plan.channels == channel_groups_from_contract(contract)
        messages = dict(zip((spec.identifier for spec in plan.specs), plan.channels, strict=True))
        specs = {spec.identifier: spec for spec in plan.specs}

        first_second = [
            (blob, chunk)
            for blob, chunk in zip(result.canfd, result.canfd_manifest, strict=True)
            if chunk.t_s < 1.0
        ]
        # One frame per release, and only the releases the plan schedules: at full
        # sensor rate every release lands on a frame, so this is the plan's own count
        # per simulated second, read back off the wire.
        assert len(first_second) == sum(_ONE_SECOND_US // s.period_us for s in plan.specs) == 730
        for blob, chunk in first_second:
            frame = decode_frame(blob)
            spec = specs[frame.identifier]
            # Same identifier, same DLC, same channels, same schedule: the emitted
            # traffic is the plan, not a per-frame repack of the record.
            assert DLC_PAYLOAD_BYTES[frame.dlc] == spec.payload_bytes
            assert frame.timestamp_us == round(chunk.t_s * _ONE_SECOND_US)
            assert chunk.channels == messages[frame.identifier]

    def test_full_rate_wire_load_is_under_70_percent_of_the_bitrate(
        self, contract: ChannelContract
    ) -> None:
        result = run_bridge(_full_rate_record(contract), seed=4)
        plan = result.bus_plan
        specs = {spec.identifier: spec for spec in plan.specs}
        first_second = [
            blob
            for blob, chunk in zip(result.canfd, result.canfd_manifest, strict=True)
            if chunk.t_s < 1.0
        ]
        # 730 frames in one simulated second, costed by the bus model at the DLC each
        # frame actually carries.
        assert len(first_second) == 730
        bits = sum(specs[decode_frame(blob).identifier].wire_bits for blob in first_second)
        # One simulated second of wire, as a share of the bits the bus carries in one
        # second: no simulated-seconds-per-bit factor belongs in a fraction.
        load = bits / plan.bitrate_bps
        # Measured off the emitted frames, and equal to what the plan predicted.
        assert load == pytest.approx(plan.utilisation, abs=1e-12)
        assert load < 0.70


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
        assert plan.channels == channel_groups_from_contract(contract)
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


def _probe_scenario(config: KernelConfig) -> scenarios.Scenario:
    """One whole control interval of coasting from rest: the shortest run the runner takes.

    The duration is derived from the configured step, so the runner's
    whole-control-interval rule holds whatever ``dt_s`` the car spec carries, and a
    declared start speed of zero is a starting condition rather than a performance
    figure. The test below is about determinism of the saved bytes, so nothing here
    says anything about the car.
    """
    return scenarios.Scenario(
        name="p5_manifest_byte_probe",
        initial_speed_m_s=0.0,
        description="attribution probe: coasting from rest with no pedal or gear request",
        segments=(scenarios.ScenarioSegment(scenarios.CONTROL_STEPS * config.dt_s),),
    )


def _manifest(scenario_version: str = "p5-byte-probe-2026-10-08") -> RunManifest:
    """The five facts this probe cites, all supplied here - the writer resolves none."""
    return RunManifest(
        seed=7,
        car_spec_version="fia-2026-c-issue-20",
        scenario_version=scenario_version,
        setup_hash="ab" * 32,
        git_sha="3f1c0a9e2b7d5468af0c1d3e5b7a9f2046813c5d",
    )


def _text_metadata(metadata: Mapping[bytes, bytes] | None) -> dict[str, str]:
    """Arrow's byte key-value metadata as the text it was written from."""
    return {key.decode("utf-8"): value.decode("utf-8") for key, value in (metadata or {}).items()}


class TestRunManifestByteEquality:
    """Two runs citing one manifest are one file; a different manifest is one key.

    The saved bytes are the storage gate (PLAN.md section 8.3), and a run manifest is
    a citation *about* a run rather than a value resolved from one, so it has to be
    inert: repeating a run must reproduce the file exactly, and changing only what the
    run cites must change only the citation.
    """

    def test_two_runs_citing_one_manifest_serialise_byte_identically(self, spec: CarSpec) -> None:
        config = spec.kernel_config()
        plan = _probe_scenario(config)
        manifest = _manifest()
        first = scenarios.run_scenario(config, plan, manifest=manifest)
        second = scenarios.run_scenario(config, plan, manifest=manifest)

        assert first.record == second.record
        first_bytes = serialise_frames(first.record, metadata_with_manifest(manifest))
        second_bytes = serialise_frames(second.record, metadata_with_manifest(manifest))
        assert first_bytes == second_bytes
        table = pq.read_table(pa.BufferReader(first_bytes))
        assert from_metadata(table.schema.metadata) == manifest
        assert _text_metadata(table.schema.metadata)[MANIFEST_METADATA_KEY] == manifest.canonical()
        # The citation merges over the contract keys rather than replacing them.
        written = _text_metadata(table.schema.metadata)
        assert {name: written[name] for name in METADATA} == METADATA

    def test_differing_manifests_differ_only_in_the_manifest_key(self, spec: CarSpec) -> None:
        config = spec.kernel_config()
        plan = _probe_scenario(config)
        cited = _manifest()
        # Only the scenario version moves: a different run of the same inputs.
        restated = _manifest(scenario_version="p5-byte-probe-2026-10-09")

        run = scenarios.run_scenario(config, plan, manifest=cited)
        other = scenarios.run_scenario(config, plan, manifest=restated)
        assert run.record == other.record

        cited_bytes = serialise_frames(run.record, metadata_with_manifest(cited))
        restated_bytes = serialise_frames(other.record, metadata_with_manifest(restated))
        assert cited_bytes != restated_bytes

        cited_table = pq.read_table(pa.BufferReader(cited_bytes))
        restated_table = pq.read_table(pa.BufferReader(restated_bytes))
        # Same rows, same schema, same every other key: the files differ in the citation
        # and in nothing else, which is what makes the citation trustworthy.
        assert cited_table.equals(restated_table, check_metadata=False)
        cited_keys = _text_metadata(cited_table.schema.metadata)
        restated_keys = _text_metadata(restated_table.schema.metadata)
        assert set(cited_keys) == set(restated_keys)
        differing = {name for name in cited_keys if cited_keys[name] != restated_keys[name]}
        assert differing == {MANIFEST_METADATA_KEY}
        assert from_metadata(cited_table.schema.metadata) == cited
        assert from_metadata(restated_table.schema.metadata) == restated
        assert read_frames(cited_bytes) == read_frames(restated_bytes) == run.record.frames
