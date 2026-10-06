"""Unified P5 bridge: record -> sensor processing -> CAN-FD -> Parquet -> replay.

One narrow path, no new storage abstraction: a finished :class:`SampleRecord`
is pushed through contract-driven sensor processing (quantise, seeded noise,
seeded fault injection with separate ground-truth annotations), packed into
decodable CAN-FD frames, serialised byte-identically with the existing
Parquet writer, and re-read for replay. Simulated time only; nothing here
reads a wall clock or re-runs physics.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from f1telemetry.contracts.channels import (
    ChannelContract,
    load_channel_contract,
)
from f1telemetry.telemetry.bus import (
    DEFAULT_BITRATE_BPS,
    MessageSpec,
    ScheduledMessage,
    bus_utilisation,
    message_specs_from_contract,
    schedule,
)
from f1telemetry.telemetry.canfd import decode_frame, encode_frame, next_counter
from f1telemetry.telemetry.sensors import (
    FAULT_TYPES_ORDERED,
    FaultAnnotation,
    FaultParameters,
    apply_fault,
    inject_noise,
    quantise_array,
)
from f1telemetry.testing.parquet_io import (
    METADATA,
    read_frames,
    serialise_frames,
)
from f1telemetry.testing.records import SampleRecord, SensorFrame
from f1telemetry.testing.replay import ReplaySource

__all__ = [
    "BusPlan",
    "CanChunk",
    "ProcessedTelemetry",
    "bus_plan_for_record",
    "process_to_parquet",
    "read_annotations",
    "replay_bytes",
    "run_bridge",
]

# 64 payload bytes is the CAN-FD maximum and also the widest legal DLC for
# this columnar codec. Group channels into whole float32 slots the codec
# accepts, largest first, so every chunk is encodable on the wire.
_SLOT_FLOATS: Final[tuple[int, ...]] = (16, 12, 8, 6, 5, 4, 3, 2, 1)
_BASE_IDENTIFIER: Final[int] = 0x100


def _pack_slots(count: int) -> list[int]:
    parts: list[int] = []
    remaining = count
    while remaining > 0:
        take = max(size for size in _SLOT_FLOATS if size <= remaining)
        parts.append(take)
        remaining -= take
    return parts


@dataclass(frozen=True, slots=True)
class CanChunk:
    """Manifest entry for one CAN-FD frame on the wire."""

    t_s: float
    channels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BusPlan:
    """Contract-derived CAN schedule for one run, in simulation time only.

    ``specs`` packs every periodic contract channel by its declared rate;
    ``schedule`` realizes those releases over the run's simulated span with
    identifier-priority arbitration; ``utilisation`` is the serialized bus
    occupancy at ``bitrate_bps``. No wall clock is read anywhere.
    """

    specs: tuple[MessageSpec, ...] = ()
    schedule: tuple[ScheduledMessage, ...] = ()
    utilisation: float = 0.0
    duration_us: int = 0
    bitrate_bps: int = DEFAULT_BITRATE_BPS


@dataclass(frozen=True, slots=True)
class ProcessedTelemetry:
    """The five artefacts of one unified pass through the P5 bridge."""

    ground_truth: tuple[SensorFrame, ...]
    frames: tuple[SensorFrame, ...]
    annotations: tuple[FaultAnnotation, ...]
    canfd: tuple[bytes, ...]
    canfd_manifest: tuple[CanChunk, ...]
    parquet: bytes
    name_order: tuple[str, ...] = field(default=())
    bus_plan: BusPlan = field(default_factory=BusPlan)

    def replay(self) -> tuple[SensorFrame, ...]:
        return read_frames(self.parquet)

    def saved_annotations(self) -> tuple[FaultAnnotation, ...]:
        return read_annotations(self.parquet)

    def replay_source(self, path: Path) -> ReplaySource:
        path.write_bytes(self.parquet)
        return ReplaySource(path, annotations=read_annotations(path))


# Parquet key-value metadata key carrying the fault ground truth. The
# payload is a compact JSON array, same order as ``annotations``, so the
# saved bytes stay deterministic and old Parquet files (no key) read back
# as "no annotations".
_ANNOTATIONS_KEY: Final[str] = "f1telemetry/fault_annotations"


def _annotations_payload(annotations: Sequence[FaultAnnotation]) -> str:
    rows = [
        {
            "fault_type": a.fault_type,
            "severity": float(a.severity),
            "onset_t_s": float(a.onset_t_s),
            "ground_truth_t_s": float(a.ground_truth_t_s),
            "label": bool(a.label),
            "channel_name": a.channel_name,
            "duration_samples": a.duration_samples,
            "seed": a.seed,
        }
        for a in annotations
    ]
    return json.dumps(rows, separators=(",", ":"))


def read_annotations(source: bytes | Path) -> tuple[FaultAnnotation, ...]:
    """Fault ground truth persisted with a saved run, oldest first.

    Absent key (legacy Parquet bytes) or an empty payload reads as ``()``.
    """
    table = (
        pq.read_table(pa.BufferReader(source))
        if isinstance(source, bytes)
        else pq.read_table(source)
    )
    metadata = table.schema.metadata or {}
    raw = metadata.get(_ANNOTATIONS_KEY.encode())
    if raw is None:
        return ()
    rows = json.loads(raw.decode("utf-8"))
    return tuple(
        FaultAnnotation(
            fault_type=str(row["fault_type"]),
            severity=float(row["severity"]),
            onset_t_s=float(row["onset_t_s"]),
            ground_truth_t_s=float(row["ground_truth_t_s"]),
            label=bool(row["label"]),
            channel_name=str(row["channel_name"]),
            duration_samples=(
                None if row["duration_samples"] is None else int(row["duration_samples"])
            ),
            seed=None if row["seed"] is None else int(row["seed"]),
        )
        for row in rows
    )


def _channel_arrays(record: SampleRecord) -> dict[str, np.ndarray]:
    if not record.frames:
        return {}
    names = list(record.frames[0].values)
    return {
        name: np.asarray([frame.values[name] for frame in record.frames], dtype=np.float64)
        for name in names
    }


def process_to_parquet(
    record: SampleRecord,
    contract: ChannelContract | None = None,
    *,
    seed: int = 0,
    active_faults: Sequence[tuple[str, FaultParameters]] | None = None,
) -> tuple[tuple[SensorFrame, ...], tuple[FaultAnnotation, ...]]:
    """Record frames -> contract sensor processing -> corrupted frames.

    The record's own frames are the decimated/published stream, so this stage
    applies quantisation, seeded noise and seeded fault injection per channel,
    keeping the original frames untouched as ground truth.
    """
    contract = load_channel_contract() if contract is None else contract
    channels = _channel_arrays(record)
    rng = np.random.default_rng(int(seed))
    clean: dict[str, np.ndarray] = {}
    for name, values in channels.items():
        ch = contract.by_name(name)  # KeyError -> contract error at boundary
        quantised = quantise_array(values, ch.quantisation)
        clean[name] = np.asarray(
            inject_noise(quantised, float(ch.sigma), ch.noise_model, rng=rng),
            dtype=np.float32,
        )

    corrupted = {name: values.copy() for name, values in clean.items()}
    annotations: list[FaultAnnotation] = []
    dt = float(record.dt_s)
    for fault_type, params in active_faults or ():
        if fault_type not in FAULT_TYPES_ORDERED:
            raise ValueError(f"unknown fault_type {fault_type!r}")
        seen_pairs: set[tuple[str, str]] = set()
        for name in channels:
            ch = contract.by_name(name)
            if fault_type not in ch.fault_eligible:
                continue
            start = min(params.onset_index, len(corrupted[name]))
            stop = (
                len(corrupted[name])
                if params.duration_samples is None
                else min(len(corrupted[name]), params.onset_index + params.duration_samples)
            )
            if start >= stop:
                continue
            if fault_type == "swap":
                if ch.corner is None or params.swap_target_corner is None:
                    raise ValueError("swap requires a corner channel and swap_target_corner")
                target = f"{ch.base_name}_{params.swap_target_corner.lower()}"
                if (
                    target not in corrupted
                    or fault_type not in contract.by_name(target).fault_eligible
                ):
                    raise ValueError(f"swap target {target!r} is not an eligible channel")
                pair = (min(name, target), max(name, target))
                if pair in seen_pairs or len(corrupted[name]) != len(corrupted[target]):
                    continue
                left, right = corrupted[name].copy(), corrupted[target].copy()
                corrupted[name][start:stop] = right[start:stop]
                corrupted[target][start:stop] = left[start:stop]
                seen_pairs.add(pair)
                targets = (name, target)
            else:
                snapshot = corrupted[name].copy()
                for index in range(start, stop):
                    corrupted[name][index] = apply_fault(
                        float(snapshot[index]),
                        ch,
                        fault_type,
                        params,
                        index=index,
                        ground_truth_array=snapshot,
                        seed=seed,
                    )
                targets = (name,)
            for target_name in targets:
                annotations.append(
                    FaultAnnotation(
                        fault_type=fault_type,
                        severity=params.severity,
                        onset_t_s=params.onset_index * dt,
                        ground_truth_t_s=max(
                            0,
                            params.onset_index - params.stale_samples
                            if fault_type == "stale"
                            else params.onset_index,
                        )
                        * dt,
                        channel_name=target_name,
                        duration_samples=params.duration_samples,
                        seed=seed,
                    )
                )

    frames = tuple(
        SensorFrame(
            t_s=float(record.frames[i].t_s),
            values={name: float(corrupted[name][i]) for name in channels},
        )
        for i in range(len(record.frames))
    )
    return frames, tuple(annotations)


def encode_canfd(frames: Sequence[SensorFrame]) -> tuple[tuple[bytes, ...], tuple[CanChunk, ...]]:
    """Pack processed frames column-wise into decodable CAN-FD payloads."""
    wire: list[bytes] = []
    manifest: list[CanChunk] = []
    counter = 0
    chunk_index = 0
    for frame in frames:
        names = sorted(frame.values)
        timestamp_us = round(float(frame.t_s) * 1_000_000)
        for group in _iter_chunks(names):
            payload = struct.pack(f"<{len(group)}f", *(float(frame.values[n]) for n in group))
            wire.append(
                encode_frame(
                    _BASE_IDENTIFIER + chunk_index,
                    counter,
                    timestamp_us,
                    payload,
                )
            )
            manifest.append(CanChunk(t_s=float(frame.t_s), channels=tuple(group)))
            counter = next_counter(counter)
            chunk_index += 1
    return tuple(wire), tuple(manifest)


def _iter_chunks(names: Sequence[str]) -> list[tuple[str, ...]]:
    groups: list[tuple[str, ...]] = []
    offset = 0
    for take in _pack_slots(len(names)):
        groups.append(tuple(names[offset : offset + take]))
        offset += take
    return groups


def decode_canfd_values(
    wire: Sequence[bytes], manifest: Sequence[CanChunk]
) -> list[tuple[float, dict[str, float]]]:
    """Decode wire frames back into ``(t_s, {channel: value})`` pairs."""
    if len(wire) != len(manifest):
        raise ValueError("wire/manifest length mismatch")
    out: list[tuple[float, dict[str, float]]] = []
    for blob, chunk in zip(wire, manifest, strict=True):
        frame = decode_frame(blob)
        values = struct.unpack(f"<{len(chunk.channels)}f", frame.payload)
        if frame.timestamp_us != round(chunk.t_s * 1_000_000):
            raise ValueError("CAN-FD timestamp does not match the manifest")
        out.append((chunk.t_s, dict(zip(chunk.channels, values, strict=True))))
    return out


def bus_plan_for_record(
    record: SampleRecord,
    contract: ChannelContract | None = None,
    *,
    bitrate_bps: int = DEFAULT_BITRATE_BPS,
) -> BusPlan:
    """Derive the contract CAN schedule over a run's simulated span.

    The duration comes from the record's simulation timestamps
    (``frames[-1].t_s - frames[0].t_s + dt_s``), so replaying the same
    record always yields the same releases. Frame encoding itself is
    unchanged; this only schedules the declared rates and measures load.
    """
    contract = load_channel_contract() if contract is None else contract
    specs = message_specs_from_contract(contract)
    if not record.frames:
        duration_us = 0
    else:
        span_s = float(record.frames[-1].t_s) - float(record.frames[0].t_s) + float(record.dt_s)
        duration_us = max(0, round(span_s * 1_000_000))
    if not specs:
        return BusPlan((), (), 0.0, duration_us, bitrate_bps)
    utilisation = bus_utilisation(specs, bitrate_bps)
    events = schedule(specs, duration_us, bitrate_bps) if duration_us else ()
    return BusPlan(specs, events, utilisation, duration_us, bitrate_bps)


def run_bridge(
    record: SampleRecord,
    contract: ChannelContract | None = None,
    *,
    seed: int = 0,
    active_faults: Sequence[tuple[str, FaultParameters]] | None = None,
    bitrate_bps: int = DEFAULT_BITRATE_BPS,
) -> ProcessedTelemetry:
    """The whole unified path: sensors -> CAN-FD -> Parquet, ready to replay."""
    contract = load_channel_contract() if contract is None else contract
    frames, annotations = process_to_parquet(
        record, contract, seed=seed, active_faults=active_faults
    )
    wire, manifest = encode_canfd(frames)
    processed_record = SampleRecord(
        name=record.name,
        dt_s=record.dt_s,
        description=record.description,
        ground_truth=record.ground_truth,
        frames=frames,
    )
    parquet = serialise_frames(
        processed_record,
        metadata={
            **METADATA,
            _ANNOTATIONS_KEY: _annotations_payload(annotations),
        },
    )
    return ProcessedTelemetry(
        ground_truth=record.frames,
        frames=frames,
        annotations=annotations,
        canfd=wire,
        canfd_manifest=manifest,
        parquet=parquet,
        name_order=tuple(frames[0].values) if frames else (),
        bus_plan=bus_plan_for_record(record, contract, bitrate_bps=bitrate_bps),
    )


def replay_bytes(parquet: bytes) -> tuple[SensorFrame, ...]:
    """Logical frames straight from a saved Parquet payload, no physics."""
    return read_frames(parquet)
