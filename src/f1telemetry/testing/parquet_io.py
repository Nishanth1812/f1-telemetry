"""Deterministic Parquet serialisation of a record's published frames.

The warm tier writes one Parquet file per group per run (PLAN.md section 8.3), and
invariant 8 is the gate on the whole storage path: same input, byte-identical file. That
property is easy to lose to a wall-clock value in key-value metadata, a non-deterministic
row order, or dictionary-derived column ordering, so it is built and tested here at P0
rather than discovered in P5.

What this module deliberately does *not* do is invent a run: it serialises exactly the
frames a record carries, in contract order, and nothing else. Key-value metadata follows the
same rule. :data:`METADATA` is the floor every file carries, caller keys merge over it, and a
:class:`~f1telemetry.testing.run_manifest.RunManifest` is merged in last - a citation of the
five facts a run declares, never a value resolved from this side of the writer.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pyarrow as pa
import pyarrow.parquet as pq

from f1telemetry.contracts.channels import DTYPES
from f1telemetry.testing.records import SampleRecord, SensorFrame
from f1telemetry.testing.run_manifest import RunManifest

__all__ = [
    "ARROW_TYPES",
    "COMPRESSION",
    "METADATA",
    "TIME_COLUMN",
    "build_table",
    "metadata_with_manifest",
    "read_frames",
    "serialise_frames",
    "write_frames",
]

COMPRESSION: Final[str] = "zstd"
COMPRESSION_LEVEL: Final[int] = 3
TIME_COLUMN: Final[str] = "t_s"
METADATA: Final[dict[str, str]] = {
    "contract": "f1-telemetry/phase0",
    "note": "deterministic fixture serialisation; no wall-clock value may be added here",
}


def _arrow_type(dtype: str) -> pa.DataType:
    """Contract dtype -> Arrow type. Used by the P5 writer; see the note on build_table."""
    return {
        "float32": pa.float32(),
        "float64": pa.float64(),
        "int8": pa.int8(),
        "int16": pa.int16(),
        "int32": pa.int32(),
        "uint8": pa.uint8(),
        "uint16": pa.uint16(),
        "uint32": pa.uint32(),
        "bool": pa.bool_(),
    }[dtype]


ARROW_TYPES: Final[Mapping[str, pa.DataType]] = MappingProxyType(
    {dtype: _arrow_type(dtype) for dtype in DTYPES}
)


def metadata_with_manifest(
    manifest: RunManifest | None = None,
    metadata: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Key-value metadata for a file that cites ``manifest``: defaults, then caller, then manifest.

    The three layers merge rather than replace, in that order, so :data:`METADATA` - the contract
    name and the determinism note every file carries - survives on a file that adds anything of its
    own. ``metadata`` is whatever the caller wanted alongside the defaults; ``manifest`` is written
    last so :data:`~f1telemetry.testing.run_manifest.MANIFEST_METADATA_KEY` always holds the
    manifest the caller supplied. Every value is the caller's: nothing here resolves a car spec,
    a scenario, a repository or a clock.

    Returns a plain dict rather than writing a file, so a caller can inspect or extend it first -
    the writer merges again and cannot lose a key this way.
    """
    merged = {**METADATA, **(metadata or {})}
    if manifest is not None:
        merged.update(manifest.to_metadata())
    return merged


def build_table(record: SampleRecord, metadata: Mapping[str, str] | None = None) -> pa.Table:
    """Arrow table of the record's published frames: time column, then contract order.

    Columns are float64 regardless of the contract dtype. That is deliberate: this
    serialiser exists to prove byte-identical output, and narrowing a fixture value to
    the sensor's real bit depth is *quantisation*, which is P5's `sensors` job driven by
    `ARROW_TYPES` and the per-group schema in `generated/parquet_schema.py`. Rounding here
    would put an invented sensor model in the determinism gate.

    `metadata` is caller-supplied Parquet key-value metadata, and it is *added to*
    :data:`METADATA` rather than put in its place: a caller naming its own keys has not
    asked to drop the contract name or the determinism note, and a file that cites a run
    manifest still has to say which contract it obeys. P6-T7 can put the seed, car-spec
    version, scenario version, setup hash and git SHA in here - :func:`metadata_with_manifest`
    is the shape of that call - and this writer still infers none of them. A wall-clock value
    in this mapping is what invariant 8 exists to catch, and the test for that leak is in the
    suite.
    """
    channels = record.channels
    merged = {**METADATA, **(metadata or {})}
    schema = pa.schema(
        [pa.field(TIME_COLUMN, pa.float64(), nullable=False)]
        + [pa.field(name, pa.float64(), nullable=False) for name in channels],
        metadata={k.encode(): v.encode() for k, v in sorted(merged.items())},
    )
    columns: list[pa.Array] = [pa.array([frame.t_s for frame in record.frames], pa.float64())]
    columns.extend(
        pa.array([frame.values[name] for frame in record.frames], pa.float64()) for name in channels
    )
    return pa.Table.from_arrays(columns, schema=schema)


def write_frames(
    record: SampleRecord, metadata: Mapping[str, str] | None = None
) -> pa.BufferOutputStream:
    """Serialise to an in-memory Parquet buffer with deterministic settings."""
    sink = pa.BufferOutputStream()
    pq.write_table(
        build_table(record, metadata),
        sink,
        compression=COMPRESSION,
        compression_level=COMPRESSION_LEVEL,
        version="2.6",
        write_statistics=True,
        store_schema=True,
    )
    return sink


def serialise_frames(record: SampleRecord, metadata: Mapping[str, str] | None = None) -> bytes:
    """Byte-identical Parquet encoding of the record's published frames."""
    return write_frames(record, metadata).getvalue().to_pybytes()


def read_frames(source: bytes | Path) -> tuple[SensorFrame, ...]:
    """Read saved logical frames for replay without rerunning physics."""
    table = (
        pq.read_table(pa.BufferReader(source))
        if isinstance(source, bytes)
        else pq.read_table(source)
    )
    if TIME_COLUMN not in table.column_names:
        raise ValueError(f"Parquet replay requires {TIME_COLUMN!r}")
    times = table[TIME_COLUMN].to_pylist()
    names = [name for name in table.column_names if name != TIME_COLUMN]
    columns = {name: table[name].to_pylist() for name in names}
    frames: list[SensorFrame] = []
    previous = -1.0
    for index, timestamp in enumerate(times):
        if timestamp is None or not math.isfinite(timestamp) or timestamp <= previous:
            raise ValueError("Parquet replay timestamps must be finite and strictly increasing")
        previous = float(timestamp)
        frames.append(
            SensorFrame(
                t_s=float(timestamp),
                values={name: float(columns[name][index]) for name in names},
            )
        )
    return tuple(frames)
