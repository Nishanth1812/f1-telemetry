"""Deterministic Parquet serialisation of a record's published frames.

The warm tier writes one Parquet file per group per run (PLAN.md section 8.3), and
invariant 8 is the gate on the whole storage path: same input, byte-identical file. That
property is easy to lose to a wall-clock value in key-value metadata, a non-deterministic
row order, or dictionary-derived column ordering, so it is built and tested here at P0
rather than discovered in P5.

What this module deliberately does *not* do is invent a run: it serialises exactly the
frames a record carries, in contract order, and nothing else.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import pyarrow as pa
import pyarrow.parquet as pq

from f1telemetry.contracts.channels import DTYPES
from f1telemetry.testing.records import SampleRecord

__all__ = [
    "ARROW_TYPES",
    "COMPRESSION",
    "METADATA",
    "TIME_COLUMN",
    "build_table",
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


def build_table(record: SampleRecord, metadata: Mapping[str, str] | None = None) -> pa.Table:
    """Arrow table of the record's published frames: time column, then contract order.

    Columns are float64 regardless of the contract dtype. That is deliberate: this
    serialiser exists to prove byte-identical output, and narrowing a fixture value to
    the sensor's real bit depth is *quantisation*, which is P5's `sensors` job driven by
    `ARROW_TYPES` and the per-group schema in `generated/parquet_schema.py`. Rounding here
    would put an invented sensor model in the determinism gate.

    `metadata` is Parquet key-value metadata. P6-T7 puts the seed, spec version, scenario
    version and git SHA in here, which is legitimate because they are the *same* on every
    re-run of the same scenario. A wall-clock value in this mapping is what invariant 8
    exists to catch, and the test for that leak is in the suite.
    """
    channels = record.channels
    merged = dict(METADATA if metadata is None else metadata)
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
