"""Durable Scenario Lab runs: one immutable Parquet file per saved run.

A saved run is self-contained. Its frames are the deterministic Parquet written by
:func:`~f1telemetry.testing.parquet_io.serialise_frames`, and its summary, request settings and
configuration/code provenance travel in the same file's key-value metadata. Files are named by an
opaque UUID hex, never by caller input, and are published by writing a temporary file, fsyncing
it, and hard-linking it into place. ``os.link`` refuses an existing name, so a run is never
replaced.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Final

import pyarrow as pa
import pyarrow.parquet as pq

from f1telemetry.synthetic import Frame, FrameEvent
from f1telemetry.testing.parquet_io import read_frames, serialise_frames
from f1telemetry.testing.records import SampleRecord, SensorFrame
from f1telemetry.testing.scenario_lab import LabRun

__all__ = ["RunStore", "RunStoreError"]

FORMAT_VERSION: Final[int] = 1
DOCUMENT_KEY: Final[str] = "f1telemetry/scenario_run"
EVENTS_KEY: Final[str] = "f1telemetry/frame_events"
_RUN_ID: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{32}")
_RUN_FILE: Final[re.Pattern[str]] = re.compile(r"([0-9a-f]{32})\.parquet")


def _invalid_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value}")


class RunStoreError(ValueError):
    """A run cannot be saved, found, or read back; the message says which and why."""


def _new_run_id() -> str:
    return uuid.uuid4().hex


def _checked_id(run_id: object) -> str:
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise RunStoreError("run id must be 32 lower-case hexadecimal characters")
    return run_id


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":  # Windows cannot open a directory to fsync it; the file fsync stands.
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _document(
    metadata: dict[bytes, bytes] | None, run_id: str
) -> tuple[dict[str, object], dict[str, object] | None]:
    """The summary and provenance a file carries. Anything else in the document is corruption."""
    raw = None if metadata is None else metadata.get(DOCUMENT_KEY.encode())
    if raw is None:
        raise RunStoreError(f"run {run_id} is corrupt: no run document")
    try:
        document: object = json.loads(raw, parse_constant=_invalid_json_constant)
    except ValueError as exc:
        raise RunStoreError(f"run {run_id} is corrupt: run document is not JSON") from exc
    if not isinstance(document, dict) or set(document) != {
        "format_version",
        "run_id",
        "result",
        "provenance",
    }:
        raise RunStoreError(f"run {run_id} is corrupt: run document has the wrong fields")
    result: object = document["result"]
    provenance: object = document["provenance"]
    if (
        type(document["format_version"]) is not int
        or document["format_version"] != FORMAT_VERSION
        or document["run_id"] != run_id
        or not isinstance(result, dict)
        or not (provenance is None or isinstance(provenance, dict))
    ):
        raise RunStoreError(f"run {run_id} is corrupt: run document does not match its file")
    return result, provenance


def _record(run_id: str, frames: tuple[Frame, ...]) -> SampleRecord:
    """Published frames as a record. Times are stored as seconds of the exact microsecond count."""
    if not frames:
        raise RunStoreError("run must contain frames")
    channels = tuple(frames[0].channels)
    sensor_frames: list[SensorFrame] = []
    previous = -1
    for frame in frames:
        if frame.samples:
            raise RunStoreError("run frames must be original samples, not delivery batches")
        if isinstance(frame.time_us, bool) or not isinstance(frame.time_us, int):
            raise RunStoreError("run timestamps must be integer microseconds")
        if frame.time_us <= previous:
            raise RunStoreError("run timestamps must be non-negative and strictly increasing")
        previous = frame.time_us
        if tuple(frame.channels) != channels:
            raise RunStoreError("run frames must all carry the same channels")
        if not all(math.isfinite(value) for value in frame.channels.values()):
            raise RunStoreError("run frames must hold only finite values")
        sensor_frames.append(SensorFrame(frame.time_us / 1_000_000, dict(frame.channels)))
    dt_s = sensor_frames[1].t_s - sensor_frames[0].t_s if len(sensor_frames) > 1 else 0.0
    return SampleRecord(run_id, dt_s, "", (), tuple(sensor_frames))


class RunStore:
    """Saved runs under one server-owned directory. Ids are opaque; callers never give paths."""

    def __init__(self, directory: Path) -> None:
        self._directory: Path = Path(directory)

    def save(self, lab_run: LabRun) -> dict[str, object]:
        """Publish a run and return its summary with ``run_id``. Nothing is replaced."""
        run_id = _new_run_id()
        result = {**lab_run.result, "run_id": run_id}
        try:
            payload = json.dumps(
                {
                    "format_version": FORMAT_VERSION,
                    "run_id": run_id,
                    "result": result,
                    "provenance": lab_run.provenance,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise RunStoreError(f"run {run_id} cannot be stored as JSON: {exc}") from exc
        events = {
            str(index): [asdict(event) for event in frame.events]
            for index, frame in enumerate(lab_run.frames)
            if frame.events
        }
        try:
            events_payload = json.dumps(events, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise RunStoreError(f"run {run_id} has invalid events: {exc}") from exc
        data = serialise_frames(
            _record(run_id, lab_run.frames), {DOCUMENT_KEY: payload, EVENTS_KEY: events_payload}
        )
        self._publish(run_id, data)
        return result

    def _publish(self, run_id: str, data: bytes) -> None:
        final = self._directory / f"{run_id}.parquet"
        temporary: str | None = None
        published = False
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(dir=self._directory, prefix=".", suffix=".tmp")
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, final)  # Raises if the name exists: never a replacement.
            published = True
            _fsync_directory(self._directory)
        except OSError as exc:
            if published:  # Undo a publication whose directory entry could not be made durable.
                final.unlink(missing_ok=True)
            raise RunStoreError(f"run {run_id} was not published: {exc}") from exc
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    def list_runs(self) -> list[dict[str, object]]:
        """Summaries of every saved run, newest first. Corrupt files raise, naming the run."""
        if not self._directory.is_dir():
            return []
        found = [
            (path, match.group(1))
            for path in self._directory.iterdir()
            if (match := _RUN_FILE.fullmatch(path.name)) is not None
        ]
        found.sort(key=lambda item: (item[0].stat().st_mtime_ns, item[1]), reverse=True)
        summaries: list[dict[str, object]] = []
        for path, run_id in found:
            try:
                metadata = pq.read_metadata(path).metadata
            except (OSError, ValueError) as exc:
                raise RunStoreError(f"run {run_id} is corrupt: {exc}") from exc
            summaries.append(_document(metadata, run_id)[0])
        return summaries

    def load(self, run_id: str) -> LabRun:
        """Read a run back with its original frames, summary and provenance."""
        checked = _checked_id(run_id)
        try:
            data = (self._directory / f"{checked}.parquet").read_bytes()
        except FileNotFoundError as exc:
            raise RunStoreError(f"run {checked} not found") from exc
        except OSError as exc:
            raise RunStoreError(f"run {checked} could not be read: {exc}") from exc
        try:
            metadata = pq.read_metadata(pa.BufferReader(data)).metadata
            frames = read_frames(data)
        except (OSError, ValueError) as exc:
            raise RunStoreError(f"run {checked} is corrupt: {exc}") from exc
        result, provenance = _document(metadata, checked)
        try:
            event_rows = json.loads(
                (metadata or {}).get(EVENTS_KEY.encode(), b"{}"),
                parse_constant=_invalid_json_constant,
            )
            if not isinstance(event_rows, dict):
                raise ValueError("event document must be a mapping")
            events = {
                int(index): tuple(FrameEvent(**row) for row in rows)
                for index, rows in event_rows.items()
            }
            if any(index < 0 or index >= len(frames) for index in events):
                raise ValueError("event frame index is outside the run")
            if any(
                not isinstance(event.kind, str)
                or type(event.time_us) is not int
                or event.time_us < 0
                for rows in events.values()
                for event in rows
            ):
                raise ValueError("invalid event kind or timestamp")
        except (ValueError, TypeError, KeyError) as exc:
            raise RunStoreError(f"run {checked} is corrupt: invalid events") from exc
        loaded = LabRun(
            result=result,
            frames=tuple(
                Frame(
                    time_us=round(frame.t_s * 1_000_000),
                    channels=dict(frame.values),
                    events=events.get(index, ()),
                )
                for index, frame in enumerate(frames)
            ),
            provenance=provenance,
        )
        _record(checked, loaded.frames)
        return loaded
