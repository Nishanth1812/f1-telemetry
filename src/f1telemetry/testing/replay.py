"""Finite source for replaying saved sensor frames through the live transport.

Reads a saved Parquet run once and re-emits it as :class:`~f1telemetry.synthetic.Frame`
batches on exactly the live contract, so the dashboard cannot tell replay from a
running car. A run's saved fault annotations - the P5-T4 ground truth persisted in
the Parquet metadata by :mod:`f1telemetry.telemetry.pipeline` - can be attached at
construction and ride out as optional frame events at their simulated onset, so the
dashboard's event view shows what was injected without rerunning anything. Session
event channels (``lap_index`` and friends) need no special handling here: they are
ordinary columns in the saved run and pass through in ``channels`` untouched.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from pathlib import Path

from f1telemetry.synthetic import Frame, FrameEvent
from f1telemetry.telemetry.sensors import FaultAnnotation
from f1telemetry.testing.parquet_io import read_frames
from f1telemetry.testing.records import SensorFrame


def _onset_us(annotation: FaultAnnotation) -> int:
    """The annotation's simulated onset in integer microseconds."""
    return round(annotation.onset_t_s * 1_000_000)


class ReplaySource:
    """Read a Parquet run once and expose its values as dashboard frames.

    Args:
        path: the saved run to replay.
        annotations: fault ground truth persisted with the run, typically from
            :func:`f1telemetry.telemetry.pipeline.read_annotations`. Each one is
            emitted exactly once, in onset order, as a ``kind="fault"`` frame
            event: attached to the first frame emitted at or after its onset, or -
            when the stored frames run out first - as an event-only frame stamped
            at the onset itself. Event emission is never rate-limited: events are
            sparse, and dropping one would hide an injected fault from the
            dashboard.
    """

    def __init__(self, path: Path, *, annotations: Iterable[FaultAnnotation] = ()) -> None:
        self._frames: tuple[SensorFrame, ...] = read_frames(path)
        self._annotations: tuple[FaultAnnotation, ...] = tuple(
            sorted(annotations, key=lambda annotation: annotation.onset_t_s)
        )
        self._index: int = 0
        self._annotation_index: int = 0
        self._now_us: int = int(self._frames[0].t_s * 1_000_000) if self._frames else 0
        self._last_emit_us: int | None = None

    @property
    def now_us(self) -> int:
        return self._now_us

    @property
    def annotations(self) -> tuple[FaultAnnotation, ...]:
        """The attached fault annotations, in onset order."""
        return self._annotations

    def reset(self) -> None:
        """Rewind to the first frame so replay can start again."""
        self._index = 0
        self._annotation_index = 0
        self._now_us = int(self._frames[0].t_s * 1_000_000) if self._frames else 0
        self._last_emit_us = None

    @property
    def exhausted(self) -> bool:
        return self._index >= len(self._frames) and self._annotation_index >= len(self._annotations)

    def iter_frames(self, duration_us: int, *, max_hz: float) -> Iterator[Frame]:
        """Yield stored samples in a half-open time window, limited to ``max_hz``.

        Stored frames keep the existing tick-limited behaviour; fault events ride
        the frames that survive the tick, and any annotation still pending when the
        stored frames run out is yielded as an event-only frame (``channels`` is
        empty) at its onset, provided the onset falls inside the window.
        """
        if duration_us < 0:
            raise ValueError("duration_us must be non-negative")
        if not math.isfinite(max_hz) or max_hz <= 0.0:
            raise ValueError("max_hz must be positive and finite")
        tick_us = math.ceil(1_000_000 / max_hz)
        end_us = self._now_us + duration_us
        while self._index < len(self._frames):
            frame = self._frames[self._index]
            time_us = round(frame.t_s * 1_000_000)
            if time_us >= end_us:
                break
            self._index += 1
            self._now_us = time_us
            if self._last_emit_us is not None and time_us - self._last_emit_us < tick_us:
                continue
            self._last_emit_us = time_us
            yield Frame(time_us, dict(frame.values), self._due_events(time_us))
        while self._annotation_index < len(self._annotations):
            onset_us = _onset_us(self._annotations[self._annotation_index])
            if onset_us >= end_us:
                break
            # Drains every annotation sharing this onset, so each distinct onset
            # yields at most one frame and wire timestamps stay non-decreasing:
            # everything already emitted was drained at a strictly earlier time.
            events = self._due_events(onset_us)
            self._now_us = max(self._now_us, onset_us)
            yield Frame(onset_us, {}, events)
        self._now_us = max(self._now_us, end_us)

    def _due_events(self, time_us: int) -> tuple[FrameEvent, ...]:
        """Drain pending annotations with onset at or before ``time_us``, in order."""
        events: list[FrameEvent] = []
        while self._annotation_index < len(self._annotations):
            annotation = self._annotations[self._annotation_index]
            onset_us = _onset_us(annotation)
            if onset_us > time_us:
                break
            events.append(
                FrameEvent(
                    kind="fault",
                    time_us=onset_us,
                    fault_type=annotation.fault_type,
                    channel=annotation.channel_name or None,
                    severity=float(annotation.severity),
                    duration_samples=annotation.duration_samples,
                    label=bool(annotation.label),
                )
            )
            self._annotation_index += 1
        return tuple(events)
