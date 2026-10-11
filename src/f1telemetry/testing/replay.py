"""Finite source for replaying saved sensor frames through the live transport.

Reads a saved Parquet run once and re-emits it as :class:`~f1telemetry.synthetic.Frame`
batches on exactly the live contract, so the dashboard cannot tell replay from a
running car. The delivery grid is the one
:meth:`~f1telemetry.synthetic.SyntheticSource.iter_frames` uses - anchored at ``t = 0``,
``ceil(1e6 / max_hz)`` microseconds per tick - with the newest stored value of every
channel on each tick and the stored frames that tick aggregated into
:attr:`~f1telemetry.synthetic.Frame.samples`, so a saved 200 Hz run is still countable
at 200 samples per simulated second behind a 30 Hz feed. A run's saved fault
annotations - the P5-T4 ground truth persisted in the Parquet metadata by
:mod:`f1telemetry.telemetry.pipeline` - can be attached at construction and ride out as
optional frame events at their simulated onset, so the dashboard's event view shows
what was injected without rerunning anything. Session event channels (``lap_index`` and
friends) need no special handling here: they are ordinary columns in the saved run and
pass through in ``channels`` untouched.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from pathlib import Path

from f1telemetry.synthetic import BROADCAST_HZ, Frame, FrameEvent
from f1telemetry.telemetry.sensors import FaultAnnotation
from f1telemetry.testing.parquet_io import read_frames
from f1telemetry.testing.records import SensorFrame

_ONE_SECOND_US = 1_000_000


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
            event: attached to the first delivery frame at or after its onset, which
            may be long after the stored frames run out. Event emission is never
            rate-limited: events are sparse, and dropping one would hide an injected
            fault from the dashboard.
    """

    def __init__(self, path: Path, *, annotations: Iterable[FaultAnnotation] = ()) -> None:
        self._frames: tuple[SensorFrame, ...] = read_frames(path)
        self._times: tuple[int, ...] = tuple(
            round(float(frame.t_s) * _ONE_SECOND_US) for frame in self._frames
        )
        self._annotations: tuple[FaultAnnotation, ...] = tuple(
            sorted(annotations, key=lambda annotation: annotation.onset_t_s)
        )
        self._rewind()

    def _rewind(self) -> None:
        """Simulated time back to ``t = 0``, as :meth:`SyntheticSource.reset` does it."""
        self._index: int = 0
        self._annotation_index: int = 0
        self._now_us: int = 0
        self._latest: dict[str, float] = {}

    @property
    def now_us(self) -> int:
        return self._now_us

    @property
    def annotations(self) -> tuple[FaultAnnotation, ...]:
        """The attached fault annotations, in onset order."""
        return self._annotations

    def reset(self) -> None:
        """Rewind to the first frame so replay can start again."""
        self._rewind()

    @property
    def exhausted(self) -> bool:
        return self._index >= len(self._frames) and self._annotation_index >= len(self._annotations)

    def iter_frames(self, duration_us: int, *, max_hz: float = BROADCAST_HZ) -> Iterator[Frame]:
        """Yield delivery frames on the live grid, aggregated from the stored frames.

        The tick is ``ceil(1e6 / max_hz)`` microseconds on a grid anchored at
        ``t = 0``, exactly as in :meth:`~f1telemetry.synthetic.SyntheticSource.iter_frames`,
        and the window is half-open, so slicing a run into consecutive chunks neither
        repeats a timestamp nor skips one. Each frame carries the newest stored value
        of every channel, and every stored frame the tick aggregated goes out in
        :attr:`~f1telemetry.synthetic.Frame.samples` - the declared rates of the saved
        run stay countable downstream while ``channels`` stays the newest-value
        snapshot the tiles read.

        Unlike the live source, which samples forever, replay is finite: a tick with
        no stored frame behind it and no event due says nothing, so it is not
        delivered. The grid is therefore the live grid over the run's span rather than
        a restatement of the last values forever, and the transport still sees the
        source finish.
        """
        if duration_us < 0:
            raise ValueError("duration_us must be non-negative")
        if not math.isfinite(max_hz) or max_hz <= 0.0:
            raise ValueError("max_hz must be positive and finite")
        tick_us = math.ceil(_ONE_SECOND_US / max_hz)
        end_us = self._now_us + duration_us
        target = -(-self._now_us // tick_us) * tick_us  # first grid tick at or after now
        while target < end_us and not self.exhausted:
            events = self._due_events(target)
            samples = self._advance_to(target)
            if samples or events:
                yield Frame(target, dict(self._latest), events, samples)
            target += tick_us
        self._now_us = max(self._now_us, end_us)

    def _advance_to(self, limit_us: int) -> tuple[Frame, ...]:
        """Fold every stored frame at or before ``limit_us`` into the snapshot.

        Returns those stored frames as the tick's batch, in stream order, which is
        what lets the feed publish full-rate evidence without re-reading anything:
        the batch is collected from the same frames that fill the snapshot.
        """
        batch: list[Frame] = []
        while self._index < len(self._frames) and self._times[self._index] <= limit_us:
            frame = self._frames[self._index]
            self._latest.update(frame.values)
            batch.append(Frame(self._times[self._index], dict(frame.values)))
            self._index += 1
        self._now_us = limit_us
        return tuple(batch)

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
