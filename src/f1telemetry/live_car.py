"""A live telemetry source that shows the simulated car instead of random numbers.

Until a test is started the car sits at rest. :meth:`DrivenCarSource.start_run` queues a
finished headless run; the source then plays its 100 Hz frames back in real time through the
same wire shape the dashboard already reads, and holds the last values when the run ends so the
final state stays on screen. The clock only moves forward, so a second run follows the first
rather than rewinding it.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from typing import Final

from f1telemetry.synthetic import BROADCAST_HZ, Frame

__all__ = ["DrivenCarSource"]

_US_PER_SECOND: Final[int] = 1_000_000


class DrivenCarSource:
    """Frame source that idles until a recorded run is queued, then plays it live."""

    def __init__(self, idle: Mapping[str, float]) -> None:
        self._latest: dict[str, float] = dict(idle)
        self._now_us: int = 0
        self._emitted_us: int = -1
        self._run: tuple[Frame, ...] = ()
        self._cursor: int = 0
        self._origin_us: int = 0

    @property
    def now_us(self) -> int:
        return self._now_us

    @property
    def running(self) -> bool:
        return self._cursor < len(self._run)

    def start_run(self, frames: Sequence[Frame]) -> None:
        """Begin playing ``frames`` (times from 0) right after the last delivered frame."""
        self._run = tuple(frames)
        self._cursor = 0
        self._origin_us = self._now_us + 1

    def iter_frames(self, duration_us: int, *, max_hz: float = BROADCAST_HZ) -> Iterator[Frame]:
        if not math.isfinite(max_hz) or max_hz <= 0.0:
            raise ValueError(f"max_hz must be a positive finite number, got {max_hz!r}")
        tick_us = math.ceil(_US_PER_SECOND / max_hz)
        end = self._now_us + max(0, int(duration_us))
        target = -(-self._now_us // tick_us) * tick_us
        while target < end:
            if target != self._emitted_us:
                samples = self._advance_to(target)
                self._emitted_us = target
                self._now_us = target
                yield Frame(target, dict(self._latest), samples=samples)
            target += tick_us

    def _advance_to(self, limit_us: int) -> tuple[Frame, ...]:
        """Take every recorded frame due by ``limit_us``; idle ticks repeat the held values."""
        due: list[Frame] = []
        relative_limit = limit_us - self._origin_us
        while self._cursor < len(self._run) and self._run[self._cursor].time_us <= relative_limit:
            frame = self._run[self._cursor]
            self._latest.update(frame.channels)
            due.append(Frame(self._origin_us + frame.time_us, dict(frame.channels)))
            self._cursor += 1
        if due:
            return tuple(due)
        return (Frame(limit_us, dict(self._latest)),)
