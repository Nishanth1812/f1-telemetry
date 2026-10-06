"""Fixed-capacity, monotonic telemetry window for live viewing and scrubbing."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Protocol


class _TimedFrame(Protocol):
    """A frame carrying a timestamp. Declared read-only so the frozen
    :class:`f1telemetry.testing.records.SensorFrame` satisfies it."""

    @property
    def t_s(self) -> float: ...

    @property
    def values(self) -> object: ...


class LiveBuffer:
    """Keep the newest frames up to a declared capacity; append overwrites the oldest."""

    def __init__(self, capacity: int = 600_000) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be an integer >= 1")
        self.capacity: int = capacity
        self._frames: deque[_TimedFrame] = deque(maxlen=capacity)
        self.overwritten: int = 0

    def append(self, frame: _TimedFrame) -> None:
        if self._frames and frame.t_s <= self._frames[-1].t_s:
            raise ValueError("frame timestamps must increase strictly")
        if len(self._frames) == self.capacity:
            self.overwritten += 1
        self._frames.append(frame)

    def extend(self, frames: Iterable[_TimedFrame]) -> None:
        for frame in frames:
            self.append(frame)

    def snapshot(
        self, start_s: float | None = None, end_s: float | None = None
    ) -> tuple[_TimedFrame, ...]:
        """Return an immutable time-window snapshot in timestamp order."""
        if start_s is not None and end_s is not None and start_s > end_s:
            raise ValueError("start_s must be <= end_s")
        return tuple(
            frame
            for frame in self._frames
            if (start_s is None or frame.t_s >= start_s) and (end_s is None or frame.t_s <= end_s)
        )

    def __len__(self) -> int:
        return len(self._frames)
