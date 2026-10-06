"""Finite source for replaying saved sensor frames through the live transport."""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

from f1telemetry.synthetic import Frame
from f1telemetry.testing.parquet_io import read_frames
from f1telemetry.testing.records import SensorFrame


class ReplaySource:
    """Read a Parquet run once and expose its values as dashboard frames."""

    def __init__(self, path: Path) -> None:
        self._frames: tuple[SensorFrame, ...] = read_frames(path)
        self._index: int = 0
        self._now_us: int = int(self._frames[0].t_s * 1_000_000) if self._frames else 0
        self._last_emit_us: int | None = None

    @property
    def now_us(self) -> int:
        return self._now_us

    @property
    def exhausted(self) -> bool:
        return self._index >= len(self._frames)

    def iter_frames(self, duration_us: int, *, max_hz: float) -> Iterator[Frame]:
        """Yield stored samples in a half-open time window, limited to ``max_hz``."""
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
            yield Frame(time_us, dict(frame.values))
        self._now_us = max(self._now_us, end_us)
