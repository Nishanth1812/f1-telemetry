"""Capacity, overwrite, ordering, and time-window behavior for the live telemetry buffer."""

from __future__ import annotations

from typing import cast

import pytest

from f1telemetry.telemetry.live_buffer import LiveBuffer
from f1telemetry.testing.records import SensorFrame


def _frame(t_s: float) -> SensorFrame:
    return SensorFrame(t_s=t_s, values={"speed": t_s})


def test_buffer_retains_exact_capacity_and_counts_overwrites() -> None:
    buffer = LiveBuffer(capacity=3)
    buffer.extend(_frame(t_s) for t_s in range(5))

    assert len(buffer) == 3
    assert tuple(frame.t_s for frame in buffer.snapshot()) == (2.0, 3.0, 4.0)
    assert buffer.overwritten == 2


def test_snapshot_uses_an_inclusive_time_window_and_is_immutable() -> None:
    buffer = LiveBuffer(capacity=4)
    buffer.extend(_frame(t_s) for t_s in range(4))

    result = buffer.snapshot(start_s=1.0, end_s=2.0)

    assert tuple(frame.t_s for frame in result) == (1.0, 2.0)
    assert isinstance(result, tuple)


@pytest.mark.parametrize("timestamp", [1.0, 0.5])
def test_append_rejects_duplicate_or_decreasing_timestamps(timestamp: float) -> None:
    buffer = LiveBuffer(capacity=2)
    buffer.append(_frame(1.0))

    with pytest.raises(ValueError, match="timestamps must increase strictly"):
        buffer.append(_frame(timestamp))


def test_snapshot_rejects_a_reversed_time_window() -> None:
    buffer = LiveBuffer(capacity=2)

    with pytest.raises(ValueError, match="start_s must be <= end_s"):
        buffer.snapshot(start_s=2.0, end_s=1.0)


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_capacity_must_be_a_positive_integer(capacity: int | float) -> None:
    with pytest.raises(ValueError, match="capacity must be an integer >= 1"):
        LiveBuffer(capacity=cast(int, capacity))
