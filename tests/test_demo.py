from itertools import pairwise
from pathlib import Path

import pytest

from f1telemetry.demo import LoopingReplaySource, generate_sample
from f1telemetry.testing.parquet_io import serialise_frames
from f1telemetry.testing.records import SampleRecord, SensorFrame


def test_demo_loops_without_rewinding_the_wire_clock(tmp_path: Path) -> None:
    path = tmp_path / "sample.parquet"
    frames = (SensorFrame(0.0, {"speed": 1.0}), SensorFrame(0.1, {"speed": 2.0}))
    path.write_bytes(serialise_frames(SampleRecord("demo", 0.1, "", (), frames)))
    source = LoopingReplaySource(path)
    deliveries = [frame for _ in range(4) for frame in source.iter_frames(1_000_000, max_hz=30)]
    assert len(deliveries) >= 8
    times = [frame.time_us for frame in deliveries]
    assert times == sorted(set(times))
    assert max(b - a for a, b in pairwise(times)) < 200_000
    assert sum(frame.channels["speed"] == 2.0 for frame in deliveries) >= 4


def test_sample_generation_never_replaces_a_recording(tmp_path: Path) -> None:
    path = tmp_path / "keep.parquet"
    path.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        generate_sample(path)
    assert path.read_bytes() == b"original"
