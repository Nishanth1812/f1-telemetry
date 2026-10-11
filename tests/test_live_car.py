"""The live feed shows the simulated car: at rest, then driving a queued run, then holding it."""

from __future__ import annotations

from f1telemetry.live_car import DrivenCarSource
from f1telemetry.synthetic import Frame

_TICK = 33_334


def _run(count: int) -> tuple[Frame, ...]:
    return tuple(Frame(index * 10_000, {"speed": float(index)}) for index in range(count))


def _deliver(source: DrivenCarSource, ticks: int) -> list[Frame]:
    return list(source.iter_frames(ticks * _TICK, max_hz=30.0))


def test_idle_car_repeats_resting_values() -> None:
    source = DrivenCarSource({"speed": 0.0, "gear": 1.0})
    frames = _deliver(source, 3)
    assert [frame.channels for frame in frames] == [{"speed": 0.0, "gear": 1.0}] * 3
    assert not source.running


def test_run_plays_back_in_order_then_holds_the_final_values() -> None:
    source = DrivenCarSource({"speed": 0.0, "gear": 1.0})
    _deliver(source, 2)
    source.start_run(_run(30))
    assert source.running
    frames = _deliver(source, 20)
    speeds = [frame.channels["speed"] for frame in frames]
    assert speeds == sorted(speeds)
    assert speeds[-1] == 29.0
    assert not source.running
    assert frames[-1].channels["gear"] == 1.0
    played = [sample.channels["speed"] for frame in frames for sample in frame.samples]
    assert 0.0 in played and 29.0 in played


def test_time_only_moves_forward_across_runs() -> None:
    source = DrivenCarSource({"speed": 0.0})
    times = [frame.time_us for frame in _deliver(source, 4)]
    source.start_run(_run(10))
    times += [frame.time_us for frame in _deliver(source, 6)]
    source.start_run(_run(10))
    times += [frame.time_us for frame in _deliver(source, 6)]
    assert times == sorted(set(times))
