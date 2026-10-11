"""P0-T10: the contract drives the cadence, and nothing unencodable is published.

One test, two claims, because they are the same claim seen from both sides: the wire
content is a pure function of ``channels.yaml``.

1. Editing ``rate_hz`` in a temporary contract changes the sample cadence, with no
   other edit. If any rate were hardcoded, changing 200 to 50 would not change the
   sample count from 200 to 50.
2. A :data:`~f1telemetry.synthetic.SampleFn` that returns ``nan``, ``inf`` or a
   non-number never reaches a frame. This exercises the frame guard rather than the
   contract sampler, because the contract sampler *raises* on a non-finite draw
   instead of producing one. The frame guard exists for the P5 sensor and replay path
   that will replace that sampler, so it is tested here rather than trusted.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from pathlib import Path
from typing import cast

import numpy as np

from f1telemetry.contracts.channels import Channel, load_channel_contract
from f1telemetry.synthetic import SampleFn, SyntheticSource, load_source

_ONE_SECOND_US = 1_000_000
_HALF_SECOND_US = 500_000

_CONTRACT_TEMPLATE = """\
version: 1
description: temporary contract for the P0-T10 cadence test
channels:
  - name: fast
    group: chassis
    unit: m/s^2
    rate_hz: {fast_hz}
    dtype: float32
    range: [-60.0, 60.0]
    noise_model: gaussian
    sigma: 0.5
    fault_eligible: [dropout, freeze, spike, noise]
  - name: gear
    group: powertrain
    unit: enum
    rate_hz: {fast_hz}
    dtype: int8
    range: [-1, 8]
    noise_model: none
    sigma: 0.0
    fault_eligible: [dropout, freeze, spike, noise]
  - name: slow
    group: thermal
    unit: degC
    rate_hz: {slow_hz}
    dtype: float32
    range: [-20.0, 130.0]
    noise_model: none
    sigma: 0.0
    fault_eligible: [dropout, freeze, spike, noise]
  - name: lap_index
    group: session
    unit: count
    event: true
    rate_hz: null
    dtype: int16
    range: [-1, 5000]
    noise_model: none
    sigma: 0.0
    fault_eligible: [dropout, stale]
"""

_EVENT_ONLY_CONTRACT = """\
version: 1
description: contract with no periodic channel at all
channels:
  - name: lap_index
    group: session
    unit: count
    event: true
    rate_hz: null
    dtype: int16
    range: [-1, 5000]
    noise_model: none
    sigma: 0.0
    fault_eligible: [dropout, stale]
"""


def _write(path: Path, text: str) -> Path:
    _ = path.write_text(text, encoding="utf-8")
    return path


def _cadence(source: SyntheticSource, duration_us: int = _ONE_SECOND_US) -> Counter[str]:
    """How many frames each channel appeared in over a window of simulated time."""
    counts: Counter[str] = Counter()
    for frame in source.iter_samples(duration_us):
        counts.update(frame.channels.keys())
    return counts


def _counting(counts: Counter[str]) -> SampleFn:
    """A well-behaved :data:`SampleFn` that records how often each channel is drawn."""

    def sample(channel: Channel, rng: np.random.Generator) -> float:
        counts[channel.name] += 1
        return float(rng.uniform(channel.range_min, channel.range_max))

    return sample


def _poisoning() -> SampleFn:
    """A :data:`SampleFn` that returns something unpublishable for one channel."""
    # 1e400 is not a separate case: Python folds it to inf, already covered.
    unusable: list[object] = [math.nan, math.inf, -math.inf, "12.5", None]

    def sample(channel: Channel, rng: np.random.Generator) -> float:
        if channel.name == "fast":
            # Deliberate type lie: the annotation says float, the point is that the
            # guard has to survive callers who break it.
            return cast("float", unusable[int(rng.integers(0, len(unusable)))])
        return 1.0

    return sample


def test_rate_change_in_channels_yaml_drives_cadence_and_frames_stay_finite(
    tmp_path: Path,
) -> None:
    """A rate edit changes the cadence; malformed values never leave the source."""
    baseline = _write(
        tmp_path / "baseline.yaml", _CONTRACT_TEMPLATE.format(fast_hz=200, slow_hz=10)
    )
    edited = _write(tmp_path / "edited.yaml", _CONTRACT_TEMPLATE.format(fast_hz=50, slow_hz=20))
    events_only = _write(tmp_path / "events.yaml", _EVENT_ONLY_CONTRACT)

    # 1a. Declared rate == realised rate, sample for sample, over one simulated second.
    base_counts = _cadence(load_source(baseline))
    assert base_counts["fast"] == 200
    assert base_counts["gear"] == 200
    assert base_counts["slow"] == 10

    # 1b. One rate changed in channels.yaml, no other edit: fast and gear drop to
    #     50/s and slow doubles to 20/s.
    edited_counts = _cadence(load_source(edited))
    assert edited_counts["fast"] == 50
    assert edited_counts["gear"] == 50
    assert edited_counts["slow"] == 20

    # 1c. An event channel is on no timer, whatever the periodic rates are.
    assert set(load_source(edited).event_names) == {"lap_index"}
    assert all(
        "lap_index" not in frame.channels
        for frame in load_source(edited).iter_samples(_ONE_SECOND_US)
    )
    #     A contract with nothing but event channels schedules nothing at all, and
    #     must not trip over the empty schedule.
    assert list(load_source(events_only).iter_samples(_ONE_SECOND_US)) == []
    assert all(
        not frame.channels for frame in load_source(events_only).iter_frames(_HALF_SECOND_US)
    )

    # 1d. Every value is finite, inside its own channel's declared range, and an
    #     integer dtype stays integer - a discrete state channel never emits 3.7.
    contract = load_channel_contract(edited)
    seen_gear: set[float] = set()
    for frame in load_source(edited, seed=7).iter_samples(200_000):
        for name, value in frame.channels.items():
            channel = contract.by_name(name)
            assert math.isfinite(value)
            assert channel.range_min <= value <= channel.range_max
        if "gear" in frame.channels:
            gear = frame.value("gear")
            assert gear == round(gear)
            seen_gear.add(gear)
    assert len(seen_gear) > 1 and seen_gear <= set(range(-1, 9))

    # 1e. 30 Hz aggregation for the browser changes delivery, never sampling: every
    #     frame carries every channel, and the declared rates are still drawn.
    frames = list(load_source(baseline).iter_frames(_HALF_SECOND_US))
    assert len(frames) == 15  # ceil(1e6/30) = 33334 us ticks, half-open over 500 ms
    assert all(set(frame.channels) == {"fast", "slow", "gear"} for frame in frames)
    assert [frame.time_us for frame in frames[:2]] == [0, 33_334]
    assert all(b.time_us - a.time_us == 33_334 for a, b in itertools.pairwise(frames))

    # 15 frames were delivered, but sampling is untouched: over the span the last
    # tick reached, each channel was drawn exactly rate_hz times over it - 94 for the
    # 200 Hz pair and 5 for the 10 Hz channel.
    drawn: Counter[str] = Counter()
    aggregated = SyntheticSource(load_channel_contract(baseline), sample_fn=_counting(drawn))
    _ = list(aggregated.iter_frames(_HALF_SECOND_US))
    reached = aggregated.now_us  # last tick, since the window is half-open
    for name, rate_hz in (("fast", 200.0), ("gear", 200.0), ("slow", 10.0)):
        assert drawn[name] == int(reached * rate_hz) // _ONE_SECOND_US + 1

    # 1f. A transport pulls fixed-size chunks, so consecutive chunks must tile the
    #     grid exactly: no repeated timestamp at the seam, no skipped tick. (The tick
    #     count is not 2 x 15 - a 500 ms chunk is not a whole number of 33334 us
    #     ticks - but the spacing across the seam is.)
    chunked = SyntheticSource(load_channel_contract(baseline))
    tiled = [f.time_us for _ in range(2) for f in chunked.iter_frames(_HALF_SECOND_US)]
    assert tiled[0] == 0
    assert len(set(tiled)) == len(tiled)
    assert all(b - a == 33_334 for a, b in itertools.pairwise(tiled))

    # 2. A sample function that misbehaves cannot poison the wire, on either stream.
    poisoned = _poisoning()
    for source in (
        load_source(baseline, sample_fn=poisoned),
        load_source(edited, sample_fn=poisoned),
    ):
        published = [
            *source.iter_frames(_ONE_SECOND_US),
            *source.iter_samples(_ONE_SECOND_US),
        ]
        assert published, "expected frames even when every fast sample is unusable"
        for frame in published:
            assert "fast" not in frame.channels
            assert set(frame.channels) <= {"slow", "gear"}
            payload = frame.to_wire()
            assert isinstance(payload["time_us"], int)
            wire = cast("dict[str, float]", payload["channels"])
            for name, value in wire.items():
                assert isinstance(name, str)
                assert isinstance(value, float) and math.isfinite(value)
