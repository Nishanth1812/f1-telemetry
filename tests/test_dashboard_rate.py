"""P0 full-rate transport: a delivery frame carries the samples it aggregated.

The 30 Hz feed is a *view* of a full-rate source, and a browser can only retain or
count what actually arrives. These tests pin the wire evidence for that claim:

* every delivery frame carries an optional ``samples`` batch, so a 200 Hz channel is
  visible 200 times a simulated second even though only 30 frames are delivered, and
  the snapshot value is the newest sample at or before that frame;
* editing ``rate_hz`` in a temporary contract changes that count with no other edit,
  which is the P0 exit gate read straight off the wire;
* the batch is ordered, and each instant is delivered exactly once across the
  fixed-size chunk boundaries a transport pulls;
* the finite-value guard applies inside the batch exactly as it applies to a frame,
  and a frame with no batch still serialises byte-identically to the legacy shape.

The batch is read off ``Frame.to_wire()`` rather than a typed attribute on purpose:
these tests are written before the field exists, so they fail on the missing
``samples`` key and not on an import or attribute error.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import cast

import numpy as np

from f1telemetry.contracts.channels import Channel
from f1telemetry.synthetic import Frame, SampleFn, SyntheticSource, load_source

_ONE_SECOND_US = 1_000_000
_HALF_SECOND_US = 500_000
_FAST_HZ = 200.0
_SLOW_HZ = 10.0
_FAST_PERIOD_US = 5_000  # 200 Hz
_SLOW_PERIOD_US = 100_000  # 10 Hz

_CONTRACT_TEMPLATE = """\
version: 1
description: temporary contract for the P0 full-rate transport test
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

_Batch = list[tuple[int, dict[str, float]]]
"""``(time_us, channels)`` pairs: the wire shape of one sample entry."""


def _write(path: Path, *, fast_hz: float, slow_hz: float) -> Path:
    _ = path.write_text(
        _CONTRACT_TEMPLATE.format(fast_hz=fast_hz, slow_hz=slow_hz), encoding="utf-8"
    )
    return path


def _samples(frame: Frame, *, required: bool = True) -> _Batch:
    """The frame's full-rate batch, read off the wire payload.

    ``samples`` is optional on the wire, so a frame with nothing new to report omits
    the key. ``required`` is for the frames that must carry a batch; the poisoned case
    checks the omission instead.
    """
    payload = frame.to_wire()
    if not required and "samples" not in payload:
        return []
    assert "samples" in payload, (
        f"frame at t={frame.time_us}us carries no 'samples' key: the feed is still a "
        "newest-value snapshot, so a browser cannot retain or count the samples in between"
    )
    batch = cast("list[dict[str, object]]", payload["samples"])
    pairs: _Batch = []
    for entry in batch:
        assert set(entry) == {"time_us", "channels"}, "sample entries must stay flat"
        time_us = entry.get("time_us")
        assert isinstance(time_us, int), f"sample time_us must be an int, got {time_us!r}"
        pairs.append((time_us, cast("dict[str, float]", entry["channels"])))
    return pairs


def _chunk(
    source: SyntheticSource, duration_us: int, *, required: bool = True
) -> list[tuple[Frame, _Batch]]:
    """One fixed-size pull from the source, as a transport makes it."""
    return [
        (frame, _samples(frame, required=required)) for frame in source.iter_frames(duration_us)
    ]


def _delivered(
    source: SyntheticSource, duration_us: int, *, required: bool = True
) -> list[tuple[Frame, _Batch]]:
    """Frames for one chunk, plus the next delivery frame after the chunk ends.

    The samples in ``[0, duration_us)`` do not all sit in frames whose timestamp is
    below ``duration_us``: the tail of the window is aggregated into the *next*
    delivery frame. Consuming one frame past the chunk boundary is what makes a whole
    simulated window countable.
    """
    frames = _chunk(source, duration_us, required=required)
    closing = next(iter(source.iter_frames(duration_us)), None)
    if closing is None:
        return frames
    return [*frames, (closing, _samples(closing, required=required))]


def _window_pairs(
    delivered: list[tuple[Frame, _Batch]], duration_us: int
) -> list[tuple[int, dict[str, float]]]:
    """Every delivered sample instant in ``[0, duration_us)``, in delivery order."""
    flat = [pair for _frame, batch in delivered for pair in batch]
    return [pair for pair in flat if pair[0] < duration_us]


def _poisoning() -> SampleFn:
    """A :data:`SampleFn` that returns something unpublishable for the fast channel."""
    unusable: list[object] = [math.nan, math.inf, -math.inf, "12.5", None]

    def sample(channel: Channel, rng: np.random.Generator) -> float:
        if channel.name == "fast":
            # Deliberate type lie: the annotation says float, the point is that the
            # guard inside the batch has to survive callers who break it.
            return cast("float", unusable[int(rng.integers(0, len(unusable)))])
        return 1.0

    return sample


def test_delivery_frames_carry_the_full_rate_samples_they_aggregated(tmp_path: Path) -> None:
    """A 200 Hz channel is countable 200 times a second behind a 30 Hz feed."""
    contract = _write(tmp_path / "fast.yaml", fast_hz=_FAST_HZ, slow_hz=_SLOW_HZ)
    delivered = _delivered(load_source(contract, seed=3), _ONE_SECOND_US)

    # 30 Hz delivery is unchanged: 30 frames over one simulated second, plus the one
    # frame that closes the window.
    assert [frame.time_us for frame, _batch in delivered[:2]] == [0, 33_334]
    assert len(delivered) == 31
    assert all(batch for _frame, batch in delivered), "every delivered frame needs a batch"

    pairs = _window_pairs(delivered, _ONE_SECOND_US)
    instants = [time_us for time_us, _channels in pairs]
    assert instants == sorted(instants)
    assert len(set(instants)) == len(instants), "an instant must not be delivered twice"
    assert instants == sorted({index * _FAST_PERIOD_US for index in range(200)})

    # Each entry carries only the channels due at that instant, so the 10 Hz channel
    # is not given 200 fake samples and the fast channel is not dropped.
    for time_us, channels in pairs:
        assert set(channels) == ({"fast", "slow"} if time_us % _SLOW_PERIOD_US == 0 else {"fast"})
    assert sum("fast" in channels for _time_us, channels in pairs) == 200
    assert sum("slow" in channels for _time_us, channels in pairs) == 10

    # Samples sit at or before the snapshot that encloses them, and the snapshot is
    # the newest value seen *so far*: a channel slower than delivery may not be
    # resampled in a given batch, so its latest value is carried forward, not
    # re-sampled. Tracked cumulatively across the delivered batches.
    latest: dict[str, float] = {}
    for frame, batch in delivered:
        for time_us, channels in batch:
            assert time_us <= frame.time_us
            latest.update(channels)
        assert frame.channels == latest, (
            f"snapshot at t={frame.time_us}us is not the newest sampled value per channel"
        )


def test_a_rate_edit_in_the_contract_changes_the_delivered_sample_rate(tmp_path: Path) -> None:
    """One ``rate_hz`` edit, no other edit: the observable sample rate follows it."""
    baseline = _write(tmp_path / "baseline.yaml", fast_hz=_FAST_HZ, slow_hz=_SLOW_HZ)
    edited = _write(tmp_path / "edited.yaml", fast_hz=50, slow_hz=20)

    for path, fast_count, slow_count in ((baseline, 200, 10), (edited, 50, 20)):
        delivered = _delivered(load_source(path), _ONE_SECOND_US)
        pairs = _window_pairs(delivered, _ONE_SECOND_US)
        assert sum("fast" in channels for _time_us, channels in pairs) == fast_count
        assert sum("slow" in channels for _time_us, channels in pairs) == slow_count
        assert len(delivered) == 31, "delivery rate is 30 Hz whatever the sampling rates are"


def test_sample_instants_are_ordered_and_delivered_once_across_chunk_seams(tmp_path: Path) -> None:
    """A transport pulling fixed-size chunks sees one gapless, ordered sample stream."""
    contract = _write(tmp_path / "fast.yaml", fast_hz=_FAST_HZ, slow_hz=_SLOW_HZ)
    source = load_source(contract, seed=5)

    # Two consecutive chunks, exactly as the server pulls them. The closing frame of
    # one chunk is the opening frame of the next, so it is left out here to avoid
    # counting one delivery twice.
    chunks = [_chunk(source, _HALF_SECOND_US) for _ in range(2)]
    frame_times = [frame.time_us for chunk in chunks for frame, _batch in chunk]
    assert frame_times[0] == 0
    assert len(set(frame_times)) == len(frame_times), "a delivery timestamp was repeated"
    assert all(b - a == 33_334 for a, b in itertools.pairwise(frame_times))

    # The sample instants stay one gapless ordered series across the same seam: every
    # 200 Hz instant delivered once, none repeated, none skipped at the boundary.
    instants = [time_us for chunk in chunks for _frame, batch in chunk for time_us, _ch in batch]
    assert instants[0] == 0
    assert len(set(instants)) == len(instants), "a sample instant was delivered twice"
    assert all(b - a == _FAST_PERIOD_US for a, b in itertools.pairwise(instants))


def test_unusable_sample_values_never_enter_a_batch_and_legacy_frames_are_unchanged(
    tmp_path: Path,
) -> None:
    """The finite-value guard applies inside the batch, and an empty batch is omitted."""
    contract = _write(tmp_path / "fast.yaml", fast_hz=_FAST_HZ, slow_hz=_SLOW_HZ)
    delivered = _delivered(
        load_source(contract, sample_fn=_poisoning()), _ONE_SECOND_US, required=False
    )

    assert delivered, "expected frames even when every fast sample is unusable"
    # Only the 10 Hz channel survives the guard, so most 30 Hz frames have no new
    # periodic sample to report and must omit the optional key rather than ship an
    # empty batch. The key is present exactly when the batch has an entry.
    present = [(frame, "samples" in frame.to_wire()) for frame, _batch in delivered]
    assert present[0][1], "the t=0 frame samples every periodic channel"
    assert any(not flag for _frame, flag in present), "expected frames with nothing new to report"
    for frame, batch in delivered:
        assert ("samples" in frame.to_wire()) == bool(batch)

    # The surviving instants are exactly the 10 Hz ones, once each and in order.
    instants = [time_us for _frame, batch in delivered for time_us, _channels in batch]
    assert instants == sorted(instants)
    assert len(set(instants)) == len(instants)
    assert instants == [index * _SLOW_PERIOD_US for index in range(11)]

    for frame, batch in delivered:
        assert "fast" not in frame.channels
        assert set(frame.channels) <= {"slow"}
        for _time_us, channels in batch:
            assert set(channels) <= {"slow"}, f"unusable value leaked into the batch: {channels}"
            for name, value in channels.items():
                assert isinstance(name, str)
                assert isinstance(value, float) and math.isfinite(value)

    # A frame constructed without a batch is byte-identical to the legacy payload.
    legacy = Frame(time_us=1_234, channels={"fast": 1.5, "slow": 2.5})
    assert legacy.to_wire() == {"time_us": 1_234, "channels": {"fast": 1.5, "slow": 2.5}}
