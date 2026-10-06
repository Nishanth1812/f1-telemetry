"""Synthetic telemetry source - PHASES.md P0-T10, zero physics.

The point of this module is to *prove the rate contract*, not to simulate a car. It
reads :mod:`f1telemetry.contracts.channels` and does two things and nothing else:

1. **Schedules.** Every periodic channel is sampled on its own clock, derived solely
   from that channel's ``rate_hz`` in ``channels.yaml``. No rate is written down here,
   so editing ``rate_hz`` is the only way to change a cadence (PLAN.md section 5.2,
   PHASES.md exit gate: "changing a rate in ``channels.yaml`` changes it in the
   dashboard with no other edit"). Event channels carry ``event: true`` and
   ``rate_hz: null``; per the contract they are *never* placed on a timer, so they
   appear in :attr:`SyntheticSource.event_names` and nowhere else.
2. **Values.** Each sample is a finite scalar inside the channel's declared range,
   drawn from a seeded :class:`numpy.random.Generator` using the channel's declared
   ``noise_model`` and ``sigma``.

Two iterators, deliberately:

* :meth:`SyntheticSource.iter_samples` is the ground truth. It yields one frame per
  *scheduled instant* containing only the channels due at that instant, so the
  declared rate of every channel is directly observable in the stream. This is what
  the cadence test measures and what a P5 decimator would consume.
* :meth:`SyntheticSource.iter_frames` is the P0 browser feed. It samples internally at
  the declared rates exactly as above and then *aggregates* those samples into frames
  on a ``max_hz`` grid (30 Hz by default, matching PLAN.md section 8.4 and
  ``web/src/telemetry/types.ts``), each frame carrying the most recent value of every
  channel. Aggregation only changes delivery, never sampling, so a 200 Hz channel is
  still drawn from the RNG 200 times a second and simply appears 30 times.

Two invariants the rest of the project depends on:

* **Simulated time only.** Every timestamp is integer microseconds counted from
  ``t = 0`` by the generator itself. Nothing here reads a wall clock, so a run is
  reproducible byte-for-byte from ``(seed, contract)`` (PLAN.md section 8.6). The
  transport in :mod:`f1telemetry.server` paces *delivery* against a monotonic clock
  and never touches the values or the timestamps.
* **Only finite values reach a frame.** The contract sampler raises rather than
  emitting a non-finite value, and every frame is filtered again on the way out so a
  foreign :data:`SampleFn` - the seam where P5's virtual sensors and P5-T9's replay
  will plug in - cannot put ``NaN`` on the wire.

State is a single iterator at a time; :meth:`SyntheticSource.reset` rewinds simulated
time and reseeds.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real
from pathlib import Path
from typing import Final

import numpy as np

from f1telemetry.contracts.channels import (
    Channel,
    ChannelContract,
    load_channel_contract,
)

__all__ = [
    "BROADCAST_HZ",
    "Frame",
    "FrameEvent",
    "SampleFn",
    "SyntheticError",
    "SyntheticSource",
    "contract_sample",
    "load_source",
]

BROADCAST_HZ: Final[float] = 30.0
"""Default aggregate delivery rate for :meth:`SyntheticSource.iter_frames`.

PLAN.md section 8.4 gives the browser a 30 Hz ``live`` stream; the dashboard's tile
refresh constant is the same number. 30 Hz is an upper bound on *delivery*, never on
sampling.
"""

_US_PER_SECOND: Final[int] = 1_000_000
_SQRT3: Final[float] = math.sqrt(3.0)


class SyntheticError(RuntimeError):
    """Raised when the contract cannot be turned into a usable schedule or sample."""


SampleFn = Callable[[Channel, np.random.Generator], float]
"""Signature of the per-sample value hook.

``(channel, rng) -> float``. The default is :func:`contract_sample`; P5 replaces it
with the virtual-sensor pipeline, which owns decimation, quantisation and fault
injection. The hook is handed the channel so it can read ``range``, ``noise_model``,
``sigma``, ``dtype`` and ``quantise`` off the contract rather than hardcoding them.
"""


@dataclass(frozen=True, slots=True)
class FrameEvent:
    """One occurrence attached to a frame, serialised under the optional ``events`` key.

    The ``channels`` mapping carries *values*; events carry *occurrences*. Today the
    only producer is the replay path, which surfaces a saved run's injected-fault
    annotations as ``kind == "fault"`` events; the session event channels
    (``lap_index`` and friends) stay ordinary channels on the wire and need no kind
    of their own. Every field after ``time_us`` is optional and is omitted from the
    payload when ``None``, so an event costs exactly the keys it needs and a frame
    with no events serialises exactly as it always has.
    """

    kind: str
    time_us: int
    fault_type: str | None = None
    channel: str | None = None
    severity: float | None = None
    duration_samples: int | None = None
    label: bool | None = None

    def to_wire(self) -> dict[str, object]:
        """JSON-ready event; ``None`` fields are omitted, ``False`` and ``0`` kept."""
        payload: dict[str, object] = {"kind": self.kind, "time_us": int(self.time_us)}
        if self.fault_type is not None:
            payload["fault_type"] = self.fault_type
        if self.channel is not None:
            payload["channel"] = self.channel
        if self.severity is not None:
            payload["severity"] = float(self.severity)
        if self.duration_samples is not None:
            payload["duration_samples"] = int(self.duration_samples)
        if self.label is not None:
            payload["label"] = bool(self.label)
        return payload


@dataclass(frozen=True, slots=True)
class Frame:
    """One batch of samples, and exactly the WebSocket payload shape.

    ``time_us`` is an integer count of simulated microseconds from ``t = 0``;
    ``channels`` maps channel name to a finite float. Event channels ride in the same
    mapping when something publishes them, but the generator never schedules them.
    ``events`` carries discrete occurrences - saved fault annotations on the replay
    path - and is serialised only when non-empty, so a frame without events is
    byte-identical to the pre-event contract and live and replay frames keep one
    shape.
    """

    time_us: int
    channels: dict[str, float]
    events: tuple[FrameEvent, ...] = ()

    def to_wire(self) -> dict[str, object]:
        """JSON-ready payload, matching ``TelemetryFrame`` in the web client."""
        payload: dict[str, object] = {
            "time_us": int(self.time_us),
            "channels": {name: float(value) for name, value in self.channels.items()},
        }
        if self.events:
            payload["events"] = [event.to_wire() for event in self.events]
        return payload

    def value(self, name: str) -> float:
        return self.channels[name]


@dataclass(frozen=True, slots=True)
class _Slot:
    """A periodic channel and the interval between its samples."""

    channel: Channel
    period: Fraction


def _period_us(rate_hz: float) -> Fraction:
    """Exact sample interval in simulated microseconds, for any ``rate_hz``.

    Rational rather than float so that rates which do not divide 1e6 evenly (3.7 Hz
    from OpenF1, say) do not accumulate timing drift, and so that two channels at
    harmonically related rates come due at the *same* instant and batch together.
    """
    rate = Fraction(rate_hz).limit_denominator(_US_PER_SECOND)
    if rate <= 0:
        raise SyntheticError(f"rate_hz must be > 0, got {rate_hz!r}")
    return Fraction(_US_PER_SECOND) / rate


def _noise(channel: Channel, rng: np.random.Generator) -> float:
    """Draw from the channel's declared noise model, zero-mean with stddev ``sigma``."""
    if channel.noise_model == "none" or channel.sigma == 0.0:
        return 0.0
    if channel.noise_model == "gaussian":
        return float(rng.normal(0.0, channel.sigma))
    # Uniform on +-sqrt(3)*sigma, so the field named sigma stays the standard deviation.
    bound = channel.sigma * _SQRT3
    return float(rng.uniform(-bound, bound))


def _fit(channel: Channel, value: float) -> float:
    """Force a draw inside the declared range, as an int for integer dtypes.

    Reading ``channel.is_integer`` off the contract is what keeps a discrete state
    channel such as ``gear`` from emitting 3.7 without naming it anywhere.
    """
    if not math.isfinite(value):
        raise SyntheticError(f"{channel.name}: generated a non-finite value {value!r}")
    low, high = channel.range_min, channel.range_max
    if value < low:
        value = low
    elif value > high:
        value = high
    if channel.is_integer:
        rounded = float(round(value))
        value = min(max(rounded, math.floor(low)), math.ceil(high))
    return value


def contract_sample(channel: Channel, rng: np.random.Generator) -> float:
    """Default :data:`SampleFn`: a uniform draw in range, plus the declared noise.

    White noise in the declared range is the honest P0 answer. Nothing here tries to
    look like a car, because a plausible-looking but unphysical signal would defeat
    the point of the phase: the dashboard is proving that the *contract* drives the
    wire, not that the physics works.
    """
    low, high = channel.range_min, channel.range_max
    draw = low if high <= low else float(rng.uniform(low, high))
    return _fit(channel, draw + _noise(channel, rng))


def _publishable(values: Mapping[str, object]) -> dict[str, float]:
    """Keep only finite real numbers, so nothing unencodable can reach the wire.

    The second of the two guards described in the module docstring: ``contract_sample``
    raises on a non-finite draw, and this drops one anyway if a foreign ``SampleFn``
    returns ``nan``/``inf``, a NumPy scalar, or a non-number. Dropping a channel is
    safe - the web client treats a missing key as "not updated this frame" - whereas
    emitting ``NaN`` would be invalid JSON that the browser rejects as malformed.
    """
    keep: dict[str, float] = {}
    for name, value in values.items():
        # numbers.Real covers int, float and every NumPy scalar, and excludes str,
        # None, complex and Decimal-by-str. bool is a Real, so it is excluded by hand.
        if isinstance(value, Real) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number):
                keep[name] = number
    return keep


class SyntheticSource:
    """Deterministic finite samples for every periodic channel in a contract.

    Args:
        contract: the loaded contract; the only source of rates, ranges and noise.
        seed: seed for the PCG64 generator. ``(seed, contract)`` fixes the whole run.
        sample_fn: overrides :func:`contract_sample`. The injection seam for P5.
    """

    def __init__(
        self,
        contract: ChannelContract,
        *,
        seed: int = 0,
        sample_fn: SampleFn | None = None,
    ) -> None:
        self._contract: ChannelContract = contract
        self._sample_fn: SampleFn = contract_sample if sample_fn is None else sample_fn
        self._seed: int = int(seed)
        self._slots: tuple[_Slot, ...] = tuple(
            _Slot(channel, _period_us(channel.rate_hz))
            for channel in contract.channels
            if channel.rate_hz is not None
        )
        self._rng: np.random.Generator = np.random.default_rng(self._seed)
        self._next_due: list[Fraction] = [Fraction(0)] * len(self._slots)
        self._latest: dict[str, float] = {}
        self._now_us: int = 0
        self._emitted_us: int | None = None

    @property
    def contract(self) -> ChannelContract:
        return self._contract

    @property
    def now_us(self) -> int:
        """Current simulated time in integer microseconds. Never a wall clock."""
        return self._now_us

    @property
    def event_names(self) -> tuple[str, ...]:
        """Event channels. Published on occurrence by the caller; never scheduled."""
        return tuple(channel.name for channel in self._contract.channels if channel.event)

    @property
    def scheduled_names(self) -> tuple[str, ...]:
        """Names of the channels this source samples on a timer, in contract order."""
        return tuple(slot.channel.name for slot in self._slots)

    def reset(self, seed: int | None = None) -> None:
        """Rewind to ``t = 0`` and reseed, so two runs can be compared exactly."""
        if seed is not None:
            self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self._next_due = [Fraction(0)] * len(self._slots)
        self._latest = {}
        self._now_us = 0
        self._emitted_us = None

    def iter_samples(self, duration_us: int) -> Iterator[Frame]:
        """Yield one frame per scheduled instant, holding only the channels due then.

        The declared rate of every channel is directly countable in this stream: a
        200 Hz channel appears in 200 frames per simulated second. The window is
        half-open, ``[t0, t0 + duration_us)``, so a 200 Hz channel yields exactly 200
        samples per second of simulated time - one per period, with the endpoint not
        double-counted. Instants that fall on the same microsecond are batched into
        one frame, so 200 Hz and 100 Hz channels share every 10 ms.
        """
        end = self._now_us + max(0, int(duration_us))
        if not self._slots:
            return
        while True:
            due = min(self._next_due)
            if due >= end:
                return
            self._now_us = round(due)
            batch: dict[str, object] = {}
            for index, slot in enumerate(self._slots):
                if self._next_due[index] == due:
                    batch[slot.channel.name] = self._sample_fn(slot.channel, self._rng)
                    self._next_due[index] = due + slot.period
            yield Frame(self._now_us, _publishable(batch))

    def iter_frames(
        self,
        duration_us: int,
        *,
        max_hz: float = BROADCAST_HZ,
    ) -> Iterator[Frame]:
        """Yield aggregated delivery frames on a ``max_hz`` grid.

        Every channel is still sampled on its own declared schedule - including
        several times per grid tick for a channel faster than ``max_hz``, in which
        case the frame carries the newest of those samples. Only delivery is
        decimated, so this is a faithful 30 Hz view of a full-rate source and not a
        30 Hz source.

        The tick is ``ceil(1e6 / max_hz)`` microseconds, so the realised rate is at
        most ``max_hz`` - 29.999 Hz rather than a rounding-prone 30.0003 Hz. Ticks sit
        on a grid anchored at ``t = 0`` and no instant is ever emitted twice, so
        slicing a long run into consecutive chunks neither repeats a timestamp nor
        skips one - which is what a transport that pulls fixed-size chunks needs. As
        in :meth:`iter_samples` the window is half-open, so one second of simulated
        time at 30 Hz is exactly 30 frames.
        """
        rate = float(max_hz)
        if not math.isfinite(rate) or rate <= 0.0:
            raise SyntheticError(f"max_hz must be a positive finite number, got {max_hz!r}")
        tick_us = math.ceil(_US_PER_SECOND / rate)
        anchor = self._now_us
        end = anchor + max(0, int(duration_us))
        target = -(-anchor // tick_us) * tick_us  # first grid tick at or after anchor
        while target < end:
            if target != self._emitted_us:
                self._advance_to(target)
                self._emitted_us = target
                yield Frame(target, _publishable(self._latest))
            target += tick_us

    def _advance_to(self, limit_us: int) -> None:
        """Sample every channel whose next due instant has arrived, then set time."""
        for index, slot in enumerate(self._slots):
            due = self._next_due[index]
            while due <= limit_us:
                self._latest[slot.channel.name] = self._sample_fn(slot.channel, self._rng)
                due = due + slot.period
            self._next_due[index] = due
        self._now_us = limit_us


def load_source(
    path: Path | None = None,
    *,
    seed: int = 0,
    sample_fn: SampleFn | None = None,
) -> SyntheticSource:
    """Load a contract and wrap it in a :class:`SyntheticSource`.

    ``path`` defaults to the repository ``channels.yaml`` located by the contract
    loader; pass a :class:`~pathlib.Path` to drive a temporary contract instead.
    """
    contract = load_channel_contract(path)
    return SyntheticSource(contract, seed=seed, sample_fn=sample_fn)
