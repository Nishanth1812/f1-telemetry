"""Deterministic periodic CAN-FD release scheduling and bus-load estimates."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from f1telemetry.telemetry.canfd import DLC_PAYLOAD_BYTES, ID_MAX

if TYPE_CHECKING:
    from f1telemetry.contracts.channels import ChannelContract

_HEADER_BYTES: Final = 18
_CRC_BYTES: Final = 4
_BUS_OVERHEAD_BITS: Final = 67  # arbitration/control/ACK/EOF/intermission estimate

# Virtual bus sizing: CAN-FD 2 Mbit/s data rate, the common automotive
# default. Simulation time only; this is never a wall-clock rate.
DEFAULT_BITRATE_BPS: Final = 2_000_000
DEFAULT_BASE_IDENTIFIER: Final = 0x100
_ONE_SECOND_US: Final = 1_000_000
# Legal columnar payloads are CAN-FD DLC lengths that hold whole float32
# slots, largest first, so every scheduled message is encodable on the wire.
_SLOT_FLOATS: Final[tuple[int, ...]] = tuple(
    size // 4
    for size in sorted(
        (length for length in DLC_PAYLOAD_BYTES if length % 4 == 0 and length > 0),
        reverse=True,
    )
)


@dataclass(frozen=True, slots=True)
class MessageSpec:
    name: str
    identifier: int
    period_us: int
    payload_bytes: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("message name must not be empty")
        if not 0 <= self.identifier <= ID_MAX:
            raise ValueError("identifier must be a 29-bit unsigned integer")
        if self.period_us < 1:
            raise ValueError("period_us must be >= 1")
        if self.payload_bytes not in DLC_PAYLOAD_BYTES or self.payload_bytes % 4:
            raise ValueError("payload_bytes must be a whole-float32 DLC payload length")

    @property
    def wire_bits(self) -> int:
        return 8 * (_HEADER_BYTES + self.payload_bytes + _CRC_BYTES) + _BUS_OVERHEAD_BITS


@dataclass(frozen=True, slots=True)
class ScheduledMessage:
    name: str
    identifier: int
    release_us: int
    start_us: int
    finish_us: int


def _pack_slot_counts(count: int) -> list[int]:
    """Split a periodic channel count into legal columnar DLC frames, largest first."""
    parts: list[int] = []
    remaining = count
    while remaining > 0:
        take = max(size for size in _SLOT_FLOATS if size <= remaining)
        parts.append(take)
        remaining -= take
    return parts


def message_specs_from_contract(
    contract: ChannelContract,
    *,
    base_identifier: int = DEFAULT_BASE_IDENTIFIER,
) -> tuple[MessageSpec, ...]:
    """Derive deterministic periodic CAN schedule specs from the channel contract.

    Each periodic channel occupies one float32 slot (4 bytes); narrow dtypes
    are padded because the codec only carries whole float32 slots. Channels
    sharing a rate are packed greedily into legal DLC payloads, fastest rate
    first so faster traffic holds lower identifiers (higher arbitration
    priority). Event channels (``rate_hz is None``) are never scheduled on a
    timer. Periods come from the declared rates; no wall clock is read.
    """
    slots_by_rate: dict[float, int] = {}
    for channel in contract.channels:
        rate = channel.rate_hz
        if rate is None:  # event channels are never scheduled on a timer
            continue
        rate_hz = float(rate)
        if not math.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError(f"channel {channel.name!r} has invalid rate_hz {rate!r}")
        slots_by_rate[rate_hz] = slots_by_rate.get(rate_hz, 0) + 1
    specs: list[MessageSpec] = []
    for rank, rate_hz in enumerate(sorted(slots_by_rate, reverse=True)):
        period_us = round(_ONE_SECOND_US / rate_hz)
        if period_us < 1:
            raise ValueError(f"rate {rate_hz!r} Hz needs a sub-microsecond period")
        label = str(int(rate_hz)) if rate_hz.is_integer() else str(rate_hz)
        identifier_base = base_identifier * (rank + 1)
        for frame_index, frame_slots in enumerate(_pack_slot_counts(slots_by_rate[rate_hz])):
            identifier = identifier_base + frame_index
            if identifier > ID_MAX:
                raise ValueError("contract schedule needs identifiers beyond 29 bits")
            specs.append(
                MessageSpec(
                    f"telemetry_{label}hz_{frame_index}",
                    identifier,
                    period_us,
                    frame_slots * 4,
                )
            )
    return tuple(specs)


def bus_utilisation(specs: tuple[MessageSpec, ...], bitrate_bps: int) -> float:
    """Return estimated serialized bus occupancy over the repeating release periods."""
    if not specs or bitrate_bps < 1:
        raise ValueError("specs and positive bitrate_bps are required")
    return sum(spec.wire_bits * 1_000_000 / spec.period_us for spec in specs) / bitrate_bps


def schedule(
    specs: tuple[MessageSpec, ...], duration_us: int, bitrate_bps: int
) -> tuple[ScheduledMessage, ...]:
    """Schedule periodic releases using identifier priority and simulated time only."""
    if duration_us < 0 or bitrate_bps < 1:
        raise ValueError("duration_us must be >= 0 and bitrate_bps must be positive")
    releases: list[tuple[int, int, int, MessageSpec]] = []
    for index, spec in enumerate(specs):
        heapq.heappush(releases, (0, spec.identifier, index, spec))
    pending: list[tuple[int, int, int, MessageSpec]] = []
    now = 0
    out: list[ScheduledMessage] = []
    while releases or pending:
        if not pending and releases and releases[0][0] > now:
            now = releases[0][0]
        while releases and releases[0][0] <= now:
            heapq.heappush(pending, heapq.heappop(releases))
        release_us, identifier, index, spec = heapq.heappop(pending)
        duration = math.ceil(spec.wire_bits * 1_000_000 / bitrate_bps)
        finish = now + duration
        if now >= duration_us:
            break
        out.append(ScheduledMessage(spec.name, identifier, release_us, now, finish))
        now = finish
        next_release = release_us + spec.period_us
        if next_release < duration_us:
            heapq.heappush(releases, (next_release, spec.identifier, index, spec))
    return tuple(out)
