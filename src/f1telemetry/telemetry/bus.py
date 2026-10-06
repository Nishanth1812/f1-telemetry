"""Deterministic periodic CAN-FD release scheduling and bus-load estimates."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Final

from f1telemetry.telemetry.canfd import ID_MAX

_HEADER_BYTES: Final = 18
_CRC_BYTES: Final = 4
_BUS_OVERHEAD_BITS: Final = 67  # arbitration/control/ACK/EOF/intermission estimate


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
        if self.payload_bytes < 0 or self.payload_bytes > 64 or self.payload_bytes % 4:
            raise ValueError("payload_bytes must be a multiple of 4 in 0..64")

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
