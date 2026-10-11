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
_BUS_OVERHEAD_BITS: Final = 67
"""Fixed per-frame wire cost *outside* the byte-oriented body, in bits.

ISO 11898-1 spells a frame out as start of frame, arbitration field, control
field, data field, CRC sequence, CRC delimiter, acknowledge slot and delimiter,
end of frame and intermission. This model accounts for the arbitration-to-CRC
part as the 18 header bytes the codec puts on the wire plus the payload plus the
4 CRC bytes, so what is left to count by hand is the frame delimiters, the
acknowledge and the intermission: 67 bits, once per frame, whatever it carries.

**Bit stuffing is deliberately not in this number.** ISO 11898-1 stuffs one bit of
opposite polarity after every run of five equal bits, so the worst case for a body
of ``n`` bits is ``n // 5`` extra bits - 20 % of the body. Telemetry payloads never
get near that, but a load *gate* should not quietly leave it out, so the body is
exposed as :attr:`MessageSpec.stuffed_bits` and the worst case is exactly
``stuffed_bits // 5``. Real stuffing is content-dependent, and only a bus analyser
can measure it; a deterministic bound is what this model can honestly claim.
"""

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
    def slots(self) -> int:
        """Whole float32 values this message carries."""
        return self.payload_bytes // 4

    @property
    def stuffed_bits(self) -> int:
        """Body bits this frame puts on the wire - the region bit stuffing can inflate.

        The worst-case stuffing allowance is ``stuffed_bits // 5`` (see
        :data:`_BUS_OVERHEAD_BITS`); real payloads stuff far less, and the bound is
        the figure a load estimate can rely on.
        """
        return 8 * (_HEADER_BYTES + self.payload_bytes + _CRC_BYTES)

    @property
    def wire_bits(self) -> int:
        """Serialized cost in bits, *without* bit stuffing - see :data:`_BUS_OVERHEAD_BITS`."""
        return self.stuffed_bits + _BUS_OVERHEAD_BITS


@dataclass(frozen=True, slots=True)
class ScheduledMessage:
    name: str
    identifier: int
    release_us: int
    start_us: int
    finish_us: int


def pack_slot_counts(count: int) -> tuple[int, ...]:
    """Split a periodic channel count into legal columnar DLC frames, largest first.

    One channel is one float32 slot; narrow dtypes are padded because the codec
    only carries whole slots. Exported so the encoder that realises a schedule on
    the wire packs payloads by this rule and no other, which is what keeps the
    emitted DLCs inside what :func:`schedule` budgeted.
    """
    parts: list[int] = []
    remaining = count
    while remaining > 0:
        take = max(size for size in _SLOT_FLOATS if size <= remaining)
        parts.append(take)
        remaining -= take
    return tuple(parts)


def _rate_groups(contract: ChannelContract) -> tuple[tuple[float, tuple[str, ...]], ...]:
    """Periodic channels grouped by declared rate: fastest first, contract order within a rate.

    Event channels (``rate_hz is None``) are never scheduled, so they are in no group.
    This is the one pass that decides both the message identifiers and which channels
    each message carries, so the schedule and its channel layout cannot drift apart.
    """
    names_by_rate: dict[float, list[str]] = {}
    for channel in contract.channels:
        rate = channel.rate_hz
        if rate is None:  # event channels are never scheduled on a timer
            continue
        rate_hz = float(rate)
        if not math.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError(f"channel {channel.name!r} has invalid rate_hz {rate!r}")
        names_by_rate.setdefault(rate_hz, []).append(channel.name)
    return tuple(
        (rate_hz, tuple(names_by_rate[rate_hz])) for rate_hz in sorted(names_by_rate, reverse=True)
    )


def message_specs_from_contract(
    contract: ChannelContract,
    *,
    base_identifier: int = DEFAULT_BASE_IDENTIFIER,
) -> tuple[MessageSpec, ...]:
    """Derive deterministic periodic CAN schedule specs from the channel contract.

    Channels sharing a rate are packed greedily into legal DLC payloads by
    :func:`pack_slot_counts`, fastest rate first so faster traffic holds lower
    identifiers (higher arbitration priority). Periods come from the declared
    rates; no wall clock is read. The channel names behind each message are the
    same list :func:`channel_groups_from_contract` returns, index for index.
    """
    specs: list[MessageSpec] = []
    for rank, (rate_hz, names) in enumerate(_rate_groups(contract)):
        period_us = round(_ONE_SECOND_US / rate_hz)
        if period_us < 1:
            raise ValueError(f"rate {rate_hz!r} Hz needs a sub-microsecond period")
        label = str(int(rate_hz)) if rate_hz.is_integer() else str(rate_hz)
        identifier_base = base_identifier * (rank + 1)
        for frame_index, frame_slots in enumerate(pack_slot_counts(len(names))):
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


def channel_groups_from_contract(contract: ChannelContract) -> tuple[tuple[str, ...], ...]:
    """Channel names per scheduled message, index-aligned with the specs.

    Entry *i* holds the channels carried by the *i*-th message of
    :func:`message_specs_from_contract`, packed by the same :func:`pack_slot_counts`
    rule over the same :func:`_rate_groups` pass. A caller that holds both can bind
    a scheduled message to its values without restating how either is derived.
    """
    groups: list[tuple[str, ...]] = []
    for _rate_hz, names in _rate_groups(contract):
        offset = 0
        for take in pack_slot_counts(len(names)):
            groups.append(names[offset : offset + take])
            offset += take
    return tuple(groups)


def bus_utilisation(specs: tuple[MessageSpec, ...], bitrate_bps: int) -> float:
    """Return estimated serialized bus occupancy over the repeating release periods.

    Occupancy is measured in :attr:`MessageSpec.wire_bits`, so it excludes bit
    stuffing; add ``stuffed_bits // 5`` per message for the worst case ISO 11898-1
    allows (see :data:`_BUS_OVERHEAD_BITS`). The figure is a share of ``bitrate_bps``,
    never a wall-clock rate.
    """
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
