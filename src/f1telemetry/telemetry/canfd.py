"""Deterministic virtual CAN-FD frame codec.

The project packet layout stores a 29-bit extended identifier in two bytes, which
cannot represent the declared range. This codec uses a four-byte little-endian ID;
the remaining fields follow it. CRC-32/ISO-HDLC covers header and payload.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from enum import IntFlag
from typing import Final

ID_MAX: Final = (1 << 29) - 1
COUNTER_MAX: Final = (1 << 32) - 1
_DLC_BYTES: Final = (0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64)
_HEADER: Final = struct.Struct("<IBBIQ")  # id, DLC, flags, counter, simulated µs
_CRC: Final = struct.Struct("<I")


class FrameError(ValueError):
    """Malformed CAN-FD frame or field."""


class Flags(IntFlag):
    ERROR = 1
    EXTENDED = 2
    RTR = 4


@dataclass(frozen=True, slots=True)
class Frame:
    identifier: int
    dlc: int
    flags: Flags
    counter: int
    timestamp_us: int
    payload: bytes


def _dlc_for_length(size: int) -> int:
    try:
        return _DLC_BYTES.index(size)
    except ValueError as exc:
        raise FrameError(f"payload length {size} is not a CAN-FD DLC length") from exc


def encode_frame(
    identifier: int,
    counter: int,
    timestamp_us: int,
    payload: object,
    *,
    flags: Flags | int = Flags.EXTENDED,
) -> bytes:
    """Encode opaque CAN-FD payload bytes, validating all declared fields."""
    if (
        isinstance(identifier, bool)
        or not isinstance(identifier, int)
        or not 0 <= identifier <= ID_MAX
    ):
        raise FrameError("identifier must be a 29-bit unsigned integer")
    if isinstance(counter, bool) or not isinstance(counter, int) or not 0 <= counter <= COUNTER_MAX:
        raise FrameError("counter must be an unsigned 32-bit integer")
    if (
        isinstance(timestamp_us, bool)
        or not isinstance(timestamp_us, int)
        or not 0 <= timestamp_us < 1 << 64
    ):
        raise FrameError("timestamp_us must be an unsigned 64-bit integer")
    if not isinstance(payload, bytes):
        raise FrameError("payload must be bytes")
    if len(payload) % 4:
        raise FrameError("columnar CAN-FD payload length must be a multiple of float32 size")
    dlc = _dlc_for_length(len(payload))
    if isinstance(flags, bool) or not isinstance(flags, (Flags, int)) or int(flags) & ~7:
        raise FrameError("flags contain unsupported bits")
    body = _HEADER.pack(identifier, dlc, int(flags), counter, timestamp_us) + payload
    return body + _CRC.pack(zlib.crc32(body) & 0xFFFFFFFF)


def decode_frame(data: bytes) -> Frame:
    """Decode a complete frame and reject invalid length, DLC, flags, or CRC."""
    if not isinstance(data, bytes) or len(data) < _HEADER.size + _CRC.size:
        raise FrameError("frame is shorter than the minimum header and CRC")
    identifier, dlc, raw_flags, counter, timestamp_us = _HEADER.unpack_from(data)
    if identifier > ID_MAX:
        raise FrameError("identifier has bits outside the 29-bit range")
    if dlc >= len(_DLC_BYTES):
        raise FrameError(f"DLC {dlc} is invalid")
    payload_end = _HEADER.size + _DLC_BYTES[dlc]
    if len(data) != payload_end + _CRC.size:
        raise FrameError("frame length does not match its DLC")
    if _DLC_BYTES[dlc] % 4:
        raise FrameError("columnar CAN-FD payload length must be a multiple of float32 size")
    if raw_flags & ~7:
        raise FrameError("flags contain reserved bits")
    expected = zlib.crc32(data[:payload_end]) & 0xFFFFFFFF
    (actual,) = _CRC.unpack_from(data, payload_end)
    if actual != expected:
        raise FrameError("CRC-32/ISO-HDLC mismatch")
    return Frame(
        identifier, dlc, Flags(raw_flags), counter, timestamp_us, data[_HEADER.size : payload_end]
    )


def next_counter(counter: int) -> int:
    """Return the next uint32 rolling counter."""
    if isinstance(counter, bool) or not isinstance(counter, int) or not 0 <= counter <= COUNTER_MAX:
        raise FrameError("counter must be an unsigned 32-bit integer")
    return (counter + 1) & COUNTER_MAX
