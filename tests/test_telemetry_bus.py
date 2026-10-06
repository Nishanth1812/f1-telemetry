"""P5-T5/T6: CAN-FD frame codec and periodic bus scheduling (PLAN.md §8.2).

Covers the virtual ECU contract end to end: known-answer CRC-32 vectors,
encode/decode round-trips, corruption, DLC, identifier and counter
validation, full-rate bus load under 70 %, identifier-priority arbitration,
and the proof that the 10 Hz engine-temperature frame is not starved by
the 200 Hz IMU traffic.
"""

from __future__ import annotations

import struct
from typing import Final

import pytest

from f1telemetry.contracts.channels import ChannelContract
from f1telemetry.telemetry.bus import (
    MessageSpec,
    bus_utilisation,
    message_specs_from_contract,
    schedule,
)
from f1telemetry.telemetry.canfd import (
    COUNTER_MAX,
    DLC_PAYLOAD_BYTES,
    ID_MAX,
    Flags,
    FrameError,
    decode_frame,
    encode_frame,
    next_counter,
)

# CAN-FD 2 Mbit/s data rate: the common automotive default, and the rate
# this project's virtual bus is sized for.
_BITRATE_BPS: Final = 2_000_000
_ONE_SECOND_US: Final = 1_000_000

# Legal CAN-FD DLC payload lengths expressed in float32 slots (bytes / 4).
# The columnar codec only carries whole float32 slots, so the 1, 2 and 3
# byte DLC entries can never occur on this bus.
_DLC_SLOTS: Final = tuple(n // 4 for n in DLC_PAYLOAD_BYTES if n % 4 == 0)
# Payload lengths legal on this columnar bus: CAN-FD DLC lengths that are
# also whole numbers of float32 slots.
_COLUMNAR_PAYLOAD_BYTES: Final = tuple(slots * 4 for slots in _DLC_SLOTS)

# Known-answer vectors. Expected bytes were produced by the independent
# bitwise CRC-32/ISO-HDLC below, not by zlib and not by the codec.
# Vector 1: id=0x1ABCDEF, dlc=8, flags=EXTENDED, counter=7, t_us=987654321,
# payload=float32(1.5, -2.25).
_KNOWN_FRAME: Final = bytes.fromhex("efcdab01080207000000b168de3a000000000000c03f000010c09f4afa7e")
# Vector 2: id=0, dlc=0, flags=0, counter=0xFFFFFFFF, t_us=0, no payload.
_KNOWN_EMPTY_FRAME: Final = bytes.fromhex("000000000000ffffffff0000000000000000ddf631e3")


def _crc32_iso_hdlc(data: bytes) -> int:
    """Bitwise CRC-32/ISO-HDLC, independent of zlib and of the codec."""
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0xEDB88320 if crc & 1 else 0)
    return crc ^ 0xFFFFFFFF


def _reframe(body: bytearray) -> bytes:
    """Re-stamp a forged frame body with a self-consistent CRC."""
    body[-4:] = struct.pack("<I", _crc32_iso_hdlc(bytes(body[:-4])))
    return bytes(body)


def _pack_slots(slots: int) -> list[int]:
    """Split a slot count into legal DLC frames, largest first."""
    parts: list[int] = []
    remaining = slots
    while remaining > 0:
        take = max(size for size in _DLC_SLOTS if size <= remaining)
        parts.append(take)
        remaining -= take
    return parts


def _periodic_rates(contract: ChannelContract) -> dict[int, int]:
    """Map each periodic rate in the contract to its channel (slot) count."""
    slots_by_rate: dict[int, int] = {}
    for channel in contract.channels:
        rate = channel.rate_hz
        if rate is None:  # event channels are never scheduled on a timer
            continue
        rate = int(rate)
        slots_by_rate[rate] = slots_by_rate.get(rate, 0) + 1
    return slots_by_rate


def _full_rate_layout(contract: ChannelContract) -> tuple[MessageSpec, ...]:
    """Pack every periodic contract channel into CAN-FD messages by rate.

    Columnar packing: each periodic channel occupies one float32 slot
    (4 bytes); narrow dtypes are padded because the codec only carries
    whole float32 slots. Frames fill greedily up to 64 bytes.

    Identifiers rise as rate falls, so the 10 Hz engine-temperature
    frame carries the highest identifier - the lowest arbitration
    priority. That is the worst case for starvation, which is the point.
    """
    specs: list[MessageSpec] = []
    for rank, (rate, slots) in enumerate(sorted(_periodic_rates(contract).items(), reverse=True)):
        identifier = 0x100 * (rank + 1)
        for frame_index, frame_slots in enumerate(_pack_slots(slots)):
            specs.append(
                MessageSpec(
                    f"telemetry_{rate}hz_{frame_index}",
                    identifier + frame_index,
                    round(_ONE_SECOND_US / rate),
                    frame_slots * 4,
                )
            )
    return tuple(specs)


# ---------------------------------------------------------------------------
# Known-answer CRC
# ---------------------------------------------------------------------------


def test_crc32_reference_matches_catalog_check_value() -> None:
    assert _crc32_iso_hdlc(b"123456789") == 0xCBF43926


def test_encode_frame_is_known_answer_vector() -> None:
    payload = struct.pack("<2f", 1.5, -2.25)
    assert encode_frame(0x1ABCDEF, 7, 987654321, payload) == _KNOWN_FRAME


def test_encode_empty_payload_is_known_answer_vector() -> None:
    assert encode_frame(0, COUNTER_MAX, 0, b"", flags=0) == _KNOWN_EMPTY_FRAME


def test_known_answer_vector_decodes() -> None:
    frame = decode_frame(_KNOWN_FRAME)
    assert frame.identifier == 0x1ABCDEF
    assert frame.dlc == 8
    assert frame.flags == Flags.EXTENDED
    assert frame.counter == 7
    assert frame.timestamp_us == 987654321
    assert frame.payload == struct.pack("<2f", 1.5, -2.25)


# ---------------------------------------------------------------------------
# Encode/decode round-trip
# ---------------------------------------------------------------------------


def test_encode_decode_round_trips_every_field() -> None:
    payload = bytes(range(64))
    flags = Flags.ERROR | Flags.EXTENDED | Flags.RTR
    frame = encode_frame(ID_MAX, COUNTER_MAX, (1 << 64) - 1, payload, flags=flags)
    decoded = decode_frame(frame)
    assert decoded.identifier == ID_MAX
    assert decoded.dlc == 15
    assert decoded.flags == flags
    assert decoded.counter == COUNTER_MAX
    assert decoded.timestamp_us == (1 << 64) - 1
    assert decoded.payload == payload


@pytest.mark.parametrize(
    ("size", "dlc"),
    [(0, 0), (4, 4), (8, 8), (12, 9), (16, 10), (20, 11), (24, 12), (32, 13), (48, 14), (64, 15)],
)
def test_dlc_code_follows_the_canfd_mapping(size: int, dlc: int) -> None:
    assert decode_frame(encode_frame(0x100, 0, 0, b"\x00" * size)).dlc == dlc


@pytest.mark.parametrize("flags", [0, 1, 2, 4, 7])
def test_flags_round_trip(flags: int) -> None:
    assert decode_frame(encode_frame(0x100, 1, 1, b"\x00" * 4, flags=flags)).flags == (Flags(flags))


@pytest.mark.parametrize("timestamp_us", [0, 1, (1 << 64) - 1])
def test_timestamp_round_trips(timestamp_us: int) -> None:
    frame = encode_frame(0x100, 1, timestamp_us, b"")
    assert decode_frame(frame).timestamp_us == timestamp_us


def test_identifier_round_trips_at_the_29_bit_limit() -> None:
    frame = encode_frame(ID_MAX, 1, 1, b"\x00" * 4)
    assert decode_frame(frame).identifier == ID_MAX


# ---------------------------------------------------------------------------
# Corruption detection
# ---------------------------------------------------------------------------


def test_decode_rejects_corrupted_payload() -> None:
    frame = bytearray(encode_frame(0x1ABCDEF, 7, 987654321, struct.pack("<2f", 1.5, -2.25)))
    frame[18] ^= 0x01  # first payload byte
    with pytest.raises(FrameError, match="CRC"):
        decode_frame(bytes(frame))


def test_decode_rejects_corrupted_header() -> None:
    frame = bytearray(encode_frame(0x1ABCDEF, 7, 987654321, struct.pack("<2f", 1.5, -2.25)))
    frame[0] ^= 0x01  # identifier low byte
    with pytest.raises(FrameError, match="CRC"):
        decode_frame(bytes(frame))


def test_decode_rejects_corrupted_crc() -> None:
    frame = bytearray(encode_frame(0x1ABCDEF, 7, 987654321, struct.pack("<2f", 1.5, -2.25)))
    frame[-1] ^= 0x01
    with pytest.raises(FrameError, match="CRC"):
        decode_frame(bytes(frame))


def test_decode_rejects_truncated_and_extended_frames() -> None:
    frame = encode_frame(0x100, 0, 1, b"\x00" * 8)
    with pytest.raises(FrameError):
        decode_frame(frame[:-1])
    with pytest.raises(FrameError):
        decode_frame(frame + b"\x00")
    with pytest.raises(FrameError):
        decode_frame(frame[:10])
    with pytest.raises(FrameError):
        decode_frame(b"")


def test_decode_rejects_non_bytes_input() -> None:
    frame = encode_frame(0x100, 0, 1, b"")
    with pytest.raises(FrameError):
        decode_frame(bytearray(frame))  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# DLC enforcement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 36, 40, 44, 52, 56, 60, 68, 100])
def test_encode_rejects_illegal_payload_lengths(size: int) -> None:
    with pytest.raises(FrameError):
        encode_frame(0x100, 0, 0, b"\x00" * size)


def test_encode_rejects_non_bytes_payload() -> None:
    for payload in (bytearray(4), "0000", None, [0, 0, 0, 0]):
        with pytest.raises(FrameError, match="payload must be bytes"):
            encode_frame(0x100, 1, 1, payload)


def test_decode_rejects_dlc_length_mismatch_even_with_a_valid_crc() -> None:
    # An 8-byte payload whose DLC byte claims 64 bytes, re-stamped with a
    # correct CRC so only the DLC/length check can reject it.
    body = bytearray(encode_frame(0x100, 0, 1, b"\x00" * 8))
    body[4] = 15
    with pytest.raises(FrameError, match="DLC"):
        decode_frame(_reframe(body))


def test_decode_rejects_dlc_out_of_range_even_with_a_valid_crc() -> None:
    body = bytearray(encode_frame(0x100, 0, 1, b""))
    body[4] = 16
    with pytest.raises(FrameError, match="DLC"):
        decode_frame(_reframe(body))


def test_decode_rejects_non_float32_payload_length() -> None:
    # DLC 7 maps to 7 bytes: legal CAN-FD, but not a whole number of
    # float32 slots, so this columnar codec refuses it. The frame is
    # built by hand because encode_frame rejects 7-byte payloads.
    body = struct.pack("<IBBIQ", 0x100, 7, 0, 1, 1) + b"\x00" * 7
    frame = body + struct.pack("<I", _crc32_iso_hdlc(body))
    with pytest.raises(FrameError, match="float32"):
        decode_frame(frame)


# ---------------------------------------------------------------------------
# Identifier enforcement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("identifier", [ID_MAX + 1, 1 << 29, -1, 0x1FFFFFFFF])
def test_encode_rejects_out_of_range_identifiers(identifier: int) -> None:
    with pytest.raises(FrameError, match="29-bit"):
        encode_frame(identifier, 1, 1, b"")


@pytest.mark.parametrize("identifier", [True, 1.0, "0x100", None])
def test_encode_rejects_non_integer_identifier(identifier: object) -> None:
    with pytest.raises(FrameError, match="29-bit"):
        encode_frame(identifier, 1, 1, b"")  # pyright: ignore[reportArgumentType]


def test_decode_rejects_identifier_above_29_bits() -> None:
    body = bytearray(encode_frame(0x100, 1, 1, b"\x00" * 4))
    body[3] = 0x20  # bit 29 of the little-endian uint32 identifier
    with pytest.raises(FrameError, match="29-bit"):
        decode_frame(_reframe(body))


# ---------------------------------------------------------------------------
# Rolling counter
# ---------------------------------------------------------------------------


def test_next_counter_wraps_at_uint32() -> None:
    assert next_counter(0) == 1
    assert next_counter(42) == 43
    assert next_counter(COUNTER_MAX - 1) == COUNTER_MAX
    assert next_counter(COUNTER_MAX) == 0


def test_rolling_counter_sequence_wraps() -> None:
    counter = COUNTER_MAX - 1
    for expected in (COUNTER_MAX - 1, COUNTER_MAX, 0, 1):
        assert counter == expected
        counter = next_counter(counter)


@pytest.mark.parametrize("counter", [COUNTER_MAX + 1, -1, True, 1.0])
def test_next_counter_validates_input(counter: object) -> None:
    with pytest.raises(FrameError, match="32-bit"):
        next_counter(counter)  # pyright: ignore[reportArgumentType]


def test_counter_round_trips_at_the_maximum() -> None:
    frame = encode_frame(0x100, COUNTER_MAX, 0, b"\x00" * 4)
    assert decode_frame(frame).counter == COUNTER_MAX


# ---------------------------------------------------------------------------
# Message specification and bus-load model
# ---------------------------------------------------------------------------


def test_message_spec_rejects_payload_sizes_the_codec_refuses() -> None:
    # Multiples of four inside 0..64 that are not CAN-FD DLC lengths,
    # and DLC lengths that are not whole float32 slots: the bus model
    # must not be able to schedule a frame the codec cannot encode.
    for size in (36, 40, 44, 52, 56, 60):
        with pytest.raises(ValueError, match="DLC"):
            MessageSpec("m", 0x100, 10_000, size)
    for size in (1, 2, 3, 5, 6, 7):
        with pytest.raises(ValueError, match="float32"):
            MessageSpec("m", 0x100, 10_000, size)


@pytest.mark.parametrize("size", _COLUMNAR_PAYLOAD_BYTES)
def test_message_spec_accepts_every_columnar_payload_size(size: int) -> None:
    spec = MessageSpec("m", 0x100, 10_000, size)
    assert spec.wire_bits == 8 * (22 + size) + 67
    # A scheduled message must be encodable on the wire.
    encode_frame(spec.identifier, 0, 0, b"\x00" * spec.payload_bytes)


def test_message_spec_validates_its_fields() -> None:
    with pytest.raises(ValueError, match="name"):
        MessageSpec("", 0x100, 10_000, 4)
    with pytest.raises(ValueError, match="identifier"):
        MessageSpec("m", ID_MAX + 1, 10_000, 4)
    with pytest.raises(ValueError, match="identifier"):
        MessageSpec("m", -1, 10_000, 4)
    with pytest.raises(ValueError, match="period"):
        MessageSpec("m", 0x100, 0, 4)
    with pytest.raises(ValueError, match="period"):
        MessageSpec("m", 0x100, -1, 4)


def test_bus_utilisation_validates_its_inputs() -> None:
    spec = MessageSpec("m", 0x100, 10_000, 4)
    with pytest.raises(ValueError, match="specs"):
        bus_utilisation((), 1_000_000)
    with pytest.raises(ValueError, match="bitrate"):
        bus_utilisation((spec,), 0)
    with pytest.raises(ValueError, match="bitrate"):
        schedule((), 10, 0)
    with pytest.raises(ValueError, match="duration"):
        schedule((spec,), -1, 1_000_000)


def test_bus_utilisation_formula() -> None:
    spec = MessageSpec("m", 0x100, 1_000_000, 64)  # 755 wire bits per second
    assert bus_utilisation((spec,), 1_000_000) == pytest.approx(0.000755)


# ---------------------------------------------------------------------------
# Full-rate bus load (PLAN.md §8.2: < 70 %)
# ---------------------------------------------------------------------------


def test_full_rate_layout_covers_every_periodic_channel(
    contract: ChannelContract,
) -> None:
    slots_by_rate = _periodic_rates(contract)
    # The contract inventory from channels.yaml: 74 periodic channels,
    # 7160 channel-samples/s, event channels excluded.
    assert slots_by_rate == {200: 14, 100: 40, 20: 16, 10: 4}
    assert contract.samples_per_second() == 7160.0
    specs = _full_rate_layout(contract)
    assert sum(spec.payload_bytes for spec in specs) == 4 * sum(slots_by_rate.values())


def test_full_rate_bus_load_is_under_70_percent(contract: ChannelContract) -> None:
    utilisation = bus_utilisation(_full_rate_layout(contract), _BITRATE_BPS)
    assert utilisation < 0.70
    # Exact figure: 730 frames/s carrying 406 510 bit/s over 2 Mbit/s.
    # (At 500 kbit/s the same layout would need 81.3 %, which is why the
    # virtual bus is sized for CAN-FD 2 Mbit/s rather than classic CAN.)
    assert utilisation == pytest.approx(0.203255, abs=1e-9)


def test_full_rate_bus_load_at_1_mbit_s_is_under_70_percent(
    contract: ChannelContract,
) -> None:
    assert bus_utilisation(_full_rate_layout(contract), 1_000_000) == pytest.approx(
        0.40651, abs=1e-9
    )


def test_production_specs_match_the_full_rate_layout(contract: ChannelContract) -> None:
    """The production contract boundary derives the same schedule the oracle checks.

    All load (< 70 %), arbitration, and starvation properties proven below
    for ``_full_rate_layout`` therefore hold for the production path, which
    previously had no contract-derived schedule of its own.
    """
    assert message_specs_from_contract(contract) == _full_rate_layout(contract)


def test_production_specs_are_deterministic(contract: ChannelContract) -> None:
    assert message_specs_from_contract(contract) == message_specs_from_contract(contract)


# ---------------------------------------------------------------------------
# Identifier-priority arbitration
# ---------------------------------------------------------------------------


def test_lower_identifier_wins_arbitration() -> None:
    high = MessageSpec("high_priority", 0x050, 100_000, 16)
    low = MessageSpec("low_priority", 0x100, 100_000, 16)
    first, second = schedule((low, high), 1_000, _BITRATE_BPS)
    assert first.name == "high_priority"
    assert first.start_us == 0
    assert second.name == "low_priority"
    assert second.start_us == first.finish_us


def test_arbitration_is_non_preemptive() -> None:
    # The high-priority frame released at t=200 us lands inside the
    # low-priority frame's transmission (186..564 us) and must wait for
    # it to finish: CAN arbitration happens per frame, never mid-frame.
    low = MessageSpec("low_priority", 0x100, 1_000_000, 64)
    high = MessageSpec("high_priority", 0x050, 200, 16)
    events = schedule((low, high), 1_000, _BITRATE_BPS)
    by_release = {(m.name, m.release_us): m for m in events}
    first_low = by_release[("low_priority", 0)]
    assert first_low.start_us == 186  # behind the t=0 high-priority frame
    released_during = by_release[("high_priority", 200)]
    assert released_during.start_us == first_low.finish_us  # waited, no preemption
    assert all(
        events[index].start_us >= events[index - 1].finish_us for index in range(1, len(events))
    )


def test_schedule_is_deterministic() -> None:
    specs = (
        MessageSpec("a", 0x050, 5_000, 32),
        MessageSpec("b", 0x100, 10_000, 16),
        MessageSpec("c", 0x0FF, 20_000, 64),
    )
    assert schedule(specs, 100_000, _BITRATE_BPS) == schedule(specs, 100_000, _BITRATE_BPS)


def test_release_cadence_matches_the_period() -> None:
    spec = MessageSpec("periodic", 0x100, 5_000, 16)
    events = schedule((spec,), 100_000, _BITRATE_BPS)
    assert [m.release_us for m in events] == list(range(0, 100_000, 5_000))
    # A lone message never waits: it starts exactly at its release.
    assert all(m.start_us == m.release_us for m in events)


def test_schedule_releases_outside_the_window_are_dropped() -> None:
    spec = MessageSpec("m", 0x100, 5_000, 16)
    events = schedule((spec,), 1_000, _BITRATE_BPS)
    assert [m.release_us for m in events] == [0]


def test_schedule_empty_window_returns_no_events() -> None:
    spec = MessageSpec("m", 0x100, 10_000, 4)
    assert schedule((spec,), 0, _BITRATE_BPS) == ()


# ---------------------------------------------------------------------------
# Starvation proof: 10 Hz engine frame vs 200 Hz IMU (PLAN.md §8.2)
# ---------------------------------------------------------------------------


def test_ten_hz_engine_frame_is_not_starved_by_200_hz_imu(
    contract: ChannelContract,
) -> None:
    specs = _full_rate_layout(contract)
    periods = {spec.name: spec.period_us for spec in specs}
    # Adversarial arrangement: the engine-temperature frame holds the
    # highest identifier, i.e. the lowest arbitration priority, so the
    # 200 Hz IMU traffic outranks it on every single arbitration.
    engine_ids = {spec.identifier for spec in specs if spec.name.startswith("telemetry_10hz")}
    assert engine_ids == {max(spec.identifier for spec in specs)}

    events = schedule(specs, _ONE_SECOND_US, _BITRATE_BPS)

    engine = [m for m in events if m.name.startswith("telemetry_10hz")]
    imu = [m for m in events if m.name.startswith("telemetry_200hz")]
    # Every release is transmitted: 10 engine frames, 2 IMU frames x 200.
    assert len(engine) == 10
    assert len(imu) == 400
    # No release anywhere in the full-rate schedule misses its deadline,
    # which is its own period.
    assert all(m.start_us < m.release_us + periods[m.name] for m in events)
    # Worst case is the t=0 cold-start burst, where every frame is
    # released at once and the engine frame waits behind 3 698 wire
    # bits of higher-priority traffic (1 852 us at 2 Mbit/s).
    latencies = [m.start_us - m.release_us for m in engine]
    assert max(latencies) == pytest.approx(1_852, abs=1)
    assert max(latencies) < periods["telemetry_10hz_0"]
