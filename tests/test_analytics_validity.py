"""P7-T1 Layer 0 validity checks: hand-derived fixtures, one per rule.

Each rule trips on a crafted violation and stays silent on a clean record (negative
control). Bounds and noise floors are read off the ``contract`` fixture — the tests
never restate a number from ``channels.yaml``.
"""

from __future__ import annotations

from f1telemetry.analytics.validity import check_validity
from f1telemetry.contracts.channels import ChannelContract
from f1telemetry.testing.records import SensorFrame


def _frames(channel: str, values: tuple[float, ...], dt_s: float = 0.01) -> tuple[SensorFrame, ...]:
    return tuple(
        SensorFrame(t_s=index * dt_s, values={channel: value}) for index, value in enumerate(values)
    )


def _clean_speed() -> tuple[float, ...]:
    return tuple(100.0 + 0.5 * index for index in range(8))


def test_clean_record_stays_silent(contract: ChannelContract) -> None:
    assert check_validity(_frames("speed", _clean_speed()), contract) == ()


def test_default_contract_loads_channels_yaml() -> None:
    assert check_validity(_frames("speed", _clean_speed())) == ()


def test_range_trips_above_contract_max(contract: ChannelContract) -> None:
    ceiling = contract.by_name("speed").range_max
    values = tuple(ceiling - 4.0 + 2.0 * index for index in range(6))
    findings = check_validity(_frames("speed", values), contract)
    assert [(finding.channel, finding.index, finding.rule) for finding in findings] == [
        ("speed", 3, "range"),
        ("speed", 4, "range"),
        ("speed", 5, "range"),
    ]


def test_rate_of_change_trips_on_spike(contract: ChannelContract) -> None:
    values = (100.0, 100.5, 101.0, 150.0, 101.5, 102.0)
    findings = check_validity(_frames("speed", values), contract)
    assert [(finding.channel, finding.index, finding.rule) for finding in findings] == [
        ("speed", 3, "rate_of_change"),
        ("speed", 4, "rate_of_change"),
    ]


def test_stuck_trips_on_repeated_value(contract: ChannelContract) -> None:
    findings = check_validity(_frames("speed", (100.0,) * 6), contract)
    assert [(finding.channel, finding.index, finding.rule) for finding in findings] == [
        ("speed", 4, "stuck")
    ]


def test_timestamp_monotonicity_trips_on_duplicate_stamp(contract: ChannelContract) -> None:
    frames = list(_frames("speed", _clean_speed()))
    frames[3] = SensorFrame(t_s=frames[2].t_s, values=dict(frames[3].values))
    findings = check_validity(tuple(frames), contract)
    assert [(finding.channel, finding.index, finding.rule) for finding in findings] == [
        ("t_s", 3, "timestamp_monotonicity")
    ]


def test_non_finite_value_trips_range_only(contract: ChannelContract) -> None:
    values = (100.0, float("nan"), 101.0)
    findings = check_validity(_frames("speed", values), contract)
    assert [(finding.channel, finding.index, finding.rule) for finding in findings] == [
        ("speed", 1, "range")
    ]


def test_discrete_channel_range_without_rate(contract: ChannelContract) -> None:
    assert contract.by_name("gear").sigma == 0.0
    findings = check_validity(_frames("gear", (3.0, 4.0, 4.0, 9.0, 4.0, 5.0)), contract)
    assert [(finding.channel, finding.index, finding.rule) for finding in findings] == [
        ("gear", 3, "range")
    ]
