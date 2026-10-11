"""P7-T2/T3 Layer 1 physics residuals: hand-derived fixtures, one per model.

Clean values are worked by hand, not generated from the model formulas, so a wrong
conversion in the module fails here. ``vx = 40`` and ``vy = 30`` give a ground speed of
``hypot(40, 30) = 50`` m/s, i.e. 180 km/h. Slip-free rim speed is ``3.6 * 40 = 144`` km/h,
and 5 % slip on the rear left gives ``144 * 1.05 = 151.2`` km/h. Noise floors are read off
the ``contract`` fixture, never restated.

The last two tests use faults from :mod:`f1telemetry.telemetry.sensors` and check that the
residual equals the injected offset. That is the property PLAN.md section 9 rests on.
"""

from __future__ import annotations

import pytest

from f1telemetry.analytics.residuals import (
    MODELLED_CHANNELS,
    ResidualSample,
    attribute,
    residual_z_scores,
)
from f1telemetry.contracts.channels import ChannelContract
from f1telemetry.telemetry.sensors import FaultParameters, apply_fault
from f1telemetry.testing.records import SensorFrame

_CORNERS = ("fl", "fr", "rl", "rr")


def _clean_values() -> dict[str, float]:
    values: dict[str, float] = {"vx": 40.0, "vy": 30.0, "speed": 180.0}
    for corner in _CORNERS:
        values[f"slip_ratio_{corner}"] = 0.0
        values[f"wheel_speed_{corner}"] = 144.0
        values[f"tyre_temp_{corner}"] = 90.0
    values["slip_ratio_rl"] = 5.0
    values["wheel_speed_rl"] = 151.2
    return values


def _clean_frames(count: int = 8) -> tuple[SensorFrame, ...]:
    return tuple(SensorFrame(t_s=index * 0.005, values=_clean_values()) for index in range(count))


def _with_value(
    frames: tuple[SensorFrame, ...], index: int, channel: str, value: float
) -> tuple[SensorFrame, ...]:
    out = list(frames)
    out[index] = SensorFrame(t_s=frames[index].t_s, values={**frames[index].values, channel: value})
    return tuple(out)


def _sample(samples: tuple[ResidualSample, ...], channel: str, index: int) -> ResidualSample:
    matches = [s for s in samples if s.channel == channel and s.index == index]
    assert len(matches) == 1, f"expected one sample for {channel}@{index}, got {len(matches)}"
    return matches[0]


def test_clean_record_has_zero_z_scores(contract: ChannelContract) -> None:
    samples = residual_z_scores(_clean_frames(), contract)
    assert {sample.channel for sample in samples} == set(MODELLED_CHANNELS)
    assert max(abs(sample.z_score) for sample in samples) < 1e-9


def test_speed_uses_ground_speed_not_longitudinal_alone(contract: ChannelContract) -> None:
    speed = _sample(residual_z_scores(_clean_frames(), contract), "speed", 0)
    assert speed.predicted == pytest.approx(180.0)
    assert speed.residual == pytest.approx(0.0, abs=1e-9)


def test_wheel_speed_follows_slip_ratio(contract: ChannelContract) -> None:
    samples = residual_z_scores(_clean_frames(), contract)
    assert _sample(samples, "wheel_speed_rl", 0).predicted == pytest.approx(151.2)
    assert _sample(samples, "wheel_speed_fl", 0).predicted == pytest.approx(144.0)


def test_hold_baseline_starts_after_its_window(contract: ChannelContract) -> None:
    samples = residual_z_scores(_clean_frames(), contract)
    tyre = [s.index for s in samples if s.channel == "tyre_temp_fl"]
    assert tyre == [5, 6, 7]


def test_spiked_speed_trips_speed_only(contract: ChannelContract) -> None:
    frames = _with_value(_clean_frames(), 4, "speed", 181.5)
    samples = residual_z_scores(frames, contract)
    ranked = attribute(samples, 4)
    assert ranked[0].channel == "speed"
    assert ranked[0].residual == pytest.approx(1.5)
    assert ranked[0].z_score == pytest.approx(15.0)
    assert all(abs(sample.z_score) < 1e-9 for sample in ranked[1:])


def test_spiked_wheel_speed_trips_that_corner_only(contract: ChannelContract) -> None:
    frames = _with_value(_clean_frames(), 2, "wheel_speed_fr", 142.2)
    samples = residual_z_scores(frames, contract)
    ranked = attribute(samples, 2)
    assert ranked[0].channel == "wheel_speed_fr"
    assert ranked[0].z_score == pytest.approx(-3.0)
    assert all(abs(sample.z_score) < 1e-9 for sample in ranked[1:])


def test_tyre_spike_trips_then_hold_baseline_recovers(contract: ChannelContract) -> None:
    frames = _with_value(_clean_frames(), 6, "tyre_temp_fl", 95.0)
    samples = residual_z_scores(frames, contract)
    assert _sample(samples, "tyre_temp_fl", 6).z_score == pytest.approx(10.0)
    assert abs(_sample(samples, "tyre_temp_fl", 7).z_score) < 1e-9


def test_injected_spike_equals_residual(contract: ChannelContract) -> None:
    clean = _clean_frames()
    channel = contract.by_name("speed")
    onset = 4
    clean_value = clean[onset].values["speed"]
    faulted_value = apply_fault(
        clean_value, channel, "spike", FaultParameters(onset_index=onset), index=onset, seed=7
    )
    injected = faulted_value - clean_value
    assert abs(injected) == pytest.approx(3.0 * channel.sigma)

    samples = residual_z_scores(_with_value(clean, onset, "speed", faulted_value), contract)
    sample = _sample(samples, "speed", onset)
    assert sample.residual == pytest.approx(injected, abs=1e-9)
    assert sample.z_score == pytest.approx(injected / channel.sigma)
    assert abs(sample.z_score) == pytest.approx(3.0)


def test_injected_step_equals_residual_on_every_affected_frame(
    contract: ChannelContract,
) -> None:
    clean = _clean_frames()
    channel = contract.by_name("wheel_speed_rl")
    onset = 3
    params = FaultParameters(severity=1.0, onset_index=onset)
    frames = clean
    injected: dict[int, float] = {}
    for index in range(onset, len(clean)):
        clean_value = clean[index].values["wheel_speed_rl"]
        faulted_value = apply_fault(clean_value, channel, "step", params, index=index)
        injected[index] = faulted_value - clean_value
        frames = _with_value(frames, index, "wheel_speed_rl", faulted_value)

    # Hand-derived: 0.05 * (400 - 0) km/h * severity 1.0 = 20 km/h offset.
    assert injected[onset] == pytest.approx(20.0)
    samples = residual_z_scores(frames, contract)
    for index, offset in injected.items():
        assert _sample(samples, "wheel_speed_rl", index).residual == pytest.approx(offset, abs=1e-9)
    assert abs(_sample(samples, "wheel_speed_rl", onset - 1).z_score) < 1e-9


def test_non_finite_observation_is_skipped(contract: ChannelContract) -> None:
    frames = _with_value(_clean_frames(), 3, "speed", float("nan"))
    samples = residual_z_scores(frames, contract)
    assert not [s for s in samples if s.channel == "speed" and s.index == 3]
    assert [s for s in samples if s.channel == "speed" and s.index == 2]


def test_model_skipped_when_inputs_missing(contract: ChannelContract) -> None:
    frames = tuple(SensorFrame(t_s=i * 0.005, values={"wheel_speed_fl": 144.0}) for i in range(8))
    assert residual_z_scores(frames, contract) == ()


def test_default_contract_loads_channels_yaml(contract: ChannelContract) -> None:
    default = residual_z_scores(_clean_frames())
    assert default == residual_z_scores(_clean_frames(), contract)


def test_hold_window_must_be_positive() -> None:
    with pytest.raises(ValueError, match="hold_window"):
        residual_z_scores(_clean_frames(), hold_window=0)
