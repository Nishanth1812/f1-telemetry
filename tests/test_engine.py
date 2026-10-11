"""The P1 ICE rotational-speed primitive: net crank torque / configured inertia / dt.

``tests/test_gearbox.py`` pins the model that produces the differential-side torque, and
this file pins the engine-speed state that torque - divided back through ``gear ×
final_drive`` by the runner - pushes against. Every physical number comes from the
loaded ``car_spec.yaml`` through
:class:`~f1telemetry.contracts.car_spec.KernelConfig`, so this file tunes nothing.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from f1telemetry.physics import engine  # noqa: TID251 -- test target

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import CarSpec, KernelConfig

pytestmark = pytest.mark.powertrain


@pytest.fixture(scope="module")
def config(spec: CarSpec) -> KernelConfig:
    return spec.kernel_config()


def _expected_rpm(
    config: KernelConfig,
    ice_rpm: float,
    net_torque_nm: float,
    dt_s: float,
) -> float:
    """The unclamped integrated speed, computed independently of the module."""
    omega = ice_rpm * math.tau / 60.0
    omega += net_torque_nm / config.ice_inertia_kg_m2 * dt_s
    return omega * 60.0 / math.tau


def test_zero_net_torque_holds_speed(config: KernelConfig) -> None:
    """T_delivered == T_load: the engine coasts at exactly the speed it had."""
    assert engine.step_engine_speed(config, 8_000.0, 250.0, 250.0) == pytest.approx(8_000.0)


def test_acceleration_matches_torque_over_inertia(config: KernelConfig) -> None:
    """Positive net torque raises the speed by net/I * dt, in the configured inertia."""
    delivered, load = 400.0, 100.0
    got = engine.step_engine_speed(config, 8_000.0, delivered, load, dt_s=0.01)
    expected = _expected_rpm(config, 8_000.0, delivered - load, 0.01)
    assert got == pytest.approx(expected, rel=1e-12)
    assert got > 8_000.0


def test_deceleration_matches_torque_over_inertia(config: KernelConfig) -> None:
    """A resisting load larger than the delivered torque pulls the speed down."""
    delivered, load = 100.0, 400.0
    got = engine.step_engine_speed(config, 8_000.0, delivered, load, dt_s=0.01)
    expected = _expected_rpm(config, 8_000.0, delivered - load, 0.01)
    assert got == pytest.approx(expected, rel=1e-12)
    assert 8_000.0 > got > config.idle_rpm


def test_inertia_scales_the_speed_change_inversely(config: KernelConfig) -> None:
    """Doubling the configured inertia halves the same step's speed change."""
    heavier = replace(config, ice_inertia_kg_m2=2.0 * config.ice_inertia_kg_m2)
    base = engine.step_engine_speed(config, 8_000.0, 300.0, 0.0, dt_s=0.01) - 8_000.0
    doubled = engine.step_engine_speed(heavier, 8_000.0, 300.0, 0.0, dt_s=0.01) - 8_000.0
    assert doubled == pytest.approx(0.5 * base, rel=1e-12)


def test_default_dt_is_the_configured_step(config: KernelConfig) -> None:
    """Omitting ``dt_s`` integrates over ``config.dt_s`` so the step cannot disagree."""
    got = engine.step_engine_speed(config, 8_000.0, 300.0, 0.0)
    expected = _expected_rpm(config, 8_000.0, 300.0, config.dt_s)
    assert got == pytest.approx(expected, rel=1e-12)


def test_rev_limit_clamps_rather_than_running_away(config: KernelConfig) -> None:
    """A huge positive net torque holds the state at the rev limit, never above it."""
    got = engine.step_engine_speed(config, config.rev_limit_rpm - 1.0, 1.0e9, 0.0)
    assert got == config.rev_limit_rpm


def test_idle_floor_clamps_rather_than_stalling(config: KernelConfig) -> None:
    """A huge resisting load holds the state at idle, never at or below zero."""
    got = engine.step_engine_speed(config, config.idle_rpm + 1.0, 0.0, 1.0e9)
    assert got == config.idle_rpm
    assert got > 0.0


def test_free_acceleration_when_neutral_or_shift_cut(config: KernelConfig) -> None:
    """Zero reflected load - neutral or a shift cut - is free engine acceleration."""
    got = engine.step_engine_speed(config, 8_000.0, 300.0, 0.0, dt_s=0.01)
    assert got == pytest.approx(_expected_rpm(config, 8_000.0, 300.0, 0.01), rel=1e-12)
    assert got > 8_000.0


def test_reverse_engagement_decelerates_to_the_idle_floor(config: KernelConfig) -> None:
    """A caller-supplied reverse load can only pull the engine to idle, never negative."""
    got = engine.step_engine_speed(config, 6_000.0, 0.0, 5.0e6)
    assert got == config.idle_rpm
    assert got > 0.0


def test_engine_speed_never_leaves_the_configured_band(config: KernelConfig) -> None:
    """A sweep of adversarial torques/speeds stays finite and inside the band."""
    for rpm in (config.idle_rpm, 4_000.0, config.rev_limit_rpm):
        for delivered, load in ((0.0, 1.0e6), (1.0e6, 0.0), (-1.0e6, 0.0), (0.0, -1.0e6)):
            got = engine.step_engine_speed(config, rpm, delivered, load)
            assert math.isfinite(got)
            assert config.idle_rpm <= got <= config.rev_limit_rpm


def test_primitive_matches_the_public_step(config: KernelConfig) -> None:
    """The compiled torque-balance primitive is the same arithmetic the step uses."""
    got = engine.engine_acceleration_rad_s2(400.0, 100.0, config.ice_inertia_kg_m2)
    assert got == pytest.approx(300.0 / config.ice_inertia_kg_m2, rel=1e-15)


def test_rpm_omega_round_trip() -> None:
    """The two conversions are exact inverses so the boundary and step cannot drift."""
    assert engine.ice_rpm_from_omega_rad_s(engine.ice_omega_rad_s(12_345.0)) == pytest.approx(
        12_345.0, rel=1e-15
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), "8000", True])
def test_non_finite_or_non_numeric_inputs_are_rejected(config: KernelConfig, bad: Any) -> None:
    with pytest.raises(ValueError, match="step_engine_speed"):
        engine.step_engine_speed(config, bad, 300.0, 0.0)
    with pytest.raises(ValueError, match="step_engine_speed"):
        engine.step_engine_speed(config, 8_000.0, bad, 0.0)
    with pytest.raises(ValueError, match="step_engine_speed"):
        engine.step_engine_speed(config, 8_000.0, 300.0, bad)
    with pytest.raises(ValueError, match="step_engine_speed"):
        engine.step_engine_speed(config, 8_000.0, 300.0, 0.0, dt_s=bad)


def test_non_positive_dt_is_rejected(config: KernelConfig) -> None:
    with pytest.raises(ValueError, match="dt_s"):
        engine.step_engine_speed(config, 8_000.0, 300.0, 0.0, dt_s=0.0)
    with pytest.raises(ValueError, match="dt_s"):
        engine.step_engine_speed(config, 8_000.0, 300.0, 0.0, dt_s=-0.01)


def test_state_outside_the_band_is_rejected(config: KernelConfig) -> None:
    with pytest.raises(ValueError, match="ice_rpm"):
        engine.step_engine_speed(config, config.idle_rpm - 1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="ice_rpm"):
        engine.step_engine_speed(config, config.rev_limit_rpm + 1.0, 0.0, 0.0)


def test_zero_or_negative_inertia_config_is_rejected(config: KernelConfig) -> None:
    for bad in (0.0, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="ice_inertia_kg_m2"):
            engine.step_engine_speed(replace(config, ice_inertia_kg_m2=bad), 8_000.0, 300.0, 0.0)


def test_inverted_or_zero_band_config_is_rejected(config: KernelConfig) -> None:
    with pytest.raises(ValueError, match="idle_rpm"):
        engine.step_engine_speed(replace(config, idle_rpm=config.rev_limit_rpm), 8_000.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="idle_rpm"):
        engine.step_engine_speed(replace(config, idle_rpm=0.0), 8_000.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="idle_rpm"):
        engine.step_engine_speed(replace(config, idle_rpm=float("nan")), 8_000.0, 0.0, 0.0)


def test_bad_configured_dt_is_rejected(config: KernelConfig) -> None:
    with pytest.raises(ValueError, match="dt_s"):
        engine.step_engine_speed(replace(config, dt_s=0.0), 8_000.0, 300.0, 0.0)
