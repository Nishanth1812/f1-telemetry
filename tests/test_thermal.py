"""P3 thermal boundaries: correct heat source, pressure coupling and finite traces."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from f1telemetry.contracts.car_spec import THERMAL_NODES

# P3 boundary tests deliberately exercise the thermal physics module.
from f1telemetry.physics.thermal import (  # noqa: TID251
    THERMAL_NODE_NAMES,
    brake_heat_w,
    convective_heat_flow_w,
    lumped_temperature_step_c,
    simulate_thermal_trace,
    tyre_gas_mass_step_kg,
    tyre_pressure_pa,
    tyre_slip_heat_w,
    validated_thermal_scalars,
)

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import CarSpec, KernelConfig


@pytest.fixture(scope="module")
def config(spec: CarSpec) -> KernelConfig:
    return spec.kernel_config()


def test_tyre_heat_uses_patch_slip_velocity_not_vehicle_road_speed() -> None:
    patch_force_n = 12_000.0
    slip_velocity_m_s = 0.25
    road_speed_m_s = 70.0

    heat_w = tyre_slip_heat_w(patch_force_n, slip_velocity_m_s)

    assert heat_w == patch_force_n * slip_velocity_m_s
    assert heat_w != patch_force_n * road_speed_m_s
    assert tyre_slip_heat_w(0.0, slip_velocity_m_s) == 0.0
    assert tyre_slip_heat_w(patch_force_n, 0.0) == 0.0


def test_brake_heat_uses_brake_torque_and_wheel_speed() -> None:
    assert brake_heat_w(500.0, 100.0, 0.75) == 37_500.0
    assert brake_heat_w(500.0, 0.0, 0.75) == 0.0


def test_lumped_node_is_steady_at_ambient_and_cools_when_hot() -> None:
    assert lumped_temperature_step_c(25.0, 10_000.0, 0.0, 0.0, 0.0, 0.01) == 25.0
    loss = convective_heat_flow_w(100.0, 0.25, 100.0, 25.0)
    cooled = lumped_temperature_step_c(100.0, 10_000.0, 0.0, loss, 0.0, 0.01)
    assert cooled < 100.0


def test_tyre_pressure_tracks_temperature_and_leak_independently() -> None:
    mass_kg, volume_m3, gas_constant = 0.08, 0.03, 287.05
    cold = tyre_pressure_pa(mass_kg, volume_m3, 25.0, gas_constant)
    hot = tyre_pressure_pa(mass_kg, volume_m3, 100.0, gas_constant)
    leaked_mass = tyre_gas_mass_step_kg(mass_kg, 1e-5, 10.0)
    leaked = tyre_pressure_pa(leaked_mass, volume_m3, 100.0, gas_constant)

    assert hot > cold
    assert leaked < hot
    assert leaked_mass == mass_kg - 1e-4


def test_long_thermal_trace_is_finite_and_repeatable(config: KernelConfig) -> None:
    count = 3_000
    slip = np.full((count, 4), 20.0)
    brake = np.full((count, 4), 5.0)
    heat = np.full(count, 8_000.0)
    speed = np.full(count, 45.0)
    first = simulate_thermal_trace(
        config,
        dt_s=0.01,
        speed_m_s=speed,
        tyre_slip_work_j=slip,
        brake_work_j=brake,
        engine_heat_j=heat,
        gearbox_heat_j=heat / 10,
        ambient_temp_c=25.0,
    )
    second = simulate_thermal_trace(
        config,
        dt_s=0.01,
        speed_m_s=speed,
        tyre_slip_work_j=slip,
        brake_work_j=brake,
        engine_heat_j=heat,
        gearbox_heat_j=heat / 10,
        ambient_temp_c=25.0,
    )

    for values in (
        first.tyre_temp_c,
        first.brake_temp_c,
        first.tyre_pressure_psi,
        first.engine_temp_c,
        first.gearbox_temp_c,
    ):
        assert np.isfinite(values).all()
    np.testing.assert_array_equal(first.tyre_temp_c, second.tyre_temp_c)
    np.testing.assert_array_equal(first.brake_temp_c, second.brake_temp_c)
    np.testing.assert_array_equal(first.tyre_pressure_psi, second.tyre_pressure_psi)
    np.testing.assert_array_equal(first.engine_temp_c, second.engine_temp_c)
    np.testing.assert_array_equal(first.gearbox_temp_c, second.gearbox_temp_c)


def test_thermal_node_order_matches_the_loader() -> None:
    assert THERMAL_NODE_NAMES == THERMAL_NODES == ("tyre", "brake", "engine", "gearbox")


def test_validated_thermal_scalars_mirror_the_config(config: KernelConfig) -> None:
    values = validated_thermal_scalars(config, "test")
    np.testing.assert_array_equal(values.initial_temp_c, config.thermal_node_initial_temp_c)
    np.testing.assert_array_equal(
        values.heat_capacity_j_per_k, config.thermal_node_heat_capacity_j_per_k
    )
    np.testing.assert_array_equal(values.cooling_area_m2, config.thermal_node_cooling_area_m2)
    np.testing.assert_array_equal(values.emissivity, config.thermal_node_emissivity)
    np.testing.assert_array_equal(
        values.airflow_base_w_m2_k, config.thermal_node_airflow_base_w_m2_k
    )
    np.testing.assert_array_equal(
        values.airflow_speed_gain_w_m2_k_per_m_s,
        config.thermal_node_airflow_speed_gain_w_m2_k_per_m_s,
    )
    assert values.brake_heat_fraction == config.thermal_brake_heat_fraction
    assert values.engine_waste_heat_share == config.thermal_engine_waste_heat_share
    assert values.gearbox_loss_share == config.thermal_gearbox_loss_share
    assert values.tyre_volume_m3 == config.thermal_tyre_volume_m3
    assert values.tyre_gas_constant_j_per_kg_k == config.thermal_tyre_gas_constant_j_per_kg_k
    assert values.tyre_initial_pressure_psi_gauge == config.thermal_tyre_initial_pressure_psi_gauge
