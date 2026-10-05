"""P2 planar integration: steering, four-corner forces, yaw and body motion."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from f1telemetry.contracts.car_spec import CarSpec
from f1telemetry.kernels import longitudinal  # noqa: TID251 -- kernel boundary regression
from f1telemetry.physics import forces  # noqa: TID251 -- kernel boundary regression

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import KernelConfig


@pytest.fixture(scope="module")
def config(spec: CarSpec) -> KernelConfig:
    return spec.kernel_config()


def _run(
    config: KernelConfig,
    steps: int,
    steer_wheel_deg: np.ndarray | None = None,
) -> tuple[np.ndarray, longitudinal.StepOutputs]:
    speed = 20.0
    state = longitudinal.initial_state(
        speed_m_s=speed,
        wheel_omega_rad_s=speed / config.rolling_radius_m,
    )
    outputs = longitudinal.allocate_step_outputs(steps)
    trace = longitudinal.simulate(
        config,
        steps,
        state,
        np.zeros(steps, dtype=np.float64),
        longitudinal.allocate(steps),
        step_outputs=outputs,
        steer_wheel_deg=steer_wheel_deg,
    )
    return trace, outputs


def test_a_straight_run_has_no_lateral_body_motion(config: KernelConfig) -> None:
    trace, outputs = _run(config, 500)

    assert np.allclose(trace[:, longitudinal.Y_INDEX], 0.0, atol=1.0e-6)
    assert np.allclose(trace[:, longitudinal.VY_INDEX], 0.0, atol=1.0e-5)
    assert np.allclose(trace[:, longitudinal.PSI_INDEX], 0.0, atol=1.0e-6)
    assert np.allclose(trace[:, longitudinal.YAW_RATE_INDEX], 0.0, atol=1.0e-4)
    assert np.allclose(outputs.force_y_n[:, 0], -outputs.force_y_n[:, 1], atol=1.0)
    assert np.allclose(outputs.force_y_n[:, 2], -outputs.force_y_n[:, 3], atol=1.0)


def test_zero_camber_straight_control_has_no_lateral_forces(config: KernelConfig) -> None:
    zero_camber = replace(
        config,
        axle_static_camber_deg=np.zeros_like(config.axle_static_camber_deg),
        axle_camber_gain_deg_per_m=np.zeros_like(config.axle_camber_gain_deg_per_m),
    )

    trace, outputs = _run(zero_camber, 500)

    assert np.allclose(outputs.force_y_n, 0.0, atol=1.0)
    assert np.allclose(trace[:, longitudinal.VY_INDEX], 0.0, atol=1.0e-5)
    assert np.allclose(trace[:, longitudinal.YAW_RATE_INDEX], 0.0, atol=1.0e-4)


def test_steering_produces_leftward_force_yaw_and_load_transfer(config: KernelConfig) -> None:
    steps = 2_000
    steer = np.full(steps, 1.0, dtype=np.float64)
    trace, outputs = _run(config, steps, steer)

    assert np.any(outputs.force_y_n[:, :2] > 0.0)
    assert trace[-1, longitudinal.YAW_RATE_INDEX] > 0.0
    assert trace[-1, longitudinal.PSI_INDEX] > 0.0
    assert trace[-1, longitudinal.Y_INDEX] > 0.0
    assert trace[-1, longitudinal.PREVIOUS_AY_INDEX] > 0.0
    assert outputs.load_n[-1, forces.FR_WHEEL_INDEX] > outputs.load_n[-1, forces.FL_WHEEL_INDEX]
    assert outputs.load_n[-1, forces.RR_WHEEL_INDEX] > outputs.load_n[-1, forces.RL_WHEEL_INDEX]


def test_steering_history_is_validated_before_the_kernel(config: KernelConfig) -> None:
    bad_shape = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match=r"steer_wheel_deg.*shape"):
        _run(config, 2, bad_shape)

    too_far = np.full(2, config.max_steering_wheel_angle_deg + 1.0, dtype=np.float64)
    with pytest.raises(ValueError, match="exceeds"):
        _run(config, 2, too_far)
