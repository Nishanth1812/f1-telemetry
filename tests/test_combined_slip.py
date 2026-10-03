"""P2-T4: normalized-slip-vector similarity and its ellipse contract."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from f1telemetry.contracts.car_spec import CarSpec
from f1telemetry.physics import (  # noqa: TID251 -- exercise the compiled physics composition
    combined_slip,
    forces,
    tyres,
)

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import KernelConfig

pytestmark = pytest.mark.forces

EXPECTED_OPTIONS: dict[str, Any] = {
    "fastmath": False,
    "nopython": True,
    "nogil": True,
    "boundscheck": False,
    "error_model": "numpy",
}


@pytest.fixture(scope="module")
def config(spec: CarSpec) -> KernelConfig:
    return spec.kernel_config()


@pytest.fixture(scope="module")
def params(config: KernelConfig) -> np.ndarray:
    return combined_slip.prepare_combined_slip_parameters(config, "test")


def _longitudinal(config: KernelConfig, kappa: float, load_n: float) -> float:
    return forces.tyre_longitudinal_force(
        kappa,
        load_n,
        config.pacejka_b,
        config.pacejka_c,
        config.pacejka_e,
        config.pacejka_mu,
    )


def _lateral(config: KernelConfig, alpha_deg: float, camber_deg: float, load_n: float) -> float:
    return tyres.tyre_lateral_force(
        alpha_deg,
        camber_deg,
        load_n,
        config.lateral_pacejka_b,
        config.lateral_pacejka_c,
        config.lateral_pacejka_e,
        config.lateral_pacejka_mu,
        config.load_sensitivity_reference_n,
        config.load_sensitivity_peak,
        config.load_sensitivity_stiffness,
        config.camber_stiffness_n_per_deg,
    )


@pytest.mark.parametrize("load_n", [250.0, 4000.0, 12000.0])
@pytest.mark.parametrize("kappa", [-0.8, -0.2, -0.01, 0.0, 0.01, 0.2, 0.8])
def test_pure_longitudinal_slip_preserves_the_p1_curve(
    config: KernelConfig, params: np.ndarray, kappa: float, load_n: float
) -> None:
    fx, fy = combined_slip.combined_tyre_forces(kappa, 0.0, 0.0, load_n, params)
    assert fx == pytest.approx(_longitudinal(config, kappa, load_n), rel=2e-12, abs=1e-12)
    assert fy == 0.0


@pytest.mark.parametrize("load_n", [250.0, 4000.0, 12000.0])
@pytest.mark.parametrize("alpha_deg", [-35.0, -8.0, -0.25, 0.0, 0.25, 8.0, 35.0])
@pytest.mark.parametrize("camber_deg", [-4.0, 0.0, 4.0])
def test_pure_lateral_slip_preserves_the_p2_t3_curve(
    config: KernelConfig,
    params: np.ndarray,
    alpha_deg: float,
    camber_deg: float,
    load_n: float,
) -> None:
    fx, fy = combined_slip.combined_tyre_forces(0.0, alpha_deg, camber_deg, load_n, params)
    assert fx == 0.0
    assert fy == pytest.approx(
        _lateral(config, alpha_deg, camber_deg, load_n), rel=2e-12, abs=1e-12
    )


@pytest.mark.parametrize("load_n", [100.0, 1000.0, 4000.0, 12000.0, 40000.0])
def test_normalized_force_stays_inside_the_load_dependent_ellipse(
    config: KernelConfig, params: np.ndarray, load_n: float
) -> None:
    dx = config.pacejka_mu * load_n
    dy = (
        tyres.lateral_peak_friction(
            load_n,
            config.lateral_pacejka_mu,
            config.load_sensitivity_reference_n,
            config.load_sensitivity_peak,
        )
        * load_n
    )
    for kappa in np.linspace(-1.5, 1.5, 41):
        for alpha_deg in np.linspace(-88.0, 88.0, 45):
            fx, fy = combined_slip.combined_tyre_forces(
                float(kappa), float(alpha_deg), 0.0, load_n, params
            )
            assert math.isfinite(fx) and math.isfinite(fy)
            utilization = (fx / dx) ** 2
            if dy > 0.0:
                utilization += (fy / dy) ** 2
            else:
                assert fy == 0.0
            assert utilization <= 1.0 + 2e-12


def test_the_ellipse_is_reached_at_the_derived_combined_peak(
    config: KernelConfig, params: np.ndarray
) -> None:
    """The bound test is non-vacuous: both pure axes peak at normalized radius one."""
    z_x = combined_slip.peak_argument(config.pacejka_c, config.pacejka_e)
    z_y = combined_slip.peak_argument(config.lateral_pacejka_c, config.lateral_pacejka_e)
    load_n = config.load_sensitivity_reference_n
    dx = config.pacejka_mu * load_n
    dy = config.lateral_pacejka_mu * load_n
    for angle_deg in range(0, 91, 15):
        theta = math.radians(angle_deg)
        kappa = z_x * math.cos(theta) / config.pacejka_b
        alpha_deg = math.degrees(z_y * math.sin(theta) / config.lateral_pacejka_b)
        fx, fy = combined_slip.combined_tyre_forces(kappa, alpha_deg, 0.0, load_n, params)
        assert (fx / dx) ** 2 + (fy / dy) ** 2 == pytest.approx(1.0, abs=2e-10)


@pytest.mark.parametrize("angle_deg", [0.0, 15.0, 30.0, 45.0, 60.0, 90.0])
def test_force_envelope_rises_to_and_falls_after_the_combined_peak(
    config: KernelConfig, params: np.ndarray, angle_deg: float
) -> None:
    theta = math.radians(angle_deg)
    z_x = combined_slip.peak_argument(config.pacejka_c, config.pacejka_e)
    z_y = combined_slip.peak_argument(config.lateral_pacejka_c, config.lateral_pacejka_e)
    load_n = config.load_sensitivity_reference_n
    dx = config.pacejka_mu * load_n
    dy = config.lateral_pacejka_mu * load_n
    utilization = []
    for radius in np.linspace(0.0, 2.0, 201):
        kappa = radius * z_x * math.cos(theta) / config.pacejka_b
        alpha_deg = math.degrees(radius * z_y * math.sin(theta) / config.lateral_pacejka_b)
        fx, fy = combined_slip.combined_tyre_forces(kappa, alpha_deg, 0.0, load_n, params)
        utilization.append((fx / dx) ** 2 + (fy / dy) ** 2)
    values = np.asarray(utilization)
    assert values[0] == 0.0
    assert np.all(np.diff(values[:101]) >= -2e-12)
    assert np.all(np.diff(values[100:]) <= 2e-12)
    assert values[100] == pytest.approx(1.0, abs=2e-10)


def test_zero_and_negative_load_return_zero_on_both_axes(params: np.ndarray) -> None:
    for load_n in (0.0, -100.0):
        assert combined_slip.combined_tyre_forces(0.3, 14.0, 2.0, load_n, params) == (
            0.0,
            0.0,
        )


def test_zero_slip_returns_zero_and_opposite_slips_mirror(params: np.ndarray) -> None:
    assert combined_slip.combined_tyre_forces(0.0, 0.0, 0.0, 4000.0, params) == (
        0.0,
        0.0,
    )
    forward = combined_slip.combined_tyre_forces(0.25, 12.0, 0.0, 4000.0, params)
    mirrored = combined_slip.combined_tyre_forces(-0.25, -12.0, 0.0, 4000.0, params)
    assert mirrored == pytest.approx((-forward[0], -forward[1]), rel=1e-12, abs=1e-12)


def test_cambered_combined_force_remains_inside_the_ellipse(
    config: KernelConfig, params: np.ndarray
) -> None:
    load_n = config.load_sensitivity_reference_n
    dx = config.pacejka_mu * load_n
    dy = config.lateral_pacejka_mu * load_n
    for camber_deg in (-8.0, -3.0, 0.0, 3.0, 8.0):
        for kappa in (-0.4, -0.1, 0.1, 0.4):
            for alpha_deg in (-25.0, -8.0, 8.0, 25.0):
                fx, fy = combined_slip.combined_tyre_forces(
                    kappa, alpha_deg, camber_deg, load_n, params
                )
                assert (fx / dx) ** 2 + (fy / dy) ** 2 <= 1.0 + 2e-12


def test_invalid_shape_and_curvature_coefficients_cannot_build_peak_axes(
    config: KernelConfig,
) -> None:
    for field, value in (
        ("pacejka_c", 1.0),
        ("pacejka_c", 2.01),
        ("pacejka_e", 1.0),
        ("lateral_pacejka_c", 1.0),
        ("lateral_pacejka_c", 2.01),
        ("lateral_pacejka_e", 1.0),
    ):
        with pytest.raises(ValueError, match=field):
            combined_slip.prepare_combined_slip_parameters(
                replace(config, **{field: value}), "test"
            )


@pytest.mark.parametrize("shape,curvature", [(1.0, 0.9), (2.01, 0.9), (1.5, 1.0)])
def test_peak_argument_rejects_nonpeaking_magic_formula_parameters(
    shape: float, curvature: float
) -> None:
    with pytest.raises(ValueError):
        combined_slip.peak_argument(shape, curvature)


def test_peak_argument_is_the_exact_unit_peak_for_both_shapes(
    config: KernelConfig,
) -> None:
    for shape, curvature in (
        (config.pacejka_c, config.pacejka_e),
        (config.lateral_pacejka_c, config.lateral_pacejka_e),
    ):
        peak = combined_slip.peak_argument(shape, curvature)
        value = math.sin(shape * math.atan(peak - curvature * (peak - math.atan(peak))))
        assert value == pytest.approx(1.0, abs=2e-14)


def test_combined_force_primitive_uses_the_project_numba_options(
    params: np.ndarray,
) -> None:
    combined_slip.combined_tyre_forces(0.1, 4.0, 0.0, 4000.0, params)
    for key, expected in EXPECTED_OPTIONS.items():
        assert combined_slip.combined_tyre_forces.targetoptions.get(key) is expected
