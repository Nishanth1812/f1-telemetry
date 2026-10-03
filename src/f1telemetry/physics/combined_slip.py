"""P2-T4 normalized-slip-vector similarity for combined tire forces."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import numpy as np
from numba import njit

from . import forces, tyres  # noqa: TID251 -- same-layer tire-model composition
from .pacejka import magic_formula_shape  # noqa: TID251 -- shared physics primitive

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import KernelConfig

__all__ = [
    "PARAMETER_COUNT",
    "combined_tyre_forces",
    "peak_argument",
    "prepare_combined_slip_parameters",
]

PARAMETER_COUNT: Final[int] = 14
_LONG_B: Final[int] = 0
_LONG_C: Final[int] = 1
_LONG_E: Final[int] = 2
_LONG_MU: Final[int] = 3
_LAT_B: Final[int] = 4
_LAT_C: Final[int] = 5
_LAT_E: Final[int] = 6
_LAT_MU: Final[int] = 7
_REFERENCE_LOAD: Final[int] = 8
_PEAK_SENSITIVITY: Final[int] = 9
_STIFFNESS_SENSITIVITY: Final[int] = 10
_CAMBER_STIFFNESS: Final[int] = 11
_LONG_PEAK_ARGUMENT: Final[int] = 12
_LAT_PEAK_ARGUMENT: Final[int] = 13
_RAD_PER_DEG: Final[float] = math.pi / 180.0


def peak_argument(shape: float, curvature: float) -> float:
    """Return the positive ``z = B * slip`` at the Magic Formula's unit peak.

    The peak condition is ``z (1-E) + E atan(z) = tan(pi / (2 C))``.
    For ``1 < C <= 2`` and ``E < 1`` the left-hand side increases strictly
    from zero to infinity, so a bracketed bisection gives the unique peak
    without introducing a fitted coefficient.
    """
    if isinstance(shape, bool) or not isinstance(shape, (int, float)):
        raise ValueError(f"shape must be a finite real number, got {shape!r}")
    if isinstance(curvature, bool) or not isinstance(curvature, (int, float)):
        raise ValueError(f"curvature must be a finite real number, got {curvature!r}")
    shape_value = float(shape)
    curvature_value = float(curvature)
    if not math.isfinite(shape_value) or not 1.0 < shape_value <= 2.0:
        raise ValueError(f"shape must be finite and in (1, 2], got {shape!r}")
    if not math.isfinite(curvature_value) or curvature_value >= 1.0:
        raise ValueError(f"curvature must be finite and < 1, got {curvature!r}")

    target = math.tan(math.pi / (2.0 * shape_value))

    def residual(z: float) -> float:
        return z * (1.0 - curvature_value) + curvature_value * math.atan(z) - target

    lower = 0.0
    upper = max(1.0, target / (1.0 - curvature_value))
    while residual(upper) < 0.0:
        upper *= 2.0
        if not math.isfinite(upper):
            raise ValueError("shape and curvature do not yield a finite peak argument")
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        if residual(middle) < 0.0:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def prepare_combined_slip_parameters(config: KernelConfig, prefix: str) -> np.ndarray:
    """Validate the two pure-slip models and pack their coefficients for the kernel."""
    longitudinal = forces.validated_config_scalars(config, prefix)
    lateral = tyres.validated_lateral_scalars(config, prefix)
    parameters = np.array(
        [
            longitudinal["pacejka_b"],
            longitudinal["pacejka_c"],
            longitudinal["pacejka_e"],
            longitudinal["pacejka_mu"],
            lateral["lateral_pacejka_b"],
            lateral["lateral_pacejka_c"],
            lateral["lateral_pacejka_e"],
            lateral["lateral_pacejka_mu"],
            lateral["load_sensitivity_reference_n"],
            lateral["load_sensitivity_peak"],
            lateral["load_sensitivity_stiffness"],
            lateral["camber_stiffness_n_per_deg"],
            peak_argument(longitudinal["pacejka_c"], longitudinal["pacejka_e"]),
            peak_argument(lateral["lateral_pacejka_c"], lateral["lateral_pacejka_e"]),
        ],
        dtype=np.float64,
    )
    return np.ascontiguousarray(parameters)


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def combined_tyre_forces(
    slip_ratio: float,
    slip_angle_deg: float,
    camber_deg: float,
    load_n: float,
    parameters: np.ndarray,
) -> tuple[float, float]:
    """Return ``(Fx, Fy)`` from the normalized-slip-vector similarity model.

    The pure longitudinal and lateral curves retain their own ``B``, ``C``,
    ``E`` and peak ``D``. Their slips are normalized by each curve's derived
    peak argument; the combined radius drives each axis's own Magic Formula
    shape, and direction cosines divide the resulting force between axes.
    Thus pure-axis behavior is unchanged and the load-dependent force ellipse
    follows from ``|sin| <= 1`` without clamping the resultant force.
    ``slip_angle_deg`` and ``camber_deg`` are degrees; ``slip_ratio`` is
    dimensionless and positive for drive; positive lateral force is leftward.
    """
    if load_n <= 0.0:
        return (0.0, 0.0)

    b_x = parameters[_LONG_B]
    c_x = parameters[_LONG_C]
    e_x = parameters[_LONG_E]
    mu_x = parameters[_LONG_MU]
    b_y = parameters[_LAT_B]
    c_y = parameters[_LAT_C]
    e_y = parameters[_LAT_E]
    mu_y = parameters[_LAT_MU]
    reference_load = parameters[_REFERENCE_LOAD]
    peak_sensitivity = parameters[_PEAK_SENSITIVITY]
    stiffness_sensitivity = parameters[_STIFFNESS_SENSITIVITY]
    camber_stiffness = parameters[_CAMBER_STIFFNESS]
    z_x = parameters[_LONG_PEAK_ARGUMENT]
    z_y = parameters[_LAT_PEAK_ARGUMENT]

    peak_x_n = mu_x * load_n
    effective_mu_y = tyres.lateral_peak_friction(load_n, mu_y, reference_load, peak_sensitivity)
    peak_y_n = effective_mu_y * load_n
    effective_b_y = tyres.lateral_slip_stiffness(load_n, b_y, reference_load, stiffness_sensitivity)
    reference_cornering_stiffness = tyres.reference_cornering_stiffness_n_per_deg(
        mu_y, b_y, c_y, reference_load
    )
    camber_slip_deg = tyres.camber_equivalent_slip_deg(
        camber_deg, camber_stiffness, reference_cornering_stiffness
    )
    alpha_rad = (slip_angle_deg + camber_slip_deg) * _RAD_PER_DEG

    normalized_x = b_x * slip_ratio / z_x
    normalized_y = 0.0
    if peak_y_n > 0.0 and effective_b_y > 0.0:
        normalized_y = effective_b_y * alpha_rad / z_y
    combined_radius = math.sqrt(normalized_x * normalized_x + normalized_y * normalized_y)
    if combined_radius == 0.0:
        return (0.0, 0.0)

    direction_x = normalized_x / combined_radius
    direction_y = normalized_y / combined_radius
    shape_x = magic_formula_shape(z_x * combined_radius, c_x, e_x)
    shape_y = magic_formula_shape(z_y * combined_radius, c_y, e_y)
    return (
        peak_x_n * shape_x * direction_x,
        peak_y_n * shape_y * direction_y,
    )
