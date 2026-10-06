"""Deterministic racing-line offsets and a curvature-bound speed profile."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Final

import numpy as np

from f1telemetry.tracks import Track

__all__ = [
    "LateralOffsetSolution",
    "SpeedProfile",
    "minimum_curvature_offsets",
    "speed_profile",
]

_MAX_SWEEPS_DEFAULT: Final[int] = 100


@dataclass(frozen=True, slots=True)
class LateralOffsetSolution:
    """Bounded periodic-offset solve result.

    ``converged`` is False when the projected-gradient loop exhausts
    ``max_iterations`` before ``tolerance`` is met; the last bounded iterate is
    still reported so the failure is inspectable rather than silently retried.
    """

    s_m: np.ndarray
    lateral_m: np.ndarray
    bound_m: np.ndarray
    objective: float
    converged: bool
    iterations: int
    message: str


@dataclass(frozen=True, slots=True)
class SpeedProfile:
    """Periodic speed profile at the track waypoints."""

    s_m: np.ndarray
    speed_m_s: np.ndarray
    curvature_per_m: np.ndarray


def minimum_curvature_offsets(
    track: Track,
    *,
    margin_m: float = 0.5,
    tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> LateralOffsetSolution:
    """Solve the periodic second-difference quadratic for lateral offsets."""
    if not isinstance(track, Track):
        raise TypeError(f"track must be a Track, got {type(track).__name__}")
    margin = _finite_non_negative("margin_m", margin_m)
    tol = _finite_positive("tolerance", tolerance)
    iterations_limit = _non_negative_int("max_iterations", max_iterations)

    count = track.waypoint_count
    s_m = np.array(track.s_m, dtype=np.float64, copy=True)
    bound = np.maximum(0.5 * np.array(track.width_m, dtype=np.float64, copy=True) - margin, 0.0)
    ds = np.array(track.ds_m, dtype=np.float64, copy=True)
    curvature = np.array([track.curvature_at(float(s)) for s in s_m], dtype=np.float64)

    # Line curvature ~= centerline curvature + second derivative of the
    # lateral offset, so the quadratic is ||B d + k||^2 over the loop.
    second = np.zeros((count, count), dtype=np.float64)
    for index in range(count):
        previous = (index - 1) % count
        following = (index + 1) % count
        span = 0.5 * (float(ds[previous]) + float(ds[index]))
        second[index, previous] = -1.0 / (float(ds[previous]) * span)
        second[index, index] = 1.0 / (float(ds[previous]) * span) - 1.0 / (float(ds[index]) * span)
        second[index, following] = 1.0 / (float(ds[index]) * span)
    quadratic = second.T @ second
    linear = second.T @ curvature
    lipschitz = float(np.linalg.eigvalsh(quadratic)[-1]) * 2.0
    step = 0.0 if lipschitz <= 0.0 else 1.0 / lipschitz

    lateral = np.zeros(count, dtype=np.float64)
    converged = False
    message = f"not converged after {iterations_limit} iterations"
    iterations = 0
    for candidate in range(1, iterations_limit + 1):
        iterations = candidate
        gradient = 2.0 * ((quadratic @ lateral) + linear)
        trial = np.clip(lateral - step * gradient, -bound, bound)
        moved = float(np.max(np.abs(trial - lateral))) if count else 0.0
        lateral = trial
        if moved <= tol:
            converged = True
            message = "converged"
            break

    residual = (second @ lateral) + curvature
    return LateralOffsetSolution(
        s_m=s_m,
        lateral_m=lateral,
        bound_m=bound,
        objective=float(np.sum(residual * residual)),
        converged=converged,
        iterations=iterations,
        message=message,
    )


def speed_profile(
    track: Track,
    *,
    max_speed_m_s: float,
    max_accel_m_s2: float,
    max_brake_m_s2: float,
    lateral_accel_m_s2: float,
    tolerance_m_s: float = 1e-12,
    max_sweeps: int = _MAX_SWEEPS_DEFAULT,
) -> SpeedProfile:
    """Relax corner speeds against periodic accel and brake bounds."""
    if not isinstance(track, Track):
        raise TypeError(f"track must be a Track, got {type(track).__name__}")
    max_speed = _finite_positive("max_speed_m_s", max_speed_m_s)
    max_accel = _finite_positive("max_accel_m_s2", max_accel_m_s2)
    max_brake = _finite_positive("max_brake_m_s2", max_brake_m_s2)
    max_lateral = _finite_positive("lateral_accel_m_s2", lateral_accel_m_s2)
    tol = _finite_positive("tolerance_m_s", tolerance_m_s)
    sweeps = _non_negative_int("max_sweeps", max_sweeps)

    s_m = np.array(track.s_m, dtype=np.float64, copy=True)
    ds_m = np.array(track.ds_m, dtype=np.float64, copy=True)
    curvature = np.array([abs(track.curvature_at(float(s))) for s in s_m], dtype=np.float64)

    target = np.full_like(curvature, max_speed)
    nonzero = curvature > 0.0
    target[nonzero] = np.sqrt(max_lateral / curvature[nonzero])
    speed = np.minimum(target, max_speed)

    for _ in range(sweeps):
        previous = speed.copy()
        for index in range(track.waypoint_count):
            following = (index + 1) % track.waypoint_count
            speed[index] = min(
                speed[index],
                math.sqrt(speed[following] ** 2 + 2.0 * max_brake * float(ds_m[index])),
            )
        for index in range(track.waypoint_count):
            following = (index + 1) % track.waypoint_count
            speed[following] = min(
                speed[following],
                math.sqrt(speed[index] ** 2 + 2.0 * max_accel * float(ds_m[index])),
            )
        if float(np.max(np.abs(speed - previous))) <= tol:
            break

    return SpeedProfile(s_m=s_m, speed_m_s=speed, curvature_per_m=curvature)


def _finite_positive(label: str, value: object) -> float:
    number = _finite_real(label, value)
    if number <= 0.0:
        raise ValueError(f"{label} must be > 0, got {value!r}")
    return number


def _finite_non_negative(label: str, value: object) -> float:
    number = _finite_real(label, value)
    if number < 0.0:
        raise ValueError(f"{label} must be >= 0, got {value!r}")
    return number


def _finite_real(label: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return number


def _non_negative_int(label: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be an int >= 0, got {value!r}")
    return value
