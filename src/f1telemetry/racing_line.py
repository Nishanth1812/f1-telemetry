"""Deterministic racing-line offsets and a curvature-bound speed profile.

The speed profile consumes the *solved* offset line's curvature - the
racing line flattens corners - and enforces braking (backward) and
traction (forward) constraints around the whole lap.
"""

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

    ``bound_left_m`` / ``bound_right_m`` are the per-side offset limits the
    clip enforced: track edge distances minus ``margin_m`` on the left and
    right respectively. ``curvature_per_m`` is the solved line's signed
    curvature at the waypoint knots (centreline curvature plus the offset's
    second difference), which the speed profile consumes;
    ``centerline_curvature_per_m`` is the reference the line flattens.
    """

    s_m: np.ndarray
    lateral_m: np.ndarray
    bound_left_m: np.ndarray
    bound_right_m: np.ndarray
    curvature_per_m: np.ndarray
    centerline_curvature_per_m: np.ndarray
    objective: float
    converged: bool
    iterations: int
    message: str


@dataclass(frozen=True, slots=True)
class SpeedProfile:
    """Periodic speed profile at the track waypoints.

    ``converged`` is False when the relaxation did not settle within
    ``max_sweeps`` or the constraints could not all hold; the last
    iterate is still reported so the failure is inspectable rather
    than smoothed into an apparently valid profile.
    """

    s_m: np.ndarray
    speed_m_s: np.ndarray
    curvature_per_m: np.ndarray
    converged: bool = True


def minimum_curvature_offsets(
    track: Track,
    *,
    margin_m: float = 0.5,
    tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> LateralOffsetSolution:
    """Solve the periodic second-difference quadratic for lateral offsets."""
    candidate_track: object = track
    if not isinstance(candidate_track, Track):
        raise TypeError(  # pyright: ignore[reportUnreachable] -- preserve guard for untyped callers
            f"track must be a Track, got {type(track).__name__}"
        )
    margin = _finite_non_negative("margin_m", margin_m)
    tol = _finite_positive("tolerance", tolerance)
    iterations_limit = _non_negative_int("max_iterations", max_iterations)

    count = track.waypoint_count
    s_m = np.array(track.s_m, dtype=np.float64, copy=True)
    bound_left = np.maximum(np.array(track.width_left_m, dtype=np.float64, copy=True) - margin, 0.0)
    bound_right = np.maximum(
        np.array(track.width_right_m, dtype=np.float64, copy=True) - margin, 0.0
    )
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
        trial = np.minimum(np.maximum(lateral - step * gradient, -bound_right), bound_left)
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
        bound_left_m=bound_left,
        bound_right_m=bound_right,
        curvature_per_m=residual,
        centerline_curvature_per_m=curvature,
        objective=float(np.sum(residual * residual)),
        converged=converged,
        iterations=iterations,
        message=message,
    )


def speed_profile(
    track: Track,
    line: LateralOffsetSolution,
    *,
    max_speed_m_s: float,
    max_accel_m_s2: float,
    max_brake_m_s2: float,
    lateral_accel_m_s2: float,
    tolerance_m_s: float = 1e-12,
    max_sweeps: int = _MAX_SWEEPS_DEFAULT,
) -> SpeedProfile:
    """Relax corner speeds against periodic accel and brake bounds.

    Corner targets come from the *solved* line's curvature - the line
    ``line`` produced by :func:`minimum_curvature_offsets`, not the
    centreline's - because the car drives the line. The backward pass
    enforces braking capacity into every knot and the forward pass
    enforces the available traction/power-limited acceleration out of
    it; both repeat until the profile stops moving or ``max_sweeps``
    is exhausted, in which case ``converged`` is False instead of
    presenting the partial profile as a valid one. The line must have
    converged itself: an unsolved offset line has no trustworthy
    curvature to build a profile on.
    """
    candidate_track: object = track
    candidate_line: object = line
    if not isinstance(candidate_track, Track):
        raise TypeError(  # pyright: ignore[reportUnreachable] -- preserve guard for untyped callers
            f"track must be a Track, got {type(track).__name__}"
        )
    if not isinstance(candidate_line, LateralOffsetSolution):
        raise TypeError(  # pyright: ignore[reportUnreachable] -- preserve guard for untyped callers
            f"line must be a LateralOffsetSolution, got {type(line).__name__}"
        )
    if not line.converged:
        raise ValueError("speed_profile needs a converged racing line")
    if line.s_m.shape != track.s_m.shape or line.curvature_per_m.shape != track.s_m.shape:
        raise ValueError("racing line arrays must align with the track knots")
    max_speed = _finite_positive("max_speed_m_s", max_speed_m_s)
    max_accel = _finite_positive("max_accel_m_s2", max_accel_m_s2)
    max_brake = _finite_positive("max_brake_m_s2", max_brake_m_s2)
    max_lateral = _finite_positive("lateral_accel_m_s2", lateral_accel_m_s2)
    tol = _finite_non_negative("tolerance_m_s", tolerance_m_s)
    sweeps = _non_negative_int("max_sweeps", max_sweeps)

    s_m = np.array(track.s_m, dtype=np.float64, copy=True)
    ds_m = np.array(track.ds_m, dtype=np.float64, copy=True)
    curvature = np.abs(np.array(line.curvature_per_m, dtype=np.float64, copy=True))

    target = np.full_like(curvature, max_speed)
    nonzero = curvature > 0.0
    target[nonzero] = np.sqrt(max_lateral / curvature[nonzero])
    speed = np.minimum(target, max_speed)

    converged = False
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
            converged = True
            break

    return SpeedProfile(
        s_m=s_m,
        speed_m_s=speed,
        curvature_per_m=curvature,
        converged=converged,
    )


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
