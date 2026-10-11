"""Pure-pursuit requests for a track's computed line and speed profile."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from f1telemetry.laps import project_position
from f1telemetry.racing_line import LateralOffsetSolution, SpeedProfile
from f1telemetry.tracks import Track


@dataclass(frozen=True, slots=True)
class DriverRequest:
    steering_wheel_rad: float
    throttle: float
    brake: float
    target_speed_m_s: float


def pure_pursuit_request(
    track: Track,
    line: LateralOffsetSolution,
    profile: SpeedProfile,
    *,
    x_m: float,
    y_m: float,
    heading_rad: float,
    speed_m_s: float,
    wheelbase_m: float,
    steering_ratio: float,
    lookahead_m: float | None = None,
    grip_margin: float = 0.9,
    speed_gain: float = 0.2,
) -> DriverRequest:
    """Return steering and pedal requests without owning simulator state."""
    values = (
        x_m,
        y_m,
        heading_rad,
        speed_m_s,
        wheelbase_m,
        steering_ratio,
        grip_margin,
        speed_gain,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("driver inputs must be finite")
    if speed_m_s < 0.0 or wheelbase_m <= 0.0 or steering_ratio <= 0.0:
        raise ValueError("speed must be >= 0 and wheelbase/steering ratio must be > 0")
    if not 0.0 < grip_margin <= 1.0 or speed_gain <= 0.0:
        raise ValueError("grip_margin must be in (0, 1] and speed_gain must be > 0")
    if not line.converged:
        raise ValueError("cannot follow a racing line whose solver did not converge")
    if line.s_m.shape != line.lateral_m.shape or profile.s_m.shape != profile.speed_m_s.shape:
        raise ValueError("line and speed profile arrays have inconsistent dimensions")
    if not np.array_equal(line.s_m, track.s_m) or not np.array_equal(profile.s_m, track.s_m):
        raise ValueError("line and speed profile arc lengths must align with the track knots")
    progress = _project(track, x_m, y_m)
    lookahead = max(2.0, 0.5 * speed_m_s) if lookahead_m is None else lookahead_m
    if not math.isfinite(lookahead) or lookahead <= 0.0:
        raise ValueError("lookahead_m must be finite and > 0")
    target_s = (progress + lookahead) % track.length_m
    target_offset = float(np.interp(target_s, line.s_m, line.lateral_m, period=track.length_m))
    target_x, target_y = track.point_at(target_s, target_offset)
    dx, dy = target_x - x_m, target_y - y_m
    local_y = -math.sin(heading_rad) * dx + math.cos(heading_rad) * dy
    distance_sq = dx * dx + dy * dy
    curvature = 0.0 if distance_sq <= 1e-12 else 2.0 * local_y / distance_sq
    steering_road = math.atan(wheelbase_m * curvature)
    steering_wheel = steering_road * steering_ratio

    profile_speed = float(
        np.interp(progress, profile.s_m, profile.speed_m_s, period=track.length_m)
    )
    target_speed = max(0.0, profile_speed * grip_margin)
    error = target_speed - speed_m_s
    throttle = min(1.0, max(0.0, speed_gain * error))
    brake = min(1.0, max(0.0, -speed_gain * error))
    return DriverRequest(steering_wheel, throttle, brake, target_speed)


def _project(track: Track, x_m: float, y_m: float) -> float:
    """The position's arc length on the shared lap-timing projection.

    The driver projects through :func:`f1telemetry.laps.project_position`
    - the same dense spline-polyline projection, seam convention and
    wrapping that :func:`f1telemetry.laps.assess_lap` times a lap with -
    so the arc length the driver steers by and the arc length a sector
    or lap event is timed at are one quantity. A projection of its own,
    such as the raw waypoint chords, would drift from the timing
    module's between knots and put the driver's target on a different
    part of the lap than the one the events report.
    """
    return project_position(track, x_m, y_m).s_m
