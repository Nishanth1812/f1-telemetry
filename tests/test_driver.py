"""P4 reference-driver requests share the lap-timing projection."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from f1telemetry.laps import project_position
from f1telemetry.racing_line import (
    LateralOffsetSolution,
    SpeedProfile,
    minimum_curvature_offsets,
    speed_profile,
)
from f1telemetry.reference_driver import (
    DriverRequest,
    _project,  # pyright: ignore[reportPrivateUsage] -- pins the shared projection
    pure_pursuit_request,
)
from f1telemetry.tracks import Track, load_track


@pytest.fixture
def track() -> Track:
    source = Path(__file__).parents[1] / "tracks" / "technical_ring.yaml"
    return load_track(source)


def _profile(track: Track, line: LateralOffsetSolution) -> SpeedProfile:
    return speed_profile(
        track,
        line,
        max_speed_m_s=80.0,
        max_accel_m_s2=8.0,
        max_brake_m_s2=12.0,
        lateral_accel_m_s2=25.0,
    )


def _request(track: Track, line: LateralOffsetSolution, profile: SpeedProfile) -> DriverRequest:
    x_m, y_m = track.point_at(0.0)
    frame = track.frame_at(0.0)
    return pure_pursuit_request(
        track,
        line,
        profile,
        x_m=x_m,
        y_m=y_m,
        heading_rad=frame.heading_rad,
        speed_m_s=15.0,
        wheelbase_m=3.6,
        steering_ratio=12.0,
    )


def _on_coarser_grid[SolutionT: (LateralOffsetSolution | SpeedProfile)](
    solution: SolutionT, step: int = 2
) -> SolutionT:
    """Resample every per-knot array of a solution onto every ``step``-th knot.

    The arrays stay consistent with one another, so the driver's own
    ``s_m``/value shape checks pass; only the arc lengths stop matching
    ``track.s_m``. That is the misalignment the track-knot guard exists
    for: a line or profile solved on some other grid would otherwise be
    interpolated - and steered and paced by - as if its knots were the
    track's.
    """
    resampled = {
        item.name: getattr(solution, item.name)[::step]
        for item in dataclasses.fields(solution)
        if isinstance(getattr(solution, item.name), np.ndarray)
    }
    return dataclasses.replace(solution, **resampled)


def _on_shifted_grid[SolutionT: (LateralOffsetSolution | SpeedProfile)](
    solution: SolutionT,
) -> SolutionT:
    """Keep the knot count but move arc-length coordinates off the track grid."""
    return dataclasses.replace(solution, s_m=solution.s_m + 0.25)


def test_driver_progress_matches_lap_timing_projection(track: Track) -> None:
    x_m, y_m = track.point_at(37.5)
    assert _project(track, x_m, y_m) == project_position(track, x_m, y_m).s_m


def test_pure_pursuit_returns_bounded_pedal_requests(track: Track) -> None:
    line = minimum_curvature_offsets(track)
    request = _request(track, line, _profile(track, line))
    assert 0.0 <= request.throttle <= 1.0
    assert 0.0 <= request.brake <= 1.0
    assert request.throttle == 0.0 or request.brake == 0.0


def test_pure_pursuit_refuses_a_line_whose_knots_miss_the_track(track: Track) -> None:
    """A line solved on another grid would steer off a foreign arc length."""
    line = minimum_curvature_offsets(track)
    with pytest.raises(ValueError, match="align with the track knots"):
        _request(track, _on_coarser_grid(line), _profile(track, line))


def test_pure_pursuit_refuses_a_profile_whose_knots_miss_the_track(track: Track) -> None:
    """A profile on another grid would pace the car to the wrong speed."""
    line = minimum_curvature_offsets(track)
    with pytest.raises(ValueError, match="align with the track knots"):
        _request(track, line, _on_coarser_grid(_profile(track, line)))


def test_pure_pursuit_refuses_same_sized_line_on_another_grid(track: Track) -> None:
    line = minimum_curvature_offsets(track)
    profile = _profile(track, line)
    with pytest.raises(ValueError, match="align with the track knots"):
        _request(track, _on_shifted_grid(line), profile)


def test_pure_pursuit_refuses_same_sized_profile_on_another_grid(track: Track) -> None:
    line = minimum_curvature_offsets(track)
    profile = _profile(track, line)
    with pytest.raises(ValueError, match="align with the track knots"):
        _request(track, line, _on_shifted_grid(profile))


def test_pure_pursuit_accepts_the_track_aligned_line_and_profile(track: Track) -> None:
    """The knot guard passes what the solvers actually produce for this track."""
    line = minimum_curvature_offsets(track)
    profile = _profile(track, line)
    assert line.s_m.shape == track.s_m.shape
    assert profile.s_m.shape == track.s_m.shape
    assert math.isfinite(_request(track, line, profile).steering_wheel_rad)
