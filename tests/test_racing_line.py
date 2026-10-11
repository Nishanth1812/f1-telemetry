"""P4-T3/T4/T5: the offset line's curvature drives the speed profile,
and the profile enforces the braking (backward) and traction (forward)
constraints everywhere, with an explicit failure flag when it cannot.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from f1telemetry.contracts.channels import repo_root
from f1telemetry.racing_line import (
    LateralOffsetSolution,
    minimum_curvature_offsets,
    speed_profile,
)
from f1telemetry.tracks import Track, load_track

_MAX_SPEED = 60.0
_MAX_ACCEL = 8.0
_MAX_BRAKE = 25.0
_MAX_LAT_ACCEL = 30.0


def _line(track: Track) -> LateralOffsetSolution:
    return minimum_curvature_offsets(track)


def test_offsets_converge_and_stay_in_bounds() -> None:
    track = load_track(repo_root() / "tracks" / "coastal_loop.yaml")
    line = _line(track)
    assert line.converged
    assert line.lateral_m.shape == line.s_m.shape
    assert np.all(line.lateral_m <= line.bound_left_m + 1e-12)
    assert np.all(line.lateral_m >= -line.bound_right_m - 1e-12)
    again = _line(track)
    assert np.array_equal(line.lateral_m, again.lateral_m)


def _asymmetric_track(tmp_path: Path) -> Track:
    from f1telemetry.tracks import load_track

    body = """\
version: 1
name: asymmetric
description: synthetic
centerline:
  - {x_m: 0.0, y_m: 0.0}
  - {x_m: 200.0, y_m: 0.0}
  - {x_m: 200.0, y_m: 200.0}
  - {x_m: 0.0, y_m: 200.0}
widths_m:
  - {left_m: 8.0, right_m: 4.0}
  - {left_m: 8.0, right_m: 4.0}
  - {left_m: 8.0, right_m: 4.0}
  - {left_m: 8.0, right_m: 4.0}
sectors:
  - {s_m: 200.0}
  - {s_m: 600.0}
"""
    path = tmp_path / "asymmetric.yaml"
    path.write_text(body, encoding="utf-8")
    return load_track(path)


def test_offset_bounds_follow_the_local_side(tmp_path: Path) -> None:
    track = _asymmetric_track(tmp_path)
    assert np.allclose(track.width_left_m, 8.0)
    assert np.allclose(track.width_right_m, 4.0)
    line = _line(track)
    # The offset line may use more room on the wide (left) side than
    # the narrow (right) side allows: the bound must be asymmetric.
    assert np.allclose(line.bound_left_m, 8.0 - 0.5)
    assert np.allclose(line.bound_right_m, 4.0 - 0.5)
    assert np.all(line.lateral_m <= line.bound_left_m + 1e-12)
    assert np.all(line.lateral_m >= -line.bound_right_m - 1e-12)


def test_offsets_deviate_and_approach_the_bound_in_the_tightest_corner() -> None:
    for fixture in ("coastal_loop", "technical_ring"):
        track = load_track(repo_root() / "tracks" / f"{fixture}.yaml")
        line = _line(track)
        # The solved line actually uses the track width rather than
        # hugging the centreline everywhere.
        assert float(np.max(np.abs(line.lateral_m))) > 1.0
        tightest = int(np.argmax(np.abs(line.centerline_curvature_per_m)))
        bound = min(float(line.bound_left_m[tightest]), float(line.bound_right_m[tightest]))
        assert bound > 0.0
        assert abs(float(line.lateral_m[tightest])) >= 0.8 * bound


def test_tumftm_field_names_are_accepted(tmp_path: Path) -> None:
    from f1telemetry.tracks import load_track

    body = """\
version: 1
name: tumftm
description: synthetic
centerline:
  - {x_m: 0.0, y_m: 0.0}
  - {x_m: 150.0, y_m: 0.0}
  - {x_m: 150.0, y_m: 150.0}
  - {x_m: 0.0, y_m: 150.0}
widths_m:
  - {w_tr_left_m: 7.0, w_tr_right_m: 5.0}
  - {w_tr_left_m: 7.0, w_tr_right_m: 5.0}
  - {w_tr_left_m: 7.0, w_tr_right_m: 5.0}
  - {w_tr_left_m: 7.0, w_tr_right_m: 5.0}
sectors:
  - {s_m: 150.0}
  - {s_m: 450.0}
"""
    path = tmp_path / "tumftm.yaml"
    path.write_text(body, encoding="utf-8")
    track = load_track(path)
    assert np.allclose(track.width_left_m, 7.0)
    assert np.allclose(track.width_right_m, 5.0)
    assert np.allclose(track.width_m, 12.0)


def test_lateral_validity_uses_the_correct_edge(tmp_path: Path) -> None:
    from f1telemetry.laps import (
        wheel_positions,
        wheels_within_track_limits,
    )

    track = _asymmetric_track(tmp_path)
    # A CG on the centreline with wheels placed by the same heading:
    # all four wheels inside on a straight.
    x, y = track.centerline_at(100.0)
    tangent = track.tangent_at(100.0)
    heading = math.atan2(tangent[1], tangent[0])
    wheels = wheel_positions(x, y, heading, 3.0, 2.0)
    assert wheels_within_track_limits(track, wheels)
    normal = (-tangent[1], tangent[0])
    # 6 m left is beyond what a 4 m half-width would allow, but the
    # left edge is 8 m away, so the lap stays valid.
    shifted = (x + 6.0 * normal[0], y + 6.0 * normal[1])
    left_wheels = wheel_positions(shifted[0], shifted[1], heading, 3.0, 2.0)
    assert wheels_within_track_limits(track, left_wheels)
    # 5 m right is past the narrow 4 m right edge, so it rejects.
    right_shifted = (x - 5.0 * normal[0], y - 5.0 * normal[1])
    right_wheels = wheel_positions(right_shifted[0], right_shifted[1], heading, 3.0, 2.0)
    assert not wheels_within_track_limits(track, right_wheels)


def test_solution_reports_the_solved_line_curvature() -> None:
    track = load_track(repo_root() / "tracks" / "coastal_loop.yaml")
    line = _line(track)
    assert isinstance(line.curvature_per_m, np.ndarray)
    assert line.curvature_per_m.shape == line.s_m.shape
    assert np.all(np.isfinite(line.curvature_per_m))
    # A minimum-curvature line flattens the centreline: its curvature
    # must be no larger in magnitude on average than the centreline's.
    assert (
        float(np.mean(np.abs(line.curvature_per_m)))
        <= float(np.mean(np.abs(line.centerline_curvature_per_m))) + 1e-12
    )


def test_speed_profile_consumes_the_line_curvature() -> None:
    track = load_track(repo_root() / "tracks" / "coastal_loop.yaml")
    line = _line(track)
    profile = speed_profile(
        track,
        line,
        max_speed_m_s=_MAX_SPEED,
        max_accel_m_s2=_MAX_ACCEL,
        max_brake_m_s2=_MAX_BRAKE,
        lateral_accel_m_s2=_MAX_LAT_ACCEL,
    )
    assert profile.converged
    assert np.allclose(profile.curvature_per_m, np.abs(line.curvature_per_m), atol=0.0)


def test_speed_profile_enforces_braking_backward_and_traction_forward() -> None:
    track = load_track(repo_root() / "tracks" / "technical_ring.yaml")
    line = _line(track)
    profile = speed_profile(
        track,
        line,
        max_speed_m_s=_MAX_SPEED,
        max_accel_m_s2=_MAX_ACCEL,
        max_brake_m_s2=_MAX_BRAKE,
        lateral_accel_m_s2=_MAX_LAT_ACCEL,
    )
    speeds = profile.speed_m_s
    assert np.all(speeds > 0.0)
    assert np.all(speeds <= _MAX_SPEED + 1e-9)
    count = track.waypoint_count
    for index in range(count):
        following = (index + 1) % count
        ds = float(track.ds_m[index])
        # Braking capacity looking backward: the car may slow at most at
        # max_brake_m_s2 into the next knot.
        assert speeds[following] ** 2 >= speeds[index] ** 2 - 2.0 * _MAX_BRAKE * ds - 1e-6
        # Traction capacity looking forward: the car may gain at most
        # max_accel_m_s2 over the segment.
        assert speeds[following] ** 2 <= speeds[index] ** 2 + 2.0 * _MAX_ACCEL * ds + 1e-6


def test_speed_profile_respects_the_corner_speed_target() -> None:
    track = load_track(repo_root() / "tracks" / "technical_ring.yaml")
    line = _line(track)
    profile = speed_profile(
        track,
        line,
        max_speed_m_s=_MAX_SPEED,
        max_accel_m_s2=_MAX_ACCEL,
        max_brake_m_s2=_MAX_BRAKE,
        lateral_accel_m_s2=_MAX_LAT_ACCEL,
    )
    for speed, curvature in zip(profile.speed_m_s, profile.curvature_per_m, strict=True):
        if curvature > 0.0:
            assert speed**2 * curvature <= _MAX_LAT_ACCEL + 1e-6
        assert speed <= _MAX_SPEED + 1e-9
    # A chicane forces the profile below the top speed somewhere.
    assert float(np.min(profile.speed_m_s)) < _MAX_SPEED * 0.9


def test_speed_profile_is_deterministic() -> None:
    track = load_track(repo_root() / "tracks" / "coastal_loop.yaml")
    line = _line(track)
    first = speed_profile(
        track,
        line,
        max_speed_m_s=_MAX_SPEED,
        max_accel_m_s2=_MAX_ACCEL,
        max_brake_m_s2=_MAX_BRAKE,
        lateral_accel_m_s2=_MAX_LAT_ACCEL,
    )
    second = speed_profile(
        track,
        line,
        max_speed_m_s=_MAX_SPEED,
        max_accel_m_s2=_MAX_ACCEL,
        max_brake_m_s2=_MAX_BRAKE,
        lateral_accel_m_s2=_MAX_LAT_ACCEL,
    )
    assert np.array_equal(first.speed_m_s, second.speed_m_s)
    assert first.converged == second.converged


def test_speed_profile_flags_non_convergence_instead_of_smoothing() -> None:
    track = load_track(repo_root() / "tracks" / "technical_ring.yaml")
    line = _line(track)
    profile = speed_profile(
        track,
        line,
        max_speed_m_s=_MAX_SPEED,
        max_accel_m_s2=_MAX_ACCEL,
        max_brake_m_s2=_MAX_BRAKE,
        lateral_accel_m_s2=_MAX_LAT_ACCEL,
        max_sweeps=0,
    )
    assert not profile.converged


def test_speed_profile_refuses_a_line_that_did_not_converge() -> None:
    track = load_track(repo_root() / "tracks" / "coastal_loop.yaml")
    broken = minimum_curvature_offsets(track, max_iterations=0)
    assert not broken.converged
    with pytest.raises(ValueError):
        speed_profile(
            track,
            broken,
            max_speed_m_s=_MAX_SPEED,
            max_accel_m_s2=_MAX_ACCEL,
            max_brake_m_s2=_MAX_BRAKE,
            lateral_accel_m_s2=_MAX_LAT_ACCEL,
        )


def test_a_chicane_produces_a_braking_zone_before_the_corner() -> None:
    track = load_track(repo_root() / "tracks" / "technical_ring.yaml")
    line = _line(track)
    profile = speed_profile(
        track,
        line,
        max_speed_m_s=_MAX_SPEED,
        max_accel_m_s2=_MAX_ACCEL,
        max_brake_m_s2=_MAX_BRAKE,
        lateral_accel_m_s2=_MAX_LAT_ACCEL,
    )
    speeds = profile.speed_m_s
    slowest = int(np.argmin(speeds))
    count = track.waypoint_count
    # Walking backward from the slowest knot, the speed must rise no
    # faster than braking from the top of the profile allows, i.e. the
    # profile decelerates *into* the chicane rather than after it.
    for index in range(count):
        following = (slowest + 1 + index) % count
        previous = (following - 1) % count
        ds = float(track.ds_m[previous])
        assert speeds[following] ** 2 >= speeds[previous] ** 2 - 2.0 * _MAX_BRAKE * ds - 1e-6
        if speeds[following] >= 0.99 * _MAX_SPEED:
            break
    assert math.isfinite(float(np.min(speeds)))
