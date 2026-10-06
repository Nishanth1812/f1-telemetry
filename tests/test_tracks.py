"""P4-T1/T2: geometry behaviours the smooth centreline must satisfy.

The polyline centreline interpolated position and width but produced
curvature only as turning-angle deltas at the waypoints; the speed
profile and racing line consume curvature, so the centreline now has
to be genuinely smooth: continuous position, tangent and curvature
from one periodic curve through the waypoints, arc-length parameter
``s`` measured on that curve, and pinned continuity at the
start/finish seam.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from f1telemetry.contracts.channels import ContractError, repo_root
from f1telemetry.tracks import Track, load_track

_CIRCLE_WAYPOINTS = 24
_CIRCLE_RADIUS_M = 100.0


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "track.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _circle_track(tmp_path: Path) -> Track:
    points = []
    for index in range(_CIRCLE_WAYPOINTS):
        angle = 2.0 * math.pi * index / _CIRCLE_WAYPOINTS
        points.append(
            f"  - {{x_m: {_CIRCLE_RADIUS_M * math.cos(angle):.9f}, "
            f"y_m: {_CIRCLE_RADIUS_M * math.sin(angle):.9f}}}"
        )
    widths = ", ".join(["12.0"] * _CIRCLE_WAYPOINTS)
    body = (
        "version: 1\nname: circle\ndescription: synthetic\ncenterline:\n"
        + "\n".join(points)
        + f"\nwidths_m: [{widths}]\nsectors:\n  - {{s_m: 100.0}}\n  - {{s_m: 400.0}}\n"
    )
    return load_track(_write(tmp_path, body))


def test_centerline_passes_through_every_waypoint(tmp_path: Path) -> None:
    track = load_track(repo_root() / "tracks" / "coastal_loop.yaml")
    for index in range(track.waypoint_count):
        x, y = track.centerline_at(float(track.s_m[index]))
        assert math.isclose(x, float(track.x_m[index]), abs_tol=1e-9)
        assert math.isclose(y, float(track.y_m[index]), abs_tol=1e-9)


def test_tangent_is_continuous_across_waypoints(tmp_path: Path) -> None:
    track = load_track(repo_root() / "tracks" / "coastal_loop.yaml")
    for knot in track.s_m:
        before = track.tangent_at(float(knot) - 1e-3)
        after = track.tangent_at(float(knot) + 1e-3)
        assert math.isclose(before[0], after[0], abs_tol=1e-3)
        assert math.isclose(before[1], after[1], abs_tol=1e-3)


def test_curvature_is_continuous_across_waypoints_and_the_seam(tmp_path: Path) -> None:
    track = load_track(repo_root() / "tracks" / "coastal_loop.yaml")
    for knot in track.s_m:
        before = track.curvature_at(float(knot) - 1e-3)
        after = track.curvature_at(float(knot) + 1e-3)
        assert math.isclose(before, after, rel_tol=1e-2, abs_tol=1e-4)
    near_end = track.curvature_at(track.length_m - 1e-3)
    near_start = track.curvature_at(1e-3)
    assert math.isclose(near_end, near_start, rel_tol=1e-2, abs_tol=1e-4)


def test_geometry_is_periodic_in_s(tmp_path: Path) -> None:
    track = load_track(repo_root() / "tracks" / "technical_ring.yaml")
    for s in (37.5, 812.25, 1299.9):
        frame_a = track.frame_at(s)
        frame_b = track.frame_at(s + track.length_m)
        frame_c = track.frame_at(s - track.length_m)
        assert math.isclose(frame_a.x_m, frame_b.x_m, abs_tol=1e-9)
        assert math.isclose(frame_a.x_m, frame_c.x_m, abs_tol=1e-9)
        assert math.isclose(frame_a.curvature_per_m, frame_b.curvature_per_m, abs_tol=1e-12)
        assert math.isclose(frame_a.width_m, frame_c.width_m, abs_tol=1e-12)


def test_arc_length_is_measured_on_the_smooth_curve(tmp_path: Path) -> None:
    track = load_track(repo_root() / "tracks" / "coastal_loop.yaml")
    assert np.all(np.diff(track.s_m) > 0.0)
    assert math.isclose(float(np.sum(track.ds_m)), track.length_m, rel_tol=1e-9)
    # Chord lengths stay within the segment arc lengths: a straight segment's
    # arc length equals its chord, a curved one exceeds it slightly.
    points_x = track.x_m
    points_y = track.y_m
    for index in range(track.waypoint_count):
        follow = (index + 1) % track.waypoint_count
        chord = math.hypot(
            float(points_x[follow]) - float(points_x[index]),
            float(points_y[follow]) - float(points_y[index]),
        )
        assert float(track.ds_m[index]) >= 0.9 * chord


def test_curvature_tracks_a_circle(tmp_path: Path) -> None:
    track = _circle_track(tmp_path)
    expected = 1.0 / _CIRCLE_RADIUS_M
    for index in range(track.waypoint_count):
        mid = float(track.s_m[index]) + 0.5 * float(track.ds_m[index])
        curvature = track.curvature_at(mid)
        assert math.isclose(curvature, expected, rel_tol=5e-3), (mid, curvature)


def test_positive_curvature_is_a_left_turn(tmp_path: Path) -> None:
    track = _circle_track(tmp_path)
    assert track.curvature_at(25.0) > 0.0


def test_width_still_interpolates_between_waypoints(tmp_path: Path) -> None:
    track = load_track(repo_root() / "tracks" / "coastal_loop.yaml")
    first = float(track.width_m[0])
    second = float(track.width_m[1])
    mid = track.width_at(0.5 * float(track.s_m[1]))
    assert math.isclose(mid, 0.5 * (first + second), rel_tol=1e-9)


def test_validation_still_refuses_broken_files(tmp_path: Path) -> None:
    body = (
        "version: 1\nname: dup\ndescription: ''\ncenterline:\n"
        "  - {x_m: 0.0, y_m: 0.0}\n  - {x_m: 10.0, y_m: 0.0}\n"
        "  - {x_m: 10.0, y_m: 0.0}\n"
        "widths_m: [10.0, 10.0, 10.0]\nsectors:\n  - {s_m: 5.0}\n"
    )
    with pytest.raises(ContractError):
        load_track(_write(tmp_path, body))

    sectors_out_of_order = (
        "version: 1\nname: bad\ndescription: ''\ncenterline:\n"
        "  - {x_m: 0.0, y_m: 0.0}\n  - {x_m: 10.0, y_m: 0.0}\n"
        "  - {x_m: 10.0, y_m: 10.0}\n"
        "widths_m: [10.0, 10.0, 10.0]\nsectors:\n"
        "  - {s_m: 25.0}\n  - {s_m: 5.0}\n"
    )
    with pytest.raises(ContractError):
        load_track(_write(tmp_path, sectors_out_of_order))


def test_invalid_query_arguments_are_refused(tmp_path: Path) -> None:
    track = load_track(repo_root() / "tracks" / "coastal_loop.yaml")
    with pytest.raises(ValueError):
        track.centerline_at(float("nan"))
    with pytest.raises(ValueError):
        track.curvature_at(True)
    with pytest.raises(ValueError):
        track.width_at("soon")  # pyright: ignore[reportArgumentType]
