"""P4-T1/T2: the track file format and the geometry every P4 task builds on.

``PLAN.md`` section 8.5 needs a circuit before anything else in P4 can be
built: the minimum-curvature racing line (P4-T3), the forward-backward speed
profile (P4-T5), lap and sector timing (P4-T6) and the reference driver
(P4-T8) all consume the same three things - a periodic centerline, a
per-waypoint track width and ordered sector boundaries - so this module is
the single place they are defined, validated and interpolated. Nothing
downstream restates them, and no number in this module is a physical
constant: every metre comes from the track file.

**The file format.** A track file is a versioned YAML mapping::

    version: 1
    name: circuit
    description: >-
      What the file is, and where the waypoints came from.
    centerline:
      - {x_m: 0.0, y_m: 0.0}
      - {x_m: 100.0, y_m: 0.0}
    widths_m: [12.0, 12.0]
    sectors:
      - {s_m: 1500.0}
      - {s_m: 3200.0}

``centerline`` is the closed polyline through the waypoints: the last
waypoint is joined back to the first, so the loop closes by construction and
the first waypoint is never repeated at the end (a repeat would make the
closing segment zero-length, which the loader refuses). ``widths_m`` holds
one width per waypoint, aligned by position - the loader checks the
count, because a shifted width list would describe a different track than
the one drawn. Each entry is either a positive total width in metres, or a
mapping carrying the distance to each edge (``left_m`` / ``right_m``, or
the TUMFTM survey aliases ``w_tr_left_m`` / ``w_tr_right_m``), so
asymmetric tracks keep both sides instead of being averaged. ``left`` is
the side a positive lateral offset addresses. ``sectors`` holds the lap's
sector boundaries as arc lengths
in metres, strictly increasing and strictly inside the lap length, so no
boundary sits on the start/finish line.

**The geometry.** The centreline is a periodic cubic spline through
the waypoints, built in two passes. First a ``C2`` cubic interpolant
of the closed waypoint list is solved - each segment is a cubic between
adjacent waypoints whose first and second derivatives join continuously
across every knot, including the start/finish seam. Second, the curve is
reparameterised by arc length: a fixed dense quadrature of each segment
gives its length, and a query at arc length ``s`` finds the segment it
falls in, converts the remaining distance along that segment to the
spline's own parameter with the segment's length table, and evaluates
the spline there, with ``s`` wrapped modulo the lap length so negative
and beyond-the-lap values address the same point on an adjacent lap.
Only the width still interpolates per-waypoint: it is a property of the
track, not a derivative of the centreline.

Why a cubic spline rather than the raw polyline or a Catmull-Rom: the
speed profile (P4-T5) and the minimum-curvature line (P4-T3) both
consume curvature, and a polyline has none on its segment interiors, so
its curvature would be a sum of deltas at the waypoints. A Catmull-Rom
spline makes position and tangent continuous but leaves curvature
jumping at the knots, which would put false braking/turning events into
the speed profile. The periodic cubic is the smallest interpolant that
makes position, tangent *and* curvature continuous (it is ``C2``), it
interpolates every waypoint exactly, and it is deterministic: the same
fixture solves to the same coefficients and the same knots.

The tangent is the spline's derivative, normalised; a query exactly on
a waypoint is right-continuous: it returns the outgoing segment's
derivative. The normal is the tangent rotated a quarter turn left,
``(-t_y, t_x)``; the ground frame is right-handed, so a positive lateral
offset in :meth:`Track.point_at` is to the left of the direction of
travel. Curvature is the analytic spline quantity
``(x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)``, in 1/m, positive for a left
(counterclockwise) turn - the same sign convention as the normal.

**Lateral offset is unbounded.** :meth:`Track.point_at` adds ``lateral_m``
times the normal to the centerline and refuses nothing: the racing-line
solver is expected to explore offsets up to roughly half the width, and a
clamp here would turn an over-optimistic line into a silently different one.
Whether a point is inside the track is a lap-validity question (P4-T7),
asked later against :meth:`Track.width_at` - not one the geometry layer
should answer by clipping.

**Immutability.** :class:`Track` is a frozen dataclass whose arrays are
read-only, so a loaded track cannot be edited in place and every consumer
sees the file's geometry exactly as validated. A different track is a
different file and a new :class:`Track`.

**Known limitation (P4-T1).** Elevation and gradient are out of scope: the
centerline is planar, and every quantity here is two-dimensional until a
track format with elevation exists.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

import numpy as np
import yaml

from f1telemetry.contracts.channels import ContractError, repo_root

__all__ = [
    "ContractError",
    "Track",
    "TrackFrame",
    "load_track",
    "track_yaml_path",
]

# A closed loop needs at least three waypoints: two waypoints and the closing
# segment retrace the same line, so they cannot enclose anything.
_MIN_WAYPOINTS: Final[int] = 3
_CENTERLINE_KEYS: Final[frozenset[str]] = frozenset({"x_m", "y_m"})
_TWO_PI: Final[float] = 2.0 * math.pi
# Dense quadrature points per spline segment for the arc-length tables:
# fixed, so a fixture always solves to the same knots and length.
_SPLINE_SAMPLES: Final[int] = 64


@dataclass(frozen=True, slots=True)
class TrackFrame:
    """The frame at one arc length: position, heading, curvature and width.

    Everything a racing-line, speed-profile or driver consumer needs at a
    single point of the lap, in one value, so the interpolation is computed
    once per query instead of once per quantity.
    """

    x_m: float
    y_m: float
    tangent_x_m: float
    tangent_y_m: float
    normal_x_m: float
    normal_y_m: float
    curvature_per_m: float
    width_m: float
    width_left_m: float
    width_right_m: float

    @property
    def heading_rad(self) -> float:
        """The tangent's heading, counterclockwise from +x, in radians."""
        return math.atan2(self.tangent_y_m, self.tangent_x_m)


@dataclass(frozen=True, slots=True)
class Track:
    """An immutable circuit: periodic centerline, per-waypoint width, sectors.

    The arrays are read-only ``float64`` vectors of length
    :attr:`waypoint_count`, so the geometry a file describes cannot be edited
    in place:

    * ``x_m``, ``y_m`` - the waypoints in order, metres, in the right-handed
      ground frame. The loop closes from the last waypoint back to the first;
      the first waypoint is not repeated at the end.
    * ``width_m`` - the total track width at each waypoint, metres,
      positive: ``width_left_m[i] + width_right_m[i]``.
    * ``width_left_m``, ``width_right_m`` - the distance from the
      centreline to the left and right track edges at each waypoint,
      metres, positive. Both sides are kept separately because surveyed
      centrelines (TUMFTM's ``w_tr_left_m`` / ``w_tr_right_m``) place
      them asymmetrically; a positive lateral offset addresses the
      left side, and a position is on track only between
      ``-width_right_m`` and ``+width_left_m``.
    * ``s_m`` - the arc length at each waypoint along the spline, metres,
      strictly increasing from ``s_m[0] == 0`` to ``s_m[-1] < length_m``.
    * ``ds_m`` - the arc length of each spline segment, metres, including
      the closing segment ``ds_m[-1]`` from the last waypoint back to
      the first; all positive, and they sum to :attr:`length_m`.
    * ``tangent_x_m``, ``tangent_y_m`` - the unit tangent at each knot,
      from the spline's own derivative (right-continuous at the knot).
    * ``curvature_per_m`` - the signed centreline curvature at each
      knot, in 1/m, positive for a left turn.
    * ``spline_second_x``, ``spline_second_y`` - the second derivatives
      of the cubic spline at each knot; with the waypoints they fix
      the interpolant a query evaluates.
    * ``spline_table_parameter``, ``spline_table_length`` - the fixed
      dense quadrature of each segment: the spline parameter at each
      sample, and the cumulative arc length within that segment, so a
      distance along the curve can be mapped back to the parameter the
      spline is evaluated at.

    Built by :func:`load_track`, which enforces every invariant above; the
    dataclass itself is a record, not a second validation path.
    """

    name: str
    version: int
    description: str
    length_m: float
    sector_boundaries_m: tuple[float, ...]
    x_m: np.ndarray
    y_m: np.ndarray
    width_m: np.ndarray
    width_left_m: np.ndarray
    width_right_m: np.ndarray
    s_m: np.ndarray
    ds_m: np.ndarray
    tangent_x_m: np.ndarray
    tangent_y_m: np.ndarray
    curvature_per_m: np.ndarray
    spline_second_x: np.ndarray
    spline_second_y: np.ndarray
    spline_table_parameter: np.ndarray
    spline_table_length: np.ndarray

    @property
    def waypoint_count(self) -> int:
        """The number of centerline waypoints, and of every per-waypoint array."""
        return int(self.x_m.shape[0])

    def centerline_at(self, s_m: float) -> tuple[float, float]:
        """The centerline point at arc length ``s_m``, in metres, periodic in ``s``.

        ``s_m`` wraps modulo :attr:`length_m`, so negative values and values
        past the lap length address the same point on an adjacent lap. The
        spline is evaluated at the matched point on its own parameter, and
        a query exactly on a waypoint returns that waypoint, on the
        segment that starts there.
        """
        self._check("s_m", s_m)
        index, fraction = self._location(s_m)
        x_m, y_m, _, _, _ = self._spline_eval(index, fraction * float(self.ds_m[index]))
        return (x_m, y_m)

    def width_at(self, s_m: float) -> float:
        """The total track width at arc length ``s_m``, in metres.

        The per-waypoint widths interpolate piecewise-linearly over the
        same knots as the centreline, so a width never jumps between
        waypoints. Consumers that need the side-specific extent should
        use :meth:`width_left_at` and :meth:`width_right_at`.
        """
        self._check("s_m", s_m)
        index, fraction = self._location(s_m)
        return self._lerp(self.width_m, index, fraction)

    def width_left_at(self, s_m: float) -> float:
        """The distance from the centreline to the left edge at ``s_m``, metres."""
        self._check("s_m", s_m)
        index, fraction = self._location(s_m)
        return self._lerp(self.width_left_m, index, fraction)

    def width_right_at(self, s_m: float) -> float:
        """The distance from the centreline to the right edge at ``s_m``, metres."""
        self._check("s_m", s_m)
        index, fraction = self._location(s_m)
        return self._lerp(self.width_right_m, index, fraction)

    def tangent_at(self, s_m: float) -> tuple[float, float]:
        """The unit tangent at ``s_m``: the direction of the segment it falls in.

        Constant within a segment, so a corner shows up only as a change of
        tangent from one segment to the next. Right-continuous at waypoints:
        a query exactly on one returns the outgoing segment's direction.
        """
        self._check("s_m", s_m)
        index, fraction = self._location(s_m)
        _, _, tangent_x, tangent_y, _ = self._spline_eval(index, fraction * float(self.ds_m[index]))
        return (tangent_x, tangent_y)

    def normal_at(self, s_m: float) -> tuple[float, float]:
        """The unit normal at ``s_m``: the tangent rotated a quarter turn left.

        ``(-t_y, t_x)`` in the right-handed ground frame, so the normal
        points to the left of the direction of travel, and a positive
        lateral offset in :meth:`point_at` is to the left.
        """
        self._check("s_m", s_m)
        tangent_x, tangent_y = self.tangent_at(s_m)
        return (-tangent_y, tangent_x)

    def curvature_at(self, s_m: float) -> float:
        """The signed curvature at ``s_m``, in 1/m, positive for a left turn.

        The spline is ``C2``, so the curvature is continuous around the
        lap, including across the start/finish seam; it is evaluated
        analytically from the spline's derivatives, not interpolated
        from per-waypoint estimates.
        """
        self._check("s_m", s_m)
        index, fraction = self._location(s_m)
        _, _, _, _, curvature = self._spline_eval(index, fraction * float(self.ds_m[index]))
        return curvature

    def frame_at(self, s_m: float) -> TrackFrame:
        """Position, tangent, normal, curvature and width at ``s_m``, once.

        The one-stop frame for anything that needs the whole local geometry
        at a point: it interpolates each quantity from the same segment
        index, so the returned values cannot disagree with each other the way
        separate calls made at slightly different arc lengths could.
        """
        self._check("s_m", s_m)
        index, fraction = self._location(s_m)
        x_m, y_m, tangent_x, tangent_y, curvature = self._spline_eval(
            index, fraction * float(self.ds_m[index])
        )
        return TrackFrame(
            x_m=x_m,
            y_m=y_m,
            tangent_x_m=tangent_x,
            tangent_y_m=tangent_y,
            normal_x_m=-tangent_y,
            normal_y_m=tangent_x,
            curvature_per_m=curvature,
            width_m=self._lerp(self.width_m, index, fraction),
            width_left_m=self._lerp(self.width_left_m, index, fraction),
            width_right_m=self._lerp(self.width_right_m, index, fraction),
        )

    def point_at(self, s_m: float, lateral_m: float = 0.0) -> tuple[float, float]:
        """The point at arc length ``s_m``, offset ``lateral_m`` to the left.

        The centerline point plus ``lateral_m`` times the unit normal, in
        metres. ``lateral_m`` is unbounded and unclamped by design: the
        racing-line solver is expected to explore offsets up to roughly half
        the width, and a clamp here would turn an over-optimistic line into a
        silently different one. Track limits are a lap-validity question
        (P4-T7) asked against :meth:`width_at`, not a geometry one.
        """
        self._check("s_m", s_m)
        self._check("lateral_m", lateral_m)
        index, fraction = self._location(s_m)
        x_m, y_m, tangent_x, tangent_y, _ = self._spline_eval(
            index, fraction * float(self.ds_m[index])
        )
        return (x_m + lateral_m * (-tangent_y), y_m + lateral_m * tangent_x)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready view of the track, for inspection and transport."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "length_m": self.length_m,
            "waypoint_count": self.waypoint_count,
            "centerline": [[float(x), float(y)] for x, y in zip(self.x_m, self.y_m, strict=True)],
            "widths_m": [float(width) for width in self.width_m],
            "width_left_m": [float(width) for width in self.width_left_m],
            "width_right_m": [float(width) for width in self.width_right_m],
            "sector_boundaries_m": list(self.sector_boundaries_m),
        }

    def _location(self, s_m: float) -> tuple[int, float]:
        """The segment index and the fraction along it for one periodic arc length.

        The arc length is wrapped modulo the lap length first, then located
        with a binary search over the strictly increasing knot vector, so the
        lookup costs ``O(log n)`` however long the waypoint list is. A wrapped
        value of exactly zero addresses segment zero at fraction zero.
        """
        wrapped = s_m % self.length_m
        index = int(np.searchsorted(self.s_m, wrapped, side="right")) - 1
        fraction = (wrapped - float(self.s_m[index])) / float(self.ds_m[index])
        return index, fraction

    def _spline_eval(
        self, index: int, distance_m: float
    ) -> tuple[float, float, float, float, float]:
        """The spline's (x, y, unit tangent, signed curvature) at one segment.

        ``distance_m`` is the arc length from the start of segment
        ``index`` along the curve, in metres; the segment's dense length
        table maps it back to the spline's own parameter, and the cubic,
        its first and its second derivative are evaluated there. The
        tangent is normalised and the curvature is the analytic
        ``(x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)`` of that derivative, so
        the returned tuple is consistent with the centreline by
        construction.
        """
        table_u = self.spline_table_parameter[index]
        table_l = self.spline_table_length[index]
        # The dense length table is monotone by construction, so a
        # binary search finds the span; degenerate (stationary) spans
        # divide by zero only for a self-intersecting fixture, which the
        # loader's duplicate-waypoint check rejects upstream.
        span_index = int(np.searchsorted(table_l, distance_m, side="right")) - 1
        span_index = min(max(span_index, 0), table_l.shape[0] - 2)
        span = float(table_l[span_index + 1]) - float(table_l[span_index])
        fraction = 0.0 if span <= 0.0 else (distance_m - float(table_l[span_index])) / span
        u = float(table_u[span_index]) + fraction * (
            float(table_u[span_index + 1]) - float(table_u[span_index])
        )
        return self._spline_eval_u(index, u)

    def centerline_polyline(
        self, subdivisions: int = 16
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A dense deterministic sampling of the smooth centreline.

        Returns ``(x_m, y_m, s_m)``: the spline evaluated at
        ``subdivisions`` equal parameter steps per segment, the
        corresponding arc lengths interpolated from the integration
        table, and the closing segment's far end identified with the
        lap length so the polyline closes. Projection consumers use this
        rather than the waypoint polyline, which the spline
        deliberately no longer follows exactly between knots.
        """
        if isinstance(subdivisions, bool) or not isinstance(subdivisions, int) or subdivisions < 1:
            raise ValueError(f"subdivisions must be an int >= 1, got {subdivisions!r}")
        xs: list[float] = []
        ys: list[float] = []
        arc_s: list[float] = []
        for index in range(self.waypoint_count):
            for j in range(subdivisions):
                u = j / subdivisions
                x, y, _, _, _ = self._spline_eval_u(index, u)
                xs.append(x)
                ys.append(y)
                arc_s.append(
                    float(self.s_m[index])
                    + float(
                        np.interp(
                            u, self.spline_table_parameter[index], self.spline_table_length[index]
                        )
                    )
                )
        x_arr = np.asarray(xs, dtype=np.float64)
        y_arr = np.asarray(ys, dtype=np.float64)
        return (x_arr, y_arr, np.asarray(arc_s, dtype=np.float64))

    def _spline_eval_u(self, index: int, u: float) -> tuple[float, float, float, float, float]:
        """The spline quantities at segment ``index``'s own parameter ``u``."""
        following = (index + 1) % self.waypoint_count
        x0 = float(self.x_m[index])
        y0 = float(self.y_m[index])
        x1 = float(self.x_m[following])
        y1 = float(self.y_m[following])
        m0x = float(self.spline_second_x[index])
        m1x = float(self.spline_second_x[following])
        m0y = float(self.spline_second_y[index])
        m1y = float(self.spline_second_y[following])
        x, y = _spline_point(x0, y0, x1, y1, m0x, m1x, m0y, m1y, u)
        dx, dy, d2x, d2y = _spline_derivative(x0, y0, x1, y1, m0x, m1x, m0y, m1y, u)
        speed = math.hypot(dx, dy)
        curvature = (dx * d2y - dy * d2x) / (speed * speed * speed)
        return (x, y, dx / speed, dy / speed, curvature)

    def _lerp(self, values: np.ndarray, index: int, fraction: float) -> float:
        """One per-waypoint quantity, piecewise-linearly interpolated over ``s``."""
        following = (index + 1) % self.waypoint_count
        start = float(values[index])
        end = float(values[following])
        return start + fraction * (end - start)

    def _check(self, label: str, value: object) -> float:
        """Narrow one query argument to a finite ``float``, or fail before interpolating.

        ``bool`` is refused although it is an ``int``: a boolean arc length is
        a caller bug, not a number. This is the Python boundary's check - a
        compiled consumer calling the arithmetic directly gets nonfinite
        results instead of an exception, the same split the physics modules
        make between validating boundaries and kernels.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"track {self.name!r}: {label} must be a real number, got {value!r}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"track {self.name!r}: {label} must be finite, got {value!r}")
        return number


def track_yaml_path() -> Path:
    """The conventional track file location: ``tracks/circuit.yaml`` at the repo root.

    ``PLAN.md`` section 12 keeps circuits under ``tracks/*.yaml``; P4-T9
    supplies the first two. Until then the default names the file a circuit
    would live in, and loading it fails with the missing-file error below
    rather than with a confusing path bug.
    """
    return repo_root() / "tracks" / "coastal_loop.yaml"


def load_track(path: Path | None = None) -> Track:
    """Load, validate and arm one track file (P4-T1).

    The file is the only source of track data: every waypoint, width and
    sector boundary is read from it, and nothing in this module supplies a
    number of its own. The schema is validated field by field - types,
    counts, signs, ordering - so a bad edit raises :class:`ContractError`
    here, at the boundary, instead of surfacing later as a NaN or a division
    by zero inside a racing-line or speed-profile kernel.
    """
    source = track_yaml_path() if path is None else Path(path)
    if not source.is_file():
        raise ContractError(f"track file not found: {source}")
    where = str(source)
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _require_mapping(document, where)

    version = root.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ContractError(f"{where}: version: expected an int >= 1, got {version!r}")
    name = root.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ContractError(f"{where}: name: expected a non-empty string, got {name!r}")
    description = str(root.get("description", ""))

    centerline = _require_list(root.get("centerline"), f"{where}: centerline")
    if len(centerline) < _MIN_WAYPOINTS:
        raise ContractError(
            f"{where}: centerline: expected at least {_MIN_WAYPOINTS} waypoints, got "
            f"{len(centerline)}; two waypoints and the closing segment retrace the "
            "same line, so they cannot form a loop"
        )
    xs: list[float] = []
    ys: list[float] = []
    for index, entry in enumerate(centerline):
        entry_where = f"{where}: centerline[{index}]"
        mapping = _require_mapping(entry, entry_where)
        missing = _CENTERLINE_KEYS - set(mapping)
        if missing:
            raise ContractError(f"{entry_where}: missing key(s) {sorted(missing)}")
        xs.append(_number(mapping["x_m"], f"{entry_where}.x_m"))
        ys.append(_number(mapping["y_m"], f"{entry_where}.y_m"))

    widths_raw = _require_list(root.get("widths_m"), f"{where}: widths_m")
    if len(widths_raw) != len(centerline):
        raise ContractError(
            f"{where}: widths_m: expected exactly one width per centerline waypoint "
            f"({len(centerline)}), got {len(widths_raw)}; widths and waypoints are "
            "aligned by position, so a shifted list would describe a different "
            "track than the one drawn"
        )
    widths_left: list[float] = []
    widths_right: list[float] = []
    for index, entry in enumerate(widths_raw):
        left, right = _parse_width_entry(entry, f"{where}: widths_m[{index}]")
        widths_left.append(left)
        widths_right.append(right)

    sectors = _require_list(root.get("sectors"), f"{where}: sectors")
    if not sectors:
        raise ContractError(
            f"{where}: sectors: expected a non-empty list of {{s_m: ...}} sector boundaries"
        )
    boundaries: list[float] = []
    for index, entry in enumerate(sectors):
        entry_where = f"{where}: sectors[{index}]"
        mapping = _require_mapping(entry, entry_where)
        if "s_m" not in mapping:
            raise ContractError(
                f"{entry_where}: missing key 's_m', got {sorted(map(str, mapping))}"
            )
        boundaries.append(_number(mapping["s_m"], f"{entry_where}.s_m"))

    return _build_track(
        name=name,
        version=version,
        description=description,
        centerline_xs=xs,
        centerline_ys=ys,
        widths_left=widths_left,
        widths_right=widths_right,
        sector_boundaries=boundaries,
        where=where,
    )


def _build_track(
    *,
    name: str,
    version: int,
    description: str,
    centerline_xs: Sequence[float],
    centerline_ys: Sequence[float],
    widths_left: Sequence[float],
    widths_right: Sequence[float],
    sector_boundaries: Sequence[float],
    where: str,
) -> Track:
    """Validate parsed track data and arm it as an immutable :class:`Track`.

    Everything checked here is a check a compiled consumer cannot make: a
    zero-length segment, which would divide by zero when the tangent is
    normalised; an unordered sector list; a boundary on the start/finish
    line. The waypoints, widths and boundaries are already finite numbers by
    the time they arrive - :func:`load_track` saw to that - so the only
    arithmetic that can fail here is the one that must.
    """
    x = np.array(centerline_xs, dtype=np.float64)
    y = np.array(centerline_ys, dtype=np.float64)
    width_left = np.array(widths_left, dtype=np.float64)
    width_right = np.array(widths_right, dtype=np.float64)
    width = width_left + width_right

    # The closing segment joins the last waypoint back to the first, so the
    # loop is closed by construction; `roll` indexes every segment, including
    # that one, without the first waypoint being repeated in the file.
    delta_x = np.roll(x, -1) - x
    delta_y = np.roll(y, -1) - y
    chords = np.hypot(delta_x, delta_y)
    if not bool(np.all(chords > 0.0)):
        index = int(np.argmin(chords))
        raise ContractError(
            f"{where}: centerline: waypoints {index} and {(index + 1) % chords.shape[0]} "
            f"are the same point (segment length {float(chords[index])} m); every "
            "segment must be nonzero, so the first waypoint must not be repeated "
            "at the end either - the closing segment is the join from the last "
            "waypoint back to the first"
        )

    # The periodic cubic spline through the waypoints: C2 across every knot,
    # including the start/finish seam. Solving the second-derivative system
    # fixes the interpolant, and the dense per-segment length tables turn
    # arc-length queries into parameter evaluations.
    spline_second_x, spline_second_y = _periodic_spline_second(x, y)
    count = x.shape[0]
    table_parameter = np.zeros((count, _SPLINE_SAMPLES + 1), dtype=np.float64)
    table_length = np.zeros((count, _SPLINE_SAMPLES + 1), dtype=np.float64)
    ds = np.zeros(count, dtype=np.float64)
    for index in range(count):
        points: list[tuple[float, float]] = []
        for step in range(_SPLINE_SAMPLES + 1):
            u = step / _SPLINE_SAMPLES
            table_parameter[index, step] = u
            x_at, y_at = _spline_point(
                float(x[index]),
                float(y[index]),
                float(x[(index + 1) % count]),
                float(y[(index + 1) % count]),
                float(spline_second_x[index]),
                float(spline_second_x[(index + 1) % count]),
                float(spline_second_y[index]),
                float(spline_second_y[(index + 1) % count]),
                u,
            )
            points.append((x_at, y_at))
        for step in range(1, _SPLINE_SAMPLES + 1):
            px, py = points[step - 1]
            qx, qy = points[step]
            table_length[index, step] = table_length[index, step - 1] + math.hypot(qx - px, qy - py)
        ds[index] = table_length[index, _SPLINE_SAMPLES]

    knots = np.concatenate(([0.0], np.cumsum(ds)[:-1]))
    length_m = float(np.sum(ds))

    tangent_x = np.zeros(count, dtype=np.float64)
    tangent_y = np.zeros(count, dtype=np.float64)
    curvature = np.zeros(count, dtype=np.float64)
    for index in range(count):
        # The knot value is the outgoing segment's derivative at u=0, so a
        # query exactly on a waypoint is right-continuous, as documented.
        dx, dy, d2x, d2y = _spline_derivative(
            float(x[index]),
            float(y[index]),
            float(x[(index + 1) % count]),
            float(y[(index + 1) % count]),
            float(spline_second_x[index]),
            float(spline_second_x[(index + 1) % count]),
            float(spline_second_y[index]),
            float(spline_second_y[(index + 1) % count]),
            0.0,
        )
        speed = math.hypot(dx, dy)
        tangent_x[index] = dx / speed
        tangent_y[index] = dy / speed
        curvature[index] = (dx * d2y - dy * d2x) / (speed * speed * speed)

    if any(boundary <= previous for previous, boundary in pairwise(sector_boundaries)):
        raise ContractError(
            f"{where}: sectors: sector boundaries s_m must be strictly increasing, "
            f"got {list(sector_boundaries)}"
        )
    for boundary in sector_boundaries:
        if not 0.0 < boundary < length_m:
            raise ContractError(
                f"{where}: sectors: sector boundary s_m {boundary} m must lie "
                f"strictly inside the lap length (0, {length_m}) m; a boundary on "
                "the start/finish line would open a sector of zero length"
            )

    # Read-only from here on: a Track is the file's geometry, frozen at load.
    for array in (
        x,
        y,
        width,
        width_left,
        width_right,
        knots,
        ds,
        tangent_x,
        tangent_y,
        curvature,
        spline_second_x,
        spline_second_y,
        table_parameter,
        table_length,
    ):
        array.setflags(write=False)

    return Track(
        name=name,
        version=version,
        description=description,
        length_m=length_m,
        sector_boundaries_m=tuple(sector_boundaries),
        x_m=x,
        y_m=y,
        width_m=width,
        width_left_m=width_left,
        width_right_m=width_right,
        s_m=knots,
        ds_m=ds,
        tangent_x_m=tangent_x,
        tangent_y_m=tangent_y,
        curvature_per_m=curvature,
        spline_second_x=spline_second_x,
        spline_second_y=spline_second_y,
        spline_table_parameter=table_parameter,
        spline_table_length=table_length,
    )


def _periodic_spline_second(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The second derivatives of the closed cubic interpolant at the knots.

    One standard system per coordinate: ``M[i-1] + 4 M[i] + M[i+1] =
    6 (p[i-1] - 2 p[i] + p[i+1])`` under a uniform per-segment
    parameter, with the ring closing across the seam. Solving it gives
    the C2 interpolant through every waypoint; the uniform parameter
    is reparameterised by arc length when queries are made.
    """
    count = int(x.shape[0])
    system = np.zeros((count, count), dtype=np.float64)
    for index in range(count):
        system[index, (index - 1) % count] = 1.0
        system[index, index] = 4.0
        system[index, (index + 1) % count] = 1.0
    rhs_x = np.array(
        [6.0 * (x[(i - 1) % count] - 2.0 * x[i] + x[(i + 1) % count]) for i in range(count)],
        dtype=np.float64,
    )
    rhs_y = np.array(
        [6.0 * (y[(i - 1) % count] - 2.0 * y[i] + y[(i + 1) % count]) for i in range(count)],
        dtype=np.float64,
    )
    return (np.linalg.solve(system, rhs_x), np.linalg.solve(system, rhs_y))


def _spline_point(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    m0x: float,
    m1x: float,
    m0y: float,
    m1y: float,
    u: float,
) -> tuple[float, float]:
    """The cubic segment's position at parameter ``u`` in ``[0, 1]``."""
    one = 1.0 - u
    x = m0x * (one**3) / 6.0 + m1x * (u**3) / 6.0 + (x0 - m0x / 6.0) * one + (x1 - m1x / 6.0) * u
    y = m0y * (one**3) / 6.0 + m1y * (u**3) / 6.0 + (y0 - m0y / 6.0) * one + (y1 - m1y / 6.0) * u
    return (x, y)


def _spline_derivative(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    m0x: float,
    m1x: float,
    m0y: float,
    m1y: float,
    u: float,
) -> tuple[float, float, float, float]:
    """The cubic segment's first and second derivative at ``u``."""
    one = 1.0 - u
    dx = -m0x * (one**2) / 2.0 + m1x * (u**2) / 2.0 + (x1 - x0) - (m1x - m0x) / 6.0
    dy = -m0y * (one**2) / 2.0 + m1y * (u**2) / 2.0 + (y1 - y0) - (m1y - m0y) / 6.0
    d2x = m0x * one + m1x * u
    d2y = m0y * one + m1y * u
    return (dx, dy, d2x, d2y)


def _require_mapping(node: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(node, Mapping):
        raise ContractError(f"{where}: expected a mapping, got {node!r}")
    return node


def _require_list(node: Any, where: str) -> Sequence[Any]:
    if not isinstance(node, Sequence) or isinstance(node, (str, bytes)):
        raise ContractError(f"{where}: expected a list, got {node!r}")
    return node


def _number(node: Any, where: str) -> float:
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        raise ContractError(f"{where}: expected a number, got {node!r}")
    value = float(node)
    if not math.isfinite(value):
        raise ContractError(f"{where}: expected a finite number, got {node!r}")
    return value


def _parse_width_entry(entry: Any, where: str) -> tuple[float, float]:
    """One waypoint's width on each side of the centreline, in metres.

    Two forms are accepted. A plain positive number is the symmetric
    total width and splits in half. A mapping carries the distances
    from the centreline to each edge directly, so asymmetric surveyed
    data survives the schema: the keys ``left_m`` / ``right_m`` are
    used, and the TUMFTM survey naming ``w_tr_left_m`` / ``w_tr_right_m``
    is accepted as an alias for the same quantities. ``left`` is the
    side a positive lateral offset addresses, matching
    :meth:`Track.point_at`'s sign convention.
    """
    if isinstance(entry, Mapping):
        left = entry.get("left_m", entry.get("w_tr_left_m"))
        right = entry.get("right_m", entry.get("w_tr_right_m"))
        if left is None or right is None:
            raise ContractError(
                f"{where}: expected 'left_m' and 'right_m' (or the 'w_tr_left_m'/"
                f"'w_tr_right_m' aliases), got keys {sorted(map(str, entry))}"
            )
        return (_positive(left, f"{where}.left_m"), _positive(right, f"{where}.right_m"))
    total = _positive(entry, where)
    half = total / 2.0
    return (half, half)


def _positive(node: Any, where: str) -> float:
    value = _number(node, where)
    if value <= 0.0:
        raise ContractError(f"{where}: expected a number > 0, got {node!r}")
    return value
