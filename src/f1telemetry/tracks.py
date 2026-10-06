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
one positive width per waypoint, aligned by position - the loader checks the
count, because a shifted width list would describe a different track than
the one drawn. ``sectors`` holds the lap's sector boundaries as arc lengths
in metres, strictly increasing and strictly inside the lap length, so no
boundary sits on the start/finish line.

**The geometry.** The centerline is interpolated piecewise-linearly and
periodically: a query at arc length ``s`` finds the segment it falls in and
interpolates along it, with ``s`` wrapped modulo the lap length so negative
and beyond-the-lap values address the same point on an adjacent lap. Arc
length is measured along the polyline itself, so ``s`` is exact rather than
an approximation of some smoother curve. Per-waypoint quantities - width and
curvature - interpolate the same way, over the same knots.

The tangent is the segment's unit direction, constant within a segment, and
a query exactly on a waypoint is right-continuous: it returns the outgoing
segment's frame. The normal is the tangent rotated a quarter turn left,
``(-t_y, t_x)``; the ground frame is right-handed, so a positive lateral
offset in :meth:`Track.point_at` is to the left of the direction of travel.

Curvature needs one decision, because a polyline has none on its segment
interiors. The per-waypoint estimate is the signed turning angle at the
waypoint - the angle from the incoming segment to the outgoing one, wrapped
to ``(-pi, pi]`` - divided by the mean of the two adjacent segment lengths,
and those estimates are interpolated piecewise-linearly over ``s`` like any
other per-waypoint quantity. Positive curvature is a left (counterclockwise)
turn, the same sign convention as the normal.

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
    * ``width_m`` - the track width at each waypoint, metres, positive.
    * ``s_m`` - the arc length at each waypoint along the polyline, metres,
      strictly increasing from ``s_m[0] == 0`` to ``s_m[-1] < length_m``.
    * ``ds_m`` - the length of each segment, metres, including the closing
      segment ``ds_m[-1]`` from the last waypoint back to the first; all
      positive, and they sum to :attr:`length_m`.
    * ``tangent_x_m``, ``tangent_y_m`` - each segment's unit direction.
    * ``curvature_per_m`` - the signed per-waypoint curvature estimate, in
      1/m, positive for a left turn.

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
    s_m: np.ndarray
    ds_m: np.ndarray
    tangent_x_m: np.ndarray
    tangent_y_m: np.ndarray
    curvature_per_m: np.ndarray

    @property
    def waypoint_count(self) -> int:
        """The number of centerline waypoints, and of every per-waypoint array."""
        return int(self.x_m.shape[0])

    def centerline_at(self, s_m: float) -> tuple[float, float]:
        """The centerline point at arc length ``s_m``, in metres, periodic in ``s``.

        ``s_m`` wraps modulo :attr:`length_m`, so negative values and values
        past the lap length address the same point on an adjacent lap. The
        interpolation is piecewise-linear between waypoints; a query exactly
        on a waypoint returns that waypoint, on the segment that starts there.
        """
        self._check("s_m", s_m)
        index, fraction = self._location(s_m)
        return self._lerp_centerline(index, fraction)

    def width_at(self, s_m: float) -> float:
        """The track width at arc length ``s_m``, in metres.

        The per-waypoint widths interpolate piecewise-linearly over the same
        knots as the centerline, so a width never jumps between waypoints.
        """
        self._check("s_m", s_m)
        index, fraction = self._location(s_m)
        return self._lerp(self.width_m, index, fraction)

    def tangent_at(self, s_m: float) -> tuple[float, float]:
        """The unit tangent at ``s_m``: the direction of the segment it falls in.

        Constant within a segment, so a corner shows up only as a change of
        tangent from one segment to the next. Right-continuous at waypoints:
        a query exactly on one returns the outgoing segment's direction.
        """
        self._check("s_m", s_m)
        index, _ = self._location(s_m)
        return (float(self.tangent_x_m[index]), float(self.tangent_y_m[index]))

    def normal_at(self, s_m: float) -> tuple[float, float]:
        """The unit normal at ``s_m``: the tangent rotated a quarter turn left.

        ``(-t_y, t_x)`` in the right-handed ground frame, so the normal
        points to the left of the direction of travel, and a positive
        lateral offset in :meth:`point_at` is to the left.
        """
        self._check("s_m", s_m)
        index, _ = self._location(s_m)
        tangent_x = float(self.tangent_x_m[index])
        tangent_y = float(self.tangent_y_m[index])
        return (-tangent_y, tangent_x)

    def curvature_at(self, s_m: float) -> float:
        """The signed curvature at ``s_m``, in 1/m, positive for a left turn.

        The per-waypoint curvature estimates interpolate piecewise-linearly
        over the same knots as the centerline, so the curvature profile is
        continuous around the lap.
        """
        self._check("s_m", s_m)
        index, fraction = self._location(s_m)
        return self._lerp(self.curvature_per_m, index, fraction)

    def frame_at(self, s_m: float) -> TrackFrame:
        """Position, tangent, normal, curvature and width at ``s_m``, once.

        The one-stop frame for anything that needs the whole local geometry
        at a point: it interpolates each quantity from the same segment
        index, so the returned values cannot disagree with each other the way
        separate calls made at slightly different arc lengths could.
        """
        self._check("s_m", s_m)
        index, fraction = self._location(s_m)
        x_m, y_m = self._lerp_centerline(index, fraction)
        tangent_x = float(self.tangent_x_m[index])
        tangent_y = float(self.tangent_y_m[index])
        return TrackFrame(
            x_m=x_m,
            y_m=y_m,
            tangent_x_m=tangent_x,
            tangent_y_m=tangent_y,
            normal_x_m=-tangent_y,
            normal_y_m=tangent_x,
            curvature_per_m=self._lerp(self.curvature_per_m, index, fraction),
            width_m=self._lerp(self.width_m, index, fraction),
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
        x_m, y_m = self._lerp_centerline(index, fraction)
        normal_x = -float(self.tangent_y_m[index])
        normal_y = float(self.tangent_x_m[index])
        return (x_m + lateral_m * normal_x, y_m + lateral_m * normal_y)

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

    def _lerp(self, values: np.ndarray, index: int, fraction: float) -> float:
        """One per-waypoint quantity, piecewise-linearly interpolated over ``s``."""
        following = (index + 1) % self.waypoint_count
        start = float(values[index])
        end = float(values[following])
        return start + fraction * (end - start)

    def _lerp_centerline(self, index: int, fraction: float) -> tuple[float, float]:
        """The centerline point at ``fraction`` along the segment starting at ``index``."""
        following = (index + 1) % self.waypoint_count
        start_x = float(self.x_m[index])
        start_y = float(self.y_m[index])
        end_x = float(self.x_m[following])
        end_y = float(self.y_m[following])
        return (start_x + fraction * (end_x - start_x), start_y + fraction * (end_y - start_y))

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
    widths = [
        _positive(entry, f"{where}: widths_m[{index}]") for index, entry in enumerate(widths_raw)
    ]

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
        widths=widths,
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
    widths: Sequence[float],
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
    width = np.array(widths, dtype=np.float64)

    # The closing segment joins the last waypoint back to the first, so the
    # loop is closed by construction; `roll` indexes every segment, including
    # that one, without the first waypoint being repeated in the file.
    delta_x = np.roll(x, -1) - x
    delta_y = np.roll(y, -1) - y
    ds = np.hypot(delta_x, delta_y)
    if not bool(np.all(ds > 0.0)):
        index = int(np.argmin(ds))
        raise ContractError(
            f"{where}: centerline: waypoints {index} and {(index + 1) % ds.shape[0]} "
            f"are the same point (segment length {float(ds[index])} m); every "
            "segment must be nonzero, so the first waypoint must not be repeated "
            "at the end either - the closing segment is the join from the last "
            "waypoint back to the first"
        )

    knots = np.concatenate(([0.0], np.cumsum(ds)[:-1]))
    length_m = float(np.sum(ds))
    tangent_x = delta_x / ds
    tangent_y = delta_y / ds

    # Curvature: a polyline is straight on every segment interior, so the
    # curvature lives at the waypoints. The estimate at a waypoint is the
    # signed turning angle - the angle from the incoming segment to the
    # outgoing one, wrapped to (-pi, pi] - over the mean of the two adjacent
    # segment lengths. Positive is a left turn, matching the left normal.
    heading = np.arctan2(tangent_y, tangent_x)
    heading_periodic = np.concatenate((heading[-1:], heading))
    turning = np.diff(heading_periodic)
    turning = (turning + math.pi) % _TWO_PI - math.pi
    ds_periodic = np.concatenate((ds[-1:], ds))
    mean_ds = 0.5 * (ds_periodic[:-1] + ds_periodic[1:])
    curvature = turning / mean_ds

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
    for array in (x, y, width, knots, ds, tangent_x, tangent_y, curvature):
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
        s_m=knots,
        ds_m=ds,
        tangent_x_m=tangent_x,
        tangent_y_m=tangent_y,
        curvature_per_m=curvature,
    )


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


def _positive(node: Any, where: str) -> float:
    value = _number(node, where)
    if value <= 0.0:
        raise ContractError(f"{where}: expected a number > 0, got {node!r}")
    return value
