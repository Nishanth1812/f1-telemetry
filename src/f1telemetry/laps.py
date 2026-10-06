"""P4-T6/P4-T7: progress projection, crossing events and lap validity.

``PLAN.md`` section 8.5 item 6 needs three things once a track file
exists: where on the lap a position is, when the car crossed each
sector boundary and the start/finish line, and whether a recorded
lap counts as a valid measurement. This module answers all three
from :class:`~f1telemetry.tracks.Track` alone - every metre, every
width and every boundary comes from the track the caller hands in,
and no number here is a physical constant. It is also the answer to
the question :mod:`~f1telemetry.tracks` deliberately refuses to
answer: lateral offset is unbounded in
:func:`~f1telemetry.tracks.Track.point_at` because clamping there
would hide an over-optimistic racing line, so whether a position is
inside the track is decided here, against the local width
:func:`~f1telemetry.tracks.Track.frame_at` reports, and nowhere
else.

**Progress.** :func:`project_position` inverts
:func:`~f1telemetry.tracks.Track.point_at`: a world position is
projected onto the closed centerline polyline by searching every
segment - the closing one included, so the start/finish seam is no
different from any other join - for its nearest point. The
projection carries the arc length of that point wrapped into
``[0, length_m)`` so it addresses the same knot on every lap, the
signed lateral offset along the nearest segment's normal (positive
to the left of the direction of travel, the sign
:func:`~f1telemetry.tracks.Track.point_at` uses), and the Euclidean
distance to the polyline. The distance equals the lateral's
magnitude only where the perpendicular foot lands inside a segment;
wherever the nearest point is a clamped segment end, the lateral is
measured along that segment's normal while the true nearest point is
the shared waypoint. An exact tie between two segments resolves to
the later one - the outgoing segment at a shared waypoint, the
right-continuous convention :meth:`~f1telemetry.tracks.Track.frame_at`
uses - so a position exactly on a waypoint projects
deterministically. The search is linear in the waypoint count,
which a track file keeps small, so no spatial index exists to drift
out of sync with the polyline it indexes.

**Events.** :func:`crossing_events` turns a stream of timed samples
into :class:`TrackEvent` records. Per-sample progress is unwrapped
across the start/finish line - a drop of at least half a lap
between consecutive samples counts a forward crossing, a rise of
more than half a lap a backward one, and a jump of exactly half a
lap resolves forward - and only *forward* crossings
emit events: a sector time or a lap time is measured on forward
progress, and a car reversing over a boundary is a driver error the
``invalid`` flag exists to record, not an event to time. Each
event's timestamp is simulated: it is the bracketing samples'
simulation times interpolated linearly through the crossing, so a
boundary reached exactly at a sample takes that sample's time, and
a stream that begins exactly on a boundary emits no event for it,
there being no earlier sample to cross from. The unwrap is
unambiguous only while consecutive samples are at most half a lap
apart, a jump of exactly half a lap resolving forward; that is a
contract on the caller's sampling, stated here
rather than enforced, because what a larger jump *means* depends on
intent no module can infer.

**Validity.** :func:`evaluate_lap_validity` decides whether a
recorded lap is a measurement or a reject, and requires all of: a
*completed lap* - the sample stream contains a forward start/finish
crossing, which :func:`crossing_events` reports as a
:attr:`TrackEventKind.LAP` event - all four wheels within the local
half-width at *every* recorded sample, and neither the DNF nor the
invalid flag set. The DNF and invalid flags are inputs the caller
owns (a car that stopped, a cut corner the scenario author wants
excluded); this module reads them and never sets them. An empty
stream completes nothing and is therefore invalid, and
:func:`assess_lap` composes all of the above into one
:class:`LapAssessment`.

**Wheels.** :func:`wheel_positions` places the four contact patches
from the CG position, the heading and the car's wheel geometry -
wheel base and per-axle track, both caller-owned because deriving
them from ``car_spec.yaml`` is the caller's job - in the fixed
``FL, FR, RL, RR`` order :mod:`~f1telemetry.contracts.channels`
fixes. The body frame is the physics core's: ``x`` forward, ``y``
left, right-handed, so each wheel's lateral offset rides the
heading's left normal ``(-sin(heading), cos(heading))``.

**Determinism.** Every function is a pure function of its
arguments: no clock, no randomness, no state, nothing read from
configuration, so the same inputs give the same bits and a repeated
run is byte-identical. This is a boundary module, not a compiled
kernel, so bad arguments - a nonfinite coordinate, a non-positive
wheel base, a decreasing simulation time - raise :class:`ValueError`
here, at the boundary, instead of surfacing later as a NaN inside a
timing comparison.

**Known limitation.** Everything here observes the car at the
recorded instants only. A wheel that leaves the track between two
samples is not seen, and a boundary crossed between two samples is
interpolated as a straight line in progress; sample density is the
caller's, and the timing inherits it. Elevation is out of scope for
the same reason it is in :mod:`~f1telemetry.tracks`: the centerline
is planar, so "within limits" is a two-dimensional question until a
track format with elevation exists.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from f1telemetry.tracks import Track

__all__ = [
    "LapAssessment",
    "LapValidity",
    "TrackEvent",
    "TrackEventKind",
    "TrackProjection",
    "TrackSample",
    "WheelPositions",
    "assess_lap",
    "crossing_events",
    "evaluate_lap_validity",
    "project_position",
    "wheel_positions",
    "wheels_within_track_limits",
]

# A wrap in the wrapped progress counts as a start/finish crossing
# only when the jump exceeds half a lap; a smaller jump is ordinary
# driving (or ordinary reversing) within one lap and must not move
# the lap counter.
_WRAP_FRACTION: Final[float] = 0.5

# The reasons a lap can fail, in the fixed order LapValidity.failures
# lists them: the flags the caller declares first, then the two
# conditions this module measures.
_FAILURE_DNF: Final[str] = "dnf"
_FAILURE_INVALID: Final[str] = "invalid"
_FAILURE_INCOMPLETE_LAP: Final[str] = "incomplete_lap"
_FAILURE_OFF_TRACK: Final[str] = "off_track"


class TrackEventKind(enum.Enum):
    """What a :class:`TrackEvent` marks: a sector boundary or the lap line.

    The members are a plain :class:`~enum.Enum` with string values
    rather than an :class:`~enum.IntEnum` because, unlike the
    gearbox's request codes, no compiled kernel consumes them: they
    exist so an event can be labelled, compared and serialised
    without bare strings drifting around a caller.
    """

    SECTOR = "sector"
    LAP = "lap"


@dataclass(frozen=True, slots=True)
class TrackSample:
    """One recorded car position on the simulation clock.

    ``time_s`` is the simulation time in seconds - the clock the
    scenario steps, never the wall clock, so a repeated run replays
    the same timestamps. ``x_m``/``y_m`` are the CG position in the
    ground frame, metres, and ``heading_rad`` its heading in radians
    counterclockwise from ``+x``, the frame
    :mod:`~f1telemetry.physics.kinematics` integrates in. The
    heading defaults to zero so a position-only stream still
    projects and times; it only takes effect where wheels are
    placed, in :func:`assess_lap`.
    """

    time_s: float
    x_m: float
    y_m: float
    heading_rad: float = 0.0


@dataclass(frozen=True, slots=True)
class TrackProjection:
    """A world position's place on the closed centerline.

    ``s_m`` is the arc length of the nearest polyline point, wrapped
    into ``[0, length_m)`` so it addresses the same knot on every
    lap; ``lateral_m`` is the signed offset along the nearest
    segment's normal in metres, positive to the left of the
    direction of travel; ``distance_m`` is the Euclidean distance to
    the polyline in metres, which is ``abs(lateral_m)`` only where
    the perpendicular foot lands inside a segment and larger
    wherever the nearest point is a clamped segment end.
    """

    s_m: float
    lateral_m: float
    distance_m: float


@dataclass(frozen=True, slots=True)
class WheelPositions:
    """The four contact patches, in the fixed ``FL, FR, RL, RR`` order.

    Each field pair is one wheel's world position in metres, front
    axle first and left before right - the order
    :mod:`~f1telemetry.contracts.channels` fixes, so a caller never
    holds a bare index without a name for it.
    """

    fl_x_m: float
    fl_y_m: float
    fr_x_m: float
    fr_y_m: float
    rl_x_m: float
    rl_y_m: float
    rr_x_m: float
    rr_y_m: float

    def positions(self) -> tuple[tuple[float, float], ...]:
        """The four wheel positions as ``(x_m, y_m)`` pairs, ``FL, FR, RL, RR``."""
        return (
            (self.fl_x_m, self.fl_y_m),
            (self.fr_x_m, self.fr_y_m),
            (self.rl_x_m, self.rl_y_m),
            (self.rr_x_m, self.rr_y_m),
        )


@dataclass(frozen=True, slots=True)
class TrackEvent:
    """One sector or lap boundary crossing, with its simulated timestamp.

    ``time_s`` is the crossing's simulated time in seconds: the
    bracketing samples' simulation times interpolated linearly
    through the crossing, so it is exact for progress that is linear
    between samples and never touches the wall clock. ``s_m`` is the
    boundary's arc length wrapped into ``[0, length_m)`` - ``0.0``
    for the start/finish line. ``index`` names what the event
    completes: for :attr:`TrackEventKind.SECTOR` the 1-based
    position of the boundary in the track file's ``sectors`` list
    (sector 1 ends at the first boundary, and so on), for
    :attr:`TrackEventKind.LAP` the 1-based count of forward
    start/finish crossings the stream has produced, which is the
    number of the lap the event completes whenever the stream begins
    at the line.
    """

    time_s: float
    s_m: float
    kind: TrackEventKind
    index: int


@dataclass(frozen=True, slots=True)
class LapValidity:
    """Whether one recorded lap is a measurement or a reject (P4-T7).

    ``valid`` is the conjunction the task names: the lap was
    completed, all four wheels stayed within the local half-width,
    and neither the DNF nor the invalid flag was set. ``failures``
    lists every condition that did not hold, in the fixed order
    ``"dnf"``, ``"invalid"``, ``"incomplete_lap"``, ``"off_track"``,
    so a caller can log why a lap was rejected without re-deriving
    it; an empty tuple means the lap is valid.
    """

    valid: bool
    completed_lap: bool
    wheels_within_limits: bool
    dnf: bool
    invalid: bool
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LapAssessment:
    """The timing and the verdict for one stream of samples.

    ``events`` are the stream's sector and lap crossings, ordered by
    simulated time; ``validity`` is the verdict over the whole
    stream; ``progress_m`` is each sample's unwrapped arc length in
    metres, so ``progress_m[i] - progress_m[0]`` is the distance the
    car travelled along the direction of travel from the first
    sample (negative where it reversed across the line); and
    ``projections`` is each sample's projection, for a caller that
    wants to find where a lap went off track.
    """

    events: tuple[TrackEvent, ...]
    validity: LapValidity
    progress_m: tuple[float, ...]
    projections: tuple[TrackProjection, ...]


def project_position(track: Track, x_m: float, y_m: float) -> TrackProjection:
    """The nearest point of the closed centerline polyline to one position.

    Every segment competes, the closing one included, so the
    start/finish seam is found by the same search as any other join:
    the perpendicular foot on each segment is clamped to the segment,
    and the segment whose clamped foot is closest wins, ties
    resolving to the later segment - the outgoing one at a shared
    waypoint, the right-continuous convention
    :meth:`~f1telemetry.tracks.Track.frame_at` uses - so a position
    exactly on a waypoint projects deterministically and
    consistently with the frame at its arc length. The returned arc
    length is
    wrapped into ``[0, length_m)``, so the first waypoint is
    addressed as ``0.0`` whichever side of the seam reached it, and
    the lateral offset carries :func:`~f1telemetry.tracks.Track.point_at`'s
    sign convention - positive to the left of the direction of
    travel.

    Inputs: ``x_m``/``y_m``, a world position in the ground frame,
    metres. The projection is exact for the polyline the file
    describes; it is not a projection onto the racing line or any
    other offset curve.
    """
    x = _check("x_m", x_m)
    y = _check("y_m", y_m)
    start_x = track.x_m
    start_y = track.y_m
    # Each segment's direction, including the closing one, which
    # `roll` forms from the last waypoint back to the first without
    # the file repeating that waypoint. Segment lengths are positive -
    # `load_track` refuses zero-length segments - so the denominator
    # cannot vanish.
    delta_x = np.roll(start_x, -1) - start_x
    delta_y = np.roll(start_y, -1) - start_y
    # The closest point on a segment is its perpendicular foot,
    # clamped to the segment; every segment competes, and the
    # winner is picked from the distances below.
    parameter = ((x - start_x) * delta_x + (y - start_y) * delta_y) / (
        delta_x * delta_x + delta_y * delta_y
    )
    parameter = np.clip(parameter, 0.0, 1.0)
    closest_x = start_x + parameter * delta_x
    closest_y = start_y + parameter * delta_y
    distance_squared = (x - closest_x) ** 2 + (y - closest_y) ** 2
    # `argmin` over the reversed distances keeps the *last* minimum in
    # segment order, so an exact tie between two segments resolves to
    # the later one - the outgoing segment at a shared waypoint, the
    # same right-continuous convention `frame_at` uses - and a
    # position exactly on a waypoint projects consistently with the
    # frame at its arc length.
    index = len(distance_squared) - 1 - int(np.argmin(distance_squared[::-1]))

    fraction = float(parameter[index])
    s_m = float(track.s_m[index]) + fraction * float(track.ds_m[index])
    # The closing segment's far end is the first waypoint, whose arc
    # length is zero: wrap so both ways of reaching it address the
    # same knot on the lap.
    s_m %= track.length_m
    # Lateral offset along the nearest segment's normal, the same
    # normal `point_at` adds its `lateral_m` to, so the sign matches.
    tangent_x = float(track.tangent_x_m[index])
    tangent_y = float(track.tangent_y_m[index])
    residual_x = x - float(closest_x[index])
    residual_y = y - float(closest_y[index])
    lateral_m = residual_x * (-tangent_y) + residual_y * tangent_x
    distance_m = math.sqrt(float(distance_squared[index]))
    return TrackProjection(s_m=s_m, lateral_m=lateral_m, distance_m=distance_m)


def wheel_positions(
    x_m: float,
    y_m: float,
    heading_rad: float,
    wheel_base_m: float,
    axle_track_m: float,
) -> WheelPositions:
    """The four contact patches of a car at one CG position and heading.

    The wheels ride the body frame :mod:`~f1telemetry.physics.kinematics`
    integrates in - ``x`` forward, ``y`` left, right-handed - so the
    front axle sits half the wheel base ahead of the CG along the
    heading, the rear axle half behind it, and each wheel half the
    track width to its side along the heading's left normal
    ``(-sin(heading), cos(heading))``. One track width serves both
    axles; a car whose front and rear tracks differ is the caller's
    geometry to place, because reading it from ``car_spec.yaml`` is
    the caller's job.

    Inputs: the CG position ``x_m``/``y_m`` in metres, the heading
    ``heading_rad`` in radians counterclockwise from ``+x``, and the
    positive wheel geometry ``wheel_base_m`` and ``axle_track_m`` in
    metres. The returned order is ``FL, FR, RL, RR``.
    """
    x = _check("x_m", x_m)
    y = _check("y_m", y_m)
    heading = _check("heading_rad", heading_rad)
    wheel_base = _positive("wheel_base_m", wheel_base_m)
    axle_track = _positive("axle_track_m", axle_track_m)

    cosine = math.cos(heading)
    sine = math.sin(heading)
    forward_x = cosine
    forward_y = sine
    left_x = -sine
    left_y = cosine
    front_x = x + 0.5 * wheel_base * forward_x
    front_y = y + 0.5 * wheel_base * forward_y
    rear_x = x - 0.5 * wheel_base * forward_x
    rear_y = y - 0.5 * wheel_base * forward_y
    half_track = 0.5 * axle_track
    return WheelPositions(
        fl_x_m=front_x + half_track * left_x,
        fl_y_m=front_y + half_track * left_y,
        fr_x_m=front_x - half_track * left_x,
        fr_y_m=front_y - half_track * left_y,
        rl_x_m=rear_x + half_track * left_x,
        rl_y_m=rear_y + half_track * left_y,
        rr_x_m=rear_x - half_track * left_x,
        rr_y_m=rear_y - half_track * left_y,
    )


def wheels_within_track_limits(
    track: Track,
    wheels: WheelPositions,
    tolerance_m: float = 0.0,
) -> bool:
    """Whether all four wheels are inside the local half-width, at once.

    Each wheel is projected onto the centerline and its signed
    lateral offset compared against half the width
    :func:`~f1telemetry.tracks.Track.frame_at` reports at that arc
    length - the local half-width, interpolated piecewise-linearly
    between waypoints exactly as every other per-waypoint quantity.
    A wheel exactly on the limit is inside; ``tolerance_m`` widens
    the limit by a stated, nonnegative amount for callers whose
    positions carry rounding, and defaults to zero so the limit is
    the file's width and nothing else. The check is per wheel and
    per instant: one wheel, one sample outside is enough to return
    ``False``.
    """
    tolerance = _nonnegative("tolerance_m", tolerance_m)
    for x_m, y_m in wheels.positions():
        projection = project_position(track, x_m, y_m)
        frame = track.frame_at(projection.s_m)
        if abs(projection.lateral_m) > 0.5 * frame.width_m + tolerance:
            return False
    return True


def crossing_events(
    track: Track,
    samples: Sequence[TrackSample],
) -> tuple[TrackEvent, ...]:
    """The sector and start/finish crossings of a stream of timed samples.

    Each sample's wrapped progress is unwrapped across the
    start/finish line first (a drop of at least half a lap between
    consecutive samples counts a forward crossing, a rise of more
    than half a lap a backward one, a jump of exactly half a lap
    resolving forward), and every boundary the
    *unwrapped* progress moves forward through becomes one
    :class:`TrackEvent`: the start/finish line as
    :attr:`TrackEventKind.LAP`, each entry of the track file's
    ``sectors`` as :attr:`TrackEventKind.SECTOR`. A boundary crossed
    between two samples is timed by linear interpolation of the
    samples' simulation times through the crossing, and a boundary
    reached exactly at a sample takes that sample's time; a stream
    that begins exactly on a boundary emits no event for it. Sample
    pairs that do not advance the unwrapped progress emit nothing, so
    reversing over a boundary times nothing - the ``invalid`` flag,
    not an event, is where a reversal belongs.

    The unwrap is unambiguous only while consecutive samples are at
    most half a lap apart, a jump of exactly half a lap resolving
    forward; that is a contract on the caller's sampling density,
    stated here rather than enforced, because what a larger jump
    means depends on intent no module can infer. Events are
    returned ordered by simulated time, then arc length, so the order
    is total and repeatable. An empty or single-sample stream crosses
    nothing and returns ``()``.
    """
    checked = _checked_samples(samples)
    unwrapped = _unwrap_progress(track, _project_samples(track, checked))
    return _crossings(track, checked, unwrapped)


def evaluate_lap_validity(
    completed_lap: bool,
    wheels_within_limits: bool,
    dnf: bool = False,
    invalid: bool = False,
) -> LapValidity:
    """Whether a recorded lap is a valid measurement, from its four inputs.

    The verdict (P4-T7) requires all of: a *completed lap* - the
    sample stream contains a forward start/finish crossing, which
    :func:`crossing_events` reports as a :attr:`TrackEventKind.LAP`
    event - all four wheels within the local half-width at every
    recorded sample, and neither the DNF nor the invalid flag set.
    The two flags are the caller's to set: ``dnf`` for a car that did
    not finish, ``invalid`` for anything the caller wants excluded
    from timing comparisons - a cut corner, an excursion, a lap the
    scenario author distrusts. This module reads both and never sets
    either, so the decision to reject a lap for a reason the geometry
    cannot see stays with the caller.

    The conditions are evaluated in a fixed order - ``dnf``,
    ``invalid``, ``incomplete_lap``, ``off_track`` - and every failure
    is listed in :attr:`LapValidity.failures`, not just the first, so
    a rejected lap reports all its reasons at once.
    """
    failures: list[str] = []
    if dnf:
        failures.append(_FAILURE_DNF)
    if invalid:
        failures.append(_FAILURE_INVALID)
    if not completed_lap:
        failures.append(_FAILURE_INCOMPLETE_LAP)
    if not wheels_within_limits:
        failures.append(_FAILURE_OFF_TRACK)
    return LapValidity(
        valid=not failures,
        completed_lap=completed_lap,
        wheels_within_limits=wheels_within_limits,
        dnf=dnf,
        invalid=invalid,
        failures=tuple(failures),
    )


def assess_lap(
    track: Track,
    samples: Sequence[TrackSample],
    wheel_base_m: float,
    axle_track_m: float,
    dnf: bool = False,
    invalid: bool = False,
    tolerance_m: float = 0.0,
) -> LapAssessment:
    """Time and judge one stream of samples as one lap, in one call.

    The composition a timing consumer wants: the samples are
    validated and projected, their progress unwrapped, their sector
    and lap crossings collected as :class:`TrackEvent` records, and
    every sample's four wheels - placed by :func:`wheel_positions`
    from the sample's own heading and the given wheel geometry -
    checked against the local half-width. The lap is complete when
    the crossings include a :attr:`TrackEventKind.LAP` event, and the
    wheels are within limits only when they are within limits at
    *every* sample, so a single excursion anywhere in the stream
    rejects the lap. The DNF and invalid flags pass straight through
    to :func:`evaluate_lap_validity`, which holds the final verdict.

    Inputs: the ``track`` to time against, the ``samples`` stream,
    the positive car geometry ``wheel_base_m``/``axle_track_m`` in
    metres, the caller-owned ``dnf``/``invalid`` flags, and the
    nonnegative ``tolerance_m`` by which the half-width limits are
    widened. Returns the :class:`LapAssessment` with the events, the
    verdict, the unwrapped progress and the per-sample projections.
    """
    checked = _checked_samples(samples)
    wheel_base = _positive("wheel_base_m", wheel_base_m)
    axle_track = _positive("axle_track_m", axle_track_m)
    tolerance = _nonnegative("tolerance_m", tolerance_m)

    projections = _project_samples(track, checked)
    unwrapped = _unwrap_progress(track, projections)
    events = _crossings(track, checked, unwrapped)
    completed_lap = any(event.kind is TrackEventKind.LAP for event in events)

    wheels_within_limits = True
    for sample in checked:
        wheels = wheel_positions(
            sample.x_m,
            sample.y_m,
            sample.heading_rad,
            wheel_base,
            axle_track,
        )
        if not wheels_within_track_limits(track, wheels, tolerance):
            wheels_within_limits = False
            break

    return LapAssessment(
        events=events,
        validity=evaluate_lap_validity(
            completed_lap, wheels_within_limits, dnf, invalid
        ),
        progress_m=unwrapped,
        projections=projections,
    )


def _project_samples(
    track: Track,
    samples: Sequence[TrackSample],
) -> tuple[TrackProjection, ...]:
    """Every sample's projection, in stream order."""
    return tuple(project_position(track, sample.x_m, sample.y_m) for sample in samples)


def _unwrap_progress(
    track: Track,
    projections: Sequence[TrackProjection],
) -> tuple[float, ...]:
    """Per-sample progress unwrapped across the start/finish line.

    A drop of at least half the lap length between consecutive
    wrapped values is a forward crossing of the line and lifts the
    lap counter; a rise of more than half a lap is a backward
    crossing and lowers it, so a brief reversal does not corrupt the
    distance the car has covered. A jump of exactly half a lap is
    ambiguous in principle and resolves forward, the direction the
    streams this module times drive, so a sample that lands exactly
    on the line counts as a crossing rather than being dropped. An
    empty stream unwraps to nothing.
    """
    if not projections:
        return ()
    length = track.length_m
    threshold = length * _WRAP_FRACTION
    unwrapped: list[float] = []
    laps = 0
    previous = projections[0].s_m
    unwrapped.append(previous)
    for projection in projections[1:]:
        delta = projection.s_m - previous
        # A drop of at least half a lap is a forward crossing of the
        # line and lifts the lap counter; a rise of more than half a
        # lap is a backward crossing and lowers it. A jump of exactly
        # half a lap is ambiguous in principle - forward or backward
        # both land on the same wrapped value - and resolves forward,
        # the direction the streams this module times drive, so a
        # sample that lands exactly on the line is counted as a
        # crossing rather than dropped.
        if delta <= -threshold:
            laps += 1
        elif delta > threshold:
            laps -= 1
        unwrapped.append(projection.s_m + laps * length)
        previous = projection.s_m
    return tuple(unwrapped)


def _crossings(
    track: Track,
    samples: Sequence[TrackSample],
    unwrapped: Sequence[float],
) -> tuple[TrackEvent, ...]:
    """The forward crossings between consecutive unwrapped samples.

    A boundary is crossed between two samples when the unwrapped
    progress moves forward from strictly below it to at or past it,
    so a boundary reached exactly at a sample is timed at that
    sample, and a boundary the stream begins on is never crossed -
    there is no earlier sample to cross from. Timestamps interpolate
    linearly in progress between the samples' simulation times, and
    only forward progress is examined: a sample pair that does not
    advance emits nothing, which is what keeps a reversing car from
    manufacturing sector times.
    """
    length = track.length_m
    events: list[TrackEvent] = []
    lap_count = 0
    for index in range(len(samples) - 1):
        start = unwrapped[index]
        end = unwrapped[index + 1]
        if end <= start:
            continue
        time_s = samples[index].time_s
        # Seconds per metre of progress over this pair, so a crossing's
        # time is the pair's start time plus the crossing's share of
        # the span - the linear interpolation the simulated timestamps
        # are made of.
        seconds_per_metre = (samples[index + 1].time_s - time_s) / (end - start)
        # The start/finish line: every multiple of the lap length the
        # progress passes, counted in stream order so the index is the
        # number of forward crossings the stream has produced so far.
        multiple = math.floor(start / length) + 1
        while multiple * length <= end:
            lap_count += 1
            target = multiple * length
            events.append(
                TrackEvent(
                    time_s=time_s + (target - start) * seconds_per_metre,
                    s_m=0.0,
                    kind=TrackEventKind.LAP,
                    index=lap_count,
                )
            )
            multiple += 1
        # Each sector boundary of the lap, repeated on every lap the
        # span covers: the smallest lap count that places the boundary
        # strictly past the pair's start, then upward.
        for sector_index, boundary in enumerate(track.sector_boundaries_m, start=1):
            lap = math.floor((start - boundary) / length) + 1
            while boundary + lap * length <= end:
                target = boundary + lap * length
                events.append(
                    TrackEvent(
                        time_s=time_s + (target - start) * seconds_per_metre,
                        s_m=boundary,
                        kind=TrackEventKind.SECTOR,
                        index=sector_index,
                    )
                )
                lap += 1
    events.sort(key=lambda event: (event.time_s, event.s_m, event.kind.value, event.index))
    return tuple(events)


def _checked_samples(samples: Sequence[TrackSample]) -> tuple[TrackSample, ...]:
    """Validate a sample stream: finite fields, nondecreasing simulation time.

    A decreasing simulation time is a caller bug - a stream that runs
    backwards cannot be timed - so it fails here, at the boundary,
    with the two offending times in the message.
    """
    checked: list[TrackSample] = []
    previous_time = -math.inf
    for sample in samples:
        time_s = _check("time_s", sample.time_s)
        _check("x_m", sample.x_m)
        _check("y_m", sample.y_m)
        _check("heading_rad", sample.heading_rad)
        if time_s < previous_time:
            raise ValueError(
                f"sample time_s must be nondecreasing, got {time_s} s "
                f"after {previous_time} s"
            )
        previous_time = time_s
        checked.append(sample)
    return tuple(checked)


def _check(label: str, value: object) -> float:
    """Narrow one argument to a finite ``float``, or fail before any arithmetic.

    ``bool`` is refused although it is an ``int``: a boolean position
    is a caller bug, not a number. This is the boundary's check, the
    same one :class:`~f1telemetry.tracks.Track` makes on its own
    queries.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return number


def _positive(label: str, value: object) -> float:
    """One argument as a finite ``float`` that must be greater than zero."""
    number = _check(label, value)
    if number <= 0.0:
        raise ValueError(f"{label} must be > 0, got {value!r}")
    return number


def _nonnegative(label: str, value: object) -> float:
    """One argument as a finite ``float`` that must not be negative."""
    number = _check(label, value)
    if number < 0.0:
        raise ValueError(f"{label} must be >= 0, got {value!r}")
    return number
