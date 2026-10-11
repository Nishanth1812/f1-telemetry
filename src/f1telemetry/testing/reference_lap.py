"""Compose the reference driver and lap timing with the fixed-step scenario runner.

The P4-to-P5 handoff lives here too: :func:`publish_session_channels` maps a
run's lap/sector assessment onto the contract's session event channels
(``lap_index``, ``sector_index``, ``lap_time``, ``sector_time``, ``delta``) so
scenario records flow through the existing telemetry record/event path -
sensor processing, Parquet, live and replay frames - as ordinary channels, the
same way the dashboard's event view already consumes them (value changes on
contract event channels). No wall clock, no randomness: pure mapping of
simulated times and assessed crossings.
"""

from __future__ import annotations

from f1telemetry.laps import (
    LapAssessment,
    ReferenceLap,
    TrackEventKind,
    TrackSample,
    assess_lap,
    reference_lap_delta,
    wheel_positions,
    wheels_within_track_limits,
)
from f1telemetry.racing_line import LateralOffsetSolution, SpeedProfile
from f1telemetry.reference_driver import DriverRequest, pure_pursuit_request
from f1telemetry.testing.records import SampleRecord, SensorFrame
from f1telemetry.testing.scenarios import (
    TRACE_PSI_INDEX,
    TRACE_X_INDEX,
    TRACE_Y_INDEX,
    ControlLaw,
    ControlState,
    ScenarioRun,
)
from f1telemetry.tracks import Track

__all__ = [
    "SESSION_CHANNELS",
    "assess_scenario_lap",
    "make_reference_driver",
    "publish_session_channels",
]

#: The contract session event channels publication writes, in contract order.
#: Every published frame carries all five, so Parquet columns and replay
#: channels stay uniform across the run.
SESSION_CHANNELS: tuple[str, ...] = (
    "lap_index",
    "sector_index",
    "lap_time",
    "sector_time",
    "delta",
)

# Contract ranges (channels.yaml session group): publication clamps to them so
# a long or slow run cannot put an out-of-range value on the wire.
_LAP_INDEX_MAX: float = 5000.0
_SECTOR_INDEX_MAX: float = 3.0
_LAP_TIME_MAX_S: float = 1200.0
_SECTOR_TIME_MAX_S: float = 600.0
_DELTA_LIMIT_S: float = 5.0

# Event-to-frame join tolerance, in seconds. A crossing is timed by linear
# interpolation between its bracketing samples, so an event at the final
# sample can land microseconds after that sample's own timestamp; without a
# tolerance the closing lap would belong to no frame. 1 ms is orders of
# magnitude below any record interval while far above interpolation noise.
_EVENT_JOIN_TOLERANCE_S: float = 1.0e-3


def make_reference_driver(
    track: Track,
    line: LateralOffsetSolution,
    profile: SpeedProfile,
    *,
    wheelbase_m: float,
    steering_ratio: float,
    grip_margin: float = 0.9,
    speed_gain: float = 0.2,
) -> ControlLaw:
    """Return a scenario control law using pure pursuit and the solved speed profile."""

    def request(state: ControlState) -> DriverRequest:
        return pure_pursuit_request(
            track,
            line,
            profile,
            x_m=state.x_m,
            y_m=state.y_m,
            heading_rad=state.heading_rad,
            speed_m_s=state.speed_m_s,
            wheelbase_m=wheelbase_m,
            steering_ratio=steering_ratio,
            grip_margin=grip_margin,
            speed_gain=speed_gain,
        )

    return request


def _scenario_samples(run: ScenarioRun) -> tuple[TrackSample, ...]:
    """The run's recorded states as timed track samples, at the record rate."""
    rows = list(range(0, run.steps + 1, run.control_steps))
    if rows[-1] != run.steps:
        rows.append(run.steps)
    return tuple(
        TrackSample(
            time_s=row * run.dt_s,
            x_m=float(run.trace[row, TRACE_X_INDEX]),
            y_m=float(run.trace[row, TRACE_Y_INDEX]),
            heading_rad=float(run.trace[row, TRACE_PSI_INDEX]),
        )
        for row in rows
    )


def assess_scenario_lap(
    track: Track,
    run: ScenarioRun,
    *,
    wheelbase_m: float,
    axle_track_m: float,
    dnf: bool = False,
    invalid: bool = False,
    tolerance_m: float = 0.0,
) -> LapAssessment:
    """Project recorded scenario states into deterministic lap and sector results."""
    return assess_lap(
        track,
        _scenario_samples(run),
        wheelbase_m,
        axle_track_m,
        dnf=dnf,
        invalid=invalid,
        tolerance_m=tolerance_m,
    )


def publish_session_channels(
    run: ScenarioRun,
    track: Track,
    *,
    wheelbase_m: float,
    axle_track_m: float,
    dnf: bool = False,
    invalid: bool = False,
    tolerance_m: float = 0.0,
    reference: ReferenceLap | None = None,
) -> SampleRecord:
    """Publish lap/sector crossings and lap validity as session event channels.

    Returns a copy of ``run.record`` whose frames additionally carry the five
    contract session channels, so live and replay frames can carry P4 timing
    through the existing telemetry path (sensor processing, Parquet, replay,
    dashboard event log) with no special casing: session channels are ordinary
    columns there, and the dashboard logs their value changes.

    Per-frame semantics, all in simulated time:

    * ``lap_index``: completed forward start/finish crossings at or before the
      frame (``0`` while the opening lap is in progress). Crossings join their
      bracketing frame within a 1 ms interpolation tolerance, so a boundary
      reached at the final sample still belongs to the run.
    * ``sector_index``: sector boundaries completed in the current lap (``0``
      before the first boundary and again after each lap line).
    * ``lap_time``: duration of the most recently completed *valid* lap, held;
      ``0.0`` until one completes. A lap counts only when every sample since
      the previous lap line kept all four wheels within limits and neither
      ``dnf`` nor ``invalid`` is set, so an off-track or flagged lap advances
      the indices without publishing its time.
    * ``sector_time``: duration of the most recently completed clean sector,
      held; ``0.0`` until one completes. Gated the same way, per sector.
    * ``delta``: signed gap to ``reference`` at the frame's track position
      (positive means behind), clamped to the contract ``[-5, 5]`` s window;
      ``0.0`` when no reference is supplied.

    The first lap and sector are measured from the stream start, so a run that
    starts at the start/finish line yields true lap/sector times; a run that
    joins mid-lap times only the crossings it actually brackets. Every value
    is a pure function of the run and the track, so re-publishing is
    idempotent and two identical runs publish byte-identical channels.
    """
    frames = run.record.frames
    if not frames:
        return run.record
    assessment = assess_scenario_lap(
        track,
        run,
        wheelbase_m=wheelbase_m,
        axle_track_m=axle_track_m,
        dnf=dnf,
        invalid=invalid,
        tolerance_m=tolerance_m,
    )
    samples = _scenario_samples(run)
    within = tuple(
        _sample_within_limits(track, sample, wheelbase_m, axle_track_m, tolerance_m)
        for sample in samples
    )
    progress = assessment.progress_m
    events = assessment.events
    event_index = 0
    laps_completed = 0
    sectors_in_lap = 0
    lap_start_t = frames[0].t_s
    sector_start_t = frames[0].t_s
    lap_time_held = 0.0
    sector_time_held = 0.0
    clean_since_lap = True
    clean_since_sector = True
    published: list[SensorFrame] = []
    key_order = _session_key_order(frames[0])
    for frame_index, frame in enumerate(frames):
        sample_index = min(frame_index, len(samples) - 1)
        if not within[sample_index]:
            clean_since_lap = False
            clean_since_sector = False
        while (
            event_index < len(events)
            and events[event_index].time_s <= frame.t_s + _EVENT_JOIN_TOLERANCE_S
        ):
            event = events[event_index]
            event_index += 1
            if event.kind is TrackEventKind.LAP:
                if not dnf and not invalid and clean_since_lap:
                    lap_time_held = _clamp(event.time_s - lap_start_t, 0.0, _LAP_TIME_MAX_S)
                laps_completed += 1
                sectors_in_lap = 0
                lap_start_t = event.time_s
                sector_start_t = event.time_s
                clean_since_lap = within[sample_index]
                clean_since_sector = within[sample_index]
            else:
                if not dnf and not invalid and clean_since_sector:
                    sector_time_held = _clamp(
                        event.time_s - sector_start_t, 0.0, _SECTOR_TIME_MAX_S
                    )
                sectors_in_lap += 1
                sector_start_t = event.time_s
                clean_since_sector = within[sample_index]
        values = {
            "lap_index": min(float(laps_completed), _LAP_INDEX_MAX),
            "sector_index": min(float(sectors_in_lap), _SECTOR_INDEX_MAX),
            "lap_time": lap_time_held,
            "sector_time": sector_time_held,
            "delta": _session_delta(reference, track, progress, sample_index, frame, lap_start_t),
        }
        merged = dict(frame.values)
        merged.update(values)
        published.append(
            SensorFrame(t_s=frame.t_s, values={name: merged[name] for name in key_order})
        )
    return SampleRecord(
        name=run.record.name,
        dt_s=run.record.dt_s,
        description=(
            f"{run.record.description} Session event channels "
            f"({', '.join(SESSION_CHANNELS)}) published from the P4 lap/sector "
            "assessment; lap_time/sector_time hold only valid completions."
        ),
        ground_truth=run.record.ground_truth,
        frames=tuple(published),
    )


def _sample_within_limits(
    track: Track,
    sample: TrackSample,
    wheelbase_m: float,
    axle_track_m: float,
    tolerance_m: float,
) -> bool:
    """Whether one sample keeps all four wheels inside the local half-width."""
    wheels = wheel_positions(
        sample.x_m,
        sample.y_m,
        sample.heading_rad,
        wheelbase_m,
        axle_track_m,
    )
    return wheels_within_track_limits(track, wheels, tolerance_m)


def _session_delta(
    reference: ReferenceLap | None,
    track: Track,
    progress: tuple[float, ...],
    sample_index: int,
    frame: SensorFrame,
    lap_start_t: float,
) -> float:
    """The frame's reference-lap gap, or ``0.0`` with no reference to chase."""
    if reference is None or not progress:
        return 0.0
    s_m = progress[min(sample_index, len(progress) - 1)] % track.length_m
    elapsed = frame.t_s - lap_start_t
    return _clamp(
        reference_lap_delta(reference, s_m=s_m, lap_time_s=elapsed),
        -_DELTA_LIMIT_S,
        _DELTA_LIMIT_S,
    )


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the contract range ``[low, high]``."""
    return min(high, max(low, value))


def _session_key_order(first: SensorFrame) -> tuple[str, ...]:
    """Uniform column order: existing keys first, then any missing session key."""
    order = list(first.values)
    for name in SESSION_CHANNELS:
        if name not in first.values:
            order.append(name)
    return tuple(order)
