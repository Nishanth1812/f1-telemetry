"""Record types the invariant and golden harnesses operate on.

Two streams, matching the split in PLAN.md section 3:

* :class:`GroundTruthStep` is the physics state. It carries Fx, Fy, kappa, alpha, load
  and the power terms that no real sensor exposes, so it is deliberately *not* in
  ``channels.yaml`` - it is the ``GroundTruth`` side channel.
* :class:`SensorFrame` is what the emulated sensor chain would publish, keyed by contract
  channel name. This is the only stream the analytics, storage and dashboard layers may
  ever see, and invariant 1 and invariant 8 are checked on it.

Both are frozen. Mutating a fixture means producing a new record, which keeps the
"supplied fixture values" honest: a test cannot quietly alter a baseline in place.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Final, cast

__all__ = [
    "CORNERS",
    "GroundTruthStep",
    "SampleRecord",
    "SensorFrame",
    "WheelTruth",
    "with_channel",
    "with_frame_value",
    "with_gear_sequence",
    "with_step",
    "with_wheel",
]

CORNERS: Final[tuple[str, ...]] = ("FL", "FR", "RL", "RR")


def _unloaded_wheels() -> tuple[WheelTruth, WheelTruth, WheelTruth, WheelTruth]:
    wheels = tuple(WheelTruth(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0) for _ in CORNERS)
    return cast("tuple[WheelTruth, WheelTruth, WheelTruth, WheelTruth]", wheels)


@dataclass(frozen=True, slots=True)
class WheelTruth:
    """Per-corner physics truth, including separate longitudinal and lateral grip peaks.

    ``mu`` remains the longitudinal peak coefficient for existing P1 fixtures. ``mu_lateral``
    is the load-adjusted lateral coefficient; ``None`` means a legacy fixture uses ``mu`` for
    both axes.
    """

    fz_n: float
    fx_n: float
    fy_n: float
    mu: float
    kappa: float
    alpha_rad: float
    camber_deg: float
    mu_lateral: float | None = None
    effective_alpha_rad: float | None = None


@dataclass(frozen=True, slots=True)
class GroundTruthStep:
    """One 10 kHz-equivalent physics step, decimated for inspection.

    Sign conventions, fixed once here so the invariants have something unambiguous to
    check (PLAN.md section 11 invariants 2, 4, 5):

    * body frame is x forward, y left, z up
    * ``kappa`` positive in drive, ``fx_n`` positive forward
    * ``alpha_rad`` positive when the tyre generates ``fy_n`` in +y
    * ``drag_w`` is signed against forward motion, so aerodynamic drag power is negative
    * ``gear`` is -1 reverse, 0 neutral, 1..8
    * ``az_m_s2`` is the chassis vertical acceleration, so the load sum is
      ``mass * (g + az) + downforce``
    """

    t_s: float
    vx_m_s: float
    vy_m_s: float
    ax_m_s2: float
    ay_m_s2: float
    az_m_s2: float
    gear: int
    clutch: float
    throttle_pct: float
    ice_power_w: float
    mgu_k_power_w: float
    drag_w: float
    downforce_n: float
    steer_rad: float
    energy_residual_fraction: float | None = None
    wheels: tuple[WheelTruth, WheelTruth, WheelTruth, WheelTruth] = field(
        default_factory=_unloaded_wheels
    )
    yaw_rate_rad_s: float = 0.0
    roll_rad: float = 0.0
    pitch_rad: float = 0.0
    heave_m: float = 0.0
    suspension_travel_m: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    travel_limited: tuple[bool, bool, bool, bool] = (False, False, False, False)

    @property
    def wheel(self) -> dict[str, WheelTruth]:
        return dict(zip(CORNERS, self.wheels, strict=True))

    def wheel_index(self, corner: str) -> int:
        return CORNERS.index(corner)


@dataclass(frozen=True, slots=True)
class SensorFrame:
    """One published sample per channel, keyed by contract channel name."""

    t_s: float
    values: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class SampleRecord:
    """A complete, self-contained record: the truth stream and the sensor stream."""

    name: str
    dt_s: float
    description: str
    ground_truth: tuple[GroundTruthStep, ...]
    frames: tuple[SensorFrame, ...]

    def __len__(self) -> int:
        return len(self.ground_truth)

    @property
    def channels(self) -> tuple[str, ...]:
        if not self.frames:
            return ()
        return tuple(self.frames[0].values)

    def series(self, channel: str) -> tuple[float, ...]:
        return tuple(frame.values[channel] for frame in self.frames)


def _pick(override: float | None, current: float) -> float:
    return current if override is None else override


def with_wheel(
    record: SampleRecord,
    step: int,
    corner: str,
    *,
    fz_n: float | None = None,
    fx_n: float | None = None,
    fy_n: float | None = None,
    mu: float | None = None,
    mu_lateral: float | None = None,
    kappa: float | None = None,
    alpha_rad: float | None = None,
    camber_deg: float | None = None,
    effective_alpha_rad: float | None = None,
) -> SampleRecord:
    """Return a copy of `record` with one wheel's truth fields replaced.

    Keyword-only and explicit per field, so that a typo in a mutation is a type error
    rather than a silently ignored keyword that leaves the fixture unchanged.
    """
    target = record.ground_truth[step]
    index = target.wheel_index(corner)
    current = target.wheels[index]
    merged = WheelTruth(
        fz_n=_pick(fz_n, current.fz_n),
        fx_n=_pick(fx_n, current.fx_n),
        fy_n=_pick(fy_n, current.fy_n),
        mu=_pick(mu, current.mu),
        kappa=_pick(kappa, current.kappa),
        alpha_rad=_pick(alpha_rad, current.alpha_rad),
        camber_deg=_pick(camber_deg, current.camber_deg),
        mu_lateral=current.mu_lateral if mu_lateral is None else mu_lateral,
        effective_alpha_rad=(
            current.effective_alpha_rad if effective_alpha_rad is None else effective_alpha_rad
        ),
    )
    wheels = list(target.wheels)
    wheels[index] = merged
    return _with_ground_truth(record, step, replace(target, wheels=tuple(wheels)))


def with_step(
    record: SampleRecord,
    step: int,
    *,
    vx_m_s: float | None = None,
    vy_m_s: float | None = None,
    ax_m_s2: float | None = None,
    ay_m_s2: float | None = None,
    az_m_s2: float | None = None,
    clutch: float | None = None,
    throttle_pct: float | None = None,
    ice_power_w: float | None = None,
    mgu_k_power_w: float | None = None,
    drag_w: float | None = None,
    downforce_n: float | None = None,
    steer_rad: float | None = None,
    yaw_rate_rad_s: float | None = None,
    roll_rad: float | None = None,
    pitch_rad: float | None = None,
    heave_m: float | None = None,
    suspension_travel_m: tuple[float, float, float, float] | None = None,
    travel_limited: tuple[bool, bool, bool, bool] | None = None,
) -> SampleRecord:
    """Return a copy of `record` with one ground-truth step's scalars replaced."""
    target = record.ground_truth[step]
    updated = replace(
        target,
        vx_m_s=_pick(vx_m_s, target.vx_m_s),
        vy_m_s=_pick(vy_m_s, target.vy_m_s),
        ax_m_s2=_pick(ax_m_s2, target.ax_m_s2),
        ay_m_s2=_pick(ay_m_s2, target.ay_m_s2),
        az_m_s2=_pick(az_m_s2, target.az_m_s2),
        clutch=_pick(clutch, target.clutch),
        throttle_pct=_pick(throttle_pct, target.throttle_pct),
        ice_power_w=_pick(ice_power_w, target.ice_power_w),
        mgu_k_power_w=_pick(mgu_k_power_w, target.mgu_k_power_w),
        drag_w=_pick(drag_w, target.drag_w),
        downforce_n=_pick(downforce_n, target.downforce_n),
        steer_rad=_pick(steer_rad, target.steer_rad),
        yaw_rate_rad_s=_pick(yaw_rate_rad_s, target.yaw_rate_rad_s),
        roll_rad=_pick(roll_rad, target.roll_rad),
        pitch_rad=_pick(pitch_rad, target.pitch_rad),
        heave_m=_pick(heave_m, target.heave_m),
        suspension_travel_m=(
            target.suspension_travel_m if suspension_travel_m is None else suspension_travel_m
        ),
        travel_limited=target.travel_limited if travel_limited is None else travel_limited,
    )
    return _with_ground_truth(record, step, updated)


def with_gear_sequence(record: SampleRecord, gears: tuple[int, ...]) -> SampleRecord:
    """Return a copy of `record` with the gear column replaced."""
    if len(gears) != len(record.ground_truth):
        msg = (
            f"gear sequence of length {len(gears)} does not match {len(record.ground_truth)} steps"
        )
        raise ValueError(msg)
    return _replace_record(
        record,
        ground_truth=tuple(
            replace(step, gear=gear) for step, gear in zip(record.ground_truth, gears, strict=True)
        ),
    )


def with_frame_value(record: SampleRecord, step: int, channel: str, value: float) -> SampleRecord:
    """Return a copy of `record` with one published channel sample replaced."""
    frames = list(record.frames)
    values = dict(frames[step].values)
    if channel not in values:
        msg = f"record {record.name!r} does not publish channel {channel!r}"
        raise KeyError(msg)
    values[channel] = value
    frames[step] = replace(frames[step], values=values)
    return _replace_record(record, frames=tuple(frames))


def with_channel(record: SampleRecord, channel: str, value: float) -> SampleRecord:
    """Return a copy of `record` with `channel` forced to `value` at every step."""
    out = record
    for step in range(len(record.frames)):
        out = with_frame_value(out, step, channel, value)
    return out


def _with_ground_truth(record: SampleRecord, step: int, updated: GroundTruthStep) -> SampleRecord:
    steps = list(record.ground_truth)
    steps[step] = updated
    return _replace_record(record, ground_truth=tuple(steps))


def _replace_record(
    record: SampleRecord,
    *,
    ground_truth: tuple[GroundTruthStep, ...] | None = None,
    frames: tuple[SensorFrame, ...] | None = None,
) -> SampleRecord:
    return SampleRecord(
        name=record.name,
        dt_s=record.dt_s,
        description=record.description,
        ground_truth=record.ground_truth if ground_truth is None else ground_truth,
        frames=record.frames if frames is None else frames,
    )
