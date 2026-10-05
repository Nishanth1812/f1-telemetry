"""Hand-authored fixture records for the invariant and golden harnesses (P0-T7, P0-T8).

These are **supplied values, not simulation output**. Every number below was chosen so
that one invariant property holds exactly and the others are either trivially satisfied
or deliberately violated in a mutation fixture. Nothing here is physics: no tyre model,
no integrator, no fabricated telemetry. The point is that the eight invariant *semantics*
- the formula, the sign convention, the tolerance - are pinned down and non-vacuous
before P1 hands them real traces.

Each record satisfies the full sum rules, so the only way a checker passes is by actually
computing the right thing:

``straight_line_record``
    Four 10 ms steps of full-throttle acceleration, zero steer, zero camber, symmetric
    axle loads. Load transfer ``m * ax * h_cg / L`` moves weight rearward; aerodynamic
    downforce from ``car_spec`` at 50 m/s is added and split evenly across the four
    corners, which is a fixture simplification - invariant 3 only constrains the sum.
    Powers are the 53/47 split of ``m * vx * ax`` plus the positive drag-power magnitude;
    ``drag_w`` stores that resisting term as a negative value. The tire is exercised on
    both axes at the same time (0.20/0.10 on the loaded outside tyres, 0.15/0.06 inside).

``cornering_record``
    Four 10 ms steps of a steady left-hand corner with combined slip, so the sign
    conventions, the friction ellipse and the lateral load transfer are all non-trivial.
    Loads hold their mid-corner values across the four steps while ``vx`` changes, so the
    energy check can be evaluated at every step without inventing a load-transfer
    transient.
"""

from __future__ import annotations

from typing import Final

from f1telemetry.testing.records import (
    GroundTruthStep,
    SampleRecord,
    SensorFrame,
    WheelTruth,
)

__all__ = [
    "CORNERING",
    "DT_S",
    "STRAIGHT_LINE",
    "cornering_record",
    "corrupted_records",
    "golden_channels",
    "straight_line_record",
]

DT_S: Final[float] = 0.01
STRAIGHT_LINE: Final[str] = "straight_line"
CORNERING: Final[str] = "cornering"

_MU: Final[float] = 1.8


def _straight_line_frames(
    speeds_m_s: tuple[float, ...], ice_rpm: tuple[float, ...], wheels_speed_kmh: tuple[float, ...]
) -> tuple[SensorFrame, ...]:
    frames: list[SensorFrame] = []
    for index, speed in enumerate(speeds_m_s):
        frames.append(
            SensorFrame(
                t_s=index * DT_S,
                values={
                    "speed": speed * 3.6,
                    "vx": speed,
                    "vy": 0.0,
                    "yaw_rate": 0.0,
                    "accel_longitudinal": 10.0,
                    "accel_lateral": 0.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "ice_rpm": ice_rpm[index],
                    "gear": float(3 if index < 2 else 4),
                    "throttle_pct": 100.0,
                    "clutch_pct": 100.0,
                    "ice_torque_nm": 320.0,
                    "mgu_k_power_kw": 188.8368,
                    "fuel_flow_kg_h": 98.0,
                    "aero_mode": 0.0,
                    "downforce_n": 6546.5625,
                    "fw_flap_deg": 12.0,
                    "rw_flap_deg": 9.0,
                    "brake_pressure": 0.0,
                    "steering_angle": 0.0,
                    "pedal_travel": 100.0,
                    **{
                        f"wheel_speed_{corner.lower()}": wheels_speed_kmh[index]
                        for corner in ("FL", "FR", "RL", "RR")
                    },
                    **{
                        f"tyre_temp_{corner.lower()}": 88.0 + index
                        for corner in ("FL", "FR", "RL", "RR")
                    },
                },
            )
        )
    return tuple(frames)


def straight_line_record() -> SampleRecord:
    """Full-throttle acceleration, symmetric, four steps of 10 ms."""
    speeds = (50.0, 50.1, 50.2, 50.3)
    ice_rpm = (9103.6, 9203.0, 9302.5, 9401.9)
    wheel_kmh = tuple(round(speed * 3.6, 1) for speed in speeds)
    front = 2782.2407
    rear = 4413.7005
    wheels = (
        WheelTruth(front, 1500.0, 0.0, _MU, 0.02, 0.0, 0.0),
        WheelTruth(front, 1500.0, 0.0, _MU, 0.02, 0.0, 0.0),
        WheelTruth(rear, 2500.0, 0.0, _MU, 0.05, 0.0, 0.0),
        WheelTruth(rear, 2500.0, 0.0, _MU, 0.05, 0.0, 0.0),
    )
    ice_power = (212943.7, 213367.7, 213791.7, 214215.7)
    mgu_power = (188836.8, 189212.8, 189588.8, 189964.8)
    steps = tuple(
        GroundTruthStep(
            t_s=index * DT_S,
            vx_m_s=speeds[index],
            vy_m_s=0.0,
            ax_m_s2=10.0,
            ay_m_s2=0.0,
            az_m_s2=0.0,
            gear=3 if index < 2 else 4,
            clutch=1.0,
            throttle_pct=100.0,
            ice_power_w=ice_power[index],
            mgu_k_power_w=mgu_power[index],
            drag_w=-1780.5,
            downforce_n=6546.5625,
            steer_rad=0.0,
            wheels=wheels,
        )
        for index in range(len(speeds))
    )
    return SampleRecord(
        name=STRAIGHT_LINE,
        dt_s=DT_S,
        description=(
            "Straight-line full-throttle acceleration. Supplied values, not simulation "
            "output: the four corner loads, the 53/47 power split and the drag term are "
            "hand-computed so that invariants 2, 3, 4, 5 and 6 all hold exactly."
        ),
        ground_truth=steps,
        frames=_straight_line_frames(speeds, ice_rpm, wheel_kmh),
    )


def cornering_record() -> SampleRecord:
    """Steady left-hand corner with combined slip, four steps of 10 ms."""
    speeds = (40.0, 40.06, 40.12, 40.18)
    vy = 0.8
    steer = 0.01
    alpha_front = 0.010
    alpha_rear = 0.020
    front_left = 3276.3795
    front_right = 1526.3795
    rear_left = 4380.7805
    rear_right = 2630.7805
    wheels = (
        WheelTruth(front_left, 800.0, 800.0, _MU, 0.01, alpha_front, -3.0),
        WheelTruth(front_right, 800.0, 800.0, _MU, 0.01, alpha_front, -3.0),
        WheelTruth(rear_left, 1600.0, 1200.0, _MU, 0.02, alpha_rear, -3.0),
        WheelTruth(rear_right, 1600.0, 1200.0, _MU, 0.02, alpha_rear, -3.0),
    )
    ice_power = (104094.9, 104247.5, 104400.1, 104552.8)
    mgu_power = (92310.5, 92445.9, 92581.3, 92716.6)
    steps = tuple(
        GroundTruthStep(
            t_s=index * DT_S,
            vx_m_s=speeds[index],
            vy_m_s=vy,
            ax_m_s2=6.0,
            ay_m_s2=5.0,
            az_m_s2=0.0,
            gear=4,
            clutch=1.0,
            throttle_pct=100.0,
            ice_power_w=ice_power[index],
            mgu_k_power_w=mgu_power[index],
            drag_w=-1205.4,
            downforce_n=3969.0,
            steer_rad=steer,
            wheels=wheels,
        )
        for index in range(len(speeds))
    )
    frames = tuple(
        SensorFrame(
            t_s=index * DT_S,
            values={
                "speed": speeds[index] * 3.6,
                "vx": speeds[index],
                "vy": vy,
                "yaw_rate": 20.0,
                "accel_longitudinal": 6.0,
                "accel_lateral": 5.0,
                "roll": 2.1,
                "pitch": 1.4,
                "ice_rpm": 8300.0 + index * 40.0,
                "gear": 4.0,
                "throttle_pct": 100.0,
                "clutch_pct": 100.0,
                "ice_torque_nm": 300.0,
                "mgu_k_power_kw": 92.3105,
                "fuel_flow_kg_h": 80.0,
                "aero_mode": 0.0,
                "downforce_n": 3969.0,
                "fw_flap_deg": 14.0,
                "rw_flap_deg": 11.0,
                "brake_pressure": 0.0,
                "steering_angle": 0.572958,
                "pedal_travel": 100.0,
                **{
                    f"wheel_speed_{corner.lower()}": speeds[index] * 3.6
                    for corner in ("FL", "FR", "RL", "RR")
                },
                **{
                    f"slip_angle_{corner.lower()}": (
                        alpha_front if corner in ("FL", "FR") else alpha_rear
                    )
                    * 57.29577951308232
                    for corner in ("FL", "FR", "RL", "RR")
                },
                **{
                    f"brake_temp_{corner.lower()}": 420.0 + index
                    for corner in ("FL", "FR", "RL", "RR")
                },
            },
        )
        for index in range(len(speeds))
    )
    return SampleRecord(
        name=CORNERING,
        dt_s=DT_S,
        description=(
            "Steady left-hand corner with combined slip. Supplied values, not simulation "
            "output: lateral load transfer, corner loads and the power terms are "
            "hand-computed so that invariants 2, 3, 4 and 6 hold exactly while the "
            "friction ellipse and the sign conventions are genuinely non-trivial."
        ),
        ground_truth=steps,
        frames=frames,
    )


def golden_channels(record: SampleRecord) -> tuple[str, ...]:
    """The published channels a golden baseline is committed for."""
    return record.channels


def corrupted_records() -> dict[int, SampleRecord]:
    """One record per content invariant that violates exactly that invariant.

    Used by the non-vacuity half of every invariant test. A checker that returns "pass"
    on any of these is broken, and that is the assertion. Invariant 8 is absent by
    design: it is a property of the serialisation path, not of any record's contents, so
    its non-vacuity is proved by leaking a wall-clock value into Parquet metadata and
    asserting the byte comparison catches it.
    """
    from f1telemetry.testing.records import (
        with_frame_value,
        with_gear_sequence,
        with_step,
        with_wheel,
    )

    straight = straight_line_record()
    corner = cornering_record()
    return {
        1: with_frame_value(straight, 2, "speed", float("nan")),
        2: with_wheel(corner, 1, "FL", fy_n=9000.0),
        3: with_wheel(straight, 1, "RR", fz_n=4413.7005 - 250.0),
        4: with_wheel(corner, 0, "RL", fy_n=-1200.0),
        5: with_wheel(straight, 0, "RL", kappa=0.06),
        6: with_step(straight, 2, ice_power_w=213791.7 * 0.9),
        7: with_gear_sequence(straight, (3, 5, 4, 6)),
    }
