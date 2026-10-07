"""Phase 1 straight-line scenarios run over the existing physics APIs.

`PHASES.md` P1-T8 asks for scenarios that run from fixed initial conditions and emit traces
through the existing testing/record pattern. This
file covers that path and nothing wider:

* **The behaviour each scenario exists to show.** A standing launch starts from rest in first
  gear and seeds an engine-speed **state** that evolves during clutch slip; the gearbox moves
  *only* where the scenario asks, and each request cuts the driveline for the configured shift
  time; a neutral selection transmits exactly nothing and selecting first gear again restores it;
  a coasting car decelerates against drag alone and drag grows with speed; caller brake torque
  decelerates the car and its wheels; the MGU-K is blocked below 50 km/h in a declared standing
  start, drains the store when it deploys, refills it when it regenerates, without exceeding
  C5.2.7's 350 kW, and in the high-speed run it deploys only inside its one declared window and is
  absent from the ICE-only terminal tail.
* **The contracts the runner has to honour.** Caller-owned state, no clock and no random
  source, byte-identical repeat runs, and refusals for a duration that is not a whole number of
  kernel steps and for pedals, requests or brake torque it could not use.
* **The produced records.** Only contract channels are published, every published value stays
  inside the range `channels.yaml` declares for it, and `PLAN.md` section 11's invariants,
  including the discrete chassis and wheel energy balance, run against the real traces.
* **Two boundaries are pinned, not assumed.** The final recorded row's energy residual is the
  final complete control interval's, recomputed here from the documented wheel-boundary
  identity rather than trusted, and `ice_power_w` is pinned to its declared boundary - the
  ICE's gross crankshaft shaft power, which a shift cut does not collapse and a slipping
  clutch does not follow - with a regression for each case.
* **Straight-line references are used according to their evidence.** The coarse 0-100 km/h time
  and terminal speed are printed for review, but neither is assigned an unsupported tolerance.
  The high-speed run's transient maximum is checked against the cited 325.8 km/h reachability floor;
  it is kept separate from terminal speed.

No coefficient is tuned here. The coarse 0–100 result is reported without a performance band, and
the transient top-speed check uses the cited FIA reachability floor; all other physical inputs come
from `car_spec.yaml` or regulation limits enforced by the physics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Final

import numpy as np
import pytest

from f1telemetry.generated.channels import CHANNELS, DTYPES
from f1telemetry.kernels import longitudinal  # noqa: TID251 -- the scenarios drive the kernel
from f1telemetry.physics import (  # noqa: TID251 -- the scenarios drive this API
    engine,
    forces,
    gearbox,
    kinematics,
    steering,
)
from f1telemetry.testing import scenarios
from f1telemetry.testing.invariants import (
    check_energy_balance,
    check_gearbox_progression,
    run_all,
)
from f1telemetry.testing.records import CORNERS

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import CarSpec, KernelConfig
    from f1telemetry.testing.scenarios import Scenario, ScenarioRun

pytestmark = [pytest.mark.kernel, pytest.mark.invariant]

# The two straight-line references in `PLAN.md` §11.1 and `docs/calibration.md` §6. The 0–100
# median is reported without a pass/fail band because its ±0.30 s is feed quantisation; 325.8 km/h
# is an event speed-trap reachability floor checked against transient maximum, never terminal speed.
ZERO_TO_HUNDRED_REFERENCE_S: Final[float] = 2.32
TOP_SPEED_REACHABILITY_KMH: Final[float] = 325.8


@pytest.fixture(scope="module")
def config(spec: CarSpec) -> KernelConfig:
    """The validated, flat configuration the scenarios run on."""
    return spec.kernel_config()


@pytest.fixture(scope="module")
def suite(config: KernelConfig) -> Mapping[str, Scenario]:
    """The named scenario set, every input of which is read from `config`."""
    return scenarios.build_scenarios(config)


@pytest.fixture(scope="module")
def runs(config: KernelConfig, suite: Mapping[str, Scenario]) -> Mapping[str, ScenarioRun]:
    """Every scenario run once. A run is the expensive thing here, so it is not repeated."""
    return {name: scenarios.run_scenario(config, scenario) for name, scenario in suite.items()}


def _windows(run: ScenarioRun, scenario: Scenario) -> tuple[slice, ...]:
    """One recorded-step slice per segment, in scenario order.

    ``run_scenario`` requires every segment to be a whole number of control intervals, so the
    recorded windows line up with the segments exactly and a test can say "during this segment"
    without recomputing a time base.
    """
    windows: list[slice] = []
    first = 0
    for segment in scenario.segments:
        recorded = round(segment.duration_s / run.dt_s) // run.control_steps
        windows.append(slice(first, first + recorded))
        first += recorded
    return tuple(windows)


def _final_interval_energy_residual(run: ScenarioRun, config: KernelConfig) -> float:
    """The wheel-boundary energy residual of the run's final control interval.

    The identity the scenario module documents - the change in chassis and wheel kinetic
    energy against the work of the wheel torques, aerodynamic drag and tyre slip, each
    term at midpoint velocity and wheel speed - recomputed here straight from the run's own
    trace, torque histories and applied kernel forces rather than through the runner's
    helper, so the recorded final-row value is checked against the documented quantities
    and not against itself.

    The interval is the last complete one: ``[steps - control_steps, steps]``, which ends
    on the run's terminal state, because no interval starts there.
    """
    count = run.control_steps
    start = run.steps - count
    trace = run.trace
    values = forces.validated_config_scalars(config, "test_scenarios")
    kinetic_change = (
        0.5
        * config.mass_kg
        * (
            trace[start + count, longitudinal.V_INDEX] ** 2
            + trace[start + count, longitudinal.VY_INDEX] ** 2
            - trace[start, longitudinal.V_INDEX] ** 2
            - trace[start, longitudinal.VY_INDEX] ** 2
        )
    )
    kinetic_change += (
        0.5
        * config.yaw_inertia_kg_m2
        * (
            trace[start + count, longitudinal.YAW_RATE_INDEX] ** 2
            - trace[start, longitudinal.YAW_RATE_INDEX] ** 2
        )
    )
    kinetic_change += (
        0.5
        * values["wheel_inertia_kg_m2"]
        * math.fsum(
            float(trace[start + count, column] ** 2 - trace[start, column] ** 2)
            for column in range(
                longitudinal.WHEEL_STATE_OFFSET,
                longitudinal.WHEEL_STATE_OFFSET + forces.WHEEL_COUNT,
            )
        )
    )
    torque_work = 0.0
    drag_work = 0.0
    tyre_slip_work = 0.0
    for index in range(start, start + count):
        before = trace[index]
        after = trace[index + 1]
        yaw_mid = 0.5 * (
            float(before[longitudinal.YAW_RATE_INDEX]) + float(after[longitudinal.YAW_RATE_INDEX])
        )
        speed_before = math.hypot(
            float(before[longitudinal.V_INDEX]), float(before[longitudinal.VY_INDEX])
        )
        speed_after = math.hypot(
            float(after[longitudinal.V_INDEX]), float(after[longitudinal.VY_INDEX])
        )
        speed_mid = 0.5 * (speed_before + speed_after)
        vx_mid = 0.5 * (float(before[longitudinal.V_INDEX]) + float(after[longitudinal.V_INDEX]))
        vy_mid = 0.5 * (float(before[longitudinal.VY_INDEX]) + float(after[longitudinal.VY_INDEX]))
        _downforce_n, drag_n = forces.aero_forces(
            speed_before,
            values["air_density_kg_m3"],
            values["reference_area_m2"],
            config.aero_speed_m_s,
            config.cl,
            config.cd,
        )
        drag_work += drag_n * speed_mid * config.dt_s
        road_steer_deg = np.zeros(forces.WHEEL_COUNT, dtype=np.float64)
        steering.road_wheel_angles_deg(
            float(run.steer_wheel_deg[index]),
            config.steering_ratio,
            config.wheelbase_m,
            float(config.axle_track_m[0]),
            config.ackermann_fraction,
            road_steer_deg,
        )
        for wheel in range(forces.WHEEL_COUNT):
            column = longitudinal.WHEEL_STATE_OFFSET + wheel
            omega_mid = 0.5 * (float(before[column]) + float(after[column]))
            torque_work += (
                (
                    forces.wheel_drive_torque_nm(wheel, float(run.drive_torque_nm[index]))
                    + run.brake_torque_nm[index, wheel]
                )
                * omega_mid
                * config.dt_s
            )
            axle = wheel // forces.WHEELS_PER_AXLE_COUNT
            side = 1.0 if wheel % forces.WHEELS_PER_AXLE_COUNT == 0 else -1.0
            corner_x = config.cg_to_front_axle_m if axle == 0 else -config.cg_to_rear_axle_m
            corner_y = side * 0.5 * float(config.axle_track_m[axle])
            patch_vx, patch_vy = kinematics.contact_velocity_m_s(
                vx_mid, vy_mid, yaw_mid, corner_x, corner_y
            )
            steer_deg = float(road_steer_deg[wheel]) + float(
                config.axle_bump_steer_deg_per_m[axle] * run.step_outputs.travel_m[index, wheel]
            )
            wheel_vx, wheel_vy = kinematics.wheel_frame_velocity_m_s(patch_vx, patch_vy, steer_deg)
            tyre_slip_work += (
                float(run.step_outputs.force_x_n[index, wheel])
                * (omega_mid * values["rolling_radius_m"] - wheel_vx)
                - float(run.step_outputs.force_y_n[index, wheel]) * wheel_vy
            ) * config.dt_s
    accounted_work = torque_work + drag_work - tyre_slip_work
    scale = max(abs(kinetic_change), abs(accounted_work), 1.0)
    return abs(kinetic_change - accounted_work) / scale


def _clutch_transmitted_power(run: ScenarioRun, window: slice) -> np.ndarray:
    """The mechanical power crossing the clutch, per recorded step of ``window``.

    The clutch's output is the driveline torque against the wheel speeds - the rear-axle
    boundary C9.2.5 states the demand in - read at each recorded step's first kernel step
    from the run's own trace and torque histories. The front wheels are undriven (C9.1.1),
    so their share of the driveline torque is exactly zero and only the rear pair
    contributes.
    """
    rows = np.arange(window.start, window.stop) * run.control_steps
    transmitted_w = np.empty(rows.size)
    for index, row in enumerate(rows):
        omega = run.trace[
            row,
            longitudinal.WHEEL_STATE_OFFSET : longitudinal.WHEEL_STATE_OFFSET + forces.WHEEL_COUNT,
        ]
        transmitted_w[index] = math.fsum(
            float(forces.wheel_drive_torque_nm(wheel, float(run.drive_torque_nm[row])))
            * float(omega[wheel])
            for wheel in range(forces.WHEEL_COUNT)
        )
    return transmitted_w


# ------------------------------------------------------------------ behaviour: launch


def test_a_standing_launch_starts_from_rest_in_first_gear(runs: Mapping[str, ScenarioRun]) -> None:
    run = runs["standing_launch"]
    steps = run.record.ground_truth
    assert steps[0].vx_m_s == 0.0
    assert steps[0].gear == 1
    assert run.gears == (1,) * len(run.gears), "rpm alone must not move the gearbox (C9.8.1)"
    assert run.record.ground_truth[-1].vx_m_s > 0.0
    assert float(run.drivetrain.ice_power_w.max()) > 0.0, (
        "the ICE delivers shaft power from the declared launch speed"
    )
    assert float(run.drivetrain.accel_m_s2.max()) > 0.0


def test_thermal_scenarios_publish_finite_temperature_and_pressure_channels(
    runs: Mapping[str, ScenarioRun],
) -> None:
    for name in ("thermal_soak", "brake_duty_cycle"):
        frames = runs[name].record.frames
        assert frames
        for frame in frames:
            assert np.isfinite(
                [
                    frame.values["tyre_temp_fl"],
                    frame.values["tyre_pressure_fl"],
                    frame.values["brake_temp_fl"],
                    frame.values["engine_temp"],
                    frame.values["gearbox_temp"],
                ]
            ).all()
    brake = runs["brake_duty_cycle"].record.series("brake_temp_fl")
    assert max(brake) > brake[0]


def test_control_law_receives_scenario_pose_and_controls_interval(config: KernelConfig) -> None:
    states: list[scenarios.ControlState] = []
    plan = scenarios.Scenario(
        name="control_law_pose",
        initial_speed_m_s=10.0,
        initial_x_m=4.0,
        initial_y_m=7.0,
        initial_heading_rad=0.2,
        description="control-law boundary",
        segments=(scenarios.ScenarioSegment(duration_s=0.02),),
    )

    def law(state: scenarios.ControlState) -> scenarios.DriverRequest:
        states.append(state)
        return scenarios.DriverRequest(0.0, 0.25, 0.0, 20.0)

    run = scenarios.run_scenario(config, plan, control_steps=100, control_law=law)
    assert len(states) == 2
    assert (states[0].x_m, states[0].y_m) == (4.0, 7.0)
    assert states[0].heading_rad == 0.2
    assert run.drivetrain.throttle[0] == 0.25


def test_control_law_brake_maps_to_signed_wheel_torque(config: KernelConfig) -> None:
    plan = scenarios.Scenario(
        name="control_law_brake",
        initial_speed_m_s=20.0,
        description="normalized brake boundary",
        segments=(scenarios.ScenarioSegment(duration_s=0.01),),
    )
    request = scenarios.DriverRequest(0.0, 0.0, 0.5, 0.0)
    run = scenarios.run_scenario(
        config,
        plan,
        control_steps=100,
        control_law=lambda _: request,
        max_brake_torque_nm=100.0,
    )
    assert np.all(run.brake_torque_nm == -50.0)


def test_each_segment_applies_its_caller_supplied_clutch_state(
    runs: Mapping[str, ScenarioRun], suite: Mapping[str, Scenario]
) -> None:
    run = runs["standing_launch"]
    for window, segment in zip(
        _windows(run, suite[run.name]), suite[run.name].segments, strict=True
    ):
        assert np.all(run.drivetrain.clutch[window] == segment.clutch)


def test_a_caller_supplied_initial_engine_speed_advances_during_clutch_slip(
    config: KernelConfig,
) -> None:
    """A launch seeds a crank-speed state that evolves under delivered and clutch load torque.

    While the clutch slips the crank is not speed-locked to the wheels. The start telemetry is an
    initial condition, not a speed hold: subsequent rpm follows the configured ICE inertia and
    crank torque balance.
    """
    declared = 12_000.0

    def _run(override: float | None) -> ScenarioRun:
        segment = scenarios.ScenarioSegment(
            duration_s=0.5,
            throttle=0.5,
            clutch=0.5,
            grid_standing_start=True,
            ice_rpm_initial=override,
        )
        plan = scenarios.Scenario("slipping_launch", 0.0, (segment,), "declared launch rpm")
        return scenarios.run_scenario(config, plan)

    with_override = _run(declared)
    without = _run(None)

    assert with_override.drivetrain.ice_rpm[0] == declared
    assert with_override.drivetrain.ice_rpm[1] != declared, (
        "the declared telemetry speed seeds the engine state; it is not an override held for "
        "the whole slipping segment"
    )
    assert without.drivetrain.ice_rpm[1] >= config.idle_rpm, (
        "an unseeded engine still evolves from configured idle instead of being re-derived from "
        "the stationary wheels"
    )
    ratio = config.gear_ratios[0] * config.final_drive
    expected = engine.step_engine_speed(
        config,
        declared,
        float(with_override.drivetrain.ice_torque_nm[0]),
        float(with_override.drivetrain.drive_torque_nm[0]) / ratio,
        dt_s=with_override.control_steps * with_override.dt_s,
    )
    assert with_override.drivetrain.ice_rpm[1] == pytest.approx(expected)
    assert max(with_override.gears) == 1, "a declared 12 000 rpm is below the upshift point"
    assert float(with_override.drivetrain.ice_power_w.max()) > float(
        without.drivetrain.ice_power_w.max()
    ), "a faster engine at the same pedal delivers more shaft power"
    record_rpm = np.array([frame.values["ice_rpm"] for frame in with_override.record.frames])
    assert record_rpm[0] == declared, "the published channel carries the initial crank speed"
    assert all(step.mgu_k_power_w == 0.0 for step in with_override.record.ground_truth), (
        "no motor was asked for anything, so no store energy moved"
    )


def test_the_built_launch_scenarios_seed_the_start_telemetry_engine_speed(
    runs: Mapping[str, ScenarioRun], suite: Mapping[str, Scenario], config: KernelConfig
) -> None:
    """Both real launches seed telemetry rpm only during the initial slipping segment.

    ``ScenarioSegment.ice_rpm_initial`` is an initial condition for the rotational state. It is
    legal only for a grid-start segment with a slipping clutch; the state then evolves and fully
    engaged segments couple engine speed to the wheels.
    """
    assert scenarios.LAUNCH_ICE_RPM == 12_000.0
    assert config.idle_rpm < scenarios.LAUNCH_ICE_RPM <= config.rev_limit_rpm, (
        "the declared launch speed has to be a speed this car's engine is configured to reach"
    )
    for name in ("standing_launch", "accelerate_to_speed"):
        run = runs[name]
        scenario = suite[name]
        assert any(segment.ice_rpm_initial is None for segment in scenario.segments), (
            f"{name} declares the engine speed for the whole run, not for its launch"
        )
        published = np.array([frame.values["ice_rpm"] for frame in run.record.frames])
        for window, segment in zip(_windows(run, scenario), scenario.segments, strict=True):
            if segment.ice_rpm_initial is None:
                continue
            assert segment.grid_standing_start, "only a grid-start segment may declare launch rpm"
            assert segment.clutch < 1.0, (
                "the override is only valid while the clutch slips and decouples engine speed "
                "from the wheels"
            )
            assert segment.ice_rpm_initial == scenarios.LAUNCH_ICE_RPM
            assert run.drivetrain.ice_rpm[window.start] == scenarios.LAUNCH_ICE_RPM, (
                "the first sample reports the declared engine state, not the wheel-derived one"
            )
            assert np.all(np.isfinite(run.drivetrain.ice_rpm[window]))
            assert np.any(run.drivetrain.ice_rpm[window] != scenarios.LAUNCH_ICE_RPM), (
                "and later samples evolve from the initial condition instead of holding it"
            )
            assert published[window.start] == scenarios.LAUNCH_ICE_RPM


def test_the_zero_to_one_hundred_time_is_measured_against_the_cited_coarse_reference(
    runs: Mapping[str, ScenarioRun],
) -> None:
    """The crossing is measured and reported against the reference in ``docs/calibration.md`` §6.

    That reference is a median of six ~3.7 Hz telemetry crossings: a coarse observation with no
    published figure behind it, whose ±0.30 s is the feed's quantisation rather than a tolerance
    on this car. So this test **reports** the gap instead of asserting a pass, and there is no
    band here to widen - the P1 performance gate stays open in `docs/calibration.md` until someone
    measures both numbers deliberately and records the result.
    """
    run = runs["accelerate_to_speed"]
    crossing = np.flatnonzero(run.trace[:, 1] >= 100.0 / 3.6)
    assert run.trace[0, 1] == 0.0, "the run starts from rest, so the window includes the launch"
    assert crossing.size, "the scenario must reach 100 km/h for the crossing to be measured"
    measured_s = float(crossing[0] * run.dt_s)
    gap_s = measured_s - ZERO_TO_HUNDRED_REFERENCE_S
    print(
        f"0-100 km/h measured {measured_s:.4f} s from rest against the cited "
        f"{ZERO_TO_HUNDRED_REFERENCE_S} s reference (PLAN.md section 11.1): "
        f"{gap_s:+.4f} s. Reported, not asserted - the reference carries no tolerance."
    )


def test_full_throttle_reaches_top_gear_and_ends_in_an_ice_only_tail(
    runs: Mapping[str, ScenarioRun], suite: Mapping[str, Scenario], config: KernelConfig
) -> None:
    """The terminal speed is measured over a stretch where the motor asked for nothing.

    C5.2.7's cap, the store's 4 MJ window and the gearbox's caller-owned shift requests are
    unchanged; what changed is that the last segment of this run is explicitly motor-free, so the
    number quoted as terminal speed is a settled ICE-only asymptote rather than a boost still being
    spent.
    """
    run = runs["full_throttle"]
    scenario = suite["full_throttle"]
    windows = _windows(run, scenario)
    tail = scenario.segments[-1]
    assert max(run.gears) == config.gear_ratios.size, "the requests are what put it in eighth"
    assert tail.mgu_k_request_nm == 0.0, "the terminal tail asks the motor for nothing"
    assert tail.request is gearbox.GearRequest.HOLD, "and asks for no further gear"
    assert np.all(run.drivetrain.mgu_k_power_w[windows[-1]] == 0.0), (
        "no store energy moves during the terminal tail"
    )
    assert tail.duration_s >= 40.0, (
        "the tail has to be long enough to settle; a shorter one measures a boost, not an asymptote"
    )


def test_the_bounded_mgu_k_deployment_sits_in_the_high_speed_window(
    runs: Mapping[str, ScenarioRun], suite: Mapping[str, Scenario], config: KernelConfig
) -> None:
    """Deployment where the run is meant to reach a high speed, and nowhere else.

    Top gear is where a full C5.2.11 deployment is transmissible at all (see the module docstring).
    The longer request keeps deployment available through the high-speed stretch; the step still
    clamps delivery to charge, power, torque and relative-speed limits.
    """
    run = runs["full_throttle"]
    scenario = suite["full_throttle"]
    windows = _windows(run, scenario)
    deployment = [
        i for i, segment in enumerate(scenario.segments) if segment.mgu_k_request_nm > 0.0
    ]
    assert len(deployment) == 1, "one bounded deployment request, not a motor-assisted run"
    index = deployment[0]
    assert 0 < index < len(scenario.segments) - 1, "the deployment is followed by an ICE-only tail"
    assert scenario.segments[index].mgu_k_request_nm == pytest.approx(
        config.mgu_k_torque_limit_nm / config.mgu_k_crankshaft_ratio, rel=1e-12
    ), "the request is C5.2.11's crank-referenced limit at the shaft, not a new number"
    assert scenario.segments[index].duration_s == 20.0, (
        "the deployment request covers the high-speed stretch"
    )
    assert all(
        int(run.drivetrain.gear[window][0]) == config.gear_ratios.size for window in windows[index:]
    ), "the deployment happens in top gear, where the motor can actually be used"
    speeds_km_h = np.array([step.vx_m_s for step in run.record.ground_truth]) * 3.6
    assert speeds_km_h[windows[index]].min() > config.launch_speed_kmh, (
        "C5.2.12's launch block is not in play this far up, so the deployment is the motor's own"
    )
    power = run.drivetrain.mgu_k_power_w
    assert np.any(power[windows[index]] > 0.0), (
        "a deployment spends the store, in the positive direction"
    )
    cap_w = config.mgu_k_peak_power_kw * 1_000.0
    assert float(power[windows[index]].max()) <= cap_w, "C5.2.7's 350 kW"
    soc = run.drivetrain.soc_mj
    assert soc[windows[index].start] > soc[windows[index].stop - 1]
    assert np.all(soc >= 0.0) and np.all(soc <= config.store_energy_mj), "C5.2.9's window"
    outside = np.concatenate([power[: windows[index].start], power[windows[index].stop :]])
    assert np.all(outside == 0.0), "no other segment of this run moves the store"


def test_the_transient_maximum_speed_is_measured_apart_from_the_terminal_speed(
    runs: Mapping[str, ScenarioRun], suite: Mapping[str, Scenario]
) -> None:
    """Two quantities, two stretches, neither standing in for the other.

    The reachability floor in `PLAN.md` §11.1 is compared against the **maximum** speed the run
    reaches, deployment included; the terminal speed is a separate result with no event-trap
    target. This test measures and reports both separately, and checks only that the transient
    maximum clears the cited reachability floor; that floor is not a terminal-speed target.
    """
    run = runs["full_throttle"]
    scenario = suite["full_throttle"]
    tail_window = _windows(run, scenario)[-1]
    transient_km_h = float(run.trace[:, 1].max()) * 3.6
    terminal_km_h = float(run.trace[-1, 1]) * 3.6
    tail_km_h = float(run.record.ground_truth[tail_window.start].vx_m_s) * 3.6
    deployment = next(
        window
        for segment, window in zip(scenario.segments, _windows(run, scenario), strict=True)
        if segment.mgu_k_request_nm > 0.0
    )
    deployment_power_kw = run.drivetrain.mgu_k_power_w[deployment] / 1_000.0
    deployment_soc_mj = run.drivetrain.soc_mj[deployment]
    last_five = run.trace[-round(5.0 / run.dt_s) :, 1]
    drift_km_h = float((last_five[-1] - last_five[0]) * 3.6)
    print(
        f"full_throttle transient maximum {transient_km_h:.4f} km/h (the reachability-floor "
        f"quantity, {transient_km_h - TOP_SPEED_REACHABILITY_KMH:+.4f} km/h against the cited "
        f"{TOP_SPEED_REACHABILITY_KMH} km/h floor), terminal {terminal_km_h:.4f} km/h after a "
        f"{tail_km_h:.4f} km/h ICE-only tail entry, MGU-K peaked at "
        f"{float(deployment_power_kw.max()):.2f} kW with store "
        f"{float(deployment_soc_mj[0]):.3f} to {float(deployment_soc_mj[-1]):.3f} MJ, "
        f"{drift_km_h:+.4f} km/h over its last five seconds. The transient maximum is asserted "
        f"against its floor; terminal speed is not."
    )
    assert transient_km_h >= TOP_SPEED_REACHABILITY_KMH, (
        "the transient maximum, not the terminal speed, must reach the cited FIA speed-table floor"
    )
    assert abs(drift_km_h) < 1.0, (
        "the last five seconds of an ICE-only tail are the settled speed, not a residual climb"
    )


def test_the_launch_grip_keeps_the_rear_tyres_inside_their_peak(
    runs: Mapping[str, ScenarioRun], config: KernelConfig
) -> None:
    """The demanded torque must stay inside what the tyre can hold, or the wheel runs away.

    With no traction control (C9.1.2) a demand above the Magic Formula's peak has no
    equilibrium: the wheel accelerates, slip grows past the peak and the force falls further
    still. That is the correct behaviour of the model, but it is not a usable scenario trace,
    so the launch is required to stay below the demand that would cause it.

    The current synthetic configuration passes this check. Keep it as a regression guard, but
    treat it as model consistency rather than launch calibration; the P1 performance gate remains
    open in `docs/calibration.md`.
    """
    run = runs["standing_launch"]
    for corner in ("RL", "RR"):
        kappa = [step.wheels[CORNERS.index(corner)].kappa for step in run.record.ground_truth]
        assert max(kappa) < 1.0, f"{corner} slip ratio reached {max(kappa)}, which is past the peak"


# ---------------------------------------------------------------- behaviour: shifting


@pytest.mark.parametrize("name", ["full_throttle_shifts", "full_throttle"])
def test_the_gearbox_moves_only_where_the_scenario_asks(
    name: str, runs: Mapping[str, ScenarioRun], suite: Mapping[str, Scenario]
) -> None:
    """Every requested change, in its own segment, and no unrequested one anywhere.

    ``full_throttle`` is in this list because it now carries a motor deployment as well as the
    upshifts: a run whose mid-segment MGU-K request could move the box would be a gearbox bug, and
    this is the assertion that says the seven requests are still the only thing that does.
    """
    run = runs[name]
    scenario = suite[name]
    gears = np.asarray(run.gears)
    changes = np.flatnonzero(np.diff(gears) != 0)
    requested = [
        i
        for i, segment in enumerate(scenario.segments)
        if segment.request is gearbox.GearRequest.UP
    ]
    assert len(changes) == len(requested)
    windows = _windows(run, scenario)
    for change, index in zip(changes.tolist(), requested, strict=True):
        assert windows[index].start <= change + 1 < windows[index].stop, (
            f"the gear changed at recorded step {change + 1}, outside segment {index}"
        )
        assert int(gears[change + 1]) == int(gears[change]) + 1, (
            "one request moves one gear (C9.8.3)"
        )
    assert gears.max() == len(requested) + 1


def test_a_requested_shift_cuts_the_driveline_for_the_configured_shift_time(
    runs: Mapping[str, ScenarioRun], suite: Mapping[str, Scenario], config: KernelConfig
) -> None:
    run = runs["full_throttle_shifts"]
    scenario = suite["full_throttle_shifts"]
    cut_steps = round(config.shift_time_s / config.dt_s)
    offset = 0
    for segment in scenario.segments:
        if segment.request is gearbox.GearRequest.UP:
            active = np.flatnonzero(run.drive_torque_nm[offset:] != 0.0)
            resumed = offset + int(active[0])
            cut = resumed - offset
            assert cut_steps <= cut <= cut_steps + 1
            assert np.all(run.drive_torque_nm[offset:resumed] == 0.0)
        offset += round(segment.duration_s / run.dt_s)
    assert config.shift_time_s <= config.shift_time_max_up_s, "C9.8.4's up-change limit"


def test_ice_power_w_is_gross_crank_shaft_power_not_clutch_transmitted(
    runs: Mapping[str, ScenarioRun], suite: Mapping[str, Scenario]
) -> None:
    """The channel's declared boundary is the crankshaft, upstream of the clutch.

    ``f1telemetry.testing.scenarios`` declares ``ice_power_w`` as the ICE's gross
    shaft power at the crankshaft - the delivered torque against the engine speed
    the drivetrain sampled - and this regression pins the two cases that boundary
    changes the most:

    * **A shift cut opens the driveline, not the engine.** The driveline torque is
      exactly zero through the cut, so the clutch-transmitted power is zero, while
      the engine keeps delivering shaft power: the channel does not collapse with
      the driveline it no longer drives.
    * **A slipping clutch decouples the two boundaries.** The standing launch seeds
      12 000 rpm, then its crank speed evolves independently of wheel speed under the
      engine inertia and reflected clutch load. The clutch-transmitted power stays
      below crankshaft power because the wheels turn much slower while they slip.

    The operational form of the boundary is checked first: at every recorded step of
    every scenario the channel is exactly ``ice_torque_nm * ice_rpm * tau / 60``.
    """
    for run in runs.values():
        shaft_w = run.drivetrain.ice_torque_nm * run.drivetrain.ice_rpm * (math.tau / 60.0)
        assert np.allclose(run.drivetrain.ice_power_w, shaft_w, rtol=1e-12, atol=0.0), (
            f"{run.name}: ice_power_w is declared as the ICE's gross crankshaft "
            "shaft power, so it must be exactly the delivered torque against the "
            "sampled engine speed at every recorded step"
        )

    # A shift cut: nothing crosses the clutch, but the engine keeps turning.
    run = runs["full_throttle_shifts"]
    scenario = suite["full_throttle_shifts"]
    offset = 0
    cuts: list[slice] = []
    for segment in scenario.segments:
        if segment.request is gearbox.GearRequest.UP:
            active = np.flatnonzero(run.drive_torque_nm[offset:] != 0.0)
            resumed = offset + int(active[0])
            cuts.append(slice(offset, resumed))
        offset += round(segment.duration_s / run.dt_s)
    assert cuts, "full_throttle_shifts must contain at least one requested shift"
    for cut in cuts:
        recorded = slice(cut.start // run.control_steps, cut.stop // run.control_steps)
        assert np.all(run.drive_torque_nm[cut] == 0.0), (
            "a requested shift cuts the driveline: the clutch transmits no torque, "
            "so the power crossing it is exactly zero"
        )
        assert np.all(run.drivetrain.ice_power_w[recorded] > 0.0), (
            "ice_power_w is declared at the crankshaft, so a shift cut does not "
            "collapse it with the driveline: the engine keeps delivering shaft "
            "power through the cut, and only drive_torque_nm against wheel speed "
            "is the clutch-transmitted quantity"
        )

    # A slipping clutch: the channel follows the engine, not the wheels.
    run = runs["standing_launch"]
    scenario = suite["standing_launch"]
    window = _windows(run, scenario)[0]
    assert run.drivetrain.ice_rpm[window.start] == scenarios.LAUNCH_ICE_RPM
    assert np.any(run.drivetrain.ice_rpm[window] != scenarios.LAUNCH_ICE_RPM)
    shaft_w = run.drivetrain.ice_power_w[window]
    transmitted_w = _clutch_transmitted_power(run, window)
    assert np.all(transmitted_w < shaft_w), (
        "while the clutch slips the wheels turn far slower than the engine, so the "
        "power crossing the clutch stays below the crankshaft power the channel "
        "reports: ice_power_w is the engine-side quantity, and the clutch boundary "
        "is drive_torque_nm against wheel speed"
    )


# ------------------------------------------------------- behaviour: coast and neutral


def test_neutral_transmits_nothing_and_first_gear_restores_the_torque(
    runs: Mapping[str, ScenarioRun], suite: Mapping[str, Scenario], config: KernelConfig
) -> None:
    run = runs["coast_neutral"]
    scenario = suite["coast_neutral"]
    windows = _windows(run, scenario)
    neutral = next(
        index
        for index, segment in enumerate(scenario.segments)
        if segment.request is gearbox.GearRequest.NEUTRAL
    )
    after = next(
        index
        for index, segment in enumerate(scenario.segments)
        if index > neutral and segment.request is gearbox.GearRequest.UP
    )
    assert scenario.segments[neutral].throttle == 1.0, "the engine is still asked for full torque"
    assert np.all(run.drivetrain.drive_torque_nm[windows[neutral]] == 0.0)
    after_shift = windows[after].start + round(config.shift_time_s / run.record_dt_s) + 1
    assert np.all(run.drivetrain.drive_torque_nm[after_shift : windows[after].stop] > 0.0)
    assert all(gear == 0 for gear in run.gears[windows[neutral]])


def test_a_coasting_car_slows_against_drag_and_the_drag_grows_with_speed(
    runs: Mapping[str, ScenarioRun], suite: Mapping[str, Scenario], config: KernelConfig
) -> None:
    run = runs["coast_neutral"]
    scenario = suite["coast_neutral"]
    neutral = next(
        index
        for index, segment in enumerate(scenario.segments)
        if segment.request is gearbox.GearRequest.NEUTRAL
    )
    window = _windows(run, scenario)[neutral]
    speeds = np.array([step.vx_m_s for step in run.record.ground_truth[window]])
    drag = np.array([step.drag_w for step in run.record.ground_truth[window]])
    assert speeds[-1] < speeds[0], "drag removes net speed over the neutral segment"
    assert np.all(drag < 0.0), "drag power is signed against forward motion"
    assert abs(drag[-1]) < abs(drag[0]), "drag power falls as the car's speed falls"
    assert abs(drag[-1] / speeds[-1]) < abs(drag[0] / speeds[0]), (
        "drag force grows approximately with speed squared"
    )
    first_trace_row = window.start * run.control_steps
    last_trace_row = window.stop * run.control_steps

    def kinetic_energy(row: int) -> float:
        state = run.trace[row]
        wheel_omega = state[
            longitudinal.WHEEL_STATE_OFFSET : longitudinal.WHEEL_STATE_OFFSET + forces.WHEEL_COUNT
        ]
        return (
            0.5
            * config.mass_kg
            * (state[longitudinal.V_INDEX] ** 2 + state[longitudinal.VY_INDEX] ** 2)
            + 0.5 * config.yaw_inertia_kg_m2 * state[longitudinal.YAW_RATE_INDEX] ** 2
            + 0.5 * config.wheel_inertia_kg_m2 * float(np.dot(wheel_omega, wheel_omega))
        )

    assert kinetic_energy(last_trace_row) < kinetic_energy(first_trace_row)


# ------------------------------------------------------------------ behaviour: braking


def test_caller_brake_torque_decelerates_the_car_and_all_four_wheels(
    runs: Mapping[str, ScenarioRun], suite: Mapping[str, Scenario]
) -> None:
    run = runs["braking"]
    scenario = suite["braking"]
    braking = next(
        index for index, segment in enumerate(scenario.segments) if segment.brake_torque_nm < 0.0
    )
    window = _windows(run, scenario)[braking]
    steps = run.record.ground_truth[window]
    assert all(step.ax_m_s2 < 0.0 for step in steps[1:])
    first, last = steps[0], steps[-1]
    assert last.vx_m_s < first.vx_m_s
    for corner in CORNERS:
        before = first.wheels[CORNERS.index(corner)]
        after = last.wheels[CORNERS.index(corner)]
        assert after.fx_n < 0.0, f"{corner} produces no braking force"
        assert after.kappa < 0.0, f"{corner} does not slow relative to the road"
        assert after.fx_n != before.fx_n


# ------------------------------------------------------------------- behaviour: MGU-K


def test_the_standing_start_rule_blocks_mgu_deployment_below_fifty_km_h(
    runs: Mapping[str, ScenarioRun], suite: Mapping[str, Scenario], config: KernelConfig
) -> None:
    run = runs["mgu_k_standing_start"]
    scenario = suite["mgu_k_standing_start"]
    blocked = next(
        index
        for index, segment in enumerate(scenario.segments)
        if segment.grid_standing_start and segment.mgu_k_request_nm > 0.0
    )
    window = _windows(run, scenario)[blocked]
    speeds_km_h = np.array([step.vx_m_s for step in run.record.ground_truth[window]]) * 3.6
    assert np.all(speeds_km_h < config.launch_speed_kmh), "the rule only applies below 50 km/h"
    assert np.all(run.drivetrain.mgu_k_power_w[window] == 0.0), (
        "C5.2.12 blocks positive MGU-K torque in a declared grid standing start"
    )
    assert np.ptp(run.drivetrain.soc_mj[window]) == 0.0, "a blocked motor moves no energy"


def test_mgu_k_deployment_drains_the_store_and_regeneration_refills_it(
    runs: Mapping[str, ScenarioRun], suite: Mapping[str, Scenario], config: KernelConfig
) -> None:
    run = runs["mgu_k_deploy_regen"]
    scenario = suite["mgu_k_deploy_regen"]
    windows = _windows(run, scenario)
    deploy = next(
        index
        for index, segment in enumerate(scenario.segments)
        if segment.mgu_k_request_nm > 0.0 and not segment.grid_standing_start
    )
    regen = next(
        index for index, segment in enumerate(scenario.segments) if segment.mgu_k_request_nm < 0.0
    )
    soc = run.drivetrain.soc_mj
    assert soc[windows[deploy].start] > soc[windows[deploy].stop - 1], (
        "deployment does not drain the store"
    )
    assert np.all(run.drivetrain.mgu_k_power_w[windows[deploy]] > 0.0)
    assert soc[windows[regen].stop - 1] > soc[windows[regen].start], (
        "regeneration does not refill the store"
    )
    assert np.all(run.drivetrain.mgu_k_power_w[windows[regen]] < 0.0)
    assert np.all(run.drivetrain.lap_recharge_mj[windows[regen]] > 0.0), "C5.2.10's budget is used"
    assert np.all(soc >= 0.0) and np.all(soc <= config.store_energy_mj), (
        "the store stays inside C5.2.9's window"
    )


def test_mgu_k_power_is_reported_in_watts_and_is_driven_against_the_electrical_cap(
    runs: Mapping[str, ScenarioRun], config: KernelConfig
) -> None:
    """The reported store-side power must be the C5.2.7 quantity, in watts.

    ``-d(SOC)/dt`` is MJ/s over a buffer C5.2.9 states in MJ, so the W-named column was reporting
    a millionth of the power it claimed. Dividing that by 1 000 to compare it with a kW cap is what
    let the test pass while proving nothing: 0.35 was read as 0.35 kW against a 350 kW cap. A cap
    assertion is only worth anything if the value it measures reaches the cap.
    """
    run = runs["mgu_k_deploy_regen"]
    power_w = np.abs(run.drivetrain.mgu_k_power_w)
    cap_w = config.mgu_k_peak_power_kw * 1_000.0

    assert float(power_w.max()) <= cap_w, "C5.2.7's 350 kW"
    assert float(power_w.max()) >= 0.99 * cap_w, (
        "the motor is asked for C5.2.11's full crank-referenced torque at the run's own engine "
        "speed, so the store-side power has to reach the cap. A MJ/s figure in a W column is a "
        "millionth of this and the assertion above would pass without it moving anything"
    )
    regen_w = -float(run.drivetrain.mgu_k_power_w.min())
    assert regen_w >= 0.5 * cap_w, "regeneration moves real energy too"


def test_mgu_k_power_matches_the_stored_energy_it_claims_to_measure(
    runs: Mapping[str, ScenarioRun],
) -> None:
    """Cross-check the watts against the store trace, so the unit is fixed by two quantities.

    The SOC column and the power column are recorded from the same buffer, so the one has to be
    the time integral of the other in watts. This ties the two together independently of the cap,
    which both would pass if the units were wrong in the same way.
    """
    run = runs["mgu_k_deploy_regen"]
    soc = run.drivetrain.soc_mj
    # A recorded step spans control_steps kernel steps and the SOC delta is per step, so the
    # interval power is the whole SOC change over the interval that produced it.
    change_mj = np.diff(soc)
    elapsed_s = np.full(change_mj.shape, run.record_dt_s)
    expected_w = -change_mj / elapsed_s * 1.0e6
    reported_w = run.drivetrain.mgu_k_power_w[:-1]

    moving = np.abs(change_mj) > 0.0
    assert moving.sum() > len(change_mj) // 2, "most of the run must actually move the store"
    assert np.allclose(reported_w[moving], expected_w[moving], rtol=0.02, atol=1.0)
    assert np.all(np.sign(reported_w) == np.sign(-change_mj)), (
        "deployment spends the store and regeneration fills it, in the same direction of sign"
    )


# --------------------------------------------------------------- invariants and records


def test_p1_invariants_pass_on_the_produced_scenario_records(
    runs: Mapping[str, ScenarioRun], spec: CarSpec, suite: Mapping[str, Scenario]
) -> None:
    p1_invariants = (1, 6, 7, 8)
    for name, run in runs.items():
        for result in run_all(run.record, spec, p1_invariants):
            assert result.passed, f"{name}: {result.summary()}"


def test_p2_invariants_pass_on_produced_scenario_records(
    runs: Mapping[str, ScenarioRun], spec: CarSpec
) -> None:
    for name, run in runs.items():
        for result in run_all(run.record, spec, (2, 3, 4)):
            assert result.passed, f"{name}: {result.summary()}"


def test_invariant_seven_accepts_legal_neutral_selection(
    runs: Mapping[str, ScenarioRun], spec: CarSpec
) -> None:
    for run in runs.values():
        assert not check_gearbox_progression(run.record, spec), run.name
    assert any(gear == 0 for gear in runs["coast_neutral"].gears)


def test_the_energy_invariant_gates_the_actual_chassis_and_wheel_energy_balance(
    runs: Mapping[str, ScenarioRun], spec: CarSpec
) -> None:
    for run in runs.values():
        assert not check_energy_balance(run.record, spec), run.name
        fractions = [step.energy_residual_fraction for step in run.record.ground_truth]
        assert all(value is not None and np.isfinite(value) for value in fractions), run.name
        assert max(value for value in fractions if value is not None) < 0.01, run.name


def test_energy_invariant_rejects_a_real_run_record_over_one_percent(
    runs: Mapping[str, ScenarioRun], spec: CarSpec
) -> None:
    from dataclasses import replace

    run = runs["standing_launch"]
    steps = list(run.record.ground_truth)
    steps[1] = replace(steps[1], energy_residual_fraction=0.0101)
    record = replace(run.record, ground_truth=tuple(steps))
    result = check_energy_balance(record, spec)
    assert len(result) == 1
    assert result[0].where == "step 1"


def test_the_final_intervals_energy_residual_is_computed_not_fabricated(
    runs: Mapping[str, ScenarioRun], config: KernelConfig
) -> None:
    """The last row's residual is the final interval's, computed from its own work.

    The run's terminal state has no interval after it, so the last recorded
    row reports the residual of the final complete control interval - the one
    that *ends* on that row, which is the same interval the row before it
    describes. That value has to be the wheel-boundary identity computed from
    the interval's own work and energy, not a hard-coded pass, and this test
    holds it to all three of the things a fabricated zero is not: it is the
    same computed value the previous row carries, it is nonzero because a real
    interval's identity does not close to exactly zero in floating point, and
    it matches the identity recomputed here from the documented quantities.
    """
    for run in runs.values():
        steps = run.record.ground_truth
        final = steps[-1].energy_residual_fraction
        previous = steps[-2].energy_residual_fraction
        assert final is not None and math.isfinite(final), run.name
        assert final == previous, (
            f"{run.name}: the final row sits on the terminal state, where no "
            "interval starts, so it must report the final complete interval's "
            f"residual - the same value the previous row carries - not a "
            f"fabricated pass (recorded {final!r} against {previous!r})"
        )
        assert final == pytest.approx(
            _final_interval_energy_residual(run, config), rel=1e-6, abs=1e-10
        ), (
            f"{run.name}: the final row's residual must be the wheel-boundary "
            "identity recomputed from the final interval's own kinetic-energy "
            "change, wheel-torque work, drag work and tyre-slip work"
        )


def test_the_published_channels_are_contract_channels_inside_their_declared_ranges(
    runs: Mapping[str, ScenarioRun],
) -> None:
    for name, run in runs.items():
        for channel in run.record.channels:
            assert channel in CHANNELS, (
                f"{name} publishes {channel!r}, which is not in the contract"
            )
            spec = CHANNELS[channel]
            assert spec.dtype in DTYPES
            values = run.record.series(channel)
            assert min(values) >= spec.range_min, f"{name}.{channel} below {spec.range_min}"
            assert max(values) <= spec.range_max, f"{name}.{channel} above {spec.range_max}"


def test_a_record_is_an_even_decimation_of_the_kernel_trace(
    runs: Mapping[str, ScenarioRun],
) -> None:
    for run in runs.values():
        assert run.record.dt_s == run.control_steps * run.dt_s
        assert len(run.record) == run.steps // run.control_steps + 1
        assert run.trace.shape == (run.steps + 1, run.trace.shape[1])
        assert [step.t_s for step in run.record.ground_truth] == pytest.approx(
            [index * run.record.dt_s for index in range(len(run.record))]
        )


def test_scenario_truth_records_applied_load_force_and_travel(
    runs: Mapping[str, ScenarioRun],
) -> None:
    """Recorded corner truth comes from the applied kernel step, including terminal hold."""
    run = runs["standing_launch"]
    for record_index in (0, 1, len(run.record) - 1):
        step = run.record.ground_truth[record_index]
        trace_row = record_index * run.control_steps
        output_row = min(trace_row, run.steps - 1)
        assert tuple(wheel.fz_n for wheel in step.wheels) == tuple(
            float(value) for value in run.step_outputs.load_n[output_row]
        )
        assert tuple(wheel.fx_n for wheel in step.wheels) == tuple(
            float(value) for value in run.step_outputs.force_x_n[output_row]
        )
        assert step.suspension_travel_m == tuple(
            float(value) for value in run.step_outputs.travel_m[output_row]
        )
        assert step.travel_limited == tuple(
            bool(value) for value in run.step_outputs.travel_limited[output_row]
        )


def test_suspension_stays_within_configured_limits_for_every_scenario_step(
    runs: Mapping[str, ScenarioRun],
) -> None:
    for name, run in runs.items():
        assert all(not any(step.travel_limited) for step in run.record.ground_truth), (
            f"{name} reached a configured suspension travel limit"
        )


def test_steered_scenario_records_lateral_truth_and_contract_channels(
    config: KernelConfig,
) -> None:
    plan = scenarios.Scenario(
        name="steered_record",
        initial_speed_m_s=20.0,
        initial_gear=3,
        description="Short steering truth regression.",
        segments=(scenarios.ScenarioSegment(duration_s=0.1, steer_wheel_deg=10.0),),
    )

    run = scenarios.run_scenario(config, plan)
    step = run.record.ground_truth[1]
    frame = run.record.frames[1].values

    assert step.steer_rad == pytest.approx(math.radians(10.0))
    assert step.yaw_rate_rad_s > 0.0
    assert any(wheel.fy_n != 0.0 for wheel in step.wheels)
    assert tuple(wheel.fy_n for wheel in step.wheels) == tuple(
        float(value) for value in run.step_outputs.force_y_n[run.control_steps]
    )
    assert frame["steering_angle"] == pytest.approx(10.0)
    assert frame["yaw_rate"] == pytest.approx(math.degrees(step.yaw_rate_rad_s))
    assert frame["slip_angle_fl"] == pytest.approx(math.degrees(step.wheels[0].alpha_rad))
    assert tuple(wheel.camber_deg for wheel in step.wheels) == tuple(
        float(value) for value in run.step_outputs.camber_deg[run.control_steps]
    )
    for corner, wheel in zip(CORNERS, step.wheels, strict=True):
        assert frame[f"camber_{corner.lower()}"] == pytest.approx(wheel.camber_deg)
    final_residual = run.record.ground_truth[-1].energy_residual_fraction
    assert final_residual == pytest.approx(
        _final_interval_energy_residual(run, config), rel=1e-6, abs=1e-10
    )


def test_steady_state_circle_settles_to_its_requested_radius(
    runs: Mapping[str, ScenarioRun], config: KernelConfig
) -> None:
    run = runs["steady_state_circle"]
    tail = run.record.ground_truth[-15:]
    speed = math.fsum(math.hypot(step.vx_m_s, step.vy_m_s) for step in tail) / len(tail)
    yaw_rate = math.fsum(step.yaw_rate_rad_s for step in tail) / len(tail)
    lateral_g = math.fsum(step.ay_m_s2 for step in tail) / len(tail) / config.gravity_m_s2

    assert speed / yaw_rate == pytest.approx(50.0, rel=0.05)
    assert 0.7 < lateral_g < 0.9
    assert all(not any(step.travel_limited) for step in tail)


def test_constant_radius_speed_sweep_matches_radius_and_settles_lateral_acceleration(
    config: KernelConfig,
    spec: CarSpec,
) -> None:
    sweep = scenarios.run_constant_radius_speed_sweep(config)
    lateral_g: list[float] = []
    for run in sweep:
        for channel in run.record.channels:
            channel_spec = CHANNELS[channel]
            values = run.record.series(channel)
            assert min(values) >= channel_spec.range_min, (
                f"{run.name}.{channel} below {channel_spec.range_min}"
            )
            assert max(values) <= channel_spec.range_max, (
                f"{run.name}.{channel} above {channel_spec.range_max}"
            )
        assert all(not any(step.travel_limited) for step in run.record.ground_truth), (
            f"{run.name} reached a configured suspension travel limit"
        )
        tail_steps = max(1, round(0.15 / run.record.dt_s))
        tail = run.record.ground_truth[-tail_steps:]
        speed = math.fsum(math.hypot(step.vx_m_s, step.vy_m_s) for step in tail) / len(tail)
        yaw_rate = math.fsum(step.yaw_rate_rad_s for step in tail) / len(tail)
        lateral_acceleration = math.fsum(step.ay_m_s2 for step in tail) / len(tail)
        lateral_g.append(lateral_acceleration / config.gravity_m_s2)
        assert speed / yaw_rate == pytest.approx(200.0, abs=3.0)
        assert lateral_acceleration == pytest.approx(speed * yaw_rate, rel=0.05)
        for result in run_all(run.record, spec, (1, 2, 3, 4, 6)):
            assert result.passed, f"{run.name}: {result.summary()}"

    assert np.all(np.diff(lateral_g) > 0.0)


def test_zero_steer_symmetry_control_has_no_camber_or_bump_steer(
    config: KernelConfig, spec: CarSpec
) -> None:
    control_config = replace(
        config,
        axle_static_camber_deg=np.zeros_like(config.axle_static_camber_deg),
        axle_camber_gain_deg_per_m=np.zeros_like(config.axle_camber_gain_deg_per_m),
        axle_bump_steer_deg_per_m=np.zeros_like(config.axle_bump_steer_deg_per_m),
    )
    plan = scenarios.Scenario(
        name="zero_steer_symmetry_control",
        initial_speed_m_s=20.0,
        initial_gear=0,
        description="Symmetric control with no camber or bump steer.",
        segments=(scenarios.ScenarioSegment(duration_s=0.1),),
    )
    run = scenarios.run_scenario(control_config, plan)
    result = run_all(run.record, spec, (5,))[0]
    assert result.passed, result.summary()


def test_left_and_right_steering_runs_are_mirror_symmetric(config: KernelConfig) -> None:
    symmetric_config = replace(
        config,
        axle_static_camber_deg=np.zeros_like(config.axle_static_camber_deg),
        axle_camber_gain_deg_per_m=np.zeros_like(config.axle_camber_gain_deg_per_m),
        axle_bump_steer_deg_per_m=np.zeros_like(config.axle_bump_steer_deg_per_m),
    )

    def run_turn(steer_wheel_deg: float) -> ScenarioRun:
        plan = scenarios.Scenario(
            name="mirrored_turn",
            initial_speed_m_s=20.0,
            initial_gear=0,
            description="Matched steering input for the paired-turn symmetry check.",
            segments=(scenarios.ScenarioSegment(0.1, steer_wheel_deg=steer_wheel_deg),),
        )
        return scenarios.run_scenario(symmetric_config, plan)

    left_turn = run_turn(5.0).record.ground_truth
    right_turn = run_turn(-5.0).record.ground_truth
    assert len(left_turn) == len(right_turn)
    for left_step, right_step in zip(left_turn, right_turn, strict=True):
        assert left_step.vx_m_s == pytest.approx(right_step.vx_m_s, rel=1e-7, abs=1e-9)
        assert left_step.vy_m_s == pytest.approx(-right_step.vy_m_s, rel=1e-6, abs=1e-8)
        assert left_step.yaw_rate_rad_s == pytest.approx(
            -right_step.yaw_rate_rad_s, rel=1e-6, abs=1e-8
        )
        assert left_step.ay_m_s2 == pytest.approx(-right_step.ay_m_s2, rel=1e-6, abs=1e-7)
        mirrored_wheels = (
            right_step.wheels[1],
            right_step.wheels[0],
            right_step.wheels[3],
            right_step.wheels[2],
        )
        for left_wheel, right_wheel in zip(left_step.wheels, mirrored_wheels, strict=True):
            assert left_wheel.fz_n == pytest.approx(right_wheel.fz_n, rel=1e-6, abs=1e-4)
            assert left_wheel.fx_n == pytest.approx(right_wheel.fx_n, rel=1e-6, abs=1e-4)
            assert left_wheel.fy_n == pytest.approx(-right_wheel.fy_n, rel=1e-6, abs=1e-4)
            assert left_wheel.kappa == pytest.approx(right_wheel.kappa, rel=1e-6, abs=1e-8)
            assert left_wheel.alpha_rad == pytest.approx(-right_wheel.alpha_rad, rel=1e-6, abs=1e-8)
            assert left_wheel.camber_deg == pytest.approx(
                -right_wheel.camber_deg, rel=1e-6, abs=1e-8
            )


def test_steering_demand_is_monotonic_with_front_roll_stiffness(
    config: KernelConfig,
) -> None:
    demands: list[float] = []
    radius_m = 50.0
    speed_m_s = 20.0
    geometric_steer = math.degrees(math.atan(config.wheelbase_m / radius_m)) * (
        config.steering_ratio
    )
    for front_fraction in (0.3, 0.5, 0.7):
        candidate_config = replace(config, roll_stiffness_front_fraction=front_fraction)
        lower_deg, upper_deg = 0.0, 2.0 * geometric_steer
        for _ in range(8):
            steer_deg = 0.5 * (lower_deg + upper_deg)
            plan = scenarios.Scenario(
                name="roll_stiffness_probe",
                initial_speed_m_s=speed_m_s,
                initial_gear=0,
                description="Match the same circle while varying the roll stiffness split.",
                segments=(scenarios.ScenarioSegment(0.5, steer_wheel_deg=steer_deg),),
            )
            run = scenarios.run_scenario(candidate_config, plan)
            tail = run.record.ground_truth[-15:]
            mean_speed = math.fsum(math.hypot(step.vx_m_s, step.vy_m_s) for step in tail) / len(
                tail
            )
            mean_yaw_rate = math.fsum(step.yaw_rate_rad_s for step in tail) / len(tail)
            if mean_speed / mean_yaw_rate > radius_m:
                lower_deg = steer_deg
            else:
                upper_deg = steer_deg
        demands.append(0.5 * (lower_deg + upper_deg))

    assert np.all(np.diff(demands) < 0.0), demands


# ---------------------------------------------------------------------- determinism


def test_two_runs_of_one_scenario_are_byte_identical(
    config: KernelConfig, suite: Mapping[str, Scenario]
) -> None:
    scenario = suite["standing_launch"]
    first = scenarios.run_scenario(config, scenario)
    second = scenarios.run_scenario(config, scenario)
    assert first.trace.tobytes() == second.trace.tobytes()
    assert first.drive_torque_nm.tobytes() == second.drive_torque_nm.tobytes()
    for field in ("gear", "soc_mj", "ice_rpm", "mgu_k_power_w"):
        left = getattr(first.drivetrain, field)
        right = getattr(second.drivetrain, field)
        assert left.tobytes() == right.tobytes(), field


# ----------------------------------------------------------------------- the boundary


def test_a_duration_off_the_kernel_step_grid_is_refused(config: KernelConfig) -> None:
    segment = scenarios.ScenarioSegment(duration_s=0.10051)
    scenario = scenarios.Scenario("off_grid", 0.0, (segment,), "off grid")
    with pytest.raises(ValueError, match=r"whole number of .* kernel steps"):
        scenarios.run_scenario(config, scenario)


def test_a_segment_shorter_than_one_control_interval_is_refused(config: KernelConfig) -> None:
    segment = scenarios.ScenarioSegment(duration_s=config.dt_s * 5.0)
    scenario = scenarios.Scenario("short", 0.0, (segment,), "shorter than one interval")
    with pytest.raises(ValueError, match=r"control intervals"):
        scenarios.run_scenario(config, scenario, control_steps=100)


def test_a_pedal_outside_the_stroke_is_refused(config: KernelConfig) -> None:
    for pedal in ({"throttle": 1.5}, {"clutch": -0.1}):
        segment = scenarios.ScenarioSegment(duration_s=0.1, **pedal)  # pyright: ignore[reportArgumentType]
        scenario = scenarios.Scenario("pedal", 0.0, (segment,), "bad pedal")
        with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
            scenarios.run_scenario(config, scenario)


def test_a_brake_torque_that_drives_the_wheels_forward_is_refused(config: KernelConfig) -> None:
    segment = scenarios.ScenarioSegment(duration_s=0.1, brake_torque_nm=100.0)
    scenario = scenarios.Scenario("brake", 0.0, (segment,), "brake with the wrong sign")
    with pytest.raises(ValueError, match=r"brake torque"):
        scenarios.run_scenario(config, scenario)


def test_an_unknown_gear_request_is_refused(config: KernelConfig) -> None:
    segment = scenarios.ScenarioSegment(duration_s=0.1, request=9)  # pyright: ignore[reportArgumentType]
    scenario = scenarios.Scenario("request", 0.0, (segment,), "unknown request")
    with pytest.raises(ValueError, match=r"request"):
        scenarios.run_scenario(config, scenario)


def test_an_initial_engine_speed_outside_the_configured_range_is_refused(
    config: KernelConfig,
) -> None:
    """An initial engine state outside the operating band is refused rather than clamped.

    The derived rpm is clamped to ``[idle_rpm, rev_limit_rpm]`` because a wheel speed outside that
    band is a gear-ratio artefact, not an engine speed. A *declared* speed outside it is different:
    it is a caller bug, and clamping it would silently change the requested state.
    """
    for override in (
        config.idle_rpm - 1.0,
        config.rev_limit_rpm + 1.0,
        -12_000.0,
    ):
        segment = scenarios.ScenarioSegment(duration_s=0.1, ice_rpm_initial=override)
        scenario = scenarios.Scenario("rpm", 0.0, (segment,), "out of range")
        with pytest.raises(ValueError, match=r"ice_rpm_initial must be within"):
            scenarios.run_scenario(config, scenario)


def test_a_nonfinite_initial_engine_speed_is_refused(config: KernelConfig) -> None:
    for override in (math.nan, math.inf, -math.inf):
        segment = scenarios.ScenarioSegment(duration_s=0.1, ice_rpm_initial=override)
        scenario = scenarios.Scenario("rpm", 0.0, (segment,), "nonfinite rpm")
        with pytest.raises(ValueError, match=r"ice_rpm_initial must be finite"):
            scenarios.run_scenario(config, scenario)


def test_the_configured_idle_and_rev_limit_are_legal_initial_state_bounds(
    config: KernelConfig,
) -> None:
    for override in (config.idle_rpm, config.rev_limit_rpm):
        segment = scenarios.ScenarioSegment(
            duration_s=0.1,
            clutch=0.5,
            grid_standing_start=True,
            ice_rpm_initial=override,
        )
        scenario = scenarios.Scenario("rpm", 0.0, (segment,), "bound")
        run = scenarios.run_scenario(config, scenario)
        assert np.all(run.drivetrain.ice_rpm == override)


def test_a_nonfinite_mgu_k_request_is_refused(config: KernelConfig) -> None:
    segment = scenarios.ScenarioSegment(duration_s=0.1, mgu_k_request_nm=math.nan)
    scenario = scenarios.Scenario("mgu", 0.0, (segment,), "nan request")
    with pytest.raises(ValueError, match=r"mgu_k_request_nm must be finite"):
        scenarios.run_scenario(config, scenario)


def test_a_bias_vector_of_the_wrong_length_is_refused(config: KernelConfig) -> None:
    segment = scenarios.ScenarioSegment(duration_s=0.1)
    scenario = scenarios.Scenario(
        "bias", 0.0, (segment,), "three brakes", brake_bias=(-1.0, -1.0, -1.0)
    )
    with pytest.raises(ValueError, match=r"brake_bias"):
        scenarios.run_scenario(config, scenario)


def test_an_unknown_scenario_name_is_refused(config: KernelConfig) -> None:
    with pytest.raises(KeyError):
        scenarios.scenario(config, "reverse_on_a_straight")


def test_scenario_init_loader_resolves_parameters_and_keeps_defaults(tmp_path) -> None:
    source = tmp_path / "rolling.yaml"
    source.write_text(
        """ParameterValueDeclarations:
  - name: start_speed
    value: 12.0
  - name: start_gear
    value: 1
Init:
  name: rolling_start
  description: Rolling start
  initial_speed_m_s: "$start_speed"
  initial_gear: "$start_gear"
""",
        encoding="utf-8",
    )

    initial = scenarios.load_scenario_init(source)

    assert initial == scenarios.ScenarioInit(
        name="rolling_start",
        description="Rolling start",
        initial_speed_m_s=12.0,
        initial_gear=1,
    )


def test_scenario_init_loader_accepts_literals_and_null_soc(tmp_path) -> None:
    source = tmp_path / "literal.yaml"
    source.write_text(
        """ParameterValueDeclarations: []
Init:
  name: literal_start
  description: Literal start
  initial_speed_m_s: 0
  soc_mj: null
""",
        encoding="utf-8",
    )

    initial = scenarios.load_scenario_init(source)

    assert initial.initial_speed_m_s == 0.0
    assert initial.soc_mj is None


def test_scenario_init_loader_rejects_duplicate_and_unknown_references(tmp_path) -> None:
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        """ParameterValueDeclarations:
  - name: speed
    value: 10
  - name: speed
    value: 20
Init:
  name: duplicate
  description: Duplicate declaration
  initial_speed_m_s: "$speed"
""",
        encoding="utf-8",
    )
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text(
        """ParameterValueDeclarations: []
Init:
  name: unknown
  description: Unknown reference
  initial_speed_m_s: "$missing"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"duplicate.yaml.*speed.*duplicate"):
        scenarios.load_scenario_init(duplicate)
    with pytest.raises(ValueError, match=r"unknown.yaml.*initial_speed_m_s.*missing"):
        scenarios.load_scenario_init(unknown)


@pytest.mark.parametrize(
    "document",
    [
        """ParameterValueDeclarations: []
Init: {name: bad, description: Bad, initial_speed_m_s: 1}
Extra: true
""",
        """ParameterValueDeclarations: []
Init: {name: bad, description: Bad, initial_speed_m_s: 1, extra: true}
""",
        """ParameterValueDeclarations: []
Init: {name: bad, description: Bad}
""",
        """ParameterValueDeclarations: []
Init: {name: bad, description: Bad, initial_speed_m_s: true}
""",
        """ParameterValueDeclarations: []
Init: {name: bad, description: Bad, initial_speed_m_s: .inf}
""",
    ],
)
def test_scenario_init_loader_rejects_schema_and_nonfinite_values(tmp_path, document) -> None:
    source = tmp_path / "invalid.yaml"
    source.write_text(document, encoding="utf-8")

    with pytest.raises(ValueError, match=r"invalid.yaml"):
        scenarios.load_scenario_init(source)


def test_loaded_scenario_init_uses_run_scenario_config_validation(
    tmp_path, config: KernelConfig
) -> None:
    source = tmp_path / "out-of-range-gear.yaml"
    source.write_text(
        """ParameterValueDeclarations: []
Init:
  name: invalid_gear
  description: Gear checked by the scenario runner
  initial_speed_m_s: 0
  initial_gear: 99
""",
        encoding="utf-8",
    )
    initial = scenarios.load_scenario_init(source)
    plan = scenarios.Scenario(
        name=initial.name,
        initial_speed_m_s=initial.initial_speed_m_s,
        segments=(scenarios.ScenarioSegment(duration_s=0.1),),
        description=initial.description,
        initial_gear=initial.initial_gear,
        soc_mj=initial.soc_mj,
        brake_bias=initial.brake_bias,
        tyre_leak_rate_kg_s=initial.tyre_leak_rate_kg_s,
        initial_x_m=initial.initial_x_m,
        initial_y_m=initial.initial_y_m,
        initial_heading_rad=initial.initial_heading_rad,
    )

    with pytest.raises(ValueError, match=r"initial_gear must be a whole gear"):
        scenarios.run_scenario(config, plan)
