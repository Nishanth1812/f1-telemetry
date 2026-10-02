"""Phase 1 straight-line scenarios run over the existing physics APIs.

`PHASES.md` P1-T8 asks for scenarios, and `tasks/todo.md` Task 5 asks that each one run from
fixed initial conditions and emit traces through the existing testing/record pattern. This
file covers that path and nothing wider:

* **The behaviour each scenario exists to show.** A standing launch starts from rest in first
  gear and moves; the gearbox moves *only* where the scenario asks, and each request cuts the
  driveline for the configured shift time; a neutral selection transmits exactly nothing and
  selecting first gear again restores it; a coasting car decelerates against drag alone and
  drag grows with speed; caller brake torque decelerates the car and its wheels; the MGU-K is
  blocked below 50 km/h in a declared standing start, drains the store when it deploys and
  refills it when it regenerates, without exceeding C5.2.7's 350 kW.
* **The contracts the runner has to honour.** Caller-owned state, no clock and no random
  source, byte-identical repeat runs, and refusals for a duration that is not a whole number of
  kernel steps and for pedals, requests or brake torque it could not use.
* **The produced records.** Only contract channels are published, every published value stays
  inside the range `channels.yaml` declares for it, and `PLAN.md` section 11's invariants,
  including the discrete chassis and wheel energy balance, run against the real traces.

No coefficient is tuned here and no target is asserted: every physical number comes from the
loaded `car_spec.yaml`, and the only bounds used are the regulation values the physics already
applies and the ranges the channel contract already declares.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import pytest

from f1telemetry.generated.channels import CHANNELS, DTYPES
from f1telemetry.physics import gearbox  # noqa: TID251 -- the scenarios drive this API
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


# ------------------------------------------------------------------ behaviour: launch


def test_a_standing_launch_starts_from_rest_in_first_gear(runs: Mapping[str, ScenarioRun]) -> None:
    run = runs["standing_launch"]
    steps = run.record.ground_truth
    assert steps[0].vx_m_s == 0.0
    assert steps[0].gear == 1
    assert run.gears == (1,) * len(run.gears), "rpm alone must not move the gearbox (C9.8.1)"
    assert run.record.ground_truth[-1].vx_m_s > 0.0
    assert float(run.drivetrain.ice_power_w.max()) > 0.0, "the ICE stays at its configured idle"
    assert float(run.drivetrain.accel_m_s2.max()) > 0.0


def test_each_segment_applies_its_caller_supplied_clutch_state(
    runs: Mapping[str, ScenarioRun], suite: Mapping[str, Scenario]
) -> None:
    run = runs["standing_launch"]
    for window, segment in zip(
        _windows(run, suite[run.name]), suite[run.name].segments, strict=True
    ):
        assert np.all(run.drivetrain.clutch[window] == segment.clutch)


def test_accelerate_to_speed_crosses_one_hundred_km_h_from_rest(
    runs: Mapping[str, ScenarioRun],
) -> None:
    run = runs["accelerate_to_speed"]
    crossing = np.flatnonzero(run.trace[:, 1] >= 100.0 / 3.6)
    assert crossing.size
    assert run.trace[0, 1] == 0.0
    assert crossing[0] * run.dt_s == pytest.approx(6.8998, abs=0.001)


def test_full_throttle_run_reaches_top_gear_and_a_terminal_speed(
    runs: Mapping[str, ScenarioRun], config: KernelConfig
) -> None:
    run = runs["full_throttle"]
    assert max(run.gears) == config.gear_ratios.size
    last_five_seconds = round(5.0 / run.dt_s)
    change_kmh = (run.trace[-1, 1] - run.trace[-last_five_seconds, 1]) * 3.6
    assert 0.0 <= change_kmh < 1.0
    assert run.trace[-1, 1] * 3.6 == pytest.approx(307.419, abs=0.01)


def test_the_launch_grip_keeps_the_rear_tyres_inside_their_peak(
    runs: Mapping[str, ScenarioRun], config: KernelConfig
) -> None:
    """The demanded torque must stay inside what the tyre can hold, or the wheel runs away.

    With no traction control (C9.1.2) a demand above the Magic Formula's peak has no
    equilibrium: the wheel accelerates, slip grows past the peak and the force falls further
    still. That is the correct behaviour of the model, but it is not a usable scenario trace,
    so the launch is required to stay below the demand that would cause it.
    """
    run = runs["standing_launch"]
    for corner in ("RL", "RR"):
        kappa = [step.wheels[CORNERS.index(corner)].kappa for step in run.record.ground_truth]
        assert max(kappa) < 1.0, f"{corner} slip ratio reached {max(kappa)}, which is past the peak"


# ---------------------------------------------------------------- behaviour: shifting


def test_the_gearbox_moves_only_where_the_scenario_asks(
    runs: Mapping[str, ScenarioRun], suite: Mapping[str, Scenario]
) -> None:
    run = runs["full_throttle_shifts"]
    scenario = suite["full_throttle_shifts"]
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
    runs: Mapping[str, ScenarioRun], suite: Mapping[str, Scenario]
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
    accel = run.drivetrain.accel_m_s2[window]
    assert np.all(np.diff(speeds[1:]) < 0.0), "nothing but drag acts after neutral selection"
    assert np.all(accel[1:] < 0.0)
    assert np.all(drag < 0.0), "drag power is signed against forward motion"
    assert abs(drag[-1]) < abs(drag[0]), "drag power falls as the car's speed falls"
    assert abs(drag[-1] / speeds[-1]) < abs(drag[0] / speeds[0]), (
        "drag force grows approximately with speed squared"
    )


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


def test_mgu_k_power_stays_inside_the_absolute_electrical_cap(
    runs: Mapping[str, ScenarioRun], config: KernelConfig
) -> None:
    run = runs["mgu_k_deploy_regen"]
    power_kw = np.abs(run.drivetrain.mgu_k_power_w) / 1_000.0
    assert float(power_kw.max()) <= config.mgu_k_peak_power_kw, "C5.2.7's 350 kW"
    assert float(power_kw.max()) > 0.0, "the motor never moved, so nothing was proven"


# --------------------------------------------------------------- invariants and records


def test_every_phase_one_invariant_passes_on_the_produced_records(
    runs: Mapping[str, ScenarioRun], spec: CarSpec, suite: Mapping[str, Scenario]
) -> None:
    for name, run in runs.items():
        for result in run_all(run.record, spec, scenarios.PHASE_ONE_INVARIANTS):
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
