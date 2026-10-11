"""The headless fast path: one scenario, one compiled drivetrain loop, one post-pass.

`docs/latency.md` profiles :func:`~f1telemetry.testing.scenarios.run_scenario` at roughly 0.28x
real time and attributes 98.9% of that to runner overhead rather than to the kernel. This module
is the fast path that overhead section is asking for. It runs the *same* kernel and the *same*
drivetrain over the same two rates, and it returns the same
:class:`~f1telemetry.testing.scenarios.ScenarioRun`: the trace it returns is byte-identical to the
one ``run_scenario`` returns for the same plan, and so are the drivetrain columns, the step
outputs and the record, because each is built from the same buffers by the same functions.

**What was slow, and what changed.** The profile is not ambiguous. Per kernel step,
``run_scenario`` called :func:`~f1telemetry.physics.powertrain.step_mgu_k` and
:func:`~f1telemetry.physics.gearbox.step_gearbox`, and those two are *validation* boundaries:
together 11.1 s of a 14.2 s profiled run, of which almost none was arithmetic. Every step
re-checked the ratio table, the ICE torque curve and both MGU-K deployment curves for finiteness
and shape, re-read the gear out of the buffer to check its domain, and rebuilt dictionaries of
already-validated scalars - to hand a handful of numbers to a compiled step that trusts them.
:func:`~f1telemetry.physics.powertrain.step_ice_torque` was the same story at 3.4 s, and
``run_scenario`` called it 70 700 times to produce a value that is constant across a control
interval.

So what this path defers is not the physics and not the record: it is the *re-derivation* of the
physics' inputs. Every configuration scalar the two entry points would re-validate on every step
is read once per run, and one compiled call per control interval does the stepping, calling the
same compiled primitives - ``gearbox._step_gearbox`` and ``powertrain._step_mgu_k`` - that the
entry points themselves call. The arithmetic is not re-implemented, re-associated or approximated.
It is the identical compiled function on the identical inputs in the identical order, which is why
the result is byte-identical rather than close.

**Where the validation went, since it did not go away.** It moved from per step to per segment.
Before the run steps anything, each segment's drivetrain inputs are handed once to the public
:func:`~f1telemetry.physics.gearbox.step_gearbox` and
:func:`~f1telemetry.physics.powertrain.step_mgu_k` on *scratch copies* of the state buffers, so
every check those boundaries perform still runs, and still runs before a step is taken rather than
after one. What the compiled loop then relies on is the one rule those boundaries could not
enforce per step anyway: the gear only ever moves to an adjacent forward index, to neutral or to
reverse, because every request is a :class:`~f1telemetry.physics.gearbox.GearRequest` code and
``_checked_segments`` has already refused an out-of-domain initial gear. That is the same
``boundscheck=False`` contract :func:`~f1telemetry.physics.longitudinal.simulate` states for its
own inputs, taken for the same reason. The one quantity that is not a segment declaration is the
sampled road speed, because it comes out of the trace; it is checked for finiteness once per
control interval, where the state it came from was just written.

**The record, the thermal trace and the invariants are still assembled once, by the same code.**
``run_scenario`` already builds its record after the stepping loop rather than inside it, so
"deferred to a post-pass" is not what this module changes; what changes is that the stepping loop
no longer competes with it for the same per-step Python calls. This module deliberately *reuses*
``scenarios._build_record`` and the kernel-rate-to-recorded-rate decimation rather than writing a
cheaper record of its own, for two reasons. An energy residual or a tyre temperature produced by a
second implementation would differ from the reference one by a rounding difference, and a headless
path that quietly reports a different residual is worse than a slow one. And a record built from
the same buffers is byte-identical to the reference run's, which is what lets
``tests/test_headless.py` assert equality over the whole run rather than over the trace alone. If
headless throughput ever needs the record gone as well, the honest next step is an optional
record on this path - not a faster one.

**What this does not do.** It changes no coefficient, no ``dt_s``, no ``CONTROL_STEPS`` and no
order of arithmetic; it reads no clock and no random source; and it holds the state, the buffers
and the refusals to caller ownership exactly as ``run_scenario`` does. It is a faster *schedule*
over the existing model, not a different model. The measured speedup is in ``docs/latency.md`,
reported as measured rather than as targeted, and that document states plainly that this path
does not reach the P6 50x real-time gate.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import numpy as np
from numba import njit

from f1telemetry.kernels import longitudinal  # noqa: TID251 -- the headless path drives the kernel
from f1telemetry.physics import (  # noqa: TID251 -- and the models it steps
    engine,
    forces,
    gearbox,
    powertrain,
)

# The two compiled drivetrain steps this module drives directly, below. They are the same functions
# ``gearbox.step_gearbox`` and ``powertrain.step_mgu_k`` call, reached past those functions'
# per-call validation because this module runs that validation once per segment instead - see the
# module docstring.
from f1telemetry.physics.gearbox import (  # noqa: TID251 -- compiled step, see the docstring
    _step_gearbox as step_gearbox_step,  # pyright: ignore[reportPrivateUsage]
)
from f1telemetry.physics.powertrain import (  # noqa: TID251 -- compiled step, see the docstring
    _step_mgu_k as step_mgu_k_step,  # pyright: ignore[reportPrivateUsage]
)
from f1telemetry.testing.run_manifest import RunManifest
from f1telemetry.testing.scenarios import (
    _J_PER_MJ as J_PER_MJ,  # pyright: ignore[reportPrivateUsage]
)

# This module is a *peer* of ``scenarios``, not a second owner of its validated boundary, so it
# reads the runner's own helpers rather than restating them. The names below are the ones a peer
# needs and ``scenarios`` does not export: its segment checker, brake-history builder,
# wheel-coupled-rpm reader, driver's-request refusal, kernel-rate decimation, step-output view and
# record assembler; and its joules-per-megajoule constant, so the store-side power on this path is
# scaled by the reference runner's number rather than by a second copy of it.
from f1telemetry.testing.scenarios import (
    CONTROL_STEPS,
    ControlState,
    DrivetrainTrace,
    Scenario,
    ScenarioRun,
    ScenarioSegment,
)
from f1telemetry.testing.scenarios import (
    _brake_history as brake_history,  # pyright: ignore[reportPrivateUsage]
)
from f1telemetry.testing.scenarios import (
    _build_record as build_record,  # pyright: ignore[reportPrivateUsage]
)
from f1telemetry.testing.scenarios import (
    _check_driver_request as check_driver_request,  # pyright: ignore[reportPrivateUsage]
)
from f1telemetry.testing.scenarios import (
    _checked_segments as checked_segments,  # pyright: ignore[reportPrivateUsage]
)
from f1telemetry.testing.scenarios import (
    _held as held,  # pyright: ignore[reportPrivateUsage]
)
from f1telemetry.testing.scenarios import (
    _step_output_view as step_output_view,  # pyright: ignore[reportPrivateUsage]
)
from f1telemetry.testing.scenarios import (
    _wheel_coupled_ice_rpm as wheel_coupled_ice_rpm,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import KernelConfig
    from f1telemetry.testing.scenarios import ControlLaw

__all__ = ["run_headless"]

# The drivetrain and MGU-K state slots, bound once as plain module constants because that is the
# form Numba freezes; the values and their order are ``gearbox``'s and ``powertrain``'s.
GEAR_INDEX: Final[int] = gearbox.GEAR_INDEX
SHIFT_TIMER_INDEX: Final[int] = gearbox.SHIFT_TIMER_INDEX
CLUTCH_INDEX: Final[int] = gearbox.CLUTCH_INDEX
SOC_INDEX: Final[int] = powertrain.SOC_INDEX
LAP_RECHARGE_INDEX: Final[int] = powertrain.LAP_RECHARGE_INDEX

# The two gear states outside the forward box, and ``HOLD`` as the integer code the compiled
# gearbox step compares against.
NEUTRAL_GEAR: Final[float] = gearbox.NEUTRAL_GEAR
REVERSE_GEAR: Final[float] = gearbox.REVERSE_GEAR
REQUEST_HOLD: Final[int] = int(gearbox.GearRequest.HOLD)

# km/h per m/s. The public ``step_mgu_k`` applies this to the sampled speed before the C5.2.8
# power cap looks at it; it is stated here because ``powertrain`` keeps the constant private to
# its own boundary and duplicating a *conversion* is not duplicating a coefficient.
KMH_PER_M_S: Final[float] = 3.6


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def _drive_control_interval(
    steps: int,
    offset: int,
    interval_request: int,
    driver_throttle: float,
    sampled_rpm: float,
    sampled_speed_km_h: float,
    ice_torque_nm: float,
    gear_state: np.ndarray,
    mgu_k_state: np.ndarray,
    dt_s: float,
    mgu_k_cap_w: float,
    mgu_k_request_nm: float,
    mgu_k_dc_limit_kw: float,
    grid_standing_start: bool,
    gear_ratios: np.ndarray,
    reverse_ratio: float,
    final_drive: float,
    shift_time_s: float,
    clutch_demand_torque_nm: float,
    clutch_demand_travel_fraction: float,
    mgu_k_crankshaft_ratio: float,
    mgu_k_motor_inverter_efficiency: float,
    mgu_k_torque_limit_nm: float,
    mgu_k_relative_speed_limit_rpm: float,
    launch_speed_km_h: float,
    store_energy_mj: float,
    recharge_limit_mj_per_lap: float,
    torque: np.ndarray,
    drive_torque_nm: np.ndarray,
    rpm: np.ndarray,
    ice: np.ndarray,
    mgu_k_power_w: np.ndarray,
    gear: np.ndarray,
    clutch: np.ndarray,
    soc: np.ndarray,
    lap_recharge: np.ndarray,
    throttle: np.ndarray,
) -> tuple[float, float]:
    """Step the drivetrain across one control interval into the run's kernel-rate columns.

    The compiled twin of the innermost loop in
    :func:`~f1telemetry.testing.scenarios.run_scenario`, statement for statement and in the same
    order: read the state of charge, step the MGU-K, step the gearbox with the ICE's torque and the
    motor's summed at the crankshaft, accumulate delivered and load crank torque against the gear
    that step settled on, then publish the columns the recorded rate decimates afterwards.

    ``interval_request`` is the interval's request and is applied on the first step only; every
    later step holds. That is the segment-holds-its-input rule and not a shortcut: a request is a
    command, so re-applying it would ratchet the box a gear per step.
    ``ice_torque_nm`` is what :func:`~f1telemetry.physics.powertrain.step_ice_torque` returns for
    this interval's held rpm and pedal - constant across the interval, which is why the reference
    runner's per-step recomputation of it was 70 700 calls to produce one number. The gearbox entry
    point computes the same quantity from the same two inputs and adds the motor's torque to it,
    so passing the shared value is the reference arithmetic and not a second way of doing it.
    ``offset`` is the index of this interval's first step in the run's kernel-rate columns.

    Returns ``(delivered_torque_sum, load_torque_sum)``, accumulated in step order in ``float64``,
    which is what :func:`~f1telemetry.physics.engine.step_engine_speed` integrates when the engine
    is free.
    """
    delivered_torque_sum = 0.0
    load_torque_sum = 0.0
    for index in range(steps):
        position = offset + index
        throttle[position] = driver_throttle
        charge_before_mj = mgu_k_state[SOC_INDEX]
        mgu_k_torque_nm = step_mgu_k_step(
            mgu_k_state,
            mgu_k_request_nm,
            sampled_rpm,
            sampled_speed_km_h,
            dt_s,
            mgu_k_dc_limit_kw,
            mgu_k_crankshaft_ratio,
            mgu_k_motor_inverter_efficiency,
            mgu_k_torque_limit_nm,
            mgu_k_relative_speed_limit_rpm,
            launch_speed_km_h,
            store_energy_mj,
            grid_standing_start,
            False,
            recharge_limit_mj_per_lap,
        )
        torque[index] = step_gearbox_step(
            gear_state,
            ice_torque_nm + mgu_k_torque_nm,
            dt_s,
            gear_ratios,
            reverse_ratio,
            final_drive,
            shift_time_s,
            clutch_demand_torque_nm,
            clutch_demand_travel_fraction,
            interval_request if index == 0 else REQUEST_HOLD,
        )
        delivered_torque_sum += ice_torque_nm + mgu_k_torque_nm
        current_gear = int(gear_state[GEAR_INDEX])
        if current_gear >= 1:
            total_ratio = gear_ratios[current_gear - 1] * final_drive
        elif current_gear == REVERSE_GEAR:
            total_ratio = -reverse_ratio * final_drive
        else:
            total_ratio = 0.0
        if total_ratio != 0.0:
            load_torque_sum += torque[index] / total_ratio
        drive_torque_nm[position] = torque[index]
        rpm[position] = sampled_rpm
        ice[position] = ice_torque_nm
        # Finite differencing accumulated SOC can exceed the enforced cap by roundoff, so the
        # published store-side power is clamped exactly as the reference runner clamps it.
        reported_power_w = (charge_before_mj - mgu_k_state[SOC_INDEX]) * J_PER_MJ / dt_s
        mgu_k_power_w[position] = min(mgu_k_cap_w, max(-mgu_k_cap_w, reported_power_w))
        gear[position] = current_gear
        clutch[position] = gear_state[CLUTCH_INDEX]
        soc[position] = mgu_k_state[SOC_INDEX]
        lap_recharge[position] = mgu_k_state[LAP_RECHARGE_INDEX]
    return delivered_torque_sum, load_torque_sum


def run_headless(
    config: KernelConfig,
    plan: Scenario,
    *,
    control_steps: int = CONTROL_STEPS,
    control_law: ControlLaw | None = None,
    max_brake_torque_nm: float = 0.0,
    manifest: RunManifest | None = None,
) -> ScenarioRun:
    """Run one scenario headlessly and return its trace, its drivetrain history and its record.

    Every argument means here exactly what it means in
    :func:`~f1telemetry.testing.scenarios.run_scenario`, and every refusal that function makes
    before simulating anything - a segment that is not a whole number of control intervals, a
    pedal outside ``[0, 1]``, a gear request that is not a ``GearRequest``, brake torque signed the
    wrong way, a bias vector of the wrong length, a control law that is not callable, a braking law
    with no ``max_brake_torque_nm`` - is made here too, by the same helpers, before a step is taken.

    What differs is the schedule. One compiled call per control interval replaces the per-step
    Python drivetrain calls, and the configuration scalars those calls re-validated on every one of
    the run's kernel steps are read once per run instead - after one validation pass per segment,
    made on scratch state copies so no run step is spent validating. The kernel is called exactly
    as the reference runner calls it, with the same buffers and the same step outputs, and the
    record is assembled once at the end by the same assembler. Trace, drivetrain columns, step
    outputs and record are therefore byte-identical to the reference run's for the same plan, which
    is what ``tests/test_headless.py`` asserts on a straight-line scenario and on a steering one.

    ``manifest`` is attached to the run as :attr:`ScenarioRun.manifest` and used for nothing else,
    exactly as in ``run_scenario``: it names the five facts the caller is citing, so attaching one
    needs no seed source, no car-spec or scenario resolution, no setup hashing, no git call and no
    clock, and changes no trace, no record and no byte of the run it describes.

    The measured speedup is in ``docs/latency.md``. It is a real one, it is not the 50x the P6 gate
    asks for, and that document says so rather than the constant being tuned until it looks like
    one.
    """
    counts = checked_segments(plan, config, control_steps)
    if (
        isinstance(max_brake_torque_nm, bool)
        or not isinstance(max_brake_torque_nm, (int, float))
        or not math.isfinite(max_brake_torque_nm)
        or max_brake_torque_nm < 0.0
    ):
        raise ValueError("max_brake_torque_nm must be finite and >= 0")
    if control_law is not None:
        candidate_control_law: object = control_law
        if not callable(candidate_control_law):
            raise TypeError(  # pyright: ignore[reportUnreachable] -- preserve guard for untyped callers
                "control_law must be callable"
            )
    total = sum(counts)

    trace = np.zeros((total + 1, longitudinal.STATE_SIZE), dtype=np.float64)
    step_outputs = longitudinal.allocate_step_outputs(total)
    steer_history = np.zeros(total, dtype=np.float64)
    drive = np.zeros(total, dtype=np.float64)
    brake_torque = np.zeros((total, forces.WHEEL_COUNT), dtype=np.float64)
    rpm = np.zeros(total, dtype=np.float64)
    ice = np.zeros(total, dtype=np.float64)
    mgu_k_w = np.zeros(total, dtype=np.float64)
    gear = np.zeros(total, dtype=np.int64)
    clutch = np.zeros(total, dtype=np.float64)
    soc = np.zeros(total, dtype=np.float64)
    recharge = np.zeros(total, dtype=np.float64)
    throttle = np.zeros(total, dtype=np.float64)
    torque = np.zeros(control_steps, dtype=np.float64)

    gear_state = gearbox.initial_state(float(plan.initial_gear))
    mgu_k_state = powertrain.mgu_k_initial_state(config, plan.soc_mj)
    engine_rpm = config.idle_rpm
    state = longitudinal.initial_state(
        distance_m=plan.initial_x_m,
        speed_m_s=plan.initial_speed_m_s,
        wheel_omega_rad_s=plan.initial_speed_m_s / config.rolling_radius_m,
        y_m=plan.initial_y_m,
        heading_rad=plan.initial_heading_rad,
    )
    trace[0] = state

    row = 0
    mgu_k_cap_w = config.mgu_k_peak_power_kw * 1_000.0
    for segment, count in zip(plan.segments, counts, strict=True):
        steer_history[row : row + count] = segment.steer_wheel_deg
        if segment.ice_rpm_initial is not None:
            engine_rpm = float(segment.ice_rpm_initial)
        gear_state[CLUTCH_INDEX] = segment.clutch
        brake = brake_history(segment, plan.brake_bias, count)
        brake_torque[row : row + count] = brake
        _validate_segment_drivetrain(config, segment, gear_state, mgu_k_state, engine_rpm)
        for interval in range(0, count, control_steps):
            interval_start = row + interval
            sample = state
            driver_throttle = segment.throttle
            driver_steer_deg = segment.steer_wheel_deg
            if control_law is not None:
                request = control_law(
                    ControlState(
                        time_s=interval_start * config.dt_s,
                        x_m=float(sample[longitudinal.X_INDEX]),
                        y_m=float(sample[longitudinal.Y_INDEX]),
                        heading_rad=float(sample[longitudinal.PSI_INDEX]),
                        speed_m_s=math.hypot(
                            float(sample[longitudinal.V_INDEX]),
                            float(sample[longitudinal.VY_INDEX]),
                        ),
                    )
                )
                check_driver_request(request, config)
                driver_throttle = request.throttle
                driver_steer_deg = math.degrees(request.steering_wheel_rad)
                if request.brake > 0.0 and max_brake_torque_nm <= 0.0:
                    raise ValueError("max_brake_torque_nm must be > 0 for a braking control law")
                brake[interval : interval + control_steps] = np.asarray(
                    [-max_brake_torque_nm * request.brake * abs(share) for share in plan.brake_bias]
                )
                brake_torque[interval_start : interval_start + control_steps] = brake[
                    interval : interval + control_steps
                ]
                steer_history[interval_start : interval_start + control_steps] = driver_steer_deg
            shifting_at_start = gear_state[SHIFT_TIMER_INDEX] > 0.0
            starting_gear = int(gear_state[GEAR_INDEX])
            interval_request = segment.request if interval == 0 else gearbox.GearRequest.HOLD
            if (
                plan.upshift_at_shift_point
                and interval_request == gearbox.GearRequest.HOLD
                and not shifting_at_start
                and segment.clutch >= 1.0
                and 1 <= starting_gear < config.gear_ratios.size
                and wheel_coupled_ice_rpm(config, sample, gear_state) >= config.shift_up_rpm
            ):
                interval_request = gearbox.GearRequest.UP
            shift_requested = interval_request != gearbox.GearRequest.HOLD
            clutch_open = segment.clutch < 1.0 or starting_gear == NEUTRAL_GEAR
            engine_is_free = clutch_open or shifting_at_start or shift_requested
            if engine_is_free:
                if not clutch_open:
                    # Entering a shift cut from a locked clutch preserves the coupled speed as the
                    # initial condition of the newly free engine state.
                    engine_rpm = wheel_coupled_ice_rpm(config, sample, gear_state)
                sampled_rpm = engine_rpm
            else:
                sampled_rpm = wheel_coupled_ice_rpm(config, sample, gear_state)
                engine_rpm = sampled_rpm
            sampled_speed_km_h = _sampled_speed_km_h(sample)
            sampled_torque = powertrain.step_ice_torque(config, sampled_rpm, driver_throttle)
            mgu_k_dc_limit_kw = powertrain.mgu_k_power_limit_kw(
                sampled_speed_km_h,
                (config.ers_overtake_speed_km_h if segment.overtake else config.ers_speed_km_h),
                (config.ers_overtake_limit_kw if segment.overtake else config.ers_limit_kw),
            )
            delivered_sum, load_sum = _drive_control_interval(
                control_steps,
                interval_start,
                int(interval_request),
                driver_throttle,
                sampled_rpm,
                sampled_speed_km_h,
                sampled_torque,
                gear_state,
                mgu_k_state,
                config.dt_s,
                mgu_k_cap_w,
                segment.mgu_k_request_nm,
                mgu_k_dc_limit_kw,
                segment.grid_standing_start,
                config.gear_ratios,
                config.reverse_ratio,
                config.final_drive,
                config.shift_time_s,
                config.clutch_demand_torque_nm,
                config.clutch_demand_travel_fraction,
                config.mgu_k_crankshaft_ratio,
                config.mgu_k_motor_inverter_efficiency,
                config.mgu_k_torque_limit_nm,
                config.mgu_k_relative_speed_limit_rpm,
                config.launch_speed_kmh,
                config.store_energy_mj,
                config.recharge_limit_mj_per_lap,
                torque,
                drive,
                rpm,
                ice,
                mgu_k_w,
                gear,
                clutch,
                soc,
                recharge,
                throttle,
            )
            out = longitudinal.simulate(
                config,
                control_steps,
                state,
                torque,
                longitudinal.allocate(control_steps),
                brake[interval : interval + control_steps],
                step_outputs=step_output_view(step_outputs, interval_start, control_steps),
                steer_wheel_deg=steer_history[interval_start : interval_start + control_steps],
            )
            trace[interval_start + 1 : interval_start + control_steps + 1] = out[1:]
            state = out[control_steps].copy()
            if engine_is_free:
                engine_rpm = engine.step_engine_speed(
                    config,
                    sampled_rpm,
                    delivered_sum / control_steps,
                    load_sum / control_steps,
                    dt_s=control_steps * config.dt_s,
                )
            else:
                engine_rpm = wheel_coupled_ice_rpm(config, state, gear_state)
        row += count

    accel = trace[:, longitudinal.PREVIOUS_AX_INDEX].copy()
    recorded = held(rpm, control_steps, total).size
    drivetrain = DrivetrainTrace(
        gear=held(gear, control_steps, total),
        clutch=held(clutch, control_steps, total),
        throttle=held(throttle, control_steps, total),
        ice_rpm=held(rpm, control_steps, total),
        ice_torque_nm=held(ice, control_steps, total),
        ice_power_w=(
            held(ice, control_steps, total) * held(rpm, control_steps, total) * (math.tau / 60.0)
        ),
        mgu_k_power_w=held(mgu_k_w, control_steps, total),
        drive_torque_nm=held(drive, control_steps, total),
        soc_mj=held(soc, control_steps, total),
        lap_recharge_mj=held(recharge, control_steps, total),
        accel_m_s2=held(accel, control_steps, total),
    )
    if len(drivetrain.gear) != recorded:
        msg = f"{plan.name}: the drivetrain history and the record disagree on length"
        raise AssertionError(msg)
    return ScenarioRun(
        name=plan.name,
        description=plan.description,
        dt_s=config.dt_s,
        control_steps=control_steps,
        steps=total,
        trace=trace,
        drive_torque_nm=drive,
        brake_torque_nm=brake_torque,
        drivetrain=drivetrain,
        record=build_record(
            plan,
            config,
            trace,
            drivetrain,
            drive,
            brake_torque,
            control_steps,
            step_outputs,
            steer_history,
            plan.tyre_leak_rate_kg_s,
        ),
        step_outputs=step_outputs,
        steer_wheel_deg=steer_history,
        manifest=manifest,
    )


def _sampled_speed_km_h(sample: np.ndarray) -> float:
    """The interval's held road speed in km/h, absolute, as the MGU-K's power cap reads it.

    ``step_mgu_k`` derives this from the sampled body-frame ``vx`` rather than from the ground
    speed, because the cap is stated against road speed and a reverse interval has a negative one.
    Re-derived here rather than called through, because it is the one drivetrain input on this path
    that is not a segment declaration - it comes out of the trace - and so the one value the
    once-per-segment validation pass cannot have seen. Checking it once per control interval,
    where the state it came from was just written, is what keeps a non-finite speed an error
    rather than a torque the model happily propagates.
    """
    speed_km_h = KMH_PER_M_S * float(sample[longitudinal.V_INDEX])
    if not math.isfinite(speed_km_h):
        msg = (
            "run_headless: the sampled road speed is not finite. A NaN reaching the MGU-K would be "
            "a torque and a store state written from it, so it is refused where it is sampled"
        )
        raise ValueError(msg)
    return abs(speed_km_h)


def _validate_segment_drivetrain(
    config: KernelConfig,
    segment: ScenarioSegment,
    gear_state: np.ndarray,
    mgu_k_state: np.ndarray,
    engine_rpm: float,
) -> None:
    """Run the drivetrain's Python validation once per segment, on scratch state copies.

    The compiled loop is handed scalars read from ``config`` and steps the state buffers without
    re-checking them, so the checks those two boundaries would otherwise collect on every one of
    the run's kernel steps are collected here instead - once per segment, which is the coarsest
    interval over which a drivetrain declaration can change. Both public entry points are called
    on *copies* of the buffers, so this run's own ``gear_state`` and ``mgu_k_state`` are untouched:
    a step taken only to be validated must not be a step of the run.

    ``engine_rpm`` is the run's engine speed as the segment starts, so the ICE curve and the
    MGU-K's part-speed cap are checked at a speed this run actually visits.
    """
    gearbox.step_gearbox(
        config,
        gear_state.copy(),
        engine_rpm,
        segment.throttle,
        gearbox.GearRequest.HOLD,
    )
    powertrain.step_mgu_k(
        config,
        mgu_k_state.copy(),
        segment.mgu_k_request_nm,
        engine_rpm,
        0.0,
        grid_standing_start=segment.grid_standing_start,
        overtake=segment.overtake,
    )
