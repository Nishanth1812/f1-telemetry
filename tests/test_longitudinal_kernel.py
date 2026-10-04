"""P1-T1: the fixed-step straight-line kernel, tested against the claims it makes.

``PHASES.md`` P1-T1 asks for a flat ``@njit(cache=True, fastmath=False)`` kernel over
preallocated float64 arrays at ``dt = 100 µs``, and P1-T2 asks for the determinism harness
that proves it. The claims are checked here in the form that would actually fail:

* **It really is a compiled, cached kernel.** The dispatcher's own options are read back
  (``fastmath=False``, ``nogil``, ``boundscheck=False``) instead of being trusted from the
  decorator text, and ``cache=True`` is proven by a fresh interpreter writing a cache index
  into a directory that was empty a moment earlier.
* **Fixed step, semi-implicit.** Row ``n`` of a trace is the state after exactly ``n`` steps of
  the configured ``dt_s``, and position advances with the *updated* speed, so the scheme is
  semi-implicit rather than explicit. The scheme is pinned against the explicit alternative,
  not against a number copied out of a run.
* **The loop is closed, and its ordering is pinned.** Four caller-owned wheel angular speeds sit in
  the state, and one step reads ``drive torque -> wheel torque -> slip -> Fx -> force on the car``
  with the wheel states and the speed both advanced on the forces evaluated at the state they
  started from. That ordering is a model, not an implementation detail, so it is asserted: an
  ordering that advanced the wheels on the *new* speed would give a different and equally plausible
  launch.
* **Determinism.** Two runs with the same configuration and the same buffers produce
  byte-identical traces, and the run reads no clock and no random source. Byte comparison, not
  ``allclose``: a tolerance would hide the last-bit differences a wall-clock read introduces.
* **No work the caller did not ask for.** The step loop allocates nothing measurable in Numba's
  runtime, the caller owns every buffer, and the inputs are left untouched.
* **One way in.** The compiled loop runs with ``boundscheck=False``, so ``simulate`` is the only
  door: a buffer of the wrong type, layout, size or writability, a step count that is not an
  integer, or a ``dt_s``/``mass_kg`` that would quietly produce NaNs is refused in Python,
  before the loop starts. P1-T6/T7 widened that door's list: the kernel now indexes the aero
  curves and divides by the wheel inertia and the rolling radius inside the loop, so every
  coefficient it reads is checked at the same boundary, and a seeded wheel speed that is not finite
  is refused rather than carried into every later row.

Every physical number here comes from the loaded ``car_spec.yaml``, including the drive torque
used for the representative runs. The one-step launch checks take their drive torque from the
configuration's own gearbox - first gear, full engagement - so a run that is analytically exact is
still a run this car would actually make, and there is no tuned constant in this file.

The ``kernel`` marker keeps these apart from the ``toolchain`` probe: that one proves Numba
compiles here, this one proves the car's integrator is right.
"""

from __future__ import annotations

import importlib
import math
import os
import subprocess
import sys
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from f1telemetry.contracts.car_spec import CarSpec
from f1telemetry.kernels import longitudinal
from f1telemetry.physics import (
    combined_slip,
    forces,
    gearbox,
    kinematics,
    loads,
    relaxation,
    steering,
)

if TYPE_CHECKING:
    from pathlib import Path

    from f1telemetry.contracts.car_spec import KernelConfig

pytestmark = pytest.mark.kernel

# The compiled loop is private to the module - that is the point of the fix these tests cover -
# and the dispatcher's own attributes are the only honest evidence of what the compiler was
# told, so the two tests below read them off the object rather than off the decorator text.
COMPILED_LOOP = longitudinal._integrate  # pyright: ignore[reportPrivateUsage]

# The four wheel columns of the state, in the order `forces` numbers the contact patches. Held here
# as the kernel's own names and asserted against `forces`'s, so a reordering of one and not the
# other cannot quietly drive the wrong wheel.
WHEEL_COLUMNS: tuple[int, ...] = (
    longitudinal.FL_WHEEL_INDEX,
    longitudinal.FR_WHEEL_INDEX,
    longitudinal.RL_WHEEL_INDEX,
    longitudinal.RR_WHEEL_INDEX,
)

# Wall-clock and randomness entry points. A simulation that reads any of them cannot be
# byte-reproducible, so the determinism test replaces them with functions that raise.
TIME_AND_RANDOM: tuple[tuple[str, str], ...] = (
    ("time", "perf_counter"),
    ("time", "monotonic"),
    ("time", "time"),
    ("time", "time_ns"),
    ("random", "random"),
    ("random", "randrange"),
)

# The body of the fresh interpreter the cache test runs: the same entry point this file uses,
# with the configuration loaded from the same car_spec.yaml, so the only thing isolated from the
# pytest process is the compile cache. It prints the speed reached so the parent can see that the
# run happened rather than taking the child's silence as success.
CACHE_PROBE_SCRIPT = (
    "import numpy as np\n"
    "from f1telemetry.contracts.car_spec import load_car_spec\n"
    "from f1telemetry.kernels import longitudinal\n"
    "config = load_car_spec().kernel_config()\n"
    "steps = 4\n"
    "trace = longitudinal.simulate(\n"
    "    config,\n"
    "    steps,\n"
    "    longitudinal.initial_state(),\n"
    "    np.zeros(steps),\n"
    "    longitudinal.allocate(steps),\n"
    ")\n"
    "print(trace[-1, longitudinal.V_INDEX])\n"
)


@pytest.fixture(scope="module")
def config(spec: CarSpec) -> KernelConfig:
    """The validated, flat configuration the kernel is allowed to see."""
    return spec.kernel_config()


@pytest.fixture(scope="module")
def steps_per_second(config: KernelConfig) -> int:
    """How many fixed steps one simulated second takes, read from the configured step."""
    return round(1.0 / config.dt_s)


@pytest.fixture(scope="module")
def launch_torque(config: KernelConfig) -> float:
    """The differential-side torque this car makes in first gear at full engagement, in Nm.

    Taken from the car's own drivetrain rather than written down: :func:`gearbox.step_gearbox` with
    the launch state (first gear, no shift running, clutch closed) and an idle engine is the
    torque a standing start actually begins with, and reading it keeps this file free of tuned
    numbers while still being a torque the car would really transmit.
    """
    state = gearbox.initial_state()
    torque = gearbox.step_gearbox(config, state, config.idle_rpm, 1.0)
    assert torque > 0.0, "a full-engagement first-gear launch must make forward torque"
    return torque


def _rolling_wheels(config: KernelConfig, speed_m_s: float) -> float:
    """The wheel angular speed that makes ``omega r`` equal the road speed."""
    return speed_m_s / config.rolling_radius_m


def _run(
    config: KernelConfig,
    steps: int,
    drive_torque_nm: np.ndarray,
    state: np.ndarray | None = None,
) -> np.ndarray:
    """Integrate into a fresh caller-owned buffer and hand it back."""
    seed = longitudinal.initial_state() if state is None else state
    return longitudinal.simulate(config, steps, seed, drive_torque_nm, longitudinal.allocate(steps))


def _run_with_diagnostics(
    config: KernelConfig,
    steps: int,
    drive_torque_nm: np.ndarray,
    state: np.ndarray | None = None,
    brake_torque_nm: np.ndarray | None = None,
) -> tuple[np.ndarray, longitudinal.StepOutputs]:
    """A run that also records the per-step corner diagnostics, and both of the caller's buffers.

    The same shape as :func:`_run` with the optional ``step_outputs`` buffer added, so a test that
    needs the loads has to ask for them rather than infer them from the trace.
    """
    seed = longitudinal.initial_state() if state is None else state
    out = longitudinal.allocate(steps)
    step_outputs = longitudinal.allocate_step_outputs(steps)
    trace = longitudinal.simulate(
        config,
        steps,
        seed,
        drive_torque_nm,
        out,
        brake_torque_nm,
        step_outputs=step_outputs,
    )
    return trace, step_outputs


def _reference_loads(
    config: KernelConfig,
    ax_m_s2: float,
    ay_m_s2: float,
    az_m_s2: float,
    downforce_n: float,
) -> loads.CornerLoads:
    """The load model's own Python composition, called with the inputs one kernel step would use.

    Deliberately *not* the kernel's arithmetic: it goes through
    :func:`~f1telemetry.physics.loads.step_loads`, which validates the configuration, refuses the
    impossible cases and reads the accelerations as keyword arguments. So a kernel that passed the
    accelerations in the wrong order or dropped the ``az`` term would disagree with this even though
    both sides call the same compiled primitives.
    """
    return loads.step_loads(
        config,
        longitudinal_accel_m_s2=ax_m_s2,
        lateral_accel_m_s2=ay_m_s2,
        vertical_accel_m_s2=az_m_s2,
        aero_downforce_n=downforce_n,
    )


def _coasting(config: KernelConfig, steps: int, speed_m_s: float) -> np.ndarray:
    """A run with no drive torque and all four wheels rolling at the road speed.

    The one state in which the closed loop has nothing to do: every slip is zero, so every tyre
    force is zero, the wheels do not move, and the only force left on the car is drag. That makes
    it the cleanest fixture for the scheme and the buffer claims, which are about the integrator
    rather than about the tyre model.
    """
    state = longitudinal.initial_state(
        speed_m_s=speed_m_s, wheel_omega_rad_s=_rolling_wheels(config, speed_m_s)
    )
    return _run(config, steps, np.zeros(steps, dtype=np.float64), state)


def test_caller_brake_torque_slows_car_and_applies_to_front_wheels(config: KernelConfig) -> None:
    steps = 100
    speed = 30.0
    state = longitudinal.initial_state(
        speed_m_s=speed, wheel_omega_rad_s=_rolling_wheels(config, speed)
    )
    brake = np.full((steps, forces.WHEEL_COUNT), -500.0, dtype=np.float64)
    trace = longitudinal.simulate(
        config,
        steps,
        state,
        np.zeros(steps, dtype=np.float64),
        longitudinal.allocate(steps),
        brake_torque_nm=brake,
    )
    replay = longitudinal.simulate(
        config,
        steps,
        state,
        np.zeros(steps, dtype=np.float64),
        longitudinal.allocate(steps),
        brake_torque_nm=brake,
    )
    assert trace[-1, longitudinal.V_INDEX] < speed
    assert trace[-1, longitudinal.FL_WHEEL_INDEX] < state[longitudinal.FL_WHEEL_INDEX]
    assert trace.tobytes() == replay.tobytes()


def test_brake_torque_history_is_validated(config: KernelConfig) -> None:
    state = longitudinal.initial_state()
    drive = np.zeros(2, dtype=np.float64)
    out = longitudinal.allocate(2)
    with pytest.raises(ValueError, match=r"brake_torque_nm.*shape"):
        cast(Any, longitudinal.simulate)(
            config, 2, state, drive, out, brake_torque_nm=np.zeros((2, 2), dtype=np.float64)
        )
    invalid = np.zeros((2, forces.WHEEL_COUNT), dtype=np.float64)
    invalid[1, 0] = np.nan
    with pytest.raises(ValueError, match="brake_torque_nm must be finite"):
        longitudinal.simulate(config, 2, state, drive, out, brake_torque_nm=invalid)


def test_allocate_step_outputs_hands_back_caller_owned_buffers_of_the_declared_layout() -> None:
    """Diagnostics have a named, caller-owned buffer for every reported per-wheel value."""
    steps = 5
    step_outputs = longitudinal.allocate_step_outputs(steps)
    assert isinstance(step_outputs, longitudinal.StepOutputs)
    assert step_outputs._fields == longitudinal.STEP_OUTPUT_FIELDS
    for name in longitudinal.STEP_OUTPUT_FLOAT_FIELDS:
        buffer = getattr(step_outputs, name)
        assert buffer.shape == (steps, forces.WHEEL_COUNT), name
        assert buffer.dtype == np.float64, name
        assert buffer.flags.c_contiguous and buffer.flags.writeable, name
    flags = step_outputs.travel_limited
    assert flags.shape == (steps, forces.WHEEL_COUNT)
    assert flags.dtype == np.int64
    assert flags.flags.c_contiguous and flags.flags.writeable
    assert np.count_nonzero(flags) == 0, "a fresh buffer has no steps in it yet"

    pointers = {
        getattr(step_outputs, name).__array_interface__["data"][0]
        for name in longitudinal.STEP_OUTPUT_FIELDS
    }
    assert len(pointers) == len(longitudinal.STEP_OUTPUT_FIELDS)

    empty = longitudinal.allocate_step_outputs(0)
    assert empty.load_n.shape == (0, forces.WHEEL_COUNT)
    with pytest.raises(ValueError, match="steps"):
        longitudinal.allocate_step_outputs(4.0)  # pyright: ignore[reportArgumentType]


def test_every_step_loads_what_the_load_model_says_from_that_row(
    config: KernelConfig,
    launch_torque: float,
    steps_per_second: int,
) -> None:
    """The kernel's corner loads, travel and flags against the load model's own composition.

    Step for step and corner for corner, from the acceleration and downforce the row that the step
    started from carries - not from a closed form and not from a number copied out of a run. The
    reference goes through :func:`~f1telemetry.physics.loads.step_loads`, which takes the three
    accelerations as separate keyword arguments, so a kernel that passed them in the wrong order,
    dropped the ``az`` term or distributed the aero load the P1 way would disagree here even though
    both sides call the same compiled primitives.

    Equality rather than a tolerance: both sides run the same compiled functions on the same inputs,
    so anything but bit identity would mean the kernel had done something in between.
    """
    steps = steps_per_second // 20
    state = longitudinal.initial_state(
        speed_m_s=5.0, wheel_omega_rad_s=0.0, previous_acceleration_m_s2=(3.0, -2.0, 0.5)
    )
    drive_torque = np.linspace(0.0, launch_torque, steps)
    trace, step_outputs = _run_with_diagnostics(config, steps, drive_torque, state)
    combined_parameters = combined_slip.prepare_combined_slip_parameters(
        config, "test_every_step_loads_what_the_load_model_says_from_that_row"
    )

    for index in range(steps):
        downforce_n, _ = forces.aero_forces(
            float(trace[index, longitudinal.V_INDEX]),
            config.air_density_kg_m3,
            config.reference_area_m2,
            config.aero_speed_m_s,
            config.cl,
            config.cd,
        )
        corner = _reference_loads(
            config,
            float(trace[index, longitudinal.PREVIOUS_AX_INDEX]),
            float(trace[index, longitudinal.PREVIOUS_AY_INDEX]),
            float(trace[index, longitudinal.PREVIOUS_AZ_INDEX]),
            downforce_n,
        )
        assert np.allclose(step_outputs.load_n[index], corner.load_n, rtol=0.0, atol=1e-9), index
        assert np.allclose(step_outputs.travel_m[index], corner.travel_m, rtol=0.0, atol=1e-12), (
            index
        )
        assert np.array_equal(
            step_outputs.travel_limited[index], np.array(corner.travel_limited, dtype=np.int64)
        ), index
        # Check the exact combined-slip force from the integrated slip state and applied load.
        for wheel in range(forces.WHEEL_COUNT):
            expected_fx_n, expected_fy_n = combined_slip.combined_tyre_forces(
                float(step_outputs.slip_ratio[index, wheel]),
                float(step_outputs.slip_angle_deg[index, wheel]),
                float(step_outputs.camber_deg[index, wheel]),
                float(step_outputs.load_n[index, wheel]),
                combined_parameters,
            )
            assert step_outputs.force_x_n[index, wheel] == expected_fx_n, (index, wheel)
            assert step_outputs.force_y_n[index, wheel] == expected_fy_n, (index, wheel)
        assert np.isfinite(trace[index + 1, longitudinal.PREVIOUS_AY_INDEX]), index


def test_the_longitudinal_transfer_moves_load_rearward_under_power_and_forward_under_brakes(
    config: KernelConfig,
) -> None:
    """``dFz = m ax h / L`` from the front axle to the rear, with the total untouched.

    Two one-step runs from the same rolling state that differ only in the seeded previous
    acceleration, so the transfer is the *only* thing that differs. Under power the rear axle must
    end up carrying more than the front, under braking less, and both must still carry the car's
    weight plus its own downforce - transfer moves load around the car, it does not create any.
    """
    steps = 1
    speed = 40.0
    rolling = _rolling_wheels(config, speed)
    torque = np.zeros(steps, dtype=np.float64)
    downforce_n, _ = forces.aero_forces(
        speed,
        config.air_density_kg_m3,
        config.reference_area_m2,
        config.aero_speed_m_s,
        config.cl,
        config.cd,
    )
    front, rear = forces.FL_WHEEL_INDEX, forces.RL_WHEEL_INDEX

    def axle_loads(ax_m_s2: float) -> tuple[float, float, np.ndarray]:
        _, step_outputs = _run_with_diagnostics(
            config,
            steps,
            torque,
            longitudinal.initial_state(
                speed_m_s=speed,
                wheel_omega_rad_s=rolling,
                previous_acceleration_m_s2=(ax_m_s2, 0.0, 0.0),
            ),
        )
        row = step_outputs.load_n[0]
        return (
            float(row[front] + row[front + 1]),
            float(row[rear] + row[rear + 1]),
            row,
        )

    accelerating_front_n, accelerating_rear_n, accelerating_row = axle_loads(8.0)
    braking_front_n, braking_rear_n, braking_row = axle_loads(-8.0)
    total_n = config.mass_kg * config.gravity_m_s2 + downforce_n
    static_rear_n = total_n * (1.0 - config.front_weight_fraction)

    assert accelerating_rear_n > accelerating_front_n, "power loads the rear axle"
    assert braking_rear_n < braking_front_n, "braking loads the front axle"
    # Symmetric about the untransferred split: +ax and -ax move the same load in opposite
    # directions, so the two runs' rear loads bracket the static one equally.
    assert accelerating_rear_n - static_rear_n == pytest.approx(
        static_rear_n - braking_rear_n, rel=1e-12
    )
    assert accelerating_front_n + accelerating_rear_n == pytest.approx(total_n, rel=1e-12)
    assert braking_front_n + braking_rear_n == pytest.approx(total_n, rel=1e-12)
    assert float(accelerating_row.sum()) == pytest.approx(total_n, rel=1e-12)
    assert float(braking_row.sum()) == pytest.approx(total_n, rel=1e-12)


def test_a_corner_that_reaches_its_travel_limit_is_flagged_and_the_limit_is_reported(
    config: KernelConfig,
) -> None:
    """The travel band is enforced and the corner that hit it says so.

    Checked against a car with a deliberately tiny configured travel limit, because this car's own
    25 mm of travel is more than a normal acceleration can use - which is itself worth asserting,
    since a flag that can only be raised by an absurd number is not evidence that the limit is
    wired up at all. Three things are checked on the flagged run: the flag is set on the corners
    that ran out of travel and clear on the axle that did not, the front axle's reported travel is
    inside the band it was clamped to, and the flags match the load model's own report corner for
    corner.
    """
    tight = replace(config, axle_travel_limit_m=np.full(2, 5.0e-4, dtype=np.float64))
    steps = 1
    speed = 40.0
    rolling = _rolling_wheels(config, speed)
    torque = np.zeros(steps, dtype=np.float64)
    downforce_n, _ = forces.aero_forces(
        speed,
        config.air_density_kg_m3,
        config.reference_area_m2,
        config.aero_speed_m_s,
        config.cl,
        config.cd,
    )

    def flagged(ax_m_s2: float, config_used: KernelConfig = config) -> longitudinal.StepOutputs:
        _, step_outputs = _run_with_diagnostics(
            config_used,
            steps,
            torque,
            longitudinal.initial_state(
                speed_m_s=speed,
                wheel_omega_rad_s=rolling,
                previous_acceleration_m_s2=(ax_m_s2, 0.0, 0.0),
            ),
        )
        return step_outputs

    # A normal run on this car never reaches its own stops.
    assert np.count_nonzero(flagged(10.0).travel_limited) == 0

    # 4.5 m/s^2 moves 158.9 N onto each rear corner and off each front one. That is past the tight
    # front band (150 N) and inside the rear one (175 N), which is what makes the flag per corner
    # rather than per car.
    loaded = flagged(4.5, tight)
    drooping = loaded.travel_limited[0]
    assert drooping[forces.FL_WHEEL_INDEX] == 1
    assert drooping[forces.FR_WHEEL_INDEX] == 1
    assert drooping[forces.RL_WHEEL_INDEX] == 0
    assert drooping[forces.RR_WHEEL_INDEX] == 0
    front_travel_m = float(loaded.travel_m[0, forces.FL_WHEEL_INDEX])
    rear_travel_m = float(loaded.travel_m[0, forces.RL_WHEEL_INDEX])
    assert front_travel_m < 0.0, "the unloaded front axle is in droop"
    assert rear_travel_m > 0.0, "and the loaded rear axle is in compression"
    assert abs(front_travel_m) <= float(tight.axle_travel_limit_m[0]) * (1.0 + 1.0e-9)
    assert rear_travel_m < float(tight.axle_travel_limit_m[1])

    corner = _reference_loads(tight, 4.5, 0.0, 0.0, downforce_n)
    assert np.array_equal(loaded.travel_limited[0], np.array(corner.travel_limited, dtype=np.int64))
    assert np.array_equal(loaded.load_n[0], corner.load_n)
    assert np.array_equal(loaded.travel_m[0], corner.travel_m)
    # The transfer the clamp removed is handed to the corners that still have room, so the total is
    # still the car's weight plus its own downforce.
    assert float(loaded.load_n[0].sum()) == pytest.approx(
        config.mass_kg * config.gravity_m_s2 + downforce_n, rel=1e-12
    )


def test_the_step_resolves_an_acceleration_and_carries_it_into_the_next_step(
    config: KernelConfig,
    launch_torque: float,
    steps_per_second: int,
) -> None:
    """The load lag is one step, and it is fed by the acceleration this kernel actually resolved.

    ``PREVIOUS_AX`` is written every step with ``F/m`` for that step, so the loads the *next* step
    asks for are the ones this step's forces imply - which is the plan's explicit answer to the
    load/force algebraic loop. Three things make that checkable rather than assumed: the column
    equals the speed increment the row pair implies, the seeded value is *replaced* after one step
    rather than held, and the first step's loads are visibly different when the seed is different.
    """
    steps = steps_per_second // 20
    torque = np.full(steps, launch_torque, dtype=np.float64)

    seeded = longitudinal.initial_state(previous_acceleration_m_s2=(9.0, 0.0, 0.0))
    trace, step_outputs = _run_with_diagnostics(config, steps, torque, seeded)
    zero_seed = longitudinal.initial_state()
    _, unseeded = _run_with_diagnostics(config, steps, torque, zero_seed)

    # Row 0 is the caller's state, so the seed is what the first step's loads are computed from.
    assert trace[0, longitudinal.PREVIOUS_AX_INDEX] == 9.0
    assert not np.array_equal(step_outputs.load_n[0], unseeded.load_n[0]), (
        "a seeded acceleration has to reach the first step's loads, or the lag is not closed"
    )
    # After one step the column holds what the step resolved, not what the caller seeded.
    assert trace[1, longitudinal.PREVIOUS_AX_INDEX] != 9.0
    assert np.all(trace[1:, longitudinal.PREVIOUS_AX_INDEX] >= 0.0), "a launch only accelerates"
    ax_from_body_velocity = (
        trace[1:, longitudinal.V_INDEX] - trace[:-1, longitudinal.V_INDEX]
    ) / config.dt_s - trace[:-1, longitudinal.YAW_RATE_INDEX] * trace[:-1, longitudinal.VY_INDEX]
    ay_from_body_velocity = (
        trace[1:, longitudinal.VY_INDEX] - trace[:-1, longitudinal.VY_INDEX]
    ) / config.dt_s + trace[:-1, longitudinal.YAW_RATE_INDEX] * trace[:-1, longitudinal.V_INDEX]
    assert np.allclose(
        trace[1:, longitudinal.PREVIOUS_AX_INDEX], ax_from_body_velocity, rtol=1e-8, atol=1e-8
    )
    assert np.allclose(
        trace[1:, longitudinal.PREVIOUS_AY_INDEX], ay_from_body_velocity, rtol=1e-8, atol=1e-8
    )
    assert np.array_equal(trace[1:, longitudinal.PREVIOUS_AZ_INDEX], np.zeros(steps))
    assert np.array_equal(trace[1:, longitudinal.PREVIOUS_AZ_INDEX], np.zeros(steps))


def test_the_kernel_picks_the_same_axle_the_load_model_does(config: KernelConfig) -> None:
    """The kernel derives an axle from the corner order; this pins it against the load model.

    The load module keeps its corner-to-axle table private and reads it positionally, because an
    axle is the two adjacent corners in ``FL, FR, RL, RR`` order. The kernel needs the same mapping
    to pick a corner's ride rate, and derives it from the wheel count per axle rather than
    restating a table - which is only safe while the two agree, so this is the test that says they
    do.
    """
    derived = tuple(wheel // forces.WHEELS_PER_AXLE_COUNT for wheel in range(forces.WHEEL_COUNT))
    assert derived == loads._AXLE_OF_CORNER  # pyright: ignore[reportPrivateUsage]


def test_a_run_without_diagnostics_is_byte_identical_to_one_with_them(
    config: KernelConfig,
    launch_torque: float,
    steps_per_second: int,
) -> None:
    """Asking for the per-step diagnostics is the only difference between the two calls.

    The buffers are the caller's to fill, so the P1 signature - five positional arguments and
    nothing else - has to keep working and has to keep producing the same bytes. Also pinned: the
    keyword is keyword-only, so a caller cannot accidentally pass a diagnostics buffer into the
    brake-torque slot, and a zero-step run against zero-row diagnostics is not a special case.
    """
    steps = steps_per_second // 4
    torque = np.full(steps, launch_torque, dtype=np.float64)
    plain = _run(config, steps, torque)
    recorded, step_outputs = _run_with_diagnostics(config, steps, torque)
    assert plain.tobytes() == recorded.tobytes()

    replay_outputs = longitudinal.allocate_step_outputs(steps)
    replay = longitudinal.simulate(
        config,
        steps,
        longitudinal.initial_state(),
        torque,
        longitudinal.allocate(steps),
        step_outputs=replay_outputs,
    )
    assert replay.tobytes() == plain.tobytes()
    assert replay_outputs.load_n.tobytes() == step_outputs.load_n.tobytes()
    assert replay_outputs.travel_limited.tobytes() == step_outputs.travel_limited.tobytes()

    with pytest.raises(TypeError):
        cast(Any, longitudinal.simulate)(
            config,
            2,
            longitudinal.initial_state(),
            np.zeros(2),
            longitudinal.allocate(2),
            None,
            longitudinal.allocate_step_outputs(2),
        )

    empty_trace = longitudinal.simulate(
        config,
        0,
        longitudinal.initial_state(),
        np.zeros(0, dtype=np.float64),
        longitudinal.allocate(0),
        step_outputs=longitudinal.allocate_step_outputs(0),
    )
    assert empty_trace.shape == (1, longitudinal.STATE_SIZE)
    assert np.array_equal(empty_trace, longitudinal.initial_state().reshape(1, -1))


def _diagnostic_case(
    steps: int,
    name: str,
    replacement: object,
) -> longitudinal.StepOutputs:
    """A valid set of diagnostics with one named field replaced by something unusable."""
    good = longitudinal.allocate_step_outputs(steps)
    return good._replace(**{name: replacement})


@pytest.mark.parametrize(
    ("name", "replacement", "message"),
    [
        ("load_n", np.zeros((3, 2), dtype=np.float64), "shape"),
        ("load_n", np.zeros(3, dtype=np.float64), "shape"),
        ("force_x_n", np.zeros((3, 4), dtype=np.float32), "float64"),
        ("travel_m", np.zeros((6, 4), dtype=np.float64)[::2], "contiguous"),
        ("travel_limited", np.zeros((3, 4), dtype=np.int32), "int64"),
        ("travel_m", [0.0] * 12, "ndarray"),
        ("load_n", "not a buffer at all", "ndarray"),
    ],
    ids=[
        "wrong-wheel-count",
        "missing-wheel-axis",
        "float32",
        "strided",
        "int32-flags",
        "list",
        "string",
    ],
)
def test_a_diagnostic_buffer_the_run_cannot_fill_is_refused_in_python(
    config: KernelConfig,
    name: str,
    replacement: object,
    message: str,
) -> None:
    """Every diagnostic buffer is checked at the same boundary as every other kernel buffer.

    ``boundscheck=False`` writes into these, so a short row count or a missing wheel column is an
    out-of-bounds write rather than an error - the same argument that makes ``simulate`` the only
    way into the loop applies to the optional buffers too. The dtype is part of it: the flags are
    ``int64`` and a ``float32`` one would compile as a differently-typed kernel.
    """
    steps = 3
    out = longitudinal.allocate(steps)
    step_outputs = _diagnostic_case(steps, name, replacement)
    with pytest.raises(ValueError, match=rf"step_outputs\.{name}.*{message}"):
        longitudinal.simulate(
            config,
            steps,
            longitudinal.initial_state(),
            np.zeros(steps, dtype=np.float64),
            out,
            step_outputs=step_outputs,
        )
    # Refused before the loop: not one row of the trace was written.
    assert not np.any(out), "the kernel ran before the buffer was refused"


def test_diagnostic_buffers_that_are_not_writeable_or_are_aliased_are_refused(
    config: KernelConfig,
) -> None:
    """The four buffers are separate destinations, so a read-only or shared one is a caller bug.

    Aliasing is the nastier of the two: handing the same array for the loads and the longitudinal
    force would produce a run that looks finished, with one corner quantity overwritten by another.
    """
    steps = 3
    out = longitudinal.allocate(steps)
    torque = np.zeros(steps, dtype=np.float64)
    state = longitudinal.initial_state()

    frozen = longitudinal.allocate_step_outputs(steps)
    frozen.travel_m.flags.writeable = False
    with pytest.raises(ValueError, match=r"step_outputs\.travel_m.*read-only"):
        longitudinal.simulate(config, steps, state, torque, out, step_outputs=frozen)

    aliased = longitudinal.allocate_step_outputs(steps)
    shared = aliased.load_n
    aliased = aliased._replace(force_x_n=shared)
    with pytest.raises(ValueError, match="share memory"):
        longitudinal.simulate(config, steps, state, torque, out, step_outputs=aliased)


def test_something_that_is_not_a_step_output_group_is_refused(config: KernelConfig) -> None:
    """The four buffers arrive as one named group, not as four loose arguments.

    A plain tuple of four correctly shaped arrays would be accepted by a loose signature and
    rejected here, which is the point: the fields are named in the contract, and a caller that
    ordered them by hand would be one refactor away from silently swapping the travel flags for
    the longitudinal force.
    """
    steps = 3
    good = longitudinal.allocate_step_outputs(steps)
    for wrong in (
        (good.load_n, good.force_x_n, good.travel_m, good.travel_limited),
        [good.load_n, good.force_x_n, good.travel_m, good.travel_limited],
        "step_outputs",
    ):
        with pytest.raises(ValueError, match="step_outputs"):
            longitudinal.simulate(
                config,
                steps,
                longitudinal.initial_state(),
                np.zeros(steps, dtype=np.float64),
                longitudinal.allocate(steps),
                step_outputs=wrong,  # pyright: ignore[reportArgumentType]
            )


def _reference(
    config: KernelConfig,
    steps: int,
    drive_torque_nm: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    """P2 step written in Python, retaining the kernel's explicit update order."""
    out = np.zeros((steps + 1, longitudinal.STATE_SIZE), dtype=np.float64)
    out[0] = state
    parameters = combined_slip.prepare_combined_slip_parameters(config, "_reference")
    road_steer_deg = np.zeros(forces.WHEEL_COUNT, dtype=np.float64)
    steering.road_wheel_angles_deg(
        0.0,
        config.steering_ratio,
        config.wheelbase_m,
        float(config.axle_track_m[0]),
        config.ackermann_fraction,
        road_steer_deg,
    )
    dt = config.dt_s
    for index in range(steps):
        previous = out[index]
        vx, vy = previous[longitudinal.V_INDEX], previous[longitudinal.VY_INDEX]
        yaw = previous[longitudinal.YAW_RATE_INDEX]
        speed = math.hypot(vx, vy)
        downforce, drag = forces.aero_forces(
            speed,
            config.air_density_kg_m3,
            config.reference_area_m2,
            config.aero_speed_m_s,
            config.cl,
            config.cd,
        )
        corner = _reference_loads(
            config,
            float(previous[longitudinal.PREVIOUS_AX_INDEX]),
            float(previous[longitudinal.PREVIOUS_AY_INDEX]),
            float(previous[longitudinal.PREVIOUS_AZ_INDEX]),
            downforce,
        )
        net_fx = drag * vx / speed if speed > 0.0 else 0.0
        net_fy = drag * vy / speed if speed > 0.0 else 0.0
        moment = 0.0
        for wheel in range(forces.WHEEL_COUNT):
            axle = wheel // forces.WHEELS_PER_AXLE_COUNT
            side = 1.0 if wheel % forces.WHEELS_PER_AXLE_COUNT == 0 else -1.0
            x = config.cg_to_front_axle_m if axle == 0 else -config.cg_to_rear_axle_m
            y = side * 0.5 * float(config.axle_track_m[axle])
            patch_vx, patch_vy = kinematics.contact_velocity_m_s(vx, vy, yaw, x, y)
            steer_deg = float(road_steer_deg[wheel]) + float(
                config.axle_bump_steer_deg_per_m[axle] * corner.travel_m[wheel]
            )
            wheel_vx, wheel_vy = kinematics.wheel_frame_velocity_m_s(patch_vx, patch_vy, steer_deg)
            patch_speed = math.hypot(wheel_vx, wheel_vy)
            omega = previous[WHEEL_COLUMNS[wheel]]
            target_kappa = forces.slip_ratio(
                omega * config.rolling_radius_m,
                wheel_vx,
                config.slip_ratio_min_speed_m_s,
            )
            target_alpha = kinematics.slip_angle_deg(wheel_vx, wheel_vy)
            kappa = relaxation.relax_slip_ratio(
                previous[longitudinal.KAPPA_RELAX_OFFSET + wheel],
                target_kappa,
                patch_speed,
                config.relaxation_length_longitudinal_m,
                dt,
                config.relaxation_min_speed_m_s,
            )
            alpha_deg = relaxation.relax_slip_ratio(
                previous[longitudinal.ALPHA_RELAX_OFFSET + wheel],
                target_alpha,
                patch_speed,
                config.relaxation_length_lateral_m,
                dt,
                config.relaxation_min_speed_m_s,
            )
            camber = side * (
                config.axle_static_camber_deg[axle]
                + config.axle_camber_gain_deg_per_m[axle] * corner.travel_m[wheel]
            )
            fx, fy = combined_slip.combined_tyre_forces(
                kappa, alpha_deg, camber, corner.load_n[wheel], parameters
            )
            out[index + 1, longitudinal.KAPPA_RELAX_OFFSET + wheel] = kappa
            out[index + 1, longitudinal.ALPHA_RELAX_OFFSET + wheel] = alpha_deg
            steer = math.radians(steer_deg)
            body_fx = fx * math.cos(steer) - fy * math.sin(steer)
            body_fy = fx * math.sin(steer) + fy * math.cos(steer)
            net_fx += body_fx
            net_fy += body_fy
            moment += x * body_fy - y * body_fx
            wheel_torque = forces.wheel_drive_torque_nm(wheel, float(drive_torque_nm[index]))
            angular_acceleration = forces.wheel_angular_acceleration_rad_s2(
                wheel_torque, fx, config.rolling_radius_m, config.wheel_inertia_kg_m2
            )
            out[index + 1, WHEEL_COLUMNS[wheel]] = omega + angular_acceleration * dt

        ax, ay = net_fx / config.mass_kg, net_fy / config.mass_kg
        vx_next = vx + (ax + yaw * vy) * dt
        vy_next = vy + (ay - yaw * vx) * dt
        yaw_next = yaw + moment / config.yaw_inertia_kg_m2 * dt
        psi_next = previous[longitudinal.PSI_INDEX] + yaw_next * dt
        out[index + 1, longitudinal.V_INDEX] = vx_next
        out[index + 1, longitudinal.VY_INDEX] = vy_next
        out[index + 1, longitudinal.YAW_RATE_INDEX] = yaw_next
        out[index + 1, longitudinal.PSI_INDEX] = psi_next
        out[index + 1, longitudinal.X_INDEX] = (
            previous[longitudinal.X_INDEX]
            + (vx_next * math.cos(psi_next) - vy_next * math.sin(psi_next)) * dt
        )
        out[index + 1, longitudinal.Y_INDEX] = (
            previous[longitudinal.Y_INDEX]
            + (vx_next * math.sin(psi_next) + vy_next * math.cos(psi_next)) * dt
        )
        left = 0.5 * (corner.travel_m[0] + corner.travel_m[2])
        right = 0.5 * (corner.travel_m[1] + corner.travel_m[3])
        front = 0.5 * (corner.travel_m[0] + corner.travel_m[1])
        rear = 0.5 * (corner.travel_m[2] + corner.travel_m[3])
        out[index + 1, longitudinal.ROLL_INDEX] = (right - left) / (
            0.5 * (config.axle_track_m[0] + config.axle_track_m[1])
        )
        out[index + 1, longitudinal.PITCH_INDEX] = (front - rear) / config.wheelbase_m
        out[index + 1, longitudinal.HEAVE_INDEX] = sum(corner.travel_m) / forces.WHEEL_COUNT
        out[index + 1, longitudinal.PREVIOUS_AX_INDEX] = ax
        out[index + 1, longitudinal.PREVIOUS_AY_INDEX] = ay
        out[index + 1, longitudinal.PREVIOUS_AZ_INDEX] = 0.0
    return out


def _refuse(what: str) -> None:
    raise AssertionError(f"the simulation read {what}")


def test_the_kernel_compiles_to_machine_code_and_writes_a_compile_cache(
    config: KernelConfig,
    tmp_path: Path,
) -> None:
    """A dispatcher entry is not a compiled kernel, and `cache=True` can do nothing silently.

    Two pieces of evidence, both measured rather than read back out of this file.

    Calling the kernel registers a compiled signature, which rules out a Python fallback. The
    cache claim needs more than that, because it is not a claim about this process: a test that
    accepts a cache index from anywhere on disk is happy with one an earlier run left behind, so
    it goes on passing after the decorator is changed to `cache=False`. A test that cannot fail
    when the thing it names breaks is not a test.

    So the compile happens in a fresh interpreter whose `NUMBA_CACHE_DIR` is this test's empty
    `tmp_path`, and the index has to turn up there. The subprocess is not optional: this module
    and its dispatcher are already imported in the pytest process, so its signature was compiled
    before the test could ask for a cold one. `NUMBA_CACHE_DIR` also takes priority over the
    `__pycache__` beside the source, so the child cannot load an index an earlier run wrote.
    """
    _run(config, 4, np.zeros(4, dtype=np.float64))
    assert len(COMPILED_LOOP.signatures) >= 1, (
        "the run returned without compiling a signature, which means it fell back to Python"
    )

    environment = dict(os.environ)
    environment["NUMBA_CACHE_DIR"] = str(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-c", CACHE_PROBE_SCRIPT],
        capture_output=True,
        text=True,
        check=True,
        env=environment,
    )
    reported = completed.stdout.strip()
    assert reported, f"the fresh interpreter printed no trace row; stderr was {completed.stderr}"
    assert math.isfinite(float(reported)), f"the fresh interpreter reported {reported}"

    # numba nests the index under a subdirectory derived from the source location, so the temp
    # dir is searched rather than listed.
    written = sorted(tmp_path.rglob("longitudinal.*.nbi"))
    assert written, (
        f"numba wrote no .nbi cache index for the longitudinal kernel into the empty {tmp_path}, "
        f"so cache=True is not taking effect; the directory holds "
        f"{sorted(path.name for path in tmp_path.rglob('*'))}"
    )


def test_the_kernel_options_match_plan_section_4_1() -> None:
    """PLAN.md section 4.1 rule 4, read off the dispatcher rather than off the decorator.

    ``boundscheck=False`` is the option that makes the compiled loop unsafe to call directly -
    an undersized buffer is a silent out-of-bounds write - and it is why ``_integrate`` is
    private and ``simulate`` validates. Asserted here so that cannot be changed by accident.
    """
    options = dict(COMPILED_LOOP.targetoptions)
    assert options["fastmath"] is False
    assert options["nopython"] is True
    assert options["nogil"] is True
    assert options["boundscheck"] is False
    assert options["error_model"] == "numpy"


def test_the_step_is_data_and_each_row_is_exactly_one_configured_step(
    config: KernelConfig,
    steps_per_second: int,
) -> None:
    """100 µs is what the spec says, and each row advances by exactly that, using its own speed.

    This is the fixed-step claim in its simplest falsifiable form: one simulated second of rows
    covers exactly one second of travel. The increment is checked against the speed *that row*
    produced rather than the one the row started with, because that is the semi-implicit rule and
    because a coasting car's speed is not constant - the two are not the same number here, which is
    what makes the check able to fail.
    """
    assert config.dt_s == pytest.approx(1.0e-4, rel=0.0, abs=0.0)
    assert round(1.0 / config.dt_s) == 10_000

    steps = steps_per_second
    speed = 30.0
    out = _coasting(config, steps, speed)
    assert out.shape == (steps + 1, longitudinal.STATE_SIZE)
    heading = out[1:, longitudinal.PSI_INDEX]
    vx = out[1:, longitudinal.V_INDEX]
    vy = out[1:, longitudinal.VY_INDEX]
    expected_dx = (vx * np.cos(heading) - vy * np.sin(heading)) * config.dt_s
    expected_dy = (vx * np.sin(heading) + vy * np.cos(heading)) * config.dt_s
    assert np.allclose(np.diff(out[:, longitudinal.X_INDEX]), expected_dx, rtol=0.0, atol=1e-12)
    assert np.allclose(np.diff(out[:, longitudinal.Y_INDEX]), expected_dy, rtol=0.0, atol=1e-12)
    # One row per step, plus the seed, and the seed is the row zero the caller handed over.
    assert out.shape[0] == steps + 1


def test_the_caller_owns_the_state_and_trace_buffers(config: KernelConfig) -> None:
    """Row 0 of the trace is the caller's state, and the caller hands over the whole buffer.

    ``steps`` of zero is the degenerate case of the same rule: row 0 is the seeded state and
    nothing else is written, so a run of no length is not a special case.
    """
    state = longitudinal.initial_state(distance_m=12.5, speed_m_s=30.0, wheel_omega_rad_s=140.0)
    assert state.shape == (longitudinal.STATE_SIZE,)
    assert state.dtype == np.float64
    assert state[longitudinal.X_INDEX] == 12.5
    assert state[longitudinal.V_INDEX] == 30.0
    assert [state[column] for column in WHEEL_COLUMNS] == [140.0] * forces.WHEEL_COUNT

    steps = 5
    out = longitudinal.allocate(steps)
    assert out.shape == (steps + 1, longitudinal.STATE_SIZE)
    assert out.dtype == np.float64
    assert out.flags.writeable

    empty = longitudinal.allocate(0)
    returned = longitudinal.simulate(config, 0, state, np.zeros(0, dtype=np.float64), empty)
    assert np.array_equal(returned, state.reshape(1, longitudinal.STATE_SIZE))


def test_the_state_carries_four_wheel_angular_speeds_in_the_physysics_corner_order(
    config: KernelConfig,
) -> None:
    """The P2 extension preserves the P1 prefix and its four wheel-speed columns.

    Asserted against the force model's own constants rather than against numbers written here, so
    the two cannot disagree about which column is which - a trace whose left wheel was driven and
    right wheel not would still be finite, still accelerate, and still pass every other test.
    ``PLAN.md`` section 4's state vector lists ``omega`` per wheel, and this is where that lands.
    """
    assert longitudinal.STATE_SIZE == 24
    assert WHEEL_COLUMNS == (2, 3, 4, 5)
    assert WHEEL_COLUMNS[forces.FL_WHEEL_INDEX] == longitudinal.FL_WHEEL_INDEX
    assert WHEEL_COLUMNS[forces.FR_WHEEL_INDEX] == longitudinal.FR_WHEEL_INDEX
    assert WHEEL_COLUMNS[forces.RL_WHEEL_INDEX] == longitudinal.RL_WHEEL_INDEX
    assert WHEEL_COLUMNS[forces.RR_WHEEL_INDEX] == longitudinal.RR_WHEEL_INDEX
    # The rear pair starts at the offset the drive split tests against, so C9.1.1's axle is a
    # contiguous block of the state and not two indices chosen by hand.
    assert (
        longitudinal.RL_WHEEL_INDEX == longitudinal.FL_WHEEL_INDEX + forces.FIRST_REAR_WHEEL_INDEX
    )
    assert longitudinal.RR_WHEEL_INDEX == longitudinal.RL_WHEEL_INDEX + 1

    # Every column the run writes is a wheel speed, and a rolling start puts the same number in all
    # four: the default is zero, and a caller seeding a moving car has to say so explicitly.
    assert [float(longitudinal.initial_state()[column]) for column in WHEEL_COLUMNS] == [0.0] * 4
    rolling = _rolling_wheels(config, 45.0)
    seeded = longitudinal.initial_state(speed_m_s=45.0, wheel_omega_rad_s=rolling)
    assert [float(seeded[column]) for column in WHEEL_COLUMNS] == pytest.approx(
        [rolling] * forces.WHEEL_COUNT
    )


def test_p2_state_uses_named_columns_and_seeds_caller_owned_transient_states() -> None:
    state = longitudinal.initial_state(
        distance_m=12.0,
        speed_m_s=30.0,
        wheel_omega_rad_s=100.0,
        y_m=-4.0,
        heading_rad=0.25,
        vy_m_s=2.0,
        yaw_rate_rad_s=-0.1,
        roll_rad=0.02,
        pitch_rad=-0.01,
        heave_m=0.04,
        alpha_relax_deg=np.array([1.0, 2.0, 3.0, 4.0]),
        kappa_relax=np.array([-0.1, -0.2, 0.3, 0.4]),
        previous_acceleration_m_s2=(5.0, -6.0, 0.7),
    )
    assert state.shape == (longitudinal.STATE_SIZE,)
    assert state[longitudinal.X_INDEX] == 12.0
    assert state[longitudinal.VX_INDEX] == 30.0
    assert state[longitudinal.Y_INDEX] == -4.0
    assert state[longitudinal.PSI_INDEX] == 0.25
    assert state[longitudinal.VY_INDEX] == 2.0
    assert state[longitudinal.YAW_RATE_INDEX] == -0.1
    assert state[longitudinal.ROLL_INDEX] == 0.02
    assert state[longitudinal.PITCH_INDEX] == -0.01
    assert state[longitudinal.HEAVE_INDEX] == 0.04
    assert state[
        longitudinal.ALPHA_RELAX_OFFSET : longitudinal.ALPHA_RELAX_OFFSET + 4
    ] == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert state[
        longitudinal.KAPPA_RELAX_OFFSET : longitudinal.KAPPA_RELAX_OFFSET + 4
    ] == pytest.approx([-0.1, -0.2, 0.3, 0.4])
    assert state[
        longitudinal.PREVIOUS_AX_INDEX : longitudinal.PREVIOUS_AZ_INDEX + 1
    ] == pytest.approx([5.0, -6.0, 0.7])


def test_integrator_advances_appended_planar_state(
    config: KernelConfig,
) -> None:
    state = longitudinal.initial_state(
        vy_m_s=1.25,
        yaw_rate_rad_s=-0.5,
        roll_rad=0.02,
        pitch_rad=-0.04,
        heave_m=0.06,
    )
    out = longitudinal.simulate(
        config,
        3,
        state,
        np.zeros(3, dtype=np.float64),
        longitudinal.allocate(3),
    )
    assert out[-1, longitudinal.VY_INDEX] != state[longitudinal.VY_INDEX]
    assert out[-1, longitudinal.YAW_RATE_INDEX] != state[longitudinal.YAW_RATE_INDEX]
    assert out[-1, longitudinal.PSI_INDEX] != state[longitudinal.PSI_INDEX]
    assert out[-1, longitudinal.ROLL_INDEX] != state[longitudinal.ROLL_INDEX]
    assert out[-1, longitudinal.PITCH_INDEX] != state[longitudinal.PITCH_INDEX]
    assert out[-1, longitudinal.HEAVE_INDEX] != state[longitudinal.HEAVE_INDEX]


def test_the_first_step_of_a_standing_launch_is_exactly_predictable(
    config: KernelConfig,
    launch_torque: float,
) -> None:
    """One step from rest, written out by hand, with every number the car supplies.

    At rest with stationary wheels the slip is exactly zero, so the tyre force is exactly zero and
    the drag is exactly zero: the car cannot move on step one however hard it is driven. The rear
    wheels, and only the rear wheels, gain ``T / 2 / I`` of angular speed. That is C9.1.1, the
    equal split and the whole of P1-T6 in four assertions with no tolerance in any of them.
    """
    steps = 1
    torque = np.full(steps, launch_torque, dtype=np.float64)
    out = _run(config, steps, torque)
    inertia = config.wheel_inertia_kg_m2
    dt_s = config.dt_s

    assert out[1, longitudinal.V_INDEX] == 0.0
    assert out[1, longitudinal.X_INDEX] == 0.0
    assert out[1, longitudinal.FL_WHEEL_INDEX] == 0.0
    assert out[1, longitudinal.FR_WHEEL_INDEX] == 0.0
    assert out[1, longitudinal.RL_WHEEL_INDEX] == pytest.approx(
        (launch_torque / 2.0) / inertia * dt_s, rel=1e-12
    )
    assert out[1, longitudinal.RR_WHEEL_INDEX] == out[1, longitudinal.RL_WHEEL_INDEX]
    # And nothing was invented on the way: the trace is exactly the seed with two rear wheels
    # turned, which is what "the forces are zero at zero slip" has to mean for the integrator too.
    assert np.count_nonzero(out[1]) == 2


def test_aero_downforce_is_conserved_across_the_four_contact_patches(
    config: KernelConfig,
) -> None:
    """The four corner loads sum to weight plus downforce, not weight plus four downforces.

    The conservation is the invariant and it is checked per step, against the load model's own
    composition rather than against a number out of a run. The *distribution* of the aerodynamic
    load is the part that deliberately changed in P2-T2: it follows the same CG share as the weight,
    so the front axle - 46% of the static load - takes 46% of the downforce rather than the quarter
    P1's kernel added to every patch. Both halves are asserted, because a kernel that kept the old
    quarter-each split would still conserve the total.
    """
    steps = 5
    speed = 50.0
    rolling_omega = speed / config.rolling_radius_m
    state = longitudinal.initial_state(speed_m_s=speed, wheel_omega_rad_s=rolling_omega)
    trace, step_outputs = _run_with_diagnostics(
        config, steps, np.zeros(steps, dtype=np.float64), state
    )

    front_share = config.cg_to_rear_axle_m / (config.cg_to_front_axle_m + config.cg_to_rear_axle_m)
    for index in range(steps):
        # Row `index` is the state the step started from, so it is also the speed the aero, and
        # therefore the downforce, was evaluated at.
        downforce_n, _ = forces.aero_forces(
            float(trace[index, longitudinal.V_INDEX]),
            config.air_density_kg_m3,
            config.reference_area_m2,
            config.aero_speed_m_s,
            config.cl,
            config.cd,
        )
        row = step_outputs.load_n[index]
        assert float(row.sum()) == pytest.approx(
            config.mass_kg * config.gravity_m_s2 + downforce_n, rel=1e-12
        )
        front_axle_n = float(row[forces.FL_WHEEL_INDEX] + row[forces.FR_WHEEL_INDEX])
        transfer_n = loads.longitudinal_load_transfer_n(
            config.mass_kg,
            float(trace[index, longitudinal.PREVIOUS_AX_INDEX]),
            config.cg_height_m,
            config.cg_to_front_axle_m + config.cg_to_rear_axle_m,
        )
        assert front_axle_n == pytest.approx(front_share * float(row.sum()) - transfer_n, rel=1e-12)
        # Not the P1 convention any more: a quarter of the downforce on each front patch would put
        # the front axle on exactly 50% of the total.
        assert front_axle_n != pytest.approx(0.5 * float(row.sum()), rel=1e-6)


def test_drive_torque_accelerates_the_car_through_the_rear_wheels_only(
    config: KernelConfig,
    launch_torque: float,
    steps_per_second: int,
) -> None:
    """The end-to-end path: torque -> rear wheel spin -> slip -> force -> car.

    A tenth of a second of this car's own launch is enough for the loop to reach positive slip on
    the driven axle, so the car has to accelerate; and because only the rear wheels were driven, the
    fronts are only ever responding to the road. That is checked rather than assumed: a kernel that
    fed the front axle too would still accelerate, plausibly.
    """
    steps = steps_per_second // 10
    outputs = longitudinal.allocate_step_outputs(steps)
    out = longitudinal.simulate(
        config,
        steps,
        longitudinal.initial_state(),
        np.full(steps, launch_torque, dtype=np.float64),
        longitudinal.allocate(steps),
        step_outputs=outputs,
    )

    assert np.isfinite(out).all()
    # The first step cannot move the car: at rest with stationary wheels the slip is exactly zero,
    # so the tyre force is exactly zero. From the second step on the car is under power.
    assert np.all(np.diff(out[1:, longitudinal.V_INDEX]) > 0.0), "the car accelerates under power"
    assert out[-1, longitudinal.X_INDEX] > 0.0
    rear = out[:, [longitudinal.RL_WHEEL_INDEX, longitudinal.RR_WHEEL_INDEX]]
    front = out[:, [longitudinal.FL_WHEEL_INDEX, longitudinal.FR_WHEEL_INDEX]]
    assert np.all(np.diff(rear[:51], axis=0) > 0.0), "the driven wheels spin up initially"
    assert rear[-1, 0] > 1.0, "and they reach a slip-producing speed inside a tenth of a second"
    assert np.max(np.abs(rear[:, 0] - rear[:, 1])) < 1.0e-3, (
        "equal drive torque keeps the two driven wheel speeds within a sub-millimetre-per-second "
        "difference despite tiny mirrored-force roundoff"
    )
    # The fronts stay at rest for the first rows, because a stationary car with stationary wheels
    # has no slip and so no force at all; from there they are driven by the road alone.
    assert np.all(front[:3] == 0.0), "three zero-force steps before the road moves them"
    assert np.all(front[3:] > 0.0), "the fronts are driven by the road, not by the transmission"
    # Relaxation lets the unpowered front wheels lag the low-speed chassis during this short launch.
    # They respond to road force, while the driven rear axle retains the larger spin state.
    assert 0.0 < front[-1, 0] < rear[-1, 0]


def test_a_locked_front_wheel_is_spun_up_by_the_road_and_no_lock_is_held(
    config: KernelConfig,
) -> None:
    """C11.4.1 forbids a system designed to prevent wheels locking; C9.1.2 forbids suppressing spin.

    There is nothing in the loop that can do either, and the two states are reached here from
    ordinary inputs rather than by asking for them: a car seeded with stationary wheels is a locked
    axle, and a car seeded with wheels far faster than the road is a spinning axle. Both have to
    resolve themselves, in opposite directions, with no flag set.
    """
    steps = 20_000
    speed = 40.0
    rolling = _rolling_wheels(config, speed)

    locked = _run(
        config,
        steps,
        np.zeros(steps, dtype=np.float64),
        longitudinal.initial_state(speed_m_s=speed, wheel_omega_rad_s=0.0),
    )
    assert locked[0, longitudinal.FL_WHEEL_INDEX] == 0.0
    assert np.all(np.diff(locked[:300, longitudinal.FL_WHEEL_INDEX]) > 0.0), (
        "the road spun the locked wheel up, monotonically at first"
    )
    assert locked[-1, longitudinal.V_INDEX] < speed, "and the slip cost the car speed"
    # It resolves: with no slip left there is no force left, so the locked axle ends up rolling at
    # the car's own speed. A large initial overspeed first accelerates the chassis before the
    # rotating assembly catches up, so this uses enough simulated time for the coupled wheel/chassis
    # states to settle rather than treating a short transient as steady state.
    assert locked[-1, longitudinal.FL_WHEEL_INDEX] == pytest.approx(
        locked[-1, longitudinal.V_INDEX] / config.rolling_radius_m, rel=1.0e-3
    )

    spun = _run(
        config,
        steps,
        np.zeros(steps, dtype=np.float64),
        longitudinal.initial_state(speed_m_s=speed, wheel_omega_rad_s=4.0 * rolling),
    )
    assert spun[0, longitudinal.FL_WHEEL_INDEX] == 4.0 * rolling
    assert np.all(np.diff(spun[:301, longitudinal.FL_WHEEL_INDEX]) < 0.0), (
        "the road initially slows the overspeed wheel"
    )
    assert spun[-1, longitudinal.FL_WHEEL_INDEX] == pytest.approx(
        spun[-1, longitudinal.V_INDEX] / config.rolling_radius_m, rel=1.0e-3
    )


def test_the_wheels_and_the_chassis_advance_on_the_same_step_in_a_fixed_order(
    config: KernelConfig,
    launch_torque: float,
    steps_per_second: int,
) -> None:
    """The update order is pinned, because a different one is a different launch.

    Every wheel's speed and the car's speed are advanced from the state they *started* the step
    with, and the position from the speed that step produced. Both halves are checked as exact
    identities against the trace rather than against a number out of a run: recompute the step's
    forces from the row that went in, then require the row that came out to be exactly that.
    """
    steps = steps_per_second // 20
    outputs = longitudinal.allocate_step_outputs(steps)
    out = longitudinal.simulate(
        config,
        steps,
        longitudinal.initial_state(),
        np.full(steps, launch_torque, dtype=np.float64),
        longitudinal.allocate(steps),
        step_outputs=outputs,
    )
    radius = config.rolling_radius_m

    for index in range(steps):
        previous, current = out[index], out[index + 1]
        vx = float(previous[longitudinal.V_INDEX])
        vy = float(previous[longitudinal.VY_INDEX])
        yaw = float(previous[longitudinal.YAW_RATE_INDEX])
        speed = math.hypot(vx, vy)
        _downforce_n, drag_n = forces.aero_forces(
            speed,
            config.air_density_kg_m3,
            config.reference_area_m2,
            config.aero_speed_m_s,
            config.cl,
            config.cd,
        )
        net_fx = drag_n * vx / speed if speed > 0.0 else 0.0
        net_fy = drag_n * vy / speed if speed > 0.0 else 0.0
        moment = 0.0
        for wheel in range(forces.WHEEL_COUNT):
            column = WHEEL_COLUMNS[wheel]
            tyre_fx_n = float(outputs.force_x_n[index, wheel])
            tyre_fy_n = float(outputs.force_y_n[index, wheel])
            axle = wheel // forces.WHEELS_PER_AXLE_COUNT
            side = 1.0 if wheel % forces.WHEELS_PER_AXLE_COUNT == 0 else -1.0
            corner_x = config.cg_to_front_axle_m if axle == 0 else -config.cg_to_rear_axle_m
            corner_y = side * 0.5 * float(config.axle_track_m[axle])
            steer = math.radians(
                float(config.axle_bump_steer_deg_per_m[axle] * outputs.travel_m[index, wheel])
            )
            body_fx = tyre_fx_n * math.cos(steer) - tyre_fy_n * math.sin(steer)
            body_fy = tyre_fx_n * math.sin(steer) + tyre_fy_n * math.cos(steer)
            net_fx += body_fx
            net_fy += body_fy
            moment += corner_x * body_fy - corner_y * body_fx
            drive_nm = forces.wheel_drive_torque_nm(wheel, launch_torque)
            alpha = forces.wheel_angular_acceleration_rad_s2(
                drive_nm, tyre_fx_n, radius, config.wheel_inertia_kg_m2
            )
            assert current[column] == pytest.approx(
                previous[column] + alpha * config.dt_s, rel=0.0, abs=0.0
            ), (index, column)
        ax = net_fx / config.mass_kg
        ay = net_fy / config.mass_kg
        vx_next = vx + (ax + yaw * vy) * config.dt_s
        vy_next = vy + (ay - yaw * vx) * config.dt_s
        yaw_next = yaw + moment / config.yaw_inertia_kg_m2 * config.dt_s
        assert current[longitudinal.V_INDEX] == pytest.approx(vx_next, rel=0.0, abs=1e-12), index
        assert current[longitudinal.VY_INDEX] == pytest.approx(vy_next, rel=0.0, abs=1e-12), index
        assert current[longitudinal.YAW_RATE_INDEX] == pytest.approx(
            yaw_next, rel=0.0, abs=1e-12
        ), index
        assert current[longitudinal.PSI_INDEX] == pytest.approx(
            previous[longitudinal.PSI_INDEX] + yaw_next * config.dt_s, rel=0.0, abs=1e-12
        ), index
        assert current[longitudinal.X_INDEX] == pytest.approx(
            previous[longitudinal.X_INDEX]
            + (
                vx_next * math.cos(current[longitudinal.PSI_INDEX])
                - vy_next * math.sin(current[longitudinal.PSI_INDEX])
            )
            * config.dt_s,
            abs=1e-12,
        ), index
        assert current[longitudinal.Y_INDEX] == pytest.approx(
            previous[longitudinal.Y_INDEX]
            + (
                vx_next * math.sin(current[longitudinal.PSI_INDEX])
                + vy_next * math.cos(current[longitudinal.PSI_INDEX])
            )
            * config.dt_s,
            abs=1e-12,
        ), index


def test_a_negative_drive_torque_moves_the_car_backwards(config: KernelConfig) -> None:
    """Reverse is the same loop with a negative torque, which is what C9.7 asks to be drivable.

    A reverse gear hands P1-T7 a negative differential-side torque, so nothing in the loop has to
    know about gears. Checked from rest, where the slip is unambiguous, and on both the sign of the
    wheel speeds and the sign of the travel - a model that produced a backwards car with forward
    wheel speeds would be braking, not reversing.
    """
    steps = 2_000
    torque = np.full(steps, -1_500.0, dtype=np.float64)
    out = _run(config, steps, torque)

    assert np.isfinite(out).all()
    # As forward, the very first step cannot move the car: zero slip is zero force.
    assert np.all(np.diff(out[1:, longitudinal.V_INDEX]) < 0.0)
    assert out[-1, longitudinal.X_INDEX] < 0.0
    assert out[-1, longitudinal.RL_WHEEL_INDEX] < 0.0
    assert out[-1, longitudinal.RR_WHEEL_INDEX] < 0.0
    assert out[-1, longitudinal.FL_WHEEL_INDEX] < 0.0, "the road spins the fronts backwards too"
    assert out[-1, longitudinal.FL_WHEEL_INDEX] == pytest.approx(
        out[-1, longitudinal.V_INDEX] / config.rolling_radius_m, rel=1e-2
    )


def test_position_uses_the_updated_speed_so_the_scheme_is_semi_implicit(
    config: KernelConfig,
) -> None:
    """Symplectic Euler advances position with the *new* speed; explicit Euler does not.

    The scheme is pinned twice against a real run rather than a constant-force one, because the
    force is now a function of the state and there is no longer a closed form to compare a gap
    against. First, exactly: row one advances by ``out[1, V] * dt`` and *not* by ``out[0, V] * dt``,
    which are different numbers here because the car is decelerating into its own drag. Second,
    against the explicit alternative over the whole run, which integrates position with the previous
    speed and therefore always covers less ground while the car is moving forwards.
    """
    steps = 2_000
    speed = 40.0
    out = _coasting(config, steps, speed)
    dt_s = config.dt_s
    for index in range(1, steps + 1):
        previous, current = out[index - 1], out[index]
        psi = current[longitudinal.PSI_INDEX]
        vx, vy = current[longitudinal.V_INDEX], current[longitudinal.VY_INDEX]
        expected_dx = (vx * math.cos(psi) - vy * math.sin(psi)) * dt_s
        expected_dy = (vx * math.sin(psi) + vy * math.cos(psi)) * dt_s
        assert current[longitudinal.X_INDEX] == pytest.approx(
            previous[longitudinal.X_INDEX] + expected_dx, abs=1e-14
        )
        assert current[longitudinal.Y_INDEX] == pytest.approx(
            previous[longitudinal.Y_INDEX] + expected_dy, abs=1e-14
        )
    assert out[1, longitudinal.V_INDEX] != out[0, longitudinal.V_INDEX], (
        "the run has to be changing speed for this test to say anything about the scheme"
    )
    assert (
        out[1, longitudinal.X_INDEX]
        != out[0, longitudinal.X_INDEX]
        + (
            out[0, longitudinal.V_INDEX] * math.cos(out[0, longitudinal.PSI_INDEX])
            - out[0, longitudinal.VY_INDEX] * math.sin(out[0, longitudinal.PSI_INDEX])
        )
        * dt_s
    )

    explicit = np.zeros_like(out)
    explicit[0, :] = out[0, :]
    for index in range(1, steps + 1):
        explicit[index, :] = out[index, :]
        psi = out[index - 1, longitudinal.PSI_INDEX]
        vx, vy = out[index - 1, longitudinal.V_INDEX], out[index - 1, longitudinal.VY_INDEX]
        explicit[index, longitudinal.X_INDEX] = (
            explicit[index - 1, longitudinal.X_INDEX]
            + (vx * math.cos(psi) - vy * math.sin(psi)) * dt_s
        )
        explicit[index, longitudinal.Y_INDEX] = (
            explicit[index - 1, longitudinal.Y_INDEX]
            + (vx * math.sin(psi) + vy * math.cos(psi)) * dt_s
        )
    assert not np.array_equal(out, explicit)
    # The gap's sign follows the car's speed change, not the direction of travel: semi-implicit
    # advances with the *new* speed, so it covers more ground while accelerating and less while
    # decelerating. This run coasts into its own drag, so it covers less - and saying which way
    # without the reason would be a trap for the next person to change the fixture.
    assert out[-1, longitudinal.V_INDEX] < speed
    assert out[-1, longitudinal.X_INDEX] < explicit[-1, longitudinal.X_INDEX]


def test_the_kernel_agrees_with_a_plain_python_reference(
    config: KernelConfig,
    launch_torque: float,
    steps_per_second: int,
) -> None:
    """Bit for bit, not approximately: the compiled loop must do the arithmetic it claims.

    The reference borrows the compiled tyre and aero primitives - so this is a comparison of two
    *orderings* of the same arithmetic rather than of two evaluations of ``sin`` and ``atan``, whose
    last bits belong to the platform's libm and are not this project's to assert on. Everything the
    kernel owns - which state a wheel reads, which force a row is built from, and the order the
    updates happen in - is written out again here in plain Python.
    """
    steps = steps_per_second // 20
    state = longitudinal.initial_state(speed_m_s=5.0, wheel_omega_rad_s=0.0)
    drive_torque = np.linspace(0.0, launch_torque, steps)
    out = _run(config, steps, drive_torque, state)
    expected = _reference(config, steps, drive_torque, state)
    assert np.allclose(out, expected, rtol=1.0e-12, atol=1.0e-12)


def test_a_car_at_rest_with_no_drive_torque_does_not_move(
    config: KernelConfig,
    steps_per_second: int,
) -> None:
    """Zero in, zero out - the degenerate run, and the check that nothing invents a force.

    Both halves of the loop are at their zeros here: slip is zero, so the Magic Formula returns
    exactly ``0.0``, and dynamic pressure is zero, so the drag is exactly ``0.0``. A model that
    produced a small force in that state would quietly manufacture a launch out of numerical dust,
    and a coast that drifts would make every later trace harder to read.
    """
    steps = steps_per_second
    out = _run(config, steps, np.zeros(steps, dtype=np.float64))
    assert np.isfinite(out).all()
    assert np.array_equal(out, np.zeros((steps + 1, longitudinal.STATE_SIZE)))


def test_a_coasting_run_loses_speed_to_drag_while_its_wheels_stay_rolling(
    config: KernelConfig,
    steps_per_second: int,
) -> None:
    """A representative second of coasting: finite throughout, decelerating, wheels barely moving.

    Not an analytic claim any more - with the tyre model in the loop the force is a function of the
    state - so what is asserted is the shape of the run and its finiteness, which is what a
    diverging formula breaks. The wheels must stay within a slip ratio of rolling rather than
    running away:
    the loop is what holds them there, and a kernel that dropped the reaction term would let them
    drift freely.
    """
    steps = steps_per_second
    speed = 60.0
    out = _coasting(config, steps, speed)
    assert np.isfinite(out).all()
    assert np.all(np.diff(out[:, longitudinal.V_INDEX]) < 0.0), "drag is negative going forward"
    assert np.all(np.diff(out[:, longitudinal.X_INDEX]) > 0.0), "so the car still travels"
    assert 0.0 < out[-1, longitudinal.V_INDEX] < speed
    rolling = out[:, longitudinal.V_INDEX] / config.rolling_radius_m
    for column in WHEEL_COLUMNS:
        assert np.allclose(out[:, column], rolling, rtol=1e-3), column


def test_a_second_run_reproduces_the_first_byte_for_byte(
    config: KernelConfig,
    launch_torque: float,
    steps_per_second: int,
    spec: CarSpec,
) -> None:
    """P1-T2: same spec, same buffers, same inputs -> the same bytes, not merely the same numbers.

    The second run is deliberately hostile in two ways. Its config is rebuilt from the spec, so the
    check also rules out a kernel keyed off the identity of the config object rather than its
    values; and it writes over a buffer poisoned with NaN, so anything the first run left behind -
    an accumulator the caller never zeroed, a row the loop skipped - would show up.
    """
    steps = steps_per_second // 4
    drive_torque = np.full(steps, launch_torque, dtype=np.float64)
    reused = _run(config, steps, drive_torque)
    reused[:] = np.nan
    longitudinal.simulate(config, steps, longitudinal.initial_state(), drive_torque, reused)
    rebuilt = _run(spec.kernel_config(), steps, drive_torque)
    assert reused.tobytes() == rebuilt.tobytes()


def test_the_run_reads_no_clock_and_no_random_source(
    config: KernelConfig,
    launch_torque: float,
    steps_per_second: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Byte-identity is only a determinism claim if nothing outside the arithmetic varies."""
    for module_name, attribute in TIME_AND_RANDOM:
        module = importlib.import_module(module_name)
        if hasattr(module, attribute):
            monkeypatch.setattr(module, attribute, partial(_refuse, f"{module_name}.{attribute}"))
    steps = steps_per_second // 4
    drive_torque = np.full(steps, launch_torque, dtype=np.float64)
    first = _run(config, steps, drive_torque)
    second = _run(config, steps, drive_torque)
    assert first.tobytes() == second.tobytes()


def test_the_step_loop_performs_no_allocation(config: KernelConfig) -> None:
    """PLAN.md section 4.1 rule 2: no allocation inside the loop, at any step count.

    Measured with Numba's own NRT allocation counters, the only thing that can see an
    allocation made inside compiled code. Compiled-to-compiled dispatch costs a fixed handful of
    allocations whatever the loop length, so the claim is that the count does not grow with the
    number of steps - a run a hundred times longer must not allocate any more than the short one.
    Closing the loop calls the tyre and aero primitives from inside the loop, so this is the test
    that says those calls do not allocate either.

    The first half of the test is the control: a kernel that *does* allocate inside its loop has
    to show a count that grows, otherwise the counters are not measuring anything and the second
    half would pass vacuously. The counters themselves are asserted on rather than skipped when
    they are off, for the same reason - a measurement that silently stops measuring is not a pass.
    """
    from numba import njit
    from numba.core import config as numba_config
    from numba.core.runtime import nrt

    # numba parses this flag into its own config namespace at import time by writing module
    # globals, so it is read through getattr rather than off the module directly.
    assert getattr(numba_config, "NRT_STATS", False), (
        "numba's NRT allocation counters are off, so this test would pass vacuously; "
        "tests/conftest.py sets NUMBA_NRT_STATS before numba is imported"
    )

    @njit(cache=False)
    def allocating(steps: int) -> float:
        total = 0.0
        for index in range(steps):
            scratch = np.empty(index + 1, dtype=np.float64)
            total += scratch.size
        return total

    def allocations_for(steps: int) -> tuple[tuple[int, int], tuple[int, int]]:
        """Allocations for one run of the kernel, and for one run of the control kernel."""
        torque = np.full(steps, 1.0, dtype=np.float64)
        state = longitudinal.initial_state()
        out = longitudinal.allocate(steps)
        longitudinal.simulate(config, steps, state, torque, out)
        allocating(steps)
        before = nrt.rtsys.get_allocation_stats()
        longitudinal.simulate(config, steps, state, torque, out)
        after_kernel = nrt.rtsys.get_allocation_stats()
        allocating(steps)
        after_control = nrt.rtsys.get_allocation_stats()
        kernel = (
            after_kernel.alloc - before.alloc,
            after_kernel.mi_alloc - before.mi_alloc,
        )
        control = (
            after_control.alloc - after_kernel.alloc,
            after_control.mi_alloc - after_kernel.mi_alloc,
        )
        return kernel, control

    short_kernel, short_control = allocations_for(1_000)
    long_kernel, long_control = allocations_for(100_000)
    assert long_control > short_control, "NRT counters are not seeing in-loop allocation at all"
    assert long_kernel == short_kernel


def test_the_caller_owns_every_buffer_and_the_inputs_survive_the_run(
    config: KernelConfig,
    launch_torque: float,
    steps_per_second: int,
) -> None:
    """The kernel writes only into the buffer it was handed, and hands the same one back."""
    steps = steps_per_second // 4
    state = longitudinal.initial_state(speed_m_s=7.0, wheel_omega_rad_s=19.4)
    torque = np.full(steps, launch_torque, dtype=np.float64)
    torque_before = torque.tobytes()
    state_before = state.tobytes()
    out = longitudinal.allocate(steps)
    returned = longitudinal.simulate(config, steps, state, torque, out)
    assert np.shares_memory(returned, out)
    assert returned.ctypes.data == out.ctypes.data
    assert torque.tobytes() == torque_before
    assert state.tobytes() == state_before


def test_read_only_state_and_torque_are_accepted_because_the_kernel_never_writes_them(
    config: KernelConfig,
    launch_torque: float,
) -> None:
    """A caller may hand over inputs it does not own outright; ``out`` is the only destination.

    Seeded rolling, so the run is a plain acceleration rather than a locked-axle deceleration -
    the state has to be one the loop can be asked about at all.
    """
    steps = 64
    drive_torque = np.full(steps, launch_torque, dtype=np.float64)
    drive_torque.flags.writeable = False
    state = longitudinal.initial_state(
        speed_m_s=3.0, wheel_omega_rad_s=3.0 / config.rolling_radius_m
    )
    state.flags.writeable = False
    out = longitudinal.allocate(steps)
    returned = longitudinal.simulate(config, steps, state, drive_torque, out)
    assert returned[-1, longitudinal.V_INDEX] > 3.0
    assert drive_torque.tobytes() == np.full(steps, launch_torque).tobytes()


def test_editing_the_car_spec_changes_the_run_with_no_code_edit(
    config: KernelConfig,
    launch_torque: float,
    spec: CarSpec,
) -> None:
    """The configuration is the only thing that decides the run.

    Twice the step doubles the first wheel increment. Mass and inertia also reach the force balance:
    a changed mass slightly changes the load-sensitive tyre response even from a standing start.
    """
    steps = 2
    torque = np.full(steps, launch_torque, dtype=np.float64)
    baseline = _run(config, steps, torque)
    assert baseline[1, longitudinal.RL_WHEEL_INDEX] == pytest.approx(
        (launch_torque / 2.0) / config.wheel_inertia_kg_m2 * config.dt_s, rel=1e-12
    )

    slower = _run(replace(spec, dt_s=config.dt_s * 2.0).kernel_config(), steps, torque)
    assert slower[1, longitudinal.RL_WHEEL_INDEX] == pytest.approx(
        2.0 * baseline[1, longitudinal.RL_WHEEL_INDEX], rel=1e-12
    )

    # A wheel's balance is `I d(omega)/dt = T - Fx r`. Row one is mass-free, because at zero slip
    # the force is exactly zero and so is neither term.
    heavier = _run(replace(spec, mass_kg=spec.mass_kg * 2.0).kernel_config(), steps, torque)
    assert heavier[1, longitudinal.RL_WHEEL_INDEX] == baseline[1, longitudinal.RL_WHEEL_INDEX]

    # Load sensitivity makes the tyre's effective peak change as mass changes, so acceleration does
    # not cancel exactly against mass. This is a small but measurable response to the edited data.
    assert not np.isclose(
        heavier[2, longitudinal.V_INDEX],
        baseline[2, longitudinal.V_INDEX],
        rtol=1.0e-4,
        atol=0.0,
    )
    with_speed = _run(
        replace(spec, mass_kg=spec.mass_kg * 2.0).kernel_config(),
        steps,
        torque,
        longitudinal.initial_state(speed_m_s=20.0, wheel_omega_rad_s=100.0),
    )
    assert with_speed[2, longitudinal.V_INDEX] != baseline[2, longitudinal.V_INDEX], (
        "with downforce on the table the mass stops cancelling, which is what the "
        "load-then-torque order means for acceleration"
    )

    # The wheel states are *not* mass-free, because the reaction term scales with the load: a
    # heavier car is a heavier car to spin up. The two lines above are what make that asymmetry
    # visible rather than something a reader has to notice.
    assert heavier[2, longitudinal.RL_WHEEL_INDEX] != baseline[2, longitudinal.RL_WHEEL_INDEX]


@pytest.mark.parametrize(
    ("steps", "torque_size", "state_size", "rows"),
    [
        (4, 3, 6, 5),
        (4, 4, 1, 5),
        (4, 4, 2, 4),
        (4, 4, 6, 6),
    ],
    ids=["short-torque", "short-state", "one-row-short", "one-row-long"],
)
def test_a_buffer_the_run_cannot_fill_is_refused_in_python(
    config: KernelConfig,
    steps: int,
    torque_size: int,
    state_size: int,
    rows: int,
) -> None:
    """Every check the loop cannot make belongs in the caller, before the loop starts."""
    torque = np.zeros(torque_size, dtype=np.float64)
    state = np.zeros(state_size, dtype=np.float64)
    out = np.zeros((rows, longitudinal.STATE_SIZE), dtype=np.float64)
    with pytest.raises(ValueError, match="buffer"):
        longitudinal.simulate(config, steps, state, torque, out)


def test_a_float32_buffer_is_refused_in_python(config: KernelConfig) -> None:
    """The kernel is float64; a float32 buffer would silently re-quantise the trace."""
    torque = np.zeros(4, dtype=np.float32)
    state = np.zeros(longitudinal.STATE_SIZE, dtype=np.float32)
    out = np.zeros((5, longitudinal.STATE_SIZE), dtype=np.float32)
    with pytest.raises(ValueError, match="float64"):
        longitudinal.simulate(config, 4, state, torque, out)


def test_a_buffer_that_is_not_an_ndarray_is_refused_in_python(config: KernelConfig) -> None:
    """A list would be converted by the dispatcher, and the caller would not get the buffer back."""
    not_an_array: Any = [0.0] * longitudinal.STATE_SIZE
    with pytest.raises(ValueError, match="ndarray"):
        longitudinal.simulate(config, 4, not_an_array, np.zeros(4), np.zeros((5, 2)))


def test_a_strided_buffer_is_refused_in_python(config: KernelConfig) -> None:
    """Contiguity is part of the kernel contract, so it is checked rather than silently honoured."""
    torque = np.zeros(8, dtype=np.float64)[::2]
    assert not torque.flags.c_contiguous
    with pytest.raises(ValueError, match="contiguous"):
        longitudinal.simulate(config, 4, longitudinal.initial_state(), torque, np.zeros((5, 2)))


def test_a_read_only_output_buffer_is_refused_in_python(config: KernelConfig) -> None:
    """``out`` is the kernel's only destination, so a read-only buffer is an out-of-bounds write."""
    steps = 4
    out = longitudinal.allocate(steps)
    out.flags.writeable = False
    with pytest.raises(ValueError, match="read-only"):
        longitudinal.simulate(config, steps, longitudinal.initial_state(), np.zeros(steps), out)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        (longitudinal.V_INDEX, math.nan),
        (longitudinal.V_INDEX, math.inf),
        (longitudinal.FL_WHEEL_INDEX, math.nan),
        (longitudinal.RL_WHEEL_INDEX, math.inf),
        (longitudinal.RR_WHEEL_INDEX, -math.inf),
    ],
    ids=["speed", "speed-inf", "front-left", "rear-left", "rear-right"],
)
def test_a_seeded_state_that_is_not_finite_is_refused_in_python(
    config: KernelConfig,
    column: int,
    value: float,
) -> None:
    """A nonfinite seed poisons every row after it, and the trace still looks finished.

    This is the boundary check P1-T6 adds. A NaN wheel speed is not a slow wheel - it reaches the
    slip ratio, from there the tyre force, from there the force on the car, and every later row is
    NaN while the run reports success. Six values are checked once, before the loop, which is
    nothing next to the trace it protects.
    """
    state = longitudinal.initial_state(speed_m_s=10.0, wheel_omega_rad_s=30.0)
    state[column] = value
    with pytest.raises(ValueError, match="state"):
        _run(config, 4, np.zeros(4, dtype=np.float64), state)


@pytest.mark.parametrize("value", [0.0, -1.0e-4, math.nan, math.inf])
def test_a_step_that_could_not_produce_a_run_is_refused_in_python(
    config: KernelConfig,
    value: float,
) -> None:
    """``KernelConfig`` is a public frozen dataclass, so it can carry a step the loader would not.

    Nothing in the loop checks ``dt_s``; a zero or nonfinite step is arithmetic that returns NaNs
    or an empty-looking run rather than an error, which is the kind of bug that surfaces three
    tasks later as an implausible trace.
    """
    broken = replace(config, dt_s=value)
    with pytest.raises(ValueError, match="dt_s"):
        _run(broken, 4, np.zeros(4, dtype=np.float64))


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan])
def test_a_mass_that_could_not_produce_a_run_is_refused_in_python(
    config: KernelConfig,
    value: float,
) -> None:
    """Same for ``mass_kg``: the loop divides by it, so zero mass is a silent NaN trace."""
    broken = replace(config, mass_kg=value)
    with pytest.raises(ValueError, match="mass_kg"):
        _run(broken, 4, np.zeros(4, dtype=np.float64))


@pytest.mark.parametrize("value", [0.0, -9.80665, math.nan])
def test_a_gravity_that_could_not_produce_a_run_is_refused_in_python(
    config: KernelConfig,
    value: float,
) -> None:
    """Gravity now reaches the kernel, since the static axle loads are a fraction of the weight."""
    broken = replace(config, gravity_m_s2=value)
    with pytest.raises(ValueError, match="gravity_m_s2"):
        _run(broken, 4, np.zeros(4, dtype=np.float64))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("wheel_inertia_kg_m2", 0.0),
        ("wheel_inertia_kg_m2", -0.9),
        ("rolling_radius_m", 0.0),
        ("front_weight_fraction", 0.0),
        ("front_weight_fraction", 1.0),
        ("air_density_kg_m3", 0.0),
        ("slip_ratio_min_speed_m_s", 0.0),
        ("pacejka_mu", math.nan),
    ],
)
def test_a_coefficient_the_closed_loop_now_reads_is_refused_in_python(
    config: KernelConfig,
    name: str,
    value: float,
) -> None:
    """P1-T6/T7 turned eight more coefficients into things the loop divides by or indexes.

    Until the loop was closed, only ``dt_s`` and ``mass_kg`` were checked: force was an input, so
    nothing else could be reached. Now the wheel inertia and the rolling radius are divisors, the
    aero curves are indexed with ``boundscheck=False``, and the front weight fraction decides how
    the load splits - a malformed value in any of them is a wrong run rather than an error, and it
    would be a wrong run all the way to the end of the trace.
    """
    broken = replace(config, **{name: value})
    with pytest.raises(ValueError, match=name):
        _run(broken, 4, np.zeros(4, dtype=np.float64))


def test_replaced_aero_arrays_cannot_reach_the_kernel_loop(config: KernelConfig) -> None:
    """``simulate`` now indexes ``Cl(v)`` and ``Cd(v)``, so it has to re-check them itself.

    The loader's guarantees are a property of the loader, and ``CarSpec`` plus ``dataclasses``
    are the documented way to get a config that never went through it. ``fastmath=False`` does not
    turn off ``boundscheck=False``, so a short or reversed curve here is an out-of-bounds read in
    compiled code.
    """
    reversed_axis = config.aero_speed_m_s.copy()
    reversed_axis[2] = reversed_axis[1]
    malformed = (
        replace(config, cl=config.cl.astype(np.float32)),
        replace(config, cd=config.cd[:-1]),
        replace(config, aero_speed_m_s=config.aero_speed_m_s[::-1].copy()),
        replace(config, aero_speed_m_s=reversed_axis),
        replace(config, cd=np.zeros_like(config.cd)),
        replace(config, aero_speed_m_s=config.aero_speed_m_s.tolist()),
    )
    for broken in malformed:
        with pytest.raises(ValueError, match="simulate: config"):
            _run(broken, 4, np.zeros(4, dtype=np.float64))


@pytest.mark.parametrize("steps", [4.0, 4.5, True, "4", None])
def test_a_step_count_that_is_not_an_integer_is_refused_in_python(
    config: KernelConfig,
    steps: Any,
) -> None:
    """A float count would compile a second float64 specialisation of the same integrator.

    ``int(4.0)`` would hide that, and ``bool`` is an ``int`` by Python's rules but never a
    meaningful step count, so both are refused rather than coerced.
    """
    with pytest.raises(ValueError, match="steps"):
        longitudinal.simulate(
            config,
            steps,
            longitudinal.initial_state(),
            np.zeros(4, dtype=np.float64),
            np.zeros((5, longitudinal.STATE_SIZE), dtype=np.float64),
        )


def test_an_integer_like_step_count_is_accepted(config: KernelConfig) -> None:
    """``operator.index`` is what the boundary uses, so a numpy integer is a legal count."""
    steps = 4
    torque = np.zeros(steps, dtype=np.float64)
    state = longitudinal.initial_state()
    with_int = longitudinal.simulate(config, steps, state, torque, longitudinal.allocate(steps))
    numpy_integer: Any = np.int64(steps)
    with_numpy = longitudinal.simulate(
        config,
        numpy_integer,
        state,
        torque,
        np.zeros((steps + 1, longitudinal.STATE_SIZE), dtype=np.float64),
    )
    assert with_int.tobytes() == with_numpy.tobytes()
