"""P1-T1: the fixed-step straight-line kernel, tested against the claims it makes.

``PHASES.md`` P1-T1 asks for a flat ``@njit(cache=True, fastmath=False)`` kernel over
preallocated float64 arrays at ``dt = 100 µs``, and P1-T2 asks for the determinism harness
that proves it. Three claims are checked here, each in the form that would actually fail:

* **Fixed step, semi-implicit.** Row ``n`` of a trace is the state after exactly ``n`` steps
  of the configured ``dt_s``, and position advances with the *updated* speed, so the scheme is
  semi-implicit rather than explicit. The scheme is pinned against the explicit alternative,
  not against a number copied out of a run.
* **Determinism.** Two runs with the same configuration and the same buffers produce
  byte-identical traces, and the run reads no clock and no random source. Byte comparison, not
  ``allclose``: a tolerance would hide the last-bit differences a wall-clock read introduces.
* **No work the caller did not ask for.** The step loop allocates nothing measurable in
  Numba's runtime, the caller owns every buffer, and the inputs are left untouched.

Every physical number here comes from the loaded ``car_spec.yaml``, including the drive force
used for the representative run - it is the car's own weight, so the run accelerates at one
gravity. There is no tuned constant in this file.

The ``kernel`` marker keeps these apart from the ``toolchain`` probe: that one proves Numba
compiles here, this one proves the car's integrator is right.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pytest

from f1telemetry.contracts.car_spec import CarSpec
from f1telemetry.kernels import longitudinal

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import KernelConfig

pytestmark = pytest.mark.kernel

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


@pytest.fixture(scope="module")
def config(spec: CarSpec) -> KernelConfig:
    """The validated, flat configuration the kernel is allowed to see."""
    return spec.kernel_config()


@pytest.fixture(scope="module")
def steps_per_second(config: KernelConfig) -> int:
    """How many fixed steps one simulated second takes, read from the configured step."""
    return round(1.0 / config.dt_s)


@pytest.fixture(scope="module")
def drive_force(config: KernelConfig, steps_per_second: int) -> np.ndarray:
    """A constant longitudinal force equal to the car's weight, so the run accelerates at 1 g.

    Taken from the configuration rather than written down, which keeps this file free of tuned
    numbers and makes the analytic expectation exact.
    """
    return longitudinal.constant_longitudinal_force(
        steps=steps_per_second, force_n=config.mass_kg * config.gravity_m_s2
    )


def _run(
    config: KernelConfig,
    steps: int,
    force_n: np.ndarray,
    state: np.ndarray | None = None,
) -> np.ndarray:
    """Integrate into a fresh caller-owned buffer and hand it back."""
    seed = longitudinal.initial_state() if state is None else state
    return longitudinal.simulate(config, steps, seed, force_n, longitudinal.allocate(steps))


def _reference(
    steps: int,
    dt_s: float,
    mass_kg: float,
    force_n: float,
    state: np.ndarray,
) -> np.ndarray:
    """The same integrator in plain Python, written out one step at a time.

    Deliberately not a closed form: a compiled kernel compared against arithmetic done the same
    way catches a wrong update order, which a closed form would only show as a small difference.
    """
    out = np.zeros((steps + 1, longitudinal.STATE_SIZE), dtype=np.float64)
    out[0, :] = state
    acceleration = force_n / mass_kg
    for index in range(1, steps + 1):
        out[index, longitudinal.V_INDEX] = (
            out[index - 1, longitudinal.V_INDEX] + acceleration * dt_s
        )
        out[index, longitudinal.X_INDEX] = (
            out[index - 1, longitudinal.X_INDEX] + out[index, longitudinal.V_INDEX] * dt_s
        )
    return out


def _refuse(what: str) -> None:
    raise AssertionError(f"the simulation read {what}")


def test_the_kernel_compiles_to_machine_code(config: KernelConfig) -> None:
    """A dispatcher entry is not a compiled kernel: a signature is the evidence."""
    state = longitudinal.initial_state()
    force = np.zeros(4, dtype=np.float64)
    out = longitudinal.allocate(4)
    longitudinal.simulate(config, 4, state, force, out)
    scratch = np.zeros(longitudinal.STATE_SIZE, dtype=np.float64)
    longitudinal.step(state, 0.0, scratch, config.dt_s, config.mass_kg)
    assert len(longitudinal.step.signatures) >= 1
    assert len(longitudinal.integrate.signatures) >= 1


def test_the_kernel_options_match_plan_section_4_1() -> None:
    """PLAN.md section 4.1 rule 4: cache on, fastmath off, and both assertable."""
    options = dict(longitudinal.integrate.targetoptions)
    assert options["fastmath"] is False
    assert options["nopython"] is True
    assert options["nogil"] is True
    assert "cache" not in options
    assert longitudinal.TARGET_OPTIONS["cache"] is True
    assert longitudinal.TARGET_OPTIONS["fastmath"] is False


def test_the_fixed_step_is_ten_kilohertz_and_comes_from_the_car_spec(
    config: KernelConfig,
) -> None:
    """The step is data, not a constant in the kernel: 100 µs is what the spec says."""
    assert config.dt_s == pytest.approx(1.0e-4, rel=0.0, abs=0.0)
    assert 1.0 / config.dt_s == pytest.approx(10_000.0, rel=0.0, abs=0.0)
    assert round(1.0 / config.dt_s) == 10_000


def test_the_caller_seeds_the_state_and_owns_the_trace_buffer() -> None:
    """Row 0 of the trace is the caller's state; the caller hands over the whole buffer."""
    state = longitudinal.initial_state(distance_m=12.5, speed_m_s=30.0)
    assert state.shape == (longitudinal.STATE_SIZE,)
    assert state.dtype == np.float64
    assert state[longitudinal.X_INDEX] == 12.5
    assert state[longitudinal.V_INDEX] == 30.0
    out = longitudinal.allocate(5)
    assert out.shape == (6, longitudinal.STATE_SIZE)
    assert out.dtype == np.float64
    assert out.flags.writeable


def test_a_zero_step_run_writes_only_the_initial_state(config: KernelConfig) -> None:
    """A run of no steps is not a special case: row 0 is the caller's state, and nothing else."""
    state = longitudinal.initial_state(distance_m=3.0, speed_m_s=11.0)
    out = longitudinal.allocate(0)
    assert out.shape == (1, longitudinal.STATE_SIZE)
    returned = longitudinal.simulate(config, 0, state, np.zeros(0, dtype=np.float64), out)
    assert np.array_equal(returned, state.reshape(1, longitudinal.STATE_SIZE))


def test_a_coasting_run_holds_speed_and_advances_one_step_per_row(
    config: KernelConfig,
    steps_per_second: int,
) -> None:
    """With no force, every row is the previous row advanced by exactly ``dt_s``.

    This is the fixed-step claim in its simplest falsifiable form: one simulated second of rows
    covers exactly one second of travel, at the speed the caller seeded.
    """
    steps = steps_per_second
    speed = 30.0
    state = longitudinal.initial_state(speed_m_s=speed)
    out = _run(config, steps, np.zeros(steps, dtype=np.float64), state)
    assert out.shape == (steps + 1, longitudinal.STATE_SIZE)
    assert np.array_equal(out[0], state)
    assert np.array_equal(out[:, longitudinal.V_INDEX], np.full(steps + 1, speed))
    travelled = out[:, longitudinal.X_INDEX]
    assert np.allclose(np.diff(travelled), speed * config.dt_s, rtol=1.0e-12, atol=0.0)
    assert travelled[-1] == pytest.approx(speed * config.dt_s * steps, rel=1.0e-12)


def test_the_trace_advances_by_exactly_one_configured_step_per_row(
    config: KernelConfig,
    drive_force: np.ndarray,
) -> None:
    """Every row adds the same velocity increment, and the totals match the analytic run."""
    steps = drive_force.size
    force_n = config.mass_kg * config.gravity_m_s2
    out = _run(config, steps, drive_force)
    acceleration = force_n / config.mass_kg
    increments = np.diff(out[:, longitudinal.V_INDEX])
    assert np.allclose(increments, acceleration * config.dt_s, rtol=1.0e-9, atol=0.0)
    elapsed = steps * config.dt_s
    assert out[-1, longitudinal.V_INDEX] == pytest.approx(acceleration * elapsed, rel=1.0e-12)
    # Semi-implicit Euler sums v_1..v_n where the closed form assumes the mean speed, so a
    # constant-acceleration run lands half a step of velocity above 0.5*a*T^2. That offset is
    # the scheme's own leading error, not slack in the test: it is asserted, not absorbed.
    expected_distance = 0.5 * acceleration * elapsed * elapsed + 0.5 * acceleration * elapsed * (
        config.dt_s
    )
    assert out[-1, longitudinal.X_INDEX] == pytest.approx(expected_distance, rel=1.0e-12)
    closed_form = 0.5 * acceleration * elapsed * elapsed
    assert out[-1, longitudinal.X_INDEX] - closed_form == pytest.approx(
        0.5 * acceleration * elapsed * config.dt_s,
        rel=1.0e-9,
    )


def test_the_kernel_agrees_with_a_plain_python_reference(
    config: KernelConfig,
    drive_force: np.ndarray,
) -> None:
    """Bit for bit, not approximately: the compiled loop must do the arithmetic it claims."""
    steps = drive_force.size
    state = longitudinal.initial_state(speed_m_s=5.0)
    out = _run(config, steps, drive_force, state)
    expected = _reference(
        steps,
        config.dt_s,
        config.mass_kg,
        config.mass_kg * config.gravity_m_s2,
        state,
    )
    assert np.array_equal(out, expected)


def test_position_uses_the_updated_speed_so_the_scheme_is_semi_implicit(
    config: KernelConfig,
    drive_force: np.ndarray,
) -> None:
    """Symplectic Euler advances position with the *new* speed; explicit Euler does not.

    Both schemes agree on the speed column to the last bit, so a test that only checks speed
    cannot tell them apart. In the position column they differ by one step of velocity each step,
    so the scheme is pinned here twice: the trace must reproduce the semi-implicit update bit for
    bit, and it must differ from the explicit one by exactly the amount that predicts.
    """
    steps = 1_000
    out = _run(config, steps, drive_force[:steps])
    dt_s = config.dt_s
    for index in range(1, steps + 1):
        previous = out[index - 1]
        current = out[index]
        assert current[longitudinal.X_INDEX] == (
            previous[longitudinal.X_INDEX] + current[longitudinal.V_INDEX] * dt_s
        )
    explicit = np.zeros_like(out)
    explicit[0, :] = out[0, :]
    for index in range(1, steps + 1):
        explicit[index, longitudinal.V_INDEX] = out[index, longitudinal.V_INDEX]
        explicit[index, longitudinal.X_INDEX] = (
            explicit[index - 1, longitudinal.X_INDEX] + out[index - 1, longitudinal.V_INDEX] * dt_s
        )
    assert not np.array_equal(out, explicit)
    assert out[-1, longitudinal.X_INDEX] > explicit[-1, longitudinal.X_INDEX]
    gap = out[-1, longitudinal.X_INDEX] - explicit[-1, longitudinal.X_INDEX]
    # Each step the two schemes advance position by speeds one step apart, so the gap grows
    # by exactly `acceleration * dt^2` per step.
    expected_gap = steps * config.gravity_m_s2 * dt_s * dt_s
    assert gap == pytest.approx(expected_gap, rel=1.0e-12)


def test_a_representative_run_is_finite(
    config: KernelConfig,
    drive_force: np.ndarray,
) -> None:
    """One simulated second at 10 kHz: finite throughout, and accelerating the whole way."""
    steps = drive_force.size
    out = _run(config, steps, drive_force)
    assert np.isfinite(out).all()
    assert np.all(np.diff(out[:, longitudinal.V_INDEX]) > 0.0)
    assert np.all(np.diff(out[:, longitudinal.X_INDEX]) > 0.0)
    assert out[-1, longitudinal.V_INDEX] > 0.0
    assert out[-1, longitudinal.X_INDEX] > 0.0


def test_two_runs_with_the_same_inputs_are_byte_identical(
    config: KernelConfig,
    drive_force: np.ndarray,
    spec: CarSpec,
) -> None:
    """P1-T2: same spec, same buffers, same seed -> the same bytes, not merely the same numbers.

    The configuration is rebuilt from the spec for the second run, so the check also rules out
    a kernel that keys off the identity of the config object rather than its values.
    """
    first = _run(config, drive_force.size, drive_force)
    second = _run(spec.kernel_config(), drive_force.size, drive_force)
    assert first.tobytes() == second.tobytes()


def test_the_same_buffer_can_be_reused_for_a_second_identical_run(
    config: KernelConfig,
    drive_force: np.ndarray,
) -> None:
    """Reuse is the point of caller-owned buffers: the second run must not inherit the first."""
    steps = drive_force.size
    state = longitudinal.initial_state()
    out = longitudinal.allocate(steps)
    longitudinal.simulate(config, steps, state, drive_force, out)
    first = out.tobytes()
    out[:] = np.nan
    longitudinal.simulate(config, steps, state, drive_force, out)
    assert out.tobytes() == first


def test_the_run_reads_no_clock_and_no_random_source(
    config: KernelConfig,
    drive_force: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Byte-identity is only a determinism claim if nothing outside the arithmetic varies."""
    for module_name, attribute in TIME_AND_RANDOM:
        module = importlib.import_module(module_name)
        if hasattr(module, attribute):
            monkeypatch.setattr(module, attribute, partial(_refuse, f"{module_name}.{attribute}"))
    first = _run(config, drive_force.size, drive_force)
    second = _run(config, drive_force.size, drive_force)
    assert first.tobytes() == second.tobytes()


def test_the_step_loop_performs_no_allocation(config: KernelConfig) -> None:
    """PLAN.md section 4.1 rule 2: no allocation inside the loop, at any step count.

    Measured with Numba's own NRT allocation counters, the only thing that can see an
    allocation made inside compiled code. Compiled-to-compiled dispatch costs a fixed handful of
    allocations whatever the loop length, so the claim is that the count does not grow with the
    number of steps - a run a hundred times longer must not allocate any more than the short one.

    The first half of the test is the control: a kernel that *does* allocate inside its loop has
    to show a count that grows, otherwise the counters are not measuring anything and the second
    half would pass vacuously.
    """
    if os.environ.get("NUMBA_NRT_STATS") != "1":
        pytest.skip("numba NRT allocation counters are off; tests/conftest.py sets NUMBA_NRT_STATS")

    from numba import njit
    from numba.core.runtime import nrt

    @njit(cache=False)
    def allocating(steps: int) -> float:
        total = 0.0
        for index in range(steps):
            scratch = np.empty(index + 1, dtype=np.float64)
            total += scratch.size
        return total

    def allocations_for(steps: int) -> tuple[tuple[int, int], tuple[int, int]]:
        """Allocations for one run of the kernel, and for one run of the control kernel."""
        force = longitudinal.constant_longitudinal_force(steps=steps, force_n=1.0)
        state = longitudinal.initial_state()
        out = longitudinal.allocate(steps)
        longitudinal.simulate(config, steps, state, force, out)
        allocating(steps)
        before = nrt.rtsys.get_allocation_stats()
        longitudinal.simulate(config, steps, state, force, out)
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
    assert long_kernel[0] <= 8, f"unexpected per-call overhead: {long_kernel[0]} allocations"


def test_the_caller_owns_every_buffer_and_the_inputs_survive_the_run(
    config: KernelConfig,
    drive_force: np.ndarray,
) -> None:
    """The kernel writes only into the buffer it was handed, and hands the same one back."""
    steps = drive_force.size
    state = longitudinal.initial_state(speed_m_s=7.0)
    force_before = drive_force.tobytes()
    state_before = state.tobytes()
    out = longitudinal.allocate(steps)
    returned = longitudinal.simulate(config, steps, state, drive_force, out)
    assert np.shares_memory(returned, out)
    assert returned.ctypes.data == out.ctypes.data
    assert drive_force.tobytes() == force_before
    assert state.tobytes() == state_before


def test_editing_the_car_spec_changes_the_run_with_no_code_edit(
    config: KernelConfig,
    drive_force: np.ndarray,
    spec: CarSpec,
) -> None:
    """P1-T1b: the configuration is the only thing that decides the run.

    Mass and step are the two inputs the Task 2 kernel consumes, so editing either through
    ``dataclasses.replace`` - the same path a ``car_spec.yaml`` edit takes through the loader -
    must move the trace: twice the mass for half the acceleration per newton, twice the step
    for twice the velocity increment.
    """
    steps = drive_force.size
    baseline = _run(config, steps, drive_force)
    baseline_increment = baseline[1, longitudinal.V_INDEX] - baseline[0, longitudinal.V_INDEX]

    heavier = _run(replace(spec, mass_kg=spec.mass_kg * 2.0).kernel_config(), steps, drive_force)
    assert heavier.tobytes() != baseline.tobytes()
    heavier_increment = heavier[1, longitudinal.V_INDEX] - heavier[0, longitudinal.V_INDEX]
    assert heavier_increment == pytest.approx(baseline_increment / 2.0, rel=1.0e-12)

    slower = _run(replace(spec, dt_s=config.dt_s * 2.0).kernel_config(), steps, drive_force)
    slower_increment = slower[1, longitudinal.V_INDEX] - slower[0, longitudinal.V_INDEX]
    assert slower_increment == pytest.approx(baseline_increment * 2.0, rel=1.0e-12)


@pytest.mark.parametrize(
    ("steps", "force_size", "state_size", "rows"),
    [
        (4, 3, 2, 5),
        (4, 4, 1, 5),
        (4, 4, 2, 4),
        (4, 4, 2, 6),
    ],
    ids=["short-force", "short-state", "one-row-short", "one-row-long"],
)
def test_a_buffer_the_run_cannot_fill_is_refused_in_python(
    config: KernelConfig,
    steps: int,
    force_size: int,
    state_size: int,
    rows: int,
) -> None:
    """Every check the kernel cannot make belongs in the caller, before the loop starts."""
    force = np.zeros(force_size, dtype=np.float64)
    state = np.zeros(state_size, dtype=np.float64)
    out = np.zeros((rows, longitudinal.STATE_SIZE), dtype=np.float64)
    with pytest.raises(ValueError, match="buffer"):
        longitudinal.simulate(config, steps, state, force, out)


def test_a_float32_buffer_is_refused_in_python(config: KernelConfig) -> None:
    """The kernel is float64; a float32 buffer would silently re-quantise the trace."""
    force = np.zeros(4, dtype=np.float32)
    state = np.zeros(longitudinal.STATE_SIZE, dtype=np.float32)
    out = np.zeros((5, longitudinal.STATE_SIZE), dtype=np.float32)
    with pytest.raises(ValueError, match="float64"):
        longitudinal.simulate(config, 4, state, force, out)


def test_a_negative_step_count_is_refused_in_python(config: KernelConfig) -> None:
    """A negative step count is a caller mistake, and is caught before any buffer is sized."""
    with pytest.raises(ValueError, match="steps"):
        longitudinal.simulate(
            config,
            -1,
            longitudinal.initial_state(),
            np.zeros(0, dtype=np.float64),
            longitudinal.allocate(0),
        )
