"""P1-T1: the fixed-step straight-line kernel, tested against the claims it makes.

``PHASES.md`` P1-T1 asks for a flat ``@njit(cache=True, fastmath=False)`` kernel over
preallocated float64 arrays at ``dt = 100 µs``, and P1-T2 asks for the determinism harness
that proves it. The claims are checked here in the form that would actually fail:

* **It really is a compiled, cached kernel.** The dispatcher's own options are read back
  (``fastmath=False``, ``nogil``, ``boundscheck=False``) instead of being trusted from the
  decorator text, and ``cache=True`` is proven the way P0 proved it for the probe: numba wrote
  a cache index next to the module.
* **Fixed step, semi-implicit.** Row ``n`` of a trace is the state after exactly ``n`` steps of
  the configured ``dt_s``, and position advances with the *updated* speed, so the scheme is
  semi-implicit rather than explicit. The scheme is pinned against the explicit alternative,
  not against a number copied out of a run.
* **Determinism.** Two runs with the same configuration and the same buffers produce
  byte-identical traces, and the run reads no clock and no random source. Byte comparison, not
  ``allclose``: a tolerance would hide the last-bit differences a wall-clock read introduces.
* **No work the caller did not ask for.** The step loop allocates nothing measurable in Numba's
  runtime, the caller owns every buffer, and the inputs are left untouched.
* **One way in.** The compiled loop runs with ``boundscheck=False``, so ``simulate`` is the only
  door: a buffer of the wrong type, layout, size or writability, a step count that is not an
  integer, or a ``dt_s``/``mass_kg`` that would quietly produce NaNs is refused in Python,
  before the loop starts.

Every physical number here comes from the loaded ``car_spec.yaml``, including the drive force
used for the representative run - it is the car's own weight, so the run accelerates at one
gravity. There is no tuned constant in this file.

The ``kernel`` marker keeps these apart from the ``toolchain`` probe: that one proves Numba
compiles here, this one proves the car's integrator is right.
"""

from __future__ import annotations

import importlib
import math
import os
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from f1telemetry.contracts.car_spec import CarSpec
from f1telemetry.kernels import longitudinal

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import KernelConfig

pytestmark = pytest.mark.kernel

# The compiled loop is private to the module - that is the point of the fix these tests cover -
# and the dispatcher's own attributes are the only honest evidence of what the compiler was
# told, so the two tests below read them off the object rather than off the decorator text.
COMPILED_LOOP = longitudinal._integrate  # pyright: ignore[reportPrivateUsage]

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
    return np.full(steps_per_second, config.mass_kg * config.gravity_m_s2, dtype=np.float64)


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


def _cache_index_files() -> list[Path]:
    """The `.nbi` cache index files numba has written for this kernel module.

    ``cache=True`` writes one ``<module>.<signature>.nbi`` index next to the source, unless
    ``NUMBA_CACHE_DIR`` redirects it, and both are searched so the test does not depend on
    which is in force - the same search :mod:`f1telemetry.kernels.probe` documents for P0.
    """
    here = Path(str(longitudinal.__file__)).resolve()
    roots = [here.parent / "__pycache__"]
    override = os.environ.get("NUMBA_CACHE_DIR")
    if override:
        roots.insert(0, Path(override))
    found: list[Path] = []
    for root in roots:
        if root.is_dir():
            found.extend(sorted(root.glob("longitudinal.*.nbi")))
    return found


def _refuse(what: str) -> None:
    raise AssertionError(f"the simulation read {what}")


def test_the_kernel_compiles_to_machine_code_and_writes_a_compile_cache(
    config: KernelConfig,
) -> None:
    """A dispatcher entry is not a compiled kernel, and `cache=True` can do nothing silently.

    Two pieces of evidence, both measured rather than read back out of this file: calling the
    kernel registers a compiled signature, and that compile leaves a cache index for it beside
    the module. This is the check P0-T1b makes for the probe; the reason it is repeated here is
    that the claim under test is this kernel's, not the toolchain's.
    """
    _run(config, 4, np.zeros(4, dtype=np.float64))
    assert len(COMPILED_LOOP.signatures) >= 1, (
        "the run returned without compiling a signature, which means it fell back to Python"
    )
    assert _cache_index_files(), (
        "numba wrote no .nbi cache index for the longitudinal kernel, so cache=True is not "
        "taking effect"
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
    """100 µs is what the spec says, and a coasting row is the previous row by exactly that.

    This is the fixed-step claim in its simplest falsifiable form: one simulated second of rows
    covers exactly one second of travel, at the speed the caller seeded.
    """
    assert config.dt_s == pytest.approx(1.0e-4, rel=0.0, abs=0.0)
    assert round(1.0 / config.dt_s) == 10_000

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


def test_the_caller_owns_the_state_and_trace_buffers(config: KernelConfig) -> None:
    """Row 0 of the trace is the caller's state, and the caller hands over the whole buffer.

    ``steps`` of zero is the degenerate case of the same rule: row 0 is the seeded state and
    nothing else is written, so a run of no length is not a special case.
    """
    state = longitudinal.initial_state(distance_m=12.5, speed_m_s=30.0)
    assert state.shape == (longitudinal.STATE_SIZE,)
    assert state.dtype == np.float64
    assert state[longitudinal.X_INDEX] == 12.5
    assert state[longitudinal.V_INDEX] == 30.0

    steps = 5
    out = longitudinal.allocate(steps)
    assert out.shape == (steps + 1, longitudinal.STATE_SIZE)
    assert out.dtype == np.float64
    assert out.flags.writeable

    empty = longitudinal.allocate(0)
    returned = longitudinal.simulate(config, 0, state, np.zeros(0, dtype=np.float64), empty)
    assert np.array_equal(returned, state.reshape(1, longitudinal.STATE_SIZE))


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


def test_a_representative_run_is_finite_and_lands_on_the_analytic_run(
    config: KernelConfig,
    drive_force: np.ndarray,
) -> None:
    """One simulated second at 10 kHz: finite throughout, accelerating, and exactly where the
    scheme says it should be.

    Semi-implicit Euler sums v_1..v_n where the closed form assumes the mean speed, so a
    constant-acceleration run lands half a step of velocity above 0.5*a*T^2. That offset is the
    scheme's own leading error, asserted rather than absorbed by a tolerance.
    """
    steps = drive_force.size
    out = _run(config, steps, drive_force)
    acceleration = config.gravity_m_s2
    elapsed = steps * config.dt_s
    assert np.isfinite(out).all()
    assert np.all(np.diff(out[:, longitudinal.V_INDEX]) > 0.0)
    assert np.all(np.diff(out[:, longitudinal.X_INDEX]) > 0.0)
    assert np.allclose(
        np.diff(out[:, longitudinal.V_INDEX]),
        acceleration * config.dt_s,
        rtol=1.0e-9,
        atol=0.0,
    )
    assert out[-1, longitudinal.V_INDEX] == pytest.approx(acceleration * elapsed, rel=1.0e-12)
    closed_form = 0.5 * acceleration * elapsed * elapsed
    assert out[-1, longitudinal.X_INDEX] == pytest.approx(
        closed_form + 0.5 * acceleration * elapsed * config.dt_s,
        rel=1.0e-12,
    )


def test_a_second_run_reproduces_the_first_byte_for_byte(
    config: KernelConfig,
    drive_force: np.ndarray,
    spec: CarSpec,
) -> None:
    """P1-T2: same spec, same buffers, same inputs -> the same bytes, not merely the same numbers.

    The second run is deliberately hostile in two ways. Its config is rebuilt from the spec, so
    the check also rules out a kernel keyed off the identity of the config object rather than
    its values; and it writes over a buffer poisoned with NaN, so anything the first run left
    behind - an accumulator the caller never zeroed, a row the loop skipped - would show up.
    """
    steps = drive_force.size
    reused = _run(config, steps, drive_force)
    reused[:] = np.nan
    longitudinal.simulate(config, steps, longitudinal.initial_state(), drive_force, reused)
    rebuilt = _run(spec.kernel_config(), steps, drive_force)
    assert reused.tobytes() == rebuilt.tobytes()


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
        force = np.full(steps, 1.0, dtype=np.float64)
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


def test_read_only_state_and_force_are_accepted_because_the_kernel_never_writes_them(
    config: KernelConfig,
) -> None:
    """A caller may hand over inputs it does not own outright; ``out`` is the only destination."""
    steps = 64
    force_n = np.full(steps, config.mass_kg * config.gravity_m_s2, dtype=np.float64)
    force_n.flags.writeable = False
    state = longitudinal.initial_state(speed_m_s=3.0)
    state.flags.writeable = False
    out = longitudinal.allocate(steps)
    returned = longitudinal.simulate(config, steps, state, force_n, out)
    assert returned[-1, longitudinal.V_INDEX] == pytest.approx(
        3.0 + config.gravity_m_s2 * steps * config.dt_s,
        rel=1.0e-12,
    )
    assert force_n.tobytes() == np.full(steps, config.mass_kg * config.gravity_m_s2).tobytes()


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
    """Every check the loop cannot make belongs in the caller, before the loop starts."""
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


def test_a_buffer_that_is_not_an_ndarray_is_refused_in_python(config: KernelConfig) -> None:
    """A list would be converted by the dispatcher, and the caller would not get the buffer back."""
    not_an_array: Any = [0.0, 0.0]
    with pytest.raises(ValueError, match="ndarray"):
        longitudinal.simulate(config, 4, not_an_array, np.zeros(4), np.zeros((5, 2)))


def test_a_strided_buffer_is_refused_in_python(config: KernelConfig) -> None:
    """Contiguity is part of the kernel contract, so it is checked rather than silently honoured."""
    force = np.zeros(8, dtype=np.float64)[::2]
    assert not force.flags.c_contiguous
    with pytest.raises(ValueError, match="contiguous"):
        longitudinal.simulate(config, 4, longitudinal.initial_state(), force, np.zeros((5, 2)))


def test_a_read_only_output_buffer_is_refused_in_python(config: KernelConfig) -> None:
    """``out`` is the kernel's only destination, so a read-only buffer is an out-of-bounds write."""
    steps = 4
    out = longitudinal.allocate(steps)
    out.flags.writeable = False
    with pytest.raises(ValueError, match="read-only"):
        longitudinal.simulate(config, steps, longitudinal.initial_state(), np.zeros(steps), out)


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
    force = np.zeros(steps, dtype=np.float64)
    state = longitudinal.initial_state()
    with_int = longitudinal.simulate(config, steps, state, force, longitudinal.allocate(steps))
    numpy_integer: Any = np.int64(steps)
    with_numpy = longitudinal.simulate(
        config,
        numpy_integer,
        state,
        force,
        np.zeros((steps + 1, longitudinal.STATE_SIZE), dtype=np.float64),
    )
    assert with_int.tobytes() == with_numpy.tobytes()
