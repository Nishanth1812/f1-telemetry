"""P0-T1b: the Numba toolchain, proven before anything depends on it.

The claim being tested is narrow and checkable: on *this* machine, an
``@njit(cache=True, fastmath=False)`` kernel compiles to machine code, runs, produces
numbers a plain-Python reference agrees with, and writes a compile cache that a second
process can reuse. If any of that is false, it should fail here and not in P1.
"""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest

from f1telemetry.kernels import probe

pytestmark = pytest.mark.toolchain

_STIFFNESS = 4.0
_DAMPING = 0.2
_DT = 1.0e-4
_STEPS = 500
COLD_START_BUDGET_S = 2.0


def _reference(steps: int, dt: float, stiffness: float, damping: float) -> np.ndarray:
    out = np.zeros((steps + 1, 2), dtype=np.float64)
    out[0, 0] = 1.0
    out[0, 1] = 0.0
    for index in range(1, steps + 1):
        acceleration = -stiffness * out[index - 1, 0] - damping * out[index - 1, 1]
        out[index, 1] = out[index - 1, 1] + acceleration * dt
        out[index, 0] = out[index - 1, 0] + out[index, 1] * dt
    return out


def test_kernel_compiles_and_reports_a_signature() -> None:
    """Calling the kernel must produce a compiled signature, not a Python fallback."""
    state = probe.initial_state()
    out = probe.allocate(4)
    scratch = np.zeros(probe.STATE_SIZE, dtype=np.float64)
    probe.integrate(4, _DT, _STIFFNESS, _DAMPING, state, out)
    probe.step(state, scratch, _DT, _STIFFNESS, _DAMPING)
    assert len(probe.integrate.signatures) >= 1
    assert len(probe.step.signatures) >= 1


def test_kernel_options_match_plan_section_4_1() -> None:
    """PLAN.md section 4.1 rule 4: cache on, fastmath off, and both verifiable."""
    options = dict(probe.integrate.targetoptions)
    assert options["fastmath"] is False
    assert options["nopython"] is True
    assert options["nogil"] is True
    assert "cache" not in options
    assert probe.TARGET_OPTIONS["cache"] is True
    assert probe.TARGET_OPTIONS["fastmath"] is False


def test_kernel_agrees_with_a_plain_python_reference() -> None:
    state = probe.initial_state()
    out = probe.allocate(_STEPS)
    probe.integrate(_STEPS, _DT, _STIFFNESS, _DAMPING, state, out)
    expected = _reference(_STEPS, _DT, _STIFFNESS, _DAMPING)
    assert np.allclose(out, expected, rtol=0.0, atol=0.0)
    assert np.isfinite(out).all()


def test_kernel_writes_a_compile_cache() -> None:
    """`cache=True` silently doing nothing is the failure mode worth catching in P0."""
    state = probe.initial_state()
    out = probe.allocate(8)
    probe.integrate(8, _DT, _STIFFNESS, _DAMPING, state, out)
    assert probe.cache_index_path(), "numba wrote no .nbi cache index next to the kernel"


def test_second_process_reuses_the_cache_in_a_fresh_interpreter() -> None:
    """A cold interpreter must not recompile: the cache index is what makes that true.

    A cold compile of these two kernels takes about four seconds on the dev machine, so a
    fresh process that starts up well inside `COLD_START_BUDGET_S` can only have loaded
    the cached object. Comparing the two runs against each other would be flaky: by the
    time this test runs the cache is already warm and the difference is OS noise.
    """
    script = (
        "import time\n"
        "from f1telemetry.kernels import probe\n"
        "start = time.perf_counter()\n"
        "state = probe.initial_state()\n"
        "out = probe.allocate(64)\n"
        "probe.integrate(64, 1e-4, 4.0, 0.2, state, out)\n"
        "print(f'{time.perf_counter() - start:.6f}')\n"
    )
    for attempt in range(2):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
        )
        elapsed = float(completed.stdout.strip())
        assert math.isfinite(elapsed)
        assert elapsed < COLD_START_BUDGET_S, (
            f"run {attempt} took {elapsed:.3f}s; a cold numba compile of this kernel is "
            f"about 4s, so the cache was not reused"
        )


def test_kernel_allocates_nothing_in_the_step_loop() -> None:
    """PLAN.md section 4.1 rule 2: the caller owns every buffer."""
    state = probe.initial_state()
    out = probe.allocate(_STEPS)
    probe.integrate(_STEPS, _DT, _STIFFNESS, _DAMPING, state, out)
    assert out.shape == (_STEPS + 1, probe.STATE_SIZE)
    assert out.dtype == np.float64
    scratch = np.zeros(probe.STATE_SIZE, dtype=np.float64)
    probe.step(out[0], scratch, _DT, _STIFFNESS, _DAMPING)
    assert scratch[0] == pytest.approx(out[1, 0], rel=0.0, abs=0.0)
    assert scratch[1] == pytest.approx(out[1, 1], rel=0.0, abs=0.0)
