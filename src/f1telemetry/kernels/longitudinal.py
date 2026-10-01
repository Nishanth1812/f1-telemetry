"""P1-T1: the straight-line longitudinal kernel - P0 Numba conventions, one car at a time.

:mod:`f1telemetry.kernels.probe` proved the toolchain with a damped oscillator. This is the
first module that simulates a car, and it is deliberately almost the same shape: a flat
numeric kernel over preallocated ``float64`` arrays, written to the six rules of ``PLAN.md``
section 4.1.

**State.** Two scalars, in the fixed order ``[distance_m, speed_m_s]`` and named by
:data:`X_INDEX` / :data:`V_INDEX` so a caller never indexes a raw column. ``speed_m_s`` is the
longitudinal velocity - the ``vx`` of ``PLAN.md`` section 4's state vector, which is all a
straight-line model has - so the two views line up when Task 4 writes a real scenario.
Everything else P1 needs is not here yet: the aero, tyre and drivetrain states arrive with the
tasks that own them, and the layout is expected to grow. :data:`STATE_SIZE` and the index
constants are the thing that growth must update.

**Forces are an input, not a model.** The kernel is told the net longitudinal force for each
step through a caller-owned array; deciding what that force is belongs to the aero and tyre
work of Task 3 and the drivetrain work of Task 4. That split is why the integrator can be
tested exactly here, against arithmetic that is written out one step at a time.

**The step is data.** ``dt`` comes from ``car_spec.yaml`` through :class:`KernelConfig`, so
changing the integration rate is a data edit (the cross-phase rule that no physical constant is
hardcoded in Python). At 100 µs this is a 10 kHz kernel, which is what ``PLAN.md`` section 4
fixes; the tests assert the file still says so.

**Determinism.** No clock, no random source, no allocation, no ``prange``: same
:class:`KernelConfig`, same buffers, same inputs, byte-identical trace. ``cache=True`` and
``fastmath=False`` are not preferences here - ``fastmath`` permits floating-point
reassociation, which would make the byte-identity claim version-dependent rather than a
property of the kernel.

Neither kernel validates anything, and neither can: a compiled function that raises would be a
different kind of kernel. Everything the loop cannot check - buffer shapes, dtypes and sizes -
is checked by :func:`simulate` before the loop starts, which is where ``PHASES.md`` says such
checks belong, so a mistake surfaces as a :class:`ValueError` naming the buffer instead of as
an out-of-bounds read at speed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
from numba import njit

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import KernelConfig

__all__ = [
    "STATE_SIZE",
    "TARGET_OPTIONS",
    "V_INDEX",
    "X_INDEX",
    "allocate",
    "constant_longitudinal_force",
    "initial_state",
    "integrate",
    "simulate",
    "step",
]

STATE_SIZE: Final[int] = 2
X_INDEX: Final[int] = 0
V_INDEX: Final[int] = 1

# Mirrors the probe's options. Kept as data so a test can assert the compiler was told what
# this module claims it was told, rather than trusting the decorator text.
TARGET_OPTIONS: Final[dict[str, object]] = {
    "cache": True,
    "fastmath": False,
    "nogil": True,
    "boundscheck": False,
    "error_model": "numpy",
}


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def step(state: np.ndarray, force_n: float, out: np.ndarray, dt_s: float, mass_kg: float) -> None:
    """One semi-implicit Euler step of the straight-line point mass.

    ``state`` and ``out`` are ``[distance_m, speed_m_s]``. Semi-implicit means the position is
    advanced with the speed this step produced, not the speed it started with; the explicit
    alternative is the same arithmetic with ``state[V_INDEX]`` on the second line, and the two
    schemes part company by one step of velocity - ``acceleration * dt_s**2`` per step, growing
    with the run - so the choice is pinned by a test rather than left to taste.
    """
    acceleration = force_n / mass_kg
    out[V_INDEX] = state[V_INDEX] + acceleration * dt_s
    out[X_INDEX] = state[X_INDEX] + out[V_INDEX] * dt_s


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def integrate(
    steps: int,
    dt_s: float,
    mass_kg: float,
    force_n: np.ndarray,
    state: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    """Run ``steps`` fixed steps into caller-owned ``out``, shape ``(steps + 1, STATE_SIZE)``.

    Row 0 of ``out`` is seeded from ``state`` and row ``n`` is the state after exactly ``n``
    steps of ``dt_s``, so row ``n`` is simulated time ``n * dt_s`` and the trace has one row
    per step plus the initial condition. ``force_n`` holds the net longitudinal force for each
    step, so no force is computed here and none is allocated.

    The loop reads and writes only the three arrays it was given, and returns the caller's
    ``out`` so a scenario can hold the buffer it wrote into.
    """
    out[0, X_INDEX] = state[X_INDEX]
    out[0, V_INDEX] = state[V_INDEX]
    for index in range(steps):
        acceleration = force_n[index] / mass_kg
        speed = out[index, V_INDEX] + acceleration * dt_s
        out[index + 1, V_INDEX] = speed
        out[index + 1, X_INDEX] = out[index, X_INDEX] + speed * dt_s
    return out


def initial_state(distance_m: float = 0.0, speed_m_s: float = 0.0) -> np.ndarray:
    """Caller-owned float64 state buffer, ``[distance_m, speed_m_s]``."""
    return np.array([distance_m, speed_m_s], dtype=np.float64)


def allocate(steps: int) -> np.ndarray:
    """Caller-owned output buffer, shape ``(steps + 1, STATE_SIZE)``."""
    _check_steps(steps)
    return np.zeros((steps + 1, STATE_SIZE), dtype=np.float64)


def constant_longitudinal_force(steps: int, force_n: float = 0.0) -> np.ndarray:
    """Caller-owned force history, shape ``(steps,)``, constant over the run.

    Task 3 and Task 4 replace this with computed forces. It exists now because the integrator
    takes force as an input, so a run needs a force array whether or not anything can yet
    produce a physically interesting one.
    """
    _check_steps(steps)
    return np.full(steps, force_n, dtype=np.float64)


def simulate(
    config: KernelConfig,
    steps: int,
    state: np.ndarray,
    force_n: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    """Integrate a straight-line run from a validated :class:`KernelConfig`, in fixed steps.

    The one entry point a scenario uses. It reads ``dt_s`` and ``mass_kg`` from the config once,
    here in Python, and hands the compiled loop nothing but numbers - the kernel never sees the
    config object, a YAML document or a keyword argument.

    Every buffer is the caller's, and every one is validated before the loop starts rather than
    inside it: a bad shape or dtype raises :class:`ValueError` naming the buffer, which is the
    one-way ``@njit`` boundary ``PHASES.md`` asks for. ``steps`` may be zero, which writes the
    initial state into row 0 and nothing else.

    Returns ``out``, so a caller can write ``out = simulate(...)`` without giving up the buffer.
    """
    _check_steps(steps)
    _check_buffer("state", state, (STATE_SIZE,))
    _check_buffer("force_n", force_n, (steps,))
    _check_buffer("out", out, (steps + 1, STATE_SIZE))
    return integrate(steps, config.dt_s, config.mass_kg, force_n, state, out)


def _check_steps(steps: int) -> None:
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}")


def _check_buffer(name: str, array: np.ndarray, shape: tuple[int, ...]) -> None:
    """Refuse a buffer the kernel would read out of bounds or quietly re-quantise."""
    if array.dtype != np.float64:
        raise ValueError(
            f"simulate: {name} buffer must be float64, got {array.dtype}. The kernel is "
            "float64 end to end and a narrower buffer would re-quantise the trace"
        )
    if array.shape != shape:
        raise ValueError(f"simulate: {name} buffer must have shape {shape}, got {array.shape}")
