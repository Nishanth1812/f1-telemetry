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

The compiled loop is :func:`_integrate`, and it is private on purpose: ``boundscheck=False``
means an undersized buffer is an out-of-bounds write with no error at all, so the only way
that can be safe is for there to be exactly one way in. :func:`simulate` is that way. It is the
whole boundary - the one-way ``@njit`` boundary ``PHASES.md`` asks for - and it refuses a
nonfinite or nonpositive ``dt_s`` or ``mass_kg``, a step count that is not an integer, and any
buffer that is not a C-contiguous ``float64`` ``ndarray`` of exactly the size the run needs. A
mistake surfaces as a :class:`ValueError` naming the buffer instead of as a silent write past
the end of an array.
"""

from __future__ import annotations

import math
import operator
from typing import TYPE_CHECKING, Final

import numpy as np
from numba import njit

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import KernelConfig

__all__ = [
    "STATE_SIZE",
    "V_INDEX",
    "X_INDEX",
    "allocate",
    "initial_state",
    "simulate",
]

STATE_SIZE: Final[int] = 2
X_INDEX: Final[int] = 0
V_INDEX: Final[int] = 1


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def _integrate(
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

    Semi-implicit means the position is advanced with the speed this step produced, not the
    speed it started with; the explicit alternative is the same arithmetic with ``out[index,
    V_INDEX]`` on the last line, and the two schemes part company by one step of velocity per
    step, so the choice is pinned by a test rather than left to taste.

    The loop reads and writes only the three arrays it was given, and returns the caller's
    ``out`` so a scenario can hold the buffer it wrote into. Nothing is checked here - see the
    module docstring - so it must only ever be reached through :func:`simulate`.
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
    count = _checked_steps(steps)
    return np.zeros((count + 1, STATE_SIZE), dtype=np.float64)


def simulate(
    config: KernelConfig,
    steps: int,
    state: np.ndarray,
    force_n: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    """Integrate a straight-line run from a validated :class:`KernelConfig`, in fixed steps.

    The one entry point a scenario uses, and the only thing in this module that touches the
    compiled loop. It reads ``dt_s`` and ``mass_kg`` out of the config once, here in Python, and
    hands the compiled loop nothing but numbers - the kernel never sees the config object, a
    YAML document or a keyword argument.

    Every buffer is the caller's, and every one is validated before the loop starts rather than
    inside it, because ``boundscheck=False`` means the loop cannot. So are the two config
    scalars and the step count: :class:`KernelConfig` is normally range-checked by the loader,
    but it is a public frozen dataclass and can be built or replaced directly, and a zero mass
    or a NaN step is arithmetic that returns NaNs rather than an error. ``steps`` may be zero,
    which writes the initial state into row 0 and nothing else.

    Returns ``out``, so a caller can write ``out = simulate(...)`` without giving up the buffer.
    """
    count = _checked_steps(steps)
    for name, value in (("dt_s", config.dt_s), ("mass_kg", config.mass_kg)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"simulate: config.{name} must be finite and > 0, got {value!r}. The kernel "
                "divides by mass_kg and multiplies by dt_s, so neither value is checked for "
                "you once the loop starts"
            )
    _check_buffer("state", state, (STATE_SIZE,), writable=False)
    _check_buffer("force_n", force_n, (count,), writable=False)
    _check_buffer("out", out, (count + 1, STATE_SIZE), writable=True)
    return _integrate(count, config.dt_s, config.mass_kg, force_n, state, out)


def _checked_steps(steps: int) -> int:
    """The step count as a genuine ``int``, or a :class:`ValueError`.

    ``operator.index`` rather than ``int()`` on purpose: a float count would otherwise be
    passed straight to the compiled loop, where it compiles a *second* float64 specialisation
    of the same integrator and a ``4.0`` step count would quietly succeed. ``bool`` is refused
    even though it is an ``int``, because ``simulate(config, True, ...)`` is a bug.
    """
    if isinstance(steps, bool):
        raise ValueError(f"steps must be an integer, got {steps!r} of type {type(steps).__name__}")
    try:
        count = operator.index(steps)
    except TypeError as error:
        raise ValueError(
            f"steps must be an integer, got {steps!r} of type {type(steps).__name__}"
        ) from error
    if count < 0:
        raise ValueError(f"steps must be >= 0, got {count}")
    return count


def _check_buffer(name: str, array: object, shape: tuple[int, ...], *, writable: bool) -> None:
    """Refuse a buffer the kernel would read out of bounds or quietly re-quantise.

    ``state`` and ``force_n`` are read-only as far as the kernel is concerned, so a caller may
    hand over a buffer it does not want written; ``out`` is the kernel's only destination and
    must be writable. Contiguity is part of the contract in ``car_spec.KernelConfig`` and in
    ``PLAN.md`` section 4.1, and a strided buffer would compile as a differently-typed kernel
    rather than being rejected.
    """
    if not isinstance(array, np.ndarray):
        raise ValueError(
            f"simulate: {name} must be a numpy ndarray, got {type(array).__name__}. Anything "
            "else would be copied into the kernel, so the buffer the caller holds is not the "
            "one the run writes"
        )
    if array.dtype != np.float64:
        raise ValueError(
            f"simulate: {name} buffer must be float64, got {array.dtype}. The kernel is "
            "float64 end to end and a narrower buffer would re-quantise the trace"
        )
    if array.shape != shape:
        raise ValueError(f"simulate: {name} buffer must have shape {shape}, got {array.shape}")
    if not array.flags.c_contiguous:
        raise ValueError(
            f"simulate: {name} buffer must be C-contiguous; a strided buffer compiles as a "
            "different kernel signature and the run stops being the one the tests pin"
        )
    if writable and not array.flags.writeable:
        raise ValueError(
            f"simulate: {name} buffer is read-only. It is the kernel's only destination, so a "
            "read-only buffer makes the run write out of bounds"
        )
