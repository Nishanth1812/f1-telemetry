"""P0-T1b - prove the Numba toolchain end to end on this machine, in P0.

This is **not** vehicle dynamics and makes no claim to be. It is a two-state
semi-implicit Euler integrator over a damped harmonic oscillator, written to the six
rules of PLAN.md section 4.1 so that the rules themselves are proven before the physics
core depends on them:

1. flat numeric kernel - float64 numpy arrays in, float64 numpy arrays out
2. all state in preallocated arrays owned by the caller, no allocation in the loop
3. configuration passed in as scalars from ``car_spec.yaml``-shaped data, never looked up
   by name inside the kernel
4. ``cache=True``, ``fastmath=False``
5. no ``prange``
6. everything outside the kernel is ordinary Python

Windows 64-bit is supported and LLVM ships inside the llvmlite wheel, so no compiler
should be needed. The test asserts that on this machine rather than trusting it, and also
asserts the compile cache is actually written, because ``cache=True`` silently doing
nothing is the failure mode worth catching now.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = [
    "STATE_SIZE",
    "TARGET_OPTIONS",
    "allocate",
    "cache_index_path",
    "initial_state",
    "integrate",
    "step",
]

STATE_SIZE = 2

TARGET_OPTIONS = {
    "cache": True,
    "fastmath": False,
    "nogil": True,
    "boundscheck": False,
    "error_model": "numpy",
}


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def step(state: np.ndarray, out: np.ndarray, dt: float, stiffness: float, damping: float) -> None:
    """One semi-implicit Euler step. `state` and `out` are [position, velocity]."""
    acceleration = -stiffness * state[0] - damping * state[1]
    out[1] = state[1] + acceleration * dt
    out[0] = state[0] + out[1] * dt


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def integrate(
    steps: int,
    dt: float,
    stiffness: float,
    damping: float,
    state: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    """Run `steps` fixed steps into caller-owned `out`, shape (steps + 1, 2).

    Row 0 of `out` is seeded from `state`, so the caller owns every buffer and never has
    to write into the output before handing it over.
    """
    out[0, 0] = state[0]
    out[0, 1] = state[1]
    for index in range(1, steps + 1):
        acceleration = -stiffness * out[index - 1, 0] - damping * out[index - 1, 1]
        out[index, 1] = out[index - 1, 1] + acceleration * dt
        out[index, 0] = out[index - 1, 0] + out[index, 1] * dt
    return out


def initial_state() -> np.ndarray:
    """Caller-owned float64 state buffer: displaced and stationary."""
    return np.array([1.0, 0.0], dtype=np.float64)


def allocate(steps: int) -> np.ndarray:
    """Caller-owned output buffer, shape (steps + 1, 2)."""
    return np.zeros((steps + 1, STATE_SIZE), dtype=np.float64)


def cache_index_path() -> list[str]:
    """Candidate locations of the Numba compile cache index for this module.

    ``cache=True`` writes a ``<module>.<signature>.nbi`` index next to the source by
    default; ``NUMBA_CACHE_DIR`` overrides that. Both are searched so the test does not
    depend on which one is in force.
    """
    import os
    from pathlib import Path

    here = Path(__file__).resolve()
    roots = [here.parent / "__pycache__"]
    override = os.environ.get("NUMBA_CACHE_DIR")
    if override:
        roots.insert(0, Path(override))
    found: list[str] = []
    for root in roots:
        if not root.is_dir():
            continue
        found.extend(str(p) for p in sorted(root.glob(f"{here.stem}.*.nbi")))
    return found
