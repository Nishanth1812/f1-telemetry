"""Shared normalized Magic Formula shape used by both tire axes."""

from __future__ import annotations

import math

from numba import njit

__all__ = ["magic_formula_shape"]


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def magic_formula_shape(scaled_slip: float, shape: float, curvature: float) -> float:
    """Return ``sin(C atan(x - E (x - atan(x))))`` for the already-scaled slip ``x``."""
    return math.sin(
        shape * math.atan(scaled_slip - curvature * (scaled_slip - math.atan(scaled_slip)))
    )
