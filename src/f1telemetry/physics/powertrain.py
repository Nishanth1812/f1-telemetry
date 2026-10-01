"""P1-T4: the synthesised ICE torque curve and the turbo-lag multiplier that shapes it.

``PHASES.md`` P1-T4 asks for an ICE torque curve that is **synthesised rather than sourced**, with
a turbo-lag multiplier collapsing below about 4 000 rpm. The eight knots and both lag numbers are
``car_spec.yaml`` data reached through
:class:`~f1telemetry.contracts.car_spec.KernelConfig` like every other coefficient, and
``docs/calibration.md`` section 3 carries their provenance. What lives here is the two decisions
that turn a table into a torque any engine speed can ask for.

**Interpolation is inherited, not reinvented.** :func:`ice_torque_nm` reads the curve through
:func:`~f1telemetry.physics.forces.speed_curve` - piecewise linear between knots, held flat at
both ends - because that is the rule Task 3 froze for ``Cl(v)`` and ``Cd(v)`` against the same
reasoning: a table of points is a table, not a fit. One interpolator rather than two that agree
today.

**The lag is a threshold, and the threshold is configured data.** :func:`turbo_lag_factor`
returns the multiplier strictly below ``collapse_below_rpm`` and exactly ``1.0`` at and above it,
which is the literal reading of ``PLAN.md`` section 6's "falls off sharply below ~4 000 rpm" and
of the threshold ``car_spec.yaml`` records. ``PHASES.md`` P1-T4 asks for a multiplier "collapsing
below ~4 000 rpm" and names no ramp back, so a step is what was asked for. Torque is therefore
discontinuous at that rpm, and that is a property of the synthesised data rather than of this
code: the file names where the lag stops and not where full boost returns, so there is nothing to
recover over, and a recovery ramp invented here would put a number in Python that no file
supplies. ``docs/calibration.md`` section 3 states the same shape to a reader checking the file
against this code.

The committed lag threshold and ``idle_rpm`` are both 4 000 rpm, so the committed curve can
only be reduced below idle. Its lowest knot is 2 000 rpm, so the multiplier is live over
2 000-4 000 rpm and held flat below that, on the same clamped-end rule the curve follows.

**Two ways in, as in :mod:`f1telemetry.physics.forces`.** The compiled primitives take flat
scalars and ``float64`` arrays and are what a kernel calls, unvalidated because a loop cannot
afford Python in it; :func:`step_ice_torque` is the Python-facing composition that reads them off
the configuration and is the one place those values are checked before any arithmetic happens.
``KernelConfig`` is public and replaceable, so a loader's guarantees are a property of the loader
rather than a guarantee about the object. Nothing here has state, allocates, or reads a clock, so
a run's determinism does not have to be re-earned on this path.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

from f1telemetry.physics.forces import speed_curve  # noqa: TID251 -- same-layer kernel helper

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import KernelConfig

__all__ = [
    "ice_torque_nm",
    "step_ice_torque",
    "turbo_lag_factor",
]


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def turbo_lag_factor(
    rpm: float,
    collapse_below_rpm: float,
    multiplier_at_collapse: float,
) -> float:
    """``multiplier_at_collapse`` strictly below ``collapse_below_rpm``.

    Exactly ``1.0`` at or above.

    The turbo-lag reduction of ``PLAN.md`` section 6, expressed as the one decision it leaves
    open: a threshold at a configured engine speed rather than a spool characteristic, which
    would need a boost curve the regulations do not publish. ``1.0`` at the threshold rather than
    a hair under it, so ``collapse_below_rpm`` means what it says - and a caller can see the
    reduction in this return value rather than inferring it from the torque.
    """
    if rpm < collapse_below_rpm:
        return multiplier_at_collapse
    return 1.0


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def ice_torque_nm(
    rpm: float,
    torque_rpm: np.ndarray,
    torque_nm: np.ndarray,
    collapse_below_rpm: float,
    multiplier_at_collapse: float,
) -> float:
    """ICE torque in newton-metres at ``rpm``, with the turbo lag applied.

    The synthesised curve interpolated between its knots and held flat outside them, multiplied by
    :func:`turbo_lag_factor`. **Throttle is not applied here**: this is the engine's delivered
    torque at full request, and scaling it by throttle position is the drivetrain's call in
    P1-T5.

    Negative and zero ``rpm`` are not refused, because the ends are held flat and a stalled or
    backfiring engine should get a finite number rather than an index error. ``torque_rpm`` must
    be strictly increasing and share a length with ``torque_nm``, which the ``car_spec.yaml``
    loader checks before it builds a ``KernelConfig`` and :func:`step_ice_torque` checks again
    before it hands over a config that may have been replaced. The one rule no loader applies is
    the sign of ``torque_nm``, so it is :func:`step_ice_torque`'s alone.
    """
    return speed_curve(rpm, torque_rpm, torque_nm) * turbo_lag_factor(
        rpm, collapse_below_rpm, multiplier_at_collapse
    )


def step_ice_torque(config: KernelConfig, rpm: float) -> float:
    """ICE torque in newton-metres at ``rpm``, read off a validated configuration.

    The Python-facing entry point, and the reason it exists: a caller inside a kernel reads the
    curve and the two lag scalars out of the config once, outside its loop, and calls
    :func:`ice_torque_nm` directly - the dataclass is not something a compiled loop can hold.
    That a caller outside a kernel can ask for the torque in one call is the whole of the job.

    ``KernelConfig`` is public and replaceable, so the values are checked here rather than
    assumed to have been checked by whoever built it. ``ice_torque_nm`` is compiled with
    ``boundscheck=False`` and no finiteness checks, so an empty curve reads uninitialised memory,
    a mismatched length reads past the end of ``torque_nm``, a reversed axis finds a segment that
    is not there, and a NaN ``rpm`` weighs nothing - each returning a plausible torque rather than
    an error, and a wrong torque is exactly what this model is asked to produce. A negative knot
    reads just as quietly and for the other reason: it is not a lookup failure but an engine that
    pushes the wrong way. The primitives stay unvalidated because a kernel calls them once per
    step and cannot afford Python.

    ``rpm`` and both lag scalars are narrowed to ``float`` so equivalent ``int`` and ``float``
    callers share one compiled specialisation rather than compiling a second signature for each.
    """
    rpm = _checked_float("rpm", rpm)
    collapse_rpm = _checked_float("config.turbo_lag_collapse_rpm", config.turbo_lag_collapse_rpm)
    multiplier = _checked_float("config.turbo_lag_multiplier", config.turbo_lag_multiplier)
    if collapse_rpm <= 0.0:
        raise ValueError(
            "step_ice_torque: config.turbo_lag_collapse_rpm must be finite and > 0, "
            f"got {collapse_rpm!r}"
        )
    if not 0.0 < multiplier <= 1.0:
        raise ValueError(
            f"step_ice_torque: config.turbo_lag_multiplier must be in (0, 1], got {multiplier!r}"
        )
    _check_torque_arrays(config)
    return ice_torque_nm(
        rpm,
        config.torque_rpm,
        config.torque_nm,
        collapse_rpm,
        multiplier,
    )


def _checked_float(label: str, value: object) -> float:
    """Return a finite numeric scalar as float, or fail before it reaches Numba."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"step_ice_torque: {label} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"step_ice_torque: {label} must be finite, got {value!r}")
    return number


def _check_torque_arrays(config: KernelConfig) -> None:
    """Validate the torque curve before it reaches the compiled lookup.

    Every rule here is one ``speed_curve`` cannot make for itself once the bounds check is off.
    The sign rule is the one that is not about reading the table at all: ``ice_torque_nm``
    returns whatever the table says, so a negative knot is a backwards push at full throttle
    request, and a drivetrain that cannot tell engine braking from drive torque is exactly the
    drivetrain P1-T5 builds. Zero is allowed, because an engine that makes no torque is a real
    operating point rather than a malformed table - unlike ``Cl`` and ``Cd``, which have no zero.
    """
    axis, values = config.torque_rpm, config.torque_nm
    arrays = (("torque_rpm", axis), ("torque_nm", values))
    for name, array in arrays:
        if (
            not isinstance(array, np.ndarray)
            or array.dtype != np.float64
            or array.ndim != 1
            or array.size == 0
            or not array.flags.c_contiguous
        ):
            raise ValueError(
                f"step_ice_torque: config.{name} must be a non-empty C-contiguous float64 vector"
            )
    if axis.shape != values.shape:
        raise ValueError(
            "step_ice_torque: config.torque_rpm and config.torque_nm must share a length, "
            f"got {axis.shape} and {values.shape}"
        )
    for name, array in arrays:
        if not np.isfinite(array).all():
            raise ValueError(f"step_ice_torque: config.{name} must be finite")
    if not np.all(np.diff(axis) > 0.0):
        raise ValueError("step_ice_torque: config.torque_rpm must be strictly increasing")
    if np.any(values < 0.0):
        raise ValueError("step_ice_torque: config.torque_nm must be non-negative")
