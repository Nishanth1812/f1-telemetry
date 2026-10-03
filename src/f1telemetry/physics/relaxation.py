"""P2-T5: tyre relaxation - the transient slip states that chase their targets.

``PLAN.md`` section 4 puts the relaxation length on both slip axes (
``sigma = C / Cy`` in Limebeer & Tremlett's terms), and ``car_spec.yaml``
carries the configured lengths as synthetised placeholders. This module is
that machinery and nothing else: no combined slip, no chassis integration,
no force outputs. It answers one question - given the caller-owned slip
state, the steady target the current kinematics imply, the patch speed and
the configured lengths, what is the slip one fixed 100 us step later - and
it answers it as an exact exponential update with a finite low-speed
floor.

**The update**, integrated exactly rather than as forward Euler::

    v_eff  = max(|v_patch|, v_min)                  m/s
    s_{n+1} = s_target + (s_n - s_target) exp(-v_eff dt / L)

The time constant is ``L / v_eff``: a corner that slips its patch for one
relaxation length has gone ``1 - e^-1`` of the way to the target, which is
what a relaxation *length* means. The exact exponential is used rather
than the ``dt / tau`` first-order step because the fixed 100 us step is
not always small against ``tau`` - at a 20 m/s patch with a 0.3 m length
``dt / tau`` is a little inside 7e-3, but near the low-speed floor it
approaches ``v_min dt / L``, and Euler would overshoot the target there
while the exponential cannot: the multiplier ``exp(-v_eff dt / L)`` is in
``(0, 1]`` for every nonnegative ``dt``, so the state approaches the
target monotonically and never crosses it. That property is the whole
point of relaxing slip instead of copying it.

**Units and signs.** Longitudinal slip ratio ``kappa`` is dimensionless,
positive for drive (the P1 convention). Lateral slip angle ``alpha`` is
in degrees at this boundary, matching ``tyres.step_lateral_force`` - the
same quantity the relaxed state is eventually handed to the lateral Magic
Formula. Patch speed is in m/s and enters as a magnitude: the lag is a
property of how much rubber has passed under the patch, not of which way
it travelled. Relaxation lengths are in metres, ``dt`` in seconds, and
the per-wheel target state is the caller's: this module relaxes a slip it
is handed and returns nothing that looks like a force.

**The low-speed floor is a model choice, labelled as one.** A stopped
wheel has no patch travel, so ``L / v`` would divide by zero; the
configured ``min_speed_m_s`` keeps the lag finite and simply means a
stationary patch relaxes at the floor rate. Below the floor the decay is
pinned to the floor; above it the actual magnitude wins. The floor is the
same kind of simulation guard as ``slip_ratio_min_speed_m_s`` in the P1
slip-ratio guard, and it is the caller's, not a hidden constant.

**Zero load is the tyre model's problem, deliberately.** No function
here reads a vertical load: a lifted wheel's slip state keeps tracking
its target, and the force that wheel produces is exactly zero because
the tyre primitives guard the load. Mixing the load guard into the slip
state would make the state's history depend on the force model instead
of on the kinematics it describes, and the plan wants the order
explicit: relaxation first, then combined slip, then the tyre force.

**Two ways in, deliberately.** The primitives take flat scalars and
caller-owned ``float64`` vectors and are what a compiled kernel calls;
:func:`step_relaxation` is the Python-facing composition that checks the
configuration and the state before any arithmetic happens. The kernel
reads the configuration once, outside its loop, and hands the primitives
numbers. The state vectors are *caller-owned*: nothing in this module
allocates a slip state, and the step writes through the arrays it is
given, because a Numba loop owns no buffers of its own. Nothing here
reads a regulation: the lengths and the floor are synthetised
placeholders in ``car_spec.yaml``, labelled as such in the file itself.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import numpy as np
from numba import njit

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import KernelConfig

__all__ = [
    "decay_factor",
    "relax_slip_ratio",
    "relax_slip_state",
    "step_relaxation",
    "validated_relaxation_scalars",
]

# The four per-wheel vectors are length four, in the corner order every
# other per-wheel quantity uses. The name keeps a bare 4 out of the
# validation below.
_WHEEL_COUNT: Final[int] = 4

# The njit options are written out on every function rather than shared
# through one dict, for the same reason `tests/test_relaxation.py` reads
# them off the dispatcher: an indirection between the reader and the
# options is one more place for them to be wrong, and these are the
# options the project's determinism claim rests on.


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def decay_factor(
    patch_speed_m_s: float,
    relaxation_length_m: float,
    dt_s: float,
    min_speed_m_s: float,
) -> float:
    """``exp(-v_eff dt / L)``: the fraction of the gap that survives one step.

    ``v_eff`` is the patch speed's magnitude floored at the configured
    low-speed value, so a stopped patch still relaxes - at the floor rate
    - rather than stalling or dividing by zero. The returned multiplier
    is in ``(0, 1]`` for every nonnegative ``dt``, which is what keeps the
    relaxed state monotone toward its target and unable to overshoot.
    """
    speed = abs(patch_speed_m_s)
    if speed < min_speed_m_s:
        speed = min_speed_m_s
    return math.exp(-speed * dt_s / relaxation_length_m)


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def relax_slip_ratio(
    current: float,
    target: float,
    patch_speed_m_s: float,
    relaxation_length_m: float,
    dt_s: float,
    min_speed_m_s: float,
) -> float:
    """One exact-exponential relaxation step for one scalar slip.

    ``s' = target + (s - target) exp(-v_eff dt / L)``: the new slip is the
    target plus whatever fraction of the old offset survived the step.
    Same arithmetic for the longitudinal slip ratio and the lateral slip
    angle - only the caller's relaxation length differs - so a single
    scalar primitive serves both axes rather than two that could drift.
    """
    return target + (current - target) * decay_factor(
        patch_speed_m_s, relaxation_length_m, dt_s, min_speed_m_s
    )


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def relax_slip_state(
    slip_ratio_state: np.ndarray,
    slip_ratio_target: np.ndarray,
    slip_angle_state: np.ndarray,
    slip_angle_target: np.ndarray,
    patch_speed_m_s: np.ndarray,
    longitudinal_length_m: float,
    lateral_length_m: float,
    dt_s: float,
    min_speed_m_s: float,
) -> None:
    """Relax both slip axes, all four wheels, one step, in place.

    Every vector is caller-owned, length four, in ``FL, FR, RL, RR`` order,
    updated through the passed arrays rather than by returning new ones -
    a kernel loop owns no buffers of its own. Both axes share one patch
    speed per wheel: the travel past the contact patch sets the lag for
    the longitudinal and the lateral state alike, so only the configured
    length differs between them.
    """
    for wheel in range(_WHEEL_COUNT):
        slip_ratio_state[wheel] = relax_slip_ratio(
            slip_ratio_state[wheel],
            slip_ratio_target[wheel],
            patch_speed_m_s[wheel],
            longitudinal_length_m,
            dt_s,
            min_speed_m_s,
        )
        slip_angle_state[wheel] = relax_slip_ratio(
            slip_angle_state[wheel],
            slip_angle_target[wheel],
            patch_speed_m_s[wheel],
            lateral_length_m,
            dt_s,
            min_speed_m_s,
        )


def validated_relaxation_scalars(config: KernelConfig, prefix: str) -> dict[str, float]:
    """Every relaxation scalar the model reads, narrowed to ``float`` and range-checked.

    One validator for the model, shared by the Python entry point and by
    any kernel that reads the configuration outside its loop. ``prefix``
    names the calling entry point in the message, because a shared
    boundary is a worse place for an ambiguous error than a named one.

    All three values must be positive and finite: a zero or negative
    length makes the exponent's time constant nonpositive, and a zero or
    negative floor rethrows the stopped-wheel divide it exists to prevent.
    """
    names = (
        "relaxation_length_longitudinal_m",
        "relaxation_length_lateral_m",
        "relaxation_min_speed_m_s",
    )
    values = {}
    for name in names:
        value = _checked_float(f"config.{name}", getattr(config, name), prefix=prefix)
        if value <= 0.0:
            raise ValueError(
                f"{prefix}: config.{name} must be finite and > 0, got {value!r}. "
                "The relaxation lengths set the lag in exp(-v dt / L) and the "
                "floor keeps a stopped patch finite; a zero or negative one makes "
                "the time constant infinite, the exponent negative, or rethrows "
                "the zero divide the floor exists to prevent"
            )
        values[name] = value
    return values


def step_relaxation(
    config: KernelConfig,
    slip_ratio_state: np.ndarray,
    slip_ratio_target: np.ndarray,
    slip_angle_state: np.ndarray,
    slip_angle_target: np.ndarray,
    patch_speed_m_s: np.ndarray,
    dt_s: float,
) -> None:
    """One fixed step of relaxation for both slip axes, checked before it runs.

    The Python-facing composition. A caller inside a kernel does not come
    through here - it reads the configuration outside its loop through
    :func:`validated_relaxation_scalars` and calls :func:`relax_slip_state`
    directly, because this function's checks are Python and a compiled
    loop cannot make them.

    All five vectors are caller-owned ``float64`` C-contiguous length-four
    arrays; the two state arrays are updated in place and the targets and
    speeds are read only. ``dt_s`` must be finite and nonnegative - the
    production integrator is the fixed 100 us step, so a negative or
    nonfinite step here is a caller bug rather than a different model.
    Every scalar is narrowed to ``float`` so that equivalent ``int`` and
    ``float`` callers share one compiled specialisation, and every value
    is refused - rather than coerced or clamped - when it is nonfinite or
    in the wrong shape: a NaN that reaches an integrator is a run that
    looks like it finished.
    """
    values = validated_relaxation_scalars(config, "step_relaxation")
    dt = _checked_float("dt_s", dt_s, prefix="step_relaxation")
    if dt < 0.0:
        raise ValueError(f"step_relaxation: dt_s must be >= 0, got {dt!r}")
    for label, value in (
        ("slip_ratio_state", slip_ratio_state),
        ("slip_ratio_target", slip_ratio_target),
        ("slip_angle_state", slip_angle_state),
        ("slip_angle_target", slip_angle_target),
        ("patch_speed_m_s", patch_speed_m_s),
    ):
        _checked_wheel_vector(label, value, prefix="step_relaxation")
        if not np.isfinite(value).all():
            raise ValueError(f"step_relaxation: {label} must be finite, got {list(value)}")
    relax_slip_state(
        slip_ratio_state,
        slip_ratio_target,
        slip_angle_state,
        slip_angle_target,
        patch_speed_m_s,
        values["relaxation_length_longitudinal_m"],
        values["relaxation_length_lateral_m"],
        dt,
        values["relaxation_min_speed_m_s"],
    )


def _checked_wheel_vector(label: str, value: object, *, prefix: str) -> np.ndarray:
    """A per-wheel vector: length four, C-contiguous ``float64``.

    A length-three or length-five vector, a ``float32`` one, a nested one,
    or not an array at all are all refused: with bounds checking off, a
    wrong length is an out-of-bounds read rather than an error, and the
    state the caller thinks it relaxed is not the state that was written.
    Tests match on the substring ``float64`` for the dtype and shape
    refusals, so the message says it.
    """
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.float64
        or value.ndim != 1
        or value.size != _WHEEL_COUNT
        or not value.flags.c_contiguous
    ):
        raise ValueError(
            f"{prefix}: {label} must be a C-contiguous float64 vector of length "
            f"{_WHEEL_COUNT}, got {type(value).__name__} of shape "
            f"{getattr(value, 'shape', None)} and dtype {getattr(value, 'dtype', None)}"
        )
    return value


def _checked_float(label: str, value: object, *, prefix: str = "step_relaxation") -> float:
    """Return a finite real scalar as ``float``, or fail before it reaches Numba.

    ``bool`` is refused although it is an ``int``, because ``True`` as a
    time step is a caller bug rather than a number. This mirrors
    ``tyres._checked_float`` and ``loads._checked_float`` rather than
    sharing either: the modules validate different field sets for
    different models, and a shared private helper between them would be a
    coupling for the sake of six lines.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{prefix}: {label} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{prefix}: {label} must be finite, got {value!r}")
    return number
