"""P1-T3/P1-T6: aerodynamic and longitudinal tyre forces, as flat numeric Numba functions.

Two models, one module, because they are two halves of the same step and are documented against
the same sign conventions.

**Aerodynamics** is ``PLAN.md`` section 5.1 written out: dynamic pressure ``q = 1/2 rho v^2``,
downforce ``Fz_aero = q A Cl(v)`` and drag ``Fdrag = -q A Cd(v)`` signed along +x. ``Cl`` and
``Cd`` are *functions of speed* read from ``car_spec.yaml``, evaluated by linear interpolation
on the shared speed axis and held flat outside it. That interpolation is the decision Task 1
deliberately left open; see :func:`speed_curve` for why linear and clamped.

**The longitudinal tyre force** is the Pacejka Magic Formula behind a guarded slip ratio,
``kappa = (omega r - v) / max(v, eps)``. Three things about it are decisions rather than
formulae, and each is argued where it is implemented:

* :func:`tyre_longitudinal_force` returns exactly zero for a patch carrying no load, including a
  *negative* load - a load that has gone negative would otherwise invert the sign of the whole
  expression and push the car the wrong way from a tyre that is carrying nothing.
* The grip limit ``|Fx| <= mu Fz`` is a property of the Magic Formula, not a clamp added to it.
  A clamp that the formula can never reach would be untestable dead code, so the bound is
  measured across a slip sweep instead - and the peak is required to *reach* ``mu Fz``, so the
  measurement cannot pass on a formula that returns a constant fraction of it.
* The P1 force is linear in ``Fz`` (``D = mu Fz``), with no load sensitivity. PLAN.md section 4
  says an F1 tyre without load sensitivity understates high-speed downforce badly, and that is
  true - but load sensitivity is P2-T3's work, and a silent non-proportional model added before
  then would be harder to notice than an explicit one.

**Two ways in, deliberately.** The primitives take flat scalars and ``float64`` arrays and are
what a compiled kernel calls: ``longitudinal.simulate`` reads ``dt_s`` and ``mass_kg`` out of the
config once, in Python, and the loop is handed numbers, and the same rule applies here.
:func:`step_forces` is the Python-facing composition of the three models for one step, and the
one place the configuration and the state are checked before any arithmetic happens. It is not a
hot-path call: a kernel reads the config outside its loop and calls the primitives.

**No state, no allocation, no clock.** Every function is a pure function of its arguments, so
the same inputs give the same bits, and a P1 run's determinism claim does not have to be
re-earned here.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import numpy as np
from numba import njit

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import KernelConfig

__all__ = [
    "aero_forces",
    "dynamic_pressure_pa",
    "slip_ratio",
    "speed_curve",
    "step_forces",
    "tyre_longitudinal_force",
]

# The scalars `step_forces` refuses to divide or multiply by, and the sign each one must have.
# `pacejka_e` is the odd one out: the curvature factor carries a sign in the Magic Formula, so
# it is only required to be finite. Read as data rather than as a list of names because a name
# that is not checked here is a name nobody notices is unchecked.
_POSITIVE_CONFIG_SCALARS: Final[tuple[str, ...]] = (
    "air_density_kg_m3",
    "reference_area_m2",
    "pacejka_b",
    "pacejka_c",
    "pacejka_mu",
    "slip_ratio_min_speed_m_s",
)
_FINITE_CONFIG_SCALARS: Final[tuple[str, ...]] = ("pacejka_e",)

# The njit options are written out on every function rather than shared through one dict, for the
# same reason `tests/test_longitudinal_kernel.py` reads them off the dispatcher: an indirection
# between the reader and the options is one more place for them to be wrong, and these are the
# options the project's determinism claim rests on.


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def dynamic_pressure_pa(speed_m_s: float, air_density_kg_m3: float) -> float:
    """``q = 1/2 rho v^2``, the pressure every aerodynamic force in this module is built on.

    Zero at rest, and exactly quadratic in speed for every other value - which is the only
    property it claims, and the reason the speed dependence of the *forces* has to come from
    ``Cl(v)`` and ``Cd(v)`` rather than from this function (PLAN.md section 5.1).
    """
    return 0.5 * air_density_kg_m3 * speed_m_s * speed_m_s


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def speed_curve(speed_m_s: float, breakpoints_m_s: np.ndarray, values: np.ndarray) -> float:
    """Piecewise-linear lookup on a speed-indexed curve, held flat outside its own range.

    **Linear between knots, clamped at both ends.** Task 1 froze one shared speed axis for ``Cl``
    and ``Cd`` and deliberately left the interpolation undecided; this is that decision.

    * *Linear* because a table of six points is a table, not a fit. A spline through knots the
      project itself labelled "order of magnitude" would invent curvature the data does not
      have, and would be harder to read back than the arithmetic that produced it. It is also
      the arithmetic ``tests/test_car_spec.py`` already uses to sample the ERS curves, so the
      kernel and the contract tests interpolate identically rather than approximately.
    * *Clamped* because the top knot is 105 m/s and a car can be dragged past it. Holding the
      end value is finite, monotone and conservative; extending the last segment would make the
      drag of a fast car depend on a two-point line, which is the kind of silent extrapolation
      that shows up as a plausible-looking top speed.

    Both arrays must be the same length, non-empty, and strictly increasing in speed - which
    :func:`~f1telemetry.contracts.car_spec.KernelConfig` guarantees for ``cl``/``cd`` against the
    shared ``aero_speed_m_s`` axis, and :func:`step_forces` re-checks.
    """
    last = breakpoints_m_s.size - 1
    if speed_m_s <= breakpoints_m_s[0]:
        return values[0]
    if speed_m_s >= breakpoints_m_s[last]:
        return values[last]
    index = 0
    while speed_m_s > breakpoints_m_s[index + 1]:
        index += 1
    span = breakpoints_m_s[index + 1] - breakpoints_m_s[index]
    weight = (speed_m_s - breakpoints_m_s[index]) / span
    return values[index] * (1.0 - weight) + values[index + 1] * weight


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def aero_forces(
    speed_m_s: float,
    air_density_kg_m3: float,
    reference_area_m2: float,
    speed_axis_m_s: np.ndarray,
    cl: np.ndarray,
    cd: np.ndarray,
) -> tuple[float, float]:
    """``(downforce, drag)`` in newtons at ``speed_m_s``, signed along +x.

    ``downforce = 1/2 rho v^2 A Cl(v)`` and ``drag = -1/2 rho v^2 A Cd(v)``, with ``Cl`` and
    ``Cd`` evaluated by :func:`speed_curve` on the shared speed axis. The returned downforce is
    a positive magnitude to be *added* to a contact patch's vertical load; the returned drag is a
    signed force, negative when the car is moving forward and positive when it is moving
    backwards, because the drag always opposes the direction of travel. That is the only
    difference between the two halves of the return value, and it is why the caller cannot
    simply add them to different things without reading this.

    Downforce is even in speed and drag is odd in it, both as ``PLAN.md`` section 5.1 and
    ``GroundTruthStep.downforce_n`` / ``.drag_w`` require. That symmetry is why the coefficients
    are looked up at ``abs(speed_m_s)``: the curves are tabulated for forward speeds, and a car
    rolling backwards at 80 m/s meets the same air at the same dynamic pressure as one rolling
    forwards, so reading the curve at -80 m/s would take the below-range end value and give it
    the drag of a stationary car. ``speed_axis_m_s``, ``cl`` and ``cd`` share a length, which the
    configuration boundary enforces.
    """
    pressure = dynamic_pressure_pa(abs(speed_m_s), air_density_kg_m3)
    downforce_n = pressure * reference_area_m2 * speed_curve(abs(speed_m_s), speed_axis_m_s, cl)
    drag_n = -pressure * reference_area_m2 * speed_curve(abs(speed_m_s), speed_axis_m_s, cd)
    if speed_m_s < 0.0:
        drag_n = -drag_n
    return downforce_n, drag_n


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def slip_ratio(wheel_speed_m_s: float, speed_m_s: float, min_speed_m_s: float) -> float:
    """``kappa = (omega r - v) / max(v, eps)``, the longitudinal slip ratio.

    Both speeds are circumferential: ``wheel_speed_m_s`` is already ``omega`` times the rolling
    radius, which is the product Task 4 builds when it owns the wheel state, and this function
    is handed the product rather than a wheel speed it would have to multiply out.

    The ``max(v, eps)`` guard is the whole point of the function and the reason ``eps`` is a
    configured number rather than a literal: at a standing start the unguarded denominator is
    zero, and a sliding wheel at rest is not a hypothetical state - it is a launch, or a locked
    wheel. ``eps`` is in ``car_spec.yaml``, so the low-speed behaviour is calibration data.

    The slip ratio is positive in drive (wheel faster than road) and negative under braking, which
    is the convention :class:`~f1telemetry.testing.records.GroundTruthStep` documents and
    invariant 4 checks. A consequence worth knowing before Task 4 models a launch: with the
    denominator pinned at ``eps``, a launch sits at a *large* slip ratio, on the falling branch
    of the Magic Formula rather than the part of the curve that rises.
    """
    denominator = speed_m_s if speed_m_s > min_speed_m_s else min_speed_m_s
    return (wheel_speed_m_s - speed_m_s) / denominator


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def _magic_formula(slip: float, stiffness: float, shape: float, curvature: float) -> float:
    """The Magic Formula normalised by its own peak: ``sin(C atan(Bx - E(Bx - atan Bx)))``.

    Private, and separate from :func:`tyre_longitudinal_force` for the reason the load guard is
    in that function rather than here: the normalised shape is meaningless without a peak
    ``D``, so the only safe way to expose the formula is with ``D = mu Fz`` already applied. A
    caller that reached for this would get a dimensionless number in [-1, 1] and would still have
    to remember the guard.
    """
    scaled = stiffness * slip
    return math.sin(shape * math.atan(scaled - curvature * (scaled - math.atan(scaled))))


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def tyre_longitudinal_force(
    slip: float,
    load_n: float,
    stiffness: float,
    shape: float,
    curvature: float,
    friction: float,
) -> float:
    """``Fx = mu Fz sin(C atan(B kappa - E(B kappa - atan(B kappa))))``, in newtons.

    ``D = mu Fz`` with no load sensitivity, which is the P1 simplification PLAN.md section 4
    warns about and P2-T3 corrects: a constant-``mu`` tyre understates high-speed downforce. It
    is stated here as a linear model rather than left implicit so that P2-T3's change is a
    visible edit to this function and its tests.

    **The load guard.** A patch carrying no load returns exactly ``0.0``, and so does one
    carrying a negative load. Zero load is a wheel in the air; a negative load is a solver or a
    fixture that has gone wrong, and it is exactly the case where the unguarded expression
    inverts - ``D = mu Fz`` would be negative, the force would come out reversed, and the car
    would be pushed the wrong way by a tyre that is carrying nothing. The error would surface
    several steps later looking like a traction problem.

    **The grip limit is the formula's, not a clamp's.** ``|sin(...)| <= 1`` bounds the force at
    ``mu Fz``, so adding a clamp would add a branch that can never fire - untestable dead code,
    and a second place for the grip limit to be wrong. The bound is therefore asserted over a
    slip sweep, with the peak required to reach it, rather than enforced here.

    ``stiffness``, ``shape`` and ``friction`` are magnitudes and ``curvature`` carries a sign;
    all four come from ``car_spec.yaml``, and :func:`step_forces` refuses a configuration that
    does not satisfy that.
    """
    if load_n <= 0.0:
        return 0.0
    peak_n = friction * load_n
    return peak_n * _magic_formula(slip, stiffness, shape, curvature)


def step_forces(
    config: KernelConfig,
    speed_m_s: float,
    wheel_speed_m_s: float,
    load_n: float,
) -> tuple[float, float, float]:
    """``(downforce_n, drag_n, tyre_fx_n)`` for one step, from a validated configuration.

    The Python-facing composition of the three models, and the single place where the
    configuration and the state are checked against each other before any arithmetic happens.
    A caller inside a kernel does not come through here - it reads the configuration outside its
    loop, exactly as :func:`~f1telemetry.kernels.longitudinal.simulate` reads ``dt_s`` and
    ``mass_kg``, and calls the primitives directly, because this function's checks are Python and
    a compiled loop cannot make them.

    ``load_n`` is the vertical load *the contact patch carries*: the caller owns how downforce
    is shared between the axles, which is P2-T2's static split and not this task's. Passing
    ``static_load + downforce_n`` is the straight-line approximation; the sum of the corner loads
    then satisfies invariant 3 by construction.

    The checks are the ones whose absence produces NaNs rather than errors: a nonfinite state
    value, a zero or nonfinite air density or reference area, a shape factor or friction
    coefficient of zero (which would make the tyre force identically zero whatever the slip), and
    a zero or nonfinite ``slip_ratio_min_speed_m_s`` - the divide-by-zero this whole guard
    exists to prevent, reachable through a hand-built or replaced
    :class:`~f1telemetry.contracts.car_spec.KernelConfig` that never went through the loader.
    """
    _check_state("speed_m_s", speed_m_s)
    _check_state("wheel_speed_m_s", wheel_speed_m_s)
    _check_state("load_n", load_n)
    for name in _POSITIVE_CONFIG_SCALARS:
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"step_forces: config.{name} must be finite and > 0, got {value!r}. It is a "
                "multiplier or a denominator, so nothing downstream can recover from it"
            )
    for name in _FINITE_CONFIG_SCALARS:
        value = getattr(config, name)
        if not math.isfinite(value):
            raise ValueError(
                f"step_forces: config.{name} must be finite, got {value!r}. The curvature "
                "factor carries a sign, but a nonfinite one is a NaN in the force"
            )
    _check_aero_arrays(config)
    downforce_n, drag_n = aero_forces(
        speed_m_s,
        config.air_density_kg_m3,
        config.reference_area_m2,
        config.aero_speed_m_s,
        config.cl,
        config.cd,
    )
    slip = slip_ratio(wheel_speed_m_s, speed_m_s, config.slip_ratio_min_speed_m_s)
    tyre_fx_n = tyre_longitudinal_force(
        slip,
        load_n,
        config.pacejka_b,
        config.pacejka_c,
        config.pacejka_e,
        config.pacejka_mu,
    )
    return downforce_n, drag_n, tyre_fx_n


def _check_state(name: str, value: float) -> None:
    """Refuse a state value that would make a force NaN rather than wrong."""
    if not math.isfinite(value):
        raise ValueError(
            f"step_forces: {name} must be finite, got {value!r}. A NaN or an infinity here "
            "propagates into the integrator, where it looks like a run that finished"
        )


def _check_aero_arrays(config: KernelConfig) -> None:
    """Refuse a curve set the interpolated lookup would read out of bounds.

    The loader already checks that ``cl`` and ``cd`` share the speed axis, that the axis is
    strictly increasing, and that every coefficient is positive and finite. This repeats the
    parts that make :func:`speed_curve` safe, because ``KernelConfig`` is a public frozen
    dataclass that ``dataclasses.replace`` can put a hand-built array into, and an
    out-of-bounds read there is silent - the same reason ``longitudinal.simulate`` re-checks
    ``dt_s`` and ``mass_kg`` rather than trusting the loader.
    """
    axis = config.aero_speed_m_s
    if axis.ndim != 1 or axis.size == 0:
        raise ValueError(
            "step_forces: config.aero_speed_m_s must be a non-empty 1-D array, got shape "
            f"{axis.shape}"
        )
    if axis.shape != config.cl.shape or axis.shape != config.cd.shape:
        raise ValueError(
            "step_forces: config.aero_speed_m_s, config.cl and config.cd must share a length, "
            f"got {axis.shape}, {config.cl.shape} and {config.cd.shape}. speed_curve indexes "
            "both curves with the one axis."
        )
    for index in range(axis.size):
        if not math.isfinite(float(axis[index])) or not math.isfinite(float(config.cl[index])):
            raise ValueError(
                f"step_forces: config.aero_speed_m_s[{index}] and config.cl[{index}] must be finite"
            )
        if not math.isfinite(float(config.cd[index])):
            raise ValueError(f"step_forces: config.cd[{index}] must be finite")
        if index and float(axis[index]) <= float(axis[index - 1]):
            raise ValueError(
                f"step_forces: config.aero_speed_m_s must be strictly increasing, but "
                f"[{index}] is {float(axis[index])} after {float(axis[index - 1])}. A repeated "
                "speed would divide by zero in speed_curve."
            )
