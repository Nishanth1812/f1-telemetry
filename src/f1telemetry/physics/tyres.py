"""P2-T3: per-wheel steady lateral tyre response - the lateral Magic Formula.

``PLAN.md`` section 4 gives the lateral model its own Pacejka set, with load
sensitivity on ``D`` and ``B`` and a camber response, and ``PHASES.md`` P2-T3
asks for exactly that and nothing more. This module is that task's work and is
deliberately nothing else: no combined-slip ellipse (Task 4), no relaxation
state (Task 5), no chassis integration, no per-corner geometry. It answers one
question - what steady lateral force does one contact patch produce for its
slip angle, camber angle and vertical load - and it answers it as a pure
function of those three inputs and the validated configuration.

**The formula**, with every term dimensioned in the units the configuration
    and the channels already use (input angles in degrees, converted to radians
    for the Magic Formula; forces and loads in newtons)::

    Fy = D(Fz) sin(C atan(B_eff radians(alpha + alpha_gamma)
                           - E (B_eff radians(alpha + alpha_gamma) - atan(...))))

    alpha_gamma = K_camber gamma / C_alpha_ref      deg, the camber's worth of slip
    C_alpha_ref = mu Fz_ref B C pi/180              N/deg, the reference cornering stiffness
    D(Fz)       = mu_eff(Fz) Fz                     N, the available peak
    mu_eff      = mu max(0, 1 - s_peak (Fz/Fz_ref - 1))
    B_eff       = B  max(0, 1 - s_stiff (Fz/Fz_ref - 1))

Five decisions are written out here rather than left implicit:

* **Camber enters as an equivalent slip angle, not as an added force.** The
  configured ``camber_stiffness_n_per_deg`` is a force per degree, and an
  additive camber force would push the total past the friction peak whenever
  slip and camber peak together. The reference cornering stiffness
  ``mu Fz_ref B C pi/180`` - the Magic Formula's initial slope per degree at the reference
  load, where both sensitivities are exactly one - converts the configured
  newtons-per-degree into the slip angle that would buy the same force, so
  the camber walks the *same* curve a slip angle walks: it rises to the peak
  and falls back to the curve's asymptote. The configured stiffness is
  realised exactly in the linear range at the reference load
  (``Fy(0, gamma) = K gamma`` there), it scales with the load-sensitive
  cornering stiffness away from it, and it saturates at the same peak as
  slip. The bound ``|Fy| <= mu_eff(Fz) Fz`` is then the formula's own
  property - ``|sin| <= 1`` - rather than a clamp behind the formula, which
  is the same reason P1's longitudinal grip limit is not a clamp either.
* **Load sensitivity is a linear fall per unit of normalised load, clamped at
  zero.** ``mu_eff`` and ``B_eff`` each fall by their configured fraction for
  every reference load of vertical load above ``Fz_ref``, and rise below it -
  lighter-loaded corners grip more per newton, which is the effect PLAN.md
  section 4 says a constant-``mu`` tyre misses. The linear form would reach
  zero at ``(1/s + 1) Fz_ref`` and reverse beyond it, so the effective
  values are clamped to nonnegative bounds: past that load the tyre is
  modelled as having *no* lateral capacity rather than a negative one, which
  is the safe direction - a reversed response at absurd load would push the
  car the wrong way from a tyre that is still carrying load.
* **The load guard is in the primitive, not the caller.** A patch carrying no
  load returns exactly ``0.0``, and so does one carrying a negative load. Zero
  load is a wheel in the air; a negative load is a solver or a fixture that
  has gone wrong, and it is exactly the case where an unguarded ``D`` would
  invert the sign of the whole expression. The error would surface several
  steps later looking like a grip problem.
* **The shape and curvature factors are bounded so the contract is
  unconditional.** The response is sign-preserving at *every* finite slip
  angle - positive slip angle and positive camber produce positive leftward
  ``Fy``, with no reversal past the peak - exactly when the shape factor is
  in ``(0, 2]`` and the curvature factor is below 1. Those are the conditions
  the Magic Formula's own algebra needs: a shape factor above 2 lets
  ``C atan(...)`` cross ``pi`` and the curve cross zero, and a curvature
  factor at or above 1 keeps the curve from ever reaching its peak. The
  boundary refuses anything else, so the sign contract holds for every
  configuration the model accepts rather than only for the committed one.
* **Two ways in, deliberately.** The primitives take flat scalars and are
  what a compiled kernel calls; :func:`step_lateral_force` is the Python-facing
  composition that checks the configuration and the state before any
  arithmetic happens. The kernel reads the configuration once, outside its
  loop, through :func:`validated_lateral_scalars`, and hands the primitives
  numbers - ``KernelConfig`` is a public frozen dataclass that
  ``dataclasses.replace`` can make inconsistent, and a ``boundscheck=False``
  function cannot defend itself.

**No state, no allocation, no clock, no claims.** Every function is a pure
function of its arguments, so the same inputs give the same bits. Nothing here
reads a regulation: the lateral coefficient set, the load sensitivity and the
camber stiffness are synthesised placeholders in ``car_spec.yaml``, labelled
as such in the file itself, and this module treats them as data to be
calibrated rather than as numbers with any authority behind them. Steady-state
means steady-state: a transient slip input is answered with its steady force,
which is the task's scope and Task 4's and Task 5's to widen.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

from numba import njit

from .pacejka import magic_formula_shape  # noqa: TID251 -- shared physics primitive

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import KernelConfig

__all__ = [
    "camber_equivalent_slip_deg",
    "lateral_peak_friction",
    "lateral_slip_stiffness",
    "load_sensitivity_multiplier",
    "reference_cornering_stiffness_n_per_deg",
    "step_lateral_force",
    "tyre_lateral_force",
    "validated_lateral_scalars",
]

# The shape factor C is bounded above by 2 and the curvature factor E below 1
# because those are the conditions under which the Magic Formula's own shape
# stays sign-preserving and reaches its peak at a finite slip angle. With
# C > 2 the argument C atan(...) can cross pi and the curve crosses zero into
# a reversed response; with E >= 1 the curve's inner term saturates below the
# value the peak needs, so the peak is never reached. Both bounds are the
# model's contract made checkable, not tuned coefficients.
_SHAPE_FACTOR_MIN: Final[float] = 1.0
_SHAPE_FACTOR_MAX: Final[float] = 2.0
_CURVATURE_FACTOR_MAX: Final[float] = 1.0
_RAD_PER_DEG: Final[float] = math.pi / 180.0

# The scalars the lateral model reads and the sign each one has to have, read
# as data rather than as a list of names because a name that is not checked
# here is a name nobody notices is unchecked. `lateral_pacejka_e` is the odd
# one out: the curvature factor carries a sign in the Magic Formula, so it is
# required to be finite and *below* 1 rather than positive. The two
# sensitivities are fractions of a fall per unit of normalised load, so the
# interesting boundary is 1 as well as 0: a sensitivity of exactly 1 would
# make the load-sensitive tyre reach zero grip at twice the reference load,
# and a negative one would make grip *rise* with load.
_LATERAL_POSITIVE_CONFIG_SCALARS: Final[tuple[str, ...]] = (
    "lateral_pacejka_b",
    "lateral_pacejka_mu",
    "load_sensitivity_reference_n",
    "camber_stiffness_n_per_deg",
)
_LATERAL_FINITE_CONFIG_SCALARS: Final[tuple[str, ...]] = ("lateral_pacejka_e",)
_LATERAL_SENSITIVITY_CONFIG_SCALARS: Final[tuple[str, ...]] = (
    "load_sensitivity_peak",
    "load_sensitivity_stiffness",
)

# The njit options are written out on every function rather than shared
# through one dict, for the same reason `tests/test_tyres.py` reads them off
# the dispatcher: an indirection between the reader and the options is one
# more place for them to be wrong, and these are the options the project's
# determinism claim rests on.


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def load_sensitivity_multiplier(
    load_n: float,
    reference_load_n: float,
    sensitivity: float,
) -> float:
    """``max(0, 1 - s (Fz/Fz_ref - 1))``: the fraction of grip load still buys.

    The reference load is where the multiplier is exactly one, so a corner at
    the reference load keeps every newton of its grip. Above the reference the
    multiplier falls by the configured fraction per reference load of extra
    load - grip is *not* proportional to load, which is the effect PLAN.md
    section 4 says a constant-``mu`` tyre misses. Below the reference it rises
    the same way, and it is bounded there by ``1 + sensitivity`` because a
    load cannot go below zero.

    The clamp at zero is the safe direction. The linear fall would reach zero
    at ``(1/s + 1) Fz_ref`` and go negative beyond it, which would model a
    tyre that pushes the car the *wrong way* under an absurd load; clamped,
    the same tyre is modelled as having no lateral capacity left. A sensitivity
    of zero is the P1 model - no load dependence at all - so this function
    degenerates to the constant one rather than to a special case.

    ``reference_load_n`` is a divisor and is not guarded here: a zero or
    negative reference load is a configuration the boundary refuses before any
    of this is called.
    """
    multiplier = 1.0 - sensitivity * (load_n / reference_load_n - 1.0)
    if multiplier < 0.0:
        return 0.0
    return multiplier


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def lateral_peak_friction(
    load_n: float,
    friction: float,
    reference_load_n: float,
    peak_sensitivity: float,
) -> float:
    """``mu_eff(Fz)``: the load-sensitive peak friction coefficient.

    ``friction`` is the configured peak friction at the reference load and
    ``peak_sensitivity`` is the configured fractional fall per unit of
    normalised load above it, so this is the ``mu`` of ``D = mu_eff Fz`` -
    the load sensitivity PLAN.md section 4 puts on ``D``. At the reference
    load it returns ``friction`` exactly; at twice the reference load,
    ``friction (1 - s)``.
    """
    return friction * load_sensitivity_multiplier(load_n, reference_load_n, peak_sensitivity)


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def lateral_slip_stiffness(
    load_n: float,
    stiffness: float,
    reference_load_n: float,
    stiffness_sensitivity: float,
) -> float:
    """``B_eff(Fz)``: the load-sensitive slip stiffness.

    The second half of the load sensitivity, the one PLAN.md section 4 puts on
    ``B``: the curve's initial slope falls with load by the configured
    fraction per unit of normalised load, so a heavily loaded corner not only
    peaks lower, it peaks *later* - the whole curve is softer, not just
    scaled down. At the reference load it returns ``stiffness`` exactly.
    """
    return stiffness * load_sensitivity_multiplier(load_n, reference_load_n, stiffness_sensitivity)


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def reference_cornering_stiffness_n_per_deg(
    friction: float,
    stiffness: float,
    shape: float,
    reference_load_n: float,
) -> float:
    """Reference-load Magic Formula slope in N/deg, with its input angle in radians.

    For small slip angles the Magic Formula is linear, ``Fy = D C B alpha``,
    so its slope at the origin is the peak force times the shape and stiffness
    factors. At the reference load both sensitivities are exactly one, which
    makes this the cornering stiffness the configuration describes *at* the
    reference load - the place where the configured camber stiffness is
    defined to be realised. It is the denominator of the camber equivalent
    slip angle, and it is derived from configured values alone: it is not a
    second coefficient and no new number enters the model through it.
    """
    return friction * reference_load_n * stiffness * shape * _RAD_PER_DEG


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def camber_equivalent_slip_deg(
    camber_deg: float,
    camber_stiffness_n_per_deg: float,
    cornering_stiffness_n_per_deg: float,
) -> float:
    """``K gamma / C_alpha_ref`` in degrees: the slip angle a camber angle is worth.

    Dimensionally, ``camber_stiffness_n_per_deg`` times a camber angle in
    degrees is a force in newtons, and dividing that force by a cornering
    stiffness in newtons per degree leaves a slip angle in degrees. That slip
    angle is what the camber contributes to the Magic Formula's argument, so
    the camber response saturates with the same peak as the slip response
    rather than adding force on top of it - which is what keeps the total
    lateral force inside the available peak by construction.

    ``cornering_stiffness_n_per_deg`` is a divisor and is not guarded here:
    it is built from validated-positive configuration values, and the
    boundary that can refuse a zero one is :func:`step_lateral_force`.
    """
    return camber_stiffness_n_per_deg * camber_deg / cornering_stiffness_n_per_deg


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def _magic_formula(slip: float, stiffness: float, shape: float, curvature: float) -> float:
    """The Magic Formula normalised by its own peak: ``sin(C atan(Bx - E(Bx - atan Bx)))``.

    Private, and separate from :func:`tyre_lateral_force` for the reason the
    load guard is in that function rather than here: the normalised shape is
    meaningless without a peak ``D``, so the only safe way to expose the
    formula is with ``D = mu_eff(Fz) Fz`` and the load guard already applied.
    A caller that reached for this would get a dimensionless number in
    ``[-1, 1]`` and would still have to remember both.

    The shape is odd in its argument, bounded by one in magnitude for every
    finite input, and - for a shape factor in ``(0, 2]`` and a curvature
    factor below 1, which :func:`validated_lateral_scalars` enforces -
    sign-preserving, so a positive equivalent slip angle is a positive force
    at every finite slip angle rather than only near the origin.
    """
    return magic_formula_shape(stiffness * slip, shape, curvature)


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def tyre_lateral_force(
    slip_angle_deg: float,
    camber_deg: float,
    load_n: float,
    stiffness: float,
    shape: float,
    curvature: float,
    friction: float,
    reference_load_n: float,
    peak_sensitivity: float,
    stiffness_sensitivity: float,
    camber_stiffness_n_per_deg: float,
) -> float:
    """``Fy`` for one contact patch, in newtons: the steady lateral Magic Formula.

    The formula is the one in the module docstring, in the order the reference
    in ``tests/test_tyres.py`` transcribes it:

    1. the load-sensitive, clamped peak friction and slip stiffness, and the
       peak force ``D = mu_eff(Fz) Fz`` they produce;
    2. the reference cornering stiffness ``mu Fz_ref B C pi/180``, and the camber
       angle's worth of equivalent slip bought with the configured camber
       stiffness over it;
    3. the Magic Formula of the slip angle plus that equivalent slip angle,
       scaled by the load-sensitive stiffness, and peaked by ``D``.

    **The load guard.** A patch carrying no load returns exactly ``0.0``, and
    so does one carrying a negative load. Zero load is a wheel in the air; a
    negative load is a solver or a fixture that has gone wrong, and it is
    exactly the case where the unguarded expression inverts - ``D`` would be
    negative, the force would come out reversed, and the car would be pushed
    the wrong way by a tyre that is carrying nothing.

    **The bound is the formula's.** ``|sin| <= 1`` bounds the force at
    ``mu_eff(Fz) Fz`` for every combination of slip angle, camber angle and
    load, because the camber enters through the same shape function rather
    than around it. Adding a clamp would add a branch that can never fire -
    untestable dead code, and a second place for the grip limit to be wrong -
    so the bound is asserted over a sweep in the tests instead, with the peak
    required to reach it.

    ``stiffness``, ``shape``, ``friction``, ``reference_load_n`` and
    ``camber_stiffness_n_per_deg`` are magnitudes, ``curvature`` carries a
    sign, and the two sensitivities are fractions; all of them come from
    ``car_spec.yaml``, and :func:`step_lateral_force` refuses a configuration
    that does not satisfy that. Channel and camber inputs are degrees; slip is
    converted to radians for the dimensionless ``B`` factor.
    """
    if load_n <= 0.0:
        return 0.0
    peak_n = lateral_peak_friction(load_n, friction, reference_load_n, peak_sensitivity) * load_n
    cornering_n_per_deg = reference_cornering_stiffness_n_per_deg(
        friction, stiffness, shape, reference_load_n
    )
    equivalent_slip_deg = slip_angle_deg + camber_equivalent_slip_deg(
        camber_deg, camber_stiffness_n_per_deg, cornering_n_per_deg
    )
    effective_stiffness = lateral_slip_stiffness(
        load_n, stiffness, reference_load_n, stiffness_sensitivity
    )
    return peak_n * _magic_formula(
        equivalent_slip_deg * _RAD_PER_DEG, effective_stiffness, shape, curvature
    )


def validated_lateral_scalars(config: KernelConfig, prefix: str) -> dict[str, float]:
    """Every lateral scalar the model reads, narrowed to ``float`` and range-checked.

    **One validator for the model, shared by the Python entry point and by the
    kernel.** :func:`step_lateral_force` needs this, and a compiled kernel
    needs it called once outside its loop, which is what ``PLAN.md`` section
    4.1 rule 3 requires: the compiled loop is handed numbers and never sees
    the config object. ``prefix`` names the calling entry point in the
    message, because a shared boundary is a worse place for an ambiguous error
    than a named one.

    The sign rules are the ones a compiled function cannot make for itself once
    ``boundscheck`` is off, and they are the conditions the module's contract
    needs rather than a copy of the loader's rules:

    * magnitudes strictly positive - ``b``, ``mu``, the reference load and the
      camber stiffness; a zero or negative one is a zero denominator, a
      reversed sign or a tyre with no grip to divide out;
    * the shape factor ``c`` in ``(0, 2]`` - above 2 the Magic Formula's
      argument can cross ``pi`` and the response reverses at large slip
      angles, which would break the sign contract the module promises;
    * the curvature factor ``e`` finite and below 1 - at or above 1 the curve
      never reaches its peak, so the grip limit the tests measure would be
      out of reach and the model would quietly under-report its own capacity;
    * the two sensitivities in ``[0, 1)`` - a fraction of a fall per unit of
      normalised load, where 1 would reach zero grip at twice the reference
      load and a negative value would make grip rise with load.

    It is deliberately *wider* than any one caller's read, because
    ``KernelConfig`` is a public frozen dataclass that ``dataclasses.replace``
    can make inconsistent, and a boundary that checked only what this call
    happened to touch would let a bad value sit in the same object the next
    call reads.
    """
    names = (
        *_LATERAL_POSITIVE_CONFIG_SCALARS,
        *_LATERAL_FINITE_CONFIG_SCALARS,
        "lateral_pacejka_c",
        *_LATERAL_SENSITIVITY_CONFIG_SCALARS,
    )
    values = {
        name: _checked_float(f"config.{name}", getattr(config, name), prefix=prefix)
        for name in names
    }
    for name in _LATERAL_POSITIVE_CONFIG_SCALARS:
        if values[name] <= 0.0:
            raise ValueError(
                f"{prefix}: config.{name} must be finite and > 0, got {values[name]!r}"
            )
    shape = values["lateral_pacejka_c"]
    if not _SHAPE_FACTOR_MIN < shape <= _SHAPE_FACTOR_MAX:
        raise ValueError(
            f"{prefix}: config.lateral_pacejka_c must be in "
            f"({_SHAPE_FACTOR_MIN}, {_SHAPE_FACTOR_MAX}], got {shape!r}. "
            "The combined-slip peak normalization needs a finite, reachable peak; "
            "above 2 the Magic Formula can reverse sign past its peak"
        )
    curvature = values["lateral_pacejka_e"]
    if not curvature < _CURVATURE_FACTOR_MAX:
        raise ValueError(
            f"{prefix}: config.lateral_pacejka_e must be finite and < "
            f"{_CURVATURE_FACTOR_MAX}, got {curvature!r}. The curvature factor carries "
            "a sign, but at or above 1 the Magic Formula never reaches its peak, so "
            "the grip limit the model reports would be out of reach"
        )
    for name in _LATERAL_SENSITIVITY_CONFIG_SCALARS:
        if not 0.0 <= values[name] < 1.0:
            raise ValueError(
                f"{prefix}: config.{name} must be in [0, 1), got {values[name]!r}. "
                "It is a fractional fall in grip per unit of normalised load; 1 would "
                "make the load-sensitive tyre reach zero grip at twice the reference "
                "load, and a negative one would make grip rise with load"
            )
    return values


def step_lateral_force(
    config: KernelConfig,
    slip_angle_deg: float,
    camber_deg: float,
    load_n: float,
) -> float:
    """``Fy`` for one wheel, one step, from a validated configuration.

    The Python-facing composition, and the single place where the
    configuration and the state are checked against each other before any
    arithmetic happens. A caller inside a kernel does not come through here -
    it reads the configuration outside its loop through
    :func:`validated_lateral_scalars` and calls :func:`tyre_lateral_force`
    directly, because this function's checks are Python and a compiled loop
    cannot make them.

    ``slip_angle_deg`` and ``camber_deg`` are the contact patch's own angles
    in degrees, positive slip angle and positive camber each producing a
    positive (leftward) force; ``load_n`` is the vertical load the patch
    carries, which the caller owns - static split, transfer and downforce are
    other tasks' work. Steady state only: the force for a slip angle is the
    force for that slip angle, with no relaxation lag and no combination with
    the longitudinal force, both of which are Task 4's and Task 5's to add.

    Every scalar is narrowed to ``float`` so that equivalent ``int`` and
    ``float`` callers share one compiled specialisation, and every value is
    refused - rather than coerced or clamped - when it is nonfinite or the
    configuration cannot be used: a nonfinite slip angle is arithmetic that
    returns NaN rather than an error, and a NaN that reaches an integrator is
    a run that looks like it finished.
    """
    slip_angle = _checked_float("slip_angle_deg", slip_angle_deg, prefix="step_lateral_force")
    camber = _checked_float("camber_deg", camber_deg, prefix="step_lateral_force")
    load = _checked_float("load_n", load_n, prefix="step_lateral_force")
    values = validated_lateral_scalars(config, "step_lateral_force")
    return tyre_lateral_force(
        slip_angle,
        camber,
        load,
        values["lateral_pacejka_b"],
        values["lateral_pacejka_c"],
        values["lateral_pacejka_e"],
        values["lateral_pacejka_mu"],
        values["load_sensitivity_reference_n"],
        values["load_sensitivity_peak"],
        values["load_sensitivity_stiffness"],
        values["camber_stiffness_n_per_deg"],
    )


def _checked_float(label: str, value: object, *, prefix: str = "step_lateral_force") -> float:
    """Return a finite real scalar as ``float``, or fail before it reaches Numba.

    ``bool`` is refused although it is an ``int``, because ``True`` as a slip
    angle or a load is a caller bug rather than a number. This mirrors
    ``forces._checked_float`` and ``loads._checked_float`` rather than sharing
    either: the three modules validate different field sets for different
    models, and a shared private helper between them would be a coupling for
    the sake of six lines.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{prefix}: {label} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{prefix}: {label} must be finite, got {value!r}")
    return number
