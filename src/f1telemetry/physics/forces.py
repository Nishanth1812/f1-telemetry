"""P1-T3/P1-T6/P1-T7: aerodynamic and longitudinal tyre forces, and the wheel state they close with.

Three models, one module, because they are halves of the same step and are documented against the
same sign conventions.

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

**The wheel rotational state** is what makes that force a *closed loop* rather than an output.
P1-T6 adds four caller-owned angular speeds and closes ``I_w d(omega)/dt = T_drive - Fx r``, and
P1-T7 is the assembly that turns the gearbox's differential-side torque into those wheel states.
Four decisions belong here rather than in the kernel:

* **Drive torque reaches the two rear wheels and nowhere else (C9.1.1).**
  :func:`wheel_drive_torque_nm` returns exactly ``0.0`` for a front index rather than a small
  number, and the split is :data:`REAR_DRIVE_SHARE`, a derived ``1 / REAR_WHEEL_COUNT`` rather
  than a configured share. The clause fixes the axle; it says nothing about left against right,
  because left against right is a *differential*, and P1 has no differential model. So the equal
  split is a stated synthetic assumption, and writing it as a constant rather than as data is what
  stops it being read as a calibrated or regulated figure.
* **Nothing intervenes (C9.1.2, C9.9.1, C11.4.1).** :func:`wheel_drive_torque_nm` takes no wheel
  speed at all, which is the whole of "no system capable of preventing driven wheels from spinning
  under power": traction control would appear here as a term that reads slip, and a limited-slip
  differential would appear here as a faster wheel being handed more than its share. Spin and lock
  are still real states - the Magic Formula's own fall-off past its peak produces them - so the
  absence of intervention is a statement about the torque, not about the absence of wheelspin.
* **The load split is static and symmetric.** :func:`static_wheel_load_n` is
  ``weight x axle fraction / 2`` with no speed dependence at all; downforce is added by the
  caller because the front/rear share of the aero load is P2-T2's decision, exactly as the
  corner loads are. Summing the four returns the car's own weight, which is what invariant 3
  checks.
* **The reaction is not optional.** :func:`wheel_angular_acceleration_rad_s2` is the balance above,
  and it carries the ``- Fx r`` term because integrating a wheel on drive torque alone manufactures
  energy: the tyre that pushed the car forward is pushing back on the wheel by ``Fx r``.

**Two ways in, deliberately.** The primitives take flat scalars and ``float64`` arrays and are
what a compiled kernel calls: ``longitudinal.simulate`` reads every coefficient out of the config
once, in Python, and the loop is handed numbers, and the same rule applies here.
:func:`step_forces` and :func:`step_wheel` are the Python-facing compositions for one step, and the
places the configuration and the state are checked before any arithmetic happens. They are not
hot-path calls: a kernel reads the config outside its loop and calls the primitives, through
:func:`validated_config_scalars` and :func:`validate_aero_arrays`.

**No state, no allocation, no clock.** Every function is a pure function of its arguments, so the
same inputs give the same bits, and a P1 run's determinism claim does not have to be
re-earned here. The wheel state that *does* live somewhere is caller-owned, as every state in
this project is.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import numpy as np
from numba import njit

from .pacejka import magic_formula_shape  # noqa: TID251 -- shared physics primitive

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import KernelConfig

__all__ = [
    "FIRST_REAR_WHEEL_INDEX",
    "FL_WHEEL_INDEX",
    "FR_WHEEL_INDEX",
    "REAR_DRIVE_SHARE",
    "REAR_WHEEL_COUNT",
    "RL_WHEEL_INDEX",
    "RR_WHEEL_INDEX",
    "WHEEL_COUNT",
    "aero_forces",
    "dynamic_pressure_pa",
    "slip_ratio",
    "speed_curve",
    "static_wheel_load_n",
    "step_forces",
    "step_wheel",
    "tyre_longitudinal_force",
    "validate_aero_arrays",
    "validated_config_scalars",
    "wheel_angular_acceleration_rad_s2",
    "wheel_drive_torque_nm",
    "wheel_tyre_force_n",
]

# The four contact patches, in the fixed order every wheel state and every trace uses. Two wheels
# per axle, front pair first, so a caller never holds a bare `0..3` without a name for it.
WHEEL_COUNT: Final[int] = 4
FL_WHEEL_INDEX: Final[int] = 0
FR_WHEEL_INDEX: Final[int] = 1
RL_WHEEL_INDEX: Final[int] = 2
RR_WHEEL_INDEX: Final[int] = 3

# C9.1.1 (page 100): "The transmission may only drive the two rear wheels." Read as an index
# boundary rather than as a membership test, so the primitive that divides a drivetrain torque
# between wheels is one comparison and cannot disagree with the one that picks a vertical load.
FIRST_REAR_WHEEL_INDEX: Final[int] = 2
REAR_WHEEL_COUNT: Final[int] = WHEEL_COUNT - FIRST_REAR_WHEEL_INDEX
WHEELS_PER_AXLE_COUNT: Final[int] = 2

# Equal left/right drive split, and equal left/right static load. Both are *synthetic*: the clause
# that requires rear-wheel drive fixes the axle and nothing else, and neither figure is a number
# any file supplies. Derived from the wheel count so there is one place that says "half" and a
# change of layout moves both together.
REAR_DRIVE_SHARE: Final[float] = 1.0 / REAR_WHEEL_COUNT
_AXLE_LOAD_SHARE: Final[float] = 1.0 / WHEELS_PER_AXLE_COUNT

# The scalars the force and wheel model reads and the sign each one has to have, read as data
# rather than as a list of names because a name that is not checked here is a name nobody notices
# is unchecked. `pacejka_e` is the odd one out: the curvature factor carries a sign in the Magic
# Formula, so it is only required to be finite. `rolling_radius_m` and `wheel_inertia_kg_m2` are
# P1-T6's additions and are here because they are the two divisors of the wheel equation; a caller
# that does not read them is still held to them, which is what a shared boundary over a replaceable
# `KernelConfig` has to mean.
_POSITIVE_CONFIG_SCALARS: Final[tuple[str, ...]] = (
    "air_density_kg_m3",
    "reference_area_m2",
    "pacejka_b",
    "pacejka_c",
    "pacejka_mu",
    "rolling_radius_m",
    "slip_ratio_min_speed_m_s",
    "wheel_inertia_kg_m2",
)
_FINITE_CONFIG_SCALARS: Final[tuple[str, ...]] = ("pacejka_e",)
# A fraction of a total weight, so the interesting boundary is the top as well as the bottom:
# exactly 1 would leave the rear axle carrying nothing, which is a fraction of nothing.
_FRACTION_CONFIG_SCALARS: Final[tuple[str, ...]] = ("front_weight_fraction",)

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
    return magic_formula_shape(stiffness * slip, shape, curvature)


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


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def wheel_drive_torque_nm(wheel_index: int, drivetrain_torque_nm: float) -> float:
    """The torque this one wheel is handed, in newton-metres. Rear wheels only (C9.1.1).

    **A front wheel returns exactly ``0.0``**, not a small number, and the four torques sum back to
    the drivetrain torque that went in. C9.1.1 (page 100) reads "The transmission may only drive the
    two rear wheels", and the sharpest form of that is a front index producing a hard zero: a
    drive force quietly applied to the front axle produces a straight-line run that still
    accelerates plausibly, which is exactly why the claim is worth a zero and not a tolerance.

    **The split is half, and it is synthetic.** :data:`REAR_DRIVE_SHARE` is
    ``1 / REAR_WHEEL_COUNT``: no clause fixes a left-to-right distribution, because that is what a
    differential decides and P1 has no differential model. Equal left and right is therefore a
    stated assumption rather than a prediction or a regulated figure, and it is derived from the
    wheel count rather than written down so that a change of layout moves it with everything else.

    **The wheel's own speed is not an argument**, and that is the C9.1.2 claim stated in a
    signature rather than in prose: no system capable of preventing driven wheels from spinning
    under power would read slip here. A limited-slip differential (C9.9.1) would appear as a faster
    rear wheel being handed more than its share; neither does. Wheelspin is still a real state -
    the Magic Formula's fall-off past its peak produces it - so the absence of intervention is a
    statement about the *torque*, not a claim that wheels never slip.

    ``wheel_index`` is not range-checked here: an index at or past
    :data:`FIRST_REAR_WHEEL_INDEX` is a rear wheel, and a caller cannot be handed a corner that
    does not exist because :func:`step_wheel` refuses the index before it gets here, exactly as
    :func:`~f1telemetry.physics.gearbox.step_gearbox` is the only way into a
    ``boundscheck=False`` ratio lookup.
    """
    if wheel_index < FIRST_REAR_WHEEL_INDEX:
        return 0.0
    return drivetrain_torque_nm * REAR_DRIVE_SHARE


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def static_wheel_load_n(weight_n: float, front_weight_fraction: float, wheel_index: int) -> float:
    """Static per-wheel vertical load, in newtons: ``weight x axle fraction / 2``.

    **Static, and symmetric, on purpose.** Load transfer is P2-T2's work (the longitudinal part of
    it depends on acceleration, which this function is not given), so P1 splits the car's weight by
    ``front_weight_fraction`` from ``car_spec.yaml`` and divides evenly inside each axle. The four
    results sum to ``weight_n``, which is what invariant 3 checks and what makes the straight-line
    approximation honest: a caller adds downforce on top and the corner loads still sum to weight
    plus downforce.

    Which share of the *aero* load goes where is deliberately absent. ``step_forces`` documents the
    same boundary from the other side - the caller owns the corner load - and P2-T2 is what turns
    both of these into a per-corner transfer.

    ``front_weight_fraction`` must be strictly inside ``(0, 1)``, which
    :func:`validated_config_scalars` checks. Exactly ``1`` would leave the rear axle carrying
    nothing at all, and a rear axle carrying nothing is not a car that is light on drive.
    """
    if wheel_index < FIRST_REAR_WHEEL_INDEX:
        return weight_n * front_weight_fraction * _AXLE_LOAD_SHARE
    return weight_n * (1.0 - front_weight_fraction) * _AXLE_LOAD_SHARE


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def wheel_tyre_force_n(
    speed_m_s: float,
    wheel_omega_rad_s: float,
    load_n: float,
    rolling_radius_m: float,
    min_speed_m_s: float,
    stiffness: float,
    shape: float,
    curvature: float,
    friction: float,
) -> float:
    """``Fx`` for one wheel from its own angular speed: the tyre model with ``omega r`` resolved.

    :func:`slip_ratio` is documented as taking a *circumferential* wheel speed, and this is where
    that product is formed - ``omega`` is a state and ``r`` is a configuration, so neither one
    alone is the input the tyre model wants. Everything after the product is
    :func:`tyre_longitudinal_force` unchanged, including the zero-and-negative load guard, so a
    wheel in the air transmits nothing here too.

    Nine scalars in a row is the kind of call where a transposed pair still compiles and still
    returns a plausible force, which is why the composed result is asserted against the two
    primitives reached independently rather than against a number copied out of a run.
    """
    slip = slip_ratio(wheel_omega_rad_s * rolling_radius_m, speed_m_s, min_speed_m_s)
    return tyre_longitudinal_force(slip, load_n, stiffness, shape, curvature, friction)


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def wheel_angular_acceleration_rad_s2(
    drive_torque_nm: float,
    tyre_fx_n: float,
    rolling_radius_m: float,
    wheel_inertia_kg_m2: float,
    brake_torque_nm: float = 0.0,
) -> float:
    """``d(omega)/dt = (T_drive + T_brake - Fx r) / I_w``, in rad/s^2.

    **The ``- Fx r`` term is the whole reason the wheel is a state.** The tyre that pushed the car
    forward by ``Fx`` pushes back on the wheel by ``Fx r``, and a wheel integrated on drive torque
    alone would manufacture energy out of nothing - accelerating while the road does the opposite,
    with nothing to say so. The reaction is also what makes the fronts *free-rolling* rather than
    inert: with ``T_drive = 0`` this reduces to the road spinning an undriven wheel up or down,
    and a wheel rolling at ``omega r = v`` feels exactly nothing because its force is exactly zero.

    The caller supplies signed brake torque separately; it acts on that wheel only. There is no
    ABS feedback or differential coupling, so this term never changes another wheel's torque.

    ``wheel_inertia_kg_m2`` is the divisor and is ``car_spec.yaml`` data: no clause fixes a
    rotational inertia, so the value is a labelled synthetic placeholder inside ``PLAN.md``
    section 4's band, and :func:`validated_config_scalars` refuses zero, negative and nonfinite.
    """
    return (drive_torque_nm + brake_torque_nm - tyre_fx_n * rolling_radius_m) / wheel_inertia_kg_m2


def validated_config_scalars(config: KernelConfig, prefix: str) -> dict[str, float]:
    """Every scalar the force and wheel model reads, narrowed to ``float`` and range-checked.

    **One validator for the model, shared by every Python entry point and by the kernel.**
    :func:`step_forces`, :func:`step_wheel` and ``longitudinal.simulate`` all need this, and three
    copies of the same eight names and the same sign rules would be three places for the rule to be
    wrong. The kernel calls it once, outside its loop, which is what
    ``PLAN.md`` section 4.1 rule 3 requires: the compiled loop is handed numbers and never sees the
    config object.

    ``prefix`` names the calling entry point in the message, because three functions here share
    these helpers and ``slip_ratio_min_speed_m_s must be > 0`` is a worse error than one that says
    which function refused it.

    The sign rules are the ones a compiled function cannot make for itself once ``boundscheck`` is
    off: magnitudes strictly positive, the Magic Formula's curvature factor finite but signed
    because ``E`` carries a sign, and the front weight fraction strictly inside ``(0, 1)``. It is
    deliberately *wider* than any one caller's read - :func:`step_forces` does not use the wheel
    inertia, and is still held to it - because ``KernelConfig`` is a public frozen dataclass that
    ``dataclasses.replace`` can make inconsistent, and a boundary that checked only what this call
    happened to touch would let a bad value sit in the same object the next call reads.
    """
    names = (*_POSITIVE_CONFIG_SCALARS, *_FRACTION_CONFIG_SCALARS, *_FINITE_CONFIG_SCALARS)
    values = {
        name: _checked_float(f"config.{name}", getattr(config, name), prefix=prefix)
        for name in names
    }
    for name in _POSITIVE_CONFIG_SCALARS:
        if values[name] <= 0.0:
            raise ValueError(
                f"{prefix}: config.{name} must be finite and > 0, got {values[name]!r}"
            )
    for name in _FRACTION_CONFIG_SCALARS:
        if not 0.0 < values[name] < 1.0:
            raise ValueError(f"{prefix}: config.{name} must be in (0, 1), got {values[name]!r}")
    shape = values["pacejka_c"]
    if not 1.0 < shape <= 2.0:
        raise ValueError(
            f"{prefix}: config.pacejka_c must be in (1, 2], got {shape!r}. "
            "The combined-slip peak normalization requires a finite peak argument"
        )
    curvature = values["pacejka_e"]
    if curvature >= 1.0:
        raise ValueError(
            f"{prefix}: config.pacejka_e must be finite and < 1, got {curvature!r}. "
            "The combined-slip peak equation is strictly increasing only below 1"
        )
    return values


def validate_aero_arrays(config: KernelConfig, prefix: str = "step_forces") -> None:
    """Validate the three aero arrays before they reach a compiled lookup.

    Public because ``longitudinal.simulate`` now indexes them too: the kernel reads ``Cl(v)`` and
    ``Cd(v)`` inside its step loop with ``boundscheck=False``, so the same guarantees the loader
    gives have to be re-established at the boundary that hands it the config. ``prefix`` names
    that boundary in the message for the same reason as above.
    """
    axis, cl, cd = config.aero_speed_m_s, config.cl, config.cd
    arrays = (("aero_speed_m_s", axis), ("cl", cl), ("cd", cd))
    for name, array in arrays:
        if (
            not isinstance(array, np.ndarray)
            or array.dtype != np.float64
            or array.ndim != 1
            or array.size == 0
            or not array.flags.c_contiguous
        ):
            raise ValueError(
                f"{prefix}: config.{name} must be a non-empty C-contiguous float64 vector"
            )
    if axis.shape != cl.shape or axis.shape != cd.shape:
        raise ValueError(
            f"{prefix}: config.aero_speed_m_s, config.cl and config.cd must share a length, "
            f"got {axis.shape}, {cl.shape} and {cd.shape}"
        )
    for name, array in arrays:
        if not np.isfinite(array).all():
            raise ValueError(f"{prefix}: config.{name} must be finite")
    if not np.all(np.diff(axis) > 0.0):
        raise ValueError(f"{prefix}: config.aero_speed_m_s must be strictly increasing")
    if not np.all(cl > 0.0) or not np.all(cd > 0.0):
        raise ValueError(f"{prefix}: config.cl and config.cd must be positive")


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
    then satisfies invariant 3 by construction, and
    :func:`static_wheel_load_n` is what produces the static part.

    ``KernelConfig`` is public and replaceable, so values are checked here before they reach Numba.
    Scalars are narrowed to ``float`` to keep one compiled specialization; aero arrays must be
    valid, contiguous float64 curves on a shared speed axis.
    """
    speed_m_s = _checked_float("speed_m_s", speed_m_s)
    wheel_speed_m_s = _checked_float("wheel_speed_m_s", wheel_speed_m_s)
    load_n = _checked_float("load_n", load_n)
    values = validated_config_scalars(config, "step_forces")
    validate_aero_arrays(config)
    downforce_n, drag_n = aero_forces(
        speed_m_s,
        values["air_density_kg_m3"],
        values["reference_area_m2"],
        config.aero_speed_m_s,
        config.cl,
        config.cd,
    )
    slip = slip_ratio(wheel_speed_m_s, speed_m_s, values["slip_ratio_min_speed_m_s"])
    tyre_fx_n = tyre_longitudinal_force(
        slip,
        load_n,
        values["pacejka_b"],
        values["pacejka_c"],
        values["pacejka_e"],
        values["pacejka_mu"],
    )
    return downforce_n, drag_n, tyre_fx_n


def step_wheel(
    config: KernelConfig,
    wheel_index: int,
    drivetrain_torque_nm: float,
    speed_m_s: float,
    wheel_omega_rad_s: float,
    load_n: float,
    brake_torque_nm: float = 0.0,
) -> tuple[float, float, float]:
    """``(drive_torque_nm, tyre_fx_n, angular_acceleration_rad_s2)`` for one wheel, one step.

    The wheel-state half of the step, in the same shape as :func:`step_forces`: read a validated
    configuration, check the state against it, and hand the caller the three numbers a caller
    outside a kernel needs rather than the arguments to rebuild them.

    ``drivetrain_torque_nm`` is the gearbox's differential-side torque for this step -
    ``gearbox.step_gearbox`` returns exactly that - and ``wheel_omega_rad_s`` is the wheel's own
    state, which is what turns the torque into slip and therefore into force. ``load_n`` is the
    vertical load the patch carries, from :func:`static_wheel_load_n` plus the caller's share of
    downforce. ``brake_torque_nm`` is a separate signed wheel torque, normally negative while the
    car rolls forward.

    **The returned angular acceleration is explicit**: the caller advances
    ``omega += alpha * dt_s``. That is the same scheme the speed uses, and the reason the wheel
    states and the chassis speed stay consistent with each other inside one loop iteration.

    Every scalar is narrowed to ``float`` and the corner index to ``int`` so that equivalent ``int``
    and ``float`` callers share one compiled specialisation, and ``wheel_index`` is refused
    outside ``0..WHEEL_COUNT-1`` rather than coerced: an out-of-range index would be read by
    :func:`wheel_drive_torque_nm` as a rear wheel and by :func:`static_wheel_load_n` as the other
    axle's load, so a bad corner would quietly drive the wrong wheel with the wrong load and still
    return a finite answer.
    """
    index = _checked_wheel_index(wheel_index)
    drivetrain_torque = _checked_float(
        "drivetrain_torque_nm", drivetrain_torque_nm, prefix="step_wheel"
    )
    speed = _checked_float("speed_m_s", speed_m_s, prefix="step_wheel")
    omega = _checked_float("wheel_omega_rad_s", wheel_omega_rad_s, prefix="step_wheel")
    load = _checked_float("load_n", load_n, prefix="step_wheel")
    brake = _checked_float("brake_torque_nm", brake_torque_nm, prefix="step_wheel")
    values = validated_config_scalars(config, "step_wheel")

    drive_torque_nm = wheel_drive_torque_nm(index, drivetrain_torque)
    tyre_fx_n = wheel_tyre_force_n(
        speed,
        omega,
        load,
        values["rolling_radius_m"],
        values["slip_ratio_min_speed_m_s"],
        values["pacejka_b"],
        values["pacejka_c"],
        values["pacejka_e"],
        values["pacejka_mu"],
    )
    alpha_rad_s2 = wheel_angular_acceleration_rad_s2(
        drive_torque_nm,
        tyre_fx_n,
        values["rolling_radius_m"],
        values["wheel_inertia_kg_m2"],
        brake,
    )
    return float(drive_torque_nm), float(tyre_fx_n), float(alpha_rad_s2)


def _checked_wheel_index(value: object) -> int:
    """A contact patch as a whole ``int`` in ``0..WHEEL_COUNT - 1``.

    Refused rather than coerced. A ``float`` would compile a second Numba specialisation beside the
    integer one, so every later caller would have to be checked against both; ``bool`` is refused
    although it is an ``int``, because ``True`` is front left by value and a corner arriving as a
    truth value is a bug rather than a corner. The range check is the load-bearing one: an index
    past the end would be treated as a rear wheel by the drive split and the other axle's load by
    the static split, so it would return a finite, plausible, wrong answer.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"step_wheel: wheel_index must be an int in 0..{WHEEL_COUNT - 1}, got {value!r} of "
            f"type {type(value).__name__}"
        )
    index = int(value)
    if not 0 <= index < WHEEL_COUNT:
        raise ValueError(
            f"step_wheel: wheel_index must be an int in 0..{WHEEL_COUNT - 1}, got {index!r}. The "
            "index picks which axle C9.1.1 lets be driven and which axle's static load is read, so "
            "a corner outside the car is a wrong model rather than an error worth returning"
        )
    return index


def _checked_float(label: str, value: object, *, prefix: str = "step_forces") -> float:
    """Return a finite numeric scalar as float, or fail before it reaches Numba."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{prefix}: {label} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{prefix}: {label} must be finite, got {value!r}")
    return number
