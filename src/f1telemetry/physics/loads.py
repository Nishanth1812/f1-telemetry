"""P2-T2: the four corner vertical loads, and the quasi-static suspension travel they imply.

``PLAN.md`` section 2 fixes suspension as quasi-static and this module is that decision written
out: four vertical loads from validated CG/axle/track geometry, plus the suspension travel those
loads imply through the configured wheel rates. No linkage state, no damper, no roll rate - a
wheel is carrying a number, and the number is what the tyre model is handed.

**The coupling, and why it is an input.** Load transfer depends on acceleration, acceleration
depends on tyre force, and tyre force depends on load. Resolving that inside one 100 us step is an
algebraic loop. The plan's answer is an explicit one, and this module implements only that: the
caller passes the *previous* step's body accelerations, and the loads those numbers produce are
returned. Nothing here estimates, iterates or substitutes an acceleration, so a run seeds the
coupling to zero and stores the accelerations it resolves for the next step. The consequence is
stated rather than hidden: the load a tyre is asked about is one 100 us step behind the force it
produced, which is why this is a documented fixed-lag coupling and not an implicit solve.

**The formulae**, in the corner order :mod:`f1telemetry.physics.forces` already fixes,
``FL, FR, RL, RR`` (the indices are imported from there, not restated, so the load order and the
wheel-state order cannot drift apart):

* Total vertical load, with positive ``az`` upward in the z-up frame::

      Fz_total = m (g + az) + Fz_aero

  This is P1's accounting convention carried forward: the four loads sum to weight plus downforce,
  which is what invariant 3 checks. ``az`` is a *body* acceleration, so a body accelerating upward
  presses harder on the road; that is why it adds to the load instead of subtracting from it.
* The untransferred load is distributed by the CG position, the same share P1 used for the static
  split: the front axle carries ``lr / L`` of the total and the rear ``lf / L``, and each axle
  splits evenly between its two wheels. The two arms are summed to form the wheelbase, so the
  model cannot be handed a wheelbase and two arms that disagree.
* Longitudinal transfer is the textbook geometric form the plan names,
  ``dFz = m ax h / L``, moved from the front axle to the rear axle and halved onto each wheel.
* Lateral transfer is a roll moment split between the axles by the configured roll-stiffness
  fraction and converted to a load through each axle's *own* track::

      M_roll  = m ay h
      dFz_f   = phi_front  M_roll / t_front
      dFz_r   = (1 - phi)  M_roll / t_rear

  ``ay > 0`` is leftward, so it is a left-hand turn, and the outside of a left-hand turn is the
  *right-hand* side of the car: positive ``ay`` loads the right wheels and unloads the left ones.
  The split is by roll *stiffness*, not by roll moment share of load, because the anti-roll bars
  that set it are the distribution the configured fraction describes.
* **Lift-off is a result, not a negative load.** A corner whose raw load goes below zero is lifted
  and stays at exactly ``0.0``; the axle's other wheel is relieved by the same amount, so the
  total is unchanged and no tyre is ever asked to push on the road from below. The direction of
  that relief matters: a corner at ``-2000 N`` was arithmetically *taking* 2000 N from its partner
  on the same axle, so lifting it frees the partner rather than loading it further, and the whole
  axle load ends up on the wheel that is still down. P1's tyre primitives already return exactly
  zero force for a zero or negative load, so a lifted wheel costs the car grip rather than pulling
  it into the road.

**Suspension travel is reported, and its limit is enforced.** Travel is the corner's load
deviation from its own weight-and-aero share over the configured per-axle wheel rate, positive in
compression. Where the deviation would exceed the configured travel limit, the load is clamped to
the limit and the clamped-away load is shared out over the corners that still have room for it, so
the total survives the clamp. Every corner that *reached* its limit is flagged in the result, and
so is any corner the redistribution could not bring back inside its band - the plan's "expose a
failure/limit result rather than silently clipping an impossible load". The band is a declared
limit on suspension travel, so clamping it is a statement that the configuration cannot represent
that load case rather than a bump-stop force: a real stop adds force, this moves it sideways
instead, and that is said here rather than dressed up as a stop.

**Two things this module deliberately does not do.** It does not use the configured pitch-stiffness
split: the longitudinal transfer above is the plan's ``m ax h / L`` geometric form, and a pitch
stiffness split belongs to a body pitch *angle*, which is a state this task does not own. It also
does not read camber gain or bump steer, which are the quasi-static *kinematics* of Task 6 and are
not needed to say what a wheel is carrying. Both are named here so that their absence from this
module reads as a decision rather than an oversight.

**Limitations worth stating plainly.** The roll moment uses the full CG height with no roll-centre
offset (no roll-centre height is configured), and the sprung/unsprung split is not resolved in the
transfer - the total mass and the total CG height are used throughout. Aerodynamic load is
distributed by the same CG share as the weight, because no aerodynamic centre-of-pressure position
is configured; P1's kernel added downforce equally to all four wheels, so this is a deliberate
change of convention and is called out rather than inherited silently. Total vertical load is
required to be positive, since there is no distribution of a negative total.

**Two ways in, deliberately.** The primitives take flat scalars and caller-owned ``float64``
vectors and are what a compiled kernel calls; :func:`step_loads` is the Python-facing composition
that checks the configuration and the state before any arithmetic happens. Its small
:class:`CornerLoads` result is immutable, so a caller cannot adjust a load after the fact and then
report the total it adjusted - the total is the sum of what was returned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from numba import njit

from f1telemetry.physics.forces import (  # noqa: TID251 -- the corner order must be one definition
    FL_WHEEL_INDEX,
    FR_WHEEL_INDEX,
    RL_WHEEL_INDEX,
    RR_WHEEL_INDEX,
    WHEEL_COUNT,
)

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import KernelConfig

__all__ = [
    "CornerLoads",
    "clamp_travel_to_limits",
    "corner_loads_n",
    "lateral_roll_moment_nm",
    "longitudinal_load_transfer_n",
    "step_loads",
    "suspension_travel_m",
    "vertical_load_total_n",
]

# One half of an axle: both wheels of an axle start from the same axle load, and a transfer that
# applies to the axle applies to half of it on each side. Written as a name so the symmetry is
# visible at every use rather than being a bare 0.5 that reads as an arbitrary factor.
_HALF: Final[float] = 0.5
# The four corner loads are in FL, FR, RL, RR order, which is the order
# `f1telemetry.physics.forces` indexes its wheel states in; the indices are imported from there so
# this module cannot quietly adopt a layout of its own.
#
# An axle is the two adjacent corners, which is the one place that order is read positionally.
_AXLE_OF_CORNER: Final[tuple[int, int, int, int]] = (0, 0, 1, 1)
# How far outside its own band a corner may sit before it is reported as still outside. Proportional
# to the band rather than an absolute number of metres, so it cannot matter at one scale and matter
# at another; it exists only so a rounding-level residue is not reported as a travel violation.
_BAND_TOLERANCE: Final[float] = 1e-9
# The sum of the four loads is allowed to differ from the sum of their base by this proportion of
# the car's own load before the difference is treated as load rather than as arithmetic rounding.
# Total-load conservation is therefore exact to about 1e-12 relative, which is far tighter than any
# invariant tolerance and is bounded here so a lifted corner cannot be nudged off zero by dust.
_SUM_TOLERANCE: Final[float] = 1e-12


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def vertical_load_total_n(
    mass_kg: float,
    gravity_m_s2: float,
    vertical_accel_m_s2: float,
    aero_downforce_n: float,
) -> float:
    """``Fz_total = m (g + az) + Fz_aero``, in newtons - the number the four loads must sum to.

    ``az`` is positive upward in the z-up frame, so it *raises* the total: a body accelerating
    upward presses harder on the road. ``aero_downforce_n`` is a magnitude (P1's
    ``aero_forces`` returns one), so it can only add.

    This is a pure arithmetic primitive with no guard, because a compiled function cannot raise a
    useful error and :func:`step_loads` refuses a nonpositive total before any of this is called.
    """
    return mass_kg * (gravity_m_s2 + vertical_accel_m_s2) + aero_downforce_n


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def longitudinal_load_transfer_n(
    mass_kg: float,
    longitudinal_accel_m_s2: float,
    cg_height_m: float,
    wheelbase_m: float,
) -> float:
    """``dFz = m ax h / L`` in newtons: the load that moves from the front axle to the rear.

    Positive acceleration moves load rearward, so the caller subtracts this from the front axle
    and adds it to the rear, then halves it onto each wheel. This is the geometric form the plan
    names explicitly, and it is deliberately *not* a pitch-stiffness split: the configured
    ``pitch_stiffness_front_fraction`` describes how a pitch *moment* divides between the axles of
    a body that has a pitch angle, which is a state this task does not own.

    No guard on ``wheelbase_m``: a zero wheelbase is a zero denominator, and the boundary that can
    say so is :func:`step_loads`.
    """
    return mass_kg * longitudinal_accel_m_s2 * cg_height_m / wheelbase_m


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def lateral_roll_moment_nm(
    mass_kg: float,
    lateral_accel_m_s2: float,
    cg_height_m: float,
) -> float:
    """``M = m ay h`` in newton-metres: the moment the tyres must react to roll the car.

    The full CG height is used, with no roll-centre offset, because no roll-centre height is
    configured. A roll centre below the CG reduces the effective arm; omitting it makes the
    predicted transfer *larger* than a car with a real roll centre would produce, which is the
    conservative direction for a model with no roll-rate state.
    """
    return mass_kg * lateral_accel_m_s2 * cg_height_m


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def _redistribute_load(loads_n: np.ndarray, delta_n: float) -> None:
    """Apply a signed ``delta_n`` across the loaded corners in proportion to their load.

    Shared out proportionally rather than dumped on one wheel because there is no basis in the
    configuration for choosing one: anything else would be a parameter invented to move a number.
    Corners at exactly zero are excluded, so a lifted wheel stays lifted - it has no contact patch
    to hand load to or take it from.

    The sum changes by exactly ``delta_n`` up to floating-point rounding. A zero delta and a set
    with nothing carrying load both return without touching anything, so this cannot divide by
    zero.
    """
    if delta_n == 0.0:
        return
    carrying = 0.0
    for index in range(loads_n.size):
        if loads_n[index] > 0.0:
            carrying += loads_n[index]
    if carrying <= 0.0:
        return
    for index in range(loads_n.size):
        if loads_n[index] > 0.0:
            loads_n[index] += loads_n[index] * delta_n / carrying


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def _band_room_n(
    loads_n: np.ndarray,
    base_n: np.ndarray,
    axle_ride_rate_n_per_m: np.ndarray,
    axle_travel_limit_m: np.ndarray,
    index: int,
    direction_n: float,
) -> float:
    """How much this corner's load may still move in the direction ``direction_n`` asks for.

    Directional, not a magnitude of slack: a corner already pinned against the bottom of its band
    has all of its room above it, and a magnitude-only "room" would report it as full. Zero when
    the corner is pinned in the direction asked for, and negative when it is pinned on the far side
    of it, which the caller skips rather than treating as room.

    The band itself is ``rate x limit`` in newtons, which is the largest load deviation the
    configured travel represents at this axle.
    """
    axle = _AXLE_OF_CORNER[index]
    band_n = axle_ride_rate_n_per_m[axle] * axle_travel_limit_m[axle]
    deviation_n = loads_n[index] - base_n[index]
    if direction_n > 0.0:
        return band_n - deviation_n
    return band_n + deviation_n


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def _restore_load_sum(
    loads_n: np.ndarray,
    base_n: np.ndarray,
    axle_ride_rate_n_per_m: np.ndarray,
    axle_travel_limit_m: np.ndarray,
) -> None:
    """Undo whatever clamping did to the *sum*, without pushing any corner out of its band.

    Clamping is not conservative. Pulling a corner back to its band changes the sum, and clipping a
    large droop to a small one can *raise* it: four corners at ``-706, -706, +706, +706`` N clip to
    ``-300, -300, +700, +700`` and the sum is 800 N heavier than it was. That load has to come back
    off the corners or the total vertical load stops being the car's weight.

    **The residual is the error; the correction is its negation.** Both the direction the room is
    measured in and the amount applied are the correction, which is the whole subtlety here -
    distributing the residual itself moves the sum further in the direction it was already wrong.

    The correction is shared over the corners with *directional* room in that direction: a corner
    pinned at the bottom of its band has all of its room above it, which a magnitude-only "room"
    would miss, and that is exactly the case after an axle-level clamp. Where no corner has any
    room the correction is shared over the loaded corners in proportion to their load instead,
    which cannot satisfy the band - so the caller reports the corners it pushed outside rather than
    pretending the limit was met. An honest violation the kernel can surface beats a band that is
    satisfied by not checking it.

    The sum is restored up to floating-point rounding; two passes leave it at the rounding level,
    which is why the loop is bounded rather than run to convergence.
    """
    for _ in range(2):
        residual_n = 0.0
        base_total_n = 0.0
        for index in range(loads_n.size):
            residual_n += loads_n[index] - base_n[index]
            base_total_n += base_n[index]
        # A residual far below the car's own load is arithmetic rounding, not load. Correcting it
        # would spread rounding dust onto the lifted wheel and take it off zero, which is the one
        # number in this model that has to stay exact.
        if abs(residual_n) <= _SUM_TOLERANCE * abs(base_total_n):
            return
        correction_n = -residual_n
        available_n = 0.0
        for index in range(loads_n.size):
            room_n = _band_room_n(
                loads_n, base_n, axle_ride_rate_n_per_m, axle_travel_limit_m, index, correction_n
            )
            if room_n > 0.0:
                available_n += room_n
        if available_n <= 0.0:
            _redistribute_load(loads_n, correction_n)
            return
        for index in range(loads_n.size):
            room_n = _band_room_n(
                loads_n, base_n, axle_ride_rate_n_per_m, axle_travel_limit_m, index, correction_n
            )
            if room_n > 0.0:
                loads_n[index] += correction_n * room_n / available_n


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def _clip_lift_off(loads_n: np.ndarray) -> None:
    """A corner that would carry a negative load is lifted to exactly zero and stays there.

    A negative vertical load is not a small load, it is an impossible one: the tyre would be
    pulling the car into the road. The lift is taken out of the *same axle's* other wheel, and the
    direction of that relief is the physical one - a corner at ``-2000 N`` was, arithmetically,
    taking 2000 N from its partner, so lifting it frees the partner rather than loading it
    further. With two wheels on an axle that makes the whole axle load rest on the one that is
    still down, which is what a wheel in the air does to a car's load balance.

    **Relief stays inside the axle** because an axle's vertical load is fixed by the CG position in
    a quasi-static model: lateral transfer moves load left-to-right within an axle, it does not
    change what the axle carries. Only when a whole axle's raw total is negative - a longitudinal
    transfer larger than the axle's own load - is there no partner left on that axle, and the
    relief is then shared over whatever is still loaded elsewhere on the car.

    The loop is bounded by the wheel count because each pass lifts at least one corner, and a
    relieved partner can itself go negative and be lifted in the next pass.
    """
    for _ in range(loads_n.size):
        lifted_any = False
        for axle in range(2):
            first = axle * 2
            second = first + 1
            deficit_n = 0.0
            if loads_n[first] < 0.0:
                deficit_n -= loads_n[first]
                loads_n[first] = 0.0
                lifted_any = True
            if loads_n[second] < 0.0:
                deficit_n -= loads_n[second]
                loads_n[second] = 0.0
                lifted_any = True
            if deficit_n <= 0.0:
                continue
            if loads_n[first] > 0.0:
                loads_n[first] -= deficit_n
            elif loads_n[second] > 0.0:
                loads_n[second] -= deficit_n
            else:
                _redistribute_load(loads_n, -deficit_n)
        if not lifted_any:
            return


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def corner_loads_n(
    loads_n: np.ndarray,
    base_loads_n: np.ndarray,
    mass_kg: float,
    gravity_m_s2: float,
    aero_downforce_n: float,
    vertical_accel_m_s2: float,
    longitudinal_accel_m_s2: float,
    lateral_accel_m_s2: float,
    cg_to_front_axle_m: float,
    cg_to_rear_axle_m: float,
    cg_height_m: float,
    axle_track_m: np.ndarray,
    roll_stiffness_front_fraction: float,
) -> None:
    """Write the four corner loads and their untransferred base into caller-owned vectors.

    Both outputs are caller-owned ``float64`` vectors of length four in ``FL, FR, RL, RR`` order,
    because a kernel must not allocate inside its loop. ``base_loads_n`` receives the weight-plus-
    aero-plus-``az`` share of each corner - the load that corner would carry with no inertial
    transfer - and it is the datum suspension travel is measured from, so returning it here saves
    the caller a second pass over the same arithmetic.

    The body of the function is the three documented steps in order: distribute the total by the CG
    position, move the longitudinal transfer between the axles, then move the lateral transfer
    within each axle across that axle's own track. The order matters for conservation rather than
    for taste: the axle totals are formed once and each axle's lateral transfer is applied as a
    symmetric pair around its own total, so the four loads sum to the total by construction.

    **Lift-off is handled here, not by the caller.** The raw corner loads can go negative, and a
    ``boundscheck=False`` kernel cannot ask what to do about that, so the clip and the relief it
    causes happen in the same place as the arithmetic that produced it.
    """
    wheelbase_m = cg_to_front_axle_m + cg_to_rear_axle_m
    total_n = vertical_load_total_n(mass_kg, gravity_m_s2, vertical_accel_m_s2, aero_downforce_n)
    # The front share of the static load is the fraction of the wheelbase *behind* the CG, which
    # is why the rear arm is the one that appears here. The rear total is taken as the remainder
    # rather than as `total * (1 - share)` so the two axles account for the whole total exactly.
    front_axle_n = total_n * cg_to_rear_axle_m / wheelbase_m
    rear_axle_n = total_n - front_axle_n

    base_loads_n[FL_WHEEL_INDEX] = _HALF * front_axle_n
    base_loads_n[FR_WHEEL_INDEX] = _HALF * front_axle_n
    base_loads_n[RL_WHEEL_INDEX] = _HALF * rear_axle_n
    base_loads_n[RR_WHEEL_INDEX] = _HALF * rear_axle_n

    longitudinal_n = longitudinal_load_transfer_n(
        mass_kg, longitudinal_accel_m_s2, cg_height_m, wheelbase_m
    )
    front_axle_n -= longitudinal_n
    rear_axle_n += longitudinal_n

    roll_moment_nm = lateral_roll_moment_nm(mass_kg, lateral_accel_m_s2, cg_height_m)
    front_lateral_n = roll_moment_nm * roll_stiffness_front_fraction / axle_track_m[0]
    rear_lateral_n = roll_moment_nm * (1.0 - roll_stiffness_front_fraction) / axle_track_m[1]

    # Positive `ay` is a leftward acceleration, so the right-hand wheels gain.
    loads_n[FL_WHEEL_INDEX] = _HALF * front_axle_n - _HALF * front_lateral_n
    loads_n[FR_WHEEL_INDEX] = _HALF * front_axle_n + _HALF * front_lateral_n
    loads_n[RL_WHEEL_INDEX] = _HALF * rear_axle_n - _HALF * rear_lateral_n
    loads_n[RR_WHEEL_INDEX] = _HALF * rear_axle_n + _HALF * rear_lateral_n
    _clip_lift_off(loads_n)


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def suspension_travel_m(load_n: float, base_load_n: float, ride_rate_n_per_m: float) -> float:
    """``(Fz - Fz_base) / k``: quasi-static suspension travel in metres, positive in compression.

    The corner's load *deviation* from its own weight-and-aero share is what the spring reacts, so
    that is what the ride rate divides. Measuring from the aerodynamically loaded base rather than
    from the static corner load is a stated convention: it keeps the travel band from being an
    absolute limit that no amount of downforce could be represented inside, and it makes the
    datum a number this module already has.

    Positive is compression, so a corner carrying more than its base is compressed and one
    carrying less is in droop. No guard: a zero ride rate is a zero denominator, which
    :func:`step_loads` refuses before this is called.
    """
    return (load_n - base_load_n) / ride_rate_n_per_m


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def clamp_travel_to_limits(
    loads_n: np.ndarray,
    base_loads_n: np.ndarray,
    axle_ride_rate_n_per_m: np.ndarray,
    axle_travel_limit_m: np.ndarray,
    travel_limited_n: np.ndarray,
) -> None:
    """Clamp each corner's load to its configured travel band, paying for it out of the others.

    ``travel_limited_n`` is a caller-owned ``int64`` vector of length four, set where a corner is
    against its mechanical limit: either its travel exceeded the configured limit before the
    clamp, or it is still outside the band afterwards. The first is the ordinary, reachable case -
    the car ran out of travel and the model put the wheel on its stop. The second is the one the
    plan asks to be visible rather than silent: the load could not be brought inside the band at
    all, because every corner was pinned against it in the direction the balance needed.

    The order is clamp, restore the sum, clip lift-off, then report. A purely lateral transfer
    clamps an axle's two corners in opposite directions, so its own residual cancels; an axle whose
    *total* moves outside the band is the case that needs the sum restored, and restoring it is
    itself able to push a corner back outside, which is what the second flag condition exists to
    catch. Clipping lift-off again afterwards is not paranoia: the redistribution changes all four
    loads at once.

    The band is a *declared* limit on suspension travel, so a clamp here is a statement that the
    configuration cannot represent the load case rather than a bump-stop force. A real stop adds
    force; this moves it sideways instead. That is stated rather than dressed up, because the
    alternative - leaving the tyre load outside the band - would be the same clamp with neither
    the conservation nor the report.
    """
    for index in range(loads_n.size):
        axle = _AXLE_OF_CORNER[index]
        band_n = axle_ride_rate_n_per_m[axle] * axle_travel_limit_m[axle]
        deviation_n = loads_n[index] - base_loads_n[index]
        travel_limited_n[index] = 1 if abs(deviation_n) > band_n else 0
        clamped_n = deviation_n
        if clamped_n > band_n:
            clamped_n = band_n
        elif clamped_n < -band_n:
            clamped_n = -band_n
        loads_n[index] += clamped_n - deviation_n
    _restore_load_sum(loads_n, base_loads_n, axle_ride_rate_n_per_m, axle_travel_limit_m)
    _clip_lift_off(loads_n)
    for index in range(loads_n.size):
        axle = _AXLE_OF_CORNER[index]
        travel_m = suspension_travel_m(
            loads_n[index], base_loads_n[index], axle_ride_rate_n_per_m[axle]
        )
        # The tolerance is a proportion of the band, so it cannot turn a corner that is genuinely
        # outside its stop into one reported as inside it.
        outside = abs(travel_m) > axle_travel_limit_m[axle] * (1.0 + _BAND_TOLERANCE)
        travel_limited_n[index] = 1 if travel_limited_n[index] == 1 or outside else 0


@dataclass(frozen=True, slots=True)
class CornerLoads:
    """What one load calculation produced: four loads, their base, travel and the limit flags.

    Immutable, and every field is a plain tuple of floats so the result cannot be adjusted after
    the fact and then reported with a total that no longer matches it. The order of every tuple is
    ``FL, FR, RL, RR``.

    ``load_n`` is what the tyre model is handed and is never negative; ``base_load_n`` is the
    untransferred share it is measured against; ``travel_m`` is positive in compression;
    ``travel_limited`` is ``True`` for a corner that reached its configured travel limit, or that is
    still outside it after the redistribution. It is a reportable model limitation, not an error.
    """

    load_n: tuple[float, float, float, float]
    base_load_n: tuple[float, float, float, float]
    travel_m: tuple[float, float, float, float]
    travel_limited: tuple[bool, bool, bool, bool]

    @property
    def total_load_n(self) -> float:
        """The four loads added up: weight plus aero plus the vertical-acceleration term."""
        return math.fsum(self.load_n)

    @property
    def lift_off(self) -> tuple[int, ...]:
        """Corners carrying exactly no load, in corner order.

        Derived rather than stored, because a corner at exactly ``0.0`` *is* a lifted wheel in this
        model: nothing else can produce one.
        """
        return tuple(index for index, value in enumerate(self.load_n) if value == 0.0)


def step_loads(
    config: KernelConfig,
    *,
    longitudinal_accel_m_s2: float,
    lateral_accel_m_s2: float,
    vertical_accel_m_s2: float = 0.0,
    aero_downforce_n: float = 0.0,
) -> CornerLoads:
    """The Python-facing composition: check the configuration and the state, then compute.

    ``longitudinal_accel_m_s2`` and ``lateral_accel_m_s2`` are the previous step's body
    accelerations and are required, not defaulted to zero, because this is the explicit coupling
    the plan fixes and a default would let a caller believe it had supplied one.
    ``vertical_accel_m_s2`` and ``aero_downforce_n`` default to zero because a straight-line caller
    at rest genuinely has nothing to pass for them.

    Everything that could divide by zero, index past the end of a length-two vector or poison a
    load with a NaN is refused here, before any arithmetic: a replaceable public ``KernelConfig``
    can be made inconsistent with ``dataclasses.replace``, and a compiled function cannot defend
    itself once ``boundscheck`` is off. A nonpositive total vertical load is refused rather than
    distributed, because there is no distribution of a negative total and sharing one out would
    hand every corner a load the tyre model has to defend against.
    """
    mass = _checked_float("mass_kg", config.mass_kg, prefix="step_loads")
    gravity = _checked_float("gravity_m_s2", config.gravity_m_s2, prefix="step_loads")
    cg_to_front = _checked_float(
        "cg_to_front_axle_m", config.cg_to_front_axle_m, prefix="step_loads"
    )
    cg_to_rear = _checked_float("cg_to_rear_axle_m", config.cg_to_rear_axle_m, prefix="step_loads")
    cg_height = _checked_float("cg_height_m", config.cg_height_m, prefix="step_loads")
    roll_front = _checked_float(
        "roll_stiffness_front_fraction", config.roll_stiffness_front_fraction, prefix="step_loads"
    )
    for name, value in (
        ("mass_kg", mass),
        ("gravity_m_s2", gravity),
        ("cg_to_front_axle_m", cg_to_front),
        ("cg_to_rear_axle_m", cg_to_rear),
        ("cg_height_m", cg_height),
    ):
        if value <= 0.0:
            raise ValueError(
                f"step_loads: config.{name} must be > 0, got {value!r}. Every one of these is a "
                "magnitude or a divisor in the transfer: a zero CG arm or CG height is a zero "
                "wheelbase or a car with no leverage over its own weight"
            )
    if not 0.0 < roll_front < 1.0:
        raise ValueError(
            f"step_loads: config.roll_stiffness_front_fraction must be in (0, 1), got "
            f"{roll_front!r}. It is the share of the roll moment the front axle takes; 1 would "
            "leave the rear axle carrying none of its own transfer"
        )

    track_m = _checked_axle_pair(config.axle_track_m, "axle_track_m")
    ride_rate = _checked_axle_pair(config.axle_ride_rate_n_per_m, "axle_ride_rate_n_per_m")
    travel_limit = _checked_axle_pair(config.axle_travel_limit_m, "axle_travel_limit_m")

    longitudinal_accel = _checked_float(
        "longitudinal_accel_m_s2", longitudinal_accel_m_s2, prefix="step_loads"
    )
    lateral_accel = _checked_float("lateral_accel_m_s2", lateral_accel_m_s2, prefix="step_loads")
    vertical_accel = _checked_float("vertical_accel_m_s2", vertical_accel_m_s2, prefix="step_loads")
    downforce = _checked_float("aero_downforce_n", aero_downforce_n, prefix="step_loads")
    if downforce < 0.0:
        raise ValueError(
            f"step_loads: aero_downforce_n must be >= 0, got {downforce!r}. It is the positive "
            "magnitude `aero_forces` returns; aerodynamic lift is not a configured input"
        )

    total_n = vertical_load_total_n(mass, gravity, vertical_accel, downforce)
    if total_n <= 0.0:
        raise ValueError(
            f"step_loads: the total vertical load would be {total_n!r} N for az={vertical_accel!r} "
            f"and {downforce!r} N of downforce. There is no distribution of a total the tyres "
            "cannot carry, so this is refused rather than shared out into four negative loads"
        )

    loads_n = np.zeros(WHEEL_COUNT, dtype=np.float64)
    base_n = np.zeros(WHEEL_COUNT, dtype=np.float64)
    limited_n = np.zeros(WHEEL_COUNT, dtype=np.int64)
    corner_loads_n(
        loads_n,
        base_n,
        mass,
        gravity,
        downforce,
        vertical_accel,
        longitudinal_accel,
        lateral_accel,
        cg_to_front,
        cg_to_rear,
        cg_height,
        track_m,
        roll_front,
    )
    clamp_travel_to_limits(loads_n, base_n, ride_rate, travel_limit, limited_n)
    # Spelled out one corner at a time, in FL, FR, RL, RR order, rather than as a comprehension over
    # the vector: the result is a fixed four-tuple, the order is the contract every caller reads it
    # in, and a comprehension would hand back a length the type checker cannot vouch for.
    return CornerLoads(
        load_n=(
            float(loads_n[FL_WHEEL_INDEX]),
            float(loads_n[FR_WHEEL_INDEX]),
            float(loads_n[RL_WHEEL_INDEX]),
            float(loads_n[RR_WHEEL_INDEX]),
        ),
        base_load_n=(
            float(base_n[FL_WHEEL_INDEX]),
            float(base_n[FR_WHEEL_INDEX]),
            float(base_n[RL_WHEEL_INDEX]),
            float(base_n[RR_WHEEL_INDEX]),
        ),
        travel_m=(
            float(
                suspension_travel_m(
                    loads_n[FL_WHEEL_INDEX],
                    base_n[FL_WHEEL_INDEX],
                    ride_rate[_AXLE_OF_CORNER[FL_WHEEL_INDEX]],
                )
            ),
            float(
                suspension_travel_m(
                    loads_n[FR_WHEEL_INDEX],
                    base_n[FR_WHEEL_INDEX],
                    ride_rate[_AXLE_OF_CORNER[FR_WHEEL_INDEX]],
                )
            ),
            float(
                suspension_travel_m(
                    loads_n[RL_WHEEL_INDEX],
                    base_n[RL_WHEEL_INDEX],
                    ride_rate[_AXLE_OF_CORNER[RL_WHEEL_INDEX]],
                )
            ),
            float(
                suspension_travel_m(
                    loads_n[RR_WHEEL_INDEX],
                    base_n[RR_WHEEL_INDEX],
                    ride_rate[_AXLE_OF_CORNER[RR_WHEEL_INDEX]],
                )
            ),
        ),
        travel_limited=(
            bool(limited_n[FL_WHEEL_INDEX]),
            bool(limited_n[FR_WHEEL_INDEX]),
            bool(limited_n[RL_WHEEL_INDEX]),
            bool(limited_n[RR_WHEEL_INDEX]),
        ),
    )


def _checked_axle_pair(value: object, name: str) -> np.ndarray:
    """A per-axle quantity: a length-two, C-contiguous ``float64`` vector of positive values.

    Front first, matching every other per-axle array in ``car_spec.yaml`` and every kernel that
    indexes an axle by position with ``boundscheck`` off. A length-one or length-three vector, a
    ``float32`` one, a nested one, a NaN or a non-positive entry are all refused: with bounds
    checking off, a wrong length is an out-of-bounds read rather than an error, and a zero track
    or ride rate is the zero denominator of a transfer or a travel.
    """
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.float64
        or value.ndim != 1
        or value.size != 2
        or not value.flags.c_contiguous
    ):
        raise ValueError(
            f"step_loads: config.{name} must be a C-contiguous float64 vector of length 2, "
            f"front first; got {type(value).__name__} of shape "
            f"{getattr(value, 'shape', None)} and dtype {getattr(value, 'dtype', None)}"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"step_loads: config.{name} must be finite, got {list(value)}")
    if not np.all(value > 0.0):
        raise ValueError(f"step_loads: config.{name} must be > 0 at every axle, got {list(value)}")
    return value


def _checked_float(label: str, value: object, *, prefix: str = "step_loads") -> float:
    """A finite real scalar as ``float``, or a refusal before it reaches Numba.

    ``bool`` is refused although it is an ``int``, because ``True`` as a mass or an acceleration is
    a caller bug rather than a number. This mirrors ``forces._checked_float`` rather than sharing
    it: the two modules validate different field sets for different models, and a shared private
    helper between them would be a coupling for the sake of six lines.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{prefix}: {label} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{prefix}: {label} must be finite, got {value!r}")
    return number
