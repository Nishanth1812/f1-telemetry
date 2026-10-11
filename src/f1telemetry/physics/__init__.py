"""The physics core: the force models a P1 run is built from.

P0 shipped no physics and proved the Numba toolchain with a toolchain probe. Task 2 added the
first car in :mod:`f1telemetry.kernels.longitudinal` - and deliberately left every force out of
it, because deciding what the net longitudinal force is belongs to the tasks that own the models
behind it. This package is the first of those.

:mod:`f1telemetry.physics.forces` holds P1-T3's two models - speed-dependent aerodynamics and
the longitudinal tyre force - and P1-T6/P1-T7's wheel rotational state, all driven by the
validated arrays in :class:`~f1telemetry.contracts.car_spec.KernelConfig`. It carries the P0 Numba
conventions - ``@njit(cache=True, fastmath=False)``, flat numeric arguments, no allocation and no
clock - so there is one set of kernel rules to read, not two. The wheel half closes
``I_w d(omega)/dt = T_drive - Fx r`` for four caller-owned angular speeds, hands the drivetrain's
differential-side torque to the two rear wheels as an equal split and exactly zero to the fronts
(C9.1.1), and derives each corner's static vertical load from ``front_weight_fraction`` with no
speed dependence at all.

:mod:`f1telemetry.physics.powertrain` holds P1-T4's and P1-T6's: the synthesised ICE torque curve
and the turbo-lag multiplier below about 4 000 rpm, plus the 2026 limits that bound the whole power
unit - because Section C Issue 20 states no ICE power in kW at all and bounds the engine through
fuel energy flow (C5.2.3, C5.2.4, C5.2.5) instead. The same module holds the MGU-K: the 350 kW DC
and speed-dependent propulsion caps, the crankshaft-referenced 500 Nm torque limit, the 60 000 rpm
part-speed ceiling, the grid standing-start rule, the 4 MJ usable window and the event-conditioned
per-lap recharge limit. It reuses :func:`~f1telemetry.physics.forces.speed_curve` for the
interpolation rather than carrying a second interpolator, and it splits the two ways in the same
way: compiled primitives that take bare numbers, and Python entry points that validate what they
read before handing it over.

:mod:`f1telemetry.physics.gearbox` holds P1-T5's driver-requested shifts and clutch demand, and
P1-T6's placement of the MGU-K. RPM does not shift the gearbox. The caller owns gear, shift timer,
clutch engagement, and the one-step gear request; the compiled step writes the first two state
values and returns differential-side torque using the configured ratios, final drive, and C9.2.5
clutch demand. The MGU-K joins at the **crankshaft**, ahead of the gear and the clutch, because
C5.18.2 fixes its coupling to the crankshaft at a fixed ratio, so the sum
``(throttle * ice + mgu_k) * ratio * final_drive`` is what the clutch demand sees;
:func:`~f1telemetry.physics.gearbox.step_drivetrain` is the composition that applies the motor's
limits before joining it. The torque it returns is what
:func:`~f1telemetry.physics.forces.wheel_drive_torque_nm` splits between the two rear wheels.

Of the three state slots the step advances two. The gear and the shift timer move on every call;
the clutch engagement is **supplied by the caller and read, never written**, so a caller ramps a
launch or lifts off between steps and the shift's boost cut - which is applied to a local copy -
leaves the caller's value untouched. The MGU-K's two slots - state of charge and the per-lap
recharge accumulator - are likewise advanced in place by the MGU-K step.

**Sign conventions**, fixed here and matching
:class:`~f1telemetry.testing.records.GroundTruthStep` so the invariants have something
unambiguous to check (PLAN.md section 11, invariant 4):

* x is forward, so a positive force accelerates the car and drag is negative going forward;
* ``kappa`` is positive in drive, and the longitudinal tyre force has the sign of the slip;
* downforce is a positive magnitude added to the vertical load, so it is even in speed;
* ``gear`` is ``-1`` in reverse, ``0`` in neutral, and ``1..n_gears`` forward. Drive torque is
  positive and reverse torque negative under positive throttle;
* a wheel's angular speed is positive when it turns the way a forward-rolling wheel turns, so a
  wheel spinning faster than the road is driving slip and a wheel turning slower is braking slip -
  which is the same convention the slip ratio uses, one product up.

**P2 primitives are not yet the integrated vehicle.** ``loads``, ``kinematics``, ``steering``,
``tyres``, ``combined_slip`` and ``relaxation`` now provide tested pieces, but the P1 longitudinal
kernel still uses static corner loads and does not advance lateral body motion. The assembled P2
kernel and real steering scenarios remain open. Brake input is caller-supplied torque without a
brake-capacity model; no differential, traction control or ABS is modeled (C9.9.1, C9.1.2 and
C11.4.1). Nothing here reads a clock, a random source or a dict, and no number is written in Python
that ``car_spec.yaml`` does not supply - the two symmetry assumptions (equal rear drive split, equal
load inside an axle) are derived from the wheel count rather than configured, precisely so that
they cannot be read as coefficients.

**Layer isolation.** Nothing in ``analytics``, ``server`` or the web layer may import from here
(PLAN.md section 3); ruff's banned-api list enforces it, and the tests that must import a kernel
carry a per-file ``TID251`` carve-out for exactly that reason.
"""

from __future__ import annotations

__all__ = ["__doc__"]
