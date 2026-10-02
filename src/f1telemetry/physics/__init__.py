"""The physics core: the force models a P1 run is built from.

P0 shipped no physics and proved the Numba toolchain with a toolchain probe. Task 2 added the
first car in :mod:`f1telemetry.kernels.longitudinal` - and deliberately left every force out of
it, because deciding what the net longitudinal force is belongs to the tasks that own the models
behind it. This package is the first of those.

:mod:`f1telemetry.physics.forces` holds P1-T3's two models: speed-dependent aerodynamics and
the longitudinal tyre force, both driven by the validated arrays in
:class:`~f1telemetry.contracts.car_spec.KernelConfig`. It carries the P0 Numba conventions -
``@njit(cache=True, fastmath=False)``, flat numeric arguments, no allocation, no clock - so
there is one set of kernel rules to read, not two.

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
limits before joining it. Wheel speed and force assembly belong to P1-T6 and P1-T7.

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
  positive and reverse torque negative under positive throttle.

**What this package does not do.** No lateral force, no combined slip, no load sensitivity and no
load split: those are P2's (P2-T2 owns the static axial split, P2-T3 the load sensitivity on
``D`` and ``B``, P2-T4 the similarity method). No wheel rotational state and no assembly of drive
or brake torque into a force either - P1-T6 and P1-T7 own those, and the gearbox hands P1-T7 a
torque rather than a newton. Nothing here reads a clock, a random source or a dict, and no number
is written in Python that ``car_spec.yaml`` does not supply.

**Layer isolation.** Nothing in ``analytics``, ``server`` or the web layer may import from here
(PLAN.md section 3); ruff's banned-api list enforces it, and the tests that must import a kernel
carry a per-file ``TID251`` carve-out for exactly that reason.
"""

from __future__ import annotations

__all__ = ["__doc__"]
