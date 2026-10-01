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

:mod:`f1telemetry.physics.powertrain` holds P1-T4's: the synthesised ICE torque curve and the
turbo-lag multiplier below about 4 000 rpm, read from the same ``KernelConfig`` and under the
same conventions. It reuses :func:`~f1telemetry.physics.forces.speed_curve` for the interpolation
rather than carrying a second interpolator, and it splits the two ways in the same way: compiled
primitives that take bare numbers, and one Python entry point that validates what it reads
before handing it over.

:mod:`f1telemetry.physics.gearbox` holds P1-T5's: gear selection against the configured shift
points, a shift timer that freezes the gear while it runs, and the clutch that both the boost cut
and trailing throttle pass through. Its state - the gear, the shift timer and the clutch engagement
- is one caller-owned ``float64`` buffer written in place, so a run steps a single array with no
allocation, and its engine torque comes from :func:`~f1telemetry.physics.powertrain.step_ice_torque`
rather than from a second copy of the torque lookup.

It returns **differential-side** torque: the throttled engine torque multiplied by the selected
gear's ratio and by `final_drive`, and only then passed through the clutch ceiling — `PLAN.md`
section 6 puts the clutch downstream of the gearbox, so that capacity is a differential-side figure.
The committed value sits inside the range the box produces (3 000 Nm, against 3 844 Nm in first and
1 618 Nm in eighth), which leaves gears 1-3 clutch-limited and gears 4-8 transparent, so the eight
ratios are read rather than merely counted. Wheel speed and force assembly are not in it: those are
P1-T6's and P1-T7's, which is why `ice_rpm` is an argument to
`f1telemetry.physics.gearbox.step_gearbox` and not something it computes.

Of the three state slots the step advances two. The gear and the shift timer move on every call;
the clutch engagement is **supplied by the caller and read, never written**, so a caller ramps a
launch or lifts off between steps and the shift's boost cut - which is applied to a local copy -
leaves the caller's value untouched.

**Sign conventions**, fixed here and matching
:class:`~f1telemetry.testing.records.GroundTruthStep` so the invariants have something
unambiguous to check (PLAN.md section 11, invariant 4):

* x is forward, so a positive force accelerates the car and drag is negative going forward;
* ``kappa`` is positive in drive, and the longitudinal tyre force has the sign of the slip;
* downforce is a positive magnitude added to the vertical load, so it is even in speed;
* ``gear`` is ``1..n_gears`` with no neutral and no reverse, and the transmitted drive torque is
  positive in drive. That pair is what makes ``PLAN.md`` section 11's invariant 7 - monotonic
  progression, no reverse under positive throttle - a property of the model rather than of a
  record that happens to be monotone.

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
