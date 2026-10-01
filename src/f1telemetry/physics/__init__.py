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

**Sign conventions**, fixed here and matching
:class:`~f1telemetry.testing.records.GroundTruthStep` so the invariants have something
unambiguous to check (PLAN.md section 11, invariant 4):

* x is forward, so a positive force accelerates the car and drag is negative going forward;
* ``kappa`` is positive in drive, and the longitudinal tyre force has the sign of the slip;
* downforce is a positive magnitude added to the vertical load, so it is even in speed.

**What this package does not do.** No lateral force, no combined slip, no load sensitivity and no
load split: those are P2's (P2-T2 owns the static axial split, P2-T3 the load sensitivity on
``D`` and ``B``, P2-T4 the similarity method). Nothing here reads a clock, a random source or a
dict, and no number is written in Python that ``car_spec.yaml`` does not supply.

**Layer isolation.** Nothing in ``analytics``, ``server`` or the web layer may import from here
(PLAN.md section 3); ruff's banned-api list enforces it, and the tests that must import a kernel
carry a per-file ``TID251`` carve-out for exactly that reason.
"""

from __future__ import annotations

__all__ = ["__doc__"]
