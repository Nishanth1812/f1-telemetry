"""Compiled kernels.

P0 shipped one kernel here, a toolchain probe rather than physics: P0-T1b exists so that "the
Numba toolchain works here" is a *measured* fact before P1 builds a physics core on top of it.
See :mod:`f1telemetry.kernels.probe`.

P1 adds the first kernel that is actually a car - the fixed-step straight-line longitudinal
integrator, in :mod:`f1telemetry.kernels.longitudinal`. It keeps the probe's conventions
(``cache=True``, ``fastmath=False``, caller-owned float64 buffers) so there is one set of rules
to read, not two.

Nothing outside this package may import it. ``analytics``, ``server`` and the web layer
depend on the contract and the GroundTruth side channel only (PLAN.md section 3), and ruff's
banned-api list enforces that.
"""

from __future__ import annotations

__all__ = ["__doc__"]
