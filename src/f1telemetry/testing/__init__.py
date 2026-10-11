"""Validation harness: ground-truth records, the 8 invariants, and golden traces.

Nothing here simulates anything. The records are hand-authored fixtures with values
chosen so that exactly one invariant property holds at a time, which is what makes the
invariant tests non-vacuous: every checker is run against a clean record *and* against a
record with a deliberate violation, and the test fails if the checker does not notice.

The physics these invariants are written for arrives in P1 and P2. Until then each
checker runs against representative fixture values and its result records which phase
turns it into a physics-backed check.
"""

from __future__ import annotations

__all__ = ["__doc__"]
