"""Analytics layers (PLAN.md section 9). Residual first, validity before everything.

Layer 0 is deterministic: range, rate-of-change, stuck-value and timestamp checks
over published :class:`~f1telemetry.testing.records.SensorFrame` sequences. Per
PLAN.md section 3 nothing in here imports the physics core; thresholds come from
``channels.yaml``, never from hardcoded numbers.
"""

from __future__ import annotations

__all__: list[str] = []
