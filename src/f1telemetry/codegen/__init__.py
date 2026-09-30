"""Code generation from the contract.

``channels.yaml`` is read here and nowhere else. Three artifacts are produced:

* ``src/f1telemetry/generated/channels.py`` - the ``CHANNELS`` registry, per-group
  dataclasses and a runtime loader (P0-T4).
* ``src/f1telemetry/generated/parquet_schema.py`` - one Parquet schema per group (P0-T5).
* ``web/src/generated/channels.ts`` - TypeScript channel types (P0-T5).

Generation is idempotent: iteration follows the declaration order in ``channels.yaml``,
no timestamp or environment value is written into any output, and ``--check`` fails when
a committed artifact is stale. CI runs ``--check`` so generated code can never drift from
the contract (P0-T6).
"""

from __future__ import annotations

__all__ = ["__doc__"]
