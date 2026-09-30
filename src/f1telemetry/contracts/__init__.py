"""Contract loading.

``channels.yaml`` is the single source of truth for every field's metadata
(PLAN.md section 5.2). The loader is the only place that knows the YAML shape; the
code generator and the runtime both consume the dataclasses defined here, so field
metadata is never restated in a second file.
"""

from __future__ import annotations

__all__ = ["__doc__"]
