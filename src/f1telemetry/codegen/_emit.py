"""Shared emission helpers.

Emitters must be pure functions of the contract: same contract in, byte-identical files
out. Nothing here reads the clock, the environment, or the filesystem outside the
explicit paths passed in.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

from f1telemetry.contracts.channels import Channel, ChannelContract

__all__ = [
    "ARROW_TYPES",
    "BANNER",
    "BANNER_TS",
    "TS_TYPES",
    "Raw",
    "class_name",
    "py_repr",
    "render_py",
    "render_ts",
]


@dataclass(frozen=True, slots=True)
class Raw:
    """Literal source text that is already a Python expression."""

    text: str


BANNER: Final[str] = (
    "GENERATED FILE - DO NOT EDIT.\n"
    "Regenerate with: uv run f1-codegen\n"
    "Source of truth: channels.yaml (PLAN.md section 5.2)\n"
    "CI runs `uv run f1-codegen --check` and fails on any diff."
)
BANNER_TS: Final[str] = BANNER

ARROW_TYPES: Final[dict[str, str]] = {
    "float32": "pa.float32()",
    "float64": "pa.float64()",
    "int8": "pa.int8()",
    "int16": "pa.int16()",
    "int32": "pa.int32()",
    "uint8": "pa.uint8()",
    "uint16": "pa.uint16()",
    "uint32": "pa.uint32()",
    "bool": "pa.bool_()",
}
TS_TYPES: Final[dict[str, str]] = {
    "float32": "number",
    "float64": "number",
    "int8": "number",
    "int16": "number",
    "int32": "number",
    "uint8": "number",
    "uint16": "number",
    "uint32": "number",
    "bool": "boolean",
}

_GROUP_CLASS_SUFFIX: Final[dict[str, str]] = {
    "chassis": "Channels",
    "imu": "Channels",
    "powertrain": "Channels",
    "aero": "Channels",
    "wheel": "Channels",
    "wheel_thermal": "Channels",
    "thermal": "Channels",
    "session": "Channels",
    "driver": "Channels",
}


def class_name(group: str) -> str:
    """``wheel_thermal`` -> ``WheelThermalChannels``."""
    suffix = _GROUP_CLASS_SUFFIX.get(group, "Channels")
    return "".join(part.capitalize() for part in group.split("_")) + suffix


def py_repr(value: object) -> str:
    """Deterministic Python literal for a contract value."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (int, str)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        body = ", ".join(py_repr(item) for item in value)
        if not body:
            return "()" if isinstance(value, tuple) else "[]"
        return f"({body},)" if isinstance(value, tuple) else f"[{body}]"
    msg = f"cannot render {type(value).__name__} as a Python literal"
    raise TypeError(msg)


def ts_str(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ").replace('"', '\\"')
    return f"'{escaped}'"


def render_py(body: str) -> str:
    return f'"""{BANNER}"""\n\nfrom __future__ import annotations\n\n{body}'


def render_ts(body: str) -> str:
    return f"/**\n * {BANNER_TS.replace(chr(10), chr(10) + ' * ')}\n */\n\n{body}"


def iter_expanded(contract: ChannelContract) -> Iterable[tuple[Channel, ...]]:
    """Corner-expanded channels, grouped by their declared base name."""
    current_base: str | None = None
    bucket: list[Channel] = []
    for channel in contract.channels:
        if channel.base_name != current_base:
            if bucket:
                yield tuple(bucket)
            current_base = channel.base_name
            bucket = [channel]
        else:
            bucket.append(channel)
    if bucket:
        yield tuple(bucket)
