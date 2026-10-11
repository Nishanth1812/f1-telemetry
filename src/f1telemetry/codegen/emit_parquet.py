"""Emit the per-group Parquet schema (P0-T5).

One file per group on disk (PLAN.md section 8.3 warm tier), so each group gets its own
Arrow schema here. Every group file opens with a `t_s` simulation-timestamp column, then
its own channels in contract order. Event-rate groups (session) are not row-aligned to a
fixed rate, so their schema is emitted too and the writer owns the irregular indexing.
"""

from __future__ import annotations

from f1telemetry.codegen._emit import ARROW_TYPES, render_py
from f1telemetry.contracts.channels import ChannelContract

__all__ = ["TIME_COLUMN", "emit_parquet_schema"]

_INDENT = " " * 4
TIME_COLUMN = "t_s"

_DATACLASSES = (
    "@dataclass(frozen=True, slots=True)\n"
    "class ParquetColumn:\n"
    '    """One Parquet column and the contract metadata a reader needs."""\n\n'
    f"{_INDENT}name: str\n"
    f"{_INDENT}arrow_type: str\n"
    f"{_INDENT}unit: str\n"
    f"{_INDENT}rate_hz: float | None\n"
    f"{_INDENT}event: bool\n"
    f"{_INDENT}dtype: str\n"
)

_GROUP_SCHEMA = (
    "@dataclass(frozen=True, slots=True)\n"
    "class ParquetGroupSchema:\n"
    '    """Schema of one group file: the time column plus that group\'s channels."""\n\n'
    f"{_INDENT}group: str\n"
    f"{_INDENT}rate_hz: float | None\n"
    f"{_INDENT}event: bool\n"
    f"{_INDENT}columns: tuple[ParquetColumn, ...]\n\n"
    f"{_INDENT}@property\n"
    f"{_INDENT}def names(self) -> tuple[str, ...]:\n"
    f"{_INDENT * 2}return tuple(column.name for column in self.columns)\n\n"
    f"{_INDENT}def arrow_schema(self) -> pa.Schema:\n"
    f'{_INDENT * 2}"""Arrow schema for the group, time column first."""\n'
    f"{_INDENT * 2}fields = [pa.field(TIME_COLUMN, pa.float64(), nullable=False)]\n"
    f"{_INDENT * 2}for column in self.columns:\n"
    f"{_INDENT * 3}fields.append(pa.field(column.name, column.arrow_type, nullable=False))\n"
    f"{_INDENT * 2}return pa.schema(fields)\n\n"
    f"{_INDENT}def arrow_type_map(self) -> dict[str, pa.DataType]:\n"
    f"{_INDENT * 2}return {{name: _ARROW[name] for name in self.names}}"
)


def emit_parquet_schema(contract: ChannelContract) -> str:
    arrow_map = "_ARROW: Final[Mapping[str, pa.DataType]] = MappingProxyType(\n{\n"
    arrow_map += "".join(
        f"{_INDENT * 2}{dtype!r}: {expression},\n" for dtype, expression in ARROW_TYPES.items()
    )
    arrow_map += "})\n"

    group_schemas: list[str] = []
    group_functions: list[str] = []
    for group in contract.groups:
        members = contract.group_members(group)
        columns = "".join(
            f"{_INDENT * 2}ParquetColumn(\n"
            f"{_INDENT * 3}name={channel.name!r},\n"
            f"{_INDENT * 3}arrow_type={ARROW_TYPES[channel.dtype]},\n"
            f"{_INDENT * 3}unit={channel.unit!r},\n"
            f"{_INDENT * 3}rate_hz={channel.rate_hz!r},\n"
            f"{_INDENT * 3}event={channel.event!r},\n"
            f"{_INDENT * 3}dtype={channel.dtype!r},\n"
            f"{_INDENT * 2}),\n"
            for channel in members
        )
        group_schemas.append(
            f"{group.upper()}: Final[ParquetGroupSchema] = ParquetGroupSchema(\n"
            f"{_INDENT}group={group!r},\n"
            f"{_INDENT}rate_hz={members[0].rate_hz!r},\n"
            f"{_INDENT}event={members[0].event!r},\n"
            f"{_INDENT}columns=(\n{columns}{_INDENT}),\n"
            f")"
        )
        group_functions.append(
            f"def {group}_schema() -> ParquetGroupSchema:\n"
            f"{_INDENT}return {group.upper()}\n\n\n"
            f"def {group}_arrow_schema() -> pa.Schema:\n"
            f"{_INDENT}return {group.upper()}.arrow_schema()"
        )

    table = "PARQUET_SCHEMAS: Final[Mapping[str, ParquetGroupSchema]] = MappingProxyType(\n{\n"
    table += "".join(f"{_INDENT}{group!r}: {group.upper()},\n" for group in contract.groups)
    table += "})\n"

    functions = (
        "def parquet_group_schemas() -> Mapping[str, ParquetGroupSchema]:\n"
        f"{_INDENT}return PARQUET_SCHEMAS\n\n\n"
        "def parquet_arrow_schema(group: str) -> pa.Schema:\n"
        f"{_INDENT}return PARQUET_SCHEMAS[group].arrow_schema()"
    )

    body = f"{_DATACLASSES}\n\n{_GROUP_SCHEMA}\n\n{arrow_map}\n\n"
    body += "\n\n".join(group_schemas)
    body += f"\n\n\n{table}\n\n\n" + "\n\n\n".join(group_functions)
    body += f"\n\n\n{functions}\n"

    header = (
        "from collections.abc import Mapping\n"
        "from dataclasses import dataclass\n"
        "from types import MappingProxyType\n"
        "from typing import Final\n\n"
        "import pyarrow as pa\n"
        f"\nTIME_COLUMN: Final[str] = {TIME_COLUMN!r}\n"
    )
    return render_py(header + "\n" + body)
