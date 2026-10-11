"""Emit the Python channel registry and per-group dataclasses (P0-T4)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from f1telemetry.codegen._emit import Raw, class_name, iter_expanded, py_repr, render_py
from f1telemetry.contracts.channels import (
    CORNERS,
    DTYPES,
    FAULT_TYPES,
    NOISE_MODELS,
    Channel,
    ChannelContract,
)

__all__ = ["emit_python_channels"]

_INDENT = " " * 4


def _literal(name: str, values: Iterable[str]) -> str:
    body = ", ".join(py_repr(v) for v in values)
    return f"{name}: TypeAlias = Literal[{body}]"


def _fields(channel: Channel) -> list[tuple[str, object]]:
    return [
        ("name", channel.name),
        ("base_name", channel.base_name),
        ("group", channel.group),
        ("unit", channel.unit),
        ("dtype", channel.dtype),
        ("rate_hz", channel.rate_hz),
        ("event", channel.event),
        ("range_min", channel.range_min),
        ("range_max", channel.range_max),
        ("noise_model", channel.noise_model),
        ("sigma", channel.sigma),
        (
            "quantisation",
            None
            if channel.quantisation is None
            else Raw(
                f"Quantisation(bits={channel.quantisation.bits}, "
                f"full_scale={channel.quantisation.full_scale!r})"
            ),
        ),
        ("fault_eligible", tuple(channel.fault_eligible)),
        ("corners", tuple(channel.corners)),
        ("corner", channel.corner),
        ("description", channel.description),
        ("source", channel.source),
    ]


def _channel_literal(channel: Channel, indent: str) -> str:
    fields = _fields(channel)
    width = max(len(key) for key, _ in fields) + 1
    hang = " " * (len(indent) + len("Channel("))
    lines: list[str] = []
    for index, (key, value) in enumerate(fields):
        rendered = value.text if isinstance(value, Raw) else py_repr(value)
        closing = ")" if index == len(fields) - 1 else ""
        lines.append(f"{key.ljust(width)} = {rendered}{closing},")
    return "\n".join([f"{indent}Channel({lines[0]}", *(f"{hang}{line}" for line in lines[1:])])


def _emit_dataclass(group: str, members: Sequence[Channel]) -> str:
    name = class_name(group)
    width = max(len(c.name) for c in members) + 1
    fields = "\n".join(f"{_INDENT}{c.name.ljust(width)}: float" for c in members)
    joined = ", ".join(f'"{c.name}"' for c in members)
    return (
        f"@dataclass(frozen=True, slots=True)\n"
        f"class {name}:\n"
        f'    """One frame of every {group.replace("_", " ")} channel. '
        f'Field order matches the contract."""\n\n'
        f"{fields}\n\n"
        f"{_INDENT}@classmethod\n"
        f"{_INDENT}def from_mapping(cls, values: Mapping[str, float]) -> Self:\n"
        f'{_INDENT * 2}"""Build from a channel-name mapping; a missing channel is an error."""\n'
        f"{_INDENT * 2}return cls(**{{name: values[name] for name in cls.field_names()}})\n\n"
        f"{_INDENT}@classmethod\n"
        f"{_INDENT}def field_names(cls) -> tuple[str, ...]:\n"
        f"{_INDENT * 2}return ({joined},)\n\n"
        f"{_INDENT}def as_mapping(self) -> dict[str, float]:\n"
        f"{_INDENT * 2}return {{name: getattr(self, name) for name in self.field_names()}}\n"
    )


def emit_python_channels(contract: ChannelContract) -> str:
    """Render ``generated/channels.py`` from the loaded contract."""
    blocks: list[str] = []
    blocks.append(f"CONTRACT_VERSION: Final[str] = {py_repr(contract.version)}")
    blocks.append(f"CONTRACT_DESCRIPTION: Final[str] = {py_repr(contract.description)}")
    blocks.append(f"CORNERS: Final[tuple[str, ...]] = {py_repr(CORNERS)}")
    blocks.append(f"FAULT_TYPES: Final[tuple[str, ...]] = {py_repr(FAULT_TYPES)}")
    blocks.append(f"NOISE_MODELS: Final[tuple[str, ...]] = {py_repr(NOISE_MODELS)}")
    blocks.append(f"DTYPES: Final[tuple[str, ...]] = {py_repr(DTYPES)}")
    blocks.append(f"GROUPS: Final[tuple[str, ...]] = {py_repr(contract.groups)}")
    rates = tuple(sorted({int(c.rate_hz) for c in contract.channels if c.rate_hz is not None}))
    blocks.append(f"RATES_HZ: Final[tuple[int, ...]] = {py_repr(rates)}")
    blocks.append(f"SAMPLES_PER_SECOND: Final[float] = {contract.samples_per_second()!r}")

    aliases = [
        _literal("NoiseModel", NOISE_MODELS),
        _literal("ChannelGroup", contract.groups),
        _literal("Corner", CORNERS),
        _literal("ChannelName", contract.names),
        _literal(
            "CornerChannelName",
            (c.name for c in contract.channels if c.is_corner_channel),
        ),
        _literal(
            "QuantisedBaseName",
            (b[0].base_name for b in iter_expanded(contract) if b[0].quantisation is not None),
        ),
    ]
    blocks.append("\n".join(aliases))

    dataclass_defs = (
        "@dataclass(frozen=True, slots=True)\n"
        "class Quantisation:\n"
        '    """Analogue-to-digital resolution of the emulated sensor."""\n\n'
        f"{_INDENT}bits: int\n"
        f"{_INDENT}full_scale: float\n\n"
        f"{_INDENT}@property\n"
        f"{_INDENT}def step(self) -> float:\n"
        f"{_INDENT * 2}return self.full_scale / float(1 << (self.bits - 1))\n\n"
        f"{_INDENT}def to_dict(self) -> dict[str, float]:\n"
        f"{_INDENT * 2}bits = self.bits\n"
        f"{_INDENT * 2}full_scale = self.full_scale\n"
        f"{_INDENT * 2}return {{'bits': bits, 'full_scale': full_scale, 'step': self.step}}\n"
    )
    channel_def = (
        "@dataclass(frozen=True, slots=True)\n"
        "class Channel:\n"
        '    """One addressable channel. `name` is unique across the contract."""\n\n'
        f"{_INDENT}name: ChannelName\n"
        f"{_INDENT}base_name: str\n"
        f"{_INDENT}group: ChannelGroup\n"
        f"{_INDENT}unit: str\n"
        f"{_INDENT}dtype: str\n"
        f"{_INDENT}rate_hz: float | None\n"
        f"{_INDENT}event: bool\n"
        f"{_INDENT}range_min: float\n"
        f"{_INDENT}range_max: float\n"
        f"{_INDENT}noise_model: NoiseModel\n"
        f"{_INDENT}sigma: float\n"
        f"{_INDENT}quantisation: Quantisation | None\n"
        f"{_INDENT}fault_eligible: tuple[str, ...]\n"
        f"{_INDENT}corners: tuple[Corner, ...]\n"
        f"{_INDENT}corner: Corner | None\n"
        f"{_INDENT}description: str\n"
        f"{_INDENT}source: str\n\n"
        f"{_INDENT}@property\n"
        f"{_INDENT}def period_s(self) -> float | None:\n"
        f"{_INDENT * 2}return None if self.rate_hz is None else 1.0 / self.rate_hz\n\n"
        f"{_INDENT}@property\n"
        f"{_INDENT}def is_corner_channel(self) -> bool:\n"
        f"{_INDENT * 2}return self.corner is not None\n\n"
        f"{_INDENT}def to_dict(self) -> dict[str, object]:\n"
        f"{_INDENT * 2}quant = None if self.quantisation is None else self.quantisation.to_dict()\n"
        f"{_INDENT * 2}return {{\n"
        f'{_INDENT * 3}"name": self.name,\n'
        f'{_INDENT * 3}"base_name": self.base_name,\n'
        f'{_INDENT * 3}"group": self.group,\n'
        f'{_INDENT * 3}"unit": self.unit,\n'
        f'{_INDENT * 3}"dtype": self.dtype,\n'
        f'{_INDENT * 3}"rate_hz": self.rate_hz,\n'
        f'{_INDENT * 3}"event": self.event,\n'
        f'{_INDENT * 3}"range": [self.range_min, self.range_max],\n'
        f'{_INDENT * 3}"noise_model": self.noise_model,\n'
        f'{_INDENT * 3}"sigma": self.sigma,\n'
        f'{_INDENT * 3}"quantise": quant,\n'
        f'{_INDENT * 3}"fault_eligible": list(self.fault_eligible),\n'
        f'{_INDENT * 3}"corners": list(self.corners),\n'
        f'{_INDENT * 3}"corner": self.corner,\n'
        f'{_INDENT * 3}"description": self.description,\n'
        f'{_INDENT * 3}"source": self.source,\n'
        f"{_INDENT * 2}}}"
    )

    channel_literals = "\n".join(
        _channel_literal(channel, _INDENT) for channel in contract.channels
    )
    registry = (
        f"_CHANNELS: Final[tuple[Channel, ...]] = (\n{channel_literals}\n)\n\n"
        f"CHANNELS: Final[Mapping[str, Channel]] = MappingProxyType(\n"
        f"{_INDENT}{{channel.name: channel for channel in _CHANNELS}}\n"
        f")\n\n"
        f"CHANNELS_BY_GROUP: Final[Mapping[str, tuple[Channel, ...]]] = MappingProxyType(\n"
        f"{_INDENT}{{\n"
        + "".join(
            f"{_INDENT * 2}{py_repr(group)}: (\n"
            + "".join(
                f"{_INDENT * 3}CHANNELS[{py_repr(c.name)}],\n"
                for c in contract.group_members(group)
            )
            + f"{_INDENT * 2}),\n"
            for group in contract.groups
        )
        + f"{_INDENT}}}\n"
        f")"
    )

    group_classes = "\n\n".join(
        _emit_dataclass(group, contract.group_members(group)) for group in contract.groups
    )

    contract_class = (
        "@dataclass(frozen=True, slots=True)\n"
        "class Contract:\n"
        '    """Runtime handle on the contract, for callers that need to re-read the YAML."""\n\n'
        f"{_INDENT}version: str\n"
        f"{_INDENT}channels: Mapping[str, Channel]\n\n"
        f"{_INDENT}def __len__(self) -> int:\n"
        f"{_INDENT * 2}return len(self.channels)\n\n"
        f"{_INDENT}def channel_names(self) -> tuple[str, ...]:\n"
        f"{_INDENT * 2}return tuple(self.channels)\n\n"
        f"{_INDENT}def by_name(self, name: str) -> Channel:\n"
        f"{_INDENT * 2}return self.channels[name]\n\n"
        f"{_INDENT}def group(self, group: str) -> tuple[Channel, ...]:\n"
        f"{_INDENT * 2}return CHANNELS_BY_GROUP[group]\n\n"
        f"{_INDENT}def base_names(self) -> tuple[str, ...]:\n"
        f"{_INDENT * 2}seen: dict[str, None] = {{}}\n"
        f"{_INDENT * 2}for channel in self.channels.values():\n"
        f"{_INDENT * 3}seen.setdefault(channel.base_name, None)\n"
        f"{_INDENT * 2}return tuple(seen)\n\n"
        f"{_INDENT}def corner_channels(self, base_name: str) -> tuple[Channel, ...]:\n"
        f"{_INDENT * 2}matched = (c for c in self.channels.values() if c.base_name == base_name)\n"
        f"{_INDENT * 2}return tuple(matched)\n\n"
        f"{_INDENT}def corner_map(self, base_name: str) -> dict[Corner, Channel]:\n"
        f"{_INDENT * 2}found: dict[Corner, Channel] = {{}}\n"
        f"{_INDENT * 2}for channel in self.corner_channels(base_name):\n"
        f"{_INDENT * 3}if channel.corner is not None:\n"
        f"{_INDENT * 4}found[channel.corner] = channel\n"
        f"{_INDENT * 2}return found\n\n"
        f"{_INDENT}def corner_values(\n"
        f"{_INDENT * 2}self, values: Mapping[str, float], base_name: str\n"
        f"{_INDENT}) -> dict[str, float]:\n"
        f"{_INDENT * 2}channels = self.corner_channels(base_name)\n"
        f"{_INDENT * 2}return {{str(c.corner).lower(): values[c.name] for c in channels}}\n\n"
        f"{_INDENT}def samples_per_second(self) -> float:\n"
        f"{_INDENT * 2}return SAMPLES_PER_SECOND\n\n"
        f"{_INDENT}def event_channels(self) -> tuple[Channel, ...]:\n"
        f"{_INDENT * 2}return tuple(c for c in self.channels.values() if c.event)"
    )

    loader = (
        "def load_contract(channels_yaml: Path | None = None) -> Contract:\n"
        '    """Re-read `channels.yaml` and assert it still matches this generated file.\n\n'
        "    Raises if the contract has drifted, which means codegen was not re-run.\n"
        '    """\n'
        f"{_INDENT}from f1telemetry.contracts.channels import (\n"
        f"{_INDENT * 2}ContractError,\n"
        f"{_INDENT * 2}load_channel_contract,\n"
        f"{_INDENT})\n\n"
        f"{_INDENT}loaded = load_channel_contract(channels_yaml)\n"
        f"{_INDENT}if loaded.version != CONTRACT_VERSION:\n"
        f"{_INDENT * 2}msg = f'contract version {{loaded.version}} != '\n"
        f"{_INDENT * 2}msg += f'generated {{CONTRACT_VERSION}}'\n"
        f"{_INDENT * 2}raise ContractError(msg)\n"
        f"{_INDENT}if loaded.names != tuple(CHANNELS):\n"
        f"{_INDENT * 2}msg = 'channels.yaml no longer matches the generated registry; '\n"
        f"{_INDENT * 2}msg += 'run uv run f1-codegen'\n"
        f"{_INDENT * 2}raise ContractError(msg)\n"
        f"{_INDENT}return Contract(version=loaded.version, channels=CHANNELS)"
    )

    body = "\n".join(blocks)
    body += (
        "\n\n"
        + dataclass_defs
        + "\n\n"
        + channel_def
        + "\n\n\n"
        + registry
        + "\n\n\n"
        + group_classes
        + "\n\n\n"
        + contract_class
        + "\n\n\n"
        + loader
        + "\n"
    )
    header = (
        "from collections.abc import Mapping\n"
        "from dataclasses import dataclass\n"
        "from pathlib import Path\n"
        "from types import MappingProxyType\n"
        "from typing import Final, Literal, Self, TypeAlias\n"
    )
    return render_py(header + "\n" + body)
