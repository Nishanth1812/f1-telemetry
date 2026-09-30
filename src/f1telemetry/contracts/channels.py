"""Typed view of ``channels.yaml`` - the project contract (PLAN.md section 5.2).

The YAML file is hand-authored; this module is its only schema. Two products come
out of it and nothing else in the repository is allowed to restate field metadata:

* ``Channel`` - one addressable channel, after per-corner expansion.
* ``ChannelContract`` - the whole dictionary, grouped and indexed.

Per PLAN.md section 3 the ``GroundTruth`` side channel is deliberately *not* here:
it carries physics state (Fx, Fy, kappa, ...) that no real sensor exposes. It lives
in :mod:`f1telemetry.testing.records`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

__all__ = [
    "CORNERS",
    "DTYPES",
    "FAULT_TYPES",
    "NOISE_MODELS",
    "Channel",
    "ChannelContract",
    "ContractError",
    "Quantisation",
    "channels_yaml_path",
    "load_channel_contract",
    "repo_root",
]

CORNERS: Final[tuple[str, ...]] = ("FL", "FR", "RL", "RR")
FAULT_TYPES: Final[tuple[str, ...]] = (
    "dropout",
    "freeze",
    "spike",
    "step",
    "gain",
    "noise",
    "quantise",
    "swap",
    "saturate",
    "stale",
)
NOISE_MODELS: Final[tuple[str, ...]] = ("gaussian", "uniform", "none")
DTYPES: Final[tuple[str, ...]] = (
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "uint8",
    "uint16",
    "uint32",
    "bool",
)
INTEGER_DTYPES: Final[frozenset[str]] = frozenset(
    d for d in DTYPES if d != "bool" and not d.startswith("float")
)

_REQUIRED_KEYS: Final[tuple[str, ...]] = (
    "name",
    "group",
    "unit",
    "dtype",
    "noise_model",
)


class ContractError(ValueError):
    """Raised when ``channels.yaml`` violates the contract schema."""


def repo_root() -> Path:
    """Repository root, located by walking up from this file until the marker is found."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "channels.yaml").is_file():
            return candidate
    msg = "could not locate repository root (no channels.yaml found above src/)"
    raise ContractError(msg)


def channels_yaml_path() -> Path:
    return repo_root() / "channels.yaml"


@dataclass(frozen=True, slots=True)
class Quantisation:
    """Analogue-to-digital resolution of the emulated sensor."""

    bits: int
    full_scale: float

    @property
    def step(self) -> float:
        return self.full_scale / float(1 << (self.bits - 1))

    def to_dict(self) -> dict[str, Any]:
        return {"bits": self.bits, "full_scale": self.full_scale, "step": self.step}


@dataclass(frozen=True, slots=True)
class Channel:
    """One addressable channel. ``name`` is unique across the whole contract."""

    name: str
    base_name: str
    group: str
    unit: str
    dtype: str
    rate_hz: float | None
    event: bool
    range_min: float
    range_max: float
    noise_model: str
    sigma: float
    quantisation: Quantisation | None
    fault_eligible: tuple[str, ...]
    corners: tuple[str, ...]
    corner: str | None
    description: str
    source: str

    @property
    def is_corner_channel(self) -> bool:
        return self.corner is not None

    @property
    def is_integer(self) -> bool:
        return self.dtype in INTEGER_DTYPES

    @property
    def period_s(self) -> float | None:
        return None if self.rate_hz is None else 1.0 / self.rate_hz

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_name": self.base_name,
            "group": self.group,
            "unit": self.unit,
            "dtype": self.dtype,
            "rate_hz": self.rate_hz,
            "event": self.event,
            "range": [self.range_min, self.range_max],
            "noise_model": self.noise_model,
            "sigma": self.sigma,
            "quantise": None if self.quantisation is None else self.quantisation.to_dict(),
            "fault_eligible": list(self.fault_eligible),
            "corners": list(self.corners),
            "corner": self.corner,
            "description": self.description,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class ChannelContract:
    """The whole channel dictionary, with corner expansion already applied."""

    version: str
    description: str
    channels: tuple[Channel, ...]
    groups: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.channels)

    def __iter__(self) -> Any:
        return iter(self.channels)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.channels)

    def by_name(self, name: str) -> Channel:
        for channel in self.channels:
            if channel.name == name:
                return channel
        raise KeyError(name)

    def group_members(self, group: str) -> tuple[Channel, ...]:
        return tuple(c for c in self.channels if c.group == group)

    @property
    def base_names(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for channel in self.channels:
            seen.setdefault(channel.base_name, None)
        return tuple(seen)

    def samples_per_second(self) -> float:
        """Contract sizing figure; event channels contribute nothing when unscheduled."""
        return sum(0.0 if c.rate_hz is None else c.rate_hz for c in self.channels)


def _require_mapping(node: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(node, Mapping):
        raise ContractError(f"{where}: expected a mapping, got {type(node).__name__}")
    return node


def _require_float(node: Any, where: str) -> float:
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        raise ContractError(f"{where}: expected a number, got {node!r}")
    value = float(node)
    if not math.isfinite(value):
        raise ContractError(f"{where}: expected a finite number, got {node!r}")
    return value


def _parse_quantisation(node: Any, where: str) -> Quantisation:
    mapping = _require_mapping(node, where)
    missing = {"bits", "full_scale"} - set(mapping)
    if missing:
        raise ContractError(f"{where}: missing key(s) {sorted(missing)}")
    bits = mapping["bits"]
    if isinstance(bits, bool) or not isinstance(bits, int) or not 1 <= bits <= 32:
        raise ContractError(f"{where}.bits: expected an int in 1..32, got {bits!r}")
    full_scale = _require_float(mapping["full_scale"], f"{where}.full_scale")
    if full_scale <= 0.0:
        raise ContractError(f"{where}.full_scale: expected > 0, got {full_scale}")
    return Quantisation(bits=bits, full_scale=full_scale)


def _parse_faults(node: Any, where: str) -> tuple[str, ...]:
    if not isinstance(node, Sequence) or isinstance(node, (str, bytes)):
        raise ContractError(f"{where}: expected a list, got {node!r}")
    unknown = [f for f in node if f not in FAULT_TYPES]
    if unknown:
        raise ContractError(
            f"{where}: unknown fault type(s) {unknown}; allowed {list(FAULT_TYPES)}"
        )
    ordered = tuple(f for f in FAULT_TYPES if f in set(node))
    return ordered


def _expand(spec: Mapping[str, Any], version: str) -> list[Channel]:
    missing = [key for key in _REQUIRED_KEYS if key not in spec]
    if missing:
        raise ContractError(f"channel {spec.get('name')!r}: missing key(s) {missing}")

    name = spec["name"]
    where = f"channel {name!r}"
    if not isinstance(name, str) or not name.isidentifier() or not name.islower():
        raise ContractError(
            f"{where}.name: expected a lowercase snake_case identifier, got {name!r}"
        )
    if "." in name or "-" in name:
        raise ContractError(f"{where}.name: '.' and '-' are not valid in a channel name")

    group = spec["group"]
    if not isinstance(group, str) or not group:
        raise ContractError(f"{where}.group: expected a non-empty string, got {group!r}")

    unit = spec["unit"]
    if not isinstance(unit, str) or not unit:
        raise ContractError(f"{where}.unit: expected a non-empty string, got {unit!r}")

    dtype = spec["dtype"]
    if dtype not in DTYPES:
        raise ContractError(f"{where}.dtype: unknown dtype {dtype!r}; allowed {list(DTYPES)}")

    event = spec.get("event", False)
    if not isinstance(event, bool):
        raise ContractError(f"{where}.event: expected a bool, got {event!r}")

    rate_raw = spec.get("rate_hz", None)
    if event:
        if rate_raw is not None:
            raise ContractError(f"{where}.rate_hz: event channels must set rate_hz: null")
        rate_hz: float | None = None
    else:
        if rate_raw is None:
            raise ContractError(f"{where}.rate_hz: required, or set event: true with rate_hz: null")
        rate_hz = _require_float(rate_raw, f"{where}.rate_hz")
        if rate_hz <= 0.0:
            raise ContractError(f"{where}.rate_hz: expected > 0, got {rate_hz}")

    bounds = spec.get("range", None)
    if bounds is None:
        range_min, range_max = 0.0, 1.0
    else:
        if not isinstance(bounds, Sequence) or isinstance(bounds, (str, bytes)) or len(bounds) != 2:
            raise ContractError(f"{where}.range: expected [min, max], got {bounds!r}")
        range_min = _require_float(bounds[0], f"{where}.range[0]")
        range_max = _require_float(bounds[1], f"{where}.range[1]")
        if range_min > range_max:
            raise ContractError(f"{where}.range: min {range_min} > max {range_max}")

    noise_model = spec["noise_model"]
    if noise_model not in NOISE_MODELS:
        raise ContractError(
            f"{where}.noise_model: unknown model {noise_model!r}; allowed {list(NOISE_MODELS)}"
        )
    sigma = _require_float(spec.get("sigma", 0.0), f"{where}.sigma")
    if sigma < 0.0:
        raise ContractError(f"{where}.sigma: expected >= 0, got {sigma}")
    if noise_model == "none" and sigma != 0.0:
        raise ContractError(f"{where}.sigma: noise_model 'none' requires sigma: 0")

    quant_raw = spec.get("quantise", None)
    quantisation = (
        None if quant_raw is None else _parse_quantisation(quant_raw, f"{where}.quantise")
    )
    if quantisation is not None and quantisation.full_scale < abs(range_max):
        raise ContractError(
            f"{where}.quantise.full_scale: {quantisation.full_scale} does not cover "
            f"range max {range_max}; a fault-free channel would sit at full scale"
        )

    faults_raw = spec.get("fault_eligible", list(FAULT_TYPES))
    fault_eligible = _parse_faults(faults_raw, f"{where}.fault_eligible")
    if "swap" in fault_eligible and not spec.get("corners"):
        raise ContractError(f"{where}.fault_eligible: 'swap' needs a non-empty corners list")

    corners_raw = spec.get("corners", None)
    if corners_raw is None:
        corners: tuple[str, ...] = ()
    else:
        if not isinstance(corners_raw, Sequence) or isinstance(corners_raw, (str, bytes)):
            raise ContractError(f"{where}.corners: expected a list, got {corners_raw!r}")
        bad = [c for c in corners_raw if c not in CORNERS]
        if bad:
            raise ContractError(
                f"{where}.corners: unknown corner(s) {bad}; allowed {list(CORNERS)}"
            )
        if len(set(corners_raw)) != len(corners_raw):
            raise ContractError(f"{where}.corners: duplicate corner(s) in {list(corners_raw)}")
        corners = tuple(c for c in CORNERS if c in set(corners_raw))

    description = str(spec.get("description", ""))
    source = str(spec.get("source", f"channels.yaml v{version}"))

    def build(channel_name: str, corner: str | None) -> Channel:
        return Channel(
            name=channel_name,
            base_name=name,
            group=group,
            unit=unit,
            dtype=dtype,
            rate_hz=rate_hz,
            event=event,
            range_min=range_min,
            range_max=range_max,
            noise_model=noise_model,
            sigma=sigma,
            quantisation=quantisation,
            fault_eligible=fault_eligible,
            corners=corners,
            corner=corner,
            description=description,
            source=source,
        )

    if not corners:
        return [build(name, None)]
    return [build(f"{name}_{corner.lower()}", corner) for corner in corners]


def load_channel_contract(path: Path | None = None) -> ChannelContract:
    """Load, validate and corner-expand ``channels.yaml``."""
    source = channels_yaml_path() if path is None else Path(path)
    if not source.is_file():
        raise ContractError(f"channel contract not found: {source}")
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _require_mapping(document, str(source))
    version = str(root.get("version", "0"))
    entries = root.get("channels")
    if not isinstance(entries, list) or not entries:
        raise ContractError(f"{source}: 'channels' must be a non-empty list")

    channels: list[Channel] = []
    groups: dict[str, None] = {}
    for index, entry in enumerate(entries):
        mapping = _require_mapping(entry, f"{source}: channels[{index}]")
        expanded = _expand(mapping, version)
        channels.extend(expanded)
        groups.setdefault(str(mapping["group"]), None)

    seen: dict[str, None] = {}
    duplicates: list[str] = []
    for channel in channels:
        if channel.name in seen:
            duplicates.append(channel.name)
        seen[channel.name] = None
    if duplicates:
        raise ContractError(f"{source}: duplicate channel name(s) {sorted(set(duplicates))}")

    return ChannelContract(
        version=version,
        description=str(root.get("description", "")),
        channels=tuple(channels),
        groups=tuple(groups),
    )
