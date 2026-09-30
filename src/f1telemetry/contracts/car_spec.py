"""Typed view of ``car_spec.yaml``.

P0 loads only what the invariant harness and the P1 kernel interface need. P1-T1b
extends this with the full curve/array extraction; the shape below (validate on load,
hand plain numbers to the kernel) is already the shape P1 needs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

import yaml

from f1telemetry.contracts.channels import ContractError, repo_root

__all__ = ["AeroCurve", "CarSpec", "car_spec_path", "load_car_spec", "provenance_audit"]

_TOP_LEVEL_SECTIONS: Final[tuple[str, ...]] = (
    "constants",
    "mass",
    "aero",
    "powertrain",
    "gearbox",
    "tyres",
    "chassis",
)
_NEEDS_SOURCE_DATE: Final[frozenset[str]] = frozenset({"provisional", "synthesised", "plan"})


def car_spec_path() -> Path:
    return repo_root() / "car_spec.yaml"


@dataclass(frozen=True, slots=True)
class AeroCurve:
    """A validated speed-indexed coefficient curve.

    P0 checks only the shape: one coefficient per speed, speeds strictly increasing.
    Evaluating the curve - which interpolation, if any - is P1's call, so nothing here
    decides it.
    """

    speed_m_s: tuple[float, ...]
    value: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CarSpec:
    """Provisional P0 car spec. Every field is a placeholder pending P1 calibration."""

    spec: Mapping[str, Any]
    mass_kg: float
    gravity_m_s2: float
    air_density_kg_m3: float
    reference_area_m2: float
    ice_peak_power_kw: float
    mgu_k_peak_power_kw: float
    power_split_ice: float
    rev_limit_rpm: float
    gear_ratios: tuple[float, ...]
    final_drive: float
    rolling_radius_m: float
    torque_curve: tuple[tuple[float, float], ...] = field(default=())
    cl_curve: AeroCurve | None = None
    cd_curve: AeroCurve | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def provenance(self) -> str:
        return str(self.spec.get("provenance", "unknown"))

    @property
    def source_date(self) -> str:
        return str(self.spec.get("source_date", "unknown"))

    @property
    def weight_n(self) -> float:
        return self.mass_kg * self.gravity_m_s2

    def wheel_rpm_at(self, gear: int, ice_rpm: float) -> float:
        if not 1 <= gear <= len(self.gear_ratios):
            raise ContractError(f"gear {gear} outside 1..{len(self.gear_ratios)}")
        return ice_rpm / (self.gear_ratios[gear - 1] * self.final_drive)

    def speed_at(self, gear: int, ice_rpm: float) -> float:
        return self.wheel_rpm_at(gear, ice_rpm) * 2.0 * math.pi * self.rolling_radius_m / 60.0


def _number(node: Any, where: str) -> float:
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        raise ContractError(f"{where}: expected a number, got {node!r}")
    value = float(node)
    if not math.isfinite(value):
        raise ContractError(f"{where}: expected a finite number, got {node!r}")
    return value


def _section(root: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    node = root.get(key)
    if not isinstance(node, Mapping):
        raise ContractError(f"car_spec: section {key!r} missing or not a mapping")
    return node


def _curve(entries: Any, where: str, coefficient: str) -> AeroCurve:
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)) or not entries:
        raise ContractError(f"{where}: expected a non-empty list of {{speed_m_s, {coefficient}}}")
    speeds: list[float] = []
    values: list[float] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ContractError(f"{where}[{index}]: expected a mapping, got {entry!r}")
        if coefficient not in entry:
            raise ContractError(
                f"{where}[{index}]: expected key {coefficient!r}, got {sorted(map(str, entry))}"
            )
        speeds.append(_number(entry["speed_m_s"], f"{where}[{index}].speed_m_s"))
        values.append(_number(entry[coefficient], f"{where}[{index}].{coefficient}"))
    if any(b <= a for a, b in pairwise(speeds)):
        raise ContractError(f"{where}: speed_m_s must be strictly increasing")
    return AeroCurve(speed_m_s=tuple(speeds), value=tuple(values))


def load_car_spec(path: Path | None = None) -> CarSpec:
    source = car_spec_path() if path is None else Path(path)
    if not source.is_file():
        raise ContractError(f"car spec not found: {source}")
    root = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(root, Mapping):
        raise ContractError(f"{source}: expected a top-level mapping")

    spec = _section(root, "spec")
    missing_sections = [s for s in _TOP_LEVEL_SECTIONS if s not in root]
    if missing_sections:
        raise ContractError(f"{source}: missing section(s) {missing_sections}")

    constants = _section(root, "constants")
    mass = _section(root, "mass")
    aero = _section(root, "aero")
    powertrain = _section(root, "powertrain")
    ice = _section(powertrain, "ice")
    gearbox = _section(root, "gearbox")
    tyres = _section(root, "tyres")

    ratios_raw = gearbox.get("ratios")
    if not isinstance(ratios_raw, Sequence) or isinstance(ratios_raw, (str, bytes)):
        raise ContractError("car_spec: gearbox.ratios must be a non-empty list")
    if not ratios_raw:
        raise ContractError("car_spec: gearbox.ratios must be a non-empty list")
    ratios = tuple(_number(r, f"car_spec: gearbox.ratios[{i}]") for i, r in enumerate(ratios_raw))
    if any(b >= a for a, b in pairwise(ratios)):
        raise ContractError("car_spec: gearbox.ratios must be strictly decreasing")

    torque_entries = ice.get("torque_curve_nm")
    if not isinstance(torque_entries, Sequence) or isinstance(torque_entries, (str, bytes)):
        raise ContractError("car_spec: powertrain.ice.torque_curve_nm must be a list")
    torque_curve: list[tuple[float, float]] = []
    for index, entry in enumerate(torque_entries):
        if not isinstance(entry, Mapping):
            raise ContractError(f"car_spec: powertrain.ice.torque_curve_nm[{index}] not a mapping")
        torque_curve.append(
            (
                _number(entry.get("rpm"), f"car_spec: torque_curve_nm[{index}].rpm"),
                _number(entry.get("torque_nm"), f"car_spec: torque_curve_nm[{index}].torque_nm"),
            )
        )
    if any(b[0] <= a[0] for a, b in pairwise(torque_curve)):
        raise ContractError("car_spec: ICE torque curve rpm must be strictly increasing")

    rolling_radius = _number(tyres.get("rolling_radius_m"), "car_spec: tyres.rolling_radius_m")
    if rolling_radius <= 0.0:
        raise ContractError("car_spec: tyres.rolling_radius_m must be > 0")

    return CarSpec(
        spec=spec,
        mass_kg=_number(mass.get("total_kg"), "car_spec: mass.total_kg"),
        gravity_m_s2=_number(constants.get("gravity_m_s2"), "car_spec: constants.gravity_m_s2"),
        air_density_kg_m3=_number(
            constants.get("air_density_kg_m3"), "car_spec: constants.air_density_kg_m3"
        ),
        reference_area_m2=_number(
            aero.get("reference_area_m2"), "car_spec: aero.reference_area_m2"
        ),
        ice_peak_power_kw=_number(
            ice.get("peak_power_kw"), "car_spec: powertrain.ice.peak_power_kw"
        ),
        mgu_k_peak_power_kw=_number(
            _section(powertrain, "mgu_k").get("peak_power_kw"),
            "car_spec: powertrain.mgu_k.peak_power_kw",
        ),
        power_split_ice=_number(
            powertrain.get("power_split_ice"), "car_spec: powertrain.power_split_ice"
        ),
        rev_limit_rpm=_number(ice.get("rev_limit_rpm"), "car_spec: powertrain.ice.rev_limit_rpm"),
        gear_ratios=ratios,
        final_drive=_number(gearbox.get("final_drive"), "car_spec: gearbox.final_drive"),
        rolling_radius_m=rolling_radius,
        torque_curve=tuple(torque_curve),
        cl_curve=_curve(aero.get("cl_curve"), "car_spec: aero.cl_curve", "cl"),
        cd_curve=_curve(aero.get("cd_curve"), "car_spec: aero.cd_curve", "cd"),
        raw=root,
    )


def _audit_provenance(node: Mapping[str, Any], path: str) -> list[str]:
    """Findings for one mapping, plus every nested mapping that declares its own provenance.

    A nested block that carries a ``provenance`` line is making its own claim and owes the
    same ``source_date`` a top-level section does, so it is audited as a section in its own
    right and reported at its dotted path. A nested block with no provenance line - a
    parameter group like ``powertrain.ice.turbo_lag`` - is covered by its parent's line and
    is left alone.
    """
    findings: list[str] = []
    provenance = node.get("provenance")
    if provenance is None:
        findings.append(f"{path}: no provenance line")
    elif str(provenance) in _NEEDS_SOURCE_DATE and node.get("source_date") is None:
        findings.append(f"{path}: provenance {provenance!r} requires a source_date")
    for key, value in node.items():
        if isinstance(value, Mapping) and "provenance" in value:
            findings.extend(_audit_provenance(value, f"{path}.{key}"))
    return findings


def provenance_audit(root: Mapping[str, Any]) -> list[str]:
    """Report sections that break the standing rule 'every coefficient gets a provenance line'.

    Covers the top-level sections and any nested mapping that declares its own provenance
    (notably ``powertrain.ice`` and ``powertrain.mgu_k``), reported at their dotted path.
    Returns a list of human-readable findings; empty means the rule holds.
    """
    findings: list[str] = []
    for key in _TOP_LEVEL_SECTIONS:
        section = root.get(key)
        if not isinstance(section, Mapping):
            findings.append(f"{key}: section missing")
            continue
        findings.extend(_audit_provenance(section, key))
    spec = root.get("spec")
    if isinstance(spec, Mapping):
        if spec.get("source_date") is None:
            findings.append("spec: no source_date")
        if spec.get("calibration_status") not in {None, "draft", "uncalibrated"}:
            findings.append(
                f"spec: calibration_status is {spec.get('calibration_status')!r}; P0 must not "
                "ship a car spec that claims to be calibrated"
            )
    return findings
