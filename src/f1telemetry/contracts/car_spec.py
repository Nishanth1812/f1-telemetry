"""Typed view of ``car_spec.yaml`` - the P1 physics configuration (PLAN.md section 5).

Two things leave this module for the kernel, and nothing else does:

* :class:`CarSpec` - the validated view, plus :meth:`CarSpec.citations`, which is the
  machine-checkable record of which clause each regulation value was read from.
* :class:`KernelConfig` - flat scalars and float64 arrays, built outside Numba per
  ``PLAN.md`` section 4.1 rules 1 and 3. A kernel receives numbers; it never sees YAML.

Two rules shape the file schema and the loader together:

**Every number is claimed.** Each value is claimed by exactly one of ``regulation``
(clause + page + quoted text, so a reviewer can check it without opening the PDF) or
``not_regulated`` (why it is not a regulation number). :func:`provenance_audit` fails on an
unclaimed number, which is what stops a synthesised coefficient from being read as a
regulation limit - the failure mode that matters most in this project.

**Everything is validated here.** Masses, densities, ratios and curves are checked in Python
before any array is built, so a bad edit raises ``ContractError`` at the boundary rather than
producing a NaN inside a compiled kernel.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

import numpy as np
import yaml

from f1telemetry.contracts.channels import ContractError, repo_root

__all__ = [
    "AeroCurve",
    "CarSpec",
    "ContractError",
    "KernelConfig",
    "car_spec_path",
    "load_car_spec",
    "provenance_audit",
]

_TOP_LEVEL_SECTIONS: Final[tuple[str, ...]] = (
    "constants",
    "mass",
    "aero",
    "powertrain",
    "gearbox",
    "tyres",
    "chassis",
    "integration",
)
_NEEDS_SOURCE_DATE: Final[frozenset[str]] = frozenset(
    {"provisional", "synthesised", "plan", "mixed"}
)
_REGULATED_PROVENANCE: Final[frozenset[str]] = frozenset({"regulated", "mixed"})
_CLAIM_BLOCKS: Final[tuple[str, ...]] = ("regulation", "not_regulated")


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
class DeploymentCurve:
    """A validated speed-indexed ERS-K power limit.

    The same shape as :class:`AeroCurve` with different units, kept separate so the unit
    suffix is visible at every use: speed in km/h, limit in kW, exactly as C5.2.8 states them.
    """

    speed_km_h: tuple[float, ...]
    limit_kw: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class KernelConfig:
    """Flat numeric configuration for the longitudinal kernel.

    Scalars and 1-D ``float64`` arrays only, all C-contiguous and writable, so the kernel
    contract of ``PLAN.md`` section 4.1 holds: arrays in, arrays out, no dictionaries, no
    allocation, no string lookups. The caller owns every buffer.
    """

    dt_s: float
    mass_kg: float
    gravity_m_s2: float
    air_density_kg_m3: float
    air_temperature_k: float
    reference_area_m2: float
    ride_height_sensitivity: float
    aero_speed_m_s: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    ice_peak_power_kw: float
    rev_limit_rpm: float
    idle_rpm: float
    torque_rpm: np.ndarray
    torque_nm: np.ndarray
    turbo_lag_collapse_rpm: float
    turbo_lag_multiplier: float
    fuel_energy_flow_max_mj_h: float
    fuel_energy_flow_low_rpm_limit_rpm: float
    fuel_energy_flow_low_rpm_gain: float
    fuel_energy_flow_low_rpm_offset_mj_h: float
    mgu_k_peak_power_kw: float
    mgu_k_torque_limit_nm: float
    ers_speed_km_h: np.ndarray
    ers_limit_kw: np.ndarray
    ers_overtake_speed_km_h: np.ndarray
    ers_overtake_limit_kw: np.ndarray
    store_energy_mj: float
    recharge_limit_mj_per_lap: float
    superclip_s: float
    launch_speed_kmh: float
    power_split_ice: float
    fuel_lhv_kj_kg: float
    gear_ratios: np.ndarray
    final_drive: float
    shift_up_rpm: float
    shift_down_rpm: float
    shift_time_s: float
    rolling_radius_m: float
    wheel_diameter_m: float
    front_width_mm: float
    rear_width_mm: float
    wheelbase_m: float
    overall_width_m: float
    front_weight_fraction: float


@dataclass(frozen=True, slots=True)
class CarSpec:
    """Validated P1 car spec, versioned to an FIA regulation issue."""

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
    dt_s: float
    torque_curve: tuple[tuple[float, float], ...] = field(default=())
    cl_curve: AeroCurve | None = None
    cd_curve: AeroCurve | None = None
    deployment_curve: DeploymentCurve | None = None
    overtake_curve: DeploymentCurve | None = None
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

    def citations(self) -> dict[str, tuple[str, int]]:
        """``dotted.path`` -> ``(clause, page)`` for every value read from the regulation.

        Keyed by the YAML path so a claim can be traced back to the exact block that made it,
        including nested blocks such as ``powertrain.mgu_k``.
        """
        found: dict[str, tuple[str, int]] = {}
        for section in _TOP_LEVEL_SECTIONS:
            block = self.raw.get(section)
            if isinstance(block, Mapping):
                _collect_citations(block, section, found)
        return found

    def wheel_rpm_at(self, gear: int, ice_rpm: float) -> float:
        if not 1 <= gear <= len(self.gear_ratios):
            raise ContractError(f"gear {gear} outside 1..{len(self.gear_ratios)}")
        return ice_rpm / (self.gear_ratios[gear - 1] * self.final_drive)

    def speed_at(self, gear: int, ice_rpm: float) -> float:
        return self.wheel_rpm_at(gear, ice_rpm) * 2.0 * math.pi * self.rolling_radius_m / 60.0

    def kernel_config(self) -> KernelConfig:
        """Validated flat arrays for the kernel.

        Raises ``ContractError`` for any value that is missing, non-finite, or outside the
        range where the physics it feeds is defined. The checks that matter are the ones a
        compiled kernel cannot make: a division by a zero rolling radius, a slip denominator
        that vanishes, a gear count the state vector was not sized for.
        """
        raw = self.raw
        constants = _section(raw, "constants")
        aero = _section(raw, "aero")
        powertrain = _section(raw, "powertrain")
        ice = _section(powertrain, "ice")
        turbo_lag = _section(ice, "turbo_lag")
        mgu_k = _section(powertrain, "mgu_k")
        gearbox = _section(raw, "gearbox")
        tyres = _section(raw, "tyres")
        chassis = _section(raw, "chassis")
        integration = _section(raw, "integration")

        cl = self.cl_curve or AeroCurve(speed_m_s=(0.0,), value=(0.0,))
        cd = self.cd_curve or AeroCurve(speed_m_s=(0.0,), value=(0.0,))
        deployment = self.deployment_curve or DeploymentCurve(speed_km_h=(0.0,), limit_kw=(0.0,))
        overtake = self.overtake_curve or DeploymentCurve(speed_km_h=(0.0,), limit_kw=(0.0,))

        mass_kg = _positive(self.mass_kg, "car_spec: mass.total_kg")
        gravity = _positive(self.gravity_m_s2, "car_spec: constants.gravity_m_s2")
        air_density = _positive(self.air_density_kg_m3, "car_spec: constants.air_density_kg_m3")
        air_temperature = _number(
            integration.get("air_temperature_k", constants.get("air_temperature_k")),
            "car_spec: constants.air_temperature_k",
        )
        reference_area = _positive(self.reference_area_m2, "car_spec: aero.reference_area_m2")
        ride_height = _non_negative(
            aero.get("ride_height_sensitivity"), "car_spec: aero.ride_height_sensitivity"
        )
        _positive_values(cl.value, "car_spec: aero.cl_curve: every cl must be > 0")
        _positive_values(cd.value, "car_spec: aero.cd_curve: every cd must be > 0")

        dt_s = _positive(integration.get("dt_s"), "car_spec: integration.dt_s")
        ice_peak = _positive(ice.get("peak_power_kw"), "car_spec: powertrain.ice.peak_power_kw")
        idle_rpm = _positive(ice.get("idle_rpm"), "car_spec: powertrain.ice.idle_rpm")
        rev_limit = _positive(ice.get("rev_limit_rpm"), "car_spec: powertrain.ice.rev_limit_rpm")
        if idle_rpm >= rev_limit:
            raise ContractError(
                f"car_spec: powertrain.ice.rev_limit_rpm ({rev_limit}) must exceed idle_rpm "
                f"({idle_rpm})"
            )
        turbo_collapse = _positive(
            turbo_lag.get("collapse_below_rpm"),
            "car_spec: powertrain.ice.turbo_lag.collapse_below_rpm",
        )
        turbo_multiplier = _number(
            turbo_lag.get("multiplier_at_collapse"),
            "car_spec: powertrain.ice.turbo_lag.multiplier_at_collapse",
        )
        if not 0.0 < turbo_multiplier <= 1.0:
            raise ContractError(
                "car_spec: powertrain.ice.turbo_lag.multiplier_at_collapse must be in (0, 1], "
                f"got {turbo_multiplier}"
            )

        mgu_k_power = _positive(
            mgu_k.get("peak_power_kw"), "car_spec: powertrain.mgu_k.peak_power_kw"
        )
        store_energy = _positive(
            mgu_k.get("store_energy_mj"), "car_spec: powertrain.mgu_k.store_energy_mj"
        )
        power_split = _number(
            powertrain.get("power_split_ice"), "car_spec: powertrain.power_split_ice"
        )
        if not 0.0 < power_split < 1.0:
            raise ContractError(
                f"car_spec: powertrain.power_split_ice must be in (0, 1), got {power_split}"
            )

        ratios = np.array(self.gear_ratios, dtype=np.float64)
        final_drive = _positive(self.final_drive, "car_spec: gearbox.final_drive")
        shift_up = _positive(gearbox.get("shift_up_rpm"), "car_spec: gearbox.shift_up_rpm")
        shift_down = _positive(gearbox.get("shift_down_rpm"), "car_spec: gearbox.shift_down_rpm")
        shift_time = _non_negative(gearbox.get("shift_time_s"), "car_spec: gearbox.shift_time_s")
        if shift_up > rev_limit:
            raise ContractError(
                f"car_spec: gearbox.shift_up_rpm ({shift_up}) exceeds rev_limit_rpm ({rev_limit})"
            )
        if shift_down >= shift_up:
            raise ContractError(
                f"car_spec: gearbox.shift_down_rpm ({shift_down}) must be below shift_up_rpm "
                f"({shift_up})"
            )

        rolling_radius = _positive(
            tyres.get("rolling_radius_m"), "car_spec: tyres.rolling_radius_m"
        )
        wheel_diameter = _positive(
            tyres.get("wheel_diameter_m"), "car_spec: tyres.wheel_diameter_m"
        )
        wheelbase = _positive(chassis.get("wheelbase_m"), "car_spec: chassis.wheelbase_m")

        return KernelConfig(
            dt_s=dt_s,
            mass_kg=mass_kg,
            gravity_m_s2=gravity,
            air_density_kg_m3=air_density,
            air_temperature_k=air_temperature,
            reference_area_m2=reference_area,
            ride_height_sensitivity=ride_height,
            aero_speed_m_s=np.array(cl.speed_m_s, dtype=np.float64),
            cl=np.array(cl.value, dtype=np.float64),
            cd=np.array(cd.value, dtype=np.float64),
            ice_peak_power_kw=ice_peak,
            rev_limit_rpm=rev_limit,
            idle_rpm=idle_rpm,
            torque_rpm=np.array([rpm for rpm, _ in self.torque_curve], dtype=np.float64),
            torque_nm=np.array([nm for _, nm in self.torque_curve], dtype=np.float64),
            turbo_lag_collapse_rpm=turbo_collapse,
            turbo_lag_multiplier=turbo_multiplier,
            fuel_energy_flow_max_mj_h=_positive(
                ice.get("fuel_energy_flow_max_mj_h"),
                "car_spec: powertrain.ice.fuel_energy_flow_max_mj_h",
            ),
            fuel_energy_flow_low_rpm_limit_rpm=_positive(
                ice.get("fuel_energy_flow_low_rpm_limit_rpm"),
                "car_spec: powertrain.ice.fuel_energy_flow_low_rpm_limit_rpm",
            ),
            fuel_energy_flow_low_rpm_gain=_positive(
                ice.get("fuel_energy_flow_low_rpm_gain_mj_h_per_rpm"),
                "car_spec: powertrain.ice.fuel_energy_flow_low_rpm_gain_mj_h_per_rpm",
            ),
            fuel_energy_flow_low_rpm_offset_mj_h=_number(
                ice.get("fuel_energy_flow_low_rpm_offset_mj_h"),
                "car_spec: powertrain.ice.fuel_energy_flow_low_rpm_offset_mj_h",
            ),
            mgu_k_peak_power_kw=mgu_k_power,
            mgu_k_torque_limit_nm=_positive(
                mgu_k.get("torque_limit_nm"), "car_spec: powertrain.mgu_k.torque_limit_nm"
            ),
            ers_speed_km_h=np.array(deployment.speed_km_h, dtype=np.float64),
            ers_limit_kw=np.array(deployment.limit_kw, dtype=np.float64),
            ers_overtake_speed_km_h=np.array(overtake.speed_km_h, dtype=np.float64),
            ers_overtake_limit_kw=np.array(overtake.limit_kw, dtype=np.float64),
            store_energy_mj=store_energy,
            recharge_limit_mj_per_lap=_positive(
                mgu_k.get("recharge_limit_mj_per_lap"),
                "car_spec: powertrain.mgu_k.recharge_limit_mj_per_lap",
            ),
            superclip_s=_positive(
                mgu_k.get("superclip_s"), "car_spec: powertrain.mgu_k.superclip_s"
            ),
            launch_speed_kmh=_non_negative(
                mgu_k.get("launch_speed_kmh"), "car_spec: powertrain.mgu_k.launch_speed_kmh"
            ),
            power_split_ice=power_split,
            fuel_lhv_kj_kg=_positive(
                powertrain.get("fuel_lhv_kj_kg"), "car_spec: powertrain.fuel_lhv_kj_kg"
            ),
            gear_ratios=ratios,
            final_drive=final_drive,
            shift_up_rpm=shift_up,
            shift_down_rpm=shift_down,
            shift_time_s=shift_time,
            rolling_radius_m=rolling_radius,
            wheel_diameter_m=wheel_diameter,
            front_width_mm=_positive(tyres.get("front_width_mm"), "car_spec: tyres.front_width_mm"),
            rear_width_mm=_positive(tyres.get("rear_width_mm"), "car_spec: tyres.rear_width_mm"),
            wheelbase_m=wheelbase,
            overall_width_m=_positive(
                chassis.get("overall_width_m"), "car_spec: chassis.overall_width_m"
            ),
            front_weight_fraction=_number(
                chassis.get("front_weight_fraction"),
                "car_spec: chassis.front_weight_fraction",
            ),
        )


def _number(node: Any, where: str) -> float:
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        raise ContractError(f"{where}: expected a number, got {node!r}")
    value = float(node)
    if not math.isfinite(value):
        raise ContractError(f"{where}: expected a finite number, got {node!r}")
    return value


def _positive(node: Any, where: str) -> float:
    value = _number(node, where)
    if value <= 0.0:
        raise ContractError(f"{where}: expected a number > 0, got {node!r}")
    return value


def _non_negative(node: Any, where: str) -> float:
    value = _number(node, where)
    if value < 0.0:
        raise ContractError(f"{where}: expected a number >= 0, got {node!r}")
    return value


def _positive_values(values: Sequence[float], where: str) -> None:
    for index, value in enumerate(values):
        if value <= 0.0:
            raise ContractError(f"{where}: expected values > 0, got {value!r} at index {index}")


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


def _deployment_curve(entries: Any, where: str) -> DeploymentCurve:
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)) or not entries:
        raise ContractError(f"{where}: expected a non-empty list of {{speed_km_h, limit_kw}}")
    speeds: list[float] = []
    limits: list[float] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ContractError(f"{where}[{index}]: expected a mapping, got {entry!r}")
        for key in ("speed_km_h", "limit_kw"):
            if key not in entry:
                raise ContractError(
                    f"{where}[{index}]: expected key {key!r}, got {sorted(map(str, entry))}"
                )
        speeds.append(_number(entry["speed_km_h"], f"{where}[{index}].speed_km_h"))
        limits.append(_number(entry["limit_kw"], f"{where}[{index}].limit_kw"))
    if any(b <= a for a, b in pairwise(speeds)):
        raise ContractError(f"{where}: speed_km_h must be strictly increasing")
    if any(value < 0.0 for value in limits):
        raise ContractError(f"{where}: expected a non-negative limit_kw at every entry")
    return DeploymentCurve(speed_km_h=tuple(speeds), limit_kw=tuple(limits))


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
    integration = _section(root, "integration")

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
        dt_s=_number(integration.get("dt_s"), "car_spec: integration.dt_s"),
        torque_curve=tuple(torque_curve),
        cl_curve=_curve(aero.get("cl_curve"), "car_spec: aero.cl_curve", "cl"),
        cd_curve=_curve(aero.get("cd_curve"), "car_spec: aero.cd_curve", "cd"),
        deployment_curve=_deployment_curve(
            _section(powertrain, "mgu_k").get("deployment_curve_kw"),
            "car_spec: powertrain.mgu_k.deployment_curve_kw",
        ),
        overtake_curve=_deployment_curve(
            _section(powertrain, "mgu_k").get("overtake_curve_kw"),
            "car_spec: powertrain.mgu_k.overtake_curve_kw",
        ),
        raw=root,
    )


def _is_value_node(node: Any) -> bool:
    """A coefficient: a number, a curve, or a parameter group that makes no claim itself.

    A nested mapping without its own ``provenance`` is a parameter group such as
    ``powertrain.ice.turbo_lag``; its parent's claim covers it, but the group is still a
    named thing a claim is allowed to name, so ``turbo_lag`` is claimable while ``quote`` is
    not (a claim block names values, never prose).
    """
    if isinstance(node, bool):
        return False
    if isinstance(node, (int, float)):
        return True
    if isinstance(node, Sequence) and not isinstance(node, (str, bytes)) and node:
        return all(isinstance(entry, (int, float, Mapping)) for entry in node)
    if isinstance(node, Mapping):
        return "provenance" not in node and not set(node) & set(_CLAIM_BLOCKS)
    return False


def _collect_citations(
    node: Mapping[str, Any], path: str, found: dict[str, tuple[str, int]]
) -> None:
    """Walk ``regulation`` claim blocks, including nested ones such as ``powertrain.ice``."""
    block = node.get("regulation")
    if isinstance(block, Mapping):
        for name, entry in block.items():
            clause, page = _citation(entry, f"{path}.regulation.{name}")
            found[f"{path}.{name}"] = (clause, page)
    for key, value in node.items():
        if isinstance(value, Mapping) and "provenance" in value:
            _collect_citations(value, f"{path}.{key}", found)


def _citation(entry: Any, where: str) -> tuple[str, int]:
    if not isinstance(entry, Mapping):
        raise ContractError(f"{where}: expected a mapping with clause and page, got {entry!r}")
    missing = {"clause", "page"} - set(entry)
    if missing:
        raise ContractError(f"{where}: missing key(s) {sorted(missing)}")
    clause = str(entry["clause"])
    page = int(_number(entry["page"], f"{where}.page"))
    if not clause.startswith("C"):
        raise ContractError(
            f"{where}.clause: expected an article reference like C4.1, got {clause!r}"
        )
    if page <= 0:
        raise ContractError(f"{where}.page: expected a positive page number, got {page}")
    return clause, page


def _audit_claims(node: Mapping[str, Any], path: str, findings: list[str]) -> None:
    """Every value in the block is claimed by exactly one of the two claim blocks."""
    regulated = str(node.get("provenance", "")) in _REGULATED_PROVENANCE
    claims: dict[str, Mapping[str, Any]] = {
        name: block for name in _CLAIM_BLOCKS if isinstance(block := node.get(name), Mapping)
    }
    values = {
        key: value
        for key, value in node.items()
        if key not in _CLAIM_BLOCKS and _is_value_node(value)
    }
    claimed: dict[str, str] = {}
    for name, block in claims.items():
        for key, entry in block.items():
            where = f"{path}.{name}.{key}"
            if key not in values:
                findings.append(f"{path}.{name}: {key!r} is not a value or curve in this block")
                continue
            if key in claimed:
                findings.append(
                    f"{path}: {key!r} is claimed by both {claimed[key]!r} and {name!r}; a value "
                    "has one basis"
                )
                continue
            claimed[key] = name
            if name == "regulation":
                findings.extend(_audit_citation(entry, where))
            elif not isinstance(entry, str) or not entry.strip():
                findings.append(f"{where}: expected a sentence saying why this is not regulated")
    for key in values:
        if key not in claimed:
            if regulated:
                findings.append(f"{path}: {key!r} is a value with no regulation citation")
            else:
                findings.append(
                    f"{path}: {key!r} is a value and is claimed by neither 'regulation' nor "
                    "'not_regulated'"
                )
    if regulated and not claims.get("regulation"):
        findings.append(f"{path}: provenance is 'regulated' but there is no regulation block")


def _audit_citation(entry: Any, where: str) -> list[str]:
    if not isinstance(entry, Mapping):
        return [f"{where}: expected a mapping with clause and page"]
    missing = {"clause", "page"} - set(entry)
    if missing:
        return [f"{where}: missing key(s) {sorted(missing)}"]
    clause = entry["clause"]
    page = entry["page"]
    if not isinstance(clause, str) or not clause.startswith("C"):
        return [f"{where}.clause: expected an article reference like C4.1, got {clause!r}"]
    if isinstance(page, bool) or not isinstance(page, int) or page <= 0:
        return [f"{where}.page: expected a positive page number, got {page!r}"]
    quote = entry.get("quote")
    if quote is not None and not str(quote).strip():
        return [f"{where}.quote: present but empty; drop the key or quote the clause"]
    return []


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
    _audit_claims(node, path, findings)
    for key, value in node.items():
        if isinstance(value, Mapping) and "provenance" in value:
            findings.extend(_audit_provenance(value, f"{path}.{key}"))
    return findings


def provenance_audit(root: Mapping[str, Any]) -> list[str]:
    """Report sections that break the standing rule 'every coefficient gets a provenance line'.

    Covers the top-level sections and any nested mapping that declares its own provenance
    (notably ``powertrain.ice`` and ``powertrain.mgu_k``), reported at their dotted path.
    Two failures are reported: a value with no claim at all, and a ``regulation`` claim
    missing a clause or page. Returns a list of human-readable findings; empty means the rule
    holds.
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
