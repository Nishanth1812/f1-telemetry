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

    Built once by :func:`load_car_spec` and reachable through
    :meth:`CarSpec.kernel_config`. Every field is already range-checked at that point, so this
    object carries no validation logic of its own and cannot disagree with the loader.
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
    pacejka_b: float
    pacejka_c: float
    pacejka_e: float
    pacejka_mu: float
    slip_ratio_min_speed_m_s: float
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
    fuel_energy_flow_partial_load_threshold_kw: float
    fuel_energy_flow_partial_load_gain: float
    fuel_energy_flow_partial_load_offset_mj_h: float
    fuel_energy_flow_partial_load_min_mj_h: float
    fuel_to_shaft_efficiency: float
    mgu_k_peak_power_kw: float
    mgu_k_torque_limit_nm: float
    mgu_k_crankshaft_ratio: float
    mgu_k_relative_speed_limit_rpm: float
    mgu_k_motor_inverter_efficiency: float
    ers_speed_km_h: np.ndarray
    ers_limit_kw: np.ndarray
    ers_overtake_speed_km_h: np.ndarray
    ers_overtake_limit_kw: np.ndarray
    store_energy_mj: float
    recharge_limit_mj_per_lap: float
    recharge_limit_reduced_mj_per_lap: float
    recharge_limit_qualifying_floor_mj_per_lap: float
    recharge_allowance_mj_per_lap: float
    superclip_s: float
    launch_speed_kmh: float
    power_split_ice: float
    fuel_lhv_kj_kg: float
    gear_ratios: np.ndarray
    final_drive: float
    reverse_ratio: float
    shift_up_rpm: float
    shift_down_rpm: float
    shift_time_s: float
    shift_time_max_up_s: float
    shift_time_max_down_s: float
    shift_disengage_max_s: float
    clutch_torque_capacity_nm: float
    clutch_demand_torque_nm: float
    clutch_demand_travel_fraction: float
    clutch_control_error_max_nm: float
    clutch_launch_exception_s: float
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
        """Flat, validated numeric arrays for the kernel.

        Builds on access from this spec's own fields, via :meth:`build_kernel_config`, so the
        result can never disagree with the spec it was asked about. An earlier version cached
        the config on the instance at load time, which desynchronised under
        ``dataclasses.replace``: the typed field changed and the cached config did not.
        """
        return self.build_kernel_config()

    def derived_points(self) -> dict[str, tuple[float, ...]]:
        """``dotted.path`` -> curve speeds that the cited clause does **not** state.

        A claim covers the value it names, but a piecewise curve usually needs at least one
        breakpoint that is not in the regulation text - where two stated segments meet a third
        limit, for instance. Claiming the whole curve to the clause would read as if every
        point came from it. Those points are declared in the claim's ``derived_points`` list
        and collected here, so a reader asking "which of these numbers are the regulation's?"
        gets the partial answer from the same API as the citation table.
        """
        found: dict[str, tuple[float, ...]] = {}
        for section in _TOP_LEVEL_SECTIONS:
            block = self.raw.get(section)
            if isinstance(block, Mapping):
                _collect_derived(block, section, found)
        return found

    def build_kernel_config(self) -> KernelConfig:
        """The one place every P1 kernel input is read and range-checked.

        Values that :class:`CarSpec` holds as typed fields are read from those fields, never
        re-parsed out of ``raw``; only the P1 inputs beyond the P0 field set are parsed here.
        Called by :meth:`kernel_config` on every access rather than cached, so the config is
        always derived from the fields currently on the spec - a spec built by hand, or
        modified with ``dataclasses.replace``, is validated exactly as a loaded one is.

        The checks are the ones a compiled kernel cannot make: a division by a zero rolling
        radius, a slip denominator that vanishes, a speed grid that would be indexed out of
        bounds, and a power curve that exceeds a regulatory cap.
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

        cl = self.cl_curve
        cd = self.cd_curve
        deployment = self.deployment_curve
        overtake = self.overtake_curve
        if cl is None or cd is None or deployment is None or overtake is None:
            raise ContractError("car spec was loaded without its curves")

        _positive_values(cl.value, "car_spec: aero.cl_curve: every cl must be > 0")
        _positive_values(cd.value, "car_spec: aero.cd_curve: every cd must be > 0")
        if cl.speed_m_s != cd.speed_m_s:
            raise ContractError(
                "car_spec: aero.cl_curve and aero.cd_curve must use the same speed_m_s "
                f"breakpoints; got cl {list(cl.speed_m_s)} and cd {list(cd.speed_m_s)}. "
                "KernelConfig exposes one shared speed axis, so a mismatch would index the "
                "wrong curve."
            )

        mass_kg = _positive(self.mass_kg, "car_spec: mass.total_kg")
        gravity = _positive(self.gravity_m_s2, "car_spec: constants.gravity_m_s2")
        air_density = _positive(self.air_density_kg_m3, "car_spec: constants.air_density_kg_m3")
        air_temperature = _number(
            constants.get("air_temperature_k"), "car_spec: constants.air_temperature_k"
        )
        reference_area = _positive(self.reference_area_m2, "car_spec: aero.reference_area_m2")
        ride_height = _non_negative(
            aero.get("ride_height_sensitivity"), "car_spec: aero.ride_height_sensitivity"
        )
        dt_s = _positive(self.dt_s, "car_spec: integration.dt_s")
        ice_peak = _positive(self.ice_peak_power_kw, "car_spec: powertrain.ice.peak_power_kw")
        idle_rpm = _positive(ice.get("idle_rpm"), "car_spec: powertrain.ice.idle_rpm")
        rev_limit = _positive(self.rev_limit_rpm, "car_spec: powertrain.ice.rev_limit_rpm")
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

        # C5.2.5 (page 64) bounds the ICE's fuel energy flow as a function of *engine power*
        # rather than of rpm, with a flat arm at or below -50 kW. The threshold is therefore the
        # one number in this block that is allowed to be negative - it is a power, and the
        # clause's constant arm starts at or below it - so it is checked for finiteness and the
        # other three for sign. The arms are recorded as four numbers because the clause states
        # two formulas; that they meet at the threshold is a property of the data, asserted where
        # the physics is tested rather than enforced here.
        partial_load_threshold = _number(
            ice.get("fuel_energy_flow_partial_load_threshold_kw"),
            "car_spec: powertrain.ice.fuel_energy_flow_partial_load_threshold_kw",
        )
        partial_load_gain = _positive(
            ice.get("fuel_energy_flow_partial_load_gain_mj_h_per_kw"),
            "car_spec: powertrain.ice.fuel_energy_flow_partial_load_gain_mj_h_per_kw",
        )
        partial_load_offset = _positive(
            ice.get("fuel_energy_flow_partial_load_offset_mj_h"),
            "car_spec: powertrain.ice.fuel_energy_flow_partial_load_offset_mj_h",
        )
        partial_load_min = _positive(
            ice.get("fuel_energy_flow_partial_load_min_mj_h"),
            "car_spec: powertrain.ice.fuel_energy_flow_partial_load_min_mj_h",
        )
        # P1-T6: the conversion between the MJ/h C5.2.3/.4/.5 bound the engine by and the kW of
        # shaft power everything downstream computes. No clause publishes it, so it is checked as
        # a fraction rather than a magnitude: at or above one the model would report more shaft
        # power than the fuel it burned, and every fuel-energy-flow limit would be unreachable.
        fuel_to_shaft = _fraction(
            ice.get("fuel_to_shaft_efficiency"),
            "car_spec: powertrain.ice.fuel_to_shaft_efficiency",
        )

        mgu_k_power = _positive(
            self.mgu_k_peak_power_kw, "car_spec: powertrain.mgu_k.peak_power_kw"
        )
        # C5.2.7 caps absolute ERS-K power; C5.2.8's speed curves are the *propulsion* limit and
        # sit above it at low speed. The file stores the effective limit (the smaller of the
        # two), and this check keeps that true for any curve that reaches the builder - the
        # curves in car_spec.yaml are already capped, but a clamp in a file is a claim about
        # today's file, not a guarantee, and CarSpec is public so a curve can be replaced
        # without ever going through the loader.
        ers_limits = _capped(deployment, mgu_k_power, "deployment_curve_kw")
        ers_overtake_limits = _capped(overtake, mgu_k_power, "overtake_curve_kw")
        store_energy = _positive(
            mgu_k.get("store_energy_mj"), "car_spec: powertrain.mgu_k.store_energy_mj"
        )
        # C5.18.2 permanently gears the MGU-K to the crankshaft at a *fixed* ratio and states no
        # value for it, so the number is this project's. It is still a magnitude: a zero ratio
        # would leave the MGU-K with no coupling to join at, and a negative one would drive the
        # crankshaft backwards. C5.18.5's 60 000 rpm ceiling is on the *relative* speed, i.e. the
        # product of this ratio and the engine speed, so the two are read together.
        crankshaft_ratio = _positive(
            mgu_k.get("crankshaft_ratio"), "car_spec: powertrain.mgu_k.crankshaft_ratio"
        )
        relative_speed_limit = _positive(
            mgu_k.get("relative_speed_limit_rpm"),
            "car_spec: powertrain.mgu_k.relative_speed_limit_rpm",
        )
        # The same fraction rule as the ICE's: C5.2.7 caps the MGU-K in electrical DC power and
        # the model holds mechanical shaft torque, so an efficiency above one would let the motor
        # deliver more than the clause caps and the DC limit would never bind.
        motor_inverter = _fraction(
            mgu_k.get("motor_inverter_efficiency"),
            "car_spec: powertrain.mgu_k.motor_inverter_efficiency",
        )
        # C5.2.10 (page 64) states a per-lap recharge *baseline* and then reduces it under
        # conditions it lists: 7 MJ, a 4 MJ qualifying floor, and a conditional 0.5 MJ
        # allowance. Reading the 8.5 MJ as a standing cap is the mistake this ordering exists to
        # catch, so an event-conditioned value above the baseline it modifies is refused rather
        # than accepted as a stricter or looser limit.
        recharge_baseline = _positive(
            mgu_k.get("recharge_limit_mj_per_lap"),
            "car_spec: powertrain.mgu_k.recharge_limit_mj_per_lap",
        )
        recharge_reduced = _positive(
            mgu_k.get("recharge_limit_reduced_mj_per_lap"),
            "car_spec: powertrain.mgu_k.recharge_limit_reduced_mj_per_lap",
        )
        recharge_qualifying = _positive(
            mgu_k.get("recharge_limit_qualifying_floor_mj_per_lap"),
            "car_spec: powertrain.mgu_k.recharge_limit_qualifying_floor_mj_per_lap",
        )
        recharge_allowance = _positive(
            mgu_k.get("recharge_allowance_mj_per_lap"),
            "car_spec: powertrain.mgu_k.recharge_allowance_mj_per_lap",
        )
        if recharge_reduced > recharge_baseline:
            raise ContractError(
                f"car_spec: powertrain.mgu_k.recharge_limit_reduced_mj_per_lap "
                f"({recharge_reduced}) exceeds the C5.2.10 baseline of {recharge_baseline} MJ/lap. "
                "C5.2.10's reduced figure is a reduction of the baseline, not a second one."
            )
        if recharge_qualifying > recharge_reduced:
            raise ContractError(
                "car_spec: powertrain.mgu_k.recharge_limit_qualifying_floor_mj_per_lap "
                f"({recharge_qualifying}) exceeds the C5.2.10 reduced limit of {recharge_reduced} "
                "MJ/lap. The qualifying floor is the lowest of the article's three figures."
            )
        power_split = _number(self.power_split_ice, "car_spec: powertrain.power_split_ice")
        if not 0.0 < power_split < 1.0:
            raise ContractError(
                f"car_spec: powertrain.power_split_ice must be in (0, 1), got {power_split}"
            )

        # P1-T5 is the first task to read the ratios, and it reads them as an engine-speed to
        # wheel-speed factor. A ratio at or below zero maps an engine speed onto a wheel speed with
        # a sign or a scale the drivetrain cannot use, and no existing rule here covers that, so it
        # is covered here rather than by the model that consumes it.
        ratios = np.array(self.gear_ratios, dtype=np.float64)
        if not np.all(ratios > 0.0):
            raise ContractError(
                f"car_spec: gearbox.ratios must all be > 0, got {list(self.gear_ratios)}. "
                "A zero ratio would stop the engine from turning the wheel and a negative one "
                "would drive it the wrong way."
            )
        final_drive = _positive(self.final_drive, "car_spec: gearbox.final_drive")
        # C9.7 (page 104) requires the car to be drivable in reverse at any time, but states no
        # reverse ratio, so the number is synthesised. It is a magnitude like `ratios`: a negative
        # reverse ratio would be the same ratio counted twice, and a zero one would transmit
        # nothing however hard the driver asked. What makes reverse work is that it negates the
        # transmitted torque, never that it indexes the forward table - which stays 1..8 and
        # must not be indexed by 0 or -1.
        reverse_ratio = _positive(gearbox.get("reverse_ratio"), "car_spec: gearbox.reverse_ratio")
        shift_up = _positive(gearbox.get("shift_up_rpm"), "car_spec: gearbox.shift_up_rpm")
        shift_down = _positive(gearbox.get("shift_down_rpm"), "car_spec: gearbox.shift_down_rpm")
        # P1-T5: strictly positive, not merely non-negative. The shift timer is the only thing that
        # freezes the gear while a shift runs, so `physics.gearbox.step_gearbox` refuses a zero
        # shift time - at zero an rpm sitting on the upshift point advances the box a gear per
        # step. The loader and the step have to agree, or the file could hold a value the model
        # will not take.
        shift_time = _positive(gearbox.get("shift_time_s"), "car_spec: gearbox.shift_time_s")
        # C9.8.4 (page 104) bounds a gear change from above - 200 ms up, 300 ms down - and bounds
        # the request-to-disengage time separately at 80 ms. `shift_time_s` is a project number,
        # so the clause is the only thing that keeps it honest: the loader checks the duration
        # against both direction limits rather than trusting the committed 40 ms. The bound is
        # inclusive, because the up limit is itself a legal value for a shared duration.
        shift_time_max_up = _positive(
            gearbox.get("shift_time_max_up_s"), "car_spec: gearbox.shift_time_max_up_s"
        )
        shift_time_max_down = _positive(
            gearbox.get("shift_time_max_down_s"), "car_spec: gearbox.shift_time_max_down_s"
        )
        shift_disengage = _positive(
            gearbox.get("shift_disengage_max_s"), "car_spec: gearbox.shift_disengage_max_s"
        )
        if shift_time_max_up > 0.2:
            raise ContractError(
                f"car_spec: gearbox.shift_time_max_up_s ({shift_time_max_up}) exceeds the "
                "C9.8.4 200 ms maximum"
            )
        if shift_time_max_down > 0.3:
            raise ContractError(
                f"car_spec: gearbox.shift_time_max_down_s ({shift_time_max_down}) exceeds the "
                "C9.8.4 300 ms maximum"
            )
        if shift_disengage > 0.08:
            raise ContractError(
                f"car_spec: gearbox.shift_disengage_max_s ({shift_disengage}) exceeds the "
                "C9.8.4 80 ms maximum"
            )
        fastest_change = min(shift_time_max_up, shift_time_max_down)
        if shift_time > fastest_change:
            raise ContractError(
                f"car_spec: gearbox.shift_time_s ({shift_time}) exceeds the C9.8.4 gear-change "
                f"limit of {fastest_change} s. C9.8.4 allows 200 ms up and 300 ms down; one "
                "shared duration is only legal below the smaller of the two."
            )
        # P1-T5: the clutch capacity is the ceiling every transmitted torque is measured against, so
        # it has to be positive. Zero would let the model apply it at all; a negative one would put
        # the sign of the capacity on the wrong side of the engine's.
        clutch_capacity = _positive(
            gearbox.get("clutch_torque_capacity_nm"),
            "car_spec: gearbox.clutch_torque_capacity_nm",
        )
        # C9.2.5 (page 101) states the driver's clutch *demand* as rear-axle torque: a 5200 Nm gain
        # over 90% of engagement travel, tracked to within +/-150 Nm, with the first 85 ms of a
        # launch step excepted from that band. None of these is the capacity above, and none of
        # them is a gain on engine torque either - they are rear-axle quantities.
        clutch_demand_torque = _positive(
            gearbox.get("clutch_demand_torque_nm"),
            "car_spec: gearbox.clutch_demand_torque_nm",
        )
        clutch_demand_travel = _fraction(
            gearbox.get("clutch_demand_travel_fraction"),
            "car_spec: gearbox.clutch_demand_travel_fraction",
        )
        clutch_error = _positive(
            gearbox.get("clutch_control_error_max_nm"),
            "car_spec: gearbox.clutch_control_error_max_nm",
        )
        clutch_launch_exception = _non_negative(
            gearbox.get("clutch_launch_exception_s"),
            "car_spec: gearbox.clutch_launch_exception_s",
        )
        if shift_up > rev_limit:
            raise ContractError(
                f"car_spec: gearbox.shift_up_rpm ({shift_up}) exceeds rev_limit_rpm ({rev_limit})"
            )
        if shift_down >= shift_up:
            raise ContractError(
                f"car_spec: gearbox.shift_down_rpm ({shift_down}) must be below shift_up_rpm "
                f"({shift_up})"
            )

        rolling_radius = _positive(self.rolling_radius_m, "car_spec: tyres.rolling_radius_m")
        wheel_diameter = _positive(
            tyres.get("wheel_diameter_m"), "car_spec: tyres.wheel_diameter_m"
        )
        # P1-T3: the longitudinal Magic Formula and the guard on its slip denominator. `e` is
        # checked for finiteness rather than for a sign, because the curvature factor carries
        # one; `b`, `c` and `mu` are magnitudes. The slip guard has to be positive, since it is
        # the denominator of the one division in the tyre model that can otherwise divide by
        # zero at a standing start.
        pacejka = _section(tyres, "longitudinal_pacejka")
        pacejka_b = _positive(pacejka.get("b"), "car_spec: tyres.longitudinal_pacejka.b")
        pacejka_c = _positive(pacejka.get("c"), "car_spec: tyres.longitudinal_pacejka.c")
        pacejka_e = _number(pacejka.get("e"), "car_spec: tyres.longitudinal_pacejka.e")
        pacejka_mu = _positive(pacejka.get("mu"), "car_spec: tyres.longitudinal_pacejka.mu")
        slip_guard = _positive(
            tyres.get("slip_ratio_min_speed_m_s"),
            "car_spec: tyres.slip_ratio_min_speed_m_s",
        )
        wheelbase = _positive(chassis.get("wheelbase_m"), "car_spec: chassis.wheelbase_m")
        front_weight = _number(
            chassis.get("front_weight_fraction"), "car_spec: chassis.front_weight_fraction"
        )
        if not 0.0 < front_weight < 1.0:
            raise ContractError(
                "car_spec: chassis.front_weight_fraction must be in (0, 1), got "
                f"{front_weight}. It is a fraction of a total mass. Note that C4.2's 0.44 / 0.54 "
                "are fractions of the C4.1 Minimum Mass including the separately published "
                "Nominal Tyre Mass, not of mass.total_kg, so they are not enforced here - see "
                "chassis.c42_enforcement."
            )

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
            pacejka_b=pacejka_b,
            pacejka_c=pacejka_c,
            pacejka_e=pacejka_e,
            pacejka_mu=pacejka_mu,
            slip_ratio_min_speed_m_s=slip_guard,
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
            fuel_energy_flow_partial_load_threshold_kw=partial_load_threshold,
            fuel_energy_flow_partial_load_gain=partial_load_gain,
            fuel_energy_flow_partial_load_offset_mj_h=partial_load_offset,
            fuel_energy_flow_partial_load_min_mj_h=partial_load_min,
            fuel_to_shaft_efficiency=fuel_to_shaft,
            mgu_k_peak_power_kw=mgu_k_power,
            mgu_k_torque_limit_nm=_positive(
                mgu_k.get("torque_limit_nm"), "car_spec: powertrain.mgu_k.torque_limit_nm"
            ),
            mgu_k_crankshaft_ratio=crankshaft_ratio,
            mgu_k_relative_speed_limit_rpm=relative_speed_limit,
            mgu_k_motor_inverter_efficiency=motor_inverter,
            ers_speed_km_h=np.array(deployment.speed_km_h, dtype=np.float64),
            ers_limit_kw=ers_limits,
            ers_overtake_speed_km_h=np.array(overtake.speed_km_h, dtype=np.float64),
            ers_overtake_limit_kw=ers_overtake_limits,
            store_energy_mj=store_energy,
            recharge_limit_mj_per_lap=recharge_baseline,
            recharge_limit_reduced_mj_per_lap=recharge_reduced,
            recharge_limit_qualifying_floor_mj_per_lap=recharge_qualifying,
            recharge_allowance_mj_per_lap=recharge_allowance,
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
            reverse_ratio=reverse_ratio,
            shift_up_rpm=shift_up,
            shift_down_rpm=shift_down,
            shift_time_s=shift_time,
            shift_time_max_up_s=shift_time_max_up,
            shift_time_max_down_s=shift_time_max_down,
            shift_disengage_max_s=shift_disengage,
            clutch_torque_capacity_nm=clutch_capacity,
            clutch_demand_torque_nm=clutch_demand_torque,
            clutch_demand_travel_fraction=clutch_demand_travel,
            clutch_control_error_max_nm=clutch_error,
            clutch_launch_exception_s=clutch_launch_exception,
            rolling_radius_m=rolling_radius,
            wheel_diameter_m=wheel_diameter,
            front_width_mm=_positive(tyres.get("front_width_mm"), "car_spec: tyres.front_width_mm"),
            rear_width_mm=_positive(tyres.get("rear_width_mm"), "car_spec: tyres.rear_width_mm"),
            wheelbase_m=wheelbase,
            overall_width_m=_positive(
                chassis.get("overall_width_m"), "car_spec: chassis.overall_width_m"
            ),
            front_weight_fraction=front_weight,
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


def _fraction(node: Any, where: str) -> float:
    """A value strictly inside ``(0, 1]`` - a share of something, not a magnitude.

    Distinct from ``_positive`` because the interesting boundary is the top: C9.2.5's 90 %
    engagement-travel span is legal, 0 % is not, and a rule that accepted zero would let the
    model divide by it.
    """
    value = _number(node, where)
    if not 0.0 < value <= 1.0:
        raise ContractError(f"{where}: expected a number in (0, 1], got {node!r}")
    return value


def _positive_values(values: Sequence[float], where: str) -> None:
    for index, value in enumerate(values):
        if value <= 0.0:
            raise ContractError(f"{where}: expected values > 0, got {value!r} at index {index}")


def _capped(curve: DeploymentCurve, cap_kw: float, name: str) -> np.ndarray:
    """An ERS deployment curve's limits as a float64 array, refusing any point above the cap.

    C5.2.8's formulas are the *propulsion* limit and permit far more than the car may deliver:
    ``1800 - 5v`` allows 1800 kW at rest, ``7100 - 20v`` allows 7100 kW. C5.2.7 caps absolute
    ERS-K electrical DC power at 350 kW whatever the speed, so the effective limit is the
    smaller of the two clauses and the curve in ``car_spec.yaml`` must already be clamped.

    This enforces that at the shared boundary rather than trusting the committed data. Data
    being correct today is not a guarantee, and ``CarSpec`` is a public frozen dataclass a caller
    can construct or ``dataclasses.replace`` directly, so a check confined to ``load_car_spec``
    would be bypassable. Rejecting rather than clamping is deliberate: silently clamping would
    hide a bad edit from whoever made it and leave the file's knots disagreeing with the array
    the kernel actually reads.

    The bound is inclusive, because the committed curve sits exactly on the cap at low speed.
    """
    limits = np.array(curve.limit_kw, dtype=np.float64)
    over_cap = limits > cap_kw
    if bool(np.any(over_cap)):
        index = int(np.argmax(over_cap))
        speed = curve.speed_km_h[index] if index < len(curve.speed_km_h) else float("nan")
        raise ContractError(
            f"car_spec: powertrain.mgu_k.{name}: limit_kw[{index}] is {limits[index]} kW at "
            f"{speed} km/h, above the C5.2.7 absolute cap of {cap_kw} kW. C5.2.8's speed curve is "
            "the propulsion limit only; the effective limit is the smaller of C5.2.8 and "
            "C5.2.7, so the curve in car_spec.yaml must already be clamped."
        )
    return limits


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

    # Range checks live in `build_kernel_config`, which `kernel_config` calls on every access.
    # Repeating them here would be the second validation path over the same numbers that
    # round 1 removed, and it would still miss a hand-built or replaced spec.
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
        rolling_radius_m=_number(tyres.get("rolling_radius_m"), "car_spec: tyres.rolling_radius_m"),
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
        if "provenance" in node or set(node) & set(_CLAIM_BLOCKS):
            return False
        # A group of numbers (or curves) is a parameter group a claim may name; a group of
        # prose and metadata is documentation, which no claim should have to explain.
        return any(_is_value_node(value) for value in node.values())
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


def _collect_derived(
    node: Mapping[str, Any], path: str, found: dict[str, tuple[float, ...]]
) -> None:
    """Collect ``derived_points`` declared inside a ``regulation`` claim."""
    block = node.get("regulation")
    if isinstance(block, Mapping):
        for name, entry in block.items():
            speeds = _derived_speeds(entry, f"{path}.regulation.{name}")
            if speeds:
                found[f"{path}.{name}"] = speeds
    for key, value in node.items():
        if isinstance(value, Mapping) and "provenance" in value:
            _collect_derived(value, f"{path}.{key}", found)


def _derived_speeds(entry: Any, where: str) -> tuple[float, ...]:
    if not isinstance(entry, Mapping):
        return ()
    points = entry.get("derived_points")
    if points is None:
        return ()
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)) or not points:
        raise ContractError(f"{where}.derived_points: expected a non-empty list of points")
    speeds: list[float] = []
    for index, point in enumerate(points):
        point_where = f"{where}.derived_points[{index}]"
        if not isinstance(point, Mapping) or "speed_km_h" not in point:
            raise ContractError(f"{point_where}: expected a mapping with speed_km_h")
        speeds.append(_number(point["speed_km_h"], f"{point_where}.speed_km_h"))
    if any(b <= a for a, b in pairwise(speeds)):
        raise ContractError(f"{where}.derived_points: speed_km_h must be strictly increasing")
    return tuple(speeds)


def _audit_derived(entry: Mapping[str, Any], where: str, value: Any) -> list[str]:
    """Every declared derived point must be in the curve, and must say why it is there.

    ``derived_points`` is how a claim stays honest when a cited curve needs a breakpoint the
    clause does not state. It only works if it is checked, so a point naming a speed the curve
    does not have is an audit failure rather than dead metadata.

    Everything here is a *finding*. The audit's job is to report what is wrong with a file, so
    a malformed entry - a string where a speed belongs, a missing ``basis`` - must be reported
    rather than raised: raising takes the audit down and tells the reader nothing about the
    other 22 citations. That is why this does not use the ``_number`` helper, which raises.
    """
    points = entry.get("derived_points")
    if points is None:
        return []
    findings: list[str] = []
    name = where.rsplit(".", 1)[-1]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return [f"{where}.derived_points: only a curve can declare derived points"]
    speeds = _curve_speeds(value)
    for index, point in enumerate(points):
        point_where = f"{where}.derived_points[{index}]"
        if not _has_speed(point):
            findings.append(f"{point_where}: expected a mapping with speed_km_h")
            continue
        speed, complaint = _speed(point, point_where)
        if speed is None:
            findings.append(complaint or f"{point_where}.speed_km_h: expected a number")
            continue
        point_where = f"{where}.derived_points[{speed}]"
        if speed not in speeds:
            findings.append(
                f"{point_where}: speed_km_h {speed} is not a breakpoint of {name}; it has {speeds}"
            )
        basis = point.get("basis")
        if not isinstance(basis, str) or not basis.strip():
            findings.append(
                f"{point_where}.basis: expected a sentence saying why this breakpoint is not "
                "in the clause"
            )
    return findings


def _has_speed(point: Any) -> bool:
    return isinstance(point, Mapping) and "speed_km_h" in point


def _curve_speeds(points: Sequence[Any]) -> list[float]:
    """The usable speeds of a curve, skipping any entry that does not have one."""
    found: list[float] = []
    for point in points:
        if not _has_speed(point):
            continue
        speed = _speed(point, "curve")[0]
        if speed is not None:
            found.append(speed)
    return found


def _speed(point: Mapping[str, Any], where: str) -> tuple[float | None, str | None]:
    """The point's speed and, when it is malformed, the finding describing why.

    ``(None, message)`` for anything :func:`_number` would have raised on, so the audit reports
    the bad entry and carries on to the next one instead of stopping.
    """
    node = point["speed_km_h"]
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        return None, f"{where}.speed_km_h: expected a number, got {node!r}"
    value = float(node)
    if not math.isfinite(value):
        return None, f"{where}.speed_km_h: expected a finite number, got {node!r}"
    return value, None


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
                if isinstance(entry, Mapping):
                    findings.extend(_audit_derived(entry, where, values[key]))
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
                f"spec: calibration_status is {spec.get('calibration_status')!r}; this phase "
                "must not ship a car spec that claims to be calibrated"
            )
    return findings
