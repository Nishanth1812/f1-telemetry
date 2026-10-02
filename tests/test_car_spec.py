"""The P1 configuration boundary: ``car_spec.yaml`` provenance and kernel-ready arrays.

P1-T1's claim is that ``car_spec.yaml`` is the only place a physical coefficient lives.
Two things have to be true for that to be more than a convention:

* every regulated value carries the FIA issue, clause and page it was read from, and every
  value that is *not* regulated says why, so a reviewer can tell a regulation number from a
  synthesised one without opening the PDF;
* the loader hands the kernel plain numeric arrays and refuses to hand over anything invalid,
  so a bad edit fails in Python instead of inside Numba.

The citation table below is pinned deliberately. It is the machine-checkable form of "read
these clauses out of the current issue", and it fails loudly when the file drifts from the
document it claims to cite.

``KernelConfig`` is referenced through the module rather than imported by name. For the P0
regression run, the ``contract`` pytest marker introduced with P1 was copied into the temporary
P0 worktree so collection could proceed. With that marker registered, the P1 tests failed on
behaviour rather than a missing symbol: no ``citations()``, no ``kernel_config()``, no grid
check, and no ``front_weight_fraction`` range.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import yaml

from f1telemetry.contracts import car_spec as car_spec_module
from f1telemetry.contracts.car_spec import (
    CarSpec,
    ContractError,
    load_car_spec,
    provenance_audit,
)

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import KernelConfig

pytestmark = pytest.mark.contract

FIA_ISSUE = 20
FIA_ISSUE_DATE = "2026-08-05"
FIA_DOCUMENT_URL = (
    "https://www.fia.com/system/files/documents/"
    "fia_2026_f1_regulations_-_section_c_technical_-_iss_20_-_2026-08-05.pdf"
)
FIA_LISTING_URL = "https://www.fia.com/regulation/category/110/technical-regulations"

# (dotted path, clause, page) for every value read out of FIA 2026 Section C Issue 20.
EXPECTED_CITATIONS: dict[str, tuple[str, int]] = {
    "mass.minimum_mass_kg": ("C4.1", 59),
    "mass.minimum_mass_qualifying_kg": ("C4.1", 59),
    "mass.driver_reference_mass_kg": ("C4.5.2", 60),
    "powertrain.ice.idle_rpm": ("C5.13.4", 75),
    "powertrain.ice.fuel_energy_flow_max_mj_h": ("C5.2.3", 64),
    "powertrain.ice.fuel_energy_flow_per_cylinder_max_mj_h": ("C5.2.3", 64),
    "powertrain.ice.fuel_energy_flow_low_rpm_gain_mj_h_per_rpm": ("C5.2.4", 64),
    "powertrain.ice.fuel_energy_flow_low_rpm_offset_mj_h": ("C5.2.4", 64),
    "powertrain.ice.fuel_energy_flow_low_rpm_limit_rpm": ("C5.2.4", 64),
    "powertrain.ice.fuel_energy_flow_partial_load_gain_mj_h_per_kw": ("C5.2.5", 64),
    "powertrain.ice.fuel_energy_flow_partial_load_offset_mj_h": ("C5.2.5", 64),
    "powertrain.ice.fuel_energy_flow_partial_load_min_mj_h": ("C5.2.5", 64),
    "powertrain.ice.fuel_energy_flow_partial_load_threshold_kw": ("C5.2.5", 64),
    "powertrain.mgu_k.peak_power_kw": ("C5.2.7", 64),
    "powertrain.mgu_k.deployment_curve_kw": ("C5.2.8", 64),
    "powertrain.mgu_k.overtake_curve_kw": ("C5.2.8", 64),
    "powertrain.mgu_k.store_energy_mj": ("C5.2.9", 64),
    "powertrain.mgu_k.recharge_limit_mj_per_lap": ("C5.2.10", 64),
    "powertrain.mgu_k.recharge_limit_reduced_mj_per_lap": ("C5.2.10", 64),
    "powertrain.mgu_k.recharge_limit_qualifying_floor_mj_per_lap": ("C5.2.10", 64),
    "powertrain.mgu_k.recharge_allowance_mj_per_lap": ("C5.2.10", 64),
    "powertrain.mgu_k.torque_limit_nm": ("C5.2.11", 65),
    "powertrain.mgu_k.launch_speed_kmh": ("C5.2.12", 65),
    "powertrain.mgu_k.transient_torque_limiter_threshold_nm": ("C5.18.4", 78),
    "powertrain.mgu_k.relative_speed_limit_rpm": ("C5.18.5", 78),
    "gearbox.clutch_demand_torque_nm": ("C9.2.5", 101),
    "gearbox.clutch_demand_travel_fraction": ("C9.2.5", 101),
    "gearbox.clutch_control_error_max_nm": ("C9.2.5", 101),
    "gearbox.clutch_launch_exception_s": ("C9.2.5", 101),
    "gearbox.shift_time_max_up_s": ("C9.8.4", 104),
    "gearbox.shift_time_max_down_s": ("C9.8.4", 104),
    "gearbox.shift_disengage_max_s": ("C9.8.4", 104),
    "tyres.front_width_mm": ("C10.7.2", 111),
    "tyres.rear_width_mm": ("C10.7.2", 111),
    "tyres.rim_diameter_mm": ("C10.7.2", 111),
    "chassis.overall_width_m": ("C2.3.1", 10),
    "chassis.wheelbase_m": ("C2.3.3", 11),
    "chassis.minimum_front_axle_fraction": ("C4.2", 59),
    "chassis.minimum_rear_axle_fraction": ("C4.2", 59),
}

# Sections that state no FIA basis at all, so every number in them is a project decision.
# `gearbox` is not one of them: C9.2.5 and C9.8.4 fix the clutch demand model and the shifting
# timings, so the section carries both claim blocks. See
# ``test_the_gearbox_section_is_mixed_and_says_which_number_is_which``.
NOT_REGULATED_SECTIONS = ("aero", "integration")


def _root(repo: Path) -> dict[str, Any]:
    document = yaml.safe_load((repo / "car_spec.yaml").read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _at(root: Mapping[str, Any], path: tuple[Any, ...]) -> Any:
    """Walk a YAML document by key path. Returns ``Any``: the document is unvalidated here."""
    node: Any = root
    for part in path:
        node = node[part]
    return node


def _write(root: dict[str, Any], tmp_path: Path) -> Path:
    path = tmp_path / "car_spec.yaml"
    path.write_text(yaml.safe_dump(root, sort_keys=False), encoding="utf-8")
    return path


def _arrays(config: KernelConfig) -> dict[str, np.ndarray]:
    return {
        field.name: value
        for field in fields(config)
        if isinstance((value := getattr(config, field.name)), np.ndarray)
    }


def test_spec_is_pinned_to_the_current_fia_issue(spec: CarSpec) -> None:
    """The citation table below only means anything against one named document."""
    assert spec.spec["issue"] == FIA_ISSUE
    assert str(spec.spec["issue_date"]) == FIA_ISSUE_DATE
    assert spec.spec["document_url"] == FIA_DOCUMENT_URL
    assert spec.spec["listing_url"] == FIA_LISTING_URL
    assert spec.spec["verified_against_document"] is True
    assert str(spec.spec["verified_on"])


def test_every_regulated_value_cites_the_clause_it_was_read_from(spec: CarSpec) -> None:
    assert spec.citations() == EXPECTED_CITATIONS


def test_a_curve_claim_declares_its_derived_breakpoints(spec: CarSpec) -> None:
    """A breakpoint the clause does not state must be machine-visible, not buried in prose.

    C5.2.8 states the deployment limit as two linear segments meeting at 340 km/h, and C5.2.7
    separately caps absolute ERS-K power at 350 kW. Below 290 km/h the first segment's
    ``1800 - 5v`` would allow up to 1800 kW, so the effective limit is the smaller of the two
    clauses and the curve has to turn where they cross. That crossing speed is derived, and
    declaring it keeps the curve from reading as four regulated points.
    """
    assert spec.derived_points() == {
        "powertrain.mgu_k.deployment_curve_kw": (290.0,),
        "powertrain.mgu_k.overtake_curve_kw": (337.5,),
    }

    claim = _at(spec.raw, ("powertrain", "mgu_k", "regulation", "deployment_curve_kw"))
    assert claim["clause"] == "C5.2.8"
    entry = next(p for p in claim["derived_points"] if p["speed_km_h"] == 290.0)
    assert "C5.2.7" in entry["basis"]


def test_the_ers_curves_never_exceed_the_absolute_cap(spec: CarSpec) -> None:
    """C5.2.7 caps absolute ERS-K power at 350 kW, and C5.2.8 sits under it at low speed.

    Taken alone, ``1800 - 5v`` permits 1800 kW at rest and ``7100 - 20v`` permits 7100 kW.
    Both formulas are only ever the *propulsion* limit; the absolute cap still applies, so the
    effective limit is the smaller of the two. A kernel reading the raw curve would be handed
    five times the power the regulation allows.
    """
    config = spec.kernel_config()
    cap = spec.mgu_k_peak_power_kw
    assert cap == 350.0

    for speeds, limits, label in (
        (config.ers_speed_km_h, config.ers_limit_kw, "deployment"),
        (config.ers_overtake_speed_km_h, config.ers_overtake_limit_kw, "overtake"),
    ):
        assert float(np.max(limits)) == pytest.approx(cap), label
        assert float(limits[0]) == pytest.approx(cap), label
        # Interpolating anywhere on the curve must stay under the cap, not just at the knots.
        for speed in np.linspace(0.0, float(speeds[-1]), 201):
            assert _interpolate(speeds, limits, float(speed)) <= cap + 1e-9, (label, speed)
        assert float(limits[-1]) == 0.0, label


def test_the_ers_curve_knots_are_where_c52_8_and_c52_7_cross(spec: CarSpec) -> None:
    """The crossover speeds are arithmetic, not fitted, so they are pinned exactly."""
    config = spec.kernel_config()
    # 1800 - 5v = 350 -> v = 290 km/h.  7100 - 20v = 350 -> v = 337.5 km/h.
    assert list(config.ers_speed_km_h) == [0.0, 290.0, 340.0, 345.0]
    assert list(config.ers_limit_kw) == [350.0, 350.0, 100.0, 0.0]
    assert list(config.ers_overtake_speed_km_h) == [0.0, 337.5, 355.0]
    assert list(config.ers_overtake_limit_kw) == [350.0, 350.0, 0.0]
    # The clauses' own formulas must reproduce the knots they are stated to give, and the cap
    # must be what binds below each crossover. C5.2.8's first segment agrees with the second at
    # 340 km/h, which is why the sampled curve needs no knot there for the clause's own sake.
    assert _close(6900.0 - 20.0 * 340.0, 100.0)  # the 100 kW knot
    assert _close(1800.0 - 5.0 * 340.0, 100.0)  # same point from the first segment
    assert _close(6900.0 - 20.0 * 345.0, 0.0)  # zero from 345 km/h
    assert _close(7100.0 - 20.0 * 355.0, 0.0)  # Overtake zero from 355 km/h
    # Where each clause's formula falls to the C5.2.7 cap - the derived crossover speeds.
    assert _close(1800.0 - 5.0 * 290.0, 350.0)
    assert _close(7100.0 - 20.0 * 337.5, 350.0)


@pytest.mark.parametrize("curve", ["deployment_curve_kw", "overtake_curve_kw"])
def test_an_ers_curve_point_above_the_absolute_cap_is_rejected(
    tmp_path: Path, repo: Path, curve: str
) -> None:
    """The cap is enforced at the boundary, not merely correct in today's committed file.

    Both curves in ``car_spec.yaml`` sit at or below 350 kW, but that is a property of the data,
    not a guarantee. A later edit that restores a raw C5.2.8 value - 1800 kW at rest from
    ``1800 - 5v``, 7100 kW from ``7100 - 20v`` - would otherwise reach the kernel unchecked.
    351 kW is one kW over the cap, so this tests the boundary rather than a tolerance.
    """
    root = _root(repo)
    _at(root, ("powertrain", "mgu_k", curve))[0]["limit_kw"] = 351.0
    with pytest.raises(ContractError, match=rf"{curve}.*C5\.2\.7 absolute cap"):
        load_car_spec(_write(root, tmp_path)).kernel_config()


def test_the_cap_check_covers_a_hand_built_and_a_replaced_curve(spec: CarSpec) -> None:
    """The check must reach curves that never went through the YAML file.

    ``CarSpec`` is a public frozen dataclass and ``dataclasses.replace`` is the obvious way to
    try a variant, so a cap check confined to ``load_car_spec`` would be trivially bypassable.
    Also pins that the bound is inclusive, because the committed curve sits exactly on it and a
    strict bound would fail the project's own data.
    """
    over_cap = car_spec_module.DeploymentCurve(
        speed_km_h=(0.0, 290.0, 340.0, 345.0),
        limit_kw=(1800.0, 350.0, 100.0, 0.0),
    )
    with pytest.raises(ContractError, match=r"deployment_curve_kw.*C5\.2\.7 absolute cap"):
        replace(spec, deployment_curve=over_cap).kernel_config()

    at_cap = car_spec_module.DeploymentCurve(speed_km_h=(0.0, 100.0), limit_kw=(350.0, 0.0))
    config = replace(spec, deployment_curve=at_cap).kernel_config()
    assert list(config.ers_limit_kw) == [350.0, 0.0]


def _close(actual: float, expected: float) -> bool:
    return abs(actual - expected) < 1e-9


def _interpolate(speeds: np.ndarray, limits: np.ndarray, speed: float) -> float:
    """Piecewise-linear evaluation, matching what a kernel will do with these arrays."""
    index = int(np.searchsorted(speeds, speed, side="right")) - 1
    index = min(max(index, 0), speeds.size - 2)
    span = float(speeds[index + 1] - speeds[index])
    weight = 0.0 if span == 0.0 else (speed - float(speeds[index])) / span
    return float(limits[index]) * (1.0 - weight) + float(limits[index + 1]) * weight


def test_a_derived_point_that_is_not_in_the_curve_is_an_audit_failure(
    tmp_path: Path, repo: Path
) -> None:
    root = _root(repo)
    _at(root, ("powertrain", "mgu_k", "regulation", "deployment_curve_kw"))["derived_points"] = [
        {"speed_km_h": 123.0, "basis": "invented corner"}
    ]
    findings = provenance_audit(load_car_spec(_write(root, tmp_path)).raw)
    assert findings == [
        "powertrain.mgu_k.regulation.deployment_curve_kw.derived_points[123.0]: speed_km_h "
        "123.0 is not a breakpoint of deployment_curve_kw; it has [0.0, 290.0, 340.0, 345.0]"
    ]


def test_a_malformed_derived_point_speed_is_an_audit_finding_not_an_exception(
    tmp_path: Path, repo: Path
) -> None:
    """The audit reports; it does not raise. A finding and a crash are different failures.

    Every other malformed claim shape - a missing clause, a citation naming something that is
    not a value - is already reported as a finding. A derived point with a string where a speed
    belongs is the same class of mistake and must behave the same way, or a typo in the file
    takes out the audit instead of telling anyone about it.
    """
    root = _root(repo)
    _at(root, ("powertrain", "mgu_k", "regulation", "deployment_curve_kw"))["derived_points"] = [
        {"speed_km_h": "290", "basis": "the crossing"}
    ]
    findings = provenance_audit(load_car_spec(_write(root, tmp_path)).raw)
    assert findings == [
        "powertrain.mgu_k.regulation.deployment_curve_kw.derived_points[0].speed_km_h: expected "
        "a number, got '290'"
    ]


def test_a_derived_point_without_a_basis_is_an_audit_failure(tmp_path: Path, repo: Path) -> None:
    root = _root(repo)
    _at(root, ("powertrain", "mgu_k", "regulation", "deployment_curve_kw"))["derived_points"] = [
        {"speed_km_h": 290.0, "basis": "  "}
    ]
    findings = provenance_audit(load_car_spec(_write(root, tmp_path)).raw)
    assert findings == [
        "powertrain.mgu_k.regulation.deployment_curve_kw.derived_points[290.0].basis: expected "
        "a sentence saying why this breakpoint is not in the clause"
    ]


def test_the_overspecified_derived_point_is_the_only_one(spec: CarSpec) -> None:
    """Both ERS curves have exactly one derived knot, the C5.2.7 crossover. Nothing else.

    Every other knot - 340, 345 and 355 km/h, and the 100 kW and 0 kW limits - is stated by
    C5.2.8 verbatim, so declaring more would over-report the synthesis.
    """
    derived = spec.derived_points()
    assert derived == {
        "powertrain.mgu_k.deployment_curve_kw": (290.0,),
        "powertrain.mgu_k.overtake_curve_kw": (337.5,),
    }
    for speeds in derived.values():
        assert speeds in ((290.0,), (337.5,))


def test_the_mass_minimum_carries_the_nominal_tyre_mass_and_the_session(spec: CarSpec) -> None:
    """C4.1's floor is 724 kg *plus* a supplier figure, and it is 726 kg in Qualifying.

    Both halves are in the clause's own wording and both matter to anything that later wants
    to check the floor: 724 kg on its own is not the minimum, and the minimum is a function of
    the session. ``mass.total_kg`` sits well above either, which is why nothing enforces the
    floor in P1 - see ``test_the_c42_floor_is_not_enforced_against_total_mass``.

    The driver reference mass is cited to **C4.5.2**, not C4.5. The Phase 1 design records
    that as a correction (gap G9), and a citation that points one clause out is a wrong claim
    even when the number it carries is right.
    """
    mass = _at(spec.raw, ("mass",))
    assert "Nominal Tyre Mass" in mass["regulation"]["minimum_mass_kg"]["quote"]
    qualifying = mass["regulation"]["minimum_mass_qualifying_kg"]["quote"]
    assert "Nominal Tyre Mass" in qualifying
    assert "Qualifying" in qualifying

    driver = mass["regulation"]["driver_reference_mass_kg"]
    assert (driver["clause"], driver["page"]) == ("C4.5.2", 60)
    assert mass["driver_reference_mass_kg"] == 82.0


def test_the_clutch_demand_is_the_regulated_rear_axle_torque_demand(spec: CarSpec) -> None:
    """C9.2.5 defines what the clutch is *asked* for, not what it can take.

    The committed gearbox clamps transmitted torque to a synthetic 3000 Nm capacity. C9.2.5
    says something else: the driver's clutch request is expressed as torque at the rear axle
    by applying a gain of 5200 Nm over the 5-95 % engagement range - 90 % of travel - the
    controller must track it within +/-150 Nm at the rear axle, and the first 85 ms of a launch
    step is excepted from that band. Four numbers that used to live only in the design note are
    configuration with a clause behind them, so the gearbox task consumes data rather than
    hardcoding a constant.
    """
    config = spec.kernel_config()
    assert config.clutch_demand_torque_nm == 5200.0
    assert config.clutch_demand_travel_fraction == 0.9
    assert config.clutch_control_error_max_nm == 150.0
    assert config.clutch_launch_exception_s == 0.085

    claim = _at(spec.raw, ("gearbox", "regulation", "clutch_demand_torque_nm"))
    assert (claim["clause"], claim["page"]) == ("C9.2.5", 101)


def test_the_shift_time_is_bounded_by_the_c98_4_direction_limits(spec: CarSpec) -> None:
    """The one synthetic shift duration is only allowed to exist because of C9.8.4.

    ``shift_time_s`` is a project number, so the clause is what keeps it honest: an up change
    must complete within 200 ms, a down change within 300 ms, and the original gear must be
    disengaged within 80 ms of the request. A single shared duration is therefore only legal
    below the *smaller* of the two direction limits, which is what the loader now enforces.
    """
    config = spec.kernel_config()
    assert (config.shift_time_max_up_s, config.shift_time_max_down_s) == (0.2, 0.3)
    assert config.shift_disengage_max_s == 0.08
    assert config.shift_time_s <= min(config.shift_time_max_up_s, config.shift_time_max_down_s)
    assert config.shift_disengage_max_s <= 0.08


def test_a_shift_time_beyond_the_regulated_direction_limit_is_rejected(
    tmp_path: Path, repo: Path
) -> None:
    """The bound is inclusive at 200 ms, because the faster direction's limit is the one.

    ``tests/test_gearbox.py`` already builds a variant at five times the committed duration -
    exactly 200 ms - and it has to keep loading, so this pins which side of the bound that is
    and rules out a stricter one.
    """
    root = _root(repo)
    _at(root, ("gearbox",))["shift_time_s"] = 0.2
    at_limit = load_car_spec(_write(root, tmp_path)).kernel_config()
    assert at_limit.shift_time_s == 0.2

    root = _root(repo)
    _at(root, ("gearbox",))["shift_time_s"] = 0.201
    with pytest.raises(ContractError, match=r"shift_time_s.*C9\.8\.4"):
        load_car_spec(_write(root, tmp_path)).kernel_config()


def test_the_reverse_ratio_is_synthetic_and_never_claimed_to_a_clause(spec: CarSpec) -> None:
    """C9.7 requires the car to be drivable in reverse; it states no reverse *ratio*.

    So the ratio is a project choice with a clause behind the requirement and not behind the
    number. Recording it as a value with no `regulation` claim is the point: a reverse gear is
    required, a reverse ratio is invented, and only one of those two is the FIA's.
    """
    gearbox = _at(spec.raw, ("gearbox",))
    assert "reverse_ratio" not in _at(spec.raw, ("gearbox", "regulation"))

    config = spec.kernel_config()
    assert config.reverse_ratio > 0.0
    claim = _at(spec.raw, ("gearbox", "not_regulated", "reverse_ratio"))
    assert "C9.7" in claim
    assert "ynthesised" in claim or "ynthetic" in claim


def test_the_mgu_k_torque_limit_is_referenced_to_crankshaft_speed(spec: CarSpec) -> None:
    """500 Nm is a crankshaft-referenced limit, and 520 Nm is not another cap.

    The reference is what makes the number comparable at all: a motor-shaft figure and a
    crankshaft figure are different quantities, and the same sentence states both, so the quote
    has to carry it. C5.18.4's 520 Nm is a threshold above which an *optional* torque-limiting
    device may act; the Phase 1 design calls out treating it as a cap as a specific failure, so
    it is recorded with that wording and deliberately kept out of ``KernelConfig`` - handing a
    number that must not be used as a limit to the physics would invite exactly that.
    """
    mgu_k = _at(spec.raw, ("powertrain", "mgu_k"))
    claim = _at(spec.raw, ("powertrain", "mgu_k", "regulation", "torque_limit_nm"))
    assert (claim["clause"], claim["page"]) == ("C5.2.11", 65)
    assert "crankshaft" in claim["quote"]

    threshold = mgu_k["regulation"]["transient_torque_limiter_threshold_nm"]
    assert (threshold["clause"], threshold["page"]) == ("C5.18.4", 78)
    assert "not an MGU-K torque cap" in threshold["basis"]
    assert "transient_torque_limiter_threshold_nm" not in fields(car_spec_module.KernelConfig)


def test_the_mgu_k_crank_ratio_is_synthetic_and_the_relative_speed_cap_is_not(
    spec: CarSpec,
) -> None:
    """C5.18.5 caps a number; C5.18.2 requires a property. Only the first states a value.

    The MGU-K is permanently geared to the crankshaft, so joining its torque at the crankshaft
    needs a ratio - and the clause fixes that the ratio is *fixed* without ever stating what it
    is. The relative speed limit is the other half of the same coupling, and it is a real
    number: 60 000 rpm of MGU-K part speed.
    """
    config = spec.kernel_config()
    assert config.mgu_k_crankshaft_ratio > 0.0
    assert config.mgu_k_relative_speed_limit_rpm == 60000.0

    ratio_claim = _at(spec.raw, ("powertrain", "mgu_k", "not_regulated", "crankshaft_ratio"))
    assert "C5.18.2" in ratio_claim
    speed_claim = _at(spec.raw, ("powertrain", "mgu_k", "regulation", "relative_speed_limit_rpm"))
    assert (speed_claim["clause"], speed_claim["page"]) == ("C5.18.5", 78)


def test_the_c52_10_recharge_limits_are_event_conditioned_not_one_universal_cap(
    spec: CarSpec,
) -> None:
    """C5.2.10 states a baseline and three event-conditioned values, not a single limit.

    The Phase 1 design calls out reading the 8.5 MJ baseline as a universal cap as the failure
    this replaces: the article lowers it to 7 MJ in defined circumstances, names 4 MJ as the
    qualifying floor, and allows a conditional extra 0.5 MJ. All four are recorded, and the
    loader refuses a file whose event-conditioned value is *above* the baseline it reduces -
    which is the mistake a single-number reading invites.
    """
    config = spec.kernel_config()
    assert config.recharge_limit_mj_per_lap == 8.5
    assert config.recharge_limit_reduced_mj_per_lap == 7.0
    assert config.recharge_limit_qualifying_floor_mj_per_lap == 4.0
    assert config.recharge_allowance_mj_per_lap == 0.5

    claim = _at(spec.raw, ("powertrain", "mgu_k", "regulation", "recharge_limit_mj_per_lap"))
    assert (claim["clause"], claim["page"]) == ("C5.2.10", 64)
    assert "baseline" in claim["scope"]


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("recharge_limit_reduced_mj_per_lap", 9.0, "exceeds the C5.2.10 baseline"),
        ("recharge_limit_qualifying_floor_mj_per_lap", 7.5, "exceeds the C5.2.10 reduced limit"),
        ("recharge_allowance_mj_per_lap", 0.0, "recharge_allowance_mj_per_lap"),
    ],
)
def test_an_unordered_c52_10_recharge_limit_is_rejected(
    tmp_path: Path, repo: Path, key: str, value: float, message: str
) -> None:
    root = _root(repo)
    _at(root, ("powertrain", "mgu_k"))[key] = value
    with pytest.raises(ContractError, match=message):
        load_car_spec(_write(root, tmp_path)).kernel_config()


def test_the_c52_5_partial_load_limit_is_expressed_in_engine_power(spec: CarSpec) -> None:
    """The other fuel-energy-flow limit is a function of power, and its arms have to meet.

    C5.2.3 and C5.2.4 bound the ICE by rpm; C5.2.5 bounds it by engine power instead, with a
    flat 380 MJ/h arm at or below -50 kW and ``9.78*P + 869`` above it. The threshold is
    therefore negative and must not be caught by a positivity rule - the only value in the ICE
    block that may be below zero.

    The two arms meeting at the threshold is arithmetic, not coincidence, so it is checked
    rather than assumed: a mis-transcribed gain or offset would leave a step in the limit that
    no citation would reveal.
    """
    config = spec.kernel_config()
    assert config.fuel_energy_flow_partial_load_threshold_kw == -50.0
    assert config.fuel_energy_flow_partial_load_min_mj_h == 380.0
    assert config.fuel_energy_flow_partial_load_gain == 9.78
    assert config.fuel_energy_flow_partial_load_offset_mj_h == 869.0

    arms = (
        config.fuel_energy_flow_partial_load_gain
        * config.fuel_energy_flow_partial_load_threshold_kw
    )
    assert _close(
        arms + config.fuel_energy_flow_partial_load_offset_mj_h,
        config.fuel_energy_flow_partial_load_min_mj_h,
    ), f"C5.2.5's arms leave a step: {arms} + offset != min"

    claim = _at(
        spec.raw,
        ("powertrain", "ice", "regulation", "fuel_energy_flow_partial_load_gain_mj_h_per_kw"),
    )
    assert (claim["clause"], claim["page"]) == ("C5.2.5", 64)


def test_the_two_efficiency_conversions_are_declared_synthetic_and_read_as_data(
    spec: CarSpec,
) -> None:
    """The regulations bound power in fuel MJ/h and electrical kW, and state neither efficiency.

    C5.2.3 through C5.2.5 cap the ICE by fuel *energy* flow while everything the model computes
    at the crank is *shaft* power, and C5.2.7 caps the MGU-K in electrical DC power while the
    model holds mechanical shaft torque. Both conversions need an efficiency the document does
    not publish, so both are project numbers and are labelled as such rather than quoted to a
    clause. A test that let them look regulated would be claiming a number the FIA never stated.
    """
    config = spec.kernel_config()
    assert 0.0 < config.fuel_to_shaft_efficiency < 1.0
    assert 0.0 < config.mgu_k_motor_inverter_efficiency < 1.0

    ice_claim = _at(spec.raw, ("powertrain", "ice", "not_regulated", "fuel_to_shaft_efficiency"))
    mgu_claim = _at(spec.raw, ("powertrain", "mgu_k", "not_regulated", "motor_inverter_efficiency"))
    assert "C5.2.3" in ice_claim and "C5.2.5" in ice_claim
    assert "C5.2.7" in mgu_claim
    assert "not a regulated value" in ice_claim.lower()
    assert "not a regulated value" in mgu_claim.lower()

    citations = spec.citations()
    assert "powertrain.ice.fuel_to_shaft_efficiency" not in citations
    assert "powertrain.mgu_k.motor_inverter_efficiency" not in citations


def test_the_c52_3_per_cylinder_arm_is_recorded_and_the_file_says_it_is_not_enforced(
    spec: CarSpec,
) -> None:
    """C5.2.3's per-cylinder figure is cited but stays out of the physics, and the file says why.

    The per-cylinder limit is a real clause and its number is in the file with its quote, but
    enforcing it needs a cylinder count, which no block of ``car_spec.yaml`` carries and which
    this project's verified clause set does not cite. Rather than invent one, the arm is recorded
    and named as unenforced: an unrecorded omission is indistinguishable from an oversight, and a
    silently invented cylinder count would be an unsourced number doing regulatory work.
    """
    value = _at(spec.raw, ("powertrain", "ice", "fuel_energy_flow_per_cylinder_max_mj_h"))
    assert value == 550.0
    note = _at(spec.raw, ("powertrain", "ice", "note"))
    assert "per-cylinder" in note
    assert "C5.2.3" in note

    config = spec.kernel_config()
    assert "fuel_energy_flow_per_cylinder_max_mj_h" not in config.__dataclass_fields__


def test_the_gearbox_section_is_mixed_and_says_which_number_is_which(spec: CarSpec) -> None:
    """`gearbox` moved from `synthesised` to `mixed`, and both claim blocks are load-bearing.

    The ratios, the shift points, the shift duration, the reverse ratio and the clutch
    capacity are all project choices. C9.2.5's demand model, C9.8.4's shifting timings and
    C9.6.1's eight forward ratios are not, so a section claiming `synthesised` over all of it
    would describe the opposite of the truth.
    """
    gearbox = _at(spec.raw, ("gearbox",))
    assert gearbox["provenance"] == "mixed"
    assert set(gearbox["regulation"]) <= set(gearbox)
    assert set(gearbox["not_regulated"]) <= set(gearbox)
    assert "C9.2.5" in "".join(str(value) for value in gearbox["regulation"].values())


def test_the_p1_section_that_carries_the_regulated_powertrain_limits_is_populated(
    spec: CarSpec,
) -> None:
    """Guards the shape of the deployment curves the kernel will index."""
    config = spec.kernel_config()
    assert config.ers_speed_km_h[0] == 0.0
    assert config.ers_limit_kw[-1] == 0.0
    assert config.ers_overtake_limit_kw[-1] == 0.0
    assert np.all(np.diff(config.ers_speed_km_h) > 0.0)
    assert np.all(np.diff(config.ers_overtake_speed_km_h) > 0.0)
    assert config.store_energy_mj == 4.0
    assert config.launch_speed_kmh == 50.0


def test_every_value_is_either_cited_or_explained_as_not_regulated(spec: CarSpec) -> None:
    assert provenance_audit(spec.raw) == []


def test_sections_without_a_regulation_basis_say_so(spec: CarSpec) -> None:
    """`constants` is a standards block; the rest are outright project decisions.

    Both states are recorded rather than left implicit, so "no FIA article covers this"
    cannot be confused with "nobody checked".
    """
    assert set(NOT_REGULATED_SECTIONS).isdisjoint(EXPECTED_CITATIONS)
    for section in NOT_REGULATED_SECTIONS:
        block = _at(spec.raw, (section,))
        assert "regulation" not in block, section
        assert "not_regulated" in block, section
    constants = _at(spec.raw, ("constants",))
    assert constants["provenance"] == "standard"
    assert "regulation" not in constants


def test_a_citation_without_a_clause_is_an_audit_failure(tmp_path: Path, repo: Path) -> None:
    root = _root(repo)
    del _at(root, ("mass", "regulation", "minimum_mass_kg"))["clause"]
    findings = provenance_audit(load_car_spec(_write(root, tmp_path)).raw)
    assert findings == ["mass.regulation.minimum_mass_kg: missing key(s) ['clause']"]


def test_a_citation_for_a_value_that_is_not_there_is_an_audit_failure(
    tmp_path: Path, repo: Path
) -> None:
    root = _root(repo)
    _at(root, ("mass", "regulation"))["minimum_mass_tyres_kg"] = {"clause": "C4.1", "page": 59}
    findings = provenance_audit(load_car_spec(_write(root, tmp_path)).raw)
    assert findings == [
        "mass.regulation: 'minimum_mass_tyres_kg' is not a value or curve in this block"
    ]


def test_an_uncited_number_in_a_regulated_section_is_an_audit_failure(
    tmp_path: Path, repo: Path
) -> None:
    """A section declared `mixed` may only hold values that carry a clause or a reason."""
    root = _root(repo)
    _at(root, ("mass",))["tyre_mass_kg"] = 40.0
    findings = provenance_audit(load_car_spec(_write(root, tmp_path)).raw)
    assert findings == ["mass: 'tyre_mass_kg' is a value with no regulation citation"]


@pytest.mark.parametrize(
    ("drop", "message"),
    [
        (2, "aero.cl_curve and aero.cd_curve must use the same speed_m_s breakpoints"),
        (1, "aero.cl_curve and aero.cd_curve must use the same speed_m_s breakpoints"),
    ],
)
def test_a_drag_curve_on_a_different_speed_grid_is_rejected(
    tmp_path: Path, repo: Path, drop: int, message: str
) -> None:
    """Cl and Cd share one speed axis in ``KernelConfig``, so the grids must agree.

    A Cd curve with different breakpoints would silently produce a ``cd`` array of a
    different length to the ``aero_speed_m_s`` axis it is indexed by, which is an
    out-of-bounds read inside the kernel rather than a load error. The loader is the only
    place that can catch it.
    """
    root = _root(repo)
    del _at(root, ("aero",))["cd_curve"][drop]
    with pytest.raises(ContractError, match=message):
        load_car_spec(_write(root, tmp_path)).kernel_config()


def test_a_cl_and_cd_grid_mismatch_is_caught_even_when_the_lengths_agree(
    tmp_path: Path, repo: Path
) -> None:
    """Equal length is not enough: the same count at different speeds is still a mismatch."""
    root = _root(repo)
    _at(root, ("aero",))["cd_curve"][2]["speed_m_s"] = 41.0
    with pytest.raises(ContractError, match=r"aero\.cl_curve and aero\.cd_curve"):
        load_car_spec(_write(root, tmp_path)).kernel_config()


@pytest.mark.parametrize("value", [0.0, 1.0, -0.2, 1.5])
def test_front_weight_fraction_must_be_a_strict_fraction(
    tmp_path: Path, repo: Path, value: float
) -> None:
    """It is a fraction of a total mass, so it has no meaning outside (0, 1).

    Note what is deliberately *not* checked: C4.2's 0.44 / 0.54 are fractions of the
    regulatory Minimum Mass, which includes a separately published Nominal Tyre Mass, not
    fractions of ``mass.total_kg``. See ``test_the_c42_floor_is_not_enforced_against_total_mass``.
    """
    root = _root(repo)
    _at(root, ("chassis",))["front_weight_fraction"] = value
    with pytest.raises(ContractError, match=r"chassis\.front_weight_fraction must be in \(0, 1\)"):
        load_car_spec(_write(root, tmp_path)).kernel_config()


def test_the_c42_floor_is_not_enforced_against_total_mass(spec: CarSpec) -> None:
    """C4.2 cannot be enforced here, and pretending otherwise would be wrong.

    C4.2 reads "the mass measured at the front axle must not be less than the Minimum Mass
    specified in Article C4.1 factored by 0.44". The Minimum Mass of C4.1 is *724 kg plus the
    Nominal Tyre Mass*, and the Nominal Tyre Mass is published by the tyre supplier after the
    final tyre-testing camp (C4.7) - it is not in the regulations and this project does not
    have it. So the floor applies to ``minimum_mass_kg + nominal_tyre_mass_kg``, not to
    ``mass.total_kg``, and comparing 0.46 against 0.44 as if both were fractions of the same
    quantity would enforce a rule that does not exist.

    What *is* enforced is the part that needs no missing input: the value is a fraction. The
    two floors stay in ``car_spec.yaml`` as cited regulation data, and neither reaches
    ``KernelConfig`` - the longitudinal kernel has no axle, so a P1 kernel would only misuse
    them. P2-T2, which owns the axial split, is where C4.2 becomes checkable.
    """
    config = spec.kernel_config()
    assert 0.0 < config.front_weight_fraction < 1.0

    assert "minimum_front_axle_fraction" not in fields(car_spec_module.KernelConfig)
    assert "minimum_rear_axle_fraction" not in fields(car_spec_module.KernelConfig)
    # The floors are still recorded, with their clause, even though nothing consumes them yet.
    chassis = _at(spec.raw, ("chassis",))
    assert chassis["minimum_front_axle_fraction"] == 0.44
    assert chassis["minimum_rear_axle_fraction"] == 0.54
    assert chassis["regulation"]["minimum_front_axle_fraction"]["clause"] == "C4.2"
    # C4.2 is a Qualifying-only check, which is most of why P1 scenarios cannot be judged by it.
    assert (
        "Qualifying and Sprint Qualifying"
        in chassis["regulation"]["minimum_front_axle_fraction"]["quote"]
    )
    # And the missing input that blocks enforcement is named in the file, not glossed over -
    # as an obtainable published figure, not as something fundamentally unavailable.
    enforcement = chassis["c42_enforcement"]
    assert enforcement["status"] == "not_enforced"
    assert "Nominal Tyre Mass" in enforcement["scope"]
    missing = enforcement["missing_input"]
    assert missing["name"] == "nominal_tyre_mass_kg"
    assert "C4.7" in missing["source"]
    assert "obtainable" in missing["availability"]
    assert enforcement["becomes_checkable_at"].startswith("P2-T2")
    assert any("nominal_tyre_mass_kg" in item for item in enforcement["blocked_by"])


def test_kernel_config_hands_over_plain_float64_arrays(spec: CarSpec) -> None:
    config = spec.kernel_config()
    arrays = _arrays(config)
    assert arrays, "the kernel config must hand over arrays, not only scalars"
    for name, array in arrays.items():
        assert array.dtype == np.float64, name
        assert array.ndim == 1, name
        assert array.flags["C_CONTIGUOUS"], name
        assert array.flags["WRITEABLE"], name
        assert np.all(np.isfinite(array)), name
    assert config.aero_speed_m_s.shape == config.cl.shape == config.cd.shape
    assert config.torque_rpm.shape == config.torque_nm.shape
    assert config.gear_ratios.shape == (8,)
    assert config.ers_speed_km_h.shape == config.ers_limit_kw.shape
    assert config.ers_overtake_speed_km_h.shape == config.ers_overtake_limit_kw.shape
    assert config.dt_s == pytest.approx(1.0e-4)
    assert config.mass_kg == spec.mass_kg
    assert config.rolling_radius_m == spec.rolling_radius_m


def test_editing_the_yaml_changes_the_kernel_config_with_no_code_edit(
    tmp_path: Path, repo: Path
) -> None:
    """P1-T1b: calibration is a data edit, not a refactor."""
    root = _root(repo)
    before = load_car_spec(_write(root, tmp_path)).kernel_config()

    _at(root, ("mass",))["total_kg"] = 780.0
    _at(root, ("gearbox",))["final_drive"] = 3.4
    _at(root, ("aero", "cl_curve", 2))["cl"] = 2.9
    _at(root, ("tyres", "longitudinal_pacejka"))["mu"] = 1.85
    _at(root, ("tyres",))["slip_ratio_min_speed_m_s"] = 2.0
    _at(root, ("powertrain", "ice"))["torque_curve_nm"][4]["torque_nm"] = 340.0
    after = load_car_spec(_write(root, tmp_path)).kernel_config()

    assert before.mass_kg == 800.0
    assert after.mass_kg == 780.0
    assert after.final_drive == 3.4
    assert after.cl[2] == 2.9
    assert after.pacejka_mu == 1.85
    assert after.slip_ratio_min_speed_m_s == 2.0
    assert after.torque_nm[4] == 340.0


def test_the_longitudinal_tyre_coefficients_reach_the_kernel_config(spec: CarSpec) -> None:
    """P1-T3's four Magic Formula coefficients and the slip guard are handed over as numbers.

    Scalars, not a mapping, because a kernel cannot read a dict (PLAN.md section 4.1 rule 1) and
    because the values are asserted against the file rather than against themselves: a builder
    that read the right keys and attached them to the wrong fields would still satisfy a test
    that only checked they were present. What the coefficients *do* is the force model's business
    and is asserted in ``tests/test_forces.py``.
    """
    config = spec.kernel_config()
    pacejka = _at(spec.raw, ("tyres", "longitudinal_pacejka"))
    assert config.pacejka_b == pacejka["b"] > 0.0
    assert config.pacejka_c == pacejka["c"] > 0.0
    assert config.pacejka_e == pacejka["e"]
    assert config.pacejka_mu == pacejka["mu"] > 0.0
    guard = _at(spec.raw, ("tyres",))["slip_ratio_min_speed_m_s"]
    assert config.slip_ratio_min_speed_m_s == guard > 0.0
    # The wheel inertia is the fourth number the P1-T6 rotational state needs, and it is claimed
    # as synthesised for the reason the claim block says: PLAN.md section 4 gives a band and no
    # clause fixes it.
    inertia = _at(spec.raw, ("tyres",))["wheel_inertia_kg_m2"]
    assert config.wheel_inertia_kg_m2 == inertia > 0.0
    inertia_claim = _at(spec.raw, ("tyres", "not_regulated", "wheel_inertia_kg_m2"))
    assert "inertia" in inertia_claim
    assert "PLAN.md" in inertia_claim
    # Both are claimed as synthesised, and the claim says which model it is, so a later edit
    # that imports a published parameter set without changing the claim fails the audit's eye.
    claim = _at(spec.raw, ("tyres", "not_regulated", "longitudinal_pacejka"))
    assert "Pacejka" in claim
    assert "ynthesised" in claim
    assert "eps" in _at(spec.raw, ("tyres", "not_regulated", "slip_ratio_min_speed_m_s"))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("mass", "total_kg"), 0.0, "mass.total_kg"),
        (("mass", "total_kg"), -5.0, "mass.total_kg"),
        (("constants", "air_density_kg_m3"), 0.0, "air_density_kg_m3"),
        (("aero", "reference_area_m2"), 0.0, "reference_area_m2"),
        (("aero", "cl_curve", 1, "cl"), 0.0, "aero.cl_curve"),
        (("aero", "cd_curve", 0, "cd"), -0.5, "aero.cd_curve"),
        (("integration", "dt_s"), 0.0, "integration.dt_s"),
        (("powertrain", "power_split_ice"), 1.5, "power_split_ice"),
        (("powertrain", "ice", "rev_limit_rpm"), 2000.0, "rev_limit_rpm"),
        (("powertrain", "ice", "peak_power_kw"), 0.0, "peak_power_kw"),
        (
            ("powertrain", "ice", "turbo_lag", "multiplier_at_collapse"),
            1.5,
            "multiplier_at_collapse",
        ),
        (
            ("powertrain", "ice", "fuel_energy_flow_partial_load_gain_mj_h_per_kw"),
            0.0,
            "fuel_energy_flow_partial_load_gain_mj_h_per_kw",
        ),
        (
            ("powertrain", "ice", "fuel_energy_flow_partial_load_offset_mj_h"),
            0.0,
            "fuel_energy_flow_partial_load_offset_mj_h",
        ),
        (
            ("powertrain", "ice", "fuel_energy_flow_partial_load_min_mj_h"),
            0.0,
            "fuel_energy_flow_partial_load_min_mj_h",
        ),
        # C5.2.5's threshold is the one number in the ICE block that may be negative: it is an
        # engine *power*, and the clause's flat arm starts at or below -50 kW. Only a
        # non-finite value is refused for it, never a negative one.
        (
            ("powertrain", "ice", "fuel_energy_flow_partial_load_threshold_kw"),
            math.inf,
            "fuel_energy_flow_partial_load_threshold_kw",
        ),
        (("powertrain", "mgu_k", "peak_power_kw"), 0.0, "mgu_k.peak_power_kw"),
        (("powertrain", "mgu_k", "store_energy_mj"), 0.0, "store_energy_mj"),
        (("powertrain", "mgu_k", "deployment_curve_kw", 2, "limit_kw"), -1.0, "limit_kw"),
        (("powertrain", "mgu_k", "crankshaft_ratio"), 0.0, "crankshaft_ratio"),
        (
            ("powertrain", "mgu_k", "relative_speed_limit_rpm"),
            -1.0,
            "relative_speed_limit_rpm",
        ),
        # The two efficiency conversions are the only numbers in the powertrain block that are
        # neither regulated nor bounded by a clause, and a fraction is the rule: a zero divides by
        # nothing useful and an above-one would report more energy out of the motor than the
        # regulations cap it delivering.
        (
            ("powertrain", "ice", "fuel_to_shaft_efficiency"),
            0.0,
            "fuel_to_shaft_efficiency",
        ),
        (
            ("powertrain", "ice", "fuel_to_shaft_efficiency"),
            1.01,
            "fuel_to_shaft_efficiency",
        ),
        (
            ("powertrain", "mgu_k", "motor_inverter_efficiency"),
            0.0,
            "motor_inverter_efficiency",
        ),
        (
            ("powertrain", "mgu_k", "motor_inverter_efficiency"),
            math.nan,
            "motor_inverter_efficiency",
        ),
        (("gearbox", "final_drive"), 0.0, "final_drive"),
        (("gearbox", "reverse_ratio"), 0.0, "reverse_ratio"),
        (("gearbox", "clutch_demand_torque_nm"), 0.0, "clutch_demand_torque_nm"),
        (("gearbox", "clutch_demand_travel_fraction"), 1.5, "clutch_demand_travel_fraction"),
        (("gearbox", "clutch_control_error_max_nm"), 0.0, "clutch_control_error_max_nm"),
        (("gearbox", "clutch_launch_exception_s"), -0.01, "clutch_launch_exception_s"),
        (("gearbox", "shift_time_max_up_s"), 0.0, "shift_time_max_up_s"),
        (("gearbox", "shift_time_max_down_s"), 0.0, "shift_time_max_down_s"),
        (("gearbox", "shift_disengage_max_s"), 0.0, "shift_disengage_max_s"),
        (("gearbox", "shift_time_max_up_s"), 0.201, "C9.8.4.*200 ms"),
        (("gearbox", "shift_time_max_down_s"), 0.301, "C9.8.4.*300 ms"),
        (("gearbox", "shift_disengage_max_s"), 0.081, "C9.8.4.*80 ms"),
        (("gearbox", "shift_up_rpm"), 20000.0, "shift_up_rpm"),
        (("gearbox", "shift_down_rpm"), 13000.0, "shift_down_rpm"),
        (("gearbox", "shift_time_s"), -0.01, "shift_time_s"),
        # P1-T5: zero is now refused too, not just negative values. `physics.gearbox.step_gearbox`
        # needs the shift timer to be strictly positive, because that timer is the only thing that
        # freezes the gear while a shift runs; at zero an rpm sitting on the upshift point advances
        # the box a gear per step. The loader and the step have to agree on the rule.
        (("gearbox", "shift_time_s"), 0.0, "shift_time_s"),
        (("tyres", "rolling_radius_m"), 0.0, "rolling_radius_m"),
        (("tyres", "wheel_diameter_m"), 0.0, "wheel_diameter_m"),
        (("tyres", "slip_ratio_min_speed_m_s"), 0.0, "slip_ratio_min_speed_m_s"),
        (("tyres", "slip_ratio_min_speed_m_s"), math.inf, "slip_ratio_min_speed_m_s"),
        (("tyres", "longitudinal_pacejka", "b"), 0.0, "longitudinal_pacejka.b"),
        (("tyres", "longitudinal_pacejka", "c"), -1.0, "longitudinal_pacejka.c"),
        (("tyres", "longitudinal_pacejka", "e"), math.inf, "longitudinal_pacejka.e"),
        (("tyres", "longitudinal_pacejka", "mu"), 0.0, "longitudinal_pacejka.mu"),
        # P1-T6/T7: the wheel rotational state divides by the inertia, so zero is a NaN wheel
        # rather than a car that happens to have no wheel mass. It is synthesised (PLAN.md section
        # 4 gives 0.5-1.2 kg.m^2 for wheel plus tyre), so the loader's only job is to keep it a
        # usable magnitude.
        (("tyres", "wheel_inertia_kg_m2"), 0.0, "wheel_inertia_kg_m2"),
        (("tyres", "wheel_inertia_kg_m2"), -0.9, "wheel_inertia_kg_m2"),
        (("chassis", "wheelbase_m"), 0.0, "wheelbase_m"),
    ],
)
def test_invalid_p1_configuration_fails_in_python_before_the_kernel(
    tmp_path: Path, repo: Path, path: tuple[Any, ...], value: float, message: str
) -> None:
    root = _root(repo)
    _at(root, path[:-1])[path[-1]] = value
    with pytest.raises(ContractError, match=message):
        load_car_spec(_write(root, tmp_path)).kernel_config()


@pytest.mark.parametrize(
    ("path", "message"),
    [
        (("gearbox", "ratios"), "ratios"),
        (("powertrain", "ice", "torque_curve_nm"), "torque_curve_nm"),
        (("powertrain", "mgu_k", "deployment_curve_kw"), "limit_kw"),
        (("tyres", "longitudinal_pacejka"), "longitudinal_pacejka"),
    ],
)
def test_a_missing_curve_or_ratio_list_fails_at_the_python_boundary(
    tmp_path: Path, repo: Path, path: tuple[str, ...], message: str
) -> None:
    root = _root(repo)
    del _at(root, path[:-1])[path[-1]]
    with pytest.raises(ContractError, match=message):
        load_car_spec(_write(root, tmp_path)).kernel_config()


@pytest.mark.parametrize("section", ["mass", "aero", "gearbox", "integration", "tyres"])
def test_a_missing_section_is_rejected_before_any_array_is_built(
    tmp_path: Path, repo: Path, section: str
) -> None:
    root = _root(repo)
    del root[section]
    with pytest.raises(ContractError, match="missing section"):
        load_car_spec(_write(root, tmp_path))


def test_kernel_config_arrays_are_writable_so_the_caller_owns_them(spec: CarSpec) -> None:
    config = spec.kernel_config()
    config.gear_ratios[0] = 4.0
    assert config.gear_ratios[0] == 4.0


def test_two_builds_from_one_spec_agree(spec: CarSpec) -> None:
    """Reproducibility starts here: the same spec must produce the same configuration."""
    first = spec.kernel_config()
    second = spec.kernel_config()
    for field in fields(car_spec_module.KernelConfig):
        value = getattr(first, field.name)
        other = getattr(second, field.name)
        if isinstance(value, np.ndarray):
            assert np.array_equal(value, other), field.name
        else:
            assert value == other, field.name


def test_a_replaced_spec_rebuilds_its_kernel_config(spec: CarSpec) -> None:
    """`dataclasses.replace` must not leave a stale configuration behind.

    A cached `KernelConfig` built at load time survives `replace`, so a replaced spec reports
    the *old* mass while carrying a new one - the typed field and the config disagree, and
    nothing notices. Building on access means the config is always derived from the fields it
    is asked about.
    """
    replaced = replace(spec, mass_kg=900.0)
    assert replaced.kernel_config().mass_kg == 900.0
    assert spec.kernel_config().mass_kg == 800.0


def test_a_replaced_spec_still_gets_its_values_validated(spec: CarSpec) -> None:
    """Validation must not depend on how the spec was constructed.

    `CarSpec` is a public frozen dataclass and `replace` is the obvious way to build a variant
    - a scenario setting a different mass, say. If validation only ran inside `load_car_spec`,
    a replaced spec would hand an unchecked value straight to the kernel.
    """
    with pytest.raises(ContractError, match="rolling_radius_m"):
        replace(spec, rolling_radius_m=0.0).kernel_config()
    with pytest.raises(ContractError, match="power_split_ice"):
        replace(spec, power_split_ice=1.5).kernel_config()
    with pytest.raises(ContractError, match="front_weight_fraction"):
        replace(
            spec,
            raw={
                **spec.raw,
                "chassis": {**_at(spec.raw, ("chassis",)), "front_weight_fraction": 1.5},
            },
        ).kernel_config()


def test_the_audit_rejects_a_spec_claiming_to_be_calibrated(tmp_path: Path, repo: Path) -> None:
    root = _root(repo)
    _at(root, ("spec",))["calibration_status"] = "calibrated"
    findings = provenance_audit(load_car_spec(_write(root, tmp_path)).raw)
    assert findings == [
        "spec: calibration_status is 'calibrated'; this phase must not ship a car spec that claims "
        "to be calibrated"
    ]
