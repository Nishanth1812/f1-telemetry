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
document it claims to cite. The values themselves are re-checked against the clause wording in
``test_the_cited_clauses_still_say_what_the_values_claim``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from f1telemetry.contracts.car_spec import (
    CarSpec,
    ContractError,
    KernelConfig,
    load_car_spec,
    provenance_audit,
)

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
    "mass.driver_reference_mass_kg": ("C4.5", 60),
    "powertrain.ice.idle_rpm": ("C5.13.4", 75),
    "powertrain.ice.fuel_energy_flow_max_mj_h": ("C5.2.3", 64),
    "powertrain.ice.fuel_energy_flow_per_cylinder_max_mj_h": ("C5.2.3", 64),
    "powertrain.ice.fuel_energy_flow_low_rpm_gain_mj_h_per_rpm": ("C5.2.4", 64),
    "powertrain.ice.fuel_energy_flow_low_rpm_offset_mj_h": ("C5.2.4", 64),
    "powertrain.ice.fuel_energy_flow_low_rpm_limit_rpm": ("C5.2.4", 64),
    "powertrain.mgu_k.peak_power_kw": ("C5.2.7", 64),
    "powertrain.mgu_k.deployment_curve_kw": ("C5.2.8", 64),
    "powertrain.mgu_k.overtake_curve_kw": ("C5.2.8", 64),
    "powertrain.mgu_k.store_energy_mj": ("C5.2.9", 64),
    "powertrain.mgu_k.recharge_limit_mj_per_lap": ("C5.2.10", 64),
    "powertrain.mgu_k.torque_limit_nm": ("C5.2.11", 65),
    "powertrain.mgu_k.launch_speed_kmh": ("C5.2.12", 65),
    "tyres.front_width_mm": ("C10.7.2", 111),
    "tyres.rear_width_mm": ("C10.7.2", 111),
    "tyres.rim_diameter_mm": ("C10.7.2", 111),
    "chassis.overall_width_m": ("C2.3.1", 10),
    "chassis.wheelbase_m": ("C2.3.3", 11),
    "chassis.minimum_front_axle_fraction": ("C4.2", 59),
    "chassis.minimum_rear_axle_fraction": ("C4.2", 59),
}

# Sections that state no FIA basis at all, so every number in them is a project decision.
NOT_REGULATED_SECTIONS = ("aero", "gearbox", "integration")


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


def test_the_cited_clauses_still_say_what_the_values_claim(spec: CarSpec) -> None:
    """Re-check the citations against the wording of Issue 20.

    A citation table can be internally consistent and still cite the wrong clause. These
    assertions pin the numbers the quoted clauses actually state.
    """
    mass = _at(spec.raw, ("mass",))
    ice = _at(spec.raw, ("powertrain", "ice"))
    mgu_k = _at(spec.raw, ("powertrain", "mgu_k"))
    tyres = _at(spec.raw, ("tyres",))
    chassis = _at(spec.raw, ("chassis",))

    # C4.1: 724kg plus nominal tyre mass; 726kg in Qualifying and Sprint Qualifying.
    assert mass["minimum_mass_kg"] == 724.0
    assert mass["minimum_mass_qualifying_kg"] == 726.0
    # C4.5.2: reference driver mass plus driver ballast is not less than 82kg.
    assert mass["driver_reference_mass_kg"] == 82.0
    # C4.2: front axle at least 0.44 of the minimum mass, rear axle at least 0.54.
    assert chassis["minimum_front_axle_fraction"] == 0.44
    assert chassis["minimum_rear_axle_fraction"] == 0.54
    # C5.13.4: the idle speed control target may not exceed 4,000rpm.
    assert ice["idle_rpm"] == 4000.0
    # C5.2.3 and C5.2.4: fuel energy flow limits and the limit curve below 10,500rpm.
    assert ice["fuel_energy_flow_max_mj_h"] == 3000.0
    assert ice["fuel_energy_flow_per_cylinder_max_mj_h"] == 550.0
    assert ice["fuel_energy_flow_low_rpm_limit_rpm"] == 10500.0
    assert ice["fuel_energy_flow_low_rpm_gain_mj_h_per_rpm"] == 0.27
    assert ice["fuel_energy_flow_low_rpm_offset_mj_h"] == 165.0
    # C5.2.7 to C5.2.12.
    assert mgu_k["peak_power_kw"] == 350.0
    assert mgu_k["store_energy_mj"] == 4.0
    assert mgu_k["recharge_limit_mj_per_lap"] == 8.5
    assert mgu_k["torque_limit_nm"] == 500.0
    assert mgu_k["launch_speed_kmh"] == 50.0
    # C5.2.8.i: P(kW) = 1800 - 5v below 340kph, 6900 - 20v to 345kph, zero from 345kph.
    normal = {float(p["speed_km_h"]): float(p["limit_kw"]) for p in mgu_k["deployment_curve_kw"]}
    assert normal == {0.0: 1800.0, 290.0: 350.0, 340.0: 100.0, 345.0: 0.0}
    # C5.2.8.ii: Overtake, P(kW) = 7100 - 20v below 355kph, zero from 355kph.
    overtake = {float(p["speed_km_h"]): float(p["limit_kw"]) for p in mgu_k["overtake_curve_kw"]}
    assert overtake == {0.0: 7100.0, 355.0: 0.0}
    # C10.7.2: tyre mounting width 315mm front and 401.3mm rear on a 462.5mm rim.
    assert tyres["rim_diameter_mm"] == 462.5
    assert tyres["front_width_mm"] == 315.0
    assert tyres["rear_width_mm"] == 401.3
    # C2.3.1 and C2.3.3: 950mm from the centreline, wheelbase no more than 3400mm.
    assert chassis["overall_width_m"] == 1.9
    assert chassis["wheelbase_m"] == 3.4


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
    _at(root, ("aero",))["cl_curve"][2]["cl"] = 2.9
    _at(root, ("powertrain", "ice"))["torque_curve_nm"][4]["torque_nm"] = 340.0
    after = load_car_spec(_write(root, tmp_path)).kernel_config()

    assert before.mass_kg == 800.0
    assert after.mass_kg == 780.0
    assert after.final_drive == 3.4
    assert after.cl[2] == 2.9
    assert after.torque_nm[4] == 340.0


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
        (("powertrain", "mgu_k", "peak_power_kw"), 0.0, "mgu_k.peak_power_kw"),
        (("powertrain", "mgu_k", "store_energy_mj"), 0.0, "store_energy_mj"),
        (("powertrain", "mgu_k", "deployment_curve_kw", 2, "limit_kw"), -1.0, "limit_kw"),
        (("gearbox", "final_drive"), 0.0, "final_drive"),
        (("gearbox", "shift_up_rpm"), 20000.0, "shift_up_rpm"),
        (("gearbox", "shift_down_rpm"), 13000.0, "shift_down_rpm"),
        (("gearbox", "shift_time_s"), -0.01, "shift_time_s"),
        (("tyres", "rolling_radius_m"), 0.0, "rolling_radius_m"),
        (("tyres", "wheel_diameter_m"), 0.0, "wheel_diameter_m"),
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
    for field in fields(KernelConfig):
        value = getattr(first, field.name)
        other = getattr(second, field.name)
        if isinstance(value, np.ndarray):
            assert np.array_equal(value, other), field.name
        else:
            assert value == other, field.name


def test_the_audit_rejects_a_spec_claiming_to_be_calibrated(tmp_path: Path, repo: Path) -> None:
    root = _root(repo)
    _at(root, ("spec",))["calibration_status"] = "calibrated"
    findings = provenance_audit(load_car_spec(_write(root, tmp_path)).raw)
    assert findings == [
        "spec: calibration_status is 'calibrated'; P0 must not ship a car spec that claims "
        "to be calibrated"
    ]
