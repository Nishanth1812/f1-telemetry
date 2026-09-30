"""The contract, its loader, and everything code-generated from it (P0-T2 to P0-T6).

The load-bearing claim of P0 is that ``channels.yaml`` is the single source of truth and
the generated artifacts cannot drift from it. These tests are what make that claim
checkable rather than aspirational: they assert the contract's own completeness, that the
generated registry, Parquet schema and TypeScript output all cover the same channel set,
and that the CI gate -- "regenerate and fail on diff" -- actually fires.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from f1telemetry.codegen.__main__ import build_artifacts, check, main
from f1telemetry.contracts.car_spec import CarSpec, load_car_spec, provenance_audit
from f1telemetry.contracts.channels import (
    CORNERS,
    DTYPES,
    FAULT_TYPES,
    ChannelContract,
    ContractError,
    load_channel_contract,
)
from f1telemetry.generated.channels import CHANNELS, SAMPLES_PER_SECOND, load_contract
from f1telemetry.generated.parquet_schema import PARQUET_SCHEMAS

pytestmark = pytest.mark.codegen

# The generated surface PHASES.md P0-T4 and P0-T5 ask for, and nothing else. The CAN-FD
# frame layout is P5-T5: it needs an encode/decode path that does not exist yet, so
# pinning it here is how it came to exist.
P0_GENERATED_PATHS = frozenset(
    {
        "src/f1telemetry/generated/channels.py",
        "src/f1telemetry/generated/parquet_schema.py",
        "web/src/generated/channels.ts",
    }
)

PLAN_GROUPS = (
    "chassis",
    "imu",
    "powertrain",
    "aero",
    "wheel",
    "wheel_thermal",
    "thermal",
    "session",
    "driver",
)

PLAN_CHANNELS = {
    "chassis": (
        "speed",
        "vx",
        "vy",
        "yaw_rate",
        "accel_lateral",
        "accel_longitudinal",
        "roll",
        "pitch",
    ),
    "imu": ("imu_accel_x", "imu_accel_y", "imu_accel_z", "imu_gyro_x", "imu_gyro_y", "imu_gyro_z"),
    "powertrain": (
        "ice_rpm",
        "mgu_k_rpm",
        "gear",
        "throttle_pct",
        "clutch_pct",
        "ice_torque_nm",
        "mgu_k_power_kw",
        "boost_remaining_bar",
        "eso_pct",
        "fuel_flow_kg_h",
    ),
    "aero": ("aero_mode", "fw_flap_deg", "rw_flap_deg", "zone_id", "downforce_n", "cl_a", "cd_a"),
    "wheel": ("wheel_speed", "slip_ratio", "slip_angle", "vertical_load", "camber"),
    "wheel_thermal": ("suspension_travel", "tyre_pressure", "tyre_temp", "brake_temp"),
    "thermal": ("engine_temp", "gearbox_temp", "coolant_temp", "brake_duct_air_temp"),
    "session": ("lap_index", "sector_index", "lap_time", "sector_time", "delta"),
    "driver": ("brake_pressure", "steering_angle", "pedal_travel"),
}

PLAN_RATES = {
    "chassis": 200,
    "imu": 200,
    "powertrain": 100,
    "aero": 100,
    "wheel": 100,
    "wheel_thermal": 20,
    "thermal": 10,
    "session": None,
    "driver": 100,
}

_PER_CORNER_SIGNALS = {
    "wheel_speed",
    "slip_ratio",
    "slip_angle",
    "vertical_load",
    "camber",
    "suspension_travel",
    "tyre_pressure",
    "tyre_temp",
    "brake_temp",
}

# The only two provenance values a channel may carry. `provenance` is not part of the
# loader's `Channel` dataclass - it is a contract claim about where a *number* came from,
# not a sampling property - so these two tests read the raw YAML rather than the contract.
PROVENANCE_VALUES = frozenset({"illustrative", "fia_limit"})

# Channels whose bounds are a regulation operating limit rather than a simulator setting,
# mapped to the clause each one is read out of. Issue 20 is the current issue; a bump in
# issue number means re-reading the clause text, not a different number here.
FIA_LIMIT_CLAUSES = {
    "gear": ("C9.6.1", "C9.7"),
    "mgu_k_power_kw": ("C5.2.7",),
}

# The regulation issue these clause numbers are quoted from, as it must appear in the
# `source` line of every fia_limit channel.
FIA_SOURCE_CITATION = ("FIA 2026", "Section C", "Issue 20", "2026-08-05")


def _minimal_channel(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "name": "probe_value",
        "group": "probe",
        "unit": "m/s^2",
        "rate_hz": 100,
        "dtype": "float32",
        "range": [-1.0, 1.0],
        "noise_model": "gaussian",
        "sigma": 0.01,
        "quantise": {"bits": 12, "full_scale": 1.0},
        "fault_eligible": [f for f in FAULT_TYPES if f != "swap"],
    }
    base.update(overrides)
    return base


def _write_contract(tmp_path: Path, channels: list[dict[str, object]]) -> Path:
    path = tmp_path / "channels.yaml"
    path.write_text(
        yaml.safe_dump({"version": "test", "channels": channels}, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _raw_channel_entries(repo: Path) -> list[dict[str, object]]:
    """The channel list as written, before the loader expands per-corner signals.

    Provenance is a declaration about where a bound came from, so it is asserted on the
    file rather than on the corner-expanded contract.
    """
    root = yaml.safe_load((repo / "channels.yaml").read_text(encoding="utf-8"))
    return list(root["channels"])


def test_every_plan_section_5_2_group_is_present(contract: ChannelContract) -> None:
    assert contract.groups == PLAN_GROUPS
    for group, expected in PLAN_CHANNELS.items():
        bases = {c.base_name for c in contract.group_members(group)}
        missing = set(expected) - bases
        assert not missing, (
            f"group {group} is missing PLAN.md section 5.2 channels {sorted(missing)}"
        )


def test_group_rates_match_plan_section_5_2(contract: ChannelContract) -> None:
    for group, rate in PLAN_RATES.items():
        for channel in contract.group_members(group):
            assert channel.rate_hz == rate, f"{channel.name} rate {channel.rate_hz} != {rate}"


def test_event_channels_encode_rate_as_null(contract: ChannelContract) -> None:
    session = contract.group_members("session")
    assert session, "the session group must exist"
    for channel in session:
        assert channel.event is True
        assert channel.rate_hz is None
        assert channel.period_s is None
    for channel in contract.channels:
        if not channel.event:
            assert channel.rate_hz is not None, f"{channel.name} has no rate"


def test_corner_expansion_is_a_function_not_a_hand_written_list(contract: ChannelContract) -> None:
    corner_bases = {c.base_name for c in contract.channels if c.base_name in _PER_CORNER_SIGNALS}
    assert corner_bases == _PER_CORNER_SIGNALS
    for base in sorted(_PER_CORNER_SIGNALS):
        members = [c for c in contract.channels if c.base_name == base]
        assert len(members) == len(CORNERS), f"{base} expanded to {len(members)} channels"
        assert [c.corner for c in members] == list(CORNERS)
        for corner in CORNERS:
            assert f"{base}_{corner.lower()}" in CHANNELS
            assert CHANNELS[f"{base}_{corner.lower()}"].group == CHANNELS[f"{base}_fl"].group


def test_swap_eligibility_requires_corners(contract: ChannelContract) -> None:
    for channel in contract.channels:
        if "swap" in channel.fault_eligible:
            assert channel.corners, f"{channel.name} is swap-eligible with no corners"
        if not channel.corners:
            assert "swap" not in channel.fault_eligible


def test_every_channel_declares_the_full_field_metadata(contract: ChannelContract) -> None:
    for channel in contract.channels:
        assert channel.unit, f"{channel.name} has no unit"
        assert channel.dtype in DTYPES
        assert channel.range_min < channel.range_max
        assert channel.noise_model in ("gaussian", "uniform", "none")
        assert channel.sigma >= 0.0
        assert channel.fault_eligible, f"{channel.name} is not eligible for any fault"
        assert set(channel.fault_eligible) <= set(FAULT_TYPES)
        assert channel.description, f"{channel.name} has no description"
        assert channel.source, f"{channel.name} has no provenance line"
        if channel.quantisation is not None:
            assert channel.quantisation.step > 0.0
            assert channel.quantisation.full_scale >= abs(channel.range_max)


def test_every_channel_declares_where_its_numbers_came_from(repo: Path) -> None:
    """No channel may claim `provenance: plan` any more.

    A plan provenance line said "PLAN.md section 5.2 lists this channel", which is true of
    the inventory and silent about the bounds - and the bounds are the part a reader is
    most likely to mistake for measured data. So every channel now answers one of two
    questions instead: `illustrative` (a simulator setting we chose) or `fia_limit` (a
    regulation operating limit). Nothing in between, and nothing unlabelled.
    """
    entries = _raw_channel_entries(repo)
    unlabelled = sorted(
        f"{entry['name']}={entry.get('provenance')!r}"
        for entry in entries
        if entry.get("provenance") not in PROVENANCE_VALUES
    )
    assert not unlabelled, f"channels without an illustrative/fia_limit provenance: {unlabelled}"

    fia_limit = {entry["name"] for entry in entries if entry.get("provenance") == "fia_limit"}
    assert fia_limit == set(FIA_LIMIT_CLAUSES), (
        f"fia_limit must be exactly {sorted(FIA_LIMIT_CLAUSES)}, found {sorted(fia_limit)}"
    )


def test_fia_limit_bounds_are_quoted_from_the_current_issue(repo: Path) -> None:
    """The two rule-derived bounds must say which clause, and which issue, they came from.

    A bound with no clause behind it is indistinguishable from a guess, and a bound quoted
    from a superseded issue silently goes stale. So `gear` and `mgu_k_power_kw` are pinned
    to the clause numbers and to Issue 20 of FIA 2026 Section C, and their declared ranges
    are pinned to those clauses rather than to whatever the generator happens to emit.
    """
    entries = {entry["name"]: entry for entry in _raw_channel_entries(repo)}

    for name, clauses in FIA_LIMIT_CLAUSES.items():
        entry = entries[name]
        assert entry["provenance"] == "fia_limit", f"{name} is not fia_limit"
        source = str(entry["source"])
        missing = [c for c in (*FIA_SOURCE_CITATION, *clauses) if c not in source]
        assert not missing, f"{name} source is missing {missing}: {source!r}"
        assert entry["source_date"], f"{name} has no source_date"

    assert entries["gear"]["range"] == [-1, 8]
    assert entries["mgu_k_power_kw"]["range"] == [-350.0, 350.0]
    assert entries["mgu_k_power_kw"]["quantise"]["full_scale"] == 350.0


def test_sizing_is_computed_from_the_contract_not_estimated(contract: ChannelContract) -> None:
    expected = sum(0.0 if c.rate_hz is None else c.rate_hz for c in contract.channels)
    assert expected == SAMPLES_PER_SECOND
    assert expected > 4600, "PLAN.md section 5.2 estimated ~4600/s; the contract must exceed it"
    raw_bytes_per_hour = expected * 4 * 3600
    assert 90e6 < raw_bytes_per_hour < 120e6


def test_generated_registry_matches_the_contract(contract: ChannelContract) -> None:
    runtime = load_contract()
    assert runtime.channel_names() == contract.names
    for channel in contract.channels:
        generated = CHANNELS[channel.name]
        assert generated.unit == channel.unit
        assert generated.dtype == channel.dtype
        assert generated.rate_hz == channel.rate_hz
        assert generated.range_min == channel.range_min
        assert generated.sigma == channel.sigma
        assert generated.fault_eligible == channel.fault_eligible
        assert generated.corner == channel.corner


def test_generated_dataclasses_cover_every_channel(contract: ChannelContract) -> None:
    from f1telemetry.generated import channels as generated

    for group in contract.groups:
        members = contract.group_members(group)
        cls = getattr(generated, "".join(p.capitalize() for p in group.split("_")) + "Channels")
        assert cls.field_names() == tuple(c.name for c in members)
        values = {c.name: float(i) for i, c in enumerate(members)}
        frame = cls.from_mapping(values)
        assert frame.as_mapping() == values


def test_parquet_schema_partitions_every_channel(contract: ChannelContract) -> None:
    assert tuple(PARQUET_SCHEMAS) == contract.groups
    seen: list[str] = []
    for group, schema in PARQUET_SCHEMAS.items():
        names = schema.names
        assert names == tuple(c.name for c in contract.group_members(group))
        seen.extend(names)
        arrow = schema.arrow_schema()
        assert arrow.names[0] == "t_s"
        assert arrow.field("t_s").nullable is False
        assert len(arrow) == len(names) + 1
    assert sorted(seen) == sorted(contract.names)


def test_typescript_output_declares_every_channel(contract: ChannelContract, repo: Path) -> None:
    text = (repo / "web" / "src" / "generated" / "channels.ts").read_text(encoding="utf-8")
    assert text.startswith("/**")
    for channel in contract.channels:
        assert f"name: '{channel.name}'" in text, f"{channel.name} missing from channels.ts"
    assert text.count("baseName: '") == len(contract)
    assert "export type ChannelName =" in text
    assert "export const CHANNELS_BY_GROUP" in text


def test_generated_surface_is_the_p0_contract_and_is_current(repo: Path) -> None:
    """P0-T4 and P0-T5 are the registry, the Parquet schema and the TypeScript types.

    The set is asserted as well as the freshness, so an artifact that belongs to a later
    phase cannot be added to codegen without a test noticing.
    """
    paths = {artifact.path.relative_to(repo).as_posix() for artifact in build_artifacts(repo)}
    assert paths == P0_GENERATED_PATHS
    assert check(repo) == [], "committed generated code does not match channels.yaml"


def test_codegen_is_idempotent_and_the_check_gate_fires(repo: Path) -> None:
    assert check(repo) == [], "committed generated code does not match channels.yaml"
    before = {path: path.read_bytes() for path in _generated_paths(repo)}
    assert main([]) == 0
    assert main([]) == 0
    after = {path: path.read_bytes() for path in before}
    assert after == before, "running codegen twice changed the output"
    assert check(repo) == []


def test_check_gate_reports_a_stale_artifact(tmp_path: Path) -> None:
    """The CI gate must actually fail, not quietly pass on a missing or edited file."""
    root = tmp_path / "repo"
    (root / "src" / "f1telemetry" / "generated").mkdir(parents=True)
    (root / "web" / "src" / "generated").mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "channels.yaml"
    (root / "channels.yaml").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    for artifact in build_artifacts(root):
        artifact.path.parent.mkdir(parents=True, exist_ok=True)
        artifact.path.write_text(artifact.content, encoding="utf-8", newline="\n")
    assert check(root) == []

    target = root / "src" / "f1telemetry" / "generated" / "channels.py"
    edited = target.read_text(encoding="utf-8") + "\nHAND_EDITED = True\n"
    target.write_text(edited, encoding="utf-8")
    stale = check(root)
    assert len(stale) == 1
    assert "stale" in stale[0]

    target.unlink()
    assert check(root) == [f"missing: {target}"]


def test_car_spec_provenance_is_complete(spec: CarSpec) -> None:
    findings = provenance_audit(spec.raw)
    assert findings == [], f"car_spec.yaml provenance gaps: {findings}"
    assert spec.provenance == "provisional"
    assert spec.source_date
    assert spec.spec["calibration_status"] == "uncalibrated"
    assert spec.spec["issue"] == 20
    assert str(spec.spec["issue_date"]) == "2026-08-05"


def test_provenance_audit_catches_a_nested_section_without_a_source_date(
    tmp_path: Path, repo: Path
) -> None:
    """`powertrain.ice` carries its own provenance line, so it owes its own source_date.

    The audit used to read only the top-level sections, so a nested block could claim
    `provenance: synthesised` with nothing dating it and pass.
    """
    root = yaml.safe_load((repo / "car_spec.yaml").read_text(encoding="utf-8"))
    del root["powertrain"]["ice"]["source_date"]
    path = tmp_path / "car_spec.yaml"
    path.write_text(yaml.safe_dump(root, sort_keys=False), encoding="utf-8")
    findings = provenance_audit(load_car_spec(path).raw)
    assert findings == ["powertrain.ice: provenance 'synthesised' requires a source_date"]


def test_gearbox_lands_in_the_plan_top_speed_band(spec: CarSpec) -> None:
    """The gearbox is provisional, but it must not contradict PLAN.md section 11."""
    top_speed_kmh = spec.speed_at(len(spec.gear_ratios), spec.rev_limit_rpm) * 3.6
    assert 350.0 <= top_speed_kmh <= 370.0, top_speed_kmh
    first_gear_kmh = spec.speed_at(1, 10000.0) * 3.6
    assert 100.0 < first_gear_kmh < 140.0, first_gear_kmh
    assert spec.ice_peak_power_kw == 400.0
    assert spec.mgu_k_peak_power_kw == 350.0
    assert spec.power_split_ice == 0.53


def test_loader_rejects_a_cd_coefficient_in_the_cl_curve(tmp_path: Path, repo: Path) -> None:
    """A drag coefficient under cl_curve is a typo, not a Cl value.

    The loader used to accept either key on either curve, so a misplaced coefficient
    entered the aero model as the wrong quantity and nothing failed.
    """
    root = yaml.safe_load((repo / "car_spec.yaml").read_text(encoding="utf-8"))
    entry = root["aero"]["cl_curve"][0]
    entry["cd"] = entry.pop("cl")
    path = tmp_path / "car_spec.yaml"
    path.write_text(yaml.safe_dump(root, sort_keys=False), encoding="utf-8")
    with pytest.raises(ContractError, match=r"aero\.cl_curve\[0\].*expected key 'cl'"):
        load_car_spec(path)


def test_loader_rejects_an_incomplete_channel(tmp_path: Path) -> None:
    path = _write_contract(tmp_path, [{"name": "no_unit", "group": "g", "rate_hz": 10}])
    with pytest.raises(ContractError, match="missing key"):
        load_channel_contract(path)


def test_loader_rejects_a_quantiser_that_does_not_cover_the_range(tmp_path: Path) -> None:
    path = _write_contract(tmp_path, [_minimal_channel(quantise={"bits": 12, "full_scale": 0.5})])
    with pytest.raises(ContractError, match="does not cover"):
        load_channel_contract(path)


def test_loader_rejects_a_rate_on_an_event_channel(tmp_path: Path) -> None:
    path = _write_contract(tmp_path, [_minimal_channel(event=True)])
    with pytest.raises(ContractError, match="event channels must set rate_hz: null"):
        load_channel_contract(path)


def test_loader_rejects_duplicate_channel_names(tmp_path: Path) -> None:
    path = _write_contract(
        tmp_path,
        [_minimal_channel(), _minimal_channel(group="other")],
    )
    with pytest.raises(ContractError, match="duplicate channel name"):
        load_channel_contract(path)


def test_loader_rejects_an_unknown_fault_type(tmp_path: Path) -> None:
    path = _write_contract(tmp_path, [_minimal_channel(fault_eligible=["dropout", "vibes"])])
    with pytest.raises(ContractError, match="unknown fault type"):
        load_channel_contract(path)


def test_loader_rejects_an_unknown_corner(tmp_path: Path) -> None:
    path = _write_contract(tmp_path, [_minimal_channel(corners=["FL", "MIDDLE"])])
    with pytest.raises(ContractError, match="unknown corner"):
        load_channel_contract(path)


def test_loader_rejects_noise_model_none_with_a_sigma(tmp_path: Path) -> None:
    path = _write_contract(tmp_path, [_minimal_channel(noise_model="none", sigma=0.5)])
    with pytest.raises(ContractError, match="requires sigma: 0"):
        load_channel_contract(path)


def _generated_paths(repo: Path) -> tuple[Path, ...]:
    return tuple(repo / relative for relative in sorted(P0_GENERATED_PATHS))
