"""Run-manifest tests (P6-T7). Owns this file, the manifest wiring and run_manifest.py only.

The first block is the manifest on its own: what it validates, how it serialises, and how it
comes back out of a Parquet file. The second block is the wiring built on top of it - the
manifest attached to a run, and the metadata merge that carries a manifest into a file
without displacing the contract keys every file is supposed to keep.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from f1telemetry.testing import scenarios
from f1telemetry.testing.fixtures import straight_line_record
from f1telemetry.testing.parquet_io import (
    METADATA,
    build_table,
    metadata_with_manifest,
    read_frames,
    serialise_frames,
)
from f1telemetry.testing.run_manifest import (
    MANIFEST_METADATA_KEY,
    RunManifest,
    from_metadata,
)

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import CarSpec, KernelConfig

GIT_SHA = "3f1c0a9e2b7d5468af0c1d3e5b7a9f2046813c5d"
SETUP_HASH = "ab" * 32

# Control intervals in the attribution probe. One interval is the least a run can be, so the
# wiring is proved without spending a scenario's wall-clock on a test that asserts no physics.
_PROBE_CONTROL_STEPS: Final[int] = 1


def _manifest(**overrides: object) -> RunManifest:
    fields: dict[str, object] = {
        "seed": 7,
        "car_spec_version": "fia-2026-c-issue-20",
        "scenario_version": "launch-2026-10-06",
        "setup_hash": SETUP_HASH,
        "git_sha": GIT_SHA,
    }
    fields.update(overrides)
    return RunManifest(**fields)  # pyright: ignore[reportArgumentType]


def _text_metadata(metadata: Mapping[bytes, bytes] | None) -> dict[str, str]:
    """Arrow's byte key-value metadata as the text it was written from."""
    return {key.decode("utf-8"): value.decode("utf-8") for key, value in (metadata or {}).items()}


def _probe_config(spec: CarSpec) -> KernelConfig:
    """The validated configuration every scenario runs on; the probe reads nothing else."""
    return spec.kernel_config()


def _probe_scenario(config: KernelConfig) -> scenarios.Scenario:
    """The shortest run the runner accepts: one control interval, no pedal and no request.

    The duration is derived from the configured step rather than chosen, so the runner's
    whole-control-interval rule holds whatever ``dt_s`` the spec carries, and a declared start
    speed of zero is a starting condition rather than a performance figure. Nothing here says
    anything about the car.
    """
    return scenarios.Scenario(
        name="manifest_attribution_probe",
        initial_speed_m_s=0.0,
        description="attribution probe: coasting from rest with no pedal or gear request",
        segments=(scenarios.ScenarioSegment(_PROBE_CONTROL_STEPS * config.dt_s),),
    )


def test_fields_carried_and_frozen() -> None:
    manifest = _manifest()
    assert manifest.seed == 7
    assert manifest.car_spec_version == "fia-2026-c-issue-20"
    assert manifest.scenario_version == "launch-2026-10-06"
    assert manifest.setup_hash == SETUP_HASH
    assert manifest.git_sha == GIT_SHA
    with pytest.raises(AttributeError):
        manifest.seed = 8  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize("seed", [-1, 1.5, True, "7", None])
def test_refuses_a_seed_no_generator_accepts(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        _manifest(seed=seed)


@pytest.mark.parametrize("value", ["", " leading", "trailing ", None, 3])
def test_refuses_an_unusable_car_spec_version(value: object) -> None:
    with pytest.raises(ValueError, match="car_spec_version"):
        _manifest(car_spec_version=value)


@pytest.mark.parametrize("value", ["", " leading", "trailing ", None, 3])
def test_refuses_an_unusable_scenario_version(value: object) -> None:
    with pytest.raises(ValueError, match="scenario_version"):
        _manifest(scenario_version=value)


@pytest.mark.parametrize("value", ["", "NOTHEX", "ABCDEF", f" {SETUP_HASH}", None])
def test_refuses_a_non_lower_case_hex_setup_hash(value: object) -> None:
    with pytest.raises(ValueError, match="setup_hash"):
        _manifest(setup_hash=value)


@pytest.mark.parametrize("value", ["", "abc123", GIT_SHA.upper(), "g" * 40, "a" * 41, None])
def test_refuses_a_git_sha_that_is_not_a_commit_id(value: object) -> None:
    with pytest.raises(ValueError, match="git_sha"):
        _manifest(git_sha=value)


def test_canonical_is_compact_key_sorted_json() -> None:
    manifest = _manifest()
    canonical = manifest.canonical()
    assert ", " not in canonical
    assert '": ' not in canonical
    # Key order is sorted, not insertion order: to_dict() lists seed first.
    assert list(json.loads(canonical)) == sorted(manifest.to_dict())
    assert canonical.index('"car_spec_version"') < canonical.index('"git_sha"')
    assert json.loads(canonical) == manifest.to_dict()


def test_canonical_does_not_depend_on_field_insertion_order() -> None:
    """Same five values, different construction order, identical bytes."""
    forward = _manifest()
    reversed_ = RunManifest(
        seed=7,
        car_spec_version="fia-2026-c-issue-20",
        scenario_version="launch-2026-10-06",
        setup_hash=SETUP_HASH,
        git_sha=GIT_SHA,
    )
    assert forward.canonical() == reversed_.canonical()


def test_json_round_trip() -> None:
    manifest = _manifest()
    assert RunManifest.from_json(manifest.canonical()) == manifest


def test_from_dict_rejects_unknown_missing_and_bad_version() -> None:
    document = _manifest().to_dict()
    unknown = {**document, "git_sah": document["git_sha"]}
    del unknown["git_sha"]
    with pytest.raises(ValueError, match="unknown field"):
        RunManifest.from_dict(unknown)
    with pytest.raises(ValueError, match="field names"):
        RunManifest.from_dict({**document, 1: "invalid"})
    with pytest.raises(ValueError, match="missing field"):
        RunManifest.from_dict({k: v for k, v in document.items() if k != "seed"})
    with pytest.raises(ValueError, match="manifest_version"):
        RunManifest.from_dict({**document, "manifest_version": 99})
    with pytest.raises(ValueError, match="manifest_version"):
        RunManifest.from_dict({**document, "manifest_version": True})
    with pytest.raises(ValueError, match="seed"):
        RunManifest.from_dict({**document, "seed": -1})


def test_from_dict_refuses_a_non_mapping_document() -> None:
    for document in ([], ["seed"], 7, "manifest", None):
        with pytest.raises(ValueError, match="document"):
            RunManifest.from_dict(document)


def test_from_json_refuses_deep_nesting() -> None:
    """A too-deep payload is unreadable, not an escape from the ValueError contract."""
    with pytest.raises(ValueError, match="nested too deeply"):
        RunManifest.from_json("[" * 5000 + "]" * 5000)


def test_metadata_is_one_namespaced_key_and_round_trips() -> None:
    manifest = _manifest()
    metadata = manifest.to_metadata()
    assert list(metadata) == [MANIFEST_METADATA_KEY]
    assert from_metadata(metadata) == manifest


def test_absent_key_reads_as_no_manifest() -> None:
    assert from_metadata(None) is None
    assert from_metadata({}) is None
    assert from_metadata({b"other": b"value"}) is None


def test_present_but_unreadable_manifest_is_an_error() -> None:
    with pytest.raises(ValueError, match="document"):
        from_metadata({MANIFEST_METADATA_KEY.encode("utf-8"): b"{not json"})
    with pytest.raises(ValueError, match="document"):
        from_metadata({MANIFEST_METADATA_KEY: '{"manifest_version": 99}'})
    with pytest.raises(ValueError, match="UTF-8"):
        from_metadata({MANIFEST_METADATA_KEY.encode("utf-8"): b"\xff\xfe"})


def test_parquet_metadata_is_read_as_bytes_by_arrow() -> None:
    """Arrow hands back byte keys and byte values; that path is the real one."""
    manifest = _manifest()
    table = pa.table({"t": [0.0]}, schema=pa.schema([("t", pa.float64())], metadata=None))
    table = table.replace_schema_metadata(
        {k.encode("utf-8"): v.encode("utf-8") for k, v in manifest.to_metadata().items()}
    )
    assert from_metadata(table.schema.metadata) == manifest


def test_survives_a_real_parquet_round_trip() -> None:
    manifest = _manifest()
    payload = serialise_frames(straight_line_record(), manifest.to_metadata())
    table = pq.read_table(pa.BufferReader(payload))
    assert from_metadata(table.schema.metadata) == manifest


# ------------------------------------------------------------------
# Wiring: a manifest on a run, and a merge that keeps the default keys
# ------------------------------------------------------------------


def test_wiring_keeps_the_default_keys_while_the_manifest_round_trips() -> None:
    """The manifest reaches the file by addition: the contract keys are still there."""
    manifest = _manifest()
    record = straight_line_record()
    payload = serialise_frames(record, metadata_with_manifest(manifest))
    table = pq.read_table(pa.BufferReader(payload))
    assert from_metadata(table.schema.metadata) == manifest
    written = _text_metadata(table.schema.metadata)
    # Defaults kept, not replaced: a file that cites a run still names its contract.
    assert {name: written[name] for name in METADATA} == METADATA
    assert written[MANIFEST_METADATA_KEY] == manifest.canonical()
    # The citation is metadata only - the frames come back exactly as they went in, and the
    # manifest carries no wall-clock value, so repeating the serialisation is byte-identical.
    assert read_frames(payload) == record.frames
    assert serialise_frames(record, metadata_with_manifest(manifest)) == payload


def test_caller_metadata_merges_over_the_defaults_instead_of_replacing_them() -> None:
    """Adding a key is not a request to drop ``contract`` and ``note``."""
    table = build_table(straight_line_record(), {"origin": "p6-t7"})
    written = _text_metadata(table.schema.metadata)
    assert {name: written[name] for name in METADATA} == METADATA
    assert written["origin"] == "p6-t7"


def test_run_scenario_attaches_the_caller_manifest_and_claims_nothing_else(
    spec: CarSpec,
) -> None:
    """``run_scenario`` carries the manifest it is given, and a run without one cites nothing."""
    config = _probe_config(spec)
    plan = _probe_scenario(config)
    manifest = _manifest()
    run = scenarios.run_scenario(
        config, plan, control_steps=_PROBE_CONTROL_STEPS, manifest=manifest
    )
    bare = scenarios.run_scenario(config, plan, control_steps=_PROBE_CONTROL_STEPS)
    assert run.manifest == manifest
    assert bare.manifest is None
    # Attributing a run changes what it says about itself, not what it computed: the records are
    # identical, and the difference between the two serialisations is the metadata key alone.
    assert run.record == bare.record
    payload = serialise_frames(run.record, metadata_with_manifest(run.manifest))
    table = pq.read_table(pa.BufferReader(payload))
    assert from_metadata(table.schema.metadata) == manifest
    written = _text_metadata(table.schema.metadata)
    assert {name: written[name] for name in METADATA} == METADATA
    assert serialise_frames(bare.record) != payload
