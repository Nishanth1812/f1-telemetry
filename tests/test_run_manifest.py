"""Run-manifest tests (P6-T7). Owns this file and run_manifest.py only."""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from f1telemetry.testing.fixtures import straight_line_record
from f1telemetry.testing.parquet_io import serialise_frames
from f1telemetry.testing.run_manifest import (
    MANIFEST_METADATA_KEY,
    RunManifest,
    from_metadata,
)

GIT_SHA = "3f1c0a9e2b7d5468af0c1d3e5b7a9f2046813c5d"
SETUP_HASH = "ab" * 32


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
