"""Durable recorded Scenario Lab runs (Task 1 of the portfolio release).

Owns this file, ``f1telemetry.runs`` and ``f1telemetry.testing.scenario_lab``'s provenance
capture. Each test names one invariant: exact round-trip across a restart, provenance,
identifier and corruption handling, no-overwrite, and interrupted publication.
"""

from __future__ import annotations

import hashlib
import os
import re
import struct
import subprocess
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from f1telemetry import runs
from f1telemetry.contracts.car_spec import car_spec_path, load_car_spec
from f1telemetry.contracts.channels import channels_yaml_path
from f1telemetry.runs import RunStore, RunStoreError
from f1telemetry.synthetic import Frame, FrameEvent
from f1telemetry.testing import scenario_lab
from f1telemetry.testing.parquet_io import serialise_frames
from f1telemetry.testing.records import SampleRecord, SensorFrame
from f1telemetry.testing.scenario_lab import LabRun

_RUN_FILE = re.compile(r"[0-9a-f]{32}\.parquet")
_VALUES = (1 / 3, -0.0, 5e-324, -1.7976931348623157e308, 12.5)


def _result() -> dict[str, object]:
    return {
        "type": "scenario_result",
        "request_id": "req-1",
        "scenario_id": "launch",
        "title": "Standing launch",
        "description": "test run",
        "setting_name": "Throttle",
        "setting": 50.0,
        "duration_s": 0.02,
        "peak_speed_kmh": 12.5,
        "peak_longitudinal_g": 0.25,
        "peak_lateral_g": 0.0,
        "simulation_label": "Uncalibrated simulation",
        "points": [{"time_s": 0.0, "speed_kmh": 0.0}, {"time_s": 0.01, "speed_kmh": 12.5}],
    }


def _provenance() -> dict[str, object]:
    return {
        "scenario": {"id": "launch", "version": "1", "seed": None},
        "request": {"scenario_id": "launch", "setting": 50.0},
        "car_spec": {"sha256": "ab" * 32, "text": "spec: {}\n"},
        "channels": {"sha256": "cd" * 32, "text": "channels: []\n"},
        "code": {"revision": None, "source": "unknown", "dirty": None},
    }


def _lab_run(provenance: dict[str, object] | None = None) -> LabRun:
    frames = tuple(
        Frame(
            time_us=10_000 * index,
            channels={"speed": value, "accel_longitudinal": -value},
        )
        for index, value in enumerate(_VALUES)
    )
    return LabRun(_result(), frames, provenance)


def _bits(value: float) -> bytes:
    return struct.pack("<d", value)


def test_round_trip_survives_restart(tmp_path: Path) -> None:
    original = _lab_run(_provenance())
    saved = RunStore(tmp_path).save(original)

    run_id = saved["run_id"]
    assert isinstance(run_id, str) and re.fullmatch(r"[0-9a-f]{32}", run_id)
    assert "run_id" not in original.result, "save must not mutate the caller's result"

    restarted = RunStore(tmp_path)
    assert restarted.list_runs() == [saved]
    loaded = restarted.load(run_id)
    assert loaded.result == saved
    assert loaded.provenance == _provenance()
    assert [frame.time_us for frame in loaded.frames] == [
        frame.time_us for frame in original.frames
    ]
    for loaded_frame, original_frame in zip(loaded.frames, original.frames, strict=True):
        assert loaded_frame.channels.keys() == original_frame.channels.keys()
        for name, value in original_frame.channels.items():
            assert _bits(loaded_frame.channels[name]) == _bits(value)


def test_listing_an_absent_store_is_empty(tmp_path: Path) -> None:
    assert RunStore(tmp_path / "never-created").list_runs() == []


def test_run_without_provenance_round_trips_as_none(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    run_id = store.save(_lab_run())["run_id"]
    assert store.load(str(run_id)).provenance is None


def test_save_never_replaces_an_existing_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runs, "_new_run_id", lambda: "0" * 32)
    store = RunStore(tmp_path)
    store.save(_lab_run(_provenance()))
    before = (tmp_path / ("0" * 32 + ".parquet")).read_bytes()

    other = _lab_run(_provenance())
    other.result["title"] = "Replacement"
    with pytest.raises(RunStoreError):
        store.save(other)

    assert (tmp_path / ("0" * 32 + ".parquet")).read_bytes() == before
    assert store.load("0" * 32).result["title"] == "Standing launch"


@pytest.mark.parametrize(
    "run_id",
    [
        "",
        "../" + "a" * 32,
        "..\\" + "a" * 32,
        "a" * 31,
        "a" * 33,
        "A" * 32,
        "g" * 32,
        "a" * 32 + ".parquet",
        "a" * 32 + "\n",
        "/etc/passwd",
    ],
)
def test_invalid_run_identifiers_are_rejected(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(RunStoreError):
        RunStore(tmp_path).load(run_id)


def test_unknown_but_well_formed_identifier_is_an_explicit_error(tmp_path: Path) -> None:
    with pytest.raises(RunStoreError, match="not found"):
        RunStore(tmp_path).load("f" * 32)


def test_corrupt_file_fails_explicitly(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    run_id = str(store.save(_lab_run(_provenance()))["run_id"])
    path = tmp_path / f"{run_id}.parquet"
    path.write_bytes(path.read_bytes()[: len(path.read_bytes()) // 2])

    with pytest.raises(RunStoreError, match="corrupt"):
        store.load(run_id)
    with pytest.raises(RunStoreError, match="corrupt"):
        store.list_runs()


def test_plain_parquet_without_a_run_document_is_corrupt(tmp_path: Path) -> None:
    samples = (SensorFrame(0.0, {"speed": 1.0}), SensorFrame(0.01, {"speed": 2.0}))
    record = SampleRecord("plain", 0.01, "no run document", (), samples)
    (tmp_path / ("1" * 32 + ".parquet")).write_bytes(serialise_frames(record))

    with pytest.raises(RunStoreError, match="corrupt"):
        RunStore(tmp_path).load("1" * 32)


def test_renamed_run_file_is_rejected(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    run_id = str(store.save(_lab_run(_provenance()))["run_id"])
    (tmp_path / f"{run_id}.parquet").rename(tmp_path / ("2" * 32 + ".parquet"))

    with pytest.raises(RunStoreError, match="corrupt"):
        store.load("2" * 32)


def test_interrupted_fsync_leaves_no_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_fd: int) -> None:
        raise OSError("simulated power loss")

    monkeypatch.setattr(os, "fsync", fail)
    store = RunStore(tmp_path)
    with pytest.raises(RunStoreError):
        store.save(_lab_run(_provenance()))

    assert list(tmp_path.iterdir()) == []
    assert store.list_runs() == []


def test_interrupted_publication_leaves_no_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(_source: object, _target: object) -> None:
        raise OSError("simulated crash before publication")

    monkeypatch.setattr(os, "link", fail)
    store = RunStore(tmp_path)
    with pytest.raises(RunStoreError):
        store.save(_lab_run(_provenance()))

    assert list(tmp_path.iterdir()) == []
    assert store.list_runs() == []


def test_leftover_temporary_file_is_not_listed(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.save(_lab_run(_provenance()))
    (tmp_path / ".leftover.tmp").write_bytes(b"partial")

    listed = store.list_runs()
    assert len(listed) == 1
    assert _RUN_FILE.fullmatch(f"{listed[0]['run_id']}.parquet")


def test_non_finite_samples_are_refused_before_publication(tmp_path: Path) -> None:
    run = LabRun(_result(), (Frame(time_us=0, channels={"speed": float("nan")}),), None)

    with pytest.raises(RunStoreError):
        RunStore(tmp_path).save(run)
    assert list(tmp_path.iterdir()) == []


def test_saved_file_is_plain_parquet_with_the_run_document(tmp_path: Path) -> None:
    saved = RunStore(tmp_path).save(_lab_run(_provenance()))
    path = tmp_path / f"{saved['run_id']}.parquet"
    metadata = pq.read_metadata(path).metadata
    assert metadata is not None
    assert b"f1telemetry/scenario_run" in metadata
    assert pq.read_table(path).num_rows == len(_VALUES)


def test_run_lab_captures_configuration_and_code_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("F1_TELEMETRY_REVISION", "b" * 40)
    run = scenario_lab.run_lab(
        {"type": "scenario_run", "request_id": "prov-1", "scenario_id": "braking", "setting": 50}
    )

    assert run.provenance is not None
    spec_bytes = car_spec_path().read_bytes()
    channel_bytes = channels_yaml_path().read_bytes()
    car_spec = run.provenance["car_spec"]
    channels = run.provenance["channels"]
    assert isinstance(car_spec, dict) and isinstance(channels, dict)
    assert car_spec["sha256"] == hashlib.sha256(spec_bytes).hexdigest()
    assert car_spec["text"] == spec_bytes.decode("utf-8")
    assert channels["sha256"] == hashlib.sha256(channel_bytes).hexdigest()
    assert run.provenance["scenario"] == {
        "id": "braking",
        "version": scenario_lab.SCENARIO_LAB_VERSION,
        "seed": None,
    }
    assert run.provenance["code"] == {"revision": "b" * 40, "source": "build_env", "dirty": None}


def test_code_revision_is_explicitly_unknown_without_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("F1_TELEMETRY_REVISION", raising=False)

    def no_git(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    assert scenario_lab._code_provenance() == {  # pyright: ignore[reportPrivateUsage]
        "revision": None,
        "source": "unknown",
        "dirty": None,
    }


def test_code_revision_comes_from_git_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("F1_TELEMETRY_REVISION", raising=False)
    code = scenario_lab._code_provenance()  # pyright: ignore[reportPrivateUsage]
    assert code["source"] == "git"
    assert isinstance(code["revision"], str) and re.fullmatch(r"[0-9a-f]{40}", code["revision"])
    assert isinstance(code["dirty"], bool)


def test_configuration_snapshot_can_be_parsed_after_source_changes(tmp_path: Path) -> None:
    snapshot = car_spec_path().read_bytes()
    changed = tmp_path / "car_spec.yaml"
    changed.write_text("invalid: replaced after capture", encoding="utf-8")
    parsed = load_car_spec(changed, content=snapshot)
    assert parsed.mass_kg == load_car_spec().mass_kg


@pytest.mark.parametrize("times", [(), (1, 1), (2, 1), (-1, 1)])
def test_invalid_timestamps_cannot_publish_an_unreadable_run(
    tmp_path: Path, times: tuple[int, ...]
) -> None:
    run = LabRun({"type": "scenario_result"}, tuple(Frame(t, {"speed": 1.0}) for t in times))
    with pytest.raises(RunStoreError, match=r"frames|timestamp"):
        RunStore(tmp_path).save(run)
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []


def test_blocked_store_path_reports_a_storage_error(tmp_path: Path) -> None:
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("keep this file", encoding="utf-8")
    with pytest.raises(RunStoreError, match="not published"):
        RunStore(blocked).save(_lab_run())
    assert blocked.read_text(encoding="utf-8") == "keep this file"


def test_fault_events_survive_saved_run_replay(tmp_path: Path) -> None:
    event = FrameEvent("fault", 0, fault_type="spike", channel="speed", label=True)
    original = LabRun(_result(), (Frame(0, {"speed": 1.0}, (event,)),))
    store = RunStore(tmp_path)
    run_id = str(store.save(original)["run_id"])
    assert store.load(run_id).frames == original.frames
