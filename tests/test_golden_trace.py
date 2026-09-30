"""P0-T8: golden-trace comparison, committed baselines, and threshold reporting.

PLAN.md section 14 lists "baseline churns on every coefficient tune" as a known risk, and
the mitigation is explicit: flag trace diffs over a stated threshold for review rather than
auto-failing. So these tests do three separate things and it matters which is which:

* the committed baselines are compared, and a flagged diff is **reported** - written to
  ``reports/golden/`` and printed - rather than failed. Only ``golden_strict = true`` in
  the pytest config, or a deliberate ``--golden-update``, changes that.
* the harness is proved non-vacuous: a trace perturbed past a threshold must be flagged, a
  trace nudged inside the noise floor must not, and a channel vanishing from a trace counts
  as a diff rather than a pass.
* thresholds are proved to come from ``channels.yaml``, not from a second hand-kept list.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from f1telemetry.generated.channels import CHANNELS
from f1telemetry.testing.fixtures import cornering_record, straight_line_record
from f1telemetry.testing.golden import (
    GoldenReport,
    compare,
    contract_thresholds,
    load_baseline,
    threshold_for,
    write_baseline,
)
from f1telemetry.testing.records import SampleRecord, SensorFrame, with_frame_value

pytestmark = pytest.mark.golden

BASELINES = {
    "straight_line": "tests/golden/straight_line.json",
    "cornering": "tests/golden/cornering.json",
}
BASELINE_NOTES = {
    "straight_line": (
        "Supplied fixture values, not simulation output. The baseline exists to prove the "
        "comparison machinery works and to catch a change to the fixture or the harness; "
        "P1 replaces it with a simulation run."
    ),
    "cornering": (
        "Supplied fixture values, not simulation output. See straight_line.json for what a "
        "P0 baseline is and is not."
    ),
}


def _record(name: str) -> SampleRecord:
    return straight_line_record() if name == "straight_line" else cornering_record()


def _report_for(repo: Path, name: str) -> GoldenReport:
    record = _record(name)
    return compare(
        name,
        load_baseline(repo / BASELINES[name]),
        record,
        contract_thresholds(record.channels),
    )


def test_baselines_are_committed_and_describe_the_fixture_exactly(
    repo: Path, golden_update: bool
) -> None:
    for name, relative in BASELINES.items():
        path = repo / relative
        record = _record(name)
        if golden_update:
            write_baseline(path, record, BASELINE_NOTES[name])
        assert path.is_file(), (
            f"missing committed baseline {relative}; "
            f"regenerate deliberately with `uv run pytest --golden-update`"
        )
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["name"] == name
        assert document["steps"] == len(record.frames)
        assert document["dt_s"] == record.dt_s
        assert set(document["series"]) == set(record.channels)
        for channel, values in document["series"].items():
            assert values == list(record.series(channel)), f"{relative}:{channel} drifted"


def test_golden_diff_is_reported_for_review_not_failed(
    repo: Path, report_dir: Path, golden_strict: bool
) -> None:
    """The default policy: report. A flagged diff fails only under `golden_strict`."""
    flagged_total = 0
    for name in BASELINES:
        report = _report_for(repo, name)
        report.write(report_dir / f"{name}.json")
        print(report.render())
        flagged_total += len(report.flagged)
        if flagged_total and golden_strict:
            pytest.fail(
                f"{name}: {len(report.flagged)} channel(s) over threshold with golden_strict "
                f"on. Review {report_dir / f'{name}.json'}, then either accept the change by "
                f"re-running `uv run pytest --golden-update`, or treat it as a regression."
            )
    assert report_dir.is_dir()


def test_harness_flags_a_perturbation_past_the_threshold(repo: Path) -> None:
    """Non-vacuity: a trace that moves more than the contract noise floor is flagged."""
    record = _record("straight_line")
    baseline = load_baseline(repo / BASELINES["straight_line"])
    clean = compare("clean", baseline, record, contract_thresholds(record.channels))
    assert clean.clean, clean.render()

    sigma = CHANNELS["speed"].sigma
    moved = 10.0 * sigma
    perturbed = with_frame_value(record, 1, "speed", record.frames[1].values["speed"] + moved)
    report = compare("perturbed", baseline, perturbed, contract_thresholds(record.channels))
    assert not report.clean
    assert {diff.channel for diff in report.flagged} == {"speed"}
    worst = next(diff for diff in report.diffs if diff.channel == "speed")
    assert worst.at_step == 1
    assert worst.max_abs_diff == pytest.approx(moved, rel=1.0e-9)
    assert "exceeds" in worst.reason
    assert "over threshold" in report.render()


def test_harness_does_not_flag_a_change_inside_the_noise_floor(repo: Path) -> None:
    record = _record("cornering")
    baseline = load_baseline(repo / BASELINES["cornering"])
    sigma = CHANNELS["wheel_speed_fl"].sigma
    nudged = with_frame_value(
        record, 2, "wheel_speed_fl", record.frames[2].values["wheel_speed_fl"] + sigma
    )
    report = compare("nudged", baseline, nudged, contract_thresholds(record.channels))
    assert report.clean, report.render()


def test_thresholds_come_from_the_contract_noise_floor() -> None:
    assert threshold_for("speed").abs_tol == pytest.approx(6.0 * CHANNELS["speed"].sigma)
    assert threshold_for("vx").abs_tol == pytest.approx(6.0 * CHANNELS["vx"].sigma)
    thresholds = contract_thresholds(["speed", "vx", "ice_rpm"])
    assert [t.channel for t in thresholds] == ["speed", "vx", "ice_rpm"]
    assert all(t.abs_tol > 0.0 for t in thresholds)
    assert threshold_for("speed").abs_tol != threshold_for("ice_rpm").abs_tol
    assert threshold_for("gear").abs_tol == pytest.approx(
        max(1.0e-9, 6.0 * CHANNELS["gear"].sigma)
    ), "a discrete channel with sigma 0 still needs a non-zero band"


def test_a_channel_disappearing_from_the_trace_is_flagged(repo: Path) -> None:
    """A channel vanishing is a silent regression, so it is a diff and not a pass."""
    record = _record("straight_line")
    baseline = load_baseline(repo / BASELINES["straight_line"])
    report = compare("short", baseline, _truncated(record), contract_thresholds(record.channels))
    assert not report.clean
    assert "speed" in {diff.channel for diff in report.flagged}
    lost = [diff for diff in report.flagged if diff.channel != "speed"]
    assert lost and all(math.isinf(diff.max_abs_diff) for diff in report.flagged)


def test_write_baseline_round_trips(tmp_path: Path) -> None:
    record = straight_line_record()
    path = write_baseline(tmp_path / "out.json", record, "unit test")
    reloaded = load_baseline(path)
    assert reloaded == {channel: list(record.series(channel)) for channel in record.channels}
    assert json.loads(path.read_text(encoding="utf-8"))["dt_s"] == record.dt_s
    assert path.read_text(encoding="utf-8").endswith("\n")


def _truncated(record: SampleRecord) -> SampleRecord:
    """A record that publishes almost nothing, as a lost-channel bug would."""
    return SampleRecord(
        name=record.name,
        dt_s=record.dt_s,
        description=record.description,
        ground_truth=record.ground_truth,
        frames=tuple(
            SensorFrame(t_s=frame.t_s, values={"speed": frame.values["speed"]})
            for frame in record.frames[:1]
        ),
    )
