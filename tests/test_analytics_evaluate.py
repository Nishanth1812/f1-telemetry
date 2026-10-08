"""P7-T9/T10/T12 evaluation harness tests (PLAN.md section 9.1).

The detection table in ``docs/detection.md`` is generated from
:mod:`f1telemetry.analytics.evaluate`, never hand-edited. The last test
here diffs the committed file against fresh harness output, which is what
makes the table CI-regenerable.
"""

from __future__ import annotations

import pytest

from f1telemetry.analytics.evaluate import (
    BASELINE_L2,
    BASELINE_L2_PERSISTENCE,
    ONSET_INDEX,
    SEEDS,
    SEVERITIES,
    TARGET_CHANNEL,
    baseline_names,
    evaluate_all,
    fit_evaluation_model,
    has_sklearn,
    make_faulted_frames,
    render_markdown,
)
from f1telemetry.analytics.multivariate import score_multivariate
from f1telemetry.analytics.residuals import attribute, residual_z_scores
from f1telemetry.analytics.validity import check_validity
from f1telemetry.contracts.channels import ChannelContract, channels_yaml_path
from f1telemetry.telemetry.sensors import FAULT_TYPES_ORDERED


@pytest.mark.analytics
def test_sweep_covers_every_fault_type_severity_and_seed(contract: ChannelContract) -> None:
    report = evaluate_all(contract)
    assert report.fault_types == tuple(FAULT_TYPES_ORDERED)
    assert len(report.fault_types) == 10
    assert report.severities == SEVERITIES
    assert report.seeds == SEEDS
    assert len(report.cells) == 10 * len(SEVERITIES) * len(report.baselines)
    for cell in report.cells:
        assert cell.n_seeds == len(SEEDS)
        assert cell.n_detected <= cell.n_seeds
        assert 0.0 <= cell.miss_rate <= 1.0


@pytest.mark.analytics
def test_strong_step_fault_detected_at_onset_and_localised(contract: ChannelContract) -> None:
    report = evaluate_all(contract)
    cell = next(
        cell
        for cell in report.cells
        if cell.fault_type == "step" and cell.severity == 1.0 and cell.baseline == BASELINE_L2
    )
    assert cell.miss_rate == pytest.approx(0.0)
    assert cell.mean_latency_frames == pytest.approx(0.0)
    assert cell.localisation_accuracy == pytest.approx(1.0)


@pytest.mark.analytics
def test_spike_survives_single_window_but_not_persistence(contract: ChannelContract) -> None:
    report = evaluate_all(contract)
    single = next(
        cell
        for cell in report.cells
        if cell.fault_type == "spike" and cell.severity == 1.0 and cell.baseline == BASELINE_L2
    )
    persisted = next(
        cell
        for cell in report.cells
        if cell.fault_type == "spike"
        and cell.severity == 1.0
        and cell.baseline == BASELINE_L2_PERSISTENCE
    )
    assert single.miss_rate == pytest.approx(0.0)
    assert persisted.miss_rate == pytest.approx(1.0)


@pytest.mark.analytics
def test_dropout_is_a_layer2_miss_and_a_layer0_catch(contract: ChannelContract) -> None:
    frames = make_faulted_frames(contract, "dropout", 1.0, SEEDS[0])
    findings = check_validity(frames, contract)
    post_onset = [
        (finding.channel, finding.rule) for finding in findings if finding.index >= ONSET_INDEX
    ]
    assert post_onset[:1] == [(TARGET_CHANNEL, "range")]
    model = fit_evaluation_model(contract)
    scored = score_multivariate(frames, model)
    assert all(sample.index < ONSET_INDEX for sample in scored if sample.t2_exceeds)


@pytest.mark.analytics
def test_clean_records_and_negative_control_stay_silent(contract: ChannelContract) -> None:
    report = evaluate_all(contract)
    for summary in report.summaries:
        assert summary.clean_false_alarms == 0
        assert summary.false_alarms_per_hour == pytest.approx(0.0)
    for result in report.negative_control:
        assert result.silent


@pytest.mark.analytics
def test_harness_output_is_deterministic(contract: ChannelContract) -> None:
    assert render_markdown(evaluate_all(contract)) == render_markdown(evaluate_all(contract))


@pytest.mark.analytics
def test_attribution_names_the_faulted_channel_on_step(contract: ChannelContract) -> None:
    frames = make_faulted_frames(contract, "step", 1.0, SEEDS[0])
    samples = residual_z_scores(frames, contract)
    assert attribute(samples, ONSET_INDEX)[0].channel == TARGET_CHANNEL


@pytest.mark.analytics
def test_two_baselines_without_sklearn_gap_documented(contract: ChannelContract) -> None:
    report = evaluate_all(contract)
    assert report.baselines == baseline_names()
    text = render_markdown(report)
    if not has_sklearn():
        assert report.baselines == (BASELINE_L2, BASELINE_L2_PERSISTENCE)
        assert "not a project dependency" in text
    assert "Negative control" in text


@pytest.mark.analytics
def test_detection_md_matches_harness_output(contract: ChannelContract) -> None:
    path = channels_yaml_path().parent / "docs" / "detection.md"
    assert path.is_file(), "docs/detection.md is missing; regenerate it from the harness"
    fresh = render_markdown(evaluate_all(contract))
    committed = path.read_text(encoding="utf-8")
    assert committed == fresh, (
        "docs/detection.md differs from harness output; regenerate with "
        "`uv run python -m f1telemetry.analytics.evaluate --regenerate docs/detection.md`"
    )
