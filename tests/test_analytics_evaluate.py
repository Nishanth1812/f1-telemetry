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
    BASELINE_L2_PERSISTENCE_IFOREST,
    IF_N_ESTIMATORS,
    IF_RANDOM_STATE,
    ONSET_INDEX,
    SEEDS,
    SEVERITIES,
    TARGET_CHANNEL,
    DetectionCell,
    baseline_names,
    calibration_frames,
    evaluate_all,
    fit_evaluation_model,
    fit_isolation_gate,
    has_sklearn,
    isolation_samples,
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
def test_three_baselines_reported_and_gap_section_documented(contract: ChannelContract) -> None:
    report = evaluate_all(contract)
    assert report.baselines == baseline_names()
    text = render_markdown(report)
    assert "## Isolation Forest gap" in text
    if has_sklearn():
        assert report.baselines == (
            BASELINE_L2,
            BASELINE_L2_PERSISTENCE,
            BASELINE_L2_PERSISTENCE_IFOREST,
        )
        assert f"`n_estimators={IF_N_ESTIMATORS}`" in text
        assert f"`random_state={IF_RANDOM_STATE}`" in text
    else:
        assert report.baselines == (BASELINE_L2, BASELINE_L2_PERSISTENCE)
        assert "is not importable" in text
    assert "Negative control" in text


@pytest.mark.analytics
def test_isolation_forest_is_deterministic_under_fixed_random_state(
    contract: ChannelContract,
) -> None:
    model = fit_evaluation_model(contract)
    calibration = calibration_frames(contract)
    frames = make_faulted_frames(contract, "step", 1.0, SEEDS[0])
    first = fit_isolation_gate(calibration, model)
    second = fit_isolation_gate(calibration, model)
    assert first.threshold == second.threshold
    assert isolation_samples(frames, first) == isolation_samples(frames, second)


@pytest.mark.analytics
def test_isolation_forest_fits_only_the_clean_calibration_window(
    contract: ChannelContract,
) -> None:
    model = fit_evaluation_model(contract)
    calibration = calibration_frames(contract)
    gate = fit_isolation_gate(calibration, model)
    # Six calibration frames: max_samples resolves to all of them, as documented.
    assert gate.estimator.max_samples_ == len(calibration) == 6
    scores = [sample.score for sample in isolation_samples(calibration, gate)]
    assert len(scores) == len(calibration)
    # The threshold is a lower-tail quantile of the same six scores, so it lies
    # inside their range.
    assert min(scores) <= gate.threshold <= max(scores)


@pytest.mark.analytics
def test_isolation_forest_alarms_on_one_of_its_own_calibration_frames(
    contract: ChannelContract,
) -> None:
    # Measured, not tuned away. A quantile of six points interpolates just above
    # the calibration minimum, so the most anomalous clean frame falls below the
    # threshold. The classical baselines are not asserted here; this pins only the
    # IF gate's own calibration behaviour.
    model = fit_evaluation_model(contract)
    calibration = calibration_frames(contract)
    gate = fit_isolation_gate(calibration, model)
    alarms = [sample.alarm for sample in isolation_samples(calibration, gate)]
    assert alarms.count(True) == 1


@pytest.mark.analytics
def test_isolation_forest_misses_step_that_classical_baselines_catch(
    contract: ChannelContract,
) -> None:
    report = evaluate_all(contract)

    def find(baseline: str) -> DetectionCell:
        return next(
            found
            for found in report.cells
            if found.fault_type == "step" and found.severity == 1.0 and found.baseline == baseline
        )

    assert find(BASELINE_L2).miss_rate == pytest.approx(0.0)
    assert find(BASELINE_L2_PERSISTENCE).miss_rate == pytest.approx(0.0)
    assert find(BASELINE_L2_PERSISTENCE_IFOREST).miss_rate == pytest.approx(1.0)


@pytest.mark.analytics
def test_isolation_forest_summary_is_worse_than_classical_on_the_sweep(
    contract: ChannelContract,
) -> None:
    report = evaluate_all(contract)
    summaries = {summary.baseline: summary for summary in report.summaries}
    isolation = summaries[BASELINE_L2_PERSISTENCE_IFOREST]
    assert isolation.miss_rate == pytest.approx(1.0)
    assert isolation.miss_rate > summaries[BASELINE_L2].miss_rate
    assert isolation.miss_rate > summaries[BASELINE_L2_PERSISTENCE].miss_rate
    # Zero clean false alarms is not evidence of quality for this column: it misses
    # every sweep fault too, so it is silent on clean records for the same reason.
    assert isolation.clean_false_alarms == 0


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
