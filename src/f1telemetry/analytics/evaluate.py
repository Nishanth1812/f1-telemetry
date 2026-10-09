"""P7 evaluation harness: clean + faulted records through Layers 0-3.

PLAN.md section 9.1, tasks P7-T9/T10/T12. Deterministic: no physics, no
kernels. The one learned baseline is scikit-learn's Isolation Forest with a
fixed ``random_state``. Every record is built from the channel
contract (``channels.yaml``) plus the seeded fault primitives in
:mod:`f1telemetry.telemetry.sensors`, and every threshold comes from the
contract or from a clean calibration window. Injected faults are never in
the training (calibration) set.

Sweep: every fault type x every severity x 3 seeds, one faulted channel
(``tyre_temp_fl``). The multivariate pair is the two front tyre
temperatures; the front-right base carries a 1 C static offset so that a
``swap`` fault exchanges two distinguishable values instead of two
near-identical ones. Metrics per case, per baseline:

* detection latency in frames from onset (``None`` when missed),
* miss / detection on the faulted record,
* localisation accuracy (attributed channel == faulted channel, where the
  attribution is the Layer 1 ranking at the detection frame with a Layer 0
  fallback; ``swap`` accepts either corner),
* false-alarm rate on clean records, in alarms per hour.

Baselines, reported side by side: ``l2`` (single-window T2/Q exceedance),
``l2+persistence`` (k-of-n confirmation) and ``l2+persistence+isolationforest``
(an Isolation Forest fitted on the clean calibration window's standardised
residuals, alarmed on its own calibration-quantile threshold and then passed
through the same k-of-n rule). The Isolation Forest is included when
``sklearn`` is importable; it is a project dependency (see ``pyproject.toml``),
and it is imported lazily so the classical baselines never touch it.

``docs/detection.md`` is generated from this harness, never hand-edited::

    uv run python -m f1telemetry.analytics.evaluate --regenerate docs/detection.md

``tests/test_analytics_evaluate.py`` diffs the committed file against fresh
harness output, so CI regenerability is a test, not a convention.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

from f1telemetry.analytics.multivariate import (
    DEFAULT_LEVEL,
    MultivariateModel,
    MultivariateSample,
    fit_multivariate,
    persistent_alarms,
    score_multivariate,
)
from f1telemetry.analytics.residuals import (
    ResidualSample,
    attribute,
    residual_z_scores,
)
from f1telemetry.analytics.validity import (
    ValidityFinding,
    check_validity,
)
from f1telemetry.contracts.channels import (
    ChannelContract,
    channels_yaml_path,
    load_channel_contract,
)
from f1telemetry.telemetry.sensors import (
    FAULT_TYPES_ORDERED,
    FaultParameters,
    apply_fault,
)
from f1telemetry.testing.records import SensorFrame

if TYPE_CHECKING:
    from sklearn.ensemble import IsolationForest

__all__ = [
    "BASELINE_L2",
    "BASELINE_L2_PERSISTENCE",
    "BASELINE_L2_PERSISTENCE_IFOREST",
    "DT_S",
    "EVALUATION_CHANNELS",
    "IF_N_ESTIMATORS",
    "IF_RANDOM_STATE",
    "NEGATIVE_CONTROL_SEED",
    "N_FRAMES",
    "ONSET_INDEX",
    "PERSISTENCE_K",
    "PERSISTENCE_N",
    "SEEDS",
    "SEVERITIES",
    "SWAP_PARTNER",
    "TARGET_CHANNEL",
    "BaselineSummary",
    "DetectionCell",
    "EvaluationReport",
    "IsolationGate",
    "IsolationSample",
    "NegativeControlResult",
    "baseline_names",
    "calibration_frames",
    "evaluate_all",
    "fit_evaluation_model",
    "fit_isolation_gate",
    "has_sklearn",
    "isolation_samples",
    "make_clean_frames",
    "make_faulted_frames",
    "make_negative_control_frames",
    "render_markdown",
    "run_layers",
]

BASELINE_L2: Final[str] = "l2"
BASELINE_L2_PERSISTENCE: Final[str] = "l2+persistence"
BASELINE_L2_PERSISTENCE_IFOREST: Final[str] = "l2+persistence+isolationforest"

EVALUATION_CHANNELS: Final[tuple[str, ...]] = ("tyre_temp_fl", "tyre_temp_fr")
TARGET_CHANNEL: Final[str] = "tyre_temp_fl"
SWAP_PARTNER: Final[str] = "tyre_temp_fr"

BASE_FL_C: Final[float] = 90.0
BASE_FR_C: Final[float] = 91.0
DT_S: Final[float] = 0.05
N_FRAMES: Final[int] = 60
ONSET_INDEX: Final[int] = 20
SEVERITIES: Final[tuple[float, ...]] = (0.5, 1.0)
SEEDS: Final[tuple[int, ...]] = (11, 22, 33)
NEGATIVE_CONTROL_SEED: Final[int] = 99
PERSISTENCE_K: Final[int] = 2
PERSISTENCE_N: Final[int] = 3
IF_N_ESTIMATORS: Final[int] = 100
IF_RANDOM_STATE: Final[int] = 0

_CALIBRATION_ZPAIRS: Final[tuple[tuple[float, float], ...]] = (
    (1.0, 1.0),
    (-1.0, -1.0),
    (2.0, 2.0),
    (-2.0, -2.0),
    (0.1, -0.1),
    (-0.1, 0.1),
)

_NEGATIVE_CONTROL_RAMP_C: Final[float] = 0.5


@dataclass(frozen=True, slots=True)
class IsolationGate:
    """Isolation Forest fitted on the clean calibration window's standardised vectors.

    ``model`` supplies the Layer 2 standardisation so the feature vector is the
    same ``(y - mu) / sigma`` the manifold sees. An alarm is a score strictly
    below ``threshold``; lower scores are more anomalous.
    """

    model: MultivariateModel
    estimator: IsolationForest
    threshold: float


@dataclass(frozen=True, slots=True)
class IsolationSample:
    """One frame's Isolation Forest score and its k-of-n persisted alarm.

    ``attributed`` is the channel whose reset to its calibration mean raises the
    estimator's normality score the most, or ``None`` when no reset helps.
    """

    index: int
    score: float
    alarm: bool
    persisted: bool
    attributed: str | None


@dataclass(frozen=True, slots=True)
class LayerOutputs:
    """Everything Layers 0-3 compute for one record."""

    findings: tuple[ValidityFinding, ...]
    residuals: tuple[ResidualSample, ...]
    scores: tuple[MultivariateSample, ...]
    exceeds: tuple[bool, ...]
    persisted: tuple[bool, ...]
    isolation: tuple[IsolationSample, ...] | None


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    """One (fault, severity, seed, baseline) evaluation."""

    fault_type: str
    severity: float
    seed: int
    baseline: str
    detected: bool
    latency_frames: int | None
    attributed_channel: str | None
    correctly_localised: bool


@dataclass(frozen=True, slots=True)
class DetectionCell:
    """Seed-aggregated row of the detection matrix."""

    fault_type: str
    severity: float
    baseline: str
    n_seeds: int
    n_detected: int
    miss_rate: float
    mean_latency_frames: float | None
    localisation_accuracy: float | None


@dataclass(frozen=True, slots=True)
class BaselineSummary:
    """Whole-sweep summary plus the clean-record false-alarm rate."""

    baseline: str
    miss_rate: float
    mean_latency_frames: float | None
    localisation_accuracy: float | None
    clean_false_alarms: int
    clean_frames: int
    false_alarms_per_hour: float


@dataclass(frozen=True, slots=True)
class NegativeControlResult:
    """Operational-variability record with no injected fault."""

    baseline: str
    alarms: int
    frames: int
    silent: bool


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Full harness output; the sole input to the markdown renderer."""

    channels: tuple[str, ...]
    target_channel: str
    fault_types: tuple[str, ...]
    severities: tuple[float, ...]
    seeds: tuple[int, ...]
    n_frames: int
    onset_index: int
    dt_s: float
    persistence_k: int
    persistence_n: int
    baselines: tuple[str, ...]
    sklearn_available: bool
    cells: tuple[DetectionCell, ...]
    summaries: tuple[BaselineSummary, ...]
    negative_control: tuple[NegativeControlResult, ...]


def has_sklearn() -> bool:
    """Whether the Isolation Forest baseline can run (no import, just a probe)."""
    return importlib.util.find_spec("sklearn") is not None


def baseline_names() -> tuple[str, ...]:
    """Active baselines: two without sklearn, three with it."""
    if has_sklearn():
        return (BASELINE_L2, BASELINE_L2_PERSISTENCE, BASELINE_L2_PERSISTENCE_IFOREST)
    return (BASELINE_L2, BASELINE_L2_PERSISTENCE)


def calibration_frames(contract: ChannelContract) -> tuple[SensorFrame, ...]:
    """Clean six-frame calibration window; means are the per-corner bases."""
    sigma_fl = contract.by_name("tyre_temp_fl").sigma
    sigma_fr = contract.by_name("tyre_temp_fr").sigma
    return tuple(
        SensorFrame(
            t_s=index * DT_S,
            values={
                "tyre_temp_fl": BASE_FL_C + zfl * sigma_fl,
                "tyre_temp_fr": BASE_FR_C + zfr * sigma_fr,
            },
        )
        for index, (zfl, zfr) in enumerate(_CALIBRATION_ZPAIRS)
    )


def fit_evaluation_model(contract: ChannelContract) -> MultivariateModel:
    """Fit Layer 2 on the clean calibration window only."""
    return fit_multivariate(calibration_frames(contract), EVALUATION_CHANNELS, contract)


def _standardised(frame: SensorFrame, model: MultivariateModel) -> list[float] | None:
    """Layer 2 z-vector for one frame, or ``None`` when a model channel is unusable."""
    row: list[float] = []
    for name, mean, sigma in zip(model.channels, model.means, model.sigmas, strict=True):
        value = frame.values.get(name)
        if value is None or not math.isfinite(value):
            return None
        row.append((value - mean) / sigma)
    return row


def fit_isolation_gate(
    frames: Sequence[SensorFrame],
    model: MultivariateModel,
    *,
    level: float = DEFAULT_LEVEL,
) -> IsolationGate:
    """Fit the Isolation Forest on clean calibration ``frames`` only.

    The threshold is the ``1 - level`` quantile of the calibration scores, the
    lower-tail mirror of Layer 2's upper-quantile threshold. With six frames
    ``max_samples`` resolves to six, so every tree sees the whole window: a
    deliberately small fit, documented in the report, not a tuned one.
    """
    # Imported here so the classical baselines never touch sklearn.
    from sklearn.ensemble import IsolationForest

    rows: list[list[float]] = []
    for index, frame in enumerate(frames):
        row = _standardised(frame, model)
        if row is None:
            msg = f"calibration frame {index} has no usable value for the isolation features"
            raise ValueError(msg)
        rows.append(row)
    matrix = np.array(rows, dtype=float)
    estimator = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        max_samples="auto",
        random_state=IF_RANDOM_STATE,
    )
    estimator.fit(matrix)
    calibration_scores = estimator.score_samples(matrix)
    return IsolationGate(
        model=model,
        estimator=estimator,
        threshold=float(np.quantile(calibration_scores, 1.0 - level)),
    )


def isolation_samples(
    frames: Sequence[SensorFrame],
    gate: IsolationGate,
    *,
    k: int = PERSISTENCE_K,
    n: int = PERSISTENCE_N,
) -> tuple[IsolationSample, ...]:
    """Score ``frames`` with the gate; frames with unusable values are skipped.

    Attribution is leave-one-channel-out on the estimator score: each channel is
    reset to its calibration mean in turn and the channel whose reset gives the
    largest normality gain is named. Persistence is applied across the scored
    frames in order, the same way Layer 3 treats the Layer 2 flags.
    """
    indexed: list[tuple[int, list[float]]] = []
    for index, frame in enumerate(frames):
        row = _standardised(frame, gate.model)
        if row is not None:
            indexed.append((index, row))
    if not indexed:
        return ()
    matrix = np.array([row for _, row in indexed], dtype=float)
    scores = gate.estimator.score_samples(matrix)
    gains = np.zeros(matrix.shape, dtype=float)
    for column in range(matrix.shape[1]):
        reset = matrix.copy()
        reset[:, column] = 0.0
        gains[:, column] = gate.estimator.score_samples(reset) - scores
    alarms = tuple(float(score) < gate.threshold for score in scores)
    persisted = persistent_alarms(alarms, k=k, n=n)
    samples: list[IsolationSample] = []
    for position, (index, _row) in enumerate(indexed):
        best = int(np.argmax(gains[position]))
        samples.append(
            IsolationSample(
                index=index,
                score=float(scores[position]),
                alarm=alarms[position],
                persisted=persisted[position],
                attributed=gate.model.channels[best] if gains[position, best] > 0.0 else None,
            )
        )
    return tuple(samples)


def make_clean_frames(
    contract: ChannelContract, seed: int, *, n_frames: int = N_FRAMES
) -> tuple[SensorFrame, ...]:
    """Clean record: common-mode drift plus tiny independent jitter.

    The independent component stays small so the clean record scores silent
    against the calibration thresholds; the common mode is what a real soak
    looks like on two front tyres.
    """
    sigma_fl = contract.by_name("tyre_temp_fl").sigma
    sigma_fr = contract.by_name("tyre_temp_fr").sigma
    rng = np.random.default_rng(seed)
    frames: list[SensorFrame] = []
    for index in range(n_frames):
        common = float(rng.uniform(-0.4, 0.4))
        jitter_fl = float(rng.uniform(-0.04, 0.04))
        jitter_fr = float(rng.uniform(-0.04, 0.04))
        frames.append(
            SensorFrame(
                t_s=index * DT_S,
                values={
                    "tyre_temp_fl": BASE_FL_C + (common + jitter_fl) * sigma_fl,
                    "tyre_temp_fr": BASE_FR_C + (common + jitter_fr) * sigma_fr,
                },
            )
        )
    return tuple(frames)


def make_negative_control_frames(
    contract: ChannelContract, seed: int = NEGATIVE_CONTROL_SEED
) -> tuple[SensorFrame, ...]:
    """Operational variability with no fault: a slow 0.5 C warm-up ramp."""
    sigma_fl = contract.by_name("tyre_temp_fl").sigma
    sigma_fr = contract.by_name("tyre_temp_fr").sigma
    rng = np.random.default_rng(seed)
    frames: list[SensorFrame] = []
    for index in range(N_FRAMES):
        drift = _NEGATIVE_CONTROL_RAMP_C * index / (N_FRAMES - 1)
        common = float(rng.uniform(-0.4, 0.4))
        jitter_fl = float(rng.uniform(-0.04, 0.04))
        jitter_fr = float(rng.uniform(-0.04, 0.04))
        frames.append(
            SensorFrame(
                t_s=index * DT_S,
                values={
                    "tyre_temp_fl": BASE_FL_C + drift + (common + jitter_fl) * sigma_fl,
                    "tyre_temp_fr": BASE_FR_C + drift + (common + jitter_fr) * sigma_fr,
                },
            )
        )
    return tuple(frames)


def _fault_params(fault_type: str, severity: float) -> FaultParameters | None:
    """Severity-aware parameters; ``swap`` is handled structurally, not here."""
    if fault_type == "swap":
        return None
    if fault_type == "spike":
        return FaultParameters(
            severity=severity,
            onset_index=ONSET_INDEX,
            spike_sigma_factor=6.0 * severity,
        )
    if fault_type == "freeze":
        return FaultParameters(
            severity=severity,
            onset_index=ONSET_INDEX,
            freeze_hold_index=ONSET_INDEX - 1,
        )
    if fault_type == "stale":
        return FaultParameters(
            severity=severity,
            onset_index=ONSET_INDEX,
            stale_samples=max(1, round(4.0 * severity)),
        )
    return FaultParameters(severity=severity, onset_index=ONSET_INDEX)


def make_faulted_frames(
    contract: ChannelContract, fault_type: str, severity: float, seed: int
) -> tuple[SensorFrame, ...]:
    """Clean record with one fault injected on the target channel from onset."""
    if fault_type not in FAULT_TYPES_ORDERED:
        msg = f"unknown fault_type {fault_type!r}"
        raise ValueError(msg)
    clean = make_clean_frames(contract, seed)
    if fault_type == "swap":
        return tuple(
            SensorFrame(
                t_s=frame.t_s,
                values=(
                    dict(frame.values)
                    if frame_index < ONSET_INDEX
                    else {
                        TARGET_CHANNEL: frame.values[SWAP_PARTNER],
                        SWAP_PARTNER: frame.values[TARGET_CHANNEL],
                    }
                ),
            )
            for frame_index, frame in enumerate(clean)
        )
    channel = contract.by_name(TARGET_CHANNEL)
    params = _fault_params(fault_type, severity)
    assert params is not None
    truth = np.array([frame.values[TARGET_CHANNEL] for frame in clean], dtype=float)
    out: list[SensorFrame] = []
    for frame_index, frame in enumerate(clean):
        values = dict(frame.values)
        if frame_index >= ONSET_INDEX:
            values[TARGET_CHANNEL] = apply_fault(
                float(values[TARGET_CHANNEL]),
                channel,
                fault_type,
                params,
                index=frame_index,
                ground_truth_array=truth,
                seed=seed,
            )
        out.append(SensorFrame(t_s=frame.t_s, values=values))
    return tuple(out)


def run_layers(
    frames: Sequence[SensorFrame],
    model: MultivariateModel,
    contract: ChannelContract,
    gate: IsolationGate | None = None,
) -> LayerOutputs:
    """Run Layers 0-3 over one record in frame order.

    ``gate`` is only needed for the Isolation Forest baseline; without it the
    ``isolation`` field is ``None`` and the classical baselines are unchanged.
    """
    findings = check_validity(frames, contract)
    residuals = residual_z_scores(frames, contract)
    scores = score_multivariate(frames, model)
    exceeds = tuple(sample.t2_exceeds or sample.q_exceeds for sample in scores)
    persisted = persistent_alarms(exceeds, k=PERSISTENCE_K, n=PERSISTENCE_N)
    isolation = None if gate is None else isolation_samples(frames, gate)
    return LayerOutputs(
        findings=findings,
        residuals=residuals,
        scores=scores,
        exceeds=exceeds,
        persisted=persisted,
        isolation=isolation,
    )


def _alarm_indices(outputs: LayerOutputs, baseline: str) -> tuple[int, ...]:
    """Frame indices alarming under one baseline, in frame order."""
    if baseline == BASELINE_L2:
        return tuple(
            sample.index
            for sample, flag in zip(outputs.scores, outputs.exceeds, strict=True)
            if flag
        )
    if baseline == BASELINE_L2_PERSISTENCE:
        return tuple(
            sample.index
            for sample, flag in zip(outputs.scores, outputs.persisted, strict=True)
            if flag
        )
    if baseline == BASELINE_L2_PERSISTENCE_IFOREST:
        if outputs.isolation is None:
            msg = "isolation-forest baseline needs an IsolationGate passed to run_layers"
            raise ValueError(msg)
        return tuple(sample.index for sample in outputs.isolation if sample.persisted)
    msg = f"unknown baseline {baseline!r}"
    raise ValueError(msg)


def _attribute_channel(outputs: LayerOutputs, index: int) -> str | None:
    """Attribution at one frame: Layer 1 ranking first, Layer 0 fallback."""
    ranked = attribute(outputs.residuals, index)
    if ranked:
        return ranked[0].channel
    at_index = sorted({finding.channel for finding in outputs.findings if finding.index == index})
    for channel in at_index:
        if channel != "t_s":
            return channel
    return None


def _isolation_attribution(outputs: LayerOutputs, index: int) -> str | None:
    """Estimator-score attribution at one frame, as scored by the Isolation Forest."""
    for sample in outputs.isolation or ():
        if sample.index == index:
            return sample.attributed
    return None


def _expected_channels(fault_type: str) -> tuple[str, ...]:
    if fault_type == "swap":
        return (TARGET_CHANNEL, SWAP_PARTNER)
    return (TARGET_CHANNEL,)


def _evaluate_case(
    outputs: LayerOutputs,
    *,
    fault_type: str,
    severity: float,
    seed: int,
    baseline: str,
) -> CaseOutcome:
    alarms = tuple(index for index in _alarm_indices(outputs, baseline) if index >= ONSET_INDEX)
    if not alarms:
        return CaseOutcome(
            fault_type=fault_type,
            severity=severity,
            seed=seed,
            baseline=baseline,
            detected=False,
            latency_frames=None,
            attributed_channel=None,
            correctly_localised=False,
        )
    detection = alarms[0]
    if baseline == BASELINE_L2_PERSISTENCE_IFOREST:
        attributed = _isolation_attribution(outputs, detection)
    else:
        attributed = _attribute_channel(outputs, detection)
    expected = _expected_channels(fault_type)
    return CaseOutcome(
        fault_type=fault_type,
        severity=severity,
        seed=seed,
        baseline=baseline,
        detected=True,
        latency_frames=detection - ONSET_INDEX,
        attributed_channel=attributed,
        correctly_localised=attributed in expected,
    )


def _aggregate_cell(outcomes: Sequence[CaseOutcome]) -> DetectionCell:
    first = outcomes[0]
    detected = [outcome for outcome in outcomes if outcome.detected]
    latencies = [
        outcome.latency_frames for outcome in detected if outcome.latency_frames is not None
    ]
    localised = sum(1 for outcome in detected if outcome.correctly_localised)
    return DetectionCell(
        fault_type=first.fault_type,
        severity=first.severity,
        baseline=first.baseline,
        n_seeds=len(outcomes),
        n_detected=len(detected),
        miss_rate=(len(outcomes) - len(detected)) / len(outcomes),
        mean_latency_frames=(sum(latencies) / len(latencies) if latencies else None),
        localisation_accuracy=(localised / len(detected) if detected else None),
    )


def evaluate_all(contract: ChannelContract | None = None) -> EvaluationReport:
    """Run the full fault x severity x seed sweep and aggregate the report."""
    resolved = load_channel_contract(channels_yaml_path()) if contract is None else contract
    model = fit_evaluation_model(resolved)
    baselines = baseline_names()
    gate = (
        fit_isolation_gate(calibration_frames(resolved), model)
        if BASELINE_L2_PERSISTENCE_IFOREST in baselines
        else None
    )
    fault_types = tuple(FAULT_TYPES_ORDERED)

    outcomes: list[CaseOutcome] = []
    for fault_type in fault_types:
        for severity in SEVERITIES:
            for seed in SEEDS:
                frames = make_faulted_frames(resolved, fault_type, severity, seed)
                outputs = run_layers(frames, model, resolved, gate)
                for baseline in baselines:
                    outcomes.append(
                        _evaluate_case(
                            outputs,
                            fault_type=fault_type,
                            severity=severity,
                            seed=seed,
                            baseline=baseline,
                        )
                    )

    cells = tuple(
        _aggregate_cell(
            tuple(
                outcome
                for outcome in outcomes
                if outcome.fault_type == fault_type
                and outcome.severity == severity
                and outcome.baseline == baseline
            )
        )
        for fault_type in fault_types
        for severity in SEVERITIES
        for baseline in baselines
    )

    summaries: list[BaselineSummary] = []
    for baseline in baselines:
        scoped = [outcome for outcome in outcomes if outcome.baseline == baseline]
        detected = [outcome for outcome in scoped if outcome.detected]
        latencies = [
            outcome.latency_frames for outcome in detected if outcome.latency_frames is not None
        ]
        localised = sum(1 for outcome in detected if outcome.correctly_localised)
        clean_alarms = 0
        clean_frames = 0
        for seed in SEEDS:
            clean = make_clean_frames(resolved, seed)
            clean_outputs = run_layers(clean, model, resolved, gate)
            clean_alarms += len(_alarm_indices(clean_outputs, baseline))
            clean_frames += len(clean)
        total_hours = clean_frames * DT_S / 3600.0
        summaries.append(
            BaselineSummary(
                baseline=baseline,
                miss_rate=(len(scoped) - len(detected)) / len(scoped),
                mean_latency_frames=(sum(latencies) / len(latencies) if latencies else None),
                localisation_accuracy=(localised / len(detected) if detected else None),
                clean_false_alarms=clean_alarms,
                clean_frames=clean_frames,
                false_alarms_per_hour=(clean_alarms / total_hours if total_hours > 0 else 0.0),
            )
        )

    control_frames = make_negative_control_frames(resolved)
    control_outputs = run_layers(control_frames, model, resolved, gate)
    control_results = tuple(
        NegativeControlResult(
            baseline=baseline,
            alarms=len(_alarm_indices(control_outputs, baseline)),
            frames=len(control_frames),
            silent=len(_alarm_indices(control_outputs, baseline)) == 0,
        )
        for baseline in baselines
    )
    return EvaluationReport(
        channels=EVALUATION_CHANNELS,
        target_channel=TARGET_CHANNEL,
        fault_types=fault_types,
        severities=SEVERITIES,
        seeds=SEEDS,
        n_frames=N_FRAMES,
        onset_index=ONSET_INDEX,
        dt_s=DT_S,
        persistence_k=PERSISTENCE_K,
        persistence_n=PERSISTENCE_N,
        baselines=baselines,
        sklearn_available=has_sklearn(),
        cells=cells,
        summaries=tuple(summaries),
        negative_control=control_results,
    )


def _format_optional(value: float | None, decimals: int) -> str:
    if value is None:
        return "--"
    return f"{value:.{decimals}f}"


def _cell_lookup(
    report: EvaluationReport, fault: str, severity: float, baseline: str
) -> DetectionCell:
    for cell in report.cells:
        if cell.fault_type == fault and cell.severity == severity and cell.baseline == baseline:
            return cell
    msg = f"no cell for {(fault, severity, baseline)}"
    raise KeyError(msg)


def render_markdown(report: EvaluationReport) -> str:
    """Render the committed detection table from harness output (deterministic)."""
    lines = [
        "# Detection table (generated)",
        "",
        "Generated from `src/f1telemetry/analytics/evaluate.py`; do not hand-edit. "
        "Regenerate with:",
        "",
        "```text",
        "uv run python -m f1telemetry.analytics.evaluate --regenerate docs/detection.md",
        "```",
        "",
        "## Method",
        "",
        f"Channels: `{', '.join(report.channels)}` (multivariate pair); "
        f"faulted channel: `{report.target_channel}`. "
        f"Records hold {report.n_frames} frames at {report.dt_s} s "
        f"with fault onset at frame {report.onset_index}. "
        f"Sweep: {len(report.fault_types)} fault types x "
        f"{len(report.severities)} severities "
        f"({', '.join(f'{severity:.1f}' for severity in report.severities)}) x "
        f"{len(report.seeds)} seeds ({', '.join(str(seed) for seed in report.seeds)}). "
        f"Calibration uses a clean six-frame window only; injected faults are "
        f"never in the training set. Persistence rule: "
        f"{report.persistence_k}-of-{report.persistence_n}. "
        "Localisation is the Layer 1 ranking at the detection frame with a "
        "Layer 0 fallback; `swap` accepts either corner.",
        "",
        "## Detection matrix (per fault x severity, seeds aggregated)",
        "",
        "Latency is mean frames from onset over detected seeds; `--` means no "
        "seed in the group was detected (miss) or localised.",
        "",
    ]
    metric_headers = [
        header
        for baseline in report.baselines
        for header in (f"{baseline} miss", f"{baseline} latency", f"{baseline} loc")
    ]
    header = "| " + " | ".join(["fault", "severity", "seeds", *metric_headers]) + " |"
    divider_cells = ["---", "---", "---:", *["---:", "---:", "---:"] * len(report.baselines)]
    divider = "|" + "|".join(divider_cells) + "|"
    lines.extend([header, divider])
    for fault in report.fault_types:
        for severity in report.severities:
            row = [fault, f"{severity:.1f}"]
            first = _cell_lookup(report, fault, severity, report.baselines[0])
            row.append(str(first.n_seeds))
            for baseline in report.baselines:
                cell = _cell_lookup(report, fault, severity, baseline)
                row.extend(
                    [
                        f"{cell.miss_rate:.2f}",
                        _format_optional(cell.mean_latency_frames, 1),
                        _format_optional(cell.localisation_accuracy, 2),
                    ]
                )
            lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "## Baseline summaries",
            "",
            "| baseline | miss rate | mean latency (frames) | "
            "localisation accuracy | clean false alarms | false alarms / hour |",
            "|---|---|---|---|---|---|",
        ]
    )
    for summary in report.summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    summary.baseline,
                    f"{summary.miss_rate:.3f}",
                    _format_optional(summary.mean_latency_frames, 2),
                    _format_optional(summary.localisation_accuracy, 3),
                    f"{summary.clean_false_alarms}/{summary.clean_frames}",
                    f"{summary.false_alarms_per_hour:.1f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Negative control",
            "",
            "Operational-variability record (slow 0.5 C warm-up ramp, seeded "
            "jitter, no injected fault, no genuine abnormality). "
            "The detector must stay silent.",
            "",
            "| baseline | alarms | frames | silent |",
            "|---|---|---|:---:|",
        ]
    )
    for result in report.negative_control:
        lines.append(
            f"| {result.baseline} | {result.alarms} | {result.frames} "
            f"| {'yes' if result.silent else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "## Isolation Forest gap",
            "",
        ]
    )
    if report.sklearn_available:
        lines.extend(
            [
                f"The `{BASELINE_L2_PERSISTENCE_IFOREST}` column is scikit-learn's "
                f"IsolationForest (`n_estimators={IF_N_ESTIMATORS}`, "
                f"`random_state={IF_RANDOM_STATE}`), fit on the six clean calibration "
                "frames only. Its feature vector is the Layer 2 standardised pair, "
                "so the forest sees the same `(y - mu) / sigma` the manifold does. "
                "The alarm threshold is the calibration window's own "
                f"{1.0 - DEFAULT_LEVEL:.3f} score quantile, the lower-tail mirror "
                "of Layer 2's upper-quantile threshold, and the same k-of-n "
                "persistence rule as the other columns applies.",
                "",
                "Caveats. `max_samples` resolves to the calibration size, so each "
                "tree sees every calibration frame: a six-point fit is a thin "
                "basis, not a learned density. Attribution is the estimator-score "
                "gain from resetting one channel to its calibration mean. The "
                "hyperparameters were fixed before the sweep and not tuned on it. "
                "The sweep is simulated, so this is not a held-out real-data result.",
            ]
        )
    else:
        lines.extend(
            [
                "scikit-learn is not importable in this environment, so the "
                f"`{BASELINE_L2_PERSISTENCE_IFOREST}` column is omitted and only "
                f"`{BASELINE_L2}` and `{BASELINE_L2_PERSISTENCE}` are reported. "
                "scikit-learn is a declared project dependency, so a synced "
                "environment should always produce the full table.",
            ]
        )
    lines.extend(
        [
            "",
            "## Reading notes",
            "",
            "* Layer 2 scores every frame with data; frames with a missing or "
            "non-finite value are skipped by design (Layer 0 owns those "
            "values), so a sustained `dropout` is a Layer 2 miss and a "
            "Layer 0 catch.",
            "* Single-frame `spike` trips the single-window baseline but "
            "cannot survive a k-of-n rule with k >= 2; that miss under "
            "persistence is the documented cost of false-alarm suppression.",
            "* `freeze` and `stale` hold plausible values, so a marginal "
            "manifold often stays silent on them; the stuck-value rule in "
            "Layer 0 is the layer that owns `freeze`.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: print the table, or rewrite the committed file."""
    parser = argparse.ArgumentParser(description="P7 detection-table evaluation harness.")
    parser.add_argument(
        "--regenerate",
        type=Path,
        default=None,
        help="write the rendered detection table to PATH",
    )
    parser.add_argument(
        "--check",
        type=Path,
        default=None,
        help="fail when PATH differs from fresh harness output",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = evaluate_all()
    text = render_markdown(report)
    if args.regenerate is not None:
        args.regenerate.write_text(text, encoding="utf-8")
    if args.check is not None:
        committed = args.check.read_text(encoding="utf-8")
        if committed != text:
            parser.exit(1, f"{args.check} differs from harness output; regenerate it.\n")
    if args.regenerate is None:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
