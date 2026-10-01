"""Golden-trace comparison (P0-T8, PLAN.md section 11 and section 14).

Coefficient tuning is *expected* to move traces, so a golden diff must not be an
automatic failure - PLAN.md section 14 names "baseline churns on every coefficient tune"
as a known risk and the mitigation is to flag diffs over a stated threshold for review.
That is what this module does: compare a candidate trace against a committed baseline,
report the per-channel diff, and hand the decision to a human.

Two design points that make it defensible rather than decorative:

* thresholds come from ``channels.yaml`` via the channel's own ``sigma``, not from a
  hand-kept list, so a retune of the noise model moves the gate with it and there is no
  second place where a per-channel number lives;
* ``GoldenReport`` is a *report*, not an assertion. The pytest stage in PLAN.md section 12
  runs ``pytest -m golden`` and the report is written to disk for review; only
  ``--golden-strict`` turns a flagged diff into a failure.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from f1telemetry.generated.channels import CHANNELS
from f1telemetry.testing.records import SampleRecord

__all__ = [
    "ChannelDiff",
    "GoldenReport",
    "Threshold",
    "compare",
    "contract_thresholds",
    "load_baseline",
    "threshold_for",
    "write_baseline",
]

SIGMA_MULTIPLE: Final[float] = 6.0
RELATIVE_FLOOR: Final[float] = 1.0e-6
ABSOLUTE_FLOOR: Final[float] = 1.0e-9


@dataclass(frozen=True, slots=True)
class Threshold:
    """Per-channel acceptance band, derived from the contract's declared noise floor."""

    channel: str
    abs_tol: float
    rel_tol: float

    def to_dict(self) -> dict[str, float | str]:
        return {"channel": self.channel, "abs_tol": self.abs_tol, "rel_tol": self.rel_tol}


@dataclass(frozen=True, slots=True)
class ChannelDiff:
    """How far one channel moved, and whether that clears the review threshold."""

    channel: str
    threshold: Threshold
    max_abs_diff: float
    max_rel_diff: float
    at_step: int
    baseline_value: float
    candidate_value: float

    @property
    def flagged(self) -> bool:
        return self.max_abs_diff > self.threshold.abs_tol

    @property
    def reason(self) -> str:
        if not self.flagged:
            return "within threshold"
        return (
            f"abs diff {self.max_abs_diff:.6g} exceeds {self.threshold.abs_tol:.6g} "
            f"({self.max_rel_diff:.3%} of {self.baseline_value:.6g})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "abs_tol": self.threshold.abs_tol,
            "rel_tol": self.threshold.rel_tol,
            "max_abs_diff": self.max_abs_diff,
            "max_rel_diff": self.max_rel_diff,
            "at_step": self.at_step,
            "baseline_value": self.baseline_value,
            "candidate_value": self.candidate_value,
            "flagged": self.flagged,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class GoldenReport:
    """The whole comparison, ready to be written to disk and read by a person."""

    name: str
    diffs: tuple[ChannelDiff, ...]

    @property
    def flagged(self) -> tuple[ChannelDiff, ...]:
        return tuple(diff for diff in self.diffs if diff.flagged)

    @property
    def clean(self) -> bool:
        return not self.flagged

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "channels": len(self.diffs),
            "flagged": len(self.flagged),
            "diffs": [diff.to_dict() for diff in self.diffs],
        }

    def render(self) -> str:
        if self.clean:
            return f"golden {self.name}: clean, {len(self.diffs)} channel(s) compared"
        head = (
            f"golden {self.name}: {len(self.flagged)} of {len(self.diffs)} "
            "channel(s) over threshold"
        )
        lines = [head]
        lines.extend(f"  {diff.channel}: {diff.reason}" for diff in self.flagged)
        return "\n".join(lines)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        path.write_text(payload, encoding="utf-8")
        return path


def threshold_for(channel: str, sigma_multiple: float = SIGMA_MULTIPLE) -> Threshold:
    """Acceptance band for one channel, taken from its contract ``sigma``."""
    if channel not in CHANNELS:
        msg = f"channel {channel!r} is not in the contract"
        raise KeyError(msg)
    spec = CHANNELS[channel]
    abs_tol = sigma_multiple * spec.sigma
    if abs_tol < ABSOLUTE_FLOOR:
        abs_tol = ABSOLUTE_FLOOR
    return Threshold(channel=channel, abs_tol=abs_tol, rel_tol=RELATIVE_FLOOR)


def contract_thresholds(
    channels: Iterable[str], sigma_multiple: float = SIGMA_MULTIPLE
) -> tuple[Threshold, ...]:
    return tuple(threshold_for(channel, sigma_multiple) for channel in channels)


def compare(
    name: str,
    baseline: Mapping[str, Sequence[float]],
    candidate: SampleRecord,
    thresholds: Sequence[Threshold] | None = None,
) -> GoldenReport:
    """Diff `candidate` against `baseline`, one entry per channel.

    The baseline is keyed by channel name with one value per step. A channel present in
    the baseline but absent from the candidate is itself a flagged diff, because a channel
    disappearing from a trace is exactly the kind of silent regression this harness exists
    to surface.
    """
    by_channel = {threshold.channel: threshold for threshold in thresholds or ()}
    diffs: list[ChannelDiff] = []
    for channel in sorted(baseline):
        expected = tuple(float(v) for v in baseline[channel])
        if channel not in candidate.channels:
            diffs.append(
                ChannelDiff(
                    channel=channel,
                    threshold=by_channel.get(channel, threshold_for(channel)),
                    max_abs_diff=math.inf,
                    max_rel_diff=math.inf,
                    at_step=-1,
                    baseline_value=expected[-1] if expected else 0.0,
                    candidate_value=math.nan,
                )
            )
            continue
        actual = candidate.series(channel)
        if len(actual) != len(expected):
            diffs.append(
                ChannelDiff(
                    channel=channel,
                    threshold=by_channel.get(channel, threshold_for(channel)),
                    max_abs_diff=math.inf,
                    max_rel_diff=math.inf,
                    at_step=min(len(actual), len(expected)) - 1,
                    baseline_value=expected[min(len(actual), len(expected)) - 1],
                    candidate_value=actual[min(len(actual), len(expected)) - 1],
                )
            )
            continue
        worst = max(
            ((abs(a - b), i, a, b) for i, (a, b) in enumerate(zip(actual, expected, strict=True))),
            key=lambda item: item[0],
        )
        abs_diff, index, candidate_value, baseline_value = worst
        scale = abs(baseline_value)
        if scale > ABSOLUTE_FLOOR:
            rel_diff = abs_diff / scale
        else:
            rel_diff = 0.0 if abs_diff == 0.0 else math.inf
        diffs.append(
            ChannelDiff(
                channel=channel,
                threshold=by_channel.get(channel, threshold_for(channel)),
                max_abs_diff=abs_diff,
                max_rel_diff=rel_diff,
                at_step=index,
                baseline_value=baseline_value,
                candidate_value=candidate_value,
            )
        )
    return GoldenReport(name=name, diffs=tuple(diffs))


def load_baseline(path: Path) -> dict[str, list[float]]:
    """Read a committed baseline file."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        msg = f"{path}: expected a JSON object"
        raise ValueError(msg)
    series = document.get("series")
    if not isinstance(series, Mapping):
        msg = f"{path}: expected a 'series' object of channel -> values"
        raise ValueError(msg)
    out: dict[str, list[float]] = {}
    for key, values in series.items():
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            msg = f"{path}: series[{key!r}] is not a list of numbers"
            raise ValueError(msg)
        numbers: list[float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                msg = f"{path}: series[{key!r}] contains a non-numeric entry {value!r}"
                raise ValueError(msg)
            numbers.append(float(value))
        out[str(key)] = numbers
    return out


def write_baseline(
    path: Path,
    record: SampleRecord,
    description: str,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Commit a baseline for a record. Only ever called deliberately, never by a test."""
    document = {
        "name": record.name,
        "description": description,
        "record_description": record.description,
        "dt_s": record.dt_s,
        "steps": len(record.frames),
        "series": {channel: list(record.series(channel)) for channel in record.channels},
        "thresholds": {c.channel: c.to_dict() for c in contract_thresholds(record.channels)},
    }
    if extra:
        document.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
