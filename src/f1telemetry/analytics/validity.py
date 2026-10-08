"""Layer 0 validity checks over published sensor frames (PLAN.md section 9, P7-T1).

Deterministic, no ML, no physics: every threshold is derived from ``channels.yaml``
through :mod:`f1telemetry.contracts.channels`, so a contract edit moves the checks
with no other code change. The four rules, per channel unless stated:

* ``range`` — sample outside the contract ``[range_min, range_max]`` (non-finite
  counts as out of range, since no NaN or Inf may ever appear on the wire);
* ``rate_of_change`` — per-step ``|delta|`` above ``rate_sigma_multiple * sigma``
  plus one quantisation step, where ``sigma`` and the step come from the contract.
  Channels with ``sigma == 0`` (discrete state such as ``gear``) carry no per-step
  noise scale, so this rule does not apply to them;
* ``stuck`` — value bit-identical for ``stuck_window`` consecutive frames;
* ``timestamp_monotonicity`` — frame timestamp not strictly increasing (reported
  on ``t_s``, the one finding that is not per-channel).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

from f1telemetry.contracts.channels import (
    Channel,
    ChannelContract,
    channels_yaml_path,
    load_channel_contract,
)
from f1telemetry.testing.records import SensorFrame

__all__ = [
    "TIME_CHANNEL",
    "ValidityFinding",
    "ValidityRule",
    "check_validity",
]

ValidityRule = Literal["range", "rate_of_change", "stuck", "timestamp_monotonicity"]

TIME_CHANNEL: Final[str] = "t_s"


@dataclass(frozen=True, slots=True)
class ValidityFinding:
    """One tripped Layer 0 rule: which channel, which frame index, which rule."""

    channel: str
    index: int
    rule: ValidityRule
    value: float | None
    detail: str


def check_validity(
    frames: Sequence[SensorFrame],
    contract: ChannelContract | None = None,
    *,
    rate_sigma_multiple: float = 25.0,
    stuck_window: int = 5,
) -> tuple[ValidityFinding, ...]:
    """Run all four Layer 0 rules over ``frames`` in frame order.

    ``contract`` defaults to ``channels.yaml`` loaded from the repository root.
    Only channels present in the frames *and* known to the contract are checked;
    anything else is skipped, so a subset record stays analysable. Findings are
    sorted by ``(index, channel, rule)`` for deterministic output.
    """
    if rate_sigma_multiple <= 0.0:
        msg = f"rate_sigma_multiple must be > 0, got {rate_sigma_multiple}"
        raise ValueError(msg)
    if stuck_window < 2:
        msg = f"stuck_window must be >= 2, got {stuck_window}"
        raise ValueError(msg)
    if len(frames) == 0:
        return ()
    resolved = load_channel_contract(channels_yaml_path()) if contract is None else contract
    known = {channel.name: channel for channel in resolved.channels}
    findings: list[ValidityFinding] = []
    findings.extend(_check_timestamps(frames))
    for name in frames[0].values:
        spec = known.get(name)
        if spec is None:
            continue
        series = tuple(float(frame.values[name]) for frame in frames)
        findings.extend(_check_range(name, series, spec))
        findings.extend(_check_rate(name, series, spec, rate_sigma_multiple))
        findings.extend(_check_stuck(name, series, stuck_window))
    findings.sort(key=lambda finding: (finding.index, finding.channel, finding.rule))
    return tuple(findings)


def _check_timestamps(frames: Sequence[SensorFrame]) -> list[ValidityFinding]:
    findings: list[ValidityFinding] = []
    for index in range(1, len(frames)):
        stamp = frames[index].t_s
        if not math.isfinite(stamp) or stamp <= frames[index - 1].t_s:
            findings.append(
                ValidityFinding(
                    channel=TIME_CHANNEL,
                    index=index,
                    rule="timestamp_monotonicity",
                    value=stamp,
                    detail=f"t_s {stamp} is not strictly after {frames[index - 1].t_s}",
                )
            )
    return findings


def _check_range(channel: str, series: Sequence[float], spec: Channel) -> list[ValidityFinding]:
    findings: list[ValidityFinding] = []
    for index, value in enumerate(series):
        if not math.isfinite(value):
            findings.append(
                ValidityFinding(
                    channel=channel,
                    index=index,
                    rule="range",
                    value=value,
                    detail=f"value {value} is non-finite",
                )
            )
        elif value < spec.range_min or value > spec.range_max:
            findings.append(
                ValidityFinding(
                    channel=channel,
                    index=index,
                    rule="range",
                    value=value,
                    detail=(
                        f"value {value} outside contract range [{spec.range_min}, {spec.range_max}]"
                    ),
                )
            )
    return findings


def _check_rate(
    channel: str, series: Sequence[float], spec: Channel, multiple: float
) -> list[ValidityFinding]:
    if spec.sigma <= 0.0:
        return []
    quantum = spec.quantisation.step if spec.quantisation is not None else 0.0
    limit = multiple * spec.sigma + quantum
    findings: list[ValidityFinding] = []
    for index in range(1, len(series)):
        previous, current = series[index - 1], series[index]
        if not math.isfinite(previous) or not math.isfinite(current):
            continue
        jump = abs(current - previous)
        if jump > limit:
            findings.append(
                ValidityFinding(
                    channel=channel,
                    index=index,
                    rule="rate_of_change",
                    value=current,
                    detail=(
                        f"|delta| {jump} exceeds {multiple} * sigma "
                        f"({spec.sigma}) + quantum ({quantum})"
                    ),
                )
            )
    return findings


def _check_stuck(channel: str, series: Sequence[float], window: int) -> list[ValidityFinding]:
    findings: list[ValidityFinding] = []
    run = 1
    for index in range(1, len(series)):
        if series[index] == series[index - 1]:
            run += 1
            if run == window:
                findings.append(
                    ValidityFinding(
                        channel=channel,
                        index=index,
                        rule="stuck",
                        value=series[index],
                        detail=f"value unchanged for {window} consecutive frames",
                    )
                )
        else:
            run = 1
    return findings
