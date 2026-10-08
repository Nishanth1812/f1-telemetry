"""Layer 1 physics residuals over published sensor frames (PLAN.md section 9, P7-T2/T3).

Each modelled channel is predicted from other published channels by a small reduced-order
formula, reimplemented here. Nothing is imported from the physics core or the kernels, so
the prediction is independent of the simulator that produced the record. For each sample

    r(t) = y(t) - yhat(t),      z(t) = r(t) / sigma_y

where ``sigma_y`` is the channel's noise floor from ``channels.yaml``. In simulation the
residual *is* the injected fault in physical units (PLAN.md section 9), so the same
number gives detection, per-channel attribution and truth labels.

Models, and the reason each one is what it is:

* ``speed`` - ground speed from body velocity: ``3.6 * hypot(vx, vy)`` km/h.
* ``wheel_speed_<c>`` - rim speed from longitudinal velocity and slip ratio:
  ``3.6 * vx * (1 + slip_ratio_<c> / 100)`` km/h. Slip ratio is the wheel's own state,
  so it is an input here. The wheel radius does not appear because ``wheel_speed`` is a
  linear rim speed in km/h, not an angular speed. Consequence: a fault on ``slip_ratio_<c>``
  also shows as a residual on ``wheel_speed_<c>``. Attribution cannot separate the two
  until the tyre model that predicts slip exists.
* ``tyre_temp_<c>`` - hold-baseline. The P3 lumped node lives in the physics core, which
  this layer may not import, so the prediction is the median of the previous
  ``hold_window`` published samples. The median keeps one spike from contaminating the
  next prediction. The baseline lags a genuine ramp by about half the window, and it
  follows a slow bias drift, so step faults on tyre temperature stay largely invisible.

Channels outside these models carry no residual. That is deliberate: a formula that
cannot be justified is not shipped as a detector. Samples with a missing or non-finite
input or observation are skipped. Layer 0 owns the report of those values.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from f1telemetry.contracts.channels import (
    CORNERS,
    ChannelContract,
    channels_yaml_path,
    load_channel_contract,
)
from f1telemetry.testing.records import SensorFrame

__all__ = [
    "HOLD_WINDOW",
    "MODELLED_CHANNELS",
    "ResidualSample",
    "attribute",
    "residual_z_scores",
]

HOLD_WINDOW: Final[int] = 5

# Unit conversion (m/s to km/h) and percent-to-fraction. Physical constants of the
# representation, not noise floors or thresholds.
KMH_PER_MS: Final[float] = 3.6
PERCENT: Final[float] = 100.0

_SPEED: Final[str] = "speed"
_WHEEL_SPEED: Final[tuple[str, ...]] = tuple(f"wheel_speed_{c.lower()}" for c in CORNERS)
_TYRE_TEMP: Final[tuple[str, ...]] = tuple(f"tyre_temp_{c.lower()}" for c in CORNERS)

MODELLED_CHANNELS: Final[tuple[str, ...]] = (_SPEED, *_WHEEL_SPEED, *_TYRE_TEMP)


@dataclass(frozen=True, slots=True)
class ResidualSample:
    """One channel's residual at one frame, with its z-score for attribution."""

    channel: str
    index: int
    observed: float
    predicted: float
    residual: float
    z_score: float


def residual_z_scores(
    frames: Sequence[SensorFrame],
    contract: ChannelContract | None = None,
    *,
    hold_window: int = HOLD_WINDOW,
) -> tuple[ResidualSample, ...]:
    """Predict every modelled channel at every frame and return the normalised residuals.

    ``contract`` defaults to ``channels.yaml`` loaded from the repository root. A model is
    evaluated only when all of its inputs are present, so a subset record stays analysable.
    Output is sorted by ``(index, channel)``.
    """
    if hold_window < 1:
        msg = f"hold_window must be >= 1, got {hold_window}"
        raise ValueError(msg)
    if len(frames) == 0:
        return ()
    resolved = load_channel_contract(channels_yaml_path()) if contract is None else contract
    known = {channel.name: channel for channel in resolved.channels}
    samples: list[ResidualSample] = []
    for index, frame in enumerate(frames):
        predictions = _kinematic_predictions(frame.values)
        predictions.update(_hold_predictions(frames, index, hold_window))
        for channel, predicted in predictions.items():
            spec = known.get(channel)
            observed = frame.values.get(channel)
            if spec is None or observed is None or spec.sigma <= 0.0:
                continue
            if not (math.isfinite(observed) and math.isfinite(predicted)):
                continue
            residual = observed - predicted
            samples.append(
                ResidualSample(
                    channel=channel,
                    index=index,
                    observed=observed,
                    predicted=predicted,
                    residual=residual,
                    z_score=residual / spec.sigma,
                )
            )
    samples.sort(key=lambda sample: (sample.index, sample.channel))
    return tuple(samples)


def attribute(samples: Sequence[ResidualSample], index: int) -> tuple[ResidualSample, ...]:
    """Channels ranked by ``|z|`` at one frame, largest first, ties broken by name."""
    at_index = [sample for sample in samples if sample.index == index]
    at_index.sort(key=lambda sample: (-abs(sample.z_score), sample.channel))
    return tuple(at_index)


def _kinematic_predictions(values: Mapping[str, float]) -> dict[str, float]:
    predictions: dict[str, float] = {}
    if "vx" not in values:
        return predictions
    vx = values["vx"]
    if "vy" in values:
        predictions[_SPEED] = KMH_PER_MS * math.hypot(vx, values["vy"])
    for corner, channel in zip(CORNERS, _WHEEL_SPEED, strict=True):
        slip = values.get(f"slip_ratio_{corner.lower()}")
        if slip is not None:
            predictions[channel] = KMH_PER_MS * vx * (1.0 + slip / PERCENT)
    return predictions


def _hold_predictions(frames: Sequence[SensorFrame], index: int, window: int) -> dict[str, float]:
    predictions: dict[str, float] = {}
    if index < window:
        return predictions
    for channel in _TYRE_TEMP:
        history: list[float] = []
        for past in range(index - window, index):
            value = frames[past].values.get(channel)
            if value is None or not math.isfinite(value):
                break
            history.append(value)
        else:
            predictions[channel] = statistics.median(history)
    return predictions
