"""Layer 2 multivariate SPC and Layer 3 temporal persistence (PLAN.md section 9, P7).

Layer 2 stacks per-sample z-score vectors into a feature matrix over a clean
calibration window, then runs PCA via numpy SVD (no new dependencies). For each
channel ``j`` the z-score is ``(y - mu_j) / sigma_j`` where ``mu_j`` is the
calibration-window mean and ``sigma_j`` is the channel's noise floor from
``channels.yaml``. With ``X = U S V^T`` the kept-component eigenvalues are
``lambda = S^2 / (n - 1)``, and each sample scores:

* ``T2`` (Hotelling, in-manifold): ``sum(t_j^2 / lambda_j)`` over kept scores;
* ``Q``/SPE (out-of-manifold): ``||z||^2 - ||t||^2``, clipped at zero.

Both thresholds are the empirical ``level`` quantile (default 99.7%) of the
calibration window, not theory values. Alarms fire on strict exceedance, so the
calibration window itself scores silent by construction.

Layer 3 runs CUSUM or EWMA on a statistic series for slow drift (a thermal ramp
never trips a static threshold), plus a k-of-n persistence rule: alarm only if
the statistic exceeds threshold in at least ``k`` of the last ``n`` windows.
That rule is what drives false alarms toward zero on a fast stream.

Nothing here is imported from the physics core or the kernels. Frames with a
missing or non-finite value are skipped at score time; Layer 0 owns reporting
those values. A dirty calibration window is a programming error, so fitting
raises instead of skipping.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from f1telemetry.contracts.channels import (
    ChannelContract,
    channels_yaml_path,
    load_channel_contract,
)
from f1telemetry.testing.records import SensorFrame

__all__ = [
    "DEFAULT_LEVEL",
    "MultivariateModel",
    "MultivariateSample",
    "cusum",
    "ewma",
    "fit_multivariate",
    "persistent_alarms",
    "score_multivariate",
]

DEFAULT_LEVEL: Final[float] = 0.997


@dataclass(frozen=True, slots=True)
class MultivariateModel:
    """PCA manifold fitted on a clean calibration window."""

    channels: tuple[str, ...]
    means: tuple[float, ...]
    sigmas: tuple[float, ...]
    components: tuple[tuple[float, ...], ...]
    eigenvalues: tuple[float, ...]
    t2_threshold: float
    q_threshold: float
    level: float

    @property
    def n_components(self) -> int:
        """Number of kept principal components."""
        return len(self.components)


@dataclass(frozen=True, slots=True)
class MultivariateSample:
    """One frame's Layer 2 statistics and single-window exceedance flags."""

    index: int
    t2: float
    q: float
    t2_exceeds: bool
    q_exceeds: bool


def fit_multivariate(
    frames: Sequence[SensorFrame],
    channels: Sequence[str],
    contract: ChannelContract | None = None,
    *,
    n_components: int = 1,
    level: float = DEFAULT_LEVEL,
) -> MultivariateModel:
    """Fit the Layer 2 PCA manifold on clean calibration ``frames``.

    ``contract`` defaults to ``channels.yaml`` loaded from the repository root
    and supplies the per-channel noise floors. Raises ``ValueError`` on fewer
    than two channels, duplicate or unknown channels, channels with no noise
    scale, fewer than two frames, a missing or non-finite calibration value, a
    degenerate (zero-variance) kept component, or an out-of-range argument.
    """
    names = tuple(channels)
    if len(names) < 2:
        msg = f"multivariate fit needs at least 2 channels, got {len(names)}"
        raise ValueError(msg)
    if len(set(names)) != len(names):
        msg = f"duplicate channels in {list(names)}"
        raise ValueError(msg)
    if not 1 <= n_components < len(names):
        msg = f"n_components must satisfy 1 <= n_components < {len(names)}, got {n_components}"
        raise ValueError(msg)
    if not 0.0 < level < 1.0:
        msg = f"level must be in (0, 1), got {level}"
        raise ValueError(msg)
    if len(frames) < 2:
        msg = f"calibration needs at least 2 frames, got {len(frames)}"
        raise ValueError(msg)
    resolved = load_channel_contract(channels_yaml_path()) if contract is None else contract
    sigmas: list[float] = []
    for name in names:
        try:
            spec = resolved.by_name(name)
        except KeyError as exc:
            msg = f"unknown channel {name!r}"
            raise ValueError(msg) from exc
        if spec.sigma <= 0.0:
            msg = f"channel {name!r} has no noise scale (sigma={spec.sigma})"
            raise ValueError(msg)
        sigmas.append(spec.sigma)
    raw: list[list[float]] = []
    for index, frame in enumerate(frames):
        row: list[float] = []
        for name in names:
            value = frame.values.get(name)
            if value is None or not math.isfinite(value):
                msg = f"calibration frame {index} has no usable value for {name!r}"
                raise ValueError(msg)
            row.append(value)
        raw.append(row)
    matrix = np.array(raw, dtype=float)
    means = tuple(float(value) for value in matrix.mean(axis=0))
    standardized = (matrix - np.array(means)) / np.array(sigmas)
    _left, singular, loadings_t = np.linalg.svd(standardized, full_matrices=False)
    eigen = tuple(float(s * s) / float(len(frames) - 1) for s in singular)
    for component in range(n_components):
        if eigen[component] <= 0.0:
            msg = "degenerate calibration: zero variance along a kept component"
            raise ValueError(msg)
    kept = tuple(
        tuple(float(value) for value in loadings_t[component]) for component in range(n_components)
    )
    basis = loadings_t[:n_components]
    weights = standardized @ basis.T
    scales = np.array(eigen[:n_components])
    t2 = np.sum(weights * weights / scales, axis=1)
    q = np.maximum(
        np.sum(standardized * standardized, axis=1) - np.sum(weights * weights, axis=1), 0.0
    )
    return MultivariateModel(
        channels=names,
        means=means,
        sigmas=tuple(sigmas),
        components=kept,
        eigenvalues=eigen[:n_components],
        t2_threshold=float(np.quantile(t2, level)),
        q_threshold=float(np.quantile(q, level)),
        level=level,
    )


def score_multivariate(
    frames: Sequence[SensorFrame], model: MultivariateModel
) -> tuple[MultivariateSample, ...]:
    """Score ``frames`` against a fitted manifold; T2/Q exceed on strict ``>``.

    Frames with a missing or non-finite value on any model channel are skipped.
    Output is in frame order.
    """
    basis = np.array(model.components, dtype=float)
    scales = np.array(model.eigenvalues, dtype=float)
    samples: list[MultivariateSample] = []
    for index, frame in enumerate(frames):
        row: list[float] = []
        for name, mean, sigma in zip(model.channels, model.means, model.sigmas, strict=True):
            value = frame.values.get(name)
            if value is None or not math.isfinite(value):
                break
            row.append((value - mean) / sigma)
        else:
            point = np.array(row, dtype=float)
            weights = basis @ point
            t2 = float(np.sum(weights * weights / scales))
            residual = float(np.sum(point * point) - np.sum(weights * weights))
            scored_q = max(residual, 0.0)
            samples.append(
                MultivariateSample(
                    index=index,
                    t2=t2,
                    q=scored_q,
                    t2_exceeds=t2 > model.t2_threshold,
                    q_exceeds=scored_q > model.q_threshold,
                )
            )
    return tuple(samples)


def cusum(values: Sequence[float], *, drift: float, threshold: float) -> tuple[float, ...]:
    """One-sided upper CUSUM: ``g_t = max(0, g_{t-1} + x_t - drift)``, ``g`` from 0.

    Trips when ``g_t > threshold``. Raises ``ValueError`` on a non-positive
    threshold or a non-finite drift, threshold or sample.
    """
    if not math.isfinite(drift):
        msg = f"drift must be finite, got {drift}"
        raise ValueError(msg)
    if not math.isfinite(threshold) or threshold <= 0.0:
        msg = f"threshold must be finite and > 0, got {threshold}"
        raise ValueError(msg)
    stats: list[float] = []
    running = 0.0
    for value in values:
        if not math.isfinite(value):
            msg = f"CUSUM input must be finite, got {value}"
            raise ValueError(msg)
        running = max(0.0, running + value - drift)
        stats.append(running)
    return tuple(stats)


def ewma(values: Sequence[float], *, alpha: float) -> tuple[float, ...]:
    """Exponentially weighted moving average: ``s_0 = x_0`` then ``s_t = a x_t + (1-a) s``.

    Raises ``ValueError`` unless ``0 < alpha <= 1`` or on a non-finite sample.
    """
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        msg = f"alpha must satisfy 0 < alpha <= 1, got {alpha}"
        raise ValueError(msg)
    stats: list[float] = []
    for value in values:
        if not math.isfinite(value):
            msg = f"EWMA input must be finite, got {value}"
            raise ValueError(msg)
        stats.append(value if not stats else alpha * value + (1.0 - alpha) * stats[-1])
    return tuple(stats)


def persistent_alarms(exceeds: Sequence[bool], *, k: int, n: int) -> tuple[bool, ...]:
    """K-of-n persistence: alarm at ``i`` iff at least ``k`` of the last ``n`` flags hold.

    The window is inclusive of the current sample and truncated at the start of
    the series. Raises ``ValueError`` unless ``1 <= k <= n``.
    """
    if n < 1:
        msg = f"n must be >= 1, got {n}"
        raise ValueError(msg)
    if not 1 <= k <= n:
        msg = f"k must satisfy 1 <= k <= n ({n}), got {k}"
        raise ValueError(msg)
    flags = tuple(bool(flag) for flag in exceeds)
    alarms: list[bool] = []
    for index in range(len(flags)):
        window = flags[max(0, index - n + 1) : index + 1]
        alarms.append(sum(window) >= k)
    return tuple(alarms)
