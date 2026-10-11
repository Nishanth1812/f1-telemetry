"""P5 sensor slice — deterministic fixed-rate decimation, quantisation, noise,
seeded fault injection, and ground-truth separation (PLAN.md §8.1, PHASES.md P5).

Reads rates, ranges, quantisation, noise sigma and fault eligibility from the
contract (`channels.yaml` / `f1telemetry.generated.channels`). Produces
corrupted `SensorFrame` samples from clean `GroundTruthStep` arrays, using
simulated (`t_s`) timestamps only. Does not import physics or kernels.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from f1telemetry.contracts.channels import (
    FAULT_TYPES,
    Channel,
    ChannelContract,
    ContractError,
    Quantisation,
    load_channel_contract,
)

# ---------------------------------------------------------------------------
# Contract-level constants re-exported for consumer convenience
# ---------------------------------------------------------------------------
FAULT_TYPES_ORDERED: Final[tuple[str, ...]] = tuple(FAULT_TYPES)

# ---------------------------------------------------------------------------
# Fault taxonomy metadata (PLAN.md §8.1 / PHASES.md P5-T4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FaultMetadata:
    """Static contract-level metadata for one of the ten fault types."""

    name: str
    description: str
    requires_corner: bool  # `swap` requires non-empty `corners`

    def is_eligible_for(self, channel: Channel) -> bool:
        return self.name in channel.fault_eligible


_FAULT_DESC: Final[dict[str, str]] = {
    "dropout": "N consecutive frames missing.",
    "freeze": "Value holds; frame counter increments.",
    "spike": "Single-sample impulse, ±kσ.",
    "step": "Slow bias drift (step offset).",
    "gain": "Scale error: 1 ± k.",
    "noise": "Variance inflation (excess noise).",
    "quantise": "Coarser effective bit depth.",
    "swap": "Per-corner channel swaps (e.g. FL↔FR).",
    "saturate": "Clipping at full-scale.",
    "stale": "Value timestamped t−k, re-sent live.",
}

FAULT_METAS: Final[tuple[FaultMetadata, ...]] = tuple(
    FaultMetadata(
        name=f,
        description=_FAULT_DESC.get(f, ""),
        requires_corner=f == "swap",
    )
    for f in FAULT_TYPES_ORDERED
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_positive_float(value: object, label: str) -> float:
    if not isinstance(value, (int, float)):
        raise ContractError(f"{label}: expected float, got {type(value)}")
    v = float(value)
    if not math.isfinite(v):
        raise ContractError(f"{label}: must be finite, got {v!r}")
    if v < 0:
        raise ContractError(f"{label}: must be non-negative, got {v!r}")
    return v


# ---------------------------------------------------------------------------
# Anti-alias FIR coefficients for fixed-rate decimation from 10 kHz base
# ---------------------------------------------------------------------------

# Simple truncated-sinc low-pass FIR. Cutoff = target_rate / 2 (Nyquist).
# Filter length = 2 * decimation_factor to keep latency modest but provide
# basic alias suppression.  For 200 Hz (factor 50): 101 taps.  For 100 Hz
# (factor 100): 201 taps.  Implemented via np.convolve for clarity; in a hot
# loop it could be replaced by an overlap-save FFT, but stdlib/numpy only
# keeps the dependency footprint minimal.


def _lowpass_fir(decimation_factor: int) -> np.ndarray:
    """Return a normalised truncated-sinc FIR for half-Nyquist cutoff."""
    n_taps = 2 * decimation_factor + 1
    cutoff = 0.5 / decimation_factor  # in units of 10 kHz
    # Windowed sinc
    t = np.arange(n_taps) - decimation_factor
    # Guard division by zero at centre
    h = np.sinc(2 * cutoff * t)  # np.sinc is sin(pi*x)/(pi*x)
    h *= np.hamming(n_taps)
    h = h / np.sum(h)
    return h.astype(np.float64)


def _filter_and_decimate(signal: np.ndarray, factor: int) -> np.ndarray:
    taps = _lowpass_fir(factor)
    edge = (len(taps) - 1) // 2
    padded = np.pad(signal, (edge, edge), mode="edge")
    return np.convolve(padded, taps, mode="valid")[::factor]


# ---------------------------------------------------------------------------
# Decimation
# ---------------------------------------------------------------------------


def decimate_to_rate(
    signal: np.ndarray,
    source_rate: float,
    target_rate: float,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Fixed-rate deterministic decimation with anti-alias FIR.

    Args:
        signal: 1-D float array sampled at ``source_rate`` Hz.  The base
            physics rate is 10 kHz, but the filter is selected by the
            ``target_rate`` argument.
        source_rate: rate of ``signal`` (Hz).  Used only for validation.
        target_rate: output rate (Hz).  Supported contract rates are
            10, 20, 100, 200; anti-alias is guaranteed for 100/200 Hz.
        seed: unused in pure decimation, but kept for API parity with
            the noise/quantisation path.  Ignored for determinism.

    Returns:
        Downsampled array at ``target_rate`` Hz.
    """
    source_rate = _validate_positive_float(float(source_rate), "source_rate")
    target_rate = _validate_positive_float(float(target_rate), "target_rate")
    if target_rate > source_rate:
        raise ContractError(f"target_rate ({target_rate}) exceeds source_rate ({source_rate})")
    if signal.ndim != 1:
        raise ContractError(f"signal must be 1-D, got shape {signal.shape}")
    if len(signal) == 0:
        return np.array([], dtype=np.float32)

    ratio = source_rate / target_rate
    factor = round(ratio)
    if factor < 1 or not math.isclose(ratio, factor, rel_tol=0.0, abs_tol=1e-9):
        raise ContractError("source_rate must be an integer multiple of target_rate")
    down = signal.astype(np.float64)
    # Split large ratios into bounded FIR stages so low-rate channels get
    # their own Nyquist filters without a thousands-tap 10 kHz convolution.
    while factor > 100:
        stage = next((n for n in range(100, 1, -1) if factor % n == 0), 0)
        if stage == 0:
            raise ContractError(f"cannot split decimation ratio {factor} into FIR stages")
        down = _filter_and_decimate(down, stage)
        factor //= stage
    if factor > 1:
        down = _filter_and_decimate(down, factor)
    # Return float32 to match contract dtype, but keep float64 through filter.
    return down.astype(np.float32)


# ---------------------------------------------------------------------------
# Quantisation (contract ``quantise: {bits, full_scale}``)
# ---------------------------------------------------------------------------


def quantise(
    value: float | np.ndarray,
    quantisation: Quantisation | None,
    *,
    seed: int | None = None,
) -> float | np.ndarray:
    """Quantise a scalar to the declared sensor resolution.

    Uses rounded-to-nearest with deterministic tie-breaking (ties to even)
    so the operation is reproducible regardless of RNG seed.
    """
    if np.ndim(value) != 0:
        return quantise_array(np.asarray(value), quantisation, seed=seed)
    if quantisation is None:
        return float(value)
    step = quantisation.step
    # Scale to integer grid, round, scale back.
    scaled = float(value) / step
    rounded = float(np.round(scaled))
    quantised = rounded * step
    # Clamp to full-scale range implied by bits/full_scale (the contract
    # already validates that full_scale covers the range).
    max_scale = float(quantisation.full_scale)
    quantised = max(-max_scale, min(max_scale, quantised))
    return float(quantised)


def quantise_array(
    values: np.ndarray,
    quantisation: Quantisation | None,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Vectorised quantisation."""
    if quantisation is None:
        return values.astype(np.float32) if values.dtype != np.float32 else values
    step = float(quantisation.step)
    max_scale = float(quantisation.full_scale)
    out = np.round(values / step) * step
    out = np.clip(out, -max_scale, max_scale)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Noise (contract ``noise_model`` / ``sigma``)
# ---------------------------------------------------------------------------


def inject_noise(
    value: float | np.ndarray,
    sigma: float,
    noise_model: str,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> float | np.ndarray:
    """Add deterministic noise drawn from the contract noise model.

    ``seed`` is used only if ``rng`` is not provided; supplying ``rng``
    explicitly keeps noise streams separable from quantisation streams,
    which is required when both are applied in sequence.
    """
    if noise_model == "none" or float(sigma) <= 0.0:
        return float(value) if np.ndim(value) == 0 else value

    if rng is None:
        rng = np.random.default_rng(seed if seed is not None else 0)

    if noise_model == "gaussian":
        noise_arr = rng.normal(0.0, float(sigma), size=np.shape(value))
    elif noise_model == "uniform":
        # Uniform on ±sqrt(3)*sigma so stddev equals sigma.
        bound = float(sigma) * math.sqrt(3.0)
        noise_arr = rng.uniform(-bound, bound, size=np.shape(value))
    else:
        raise ContractError(f"unknown noise_model: {noise_model!r}")

    result = np.array(value, dtype=np.float64) + noise_arr
    return float(result) if np.ndim(value) == 0 else result.astype(np.float32)


# ---------------------------------------------------------------------------
# Fault injection (PLAN.md §8.1 / PHASES.md P5-T4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FaultEvent:
    """A single injected-fault occurrence with ground-truth provenance."""

    fault_type: str
    severity: float  # 0.0..1.0 relative intensity; 1.0 = full contract severity
    onset_sample_index: int
    duration_samples: int | None  # None = indefinite / sustained
    ground_truth_label: bool = True  # True = this is the injected fault


# Per-fault parameter structures (kept minimal; full parameterisation could
# be added in P7 analytics without changing this API).


@dataclass(frozen=True, slots=True)
class FaultParameters:
    """Runtime parameters controlling how a fault manifests."""

    severity: float = 1.0
    onset_index: int = 0
    duration_samples: int | None = None
    swap_target_corner: str | None = None  # for ``swap`` only
    step_offset_fraction: float = 0.05  # fraction of range for ``step``
    spike_sigma_factor: float = 3.0  # kσ multiplier for ``spike``
    freeze_hold_index: int | None = None  # which sample to freeze on
    stale_samples: int = 1

    def __post_init__(self) -> None:
        if not math.isfinite(self.severity) or not 0.0 <= self.severity <= 1.0:
            raise ContractError("fault severity must be finite and in [0, 1]")
        if self.onset_index < 0 or (
            self.duration_samples is not None and self.duration_samples < 1
        ):
            raise ContractError("fault onset must be non-negative and duration must be positive")
        if self.stale_samples < 1:
            raise ContractError("stale_samples must be >= 1")


def apply_fault(
    value: float,
    channel: Channel,
    fault_type: str,
    params: FaultParameters,
    *,
    index: int = 0,
    ground_truth_array: np.ndarray | None = None,
    seed: int | None = None,
) -> float:
    """Apply a single fault to one scalar sample.

    Args:
        value: the *corrupted* sensor value from the previous pipeline stage
            (after decimation/noise/quantisation).  The ground-truth value
            should be passed separately via ``ground_truth_array``; this
            function does not alter it.
        channel: the contract channel (for range, corners, eligibility).
        fault_type: one of the ten contract fault names.
        params: runtime severity/onset/duration settings.
        index: current sample index in the output stream (for time-based
            faults such as ``step`` or ``stale``).
        ground_truth_array: optional reference truth array; kept separate
            from the corrupted ``value``.  Not modified.
        seed: deterministic seed for any stochastic components (spike
            direction, noise amplification).

    Returns:
        The corrupted scalar value.
    """
    if fault_type not in FAULT_TYPES_ORDERED:
        raise ContractError(
            f"unknown fault_type {fault_type!r}; allowed {list(FAULT_TYPES_ORDERED)}"
        )

    # Eligibility check: the contract records which faults are permitted.
    # We enforce this at the pipeline level, but defend here as well.
    if fault_type not in channel.fault_eligible:
        # If ineligible, return unchanged (silent pass-through).
        return float(value)

    # Time-based onset/duration gate.
    active = True
    if params.duration_samples is not None:
        active = params.onset_index <= index < params.onset_index + params.duration_samples
    else:
        active = index >= params.onset_index

    if not active:
        return float(value)

    # Fault-specific transformations.
    # Note: ``value`` is already the decimated/noisy/quantised sample.
    # We apply the corruption *after* those stages, which matches the
    # pipeline order: physics → decimate → quantise/noise → fault → wire.

    if fault_type == "dropout":
        # Return NaN as a sentinel for missing; wire-level must filter it.
        return float("nan")

    elif fault_type == "freeze":
        hold_index = (
            params.onset_index if params.freeze_hold_index is None else params.freeze_hold_index
        )
        if ground_truth_array is not None and 0 <= hold_index < len(ground_truth_array):
            return float(ground_truth_array[hold_index])
        return float(value)

    elif fault_type == "spike":
        # Single-sample impulse.  If called repeatedly at the same index,
        # we inject; otherwise we return unchanged.
        if index == params.onset_index:
            rng = np.random.default_rng(None if seed is None else seed + index)
            direction = 1.0 if rng.random() > 0.5 else -1.0
            spike_mag = float(params.spike_sigma_factor * float(channel.sigma))
            return float(value) + direction * spike_mag
        return float(value)

    elif fault_type == "step":
        # Slow bias drift: add a fraction of the declared range.
        range_span = float(channel.range_max - channel.range_min)
        offset = float(params.step_offset_fraction) * range_span * params.severity
        return float(value) + offset

    elif fault_type == "gain":
        # Scale error.
        gain_factor = 1.0 + float(params.severity) * 0.2
        return float(value) * gain_factor

    elif fault_type == "noise":
        # Variance inflation: add extra noise with amplified sigma.
        extra_sigma = float(channel.sigma) * float(params.severity) * 2.0
        rng = np.random.default_rng(None if seed is None else seed + index)
        extra = float(rng.normal(0.0, extra_sigma))
        return float(value) + extra

    elif fault_type == "quantise":
        # Coarser effective bit depth: reduce quantisation bits by severity.
        if channel.quantisation is not None:
            reduced_bits = max(1, int(channel.quantisation.bits - params.severity * 4))
            # Re-quantise with coarser step.
            reduced_step = float(channel.quantisation.full_scale) / float(1 << (reduced_bits - 1))
            quantised = float(np.round(float(value) / reduced_step)) * reduced_step
            max_scale = float(channel.quantisation.full_scale)
            return max(-max_scale, min(max_scale, quantised))
        return float(value)

    elif fault_type == "swap":
        # Swap requires non-empty corners (enforced by contract loader).
        # This implementation operates on the scalar value; for per-corner
        # channels (e.g. wheel_speed_fl) the caller swaps the array
        # indices before feeding this pipeline, or passes the swapped
        # value directly as ``value``.  We document this clearly.
        if params.swap_target_corner is not None:
            # If ``value`` came from a different corner, return it as-is
            # (the swap is performed upstream in the array layer).
            pass
        return float(value)

    elif fault_type == "saturate":
        midpoint = 0.5 * (channel.range_min + channel.range_max)
        rail = channel.range_max if value >= midpoint else channel.range_min
        return float(value + params.severity * (rail - value))

    elif fault_type == "stale":
        source_index = index - params.stale_samples
        if ground_truth_array is not None and 0 <= source_index < len(ground_truth_array):
            return float(ground_truth_array[source_index])
        # Replay value from ``stale_offset`` samples ago.  The full pipeline
        # manages an offset ring buffer; here we approximate by returning
        # the current value unchanged and documenting that the caller must
        # manage the delay buffer externally.
        return float(value)

    else:
        # Defensive fallback for any future extension.
        return float(value)


# ---------------------------------------------------------------------------
# Ground-truth / corrupted separation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GroundTruthStep:
    """Physics truth at one instant (not a sensor reading)."""

    t_s: float  # simulated time in seconds
    values: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class SensorFrame:
    """One published sample per channel after the sensor pipeline."""

    t_s: float  # simulated time in seconds (same time base as GroundTruthStep)
    values: Mapping[str, float]

    def to_wire(self) -> dict[str, object]:
        return {"t_s": float(self.t_s), "channels": dict(self.values)}


@dataclass(frozen=True, slots=True)
class FaultAnnotation:
    """Provenance metadata linking a corrupted frame back to its ground truth."""

    fault_type: str
    severity: float
    onset_t_s: float
    ground_truth_t_s: float
    label: bool = True
    channel_name: str = ""
    duration_samples: int | None = None
    seed: int | None = None


# ---------------------------------------------------------------------------
# Pipeline class (P5-T5 / P5-T6)
# ---------------------------------------------------------------------------


class SensorPipeline:
    """End-to-end sensor chain: decimate → quantise → noise → fault → wire.

    Keeps ``ground_truth`` arrays separate from ``corrupted`` outputs, and
    attaches deterministic ``FaultAnnotation`` records for analytics (P7).
    """

    def __init__(
        self,
        contract_path: Path | str | None = None,
        *,
        base_rate_hz: float = 10000.0,
        seed: int = 42,
    ) -> None:
        if contract_path is None:
            contract_path = _find_contract_root(Path(__file__)) / "channels.yaml"
        self.contract: ChannelContract = load_channel_contract(Path(contract_path))
        self.base_rate_hz: float = _validate_positive_float(float(base_rate_hz), "base_rate_hz")
        self.seed: int = int(seed)
        self._rng: np.random.Generator = np.random.default_rng(self.seed)

    # --- Internal helpers ---------------------------------------------------

    def _channel_by_name(self, name: str) -> Channel:
        try:
            return self.contract.by_name(name)
        except KeyError as exc:
            raise ContractError(f"channel {name!r} not in contract") from exc

    # --- Pipeline stages -----------------------------------------------------

    def decimate_channel(
        self,
        signal: np.ndarray,
        channel_name: str,
        source_rate: float | None = None,
    ) -> np.ndarray:
        ch = self._channel_by_name(channel_name)
        rate = ch.rate_hz
        if rate is None:
            raise ContractError(
                f"channel {channel_name!r} is event-driven (rate_hz: null); cannot decimate"
            )
        src = float(source_rate) if source_rate is not None else self.base_rate_hz
        return decimate_to_rate(signal, src, float(rate), seed=self.seed)

    def quantise_channel(
        self,
        value: float | np.ndarray,
        channel_name: str,
        *,
        seed: int | None = None,
    ) -> float | np.ndarray:
        ch = self._channel_by_name(channel_name)
        return quantise(value, ch.quantisation, seed=seed)

    def noise_channel(
        self,
        value: float | np.ndarray,
        channel_name: str,
        *,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> float | np.ndarray:
        ch = self._channel_by_name(channel_name)
        return inject_noise(
            value,
            float(ch.sigma),
            ch.noise_model,
            seed=seed,
            rng=rng,
        )

    def fault_channel(
        self,
        value: float,
        channel_name: str,
        fault_type: str,
        params: FaultParameters,
        *,
        index: int = 0,
        ground_truth_array: np.ndarray | None = None,
        seed: int | None = None,
    ) -> float:
        ch = self._channel_by_name(channel_name)
        # Check eligibility; pipeline skips silently if ineligible.
        return apply_fault(
            value,
            ch,
            fault_type,
            params,
            index=index,
            ground_truth_array=ground_truth_array,
            seed=seed,
        )

    # --- Full pipeline --------------------------------------------------------

    def process(
        self,
        ground_truth: Mapping[str, np.ndarray],
        *,
        active_faults: Sequence[tuple[str, FaultParameters]] | None = None,
        seed: int | None = None,
    ) -> tuple[Mapping[str, np.ndarray], Mapping[str, Sequence[FaultAnnotation]]]:
        """Run the full P5 pipeline over ground-truth arrays.

        Args:
            ground_truth: mapping ``channel_name → np.ndarray`` of clean
                samples at ``base_rate_hz``.  This array is **not modified**.
            active_faults: optional list of ``(fault_type, params)``; in a
                full pipeline each fault would also specify the target
                channel(s) and onset/duration.  For this slice we apply
                globally to demonstrate the mechanism.
            seed: overrides the pipeline seed for reproducibility.

        Returns:
            ``(corrupted_samples, annotations)``.  ``annotations`` is a
            mapping ``fault_type → [FaultAnnotation, ...]`` tracking every
            injected event for audit/replay.
        """
        annotations: dict[str, list[FaultAnnotation]] = {f: [] for f in FAULT_TYPES_ORDERED}
        run_seed = self.seed if seed is None else int(seed)
        rng = np.random.default_rng(run_seed)
        clean: dict[str, np.ndarray] = {}
        channels: dict[str, Channel] = {}
        for name, values in ground_truth.items():
            ch = self._channel_by_name(name)
            source = np.asarray(values)
            if source.ndim != 1 or not np.issubdtype(source.dtype, np.number):
                raise ContractError(f"{name}: ground truth must be a numeric 1-D array")
            if not np.all(np.isfinite(source)):
                raise ContractError(f"{name}: ground truth values must be finite")
            channels[name] = ch
            if ch.event:
                clean[name] = source.astype(np.float32, copy=True)
                continue
            sampled = self.decimate_channel(source, name)
            quantised = quantise_array(sampled, ch.quantisation)
            clean[name] = np.asarray(
                self.noise_channel(quantised, name, seed=run_seed, rng=rng), dtype=np.float32
            )

        corrupted = {name: values.copy() for name, values in clean.items()}
        for fault_type, params in active_faults or ():
            if fault_type not in FAULT_TYPES_ORDERED:
                raise ContractError(f"unknown fault_type {fault_type!r}")
            seen_pairs: set[tuple[str, str]] = set()
            for name, channel in channels.items():
                if fault_type not in channel.fault_eligible:
                    continue
                start = min(params.onset_index, len(corrupted[name]))
                stop = (
                    len(corrupted[name])
                    if params.duration_samples is None
                    else min(len(corrupted[name]), params.onset_index + params.duration_samples)
                )
                if start >= stop:
                    continue
                if fault_type == "swap":
                    if channel.corner is None or params.swap_target_corner is None:
                        raise ContractError("swap requires a corner channel and swap_target_corner")
                    target = f"{channel.base_name}_{params.swap_target_corner.lower()}"
                    if target not in corrupted or fault_type not in channels[target].fault_eligible:
                        raise ContractError(f"swap target {target!r} is not an eligible channel")
                    pair = (min(name, target), max(name, target))
                    if pair in seen_pairs:
                        continue
                    if len(corrupted[name]) != len(corrupted[target]):
                        raise ContractError("swap channels must have the same sample count")
                    left, right = corrupted[name].copy(), corrupted[target].copy()
                    corrupted[name][start:stop] = right[start:stop]
                    corrupted[target][start:stop] = left[start:stop]
                    seen_pairs.add(pair)
                    targets = (name, target)
                else:
                    snapshot = corrupted[name].copy()
                    for index in range(start, stop):
                        corrupted[name][index] = self.fault_channel(
                            float(snapshot[index]),
                            name,
                            fault_type,
                            params,
                            index=index,
                            ground_truth_array=snapshot,
                            seed=run_seed,
                        )
                    targets = (name,)
                for target_name in targets:
                    target_channel = channels[target_name]
                    target_rate = target_channel.rate_hz or 1.0
                    annotations[fault_type].append(
                        FaultAnnotation(
                            fault_type=fault_type,
                            severity=params.severity,
                            onset_t_s=params.onset_index / target_rate,
                            ground_truth_t_s=max(
                                0,
                                params.onset_index - params.stale_samples
                                if fault_type == "stale"
                                else params.onset_index,
                            )
                            / target_rate,
                            channel_name=target_name,
                            duration_samples=params.duration_samples,
                            seed=run_seed,
                        )
                    )
        return corrupted, {kind: tuple(events) for kind, events in annotations.items()}


# ---------------------------------------------------------------------------
# Convenience: load contract from repo root by walking up
# ---------------------------------------------------------------------------


def _find_contract_root(start: Path | None = None) -> Path:
    origin = Path(__file__) if start is None else start
    for parent in origin.resolve().parents:
        if (parent / "channels.yaml").is_file():
            return parent
    raise ContractError("could not locate channels.yaml above sensors.py")


# ---------------------------------------------------------------------------
# Module-level convenience exports
# ---------------------------------------------------------------------------

__all__ = [
    "FAULT_METAS",
    "FAULT_TYPES_ORDERED",
    "FaultAnnotation",
    "FaultEvent",
    "FaultMetadata",
    "FaultParameters",
    "GroundTruthStep",
    "SensorFrame",
    "SensorPipeline",
    "apply_fault",
    "decimate_to_rate",
    "inject_noise",
    "quantise",
    "quantise_array",
]
