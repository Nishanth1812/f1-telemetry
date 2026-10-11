"""P10 first slice: setup DoE + quadratic surrogate (PLAN.md §10 item 5).

That item needs a designed experiment over the setup vector, a fitted surrogate,
a predicted optimum, a re-simulation and a published predicted-vs-actual error.
This module is the first slice of it: the design (a seeded Latin hypercube over a
declared setup box), the per-run setup application, the response assembly from
sweep rows, and the least-squares quadratic response surface. No CLI, no docs page
yet.

**Setup, not coefficients.** Every dimension below is a quantity the garage sets
between runs, and every one is applied as a per-run in-memory override: a
:class:`~f1telemetry.testing.scenarios.Scenario` copy, a
:class:`~f1telemetry.contracts.car_spec.KernelConfig` copy, or both. ``car_spec.yaml``
is never edited, no coefficient is retuned, and the physics is untouched:

* ``brake_bias_front`` — front share of the brake-bias vector, in ``[0, 1]``.
  Maps to ``Scenario.brake_bias`` as ``(f/2, f/2, (1-f)/2, (1-f)/2)``, so the four
  shares still sum to one and the total brake demand is unchanged.
* ``roll_stiffness_front_fraction`` — the suspension balance knob, in ``(0, 1)``.
  Maps to ``KernelConfig.roll_stiffness_front_fraction`` on a per-run copy.
* ``front_static_camber_deg`` — front static camber, any finite value. Maps to the
  front entry of ``KernelConfig.axle_static_camber_deg`` on a per-run copy; the
  base config's array is never mutated.
* ``cold_tyre_pressure_psi`` — cold inflation pressure, ``>= 0``. Maps to
  ``KernelConfig.thermal_tyre_initial_pressure_psi_gauge`` on a per-run copy. It
  seeds the tyre gas mass (a supplier-declared setup value, not a model
  coefficient) and, in the current lumped thermal model, moves only the published
  ``tyre_pressure_*`` channels — never the motion proxies. The tests pin that
  null dynamics effect rather than fitting through it silently.

Wing (flap) angles are deliberately *not* a dimension: the model has no
aero-balance input — the frame publication omits flap channels and says so — so a
wing dimension would have nothing to act on short of a physics change. Asking for
an unknown dimension raises ``KeyError``.

**Responses.** :func:`response_table` assembles one row per
:class:`~f1telemetry.testing.sweep.SweepRow`. Without a track it reports the
documented proxy metrics with the same definitions as
:func:`~f1telemetry.testing.sweep.results_table` (final speed, settled lateral
acceleration over the final 0.2 s). With a track it projects the run through
:func:`~f1telemetry.testing.reference_lap.assess_scenario_lap` and reports the
first-lap sector durations, the first completed lap time (``None`` when the stream
contains no lap crossing), and the validity verdict.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from f1telemetry.contracts.car_spec import KernelConfig
from f1telemetry.laps import TrackEventKind
from f1telemetry.testing.reference_lap import assess_scenario_lap
from f1telemetry.testing.run_manifest import RunManifest
from f1telemetry.testing.scenarios import (
    CONTROL_STEPS,
    ControlLaw,
    Scenario,
    ScenarioRun,
    run_scenario,
)
from f1telemetry.testing.sweep import SweepRow
from f1telemetry.tracks import Track

__all__ = [
    "FitScore",
    "LapResponse",
    "QuadraticSurrogate",
    "SetupBox",
    "SetupDim",
    "apply_setup",
    "fit_quadratic",
    "lap_responses",
    "latin_hypercube",
    "predict_optimum",
    "proxy_responses",
    "response_table",
    "run_design_points",
]

#: Settled lateral acceleration averages this window at the end of the run,
#: the same window :mod:`f1telemetry.testing.sweep` uses.
_SETTLED_WINDOW_S: Final[float] = 0.2

#: Supported setup dimensions and where each one is applied per run.
_SETUP_DIMS: Final[frozenset[str]] = frozenset(
    {
        "brake_bias_front",
        "roll_stiffness_front_fraction",
        "front_static_camber_deg",
        "cold_tyre_pressure_psi",
    }
)

ManifestFactory = Callable[[int, Mapping[str, float]], RunManifest | None]


@dataclass(frozen=True, slots=True)
class SetupDim:
    """One setup axis: its name and its closed ``[low, high]`` box bounds."""

    name: str
    low: float
    high: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("setup dimension needs a nonempty name")
        for label, value in (("low", self.low), ("high", self.high)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"setup dimension {self.name!r}: {label} must be a number")
            if not math.isfinite(float(value)):
                raise ValueError(f"setup dimension {self.name!r}: {label} must be finite")
        if not float(self.low) < float(self.high):
            raise ValueError(
                f"setup dimension {self.name!r}: low ({self.low}) must be below high ({self.high})"
            )


@dataclass(frozen=True, slots=True)
class SetupBox:
    """The declared setup box: one or more named, bounded setup dimensions."""

    dims: tuple[SetupDim, ...]

    def __post_init__(self) -> None:
        if not self.dims:
            raise ValueError("a setup box needs at least one dimension")
        names = [dim.name for dim in self.dims]
        if len(set(names)) != len(names):
            raise ValueError(f"a setup box needs unique dimension names, got {names}")

    @property
    def names(self) -> tuple[str, ...]:
        """Dimension names, in box order."""
        return tuple(dim.name for dim in self.dims)

    @property
    def ndim(self) -> int:
        """Number of setup dimensions."""
        return len(self.dims)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """``(low, high)`` vectors, in box order."""
        low = np.array([dim.low for dim in self.dims], dtype=np.float64)
        high = np.array([dim.high for dim in self.dims], dtype=np.float64)
        return low, high


def latin_hypercube(box: SetupBox, n_samples: int, seed: int) -> np.ndarray:
    """Seeded Latin-hypercube samples over ``box``, in physical units.

    Each dimension's ``[0, 1)`` range is split into ``n_samples`` equal strata;
    every stratum contributes exactly one sample per dimension (a random
    permutation), placed uniformly at random inside the stratum, then mapped to
    ``[low, high]``. The seed is required — ``None`` would make the design
    unrepeatable, which fails the determinism rule the sweep table relies on.
    Returns an ``(n_samples, box.ndim)`` ``float64`` array, one design point per
    row, columns in :attr:`SetupBox.names` order.
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError(f"n_samples must be an int >= 1, got {n_samples!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an int, got {seed!r}")
    low, high = box.bounds()
    rng = np.random.default_rng(seed)
    unit = np.empty((n_samples, box.ndim), dtype=np.float64)
    for dim in range(box.ndim):
        strata = rng.permutation(n_samples).astype(np.float64)
        unit[:, dim] = (strata + rng.random(n_samples)) / float(n_samples)
    return low + unit * (high - low)


def apply_setup(
    base_scenario: Scenario,
    base_config: KernelConfig,
    point: Mapping[str, float],
) -> tuple[Scenario, KernelConfig]:
    """Apply one setup point as per-run overrides; the inputs are never mutated.

    Unknown dimension names raise ``KeyError`` (wing angles land here: there is
    no aero-balance input for them to act on); out-of-range values raise
    ``ValueError``. Returns the ``(scenario, config)`` pair to simulate.
    """
    scenario = base_scenario
    config = base_config
    for name, raw in point.items():
        if name not in _SETUP_DIMS:
            raise KeyError(
                f"unknown setup dimension {name!r}; supported dimensions are {sorted(_SETUP_DIMS)}"
            )
        value = _checked_setup_value(name, raw)
        if name == "brake_bias_front":
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"brake_bias_front must be in [0, 1], got {value!r}")
            scenario = dataclasses.replace(
                scenario,
                brake_bias=(value / 2.0, value / 2.0, (1.0 - value) / 2.0, (1.0 - value) / 2.0),
            )
        elif name == "roll_stiffness_front_fraction":
            if not 0.0 < value < 1.0:
                raise ValueError(f"roll_stiffness_front_fraction must be in (0, 1), got {value!r}")
            config = dataclasses.replace(config, roll_stiffness_front_fraction=value)
        elif name == "front_static_camber_deg":
            camber = np.array(config.axle_static_camber_deg, dtype=np.float64)
            camber[0] = value
            config = dataclasses.replace(config, axle_static_camber_deg=camber)
        else:  # cold_tyre_pressure_psi
            if value < 0.0:
                raise ValueError(f"cold_tyre_pressure_psi must be >= 0, got {value!r}")
            config = dataclasses.replace(config, thermal_tyre_initial_pressure_psi_gauge=value)
    return scenario, config


def _checked_setup_value(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"setup dimension {name!r} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"setup dimension {name!r} must be finite, got {value!r}")
    return number


def proxy_responses(run: ScenarioRun) -> dict[str, float]:
    """Documented proxy metrics for a run with no track: final speed, settled ay."""
    last = run.record.ground_truth[-1]
    final_speed = math.hypot(last.vx_m_s, last.vy_m_s)
    steps = run.record.ground_truth
    window = max(1, round(_SETTLED_WINDOW_S / run.record.dt_s))
    tail = steps[-window:]
    settled_ay = math.fsum(step.ay_m_s2 for step in tail) / len(tail)
    return {"final_speed_m_s": final_speed, "settled_ay_m_s2": settled_ay}


@dataclass(frozen=True, slots=True)
class LapResponse:
    """First-lap timing of one run projected onto a track.

    ``sector_times_s`` holds the consecutive boundary-to-boundary durations from
    the stream start up to and including the first lap crossing; ``lap_time_s``
    is the stream-start-to-lap-line duration, or ``None`` when the stream
    contains no lap crossing (a partial lap still times the sectors it crossed).
    ``valid``/``failures`` are the
    :func:`~f1telemetry.testing.reference_lap.assess_scenario_lap` verdict.
    """

    sector_times_s: tuple[float, ...]
    lap_time_s: float | None
    valid: bool
    failures: tuple[str, ...]


def lap_responses(
    run: ScenarioRun,
    track: Track,
    *,
    wheelbase_m: float,
    axle_track_m: float,
) -> LapResponse:
    """Project one run onto ``track`` and time its first lap and sectors."""
    assessment = assess_scenario_lap(track, run, wheelbase_m=wheelbase_m, axle_track_m=axle_track_m)
    start_s = run.record.ground_truth[0].t_s
    crossings: list[float] = []
    saw_lap = False
    for event in assessment.events:
        crossings.append(event.time_s)
        if event.kind is TrackEventKind.LAP:
            saw_lap = True
            break
    boundaries = [start_s, *crossings]
    sector_times = tuple(later - earlier for earlier, later in itertools.pairwise(boundaries))
    return LapResponse(
        sector_times_s=sector_times,
        lap_time_s=(crossings[-1] - start_s) if saw_lap else None,
        valid=assessment.validity.valid,
        failures=assessment.validity.failures,
    )


def response_table(
    rows: Sequence[SweepRow],
    *,
    track: Track | None = None,
    wheelbase_m: float = 0.0,
    axle_track_m: float = 0.0,
) -> list[dict[str, object]]:
    """One response row per sweep row: setup params plus measured responses.

    Without a track the row carries the proxy metrics (``final_speed_m_s``,
    ``settled_ay_m_s2``); with a track it carries ``sector_1_s`` …,
    ``lap_time_s`` (``None`` without a completed lap) and ``valid``. A track run
    needs the car geometry the projection places wheels with.
    """
    table: list[dict[str, object]] = []
    for row in rows:
        entry: dict[str, object] = dict(row.params)
        if track is None:
            entry.update(proxy_responses(row.run))
        else:
            response = lap_responses(
                row.run, track, wheelbase_m=wheelbase_m, axle_track_m=axle_track_m
            )
            for index, duration in enumerate(response.sector_times_s, start=1):
                entry[f"sector_{index}_s"] = duration
            entry["lap_time_s"] = response.lap_time_s
            entry["valid"] = response.valid
            entry["failures"] = response.failures
        table.append(entry)
    return table


def run_design_points(
    base_scenario: Scenario,
    base_config: KernelConfig,
    box: SetupBox,
    samples: np.ndarray,
    manifest_factory: ManifestFactory | None = None,
    *,
    control_law: ControlLaw | None = None,
    max_brake_torque_nm: float = 0.0,
    control_steps: int = CONTROL_STEPS,
) -> tuple[SweepRow, ...]:
    """Simulate one run per design point; each point rides per-run setup overrides.

    ``samples`` is an ``(n, box.ndim)`` array in physical units — the output of
    :func:`latin_hypercube` — with columns in :attr:`SetupBox.names` order. Every
    sample must lie inside the box. ``manifest_factory(index, params)`` cites each
    run the way :func:`~f1telemetry.testing.sweep.run_sweep` does, or ``None``
    for uncited runs. ``control_law``/``max_brake_torque_nm`` drive closed-loop
    track runs; without them the scenario's own segments drive.
    """
    matrix = np.asarray(samples, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != box.ndim:
        raise ValueError(f"samples must be an (n, {box.ndim}) array, got shape {matrix.shape}")
    if matrix.shape[0] < 1:
        raise ValueError("samples must hold at least one design point")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("samples must be finite")
    low, high = box.bounds()
    if bool(np.any(matrix < low)) or bool(np.any(matrix > high)):
        raise ValueError("every design point must lie inside the setup box")
    rows: list[SweepRow] = []
    for index, sample in enumerate(matrix.tolist()):
        params = dict(zip(box.names, sample, strict=True))
        plan, config = apply_setup(base_scenario, base_config, params)
        manifest = manifest_factory(index, params) if manifest_factory is not None else None
        run = run_scenario(
            config,
            plan,
            control_steps=control_steps,
            control_law=control_law,
            max_brake_torque_nm=max_brake_torque_nm,
            manifest=manifest,
        )
        rows.append(SweepRow(params=params, run=run, manifest=manifest))
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class QuadraticSurrogate:
    """Least-squares quadratic response surface over a setup box.

    Coordinates are normalised to ``[-1, 1]`` per dimension before fitting, so
    the coefficients share a scale; :meth:`predict` takes physical units.
    Features are ``[1, xi, xi², xi·xj (i<j)]``.
    """

    response: str
    names: tuple[str, ...]
    low: tuple[float, ...]
    high: tuple[float, ...]
    coefficients: tuple[float, ...]

    @property
    def ndim(self) -> int:
        """Number of setup dimensions the surface spans."""
        return len(self.names)

    def _normalise(self, matrix: np.ndarray) -> np.ndarray:
        low = np.array(self.low, dtype=np.float64)
        high = np.array(self.high, dtype=np.float64)
        return 2.0 * (matrix - low) / (high - low) - 1.0

    def predict_matrix(self, samples: np.ndarray) -> np.ndarray:
        """Predict the response at each row of an ``(n, ndim)`` physical array."""
        matrix = np.asarray(samples, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.ndim:
            raise ValueError(f"samples must be an (n, {self.ndim}) array")
        return _quadratic_features(self._normalise(matrix)) @ np.array(
            self.coefficients, dtype=np.float64
        )

    def predict(self, point: Mapping[str, float]) -> float:
        """Predict the response at one named setup point."""
        try:
            ordered = [float(point[name]) for name in self.names]
        except KeyError as exc:
            raise KeyError(f"surrogate point is missing dimension {exc}") from exc
        return float(self.predict_matrix(np.array([ordered], dtype=np.float64))[0])


def _quadratic_features(unit: np.ndarray) -> np.ndarray:
    """``[1, xi, xi², xi·xj (i<j)]`` columns for ``[-1, 1]`` coordinates."""
    n_dims = unit.shape[1]
    columns = [np.ones(unit.shape[0], dtype=np.float64)]
    columns.extend(unit[:, dim] for dim in range(n_dims))
    columns.extend(unit[:, dim] ** 2 for dim in range(n_dims))
    for first, second in itertools.combinations(range(n_dims), 2):
        columns.append(unit[:, first] * unit[:, second])
    return np.column_stack(columns)


def n_quadratic_features(ndim: int) -> int:
    """Number of quadratic-surface coefficients over ``ndim`` dimensions."""
    if isinstance(ndim, bool) or not isinstance(ndim, int) or ndim < 1:
        raise ValueError(f"ndim must be an int >= 1, got {ndim!r}")
    return 1 + 2 * ndim + ndim * (ndim - 1) // 2


def fit_quadratic(
    box: SetupBox,
    samples: np.ndarray,
    values: np.ndarray | Sequence[float],
    response: str,
) -> QuadraticSurrogate:
    """Fit the quadratic surface through ``(samples, values)`` by least squares.

    ``samples`` is ``(n, box.ndim)`` in physical units, ``values`` the measured
    response at each row. Needs at least as many points as coefficients
    (:func:`n_quadratic_features`); fewer would fit noise exactly and report
    nothing about the surface between the points.
    """
    matrix = np.asarray(samples, dtype=np.float64)
    target = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[1] != box.ndim:
        raise ValueError(f"samples must be an (n, {box.ndim}) array")
    if target.shape != (matrix.shape[0],):
        raise ValueError("values must hold one response per sample row")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(target)):
        raise ValueError("samples and values must be finite")
    if not response:
        raise ValueError("response needs a nonempty name")
    n_needed = n_quadratic_features(box.ndim)
    if matrix.shape[0] < n_needed:
        raise ValueError(
            f"a {box.ndim}-D quadratic surface has {n_needed} coefficients "
            f"but only {matrix.shape[0]} sample(s) were supplied"
        )
    low, high = box.bounds()
    unit = 2.0 * (matrix - low) / (high - low) - 1.0
    coefficients, _, _, _ = np.linalg.lstsq(_quadratic_features(unit), target, rcond=None)
    return QuadraticSurrogate(
        response=response,
        names=box.names,
        low=tuple(low.tolist()),
        high=tuple(high.tolist()),
        coefficients=tuple(float(value) for value in coefficients),
    )


@dataclass(frozen=True, slots=True)
class FitScore:
    """Held-out quality of a fitted surface: R² and worst absolute error."""

    r2: float
    max_abs_err: float
    n: int


def score_fit(
    fit: QuadraticSurrogate, samples: np.ndarray, values: np.ndarray | Sequence[float]
) -> FitScore:
    """Score ``fit`` on held-out ``(samples, values)`` in response units.

    ``r2`` is ``1 - ss_res/ss_tot``; a constant held-out response has no variance
    to explain, so it scores ``1.0`` when predicted exactly and ``0.0`` otherwise.
    """
    matrix = np.asarray(samples, dtype=np.float64)
    target = np.asarray(list(values), dtype=np.float64).reshape(-1)
    predicted = fit.predict_matrix(matrix)
    residual = target - predicted
    max_abs_err = float(np.max(np.abs(residual))) if residual.size else 0.0
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((target - float(np.mean(target))) ** 2))
    r2 = (1.0 if ss_res == 0.0 else 0.0) if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return FitScore(r2=r2, max_abs_err=max_abs_err, n=int(target.size))


def predict_optimum(
    fit: QuadraticSurrogate,
    box: SetupBox,
    *,
    minimise: bool = True,
    seed: int = 0,
    n_candidates: int = 20000,
) -> tuple[dict[str, float], float]:
    """Predict the box optimum of the surrogate surface.

    Evaluates the surface on a seeded uniform candidate cloud plus every box
    corner and returns the best ``(point, predicted value)``. Seeded, so the
    prediction is repeatable; the re-simulation in the test owns the question of
    whether the surface optimum is the car's.
    """
    if fit.names != box.names:
        raise ValueError("the surrogate was fitted over a different setup box")
    if isinstance(n_candidates, bool) or not isinstance(n_candidates, int) or n_candidates < 1:
        raise ValueError(f"n_candidates must be an int >= 1, got {n_candidates!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an int, got {seed!r}")
    low, high = box.bounds()
    rng = np.random.default_rng(seed)
    cloud = low + rng.random((n_candidates, box.ndim)) * (high - low)
    corners = np.array(
        [list(corner) for corner in itertools.product(*zip(low, high, strict=True))],
        dtype=np.float64,
    )
    candidates = np.vstack([cloud, corners])
    predicted = fit.predict_matrix(candidates)
    best = int(np.argmin(predicted) if minimise else np.argmax(predicted))
    point = dict(zip(box.names, (float(value) for value in candidates[best]), strict=True))
    return point, float(predicted[best])
