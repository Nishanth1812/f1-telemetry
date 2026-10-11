"""P10 full version: sector-time DoE over the setup box (PLAN.md §10 item 5).

The first slice (:mod:`f1telemetry.doe.design`) proved the machinery — seeded
Latin hypercubes, per-run setup application, quadratic response surfaces — on
proxy metrics and short open-loop runs. This module is the full version: the
same machinery pointed at real lap time. The responses are per-sector durations
of a full closed-loop coastal-loop reference-driver lap, timed by projecting
each run through
:func:`~f1telemetry.testing.reference_lap.assess_scenario_lap`, with one
quadratic surrogate per sector plus one for the lap total.

**Setup, not coefficients.** The box spans two garage knobs only —
``roll_stiffness_front_fraction`` and ``brake_bias_front`` — and both ride the
per-run overrides in :func:`~f1telemetry.doe.design.apply_setup`: a
:class:`~f1telemetry.testing.scenarios.Scenario` copy, a
:class:`~f1telemetry.contracts.car_spec.KernelConfig` copy, or both.
``car_spec.yaml`` is never edited, no coefficient is retuned, and the physics
is untouched. The closed-loop profile is the measured P2 capability
(~7.5 m/s²) from the phase 3–5 demo record, not a tuning input.

**Incomplete laps are skipped and reported, never fabricated.** A run whose
stream contains no completed valid lap contributes no sector times:
:func:`collect_sector_table` records its params and failures in
:attr:`SectorTable.skipped` and fits only completed laps.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast

import numpy as np

from f1telemetry.contracts.car_spec import KernelConfig
from f1telemetry.doe.design import (
    FitScore,
    QuadraticSurrogate,
    SetupBox,
    SetupDim,
    fit_quadratic,
    lap_responses,
    latin_hypercube,
    predict_optimum,
    score_fit,
)
from f1telemetry.racing_line import (
    LateralOffsetSolution,
    SpeedProfile,
    minimum_curvature_offsets,
    speed_profile,
)
from f1telemetry.testing.reference_lap import make_reference_driver
from f1telemetry.testing.scenarios import (
    ControlLaw,
    Scenario,
    ScenarioSegment,
)
from f1telemetry.testing.sweep import SweepRow
from f1telemetry.tracks import Track

__all__ = [
    "AXLE_TRACK_M",
    "DURATION_S",
    "GRIP_MARGIN",
    "HOLD_N",
    "HOLD_SEED",
    "LATERAL_ACCEL_M_S2",
    "LINE_MARGIN_M",
    "MAX_ACCEL_M_S2",
    "MAX_BRAKE_M_S2",
    "MAX_BRAKE_TORQUE_NM",
    "MAX_SPEED_M_S",
    "N_SECTORS",
    "OPTIMUM_SEED",
    "RESPONSE_NAMES",
    "SECTOR_BOX",
    "TRAIN_N",
    "TRAIN_SEED",
    "SectorTable",
    "SkippedLap",
    "base_plan",
    "collect_sector_table",
    "design_samples",
    "fit_sector_surrogates",
    "line_and_profile",
    "predict_lap_optimum",
    "reference_law",
    "score_sector_holds",
]

#: The 2-D setup box: suspension balance × brake balance, the two garage knobs
#: with a first-order effect on cornering balance and braking stability.
SECTOR_BOX: Final[SetupBox] = SetupBox(
    dims=(
        SetupDim("roll_stiffness_front_fraction", 0.5, 0.7),
        SetupDim("brake_bias_front", 0.5, 0.7),
    )
)

#: 8 train points fit the 6-coefficient 2-D quadratic with 2 spare degrees of
#: freedom; 4 holdout points score it.
TRAIN_N: Final[int] = 8
TRAIN_SEED: Final[int] = 41
HOLD_N: Final[int] = 4
HOLD_SEED: Final[int] = 97
#: Seeded surrogate-optimum search, so the prediction is repeatable.
OPTIMUM_SEED: Final[int] = 5

#: Closed-loop configuration, shared with the P4 reference-driver evidence:
#: the solved minimum-curvature line clipped this far inside the painted edge,
#: paced by the speed profile at the measured ~7.5 m/s² P2 capability.
LATERAL_ACCEL_M_S2: Final[float] = 7.5
MAX_SPEED_M_S: Final[float] = 60.0
MAX_ACCEL_M_S2: Final[float] = 8.0
MAX_BRAKE_M_S2: Final[float] = 25.0
LINE_MARGIN_M: Final[float] = 2.0
GRIP_MARGIN: Final[float] = 0.9
#: 115 s closes coastal_loop's ~88 s first lap with room for one more sector
#: crossing to confirm the sector stream.
DURATION_S: Final[float] = 115.0
MAX_BRAKE_TORQUE_NM: Final[float] = 9000.0
AXLE_TRACK_M: Final[float] = 2.0
INITIAL_GEAR: Final[int] = 4

#: Coastal_loop declares two sector boundaries plus the lap line: three sectors.
N_SECTORS: Final[int] = 3
#: One response per sector plus the lap total, fitted as four surrogates.
RESPONSE_NAMES: Final[tuple[str, ...]] = ("sector_1_s", "sector_2_s", "sector_3_s", "lap_time_s")


@dataclass(frozen=True, slots=True)
class SkippedLap:
    """A design point that contributed no timing: its params and why."""

    params: dict[str, float]
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SectorTable:
    """Completed-lap sector timing for one design: samples plus responses.

    ``samples`` is ``(n_complete, box.ndim)`` in physical units, columns in box
    order; ``values`` maps each name in :data:`RESPONSE_NAMES` to its measured
    durations in seconds. ``skipped`` reports every incomplete or invalid lap
    the table left out, so nothing is silently dropped and nothing is
    fabricated.
    """

    samples: np.ndarray
    values: dict[str, np.ndarray]
    skipped: tuple[SkippedLap, ...]

    @property
    def n_complete(self) -> int:
        """Number of design points that produced a completed valid lap."""
        return int(self.samples.shape[0])


def line_and_profile(track: Track) -> tuple[LateralOffsetSolution, SpeedProfile]:
    """Solve the minimum-curvature line and its 7.5 m/s² speed profile."""
    line = minimum_curvature_offsets(track, margin_m=LINE_MARGIN_M)
    profile = speed_profile(
        track,
        line,
        max_speed_m_s=MAX_SPEED_M_S,
        max_accel_m_s2=MAX_ACCEL_M_S2,
        max_brake_m_s2=MAX_BRAKE_M_S2,
        lateral_accel_m_s2=LATERAL_ACCEL_M_S2,
    )
    return line, profile


def reference_law(
    track: Track,
    line: LateralOffsetSolution,
    profile: SpeedProfile,
    config: KernelConfig,
) -> ControlLaw:
    """The P4 pure-pursuit reference driver over the solved line and profile."""
    return make_reference_driver(
        track,
        line,
        profile,
        wheelbase_m=config.wheelbase_m,
        steering_ratio=config.steering_ratio,
        grip_margin=GRIP_MARGIN,
    )


def base_plan(track: Track, line: LateralOffsetSolution, profile: SpeedProfile) -> Scenario:
    """The un-tuned closed-loop lap: start/finish pose on the solved line."""
    v0 = float(np.interp(0.0, profile.s_m, profile.speed_m_s, period=track.length_m))
    x0, y0 = track.point_at(0.0, float(line.lateral_m[0]))
    tangent_x, tangent_y = track.tangent_at(0.0)
    return Scenario(
        name="sector_doe_coastal_lap",
        initial_speed_m_s=v0,
        initial_gear=INITIAL_GEAR,
        upshift_at_shift_point=True,
        initial_x_m=x0,
        initial_y_m=y0,
        initial_heading_rad=math.atan2(tangent_y, tangent_x),
        description="Sector-time DoE: full closed-loop coastal_loop reference-driver lap.",
        segments=(ScenarioSegment(DURATION_S),),
    )


def design_samples() -> tuple[np.ndarray, np.ndarray]:
    """Seeded ``(train, holdout)`` LHS point sets over :data:`SECTOR_BOX`."""
    train = latin_hypercube(SECTOR_BOX, TRAIN_N, TRAIN_SEED)
    hold = latin_hypercube(SECTOR_BOX, HOLD_N, HOLD_SEED)
    return train, hold


def collect_sector_table(
    box: SetupBox,
    rows: Sequence[SweepRow],
    track: Track,
    config: KernelConfig,
) -> SectorTable:
    """Time each row's first lap and keep only completed valid laps.

    A row whose stream holds no lap crossing, fails validity, or crosses an
    unexpected number of sector boundaries lands in :attr:`SectorTable.skipped`
    with its params and failures; its sector times are never guessed.
    """
    samples: list[list[float]] = []
    columns: dict[str, list[float]] = {name: [] for name in RESPONSE_NAMES}
    skipped: list[SkippedLap] = []
    for row in rows:
        # Design points build params from float sample rows; the cast recovers
        # that construction type from SweepRow's wider Mapping[str, object].
        params = cast(dict[str, float], dict(row.params))
        response = lap_responses(
            row.run, track, wheelbase_m=config.wheelbase_m, axle_track_m=AXLE_TRACK_M
        )
        failures = response.failures
        if (
            response.lap_time_s is None
            or not response.valid
            or len(response.sector_times_s) != N_SECTORS
        ):
            if response.lap_time_s is not None and response.valid:
                failures = (*failures, f"expected_{N_SECTORS}_sectors")
            skipped.append(SkippedLap(params=params, failures=tuple(failures)))
            continue
        assert response.lap_time_s is not None
        samples.append([params[name] for name in box.names])
        for index, duration in enumerate(response.sector_times_s, start=1):
            columns[f"sector_{index}_s"].append(duration)
        columns["lap_time_s"].append(response.lap_time_s)
    table = SectorTable(
        samples=np.array(samples, dtype=np.float64).reshape(-1, box.ndim),
        values={name: np.array(entries, dtype=np.float64) for name, entries in columns.items()},
        skipped=tuple(skipped),
    )
    return table


def fit_sector_surrogates(box: SetupBox, table: SectorTable) -> dict[str, QuadraticSurrogate]:
    """Fit one quadratic surface per sector plus one for the lap total."""
    fits: dict[str, QuadraticSurrogate] = {}
    for name in RESPONSE_NAMES:
        fits[name] = fit_quadratic(box, table.samples, table.values[name], name)
    return fits


def score_sector_holds(
    fits: Mapping[str, QuadraticSurrogate], hold: SectorTable
) -> dict[str, FitScore]:
    """Held-out max-abs-error and R² per response, in seconds."""
    return {name: score_fit(fits[name], hold.samples, hold.values[name]) for name in RESPONSE_NAMES}


def predict_lap_optimum(
    lap_fit: QuadraticSurrogate, box: SetupBox, *, seed: int = OPTIMUM_SEED
) -> tuple[dict[str, float], float]:
    """Predict the box setup that minimises the lap-time surrogate surface."""
    return predict_optimum(lap_fit, box, minimise=True, seed=seed)
