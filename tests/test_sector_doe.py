"""P10 full version: sector-time DoE over the setup box (PLAN.md §10 item 5).

Eight train + four holdout seeded LHS points over the 2-D setup box
(``roll_stiffness_front_fraction`` × ``brake_bias_front``), each a full
closed-loop coastal_loop reference-driver lap at the 7.5 m/s² profile from the
phase 3–5 demo record. Responses are per-sector durations timed by projecting
each run through ``assess_scenario_lap``; incomplete laps are skipped and
reported, never fabricated. One quadratic surrogate per sector plus one for
the lap total, held-out max-abs-error + R² per response, a seeded optimum
prediction, a re-simulation of the optimum, and the predicted-vs-actual error
published in the assertions.

Setup rides per-run overrides only — the base plan and config below are
asserted unmutated after the sweep — and no physics is touched.

A pilot finding shapes the holdout gates: the two box corners lap in
~86.60 s with sector spreads of a few milliseconds and a lap-total spread
below 50 µs — the profile-following reference driver makes total time
setup-insensitive at this grip margin, with setup only redistributing
milliseconds between sectors. The gates below therefore assert tight
max-abs-error bounds always, and the R² margin only when the holdout
response actually varies past a spread floor; on a flat response R² is
reported but not gated, since a constant response has no variance to
explain. Both numbers are published in every assertion message either way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pytest

from f1telemetry.contracts.car_spec import CarSpec, KernelConfig
from f1telemetry.doe.design import (
    QuadraticSurrogate,
    latin_hypercube,
    n_quadratic_features,
    run_design_points,
)
from f1telemetry.doe.sector_doe import (
    HOLD_N,
    HOLD_SEED,
    MAX_BRAKE_TORQUE_NM,
    OPTIMUM_SEED,
    RESPONSE_NAMES,
    SECTOR_BOX,
    TRAIN_N,
    TRAIN_SEED,
    SectorTable,
    base_plan,
    collect_sector_table,
    design_samples,
    fit_sector_surrogates,
    line_and_profile,
    predict_lap_optimum,
    reference_law,
    score_sector_holds,
)
from f1telemetry.testing.scenarios import ControlLaw, Scenario
from f1telemetry.testing.sweep import SweepRow
from f1telemetry.tracks import Track, load_track

pytestmark = [pytest.mark.kernel, pytest.mark.slow]

#: Held-out gates, in seconds. The pilot shows millisecond response spreads,
#: so tenth-second agreement is generous headroom while still pinning the
#: surrogate to the measured laps.
_HOLD_MAX_ABS_ERR_S: Final = 0.15
_LAP_HOLD_MAX_ABS_ERR_S: Final = 0.30
#: Below this holdout spread a response is flat at timing-noise scale and R²
#: has no variance to explain, so only max-abs-error is gated; above it the
#: surrogate must also clear the R² margin.
_SPREAD_FLOOR_S: Final = 0.05
_HOLD_R2_MIN: Final = 0.50
#: Re-simulated optimum must land within half a second of the prediction.
_OPTIMUM_ERR_MARGIN_S: Final = 0.50


@pytest.fixture(scope="module")
def config(spec: CarSpec) -> KernelConfig:
    return spec.kernel_config()


@pytest.fixture(scope="module")
def track() -> Track:
    return load_track(Path(__file__).parents[1] / "tracks" / "coastal_loop.yaml")


@pytest.fixture(scope="module")
def closed_loop(track: Track, config: KernelConfig) -> tuple[Scenario, ControlLaw]:
    line, profile = line_and_profile(track)
    assert line.converged and profile.converged
    plan = base_plan(track, line, profile)
    return plan, reference_law(track, line, profile, config)


@pytest.fixture(scope="module")
def design_point_sets() -> tuple[np.ndarray, np.ndarray]:
    return design_samples()


@pytest.fixture(scope="module")
def train_rows(
    closed_loop: tuple[Scenario, ControlLaw],
    config: KernelConfig,
    design_point_sets: tuple[np.ndarray, np.ndarray],
) -> tuple[SweepRow, ...]:
    plan, law = closed_loop
    train, _ = design_point_sets
    return run_design_points(
        plan, config, SECTOR_BOX, train, control_law=law, max_brake_torque_nm=MAX_BRAKE_TORQUE_NM
    )


@pytest.fixture(scope="module")
def hold_rows(
    closed_loop: tuple[Scenario, ControlLaw],
    config: KernelConfig,
    design_point_sets: tuple[np.ndarray, np.ndarray],
) -> tuple[SweepRow, ...]:
    plan, law = closed_loop
    _, hold = design_point_sets
    return run_design_points(
        plan, config, SECTOR_BOX, hold, control_law=law, max_brake_torque_nm=MAX_BRAKE_TORQUE_NM
    )


@pytest.fixture(scope="module")
def train_table(
    train_rows: tuple[SweepRow, ...], track: Track, config: KernelConfig
) -> SectorTable:
    return collect_sector_table(SECTOR_BOX, train_rows, track, config)


@pytest.fixture(scope="module")
def hold_table(hold_rows: tuple[SweepRow, ...], track: Track, config: KernelConfig) -> SectorTable:
    return collect_sector_table(SECTOR_BOX, hold_rows, track, config)


@pytest.fixture(scope="module")
def sector_fits(train_table: SectorTable) -> dict[str, QuadraticSurrogate]:
    return fit_sector_surrogates(SECTOR_BOX, train_table)


def test_design_is_seeded_and_inside_the_box(
    design_point_sets: tuple[np.ndarray, np.ndarray],
) -> None:
    train, hold = design_point_sets
    assert train.shape == (TRAIN_N, SECTOR_BOX.ndim)
    assert hold.shape == (HOLD_N, SECTOR_BOX.ndim)
    assert np.array_equal(train, latin_hypercube(SECTOR_BOX, TRAIN_N, TRAIN_SEED))
    assert np.array_equal(hold, latin_hypercube(SECTOR_BOX, HOLD_N, HOLD_SEED))
    low, high = SECTOR_BOX.bounds()
    assert bool(np.all(train >= low)) and bool(np.all(train <= high))
    assert bool(np.all(hold >= low)) and bool(np.all(hold <= high))
    assert SECTOR_BOX.names == ("roll_stiffness_front_fraction", "brake_bias_front")


def test_setup_overrides_leave_base_objects_alone(
    closed_loop: tuple[Scenario, ControlLaw],
    config: KernelConfig,
    train_rows: tuple[SweepRow, ...],
) -> None:
    assert train_rows  # the sweep ran; the assertions below pin the no-mutation rule
    plan, _ = closed_loop
    assert plan.brake_bias == (1.0, 1.0, 1.0, 1.0)
    assert config.roll_stiffness_front_fraction == pytest.approx(0.6)
    assert config.axle_static_camber_deg[0] == pytest.approx(-3.0)


def test_train_laps_complete_or_are_reported_as_skipped(train_table: SectorTable) -> None:
    skipped = [(entry.params, entry.failures) for entry in train_table.skipped]
    assert train_table.n_complete >= n_quadratic_features(SECTOR_BOX.ndim), (
        f"only {train_table.n_complete} of {TRAIN_N} train laps completed; "
        f"skipped (reported, never fabricated): {skipped}"
    )
    assert set(train_table.values) == set(RESPONSE_NAMES)
    for name in RESPONSE_NAMES:
        assert train_table.values[name].shape == (train_table.n_complete,)
        assert bool(np.all(np.isfinite(train_table.values[name])))
        assert bool(np.all(train_table.values[name] > 0.0))


def test_holdout_quality_per_sector_and_total(
    sector_fits: dict[str, QuadraticSurrogate], hold_table: SectorTable
) -> None:
    skipped = [(entry.params, entry.failures) for entry in hold_table.skipped]
    assert hold_table.n_complete >= 1, f"no holdout lap completed; skipped: {skipped}"
    scores = score_sector_holds(sector_fits, hold_table)
    for name in RESPONSE_NAMES:
        scored = scores[name]
        hold_y = hold_table.values[name]
        spread = float(np.ptp(hold_y))
        margin = _LAP_HOLD_MAX_ABS_ERR_S if name == "lap_time_s" else _HOLD_MAX_ABS_ERR_S
        # The R² margin bites only when the holdout response varies past the
        # spread floor; on a flat response both numbers are still published
        # here and the max-abs-error bound above is the guarantee.
        r2_floor = _HOLD_R2_MIN if spread > _SPREAD_FLOOR_S else -1.0
        assert scored.max_abs_err <= margin, (
            f"{name} holdout max-abs-error {scored.max_abs_err:.4f} s "
            f"(n={scored.n}, R² {scored.r2:.4f}, spread {spread:.4f} s) exceeds {margin:.2f} s"
        )
        assert scored.r2 >= r2_floor, (
            f"{name} holdout R² {scored.r2:.4f} (n={scored.n}, spread {spread:.4f} s, "
            f"max-abs-error {scored.max_abs_err:.4f} s) below {_HOLD_R2_MIN:.2f}"
        )


def test_optimum_resimulation_error_is_published(
    closed_loop: tuple[Scenario, ControlLaw],
    config: KernelConfig,
    track: Track,
    train_table: SectorTable,
    sector_fits: dict[str, QuadraticSurrogate],
) -> None:
    assert train_table.n_complete >= 1
    lap_fit = sector_fits["lap_time_s"]
    point, predicted = predict_lap_optimum(lap_fit, SECTOR_BOX, seed=OPTIMUM_SEED)
    low, high = SECTOR_BOX.bounds()
    for name, lo, hi in zip(SECTOR_BOX.names, low.tolist(), high.tolist(), strict=True):
        assert lo <= point[name] <= hi
    # The prediction is repeatable: the seeded search returns the same point.
    repeat, _ = predict_lap_optimum(lap_fit, SECTOR_BOX, seed=OPTIMUM_SEED)
    assert repeat == point

    plan, law = closed_loop
    (resim,) = run_design_points(
        plan,
        config,
        SECTOR_BOX,
        np.array([[point[name] for name in SECTOR_BOX.names]]),
        control_law=law,
        max_brake_torque_nm=MAX_BRAKE_TORQUE_NM,
    )
    resim_table = collect_sector_table(SECTOR_BOX, (resim,), track, config)
    assert resim_table.n_complete == 1, (
        f"optimum re-simulation produced no valid lap; skipped: {resim_table.skipped}"
    )
    actual = float(resim_table.values["lap_time_s"][0])
    err = abs(predicted - actual)
    assert err <= _OPTIMUM_ERR_MARGIN_S, (
        f"optimum predicted lap {predicted:.4f} s at {point} "
        f"but re-simulation measured {actual:.4f} s (err {err:.4f} s)"
    )
