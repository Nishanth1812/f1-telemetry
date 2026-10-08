"""P10 first slice: Latin-hypercube setup design + quadratic surrogate.

Covers ``f1telemetry.doe.design`` end to end: the sampler's stratification and
determinism, per-run setup application (setup only — ``car_spec.yaml`` is never
edited and the base objects are never mutated), response assembly from sweep rows
(proxies where no track run exists, sector times where one does), surrogate fit
quality on held-out points (max-abs-error + R²), and optimum prediction with a
re-simulation whose predicted-vs-actual error is published in the assertions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from f1telemetry.contracts.car_spec import CarSpec, KernelConfig
from f1telemetry.doe.design import (
    FitScore,
    SetupBox,
    SetupDim,
    apply_setup,
    fit_quadratic,
    lap_responses,
    latin_hypercube,
    n_quadratic_features,
    predict_optimum,
    proxy_responses,
    response_table,
    run_design_points,
    score_fit,
)
from f1telemetry.racing_line import minimum_curvature_offsets, speed_profile
from f1telemetry.testing.reference_lap import make_reference_driver
from f1telemetry.testing.run_manifest import RunManifest
from f1telemetry.testing.scenarios import Scenario, ScenarioSegment
from f1telemetry.testing.sweep import SweepRow, results_table
from f1telemetry.tracks import Track, load_track

_SETUP_HASH = "ab" * 32
_GIT_SHA = "cd" * 20

_LATERAL_BOX = SetupBox(
    dims=(
        SetupDim("roll_stiffness_front_fraction", 0.5, 0.7),
        SetupDim("front_static_camber_deg", -4.0, -2.0),
        SetupDim("cold_tyre_pressure_psi", 19.0, 25.0),
    )
)
_BRAKING_BOX = SetupBox(dims=(SetupDim("brake_bias_front", 0.5, 0.7),))

_LATERAL_TRAIN_SEED = 7
_LATERAL_TRAIN_N = 10
_LATERAL_HOLD_SEED = 99
_LATERAL_HOLD_N = 4
_BRAKING_TRAIN_SEED = 11
_BRAKING_TRAIN_N = 5
_BRAKING_HOLD_SEED = 23
_BRAKING_HOLD_N = 2


@pytest.fixture(scope="module")
def config(spec: CarSpec) -> KernelConfig:
    return spec.kernel_config()


@pytest.fixture(scope="module")
def circle(config: KernelConfig) -> Scenario:
    steer_wheel_deg = math.degrees(math.atan(config.wheelbase_m / 50.0)) * config.steering_ratio
    return Scenario(
        name="doe_circle",
        initial_speed_m_s=20.0,
        initial_gear=0,
        description="DoE lateral slice: 50 m neutral circle.",
        segments=(ScenarioSegment(1.0, steer_wheel_deg=steer_wheel_deg),),
    )


@pytest.fixture(scope="module")
def braking() -> Scenario:
    return Scenario(
        name="doe_braking",
        initial_speed_m_s=30.0,
        initial_gear=4,
        description="DoE braking slice: approach then a grip-limited stop.",
        segments=(
            ScenarioSegment(0.4, throttle=0.3),
            ScenarioSegment(2.0, brake_torque_nm=-8000.0),
        ),
    )


def _factory(index: int, _params: Mapping[str, float]) -> RunManifest:
    return RunManifest(
        seed=index,
        car_spec_version="test",
        scenario_version="1",
        setup_hash=_SETUP_HASH,
        git_sha=_GIT_SHA,
    )


def _column(rows: tuple[SweepRow, ...], response: str) -> np.ndarray:
    return np.array([proxy_responses(row.run)[response] for row in rows], dtype=np.float64)


def test_latin_hypercube_is_stratified_and_deterministic() -> None:
    samples = latin_hypercube(_LATERAL_BOX, 8, seed=4)

    assert samples.shape == (8, 3)
    low, high = _LATERAL_BOX.bounds()
    assert bool(np.all(samples >= low)) and bool(np.all(samples <= high))
    # Every stratum contributes exactly one sample per dimension.
    for dim, (lo, hi) in enumerate(zip(low.tolist(), high.tolist(), strict=True)):
        strata = np.floor((samples[:, dim] - lo) / (hi - lo) * 8).astype(int)
        assert sorted(strata.tolist()) == list(range(8))

    assert np.array_equal(samples, latin_hypercube(_LATERAL_BOX, 8, seed=4))
    assert not np.array_equal(samples, latin_hypercube(_LATERAL_BOX, 8, seed=5))


def test_setup_box_rejects_bad_declarations() -> None:
    with pytest.raises(ValueError, match="at least one dimension"):
        SetupBox(dims=())
    with pytest.raises(ValueError, match="unique dimension names"):
        SetupBox(
            dims=(SetupDim("roll_stiffness_front_fraction", 0.5, 0.7),) * 2,
        )
    with pytest.raises(ValueError, match="must be below high"):
        SetupDim("brake_bias_front", 0.7, 0.5)
    with pytest.raises(ValueError, match="n_samples"):
        latin_hypercube(_BRAKING_BOX, 0, seed=1)


def test_apply_setup_routes_each_dim_and_leaves_base_objects_alone(
    config: KernelConfig, circle: Scenario
) -> None:
    plan, tuned = apply_setup(
        circle,
        config,
        {
            "brake_bias_front": 0.62,
            "roll_stiffness_front_fraction": 0.68,
            "front_static_camber_deg": -2.5,
            "cold_tyre_pressure_psi": 24.0,
        },
    )

    assert plan.brake_bias == pytest.approx((0.31, 0.31, 0.19, 0.19))
    assert tuned.roll_stiffness_front_fraction == pytest.approx(0.68)
    assert tuned.axle_static_camber_deg[0] == pytest.approx(-2.5)
    assert tuned.axle_static_camber_deg[1] == pytest.approx(config.axle_static_camber_deg[1])
    assert tuned.thermal_tyre_initial_pressure_psi_gauge == pytest.approx(24.0)
    # The base objects are never mutated: setup rides per-run copies.
    assert circle.brake_bias == (1.0, 1.0, 1.0, 1.0)
    assert config.roll_stiffness_front_fraction == pytest.approx(0.6)
    assert config.axle_static_camber_deg[0] == pytest.approx(-3.0)
    assert config.thermal_tyre_initial_pressure_psi_gauge == pytest.approx(22.0)


def test_apply_setup_rejects_unknown_dims_and_out_of_range_values(
    config: KernelConfig, circle: Scenario
) -> None:
    # Wing angles have no setup path: the model carries no aero-balance input,
    # so a flap dimension would have nothing to act on short of a physics change.
    with pytest.raises(KeyError, match="fw_flap_deg"):
        apply_setup(circle, config, {"fw_flap_deg": 5.0})
    with pytest.raises(ValueError, match="brake_bias_front"):
        apply_setup(circle, config, {"brake_bias_front": 1.5})
    with pytest.raises(ValueError, match="roll_stiffness_front_fraction"):
        apply_setup(circle, config, {"roll_stiffness_front_fraction": 1.0})
    with pytest.raises(ValueError, match="cold_tyre_pressure_psi"):
        apply_setup(circle, config, {"cold_tyre_pressure_psi": -1.0})


def test_cold_pressure_moves_only_thermal_channels(config: KernelConfig, circle: Scenario) -> None:
    samples = np.array([[0.6, -3.0, 19.0], [0.6, -3.0, 25.0]])
    (cool, hot) = run_design_points(circle, config, _LATERAL_BOX, samples)

    cool_proxies = proxy_responses(cool.run)
    hot_proxies = proxy_responses(hot.run)
    assert cool_proxies == hot_proxies
    cool_psi = cool.run.record.frames[-1].values["tyre_pressure_fl"]
    hot_psi = hot.run.record.frames[-1].values["tyre_pressure_fl"]
    assert 5.9 < hot_psi - cool_psi < 6.1


def test_design_runs_are_deterministic_and_cited(config: KernelConfig, circle: Scenario) -> None:
    samples = latin_hypercube(_LATERAL_BOX, 2, seed=21)
    first = run_design_points(circle, config, _LATERAL_BOX, samples, _factory)
    second = run_design_points(circle, config, _LATERAL_BOX, samples, _factory)

    assert [row.manifest.seed if row.manifest is not None else -1 for row in first] == [0, 1]
    for lhs, rhs in zip(first, second, strict=True):
        assert lhs.run.trace.tobytes() == rhs.run.trace.tobytes()
        assert lhs.params == rhs.params
    assert response_table(first) == response_table(second)


def test_proxy_table_matches_sweep_results_table(config: KernelConfig, circle: Scenario) -> None:
    rows = run_design_points(circle, config, _LATERAL_BOX, latin_hypercube(_LATERAL_BOX, 2, 21))

    compact = results_table(rows)
    assembled = response_table(rows)
    assert len(assembled) == len(compact) == 2
    for lhs, rhs in zip(assembled, compact, strict=True):
        assert lhs["final_speed_m_s"] == rhs["final_speed_m_s"]
        assert lhs["settled_ay_m_s2"] == rhs["settled_ay_m_s2"]


@pytest.fixture(scope="module")
def lateral_fit(
    config: KernelConfig, circle: Scenario
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_samples = latin_hypercube(_LATERAL_BOX, _LATERAL_TRAIN_N, _LATERAL_TRAIN_SEED)
    hold_samples = latin_hypercube(_LATERAL_BOX, _LATERAL_HOLD_N, _LATERAL_HOLD_SEED)
    train = run_design_points(circle, config, _LATERAL_BOX, train_samples)
    hold = run_design_points(circle, config, _LATERAL_BOX, hold_samples)
    return (
        train_samples,
        _column(train, "settled_ay_m_s2"),
        hold_samples,
        _column(hold, "settled_ay_m_s2"),
    )


def test_lateral_surrogate_holdout_quality(
    lateral_fit: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    train_samples, train_y, hold_samples, hold_y = lateral_fit
    assert float(np.ptp(train_y)) > 0.2  # the box must actually move the response
    fit = fit_quadratic(_LATERAL_BOX, train_samples, train_y, "settled_ay_m_s2")
    assert len(fit.coefficients) == n_quadratic_features(3)

    scored: FitScore = score_fit(fit, hold_samples, hold_y)
    assert scored.r2 >= 0.99, f"lateral holdout R² {scored.r2:.4f} on {scored.n} point(s)"
    assert scored.max_abs_err <= 0.05, (
        f"lateral holdout max-abs-error {scored.max_abs_err:.4f} m/s² on {scored.n} point(s)"
    )


def test_lateral_optimum_resimulation_error_is_published(
    config: KernelConfig,
    circle: Scenario,
    lateral_fit: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    train_samples, train_y, _, _ = lateral_fit
    fit = fit_quadratic(_LATERAL_BOX, train_samples, train_y, "settled_ay_m_s2")

    point, predicted = predict_optimum(fit, _LATERAL_BOX, minimise=False, seed=3)
    low, high = _LATERAL_BOX.bounds()
    for name, lo, hi in zip(_LATERAL_BOX.names, low.tolist(), high.tolist(), strict=True):
        assert lo <= point[name] <= hi

    (resim,) = run_design_points(
        circle, config, _LATERAL_BOX, np.array([[point[name] for name in _LATERAL_BOX.names]])
    )
    actual = proxy_responses(resim.run)["settled_ay_m_s2"]
    err = abs(predicted - actual)
    assert err <= 0.10, (
        f"lateral optimum predicted settled_ay {predicted:.4f} m/s² "
        f"but re-simulation measured {actual:.4f} m/s² (err {err:.4f} m/s²)"
    )


@pytest.fixture(scope="module")
def braking_fit(
    config: KernelConfig, braking: Scenario
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_samples = latin_hypercube(_BRAKING_BOX, _BRAKING_TRAIN_N, _BRAKING_TRAIN_SEED)
    hold_samples = latin_hypercube(_BRAKING_BOX, _BRAKING_HOLD_N, _BRAKING_HOLD_SEED)
    train = run_design_points(braking, config, _BRAKING_BOX, train_samples)
    hold = run_design_points(braking, config, _BRAKING_BOX, hold_samples)
    return (
        train_samples,
        _column(train, "final_speed_m_s"),
        hold_samples,
        _column(hold, "final_speed_m_s"),
    )


def test_braking_surrogate_holdout_quality_and_optimum(
    config: KernelConfig,
    braking: Scenario,
    braking_fit: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    train_samples, train_y, hold_samples, hold_y = braking_fit
    fit = fit_quadratic(_BRAKING_BOX, train_samples, train_y, "final_speed_m_s")

    scored = score_fit(fit, hold_samples, hold_y)
    assert scored.r2 >= 0.95, f"braking holdout R² {scored.r2:.4f} on {scored.n} point(s)"
    assert scored.max_abs_err <= 0.10, (
        f"braking holdout max-abs-error {scored.max_abs_err:.4f} m/s² on {scored.n} point(s)"
    )

    point, predicted = predict_optimum(fit, _BRAKING_BOX, minimise=True, seed=5)
    (resim,) = run_design_points(
        braking, config, _BRAKING_BOX, np.array([[point["brake_bias_front"]]])
    )
    actual = proxy_responses(resim.run)["final_speed_m_s"]
    err = abs(predicted - actual)
    assert err <= 0.50, (
        f"braking optimum predicted final speed {predicted:.4f} m/s "
        f"but re-simulation measured {actual:.4f} m/s (err {err:.4f} m/s)"
    )


def test_quadratic_surface_recovers_a_known_surface_and_rejects_bad_inputs() -> None:
    box = SetupBox(dims=(SetupDim("x1", -1.0, 1.0), SetupDim("x2", -1.0, 1.0)))
    samples = latin_hypercube(box, 12, seed=2)
    values = (
        1.0
        + 2.0 * samples[:, 0]
        + 3.0 * samples[:, 0] ** 2
        - samples[:, 1]
        + 0.5 * samples[:, 0] * samples[:, 1]
    )

    fit = fit_quadratic(box, samples, values, "synthetic")
    assert len(fit.coefficients) == n_quadratic_features(2) == 6
    np.testing.assert_allclose(fit.predict_matrix(samples), values, rtol=0, atol=1e-8)
    assert fit.predict({"x1": 0.5, "x2": -0.25}) == pytest.approx(
        1.0 + 2.0 * 0.5 + 3.0 * 0.25 + 0.25 + 0.5 * 0.5 * -0.25
    )

    scored = score_fit(fit, samples, values)
    assert scored.n == 12
    assert scored.r2 == pytest.approx(1.0)
    assert scored.max_abs_err == pytest.approx(0.0, abs=1e-8)
    constant = score_fit(fit, samples[:2], np.full(2, 3.0))
    assert constant.r2 == 0.0

    with pytest.raises(KeyError, match="x2"):
        fit.predict({"x1": 0.0})
    with pytest.raises(ValueError, match="coefficients"):
        fit_quadratic(box, samples[:4], values[:4], "synthetic")
    with pytest.raises(ValueError, match="different setup box"):
        predict_optimum(fit, _BRAKING_BOX)
    with pytest.raises(ValueError, match="n_candidates"):
        predict_optimum(fit, box, n_candidates=0)


def test_run_design_points_rejects_out_of_box_samples(
    config: KernelConfig, braking: Scenario
) -> None:
    with pytest.raises(ValueError, match="inside the setup box"):
        run_design_points(braking, config, _BRAKING_BOX, np.array([[0.9]]))
    with pytest.raises(ValueError, match=r"\(n, 1\)"):
        run_design_points(braking, config, _BRAKING_BOX, np.array([0.6]))


@pytest.fixture(scope="module")
def coastal_track() -> Track:
    return load_track(Path(__file__).parents[1] / "tracks" / "coastal_loop.yaml")


@pytest.fixture(scope="module")
def coastal_partial_run(config: KernelConfig, coastal_track: Track) -> tuple[Track, SweepRow]:
    line = minimum_curvature_offsets(coastal_track, margin_m=2.0)
    profile = speed_profile(
        coastal_track,
        line,
        max_speed_m_s=60.0,
        max_accel_m_s2=8.0,
        max_brake_m_s2=25.0,
        lateral_accel_m_s2=7.5,
    )
    assert line.converged and profile.converged
    v0 = float(np.interp(0.0, profile.s_m, profile.speed_m_s, period=coastal_track.length_m))
    x0, y0 = coastal_track.point_at(0.0, float(line.lateral_m[0]))
    tangent_x, tangent_y = coastal_track.tangent_at(0.0)
    law = make_reference_driver(
        coastal_track,
        line,
        profile,
        wheelbase_m=config.wheelbase_m,
        steering_ratio=config.steering_ratio,
        grip_margin=0.9,
    )
    plan = Scenario(
        name="doe_coastal_partial",
        initial_speed_m_s=v0,
        initial_gear=4,
        upshift_at_shift_point=True,
        initial_x_m=x0,
        initial_y_m=y0,
        initial_heading_rad=math.atan2(tangent_y, tangent_x),
        description="DoE track slice: 45 s of the reference lap (sector 1 only).",
        segments=(ScenarioSegment(45.0),),
    )
    (row,) = run_design_points(
        plan,
        config,
        _LATERAL_BOX,
        np.array([[0.6, -3.0, 22.0]]),
        control_law=law,
        max_brake_torque_nm=9000.0,
    )
    return coastal_track, row


def test_track_run_reports_sector_times_without_a_completed_lap(
    config: KernelConfig, coastal_partial_run: tuple[Track, SweepRow]
) -> None:
    track, row = coastal_partial_run
    response = lap_responses(row.run, track, wheelbase_m=config.wheelbase_m, axle_track_m=2.0)

    assert len(response.sector_times_s) == 1
    assert 24.0 < response.sector_times_s[0] < 27.0
    assert response.lap_time_s is None
    assert response.valid is False
    assert "incomplete_lap" in response.failures

    (entry,) = response_table((row,), track=track, wheelbase_m=config.wheelbase_m, axle_track_m=2.0)
    assert entry["sector_1_s"] == response.sector_times_s[0]
    assert entry["lap_time_s"] is None
    assert entry["valid"] is False
