from types import SimpleNamespace

import pytest

from f1telemetry.testing import scenario_lab, scenarios
from f1telemetry.testing.records import SensorFrame


def _plan(name: str) -> scenarios.Scenario:
    return scenarios.Scenario(
        name=name,
        initial_speed_m_s=20.0,
        segments=(
            scenarios.ScenarioSegment(0.1, throttle=0.25, steer_wheel_deg=2.0),
            scenarios.ScenarioSegment(0.2, throttle=1.0, brake_torque_nm=-100.0),
        ),
        description="test run",
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "scenario_run", "request_id": "x", "scenario_id": "arbitrary", "setting": 50},
        {"type": "scenario_run", "request_id": "x", "scenario_id": "launch", "setting": 101},
        {"type": "scenario_run", "request_id": "x", "scenario_id": "launch", "setting": True},
        {
            "type": "scenario_run",
            "request_id": "x",
            "scenario_id": "launch",
            "setting": float("nan"),
        },
        {"type": "scenario_run", "request_id": "x", "scenario_id": "launch", "setting": 10**1000},
        {
            "type": "scenario_run",
            "request_id": "x",
            "scenario_id": "launch",
            "setting": 80,
            "path": "file",
        },
    ],
)
def test_rejects_unbounded_scenario_requests(payload: object) -> None:
    with pytest.raises(scenario_lab.ScenarioLabError):
        scenario_lab.run_lab_request(payload)


@pytest.mark.parametrize(
    ("scenario_id", "setting", "scenario_name", "segment_index", "field", "expected"),
    [
        ("launch", 80, "accelerate_to_speed", 1, "throttle", 0.8),
        ("braking", 50, "braking", 1, "brake_torque_nm", -50.0),
        ("cornering", 150, "steady_state_circle", 0, "steer_wheel_deg", 3.0),
    ],
)
def test_runs_only_the_selected_bounded_adjustment(
    monkeypatch: pytest.MonkeyPatch,
    scenario_id: str,
    setting: float,
    scenario_name: str,
    segment_index: int,
    field: str,
    expected: float,
) -> None:
    plans = {
        "accelerate_to_speed": _plan("accelerate_to_speed"),
        "braking": _plan("braking"),
        "steady_state_circle": _plan("steady_state_circle"),
    }
    monkeypatch.setattr(
        scenario_lab,
        "load_car_spec",
        lambda: SimpleNamespace(kernel_config=lambda: object()),
    )
    monkeypatch.setattr(scenario_lab, "build_scenarios", lambda _config: plans)  # pyright: ignore[reportUnknownLambdaType]
    captured: dict[str, scenarios.Scenario] = {}

    def fake_run(_config: object, plan: scenarios.Scenario) -> object:
        captured["plan"] = plan
        record = SimpleNamespace(
            frames=(
                SensorFrame(
                    0.0,
                    {"speed": 0.0, "accel_longitudinal": 0.0, "accel_lateral": 0.0},
                ),
                SensorFrame(0.1, {"speed": 36.0, "accel_longitudinal": 2.0, "accel_lateral": 1.0}),
            )
        )
        return SimpleNamespace(
            record=record,
            description=plan.description,
            steps=10,
            dt_s=0.01,
        )

    monkeypatch.setattr(scenario_lab, "run_headless", fake_run)
    response = scenario_lab.run_lab_request(
        {
            "type": "scenario_run",
            "request_id": "abc-1",
            "scenario_id": scenario_id,
            "setting": setting,
        }
    )
    assert response["type"] == "scenario_result"
    assert response["request_id"] == "abc-1"
    points = response["points"]
    assert isinstance(points, list)
    assert len(points) == 2
    assert captured["plan"].name == scenario_name
    actual = captured["plan"].segments[segment_index]
    assert getattr(actual, field) == pytest.approx(expected)
