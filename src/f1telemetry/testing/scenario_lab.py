"""Small, bounded request boundary for the local dashboard's Scenario Lab."""

from __future__ import annotations

import math
import re
from dataclasses import replace
from typing import Final

from f1telemetry.contracts.car_spec import load_car_spec
from f1telemetry.testing.headless import run_headless
from f1telemetry.testing.scenarios import Scenario, ScenarioRun, build_scenarios

__all__ = ["ScenarioLabError", "run_lab_request"]

_REQUEST_ID: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9._-]{1,64}\Z")
_POINT_LIMIT: Final[int] = 120
_SCENARIOS: Final[dict[str, tuple[str, str, str, float, float]]] = {
    "launch": (
        "accelerate_to_speed",
        "Standing launch",
        "Throttle",
        50.0,
        100.0,
    ),
    "braking": ("braking", "Braking run", "Brake", 25.0, 100.0),
    "cornering": (
        "steady_state_circle",
        "Constant-radius corner",
        "Steering",
        50.0,
        150.0,
    ),
}


class ScenarioLabError(ValueError):
    """A request does not match the small Scenario Lab contract."""


def _read_request(value: object) -> tuple[str, str, float]:
    if not isinstance(value, dict):
        raise ScenarioLabError("Request must be a JSON object.")
    if set(value) != {"type", "request_id", "scenario_id", "setting"}:
        raise ScenarioLabError(
            "Request must include type, request_id, scenario_id, and setting only."
        )
    request_id = value.get("request_id")
    if not isinstance(request_id, str) or _REQUEST_ID.fullmatch(request_id) is None:
        raise ScenarioLabError(
            "Request ID must be 1–64 letters, numbers, dots, dashes, or underscores."
        )
    if value.get("type") != "scenario_run":
        raise ScenarioLabError("Unknown Scenario Lab request type.")
    scenario_id = value.get("scenario_id")
    if not isinstance(scenario_id, str) or scenario_id not in _SCENARIOS:
        raise ScenarioLabError("Choose launch, braking, or cornering.")
    setting = value.get("setting")
    if isinstance(setting, bool) or not isinstance(setting, (int, float)):
        raise ScenarioLabError("Setting must be a finite number.")
    try:
        finite = math.isfinite(setting)
    except OverflowError:
        finite = False
    if not finite:
        raise ScenarioLabError("Setting must be a finite number.")
    _, _, _, minimum, maximum = _SCENARIOS[scenario_id]
    if not minimum <= setting <= maximum:
        raise ScenarioLabError(f"Setting must be between {minimum:g} and {maximum:g}.")
    return request_id, scenario_id, float(setting)


def _tuned_plan(scenarios: dict[str, Scenario], scenario_id: str, setting: float) -> Scenario:
    name, _, _, _, _ = _SCENARIOS[scenario_id]
    plan = scenarios[name]
    segments = list(plan.segments)
    if scenario_id == "launch":
        segments[1] = replace(segments[1], throttle=setting / 100.0)
    elif scenario_id == "braking":
        segments[1] = replace(
            segments[1], brake_torque_nm=segments[1].brake_torque_nm * setting / 100.0
        )
    else:
        steer = segments[0].steer_wheel_deg * setting / 100.0
        segments[0] = replace(segments[0], steer_wheel_deg=steer)
    return replace(plan, segments=tuple(segments))


def _result_points(run: ScenarioRun) -> list[dict[str, float]]:
    frames = run.record.frames
    if not frames:
        return []
    point_count = min(len(frames), _POINT_LIMIT)
    indices = (
        [round(index * (len(frames) - 1) / (point_count - 1)) for index in range(point_count)]
        if point_count > 1
        else [0]
    )
    points: list[dict[str, float]] = []
    for index in indices:
        frame = frames[index]
        points.append(
            {
                "time_s": frame.t_s,
                "speed_kmh": frame.values["speed"],
                "accel_longitudinal_m_s2": frame.values["accel_longitudinal"],
                "accel_lateral_m_s2": frame.values["accel_lateral"],
            }
        )
    return points


def run_lab_request(value: object) -> dict[str, object]:
    """Validate and run one supported Scenario Lab experiment."""
    request_id, scenario_id, setting = _read_request(value)
    config = load_car_spec().kernel_config()
    _, label, setting_name, _, _ = _SCENARIOS[scenario_id]
    plan = _tuned_plan(dict(build_scenarios(config)), scenario_id, setting)
    run = run_headless(config, plan)
    points = _result_points(run)
    if not points:
        raise ScenarioLabError("The scenario produced no samples.")
    frames = run.record.frames
    longitudinal_g = max(abs(frame.values["accel_longitudinal"]) for frame in frames) / 9.80665
    lateral_g = max(abs(frame.values["accel_lateral"]) for frame in frames) / 9.80665
    peak_speed = max(frame.values["speed"] for frame in frames)
    return {
        "type": "scenario_result",
        "request_id": request_id,
        "scenario_id": scenario_id,
        "title": label,
        "description": run.description,
        "setting_name": setting_name,
        "setting": setting,
        "duration_s": run.steps * run.dt_s,
        "peak_speed_kmh": peak_speed,
        "peak_longitudinal_g": longitudinal_g,
        "peak_lateral_g": lateral_g,
        "simulation_label": (
            "Uncalibrated simulation · results are for learning, not real-car prediction."
        ),
        "points": points,
    }
