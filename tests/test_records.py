"""P2 truth fields on records, with the P1 constructor and immutable update contract intact."""

from __future__ import annotations

from f1telemetry.testing.records import GroundTruthStep, SampleRecord, WheelTruth, with_step


def _legacy_step() -> GroundTruthStep:
    wheel = WheelTruth(3_000.0, 100.0, 0.0, 1.7, 0.02, 0.0, 0.0)
    return GroundTruthStep(
        0.01,
        20.0,
        0.0,
        1.0,
        0.0,
        0.0,
        3,
        1.0,
        80.0,
        100_000.0,
        0.0,
        1_000.0,
        2_000.0,
        0.0,
        0.001,
        (wheel, wheel, wheel, wheel),
    )


def _record() -> SampleRecord:
    return SampleRecord("record-test", 0.01, "test record", (_legacy_step(),), ())


def test_p2_fields_default_without_changing_legacy_positional_construction() -> None:
    step = _legacy_step()

    assert step.t_s == 0.01
    assert step.energy_residual_fraction == 0.001
    assert len(step.wheels) == 4
    assert step.yaw_rate_rad_s == 0.0
    assert step.roll_rad == 0.0
    assert step.pitch_rad == 0.0
    assert step.heave_m == 0.0
    assert step.suspension_travel_m == (0.0, 0.0, 0.0, 0.0)
    assert step.travel_limited == (False, False, False, False)


def test_with_step_replaces_every_p2_field_immutably() -> None:
    original = _record()
    travel = (0.001, 0.002, -0.001, -0.002)
    limited = (False, True, False, True)

    updated = with_step(
        original,
        0,
        yaw_rate_rad_s=0.25,
        roll_rad=-0.04,
        pitch_rad=0.01,
        heave_m=0.003,
        suspension_travel_m=travel,
        travel_limited=limited,
    )

    assert updated.ground_truth[0].yaw_rate_rad_s == 0.25
    assert updated.ground_truth[0].roll_rad == -0.04
    assert updated.ground_truth[0].pitch_rad == 0.01
    assert updated.ground_truth[0].heave_m == 0.003
    assert updated.ground_truth[0].suspension_travel_m == travel
    assert updated.ground_truth[0].travel_limited == limited
    assert original.ground_truth[0].yaw_rate_rad_s == 0.0
    assert original.ground_truth[0].suspension_travel_m == (0.0, 0.0, 0.0, 0.0)


def test_with_step_preserves_p2_fields_when_they_are_omitted() -> None:
    initial = with_step(
        _record(),
        0,
        yaw_rate_rad_s=0.5,
        roll_rad=0.02,
        pitch_rad=-0.01,
        heave_m=0.004,
        suspension_travel_m=(0.001, 0.002, 0.003, 0.004),
        travel_limited=(True, False, True, False),
    )

    updated = with_step(initial, 0, vx_m_s=22.0)

    assert updated.ground_truth[0].vx_m_s == 22.0
    assert updated.ground_truth[0].yaw_rate_rad_s == 0.5
    assert updated.ground_truth[0].roll_rad == 0.02
    assert updated.ground_truth[0].pitch_rad == -0.01
    assert updated.ground_truth[0].heave_m == 0.004
    assert updated.ground_truth[0].suspension_travel_m == (0.001, 0.002, 0.003, 0.004)
    assert updated.ground_truth[0].travel_limited == (True, False, True, False)
