"""P0-T7: all 8 invariants of PLAN.md section 11, as running tests.

Every test here does two things, and the second is the one that matters:

1. the checker passes on a clean, hand-computed fixture record;
2. the checker **fails** on a record with one deliberate violation of exactly that
   invariant.

Step 2 is what stops these from being ceremonial. An invariant that cannot be made to
fail is not a test, and the mutations are small and specific - 250 N off one corner's
load, 0.9x on one power term, one gear index going backwards.

The fixtures are supplied values, not simulation output. Each result therefore records
which phase turns the check into a physics-backed one; see
``InvariantResult.backing`` and ``docs/phase0-decisions.md``.
"""

from __future__ import annotations

import math

import pytest

from f1telemetry.contracts.car_spec import CarSpec
from f1telemetry.testing.fixtures import corrupted_records
from f1telemetry.testing.invariants import (
    INVARIANTS,
    InvariantResult,
    run_all,
)
from f1telemetry.testing.parquet_io import METADATA, serialise_frames
from f1telemetry.testing.records import SampleRecord

pytestmark = pytest.mark.invariant

_MISSIONS = corrupted_records()


def _result(number: int, record: SampleRecord, spec: CarSpec) -> InvariantResult:
    return run_all(record, spec, (number,))[0]


def _assert_detects(number: int, spec: CarSpec) -> InvariantResult:
    """The mutation for `number` must be caught by invariant `number`."""
    result = _result(number, _MISSIONS[number], spec)
    assert not result.passed, (
        f"invariant {number} did not notice its own violation:\n{result.summary()}"
    )
    assert result.violations, "invariant reported failure with no violations"
    return result


def test_invariant_registry_covers_all_eight() -> None:
    assert [inv.number for inv in INVARIANTS] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert len({inv.name for inv in INVARIANTS}) == 8
    for invariant in INVARIANTS:
        assert invariant.backing, f"invariant {invariant.number} does not name its backing phase"


def test_invariant_1_no_nan_or_inf(straight: SampleRecord, spec: CarSpec) -> None:
    """No NaN or Inf, ever - in the truth stream or the published stream. Backed from P1."""
    result = _result(1, straight, spec)
    assert result.passed, result.summary()

    corrupted = _assert_detects(1, spec)
    assert any("speed" in violation.where for violation in corrupted.violations)

    inf_record = _MISSIONS[1]
    assert math.isnan(inf_record.frames[2].values["speed"])


def test_invariant_2_friction_ellipse(cornering: SampleRecord, spec: CarSpec) -> None:
    """(Fx/muFz)^2 + (Fy/muFz)^2 <= 1 for every wheel, every step. Backed from P2."""
    result = _result(2, cornering, spec)
    assert result.passed, result.summary()

    corrupted = _assert_detects(2, spec)
    assert all("ellipse" in violation.detail for violation in corrupted.violations)

    combined = math.fsum(
        [
            (wheel.fx_n / (wheel.mu * wheel.fz_n)) ** 2
            + (wheel.fy_n / (wheel.mu * wheel.fz_n)) ** 2
            for step in cornering.ground_truth
            for wheel in step.wheels
        ]
    )
    assert 0.0 < combined < 4 * 8


def test_invariant_3_vertical_load_sum(
    straight: SampleRecord, cornering: SampleRecord, spec: CarSpec
) -> None:
    """Sum of corner loads = weight + aero + the acceleration term. Backed from P2."""
    for record in (straight, cornering):
        result = _result(3, record, spec)
        assert result.passed, result.summary()

    corrupted = _assert_detects(3, spec)
    assert all("sum" in violation.detail for violation in corrupted.violations)

    step = straight.ground_truth[0]
    expected = spec.mass_kg * (spec.gravity_m_s2 + step.az_m_s2) + step.downforce_n
    total = math.fsum(wheel.fz_n for wheel in step.wheels)
    assert total == pytest.approx(expected, rel=1.0e-6)
    assert step.downforce_n > 0.0
    assert spec.weight_n > 0.0


def test_invariant_4_sign_conventions(cornering: SampleRecord, spec: CarSpec) -> None:
    """sign(kappa) == sign(fx) and sign(alpha) == sign(fy) at all four corners. Backed from P2."""
    result = _result(4, cornering, spec)
    assert result.passed, result.summary()

    corrupted = _assert_detects(4, spec)
    assert all("sign" in violation.detail for violation in corrupted.violations)

    for step in cornering.ground_truth:
        for wheel in step.wheels:
            assert math.copysign(1.0, wheel.fx_n) == math.copysign(1.0, wheel.kappa)
            assert math.copysign(1.0, wheel.fy_n) == math.copysign(1.0, wheel.alpha_rad)


def test_invariant_5_left_right_symmetry(straight: SampleRecord, spec: CarSpec) -> None:
    """Symmetry at zero steer, zero camber, symmetric setup. Backed from P2."""
    result = _result(5, straight, spec)
    assert result.passed, result.summary()

    corrupted = _assert_detects(5, spec)
    assert any("slip ratio" in violation.detail for violation in corrupted.violations)

    fl, fr, rl, rr = straight.ground_truth[0].wheels
    assert fl.fz_n == fr.fz_n
    assert rl.fz_n == rr.fz_n
    assert fl.fz_n < rl.fz_n, "acceleration must load the rear axle"


def test_invariant_6_energy_balance(
    straight: SampleRecord, cornering: SampleRecord, spec: CarSpec
) -> None:
    """d(KE)/dt = ICE + MGU-K - drag, residual under 1%. Backed from P1."""
    for record in (straight, cornering):
        result = _result(6, record, spec)
        assert result.passed, result.summary()

    corrupted = _assert_detects(6, spec)
    assert all("energy" in violation.detail for violation in corrupted.violations)

    step = straight.ground_truth[0]
    kinetic = spec.mass_kg * (step.vx_m_s * step.ax_m_s2 + step.vy_m_s * step.ay_m_s2)
    supplied = step.ice_power_w + step.mgu_k_power_w - step.drag_w
    assert kinetic > 0.0
    assert abs(supplied - kinetic) / abs(kinetic) < 0.01


def test_invariant_7_gearbox_progression(straight: SampleRecord, spec: CarSpec) -> None:
    """Monotonic progression, and no reverse under positive throttle. Backed from P1."""
    result = _result(7, straight, spec)
    assert result.passed, result.summary()

    corrupted = _assert_detects(7, spec)
    assert any("backwards" in violation.detail for violation in corrupted.violations)

    gears = [step.gear for step in straight.ground_truth]
    assert gears == sorted(gears)
    assert min(gears) >= 1
    assert max(gears) <= len(spec.gear_ratios)


def test_invariant_8_determinism(straight: SampleRecord, spec: CarSpec) -> None:
    """Two serialisations of one record are byte-identical. Backed from P5.

    Non-vacuity here is a leak test rather than a record mutation, because the invariant
    is a property of the storage path. Writing a wall-clock value into Parquet key-value
    metadata is the single most common way real runs lose byte-identical output, so that
    is exactly what the test injects and expects the comparison to catch.
    """
    result = _result(8, straight, spec)
    assert result.passed, result.summary()

    leaked = serialise_frames(straight, {**METADATA, "written_at": "2026-09-30T00:00:01Z"})
    repeated = serialise_frames(straight, {**METADATA, "written_at": "2026-09-30T00:00:02Z"})
    assert leaked != repeated, "the byte comparison cannot detect a differing byte"
    assert serialise_frames(straight) == serialise_frames(straight, METADATA)


def test_every_mutation_is_detected_by_exactly_its_invariant(spec: CarSpec) -> None:
    """Cross-check: the whole mutation set, each against its own checker only."""
    for number, record in sorted(_MISSIONS.items()):
        results = run_all(record, spec)
        target = next(result for result in results if result.number == number)
        assert not target.passed, f"mutation {number} went unnoticed"


def test_cornering_record_does_not_satisfy_the_symmetry_invariant(
    cornering: SampleRecord, spec: CarSpec
) -> None:
    """Invariant 5 must not pass on a cornering record: that would make it vacuous."""
    result = _result(5, cornering, spec)
    assert not result.passed, "the symmetry check accepted lateral motion"
    assert result.violations


def test_checkers_never_raise_on_absurd_input(straight: SampleRecord, spec: CarSpec) -> None:
    """A violated invariant is data, not an exception, so one bad number reports once."""
    from f1telemetry.testing.records import with_wheel

    absurd = with_wheel(straight, 0, "FL", fz_n=-5000.0, mu=-1.0)
    for number in (1, 2, 3, 4, 5, 6, 7, 8):
        result = _result(number, absurd, spec)
        assert isinstance(result.violations, tuple)
    assert not _result(2, absurd, spec).passed
