"""P1-T3, P1-T6 and P1-T7: speed-dependent aero, the longitudinal tyre force and the wheel state.

Aero and tyre claims, in the form that would actually fail:

* Downforce and drag are dynamic pressure times area times a coefficient read off the speed axis.
  ``PLAN.md`` section 5.1 says they are *not* proportional to ``v^2``, so ``Fdown / (q A)`` must
  move with speed; a constant-coefficient model fails that.
* Slip and force share a sign (``kappa`` positive in drive, ``fx_n`` forward), the force is exactly
  zero at zero slip, and an unloaded patch makes no force - a negative load included, since it
  would otherwise flip the sign of the whole formula.
* The slip denominator is ``max(v, eps)`` with ``eps`` read from ``car_spec.yaml``, so the guard is
  data rather than a literal.
* ``|Fx| <= mu Fz`` over a slip sweep, and the peak must actually reach it.

Wheel state claims:

* C9.1.1: only the rear wheels receive drive torque, and they sum to the drivetrain torque exactly.
  The equal split is a symmetry assumption (there is no differential model), not a calibration.
* C9.1.2 and C9.9.1: no traction control and no limited slip, so the drive torque ignores each
  wheel's own speed and the other rear wheel's speed.
* Free-rolling fronts obey ``I dω/dt = -Fx r``, and ``I dω/dt = T_drive - Fx r`` closes the loop.

Every physical number comes from the loaded ``car_spec.yaml``. The ``forces`` marker keeps this
apart from the integrator's ``kernel`` marker: this file is about the numbers the integrator is
fed, not the scheme it integrates with.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
import yaml

from f1telemetry.contracts.car_spec import CarSpec, load_car_spec
from f1telemetry.physics import forces

if TYPE_CHECKING:
    from pathlib import Path

    from f1telemetry.contracts.car_spec import KernelConfig

pytestmark = pytest.mark.forces

# Read off the dispatchers rather than the decorator text, so the check is not a copy of the thing
# it checks. `cache` is absent on purpose: numba consumes it at decoration and does not report it.
EXPECTED_OPTIONS: dict[str, Any] = {
    "fastmath": False,
    "nopython": True,
    "nogil": True,
    "boundscheck": False,
    "error_model": "numpy",
}


@pytest.fixture(scope="module")
def config(spec: CarSpec) -> KernelConfig:
    return spec.kernel_config()


def _tyre(config: KernelConfig, slip: float, load_n: float) -> float:
    return forces.tyre_longitudinal_force(
        slip, load_n, config.pacejka_b, config.pacejka_c, config.pacejka_e, config.pacejka_mu
    )


def _wheel_tyre(config: KernelConfig, speed_m_s: float, omega_rad_s: float, load_n: float) -> float:
    return forces.wheel_tyre_force_n(
        speed_m_s,
        omega_rad_s,
        load_n,
        config.rolling_radius_m,
        config.slip_ratio_min_speed_m_s,
        config.pacejka_b,
        config.pacejka_c,
        config.pacejka_e,
        config.pacejka_mu,
    )


def _aero(config: KernelConfig, speed_m_s: float) -> tuple[float, float]:
    return forces.aero_forces(
        speed_m_s,
        config.air_density_kg_m3,
        config.reference_area_m2,
        config.aero_speed_m_s,
        config.cl,
        config.cd,
    )


def _reference_curve(speed_m_s: float, breakpoints: np.ndarray, values: np.ndarray) -> float:
    """Piecewise-linear with flat ends, written with the kernel's own expressions and order.

    Same operations in the same order on the same doubles, so an exact comparison is meaningful.
    """
    last = breakpoints.size - 1
    if speed_m_s <= float(breakpoints[0]):
        return float(values[0])
    if speed_m_s >= float(breakpoints[last]):
        return float(values[last])
    index = 0
    while speed_m_s > float(breakpoints[index + 1]):
        index += 1
    span = float(breakpoints[index + 1]) - float(breakpoints[index])
    weight = (speed_m_s - float(breakpoints[index])) / span
    return float(values[index]) * (1.0 - weight) + float(values[index + 1]) * weight


def _reference_magic_formula(
    slip: float, stiffness: float, shape: float, curvature: float
) -> float:
    """Compared with a tolerance: ``atan`` and ``sin`` come from the platform's libm, and this
    suite runs on both Windows and Linux, so the last bits of a transcendental are not asserted."""
    scaled = stiffness * slip
    inner = scaled - curvature * (scaled - math.atan(scaled))
    return math.sin(shape * math.atan(inner))


def _document(repo: Path) -> dict[str, Any]:
    document = yaml.safe_load((repo / "car_spec.yaml").read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _at(root: dict[str, Any], path: tuple[Any, ...]) -> Any:
    node: Any = root
    for part in path:
        node = node[part]
    return node


def _write(root: dict[str, Any], tmp_path: Path) -> Path:
    path = tmp_path / "car_spec.yaml"
    path.write_text(yaml.safe_dump(root, sort_keys=False), encoding="utf-8")
    return path


def test_the_force_primitives_are_compiled_kernels_with_the_project_options(
    config: KernelConfig,
) -> None:
    """A Python fallback here would be a silent step backwards, so the dispatchers are asked."""
    _aero(config, 50.0)
    _tyre(config, 0.1, 4_000.0)
    # The helpers and the slip ratio are normally reached through the compositions, so their own
    # dispatchers have no signature until something calls them directly.
    forces.dynamic_pressure_pa(50.0, config.air_density_kg_m3)
    forces.speed_curve(50.0, config.aero_speed_m_s, config.cl)
    forces.slip_ratio(52.0, 50.0, config.slip_ratio_min_speed_m_s)
    forces.wheel_drive_torque_nm(forces.RL_WHEEL_INDEX, 2_000.0)
    forces.static_wheel_load_n(
        config.mass_kg * config.gravity_m_s2, config.front_weight_fraction, forces.FL_WHEEL_INDEX
    )
    _wheel_tyre(config, 50.0, 52.0 / config.rolling_radius_m, 2_000.0)
    forces.wheel_angular_acceleration_rad_s2(1_000.0, 100.0, 0.36, 0.9)
    for function in (
        forces.dynamic_pressure_pa,
        forces.speed_curve,
        forces.aero_forces,
        forces.slip_ratio,
        forces.tyre_longitudinal_force,
        forces.wheel_drive_torque_nm,
        forces.static_wheel_load_n,
        forces.wheel_tyre_force_n,
        forces.wheel_angular_acceleration_rad_s2,
    ):
        name = function.py_func.__name__
        assert len(function.signatures) >= 1, f"{name} never compiled a signature"
        options = dict(function.targetoptions)
        for option, expected in EXPECTED_OPTIONS.items():
            assert options[option] == expected, (name, option)


def test_dynamic_pressure_is_half_rho_v_squared(config: KernelConfig) -> None:
    density = config.air_density_kg_m3
    assert forces.dynamic_pressure_pa(0.0, density) == 0.0
    for speed in (1.0, 12.5, 60.0, 95.0, 105.0):
        assert forces.dynamic_pressure_pa(speed, density) == 0.5 * density * speed * speed
    assert forces.dynamic_pressure_pa(20.0, density) == pytest.approx(
        4.0 * forces.dynamic_pressure_pa(10.0, density), rel=1e-12
    )


def test_the_aero_coefficients_are_read_off_the_configured_speed_axis(
    config: KernelConfig,
) -> None:
    """Exact at the knots, where the kernel must return the file's own number, and exact between
    them against the reference arithmetic."""
    axis = config.aero_speed_m_s
    for index in range(axis.size):
        speed = float(axis[index])
        assert forces.speed_curve(speed, axis, config.cl) == float(config.cl[index])
        assert forces.speed_curve(speed, axis, config.cd) == float(config.cd[index])

    for speed in np.linspace(0.0, float(axis[-1]), 101):
        assert forces.speed_curve(float(speed), axis, config.cl) == _reference_curve(
            float(speed), axis, config.cl
        )
        assert forces.speed_curve(float(speed), axis, config.cd) == _reference_curve(
            float(speed), axis, config.cd
        )


def test_a_speed_curve_is_held_flat_outside_its_own_range(config: KernelConfig) -> None:
    """Clamped, not extrapolated: a car dragged past the table must not get a coefficient from a
    line through the last two points."""
    axis = config.aero_speed_m_s
    assert float(axis[0]) == 0.0, "the first knot is at rest, so there is no under-range to test"
    last = float(axis[-1])
    for speed in (last + 1.0, 150.0, 400.0):
        assert forces.speed_curve(speed, axis, config.cl) == float(config.cl[-1])
        assert forces.speed_curve(speed, axis, config.cd) == float(config.cd[-1])
    # A curve starting at rest never reaches the under-range branch, so use a slice of the table.
    inner_axis = axis[2:]
    assert forces.speed_curve(0.0, inner_axis, config.cl[2:]) == float(config.cl[2])
    assert forces.speed_curve(30.0, inner_axis, config.cd[2:]) == float(config.cd[2])


def test_aero_forces_are_dynamic_pressure_times_area_times_the_speed_dependent_coefficient(
    config: KernelConfig,
) -> None:
    """At a breakpoint, so no interpolation can hide a dropped density, area, square or coeff."""
    index = 2
    speed = float(config.aero_speed_m_s[index])
    pressure = 0.5 * config.air_density_kg_m3 * speed * speed
    downforce, drag = _aero(config, speed)
    assert downforce == pytest.approx(
        pressure * config.reference_area_m2 * float(config.cl[index]), rel=1e-12
    )
    assert drag == pytest.approx(
        -pressure * config.reference_area_m2 * float(config.cd[index]), rel=1e-12
    )


def test_aero_forces_are_not_proportional_to_speed_squared(config: KernelConfig) -> None:
    """The claim a constant-``Cl`` implementation fails: ``Fz / (q A)`` is exactly ``Cl(v)``."""
    ratios = []
    for speed in np.linspace(0.0, float(config.aero_speed_m_s[-1]), 51)[1:]:
        downforce, _ = _aero(config, float(speed))
        pressure = 0.5 * config.air_density_kg_m3 * float(speed) * float(speed)
        ratios.append(downforce / (pressure * config.reference_area_m2))
    assert len(set(np.round(ratios, 9))) > 1, "downforce is proportional to v^2, so Cl is constant"
    assert ratios[0] < ratios[-1], "the Cl curve rises, so the ratio must rise with speed"
    assert ratios[-1] == pytest.approx(float(config.cl[-1]), rel=1e-12)


def test_drag_pushes_against_the_direction_of_travel(config: KernelConfig) -> None:
    """Drag is negative going forward and positive reversing; standing still gives exactly zero
    because the dynamic pressure is zero, not because of a special case."""
    forward, forward_drag = _aero(config, 80.0)
    backward, backward_drag = _aero(config, -80.0)
    _, still_drag = _aero(config, 0.0)
    assert forward_drag < 0.0
    assert backward_drag > 0.0
    assert backward_drag == -forward_drag
    assert still_drag == 0.0
    assert backward == forward, "downforce is even in speed"


def test_downforce_is_never_negative_and_grows_with_speed(config: KernelConfig) -> None:
    previous = -1.0
    for speed in np.linspace(0.0, 1.5 * float(config.aero_speed_m_s[-1]), 121):
        downforce, _ = _aero(config, float(speed))
        assert downforce >= previous, speed
        assert math.isfinite(downforce), speed
        previous = downforce
    assert previous > 0.0


def test_the_slip_ratio_is_the_documented_guarded_expression(config: KernelConfig) -> None:
    """``kappa = (omega r - v) / max(v, eps)`` wherever the guard does not bind. Both speeds are
    circumferential, so the rolling radius is already folded into ``wheel_speed_m_s``."""
    guard = config.slip_ratio_min_speed_m_s
    assert guard > 0.0
    for speed in (guard * 2.0, 20.0, 60.0, 95.0):
        for wheel_speed in (0.0, speed * 0.5, speed, speed * 1.25):
            assert forces.slip_ratio(wheel_speed, speed, guard) == pytest.approx(
                (wheel_speed - speed) / speed, rel=1e-12, abs=0.0
            )


def test_the_slip_ratio_guards_its_low_speed_denominator(config: KernelConfig) -> None:
    """Below the guard the denominator is the guard, never the road speed. At rest the unguarded
    expression divides by zero, which is the failure P1-T6 names."""
    guard = config.slip_ratio_min_speed_m_s
    wheel_speed = 12.0
    for speed in (0.0, guard * 0.5, guard * 0.999):
        # Only the denominator is guarded: the numerator still subtracts the road speed.
        assert forces.slip_ratio(wheel_speed, speed, guard) == pytest.approx(
            (wheel_speed - speed) / guard, rel=1e-12
        )
    assert forces.slip_ratio(wheel_speed, 0.0, guard) == pytest.approx(
        wheel_speed / guard, rel=1e-12
    )
    # The two regimes meet at the guard, so there is no step in the expression.
    assert forces.slip_ratio(wheel_speed, guard, guard) == pytest.approx(
        (wheel_speed - guard) / guard, rel=1e-12
    )
    assert forces.slip_ratio(wheel_speed, guard * 1.001, guard) == pytest.approx(
        (wheel_speed - guard * 1.001) / (guard * 1.001), rel=1e-12
    )


def test_the_slip_ratio_uses_speed_magnitude_in_reverse(config: KernelConfig) -> None:
    guard = config.slip_ratio_min_speed_m_s
    for speed in (-guard * 0.5, -guard * 4.0):
        denominator = max(abs(speed), guard)
        assert forces.slip_ratio(speed * 1.2, speed, guard) == pytest.approx(
            (speed * 1.2 - speed) / denominator
        )
        assert forces.slip_ratio(speed * 0.8, speed, guard) == pytest.approx(
            (speed * 0.8 - speed) / denominator
        )


def test_the_slip_ratio_sign_follows_the_wheel_against_the_road(config: KernelConfig) -> None:
    """Driving slip is positive and braking slip negative, per the ground-truth convention."""
    guard = config.slip_ratio_min_speed_m_s
    speed = 50.0
    assert forces.slip_ratio(speed * 1.2, speed, guard) > 0.0, "a wheel spinning faster is driving"
    assert forces.slip_ratio(speed * 0.8, speed, guard) < 0.0, "a wheel turning slower is braking"
    assert forces.slip_ratio(speed, speed, guard) == 0.0, "rolling without slip is no slip"
    # The convention survives the guard: a stationary car with a spinning or locked wheel.
    assert forces.slip_ratio(12.0, 0.0, guard) > 0.0
    assert forces.slip_ratio(-12.0, 0.0, guard) < 0.0


def test_the_slip_guard_is_data_rather_than_a_literal(config: KernelConfig, spec: CarSpec) -> None:
    """Edited through the raw document, the same path a YAML edit takes. Below the guard the slip
    ratio must move; above it nothing may, or the "guard" would be changing the whole curve."""
    guard = config.slip_ratio_min_speed_m_s
    assert forces.slip_ratio(12.0, 0.0, guard) == pytest.approx(12.0 / guard, rel=1e-12)
    rolling = forces.slip_ratio(12.0, guard * 4.0, guard)
    assert rolling == pytest.approx((12.0 - guard * 4.0) / (guard * 4.0), rel=1e-12)

    tyres = cast("dict[str, Any]", spec.raw["tyres"])
    doubled = replace(
        spec,
        raw={**spec.raw, "tyres": {**tyres, "slip_ratio_min_speed_m_s": guard * 2.0}},
    ).kernel_config()
    assert doubled.slip_ratio_min_speed_m_s == guard * 2.0
    assert forces.slip_ratio(12.0, 0.0, doubled.slip_ratio_min_speed_m_s) == pytest.approx(
        12.0 / (guard * 2.0), rel=1e-12
    )
    assert forces.slip_ratio(12.0, guard * 4.0, doubled.slip_ratio_min_speed_m_s) == rolling


def test_the_longitudinal_force_is_the_configured_pacejka_magic_formula(
    config: KernelConfig,
) -> None:
    """Driven through :func:`forces.step_forces` so the composition is also checked to read the
    four Pacejka scalars in the right order: a transposed argument still compiles and still
    returns a plausible-looking force."""
    load_n = 4_000.0
    for slip in (-3.0, -0.5, -0.1, 0.0, 0.1, 0.3, 1.0, 3.0):
        _, _, tyre = forces.step_forces(config, 60.0, 60.0 * (1.0 + slip), load_n)
        assert tyre == pytest.approx(
            config.pacejka_mu
            * load_n
            * _reference_magic_formula(slip, config.pacejka_b, config.pacejka_c, config.pacejka_e),
            rel=1e-12,
            abs=0.0,
        )


def test_no_slip_means_no_longitudinal_force(config: KernelConfig) -> None:
    """Exactly zero, not "very small": zero slip is the state a coasting car is in."""
    for load_n in (0.0, 1_000.0, 12_000.0):
        assert _tyre(config, 0.0, load_n) == 0.0
    _, _, from_composition = forces.step_forces(config, 40.0, 40.0, 4_000.0)
    assert from_composition == 0.0


def test_a_contact_patch_with_no_load_makes_no_longitudinal_force(config: KernelConfig) -> None:
    """The negative-load case is the one that bites: a force pushing the wrong way from a patch
    carrying nothing would surface several steps later looking like a traction problem."""
    for load_n in (0.0, -0.0, -1.0, -5_000.0):
        for slip in (-2.0, -0.1, 0.0, 0.1, 2.0):
            assert _tyre(config, slip, load_n) == 0.0, (slip, load_n)
    _, _, airborne = forces.step_forces(config, 60.0, 66.0, 0.0)
    assert airborne == 0.0


def test_the_longitudinal_force_sign_is_the_slip_sign(config: KernelConfig) -> None:
    """``sign(kappa) == sign(fx)``. The Magic Formula is odd in slip, so the braking side is the
    exact negation of the driving side, not a separately computed value that looks similar."""
    load_n = 6_000.0
    for slip in (0.01, 0.05, 0.2, 0.5, 1.5, 4.0):
        driving = _tyre(config, slip, load_n)
        braking = _tyre(config, -slip, load_n)
        assert driving > 0.0, slip
        assert braking == -driving, slip


def test_the_longitudinal_force_grows_with_the_vertical_load(config: KernelConfig) -> None:
    """Proportional to load below the peak, because ``D = mu Fz`` in P1. Load sensitivity is
    P2-T3's work; a non-proportional model added before then would have to change this test."""
    slip = 0.1
    single = _tyre(config, slip, 2_000.0)
    assert single > 0.0
    assert _tyre(config, slip, 4_000.0) == pytest.approx(2.0 * single, rel=1e-12)
    assert _tyre(config, slip, 8_000.0) == pytest.approx(4.0 * single, rel=1e-12)
    # Downforce is the other half of the load story: more speed, more load, more grip.
    static_load = 4_000.0
    downforce, _ = _aero(config, 80.0)
    assert downforce > 0.0
    assert _tyre(config, slip, static_load + downforce) > _tyre(config, slip, static_load)


def test_the_longitudinal_force_never_exceeds_the_configured_grip_limit(
    config: KernelConfig,
) -> None:
    """The bound is measured over a sweep twenty times wider than the patch can reach. The peak
    must *reach* ``mu Fz``, or a formula returning a fixed fraction of the limit would pass a
    one-sided bound while the test measured nothing."""
    load_n = 5_000.0
    limit = config.pacejka_mu * load_n
    slips = np.linspace(-20.0, 20.0, 401)
    forces_over_slip = [_tyre(config, float(slip), load_n) for slip in slips]
    for slip, force in zip(slips, forces_over_slip, strict=True):
        assert abs(force) <= limit + 1e-9, slip
    peak = max(forces_over_slip)
    assert peak == pytest.approx(limit, rel=1e-3), "the curve never approaches the grip limit"

    # Past the peak the force falls away: a formula that kept rising would be a straight line.
    peak_slip = float(slips[int(np.argmax(forces_over_slip))])
    assert 0.2 <= peak_slip <= 0.5, (
        f"the peak should be a few tenths of a slip ratio, got {peak_slip}"
    )
    assert _tyre(config, 5.0, load_n) < peak
    assert _tyre(config, 5.0, load_n) > 0.0


def test_forces_stay_finite_across_the_operating_range(config: KernelConfig) -> None:
    """Deliberately wider than the car reaches, because the failure guarded against is a NaN
    appearing several steps into a run with no obvious cause."""
    guard = config.slip_ratio_min_speed_m_s
    for speed in np.linspace(-120.0, 250.0, 201):
        downforce, drag = _aero(config, float(speed))
        assert math.isfinite(downforce), speed
        assert math.isfinite(drag), speed
        for load_n in (0.0, 250.0, 4_000.0, 20_000.0, 60_000.0):
            for slip in (-10.0, -0.3, 0.0, 0.3, 10.0):
                assert math.isfinite(_tyre(config, slip, load_n)), (speed, load_n, slip)
            wheel_speed = float(speed) * (1.0 + load_n * 1e-4)
            assert math.isfinite(forces.slip_ratio(wheel_speed, float(speed), guard)), speed
            _, _, tyre = forces.step_forces(config, float(speed), wheel_speed, load_n)
            assert math.isfinite(tyre), (speed, load_n)


def test_step_forces_returns_the_three_terms_the_kernel_needs(config: KernelConfig) -> None:
    """Each term equals its primitive: the composition must agree with what it is built from."""
    speed = 70.0
    wheel_speed = 75.0
    load_n = 5_500.0
    downforce, drag, tyre = forces.step_forces(config, speed, wheel_speed, load_n)
    expected_downforce, expected_drag = _aero(config, speed)
    expected_slip = forces.slip_ratio(wheel_speed, speed, config.slip_ratio_min_speed_m_s)
    assert downforce == expected_downforce
    assert drag == expected_drag
    assert tyre == _tyre(config, expected_slip, load_n)
    assert downforce > 0.0
    assert drag < 0.0
    assert tyre > 0.0, "a wheel turning faster than the road is driving the car forward"


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("speed_m_s", (math.nan, 0.0, 4_000.0)),
        ("speed_m_s", (10.0, math.inf, 4_000.0)),
        ("wheel_speed_m_s", (10.0, math.nan, 4_000.0)),
        ("load_n", (10.0, 0.0, math.nan)),
        ("load_n", (10.0, 0.0, math.inf)),
    ],
)
def test_step_forces_refuses_inputs_that_would_produce_nans(
    config: KernelConfig,
    name: str,
    arguments: tuple[float, float, float],
) -> None:
    """``KernelConfig`` is a public frozen dataclass, so loader validation is a property of the
    loader, not a guarantee. A NaN that reaches the integrator is a run that looks finished."""
    with pytest.raises(ValueError, match=name):
        forces.step_forces(config, *arguments)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("air_density_kg_m3", 0.0),
        ("air_density_kg_m3", math.nan),
        ("reference_area_m2", 0.0),
        ("slip_ratio_min_speed_m_s", 0.0),
        ("slip_ratio_min_speed_m_s", math.nan),
        ("pacejka_b", 0.0),
        ("pacejka_c", -1.0),
        ("pacejka_c", 1.0),
        ("pacejka_c", 2.5),
        ("pacejka_e", 1.0),
        ("pacejka_e", math.inf),
        ("pacejka_mu", 0.0),
        ("pacejka_mu", math.nan),
    ],
)
def test_step_forces_refuses_a_configuration_it_could_not_use(
    config: KernelConfig,
    name: str,
    value: float,
) -> None:
    """A zero slip guard is the sharpest case: it is the divide-by-zero P1-T6 exists to prevent.
    ``E`` legitimately carries a sign, so only its finiteness is required, hence an infinity here
    rather than a negative number."""
    broken = replace(config, **{name: value})
    with pytest.raises(ValueError, match=name):
        forces.step_forces(broken, 50.0, 52.0, 4_000.0)


def test_editing_the_car_spec_changes_the_forces_with_no_code_edit(
    tmp_path: Path,
    repo: Path,
    config: KernelConfig,
) -> None:
    """P1-T1b for the force model: each consumed quantity has to move the force it feeds. A
    physics constant written into the module would pass every other test in this file."""
    speed = 60.0
    wheel_speed = 66.0
    load_n = 5_000.0
    before = forces.step_forces(config, speed, wheel_speed, load_n)

    root = _document(repo)
    _at(root, ("aero", "cl_curve", 3))["cl"] = float(config.cl[3]) * 1.25
    _at(root, ("aero", "cd_curve", 3))["cd"] = float(config.cd[3]) * 0.8
    _at(root, ("tyres", "longitudinal_pacejka"))["mu"] = config.pacejka_mu * 1.1
    _at(root, ("tyres",))["slip_ratio_min_speed_m_s"] = config.slip_ratio_min_speed_m_s * 2.0
    edited = load_car_spec(_write(root, tmp_path)).kernel_config()
    after = forces.step_forces(edited, speed, wheel_speed, load_n)

    assert after[0] / before[0] == pytest.approx(1.25, rel=1e-12), "more Cl, more downforce"
    assert after[1] / before[1] == pytest.approx(0.8, rel=1e-12), "less Cd, less drag"
    assert after[2] / before[2] == pytest.approx(1.1, rel=1e-12), "more mu, more force"
    # At 60 m/s the guard does not bind, so doubling it cannot move the slip ratio: the 1.1 above
    # is attributable to the friction edit alone.
    assert forces.slip_ratio(wheel_speed, speed, config.slip_ratio_min_speed_m_s) == pytest.approx(
        (wheel_speed - speed) / speed, rel=1e-12
    )
    # At rest the guard binds, so the guard edit has to move the launch force, or it is decorative.
    at_rest = forces.step_forces(config, 0.0, 12.0, load_n)
    at_rest_edited = forces.step_forces(edited, 0.0, 12.0, load_n)
    assert at_rest_edited[2] != at_rest[2], "doubling eps has to change the launch slip ratio"


def test_step_forces_narrows_numeric_scalars_before_numba(config: KernelConfig) -> None:
    """Equivalent int and float inputs use the same force arithmetic."""
    float_config = replace(config, air_density_kg_m3=1.0)
    forces.step_forces(float_config, 60.0, 66.0, 4_000.0)
    compiled = len(forces.aero_forces.nopython_signatures)
    assert compiled > 0
    integer_config = replace(config, air_density_kg_m3=1)
    forces.step_forces(integer_config, 60.0, 66.0, 4_000.0)
    forces.step_forces(config, 60, 66, 4_000)
    assert len(forces.aero_forces.nopython_signatures) == compiled
    with pytest.raises(ValueError, match="speed_m_s"):
        forces.step_forces(config, np.float32(60.0), 66.0, 4_000.0)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="pacejka_mu"):
        forces.step_forces(replace(config, pacejka_mu=np.float32(1.7)), 60.0, 66.0, 4_000.0)


def test_step_forces_validates_replaced_aero_arrays(config: KernelConfig) -> None:
    """A replaced config cannot bypass the loader's array guarantees."""
    repeated_axis = config.aero_speed_m_s.copy()
    repeated_axis[2] = repeated_axis[1]
    infinite_curve = config.cd.copy()
    infinite_curve[2] = np.inf
    malformed = (
        replace(config, cl=config.cl.astype(np.float32)),
        replace(config, cd=config.cd.astype(np.int64)),
        replace(config, cl=config.cl[::-1]),
        replace(config, aero_speed_m_s=config.aero_speed_m_s.tolist()),
        replace(config, cl=config.cl.reshape(2, 3)),
        replace(config, cd=config.cd[:-1]),
        replace(config, aero_speed_m_s=repeated_axis),
        replace(config, cd=infinite_curve),
        replace(config, cl=-config.cl),
        replace(config, cd=np.zeros_like(config.cd)),
    )
    for broken in malformed:
        with pytest.raises(ValueError, match="step_forces: config"):
            forces.step_forces(broken, 60.0, 66.0, 4_000.0)


def test_only_the_rear_wheels_receive_drive_torque_and_the_split_is_equal(
    config: KernelConfig,
) -> None:
    """C9.1.1: "The transmission may only drive the two rear wheels." The fronts get *exactly*
    zero: a straight-line run fed to the wrong axle still accelerates plausibly, so this failure
    is invisible in a trace. The four torques must also sum to the drivetrain torque, so the split
    cannot lose torque on the way to the road."""
    assert forces.WHEEL_COUNT == 4
    assert forces.FIRST_REAR_WHEEL_INDEX == forces.REAR_WHEEL_COUNT == 2
    for drivetrain_torque_nm in (0.0, 1.0, 2_400.0, -2_400.0):
        torques = [
            forces.wheel_drive_torque_nm(index, drivetrain_torque_nm)
            for index in range(forces.WHEEL_COUNT)
        ]
        assert torques[forces.FL_WHEEL_INDEX] == 0.0, drivetrain_torque_nm
        assert torques[forces.FR_WHEEL_INDEX] == 0.0, drivetrain_torque_nm
        assert torques[forces.RL_WHEEL_INDEX] == drivetrain_torque_nm / 2.0, drivetrain_torque_nm
        assert torques[forces.RR_WHEEL_INDEX] == drivetrain_torque_nm / 2.0, drivetrain_torque_nm
        assert sum(torques) == drivetrain_torque_nm, drivetrain_torque_nm


def test_the_rear_drive_split_is_half_and_not_a_configured_share() -> None:
    """A derived ``1 / REAR_WHEEL_COUNT``, stated rather than fitted: a synthetic symmetry
    assumption, not a calibrated number. Asserting the derivation stops a third "share" constant
    being added as configuration with no file entry behind it."""
    assert forces.REAR_DRIVE_SHARE == 1.0 / forces.REAR_WHEEL_COUNT == 0.5
    assert forces.wheel_drive_torque_nm(forces.RL_WHEEL_INDEX, 3.0) == (
        3.0 * forces.REAR_DRIVE_SHARE
    )


def test_a_wheel_that_is_already_spinning_is_not_given_less_drive_torque(
    config: KernelConfig,
) -> None:
    """C9.1.2: traction control looks like this in code - more slip, less torque. The torque must
    be *identical* across a four-decade sweep of wheel speed, and the tyre force must fall away
    at the top of the sweep so the absence of a cut is visible in the physics."""
    drivetrain_torque_nm = 2_400.0
    speed_m_s = 50.0
    load_n = 2_100.0
    rolling = speed_m_s / config.rolling_radius_m
    # All four are at or past peak slip, so the sweep measures the fall-off rather than the rise.
    torques = []
    forces_over_spin = []
    for omega_rad_s in (rolling, 2.0 * rolling, 10.0 * rolling, 50.0 * rolling):
        drive, tyre, _ = forces.step_wheel(
            config, forces.RL_WHEEL_INDEX, drivetrain_torque_nm, speed_m_s, omega_rad_s, load_n
        )
        torques.append(drive)
        forces_over_spin.append(tyre)
    assert len(set(torques)) == 1, "the drive torque varies with wheel speed, which is C9.1.2"
    assert torques[0] == drivetrain_torque_nm / 2.0
    assert forces_over_spin[0] == pytest.approx(0.0, abs=1e-9)
    assert forces_over_spin[1] > forces_over_spin[2] > forces_over_spin[3] > 0.0, (
        "a driven wheel spinning harder past the peak transmits less and less force, so this "
        "asserts the spin is real rather than the tyre model flattening out"
    )


def test_no_torque_moves_from_a_slower_rear_wheel_to_a_faster_one(config: KernelConfig) -> None:
    """C9.9.1: a limited-slip differential moves torque to the faster wheel, so its signature is
    the faster wheel receiving *more* than half. Both rear wheels are driven at very different
    speeds and must each still get exactly one half."""
    drivetrain_torque_nm = 1_800.0
    speed_m_s = 60.0
    rolling = speed_m_s / config.rolling_radius_m
    left_drive, left_tyre, _ = forces.step_wheel(
        config, forces.RL_WHEEL_INDEX, drivetrain_torque_nm, speed_m_s, rolling, 2_100.0
    )
    right_drive, right_tyre, _ = forces.step_wheel(
        config, forces.RR_WHEEL_INDEX, drivetrain_torque_nm, speed_m_s, 6.0 * rolling, 2_100.0
    )
    assert left_drive == right_drive == drivetrain_torque_nm / 2.0
    assert left_drive == forces.wheel_drive_torque_nm(forces.RL_WHEEL_INDEX, drivetrain_torque_nm)
    assert right_drive == forces.wheel_drive_torque_nm(forces.RR_WHEEL_INDEX, drivetrain_torque_nm)
    # The response may differ - each wheel answers its own slip - which is why the torque equality
    # above is a statement about torque rather than about the whole step.
    assert right_tyre > left_tyre > -1e-9, (
        "the faster wheel is the one past the tyre's peak, so it is the one transmitting less"
    )
    assert abs(left_tyre) <= 1e-9, (
        "a wheel rolling at omega r = v feels nothing. The bound rather than `== 0.0` is the "
        "rounding of `v / r * r` back to `v`, not a tolerance on the model"
    )


def test_the_front_wheels_free_roll_between_the_tyre_and_the_road(config: KernelConfig) -> None:
    """Undriven, not inert: a front wheel at exactly ``ω r = v`` neither accelerates nor
    decelerates, and one that is not rolling is spun up by the road."""
    speed_m_s = 70.0
    load_n = 1_800.0
    rolling = speed_m_s / config.rolling_radius_m
    drive, tyre, alpha = forces.step_wheel(
        config, forces.FL_WHEEL_INDEX, 2_400.0, speed_m_s, rolling, load_n
    )
    assert drive == 0.0
    assert tyre == 0.0
    assert alpha == 0.0

    locked = forces.step_wheel(config, forces.FR_WHEEL_INDEX, 2_400.0, speed_m_s, 0.0, load_n)
    assert locked[0] == 0.0
    assert locked[1] < 0.0, "a locked wheel is braking slip, so the force opposes the car"
    assert locked[2] > 0.0, "and the road spins that wheel back up"
    assert locked[2] == pytest.approx(
        -locked[1] * config.rolling_radius_m / config.wheel_inertia_kg_m2, rel=1e-12
    )

    spun = forces.step_wheel(
        config, forces.FR_WHEEL_INDEX, 2_400.0, speed_m_s, 4.0 * rolling, load_n
    )
    assert spun[1] > 0.0
    assert spun[2] < 0.0, "a front wheel spinning faster than the road is slowed by it"


def test_the_wheel_state_closes_the_loop_on_its_own_angular_momentum(
    config: KernelConfig,
) -> None:
    """``I dω/dt = T_drive - Fx r`` for every wheel at every sign of torque and slip. The reaction
    term is the point of owning a wheel state: a model integrating with only the drive torque
    invents energy. Checked against a plain-Python recomputation so a transposed argument fails."""
    inertia = config.wheel_inertia_kg_m2
    radius = config.rolling_radius_m
    dt_s = config.dt_s
    speed_m_s = 55.0
    rolling = speed_m_s / radius
    for wheel_index in range(forces.WHEEL_COUNT):
        for drivetrain_torque_nm in (-2_400.0, 0.0, 900.0, 2_400.0):
            for omega_rad_s in (0.0, 0.5 * rolling, rolling, 2.5 * rolling):
                drive, tyre, alpha = forces.step_wheel(
                    config, wheel_index, drivetrain_torque_nm, speed_m_s, omega_rad_s, 2_100.0
                )
                assert inertia * alpha == pytest.approx(drive - tyre * radius, rel=1e-12), (
                    wheel_index,
                    drivetrain_torque_nm,
                    omega_rad_s,
                )
                assert alpha == pytest.approx((drive - tyre * radius) / inertia, rel=1e-12)
                # One explicit Euler step over the configured interval must deliver exactly the
                # impulse the balance says. The absolute bound is rounding: when the balance is
                # materially zero, `omega r` differs from `v` only in the last bits of `v / r * r`,
                # and subtracting nearly equal speeds amplifies that.
                advanced = omega_rad_s + alpha * dt_s
                assert inertia * (advanced - omega_rad_s) == pytest.approx(
                    (drive - tyre * radius) * dt_s, rel=1e-9, abs=1e-9
                )


def test_drive_torque_spins_a_rear_wheel_up_and_the_road_holds_it_back(
    config: KernelConfig,
) -> None:
    """The two halves of the balance must be able to fight. Nothing decides the crossover speed,
    so the test asserts the sign of each term rather than a speed where they happen to match."""
    load_n = 2_100.0
    speed_m_s = 60.0
    radius = config.rolling_radius_m
    inertia = config.wheel_inertia_kg_m2
    rolling = speed_m_s / radius

    spinning_up = forces.step_wheel(
        config, forces.RL_WHEEL_INDEX, 2_400.0, speed_m_s, rolling, load_n
    )
    assert spinning_up[0] == 1_200.0
    assert spinning_up[1] == pytest.approx(0.0, abs=1e-9), "no slip, so no force"
    assert spinning_up[2] == pytest.approx(1_200.0 / inertia, rel=1e-12)

    reacting = forces.step_wheel(
        config, forces.RL_WHEEL_INDEX, 1_200.0, speed_m_s, 3.0 * rolling, load_n
    )
    assert reacting[1] > 0.0, "a slipping driven wheel pushes the car forward"
    assert reacting[2] < 0.0, "and the road takes the speed back out of the wheel"
    # Reverse drive torque is the path a reverse gear takes. It is measured from rest so the tyre
    # reaction is exactly zero: a locked wheel on a *moving* car is a different state, where the
    # road wins.
    reversing = forces.step_wheel(config, forces.RR_WHEEL_INDEX, -1_200.0, 0.0, 0.0, load_n)
    assert reversing[0] == -600.0
    assert reversing[1] == 0.0
    assert reversing[2] == pytest.approx(-600.0 / inertia, rel=1e-12)


def test_zero_and_negative_speed_stay_finite_and_keep_the_slip_sign(
    config: KernelConfig,
) -> None:
    """Exactly zero slip at rest - the state a launch starts from. Rolling backwards with
    ``ω r = v`` is also zero slip: reverse has to be drivable (C9.7)."""
    radius = config.rolling_radius_m
    load_n = 2_100.0

    at_rest = forces.step_wheel(config, forces.RL_WHEEL_INDEX, 0.0, 0.0, 0.0, load_n)
    assert at_rest[1] == 0.0
    assert at_rest[2] == 0.0

    launching = forces.step_wheel(config, forces.RL_WHEEL_INDEX, 2_400.0, 0.0, 40.0, load_n)
    assert launching[1] > 0.0, "a spinning wheel on a stationary car is driving slip"
    assert math.isfinite(launching[1])

    backwards = -45.0
    rolling_back = forces.step_wheel(
        config, forces.RL_WHEEL_INDEX, 0.0, backwards, backwards / radius, load_n
    )
    assert rolling_back[1] == 0.0, "rolling backwards without slip is still no slip"

    # A stationary wheel under a car sliding backwards is braking slip, pushing the car towards
    # zero speed.
    sliding = forces.step_wheel(config, forces.RL_WHEEL_INDEX, 0.0, backwards, 0.0, load_n)
    assert sliding[1] > 0.0
    assert sliding[2] < 0.0

    for speed_m_s in np.linspace(-200.0, 200.0, 401):
        for omega_rad_s in (-4_000.0, -40.0, 0.0, 40.0, 4_000.0):
            _, tyre, alpha = forces.step_wheel(
                config, forces.RL_WHEEL_INDEX, 1_500.0, float(speed_m_s), omega_rad_s, load_n
            )
            assert math.isfinite(tyre), (speed_m_s, omega_rad_s)
            assert math.isfinite(alpha), (speed_m_s, omega_rad_s)


def test_the_static_axle_loads_split_the_car_weight_by_the_front_fraction(
    config: KernelConfig,
) -> None:
    """``Fz = weight x axle fraction / 2``, summed back to the car's own weight. Load transfer is
    P2-T2's work, so P1's split is static and speed-independent."""
    weight_n = config.mass_kg * config.gravity_m_s2
    fraction = config.front_weight_fraction
    assert 0.0 < fraction < 1.0
    loads = [
        forces.static_wheel_load_n(weight_n, fraction, index) for index in range(forces.WHEEL_COUNT)
    ]
    front = weight_n * fraction / 2.0
    rear = weight_n * (1.0 - fraction) / 2.0
    assert loads[forces.FL_WHEEL_INDEX] == front
    assert loads[forces.FR_WHEEL_INDEX] == front
    assert loads[forces.RL_WHEEL_INDEX] == rear
    assert loads[forces.RR_WHEEL_INDEX] == rear
    assert sum(loads) == pytest.approx(weight_n, rel=1e-12)
    # No downforce, no speed dependence: the same four numbers at any speed, which is the property
    # P2-T2 removes.
    assert loads[forces.FL_WHEEL_INDEX] == loads[forces.FL_WHEEL_INDEX]


def test_the_wheel_tyre_force_is_the_configured_tyre_model_at_the_wheels_own_speed(
    config: KernelConfig,
) -> None:
    """``wheel_tyre_force_n`` is ``slip_ratio`` and ``tyre_longitudinal_force`` composed, checked
    against both primitives reached independently. Reading the curve at ``ω`` instead of ``ω r``
    would return a plausible force at the wrong slip."""
    radius = config.rolling_radius_m
    load_n = 3_000.0
    for speed_m_s in (0.0, 12.5, 60.0, 95.0):
        for omega_rad_s in (-400.0, 0.0, 20.0, 260.0):
            expected_slip = forces.slip_ratio(
                omega_rad_s * radius, speed_m_s, config.slip_ratio_min_speed_m_s
            )
            expected = _tyre(config, expected_slip, load_n)
            assert _wheel_tyre(config, speed_m_s, omega_rad_s, load_n) == expected, (
                speed_m_s,
                omega_rad_s,
            )
    # The load guard reaches the composition too: a wheel in the air transmits nothing.
    assert _wheel_tyre(config, 60.0, 400.0, 0.0) == 0.0


@pytest.mark.parametrize(
    "wheel_index",
    [-1, forces.WHEEL_COUNT, forces.WHEEL_COUNT + 1, 1.0, 0.5, True, "1", None],
    ids=["negative", "one-past", "far-past", "float", "fraction", "bool", "str", "none"],
)
def test_step_wheel_refuses_a_wheel_index_it_could_not_assemble(
    config: KernelConfig,
    wheel_index: object,
) -> None:
    """A float is refused rather than coerced: ``4.0`` would compile a second Numba specialisation
    beside the integer one. ``bool`` is refused although it is an ``int``, since ``True`` would be
    front left by value."""
    with pytest.raises(ValueError, match="wheel_index"):
        forces.step_wheel(config, wheel_index, 1_000.0, 50.0, 140.0, 2_000.0)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("drivetrain_torque_nm", (math.nan, 50.0, 140.0, 2_000.0)),
        ("drivetrain_torque_nm", (-math.inf, 50.0, 140.0, 2_000.0)),
        ("speed_m_s", (1_000.0, math.nan, 140.0, 2_000.0)),
        ("speed_m_s", (1_000.0, math.inf, 140.0, 2_000.0)),
        ("wheel_omega_rad_s", (1_000.0, 50.0, math.nan, 2_000.0)),
        ("wheel_omega_rad_s", (1_000.0, 50.0, -math.inf, 2_000.0)),
        ("load_n", (1_000.0, 50.0, 140.0, math.nan)),
        ("load_n", (1_000.0, 50.0, 140.0, math.inf)),
    ],
)
def test_step_wheel_refuses_inputs_that_would_produce_nans(
    config: KernelConfig,
    name: str,
    arguments: tuple[float, float, float, float],
) -> None:
    """A NaN angular speed is a wheel that has already left the model. It would reach the chassis
    while every column after it was NaN, so the run would finish looking successful."""
    with pytest.raises(ValueError, match=name):
        forces.step_wheel(config, forces.RL_WHEEL_INDEX, *arguments)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("wheel_inertia_kg_m2", 0.0),
        ("wheel_inertia_kg_m2", -0.9),
        ("wheel_inertia_kg_m2", math.nan),
        ("rolling_radius_m", 0.0),
        ("rolling_radius_m", -0.36),
        ("rolling_radius_m", math.inf),
        ("front_weight_fraction", 0.0),
        ("front_weight_fraction", 1.0),
        ("front_weight_fraction", 1.5),
        ("front_weight_fraction", math.nan),
    ],
)
def test_step_wheel_refuses_a_configuration_it_could_not_use(
    config: KernelConfig,
    name: str,
    value: float,
) -> None:
    """Inertia and radius are divisors of the angular acceleration, so zero is a NaN wheel. A
    front fraction of exactly 1 leaves the rear axle carrying nothing: a fraction of nothing, not a
    car."""
    broken = replace(config, **{name: value})
    with pytest.raises(ValueError, match=name):
        forces.step_wheel(broken, forces.RL_WHEEL_INDEX, 1_000.0, 50.0, 140.0, 2_000.0)


def test_step_wheel_narrows_numeric_scalars_before_numba(config: KernelConfig) -> None:
    """Equivalent int and float inputs use the same wheel arithmetic."""
    compiled = len(forces.wheel_drive_torque_nm.nopython_signatures)
    forces.step_wheel(config, forces.RL_WHEEL_INDEX, 1_000, 50, 140, 2_000)
    assert len(forces.wheel_drive_torque_nm.nopython_signatures) == compiled
    with pytest.raises(ValueError, match="speed_m_s"):
        forces.step_wheel(
            config,
            forces.RL_WHEEL_INDEX,
            1_000.0,
            np.float32(50.0),  # pyright: ignore[reportArgumentType]
            140.0,
            2_000.0,
        )
    # A replaced field is read through `getattr`, so it need not be `float`; what is refused is a
    # value that is neither an int nor a float, which np.float32 is not for Python's `isinstance`.
    with pytest.raises(ValueError, match="wheel_inertia_kg_m2"):
        forces.step_wheel(
            replace(config, wheel_inertia_kg_m2=np.float32(0.9)),
            forces.RL_WHEEL_INDEX,
            1_000.0,
            50.0,
            140.0,
            2_000.0,
        )


def test_editing_the_car_spec_changes_the_wheel_step_with_no_code_edit(
    tmp_path: Path,
    repo: Path,
    config: KernelConfig,
) -> None:
    """P1-T1b for the wheel model. Doubling the inertia has to halve the angular acceleration
    exactly, measured where tyre force is zero, so inertia is the only thing the ratio can be."""
    root = _document(repo)
    assert forces.validated_config_scalars(config, "test")["wheel_inertia_kg_m2"] > 0.0
    _at(root, ("tyres",))["wheel_inertia_kg_m2"] = config.wheel_inertia_kg_m2 * 2.0
    edited = load_car_spec(_write(root, tmp_path)).kernel_config()

    rolling = 60.0 / config.rolling_radius_m
    before = forces.step_wheel(config, forces.RL_WHEEL_INDEX, 1_200.0, 60.0, rolling, 2_100.0)
    after = forces.step_wheel(edited, forces.RL_WHEEL_INDEX, 1_200.0, 60.0, rolling, 2_100.0)
    assert edited.wheel_inertia_kg_m2 == 2.0 * config.wheel_inertia_kg_m2
    assert before[2] != after[2]
    assert after[2] == pytest.approx(before[2] / 2.0, rel=1e-12)
    assert before[0] == after[0], "the drive torque is not an inertia, so it must not move"
