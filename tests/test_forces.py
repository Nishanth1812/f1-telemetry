"""P1-T3: speed-dependent aero and the longitudinal tyre force, tested against their claims.

``PHASES.md`` P1-T3 and P1-T6 ask for ``Fz_aero = 1/2 rho v^2 A Cl(v)`` and
``Fdrag = 1/2 rho v^2 A Cd(v)`` with ``Cl``/``Cd`` as *functions of speed* rather than constants,
and for a longitudinal Pacejka force behind a slip ratio whose low-speed denominator is guarded.
The claims are checked here in the form that would actually fail:

* **Dynamic pressure, then a speed-dependent coefficient.** The force is the dynamic pressure
  times the reference area times a coefficient read off the configured speed axis, checked
  against that arithmetic written out step by step, and checked at a curve knot where no
  interpolation is involved at all.
* **Not proportional to v^2.** ``PLAN.md`` section 5.1 says downforce and drag are *not*
  proportional to ``v^2`` over the range, which is a claim about the ratio, so the ratio is
  what is tested: ``Fdown / (q A)`` has to move with speed, and a constant-coefficient
  implementation fails it.
* **Sign conventions.** ``GroundTruthStep`` fixes ``kappa`` positive in drive and ``fx_n``
  positive forward, so slip and force share a sign, the force is exactly zero at zero slip, and
  drag pushes against the direction of travel. Downforce is even in speed and never negative.
* **The low-speed guard.** ``kappa = (omega r - v) / max(v, eps)`` with ``eps`` from
  ``car_spec.yaml``. At rest the denominator is ``eps`` and nowhere else, and the guard is a
  number in the data rather than a literal in the code.
* **Load behaviour, and the grip limit.** A contact patch carrying no load makes no
  longitudinal force - including a negative load, which would otherwise reverse the sign of the
  whole formula - and the force is bounded by ``mu * Fz`` across a wide slip sweep, with the
  peak actually reaching that bound rather than merely sitting under it.
* **Finite output.** Every force is finite over a range wider than the car can reach, including
  past the top of the aero table, where the curves are held flat instead of extrapolated.

Every physical number comes from the loaded ``car_spec.yaml``, so this file contains no tuned
constant. The one-way ``@njit`` boundary is the P0 convention: the primitives take flat scalars
and ``float64`` arrays and are what a compiled kernel calls, while
:func:`forces.step_forces` is the Python-facing composition that re-checks the configuration
before use.

The ``forces`` marker keeps this apart from the ``kernel`` marker on the integrator: this file
is about the numbers the integrator is fed, not about the scheme it integrates with.
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

# The njit options the physics core uses (PLAN.md section 4.1 rule 4), read off the dispatchers
# rather than off the decorator text: a copy of the decorator in a test can only ever agree with
# the decorator, which is the thing that needed checking. `cache` is absent on purpose - numba
# consumes it at decoration and does not report it back on `targetoptions`, so asserting it here
# would be asserting nothing.
EXPECTED_OPTIONS: dict[str, Any] = {
    "fastmath": False,
    "nopython": True,
    "nogil": True,
    "boundscheck": False,
    "error_model": "numpy",
}


@pytest.fixture(scope="module")
def config(spec: CarSpec) -> KernelConfig:
    """The validated, flat configuration the force model is allowed to see."""
    return spec.kernel_config()


def _tyre(config: KernelConfig, slip: float, load_n: float) -> float:
    """``tyre_longitudinal_force`` with the configured coefficients, in the documented order."""
    return forces.tyre_longitudinal_force(
        slip,
        load_n,
        config.pacejka_b,
        config.pacejka_c,
        config.pacejka_e,
        config.pacejka_mu,
    )


def _aero(config: KernelConfig, speed_m_s: float) -> tuple[float, float]:
    """``aero_forces`` with the configured scalars and the shared speed axis."""
    return forces.aero_forces(
        speed_m_s,
        config.air_density_kg_m3,
        config.reference_area_m2,
        config.aero_speed_m_s,
        config.cl,
        config.cd,
    )


def _reference_curve(speed_m_s: float, breakpoints: np.ndarray, values: np.ndarray) -> float:
    """Piecewise-linear, ends held flat - the arithmetic the kernel does, in plain Python.

    Written with the same expressions in the same order, so an exact comparison is meaningful.
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
    slip: float,
    stiffness: float,
    shape: float,
    curvature: float,
) -> float:
    """The longitudinal Magic Formula normalised by its own peak, in plain Python.

    Compared with a tolerance rather than for bit equality: ``atan`` and ``sin`` come from the
    platform's libm and this suite runs on both Windows and Linux, so the last bits of a
    transcendental are not this project's to assert on. The formula is what is being tested.
    """
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
    """A Python fallback here would be a silent step backwards, so the dispatchers are asked.

    What this pins is that each primitive is a compiled dispatcher with the project's options -
    ``fastmath`` off, ``boundscheck`` off, ``error_model="numpy"`` - and that a call registered a
    signature rather than falling back to Python. ``cache=True`` is *not* asserted here: numba
    consumes it at decoration time, so it cannot be read back off ``targetoptions``, and the
    honest proof of it needs a fresh interpreter and an empty cache directory, which
    ``tests/test_longitudinal_kernel.py`` does once for the kernel convention this module follows.
    """
    _aero(config, 50.0)
    _tyre(config, 0.1, 4_000.0)
    # Both helpers and the slip ratio are normally reached through `aero_forces` or
    # `step_forces`, so their own dispatchers have no signature until something calls them
    # directly - which is what this test is doing.
    forces.dynamic_pressure_pa(50.0, config.air_density_kg_m3)
    forces.speed_curve(50.0, config.aero_speed_m_s, config.cl)
    forces.slip_ratio(52.0, 50.0, config.slip_ratio_min_speed_m_s)
    for function in (
        forces.dynamic_pressure_pa,
        forces.speed_curve,
        forces.aero_forces,
        forces.slip_ratio,
        forces.tyre_longitudinal_force,
    ):
        name = function.py_func.__name__
        assert len(function.signatures) >= 1, f"{name} never compiled a signature"
        options = dict(function.targetoptions)
        for option, expected in EXPECTED_OPTIONS.items():
            assert options[option] == expected, (name, option)


def test_dynamic_pressure_is_half_rho_v_squared(config: KernelConfig) -> None:
    """The documented relationship, checked against arithmetic written out, at rest too."""
    density = config.air_density_kg_m3
    assert forces.dynamic_pressure_pa(0.0, density) == 0.0
    for speed in (1.0, 12.5, 60.0, 95.0, 105.0):
        assert forces.dynamic_pressure_pa(speed, density) == 0.5 * density * speed * speed
    # Proportional to the square of speed, which is the only thing dynamic pressure claims.
    assert forces.dynamic_pressure_pa(20.0, density) == pytest.approx(
        4.0 * forces.dynamic_pressure_pa(10.0, density), rel=1e-12
    )


def test_the_aero_coefficients_are_read_off_the_configured_speed_axis(
    config: KernelConfig,
) -> None:
    """``Cl(v)`` and ``Cd(v)`` are interpolations of the file's curves, at knots and between.

    The knot comparison is exact: no interpolation is involved, so the kernel has to return the
    file's own number. The between-knot comparison is exact too, because the reference performs
    the same two multiplications and one addition in the same order on the same doubles.
    """
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
    """Above the top knot the curve is clamped, not extrapolated.

    ``car_spec.yaml`` stops at 105 m/s, and a car being dragged past that must not be handed a
    coefficient from a line through the last two points. Holding the end value is finite,
    monotone, and the conservative choice; a test that only checked speeds inside the table
    would leave the out-of-range behaviour to chance.
    """
    axis = config.aero_speed_m_s
    assert float(axis[0]) == 0.0, "the first knot is at rest, so there is no under-range to test"
    last = float(axis[-1])
    for speed in (last + 1.0, 150.0, 400.0):
        assert forces.speed_curve(speed, axis, config.cl) == float(config.cl[-1])
        assert forces.speed_curve(speed, axis, config.cd) == float(config.cd[-1])
    # The same rule applies below the first knot, which a curve starting at rest never reaches -
    # so it is exercised on a slice of the committed table rather than left untested.
    inner_axis = axis[2:]
    assert forces.speed_curve(0.0, inner_axis, config.cl[2:]) == float(config.cl[2])
    assert forces.speed_curve(30.0, inner_axis, config.cd[2:]) == float(config.cd[2])


def test_aero_forces_are_dynamic_pressure_times_area_times_the_speed_dependent_coefficient(
    config: KernelConfig,
) -> None:
    """``Fz_aero = 1/2 rho v^2 A Cl(v)`` and ``Fdrag = 1/2 rho v^2 A Cd(v)``, at a knot.

    Evaluated at a breakpoint on purpose: with no interpolation in the way, the only ways to get
    this wrong are to drop the density, the area, the square or the coefficient.
    """
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
    """PLAN.md section 5.1: the coefficients are not constant over the speed range.

    This is the claim a constant-``Cl`` implementation fails. ``Fz / (q A)`` has to be exactly
    ``Cl(v)``, so it has to move as the speed moves, and the movement is what the curve says.
    """
    ratios = []
    for speed in np.linspace(0.0, float(config.aero_speed_m_s[-1]), 51)[1:]:
        downforce, _ = _aero(config, float(speed))
        pressure = 0.5 * config.air_density_kg_m3 * float(speed) * float(speed)
        ratios.append(downforce / (pressure * config.reference_area_m2))
    assert len(set(np.round(ratios, 9))) > 1, "downforce is proportional to v^2, so Cl is constant"
    assert ratios[0] < ratios[-1], "the Cl curve rises, so the ratio must rise with speed"
    assert ratios[-1] == pytest.approx(float(config.cl[-1]), rel=1e-12)


def test_drag_pushes_against_the_direction_of_travel(config: KernelConfig) -> None:
    """``GroundTruthStep`` signs force along +x, so drag is signed: negative going forward.

    Forward motion gives a negative drag force, reversing gives a positive one - the same
    quadratic magnitude either way - and standing still gives exactly zero, because the dynamic
    pressure is zero rather than because of a special case.
    """
    forward, forward_drag = _aero(config, 80.0)
    backward, backward_drag = _aero(config, -80.0)
    _, still_drag = _aero(config, 0.0)
    assert forward_drag < 0.0
    assert backward_drag > 0.0
    assert backward_drag == -forward_drag
    assert still_drag == 0.0
    assert backward == forward, "downforce is even in speed"


def test_downforce_is_never_negative_and_grows_with_speed(config: KernelConfig) -> None:
    """Downforce is added to the vertical load, so it has to be a non-negative quantity."""
    previous = -1.0
    for speed in np.linspace(0.0, 1.5 * float(config.aero_speed_m_s[-1]), 121):
        downforce, _ = _aero(config, float(speed))
        assert downforce >= previous, speed
        assert math.isfinite(downforce), speed
        previous = downforce
    assert previous > 0.0


def test_the_slip_ratio_is_the_documented_guarded_expression(config: KernelConfig) -> None:
    """``kappa = (omega r - v) / max(v, eps)``, exactly, wherever the guard does not bind.

    Both speeds are circumferential, so ``wheel_speed_m_s`` is already ``omega * r``: the rolling
    radius enters when Task 4 builds the wheel state, and this function is handed the product.
    """
    guard = config.slip_ratio_min_speed_m_s
    assert guard > 0.0
    for speed in (guard * 2.0, 20.0, 60.0, 95.0):
        for wheel_speed in (0.0, speed * 0.5, speed, speed * 1.25):
            assert forces.slip_ratio(wheel_speed, speed, guard) == pytest.approx(
                (wheel_speed - speed) / speed, rel=1e-12, abs=0.0
            )


def test_the_slip_ratio_guards_its_low_speed_denominator(config: KernelConfig) -> None:
    """Below the guard speed the denominator is the guard, never the road speed.

    At rest the unguarded expression divides by zero, which is the specific failure P1-T6 names.
    A sliding wheel at rest is a real state - it is a launch, or a locked wheel - so the
    denominator has to be the configured floor rather than something that happens to be large.
    """
    guard = config.slip_ratio_min_speed_m_s
    wheel_speed = 12.0
    for speed in (0.0, guard * 0.5, guard * 0.999):
        # Only the denominator is guarded: the numerator still subtracts the road speed, so a
        # car creeping forward at half the guard has less slip, not more, than one at rest.
        assert forces.slip_ratio(wheel_speed, speed, guard) == pytest.approx(
            (wheel_speed - speed) / guard, rel=1e-12
        )
    assert forces.slip_ratio(wheel_speed, 0.0, guard) == pytest.approx(
        wheel_speed / guard, rel=1e-12
    )
    # The guard stops applying the moment the road outruns it, so the two regimes meet rather
    # than leaving a step in the expression.
    assert forces.slip_ratio(wheel_speed, guard, guard) == pytest.approx(
        (wheel_speed - guard) / guard, rel=1e-12
    )
    assert forces.slip_ratio(wheel_speed, guard * 1.001, guard) == pytest.approx(
        (wheel_speed - guard * 1.001) / (guard * 1.001), rel=1e-12
    )


def test_the_slip_ratio_sign_follows_the_wheel_against_the_road(config: KernelConfig) -> None:
    """Driving slip is positive and braking slip is negative, per the ground-truth convention."""
    guard = config.slip_ratio_min_speed_m_s
    speed = 50.0
    assert forces.slip_ratio(speed * 1.2, speed, guard) > 0.0, "a wheel spinning faster is driving"
    assert forces.slip_ratio(speed * 0.8, speed, guard) < 0.0, "a wheel turning slower is braking"
    assert forces.slip_ratio(speed, speed, guard) == 0.0, "rolling without slip is no slip"
    # The convention has to survive the guard too: a stationary car with a spinning wheel is
    # driving slip, and a stationary car with a locked wheel is braking slip.
    assert forces.slip_ratio(12.0, 0.0, guard) > 0.0
    assert forces.slip_ratio(-12.0, 0.0, guard) < 0.0


def test_the_slip_guard_is_data_rather_than_a_literal(config: KernelConfig, spec: CarSpec) -> None:
    """``eps`` comes from ``car_spec.yaml``, so the low-speed behaviour is a calibration knob.

    Edited through the raw document, the same path a YAML edit takes, and asserted twice: below
    the guard the slip ratio has to move, and above it nothing may move at all - otherwise the
    "guard" would be quietly changing the whole curve rather than only its floor.
    """
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
    """``Fx = mu Fz sin(C atan(B kappa - E(B kappa - atan(B kappa))))``, term for term.

    Driven through :func:`forces.step_forces` rather than the primitive, so this also pins that
    the composition reads ``pacejka_b``, ``_c``, ``_e`` and ``_mu`` in the right order. Six small
    scalars in a row is exactly the kind of call where a transposed argument still compiles and
    still returns a plausible-looking force.
    """
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
    """``Fz <= 0`` gives exactly zero, and a negative load does not reverse the formula's sign.

    The zero-load case is a wheel in the air. The negative-load case is the one that bites: a
    load that has gone negative, or a fixture that starts one at -0.0, would otherwise get a
    force pushing the *wrong* way from a patch carrying nothing, and the error would surface
    several steps later looking like a traction problem.
    """
    for load_n in (0.0, -0.0, -1.0, -5_000.0):
        for slip in (-2.0, -0.1, 0.0, 0.1, 2.0):
            assert _tyre(config, slip, load_n) == 0.0, (slip, load_n)
    _, _, airborne = forces.step_forces(config, 60.0, 66.0, 0.0)
    assert airborne == 0.0


def test_the_longitudinal_force_sign_is_the_slip_sign(config: KernelConfig) -> None:
    """Invariant 4's convention at the force level: ``sign(kappa) == sign(fx)``.

    The Magic Formula is odd in slip, so the negative side is the exact negation of the positive
    side rather than a separately computed value that happens to be similar.
    """
    load_n = 6_000.0
    for slip in (0.01, 0.05, 0.2, 0.5, 1.5, 4.0):
        driving = _tyre(config, slip, load_n)
        braking = _tyre(config, -slip, load_n)
        assert driving > 0.0, slip
        assert braking == -driving, slip


def test_the_longitudinal_force_grows_with_the_vertical_load(config: KernelConfig) -> None:
    """Below the peak the force is proportional to load, because ``D = mu Fz`` in P1.

    Load sensitivity on ``D`` - grip *falling* as load rises - is P2-T3's work, and this is the
    assertion that says so: a non-proportional model added before then would have to change
    this test rather than quietly contradict it.
    """
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
    """``|Fx| <= mu Fz`` is a property of the Magic Formula, so it is measured rather than clamped.

    The bound is asserted over a slip sweep twenty times wider than the contact patch can reach,
    on both signs, and the peak is then required to *reach* ``mu Fz`` rather than merely sit
    under it - otherwise a formula returning a constant fraction of the limit would pass a
    one-sided bound and the test would be measuring nothing.
    """
    load_n = 5_000.0
    limit = config.pacejka_mu * load_n
    slips = np.linspace(-20.0, 20.0, 401)
    forces_over_slip = [_tyre(config, float(slip), load_n) for slip in slips]
    for slip, force in zip(slips, forces_over_slip, strict=True):
        assert abs(force) <= limit + 1e-9, slip
    peak = max(forces_over_slip)
    assert peak == pytest.approx(limit, rel=1e-3), "the curve never approaches the grip limit"

    # Past the peak the force falls away, which is the branch a launch sits on: a badly slipping
    # wheel is not a grip-limited wheel, and a formula that kept rising would be a straight line.
    peak_slip = float(slips[int(np.argmax(forces_over_slip))])
    assert 0.2 <= peak_slip <= 0.5, (
        f"the peak should be a few tenths of a slip ratio, got {peak_slip}"
    )
    assert _tyre(config, 5.0, load_n) < peak
    assert _tyre(config, 5.0, load_n) > 0.0


def test_forces_stay_finite_across_the_operating_range(config: KernelConfig) -> None:
    """Finite everywhere a P1 run can put them, including past the top of the aero table.

    The range is deliberately wider than the car reaches - speeds past the last aero knot, loads
    to 60 kN on one patch, slip to ten times anything a real contact patch sees - because the
    failure this guards against is a NaN appearing several steps into a run with no obvious
    cause. A diverging formula shows up here rather than in a golden trace.
    """
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
    """``(downforce, drag, tyre force)`` for one step, each equal to its primitive.

    The tuple is the whole Python-facing surface: a scenario wants the three numbers rather than
    the arguments to rebuild them, and the composition has to agree with the primitives it is
    built from rather than approximating them.
    """
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
    """The composition is where configuration and state meet, so it checks both before arithmetic.

    ``KernelConfig`` is a public frozen dataclass and a scenario may build one with
    ``dataclasses.replace``, so loader validation is a property of the loader rather than a
    guarantee. A nonfinite load is arithmetic that returns NaN rather than an error, and a NaN
    that reaches the integrator is a run that looks like it finished.
    """
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
    """Zero density, a zero slip guard and a nonpositive shape factor are all NaN arithmetic.

    The zero slip guard is the sharpest of them: it is exactly the divide-by-zero P1-T6 exists
    to prevent, and a guard of 0.0 would turn every launch step into a NaN instead of a large
    slip ratio. The curvature factor is the odd one out - ``E`` legitimately carries a sign, so
    only finiteness is required of it, which is why its case here is an infinity and not a
    negative number.
    """
    broken = replace(config, **{name: value})
    with pytest.raises(ValueError, match=name):
        forces.step_forces(broken, 50.0, 52.0, 4_000.0)


def test_editing_the_car_spec_changes_the_forces_with_no_code_edit(
    tmp_path: Path,
    repo: Path,
    config: KernelConfig,
) -> None:
    """P1-T1b for the force model: the file is the only place a coefficient lives.

    One edit per quantity the model consumes - a downforce coefficient, a drag coefficient, the
    peak friction and the slip guard - and each has to move the force it is supposed to move. A
    physics constant written into the module would pass every other test in this file.
    """
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
    # At 60 m/s the guard does not bind, so doubling it cannot move the slip ratio - which is
    # what makes the 1.1 above attributable to the friction edit alone.
    assert forces.slip_ratio(wheel_speed, speed, config.slip_ratio_min_speed_m_s) == pytest.approx(
        (wheel_speed - speed) / speed, rel=1e-12
    )
    # At rest it binds, so the same edit has to move the launch slip ratio and with it the force.
    # Without this the guard edit would be decorative: four of the five edits would be pinned and
    # the fifth would only be checked for not breaking anything.
    at_rest = forces.step_forces(config, 0.0, 12.0, load_n)
    at_rest_edited = forces.step_forces(edited, 0.0, 12.0, load_n)
    assert at_rest_edited[2] != at_rest[2], "doubling eps has to change the launch slip ratio"
