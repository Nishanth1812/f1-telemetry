"""P2-T3: per-wheel steady lateral tyre response, tested against its claims.

``PHASES.md`` P2-T3 and the plan's Task 3 ask for a lateral Magic Formula with
documented load sensitivity on ``D`` and ``B`` and a configured camber response,
with exactly zero force for an unloaded wheel. The claims are checked here in the
form that would actually fail:

* **The formula, term for term.** The steady force is
  ``Fy = mu_eff(Fz) Fz sin(C atan(B_eff (alpha + alpha_gamma) - E(...)))``
  with the load-sensitive, clamped ``mu_eff`` and ``B_eff`` and the camber
  entered as an equivalent slip angle through the reference cornering
  stiffness. It is compared against the same arithmetic written out in plain
  Python, driven through :func:`tyres.step_lateral_force`, so a transposed
  argument in the eleven-scalar call still compiles and still returns a
  plausible-looking force.
* **Odd symmetry at zero camber.** With no camber the response is exactly odd
  in the slip angle, which is the symmetry a mirrored left/right turn depends
  on.
* **Sign.** Positive slip angle and positive camber each produce positive
  (leftward) ``Fy``, alone and together, over the whole swept range - not
  only near the origin.
* **Zero and negative load.** ``Fz <= 0`` gives exactly ``0.0`` for every slip
  and camber, including ``-0.0``: a negative load would otherwise invert the
  whole expression and push the car from a patch carrying nothing.
* **Reference-load peak.** At the configured reference load the swept peak
  reaches ``mu Fz_ref`` rather than merely sitting under it, and falls away
  past it.
* **Saturation and the bound after camber.** ``|Fy| <= mu_eff(Fz) Fz`` holds
  over slip angles, camber angles and loads far past anything a contact patch
  sees - including camber large enough to saturate on its own - because the
  bound is the formula's own, not a clamp behind it.
* **Load sensitivity.** Grip is not proportional to load: the peak at twice
  the reference load is ``2 (1 - s_peak)`` of it, the small-slip force grows
  by ``2 (1 - s_peak)(1 - s_stiff)``, and past the load where the sensitivity
  clamps at zero the force is exactly zero on both signs rather than reversed.
* **Camber contribution.** At the reference load the configured camber
  stiffness is realized - ``Fy(0, gamma) = K gamma`` in the linear range -
  camber alone reaches the same peak as slip alone, and its sign follows the
  camber sign.
* **Finite output.** Every force is finite over a range wider than the car can
  reach, including slip and camber angles of a million degrees and loads a
  billion times the car's weight.
* **Invalid input.** Nonfinite slip, camber and load, boolean inputs, and
  every configuration the model could not use - a nonpositive magnitude, a
  shape factor outside ``(0, 2]``, a curvature factor at or above 1, a
  sensitivity outside ``[0, 1)`` - are refused at the Python boundary before
  any arithmetic happens.

Every physical number comes from the loaded ``car_spec.yaml``, so this file
contains no tuned constant. The ``forces`` marker keeps it with the other
physics-core tests. The module is steady-state only, so there is no
combined-slip, relaxation or integration coverage here - those belong to
Tasks 4 and 5.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import yaml

from f1telemetry.contracts.car_spec import CarSpec, load_car_spec

# The layer-isolation rule bans importing the physics core from layers 4-6
# (PLAN.md section 3). This file is the boundary test that proves the core
# itself, so the import is carved out the same way the kernel's is.
from f1telemetry.physics import tyres  # noqa: TID251 -- the tyre tests test the core

if TYPE_CHECKING:
    from pathlib import Path

    from f1telemetry.contracts.car_spec import KernelConfig

pytestmark = pytest.mark.forces

# Read off the dispatchers rather than copied out of the decorator text: a
# copy in a test can only agree with the decorator, which is the thing that
# needed checking. `cache` is absent on purpose - numba consumes it at
# decoration time and does not report it back on `targetoptions`.
EXPECTED_OPTIONS: dict[str, Any] = {
    "fastmath": False,
    "nopython": True,
    "nogil": True,
    "boundscheck": False,
    "error_model": "numpy",
}


@pytest.fixture(scope="module")
def config(spec: CarSpec) -> KernelConfig:
    """The validated, flat configuration the tyre model is allowed to see."""
    return spec.kernel_config()


def _force(
    config: KernelConfig,
    slip_angle_deg: float,
    camber_deg: float,
    load_n: float,
) -> float:
    """``tyre_lateral_force`` with the configured coefficients, in the documented order."""
    return tyres.tyre_lateral_force(
        slip_angle_deg,
        camber_deg,
        load_n,
        config.lateral_pacejka_b,
        config.lateral_pacejka_c,
        config.lateral_pacejka_e,
        config.lateral_pacejka_mu,
        config.load_sensitivity_reference_n,
        config.load_sensitivity_peak,
        config.load_sensitivity_stiffness,
        config.camber_stiffness_n_per_deg,
    )


def _reference_magic_formula(scaled: float, shape: float, curvature: float) -> float:
    """The Magic Formula normalised by its own peak, in plain Python.

    Compared for tight rather than bit equality: ``atan`` and ``sin`` come
    from the platform's libm and this suite runs on both Windows and Linux,
    so the last bits of a transcendental are not this project's to assert
    on. The formula is what is being tested.
    """
    return math.sin(shape * math.atan(scaled - curvature * (scaled - math.atan(scaled))))


def _reference_lateral_force(
    config: KernelConfig,
    slip_angle_deg: float,
    camber_deg: float,
    load_n: float,
) -> float:
    """The documented steady lateral formula, written out in plain Python.

    A transcription of the module's documented arithmetic in the same order,
    so a difference is a difference in the *model* rather than in the order
    of two floating-point operations. It is not a captured output: a
    reference that copied a run's numbers could only prove the code still
    does what it did.
    """
    if load_n <= 0.0:
        return 0.0
    reference_load_n = config.load_sensitivity_reference_n
    peak_multiplier = max(
        0.0, 1.0 - config.load_sensitivity_peak * (load_n / reference_load_n - 1.0)
    )
    stiffness_multiplier = max(
        0.0,
        1.0 - config.load_sensitivity_stiffness * (load_n / reference_load_n - 1.0),
    )
    peak_n = config.lateral_pacejka_mu * peak_multiplier * load_n
    # The reference-load slope per degree; Magic Formula slip is evaluated in radians.
    cornering_n_per_deg = (
        config.lateral_pacejka_mu
        * reference_load_n
        * config.lateral_pacejka_b
        * config.lateral_pacejka_c
        * math.pi
        / 180.0
    )
    equivalent_slip_deg = slip_angle_deg + (
        config.camber_stiffness_n_per_deg * camber_deg / cornering_n_per_deg
    )
    scaled = config.lateral_pacejka_b * stiffness_multiplier * math.radians(equivalent_slip_deg)
    return peak_n * _reference_magic_formula(
        scaled, config.lateral_pacejka_c, config.lateral_pacejka_e
    )


def _peak_slip_deg(config: KernelConfig) -> float:
    """The slip angle that peaks, found on a fine sweep of the whole rising branch."""
    reference_load_n = config.load_sensitivity_reference_n
    slips = np.linspace(0.0, 90.0, 901)
    forces = [_force(config, float(slip), 0.0, reference_load_n) for slip in slips]
    return float(slips[int(np.argmax(forces))])


def _document(repo: Path) -> dict[str, Any]:
    document = yaml.safe_load((repo / "car_spec.yaml").read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _at(root: dict[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = root
    for part in path:
        node = node[part]
    return node


def _write(root: dict[str, Any], tmp_path: Path) -> Path:
    path = tmp_path / "car_spec.yaml"
    path.write_text(yaml.safe_dump(root, sort_keys=False), encoding="utf-8")
    return path


def test_the_lateral_primitives_are_compiled_kernels_with_the_project_options(
    config: KernelConfig,
) -> None:
    """A Python fallback here would be a silent step backwards, so the dispatchers are asked.

    What this pins is that each primitive is a compiled dispatcher with the
    project's options - ``fastmath`` off, ``boundscheck`` off,
    ``error_model="numpy"`` - and that a call registered a signature rather
    than falling back to Python.
    """
    reference_load_n = config.load_sensitivity_reference_n
    _force(config, 0.1, 0.0, reference_load_n)
    tyres.load_sensitivity_multiplier(reference_load_n, reference_load_n, 0.1)
    tyres.lateral_peak_friction(reference_load_n, config.lateral_pacejka_mu, reference_load_n, 0.1)
    tyres.lateral_slip_stiffness(reference_load_n, config.lateral_pacejka_b, reference_load_n, 0.05)
    tyres.reference_cornering_stiffness_n_per_deg(
        config.lateral_pacejka_mu,
        config.lateral_pacejka_b,
        config.lateral_pacejka_c,
        reference_load_n,
    )
    tyres.camber_equivalent_slip_deg(0.5, config.camber_stiffness_n_per_deg, 10_000.0)
    for function in (
        tyres.load_sensitivity_multiplier,
        tyres.lateral_peak_friction,
        tyres.lateral_slip_stiffness,
        tyres.reference_cornering_stiffness_n_per_deg,
        tyres.camber_equivalent_slip_deg,
        tyres.tyre_lateral_force,
    ):
        name = function.py_func.__name__
        assert len(function.signatures) >= 1, f"{name} never compiled a signature"
        options = dict(function.targetoptions)
        for option, expected in EXPECTED_OPTIONS.items():
            assert options[option] == expected, (name, option)


def test_the_lateral_force_is_the_documented_magic_formula(
    config: KernelConfig,
) -> None:
    """``Fy = mu_eff(Fz) Fz sin(C atan(B_eff alpha_eq - E(...)))``, term for term.

    Driven through :func:`tyres.step_lateral_force` rather than the
    primitive, so this also pins that the composition reads the lateral
    coefficients, the load-sensitivity triple and the camber stiffness in
    the right order. Eleven scalars in a row is exactly the kind of call
    where a transposed pair still compiles and still returns a
    plausible-looking force.
    """
    reference_load_n = config.load_sensitivity_reference_n
    for slip in (-30.0, -5.0, -0.5, 0.0, 0.5, 5.0, 30.0):
        for camber in (-5.0, -0.5, 0.0, 0.5, 5.0):
            for load in (0.5 * reference_load_n, reference_load_n, 2.0 * reference_load_n):
                force = tyres.step_lateral_force(config, slip, camber, load)
                assert force == pytest.approx(
                    _reference_lateral_force(config, slip, camber, load),
                    rel=1e-12,
                    abs=0.0,
                )


def test_the_lateral_force_is_odd_in_slip_angle_at_zero_camber(
    config: KernelConfig,
) -> None:
    """The symmetry a mirrored turn depends on: ``Fy(-alpha, 0) == -Fy(alpha, 0)``.

    The Magic Formula is odd in its argument, so the negative side is the
    exact negation of the positive side rather than a separately computed
    value that happens to be similar. Camber breaks the oddness - which is
    why it is pinned to zero here and gets its own sign test below.
    """
    reference_load_n = config.load_sensitivity_reference_n
    for load in (0.25 * reference_load_n, reference_load_n, 4.0 * reference_load_n):
        for slip in (0.001, 0.05, 0.3, 1.0, 10.0, 60.0):
            leftward = _force(config, slip, 0.0, load)
            rightward = _force(config, -slip, 0.0, load)
            assert rightward == -leftward, (slip, load)


def test_positive_slip_angle_and_camber_push_left(config: KernelConfig) -> None:
    """The sign contract, checked far from the origin as well as near it.

    Positive slip angle and positive camber each produce a positive
    (leftward) force, alone and together, at every load that still carries
    lateral capacity. The response is sign-preserving at *every* finite slip
    angle, not only in the linear range: that is what the shape-factor and
    curvature-factor bounds in the configuration exist to guarantee, and a
    model that reversed past its peak would pass a near-origin-only sign
    test.
    """
    reference_load_n = config.load_sensitivity_reference_n
    for load in (reference_load_n, 2.0 * reference_load_n):
        for slip in (0.001, 0.1, 0.5, 2.0, 30.0, 90.0):
            assert _force(config, slip, 0.0, load) > 0.0, (slip, load)
        for camber in (0.001, 0.1, 0.5, 2.0, 30.0):
            assert _force(config, 0.0, camber, load) > 0.0, (camber, load)
        assert _force(config, 0.3, 0.3, load) > 0.0
        # And the mirror image pushes right.
        assert _force(config, -0.3, -0.3, load) < 0.0


def test_zero_slip_angle_and_zero_camber_make_no_lateral_force(
    config: KernelConfig,
) -> None:
    """Exactly zero, not "very small": a straight, uncambered wheel makes no lateral force."""
    reference_load_n = config.load_sensitivity_reference_n
    for load in (0.0, reference_load_n, 12_000.0):
        assert _force(config, 0.0, 0.0, load) == 0.0
    assert tyres.step_lateral_force(config, 0.0, 0.0, reference_load_n) == 0.0


def test_an_unloaded_patch_makes_no_lateral_force(config: KernelConfig) -> None:
    """``Fz <= 0`` gives exactly zero, and a negative load does not reverse the formula's sign.

    The zero-load case is a wheel in the air. The negative-load case is the
    one that bites: a load that has gone negative, or a fixture that starts
    one at ``-0.0``, would otherwise get a force pushing the *wrong* way
    from a patch carrying nothing, and the error would surface several
    steps later looking like a grip problem.
    """
    for load in (0.0, -0.0, -1.0, -5_000.0):
        for slip in (-30.0, -0.5, 0.0, 0.5, 30.0):
            for camber in (-5.0, 0.0, 5.0):
                assert _force(config, slip, camber, load) == 0.0, (slip, camber, load)
    assert tyres.step_lateral_force(config, 5.0, 2.0, 0.0) == 0.0


def test_the_peak_at_the_reference_load_reaches_the_configured_grip(
    config: KernelConfig,
) -> None:
    """``max |Fy| = mu Fz_ref`` at the reference load, reached rather than approached.

    The bound is asserted over a slip sweep far wider than the
    contact patch can reach, on both signs, and the peak is then required
    to *reach* ``mu Fz_ref`` rather than merely sit under it - otherwise a
    formula returning a constant fraction of the limit would pass a
    one-sided bound and the test would be measuring nothing. Past the peak
    the force falls away, which is the branch a badly slipping wheel is on.
    """
    reference_load_n = config.load_sensitivity_reference_n
    limit = config.lateral_pacejka_mu * reference_load_n
    slips = np.linspace(-90.0, 90.0, 1801)
    forces_over_slip = [_force(config, float(slip), 0.0, reference_load_n) for slip in slips]
    for slip, force in zip(slips, forces_over_slip, strict=True):
        assert abs(force) <= limit * (1.0 + 1e-12), slip
    peak = max(forces_over_slip)
    assert peak == pytest.approx(limit, rel=1e-3), "the curve never approaches the grip limit"

    peak_slip = float(slips[int(np.argmax(forces_over_slip))])
    assert 0.0 < peak_slip < 90.0, f"the peak should be inside the swept range, got {peak_slip}"
    assert _force(config, 90.0, 0.0, reference_load_n) < peak
    assert _force(config, 90.0, 0.0, reference_load_n) > 0.0


def test_the_lateral_force_never_exceeds_the_available_peak_after_camber(
    config: KernelConfig,
) -> None:
    """``|Fy| <= mu_eff(Fz) Fz`` is a property of the formula, so it is measured, not clamped.

    The bound is asserted over slip angles to ninety degrees, camber angles
    past the whole declared channel range, and loads from a thousandth of
    the reference to a hundred times it - which is past the load where the
    sensitivity clamps grip at zero. The camber contribution is inside the
    bound *by construction*: it enters as an equivalent slip angle, so a
    camber-only sweep has to respect the same limit a slip-only sweep does.
    """
    reference_load_n = config.load_sensitivity_reference_n
    slips = np.linspace(-90.0, 90.0, 361)
    cambers = (-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0)
    load_fractions = (1e-3, 0.1, 0.5, 1.0, 2.0, 5.0, 11.0, 20.0, 100.0)
    for load_fraction in load_fractions:
        load_n = load_fraction * reference_load_n
        # The available peak, with the same clamped load sensitivity the
        # model applies.
        peak_multiplier = max(0.0, 1.0 - config.load_sensitivity_peak * (load_fraction - 1.0))
        limit = config.lateral_pacejka_mu * peak_multiplier * load_n
        for camber in cambers:
            for slip in slips:
                force = _force(config, float(slip), camber, load_n)
                assert abs(force) <= limit * (1.0 + 1e-12), (slip, camber, load_n)


def test_grip_falls_with_load_and_never_reverses(config: KernelConfig) -> None:
    """Load sensitivity is non-proportional, and it fades to zero rather than reversing.

    At twice the reference load the peak is ``2 (1 - s_peak)`` of the
    reference-load peak rather than twice it, and at half the reference load
    it is ``0.5 (1 + s_peak/2)`` rather than half - grip per newton rises
    as load falls, which is the whole point of carrying the sensitivity as
    data. Above the load where the linear fall would reach zero the
    multiplier is clamped at zero, so the force is exactly zero on both
    signs: a reversed response at absurd load is the failure the clamp
    exists to prevent.
    """
    reference_load_n = config.load_sensitivity_reference_n
    sensitivity = config.load_sensitivity_peak
    at_reference = max(
        _force(config, float(slip), 0.0, reference_load_n) for slip in np.linspace(0.0, 90.0, 901)
    )
    at_double = max(
        _force(config, float(slip), 0.0, 2.0 * reference_load_n)
        for slip in np.linspace(0.0, 90.0, 901)
    )
    at_half = max(
        _force(config, float(slip), 0.0, 0.5 * reference_load_n)
        for slip in np.linspace(0.0, 90.0, 901)
    )
    assert at_reference == pytest.approx(config.lateral_pacejka_mu * reference_load_n, rel=1e-3)
    assert at_double == pytest.approx(
        2.0 * (1.0 - sensitivity) * config.lateral_pacejka_mu * reference_load_n,
        rel=1e-3,
    )
    assert at_half == pytest.approx(
        0.5 * (1.0 + 0.5 * sensitivity) * config.lateral_pacejka_mu * reference_load_n,
        rel=1e-3,
    )
    # Non-proportional, which is the assertion the P1 linear model could not pass.
    assert at_double != pytest.approx(2.0 * at_reference, rel=1e-3)
    assert at_half != pytest.approx(0.5 * at_reference, rel=1e-3)

    # The clamp: the linear fall reaches zero at 1/sensitivity reference
    # loads above the reference, and past it the force is exactly zero,
    # not negative.
    clamped_load_n = (1.0 / sensitivity + 1.0) * reference_load_n
    for load in (clamped_load_n, 2.0 * clamped_load_n, 20.0 * clamped_load_n):
        for slip in (0.001, 0.5, 30.0):
            assert _force(config, slip, 0.0, load) == 0.0, (slip, load)


def test_the_slip_stiffness_falls_with_load(config: KernelConfig) -> None:
    """The second sensitivity: the curve softens with load, so the peak moves outward.

    ``s_stiff`` scales ``B``, so the slip angle that peaks is inversely
    proportional to the load-sensitive stiffness - the peak at twice the
    reference load sits ``1 / (1 - s_stiff)`` times further out - and the
    small-slip force grows by ``2 (1 - s_peak)(1 - s_stiff)``, less than
    the load ratio, because both the peak and the stiffness fall. A model
    with stiffness sensitivity but no peak sensitivity would still show the
    outward move; a model with neither would show neither.
    """
    reference_load_n = config.load_sensitivity_reference_n
    stiffness_sensitivity = config.load_sensitivity_stiffness
    fine_slips = np.linspace(0.0, 90.0, 18001)
    at_reference = [_force(config, float(slip), 0.0, reference_load_n) for slip in fine_slips]
    at_double = [_force(config, float(slip), 0.0, 2.0 * reference_load_n) for slip in fine_slips]
    peak_slip_reference = float(fine_slips[int(np.argmax(at_reference))])
    peak_slip_double = float(fine_slips[int(np.argmax(at_double))])
    assert peak_slip_double == pytest.approx(
        peak_slip_reference / (1.0 - stiffness_sensitivity), rel=1e-2
    )

    # Small-slip force: the compound factor, well below the load ratio of two.
    tiny_slip = 1e-4
    small_reference = _force(config, tiny_slip, 0.0, reference_load_n)
    small_double = _force(config, tiny_slip, 0.0, 2.0 * reference_load_n)
    compound = 2.0 * (1.0 - config.load_sensitivity_peak) * (1.0 - stiffness_sensitivity)
    assert small_double == pytest.approx(compound * small_reference, rel=1e-6)
    assert small_double < 2.0 * small_reference


def test_the_load_sensitivity_multiplier_is_clamped_and_nonnegative(
    config: KernelConfig,
) -> None:
    """The multiplier in isolation: one at the reference, falling, clamped at zero.

    Below the reference load the multiplier rises above one - lighter-loaded
    corners grip more per newton - and it is bounded there by
    ``1 + sensitivity`` because the load cannot go below zero. Above the
    load where the linear fall would reach zero it is exactly zero, and it
    stays zero rather than going negative.
    """
    reference_load_n = config.load_sensitivity_reference_n
    for name, sensitivity in (
        ("peak", config.load_sensitivity_peak),
        ("stiffness", config.load_sensitivity_stiffness),
    ):
        assert name in ("peak", "stiffness")
        assert (
            tyres.load_sensitivity_multiplier(reference_load_n, reference_load_n, sensitivity)
            == 1.0
        )
        assert tyres.load_sensitivity_multiplier(
            2.0 * reference_load_n, reference_load_n, sensitivity
        ) == pytest.approx(1.0 - sensitivity, rel=1e-12)
        assert tyres.load_sensitivity_multiplier(
            0.5 * reference_load_n, reference_load_n, sensitivity
        ) == pytest.approx(1.0 + 0.5 * sensitivity, rel=1e-12)
        assert tyres.load_sensitivity_multiplier(
            0.0, reference_load_n, sensitivity
        ) == pytest.approx(1.0 + sensitivity, rel=1e-12)
        clamped = (1.0 / sensitivity + 1.0) * reference_load_n
        assert tyres.load_sensitivity_multiplier(clamped, reference_load_n, sensitivity) == 0.0
        assert (
            tyres.load_sensitivity_multiplier(10.0 * clamped, reference_load_n, sensitivity) == 0.0
        )
        # A zero sensitivity is the P1 model: no load dependence at all.
        assert (
            tyres.load_sensitivity_multiplier(3.0 * reference_load_n, reference_load_n, 0.0) == 1.0
        )


def test_the_reference_cornering_stiffness_and_camber_equivalent_slip(
    config: KernelConfig,
) -> None:
    """The reference-load slope per degree and the camber angle it buys.

    The reference cornering stiffness is the initial slope of the Magic
    Formula at the reference load - the place where both sensitivities are
    exactly one - so the equivalent slip angle a camber angle is worth is
    the force the configured camber stiffness produces divided by that
    slope. Both are dimensioned in the units the configuration and the
    channels already use: newtons, degrees, newtons per degree.
    """
    cornering_n_per_deg = tyres.reference_cornering_stiffness_n_per_deg(
        config.lateral_pacejka_mu,
        config.lateral_pacejka_b,
        config.lateral_pacejka_c,
        config.load_sensitivity_reference_n,
    )
    assert cornering_n_per_deg == pytest.approx(
        config.lateral_pacejka_mu
        * config.load_sensitivity_reference_n
        * config.lateral_pacejka_b
        * config.lateral_pacejka_c
        * math.pi
        / 180.0,
        rel=1e-12,
    )
    # A degree of camber is worth K/C_alpha degrees of slip: K N/deg times
    # deg, divided by N/deg, leaves degrees.
    for camber in (-5.0, -0.5, 0.0, 0.5, 5.0):
        equivalent = tyres.camber_equivalent_slip_deg(
            camber, config.camber_stiffness_n_per_deg, cornering_n_per_deg
        )
        assert equivalent == pytest.approx(
            config.camber_stiffness_n_per_deg * camber / cornering_n_per_deg,
            rel=1e-12,
        )


def test_camber_contribution_is_the_configured_stiffness_at_the_reference_load(
    config: KernelConfig,
) -> None:
    """``Fy(0, gamma) = K gamma`` at the reference load, in the linear range.

    This is the sentence ``camber_stiffness_n_per_deg`` writes: at the
    reference load, a small camber angle produces the configured newtons
    per degree. Away from the reference load the same stiffness is realised
    through the load-sensitive cornering stiffness, so it grows by the
    compound factor rather than with the load, and it saturates at the same
    peak as slip rather than adding on top of it.
    """
    reference_load_n = config.load_sensitivity_reference_n
    # Cambers small enough that the Magic Formula is still linear, which is
    # the regime the configured stiffness is defined to be realized in.
    for camber in (0.02, 0.05, 0.1, 0.2):
        force = _force(config, 0.0, camber, reference_load_n)
        assert force == pytest.approx(config.camber_stiffness_n_per_deg * camber, rel=1e-3)
        assert _force(config, 0.0, -camber, reference_load_n) == -force
    # At twice the load the camber force grows by the same compound factor
    # as the slip response: both sensitivities apply to it too, because the
    # camber enters through the same cornering stiffness.
    compound = (1.0 - config.load_sensitivity_peak) * (1.0 - config.load_sensitivity_stiffness)
    for camber in (0.1, 0.5):
        at_reference = _force(config, 0.0, camber, reference_load_n)
        at_double = _force(config, 0.0, camber, 2.0 * reference_load_n)
        assert at_double == pytest.approx(2.0 * compound * at_reference, rel=1e-3)


def test_camber_alone_saturates_at_the_same_peak_as_slip(
    config: KernelConfig,
) -> None:
    """A camber-only sweep reaches ``mu Fz_ref`` too, and never passes it.

    The camber enters as an equivalent slip angle, so a large camber walks
    the same Magic Formula curve a large slip angle walks: it rises to the
    peak and falls back to the curve's asymptote. That is the whole reason
    the camber contribution is inside the friction bound by construction
    rather than by a clamp - and it is why a camber of tens of degrees
    cannot manufacture force a slip angle of the same equivalent size could
    not.
    """
    reference_load_n = config.load_sensitivity_reference_n
    limit = config.lateral_pacejka_mu * reference_load_n
    cambers = np.linspace(0.0, 200.0, 401)
    forces_over_camber = [
        _force(config, 0.0, float(camber), reference_load_n) for camber in cambers
    ]
    for camber, force in zip(cambers, forces_over_camber, strict=True):
        assert abs(force) <= limit * (1.0 + 1e-12), camber
    peak = max(forces_over_camber)
    assert peak == pytest.approx(limit, rel=1e-3), "camber alone never reaches the peak"
    # Past its own peak the camber-only force falls back, as a slip angle
    # past its peak does.
    assert _force(config, 0.0, 200.0, reference_load_n) < peak
    assert _force(config, 0.0, 200.0, reference_load_n) > 0.0


def test_forces_stay_finite_across_the_operating_range(config: KernelConfig) -> None:
    """Finite everywhere a run could put them, and far beyond it.

    The range is deliberately wider than the car reaches - slip and camber
    angles of a million degrees, loads from a micronewton to a billion
    times the reference load - because the failure this guards against is a
    NaN appearing several steps into a run with no obvious cause. The
    formula has no input-dependent division and no divergence: every
    operation is addition, multiplication, division by validated-positive
    configuration, ``atan`` and ``sin``.
    """
    reference_load_n = config.load_sensitivity_reference_n
    for load_n in (1e-6, 1.0, reference_load_n, 1e6 * reference_load_n):
        for slip in (-1e6, -90.0, -0.3, 0.0, 0.3, 90.0, 1e6):
            for camber in (-1e6, -10.0, 0.0, 10.0, 1e6):
                assert math.isfinite(_force(config, float(slip), float(camber), load_n)), (
                    slip,
                    camber,
                    load_n,
                )
    assert math.isfinite(tyres.step_lateral_force(config, 1e6, -1e6, 1e12))


def test_step_lateral_force_returns_the_one_force_the_kernel_needs(
    config: KernelConfig,
) -> None:
    """The composition agrees with the primitive it is built from.

    The single float is the whole Python-facing surface: a scenario or a
    kernel wants the force, not the arguments to rebuild it, and the
    composition has to agree with the primitive rather than approximate it.
    """
    reference_load_n = config.load_sensitivity_reference_n
    for slip in (-12.0, -0.7, 0.0, 0.7, 12.0):
        for camber in (-3.0, 0.0, 3.0):
            for load in (0.5 * reference_load_n, reference_load_n, 3.0 * reference_load_n):
                composed = tyres.step_lateral_force(config, slip, camber, load)
                assert composed == _force(config, slip, camber, load)


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("slip_angle_deg", (math.nan, 0.0, 4_000.0)),
        ("slip_angle_deg", (math.inf, 0.0, 10.0)),
        ("camber_deg", (10.0, math.nan, 4_000.0)),
        ("camber_deg", (10.0, -math.inf, 4_000.0)),
        ("load_n", (10.0, 0.0, math.nan)),
        ("load_n", (10.0, 0.0, math.inf)),
        ("slip_angle_deg", (True, 0.0, 4_000.0)),
    ],
)
def test_step_lateral_force_refuses_inputs_that_would_produce_nans(
    config: KernelConfig,
    name: str,
    arguments: tuple[float, float, float],
) -> None:
    """The composition is where configuration and state meet, so it checks both.

    ``KernelConfig`` is a public frozen dataclass and a scenario may build
    one with ``dataclasses.replace``, so loader validation is a property of
    the loader rather than a guarantee. A nonfinite slip angle is arithmetic
    that returns NaN rather than an error, and a NaN that reaches an
    integrator is a run that looks like it finished. ``bool`` is refused
    although it is an ``int``, because ``True`` as a slip angle is a caller
    bug rather than a number.
    """
    with pytest.raises(ValueError, match=name):
        tyres.step_lateral_force(config, *arguments)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("lateral_pacejka_b", 0.0),
        ("lateral_pacejka_b", -9.0),
        ("lateral_pacejka_b", math.nan),
        ("lateral_pacejka_c", 0.0),
        ("lateral_pacejka_c", -1.5),
        ("lateral_pacejka_c", 2.5),
        ("lateral_pacejka_c", math.nan),
        ("lateral_pacejka_e", 1.0),
        ("lateral_pacejka_e", 1.5),
        ("lateral_pacejka_e", math.inf),
        ("lateral_pacejka_mu", 0.0),
        ("lateral_pacejka_mu", math.nan),
        ("load_sensitivity_reference_n", 0.0),
        ("load_sensitivity_reference_n", -4_000.0),
        ("load_sensitivity_reference_n", math.nan),
        ("load_sensitivity_peak", -0.01),
        ("load_sensitivity_peak", 1.0),
        ("load_sensitivity_peak", math.nan),
        ("load_sensitivity_stiffness", -0.01),
        ("load_sensitivity_stiffness", 1.0),
        ("load_sensitivity_stiffness", math.nan),
        ("camber_stiffness_n_per_deg", 0.0),
        ("camber_stiffness_n_per_deg", -800.0),
        ("camber_stiffness_n_per_deg", math.nan),
    ],
)
def test_step_lateral_force_refuses_a_configuration_it_could_not_use(
    config: KernelConfig,
    name: str,
    value: float,
) -> None:
    """Every coefficient the model reads is held to the sign rule that makes it safe.

    The shape factor is bounded above by 2 and the curvature factor below 1
    because those are the conditions under which the lateral response stays
    sign-preserving and reaches its peak at every finite slip angle; a
    larger shape factor would let the curve cross zero and reverse at large
    slip, and a curvature factor at or above 1 would keep the peak out of
    reach. The sensitivities are bounded inside ``[0, 1)`` by the loader
    already; the boundary re-checks them because a replaced config can
    bypass the loader. The camber stiffness is a magnitude, so a negative
    one would reverse the camber sign contract.
    """
    broken = replace(config, **{name: value})
    with pytest.raises(ValueError, match=name):
        tyres.step_lateral_force(broken, 5.0, 1.0, 4_000.0)


def test_step_lateral_force_narrows_numeric_scalars_before_numba(
    config: KernelConfig,
) -> None:
    """Equivalent int and float inputs use the same force arithmetic."""
    reference_load_n = config.load_sensitivity_reference_n
    tyres.step_lateral_force(config, 0.5, 1.0, reference_load_n)
    compiled = len(tyres.tyre_lateral_force.nopython_signatures)
    assert compiled > 0
    tyres.step_lateral_force(config, 0, 1, int(reference_load_n))
    assert len(tyres.tyre_lateral_force.nopython_signatures) == compiled
    with pytest.raises(ValueError, match="slip_angle_deg"):
        tyres.step_lateral_force(config, np.float32(0.5), 1.0, reference_load_n)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="lateral_pacejka_mu"):
        tyres.step_lateral_force(
            replace(config, lateral_pacejka_mu=np.float32(1.55)), 0.5, 1.0, reference_load_n
        )


def test_editing_the_car_spec_changes_the_lateral_force_with_no_code_edit(
    tmp_path: Path,
    repo: Path,
    config: KernelConfig,
) -> None:
    """The file is the only place a lateral coefficient lives.

    One edit per quantity the model consumes - the peak friction, the shape
    and stiffness factors, the camber stiffness, the sensitivity reference
    and both sensitivities - and each has to move the force it is supposed
    to move, by the factor the documented formula says it must. A physics
    constant written into the module would pass every other test in this
    file.

    Each edit is loaded from its own copy of the file so the effects stay
    attributable: at a slip angle small enough that the Magic Formula is
    still linear, the force is ``mu Fz C B alpha``, so a peak-friction,
    shape or stiffness edit scales it by exactly the edit, and at the
    reference load a camber edit scales the camber force by exactly the
    edit.
    """
    reference_load_n = config.load_sensitivity_reference_n
    tiny_slip = 1e-4
    baseline = _force(config, tiny_slip, 0.0, reference_load_n)
    camber = 0.5
    camber_baseline = _force(config, 0.0, camber, reference_load_n)
    double_load = 2.0 * reference_load_n
    peak_baseline = _force(config, tiny_slip, 0.0, double_load)
    stiffness_baseline = _force(config, tiny_slip, 0.0, double_load)

    def edited(
        section: tuple[str, ...],
        key: str,
        value: float,
    ) -> KernelConfig:
        root = _document(repo)
        _at(root, section)[key] = value
        return load_car_spec(_write(root, tmp_path)).kernel_config()

    # Peak friction: the peak, and with it the whole curve, scales with mu.
    more_grip = edited(("tyres", "lateral_pacejka"), "mu", config.lateral_pacejka_mu * 1.1)
    assert _force(more_grip, tiny_slip, 0.0, reference_load_n) == pytest.approx(
        1.1 * baseline, rel=1e-6
    )
    # Shape factor: the linear-range slope is B C, so C scales the force.
    sharper = edited(("tyres", "lateral_pacejka"), "c", config.lateral_pacejka_c * 1.1)
    assert _force(sharper, tiny_slip, 0.0, reference_load_n) == pytest.approx(
        1.1 * baseline, rel=1e-6
    )
    # Stiffness factor: B scales the linear-range slope the same way.
    stiffer = edited(("tyres", "lateral_pacejka"), "b", config.lateral_pacejka_b * 1.2)
    assert _force(stiffer, tiny_slip, 0.0, reference_load_n) == pytest.approx(
        1.2 * baseline, rel=1e-6
    )
    # Camber stiffness: the camber force at the reference load is K gamma.
    softer_camber = edited(
        ("tyres",), "camber_stiffness_n_per_deg", config.camber_stiffness_n_per_deg * 0.8
    )
    assert _force(softer_camber, 0.0, camber, reference_load_n) == pytest.approx(
        0.8 * camber_baseline, rel=1e-3
    )
    # Sensitivity reference: at the committed reference load, a larger
    # reference makes that load a *below-reference* one, and lighter-loaded
    # corners grip more per newton - through both sensitivities, because the
    # small-slip force is the product of the peak and the stiffness.
    further_reference = edited(
        ("tyres", "load_sensitivity"),
        "reference_load_n",
        reference_load_n * 1.25,
    )
    assert _force(further_reference, tiny_slip, 0.0, reference_load_n) == pytest.approx(
        (1.0 + 0.2 * config.load_sensitivity_peak)
        * (1.0 + 0.2 * config.load_sensitivity_stiffness)
        * baseline,
        rel=1e-6,
    )
    # Peak sensitivity: at twice the reference load the peak friction is
    # 1 - s_peak, so doubling the sensitivity moves it to 1 - 2 s_peak.
    more_sensitive_peak = edited(
        ("tyres", "load_sensitivity"), "peak", config.load_sensitivity_peak * 2.0
    )
    assert _force(more_sensitive_peak, tiny_slip, 0.0, double_load) == pytest.approx(
        (1.0 - 2.0 * config.load_sensitivity_peak)
        / (1.0 - config.load_sensitivity_peak)
        * peak_baseline,
        rel=1e-6,
    )
    # Stiffness sensitivity: the peak is untouched, the curve softens.
    more_sensitive_stiffness = edited(
        ("tyres", "load_sensitivity"), "stiffness", config.load_sensitivity_stiffness * 2.0
    )
    assert _force(more_sensitive_stiffness, tiny_slip, 0.0, double_load) == pytest.approx(
        (1.0 - 2.0 * config.load_sensitivity_stiffness)
        / (1.0 - config.load_sensitivity_stiffness)
        * stiffness_baseline,
        rel=1e-6,
    )
    # Curvature: a larger E makes the curve's inner term grow more slowly,
    # which stretches the Magic Formula and moves its peak outward. The
    # committed E is below 1 and the edit keeps it there, because 1 is the
    # boundary the model refuses.
    sharper_curve = edited(("tyres", "lateral_pacejka"), "e", config.lateral_pacejka_e * 1.05)
    assert _peak_slip_deg(sharper_curve) > _peak_slip_deg(config)
    # And the baselines themselves are nonzero, or every ratio above is vacuous.
    assert baseline > 0.0
    assert camber_baseline > 0.0
    assert peak_baseline > 0.0
    assert stiffness_baseline > 0.0
