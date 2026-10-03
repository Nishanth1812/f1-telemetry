"""P2-T2: four-corner vertical loads and the quasi-static suspension travel they imply.

``PHASES.md`` P2-T2 and the plan's "Force/load contract" ask for corner loads from validated
CG/axle/track geometry, for both load-transfer directions to move load the right way, and for the
four loads to sum to weight plus aero - the same total-load accounting P1's invariant 3 checks.
The claims are checked here in the form that would actually fail:

* **The explicit coupling.** ``load`` depends on acceleration and acceleration depends on load,
  so the previous step's acceleration is an *input*. It is not estimated, not iterated on and not
  zeroed behind the caller's back: ``step_loads`` is handed ``ax``/``ay`` and returns the loads
  those numbers produce, so a run can seed them to zero and store the accelerations it resolves.
* **Symmetry, and the CG position.** At zero acceleration the four loads are equal within each
  axle, the two axles differ by the CG position, and the pair sums back to exactly
  ``forces.static_wheel_load_n`` - the P1 convention this model has to reduce to.
* **Direction and magnitude of both transfers.** ``ax > 0`` (forward) moves load to the rear by
  exactly ``m ax h / L``; ``ay > 0`` (leftward, so a left-hand turn) moves load to the *right*,
  because the outside of a left turn is the right-hand side. The lateral transfer is split between
  the axles by the configured roll-stiffness fraction and converted to a per-wheel change through
  each axle's own track, so the front and rear axles cannot silently share one width.
* **Mirroring.** Reversing the lateral acceleration reverses the corner order left-to-right and
  changes nothing else; reversing the longitudinal acceleration swaps the two axles. A model with
  one sign backwards passes the static tests and fails these.
* **Total-load conservation, always.** Over a sweep that includes accelerations the car cannot
  achieve, the four loads sum to weight plus aero plus the vertical-acceleration term, to floating
  tolerance - including the cases where a corner wants to carry a negative load.
* **Lift-off rather than a negative tyre load.** A corner whose raw load goes below zero is lifted
  and stays at exactly ``0.0``; the axle's other wheel is relieved by the same amount, because a
  corner at ``-2000 N`` was arithmetically taking load from its partner. The total is unchanged and
  no wheel is left pushing on the road from below.
* **Suspension travel is reported, not invented.** Travel is the corner's load deviation from its
  own weight-and-aero share over the configured wheel rate. Where the deviation would exceed the
  configured travel limit, the load is clamped to the limit and the clamped-away load is shared out
  rather than dropped, so the total survives; every corner that reached its limit is reported as
  limited, and so is any corner the sharing could not bring back inside its band.
* **Invalid input.** A nonpositive total vertical load, a nonfinite or non-numeric acceleration,
  downforce, a zero CG arm, a roll fraction outside ``(0, 1)`` and an axle array that is not a
  length-two contiguous ``float64`` vector are all refused at the Python boundary, before a
  ``boundscheck=False`` kernel can read past the end of one.
* **One validator for the model.** ``validated_load_scalars`` is the shared boundary every Python
  entry point and a compiled kernel call outside their loop: it hands back narrowed scalars and
  the per-axle vectors the kernels index, and it refuses everything a compiled function cannot
  defend against itself - nonfinite and non-numeric scalars, a nonpositive divisor, a roll
  fraction outside ``(0, 1)``, and any per-axle vector of the wrong length, dtype, layout or sign.

Every physical number comes from the loaded ``car_spec.yaml``, so this file contains no tuned
constant. The handful of cases that need geometry the committed car cannot produce - a lateral
acceleration high enough to lift a corner, a travel limit tight enough to bind - are reached with
the committed coefficients and an explicit statement of what is being asked for, not by replacing
physics to make a test comfortable.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from f1telemetry.contracts.car_spec import CarSpec
from f1telemetry.physics import forces, loads

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import KernelConfig

pytestmark = pytest.mark.forces

# Read off the dispatchers rather than copied out of the decorator text: a copy in a test can only
# agree with the decorator, which is the thing that needed checking. `cache` is absent on purpose -
# numba consumes it at decoration time and does not report it back.
EXPECTED_OPTIONS: dict[str, Any] = {
    "fastmath": False,
    "nopython": True,
    "nogil": True,
    "boundscheck": False,
    "error_model": "numpy",
}

# Enough lateral acceleration to lift a corner on the committed geometry: the front-left corner
# reaches zero at about 4.1 g and the rear-left at about 6.4 g, so 8 g lifts both. Computed from
# the configuration in `test_lift_off_needs_more_than_four_g` rather than quoted from a run.
LATERAL_LIFT_OFF_M_S2 = 8.0 * 9.80665


@pytest.fixture(scope="module")
def config(spec: CarSpec) -> KernelConfig:
    """The validated, flat configuration the load model is allowed to see."""
    return spec.kernel_config()


def _reference(
    config: KernelConfig,
    longitudinal_accel_m_s2: float,
    lateral_accel_m_s2: float,
    vertical_accel_m_s2: float = 0.0,
    aero_downforce_n: float = 0.0,
) -> tuple[float, float, float, float]:
    """The four corner loads as arithmetic written out in Python, in FL, FR, RL, RR order.

    Written with the same expressions in the same order as the primitive, so a difference is a
    difference in the *model* rather than in the order of two floating-point operations. It is a
    transcription of the documented formulae, not a captured output: a reference that copied a
    run's numbers could only prove the code still does what it did.
    """
    wheelbase = _wheelbase(config)
    total = config.mass_kg * (config.gravity_m_s2 + vertical_accel_m_s2) + aero_downforce_n
    front_axle = total * config.cg_to_rear_axle_m / wheelbase
    rear_axle = total - front_axle
    longitudinal = config.mass_kg * longitudinal_accel_m_s2 * config.cg_height_m / wheelbase
    roll_moment = config.mass_kg * lateral_accel_m_s2 * config.cg_height_m
    front_lateral = (
        roll_moment * config.roll_stiffness_front_fraction / float(config.axle_track_m[0])
    )
    rear_lateral = (
        roll_moment * (1.0 - config.roll_stiffness_front_fraction) / float(config.axle_track_m[1])
    )
    return (
        0.5 * (front_axle - longitudinal) - 0.5 * front_lateral,
        0.5 * (front_axle - longitudinal) + 0.5 * front_lateral,
        0.5 * (rear_axle + longitudinal) - 0.5 * rear_lateral,
        0.5 * (rear_axle + longitudinal) + 0.5 * rear_lateral,
    )


def _total_load_n(config: KernelConfig) -> float:
    return config.mass_kg * config.gravity_m_s2


def _wheelbase(config: KernelConfig) -> float:
    """The CG arms added up, which is what the model divides by.

    Not ``config.wheelbase_m``: the arms are derived from the wheelbase and the static split
    separately, and on this data their sum differs from the file's wheelbase in the last bit. Using
    the file's value here would compare against a wheelbase the model never sees.
    """
    return config.cg_to_front_axle_m + config.cg_to_rear_axle_m


def test_the_load_primitives_are_compiled_kernels_with_the_project_options(
    config: KernelConfig,
) -> None:
    """A Python fallback here would be a silent step backwards, so the dispatchers are asked."""
    loads_n = np.zeros(4, dtype=np.float64)
    base_n = np.zeros(4, dtype=np.float64)
    limited_n = np.zeros(4, dtype=np.int64)
    loads.vertical_load_total_n(config.mass_kg, config.gravity_m_s2, 0.0, 0.0)
    loads.longitudinal_load_transfer_n(config.mass_kg, 10.0, config.cg_height_m, _wheelbase(config))
    loads.lateral_roll_moment_nm(config.mass_kg, 10.0, config.cg_height_m)
    loads.corner_loads_n(
        loads_n,
        base_n,
        config.mass_kg,
        config.gravity_m_s2,
        0.0,
        0.0,
        10.0,
        10.0,
        config.cg_to_front_axle_m,
        config.cg_to_rear_axle_m,
        config.cg_height_m,
        config.axle_track_m,
        config.roll_stiffness_front_fraction,
    )
    loads.suspension_travel_m(2_000.0, 1_800.0, float(config.axle_ride_rate_n_per_m[0]))
    loads.clamp_travel_to_limits(
        loads_n, base_n, config.axle_ride_rate_n_per_m, config.axle_travel_limit_m, limited_n
    )
    for function in (
        loads.vertical_load_total_n,
        loads.longitudinal_load_transfer_n,
        loads.lateral_roll_moment_nm,
        loads.corner_loads_n,
        loads.suspension_travel_m,
        loads.clamp_travel_to_limits,
    ):
        name = function.py_func.__name__
        assert len(function.signatures) >= 1, f"{name} never compiled a signature"
        options = dict(function.targetoptions)
        for option, expected in EXPECTED_OPTIONS.items():
            assert options[option] == expected, (name, option)


def test_static_corner_loads_are_symmetric_and_split_by_the_cg_position(
    config: KernelConfig,
) -> None:
    """Zero acceleration: equal within each axle, the two axles split by the CG position."""
    result = loads.step_loads(config, longitudinal_accel_m_s2=0.0, lateral_accel_m_s2=0.0)
    front_left, front_right, rear_left, rear_right = result.load_n
    assert front_left == pytest.approx(front_right, rel=0.0, abs=0.0)
    assert rear_left == pytest.approx(rear_right, rel=0.0, abs=0.0)
    assert front_left != rear_left, "a 46/54 mass split does not put equal load on both axles"

    weight_n = _total_load_n(config)
    front_axle_n = front_left + front_right
    assert front_axle_n == pytest.approx(weight_n * config.cg_to_rear_axle_m / _wheelbase(config))
    assert rear_left + rear_right == pytest.approx(weight_n - front_axle_n, rel=1e-12, abs=1e-9), (
        "the two axles must account for the whole weight, not a rounded share of it"
    )
    assert sum(result.load_n) == pytest.approx(weight_n, rel=1e-12, abs=1e-9)
    assert result.total_load_n == pytest.approx(weight_n, rel=1e-12, abs=1e-9)
    assert not any(result.travel_limited)
    assert all(travel == pytest.approx(0.0, abs=0.0) for travel in result.travel_m), (
        "an unloaded-transfer static car is at its design ride height at every corner"
    )


def test_static_loads_reduce_to_the_p1_static_convention(config: KernelConfig) -> None:
    """The P1 model gave ``weight x axle fraction / 2``; P2 must still give that at zero transfer.

    This is the cross-check that catches a CG-arm convention flipped: P1 read
    ``front_weight_fraction`` straight from ``car_spec.yaml``, so if the P2 arms disagree with it
    the two models would disagree about which axle is heavier while both looked plausible.
    """
    weight_n = _total_load_n(config)
    result = loads.step_loads(config, longitudinal_accel_m_s2=0.0, lateral_accel_m_s2=0.0)
    for index, load_n in enumerate(result.load_n):
        assert load_n == pytest.approx(
            forces.static_wheel_load_n(weight_n, config.front_weight_fraction, index),
            rel=1e-15,
        )
    assert config.cg_to_rear_axle_m == pytest.approx(
        config.wheelbase_m * config.front_weight_fraction, rel=1e-15
    ), "the config derives the arms from the split, so they cannot be a second opinion"


def test_aero_and_vertical_acceleration_set_the_total_without_changing_the_split(
    config: KernelConfig,
) -> None:
    """Downforce and vertical acceleration change the total; the CG split is unchanged.

    Positive ``az`` is upward in the z-up frame (the plan's convention), so it *adds* to the load
    the tyres carry - the body pushing up on the road is the road pushing harder on the body.
    """
    weight_n = _total_load_n(config)
    downforce_n = 30_000.0
    vertical_m_s2 = 2.0
    result = loads.step_loads(
        config,
        longitudinal_accel_m_s2=0.0,
        lateral_accel_m_s2=0.0,
        vertical_accel_m_s2=vertical_m_s2,
        aero_downforce_n=downforce_n,
    )
    expected_total = weight_n + config.mass_kg * vertical_m_s2 + downforce_n
    assert result.total_load_n == pytest.approx(expected_total, rel=1e-15)
    assert result.base_load_n[0] == pytest.approx(result.base_load_n[1], rel=0.0, abs=0.0), (
        "an axle with no transfer carries the same on both sides whatever the total is"
    )
    assert result.base_load_n[0] == pytest.approx(
        expected_total * config.cg_to_rear_axle_m / config.wheelbase_m / 2.0, rel=1e-15
    )
    assert result.total_load_n == pytest.approx(
        loads.vertical_load_total_n(
            config.mass_kg,
            config.gravity_m_s2,
            vertical_m_s2,
            downforce_n,
        ),
        rel=1e-15,
    )


def test_longitudinal_acceleration_moves_load_rearward_by_the_planned_transfer(
    config: KernelConfig,
) -> None:
    """``dFz = m ax h / L`` forward to the rear axle, on every wheel of that axle."""
    accel_m_s2 = 12.0
    transfer_n = config.mass_kg * accel_m_s2 * config.cg_height_m / _wheelbase(config)
    assert loads.longitudinal_load_transfer_n(
        config.mass_kg, accel_m_s2, config.cg_height_m, _wheelbase(config)
    ) == pytest.approx(transfer_n, rel=1e-15)

    ahead = loads.step_loads(config, longitudinal_accel_m_s2=accel_m_s2, lateral_accel_m_s2=0.0)
    coast = loads.step_loads(config, longitudinal_accel_m_s2=0.0, lateral_accel_m_s2=0.0)
    braking = loads.step_loads(config, longitudinal_accel_m_s2=-accel_m_s2, lateral_accel_m_s2=0.0)

    for axle in (0, 1):
        assert ahead.base_load_n[axle * 2] - coast.base_load_n[axle * 2] == pytest.approx(
            0.0, abs=0.0
        ), "the base is the untransferred load; transfer moves load, it does not create it"
    for wheel in (forces.FL_WHEEL_INDEX, forces.FR_WHEEL_INDEX):
        assert coast.load_n[wheel] - ahead.load_n[wheel] == pytest.approx(0.5 * transfer_n)
    for wheel in (forces.RL_WHEEL_INDEX, forces.RR_WHEEL_INDEX):
        assert ahead.load_n[wheel] - coast.load_n[wheel] == pytest.approx(0.5 * transfer_n)

    assert ahead.load_n[forces.RL_WHEEL_INDEX] > coast.load_n[forces.RL_WHEEL_INDEX]
    assert ahead.load_n[forces.FL_WHEEL_INDEX] < coast.load_n[forces.FL_WHEEL_INDEX]
    assert braking.load_n[forces.FL_WHEEL_INDEX] > coast.load_n[forces.FL_WHEEL_INDEX], (
        "under braking the load moves forward, onto the axle that has not been asked to slow down"
    )
    assert ahead.total_load_n == pytest.approx(coast.total_load_n, rel=1e-15)
    assert braking.total_load_n == pytest.approx(coast.total_load_n, rel=1e-15)


def test_lateral_acceleration_moves_load_to_the_right_and_splits_it_by_roll_stiffness(
    config: KernelConfig,
) -> None:
    """``ay > 0`` is a leftward acceleration, so load moves to the right-hand wheels.

    The outside of a left-hand turn is the right-hand side of the car, and this is the claim a
    sign error breaks most visibly: the loads stay finite, symmetric and conserved with the sign
    flipped, and the car would simply understeer into every corner.
    """
    accel_m_s2 = 10.0
    result = loads.step_loads(config, longitudinal_accel_m_s2=0.0, lateral_accel_m_s2=accel_m_s2)
    coast = loads.step_loads(config, longitudinal_accel_m_s2=0.0, lateral_accel_m_s2=0.0)

    roll_moment_nm = config.mass_kg * accel_m_s2 * config.cg_height_m
    assert loads.lateral_roll_moment_nm(config.mass_kg, accel_m_s2, config.cg_height_m) == (
        pytest.approx(roll_moment_nm, rel=1e-15)
    )
    front_transfer_n = (
        roll_moment_nm * config.roll_stiffness_front_fraction / float(config.axle_track_m[0])
    )
    rear_transfer_n = (
        roll_moment_nm
        * (1.0 - config.roll_stiffness_front_fraction)
        / float(config.axle_track_m[1])
    )

    assert coast.load_n[forces.FL_WHEEL_INDEX] - result.load_n[forces.FL_WHEEL_INDEX] == (
        pytest.approx(0.5 * front_transfer_n)
    ), "the front-left corner is the inside of a left turn and unloads"
    assert result.load_n[forces.FR_WHEEL_INDEX] - coast.load_n[forces.FR_WHEEL_INDEX] == (
        pytest.approx(0.5 * front_transfer_n)
    )
    assert coast.load_n[forces.RL_WHEEL_INDEX] - result.load_n[forces.RL_WHEEL_INDEX] == (
        pytest.approx(0.5 * rear_transfer_n)
    )
    assert result.load_n[forces.RR_WHEEL_INDEX] - coast.load_n[forces.RR_WHEEL_INDEX] == (
        pytest.approx(0.5 * rear_transfer_n)
    )

    front_share = (0.5 * front_transfer_n) / (0.5 * front_transfer_n + 0.5 * rear_transfer_n)
    assert front_share == pytest.approx(
        config.roll_stiffness_front_fraction
        * float(config.axle_track_m[1])
        / (
            config.roll_stiffness_front_fraction * float(config.axle_track_m[1])
            + (1.0 - config.roll_stiffness_front_fraction) * float(config.axle_track_m[0])
        ),
        rel=1e-12,
    ), "the roll-moment split is by roll stiffness; the load split follows the tracks"
    assert float(config.axle_track_m[0]) != float(config.axle_track_m[1]), (
        "the two tracks differ, so a model that used one width for both axles cannot pass"
    )
    assert result.total_load_n == pytest.approx(coast.total_load_n, rel=1e-15)


def test_mirroring_the_lateral_acceleration_mirrors_the_corner_loads(
    config: KernelConfig,
) -> None:
    """``-ay`` is the same cornering mirrored: FL<->FR and RL<->RR, nothing else changed."""
    left = loads.step_loads(config, longitudinal_accel_m_s2=0.0, lateral_accel_m_s2=12.0)
    right = loads.step_loads(config, longitudinal_accel_m_s2=0.0, lateral_accel_m_s2=-12.0)
    for inside, outside in (
        (forces.FL_WHEEL_INDEX, forces.FR_WHEEL_INDEX),
        (forces.RL_WHEEL_INDEX, forces.RR_WHEEL_INDEX),
    ):
        assert left.load_n[outside] == pytest.approx(right.load_n[inside], rel=1e-15)
        assert left.load_n[inside] == pytest.approx(right.load_n[outside], rel=1e-15)
        assert left.travel_m[outside] == pytest.approx(right.travel_m[inside], rel=1e-15)
        assert left.travel_m[inside] == pytest.approx(right.travel_m[outside], rel=1e-15)
    assert left.total_load_n == pytest.approx(right.total_load_n, rel=1e-15)
    assert left.load_n[forces.FL_WHEEL_INDEX] != pytest.approx(left.load_n[forces.FR_WHEEL_INDEX])


def test_mirroring_the_longitudinal_acceleration_swaps_the_axles(config: KernelConfig) -> None:
    """``-ax`` is the mirror fore-and-aft, so the front axle sees what the rear axle saw."""
    ahead = loads.step_loads(config, longitudinal_accel_m_s2=9.0, lateral_accel_m_s2=0.0)
    braking = loads.step_loads(config, longitudinal_accel_m_s2=-9.0, lateral_accel_m_s2=0.0)
    coast = loads.step_loads(config, longitudinal_accel_m_s2=0.0, lateral_accel_m_s2=0.0)
    half_transfer_n = 0.5 * config.mass_kg * 9.0 * config.cg_height_m / _wheelbase(config)
    # The two axles do not carry the same load statically, so the mirroring claim is about the
    # *change*, not the absolute load: braking moves exactly what power moved, backwards.
    for front, rear in ((forces.FL_WHEEL_INDEX, forces.RL_WHEEL_INDEX),):
        assert ahead.load_n[rear] - coast.load_n[rear] == pytest.approx(half_transfer_n, rel=1e-12)
        assert braking.load_n[front] - coast.load_n[front] == pytest.approx(
            half_transfer_n, rel=1e-12
        )
        assert ahead.load_n[front] - coast.load_n[front] == pytest.approx(
            braking.load_n[rear] - coast.load_n[rear], rel=1e-12
        ), "power unloads the front by exactly what braking loads it"
        # Each axle has its own wheel rate, so the two travel figures are the same load swing
        # divided by different rates rather than equal numbers.
        assert ahead.travel_m[rear] == pytest.approx(
            half_transfer_n / float(config.axle_ride_rate_n_per_m[1]), rel=1e-15
        )
        assert braking.travel_m[front] == pytest.approx(
            half_transfer_n / float(config.axle_ride_rate_n_per_m[0]), rel=1e-15
        )
        assert ahead.travel_m[front] == pytest.approx(-braking.travel_m[front], rel=1e-12), (
            "the front axle's travel under power is the mirror of its travel under braking"
        )
    assert ahead.total_load_n == pytest.approx(braking.total_load_n, rel=1e-15)


def test_the_total_load_is_conserved_and_finite_across_an_extreme_sweep(
    config: KernelConfig,
) -> None:
    """Weight + aero + the vertical term, exactly, across accelerations the car cannot achieve.

    Invariant 3 is the load sum, and the interesting cases are the ones where the raw arithmetic
    asks for a negative load: conservation has to survive the clipping, not just the tidy interior
    of the envelope.
    """
    for longitudinal_m_s2 in (-40.0, -12.0, 0.0, 12.0, 40.0):
        for lateral_m_s2 in (-80.0, -30.0, 0.0, 30.0, 80.0):
            for downforce_n in (0.0, 12_000.0, 60_000.0):
                result = loads.step_loads(
                    config,
                    longitudinal_accel_m_s2=longitudinal_m_s2,
                    lateral_accel_m_s2=lateral_m_s2,
                    vertical_accel_m_s2=0.0,
                    aero_downforce_n=downforce_n,
                )
                expected = _total_load_n(config) + downforce_n
                assert all(math.isfinite(value) for value in result.load_n)
                assert all(value >= 0.0 for value in result.load_n)
                assert sum(result.load_n) == pytest.approx(expected, rel=1e-12, abs=1e-9)
                assert result.total_load_n == pytest.approx(expected, rel=1e-12, abs=1e-9)


def test_the_corner_loads_match_the_reference_arithmetic(config: KernelConfig) -> None:
    """The primitive computes exactly the documented formulae where no clipping intervenes."""
    for longitudinal_m_s2 in (-5.0, 0.0, 3.0):
        for lateral_m_s2 in (-4.0, 0.0, 2.5):
            result = loads.step_loads(
                config,
                longitudinal_accel_m_s2=longitudinal_m_s2,
                lateral_accel_m_s2=lateral_m_s2,
            )
            assert result.load_n == pytest.approx(
                _reference(config, longitudinal_m_s2, lateral_m_s2), rel=1e-14
            )


def test_lift_off_needs_more_than_four_g(config: KernelConfig) -> None:
    """The lift-off boundary is where the arithmetic says it is, not where a run happened to trip.

    Read from the configured geometry rather than quoted from a run: the front-left corner unloads
    at ``2 Fz_front ay h phi / t_f`` per newton of lateral acceleration, which is a number this
    project can compute, and pinning it means a change to the CG height or the roll split moves
    this test with the physics instead of leaving it as a coincidence.
    """
    weight_n = _total_load_n(config)
    front_corner_n = weight_n * config.cg_to_rear_axle_m / _wheelbase(config) / 2.0
    rate = (
        config.mass_kg
        * config.cg_height_m
        * config.roll_stiffness_front_fraction
        / (float(config.axle_track_m[0]) * 2.0)
    )
    boundary_m_s2 = front_corner_n / rate
    below = loads.step_loads(
        config, longitudinal_accel_m_s2=0.0, lateral_accel_m_s2=0.9 * boundary_m_s2
    )
    above = loads.step_loads(
        config, longitudinal_accel_m_s2=0.0, lateral_accel_m_s2=1.1 * boundary_m_s2
    )
    assert below.load_n[forces.FL_WHEEL_INDEX] > 0.0
    assert above.load_n[forces.FL_WHEEL_INDEX] == 0.0
    assert 4.0 * config.gravity_m_s2 < boundary_m_s2 < 5.0 * config.gravity_m_s2, (
        "the front inside wheel lifts between 4 and 5 g of lateral acceleration on the committed "
        "geometry, which is the range a downforce car is expected to reach"
    )


def test_a_corner_that_would_carry_a_negative_load_is_lifted_and_stays_at_zero(
    config: KernelConfig,
) -> None:
    """No wheel pushes on the road from below, and the total does not move when one leaves it.

    A corner at ``-2000 N`` was arithmetically *taking* load from its partner on the same axle, so
    lifting it relieves that partner: the whole axle load ends up on the wheel still down. A model
    that instead loaded the outside wheels further to "make up" the inside wheel would conserve the
    total too, and would be wrong - it would have a car lifting a wheel put more load on the ground
    than its own weight allows.
    """
    coast = loads.step_loads(config, longitudinal_accel_m_s2=0.0, lateral_accel_m_s2=0.0)
    lifted = loads.step_loads(
        config, longitudinal_accel_m_s2=0.0, lateral_accel_m_s2=LATERAL_LIFT_OFF_M_S2
    )
    assert lifted.load_n[forces.FL_WHEEL_INDEX] == 0.0
    assert lifted.load_n[forces.RL_WHEEL_INDEX] == 0.0
    assert lifted.lift_off == (forces.FL_WHEEL_INDEX, forces.RL_WHEEL_INDEX)
    assert lifted.total_load_n == pytest.approx(coast.total_load_n, rel=1e-12, abs=1e-9)
    assert not coast.lift_off

    roll_moment_nm = config.mass_kg * LATERAL_LIFT_OFF_M_S2 * config.cg_height_m
    front_transfer_n = (
        roll_moment_nm * config.roll_stiffness_front_fraction / float(config.axle_track_m[0])
    )
    rear_transfer_n = (
        roll_moment_nm
        * (1.0 - config.roll_stiffness_front_fraction)
        / float(config.axle_track_m[1])
    )
    front_axle_n = coast.load_n[forces.FL_WHEEL_INDEX] + coast.load_n[forces.FR_WHEEL_INDEX]
    rear_axle_n = coast.load_n[forces.RL_WHEEL_INDEX] + coast.load_n[forces.RR_WHEEL_INDEX]
    assert 0.5 * front_axle_n - 0.5 * front_transfer_n < 0.0, (
        "the front-left corner has to start from a genuinely negative raw load"
    )
    assert 0.5 * rear_axle_n - 0.5 * rear_transfer_n < 0.0, (
        "the rear-left corner has to start from a genuinely negative raw load"
    )

    # The axle total is unchanged by lateral transfer, so the loaded wheel carries all of it.
    assert lifted.load_n[forces.FR_WHEEL_INDEX] == pytest.approx(front_axle_n, rel=1e-15)
    assert lifted.load_n[forces.RR_WHEEL_INDEX] == pytest.approx(rear_axle_n, rel=1e-15)
    assert lifted.load_n[forces.FR_WHEEL_INDEX] > coast.load_n[forces.FR_WHEEL_INDEX]
    assert lifted.load_n[forces.RR_WHEEL_INDEX] > coast.load_n[forces.RR_WHEEL_INDEX]


def test_lift_off_that_empties_a_whole_axle_relieves_the_other_one(
    config: KernelConfig,
) -> None:
    """A longitudinal transfer larger than an axle's load has no partner left on that axle.

    Both front corners go negative together, so there is nothing on the front axle to relieve; the
    load comes off the rear axle instead, which is the only remaining place it can come from if
    the total is to be conserved.
    """
    accel_m_s2 = 12.0 * config.gravity_m_s2
    transfer_n = config.mass_kg * accel_m_s2 * config.cg_height_m / _wheelbase(config)
    front_axle_n = _total_load_n(config) * config.cg_to_rear_axle_m / _wheelbase(config)
    assert transfer_n > front_axle_n, "this case needs the transfer to exceed the front axle load"

    result = loads.step_loads(config, longitudinal_accel_m_s2=accel_m_s2, lateral_accel_m_s2=0.0)
    assert result.load_n[forces.FL_WHEEL_INDEX] == 0.0
    assert result.load_n[forces.FR_WHEEL_INDEX] == 0.0
    assert result.lift_off == (forces.FL_WHEEL_INDEX, forces.FR_WHEEL_INDEX)
    assert result.total_load_n == pytest.approx(_total_load_n(config), rel=1e-12, abs=1e-9)
    assert result.load_n[forces.RL_WHEEL_INDEX] > 0.0
    assert result.load_n[forces.RR_WHEEL_INDEX] > 0.0
    assert result.load_n[forces.RL_WHEEL_INDEX] == pytest.approx(
        result.load_n[forces.RR_WHEEL_INDEX]
    )


def test_travel_is_the_load_deviation_over_the_configured_wheel_rate(
    config: KernelConfig,
) -> None:
    """Travel is a quasi-static consequence of the load, not a separate state.

    Positive is compression, so the driven wheels compress under power and extend under braking,
    and the rate that converts one into the other is the configured per-axle wheel rate.
    """
    for wheel in range(forces.WHEEL_COUNT):
        assert loads.suspension_travel_m(
            2_100.0, 1_800.0, float(config.axle_ride_rate_n_per_m[wheel // 2])
        ) == pytest.approx(300.0 / float(config.axle_ride_rate_n_per_m[wheel // 2]), rel=1e-15)
        assert loads.suspension_travel_m(1_500.0, 1_800.0, 300_000.0) < 0.0, "droop is extension"

    accel_m_s2 = 8.0
    ahead = loads.step_loads(config, longitudinal_accel_m_s2=accel_m_s2, lateral_accel_m_s2=0.0)
    transfer_n = config.mass_kg * accel_m_s2 * config.cg_height_m / _wheelbase(config) / 2.0
    assert ahead.travel_m[forces.RL_WHEEL_INDEX] == pytest.approx(
        transfer_n / float(config.axle_ride_rate_n_per_m[1]), rel=1e-15
    )
    assert ahead.travel_m[forces.FL_WHEEL_INDEX] == pytest.approx(
        -transfer_n / float(config.axle_ride_rate_n_per_m[0]), rel=1e-15
    )
    assert ahead.travel_m[forces.RL_WHEEL_INDEX] > 0.0
    assert ahead.travel_m[forces.FL_WHEEL_INDEX] < 0.0
    assert ahead.travel_limited == (False, False, False, False)
    assert ahead.travel_m[forces.RL_WHEEL_INDEX] < float(config.axle_travel_limit_m[1]), (
        "the configured limit is wider than this transfer, so nothing is limited"
    )


def test_a_tight_travel_limit_is_reported_rather_than_silently_corrected(
    config: KernelConfig,
) -> None:
    """A declared limit narrower than the transfer is enforced, and what it cannot fix is reported.

    The committed limits (25 mm at 300-350 kN/m, so about 7.5-8.75 kN of load swing) are wider
    than the static corner load, so lift-off always arrives before the band on this
    configuration. The band is therefore exercised on the same geometry with a tighter declared
    limit rather than by replacing the physics to make the case reachable: what is under test is
    the behaviour at the limit, not a claim that the committed car reaches it. A purely lateral
    transfer clamps both wheels of an axle in opposite directions, so it needs no redistribution;
    a longitudinal one lifts the whole axle, which is the case that does.
    """
    tight = replace(config, axle_travel_limit_m=np.array([0.002, 0.002]))
    transfer_n = config.mass_kg * 20.0 * config.cg_height_m / _wheelbase(config) / 2.0
    assert transfer_n > 0.5 * float(tight.axle_travel_limit_m[0]) * float(
        tight.axle_ride_rate_n_per_m[0]
    ), "the declared transfer has to exceed the declared band for this case to mean anything"

    lateral_only = loads.step_loads(tight, longitudinal_accel_m_s2=0.0, lateral_accel_m_s2=6.0)
    assert not any(lateral_only.travel_limited), (
        "6 m/s2 lateral clamps each axle symmetrically, so no corner is left outside its band"
    )

    result = loads.step_loads(tight, longitudinal_accel_m_s2=20.0, lateral_accel_m_s2=0.0)
    assert any(result.travel_limited), (
        "both corners of each axle ran out of travel, and the model has to say so"
    )
    assert all(abs(travel) <= 0.002 for travel in result.travel_m), (
        "every corner is inside its declared band once the clamped-away load has been shared out"
    )
    assert result.total_load_n == pytest.approx(_total_load_n(config), rel=1e-12)
    assert all(value >= 0.0 for value in result.load_n)
    assert all(math.isfinite(value) for value in result.travel_m)


def test_the_configured_travel_limits_are_honoured_when_they_can_be(
    config: KernelConfig,
) -> None:
    """Where the band does not bind, travel is exactly the deviation over the rate."""
    tight = replace(config, axle_travel_limit_m=np.array([0.002, 0.002]))
    result = loads.step_loads(tight, longitudinal_accel_m_s2=1.0, lateral_accel_m_s2=0.5)
    assert not any(result.travel_limited), "1 m/s2 longitudinal is inside a 2 mm band"
    assert all(abs(travel) <= 0.002 for travel in result.travel_m)
    assert result.total_load_n == pytest.approx(_total_load_n(config), rel=1e-15)


def test_travel_clamping_redistributes_and_never_leaves_a_corner_below_zero(
    config: KernelConfig,
) -> None:
    """The clamped-away load is shared over the corners still carrying load, not discarded."""
    tight = replace(config, axle_travel_limit_m=np.array([0.002, 0.002]))
    for longitudinal_m_s2 in (-30.0, -20.0, 20.0, 30.0):
        for lateral_m_s2 in (-20.0, 0.0, 20.0):
            result = loads.step_loads(
                tight,
                longitudinal_accel_m_s2=longitudinal_m_s2,
                lateral_accel_m_s2=lateral_m_s2,
            )
            assert all(value >= 0.0 for value in result.load_n)
            assert sum(result.load_n) == pytest.approx(_total_load_n(config), rel=1e-12, abs=1e-9)
            assert all(math.isfinite(travel) for travel in result.travel_m)


def test_a_nonpositive_total_vertical_load_is_refused(config: KernelConfig) -> None:
    """A car whose tyres would have to pull it down is refused rather than distributed.

    ``az`` is upward-positive, so an ``az`` below ``-g`` asks for less than zero load in total.
    There is no distribution of a negative total, and quietly sharing it out would hand every
    corner a negative load - exactly the state the tyre model has to be protected from.
    """
    with pytest.raises(ValueError, match="total vertical load"):
        loads.step_loads(
            config,
            longitudinal_accel_m_s2=0.0,
            lateral_accel_m_s2=0.0,
            vertical_accel_m_s2=-1.01 * config.gravity_m_s2,
        )
    with pytest.raises(ValueError, match="total vertical load"):
        loads.step_loads(
            config,
            longitudinal_accel_m_s2=0.0,
            lateral_accel_m_s2=0.0,
            vertical_accel_m_s2=-config.gravity_m_s2,
            aero_downforce_n=0.0,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"longitudinal_accel_m_s2": math.nan, "lateral_accel_m_s2": 0.0},
        {"longitudinal_accel_m_s2": 0.0, "lateral_accel_m_s2": math.inf},
        {"longitudinal_accel_m_s2": -math.inf, "lateral_accel_m_s2": 0.0},
        {"longitudinal_accel_m_s2": "1.0", "lateral_accel_m_s2": 0.0},
        {"longitudinal_accel_m_s2": True, "lateral_accel_m_s2": 0.0},
        {"longitudinal_accel_m_s2": 0.0, "lateral_accel_m_s2": 0.0, "aero_downforce_n": math.nan},
        {"longitudinal_accel_m_s2": 0.0, "lateral_accel_m_s2": 0.0, "aero_downforce_n": -1.0},
        {
            "longitudinal_accel_m_s2": 0.0,
            "lateral_accel_m_s2": 0.0,
            "vertical_accel_m_s2": math.nan,
        },
    ],
)
def test_nonfinite_and_non_numeric_state_is_refused_at_the_boundary(
    config: KernelConfig, kwargs: dict[str, Any]
) -> None:
    """Nonfinite or non-numeric state never reaches ``boundscheck=False`` code."""
    with pytest.raises(ValueError):
        loads.step_loads(config, **kwargs)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mass_kg", 0.0, "mass_kg"),
        ("mass_kg", -1.0, "mass_kg"),
        ("gravity_m_s2", 0.0, "gravity_m_s2"),
        ("cg_to_front_axle_m", 0.0, "cg_to_front_axle_m"),
        ("cg_to_rear_axle_m", -0.5, "cg_to_rear_axle_m"),
        ("cg_height_m", 0.0, "cg_height_m"),
        ("roll_stiffness_front_fraction", 0.0, "roll_stiffness_front_fraction"),
        ("roll_stiffness_front_fraction", 1.0, "roll_stiffness_front_fraction"),
        ("roll_stiffness_front_fraction", 1.5, "roll_stiffness_front_fraction"),
        ("axle_track_m", np.array([1.6, 0.0]), "axle_track_m"),
        ("axle_track_m", np.array([1.6]), "axle_track_m"),
        ("axle_track_m", np.array([1.6, np.nan]), "axle_track_m"),
        ("axle_track_m", np.array([[1.6, 1.55]]), "axle_track_m"),
        ("axle_track_m", np.array([1.6, 1.55], dtype=np.float32), "axle_track_m"),
        ("axle_ride_rate_n_per_m", np.array([0.0, 350_000.0]), "axle_ride_rate_n_per_m"),
        ("axle_travel_limit_m", np.array([-0.025, 0.025]), "axle_travel_limit_m"),
    ],
)
def test_invalid_geometry_and_stiffness_configuration_is_refused(
    config: KernelConfig, field: str, value: Any, message: str
) -> None:
    """A replaceable public ``KernelConfig`` cannot carry a zero divisor into the kernel."""
    broken = replace(config, **{field: value})
    with pytest.raises(ValueError, match=message):
        loads.step_loads(broken, longitudinal_accel_m_s2=0.0, lateral_accel_m_s2=0.0)


# The six scalars the load model reads. Kept as a name here rather than repeated per test so a
# scalar added to the validator cannot be forgotten by the test that says the validator is wider
# than one caller's read.
_LOAD_SCALAR_NAMES = (
    "mass_kg",
    "gravity_m_s2",
    "cg_to_front_axle_m",
    "cg_to_rear_axle_m",
    "cg_height_m",
    "roll_stiffness_front_fraction",
)

# The three per-axle vectors the model reads, front first.
_AXLE_VECTOR_NAMES = ("axle_track_m", "axle_ride_rate_n_per_m", "axle_travel_limit_m")


def test_the_load_scalars_validator_hands_back_every_geometry_the_kernels_read(
    config: KernelConfig,
) -> None:
    """Narrowed scalars plus the caller-owned per-axle vectors, with nothing left to re-check.

    The primitives take ``float`` and ``float64`` arrays, so what crosses this boundary has to be
    exactly those types: a value that is only *equal* to the config's is not enough, because an
    ``int`` or a non-contiguous view reaching a ``boundscheck=False`` kernel is a different
    specialization or a silent copy rather than the number the model was validated against.
    """
    values = loads.validated_load_scalars(config, "test")

    for name in _LOAD_SCALAR_NAMES:
        value = getattr(values, name)
        assert type(value) is float, (
            f"{name} reached the kernel as {type(value).__name__}; the primitives are compiled "
            "against float and narrowing here is what keeps one specialization per call site"
        )
        assert value == getattr(config, name)

    for name in _AXLE_VECTOR_NAMES:
        array = getattr(values, name)
        assert array is getattr(config, name), (
            f"{name} is handed over rather than copied: the boundary allocates nothing, so a "
            "kernel reads the same caller-owned vector the config holds"
        )
        assert array.dtype == np.float64
        assert array.ndim == 1
        assert array.size == 2
        assert array.flags.c_contiguous
        assert np.isfinite(array).all()
        assert np.all(array > 0.0)


def test_the_validated_scalars_reproduce_the_loads_step_loads_computes(
    config: KernelConfig,
) -> None:
    """What the validator hands a kernel is what :func:`step_loads` fed its own kernels.

    ``step_loads`` goes through the shared boundary, and this drives the primitives from that same
    boundary's output. If the two disagreed, a kernel caller and the Python entry point would be
    reading the configuration through different rules - which is exactly the drift one shared
    validator exists to prevent, and the failure would only show up as a load case that nobody
    stepped through :func:`step_loads`.
    """
    values = loads.validated_load_scalars(config, "test")
    longitudinal_m_s2 = 9.0
    lateral_m_s2 = 4.0
    loads_n = np.zeros(4, dtype=np.float64)
    base_n = np.zeros(4, dtype=np.float64)
    limited_n = np.zeros(4, dtype=np.int64)

    loads.corner_loads_n(
        loads_n,
        base_n,
        values.mass_kg,
        values.gravity_m_s2,
        0.0,
        0.0,
        longitudinal_m_s2,
        lateral_m_s2,
        values.cg_to_front_axle_m,
        values.cg_to_rear_axle_m,
        values.cg_height_m,
        values.axle_track_m,
        values.roll_stiffness_front_fraction,
    )
    loads.clamp_travel_to_limits(
        loads_n,
        base_n,
        values.axle_ride_rate_n_per_m,
        values.axle_travel_limit_m,
        limited_n,
    )

    reference = loads.step_loads(
        config,
        longitudinal_accel_m_s2=longitudinal_m_s2,
        lateral_accel_m_s2=lateral_m_s2,
    )
    assert tuple(float(value) for value in loads_n) == pytest.approx(reference.load_n, rel=1e-15)
    assert tuple(float(value) for value in base_n) == pytest.approx(
        reference.base_load_n, rel=1e-15
    )
    assert tuple(bool(value) for value in limited_n) == reference.travel_limited


def test_the_load_scalars_validator_names_the_entry_point_that_refused(
    config: KernelConfig,
) -> None:
    """``prefix`` says *who* refused, because this boundary is shared by several callers.

    ``step_loads`` and a compiled kernel read the same fields, so ``mass_kg must be > 0`` is a
    worse error than one that says which caller was handed the impossible value.
    """
    broken = replace(config, cg_height_m=0.0)
    with pytest.raises(ValueError, match=r"kernel_boundary: config\.cg_height_m"):
        loads.validated_load_scalars(broken, "kernel_boundary")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mass_kg", 0.0),
        ("mass_kg", -1.0),
        ("mass_kg", math.nan),
        ("mass_kg", math.inf),
        ("mass_kg", "800.0"),
        ("mass_kg", True),
        ("gravity_m_s2", 0.0),
        ("gravity_m_s2", -9.80665),
        ("cg_to_front_axle_m", 0.0),
        ("cg_to_rear_axle_m", -0.5),
        ("cg_height_m", 0.0),
        ("cg_height_m", math.nan),
        ("roll_stiffness_front_fraction", 0.0),
        ("roll_stiffness_front_fraction", 1.0),
        ("roll_stiffness_front_fraction", 1.5),
        ("roll_stiffness_front_fraction", -0.1),
        ("roll_stiffness_front_fraction", math.nan),
    ],
)
def test_invalid_load_scalars_are_refused_by_the_shared_validator(
    config: KernelConfig, field: str, value: Any
) -> None:
    """Zero, negative, nonfinite and non-numeric geometry never reaches ``boundscheck=False``."""
    broken = replace(config, **{field: value})
    with pytest.raises(ValueError, match=field):
        loads.validated_load_scalars(broken, "test")


@pytest.mark.parametrize("field", _AXLE_VECTOR_NAMES)
@pytest.mark.parametrize(
    "value",
    [
        np.array([1.6]),
        np.array([1.6, 1.55, 1.5]),
        np.array([1.6, 1.55], dtype=np.float32),
        np.array([[1.6, 1.55]]),
        np.array([1.6, np.nan]),
        np.array([1.6, math.inf]),
        np.array([np.nan, 1.55]),
        np.array([1.6, 0.0]),
        np.array([1.6, -1.55]),
        [1.6, 1.55],
        np.zeros((2, 2), dtype=np.float64)[:, 0],
    ],
)
def test_a_per_axle_vector_the_kernel_could_read_past_its_end_is_refused(
    config: KernelConfig, field: str, value: Any
) -> None:
    """Length, dtype, layout, finiteness and sign, for each of the three per-axle vectors.

    With bounds checking off, a length-one or ``float32`` vector is an out-of-bounds read rather
    than an error, and a zero or negative entry is the zero denominator of a transfer, a travel or
    a band. The vectors are validated *together* rather than per kernel: ``axle_travel_limit_m`` is
    not read by ``corner_loads_n``, but it is read by the clamp in the same step, and a boundary
    that checked only what one call happened to touch would let a bad value sit in the object the
    next call reads.
    """
    broken = replace(config, **{field: value})
    with pytest.raises(ValueError, match=field):
        loads.validated_load_scalars(broken, "test")
