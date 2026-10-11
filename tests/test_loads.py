"""P2-T2: four-corner vertical loads and the quasi-static suspension travel they imply.

Every physical number comes from the loaded ``car_spec.yaml``, so this file holds no tuned
constant. ``step_loads`` takes the previous step's accelerations as inputs rather than estimating
or iterating on them, so each test states the accelerations it applies.
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

FL, FR, RL, RR = (
    forces.FL_WHEEL_INDEX,
    forces.FR_WHEEL_INDEX,
    forces.RL_WHEEL_INDEX,
    forces.RR_WHEEL_INDEX,
)

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
# reaches zero at about 4.1 g and the rear-left at about 6.4 g, so 8 g lifts both.
LATERAL_LIFT_OFF_M_S2 = 8.0 * 9.80665


@pytest.fixture(scope="module")
def config(spec: CarSpec) -> KernelConfig:
    return spec.kernel_config()


def _total_load_n(config: KernelConfig) -> float:
    return config.mass_kg * config.gravity_m_s2


def _wheelbase(config: KernelConfig) -> float:
    # Not config.wheelbase_m: the arms are derived from the wheelbase and the static split, and on
    # this data their sum differs from the file's wheelbase in the last bit. The model divides by
    # the sum, so the comparison has to as well.
    return config.cg_to_front_axle_m + config.cg_to_rear_axle_m


def _lateral_transfers_n(config: KernelConfig, lateral_accel_m_s2: float) -> tuple[float, float]:
    """Full left-to-right load difference on the front and then the rear axle."""
    roll_moment_nm = config.mass_kg * lateral_accel_m_s2 * config.cg_height_m
    front = roll_moment_nm * config.roll_stiffness_front_fraction / float(config.axle_track_m[0])
    rear = (
        roll_moment_nm
        * (1.0 - config.roll_stiffness_front_fraction)
        / float(config.axle_track_m[1])
    )
    return front, rear


def _reference(
    config: KernelConfig, longitudinal_accel_m_s2: float, lateral_accel_m_s2: float
) -> tuple[float, float, float, float]:
    # Same operations in the same order as the primitive, so a mismatch is a difference in the
    # model rather than two floating-point roundings. Transcribed from the documented formulae,
    # not captured from a run, which could only prove the code still does what it did.
    total = _total_load_n(config)
    wheelbase = _wheelbase(config)
    front_axle = total * config.cg_to_rear_axle_m / wheelbase
    rear_axle = total - front_axle
    longitudinal = config.mass_kg * longitudinal_accel_m_s2 * config.cg_height_m / wheelbase
    front_lateral, rear_lateral = _lateral_transfers_n(config, lateral_accel_m_s2)
    return (
        0.5 * (front_axle - longitudinal) - 0.5 * front_lateral,
        0.5 * (front_axle - longitudinal) + 0.5 * front_lateral,
        0.5 * (rear_axle + longitudinal) - 0.5 * rear_lateral,
        0.5 * (rear_axle + longitudinal) + 0.5 * rear_lateral,
    )


def test_the_load_primitives_are_compiled_kernels_with_the_project_options(
    config: KernelConfig,
) -> None:
    # A Python fallback here would be a silent step backwards, so the dispatchers are asked.
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
    # P1 read front_weight_fraction straight from car_spec.yaml, so if the P2 arms disagree with it
    # the two models would disagree about which axle is heavier while both looked plausible.
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
    # Positive az is upward in the z-up frame, so it adds to the load the tyres carry.
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
    # dFz = m ax h / L, forward to the rear axle, on every wheel of that axle.
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
    for wheel in (FL, FR):
        assert coast.load_n[wheel] - ahead.load_n[wheel] == pytest.approx(0.5 * transfer_n)
    for wheel in (RL, RR):
        assert ahead.load_n[wheel] - coast.load_n[wheel] == pytest.approx(0.5 * transfer_n)

    assert ahead.load_n[RL] > coast.load_n[RL]
    assert ahead.load_n[FL] < coast.load_n[FL]
    assert braking.load_n[FL] > coast.load_n[FL], (
        "under braking the load moves forward, onto the axle that has not been asked to slow down"
    )
    assert ahead.total_load_n == pytest.approx(coast.total_load_n, rel=1e-15)
    assert braking.total_load_n == pytest.approx(coast.total_load_n, rel=1e-15)


def test_lateral_acceleration_moves_load_to_the_right_and_splits_it_by_roll_stiffness(
    config: KernelConfig,
) -> None:
    # ay > 0 is a leftward acceleration, so load moves to the right-hand wheels. A sign error here
    # leaves the loads finite, symmetric and conserved, and the car simply understeers into corners.
    accel_m_s2 = 10.0
    result = loads.step_loads(config, longitudinal_accel_m_s2=0.0, lateral_accel_m_s2=accel_m_s2)
    coast = loads.step_loads(config, longitudinal_accel_m_s2=0.0, lateral_accel_m_s2=0.0)

    roll_moment_nm = config.mass_kg * accel_m_s2 * config.cg_height_m
    assert loads.lateral_roll_moment_nm(config.mass_kg, accel_m_s2, config.cg_height_m) == (
        pytest.approx(roll_moment_nm, rel=1e-15)
    )
    front_transfer_n, rear_transfer_n = _lateral_transfers_n(config, accel_m_s2)

    assert coast.load_n[FL] - result.load_n[FL] == pytest.approx(0.5 * front_transfer_n), (
        "the front-left corner is the inside of a left turn and unloads"
    )
    assert result.load_n[FR] - coast.load_n[FR] == pytest.approx(0.5 * front_transfer_n)
    assert coast.load_n[RL] - result.load_n[RL] == pytest.approx(0.5 * rear_transfer_n)
    assert result.load_n[RR] - coast.load_n[RR] == pytest.approx(0.5 * rear_transfer_n)

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
    # -ay is the same cornering mirrored: FL<->FR and RL<->RR, nothing else changed.
    left = loads.step_loads(config, longitudinal_accel_m_s2=0.0, lateral_accel_m_s2=12.0)
    right = loads.step_loads(config, longitudinal_accel_m_s2=0.0, lateral_accel_m_s2=-12.0)
    for inside, outside in ((FL, FR), (RL, RR)):
        assert left.load_n[outside] == pytest.approx(right.load_n[inside], rel=1e-15)
        assert left.load_n[inside] == pytest.approx(right.load_n[outside], rel=1e-15)
        assert left.travel_m[outside] == pytest.approx(right.travel_m[inside], rel=1e-15)
        assert left.travel_m[inside] == pytest.approx(right.travel_m[outside], rel=1e-15)
    assert left.total_load_n == pytest.approx(right.total_load_n, rel=1e-15)
    assert left.load_n[FL] != pytest.approx(left.load_n[FR])


def test_mirroring_the_longitudinal_acceleration_swaps_the_axles(config: KernelConfig) -> None:
    # -ax is the mirror fore-and-aft, so the front axle sees what the rear axle saw.
    ahead = loads.step_loads(config, longitudinal_accel_m_s2=9.0, lateral_accel_m_s2=0.0)
    braking = loads.step_loads(config, longitudinal_accel_m_s2=-9.0, lateral_accel_m_s2=0.0)
    coast = loads.step_loads(config, longitudinal_accel_m_s2=0.0, lateral_accel_m_s2=0.0)
    half_transfer_n = 0.5 * config.mass_kg * 9.0 * config.cg_height_m / _wheelbase(config)

    # The axles do not carry the same load statically, so the claim is about the change, not the
    # absolute load: braking moves exactly what power moved, backwards.
    assert ahead.load_n[RL] - coast.load_n[RL] == pytest.approx(half_transfer_n, rel=1e-12)
    assert braking.load_n[FL] - coast.load_n[FL] == pytest.approx(half_transfer_n, rel=1e-12)
    assert ahead.load_n[FL] - coast.load_n[FL] == pytest.approx(
        braking.load_n[RL] - coast.load_n[RL], rel=1e-12
    ), "power unloads the front by exactly what braking loads it"
    # Each axle has its own wheel rate, so the travel figures are one load swing over two rates.
    assert ahead.travel_m[RL] == pytest.approx(
        half_transfer_n / float(config.axle_ride_rate_n_per_m[1]), rel=1e-15
    )
    assert braking.travel_m[FL] == pytest.approx(
        half_transfer_n / float(config.axle_ride_rate_n_per_m[0]), rel=1e-15
    )
    assert ahead.travel_m[FL] == pytest.approx(-braking.travel_m[FL], rel=1e-12), (
        "the front axle's travel under power is the mirror of its travel under braking"
    )
    assert ahead.total_load_n == pytest.approx(braking.total_load_n, rel=1e-15)


def test_the_total_load_is_conserved_and_finite_across_an_extreme_sweep(
    config: KernelConfig,
) -> None:
    # The interesting cases are the ones where the raw arithmetic asks for a negative load:
    # conservation has to survive the clipping, not just the tidy interior of the envelope.
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
    # The primitive computes exactly the documented formulae where no clipping intervenes.
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
    # Read from the configured geometry rather than quoted from a run: the front-left corner
    # unloads at 2 Fz_front ay h phi / t_f per newton of lateral acceleration, so a change to the CG
    # height or the roll split moves this test with the physics.
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
    assert below.load_n[FL] > 0.0
    assert above.load_n[FL] == 0.0
    assert 4.0 * config.gravity_m_s2 < boundary_m_s2 < 5.0 * config.gravity_m_s2, (
        "the front inside wheel lifts between 4 and 5 g of lateral acceleration on the committed "
        "geometry, which is the range a downforce car is expected to reach"
    )


def test_a_corner_that_would_carry_a_negative_load_is_lifted_and_stays_at_zero(
    config: KernelConfig,
) -> None:
    # A corner at -2000 N was arithmetically taking load from its partner on the same axle, so
    # lifting it relieves that partner. A model that loaded the outside wheels further instead would
    # conserve the total and still be wrong: a lifted wheel would put more load on the ground than
    # the car's own weight allows.
    coast = loads.step_loads(config, longitudinal_accel_m_s2=0.0, lateral_accel_m_s2=0.0)
    lifted = loads.step_loads(
        config, longitudinal_accel_m_s2=0.0, lateral_accel_m_s2=LATERAL_LIFT_OFF_M_S2
    )
    assert lifted.load_n[FL] == 0.0
    assert lifted.load_n[RL] == 0.0
    assert lifted.lift_off == (FL, RL)
    assert lifted.total_load_n == pytest.approx(coast.total_load_n, rel=1e-12, abs=1e-9)
    assert not coast.lift_off

    front_transfer_n, rear_transfer_n = _lateral_transfers_n(config, LATERAL_LIFT_OFF_M_S2)
    front_axle_n = coast.load_n[FL] + coast.load_n[FR]
    rear_axle_n = coast.load_n[RL] + coast.load_n[RR]
    assert 0.5 * front_axle_n - 0.5 * front_transfer_n < 0.0, (
        "the front-left corner has to start from a genuinely negative raw load"
    )
    assert 0.5 * rear_axle_n - 0.5 * rear_transfer_n < 0.0, (
        "the rear-left corner has to start from a genuinely negative raw load"
    )

    # The axle total is unchanged by lateral transfer, so the loaded wheel carries all of it.
    assert lifted.load_n[FR] == pytest.approx(front_axle_n, rel=1e-15)
    assert lifted.load_n[RR] == pytest.approx(rear_axle_n, rel=1e-15)
    assert lifted.load_n[FR] > coast.load_n[FR]
    assert lifted.load_n[RR] > coast.load_n[RR]


def test_lift_off_that_empties_a_whole_axle_relieves_the_other_one(
    config: KernelConfig,
) -> None:
    # Both front corners go negative together, so the load comes off the rear axle instead: the only
    # place it can come from if the total is to be conserved.
    accel_m_s2 = 12.0 * config.gravity_m_s2
    transfer_n = config.mass_kg * accel_m_s2 * config.cg_height_m / _wheelbase(config)
    front_axle_n = _total_load_n(config) * config.cg_to_rear_axle_m / _wheelbase(config)
    assert transfer_n > front_axle_n, "this case needs the transfer to exceed the front axle load"

    result = loads.step_loads(config, longitudinal_accel_m_s2=accel_m_s2, lateral_accel_m_s2=0.0)
    assert result.load_n[FL] == 0.0
    assert result.load_n[FR] == 0.0
    assert result.lift_off == (FL, FR)
    assert result.total_load_n == pytest.approx(_total_load_n(config), rel=1e-12, abs=1e-9)
    assert result.load_n[RL] > 0.0
    assert result.load_n[RR] > 0.0
    assert result.load_n[RL] == pytest.approx(result.load_n[RR])


def test_travel_is_the_load_deviation_over_the_configured_wheel_rate(
    config: KernelConfig,
) -> None:
    # Positive is compression: the driven wheels compress under power and extend under braking, and
    # the configured per-axle wheel rate converts one into the other.
    for wheel in range(forces.WHEEL_COUNT):
        assert loads.suspension_travel_m(
            2_100.0, 1_800.0, float(config.axle_ride_rate_n_per_m[wheel // 2])
        ) == pytest.approx(300.0 / float(config.axle_ride_rate_n_per_m[wheel // 2]), rel=1e-15)
        assert loads.suspension_travel_m(1_500.0, 1_800.0, 300_000.0) < 0.0, "droop is extension"

    accel_m_s2 = 8.0
    ahead = loads.step_loads(config, longitudinal_accel_m_s2=accel_m_s2, lateral_accel_m_s2=0.0)
    transfer_n = config.mass_kg * accel_m_s2 * config.cg_height_m / _wheelbase(config) / 2.0
    assert ahead.travel_m[RL] == pytest.approx(
        transfer_n / float(config.axle_ride_rate_n_per_m[1]), rel=1e-15
    )
    assert ahead.travel_m[FL] == pytest.approx(
        -transfer_n / float(config.axle_ride_rate_n_per_m[0]), rel=1e-15
    )
    assert ahead.travel_m[RL] > 0.0
    assert ahead.travel_m[FL] < 0.0
    assert ahead.travel_limited == (False, False, False, False)
    assert ahead.travel_m[RL] < float(config.axle_travel_limit_m[1]), (
        "the configured limit is wider than this transfer, so nothing is limited"
    )


def test_a_tight_travel_limit_is_reported_rather_than_silently_corrected(
    config: KernelConfig,
) -> None:
    # The committed 25 mm bands are wider than the static corner load, so lift-off always comes
    # first on this car. The band is exercised with a tighter declared limit on the same geometry,
    # not by changing the physics to make it reachable. A lateral transfer clamps the two wheels of
    # an axle in opposite directions and needs no redistribution; a longitudinal one lifts the whole
    # axle, which is the case that does.
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
    tight = replace(config, axle_travel_limit_m=np.array([0.002, 0.002]))
    result = loads.step_loads(tight, longitudinal_accel_m_s2=1.0, lateral_accel_m_s2=0.5)
    assert not any(result.travel_limited), "1 m/s2 longitudinal is inside a 2 mm band"
    assert all(abs(travel) <= 0.002 for travel in result.travel_m)
    assert result.total_load_n == pytest.approx(_total_load_n(config), rel=1e-15)


def test_travel_clamping_redistributes_and_never_leaves_a_corner_below_zero(
    config: KernelConfig,
) -> None:
    # The clamped-away load is shared over the corners still carrying load, not discarded.
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
    # az is upward-positive, so an az below -g asks for less than zero load in total. Sharing out a
    # negative total would hand every corner a negative load, which the tyre model must never see.
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
    # Nonfinite or non-numeric state never reaches boundscheck=False code.
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
    # A replaceable public KernelConfig cannot carry a zero divisor into the kernel.
    broken = replace(config, **{field: value})
    with pytest.raises(ValueError, match=message):
        loads.step_loads(broken, longitudinal_accel_m_s2=0.0, lateral_accel_m_s2=0.0)


# Names rather than repeated per test, so a scalar added to the validator is checked here too.
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
    # The primitives are compiled against float and float64 arrays. An int or a non-contiguous view
    # reaching a boundscheck=False kernel is a different specialization or a silent copy, so
    # "equal to the config" is not enough.
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
    # step_loads goes through the shared boundary, and this drives the primitives from that same
    # boundary's output. If the two disagreed, a kernel caller and the Python entry point would read
    # the configuration by different rules, and the drift would only surface in a load case nobody
    # stepped through step_loads.
    values = loads.validated_load_scalars(config, "test")
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
        9.0,
        4.0,
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

    reference = loads.step_loads(config, longitudinal_accel_m_s2=9.0, lateral_accel_m_s2=4.0)
    assert tuple(float(value) for value in loads_n) == pytest.approx(reference.load_n, rel=1e-15)
    assert tuple(float(value) for value in base_n) == pytest.approx(
        reference.base_load_n, rel=1e-15
    )
    assert tuple(bool(value) for value in limited_n) == reference.travel_limited


def test_the_load_scalars_validator_names_the_entry_point_that_refused(
    config: KernelConfig,
) -> None:
    # prefix says who refused, because step_loads and a compiled kernel share this boundary: a
    # message naming only mass_kg would not say which caller was handed the impossible value.
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
    # With bounds checking off, a short or float32 vector is an out-of-bounds read rather than an
    # error, and a zero or negative entry is a zero divisor. The three vectors are validated
    # together because the clamp in the same step reads axle_travel_limit_m even when
    # corner_loads_n does not, so a bad value must not sit unchecked in the next call's input.
    broken = replace(config, **{field: value})
    with pytest.raises(ValueError, match=field):
        loads.validated_load_scalars(broken, "test")
