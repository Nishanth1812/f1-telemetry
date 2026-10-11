"""P2-T7: steering geometry - Ackermann, steering ratio, steering limit.

``PHASES.md`` P2-T7 and the phase-2 plan give the scenario input a
steering *wheel* contract and ask for the four road-wheel angles the
contact-patch kinematics read: a positive-left steering-wheel angle is
divided by the configured ratio into a mean front road-wheel angle, the
inner and outer front wheels split away from that mean by the ideal
Ackermann geometry of the wheelbase and front track, and the configured
``ackermann_fraction`` blends that ideal split with parallel steering.
The limit is a limit: past ``max_steering_wheel_angle_deg`` the boundary
rejects rather than clips. The claims are checked here in the form that
would actually fail:

* **The contract, term for term.** The boundary output for a sweep of
  left/right, small/large steering-wheel angles is compared against the
  documented arithmetic written out in plain Python, so a transposed
  argument in any of the primitives still compiles and still returns a
  plausible-looking pair.
* **Zero, parallel, ideal.** Exactly zero steering gives exactly zero
  angles everywhere; ``ackermann_fraction == 0`` is parallel steering
  (both fronts at the mean); ``== 1`` is the ideal Ackermann pair.
* **Cotangent relation.** The ideal pair satisfies
  ``cot(delta_outer) - cot(delta_inner) == front_track / wheelbase`` for
  both turn directions - the defining Ackermann relation, checked against
  geometry rather than prose.
* **Mirror symmetry.** Negating the input negates and swaps the two
  front angles and leaves the rear axle at zero: a model with one sign
  backwards passes the static tests and fails these.
* **Sign and corner order.** A positive-left input gives two positive
  front road-wheel angles with the inner (FL on a left turn) steeper
  than the outer (FR), and the rear pair exactly zero.
* **The limit is a limit.** Magnitudes past the configured maximum, and
  nonfinite inputs, are refused; nothing is clipped to look valid.
* **Invalid parameters.** A nonpositive steering ratio, a nonpositive
  maximum, an ``ackermann_fraction`` outside ``[0, 1]``, and a
  nonpositive wheelbase or track are all refused before arithmetic.
* **Geometric edge cases.** A mean road-wheel angle so large that the
  turn centre reaches the inner wheel keeps the denominator nonpositive
  territory finite: the inner angle pins to its geometric limit rather
  than producing a NaN, and the boundary's own validation keeps that
  territory out of a validated run.

Every physical number comes from the loaded ``car_spec.yaml``, so this
file contains no tuned constant. The ``forces`` marker keeps it with the
other physics-core tests. No lateral force is assembled here - that is
the tyre model's work - and no integration happens: the tests call the
compiled primitives directly, the way a kernel loop would.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from f1telemetry.contracts.car_spec import CarSpec

# The layer-isolation rule bans importing the physics core from layers 4-6
# (PLAN.md section 3). This file is the boundary test that proves the core
# itself, so the import is carved out the same way the relaxation and
# kinematics tests carve theirs out.
from f1telemetry.physics import steering  # noqa: TID251 -- the steering tests test the core

if TYPE_CHECKING:
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

_RAD_PER_DEG = math.pi / 180.0


@pytest.fixture(scope="module")
def config(spec: CarSpec) -> KernelConfig:
    """The validated, flat configuration the steering model is allowed to see."""
    return spec.kernel_config()


def _reference_angles_deg(
    steer_wheel_deg: float,
    steering_ratio: float,
    wheelbase_m: float,
    front_track_m: float,
    ackermann_fraction: float,
) -> tuple[float, float, float, float]:
    """The documented mapping in plain Python - a transcription, not a capture."""
    mean = steer_wheel_deg / steering_ratio
    if mean == 0.0:
        fl_ideal = 0.0
        fr_ideal = 0.0
    else:
        mag = abs(mean)
        radius = wheelbase_m / math.tan(mag * _RAD_PER_DEG)
        inner_radius = radius - 0.5 * front_track_m
        outer_radius = radius + 0.5 * front_track_m
        inner_mag = (
            90.0 if inner_radius <= 0.0 else math.atan(wheelbase_m / inner_radius) / _RAD_PER_DEG
        )
        outer_mag = math.atan(wheelbase_m / outer_radius) / _RAD_PER_DEG
        if mean > 0.0:
            fl_ideal, fr_ideal = inner_mag, outer_mag
        else:
            fl_ideal, fr_ideal = -outer_mag, -inner_mag
    fl = mean + ackermann_fraction * (fl_ideal - mean)
    fr = mean + ackermann_fraction * (fr_ideal - mean)
    return fl, fr, 0.0, 0.0


def test_the_steering_primitives_are_compiled_kernels_with_the_project_options(
    config: KernelConfig,
) -> None:
    """A Python fallback here would be a silent step backwards, so the dispatchers are asked."""
    steering.ideal_ackermann_front_angles_deg(
        10.0, config.wheelbase_m, float(config.axle_track_m[0])
    )
    out = np.zeros(4, dtype=np.float64)
    steering.road_wheel_angles_deg(
        120.0,
        config.steering_ratio,
        config.wheelbase_m,
        float(config.axle_track_m[0]),
        config.ackermann_fraction,
        out,
    )
    for fn in (steering.ideal_ackermann_front_angles_deg, steering.road_wheel_angles_deg):
        options = fn.targetoptions
        for key, expected in EXPECTED_OPTIONS.items():
            assert options.get(key) is expected, f"{fn.py_func.__name__} targetoptions[{key}]"


@pytest.mark.parametrize("steer_wheel_deg", [-320.0, -120.0, -5.0, 0.0, 5.0, 120.0, 320.0])
def test_the_boundary_matches_the_documented_arithmetic(
    config: KernelConfig, steer_wheel_deg: float
) -> None:
    """The full mapping, for left, right, small, large and zero inputs alike."""
    front_track = float(config.axle_track_m[0])
    got = steering.steering_angles_deg(config, steer_wheel_deg)
    expected = _reference_angles_deg(
        steer_wheel_deg,
        config.steering_ratio,
        config.wheelbase_m,
        front_track,
        config.ackermann_fraction,
    )
    assert got.shape == (4,)
    assert got.dtype == np.float64
    for actual, reference in zip(got, expected, strict=True):
        assert actual == pytest.approx(reference, rel=1e-12, abs=1e-12)


def test_zero_input_is_exactly_zero_everywhere(config: KernelConfig) -> None:
    """No steer means no road-wheel angle, on any axle, bit for bit."""
    got = steering.steering_angles_deg(config, 0.0)
    np.testing.assert_array_equal(got, np.zeros(4))


def test_parallel_limit_puts_both_front_wheels_at_the_mean(config: KernelConfig) -> None:
    """ackermann_fraction == 0 is parallel steering: FL == FR == wheel / ratio."""
    parallel = replace(config, ackermann_fraction=0.0)
    for steer_wheel_deg in (-240.0, -7.5, 0.0, 7.5, 240.0):
        fl, fr, rl, rr = steering.steering_angles_deg(parallel, steer_wheel_deg)
        mean = steer_wheel_deg / config.steering_ratio
        assert fl == pytest.approx(mean, rel=1e-15, abs=1e-15)
        assert fr == pytest.approx(mean, rel=1e-15, abs=1e-15)
        assert (rl, rr) == (0.0, 0.0)


@pytest.mark.parametrize("steer_wheel_deg", [-200.0, -30.0, 30.0, 200.0])
def test_ideal_ackermann_satisfies_the_cotangent_relation(
    config: KernelConfig, steer_wheel_deg: float
) -> None:
    """cot(delta_outer) - cot(delta_inner) == track / wheelbase, both turn ways."""
    ideal = replace(config, ackermann_fraction=1.0)
    fl, fr, _, _ = steering.steering_angles_deg(ideal, steer_wheel_deg)
    if steer_wheel_deg > 0.0:
        inner_deg, outer_deg = fl, fr
    else:
        inner_deg, outer_deg = fr, fl
    relation = 1.0 / math.tan(abs(outer_deg) * _RAD_PER_DEG) - 1.0 / math.tan(
        abs(inner_deg) * _RAD_PER_DEG
    )
    expected = float(config.axle_track_m[0]) / config.wheelbase_m
    assert relation == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_ideal_ackermann_inner_is_steeper_and_both_carry_the_input_sign(
    config: KernelConfig,
) -> None:
    """A left turn: FL inner and steeper, FR outer, both positive (positive-left)."""
    ideal = replace(config, ackermann_fraction=1.0)
    fl, fr, rl, rr = steering.steering_angles_deg(ideal, 120.0)
    assert fl > fr > 0.0
    assert (rl, rr) == (0.0, 0.0)
    fl_r, fr_r, rl_r, rr_r = steering.steering_angles_deg(ideal, -120.0)
    assert fr_r < fl_r < 0.0
    assert (rl_r, rr_r) == (0.0, 0.0)


@pytest.mark.parametrize("steer_wheel_deg", [-360.0, -120.0, -12.0, 0.0, 12.0, 120.0, 360.0])
def test_mirror_symmetry_is_exact(config: KernelConfig, steer_wheel_deg: float) -> None:
    """angles(-d) == -(FR, FL, RR, RL) of angles(d): the mirror of a left turn is a right turn."""
    left = steering.steering_angles_deg(config, steer_wheel_deg)
    right = steering.steering_angles_deg(config, -steer_wheel_deg)
    np.testing.assert_allclose(
        right,
        np.array([-left[1], -left[0], -left[3], -left[2]]),
        rtol=0.0,
        atol=0.0,
    )


def test_the_rear_angles_are_always_zero(config: KernelConfig) -> None:
    """No rear steer in this model: every input leaves RL and RR at zero."""
    for steer_wheel_deg in (-400.0, -1.0, 0.0, 1.0, 400.0):
        out = steering.steering_angles_deg(config, steer_wheel_deg)
        assert out[2] == 0.0
        assert out[3] == 0.0


def test_the_limit_rejects_rather_than_clips(config: KernelConfig) -> None:
    """A steering-wheel angle at the configured maximum is legal; one past it is an error."""
    limit = config.max_steering_wheel_angle_deg
    steering.steering_angles_deg(config, limit)
    steering.steering_angles_deg(config, -limit)
    for bad in (limit + 1e-9, -(limit + 1e-9), limit * 2.0):
        with pytest.raises(ValueError, match="max_steering_wheel_angle"):
            steering.steering_angles_deg(config, bad)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_input_is_rejected(config: KernelConfig, bad: float) -> None:
    """A NaN steering-wheel angle is a rejected command, never a silently propagated one."""
    with pytest.raises(ValueError, match="finite"):
        steering.steering_angles_deg(config, bad)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, "120", None])
def test_non_numeric_or_nonfinite_input_is_rejected(config: KernelConfig, bad: Any) -> None:
    """The boundary refuses anything that is not a real finite number."""
    with pytest.raises(ValueError, match=r"real number|finite"):
        steering.steering_angles_deg(config, bad)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("steering_ratio", 0.0, "steering_ratio"),
        ("steering_ratio", -12.0, "steering_ratio"),
        ("steering_ratio", math.nan, "steering_ratio"),
        ("max_steering_wheel_angle_deg", 0.0, "max_steering_wheel_angle_deg"),
        ("max_steering_wheel_angle_deg", -400.0, "max_steering_wheel_angle_deg"),
        ("ackermann_fraction", -0.1, "ackermann_fraction"),
        ("ackermann_fraction", 1.1, "ackermann_fraction"),
        ("ackermann_fraction", math.nan, "ackermann_fraction"),
        ("wheelbase_m", 0.0, "wheelbase_m"),
        ("wheelbase_m", -3.4, "wheelbase_m"),
        ("wheelbase_m", math.inf, "wheelbase_m"),
    ],
)
def test_invalid_parameters_are_refused(
    config: KernelConfig, field: str, value: float, match: str
) -> None:
    """Every configured scalar the model reads is range-checked before arithmetic."""
    broken = replace(config, **{field: value})
    with pytest.raises(ValueError, match=match):
        steering.steering_angles_deg(broken, 120.0)


def test_invalid_front_track_is_refused(config: KernelConfig) -> None:
    """A zero or negative front track would put the inner wheel outside the turn centre."""
    for bad_track in (0.0, -1.6, math.nan):
        broken_tracks = np.array([bad_track, config.axle_track_m[1]], dtype=np.float64)
        broken = replace(config, axle_track_m=broken_tracks)
        with pytest.raises(ValueError, match="axle_track_m"):
            steering.steering_angles_deg(broken, 120.0)


def test_the_geometric_denominator_edge_stays_finite() -> None:
    """A mean road-wheel angle whose turn centre reaches the inner wheel cannot NaN the pair.

    Ideal Ackermann needs ``radius_m - track/2 > 0``. Past
    ``atan(wheelbase / (track/2))`` that denominator is zero or negative,
    and the inner angle pins to its geometric limit of 90 degrees rather
    than flipping sign or returning NaN. The boundary's own limit keeps a
    validated run far from here; this pins the primitive's behaviour if a
    caller sweeps past it directly.
    """
    wheelbase = 3.4
    track = 1.6
    fl, fr = steering.ideal_ackermann_front_angles_deg(80.0, wheelbase, track)
    assert math.isfinite(fl)
    assert math.isfinite(fr)
    assert fl == pytest.approx(90.0, abs=1e-12)
    assert 0.0 < fr < 90.0
    fl_r, fr_r = steering.ideal_ackermann_front_angles_deg(-80.0, wheelbase, track)
    assert fr_r == pytest.approx(-90.0, abs=1e-12)
    assert -90.0 < fl_r < 0.0


def test_large_but_in_limit_angles_are_finite_and_ordered(config: KernelConfig) -> None:
    """The biggest legal steering-wheel angle still gives a sane, ordered pair."""
    limit = config.max_steering_wheel_angle_deg
    fl, fr, rl, rr = steering.steering_angles_deg(config, limit)
    assert all(math.isfinite(v) for v in (fl, fr, rl, rr))
    assert fl > fr > 0.0
    assert (rl, rr) == (0.0, 0.0)


def test_the_compiled_step_writes_a_caller_owned_buffer_in_place(config: KernelConfig) -> None:
    """The kernel primitive returns nothing; it writes the caller's float64 buffer."""
    out = np.full(4, -1.0, dtype=np.float64)
    before = out
    result = steering.road_wheel_angles_deg(
        120.0,
        config.steering_ratio,
        config.wheelbase_m,
        float(config.axle_track_m[0]),
        config.ackermann_fraction,
        out,
    )
    assert result is None
    assert out is before
    again = np.full(4, -1.0, dtype=np.float64)
    steering.road_wheel_angles_deg(
        120.0,
        config.steering_ratio,
        config.wheelbase_m,
        float(config.axle_track_m[0]),
        config.ackermann_fraction,
        again,
    )
    np.testing.assert_array_equal(out, again)


def test_equivalent_int_and_float_ratio_inputs_share_one_answer(config: KernelConfig) -> None:
    """An int steering-wheel angle and the same value as float map to the same angles."""
    a = steering.steering_angles_deg(config, 120)
    b = steering.steering_angles_deg(config, 120.0)
    np.testing.assert_array_equal(a, b)


def test_step_matches_the_boundary_composition(config: KernelConfig) -> None:
    """step_road_wheel_angles runs the same model as steering_angles_deg, in place."""
    for steer_wheel_deg in (-320.0, -30.0, 0.0, 30.0, 320.0):
        out = np.full(4, -1.0, dtype=np.float64)
        result = steering.step_road_wheel_angles(config, steer_wheel_deg, out)
        assert result is None
        np.testing.assert_array_equal(out, steering.steering_angles_deg(config, steer_wheel_deg))


def test_step_writes_the_caller_owned_buffer_in_place(config: KernelConfig) -> None:
    """The caller's vector is updated, never replaced or reallocated."""
    out = np.zeros(4, dtype=np.float64)
    before = out
    steering.step_road_wheel_angles(config, 120.0, out)
    assert out is before
    fl, fr, rl, rr = out
    assert fl > fr > 0.0
    assert (rl, rr) == (0.0, 0.0)


def test_step_rejects_out_of_limit_and_nonfinite_scalars(config: KernelConfig) -> None:
    """The scalar rules are the boundary's rules: refused, never clipped."""
    out = np.zeros(4, dtype=np.float64)
    limit = config.max_steering_wheel_angle_deg
    steering.step_road_wheel_angles(config, limit, out)
    steering.step_road_wheel_angles(config, -limit, out)
    bad_values: tuple[Any, ...] = (
        limit + 1e-9,
        -(limit + 1e-9),
        math.nan,
        math.inf,
        -math.inf,
        True,
        "120",
        None,
    )
    for bad in bad_values:
        with pytest.raises(ValueError):
            steering.step_road_wheel_angles(config, bad, out)


@pytest.mark.parametrize(
    "bad_out",
    [
        np.zeros(4, dtype=np.float32),  # wrong dtype
        np.zeros(3, dtype=np.float64),  # wrong length
        np.zeros((2, 2), dtype=np.float64),  # wrong ndim
        np.zeros(8, dtype=np.float64)[::2],  # non-contiguous
        [0.0, 0.0, 0.0, 0.0],  # not an ndarray
    ],
    ids=["float32", "length-3", "2d", "strided", "list"],
)
def test_step_rejects_a_bad_output_buffer(config: KernelConfig, bad_out: Any) -> None:
    """A wrong-shape, wrong-dtype, or strided buffer would silently write the wrong state."""
    with pytest.raises(ValueError, match=r"float64|C-contiguous"):
        steering.step_road_wheel_angles(config, 120.0, bad_out)


def test_step_rejects_a_read_only_output_buffer(config: KernelConfig) -> None:
    out = np.zeros(4, dtype=np.float64)
    out.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        steering.step_road_wheel_angles(config, 120.0, out)


def test_step_reads_the_config_outside_the_loop(config: KernelConfig) -> None:
    """Config validation happens on the boundary, not per compiled call."""
    broken = replace(config, steering_ratio=0.0)
    out = np.zeros(4, dtype=np.float64)
    with pytest.raises(ValueError, match="steering_ratio"):
        steering.step_road_wheel_angles(broken, 120.0, out)
