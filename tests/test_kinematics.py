"""P2-T3: contact-patch kinematics, tested against its claims.

``PHASES.md`` P2-T3's lateral work needs the slip angle the tyre model reads,
and this module owns the kinematics that produce it: the contact patch's
velocity from the body velocity, yaw rate and corner position; the rotation of
that velocity into road-wheel coordinates for positive-left steering; and the
slip angle itself, in degrees, positive when it drives a positive leftward
``Fy`` in :func:`~f1telemetry.physics.tyres.tyre_lateral_force`. The claims
are checked here in the form that would actually fail:

* **The formula, term for term.** Each step is compared against the same
  arithmetic written out in plain Python, over sweeps that include zero,
  negative and rearward values, so a transposed argument in any of the
  primitives still compiles and still returns a plausible-looking angle.
* **Zero steer, straight motion.** Every corner of the committed geometry
  reports exactly zero slip angle, and the contact velocity is exactly the
  body velocity.
* **Steering sign.** A left-steered wheel rolling straight reports a positive
  slip angle equal to its steer angle, a right-steered wheel the exact
  negative - and, wired into the real lateral tyre model, positive slip
  produces positive leftward ``Fy`` and negative slip the opposite. That is
  the sign contract the ``slip_angle`` channel declares, checked against the
  consumer rather than asserted in prose.
* **Yaw.** A left yaw rate signs the front and rear axles oppositely (the
  front axle is dragged left of its wheels' heading, the rear trails right),
  matches the rigid-body formula at every corner, matches the first-order
  bicycle-model values ``-r*arm/vx`` and ``+r*arm/vx`` in degrees, slows the
  inside wheel and gives both corners of one axle the same lateral velocity.
* **Mirrored left/right corners.** Mirrored steering at mirrored corners is
  the exact negative slip angle, and the full mirror symmetry - mirrored
  motion, mirrored corner, mirrored steer - holds exactly over a sweep,
  including yaw and sideslip. A model with one sign backwards passes the
  static tests and fails these.
* **Finite low-speed handling, without a hidden parameter.** A standing car
  reports exactly zero slip angle whatever its steering; the slip angle is
  scale-free (a velocity direction is a velocity direction at 1e-9 m/s as at
  30 m/s, which is what an epsilon-guarded denominator would break); pure
  sideslip at zero forward speed is a finite right angle; and every finite
  input produces a finite angle.

Every physical number comes from the loaded ``car_spec.yaml``, so this file
contains no tuned constant. The ``forces`` marker keeps it with the other
physics-core tests. No force is assembled here - that is the tyre model's
work - and no integration happens: the tests call the compiled primitives
directly, the way a kernel loop would.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import pytest

from f1telemetry.contracts.car_spec import CarSpec

# The layer-isolation rule bans importing the physics core from layers 4-6
# (PLAN.md section 3). This file is the boundary test that proves the core
# itself, so the import is carved out the same way the kernel's is. `tyres`
# is here because the slip-angle sign contract is checked against the consumer.
from f1telemetry.physics import (  # noqa: TID251 -- the kinematics tests test the core
    kinematics,
    tyres,
)

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


@pytest.fixture(scope="module")
def config(spec: CarSpec) -> KernelConfig:
    """The validated, flat configuration the kinematics tests read geometry from."""
    return spec.kernel_config()


def _corners(config: KernelConfig) -> dict[str, tuple[float, float]]:
    """The four contact-patch positions, CG-relative, in the ``FL, FR, RL, RR`` order.

    Built from the configured CG arms and per-axle tracks - the same geometry
    P2-T2's load model reads - so the tests never restate a wheelbase or a
    track width of their own.
    """
    half_front = float(config.axle_track_m[0]) / 2.0
    half_rear = float(config.axle_track_m[1]) / 2.0
    return {
        "FL": (config.cg_to_front_axle_m, half_front),
        "FR": (config.cg_to_front_axle_m, -half_front),
        "RL": (-config.cg_to_rear_axle_m, half_rear),
        "RR": (-config.cg_to_rear_axle_m, -half_rear),
    }


def _lateral_force(config: KernelConfig, slip_angle_deg: float) -> float:
    """``tyres.tyre_lateral_force`` with the configured coefficients, zero camber.

    The sign contract between kinematics and the tyre model is checked through
    the real consumer at the reference load, where the response is well inside
    its linear range and the sign is the whole story.
    """
    return tyres.tyre_lateral_force(
        slip_angle_deg,
        0.0,
        config.load_sensitivity_reference_n,
        config.lateral_pacejka_b,
        config.lateral_pacejka_c,
        config.lateral_pacejka_e,
        config.lateral_pacejka_mu,
        config.load_sensitivity_reference_n,
        config.load_sensitivity_peak,
        config.load_sensitivity_stiffness,
        config.camber_stiffness_n_per_deg,
    )


# A sweep that crosses zero in every input, includes a reversing car, a
# standing car and a crawl, and puts the corner on both sides of the CG. The
# states where a sign or a term silently stops mattering are the states the
# reference transcription has to survive.
_RIGID_BODY_SWEEP = (
    (20.0, 0.0, 0.0, 1.836, 0.8),
    (20.0, 1.5, 0.3, 1.836, -0.8),
    (5.0, -2.0, -0.8, -1.564, 0.775),
    (0.5, 0.25, 1.5, 0.0, 0.0),
    (-3.0, 1.0, -0.2, -1.564, -0.775),
    (0.0, 0.0, 0.0, 1.836, 0.8),
    (1e-9, 1e-9, 1e-9, 1.0, 1.0),
)

# A sweep for the composed slip angle: the same shape, plus a steering angle,
# spanning left and right lock, sideslip, both turn directions and reverse.
_SLIP_SWEEP = (
    (20.0, 0.0, 0.0, 1.836, 0.8, 0.0),
    (20.0, 0.0, 0.0, 1.836, 0.8, 15.0),
    (20.0, 0.0, 0.0, 1.836, -0.8, -15.0),
    (20.0, 1.5, 0.3, 1.836, -0.8, 12.0),
    (5.0, -2.0, -0.8, -1.564, 0.775, 25.0),
    (0.5, 0.25, 1.5, 0.0, 0.0, -8.0),
    (-3.0, 1.0, -0.2, -1.564, -0.775, -5.0),
    (1e-9, 1e-9, 1e-9, 1.0, 1.0, 30.0),
)


def test_the_kinematics_primitives_are_compiled_kernels_with_the_project_options(
    config: KernelConfig,
) -> None:
    """A Python fallback here would be a silent step backwards, so the dispatchers are asked.

    What this pins is that each primitive is a compiled dispatcher with the
    project's options - ``fastmath`` off, ``boundscheck`` off,
    ``error_model="numpy"`` - and that a call registered a signature rather
    than falling back to Python. ``cache=True`` is *not* asserted here: numba
    consumes it at decoration time, so it cannot be read back off
    ``targetoptions``, exactly as ``tests/test_forces.py`` records.
    """
    kinematics.contact_velocity_m_s(20.0, 0.0, 0.0, 1.0, 0.5)
    kinematics.wheel_frame_velocity_m_s(20.0, 0.0, 10.0)
    kinematics.slip_angle_deg(20.0, 0.0)
    kinematics.contact_slip_angle_deg(20.0, 0.0, 0.0, 1.0, 0.5, 10.0)
    for function in (
        kinematics.contact_velocity_m_s,
        kinematics.wheel_frame_velocity_m_s,
        kinematics.slip_angle_deg,
        kinematics.contact_slip_angle_deg,
    ):
        name = function.py_func.__name__
        assert len(function.signatures) >= 1, f"{name} never compiled a signature"
        options = dict(function.targetoptions)
        for option, expected in EXPECTED_OPTIONS.items():
            assert options[option] == expected, (name, option)


def test_zero_steer_straight_motion_gives_zero_slip_at_every_corner(
    config: KernelConfig,
) -> None:
    """Rolling straight with centred steering: no corner moves anywhere but forward.

    The contact velocity is exactly the body velocity at every corner of the
    committed geometry, a zero steer rotation leaves it unchanged, and the
    slip angle is exactly zero - not approximately, because no transcendental
    is involved once the state is zero.
    """
    for corner_x_m, corner_y_m in _corners(config).values():
        patch_vx_m_s, patch_vy_m_s = kinematics.contact_velocity_m_s(
            20.0, 0.0, 0.0, corner_x_m, corner_y_m
        )
        assert (patch_vx_m_s, patch_vy_m_s) == (20.0, 0.0)
        wheel_vx_m_s, wheel_vy_m_s = kinematics.wheel_frame_velocity_m_s(
            patch_vx_m_s, patch_vy_m_s, 0.0
        )
        assert (wheel_vx_m_s, wheel_vy_m_s) == (20.0, 0.0)
        assert kinematics.slip_angle_deg(wheel_vx_m_s, wheel_vy_m_s) == 0.0
        assert kinematics.contact_slip_angle_deg(20.0, 0.0, 0.0, corner_x_m, corner_y_m, 0.0) == 0.0


@pytest.mark.parametrize("steer_deg", [-33.0, -15.0, -5.0, -0.5, 0.5, 5.0, 15.0, 33.0])
def test_road_wheel_steer_sets_the_slip_angle(config: KernelConfig, steer_deg: float) -> None:
    """Straight motion: the slip angle *is* the road-wheel steer angle, positive left.

    With no sideslip and no yaw, the contact patch moves exactly along the
    body's forward axis, so the angle between that direction and the wheel's
    heading is the steer angle itself - the geometric limit the linear tyre
    range is built on, and the reason a left steer must produce a positive
    slip angle for the car to turn left.
    """
    corner_x_m, corner_y_m = _corners(config)["FL"]
    alpha_deg = kinematics.contact_slip_angle_deg(20.0, 0.0, 0.0, corner_x_m, corner_y_m, steer_deg)
    assert alpha_deg == pytest.approx(steer_deg, rel=1e-12)


def test_a_positive_slip_angle_drives_a_positive_leftward_lateral_force(
    config: KernelConfig,
) -> None:
    """The sign contract, checked against the consumer rather than asserted in prose.

    The ``slip_angle`` channel declares "positive generating +y force", and
    :func:`~f1telemetry.physics.tyres.tyre_lateral_force` is the function that
    has to mean it: every slip angle this module produces, fed into the real
    lateral model at the reference load with zero camber, must return a
    lateral force with the same sign. A kinematics module that flips the sign
    passes every geometric test above and fails this one.
    """
    for vx_m_s, vy_m_s, yaw_rate_rad_s, steer_deg in (
        (20.0, 0.0, 0.0, 8.0),
        (20.0, 0.0, 0.0, -8.0),
        (20.0, 0.0, math.radians(15.0), 12.0),
        (20.0, 1.5, math.radians(-10.0), -5.0),
        (5.0, -2.0, math.radians(30.0), 20.0),
    ):
        corner_x_m, corner_y_m = _corners(config)["RL"]
        alpha_deg = kinematics.contact_slip_angle_deg(
            vx_m_s, vy_m_s, yaw_rate_rad_s, corner_x_m, corner_y_m, steer_deg
        )
        force_n = _lateral_force(config, alpha_deg)
        assert math.copysign(1.0, force_n) == math.copysign(1.0, alpha_deg)
    # The magnitudes are real too, so the sign is not the only thing being
    # checked: a 5-degree slip at the reference load buys over half the peak
    # force, not a rounding error.
    corner_x_m, corner_y_m = _corners(config)["FL"]
    alpha_deg = kinematics.contact_slip_angle_deg(20.0, 0.0, 0.0, corner_x_m, corner_y_m, 5.0)
    half_peak_n = 0.5 * config.lateral_pacejka_mu * config.load_sensitivity_reference_n
    assert _lateral_force(config, alpha_deg) > half_peak_n


def test_a_left_yaw_rate_signs_the_front_and_rear_axles_oppositely(
    config: KernelConfig,
) -> None:
    """A left turn drags the front axle left and trails the rear axle right.

    With centred steering and no sideslip, a positive (left) yaw rate makes
    the front contact patches move to the *left* of their wheels' heading -
    a negative slip angle, because the patch slides left and the road pushes
    right - and the rear patches move to the *right* of theirs, a positive
    slip angle. Those are the signs the steady-state cornering model is built
    on, and a yaw-rate sign backwards here would turn the car into itself.
    """
    vx_m_s = 20.0
    yaw_rate_rad_s = math.radians(10.0)
    corners = _corners(config)
    for name in ("FL", "FR"):
        corner_x_m, corner_y_m = corners[name]
        alpha_deg = kinematics.contact_slip_angle_deg(
            vx_m_s, 0.0, yaw_rate_rad_s, corner_x_m, corner_y_m, 0.0
        )
        assert alpha_deg < 0.0, f"{name} should slip negative in a left turn"
    for name in ("RL", "RR"):
        corner_x_m, corner_y_m = corners[name]
        alpha_deg = kinematics.contact_slip_angle_deg(
            vx_m_s, 0.0, yaw_rate_rad_s, corner_x_m, corner_y_m, 0.0
        )
        assert alpha_deg > 0.0, f"{name} should slip positive in a left turn"
    # A right turn (negative yaw rate) flips every sign.
    for corner_x_m, corner_y_m in corners.values():
        left_turn_deg = kinematics.contact_slip_angle_deg(
            vx_m_s, 0.0, yaw_rate_rad_s, corner_x_m, corner_y_m, 0.0
        )
        right_turn_deg = kinematics.contact_slip_angle_deg(
            vx_m_s, 0.0, -yaw_rate_rad_s, corner_x_m, corner_y_m, 0.0
        )
        assert math.copysign(1.0, right_turn_deg) == -math.copysign(1.0, left_turn_deg)


def test_yaw_slip_angles_match_the_first_order_cornering_values(
    config: KernelConfig,
) -> None:
    """The small-angle values are ``-r*arm/vx`` at the front and ``+r*arm/vx`` at the rear.

    This is the kinematic bicycle model's steady-turn slip angle, the quantity
    every understeer gradient is computed from, checked to a few percent. The
    residue is the second-order ``r*y/vx`` term in the longitudinal component,
    which is real physics - the inside wheel of a turn moves slower - rather
    than an approximation error worth hiding.
    """
    vx_m_s = 20.0
    yaw_rate_rad_s = math.radians(10.0)
    first_order_front_deg = -math.degrees(yaw_rate_rad_s * config.cg_to_front_axle_m / vx_m_s)
    first_order_rear_deg = math.degrees(yaw_rate_rad_s * config.cg_to_rear_axle_m / vx_m_s)
    corners = _corners(config)
    for name in ("FL", "FR"):
        corner_x_m, corner_y_m = corners[name]
        alpha_deg = kinematics.contact_slip_angle_deg(
            vx_m_s, 0.0, yaw_rate_rad_s, corner_x_m, corner_y_m, 0.0
        )
        assert alpha_deg == pytest.approx(first_order_front_deg, rel=0.02)
    for name in ("RL", "RR"):
        corner_x_m, corner_y_m = corners[name]
        alpha_deg = kinematics.contact_slip_angle_deg(
            vx_m_s, 0.0, yaw_rate_rad_s, corner_x_m, corner_y_m, 0.0
        )
        assert alpha_deg == pytest.approx(first_order_rear_deg, rel=0.02)


@pytest.mark.parametrize(
    "vx_m_s,vy_m_s,yaw_rate_rad_s,corner_x_m,corner_y_m",
    _RIGID_BODY_SWEEP,
)
def test_contact_velocity_is_the_rigid_body_formula(
    vx_m_s: float,
    vy_m_s: float,
    yaw_rate_rad_s: float,
    corner_x_m: float,
    corner_y_m: float,
) -> None:
    """``v_patch = v_body + omega x r``, written out in plain Python.

    The reference is the same arithmetic in the same order, so a difference is
    a difference in the model rather than in the order of two floating-point
    operations. The sweep includes a standing car, a reversing car and a car
    moving a nanometre per second, because those are the states where a sign
    or a term silently stops mattering.
    """
    expected = (
        vx_m_s - yaw_rate_rad_s * corner_y_m,
        vy_m_s + yaw_rate_rad_s * corner_x_m,
    )
    assert kinematics.contact_velocity_m_s(
        vx_m_s, vy_m_s, yaw_rate_rad_s, corner_x_m, corner_y_m
    ) == pytest.approx(expected, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize(
    "vx_m_s,vy_m_s,steer_deg",
    [
        (20.0, 0.0, 0.0),
        (20.0, 1.5, 10.0),
        (5.0, -2.0, -25.0),
        (0.0, 3.0, 15.0),
        (-3.0, 1.0, 8.0),
        (0.0, 0.0, 30.0),
    ],
)
def test_wheel_frame_velocity_is_the_documented_rotation(
    vx_m_s: float, vy_m_s: float, steer_deg: float
) -> None:
    """The positive-left rotation matrix, written out in plain Python.

    The wheel frame is the body frame rotated by the steer angle, so the
    velocity's components in it are the body components rotated the same way:
    ``(vx cos d + vy sin d, -vx sin d + vy cos d)`` for a positive-left
    steer angle ``d``.
    """
    delta = math.radians(steer_deg)
    expected = (
        vx_m_s * math.cos(delta) + vy_m_s * math.sin(delta),
        -vx_m_s * math.sin(delta) + vy_m_s * math.cos(delta),
    )
    assert kinematics.wheel_frame_velocity_m_s(vx_m_s, vy_m_s, steer_deg) == (
        pytest.approx(expected[0], rel=1e-12, abs=1e-15),
        pytest.approx(expected[1], rel=1e-12, abs=1e-15),
    )


def test_wheel_frame_rotation_special_angles() -> None:
    """Identity at zero steer, a quarter turn, and an exact round trip back."""
    # Zero steer is the identity.
    assert kinematics.wheel_frame_velocity_m_s(17.0, -4.0, 0.0) == pytest.approx(
        (17.0, -4.0), rel=1e-12
    )
    # A left quarter turn: the wheel's x axis points body-left, so body
    # forward maps to wheel -y and body left maps to wheel +x.
    assert kinematics.wheel_frame_velocity_m_s(12.0, 0.0, 90.0) == pytest.approx(
        (0.0, -12.0), rel=1e-9
    )
    assert kinematics.wheel_frame_velocity_m_s(0.0, 5.0, 90.0) == pytest.approx(
        (5.0, 0.0), rel=1e-12
    )
    # Rotating into the wheel frame and back out again is the identity.
    wheel_vx_m_s, wheel_vy_m_s = kinematics.wheel_frame_velocity_m_s(12.0, -5.0, 18.0)
    assert kinematics.wheel_frame_velocity_m_s(wheel_vx_m_s, wheel_vy_m_s, -18.0) == pytest.approx(
        (12.0, -5.0), rel=1e-12
    )


@pytest.mark.parametrize(
    "vx_wheel_m_s,vy_wheel_m_s",
    [
        (20.0, 0.0),
        (20.0, 1.5),
        (20.0, -1.5),
        (0.5, 3.0),
        (0.0, 5.0),
        (0.0, -5.0),
        (0.0, 0.0),
        (-3.0, 1.0),
        (1e-12, -1e-12),
    ],
)
def test_slip_angle_is_the_angle_between_heading_and_motion(
    vx_wheel_m_s: float, vy_wheel_m_s: float
) -> None:
    """``atan2(-v_wy, v_wx)`` in degrees, written out in plain Python.

    The sweep includes a patch with no velocity at all, one moving exactly
    sideways, and one moving rearward, because those are where a guarded
    ratio would divide by zero and an unguarded one would return NaN.
    """
    expected = math.degrees(math.atan2(-vy_wheel_m_s, vx_wheel_m_s))
    assert kinematics.slip_angle_deg(vx_wheel_m_s, vy_wheel_m_s) == pytest.approx(
        expected, rel=1e-12
    )


def test_the_slip_angle_is_a_principal_value() -> None:
    """The angle lives in ``[-180, 180]`` degrees, whatever the motion."""
    for vx_wheel_m_s, vy_wheel_m_s in (
        (20.0, 0.0),
        (-20.0, 0.0),
        (0.0, 5.0),
        (0.0, -5.0),
        (1e-9, -1e-9),
        (-1e-9, 1e-9),
    ):
        alpha_deg = kinematics.slip_angle_deg(vx_wheel_m_s, vy_wheel_m_s)
        assert -180.0 <= alpha_deg <= 180.0


def test_pure_sideslip_at_zero_forward_speed_is_a_finite_right_angle() -> None:
    """A patch sliding sideways with no forward speed reports a finite +/-90 degrees.

    No guard is needed to keep this finite: ``atan2`` answers the angle
    directly, and a contact patch moving exactly leftward of its wheel's
    heading is slipping at a right angle in whichever direction it slides.
    """
    assert kinematics.slip_angle_deg(0.0, 5.0) == pytest.approx(-90.0)
    assert kinematics.slip_angle_deg(0.0, -5.0) == pytest.approx(90.0)


@pytest.mark.parametrize(
    "vx_m_s,vy_m_s,yaw_rate_rad_s,corner_x_m,corner_y_m,steer_deg",
    _SLIP_SWEEP,
)
def test_the_composed_slip_angle_is_the_chained_primitives(
    vx_m_s: float,
    vy_m_s: float,
    yaw_rate_rad_s: float,
    corner_x_m: float,
    corner_y_m: float,
    steer_deg: float,
) -> None:
    """One call through ``contact_slip_angle_deg`` is the three primitives in order.

    The composition has to be the chain - contact velocity, wheel-frame
    rotation, slip angle - rather than a second implementation of it, because
    a kernel loop calls the composition and a test calls the chain, and the
    two must agree on every state including a standing and a reversing car.
    """
    patch_vx_m_s, patch_vy_m_s = kinematics.contact_velocity_m_s(
        vx_m_s, vy_m_s, yaw_rate_rad_s, corner_x_m, corner_y_m
    )
    wheel_vx_m_s, wheel_vy_m_s = kinematics.wheel_frame_velocity_m_s(
        patch_vx_m_s, patch_vy_m_s, steer_deg
    )
    expected = kinematics.slip_angle_deg(wheel_vx_m_s, wheel_vy_m_s)
    assert kinematics.contact_slip_angle_deg(
        vx_m_s, vy_m_s, yaw_rate_rad_s, corner_x_m, corner_y_m, steer_deg
    ) == pytest.approx(expected, rel=1e-12)


def test_the_cg_is_its_own_corner(config: KernelConfig) -> None:
    """A corner at the CG reduces to the body slip angle, exactly.

    With no yaw arm and no lateral offset, the contact velocity *is* the body
    velocity whatever the yaw rate, so the slip angle is the angle between
    the body's motion and the wheel's heading - the definition the whole
    module is built on, and a check that neither ``r`` term in the rigid-body
    formula is transposed, because a nonzero yaw rate here must change
    nothing.
    """
    vx_m_s, vy_m_s, yaw_rate_rad_s = 20.0, 3.0, math.radians(40.0)
    for steer_deg in (-20.0, 0.0, 20.0):
        alpha_deg = kinematics.contact_slip_angle_deg(
            vx_m_s, vy_m_s, yaw_rate_rad_s, 0.0, 0.0, steer_deg
        )
        wheel_vx_m_s, wheel_vy_m_s = kinematics.wheel_frame_velocity_m_s(vx_m_s, vy_m_s, steer_deg)
        expected = math.degrees(math.atan2(-wheel_vy_m_s, wheel_vx_m_s))
        assert alpha_deg == pytest.approx(expected, rel=1e-12)


def test_yaw_moves_the_inside_wheel_slower_and_drags_the_front_axle_left(
    config: KernelConfig,
) -> None:
    """The physical sign checks, on the committed geometry.

    In a left turn the left wheels are the inside wheels: their longitudinal
    contact velocity is *below* the body's, the right wheels' is above it,
    and the two front corners - one axle, one yaw arm - gain exactly the same
    leftward velocity. A differential is what lets the inside wheel turn
    slower; the kinematics only have to report it.
    """
    vx_m_s = 20.0
    yaw_rate_rad_s = math.radians(15.0)
    corners = _corners(config)
    front_left = kinematics.contact_velocity_m_s(vx_m_s, 0.0, yaw_rate_rad_s, *corners["FL"])
    front_right = kinematics.contact_velocity_m_s(vx_m_s, 0.0, yaw_rate_rad_s, *corners["FR"])
    rear_left = kinematics.contact_velocity_m_s(vx_m_s, 0.0, yaw_rate_rad_s, *corners["RL"])
    rear_right = kinematics.contact_velocity_m_s(vx_m_s, 0.0, yaw_rate_rad_s, *corners["RR"])
    # Inside (left) wheels slower, outside (right) wheels faster.
    assert front_left[0] < vx_m_s < front_right[0]
    assert rear_left[0] < vx_m_s < rear_right[0]
    # One axle, one yaw arm: both corners gain the same lateral velocity.
    assert front_left[1] == front_right[1]
    assert front_left[1] == pytest.approx(yaw_rate_rad_s * config.cg_to_front_axle_m)
    assert rear_left[1] == rear_right[1]
    assert rear_left[1] == pytest.approx(-yaw_rate_rad_s * config.cg_to_rear_axle_m)


def test_mirrored_steer_at_mirrored_corners_mirrors_the_slip_angle(
    config: KernelConfig,
) -> None:
    """A left turn and a right turn are mirror images, and the slip angles say so.

    Straight motion, the left corner steered left and the right corner
    steered right by the same angle: the two slip angles are exact negatives,
    which is the symmetry invariant 5 (longitudinal/lateral symmetry) is
    checked against, and the reason a mirrored setup must produce mirrored
    behaviour.
    """
    vx_m_s = 20.0
    corners = _corners(config)
    for left_name, right_name in (("FL", "FR"), ("RL", "RR")):
        left_x_m, left_y_m = corners[left_name]
        right_x_m, right_y_m = corners[right_name]
        for steer_deg in (0.5, 8.0, 25.0):
            alpha_left_deg = kinematics.contact_slip_angle_deg(
                vx_m_s, 0.0, 0.0, left_x_m, left_y_m, steer_deg
            )
            alpha_right_deg = kinematics.contact_slip_angle_deg(
                vx_m_s, 0.0, 0.0, right_x_m, right_y_m, -steer_deg
            )
            assert alpha_right_deg == pytest.approx(-alpha_left_deg, rel=1e-12)


@pytest.mark.parametrize(
    "vx_m_s,vy_m_s,yaw_rate_rad_s,steer_deg",
    [
        (20.0, 0.0, 0.0, 10.0),
        (20.0, 1.5, 0.3, -12.0),
        (5.0, -2.0, -0.8, 25.0),
        (0.5, 0.25, 1.5, 0.0),
        (-3.0, 1.0, -0.2, -5.0),
        (1e-6, 1e-6, 1e-6, 30.0),
    ],
)
def test_the_full_mirror_symmetry_holds(
    vx_m_s: float, vy_m_s: float, yaw_rate_rad_s: float, steer_deg: float
) -> None:
    """Mirroring the motion, the corner and the steer negates the slip angle, exactly.

    The mirror of a left-hand turn is a right-hand turn: ``vy`` and the yaw
    rate change sign, the corner's lateral position changes sign, and the
    steer changes sign. Everything else - the longitudinal velocity and the
    corner's forward position - does not. Under that reflection the wheel's
    longitudinal velocity is unchanged and its lateral velocity negates, so
    the slip angle must negate exactly. This is the sharpest sign test in the
    file: one backwards term anywhere in the chain breaks it.
    """
    corner_x_m, corner_y_m = 1.836, 0.8
    alpha_deg = kinematics.contact_slip_angle_deg(
        vx_m_s, vy_m_s, yaw_rate_rad_s, corner_x_m, corner_y_m, steer_deg
    )
    alpha_mirrored_deg = kinematics.contact_slip_angle_deg(
        vx_m_s, -vy_m_s, -yaw_rate_rad_s, corner_x_m, -corner_y_m, -steer_deg
    )
    assert alpha_mirrored_deg == pytest.approx(-alpha_deg, rel=1e-12)


def test_a_standing_car_has_no_slip_angle_whatsoever_the_steer(
    config: KernelConfig,
) -> None:
    """No patch motion, no slip: a standing tyre is not slipping, steered or not.

    This is the finite low-speed answer, and it is exact zero rather than a
    small number, because the slip angle is an angle between two directions
    and a patch with no velocity has none to be between.
    """
    for corner_x_m, corner_y_m in _corners(config).values():
        for steer_deg in (-450.0, -30.0, 0.0, 30.0, 450.0):
            assert (
                kinematics.contact_slip_angle_deg(0.0, 0.0, 0.0, corner_x_m, corner_y_m, steer_deg)
                == 0.0
            )


def test_the_slip_angle_is_scale_free_because_it_is_an_angle(
    config: KernelConfig,
) -> None:
    """A velocity direction is a direction at any speed, including far below any guard.

    An implementation that guarded its denominator with ``max(vx, eps)`` would
    change the answer once the speed dropped below ``eps`` - a hidden parameter
    deciding the physics at walking pace. This pins the answer at a billionth
    of a metre per second against the answer at full speed for the same
    velocity *state* (every component scaled together, so the direction is
    unchanged): the slip angle has to agree, because nothing in it may depend
    on how fast the car happens to be going.
    """
    corner_x_m, corner_y_m = _corners(config)["FL"]
    vx_m_s, vy_m_s, yaw_rate_rad_s, steer_deg = 30.0, 2.0, math.radians(20.0), 8.0
    full_speed_deg = kinematics.contact_slip_angle_deg(
        vx_m_s, vy_m_s, yaw_rate_rad_s, corner_x_m, corner_y_m, steer_deg
    )
    crawl_deg = kinematics.contact_slip_angle_deg(
        vx_m_s * 1e-9,
        vy_m_s * 1e-9,
        yaw_rate_rad_s * 1e-9,
        corner_x_m,
        corner_y_m,
        steer_deg,
    )
    assert crawl_deg == pytest.approx(full_speed_deg, rel=1e-9)


def test_every_finite_input_gives_a_finite_slip_angle(config: KernelConfig) -> None:
    """Finite in, finite out - including a standing car, a reversing car and full lock.

    The sweep crosses zero in every input (the states where a guarded
    denominator would divide by zero or an unguarded one would return NaN),
    spans the ``yaw_rate`` and ``steering_angle`` channel ranges, and includes
    a negative longitudinal speed, because a simulator meets all of them.
    """
    corner_x_m, corner_y_m = _corners(config)["RR"]
    for vx_m_s in (0.0, 1e-12, 0.5, 30.0, -8.0):
        for vy_m_s in (0.0, -1.5, 3.0):
            for yaw_rate_deg_s in (-300.0, 0.0, 150.0):
                for steer_deg in (-450.0, 0.0, 450.0):
                    alpha_deg = kinematics.contact_slip_angle_deg(
                        vx_m_s,
                        vy_m_s,
                        math.radians(yaw_rate_deg_s),
                        corner_x_m,
                        corner_y_m,
                        steer_deg,
                    )
                    assert math.isfinite(alpha_deg)
