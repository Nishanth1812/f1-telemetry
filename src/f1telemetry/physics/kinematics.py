"""P2-T3: contact-patch kinematics - the slip angle the lateral tyre model reads.

``PLAN.md`` section 4 puts four Pacejka tyres on the car, and every one of
them asks the same kinematic question first: *which way is this contact patch
actually moving, relative to the direction its wheel is pointing?* This module
answers that question and nothing else. It is the kinematics between the
chassis state (``vx``, ``vy``, yaw rate) and the slip-angle input of
:func:`~f1telemetry.physics.tyres.tyre_lateral_force`, computed per corner
from the corner's CG-relative position. No force is assembled here - that is
the tyre model's work - no corner position is derived from configuration (the
caller owns the geometry, which is P2-T2's and P2-T7's to supply), and no
state is kept: every function is a pure function of its arguments, so the same
inputs give the same bits.

**The three steps**, each one function, so a caller - and a compiled kernel
loop - can stop at whichever level it needs:

1. :func:`contact_velocity_m_s` - the contact patch's velocity in body
   coordinates, from the rigid-body relation
   ``v_patch = v_body + omega x r_patch``. With the body frame ``x`` forward,
   ``y`` left, ``z`` up and right-handed, and the yaw rate taken about the
   vertical axis positive for a *left* turn (the nose rotating toward ``+y``),
   that is ``(vx - r*y, vy + r*x)``.
2. :func:`wheel_frame_velocity_m_s` - that velocity rotated into the
   road-wheel's own coordinates for positive-left steering: the wheel's ``x``
   axis lies along the steered heading and its ``y`` axis left of it, so a
   positive steer rotates the frame the same way the wheel turns.
3. :func:`slip_angle_deg` - the signed angle between the wheel's forward
   direction and the direction the patch moves, positive when the patch moves
   to the *right* of the wheel's heading, because that is the deformation
   direction that makes the road push the patch - and the car - leftward. It
   is ``atan2(-v_wy, v_wx)`` in degrees, and a positive value is exactly the
   input that makes :func:`~f1telemetry.physics.tyres.tyre_lateral_force`
   return a positive leftward ``Fy``, which is the convention the
   ``slip_angle`` channel declares ("positive generating +y force").

:func:`contact_slip_angle_deg` is the three steps in one call, the per-corner
primitive a kernel loop reaches for.

**Low speed, and why there is no epsilon.** The slip angle is an angle, so it
is computed from the velocity's *direction* with ``atan2`` rather than from a
ratio with a guarded denominator. ``atan2`` is finite for every finite input -
including a patch with no velocity at all, for which the slip angle is exactly
zero, because a standing tyre is not slipping whatever its steering angle -
and it needs no ``max(v, eps)`` floor: no hidden parameter, no configuration
entry, nothing invented that a later calibration would have to discover. The
price is stated rather than hidden: a patch moving *rearward* (``v_wx < 0``)
reports a slip angle larger in magnitude than 90 degrees, which is the geometric truth of the angle
between heading and motion, and reversing dynamics belong to the tasks that
own them rather than to a guard smuggled in here. The returned angle is the
principal value in ``[-180, 180]`` degrees.

**Units.** Velocities are m/s and the yaw rate is rad/s - the SI the chassis
state carries. The ``yaw_rate`` *channel* is deg/s and the ``steering_angle``
channel is deg; the caller converts at the telemetry boundary. This module's
one angle input (the road-wheel steer) is in degrees, the ``channels.yaml``
convention "angle: deg unless the unit says otherwise", and is converted to
radians internally exactly as :mod:`~f1telemetry.physics.tyres` converts its
angle inputs.

**Corner order and steering.** The functions are per-corner and take the
corner's position and its own road-wheel steer angle as arguments, so they
carry no order and no geometry of their own. Callers that hold corner vectors
use the fixed ``FL, FR, RL, RR`` order :mod:`~f1telemetry.physics.forces`
establishes, and callers that hold one steer angle per corner get it from P2-T7
(Ackermann and the steering ratio); a single mean steer applied to both front
corners is the parallel-steering limit, not a default this module assumes.

**No state, no allocation, no clock, no config.** Nothing here reads a
configuration, because there is nothing to configure: the kinematics needs
only the state and the corner's place on the car, both of which the caller
has. Everything is a pure function of its arguments, and a nonfinite input
produces a nonfinite output rather than an exception, which is the
``error_model="numpy"`` contract the rest of the physics core runs under.
"""

from __future__ import annotations

import math
from typing import Final

from numba import njit

__all__ = [
    "contact_slip_angle_deg",
    "contact_velocity_m_s",
    "slip_angle_deg",
    "wheel_frame_velocity_m_s",
]

# Angle conversions as named constants rather than `math.degrees`/`math.radians`
# calls, so every conversion in the compiled path is one multiplication - and
# so this module follows the same convention `tyres` already sets for angle
# inputs: degrees at the boundary, radians inside the arithmetic.
_RAD_PER_DEG: Final[float] = math.pi / 180.0
_DEG_PER_RAD: Final[float] = 180.0 / math.pi

# The njit options are written out on every function rather than shared through
# one dict, for the same reason `tests/test_forces.py` reads them off the
# dispatcher: an indirection between the reader and the options is one more
# place for them to be wrong, and these are the options the project's
# determinism claim rests on.


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def contact_velocity_m_s(
    vx_m_s: float,
    vy_m_s: float,
    yaw_rate_rad_s: float,
    corner_x_m: float,
    corner_y_m: float,
) -> tuple[float, float]:
    """The contact patch's velocity in body coordinates: ``(vx - r*y, vy + r*x)``.

    The rigid-body relation ``v_patch = v_body + omega x r_patch`` written out
    for the body frame ``channels.yaml`` fixes - ``x`` forward, ``y`` left,
    ``z`` up, right-handed - with the yaw rate about the vertical axis and
    positive meaning a *left* turn (the nose rotating toward ``+y``). The
    cross product is what gives each term its sign: a corner ahead of the CG
    (``corner_x_m > 0``) gains leftward velocity in a left turn, and a corner
    to the left of the CG (``corner_y_m > 0``) loses longitudinal velocity -
    the inside wheel of a turn moves slower, which is the effect a differential
    exists to accommodate and this module only reports.

    Inputs, all in SI: ``vx_m_s`` is body-frame longitudinal velocity, positive
    forward; ``vy_m_s`` is body-frame lateral velocity, positive left;
    ``yaw_rate_rad_s`` is the yaw rate about ``+z`` in rad/s, positive a left
    turn (the ``yaw_rate`` *channel* is deg/s, and the caller converts, because
    the chassis state this model integrates is in SI); ``corner_x_m`` is the
    corner's forward distance from the CG, positive forward; ``corner_y_m`` is
    its lateral distance from the CG, positive left. The returns are the
    patch's velocity ``(v_x, v_y)`` in body coordinates, m/s, in that same
    frame.

    A zero yaw rate returns the body velocity exactly, whatever the corner. The
    corner position is an argument rather than read from configuration because
    deriving it - CG arms and per-axle track from ``car_spec.yaml`` - is the
    caller's job, and the caller's choice of CG-relative or axle-relative
    origin is its own.
    """
    return (
        vx_m_s - yaw_rate_rad_s * corner_y_m,
        vy_m_s + yaw_rate_rad_s * corner_x_m,
    )


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def wheel_frame_velocity_m_s(
    vx_m_s: float,
    vy_m_s: float,
    steer_deg: float,
) -> tuple[float, float]:
    """Body-frame velocity rotated into road-wheel coordinates, positive-left steer.

    The wheel frame is the body frame rotated by the road-wheel steer angle:
    the wheel's ``x`` axis lies along the steered heading and its ``y`` axis
    left of it, so a positive (left) steer rotates the frame the same way the
    wheel turns. Expressing a body-frame velocity in that frame is therefore a
    rotation by ``-steer``::

        v_wx =  vx cos(steer) + vy sin(steer)
        v_wy = -vx sin(steer) + vy cos(steer)

    Inputs: ``vx_m_s`` and ``vy_m_s`` are a velocity in body coordinates (the
    contact patch's, from :func:`contact_velocity_m_s`, though any
    body-frame velocity rotates the same way), m/s; ``steer_deg`` is this
    corner's road-wheel steer angle in degrees, positive left - the
    ``steering_angle`` channel's convention, not the steering *wheel* angle,
    which P2-T7 divides by the steering ratio first. The returns are the
    velocity ``(v_wx, v_wy)`` in wheel coordinates, m/s: ``v_wx`` along the
    steered heading, ``v_wy`` left of it.

    A zero steer is the identity, a left quarter turn maps body-forward to
    ``(0, -v)`` and body-left to ``(v, 0)``, and rotating by ``steer`` and
    then by ``-steer`` returns the input exactly. No clamping: the steer angle
    is whatever the caller's steering model produced, including beyond any
    mechanical limit, because refusing to rotate an impossible angle is the
    steering limit's job, not the kinematics'.
    """
    delta_rad = steer_deg * _RAD_PER_DEG
    cos_delta = math.cos(delta_rad)
    sin_delta = math.sin(delta_rad)
    return (
        vx_m_s * cos_delta + vy_m_s * sin_delta,
        -vx_m_s * sin_delta + vy_m_s * cos_delta,
    )


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def slip_angle_deg(vx_wheel_m_s: float, vy_wheel_m_s: float) -> float:
    """The slip angle in degrees: ``atan2(-v_wy, v_wx)``, positive for leftward ``Fy``.

    The slip angle is the signed angle between the direction the wheel points
    and the direction its contact patch moves, positive when the patch moves to
    the *right* of the wheel's heading - the deformation direction that makes
    the road push the patch, and the car, leftward. With the wheel-frame
    velocity from :func:`wheel_frame_velocity_m_s` that is exactly
    ``atan2(-v_wy, v_wx)``: a wheel steered left while the car rolls straight
    reports a positive slip angle equal to its steer angle, and a patch sliding
    leftward of its wheel's heading reports a negative one. A positive value is
    the input that makes
    :func:`~f1telemetry.physics.tyre_lateral_force` return a positive leftward
    ``Fy`` - the convention the ``slip_angle`` channel declares ("positive
    generating +y force") and invariant 4 checks across all four corners.

    Inputs: ``vx_wheel_m_s`` and ``vy_wheel_m_s`` are the contact patch's
    velocity in wheel coordinates, m/s, in the frame :func:
    `wheel_frame_velocity_m_s` builds. The return is the slip angle in degrees,
    the principal value in ``[-180, 180]``.

    **Low speed is handled by construction, not by a guard.** The angle is read
    off the velocity's direction with ``atan2``, so it is finite for every
    finite input: a patch with no velocity at all reports exactly zero (a
    standing tyre is not slipping, whatever its steering), a patch moving
    exactly sideways reports a finite right angle, and the value depends only
    on the velocity's direction - scaling the whole velocity state scales
    nothing here, which is what a ``max(vx, eps)`` denominator would break and
    why no epsilon exists in this module. A patch moving rearward (``v_wx <
    0``) reports a slip angle larger in magnitude than 90 degrees, the geometric truth of the angle
    between heading and motion; reversing dynamics are out of this module's
    scope, and inventing a small-angle convention for them here would hide
    that behind a hidden parameter.
    """
    return math.atan2(-vy_wheel_m_s, vx_wheel_m_s) * _DEG_PER_RAD


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def contact_slip_angle_deg(
    vx_m_s: float,
    vy_m_s: float,
    yaw_rate_rad_s: float,
    corner_x_m: float,
    corner_y_m: float,
    steer_deg: float,
) -> float:
    """The per-corner slip angle in degrees: the three steps above, in one call.

    The composition a kernel loop wants: the contact patch's velocity from the
    body state and the corner's CG-relative position
    (:func:`contact_velocity_m_s`), rotated into this corner's road-wheel
    frame (:func:`wheel_frame_velocity_m_s`), and read as a slip angle
    (:func:`slip_angle_deg`). The result is identical to calling the three in
    sequence, and it carries the same sign contract: positive means a leftward
    ``Fy`` from :func:`~f1telemetry.physics.tyres.tyre_lateral_force`.

    Inputs are the three chassis-state scalars (``vx_m_s`` forward-positive,
    ``vy_m_s`` left-positive, ``yaw_rate_rad_s`` left-turn-positive, in SI),
    the corner's CG-relative position (``corner_x_m`` forward-positive,
    ``corner_y_m`` left-positive), and this corner's road-wheel steer angle
    ``steer_deg`` in degrees, positive left. The return is the slip angle in
    degrees.

    Every input is used exactly once, in the order the steps demand, so a
    transposed argument is a wrong model rather than a silent one - which is
    why the tests transcribe the whole chain in plain Python and compare.
    """
    patch_vx_m_s, patch_vy_m_s = contact_velocity_m_s(
        vx_m_s, vy_m_s, yaw_rate_rad_s, corner_x_m, corner_y_m
    )
    wheel_vx_m_s, wheel_vy_m_s = wheel_frame_velocity_m_s(patch_vx_m_s, patch_vy_m_s, steer_deg)
    return slip_angle_deg(wheel_vx_m_s, wheel_vy_m_s)
