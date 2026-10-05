"""P2-T7: steering geometry - the Ackermann pair, the ratio, and the limit.

``PLAN.md`` section 4 puts Pacejka tyres on every corner, and
:mod:`~f1telemetry.physics.kinematics` asks each corner for its own
road-wheel steer angle. This module answers the question upstream of
that: *which road-wheel angles does a steering-wheel input mean?* It
reads the scenario's steering-wheel angle in degrees, positive left, and
derives the four road-wheel angles in degrees, positive left, in the
``FL, FR, RL, RR`` order every other per-wheel quantity uses.

**The three steps**, each one function, so a caller - and a compiled
kernel loop - can stop at whichever level it needs:

1. Divide the steering-wheel angle by the configured ``steering_ratio``
   into the mean front road-wheel angle. The ratio is positive by
   validation, so the mean carries the input's sign.
2. :func:`ideal_ackermann_front_angles_deg` derives the ideal front
   pair from the validated wheelbase and front track: the turn centre
   that the mean angle implies has radius
   ``wheelbase / tan(mean)``, and the inner and outer wheels sit
   ``track / 2`` either side of it, so each turns through
   ``atan(wheelbase / radius_inner_or_outer)``. Equivalently the pair
   satisfies the defining Ackermann relation
   ``cot(delta_outer) - cot(delta_inner) == front_track / wheelbase``.
3. :func:`road_wheel_angles_deg` blends that ideal pair with parallel
   steering on the configured ``ackermann_fraction`` - 0 gives both
   front wheels the mean (parallel), 1 the ideal pair, and a value in
   between a linear interpolation of each front wheel's deviation from
   the mean - and writes both front angles plus two zeros for the
   non-steered rear axle into the caller's buffer.

**Signs.** A positive input steers left: the front road-wheel angles are
positive (left), the left wheel is inner and steeper than the right.
Negating the input negates both angles and swaps which wheel is inner,
exactly: ``angles(-d) == -swap(angles(d))``. The rear angles are zero
for every input - no rear steer is modelled.

**Units.** Both the input and the outputs are degrees, the
``steering_angle`` channel's convention; the steering ratio and the
Ackermann fraction are dimensionless; the wheelbase and front track are
metres. Radians appear only inside the trigonometry, converted with the
same ``_RAD_PER_DEG``/``_DEG_PER_RAD`` constants ``kinematics`` uses.

**The limit is a limit.** The scenario input is refused when its
magnitude exceeds ``max_steering_wheel_angle_deg`` or when it is
nonfinite; nothing is clipped to look legal. The compiled primitives
take no such check - a compiled loop cannot raise - and a nonfinite
input to them produces nonfinite arithmetic rather than an exception,
the ``error_model="numpy"`` contract the rest of the physics core runs
under. The validating Python boundary
(:func:`steering_angles_deg` / :func:`validated_steering_scalars`) is
what a scenario reaches for.

**Zero and the geometric edge.** Exactly zero input gives exactly zero
everywhere: the mean is zero, and the ideal pair is defined as zero
rather than through ``tan(0)``. An input so large that the implied turn
centre reaches the inner wheel puts ``radius - track/2`` at or below
zero; the inner angle then pins to its geometric limit of 90 degrees -
the continuous limit of ``atan(wheelbase / (radius - track/2))`` as the
denominator shrinks to zero from above - rather than flipping sign or
returning NaN. The validated run cannot reach that territory (the
configured limit keeps the mean near 33 degrees for the shipped
geometry), but the primitive stays finite if a caller sweeps past it.

**Two ways in, deliberately.** The primitives take flat scalars and a
caller-owned ``float64`` vector and are what a compiled kernel calls;
:func:`steering_angles_deg` is the Python-facing composition that checks
the configuration and the input before any arithmetic happens. Nothing
here reads a clock, a random source, or a dict, and no number is written
in Python that ``car_spec.yaml`` does not supply.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import numpy as np
from numba import njit

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import KernelConfig

__all__ = [
    "ideal_ackermann_front_angles_deg",
    "road_wheel_angles_deg",
    "steering_angles_deg",
    "step_road_wheel_angles",
    "validated_steering_scalars",
]

# Angle conversions as named constants rather than `math.degrees`/`math.radians`
# calls, matching `kinematics`: degrees at the boundary, radians inside the
# arithmetic.
_RAD_PER_DEG: Final[float] = math.pi / 180.0
_DEG_PER_RAD: Final[float] = 180.0 / math.pi

# The four per-wheel road-wheel angles are length four, in the corner order
# every other per-wheel quantity uses.
_WHEEL_COUNT: Final[int] = 4

# The njit options are written out on every function rather than shared through
# one dict, for the same reason `tests/test_steering.py` reads them off the
# dispatcher: an indirection between the reader and the options is one more
# place for them to be wrong, and these are the options the project's
# determinism claim rests on.


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def ideal_ackermann_front_angles_deg(
    road_wheel_deg: float,
    wheelbase_m: float,
    front_track_m: float,
) -> tuple[float, float]:
    """The ideal front road-wheel pair (FL, FR) for a mean front angle, degrees.

    The mean front road-wheel angle implies a turn centre at radius
    ``wheelbase / tan(mean)`` from the front-axle midpoint, and the inner
    and outer front wheels turn through ``atan(wheelbase /
    (radius -/+ track/2))`` respectively - which is what makes the pair
    satisfy ``cot(delta_outer) - cot(delta_inner) == front_track /
    wheelbase``. On a left turn (positive input) the FL is inner and the
    FR outer; on a right turn they swap, and both angles carry the mean's
    sign.

    Zero mean is exactly zero pair - the turn radius is infinite, and the
    division by ``tan(0)`` is never taken. A mean so large that the
    implied turn centre reaches the inner wheel (``radius - track/2 <=
    0``) pins the inner angle to its 90-degree geometric limit rather
    than flipping sign or returning NaN. The outer angle is always
    finite: its denominator is the beyond-centre radius, strictly larger.
    """
    if road_wheel_deg == 0.0:
        return (0.0, 0.0)
    magnitude_deg = abs(road_wheel_deg)
    radius_m = wheelbase_m / math.tan(magnitude_deg * _RAD_PER_DEG)
    inner_radius_m = radius_m - 0.5 * front_track_m
    outer_radius_m = radius_m + 0.5 * front_track_m
    if inner_radius_m <= 0.0:
        inner_magnitude_deg = 90.0
    else:
        inner_magnitude_deg = math.atan(wheelbase_m / inner_radius_m) * _DEG_PER_RAD
    outer_magnitude_deg = math.atan(wheelbase_m / outer_radius_m) * _DEG_PER_RAD
    if road_wheel_deg > 0.0:
        return (inner_magnitude_deg, outer_magnitude_deg)
    return (-outer_magnitude_deg, -inner_magnitude_deg)


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def road_wheel_angles_deg(
    steer_wheel_deg: float,
    steering_ratio: float,
    wheelbase_m: float,
    front_track_m: float,
    ackermann_fraction: float,
    out: np.ndarray,
) -> None:
    """All four road-wheel angles for one steering-wheel input, written in place.

    ``out`` is the caller-owned ``float64`` vector of length four in
    ``FL, FR, RL, RR`` order; this function returns nothing and allocates
    nothing, because a Numba loop owns no buffers of its own. The mean
    front angle is ``steer_wheel_deg / steering_ratio``; each front wheel
    blends the mean toward its ideal Ackermann angle on
    ``ackermann_fraction`` - 0 is parallel steering (both fronts at the
    mean), 1 the ideal pair, and in between a linear interpolation of
    each deviation from the mean. The rear axle is written as exactly two
    zeros. No validation happens here: the boundary does that, and a
    nonfinite argument yields nonfinite arithmetic, not an exception.
    """
    mean_deg = steer_wheel_deg / steering_ratio
    fl_ideal_deg, fr_ideal_deg = ideal_ackermann_front_angles_deg(
        mean_deg, wheelbase_m, front_track_m
    )
    out[0] = mean_deg + ackermann_fraction * (fl_ideal_deg - mean_deg)
    out[1] = mean_deg + ackermann_fraction * (fr_ideal_deg - mean_deg)
    out[2] = 0.0
    out[3] = 0.0


def validated_steering_scalars(config: KernelConfig, prefix: str) -> dict[str, float]:
    """Every steering scalar the model reads, narrowed to ``float`` and range-checked.

    One validator for the model, shared by the Python entry point and by
    any kernel that reads the configuration outside its loop. ``prefix``
    names the calling entry point in the message, because a shared
    boundary is a worse place for an ambiguous error than a named one.

    The ratio must be positive - it divides the input - and the maximum
    positive, because the boundary rejects beyond it. The Ackermann
    fraction must lie in ``[0, 1]``: 0 is parallel steering, 1 the ideal
    pair, and a value outside that interval interpolates away from the
    documented two limits. The wheelbase and the front track must be
    positive, because each divides into the ideal-geometry radii.
    """
    values: dict[str, float] = {}
    for name in ("steering_ratio", "max_steering_wheel_angle_deg", "wheelbase_m"):
        value = _checked_float(f"config.{name}", getattr(config, name), prefix=prefix)
        if value <= 0.0:
            raise ValueError(
                f"{prefix}: config.{name} must be finite and > 0, got {value!r}. "
                "The ratio divides the steering input, the limit bounds it, and "
                "the wheelbase divides into every ideal-geometry radius"
            )
        values[name] = value
    ackermann = _checked_float(
        "config.ackermann_fraction", config.ackermann_fraction, prefix=prefix
    )
    if not 0.0 <= ackermann <= 1.0:
        raise ValueError(
            f"{prefix}: config.ackermann_fraction must be in [0, 1], got {ackermann!r}. "
            "0 is parallel steering and 1 is ideal Ackermann"
        )
    values["ackermann_fraction"] = ackermann
    tracks = config.axle_track_m
    if (
        not isinstance(tracks, np.ndarray)
        or tracks.shape != (2,)
        or not np.isfinite(tracks).all()
        or (tracks <= 0.0).any()
    ):
        raise ValueError(
            f"{prefix}: config.axle_track_m must be two finite positive metres, "
            f"got {getattr(tracks, 'shape', None)} / {getattr(tracks, 'dtype', None)}"
        )
    values["front_track_m"] = float(tracks[0])
    return values


def steering_angles_deg(config: KernelConfig, steer_wheel_deg: float) -> np.ndarray:
    """The four road-wheel angles in degrees for a steering-wheel input, checked first.

    The Python-facing composition: a caller inside a kernel does not come
    through here - it reads the configuration outside its loop through
    :func:`validated_steering_scalars` and calls :func:`road_wheel_angles_deg`
    directly, because this function's checks are Python and a compiled
    loop cannot make them. The input is a steering-*wheel* angle in
    degrees, positive left; the return is the caller's four-corner
    road-wheel answer in degrees in ``FL, FR, RL, RR`` order.

    The input must be a real, finite number within the configured
    magnitude limit - refused, never clipped: a command past the limit is
    a caller bug, and a clamped one is a wrong model wearing a legal
    shape. Every scalar is narrowed to ``float`` so that equivalent ``int``
    and ``float`` callers share one compiled specialisation.
    """
    values = validated_steering_scalars(config, "steering_angles_deg")
    steer = _checked_float("steer_wheel_deg", steer_wheel_deg, prefix="steering_angles_deg")
    limit = values["max_steering_wheel_angle_deg"]
    if abs(steer) > limit:
        raise ValueError(
            f"steering_angles_deg: steer_wheel_deg magnitude {abs(steer)!r} exceeds "
            f"config.max_steering_wheel_angle_deg {limit!r}; the limit rejects, it "
            "does not clip"
        )
    out = np.empty(_WHEEL_COUNT, dtype=np.float64)
    road_wheel_angles_deg(
        steer,
        values["steering_ratio"],
        values["wheelbase_m"],
        values["front_track_m"],
        values["ackermann_fraction"],
        out,
    )
    return out


def step_road_wheel_angles(config: KernelConfig, steer_wheel_deg: float, out: np.ndarray) -> None:
    """One road-wheel angle step, checked before it runs.

    The Python-facing composition: a caller inside a kernel does not come
    through here - it reads the configuration outside its loop through
    :func:`validated_steering_scalars` and calls :func:`road_wheel_angles_deg`
    directly, because this function's checks are Python and a compiled
    loop cannot make them. The input is a steering-*wheel* angle in
    degrees, positive left; ``out`` is the caller-owned ``float64``
    vector of length four in ``FL, FR, RL, RR`` order that the angles
    are written into in place, and nothing is returned or allocated.

    The input must be a real, finite number within the configured
    magnitude limit - refused, never clipped - and the buffer must be a
    writable C-contiguous ``float64`` length-four vector: with bounds
    checking off, a wrong length is an out-of-bounds write rather than
    an error, and a strided or ``float32`` buffer compiles as the wrong
    memory rather than failing. Equivalent ``int`` and ``float``
    callers are narrowed to ``float`` so they share one compiled
    specialisation.
    """
    values = validated_steering_scalars(config, "step_road_wheel_angles")
    steer = _checked_float("steer_wheel_deg", steer_wheel_deg, prefix="step_road_wheel_angles")
    limit = values["max_steering_wheel_angle_deg"]
    if abs(steer) > limit:
        raise ValueError(
            f"step_road_wheel_angles: steer_wheel_deg magnitude {abs(steer)!r} exceeds "
            f"config.max_steering_wheel_angle_deg {limit!r}; the limit rejects, it "
            "does not clip"
        )
    if (
        not isinstance(out, np.ndarray)
        or out.dtype != np.float64
        or out.ndim != 1
        or out.size != _WHEEL_COUNT
        or not out.flags.c_contiguous
        or not out.flags.writeable
    ):
        raise ValueError(
            f"step_road_wheel_angles: out must be a writable C-contiguous float64 vector of length "
            f"{_WHEEL_COUNT}, got {type(out).__name__} of shape "
            f"{getattr(out, 'shape', None)} and dtype {getattr(out, 'dtype', None)}"
        )
    road_wheel_angles_deg(
        steer,
        values["steering_ratio"],
        values["wheelbase_m"],
        values["front_track_m"],
        values["ackermann_fraction"],
        out,
    )


def _checked_float(label: str, value: object, *, prefix: str) -> float:
    """Return a finite real scalar as ``float``, or fail before it reaches Numba.

    ``bool`` is refused although it is an ``int``, because ``True`` as a
    steering angle is a caller bug rather than a number. This mirrors
    ``tyres._checked_float`` and ``relaxation._checked_float`` rather than
    sharing either: the modules validate different field sets for
    different models, and a shared private helper between them would be a
    coupling for the sake of six lines.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{prefix}: {label} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{prefix}: {label} must be finite, got {value!r}")
    return number
