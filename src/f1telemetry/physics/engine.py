"""P1 engine rotational-speed primitive: the ICE as a state instead of a derived value.

``PHASES.md``/``PLAN.md`` Task 0 require the ICE to carry rotational speed and inertia
rather than deriving rpm from the wheel speed with a 4 000 rpm floor. This module is that
state, one step at a time: net crank torque divided by the configured
``ice_inertia_kg_m2`` integrated over a caller-supplied ``dt``, clamped to the configured
idle and rev-limit band.

**Units and sign conventions.** Torques are newton-metres at the crankshaft, engine speed
is rpm at the public boundary and rad/s inside the step, inertia is kg·m², dt is
seconds. Positive torque accelerates the ICE in its driven direction; the caller-reflected
load torque is positive when it *resists* drive, so the net is
``delivered - load``. A negative net - engine braking, or a heavy clutch engagement -
decelerates the engine.

**The clutch/gearbox load arrives already reflected.** The runner divides the
differential-side torque by the current ``gear × final_drive`` ratio to get a
crank-reflected load torque; this module does no ratio lookup and invents no clutch
friction law. Zero load - neutral, or a shift cut with the clutch open - means free
engine acceleration: ``delivered - 0`` through the same integration. Reverse is the
same contract in the other direction: the caller supplies the reflected torque for
reverse engagement, and the engine speed itself is never driven negative because the
band floor holds it at idle.

**Bounds are a hold, not a model.** Below ``idle_rpm`` the engine is held at idle
(a governed floor, not engine braking to zero), and above ``rev_limit_rpm`` it is held
at the limit. Both holds are exact clamps on the integrated value, so the state can
never leave ``[idle_rpm, rev_limit_rpm]`` and a finite step can never produce a NaN.
A nonfinite result is therefore impossible from finite inputs - and the inputs are
all validated before the compiled step runs.

Two ways in, as elsewhere. The compiled primitives take flat scalars, are what a
future kernel calls, and are unvalidated because a loop cannot afford Python in it;
:func:`step_engine_speed` is the Python-facing composition that checks the config and
every argument before any arithmetic happens. No inertia is hidden here and no new
parameter is invented: the inertia, both bounds and the step are read off
:class:`~f1telemetry.contracts.car_spec.KernelConfig` like every other coefficient.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

from numba import njit

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import KernelConfig

__all__ = [
    "engine_acceleration_rad_s2",
    "ice_omega_rad_s",
    "ice_rpm_from_omega_rad_s",
    "step_engine_speed",
]

# rad/s per rpm - the one conversion the boundary and the step share, so they cannot
# disagree by a rounding difference.
_RAD_PER_S_PER_RPM: Final[float] = math.tau / 60.0

# Every message this module raises names the entry point that raised it.
_STEP: Final[str] = "step_engine_speed"


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def ice_omega_rad_s(ice_rpm: float) -> float:
    """Crankshaft angular speed in rad/s for ``ice_rpm`` in rpm."""
    return ice_rpm * _RAD_PER_S_PER_RPM


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def ice_rpm_from_omega_rad_s(omega_rad_s: float) -> float:
    """Engine speed in rpm for a crankshaft angular speed in rad/s."""
    return omega_rad_s / _RAD_PER_S_PER_RPM


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def engine_acceleration_rad_s2(
    delivered_crank_torque_nm: float,
    load_crank_torque_nm: float,
    ice_inertia_kg_m2: float,
) -> float:
    """``d(omega)/dt = (T_delivered - T_load) / I_ice``, in rad/s².

    The net crank torque is the engine's own delivered torque minus the caller-reflected
    clutch/driveline load. ``ice_inertia_kg_m2`` is the divisor and is ``car_spec.yaml``
    data; :func:`step_engine_speed` refuses zero, negative and nonfinite values before
    this runs.
    """
    return (delivered_crank_torque_nm - load_crank_torque_nm) / ice_inertia_kg_m2


def step_engine_speed(
    config: KernelConfig,
    ice_rpm: float,
    delivered_crank_torque_nm: float,
    load_crank_torque_nm: float,
    dt_s: float | None = None,
) -> float:
    """One fixed step of the engine speed, in rpm - the ICE rotational state advanced.

    ``omega_{n+1} = omega_n + (T_delivered - T_load)/I_ice * dt``, converted back to rpm
    and clamped to ``[config.idle_rpm, config.rev_limit_rpm]``. Both torques are at the
    crankshaft in N·m, with the load positive when it resists drive (see the module
    docstring for the caller's ratio division and the neutral/reverse contract).

    ``dt_s`` defaults to ``config.dt_s`` so a caller running at the file's rate does not
    have to pass it, and the step it integrates cannot then disagree with the integrator's.

    **The bounds are holds, applied after integration.** A load that would drag the engine
    below idle leaves it at idle; a torque that would push it past the rev limit leaves it
    at the limit. The state is therefore always in the configured band and always finite.

    ``KernelConfig`` is public and replaceable, so the values are checked here rather than
    assumed to have been checked by whoever built it: the inertia must be finite and
    positive, the idle/rev-limit band must be a real interval with ``idle < rev``, and every
    argument must be finite before it reaches the compiled step.
    """
    rpm = _checked_float("ice_rpm", ice_rpm)
    delivered = _checked_float("delivered_crank_torque_nm", delivered_crank_torque_nm)
    load = _checked_float("load_crank_torque_nm", load_crank_torque_nm)
    step = config.dt_s if dt_s is None else _checked_float("dt_s", dt_s)
    if step <= 0.0:
        raise ValueError(
            f"{_STEP}: dt_s must be finite and > 0, got {step!r}. It is the interval the "
            "torque is integrated over, so a zero step would change nothing and a negative "
            "one would run the engine backwards"
        )
    inertia = _checked_float("config.ice_inertia_kg_m2", config.ice_inertia_kg_m2)
    if inertia <= 0.0:
        raise ValueError(
            f"{_STEP}: config.ice_inertia_kg_m2 must be finite and > 0, got {inertia!r}. "
            "It is the divisor of the torque balance, so zero would divide by nothing and "
            "a negative one would accelerate the engine backwards"
        )
    idle = _checked_float("config.idle_rpm", config.idle_rpm)
    rev_limit = _checked_float("config.rev_limit_rpm", config.rev_limit_rpm)
    if not 0.0 < idle < rev_limit:
        raise ValueError(
            f"{_STEP}: config.idle_rpm ({idle!r}) must be > 0 and below "
            f"config.rev_limit_rpm ({rev_limit!r}); the clamp band has to be a real "
            "interval for the bounds to hold meaningfully"
        )
    if not idle <= rpm <= rev_limit:
        raise ValueError(
            f"{_STEP}: ice_rpm must lie within the configured band "
            f"[{idle!r}, {rev_limit!r}], got {rpm!r}. A state outside the band is a "
            "caller bug, not a new clamp case"
        )

    omega = ice_omega_rad_s(rpm)
    omega += engine_acceleration_rad_s2(delivered, load, inertia) * step
    new_rpm = ice_rpm_from_omega_rad_s(omega)
    if math.isnan(new_rpm):
        # Unreachable from finite inputs and a finite inertia, but stated rather than
        # assumed: the documented guarantee is "never a NaN", so it gets its own guard.
        raise ArithmeticError(
            f"{_STEP}: engine speed integrated to NaN from rpm={rpm!r}, "
            f"delivered={delivered!r}, load={load!r}, dt_s={step!r}"
        )
    if new_rpm < idle:
        new_rpm = idle
    elif new_rpm > rev_limit:
        new_rpm = rev_limit
    return new_rpm


def _checked_float(label: str, value: object) -> float:
    """Return a finite numeric scalar as float, or fail before it reaches Numba."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{_STEP}: {label} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{_STEP}: {label} must be finite, got {value!r}")
    return number
