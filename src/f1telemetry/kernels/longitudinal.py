"""P1-T1 with P1-T6/P1-T7: the straight-line longitudinal kernel and the wheel state it closes with.

:mod:`f1telemetry.kernels.probe` proved the toolchain with a damped oscillator. This is the
first module that simulates a car, and it is deliberately almost the same shape: a flat
numeric kernel over preallocated ``float64`` arrays, written to the six rules of ``PLAN.md``
section 4.1.

**State.** The first six columns preserve the P1 order ``[x, vx, omega_fl, omega_fr, omega_rl,
omega_rr]``. P2 body, relaxation and previous-acceleration state is appended and named by the
indices below. ``STATE_SIZE`` is the one buffer contract for both models.

The wheel columns are :data:`forces`' corner order plus :data:`WHEEL_STATE_OFFSET`, derived rather
than written out: one module decides which wheel is left and which is rear, and the other asks it.

**The loop is closed.** Force is no longer an input array. One step reads the drivetrain torque the
caller computed for it, hands half of it to each rear wheel (C9.1.1, equal split - see
:func:`~f1telemetry.physics.forces.wheel_drive_torque_nm`), turns each wheel's own speed into slip
and slip into a longitudinal force through the existing tyre model, sums those four forces plus the
drag onto the car, and advances four wheel speeds and a chassis. Nothing intervenes: no traction
control, no differential, no ABS (C9.1.2, C9.9.1, C11.4.1), so wheelspin and lock are states the
loop reaches on its own.

That closing is why this kernel now reads the configuration so much more of. Every coefficient is
still read **once, in Python, outside the loop** - ``PLAN.md`` section 4.1 rule 3 - and the compiled
loop is handed nothing but numbers: no config object, no YAML document, no keyword argument. It is
handed them through :func:`~f1telemetry.physics.forces.validated_config_scalars` and
:func:`~f1telemetry.physics.forces.validate_aero_arrays`, which is also why the boundary has to
grow: the loop divides by the wheel inertia and the rolling radius and indexes the aero curves
with ``boundscheck=False``, so a configuration it cannot use has to be refused before it starts.

**The step is data.** ``dt`` comes from ``car_spec.yaml`` through :class:`KernelConfig`, so
changing the integration rate is a data edit (the cross-phase rule that no physical constant is
hardcoded in Python). At 100 µs this is a 10 kHz kernel, which is what ``PLAN.md`` section 4
fixes; the tests assert the file still says so.

**Determinism.** No clock, no random source, no allocation, no ``prange``: same
:class:`KernelConfig`, same buffers, same inputs, byte-identical trace. ``cache=True`` and
``fastmath=False`` are not preferences here - ``fastmath`` permits floating-point
reassociation, which would make the byte-identity claim version-dependent rather than a
property of the kernel.

The compiled loop is :func:`_integrate`, and it is private on purpose: ``boundscheck=False``
means an undersized buffer is an out-of-bounds write with no error at all, so the only way that
can be safe is for there to be exactly one way in. :func:`simulate` is that way. It is the whole
boundary - the one-way ``@njit`` boundary ``PHASES.md`` asks for - and it refuses a nonfinite or
nonpositive ``dt_s`` or ``mass_kg``, a step count that is not an integer, a seeded state carrying
a nonfinite wheel speed, and any buffer that is not a C-contiguous ``float64`` ``ndarray`` of
exactly the size the run needs. A mistake surfaces as a :class:`ValueError` naming the buffer
instead of as a silent write past the end of an array.
"""

from __future__ import annotations

import math
import operator
from typing import TYPE_CHECKING, Final

import numpy as np
from numba import njit

# The compiled loop calls the force model's primitives - the same direction as `PLAN.md` section 6's
# `torque_curve -> gearbox -> clutch -> differential -> wheels`, read as models feeding the
# integrator rather than the other way round. Layer isolation exists to keep analytics, the server
# and the web layer out of the physics core; a kernel composing the core is the composition.
from f1telemetry.physics import forces  # noqa: TID251 -- the kernel is what composes the models

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import KernelConfig

__all__ = [
    "ALPHA_RELAX_OFFSET",
    "FL_WHEEL_INDEX",
    "FR_WHEEL_INDEX",
    "HEAVE_INDEX",
    "KAPPA_RELAX_OFFSET",
    "PITCH_INDEX",
    "PREVIOUS_AX_INDEX",
    "PREVIOUS_AY_INDEX",
    "PREVIOUS_AZ_INDEX",
    "PSI_INDEX",
    "RL_WHEEL_INDEX",
    "ROLL_INDEX",
    "RR_WHEEL_INDEX",
    "STATE_SIZE",
    "VX_INDEX",
    "VY_INDEX",
    "V_INDEX",
    "WHEEL_STATE_OFFSET",
    "X_INDEX",
    "YAW_RATE_INDEX",
    "Y_INDEX",
    "allocate",
    "initial_state",
    "simulate",
]

# Preserve the P1 prefix exactly: x, vx, then four wheel angular speeds in force corner order.
# P2 state is appended so existing callers that use the named P1 columns remain valid.
WHEEL_STATE_OFFSET: Final[int] = 2
X_INDEX: Final[int] = 0
VX_INDEX: Final[int] = 1
V_INDEX: Final[int] = VX_INDEX
FL_WHEEL_INDEX: Final[int] = WHEEL_STATE_OFFSET + forces.FL_WHEEL_INDEX
FR_WHEEL_INDEX: Final[int] = WHEEL_STATE_OFFSET + forces.FR_WHEEL_INDEX
RL_WHEEL_INDEX: Final[int] = WHEEL_STATE_OFFSET + forces.RL_WHEEL_INDEX
RR_WHEEL_INDEX: Final[int] = WHEEL_STATE_OFFSET + forces.RR_WHEEL_INDEX
Y_INDEX: Final[int] = WHEEL_STATE_OFFSET + forces.WHEEL_COUNT
PSI_INDEX: Final[int] = Y_INDEX + 1
VY_INDEX: Final[int] = PSI_INDEX + 1
YAW_RATE_INDEX: Final[int] = VY_INDEX + 1
ROLL_INDEX: Final[int] = YAW_RATE_INDEX + 1
PITCH_INDEX: Final[int] = ROLL_INDEX + 1
HEAVE_INDEX: Final[int] = PITCH_INDEX + 1
ALPHA_RELAX_OFFSET: Final[int] = HEAVE_INDEX + 1
KAPPA_RELAX_OFFSET: Final[int] = ALPHA_RELAX_OFFSET + forces.WHEEL_COUNT
PREVIOUS_AX_INDEX: Final[int] = KAPPA_RELAX_OFFSET + forces.WHEEL_COUNT
PREVIOUS_AY_INDEX: Final[int] = PREVIOUS_AX_INDEX + 1
PREVIOUS_AZ_INDEX: Final[int] = PREVIOUS_AY_INDEX + 1
STATE_SIZE: Final[int] = PREVIOUS_AZ_INDEX + 1


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def _integrate(
    steps: int,
    dt_s: float,
    mass_kg: float,
    weight_n: float,
    drive_torque_nm: np.ndarray,
    brake_torque_nm: np.ndarray,
    state: np.ndarray,
    out: np.ndarray,
    air_density_kg_m3: float,
    reference_area_m2: float,
    aero_speed_m_s: np.ndarray,
    cl: np.ndarray,
    cd: np.ndarray,
    pacejka_b: float,
    pacejka_c: float,
    pacejka_e: float,
    pacejka_mu: float,
    slip_ratio_min_speed_m_s: float,
    rolling_radius_m: float,
    wheel_inertia_kg_m2: float,
    front_weight_fraction: float,
) -> np.ndarray:
    """Run ``steps`` fixed steps into caller-owned ``out``, shape ``(steps + 1, STATE_SIZE)``.

    Row 0 of ``out`` is seeded from ``state`` and row ``n`` is the state after exactly ``n``
    steps of ``dt_s``, so row ``n`` is simulated time ``n * dt_s`` and the trace has one row
    per step plus the initial condition. ``drive_torque_nm`` holds the drivetrain torque the caller
    computed for each step - what ``gearbox.step_gearbox`` returns - so no torque is computed here
    and none is allocated.

    **The order inside the step is the model**, and it is three rules:

    * **Every force is evaluated at the state the step started from.** Aero and tyre forces are
      read from row ``index``, and the wheel states are advanced from row ``index``. Updating a
      wheel from the speed the same step produced would be an implicit scheme on the wheel and an
      explicit one on the chassis, which is neither of the two the integrator claims to be.
    * **The wheels are advanced inside the same single pass** that accumulates the force on the
      car. Recomputing the tyre forces in a second loop would double four transcendental
      evaluations per step to avoid holding four numbers, and the only place this loop could hold
      anything is the allocator.
    * **Position advances with the speed this step produced.** Semi-implicit means exactly that;
      the explicit alternative is the same arithmetic with ``out[index, V_INDEX]`` on the last line,
      and the two part company by one step of velocity per step, so the choice is pinned by a test
      rather than left to taste.

    The loop reads and writes only the three arrays it was given, and returns the caller's
    ``out`` so a scenario can hold the buffer it wrote into. Nothing is checked here - see the
    module docstring - so it must only ever be reached through :func:`simulate`.
    """
    out[0, :] = state
    for index in range(steps):
        # Carry all P2-owned states until the P2 force/integration path updates them. This keeps
        # caller-seeded transients deterministic and prevents uninitialized output columns.
        out[index + 1, :] = out[index, :]
        speed_m_s = out[index, V_INDEX]
        downforce_n, drag_n = forces.aero_forces(
            speed_m_s,
            air_density_kg_m3,
            reference_area_m2,
            aero_speed_m_s,
            cl,
            cd,
        )
        drivetrain_torque_nm = drive_torque_nm[index]
        # Drag is signed along +x and is negative going forward, so it starts the sum rather than
        # being subtracted; adding it keeps one sign convention instead of two.
        net_force_n = drag_n
        for wheel in range(forces.WHEEL_COUNT):
            column = WHEEL_STATE_OFFSET + wheel
            load_n = (
                forces.static_wheel_load_n(weight_n, front_weight_fraction, wheel)
                + downforce_n / forces.WHEEL_COUNT
            )
            tyre_fx_n = forces.wheel_tyre_force_n(
                speed_m_s,
                out[index, column],
                load_n,
                rolling_radius_m,
                slip_ratio_min_speed_m_s,
                pacejka_b,
                pacejka_c,
                pacejka_e,
                pacejka_mu,
            )
            net_force_n += tyre_fx_n
            drive_nm = forces.wheel_drive_torque_nm(wheel, drivetrain_torque_nm)
            alpha_rad_s2 = forces.wheel_angular_acceleration_rad_s2(
                drive_nm,
                tyre_fx_n,
                rolling_radius_m,
                wheel_inertia_kg_m2,
                brake_torque_nm[index, wheel],
            )
            out[index + 1, column] = out[index, column] + alpha_rad_s2 * dt_s
        acceleration_m_s2 = net_force_n / mass_kg
        out[index + 1, V_INDEX] = speed_m_s + acceleration_m_s2 * dt_s
        out[index + 1, X_INDEX] = out[index, X_INDEX] + out[index + 1, V_INDEX] * dt_s
    return out


def initial_state(
    distance_m: float = 0.0,
    speed_m_s: float = 0.0,
    wheel_omega_rad_s: float = 0.0,
    *,
    y_m: float = 0.0,
    heading_rad: float = 0.0,
    vy_m_s: float = 0.0,
    yaw_rate_rad_s: float = 0.0,
    roll_rad: float = 0.0,
    pitch_rad: float = 0.0,
    heave_m: float = 0.0,
    alpha_relax_deg: np.ndarray | None = None,
    kappa_relax: np.ndarray | None = None,
    previous_acceleration_m_s2: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Caller-owned float64 state buffer with the P1 prefix and named P2 state appended.

    The stable first six columns are ``[x, vx, omega_fl, omega_fr, omega_rl, omega_rr]``.
    The appended columns are ``y, psi, vy, yaw_rate, roll, pitch, heave, alpha_relax[4],
    kappa_relax[4], previous_acceleration_xyz``. Roll, pitch and heave are quasi-static body
    outputs; suspension travel is computed from load and ride rate. Positions are metres, angles
    radians, velocities m/s or rad/s as named, and acceleration m/s².

    **One number for all four wheels, on purpose.** P1 has no lateral dynamics, so the only wheel
    asymmetry it can express is the difference a differential would make - and there is not one
    (C9.9.1). Seeding four columns from four numbers would suggest the model can carry a left/right
    difference it cannot produce. A caller that does want to start from a different state on
    different corners writes the columns itself.

    ``wheel_omega_rad_s`` defaults to ``0.0`` rather than to the wheel speed that would roll at
    ``speed_m_s``, so the default state is a standing start - the state a launch is in, and the one
    whose first step is exactly predictable. **Seeding ``speed_m_s > 0`` without a wheel speed
    therefore starts the car with a locked axle**, which the loop resolves rather than refuses; a
    rolling start is ``speed_m_s / rolling_radius_m``, and
    :func:`~f1telemetry.physics.forces.step_wheel` takes the same angular speed this takes.

    Nothing is checked here, exactly as in
    :func:`~f1telemetry.physics.gearbox.initial_state`: this only manufactures a buffer, and
    :func:`simulate` is the boundary that decides whether the numbers in it are usable. A nonfinite
    wheel speed therefore fails at the first step rather than at construction.
    """
    state = np.zeros(STATE_SIZE, dtype=np.float64)
    state[X_INDEX] = distance_m
    state[V_INDEX] = speed_m_s
    for wheel in range(forces.WHEEL_COUNT):
        state[WHEEL_STATE_OFFSET + wheel] = wheel_omega_rad_s
    state[Y_INDEX] = y_m
    state[PSI_INDEX] = heading_rad
    state[VY_INDEX] = vy_m_s
    state[YAW_RATE_INDEX] = yaw_rate_rad_s
    state[ROLL_INDEX] = roll_rad
    state[PITCH_INDEX] = pitch_rad
    state[HEAVE_INDEX] = heave_m
    if alpha_relax_deg is not None:
        state[ALPHA_RELAX_OFFSET : ALPHA_RELAX_OFFSET + forces.WHEEL_COUNT] = alpha_relax_deg
    if kappa_relax is not None:
        state[KAPPA_RELAX_OFFSET : KAPPA_RELAX_OFFSET + forces.WHEEL_COUNT] = kappa_relax
    state[PREVIOUS_AX_INDEX : PREVIOUS_AZ_INDEX + 1] = previous_acceleration_m_s2
    return state


def allocate(steps: int) -> np.ndarray:
    """Caller-owned output buffer, shape ``(steps + 1, STATE_SIZE)``."""
    count = _checked_steps(steps)
    return np.zeros((count + 1, STATE_SIZE), dtype=np.float64)


def simulate(
    config: KernelConfig,
    steps: int,
    state: np.ndarray,
    drive_torque_nm: np.ndarray,
    out: np.ndarray,
    brake_torque_nm: np.ndarray | None = None,
) -> np.ndarray:
    """Run a straight-line scenario from a validated :class:`KernelConfig`, in fixed steps.

    The one entry point a scenario uses, and the only thing in this module that touches the
    compiled loop. It reads every coefficient it needs out of the config once, here in Python, and
    hands the compiled loop nothing but numbers - the kernel never sees the config object, a YAML
    document or a keyword argument.

    ``drive_torque_nm`` is the **drivetrain torque per step**, in newton-metres at the
    differential, which is what :func:`~f1telemetry.physics.gearbox.step_gearbox` returns. It is
    the only thing the caller has to supply: the aero, the tyre forces, the slip, the wheel states
    and the static axle loads are all derived inside the loop from the state and the config.

    Every buffer is the caller's, and every one is validated before the loop starts rather than
    inside it, because ``boundscheck=False`` means the loop cannot. So are the config scalars the
    loop divides by and indexes, and the step count: :class:`KernelConfig` is normally
    range-checked by the loader, but it is a public frozen dataclass and can be built or replaced
    directly, and a zero mass, a zero wheel inertia or a NaN step is arithmetic that returns NaNs
    rather than an error. ``steps`` may be zero, which writes the seeded state into row 0 and
    nothing else.

    The torque history is *not* scanned. It is the one input a caller owns for the whole run, and
    walking it in Python is O(steps) work before any simulation happens; a nonfinite torque produces
    a nonfinite trace, which is the same position Task 2 took when force was an input array. The
    seeded state is six values and *is* checked, because a nonfinite wheel speed is not a slow
    wheel but a wheel that has already left the model.

    Returns ``out``, so a caller can write ``out = simulate(...)`` without giving up the buffer.
    ``brake_torque_nm`` is an optional caller-owned signed per-wheel torque history with shape
    ``(steps, 4)``. Negative values brake forward rotation; it has no ABS or force transfer.
    Omitted input means zero brake torque.
    """
    count = _checked_steps(steps)
    for name, value in (
        ("dt_s", config.dt_s),
        ("mass_kg", config.mass_kg),
        ("gravity_m_s2", config.gravity_m_s2),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"simulate: config.{name} must be finite and > 0, got {value!r}. The kernel "
                "divides by mass_kg and multiplies by dt_s, and the static axle loads are a "
                "fraction of mass_kg * gravity_m_s2, so none of the three is checked for you once "
                "the loop starts"
            )
    # The rest of the configuration, through the force model's own shared validator: one list of
    # names and one set of sign rules for `simulate`, `step_forces` and `step_wheel` rather than
    # three copies that could disagree. The aero curves are checked here for the same reason -
    # the loop indexes them with `boundscheck=False`.
    values = forces.validated_config_scalars(config, "simulate")
    forces.validate_aero_arrays(config, "simulate")

    _check_buffer("state", state, (STATE_SIZE,), writable=False)
    _check_buffer("drive_torque_nm", drive_torque_nm, (count,), writable=False)
    if brake_torque_nm is None:
        brake_torque_nm = np.zeros((count, forces.WHEEL_COUNT), dtype=np.float64)
    _check_buffer("brake_torque_nm", brake_torque_nm, (count, forces.WHEEL_COUNT), writable=False)
    if not np.isfinite(brake_torque_nm).all():
        raise ValueError("simulate: brake_torque_nm must be finite")
    _check_buffer("out", out, (count + 1, STATE_SIZE), writable=True)
    if not np.isfinite(state).all():
        raise ValueError(
            "simulate: state must be finite. A NaN wheel speed is not a slow wheel: it reaches the "
            "slip ratio, from there the tyre force, and from there every later row of the trace"
        )
    return _integrate(
        count,
        config.dt_s,
        config.mass_kg,
        config.mass_kg * config.gravity_m_s2,
        drive_torque_nm,
        brake_torque_nm,
        state,
        out,
        values["air_density_kg_m3"],
        values["reference_area_m2"],
        config.aero_speed_m_s,
        config.cl,
        config.cd,
        values["pacejka_b"],
        values["pacejka_c"],
        values["pacejka_e"],
        values["pacejka_mu"],
        values["slip_ratio_min_speed_m_s"],
        values["rolling_radius_m"],
        values["wheel_inertia_kg_m2"],
        values["front_weight_fraction"],
    )


def _checked_steps(steps: int) -> int:
    """The step count as a genuine ``int``, or a :class:`ValueError`.

    ``operator.index`` rather than ``int()`` on purpose: a float count would otherwise be
    passed straight to the compiled loop, where it compiles a *second* float64 specialisation
    of the same integrator and a ``4.0`` step count would quietly succeed. ``bool`` is refused
    even though it is an ``int``, because ``simulate(config, True, ...)`` is a bug.
    """
    if isinstance(steps, bool):
        raise ValueError(f"steps must be an integer, got {steps!r} of type {type(steps).__name__}")
    try:
        count = operator.index(steps)
    except TypeError as error:
        raise ValueError(
            f"steps must be an integer, got {steps!r} of type {type(steps).__name__}"
        ) from error
    if count < 0:
        raise ValueError(f"steps must be >= 0, got {count}")
    return count


def _check_buffer(name: str, array: object, shape: tuple[int, ...], *, writable: bool) -> None:
    """Refuse a buffer the kernel would read out of bounds or quietly re-quantise.

    ``state`` and ``drive_torque_nm`` are read-only as far as the kernel is concerned, so a caller
    may hand over a buffer it does not want written; ``out`` is the kernel's only destination and
    must be writable. Contiguity is part of the contract in ``car_spec.KernelConfig`` and in
    ``PLAN.md`` section 4.1, and a strided buffer would compile as a differently-typed kernel
    rather than being rejected.
    """
    if not isinstance(array, np.ndarray):
        raise ValueError(
            f"simulate: {name} must be a numpy ndarray, got {type(array).__name__}. Anything "
            "else would be copied into the kernel, so the buffer the caller holds is not the "
            "one the run writes"
        )
    if array.dtype != np.float64:
        raise ValueError(
            f"simulate: {name} buffer must be float64, got {array.dtype}. The kernel is "
            "float64 end to end and a narrower buffer would re-quantise the trace"
        )
    if array.shape != shape:
        raise ValueError(f"simulate: {name} buffer must have shape {shape}, got {array.shape}")
    if not array.flags.c_contiguous:
        raise ValueError(
            f"simulate: {name} buffer must be C-contiguous; a strided buffer compiles as a "
            "different kernel signature and the run stops being the one the tests pin"
        )
    if writable and not array.flags.writeable:
        raise ValueError(
            f"simulate: {name} buffer is read-only. It is the kernel's only destination, so a "
            "read-only buffer makes the run write out of bounds"
        )
