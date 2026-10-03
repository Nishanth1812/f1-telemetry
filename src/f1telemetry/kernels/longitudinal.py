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

**What a tyre is asked about, and where it came from.** Each wheel's vertical load used to be
``forces.static_wheel_load_n`` plus a quarter of the downforce - a P1 approximation that could not
express a load transfer, because transfer depends on acceleration and acceleration depends on tyre
force. The kernel now calls :func:`~f1telemetry.physics.loads.corner_loads_n` and
:func:`~f1telemetry.physics.loads.clamp_travel_to_limits` instead, reading the **previous** step's
``ax``, ``ay`` and ``az`` out of the three state columns the loop writes at the end of each step.
That is the plan's explicit answer to the algebraic loop, and it is stated rather than hidden: the
load a tyre is asked about is one 100 us step behind the force it produced. Two consequences are
deliberate and visible in the tests. The aerodynamic load now follows the same CG share as the
weight rather than being spread a quarter per patch, and the total still sums to weight plus
downforce. And the diagonal is *closed*: a run seeds the coupling to zero through
:func:`initial_state`, and every step after that feeds its own resolved acceleration forward, so the
transfer is a result of the run rather than an input a caller has to keep supplying.

**Diagnostics are optional and caller-owned.** :class:`StepOutputs` is four ``(steps, 4)`` buffers -
load, longitudinal force, suspension travel and the travel-limit flag - filled per step when a
caller asks for them and left alone when it does not. They are an *output*, not a state: the
loads are computed either way, because they are an input to the tyre model, so omitting them cannot
change a byte of the trace. That is what lets them stay optional rather than becoming part of
the trace itself, and it is asserted rather than assumed.

That closing is why this kernel now reads the configuration so much more of. Every coefficient is
still read **once, in Python, outside the loop** - ``PLAN.md`` section 4.1 rule 3 - and the compiled
loop is handed nothing but numbers: no config object, no YAML document, no keyword argument. It is
handed them through :func:`~f1telemetry.physics.forces.validated_config_scalars`,
:func:`~f1telemetry.physics.forces.validate_aero_arrays` and
:func:`~f1telemetry.physics.loads.validated_load_scalars`, which is also why the boundary has to
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
exactly the size the run needs. The optional :class:`StepOutputs` buffers are held to the same
rules,
and to one extra: all four are validated or none of them is used. A mistake surfaces as a
:class:`ValueError` naming the buffer instead of as a silent write past the end of an array.
"""

from __future__ import annotations

import math
import operator
from itertools import combinations
from typing import TYPE_CHECKING, Final, NamedTuple

import numpy as np
from numba import njit

# The compiled loop calls the force and load models' primitives - the same direction as `PLAN.md`
# section 6's `torque_curve -> gearbox -> clutch -> differential -> wheels`, read as models feeding
# the integrator rather than the other way round. Layer isolation exists to keep analytics, the
# server and the web layer out of the physics core; a kernel composing the core is the composition.
from f1telemetry.physics import forces, loads  # noqa: TID251 -- the kernel composes the models

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
    "StepOutputs",
    "allocate",
    "allocate_step_outputs",
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

# The per-step diagnostics, as names, in the order `StepOutputs` declares them. Written as data so
# the validation loop and the error messages cannot disagree about which buffer is which, and so a
# field added to the group is one line rather than four.
STEP_OUTPUT_FLOAT_FIELDS: Final[tuple[str, ...]] = ("load_n", "force_x_n", "travel_m")
STEP_OUTPUT_FLAG_FIELD: Final[str] = "travel_limited"
STEP_OUTPUT_FIELDS: Final[tuple[str, ...]] = (*STEP_OUTPUT_FLOAT_FIELDS, STEP_OUTPUT_FLAG_FIELD)
FLOAT64_DTYPE: Final = np.dtype(np.float64)


class StepOutputs(NamedTuple):
    """The caller's per-step corner diagnostics: four ``(steps, 4)`` buffers, one per quantity.

    **A named group of buffers, not a fourth return value.** A ``(steps, 4)`` load and a
    ``(steps, 4)`` longitudinal force are the same shape and the same dtype, so four loose
    arguments would be one refactor away from silently swapped - and a caller that swapped them
    would get a trace that looks finished. Naming them means the kernel and the caller agree on what
    each one holds, and :func:`simulate` can refuse a group that is not this type at all.

    Three ``float64`` fields and one ``int64``: ``travel_limited`` is a 0/1 report about the
    configured mechanical limit rather than a measurement, and a float field would say otherwise
    through its dtype.

    Every buffer is the caller's, and every one is C-contiguous and writeable because
    ``boundscheck=False`` writes into it: a short row count, a missing wheel column, a strided view
    or a read-only buffer is an out-of-bounds write rather than an error. All four are validated
    before the loop starts - see :func:`simulate`.

    Row ``n`` is the step that *started* at trace row ``n`` and produced trace row ``n + 1``, so the
    diagnostics and the trace can be read side by side without an off-by-one: the loads in row ``n``
    are the ones the tyre forces in row ``n`` were computed from.
    """

    load_n: np.ndarray
    force_x_n: np.ndarray
    travel_m: np.ndarray
    travel_limited: np.ndarray


@njit(cache=True, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
def _integrate(
    steps: int,
    dt_s: float,
    mass_kg: float,
    gravity_m_s2: float,
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
    cg_to_front_axle_m: float,
    cg_to_rear_axle_m: float,
    cg_height_m: float,
    axle_track_m: np.ndarray,
    roll_stiffness_front_fraction: float,
    axle_ride_rate_n_per_m: np.ndarray,
    axle_travel_limit_m: np.ndarray,
    load_n_diagnostics: np.ndarray,
    force_x_n_diagnostics: np.ndarray,
    travel_m_diagnostics: np.ndarray,
    travel_limited_diagnostics: np.ndarray,
    record_diagnostics: bool,
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
    # The four-wheel scratch the load model writes, allocated once here and reused by every step.
    # A compiled loop owns no buffers, and `corner_loads_n`/`clamp_travel_to_limits` both take
    # caller-owned vectors precisely so this can live here rather than on the allocator: four small
    # vectors once beats 3 allocations per step times a hundred thousand steps.
    load_scratch_n = np.empty(forces.WHEEL_COUNT, dtype=np.float64)
    base_load_scratch_n = np.empty(forces.WHEEL_COUNT, dtype=np.float64)
    travel_limited_scratch_n = np.empty(forces.WHEEL_COUNT, dtype=np.int64)
    for index in range(steps):
        # Carry the P2-owned states this step does not integrate until a later task owns them. This
        # keeps caller-seeded transients deterministic and prevents uninitialized output columns.
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
        # **The corner loads, from the acceleration the previous step resolved.** The load a tyre
        # is asked about is one 100 us step behind the force it produced, which is the plan's
        # explicit answer to the load/force algebraic loop: the same `previous_acceleration_m_s2`
        # argument `loads.step_loads` takes as a keyword, read here out of the state columns the
        # kernel writes at the end of each step. A run therefore seeds the coupling to zero through
        # `initial_state` and carries it forward itself, and the aero load is distributed by the
        # same CG share as the weight rather than a quarter on each patch - P2-T2's deliberate
        # change of convention, and the reason the loads no longer come from
        # `forces.static_wheel_load_n` at all.
        loads.corner_loads_n(
            load_scratch_n,
            base_load_scratch_n,
            mass_kg,
            gravity_m_s2,
            downforce_n,
            out[index, PREVIOUS_AZ_INDEX],
            out[index, PREVIOUS_AX_INDEX],
            out[index, PREVIOUS_AY_INDEX],
            cg_to_front_axle_m,
            cg_to_rear_axle_m,
            cg_height_m,
            axle_track_m,
            roll_stiffness_front_fraction,
        )
        # Clamping is a result, not a decision: a corner that ran out of travel is put on its stop
        # and flagged, and the load the clamp removed is shared over the corners that still have
        # room so the four still sum to the car's total. Both the flag and the clamped travel are
        # reported rather than folded away.
        loads.clamp_travel_to_limits(
            load_scratch_n,
            base_load_scratch_n,
            axle_ride_rate_n_per_m,
            axle_travel_limit_m,
            travel_limited_scratch_n,
        )
        drivetrain_torque_nm = drive_torque_nm[index]
        # Drag is signed along +x and is negative going forward, so it starts the sum rather than
        # being subtracted; adding it keeps one sign convention instead of two.
        net_force_n = drag_n
        for wheel in range(forces.WHEEL_COUNT):
            column = WHEEL_STATE_OFFSET + wheel
            # The axle a corner belongs to, in the same FL, FR, RL, RR order the load model reads:
            # an axle is the two adjacent corners, so the mapping is derived from the wheel count
            # per axle rather than restated. It is what picks this corner's ride rate below.
            axle = wheel // forces.WHEELS_PER_AXLE_COUNT
            load_n = load_scratch_n[wheel]
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
            if record_diagnostics:
                # Row `index` is the step that started at trace row `index`, so the diagnostics and
                # the trace line up row for row: these are the loads the force below was computed
                # from, not the loads the row that comes out was integrated with.
                load_n_diagnostics[index, wheel] = load_n
                force_x_n_diagnostics[index, wheel] = tyre_fx_n
                travel_m_diagnostics[index, wheel] = loads.suspension_travel_m(
                    load_n, base_load_scratch_n[wheel], axle_ride_rate_n_per_m[axle]
                )
                travel_limited_diagnostics[index, wheel] = travel_limited_scratch_n[wheel]
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
        # **Close the load lag with what this step actually resolved.** Only the longitudinal
        # component exists in this kernel: there is no lateral force and no heave state yet, so
        # `ay` and `az` are written zero rather than left holding a caller seed the loop would keep
        # re-reading forever. Writing the zero is the honest value - a body that is not accelerating
        # sideways or vertically carries no `m * ay` or `m * az` term - and it means a lateral task
        # only has to start writing its own column.
        out[index + 1, PREVIOUS_AX_INDEX] = acceleration_m_s2
        out[index + 1, PREVIOUS_AY_INDEX] = 0.0
        out[index + 1, PREVIOUS_AZ_INDEX] = 0.0
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


def allocate_step_outputs(steps: int) -> StepOutputs:
    """Caller-owned per-step corner diagnostics for a run of ``steps``, ready to pass to simulate.

    Zero-initialised rather than empty, because a caller that passes these and then reads a row the
    run did not reach gets a number that says "nothing happened" instead of a number that says
    "nothing was written". A zero step count is legal and gives four zero-row buffers, which is the
    degenerate case of the same rule rather than a special one.

    Allocated here rather than inside the kernel because a compiled loop must not allocate, and
    allocated by the caller rather than by :func:`simulate` because a run that does not ask for the
    diagnostics should not pay for them.
    """
    count = _checked_steps(steps)
    shape = (count, forces.WHEEL_COUNT)
    return StepOutputs(
        load_n=np.zeros(shape, dtype=np.float64),
        force_x_n=np.zeros(shape, dtype=np.float64),
        travel_m=np.zeros(shape, dtype=np.float64),
        travel_limited=np.zeros(shape, dtype=np.int64),
    )


def simulate(
    config: KernelConfig,
    steps: int,
    state: np.ndarray,
    drive_torque_nm: np.ndarray,
    out: np.ndarray,
    brake_torque_nm: np.ndarray | None = None,
    *,
    step_outputs: StepOutputs | None = None,
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

    ``step_outputs`` is the optional :class:`StepOutputs` group from
    :func:`allocate_step_outputs`, and it is optional for the same reason the load diagnostics are
    not part of the trace: most callers want the state history, not four per-step corner
    quantities. It is keyword-only because a seventh positional argument would be indistinguishable
    from a future input buffer. All four of its buffers are validated here, before the loop, exactly
    like every other buffer - and unlike the others they are optional, so *all* of the group is
    validated or none of it is used. Omitting it changes nothing about the run: the trace is
    byte-identical, and the loads the tyre model needs are computed either way, because they are an
    input to the physics rather than an output of it.
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
                "divides by mass_kg and multiplies by dt_s, and the corner loads are built from "
                "mass_kg * gravity_m_s2, so none of the three is checked for you once the loop "
                "starts"
            )
    # The rest of the configuration, through the two models' own shared validators: one list of
    # names and one set of sign rules for `simulate`, `step_forces` and `step_wheel` rather than
    # three copies that could disagree, and one validated form of every CG, track, ride-rate and
    # travel-limit scalar for the load model. The aero curves are checked here for the same reason -
    # the loop indexes them with `boundscheck=False`.
    values = forces.validated_config_scalars(config, "simulate")
    geometry = loads.validated_load_scalars(config, "simulate")
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
    if step_outputs is None:
        # One row of each buffer, allocated here rather than inside the loop, so the compiled
        # signature stays the same whether or not the caller asked for diagnostics. `record` is
        # false, so the loop never writes to them - and a single row keeps a stray write inside the
        # allocation rather than past it.
        diagnostics = _discarded_step_outputs()
        record_diagnostics = False
    else:
        diagnostics = _checked_step_outputs(step_outputs, count)
        record_diagnostics = True
    return _integrate(
        count,
        config.dt_s,
        config.mass_kg,
        geometry.gravity_m_s2,
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
        geometry.cg_to_front_axle_m,
        geometry.cg_to_rear_axle_m,
        geometry.cg_height_m,
        geometry.axle_track_m,
        geometry.roll_stiffness_front_fraction,
        geometry.axle_ride_rate_n_per_m,
        geometry.axle_travel_limit_m,
        diagnostics.load_n,
        diagnostics.force_x_n,
        diagnostics.travel_m,
        diagnostics.travel_limited,
        record_diagnostics,
    )


def _discarded_step_outputs() -> StepOutputs:
    """A one-row stand-in for a :class:`StepOutputs` the caller did not ask for.

    The compiled loop takes four diagnostic arrays whether or not anyone wants them, because a
    signature that changed shape depending on a Python-level ``if`` would compile two
    specialisations
    of the same integrator and the byte-identity claim would become version-dependent. One row each,
    never written, allocated once per run outside the loop: four tiny Python allocations against a
    kernel that makes thousands of steps.
    """
    shape = (1, forces.WHEEL_COUNT)
    return StepOutputs(
        load_n=np.zeros(shape, dtype=np.float64),
        force_x_n=np.zeros(shape, dtype=np.float64),
        travel_m=np.zeros(shape, dtype=np.float64),
        travel_limited=np.zeros(shape, dtype=np.int64),
    )


def _checked_step_outputs(step_outputs: object, count: int) -> StepOutputs:
    """Four diagnostic buffers the kernel can write into, or a :class:`ValueError` naming the field.

    Every rule :func:`simulate` applies to the buffers it has always checked applies here too, for
    the same reason: the loop writes into these with ``boundscheck=False``, so a wrong shape, a
    strided view or a read-only buffer is an out-of-bounds write rather than an error. The flags
    are checked for ``int64`` rather than ``float64`` because a differently-typed buffer compiles as
    a differently-typed kernel and writes the wrong number of bytes.

    The group itself has to be a :class:`StepOutputs`, and the four may not share memory. Three
    ``(steps, 4)`` float64 buffers passed as loose arguments are one refactor away from being
    swapped, and two of them aliased would produce a run that looks finished with one corner
    quantity quietly overwritten by another - which is exactly the failure this boundary exists to
    refuse.
    """
    if not isinstance(step_outputs, StepOutputs):
        raise ValueError(
            f"simulate: step_outputs must be a StepOutputs from allocate_step_outputs, got "
            f"{type(step_outputs).__name__}. The four buffers are named in the contract and passed "
            "as one group, because a (steps, 4) load and a (steps, 4) longitudinal force have the "
            "same shape and the same dtype and are otherwise indistinguishable at the call site"
        )
    shape = (count, forces.WHEEL_COUNT)
    for name in STEP_OUTPUT_FIELDS:
        dtype = np.dtype(np.int64 if name == STEP_OUTPUT_FLAG_FIELD else np.float64)
        _check_buffer(
            f"step_outputs.{name}",
            getattr(step_outputs, name),
            shape,
            writable=True,
            dtype=dtype,
        )
    for first, second in combinations(STEP_OUTPUT_FIELDS, 2):
        if np.shares_memory(getattr(step_outputs, first), getattr(step_outputs, second)):
            raise ValueError(
                f"simulate: step_outputs.{first} must not share memory with step_outputs.{second}. "
                "Each is a separate destination the kernel writes on the same step, so a shared "
                "buffer would leave one corner quantity overwritten by another in a run that looks "
                "finished"
            )
    return step_outputs


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


def _check_buffer(
    name: str,
    array: object,
    shape: tuple[int, ...],
    *,
    writable: bool,
    dtype: np.dtype = FLOAT64_DTYPE,
) -> None:
    """Refuse a buffer the kernel would read out of bounds or quietly re-quantise.

    ``state`` and ``drive_torque_nm`` are read-only as far as the kernel is concerned, so a caller
    may hand over a buffer it does not want written; ``out`` and the diagnostic buffers are the
    kernel's destinations and must be writable. Contiguity is part of the contract in
    ``car_spec.KernelConfig`` and in ``PLAN.md`` section 4.1, and a strided buffer would compile
    as a
    differently-typed kernel rather than being rejected. ``dtype`` is a parameter rather than a
    constant because the ``travel_limited`` flags are ``int64`` - a 0/1 report, not a measurement -
    and a ``float64`` buffer for them would write the wrong bytes into the wrong kernel.
    """
    if not isinstance(array, np.ndarray):
        raise ValueError(
            f"simulate: {name} must be a numpy ndarray, got {type(array).__name__}. Anything "
            "else would be copied into the kernel, so the buffer the caller holds is not the "
            "one the run writes"
        )
    if array.dtype != dtype:
        raise ValueError(
            f"simulate: {name} buffer must be {dtype.name}, got {array.dtype}. The kernel is "
            f"{dtype.name} end to end and anything else re-quantises the trace or compiles as a "
            "differently-typed kernel"
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
