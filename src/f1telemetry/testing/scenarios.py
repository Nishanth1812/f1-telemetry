"""Phase 1 straight-line scenarios: caller-driven traces over the APIs that already exist.

`PHASES.md` P1-T8 asks for scenarios and `tasks/todo.md` Task 5 asks that each one start from
fixed initial conditions and emit a trace through the existing testing/record pattern. This
module is that path, and it is deliberately thin: the drivetrain comes from
:func:`~f1telemetry.physics.gearbox.step_gearbox` and
:func:`~f1telemetry.physics.powertrain.step_mgu_k`, the four-wheel loop from
:func:`~f1telemetry.kernels.longitudinal.simulate`, and every coefficient from the loaded
``car_spec.yaml``. It tunes nothing and defines no configuration-matched target. The one event
speed-trap observation is only a transient reachability floor, so these scenarios demonstrate
model behaviour rather than validated car performance.

**Seven decisions are worth naming.**

* **The drivetrain is sampled once per control interval and held across it.**
  ``simulate`` takes a caller-owned torque history for the whole run, while the torque for step
  *k* depends on the state at step *k*, so the two cannot both be exact. The runner walks the run
  in control intervals of :data:`CONTROL_STEPS` kernel steps: engine speed and road speed are
  read from the trace at the interval's first row, and the drivetrain is then stepped once per
  kernel step against those held values. The gearbox's shift timer and the energy store are
  still integrated at ``config.dt_s``, because each of those APIs owns its own step - holding
  the torque over a 10 ms interval is what a real torque map does, and it is a stated
  approximation rather than an exact integration of the closed loop.

* **A segment's gear request is one request, not one per step.** ``GearRequest`` is documented
  as a command rather than a level, so the request is passed on the segment's first kernel step
  and every later step in that segment holds. Latching it is what makes the scenario say "the
  driver pulled the paddle once" - without it a single ``UP`` in a one-second segment walks the
  box to top gear as fast as the shift timer allows, which is a scenario bug the gearbox is
  right to refuse to model away.

* **The ICE has a speed state while its clutch is open.** ``ScenarioSegment.ice_rpm_initial`` seeds
  the state from the near-12 000 rpm reported by the 2026 start telemetry. The engine then evolves
  from delivered crank torque, clutch load reflected through the active ratio and configured ICE
  inertia. With a fully engaged clutch the model applies an ideal speed-lock constraint to the rear
  axle; an open clutch or shift cut leaves the engine independent. The seed is a scenario initial
  condition, not a speed hold or a car coefficient.

* **Per-corner inputs are unit-signed bias *magnitudes*.** A throttle bias of ``1.0`` and a brake
  bias of ``1.0`` mean "the pedal as given, on every wheel", which is how a straight-line driver
  and C11.1.2's left/right symmetry actually present. The sign is not part of the bias: a brake
  torque is already signed against forward rotation, so a ``-1`` share would have *driven* the
  wheels. Making them magnitudes rather than absent lets a later scenario bias front against rear
  without this module's shape changing.

* **A record carries what the model computes and nothing else, and each power
  channel names the boundary it is measured at.** ``ice_power_w`` is the ICE's
  *gross shaft power at the crankshaft*: the delivered torque against the engine
  speed the drivetrain sampled, upstream of the gearbox and the clutch. That
  boundary is a choice, made and documented here, and it is deliberate: the
  fuel-energy-flow clauses that bound the ICE (C5.2.3, C5.2.4, C5.2.5) are
  stated against crankshaft power, and the shaft power is exactly derivable
  from the state the runner already computes - the same test
  ``mgu_k_power_w`` passes for its own boundary. It is deliberately *not* the
  power transmitted through the clutch: a shift cut opens the driveline while
  the engine keeps making power, and a slipping clutch transmits less than the
  engine delivers, so the two boundaries differ by the whole clutch. The
  clutch-transmitted quantity is ``drive_torque_nm`` against wheel speed, which
  the run already carries. An ICE-only clutch-transmitted power is not reported
  at all, because the MGU-K joins the crankshaft upstream of the clutch
  (C5.18.2) and the model sums the two sources before it: the torque that
  crosses the clutch belongs to both, and attributing the clutch's limit to one
  of them would invent a split the model does not compute. ``mgu_k_power_w`` is
  the *store-side* electrical power, ``-d(SOC)/dt``, because that is the
  boundary C5.2.7 and C5.2.9 bound and it is exactly derivable from the state
  the MGU-K step wrote. Invariant 6 instead uses wheel-side work because the
  kernel has no engine or motor rotor state.

* **The energy invariant uses the modeled boundary.** The kernel has four wheel states and no
  engine-speed state, so the check balances chassis and wheel kinetic energy against wheel-torque
  work, aerodynamic drag and tyre-slip work. It does not compare crankshaft power directly with
  chassis acceleration, and it is never relabelled as a fuel-to-vehicle conservation law. Every
  interval of every record is computed from those quantities, the last included: the final
  recorded row sits on the run's terminal state, where no interval *starts*, so it reports the
  residual of the final complete control interval - the one that *ends* on that row - and no row
  is a hard-coded pass.

* **Segments are timed to stay inside the tyres' grip, and two of them are seeded rolling.**
  With no traction control (C9.1.2) a drive demand above the Magic Formula's peak has no
  equilibrium: the wheel accelerates, slip passes the peak and the force falls further, so the
  spin runs away. On the committed curves that happens in first and second gear in the middle of
  the speed range and nowhere else, so the ladder scenario upshifts before it gets there and the
  MGU-K scenario runs from a declared rolling state in the top gear - where a full C5.2.11
  deployment is transmissible at all. Both are properties of the synthetic ratios, tyres and
  torque curve rather than of a real car, and neither is a performance figure. Declaring
  :data:`LAUNCH_ICE_RPM` raises the demand through that same first-gear window by construction, and
  whether the grid launch still clears the tyre peak on the uncalibrated curves has not been
  re-measured since it was declared - `tasks/todo.md` Task 5 keeps that open.

Every drivetrain column is reported **for the step that starts at the recorded trace row**, next to
the torque that step produced, so a recorded row is internally consistent and a segment's window
lines up with the steps that segment drove. The state is caller-owned throughout: the kernel
buffer, the gearbox buffer, the MGU-K buffer and the torque histories are allocated here and
stepped in place, no clock or random source is read, and two runs of one scenario are
byte-identical.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import numpy as np

from f1telemetry.kernels import longitudinal  # noqa: TID251 -- scenarios drive the kernel
from f1telemetry.physics import (  # noqa: TID251 -- and the models
    engine,
    forces,
    gearbox,
    powertrain,
)
from f1telemetry.testing.records import (
    CORNERS,
    GroundTruthStep,
    SampleRecord,
    SensorFrame,
    WheelTruth,
)

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import KernelConfig

__all__ = [
    "CONTROL_STEPS",
    "LAUNCH_ICE_RPM",
    "PHASE_ONE_INVARIANTS",
    "DrivetrainTrace",
    "Scenario",
    "ScenarioRun",
    "ScenarioSegment",
    "build_scenarios",
    "run_scenario",
    "scenario",
]

# Kernel steps between two drivetrain samples. At the committed ``dt_s`` of 100 us this is a
# 100 Hz control rate, and it is the resolution at which the drivetrain inputs are read - see
# the module docstring. It is a harness rate, not a car coefficient: nothing in ``car_spec.yaml``
# states one, and the trace does not depend on it being any particular value.
CONTROL_STEPS: Final[int] = 100

# rad/s per rpm, the one conversion a runner needs to report engine speed from a wheel speed.
_RPM_PER_RAD_S: Final[float] = 60.0 / math.tau

# Initial crank speed for the built grid-start scenarios, from the 2026 start telemetry already
# cited on :attr:`ScenarioSegment.ice_rpm_initial`. It seeds an evolving engine state during clutch
# slip; it is not held through the segment and is not a car coefficient.
LAUNCH_ICE_RPM: Final[float] = 12_000.0

# J per MJ. C5.2.9's usable window and the MGU-K's state of charge are both held in MJ, while the
# store-side power is reported in watts because C5.2.7 caps the motor in kW. The delta of that
# buffer over one step is therefore an MJ/s rate, and it has to be converted before it can be
# compared with anything the regulation states.
_J_PER_MJ: Final[float] = 1.0e6

# All eight checks run against each produced Phase 1 scenario record.
PHASE_ONE_INVARIANTS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6, 7, 8)

_UNIT_BRAKE: Final[tuple[float, ...]] = (1.0, 1.0, 1.0, 1.0)
_REQUEST_CODES: Final[frozenset[int]] = frozenset(int(member) for member in gearbox.GearRequest)
_SEGMENT_GRID_TOLERANCE_S: Final[float] = 1.0e-9


@dataclass(frozen=True, slots=True)
class ScenarioSegment:
    """One constant stretch of a scenario: what the driver asks for over ``duration_s``.

    Every field is a declaration about the run rather than a coefficient the physics reads,
    which is why the MGU-K request is its own actuator command - shaft Nm, positive for
    deployment - instead of a share of ``throttle``: the motor has a pedal of its own and the ICE
    does not have it.

    ``ice_rpm_initial`` seeds the ICE speed state at the start of this segment, for cases where the
    crank is not locked to the rear wheels (such as a slipping launch clutch). The engine speed is
    then integrated from configured inertia and net crank torque for each control interval. With a
    fully engaged clutch outside a shift cut, speed follows an ideal gear lock to the rear axle.
    ``None``, the default, keeps the current state or lets that gear-lock constraint determine it.

    It is the last field because the ones above it are positional in existing callers: inserted
    earlier it would silently move ``request`` and every argument after it.
    """

    duration_s: float
    throttle: float = 0.0
    clutch: float = 1.0
    request: gearbox.GearRequest = gearbox.GearRequest.HOLD
    mgu_k_request_nm: float = 0.0
    brake_torque_nm: float = 0.0
    grid_standing_start: bool = False
    overtake: bool = False
    ice_rpm_initial: float | None = None


@dataclass(frozen=True, slots=True)
class Scenario:
    """A named straight-line run from fixed initial conditions.

    ``initial_gear`` seeds the gearbox buffer, and a rolling start seeds every wheel at the speed
    that would roll without slip, so a rolling scenario does not start with a locked axle.
    ``soc_mj`` seeds the MGU-K's ``[state_of_charge, lap_recharge]`` buffer; ``None`` means the
    whole of C5.2.9's window, which is what a car at the line has. ``brake_bias`` holds four
    per-corner share magnitudes.
    """

    name: str
    initial_speed_m_s: float
    segments: tuple[ScenarioSegment, ...]
    description: str
    initial_gear: int = 1
    soc_mj: float | None = None
    brake_bias: tuple[float, ...] = _UNIT_BRAKE


@dataclass(frozen=True, slots=True)
class DrivetrainTrace:
    """What the drivetrain was asked and what it did, at the recorded rate.

    Arrays rather than objects so a check can read a whole stretch of a run without a Python loop
    over it, and so a repeat run can be compared byte for byte. ``accel_m_s2`` is the chassis
    acceleration the kernel produced, which ``GroundTruthStep`` does not carry. ``ice_power_w``
    is the ICE's gross crankshaft shaft power - upstream of the clutch by declaration, see the
    module docstring - while ``drive_torque_nm`` is the post-clutch driveline torque, so the
    power the clutch actually transmits is that torque against wheel speed.
    """

    gear: np.ndarray
    clutch: np.ndarray
    throttle: np.ndarray
    ice_rpm: np.ndarray
    ice_torque_nm: np.ndarray
    ice_power_w: np.ndarray
    mgu_k_power_w: np.ndarray
    drive_torque_nm: np.ndarray
    soc_mj: np.ndarray
    lap_recharge_mj: np.ndarray
    accel_m_s2: np.ndarray


@dataclass(frozen=True, slots=True)
class ScenarioRun:
    """A completed run: the kernel's own trace, the drivetrain's, and the record built from them.

    ``trace`` is the kernel buffer exactly as ``simulate`` left it - the raw, unsummed artefact a
    determinism claim is about. The caller-owned wheel torque histories stay at kernel rate for
    the discrete energy balance and because a shift cut is a 40 ms event while records are 10 ms.
    """

    name: str
    description: str
    dt_s: float
    control_steps: int
    steps: int
    trace: np.ndarray
    drive_torque_nm: np.ndarray
    brake_torque_nm: np.ndarray
    drivetrain: DrivetrainTrace
    record: SampleRecord

    @property
    def record_dt_s(self) -> float:
        """The interval between two recorded steps, in seconds."""
        return self.control_steps * self.dt_s

    @property
    def gears(self) -> tuple[int, ...]:
        """The gear column, one entry per recorded step."""
        return tuple(int(gear) for gear in self.drivetrain.gear)


def build_scenarios(config: KernelConfig) -> Mapping[str, Scenario]:
    """The Phase 1 scenario set, every input of it read from ``config``.

    The two inputs a scenario cannot invent for itself come from the car file: the brake torque
    is the rear-wheel share of this car's own first-gear full-engagement launch torque, and the
    MGU-K request is C5.2.11's crank-referenced limit expressed at the motor shaft. Both are
    synthetic *selections* of configured values rather than new coefficients, and neither is a
    claim about how fast the car stops or how hard it deploys. The third, :data:`LAUNCH_ICE_RPM`,
    is not from the car file at all and says so where it is defined.
    """
    brake_nm = -_rear_wheel_torque_nm(config)
    mgu_k_nm = config.mgu_k_torque_limit_nm / config.mgu_k_crankshaft_ratio
    top_gear = int(config.gear_ratios.size)
    built = (
        Scenario(
            name="standing_launch",
            initial_speed_m_s=0.0,
            description=(
                "Grid standing start: clutch released on a partial pedal, then applied, then a "
                "partial-throttle settle, with the initial slipping segment declaring the near-"
                "12 000 rpm reported by start telemetry (LAUNCH_ICE_RPM) rather than idling. "
                "First gear throughout, no request, no MGU-K."
            ),
            segments=(
                ScenarioSegment(
                    0.5,
                    throttle=0.25,
                    clutch=0.75,
                    grid_standing_start=True,
                    ice_rpm_initial=LAUNCH_ICE_RPM,
                ),
                ScenarioSegment(
                    1.0,
                    throttle=1.0,
                    clutch=1.0,
                    grid_standing_start=True,
                ),
                ScenarioSegment(0.5, throttle=0.2, clutch=1.0),
            ),
        ),
        Scenario(
            name="accelerate_to_speed",
            initial_speed_m_s=0.0,
            description=(
                "Standing start with LAUNCH_ICE_RPM declared during the initial clutch-slip "
                "segment, then four "
                "driver-requested upshifts and sustained full throttle; the trace is long enough "
                "to measure its first 100 km/h crossing. Past the launch every segment takes its "
                "engine speed from the wheels again."
            ),
            segments=(
                ScenarioSegment(
                    0.5,
                    throttle=0.25,
                    clutch=0.75,
                    grid_standing_start=True,
                    ice_rpm_initial=LAUNCH_ICE_RPM,
                ),
                ScenarioSegment(
                    1.0,
                    throttle=1.0,
                    grid_standing_start=True,
                ),
                ScenarioSegment(1.0, throttle=1.0, request=gearbox.GearRequest.UP),
                ScenarioSegment(1.0, throttle=1.0, request=gearbox.GearRequest.UP),
                ScenarioSegment(1.0, throttle=1.0, request=gearbox.GearRequest.UP),
                ScenarioSegment(1.0, throttle=1.0, request=gearbox.GearRequest.UP),
                ScenarioSegment(1.5, throttle=1.0),
            ),
        ),
        Scenario(
            name="full_throttle_shifts",
            initial_speed_m_s=12.0,
            description=(
                "Rolling start at 12 m/s, full throttle, four driver up-shift requests through a "
                "gear ladder that stays inside the tyres' grip. Nothing else changes gear: rpm "
                "alone cannot (C9.8.1)."
            ),
            segments=(
                ScenarioSegment(0.6, throttle=1.0),
                ScenarioSegment(1.0, throttle=1.0, request=gearbox.GearRequest.UP),
                ScenarioSegment(1.2, throttle=1.0, request=gearbox.GearRequest.UP),
                ScenarioSegment(1.2, throttle=1.0, request=gearbox.GearRequest.UP),
                ScenarioSegment(1.2, throttle=1.0, request=gearbox.GearRequest.UP),
                ScenarioSegment(0.8, throttle=1.0),
            ),
        ),
        Scenario(
            name="full_throttle",
            initial_speed_m_s=12.0,
            description=(
                "Rolling full-throttle run with one driver request for each upshift through "
                "eighth gear, a store-bounded MGU-K deployment during the top-gear acceleration, "
                "then a sustained ICE-only top-gear tail for the terminal-speed check. The "
                "deployment and the tail are separate segments on purpose: the maximum speed the "
                "run reaches and the speed it settles at are different quantities measured over "
                "different stretches, so neither is substituted for the other."
            ),
            segments=(
                ScenarioSegment(0.6, throttle=1.0),
                *(
                    ScenarioSegment(1.2, throttle=1.0, request=gearbox.GearRequest.UP)
                    for _ in range(7)
                ),
                ScenarioSegment(6.0, throttle=1.0),
                ScenarioSegment(20.0, throttle=1.0, mgu_k_request_nm=mgu_k_nm),
                ScenarioSegment(41.0, throttle=1.0),
            ),
        ),
        Scenario(
            name="coast_neutral",
            initial_speed_m_s=12.0,
            description=(
                "Rolling start, full throttle in first gear through a partially engaged clutch, "
                "then neutral with the pedal still "
                "down, then first gear again. Only drag acts in neutral."
            ),
            segments=(
                ScenarioSegment(0.6, throttle=1.0, clutch=0.1),
                ScenarioSegment(1.0, throttle=1.0, clutch=0.0, request=gearbox.GearRequest.NEUTRAL),
                ScenarioSegment(0.8, throttle=1.0, request=gearbox.GearRequest.UP),
            ),
        ),
        Scenario(
            name="braking",
            initial_speed_m_s=25.0,
            description=(
                "Rolling start, a short partial-throttle approach, then caller-supplied signed "
                "brake torque on all four wheels. Equal four-wheel torque is the synthetic "
                "symmetry assumption; brake capacity, bias and hydraulics are not modelled."
            ),
            segments=(
                ScenarioSegment(0.4, throttle=0.3),
                ScenarioSegment(1.2, brake_torque_nm=brake_nm),
            ),
        ),
        Scenario(
            name="mgu_k_standing_start",
            initial_speed_m_s=0.0,
            soc_mj=1.0,
            description=(
                "Grid standing start that asks for full MGU-K deployment from a part-charged store "
                "and never leaves first gear below C5.2.12's 50 km/h threshold, so the motor is "
                "blocked for the whole run while the ICE launches the car."
            ),
            segments=(
                ScenarioSegment(
                    0.8, throttle=1.0, mgu_k_request_nm=mgu_k_nm, grid_standing_start=True
                ),
                ScenarioSegment(
                    1.2, throttle=1.0, mgu_k_request_nm=mgu_k_nm, grid_standing_start=True
                ),
            ),
        ),
        Scenario(
            name="mgu_k_deploy_regen",
            initial_speed_m_s=40.0,
            initial_gear=top_gear,
            soc_mj=1.5,
            description=(
                "Declared rolling state in top gear with the ICE closed: full deployment drains "
                "the store at the C5.2.7 electrical cap, then full regeneration refills it "
                "against C5.2.10's lap budget. The rolling start is what makes a full deployment "
                "transmissible at all on these ratios and tyres."
            ),
            segments=(
                ScenarioSegment(1.2, mgu_k_request_nm=mgu_k_nm),
                ScenarioSegment(1.2, mgu_k_request_nm=-mgu_k_nm),
            ),
        ),
    )
    return MappingProxyType({entry.name: entry for entry in built})


def scenario(config: KernelConfig, name: str) -> Scenario:
    """One named scenario from :func:`build_scenarios`, by name."""
    suite = build_scenarios(config)
    if name not in suite:
        msg = f"unknown scenario {name!r}; the suite has {', '.join(sorted(suite))}"
        raise KeyError(msg)
    return suite[name]


def run_scenario(
    config: KernelConfig,
    plan: Scenario,
    *,
    control_steps: int = CONTROL_STEPS,
) -> ScenarioRun:
    """Run one scenario and return its trace, its drivetrain history and its record.

    The only entry point, and the only place the two rates are reconciled: the drivetrain is
    stepped once per kernel step, and its *inputs* are read once per control interval from the
    trace row the interval starts on. See the module docstring for why that is the arrangement.

    Refused before any simulation starts: a duration that is not a whole number of kernel steps
    or of control intervals, a pedal outside ``[0, 1]``, a gear request that is not a
    ``GearRequest``, a nonfinite MGU-K request, brake torque that is not signed against forward
    rotation, a bias vector of the wrong length, and an initial speed or gear that the drivetrain
    could not start from.
    """
    counts = _checked_segments(plan, config, control_steps)
    total = sum(counts)

    trace = np.zeros((total + 1, longitudinal.STATE_SIZE), dtype=np.float64)
    drive = np.zeros(total, dtype=np.float64)
    brake_torque = np.zeros((total, forces.WHEEL_COUNT), dtype=np.float64)
    rpm = np.zeros(total, dtype=np.float64)
    ice = np.zeros(total, dtype=np.float64)
    mgu_k_w = np.zeros(total, dtype=np.float64)
    gear = np.zeros(total, dtype=np.int64)
    clutch = np.zeros(total, dtype=np.float64)
    soc = np.zeros(total, dtype=np.float64)
    recharge = np.zeros(total, dtype=np.float64)
    throttle = np.zeros(total, dtype=np.float64)

    gear_state = gearbox.initial_state(float(plan.initial_gear))
    mgu_k_state = powertrain.mgu_k_initial_state(config, plan.soc_mj)
    engine_rpm = config.idle_rpm
    state = longitudinal.initial_state(
        speed_m_s=plan.initial_speed_m_s,
        wheel_omega_rad_s=plan.initial_speed_m_s / config.rolling_radius_m,
    )
    trace[0] = state

    row = 0
    mgu_k_cap_w = config.mgu_k_peak_power_kw * 1_000.0
    for segment, count in zip(plan.segments, counts, strict=True):
        if segment.ice_rpm_initial is not None:
            engine_rpm = float(segment.ice_rpm_initial)
        gear_state[gearbox.CLUTCH_INDEX] = segment.clutch
        brake = _brake_history(segment, plan.brake_bias, count)
        brake_torque[row : row + count] = brake
        for interval in range(0, count, control_steps):
            interval_start = row + interval
            sample = state
            shifting_at_start = gear_state[gearbox.SHIFT_TIMER_INDEX] > 0.0
            shift_requested = interval == 0 and int(segment.request) != int(
                gearbox.GearRequest.HOLD
            )
            starting_gear = int(gear_state[gearbox.GEAR_INDEX])
            clutch_open = segment.clutch < 1.0 or starting_gear == gearbox.NEUTRAL_GEAR
            engine_is_free = clutch_open or shifting_at_start or shift_requested
            if engine_is_free:
                if not clutch_open:
                    # Entering a shift cut from a locked clutch preserves the coupled speed as the
                    # initial condition of the newly free engine state.
                    engine_rpm = _wheel_coupled_ice_rpm(config, sample, gear_state)
                sampled_rpm = engine_rpm
            else:
                sampled_rpm = _wheel_coupled_ice_rpm(config, sample, gear_state)
                engine_rpm = sampled_rpm
            sampled_torque = powertrain.step_ice_torque(config, sampled_rpm, segment.throttle)
            sample_speed = float(sample[longitudinal.V_INDEX])
            torque = np.zeros(control_steps, dtype=np.float64)
            delivered_torque_sum = 0.0
            load_torque_sum = 0.0
            for offset in range(control_steps):
                position = interval_start + offset
                throttle[position] = segment.throttle
                charge_before = float(mgu_k_state[powertrain.SOC_INDEX])
                sampled_mgu_k = powertrain.step_mgu_k(
                    config,
                    mgu_k_state,
                    segment.mgu_k_request_nm,
                    sampled_rpm,
                    sample_speed,
                    grid_standing_start=segment.grid_standing_start,
                    overtake=segment.overtake,
                )
                torque[offset] = gearbox.step_gearbox(
                    config,
                    gear_state,
                    sampled_rpm,
                    segment.throttle,
                    segment.request if interval == 0 and offset == 0 else gearbox.GearRequest.HOLD,
                    mgu_k_torque_nm=sampled_mgu_k,
                )
                delivered_torque_sum += sampled_torque + sampled_mgu_k
                current_gear = int(gear_state[gearbox.GEAR_INDEX])
                if current_gear >= 1:
                    total_ratio = config.gear_ratios[current_gear - 1] * config.final_drive
                elif current_gear == gearbox.REVERSE_GEAR:
                    total_ratio = -config.reverse_ratio * config.final_drive
                else:
                    total_ratio = 0.0
                if total_ratio != 0.0:
                    load_torque_sum += torque[offset] / total_ratio
                drive[position] = torque[offset]
                rpm[position] = sampled_rpm
                ice[position] = sampled_torque
                reported_power_w = (
                    (charge_before - float(mgu_k_state[powertrain.SOC_INDEX]))
                    * _J_PER_MJ
                    / config.dt_s
                )
                # Finite differencing accumulated SOC can exceed the enforced cap by roundoff.
                mgu_k_w[position] = min(mgu_k_cap_w, max(-mgu_k_cap_w, reported_power_w))
                gear[position] = int(gear_state[gearbox.GEAR_INDEX])
                clutch[position] = float(gear_state[gearbox.CLUTCH_INDEX])
                soc[position] = float(mgu_k_state[powertrain.SOC_INDEX])
                recharge[position] = float(mgu_k_state[powertrain.LAP_RECHARGE_INDEX])
            out = longitudinal.simulate(
                config,
                control_steps,
                state,
                torque,
                longitudinal.allocate(control_steps),
                brake[interval : interval + control_steps],
            )
            trace[interval_start + 1 : interval_start + control_steps + 1] = out[1:]
            state = out[control_steps].copy()
            if engine_is_free:
                engine_rpm = engine.step_engine_speed(
                    config,
                    sampled_rpm,
                    delivered_torque_sum / control_steps,
                    load_torque_sum / control_steps,
                    dt_s=control_steps * config.dt_s,
                )
            else:
                engine_rpm = _wheel_coupled_ice_rpm(config, state, gear_state)
        row += count

    accel = np.concatenate((np.diff(trace[:, longitudinal.V_INDEX]) / config.dt_s, np.zeros(1)))
    recorded = _held(rpm, control_steps, total).size
    drivetrain = DrivetrainTrace(
        gear=_held(gear, control_steps, total),
        clutch=_held(clutch, control_steps, total),
        throttle=_held(throttle, control_steps, total),
        ice_rpm=_held(rpm, control_steps, total),
        ice_torque_nm=_held(ice, control_steps, total),
        ice_power_w=(
            _held(ice, control_steps, total) * _held(rpm, control_steps, total) * (math.tau / 60.0)
        ),
        mgu_k_power_w=_held(mgu_k_w, control_steps, total),
        drive_torque_nm=_held(drive, control_steps, total),
        soc_mj=_held(soc, control_steps, total),
        lap_recharge_mj=_held(recharge, control_steps, total),
        accel_m_s2=_held(accel, control_steps, total),
    )
    if len(drivetrain.gear) != recorded:
        msg = f"{plan.name}: the drivetrain history and the record disagree on length"
        raise AssertionError(msg)
    return ScenarioRun(
        name=plan.name,
        description=plan.description,
        dt_s=config.dt_s,
        control_steps=control_steps,
        steps=total,
        trace=trace,
        drive_torque_nm=drive,
        brake_torque_nm=brake_torque,
        drivetrain=drivetrain,
        record=_build_record(plan, config, trace, drivetrain, drive, brake_torque, control_steps),
    )


def _held(values: np.ndarray, control_steps: int, total: int) -> np.ndarray:
    """A per-kernel-step array decimated to the recorded rate.

    Entry ``k`` is the step that *starts* at trace row ``k``, so the last row's step does not
    exist and the final entry repeats the last one. Shifting rather than sampling at ``k`` is what
    puts a segment's recorded window on the steps that segment actually drove.
    """
    return np.concatenate((values[:total:control_steps], values[-1:]))


def _wheel_coupled_ice_rpm(
    config: KernelConfig,
    row: np.ndarray,
    gear_state: np.ndarray,
) -> float:
    """Crank speed imposed by an ideally locked clutch in ``row``'s wheel state."""
    wheel_omega = 0.5 * (row[longitudinal.RL_WHEEL_INDEX] + row[longitudinal.RR_WHEEL_INDEX])
    gear = int(gear_state[gearbox.GEAR_INDEX])
    ratio = (
        config.gear_ratios[gear - 1] * config.final_drive
        if gear >= 1
        else config.reverse_ratio * config.final_drive
    )
    coupled_rpm = abs(wheel_omega * ratio) * _RPM_PER_RAD_S
    return float(min(config.rev_limit_rpm, max(config.idle_rpm, coupled_rpm)))


def _brake_history(segment: ScenarioSegment, bias: tuple[float, ...], count: int) -> np.ndarray:
    """The signed per-wheel brake torque history for one segment, shape ``(count, 4)``.

    The shares are magnitudes: ``segment.brake_torque_nm`` is already signed against forward
    rotation, so applying a ``-1`` share to it would hand the wheels a driving torque and the
    trace would accelerate under braking.
    """
    if segment.brake_torque_nm == 0.0:
        return np.zeros((count, forces.WHEEL_COUNT), dtype=np.float64)
    per_wheel = np.array([segment.brake_torque_nm * abs(share) for share in bias], dtype=np.float64)
    return np.tile(per_wheel, (count, 1))


def _build_record(
    plan: Scenario,
    config: KernelConfig,
    trace: np.ndarray,
    drivetrain: DrivetrainTrace,
    drive_torque_nm: np.ndarray,
    brake_torque_nm: np.ndarray,
    control_steps: int,
) -> SampleRecord:
    """Assemble the :class:`SampleRecord` a scenario produces, decimated to the recorded rate.

    The per-corner numbers are recomputed through the same public primitives the kernel loop
    calls, at the state each recorded row holds - so the record agrees with the trace it was built
    from rather than describing a second, slightly different run. ``vy``, ``ay``, ``az``,
    ``steer_rad``, ``alpha_rad``, ``fy_n`` and ``camber_deg`` are exactly zero: this slice has no
    lateral or vertical dynamics to report, and a fabricated value for any of them would make the
    record claim something the model did not compute.
    """
    values = forces.validated_config_scalars(config, "scenarios")
    weight_n = config.mass_kg * config.gravity_m_s2
    steps: list[GroundTruthStep] = []
    frames: list[SensorFrame] = []
    for index in range(len(drivetrain.gear)):
        row = index * control_steps
        speed = float(trace[row, longitudinal.V_INDEX])
        downforce_n, drag_n = forces.aero_forces(
            speed,
            values["air_density_kg_m3"],
            values["reference_area_m2"],
            config.aero_speed_m_s,
            config.cl,
            config.cd,
        )
        wheels: list[WheelTruth] = []
        for wheel in range(forces.WHEEL_COUNT):
            omega = float(trace[row, longitudinal.WHEEL_STATE_OFFSET + wheel])
            load_n = (
                forces.static_wheel_load_n(weight_n, values["front_weight_fraction"], wheel)
                + downforce_n / forces.WHEEL_COUNT
            )
            kappa = forces.slip_ratio(
                omega * values["rolling_radius_m"],
                speed,
                values["slip_ratio_min_speed_m_s"],
            )
            wheels.append(
                WheelTruth(
                    fz_n=load_n,
                    fx_n=forces.wheel_tyre_force_n(
                        speed,
                        omega,
                        load_n,
                        values["rolling_radius_m"],
                        values["slip_ratio_min_speed_m_s"],
                        values["pacejka_b"],
                        values["pacejka_c"],
                        values["pacejka_e"],
                        values["pacejka_mu"],
                    ),
                    fy_n=0.0,
                    mu=values["pacejka_mu"],
                    kappa=kappa,
                    alpha_rad=0.0,
                    camber_deg=0.0,
                )
            )
        accel = float(drivetrain.accel_m_s2[index])
        step = GroundTruthStep(
            t_s=index * config.dt_s * control_steps,
            vx_m_s=speed,
            vy_m_s=0.0,
            ax_m_s2=accel,
            ay_m_s2=0.0,
            az_m_s2=0.0,
            gear=int(drivetrain.gear[index]),
            clutch=float(drivetrain.clutch[index]),
            throttle_pct=float(drivetrain.throttle[index]) * 100.0,
            ice_power_w=float(drivetrain.ice_power_w[index]),
            mgu_k_power_w=float(drivetrain.mgu_k_power_w[index]),
            drag_w=drag_n * speed,
            downforce_n=downforce_n,
            steer_rad=0.0,
            energy_residual_fraction=_energy_residual_fraction(
                trace,
                drive_torque_nm,
                brake_torque_nm,
                row,
                control_steps,
                config,
                values,
            ),
            wheels=(wheels[0], wheels[1], wheels[2], wheels[3]),
        )
        steps.append(step)
        frames.append(
            SensorFrame(
                t_s=step.t_s,
                values=_frame_values(step, wheels, drivetrain, index, config, trace, row),
            )
        )
    return SampleRecord(
        name=plan.name,
        dt_s=config.dt_s * control_steps,
        description=(
            f"{plan.description} Produced by f1telemetry.testing.scenarios at the "
            f"{config.dt_s * control_steps!r} s record rate; every value is recomputed from the "
            "kernel trace at the row it describes."
        ),
        ground_truth=tuple(steps),
        frames=tuple(frames),
    )


def _energy_residual_fraction(
    trace: np.ndarray,
    drive_torque_nm: np.ndarray,
    brake_torque_nm: np.ndarray,
    start: int,
    count: int,
    config: KernelConfig,
    values: Mapping[str, float],
) -> float:
    """Discrete energy residual for the closed chassis and four-wheel loop.

    Work uses midpoint velocity and wheel speed, matching the explicit Euler update exactly.
    Wheel torque supplies energy; aero drag and tyre slip remove it. ICE/MGU-K crank power is
    intentionally excluded because P1 has no engine or driveline rotational state.

    The interval is ``[start, start + count)``. The run's final recorded row sits on the
    terminal state, where no interval starts, so it reports the residual of the final
    complete control interval - the one that *ends* on that row, and the same interval the
    row before it describes - by shifting the window back one interval. That interval is
    computed from the same quantities as every other one: no row is a hard-coded pass.
    """
    if start + count >= trace.shape[0]:
        # Only the final recorded row lands here, and it is the one case in
        # which the interval has to be read backwards: the row describes the
        # run's terminal state, so the residual it can truthfully carry is the
        # final complete interval's, the one ending on that state. Every
        # segment is a whole number of control intervals, so `start` (the
        # terminal row) is at least `count`, and the shifted window stays
        # inside the trace.
        start = trace.shape[0] - 1 - count
    initial = trace[start]
    final = trace[start + count]
    wheel_columns = range(longitudinal.WHEEL_STATE_OFFSET, longitudinal.STATE_SIZE)
    kinetic_change = (
        0.5
        * config.mass_kg
        * (final[longitudinal.V_INDEX] ** 2 - initial[longitudinal.V_INDEX] ** 2)
    )
    inertia = values["wheel_inertia_kg_m2"]
    kinetic_change += (
        0.5
        * inertia
        * math.fsum(float(final[column] ** 2 - initial[column] ** 2) for column in wheel_columns)
    )
    torque_work = 0.0
    drag_work = 0.0
    tyre_slip_work = 0.0
    dt_s = config.dt_s
    for index in range(start, start + count):
        before = trace[index]
        after = trace[index + 1]
        speed_mid = 0.5 * (before[longitudinal.V_INDEX] + after[longitudinal.V_INDEX])
        downforce_n, drag_n = forces.aero_forces(
            float(before[longitudinal.V_INDEX]),
            values["air_density_kg_m3"],
            values["reference_area_m2"],
            config.aero_speed_m_s,
            config.cl,
            config.cd,
        )
        drag_work += drag_n * speed_mid * dt_s
        for wheel in range(forces.WHEEL_COUNT):
            column = longitudinal.WHEEL_STATE_OFFSET + wheel
            omega_mid = 0.5 * (before[column] + after[column])
            torque_work += (
                (
                    forces.wheel_drive_torque_nm(wheel, float(drive_torque_nm[index]))
                    + brake_torque_nm[index, wheel]
                )
                * omega_mid
                * dt_s
            )
            load_n = forces.static_wheel_load_n(
                config.mass_kg * config.gravity_m_s2,
                values["front_weight_fraction"],
                wheel,
            )
            load_n += downforce_n / forces.WHEEL_COUNT
            fx_n = forces.wheel_tyre_force_n(
                float(before[longitudinal.V_INDEX]),
                float(before[column]),
                load_n,
                values["rolling_radius_m"],
                values["slip_ratio_min_speed_m_s"],
                values["pacejka_b"],
                values["pacejka_c"],
                values["pacejka_e"],
                values["pacejka_mu"],
            )
            tyre_slip_work += fx_n * (omega_mid * values["rolling_radius_m"] - speed_mid) * dt_s
    accounted_work = torque_work + drag_work - tyre_slip_work
    scale = max(abs(kinetic_change), abs(accounted_work), 1.0)
    return abs(kinetic_change - accounted_work) / scale


def _frame_values(
    step: GroundTruthStep,
    wheels: Sequence[WheelTruth],
    drivetrain: DrivetrainTrace,
    index: int,
    config: KernelConfig,
    trace: np.ndarray,
    row: int,
) -> dict[str, float]:
    """The contract channels one produced step publishes, and nothing else.

    Every name here is a real channel and every value is one this run computed: the channels
    ``channels.yaml`` declares for parts of the car P1 has no model for - brake pressure, flap
    angles, temperatures, boost pressure - are absent rather than published as zeros, because a
    published zero is a claim that the sensor read zero.
    """
    return {
        "speed": step.vx_m_s * 3.6,
        "vx": step.vx_m_s,
        "accel_longitudinal": step.ax_m_s2,
        "ice_rpm": float(drivetrain.ice_rpm[index]),
        "ice_torque_nm": float(drivetrain.ice_torque_nm[index]),
        "mgu_k_rpm": float(drivetrain.ice_rpm[index]) * config.mgu_k_crankshaft_ratio,
        "mgu_k_power_kw": step.mgu_k_power_w / 1_000.0,
        "gear": float(step.gear),
        "throttle_pct": step.throttle_pct,
        "clutch_pct": step.clutch * 100.0,
        "downforce_n": step.downforce_n,
        **{
            f"wheel_speed_{corner.lower()}": float(
                trace[row, longitudinal.WHEEL_STATE_OFFSET + wheel]
            )
            * config.rolling_radius_m
            * 3.6
            for wheel, corner in enumerate(CORNERS)
        },
        **{
            f"slip_ratio_{corner.lower()}": wheels[wheel].kappa * 100.0
            for wheel, corner in enumerate(CORNERS)
        },
        **{
            f"vertical_load_{corner.lower()}": wheels[wheel].fz_n
            for wheel, corner in enumerate(CORNERS)
        },
    }


def _rear_wheel_torque_nm(config: KernelConfig) -> float:
    """The rear-wheel share of this car's own first-gear, full-engagement launch torque.

    Read through the drivetrain rather than written down, which is what makes the braking
    scenario a data edit: ``car_spec.yaml``'s torque curve, ratios and final drive set it, and
    nothing here states a brake capacity or a stopping distance.
    """
    launch = gearbox.step_gearbox(config, gearbox.initial_state(), config.idle_rpm, 1.0)
    return launch * forces.REAR_DRIVE_SHARE


def _checked_segments(plan: Scenario, config: KernelConfig, control_steps: int) -> tuple[int, ...]:
    """The kernel step count of every segment, or a :class:`ValueError` naming one of them."""
    if isinstance(control_steps, bool) or not isinstance(control_steps, int) or control_steps < 1:
        msg = f"{plan.name}: control_steps must be an int >= 1, got {control_steps!r}"
        raise ValueError(msg)
    speed = plan.initial_speed_m_s
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed):
        msg = f"{plan.name}: initial_speed_m_s must be a finite number, got {speed!r}"
        raise ValueError(msg)
    if speed < 0.0:
        msg = f"{plan.name}: initial_speed_m_s must be >= 0, got {speed!r}"
        raise ValueError(msg)
    gear_count = float(config.gear_ratios.size)
    if (
        isinstance(plan.initial_gear, bool)
        or not isinstance(plan.initial_gear, int)
        or not float(gearbox.REVERSE_GEAR) <= float(plan.initial_gear) <= gear_count
    ):
        msg = (
            f"{plan.name}: initial_gear must be a whole gear in "
            f"{int(gearbox.REVERSE_GEAR)}..{int(gear_count)}, got {plan.initial_gear!r}. The "
            "forward part indexes the ratio table the drivetrain reads on its first step"
        )
        raise ValueError(msg)
    for label, bias in (("brake_bias", plan.brake_bias),):
        if len(bias) != forces.WHEEL_COUNT:
            msg = (
                f"{plan.name}: {label} must name {forces.WHEEL_COUNT} corners in the order "
                f"{', '.join(CORNERS)}, got {len(bias)}"
            )
            raise ValueError(msg)
        for share in bias:
            if not math.isfinite(share) or share < 0.0:
                msg = f"{plan.name}: {label} shares must be finite and >= 0, got {share!r}"
                raise ValueError(msg)
    if plan.soc_mj is not None and not math.isfinite(plan.soc_mj):
        msg = f"{plan.name}: soc_mj must be a finite number, got {plan.soc_mj!r}"
        raise ValueError(msg)
    if not plan.segments:
        msg = f"{plan.name}: a scenario needs at least one segment"
        raise ValueError(msg)
    return tuple(
        _checked_segment(plan.name, index, segment, config, control_steps)
        for index, segment in enumerate(plan.segments)
    )


def _checked_segment(
    name: str,
    index: int,
    segment: ScenarioSegment,
    config: KernelConfig,
    control_steps: int,
) -> int:
    """One segment's kernel step count, or a :class:`ValueError` naming that segment."""
    label = f"{name}: segment {index}"
    duration = segment.duration_s
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        msg = f"{label}: duration_s must be a real number, got {duration!r}"
        raise ValueError(msg)
    if not math.isfinite(duration) or duration <= 0.0:
        msg = f"{label}: duration_s must be finite and > 0, got {duration!r}"
        raise ValueError(msg)
    count = round(duration / config.dt_s)
    if count < 1 or abs(count * config.dt_s - duration) > _SEGMENT_GRID_TOLERANCE_S:
        msg = (
            f"{label}: duration_s={duration!r} is not a whole number of {config.dt_s!r} kernel "
            "steps. The integrator is fixed step (PLAN.md section 4.1), so a segment length has "
            "to land on the step grid rather than be rounded by the runner"
        )
        raise ValueError(msg)
    if count % control_steps:
        msg = (
            f"{label}: {count} steps is not a whole number of {control_steps}-step control "
            "intervals. A segment's inputs are held across a whole interval, so one that ended "
            "inside an interval would leave a step the record rate never samples"
        )
        raise ValueError(msg)
    for pedal in (segment.throttle, segment.clutch):
        if (
            isinstance(pedal, bool)
            or not isinstance(pedal, (int, float))
            or not math.isfinite(pedal)
        ):
            msg = f"{label}: throttle and clutch must be finite numbers, got {pedal!r}"
            raise ValueError(msg)
        if not 0.0 <= pedal <= 1.0:
            msg = f"{label}: throttle and clutch must be in [0, 1], got {pedal!r}"
            raise ValueError(msg)
    if isinstance(segment.request, bool) or not isinstance(
        segment.request, (gearbox.GearRequest, int)
    ):
        msg = f"{label}: request must be a GearRequest or its integer code, got {segment.request!r}"
        raise ValueError(msg)
    if int(segment.request) not in _REQUEST_CODES:
        msg = (
            f"{label}: request must be one of "
            f"{', '.join(member.name for member in gearbox.GearRequest)}, "
            f"got {int(segment.request)!r}"
        )
        raise ValueError(msg)
    initial_rpm = segment.ice_rpm_initial
    if initial_rpm is not None:
        if (
            isinstance(initial_rpm, bool)
            or not isinstance(initial_rpm, (int, float))
            or not math.isfinite(initial_rpm)
        ):
            msg = f"{label}: ice_rpm_initial must be finite, got {initial_rpm!r}"
            raise ValueError(msg)
        if not config.idle_rpm <= float(initial_rpm) <= config.rev_limit_rpm:
            msg = (
                f"{label}: ice_rpm_initial must be within the configured idle..rev limit of "
                f"{config.idle_rpm!r}..{config.rev_limit_rpm!r} rpm, got {initial_rpm!r}. The "
                "engine state must start inside its configured operating band"
            )
            raise ValueError(msg)
        if not segment.grid_standing_start or segment.clutch >= 1.0:
            msg = (
                f"{label}: ice_rpm_initial is only valid for a grid-start segment with clutch slip"
            )
            raise ValueError(msg)
    for quantity, value in (
        ("mgu_k_request_nm", segment.mgu_k_request_nm),
        ("brake_torque_nm", segment.brake_torque_nm),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            msg = f"{label}: {quantity} must be finite, got {value!r}"
            raise ValueError(msg)
    if segment.brake_torque_nm > 0.0:
        msg = (
            f"{label}: brake torque is signed against forward rotation, so it must be <= 0, got "
            f"{segment.brake_torque_nm!r}. Positive torque on the wheels is drive_torque_nm, "
            "which is what the drivetrain returns"
        )
        raise ValueError(msg)
    for flag in (segment.grid_standing_start, segment.overtake):
        if not isinstance(flag, bool):
            msg = f"{label}: grid_standing_start and overtake must be True or False, got {flag!r}"
            raise ValueError(msg)
    return count
