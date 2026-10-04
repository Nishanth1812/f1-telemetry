"""The 8 blocking invariants of PLAN.md section 11, as explicit checkers (P0-T7).

Each checker takes a :class:`~f1telemetry.testing.records.SampleRecord` and returns an
:class:`InvariantResult`. None of them raises on bad data - a violated invariant is data,
not an exception, and the whole point is to report every violation in one pass.

``InvariantResult.backing`` names the phase in which the checker stops being a fixture
check and becomes a physics-backed check. Until then each one validates the *semantics*
of its invariant against supplied values: that the formula, the sign convention and the
tolerance are right, so that when P1/P2 feed real traces in, a failure means a physics bug
rather than a disagreement about what the invariant means.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, override

from f1telemetry.contracts.car_spec import CarSpec
from f1telemetry.testing.records import CORNERS, SampleRecord

__all__ = [
    "INVARIANTS",
    "Invariant",
    "InvariantResult",
    "Violation",
    "check_determinism",
    "check_energy_balance",
    "check_finite",
    "check_friction_ellipse",
    "check_gearbox_progression",
    "check_sign_conventions",
    "check_symmetry",
    "check_vertical_load_sum",
    "run_all",
]

_FRICTION_TOLERANCE: Final[float] = 1.0 + 1.0e-12
_LOAD_RELATIVE_TOLERANCE: Final[float] = 1.0e-3
_LOAD_ABSOLUTE_FLOOR_N: Final[float] = 1.0
_ENERGY_RESIDUAL_LIMIT: Final[float] = 0.01
_SYMMETRY_TOLERANCE: Final[float] = 1.0e-9
_SYMMETRY_LONGITUDINAL_FORCE_ABSOLUTE_N: Final[float] = 1.0
_SYMMETRY_LATERAL_FORCE_ABSOLUTE_N: Final[float] = 1.0
_SYMMETRY_RELATIVE_TOLERANCE: Final[float] = 1.0e-6
_SYMMETRY_LATERAL_ACCEL_ABSOLUTE_M_S2: Final[float] = 1.0e-3
_FINITE_TOLERANCE: Final[float] = 1.0e12


@dataclass(frozen=True, slots=True)
class Violation:
    """One thing wrong, located precisely enough to debug without a debugger."""

    where: str
    detail: str
    value: float
    limit: float

    @override
    def __str__(self) -> str:
        return f"{self.where}: {self.detail} (value {self.value!r}, limit {self.limit!r})"


@dataclass(frozen=True, slots=True)
class InvariantResult:
    """Outcome of one invariant over one record."""

    number: int
    name: str
    backing: str
    violations: tuple[Violation, ...]

    @property
    def passed(self) -> bool:
        return not self.violations

    def __bool__(self) -> bool:
        return self.passed

    def summary(self) -> str:
        head = f"invariant {self.number} ({self.name})"
        if self.passed:
            return f"{head}: pass - physics-backed from {self.backing}"
        lines = [f"{head}: {len(self.violations)} violation(s)"]
        lines.extend(f"  {violation}" for violation in self.violations)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Invariant:
    number: int
    name: str
    backing: str
    check: Callable[[SampleRecord, CarSpec], tuple[Violation, ...]]


def _friction_limit(mu: float, fz_n: float) -> float:
    return mu * fz_n


def check_finite(record: SampleRecord, _spec: CarSpec) -> tuple[Violation, ...]:
    """Invariant 1: no NaN or Inf, ever - in the truth stream or the published stream."""
    found: list[Violation] = []
    for index, step in enumerate(record.ground_truth):
        scalars = {
            "vx": step.vx_m_s,
            "vy": step.vy_m_s,
            "ax": step.ax_m_s2,
            "ay": step.ay_m_s2,
            "az": step.az_m_s2,
            "clutch": step.clutch,
            "throttle_pct": step.throttle_pct,
            "ice_power_w": step.ice_power_w,
            "mgu_k_power_w": step.mgu_k_power_w,
            "drag_w": step.drag_w,
            "downforce_n": step.downforce_n,
            "steer_rad": step.steer_rad,
        }
        if step.energy_residual_fraction is not None:
            scalars["energy_residual_fraction"] = step.energy_residual_fraction
        for label, value in scalars.items():
            if not math.isfinite(value) or abs(value) > _FINITE_TOLERANCE:
                found.append(
                    Violation(
                        where=f"step {index} ground_truth.{label}",
                        detail="not a finite value",
                        value=float(value),
                        limit=_FINITE_TOLERANCE,
                    )
                )
        for corner, wheel in zip(CORNERS, step.wheels, strict=True):
            for label, value in (
                ("fz_n", wheel.fz_n),
                ("fx_n", wheel.fx_n),
                ("fy_n", wheel.fy_n),
                ("mu", wheel.mu),
                ("mu_lateral", wheel.mu_lateral)
                if wheel.mu_lateral is not None
                else ("mu_lateral", wheel.mu),
                ("kappa", wheel.kappa),
                ("alpha_rad", wheel.alpha_rad),
                ("effective_alpha_rad", wheel.effective_alpha_rad)
                if wheel.effective_alpha_rad is not None
                else ("effective_alpha_rad", wheel.alpha_rad),
                ("camber_deg", wheel.camber_deg),
            ):
                if not math.isfinite(value) or abs(value) > _FINITE_TOLERANCE:
                    found.append(
                        Violation(
                            where=f"step {index} wheel {corner}.{label}",
                            detail="not a finite value",
                            value=float(value),
                            limit=_FINITE_TOLERANCE,
                        )
                    )
    for index, frame in enumerate(record.frames):
        for channel, value in frame.values.items():
            if not math.isfinite(value) or abs(value) > _FINITE_TOLERANCE:
                found.append(
                    Violation(
                        where=f"frame {index} channel {channel}",
                        detail="not a finite value",
                        value=float(value),
                        limit=_FINITE_TOLERANCE,
                    )
                )
    return tuple(found)


def check_friction_ellipse(record: SampleRecord, _spec: CarSpec) -> tuple[Violation, ...]:
    """Invariant 2: each force is normalized by its own load-dependent peak."""
    found: list[Violation] = []
    for index, step in enumerate(record.ground_truth):
        for corner, wheel in zip(CORNERS, step.wheels, strict=True):
            if wheel.fz_n < 0.0:
                found.append(
                    Violation(
                        where=f"step {index} wheel {corner}",
                        detail="negative vertical load is invalid",
                        value=wheel.fz_n,
                        limit=0.0,
                    )
                )
                continue
            if wheel.fz_n == 0.0:
                if wheel.fx_n != 0.0 or wheel.fy_n != 0.0:
                    found.append(
                        Violation(
                            where=f"step {index} wheel {corner}",
                            detail="unloaded wheel carries tire force",
                            value=math.hypot(wheel.fx_n, wheel.fy_n),
                            limit=0.0,
                        )
                    )
                continue
            mu_lateral = wheel.mu if wheel.mu_lateral is None else wheel.mu_lateral
            if wheel.mu <= 0.0 or mu_lateral <= 0.0:
                found.append(
                    Violation(
                        where=f"step {index} wheel {corner}",
                        detail="non-positive friction coefficient on a loaded axis",
                        value=min(wheel.mu, mu_lateral),
                        limit=0.0,
                    )
                )
                continue
            limit_x = _friction_limit(wheel.mu, wheel.fz_n)
            limit_y = _friction_limit(mu_lateral, wheel.fz_n)
            utilisation = (wheel.fx_n / limit_x) ** 2 + (wheel.fy_n / limit_y) ** 2
            if utilisation > _FRICTION_TOLERANCE:
                found.append(
                    Violation(
                        where=f"step {index} wheel {corner}",
                        detail="combined slip is outside the friction ellipse",
                        value=utilisation,
                        limit=_FRICTION_TOLERANCE,
                    )
                )
    return tuple(found)


def check_vertical_load_sum(record: SampleRecord, spec: CarSpec) -> tuple[Violation, ...]:
    """Invariant 3: sum of corner loads = mass * (g + az) + aerodynamic downforce."""
    found: list[Violation] = []
    for index, step in enumerate(record.ground_truth):
        total = math.fsum(wheel.fz_n for wheel in step.wheels)
        expected = spec.mass_kg * (spec.gravity_m_s2 + step.az_m_s2) + step.downforce_n
        tolerance = max(_LOAD_ABSOLUTE_FLOOR_N, abs(expected) * _LOAD_RELATIVE_TOLERANCE)
        if abs(total - expected) > tolerance:
            found.append(
                Violation(
                    where=f"step {index}",
                    detail="corner vertical load does not sum to weight + downforce",
                    value=total,
                    limit=expected + tolerance,
                )
            )
    return tuple(found)


def _sign(value: float, tolerance: float = 0.0) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def check_sign_conventions(record: SampleRecord, _spec: CarSpec) -> tuple[Violation, ...]:
    """Invariant 4: slip and force sign conventions hold identically at all four corners.

    Two conventions, fixed in :class:`~f1telemetry.testing.records.GroundTruthStep`:
    ``sign(kappa) == sign(fx)`` and ``sign(alpha) == sign(fy)``. A corner that breaks
    either one is a sign bug that no range check and no residual model would ever see.
    """
    issues: list[Violation] = []
    for index, step in enumerate(record.ground_truth):
        for corner, wheel in zip(CORNERS, step.wheels, strict=True):
            if _sign(wheel.kappa) != _sign(wheel.fx_n):
                issues.append(
                    Violation(
                        where=f"step {index} wheel {corner}",
                        detail="slip ratio and longitudinal force disagree in sign",
                        value=wheel.fx_n,
                        limit=wheel.kappa,
                    )
                )
            effective_alpha_rad = (
                wheel.alpha_rad if wheel.effective_alpha_rad is None else wheel.effective_alpha_rad
            )
            if _sign(effective_alpha_rad) != _sign(wheel.fy_n):
                issues.append(
                    Violation(
                        where=f"step {index} wheel {corner}",
                        detail="effective slip and lateral force disagree in sign",
                        value=wheel.fy_n,
                        limit=effective_alpha_rad,
                    )
                )
    return tuple(issues)


def check_symmetry(record: SampleRecord, _spec: CarSpec) -> tuple[Violation, ...]:
    """Invariant 5: zero steer preserves mirrored corner outputs on a symmetric car.

    Static camber can create lateral force at zero slip. The physical symmetry check
    therefore compares mirrored values rather than requiring every lateral quantity to
    be zero.
    """
    issues: list[Violation] = []
    for index, step in enumerate(record.ground_truth):
        if abs(step.steer_rad) > _SYMMETRY_TOLERANCE:
            continue
        if abs(step.vy_m_s) > 1.0e-4:
            issues.append(
                Violation(
                    where=f"step {index}",
                    detail="straight-line record has lateral velocity",
                    value=abs(step.vy_m_s),
                    limit=1.0e-4,
                )
            )
        if abs(step.ay_m_s2) > _SYMMETRY_LATERAL_ACCEL_ABSOLUTE_M_S2:
            issues.append(
                Violation(
                    where=f"step {index}",
                    detail="straight-line record has lateral acceleration",
                    value=abs(step.ay_m_s2),
                    limit=_SYMMETRY_LATERAL_ACCEL_ABSOLUTE_M_S2,
                )
            )
        fl, fr, rl, rr = step.wheels
        for left, right, axle in ((fl, fr, "front"), (rl, rr, "rear")):
            if abs(left.fz_n - right.fz_n) > _LOAD_ABSOLUTE_FLOOR_N:
                issues.append(
                    Violation(
                        where=f"step {index} {axle} axle",
                        detail="left and right vertical load differ with no lateral input",
                        value=left.fz_n,
                        limit=right.fz_n,
                    )
                )
            for left_value, right_value, sign, quantity, quantity_tolerance in (
                (left.kappa, right.kappa, 1.0, "slip ratio", 1.0e-5),
                (
                    left.fx_n,
                    right.fx_n,
                    1.0,
                    "longitudinal force",
                    _SYMMETRY_LONGITUDINAL_FORCE_ABSOLUTE_N,
                ),
                (left.alpha_rad, right.alpha_rad, 1.0, "slip angle", 1.0e-4),
                (left.camber_deg, right.camber_deg, -1.0, "camber", 1.0e-3),
                (
                    left.fy_n,
                    right.fy_n,
                    -1.0,
                    "lateral force",
                    _SYMMETRY_LATERAL_FORCE_ABSOLUTE_N,
                ),
            ):
                tolerance = max(
                    quantity_tolerance,
                    max(abs(left_value), abs(right_value)) * _SYMMETRY_RELATIVE_TOLERANCE,
                )
                if abs(left_value - sign * right_value) > tolerance:
                    issues.append(
                        Violation(
                            where=f"step {index} {axle} axle",
                            detail=f"mirrored {quantity} differs with no lateral input",
                            value=left_value,
                            limit=sign * right_value,
                        )
                    )
    return tuple(issues)


def check_energy_balance(record: SampleRecord, spec: CarSpec) -> tuple[Violation, ...]:
    """Invariant 6: planar, yaw, and wheel kinetic-energy residual stays under 1%."""
    issues: list[Violation] = []
    for index, step in enumerate(record.ground_truth):
        if step.energy_residual_fraction is not None:
            relative = step.energy_residual_fraction
        else:
            kinetic_rate = spec.mass_kg * (step.vx_m_s * step.ax_m_s2 + step.vy_m_s * step.ay_m_s2)
            power_in = step.ice_power_w + step.mgu_k_power_w
            # ``drag_w`` is signed negative in the forward direction, so it is added as
            # an external power term. Legacy records without a complete P2 energy residual
            # have no wheel or yaw state; compare only the translational chassis balance.
            residual = power_in + step.drag_w - kinetic_rate
            relative = abs(residual) / max(abs(kinetic_rate), 1.0)
        if relative > _ENERGY_RESIDUAL_LIMIT:
            issues.append(
                Violation(
                    where=f"step {index}",
                    detail="energy balance residual over 1%",
                    value=relative,
                    limit=_ENERGY_RESIDUAL_LIMIT,
                )
            )
    return tuple(issues)


def check_gearbox_progression(record: SampleRecord, spec: CarSpec) -> tuple[Violation, ...]:
    """Invariant 7: a gear change is one neighbouring gear, and no reverse under positive throttle.

    **A downshift is legal.** :func:`~f1telemetry.physics.gearbox.step_requested_gear` answers a
    ``DOWN`` paddle by stepping one forward gear down, so 6 -> 5 is a driver request the model
    represents and C9.8.3's one change at a time still holds around it. An earlier version of this
    checker rejected *every* decrease, which made any record containing an ordinary downshift a
    failure - including the P1 acceleration record the moment a braking scenario was run through it.

    **What is still illegal is a transition the gearbox cannot perform.** One request moves at most
    one gear and only ever to a neighbour, so a change of two or more gears between two recorded
    steps is an indexing or state bug, in either direction. That is the check, not monotonicity:
    requiring a non-decreasing gear column would forbid the downshift the model is built to answer.

    Neutral and reverse are absolute selections that apply from anywhere - ``NEUTRAL`` and
    ``REVERSE`` are states rather than steps along the box, and ``UP`` from either re-enters at
    first - so a record is not required to walk the whole ladder to reach one. Neutral therefore
    resets the neighbour comparison, and no distance is checked across a selection. Reverse stays
    forbidden under positive throttle regardless of the preceding gear, which is the other half of
    the original contract and is unchanged.

    The neighbour rule is read across adjacent *recorded* steps, so it assumes a record is sampled
    finely enough that two shifts cannot complete inside one interval. The committed harness records
    the drivetrain every 100 kernel steps (10 ms) against a 40 ms ``shift_time_s``, which leaves
    that margin; a coarser record would need the rule weakened to "no skipped gear per interval at
    the recorded rate".
    """
    issues: list[Violation] = []
    top = len(spec.gear_ratios)
    previous = record.ground_truth[0].gear if record.ground_truth else 0
    for index, step in enumerate(record.ground_truth):
        gear = step.gear
        if not -1 <= gear <= top:
            issues.append(
                Violation(
                    where=f"step {index}",
                    detail="gear outside -1..n_gears",
                    value=float(gear),
                    limit=float(top),
                )
            )
        if gear == 0:
            previous = 0
            continue
        if gear < 0 and step.throttle_pct > 0.0:
            issues.append(
                Violation(
                    where=f"step {index}",
                    detail="reverse engaged under positive throttle",
                    value=step.throttle_pct,
                    limit=0.0,
                )
            )
        if index > 0 and previous >= 1 and gear >= 1 and abs(gear - previous) > 1:
            # Both ends are forward gears, so this is a skipped gear rather than a neutral or
            # reverse selection, which are legal from anywhere and are not distance-checked.
            descending = gear < previous
            issues.append(
                Violation(
                    where=f"step {index}",
                    detail=(
                        "gearbox went backwards, skipping gears"
                        if descending
                        else "gearbox skipped gears going up"
                    ),
                    value=float(gear),
                    limit=float(previous),
                )
            )
        previous = gear
    return tuple(issues)


def check_determinism(record: SampleRecord, spec: CarSpec) -> tuple[Violation, ...]:
    """Invariant 8: two serialisations of the same record are byte-identical.

    Physics-backed from P5, where the input is a simulation run rather than a fixture.
    What is already testable now, and is the part that actually breaks, is the storage
    path: a wall-clock value written into Parquet key-value metadata, a dictionary
    iterated in insertion order that differs between processes, or a compression setting
    that embeds a timestamp all produce a difference here.
    """
    from f1telemetry.testing.parquet_io import serialise_frames

    first = serialise_frames(record)
    second = serialise_frames(record)
    if first == second:
        return ()
    differing = [i for i, (a, b) in enumerate(zip(first, second, strict=True)) if a != b]
    issues = [
        Violation(
            where=f"byte {index}",
            detail="serialised bytes differ between two runs of the same record",
            value=float(index),
            limit=0.0,
        )
        for index in differing[:8]
    ]
    if len(differing) > 8:
        issues.append(
            Violation(
                where="bytes",
                detail=f"{len(differing)} differing bytes in total",
                value=float(len(differing)),
                limit=8.0,
            )
        )
    return tuple(issues)


INVARIANTS: Final[tuple[Invariant, ...]] = (
    Invariant(1, "no_nan_or_inf", "P1", check_finite),
    Invariant(2, "friction_ellipse", "P2", check_friction_ellipse),
    Invariant(3, "vertical_load_sum", "P2", check_vertical_load_sum),
    Invariant(4, "sign_conventions", "P2", check_sign_conventions),
    Invariant(5, "left_right_symmetry", "P2", check_symmetry),
    Invariant(6, "energy_balance", "P1", check_energy_balance),
    Invariant(7, "gearbox_progression", "P1", check_gearbox_progression),
    Invariant(8, "determinism", "P5", check_determinism),
)

_BY_NUMBER: Final[dict[int, Invariant]] = {inv.number: inv for inv in INVARIANTS}


def _one(record: SampleRecord, spec: CarSpec, number: int) -> InvariantResult:
    invariant = _BY_NUMBER[number]
    return InvariantResult(
        number=invariant.number,
        name=invariant.name,
        backing=invariant.backing,
        violations=invariant.check(record, spec),
    )


def run_all(
    record: SampleRecord, spec: CarSpec, numbers: Sequence[int] | None = None
) -> tuple[InvariantResult, ...]:
    """Run every invariant, or only the requested numbers, over one record."""
    selected = tuple(numbers) if numbers is not None else tuple(_BY_NUMBER)
    return tuple(_one(record, spec, number) for number in selected)
