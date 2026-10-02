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

_FRICTION_TOLERANCE: Final[float] = 1.0
_LOAD_RELATIVE_TOLERANCE: Final[float] = 1.0e-3
_LOAD_ABSOLUTE_FLOOR_N: Final[float] = 1.0
_ENERGY_RESIDUAL_LIMIT: Final[float] = 0.01
_SYMMETRY_TOLERANCE: Final[float] = 1.0e-9
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
                ("kappa", wheel.kappa),
                ("alpha_rad", wheel.alpha_rad),
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
    """Invariant 2: (Fx/muFz)^2 + (Fy/muFz)^2 <= 1 for every wheel, every step."""
    found: list[Violation] = []
    for index, step in enumerate(record.ground_truth):
        for corner, wheel in zip(CORNERS, step.wheels, strict=True):
            if wheel.fz_n <= 0.0:
                found.append(
                    Violation(
                        where=f"step {index} wheel {corner}",
                        detail="non-positive vertical load leaves the friction limit undefined",
                        value=wheel.fz_n,
                        limit=0.0,
                    )
                )
                continue
            if wheel.mu <= 0.0:
                found.append(
                    Violation(
                        where=f"step {index} wheel {corner}",
                        detail="non-positive friction coefficient",
                        value=wheel.mu,
                        limit=0.0,
                    )
                )
                continue
            limit = _friction_limit(wheel.mu, wheel.fz_n)
            utilisation = (wheel.fx_n / limit) ** 2 + (wheel.fy_n / limit) ** 2
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
            if _sign(wheel.alpha_rad) != _sign(wheel.fy_n):
                issues.append(
                    Violation(
                        where=f"step {index} wheel {corner}",
                        detail="slip angle and lateral force disagree in sign",
                        value=wheel.fy_n,
                        limit=wheel.alpha_rad,
                    )
                )
    return tuple(issues)


def check_symmetry(record: SampleRecord, _spec: CarSpec) -> tuple[Violation, ...]:
    """Invariant 5: left/right symmetry at zero steer, zero camber and a symmetric setup.

    Applied to a straight-line record, where the correct answer is exact: no lateral
    velocity, no lateral acceleration, no slip angle, no lateral force, and matching
    loads within each axle. A left/right asymmetry here is a load-transfer or indexing
    bug and is the cheapest bug-finder in the project.
    """
    issues: list[Violation] = []
    for index, step in enumerate(record.ground_truth):
        if abs(step.steer_rad) > _SYMMETRY_TOLERANCE:
            issues.append(
                Violation(
                    where=f"step {index}",
                    detail="symmetry check needs zero steering input",
                    value=step.steer_rad,
                    limit=_SYMMETRY_TOLERANCE,
                )
            )
        if abs(step.vy_m_s) > _SYMMETRY_TOLERANCE or abs(step.ay_m_s2) > _SYMMETRY_TOLERANCE:
            issues.append(
                Violation(
                    where=f"step {index}",
                    detail="straight-line record has lateral motion",
                    value=abs(step.vy_m_s),
                    limit=_SYMMETRY_TOLERANCE,
                )
            )
        fl, fr, rl, rr = step.wheels
        for corner, wheel in zip(CORNERS, step.wheels, strict=True):
            if abs(wheel.alpha_rad) > _SYMMETRY_TOLERANCE:
                issues.append(
                    Violation(
                        where=f"step {index} wheel {corner}",
                        detail="symmetric setup must produce no slip angle",
                        value=wheel.alpha_rad,
                        limit=_SYMMETRY_TOLERANCE,
                    )
                )
            if abs(wheel.camber_deg) > _SYMMETRY_TOLERANCE:
                issues.append(
                    Violation(
                        where=f"step {index} wheel {corner}",
                        detail="symmetric setup must have zero camber",
                        value=wheel.camber_deg,
                        limit=_SYMMETRY_TOLERANCE,
                    )
                )
            if abs(wheel.fy_n) > _SYMMETRY_TOLERANCE:
                issues.append(
                    Violation(
                        where=f"step {index} wheel {corner}",
                        detail="symmetric straight-line run must produce no lateral force",
                        value=wheel.fy_n,
                        limit=_SYMMETRY_TOLERANCE,
                    )
                )
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
            if abs(left.kappa - right.kappa) > _SYMMETRY_TOLERANCE:
                issues.append(
                    Violation(
                        where=f"step {index} {axle} axle",
                        detail="left and right slip ratio differ with no lateral input",
                        value=left.kappa,
                        limit=right.kappa,
                    )
                )
    return tuple(issues)


def check_energy_balance(record: SampleRecord, spec: CarSpec) -> tuple[Violation, ...]:
    """Invariant 6: modeled longitudinal kinetic-energy residual stays under 1%."""
    issues: list[Violation] = []
    for index, step in enumerate(record.ground_truth):
        if step.energy_residual_fraction is not None:
            relative = step.energy_residual_fraction
        else:
            kinetic_rate = spec.mass_kg * (step.vx_m_s * step.ax_m_s2 + step.vy_m_s * step.ay_m_s2)
            power_in = step.ice_power_w + step.mgu_k_power_w
            residual = power_in - step.drag_w - kinetic_rate
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
    """Invariant 7: no uncommanded downshift across forward gears or reverse under throttle.

    Neutral is a legal driver-selected state between forward gears. The record does not retain
    the driver's request, so neutral resets the forward progression check; reverse remains
    forbidden under positive throttle regardless of the preceding gear.
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
        if index > 0 and gear < previous:
            issues.append(
                Violation(
                    where=f"step {index}",
                    detail="gearbox went backwards",
                    value=float(gear),
                    limit=float(previous),
                )
            )
        if gear < 0 and step.throttle_pct > 0.0:
            issues.append(
                Violation(
                    where=f"step {index}",
                    detail="reverse engaged under positive throttle",
                    value=step.throttle_pct,
                    limit=0.0,
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
