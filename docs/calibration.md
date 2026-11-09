# Calibration and model limits

The vehicle is an **uncalibrated synthetic Formula car**. Passing a software test
does not establish agreement with a real car. The project demonstrates telemetry
engineering, numerical invariants and repeatable experiments.

## 1. Coefficient ownership

[`car_spec.yaml`](../car_spec.yaml) owns the vehicle coefficients. Each numeric
field is accounted for by a `regulation` claim or a `not_regulated` explanation,
including nested powertrain records. The loader validates the contract and
`provenance_audit()` checks that the claims are complete. The YAML is the detailed
coefficient catalogue; this document does not duplicate its numbers.

```powershell
uv run python -c "from f1telemetry.contracts.car_spec import load_car_spec, provenance_audit; findings = provenance_audit(load_car_spec().raw); print(findings); assert not findings"
uv run pytest tests/test_car_spec.py
```

The audit returned `[]` on 2026-11-03. This verifies annotation completeness,
not the truth of every external source or the model's calibration.

## 2. Regulation claims

The specification records FIA 2026 Section C, Issue 20, dated 2026-08-05,
with its source URL, article, page and quoted/basis text next to each claim.
Examples include mass floors, permitted powertrain behavior, wheel geometry
and gearbox timing bounds. A limit is not a measurement of this simulated car.
Nominal tyre mass and session-specific conditions prevent some recorded limits
from becoming complete compliance checks. No regulatory-compliance claim is made.

## 3. Synthetic physical parameters

| Group | What is assumed | What would establish calibration |
|---|---|---|
| Mass and chassis | Running mass, static split, track widths, CG and inertias | Configuration-matched measurements |
| Aero | Reference area, speed-dependent lift/drag tables and ride-height sensitivity | A cited coefficient set or measured force/speed data |
| Powertrain | Torque curve, transient response, efficiency and deployment assumptions | Matched torque/power and acceleration traces |
| Gearbox | Ratios, final drive, reverse ratio, shift schedule and duration | Actual gearing and shift traces |
| Tyres | Pacejka coefficients, load sensitivity, relaxation lengths, camber stiffness and inertia | Measured force/slip/load data |
| Suspension and steering | Ride rates, stiffness splits, kinematic gradients and steering geometry | Geometry/setup measurements and settled-response checks |
| Thermal and pressure | Lumped capacities, heat splits, cooling and initial gas conditions | Temperature/pressure traces under matched conditions |

Physical constants and unit definitions are identified separately in the YAML.
Some comments in historical plans predate later model slices; the implemented
contract and tests determine current behavior. No public F1 tyre/aero parameter
set has been imported merely because an earlier plan mentioned it.

## 4. Numerical checks

The invariant suite checks such properties as finite output, wheel-force bounds,
load balance, symmetry, gearbox state, energy accounting and deterministic
serialization. The modeled energy boundary covers chassis/wheel work, not a
complete fuel-to-road powertrain. Golden traces are regression evidence;
advisory golden comparisons are not independent physical validation.

```powershell
uv run pytest -m invariant
uv run pytest -m golden
```

## 5. Deliberate model limits

Suspension is quasi-static, thermal nodes are lumped, tyre gas volume is fixed,
and scenario brake torque is caller-supplied rather than constrained by a
validated brake-capacity model. Racing-line and driver checks do not establish
a calibrated lap-time predictor. Track geometry is approximate; see
[source attribution](../tracks/source_data/README.md).

The dashboard computes a bounded headless scenario first, then streams its
recorded frames at wall-clock pace. It is not an interactive driving simulator
and its simulation step frequency is not its WebSocket delivery rate.

## 6. Performance references and open gates

[`PHASES.md`](roadmap/PHASES.md) preserves historical reports, including the
longitudinal acceleration miss and open source-matched lateral/thermal targets.
Those reported measurements have not been re-established as calibration by
this release. Earlier phase-demo documents are absent from this checkout;
their contents cannot be reconstructed as verified evidence.

Before claiming a calibrated car, fix a configuration-matched dataset and
acceptance bounds, record its uncertainty, fit on one portion, and evaluate on
separate observations. The backend portfolio release does not close those gates.

## 7. Detection results

[The generated detection report](detection.md) describes a small synthetic
evaluation over two tyre-temperature channels. It includes failures and a
negative control. Its short clean exposure does not establish a dependable
real-world false-alarm rate, and the learned baseline's misses must remain visible.
It supports testing the diagnostic pipeline, not production fault-detection claims.
