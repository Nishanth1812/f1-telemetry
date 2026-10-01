# Phase 1 fidelity implementation plan

- **Goal:** Implement the approved generic 2026 F1 Phase 1 longitudinal fidelity spec on `feat/phase-1`.
- **Architecture:** Keep the existing compiled longitudinal kernel and YAML car/channel contracts. Add the minimum caller-owned controls and drivetrain state needed for driver-requested gear changes, clutch demand, MGU-K limits, and four-wheel longitudinal force assembly. Scenarios remain deterministic and caller-driven. Lateral dynamics, team-specific setup, and full compliance certification remain out of scope.
- **Tech stack:** Existing Python, Numba, PyYAML, and pytest dependencies; no new dependencies.
- **Spec:** `docs/superpowers/specs/2026-10-01-f1-phase-1-fidelity-design.md`
- **Global constraints:** Keep synthetic assumptions labeled and configurable; preserve kernel validation and deterministic behavior; enforce FIA values only where the cited regulation supports them; keep each task independently reviewable and commit it after its verification passes.

## Review focus

1. Gear changes must never occur from RPM alone; test explicit adjacent forward requests and neutral/reverse behavior in `tests/test_gearbox.py`.
2. Clutch command must be caller-owned and map engagement travel to the regulated rear-axle demand without the current 3000 Nm clamp; test endpoints and launch tracking in `tests/test_gearbox.py` and `tests/test_powertrain.py`.
3. MGU-K speed, DC power, torque, start-speed, SOC, and event recharge limits must be applied at their correct points; test caps and boundary cases in `tests/test_powertrain.py` and `tests/test_car_spec.py`.
4. Wheel torque and force assembly must drive only the rear axle and must not transfer torque from a slower wheel to a faster one; test sign, split, and zero/negative speed handling in `tests/test_forces.py`.
5. Updated settings and telemetry metadata must not claim synthetic choices are FIA limits; test parsed provenance and generated channel contracts in `tests/test_car_spec.py` and `tests/test_contract.py`.

## Implementation tasks

### 1. Make the car specification express supported 2026 limits

Files: `car_spec.yaml`, `src/f1telemetry/contracts/car_spec.py`, `channels.yaml`, generated contract outputs, `tests/test_car_spec.py`, `tests/test_contract.py`.

Add only the event/config fields required by the approved spec, correct provenance and descriptions for synthetic reverse/clutch/MGU-K assumptions, and regenerate checked-in outputs with the existing code generator. Start with focused failing contract tests, run them to confirm failure, implement the schema/data changes, regenerate, rerun the focused tests, review the diff, then commit as `Record 2026 powertrain limits and provenance`.

### 2. Replace automatic gearbox and synthetic clutch control

Files: `src/f1telemetry/physics/gearbox.py`, `src/f1telemetry/kernels/longitudinal.py`, `tests/test_gearbox.py`, `tests/test_longitudinal_kernel.py`, `tests/test_powertrain.py`.

Make requested gear and normalized caller-owned clutch engagement the inputs to a compiled drivetrain step. Preserve caller-owned gear, clutch, and shift state; remove RPM-triggered shifting and the 3000 Nm output clamp; represent neutral and reverse explicitly while keeping reverse ratio labeled synthetic. Add focused tests first, confirm they fail, implement the smallest compatible API change, rerun focused tests and affected kernel tests, then commit as `Use driver commands for gear and clutch control`.

### 3. Apply ICE, MGU-K, fuel-flow, and energy-store limits

Files: `src/f1telemetry/physics/powertrain.py`, `src/f1telemetry/kernels/longitudinal.py`, `src/f1telemetry/contracts/car_spec.py`, `tests/test_powertrain.py`, `tests/test_car_spec.py`, `tests/test_longitudinal_kernel.py`.

Apply the 350 kW DC and speed-dependent MGU-K power caps, crankshaft-referenced torque limit, 60,000 rpm relative-speed cap, standing-start deployment rule and stated exception, ICE fuel-energy-flow limits, 4 MJ usable SOC range, and configurable event recharge allowance. Keep the MGU-K upstream of the gearbox/clutch through a documented synthetic fixed ratio. Write cap and transition tests first, observe failure, implement and rerun focused tests plus kernel integration tests, then commit as `Enforce 2026 hybrid powertrain limits`.

### 4. Assemble longitudinal force from four wheel states

Files: `src/f1telemetry/physics/forces.py`, `src/f1telemetry/kernels/longitudinal.py`, `src/f1telemetry/contracts/car_spec.py`, `tests/test_forces.py`, `tests/test_longitudinal_kernel.py`.

Add caller-owned wheel angular speeds and rear-only drive torque assembly. Use the existing tyre model and static axle load in Phase 1; split rear torque equally as an explicit synthetic assumption, with no differential transfer, traction control, or ABS. Test conservation, direction, free-rolling fronts, and invalid states first; confirm failure; implement and rerun force and kernel tests; commit as `Connect drivetrain torque to four wheel states`.

### 5. Add deterministic Phase 1 scenarios and calibration checks

Files: existing scenario/calibration modules under `src/f1telemetry/testing/`, `docs/calibration.md`, and focused tests under `tests/`.

Use existing scenario patterns to exercise standing launch, acceleration and requested shifts, neutral/coast, braking, and MGU-K deployment/recharge transitions. Record public target source, assumptions, and tolerance for every calibration check. Add failing scenario checks first, implement only the missing scenario/calibration path, rerun those checks, then commit as `Add Phase 1 drivetrain calibration scenarios`.

### 6. Close Phase 1 invariants and exit criteria

Files: `src/f1telemetry/testing/invariants.py`, relevant `tests/test_invariants.py` and kernel/physics tests, `PHASES.md`.

Add only invariants needed for the new gear, clutch, power, energy, and wheel-force states. Verify the scenario suite and Phase 1 exit criteria against the approved spec, mark only supported criteria complete, and leave Phase 2 work open. Add invariant tests first, confirm failure, implement and rerun the smallest affected suite, review all milestone diffs and repository status, then commit as `Complete Phase 1 longitudinal fidelity checks`.

## Execution and commits

For each task: add a focused failing test, run it to see the expected failure, implement the minimum change, rerun focused verification, inspect the diff and status, then commit using the exact human-written subject listed above. Run the full existing verification set after the final task. Do not begin implementation until this plan is approved.
