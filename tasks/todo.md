# Phase 1 task list

## Task 1: Confirm P1 inputs and kernel configuration

**Status:** Done (fix round 1 applied). Commits `a063b1e`, `b9fc915`. Report: `.superpowers/sdd/phase-1/reports/task-1.md`.

**Description:** Verify P1 car inputs against the current FIA regulations and published sources, record provenance, and extend the existing car-spec parsing path to provide plain numeric arrays to the kernel.

**Acceptance criteria:**
- [x] Each regulated P1 value records the FIA issue and clause/page; synthesized values are labeled as such.
- [x] Editing a P1 value in `car_spec.yaml` changes simulation configuration without a code edit.
- [x] Invalid or missing P1 configuration fails at the Python boundary before entering Numba.

**Verification:** Focused car-spec tests; `uv run --frozen f1-check-contract`.

**Dependencies:** None.

**Files likely touched:** `car_spec.yaml`, `src/f1telemetry/contracts/car_spec.py`, tests for car-spec parsing, `docs/calibration.md`.

**Estimated scope:** Medium.

## Task 2: Add deterministic fixed-step straight-line kernel

**Description:** Add the P1 longitudinal state, caller-owned output buffers, and a flat 10 kHz semi-implicit integrator using the P0 Numba conventions.

**Acceptance criteria:**
- [ ] Kernel uses preallocated numeric arrays, fixed `dt = 100 µs`, `cache=True`, and `fastmath=False`.
- [ ] Two runs with the same seed, configuration, and inputs produce byte-identical traces.
- [ ] A representative run stays finite and the loop performs no Python-side allocation or wall-clock reads.

**Verification:** Focused kernel tests, including a byte comparison between repeated outputs; `uv run --frozen pytest <focused-test-file>`.

**Dependencies:** Task 1.

**Files likely touched:** `src/f1telemetry/kernels/`, new `src/f1telemetry/physics/` longitudinal module if needed, focused kernel tests.

**Estimated scope:** Medium.

## Checkpoint: Kernel

- [ ] Focused tests pass for determinism, fixed-step progression, and finite outputs.
- [ ] Existing `just check` gates remain green.

## Task 3: Add aero and longitudinal tire forces

**Description:** Calculate speed-dependent drag/downforce and longitudinal Pacejka force from wheel slip and vertical load.

**Acceptance criteria:**
- [ ] `Cl(v)` and `Cd(v)` are configuration-driven functions and aero forces use the documented dynamic-pressure relationship.
- [ ] Slip ratio guards its low-speed denominator; force sign and tire load behavior are covered by tests.
- [ ] Force calculations remain finite and respect the configured longitudinal grip limit.

**Verification:** Focused force tests for low speed, zero slip, increasing speed, and load changes.

**Dependencies:** Tasks 1–2.

**Files likely touched:** `car_spec.yaml`, `src/f1telemetry/physics/`, focused physics tests.

**Estimated scope:** Medium.

## Task 4: Add powertrain, clutch, gearbox, and wheel rotation

**Description:** Connect the synthesized ICE torque curve and MGU-K drive to the wheels through the specified drivetrain, including eight-speed shifts and clutch state.

**Acceptance criteria:**
- [ ] Gear ratios and limits come from `car_spec.yaml`; gearbox progression and positive-throttle reverse behavior satisfy invariant 7.
- [ ] Launch, trailing throttle, boost-cut/upshift, and low-speed cases exercise clutch and wheel-speed state.
- [ ] ICE curve provenance and synthesis assumptions are documented; motor power is accounted for separately from ICE power.

**Verification:** Focused drivetrain tests for launch, shift boundaries, clutch transition, and wheel force direction.

**Dependencies:** Tasks 1–3.

**Files likely touched:** `car_spec.yaml`, `src/f1telemetry/physics/`, drivetrain tests, `docs/calibration.md`.

**Estimated scope:** Medium.

## Checkpoint: Drive path

- [ ] Launch and shift tests pass with finite values and expected force/gear direction.
- [ ] Review any remaining P1/P2 boundary decisions before adding scenarios.

## Task 5: Add P1 scenarios and calibrate performance

**Description:** Implement `accelerate_to_speed` and `full_throttle` with deterministic traces; tune only documented car-spec coefficients to source-backed acceleration and top-speed targets.

**Acceptance criteria:**
- [ ] Each scenario runs from fixed initial conditions and emits traces through the existing testing/record pattern.
- [ ] 0–100 km/h and top-speed targets and tolerances are chosen from cited sources before tuning and recorded in `docs/calibration.md`.
- [ ] Calibration changes are data edits where possible; the torque curve remains labeled as synthesized.

**Verification:** Scenario tests plus golden traces for straight-line acceleration; record measured 0–100 km/h and top speed against chosen targets.

**Dependencies:** Tasks 1–4.

**Files likely touched:** `src/f1telemetry/`, `car_spec.yaml`, `tests/`, `tests/golden/`, `docs/calibration.md`.

**Estimated scope:** Medium.

## Task 6: Enforce P1 invariants and record independent comparison

**Description:** Add the longitudinal energy-balance gate and run P1 scenarios through the relevant existing invariants; compare against `fastest-lap` when it can be built.

**Acceptance criteria:**
- [ ] Energy residual stays below 1% on representative P1 runs and is enforced in CI.
- [ ] Invariants 1, 3, 6, and 7 pass on real scenario runs; regression output is deterministic.
- [ ] `fastest-lap` comparison is recorded within a few percent or its discrepancy/build limitation is documented without blocking P1.

**Verification:** Focused invariant and golden tests, then `just check`; save comparison method, result, and limitation in `docs/calibration.md`.

**Dependencies:** Tasks 1–5.

**Files likely touched:** `src/f1telemetry/testing/`, `tests/`, `.github/workflows/ci.yml` if needed, `docs/calibration.md`.

**Estimated scope:** Medium.

## Checkpoint: Phase 1 exit

- [ ] All `PHASES.md` P1 exit-gate items pass or have a written evidence-based explanation.
- [ ] `just check` passes.
- [ ] Calibration inputs, target sources, and optional solver comparison are recorded for review.


