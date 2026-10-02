# Phase 1 task list

## Task 1: Confirm P1 inputs and kernel configuration

**Status:** Done (fix rounds 1 and 2 applied). Commits: `a063b1e`, `b9fc915`, `a360e1f`, `c9bf718`, `782bdb9`. Report: `.superpowers/sdd/phase-1/reports/task-1.md`; round-2 report: `.superpowers/sdd/phase-1/reports/task-1-fix-round-2.md`.

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

**Status:** Done after two review fixes. Commits: `dbbf767`, `9a5503b`, `27fb512`. Kernel `src/f1telemetry/kernels/longitudinal.py`, focused tests `tests/test_longitudinal_kernel.py` (34 tests, `kernel` marker).

**Description:** Add the P1 longitudinal state, caller-owned output buffers, and a flat 10 kHz semi-implicit integrator using the P0 Numba conventions.

**Acceptance criteria:**
- [x] Kernel uses preallocated numeric arrays, fixed `dt = 100 µs`, `cache=True`, and `fastmath=False`.
- [x] Two runs with the same seed, configuration, and inputs produce byte-identical traces.
- [x] A representative run stays finite and the loop performs no Python-side allocation or wall-clock reads.

**Verification:** Focused kernel tests, including a byte comparison between repeated outputs; `uv run --frozen pytest <focused-test-file>`.

**Dependencies:** Task 1.

**Files likely touched:** `src/f1telemetry/kernels/`, new `src/f1telemetry/physics/` longitudinal module if needed, focused kernel tests.

**Estimated scope:** Medium.

## Checkpoint: Kernel

- [x] Focused tests pass for determinism, fixed-step progression, and finite outputs.
- [x] Existing `just check` gates remain green.

## Task 3: Add aero and longitudinal tire forces

**Status:** Done. Commits: `1f54838`, `c47ba59`. `src/f1telemetry/physics/forces.py` (new) and
`tests/test_forces.py` (38 tests, `forces` marker). Also touched: `car_spec.yaml` (`aero` provenance,
`tyres.longitudinal_pacejka`, `tyres.slip_ratio_min_speed_m_s`), `KernelConfig` and its builder,
`docs/calibration.md`, `pyproject.toml`. The Task 2 kernel is untouched. Ledger:
`.superpowers/sdd/phase-1/progress.md` § Task 3 status.

**Description:** Calculate speed-dependent drag/downforce and longitudinal Pacejka force from wheel slip and vertical load.

**Acceptance criteria:**
- [x] `Cl(v)` and `Cd(v)` are configuration-driven functions and aero forces use the documented dynamic-pressure relationship.
- [x] Slip ratio guards its low-speed denominator; force sign and tire load behavior are covered by tests.
- [x] Force calculations remain finite and respect the configured longitudinal grip limit.

**Verification:** `uv run --frozen pytest tests/test_forces.py` (38 passed); `uv run --frozen pytest
-m forces`; `uv run --frozen pytest`; `just check`.

**Dependencies:** Tasks 1–2.

**Files likely touched:** `car_spec.yaml`, `src/f1telemetry/physics/`, focused physics tests.

**Estimated scope:** Medium.

## Task 4: Add powertrain, clutch, gearbox, and wheel rotation

**Description:** Connect the synthesized ICE torque curve and MGU-K drive to the wheels through the specified drivetrain, including eight-speed shifts and clutch state.

**Progress:** P1-T4 ICE torque curve is committed (`5ce583f`). P1-T5 gearbox/clutch (`ccd0f35`), P1-T4/P1-T6 powertrain limits (`34ac0e3`), P1-T6/P1-T7 wheel state and force assembly (`753b0b1`), and a caller-supplied signed per-wheel brake-torque path (`dee4226`) are implemented. Brake capacity, hydraulics and brake bias remain synthetic/unmodelled, so this is not a C11 compliance demonstration. Ledger: `.superpowers/sdd/phase-1/progress.md` § Phase 1 slices 4–6.

**Acceptance criteria:**
- [x] Gear ratios, final drive, shift limits, and clutch capacity come from `car_spec.yaml`; gearbox progression and positive-throttle reverse behavior satisfy invariant 7.
- [ ] Launch, trailing throttle, and boost-cut/upshift exercise clutch state; low-speed wheel cases are covered by P1-T6/P1-T7.
- [ ] ICE curve provenance and synthesis assumptions are documented; motor power is accounted for separately from ICE power.
- [x] Caller-supplied brake torque reaches each wheel independently; a symmetric case and deceleration are tested. Brake capacity and full C11 compliance remain open.

**Verification:** Focused drivetrain tests for launch, shift boundaries, clutch transition, and wheel force direction.

**Dependencies:** Tasks 1–3.

**Files likely touched:** `car_spec.yaml`, `src/f1telemetry/physics/`, drivetrain tests, `docs/calibration.md`.

**Estimated scope:** Medium.

## Checkpoint: Drive path

- [x] P1-T5 launch/shift tests pass with finite torque and expected gear direction; P1-T6/P1-T7 then closed the wheel loop with rear-only drive and a signed tyre reaction.
- [x] The rear/left-right drive split and the static load split are recorded as synthetic in `docs/calibration.md` §3; no differential, traction control or ABS is modelled.
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
