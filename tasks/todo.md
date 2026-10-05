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

**Progress:** P1-T4 ICE torque curve is committed (`5ce583f`). P1-T5 gearbox/clutch (`ccd0f35`), P1-T4/P1-T6 powertrain limits (`34ac0e3`), P1-T6/P1-T7 wheel state and force assembly (`753b0b1`), and a caller-supplied signed per-wheel brake-torque path (`dee4226`) are implemented. New scenario records now exercise caller-supplied clutch state and brake torque. Brake capacity, hydraulics and brake bias remain synthetic/unmodelled, so this is not a C11 compliance demonstration. Ledger: `.superpowers/sdd/phase-1/progress.md` § Phase 1 slices 4–6.

**Acceptance criteria:**
- [x] Gear ratios, final drive, shift limits, and clutch capacity come from `car_spec.yaml`; gearbox progression and positive-throttle reverse behavior satisfy invariant 7.
- [x] Launch, trailing throttle, and boost-cut/upshift exercise clutch state; low-speed wheel cases are covered by P1-T6/P1-T7.
- [x] ICE curve provenance and synthesis assumptions are documented; motor power is accounted for separately from ICE power.
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

**Progress:** Deterministic `accelerate_to_speed`, `full_throttle`, `full_throttle_shifts`, coast/neutral, braking, standing-start MGU-K block and deployment/regen scenarios run through the real drivetrain/kernel and produce records. The prior pinned outputs (6.8998 s to 100 km/h and 307.4189 km/h terminal speed) predate scenario wiring and remain historical only. After the Phase 2 config edit and engine-state integration, the suite reproduced 6.6598 s to 100 km/h, a 338.4295 km/h transient maximum during 20 s of MGU-K deployment, and a 307.6027 km/h ICE-only tail. This is a working baseline before longitudinal load-transfer integration, not calibrated performance. The 0–100 result misses the coarse 2.32 s median from 2026 Belgian GP OpenF1 `car_data`; 325.8 km/h is a reachability floor from the FIA 2026 Australian GP speed table. Both were fixed before tuning and neither validates configuration-matched performance. Longitudinal load transfer remains open; the ~1.265 g brake result uses caller-supplied torque and is not a brake-capacity claim. No coefficient has been tuned. See `docs/calibration.md` § Phase 1 deterministic scenarios.

**Scenario wiring added since those numbers were measured:**

- `standing_launch` and `accelerate_to_speed` seed `scenarios.LAUNCH_ICE_RPM` (12 000 rpm) as an **initial engine state derived from the cited 2026 start telemetry** on `ScenarioSegment.ice_rpm_initial`, not as a held speed or car coefficient. During clutch slip and shift cuts the engine integrates crank torque minus reflected gearbox load using configured ICE inertia; a fully engaged clutch applies an ideal wheel-speed lock. This bounded model has no clutch friction law or engine-inertia feedback into wheel acceleration. The measurements below reproduce this engine-state baseline.
- `full_throttle` now requests MGU-K deployment for 20 s at C5.2.11's crank-referenced limit during top-gear acceleration, followed by a 41 s ICE-only tail. The store and regulation limits clamp actual delivery; the run is 76 s. The shift requests are unchanged and are still the only thing that moves the gearbox; no kernel interface changed. This is what separates the two speed quantities: the **transient maximum** is the reachability-floor quantity, and the **terminal speed** is measured over the motor-free tail. The previous 3 s variant failed CI at 308.0353 km/h; CI run 37095870013 passed the revised transient-floor assertion.

**Reporting status:**

- [x] **Record the 0–100 km/h result:** 6.6598 s against the coarse 2.32 s reference. This is a large miss; its ±0.30 s sampling uncertainty is not a pass/fail tolerance.
- [x] **The transient-speed reachability floor passes:** 338.4295 km/h against ≥325.8 km/h.
- [x] **Launch grip and terminal-tail settling pass:** the launch-grip assertion and the corrected tail-settling assertion pass with the engine-state integration.

**Description:** Implement `accelerate_to_speed` and `full_throttle` with deterministic traces; tune only documented car-spec coefficients toward source-backed acceleration and top-speed references.

**Acceptance criteria:**
- [x] Each implemented scenario runs from fixed initial conditions and emits traces through the existing testing/record pattern.
- [x] 0–100 km/h and top-speed reference points are chosen from cited sources before tuning and recorded in `docs/calibration.md`. **Done** — `PLAN.md` §11.1 and `docs/calibration.md` §6 record both with sources, derivation, and the limits of the evidence. **This revises the earlier ruling** that the 2022 Emilia Romagna start figures and event-specific 2026 speed-trap observations were not targets: the Australian GP speed-table figure is now used, but explicitly only as a reachability floor, since a speed trap mid-straight is not comparable to a terminal or asymptotic speed. **Changed since this item was first written**: the 0–100 figure is a coarse ~3.7 Hz telemetry-derived median whose ±0.30 s is feed quantisation rather than a confidence interval, not a precise acceptance target, and the first-motion window includes the physical launch and excludes only the pre-motion delay.
- [ ] Close the performance gate: 0–100 currently measures 6.6598 s against 2.32 s, while the 338.4295 km/h transient maximum clears the 325.8 km/h reachability floor. See `docs/calibration.md`; the acceleration miss remains open for load transfer and calibration work.
- [x] Calibration parameters remain data-driven and the torque curve is labeled as synthesized. **Performance calibration is not complete** until the open item above passes against the recorded reference points.

**Verification:** Scenario tests plus golden traces for straight-line acceleration; record measured 0–100 km/h and top speed against the reference points in `docs/calibration.md` §6, with the caveat that neither is a published figure.

**Dependencies:** Tasks 1–4.

**Files likely touched:** `src/f1telemetry/`, `car_spec.yaml`, `tests/`, `tests/golden/`, `docs/calibration.md`.

**Estimated scope:** Medium.

## Task 6: Enforce P1 invariants and record independent comparison

**Progress:** Invariant 6 now checks sampled intervals using total chassis-plus-wheel kinetic-energy change against wheel torque work, aero drag and tyre-slip work. All eight invariants pass on real scenario records, including the <1% energy gate. The official Windows `fastest-lap` v0.5 binary ran its bundled 2014 F1/Catalunya case and returned 77.9119 s / 344.377 km/h; its different car model is documented as a mismatch rather than a validation. See `docs/calibration.md`.

**Description:** Add the longitudinal energy-balance gate and run P1 scenarios through the relevant existing invariants; compare against `fastest-lap` when it can be built.

**Acceptance criteria:**
- [x] Energy residual stays below 1% on representative P1 runs and is enforced in CI.
- [x] Invariants 1, 3, 6, and 7 pass on real scenario runs; regression output is deterministic.
- [x] `fastest-lap` comparison is recorded with its 2014 model mismatch and is not used to claim 2026 fidelity.

**Verification:** Focused invariant and golden tests, then `just check`; save comparison method, result, and limitation in `docs/calibration.md`.

**Dependencies:** Tasks 1–5.

**Files likely touched:** `src/f1telemetry/testing/`, `tests/`, `.github/workflows/ci.yml` if needed, `docs/calibration.md`.

**Estimated scope:** Medium.

## Checkpoint: Phase 1 exit

- [ ] P1 performance reporting gate closes. **Open pending recorded results.** The transient floor, launch-grip check and tail-settling assertion all passed CI run 37095870013. Record the 0–100 result and exact transient maximum; terminal speed is separate and is not the reachability-floor quantity.
- [x] All `Justfile` check components pass when run directly; the `just` executable is not installed.
- [x] Calibration inputs, available historical/event-specific references, and the optional solver comparison are recorded for review.

## Checkpoint: Phase 2 implementation

**Progress:** The feature branch now has planar/yaw integration, longitudinal and lateral load
transfer, load-sensitive combined-slip forces, per-wheel relaxation, steering/Ackermann geometry,
quasi-static suspension outputs, signed camber truth, and deterministic circle/speed-sweep scenarios.
Produced runs check force/load/sign invariants; a clean zero-steer control and mirrored-turn pair check
symmetry; energy records include planar, yaw and wheel kinetic energy plus signed aero drag and tire
slip work. Travel-limit flags are checked across all recorded scenario and sweep steps. The 50 m
circle returns about 0.76 g, and the 40–105 m/s sweep's last point produces about 4.11 g. The
historical 4.0 g plausibility floor is not a calibrated target.

**Still open:** Find a configuration-matched lateral-g reference and test condition. Initial
advisory P2 simulation snapshots are pinned and the load-transfer/steering demo is recorded.
Pirelli's 2011 4G Pouhon report is recorded only as historical plausibility context; the sweep now
stays within the published channel ranges. P1 remains open at 6.6598 s to 100 km/h against the
coarse 2.32 s reference; the engine curve is still synthetic and untuned. Do not close either
calibration gate from the P2 plausibility floor.

**Verification (2026-10-05):** `uv run --frozen pytest -q` — 970 passed. Ruff lint and formatting,
basedpyright (0 errors), generated contract check, `npm ci --prefix web` (0 vulnerabilities), and
`npm run build --prefix web` all pass. `just` is unavailable, so each component of the `Justfile`
`check` recipe was run directly.
