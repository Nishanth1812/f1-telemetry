# Phase 2 Lateral Physics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the deterministic P1 longitudinal simulator into a four-wheel lateral model with load transfer, load-sensitive combined-slip tyres, quasi-static suspension, and reproducible circle/speed-sweep scenarios.

**Architecture:** Keep the validated YAML-to-`KernelConfig` boundary and caller-owned Numba buffers. Extend the existing wheel force path so each corner consumes local contact velocity, steering/camber, vertical load and relaxation state; assemble those forces in body axes and advance planar motion plus the explicitly scoped body states. Treat suspension travel and geometry as quasi-static, as `PLAN.md` §2 requires. Keep sensors, thermal behavior, track following and the web dashboard out of this phase.

**Tech Stack:** Existing Python 3.12, NumPy, Numba, PyYAML and pytest. No new dependencies.

**Spec:** `PHASES.md` §P2; supporting scope and model decisions in `PLAN.md` §§2, 4, 11. `docs/calibration.md` and `car_spec.yaml` remain the provenance record.

## Global Constraints

- Preserve the 100 µs fixed step, `@njit(cache=True, fastmath=False)`, preallocated float64 state/output buffers, and Python-side validation boundary.
- Keep each numeric coefficient in `car_spec.yaml`, with regulated values cited and synthesized values explicitly labeled; do not silently substitute team-specific data.
- Preserve the P1 planar sign conventions: body x forward, y left, z up; positive drive slip and positive lateral force for positive slip angle.
- Do not add a multi-body suspension solver, differential, traction control, ABS, thermal model, circuit/track model, or driver-assist system.
- Keep golden comparisons advisory for expected coefficient changes; use blocking physical invariants on real P2 scenario output.
- Do not treat the coarse P1 telemetry reference as a tight calibration tolerance.

## Review Focus

- Corner order and sign changes under left/right mirroring: test all four wheels and both turn directions.
- Zero or negative contact load, lift-off and invalid configuration: prevent negative tire capacity, nonfinite state and silent kernel indexing errors.
- Low-speed slip and zero longitudinal speed: retain finite tire force as the P1 slip regularization is revised for lateral/combined slip.
- Load conservation during braking, acceleration and cornering: verify the four loads sum to static weight plus aero and the modeled inertial transfer.
- Simultaneous Fx/Fy near the tire limit: check the configured combined-slip model at every wheel and every integration step.

---

## Current State and P1 Issues to Carry

**Implementation update (2026-10-03):** The P2 vehicle inputs are in `car_spec.yaml` and the
validated `KernelConfig`. Tested `physics/loads.py` calculates per-corner static, longitudinal and
lateral loads plus quasi-static travel; it reports travel-limit flags and documents its geometric
transfer, aero split and lack of roll-centre/unsprung-mass detail. Tested `physics/tyres.py` provides
steady lateral Magic Formula response with load sensitivity and camber; channel degrees are
converted to radians for dimensionless `B`. New `physics/kinematics.py` and
`physics/relaxation.py` primitives now cover per-corner contact/wheel-frame slip angle and exact
exponential longitudinal/lateral slip-state relaxation. `physics/steering.py` validates the
steering-wheel limit and maps the command to the Ackermann road-wheel pair. The new
`physics/combined_slip.py` implements normalized-slip-vector similarity: each pure-axis curve keeps
its own peak and shape, while the normalized combined radius drives the respective curve and its
direction cosines allocate force. Its exact ellipse follows from both axes sharing the same peak
radius, not a force clamp. A high-load discontinuity found in review is fixed: a lateral peak that
has fallen to zero still consumes lateral slip in the shared radius, so longitudinal grip is not
restored abruptly. The primitive now checks its packed parameter count, has a validated Python
entry point, and computes the reference cornering stiffness once outside the hot loop. The focused
combined-slip suite passes (135); Ruff and basedpyright pass. The P1 state prefix is preserved in a
24-column state buffer with named planar, quasi-static body-output, relaxation and prior-acceleration
indices; the old wheel-seeding and energy code now address only the four wheel-speed columns. This
state expansion is scaffolding: the lateral body update, truthful force records, and steering
scenarios are still open. Roll, pitch and heave are resolved as quasi-static derived outputs, not
independent second-order states, consistent with PLAN.md §2. Relaxation lengths and lateral tire
coefficients remain synthesized and uncalibrated; the current lateral coefficient placeholder
predicts a peak at an implausibly large slip angle and must be calibrated before any handling-
performance gate is treated as meaningful. The Magic Formula C/E validation is now stricter to
support a finite, reachable normalized peak; older custom specs outside those bounds must be
updated before using combined slip.

The scenario runner now advances ICE speed during clutch slip and shift cuts from crank torque,
inertia and reflected load, then applies an ideal wheel-speed lock when the clutch is engaged. This
bounded runner model does not add a clutch friction law or engine-inertia feedback to wheel
acceleration. The reproduced pre-load-transfer baseline is 6.6598 s to 100 km/h, 338.4295 km/h
transient maximum, and 307.6027 km/h ICE-only tail; the acceleration result still misses its coarse
reference. P1 downshift validation, the `ice_power_w` boundary and the final-interval energy residual
have been corrected and regression tested. The P1 performance gate remains open, the `just check`
command is unavailable, and the brake result remains a caller-supplied-torque tire probe. See
`docs/calibration.md` and `tasks/todo.md` for the baseline and remaining gates.

**Initial audit snapshot:** The branch was clean when first inspected. The P1 runtime path is in
`src/f1telemetry/kernels/longitudinal.py`, `src/f1telemetry/physics/forces.py`,
`src/f1telemetry/physics/gearbox.py`, and `src/f1telemetry/physics/powertrain.py`. At that point the
state contained distance, longitudinal speed and four wheel angular speeds, and `KernelConfig` had
no P2 geometry or lateral tire inputs. The typed `GroundTruthStep` already carried `vy`, `ay`, steer
and per-wheel `Fy`, camber and load fields; `channels.yaml` already declared most P2-facing outputs.
These were baseline findings; the implementation status above is current.

P1 is not complete. The following list records findings from the initial audit; corrected items are
described in the implementation update above, and open acceptance issues remain until explicitly
closed or dispositioned:

1. A measured 0–100 km/h time is **6.6598 s** against the coarse 2.32 s reference. The ICE model explains part of the miss: the kernel has no ICE rotational state, derives RPM from wheel speed, and floors it at 4,000 rpm; 370/701 recorded samples were at idle, with a discontinuous drop from the 12,000 rpm launch override. Forcing high RPM changes the time but produces rear slip ratio over 51, so coefficient tuning is not a remedy.
2. The remaining acceleration gap is also structural: fixed static axle loads omit longitudinal load transfer. A bounded probe estimated a 2.896 s ideal-torque/free-engine lower bound without transfer versus 1.612 s when ideal transfer was added. Treat these as diagnostic probes, not calibrated performance predictions. P1's 0–100 gate cannot be closed until the engine state and longitudinal load transfer are represented or the reference is formally dispositioned. Decide whether the ICE/clutch state belongs in P1 and whether P1 acceptance waits for Task 2; do not silently claim that the lateral model alone fixes this.
3. The measured transient maximum is **338.43 km/h** during the configured 20 s MGU-K request, above the ≥325.8 km/h reachability floor; its ICE-only tail settles near **307.6 km/h**. The floor therefore depends on this deployment scenario. Record transient and terminal figures separately and do not call terminal speed the speed-trap comparison. Pin the advisory `full_throttle` simulation baseline before P2 changes the kernel.
4. `tasks/todo.md` still has the P1 `just check` checkpoint unchecked. Run the project check suite at the P1 baseline and record its exact result before P2 changes.
5. The 0–100 reference is a coarse ~3.7 Hz telemetry median with no acceptance tolerance. Record the discrepancy and choose an evidence-based disposition; do not invent a tolerance to close the gate.
6. The launch grip check is too weak: it asserts slip ratio below 1.0 while the Magic Formula peak is roughly 0.2–0.5; observed launch slip is only 0.0459. Replace it with a test that checks expected traction behavior around the modeled peak and catches the free-rev failure.
7. P1's truth channel `ice_power_w` currently reports engine output rather than clutch-transmitted power: shift-cut samples transmit zero while the channel can show up to 280.74 kW. Correct its boundary and add a shift-cut regression before downstream consumers rely on it.
8. Invariant 7 currently rejects every gear decrease, so an ordinary 6→5 downshift violates it. Define legal downshift conditions, correct the invariant and add a downshift regression before P2 braking scenarios.
9. The implemented P1 energy check is a wheel-boundary identity (chassis plus wheel kinetic energy against wheel work, aero drag and slip work) and closes numerically to about 4e-9. `PHASES.md` instead specifies fuel/motor shaft power versus kinetic-energy rate; that documented comparison has a median relative residual of 29.1 and up to 982.9 kW absolute mismatch in the probe. The final scenario interval also hard-codes the energy residual to zero. Keep the implemented wheel-boundary identity as the invariant, update `PHASES.md` and calibration notes to match it, remove the artificial last-interval pass, and only then extend energy accounting for P2 body rotation/heave.
10. Braking currently reuses idle launch torque (873.7 Nm per wheel), has no brake-capacity configuration, and reaches about 1.265 g in the measured scenario versus the rough 5–6 g plan note. Brake-system capacity and performance are outside P2 acceptance; describe caller-supplied brake torques as tire-model probes and remove unqualified braking-performance claims.
11. Current aero, tire, mass split, brake distribution and ICE curve remain synthetic or uncalibrated. The `fastest-lap` comparison uses a mismatched 2014 car and is context only, not validation of this 2026 model. C5.2.3's per-cylinder fuel-energy cap remains unenforced because cylinder count is absent; do not invent a count as a P2 fix.
12. Existing P1 invariants 2/3/4 are structurally weak for P2: Fy and vertical acceleration are zero in those scenarios and symmetric loading makes the current checks trivial. Validate them on real asymmetric circle/sweep simulations. Also define the z-up `az` sign convention before using it for heave/load accounting.

The existing `tests/golden/cornering.json` and its record describe hand-supplied values, not a simulated corner. Retain it as an invariant fixture; P2 acceptance must come from scenarios run through the real kernel. Golden strict mode is currently disabled, so pin the advisory simulated `full_throttle` baseline and preserve its measured values in the report; do not imply that the fixture or advisory CI stage is a blocking regression gate.

## Task 0 — P1 baseline status

**Scope/limitation:** Summary written from the measured values recorded in this plan and the unchanged P1 baseline as inspected on `feat/phase-2`; no code, tests, or services were run to re-verify the figures.

**Baseline check (unchanged).** `just` is unavailable, so the `just check` recipe could not be executed and its checkpoint in `tasks/todo.md` remains open. Equivalent recipe checks on the unchanged baseline passed: Ruff check, format check (38 files), basedpyright 0 errors/warnings, pytest 428 passed, contract/codegen current, `npm ci` 0 vulnerabilities, web build passed.

**Measured values and their limitations.**

- **0–100 km/h: 6.6598 s** against the coarse 2.32 s reference (Plan §11.1; a ~3.7 Hz telemetry-derived median with ±0.30 s quantisation resolution, not a pass/fail tolerance). The miss is structural rather than coefficient error:
  - **ICE:** the kernel has no ICE rotational state, derives RPM from wheel speed and floors it at 4,000 rpm; 370/701 recorded samples were at idle, with a discontinuous drop from the 12,000 rpm launch override. Forcing 12,000 rpm reduces the measured time to about 5.26 s but drives rear slip ratio over 51, so coefficient tuning is not a remedy.
  - **Longitudinal load transfer omitted:** axle loads are fixed static values. A bounded probe gave a 2.896 s ideal-torque/free-engine lower bound without transfer versus 1.612 s with ideal transfer — diagnostic probes, not calibrated predictions.
  - **Disposition:** the acceleration gate is not closed by the lateral model alone; it requires the engine state and longitudinal load transfer to be implemented, or the reference to be formally dispositioned, and acceptance waits for Task 2 if so decided.
- **Full-throttle: 338.43 km/h transient maximum** during the configured 20 s MGU-K request (meets the ≥325.8 km/h reachability floor); the ICE-only tail settles near **307.6 km/h**. Transient and terminal figures are recorded separately; terminal speed is not the speed-trap comparison and the floor depends on this deployment scenario. The advisory simulated `full_throttle` baseline is pinned before P2 kernel changes.
- **Braking: ~1.265 g** vs the rough 5–6 g plan note. Caller-torque-only — it reuses idle launch torque (873.7 Nm per wheel) with no brake-capacity configuration — so it is a tire-model probe, not a brake-capacity claim. Whether an explicit brake capacity is in scope is a pending decision.
- **Energy invariant:** implemented as a wheel-boundary identity (chassis plus wheel kinetic energy vs wheel work, aero drag and slip work) closing to about 4e-9; `PHASES.md` instead specifies fuel/motor shaft power vs kinetic-energy rate, whose documented comparison shows a median relative residual of 29.1 and up to 982.9 kW absolute mismatch in the probe. The final scenario interval hard-codes the energy residual to zero (an artificial pass). The boundary must be reconciled and the artificial last-interval pass removed before extending energy accounting to P2 body rotation/heave.
- **Golden policy:** advisory only for expected coefficient changes; blocking physical invariants apply to real P2 scenario output. Golden strict mode is disabled, so the fixture and advisory CI stage are not a blocking regression gate.

**Disposition.** P1 is not closed. ICE behavior and load-transfer acceptance remain open pending implementation and evidence; the lateral P2 phase does not claim to close the acceleration gate.

## P2 Decisions and Interfaces

Before implementation, settle these once in the plan/spec and `car_spec.yaml` rather than letting each task invent them:

- **Body state contract:** preserve the P1 prefix `[x, vx, omega_fl, omega_fr, omega_rl, omega_rr]`; append named `y`, `psi`, `vy`, one yaw-rate state, quasi-static roll/pitch/heave outputs, four lateral relaxation angles, four longitudinal relaxation ratios and previous `ax/ay/az`. Integrate planar motion and yaw; derive roll, pitch and heave from the quasi-static load/suspension model, without adding body angular-rate states. Remove the duplicate `yaw_rate_blend` from `PLAN.md` §4. Suspension travel remains quasi-static.
- **Force/load coupling:** load transfer depends on acceleration, while acceleration depends on tire force and tire force depends on load. Use the previous-step acceleration as an explicit input to the current 100 µs load calculation; seed it to zero and store the resulting acceleration for the next step. Test against a reference calculation and confirm total-load conservation. This fixed explicit update avoids a same-step algebraic loop without an iterative solve.
- **Reference geometry:** define CG-relative front/rear axle distances, front/rear track, sprung-mass roll/pitch inertia, CG heights, static axle/corner load split and suspension roll/pitch stiffness distribution. The current spec only supplies wheelbase and a provisional front weight fraction, so all additions require provenance and validation.
- **Tire input contract:** define lateral Pacejka parameters, load sensitivity for peak and stiffness, camber response, relaxation lengths, and the similarity-based combined-slip equations. State the zero-load behavior and the order in which relaxation, combined slip and force assembly occur.
- **Steering contract:** scenario input is steering-wheel angle in degrees, positive left; divide by the configured steering ratio, apply Ackermann geometry and validate the configured steering limit before kernel entry.
- **Force/load contract:** corner order stays `FL, FR, RL, RR`; compute contact-patch velocities from body motion and yaw rate, rotate wheel-frame Fx/Fy into body axes, and calculate suspension/load transfer from previous-step acceleration before tire force. Positive `az` is upward in the z-up frame. Load redistribution must conserve total vertical load.
- **Numeric calibration:** Pirelli's 2011 report states 4G lateral acceleration at Spa's Pouhon with cars at 290 km/h. Keep it as historical context, not a matched target: the model uses synthetic coefficients and a neutral 200 m circle with no Spa surface or aero configuration, and its settled sweep remains below 4.0 g. A configuration-matched P2 calibration target remains open. The understeer check compares steering required at matched radius/speed while changing only `roll_stiffness_front_fraction`; the current config has no front/rear aero-balance input.

## File Map

- `car_spec.yaml`, `src/f1telemetry/contracts/car_spec.py`: validated geometry, tire, suspension, steering and integration configuration to flat kernel inputs.
- `src/f1telemetry/physics/forces.py`: four-corner kinematics, tire forces, combined slip and load transfer primitives; preserve existing P1 functions where useful.
- `src/f1telemetry/kernels/longitudinal.py`: widen/rename state indices and compose P1/P2 per-corner forces within the fixed-step integrator.
- `src/f1telemetry/testing/scenarios.py`, `records.py`: deterministic steering scenarios, per-corner initial states and truthful per-step state recording. The current `initial_state` seeds all wheel speeds equally and `Scenario` is documented as straight-line only.
- `src/f1telemetry/testing/invariants.py`: check physical invariants on the real scenario records.
- `channels.yaml` and generated contract outputs: change only if the existing channel names/rates/units cannot represent the emitted P2 outputs; regenerate with the existing generator.
- Tests: extend `tests/test_forces.py`, `tests/test_longitudinal_kernel.py`, `tests/test_scenarios.py`, `tests/test_invariants.py`, `tests/test_car_spec.py`, and `tests/test_contract.py` at the owning boundary.
- `PHASES.md`, `PLAN.md`, `docs/calibration.md`, `tasks/todo.md`: reconcile state definitions, record assumptions/targets, P1 disposition and P2 exit evidence.

## Implementation Tasks

### Task 0: Close the P1 baseline and pin P2 scope

**Files:** `tasks/todo.md`, `PHASES.md`, `docs/calibration.md`, `PLAN.md`.

- [ ] Run `just check` on the unchanged P1 baseline and record the command/result.
- [ ] Run the current acceleration and full-throttle scenarios; record the measured 0–100 crossing and exact transient maximum separately from terminal speed.
- [ ] Trace the low-speed ICE rpm clamp and launch override across the scenario/kernel boundary; implement an explicit ICE speed/inertia and clutch-transfer model or document a bounded alternative with a regression test. Do not tune tire/aero coefficients to compensate for an engine state pinned at idle.
- [ ] Decompose the 0–100 miss into engine-speed and load-transfer causes. Choose whether longitudinal load transfer is implemented in P1 now or whether the P1 acceptance gate remains open until Task 2; do not close the acceleration gate before both causes are addressed or explicitly dispositioned.
- [ ] Correct `ice_power_w` to report the declared power boundary (engine output or clutch-transmitted power, chosen and documented) and pin shift-cut behavior in a regression.
- [ ] Correct invariant 7 to accept physically legal downshifts and reject only invalid gear transitions; add a 6→5 regression before any P2 braking scenario is used for acceptance.
- [ ] Replace the vacuous launch slip threshold with coverage around the configured longitudinal tire peak and a standing-start regression.
- [ ] Record the discrepancy against the coarse 2.32 s reference and disposition the open gate without treating ±0.30 s as an acceptance band.
- [ ] Confirm the transient reachability floor, full-throttle terminal-tail settling, and P1 invariant status against current code, not old notes.
- [ ] Pin the advisory simulated `full_throttle` baseline and report its values before changing kernel behavior; identify that golden strict mode is disabled.
- [ ] Align the energy invariant contract across `PHASES.md`, `tasks/todo.md`, scenario code and `docs/calibration.md` around the implemented wheel-boundary identity; remove the hard-coded final-interval zero before extending the boundary for P2 states.
- [ ] Keep brake capacity outside P2 acceptance; label existing braking as caller-supplied torque/tire behavior only and remove it from physical brake-performance claims.
- [x] Resolve the state contract and model assumptions in “P2 Decisions and Interfaces”; update stale/duplicate `r` and `yaw_rate_blend` descriptions.
- [x] Resolve which body values are true integration states versus derived outputs: roll/pitch/heave are quasi-static; planar motion and yaw are integrated. Keep suspension geometry quasi-static and report travel-limit flags.
- [x] Define positive/negative `az` in the existing z-up convention before connecting vertical acceleration to load or heave.
- [ ] Do not tune P1 coefficients inside this task. If the P1 check or required scenario fails, fix and commit that P1 issue before advancing.

**Done when:** P1 check status and advisory baselines are reproducible and recorded; ICE/power reporting and downshift invariants have regressions; the energy contract is internally consistent; the 0–100 gate is either supported by engine plus transfer behavior or remains explicitly open pending Task 2; every P2 state/config field has declared units, meaning and provenance.

### Task 1: Add validated P2 vehicle, suspension and tire inputs

**Files:** `car_spec.yaml`, `src/f1telemetry/contracts/car_spec.py`, `tests/test_car_spec.py`, `docs/calibration.md`.

- [x] Add contract tests for missing/invalid geometry, stiffness distribution, steering limits, lateral tire parameters, load-sensitivity inputs and relaxation lengths.
- [x] Add only the inputs required by Tasks 2–5. Keep each empirical/synthetic input claimed in `not_regulated`; cite actual regulatory values without inferring geometry or performance from unrelated clauses.
- [x] Extend `KernelConfig` with flat scalar/array fields and build them outside Numba, following the existing P1 config-validation path.
- [x] Verify that invalid/zero physical denominators and mismatched arrays fail before kernel entry, and that changing a YAML coefficient changes the generated config.

**Done when:** P2 config is fully validated and provenance-audited with no hardcoded model coefficients in kernels.

### Task 2: Compute four-corner static and dynamic loads with quasi-static suspension

**Files:** `src/f1telemetry/physics/forces.py`, `car_spec.yaml`, `tests/test_forces.py`, `tests/test_car_spec.py`.

- [x] Add tests for symmetric static loads, positive/negative longitudinal transfer, left/right lateral transfer, combined transfer and zero-load/lift-off boundaries.
- [x] Include longitudinal acceleration load transfer, conserving total load; the P1 0–100 performance gate remains open pending engine/calibration work.
- [x] Implement corner loads from validated CG/axle/track geometry and roll/pitch stiffness distribution; preserve the existing total-load accounting convention for weight, vertical acceleration and aero.
- [x] Derive roll/pitch response, camber gain, bump steer and suspension travel from the documented quasi-static kinematics. Enforce configured travel limits explicitly and expose travel-limit flags.
- [x] Test that mirrored turn/load inputs mirror corner loads, and that front/rear stiffness distribution moves lateral load transfer in the expected direction without changing total load.

**Done when:** all loads remain finite, physically signed and conserved across the tested range; suspension travel remains within declared mechanical limits.

### Task 3: Add per-wheel slip kinematics and load-sensitive lateral Pacejka

**Files:** `src/f1telemetry/physics/forces.py`, `tests/test_forces.py`, `tests/test_longitudinal_kernel.py`.

- [x] Test contact velocity at each corner using `vx`, `vy` and yaw rate, including inside/outside wheel velocity and left/right steering signs.
- [x] Implement front/rear steering angles from the selected input contract and Ackermann geometry; calculate slip angle and camber consistently in the wheel frame.
- [x] Implement lateral Magic Formula with documented load sensitivity on `D` and `B` and configured camber response. Preserve exactly zero force for an unloaded wheel.
- [x] Test force sign, zero slip, camber sign, load sensitivity (non-proportional grip), left/right symmetry and finite output near zero speed.
- [x] Review the low-speed longitudinal slip denominator and cover reverse speed with a regression; retain standing-start and braking coverage.

**Done when:** each wheel produces a correctly signed lateral force from its local states and validated config, with no kernel state integration yet.

### Task 4: Add combined slip and relaxation states

**Files:** `src/f1telemetry/physics/forces.py`, `src/f1telemetry/kernels/longitudinal.py`, `tests/test_forces.py`, `tests/test_longitudinal_kernel.py`.

- [x] Test pure longitudinal, pure lateral, mixed slip, rapid slip transitions, wheel lock/spin, zero load and force bounds for the selected similarity method.
- [x] Implement the documented similarity combination so the ellipse emerges from the configured tire model; do not add a post-hoc force clamp that masks invalid model parameters.
- [x] Add caller-owned lateral and longitudinal relaxation states per wheel, using configured relaxation lengths and the actual patch travel speed with a finite low-speed limit.
- [x] Integrate relaxation in the fixed 100 µs step and pin update order with a one-step expected-value test; keep all state/output allocation outside the kernel.
- [x] Test that at every corner and step, normalized combined force satisfies the ellipse bound within a documented floating-point tolerance.

**Done when:** combined tire force stays bounded and transient slip response is deterministic and covered at the kernel boundary.

### Task 5: Extend the chassis kernel and integrate per-wheel forces

**Files:** `src/f1telemetry/kernels/longitudinal.py`, `src/f1telemetry/physics/forces.py`, `tests/test_longitudinal_kernel.py`.

- [x] Pin the expanded state layout and named index constants; preserve existing P1 `vx` and wheel angular-speed semantics.
- [x] Compute each patch's force in the wheel frame, transform to body axes, sum force and yaw moment using corner position, and integrate planar motion, yaw rate, world position and heading.
- [x] Derive roll/pitch/heave from the quasi-static load/suspension solution and integrate only planar motion and yaw; calculate suspension motion without adding a multi-body linkage state.
- [x] Extend Python-side buffer/state/config validation for every new state column before calling the `boundscheck=False` kernel.
- [x] Add reference-step and determinism tests, plus a zero-steer/zero-camber control for the symmetric P2 setup.
- [x] Add a caller-owned P2 initializer for named body values, four wheel speeds and relaxation states; validate it through the same Python boundary as `simulate`.

**Done when:** a caller-owned P2 state can complete a fixed-step run with finite values and repeatable bytes, and P1 longitudinal scenarios remain covered.

### Task 6: Record P2 truth and build steady-circle/speed-sweep scenarios

**Files:** `src/f1telemetry/testing/scenarios.py`, `src/f1telemetry/testing/records.py`, `tests/test_scenarios.py`, `channels.yaml` only if needed, generated outputs only if changed.

- [x] Extend `GroundTruthStep` with yaw, roll/pitch/heave and suspension outputs needed by the invariants; do not infer truth from sensor channels.
- [x] Record actual P2 values into the existing declared channels (`vy`, `yaw_rate`, lateral acceleration, roll, pitch, per-wheel slip angle/load/camber/travel) at their declared rates.
- [x] Implement deterministic `steady_state_circle` and `constant_radius_speed_sweep` inputs with named radius/speed ranges and controlled steering; validate scenario inputs before kernel entry.
- [x] Add a two-direction turn pair and a separate zero-steer symmetric control. The hand-supplied cornering fixture remains a fixture, not scenario output.
- [x] Widen the signed camber channel range in `channels.yaml` and regenerate the existing contract outputs.

**Done when:** the scenarios produce real four-wheel truth and sensor records, repeat byte-identically, and report achieved radius/lateral acceleration as well as commands.

### Task 7: Add P2 invariants, calibration evidence and exit gate

**Files:** `src/f1telemetry/testing/invariants.py`, `tests/test_invariants.py`, `tests/test_scenarios.py`, `docs/calibration.md`, `PHASES.md`, `tasks/todo.md`, `tests/golden/` only after output is reviewed.

- [x] Exercise invariants 2 (combined-slip bound), 3 (load sum), and 4 (sign conventions) against produced scenarios; check invariant 5 using mirrored simulations and a zero-steer control.
- [x] Extend invariant 6 to account for P2 translational/rotational body kinetic energy and tire slip work; use signed drag consistently in fixture and scenario records.
- [x] Check symmetry with mirrored left/right simulations; retain a zero-steer/zero-camber case as its own control.
- [x] Check travel-limit flags through every recorded step of every scenario and every speed-sweep run.
- [x] Compare the settled sweep with the historical 4.0 g reference and record its shortfall; keep configuration-matched P2 calibration open. The measured sweep stays inside every published channel range.
- [x] Sweep `roll_stiffness_front_fraction` at matched radius/ay and assert steering demand changes monotonically with other inputs fixed.
- [x] Save reviewed steady-circle and final sweep scenario outputs as initial advisory golden traces. They record current synthetic, untuned outputs and do not act as calibration targets.
- [x] Run the complete project quality checks and record the exact result; capture the four-corner load-transfer demo plus steering-sensitivity curve. `just` is unavailable in this environment, so run the `Justfile` check components directly.

**Implementation done when:** real P2 simulation output passes the implemented force/load/sign/symmetry/travel checks, demo and advisory baselines are reviewable, and project checks pass. The separate calibration exit gate remains open until matched source data and test conditions are available.

## P2 Exit Gate

- Constant-radius speed sweep reaches the agreed, source-backed lateral-g target under recorded conditions.
- Combined-slip bound is respected for every wheel and step; zero-load wheels produce no tire force.
- Mirror symmetry and sign conventions pass on simulated left/right turns and zero-steer symmetry cases.
- Steering demand is monotonic when only the modeled front/rear roll-stiffness split changes at matched radius and speed. Aero-balance sensitivity remains deferred because this configuration has no aero-balance input.
- Quasi-static suspension travel remains inside configured travel limits across the stated tested load range.
- Existing P1 checks remain green and the P1 metric disposition remains documented.
- The full check components pass; the demo shows all four wheel loads and the steering-sensitivity curve. `just check` itself is not runnable in this environment because `just` is not installed.

## Verification and Commit Discipline

For every implementation task, add focused failing tests first, run them to confirm the expected failure, implement the minimum change, rerun focused tests, inspect `git diff` and status, then commit one coherent milestone using a plain-language subject. After Task 7, run every component listed in `Justfile`'s `check` recipe. Do not mark an exit criterion complete from a hand-built fixture or a CI pass that omits the measured value the criterion asks to record.
