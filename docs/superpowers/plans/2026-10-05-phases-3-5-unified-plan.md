# Phases 3–5 Unified Implementation Plan

> **For agentic workers:** Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task by task. Keep the work in small commits and stop at each gate if its evidence is missing.

**Goal:** Deliver thermal behavior, track/lap simulation, and a deterministic sensor-to-storage telemetry path as one integrated milestone with clean intermediate gates.

**Architecture:** Keep the physics core fixed-step and deterministic. Add thermal models at its numeric boundary; keep track geometry, racing-line planning, and driver logic in the higher-level layers without importing physics kernels; then publish immutable simulation records through contract-driven sensor processing, CAN-FD, storage, and replay. Retain the existing `channels.yaml` contract and generated artifacts as the source of channel metadata.

**Tech Stack:** Python 3.12, NumPy, Numba kernels already in the repository, PyYAML, PyArrow, DuckDB, the existing React/TypeScript dashboard, and standard-library test/support code. Do not add a runtime dependency unless the track optimizer or CAN-FD requirements cannot be met with the installed stack and a measured need is recorded.

**Spec:** `PHASES.md` P3–P5; `PLAN.md` §§3, 7–8.6, 10–12; current contract in `channels.yaml`; P1/P2 implementation and limitations in `docs/calibration.md` and `docs/phase2-demo.md`.

## Global Constraints

- Keep fixed `dt = 100 µs`, fixed seeds, and no wall-clock input to simulation, sensor generation, bus timestamps, or replay.
- Preserve existing generated-code workflow: edit `channels.yaml`, then run `just codegen` and verify with `just codegen-check`.
- Keep layers 4–6 independent of `f1telemetry.physics` and `f1telemetry.kernels`; exchange typed records and arrays at explicit boundaries.
- Label synthetic coefficients and calibration targets; do not present a plausible output as validation or FIA compliance.
- Keep each intermediate state runnable; one final release tag only after all combined-phase exit criteria pass.
- No new dependency, general-purpose framework, or thermal base-class hierarchy without a demonstrated need.

## Review Focus

- Thermal energy inputs, unit conversions, and Kelvin-only radiation terms must remain finite and physically directed across ambient and extreme states.
- Tyre heating must use patch slip velocity, and zero-speed/zero-load cases must not divide by zero or create heat from road speed alone.
- Closed tracks, spline seams, narrow widths, and near-zero curvature must not create discontinuous frames or invalid racing lines.
- The speed planner and reference driver must respect wheel-load/friction limits, braking distance, and track boundaries rather than smoothing through impossible corners.
- Sampling and fault injection must be seed-reproducible; bus timing/arbitration must not starve lower-rate messages; Parquet must round-trip and replay with identical logical frames.

---

## Unified delivery and dependency order

Treat this as one umbrella milestone, internally split into three clean gates:

1. **Thermal gate (P3 content):** thermal and pressure behavior works on established straight-line and braking inputs.
2. **Track gate (P4 content):** valid track, racing line, speed profile, reference driver, and lap events work end to end.
3. **Telemetry gate (P5 content):** the complete record-to-sensor-to-bus-to-Parquet-to-replay path is deterministic and visible in the existing dashboard.

P3 can proceed from existing P1 wheel speed/force/power outputs. Track parsing, periodic geometry, and racing-line work can start alongside P3; the speed profile and driver integration require the P2 lateral/load/combined-slip boundary. P5 channel processing and storage scaffolding can proceed alongside both, but final signal wiring requires stable P1–P3 outputs, and lap/sector replay checks require P4 events.

Before interpreting P4 performance as a model result, review and close (or explicitly leave advisory) the existing P1 0–100/power-curve and P2 matched lateral-g calibration gates. Work on geometry and telemetry does not silently close those older gates. If they remain open, lap-time comparisons stay plausibility checks and the demo must disclose that limitation.

## File and boundary map

Follow nearby naming and split files only when a responsibility is genuinely new:

- `src/f1telemetry/physics/thermal.py` (new): pure numeric thermal/pressure update functions; consume force, slip, wheel speed, fuel/shaft-power, airflow, ambient, and configuration values; write caller-owned state.
- `src/f1telemetry/contracts/` and `car_spec.yaml`: validate and store synthetic thermal capacities, heat fractions, cooling areas/coefficients, tyre volume, and initial conditions. Reuse existing validation patterns.
- `src/f1telemetry/tracks.py` (new) and `tracks/*.yaml` (new): track-file loading/validation, periodic centerline geometry, width, surface, and sector definitions.
- `src/f1telemetry/racing_line.py` (new): sectioned minimum-curvature solve and deterministic re-linearization; consume track arrays and car capability inputs, not physics-kernel imports.
- `src/f1telemetry/driver.py` (new): speed-profile-following reference driver; output steering/throttle/brake requests at the existing scenario/control boundary.
- `src/f1telemetry/telemetry/` (new package only if needed): decimation, quantization, seeded noise, fault injection/ground truth, CAN-FD, and run pipeline modules with narrow responsibilities.
- `src/f1telemetry/testing/`: add thermal-soak, brake-duty, flying-lap, and deterministic telemetry scenarios/records using established scenario patterns.
- `src/f1telemetry/testing/parquet_io.py` and `src/f1telemetry/generated/parquet_schema.py`: extend the existing Parquet path/schema; avoid a competing storage abstraction.
- `src/f1telemetry/server.py`, `web/src/telemetry/`, and `web/src/components/`: connect recorded/live frames and replay through existing WebSocket/store/history patterns; add only the minimum lap/sector and fault indicators needed for the demo.
- `tests/`: focused `test_thermal.py`, `test_tracks.py`, `test_racing_line.py`, `test_driver.py`, and telemetry/bus/storage/replay tests. Keep kernel tests in the existing kernel/physics test boundary.
- `docs/calibration.md`, `docs/phase3-5-demo.md` (new), `README.md`, and `PHASES.md`: record synthetic values, limitations, evidence, combined milestone status, and final demo instructions.

## Task plan

### Task 0: Pin the integration contract and readiness baseline

**Dependencies:** None. **Deliverable:** written boundary choices and reproducible baseline for this milestone.

- [ ] Record the fields and ownership for one internal simulation sample, one timestamp/tick, run seed/config identity, and event/ground-truth record. Use existing records where possible; do not create a second channel registry.
- [ ] Confirm the source of each planned signal (P1/P2 kernel output, P3 thermal state, P4 track event) and the destination rate from `channels.yaml`.
- [ ] Capture current `just check`, `just codegen-check`, and representative deterministic kernel/scenario outputs before changes.
- [ ] Record P1/P2 open calibration gates and the conditions under which P4 lap-time checks are advisory.

### Task 1: Add lumped thermal state and updates (P3-T1, T4)

**Dependencies:** Task 0. **Deliverable:** engine, gearbox, and per-corner thermal states advance stably.

- [ ] Add focused tests for heating, cooling, ambient equilibrium, timestep stability, and finite values at low/high temperatures.
- [ ] Implement small pure update functions around `m·c·dT/dt = Q_in − Q_conv − Q_rad`; convert Celsius state to Kelvin only for radiation, and use airflow-dependent cooling where speed is available.
- [ ] Parse and validate synthetic thermal parameters outside the Numba hot loop; keep names, units, provenance, and source notes in `car_spec.yaml` / calibration docs.
- [ ] Wire outputs through the existing simulation record boundary without adding allocations to the fixed-step kernel.
- [ ] Run focused tests, then `just lint`, `just type`, and `just test`.

### Task 2: Couple tyre, brake, and pressure physics (P3-T2, T3, T5)

**Dependencies:** Task 1. **Deliverable:** braking and slip change per-corner tyre/brake temperatures and pressures.

- [ ] Add a regression test where tyre heat is driven by `abs(F_patch · v_slip)` and demonstrably differs from road-speed-based heating; cover zero slip and zero normal load.
- [ ] Implement per-corner brake heat from `τ_brake · ω_wheel`, with explicit heat fractions and cooling assumptions marked synthetic.
- [ ] Implement tyre pressure from absolute temperature, fixed/configured volume, and a declared gas-mass state; make a bounded leak reduce mass and pressure without coupling the injector to the thermal solver.
- [ ] Keep thermal state finite through sustained updates and define/clamp only physically declared state boundaries; avoid silent magic-number clamps.
- [ ] Run focused tests and all invariant tests.

### Task 3: Add thermal scenarios and calibration evidence (P3-T6, T7)

**Dependencies:** Task 2. **Deliverable:** repeatable thermal-soak and brake-duty evidence.

- [ ] Add `thermal_soak` and `brake_duty_cycle` scenario coverage using fixed initial state, seed, and timestep.
- [ ] Calibrate broad equilibrium windows per configured compound/surface; record the data source or explicitly label every value synthetic.
- [ ] Verify leak response and long-run finite state; compare identical runs byte-for-byte at the state/record level.
- [ ] Record the P3 gate evidence and remaining approximations in `docs/calibration.md` and the combined demo notes.
- [ ] Run `just check` and the targeted invariant/golden checks.

### Task 4: Define and validate track geometry (P4-T1, T2)

**Dependencies:** Task 0; can proceed in parallel with Tasks 1–3. **Deliverable:** two validated closed tracks with continuous geometry.

- [ ] Define the minimal versioned track format: closed centerline waypoints, per-waypoint width, surface parameters, and ordered sector boundaries. Elevation/gradient remain explicitly out of scope.
- [ ] Add validation for too few points, duplicate/degenerate segments, non-positive width, invalid sectors, and closure mismatch; report the field and offending waypoint.
- [ ] Implement deterministic periodic centerline interpolation, arc length `s`, tangent/normal frame, curvature, and lateral offset conversion; pin continuity at the seam.
- [ ] Add two track fixtures with different corner/width patterns and documented provenance; do not claim licensed or exact track survey data unless supplied.
- [ ] Run geometry tests at vertices, between vertices, at the start/finish seam, and on invalid input.

### Task 5: Solve the racing line and speed profile (P4-T3, T4, T5)

**Dependencies:** Task 4 and functional P2 load/combined-slip outputs; calibrated P1/P2 for absolute lap-time claims. **Deliverable:** smooth bounded line and feasible per-distance speed.

- [ ] Add tests for a straight, constant-radius loop, chicane, narrow section, and start/finish seam; assert width bounds, bounded curvature, and deterministic results.
- [ ] Implement sectioned minimum-curvature QP with deterministic iterative re-linearization and explicit convergence/iteration limits. Use the installed numerical stack first; if it cannot solve the required formulation robustly, document measured failure before proposing a dependency.
- [ ] Implement a backward braking pass and forward traction/power pass along `s`, constrained by the model's available loads, combined-slip capability, and tire-relaxation allowance.
- [ ] Detect non-convergence or infeasible sections and return a clear failure result; never smooth an infeasible corner into an apparently valid profile.
- [ ] Compare representative lines visually against cited public imagery; record the comparison and source, without treating visual resemblance as numerical validation.
- [ ] Run focused track/racing-line tests and compare stable speeds/curvature at repeated runs.

### Task 6: Follow the line and produce valid laps/sectors (P4-T6–T10)

**Dependencies:** Task 5. **Deliverable:** reference driver completes deterministic valid laps and events.

- [ ] Add a pure pursuit reference driver with speed-profile feedforward and configurable grip-margin derate; emit control requests through the existing scenario boundary.
- [ ] Add progress projection, lap crossing, sector crossings, elapsed times, reference-lap delta, and lap validity based on all four wheels, full lap completion, and DNF/invalid states.
- [ ] Test stable timing across identical runs, sector boundaries at wraparound, a deliberate track-limits excursion, incomplete lap, and a clean flying lap.
- [ ] Run both tracks; reject invalid laps from timing comparisons and downstream dataset eligibility.
- [ ] Compare lap times with published circuit records only as a loose bug-finding plausibility check; report P1/P2 calibration limitations beside results.
- [ ] Record the P4 gate and a line/lap artifact for the demo.

### Task 7: Build contract-driven sampling and sensor effects (P5-T1–T4)

**Dependencies:** Task 0; wire after Task 3 for thermal channels. **Deliverable:** deterministic sensor frames at declared rates.

- [ ] Add tests for every configured channel rate and for alias rejection on the 200 Hz/100 Hz paths using known high-frequency signals.
- [ ] Implement per-channel decimation with documented anti-alias filtering, followed by contract-driven quantization and seeded Gaussian noise; avoid hard-coded channel rates or sigmas.
- [ ] Implement the ten contract fault types (`dropout`, `freeze`, `spike`, `step`, `gain`, `noise`, `quantise`, `swap`, `saturate`, `stale`) only for eligible channels; record onset, duration, severity, seed, and ground-truth labels independently of corrupted values.
- [ ] Test each fault for deterministic output, expected direction/shape, eligible/ineligible channels, and correct ground-truth timing.
- [ ] Run code generation/checks if contract metadata changes; verify generated Python, Parquet, and TypeScript views stay in sync.

### Task 8: Encode, schedule, and measure the CAN-FD bus (P5-T5, T6)

**Dependencies:** Task 7. **Deliverable:** valid frames and measured bus load under arbitration.

- [ ] Pin the frame layout from `PLAN.md` §8.2: 29-bit ID, DLC, rolling counter, CRC, and simulation timestamp; document byte order and CRC parameters once.
- [ ] Add encode/decode round-trip tests, known CRC vectors, invalid DLC/ID rejection, counter wraparound, and corrupted-frame detection.
- [ ] Schedule messages by configured cycle time and deterministic arbitration; calculate utilization from serialized frame sizes and declared rates.
- [ ] Verify full-load utilization stays below 70% and a 200 Hz IMU stream cannot starve 10 Hz engine traffic.
- [ ] Keep bus time on simulation ticks; no OS timer, socket timing, or wall-clock value enters saved records.

### Task 9: Persist runs and add a bounded live buffer (P5-T7, T8)

**Dependencies:** Tasks 7–8. **Deliverable:** one self-describing Parquet artifact per run and a 60-second full-rate live window.

- [ ] Extend the existing generated Parquet schema and `testing/parquet_io.py` path with run/tick identity, channel-family columns, and fault/event ground truth; retain one file per run.
- [ ] Add a schema-level round-trip check for scalar, four-corner, discrete, and event channels; a new contract channel must flow through codegen and persistence without handwritten schema edits.
- [ ] Implement a fixed-capacity 60-second ring buffer with explicit overwrite behavior and test exact capacity, wraparound, and monotonic tick ordering.
- [ ] Use existing PyArrow/Parquet compression and DuckDB query patterns; avoid a database service or alternate file format.
- [ ] Confirm invalid or incomplete runs retain a validity status and do not become eligible reference/DoE data.

### Task 10: Replay, queries, and end-to-end determinism (P5-T9–T11)

**Dependencies:** Tasks 6–9. **Deliverable:** repeatable stored runs replay as the same logical live/event frames.

- [ ] Replay Parquet through the existing WebSocket contract at original simulated timestamps; live and replay use the same frame mapping.
- [ ] Add DuckDB views and at least one verification query each for thermal threshold, lap/sector result, and fault/ground-truth selection.
- [ ] Run two identical full scenarios and require byte-identical Parquet output. Normalize/omit nondeterministic container metadata rather than relaxing the requirement.
- [ ] Compare live and replay logical frames/events; assert channel values, timestamps, event order, and lap validity match.
- [ ] Confirm replay needs no physics re-simulation and can feed the existing dashboard.

### Task 11: Integrate the demonstration and close the unified gate

**Dependencies:** Tasks 1–10. **Deliverable:** one documented, reproducible end-to-end demo and clean release state.

- [ ] Connect the minimal existing dashboard display to live/replayed telemetry: thermal ramp/cooldown, one fault with ground-truth timing, and lap/sector deltas. Reuse current store/history/connection patterns.
- [ ] Demonstrate a clean flying lap and a separate intentional track-limits rejection; show the same run replayed from Parquet.
- [ ] Run `just check`, `just codegen-check`, and the required deterministic two-run end-to-end comparison; save command/output evidence in `docs/phase3-5-demo.md`.
- [ ] Update `PHASES.md` to mark each P3/P4/P5 sub-gate accurately, record deferred calibration gates/limitations, and apply one combined-phase tag only when all combined exit criteria pass.
- [ ] Leave the repository at a clean, runnable checkpoint with no partial subsystem hidden behind an untested path.

## Combined exit criteria

- Thermal equilibrium is within documented plausible ranges for each configured compound; leak changes pressure; tyre heating is proven to use slip velocity; long runs remain finite.
- Two validated tracks yield a smooth, width-bounded line and a physically feasible profile; reference-driver timing is repeatable; invalid laps are excluded.
- All ten eligible fault modes are seed-reproducible with onset/severity/ground-truth records; decimation anti-alias checks pass.
- CAN-FD frames round-trip and reject corruption; bus load is below 70% with low-rate messages not starved.
- Contract channels round-trip through Parquet; identical complete runs yield byte-identical files; replay matches live logical frames and drives the existing dashboard.
- `just check`, `just codegen-check`, and end-to-end determinism pass; calibration gaps from P1/P2 remain clearly reported and do not get relabeled as completed.

## Schedule and checkpoints

Keep the roadmap's estimates as a starting range: roughly **5–7 weeks of implementation** (P3 about 1 week, P4 about 2–3, P5 about 2–3), plus integration/calibration contingency. Schedule integration in parallel where dependencies allow: Tasks 1–3 with Task 4; Task 5 starts after Task 4 and P2 readiness; Tasks 7–9 can begin once the sample/channel contract is pinned; Task 10 waits for P4 events and all persistence paths. Review at each of the three gates. If a gate misses, ship only the last clean checkpoint and keep the unified release tag open.
