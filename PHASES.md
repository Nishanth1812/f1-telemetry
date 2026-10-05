# Build Phases — task-level plan

Companion to `PLAN.md` (§13 is the summary table; this is the executable detail).

## How to use this document

**The rule of a clean phase:** at the end of it, `main` is green, the project builds, there is a
demonstrable thing, and nothing half-finished is left behind. Every phase ends with a tag
(`v0.1-contracts`, `v0.2-straight-line`, …) and a demo. If a phase cannot hit its exit gate, cut it
short and ship at the last clean gate — never merge a broken phase "because we're nearly there."

**Task IDs** are `P<phase>-T<n>` so they can be referenced in commit messages and issues. A task is
0.5–2 days. A phase is done when all its tasks are closed and the exit gate passes.

**Critical path:** 0 → 1 → 2 → 4 → 5 → 7 → 8. Phases 3 and 6 can run in parallel with their
neighbours once their dependencies clear. Stretch phases 9–11 are not on the path.

**First visible demo:** end of P0. Before that you have nothing to show, so P0 is deliberately small
and vertical.

---

## P0 — Contracts, tooling, skeleton {#p0}

**Goal:** one contract, code generation, CI, and a dashboard that displays synthetic telemetry at
the correct per-channel rates.
**Depends on:** nothing. **Effort:** ~1 week.

The purpose of P0 is to prove the contract end-to-end with zero physics. If `channels.yaml` cannot
drive a real dashboard, every later phase is built on sand.

| ID | Task | Notes |
|---|---|---|
| P0-T1 | Repo scaffold: `uv` project, `pyproject.toml`, `ruff`, `basedpyright`, `pytest` config | Pin Python 3.12. Add `just` or `make` targets: `lint`, `type`, `test`, `codegen`, `dev`. Pin `numba` and `llvmlite` explicitly — a floating `numba` has already been observed pulling an incompatible `llvmlite` |
| P0-T1b | Verify the Numba toolchain end-to-end before anything depends on it | One `@njit(cache=True, fastmath=False)` kernel compiles and runs on this machine. Windows 64-bit is supported and LLVM ships in the `llvmlite` wheel, so no compiler should be needed — **prove that here, not in P1** |
| P0-T2 | Author `channels.yaml` — every channel from `PLAN.md` §5.2 | unit, rate, width, dtype, range, noise model, sigma, quantisation, fault eligibility. This is the highest-leverage 2 days in the project |
| P0-T3 | Author `car_spec.yaml` skeleton, versioned to FIA 2026 Section C Issue 16 (2026-02-27) | Only the fields P1 needs: mass, aero area, Cl/Cd curves, ICE torque, gear ratios, final drive. Extend later |
| P0-T4 | Codegen: `channels.yaml` → Python channel registry + dataclasses | Single `CHANNELS` registry; expansion by corner is a function, not 16 hand-written fields |
| P0-T5 | Codegen: `channels.yaml` → Parquet schema (per group) and TypeScript types | Generated files committed to the repo |
| P0-T6 | CI: regenerate codegen and fail on `git diff` | Guarantees generated code never drifts from the contract |
| P0-T7 | Invariant harness — all 8 invariants from `PLAN.md` §11 as running tests | They pass trivially now. They must *exist* before there is physics to fail them |
| P0-T8 | Golden-trace harness: baseline commit, compare, flag diffs over threshold | Per `PLAN.md` §14 — flag for review, do not auto-fail on coefficient changes |
| P0-T9 | Web scaffold: Vite + React 19 + TS, `zustand`, raw `ws` client with backoff reconnect | No charting library. Canvas only |
| P0-T10 | Synthetic frame generator — emits fake frames at correct per-channel rates from `channels.yaml` | Proves the rate contract. Deliberately reads only `channels.yaml`, never hardcodes a rate |
| P0-T11 | Dashboard skeleton: tile grid + one trace panel + connection indicator | Tiles render from synthetic data at 30 Hz |

**Exit gate**
- [ ] Dashboard shows live synthetic telemetry; each channel updates at its declared rate (verify with a stopwatch, not by eye)
- [ ] `just codegen` is idempotent; CI proves committed generated code is current
- [ ] All 8 invariant tests run and pass
- [ ] Changing a rate in `channels.yaml` changes it in the dashboard with no other edit
- [ ] `ruff`, `basedpyright`, `pytest`, web build all green in CI

**Tag:** `v0.1-contracts` · **Demo:** synthetic telemetry dashboard.

---

## P1 — Physics: straight line {#p1}

**Goal:** the car accelerates, shifts, and hits its published performance numbers.
**Depends on:** P0. **Effort:** ~2–3 weeks.

| ID | Task | Notes |
|---|---|---|
| P1-T1 | Numba kernel: flat `@njit(cache=True, fastmath=False)` step function over preallocated float64 arrays | `dt = 100 µs`, semi-implicit Euler. **Fixed step only** — variable step breaks determinism. Follow the six rules in `PLAN.md` §4.1: no classes, no dicts, no allocation in the loop, config passed in as arrays from `car_spec.yaml`, `fastmath` off |
| P1-T1b | Spec parsing outside the kernel → plain arrays | Keeps "no hardcoded constants" compatible with Numba. Test that a `car_spec.yaml` edit changes simulation output with no other code change |
| P1-T2 | Determinism harness: same inputs → byte-identical output | Fixed seed, no wall-clock reads anywhere in the sim. Test it now, not later |
| P1-T3 | Aerodynamics: `Fz_aero = ½ρv²A·Cl(v)`, `Fdrag = ½ρv²A·Cd(v)` | Cl/Cd as speed functions, not constants. Non-proportional-to-v² over the range |
| P1-T4 | ICE torque curve | **Synthesised**, not sourced. Turbo-lag multiplier collapsing below ~4,000 rpm. Label it synthesised in `calibration.md` |
| P1-T5 | Gearbox: 8 ratios, geometric spacing (~1.6), final drive, shift logic, clutch state | Trailing throttle and boost-cut both route through the clutch |
| P1-T6 | Wheel rotational state + longitudinal Pacejka | `κ = (ωr − v)/max(v, ε)` — guard the divide-by-zero at low speed |
| P1-T7 | Force assembly: drive/brake torque → Fx, load = static + aero | First appearance of the vertical load term |
| P1-T8 | Scenarios: `accelerate_to_speed`, `full_throttle` | Uses the P0 scenario schema stub |
| P1-T9 | Calibration: tune aero + torque to hit 0–100 km/h and top speed | Measure against the reference points in `PLAN.md` §11.1, not the old §11 sanity bands. Record each coefficient and its source in `calibration.md` as you go |
| P1-T10 | `fastest-lap` cross-check harness | Run the same `car_spec` through it, diff lap times. Fails gracefully if the dependency won't build — note it, don't block |
| P1-T11 | **Wheel-boundary energy invariant** in CI | Chassis-plus-wheel kinetic-energy change against wheel-torque work, aero drag and tyre-slip work; residual <1%. This is not a fuel-to-vehicle balance because P1 has no engine or motor rotor state. |

**Exit gate**
- [ ] Measure and report the 0–100 km/h time against the **2.32 s** coarse reference, without treating its ±0.30 s sampling uncertainty as a pass/fail tolerance; require the high-speed scenario's transient maximum to reach **≥325.8 km/h**. The first number is a ~3.7 Hz telemetry-derived median; the second is a single-event FIA speed-table reading used only as a reachability floor, since a speed trap is not comparable to terminal or asymptotic speed. The transient maximum — not terminal speed — is checked against 325.8 km/h. `full_throttle` uses a bounded MGU-K deployment followed by an ICE-only tail. The launch scenarios seed `ice_rpm_initial: 12000` during initial clutch slip; the engine integrates delivered torque against reflected clutch load using configured inertia, while a locked clutch applies an ideal wheel-speed constraint. The reproduced engine-state measurements are in `docs/calibration.md`; the 0–100 result misses and the P1 performance gate remains open.
- [ ] Power curve shape plausible across the rev range
- [ ] Invariants 1 (no NaN), 3 (load sum), 6 (energy), 7 (gearbox) pass on real runs
- [x] Two identical kernel runs with the same state and caller-owned inputs produce byte-identical output

The P1 exit gate remains open. The reproduced engine-state baseline reports 6.6598 s to 100 km/h
(reference 2.32 s), a 338.4295 km/h transient maximum during the configured 20 s MGU-K request, and
a 307.6027 km/h ICE-only tail. The brake probe reports 1.265 g for caller-supplied wheel torque; it
is not a brake-capacity result. Longitudinal load transfer and calibration remain open acceleration
work; see `docs/calibration.md` for the baseline and model limitations.
- [ ] `fastest-lap` comparison recorded — agree within a few percent, or the discrepancy is explained

**Tag:** `v0.2-straight-line` · **Demo:** 0–100 run with real traces, or a target miss with a written
explanation of which coefficient is wrong.

**Implementation status (2026-10):** The six-slice longitudinal implementation is on
`feat/phase-1`, including explicit drivetrain controls, 2026 powertrain limits, four-wheel force
assembly, caller-supplied brake torque, deterministic acceleration scenarios and all eight
invariants on real runs. Invariant 6 balances the modeled chassis/wheel boundary and passes its
<1% gate. The optional `fastest-lap` comparison is recorded, with the bundled 2014 car mismatch.
Two straight-line reference points are now recorded in `PLAN.md` §11.1 and `docs/calibration.md` §6,
fixed before any parameter edit, so the gate has something to measure against. Both are deliberately
coarse and neither validates configuration-matched performance: a ~3.7 Hz telemetry-derived 0–100
median, and a single-event FIA speed-table figure used as a reachability floor rather than a terminal
speed. The P1 exit gate remains open. The engine-state scenario suite reproduces a 6.6598 s
0–100 km/h time and a 338.4295 km/h transient maximum against the 2.32 s reference and 325.8 km/h
reachability floor. The last pinned pair predates the launch wiring and is stale. The 0–100 result
misses; the transient floor passes. Longitudinal load transfer, plausible power-curve review and
calibration remain open. Synthetic aero, tyres, brakes and powertrain assumptions remain; this work
does not establish full F1-car fidelity or regulatory compliance.

---

## P2 — Physics: lateral + load transfer {#p2}

**Goal:** the car corners, with believable weight transfer and a real friction ellipse.
**Depends on:** P1. **Effort:** ~3–4 weeks. The largest phase.

| ID | Task | Notes |
|---|---|---|
| P2-T1 | Planar chassis `vy`, `r`, `x`, `y`, `psi`; derive roll, pitch and heave quasi-statically | One yaw-rate state; body/suspension outputs follow the quasi-static suspension contract in `PLAN.md` §2 |
| P2-T2 | Vertical load transfer: static distribution + longitudinal + lateral, per corner | The core of F1 behaviour. Get the roll stiffness distribution right — it is the balance knob |
| P2-T3 | Pacejka lateral, **with load sensitivity on D and B** | Non-proportional μ(Fz). A constant-μ tire understates high-speed downforce badly and will silently break P4 |
| P2-T4 | Combined slip via the similarity method | So the friction ellipse *emerges* rather than being clamped. Needed for P4's speed profile |
| P2-T5 | Relaxation-length states, lateral and longitudinal | `σ = Cα/Cy`. This is the justification for 10 kHz over 100 Hz |
| P2-T6 | Quasi-static suspension: travel, camber gain, bump steer, roll/pitch stiffness split | Documented as a deliberate simplification per `PLAN.md` §2 |
| P2-T7 | Steering geometry + Ackermann + steering limit | Needed for manual driving later |
| P2-T8 | Scenarios: `steady_state_circle`, `constant_radius_speed_sweep` | The sweep is the primary calibration tool for lateral g |
| P2-T9 | Calibration: peak lateral g, steering-balance sensitivity | Record a source-backed matched target before calibration |
| P2-T10 | Invariants 2 (friction ellipse), 4 (sign conventions), 5 (symmetry) | Invariant 5 is the cheapest bug-finder in the project — run it constantly |

**Exit gate**
- [ ] Constant-radius sweep reaches a configuration-matched, source-backed target peak lateral g
- [x] Compare the settled sweep with the 4.0 g historical reference; the current untuned model remains below it, and it does not close calibration
- [x] The friction ellipse and zero-load force bound pass on every wheel in the produced scenarios
- [x] Zero-steer symmetry passes with camber and bump steer disabled; paired left/right steering runs mirror chassis state and four-wheel force/load outputs
- [x] Steering demand changes monotonically with `roll_stiffness_front_fraction` at matched 50 m radius and 20 m/s; front/rear aero-balance sensitivity is deferred because the configuration has no aero-balance input
- [x] Suspension travel-limit flags remain clear throughout the produced scenarios and every sweep run

**Implementation status (2026-10):** The P2 kernel, load transfer, steering, relaxation, combined-slip
forces, scenario truth and channel publication are implemented on `feat/phase-2`. The 50 m circle
settles within 5% of its requested radius at about 0.76 g; the 40–105 m/s coasting sweep runs long
enough for lateral acceleration to settle, matches its 200 m radius, and stays within published
channel ranges. Its settled high-speed point no longer meets the 4.0 g historical plausibility
reference from Pirelli's 2011 Pouhon report. That car, corner and condition do not match this
synthetic neutral-circle sweep, but the earlier transient result above 4.0 g was not a valid
steady-state comparison. The source-backed, configuration-matched lateral-g target is still unset,
so the P2 calibration exit gate remains open. Actual simulation snapshots for the 50 m circle and
105 m/s sweep endpoint are pinned as advisory goldens, with current untuned coefficients recorded
in their descriptions.
P1's 0–100 km/h miss and uncalibrated power curve also remain open; see
`docs/calibration.md`.

**Tag:** `v0.3-lateral` · **Demo:** constant-radius sweep with load transfer visible on all four
corners, and a steering-sensitivity curve.

---

## P3 — Thermal (lumped) {#p3}

**Goal:** tyre, brake, engine and gearbox temperatures that respond to what the car is doing.
**Depends on:** P1 (needs wheel ω and force). **Parallel with:** P2. **Effort:** ~1 week.

Cheap, and it unblocks the entire tyre-temperature half of the analytics story. Do not defer it.

| ID | Task | Notes |
|---|---|---|
| P3-T1 | Lumped-capacitance node base class | `m·c·dT/dt = Q_in − hA(T−T_amb) − εσA(T⁴−T_amb⁴)` |
| P3-T2 | Tyre node: `Q_slip = \|F_patch · v_slip\|` | **Slip velocity, not road speed.** Using road speed gives temperatures an order of magnitude too low — the most common bug in tyre thermal models. Write the test that catches it |
| P3-T3 | Brake node: `Q_slip = τ_brake · ω_wheel` | Per corner |
| P3-T4 | Engine + gearbox nodes: `Q = ṁ_fuel · LHV` | Cooling scaled by airflow, which should rise with speed |
| P3-T5 | Tyre pressure node: gas law driven by carcass temperature and volume | Gives the pressure↔temperature coupling, and makes a leak/leak-fault detectable |
| P3-T6 | Scenarios: `thermal_soak`, `brake_duty_cycle` | |
| P3-T7 | Calibration: equilibrium temperatures per compound and surface | Plausible operating windows, not exact |

**Exit gate**
- [ ] Equilibrium temperatures land in plausible windows per compound
- [ ] Pressure responds to temperature and to an injected leak
- [ ] A test proves slip-velocity (not road-speed) drives tyre heating
- [ ] Thermal state survives a long headless run without NaN

**Tag:** `v0.4-thermal` · **Demo:** brake duty cycle with a visible temperature ramp and cooldown.

---

## P4 — Track, racing line, lap and sector timing {#p4}

**Goal:** a circuit, a fast line through it, a driver that can follow it, and lap/sector timing.
**Depends on:** P2 (needs combined slip and lateral dynamics for the speed profile). **Effort:** ~2–3 weeks.

This phase was missing entirely from the original plan and everything downstream depends on it. Do
not let it drift right.

| ID | Task | Notes |
|---|---|---|
| P4-T1 | Track file format + periodic centerline spline | Per-waypoint width. Elevation and gradient out of scope — record as a known limitation |
| P4-T2 | Arc-length parameterisation `s`, curvature, lateral offset frame | Foundation for everything else here |
| P4-T3 | Minimum-curvature racing line as a QP | Per **section**, not whole lap — numerically better behaved and fast |
| P4-T4 | Iterative re-linearisation loop | Heilmeier et al. Re-solve against the updated line until the path stops changing |
| P4-T5 | Forward-backward speed profile | Braking capacity backward, traction and power limited forward, tire relaxation in the combined-slip check |
| P4-T6 | Lap and sector timing; sector boundaries from the track file | Delta against a reference lap |
| P4-T7 | Lap validity detection | All four wheels on track, complete lap, no DNF. Even without racing, a cut-corner lap is a bad measurement and the DoE must exclude it automatically |
| P4-T8 | Reference driver: pure pursuit on lateral offset + feedforward from the speed profile | Configurable grip-margin derate. This is the "ideal driver" manual laps are scored against |
| P4-T9 | Two circuits; sanity-check the line against published racing-line imagery | Eyeball it. A broken QP is obvious to anyone who has seen a real circuit |
| P4-T10 | Lap-time plausibility check vs published circuit records | Loose band — the model is not validated yet, but a wild miss means a bug |

**Exit gate**
- [ ] Racing line uses available track width sensibly and is smooth
- [ ] Speed profile contains no physically impossible corner
- [ ] Reference driver completes valid laps; lap and sector times are stable run-to-run
- [ ] Lap validity correctly rejects a lap with a deliberate track-limits excursion
- [ ] Line visibly resembles a real racing line on a real circuit

**Tag:** `v0.5-track` · **Demo:** reference driver doing a clean flying lap with live sector deltas.

---

## P5 — Sensors, CAN-FD bus, storage, replay {#p5}

**Goal:** the physics becomes a telemetry system — correct rates, realistic noise, breakable sensors,
durable storage, and a replay that is bit-exact.
**Depends on:** P0 for the contract, P1–P3 for real signals. **Effort:** ~2–3 weeks.

| ID | Task | Notes |
|---|---|---|
| P5-T1 | Per-channel decimation from 10 kHz to declared rate | Anti-alias filtering on the 200 Hz and 100 Hz paths. Without it, decimation aliases and every downstream anomaly finding is a ghost |
| P5-T2 | Quantisation per `channels.yaml` | |
| P5-T3 | Noise models — gaussian, per-channel `sigma` from the contract | **Calibrate a realistic floor.** With too-clean data every detector scores trivially and the project is theatre |
| P5-T4 | Fault injection — all 10 types | Each emits onset time, severity and ground-truth label alongside the corrupted data |
| P5-T5 | CAN-FD frame encode/decode per `PLAN.md` §8.2 | 29-bit ID, DLC, rolling counter, CRC, simulated timestamp |
| P5-T6 | Arbitration, per-message cycle times, bus load metric | |
| P5-T7 | Parquet writer, one file per run, grouped by channel family | |
| P5-T8 | Hot ring buffer (60 s full rate) for live UI and scrubbing | |
| P5-T9 | Replay: Parquet → re-emit as live WebSocket frames | Unlocks fast dashboard iteration without re-simulating |
| P5-T10 | DuckDB views + one verification query per phase gate | The query that answers "find runs where peak brake temp exceeded X" |
| P5-T11 | **Determinism invariant in CI** (byte-identical Parquet across two runs) | The gate for the entire project's credibility |

**Exit gate**
- [ ] Bus load <70% at full sensor rate; a 200 Hz IMU message does not starve the 10 Hz engine message
- [ ] All 10 fault types reproduce deterministically from a seed
- [ ] Anti-aliasing verified: decimate a known high-frequency signal, confirm no alias
- [ ] Parquet round-trips; a channel added in P0 flows through with no other edit
- [ ] Two identical runs produce byte-identical Parquet
- [ ] Replay is indistinguishable from live to the dashboard

**Tag:** `v0.6-telemetry` · **Demo:** the dashboard now showing *simulated* car telemetry, and a fault
injection visibly corrupting one channel.

---

## P6 — Scenario framework {#p6}

**Goal:** tests are declarative, versioned, and runnable faster than real time.
**Depends on:** P5 (needs run manifests and storage). **Parallel with:** P7. **Effort:** ~1–2 weeks.

| ID | Task | Notes |
|---|---|---|
| P6-T1 | Scenario YAML schema, OpenSCENARIO-shaped | `ParameterValueDeclarations` → `Init` → `Storyboard` → `Trigger` |
| P6-T2 | `ParameterValueDeclarations` and `Init` | Parameterised scenarios — the mechanism the DoE in P10 depends on |
| P6-T3 | Action primitives | accelerate-to-speed, brake-to-speed, constant-radius sweep, steady circle, gear sweep, thermal soak, flying lap, lap set |
| P6-T4 | Trigger conditions | Events the scenario can branch on |
| P6-T5 | Headless runner, target >50× real time | Different process from the live runner |
| P6-T5b | **Profile the runner and record real steps/sec** | This is the measurement that reopens the Numba-vs-C++ decision (`PLAN.md` §4.1). Profile before concluding: if the integrator is not the bottleneck, do not touch it — the fix is Parquet batching or pacing |
| P6-T6 | Live runner with real-time pacing | |
| P6-T7 | Run manifest: seed, `car_spec` version, scenario version, setup hash, git SHA | Sufficient to reproduce a run years later. This is what makes results citable |

**Exit gate**
- [ ] 8+ scenario types run in both live and headless modes
- [ ] Headless sustains >50× real time
- [ ] Profile recorded: where the time actually goes, and real steps/sec
- [ ] Manifest is complete enough to reproduce a run from scratch
- [ ] A parameterised scenario can be swept from the command line

**Tag:** `v0.7-scenarios` · **Demo:** one scenario swept over 20 parameter values in under a minute.

---

## P7 — Residual analytics {#p7}

**Goal:** detect injected faults, and publish the numbers that prove it works.
**Depends on:** P5 (Parquet), P6 (scenario sweep), P2/P3 (reduced-order models need physics).
**Effort:** ~3–4 weeks.

The whole project's value concentrates here. Injected faults are the **held-out test set** — never in
the training set.

| ID | Task | Notes |
|---|---|---|
| P7-T1 | Layer 0: validity checks | Range, rate-of-change, stuck value, timestamp monotonicity. Catches most faults with no maths |
| P7-T2 | Layer 1: reduced-order models predicting each channel group from inputs + state | e.g. predict wheel speed from `vx` and `ω`; predict tyre temp from the P3 node |
| P7-T3 | Residual normalisation by the `channels.yaml` noise floor → z-scores | Reuse the contract. Don't hardcode sigmas in two places |
| P7-T4 | Layer 2: PCA on the residual feature matrix; Hotelling T² and Q/SPE | Catches left/right asymmetry and thermal+slip+load combinations that no per-channel threshold sees |
| P7-T5 | Threshold calibration at empirical 99.7% from clean runs | Empirical, not theoretical |
| P7-T6 | Layer 3: CUSUM or EWMA on T² | Thermal fade is a ramp and will never trip a static threshold |
| P7-T7 | Persistence rule, k-of-n windows; tune against the clean-run FPR | The single highest-value change in the whole analytics stack |
| P7-T8 | Localisation: per-channel residual contribution + SHAP | This is what replaces the invented "94% confidence" |
| P7-T9 | Evaluation harness → generates the `PLAN.md` §9.1 table | All 10 fault types × severities × 3 seeds × scenario types |
| P7-T10 | Three baselines side by side: L2 / L2+persistence / +Isolation Forest | If ML does not beat classical, that is a published result, not a hidden failure |
| P7-T11 | Negative control: operational variability, no injected fault, no genuine abnormality | The test that kills over-eager detectors. Must be silent |
| P7-T12 | Commit generated `docs/detection.md` | Regenerated in CI and diffed against the committed baseline |

**Exit gate**
- [ ] Full detection table generated and committed: latency, FPR/h, miss rate, localisation accuracy
- [ ] Negative control is silent
- [ ] Persistence demonstrably cuts FPR without losing detection — show both numbers
- [ ] Localisation names the correct sensor, with a contribution breakdown
- [ ] Every claim in the docs traces to a row in the generated table

**Tag:** `v0.8-analytics` · **Demo:** inject a FL↔FR swap on one corner during a lap; the dashboard
names the front axle 40 ms after onset.

---

## P8 — Dashboard {#p8}

**Goal:** the artefact a race engineer would actually use.
**Depends on:** P5 (replay), P7 (anomalies). **Effort:** ~3–4 weeks.

| ID | Task | Notes |
|---|---|---|
| P8-T1 | WebSocket server, three streams: `live` 30 Hz / `trace` 5 Hz / `event` | Binary, delta-encoded snapshot + patch |
| P8-T2 | Tile grid: powertrain (RPM pair, gear, throttle/brake, boost, ESoC) | RPM **pair** — ICE and MGU-K are separate channels |
| P8-T3 | Trace stack with per-channel rates and a shared time cursor | |
| P8-T4 | Four-corner tyre block: temp, pressure, load, slip | |
| P8-T5 | Brake temperature block | |
| P8-T6 | Aero block: mode, both flap angles, downforce in newtons | |
| P8-T7 | Anomaly panel: severity, contribution, latency, persistence | Real numbers only |
| P8-T8 | Lap/sector bar with delta + lap validity indicator | |
| P8-T9 | Run comparison overlay | Two runs, overlaid traces, metric deltas |
| P8-T10 | Scrubbing from Parquet replay | Replay infrastructure from P5-T9 |
| P8-T11 | Canvas + `OffscreenCanvas` worker rendering, min-max/LTTB decimation | Decimate to chart pixel width *before* drawing. Keep React out of the hot path entirely |
| P8-T12 | Latency instrumentation; measure and publish budgets | Physics step, ingest lag, WS end-to-end, render frame time |

**Exit gate**
- [ ] All panels present and correct
- [ ] 60 fps sustained at full sensor rate, verified on a real monitor
- [ ] Latency budgets measured and published — not estimated
- [ ] Run comparison works across two different runs
- [ ] Nothing in the React render path touches 200 Hz data directly

**Tag:** `v1.0-dashboard` · **Demo:** live run with anomaly injection, then scrub back through it.

---

## Stretch phases {#stretch}

Not on the critical path. Do not start before the core gates close.

**P9 — ML upgrade** (~2 wks, needs P7)
Isolation Forest on the residual feature vector. **Gate:** it beats Layer 2 + persistence, *or* the
result honestly reports that it does not. A TCN autoencoder on residual windows only against a
demonstrated failure of layers 0–3. Never as the starting point.

**P10 — Setup DoE + surrogate** (~2–3 wks, needs P4 + P6)
Full factorial or Latin hypercube over the setup vector. Response = per-sector lap time plus derived
metrics (peak lateral g, corner minimum speed, tyre energy, brake temperature). Fit a surrogate
(`setup → sector time`), predict the optimum, re-simulate, **publish the error**. This is the real ML
contribution — everything else is diagnostics.

**P11 — Multi-node thermal, manual-driving polish, NLP lap-time polish** (open)
Patch-level tyre carcass gradient, disc temperature gradient, steering-wheel/force-feedback input,
and an Ipopt lap-time polish seeded from the P4 QP line. Each is independent; pick by interest.

---

## Cross-phase commitments

Standing rules that apply from P0 to the end. If these erode, the project quietly stops being
defensible:

1. **No hardcoded constants.** Every number lives in `channels.yaml` or `car_spec.yaml`. CI greps for
   stray magic numbers in the physics packages.
2. **The `@njit` boundary is one-way.** State arrays in, state arrays out. Anything that needs a
   dictionary, a dataclass, a string lookup or an exception is outside the kernel. If you find
   yourself wanting to `raise` inside `step()`, the check belongs in a caller.
2. **No `import` from `physics` inside `analytics`, `server`, or `web`.** Lint-enforced. This is the
   rule that makes "swap in real sensors" a real claim rather than an aspiration.
3. **Every coefficient gets a provenance line** in `docs/calibration.md` — a citation, or the literal
   word `synthesised`.
4. **No claim without a generated number.** Documentation is written from the evaluation harness
   output, not from memory.
5. **Determinism is tested, not assumed.** Any change that makes a run non-reproducible is a bug, not
   a trade-off.
