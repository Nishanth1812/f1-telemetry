# F1 Virtual Telemetry Emulator — Engineering Plan

v2. Rewritten from `F1_Virtual_Telemetry_Emulator_Idea.md`, which is retained as the statement of
intent. This document is the buildable version: locked decisions, contracts, numbers, validation
gates.

The vision is unchanged — **simulate the car → emulate its sensors → understand the car** — because it
is correct. What v1 got wrong is recorded in §1 so the reasoning is auditable.

---

## 1. What was wrong, and how v2 addresses it

| # | Problem in the original plan | Resolution in v2 |
|---|---|---|
| 1 | Simulator never specified. 18 channels incl. suspension load, brake temp, tyre pressure, aero load implies a full dynamics **and** thermal model, with no tire model, step rate, or acceptance criteria. | §4 pins the state vector, models, coefficients and 10 kHz step. §5 gives the aero model, §6 the powertrain, §7 thermal. §11 gives numeric acceptance targets. |
| 2 | Physics rate conflated with telemetry rate. A 200 Hz IMU needs a 1–10 kHz integrator plus tire relaxation dynamics. | §4: 10 kHz fixed step, relaxation-length states, then decimation to per-sensor rates in §8. |
| 3 | ML circular — trained on data the project itself generated. `Anomaly Confidence: 94%` is not a metric anything produces. | §9: residual-first detection where the physics residual *is* the fault; multivariate SPC; persistence rule. Only real numbers reported: severity, contribution, latency, persistence. |
| 4 | No wire between ECU and processing. No channel dictionary. No storage design. **No validation plan.** | §5.1 `channels.yaml` as the single contract, code-generated everywhere. §8 CAN-FD emulation. §10 storage lifecycle. §11 validation, including a free real-world cross-check. |
| 5 | Car spec was pre-2026: single "wing angle", single `RPM` and `Gear`. 2026 has active aero and a hybrid powertrain. | §5.1: `aero_mode` + two flap angles. §6: separate ICE and MGU-K, separate rotation speeds, no shared gear path. |
| 6 | *(Found while writing v2)* Lap times and per-sector analysis need a circuit, but no track, racing line, or lap/sector timing was in scope. Section 8's setup-sweep ML deliverable is impossible without it. | §8.5: minimal track module — centerline + width, min-curvature QP racing line, forward-backward speed profile, sector timing. |

---

## 2. Scope

**In:** one simulated 2026-spec F1 car · one circuit module · virtual sensor/ECU layer with
per-channel rates, calibrated noise and a typed fault taxonomy · deterministic repeatable test
scenarios · physics-residual anomaly detection · setup sweep with surrogate optimisation · engineering
dashboard.

**Explicitly out** — enforce in review, not by good intentions:
environment/track rendering · opponents and AI racecraft · race logic, standings, strategy ·
championship data · multi-car scenarios · chassis structural FEA · audio · tyre audio · weather as a
live-varying input (fixed per scenario, with temperature as a swept parameter only).

**Resolved: manual driving is IN, scoped down.** v1 proposed cutting it. That was wrong — a keyboard
throttle-brake-steer path costs almost nothing once a track exists, and it makes the dashboard
demonstrable by hand. What it does *not* get is a lap-time quality bar. So: manual control into the
track with a stability assist and a configurable steering limit, evaluated on lap *validity* and
lap-time *distribution*, not on competitiveness. No input-device SDK work, no force feedback, no
driver model. A pure pursuit + speed-profile path-follower serves as the reference "ideal" driver, and
manual laps are compared against it.

**Resolved: suspension is quasi-static, and that is a decision, not a compromise.** Full double-wishbone
kinematics buys three channels nobody will interrogate, and it is the single largest scope trap in the
project. What we do instead: a 4-corner load-transfer model with camber gain, roll/pitch stiffness
distributions, and quasi-static travel from the same kinematics, all driven from the 6-DOF chassis.
`docs/calibration.md` records this as a known simplification with its expected error, so nobody
mistakes it for a full kinematic solver later.

---

## 3. Architecture

```text
┌───────────────────────────────────────────────────────────────────┐
│ 0. CONTRACTS   channels.yaml  ·  car_spec.yaml  ·  scenario.yaml  │
├───────────────────────────────────────────────────────────────────┤
│ 1. PHYSICS CORE   10 kHz fixed step · 6-DOF chassis                │
│                   4 × (rotation + Pacejka + relaxation)            │
│                   aero · powertrain · lumped thermal                │
├──────────────┬────────────────────────────────────────────────────┤
│ 2a. GT       │ 2b. VIRTUAL SENSORS      ← fault injection lives  │
│    channel   │     per-rate decimate · quantise · bias · noise    │
│    bus       │     dropout · freeze · spike · latency              │
├──────────────┴────────────────────────────────────────────────────┤
│ 3. VIRTUAL ECU / CAN-FD   arbitration · DLC · cycle times · load  │
├───────────────────────────────────────────────────────────────────┤
│ 4. LINK   binary frames → Arrow IPC → Parquet (+ hot ring buffer)  │
├──────────────────────┬───────────────────────────────────────────┤
│ 5. ANALYTICS          │ 6. DASHBOARD                              │
│    validity · residual │    live tiles · traces · anomaly panel   │
│    MSPC · persistence  │    run-vs-run · setup comparison          │
│    ML                  │                                           │
├──────────────────────┴───────────────────────────────────────────┤
│ 7. TEST FRAMEWORK  deterministic scenarios · live + headless      │
│ 8. TRACK           centerline · racing line · sectors · lap timing │
└───────────────────────────────────────────────────────────────────┘
```

Two hard architectural rules, both of which v1 gestured at and did not commit to:

- **The only contract between layer 1 and layers 4–6 is `channels.yaml` plus a `GroundTruth` side
  channel.** Swap in real sensors and nothing above layer 3 changes. This is the same split the
  industry uses: ASAM OSI 3.8.0 keeps `GroundTruth` and `SensorData` as separate interfaces over
  protobuf. Ours is the same idea, smaller.
- **Nothing in layers 4–6 may import from layer 1.** Enforced by a lint rule in CI, not a convention.

---

## 4. Physics core

**Language: Python + Numba `@njit`. No compiled extension, no C++, no Rust.** Rationale and the
rules it imposes in §4.1 — read that before writing `step()`.

**Step:** fixed 10 kHz (`dt = 100 µs`), semi-implicit Euler or RK4. Fixed step only — variable step
destroys determinism, and determinism is a hard requirement (§8).

### 4.1 Why the physics core is Numba, not C++

**Decision: one language for the whole project, with Numba compiling the physics kernel.**

The reasoning, since it will be questioned:

*Speed is not the constraint.* One second of driving is 10,000 steps × 4 tires × ~200 flops ≈ 2
MFLOPS — less work than loading a web page. The heaviest anticipated workload, a 500-run setup sweep
of 90-second laps, is ~9×10⁸ steps: roughly 30 minutes under Numba versus ~3 minutes under C++.
Twenty-seven minutes, once or twice across the whole project, does not justify a toolchain.

*Iteration speed is the constraint.* The physics core is where coefficients get tuned, golden traces
get diffed, and `NaN`s get chased at 10 kHz. In Python that is a print statement and a re-run. Across
a C++ boundary it is a build step, a second language, pybind11 bindings, and a C++ debugger.

*The bottleneck is elsewhere.* If a run is slow, the suspect is Parquet write throughput, DuckDB
query time, or live-renderer pacing — not the integrator. Parquet and DuckDB are already native C++
called from Python, so native code is already in the stack exactly where the performance is. A
compiled integrator would not speed up the actual bottleneck.

*Revisit condition.* At P6, when the headless runner is built, measure real steps/sec against the
>50× real-time target. **Profile before concluding anything.** If the integrator itself appears in the
profile, this decision reopens. If Parquet or the pacing loop appears, the fix is buffering or
batching and C++ would not help either.

**Rules this imposes on the core — read before writing `step()`:**

1. `step()` is a **flat numeric kernel**: numpy arrays in, numpy arrays out. No classes, no
   dictionaries, no string-keyed lookups, no exception handling in the hot path.
2. All state lives in **preallocated float64 arrays** owned by the caller. No allocation inside the
   loop — reuse buffers.
3. Configuration is parsed from `car_spec.yaml` **outside** the kernel and passed in as arrays. This
   is what keeps "no hardcoded constants" (§14) compatible with Numba's constraints.
4. Use `@njit(cache=True, fastmath=False)`. `cache=True` so recompilation is not paid on every run.
   Leave `fastmath` **off** — it permits floating-point reassociation, which is a determinism and
   invariant risk (§11), and it buys almost nothing at this workload size. Revisit only if profiling
   demands it.
5. Do not reach for `numba.prange` until a single-threaded measurement shows the integrator is the
   bottleneck. Parallelism inside a sequential ODE step needs careful reduction, and it breaks
   byte-identical determinism (§8.6) unless handled deliberately.
6. Everything outside the kernel stays ordinary Python and remains fully testable, debuggable and
   mockable. `basedpyright` will not type-check inside `@njit` bodies — that boundary is expected,
   not a bug.

Known cost: `@njit` cannot compile arbitrary Python. If a future coefficient is best expressed as
something exotic, compute it *outside* the kernel and pass the result in, rather than reaching for a
compiled extension.

**State vector:**

```text
chassis    vx, vy, r, roll, pitch, heave, yaw_rate_blend, x, y, psi
per wheel  omega, alpha_relax, kappa_relax, Fz, Fy, Fx, temp, pressure
powertrain ice_rpm, mgu_k_rpm, gear, clutch, eso, boost_remaining
```

**Tire.** Pacejka Magic Formula, lateral and longitudinal, with the four properties without which F1
behaviour is not believable:

- load sensitivity on `D` and `B` — grip is *not* proportional to load; a constant-µ tire understates
  high-speed F1 downforce badly
- combined slip via the similarity method, so the friction ellipse emerges rather than being clamped
- relaxation length `σα = Cα / Cy` for both lateral and longitudinal, so transient slip is right.
  This is the difference between a tire that responds instantly and one that behaves like rubber, and
  it is why the step rate must be 10 kHz and not 100 Hz.
- calibration: F1 slick carcass stiffness 150–250 N/mm; rolling radius 1.02–1.05 × loaded radius;
  wheel+tyre rotational inertia 0.5–1.2 kg·m²

**Seed coefficients from an existing published F1 parameter set** — Limebeer & Tremlett's
open F1 model — and record provenance in `docs/calibration.md`. Do not invent numbers you cannot
cite.

**Reusing `fastest-lap`.** It is a real, MIT-licensed C++/Python vehicle-dynamics library with a 3DOF
F1 model, an Ipopt/CppAD optimal-lap-time solver, and a Catalunya lap in about a minute. **Use it as
a calibration and validation reference, not as the runtime core**, for three reasons: its F1 model is
3DOF (no heave, roll, or pitch — we need all three for the suspension and aero channels); it models
neither 2026 active aero nor a hybrid powertrain; and it has been unmaintained since 2023 with thin
documentation. Concretely: run our `car_spec` through it and diff the resulting lap times. If our
engine agrees within a few percent on a known circuit before we add active aero, the core is sound.

---

## 5. Car spec and aerodynamics

`car_spec.yaml`, versioned to a regulation issue — e.g. **FIA 2026 F1 Regulations Section C (Technical),
Issue 16, 2026-02-27**, plus the 2026-04-21 energy-management refinements. 2027/28 changes were agreed
2026-06-10, so this file will drift; versioning it makes that a data edit rather than a refactor.
**No hardcoded constants anywhere in the codebase.**

Starting point, all to be confirmed against the regulation text during phase 1: mass ~800 kg
including driver; 400 kW ICE and 350 kW MGU-K on a 53/47 split; 3.4 m wheelbase; 1.9 m floor width;
18" wheels retained but smaller diameter and narrower tread than 2022.

### 5.1 Aerodynamics

```text
Fz_aero = ½ ρ v² A · Cl(v, mode)
Fdrag   = ½ ρ v² A · Cd(v, mode)
```

- `Cl`, `Cd` as functions of speed, with a Reynolds-ish sensitivity term. Downforce and drag are not
  proportional to `v²` over the whole range.
- **Active aero is a state machine, not a blend.** `Z-mode` (closed, high downforce, default) and
  `X-mode` (open, low drag, permitted zones only). Transitions carry hysteresis and a finite
  mechanical travel time — a wing does not teleport. Channels: `aero_mode`, `fw_flap_deg`,
  `rw_flap_deg`, `zone_id`.
- Ground effect: partially-flat floor and lower-powered diffuser for 2026, so a single `Cl` curve is
  insufficient. Model a ride-height sensitivity term; this is also what makes porpoising observable
  in the telemetry, which is a nice free anomaly case.
- Include an `A` (reference area) and `Cl·A`, `Cd·A` as derived channels so the dashboard can show
  downforce in newtons rather than a dimensionless coefficient.

### 5.2 The channel dictionary — the core artifact

This is the project's most valuable deliverable. Everything is downstream of it: Python dataclasses,
CAN-FD packet layout, Parquet schema, WebSocket protocol, TypeScript types. One source of truth,
code-generated — never hand-maintained in five places.

```yaml
- name: wheel.speed
  unit: km/h
  rate_hz: 100
  width: 4              # per-corner expansion
  corners: [FL, FR, RL, RR]
  dtype: float32
  range: [0, 400]
  noise_model: gaussian
  sigma: 0.6
  quantise: {bits: 12, full_scale: 400}
  fault_eligible: [dropout, freeze, spike, step, gain, noise, quantise, swap, saturate, stale]
```

Table form, for reading:

| Group | Channels | Unit | Rate |
|---|---|---|---:|
| Chassis | speed, vx, vy, yaw rate, lateral/long accel, roll, pitch | m/s², °/s | 200 Hz |
| IMU | accel x/y/z, gyro x/y/z | m/s², °/s | 200 Hz |
| Powertrain | ice_rpm, mgu_k_rpm, gear, throttle, clutch, ice_torque, mgu_k_power, boost_remaining, eso_pct, fuel_flow | rpm, kW, % | 100 Hz |
| Aero | mode (Z/X), fw_flap_deg, rw_flap_deg, zone_id, downforce_n | deg, enum, N | 100 Hz |
| Wheel | 4 × wheel speed, slip ratio, slip angle, vertical load, camber | km/h, %, deg, N | 100 Hz |
| Wheel thermal | 4 × suspension travel, tyre pressure, tyre temp, brake temp | mm, psi, °C | 20 Hz |
| Thermal | engine temp, gearbox temp, coolant temp, brake duct air | °C | 10 Hz |
| Session | lap index, sector index, lap time, sector time, delta | s, enum | event |
| Driver | brake pressure, steering angle, pedal travel | bar, deg, % | 100 Hz |

Naming corrections against the original doc, which are substantive:

- `gear` is ambiguous on a 2026 car — the ICE has an 8-speed gearbox, the MGU-K is not on that gear
  path. The schema forces `ice_rpm` and `mgu_k_rpm` to be separate channels.
- "Front wing angle / rear wing angle" as single setup numbers is a 2022 concept. Replaced by
  `aero_mode` plus two flap positions.

**Sizing:** ≈ 4,600 channel-samples/s ≈ 18 kB/s ≈ 66 MB per hour of driving raw at float32.
Parquet + zstd lands around 10–20 MB/h. Budget against this, not against intuition.

---

## 6. Powertrain

2026 is a hybrid. Modelling it as one engine with one RPM is wrong and produces nonsense
telemetry.

```text
ICE:  torque_curve(rpm) → gearbox(8-speed) → clutch → differential → wheels
MGU-K: motor+inverter ──────────────────────────────► wheels   (no shared gear path)
store: 7 MJ · superclip ~2–4 s · Overtake Mode +150 kW cap, +0.5 MJ
        energy taper above 290 km/h for the leading car
```

- `torque_curve(rpm)`: synthesised from published 400 kW peak power, peak-torque rpm, and a
  turbo-lag torque multiplier that falls off sharply below ~4,000 rpm. No public F1 torque curve
  exists, so this is explicitly a *synthesis* and `docs/calibration.md` says so.
- 8-speed ratios + final drive: realistic ratio spacing (≈1.6 apart, geometric), top speed from the
  drag-limited solve. Recorded in `car_spec.yaml`.
- **Energy invariant, enforced in CI:** `d(KE)/dt` must equal fuel power + MGU-K power − drag work,
  to <1%. This single test catches most powertrain bugs.
- Clutch model: launch, trailing throttle, and the boost-cut on upshift all go through it. A
  single-channel `gear` with no clutch state produces traces no one believes.

---

## 7. Thermal

Lumped-capacity nodes for tire, brake, engine, gearbox. Roughly 20 lines each and enough to power
the entire tire-temperature anomaly story.

```text
m c  dT/dt = Q_slip + Q_drag − h A (T − T_ambient) − ε σ A (T⁴ − T_amb⁴)

tire:   Q_slip = |F_contact_patch · v_slip|   (slip velocity, not road speed)
brake:  Q_slip = τ_brake · ω_wheel
engine: Q = ṁ_fuel × LHV
```

- `Q_slip` for the tire must use **slip velocity**, not road speed. Using road speed produces
  temperatures an order of magnitude too low and is the most common bug in tyre thermal models.
- Pressure: a gas-law node driven by carcass temperature and volume, so `tyre_pressure` responds to
  `tyre_temp` and to a puncture/leak fault. That coupling is what makes pressure worth logging.
- Multi-node (patch-level carcass gradient, disc temperature gradient) is stretch, phase 9.

---

## 8. Emulation, storage, transport, track, scenarios

### 8.1 Virtual sensors

Per channel: decimate from 10 kHz to the channel rate (with anti-alias filtering on the 200 Hz and
100 Hz paths), quantise to the sensor's real resolution, then apply the fault chain.

**Fault taxonomy.** Every entry carries severity, onset time and ground-truth label. This is what
turns "Sensor Failure Test" from a demo into a benchmark.

| Fault | Model |
|---|---|
| Dropout | N consecutive frames missing |
| Freeze | value holds, frame counter keeps incrementing |
| Spike | single-sample impulse, ±kσ |
| Step offset | slow bias drift |
| Gain error | scale 1 ± k |
| Excess noise | variance inflation |
| Quantisation | coarser effective bit depth |
| Swap | FL↔FR, RL↔RR |
| Saturate | clipping at full-scale |
| Stale replay | value timestamped t−k, re-sent live |

**Simulate a realistic noise floor in normal operation.** `sigma` lives in `channels.yaml` per
channel class. Without it every detector scores trivially and the project is theatre.

### 8.2 Virtual ECU / CAN-FD

29-bit IDs, DLC, per-message cycle times matching the rate table, rolling counter + checksum, and
arbitration priority. Then measure bus load and prove a 200 Hz IMU frame does not starve the 10 Hz
engine-temperature message.

```text
offset  size  field
0       2     id          (29-bit, little-endian)
2       1     dlc
3       1     flags       (error / extended / rtr)
4       4     counter     (rolling)
8       8     t_us        (simulated, not wall clock)
16      N     payload     (columnar: N/channels_per_frame × float32)
```

This is cheap to emulate (~300 lines) and it is the fidelity detail that makes the project read as
an actual telemetry stack rather than a Python script with a web UI.

### 8.3 Storage and data lifecycle

Do not put 200 Hz row-per-sample into Postgres. It will work, and it will dominate your life.

| Tier | Contents | Medium | Retention |
|---|---|---|---|
| Hot | last 60 s full rate, per active run | in-memory ring buffer | ephemeral — live UI, scrubbing |
| Warm | full-rate run data, one Parquet file per run | Parquet + zstd | **all test runs, forever**; free-drive 30 days |
| Cold | 1 Hz and 0.1 Hz downsample, feature vectors, anomaly labels | Parquet | forever |
| Query | ad-hoc analysis over all of the above | DuckDB | — |

DuckDB over Parquet is the right first choice: time-window SQL over the whole archive with no server
process, which is exactly what "compare run 412 against the 30 best brake-temperature runs" needs.
Add TimescaleDB only when concurrent live multi-user dashboards demand it. Retention tiers by
artifact value — a test run is the product; a free-drive lap is not.

### 8.4 Transport and the dashboard

- **Physics → bus:** binary, 10 kHz internal, published at sensor rates.
- **Bus → ingest:** binary columnar. JSON for config and control messages only. JSON at 200 Hz × 70
  channels is the first thing that will kill the project.
- **Server → browser:** one WebSocket, three streams — `live` (30 Hz, ~24 display channels, binary,
  delta-encoded snapshot + patch), `trace` (5 Hz downsampled history), `event` (anomalies, gear
  changes, aero mode changes, lap boundaries).
- **Do not stream full rate to the browser.** A race dashboard needs 30–60 fps of readable tiles plus
  a chart window, not 200 Hz. Fetch full rate on demand for analysis.

Panels: powertrain (RPM pair, gear, throttle/brake, boost, ESoC) · dynamics (lat/long g, yaw, slip
angle) · four-corner tire block (temp, pressure, load, slip) · brake temps · aero (mode, flaps,
downforce) · trace stack with per-channel rates and a shared time cursor · anomaly panel · lap/sector
bar with delta · run comparison overlay.

Frontend: React + TypeScript, **canvas** not SVG/DOM — thousands of points per frame will stall
reconciliation. Ring buffer per channel in a Web Worker with `OffscreenCanvas`, min-max/LTTB
decimation to chart pixel width before drawing, render loop on `requestAnimationFrame`, React state
kept entirely out of the hot path.

Write down and measure the budgets: physics step, ingest lag, WS end-to-end latency, render frame
time. "Real-time" is not a requirement you can assert, only one you can measure.

### 8.5 Track, racing line, lap and sector timing

Required by the setup-sweep deliverable (§9.4) and by manual driving (§2). Minimal and sufficient:

1. **Track representation** — centerline as a closed periodic spline, per-waypoint width, plus
   surface parameters. Elevation and gradient are out of scope initially; note that as a known
   limitation.
2. **Racing line** — minimum-curvature QP with iterative re-linearisation (Heilmeier et al.), solved
   per section, not the whole lap at once, so it stays fast and numerically well-behaved. Literature
   is clear that pure minimum-curvature is a good but not optimal proxy, and that a minimum-*time*
   line is worth roughly 4% over minimum-curvature — so:
3. **Speed profile** — forward-backward integration along the line: braking capacity backward,
   traction and power limited forward, tire relaxation included in the combined-slip check.
4. **Optional polish** — seed a full NLP (Ipopt) from the QP result if the section is a
   headline benchmark. Not required for the DoE to be valid; the QP + speed profile is deterministic
   and repeatable, which is what matters more here.
5. **Reference driver** — pure pursuit on lateral offset + feedforward from the speed profile, with a
   configurable grip-margin derate. This is the "ideal driver" that manual laps are scored against.
6. **Lap and sector timing** — progress along `s`, sector boundaries declared in the track file,
   lap validity (all four wheels on track, complete lap, no DNF), and a delta against a reference
   lap. Lap validity matters even without a race: an out-lap or a cut-corner lap is a bad
   measurement, and the DoE must exclude it automatically.

### 8.6 Test framework

Scenarios declared in YAML, OpenSCENARIO-shaped: `ParameterValueDeclarations` → `Init` →
`Storyboard` → `Trigger`. Primitives: accelerate-to-speed, brake-to-speed, constant-radius corner
with speed sweep, steady-state circle, gear-shift sweep, thermal soak, single flying lap, full lap
set.

**Determinism is a hard requirement.** Fixed seed, fixed step, no wall-clock dependence. Same
scenario + same setup = byte-identical Parquet output. Without this, "repeated under the same
conditions" is false and no result is comparable.

Two execution modes, separate processes:
- **Live** — paced at real time, feeds the dashboard.
- **Headless batch** — faster than real time, target >50×, for sweeps and training-set generation.
  You cannot produce a usable dataset at 1× on demand.

---

## 9. Analytics — residual first

The structural insight this whole design rests on:

> In simulation the physics is exact. So the residual between the reduced-order model prediction and
> the sensor reading **is** the injected fault, in physical units.

One computation yields detection, attribution and perfect labels. Build on it.

**Layer 0 — validity (deterministic, no ML).** Per channel: range check, rate-of-change check,
stuck-value check, timestamp monotonicity. Catches most sensor faults outright.

**Layer 1 — physics residual.** A reduced-order model predicts each channel from inputs + state.
`r(t) = y(t) − ŷ(t)`, normalised by the per-channel noise floor from `channels.yaml` to give a
z-score. Because the residual *is* the fault, this gives detection, attribution and truth labels at
once.

**Layer 2 — multivariate.** Stack residuals that are individually plausible but jointly impossible
— left/right asymmetry, thermal + slip + load combinations — into a PCA feature manifold. Alarm on
Hotelling `T²` (in-manifold) and `Q`/SPE (out-of-manifold). Classical multivariate SPC (Johnson &
Jackson plus Hotelling). Cheap, interpretable, and it catches exactly the fault class per-channel
thresholds miss. Calibrate the threshold at the empirical 99.7% from clean calibration runs, not from
theory.

**Layer 3 — temporal and persistence.** CUSUM or EWMA on `T²` for slow drift — thermal fade is a
ramp, not a step, and will never trip a static threshold. Plus a persistence rule: alarm only if the
statistic exceeds threshold for k of the last n windows. Published digital-twin work is unambiguous
that the persistence rule is what drives false alarms toward zero while keeping finite detection
delay. A single-threshold detector on a 200 Hz stream is unusable.

**Layer 4 — ML, only where 0–3 fall short.**
1. *Isolation Forest* on the residual feature vector. Fast, handles mixed feature types, and
   attribution falls out of the path length. This is the ML headline, not an afterthought.
2. *TCN / 1-D CNN autoencoder* on residual windows, for pattern anomalies invisible to marginal
   statistics. Justified by a **measured** failure of layers 0–3, never in advance.

Do **not** lead with an LSTM autoencoder on raw 200 Hz channels. On real vehicle engine telemetry,
benchmarks found simpler non-parametric methods (k-means distance, one-class SVM) matching or beating
deep autoencoders and DeepSVDD; the deep models only win where the pattern is genuinely complex. The
convolutional/attention autoencoder work that looks most on-topic was trained on powertrain drive-cycle
data, which is a different problem. Classical-first is both better engineering and easier to defend.

**Layer 5 — explainability and localisation.** SHAP on the Isolation Forest, plus per-channel residual
attribution. This replaces the invented `Anomaly Confidence: 94%` with four real numbers:

```text
severity            Mahalanobis distance (T²) at alarm
contribution        rear_left.slip_ratio 0.14 · rear_left.tyre_temp 11°C ·
                    3/4 lateral accel channels inconsistent
detection latency   47 ms after onset
persistence         12 of last 20 windows
```

### 9.1 Evaluation protocol

Stated up front so the numbers mean something. No detector ships without this table.

- **Training set:** clean runs only. Unsupervised.
- **Test set:** runs with injected faults. **Injected faults are never in the training set** —
  training and testing on the same injected anomalies is the classic leaky-evaluation trap, and it
  would make every number here meaningless.
- **Sweep:** every fault type × every severity × 3 seeds × every scenario type. Report as a matrix.
- **Metrics:** detection latency (ms from onset to alarm), false alarm rate (alarms per hour on
  clean runs — the number that decides whether the thing is usable), miss rate, localisation
  accuracy (did it name the right sensor?), and false-alarm-to-detection ratio.
- **Baseline comparisons, all three, reported side by side:** Layer 2 only, Layer 2 + persistence,
  Layer 2 + persistence + Isolation Forest. If ML does not beat the classical baseline, that is a
  published result, not a failure to disclose.
- **Negative control:** an operational-variability run with no injected fault and no genuine
  abnormality, where the detector must stay silent. This is the test that kills over-eager
  detectors.

### 9.2 Honest reframing of two original sections

*"Tyre Behaviour Analysis"* as written is a classifier over known physical features. That is a lookup
table, not machine learning, and training it on simulator output is circular. Reframe as what it
actually is: the lumped thermal model of §7 with a learned correction term, evaluated on horizon-h
temperature prediction error. Physical, honest, still interesting.

*"Setup A vs Setup B"* deserves to be the real ML contribution, because a two-run comparison is a
demo and a surrogate model is not.

---

## 10. Definition of done

The project is done when all of the following are true, and the claim is backed by committed
artifacts rather than a demo:

1. `channels.yaml` exists and code-generates the packet layout, Parquet schema, and TypeScript types.
2. The physics core passes all 8 CI invariants and hits its §11 numeric targets.
3. A run at full sensor rate writes valid Parquet and replays **byte-identically** from disk.
4. A published detection table exists: latency, FPR/h, miss rate, localisation accuracy, for all 10
   fault types × all severities × 3 seeds, with the three baselines side by side and a negative
   control. No result is claimed that is not in that table.
5. A DoE over the setup vector has been run, a surrogate fitted, an optimum predicted, re-simulated,
   and the predicted-vs-actual error published.
6. `docs/calibration.md` accounts for every coefficient, with a citation or an explicit "synthesised"
   label.
7. Latency budgets are measured and published, not estimated.
8. A clean `git log` telling the story in order, and a README that states plainly which parts are
   simplified and why.

---

## 11. Validation

Non-negotiable. Without it "physics-based" is a claim, not a result.

**Numeric targets.** Fix these against published figures during calibration and record the source.
Treat the values below as order-of-magnitude sanity bounds to be confirmed, not as gospel:

| Quantity | Expectation | Source to confirm |
|---|---|---|
| 0–100 km/h | ~2.5–3.0 s | published acceleration figures for 2026 PU power/mass |
| Top speed | ~350–370 km/h | drag-limited solve, cross-checked vs published top speeds |
| Peak lateral g | ~4.5–5.5 g at high downforce | downforce from `car_spec`, grip from Pacejka `D` |
| Peak longitudinal decel | ~−5 to −6 g | tire-road μ, brake torque limit, weight transfer |
| 200–0 km/h braking distance | order ~4–6 s | derived, then checked against published braking data |
| Cornering balance | understeer gradient consistent with downforce split | own model, checked for monotonicity |
| Tire equilibrium temp | compound- and surface-dependent, plausible window | published tyre operating windows |
| Energy balance | residual <1% | CI invariant, §6 |

**Cross-check against real data.** OpenF1 serves historical `car_data` — speed, throttle, brake, rpm,
gear — at ~3.7 Hz from 2023 onward, free, no auth. Run a scenario, decimate to 3.7 Hz, compare traces
against a real session. This is the cheapest credibility win available and the original plan missed
it entirely. It will not match exactly, and it should not; what it validates is that throttle and brake
*timing and shape* are realistic, and that gear-shift points and rpm ranges are in the right place.

**Independent solver check.** Run the same `car_spec` through `fastest-lap` and compare lap times on
a known circuit (§4).

**CI invariants** — all in CI, all blocking:

1. no NaN or Inf, ever
2. friction ellipse respected: `(Fx/μFx)² + (Fy/μFy)² ≤ 1` for every wheel, every step
3. Σ vertical load = weight ± aero ± acceleration term
4. slip and force sign conventions consistent across all four corners
5. longitudinal/lateral symmetry at zero steer, zero camber, symmetric setup
6. energy conservation (§6), residual <1%
7. monotonic gearbox progression; no reverse engaged under positive throttle
8. determinism: two identical runs produce byte-identical Parquet

**Regression fixtures.** Golden Parquet files with expected traces. Any physics change re-runs them
and diffs. This is how you stay honest while tuning tire coefficients, and it is the only thing
standing between you and a silent regression that invalidates a month of analytics work.

---

## 12. Repository and tooling

```text
car_spec.yaml            # versioned to a FIA reg issue date
channels.yaml            # THE contract
scenarios/*.yaml         # OpenSCENARIO-shaped scenario library
tracks/*.yaml            # centerline, width, sectors
packages/
  physics/               # 10 kHz core · tire · aero · powertrain · thermal
  track/                 # centerline · min-curvature QP · speed profile · sectors
  sensors/               # per-rate sampling · quantisation · fault injection
  bus/                   # CAN-FD emulation · arbitration · cycle times
  link/                  # framing · Arrow IPC · Parquet writer · replay
  analytics/             # validity · residual · mspc · persistence · ml/
  scenarios/             # loader + runner, live and headless
  server/                # ingest · WS fan-out · REST over DuckDB
  web/                   # React dashboard
tests/
  physics/               # invariants + golden traces
  analytics/             # the §9.1 detection tables, as generated reports
docs/
  calibration.md         # every coefficient, provenance, citation
  detection.md           # generated from the analytics test suite
```

**Stack, locked:**

- **Physics core: Python + Numba.** One language for the whole project. See §4.1 for the decision and
  the rules it imposes.
- **Python 3.12** · `uv` for env and lockfile · `numpy` + `numba` for the core · `polars` + `pyarrow`
  for Parquet · `duckdb` for query · `scikit-learn` for Isolation Forest · `pytest` + `hypothesis` for
  tests · `ruff` for lint and format · `basedpyright` for types.
- **Web:** React 19 + TypeScript + Vite, a thin raw `ws` client, `zustand` for non-hot-path state,
  canvas + worker for rendering. No heavyweight charting library — the canvas path is a feature, not a
  shortcut.

**Pin `numba` and `llvmlite` explicitly** in `pyproject.toml`. The one dependency-resolution failure
found while choosing this stack came from a floating `numba` pulling an `llvmlite` that did not
support the Python version in use. Both are locked in `uv.lock` and neither is allowed to float.

CI: `ruff` → `basedpyright` → `pytest -m invariant` → `pytest -m golden` → `pytest -m analytics`
(detection tables regenerated and diffed against the committed baseline) → web build. Every stage
blocking.

---

## 13. Milestones

> **Task-level detail lives in [`PHASES.md`](./PHASES.md)** — per-phase task IDs, exit gates, demo
> points, and the cross-phase standing rules. This section is the summary.

Single developer, full-time. Estimates assume someone who has written vehicle dynamics code before;
if not, add ~40% to phases 1–2.

| # | Phase | Est. | Exit criteria |
|---|---|---:|---|
| 0 | Contracts, tooling, skeleton | 1 wk | `channels.yaml` code-generates packet layout, Parquet schema, TS types. CI green on all 8 invariants as *tests that exist*. Dashboard renders synthetic data at correct per-channel rates. |
| 1 | Physics: straight line | 2–3 wk | Aero + ICE + gearbox + longitudinal Pacejka. 0–100, top speed, power curve within agreed tolerance. Invariants 1, 3, 6, 7 pass. `fastest-lap` cross-check set up. |
| 2 | Physics: lateral + load transfer | 3–4 wk | Slip angle, combined slip, friction ellipse, 4-corner load transfer with camber gain, roll/pitch stiffness split. Constant-radius sweep matches target lateral g. Quasi-static suspension travel and tire pressure online. Invariant 2, 4, 5 pass. |
| 3 | Thermal (lumped) | 1 wk | Tire, brake, engine, gearbox nodes. Equilibrium temperatures plausible per compound. Pressure–temperature coupling live. |
| 4 | Track, racing line, timing | 2–3 wk | Min-curvature QP line on ≥2 circuits, forward-backward speed profile, sector timing, lap validity, pure-pursuit reference driver. |
| 5 | Sensors, CAN-FD, storage, replay | 2–3 wk | Per-rate decimation with anti-aliasing, calibrated noise floor, all 10 fault types, bus load <70% at full rate, Parquet round-trip, **byte-identical deterministic replay**. Invariant 8 passes. |
| 6 | Scenarios + headless | 1–2 wk | 8+ scenario types, live and headless modes, headless >50× real time. |
| 7 | Residual analytics | 3–4 wk | The full §9.1 table generated and committed. Persistence rule demonstrated to cut FPR without losing detection. Negative control silent. |
| 8 | Dashboard | 3–4 wk | All §8.4 panels, live + scrubbing, anomaly panel with real statistics, run comparison. Latency budgets measured and published. |
| — | **Core total** | **18–24 wk** | §10 items 1–4, 6, 7 met. |
| 9 | *Stretch* — ML upgrade | 2 wk | Isolation Forest beats Layer 2 + persistence, **or is honestly reported as not beating it**. Autoencoder only against a demonstrated failure. |
| 10 | *Stretch* — setup DoE + surrogate | 2–3 wk | Full sweep, surrogate, predicted optimum re-simulated, error published. |
| 11 | *Stretch* — multi-node thermal, manual-driving polish, NLP lap-time polish | open | Optional. |

Ordering rationale: emulation and storage land *before* analytics, so analytics is built against real
Parquet rather than an in-memory array. Phase 4 (track) lands before the DoE because the DoE is
worthless without it — this is the dependency v1 missed entirely.

---

## 14. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Physics scope explodes | High | Point-mass 6-DOF with 4 load-sensitive tires. No multi-body suspension solver. Enforce via §2. Phase gates make overrun visible at week 3, not month 6. |
| The model is wrong and everything downstream inherits the error | High | Validation gate per phase; refuse to advance on a missed target. `docs/calibration.md` with citations. Invariant 8 + golden traces catch regressions the moment they happen. |
| ML is circular — learns the simulator's own quirks | Medium | Residual-based detection; unsupervised training on clean runs; injected faults strictly held out. Report latency and FPR, never "confidence." |
| Ingest + analytics + WebSocket cannot keep up on one machine | Medium | Separate processes; headless batch; explicit measured budgets; decimate before the browser. |
| 2026+ regulation churn | Certain | Versioned `car_spec.yaml` keyed to a reg issue date. Zero hardcoded constants. Expect a spec bump each off-season; it is a data edit by design. |
| Racing-line/speed-profile work turns into a research project | Medium | §8.5 is scoped to min-curvature QP + forward-backward profile. NLP polish is explicitly optional and only for headline benchmarks. |
| Scope creep into "game" | Medium | §2 out-of-scope list reviewed on every PR. No rendering, no opponents, no race logic. |
| Golden-trace baseline churns on every coefficient tune | Medium | Commit baselines per phase, not per commit. Coeff changes are expected; flag trace diffs over a stated threshold for review rather than auto-failing. |

---

## 15. References

**Vehicle dynamics**
- Pacejka, *Tyre and Vehicle Dynamics*, 2nd ed. — Magic Formula, combined slip, relaxation length
- Milliken & Milliken, *Race Car Vehicle Dynamics* — load transfer, aero, race-car practice
- Tremlett & Limebeer, "Optimal tyre usage for a Formula One car," *VSD* 54(10), 2016 — canonical F1
  optimal-lap model; the published parameter set seeds `car_spec.yaml`
- Madsen, "Validation of a Single Contact Point Tire Model Based on the Transient Pacejka Model in
  Chrono," TR-2014-16 — relaxation-length implementation
- Heilmeier et al., "Racing Line Optimization" — min-curvature QP, and the ~4% gap between
  minimum-curvature and minimum-time lines
- Kapania et al., "A Sequential Two-Step Algorithm for Fast Generation of Vehicle Racing Trajectories"
  — forward-backward speed profile

**Regulations**
- FIA 2026 F1 Regulations, Section C (Technical), Issue 16, 2026-02-27
- FIA regulatory refinements, 2026-04-21 — energy management, superclip duration, Overtake Mode
- 2027/2028 technical changes agreed 2026-06-10 — anticipated spec drift
- ASAM OSI 3.8.0 and ASAM OpenSCENARIO 1.4 / 2.x — interface and scenario-design precedent

**Data and tooling**
- OpenF1 API — free historical real-world `car_data` at ~3.7 Hz for cross-validation
- `fastest-lap` (MIT) — independent vehicle-dynamics and optimal-lap reference for validation

**Analytics**
- Johnson & Jackson, *Multivariate Statistical Process Control* — T²/Q, the Layer 2 basis
- Kapteyn & Willcox, "From physics-based models to predictive digital twin via interpretable machine
  learning" — optimal trees for interpretable twin model selection
- "Digital Twin-Driven Intrusion Detection for Industrial SCADA" — physics-residual + ML hybrid beats
  either alone; reports the false-positive rates that motivate the persistence rule
- SPECTRA, "A Physics-informed Digital Twin for Real-time Structural Anomaly Inference" — residual
  features, T²-style indices, persistent decision rule
- EngineAD, "A Real-World Vehicle Engine Anomaly Detection Dataset" — classical vs deep baselines on
  vehicle telemetry; the basis for classical-first
- Castellani, Schmitt & Squartini, "Real-World Anomaly Detection by using Digital Twin Systems" —
  digital twin generating the normal-operating training set
- Greiter et al., "A Deep Convolutional Autoencoder for Assessment of Drive-Cycle Anomalies in
  Connected Vehicle Sensor Data" — and its finding that the convolution matters, which is why §9
  proposes TCN on residuals rather than an AE on raw channels
