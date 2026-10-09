# Latency budgets

`PLAN.md` §10 item 7 requires latency budgets that are measured, not estimated. Every number
in this document was measured on the development machine whose identity is in the method
sections; nothing here is an estimate. Where no measurement exists yet, the section says so
explicitly.

Machine and toolchain for all runs below: Intel Core Ultra 9 185H (22 logical cores),
Windows 11 (10.0.26200), Python 3.12.11, Numba 0.67.0, numpy 2.5.3, pyarrow 25.0.1,
Node v24.11.1.

## Physics step time — `accelerate_to_speed`, 7 s run

**Measured.** Method: throwaway local script (not committed). Build the scenario suite from
`car_spec.yaml` via `KernelConfig`, run the `accelerate_to_speed` scenario
(`0.5 s` clutch-slip launch segment + `6.5 s` full-throttle segment = 7.0 simulated seconds)
with `scenarios.run_scenario`, time the call with `time.perf_counter`, warm-up run first to
absorb Numba JIT compile, 3 timed repetitions, median reported.

- Simulated span: 7.010 s, 701 record frames (10 ms record interval), 70,000 kernel steps
  at 10 kHz.
- Run wall time: **~25 s** (orchestrator re-measurement on a quiet machine: 25.1 s warmed;
  profiler pass median 24.78 s over 3 runs). An earlier 5.27 s figure from the first pass
  could not be reproduced and is withdrawn — it was likely measured under incomparable
  conditions, so this document no longer cites it.
- **Real-time factor: ~0.28×** (7.01 s simulated per ~25 s of wall time).
- **~355 µs per kernel step**, all-in per step (kernel + drivetrain control interval pacing,
  one `step_gearbox` / MGU-K step per control interval, buffer writes, record building).

Note: the 10 kHz kernel step itself is a fraction of that ~355 µs (the profiler pass
measured the bare `longitudinal.simulate` loop at 3.84 µs/step, 26× real time); the scenario
runner (drivetrain step, energy bookkeeping, record assembly, thermal) owns the rest — see
the headless profile section below.

## Parquet serialise time

**Measured.** Method: same session; serialise the 7 s `accelerate_to_speed` record with
`testing.parquet_io.serialise_frames` (the `METADATA` header, Parquet + zstd), 5 repetitions,
median.

- Record: 701 frames, contract channels at their declared rates.
- Serialise wall time: 53.4 / 16.5 / 21.2 / 17.2 / 17.0 ms — **median ~17 ms**.
- Output size: **~316 kB** per 7 s run (~45 MB/h projected at contract rate).

## WebSocket ingest path

Path under test (shipped source, not a copy): server JSON frame
(`server.py::_broadcast`, `json.dumps(..., allow_nan=False)`) →
`web/src/telemetry/socket.ts` (`JSON.parse`, `decodeFrame`) →
`store/telemetryStore.ts::applyFrame` (throttled tile snapshot, raw counters, events) →
`telemetry/history.ts::TraceHistory.push` (per-channel ring buffers) →
`TraceHistory.fill` (min-max decimation) called once per animation frame from
`TracePanel.tsx`.

**Measured** (throwaway Node script, same data-URL loader pattern as `web/tests/telemetry.test.mjs`,
Node v24.11.1):

- `JSON.parse` + `decodeFrame`: 2.99 ms for 100 wire frames / 700 channel samples
  (a 7 s run's worth of a 100 Hz channel, batched at 30 Hz).
- `store.applyFrame` + `TraceHistory.push`: 3.42 ms for those same 700 samples,
  i.e. ~205,000 samples/s of ingest headroom.
- `TraceHistory.fill` per call: ~0.045 ms, returning 1000 decimated points from a 6100-sample
  ring — cheap enough to run every rAF tick at 60 fps.

Not measured in this pass: client↔server end-to-end latency over the loopback WebSocket
(a server-plus-client harness), and browser-side frame times. Both marked
**not-measured**; notes only:

- Every UI thread hop is synchronous; the ingest counters keep raw values outside React state
  and only the 250 ms throttled diagnostics snapshot reaches selectors.
- `TracePanel` caps the draw buffers at the canvas pixel width and decimates in `fill`, so the
  per-frame cost is bounded by pixels, not by channel rate.

## Render frame notes (from the web tests)

`web/tests/telemetry.test.mjs` (Node test runner, no framework) covers the ingest/history
semantics — batch decode, chronological min-max decimation, per-channel retention at 60 s,
rewind clearing — but it asserts **correctness, not frame time**. There is no render
timing in the test suite: render-frame budget is **not-measured** in this pass. From the
code: `TracePanel.tsx` draws on `requestAnimationFrame`, keeps React state out of the hot
path, reuses draw buffers unless the pixel width changes, and bounds points-per-frame by
the plot width via `TraceHistory.fill`. The `~0.045 ms` fill measurement above is the only
measured per-frame cost component.

## Headless runner profile — `accelerate_to_speed`, 7 s run, kernel-only variant

**Measured on this machine.** Method: throwaway local scripts (not committed). Measure the
bare kernel loop via `longitudinal.simulate()` in isolation, disconnected from record building
and Parquet writing, using the same `accelerate_to_speed` scenario and pre-calculated torque
history. The kernel receives only state and torque arrays; it performs no I/O, record assembly
or thermal simulation. Warm JIT with one full run, then 3 timed repetitions with
`time.perf_counter` for both kernel-only and full runner, median reported.

**Kernel-only profiling:**
- **Kernel-only wall time: 0.269 s** (median over 3 runs, 70,000 steps).
- **Kernel step rate: 3.84 µs/step** (70,000 steps / 0.269 s).
- **Real-time factor (kernel only): 26.05×** (7.0 s simulated per 0.269 s wall time).

**Full runner (scenario bookkeeping, drivetrain, record building, thermal):**
- **Full runner wall time: 24.78 s** (median over 3 runs, same scenario).
- **Full runner step rate: 354.01 µs/step** average across 70,000 steps.
- **Real-time factor (full): 0.28×** (7.0 s simulated takes 24.78 s wall time).
- **Runner overhead: 98.9%** of total time (24.51 s overhead + 0.27 s kernel).

Per-control-interval cost (100 kernel steps = 10 ms simulated time, producing 1 recorded frame):
- Kernel (100 steps): ~0.38 ms wall time.
- Runner overhead (drivetrain step, record frame assembly, energy residual, thermal): ~3.50 ms.
- Ratio: **9.2× runner overhead to 1× kernel per interval**.

Parquet serialisation (reported above): ~17 ms median for 701 frames. At full runner scale,
Parquet is ~0.1% of total run time and adds negligibly to headless throughput.

**For headless operation seeking 50× real time:**

Current throughput is 0.28× real time, requiring ~180× speedup to reach 50×. The kernel is
not the bottleneck; the 98.9% overhead consists of per-control-interval drivetrain stepping
and per-recorded-frame assembly (GroundTruthStep construction, discrete energy residual
calculation, thermal simulation). Of the three candidates — Parquet batching, pacing removal,
control-interval sizing — **control-interval sizing** offers the highest leverage within the
stated options: increasing `CONTROL_STEPS` from 100 to 1000 kernel steps per interval
(stretching control intervals from 10 ms to 100 ms simulated time) would reduce drivetrain and
record operations by 10×, potentially reaching ~2.8× real time. This trades faster headless
throughput for coarser torque-control responsiveness, which is acceptable for off-line analysis
but would degrade live-control fidelity. Reaching 50× would require control intervals of ~5000
steps (500 ms simulated per control interval), beyond practical limits for closed-loop control.

## Headless runner — `testing.headless.run_headless`

**Measured on this machine.** Method: throwaway local script (not committed), plus the committed
`tests/test_headless.py`, which measures the same thing. Both paths get two warm-up runs first (to
absorb Numba JIT compile) and then 9 timed repetitions with `time.perf_counter`; minimum, median and
maximum are all reported because the machine was not idle and the spread is wide. Same scenario
objects from `build_scenarios`, same `KernelConfig`, no `control_law`.

**`accelerate_to_speed`, 7.0 s sim, 70,000 kernel steps:**

| Path | min | median | max | µs/step (min) | real-time factor (min) |
|---|---|---|---|---|---|
| `scenarios.run_scenario` | 13.92 s | 19.16 s | 28.26 s | 198.9 | 0.50× |
| `testing.headless.run_headless` | 2.18 s | 3.48 s | 6.38 s | 31.2 | 3.21× |

- **Measured speedup: 6.4× (min/min), 5.5× (median/median).**
- Reference wall time here (13.9–28.3 s) brackets the ~25 s quoted above, so the machine's load
  accounts for the spread rather than the two numbers describing different work.

**`steady_state_circle`, 1.0 s sim, 10,000 kernel steps (steering path):**

| Path | min | median | max | µs/step (min) | real-time factor (min) |
|---|---|---|---|---|---|
| `scenarios.run_scenario` | 1.29 s | 1.85 s | 3.08 s | 128.7 | 0.78× |
| `testing.headless.run_headless` | 0.25 s | 0.30 s | 0.36 s | 25.0 | 4.00× |

- **Measured speedup: 5.2× (min/min), 6.1× (median/median).** The shorter run shows the same factor
  with a much tighter spread, which is the best evidence that the speedup is the schedule and not a
  lucky sample.

**Where the time went, and what it bought.** A `cProfile` pass over `run_scenario` on the same
scenario attributes its overhead to per-kernel-step *validation*, not to arithmetic:
`step_gearbox` (6.8 s cumulative), `step_mgu_k` (4.3 s) and `step_ice_torque` (3.4 s) were 70,000
calls each, and each re-checked the config's finiteness, the ratio table, the ICE torque curve and
the MGU-K deployment curves before handing already-validated numbers to a compiled step.
`step_ice_torque` alone was called 70,700 times to produce a value that is constant across a
control interval.

`run_headless` keeps the kernel, the drivetrain steps and the record assembler exactly as they
are, and moves the re-derivation of their inputs from per step to per segment: one compiled call
per control interval drives the same `gearbox._step_gearbox` / `powertrain._step_mgu_k` primitives,
and each segment's inputs are validated once through the public entry points on scratch state
copies before any run step is taken. No coefficient, `dt_s`, `CONTROL_STEPS` or arithmetic order
changed. `tests/test_headless.py` asserts `trace.tobytes()`, every drivetrain column, every step
output and the whole assembled `SampleRecord` are byte-identical to `run_scenario` on both
scenarios, so this is a scheduling change and not a physics one.

**The P6 >50× gate is not met, and this document does not claim it is.** The honest measured factor
is roughly **5–6×**, from 0.50× to ~3.2× real time; reaching 50× would still need another ~16×.
What remains after this change is not the stepping loop. A profile of the headless path puts
`_energy_residual_fraction` first (1.21 s of a 4.16 s profiled run, one Python loop over 100 kernel
steps per recorded frame) and thermal next (`_celsius_to_kelvin` and its validation helpers,
~0.7 s each) — the record post-pass, which this path deliberately reuses unchanged so that the
record it emits is byte-identical to the reference runner's. The kernel itself is ~0.5 s of it.
Options that would move the number further, none of them taken here:

- **Compile the record post-pass.** `_energy_residual_fraction` and the thermal trace are the
  remaining Python loops and are the largest single target left. Making them compiled would keep
  the arithmetic but would be a second implementation of an invariant's assembly, which is exactly
  the risk the byte-identity assertion above is there to prevent.
- **Make the record optional on this path.** A headless run that only needs the trace could skip
  the post-pass entirely, for roughly the post-pass's share of the remaining time. That changes
  `ScenarioRun`'s contract, so it is a design decision rather than an optimisation.
- **Coarsen the control interval.** Unchanged here on purpose: the byte-identity claim and the
  100 Hz declared control rate both depend on `CONTROL_STEPS` staying at 100, and a run that
  matched no reference trace would not be worth the throughput.

## What is not yet measured

- WebSocket end-to-end latency (server broadcast → decoded frame in the browser).
- Browser render/frame time under a live stream (needs a browser harness; out of scope for
  this document).
- Live-mode ingest lag at contract rate (`f1-serve` → store) under the synthetic source.
