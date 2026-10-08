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
- Run wall time: 5.278 / 5.245 / 5.273 s — **median 5.27 s**.
- **Real-time factor: ~1.3×** (7.01 s simulated per 5.27 s of wall time).
- **~75 µs per kernel step**, all-in per step (kernel + drivetrain control interval pacing,
  one `step_gearbox` / MGU-K step per control interval, buffer writes, record building).

Note: the 10 kHz kernel step itself is a fraction of that 75 µs; the scenario runner
(drivetrain step, energy bookkeeping, record assembly) shares the run. A bare-loop
measurement of `longitudinal._integrate` alone was not taken in this pass — marked
**not-measured**.

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

## What is not yet measured

- Bare `_integrate` kernel-step time (without scenario-runner overhead).
- WebSocket end-to-end latency (server broadcast → decoded frame in the browser).
- Browser render/frame time under a live stream (needs a browser harness; out of scope for
  this document).
- Live-mode ingest lag at contract rate (`f1-serve` → store) under the synthetic source.
