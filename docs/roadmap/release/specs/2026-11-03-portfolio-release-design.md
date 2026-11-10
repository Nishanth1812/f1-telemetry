# Telemetry portfolio release

The goal is a complete software/backend portfolio project: run a bounded experiment,
save it, restart, replay it, and compare its result, with repeatable evidence about
correctness and transport performance. The vehicle remains an uncalibrated simulation.

## Decisions

Extend the existing Python WebSocket server, React dashboard, Parquet serializer,
and run manifest. A directory of immutable Parquet files is sufficient; no database,
message broker, separate API framework, new charting library, or physics expansion.
The user approved all six assessment milestones, Haiku workers, and periodic commits.
Work in the current dev checkout and preserve pre-existing edits to AGENTS.md and
docs/latency.md. Publishing or a remote push is not part of local implementation.

## Required behavior

- Save every successful dashboard scenario before returning success. Save original
  relative sample times and values, the scenario summary, request settings, seed,
  configuration snapshots/hashes, code revision and dirty-state indication. Reuse
  RunManifest and deterministic Parquet. Use atomic file publication, opaque IDs,
  server-owned paths, and explicit errors; never overwrite a previous run.
- List saved runs and replay a selected run after a restart through the same live
  frame format. Reuse the existing same-scenario comparison UI when loading results.
  Corrupt files and invalid IDs fail clearly without crashing the server.
- Only one computation/playback controls the shared car at a time. Reject attempts
  to replace active playback. Viewers may join and leave independently.
- A stalled client cannot hold up the shared pump. Bound pending output and disconnect
  lagging clients rather than silently dropping full-rate samples or fault events.
  Preserve command validation, loopback defaults, origin protection and clean shutdown.
- Prove commands, persistence/restart/replay, invalid requests, competing clients,
  slow viewers and lifecycle using real sockets plus focused deterministic tests.
- Report readiness and bounded operational counters. Benchmark real WebSocket delivery
  with monotonic clocks, client count, percentiles, payload volume, CPU/memory where
  measurable, machine and revision. Distinguish paced replay from simulation cost.
- Provide one-command local startup after locked dependency installation, sample-run
  generation, a container launch setup, and a read-only looping replay deployment
  mode. Any non-loopback exposure requires explicit read-only configuration and
  permitted browser origins. No public filesystem access or remote simulation control.
- Extend browser CI beyond noise to a real scenario and saved replay. Run CI on PRs.
  Keep existing contract-rate and drift gates. Verify the full suite before release.
- Update README, architecture/rate explanation, calibration limitations, performance
  report, case study, license/attribution guidance, screenshot and short demo recording.
  Repair stale links and distinguish historical plans from verified current behavior.

## Acceptance and limits

A fresh documented setup can run the demo; a scenario survives server restart;
replayed original samples and summaries match; the existing comparison works with
saved results; a stalled viewer does not block a healthy one; memory/output queues
are bounded; benchmark and browser checks reproduce. No invented performance,
calibration, detection, or production-readiness claims. External hosting credentials,
domain choice, legal license selection and publication remain user decisions.

## Approaches considered

Extending the existing server and Parquet path keeps ownership clear and makes the
current tested mechanisms demonstrable. Adding a REST/database/service stack would
duplicate those mechanisms. Pursuing more vehicle fidelity or ML would not close
the software/backend demonstration and reliability gaps.
