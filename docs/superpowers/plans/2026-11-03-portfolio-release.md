# Telemetry Portfolio Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox syntax for tracking.

**Goal:** Complete a reproducible telemetry experiment, recording, replay and comparison workflow with measurable reliability and a portable demo.

**Architecture:** Extend the existing WebSocket, Parquet and React flows. Store immutable self-contained runs in a server-owned directory. Use bounded per-client delivery and a read-only looping replay for public demonstrations.

**Tech Stack:** Python 3.12, asyncio/websockets, PyArrow, NumPy/Numba, React/TypeScript/Zustand, Vite, Node built-ins.

**Spec:** ../specs/2026-11-03-portfolio-release-design.md

## Global Constraints

- Preserve pre-existing AGENTS.md and docs/latency.md edits; commit only owned files.
- Haiku workers own bounded files, never commit/stash/checkout or delegate.
- No new application dependencies, physics features, or speculative infrastructure.
- Preserve loopback defaults, validation, accessibility, timestamp/sample fidelity.
- Verify each milestone and commit with a plain human-written subject.
- No remote push, public publication, or license choice without an existing authorization.

## Review Focus

- Save interrupted before publication leaves no listed partial run or replaced artifact.
- Hostile run identifiers and static paths cannot escape server-owned directories.
- Slow clients cannot block healthy clients or lose events silently.
- Disconnect/restart/concurrent commands cannot overwrite an active shared run.
- Synthetic clocks never stand in for measured network latency or real-car fidelity.

### Task 1: Durable recorded runs

Files: src/f1telemetry/runs.py; src/f1telemetry/testing/scenario_lab.py; tests/test_runs.py.

Interfaces: RunStore(directory: Path), save(lab_run: LabRun) -> dict[str, object], list_runs() -> list[dict[str, object]], load(run_id: str) -> LabRun. LabRun captures provenance when the scenario/configuration is resolved. Saved result has run_id. Metadata includes original summary and snapshots. Reuse SampleRecord, serialise_frames/read_frames and RunManifest.

- [ ] Write and run failing tests for round-trip/restart, captured metadata, invalid identifiers, corrupt files, interrupted publication and no overwrites.
- [ ] Implement immutable atomic Parquet records and actual provenance capture without wall-clock values in deterministic payloads.
- [ ] Run focused tests, lint and type checks; review diff; commit the verified slice.

### Task 2: Bounded transport and saved-run integration

Files: src/f1telemetry/server.py; tests/test_server_control.py; tests/test_server_integration.py.

Interfaces: TelemetryServer(..., run_store: RunStore | None = None). scenario_run returns persisted scenario_result. runs_list -> runs_list_result {request_id, runs}; run_replay {request_id, run_id} -> scenario_result with current request_id. Reuse scenario_error for failures. No caller-supplied filesystem paths. Add health/readiness and metrics snapshot with bounded counters.

- [ ] Write failing real-socket tests for scenario commands, restart/list/replay, bad origin/path/commands, competing playback, shutdown; write a controlled stalled-send test.
- [ ] Isolate each client's sends using a bounded queue/task; disconnect rather than lose data on overflow; account and clean up sender tasks. Preserve detailed/summary encoding reuse.
- [ ] Save before success, list/load off the event loop, reject playback replacement and commands unsupported by the source, validate all requests.
- [ ] Verify tests/lint/types and commit.

### Task 3: Dashboard saved-run workflow

Files: web/src/telemetry/socket.ts; web/src/store/labStore.ts; web/src/components/TestControls.tsx; web/src/components/SavedRuns.tsx; web/src/App.tsx; web/src/styles/testlab.css; web/tests/telemetry.test.mjs.

Interfaces: requestSavedRuns() returns ScenarioResult[]; requestSavedReplay(runId) returns ScenarioResult. Extend existing request correlation and error/timeout handling. List, select and replay recorded results; reuse lastRuns/prevResult comparison, retaining current behavior.

- [ ] Add small regression tests for message correlation/disconnect and recorded-result comparisons.
- [ ] Add accessible refresh/select/replay UI using existing controls/styles, finite-value validation, clear empty/loading/error states. No redesign.
- [ ] Verify frontend tests and production build; review and commit.

### Task 4: Reproducible performance proof

Files: src/f1telemetry/benchmark.py; tests/test_benchmark.py; docs/performance.md; pyproject.toml (coordinator-owned entry points).

Interfaces: python -m f1telemetry.benchmark --duration SECONDS --clients N --output PATH. Machine-readable output includes revision/runtime, clocks, summary/detail mode, per-client frames/rate/bytes and latency percentiles, CPU and traced memory with explicit measurement scope. Add optional slow-client workload; no external load-test dependency.

- [ ] Write tests for percentile/report math and a brief real-socket run.
- [ ] Implement repeatable wall-clock delivery benchmark; never subtract simulation timestamps from epoch/perf_counter.
- [ ] Run representative workloads and publish commands/results/limitations; verify and commit.

### Task 5: Portable read-only demo and browser CI

Files: src/f1telemetry/demo.py; src/f1telemetry/server.py; Dockerfile; .dockerignore; compose.yaml; justfile; pyproject.toml; .github/workflows/ci.yml; web/tests/dashboard-workflow.mjs; tests/test_demo.py.

Interfaces: f1-demo serves built assets and a deterministic looping sample replay through the existing server. f1-serve stays loopback by default; public read-only exposure requires explicit origin configuration. Health checks, path confinement and disabled remote mutations are tested. A launch recipe builds locked environments/assets and starts the demo.

- [ ] Write failure-first tests for readonly mutation rejection, origin policy, static traversal, readiness, looping replay and demo generation.
- [ ] Implement a native HTTP/static seam using websockets process_request, sample generation and read-only replay. Add container setup with writable recording path only for local interactive mode.
- [ ] Add PR trigger and browser scenario/save/restart/replay coverage retaining the noise-rate matrix.
- [ ] Verify local launch and browser workflow; commit.

### Task 6: Documentation and release evidence

Files: README.md; PHASES.md; docs/architecture.md; docs/calibration.md; docs/case-study.md; docs/demo.md; docs/assets/*; docs/performance.md; docs/release-checklist.md.

- [ ] Replace stale feature claims, repair local links, explain sampling/physics/display rates and precomputed scenario playback.
- [ ] Record actual coefficient provenance/limitations from car_spec, existing open calibration gates and detection evaluation limits. No fabricated recovered historical evidence.
- [ ] Record screenshot and short demo; document supported startup, saved-run comparison, failure policy, deployment limits and licensing choices.
- [ ] Run lint, format, type checks, contract/codegen, frontend tests/build, full pytest, browser workflow and demo health checks. Review the complete change, fix findings, commit, and create a local verified release tag only when all required gates pass.

## Progress ledger

Initial state: dev ahead of origin/dev by six commits. User edits: AGENTS.md and docs/latency.md. Prior focused checks passed; a fresh fast baseline is running. Execution uses the user-requested Haiku workers, with orchestrator integration/review/commits. This file records milestone completion and commit evidence as work proceeds.

Baseline: `uv run --frozen pytest -m "not slow" -q` passed: 1261 tests, 123 deselected.

### Preflight

| Tasks | Shared boundary | Ruling |
|---|---|---|
| 1 | Store tests and implementation | Atomic immutable files; preserve serializer compatibility. |
| 2 | Server and real socket tests | Queue overflow disconnects; no silent sample/event loss. |
| 3 | UI and socket protocol | Reuse scenario_result/error correlation; list has its own response. |
| 4 | Benchmark and transport | Measure monotonic elapsed delivery against expected cadence. |
| 5 | Server and demo/CI | Read-only exposure is explicit; preserve local interactive defaults. |
| 6 | Evidence and implementation | Document verified current state and unresolved external choices. |
| 1, 2 | RunStore and LabRun | Task 1 settles store API before server integration. |
| 2, 3 | runs_list/run_replay messages | Exact envelope in Task 2 is binding for Task 3. |
| 2, 4 | TelemetryServer | Benchmark consumes stable server API after Task 2. |
| 2, 5 | server.py | Sequential edits; demo extends hardened transport. |
| 3, 5 | Browser behavior | Browser test uses real UI actions and saved run API. |
| 4, 6 | docs/performance.md | Publish measured output only after integrated transport is final. |

Ruling: execute the approved six-milestone scope without another plan approval round;
the user's explicit request to complete it supplies implementation authorization.
Work in place because no isolated-worktree preference was supplied, preserving all
unrelated edits. Haiku workers execute one implementation task at a time.

### Milestones

- `23193eb`: scope and plan recorded.
- `4af7d39`: Apache 2.0 selected by the user; code license and retained track notices.
- `3e6acc8`: architecture and calibration limits; provenance audit empty, 169 car-spec tests passed.
- Benchmark implementation has five passing focused tests, clean lint/types; representative results wait for final transport.
- Task 1 worker delivered 26 storage tests. Coordinator added failure-first regression cases for invalid timestamps and snapshot consistency, and fixed both at their owning boundaries. Combined focused run: 86 passed.
- Ruling: keep the strict legacy RunManifest unchanged. Scenario provenance records seed/revision as null when absent; inventing a seed or revision to fit the older manifest would make the evidence false. This is a task-plan deviation that preserves the design's honesty requirement.
- Independent Haiku storage review completed. Fixed invalid output filtering, directory creation error wrapping, non-finite JSON/frame loading, snapshot consistency and event retention. Retained explicit named corruption errors rather than silently skipping files; this keeps listing failures visible and the existing list API stable. Deployment will include contracts in the source/container layout.
- Storage/config regression: 251 passed; storage/live-event targeted suite: 48 passed; affected types clean. Original frame events now survive persistence and the rebased live timeline.
- `f1e82f6`: complete persisted-run slice committed.
- `4c121a7`: saved dashboard workflow committed after Haiku review/fixes and 13 Node regressions; production build passed.
- `e8bc60f`: bounded server delivery and saved commands committed; 23 real/control/logging tests passed. The implementation delegate's transport later timed out; it was cancelled and confirmed interrupted before coordinator integration. Its finite fixture and early-send corrections were on disk and verified.
- `cc4022d`: Vite/source-map-js security patch updates; locked npm install, audit (zero advisories), and production build passed.
- Integrated demo/public/static/reconnect scope: 31 focused tests passed, then additional public-origin and reconnect checks passed. Real Chrome scenario/save/restart/replay/compare passed; video captured through the same harness after shared preview recording failed.
- Owner selected local/deploy-ready hosting only, with no public publication.
- Active final checks: full pytest session 16827 (one failure seen; await full summary); final review delegated; Docker rebuild needed for patched frontend; preview tab tab_1_81ff45ec-e8f9-4edd-8139-5491ef5d4218 must be closed at finish. Browser-release artifact session 25440 and container rebuild 28000 recorded above; 28000 hit registry TLS timeout after the first image built.
