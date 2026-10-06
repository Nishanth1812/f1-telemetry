# Phases 3–5 combined verification record

**Implementation and software checks: complete. Combined acceptance gate: open.** Do not apply a
P3–P5 release tag until the remaining evidence below and the relevant dependency/phase gates in the
[all-pending-phases plan](./superpowers/plans/2026-10-06-all-pending-phases-plan.md) pass.

## Implemented

- P3 thermal state updates are wired into scenarios. Tyre heat uses contact-patch slip work;
  brake heat uses brake torque and wheel speed. Thermal node, heat-share, and tyre-gas inputs
  come from the validated `thermal` section of `car_spec.yaml`.
- P4 track geometry, minimum-curvature line, speed profile, reference driver, lap/sector
  assessment, and session-channel publication are implemented. The two checked-in fixtures are
  explicitly fictional. Catalunya and Silverstone source data is retained separately under
  `tracks/source_data`; it is not the provenance for those fixtures.
- P5 contract-driven sensor processing, deterministic fault annotations, CAN-FD scheduling,
  Parquet persistence/replay, the bounded live buffer, WebSocket replay, and dashboard event
  display are implemented. The replay adapter publishes saved fault annotations as frame events;
  lap/sector session channels use the same record, Parquet, replay, and dashboard path.

## VM verification

All Python tests, simulations, and builds for this work ran on `dev4.heapvue.cloud`.

- `/home/hcs/.local/bin/uv run --frozen ruff check src tests` — passed.
- `/home/hcs/.local/bin/uv run --frozen ruff format --check src tests` — passed.
- `/home/hcs/.local/bin/uv run --frozen basedpyright` — passed with 0 errors and 0 warnings.
- `/home/hcs/.local/bin/uv run --frozen f1-check-contract` — generated contract artifacts are current.
- `/home/hcs/.local/bin/uv run --frozen pytest` — **1,179 passed in 733.67 seconds**. Golden traces were refreshed
  once with `/home/hcs/.local/bin/uv run --frozen pytest --golden-update`, then verified by this
  normal full-suite run.
- Follow-up P5 gate check, `/home/hcs/.local/bin/uv run --frozen pytest tests/test_telemetry_storage.py` — **27 passed**,
  including a high-frequency alias rejection check and same-seed output equality for every fault
  type.
- Web production build — passed using the VM's installed Node/npm toolchain.
- Three parallel stress scenarios each ran 450,000 steps (45 simulated seconds), 1.35 million
  steps total. All completed and their trace/load outputs were finite.

The VM did not have `just`, so these commands were run as the equivalent `justfile` recipes.
No test or simulation was run in the local workspace.

## Acceptance evidence still required

1. **P3 calibration:** Thermal inputs remain illustrative. No operating-temperature data has been
   matched to each configured compound and surface, and no sourced equilibrium bands have been
   measured. The thermal tests and stress runs establish finite behavior, not calibration.
2. **P4 real-circuit validation:** The active fixtures remain fictional. Both have not been run
   end-to-end with the reference driver for clean and deliberately invalid lap evidence, and no
   visual comparison against cited real-circuit racing-line imagery has been recorded. The
   retained source CSVs are geometry inputs, not evidence of this comparison.
3. **P5 dashboard demonstration:** The application builds and the software tests cover sensor,
   fault, bus, storage, and replay paths. A browser demonstration showing live and replayed
   thermal, fault ground truth, and lap/sector deltas has not been visually verified or retained.
4. **Upstream calibration:** P1/P2 calibration remains open in `PHASES.md` and
   `docs/calibration.md`; their limitations must remain explicit in downstream results.

These are explicit project acceptance criteria, not test failures. The passing VM suite does not
replace calibration or the real-track and dashboard evidence, so the combined gate remains open.
