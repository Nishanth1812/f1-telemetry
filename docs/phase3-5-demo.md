# Phases 3–5 combined checkpoint

**Status: open.** The implementation has useful subsystem work in place, but the combined
release gate is not closed. No release tag should be applied until the evidence below is
recorded and the combined exit criteria in the [unified plan](./superpowers/plans/2026-10-05-phases-3-5-unified-plan.md)
pass.

## Present in the workspace

- P3 thermal update functions, scenario wiring, and focused thermal tests exist. The heat
  inputs include patch slip work and brake torque times wheel speed. The thermal parameters
  and tyre gas state are illustrative; no operating-temperature source data has been matched
  and equilibrium bands have not been measured. See [calibration notes](./calibration.md).
- P4 track geometry, racing-line, lap and sector utilities exist, along with two fictional
  track fixtures. The retained Catalunya and Silverstone source data is documented separately
  under `tracks/source_data`; it is not claimed as the provenance of the fictional fixtures.
- P5 sensor effects, CAN-FD encoding and scheduling, Parquet I/O, a bounded live buffer,
  replay, and a record-to-Parquet bridge exist. The bridge persists fault annotations in
  Parquet metadata and can replay logical channel frames without rerunning physics.
- A targeted lap test run previously reported 9 passing tests. The telemetry bridge tests
  previously reported 11 passing tests. These are subsystem results, not evidence that the
  unified end-to-end gate passes.

## Evidence still required to close the gate

1. Measure and document plausible thermal equilibrium windows for each configured compound
   and surface, plus leak response and long-run finite-state evidence.
2. Run both track fixtures through the reference driver and scenario boundary; record clean
   lap, sector, deterministic timing, and deliberately invalid lap results. Compare the
   racing line visually against cited public imagery.
3. Verify per-channel declared sample rates and anti-alias behavior, all eligible fault modes,
   CAN-FD utilization and starvation constraints, and storage round trips for event channels.
4. Persist run validity and lap/event ground truth, compare two identical full scenarios for
   byte-identical Parquet, and compare replayed values, timestamps, and events with the source.
5. Connect replayed thermal, fault, and lap/sector information to the existing dashboard and
   retain a reproducible demonstration artifact.
6. Run the plan's final checks, update the P3/P4/P5 phase gates from their evidence, and only
   then decide whether a combined release tag is justified.

The current dashboard displays numeric channel values and traces. Fault annotations are
available in saved Parquet metadata, but a user-facing fault/event view and the integrated
thermal/lap demonstration have not been evidenced. The repository's checklists therefore
remain open.

## Verification boundary for this checkpoint

Further Python processes, simulations, and tests were not started for this checkpoint, per
the explicit workspace instruction. The process list was inspected and the outstanding
pytest processes from the delegated work were stopped. Static source and plan review does not
replace the run evidence above.
