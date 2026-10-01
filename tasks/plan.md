# Phase 1 implementation plan: straight-line vehicle physics

## Overview

Implement the P1 milestone from `PHASES.md` on top of the merged P0 baseline: a deterministic, fixed-step longitudinal F1 car simulation that produces calibrated acceleration and top-speed traces. Keep the P1 scope to straight-line physics; lateral vehicle dynamics, thermal behavior, track/lap simulation, storage/replay, scenarios beyond the two P1 cases, and analytics remain later phases.

## Starting point

- Branch: `feat/phase-1`, based on `origin/dev` at `8aa930b` (the merged P0 baseline).
- P0 already supplies the YAML car/channel contracts, a Numba toolchain probe, synthetic telemetry, golden trace helpers, and CI checks. Extend those patterns rather than introducing parallel infrastructure.
- `PHASES.md` defines P1-T1 through P1-T11 and the exit gate. `PLAN.md` §4.1, §6, and §11 set kernel, drivetrain, and validation constraints.
- Use the latest FIA 2026 Section C technical-regulations issue available when implementation begins; the FIA listing currently identifies Issue 20, dated 2026-08-05. Record the issue and clause/page for every regulatory input in `car_spec.yaml` or `docs/calibration.md`. [FIA technical regulations](https://www.fia.com/regulation/category/110/technical-regulations)

## Decisions and constraints

- Keep the runtime core in Python + Numba `@njit(cache=True, fastmath=False)`, with NumPy arrays and caller-owned state/output buffers.
- Use a fixed 100 µs step and semi-implicit Euler for the initial longitudinal integrator. Do not add adaptive steps, parallel execution, or hot-loop allocation.
- Parse `car_spec.yaml` outside the kernel and pass numeric configuration into it. Do not duplicate tunable physical constants in code.
- Start with longitudinal state only. Add the vertical load contribution required by P1; defer lateral chassis states to P2.
- Treat torque-curve values that are not public as synthesized and label their source and assumptions in `docs/calibration.md`.
- Do not add a new dependency for the optional `fastest-lap` check. Its build failure must be reported and must not block P1.

## Task list

Detailed acceptance criteria, verification, dependencies, and likely files are in [`todo.md`](todo.md).

1. Confirm regulatory inputs and map P0 car-spec data into kernel-ready arrays.
2. Add deterministic fixed-step straight-line kernel and caller-owned buffers.
3. Add speed-dependent aero and longitudinal tire force.
4. Add ICE/MGU-K drive, gears, clutch, wheel rotation, and shift behavior.
5. Add P1 scenarios and calibrate acceleration/top speed with recorded sources.
6. Enforce energy/invariant checks and record the independent solver comparison.

## Checkpoints

- **Kernel checkpoint (after Task 2):** focused tests show fixed-step behavior, finite state, and byte-identical results across identical runs.
- **Drive checkpoint (after Task 4):** both drive paths (launch and gear shift) produce finite, physically signed traces; tests cover low-speed slip and shift/clutch edges.
- **P1 exit:** all exit-gate items in `PHASES.md` pass or have a written, evidence-based explanation. Run `just check` before closing the phase.

## Exit criteria

- 0–100 km/h and top speed meet tolerances agreed from source-backed published figures; record the chosen figures and citations before tuning.
- ICE power curve shape is plausible across the rev range; synthesized regions and assumptions are identified.
- Invariants 1 (finite values), 3 (vertical load accounting), 6 (energy residual below 1%), and 7 (gearbox progression) pass on real scenario runs.
- Identical initial conditions and configuration produce byte-identical output.
- `fastest-lap` comparison is recorded; target agreement is within a few percent, or the discrepancy and limits are explained. If its optional dependency cannot build, record that result without blocking P1.

## Risks and open questions

| Risk / question | Response |
|---|---|
| P0 car-spec inputs are provisional and the roadmap's cited FIA issue is old. | Confirm relevant values against the latest official FIA issue before freezing calibration inputs; record exact issue and clause/page. |
| Public ICE torque data is unavailable. | Keep the curve explicitly synthesized from documented constraints; do not present it as measured data. |
| Current 0–100 km/h and top-speed bounds are sanity ranges, not accepted tolerances. | Select and cite published references before calibration; then record numeric tolerances in `docs/calibration.md`. |
| Scope can drift into P2 tire/lateral dynamics. | Keep this milestone to longitudinal tire behavior and load; schedule lateral force, load transfer, and chassis states in P2. |
