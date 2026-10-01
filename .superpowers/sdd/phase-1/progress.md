# SDD ledger — plan: tasks/plan.md

Branch base: `8aa930b` (`origin/dev`), current task base: `0db4c30`.
Workspace: `.superpowers/sdd/phase-1/`.

## Preflight interface scan

| Tasks | Shared input/output | Result |
|---|---|---|
| 1 → 2 | Task 1 prepares validated numeric configuration from `car_spec.yaml`; Task 2 consumes it outside the flat kernel. | Consistent; freeze names/shapes in parser tests before Task 2. |
| 1 → 3 | Task 1 records aero and tire config inputs; Task 3 consumes them in force calculations. | Consistent; all coefficients remain data-driven. |
| 1 → 4 | Task 1 records powertrain/gearing/clutch inputs; Task 4 consumes them. | Consistent; no engine implementation in Task 1. |
| 1 → 5 | Task 1 values feed scenario/calibration runs. | Consistent; numeric target selection remains Task 5. |
| 2 → 3 | Task 2 establishes straight-line state/config boundary; Task 3 adds aero and tire forces. | Consistent; keep function inputs flat numeric and caller-owned. |
| 2 → 4 | Task 2 state/buffer boundary feeds drivetrain integration. | Consistent; drivetrain can extend the established state layout. |
| 3 → 4 | Task 3 force calculations and Task 4 drivetrain both feed longitudinal acceleration. | Consistent; Task 4 consumes force API from Task 3. |
| 3 → 5 | Task 3 forces feed scenario traces. | Consistent. |
| 4 → 5 | Task 4 powertrain/shift behavior feeds P1 scenarios. | Consistent. |
| 5 → 6 | Task 5 calibrated traces are inputs to invariant checks and independent comparison. | Consistent. |

## Per-task consistency scan

| Task | Scope, tests, and files agree? | Note |
|---|---|---|
| 1 | Yes | Parser/contract tests and source/calibration records; no kernel implementation. |
| 2 | Yes | Kernel plus focused determinism tests. |
| 3 | Yes | Force behavior plus focused low-speed/load tests. |
| 4 | Yes | Drivetrain behavior plus focused launch/shift tests. |
| 5 | Yes | Scenario traces, goldens, and calibration record. |
| 6 | Yes | Existing invariant/testing path, CI gate only if required, documented solver result. |

## Task 1 status

Done after two fix rounds. Report: `reports/task-1.md`; round-2 review: `reports/task-1-review-round-2.md`. Commits: `a063b1e`, `b9fc915`, `a360e1f`, `c9bf718`, `782bdb9`.

Interface frozen for Task 2. `CarSpec.kernel_config()` returns `KernelConfig`: flat scalars
plus contiguous writable `float64` arrays, every one range-checked in Python. Shape notes for
the kernel:

- `dt_s`, `mass_kg`, `gravity_m_s2`, `air_density_kg_m3`, `reference_area_m2`,
  `ride_height_sensitivity`
- `aero_speed_m_s` / `cl` / `cd` - same length **and the same breakpoints, enforced at load**,
  so `Cl(v)` and `Cd(v)` are indexed by one shared speed axis. Interpolation remains undecided,
  which is deliberate
- `torque_rpm` / `torque_nm` - same length, rpm strictly increasing
- `gear_ratios` - length 8, strictly decreasing; `final_drive`
- `shift_up_rpm` / `shift_down_rpm` / `shift_time_s`
- `rolling_radius_m`, `wheel_diameter_m`, `front_width_mm`, `rear_width_mm`, `wheelbase_m`,
  `overall_width_m`, `front_weight_fraction` (checked to be in (0, 1); the C4.2 axle floors are
  deliberately absent - they are fractions of a regulatory minimum mass that includes the
  unknown Nominal Tyre Mass, not of `mass.total_kg`. See `docs/calibration.md` §2.)
- powertrain: `ice_peak_power_kw`, `rev_limit_rpm`, `idle_rpm`, `turbo_lag_collapse_rpm`,
  `turbo_lag_multiplier`, `fuel_energy_flow_*` (the C5.2.3/5.2.4 limits, which are the real
  constraint on the torque curve), `mgu_k_peak_power_kw`, `mgu_k_torque_limit_nm`,
  `ers_speed_km_h` / `ers_limit_kw` / `ers_overtake_speed_km_h` / `ers_overtake_limit_kw`,
  `store_energy_mj`, `recharge_limit_mj_per_lap`, `superclip_s`, `launch_speed_kmh`,
  `power_split_ice`, `fuel_lhv_kj_kg`

Two corrections Task 2-5 inherit: the MGU-K store is **4 MJ** (C5.2.9), not the 7 MJ the plan
carried; and available ERS-K power is the **effective** limit, `min(C5.2.8, C5.2.7)` — 350 kW
below 290 km/h (337.5 km/h in Overtake), 100 kW at 340 km/h, zero at 345. C5.2.8's own formulas
permit 1800 kW at rest, so a kernel reading them raw would get five times the permitted power.
Both corrections are enforced at the boundary, so a kernel cannot silently use the old numbers.

Kernel-side hazard the loader cannot cover: `CONFIG.ers_speed_km_h` and
`CONFIG.ers_overtake_speed_km_h` have different lengths (4 vs 3 breakpoints). A kernel that
reads either must use the matching array length, not a shared loop bound.

`CarSpec.kernel_config()` rebuilds the arrays from the typed fields on every access, via
`build_kernel_config()` — the single validation path. It is deliberately **not** cached: a config
built at load and stored on the spec desynchronised under `dataclasses.replace`, and a spec built
or replaced outside the loader skipped validation entirely. Task 2 should call `kernel_config()`
once at the Python boundary, outside the Numba loop, rather than per step.

## Task 1 review — fix round 1

Reviewer: Space Bunny Alpha, model `openrouter/stealth/space-bunny-alpha`.
Verdict: Needs fixes. Direction approved; six Important findings block completion. Review report: `reports/task-1-review.md`.

### Important findings to fix

1. `kernel_config()` uses one shared aero speed array without checking `Cl` and `Cd` curve breakpoints are identical. Add a failing test and a Python-boundary `ContractError` on mismatched grids.
2. C5.2.8 deployment data labels the full curve as regulated although the 290 km/h crossing is a derived breakpoint. Make that derived point machine-visible and test the label, without inventing a new generic provenance framework unless needed.
3. `docs/calibration.md` overstates coverage: its test does not pin every clause in the table. Correct the claim to the actual 23 citation entries covered.
4. TDD evidence should show a behavioral red failure, not only a test-module `ImportError`. Make the new API test collectible against P0 and record the focused behavioral failure/output in the report.
5. `kernel_config()` re-reads/revalidates already validated CarSpec values from `raw`, creating two sources of truth. Reuse the typed CarSpec fields for existing validated inputs; parse only the additional P1 kernel inputs from raw.
6. Validate `front_weight_fraction` at the boundary, or remove it from the P1 config if not used. Do not assume C4.2's minimum-axle percentages directly equal fractions of simulated total mass; document the relationship and any unavailable nominal-tyre-mass input.

### Minor findings deferred per review workflow

These are not part of the fix round: duplicate curve parsing; dead fallback curves; `KernelConfig` generated equality/hash with ndarray fields; nonexistent `integration.air_temperature_k` fallback; minimum-mass and front/rear rim field naming precision; stale P0 wording; duplicated regulation numbers across data/tests/docs; misleading/duplicated test name/assertions; missing final newlines and stray trailing blank line; inconsistent ISA density/temperature claims; incomplete missing-section parametrization; untracked progress handoff not visible in the review diff.

Reviewer could not independently confirm every regulatory quote from the diff. I checked the FIA official listing earlier, but full clause-to-page re-verification remains a separate review check before Task 1 is closed.

### Fix round 1 outcome

All six Important findings addressed in commit `b9fc915`. `tests/test_car_spec.py` references
`KernelConfig` through the module so the file **collects against the P0 base** (with the P1
`contract` marker carried over — see fix round 2), making the RED evidence behavioural rather
than an `ImportError`.

1. `Cl`/`Cd` speed grids must be identical, checked at load next to the `_positive_values` calls.
2. `derived_points` inside a `regulation` claim, with `CarSpec.derived_points()`. Only the
   deployment curve has one: the 290 km/h corner, whose *limit* is C5.2.7's 350 kW and whose
   *speed* is derived. Audited: a derived point must be in the curve and must state why.
3. `docs/calibration.md` §1 separates the 23 clauses read from the 23 citation entries asserted,
   and names which of the table's rows are read-but-not-cited.
4. Behavioural RED recorded; see report.
5. `CarSpec.build_kernel_config()` is the single validation path, called from `load_car_spec`.
   Typed `CarSpec` fields are read as fields; only P1-only inputs are parsed from raw.
6. `front_weight_fraction` is checked to be in (0, 1). C4.2's floors are **not** enforced
   against it - they are fractions of the C4.1 Minimum Mass, which includes the unknown Nominal
   Tyre Mass (C4.7), so comparing against `mass.total_kg` would enforce a rule that does not
   exist. `car_spec.yaml` records `chassis.c42_enforcement` with `status: not_enforced`.

Ruling recorded: finding #6 as written ("range-check `front_weight_fraction` against the C4.2
floors or drop it") has no valid first option. Range-checking is done; comparison against the
floors is not, and the reviewer's framing assumed the floors were fractions of the same mass
they are not.


## Task 1 review — fix round 2

Reviewer: Space Bunny Alpha, model `openrouter/stealth/space-bunny-alpha`.
Commits: `a360e1f` (cap + builder), `c9bf718` (docs), `782bdb9` (quotes and wording).

- **C5.2.7's 350 kW absolute ERS-K cap is applied to both C5.2.8 propulsion profiles.** Verified
  against the Issue 20 PDF at `spec.document_url`: C5.2.8 states the *propulsion* limit only
  (`1800 - 5v` below 340 km/h), so on its own it permits 1800 kW at rest. The effective limit is
  `min(C5.2.8, C5.2.7)`, now stored as `(0,350),(290,350),(340,100),(345,0)` and
  `(0,350),(337.5,350),(355,0)`, with both crossover speeds declared in `derived_points`.
- **The cap is enforced at the shared boundary**, not merely correct in today's file: a curve
  point above 350 kW raises `ContractError` from `build_kernel_config`, for a YAML edit and for
  a curve carried on a replaced spec. Interpolated output is checked across both curves.
- **Removed the cached kernel config.** `kernel_config()` builds on access, so
  `dataclasses.replace(spec, mass_kg=900)` cannot leave a stale array behind, and a spec built
  outside the loader is validated too. `rolling_radius_m` is checked in the shared builder.
- **Audit reports, never raises, for a malformed derived-point speed**, so one typo does not take
  the audit down and hide the other 22 citations.
- **Docs no longer overstate coverage.** The clause table has 23 rows, not 30; C5.12.2 is page 73
  and C5.12.3 page 74; C10.7.2 does not say "18 inch" (that is an inference, recorded under
  `tyres.inference`); the C10.7.2 and C5.2.10 quotes reproduce the document exactly, including
  FIA's own "diaerence" typo, with column attribution in a `quote_note`; C5.2.5's 380 MJ/h arm
  and C5.2.8.iii/.iv are named as read-but-out-of-scope.
- **C4.2 clarified.** It is a Qualifying-only check, and its denominator is the C4.1 Minimum
  Mass *including* the Nominal Tyre Mass — a figure the tyre supplier publishes before the
  Championship (C4.7), so obtainable rather than unavailable. The floors stay cited in the file,
  out of `KernelConfig`, with `chassis.c42_enforcement` naming the missing input.

### Corrected TDD evidence

The RED numbers recorded in round 1 (46 failed / 9 passed) were for the round-1 test file. Re-run
against the P0 base at `0db4c30` with the **current** test file, the actual result is
**55 failed, 8 passed** — 63 tests. The failure modes are behavioural: `AttributeError: 'CarSpec'
object has no attribute 'kernel_config' / 'citations' / 'derived_points'`, and `KeyError` where a
test edits a `regulation` block or curve the P0 file does not have.

This regression run needs one piece of P1 configuration: the `contract` pytest marker was
registered in `pyproject.toml` in `a063b1e`, and `--strict-config` rejects an unregistered
marker, so the marker line was carried into the temporary P0 worktree for collection. It does not
collect against an untouched P0 checkout.

Final verification: `pytest -q` (114 passed), `pytest -m invariant` (12 passed), `pytest -m
golden` (7 passed), `ruff check` and `ruff format --check` clean, `basedpyright` 0 errors /
0 warnings / 0 notes, `f1-codegen` 0 files changed, web production build passing.
