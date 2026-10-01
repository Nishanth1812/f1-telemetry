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

Done. Report: `reports/task-1.md`.

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
carried, and available ERS-K power is speed-limited, so it drops to 100 kW at 340 km/h and
zero at 345 km/h. Both are enforced by the loader, so a kernel cannot silently use the old
numbers.

Kernel-side hazard the loader cannot cover: `CONFIG.ers_speed_km_h` and
`CONFIG.ers_overtake_speed_km_h` have different lengths (4 vs 2 breakpoints). A kernel that
reads either must use the matching array length, not a shared loop bound.

`KernelConfig` is built once at load, not on first access, so a bad `car_spec.yaml` edit fails
at `load_car_spec` rather than at first use. Task 2 should build its config eagerly, not lazily.

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

All six Important findings addressed in commit `b9fc915`. `tests/test_car_spec.py` now
**collects against the P0 base**, so the RED evidence is behavioural rather than an
`ImportError`; recorded in `reports/task-1.md` §6.

1. `Cl`/`Cd` speed grids must be identical, checked at load next to the `_positive_values` calls.
2. `derived_points` inside a `regulation` claim, with `CarSpec.derived_points()`. Only the
   deployment curve has one: the 290 km/h corner, whose *limit* is C5.2.7's 350 kW and whose
   *speed* is derived. Audited: a derived point must be in the curve and must state why.
3. `docs/calibration.md` §1 now separates the 30 clauses read from the 23 citation entries
   asserted, and names which of the table's rows are unchecked.
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
