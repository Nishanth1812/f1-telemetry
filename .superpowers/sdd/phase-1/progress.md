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

## Task 2 status

Done. `src/f1telemetry/kernels/longitudinal.py` (new) and `tests/test_longitudinal_kernel.py`
(new, 34 tests, `kernel` marker). Also touched: `src/f1telemetry/kernels/__init__.py` (the
package docstring claimed P0 held the only kernel), `tests/conftest.py` (Numba NRT counters, see
below), `pyproject.toml` (`kernel` marker, one more `TID251` carve-out).

### What was built, and what was deliberately left out

A straight-line point mass: state is `[distance_m, speed_m_s]` (`STATE_SIZE` 2,
`X_INDEX`/`V_INDEX`), integrated by semi-implicit Euler at the configured `dt_s`. No force is
modelled - **force is an input array**, one net longitudinal force per step. That is the whole
boundary Task 3 and Task 4 work through: Task 3 computes aero and tyre force, Task 4 computes
drive and brake force, and either way the integrator is unchanged.

The kernel takes `dt_s` and `mass_kg` as scalars and nothing else. Nothing about aero, tyre or
powertrain configuration reaches it yet, so the `KernelConfig` surface is minimal and grows when
the force models do.

### Interface frozen for Task 3

- `longitudinal.simulate(config, steps, state, force_n, out) -> out` is the **only** entry point.
  It reads `config.dt_s` and `config.mass_kg` **once**, in Python, outside the loop, per the
  Task 1 handoff - it does not rebuild the config per step. The compiled loop itself is
  `_integrate`, private: `boundscheck=False` makes an undersized buffer a silent out-of-bounds
  write, so there is exactly one way in and it validates.
- What `simulate` refuses before the loop starts: a nonfinite or nonpositive `dt_s` or
  `mass_kg`; a step count that is not an integer (`operator.index`, `bool` included in the
  refusal, so `4.0` cannot compile a second float64 specialisation); and any buffer that is not
  a C-contiguous `float64` `ndarray` of the right shape, with `out` additionally required to be
  writable. Read-only `state` and `force_n` are fine - the kernel never writes them. The force
  history is **not** scanned: that is Task 3's input to check, not Task 2's.
- Buffer shapes, all caller-owned `float64`: `state` `(2,)`, `force_n` `(steps,)`,
  `out` `(steps + 1, 2)` with row 0 seeded from `state` and row *n* the state after *n* steps.
- `longitudinal.allocate(steps)` and `initial_state(distance_m, speed_m_s)` are the whole
  allocation surface. There is no force-array helper: `np.full` is one line and a helper that
  only manufactures a constant force is a placeholder Task 3 makes obsolete.
- There is no public `step`. The integrator is the API; a one-step entry point would be a second
  way in and a second thing to keep in step with the scheme.
- Tests import the kernel only from `tests/test_longitudinal_kernel.py`, which carries the
  `TID251` per-file ignore. Anything else that needs the kernel needs its own carve-out in
  `pyproject.toml`, or the layer-isolation rule will refuse it.

### Two things Task 3 inherits

1. **The kernel's state layout is expected to grow.** Task 3 needs per-wheel slip and vertical
   load. Whether that means widening `STATE_SIZE` or adding a second state array is undecided;
   `STATE_SIZE` and the index constants are the thing that has to change when it does. `speed_m_s`
   is the `vx` of `PLAN.md` section 4's state vector, so the columns line up with
   `GroundTruthStep.vx_m_s` when a real record is built.
2. **The ERS curve-length hazard from Task 1 still stands** and is untouched here: `CONFIG`'s
   `ers_speed_km_h` (4 breakpoints) and `ers_overtake_speed_km_h` (2) need their own loop bounds.
   Nothing in the Task 2 kernel reads either, which is why it could not trip over it.

### Testing note for whoever runs this next

`tests/conftest.py` sets `NUMBA_NRT_STATS=1` above the project imports, which is the only point
at which numba reads it.
`test_the_step_loop_performs_no_allocation` uses Numba's NRT counters to show the step loop
allocates nothing, and it carries its own control - a kernel that does allocate in its loop, whose
count must grow - so the measurement cannot pass vacuously if numba's internals move. It also
asserts that the counters are on instead of skipping when they are not: a measurement that
silently stopped measuring is not a pass. Measured on this machine: 3 allocations per call for the
kernel at both 1,000 and 100,000 steps, against 1,000 and 100,000 for the control.

`cache=True` is proven by a **fresh interpreter** - a subprocess with `NUMBA_CACHE_DIR` pointed at
an empty pytest `tmp_path` - which loads `car_spec.yaml`, calls `simulate`, and has to leave numba's
`.nbi` cache index in that directory. The subprocess is the point, not the ceremony: the kernel
module and its dispatcher are already imported in the pytest process, so an in-process check can
only ever find an index an earlier run left behind, and it would keep passing if the decorator were
changed to `cache=False`. `NUMBA_CACHE_DIR` takes priority over the `__pycache__` beside the
source, so the child cannot load one of those either. `fastmath`/`nogil`/`boundscheck`/
`error_model` are read back off the dispatcher's own `targetoptions`. `TARGET_OPTIONS`, a literal
copy of the decorator text, is gone: it could only ever agree with the decorator, which is the
thing that needed checking.

TDD record: RED was a behavioural probe against the pre-Task-2 tree (the only compiled kernel was
the oscillator probe, which has no force input, decelerates a forward-moving car, and knows
nothing about the car's mass), plus a collection `ImportError` for the new module - the same
two-part record Task 1 used, because a missing module cannot fail behaviourally. GREEN:
`pytest tests/test_longitudinal_kernel.py` 34 passed; `pytest -q` 148 passed; `pytest -m invariant`
12 passed; `pytest -m golden` 7 passed; `ruff check`, `ruff format --check`, `basedpyright`
(0 errors / 0 warnings / 0 notes), `f1-check-contract`, `f1-codegen` and the web build all clean.
Those two counts are the numbers after the fix round below; the first Task 2 commit was 22 and 136,
and merging and splitting the tests is what moved them.

## Task 2 review — fix round

Reviewer: Space Bunny Alpha, model `openrouter/stealth/space-bunny-alpha`. One follow-up commit on
top of `dbbf767`; the fixes below are the review's eight findings and nothing else.

1. **The cache claim was self-referential.** The old test asserted `TARGET_OPTIONS["cache"] is
   True` against a dict literal that duplicated the decorator text, so it could not fail if
   caching broke. It now compiles the kernel, requires a `.nbi` cache index to appear in a
   directory that was empty beforehand, and reads `fastmath`/`nopython`/`nogil`/`boundscheck`/
   `error_model` off `dispatcher.targetoptions`. `TARGET_OPTIONS` is deleted rather than kept as a
   second copy of the same six values. The index search this round started with looked beside the
   source, which is satisfied by an index an earlier run left there - the second follow-up below
   is about that, and about the "falsified by changing the decorator" claim that came with it.
2. **The compiled loop was public.** `integrate` was exported, callable directly, and compiled with
   `boundscheck=False`, so an undersized `out` wrote past its end without a word. It is now
   `_integrate`, out of `__all__`, and the module docstring says why the private name is the safety
   property rather than a style choice.
3. **Zero mass and a NaN step produced NaNs silently.** `simulate` now refuses a nonfinite or
   nonpositive `config.dt_s` or `config.mass_kg` before Numba sees them. `KernelConfig` is loader
   validated, but it is a public frozen dataclass and `dataclasses.replace` is the same path a
   YAML edit takes, so the kernel boundary checks the two scalars it actually uses. The force
   history is left alone: scanning it is Task 3's job.
4. **A buffer check could itself crash or lie.** `_check_buffer` takes `object` and raises
   `ValueError` naming the buffer when it is not an `ndarray`, requires C-contiguity for every
   buffer (the contract already said so), and requires `out` to be writable. Read-only `state`
   and `force_n` stay legal - the kernel only reads them - and a test says so.
5. **The duplicate `step` is gone.** It was a second public entry point implementing a scheme
   `integrate` already implements, which is two things to keep in step rather than one.
6. **`constant_longitudinal_force` is gone.** It existed only to give the tests a force array and
   called itself a temporary placeholder; the tests use `np.full` and the docstring says why.
7. **`NUMBA_NRT_STATS` was set after the imports it had to precede**, and the allocation test
   *skipped* when the counters were off - the one case where the measurement is worthless was the
   case that passed silently. The flag is now set above the project imports in `conftest.py`, and
   the test asserts on `numba.core.config.NRT_STATS` (read with `getattr`: numba writes these
   flags into its module globals at import, so there is no static attribute to read).
8. **A float step count reached Numba**, which would have compiled a second float64
   specialisation of the integrator and quietly accepted `4.0`. The boundary now goes through
   `operator.index`, refuses `bool` as well as non-integers, and returns a real `int`.

Test count went 22 -> 34, and the file got smaller in what it claims: the duplicate byte-identity
and buffer-reuse tests are one test now, the driven-run increment/finite/analytic checks are one
test, and `representative run is finite` no longer restates what the reference test proves. What
replaced them is the new boundary coverage and the falsifiable cache proof.

Final verification: `pytest tests/test_longitudinal_kernel.py` 34 passed (from a cold `__pycache__`,
so the cache test proves a real compile), `pytest -q` 148 passed, `pytest -m invariant` 12 passed,
`pytest -m golden` 7 passed, `ruff check` and `ruff format --check` clean, `basedpyright`
0 errors / 0 warnings / 0 notes, `f1-check-contract` clean, `f1-codegen` 0 files changed.

## Task 2 review — second follow-up

One finding, and it is about the fix round's own cache proof rather than about the kernel.

**The `.nbi` search it introduced could be satisfied by history.** It looked in `__pycache__`
beside the module and, if `NUMBA_CACHE_DIR` was set, there too - and `__pycache__` holds
`longitudinal._integrate-68.py312.nbi` after any run, which is every run after the first. Change
the decorator to `cache=False` with that file already on disk and the test still passes, so the
"falsified by changing the decorator to `cache=False`" sentence in finding 1 above only held from
a cold directory. It was not a false claim when written; it was a claim about a starting state the
suite does not guarantee.

`tests/test_longitudinal_kernel.py` proves `cache=True` where history cannot reach it. A fresh
interpreter subprocess with `NUMBA_CACHE_DIR` set to pytest's empty `tmp_path` imports the module,
loads `car_spec.yaml` through `load_car_spec()`, calls `simulate`, and then the index has to be in
that directory - searched recursively, because numba nests it under a subdirectory derived from the
source location. `NUMBA_CACHE_DIR` takes priority over the `__pycache__` beside the source, so the
child cannot quietly load an index compiled earlier either. The child prints the speed it reached,
so the parent can see the run happened rather than read silence as success. A subprocess is
required, not tidy: the pytest process has already imported the module and compiled the dispatcher,
so there is no cold compile left in it to observe.

Re-measured: with the decorator set to `cache=False` the test fails and the temp directory is
empty; at `cache=True` it passes. The compiled-signature half of that test stays in-process, since
a signature list is per-process and this one is honest there. The helper that searched both roots
is gone with the `Path` import it needed; `os` stays for the subprocess environment.

Not fixed here, and worth a note rather than a change: `probe.cache_index_path()` in P0 has the
same staleness hole, so `tests/test_numba_toolchain.py::test_kernel_writes_a_compile_cache` can
pass from a warm `__pycache__` too. P0 is closed and this round is scoped to the Task 2 kernel, so
it is left as it stands.

Verification for this follow-up: `pytest tests/test_longitudinal_kernel.py` 34 passed,
`pytest -q` 148 passed, `pytest -m invariant` 12 passed, `ruff check` and `ruff format --check` over
the tree clean, `basedpyright` 0 errors / 0 warnings / 0 notes.

## Task 3 status

Done. Commit: `1f54838`. `src/f1telemetry/physics/__init__.py` and
`src/f1telemetry/physics/forces.py` (new), `tests/test_forces.py` (new, 36 tests, `forces`
marker), plus `car_spec.yaml` (`tyres.longitudinal_pacejka`, `tyres.slip_ratio_min_speed_m_s`),
`KernelConfig` and its builder in `src/f1telemetry/contracts/car_spec.py`, `docs/calibration.md`,
and `pyproject.toml` (the `forces` marker and one more `TID251` carve-out). The Task 2 kernel is
untouched.

### What was built

Five `njit` primitives and one Python composition. `dynamic_pressure_pa` (`q = 1/2 rho v^2`),
`speed_curve` (the `Cl`/`Cd` lookup), `aero_forces` (`(downforce, drag)`, drag signed along +x),
`slip_ratio` (`kappa = (omega r - v) / max(v, eps)`) and `tyre_longitudinal_force` (the
longitudinal Magic Formula with the load guard). `step_forces(config, speed, wheel_speed, load)`
composes the three models for one step, returns `(downforce_n, drag_n, tyre_fx_n)`, and is the one
place the configuration and the state are checked before any arithmetic.

New configuration: `pacejka_b/c/e/mu` and `slip_ratio_min_speed_m_s` on `KernelConfig`, all
range-checked in `build_kernel_config` — the three magnitudes and the slip guard positive, the
curvature factor finite but signed, because `E` carries a sign. `car_spec.yaml` labels all five
synthesised; the audit passes unchanged.

### Rulings

1. **Interface: elemental primitives plus one composition, no axle aggregation.** Task 3 needed
   aero, slip and a longitudinal force, and nothing else. Deciding which wheels get drive torque
   and which get brake torque is Task 4's, and splitting downforce between the axles is P2-T2's,
   so `step_forces` takes one `load_n` and returns one `tyre_fx_n`. A four-corner force vector
   would have been a decision this task does not own.
2. **`STATE_SIZE` did not grow, against the Task 2 handoff's expectation.** The handoff said the
   state layout was expected to grow because Task 3 needs per-wheel slip and vertical load. It
   does not: both are *inputs* to a force model, supplied per step by the caller, not state. The
   state that produces wheel speed (`omega`) is Task 4's wheel rotational state, so Task 4 is
   where `STATE_SIZE` and the index constants change if they change. The 34 Task 2 tests are
   untouched and still green, which is the evidence that nothing about the integrator moved.
3. **Interpolation: piecewise-linear, clamped at both ends.** Task 1 froze one shared speed axis
   and left the interpolation undecided. Linear because six knots labelled "order of magnitude"
   are a table, not a fit, and because it is the arithmetic `_interpolate` in
   `tests/test_car_spec.py` already uses, so the kernel and the contract tests interpolate
   identically rather than approximately. Clamped above 105 m/s because the table stops there and
   extending the last segment would make a fast car's drag a function of a two-point line.
4. **The grip limit is measured, not clamped.** `|Fx| <= mu Fz` is the Magic Formula's own bound
   (`|sin| <= 1`), so a clamp would be a branch that can never fire. The test sweeps slip and
   requires the peak to *reach* `mu Fz`, so a formula returning a constant fraction of the limit
   cannot pass it either.
5. **`Fz <= 0` returns exactly `0.0`**, negative loads included. A negative load makes
   `D = mu Fz` negative and the whole expression invert, which would push the car the wrong way
   from a tyre carrying nothing.
6. **The aero coefficients are read at `abs(v)`.** A car rolling backwards at 80 m/s meets the
   same air at the same dynamic pressure as one rolling forwards; reading the curve at -80 m/s
   would take the below-range end value. This is what makes downforce even in speed and drag odd,
   as `GroundTruthStep` requires.
7. **The `Cl`/`Cd` curves were *not* replaced with a published parameter set.** Task 1's
   `car_spec.yaml` note and this file's section 3 both said Task 3 would substitute the
   Limebeer and Tremlett open F1 model. It does not, and does not pretend to: this project has
   not read that parameter set, and a number it cannot cite is worth less than one it labels
   (PLAN.md section 4). Both documents now say the curves are consumed as synthesised data and
   that replacing them is open work. The Pacejka coefficients are synthesised on the same
   principle.
8. **No load sensitivity, asserted as linear.** `D = mu Fz` is what P1 uses; `PLAN.md` section 4
   says a constant-`mu` tyre understates high-speed downforce, and load sensitivity on `D` and
   `B` is P2-T3's. It is stated in the docstring *and* asserted in the tests, so P2-T3's change
   is a visible edit rather than a silent correction.
9. **`cache=True` is not asserted for the new dispatchers.** Numba consumes it at decoration, so
   it is not on `targetoptions`. The fresh-interpreter proof in
   `tests/test_longitudinal_kernel.py` covers the convention these functions follow; duplicating
   that ceremony for five dispatchers was not worth the test time. The option set that *is*
   readable is asserted on every primitive.

### TDD record

RED, two parts, as Task 1 and Task 2 needed. The new module does not exist, so
`pytest tests/test_forces.py` cannot collect: `ModuleNotFoundError: No module named
'f1telemetry.physics'`. A behavioural probe against the pre-Task-3 tree showed the gap as
behaviour rather than as a missing file: `KernelConfig` carries the aero arrays but raises
`AttributeError` for `pacejka_b`, `pacejka_c`, `pacejka_e`, `pacejka_mu` and
`slip_ratio_min_speed_m_s`; the `tyres` block of `car_spec.yaml` has no tyre model coefficients
at all; and the only simulation entry point in the repository is a 2-state point mass over a
caller-supplied `force_n` array, so a caller had no way to obtain a force from a car spec.

GREEN: `pytest tests/test_forces.py` 36 passed; `pytest -q` 192 passed (148 at Task 2, plus 36
force tests and 8 new car-spec cases); `pytest -m invariant` 12 passed; `pytest -m golden` 7
passed; `pytest -m forces` 36 passed; `ruff check` and `ruff format --check` clean over the tree;
`basedpyright` 0 errors / 0 warnings / 0 notes; `f1-check-contract` clean; `f1-codegen` 0 files
changed; web production build passing.

Two test-expectation bugs surfaced as red during the green phase and are worth recording, because
both were the *test* being wrong about the physics rather than the physics being wrong:
`aero_forces` originally read the curve at signed speed, which gave a car rolling backwards at
80 m/s the drag of a stationary car, and the slip-ratio test expected `omega r / eps` below the
guard when only the denominator is guarded, so the correct value is `(omega r - v) / eps`.

### What Task 4 inherits

1. **The seam is unchanged.** `longitudinal.simulate(config, steps, state, force_n, out)` still
   takes the net longitudinal force per step as an input array, and the integrator is untouched.
   Task 4 computes that array; Task 3 deliberately does not, because which wheels receive drive
   torque and which receive brake torque is Task 4's decision.
2. **Read the configuration once, outside the loop, and call the primitives.** The kernel should
   take `air_density_kg_m3`, `reference_area_m2`, `aero_speed_m_s`, `cl`, `cd`, `pacejka_*` and
   `slip_ratio_min_speed_m_s` from `KernelConfig` before the step loop and pass plain numbers to
   `aero_forces`, `slip_ratio` and `tyre_longitudinal_force` — the same rule
   `longitudinal.simulate` already follows for `dt_s` and `mass_kg`. `step_forces` is the
   Python-facing composition and re-checks the config on every call; it is not a hot-path call.
3. **A `TID251` carve-out will be needed.** `f1telemetry.physics` is on ruff's banned-api list, so
   a kernel module that imports `f1telemetry.physics.forces` needs its own per-file ignore in
   `pyproject.toml`, as the three test files now have.
4. **The state layout is still Task 4's to change.** If wheel rotation becomes state, `STATE_SIZE`
   and the index constants are what move; nothing in Task 3 depends on their current values.
5. **The launch sits on the falling branch of the tyre curve.** With `eps = 1.0 m/s` a standing
   start gives `kappa = omega r / 1.0`, which is far past the 0.36 peak, so a launch transmits
   about 78 % of `mu Fz` rather than all of it. That is a consequence of the guard, recorded in
   `car_spec.yaml` and `docs/calibration.md`, and it belongs to whoever models the launch.
6. **Downforce has to be added to a load by the caller.** `step_forces` takes the vertical load
   the patch carries; a straight-line run passes `static + downforce_n`. The front/rear split is
   P2-T2's, so Task 4 chooses one and records it — invariant 3 only checks the sum.
7. **A drag-limited top speed with the committed `Cd` and 750 kW lands above the PLAN section 11
   band.** Rough arithmetic on the committed curves, not a measured run — there is no scenario yet
   to measure it with — so it is a flag for Task 5's calibration, not a finding. Expect `Cd` or
   the power split to move.

## Task 4 partial milestone: P1-T4 ICE torque curve

Commit `5ce583f` adds the configured, piecewise-linear ICE torque lookup and the turbo-lag
multiplier. `step_ice_torque` validates replaceable `KernelConfig` scalars and arrays before
calling Numba; the compiled primitives remain flat and unvalidated for the kernel path. The
boundary review caught malformed curves reaching a `boundscheck=False` lookup and negative
full-load torque, with regressions for malformed arrays, scalar normalization, negative torque,
and the lag threshold.

**Ruling:** Keep the lag as a hard step: 0.35 strictly below 4,000 rpm and 1.0 at and above. The
configured data has no recovery endpoint and P1-T4 asks for a multiplier collapsing below that
threshold; a smooth recovery would require a new calibration input. If a recovery ramp is later
required, add its endpoint to `car_spec.yaml` and the contract before changing the model.

The calibration guide and physics package description now match that behavior. Final verification:
`pytest -q` (213 passed), Ruff check/format, basedpyright (0 errors/warnings/notes), and
`f1-check-contract` pass. Task 4 remains open for P1-T5 through P1-T7.

## Task 4 partial milestone: P1-T5 gearbox and clutch

P1-T5 adds a compiled gearbox step in `src/f1telemetry/physics/gearbox.py`. Its caller-owned
`float64` state is `[gear, shift_remaining_s, clutch_engagement]`: the step advances gear and shift
timer, reads the caller's clutch engagement without overwriting it, and applies the shift boost cut
to a local engagement value. It uses the configured shift points and returns differential-side
torque after applying the selected gear ratio and final drive, then the clutch capacity ceiling.
Wheel-speed coupling and force assembly remain with P1-T6/T7.

The clutch capacity is synthetic and explicitly differential-side. `car_spec.yaml` sets 3000 Nm:
the committed curve and ratios offer 3844 Nm in first gear and 1618 Nm in eighth, so gears 1–3
clamp at full engagement while gears 4–8 pass through. The value is a provisional model input for
later launch calibration, not a measured or regulated figure. Zero shift duration is rejected so
the timer continues to prevent repeated shifts at a held threshold.

Verification: `pytest -q` (253 passed), Ruff check/format, basedpyright (0 errors/warnings/notes),
and `f1-check-contract` pass. Task 4 remains open for P1-T6/T7 and MGU-K drive integration.

## Task 3 review follow-up

Space Bunny Alpha's review found that replaceable `KernelConfig` arrays and scalars could create
extra Numba specializations or feed invalid curves to the lookup, and that the aero provenance
notes still promised a replacement Task 3 did not make. Commit `c47ba59` validates these inputs,
tests integer normalization and malformed arrays, and corrects the notes. The follow-up added two
force tests; `pytest` passes (194 total), Ruff and basedpyright are clean, contract/codegen checks
pass, and the web build succeeds.
