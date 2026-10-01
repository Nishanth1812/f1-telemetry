# Phase 0 decisions

The choices PHASES.md left open, what was decided, and why. Anything here that later turns
out wrong gets a new line rather than a silent edit, so the record stays auditable.

Scope of this pass: P0-T1 through P0-T8, plus the contract and codegen halves of P0-T2 to
P0-T6. P0-T9 through P0-T11 (web scaffold, synthetic frame generator, dashboard) are not
covered here.

---

## 1. Event-rate encoding

`PLAN.md` §5.2 lists the Session group as `event` and every other group as a number. A
string in a numeric field would leak into the generated Python, TypeScript and Parquet
schema, so rate is encoded as a pair instead:

```yaml
- name: lap_index
  group: session
  unit: count
  event: true
  rate_hz: null
```

* `event: true` and `rate_hz: null` together mean edge-triggered. The loader rejects both
  halves of the mismatch: a non-event channel with a null rate, and an event channel with
  a number.
* Python: `rate_hz: float | None`, `event: bool`, `period_s -> float | None`.
  TypeScript: `rateHz: number | null`, `event: boolean`. Parquet: `rate_hz: float | None`.
* Sizing excludes event channels rather than guessing a rate for them. They are excluded
  because inventing a nominal rate for them would be exactly the kind of hardcoded number
  the project forbids, and the real event count is a property of a run, not of the
  contract.

Consequence: nothing downstream may schedule a session channel on a timer. P5's decimator
has to special-case them, and the generated `Contract.event_channels()` exists so it does
not have to re-derive the list.

## 2. Corners: the source of truth is one `corners` list per channel

`PLAN.md` §5.2's worked example declares `width: 4` and `corners: [FL, FR, RL, RR]` on one
entry. That is kept: a per-corner signal is declared **once**, the generator expands it,
and there is no hand-written `wheel_speed_fl` anywhere.

* Canonical corner order is `[FL, FR, RL, RR]`, declared in the contract's `conventions`
  block and enforced by the loader. The loader also normalises input to that order and
  rejects duplicates or unknown corners, so declaration order in the file cannot change
  generated output.
* Expanded name is `<base_name>_<corner lowercased>`. Nine base signals are per-corner
  (five in `wheel`, four in `wheel_thermal`), giving 36 corner channels and 79 channels in
  total.
* Expansion is a loader function, so `CHANNELS` in the generated registry is already
  expanded and every consumer - Parquet schema, TypeScript - sees the same 79
  names. `Contract.corner_map(base_name)` is the lookup a four-corner consumer wants.
* `swap` fault eligibility requires a non-empty `corners` list. A channel with nothing to
  swap with cannot be swapped, and the loader says so instead of silently allowing a fault
  that can never be injected.
* The `width: 4` key from the `PLAN.md` example is **not** implemented. It duplicates
  `len(corners)`, and two fields that must agree are one field too many. `corners` is the
  only declaration.

## 3. YAML spelling

British spellings throughout, because the project documents already use them
(`quantise`, `Final`, `synthesised`, `behaviour`): the contract key is `quantise`, the
Python attribute is `quantisation`, the TypeScript field is `quantise`, the Parquet
column metadata is `quantisation`, and the generator's own argument is
`quantisation`. The fault taxonomy keeps `PLAN.md`'s exact list: `dropout, freeze, spike,
step, gain, noise, quantise, swap, saturate, stale`.

Channel names are `snake_case` and are the stable identifier everywhere - YAML key, Python
attribute, dataclass field, TypeScript key, Parquet column. `PLAN.md` §5.2's
`wheel.speed` is flattened to `wheel_speed`; a dotted name is a display convention and
cannot be a Python identifier or a TypeScript key without a translation step in five
places, which is the thing this project exists to avoid.

Two channel names are prefixed where ambiguity would otherwise be real: `imu_accel_x` and
friends, because the chassis group already publishes filtered `accel_lateral` and
`accel_longitudinal` and the sensor has to be identifiable for the localisation step in
`PLAN.md` §9.5; and `wheel_speed`, because `speed` in the chassis group is a different
sensor.

Where `PLAN.md` §5.1 names a channel explicitly (`aero_mode`, `fw_flap_deg`,
`rw_flap_deg`, `zone_id`, `ice_rpm`, `mgu_k_rpm`, `tyre_temp`, `tyre_pressure`) that
spelling is used verbatim. Where §5.1 says `Cl·A` and `Cd·A` must be derived channels,
they are declared as `cl_a` and `cd_a` in the aero group. `drag_n` is **not** added: §5.2's
aero row does not list it, and drag is already recoverable from `cd_a` and the speed.

Units are declared once, in `unit`, and never encoded in the name - the one exception being
`PLAN.md`'s own `eso_pct` and `*_flap_deg`, which are kept as written because a
documented name beats a tidy rule.

## 4. CI OS

`ubuntu-latest` for every job. The development machine is Windows, and `P0-T1b` proves
the Numba toolchain works there, but CI is not the place to pay for a second
platform-specific matrix: the generated artifacts are OS-independent by construction
(deterministic iteration order, no timestamps, explicit `\n` newlines, and `.gitattributes`
pinning LF) and the physics has no platform-dependent behaviour to catch.

The risk this accepts: a Windows-only failure surfaces on a developer machine rather than
in CI. Mitigation is that P1's physics runs locally ten times more often than it runs in
CI, and both platforms are the same Numba/LLVM code path. If a genuine
platform-specific bug appears, the matrix is a five-line addition to
`.github/workflows/ci.yml` and this line gets rewritten.

## 5. No analytics tests yet

The `analytics` marker is registered in `pyproject.toml`, but CI does not run it yet and
there are no analytics tests in Phase 0. `PLAN.md` §12 requires that stage, and
`PLAN.md` §9.1's detection table is a P7 deliverable.

The web stage is already a real build: CI installs the locked dependencies with `npm ci`
and runs `npm run build --prefix web`.

## 6. Provisional `car_spec` provenance

`car_spec.yaml` is now versioned to FIA 2026 Section C **Issue 20 (2026-08-05)**, the current
issue of the official listing. P0 shipped it against Issue 16 (2026-02-27) as
`PHASES.md` P0-T3 requires; P1-T1 re-verified it against Issue 20 and moved the pin. Values
that were `plan`, `synthesised` or `provisional` in P0 remain so in spirit, but the claim
mechanism is stronger - see section 8.

* `spec.provenance: provisional`
* `spec.calibration_status: uncalibrated`
* `spec.source_date: 2026-09-30` - the date the P0 placeholder was written, not a
  measurement date, and stated as such in the file
* `spec.verified_against_document: true` and `spec.verified_on` - the P0 flag
  `confirmed_against_regulation_text` was replaced by these in P1-T1, because "confirmed"
  was ambiguous: P1-T1 confirmed the *regulation claims*, and said nothing about the
  uncalibrated numbers. The new pair names one document, one date and one scope.

`source_date` on every section records the same thing, and a section whose provenance is
`plan`, `provisional`, `synthesised` or `mixed` without a `source_date` is an audit failure
that `provenance_audit()` reports and a test asserts. That is the machine-checkable form of
the cross-phase rule "every coefficient gets a provenance line". A nested block that declares
its own provenance line - `powertrain.ice` and `powertrain.mgu_k` - is audited the same way
and reported at its dotted path, because a block making its own claim owes the same dating.

`PLAN.md` §5's own figures are used verbatim where they exist (800 kg, 400 kW ICE,
350 kW MGU-K, 53/47 split, 3.4 m wheelbase, 1.9 m floor width) and marked
`provenance: plan` with a pointer to the section. Everything else is
`provenance: synthesised` and says so: the Cl/Cd curves, the ICE torque table and the
turbo-lag multiplier, the gear ratios, the final drive, the rolling radius, the front
weight fraction. No number in the file pretends to have been read out of the regulation
text, and the file says so at the top.

`test_car_spec_provenance_is_complete` fails if `calibration_status` ever claims anything
other than `draft` or `uncalibrated`, so the file cannot quietly start asserting that it
is calibrated before P1-T9 has earned that.

### Gearbox, because `PLAN.md` §6 does not add up

`PLAN.md` §6 asks for "realistic ratio spacing (≈1.6 apart, geometric)". A geometric step
of 1.6 across eight gears is a 27× span, and no combination of a sane 1st gear and a sane
final drive reaches 350-370 km/h with it - the arithmetic in the `gearbox.derivation` field
of `car_spec.yaml` shows this. The ratios are therefore a geometric set chosen so that 8th
gear at the 13000 rpm limiter lands inside the `PLAN.md` §11 top-speed band and 1st gear is
in the right place for a launch. The step is not stored: the `ratios` list is the data, and
a `ratio_step` field alongside it would be a second copy of it to drift out of agreement.
P1 replaces these provisional explicit ratios during calibration.

`test_gearbox_lands_in_the_plan_top_speed_band` asserts both numbers, so the set cannot be
edited into a contradiction without a test failing. P1 replaces the whole block with the
drag-limited solve and the published Limebeer & Tremlett parameter set.

## 7. Other choices worth recording

**`numpy`, `PyYAML`, `pyarrow` are dependencies; `polars` is not.** `PLAN.md` §12 lists
`polars` and `pyarrow` for Parquet. Only `pyarrow` is declared in P0, because the P0
storage path is a single `pq.write_table` call and one schema. `polars` arrives with P5
when there is a query to answer; declaring it now means a locked version nobody imports.

**Sizing is computed, and it disagrees with `PLAN.md`.** The contract as written carries
7160 channel-samples/s (28.6 kB/s, 103 MB/h raw at float32), not the ~4600/s that §5.2
estimates. §5.2's figure is an estimate of a channel list it does not enumerate; the
contract enumerates it, so the contract wins and the number is derived
(`CHANNELS` → `SAMPLES_PER_SECOND`), with a test asserting the total is above the estimate
rather than hardcoding either number in two places.

**No CAN-FD layout in P0.** `PHASES.md` P0-T5 asks for "Parquet schema (per group) and
TypeScript types", and the CAN-FD frame encode/decode is P5-T5. An earlier pass of this
document described a generated `can_layout.py` - arbitration ids from an identifier
base and stride in a `can` block of `channels.yaml`, a 16-byte header, columnar payloads
and a bus-load percentage against a hardcoded 1 MB/s capacity. All of it is gone, and so
is the bus capacity constant, which was a magic number of exactly the kind the
cross-phase rules forbid: CAN-FD does not have one, and the P5 exit gate's "bus load
<70%" is meaningless until the arbitration and scheduling model that determines load
exists. The P5 layout will be generated from the contract when the encode/decode path
that consumes it does, not before. P0 generates the registry, the Parquet schemas and the
TypeScript types, and a test pins that set.

**`friction ellipse` and `load sum` are P2, not P1.** `InvariantResult.backing` records
this per invariant rather than in prose: 1, 6 and 7 from P1, 2, 3, 4 and 5 from P2, 8 from
P5. The P0 checks are semantic, not physical - they pin down what each invariant means,
what its tolerance is, and what its sign conventions are, so that a P1 failure is a
physics bug and not an argument about the definition.

**The energy check uses reported accelerations, not finite differences.**
`d(KE)/dt = m * (vx * ax + vy * ay)` from the ground-truth stream, at every step, because
that is what the physics core will publish and it is exact. A difference of consecutive
positions is first-order inaccurate at the interval ends and would put a spurious residual
inside the 1% gate. The P1 harness may add the finite-difference form as a second, weaker
check once there are real traces to difference.

**`ruff format` does not touch generated files; `ruff check` does.** `ruff check` is kept
on `src/f1telemetry/generated` because it earns its keep: it is what caught a real emitter
bug where a Python string literal was emitted unquoted, and it catches an F821 in generated
code before a consumer does. `ruff format` is excluded there, because hand-formatting
generated files makes `ruff format` and `f1-codegen` disagree and the two gates then fight
each other. The generated tree is governed by the codegen diff check instead.

**`.gitattributes` pins LF.** CI compares generated files with `git diff --exit-code`, so a
contributor on Windows with `core.autocrlf=true` must not be able to produce a diff that is
not a contract change. This is a real failure mode of the P0-T6 gate, not a hypothetical
one.

**`basedpyright` is configured, not defaulted.** `typeCheckingMode = "recommended"` with
seven rules turned off, each for a stated reason: the four `reportUnknown*` rules because
pyarrow and numba ship no stubs (about 280 zero-signal warnings), `reportExplicitAny`
because `Any` is the honest annotation for a YAML or JSON node before the loader has
validated it, `reportImplicitStringConcatenation` because a code generator builds source
text out of adjacent string literals on purpose, and `reportUnusedCallResult` /
`reportUnusedParameter` because tests call functions for their side effects and the
invariant checkers share one `(record, spec)` signature whether or not a given check needs
the spec. The result is `0 errors, 0 warnings`; a rule that is silenced without a reason
here is how a type checker stops being a gate.

**`polars` is not a dependency yet.** See the first entry above; it arrives with P5.

## 8. Phase 1: every value is claimed, not just labelled (P1-T1)

P0's rule was one `provenance` line per section, which is too coarse once a section holds both
regulation limits and synthesised placeholders. `powertrain.ice` is the case that forced the
change: C5.13.4 fixes the idle ceiling at 4000 rpm, while the torque curve in the same block
is a synthesis and 400 kW is nowhere in the regulations. One section-level label cannot say
that honestly.

So each section now claims its numbers one at a time, in one of exactly two blocks:

* `regulation: {value: {clause, page, quote}}` - read out of Issue 20, with the wording quoted
  so a reviewer can check it without opening the PDF;
* `not_regulated: {value: "why"}` - not a regulation number, with the reason and the source it
  actually came from.

A value claimed by both is an audit failure, not a warning: a number has one basis, and two
claims mean one of them has gone stale. `provenance_audit()` enforces it, and
`CarSpec.citations()` exposes the resulting `path -> (clause, page)` map so a test can pin it.
`docs/calibration.md` carries the method, the corrections against Issue 20, and the synthesis
assumptions.

`provenance: mixed` is the new section-level value meaning "some values regulated, some not",
and it requires a `source_date` like the other dated ones.

**A cited curve may need a breakpoint the clause does not state.** `derived_points` inside a
`regulation` claim names the curve speeds that are *not* in the quoted text, each with the
reason. The deployment curve needs one: C5.2.8 gives two linear segments meeting at 340 km/h,
but `1800 - 5v` reaches C5.2.7's 350 kW cap at 290 km/h, so the sampled curve turns there. The
**limit** at that point is the regulation's; only the **speed** is derived. Without this the
curve reads as four regulated points, and the disclosure lives only in `quote` prose that no
check can see. `CarSpec.derived_points()` reports it beside `citations()`.

**Two P0 keys were renamed rather than re-cited.** `tyres.nominal_width_mm: 305.0` became
`front_width_mm: 315.0` and `rear_width_mm: 401.3` (C10.7.2 is a per-axle mounting width, and
305 mm is a 2022-era number), and `chassis.floor_width_m: 1.9` became `chassis.overall_width_m`
because `PLAN.md` §5's "floor width" is actually the C2.3.1 overall-width limit. Citing a
clause whose wording does not match the key name is exactly the error this mechanism exists to
prevent, so the keys were corrected rather than left with a plausible-looking citation.

**`integration.dt_s` is new.** The fixed 100 us step moved from being a kernel constant to a
data value, so changing it is a YAML edit. Nothing in Section C sets a simulation step, so it
is claimed `not_regulated` with `PLAN.md` §4 as the reason.

**`CarSpec.kernel_config()` is the one-way boundary.** It returns flat scalars and contiguous
writable `float64` arrays, range-checked in Python, so the kernel receives numbers and never
YAML. That is `PLAN.md` §4.1 rules 1 and 3 made concrete. A zero rolling radius, a negative
deployment limit or mismatched `Cl`/`Cd` speed grids raise `ContractError` before any array
reaches Numba.

**The boundary is built once, at load.** `CarSpec.build_kernel_config()` is the only place these
numbers are validated, and `load_car_spec` calls it so the `KernelConfig` is constructed during
loading rather than on first access. An earlier version of this work validated lazily inside
`kernel_config()`, reading most values back out of the raw document while reaching the typed
fields for others - two routes to every coefficient, split line by line rather than by a rule,
which is exactly the drift the cross-phase rules exist to prevent. Now a value the `CarSpec`
already holds as a typed field is read from that field, and only the P1 inputs beyond the P0
field set are parsed from raw.

**`chassis.c42_enforcement` records what cannot be enforced.** C4.2's 0.44 / 0.54 axle minima are
fractions of the C4.1 Minimum Mass, which is 724 kg *plus* a Nominal Tyre Mass the tyre supplier
publishes after the final testing camp (C4.7) and the regulations do not contain. So the floors
cannot be compared against `mass.total_kg` without enforcing a rule that does not exist. The
loader checks only that `front_weight_fraction` is in (0, 1); the floors stay in the file as
cited data, out of `KernelConfig` because a longitudinal kernel has no axle, and the gap is
recorded as `status: not_enforced` with `becomes_checkable_at: P2-T2`.

**Generated type aliases use `TypeAlias`, not the PEP 695 `type` statement.** ruff's
`UP040` prefers the new form, but ruff mis-analyses `type X = 'a' | 'b'` and reports every
string literal as an undefined name, so `UP040` is disabled for the generated tree instead
of the whole project.
