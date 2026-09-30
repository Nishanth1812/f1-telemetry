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

The `analytics` marker is registered in `pyproject.toml` and the CI stage runs
`pytest -m analytics` today, collecting zero tests and passing. `PLAN.md` §12 requires that
stage, and `PLAN.md` §9.1's detection table is a P7 deliverable. Registering the marker
now means the stage exists and is visibly empty rather than absent, which is the
difference between "not built yet" and "silently broken".

The same applies to the web build stage: it is present in the workflow and exits 0 with a
message, and becomes a real build when P0-T9 lands.

## 6. Provisional `car_spec` provenance

`car_spec.yaml` is versioned to FIA 2026 Section C Issue 16 (2026-02-27) as
`PHASES.md` P0-T3 requires, and every value in it is a **placeholder**. It ships with:

* `spec.provenance: provisional`
* `spec.calibration_status: uncalibrated`
* `spec.source_date: 2026-09-30` - the date the placeholder was written, not a measurement
  date, and stated as such in the file
* `spec.confirmed_against_regulation_text: false`, `spec.confirmed_by: null`

`source_date` on every section records the same thing, and a section whose provenance is
`plan`, `provisional` or `synthesised` without a `source_date` is an audit failure that
`provenance_audit()` reports and a test asserts. That is the machine-checkable form of the
cross-phase rule "every coefficient gets a provenance line". A nested block that declares
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

**Generated type aliases use `TypeAlias`, not the PEP 695 `type` statement.** ruff's
`UP040` prefers the new form, but ruff mis-analyses `type X = 'a' | 'b'` and reports every
string literal as an undefined name, so `UP040` is disabled for the generated tree instead
of the whole project.
