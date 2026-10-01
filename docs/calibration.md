# Calibration and provenance

Every coefficient the simulation uses lives in [`car_spec.yaml`](../car_spec.yaml). This
document records where each one came from, in the order a reviewer needs to check it: which
regulation issue the file is pinned to, what each clause actually says, and which numbers are
synthesised because no public source exists.

The file itself carries the per-value detail - a `regulation` block with clause, page and
quoted text, or a `not_regulated` block saying why a value is not a regulation number. This
document does not restate those; it explains the method, the corrections made against
Issue 20, and the synthesis assumptions that later tasks must build on.

---

## 1. Source document

**FIA 2026 Formula One Regulations, Section C (Technical), Issue 20, 5 August 2026.**
<https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_c_technical_-_iss_20_-_2026-08-05.pdf>

Located on the official FIA technical-regulations listing
(<https://www.fia.com/regulation/category/110/technical-regulations>) on 2026-10-01, which
shows Issue 20 as the current 2026 Section C issue. The listing also carries **2027 Section C
Issue 2, dated 2026-08-05**, so a spec bump for 2027 is already available; this file stays on
2026 because that is the season the P1 model simulates.

`PLAN.md` section 5 and `PHASES.md` P0-T3 both cited **Issue 16 (2026-02-27)**. That is
recorded in `car_spec.yaml` under `spec.supersedes`. The clauses this task relies on had the
same values in Issue 16, so the version bump did not move any coefficient - but the citation
was stale and is now pinned.

### Method

Each value was read out of the extracted text of the Issue 20 PDF, and the page recorded is
the **document page printed in the footer** (which matches the PDF page index in this
document). Verification for the clauses used:

| Clause | Page | What it fixes |
|---|---:|---|
| C2.3.1 | 10 | Overall car width: no part beyond 950 mm from `Y=0` |
| C2.3.3 | 11 | Wheelbase no more than 3400 mm at Legality Setup |
| C4.1 | 59 | Minimum Mass 724 kg + Nominal Tyre Mass (726 kg in Qualifying) |
| C4.2 | 59 | Front axle at least 0.44 x Minimum Mass, rear at least 0.54 |
| C4.5 | 60 | Driver reference mass + Driver Ballast not less than 82 kg |
| C4.7 | 60 | Nominal Tyre Mass is supplier-published, not fixed by the regulations |
| C5.1.2 | 61 | Engine capacity 1600 cc (+0/-10) |
| C5.1.18 | 62 | Article C5 numerical values are at ambient temperature |
| C5.2.3 | 64 | Fuel energy flow no more than 3000 MJ/h, 550 MJ/h per cylinder |
| C5.2.4 | 64 | Below 10 500 rpm, `EF(MJ/h) = 0.27*N(rpm) + 165` |
| C5.2.5 | 64 | Partial-load limit `EF(MJ/h) = 9.78*P(kW) + 869` above -50 kW |
| C5.2.7 | 64 | Absolute ERS-K electrical DC power no more than 350 kW |
| C5.2.8 | 64 | Speed-dependent ERS-K deployment limits, plus the Overtake profile |
| C5.2.9 | 64 | ES state-of-charge difference no more than 4 MJ on track |
| C5.2.10 | 64 | Recharge no more than 8.5 MJ per lap (7 MJ reduced case) |
| C5.2.11 | 65 | MGU-K mechanical torque magnitude no more than 500 Nm |
| C5.2.12 | 65 | MGU-K only above 50 km/h from a standing start |
| C5.12.2 / C5.12.3 | 74 | Torque-demand gradient and minimum-curve shapes |
| C5.13.4 | 75 | Idle speed control target no more than 4000 rpm |
| C5.14 | 76 | Rev limits inside a 750 rpm band; no absolute limiter stated |
| C10.7.2 | 111 | 18 inch rims, 462.5/463 mm rim diameter, tyre mounting widths |
| C10.10.1 | 115 | Front wheel origin not outboard of `Y=603`, rear not outboard of `Y=525` |

**What is machine-checked, precisely.** The table above is the set of clauses that were *read*
for this task. It is larger than the set that is *asserted*, and the difference is deliberate:
some rows are there to record that a clause was checked and deliberately **not** recorded as a
source - C5.14 (no absolute rev limiter stated), C5.2.5, C5.12.2/C5.12.3, C10.10.1, C4.7, C5.1.2,
C5.1.18 - because none of them fixes a P1 kernel input.

The assertions are:

| Check | Covers |
|---|---|
| `test_every_regulated_value_cites_the_clause_it_was_read_from` | The exact `path -> (clause, page)` map for all **23** cited entries, so a citation cannot be dropped, re-pointed, or invented |
| `test_the_cited_clauses_still_say_what_the_values_claim` | The numeric content of those same **23** entries: the value in `car_spec.yaml` must equal the number the quoted clause states |
| `test_a_curve_claim_declares_its_derived_breakpoints` | The one breakpoint in the deployment curve that C5.2.8 does **not** state, declared as derived |

So the coverage is 23 cited values, not every row of the table above. The unchecked rows are a
reading record, not a gate; a reviewer should not read the table as a claim that all 30 rows are
asserted by a test.

## 2. What Issue 20 changed in this file

Five P0 values were wrong or misleading against the current issue, and two keys were renamed
so that a citation could not point at the wrong clause.

**`powertrain.mgu_k.store_energy_mj`: 7.0 → 4.0 (C5.2.9, page 64).** The 7 MJ figure comes
from `PLAN.md` section 6, which conflated two different limits. C5.2.9 caps the usable store
at a 4 MJ state-of-charge swing; the 7 MJ figure is the reduced per-lap **recharge** threshold
in C5.2.10 (8.5 MJ, reduced to 7 MJ at some competitions). Both are now recorded separately:
`store_energy_mj: 4.0` and `recharge_limit_mj_per_lap: 8.5`. A simulation that deploys a
7 MJ store is deploying energy the regulation does not permit.

**`powertrain.mgu_k.overtake_mode_extra_kw` is no longer the model's Overtake definition.**
`PLAN.md` section 6 describes Overtake as a flat "+150 kW cap". C5.2.8.ii defines it as an
alternative ERS-K power profile, `P(kW) = 7100 - 20*v` below 355 km/h and zero at or above
it. That profile is now `overtake_curve_kw`. The scalar is retained with a `not_regulated`
note because Task 4 may still want the shorthand; it should not drive physics.

**`powertrain.mgu_k.deployment_curve_kw` is new.** The non-Overtake deployment limit is
piecewise linear in car speed: `1800 - 5*v` below 340 km/h, `6900 - 20*v` from 340 to
345 km/h, zero at or above 345 km/h. It is stored as breakpoints at 0, 290, 340 and
345 km/h. This matters for P1: a car at 300 km/h may deploy at most 300 kW, not 350 kW.

**The 290 km/h breakpoint is derived, and that is now machine-visible.** It is not a breakpoint
in C5.2.8: the clause gives two linear segments meeting at 340 km/h. The 290 km/h point exists
because `1800 - 5*v` reaches the 350 kW absolute ERS-K cap of C5.2.7 at that speed, so the
sampled curve has to turn there or the first segment would report more than the cap allows.
The **limit value** at 290 km/h is C5.2.7's 350 kW; only the **speed** is derived. That is
declared in the claim itself as `derived_points`, and `CarSpec.derived_points()` reports it
alongside `citations()`, so a reader asking "which of these numbers are the regulation's?" gets
a partial answer from the same API rather than having to read a `quote` string. The audit
rejects a derived point naming a speed the curve does not contain, or one with an empty
`basis`.

**`powertrain.ice` gains the fuel-energy-flow limits (C5.2.3, C5.2.4, page 64).** The
regulations do **not** state an ICE power in kW anywhere. They bound the ICE through fuel
energy flow. The 400 kW peak stays in the file as a published industry figure with a
`not_regulated` note, and the four energy-flow numbers are recorded as the citation that
actually constrains it. Task 4 should check its torque curve against the energy-flow curve
rather than against 400 kW.

**`tyres.nominal_width_mm: 305.0` → `front_width_mm: 315.0`, `rear_width_mm: 401.3`
(C10.7.2, page 111).** 305 mm is a 2022-era front width. Issue 20 specifies a tyre mounting
width of 315 +/- 0.5 mm front and 401.3 +/- 0.5 mm rear on a 462.5/463 mm rim - the 18 inch
rims `PLAN.md` section 5 says were retained. `rim_diameter_mm` is recorded alongside.

**`chassis.floor_width_m` → `chassis.overall_width_m` (C2.3.1, page 10).** `PLAN.md`
section 5's "1.9 m floor width" is really the overall-width limit; 950 mm either side of
`Y=0` is what C2.3.1 says. Keeping a key named `floor_width_m` would have cited a
floor-width clause the regulations do not contain.

Two values were left alone deliberately:

* `mass.total_kg: 800.0`. C4.1 sets a **floor**, not a mass. A real car runs ballast down to
  the floor plus a driver and starting fuel, so a figure above 724 kg is not a contradiction
  of the regulation and Task 5 tunes it against the acceleration targets.
* `chassis.front_weight_fraction: 0.46`. See below - C4.2 cannot be enforced against it, so it
  is a placeholder until P2-T2 rather than a synthesised value checked against the floors.

### C4.2 cannot be enforced, and the file says so

C4.2 reads "the mass measured at the front axle must not be less than the Minimum Mass
specified in Article C4.1 factored by 0.44" (and 0.54 for the rear). The denominator is the
C4.1 **Minimum Mass**, which is `724 kg *plus* the Nominal Tyre Mass`. The Nominal Tyre Mass
is published by the tyre supplier after the final tyre-testing camp (C4.7, page 60) - it is not
a number in the regulations, and this project does not have it.

So `minimum_front_axle_fraction: 0.44` and `minimum_rear_axle_fraction: 0.54` are fractions of
`minimum_mass_kg + nominal_tyre_mass_kg`, **not** of `mass.total_kg`. Comparing 0.46 against
0.44 as though both were fractions of the same quantity would enforce a rule that does not
exist, and would silently pass or fail for the wrong reason depending on how far
`total_kg` happens to sit above the floor.

What the loader enforces instead is the part that needs no missing input:
`front_weight_fraction` must be a finite value strictly between 0 and 1. `car_spec.yaml`
records the whole reasoning in `chassis.c42_enforcement` (`status: not_enforced`, the reason,
the two inputs that block it, and `becomes_checkable_at: P2-T2`), so the gap is a stated
limitation rather than an oversight.

Neither floor reaches `KernelConfig`. A longitudinal kernel has no axle, so shipping them there
would invite a P1 kernel to misuse a fraction of the wrong quantity; P2-T2, which owns the
static axial split, is where C4.2 becomes checkable. `test_the_c42_floor_is_not_enforced_against_total_mass`
pins all of this, including that the floors are still recorded with their clause.

## 3. Synthesised values

These have no public source and are **not** measured or regulated. `PLAN.md` section 4 says
not to invent numbers you cannot cite; where a citation does not exist, the value is labelled
rather than invented, and the assumption behind it is written down here.

**ICE torque curve (`powertrain.ice.torque_curve_nm`) — synthesised.** No public F1 torque
curve exists. The shape is `PLAN.md` section 6's synthesis: a published 400 kW peak and
peak-torque rpm shaped into a table, with torque rising from 60 Nm at 2000 rpm to 330 Nm at
10 000 rpm and tapering to 294 Nm at the 13 000 rpm limiter. Assumptions: peak power lands
near 11 000 rpm; torque at the limiter is roughly 89 % of peak, matching the falling
horsepower of a turbocharged engine past peak torque; the curve is monotonic below peak.

**Turbo lag (`powertrain.ice.turbo_lag`) — synthesised.** Torque is multiplied by 0.35 below
4000 rpm, recovering linearly above it. C5.12.2 constrains the *driver torque-demand map*
(gradient no flatter than -0.045 Nm/rpm above 4000 rpm) and C5.12.3 fixes a minimum-curve
shape, but neither publishes a turbo characteristic. 4000 rpm was chosen because it is the rpm
at which C5.12.2 starts applying and because it is a conventional F1 spool threshold.

**Gearbox (`gearbox`) — synthesised.** A geometric 8-speed chosen so 8th gear at the
13 000 rpm limiter lands in the `PLAN.md` section 11 band of 350-370 km/h given the 0.36 m
rolling radius, and so 1st gear works for a launch. Shift points and the 40 ms shift are
synthetic. `PLAN.md` section 6's "approximately 1.6 apart, geometric" cannot hold across eight
gears - 1.6^7 is a 27x span and cannot reach a 350+ km/h top speed from a sane first gear -
see `docs/phase0-decisions.md` section 6.

**Aero curves (`aero`) — synthesised.** Five-point `Cl(v)` and `Cd(v)` tables with
`Cl` rising 1.80 → 3.35 and `Cd` falling 1.15 → 0.62 across 0-105 m/s, deliberately
non-proportional to v^2 per `PLAN.md` section 5.1. `reference_area_m2: 1.5` is order of
magnitude: no FIA article defines a reference area, because the regulations constrain the car
geometrically rather than aerodynamically. `ride_height_sensitivity: 0.18` is a synthesised
ground-effect term and is what will make porpoising observable later. Task 3 replaces the
curves with the published Limebeer and Tremlett F1 parameter set.

**Rolling radius and wheel diameter (`tyres`) — synthesised.** `PLAN.md` section 4 gives
1.02-1.05 x loaded radius for F1; 0.36 m is that midpoint and is the number that couples gear
ratios to road speed, which is why it matters more than the tyre itself. C10.7.2 fixes the rim
diameter (462.5 mm, recorded) but not the loaded rolling radius or the overall tyre diameter;
the 18 inch rim to 0.72 m rolling diameter step is a tyre-supplier choice.

**MGU-K superclip (`powertrain.mgu_k.superclip_s: 3.0`) — synthesised, and the weakest value
in the file.** `PLAN.md` section 6 estimates 2-4 s from the 2026-04-21 refinements.
**Issue 20 contains no superclip duration article** - a search of the extracted document text
for "clip" returns nothing. The 3.0 s midpoint is a project assumption. If Task 4 uses it for
energy management it must be treated as unverified, and the 2027 regulations may well state a
duration that this project has not read.

**Power split and fuel LHV (`powertrain`) — plan figures.** 0.53 ICE / 0.47 ERS is
`PLAN.md` section 6; the regulations cap ERS-K power by the explicit speed curve of C5.2.8
and state no split. `fuel_lhv_kj_kg: 44000.0` is a standard F1 fuel value: C5.2.6 has the
FIA measure the real LHV through the SECU per lap, so the constant is a stand-in for a
measured quantity that is not published.

**Fixed step (`integration.dt_s: 0.0001`) — a project decision.** `PLAN.md` section 4 fixes
the step at 100 us for determinism. No FIA article sets a simulation step. It moved into
`car_spec.yaml` in this task so that changing it is a data edit rather than a code edit.

**`constants` — physical standards.** Standard gravitational acceleration (ISO 80000-3),
ISA sea-level density at 15 degC, ISA reference temperature 20 degC. These are standards,
not FIA limits and not placeholders, which is why the section reads `provenance: standard`.

## 4. How the loader keeps this honest

`provenance_audit()` in `src/f1telemetry/contracts/car_spec.py` is the machine-checkable form
of the cross-phase rule "every coefficient gets a provenance line". It reports:

* a value with no claim at all - neither `regulation` nor `not_regulated`;
* a `regulation` claim missing `clause` or `page`, or naming something that is not a value;
* a value claimed by **both** blocks, since a number has one basis and claiming two means one
  of them is stale;
* a section whose `provenance` says `regulated` or `mixed` while carrying uncited numbers;
* an empty `not_regulated` sentence, which would be a placeholder for a reason;
* a `derived_points` entry naming a speed its curve does not contain, or with an empty `basis`.

Three APIs expose the result. `CarSpec.citations()` returns the
`dotted.path -> (clause, page)` map for every regulated value. `CarSpec.derived_points()`
returns the `dotted.path -> speeds` map for curve breakpoints the clause does not state.
`tests/test_car_spec.py` pins both.

`CarSpec.kernel_config()` hands the kernel flat scalars and contiguous writable `float64`
arrays, so the kernel receives numbers and never YAML - `PLAN.md` section 4.1 rules 1 and 3.
It is built **once, at load**, by `CarSpec.build_kernel_config`, which is the single validation
path over these numbers: every value `CarSpec` already holds as a typed field is read from that
field rather than re-parsed out of the raw document, and only the P1 inputs added beyond the P0
field set are parsed there. So there is one place a coefficient can be wrong, and
`kernel_config()` itself just returns the stored result.

The checks are the ones a compiled kernel cannot make: a zero rolling radius, a non-positive
torque multiplier, a power split outside (0, 1), a shift point above the rev limiter, a
deployment curve with a negative limit, a `front_weight_fraction` outside (0, 1), and **mismatched
`Cl` and `Cd` speed grids**. That last one matters because `KernelConfig` exposes a single
`aero_speed_m_s` axis: a `Cd` curve on different breakpoints would produce a `cd` array of a
different length to the axis indexing it, which is an out-of-bounds read inside compiled code
rather than a load error. All of them raise `ContractError` at the boundary.

## 5. Still outstanding for later tasks

* **0-100 km/h and top-speed targets.** `PLAN.md` section 11 gives 2.5-3.0 s and
  350-370 km/h as sanity bounds, not agreed tolerances. Task 5 must choose published figures,
  record the sources, and set numeric tolerances here before anything is tuned.
* **ICE torque curve.** Task 4 owns it, and it should be validated against the C5.2.3/5.2.4
  energy-flow limits rather than the 400 kW shorthand.
* **Gearbox.** Task 4 replaces the provisional ratios with the drag-limited top-speed solve.
* **Aero curves.** Task 3 replaces them with the published F1 parameter set.
* **`fastest-lap` cross-check.** Task 6, per `PHASES.md` P1-T10.
* **2027 regulations.** Section C Issue 2 is already published. A 2027 run would be a new
  `car_spec.yaml` keyed to a new issue, not an edit to this one.
* **C4.2 enforcement.** Blocked on the Nominal Tyre Mass (see section 2). Becomes checkable at
  P2-T2, where the axial split lives.
* **A specification bump is a coordinated edit, not one file.** The same regulated numbers
  appear in four places: `car_spec.yaml`, the `EXPECTED_CITATIONS` map in
  `tests/test_car_spec.py`, the body of `test_the_cited_clauses_still_say_what_the_values_claim`,
  and the clause table in section 1 of this document. Re-verifying against a new issue means
  updating all four in one commit; the tests fail loudly if the first two disagree, but nothing
  links the third and fourth automatically.
