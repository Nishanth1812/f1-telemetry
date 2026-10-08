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
recorded in `car_spec.yaml` under `spec.supersedes`. Issue 16 lists the C4.1 values as 724 kg
and 726 kg plus Nominal Tyre Mass. The coefficients in this file are pinned to Issue 20; no
broader claim that every clause is unchanged between issues is needed here.

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
| C5.12.2 | 73 | Torque-demand gradient above 4000 rpm no flatter than -0.045 Nm/rpm |
| C5.12.3 | 74 | Minimum-curve shape: `Torque (Nm) = -0.0027 * rpm - 30` |
| C5.13.4 | 75 | Idle speed control target no more than 4000 rpm |
| C5.14 | 76 | Rev limits inside a 750 rpm band; no absolute limiter stated |
| C10.7.2 | 111 | 462.5/463 mm rim diameter, tyre mounting widths per axle |
| C10.10.1 | 115 | Front wheel origin not outboard of `Y=603`, rear not outboard of `Y=525` |

**Scope notes on the clauses read but not recorded.** Two things in that table are worth being
explicit about, because reading a clause is not the same as using it:

- **C5.2.5 has three arms and only one is relevant to P1.** The clause states
  `EF(MJ/h) = 380` at or below -50 kW engine power, `EF(MJ/h) = 9.78*P(kW) + 869` above it,
  and is bounded by C5.2.3. P1 models full-throttle running, so the -50 kW arm and the
  partial-load arm are both out of scope for Task 5's scenarios; only the C5.2.3 and C5.2.4
  limits are recorded. A future partial-load fuel model needs the 380 MJ/h arm added.
- **C5.2.8 has four sub-clauses and only two are recorded.** `.i` (non-Overtake) and `.ii`
  (Overtake) are the profiles in `car_spec.yaml`. `.iii` is a further, *lower* limit
  (250 kW below 310 km/h, then the `.i` curve) that applies in a Race or Sprint Session on
  specified circuit sectors during a power-limited period, subject to Article B7.2 — a
  circuit-and-session-specific regime this project has no data for. `.iv` defers to
  Low-Grip-condition curves in FIA-F1-DOC-111, a document not read here. Neither is recorded,
  and P1 scenarios should not be judged against them.
- **C10.7.2 does not say "18 inch".** The clause gives rim diameters in millimetres. 462.5 mm
  is consistent with an 18 inch rim and `PLAN.md` §5 independently states 18 inch wheels are
  retained, but that is an inference from this project, not an FIA statement, and
  `car_spec.yaml` records it under `tyres.inference` rather than as a quoted value.

**What is machine-checked, precisely.** The table above lists the **23 clauses** read for this
task. That is a reading record, not a set of gates: it is larger than the set that is
*asserted*, and the difference is deliberate. Some rows record a clause that was checked and
deliberately **not** used as a source — C5.14 (states no absolute rev limiter), C5.2.5,
C5.12.2, C5.12.3, C10.10.1, C4.7, C5.1.2, C5.1.18 — because none fixes a P1 kernel input.

The assertions are:

| Check | Covers |
|---|---|
| `test_every_regulated_value_cites_the_clause_it_was_read_from` | The exact `path -> (clause, page)` map for all **23** cited entries, so a citation cannot be dropped, re-pointed, or invented |
| `test_the_cited_clauses_still_say_what_the_values_claim` | The numeric content of those same **23** entries: the value in `car_spec.yaml` must equal the number the quoted clause states |
| `test_a_curve_claim_declares_its_derived_breakpoints` | The two breakpoints in the ERS curves that C5.2.8 does **not** state — 290 and 337.5 km/h — declared as derived |
| `test_the_ers_curves_never_exceed_the_absolute_cap` | That neither curve exceeds C5.2.7's 350 kW at any interpolated speed |
| `test_an_ers_curve_point_above_the_absolute_cap_is_rejected` | That a curve point above the cap is rejected at the boundary, for both curves |

So the coverage is **23 cited values across 16 distinct clauses**, not every row of the table
above. The other seven clauses read are a reading record, not a gate; a reviewer should not
read the table as a claim that all 23 rows are asserted by a test.

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

**`powertrain.mgu_k.deployment_curve_kw` is new.** C5.2.8.i gives the non-Overtake propulsion
limit as `1800 - 5*v` below 340 km/h, `6900 - 20*v` from 340 to 345 km/h, then zero. C5.2.7
adds an absolute 350 kW cap, so the effective limit is the smaller value from the two clauses.
The stored curve is clamped at 350 kW below the 290 km/h crossing; at 300 km/h it permits
300 kW.

**Both ERS curves apply the absolute cap, and both crossing speeds are derived.** C5.2.7 limits
absolute ERS-K electrical DC power to 350 kW. C5.2.8.i's `1800 - 5*v` reaches that cap at
290 km/h; C5.2.8.ii's Overtake formula `7100 - 20*v` reaches it at 337.5 km/h. Those speeds
are not stated in C5.2.8, so each is declared as `derived_points` in its claim. The effective
arrays in `KernelConfig` stay at or below 350 kW, and the shared builder rejects any curve knot
above the C5.2.7 cap. `CarSpec.derived_points()` reports both crossings beside `citations()`;
the audit rejects declared speeds absent from their curve and empty bases.

**`powertrain.ice` gains the fuel-energy-flow limits (C5.2.3, C5.2.4, page 64).** The
regulations do **not** state an ICE power in kW anywhere. They bound the ICE through fuel
energy flow. The 400 kW peak stays in the file as a published industry figure with a
`not_regulated` note, and the four energy-flow numbers are recorded as the citation that
actually constrains it. Task 4 should check its torque curve against the energy-flow curve
rather than against 400 kW.

**`tyres.nominal_width_mm: 305.0` → `front_width_mm: 315.0`, `rear_width_mm: 401.3`
(C10.7.2, page 111).** 305 mm is a 2022-era front width. Issue 20 states one table row,
`Tyre Mounting Width 315 ± 0.5 401.3 ± 0.5`, under the column headings "Front Wheel" and
"Rear Wheel" — 315 is the front value and 401.3 the rear, and the clause does not print them
separately. `rim_diameter_mm` (462.5, from `Rim Diameter 462.5 / 463 462.5 / 463`) is recorded
alongside. The "18 inch" description of those rims is this project's inference, not the
regulation's; see §1.

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

### C4.2 cannot be enforced yet, and the file says why

C4.2 reads "At all times during the Qualifying and Sprint Qualifying Sessions, with the car
resting on a horizontal plane: i. the mass measured at the front axle must not be less than the
Minimum Mass specified in Article C4.1 factored by 0.44" (and 0.54 for the rear). Two things
follow, and both matter:

**It is a Qualifying-only check.** The scope is in the clause's own first sentence and is part
of the quote in `car_spec.yaml`. It does not apply during a Race or Sprint Session, so it would
not govern most P1 scenarios even with every input in hand. That is a scope fact about C4.2,
not a reason to ignore it: the FIA checks it at Qualifying, and P2's static split work is where
it becomes relevant.

**The denominator is not `mass.total_kg`.** It is the C4.1 **Minimum Mass**, which is
`724 kg *plus* the Nominal Tyre Mass`. C4.7 (page 60) says the tyre provider measures new
production dry-weather tyres and publishes the mean mass of a 50-tyre-per-axle sample after the
final day of TCC opportunity, prior to the start of the Championship. So the Nominal Tyre Mass
is a **published figure this project simply has not looked up** — obtainable, not fundamentally
unavailable, and not part of the regulations proper.

So `minimum_front_axle_fraction: 0.44` and `minimum_rear_axle_fraction: 0.54` are fractions of
`minimum_mass_kg + nominal_tyre_mass_kg`, **not** of `mass.total_kg`. Comparing 0.46 against
0.44 as though both were fractions of the same quantity would enforce a rule that does not
exist, and would pass or fail for the wrong reason depending on how far `total_kg` happens to sit
above the floor.

What the loader enforces instead is the part that needs no missing input:
`front_weight_fraction` must be a finite value strictly between 0 and 1. `car_spec.yaml`
records the whole reasoning in `chassis.c42_enforcement` (`status: not_enforced`, the scope, a
`missing_input` block naming `nominal_tyre_mass_kg` and its C4.7 source, the inputs that block
enforcement, and `becomes_checkable_at: P2-T2`), so the gap is a stated limitation rather than
an oversight. A task that needs the real floor should obtain the supplier figure and add it to
`mass` as a cited value.

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

**Turbo lag (`powertrain.ice.turbo_lag`) — synthesised.** The table is multiplied by **0.35
strictly below 4000 rpm and by exactly 1.0 at 4000 rpm and above**, so the reduction is a step
rather than a ramp. That is what the committed numbers describe: `car_spec.yaml` records only
`collapse_below_rpm` and `multiplier_at_collapse` and no endpoint for a recovery, so there is no
value in any file for a multiplier to recover *to*; `PLAN.md` section 6's "falls off sharply
below ~4 000 rpm" and `PHASES.md` P1-T4's "collapsing below ~4 000 rpm" both ask for the
collapse, and neither asks for the climb. The underlying curve is unaffected by that step: the
eight knots are interpolated piecewise-linearly and held flat outside them (the rule Task 3 froze
for `Cl(v)` and `Cd(v)`), so torque still rises smoothly with rpm within each segment and the
only discontinuity is the multiplier's. C5.12.2 constrains the *driver torque-demand map*
(gradient no flatter than -0.045 Nm/rpm above 4000 rpm) and C5.12.3 fixes a minimum-curve
shape, but neither publishes a turbo characteristic. 4000 rpm was chosen because it is the rpm
at which C5.12.2 starts applying and because it is a conventional F1 spool threshold.

**Gearbox (`gearbox`) — synthesised.** A geometric 8-speed chosen so 8th gear at the
13 000 rpm limiter lands in the `PLAN.md` section 11 band of 350-370 km/h given the 0.36 m
rolling radius, and so 1st gear works for a launch. (That band is the old sanity band, since
retired; the current straight-line criterion is the 325.8 km/h reachability floor in section 6,
which is a floor rather than a landing speed. The ratios are still synthetic and are not derived
from any current reference point.) Shift points and the 40 ms shift are
synthetic. `PLAN.md` section 6's "approximately 1.6 apart, geometric" cannot hold across eight
gears - 1.6^7 is a 27x span and cannot reach a 350+ km/h top speed from a sane first gear -
see `docs/phase0-decisions.md` section 6.

The eight ratios and `final_drive` are **used**, not merely counted, and the gear reduction happens
**before** the clutch. `step_gearbox` computes `engine torque × throttle × ratio × final_drive` and
then applies the clutch ceiling, so the same 330 Nm of engine torque arrives at the differential as
3 844 Nm in first and 1 618 Nm in eighth. The order is `PLAN.md` section 6's chain read literally —
`torque_curve → gearbox → clutch → differential → wheels` — and it is not cosmetic: clamping the
engine torque first and multiplying afterwards leaves the clutch capacity **unreachable** rather
than reporting a wrong number, because the engine peak is below the capacity and the minimum never
selects it. A capacity that can never be selected is configuration no model reads.

The product is still a **torque**, and that is the boundary of P1-T5: P1-T7 divides by the rolling
radius to make `Fx`, and P1-T6 supplies the wheel speed it divides by. Both arrived in Phase 1
slice 4, which closes the loop inside the integrator rather than leaving the torque at a boundary -
see "Wheel rotational state" below.

**Clutch capacity (`gearbox.clutch_torque_capacity_nm: 3000.0`) — synthesised (P1-T5).** No FIA
article states a clutch torque capacity and no public F1 figure is cited here.

**The number is differential-side, and that is forced by `PLAN.md` section 6.** The chain there is
`torque_curve(rpm) → gearbox(8-speed) → clutch → differential → wheels`, so the clutch sits *after*
the gear reduction and the capacity is compared against the already-reduced torque:

```text
output = min(throttle × engine_torque × ratio(gear) × final_drive,  capacity × engagement)
```

Clamping the engine torque first and multiplying by the ratio afterwards is the same formula in the
wrong order, and on this data its effect is not an inflated torque but a **dead clamp**:
`min(330, 600) × 11.649` is still 3 844 Nm, because the engine peak sits below the capacity so the
minimum never selects it. A capacity that can never be selected is configuration no model reads,
which is the same failure as the ratio table being decorative.

**Why 3 000 Nm.** It sits *inside* the range the committed box produces rather than above it. At the
330 Nm engine peak the box offers, by gear:

| gear | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| offered (Nm) | 3 844 | 3 403 | 3 011 | 2 664 | 2 357 | 2 086 | 1 846 | 1 618 |

So 3 000 Nm gives the model the two regimes it needs, and the tests assert both ends of it:

* **Gears 1–3 are clutch-limited at full engagement** — the low gears, which is where a launch
  happens and where traction limiting belongs. At 0.5 engagement first gear transmits 1 500 Nm, at
  0.2 it transmits 600 Nm.
* **Gears 4–8 pass the gear train's own torque through unmodulated**, so the ratio table stays
  observable across the top half of the box.

A capacity *above* 3 844 Nm would be a ceiling no gear reaches — the dead clamp above, and the reason
a first attempt at 4 200 Nm was rejected. A capacity *below* 1 618 Nm would flatten all eight gears
to the same number. The useful band is between the eighth-gear and first-gear figures, and 3 000 Nm
sits inside it.

Putting the capacity *inside* the minimum rather than multiplying the result by the engagement is
what keeps it load-bearing: `min(torque, capacity) × engagement` would make the capacity a constant
factor the clamp could never reach, which is the dead branch the P1-T3 grip-limit ruling refused to
write.

This is the least-supported number added so far. It has no measured counterpart, nothing in Section C
constrains it, and it is the first of the gearbox figures expected to move once Task 5 runs a
standing start. Note that the units are part of the number: an engine-side capacity for the same
physical clutch would be a different physical quantity, and quoting one where the other belongs
produces the dead clamp described above rather than a wrong torque. `car_spec.yaml` says so in its
`not_regulated` claim and its `note`.

**Wheel rotational state (P1-T6/P1-T7) — synthesised, and the second half of that assumption.**
`wheel_inertia_kg_m2: 0.9` is the wheel-plus-tyre rotational inertia about the spin axis, the `I_w`
of `I_w d(omega)/dt = T_drive - Fx r`. `PLAN.md` section 4 gives **0.5-1.2 kg·m²** as the band an F1
wheel and tyre occupy and this project has read no published figure inside it, so 0.9 is a labelled
placeholder on the same principle as the Pacejka coefficients. No clause fixes a rotational inertia:
C10.7.2 (page 111) gives rim diameter and the two mounting widths, which are geometry, and no
article states a mass for the wheel assembly. The value is consequential rather than cosmetic — it is
the divisor of the whole wheel equation, so a wheel that is too light spins up and locks more
readily than one that is not, and it is the input P1-T9's acceleration work will move first.

**The two things around it are assumptions, not data.** Neither is in `car_spec.yaml`, on purpose:

* **Equal left/right drive split.** C9.1.1 (page 100) states "The transmission may only drive the two
  rear wheels" and says nothing else about the distribution. What decides left against right is a
  differential, and P1 has no differential model, so `forces.REAR_DRIVE_SHARE` is a derived
  `1 / REAR_WHEEL_COUNT` in code rather than a configured figure — a constant that no file supplies
  is a constant that cannot be mistaken for a calibrated one.
* **Equal left/right static load, and no load transfer at all.** `front_weight_fraction: 0.46` is the
  already-recorded `PLAN.md` section 5 placeholder, divided evenly inside each axle, with zero speed
  dependence. **Downforce is split equally across all four patches** because no aero balance is
  configured; this keeps the total vertical load at weight plus downforce. P2-T2 replaces both
  shortcuts with axle and per-corner transfer (see §2 on why C4.2 is not enforceable against it).

**No differential, no traction control, no ABS.** C9.1.2 (page 100) forbids any system capable of
preventing driven wheels from spinning under power or of compensating excessive driver torque
demand, C9.9.1 (page 105) forbids torque transfer from a slower wheel to a faster one, and C11.4.1
(page 117) forbids a braking system designed to prevent wheels locking. In code that means
`wheel_drive_torque_nm` takes **no wheel speed at all**: traction control would appear as a term
reading slip, and a limited-slip differential would appear as the faster rear wheel being handed
more than its half. Neither does, and the tests sweep the wheel's own speed over four orders of
magnitude to prove the torque does not move with it. Wheelspin and lock are still real states —
the Magic Formula's fall-off past its peak produces them — so this is a statement about the torque,
not a claim that wheels never slip.

**No brake model yet.** Braking reaches a wheel in this slice only as a negative drive torque from
a caller, which is the same sign convention the reverse gear already uses. C11.1.1's 2 500 Nm
per-wheel floor and the rest of Article C11 are not represented; that is a brake task, not a wheel
task.

**The loop's update order is part of the model.** One step evaluates every force at the state it
started from, advances the four wheel speeds and the car's speed on those forces, and then advances
position with the speed that step produced. The wheel states are advanced inside the same single
pass that accumulates the force on the car, because recomputing the tyre forces in a second loop
would mean either four transcendental evaluations per step done twice or a scratch array allocated
inside the step, and `PLAN.md` section 4.1 rule 2 forbids the second. The ordering is asserted
rather than left to taste, because updating the wheels on the *new* speed is a different and equally
plausible launch.

**A consequence worth naming: `mass_kg` is not an acceleration knob in P1.** The static axle load is
a fraction of `mass_kg * gravity_m_s2`, so doubling the mass doubles `mu Fz` on every patch and
therefore doubles the force on the car — and the car's inertia doubled with it. With no load transfer
(P2-T2) and no tyre load sensitivity (P2-T3) the mass cancels out of longitudinal acceleration
exactly, and the test asserts that it does rather than skipping it. The wheel states are *not*
mass-free, because the reaction term scales with the load. P2-T2 and P2-T3 will both change this.

**Aero curves (`aero`) — synthesised.** Six-point `Cl(v)` and `Cd(v)` tables with
`Cl` rising 1.80 → 3.35 and `Cd` falling 1.15 → 0.62 across 0-105 m/s, deliberately
non-proportional to v^2 per `PLAN.md` section 5.1. `reference_area_m2: 1.5` is order of
magnitude: no FIA article defines a reference area, because the regulations constrain the car
geometrically rather than aerodynamically. `ride_height_sensitivity: 0.18` is a synthesised
ground-effect term and is what will make porpoising observable later.

**These curves are what the P1-T3 force model consumes, and they stay as Task 1 left them.**
`PLAN.md` section 4 names Limebeer and Tremlett's open F1 model as the parameter set to seed
from, and Task 1 wrote a note here saying Task 3 would replace these tables with it. Task 3
did not, and did not pretend to: a coefficient set this project has not read is not a citation,
and `PLAN.md` section 4's own rule is not to invent numbers you cannot cite. The tables are
labelled synthesised, the force model reads them as data, and replacing them stays open work
(see section 5). Task 5 is where the resulting accelerations and top speed get checked against
published figures, and the coefficients that fail that check are the ones to replace.

**Longitudinal Pacejka (`tyres.longitudinal_pacejka`) — synthesised.** `b: 11.0`, `c: 1.65`,
`e: 0.97`, `mu: 1.7`, added by Task 3 because P1-T6 needs a longitudinal force. `c` and `e` are
the conventional longitudinal Magic Formula shape and curvature factors; `b` is chosen so the
peak lands near 0.36 slip ratio, which is a few tenths of a slip ratio rather than the whole
one; `mu` is a plausible F1 slick peak longitudinal friction coefficient. `e` is checked for
finiteness rather than for a sign, because the curvature factor carries one, and the other three
are checked as magnitudes. No public F1 Pacejka set has been read into this project, so all four
are placeholders for calibration.

**Slip-ratio guard (`tyres.slip_ratio_min_speed_m_s: 1.0`) — synthesised.** The `eps` in
`kappa = (omega r − v) / max(v, eps)`, which `PHASES.md` P1-T6 requires and which is a
divide-by-zero at a standing start without it. 1.0 m/s is a project choice with a visible
consequence: at rest the slip ratio is `omega r / 1.0`, so a launch sits on the falling branch of
the Magic Formula rather than the part of the curve that rises. Task 4 owns the launch model and
has to live with it.

**No load sensitivity, deliberately.** `D = mu Fz` is linear in vertical load. `PLAN.md`
section 4 says a constant-`mu` tyre understates high-speed downforce badly, and that is exactly
what this model does; load sensitivity on `D` and `B` is P2-T3's work. It is stated as a linear
model in `forces.tyre_longitudinal_force` and asserted as one in the tests, so that P2-T3's
correction is a visible edit rather than a silent one.

**Rolling radius and wheel diameter (`tyres`) — synthesised.** `PLAN.md` section 4 gives
1.02-1.05 x loaded radius for F1; 0.36 m is that midpoint and is the number that couples gear
ratios to road speed, which is why it matters more than the tyre itself. C10.7.2 fixes the rim
diameter (462.5 mm, recorded) but not the loaded rolling radius or the overall tyre diameter;
the 462.5 mm rim to 0.72 m rolling diameter step is a tyre-supplier choice; "18 inch" is a
project description, as section 1 explains.

**MGU-K superclip (`powertrain.mgu_k.superclip_s: 3.0`) — synthesised, and the weakest value
in the file.** `PLAN.md` section 6 estimates 2-4 s from the 2026-04-21 refinements.
**Issue 20 contains no superclip duration article** - a search of the extracted document text
for "clip" returns nothing. The 3.0 s midpoint is a project assumption. If Task 4 uses it for
energy management it must be treated as unverified, and the 2027 regulations may well state a
duration that this project has not read.

**Two efficiency conversions (`powertrain.ice.fuel_to_shaft_efficiency: 0.52`,
`powertrain.mgu_k.motor_inverter_efficiency: 0.95`) — synthesised, and the weakest numbers in
the powertrain.** Section C states no efficiency anywhere, and both conversions are needed before a
single clause can be applied:

* **C5.2.3, C5.2.4 and C5.2.5 bound the ICE in MJ/h of fuel *energy*** while everything the model
  computes at the crankshaft is shaft power in kW. `0.52` is the thermal efficiency a current F1
  ICE reaches at its best point, so it is a ceiling-shaped figure rather than a measured one.
* **C5.2.7 caps the ERS-K in *electrical* DC power** while the model holds mechanical torque at the
  MGU-K shaft. `0.95` covers the motor and inverter together.

They are consequential rather than incidental. C5.2.3's 3 000 MJ/h total works out to about
433 kW of shaft power at 0.52, so the committed 400 kW peak **survives the cap uncut** - which is
the state slice 3 tests assert, so that a later efficiency edit which does start clipping shows up
as a failure of that test rather than as every ICE knot assertion quietly becoming a test of the
clip. At 0.40 the same curve *is* cut back, and that is how the tests prove the cap is load-bearing
rather than a branch that can never fire.

`mgu_k.crankshaft_ratio: 3.0` converts motor-shaft speed and torque to the crankshaft. C5.2.11's
500 Nm crankshaft-referenced limit therefore allows at most `500 / ratio` Nm at the faster motor
shaft; the step returns crankshaft-equivalent torque to the gearbox. C5.2.7 and C5.2.8 separately
limit electrical DC power, converted through the declared synthetic motor/inverter efficiency.

**At 3.0, C5.18.5's 60 000 rpm part speed arrives at about 233 km/h**, which is below the speeds
where C5.2.8's curve is interesting. On the committed numbers the relative-speed ceiling therefore
binds *before* the speed-dependent power cap, and the latter is only reachable at a lower ratio.
Both are enforced; the tests isolate each by relaxing the other rather than pretending the
committed pair can exercise both at once.

**C5.2.3's per-cylinder arm is recorded and cited but not enforced.** 550 MJ/h per cylinder needs a
cylinder count to apply, and no block of `car_spec.yaml` carries one and this project's verified
clause set does not cite it. The number stays in the file with its quote and an `enforcement`
note, and the omission is named in `powertrain.ice.note` and pinned by
`test_the_c52_3_per_cylinder_arm_is_recorded_and_the_file_says_it_is_not_enforced`. Inventing a
cylinder count would have been an unsourced number sitting inside a regulatory check.

**C5.18.4's 520 Nm is cited and never read.** It is the threshold above which an *optional*
torque-limiting device may act, not a second cap, so `mgu_k.transient_torque_limiter_threshold_nm`
stays in `car_spec.yaml` with its basis and never reaches `KernelConfig`. The only MGU-K torque
limit the model applies is C5.2.11's 500 Nm.

**Two of C5.2.10's conditions are caller declarations rather than numbers.** Whether the FIA
Standard ECU mandates minimum acceleration (C5.2.12's exception) and which of the article's three
per-lap recharge figures applies are both facts about the *event*, so `step_mgu_k` takes them as
flags and an enum rather than reading them from this file. The 8.5 MJ baseline is the `RACE`
default; 7 MJ and the 4 MJ qualifying floor are `REDUCED` and `QUALIFYING`; the conditional
0.5 MJ allowance is a separate flag on top of whichever is selected.

**Power split and fuel LHV (`powertrain`) — plan figures.** 0.53 ICE / 0.47 ERS is
`PLAN.md` section 6; the regulations cap ERS-K power with both C5.2.7's absolute limit and
C5.2.8's speed profiles, and state no ICE/ERS split. `fuel_lhv_kj_kg: 44000.0` is a standard F1 fuel value: C5.2.6 has the
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
arrays, so the kernel receives numbers and never YAML - `PLAN.md` section 4.1 rules 1 and 3. It
calls `CarSpec.build_kernel_config` on every access, which is the single validation path over
these numbers: every value `CarSpec` holds as a typed field is read from that field rather than
re-parsed out of the raw document, and only the P1 inputs beyond the P0 field set are parsed
there. So there is exactly one place a coefficient can be wrong.

The config is rebuilt on access rather than cached, deliberately. An earlier version built it at
load and stored it on the spec, which desynchronised under `dataclasses.replace` -
`replace(spec, mass_kg=900)` returned a spec that reported 800 kg - and skipped validation
entirely for a spec built or replaced outside the loader. Rebuilding costs a few array
constructions per call, which is nothing beside a 10 kHz run, and removes a class of
stale-config bug that no test of the loader would have caught.

The checks are the ones a compiled kernel cannot make: a zero rolling radius, a non-positive
torque multiplier, a power split outside (0, 1), a shift point above the rev limiter, a
deployment curve with a negative limit or a point above C5.2.7's 350 kW cap, a
`front_weight_fraction` outside (0, 1), and **mismatched `Cl` and `Cd` speed grids**. That last
one matters because `KernelConfig` exposes a single `aero_speed_m_s` axis: a `Cd` curve on
different breakpoints would produce a `cd` array of a different length to the axis indexing it,
which is an out-of-bounds read inside compiled code rather than a load error. The cap check
likewise matters because C5.2.8's formulas permit far more than the absolute limit, so a curve
restoring a raw C5.2.8 value would otherwise hand the kernel power the regulation forbids.

Task 3 added the checks the tyre model needs: `slip_ratio_min_speed_m_s` finite and positive,
because it is the one denominator in the tyre model that would otherwise divide by zero at a
standing start, and the three Pacejka magnitudes positive with the curvature factor finite — the
curvature factor carries a sign, so a sign check would be the wrong rule. Slice 3 added two
fractions on the same principle: `fuel_to_shaft_efficiency` and `mgu_k_motor_inverter_efficiency`
must lie in `(0, 1]`, because both are divisors and an above-one would report more power than the
energy it came from — leaving every fuel-energy-flow limit and the whole C5.2.7 cap permanently
out of reach.

Slice 4 added the same kind of check for the two divisors the wheel equation introduced —
`wheel_inertia_kg_m2` and `rolling_radius_m` must be finite and strictly positive, and
`front_weight_fraction` strictly inside `(0, 1)` — and, more importantly, moved the check *into the
kernel boundary*. Until slice 4, force was an input array and the integrator divided by nothing
but `mass_kg`, so `simulate` validated three scalars. It now indexes the aero curves with
`boundscheck=False` and divides by the wheel inertia inside its step loop, so it re-establishes the
loader's array and sign guarantees through `forces.validated_config_scalars` and
`forces.validate_aero_arrays` before the loop starts. That is the point of one shared validator: three
entry points (`step_forces`, `step_wheel`, `simulate`) now read the force model's coefficients, and
three copies of the same eight names and the same sign rules would be three places for the rule to
be wrong. All of them raise at the boundary.

## 5. Still outstanding for later tasks

* **Performance calibration remains open.** The straight-line reference points are recorded in
  section 6 and `PLAN.md` §11.1, which supersedes the first half of the original note; the bullet
  below keeps what was rejected and why. The last measured outputs missed both, and they were
  measured before the scenario wiring in *Phase 1 scenario wiring* below, so there is currently no
  figure to judge either against. Nothing here is a passed gate.
* **Rejected as calibration targets, for the record.** Formula 1 reported 3.55 s and 3.69 s
  0-100 km/h race-start times for Russell and Hamilton at the 2022 Emilia Romagna GP: useful
  historical context, but not a 2026-car measurement, and no coefficient was tuned to them. Any
  event speed-trap figure is rejected as a **terminal-speed** target for the general reason
  recorded in section 6: a trap sits mid-straight with the car still accelerating, so it is not
  the same physical quantity as a drag-limited asymptote. The 2026 Australian GP observation is
  therefore used only as a reachability floor, and no other event's speed table is used at all.
* **ICE torque curve.** Task 4 owns it, and it should be validated against the C5.2.3/5.2.4
  energy-flow limits rather than the 400 kW shorthand. **Slice 3 now applies that check**
  (`ice_fuel_energy_flow_limit_mj_h` and `step_ice_torque`), and the committed curve survives it
  at the committed efficiency - see section 3. What is *not* done is the calibration half: the
  curve has still never been chosen to land a published acceleration figure.
* **Lap recharge enforcement.** The per-lap budget is enforced and testable, but no lap or
  harvesting scenario exists yet, so nothing calls `begin_lap` in a run. That is Task 5's.
* **C5.2.5's 380 MJ/h constant arm** is now enforced rather than merely recorded (section 3), and
  C5.2.3's **per-cylinder** arm is deliberately not - see section 3.
* **Gearbox.** Task 4 replaces the provisional ratios with the drag-limited top-speed solve.
* **Aero curves.** Task 3 consumed them; it did not replace them (see section 3). Replacing the
  synthesised `Cl`/`Cd` tables with a published F1 parameter set is still open, and should happen
  alongside the Task 5 calibration, since the same published figures decide both.
* **Longitudinal Pacejka coefficients.** `b`, `c`, `e` and `mu` are placeholders (section 3). The
  first evidence that they are wrong will be a Task 5 0-100 km/h figure outside the chosen
  tolerance, or a peak longitudinal deceleration that cannot reach the Task 6 energy balance.
* **Wheel inertia.** `tyres.wheel_inertia_kg_m2: 0.9` is a placeholder inside `PLAN.md` section 4's
  0.5-1.2 kg·m² band (section 3). No public F1 figure has been read into this project, so it is
  labelled rather than cited.
* **Brakes.** The caller can supply signed per-wheel brake torque to the longitudinal kernel.
  Equal four-wheel torque is used only by the synthetic scenario below; brake capacity, bias,
  hydraulic response, ABS and brake-force transfer remain unmodelled. No braking calibration
  target is available in the repo.
* **A differential.** Equal left/right drive and equal left/right load are both synthetic symmetry
  assumptions (section 3). C9.9.1 is satisfied by not modelling a transfer at all, which is not the
  same as modelling a real one.
* **Load sensitivity.** `D = mu Fz` has none. P2-T3, with the lateral force.
* **`fastest-lap` cross-check.** A Windows prebuilt v0.5 run completed using its bundled
  `limebeer-2014-f1.xml` and Catalunya track. It returned 77.9119 s and a 344.377 km/h peak
  speed. This is a working independent solver comparison, not a like-for-like validation: the
  bundled car is a 2014 model, while this kernel is a generic synthesized 2026 straight-line car.
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

## Phase 1 scenario wiring

The straight-line scenarios in `src/f1telemetry/testing/scenarios.py` now declare two things they
previously left implicit. This section is the single place the `PHASES.md` exit gate, the
reference points in section 6 and the measured outputs in the next section all trace to. Neither
declaration adds a car coefficient: `car_spec.yaml` is untouched, and each value is either read
from it or already declared on `ScenarioSegment`.

### Launch: an initial 12 000 rpm engine state during clutch slip

`standing_launch` and `accelerate_to_speed` seed `scenarios.LAUNCH_ICE_RPM` (12 000 rpm) on their
initial, partially engaged clutch-slip segment through `ScenarioSegment.ice_rpm_initial`. This is
the near-**12 000 rpm** reported in the cited 2026 start telemetry. While the clutch slips or a shift
cut opens the driveline, the engine speed advances from delivered ICE and MGU-K crank torque minus
the crank-reflected gearbox output, divided by configured `ice_inertia_kg_m2`. A fully engaged,
non-shifting clutch uses an ideal speed lock to the geared rear wheel speed; clutch capacity still
limits transmitted gearbox torque. This does not add a clutch-friction law, rotating drivetrain
inertia feedback, governor, or rev-cut model. The telemetry value is an initial condition rather
than a speed hold and remains a scenario assumption, not a car coefficient.

The initial value and first engine-state torque/inertia update are pinned by tests. The launch-grip
assertion passes, and the reproduced engine-state baseline is recorded below; the older pinned
measurements remain historical.

### High speed: a store-limited MGU-K request, then an ICE-only tail

`full_throttle` reaches eighth gear on a rolling start through seven driver-requested upshifts,
pulls there, and then requests MGU-K deployment for **20 s** from 15 s into the run before returning
to ICE alone for a **41 s** tail. The run is 76 s. The longer command gives the high-speed run a
meaningful store-limited deployment window; the step clamps actual delivery to available charge and
regulation limits. Shift requests are untouched; they remain the only thing that moves
the gearbox, which is now asserted for this scenario as well as for `full_throttle_shifts`.

The deployment is the existing selection — C5.2.11's 500 Nm crank-referenced limit expressed at the
motor shaft — and `step_mgu_k` bounds it in three further ways: C5.2.7's 350 kW absolute cap in
watts, C5.2.9's 4 MJ store, and the C5.18.5 relative-speed ceiling. The store and hardware limits
bound actual delivery. Top gear is the only place on these ratios and tyres where a full deployment
is transmissible at all, which is
also where the run needs to be to exercise the combined propulsion-and-aero path — ICE and MGU-K
through the gearbox, the tyre traction limit, and the `Cl`/`Cd` curves — at the speed where the
reference car was observed.

The tail is separate because that is what makes the two high-speed quantities separable, which
section 6 has required all along:

- **Transient maximum speed** — the highest speed the run reaches anywhere, deployment included.
  This is the only output comparable to the 325.8 km/h reachability floor, and it is a reachability
  check with no tolerance attached.
- **Terminal speed** — the speed the run settles at over the motor-free tail, i.e. inside the
  ICE-only drag balance. A modelled result recorded for its own sake, with no event-trap target,
  and never a substitute for the maximum.

### What the straight-line numbers are allowed to mean

`tests/test_scenarios.py` reports both speed quantities separately and asserts that the transient
maximum reaches the 325.8 km/h floor. The previous 3 s deployment variant measured 308.0353 km/h
and failed this floor; the revised 20 s request passed it. GitHub Actions run
[37639816872](https://github.com/Nishanth1812/f1-telemetry/actions/runs/37639816872), on
`527d109`, measured `accelerate_to_speed` with driver upshifts requested at `config.shift_up_rpm`:

- `accelerate_to_speed` reaches 100 km/h in **4.0919 s**, **1.7719 s slower** than the coarse
  2.32 s reference. It is a diagnostic result, not a calibration pass/fail comparison.
- `full_throttle` reaches a **338.4295 km/h transient maximum** during deployment, then settles at
  **307.6027 km/h** in the ICE-only tail. The final five seconds drift by **−0.0661 km/h**.
- MGU-K delivery reaches **350 kW** and the store moves from **4.000 MJ to 0.782 MJ**.

The earlier **6.6598 s** 0–100 result used fixed-time upshifts and is superseded by the measurement
above. These remain engine-state diagnostics before Task 2 longitudinal load-transfer integration,
not calibrated predictions. The 0–100 reference is only a coarse, sampled ~3.7 Hz median; its
±0.30 s is feed quantisation resolution, not a confidence interval or a pass/fail tolerance. The
current result still misses the reference substantially, and the source audit found no matched
official launch target, so the P1 performance gate remains open. The transient maximum passes the
reachability floor; terminal speed has no event-trap target.

Lever experiment (2026-10-08, measured, no spec change): halving the sub-4000 rpm turbo-lag cut
(0.35 → 0.675) leaves 0–100 at 4.0919 s — the launch never drops below 4000 rpm (`idle_rpm`
is the floor and the multiplier bites only strictly below it), so the lever cannot act without
moving `idle_rpm` or the threshold, both separate scope decisions. `shift_up_rpm` 11000
worsens the crossing to 4.1700 s; 12000 is indistinguishable from 12500. The ICE-only tail
is intact (MGU-K zero, 8th gear over the final 41 s). The gap stays at +1.77 s; the torque-curve
shape itself remains the untested rank-1 lever, with no public source curve to calibrate against.

## Phase 1 deterministic scenarios

The kernel scenarios use caller-owned torque histories and fixed `car_spec.yaml` coefficients.
Launch, requested shifts, neutral/coast, MGU-K deployment/recharge and braking are checked through
state direction, transition caps, deterministic replay and all eight invariant checks. Invariant 6
balances the kernel's actual modeled boundary over each sampled interval: chassis and four-wheel
kinetic-energy change against wheel torque work, aero drag and tyre-slip work. It does not treat
crankshaft or store-side electrical power as direct chassis power, since the P1 state omits engine
and motor rotor inertia.

The pinned outputs recorded below are the **last pinned ones, from before the scenario wiring in the
section above** — they describe `full_throttle` as an ICE-only run with no declared launch speed.
They are kept here so the change is visible rather than silently overwritten, and they must not be
quoted as current:

- standing-launch scenario ends at 25.68 km/h after 2.0 s;
- the seven-second `accelerate_to_speed` scenario first crosses 100 km/h at 6.8998 s;
- the six-second full-throttle shift scenario starts at 12 m/s and ends at 179.48 km/h;
- the 59-second `full_throttle` run reached eighth gear, a maximum of 307.4189 km/h and a speed
  change of 0.1448 km/h over its final five seconds, so on that run the maximum and the terminal
  speed were the same number to the digits recorded.

The previous 3 s MGU-K variant was measured in CI at a 308.0353 km/h transient maximum and failed
the 325.8 km/h reachability floor. The engine-state baseline above is reproduced after the Phase 2
configuration edit. It records the 0–100 miss and the deployment-dependent transient separately;
performance calibration remains open. The terminal speed has no event-trap target and is a modelled
result.

Before the engine-state change, RPM was derived from wheel speed and floored at idle except for the
launch override. A 12,000 rpm held override reduced the reported time to about 5.26 s but produced
rear slip ratio above 51. A separate bounded ideal-torque probe estimated 2.896 s without
longitudinal load transfer and 1.612 s with it; those are diagnostic lower-bound probes, not
calibrated predictions. Engine speed now evolves during clutch slip and shift cuts, but the ideal
locked-clutch boundary is simplified and longitudinal load transfer remains open P1 acceptance
work.

The brake probe reported about 1.265 g using caller-supplied 873.7 Nm per wheel. There is no brake
capacity model, so this measures the tire response under that supplied torque and does not establish
F1 brake performance or a stopping distance.

The external references are [FIA 2026 Australian GP race maximum speeds](https://www.fia.com/events/fia-formula-one-world-championship/season-2026/grand-prix-australia/race-qualification) — the source of the 325.8 km/h reachability floor in section 6,
[OpenF1 documentation](https://openf1.org/docs/) and the [`car_data` endpoint for
`session_key=11334`](https://api.openf1.org/v1/car_data?session_key=11334) — the source of the 0-100 reference in section 6,
[Formula 1's 2022 Emilia Romagna GP start analysis](https://www.formula1.com/en/latest/article/tremayne-has-the-advantage-swung-towards-red-bull-after-the-emilia-romagna.pRu5QK7PC2BYp7fT1ohts) — recorded in section 5 as rejected, not used,
[Fastest-lap v0.5 Windows installation](https://fastest-lap.readthedocs.io/en/latest/getting_started/installation.html), and its [Catalunya quickstart](https://fastest-lap.readthedocs.io/en/latest/getting_started/quickstart.html).

The brake scenario applies synthetic signed torque to all four wheels and checks deceleration; it
does not establish an F1 brake capacity or stopping distance. The 2014 solver comparison is
recorded above with its model mismatch, so it does not validate the 2026 coefficients.

## 6. Straight-line reference points

Recorded here, and in `PLAN.md` §11.1, **before** any parameter edit. Fixing a reference before
tuning is what stops tuning from choosing its own pass mark, so the order matters: these two were
fixed while the model still misses both, and no coefficient has been tuned toward either.

**Both are coarse observations, not published performance figures, and neither validates
configuration-matched performance.** A run that reaches one of them has matched a number this
project measured off a decimated feed or a single event's speed table — not the real car. That
distinction is the substance of this section, not a disclaimer on it.

### 0–100 km/h — reference 2.32 s, uncertainty of order ±0.30 s

**Derivation.** The **median of the six quickest clean first-motion-to-100 km/h crossings** in the
2026 Belgian Grand Prix Race, computed from OpenF1 `car_data` for `session_key=11334`:

| driver | 1 | 16 | 3 | 12 | 44 | 6 |
|---|---|---|---|---|---|---|
| crossing (s) | 2.12 | 2.12 | 2.32 | 2.32 | 2.32 | 2.60 |

The median of `2.12, 2.12, 2.32, 2.32, 2.32, 2.60` is **2.32 s**. The quoted **±0.30 s is the
size of the measurement's uncertainty, not a tolerance on the car and not a confidence interval.**

**What this evidence is.** A measurement this project made from a public feed. No organisation
states a 0–100 time for the 2026 car, so there is no published figure behind this number and none
is claimed.

**What limits it, stated so the number is not over-read:**

- **Sampling rate ~3.7 Hz, which dominates the uncertainty.** The OpenF1 documentation gives the
  `car_data` sample rate as "about 3.7 Hz". This is a decimated feed, not the car's own
  telemetry, so the sample interval is ~0.27 s. Both endpoints of the window - first motion and the
  100 km/h crossing - are resolved only to the samples bracketing them, so an individual crossing
  carries up to ~0.27 s of quantisation uncertainty before anything else is counted. Six
  crossings suppress an outlier; they do not reduce quantisation error. A model agreeing with
  2.32 s to inside ±0.30 s has therefore been shown to fall **within the resolution of the feed**,
  which is a coarse check, and this number is a reference rather than a precise acceptance target.
- **Window definition: first motion to 100 km/h *includes* the launch.** The window opens at motion
  onset, so it contains the whole physical launch - clutch take-up, gear engagement, whatever
  wheelspin the driven rear axle does - and excludes **only** the pre-motion delay before the car
  starts moving, that is driver reaction and lights out. It is a launch-and-acceleration figure,
  not a rolling-acceleration one, and it should not be compared against a model that starts already
  moving.
- **Throttle is a power percentage, not a pedal position.** The OpenF1 documentation defines the
  `car_data` `throttle` field as the "percentage of maximum engine power being used", and defines
  `brake` as whether the brake pedal is pressed. So a throttle trace from this feed is **not** a
  pedal trace, and setting it against a model's commanded throttle compares two differently-defined
  channels. The feed reports nothing about what any driver aid was doing, and this document infers
  no mechanism from it: the figure is what one car did on one afternoon.

**Citation.** [OpenF1 documentation](https://openf1.org/docs/) —
<https://api.openf1.org/v1/car_data?session_key=11334>

### Top speed — 325.8 km/h, a transient reachability floor, not a terminal target

The 2026 Australian Grand Prix Race speed table published by the FIA records **325.8 km/h for Ocon
at Intermediate 2**.

**Citation.** [FIA 2026 Australian Grand Prix, Race/Qualifying results](https://www.fia.com/events/fia-formula-one-world-championship/season-2026/grand-prix-australia/race-qualification)

**What this figure is.** One car, at one point on one circuit, in one session, used as a
**reachability floor and nothing more**: a high-speed scenario that never approaches 325.8 km/h
cannot have reproduced something a real 2026 car demonstrably did. A scenario that does reach it
has shown reachability and nothing beyond it.

**What it is not, and why the distinction governs the whole section:**

- **Not a terminal or asymptotic speed, and an event speed trap is not comparable to one.** The trap
  sits part-way down a straight where the car is still accelerating, so the trap value and the
  drag-limited asymptote are different physical quantities. Matching the first is not a validation
  of the second. Any attempt to give this figure a tolerance and treat terminal speed as the thing
  to hit would be comparing across that gap, which is why no tolerance is quoted.
- **Not a target for an exact match.** The observation's own uncertainty - fuel load, track and air
  temperature, wind, track evolution - is not quantified here, so there is nothing to hang a band
  on.
- **Not a season-wide figure.** Other events' speed tables report other numbers; this one is cited
  because it is the reference this project recorded, not because it is representative.
- **The comparable model output is the scenario's maximum speed, not its terminal speed.** This is
  a transient maximum-speed / reachability check. The model output to set against 325.8 km/h is the
  **maximum** speed reached by the high-speed `full_throttle` scenario, including any MGU-K
  deployment inside that scenario. The same run's **terminal/asymptotic** speed is a separate
  result, and it has **no direct event-trap target**: a trap is a mid-straight reading with the car
  still accelerating, so this project holds no published figure that an asymptote should be matched
  against. The terminal speed is still worth reporting — it is a different and interesting number —
  but reporting it is not this check, and it must not be the thing quoted against 325.8 km/h.
- **This checks the combined propulsion/aero scenario, not the drag-limited solve alone.** The speed
  a finite run reaches is set by the whole path together: ICE and MGU-K power through the gearbox,
  the tyre model's traction limit, and the `Cl`/`Cd` curves, all in series. A shortfall against the
  floor is therefore a statement about that combination and does not localise to drag, to power, or
  to grip on its own — a later task has to separate them. `PLAN.md` section 6 still derives an
  asymptotic speed from the drag-limited solve in `car_spec.yaml`; that remains a modelled result
  with no event-trap target attached to it.

### What this changes, and what it does not

**Changed.** The P1 exit gate now has two cited numbers to measure against, fixed before tuning,
and section 5's first bullet is revised: the 2022 Emilia Romagna start figures remain rejected, and
the 2026 Australian GP speed-table entry moves from "not a terminal-speed target" to "used, but
only as a reachability floor" - a different and weaker role than being a target, not a stronger one.

**Not changed.**

- No parameter was edited. `car_spec.yaml` is untouched and no coefficient has been tuned toward
  either reference.
- The 6.8998 s / 307.4189 km/h pair is a historical result from the earlier ICE-only scenario.
  GitHub Actions run [37639816872](https://github.com/Nishanth1812/f1-telemetry/actions/runs/37639816872)
  later measured the configured RPM-threshold policy at 4.0919 s to 100 km/h and a 338.4295 km/h
  transient maximum for `full_throttle`. The acceleration result still misses the coarse reference,
  and no matched official launch target is available, so the P1 performance gate remains open.
- Neither reference is precise enough to certify configuration-matched performance. If a later
  task finds the 0–100 window too tight to separate a real coefficient error from feed
  quantisation, the correct response is to say so and re-derive the reference - not to widen the
  band until the current output fits.

## Phase 2 lateral implementation evidence

The P2 implementation adds planar body motion, four-corner load transfer, load-sensitive combined
slip, relaxation states, Ackermann steering and quasi-static suspension geometry. This records
implementation evidence only; no P2 coefficient was tuned to these outputs.

- The 20 m/s steady-circle scenario requests a 50 m radius and settles within 5% of that radius at
  about 0.76 g. The constant-radius sweep starts from 40–105 m/s (144–378 km/h), coasts for 2.5 s
  to settle lateral response, matches a 200 m radius, stays within declared channel ranges, and
  passes the finite, friction-ellipse, vertical-load-sum, sign and energy checks. Its settled final
  point falls below the 4.0 g historical reference. Pirelli reported 4G lateral acceleration at
  Spa's Pouhon in 2011, with cars at 290 km/h
  ([Pirelli, 2011](https://press.pirelli.com/the-belgian-gran-prix-from-a-tyre-point-of-view/)).
  The reported car/corner/conditions do not match this simulator's synthetic neutral-circle case;
  the earlier above-4 g result was a sideslip transient, not a steady-state match. The figure is
  not a configuration-matched calibration target. P2 calibration remains open until matched
  source data and test conditions are available.
- At 50 m radius and 20 m/s, steering demand changes monotonically as
  `roll_stiffness_front_fraction` moves from 0.3 to 0.7 with the other inputs fixed. The test is a
  balance-sensitivity check, not an understeer-gradient calibration.
- The zero-steer symmetry control disables static camber, camber gain and bump steer. A separate
  paired left/right simulation checks mirrored planar state and wheel loads/forces. Production
  scenarios retain the synthetic suspension coefficients, so zero commanded steer can still have
  toe effects from bump steer; symmetry tolerances are not widened to hide that behavior.
- Wheel camber truth is signed per corner as emitted by the kernel. The `camber` channel range was
  widened to `[-5, 5]` degrees so both sides of the configured signed camber are representable.
- Invariant 6 uses the discrete energy balance over chassis translation, yaw and four wheel spins,
  against wheel-torque work, signed aerodynamic drag and longitudinal/lateral tire slip work. The
  independently recomputed final-interval residual matches the recorded residual on the circle and
  straight-line scenarios.

At this Phase 2 checkpoint, the fixed-time-shift 0–100 km/h result was 6.6598 s against the coarse
2.32 s reference. The later RPM-threshold result is recorded in *Phase 1 scenario wiring* above.
P1 remains open because that result still misses the reference and the synthesized ICE curve
remains uncalibrated. The 338.4295 km/h transient reachability result does not close the acceleration
miss or validate the car model.

### Frozen synthetic targets — lateral-g and thermal windows (SYNTHESISED/SELF-CONSISTENT)

Status: **SYNTHESISED / SELF-CONSISTENT — not validated against a real car.** This subsection
freezes self-consistent model outputs as regression anchors. No coefficient was changed to record
them, `car_spec.yaml` is untouched, and no P1/P2/P3 gate is closed by them.

Method — lateral-g targets:

- Scenarios: `steady_state_circle` and `constant_radius_speed_sweep`, as evidenced in
  `docs/phase2-demo.md` lines 1–75.
- Settlement protocol: each sweep point starts at its listed initial speed, coasts in neutral for
  2.5 s to settle lateral response, then targets a 200 m radius; acceleration, steering and wheel
  loads are the settled mean over the final 150 ms. The car slows during each run, so the tail
  speed differs from the initial speed.
- Coefficient refs (all synthesised, untuned): `tyres.lateral_pacejka` (`b: 9.0`, `c: 1.5`,
  `e: 0.95`, `mu: 1.55`), `tyres.load_sensitivity` (`reference_load_n: 4000.0`, `peak: 0.1`,
  `stiffness: 0.05`), and `aero` (`reference_area_m2: 1.5`, `ride_height_sensitivity: 0.18`,
  synthesised `cl_curve`/`cd_curve` over 0–105 m/s).

Frozen values:

- 200 m sweep endpoint at 105 m/s initial: **3.70 g** settled mean over the final 150 ms after
  the 2.5 s neutral coast (3.697 g in `docs/phase2-demo.md`; tail speed ~85.05 m/s, measured
  radius 199.96 m).
- 50 m circle at 20 m/s: **0.76 g** settled within 5% of the requested radius.

The 4.0 g Pirelli 2011 Pouhon figure is kept as **plausibility-only context**, not a
configuration-matched calibration target. Pirelli reported 4G lateral acceleration at Spa's
Pouhon in 2011 with cars at 290 km/h
([Pirelli, 2011](https://press.pirelli.com/the-belgian-gran-prix-from-a-tyre-point-of-view/)).
Mismatch against the synthetic neutral-circle case:

| Dimension | 4.0 g reference | This simulator |
|---|---|---|
| Car | 2011 F1 car | Synthetic 2026 car, untuned coefficients |
| Corner | Spa Pouhon, banked/cambered real corner | Flat 200 m neutral circle / 50 m steady circle |
| Aero | 2011 aero package | Synthetic `Cl` curve (`car_spec.yaml` aero, lines ~111–127) |
| Tyres | 2011 compounds (also mismatched to 2017/2026 compounds) | Synthetic 2026-compound Pacejka + load sensitivity (`car_spec.yaml` lines ~609–617) |

The earlier above-4 g result was a sideslip transient, not a steady-state match.

0–100 tuning priority order (recorded, no numeric tuning performed here): torque/turbo lag >
ratios/shift points > clutch capacity > longitudinal Pacejka mu/b > wheel inertia; mass/aero
minimal per mass-cancellation.

Synthesised thermal-window estimates (no 2026 measured band exists for any of these; Pirelli's
2026 compound announcement supplies no numeric operating-temperature bands):

- C1 tread: 105–130 °C (carcass band tracked alongside tread, synthesised).
- C3 tread: 85–110 °C (carcass band tracked alongside tread, synthesised).
- C5 tread: 70–95 °C (carcass band tracked alongside tread, synthesised).
- Brake: 500–650 °C.
- Engine: 85–105 °C.
- Gearbox: 85–100 °C.

Disclaimer: no 2026 measured equilibrium band has been matched to this car configuration for any
compound or node; the above are synthesised estimates for wiring/sensitivity use only and do not
close P3-T7 or any P1/P2 gate.

## Phase 3 thermal implementation checkpoint

The simulator now computes lumped tyre, brake, engine and gearbox temperatures and tyre pressure
from scenario energy inputs. Tyre heat uses the kernel's signed contact-patch slip work, summed as
absolute per-step dissipation; brake heat uses brake torque and midpoint wheel speed. Engine heat
uses the existing synthetic fuel-to-shaft efficiency from `car_spec.yaml`. A scenario may declare a
non-negative tyre gas leak rate, which lowers pressure through the ideal-gas state independently of
the fault injector. `thermal_soak` and `brake_duty_cycle` are available in the scenario catalog.

Cooling areas and coefficients, node heat capacities, brake heat fraction, engine coolant share,
gearbox loss share, initial thermal states, tyre volume and gas mass remain illustrative values in
the thermal implementation. No operating-temperature source data has been matched to the car
configuration, and no equilibrium bands have been measured. This checkpoint records model wiring;
it does not close P3 calibration or the P1/P2 calibration gates.

## Official-source review, 7 October 2026

The calibration source authority is FIA and Pirelli primary publications. Pirelli's
[2026 compound announcement](https://press.pirelli.com/the-range-of-compounds-for-the-2026-season-has-been-set/)
(24 November 2025) identifies dry compounds C1–C5 and describes validation with mule cars.
It supplies neither compound operating-temperature bands nor matched launch or lateral-g
measurements. The [Formula 1 tyre page](https://www.pirelli.com/tires/en-us/motorsport/car/formula-1)
describes compound suitability but supplies no numeric operating-temperature bands.

The [2011 Spa publication](https://press.pirelli.com/the-belgian-gran-prix-from-a-tyre-point-of-view/)
supports the historical Pouhon reference above; its car generation and scenario do not match
this configuration. Regulatory limits and tyre-blanket limits cannot supply these missing
performance or equilibrium measurements.

These checked publications do not establish the targets required for tuning. P1 still needs
a measured launch window with matched setup and uncertainty; P2 needs matched cornering data;
P3 needs compound/surface equilibrium bands with temperature measurement semantics. Their
gates remain open, with no new coefficients or acceptance tolerances inferred from this review.
