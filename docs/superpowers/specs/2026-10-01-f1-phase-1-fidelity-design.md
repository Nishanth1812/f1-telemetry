# Phase 1 fidelity spec — a regulation-faithful generic 2026 car

Date: 2026-10-01. Scope: the remaining `tasks/todo.md` **Task 4** work (P1-T6/T7 and MGU-K
drive), followed by the remaining P1 scenarios and checks, targeting a *generic* 2026 car. This
document sets the design; implementation follows review and approval.

**Regulatory source.** FIA 2026 Formula One Regulations, Section C (Technical), **Issue 20,
5 August 2026** — <https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_c_technical_-_iss_20_-_2026-08-05.pdf>
(listing: <https://www.fia.com/regulation/category/110/technical-regulations>). Page numbers
below are the footer pages, which match the PDF page indices in this document; all were read out
of that PDF for this spec.

---

## 1. Scope

**In:** longitudinal physics only — ICE torque, gearbox, clutch, MGU-K, four wheel rotational
states, and assembly of drive/brake/aero forces into the existing `kernels/longitudinal.py`
integrator. The transmission drives only the two rear wheels. Equal left/right drive torque is a
synthetic differential abstraction, not a team-specific differential model.

**Out:** lateral dynamics, load transfer, combined slip, thermal, track, sensors, analytics (P2+).

**Two targets, kept separate.**

1. **P1 regulatory behavior** — the modeled drivetrain may not do things the 2026 rules prohibit,
   and it must represent required behavior in this slice. This is not a claim of full-car FIA
   compliance; the model omits most of the Technical Regulations.
2. **Performance calibration** — 0–100 km/h and top speed landing in a cited band. This is
   Task 5 (`docs/calibration.md` §5), needs synthetic coefficients, and is *not* a compliance
   claim.

The P1 drivetrain can match the rules and still be badly calibrated. Gate them separately; do not
let a Task 5 coefficient tune be the evidence that a clause is satisfied.

---

## 2. What "generic 2026 car" means

Regulation fixes the envelope; the team fills it. Section C Issue 20 states **team-nominated**
ratios (C9.6.2, p103), **PU-declared** rev limits (C5.14, p76), supplier-published tyre masses
(C4.7, p60), and FIA-measured fuel LHV (C5.2.6, p64). None is a regulation number.

So the generic car is: **every in-scope regulation-determined value represented exactly; every
team-specific value declared synthetic or unavailable.** Task 5 calibrates the synthetic values
against public performance targets, not against private team data.

| Class | Meaning | Example here |
|---|---|---|
| **R** | Direct limit or required behaviour | C9.1.1 rear wheels only |
| **D** | Derived implementation rule (from an R) | one accepted request changes at most one gear |
| **S** | Synthetic / calibration-only | wheel inertia, reverse ratio, generic aero map |
| **U** | Unavailable to this project | named-team gear ratios, torque curve, aero maps |

---

## 3. Verified clause set

| Clause | Page | Rule |
|---|---|---|
| C9.1.1 | 100 | "The transmission may only drive the two rear wheels." |
| C9.1.2 | 100 | No system capable of preventing driven wheels from spinning under power, or of compensating excessive driver torque demand; no wheel-spin notification |
| C9.2.3 | 101 | Fully disengaged = "incapable of transmitting any useable torque" |
| C9.2.5 | 101 | Clutch demand "expressed … as torque at the rear axle by applying a gain of 5200Nm / 90%"; control error ≤ ±150 Nm at rear axle, except first 85 ms of a launch step; engagement solely driver-controlled, exceptions listed |
| C9.2.6 | 101–102 | Paddle returns to rest within 50 ms; input-to-output delay ≤ 50 ms |
| C9.6.1 | 103 | 8 forward ratios; no CVT |
| C9.6.2 | 103 | Ratios **team-nominated** |
| C9.7 | 104 | "All cars must be able to be driven in reverse by the driver at any time" |
| C9.8.1 | 104 | "Automatic gear changes are considered a driver aid and are therefore not permitted" |
| C9.8.3 | 104 | Each change separately driver-initiated; one at a time; fixed minimum gear while moving |
| C9.8.4 | 104 | Up change ≤ 200 ms, down ≤ 300 ms, ≤ 80 ms request→disengage |
| C9.9.1–2 | 105 | No torque transfer slower→faster wheel; no front-axle torque transfer |
| C5.2.3 | 64 | Fuel energy flow ≤ 3000 MJ/h, ≤ 550 MJ/h/cylinder |
| C5.2.4 | 64 | Below 10 500 rpm: `EF = 0.27·N + 165` MJ/h |
| C5.2.5 | 64 | Partial-load fuel-energy-flow limit as a function of engine power |
| C5.2.7 | 64 | ERS-K absolute electrical DC power ≤ 350 kW |
| C5.2.8 | 64 | Speed-dependent propulsion limit: `1800−5v` <340 km/h, `6900−20v` to 345 km/h, 0 above; Overtake `7100−20v` <355 km/h |
| **C5.2.9** | 64 | ES max−min state of charge ≤ **4 MJ** — the usable window |
| **C5.2.10** | 64–65 | Recharge baseline ≤ **8.5 MJ/lap** at the CU-K HV DC bus; event-dependent reductions include 7 MJ and a qualifying floor of **4 MJ**, with a conditional **+0.5 MJ** allowance |
| **C5.2.11** | 65 | MGU-K mechanical torque ≤ **500 Nm, referenced to crankshaft speed** |
| **C5.2.12** | 65 | From a grid standing start, MGU-K only above **50 km/h**; positive torque below 50 only when the FIA Standard ECU mandates minimum acceleration |
| C5.12.6–7 | 74 | ERS-K power-reduction rate limits, with gearshift and <210 km/h exceptions |
| C5.18.2 | 78 | MGU-K rotating parts permanently linked to the ICE at a **fixed** ratio to the crankshaft |
| C5.18.4 | 78 | Optional torque-limiting device may act only above 520 Nm at crankshaft speed; this is not a MGU-K output cap |
| C11.1.1 | 116 | Rear brakes ≥ 2500 Nm per wheel without PU/MGU-K assistance |
| C11.4.1 | 117 | No braking system designed to prevent wheels locking |

---

## 4. Current gaps

| # | Gap | Class |
|---|---|---|
| G1 | `gearbox.py:140` shifts on rpm alone — **C9.8.1 violation** | R |
| G2 | `_check_state` refuses gear outside `1..n`, while telemetry defines reverse and neutral states — **C9.7 reverse capability unmet** | R |
| G3 | `clutch_torque_capacity_nm: 3000` is a synthetic output clamp. C9.2.5 defines driver-requested rear-axle clutch torque, not a mechanical clutch capacity | R/S |
| G4 | MGU-K limits and deployment curves are parsed into `KernelConfig`, but no physics function applies them; no MGU-K crankshaft coupling or energy state exists | R |
| G5 | `channels.yaml:291` and `testing/records.py:66` encode `-1/0/1..8`; the integer codes are simulator conventions, while `provenance: fia_limit` overstates what FIA regulates | D |
| G6 | `invariants.py:389` flags any gear decrease; this rejects legal downshifts | D |
| G7 | No wheel angular-velocity state; `forces.step_forces` receives circumferential wheel speed and `simulate` receives net force as input | D |
| G8 | `rev_limit_rpm: 13000` is a synthetic PU-declared value, not a fixed FIA limit. C5.14 constrains how declared limits may vary; C5.18.5 caps MGU-K relative speed at 60 000 rpm | S/R |
| G9 | `driver_reference_mass_kg` cites C4.5 instead of C4.5.2; mass minimums also require the Nominal Tyre Mass and session context (C4.1, C4.7) | R |
| G10 | Fuel-energy-flow inputs, ERS speed/power curves, SOC window, and recharge baseline are parsed, but not enforced by P1 physics; C5.2.5 partial-load enforcement also needs a declared fuel/engine efficiency mapping | R/S |

---

## 5. Adjudications

**G1 — shifts.** rpm thresholds stay as the *driver's* shift-point schedule; the gearbox must
consume an explicit request. A trace driven by rpm alone must not change gear.

**G2 — neutral and reverse.** Keep the channel's simulator state domain `{-1, 0, 1..8}`; add a
synthetic reverse ratio and never index the forward table with 0 or −1. Reverse drivability is
required by C9.7. Neutral is an existing model state and appears in regulatory provisions, but
those clauses do not independently require the car to select neutral in ordinary operation.

**G3 — clutch control.** Retain the approved caller-owned clutch input, but represent it as driver
paddle demand that produces rear-axle torque demand using C9.2.5's 5200 Nm / 90% travel gain over
the 5–95% range. Full disengagement transmits zero torque. Remove 3000 Nm as an output clamp: it
is a synthetic value that contradicts the FIA torque-demand model when active. If a physical
clutch-capacity parameter remains useful later, separate and label it as synthetic, then calibrate
it so it does not silently override driver demand. Check the ±150 Nm tracking band after the first
85 ms of a launch step.

**G4 — MGU-K.** Join MGU-K torque at the crankshaft through its fixed ratio (C5.18.2/.3), which
is the model's inferred upstream boundary before the clutch and gearbox. Enforce 350 kW electrical
DC (C5.2.7), the speed-dependent C5.2.8 propulsion cap, and 500 Nm mechanical torque referenced
to crankshaft speed (C5.2.11). C5.18.4's 520 Nm value is only the threshold above which an
optional transient torque-limiting device may act, not a torque cap. Apply the 50 km/h rule only
to a declared grid standing start and allow the FIA-mandated minimum-acceleration exception
(C5.2.12). MGU-K wheel force reaches the rear axle through the common drivetrain; it does not
drive the front axle (C9.1.1).

**Energy.** Enforce the 4 MJ usable SOC window (C5.2.9). `car_spec.yaml` already records the
8.5 MJ/lap baseline (C5.2.10); represent the conditional 7 MJ limit, 4 MJ qualifying floor, and
+0.5 MJ allowance as event-conditioned values. The Issue 20 PDF visually strikes the prior 5 MJ
qualifying floor and replaces it with 4 MJ.

**G6/G7.** Fix the invariant to permit requested legal downshifts; add four wheel `ω` states and
close `I_w·dω/dt = T_drive − T_brake − Fx·r`. Split drive torque between rear wheels only; do not
model torque transfer from a slower wheel to a faster one (C9.9.1). No traction-control or
anti-lock feedback (C9.1.2, C11.4.1). Keep the axle load split static in P1; P2 owns load transfer.

---

## 6. Slices and acceptance criteria

**S0 Contract corrections.** Acceptance: `provenance_audit()` clean; C5.2.10 represents the
8.5 MJ/lap baseline plus an event-specific limit, rather than a universal cap; C4.5→C4.5.2 and
the mass minimum includes Nominal Tyre Mass/session context; the C5.2.11 quote includes its
crankshaft reference; gear integer encodings are marked as simulator conventions. Existing
non-physics golden fixtures remain unchanged. RPM-triggered shifts are an intentional behavior
change; fixed-input, no-shift cases remain stable.

**S1 Neutral and reverse.** Acceptance: gear 0 transmits exactly 0; gear −1 transmits negative
torque; forward table never indexed by 0/−1; fixed-gear forward torque scaling remains unchanged.

**S2 Driver inputs, clutch and shifts.** Acceptance: clutch paddle demand maps to rear-axle torque
using C9.2.5's gain, fully disengaged transmits zero, and output tracks within ±150 Nm after the
first 85 ms of a launch step. RPM alone changes no gear; each up/down request starts one shift, no
new request is accepted before completion, and the moving minimum selectable gear does not
change. The current shared synthetic `shift_time_s` remains ≤200 ms (therefore within both
direction limits); request-to-original-gear-disengagement is ≤80 ms. Over-rev protection may
reject engagement with no more than 50 ms delay, after which a fresh request is required
(C9.2.5, C9.8.3–4).

**S3 Power unit limits.** Acceptance: ICE fuel flow stays within C5.2.3/.4 and the C5.2.5
partial-load curve; any efficiency used to map fuel flow to shaft power is labeled synthetic.
MGU-K gets no positive torque below 50 km/h during a declared grid standing start except when
mandated for minimum acceleration; DC power never exceeds 350 kW or the C5.2.8 speed cap;
crankshaft-referenced torque never exceeds 500 Nm; the optional 520 Nm transient limiter is not
treated as a cap; relative MGU-K part speed stays within 60 000 rpm (C5.18.5).

**S4 Energy bookkeeping.** Acceptance: SOC swing ≤4 MJ; apply the configured event recharge limit
and 8.5 MJ baseline without embedding a single universal race limit. A straight-line scenario can
check SOC use; lap recharge enforcement remains pending a lap/harvesting scenario.

**S5 Wheel state and force assembly.** Acceptance: four `ω` states close the slip→force→chassis
loop; only rear wheels receive drive torque (C9.1.1); no torque reduction on wheel spin (C9.1.2);
left/right axle brake torques remain symmetric (C11.1.2); lock and spin can arise without
intervention. `STATE_SIZE` and index constants are updated; semi-implicit ordering is pinned.

**S6 Scenario closure → Task 5.** Acceptance: `accelerate_to_speed` and `full_throttle` drive the
full path from fixed initial conditions; ICE and MGU-K powers are recorded separately; identical
inputs produce byte-identical output.

**S7 Calibration and P1 exit.** Acceptance: choose and cite public 0–100 km/h and top-speed
targets before tuning; keep the ICE curve and team-specific inputs labeled synthetic; check
invariants 1, 3, 6, and 7 on real scenario runs; record the optional `fastest-lap` comparison or
its build limitation. Investigate the existing energy invariant's
`scale = max(|dKE/dt|, 1.0)` floor before defining its start-up tolerance.

---

## 7. Uncertain choices

1. Shift request input: use separate up/down driver commands plus a neutral request; the exact
   telemetry channel shape can be settled in the implementation plan.
2. Reverse ratio, equal rear drive torque split, brake bias, and `wheel_inertia_kg_m2` are
   synthetic; wheel inertia may be seeded from the `PLAN.md` §4 band (0.5–1.2 kg·m²).
3. Energy invariant: `PLAN.md` §6 says fuel power, which needs a synthetic efficiency; recommend
   shaft power for the blocking check and a separate C5.2.3/5.2.4 limit assertion.
4. C5.2.8.iii/iv (power-limited sectors and Low-Grip DOC-111) are not modeled in P1; P1 runs are
   not judged against them.
