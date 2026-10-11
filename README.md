# F1 Virtual Telemetry Emulator

A simulation-based Formula-car telemetry and diagnostics platform. A physics model is the virtual
car; a virtual sensor layer turns its state into the channels a real car's electronics would
publish; analytics look for anomalies and faults in those channels; and a dashboard lets an
engineer inspect the car and run controlled virtual tests, with no hardware involved.

```text
SIMULATE THE CAR  ->  EMULATE ITS SENSORS  ->  UNDERSTAND THE CAR
```

What you see in the dashboard:

- **Virtual test** bar: pick a test (standing launch, braking run, constant-radius corner), change
  its one setting and run it. The car drives the test in real time.
- **Live car data**: speed, engine RPM, gear, throttle, steering and the lateral, longitudinal and
  yaw dynamics, straight from the simulated car's sensors.
- **Charts, tyres and brakes**: speed / RPM / throttle traces and per-corner tyre and brake
  readouts, all updating live while the test runs.
- **Model analysis**: warnings when a channel leaves its expected behaviour.
- **Last test result**: peak speed, forward and sideways g, and a plain-language comparison with
  your previous run of the same test.

The car spec is explicitly provisional until it is calibrated in Phase 1, so every number is
"plausible", not "validated". The longer pitch is in `F1_Virtual_Telemetry_Emulator_Idea.md`, the
engineering plan in `PLAN.md`, and the phase breakdown in `PHASES.md`.

## Run it

Prerequisites: Python 3.12 with `uv`, and Node.js 22.12 or newer with npm. From the repository
root, install the locked dependencies:

```powershell
uv sync --frozen
npm ci --prefix web
```

Start the WebSocket server and Vite dashboard in separate terminals. The server starts with the
car at rest; pressing **Run test** in the dashboard makes it drive that test live. (Add `--noise`
to `f1-serve` to stream the old uniform random contract data instead, which only proves the
channel rates.)

```powershell
uv run f1-serve
```

```powershell
npm --prefix web run dev
```

Open <http://localhost:5173>. If `just` is installed, use `just serve` and `just dev` for the two
commands above.

Run the project checks with `just check`, or run `uv run ruff check src tests`,
`uv run basedpyright`, `uv run pytest`, `uv run f1-check-contract`, and
`npm run build --prefix web` individually.

## What is simplified

This is a physics-core telemetry emulator, not a car. To stay buildable and honest about
what it claims, several layers are deliberately simplified:

- **Aero, tyres, brakes and powertrain are synthetic.** The Pacejka coefficients, aero
  curves, gear ratios and torque curve are seeded from published parameter sets and then
  labelled as synthetic in `docs/calibration.md`. They are plausible, not validated
  against a real car.
- **The power curve is uncalibrated.** `torque_curve(rpm)` is synthesised from published
  peak figures plus a turbo-lag multiplier because no public F1 torque curve exists.
- **There is no brake-capacity model.** Caller brake torque is a number, not a limit a
  brake system enforces; braking performance is therefore not an acceptance target.
- **The racing line is QP, not NLP.** The line is a minimum-curvature QP solution with a
  forward-backward speed profile. An NLP "polish" (expected to be worth a few percent of
  lap time) is optional stretch work, not a correctness requirement.
- **Residual analytics are not built yet.** The §9 residual-first detection stack is the
  plan; what exists today is validity checks, the energy invariant, CAN-FD framing, sensor
  fault injection primitives and Parquet round-trip — not the published detection tables.

Nothing above is a performance claim; see `docs/latency.md` for measured latency budgets.
