# f1-telemetry

Local F1 telemetry dashboard. Phase 0 streams synthetic data from `channels.yaml`; the car
spec is explicitly provisional until it is calibrated in Phase 1.

## Run the Phase 0 demo

Prerequisites: Python 3.12 with `uv`, and Node.js 22.12 or newer with npm. From the repository
root, install the locked dependencies:

```powershell
uv sync --frozen
npm ci --prefix web
```

Start the WebSocket server and Vite dashboard in separate terminals:

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
