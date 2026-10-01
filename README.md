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
