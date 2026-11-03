# Latency / bundle baseline

Measured state of the web build (`npm run build --prefix web`, `tsc -b && vite build`)
on Node 24.11.1 / npm 11.6.1. This file is the home for measured bundle/latency wins;
update it whenever build output or bundle size changes.

## Baseline (no changes applied this round)

| Metric | Value | Notes |
|---|---|---|
| Build time | ~8.8 s | `tsc -b && vite build`, warm cache |
| JS bundle | 355,140 bytes (~347 KB) | `index-C3H_6lC8.js`, **single chunk** |
| CSS bundle | 29,764 bytes (~29 KB) | `index-DjV3afdT.css` |
| Entry chunks | 1 JS + 1 CSS | No vendor/app split |

The JS bundle contains the application code together with `react`, `react-dom` and
`zustand` in one chunk. Because nothing code-splits, any app change invalidates the
whole bundle for returning visitors.

## Dependency audit (react / zustand)

All three runtime dependencies are imported by shipped code and are locked in
`package-lock.json` exactly as declared in `web/package.json`:

- `react` 19.3.0 — imported across components/hooks
- `react-dom` 19.3.0 — `main.tsx`
- `zustand` 5.0.15 — `web/src/store/labStore.ts`, `web/src/store/telemetryStore.ts`

No unused dependency to remove; versions match the lockfile, so `npm ci` stays
consistent. Do not change declared versions without also regenerating the lockfile
(which is out of scope for this review).

## TypeScript settings

`web/tsconfig.*.json` already uses `noEmit: true`, `composite: true`,
`skipLibCheck: true`, `isolatedModules: true`, and strict null/unused checks.
`tsc -b` typechecks without emitting, so there is no tsc build output to trim.
No change needed.

## Codegen drift check

`uv run f1-check-contract` (`just codegen-check`) is **green**: all three generated
artifacts (`src/f1telemetry/generated/channels.py`, `parquet_schema.py`,
`web/src/generated/channels.ts`) are byte-identical to what `channels.yaml` emits.
No regeneration needed; no contract change this round.

## Recommended follow-up (out of scope for this review)

The one lever that would materially trim build output is **vite manual chunking** in
`web/vite.config.ts` (not owned by this review, so left untouched). Add a
`build.rollupOptions.output.manualChunks` that splits vendor from app code, e.g.:

```ts
export default defineConfig({
  plugins: [react()],
  build: {
    target: 'es2022',
    rollupOptions: {
      output: {
        manualChunks: (id) => {
          if (id.includes('node_modules')) {
            if (/react|zustand/.test(id)) return 'vendor';
            return 'vendor-other';
          }
        },
      },
    },
  },
});
```

Expected effect: `react` + `react-dom` + `zustand` move into a separately cached
`vendor.js`, leaving a much smaller app chunk that invalidates only on app changes.
Re-run `npm run build --prefix web` after applying and record the new chunk sizes
here.

## Verification results (this round)

- `uv run ruff check src tests` — pass
- `uv run basedpyright` — 0 errors, 0 warnings, 0 notes
- `npm run build --prefix web` — pass
- `uv run f1-check-contract` — green (generated code up to date)
- Web tests — `telemetry.test.mjs`: 10/10 pass. `dashboard-smoke.mjs` fails only
  because no Chrome binary is available in this environment (pre-existing, unrelated
  to code).

## Backend / server pump (P0-T10)

Measured per-frame cost of the 30 Hz loopback pump with one summary client
connected, on the committed build:

| Metric | Value | Notes |
|---|---|---|
| `Frame.to_wire()` (detailed + snapshot) | ~26 us/frame | builds the wire dict per payload shape |
| `json.dumps` (both payloads) | ~274 us/frame | dominant cost; compact separators, `allow_nan=False` |
| channels dict copy | ~4.9 us/frame | `{name: float(v) for ...}` over ~70 channels |

Optimizations applied to `src/f1telemetry/server.py`:

- `_encode()` helper centralizes `json.dumps(..., separators=(",", ":"),
  allow_nan=False)` so the transport has one encoder, used for frames, scenario
  results, and scenario errors. No temp arrays, no duplicated encoder arguments.
- `_broadcast()` only encodes the detailed payload when a detailed client is
  connected, and only encodes the snapshot payload when a summary client is - at
  most one encode per distinct shape per frame instead of always serialising both.
- Single-dashboard fast path: one delivery sends directly without
  `asyncio.gather`, avoiding task creation and result tuples.
- Client set is snapshotted once into a frozen tuple per frame, so a connection
  arriving or dropping mid-send cannot change who this frame went to.

Guards preserved (unchanged):

- Loopback-only bind check (`_LOOPBACK_HOSTS`) - server refuses non-loopback hosts.
- `scenario_lock` (`asyncio.Lock`) - scenario runs remain single-threaded through
  the lock; `telemetry_detail` still mutates `_detailed_clients` outside the lock.
- `allow_nan=False` NaN guard - invalid JSON tokens never reach the wire.

Server can be restarted with `just serve` (or `uv run f1-serve`) at
`ws://localhost:8765/ws`; default endpoint matches `web/src/telemetry/config.ts`.
