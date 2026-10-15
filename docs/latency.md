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
