// P0 full-rate batch repair - red tests for the browser half of the slice.
// Runs on Node's built-in test runner with no framework and no new dependency.
//
// The tested modules are real TypeScript. They are loaded through a test-only
// loader that transpiles each .ts source to a `data:` URL and rewrites relative
// `from './x'` specifiers to the transpiled dependency's data URL, so the code
// under test is the shipped source, not a copy of it.
//
// NOTE on the transpiler: the installed `typescript@7.0.2` is the native
// preview whose public entry (`./lib/version.cjs`) exports only `version`, so
// `ts.transpileModule` does not exist. `rolldown/experimental`'s `transform`,
// already installed as a Vite dependency, is the transpiler used below. If a
// `ts.transpileModule` ever reappears, the single `transform` call in
// `dataUrl` is the only thing to swap.

import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { dirname, resolve as resolvePath } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import test from 'node:test';
import { transform } from 'rolldown/experimental';

const SRC = resolvePath(dirname(fileURLToPath(import.meta.url)), '..', 'src');
const MODULE_FROM = /(from\s*)(['"])([^'"]+)\2/g;
const dataUrls = new Map();

async function dataUrl(path) {
  const source = path.endsWith('.ts') ? path : `${path}.ts`;
  const cached = dataUrls.get(source);
  if (cached !== undefined) {
    return cached;
  }
  const { code } = await transform(source, readFileSync(source, 'utf8'), { lang: 'ts' });
  const specifiers = [...code.matchAll(MODULE_FROM)];
  const resolved = new Map();
  for (const match of specifiers) {
    const specifier = match[3];
    resolved.set(
      specifier,
      specifier.startsWith('.')
        ? await dataUrl(resolvePath(dirname(source), specifier))
        : pathToFileURL(createRequire(import.meta.url).resolve(specifier)).href,
    );
  }
  const rewritten = code.replace(MODULE_FROM, (match, from, quote, specifier) => {
    return `${from}${quote}${resolved.get(specifier) ?? match.slice(from.length)}${quote}`;
  });
  const url = `data:text/javascript;base64,${Buffer.from(rewritten).toString('base64')}`;
  dataUrls.set(source, url);
  return url;
}

async function loadTs(relativePath) {
  return import(await dataUrl(resolvePath(SRC, relativePath)));
}

const { decodeFrame } = await loadTs('telemetry/types.ts');
const { TraceHistory } = await loadTs('telemetry/history.ts');
const { CHANNELS } = await loadTs('generated/channels.ts');
const { useTelemetryStore } = await loadTs('store/telemetryStore.ts');

// ---------------------------------------------------------------------------
// 1. decodeFrame preserves the optional sample batch, and legacy frames
//    (no batch) decode exactly as before.
//
// Wire shape under test: an OPTIONAL `samples` array on a frame, each entry
// holding `time_us` plus only the channels actually sampled at that instant.
// ---------------------------------------------------------------------------

test('decodeFrame preserves a full-rate sample batch with exact timestamps and values', () => {
  const frame = decodeFrame({
    time_us: 33_334,
    channels: { engine: 2, slow: 0.5 },
    samples: [
      { time_us: 30_000, channels: { engine: 1 } },
      { time_us: 33_334, channels: { engine: 2, slow: 0.5 } },
    ],
  });

  assert.notEqual(frame, null, 'a frame carrying a well-formed batch must decode');
  assert.equal(frame.time_us, 33_334);
  assert.deepEqual(frame.channels, { engine: 2, slow: 0.5 });
  // Per-sample membership must survive exactly: engine at both instants, slow
  // only where it was due. A slow channel must not be handed a fake repeated
  // sample, and a fast one must not be dropped.
  assert.deepEqual(frame.samples, [
    { time_us: 30_000, channels: { engine: 1 } },
    { time_us: 33_334, channels: { engine: 2, slow: 0.5 } },
  ]);
});

test('decodeFrame leaves a legacy no-batch frame unchanged', () => {
  const frame = decodeFrame({
    time_us: 1_000,
    channels: { engine: 1, slow: 0.5 },
    events: [{ kind: 'gear_shift', time_us: 1_000, channel: 'engine', severity: 0.4 }],
  });

  assert.notEqual(frame, null);
  assert.equal(frame.time_us, 1_000);
  assert.deepEqual(frame.channels, { engine: 1, slow: 0.5 });
  assert.equal(frame.samples, undefined, 'a frame without samples must not grow one');
  assert.deepEqual(frame.events, [
    { kind: 'gear_shift', time_us: 1_000, channel: 'engine', severity: 0.4 },
  ]);
});

// ---------------------------------------------------------------------------
// 2. Malformed batches are rejected outright (null), while unusable channel
//    values inside a batch are filtered, exactly as on the snapshot.
// ---------------------------------------------------------------------------

// Strict flat wire: `samples` is an array of frames, never nested batches, and
// every timestamp it carries must be finite, at or before the enclosing frame,
// and in nondecreasing order.
const BATCH_REJECTIONS = [
  ['samples is not an array', { time_us: 10, channels: { engine: 1 }, samples: { 0: 1 } }],
  ['samples is a number', { time_us: 10, channels: { engine: 1 }, samples: 7 }],
  ['a batch entry is null', { time_us: 10, channels: { engine: 1 }, samples: [null] }],
  ['a batch entry is a number', { time_us: 10, channels: { engine: 1 }, samples: [3] }],
  ['a batch entry has no time_us', { time_us: 10, channels: { engine: 1 }, samples: [{ channels: { engine: 1 } }] }],
  ['a batch entry nests its own samples', {
    time_us: 10,
    channels: { engine: 1 },
    samples: [{ time_us: 5, channels: { engine: 1 }, samples: [{ time_us: 1, channels: { engine: 1 } }] }],
  }],
  ['a batch entry has no channels', { time_us: 10, channels: { engine: 1 }, samples: [{ time_us: 5 }] }],
  ['a batch entry has array channels', { time_us: 10, channels: { engine: 1 }, samples: [{ time_us: 5, channels: [1] }] }],
  ['a batch entry has scalar channels', { time_us: 10, channels: { engine: 1 }, samples: [{ time_us: 5, channels: 7 }] }],
  ['a NaN batch timestamp', { time_us: 10, channels: { engine: 1 }, samples: [{ time_us: Number.NaN, channels: { engine: 1 } }] }],
  ['an infinite batch timestamp', { time_us: 10, channels: { engine: 1 }, samples: [{ time_us: Number.POSITIVE_INFINITY, channels: { engine: 1 } }] }],
  ['a batch timestamp after the enclosing frame', { time_us: 10, channels: { engine: 1 }, samples: [{ time_us: 11, channels: { engine: 1 } }] }],
  ['a decreasing batch pair', {
    time_us: 10,
    channels: { engine: 1 },
    samples: [{ time_us: 9, channels: { engine: 1 } }, { time_us: 0, channels: { engine: 1 } }],
  }],
  ['frame channels are an array', { time_us: 10, channels: [1], samples: [{ time_us: 5, channels: { engine: 1 } }] }],
];

test('decodeFrame rejects a malformed sample batch', () => {
  for (const [reason, payload] of BATCH_REJECTIONS) {
    assert.equal(decodeFrame(payload), null, `must reject: ${reason}`);
  }
});

test('decodeFrame accepts an increasing batch pair at or before the frame', () => {
  const frame = decodeFrame({
    time_us: 10,
    channels: { engine: 1 },
    samples: [{ time_us: 0, channels: { engine: 1 } }, { time_us: 9, channels: { engine: 1 } }],
  });

  assert.notEqual(frame, null, '0 then 9 under frame 10 is ordered and must decode');
  assert.deepEqual(frame.samples, [
    { time_us: 0, channels: { engine: 1 } },
    { time_us: 9, channels: { engine: 1 } },
  ]);
});

test('decodeFrame filters nonfinite channel values out of a batch without rejecting it', () => {
  const frame = decodeFrame({
    time_us: 10,
    channels: { engine: 1 },
    samples: [
      { time_us: 5, channels: { engine: Number.NaN, slow: 1 } },
      { time_us: 10, channels: { engine: 1, slow: Number.POSITIVE_INFINITY } },
    ],
  });

  assert.notEqual(frame, null, 'a usable channel keeps a poison sibling from voiding the frame');
  assert.deepEqual(frame.samples, [
    { time_us: 5, channels: { slow: 1 } },
    { time_us: 10, channels: { engine: 1 } },
  ]);
});

// ---------------------------------------------------------------------------
// 3. Decimation must respect chronological ring-buffer order.
//
// Hand-derived: capacity 3, pushed (t,v) = (0,0) (1,1) (2,2) (3,3) (4,4).
// Surviving samples are t=2,3,4 with values 2,3,4. fill() with a limit of 2
// therefore keeps one bucket spanning all three and emits its extrema in
// chronological order: t=2 value 2 (the minimum), then t=4 value 4 (the
// maximum). Emitting t=4 before t=2 draws the trace backwards in time.
// ---------------------------------------------------------------------------

test('TraceHistory fills decimated points in chronological order after the ring wraps', () => {
  const history = new TraceHistory(3);
  for (let i = 0; i <= 4; i += 1) {
    history.push({ time_us: i, channels: { engine: i } });
  }

  const times = new Float64Array(8);
  const values = new Float32Array(8);
  const count = history.fill('engine', 2, times, values);

  assert.equal(count, 2);
  assert.deepEqual(Array.from(times.slice(0, count)), [2, 4]);
  assert.deepEqual(Array.from(values.slice(0, count)), [2, 4]);
});

// ---------------------------------------------------------------------------
// 4. A low-rate channel receives only its own samples.
//
// Hand-derived: `imu` is sampled at t=0, 5000 and 10000 while `engine` is
// sampled only at t=0. Filling `engine` must yield exactly one point, at
// t=0. Three points (with NaN gaps at 5000 and 10000) or repeated values at
// another channel's instants would both be wrong: the browser would be
// drawing samples that were never taken.
// ---------------------------------------------------------------------------

test('TraceHistory gives a sparse channel only its own real samples', () => {
  const history = new TraceHistory(8);
  history.push({ time_us: 0, channels: { engine: 100, imu: 1 } });
  history.push({ time_us: 5_000, channels: { imu: 2 } });
  history.push({ time_us: 10_000, channels: { imu: 3 } });

  const times = new Float64Array(8);
  const values = new Float32Array(8);

  const engineCount = history.fill('engine', 8, times, values);
  assert.equal(engineCount, 1, 'engine was sampled once, so it has one point');
  assert.equal(times[0], 0);
  assert.equal(values[0], 100);

  const imuCount = history.fill('imu', 8, times, values);
  assert.equal(imuCount, 3, 'every imu instant is kept');
  assert.deepEqual(Array.from(times.slice(0, imuCount)), [0, 5_000, 10_000]);
  assert.deepEqual(Array.from(values.slice(0, imuCount)), [1, 2, 3]);
});

test('TraceHistory retains sixty seconds at the generated speed rate', () => {
  const rate = CHANNELS.speed.rateHz;
  const capacity = Math.ceil(rate * 60) + 1;
  const history = new TraceHistory();
  for (let i = 0; i < capacity + rate; i += 1) {
    history.push({ time_us: i * 1e6 / rate, channels: { speed: i } });
  }
  const times = new Float64Array(capacity + 1);
  const values = new Float32Array(capacity + 1);
  const count = history.fill('speed', capacity + 1, times, values);
  assert.equal(count, capacity);
  assert.equal(times[count - 1] - times[0], 60e6);
});

test('TraceHistory discards the previous timeline when replay rewinds', () => {
  const history = new TraceHistory(8);
  history.push({ time_us: 100, channels: { speed: 2 } });
  history.push({ time_us: 0, channels: { speed: 1 } });
  const times = new Float64Array(8);
  const values = new Float32Array(8);
  assert.equal(history.fill('speed', 8, times, values), 1);
  assert.equal(times[0], 0);
  assert.equal(history.latestTimeUs(), 0);
});

test('store keeps full-rate samples in history without retaining them in the UI snapshot', () => {
  const store = useTelemetryStore.getState();
  store.markConnecting();
  store.applyFrame({
    time_us: 20_000,
    channels: { speed: 2 },
    samples: [
      { time_us: 10_000, channels: { speed: 1 } },
      { time_us: 20_000, channels: { speed: 2 } },
    ],
  });

  const state = useTelemetryStore.getState();
  assert.equal(state.frame?.samples, undefined, 'the render-facing snapshot omits the full-rate batch');
  assert.deepEqual(state.frame?.channels, { speed: 2 }, 'the latest snapshot value remains available');
  assert.equal(state.diagnostics?.channels.speed.count, 2, 'both actual batch samples are counted');

  const times = new Float64Array(4);
  const values = new Float32Array(4);
  const count = state.history.fill('speed', 4, times, values);
  assert.equal(count, 2, 'both actual batch samples are retained for traces');
  assert.deepEqual(Array.from(times.slice(0, count)), [10_000, 20_000]);
  assert.deepEqual(Array.from(values.slice(0, count)), [1, 2]);
});
