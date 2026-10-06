// Dashboard smoke: native Chrome over CDP, Node 22 builtins only, for
// GitHub Actions Ubuntu. Fails hard on missing Chrome/environment.
// Env: SMOKE_ARTIFACT_DIR (default artifacts/dashboard), DASHBOARD_URL
// (default http://127.0.0.1:4173), CHROME_BIN, EXPECTED_RATE_CHANGE=channel:Hz.

import assert from 'node:assert/strict';
import { execSync, spawn } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, resolve as resolvePath } from 'node:path';
import { fileURLToPath } from 'node:url';
import { transform } from 'rolldown/experimental';

const SRC = resolvePath(dirname(fileURLToPath(import.meta.url)), '..', 'src');
const DASHBOARD_URL = process.env.DASHBOARD_URL ?? 'http://127.0.0.1:4173';
const ARTIFACT_DIR = process.env.SMOKE_ARTIFACT_DIR ?? 'artifacts/dashboard';
const CDP_TIMEOUT_MS = 5_000;
const READY_TIMEOUT_MS = 30_000;
const WINDOW_MS = 20_000;
const TOLERANCE = 0.05;

// Generated schema via the build's own TS transform; channels.ts has no
// runtime imports, so one transform + import is enough.
const channelsPath = resolvePath(SRC, 'generated', 'channels.ts');
const { code } = await transform(channelsPath, readFileSync(channelsPath, 'utf8'), { lang: 'ts' });
const { CHANNELS } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const expected = new Map(Object.entries(CHANNELS).map(([n, s]) => [n, s.rateHz === null ? null : Number(s.rateHz)]));

let edited = null;
if (process.env.EXPECTED_RATE_CHANGE) {
  const i = process.env.EXPECTED_RATE_CHANGE.lastIndexOf(':');
  edited = { channel: process.env.EXPECTED_RATE_CHANGE.slice(0, i), hz: Number(process.env.EXPECTED_RATE_CHANGE.slice(i + 1)) };
  assert.ok(edited.channel && Number.isFinite(edited.hz) && edited.hz > 0, 'EXPECTED_RATE_CHANGE must be channel:Hz');
}

function findChrome() {
  for (const c of [process.env.CHROME_BIN, '/usr/bin/google-chrome', '/usr/bin/google-chrome-stable']) {
    if (c && existsSync(c)) return c;
  }
  return null;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Minimal CDP: outstanding request Map keyed by id; exception list; timeout.
const cdpState = { nextId: 1, pending: new Map(), exceptions: [], closed: false };
let cdpWs = null;

function cdpCall(method, params = {}, timeoutMs = CDP_TIMEOUT_MS) {
  return new Promise((resolve, reject) => {
    if (cdpState.closed) return reject(new Error('CDP closed'));
    const id = cdpState.nextId++;
    const timer = setTimeout(() => {
      cdpState.pending.delete(id);
      reject(new Error(`CDP ${method} timeout ${timeoutMs} ms`));
    }, timeoutMs);
    cdpState.pending.set(id, { resolve, reject, timer });
    cdpWs.send(JSON.stringify({ id, method, params }));
  });
}

function cdpWire(ws) {
  cdpWs = ws;
  ws.addEventListener('message', (event) => {
    const msg = JSON.parse(typeof event.data === 'string' ? event.data : event.data.toString());
    if (msg.id !== undefined && cdpState.pending.has(msg.id)) {
      const { resolve, reject, timer } = cdpState.pending.get(msg.id);
      cdpState.pending.delete(msg.id);
      clearTimeout(timer);
      msg.error ? reject(new Error(msg.error.message)) : resolve(msg.result);
    } else if (msg.method === 'Runtime.exceptionThrown') {
      cdpState.exceptions.push(msg.params?.exceptionDetails?.text ?? 'exception');
    }
  });
  ws.addEventListener('close', () => {
    cdpState.closed = true;
    for (const [, p] of cdpState.pending) {
      clearTimeout(p.timer);
      p.reject(new Error('CDP closed'));
    }
    cdpState.pending.clear();
  });
}

async function evalJson(expression) {
  const r = await cdpCall('Runtime.evaluate', { expression, returnByValue: true });
  if (r.exceptionDetails) throw new Error(`page eval failed: ${r.exceptionDetails.text}`);
  return JSON.parse(r.result.value);
}

let chrome = null;
let profileDir = null;

try {
  const chromeBin = findChrome();
  assert.ok(chromeBin, 'Chrome not found: set CHROME_BIN or install /usr/bin/google-chrome[-stable]');

  profileDir = mkdtempSync(resolvePath(tmpdir(), 'f1-smoke-'));
  chrome = spawn(chromeBin, [
    '--headless=new', '--disable-gpu', '--no-sandbox', '--disable-dev-shm-usage',
    '--remote-debugging-port=0', `--user-data-dir=${profileDir}`, '--no-first-run', DASHBOARD_URL,
  ], { stdio: ['ignore', 'ignore', 'pipe'] });
  let stderr = '';
  chrome.stderr.on('data', (c) => { if (stderr.length < 4096) stderr += c; });

  // Chrome writes the chosen port + ws path to DevToolsActivePort in the profile.
  let port = null;
  const portDeadline = Date.now() + 15_000;
  while (Date.now() < portDeadline && port === null) {
    try {
      port = Number(readFileSync(resolvePath(profileDir, 'DevToolsActivePort'), 'utf8').split('\n')[0]) || null;
    } catch { /* not yet */ }
    if (port === null) await sleep(200);
  }
  assert.ok(port, `Chrome did not publish a CDP port. stderr: ${stderr.slice(-400)}`);

  let target;
  const listDeadline = Date.now() + 10_000;
  while (target === undefined && Date.now() < listDeadline) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
      target = list.find((t) => t.type === 'page' && t.url.startsWith(DASHBOARD_URL)) ?? list.find((t) => t.type === 'page');
    } catch { /* retry */ }
    if (target === undefined) await sleep(200);
  }
  assert.ok(target, 'no page target from /json/list');

  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((res, rej) => {
    const timer = setTimeout(() => rej(new Error('CDP ws open timeout 5 s')), CDP_TIMEOUT_MS);
    ws.addEventListener('open', () => { clearTimeout(timer); res(); }, { once: true });
    ws.addEventListener('error', () => { clearTimeout(timer); rej(new Error('CDP ws failed')); }, { once: true });
  });
  cdpWire(ws);
  await cdpCall('Runtime.enable');
  await cdpCall('Page.enable');
  await cdpCall('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1080, deviceScaleFactor: 1, mobile: false });
  await cdpCall('Page.navigate', { url: DASHBOARD_URL });

  const STATUS = `JSON.stringify({ c: document.querySelector('.connection')?.className ?? '', l: document.querySelector('.connection__label')?.textContent ?? '', t: document.querySelectorAll('ul.tile-grid li.tile[data-channel]').length, m: document.body.textContent.includes('Malformed') })`;
  let status = null;
  const readyDeadline = Date.now() + READY_TIMEOUT_MS;
  while (Date.now() < readyDeadline) {
    status = await evalJson(STATUS);
    if (status.c.includes('connection--open') && (status.l === 'Live' || status.l === 'Reconnected') && status.t > 0) break;
    await sleep(250);
  }
  assert.ok(status && status.c.includes('connection--open'), `no open connection within ${READY_TIMEOUT_MS} ms: ${JSON.stringify(status)}`);
  assert.ok(status.t > 0, 'no channel tiles within readiness window');
  assert.equal(status.m, false, 'UI reports malformed frames before measurement');

  const tiles = await evalJson(`JSON.stringify(Array.from(document.querySelectorAll('ul.tile-grid li.tile[data-channel]')).map((li) => [li.getAttribute('data-channel'), li.getAttribute('data-declared-hz'), li.getAttribute('data-sample-count')]))`);
  const dom = new Map(tiles.map(([n, h, c]) => [n, { declared: h === null || h === '' ? null : Number(h), count: Number(c) }]));
  for (const name of dom.keys()) assert.ok(expected.has(name), `unknown tile channel "${name}"`);
  for (const [name, hz] of expected) {
    if (hz !== null) {
      assert.ok(dom.has(name), `periodic channel "${name}" missing from tiles`);
      assert.equal(dom.get(name).declared, hz, `"${name}" data-declared-hz != generated rate`);
    } else if (dom.has(name)) {
      assert.ok(dom.get(name).declared === null || dom.get(name).declared === 0, `"${name}" must not declare a positive rate`);
    }
  }
  if (edited) {
    assert.equal(expected.get(edited.channel), edited.hz, `generated rate for "${edited.channel}" != EXPECTED_RATE_CHANGE`);
    assert.equal(dom.get(edited.channel)?.declared, edited.hz, `DOM declared rate for "${edited.channel}" != EXPECTED_RATE_CHANGE`);
  }

  const READ = `JSON.stringify({ perf: performance.now(), counts: Object.fromEntries(Array.from(document.querySelectorAll('ul.tile-grid li.tile[data-channel]')).map((li) => [li.getAttribute('data-channel'), li.getAttribute('data-sample-count')])) })`;
  const r1 = await evalJson(READ);
  await sleep(WINDOW_MS);
  const r2 = await evalJson(READ);
  const elapsed = (r2.perf - r1.perf) / 1000;
  assert.ok(elapsed > 0 && Math.abs(elapsed - WINDOW_MS / 1000) < 5, `implausible window ${elapsed} s`);

  const rates = {};
  for (const [name, hz] of expected) {
    if (!(name in r1.counts)) {
      assert.equal(hz, null, `periodic channel "${name}" missing from counters`);
      continue;
    }
    const c1 = Number(r1.counts[name]);
    const c2 = Number(r2.counts[name]);
    assert.ok(Number.isFinite(c1) && c1 >= 0, `"${name}" non-finite counter`);
    assert.ok(Number.isFinite(c2) && c2 >= c1, `"${name}" counter decreased ${c1}->${c2}`);
    const delta = c2 - c1;
    const observedHz = delta / elapsed;
    const declared = edited && edited.channel === name ? edited.hz : hz;
    if (declared !== null && declared >= 10) {
      assert.ok(delta > 0, `"${name}" declared ${declared} Hz but no samples in ${elapsed.toFixed(1)} s`);
      const err = Math.abs(observedHz - declared) / declared;
      assert.ok(err <= TOLERANCE, `"${name}" observed ${observedHz.toFixed(2)} Hz outside 5% of ${declared} Hz`);
      rates[name] = { declaredHz: declared, observedHz: +observedHz.toFixed(3), delta, verdict: 'rate-tolerance' };
    } else {
      assert.ok(c1 > 0 || delta > 0, `"${name}" counter never positive`);
      rates[name] = { declaredHz: declared, observedHz: +observedHz.toFixed(3), delta, verdict: 'positivity-monotonicity' };
    }
  }

  const end = await evalJson(STATUS);
  assert.equal(end.m, false, 'UI reports malformed frames during measurement');
  assert.deepEqual(cdpState.exceptions, [], `page exceptions: ${cdpState.exceptions.join(' | ')}`);

  mkdirSync(ARTIFACT_DIR, { recursive: true });
  const shot = await cdpCall('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true }, 15_000);
  writeFileSync(resolvePath(ARTIFACT_DIR, 'dashboard-smoke.png'), Buffer.from(shot.data, 'base64'));

  let revision = process.env.GITHUB_SHA ?? null;
  if (!revision) {
    try { revision = execSync('git rev-parse HEAD', { encoding: 'utf8' }).trim(); } catch { revision = 'unknown'; }
  }
  writeFileSync(resolvePath(ARTIFACT_DIR, 'dashboard-smoke.json'), `${JSON.stringify({
    revision, url: DASHBOARD_URL, measuredAt: new Date().toISOString(),
    elapsedSeconds: +elapsed.toFixed(3), toleranceRelative: TOLERANCE,
    expectedRateChange: edited, channels: rates,
    malformedUiCount: end.m ? 1 : 0, runtimeExceptions: cdpState.exceptions, pass: true,
  }, null, 2)}\n`);
  console.log(`dashboard smoke: PASS (${Object.keys(rates).length} channels, ${elapsed.toFixed(1)} s, ${revision})`);
} catch (err) {
  // Retain failure evidence while Chrome is still alive.
  if (cdpWs && !cdpState.closed) {
    try {
      mkdirSync(ARTIFACT_DIR, { recursive: true });
      const shot = await cdpCall('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true }, 15_000);
      writeFileSync(resolvePath(ARTIFACT_DIR, 'dashboard-smoke-failure.png'), Buffer.from(shot.data, 'base64'));
      writeFileSync(resolvePath(ARTIFACT_DIR, 'dashboard-smoke-failure.json'), `${JSON.stringify({
        url: DASHBOARD_URL, failedAt: new Date().toISOString(),
        error: err instanceof Error ? err.message : String(err),
        runtimeExceptions: cdpState.exceptions, pass: false,
      }, null, 2)}\n`);
    } catch { /* evidence best-effort */ }
  }
  throw err;
} finally {
  if (cdpWs) try { cdpWs.close(); } catch { /* ignore */ }
  if (chrome) {
    try { chrome.kill('SIGTERM'); } catch { /* ignore */ }
    await sleep(3000);
    if (chrome.exitCode === null) { try { chrome.kill('SIGKILL'); } catch { /* ignore */ } }
  }
  if (profileDir) { try { rmSync(profileDir, { recursive: true, force: true }); } catch { /* ignore */ } }
}
