// Real scenario/save/restart/replay, sharing dashboard-smoke.mjs's Chrome session.
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve, dirname, basename } from 'node:path';

const delay = (ms) => new Promise((done) => setTimeout(done, ms));

export async function startWorkflowServer() {
  const directory = mkdtempSync(join(tmpdir(), 'f1-workflow-'));
  const found = spawnSync('uv', ['run', '--frozen', 'python', '-c', 'import sys; print(sys.executable)'], { encoding: 'utf8', windowsHide: true });
  assert.equal(found.status, 0, found.stderr);
  let child;
  let log = '';
  const stop = async () => {
    if (!child || child.exitCode !== null) return;
    child.kill('SIGTERM');
    await new Promise((done) => {
      const timer = setTimeout(done, 10_000);
      child.once('exit', () => { clearTimeout(timer); done(); });
    });
    if (child.exitCode === null) child.kill('SIGKILL');
  };
  const start = async () => {
    child = spawn(found.stdout.trim(), ['-m', 'f1telemetry.server', '--host', '127.0.0.1', '--port', '8765', '--runs-dir', directory, '--web-root', 'web/dist'], { windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
    child.stdout.on('data', (chunk) => { log = (log + chunk).slice(-20_000); });
    child.stderr.on('data', (chunk) => { log = (log + chunk).slice(-20_000); });
    const deadline = Date.now() + 120_000;
    while (Date.now() < deadline) {
      assert.equal(child.exitCode, null, `backend exited: ${log}`);
      try { if ((await fetch('http://127.0.0.1:8765/healthz')).ok) return; } catch { /* startup */ }
      await delay(250);
    }
    throw new Error(`backend readiness timeout: ${log}`);
  };
  try { await start(); } catch (error) { await stop(); throw error; }
  return {
    async restart() { await stop(); await start(); },
    async close() {
      await stop();
      assert.equal(dirname(resolve(directory)), resolve(tmpdir()));
      assert.ok(basename(directory).startsWith('f1-workflow-'));
      rmSync(directory, { recursive: true, force: true });
    },
  };
}

export async function runWorkflow({ cdpCall, evalJson, server }) {
  const evaluate = async (expression) => {
    const result = await cdpCall('Runtime.evaluate', { expression, returnByValue: true });
    assert.equal(result.exceptionDetails, undefined, JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  const until = async (expression) => {
    const deadline = Date.now() + 120_000;
    while (Date.now() < deadline) {
      if (await evaluate(expression)) return;
      await delay(250);
    }
    throw new Error(`dashboard condition timed out: ${expression}`);
  };
  const click = (text) => evaluate(`Array.from(document.querySelectorAll('button')).find((b) => b.textContent.trim() === ${JSON.stringify(text)}).click()`);
  await until(`document.querySelector('.testlab__run')?.disabled === false`);
  const setting = await evaluate(`Number(document.querySelector('#testlab-setting').value)`);
  await click('Run test');
  await until(`document.querySelectorAll('#saved-test option').length === 1 && document.querySelector('.scenario-result') !== null`);
  const runId = await evaluate(`document.querySelector('#saved-test').value`);
  assert.match(runId, /^[0-9a-f]{32}$/);
  console.log('workflow: real scenario saved');
  const before = await evaluate(`document.querySelector('.scenario-result__short').textContent`);
  await server.restart();
  console.log('workflow: backend restarted');
  await until(`document.querySelector('.connection')?.className.includes('connection--open')`);
  await click('Refresh saved tests');
  await until(`document.querySelector('#saved-test')?.value === '${runId}'`);
  await click('Replay saved test');
  await until(`document.querySelector('.testlab__chip')?.textContent.includes('Driving')`);
  assert.equal(await evaluate(`document.querySelector('.scenario-result__short').textContent`), before);
  await until(`Number(document.querySelector('ul.tile-grid li.tile[data-channel="speed"]')?.getAttribute('data-sample-count')) > 2`);
  console.log('workflow: saved samples replayed');
  await until(`document.querySelector('.testlab__run')?.disabled === false`);
  await click('Run test');
  await until(`document.querySelectorAll('#saved-test option').length === 2`);
  await click('Compare results');
  await until(`document.body.textContent.includes('Compared with your last run:')`);
  const state = await evalJson(`JSON.stringify({ malformed: document.body.textContent.includes('Malformed'), saved: document.querySelectorAll('#saved-test option').length })`);
  assert.equal(state.malformed, false);
  return { setting, persistedRunId: runId, savedRuns: state.saved, restartReplay: true, comparison: true };
}
