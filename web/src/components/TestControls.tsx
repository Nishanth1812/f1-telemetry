import { useEffect, useState } from 'react';
import '../styles/testlab.css';
import { useTelemetryStore } from '../store/telemetryStore';
import { useLabStore } from '../store/labStore';
import {
  APPROX_DURATION_S,
  SCENARIOS,
  SCENARIO_BY_ID,
  formatSetting,
} from '../store/labScenarios';

const TICK_MS = 100;

/**
 * Tracks the drive that follows the run reply. The server streams the car
 * after it sends the summary, so the clock starts at startedAtMs and runs for
 * the scenario's approximate duration.
 */
function useDriveClock(startedAtMs: number | null, durationS: number): number {
  const [nowMs, setNowMs] = useState(() => performance.now());

  useEffect(() => {
    if (startedAtMs === null) {
      return;
    }
    const endMs = startedAtMs + durationS * 1000;
    setNowMs(performance.now());
    const id = window.setInterval(() => {
      const now = performance.now();
      setNowMs(now);
      if (now >= endMs) {
        window.clearInterval(id);
      }
    }, TICK_MS);
    return () => window.clearInterval(id);
  }, [startedAtMs, durationS]);

  return nowMs;
}

export function TestControls() {
  const status = useTelemetryStore((state) => state.status);
  const readOnly = useLabStore((state) => state.readOnly);
  const scenarioId = useLabStore((state) => state.scenarioId);
  const setting = useLabStore((state) => state.setting);
  const running = useLabStore((state) => state.running);
  const error = useLabStore((state) => state.error);
  const startedAtMs = useLabStore((state) => state.startedAtMs);
  const chooseScenario = useLabStore((state) => state.chooseScenario);
  const setSetting = useLabStore((state) => state.setSetting);
  const run = useLabStore((state) => state.run);

  const scenario = SCENARIO_BY_ID[scenarioId];
  const playbackDurationS = useLabStore((state) => state.playbackDurationS);
  // The drive is as long as the run that was just played, whatever test is selected now.
  const durationS = playbackDurationS ?? APPROX_DURATION_S[scenarioId];
  const nowMs = useDriveClock(startedAtMs, durationS);

  const elapsedS =
    startedAtMs === null ? 0 : Math.min(Math.max((nowMs - startedAtMs) / 1000, 0), durationS);
  const driving = startedAtMs !== null && elapsedS < durationS;
  const busy = running || driving || readOnly;
  const connected = status === 'open';
  const progressPct = durationS <= 0 ? 0 : Math.min(100, (elapsedS / durationS) * 100);

  let chip: string;
  let chipTone: string;
  if (!connected) {
    chip = 'Not connected';
    chipTone = 'offline';
  } else if (readOnly) {
    chip = 'Recorded demo · read-only';
    chipTone = 'ready';
  } else if (running) {
    chip = 'Simulating the run…';
    chipTone = 'busy';
  } else if (driving) {
    chip = `Driving · ${elapsedS.toFixed(1)} of ${durationS.toFixed(1)} s`;
    chipTone = 'busy';
  } else if (startedAtMs !== null) {
    chip = 'Test finished · last values held';
    chipTone = 'ready';
  } else {
    chip = 'Car at rest';
    chipTone = 'ready';
  }

  return (
    <section className="testlab" aria-labelledby="testlab-title">
      <div className="testlab__head">
        <div className="testlab__heading">
          <h2 id="testlab-title">{readOnly ? 'Recorded demo' : 'Virtual test'}</h2>
          <p className="testlab__detail">{scenario.detail}</p>
        </div>
        <span className={`testlab__chip testlab__chip--${chipTone}`} role="status" aria-live="polite">
          {chip}
        </span>
      </div>

      {readOnly ? <p className="testlab__hint">This demo loops a saved launch. Run the project locally to record and compare your own tests.</p> : null}

      {!readOnly ? <div className="testlab__bar">
        <div className="testlab__scenarios" role="group" aria-label="Choose a test">
          {SCENARIOS.map((item) => (
            <button
              key={item.id}
              className="testlab__choice"
              type="button"
              aria-pressed={item.id === scenarioId}
              disabled={busy}
              onClick={() => chooseScenario(item.id)}
            >
              {item.title}
            </button>
          ))}
        </div>

        <label className="testlab__setting" htmlFor="testlab-setting">
          <span>{scenario.setting}</span>
          <input
            id="testlab-setting"
            type="range"
            min={scenario.min}
            max={scenario.max}
            step="1"
            value={setting}
            disabled={busy}
            onChange={(event) => setSetting(Number(event.currentTarget.value))}
          />
          <strong>{formatSetting(scenario, setting)}</strong>
        </label>

        <button
          className="testlab__run"
          type="button"
          onClick={() => void run()}
          disabled={busy || !connected}
        >
          {running ? 'Running…' : 'Run test'}
        </button>
      </div> : null}

      {driving && !readOnly ? (
        <div className="testlab__progress" aria-hidden="true">
          <span className="testlab__progress-fill" style={{ width: `${progressPct}%` }} />
        </div>
      ) : null}

      {!connected ? (
        <p className="testlab__hint">
          Start the server in PowerShell with <code>uv run f1-serve</code>. To open the page, run{' '}
          <code>npm --prefix web run dev</code> in a second window.
        </p>
      ) : null}
      {error !== null ? (
        <p className="testlab__error" role="alert">
          {error}
        </p>
      ) : null}
    </section>
  );
}
