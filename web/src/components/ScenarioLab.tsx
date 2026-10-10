import { useState } from 'react';
import { useTelemetryStore } from '../store/telemetryStore';
import {
  requestScenarioRun,
  type LabScenarioId,
  type ScenarioResult,
} from '../telemetry/socket';

interface LabScenario {
  id: LabScenarioId;
  title: string;
  detail: string;
  setting: string;
  min: number;
  max: number;
  unit: string;
}

const SCENARIOS: LabScenario[] = [
  {
    id: 'launch',
    title: 'Standing launch',
    detail: 'See how throttle changes the first seconds from a standstill.',
    setting: 'Throttle',
    min: 50,
    max: 100,
    unit: '%',
  },
  {
    id: 'braking',
    title: 'Braking run',
    detail: 'Compare the speed trace with a lighter or stronger brake request.',
    setting: 'Brake',
    min: 25,
    max: 100,
    unit: '%',
  },
  {
    id: 'cornering',
    title: 'Constant-radius corner',
    detail: 'Change steering around a fixed 50 m radius scenario.',
    setting: 'Steering',
    min: 50,
    max: 150,
    unit: '% of baseline',
  },
];

const SCENARIO_BY_ID = Object.fromEntries(
  SCENARIOS.map((item) => [item.id, item]),
) as Record<LabScenarioId, LabScenario>;

const DEFAULT_SETTING: Record<LabScenarioId, number> = {
  launch: 100,
  braking: 100,
  cornering: 100,
};

function linePoints(values: number[]): string {
  if (values.length < 2) {
    return '';
  }
  const low = Math.min(...values);
  const high = Math.max(...values);
  const range = Math.max(high - low, 1e-6);
  return values
    .map((value, index) => {
      const x = (index / (values.length - 1)) * 560 + 20;
      const y = 150 - ((value - low) / range) * 118;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
}

function ResultChart({ result }: { result: ScenarioResult }) {
  const speed = result.points.map((point) => point.speed_kmh);
  const longitudinal = result.points.map((point) => point.accel_longitudinal_m_s2 / 9.80665);
  const lateral = result.points.map((point) => point.accel_lateral_m_s2 / 9.80665);
  return (
    <div className="scenario-result__charts">
      <figure className="scenario-chart">
        <figcaption>Speed through the run · km/h</figcaption>
        <svg viewBox="0 0 600 170" role="img" aria-label="Speed over time">
          <path className="scenario-chart__grid" d="M20 32H580 M20 91H580 M20 150H580" />
          <polyline className="scenario-chart__line" points={linePoints(speed)} />
        </svg>
      </figure>
      <figure className="scenario-chart">
        <figcaption>Acceleration through the run · g</figcaption>
        <svg viewBox="0 0 600 170" role="img" aria-label="Longitudinal and lateral acceleration over time">
          <path className="scenario-chart__grid" d="M20 32H580 M20 91H580 M20 150H580" />
          <polyline className="scenario-chart__line" points={linePoints(longitudinal)} />
          <polyline className="scenario-chart__line scenario-chart__line--secondary" points={linePoints(lateral)} />
        </svg>
        <div className="scenario-chart__legend">
          <span><i />Longitudinal</span><span><i className="scenario-chart__swatch--secondary" />Lateral</span>
        </div>
      </figure>
    </div>
  );
}

function delta(next: number, prev: number, unit: string, digits: number): string {
  const diff = next - prev;
  if (Math.abs(diff) < 0.5 * 10 ** -digits) {
    return 'no change';
  }
  return `${diff > 0 ? 'up' : 'down'} ${Math.abs(diff).toFixed(digits)} ${unit}`;
}

function describeChange(prev: ScenarioResult, next: ScenarioResult): string {
  const metrics = [
    `speed ${next.peak_speed_kmh.toFixed(1)} km/h, ${delta(next.peak_speed_kmh, prev.peak_speed_kmh, 'km/h', 1)}`,
    `forward g ${next.peak_longitudinal_g.toFixed(2)}, ${delta(next.peak_longitudinal_g, prev.peak_longitudinal_g, 'g', 2)}`,
    `sideways g ${next.peak_lateral_g.toFixed(2)}, ${delta(next.peak_lateral_g, prev.peak_lateral_g, 'g', 2)}`,
  ];
  return `${next.setting_name} ${prev.setting}% → ${next.setting}%. Peak ${metrics.join(' · ')} compared with your last run of this scenario.`;
}

export function ScenarioLab() {
  const status = useTelemetryStore((state) => state.status);
  const [scenarioId, setScenarioId] = useState<LabScenarioId>('launch');
  const [setting, setSetting] = useState(DEFAULT_SETTING.launch);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<ScenarioResult | null>(null);
  const [lastRuns, setLastRuns] = useState<Partial<Record<LabScenarioId, ScenarioResult>>>({});
  const [prevResult, setPrevResult] = useState<ScenarioResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const scenario = SCENARIO_BY_ID[scenarioId];

  const chooseScenario = (next: LabScenarioId): void => {
    setScenarioId(next);
    setSetting(DEFAULT_SETTING[next]);
    setResult(null);
    setError(null);
  };

  const run = async (): Promise<void> => {
    setRunning(true);
    setError(null);
    try {
      const newResult = await requestScenarioRun(scenarioId, setting);
      setPrevResult(lastRuns[scenarioId] ?? null);
      setLastRuns((runs) => ({ ...runs, [scenarioId]: newResult }));
      setResult(newResult);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'The run could not finish.');
    } finally {
      setRunning(false);
    }
  };

  return (
    <section className="scenario-lab" aria-labelledby="scenario-lab-title">
      <div className="scenario-lab__intro">
        <div>
          <p className="scenario-lab__eyebrow">Learn by changing one thing</p>
          <h2 id="scenario-lab-title">Scenario Lab</h2>
          <p>Pick a run, adjust its control, and see what the model produces.</p>
        </div>
        <span className="scenario-lab__badge">Simulation · not calibrated to a real car</span>
      </div>

      <div className="scenario-lab__steps">
        <section aria-label="Step 1: Pick a run" className="scenario-lab__step">
          <h3 className="scenario-lab__step-heading"><span className="scenario-lab__step-num">1</span> Pick a run</h3>
          <div className="scenario-lab__choices" role="group" aria-label="Choose a scenario">
        {SCENARIOS.map((item) => (
          <button
            key={item.id}
            className={`scenario-choice${item.id === scenarioId ? ' scenario-choice--selected' : ''}`}
            type="button"
            aria-pressed={item.id === scenarioId}
            disabled={running}
            onClick={() => chooseScenario(item.id)}
          >
            <strong>{item.title}</strong>
            <span>{item.detail}</span>
          </button>
        ))}
        </div>
      </section>

      <section aria-label="Step 2: Change one thing" className="scenario-lab__step">
        <h3 className="scenario-lab__step-heading"><span className="scenario-lab__step-num">2</span> Change one thing</h3>
        <div className="scenario-lab__controls">
        <label htmlFor="scenario-setting">
          <span>{scenario.setting}</span>
          <strong>
            {setting}%{scenario.unit === '% of baseline' ? ' of baseline' : ''}
          </strong>
        </label>
        <input
          id="scenario-setting"
          type="range"
          min={scenario.min}
          max={scenario.max}
          step="1"
          value={setting}
          disabled={running}
          onChange={(event) => {
            setSetting(Number(event.currentTarget.value));
            setResult(null);
          }}
        />
        <button
          className="scenario-lab__run"
          type="button"
          onClick={() => void run()}
          disabled={running || status !== 'open'}
        >
          {running ? 'Running…' : 'Run scenario'}
        </button>
        {status !== 'open' ? (
          <p className="scenario-lab__hint">
            Start the server in PowerShell with <code>uv run f1-serve</code>. To open the page,
            run <code>npm --prefix web run dev</code> in a second window.
          </p>
        ) : null}
      </div>
      </section>

      <section aria-label="Step 3: See the result" className="scenario-lab__step">
        <h3 className="scenario-lab__step-heading"><span className="scenario-lab__step-num">3</span> See the result</h3>

        {error !== null ? (
          <p className="scenario-lab__error" role="alert">{error}</p>
        ) : null}
        {result !== null ? (
        <div className="scenario-result" aria-live="polite">
          <div className="scenario-result__heading">
            <div>
              <h3>{result.title} result</h3>
              <p>{result.description}</p>
            </div>
            <span>{result.duration_s.toFixed(1)} s simulated</span>
          </div>
          <dl className="scenario-result__metrics">
            <div>
              <dt>Peak speed</dt>
              <dd>{result.peak_speed_kmh.toFixed(1)} km/h</dd>
              <dd className="scenario-metric__note">How fast the car went; higher means faster straight-line speed.</dd>
            </div>
            <div>
              <dt>Peak longitudinal g</dt>
              <dd>{result.peak_longitudinal_g.toFixed(2)} g</dd>
              <dd className="scenario-metric__note">How hard the car accelerated or braked straight ahead. 1 g equals your own body weight.</dd>
            </div>
            <div>
              <dt>Peak lateral g</dt>
              <dd>{result.peak_lateral_g.toFixed(2)} g</dd>
              <dd className="scenario-metric__note">How hard the car was pushed sideways in a turn. 1 g equals your own body weight.</dd>
            </div>
            <div>
              <dt>{result.setting_name}</dt>
              <dd>
                {result.setting}{scenario.unit === '% of baseline' ? '% baseline' : '%'}
              </dd>
              <dd className="scenario-metric__note">The control you changed for this run.</dd>
            </div>
          </dl>
          {prevResult !== null && prevResult.scenario_id === result.scenario_id ? (
            <p className="scenario-what-changed" aria-live="polite">
              <strong>What changed:</strong> {describeChange(prevResult, result)}
            </p>
          ) : null}
          <ResultChart result={result} />
          <p className="scenario-result__note">{result.simulation_label}</p>
        </div>
      ) : (
        <p className="scenario-lab__empty">Your results will appear here after you run a scenario.</p>
      )}
      </section>
      </div>
    </section>
  );
}
