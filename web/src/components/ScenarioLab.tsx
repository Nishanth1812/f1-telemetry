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
  story: string;
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
    story: 'The car starts from a standstill and accelerates for 7 seconds. More throttle means harder acceleration.',
    setting: 'Throttle',
    min: 50,
    max: 100,
    unit: '%',
  },
  {
    id: 'braking',
    title: 'Braking run',
    detail: 'Compare the speed trace with a lighter or stronger brake request.',
    story: 'The car is rolling at about 90 km/h, then the brakes come on. A stronger brake request stops it faster.',
    setting: 'Brake',
    min: 25,
    max: 100,
    unit: '%',
  },
  {
    id: 'cornering',
    title: 'Constant-radius corner',
    detail: 'Change steering around a fixed 50 m radius scenario.',
    story: 'The car drives around a 50 m radius circle at about 72 km/h. More steering tightens the turn, so the car is pushed harder sideways.',
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

const STANDARD_GRAVITY = 9.80665;

/** Above this sideways load the simulated car is turning or sliding. */
const SLIDE_WARNING_G = 0.3;

/** Plot area inside the chart SVG (viewBox units). */
const CHART = { left: 56, right: 590, top: 32, bottom: 150 };
const CHART_HEIGHT = 176;
const GRID_PATH = `M${CHART.left} ${CHART.top}H${CHART.right} M${CHART.left} 91H${CHART.right} M${CHART.left} ${CHART.bottom}H${CHART.right}`;

function formatSetting(scenario: LabScenario, value: number): string {
  return scenario.unit === '% of baseline' ? `${value}% of baseline` : `${value}%`;
}

function headline(id: LabScenarioId, result: ScenarioResult): string {
  if (id === 'launch') {
    return `Top speed reached ${result.peak_speed_kmh.toFixed(1)} km/h after ${result.duration_s.toFixed(1)} s at ${result.setting}% throttle.`;
  }
  if (id === 'braking') {
    return `Braking at ${result.setting}% slowed the car at up to ${result.peak_longitudinal_g.toFixed(2)} g.`;
  }
  return `The car was pushed sideways with up to ${result.peak_lateral_g.toFixed(2)} g at ${result.setting}% steering.`;
}

function linePoints(values: number[], low: number, high: number): string {
  if (values.length < 2) {
    return '';
  }
  const range = Math.max(high - low, 1e-6);
  return values
    .map((value, index) => {
      const x = CHART.left + (index / (values.length - 1)) * (CHART.right - CHART.left);
      const y = CHART.bottom - ((value - low) / range) * (CHART.bottom - CHART.top);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
}

function bounds(...series: number[][]): [number, number] {
  const all = series.flat();
  return [Math.min(...all), Math.max(...all)];
}

function ChartAxes({
  low,
  high,
  digits,
  startTime,
  endTime,
}: {
  low: number;
  high: number;
  digits: number;
  startTime: number;
  endTime: number;
}) {
  const labelY = CHART_HEIGHT - 6;
  return (
    <>
      <text className="scenario-chart__axis" x={CHART.left - 8} y={CHART.top + 5} textAnchor="end">
        {high.toFixed(digits)}
      </text>
      <text className="scenario-chart__axis" x={CHART.left - 8} y={CHART.bottom + 5} textAnchor="end">
        {low.toFixed(digits)}
      </text>
      <text className="scenario-chart__axis" x={CHART.left} y={labelY} textAnchor="start">
        {startTime.toFixed(1)}
      </text>
      <text className="scenario-chart__axis scenario-chart__axis--title" x={(CHART.left + CHART.right) / 2} y={labelY} textAnchor="middle">
        Time (s)
      </text>
      <text className="scenario-chart__axis" x={CHART.right} y={labelY} textAnchor="end">
        {endTime.toFixed(1)}
      </text>
    </>
  );
}

function ResultChart({ result }: { result: ScenarioResult }) {
  const speed = result.points.map((point) => point.speed_kmh);
  const longitudinal = result.points.map((point) => point.accel_longitudinal_m_s2 / STANDARD_GRAVITY);
  const lateral = result.points.map((point) => point.accel_lateral_m_s2 / STANDARD_GRAVITY);
  const [speedLow, speedHigh] = bounds(speed);
  // Both acceleration lines share one scale so their heights can be compared.
  const [gLow, gHigh] = bounds(longitudinal, lateral);
  const startTime = result.points[0]?.time_s ?? 0;
  const endTime = result.points[result.points.length - 1]?.time_s ?? 0;

  return (
    <div className="scenario-result__charts">
      <figure className="scenario-chart">
        <figcaption>Speed through the run · km/h</figcaption>
        <svg viewBox={`0 0 600 ${CHART_HEIGHT}`} role="img" aria-label="Speed over time">
          <path className="scenario-chart__grid" d={GRID_PATH} />
          <polyline className="scenario-chart__line" points={linePoints(speed, speedLow, speedHigh)} />
          <ChartAxes low={speedLow} high={speedHigh} digits={0} startTime={startTime} endTime={endTime} />
        </svg>
      </figure>
      <figure className="scenario-chart">
        <figcaption>Push on the car through the run · g</figcaption>
        <svg viewBox={`0 0 600 ${CHART_HEIGHT}`} role="img" aria-label="Forward-back and sideways push over time">
          <path className="scenario-chart__grid" d={GRID_PATH} />
          <polyline className="scenario-chart__line" points={linePoints(longitudinal, gLow, gHigh)} />
          <polyline
            className="scenario-chart__line scenario-chart__line--secondary"
            points={linePoints(lateral, gLow, gHigh)}
          />
          <ChartAxes low={gLow} high={gHigh} digits={2} startTime={startTime} endTime={endTime} />
        </svg>
        <div className="scenario-chart__legend">
          <span><i />Forward / back</span>
          <span><i className="scenario-chart__swatch--secondary" />Sideways</span>
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
  const scenario = SCENARIO_BY_ID[next.scenario_id];
  const metrics = [
    `top speed ${next.peak_speed_kmh.toFixed(1)} km/h, ${delta(next.peak_speed_kmh, prev.peak_speed_kmh, 'km/h', 1)}`,
    `hardest push forward/back ${next.peak_longitudinal_g.toFixed(2)} g, ${delta(next.peak_longitudinal_g, prev.peak_longitudinal_g, 'g', 2)}`,
    `hardest push sideways ${next.peak_lateral_g.toFixed(2)} g, ${delta(next.peak_lateral_g, prev.peak_lateral_g, 'g', 2)}`,
  ];
  return `${scenario.setting} ${formatSetting(scenario, prev.setting)} → ${formatSetting(scenario, next.setting)}. Compared with your last run: ${metrics.join(' · ')}.`;
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

  const resultScenario = result === null ? scenario : SCENARIO_BY_ID[result.scenario_id];
  const showSlideWarning =
    result !== null &&
    (result.scenario_id === 'launch' || result.scenario_id === 'braking') &&
    result.peak_lateral_g > SLIDE_WARNING_G;

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
          <h3 className="scenario-lab__step-heading">
            <span className="scenario-lab__step-num">1</span> Pick a run
          </h3>
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
          <h3 className="scenario-lab__step-heading">
            <span className="scenario-lab__step-num">2</span> Change one thing
          </h3>
          <div className="scenario-lab__controls">
            <label htmlFor="scenario-setting">
              <span>{scenario.setting}</span>
              <strong>{formatSetting(scenario, setting)}</strong>
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
          <h3 className="scenario-lab__step-heading">
            <span className="scenario-lab__step-num">3</span> See the result
          </h3>

          {error !== null ? (
            <p className="scenario-lab__error" role="alert">{error}</p>
          ) : null}
          {result !== null ? (
            <div className="scenario-result" aria-live="polite">
              <div className="scenario-result__heading">
                <div>
                  <h3>{resultScenario.title} result</h3>
                  <p>{resultScenario.story}</p>
                </div>
                <span>{result.duration_s.toFixed(1)} s simulated</span>
              </div>
              <p className="scenario-result__short">
                <strong>In short:</strong> {headline(result.scenario_id, result)}
              </p>
              {showSlideWarning ? (
                <p className="scenario-warning" role="status">
                  The simulated car started sliding sideways during this run, which a real straight-line
                  launch would not. This is a limit of the unfinished model, so trust the speed more than
                  the sideways number.
                </p>
              ) : null}
              <dl className="scenario-result__metrics">
                <div>
                  <dt>Top speed</dt>
                  <dd>{result.peak_speed_kmh.toFixed(1)} km/h</dd>
                  <dd className="scenario-metric__note">The fastest the car went during the run.</dd>
                </div>
                <div>
                  <dt>Hardest push forward/back</dt>
                  <dd>{result.peak_longitudinal_g.toFixed(2)} g</dd>
                  <dd className="scenario-metric__note">
                    Forward when speeding up, backward when braking. 1 g is your own body weight.
                  </dd>
                </div>
                <div>
                  <dt>Hardest push sideways</dt>
                  <dd>{result.peak_lateral_g.toFixed(2)} g</dd>
                  <dd className="scenario-metric__note">
                    Push out to the side. Only above about 0.3 g does the car count as turning or sliding.
                  </dd>
                </div>
                <div>
                  <dt>{resultScenario.setting}</dt>
                  <dd>{formatSetting(resultScenario, result.setting)}</dd>
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
