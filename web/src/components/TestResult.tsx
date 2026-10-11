import { useLabStore } from '../store/labStore';
import {
  SCENARIO_BY_ID,
  SLIDE_WARNING_G,
  STANDARD_GRAVITY,
  formatSetting,
} from '../store/labScenarios';
import type { LabScenarioId, ScenarioResult } from '../telemetry/socket';

/** Plot area inside the chart SVG (viewBox units). */
const CHART = { left: 56, right: 590, top: 32, bottom: 150 };
const CHART_HEIGHT = 176;
const GRID_PATH = `M${CHART.left} ${CHART.top}H${CHART.right} M${CHART.left} 91H${CHART.right} M${CHART.left} ${CHART.bottom}H${CHART.right}`;

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

export function TestResult() {
  const result = useLabStore((state) => state.result);
  const prevResult = useLabStore((state) => state.prevResult);

  if (result === null) {
    return (
      <section className="testlab-result" aria-labelledby="testlab-result-title">
        <h2 className="testlab-result__title" id="testlab-result-title">Last test result</h2>
        <p className="scenario-lab__empty">Your results will appear here after you run a test.</p>
      </section>
    );
  }

  const resultScenario = SCENARIO_BY_ID[result.scenario_id];
  const showSlideWarning =
    (result.scenario_id === 'launch' || result.scenario_id === 'braking') &&
    result.peak_lateral_g > SLIDE_WARNING_G;

  return (
    <section className="testlab-result" aria-labelledby="testlab-result-title">
      <h2 className="testlab-result__title" id="testlab-result-title">Last test result</h2>
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
    </section>
  );
}
