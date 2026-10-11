import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { useTelemetryStore } from '../store/telemetryStore';
import { formatValue } from '../telemetry/format';
import {
  DELTA_RANGE,
  LAPS_RETAINED,
  SECTOR_LABELS,
  inDeclaredRange,
  lapStateFromEvents,
  reduceLapState,
  referenceLap,
  sectorDeltas,
  sectorSum,
  sectorTime,
  type LapRecord,
  type LapState,
  type LapVerdict,
} from '../telemetry/laps';

/**
 * Lap and sector bar with gap to the reference lap, plus the check indicator
 * (PLAN.md section 8.4; PHASES.md P8-T8).
 *
 * Every time and gap on this panel comes from the store's session event
 * stream, and the folding, verdicts and comparisons live in `telemetry/laps.ts`
 * so they can be reasoned about without React. See that module for the ordering
 * rules and, importantly, for exactly what the check does and does not claim:
 * track limits, DNF and the invalid flag are not on this stream, so a
 * "checks out" lap is internally reconciled, not an officially valid lap.
 *
 * On screen the labels are plain English; the raw names (`delta`, `ΣS − lap`,
 * `sector_index`, `ref`) are kept in `title` tooltips for engineers.
 *
 * The bar is canvas because it is a chart, and it redraws only when the lap table
 * changes - a dozen bars never needs an animation frame.
 */

const SECTOR_FALLBACK_COLOURS = ['#58a6ff', '#d2a8ff', '#3fb950'] as const;
const ROW_HEIGHT = 26;
const BAR_HEIGHT = 14;
/** Wide enough for the row label "Reference" at the canvas font. */
const LABEL_PX = 72;
const VALUE_PX = 58;
/** Three rows: the latest lap, the reference lap, and the sector gap. */
const PADDING = { top: 8, bottom: 18 };
const FONT = '11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';

/** Plain-English check results, one per verdict. */
const CHECK_TEXT: Readonly<Record<LapVerdict, string>> = {
  consistent: 'Adds up',
  unreconciled: "Sectors don't add up to the lap time",
  unmeasured: 'No lap time reported',
  'out-of-range': 'Outside the expected range',
};

function sectorName(index: number): string {
  return `Sector ${index + 1}`;
}

function lapLabel(lap: LapRecord): string {
  return lap.lapIndex === null ? '—' : `${lap.lapIndex}`;
}

function sectorText(lap: LapRecord, index: number): string {
  const time = sectorTime(lap, index);
  return time === null ? '—' : time.toFixed(3);
}

function lapTimeText(lap: LapRecord): string {
  return lap.lapTime === null ? '—' : `${lap.lapTime.toFixed(3)} s`;
}

function secondsText(value: number | null, digits = 3): string {
  if (value === null || !Number.isFinite(value)) {
    return '—';
  }
  return `${value > 0 ? '+' : value < 0 ? '−' : '±'}${Math.abs(value).toFixed(digits)}`;
}

/**
 * Gap against the reference lap, signed so a slower lap reads as slower. `null`
 * when no gap has been published, and an explicit note when the published
 * value is outside the contract's declared range - not a silently rendered
 * number.
 */
function deltaText(delta: number | null): string {
  if (delta === null || !Number.isFinite(delta)) {
    return '—';
  }
  if (!inDeclaredRange(delta, DELTA_RANGE)) {
    return `outside expected range (${formatValue(delta)} s)`;
  }
  return `${secondsText(delta)} s`;
}

function deltaTone(delta: number | null): string {
  if (delta === null || !Number.isFinite(delta) || !inDeclaredRange(delta, DELTA_RANGE)) {
    return 'lap__delta--none';
  }
  if (delta > 0.001) {
    return 'lap__delta--slow';
  }
  return delta < -0.001 ? 'lap__delta--fast' : 'lap__delta--even';
}

export function LapBar() {
  const headingId = useId();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const events = useTelemetryStore((state) => state.events);
  const [state, setState] = useState<LapState>(() => lapStateFromEvents(events));
  // Id of the newest entry already folded into `state`. The initial state folds
  // the whole log, so the cursor starts at its end.
  const foldedUpTo = useRef<number>(events[events.length - 1]?.id ?? 0);

  // The retained log is bounded and truncated from the front, so only entries not
  // yet folded are applied. Re-folding the whole log would double-count laps.
  useEffect(() => {
    const fresh = events.filter((entry) => entry.id > foldedUpTo.current);
    if (fresh.length === 0) {
      return;
    }
    foldedUpTo.current = fresh[fresh.length - 1]?.id ?? foldedUpTo.current;
    setState((current) => fresh.reduce(reduceLapState, current));
  }, [events]);

  const { laps, open, latestDelta, deltaSeen } = state;
  const closed = useMemo(() => laps.filter((lap) => lap.closed).slice(0, LAPS_RETAINED), [laps]);
  const reference = useMemo(() => referenceLap(closed), [closed]);
  // Row 1 is the most recent complete lap, row 2 the reference it is measured
  // against. With no reconciled reference there is nothing to compare to, and the
  // canvas says so rather than drawing a bar against nothing.
  const barRows: readonly LapRecord[] = useMemo(
    () => (reference === null || closed[0] === undefined ? [] : [closed[0], reference]),
    [closed, reference],
  );
  const currentSector = open.sectorIndices.length === 0 ? null : (open.sectorIndices[open.sectorIndices.length - 1] ?? null);
  const hasData = closed.length > 0 || open.sectors.some((time) => time !== null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null) {
      return;
    }
    const context = canvas.getContext('2d', { alpha: false });
    if (context === null) {
      return;
    }
    const styles = getComputedStyle(canvas);
    const read = (name: string, fallback: string): string => styles.getPropertyValue(name).trim() || fallback;
    const surface = read('--trace-surface', '#0d1117');
    const gridColour = read('--trace-grid', 'rgba(255,255,255,0.12)');
    const textColour = read('--trace-text', '#9aa7b4');
    const sectorColours = SECTOR_LABELS.map((_label, index) =>
      read(`--sector-${index + 1}`, SECTOR_FALLBACK_COLOURS[index] ?? '#8b949e'),
    );
    const fastColour = read('--ok', '#3fb950');
    const slowColour = read('--warn', '#d8a026');

    const draw = (): void => {
      const bounds = canvas.getBoundingClientRect();
      const width = Math.max(1, Math.round(bounds.width));
      const height = Math.max(1, Math.round(bounds.height));
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.fillStyle = surface;
      context.fillRect(0, 0, width, height);

      const plotLeft = PADDING.top + LABEL_PX;
      const plotWidth = Math.max(4, width - plotLeft - VALUE_PX);
      context.font = FONT;
      context.textBaseline = 'middle';

      if (barRows.length === 0) {
        context.fillStyle = textColour;
        context.textAlign = 'center';
        context.fillText('No matched lap to compare yet', plotLeft + plotWidth / 2, height / 2);
        return;
      }

      const longest = barRows.reduce<number>((total, lap) => Math.max(total, sectorSum(lap)), 0);
      const latest = barRows[0];
      if (latest === undefined) {
        return;
      }
      const deltas = sectorDeltas(latest, reference);
      const largestDelta = deltas.reduce<number>(
        (largest, delta) => (delta === null ? largest : Math.max(largest, Math.abs(delta))),
        0,
      );

      // Rows 1 and 2: sector splits as stacked bars on a shared scale, so the
      // reference lap is directly comparable to the lap it is being read against.
      barRows.forEach((lap, row) => {
        const top = PADDING.top + row * ROW_HEIGHT;
        const total = sectorSum(lap);
        const usable = longest > 0 ? (total / longest) * plotWidth : plotWidth;
        let x = plotLeft;
        lap.sectors.forEach((time, index) => {
          const share = total > 0 && time !== null ? (time / total) * usable : usable / SECTOR_LABELS.length;
          context.fillStyle = sectorColours[index] ?? textColour;
          context.fillRect(x, top, Math.max(1, share), BAR_HEIGHT);
          // Name the sector when there is room, otherwise just its number.
          const inside = share > 70 ? sectorName(index) : share > 14 ? `${index + 1}` : '';
          if (inside !== '') {
            context.fillStyle = '#0b0f14';
            context.textAlign = 'center';
            context.fillText(inside, x + share / 2, top + BAR_HEIGHT / 2);
          }
          x += share;
        });
        context.fillStyle = textColour;
        context.textAlign = 'right';
        context.fillText(lap === reference ? 'Reference' : `Lap ${lapLabel(lap)}`, plotLeft - 6, top + BAR_HEIGHT / 2);
        context.textAlign = 'left';
        context.fillText(`${total.toFixed(3)} s`, plotLeft + plotWidth + 6, top + BAR_HEIGHT / 2);
      });

      // Row 3: per-sector gap against the reference, centred on zero so the
      // sign is the geometry. Scaled to the largest sector gap present.
      const deltaTop = PADDING.top + 2 * ROW_HEIGHT;
      const centre = plotLeft + plotWidth / 2;
      context.strokeStyle = gridColour;
      context.beginPath();
      context.moveTo(Math.round(centre) + 0.5, deltaTop - 3);
      context.lineTo(Math.round(centre) + 0.5, deltaTop + BAR_HEIGHT + 3);
      context.stroke();
      context.fillStyle = textColour;
      context.textAlign = 'right';
      context.fillText('Gap', plotLeft - 6, deltaTop + BAR_HEIGHT / 2);
      const cellWidth = plotWidth / SECTOR_LABELS.length;
      deltas.forEach((delta, index) => {
        if (delta === null) {
          return;
        }
        const magnitude = largestDelta > 0 ? (Math.abs(delta) / largestDelta) * (cellWidth / 2 - 2) : 0;
        const x = plotLeft + index * cellWidth + cellWidth / 2;
        context.fillStyle = delta > 0 ? slowColour : fastColour;
        context.fillRect(delta >= 0 ? x : x - magnitude, deltaTop, Math.max(1, magnitude), BAR_HEIGHT);
      });
      context.textAlign = 'left';
      context.fillText(
        largestDelta > 0 ? `±${largestDelta.toFixed(3)} s` : 'no sector gap',
        plotLeft + plotWidth + 6,
        deltaTop + BAR_HEIGHT / 2,
      );
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [barRows, closed, reference]);

  return (
    <section className="lap" aria-labelledby={headingId}>
      <div className="lap__header">
        <h2 className="lap__heading" id={headingId}>
          Laps
        </h2>
        <span className="lap__note">lap and sector times, gap to the reference lap, check result</span>
      </div>

      <dl className="lap__readouts">
        <div className="lap__readout">
          <dt className="lap__key" title="Current lap">Current lap</dt>
          <dd className="lap__value">{lapLabel(open)}</dd>
        </div>
        <div className="lap__readout">
          <dt className="lap__key">Last lap</dt>
          <dd className="lap__value">{closed[0] === undefined ? '—' : lapTimeText(closed[0])}</dd>
        </div>
        <div className="lap__readout">
          <dt className="lap__key" title="delta">Gap to reference</dt>
          <dd className={`lap__value ${deltaTone(latestDelta)}`}>{deltaText(latestDelta)}</dd>
        </div>
        <div className="lap__readout">
          <dt className="lap__key" title="ref">Reference lap</dt>
          <dd className="lap__value">
            {reference === null ? 'none yet' : `Lap ${lapLabel(reference)} ${lapTimeText(reference)}`}
          </dd>
        </div>
      </dl>

      {!hasData ? (
        <p className="placeholder" role="note">
          No lap times yet. Lap and sector times appear here once the simulator reports a finished
          lap.
        </p>
      ) : null}

      <figure className="lap__figure">
        <canvas
          ref={canvasRef}
          className="lap__canvas"
          role="img"
          aria-label={
            reference === null
              ? 'Sector bar chart: no matched lap to compare yet'
              : `Sector bar chart: lap ${lapLabel(closed[0] ?? reference)} against reference lap ${lapLabel(
                  reference,
                )}, with the gap in each sector`
          }
        />
        <figcaption className="lap__caption">
          <span className="lap__legend">
            {SECTOR_LABELS.map((_label, index) => (
              <span key={sectorName(index)} className="lap__legend-item">
                <span className={`lap__swatch lap__swatch--${index + 1}`} />
                {sectorName(index)}
              </span>
            ))}
            <span className="lap__legend-item">
              <span className="lap__swatch lap__swatch--delta" />
              Gap vs reference
            </span>
          </span>
        </figcaption>
      </figure>

      {hasData ? (
        <div className="lap__current">
          <span className="lap__current-label">Current lap {lapLabel(open)}</span>
          {SECTOR_LABELS.map((_label, index) => (
            <span key={sectorName(index)} className="lap__current-sector">
              <span className="lap__key">{sectorName(index)}</span>
              {sectorText(open, index)}
            </span>
          ))}
          <span className="lap__current-sector">
            <span className="lap__key" title="sector_index">Current sector</span>
            {currentSector === null ? '—' : currentSector}
          </span>
          <span className="lap__verdict lap__verdict--open">In progress</span>
        </div>
      ) : null}

      {closed.length > 0 ? (
        <table className="lap__table">
          <caption>Recent laps, newest first</caption>
          <thead>
            <tr>
              <th scope="col">Lap</th>
              {SECTOR_LABELS.map((_label, index) => (
                <th key={sectorName(index)} scope="col">
                  {sectorName(index)}
                </th>
              ))}
              <th scope="col">Lap time</th>
              <th scope="col" title="Δ lap">Gap to reference</th>
              <th scope="col" title="Δ per sector">Gap per sector</th>
              <th scope="col" title="ΣS − lap">Sector sum minus lap time</th>
              <th scope="col" title="indicator">Check</th>
            </tr>
          </thead>
          <tbody>
            {closed.map((lap, index) => {
              const deltas = sectorDeltas(lap, reference);
              return (
                <tr key={`${lapLabel(lap)}-${lap.lapTime ?? index}`} className={`lap__row--${lap.verdict}`}>
                  <th scope="row">{lapLabel(lap)}</th>
                  {SECTOR_LABELS.map((_label, sector) => (
                    <td key={sector}>{sectorText(lap, sector)}</td>
                  ))}
                  <td>{lapTimeText(lap)}</td>
                  <td className={deltaTone(lap.deltaAtClose)}>{deltaText(lap.deltaAtClose)}</td>
                  <td className="lap__sector-deltas">
                    {deltas.map((delta) => (delta === null ? '—' : secondsText(delta))).join(' / ')}
                  </td>
                  <td>{secondsText(lap.reconciliation, 4)}</td>
                  <td className={`lap__indicator lap__indicator--${lap.verdict}`}>{CHECK_TEXT[lap.verdict]}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      ) : null}

      <p className="lap__footnote">
        {deltaSeen
          ? "The gap is the simulator's own comparison with the reference lap at the current track position."
          : 'The simulator has not sent any gap figures on this stream, so no lap comparison is available.'}{' '}
        The check only confirms that the three sector times add up to the reported lap time, within
        the expected range. Track limits, did-not-finish status and the official invalid-lap flag are
        not sent on this stream, so a lap that adds up is not an officially valid lap.
      </p>
    </section>
  );
}
