import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { useTelemetryStore } from '../store/telemetryStore';
import { formatValue } from '../telemetry/format';
import {
  DELTA_RANGE,
  LAPS_RETAINED,
  SECTOR_LABELS,
  VERDICT_TEXT,
  inDeclaredRange,
  lapStateFromEvents,
  reduceLapState,
  referenceLap,
  sectorDeltas,
  sectorSum,
  sectorTime,
  type LapRecord,
  type LapState,
} from '../telemetry/laps';

/**
 * Lap and sector bar with delta, plus the validity indicator (PLAN.md section
 * 8.4; PHASES.md P8-T8).
 *
 * Every time and delta on this panel comes from the store's session event
 * stream, and the folding, verdicts and comparisons live in `telemetry/laps.ts`
 * so they can be reasoned about without React. See that module for the ordering
 * rules and, importantly, for exactly what the validity indicator does and does
 * not claim: track limits, DNF and the invalid flag are not on this stream, so a
 * `consistent` lap is internally reconciled, not a certified valid lap.
 *
 * The bar is canvas because it is a chart, and it redraws only when the lap table
 * changes - a dozen bars never needs an animation frame.
 */

const SECTOR_FALLBACK_COLOURS = ['#58a6ff', '#d2a8ff', '#3fb950'] as const;
const ROW_HEIGHT = 26;
const BAR_HEIGHT = 14;
const LABEL_PX = 34;
const VALUE_PX = 58;
/** Three rows: the latest lap, the reference lap, and the sector delta. */
const PADDING = { top: 8, bottom: 18 };
const FONT = '11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';

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
 * Delta against the reference lap, signed so a slower lap reads as slower. `null`
 * when no delta has been published, and an explicit note when the published
 * value is outside the contract's declared range - not a silently rendered
 * number.
 */
function deltaText(delta: number | null): string {
  if (delta === null || !Number.isFinite(delta)) {
    return '—';
  }
  if (!inDeclaredRange(delta, DELTA_RANGE)) {
    return `off range (${formatValue(delta)} s)`;
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
        context.fillText('No reconciled lap to plot yet', plotLeft + plotWidth / 2, height / 2);
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
          if (share > 26) {
            context.fillStyle = '#0b0f14';
            context.textAlign = 'center';
            context.fillText(SECTOR_LABELS[index] ?? '', x + share / 2, top + BAR_HEIGHT / 2);
          }
          x += share;
        });
        context.fillStyle = textColour;
        context.textAlign = 'right';
        context.fillText(lap === reference ? 'ref' : `L${lapLabel(lap)}`, plotLeft - 6, top + BAR_HEIGHT / 2);
        context.textAlign = 'left';
        context.fillText(`${total.toFixed(3)} s`, plotLeft + plotWidth + 6, top + BAR_HEIGHT / 2);
      });

      // Row 3: per-sector delta against the reference, centred on zero so the
      // sign is the geometry. Scaled to the largest sector delta present.
      const deltaTop = PADDING.top + 2 * ROW_HEIGHT;
      const centre = plotLeft + plotWidth / 2;
      context.strokeStyle = gridColour;
      context.beginPath();
      context.moveTo(Math.round(centre) + 0.5, deltaTop - 3);
      context.lineTo(Math.round(centre) + 0.5, deltaTop + BAR_HEIGHT + 3);
      context.stroke();
      context.fillStyle = textColour;
      context.textAlign = 'right';
      context.fillText('Δ', plotLeft - 6, deltaTop + BAR_HEIGHT / 2);
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
        largestDelta > 0 ? `±${largestDelta.toFixed(3)} s` : 'no sector delta',
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
        <span className="lap__note">lap and sector times, delta, consistency indicator</span>
      </div>

      <dl className="lap__readouts">
        <div className="lap__readout">
          <dt className="lap__key">lap</dt>
          <dd className="lap__value">{lapLabel(open)}</dd>
        </div>
        <div className="lap__readout">
          <dt className="lap__key">last lap</dt>
          <dd className="lap__value">{closed[0] === undefined ? '—' : lapTimeText(closed[0])}</dd>
        </div>
        <div className="lap__readout">
          <dt className="lap__key">delta</dt>
          <dd className={`lap__value ${deltaTone(latestDelta)}`}>{deltaText(latestDelta)}</dd>
        </div>
        <div className="lap__readout">
          <dt className="lap__key">reference</dt>
          <dd className="lap__value">
            {reference === null ? 'none yet' : `L${lapLabel(reference)} ${lapTimeText(reference)}`}
          </dd>
        </div>
      </dl>

      {!hasData ? (
        <p className="placeholder" role="note">
          No lap events yet. The contract&apos;s session channels are event-driven, so they appear only
          when a producer publishes them; the synthetic noise stream does not schedule them.
        </p>
      ) : null}

      <figure className="lap__figure">
        <canvas
          ref={canvasRef}
          className="lap__canvas"
          role="img"
          aria-label={
            reference === null
              ? 'Sector bar: no reconciled lap to plot yet'
              : `Sector bar: lap ${lapLabel(closed[0] ?? reference)} against reference lap ${lapLabel(
                  reference,
                )}, with per-sector delta`
          }
        />
        <figcaption className="lap__caption">
          <span className="lap__legend">
            {SECTOR_LABELS.map((label, index) => (
              <span key={label} className="lap__legend-item">
                <span className={`lap__swatch lap__swatch--${index + 1}`} />
                {label}
              </span>
            ))}
            <span className="lap__legend-item">
              <span className="lap__swatch lap__swatch--delta" />
              Δ vs reference
            </span>
          </span>
        </figcaption>
      </figure>

      {hasData ? (
        <div className="lap__current">
          <span className="lap__current-label">current lap {lapLabel(open)}</span>
          {SECTOR_LABELS.map((label, index) => (
            <span key={label} className="lap__current-sector">
              <span className="lap__key">{label}</span>
              {sectorText(open, index)}
            </span>
          ))}
          <span className="lap__current-sector">
            <span className="lap__key">sector idx</span>
            {currentSector === null ? '—' : currentSector}
          </span>
          <span className="lap__verdict lap__verdict--open">in progress</span>
        </div>
      ) : null}

      {closed.length > 0 ? (
        <table className="lap__table">
          <caption>recent laps, newest first</caption>
          <thead>
            <tr>
              <th scope="col">lap</th>
              {SECTOR_LABELS.map((label) => (
                <th key={label} scope="col">
                  {label}
                </th>
              ))}
              <th scope="col">lap time</th>
              <th scope="col">Δ lap</th>
              <th scope="col">Δ per sector</th>
              <th scope="col">ΣS − lap</th>
              <th scope="col">indicator</th>
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
                  <td className={`lap__indicator lap__indicator--${lap.verdict}`}>{VERDICT_TEXT[lap.verdict]}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      ) : null}

      <p className="lap__footnote">
        {deltaSeen
          ? 'Delta is the contract’s own reference-lap comparison at the current track position.'
          : 'No delta has been published on this stream, so no lap comparison is available.'}{' '}
        The indicator reports only what this stream can prove: three sector times reconciling with the
        reported lap time, inside the contract&apos;s declared range. Track limits, DNF and the invalid
        flag are not published here, so a consistent lap is not a certified valid lap.
      </p>
    </section>
  );
}
