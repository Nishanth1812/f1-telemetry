import { useEffect, useId, useRef } from 'react';
import { cornerChannels } from '../generated/channels';
import { useTelemetryStore } from '../store/telemetryStore';
import { formatValue } from '../telemetry/format';
import type { TelemetryFrame } from '../telemetry/types';

/**
 * Four-corner tyre panel (PLAN.md section 8.4).
 *
 * Live numbers come from the throttled store snapshot through plain selectors,
 * the same contract Tile relies on, so React never holds a sample. The dense
 * part - thirty seconds of carcass temperature and slip ratio per corner - is
 * drawn on canvas in one requestAnimationFrame loop that reads the per-channel
 * ring buffers directly. Decimation to the plot's pixel width happens in
 * TraceHistory.fill before anything reaches the canvas, and there is no React
 * state anywhere in the path.
 */

const CORNER_LABELS = ['FL', 'FR', 'RL', 'RR'] as const;
const WINDOW_SECONDS = 30;
const GRID_ROWS = 4;
const ARIA_INTERVAL_MS = 250;
const TEMP_BAND_RATIO = 0.62;
const BAND_GAP = 10;
const PADDING = { top: 10, right: 52, bottom: 18, left: 8 };
const FONT = '11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';

interface CornerChannel {
  readonly corner: string;
  readonly name: string;
  readonly unit: string;
  readonly low: number;
  readonly high: number;
}

type CornerTable = readonly (CornerChannel | undefined)[];

/**
 * One entry per corner in FL, FR, RL, RR order, resolved from the generated
 * contract, so no channel name is spelled out in this component.
 */
function cornerTable(baseName: string): CornerTable {
  const specs = cornerChannels(baseName);
  return CORNER_LABELS.map((corner) => {
    const spec = specs.find((entry) => entry.corner === corner);
    if (spec === undefined) {
      return undefined;
    }
    return {
      corner,
      name: spec.name,
      unit: spec.unit,
      low: spec.range[0],
      high: spec.range[1],
    };
  });
}

function cornerValue(
  frame: TelemetryFrame | null,
  table: CornerTable,
  index: number,
): number | undefined {
  const channel = table[index];
  if (frame === null || channel === undefined) {
    return undefined;
  }
  return frame.channels[channel.name];
}

function metric(value: number | undefined, unit: string): string {
  return value === undefined ? '—' : `${formatValue(value)} ${unit}`;
}

const TYRE_TEMP = cornerTable('tyre_temp');
const TYRE_PRESSURE = cornerTable('tyre_pressure');
const VERTICAL_LOAD = cornerTable('vertical_load');
const SLIP_RATIO = cornerTable('slip_ratio');

/** Per-channel draw buffers, reallocated only when the plot width changes. */
interface TraceBuffer {
  readonly channel: CornerChannel | undefined;
  readonly colour: string;
  times: Float64Array | null;
  values: Float32Array | null;
  count: number;
  min: number;
  max: number;
}

interface BandScale {
  low: number;
  high: number;
  min: number;
  max: number;
  /** Corners with at least one usable sample in the window. */
  series: number;
}

function css(styles: CSSStyleDeclaration, name: string, fallback: string): string {
  return styles.getPropertyValue(name).trim() || fallback;
}

export function TyreBlock() {
  const headingId = useId();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const summaryRef = useRef<HTMLSpanElement | null>(null);

  // One selector per displayed channel. Every selector returns a primitive or
  // undefined, which is what keeps the subscription referentially stable while
  // the store republishes at the 30 Hz snapshot rate.
  const flTemp = useTelemetryStore((state) => cornerValue(state.frame, TYRE_TEMP, 0));
  const frTemp = useTelemetryStore((state) => cornerValue(state.frame, TYRE_TEMP, 1));
  const rlTemp = useTelemetryStore((state) => cornerValue(state.frame, TYRE_TEMP, 2));
  const rrTemp = useTelemetryStore((state) => cornerValue(state.frame, TYRE_TEMP, 3));
  const flPressure = useTelemetryStore((state) => cornerValue(state.frame, TYRE_PRESSURE, 0));
  const frPressure = useTelemetryStore((state) => cornerValue(state.frame, TYRE_PRESSURE, 1));
  const rlPressure = useTelemetryStore((state) => cornerValue(state.frame, TYRE_PRESSURE, 2));
  const rrPressure = useTelemetryStore((state) => cornerValue(state.frame, TYRE_PRESSURE, 3));
  const flLoad = useTelemetryStore((state) => cornerValue(state.frame, VERTICAL_LOAD, 0));
  const frLoad = useTelemetryStore((state) => cornerValue(state.frame, VERTICAL_LOAD, 1));
  const rlLoad = useTelemetryStore((state) => cornerValue(state.frame, VERTICAL_LOAD, 2));
  const rrLoad = useTelemetryStore((state) => cornerValue(state.frame, VERTICAL_LOAD, 3));
  const flSlip = useTelemetryStore((state) => cornerValue(state.frame, SLIP_RATIO, 0));
  const frSlip = useTelemetryStore((state) => cornerValue(state.frame, SLIP_RATIO, 1));
  const rlSlip = useTelemetryStore((state) => cornerValue(state.frame, SLIP_RATIO, 2));
  const rrSlip = useTelemetryStore((state) => cornerValue(state.frame, SLIP_RATIO, 3));

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null) {
      return;
    }
    const context = canvas.getContext('2d', { alpha: false });
    if (context === null) {
      return;
    }
    const history = useTelemetryStore.getState().history;
    const styles = getComputedStyle(canvas);
    const surface = css(styles, '--trace-surface', '#0d1117');
    const gridColour = css(styles, '--trace-grid', 'rgba(255,255,255,0.12)');
    const textColour = css(styles, '--trace-text', '#9aa7b4');
    const zeroColour = css(styles, '--trace-line', '#58a6ff');
    const cornerColours = CORNER_LABELS.map((corner) =>
      css(styles, `--corner-${corner.toLowerCase()}`, zeroColour),
    );

    const windowUs = WINDOW_SECONDS * 1e6;
    let startUs = 0;
    let plotLeft = PADDING.left;
    let plotWidth = 2;
    let request = 0;
    let lastSummaryAt = 0;

    const makeBuffers = (table: CornerTable): TraceBuffer[] =>
      table.map((channel, index) => ({
        channel,
        colour: cornerColours[index] ?? zeroColour,
        times: null,
        values: null,
        count: 0,
        min: Number.NaN,
        max: Number.NaN,
      }));

    const tempBuffers = makeBuffers(TYRE_TEMP);
    const slipBuffers = makeBuffers(SLIP_RATIO);

    // One fill per channel per frame: min/max decimation to the pixel width
    // happens inside the ring buffer, so the loop never walks a full-rate trace.
    const collect = (buffers: TraceBuffer[], points: number): void => {
      for (const buffer of buffers) {
        if (buffer.channel === undefined) {
          buffer.count = 0;
          buffer.min = Number.NaN;
          buffer.max = Number.NaN;
          continue;
        }
        let times = buffer.times;
        let values = buffer.values;
        if (times === null || values === null || times.length !== points) {
          times = new Float64Array(points);
          values = new Float32Array(points);
          buffer.times = times;
          buffer.values = values;
        }
        buffer.count = history.fill(buffer.channel.name, points, times, values, startUs);
        let min = Number.POSITIVE_INFINITY;
        let max = Number.NEGATIVE_INFINITY;
        for (let index = 0; index < buffer.count; index += 1) {
          const value = values[index] ?? Number.NaN;
          if (Number.isNaN(value)) {
            continue;
          }
          if (value < min) {
            min = value;
          }
          if (value > max) {
            max = value;
          }
        }
        buffer.min = min;
        buffer.max = max;
      }
    };

    // Data-driven range, so a cold tyre and a cooked one both read clearly.
    // With nothing in the window the declared contract range is the axis, which
    // is honest about what the signal is allowed to do.
    const scale = (buffers: TraceBuffer[]): BandScale => {
      let min = Number.POSITIVE_INFINITY;
      let max = Number.NEGATIVE_INFINITY;
      let series = 0;
      for (const buffer of buffers) {
        if (!Number.isFinite(buffer.min) || !Number.isFinite(buffer.max)) {
          continue;
        }
        series += 1;
        min = Math.min(min, buffer.min);
        max = Math.max(max, buffer.max);
      }
      if (series === 0) {
        const channel = buffers[0]?.channel;
        return {
          low: channel?.low ?? 0,
          high: channel?.high ?? 1,
          min: Number.NaN,
          max: Number.NaN,
          series: 0,
        };
      }
      const spread = Math.max(max - min, Math.abs(max) * 0.02, 1e-6);
      return { low: min - spread * 0.08, high: max + spread * 0.08, min, max, series };
    };

    const axes = (top: number, height: number, band: BandScale): void => {
      context.strokeStyle = gridColour;
      context.fillStyle = textColour;
      context.lineWidth = 1;
      context.font = FONT;
      for (let row = 0; row <= GRID_ROWS; row += 1) {
        const ratio = row / GRID_ROWS;
        const y = Math.round(top + height * ratio) + 0.5;
        context.beginPath();
        context.moveTo(plotLeft, y);
        context.lineTo(plotLeft + plotWidth, y);
        context.stroke();
        context.textAlign = 'left';
        context.textBaseline = 'middle';
        context.fillText(formatValue(band.high - (band.high - band.low) * ratio), plotLeft + plotWidth + 6, y);
      }
      if (band.low < 0 && band.high > 0) {
        // Slip ratio is signed: an emphasised zero line is the reference a
        // driver actually reads, so it outranks the grid.
        const y = Math.round(top + height - ((0 - band.low) / (band.high - band.low)) * height) + 0.5;
        context.strokeStyle = zeroColour;
        context.beginPath();
        context.moveTo(plotLeft, y);
        context.lineTo(plotLeft + plotWidth, y);
        context.stroke();
      }
    };

    const traces = (
      buffers: TraceBuffer[],
      band: BandScale,
      top: number,
      height: number,
    ): void => {
      const span = band.high - band.low || 1;
      for (const buffer of buffers) {
        const times = buffer.times;
        const values = buffer.values;
        if (times === null || values === null || buffer.count === 0) {
          continue;
        }
        context.beginPath();
        let penDown = false;
        for (let index = 0; index < buffer.count; index += 1) {
          const time = times[index] ?? 0;
          const value = values[index] ?? Number.NaN;
          if (Number.isNaN(value)) {
            penDown = false;
            continue;
          }
          const ratioX = (time - startUs) / windowUs;
          const x = plotLeft + Math.min(1, Math.max(0, ratioX)) * plotWidth;
          const y = top + height - ((value - band.low) / span) * height;
          if (penDown) {
            context.lineTo(x, y);
          } else {
            context.moveTo(x, y);
            penDown = true;
          }
        }
        context.strokeStyle = buffer.colour;
        context.lineWidth = 1.5;
        context.lineJoin = 'round';
        context.lineCap = 'round';
        context.stroke();
      }
    };

    const empty = (top: number, height: number): void => {
      context.fillStyle = textColour;
      context.textAlign = 'center';
      context.textBaseline = 'middle';
      context.fillText('No samples in window', plotLeft + plotWidth / 2, top + height / 2);
    };

    const timeAxis = (bottom: number): void => {
      context.fillStyle = textColour;
      context.strokeStyle = gridColour;
      context.font = FONT;
      context.textAlign = 'center';
      context.textBaseline = 'top';
      for (let tick = 0; tick <= 3; tick += 1) {
        const ratio = tick / 3;
        const x = Math.round(plotLeft + plotWidth * ratio) + 0.5;
        context.beginPath();
        context.moveTo(x, bottom + 1);
        context.lineTo(x, bottom + 4);
        context.stroke();
        const seconds = Math.round((ratio - 1) * WINDOW_SECONDS);
        context.fillText(seconds === 0 ? 'now' : `${seconds}s`, x, bottom + 6);
      }
    };

    const draw = (now: number): void => {
      request = requestAnimationFrame(draw);
      const bounds = canvas.getBoundingClientRect();
      const width = Math.max(1, Math.round(bounds.width));
      const height = Math.max(1, Math.round(bounds.height));
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      if (canvas.width !== Math.round(width * ratio)) {
        canvas.width = Math.round(width * ratio);
      }
      if (canvas.height !== Math.round(height * ratio)) {
        canvas.height = Math.round(height * ratio);
      }
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.fillStyle = surface;
      context.fillRect(0, 0, width, height);

      plotLeft = PADDING.left;
      plotWidth = Math.max(2, width - PADDING.left - PADDING.right);
      const plotTop = PADDING.top;
      const plotHeight = Math.max(4, height - PADDING.top - PADDING.bottom);
      const tempHeight = Math.max(2, Math.round((plotHeight - BAND_GAP) * TEMP_BAND_RATIO));
      const slipTop = plotTop + tempHeight + BAND_GAP;
      const slipHeight = Math.max(2, plotHeight - tempHeight - BAND_GAP);

      startUs = history.latestTimeUs() - windowUs;
      const points = Math.max(2, plotWidth);
      collect(tempBuffers, points);
      collect(slipBuffers, points);

      const temp = scale(tempBuffers);
      axes(plotTop, tempHeight, temp);
      if (temp.series === 0) {
        empty(plotTop, tempHeight);
      } else {
        traces(tempBuffers, temp, plotTop, tempHeight);
      }

      const slip = scale(slipBuffers);
      axes(slipTop, slipHeight, slip);
      if (slip.series === 0) {
        empty(slipTop, slipHeight);
      } else {
        traces(slipBuffers, slip, slipTop, slipHeight);
      }

      timeAxis(slipTop + slipHeight);

      if (now - lastSummaryAt >= ARIA_INTERVAL_MS) {
        lastSummaryAt = now;
        canvas.setAttribute(
          'aria-label',
          `Carcass temperature and slip ratio per corner over the last ${WINDOW_SECONDS} seconds. ${
            temp.series === 0
              ? 'No tyre temperature samples in the window.'
              : `Tyre temperature ${formatValue(temp.min)} to ${formatValue(temp.max)} degrees Celsius.`
          } ${
            slip.series === 0
              ? 'No slip ratio samples in the window.'
              : `Slip ratio ${formatValue(slip.min)} to ${formatValue(slip.max)} percent.`
          }`,
        );
        if (summaryRef.current !== null) {
          summaryRef.current.textContent = `tyre ${
            temp.series === 0 ? '—' : `${formatValue(temp.min)} to ${formatValue(temp.max)} °C`
          } · slip ${
            slip.series === 0 ? '—' : `${formatValue(slip.min)} to ${formatValue(slip.max)} %`
          }`;
        }
      }
    };

    request = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(request);
  }, []);

  const corners = [
    { corner: 'FL', temp: flTemp, pressure: flPressure, load: flLoad, slip: flSlip },
    { corner: 'FR', temp: frTemp, pressure: frPressure, load: frLoad, slip: frSlip },
    { corner: 'RL', temp: rlTemp, pressure: rlPressure, load: rlLoad, slip: rlSlip },
    { corner: 'RR', temp: rrTemp, pressure: rrPressure, load: rrLoad, slip: rrSlip },
  ];

  return (
    <section className="block" aria-labelledby={headingId}>
      <div className="block__header">
        <h2 className="block__heading" id={headingId}>
          Tyres
        </h2>
        <span className="block__note">four-corner temp, pressure, load, slip</span>
      </div>
      <dl className="corners">
        {corners.map((corner) => (
          <div
            key={corner.corner}
            className={`corners__cell corners__cell--${corner.corner.toLowerCase()}`}
          >
            <dt className="corners__name">{corner.corner}</dt>
            <dd className="corners__metrics">
              <span className="corners__metric">
                <span className="corners__key">temp</span>
                {metric(corner.temp, '°C')}
              </span>
              <span className="corners__metric">
                <span className="corners__key">press</span>
                {metric(corner.pressure, 'psi')}
              </span>
              <span className="corners__metric">
                <span className="corners__key">load</span>
                {metric(corner.load, 'N')}
              </span>
              <span className="corners__metric">
                <span className="corners__key">slip</span>
                {metric(corner.slip, '%')}
              </span>
            </dd>
          </div>
        ))}
      </dl>
      <figure className="block__figure">
        <canvas
          ref={canvasRef}
          className="block__canvas"
          role="img"
          aria-label="Tyre carcass temperature and slip ratio history"
        />
        <figcaption className="block__caption">
          <span className="block__legend">
            <span className="corners__swatch corners__cell--fl" />
            FL
            <span className="corners__swatch corners__cell--fr" />
            FR
            <span className="corners__swatch corners__cell--rl" />
            RL
            <span className="corners__swatch corners__cell--rr" />
            RR
          </span>
          <span ref={summaryRef}>—</span>
        </figcaption>
      </figure>
    </section>
  );
}