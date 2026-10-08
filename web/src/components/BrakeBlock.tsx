import { useEffect, useId, useRef } from 'react';
import { cornerChannels } from '../generated/channels';
import { useTelemetryStore } from '../store/telemetryStore';
import { formatValue } from '../telemetry/format';
import type { TelemetryFrame } from '../telemetry/types';

/**
 * Per-corner brake disc temperature (PLAN.md section 8.4).
 *
 * Same split as the other blocks: the four live numbers arrive through plain
 * selector subscriptions to the 30 Hz store snapshot, while the thirty-second
 * history of all four corners - including the window peak-hold ticks - is drawn
 * on canvas from the ring buffers. The axis is absolute rather than data-driven
 * where it can be, because a disc at 180 degrees and one at 900 degrees are
 * different stories even when both traces look similar in shape.
 */

const CORNER_LABELS = ['FL', 'FR', 'RL', 'RR'] as const;
const WINDOW_SECONDS = 30;
const GRID_ROWS = 4;
const ARIA_INTERVAL_MS = 250;
/** Axis floor, so a cold-out stint still has resolution without hiding the top. */
const MIN_AXIS_N = 400;
const PADDING = { top: 10, right: 52, bottom: 18, left: 8 };
const FONT = '11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
const PEAK_TICK_PX = 7;

interface CornerChannel {
  readonly corner: string;
  readonly name: string;
  readonly unit: string;
  readonly low: number;
  readonly high: number;
}

type CornerTable = readonly (CornerChannel | undefined)[];

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

const BRAKE_TEMP = cornerTable('brake_temp');
const CEILING = BRAKE_TEMP[0]?.high ?? 1_100;

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
  series: number;
}

function css(styles: CSSStyleDeclaration, name: string, fallback: string): string {
  return styles.getPropertyValue(name).trim() || fallback;
}

export function BrakeBlock() {
  const headingId = useId();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const summaryRef = useRef<HTMLSpanElement | null>(null);

  const flTemp = useTelemetryStore((state) => cornerValue(state.frame, BRAKE_TEMP, 0));
  const frTemp = useTelemetryStore((state) => cornerValue(state.frame, BRAKE_TEMP, 1));
  const rlTemp = useTelemetryStore((state) => cornerValue(state.frame, BRAKE_TEMP, 2));
  const rrTemp = useTelemetryStore((state) => cornerValue(state.frame, BRAKE_TEMP, 3));

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
    const ceilingColour = css(styles, '--warn', '#d8a026');
    const cornerColours = CORNER_LABELS.map((corner) =>
      css(styles, `--corner-${corner.toLowerCase()}`, textColour),
    );

    const windowUs = WINDOW_SECONDS * 1e6;
    let startUs = 0;
    let plotLeft = PADDING.left;
    let plotWidth = 2;
    let request = 0;
    let lastSummaryAt = 0;

    const buffers: TraceBuffer[] = BRAKE_TEMP.map((channel, index) => ({
      channel,
      colour: cornerColours[index] ?? textColour,
      times: null,
      values: null,
      count: 0,
      min: Number.NaN,
      max: Number.NaN,
    }));

    const collect = (points: number): void => {
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

    // Zero is the floor, so trace height always reads as "how hot", and the
    // declared ceiling enters the axis whenever the disc gets near it.
    const scale = (): BandScale => {
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
        return { low: 0, high: CEILING, min: Number.NaN, max: Number.NaN, series: 0 };
      }
      const spread = Math.max(max - min, max * 0.02, 1);
      return {
        low: 0,
        high: Math.max(max + spread * 0.08, MIN_AXIS_N),
        min,
        max,
        series,
      };
    };

    const y = (value: number, top: number, height: number, band: BandScale): number =>
      top + height - ((value - band.low) / (band.high - band.low)) * height;

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

      startUs = history.latestTimeUs() - windowUs;
      collect(Math.max(2, plotWidth));
      const band = scale();

      context.strokeStyle = gridColour;
      context.fillStyle = textColour;
      context.lineWidth = 1;
      context.font = FONT;
      for (let row = 0; row <= GRID_ROWS; row += 1) {
        const ratioY = row / GRID_ROWS;
        const gridY = Math.round(plotTop + plotHeight * ratioY) + 0.5;
        context.beginPath();
        context.moveTo(plotLeft, gridY);
        context.lineTo(plotLeft + plotWidth, gridY);
        context.stroke();
        context.textAlign = 'left';
        context.textBaseline = 'middle';
        context.fillText(
          formatValue(band.high - (band.high - band.low) * ratioY),
          plotLeft + plotWidth + 6,
          gridY,
        );
      }

      if (CEILING <= band.high) {
        // The declared contract ceiling, only while the axis actually reaches it.
        const ceilingY = Math.round(y(CEILING, plotTop, plotHeight, band)) + 0.5;
        context.strokeStyle = ceilingColour;
        context.setLineDash([4, 3]);
        context.beginPath();
        context.moveTo(plotLeft, ceilingY);
        context.lineTo(plotLeft + plotWidth, ceilingY);
        context.stroke();
        context.setLineDash([]);
        context.fillStyle = ceilingColour;
        context.textAlign = 'left';
        context.textBaseline = 'bottom';
        context.fillText(`ceiling ${formatValue(CEILING)}`, plotLeft + 4, ceilingY - 2);
      }

      if (band.series === 0) {
        context.fillStyle = textColour;
        context.textAlign = 'center';
        context.textBaseline = 'middle';
        context.fillText('No samples in window', plotLeft + plotWidth / 2, plotTop + plotHeight / 2);
      }

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
          if (penDown) {
            context.lineTo(x, y(value, plotTop, plotHeight, band));
          } else {
            context.moveTo(x, y(value, plotTop, plotHeight, band));
            penDown = true;
          }
        }
        context.strokeStyle = buffer.colour;
        context.lineWidth = 1.5;
        context.lineJoin = 'round';
        context.lineCap = 'round';
        context.stroke();

        // Peak-hold tick at the leading edge: where this corner got to inside
        // the window, which is what a driver judges a braking zone against.
        if (Number.isFinite(buffer.max)) {
          const peakY = Math.round(y(buffer.max, plotTop, plotHeight, band)) + 0.5;
          context.beginPath();
          context.moveTo(plotLeft + plotWidth - PEAK_TICK_PX, peakY);
          context.lineTo(plotLeft + plotWidth, peakY);
          context.stroke();
        }
      }

      const axisBottom = plotTop + plotHeight;
      context.strokeStyle = gridColour;
      context.fillStyle = textColour;
      context.font = FONT;
      context.textAlign = 'center';
      context.textBaseline = 'top';
      for (let tick = 0; tick <= 3; tick += 1) {
        const ratioX = tick / 3;
        const tickX = Math.round(plotLeft + plotWidth * ratioX) + 0.5;
        context.beginPath();
        context.moveTo(tickX, axisBottom + 1);
        context.lineTo(tickX, axisBottom + 4);
        context.stroke();
        const seconds = Math.round((ratioX - 1) * WINDOW_SECONDS);
        context.fillText(seconds === 0 ? 'now' : `${seconds}s`, tickX, axisBottom + 6);
      }

      if (now - lastSummaryAt >= ARIA_INTERVAL_MS) {
        lastSummaryAt = now;
        const peaks = buffers
          .map((buffer) =>
            Number.isFinite(buffer.max)
              ? `${buffer.channel?.corner ?? '?'} ${formatValue(buffer.max)}`
              : null,
          )
          .filter((entry): entry is string => entry !== null);
        canvas.setAttribute(
          'aria-label',
          `Brake disc temperature per corner over the last ${WINDOW_SECONDS} seconds.${
            peaks.length === 0
              ? ' No brake temperature samples in the window.'
              : ` Window peaks, degrees Celsius: ${peaks.join(', ')}.`
          } Declared ceiling ${formatValue(CEILING)} degrees Celsius.`,
        );
        if (summaryRef.current !== null) {
          summaryRef.current.textContent =
            peaks.length === 0 ? 'no samples in window' : `peak ${peaks.join(' · ')} °C`;
        }
      }
    };

    request = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(request);
  }, []);

  const corners = [
    { corner: 'FL', temp: flTemp },
    { corner: 'FR', temp: frTemp },
    { corner: 'RL', temp: rlTemp },
    { corner: 'RR', temp: rrTemp },
  ];

  return (
    <section className="block" aria-labelledby={headingId}>
      <div className="block__header">
        <h2 className="block__heading" id={headingId}>
          Brakes
        </h2>
        <span className="block__note">per-corner disc temperature</span>
      </div>
      <dl className="corners corners--compact">
        {corners.map((corner) => (
          <div
            key={corner.corner}
            className={`corners__cell corners__cell--${corner.corner.toLowerCase()}`}
          >
            <dt className="corners__name">{corner.corner}</dt>
            <dd className="corners__metrics">
              <span className="corners__metric">
                <span className="corners__key">disc</span>
                {metric(corner.temp, '°C')}
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
          aria-label="Brake disc temperature history per corner"
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