import { useEffect, useId, useRef } from 'react';
import { CHANNELS } from '../generated/channels';
import { useTelemetryStore } from '../store/telemetryStore';
import { formatValue } from '../telemetry/format';

/**
 * Active aero panel (PLAN.md section 8.4): mode, front and rear wing flap
 * angle, and total downforce in newtons.
 *
 * The four live readouts are plain selector subscriptions to the throttled store
 * snapshot. The canvas carries the dense part - thirty seconds of downforce from
 * a 100 Hz channel, decimated to the pixel width inside TraceHistory.fill - plus
 * the two flap lanes, which are drawn against the declared mechanical travel of
 * each wing so a nearly-closed front wing reads differently from a wide-open
 * one. Mode tints both the lane fill and the badge. No React state in the path.
 */

const WINDOW_SECONDS = 30;
const GRID_ROWS = 3;
const ARIA_INTERVAL_MS = 250;
/** Axis floor for downforce, so a low-drag stint still has resolution. */
const MIN_AXIS_N = 2_000;
const LANE_HEIGHT = 30;
const LANE_LABEL_PX = 26;
const LANE_VALUE_PX = 34;
const LANE_THICKNESS = 8;
const PADDING = { top: 10, right: 56, bottom: 8, left: 8 };
const FONT = '11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';

const AERO_MODE = CHANNELS.aero_mode;
const FW_FLAP = CHANNELS.fw_flap_deg;
const RW_FLAP = CHANNELS.rw_flap_deg;
const DOWNFORCE = CHANNELS.downforce_n;
const DOWNFORCE_CEILING = DOWNFORCE.range[1];

const MODE_TEXT = {
  z: 'Z-mode · closed, high downforce',
  x: 'X-mode · open, low drag',
} as const;

type AeroMode = keyof typeof MODE_TEXT;

function modeOf(value: number | undefined): AeroMode | null {
  if (value === undefined) {
    return null;
  }
  const rounded = Math.round(value);
  return rounded === 0 ? 'z' : rounded === 1 ? 'x' : null;
}

function metric(value: number | undefined, unit: string): string {
  return value === undefined ? '—' : `${formatValue(value)} ${unit}`;
}

function angleText(value: number | undefined): string {
  return value === undefined ? '—' : `${formatValue(value)}°`;
}

function css(styles: CSSStyleDeclaration, name: string, fallback: string): string {
  return styles.getPropertyValue(name).trim() || fallback;
}

export function AeroBlock() {
  const headingId = useId();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const summaryRef = useRef<HTMLSpanElement | null>(null);

  const aeroMode = useTelemetryStore((state) => state.frame?.channels[AERO_MODE.name]);
  const fwFlap = useTelemetryStore((state) => state.frame?.channels[FW_FLAP.name]);
  const rwFlap = useTelemetryStore((state) => state.frame?.channels[RW_FLAP.name]);
  const downforce = useTelemetryStore((state) => state.frame?.channels[DOWNFORCE.name]);

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
    const traceColour = css(styles, '--trace-line', '#58a6ff');
    const zColour = css(styles, '--accent', '#58a6ff');
    const xColour = css(styles, '--warn', '#d8a026');

    const windowUs = WINDOW_SECONDS * 1e6;
    let startUs = 0;
    let plotLeft = PADDING.left;
    let plotWidth = 2;
    let plotTop = PADDING.top;
    let plotHeight = 4;
    let lanesTop = 0;
    let modeColour = zColour;
    let request = 0;
    let lastSummaryAt = 0;
    let times: Float64Array | null = null;
    let values: Float32Array | null = null;
    let count = 0;

    // One lane per wing, each mapped onto its declared mechanical travel. Zero is
    // inside travel on both wings, so the fill reads as a departure from neutral
    // rather than as an absolute position.
    const lane = (
      label: string,
      spec: typeof FW_FLAP,
      angle: number | undefined,
      index: number,
    ): void => {
      const centre = lanesTop + index * LANE_HEIGHT + LANE_HEIGHT / 2;
      const trackLeft = plotLeft + LANE_LABEL_PX;
      const trackRight = plotLeft + plotWidth - LANE_VALUE_PX;
      const [travelLow, travelHigh] = spec.range;
      const angleX = (angleValue: number): number =>
        trackLeft +
        Math.min(1, Math.max(0, (angleValue - travelLow) / (travelHigh - travelLow))) *
          (trackRight - trackLeft);
      const trackY = centre - LANE_THICKNESS / 2;

      context.fillStyle = textColour;
      context.font = FONT;
      context.textAlign = 'right';
      context.textBaseline = 'middle';
      context.fillText(label, trackLeft - 6, centre);

      context.globalAlpha = 0.18;
      context.fillStyle = modeColour;
      context.fillRect(trackLeft, trackY, Math.max(1, trackRight - trackLeft), LANE_THICKNESS);
      context.globalAlpha = 1;

      // Mechanical travel end stops.
      context.strokeStyle = gridColour;
      context.lineWidth = 1;
      for (const stop of [trackLeft, trackRight]) {
        context.beginPath();
        context.moveTo(stop + 0.5, trackY - 3);
        context.lineTo(stop + 0.5, trackY + LANE_THICKNESS + 3);
        context.stroke();
      }
      const zeroX = angleX(0);
      context.strokeStyle = textColour;
      context.beginPath();
      context.moveTo(zeroX + 0.5, trackY - 5);
      context.lineTo(zeroX + 0.5, trackY + LANE_THICKNESS + 5);
      context.stroke();

      if (angle !== undefined) {
        const valueX = angleX(angle);
        context.fillStyle = modeColour;
        context.fillRect(
          Math.min(zeroX, valueX),
          trackY,
          Math.max(2, Math.abs(valueX - zeroX)),
          LANE_THICKNESS,
        );
        context.fillRect(valueX - 1, trackY - 5, 2, LANE_THICKNESS + 10);
      }

      context.fillStyle = angle === undefined ? textColour : modeColour;
      context.textAlign = 'left';
      context.textBaseline = 'middle';
      context.fillText(angleText(angle), plotLeft + plotWidth + 6, centre);
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
      const lanesHeight = Math.min(height - PADDING.top - PADDING.bottom, LANE_HEIGHT * 2 + 6);
      plotTop = PADDING.top;
      plotHeight = Math.max(4, height - PADDING.top - PADDING.bottom - lanesHeight);
      lanesTop = plotTop + plotHeight + 6;

      // Latest snapshot for the lane needles: the same values the DOM readouts
      // render, read imperatively so the canvas needs no subscription.
      const snapshot = useTelemetryStore.getState().frame?.channels;
      const mode = modeOf(snapshot?.[AERO_MODE.name]);
      modeColour = mode === 'x' ? xColour : zColour;

      startUs = history.latestTimeUs() - windowUs;
      const points = Math.max(2, plotWidth);
      if (times === null || values === null || times.length !== points) {
        times = new Float64Array(points);
        values = new Float32Array(points);
      }
      count = history.fill(DOWNFORCE.name, points, times, values, startUs);
      let min = Number.NaN;
      let max = Number.NaN;
      for (let index = 0; index < count; index += 1) {
        const value = values[index] ?? Number.NaN;
        if (Number.isNaN(value)) {
          continue;
        }
        min = Number.isNaN(min) ? value : Math.min(min, value);
        max = Number.isNaN(max) ? value : Math.max(max, value);
      }
      const hasData = Number.isFinite(min) && Number.isFinite(max);
      const spread = hasData ? Math.max(max - min, max * 0.02, 1) : 1;
      const low = 0;
      const high = hasData ? Math.max(max + spread * 0.08, MIN_AXIS_N) : DOWNFORCE_CEILING;
      const y = (value: number): number =>
        plotTop + plotHeight - ((value - low) / (high - low)) * plotHeight;

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
          formatValue(high - (high - low) * ratioY),
          plotLeft + plotWidth + 6,
          gridY,
        );
      }

      if (DOWNFORCE_CEILING <= high) {
        const ceilingY = Math.round(y(DOWNFORCE_CEILING)) + 0.5;
        context.strokeStyle = xColour;
        context.setLineDash([4, 3]);
        context.beginPath();
        context.moveTo(plotLeft, ceilingY);
        context.lineTo(plotLeft + plotWidth, ceilingY);
        context.stroke();
        context.setLineDash([]);
        context.fillStyle = xColour;
        context.textAlign = 'left';
        context.textBaseline = 'bottom';
        context.fillText(`ceiling ${formatValue(DOWNFORCE_CEILING)} N`, plotLeft + 4, ceilingY - 2);
      }

      if (!hasData) {
        context.fillStyle = textColour;
        context.textAlign = 'center';
        context.textBaseline = 'middle';
        context.fillText('No samples in window', plotLeft + plotWidth / 2, plotTop + plotHeight / 2);
      } else {
        context.beginPath();
        let penDown = false;
        for (let index = 0; index < count; index += 1) {
          const time = times[index] ?? 0;
          const value = values[index] ?? Number.NaN;
          if (Number.isNaN(value)) {
            penDown = false;
            continue;
          }
          const ratioX = (time - startUs) / windowUs;
          const x = plotLeft + Math.min(1, Math.max(0, ratioX)) * plotWidth;
          if (penDown) {
            context.lineTo(x, y(value));
          } else {
            context.moveTo(x, y(value));
            penDown = true;
          }
        }
        context.strokeStyle = traceColour;
        context.lineWidth = 1.5;
        context.lineJoin = 'round';
        context.lineCap = 'round';
        context.stroke();
      }

      lane('FW', FW_FLAP, snapshot?.[FW_FLAP.name], 0);
      lane('RW', RW_FLAP, snapshot?.[RW_FLAP.name], 1);

      if (now - lastSummaryAt >= ARIA_INTERVAL_MS) {
        lastSummaryAt = now;
        const modeLabel =
          mode === null ? 'Aero mode not reported' : MODE_TEXT[mode];
        canvas.setAttribute(
          'aria-label',
          `Downforce history over the last ${WINDOW_SECONDS} seconds, ${
            hasData
              ? `${formatValue(min)} to ${formatValue(max)} newtons`
              : 'no samples in the window'
          }, declared ceiling ${formatValue(DOWNFORCE_CEILING)} newtons. ${modeLabel}. Front wing flap travel ${formatValue(FW_FLAP.range[0])} to ${formatValue(FW_FLAP.range[1])} degrees, currently ${angleText(snapshot?.[FW_FLAP.name])}. Rear wing flap travel ${formatValue(RW_FLAP.range[0])} to ${formatValue(RW_FLAP.range[1])} degrees, currently ${angleText(snapshot?.[RW_FLAP.name])}.`,
        );
        if (summaryRef.current !== null) {
          summaryRef.current.textContent = hasData
            ? `downforce ${formatValue(min)} to ${formatValue(max)} N over ${WINDOW_SECONDS}s`
            : `downforce ceiling ${formatValue(DOWNFORCE_CEILING)} N`;
        }
      }
    };

    request = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(request);
  }, []);

  const mode = modeOf(aeroMode);

  return (
    <section className="block" aria-labelledby={headingId}>
      <div className="block__header">
        <h2 className="block__heading" id={headingId}>
          Aero
        </h2>
        <span className="block__note">mode, wing flap travel, downforce</span>
      </div>
      <p className={`aero__mode aero__mode--${mode ?? 'unknown'}`}>
        {mode === null ? 'aero mode not reported' : MODE_TEXT[mode]}
      </p>
      <dl className="aero__readouts">
        <div className="aero__readout">
          <dt className="aero__key">FW flap</dt>
          <dd className="aero__value">{metric(fwFlap, '°')}</dd>
        </div>
        <div className="aero__readout">
          <dt className="aero__key">RW flap</dt>
          <dd className="aero__value">{metric(rwFlap, '°')}</dd>
        </div>
        <div className="aero__readout">
          <dt className="aero__key">downforce</dt>
          <dd className="aero__value">{metric(downforce, 'N')}</dd>
        </div>
      </dl>
      <figure className="block__figure">
        <canvas
          ref={canvasRef}
          className="block__canvas"
          role="img"
          aria-label="Downforce history with front and rear wing flap travel"
        />
        <figcaption className="block__caption">
          <span className="block__legend">downforce and flap travel</span>
          <span ref={summaryRef}>—</span>
        </figcaption>
      </figure>
    </section>
  );
}