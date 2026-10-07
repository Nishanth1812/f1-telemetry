import { useEffect, useId, useRef, useState } from 'react';
import { useTelemetryStore } from '../store/telemetryStore';
import { formatValue } from '../telemetry/format';

const MAX_POINTS = 1_200;
const WINDOW_OPTIONS = [5, 10, 30, 60] as const;
const STATS_INTERVAL_MS = 250;
const GRID_ROWS = 4;
const PADDING = { top: 12, right: 76, bottom: 22, left: 10 };

export function TracePanel() {
  const channelNames = useTelemetryStore((state) => state.channelNames);
  const [channel, setChannel] = useState('');
  const [windowSeconds, setWindowSeconds] = useState(10);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const latestRef = useRef<HTMLSpanElement | null>(null);
  const rangeRef = useRef<HTMLSpanElement | null>(null);
  const channelId = useId();
  const windowId = useId();

  const activeChannel = channelNames.includes(channel) ? channel : (channelNames[0] ?? '');

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null || activeChannel === '') {
      return;
    }
    const context = canvas.getContext('2d', { alpha: false });
    if (context === null) {
      return;
    }
    const history = useTelemetryStore.getState().history;
    const times = new Float64Array(MAX_POINTS);
    const values = new Float32Array(MAX_POINTS);
    const styles = getComputedStyle(canvas);
    const surface = styles.getPropertyValue('--trace-surface').trim() || '#0d1117';
    const line = styles.getPropertyValue('--trace-line').trim() || '#58a6ff';
    const grid = styles.getPropertyValue('--trace-grid').trim() || 'rgba(255,255,255,0.14)';
    const text = styles.getPropertyValue('--trace-text').trim() || '#9aa7b4';
    const windowUs = windowSeconds * 1e6;
    let request = 0;
    let lastStatsAt = 0;

    const inWindow = (time: number, latestUs: number) => time >= latestUs - windowUs;

    const draw = (now: number) => {
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

      const plotLeft = PADDING.left;
      const plotTop = PADDING.top;
      const plotWidth = Math.max(1, width - PADDING.left - PADDING.right);
      const plotHeight = Math.max(1, height - PADDING.top - PADDING.bottom);

      const latestUs = history.latestTimeUs();
      const count = history.fill(activeChannel, MAX_POINTS, times, values);
      let plotted = 0;
      let min = Number.POSITIVE_INFINITY;
      let max = Number.NEGATIVE_INFINITY;
      let firstUs = 0;
      let lastUs = 0;
      let lastValue = Number.NaN;
      for (let index = 0; index < count; index += 1) {
        const time = times[index] ?? 0;
        const value = values[index] ?? Number.NaN;
        if (Number.isNaN(value) || !inWindow(time, latestUs)) {
          continue;
        }
        if (plotted === 0) {
          firstUs = time;
        }
        lastUs = time;
        lastValue = value;
        min = Math.min(min, value);
        max = Math.max(max, value);
        plotted += 1;
      }

      const hasData = plotted > 0;
      const spread = hasData ? Math.max(max - min, Math.abs(max) * 1e-6, 1e-9) : 1;
      const low = hasData ? min - spread * 0.1 : 0;
      const high = hasData ? max + spread * 0.1 : 1;
      const timeSpan = hasData && lastUs > firstUs ? lastUs - firstUs : 0;

      context.strokeStyle = grid;
      context.fillStyle = text;
      context.lineWidth = 1;
      context.font = '11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
      for (let row = 0; row <= GRID_ROWS; row += 1) {
        const ratioY = row / GRID_ROWS;
        const y = Math.round(plotTop + plotHeight * ratioY) + 0.5;
        context.beginPath();
        context.moveTo(plotLeft, y);
        context.lineTo(plotLeft + plotWidth, y);
        context.stroke();
        context.textAlign = 'left';
        context.textBaseline = 'middle';
        context.fillText(formatValue(high - (high - low) * ratioY), plotLeft + plotWidth + 8, y);
        const x = Math.round(plotLeft + plotWidth * ratioY) + 0.5;
        context.beginPath();
        context.moveTo(x, plotTop);
        context.lineTo(x, plotTop + plotHeight);
        context.stroke();
        context.textAlign = 'center';
        context.textBaseline = 'top';
        const offsetSeconds = timeSpan > 0 ? (timeSpan * ratioY) / 1e6 : Number.NaN;
        context.fillText(
          Number.isFinite(offsetSeconds) ? `${offsetSeconds.toFixed(1)}s` : '—',
          x,
          plotTop + plotHeight + 5,
        );
      }

      context.beginPath();
      let penDown = false;
      for (let index = 0; index < count; index += 1) {
        const time = times[index] ?? 0;
        const value = values[index] ?? Number.NaN;
        if (Number.isNaN(value) || !inWindow(time, latestUs)) {
          penDown = false;
          continue;
        }
        const ratioX = timeSpan > 0 ? (time - firstUs) / timeSpan : index / Math.max(1, plotted - 1);
        const x = plotLeft + Math.min(1, Math.max(0, ratioX)) * plotWidth;
        const y = plotTop + plotHeight - ((value - low) / (high - low)) * plotHeight;
        if (penDown) {
          context.lineTo(x, y);
        } else {
          context.moveTo(x, y);
          penDown = true;
        }
      }
      context.strokeStyle = line;
      context.lineWidth = 1.5;
      context.lineJoin = 'round';
      context.stroke();

      if (!hasData) {
        context.fillStyle = text;
        context.textAlign = 'center';
        context.textBaseline = 'middle';
        context.fillText('No samples in window', plotLeft + plotWidth / 2, plotTop + plotHeight / 2);
      }

      if (now - lastStatsAt >= STATS_INTERVAL_MS) {
        lastStatsAt = now;
        canvas.setAttribute(
          'aria-label',
          hasData
            ? `${activeChannel}: latest ${formatValue(lastValue)}, min ${formatValue(min)}, max ${formatValue(max)}, over the last ${windowSeconds} seconds`
            : `${activeChannel}: no samples in the last ${windowSeconds} seconds`,
        );
        if (latestRef.current !== null) {
          latestRef.current.textContent = hasData ? formatValue(lastValue) : '—';
        }
        if (rangeRef.current !== null) {
          rangeRef.current.textContent = hasData
            ? `min ${formatValue(min)} · max ${formatValue(max)} · ${plotted} pts`
            : 'no samples in window';
        }
      }
    };

    request = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(request);
  }, [activeChannel, windowSeconds]);

  const empty = channelNames.length === 0;

  return (
    <section className="trace" aria-labelledby="trace-heading">
      <div className="trace__header">
        <h2 className="trace__heading" id="trace-heading">
          Trace
        </h2>
        <div className="trace__controls">
          <label htmlFor={channelId}>Channel</label>
          <select
            id={channelId}
            value={activeChannel}
            disabled={empty}
            onChange={(event) => setChannel(event.target.value)}
          >
            {empty ? <option value="">No channels</option> : null}
            {channelNames.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
          <label htmlFor={windowId}>Window</label>
          <select
            id={windowId}
            value={windowSeconds}
            onChange={(event) => setWindowSeconds(Number(event.target.value))}
          >
            {WINDOW_OPTIONS.map((seconds) => (
              <option key={seconds} value={seconds}>
                {seconds} s
              </option>
            ))}
          </select>
        </div>
      </div>
      <figure className="trace__figure">
        <canvas
          ref={canvasRef}
          className="trace__canvas"
          role="img"
          aria-label={activeChannel === '' ? 'Trace channel' : `${activeChannel} trace`}
        />
        <figcaption className="trace__caption">
          <span className="trace__caption-item">
            latest <span ref={latestRef}>—</span>
          </span>
          <span className="trace__caption-item" ref={rangeRef}>
            no samples in window
          </span>
        </figcaption>
      </figure>
    </section>
  );
}
