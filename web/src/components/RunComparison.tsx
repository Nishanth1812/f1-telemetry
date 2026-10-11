import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { useTelemetryStore } from '../store/telemetryStore';
import { TraceHistory } from '../telemetry/history';
import { formatValue } from '../telemetry/format';
import type { TelemetrySample } from '../telemetry/types';

export interface RunComparisonProps {
  replayFrames?: TelemetrySample[];
}

export function RunComparison({ replayFrames }: RunComparisonProps) {
  const replayHistory = useRef(new TraceHistory()).current;
  const [replayLoaded, setReplayLoaded] = useState(false);
  const liveHistory = useTelemetryStore((state) => state.history);
  const liveChannelNames = useTelemetryStore((state) => state.channelNames);
  const replayChannelNames = useMemo(() => {
    const names = new Set<string>();
    for (const frame of replayFrames ?? []) {
      for (const name of Object.keys(frame.channels)) {
        names.add(name);
      }
    }
    return Array.from(names).sort();
  }, [replayFrames]);
  const activeChannel = replayChannelNames[0] ?? liveChannelNames[0] ?? '';
  const channelId = useId();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!replayFrames) {
      setReplayLoaded(false);
      return;
    }
    replayHistory.clear();
    for (const frame of replayFrames) {
      replayHistory.push(frame);
    }
    setReplayLoaded(true);
  }, [replayFrames]);

  const replayLatestUs = replayHistory.latestTimeUs();
  const replayCount = replayHistory.trackedChannels;

  // Compute per-channel delta: live value at its latest time vs replay value at replay's latest time
  const deltas = useMemo(() => {
    const result: Record<string, { live: number; replay: number; delta: number } | null> = {};
    const channels = activeChannel ? [activeChannel] : (replayChannelNames.length > 0 ? replayChannelNames : liveChannelNames);
    for (const ch of channels) {
      const liveTimes = new Float64Array(4);
      const liveValues = new Float32Array(4);
      const replayTimes = new Float64Array(4);
      const replayValues = new Float32Array(4);
      const liveCount = liveHistory.fill(ch, 4, liveTimes, liveValues);
      const replayCount = replayHistory.fill(ch, 4, replayTimes, replayValues);
      const liveValue = liveCount > 0 ? liveValues[liveCount - 1] ?? Number.NaN : Number.NaN;
      const replayValue = replayCount > 0 ? replayValues[replayCount - 1] ?? Number.NaN : Number.NaN;
      if (Number.isFinite(liveValue) && Number.isFinite(replayValue)) {
        result[ch] = { live: liveValue, replay: replayValue, delta: liveValue - replayValue };
      } else {
        result[ch] = null;
      }
    }
    return result;
  }, [liveHistory, replayHistory, replayLatestUs, activeChannel, replayChannelNames, liveChannelNames]);

  const deltaReadouts = useMemo(() => {
    const entries: Array<[string, { live: number; replay: number; delta: number } | null]> = Object.entries(deltas);
    return entries.slice(0, Math.min(entries.length, 6));
  }, [deltas]);

  // Basic overlay canvas: replay trace drawn at reduced opacity over a surface
  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null || !replayLoaded || replayLatestUs <= 0) return;
    const context = canvas.getContext('2d', { alpha: false });
    if (context === null) return;
    const styles = getComputedStyle(canvas);
    const surface = styles.getPropertyValue('--trace-surface').trim() || '#0d1117';
    const replayLine = '#58a6ff';
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

    const plotLeft = 10;
    const plotTop = 12;
    const plotWidth = Math.max(2, width - 86);
    const plotHeight = Math.max(1, height - 34);

    const times = new Float64Array(Math.max(2, plotWidth));
    const values = new Float32Array(Math.max(2, plotWidth));
    const windowUs = 10 * 1e6;
    const count = replayHistory.fill(activeChannel, Math.max(2, plotWidth), times, values, replayLatestUs - windowUs);

    if (count > 0) {
      const firstUs = times[0] ?? replayLatestUs;
      const lastUs = times[count - 1] ?? replayLatestUs;
      const timeSpan = Math.max(lastUs - firstUs, 1);
      let min = Number.POSITIVE_INFINITY;
      let max = Number.NEGATIVE_INFINITY;
      for (let i = 0; i < count; i += 1) {
        const v = values[i] ?? Number.NaN;
        if (Number.isFinite(v)) {
          min = Math.min(min, v);
          max = Math.max(max, v);
        }
      }
      if (Number.isFinite(min) && Number.isFinite(max)) {
        const spread = Math.max(max - min, Math.abs(max) * 1e-6, 1e-9);
        const low = min - spread * 0.1;
        const high = max + spread * 0.1;
        context.beginPath();
        let penDown = false;
        for (let i = 0; i < count; i += 1) {
          const t = times[i] ?? 0;
          const v = values[i] ?? Number.NaN;
          if (Number.isNaN(v) || !Number.isFinite(v)) {
            penDown = false;
            continue;
          }
          const ratioX = (t - firstUs) / timeSpan;
          const x = plotLeft + Math.min(1, Math.max(0, ratioX)) * plotWidth;
          const y = plotTop + plotHeight - ((v - low) / (high - low)) * plotHeight;
          if (penDown) {
            context.lineTo(x, y);
          } else {
            context.moveTo(x, y);
            penDown = true;
          }
        }
        context.strokeStyle = replayLine;
        context.lineWidth = 1.5;
        context.lineJoin = 'round';
        context.lineCap = 'round';
        context.stroke();
      }
    }
  }, [replayLoaded, replayHistory, replayLatestUs, replayCount, activeChannel]);

  return (
    <section className="lap" aria-labelledby={channelId}>
      <div className="lap__header">
        <h2 className="lap__heading" id={channelId}>
          Run comparison
        </h2>
        <span className="lap__note">
          replay vs live · {replayLoaded ? `replay ${replayCount} channels` : 'no replay loaded'}
        </span>
      </div>
      <figure className="lap__figure">
        <canvas
          ref={canvasRef}
          className="lap__canvas"
          role="img"
          aria-label={replayLoaded ? `Replay overlay for ${activeChannel}` : 'Run comparison overlay canvas'}
        />
        <figcaption className="lap__caption">
          <span>
            replay latest {(replayLatestUs / 1e6).toFixed(3)} s · channels {replayChannelNames.join(', ') || '—'}
          </span>
        </figcaption>
      </figure>
      <div className="lap__current">
        <span className="lap__current-label">delta (live − replay)</span>
        {deltaReadouts.map(([name, deltaObj]) => (
          <span key={name} className="lap__current-sector">
            <span className="lap__key">{name}</span>
            {deltaObj === null ? (
              <span>—</span>
            ) : (
              <span>
                live {formatValue(deltaObj.live)} · replay {formatValue(deltaObj.replay)} ·
                <strong style={{ color: deltaObj.delta > 0 ? '#3fb950' : deltaObj.delta < 0 ? '#f4695f' : '#9aa7b4' }}>
                  {' '}{deltaObj.delta > 0 ? '+' : ''}{deltaObj.delta.toFixed(3)}
                </strong>
              </span>
            )}
          </span>
        ))}
      </div>
    </section>
  );
}
