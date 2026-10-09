import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { useTelemetryStore } from '../store/telemetryStore';
import { TraceHistory } from '../telemetry/history';
import { formatValue } from '../telemetry/format';
import type { TelemetrySample } from '../telemetry/types';

export interface ScrubberProps {
  replayFrames?: TelemetrySample[];
  onCursorChange?: (timeUs: number) => void;
}

export function Scrubber({ replayFrames, onCursorChange }: ScrubberProps) {
  const replayHistory = useRef(new TraceHistory()).current;
  const headingId = useId();
  const [cursorTimeUs, setCursorTimeUs] = useState(0);
  const [replayLoaded, setReplayLoaded] = useState(false);
  const replayLatestUs = replayHistory.latestTimeUs();
  const replayChannels = useMemo(() => {
    const names = new Set<string>();
    for (const frame of replayFrames ?? []) {
      for (const name of Object.keys(frame.channels)) {
        names.add(name);
      }
    }
    return Array.from(names).sort();
  }, [replayFrames]);

  useEffect(() => {
    if (!replayFrames) {
      replayHistory.clear();
      setReplayLoaded(false);
      setCursorTimeUs(0);
      return;
    }
    replayHistory.clear();
    for (const frame of replayFrames) {
      replayHistory.push(frame);
    }
    const latest = replayHistory.latestTimeUs();
    setReplayLoaded(true);
    setCursorTimeUs(latest > 0 ? latest : 0);
  }, [replayFrames]);

  useEffect(() => {
    if (typeof onCursorChange === 'function') {
      onCursorChange(cursorTimeUs);
    }
  }, [cursorTimeUs, onCursorChange]);

  const maxTimeUs = replayLatestUs > 0 ? replayLatestUs : 60 * 1e6;
  const stepMs = Math.max(1, Math.round(maxTimeUs / 1e3));

  const cursorTimeSec = cursorTimeUs / 1e6;

  const scrubbedValues = useMemo(() => {
    const result: Record<string, number | null> = {};
    const chNames = replayChannels.length > 0 ? replayChannels : Array.from(new Set(useTelemetryStore.getState().channelNames));
    for (const ch of chNames.slice(0, 6)) {
      const times = new Float64Array(4);
      const values = new Float32Array(4);
      const count = replayHistory.fill(ch, 4, times, values, cursorTimeUs);
      if (count > 0) {
        const lastValue = values[count - 1];
        result[ch] = lastValue !== undefined && Number.isFinite(lastValue) ? lastValue : null;
      } else {
        result[ch] = null;
      }
    }
    return result;
  }, [replayHistory, cursorTimeUs, replayChannels]);

  const fillAtCursor = useMemo(() => {
    const result: Record<string, { count: number; latestValue: number | null }> = {};
    for (const ch of replayChannels.slice(0, 6)) {
      const times = new Float64Array(8);
      const values = new Float32Array(8);
      const count = replayHistory.fill(ch, 8, times, values, cursorTimeUs);
      const latestValue = count > 0 && values[count - 1] !== undefined && Number.isFinite(values[count - 1]) ? values[count - 1] ?? null : null;
      result[ch] = { count, latestValue };
    }
    return result;
  }, [replayHistory, cursorTimeUs, replayChannels]);

  return (
    <section className="lap" aria-labelledby={headingId}>
      <div className="lap__header">
        <h2 className="lap__heading" id={headingId}>
          Scrubber
        </h2>
        <span className="lap__note">
          replay timeline · cursor at {(cursorTimeSec).toFixed(3)} s
        </span>
      </div>
      <div className="lap__readouts">
        <div className="lap__readout">
          <dt className="lap__key">cursor time</dt>
          <dd className="lap__value">{cursorTimeSec.toFixed(3)} s</dd>
        </div>
        <div className="lap__readout">
          <dt className="lap__key">replay span</dt>
          <dd className="lap__value">{(replayLatestUs / 1e6).toFixed(3)} s</dd>
        </div>
        <div className="lap__readout">
          <dt className="lap__key">loaded</dt>
          <dd className="lap__value">{replayLoaded ? 'yes' : 'no'}</dd>
        </div>
      </div>
      <label htmlFor="scrubber-range" style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
        Scrub time (microseconds)
      </label>
      <input
        id="scrubber-range"
        type="range"
        min={0}
        max={maxTimeUs}
        step={stepMs}
        value={cursorTimeUs}
        onChange={(e) => setCursorTimeUs(Number(e.target.value))}
        style={{ width: '100%' }}
        aria-label="Scrub through replay timeline"
      />
      <div className="lap__current">
        <span className="lap__current-label">fill at cursor</span>
        {Object.entries(fillAtCursor).map(([name, info]) => (
          <span key={name} className="lap__current-sector">
            <span className="lap__key">{name}</span>
            {info.latestValue === null ? (
              <span>—</span>
            ) : (
              <span>
                {formatValue(info.latestValue)} · {info.count} pts
              </span>
            )}
          </span>
        ))}
      </div>
      <div className="lap__current">
        <span className="lap__current-label">scrubbed value</span>
        {Object.entries(scrubbedValues).map(([name, val]) => (
          <span key={name} className="lap__current-sector">
            <span className="lap__key">{name}</span>
            {val === null ? <span>—</span> : <span>{formatValue(val)}</span>}
          </span>
        ))}
      </div>
    </section>
  );
}
