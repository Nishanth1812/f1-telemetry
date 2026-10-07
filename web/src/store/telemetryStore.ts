import { create } from 'zustand';
import {
  EVENT_CHANNEL_NAMES,
  appendEventEntries,
  channelEventEntry,
  wireEventToEntry,
  type EventLogEntry,
} from '../telemetry/events';
import { TraceHistory } from '../telemetry/history';
import { CHANNELS, type ChannelSpec } from '../generated/channels';
import {
  TILE_REFRESH_MS,
  type ConnectionStatus,
  type TelemetryFrame,
} from '../telemetry/types';

export interface ChannelDiagnostics {
  /** Declared periodic rate from the generated channel contract, Hz. */
  declaredHz: number | null;
  /**
   * Observed rate in samples per second, computed from the actual ingested
   * sample timestamps (first to last), not from wall-clock estimates. 0 when a
   * channel has fewer than two samples or a degenerate time span.
   */
  observedHz: number;
  /** Total ingested samples for this channel since the last clear. */
  count: number;
}

export interface TelemetryDiagnostics {
  totalCount: number;
  channels: Record<string, ChannelDiagnostics>;
}

export interface TelemetryState {
  status: ConnectionStatus;
  attempt: number;
  retryAt: number | null;
  detail: string | null;
  channelNames: string[];
  frame: TelemetryFrame | null;
  frameCount: number;
  malformedCount: number;
  framesPerSecond: number;
  lastFrameAt: number | null;
  history: TraceHistory;
  events: EventLogEntry[];
  diagnostics: TelemetryDiagnostics | null;
  markConnecting: () => void;
  markOpen: () => void;
  markRetry: (attempt: number, delayMs: number, reason: string | null) => void;
  markFailed: (detail: string) => void;
  applyFrame: (frame: TelemetryFrame) => void;
  countMalformed: (count: number) => void;
}

const DIAGNOSTICS_INTERVAL_MS = 250;

let lastArrivalAt = 0;
let arrivalEmaMs = 0;
let lastTileRefreshAt = 0;
let nextEventId = 1;
let lastEventTimeUs: number | null = null;
const lastEventChannelValues = new Map<string, number>();

// Raw ingestion counters live outside React state; a throttled snapshot is
// what selectors see.
let lastSampleFrameTimeUs: number | null = null;
let totalSampleCount = 0;
const channelCounters = new Map<string, { count: number; firstUs: number; lastUs: number }>();
let lastDiagnosticsAt = 0;

function resetRawCounters(): void {
  lastSampleFrameTimeUs = null;
  totalSampleCount = 0;
  channelCounters.clear();
  lastDiagnosticsAt = 0;
}

function ingestCounters(timeUs: number, channels: Record<string, number>): void {
  for (const name of Object.keys(channels)) {
    const entry = channelCounters.get(name);
    if (entry === undefined) {
      channelCounters.set(name, { count: 1, firstUs: timeUs, lastUs: timeUs });
    } else {
      entry.count += 1;
      entry.firstUs = Math.min(entry.firstUs, timeUs);
      entry.lastUs = Math.max(entry.lastUs, timeUs);
    }
    totalSampleCount += 1;
  }
}

function declaredRateHz(name: string): number | null {
  const spec = (CHANNELS as Record<string, ChannelSpec | undefined>)[name];
  return spec !== undefined && typeof spec.rateHz === 'number' && spec.rateHz > 0
    ? spec.rateHz
    : null;
}

function buildDiagnostics(): TelemetryDiagnostics {
  const channels: Record<string, ChannelDiagnostics> = {};
  for (const [name, entry] of channelCounters) {
    const spanS = (entry.lastUs - entry.firstUs) / 1e6;
    channels[name] = {
      declaredHz: declaredRateHz(name),
      observedHz: entry.count >= 2 && spanS > 0 ? (entry.count - 1) / spanS : 0,
      count: entry.count,
    };
  }
  return { totalCount: totalSampleCount, channels };
}

function collectEventEntries(frame: TelemetryFrame): EventLogEntry[] {
  if (lastEventTimeUs !== null && frame.time_us < lastEventTimeUs) {
    // A replay rewind starts a new event baseline, even when the socket stays open.
    lastEventChannelValues.clear();
  }
  lastEventTimeUs = frame.time_us;
  const entries: EventLogEntry[] = [];
  if (frame.events !== undefined) {
    for (const event of frame.events) {
      entries.push(wireEventToEntry(nextEventId, event));
      nextEventId += 1;
    }
  }
  for (const [name, value] of Object.entries(frame.channels)) {
    if (!EVENT_CHANNEL_NAMES.has(name)) {
      continue;
    }
    const previous = lastEventChannelValues.get(name);
    lastEventChannelValues.set(name, value);
    if (previous === undefined || previous === value) {
      continue;
    }
    entries.push(channelEventEntry(nextEventId, name, value, frame.time_us));
    nextEventId += 1;
  }
  return entries;
}

function measuredRate(now: number): number {
  if (lastArrivalAt === 0) {
    lastArrivalAt = now;
    return 0;
  }
  const delta = now - lastArrivalAt;
  lastArrivalAt = now;
  if (delta <= 0 || delta > 2_000) {
    return 0;
  }
  arrivalEmaMs = arrivalEmaMs === 0 ? delta : arrivalEmaMs + (delta - arrivalEmaMs) * 0.1;
  return Math.min(1_000, 1_000 / arrivalEmaMs);
}

export const useTelemetryStore = create<TelemetryState>((set) => ({
  status: 'idle',
  attempt: 0,
  retryAt: null,
  detail: null,
  channelNames: [],
  frame: null,
  frameCount: 0,
  malformedCount: 0,
  framesPerSecond: 0,
  lastFrameAt: null,
  history: new TraceHistory(),
  events: [],
  diagnostics: null,
  markConnecting: () => {
    // Reconnecting restarts everything: the first values after reconnect are a
    // baseline, not a continuation of the previous timeline.
    set((state) => {
      state.history.clear();
      resetRawCounters();
      lastEventChannelValues.clear();
      lastEventTimeUs = null;
      return { status: 'connecting', attempt: 0, retryAt: null, detail: null, diagnostics: null };
    });
  },
  markOpen: () => set({ status: 'open', attempt: 0, retryAt: null, detail: null }),
  markRetry: (attempt, delayMs, reason) =>
    set({
      status: 'reconnecting',
      attempt,
      retryAt: Date.now() + delayMs,
      detail: reason,
    }),
  markFailed: (detail) => set({ status: 'error', attempt: 0, retryAt: null, detail }),
  applyFrame: (frame) => {
    const now = Date.now();
    const framesPerSecond = measuredRate(now);
    set((state) => {
      if (lastSampleFrameTimeUs !== null && frame.time_us < lastSampleFrameTimeUs) {
        // Replay rewind: drop the stale timeline and counters.
        state.history.clear();
        resetRawCounters();
      }
      // Ingest the actual periodic sample batch when present; the frame
      // snapshot itself is the legacy fallback only. Never both, or the
      // snapshot would be counted twice.
      const units = frame.samples !== undefined ? frame.samples : [frame];
      for (const unit of units) {
        state.history.push(unit);
        ingestCounters(unit.time_us, unit.channels);
      }
      lastSampleFrameTimeUs = frame.time_us;
      const frameCount = state.frameCount + 1;
      // Events are collected on every frame, before the tile-refresh throttle:
      // a dropped occurrence is a lost fault annotation or lap boundary, not a
      // stale tile value.
      const entries = collectEventEntries(frame);
      const events = entries.length > 0 ? appendEventEntries(state.events, entries) : state.events;
      const names = Object.keys(frame.channels);
      const stale = state.frame === null || now - lastTileRefreshAt >= TILE_REFRESH_MS;
      const diagnostics =
        lastDiagnosticsAt === 0 || now - lastDiagnosticsAt >= DIAGNOSTICS_INTERVAL_MS
          ? buildDiagnostics()
          : state.diagnostics;
      if (diagnostics !== state.diagnostics) {
        lastDiagnosticsAt = now;
      }
      if (!stale) {
        return events === state.events && diagnostics === state.diagnostics
          ? { frameCount }
          : { frameCount, events, diagnostics };
      }
      lastTileRefreshAt = now;
      let channelNames = state.channelNames;
      if (names.length !== channelNames.length || names[0] !== channelNames[0]) {
        const known = new Set(channelNames);
        const added = names.filter((name) => !known.has(name));
        if (added.length > 0) {
          channelNames = [...channelNames, ...added];
        }
      }
      // Keep the 30 Hz snapshot in render state; the full-rate batch already
      // lives in history and raw counters, so do not retain it in Zustand.
      const snapshot: TelemetryFrame = { time_us: frame.time_us, channels: frame.channels };
      if (frame.events !== undefined) {
        snapshot.events = frame.events;
      }
      return {
        frame: snapshot,
        frameCount,
        channelNames,
        framesPerSecond,
        lastFrameAt: now,
        events,
        diagnostics,
      };
    });
  },
  countMalformed: (count) => set((state) => ({ malformedCount: state.malformedCount + count })),
}));
