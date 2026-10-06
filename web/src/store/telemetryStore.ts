import { create } from 'zustand';
import {
  EVENT_CHANNEL_NAMES,
  appendEventEntries,
  channelEventEntry,
  wireEventToEntry,
  type EventLogEntry,
} from '../telemetry/events';
import { TraceHistory } from '../telemetry/history';
import { TILE_REFRESH_MS, type ConnectionStatus, type TelemetryFrame } from '../telemetry/types';

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
  markConnecting: () => void;
  markOpen: () => void;
  markRetry: (attempt: number, delayMs: number, reason: string | null) => void;
  markFailed: (detail: string) => void;
  applyFrame: (frame: TelemetryFrame) => void;
  countMalformed: (count: number) => void;
}

let lastArrivalAt = 0;
let arrivalEmaMs = 0;
let lastTileRefreshAt = 0;
let nextEventId = 1;
let lastEventTimeUs: number | null = null;
const lastEventChannelValues = new Map<string, number>();

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
  markConnecting: () => {
    // The first value after reconnect is a baseline, not a new event.
    lastEventChannelValues.clear();
    lastEventTimeUs = null;
    set({ status: 'connecting', attempt: 0, retryAt: null, detail: null });
  },
  markOpen: () =>
    set({ status: 'open', attempt: 0, retryAt: null, detail: null }),
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
      state.history.push(frame);
      const frameCount = state.frameCount + 1;
      // Events are collected on every frame, before the tile-refresh throttle:
      // a dropped occurrence is a lost fault annotation or lap boundary, not a
      // stale tile value.
      const entries = collectEventEntries(frame);
      const events = entries.length > 0 ? appendEventEntries(state.events, entries) : state.events;
      const names = Object.keys(frame.channels);
      const stale = state.frame === null || now - lastTileRefreshAt >= TILE_REFRESH_MS;
      if (!stale) {
        return events === state.events ? { frameCount } : { frameCount, events };
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
      return { frame, frameCount, channelNames, framesPerSecond, lastFrameAt: now, events };
    });
  },
  countMalformed: (count) => set((state) => ({ malformedCount: state.malformedCount + count })),
}));
