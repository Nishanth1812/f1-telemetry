import { create } from 'zustand';
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
  markConnecting: () =>
    set({ status: 'connecting', attempt: 0, retryAt: null, detail: null }),
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
      const names = Object.keys(frame.channels);
      const stale = state.frame === null || now - lastTileRefreshAt >= TILE_REFRESH_MS;
      if (!stale) {
        return { frameCount };
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
      return { frame, frameCount, channelNames, framesPerSecond, lastFrameAt: now };
    });
  },
  countMalformed: (count) => set((state) => ({ malformedCount: state.malformedCount + count })),
}));
