export const TILE_REFRESH_HZ = 30;
export const TILE_REFRESH_MS = 1000 / TILE_REFRESH_HZ;

export interface FrameEvent {
  kind: string;
  time_us: number;
  fault_type?: string;
  channel?: string;
  severity?: number;
  duration_samples?: number;
  label?: boolean;
}

export interface TelemetryFrame {
  time_us: number;
  channels: Record<string, number>;
  events?: FrameEvent[];
}

export type ConnectionStatus =
  | 'idle'
  | 'connecting'
  | 'open'
  | 'reconnecting'
  | 'closed'
  | 'error';

export function decodeFrame(payload: unknown): TelemetryFrame | null {
  if (typeof payload !== 'object' || payload === null) {
    return null;
  }
  const candidate = payload as { time_us?: unknown; channels?: unknown; events?: unknown };
  if (typeof candidate.time_us !== 'number' || !Number.isFinite(candidate.time_us)) {
    return null;
  }
  if (typeof candidate.channels !== 'object' || candidate.channels === null) {
    return null;
  }
  const channels: Record<string, number> = {};
  for (const [name, value] of Object.entries(candidate.channels as Record<string, unknown>)) {
    if (typeof value === 'number' && Number.isFinite(value)) {
      channels[name] = value;
    }
  }
  const frame: TelemetryFrame = { time_us: candidate.time_us, channels };
  const events = decodeEvents(candidate.events);
  if (events === null) {
    return null;
  }
  if (events !== undefined) {
    frame.events = events;
  }
  return frame;
}

function decodeEvents(payload: unknown): FrameEvent[] | null | undefined {
  if (payload === undefined) {
    return undefined;
  }
  if (!Array.isArray(payload)) {
    return null;
  }
  const events: FrameEvent[] = [];
  for (const entry of payload as unknown[]) {
    const event = decodeEvent(entry);
    if (event !== null) {
      events.push(event);
    }
  }
  return events;
}

function decodeEvent(payload: unknown): FrameEvent | null {
  if (typeof payload !== 'object' || payload === null) {
    return null;
  }
  const candidate = payload as Record<string, unknown>;
  if (typeof candidate.kind !== 'string' || candidate.kind === '') {
    return null;
  }
  if (typeof candidate.time_us !== 'number' || !Number.isFinite(candidate.time_us)) {
    return null;
  }
  const event: FrameEvent = { kind: candidate.kind, time_us: candidate.time_us };
  if (typeof candidate.fault_type === 'string') {
    event.fault_type = candidate.fault_type;
  }
  if (typeof candidate.channel === 'string') {
    event.channel = candidate.channel;
  }
  if (typeof candidate.severity === 'number' && Number.isFinite(candidate.severity)) {
    event.severity = candidate.severity;
  }
  if (typeof candidate.duration_samples === 'number' && Number.isFinite(candidate.duration_samples)) {
    event.duration_samples = candidate.duration_samples;
  }
  if (typeof candidate.label === 'boolean') {
    event.label = candidate.label;
  }
  return event;
}
