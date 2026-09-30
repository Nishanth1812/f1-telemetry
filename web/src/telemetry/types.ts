export const TILE_REFRESH_HZ = 30;
export const TILE_REFRESH_MS = 1000 / TILE_REFRESH_HZ;

export interface TelemetryFrame {
  time_us: number;
  channels: Record<string, number>;
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
  const candidate = payload as { time_us?: unknown; channels?: unknown };
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
  return { time_us: candidate.time_us, channels };
}
