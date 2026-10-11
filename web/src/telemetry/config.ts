export const DEFAULT_WS_URL = 'ws://localhost:8765/ws';

function readEnvUrl(): string {
  const configured = import.meta.env.VITE_TELEMETRY_WS_URL?.trim();
  return configured === undefined || configured === '' ? DEFAULT_WS_URL : configured;
}

export const TELEMETRY_WS_URL = readEnvUrl();

export function isWebSocketUrl(value: string): boolean {
  try {
    const protocol = new URL(value).protocol;
    return protocol === 'ws:' || protocol === 'wss:';
  } catch {
    return false;
  }
}
