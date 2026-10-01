import { useEffect } from 'react';
import { TELEMETRY_WS_URL, isWebSocketUrl } from '../telemetry/config';
import { TelemetrySocket } from '../telemetry/socket';
import { useTelemetryStore } from '../store/telemetryStore';

export function useTelemetrySocket(): void {
  useEffect(() => {
    const store = useTelemetryStore.getState();
    if (!isWebSocketUrl(TELEMETRY_WS_URL)) {
      store.markFailed(`VITE_TELEMETRY_WS_URL is not a ws:// or wss:// URL: ${TELEMETRY_WS_URL}`);
      return;
    }
    const socket = new TelemetrySocket(TELEMETRY_WS_URL, {
      onConnecting: () => useTelemetryStore.getState().markConnecting(),
      onOpen: () => useTelemetryStore.getState().markOpen(),
      onFrame: (frame) => useTelemetryStore.getState().applyFrame(frame),
      onMalformed: (count) => useTelemetryStore.getState().countMalformed(count),
      onRetry: (attempt, delayMs, reason) =>
        useTelemetryStore.getState().markRetry(attempt, delayMs, reason),
    });
    socket.start();
    return () => {
      socket.stop();
    };
  }, []);
}
