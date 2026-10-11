import { useEffect } from 'react';
import { TELEMETRY_WS_URL, isWebSocketUrl } from '../telemetry/config';
import { TelemetrySocket } from '../telemetry/socket';
import { useTelemetryStore } from '../store/telemetryStore';
import type { TelemetryFrame } from '../telemetry/types';

export function useTelemetrySocket(): void {
  useEffect(() => {
    const store = useTelemetryStore.getState();
    if (!isWebSocketUrl(TELEMETRY_WS_URL)) {
      store.markFailed(`VITE_TELEMETRY_WS_URL is not a ws:// or wss:// URL: ${TELEMETRY_WS_URL}`);
      return;
    }
    const framesQueue: TelemetryFrame[] = [];
    let frameId: number | null = null;

    const socket = new TelemetrySocket(TELEMETRY_WS_URL, {
      onConnecting: () => useTelemetryStore.getState().markConnecting(),
      onOpen: () => useTelemetryStore.getState().markOpen(),
      onFrame: (frame) => {
        framesQueue.push(frame);
        if (frameId === null) {
          frameId = requestAnimationFrame(() => {
            frameId = null;
            const frames = [...framesQueue];
            framesQueue.length = 0;
            for (const frame of frames) {
              store.applyFrame(frame);
            }
          });
        }
      },
      onMalformed: (count) => store.countMalformed(count),
      onRetry: (attempt, delayMs, reason) =>
        store.markRetry(attempt, delayMs, reason),
    });
    socket.start();
    return () => {
      socket.stop();
      if (frameId !== null) {
        cancelAnimationFrame(frameId);
        frameId = null;
      }
      // Flush any remaining frames on unmount to avoid losing data?
      // But we are stopping the socket, so it's okay to drop them.
      framesQueue.length = 0;
    };
  }, []);
}
