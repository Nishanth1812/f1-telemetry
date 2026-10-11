import type { ReactNode } from 'react';
import { useNow } from '../hooks/useNow';
import { useTelemetryStore } from '../store/telemetryStore';
import { TELEMETRY_WS_URL } from '../telemetry/config';

const STALE_AFTER_MS = 500;
const TICK_MS = 250;

/**
 * The one connection status on the page. The header used to show two competing
 * texts (a paragraph and this pill); this component is now the only one.
 */
function headline(status: string, stalled: boolean): ReactNode {
  if (status === 'open') {
    // A stalled socket is still connected, but no frames are arriving. This is
    // kept as its own wording so the page never claims "connected" while the
    // data has stopped.
    return stalled ? 'Connected, but no data arriving right now' : 'Connected to the simulator';
  }
  if (status === 'connecting' || status === 'reconnecting') {
    return `Connecting to ${TELEMETRY_WS_URL}…`;
  }
  return (
    <>
      Not connected — run <code>uv run f1-serve</code>
    </>
  );
}

export function ConnectionIndicator() {
  const status = useTelemetryStore((state) => state.status);
  const attempt = useTelemetryStore((state) => state.attempt);
  const retryAt = useTelemetryStore((state) => state.retryAt);
  const detail = useTelemetryStore((state) => state.detail);
  const framesPerSecond = useTelemetryStore((state) => state.framesPerSecond);
  const frameCount = useTelemetryStore((state) => state.frameCount);
  const malformedCount = useTelemetryStore((state) => state.malformedCount);
  const lastFrameAt = useTelemetryStore((state) => state.lastFrameAt);
  const now = useNow(status === 'open' || status === 'reconnecting' ? TICK_MS : null);

  const ageMs = lastFrameAt === null ? null : Math.max(0, now - lastFrameAt);
  const stalled = status === 'open' && ageMs !== null && ageMs > STALE_AFTER_MS;
  const countdown = retryAt === null ? null : Math.max(0, retryAt - now);

  const retryLine =
    status === 'reconnecting' && countdown !== null
      ? `Reconnecting in ${(countdown / 1_000).toFixed(1)} s (attempt ${attempt})${
          detail === null ? '' : `: ${detail}`
        }`
      : null;

  return (
    <div className={`connection connection--${stalled ? 'stalled' : status}`}>
      <p className="connection__state">
        <span className="connection__dot" aria-hidden="true" />
        <span className="connection__label" role="status" aria-live="polite">
          {headline(status, stalled)}
        </span>
      </p>
      <dl className="connection__stats">
        <div className="connection__stat">
          <dt>Updates per second</dt>
          <dd>{framesPerSecond === 0 ? '—' : framesPerSecond.toFixed(1)}</dd>
        </div>
        <div className="connection__stat">
          <dt>Updates received</dt>
          <dd>{frameCount}</dd>
        </div>
        <div className="connection__stat">
          <dt>Time since last update</dt>
          <dd>{ageMs === null ? '—' : `${(ageMs / 1_000).toFixed(2)} s`}</dd>
        </div>
        {malformedCount > 0 ? (
          <div className="connection__stat">
            <dt>Bad messages ignored</dt>
            <dd>{malformedCount}</dd>
          </div>
        ) : null}
      </dl>
      {retryLine !== null ? <p className="connection__detail">{retryLine}</p> : null}
      {retryLine === null && detail !== null && status !== 'reconnecting' ? <p className="connection__detail">{detail}</p> : null}
    </div>
  );
}
