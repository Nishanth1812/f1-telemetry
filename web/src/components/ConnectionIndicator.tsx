import { useNow } from '../hooks/useNow';
import { useTelemetryStore } from '../store/telemetryStore';

const STALE_AFTER_MS = 500;
const TICK_MS = 250;

const LABELS = {
  idle: 'Idle',
  connecting: 'Connecting…',
  open: 'Live',
  reconnecting: 'Reconnecting',
  closed: 'Disconnected',
  error: 'Error',
} as const;

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

  let label: string = LABELS[status];
  if (status === 'reconnecting' && countdown !== null) {
    label = `Reconnecting in ${(countdown / 1_000).toFixed(1)} s (attempt ${attempt})`;
  } else if (stalled) {
    label = 'No data';
  } else if (status === 'open' && attempt > 0) {
    label = 'Reconnected';
  }

  return (
    <div className={`connection connection--${stalled ? 'stalled' : status}`}>
      <p className="connection__state">
        <span className="connection__dot" aria-hidden="true" />
        <span className="connection__label" role="status" aria-live="polite">
          {label}
        </span>
      </p>
      <dl className="connection__stats">
        <div className="connection__stat">
          <dt>Frame rate</dt>
          <dd>{framesPerSecond === 0 ? '—' : `${framesPerSecond.toFixed(1)} Hz`}</dd>
        </div>
        <div className="connection__stat">
          <dt>Frames</dt>
          <dd>{frameCount}</dd>
        </div>
        <div className="connection__stat">
          <dt>Last frame</dt>
          <dd>{ageMs === null ? '—' : `${(ageMs / 1_000).toFixed(2)} s`}</dd>
        </div>
        {malformedCount > 0 ? (
          <div className="connection__stat">
            <dt>Malformed</dt>
            <dd>{malformedCount}</dd>
          </div>
        ) : null}
      </dl>
      {detail !== null ? <p className="connection__detail">{detail}</p> : null}
    </div>
  );
}
