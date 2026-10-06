import { useTelemetryStore } from '../store/telemetryStore';
import type { EventLogEntry } from '../telemetry/events';
import { formatValue } from '../telemetry/format';

function formatSimTime(timeUs: number): string {
  return `${(timeUs / 1e6).toFixed(3)} s`;
}

function describe(entry: EventLogEntry): string {
  if (entry.kind === 'channel') {
    return `${entry.channel} = ${formatValue(entry.value)}`;
  }
  const parts = [`${entry.faultType} fault`];
  if (entry.channel !== null) {
    parts.push(`on ${entry.channel}`);
  }
  if (entry.severity !== null) {
    parts.push(`severity ${entry.severity.toFixed(2)}`);
  }
  if (entry.durationSamples !== null) {
    parts.push(`${entry.durationSamples} samples`);
  }
  if (entry.label === false) {
    parts.push('not ground truth');
  }
  return parts.join(' · ');
}

export function EventPanel() {
  const events = useTelemetryStore((state) => state.events);

  return (
    <section className="events" aria-labelledby="events-heading">
      <div className="events__header">
        <h2 className="events__heading" id="events-heading">
          Events
        </h2>
      </div>
      {events.length === 0 ? (
        <p className="placeholder" role="note">
          No events yet. Fault annotations and lap/sector boundaries appear here as they occur.
        </p>
      ) : (
        <ul
          className="events__list"
          role="log"
          aria-live="polite"
          aria-label="Run events, newest first"
        >
          {[...events].reverse().map((entry) => (
            <li key={entry.id} className={`events__item events__item--${entry.kind}`}>
              <span className="events__time">{formatSimTime(entry.timeUs)}</span>
              <span className="events__text">{describe(entry)}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
