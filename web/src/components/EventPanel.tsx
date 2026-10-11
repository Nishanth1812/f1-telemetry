import { useTelemetryStore } from '../store/telemetryStore';
import type { EventLogEntry } from '../telemetry/events';
import { formatValue } from '../telemetry/format';
import { channelLabel, faultLabel } from './Tile';

function formatSimTime(timeUs: number): string {
  return `${(timeUs / 1e6).toFixed(3)} s`;
}

function describe(entry: EventLogEntry): string {
  if (entry.kind === 'channel') {
    return `${channelLabel(entry.channel)} = ${formatValue(entry.value)}`;
  }
  const parts = [faultLabel(entry.faultType)];
  if (entry.channel !== null) {
    parts.push(`on ${channelLabel(entry.channel)}`);
  }
  if (entry.severity !== null) {
    parts.push(`severity ${entry.severity.toFixed(2)}`);
  }
  if (entry.durationSamples !== null) {
    parts.push(`${entry.durationSamples} samples`);
  }
  if (entry.label === false) {
    parts.push('not confirmed as injected');
  }
  return parts.join(' · ');
}

/** The raw identifiers behind an event, kept in a tooltip for engineers. */
function rawName(entry: EventLogEntry): string {
  if (entry.kind === 'channel') {
    return entry.channel;
  }
  return entry.channel === null ? entry.faultType : `${entry.faultType} on ${entry.channel}`;
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
          Nothing has happened yet. Injected faults and lap or sector changes will be listed here
          as they occur.
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
              <span className="events__text" title={rawName(entry)}>
                {describe(entry)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
