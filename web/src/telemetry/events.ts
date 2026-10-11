import { CHANNELS } from '../generated/channels';
import type { FrameEvent } from './types';

/**
 * Largest number of run events retained in the log. Events are sparse by nature -
 * fault injections and lap/sector boundaries - so a small bound suffices; the
 * oldest entries drop off first.
 */
export const MAX_EVENT_LOG_ENTRIES = 100;

/**
 * Channels the contract declares event-driven (`event: true`, `rateHz: null`):
 * today `lap_index`, `sector_index`, `lap_time`, `sector_time` and `delta`.
 * Derived from the generated contract metadata rather than hand-listed, so a
 * contract edit is the only change needed for a new event channel to join the log.
 */
export const EVENT_CHANNEL_NAMES: ReadonlySet<string> = new Set(
  Object.values(CHANNELS)
    .filter((spec) => spec.event)
    .map((spec) => spec.name as string),
);

/**
 * One row in the dashboard's event log. Two sources feed it:
 *
 * - `fault`: a wire event from the frame's optional `events` key - a saved fault
 *   annotation replayed from a run's Parquet metadata.
 * - `channel`: a value change on a contract event channel. A replayed run carries
 *   these as ordinary columns rather than publish-on-occurrence messages, so a
 *   *change* is the occurrence; the first sighting is the run's initial state.
 */
export type EventLogEntry =
  | {
      id: number;
      timeUs: number;
      kind: 'fault';
      faultType: string;
      channel: string | null;
      severity: number | null;
      durationSamples: number | null;
      label: boolean | null;
    }
  | {
      id: number;
      timeUs: number;
      kind: 'channel';
      channel: string;
      value: number;
    };

export function wireEventToEntry(id: number, event: FrameEvent): EventLogEntry {
  return {
    id,
    timeUs: event.time_us,
    kind: 'fault',
    faultType: event.fault_type ?? event.kind,
    channel: event.channel ?? null,
    severity: event.severity ?? null,
    durationSamples: event.duration_samples ?? null,
    label: event.label ?? null,
  };
}

export function channelEventEntry(
  id: number,
  channel: string,
  value: number,
  timeUs: number,
): EventLogEntry {
  return { id, timeUs, kind: 'channel', channel, value };
}

export function appendEventEntries(
  current: readonly EventLogEntry[],
  added: readonly EventLogEntry[],
  capacity: number = MAX_EVENT_LOG_ENTRIES,
): EventLogEntry[] {
  const merged = [...current, ...added];
  return merged.length > capacity ? merged.slice(merged.length - capacity) : merged;
}
