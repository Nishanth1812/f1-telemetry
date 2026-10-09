import type { EventLogEntry } from './events';

/**
 * Lap and sector reconstruction from the contract's session event stream
 * (PLAN.md sections 5.2, 8.5 and 8.4; PHASES.md P8-T8).
 *
 * `lap_index`, `sector_index`, `sector_time`, `lap_time` and `delta` are declared
 * `event: true` in `channels.yaml`, so they carry no periodic rate and reach the
 * browser as *changes* rather than as a stream. The store turns each change into
 * an `EventLogEntry` with the simulated timestamp of the frame it arrived on, and
 * that log is what this module folds. Every time and every delta here is a value
 * a producer published; none is derived, interpolated or estimated.
 *
 * Ordering. Both `lap_time` and `lap_index` are published at a lap boundary, so
 * the wire cannot be assumed to deliver one before the other. Closing on
 * `lap_time` and opening on an index *change* is the one rule that is correct
 * under either order: a lap is complete the moment its time arrives, and the
 * next index change names the lap after it.
 *
 * **Validity is deliberately not overclaimed.** `f1telemetry.laps.
 * evaluate_lap_validity` decides a lap on four inputs - a completed lap, all four
 * wheels within the track limits at every sample, no DNF, and no invalid flag.
 * Only the first is observable on this stream; track limits, DNF and the invalid
 * flag are properties of the P4 track projection and its caller, not of the
 * session channels. So `LapVerdict` reports the internal consistency the stream
 * *can* prove, and `consistent` means exactly that: three sector times
 * reconciling with the reported lap time, inside the contract's declared range.
 * It is not a certified valid lap, and a lap is never called valid here.
 */

export const SECTOR_COUNT = 3;
export const SECTOR_LABELS = ['S1', 'S2', 'S3'] as const;

/** Laps retained for the panel, newest first. */
export const LAPS_RETAINED = 8;

/**
 * Reconciliation tolerance between the sum of the three sector times and the
 * reported lap time: a tenth of a percent of the lap, floored at 10 ms so a short
 * lap is not held to an impossible precision. Both sides of the comparison are
 * reported, so a reader can apply a different tolerance to the same numbers.
 */
export const RECONCILE_RELATIVE = 0.001;
export const RECONCILE_FLOOR_S = 0.01;

/** Contract-declared bounds for the session channels, from `channels.yaml`. */
export const LAP_TIME_RANGE: readonly [number, number] = [0, 1200];
export const SECTOR_TIME_RANGE: readonly [number, number] = [0, 600];
export const DELTA_RANGE: readonly [number, number] = [-5, 5];

/** Channels this module consumes. Anything else in the log is not a lap event. */
const SESSION_CHANNELS = new Set(['lap_index', 'sector_index', 'sector_time', 'lap_time', 'delta']);

export type LapVerdict = 'consistent' | 'unreconciled' | 'unmeasured' | 'out-of-range';

export const VERDICT_TEXT: Record<LapVerdict, string> = {
  consistent: 'consistent',
  unreconciled: 'sectors do not reconcile',
  unmeasured: 'no lap time reported',
  'out-of-range': 'outside the declared range',
};

export interface LapRecord {
  /** `lap_index` when it was published for this lap, otherwise `null`. */
  readonly lapIndex: number | null;
  /** The three reported sector times, `null` where the stream published none. */
  readonly sectors: readonly (number | null)[];
  /** Reported `sector_index` values, in arrival order. */
  readonly sectorIndices: readonly number[];
  readonly lapTime: number | null;
  /** The last `delta` published before the lap closed. */
  readonly deltaAtClose: number | null;
  readonly closed: boolean;
  readonly verdict: LapVerdict;
  /** Sum of the reported sector times minus the reported lap time, in seconds. */
  readonly reconciliation: number | null;
}

export interface LapState {
  /** Closed laps, newest first. */
  readonly laps: readonly LapRecord[];
  readonly open: LapRecord;
  /** Most recent `delta` on the stream, live position included. */
  readonly latestDelta: number | null;
  /** Whether any delta has been published at all. */
  readonly deltaSeen: boolean;
  readonly lastEventTimeUs: number;
}

export function emptyLap(lapIndex: number | null): LapRecord {
  return {
    lapIndex,
    sectors: Array.from({ length: SECTOR_COUNT }, () => null),
    sectorIndices: [],
    lapTime: null,
    deltaAtClose: null,
    closed: false,
    verdict: 'unmeasured',
    reconciliation: null,
  };
}

export function emptyLapState(): LapState {
  return { laps: [], open: emptyLap(null), latestDelta: null, deltaSeen: false, lastEventTimeUs: 0 };
}

export function inDeclaredRange(value: number, range: readonly [number, number]): boolean {
  return Number.isFinite(value) && value >= range[0] && value <= range[1];
}

export function sectorTime(lap: LapRecord, index: number): number | null {
  return lap.sectors[index] ?? null;
}

/** Sum of the reported sector times, a missing sector counted as zero. */
export function sectorSum(lap: LapRecord): number {
  return lap.sectors.reduce<number>((total, time) => total + (time ?? 0), 0);
}

/**
 * Classify a lap from what the session stream can prove. Three checks and no
 * more: sector count, sector sum against the lap time, and the declared range.
 */
export function classifyLap(lap: LapRecord): LapRecord {
  if (lap.lapTime === null) {
    return { ...lap, verdict: 'unmeasured', reconciliation: null };
  }
  if (!inDeclaredRange(lap.lapTime, LAP_TIME_RANGE)) {
    return { ...lap, verdict: 'out-of-range', reconciliation: null };
  }
  const reported = lap.sectors.filter((time): time is number => time !== null);
  if (reported.length !== SECTOR_COUNT || reported.some((time) => !inDeclaredRange(time, SECTOR_TIME_RANGE))) {
    return { ...lap, verdict: 'unreconciled', reconciliation: null };
  }
  const reconciliation = sectorSum(lap) - lap.lapTime;
  const tolerance = Math.max(RECONCILE_RELATIVE * lap.lapTime, RECONCILE_FLOOR_S);
  const verdict: LapVerdict = Math.abs(reconciliation) <= tolerance ? 'consistent' : 'unreconciled';
  return { ...lap, verdict, reconciliation };
}

/** Fold one event into the lap state. See the module docstring on ordering. */
export function reduceLapState(state: LapState, entry: EventLogEntry): LapState {
  if (entry.kind !== 'channel' || !SESSION_CHANNELS.has(entry.channel)) {
    return state;
  }
  const { value, timeUs } = entry;
  if (!Number.isFinite(value)) {
    return state;
  }
  if (state.lastEventTimeUs > 0 && timeUs < state.lastEventTimeUs) {
    // A replay rewind restarts the session timeline; laps must not span two runs.
    // The rewinding entry itself is still part of the new timeline, so it is
    // folded onto the fresh state rather than discarded.
    return reduceLapState({ ...emptyLapState(), lastEventTimeUs: timeUs }, entry);
  }
  const next = { ...state, lastEventTimeUs: timeUs };

  switch (entry.channel) {
    case 'delta':
      return { ...next, latestDelta: value, deltaSeen: true };

    case 'lap_index': {
      const lapIndex = Math.trunc(value);
      if (state.open.lapIndex === lapIndex) {
        return next;
      }
      // An index change is a lap boundary: close what came before, open the next.
      const closable = state.open.lapTime !== null || state.open.sectorIndices.length > 0;
      const closed = closable ? classifyLap({ ...state.open, closed: true }) : null;
      return {
        ...next,
        laps: closed === null ? state.laps : [closed, ...state.laps].slice(0, LAPS_RETAINED),
        open: emptyLap(lapIndex),
      };
    }

    case 'sector_index':
      return {
        ...next,
        open: { ...state.open, sectorIndices: [...state.open.sectorIndices, Math.trunc(value)] },
      };

    case 'sector_time': {
      const slot = state.open.sectors.indexOf(null);
      if (slot < 0) {
        // A fourth sector time inside one lap: the split cannot be attributed to
        // a sector, so it is dropped rather than assigned to the wrong one.
        return next;
      }
      const sectors = state.open.sectors.slice();
      sectors[slot] = value;
      return { ...next, open: { ...state.open, sectors } };
    }

    case 'lap_time':
      return {
        ...next,
        laps: [
          classifyLap({ ...state.open, lapTime: value, deltaAtClose: state.latestDelta, closed: true }),
          ...state.laps,
        ].slice(0, LAPS_RETAINED),
        open: emptyLap(state.open.lapIndex),
      };

    default:
      return next;
  }
}

/** Fold a whole event log into a lap state, oldest first. */
export function lapStateFromEvents(events: readonly EventLogEntry[]): LapState {
  return events.reduce(reduceLapState, emptyLapState());
}

/**
 * The reference lap: the quickest closed lap that reconciled. An unreconciled
 * lap would make every delta against it meaningless, so it is never chosen.
 */
export function referenceLap(laps: readonly LapRecord[]): LapRecord | null {
  return laps.reduce<LapRecord | null>((quickest, lap) => {
    if (lap.verdict !== 'consistent' || lap.lapTime === null) {
      return quickest;
    }
    return quickest === null || quickest.lapTime === null || lap.lapTime < quickest.lapTime ? lap : quickest;
  }, null);
}

/**
 * Per-sector delta of `lap` against `reference`, in seconds. Real arithmetic on
 * two published sector times: a negative number means this lap was quicker
 * through that sector.
 */
export function sectorDeltas(lap: LapRecord, reference: LapRecord | null): (number | null)[] {
  if (reference === null) {
    return [null, null, null];
  }
  return Array.from({ length: SECTOR_COUNT }, (_unused, index) => {
    const mine = sectorTime(lap, index);
    const against = sectorTime(reference, index);
    return mine === null || against === null ? null : mine - against;
  });
}
