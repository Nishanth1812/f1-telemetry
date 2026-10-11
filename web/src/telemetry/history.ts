import { CHANNELS, type ChannelSpec } from '../generated/channels';
import type { TelemetrySample } from './types';

const MAX_TRACKED_CHANNELS = 128;
const RETENTION_SECONDS = 60;
const FALLBACK_RATE_HZ = 30;

// Per-channel capacity: 60 s at the channel's generated rate, plus one
// endpoint. An explicit constructor capacity overrides the derivation.
function derivedCapacity(name: string): number {
  const spec = (CHANNELS as Record<string, ChannelSpec | undefined>)[name];
  const rate = spec?.rateHz;
  if (typeof rate === 'number' && Number.isFinite(rate) && rate > 0) {
    return Math.ceil(rate * RETENTION_SECONDS) + 1;
  }
  return Math.ceil(FALLBACK_RATE_HZ * RETENTION_SECONDS) + 1;
}

interface Ring {
  readonly capacity: number;
  readonly times: Float64Array;
  readonly values: Float32Array;
  head: number;
  size: number;
}

export class TraceHistory {
  private readonly explicitCapacity: number | null;
  private readonly rings = new Map<string, Ring>();
  private lastTimeUs: number | null = null;

  constructor(capacity?: number) {
    this.explicitCapacity = capacity === undefined ? null : Math.max(2, Math.floor(capacity));
  }

  get length(): number {
    let total = 0;
    for (const ring of this.rings.values()) {
      total += ring.size;
    }
    return total;
  }

  get trackedChannels(): number {
    return this.rings.size;
  }

  has(channel: string): boolean {
    return this.rings.has(channel);
  }

  push(sample: TelemetrySample): void {
    const entries = Object.entries(sample.channels);
    if (entries.length === 0) {
      return;
    }
    if (this.lastTimeUs !== null && sample.time_us < this.lastTimeUs) {
      // A replay rewind starts a new timeline; the old points are stale.
      this.clear();
    }
    this.lastTimeUs = sample.time_us;
    for (const [name, value] of entries) {
      let ring = this.rings.get(name);
      if (ring === undefined) {
        if (this.rings.size >= MAX_TRACKED_CHANNELS) {
          continue;
        }
        const capacity = this.explicitCapacity ?? derivedCapacity(name);
        ring = {
          capacity,
          times: new Float64Array(capacity),
          values: new Float32Array(capacity),
          head: 0,
          size: 0,
        };
        this.rings.set(name, ring);
      }
      ring.times[ring.head] = sample.time_us;
      ring.values[ring.head] = value;
      ring.head = (ring.head + 1) % ring.capacity;
      if (ring.size < ring.capacity) {
        ring.size += 1;
      }
    }
  }

  clear(): void {
    for (const ring of this.rings.values()) {
      ring.head = 0;
      ring.size = 0;
    }
    this.lastTimeUs = null;
  }

  fill(
    channel: string,
    maxPoints: number,
    times: Float64Array,
    values: Float32Array,
    sinceTimeUs?: number,
  ): number {
    const ring = this.rings.get(channel);
    if (ring === undefined || ring.size === 0) {
      return 0;
    }
    const limit = Math.max(2, Math.min(maxPoints, times.length, values.length));
    const start = (ring.head - ring.size + ring.capacity) % ring.capacity;

    // Per-channel timestamps are chronological (a rewind clears the ring), so
    // the window cut is a first visible offset, not a scan allocation.
    let firstOffset = 0;
    if (sinceTimeUs !== undefined) {
      while (firstOffset < ring.size) {
        const index = (start + firstOffset) % ring.capacity;
        if ((ring.times[index] ?? 0) >= sinceTimeUs) {
          break;
        }
        firstOffset += 1;
      }
    }
    const visibleCount = ring.size - firstOffset;
    if (visibleCount === 0) {
      return 0;
    }
    if (visibleCount <= limit) {
      let count = 0;
      for (let offset = firstOffset; offset < ring.size; offset += 1) {
        const index = (start + offset) % ring.capacity;
        times[count] = ring.times[index] ?? 0;
        values[count] = ring.values[index] ?? Number.NaN;
        count += 1;
      }
      return count;
    }
    const buckets = Math.max(1, Math.floor(limit / 2));
    const perBucket = visibleCount / buckets;
    let count = 0;
    for (let bucket = 0; bucket < buckets && count < limit; bucket += 1) {
      const from = firstOffset + Math.floor(bucket * perBucket);
      const to = Math.min(ring.size, firstOffset + Math.floor((bucket + 1) * perBucket));
      let minOffset = -1;
      let maxOffset = -1;
      let min = Number.POSITIVE_INFINITY;
      let max = Number.NEGATIVE_INFINITY;
      for (let offset = from; offset < to; offset += 1) {
        const index = (start + offset) % ring.capacity;
        const value = ring.values[index] ?? Number.NaN;
        if (Number.isNaN(value)) {
          continue;
        }
        if (value < min) {
          min = value;
          minOffset = offset;
        }
        if (value > max) {
          max = value;
          maxOffset = offset;
        }
      }
      if (minOffset < 0 || maxOffset < 0) {
        continue;
      }
      // Chronological order: the earlier extremum in time comes first, not the
      // one at the smaller physical ring slot.
      const first = Math.min(minOffset, maxOffset);
      const second = Math.max(minOffset, maxOffset);
      const firstIndex = (start + first) % ring.capacity;
      if (count < limit) {
        times[count] = ring.times[firstIndex] ?? 0;
        values[count] = ring.values[firstIndex] ?? Number.NaN;
        count += 1;
      }
      if (second !== first) {
        const secondIndex = (start + second) % ring.capacity;
        if (count < limit) {
          times[count] = ring.times[secondIndex] ?? 0;
          values[count] = ring.values[secondIndex] ?? Number.NaN;
          count += 1;
        }
      }
    }
    return count;
  }

  latestTimeUs(): number {
    if (this.lastTimeUs === null) {
      return 0;
    }
    return this.lastTimeUs;
  }
}
