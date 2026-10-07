import type { TelemetryFrame } from './types';

const DEFAULT_CAPACITY = 1800;
const MAX_TRACKED_CHANNELS = 128;

export class TraceHistory {
  readonly capacity: number;
  private readonly times: Float64Array;
  private readonly series = new Map<string, Float32Array>();
  private head = 0;
  private size = 0;

  constructor(capacity: number = DEFAULT_CAPACITY) {
    this.capacity = Math.max(2, Math.floor(capacity));
    this.times = new Float64Array(this.capacity);
  }

  get length(): number {
    return this.size;
  }

  get trackedChannels(): number {
    return this.series.size;
  }

  has(channel: string): boolean {
    return this.series.has(channel);
  }

  push(frame: TelemetryFrame): void {
    if (Object.keys(frame.channels).length === 0) {
      return;
    }
    if (this.series.size < MAX_TRACKED_CHANNELS) {
      for (const name of Object.keys(frame.channels)) {
        if (!this.series.has(name)) {
          this.series.set(name, new Float32Array(this.capacity));
        }
      }
    }
    const index = this.head;
    this.times[index] = frame.time_us;
    for (const [name, buffer] of this.series) {
      const value = frame.channels[name];
      buffer[index] = value === undefined ? Number.NaN : value;
    }
    this.head = (this.head + 1) % this.capacity;
    if (this.size < this.capacity) {
      this.size += 1;
    }
  }

  fill(channel: string, maxPoints: number, times: Float64Array, values: Float32Array): number {
    const buffer = this.series.get(channel);
    if (buffer === undefined || this.size === 0) {
      return 0;
    }
    const limit = Math.max(2, Math.min(maxPoints, times.length, values.length));
    const start = (this.head - this.size + this.capacity) % this.capacity;
    let count = 0;
    if (this.size <= limit) {
      for (let offset = 0; offset < this.size; offset += 1) {
        const index = (start + offset) % this.capacity;
        times[count] = this.times[index] ?? 0;
        values[count] = buffer[index] ?? Number.NaN;
        count += 1;
      }
      return count;
    }
    const buckets = Math.max(1, Math.floor(limit / 2));
    const perBucket = this.size / buckets;
    for (let bucket = 0; bucket < buckets; bucket += 1) {
      const from = Math.floor(bucket * perBucket);
      const to = Math.min(this.size, Math.floor((bucket + 1) * perBucket));
      let minIndex = -1;
      let maxIndex = -1;
      let min = Number.POSITIVE_INFINITY;
      let max = Number.NEGATIVE_INFINITY;
      for (let offset = from; offset < to; offset += 1) {
        const index = (start + offset) % this.capacity;
        const value = buffer[index] ?? Number.NaN;
        if (Number.isNaN(value)) {
          continue;
        }
        if (value < min) {
          min = value;
          minIndex = index;
        }
        if (value > max) {
          max = value;
          maxIndex = index;
        }
      }
      if (minIndex < 0) {
        continue;
      }
      const first = Math.min(minIndex, maxIndex);
      const second = Math.max(minIndex, maxIndex);
      times[count] = this.times[first] ?? 0;
      values[count] = buffer[first] ?? Number.NaN;
      count += 1;
      if (second !== first) {
        times[count] = this.times[second] ?? 0;
        values[count] = buffer[second] ?? Number.NaN;
        count += 1;
      }
    }
    return count;
  }

  latestTimeUs(): number {
    if (this.size === 0) {
      return 0;
    }
    const index = (this.head - 1 + this.capacity) % this.capacity;
    return this.times[index] ?? 0;
  }
}
