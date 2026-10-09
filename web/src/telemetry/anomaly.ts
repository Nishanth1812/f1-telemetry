import { CHANNELS, CHANNEL_NAMES, type ChannelSpec } from '../generated/channels';
import type { EventLogEntry } from './events';

/**
 * Live Layer 2 multivariate SPC and Layer 3 persistence, in the browser
 * (PLAN.md section 9 layers 2, 3 and 5; PHASES.md P8-T7).
 *
 * The arithmetic mirrors `src/f1telemetry/analytics/multivariate.py`, so the
 * panel reports the same kind of number the offline sweep does and no other
 * kind:
 *
 * - z_j = (y_j - mu_j) / sigma_j, with `sigma_j` the channel's noise floor from
 *   the generated contract and `mu_j` the mean over the calibration windows.
 * - Rank-1 PCA of the calibration windows, by power iteration on their
 *   standardised covariance. `lambda_1` is the kept eigenvalue.
 * - T² (Hotelling, in-manifold) = (v·z)² / lambda_1. Q/SPE (out-of-manifold) =
 *   ||z||² - (v·z)², clipped at zero.
 * - Both thresholds are the empirical `level` quantile of the *calibration*
 *   windows, not theory values, so the calibration span scores silent by
 *   construction.
 * - Layer 3 persistence: an alarm is raised only when the statistic exceeded
 *   threshold in at least `k` of the last `n` windows, and released when that
 *   count falls below `k`.
 *
 * Every channel value it reads is a real ingested sample, read through the
 * store's ring buffer; nothing here fabricates a value, a confidence or a
 * probability. Where a number cannot be computed the panel shows a dash:
 * detection latency needs a ground-truth fault event from the event stream, and
 * it is quantised to the statistic window.
 *
 * **What this is not.** PLAN.md section 9.1 calibrates on clean runs only. A
 * live browser cannot know where the clean part of a run starts, so the first
 * `CALIBRATION_WINDOWS` windows of this connection are used instead and the
 * `contamination` flag is raised if a fault event falls inside them. The
 * published detection table (latency, FPR/h, miss rate) comes from the offline
 * sweep; this panel is the live view of the same statistic.
 *
 * Only channels the contract gives a noise scale (`sigma > 0`) *and* a declared
 * rate of at least 100 Hz take part: a 100 ms window samples those on every
 * window, while a 20 Hz or 10 Hz channel would be missing from half of them,
 * and Layer 2 skips any window with a missing input. Those slower thermal
 * channels stay visible in the trace and in the offline analytics. The
 * selection is derived from the contract, so no channel name is spelled here.
 */

/** Statistic window in simulated microseconds. 100 ms gives a 10 Hz statistic. */
export const WINDOW_US = 100_000;

/** Layer 3 persistence window: alarms need `K` of the last `N` exceedances. */
export const PERSISTENCE_N = 20;
export const PERSISTENCE_K = 12;

/**
 * Empirical quantile level for both thresholds, the same 99.7% the offline fit
 * uses (`analytics.multivariate.DEFAULT_LEVEL`). Calibrated from data, not
 * theory.
 */
export const THRESHOLD_LEVEL = 0.997;

/**
 * Windows accumulated before the manifold and the thresholds are frozen. At
 * 10 Hz this is 24 s of connection time.
 */
export const CALIBRATION_WINDOWS = 240;

/**
 * How far the oldest unscored window may lag the newest ingested sample, and the
 * per-tick cap on windows scored. The lag bound is what makes a window read
 * exact: `TraceHistory.fill` returns everything from the window's start to the
 * newest sample it holds, and deciminates once that exceeds the read buffer. With
 * the window grid never more than `MAX_BACKLOG_WINDOWS` behind, the read can be
 * sized from the contract's fastest rate. Windows the lag rule steps over are
 * counted as skipped rather than scored from stale data.
 */
export const MAX_BACKLOG_WINDOWS = 4;
export const MAX_BACKLOG_US = MAX_BACKLOG_WINDOWS * WINDOW_US;
export const MAX_WINDOWS_PER_TICK = 8;

/**
 * Upper bound on how far a scored window's start can trail the newest ingested
 * sample: the backlog the lag rule allows, plus one tick's worth of catch-up.
 *
 * `TraceHistory.fill` has no upper time bound, so a window read returns every
 * sample from the window's start to the newest sample in the ring. Sizing the
 * read buffer for this span is what keeps `fill` on its exact-copy path rather
 * than its min/max decimation path, and a decimated read would yield extrema
 * instead of the window's newest sample - a wrong answer that looks plausible.
 * Deriving the bound from the same constants that bound the loop is what keeps
 * the two from drifting apart.
 */
export const READ_SPAN_US = WINDOW_US + MAX_BACKLOG_US + MAX_WINDOWS_PER_TICK * WINDOW_US;

/**
 * How far back from an alarm onset a fault event may sit and still be offered as
 * that alarm's ground truth. Beyond it the event is a different occurrence and
 * latency would be a fiction, so the panel reports no latency instead.
 */
export const FAULT_LINK_US = 10_000_000;

/** Alarms retained for the panel's history list, newest first. */
export const RECENT_ALARMS = 6;

/** Fault events retained for latency attribution. */
const FAULT_EVENTS = 32;

/**
 * The slice of the store's ring buffer this module reads. Declared structurally
 * so the statistics have no dependency on React or on the store module.
 */
export interface SampleSource {
  latestTimeUs(): number;
  fill(
    channel: string,
    maxPoints: number,
    times: Float64Array,
    values: Float32Array,
    sinceTimeUs?: number,
  ): number;
}

export type AnomalyPhase = 'idle' | 'calibrating' | 'armed';

/** Which statistic carried a window over its threshold. */
export type AnomalyStatistic = 'T2' | 'Q' | 'both';

/**
 * One channel's contribution to a window's T², in the channel's own units.
 *
 * The breakdown is signed. T² = (v·z)² / lambda, and the cross terms of that
 * square do not belong to any single channel, so the only decomposition that
 * accounts for every part of the statistic is
 *
 *   contribution_j = (v·z) · v_j · z_j / lambda
 *
 * whose terms sum to T² exactly. A channel pressing against the dominant
 * component's direction gets a negative contribution, which is the useful part
 * of the information: it says the window is *less* anomalous than that channel
 * alone would suggest. Ranking by |contribution| is what `attribute` does in
 * `analytics/residuals.py`.
 */
export interface ChannelContribution {
  readonly channel: string;
  readonly unit: string;
  /** Observed value in the scored window. */
  readonly value: number;
  /** Calibration mean for the same channel. */
  readonly mean: number;
  /** Observed minus mean, in the channel's declared unit. */
  readonly residual: number;
  /** `residual / sigma` from the contract noise floor. */
  readonly z: number;
  /** This channel's signed contribution to T², in units of T². */
  readonly contribution: number;
  /** `contribution / T²` at the same window. The rows sum to one. */
  readonly share: number;
}

/** A ground-truth fault event attributed to an alarm, with its latency. */
export interface FaultLink {
  readonly faultType: string;
  readonly channel: string | null;
  readonly severity: number | null;
  readonly samples: number | null;
  /** `false` when the event was explicitly not ground truth. */
  readonly labelled: boolean | null;
  /** Simulated milliseconds from the fault event to the alarm onset. */
  readonly latencyMs: number;
}

export interface AnomalyAlarm {
  readonly id: number;
  readonly onsetTimeUs: number;
  readonly onsetT2: number;
  readonly onsetQ: number;
  readonly peakTimeUs: number;
  readonly peakT2: number;
  readonly peakQ: number;
  readonly statistic: AnomalyStatistic;
  /** Exceedances inside the last `n` windows when the alarm last changed. */
  readonly hits: number;
  readonly k: number;
  readonly n: number;
  /** Simulated milliseconds the alarm stayed open; `null` while it is. */
  readonly durationMs: number | null;
  readonly top: ChannelContribution | null;
  readonly contributions: readonly ChannelContribution[];
  readonly fault: FaultLink | null;
}

export interface CalibrationState {
  readonly windows: number;
  readonly required: number;
  readonly startUs: number | null;
  readonly endUs: number | null;
  /** A fault event fell inside the calibration span, so it may be dirty. */
  readonly contaminated: boolean;
  /** Leading eigenvalue of the standardised covariance, `null` until armed. */
  readonly eigenvalue: number | null;
}

export interface DetectorSnapshot {
  readonly phase: AnomalyPhase;
  /** Channels in the statistic. */
  readonly channels: number;
  readonly windowUs: number;
  readonly k: number;
  readonly n: number;
  readonly level: number;
  /** Windows scored since the connection started. */
  readonly windows: number;
  /** Windows not scored: missing input, ingest ahead of the statistic, or a full read buffer. */
  readonly skipped: number;
  readonly t2: number | null;
  readonly q: number | null;
  readonly t2Threshold: number | null;
  readonly qThreshold: number | null;
  /** Exceedances in the last `n` windows. */
  readonly hits: number;
  /** The last `n` exceedance flags, oldest first. */
  readonly flags: readonly boolean[];
  /** Alarms raised since the connection started. */
  readonly alarms: number;
  readonly calibration: CalibrationState;
  readonly active: AnomalyAlarm | null;
  readonly recent: readonly AnomalyAlarm[];
  /** Why the detector cannot score, when it cannot. */
  readonly problem: string | null;
}

interface FaultEvent {
  readonly id: number;
  readonly timeUs: number;
  readonly faultType: string;
  readonly channel: string | null;
  readonly severity: number | null;
  readonly samples: number | null;
  readonly labelled: boolean | null;
}

interface ActiveAlarm {
  id: number;
  onsetTimeUs: number;
  onsetT2: number;
  onsetQ: number;
  peakTimeUs: number;
  peakT2: number;
  peakQ: number;
  statistic: AnomalyStatistic;
  hits: number;
  peakRaw: Float64Array;
  peakZ: Float64Array;
  peakT1: number;
  fault: FaultLink | null;
}

/**
 * Contract channels eligible for the statistic: a declared noise floor to
 * normalise by, and a rate high enough that every window samples them.
 */
export function statisticChannels(): readonly ChannelSpec[] {
  return CHANNEL_NAMES.map((name) => CHANNELS[name]).filter(
    (spec) => spec.sigma > 0 && (spec.rateHz ?? 0) >= 100,
  );
}

/** Empirical quantile with linear interpolation, numpy's default method. */
function quantile(sorted: readonly number[], level: number): number {
  if (sorted.length === 0) {
    return Number.NaN;
  }
  const first = sorted[0] ?? Number.NaN;
  if (sorted.length === 1) {
    return first;
  }
  const position = level * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.min(lower + 1, sorted.length - 1);
  const fraction = position - lower;
  const lowValue = sorted[lower] ?? first;
  const highValue = sorted[upper] ?? lowValue;
  return lowValue + fraction * (highValue - lowValue);
}

/**
 * Leading eigenpair of a symmetric covariance matrix by power iteration from the
 * first canonical basis vector, so the result is a deterministic function of the
 * calibration windows alone. Rank 1 is all Layer 2 needs for a live in-manifold
 * statistic; `lambda_1` and `v` are what `fit_multivariate(n_components=1)`
 * would return for the same window.
 */
function leadingEigenpair(
  covariance: Float64Array,
  size: number,
): { value: number; vector: Float64Array } | null {
  const vector = new Float64Array(size);
  const product = new Float64Array(size);
  vector[0] = 1;
  let value = Number.NEGATIVE_INFINITY;
  for (let iteration = 0; iteration < 256; iteration += 1) {
    for (let row = 0; row < size; row += 1) {
      let sum = 0;
      for (let column = 0; column < size; column += 1) {
        sum += (covariance[row * size + column] ?? 0) * (vector[column] ?? 0);
      }
      product[row] = sum;
    }
    let norm = 0;
    for (let row = 0; row < size; row += 1) {
      norm += (product[row] ?? 0) ** 2;
    }
    norm = Math.sqrt(norm);
    if (!(norm > 0)) {
      return null;
    }
    for (let row = 0; row < size; row += 1) {
      vector[row] = (product[row] ?? 0) / norm;
    }
    // Rayleigh quotient along the current iterate is the eigenvalue estimate.
    let quotient = 0;
    for (let row = 0; row < size; row += 1) {
      let sum = 0;
      for (let column = 0; column < size; column += 1) {
        sum += (covariance[row * size + column] ?? 0) * (vector[column] ?? 0);
      }
      quotient += (vector[row] ?? 0) * sum;
    }
    if (Math.abs(quotient - value) <= 1e-9 * Math.max(1, Math.abs(quotient))) {
      value = quotient;
      break;
    }
    value = quotient;
  }
  if (!(value > 0)) {
    return null;
  }
  return { value, vector };
}

export class AnomalyDetector {
  private readonly source: SampleSource;
  private readonly specs: readonly ChannelSpec[];
  private readonly size: number;
  /** Per-window read bound: every sample inside the span, so `fill` copies exactly. */
  private readonly points: number;
  private readonly times: Float64Array;
  private readonly readValues: Float32Array;
  private readonly raw: Float64Array;
  private readonly z: Float64Array;
  private readonly sigmas: Float64Array;

  private phase: AnomalyPhase = 'idle';
  private problem: string | null = null;
  private nextEndUs = 0;
  private started = false;

  private readonly calSum: Float64Array;
  private readonly calRows: Float64Array;
  private calCount = 0;
  private calStartUs: number | null = null;
  private calEndUs: number | null = null;
  private contaminated = false;

  private mean: Float64Array | null = null;
  private loadings: Float64Array | null = null;
  private eigenvalue: number | null = null;
  private t2Threshold: number | null = null;
  private qThreshold: number | null = null;

  private windows = 0;
  private skipped = 0;
  private t2: number | null = null;
  private q: number | null = null;
  private readonly flags: boolean[] = [];
  private hits = 0;
  private alarmCount = 0;
  private nextAlarmId = 1;
  private active: ActiveAlarm | null = null;
  private readonly recent: AnomalyAlarm[] = [];

  private readonly faults: FaultEvent[] = [];
  private lastFaultId = 0;
  private lastFaultTimeUs: number | null = null;

  constructor(source: SampleSource, specs: readonly ChannelSpec[] = statisticChannels()) {
    this.source = source;
    this.specs = specs;
    this.size = specs.length;
    const rate = specs.reduce((highest, spec) => Math.max(highest, spec.rateHz ?? 0), 0);
    this.points = Math.ceil((rate * READ_SPAN_US) / 1e6) + 2;
    this.times = new Float64Array(this.points);
    this.readValues = new Float32Array(this.points);
    this.raw = new Float64Array(this.size);
    this.z = new Float64Array(this.size);
    this.sigmas = new Float64Array(this.size);
    this.calSum = new Float64Array(this.size);
    this.calRows = new Float64Array(this.size * CALIBRATION_WINDOWS);
    for (let index = 0; index < this.size; index += 1) {
      this.sigmas[index] = specs[index]?.sigma ?? 0;
    }
    if (this.size < 2) {
      this.problem = `the statistic needs at least 2 channels with a noise scale, the contract gives ${this.size}`;
    }
  }

  /**
   * Feed the store's fault-event stream. Entries are consumed once, by
   * increasing id, so the bounded event log can be re-read on every change
   * without double counting. A rewind restarts the timeline.
   */
  syncFaultEvents(entries: readonly EventLogEntry[]): void {
    for (const entry of entries) {
      if (entry.kind !== 'fault' || entry.id <= this.lastFaultId) {
        continue;
      }
      if (this.lastFaultTimeUs !== null && entry.timeUs < this.lastFaultTimeUs) {
        // Replay rewind: the annotations before it belong to another timeline.
        this.faults.length = 0;
        this.contaminated = false;
      }
      this.lastFaultId = entry.id;
      this.lastFaultTimeUs = entry.timeUs;
      this.faults.push({
        id: entry.id,
        timeUs: entry.timeUs,
        faultType: entry.faultType,
        channel: entry.channel,
        severity: entry.severity,
        samples: entry.durationSamples,
        labelled: entry.label,
      });
      if (this.faults.length > FAULT_EVENTS) {
        this.faults.shift();
      }
      if (this.calStartUs !== null && entry.timeUs >= this.calStartUs) {
        // The fault falls inside the calibration span, so the frozen manifold may
        // have learned from it. The samples that would settle the question have
        // long since left the ring buffer, so this stays flagged.
        this.contaminated = true;
      }
    }
  }

  /** Score every complete window that has become available since the last call. */
  advance(latestTimeUs: number): void {
    if (this.problem !== null || !Number.isFinite(latestTimeUs) || latestTimeUs <= 0) {
      return;
    }
    if (!this.started) {
      this.started = true;
      this.nextEndUs = latestTimeUs;
      this.phase = 'calibrating';
    }
    if (latestTimeUs - this.nextEndUs > MAX_BACKLOG_US) {
      // Ingest is running ahead of the statistic. Step the window grid forward
      // whole windows and count what was dropped rather than scoring from a read
      // that would have to be decimated.
      const behind = latestTimeUs - MAX_BACKLOG_US - this.nextEndUs;
      const drop = Math.floor(behind / WINDOW_US) + 1;
      this.skipped += drop;
      this.nextEndUs += drop * WINDOW_US;
    }
    let budget = MAX_WINDOWS_PER_TICK;
    while (this.nextEndUs <= latestTimeUs && budget > 0) {
      budget -= 1;
      const endUs = this.nextEndUs;
      this.nextEndUs = endUs + WINDOW_US;
      this.score(endUs);
    }
  }

  snapshot(): DetectorSnapshot {
    return {
      phase: this.phase,
      channels: this.size,
      windowUs: WINDOW_US,
      k: PERSISTENCE_K,
      n: PERSISTENCE_N,
      level: THRESHOLD_LEVEL,
      windows: this.windows,
      skipped: this.skipped,
      t2: this.t2,
      q: this.q,
      t2Threshold: this.t2Threshold,
      qThreshold: this.qThreshold,
      hits: this.hits,
      flags: [...this.flags],
      alarms: this.alarmCount,
      calibration: {
        windows: this.calCount,
        required: CALIBRATION_WINDOWS,
        startUs: this.calStartUs,
        endUs: this.calEndUs,
        contaminated: this.contaminated,
        eigenvalue: this.eigenvalue,
      },
      active: this.active === null ? null : this.alarm(this.active, null),
      recent: [...this.recent],
      problem: this.problem,
    };
  }

  private score(endUs: number): void {
    if (!this.read(endUs)) {
      this.skipped += 1;
      return;
    }
    if (this.phase === 'calibrating') {
      this.calibrate(endUs);
      return;
    }
    const mean = this.mean;
    const loadings = this.loadings;
    const eigenvalue = this.eigenvalue;
    if (mean === null || loadings === null || eigenvalue === null) {
      return;
    }
    let score1 = 0;
    let norm = 0;
    for (let index = 0; index < this.size; index += 1) {
      const value = (this.raw[index] ?? 0) - (mean[index] ?? 0);
      const z = value / (this.sigmas[index] ?? 1);
      this.z[index] = z;
      const weighted = (loadings[index] ?? 0) * z;
      score1 += weighted;
      norm += z * z;
    }
    const t2 = (score1 * score1) / eigenvalue;
    const q = Math.max(norm - score1 * score1, 0);
    this.windows += 1;
    this.t2 = t2;
    this.q = q;

    const t2Exceeds = t2 > (this.t2Threshold ?? Number.POSITIVE_INFINITY);
    const qExceeds = q > (this.qThreshold ?? Number.POSITIVE_INFINITY);
    const flag = t2Exceeds || qExceeds;
    this.flags.push(flag);
    if (this.flags.length > PERSISTENCE_N) {
      this.flags.shift();
    }
    this.hits = this.flags.reduce((total, entry) => total + (entry ? 1 : 0), 0);

    const sustained = this.hits >= PERSISTENCE_K;
    if (sustained && this.active === null) {
      this.raise(endUs, t2, q, score1, t2Exceeds, qExceeds);
    } else if (sustained) {
      this.extend(endUs, t2, q, score1, t2Exceeds, qExceeds);
    } else if (this.active !== null) {
      this.release(endUs);
    }
  }

  /**
   * Read the newest sample of every eligible channel inside `[end - WINDOW_US,
   * end)`. A window with a missing, stale or unreadable channel is skipped, as
   * Layer 2 skips a frame with a missing input.
   */
  private read(endUs: number): boolean {
    const startUs = endUs - WINDOW_US;
    for (let index = 0; index < this.size; index += 1) {
      const name = this.specs[index]?.name;
      if (name === undefined) {
        return false;
      }
      const count = this.source.fill(name, this.points, this.times, this.readValues, startUs);
      if (count === 0 || count >= this.points) {
        // Nothing in the window, or a buffer too full to read exactly.
        return false;
      }
      let value = Number.NaN;
      for (let sample = count - 1; sample >= 0; sample -= 1) {
        const time = this.times[sample] ?? Number.NaN;
        if (time < endUs) {
          value = this.readValues[sample] ?? Number.NaN;
          break;
        }
      }
      if (!Number.isFinite(value)) {
        return false;
      }
      this.raw[index] = value;
    }
    return true;
  }

  private calibrate(endUs: number): void {
    if (this.calCount >= CALIBRATION_WINDOWS) {
      this.arm();
      return;
    }
    this.calCount += 1;
    const row = (this.calCount - 1) * this.size;
    if (this.calStartUs === null) {
      const startUs = endUs - WINDOW_US;
      this.calStartUs = startUs;
      // Fault events can arrive before the first window does, so the retained
      // ones are re-checked against the span once there is a span to check.
      this.contaminated = this.faults.some((fault) => fault.timeUs >= startUs);
    }
    this.calEndUs = endUs;
    for (let index = 0; index < this.size; index += 1) {
      const value = this.raw[index] ?? 0;
      this.calSum[index] = (this.calSum[index] ?? 0) + value;
      // Noise-scaled but not yet centred. The mean is only known once the span is
      // complete, so `arm` centres every row on it, as the offline fit does. The
      // samples themselves have long since left the ring buffer, so the scaled
      // values are all there is to keep.
      this.calRows[row + index] = value / (this.sigmas[index] ?? 1);
    }
    if (this.calCount >= CALIBRATION_WINDOWS) {
      this.arm();
    }
  }

  private arm(): void {
    const size = this.size;
    const count = this.calCount;
    const mean = new Float64Array(size);
    for (let index = 0; index < size; index += 1) {
      mean[index] = (this.calSum[index] ?? 0) / count;
    }
    // Centre every calibration row on the final span mean, so the covariance, the
    // thresholds and the live scores all measure deviation from one and the same mean.
    for (let window = 0; window < count; window += 1) {
      const row = window * size;
      for (let index = 0; index < size; index += 1) {
        const scaledMean = (mean[index] ?? 0) / (this.sigmas[index] ?? 1);
        this.calRows[row + index] = (this.calRows[row + index] ?? 0) - scaledMean;
      }
    }
    const covariance = new Float64Array(size * size);
    for (let window = 0; window < count; window += 1) {
      const row = window * size;
      for (let a = 0; a < size; a += 1) {
        const za = this.calRows[row + a] ?? 0;
        if (za === 0) {
          continue;
        }
        for (let b = 0; b < size; b += 1) {
          covariance[a * size + b] = (covariance[a * size + b] ?? 0) + za * (this.calRows[row + b] ?? 0);
        }
      }
    }
    const scale = count > 1 ? 1 / (count - 1) : 0;
    for (let index = 0; index < covariance.length; index += 1) {
      covariance[index] = (covariance[index] ?? 0) * scale;
    }
    const eigen = leadingEigenpair(covariance, size);
    if (eigen === null) {
      this.problem = 'the calibration window is degenerate: no variance along the leading component';
      this.phase = 'idle';
      return;
    }
    const t2s: number[] = [];
    const qs: number[] = [];
    for (let window = 0; window < count; window += 1) {
      const row = window * size;
      let score1 = 0;
      let norm = 0;
      for (let index = 0; index < size; index += 1) {
        const z = this.calRows[row + index] ?? 0;
        score1 += (eigen.vector[index] ?? 0) * z;
        norm += z * z;
      }
      t2s.push((score1 * score1) / eigen.value);
      qs.push(Math.max(norm - score1 * score1, 0));
    }
    t2s.sort((a, b) => a - b);
    qs.sort((a, b) => a - b);
    this.mean = mean;
    this.loadings = eigen.vector;
    this.eigenvalue = eigen.value;
    this.t2Threshold = quantile(t2s, THRESHOLD_LEVEL);
    this.qThreshold = quantile(qs, THRESHOLD_LEVEL);
    this.phase = 'armed';
    this.flags.length = 0;
    this.hits = 0;
  }

  private raise(
    endUs: number,
    t2: number,
    q: number,
    score1: number,
    t2Exceeds: boolean,
    qExceeds: boolean,
  ): void {
    this.active = {
      id: this.nextAlarmId,
      onsetTimeUs: endUs,
      onsetT2: t2,
      onsetQ: q,
      peakTimeUs: endUs,
      peakT2: t2,
      peakQ: q,
      statistic: t2Exceeds && qExceeds ? 'both' : t2Exceeds ? 'T2' : 'Q',
      hits: this.hits,
      peakRaw: Float64Array.from(this.raw),
      peakZ: Float64Array.from(this.z),
      peakT1: score1,
      fault: this.link(endUs),
    };
    this.nextAlarmId += 1;
    this.alarmCount += 1;
  }

  private extend(
    endUs: number,
    t2: number,
    q: number,
    score1: number,
    t2Exceeds: boolean,
    qExceeds: boolean,
  ): void {
    const alarm = this.active;
    if (alarm === null) {
      return;
    }
    alarm.hits = this.hits;
    if (q > alarm.peakQ) {
      alarm.peakQ = q;
    }
    // The stored window is the T² peak and nothing else: the per-channel
    // breakdown divides by that same T², so it must come from the window T² was
    // measured on or the shares would not sum to one.
    if (t2 > alarm.peakT2) {
      alarm.peakT2 = t2;
      alarm.peakTimeUs = endUs;
      alarm.peakRaw.set(this.raw);
      alarm.peakZ.set(this.z);
      alarm.peakT1 = score1;
    }
    if (t2Exceeds && qExceeds) {
      alarm.statistic = 'both';
    } else if (t2Exceeds && alarm.statistic === 'Q') {
      alarm.statistic = 'both';
    } else if (qExceeds && alarm.statistic === 'T2') {
      alarm.statistic = 'both';
    }
  }

  private release(endUs: number): void {
    const alarm = this.active;
    if (alarm === null) {
      return;
    }
    this.recent.unshift(this.alarm(alarm, (endUs - alarm.onsetTimeUs) / 1000));
    if (this.recent.length > RECENT_ALARMS) {
      this.recent.length = RECENT_ALARMS;
    }
    this.active = null;
  }

  /**
   * The most recent ground-truth fault event at or before the onset, within
   * `FAULT_LINK_US`. Anything else would be a different occurrence, and a
   * latency measured from it would not be a detection latency.
   */
  private link(onsetTimeUs: number): FaultLink | null {
    for (let index = this.faults.length - 1; index >= 0; index -= 1) {
      const fault = this.faults[index];
      if (fault === undefined || fault.timeUs > onsetTimeUs) {
        continue;
      }
      if (onsetTimeUs - fault.timeUs > FAULT_LINK_US) {
        break;
      }
      return {
        faultType: fault.faultType,
        channel: fault.channel,
        severity: fault.severity,
        samples: fault.samples,
        labelled: fault.labelled,
        latencyMs: (onsetTimeUs - fault.timeUs) / 1000,
      };
    }
    return null;
  }

  private alarm(alarm: ActiveAlarm, durationMs: number | null): AnomalyAlarm {
    const contributions = this.contributions(alarm);
    return {
      id: alarm.id,
      onsetTimeUs: alarm.onsetTimeUs,
      onsetT2: alarm.onsetT2,
      onsetQ: alarm.onsetQ,
      peakTimeUs: alarm.peakTimeUs,
      peakT2: alarm.peakT2,
      peakQ: alarm.peakQ,
      statistic: alarm.statistic,
      hits: alarm.hits,
      k: PERSISTENCE_K,
      n: PERSISTENCE_N,
      durationMs,
      top: contributions[0] ?? null,
      contributions,
      fault: alarm.fault,
    };
  }

  /**
   * Per-channel attribution at the alarm's T² peak, ranked by |contribution|.
   * Every term is signed and the terms sum to the window's T² exactly, so the
   * table cannot quietly over- or under-account for the statistic it explains.
   */
  private contributions(alarm: ActiveAlarm): ChannelContribution[] {
    const loadings = this.loadings;
    const mean = this.mean;
    const eigenvalue = this.eigenvalue;
    if (loadings === null || mean === null || eigenvalue === null) {
      return [];
    }
    const score1 = alarm.peakT1;
    const rows: ChannelContribution[] = [];
    for (let index = 0; index < this.size; index += 1) {
      const spec = this.specs[index];
      if (spec === undefined) {
        continue;
      }
      const contribution = (score1 * (loadings[index] ?? 0) * (alarm.peakZ[index] ?? 0)) / eigenvalue;
      const value = alarm.peakRaw[index] ?? 0;
      const channelMean = mean[index] ?? 0;
      rows.push({
        channel: spec.name,
        unit: spec.unit,
        value,
        mean: channelMean,
        residual: value - channelMean,
        z: alarm.peakZ[index] ?? 0,
        contribution,
        share: alarm.peakT2 > 0 ? contribution / alarm.peakT2 : 0,
      });
    }
    rows.sort(
      (a, b) => Math.abs(b.contribution) - Math.abs(a.contribution) || a.channel.localeCompare(b.channel),
    );
    return rows;
  }
}
