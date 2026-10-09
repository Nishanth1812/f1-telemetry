import { useEffect, useId, useState } from 'react';
import { useTelemetryStore } from '../store/telemetryStore';
import {
  AnomalyDetector,
  FAULT_LINK_US,
  PERSISTENCE_K,
  type AnomalyAlarm,
  type ChannelContribution,
  type DetectorSnapshot,
} from '../telemetry/anomaly';
import { formatDurationMs, formatValue } from '../telemetry/format';

/**
 * Anomaly panel (PLAN.md section 9 layer 5; PHASES.md P8-T7).
 *
 * Four numbers, all of them produced by `telemetry/anomaly.ts` from ingested
 * samples and from the store's own fault-event stream, and nothing else:
 *
 * - **severity** as the Hotelling T² (and its out-of-manifold companion Q) at the
 *   alarm's peak window, against the empirical threshold calibrated on this
 *   connection;
 * - **per-channel contribution** as each channel's share of that T², with the
 *   residual in the channel's own declared unit;
 * - **detection latency** as simulated milliseconds from the linked ground-truth
 *   fault event to the alarm onset, quantised to the statistic window, or a dash
 *   when no fault event is attributable to the alarm;
 * - **persistence** as the k-of-n window count that raised the alarm.
 *
 * There is no confidence score and nothing that could stand in for one. A number
 * the panel cannot compute from the data is shown as a dash with the reason
 * beside it, which is why the negative control here is simply "alarms 0".
 *
 * Every 200 Hz sample is read inside the detector, straight from the store's
 * ring buffer. React only ever sees a 4 Hz snapshot of already-summarised
 * numbers, so nothing in the render path touches sample data.
 */

const PUBLISH_MS = 250;
const BREAKDOWN_ROWS = 6;

function stat(value: number | null, digits = 2): string {
  return value === null || !Number.isFinite(value) ? '—' : value.toFixed(digits);
}

function duration(timeUs: number | null): string {
  return timeUs === null ? '—' : `${(timeUs / 1e6).toFixed(3)} s`;
}

function latency(fault: AnomalyAlarm['fault']): string {
  if (fault === null) {
    return '—';
  }
  const where = fault.channel === null ? fault.faultType : `${fault.faultType} on ${fault.channel}`;
  const labelled = fault.labelled === false ? ' (not ground truth)' : '';
  return `${formatDurationMs(fault.latencyMs)} after ${where}${labelled}`;
}

/** Signed percentage of T², so the sign is never lost to formatting. */
function shareText(share: number): string {
  return `${share > 0 ? '+' : share < 0 ? '−' : '±'}${Math.abs(share * 100).toFixed(0)}`;
}

/** Tone for a contribution: driving the statistic, damping it, or negligible. */
function shareTone(share: number): 'lead' | 'negative' | 'neutral' {
  if (share >= 0.05) {
    return 'lead';
  }
  return share <= -0.005 ? 'negative' : 'neutral';
}

/** Rows of the breakdown table, padded with dashes so the table keeps its shape. */
function breakdownRows(alarm: AnomalyAlarm | null): (ChannelContribution | null)[] {
  const rows: (ChannelContribution | null)[] = [];
  for (let index = 0; index < BREAKDOWN_ROWS; index += 1) {
    rows.push(alarm === null ? null : (alarm.contributions[index] ?? null));
  }
  return rows;
}

export function AnomalyPanel() {
  const headingId = useId();
  const [snapshot, setSnapshot] = useState<DetectorSnapshot | null>(null);

  useEffect(() => {
    const store = useTelemetryStore.getState();
    const detector = new AnomalyDetector(store.history);
    setSnapshot(detector.snapshot());
    // The detector owns the sample reads; this interval only hands it the
    // newest ingested timestamp and republishes its summary.
    const timer = window.setInterval(() => {
      const current = useTelemetryStore.getState();
      detector.syncFaultEvents(current.events);
      detector.advance(current.history.latestTimeUs());
      setSnapshot(detector.snapshot());
    }, PUBLISH_MS);
    return () => window.clearInterval(timer);
  }, []);

  const phase = snapshot?.phase ?? 'idle';
  const calibrating = phase === 'calibrating';
  const armed = phase === 'armed';
  const alarm = snapshot === null ? null : (snapshot.active ?? snapshot.recent[0] ?? null);
  const open = snapshot !== null && snapshot.active !== null;
  const calibration = snapshot?.calibration ?? null;

  return (
    <section className="anomaly" aria-labelledby={headingId}>
      <div className="anomaly__header">
        <h2 className="anomaly__heading" id={headingId}>
          Anomalies
        </h2>
        <span className="anomaly__note">
          layer 2 T² and Q · k-of-n persistence · no confidence score
        </span>
      </div>

      {snapshot === null ? (
        <p className="placeholder" role="note">
          Waiting for ingested samples.
        </p>
      ) : snapshot.problem !== null ? (
        <p className="placeholder" role="note">
          {snapshot.problem}
        </p>
      ) : (
        <>
          <dl className="anomaly__stats">
            <div className="anomaly__stat">
              <dt>severity T²</dt>
              <dd>
                {stat(snapshot.t2)}
                <span className="anomaly__limit"> / {stat(snapshot.t2Threshold)}</span>
              </dd>
            </div>
            <div className="anomaly__stat">
              <dt>Q (out-of-manifold)</dt>
              <dd>
                {stat(snapshot.q)}
                <span className="anomaly__limit"> / {stat(snapshot.qThreshold)}</span>
              </dd>
            </div>
            <div className="anomaly__stat">
              <dt>windows scored</dt>
              <dd>
                {snapshot.windows}
                <span className="anomaly__limit"> · {snapshot.skipped} skipped</span>
              </dd>
            </div>
            <div className="anomaly__stat">
              <dt>alarms</dt>
              <dd>{snapshot.alarms}</dd>
            </div>
            <div className="anomaly__stat anomaly__stat--wide">
              <dt>{calibrating ? 'calibrating' : 'calibration'}</dt>
              <dd>
                {calibration === null
                  ? '—'
                  : `${calibration.windows}/${calibration.required} windows`}
                <span className="anomaly__limit">
                  {calibration?.eigenvalue === null || calibration?.eigenvalue === undefined
                    ? ''
                    : ` · λ₁ ${formatValue(calibration.eigenvalue)}`}
                  {calibration?.contaminated === true ? ' · fault event in span' : ''}
                </span>
              </dd>
            </div>
          </dl>

          <p className="anomaly__state" role="note">
            {calibrating
              ? `Calibrating the manifold on the first ${calibration?.required ?? 0} windows of this connection, ${snapshot.windowUs / 1000} ms each. No alarm can be raised before it is frozen.`
              : armed
                ? snapshot.calibration.contaminated
                  ? `Armed on ${snapshot.calibration.windows} windows. A fault event fell inside the calibration span, so PLAN.md section 9.1 clean-run calibration is not satisfied on this connection.`
                  : `Armed on ${snapshot.calibration.windows} windows, ${snapshot.windowUs / 1000} ms each. Both thresholds are the empirical ${(snapshot.level * 100).toFixed(1)}% quantile of those windows.`
                : 'Idle.'}
          </p>

          <ol
            className={`anomaly__windows anomaly__windows--${snapshot.hits >= PERSISTENCE_K ? 'tripped' : 'quiet'}`}
            aria-label={`Persistence: ${snapshot.hits} of the last ${snapshot.n} statistic windows exceeded a threshold, ${PERSISTENCE_K} required`}
          >
            {Array.from({ length: snapshot.n }, (_empty, index) => {
              const hit = snapshot.flags[index] ?? false;
              return (
                <li
                  key={index}
                  className={`anomaly__window${hit ? ' anomaly__window--hit' : ''}`}
                  data-hit={hit}
                />
              );
            })}
          </ol>

          <div className="anomaly__alarm">
            <div className="anomaly__alarm-head">
              <span className={`anomaly__badge anomaly__badge--${open ? 'open' : 'closed'}`}>
                {open ? 'alarm open' : alarm === null ? 'no alarm yet' : 'alarm released'}
              </span>
              {alarm !== null ? (
                <span className="anomaly__alarm-stat">
                  T² {stat(alarm.peakT2)} peak · Q {stat(alarm.peakQ)} peak · {alarm.statistic}
                </span>
              ) : null}
            </div>
            {alarm === null ? (
              <p className="anomaly__hint">
                {armed
                  ? `Silent so far: no window met the persistence rule. This is the state a clean run should stay in.`
                  : 'No statistic yet.'}
              </p>
            ) : (
              <dl className="anomaly__facts">
                <div className="anomaly__fact">
                  <dt>onset</dt>
                  <dd>
                    {duration(alarm.onsetTimeUs)} · T² {stat(alarm.onsetT2)}
                  </dd>
                </div>
                <div className="anomaly__fact">
                  <dt>persistence</dt>
                  <dd>
                    {alarm.hits} of {alarm.n} windows ({alarm.k} required)
                  </dd>
                </div>
                <div className="anomaly__fact">
                  <dt>detection latency</dt>
                  <dd>
                    {latency(alarm.fault)}
                    {alarm.fault === null
                      ? ` · no fault event within ${FAULT_LINK_US / 1e6} s of the onset`
                      : ''}
                  </dd>
                </div>
                <div className="anomaly__fact">
                  <dt>{open ? 'open for' : 'duration'}</dt>
                  <dd>
                    {open || alarm.durationMs === null
                      ? duration(alarm.peakTimeUs - alarm.onsetTimeUs)
                      : formatDurationMs(alarm.durationMs)}
                  </dd>
                </div>
              </dl>
            )}
          </div>

          <table className="anomaly__breakdown">
            <caption>
              signed per-channel contribution to T² at the alarm&apos;s peak window, ranked by
              magnitude. The terms sum to that T² exactly; a negative term is a channel pressing against
              the dominant component&apos;s direction.
            </caption>
            <thead>
              <tr>
                <th scope="col">channel</th>
                <th scope="col">value</th>
                <th scope="col">residual</th>
                <th scope="col">z</th>
                <th scope="col">share of T²</th>
              </tr>
            </thead>
            <tbody>
              {breakdownRows(alarm).map((row, index) => (
                <tr key={row?.channel ?? index} className={row === null ? 'anomaly__row--empty' : ''}>
                  <th scope="row">{row?.channel ?? '—'}</th>
                  <td>{row === null ? '—' : `${formatValue(row.value)} ${row.unit}`}</td>
                  <td>{row === null ? '—' : `${formatValue(row.residual)} ${row.unit}`}</td>
                  <td>{row === null ? '—' : formatValue(row.z)}</td>
                  <td className={row === null ? '' : `anomaly__share anomaly__share--${shareTone(row.share)}`}>
                    {row === null ? '—' : `${shareText(row.share)} %`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {snapshot.recent.length > 0 ? (
            <ul className="anomaly__recent" aria-label="Alarms raised earlier in this connection">
              {snapshot.recent.map((entry) => (
                <li key={entry.id} className="anomaly__recent-item">
                  <span className="anomaly__recent-time">{duration(entry.onsetTimeUs)}</span>
                  <span className="anomaly__recent-text">
                    T² {stat(entry.peakT2)} · {entry.statistic} · {entry.hits}/{entry.n} windows ·{' '}
                    {entry.fault === null ? 'latency —' : `latency ${formatDurationMs(entry.fault.latencyMs)}`}
                  </span>
                </li>
              ))}
            </ul>
          ) : null}

          <p className="anomaly__footnote">
            Statistics computed in the browser from {snapshot.channels} contract channels at{' '}
            {snapshot.windowUs / 1000} ms per window. Published latency and false-alarm rates come from
            the offline detection sweep, not from this view.
          </p>
        </>
      )}
    </section>
  );
}
