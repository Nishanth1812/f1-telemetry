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
import { channelLabel, faultLabel } from './Tile';

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
 *
 * On screen the statistics carry plain names; the technical symbol is kept in
 * the `title` tooltip so engineers can still find it.
 */

const PUBLISH_MS = 250;
const BREAKDOWN_ROWS = 6;

const STATISTIC_NAMES: Readonly<Record<AnomalyAlarm['statistic'], string>> = {
  both: 'both scores',
  T2: 'unusualness score',
  Q: 'off-pattern score',
};

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
  const where =
    fault.channel === null
      ? faultLabel(fault.faultType)
      : `${faultLabel(fault.faultType)} on ${channelLabel(fault.channel)}`;
  const labelled = fault.labelled === false ? ' (not confirmed as injected)' : '';
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
          Flags unusual combinations of sensor readings. It gives no confidence score.
        </span>
      </div>

      {snapshot === null ? (
        <p className="placeholder" role="note">
          Waiting for sensor data. Warnings appear here once the simulator is sending readings.
        </p>
      ) : snapshot.problem !== null ? (
        <p className="placeholder" role="note">
          {snapshot.problem}
        </p>
      ) : (
        <>
          <dl className="anomaly__stats">
            <div className="anomaly__stat">
              <dt title="Hotelling T² (severity)">Unusualness score</dt>
              <dd>
                {stat(snapshot.t2)}
                <span className="anomaly__limit"> / {stat(snapshot.t2Threshold)}</span>
              </dd>
            </div>
            <div className="anomaly__stat">
              <dt title="Q statistic (out-of-manifold)">Off-pattern score</dt>
              <dd>
                {stat(snapshot.q)}
                <span className="anomaly__limit"> / {stat(snapshot.qThreshold)}</span>
              </dd>
            </div>
            <div className="anomaly__stat">
              <dt>Checks made</dt>
              <dd>
                {snapshot.windows}
                <span className="anomaly__limit"> · {snapshot.skipped} skipped</span>
              </dd>
            </div>
            <div className="anomaly__stat">
              <dt>Warnings raised</dt>
              <dd>{snapshot.alarms}</dd>
            </div>
            <div className="anomaly__stat anomaly__stat--wide">
              <dt>{calibrating ? 'Learning what normal looks like' : 'Setup'}</dt>
              <dd>
                {calibration === null
                  ? '—'
                  : `${calibration.windows} of ${calibration.required} setup checks`}
                <span className="anomaly__limit">
                  {calibration?.eigenvalue === null || calibration?.eigenvalue === undefined
                    ? ''
                    : ` · main pattern strength ${formatValue(calibration.eigenvalue)}`}
                  {calibration?.contaminated === true ? ' · a fault happened during setup' : ''}
                </span>
              </dd>
            </div>
          </dl>

          <p className="anomaly__state" role="note">
            {calibrating
              ? `Learning what normal looks like from the first ${calibration?.required ?? 0} checks of this connection, ${snapshot.windowUs / 1000} ms each. No warning can be raised until this is finished.`
              : armed
                ? snapshot.calibration.contaminated
                  ? `Ready, but a fault happened during setup, so this setup may not reflect a normal run.`
                  : `Ready. Checks run every ${snapshot.windowUs / 1000} ms. Thresholds are set to the top ${(snapshot.level * 100).toFixed(1)}% of readings seen during setup.`
                : 'Not started yet.'}
          </p>

          <ol
            className={`anomaly__windows anomaly__windows--${snapshot.hits >= PERSISTENCE_K ? 'tripped' : 'quiet'}`}
            aria-label={`Checks in a row: ${snapshot.hits} of the last ${snapshot.n} checks looked unusual, ${PERSISTENCE_K} needed to raise a warning`}
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
                {open ? 'Warning active' : alarm === null ? 'No warnings yet' : 'Warning ended'}
              </span>
              {alarm !== null ? (
                <span className="anomaly__alarm-stat">
                  Peak unusualness {stat(alarm.peakT2)} · peak off-pattern {stat(alarm.peakQ)} ·{' '}
                  {STATISTIC_NAMES[alarm.statistic]}
                </span>
              ) : null}
            </div>
            {alarm === null ? (
              <p className="anomaly__hint">
                {armed
                  ? 'No warnings so far — everything looks normal.'
                  : 'Not enough data yet to make a judgement.'}
              </p>
            ) : (
              <dl className="anomaly__facts">
                <div className="anomaly__fact">
                  <dt>First unusual moment</dt>
                  <dd>
                    {duration(alarm.onsetTimeUs)} · unusualness {stat(alarm.onsetT2)}
                  </dd>
                </div>
                <div className="anomaly__fact">
                  <dt>Checks in a row</dt>
                  <dd>
                    {alarm.hits} of {alarm.n} checks looked unusual ({alarm.k} needed)
                  </dd>
                </div>
                <div className="anomaly__fact">
                  <dt>Time from fault to warning</dt>
                  <dd>
                    {latency(alarm.fault)}
                    {alarm.fault === null
                      ? ` · no logged fault within ${FAULT_LINK_US / 1e6} s of the first unusual moment`
                      : ''}
                  </dd>
                </div>
                <div className="anomaly__fact">
                  <dt>{open ? 'Active for' : 'Lasted'}</dt>
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
              How much each sensor contributed to the unusualness score at the strongest moment,
              biggest first. The contributions add up to the total score. A negative value means
              that sensor pulled against the main pattern.
            </caption>
            <thead>
              <tr>
                <th scope="col">Sensor</th>
                <th scope="col">Reading</th>
                <th scope="col">Gap from normal</th>
                <th scope="col" title="z-score">Deviation</th>
                <th scope="col">Share of score</th>
              </tr>
            </thead>
            <tbody>
              {breakdownRows(alarm).map((row, index) => (
                <tr key={row?.channel ?? index} className={row === null ? 'anomaly__row--empty' : ''}>
                  <th scope="row" title={row?.channel}>
                    {row === null ? '—' : channelLabel(row.channel)}
                  </th>
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
            <ul className="anomaly__recent" aria-label="Warnings raised earlier in this connection">
              {snapshot.recent.map((entry) => (
                <li key={entry.id} className="anomaly__recent-item">
                  <span className="anomaly__recent-time">{duration(entry.onsetTimeUs)}</span>
                  <span className="anomaly__recent-text">
                    Unusualness {stat(entry.peakT2)} · {STATISTIC_NAMES[entry.statistic]} ·{' '}
                    {entry.hits} of {entry.n} checks ·{' '}
                    {entry.fault === null
                      ? 'no fault timing'
                      : `warned ${formatDurationMs(entry.fault.latencyMs)} after the fault`}
                  </span>
                </li>
              ))}
            </ul>
          ) : null}

          <p className="anomaly__footnote">
            Scores are computed in your browser from {snapshot.channels} sensors, checked every{' '}
            {snapshot.windowUs / 1000} ms. The detection-speed and false-alarm figures published
            elsewhere come from the offline test sweep, not from this view.
          </p>
        </>
      )}
    </section>
  );
}
