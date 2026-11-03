import { useEffect, useState } from 'react';
import { useLabStore } from '../store/labStore';
import { useTelemetryStore } from '../store/telemetryStore';

export function SavedRuns() {
  const status = useTelemetryStore((state) => state.status);
  const readOnly = useLabStore((state) => state.readOnly);
  const runs = useLabStore((state) => state.savedRuns);
  const loading = useLabStore((state) => state.loadingSaved);
  const running = useLabStore((state) => state.running);
  const refresh = useLabStore((state) => state.refreshSaved);
  const replay = useLabStore((state) => state.replaySaved);
  const compare = useLabStore((state) => state.compareSaved);
  const [selected, setSelected] = useState('');
  const [baseline, setBaseline] = useState('');
  const current = runs.find((run) => run.run_id === selected) ?? runs[0];
  const candidates = runs.filter((run) => run.scenario_id === current?.scenario_id && run.run_id !== current?.run_id);
  const previous = candidates.find((run) => run.run_id === baseline) ?? candidates[0];

  useEffect(() => {
    if (status === 'open' && !readOnly) void refresh();
  }, [status, refresh, readOnly]);

  if (readOnly) return null;

  return (
    <section className="testlab saved-runs" aria-labelledby="saved-runs-title">
      <div className="testlab__head">
        <div>
          <h2 id="saved-runs-title">Saved tests</h2>
          <p className="testlab__detail">Completed tests stay available after a restart. Latest saved first.</p>
        </div>
        <button type="button" onClick={() => void refresh()} disabled={loading || status !== 'open'}>
          {loading ? 'Loading…' : 'Refresh saved tests'}
        </button>
      </div>
      {runs.length === 0 ? (
        <p className="testlab__hint" role="status">No saved tests yet. Run a virtual test to record one.</p>
      ) : (
        <div className="saved-runs__controls">
          <label htmlFor="saved-test">Saved test
            <select id="saved-test" value={current?.run_id ?? ''} onChange={(event) => setSelected(event.currentTarget.value)}>
              {runs.map((run, index) => <option key={run.run_id} value={run.run_id}>
                Saved test {runs.length - index}: {run.title} · {run.setting_name} {run.setting}% · {run.peak_speed_kmh.toFixed(1)} km/h
              </option>)}
            </select>
          </label>
          <button type="button" disabled={running || status !== 'open' || !current?.run_id}
            onClick={() => current?.run_id && void replay(current.run_id)}>
            Replay saved test
          </button>
          <label htmlFor="saved-baseline">Compare with
            <select id="saved-baseline" value={previous?.run_id ?? ''} disabled={candidates.length === 0}
              onChange={(event) => setBaseline(event.currentTarget.value)}>
              {candidates.length === 0 ? <option value="">Save another run of this test</option> : null}
              {candidates.map((run) => <option key={run.run_id} value={run.run_id}>
                Saved test {runs.length - runs.indexOf(run)}: {run.setting_name} {run.setting}% · {run.peak_speed_kmh.toFixed(1)} km/h
              </option>)}
            </select>
          </label>
          <button type="button" disabled={!previous?.run_id || !current?.run_id || running}
            onClick={() => current?.run_id && previous?.run_id && compare(current.run_id, previous.run_id)}>
            Compare results
          </button>
        </div>
      )}
    </section>
  );
}
