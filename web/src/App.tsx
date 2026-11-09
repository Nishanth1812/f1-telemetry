import { useEffect, useState } from 'react';
import { AeroBlock } from './components/AeroBlock';
import { AnomalyPanel } from './components/AnomalyPanel';
import { BrakeBlock } from './components/BrakeBlock';
import { ConnectionIndicator } from './components/ConnectionIndicator';
import { EventPanel } from './components/EventPanel';
import { LapBar } from './components/LapBar';
import { TileGrid } from './components/TileGrid';
import { TracePanel } from './components/TracePanel';
import { RunComparison } from './components/RunComparison';
import { Scrubber } from './components/Scrubber';
import { TyreBlock } from './components/TyreBlock';
import { useTelemetryStore } from './store/telemetryStore';
import { useTelemetrySocket } from './hooks/useTelemetrySocket';
import { HealthPanel } from './components/HealthPanel';
import { Instruments } from './components/Instruments';
import { TestControls } from './components/TestControls';
import { TestResult } from './components/TestResult';
import { SavedRuns } from './components/SavedRuns';
import { setDetailedTelemetry } from './telemetry/socket';

export function App() {
  useTelemetrySocket();
  const [rawOpen, setRawOpen] = useState(false);
  const status = useTelemetryStore((state) => state.status);

  // The charts, tyre and brake panels draw from the full-rate batches, so ask for them
  // whenever the dashboard is connected (the server forgets the choice on reconnect).
  useEffect(() => {
    if (status === 'open') {
      setDetailedTelemetry(true);
    }
  }, [status]);

  return (
    <div className="app">
      <a className="skip-link" href="#dashboard">
        Skip to the dashboard
      </a>
      <header className="app__header">
        <div className="app__brand">
          <p className="app__eyebrow">Telemetry operations <span aria-hidden="true">/</span> Simulation</p>
          <h1 className="app__title">F1 Telemetry Lab</h1>
          <p className="app__subtitle">
            Engineering workstation · synthetic vehicle, not calibrated to a real car
          </p>
        </div>
        <ConnectionIndicator />
      </header>
      <main className="app__main" id="dashboard">
        <div className="workstation">
          <aside className="workstation__controls" aria-label="Test operations">
            <TestControls />
            <SavedRuns />
          </aside>
          <div className="workstation__monitor">
            <Instruments />
            <section className="app__section workstation__trace" aria-labelledby="trace-heading">
              <div className="app__section-head">
                <h2 className="app__section-title" id="trace-heading">
                  Live trace
                </h2>
                <p className="app__section-hint">
                  Select a channel to inspect its recent samples.
                </p>
              </div>
              <TracePanel />
            </section>
          </div>
        </div>
        <section className="app__section" aria-labelledby="corners-heading">
          <div className="app__section-head">
            <h2 className="app__section-title" id="corners-heading">
              Tyres, brakes and aero
            </h2>
            <p className="app__section-hint">
              Every corner on its own, with thirty seconds of history behind the live number.
            </p>
          </div>
          <div className="app__blocks">
            <TyreBlock />
            <BrakeBlock />
            <AeroBlock />
          </div>
        </section>
        <section className="app__section" aria-labelledby="analysis-heading">
          <div className="app__section-head">
            <h2 className="app__section-title" id="analysis-heading">
              Car health
            </h2>
            <p className="app__section-hint">
              Each row is one rule applied to the live readings. A tripped rule stays flagged until
              the next test.
            </p>
          </div>
          <HealthPanel />
        </section>
        <TestResult />
        <details
          className="advanced-telemetry"
          onToggle={(event) => setRawOpen(event.currentTarget.open)}
        >
          <summary>
            Advanced: every sensor, laps and events{' '}
            <span className="advanced-telemetry__summary-note">(for engineers)</span>
          </summary>
          {rawOpen ? (
            <div className="advanced-telemetry__content">
              <p className="advanced-telemetry__note">
                Every sensor the simulated car publishes, with its latest value.
              </p>
              <section className="app__tiles" aria-labelledby="tiles-heading">
                <h2 className="visually-hidden" id="tiles-heading">
                  Live channels
                </h2>
                <TileGrid />
              </section>
              <div className="app__blocks">
                <LapBar />
              </div>
              <AnomalyPanel />
              <RunComparison />
              <Scrubber />
              <EventPanel />
            </div>
          ) : null}
        </details>
      </main>
    </div>
  );
}
