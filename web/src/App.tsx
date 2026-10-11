import { useState } from 'react';
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
import { useTelemetrySocket } from './hooks/useTelemetrySocket';
import { ScenarioLab } from './components/ScenarioLab';
import { setDetailedTelemetry } from './telemetry/socket';

export function App() {
  useTelemetrySocket();
  const [advancedOpen, setAdvancedOpen] = useState(false);
  return (
    <div className="app">
      <header className="app__header">
        <div>
          <h1 className="app__title">F1 Telemetry</h1>
          <p className="app__subtitle">
            Try a simulated race car: pick a run, change one setting, see what happens.
          </p>
        </div>
        <ConnectionIndicator />
      </header>
      <main className="app__main">
        <ScenarioLab />
        <details
          className="advanced-telemetry"
          onToggle={(event) => {
            const open = event.currentTarget.open;
            setAdvancedOpen(open);
            setDetailedTelemetry(open);
          }}
        >
          <summary>Advanced: live car data (for engineers)</summary>
          <p className="advanced-telemetry__note">
            These panels stream every sensor the simulator produces, about 30 times a second. You
            don&apos;t need them for the Scenario Lab above.
          </p>
          {advancedOpen ? (
            <div className="advanced-telemetry__content">
              <section className="app__tiles" aria-labelledby="tiles-heading">
                <h2 className="visually-hidden" id="tiles-heading">
                  Live channels
                </h2>
                <TileGrid />
              </section>
              <div className="app__blocks">
                <TyreBlock />
                <BrakeBlock />
                <AeroBlock />
              </div>
              <div className="app__blocks">
                <LapBar />
                <AnomalyPanel />
              </div>
              <TracePanel />
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
