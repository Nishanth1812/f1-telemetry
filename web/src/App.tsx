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
import { useTelemetryStore } from './store/telemetryStore';
import { useTelemetrySocket } from './hooks/useTelemetrySocket';
import { ScenarioLab } from './components/ScenarioLab';
import { setDetailedTelemetry } from './telemetry/socket';

function ConnectionStatusText() {
  const status = useTelemetryStore((state) => state.status);
  if (status === 'open') {
    return <p className="connection-status connection-status--open">Connected to the simulator</p>;
  }
  if (status === 'connecting' || status === 'reconnecting') {
    return <p className="connection-status connection-status--connecting">Connecting to the simulator…</p>;
  }
  return <p className="connection-status connection-status--disconnected">Not connected - start the server</p>;
}

export function App() {
  useTelemetrySocket();
  const [advancedOpen, setAdvancedOpen] = useState(false);
  return (
    <div className="app">
      <header className="app__header">
        <div>
          <h1 className="app__title">F1 Telemetry</h1>
          <p className="app__subtitle">
            A simulated racing-car dashboard: run a quick experiment, change one setting,
            and see how speed and grip respond. Not calibrated to a real car.
          </p>
          <ConnectionStatusText />
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
          <summary>Advanced telemetry · raw channels and analysis</summary>
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
