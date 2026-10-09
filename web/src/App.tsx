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

export function App() {
  useTelemetrySocket();
  return (
    <div className="app">
      <header className="app__header">
        <h1 className="app__title">F1 Telemetry</h1>
        <ConnectionIndicator />
      </header>
      <main className="app__main">
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
      </main>
    </div>
  );
}
