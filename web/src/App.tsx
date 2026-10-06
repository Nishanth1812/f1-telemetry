import { ConnectionIndicator } from './components/ConnectionIndicator';
import { EventPanel } from './components/EventPanel';
import { TileGrid } from './components/TileGrid';
import { TracePanel } from './components/TracePanel';
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
        <TracePanel />
        <EventPanel />
      </main>
    </div>
  );
}
