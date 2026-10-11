import { useTelemetryStore } from '../store/telemetryStore';
import { TILE_REFRESH_HZ } from '../telemetry/types';
import { Tile } from './Tile';

export function TileGrid() {
  const channelNames = useTelemetryStore((state) => state.channelNames);

  if (channelNames.length === 0) {
    return (
      <p className="placeholder" role="note">
        No live data yet. Once the simulator is connected, every sensor appears here and refreshes{' '}
        {TILE_REFRESH_HZ} times a second.
      </p>
    );
  }

  return (
    <ul className="tile-grid" aria-label="Live channel values">
      {channelNames.map((name) => (
        <Tile key={name} name={name} />
      ))}
    </ul>
  );
}
