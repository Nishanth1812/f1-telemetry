import { useTelemetryStore } from '../store/telemetryStore';
import { TILE_REFRESH_HZ } from '../telemetry/types';
import { Tile } from './Tile';

export function TileGrid() {
  const channelNames = useTelemetryStore((state) => state.channelNames);

  if (channelNames.length === 0) {
    return (
      <p className="placeholder" role="note">
        Waiting for telemetry frames on the display stream. Tiles refresh at {TILE_REFRESH_HZ} Hz.
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
