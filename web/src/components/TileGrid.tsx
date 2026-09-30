import { useTelemetryStore } from '../store/telemetryStore';
import { TILE_REFRESH_HZ } from '../telemetry/types';
import { Tile } from './Tile';

const MAX_TILES = 64;

export function TileGrid() {
  const channelNames = useTelemetryStore((state) => state.channelNames);

  if (channelNames.length === 0) {
    return (
      <p className="placeholder" role="note">
        Waiting for telemetry frames on the display stream. Tiles refresh at {TILE_REFRESH_HZ} Hz.
      </p>
    );
  }

  const visible = channelNames.slice(0, MAX_TILES);

  return (
    <>
      <ul className="tile-grid" aria-label="Live channel values">
        {visible.map((name) => (
          <Tile key={name} name={name} />
        ))}
      </ul>
      {channelNames.length > visible.length ? (
        <p className="placeholder">
          Showing {visible.length} of {channelNames.length} channels.
        </p>
      ) : null}
    </>
  );
}
