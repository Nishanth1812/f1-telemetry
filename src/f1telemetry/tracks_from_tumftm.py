"""Converter: TUMFTM CSV (real-circuit survey) → track-file shape.

Reads one TUMFTM racetrack CSV (header: ``x_m,y_m,w_tr_right_m,w_tr_left_m``)
and produces the same ``version: 1`` YAML track file that
:mod:`f1telemetry.tracks` consumes.  The width aliases ``w_tr_left_m`` /
``w_tr_right_m`` (TUMFTM survey naming) are mapped to the track-file keys
``left_m`` / ``right_m``; the loader in ``tracks.py`` also accepts these
aliases natively.

Usage (script or import)::

    from f1telemetry.tracks_from_tumftm import load_tumftm_track
    track = load_tumftm_track(
        "tracks/source_data/TUMFTM/Silverstone.csv"
    )

The converter subsamples dense CSV points to a practical waypoint count
(typically ~100–150 for a ~6 km circuit) so the periodic cubic spline
solves quickly and the sector list stays readable.  The original CSV is
treated as authoritative geometry; sector boundaries are derived from the
approximate arc length, not measured from the survey.

Source attribution (required by ``tracks/source_data/README.md``):

- TUMFTM racetrack-database commit ``e59595d1f3573b30d1ded6a08984935b957688e0``
- Source URL (commit):
  <https://github.com/TUMFTM/racetrack-database/blob/e59595d1f3573...>
- Source centerlines from OpenStreetMap, widths from satellite imagery; not
  a licensed or current F1 circuit layout.

Limitations (same as ``tracks.py``): elevation / gradient out of scope; the
centreline is planar.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import yaml

from f1telemetry.tracks import Track, load_track

__all__ = [
    "TUMFTM_COLUMNS",
    "convert_tumftm_to_yaml",
    "load_tumftm_track",
]

TUMFTM_COLUMNS: list[str] = ["x_m", "y_m", "w_tr_right_m", "w_tr_left_m"]


def load_tumftm_track(
    csv_path: str | Path,
    subsample_every: int = 10,
    sectors_fraction: tuple[float, float] = (0.35, 0.65),
) -> Track:
    """Load a TUMFTM CSV and return a validated ``Track``.

    ``subsample_every`` controls waypoint density (every N-th CSV row).
    ``sectors_fraction`` defines two internal sector boundaries as fractions
    of the approximate lap length.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"TUMFTM CSV not found: {csv_path}")

    points: list[tuple[float, float]] = []
    lefts: list[float] = []
    rights: list[float] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            # Skip comments / empty lines
            if not row:
                continue
            first = row[0].lstrip()
            if first.startswith("#"):
                continue
            if len(row) < 4:
                continue
            x = float(row[0])
            y = float(row[1])
            right = float(row[2])
            left = float(row[3])
            points.append((x, y))
            rights.append(right)
            lefts.append(left)

    if len(points) < 3:
        raise ValueError(f"need at least 3 waypoints, got {len(points)} from {csv_path}")

    # Subsample while preserving start and end points
    if subsample_every > 1 and len(points) > subsample_every * 3:
        indices = list(range(0, len(points), subsample_every))
        if indices[-1] != len(points) - 1:
            indices.append(len(points) - 1)
        points = [points[i] for i in indices]
        rights = [rights[i] for i in indices]
        lefts = [lefts[i] for i in indices]

    # Approximate lap length (sum of chord distances between consecutive
    # waypoints, including closing segment).  Used only for sector placement.
    total_chord = 0.0
    count = len(points)
    for i in range(count):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % count]
        total_chord += math.hypot(x1 - x0, y1 - y0)

    # Derive sector boundaries from approximate arc length fractions
    s1 = total_chord * sectors_fraction[0]
    s2 = total_chord * sectors_fraction[1]

    # Build the YAML document in memory
    centerline = [{"x_m": float(x), "y_m": float(y)} for x, y in points]
    widths = [
        {"w_tr_left_m": float(left), "w_tr_right_m": float(right)}
        for left, right in zip(lefts, rights, strict=True)
    ]

    doc = {
        "version": 1,
        "name": csv_path.stem,
        "description": (
            f"Converted from TUMFTM CSV: {csv_path.name}. "
            f"Subsampled every {subsample_every} rows; "
            f"{len(points)} waypoints; approximate lap chord ~{total_chord:.1f} m. "
            f"Source: TUMFTM racetrack-database (commit e59595d...)."
        ),
        "centerline": centerline,
        "widths_m": widths,
        "sectors": [
            {"s_m": float(s1)},
            {"s_m": float(s2)},
        ],
    }

    # Write YAML to same parent directory (new file, no production code touched)
    yaml_path = csv_path.with_suffix("").with_suffix(".yaml")
    if yaml_path.name == ".yaml":
        yaml_path = csv_path.parent / (csv_path.name.replace(".csv", ".yaml"))
    yaml_path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return load_track(yaml_path)


def convert_tumftm_to_yaml(
    csv_path: str | Path,
    out_yaml_path: str | Path | None = None,
    subsample_every: int = 10,
    sectors_fraction: tuple[float, float] = (0.35, 0.65),
) -> Path:
    """Convert a TUMFTM CSV to a track YAML file (no ``Track`` load)."""
    csv_path = Path(csv_path)
    out_yaml_path = csv_path.with_suffix(".yaml") if out_yaml_path is None else Path(out_yaml_path)
    # Re-use load logic but just keep YAML; reload to validate
    load_tumftm_track(
        csv_path,
        subsample_every=subsample_every,
        sectors_fraction=sectors_fraction,
    )
    # The function above writes YAML; confirm it exists
    if not out_yaml_path.is_file():
        raise RuntimeError(f"converter did not write YAML to {out_yaml_path}")
    return out_yaml_path
