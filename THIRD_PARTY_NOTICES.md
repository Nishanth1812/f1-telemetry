# Third-party material

The root Apache-2.0 license covers original project code. It does not replace
licenses attached to third-party data or installed dependencies.

## TUMFTM track geometry

`tracks/source_data/TUMFTM/Catalunya.csv` and `Silverstone.csv` are adapted from
[TUMFTM/racetrack-database](https://github.com/TUMFTM/racetrack-database/tree/e59595d1f3573b30d1ded6a08984935b957688e0)
at revision `e59595d1f3573b30d1ded6a08984935b957688e0`. The associated
`Silverstone.yaml` is derived geometry. Retain the upstream
[LGPL-3.0 license](tracks/source_data/TUMFTM/LICENSE) and its incorporated
[GPL-3.0 terms](tracks/source_data/TUMFTM/GPL-3.0.txt) with redistributed copies.

The upstream project describes centerlines fetched from OpenStreetMap and smoothed,
with track widths extracted from satellite imagery. Accuracy varies by circuit;
these files are not precision surveys. OpenStreetMap contributors are credited by
the upstream source. See [the track source notes](tracks/source_data/README.md).

## Installed dependencies

Python and JavaScript dependencies retain their respective licenses. Locked
versions are recorded in `uv.lock` and `web/package-lock.json`; distribution
must retain notices required by those packages. No dependency's code is relicensed
by the root license.
