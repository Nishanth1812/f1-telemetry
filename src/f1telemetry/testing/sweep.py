"""Parameterised scenario sweeps (P6 exit).

One scenario definition plus a grid of field paths and value lists produces one
:class:`~f1telemetry.testing.scenarios.Scenario` per grid point. Each point runs
through :func:`~f1telemetry.testing.scenarios.run_scenario` with its own caller-supplied
:class:`~f1telemetry.testing.run_manifest.RunManifest` (the seed varies per point), and
the results come back as a compact table: the grid parameters, the run's key metrics
(0-100 km/h time, final speed, settled lateral acceleration, Layer 0 validity), and the
manifest citation for that row.

Nothing here tunes the model. The base scenario comes from the caller (a name in
:func:`~f1telemetry.testing.scenarios.build_scenarios` or a parsed YAML document), and a
grid only replaces declared values on it; every coefficient still comes from
``car_spec.yaml``.

The YAML document is :func:`~f1telemetry.testing.scenarios.load_scenario_init`-shaped -
``ParameterValueDeclarations`` and ``Init`` - with one addition: a ``Segments`` list, one
mapping per :class:`~f1telemetry.testing.scenarios.ScenarioSegment`, each scalar allowed
to be a ``$name`` reference into the declarations. Grid overrides use dotted field paths
with optional segment indexing, e.g. ``soc_mj``, ``initial_speed_m_s`` and
``segments[0].throttle``.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import os
import re
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from f1telemetry.analytics.validity import check_validity
from f1telemetry.contracts.car_spec import KernelConfig, load_car_spec
from f1telemetry.physics import gearbox  # noqa: TID251 -- scenarios drive the gearbox enum
from f1telemetry.testing import scenarios
from f1telemetry.testing.run_manifest import RunManifest
from f1telemetry.testing.scenarios import (
    CONTROL_STEPS,
    Scenario,
    ScenarioRun,
    ScenarioSegment,
)

__all__ = [
    "SweepRow",
    "apply_sweep_overrides",
    "load_sweep_scenario",
    "main",
    "results_table",
    "run_sweep",
    "write_results",
]

# Zero-to-100 km/h crossing threshold, in m/s.
_HUNDRED_KM_H_M_S: Final[float] = 100.0 / 3.6
# Settled lateral acceleration is the mean over this window at the end of the run.
_SETTLED_WINDOW_S: Final[float] = 0.2

_ManifestFactory = Callable[[int, Mapping[str, object]], RunManifest | None]

# Fields a grid override may replace on either object.
_SCENARIO_SCALARS: Final[frozenset[str]] = frozenset(
    {
        "name",
        "description",
        "initial_speed_m_s",
        "initial_gear",
        "soc_mj",
        "tyre_leak_rate_kg_s",
        "initial_x_m",
        "initial_y_m",
        "initial_heading_rad",
        "upshift_at_shift_point",
    }
)
_SEGMENT_SCALARS: Final[frozenset[str]] = frozenset(
    {
        "duration_s",
        "throttle",
        "clutch",
        "request",
        "mgu_k_request_nm",
        "brake_torque_nm",
        "grid_standing_start",
        "overtake",
        "ice_rpm_initial",
        "steer_wheel_deg",
    }
)

_PATH_PART: Final[re.Pattern[str]] = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?")


@dataclass(frozen=True, slots=True)
class SweepRow:
    """One grid point: its parameters, the run they produced, and its manifest."""

    params: Mapping[str, object]
    run: ScenarioRun
    manifest: RunManifest | None


def _split_path(path: str) -> tuple[tuple[str, int | None], ...]:
    """``segments[0].throttle`` -> ``(("segments", 0), ("throttle", None))``."""
    parts: list[tuple[str, int | None]] = []
    for piece in path.split("."):
        match = _PATH_PART.fullmatch(piece)
        if match is None:
            raise ValueError(f"grid field path {path!r} must be dotted names with [i]")
        name, index = match.group(1), match.group(2)
        parts.append((name, int(index) if index is not None else None))
    return tuple(parts)


def _coerce(value: object, field_name: str, current: object) -> object:
    """Type a grid value against the field's current value, like the YAML loader does."""
    if isinstance(current, gearbox.GearRequest):
        if isinstance(value, gearbox.GearRequest):
            return value
        if isinstance(value, str) and not isinstance(value, bool):
            try:
                return gearbox.GearRequest[value.upper()]
            except KeyError:
                valid = ", ".join(member.name for member in gearbox.GearRequest)
                raise ValueError(f"{field_name} must name one of {valid}, got {value!r}") from None
        raise ValueError(f"{field_name} must be a GearRequest name, got {value!r}")
    if isinstance(current, bool):
        if isinstance(value, bool):
            return value
        raise ValueError(f"{field_name} must be a bool, got {value!r}")
    if isinstance(current, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field_name} must be an int, got {value!r}")
        return value
    if isinstance(current, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field_name} must be a number, got {value!r}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{field_name} must be finite, got {value!r}")
        return number
    if isinstance(current, str):
        if isinstance(value, str):
            return value
        raise ValueError(f"{field_name} must be a string, got {value!r}")
    # The nullable fields (soc_mj, ice_rpm_initial) sit at None when unset; accept a
    # number or null for those.
    if current is None:
        if value is None:
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{field_name} must be finite, got {value!r}")
            return number
        raise ValueError(f"{field_name} must be a number or null, got {value!r}")
    raise ValueError(f"{field_name} cannot take override {value!r}")


def apply_sweep_overrides(base: Scenario, overrides: Mapping[str, object]) -> Scenario:
    """One scenario per grid combination: replace declared scalars, keep the rest."""
    scenario_fields = {field.name for field in dataclasses.fields(Scenario)}
    segment_fields = {field.name for field in dataclasses.fields(ScenarioSegment)}
    scenario_kwargs: dict[str, Any] = {}
    segment_overrides: dict[int, dict[str, Any]] = {}
    for path, value in overrides.items():
        steps = _split_path(path)
        if len(steps) == 1 and steps[0][1] is None:
            name = steps[0][0]
            if name not in _SCENARIO_SCALARS or name not in scenario_fields:
                raise ValueError(f"{path!r} is not an overridable Scenario field")
            scenario_kwargs[name] = _coerce(value, path, getattr(base, name))
        elif len(steps) == 2 and steps[0][0] == "segments" and steps[0][1] is not None:
            index, name = steps[0][1], steps[1][0]
            if (
                name not in _SEGMENT_SCALARS
                or name not in segment_fields
                or steps[1][1] is not None
            ):
                raise ValueError(f"{path!r} is not an overridable segment scalar")
            if not 0 <= index < len(base.segments):
                raise ValueError(f"{path!r} is outside {len(base.segments)} segment(s)")
            segment_overrides.setdefault(index, {})[name] = _coerce(
                value, path, getattr(base.segments[index], name)
            )
        else:
            raise ValueError(
                f"{path!r} is not a supported override path; use a Scenario scalar "
                "like 'soc_mj' or a segment scalar like 'segments[0].throttle'"
            )
    segments = tuple(
        dataclasses.replace(segment, **segment_overrides.get(index, {}))
        for index, segment in enumerate(base.segments)
    )
    return dataclasses.replace(base, segments=segments, **scenario_kwargs)


def _grid_points(grid: Mapping[str, Sequence[object]]) -> Iterator[dict[str, object]]:
    if not grid:
        yield {}
        return
    names = tuple(grid)
    for values in itertools.product(*(grid[name] for name in names)):
        yield dict(zip(names, values, strict=True))


def _base_scenario(config: KernelConfig, scenario: str | Scenario) -> Scenario:
    if isinstance(scenario, str):
        suite = scenarios.build_scenarios(config)
        if scenario not in suite:
            msg = f"unknown scenario {scenario!r}; the suite has {', '.join(sorted(suite))}"
            raise KeyError(msg)
        return suite[scenario]
    return scenario


def _zero_to_hundred_s(run: ScenarioRun) -> float | None:
    speed = np.hypot(
        [step.vx_m_s for step in run.record.ground_truth],
        [step.vy_m_s for step in run.record.ground_truth],
    )
    crossed = np.nonzero(speed >= _HUNDRED_KM_H_M_S)[0]
    if crossed.size == 0:
        return None
    return float(run.record.ground_truth[int(crossed[0])].t_s)


def _final_speed_m_s(run: ScenarioRun) -> float:
    last = run.record.ground_truth[-1]
    return math.hypot(last.vx_m_s, last.vy_m_s)


def _settled_ay_m_s2(run: ScenarioRun) -> float:
    steps = run.record.ground_truth
    window = max(1, round(_SETTLED_WINDOW_S / run.record.dt_s))
    tail = steps[-window:]
    return math.fsum(step.ay_m_s2 for step in tail) / len(tail)


def _valid(run: ScenarioRun) -> bool:
    return check_validity(run.record.frames) == ()


def run_sweep(
    config: KernelConfig,
    scenario: str | Scenario,
    grid: Mapping[str, Sequence[object]],
    manifest_factory: _ManifestFactory,
    *,
    control_steps: int = CONTROL_STEPS,
) -> tuple[SweepRow, ...]:
    """Build, run and score one :class:`Scenario` per grid combination.

    ``manifest_factory(index, params)`` supplies the citation for each run - a seed that
    varies per point is the caller's choice, made here so it cannot be forgotten. The same
    grid with the same factory produces the same rows: the runs are deterministic and the
    manifests are caller-owned inputs.
    """
    base = _base_scenario(config, scenario)
    for path, values in grid.items():
        if not values:
            raise ValueError(f"grid field {path!r} has no values")
        apply_sweep_overrides(base, {path: values[0]})  # validate before running anything
    rows: list[SweepRow] = []
    for index, params in enumerate(_grid_points(grid)):
        plan = apply_sweep_overrides(base, params)
        manifest = manifest_factory(index, params)
        run = scenarios.run_scenario(config, plan, control_steps=control_steps, manifest=manifest)
        rows.append(SweepRow(params=dict(params), run=run, manifest=manifest))
    return tuple(rows)


def results_table(rows: Sequence[SweepRow]) -> list[dict[str, object]]:
    """Compact rows: grid params, key metrics, validity, manifest citation."""
    table: list[dict[str, object]] = []
    for row in rows:
        entry: dict[str, object] = dict(row.params)
        entry["t_100_kmh_s"] = _zero_to_hundred_s(row.run)
        entry["final_speed_m_s"] = _final_speed_m_s(row.run)
        entry["settled_ay_m_s2"] = _settled_ay_m_s2(row.run)
        entry["valid"] = _valid(row.run)
        entry["manifest"] = None if row.manifest is None else row.manifest.canonical()
        table.append(entry)
    return table


def write_results(rows: Sequence[SweepRow], path: Path) -> Path:
    """Write the results table as JSON or Parquet, chosen by the suffix."""
    table = results_table(rows)
    suffix = path.suffix.lower()
    if suffix == ".json":
        path.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    elif suffix == ".parquet":
        keys: list[str] = []
        for entry in table:
            for key in entry:
                if key not in keys:
                    keys.append(key)
        fields = [pa.field(key, _arrow_type_for(key, table)) for key in keys]
        schema = pa.schema(fields)
        arrays = [
            pa.array([entry.get(field.name) for entry in table], type=field.type)
            for field in fields
        ]
        pq.write_table(
            pa.Table.from_arrays(arrays, schema=schema),
            path,
            compression="zstd",
            compression_level=3,
            version="2.6",
        )
    else:
        raise ValueError(f"--out must end in .json or .parquet, got {path}")
    return path


_FLOAT_COLUMNS: Final[frozenset[str]] = frozenset(
    {"t_100_kmh_s", "final_speed_m_s", "settled_ay_m_s2"}
)


def _arrow_type_for(key: str, table: Sequence[Mapping[str, object]]) -> pa.DataType:
    if key == "valid":
        return pa.bool_()
    if key in _FLOAT_COLUMNS:
        return pa.float64()
    present = [entry[key] for entry in table if entry.get(key) is not None]
    if not present:
        return pa.string()
    first = present[0]
    if isinstance(first, bool):
        return pa.bool_()
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in present):
        return pa.float64()
    return pa.string()


def _parse_scalar(text: str) -> object:
    lowered = text.lower()
    if lowered in ("null", "none", "~"):
        return None
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(text, base=10)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _parse_grid(specs: Sequence[str]) -> dict[str, list[object]]:
    grid: dict[str, list[object]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--grid expects path=v1,v2, got {spec!r}")
        path, _, raw = spec.partition("=")
        path = path.strip()
        if not path:
            raise ValueError(f"--grid expects a field path before '=', got {spec!r}")
        values = [_parse_scalar(piece.strip()) for piece in raw.split(",")]
        if path in grid:
            grid[path].extend(values)
        else:
            grid[path] = values
    return grid


class _SweepLoader(yaml.SafeLoader):
    """Safe loader that refuses YAML merge keys, like the Init loader."""


def _reject_merge(loader: _SweepLoader, node: yaml.nodes.MappingNode, deep: bool = False) -> object:
    for key_node, _ in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            raise ValueError("YAML merge keys are not supported")
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_SweepLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _reject_merge)


def load_sweep_scenario(path: Path) -> Scenario:
    """Parse a scenario YAML: Init-style document plus a Segments list.

    Every segment scalar may be a literal or a ``$name`` reference into
    ``ParameterValueDeclarations``, resolved and range-checked the way
    :func:`~f1telemetry.testing.scenarios.load_scenario_init` resolves the Init scalars.
    Merge keys are refused.
    """
    source = Path(path)
    try:
        document = yaml.load(source.read_text(encoding="utf-8"), Loader=_SweepLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"{source}: document: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"{source}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise ValueError(f"{source}: document must be a mapping")
    unknown = set(document) - {"ParameterValueDeclarations", "Init", "Segments"}
    if unknown:
        raise ValueError(f"{source}: unknown key {sorted(unknown)[0]!r}")
    if "Init" not in document:
        raise ValueError(f"{source}: Init is required")
    if "Segments" not in document:
        raise ValueError(f"{source}: Segments is required")
    declarations = _declarations(document, source)
    init = _load_init_view(source, document)
    raw_segments = document["Segments"]
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError(f"{source}: Segments must be a non-empty list")
    template = ScenarioSegment(0.0)
    segments: list[ScenarioSegment] = []
    for index, entry in enumerate(raw_segments):
        if not isinstance(entry, Mapping):
            raise ValueError(f"{source}: Segments[{index}] must be a mapping")
        unknown_seg = set(entry) - _SEGMENT_SCALARS
        if unknown_seg:
            raise ValueError(f"{source}: Segments[{index}].{sorted(unknown_seg)[0]}: unknown field")
        if "duration_s" not in entry:
            raise ValueError(f"{source}: Segments[{index}].duration_s: required field is missing")
        fields: dict[str, Any] = {}
        for name, raw in entry.items():
            if isinstance(raw, str) and raw.startswith("$"):
                ref = raw[1:]
                if ref not in declarations:
                    raise ValueError(
                        f"{source}: Segments[{index}].{name}: unknown reference {raw!r}"
                    )
                raw = declarations[ref]
            fields[name] = _coerce(raw, f"Segments[{index}].{name}", getattr(template, name))
        try:
            segments.append(ScenarioSegment(**fields))
        except TypeError as exc:
            raise ValueError(f"{source}: Segments[{index}]: {exc}") from exc
    return Scenario(
        name=init.name,
        initial_speed_m_s=init.initial_speed_m_s,
        segments=tuple(segments),
        description=init.description,
        initial_gear=init.initial_gear,
        soc_mj=init.soc_mj,
        brake_bias=init.brake_bias,
        tyre_leak_rate_kg_s=init.tyre_leak_rate_kg_s,
        initial_x_m=init.initial_x_m,
        initial_y_m=init.initial_y_m,
        initial_heading_rad=init.initial_heading_rad,
    )


def _declarations(document: Mapping[str, Any], source: Path) -> dict[str, object]:
    raw = document.get("ParameterValueDeclarations", [])
    if not isinstance(raw, list):
        raise ValueError(f"{source}: ParameterValueDeclarations must be a list")
    out: dict[str, object] = {}
    for entry in raw:
        if not isinstance(entry, Mapping) or set(entry) != {"name", "value"}:
            raise ValueError(f"{source}: ParameterValueDeclarations entries need name and value")
        out[str(entry["name"])] = entry["value"]
    return out


def _load_init_view(source: Path, document: Mapping[str, Any]) -> scenarios.ScenarioInit:
    """Type-check the Init portion through the real loader, via a temp document."""
    view = {
        "ParameterValueDeclarations": document.get("ParameterValueDeclarations", []),
        "Init": document["Init"],
    }
    handle, name = tempfile.mkstemp(prefix="sweep-init-", suffix=".yaml")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            yaml.safe_dump(view, stream)
        # The temp copy drops the Segments key, which is the only thing the real
        # loader forbids, and the error message names the temp path's keys, so report
        # against the caller's path by wrapping: the message already names the field.
        try:
            return scenarios.load_scenario_init(Path(name))
        except ValueError as exc:
            # The temp path is junk to the caller; report against theirs. The first
            # ": " in the message follows the temp path, never a Windows drive letter.
            detail = str(exc).split(": ", 1)[1] if ": " in str(exc) else str(exc)
            raise ValueError(f"{source}: {detail}") from exc
    finally:
        Path(name).unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="f1-sweep",
        description="Run a parameterised scenario sweep and write the results table.",
    )
    parser.add_argument("scenario", type=Path, help="scenario YAML (Init + Segments)")
    parser.add_argument(
        "--grid",
        action="append",
        default=[],
        metavar="PATH=V1,V2",
        help="one grid axis; repeat for a full grid (e.g. segments[0].throttle=0.5,1.0)",
    )
    parser.add_argument("--out", type=Path, required=True, help="results .json or .parquet")
    parser.add_argument("--seed", type=int, default=0, help="first manifest seed; +1 per row")
    parser.add_argument("--car-spec-version", default="unversioned")
    parser.add_argument("--scenario-version", default="1")
    parser.add_argument("--setup-hash", default="0" * 64)
    parser.add_argument("--git-sha", default="0" * 40)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    config = load_car_spec().kernel_config()
    base = load_sweep_scenario(args.scenario)
    grid = _parse_grid(args.grid)

    def factory(index: int, params: Mapping[str, object]) -> RunManifest:
        return RunManifest(
            seed=args.seed + index,
            car_spec_version=args.car_spec_version,
            scenario_version=args.scenario_version,
            setup_hash=args.setup_hash,
            git_sha=args.git_sha,
        )

    rows = run_sweep(config, base, grid, factory)
    write_results(rows, args.out)
    print(f"f1-sweep: {len(rows)} run(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
