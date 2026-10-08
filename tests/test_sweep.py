"""P6 exit: a parameterised sweep runs every grid point with its own manifest.

The sweep is a thin composition over :func:`run_scenario`: the grid primes a series of
override sets, each becomes one :class:`Scenario`, and each run keeps the manifest its
row cites. Byte-determinism of the underlying scenarios is what makes the whole table
reproducible from the same grid and the same factory.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from f1telemetry.contracts.car_spec import CarSpec, KernelConfig
from f1telemetry.testing import scenarios
from f1telemetry.testing.run_manifest import RunManifest
from f1telemetry.testing.sweep import (
    apply_sweep_overrides,
    results_table,
    run_sweep,
    write_results,
)

_SETUP_HASH = "ab" * 32
_GIT_SHA = "cd" * 20


@pytest.fixture(scope="module")
def config(spec: CarSpec) -> KernelConfig:
    return spec.kernel_config()


def _short_scenario() -> scenarios.Scenario:
    return scenarios.Scenario(
        name="sweep_short",
        initial_speed_m_s=12.0,
        description="Short rolling scenario for the sweep test.",
        segments=(
            scenarios.ScenarioSegment(0.3, throttle=0.5),
            scenarios.ScenarioSegment(0.3, throttle=0.2),
        ),
    )


def _grid() -> dict[str, list[object]]:
    return {
        "segments[0].throttle": [0.5, 1.0],
        "segments[1].throttle": [0.0, 0.2],
    }


def _factory(index: int, params: Mapping[str, object]) -> RunManifest:
    return RunManifest(
        seed=index,
        car_spec_version="test",
        scenario_version="1",
        setup_hash=_SETUP_HASH,
        git_sha=_GIT_SHA,
    )


def test_sweep_row_count_and_manifest_per_row(config: KernelConfig) -> None:
    rows = run_sweep(config, _short_scenario(), _grid(), _factory)

    assert len(rows) == 4
    assert [row.manifest.seed for row in rows if row.manifest is not None] == [0, 1, 2, 3]
    for row in rows:
        assert row.manifest is row.run.manifest or row.manifest == row.run.manifest
    table = results_table(rows)
    assert len(table) == 4
    for entry in table:
        assert set(entry) >= {
            "segments[0].throttle",
            "segments[1].throttle",
            "t_100_kmh_s",
            "final_speed_m_s",
            "settled_ay_m_s2",
            "valid",
            "manifest",
        }
        assert isinstance(entry["valid"], bool)
        assert isinstance(entry["manifest"], str)


def test_sweep_is_deterministic(config: KernelConfig) -> None:
    first = results_table(run_sweep(config, _short_scenario(), _grid(), _factory))
    second = results_table(run_sweep(config, _short_scenario(), _grid(), _factory))

    assert first == second


def test_sweep_manifest_seed_rides_to_parquet_and_json(
    config: KernelConfig, tmp_path: Path
) -> None:
    rows = run_sweep(config, _short_scenario(), _grid(), _factory)
    as_json = tmp_path / "results.json"
    as_parquet = tmp_path / "results.parquet"

    write_results(rows, as_json)
    write_results(rows, as_parquet)

    decoded = json.loads(as_json.read_text(encoding="utf-8"))
    assert len(decoded) == 4
    assert {entry["manifest"] for entry in decoded} == {
        row.manifest.canonical() for row in rows if row.manifest is not None
    }
    import pyarrow.parquet as pq

    table = pq.read_table(as_parquet)
    assert table.num_rows == 4
    assert "manifest" in table.column_names


def test_apply_sweep_overrides_types_and_bounds(config: KernelConfig) -> None:
    base = _short_scenario()

    changed = apply_sweep_overrides(base, {"soc_mj": 1.25, "segments[1].throttle": 0.9})
    assert changed.soc_mj == 1.25
    assert changed.segments[1].throttle == 0.9
    assert base.soc_mj is None and base.segments[1].throttle == 0.2  # frozen base kept

    with pytest.raises(ValueError, match="outside 2 segment"):
        apply_sweep_overrides(base, {"segments[2].throttle": 0.5})
    with pytest.raises(ValueError, match="must be a number"):
        apply_sweep_overrides(base, {"segments[0].throttle": "wide open"})
    with pytest.raises(ValueError, match="supported override path"):
        apply_sweep_overrides(base, {"segments.throttle": 0.5})
