import asyncio

import pytest

from f1telemetry.benchmark import percentile, run_benchmark


def test_percentiles_use_linear_interpolation() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([4.0], 99) == 4.0
    assert percentile([], 95) is None


@pytest.mark.parametrize("duration,clients", [(0, 1), (float("nan"), 1), (1, 0)])
def test_invalid_workload_is_rejected(duration: float, clients: int) -> None:
    with pytest.raises(ValueError):
        asyncio.run(run_benchmark(duration_s=duration, clients=clients))


def test_benchmark_measures_real_socket_delivery() -> None:
    report = asyncio.run(run_benchmark(duration_s=0.35, clients=2))
    assert report["clients"] == 2
    assert report["clock"] == "shared-process perf_counter_ns"
    measured = report["receivers"]
    assert isinstance(measured, list)
    assert len(measured) == 2
    for receiver in measured:
        assert receiver["frames"] >= 2
        assert receiver["bytes"] > 0
        assert receiver["bytes_per_second"] == pytest.approx(
            receiver["bytes"] / report["duration_s"]
        )
        assert receiver["latency_ms"]["p95"] >= 0
    assert report["cpu_s"] > 0
