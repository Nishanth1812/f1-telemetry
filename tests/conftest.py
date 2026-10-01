"""Shared fixtures. Nothing here simulates anything; records come from
:mod:`f1telemetry.testing.fixtures`, which supplies hand-computed values.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from f1telemetry.contracts.car_spec import CarSpec, car_spec_path, load_car_spec
from f1telemetry.contracts.channels import (
    ChannelContract,
    channels_yaml_path,
    load_channel_contract,
)
from f1telemetry.testing.fixtures import cornering_record, straight_line_record
from f1telemetry.testing.records import SampleRecord

# Numba's NRT allocation counters are the only thing that can see memory allocated inside
# compiled code, and `tests/test_longitudinal_kernel.py` uses them to show that the step loop
# allocates nothing. Numba reads the flag when its config module is first imported, so it has
# to be set here, before any test module can import numba.
os.environ["NUMBA_NRT_STATS"] = "1"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--golden-update",
        action="store_true",
        default=False,
        help="rewrite the committed golden baselines instead of comparing against them",
    )
    parser.addini(
        "golden_strict",
        "fail the golden stage when a channel diff exceeds its threshold",
        default=False,
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "golden: golden-trace regression comparison; reports rather than fails by default",
    )


@pytest.fixture(scope="session")
def golden_update(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--golden-update"))


@pytest.fixture(scope="session")
def golden_strict(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getini("golden_strict"))


@pytest.fixture(scope="session")
def repo() -> Path:
    return car_spec_path().parent


@pytest.fixture(scope="session")
def spec() -> CarSpec:
    return load_car_spec()


@pytest.fixture(scope="session")
def contract() -> ChannelContract:
    return load_channel_contract(channels_yaml_path())


@pytest.fixture
def straight() -> SampleRecord:
    return straight_line_record()


@pytest.fixture
def cornering() -> SampleRecord:
    return cornering_record()


@pytest.fixture(scope="session")
def report_dir(request: pytest.FixtureRequest) -> Iterator[Path]:
    root = Path(str(request.config.rootpath))
    target = root / "reports" / "golden"
    target.mkdir(parents=True, exist_ok=True)
    yield target
