"""Shared setup for the tests."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from causalpdm.pipeline import data, preprocess  # noqa: E402

# Tests use a small slice of the data so the suite stays quick.
TEST_NORMAL_RECORDINGS = 5
TEST_GRID_SECONDS = 2.5


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: takes more than a few seconds")


@pytest.fixture(scope="session")
def coolant():
    """The coolant subsystem, loaded once and shared by every test."""
    return data.load("coolant", normal_recordings=TEST_NORMAL_RECORDINGS)


@pytest.fixture(scope="session")
def coolant_tables(coolant):
    """Preprocessed tables for the coolant subsystem."""
    tables, labels, healthy = preprocess.build_tables(
        coolant.fault_runs, coolant.normal_recordings,
        coolant.expert_graph.variable_names, TEST_GRID_SECONDS, 20)
    return tables, labels, healthy
