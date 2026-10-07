"""tests/lang: run from the repo root (`python -m pytest tests/lang -m "not sim and not gpu"`).

Markers: `sim` (needs SAPIEN), `gpu` (needs a GPU). Simulator and GPU tests are
parametrized over EMBODIMENTS.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import main.lang  # noqa: E402,F401  (main.__path__ bootstrap for main.item_envs)

EMBODIMENTS = ["ur5-wsg", "franka-panda"]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def pytest_configure(config):
    config.addinivalue_line("markers", "sim: tests that need the SAPIEN simulator")
    config.addinivalue_line("markers", "gpu: tests that need a GPU (HoloBrain encoder, cuRobo)")


@pytest.fixture
def fixtures_dir():
    return FIXTURES
