from __future__ import annotations

import logging

import pytest

from metabench.core import registry as reg
from metabench.core.interfaces import StandardizedData
from metabench.datasets.toy import ToyDataset
from metabench.tasks.steady_state_flux import SteadyStateFlux

reg.load_plugins()
logging.getLogger("cobra").setLevel(logging.WARNING)


@pytest.fixture
def toy_data() -> StandardizedData:
    data = ToyDataset().load()
    data.validate()
    return data


@pytest.fixture
def task() -> SteadyStateFlux:
    return SteadyStateFlux()
