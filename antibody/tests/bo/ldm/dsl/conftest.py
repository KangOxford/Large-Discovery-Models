"""Shared pytest fixtures for bo/ldm/dsl tests."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)