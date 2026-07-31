# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

import os

import numpy as np
import pandas as pd
import pytest

collect_ignore = [os.path.join(os.path.dirname(__file__), "test_sigkci_vs_sigchsic.py")]

DEVICE = "cpu"


@pytest.fixture
def independent_data():
    """X = f(Z) + noise, Y = g(Z) + noise (X _|_ Y | Z)."""

    def _make(n=100, seed=42):
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n, 1))
        x = np.sin(z[:, 0]) + 0.3 * rng.standard_normal(n)
        y = np.cos(z[:, 0]) + 0.3 * rng.standard_normal(n)
        return np.column_stack([x, y, z]).astype(np.float32)

    return _make


@pytest.fixture
def dependent_data():
    """X -> Y with Z as confounder (X not _|_ Y | Z)."""

    def _make(n=100, seed=42):
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n, 1))
        x = z[:, 0] + 0.3 * rng.standard_normal(n)
        y = 0.8 * x + 0.2 * rng.standard_normal(n)
        return np.column_stack([x, y, z]).astype(np.float32)

    return _make


@pytest.fixture
def small_df():
    """Small 3-variable DataFrame for algorithm smoke tests."""
    rng = np.random.default_rng(42)
    T = 80
    x0 = rng.standard_normal(T)
    x1 = np.zeros(T)
    x2 = np.zeros(T)
    for t in range(1, T):
        x1[t] = 0.7 * x0[t - 1] + 0.3 * rng.standard_normal()
        x2[t] = 0.5 * x1[t - 1] + 0.3 * rng.standard_normal()
    return pd.DataFrame({"X0": x0, "X1": x1, "X2": x2})
