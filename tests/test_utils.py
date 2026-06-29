# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for utility functions."""

import numpy as np
import pandas as pd

from causalts.utils.helpers import evaluate_graph
from causalts.utils.linearity import check_linearity


def test_evaluate_graph_perfect_match():
    G = np.zeros((3, 3, 2), dtype=np.int8)
    G[0, 1, 1] = 1
    G[1, 2, 1] = 1
    metrics = evaluate_graph(G, G)
    assert metrics["F1"] == 1.0
    assert metrics["SHD"] == 0
    assert metrics["Precision"] == 1.0
    assert metrics["TPR"] == 1.0


def test_evaluate_graph_empty_vs_nonempty():
    G_true = np.zeros((3, 3, 2), dtype=np.int8)
    G_true[0, 1, 1] = 1
    G_est = np.zeros((3, 3, 2), dtype=np.int8)
    metrics = evaluate_graph(G_est, G_true)
    assert metrics["TPR"] == 0.0
    assert metrics["SHD"] > 0


def test_evaluate_graph_all_false_positives():
    G_true = np.zeros((3, 3, 2), dtype=np.int8)
    G_est = np.ones((3, 3, 2), dtype=np.int8)
    metrics = evaluate_graph(G_est, G_true)
    assert metrics["Precision"] < 0.5
    assert metrics["SHD"] > 0


def test_check_linearity_on_linear_data():
    rng = np.random.default_rng(42)
    T = 300
    x = rng.standard_normal(T)
    y = 0.8 * x + 0.2 * rng.standard_normal(T)
    df = pd.DataFrame({"X": x, "Y": y})
    result = check_linearity(df, alpha=0.05)
    assert isinstance(result, dict)
    assert "fraction_nonlinear" in result
    assert result["fraction_nonlinear"] == 0.0


def test_check_linearity_on_nonlinear_data():
    rng = np.random.default_rng(42)
    T = 200
    x = rng.standard_normal(T)
    y = np.sin(3 * x) + 0.1 * rng.standard_normal(T)
    df = pd.DataFrame({"X": x, "Y": y})
    result = check_linearity(df, alpha=0.05)
    assert isinstance(result, dict)
    assert "fraction_nonlinear" in result
    assert result["fraction_nonlinear"] > 0
