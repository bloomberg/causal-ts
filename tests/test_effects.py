# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for DoWhy-based effect estimation (skipped if dowhy not installed)."""

import numpy as np
import pandas as pd
import pytest

dowhy = pytest.importorskip("dowhy")

from causalts.effects.graph_bridge import (  # noqa: E402
    graph_to_networkx,
    make_lagged_df,
)


def _simple_graph_and_data():
    """X0(t-1) -> X1(t), X1(t-1) -> X2(t)."""
    rng = np.random.default_rng(42)
    T = 100
    x0 = rng.standard_normal(T)
    x1 = np.zeros(T)
    x2 = np.zeros(T)
    for t in range(1, T):
        x1[t] = 0.7 * x0[t - 1] + 0.2 * rng.standard_normal()
        x2[t] = 0.5 * x1[t - 1] + 0.2 * rng.standard_normal()
    df = pd.DataFrame({"X0": x0, "X1": x1, "X2": x2})

    graph = np.zeros((3, 3, 2), dtype=np.int8)
    graph[0, 1, 1] = 1  # X0(t-1) -> X1(t)
    graph[1, 2, 1] = 1  # X1(t-1) -> X2(t)
    return graph, df


def test_graph_to_networkx():
    graph, df = _simple_graph_and_data()
    G_nx = graph_to_networkx(graph, var_names=list(df.columns))
    assert "X0_lag1" in G_nx.nodes or "X0_lag1" in str(G_nx.nodes)
    assert len(G_nx.edges) >= 2


def test_make_lagged_df():
    graph, df = _simple_graph_and_data()
    lagged_df = make_lagged_df(df, max_lag=1)
    assert "X0_lag1" in lagged_df.columns
    assert "X1" in lagged_df.columns
    assert lagged_df.shape[0] == df.shape[0] - 1


def test_estimate_effect():
    from causalts.effects.effect import estimate_effect

    graph, df = _simple_graph_and_data()
    result = estimate_effect(
        graph,
        df,
        treatment="X0",
        outcome="X1",
        treatment_lag=1,
        n_bootstraps=10,
    )
    assert hasattr(result, "value")
    assert isinstance(result.value, float)
    assert abs(result.value - 0.7) < 0.3


def test_fit_scm_and_counterfactual():
    from causalts.effects.scm import counterfactual, fit_scm

    graph, df = _simple_graph_and_data()
    scm, dag, lagged_df = fit_scm(graph, df, mechanism_type="linear")
    assert scm is not None

    result = counterfactual(scm, lagged_df, {"X0_lag1": 0.0}, target="X1")
    import pandas as pd

    assert isinstance(result, (pd.Series, float, np.floating))
