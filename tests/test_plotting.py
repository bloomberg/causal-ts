# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Smoke tests for plotting (non-interactive Agg backend)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np

from causalts.utils.helpers import evaluate_graph


def _make_graph():
    g = np.zeros((3, 3, 2), dtype=np.int8)
    g[0, 1, 1] = 1
    g[1, 2, 1] = 1
    return g


def test_plot_graph_smoke():
    from causalts.plotting import plot_graph

    g = _make_graph()
    plot_graph(g, var_names=["X", "Y", "Z"])


def test_plot_time_series_graph_smoke():
    from causalts.plotting import plot_time_series_graph

    g = _make_graph()
    plot_time_series_graph(g, var_names=["X", "Y", "Z"])


def test_plot_graph_edge_color():
    from causalts.plotting import plot_graph

    g = _make_graph()
    plot_graph(g, var_names=["X", "Y", "Z"], edge_color="steelblue")


def test_plot_graph_target_node():
    from causalts.plotting import plot_graph

    g = _make_graph()
    plot_graph(
        g,
        var_names=["X", "Y", "Z"],
        target_node="Y",
        target_in_color="blue",
        target_out_color="red",
    )


def test_plot_graph_target_node_with_val_matrix():
    from causalts.plotting import plot_graph

    g = _make_graph()
    v = np.random.default_rng(42).standard_normal(g.shape)
    plot_graph(g, val_matrix=v, var_names=["X", "Y", "Z"], target_node="Z")


def test_plot_graph_multi_target():
    from causalts.plotting import plot_graph

    g = _make_graph()
    plot_graph(
        g,
        var_names=["X", "Y", "Z"],
        target_node=["X", "Z"],
        target_node_color=["#4CAF50", "#E91E63"],
        target_in_color=["#2196F3", "#00BCD4"],
        target_out_color=["#FF9800", "#FFC107"],
        target_between_color="#9C27B0",
    )


def test_extract_subgraph_depth1():
    from causalts.utils.graph import extract_subgraph

    g = np.zeros((4, 4, 2), dtype=np.int8)
    g[0, 1, 1] = 1
    g[1, 2, 1] = 1
    g[2, 3, 1] = 1
    sub, indices, names = extract_subgraph(
        g, target="X1", depth=1, var_names=["X0", "X1", "X2", "X3"]
    )
    assert set(indices) == {0, 1, 2}
    assert names == ["X0", "X1", "X2"]
    assert sub.shape == (3, 3, 2)
    assert sub[0, 1, 1] == 1
    assert sub[1, 2, 1] == 1


def test_extract_subgraph_parents_only():
    from causalts.utils.graph import extract_subgraph

    g = np.zeros((4, 4, 2), dtype=np.int8)
    g[0, 2, 1] = 1
    g[1, 2, 1] = 1
    g[2, 3, 1] = 1
    sub, indices, _ = extract_subgraph(g, target=2, depth=1, direction="parents")
    assert set(indices) == {0, 1, 2}


def test_compare_graphs_smoke():
    from causalts.plotting import compare_graphs

    g = _make_graph()
    g2 = g.copy()
    g2[2, 0, 1] = 1
    compare_graphs(g, g2, var_names=["X", "Y", "Z"])
