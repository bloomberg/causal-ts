# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Smoke tests for plotting (non-interactive Agg backend)."""

import matplotlib
import pytest

matplotlib.use("Agg")

import numpy as np  # noqa: E402


def test_graphviz_layout_uses_pydot_without_unsupported_args(monkeypatch):
    # Regression test for https://github.com/bloomberg/causal-ts/issues/53.
    import networkx as nx
    import networkx.drawing.nx_agraph as nx_agraph
    import networkx.drawing.nx_pydot as nx_pydot

    from causalts.plotting._core import compute_node_positions

    graph = nx.path_graph(3)
    expected = {node: np.array([float(node), float(node) + 1]) for node in graph}
    calls = []

    def unavailable(*args, **kwargs):
        raise ImportError("pygraphviz is not installed")

    def pydot_layout(graph, prog="neato", root=None):
        calls.append((prog, root))
        return expected

    monkeypatch.setattr(nx_agraph, "graphviz_layout", unavailable)
    monkeypatch.setattr(nx_pydot, "graphviz_layout", pydot_layout)

    positions = compute_node_positions(
        graph,
        layout="neato",
        layout_kwargs={"args": "-Goverlap=false"},
        normalize=False,
    )

    assert calls == [("neato", None)]
    assert all(np.array_equal(positions[node], expected[node]) for node in graph)


def test_graphviz_layout_warns_before_circular_fallback(monkeypatch):
    # Regression test for https://github.com/bloomberg/causal-ts/issues/53.
    import networkx as nx
    import networkx.drawing.nx_agraph as nx_agraph
    import networkx.drawing.nx_pydot as nx_pydot

    from causalts.plotting._core import compute_node_positions

    graph = nx.path_graph(3)

    def unavailable(*args, **kwargs):
        raise ImportError("Graphviz backend is unavailable")

    monkeypatch.setattr(nx_agraph, "graphviz_layout", unavailable)
    monkeypatch.setattr(nx_pydot, "graphviz_layout", unavailable)

    with pytest.warns(UserWarning, match="falling back to circular layout"):
        positions = compute_node_positions(graph, layout="neato", normalize=False)

    assert set(positions) == set(graph)


def _make_graph():
    g = np.zeros((3, 3, 2), dtype=np.int8)
    g[0, 1, 1] = 1
    g[1, 2, 1] = 1
    return g


def test_plot_graph_smoke():
    from causalts.plotting import plot_graph

    g = _make_graph()
    plot_graph(g, var_names=["X", "Y", "Z"])


def test_plot_graph_var_names_wrong_length():
    from causalts.plotting._core import plot_graph

    g = _make_graph()

    with pytest.raises(ValueError, match="var_names"):
        plot_graph(g, var_names=["X", "Y"])


def test_plot_time_series_graph_var_names_wrong_length():
    from causalts.plotting import plot_time_series_graph

    g = _make_graph()
    with pytest.raises(ValueError, match=r"graph\[:2, :2\]"):
        plot_time_series_graph(g, var_names=["X", "Y"])


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


@pytest.mark.parametrize("mirror", ["<--", "<-o"])
def test_extract_subgraph_string_graph_follows_lag0_direction(mirror):
    from causalts.utils.graph import extract_subgraph

    # 0 -> 1 at lag 0, stored with its mirror at [1, 0, 0]; 2 -> 0 at lag 1.
    g = np.full((3, 3, 2), "", dtype="<U3")
    g[0, 1, 0], g[1, 0, 0] = "-->", mirror
    g[2, 0, 1] = "-->"

    _, parents, _ = extract_subgraph(g, target=0, direction="parents")
    _, children, _ = extract_subgraph(g, target=0, direction="children")
    _, both, _ = extract_subgraph(g, target=0)
    assert parents == [0, 2]
    assert children == [0, 1]
    assert both == [0, 1, 2]

    _, parents_of_1, _ = extract_subgraph(g, target=1, direction="parents")
    _, children_of_1, _ = extract_subgraph(g, target=1, direction="children")
    assert parents_of_1 == [0, 1]
    assert children_of_1 == [1]


def test_compare_graphs_smoke():
    from causalts.plotting import compare_graphs

    g = _make_graph()
    g2 = g.copy()
    g2[2, 0, 1] = 1
    compare_graphs(g, g2, var_names=["X", "Y", "Z"])


def _marks(edges, d=3, n_lags=2):
    """Tigramite edge-mark graph: lag-0 links are written with their mirror."""
    marks = np.full((d, d, n_lags), "", dtype="<U3")
    for i, j, tau in edges:
        marks[i, j, tau] = "-->"
        if tau == 0:
            marks[j, i, 0] = "<--"
    return marks


def _mark_graph(marks, d=2, n_lags=1):
    """Tigramite edge-mark graph from explicit ``(i, j, tau, mark)`` entries."""
    graph = np.full((d, d, n_lags), "", dtype="<U3")
    for i, j, tau, mark in marks:
        graph[i, j, tau] = mark
    return graph


def _oo_pair(mark="o-o", d=2, n_lags=1):
    """A lag-0 symmetric pair, as PCMCI+ stores an unoriented adjacency."""
    return _mark_graph([(0, 1, 0, mark), (1, 0, 0, mark)], d=d, n_lags=n_lags)


def _visible_arrow_colors(ax):
    """Edge colours of visible arrows/lines, ignoring label midpoint markers."""
    from matplotlib.patches import FancyArrowPatch

    return [
        a.get_edgecolor()
        for a in ax.get_children()
        if isinstance(a, FancyArrowPatch) and a.get_edgecolor()[3] > 0
    ]


def test_plot_lag_metrics_string_graph_mirror_is_not_an_edge():
    from causalts.plotting import plot_lag_metrics

    edges = [(0, 1, 0), (2, 1, 1)]
    binary = np.zeros((3, 3, 2), dtype=np.int8)
    for i, j, tau in edges:
        binary[i, j, tau] = 1

    _, ax = plot_lag_metrics(binary, _marks(edges))
    # Rows: Precision, Recall, F1; columns: lag 0, lag 1.
    np.testing.assert_allclose(ax.images[0].get_array(), np.ones((3, 2)))


def test_compare_graphs_string_graph_reversed_lag0_edge_is_not_tp():
    from matplotlib.colors import to_rgba
    from matplotlib.patches import FancyArrowPatch

    from causalts.plotting import compare_graphs

    _, ax = compare_graphs(_marks([(0, 1, 0)]), _marks([(1, 0, 0)]), tp_color="#00ff00")
    arrows = [a for a in ax.get_children() if isinstance(a, FancyArrowPatch)]
    assert arrows
    assert not any(a.get_facecolor() == to_rgba("#00ff00") for a in arrows)


@pytest.mark.parametrize("mirror", ["<--", "<-o"])
def test_plot_lag_metrics_one_sided_lag0_mirror_is_reversed_edge(mirror):
    import matplotlib.pyplot as plt

    from causalts.plotting import plot_lag_metrics

    # Only the mirror half of 1 -> 0 is stored, at [0, 1, 0]: still the edge 1 -> 0.
    gt = np.zeros((2, 2, 1), dtype=np.int8)
    gt[1, 0, 0] = 1
    marks = np.full((2, 2, 1), "", dtype="<U3")
    marks[0, 1, 0] = mirror

    _, ax = plot_lag_metrics(gt, marks)
    np.testing.assert_allclose(ax.images[0].get_array(), np.ones((3, 1)))
    plt.close("all")


@pytest.mark.parametrize("n_lags", [1, 3])
@pytest.mark.parametrize("as_marks", [False, True])
def test_compare_functions_single_variable_graph(n_lags, as_marks):
    import matplotlib.pyplot as plt

    from causalts.plotting import (
        compare_graphs,
        plot_lag_metrics,
        plot_pvalue_distribution,
    )

    # A (1, 1, n_lags) graph must not be squeezed down to 1-D or 0-D.
    if as_marks:
        g = np.full((1, 1, n_lags), "", dtype="<U3")
        g[0, 0, n_lags - 1] = "-->"
    else:
        g = np.zeros((1, 1, n_lags), dtype=np.int8)
        g[0, 0, n_lags - 1] = 1
    pvals = np.full((1, 1, n_lags), 0.5)

    compare_graphs(g, g)
    _, ax = plot_lag_metrics(g, g)
    assert ax.images[0].get_array().shape == (3, n_lags)
    plot_pvalue_distribution(pvals, g, graph_discovered=g)
    plt.close("all")


@pytest.mark.parametrize("tau", [0, 1])
def test_plot_lag_metrics_latent_mark_is_not_an_edge(tau):
    import matplotlib.pyplot as plt

    from causalts.plotting import plot_lag_metrics

    # "<->" asserts there is no direct edge, so it is a false negative here,
    # not a true positive plus a false positive.
    gt = np.zeros((2, 2, 2), dtype=np.int8)
    gt[0, 1, tau] = 1
    disc = np.full((2, 2, 2), "", dtype="<U3")
    disc[0, 1, tau] = "<->"
    if tau == 0:
        disc[1, 0, 0] = "<->"

    _, ax = plot_lag_metrics(gt, disc, metrics=["TP", "FP", "FN"])
    np.testing.assert_array_equal(ax.images[0].get_array()[:, tau], [0, 0, 1])
    plt.close("all")


def test_compare_functions_read_object_dtype_marks():
    import matplotlib.pyplot as plt

    from causalts.plotting import compare_graphs, plot_lag_metrics

    # Marks loaded from a DataFrame or CSV arrive as object arrays.
    gt = np.full((2, 2, 1), "", dtype=object)
    gt[0, 1, 0], gt[1, 0, 0] = "-->", "<--"
    disc = np.full((2, 2, 1), "", dtype=object)
    disc[0, 1, 0] = disc[1, 0, 0] = "<->"
    _, ax = plot_lag_metrics(gt, disc, metrics=["TP", "FP", "FN"])
    np.testing.assert_array_equal(ax.images[0].get_array()[:, 0], [0, 0, 1])
    bad = gt.copy()
    bad[0, 1, 0] = bad[1, 0, 0] = "o-o"
    with pytest.raises(ValueError, match="must be a DAG"):
        compare_graphs(bad, disc)
    plt.close("all")


@pytest.mark.parametrize("mark", ["o-o", "x-x", "<->", "o->", "<-o", "->"])
def test_compare_functions_reject_non_dag_ground_truth(mark):
    from causalts.plotting import (
        compare_graphs,
        plot_lag_metrics,
        plot_pvalue_distribution,
    )

    gt = np.full((2, 2, 1), "", dtype="<U3")
    gt[0, 1, 0] = gt[1, 0, 0] = mark
    disc = np.zeros((2, 2, 1), dtype=np.int8)
    pvals = np.full((2, 2, 1), 0.5)

    with pytest.raises(ValueError, match="must be a DAG"):
        compare_graphs(gt, disc)
    with pytest.raises(ValueError, match="must be a DAG"):
        plot_lag_metrics(gt, disc)
    with pytest.raises(ValueError, match="must be a DAG"):
        plot_pvalue_distribution(pvals, gt)


@pytest.mark.parametrize(
    "policy, expected",
    [
        # Directed TP/FP/FN, then adjacency TP/FP/FN.
        ("bidirected", [1, 1, 0, 1, 0, 0]),
        ("drop", [0, 0, 1, 0, 0, 1]),
        # The pair is found but not oriented: directed FN, adjacency TP.
        ("adjacency", [0, 0, 1, 1, 0, 0]),
    ],
)
def test_plot_lag_metrics_oo_policies(policy, expected):
    import matplotlib.pyplot as plt

    from causalts.plotting import plot_lag_metrics

    gt = np.zeros((2, 2, 1), dtype=np.int8)
    gt[0, 1, 0] = 1

    _, ax = plot_lag_metrics(
        gt,
        _oo_pair(),
        metrics=[
            "TP",
            "FP",
            "FN",
            "Adjacency TP",
            "Adjacency FP",
            "Adjacency FN",
        ],
        undirected=policy,
    )
    np.testing.assert_array_equal(ax.images[0].get_array()[:, 0], expected)
    plt.close("all")


def test_plot_lag_metrics_adjacency_rows_are_added_by_default():
    import matplotlib.pyplot as plt

    from causalts.plotting import plot_lag_metrics

    gt = np.zeros((2, 2, 1), dtype=np.int8)
    gt[0, 1, 0] = 1

    _, ax = plot_lag_metrics(gt, _oo_pair(), undirected="adjacency")
    data = ax.images[0].get_array()
    # Precision, Recall, F1, Adjacency Precision, Adjacency Recall, Adjacency F1.
    assert data.shape == (6, 1)
    np.testing.assert_allclose(data[:3, 0], np.zeros(3))
    np.testing.assert_allclose(data[3:, 0], np.ones(3))
    plt.close("all")


def test_plot_lag_metrics_spurious_adjacency_is_fp():
    import matplotlib.pyplot as plt

    from causalts.plotting import plot_lag_metrics

    gt = np.zeros((2, 2, 1), dtype=np.int8)
    _, ax = plot_lag_metrics(
        gt,
        _oo_pair(),
        metrics=["Adjacency TP", "Adjacency FP", "Adjacency FN"],
        undirected="adjacency",
    )
    np.testing.assert_array_equal(ax.images[0].get_array()[:, 0], [0, 1, 0])
    plt.close("all")


@pytest.mark.parametrize(
    "mark, parameter", [("o-o", "undirected"), ("x-x", "conflict")]
)
def test_plot_lag_metrics_adjacency_policy_per_mark(mark, parameter):
    import matplotlib.pyplot as plt

    from causalts.plotting import plot_lag_metrics

    gt = np.zeros((2, 2, 1), dtype=np.int8)
    gt[0, 1, 0] = 1

    _, ax = plot_lag_metrics(
        gt,
        _oo_pair(mark),
        metrics=[
            "TP",
            "FP",
            "FN",
            "Adjacency TP",
            "Adjacency FP",
            "Adjacency FN",
        ],
        **{parameter: "adjacency"},
    )
    np.testing.assert_array_equal(ax.images[0].get_array()[:, 0], [0, 0, 1, 1, 0, 0])
    plt.close("all")


@pytest.mark.parametrize("policy", ["drop", "bidirected", "adjacency"])
def test_plot_lag_metrics_latent_mark_dropped_under_all_policies(policy):
    import matplotlib.pyplot as plt

    from causalts.plotting import plot_lag_metrics

    gt = np.zeros((2, 2, 1), dtype=np.int8)
    gt[0, 1, 0] = 1

    _, ax = plot_lag_metrics(
        gt,
        _oo_pair("<->"),
        metrics=["TP", "FP", "FN"],
        undirected=policy,
        conflict=policy,
    )
    np.testing.assert_array_equal(ax.images[0].get_array()[:, 0], [0, 0, 1])
    plt.close("all")


@pytest.mark.parametrize("true_cell", [(0, 1), (1, 0)])
def test_plot_lag_metrics_lagged_oo_is_scored_at_its_own_cell(true_cell):
    import matplotlib.pyplot as plt

    from causalts.plotting import plot_lag_metrics

    # At lag >= 1 the reverse cell is a different temporal link, so an
    # unoriented o-o at [0, 1, 1] only matches a true edge at that same cell.
    gt = np.zeros((2, 2, 2), dtype=np.int8)
    gt[true_cell[0], true_cell[1], 1] = 1
    disc = _mark_graph([(0, 1, 1, "o-o")], n_lags=2)

    _, ax = plot_lag_metrics(
        gt,
        disc,
        metrics=["TP", "FP", "FN", "Adjacency TP", "Adjacency FP", "Adjacency FN"],
        undirected="adjacency",
    )
    expected = [0, 0, 1, 0, 1, 1] if true_cell == (1, 0) else [0, 0, 1, 1, 0, 0]
    np.testing.assert_array_equal(ax.images[0].get_array()[:, 1], expected)
    plt.close("all")


@pytest.mark.parametrize(
    "policy, expected",
    [
        # TP, FP, FN, adjacency line.
        ("bidirected", {"TP": 1, "FP": 1, "FN": 0, "adjacency": 0}),
        ("drop", {"TP": 0, "FP": 0, "FN": 1, "adjacency": 0}),
        ("adjacency", {"TP": 0, "FP": 0, "FN": 0, "adjacency": 1}),
    ],
)
def test_compare_graphs_oo_policies(policy, expected):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    from causalts.plotting import compare_graphs

    gt = np.zeros((2, 2, 1), dtype=np.int8)
    gt[0, 1, 0] = 1
    colors = {
        "TP": to_rgba("#00ff00"),
        "FP": to_rgba("#ff8800"),
        "FN": to_rgba("#ff0000"),
        "adjacency": to_rgba("#123456"),
    }

    _, ax = compare_graphs(
        gt,
        _oo_pair(),
        undirected=policy,
        tp_color="#00ff00",
        fp_color="#ff8800",
        fn_color="#ff0000",
        adjacency_color="#123456",
    )
    visible = _visible_arrow_colors(ax)
    for name, color in colors.items():
        assert visible.count(color) == expected[name], name
    plt.close("all")


def test_compare_graphs_adjacency_is_drawn_without_arrowhead():
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import FancyArrowPatch

    from causalts.plotting import compare_graphs

    gt = np.zeros((2, 2, 1), dtype=np.int8)
    gt[0, 1, 0] = 1

    _, ax = compare_graphs(
        gt, _oo_pair(), undirected="adjacency", adjacency_color="#123456"
    )
    adjacency = [
        a
        for a in ax.get_children()
        if isinstance(a, FancyArrowPatch)
        and a.get_edgecolor() == to_rgba("#123456")
        and a.get_edgecolor()[3] > 0
    ]
    assert adjacency
    assert all(type(a.get_arrowstyle()).__name__ == "Curve" for a in adjacency)
    plt.close("all")


@pytest.mark.parametrize("truth", [True, False])
def test_compare_graphs_adjacency_skips_lagged_self_links(truth):
    """A lagged o-o self-link is not drawn, as directed self-links are not."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    from causalts.plotting import compare_graphs

    gt = np.zeros((2, 2, 2), dtype=np.int8)
    if truth:
        gt[0, 0, 1] = 1
    disc = _mark_graph([(0, 0, 1, "o-o")], d=2, n_lags=2)
    _, ax = compare_graphs(gt, disc, undirected="adjacency")
    curves = [
        a
        for a in ax.get_children()
        if isinstance(a, FancyArrowPatch)
        and type(a.get_arrowstyle()).__name__ == "Curve"
    ]
    assert curves == []
    plt.close("all")


@pytest.mark.parametrize(
    "mark, parameter", [("o-o", "undirected"), ("x-x", "conflict")]
)
@pytest.mark.parametrize("truth", [True, False])
def test_compare_graphs_adjacency_lines_are_visible(mark, parameter, truth):
    """Hiding the adjacency line must change the rendered image."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    from causalts.plotting import compare_graphs

    gt = np.zeros((2, 2, 1), dtype=np.int8)
    if truth:
        gt[0, 1, 0] = 1  # correct adjacency, else spurious
    fig, ax = compare_graphs(gt, _oo_pair(mark), **{parameter: "adjacency"})
    lines = [
        a
        for a in ax.get_children()
        if isinstance(a, FancyArrowPatch)
        and type(a.get_arrowstyle()).__name__ == "Curve"
    ]
    assert lines

    def pixels():
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba()).copy()

    shown = pixels()
    for a in lines:
        a.set_visible(False)
    assert (shown != pixels()).any()
    plt.close("all")


@pytest.mark.parametrize(
    "policy, true_count, false_count",
    # A lag-0 adjacency claim contributes one p-value, not one per cell.
    [("bidirected", 1, 1), ("drop", 0, 0), ("adjacency", 1, 0)],
)
def test_plot_pvalue_distribution_oo_policies(policy, true_count, false_count):
    import matplotlib.pyplot as plt

    from causalts.plotting import plot_pvalue_distribution

    gt = np.zeros((2, 2, 1), dtype=np.int8)
    gt[0, 1, 0] = 1
    pvals = np.full((2, 2, 1), 0.5)

    _, ax = plot_pvalue_distribution(
        pvals, gt, graph_discovered=_oo_pair(), undirected=policy
    )
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert f"TP (n={true_count})" in labels
    assert f"FP (n={false_count})" in labels
    plt.close("all")


@pytest.mark.parametrize("cause, effect", [(0, 1), (1, 0)])
def test_plot_pvalue_distribution_adjacency_uses_the_true_cell(cause, effect):
    import matplotlib.pyplot as plt

    from causalts.plotting import plot_pvalue_distribution

    gt = np.zeros((2, 2, 1), dtype=np.int8)
    gt[cause, effect, 0] = 1
    pvals = np.full((2, 2, 1), 0.9)
    pvals[cause, effect, 0] = 0.01

    _, ax = plot_pvalue_distribution(
        pvals, gt, graph_discovered=_oo_pair(), undirected="adjacency"
    )
    tp = [p for p in ax.patches if p.get_height() > 0 and p.get_x() < 0.5]
    assert sum(p.get_height() for p in tp) > 0
    assert not [p for p in ax.patches if p.get_height() > 0 and p.get_x() > 0.5]
    plt.close("all")


def test_plot_pvalue_distribution_adjacency_falls_back_to_tested_mirror():
    import matplotlib.pyplot as plt

    from causalts.plotting import plot_pvalue_distribution

    gt = np.zeros((2, 2, 1), dtype=np.int8)
    gt[0, 1, 0] = 1
    pvals = np.full((2, 2, 1), np.nan)
    pvals[1, 0, 0] = 0.01

    _, ax = plot_pvalue_distribution(
        pvals, gt, graph_discovered=_oo_pair(), undirected="adjacency"
    )
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "TP (n=1)" in labels
    plt.close("all")


def test_compare_graphs_adjacency_with_zero_curvature_terminates():
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    from causalts.plotting import compare_graphs

    # A lagged directed edge and a lag-0 adjacency on the same pair need two
    # strokes; with curved_radius=0 the second must still get its own radius.
    gt = np.zeros((2, 2, 2), dtype=np.int8)
    gt[0, 1, 0] = 1
    disc = _mark_graph([(0, 1, 0, "o-o"), (0, 1, 1, "-->")], n_lags=2)

    _, ax = compare_graphs(gt, disc, undirected="adjacency", curved_radius=0)
    rads = {
        a.get_connectionstyle().rad
        for a in ax.get_children()
        if isinstance(a, FancyArrowPatch)
    }
    assert len(rads) >= 2
    plt.close("all")


@pytest.mark.parametrize("curved_radius", [0.2, 0.0])
def test_compare_graphs_strokes_on_one_pair_take_distinct_arcs(curved_radius):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    from causalts.plotting import compare_graphs

    gt = np.zeros((2, 2, 2), dtype=np.int8)
    gt[0, 1, 0] = 1
    disc = _mark_graph([(0, 1, 0, "o-o"), (0, 1, 1, "-->"), (1, 0, 1, "-->")], n_lags=2)

    _, ax = compare_graphs(
        gt, disc, undirected="adjacency", curved_radius=curved_radius
    )
    arcs = []
    for a in ax.get_children():
        if isinstance(a, FancyArrowPatch):
            (xa, ya), (xb, yb) = a._posA_posB
            rad = a.get_connectionstyle().rad
            # arc3 radius is relative to travel direction; normalise it.
            arcs.append(round(rad if (xa, ya) < (xb, yb) else -rad, 6))
    # Two directed arrows and one adjacency line: three physical arcs.
    assert len(set(arcs)) == 3
    plt.close("all")


def test_plot_lag_metrics_adjacency_rows_do_not_depend_on_the_graph():
    import matplotlib.pyplot as plt

    from causalts.plotting import plot_lag_metrics

    gt = np.zeros((2, 2, 1), dtype=np.int8)
    gt[0, 1, 0] = 1
    directed_only = _mark_graph([(0, 1, 0, "-->")])

    _, ax = plot_lag_metrics(gt, directed_only, undirected="adjacency")
    assert ax.images[0].get_array().shape[0] == 6
    plt.close("all")


def _self_loop_graphs():
    gt = np.zeros((2, 2, 3), dtype=np.int8)
    gt[0, 1, 0] = 1
    gt[0, 0, 1] = 1  # missed
    gt[1, 1, 1] = 1  # found
    disc = _mark_graph([(0, 1, 0, "-->"), (1, 1, 1, "-->"), (1, 1, 2, "-->")], n_lags=3)
    return gt, disc


def test_compare_graphs_self_loops_off_by_default():
    import matplotlib.pyplot as plt

    from causalts.plotting import compare_graphs

    gt, disc = _self_loop_graphs()
    _, ax = compare_graphs(gt, disc)
    assert not _loop_patches(ax)
    plt.close("all")


def _loop_patches(ax):
    return [a for a in ax.get_children() if a.get_gid() == "self-loop"]


def test_compare_graphs_self_loops_colour_and_label_by_status():
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    from causalts.plotting import compare_graphs

    gt, disc = _self_loop_graphs()
    _, ax = compare_graphs(gt, disc, show_self_loops=True, node_layout="circular")
    colours = sorted(mcolors.to_hex(a.get_edgecolor()) for a in _loop_patches(ax))
    # Missed X(t-1) -> X(t) and the found / spurious Y self-links.
    assert colours == sorted([mcolors.to_hex("red"), mcolors.to_hex("green")])
    texts = {t.get_text() for t in ax.texts}
    assert {"1", "1\u2713,2\u2717"} <= texts
    plt.close("all")


def test_compare_graphs_self_loop_points_away_from_the_edges():
    import matplotlib.pyplot as plt

    from causalts.plotting import compare_graphs

    gt = np.zeros((2, 2, 2), dtype=np.int8)
    gt[0, 1, 0] = 1
    gt[0, 0, 1] = 1
    node_pos = {"x": np.array([0.0, 1.0]), "y": np.array([0.0, 0.0])}
    _, ax = compare_graphs(
        gt, gt, show_self_loops=True, node_pos=node_pos, show_legend=False
    )
    labels = [t for t in ax.texts if t.get_text() == "1"]
    # The 0 -> 1 edge goes right, so node 0's loop label sits to its left.
    assert any(t.get_position()[0] < 0 for t in labels)
    plt.close("all")


def test_compare_graphs_adjacency_self_loop_has_no_arrowhead():
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    from causalts.plotting import compare_graphs

    gt = np.zeros((2, 2, 2), dtype=np.int8)
    gt[0, 0, 1] = 1
    disc = _mark_graph([(0, 0, 1, "o-o")], n_lags=2)
    _, ax = compare_graphs(gt, disc, undirected="adjacency", show_self_loops=True)
    loops = _loop_patches(ax)
    assert len(loops) == 1
    assert type(loops[0].get_arrowstyle()).__name__ == "Curve"
    assert mcolors.to_hex(loops[0].get_edgecolor()) == mcolors.to_hex("tab:blue")
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "Adjacency (correct, unoriented)" in labels
    plt.close("all")


def test_compare_graphs_self_loop_avoids_curved_edges():
    from causalts.plotting.compare import _self_loop_angle

    # Six edges from node 0, evenly spaced: every chord gap is 60 degrees,
    # but the arcs leave rotated by the curvature, so the free gap moves.
    angles = np.deg2rad(np.arange(6) * 60)
    pos = {
        0: (0.0, 0.0),
        **{k + 1: (np.cos(a), np.sin(a)) for k, a in enumerate(angles)},
    }
    edges = [(0, k, 0.2) for k in range(1, 7)]
    centre, _ = _self_loop_angle(0, pos, edges)
    tangents = []
    for _, k, rad in edges:
        xb, yb = pos[k]
        tangents.append(np.arctan2(-rad * xb + yb / 2, xb / 2 + rad * yb))
    gap = min(abs(np.angle(np.exp(1j * (centre - t)))) for t in tangents)
    assert np.isclose(gap, np.deg2rad(30))


def test_compare_graphs_legend_below_fits_in_the_figure():
    import matplotlib.pyplot as plt

    from causalts.plotting import compare_graphs

    gt = np.zeros((3, 3, 1), dtype=np.int8)
    gt[0, 1, 0] = gt[1, 2, 0] = 1
    fig, ax = compare_graphs(
        gt, gt, legend_loc="below", node_layout="circular", figsize=(4, 3)
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = ax.get_legend().get_window_extent(renderer)
    assert legend_box.y1 <= ax.get_window_extent(renderer).y0 + 1
    assert legend_box.y0 >= 0
    plt.close("all")


def test_compare_graphs_legend_below_clears_a_bulging_edge():
    import matplotlib.pyplot as plt

    from causalts.plotting import compare_graphs

    gt = np.zeros((2, 2, 1), dtype=np.int8)
    gt[1, 0, 0] = 1
    fig, ax = compare_graphs(gt, gt, node_layout="circular", legend_loc="below")
    _assert_edges_clear_legend(fig, ax)
    plt.close("all")

    # A sloped, strongly curved edge whose lowest point is off its midpoint.
    gt = np.zeros((2, 2, 1), dtype=np.int8)
    gt[0, 1, 0] = 1
    node_pos = {"x": np.array([0.0, 2.0]), "y": np.array([0.0, 1.7])}
    fig, ax = compare_graphs(
        gt, gt, node_pos=node_pos, curved_radius=1, figsize=(6, 6), legend_loc="below"
    )
    _assert_edges_clear_legend(fig, ax)
    plt.close("all")


def _assert_edges_clear_legend(fig, ax):
    from matplotlib.patches import FancyArrowPatch

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_top = ax.get_legend().get_window_extent(renderer).y1
    for a in ax.get_children():
        if isinstance(a, FancyArrowPatch) and a.get_alpha() != 0:
            assert a.get_window_extent(renderer).y0 >= legend_top - 1


@pytest.mark.parametrize(
    "kwargs, expected", [({}, 1.2), ({"self_loop_linewidth": 1.5}, 1.5)]
)
def test_compare_graphs_self_loop_linewidth(kwargs, expected):
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    from causalts.plotting import compare_graphs

    gt, disc = _self_loop_graphs()
    _, ax = compare_graphs(gt, disc, show_self_loops=True, **kwargs)
    widths = {
        mcolors.to_hex(a.get_edgecolor()): a.get_linewidth() for a in _loop_patches(ax)
    }
    # Default arrow_linewidth=3: loops use the FN width, 0.4 * 3 = 1.2.
    assert widths[mcolors.to_hex("green")] == pytest.approx(expected)
    assert widths[mcolors.to_hex("red")] == pytest.approx(min(1.2, expected))
    plt.close("all")


def test_compare_graphs_self_loops_with_zero_edge_width():
    import matplotlib.pyplot as plt

    from causalts.plotting import compare_graphs

    gt, disc = _self_loop_graphs()
    _, ax = compare_graphs(
        gt, disc, show_self_loops=True, arrow_linewidth=0, self_loop_linewidth=2
    )
    assert _loop_patches(ax)
    plt.close("all")


@pytest.mark.parametrize("show_self_loops", [False, True])
def test_compare_graphs_default_limits_depend_on_nodes_only(show_self_loops):
    import matplotlib.pyplot as plt

    from causalts.plotting import compare_graphs

    # Panels sharing node_pos must share their framing, whatever edges they draw.
    node_pos = {"x": np.array([0.0, 1.0, 0.5]), "y": np.array([0.0, 0.0, 0.8])}
    sparse = np.zeros((3, 3, 1), dtype=np.int8)
    sparse[0, 2, 0] = 1
    dense = sparse.copy()
    dense[0, 1, 0] = dense[1, 0, 0] = 1
    dense = np.concatenate([dense, np.zeros_like(dense)], axis=2)
    dense[0, 0, 1] = dense[2, 2, 1] = 1  # self-loops only in this panel
    sparse = np.concatenate([sparse, np.zeros_like(sparse)], axis=2)
    limits = []
    for graph in (sparse, dense):
        fig, ax = compare_graphs(
            sparse,
            graph,
            node_pos=node_pos,
            curved_radius=0.5,
            show_self_loops=show_self_loops,
        )
        fig.canvas.draw()
        limits.append((ax.get_xlim(), ax.get_ylim()))
        plt.close("all")
    np.testing.assert_allclose(limits[0], limits[1])


def test_compare_graphs_legend_below_fits_a_narrow_figure():
    import matplotlib.pyplot as plt

    from causalts.plotting import compare_graphs

    # An adjacency entry makes the legend wide; a 4-inch figure cannot hold
    # two columns, so it must stack them rather than clip.
    gt = np.zeros((2, 2, 1), dtype=np.int8)
    gt[0, 1, 0] = 1
    fig, ax = compare_graphs(
        gt,
        _oo_pair(),
        undirected="adjacency",
        legend_loc="below",
        figsize=(4, 3),
        legend_fontsize=10,
    )
    fig.canvas.draw()
    box = ax.get_legend().get_window_extent(fig.canvas.get_renderer())
    assert box.x0 >= 0 and box.x1 <= fig.bbox.width
    assert box.y0 >= 0
    plt.close("all")


@pytest.mark.parametrize("backend", ["pdf", "svg"])
def test_compare_graphs_legend_below_on_vector_backends(backend, tmp_path):
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    from causalts.plotting import compare_graphs

    gt, disc = _self_loop_graphs()
    fig = Figure()
    if backend == "pdf":
        from matplotlib.backends.backend_pdf import FigureCanvasPdf as Canvas
    else:
        from matplotlib.backends.backend_svg import FigureCanvasSVG as Canvas
    Canvas(fig)
    ax = fig.add_subplot(111)
    compare_graphs(gt, disc, legend_loc="below", fig_ax=(fig, ax))
    fig.savefig(tmp_path / f"out.{backend}")
    plt.close("all")


def test_compare_graphs_legend_below_fits_each_panel():
    import matplotlib.pyplot as plt

    from causalts.plotting import compare_graphs

    gt = np.zeros((2, 2, 1), dtype=np.int8)
    gt[0, 1, 0] = 1
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for ax in axes:
        compare_graphs(
            gt,
            _oo_pair(),
            undirected="adjacency",
            legend_loc="below",
            legend_fontsize=10,
            fig_ax=(fig, ax),
        )
    fig.canvas.draw()
    left, right = (ax.get_legend().get_window_extent() for ax in axes)
    assert left.x1 <= right.x0
    plt.close("all")


@pytest.mark.parametrize("legend_loc", ["below", "bottom"])
@pytest.mark.parametrize("figsize", [(12, 3), (10, 2), (4, 3), (3, 6)])
def test_compare_graphs_legend_below_stays_on_the_canvas(figsize, legend_loc):
    import matplotlib.pyplot as plt

    from causalts.plotting import compare_graphs

    gt = np.zeros((2, 2, 1), dtype=np.int8)
    gt[0, 1, 0] = 1
    fig, ax = compare_graphs(
        gt, gt, node_layout="circular", figsize=figsize, legend_loc=legend_loc
    )
    fig.canvas.draw()
    box = ax.get_legend().get_window_extent()
    assert box.y0 >= 0 and box.x0 >= 0 and box.x1 <= fig.bbox.width
    plt.close("all")


def _plot_graph_with_self_links():
    graph = np.full((3, 3, 3), "", dtype="<U3")
    graph[0, 1, 1] = "-->"
    graph[1, 2, 0], graph[2, 1, 0] = "-->", "<--"
    graph[0, 0, 1] = graph[0, 0, 2] = "-->"  # node 0: lags 1 and 2
    graph[2, 2, 1] = "-->"
    val = np.zeros(graph.shape)
    val[graph != ""] = 0.5
    val[0, 0, 2] = 0.9
    return graph, val


def test_plot_graph_self_loops_off_by_default():
    import matplotlib.pyplot as plt

    from causalts.plotting import plot_graph

    graph, val = _plot_graph_with_self_links()
    fig, ax = plot_graph(graph=graph, val_matrix=val)
    assert not _loop_patches(ax)
    plt.close("all")


def test_plot_graph_self_loops():
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    from causalts.plotting import plot_graph

    graph, val = _plot_graph_with_self_links()
    fig, ax = plot_graph(
        graph=graph, val_matrix=val, show_self_loops=True, arrow_linewidth=3
    )
    loops = _loop_patches(ax)
    assert len(loops) == 2  # nodes 0 and 2
    assert all(a.get_linewidth() == pytest.approx(1.2) for a in loops)
    texts = {t.get_text() for t in ax.texts}
    assert {"2,1", "1"} <= texts  # strongest lag first, as link labels
    # coloured with the link colour map at the strongest lag (0.9 vs 0.5)
    colours = {mcolors.to_hex(a.get_edgecolor()) for a in loops}
    assert len(colours) == 2
    plt.close("all")


def test_plot_graph_self_loop_linewidth():
    import matplotlib.pyplot as plt

    from causalts.plotting import plot_graph

    graph, val = _plot_graph_with_self_links()
    fig, ax = plot_graph(
        graph=graph, val_matrix=val, show_self_loops=True, self_loop_linewidth=2.5
    )
    assert all(a.get_linewidth() == 2.5 for a in _loop_patches(ax))
    plt.close("all")


def test_plot_graph_self_loop_on_a_single_node():
    import matplotlib.pyplot as plt

    from causalts.plotting import plot_graph

    graph = np.full((1, 1, 3), "", dtype="<U3")
    graph[0, 0, 1] = "-->"
    val = np.zeros(graph.shape)
    val[0, 0, 1] = 0.7
    fig, ax = plot_graph(graph=graph, val_matrix=val, show_self_loops=True)
    assert len(_loop_patches(ax)) == 1
    plt.close("all")


def test_plot_graph_self_loops_join_the_link_colour_scale():
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    from causalts.plotting import plot_graph

    # cross-links 0.1-0.2, loops 0.8 and 0.9: loops must not share a colour
    graph = np.full((3, 3, 2), "", dtype="<U3")
    graph[0, 1, 1], graph[1, 2, 1] = "-->", "-->"
    graph[0, 0, 1] = graph[2, 2, 1] = "-->"
    val = np.zeros(graph.shape)
    val[0, 1, 1], val[1, 2, 1], val[0, 0, 1], val[2, 2, 1] = 0.1, 0.2, 0.8, 0.9
    fig, ax = plot_graph(
        graph=graph,
        val_matrix=val,
        show_self_loops=True,
        vmin_edges=None,
        vmax_edges=None,
    )
    assert len({mcolors.to_hex(a.get_edgecolor()) for a in _loop_patches(ax)}) == 2
    plt.close("all")


def test_plot_graph_self_loop_labels_use_lag_array():
    import matplotlib.pyplot as plt

    from causalts.plotting import plot_graph

    graph, val = _plot_graph_with_self_links()
    fig, ax = plot_graph(
        graph=graph, val_matrix=val, show_self_loops=True, lag_array=[0, 10, 20]
    )
    texts = {t.get_text() for t in ax.texts}
    assert "20,10" in texts
    plt.close("all")


def test_plot_graph_self_loops_follow_the_node_transform():
    import matplotlib.pyplot as plt

    from causalts.plotting import plot_graph

    graph, val = _plot_graph_with_self_links()
    fig, ax = plt.subplots()
    node_pos = {
        "x": np.array([0.2, 0.5, 0.8]),
        "y": np.array([0.5, 0.8, 0.5]),
        "transform": ax.transAxes,
    }
    plot_graph(
        graph=graph,
        val_matrix=val,
        show_self_loops=True,
        node_pos=node_pos,
        fig_ax=(fig, ax),
    )
    assert all(a.get_transform() == ax.transAxes for a in _loop_patches(ax))
    plt.close("all")


def test_plot_graph_self_loop_ties_list_the_higher_lag_first():
    import matplotlib.pyplot as plt

    from causalts.plotting import plot_graph

    graph = np.full((2, 2, 3), "", dtype="<U3")
    graph[0, 1, 1] = graph[0, 0, 1] = graph[0, 0, 2] = "-->"
    _, ax = plot_graph(graph=graph, show_self_loops=True)  # all values tie
    assert "2,1" in {t.get_text() for t in ax.texts}
    plt.close("all")


def test_plot_graph_spurious_self_loop_is_grey():
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    from causalts.plotting import plot_graph

    graph, val = _plot_graph_with_self_links()
    attribute = np.full(graph.shape, "", dtype=object)
    attribute[0, 0, 2] = "spurious"  # the strongest lag of node 0
    _, ax = plot_graph(
        graph=graph, val_matrix=val, show_self_loops=True, link_attribute=attribute
    )
    colours = {mcolors.to_hex(a.get_edgecolor()) for a in _loop_patches(ax)}
    assert mcolors.to_hex("grey") in colours
    plt.close("all")


@pytest.mark.parametrize("self_links", [False, True])
def test_plot_graph_without_cross_links_and_a_val_matrix(self_links):
    import matplotlib.pyplot as plt

    from causalts.plotting import plot_graph, plot_time_series_graph

    graph = np.full((3, 3, 2), "", dtype="<U3")
    if self_links:
        graph[0, 0, 1] = "-->"
    val = np.where(graph != "", 0.5, 0.0)
    plot_graph(graph=graph, val_matrix=val)
    plot_time_series_graph(graph=graph, val_matrix=val)
    plt.close("all")


def test_plot_graph_spurious_target_self_loop_stays_grey():
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    from causalts.plotting import plot_graph

    graph, val = _plot_graph_with_self_links()
    attribute = np.full(graph.shape, "", dtype=object)
    attribute[0, 0, 2] = "spurious"
    _, ax = plot_graph(
        graph=graph,
        val_matrix=val,
        show_self_loops=True,
        link_attribute=attribute,
        target_node=0,
    )
    colours = {mcolors.to_hex(a.get_edgecolor()) for a in _loop_patches(ax)}
    assert mcolors.to_hex("grey") in colours
    plt.close("all")


def test_plot_graph_new_options_come_after_return_pos():
    import inspect

    from causalts.plotting import plot_graph

    names = list(inspect.signature(plot_graph).parameters)
    assert names.index("return_pos") < names.index("show_self_loops")
    assert names.index("title") < names.index("show_self_loops")


def test_plot_graph_loop_only_colour_scale_ignores_the_placeholder():
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    from causalts.plotting import plot_graph

    graph = np.full((2, 2, 2), "", dtype="<U3")
    graph[0, 0, 1] = graph[1, 1, 1] = "-->"
    val = np.zeros(graph.shape)
    val[0, 0, 1], val[1, 1, 1] = 0.8, 0.9
    _, ax = plot_graph(
        graph=graph,
        val_matrix=val,
        show_self_loops=True,
        vmin_edges=None,
        vmax_edges=None,
        cmap_edges="viridis",
    )
    colours = {mcolors.to_hex(a.get_edgecolor()) for a in _loop_patches(ax)}
    ends = {mcolors.to_hex(plt.get_cmap("viridis")(x)) for x in (0.0, 1.0)}
    assert colours == ends  # 0.8 and 0.9 span the whole scale
    plt.close("all")


def test_elliptical_loop_endpoints_keep_their_angles():
    import matplotlib.pyplot as plt

    from causalts.plotting._self_loops import draw_loops

    fig, ax = plt.subplots()
    centre, width = np.deg2rad(60), np.deg2rad(80)  # gap 20-100 degrees
    draw_loops(
        ax,
        (0, 0),
        0.2,
        (centre, width),
        [("k", "solid", 1, "", True, 0)],
        node_width=0.8,
    )
    (loop,) = [a for a in ax.get_children() if a.get_gid() == "self-loop"]
    v = loop._path_original.vertices  # the loop as given, not the arrow outline
    for x, y in (v[0], v[-1]):
        angle = np.degrees(np.arctan2(y, x))
        assert 20 < angle < 100
    plt.close("all")


def test_loop_on_a_wide_node_clears_the_node():
    import matplotlib.pyplot as plt

    from causalts.plotting._self_loops import draw_loops

    fig, ax = plt.subplots()
    # pointing right out of a node 0.8 wide: the loop must reach past x=0.4
    draw_loops(
        ax,
        (0, 0),
        0.2,
        (0.0, 2 * np.pi),
        [("k", "solid", 1, "", True, 0)],
        node_width=0.8,
    )
    (loop,) = [a for a in ax.get_children() if a.get_gid() == "self-loop"]
    p0, c1, c2, p3 = loop._path_original.vertices
    apex = (p0 + 3 * c1 + 3 * c2 + p3) / 8
    assert apex[0] > 0.4
    plt.close("all")


def test_self_loops_start_on_an_elliptical_node():
    import matplotlib.pyplot as plt

    from causalts.plotting._self_loops import draw_loops

    fig, ax = plt.subplots()
    # loop pointing right, out of a node 0.8 wide and 0.2 tall at the origin
    draw_loops(
        ax,
        (0, 0),
        0.2,
        (0.0, 2 * np.pi),
        [("k", "solid", 1, "", True, 0)],
        node_width=0.8,
    )
    (loop,) = [a for a in ax.get_children() if a.get_gid() == "self-loop"]
    x, y = loop.get_path().vertices[0]
    assert (x / 0.4) ** 2 + (y / 0.1) ** 2 == pytest.approx(1.0)
    plt.close("all")


@pytest.mark.parametrize(
    "mark, head", [("o-o", "Curve"), ("<->", "CurveFilledAB"), ("-->", "CurveFilledB")]
)
def test_plot_graph_self_loops_keep_the_mark(mark, head):
    import matplotlib.pyplot as plt

    from causalts.plotting import plot_graph

    graph = np.full((2, 2, 2), "", dtype="<U3")
    graph[0, 1, 1] = "-->"
    graph[0, 0, 1] = mark
    (loop,) = _loop_patches(plot_graph(graph=graph, show_self_loops=True)[1])
    assert type(loop.get_arrowstyle()).__name__ == head
    plt.close("all")


@pytest.mark.parametrize("colour", ["#9C27B0", (0.612, 0.153, 0.690)])
def test_plot_graph_self_loop_on_a_target_uses_the_target_colour(colour):
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    from causalts.plotting import plot_graph

    graph, val = _plot_graph_with_self_links()
    _, ax = plot_graph(
        graph=graph,
        val_matrix=val,
        show_self_loops=True,
        target_node=0,
        target_between_color=colour,
    )
    colours = {mcolors.to_hex(a.get_edgecolor()) for a in _loop_patches(ax)}
    assert mcolors.to_hex(colour) in colours
    plt.close("all")


def test_plot_graph_self_loops_follow_alpha():
    import matplotlib.pyplot as plt

    from causalts.plotting import plot_graph

    graph, val = _plot_graph_with_self_links()
    _, ax = plot_graph(graph=graph, val_matrix=val, show_self_loops=True, alpha=0.15)
    assert all(a.get_alpha() == 0.15 for a in _loop_patches(ax))
    plt.close("all")


@pytest.mark.parametrize(
    "mark, ends",
    [("o->", ["o"]), ("x-x", ["x", "x"]), ("o-o", ["o", "o"]), ("-->", [])],
)
def test_plot_graph_self_loops_draw_endpoint_symbols(mark, ends):
    import matplotlib.pyplot as plt

    from causalts.plotting import plot_graph

    graph = np.full((2, 2, 2), "", dtype="<U3")
    graph[0, 1, 1] = "-->"
    graph[0, 0, 1] = mark
    _, ax = plot_graph(graph=graph, show_self_loops=True)
    symbols = sorted(
        line.get_marker()
        for line in ax.get_lines()
        if line.get_gid() == "self-loop-end"
    )
    assert symbols == sorted(ends)
    plt.close("all")


def test_plot_graph_loop_space_reserved_without_loops():
    import matplotlib.pyplot as plt

    from causalts.plotting import plot_graph

    with_loop, val = _plot_graph_with_self_links()
    without = with_loop.copy()
    for i in range(3):
        without[i, i, :] = ""
    node_pos = {"x": np.array([0.0, 1.0, 0.5]), "y": np.array([0.0, 0.0, 0.8])}
    limits = []
    for graph in (with_loop, without):
        fig, ax = plot_graph(graph=graph, node_pos=node_pos, show_self_loops=True)
        fig.canvas.draw()
        limits.append((ax.get_xlim(), ax.get_ylim()))
        plt.close("all")
    np.testing.assert_allclose(limits[0], limits[1])


def test_compare_graphs_loops_keep_colour_transparency():
    import matplotlib.pyplot as plt

    from causalts.plotting import compare_graphs

    gt, disc = _self_loop_graphs()
    _, ax = compare_graphs(gt, disc, show_self_loops=True, tp_color="#00800040")
    greens = [
        a
        for a in _loop_patches(ax)
        if a.get_edgecolor()[:3] == pytest.approx((0, 0.502, 0), abs=0.01)
    ]
    assert greens and all(
        a.get_edgecolor()[3] == pytest.approx(0.25, abs=0.01) for a in greens
    )
    plt.close("all")


def test_compare_functions_reject_invalid_policy():
    from causalts.plotting import (
        compare_graphs,
        plot_lag_metrics,
        plot_pvalue_distribution,
    )

    gt = np.zeros((2, 2, 1), dtype=np.int8)
    gt[0, 1, 0] = 1
    disc = _oo_pair()
    pvals = np.full((2, 2, 1), 0.5)

    with pytest.raises(ValueError, match="undirected must be one of"):
        compare_graphs(gt, disc, undirected="bogus")
    with pytest.raises(ValueError, match="conflict must be one of"):
        plot_lag_metrics(gt, disc, conflict="bogus")
    with pytest.raises(ValueError, match="undirected must be one of"):
        plot_pvalue_distribution(pvals, gt, graph_discovered=disc, undirected="bogus")
    with pytest.raises(ValueError, match="undirected must be one of"):
        plot_pvalue_distribution(pvals, gt, undirected="bogus")


@pytest.mark.parametrize(
    "n_lags, wrap",
    [
        (1, lambda g: g[:, :, 0]),
        (1, lambda g: g[None]),
        (1, lambda g: g[..., None]),
        (2, lambda g: g[None]),
        (2, lambda g: g[..., None]),
        (2, lambda g: g[:, None]),
    ],
)
def test_plot_lag_metrics_squeezes_extra_singleton_axes(n_lags, wrap):
    import matplotlib.pyplot as plt

    from causalts.plotting import plot_lag_metrics

    rng = np.random.default_rng(0)
    gt = rng.integers(0, 2, size=(3, 3, n_lags))
    disc = rng.integers(0, 2, size=(3, 3, n_lags))

    _, ax = plot_lag_metrics(gt, disc)
    expected = ax.images[0].get_array()
    _, ax = plot_lag_metrics(wrap(gt), wrap(disc))
    np.testing.assert_array_equal(ax.images[0].get_array(), expected)
    plt.close("all")


def _make_frame():
    import pandas as pd

    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.standard_normal((60, 4)), columns=list("ABCD"))


def _extra_axes(**kwargs):
    """Axes added to a fresh figure by one corrplot call (1 = colorbar drawn)."""
    import matplotlib.pyplot as plt

    from causalts.plotting import corrplot

    fig, ax = plt.subplots()
    before = len(fig.axes)
    corrplot(_make_frame(), fig_ax=(fig, ax), **kwargs)
    added = len(fig.axes) - before
    plt.close(fig)
    return added


def _glyph_path_count(ax):
    from matplotlib.collections import PatchCollection

    return sum(
        len(c.get_paths()) for c in ax.collections if isinstance(c, PatchCollection)
    )


def test_corrplot_smoke_from_raw_data():
    import matplotlib.pyplot as plt

    from causalts.plotting import corrplot

    fig, ax, matrix = corrplot(_make_frame(), method="circle")
    assert matrix.shape[0] == matrix.shape[1]
    assert ax is not None
    plt.close(fig)


def test_corrplot_colorbar_false_suppresses_for_every_method():
    # Regression: 'color' and 'shade' encode magnitude in the fill alone and
    # used to draw a colorbar even when the caller passed colorbar=False.
    for method in ("circle", "square", "ellipse", "number", "color", "shade", "pie"):
        assert _extra_axes(method=method, colorbar=False) == 0, method


def test_corrplot_colorbar_default_still_draws():
    for method in ("circle", "color", "shade"):
        assert _extra_axes(method=method) == 1, method


def test_corrplot_colorbar_false_with_color_only_half():
    assert _extra_axes(upper="color", lower="number", colorbar=False) == 0


def test_corrplot_invalid_method_raises():
    from causalts.plotting import corrplot

    with pytest.raises(ValueError):
        corrplot(_make_frame(), method="typo")


def test_corrplot_invalid_diag_raises():
    from causalts.plotting import corrplot

    with pytest.raises(ValueError):
        corrplot(_make_frame(), diag="nonsense")


def test_corrplot_precomputed_with_pvalues_and_coefs():
    import matplotlib.pyplot as plt

    from causalts.plotting import compute_association_matrix, corrplot

    df = _make_frame()
    m = compute_association_matrix(df, metric="pearson")
    pvals = np.full(m.shape, 0.01)
    np.fill_diagonal(pvals, 0.0)
    fig, ax, _ = corrplot(
        m, is_corr=True, method="color", pvalues=pvals, add_coef=True, order="hclust"
    )
    plt.close(fig)


def test_corrplot_square_raw_ndarray_not_treated_as_precomputed():
    # a non-symmetric square raw matrix (n_samples == n_features) must be
    # correlation-computed, not misread as a precomputed association matrix
    import matplotlib.pyplot as plt

    from causalts.plotting import corrplot

    rng = np.random.default_rng(0)
    raw = rng.standard_normal((5, 5))  # square but not symmetric
    fig, ax, matrix = corrplot(raw)
    # association matrix has unit diagonal; the raw matrix does not
    assert np.allclose(np.diag(matrix.values), 1.0)
    plt.close(fig)


def test_compute_association_matrix_pearson():
    from causalts.plotting import compute_association_matrix

    df = _make_frame()
    m = compute_association_matrix(df, metric="pearson")
    assert m.shape == (df.shape[1], df.shape[1])
    # symmetric with unit diagonal
    assert np.allclose(m.values, m.values.T)
    assert np.allclose(np.diag(m.values), 1.0)


def test_compute_association_matrix_bad_metric_raises():
    from causalts.plotting import compute_association_matrix

    with pytest.raises(ValueError):
        compute_association_matrix(_make_frame(), metric="not_a_metric")


def test_corr_table_ci_uses_metric_appropriate_se():
    from causalts.plotting.corrplot import _fisher_ci

    r, n = 0.5, 30
    pearson_ci = _fisher_ci(r, n, metric="pearson")
    spearman_ci = _fisher_ci(r, n, metric="spearman")
    kendall_ci = _fisher_ci(r, n, metric="kendall")

    # Spearman/Kendall have a larger variance than Pearson at the same r, n,
    # so their CIs must be wider, not identical to the Pearson interval.
    assert spearman_ci != pearson_ci
    assert kendall_ci != pearson_ci
    pearson_width = pearson_ci[1] - pearson_ci[0]
    spearman_width = spearman_ci[1] - spearman_ci[0]
    assert spearman_width > pearson_width


def test_corrplot_diag_glyph_renders_the_diagonal():
    """diag='glyph' draws the diagonal as an ordinary cell.

    Needed for directed matrices, where the diagonal is real data (a
    self-loop) rather than the trivial 1.0 of a correlation matrix.
    """
    import matplotlib.pyplot as plt

    from causalts.plotting import corrplot

    n = 4

    fig, ax = plt.subplots()
    corrplot(
        _make_frame(), method="color", diag="blank", colorbar=False, fig_ax=(fig, ax)
    )
    without = _glyph_path_count(ax)
    plt.close(fig)

    fig, ax = plt.subplots()
    corrplot(
        _make_frame(), method="color", diag="glyph", colorbar=False, fig_ax=(fig, ax)
    )
    with_diag = _glyph_path_count(ax)
    plt.close(fig)

    assert with_diag == without + n


def test_corrplot_diag_glyph_leaves_split_diagonal_blank():
    """With an upper/lower split, diag="glyph" draws nothing on the diagonal.

    Documented behaviour, not an oversight: a diagonal cell belongs to neither
    half, so there is no method to borrow. Pinned so the blank diagonal cannot
    turn into an arbitrary one (e.g. silently falling back to `method`, which
    the caller never set when upper/lower are given).
    """
    import matplotlib.pyplot as plt

    from causalts.plotting import corrplot

    counts = {}
    for diag in ("blank", "glyph"):
        fig, ax = plt.subplots()
        corrplot(
            _make_frame(),
            upper="circle",
            lower="color",
            diag=diag,
            colorbar=False,
            fig_ax=(fig, ax),
        )
        counts[diag] = _glyph_path_count(ax)
        plt.close(fig)

    assert counts["glyph"] == counts["blank"]


def test_corrplot_grid_border_is_closed():
    """All four edges of the grid border must be drawn.

    The border used to be axhline/axvline at exactly the axis limits, so half
    of each boundary line fell outside the clip box and the right and bottom
    edges disappeared.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from causalts.plotting import corrplot

    n = 4
    fig, ax = plt.subplots()
    corrplot(_make_frame(), method="circle", colorbar=False, fig_ax=(fig, ax))

    border = [
        p
        for p in ax.patches
        if isinstance(p, Rectangle)
        and not p.get_fill()
        and np.isclose(p.get_width(), n)
        and np.isclose(p.get_height(), n)
    ]
    assert border, "expected a full-extent unfilled border rectangle"
    assert not border[0].get_clip_on(), "border must not be clipped at the axes edge"
    plt.close(fig)


def test_corr_table_smoke():
    from causalts.plotting import corr_table

    table = corr_table(_make_frame())
    frame = table.to_frame()
    assert frame.shape[0] == 4
    repr(table)
    table._repr_html_()


def test_corr_table_full_matrix_toggle():
    from causalts.plotting import corr_table

    lower_only = corr_table(_make_frame(), full_matrix=False).to_frame()
    full = corr_table(_make_frame(), full_matrix=True).to_frame()

    def _filled(frame):
        var_cols = frame.columns[-4:] if "M" in frame.columns else frame.columns
        return (frame[var_cols] != "").sum().sum()

    assert _filled(full) > _filled(lower_only)


def test_corr_table_show_n_toggle():
    from causalts.plotting import corr_table

    with_n = corr_table(_make_frame(), show_n=True).to_frame()
    without_n = corr_table(_make_frame(), show_n=False).to_frame()

    assert "M" in with_n.columns and "SD" in with_n.columns
    assert "M" not in without_n.columns


def test_corr_table_dcor_no_stars():
    from causalts.plotting import corr_table

    table = corr_table(_make_frame(), metric="dcor", sig_stars=True)
    assert table.sig_stars is False
    frame = table.to_frame()
    assert not frame.apply(lambda col: col.str.contains(r"\*")).any().any()


def test_corr_table_html_escapes_variable_names():
    import pandas as pd

    from causalts.plotting import corr_table

    df = pd.DataFrame({"A<script>": [1, 2, 3, 4], "B&C": [4, 3, 2, 1]}, dtype=float)
    html = corr_table(df)._repr_html_()

    assert "<script>" not in html
    assert "A&lt;script&gt;" in html
    assert "B&amp;C" in html


def test_corr_table_pvalues_accepts_ndarray():
    import numpy as np
    import pandas as pd

    from causalts.plotting import corr_table

    df = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": [4.0, 3.0, 2.0, 1.0]})
    pvals = np.array([[0.0, 0.01], [0.01, 0.0]])
    table = corr_table(df, pvalues=pvals)
    assert "*" in table.to_frame().loc["2. B", 1]


def test_corr_table_show_ci_ignored_for_unsupported_metric():
    import numpy as np
    import pandas as pd

    from causalts.plotting import corr_table

    df = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": [4.0, 3.0, 2.0, 1.0]})
    corr = df.corr()
    pvals = pd.DataFrame(
        np.array([[0.0, 0.01], [0.01, 0.0]]), index=corr.index, columns=corr.columns
    )
    table = corr_table(df, metric="dcor", pvalues=pvals, show_ci=True)
    assert table.show_ci is False
    assert "CI" not in table._footnote()
    assert "[" not in table.to_frame().to_string()


def test_corr_table_nan_correlation_is_blank_not_literal_nan():
    import pandas as pd

    from causalts.plotting import corr_table

    df = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0], "B": [4.0, 3.0, 2.0, 1.0]})
    frame = corr_table(df, show_ci=True).to_frame()
    assert "nan" not in frame.to_string().lower()


def test_corr_table_nan_mean_sd_is_blank_not_literal_nan():
    import numpy as np
    import pandas as pd

    from causalts.plotting import corr_table

    # A single valid observation makes SD (ddof=1) undefined; an all-NaN
    # column makes both M and SD undefined.
    df = pd.DataFrame({"A": [1.0, np.nan, np.nan, np.nan], "B": [4.0, 3.0, 2.0, 1.0]})
    frame = corr_table(df).to_frame()
    assert "nan" not in frame.to_string().lower()

    df_all_nan = pd.DataFrame(
        {"A": [np.nan, np.nan, np.nan, np.nan], "B": [4.0, 3.0, 2.0, 1.0]}
    )
    frame_all_nan = corr_table(df_all_nan).to_frame()
    assert "nan" not in frame_all_nan.to_string().lower()


def test_corr_table_duplicate_column_names_do_not_crash():
    import pandas as pd

    from causalts.plotting import corr_table

    df = pd.DataFrame(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]], columns=["A", "A", "B"]
    )
    frame = corr_table(df).to_frame()
    assert frame.shape[0] == 3


def test_corr_table_duplicate_column_names_with_explicit_pvalues_df():
    import pandas as pd

    from causalts.plotting import corr_table

    # r's column order after data.corr() may not match a user-supplied
    # pvalues DataFrame's order; with duplicate labels, pandas' own
    # .reindex() raises on that mismatch (label alignment is ambiguous).
    df = pd.DataFrame(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]], columns=["A", "A", "B"]
    )
    pvals = pd.DataFrame(
        [[0.0, 0.01, 0.02], [0.01, 0.0, 0.03], [0.02, 0.03, 0.0]],
        columns=["B", "A", "A"],
        index=["B", "A", "A"],
    )
    frame = corr_table(df, pvalues=pvals).to_frame()
    assert frame.shape[0] == 3


def test_fisher_ci_rejects_nonfinite_r():
    from causalts.plotting.corrplot import _fisher_ci

    assert _fisher_ci(float("nan"), 30) is None
    assert _fisher_ci(float("inf"), 30) is None


def test_pairs_panel_smoke():
    import matplotlib.pyplot as plt

    from causalts.plotting import pairs_panel

    fig, axes, matrix = pairs_panel(_make_frame())
    assert axes.shape == (4, 4)
    assert matrix.shape == (4, 4)
    plt.close(fig)


def test_pairs_panel_wide_data_warns():
    import matplotlib.pyplot as plt
    import pandas as pd

    from causalts.plotting import pairs_panel

    rng = np.random.default_rng(0)
    wide = pd.DataFrame(rng.standard_normal((30, 16)))

    with pytest.warns(UserWarning):
        fig, axes, matrix = pairs_panel(wide)
    plt.close(fig)


def test_pairs_panel_density_overlays_kde_line():
    import matplotlib.pyplot as plt

    from causalts.plotting import pairs_panel

    fig, axes, _ = pairs_panel(_make_frame(), density=True)
    diag_lines = sum(len(axes[i, i].lines) for i in range(4))
    plt.close(fig)

    fig, axes, _ = pairs_panel(_make_frame(), density=False)
    diag_lines_off = sum(len(axes[i, i].lines) for i in range(4))
    plt.close(fig)

    assert diag_lines == 4
    assert diag_lines_off == 0


def test_pairs_panel_edgecolor_toggle():
    import matplotlib.pyplot as plt

    from causalts.plotting import pairs_panel

    fig, axes, _ = pairs_panel(_make_frame(), edgecolor="black")
    scatter_ax = axes[1, 0]
    edgecolors_on = scatter_ax.collections[0].get_edgecolors()
    plt.close(fig)

    fig, axes, _ = pairs_panel(_make_frame(), edgecolor=None)
    scatter_ax = axes[1, 0]
    edgecolors_off = scatter_ax.collections[0].get_edgecolors()
    plt.close(fig)

    assert len(edgecolors_on) > 0
    assert len(edgecolors_off) == 0


def test_pairs_panel_infinite_value_does_not_crash():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from causalts.plotting import pairs_panel

    df = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, np.inf],
            "B": [4.0, 3.0, 2.0, 1.0],
            "C": [1.0, 2.0, 3.0, 4.0],
            "D": [5.0, 4.0, 3.0, 2.0],
        }
    )
    fig, axes, matrix = pairs_panel(df)
    plt.close(fig)


def test_pairs_panel_constant_column_shows_na_not_nan():
    import matplotlib.pyplot as plt
    import pandas as pd

    from causalts.plotting import pairs_panel

    df = pd.DataFrame(
        {
            "A": [1.0, 1.0, 1.0, 1.0],
            "B": [4.0, 3.0, 2.0, 1.0],
            "C": [1.0, 2.0, 1.0, 2.0],
            "D": [5.0, 4.0, 3.0, 2.0],
        }
    )
    fig, axes, matrix = pairs_panel(df)
    upper_texts = [t.get_text() for ax in axes.flat for t in ax.texts if t.get_text()]
    assert not any("nan" in t.lower() for t in upper_texts)
    plt.close(fig)


def test_compare_graphs_legend_best_location():
    from causalts.plotting import compare_graphs

    g = _make_graph()
    _, ax = compare_graphs(g, g, var_names=["X", "Y", "Z"])
    assert ax.get_legend()._loc == 0  # matplotlib code for loc="best"


def test_plot_metrics_summary_rejects_scores():
    # It cast to int8 first, so 0.7 was scored as "no edge" (#165).
    from causalts.plotting.compare import plot_metrics_summary

    gt = np.zeros((2, 2, 2), dtype=int)
    gt[0, 1, 1] = 1
    scores = gt * 0.7
    with pytest.raises(ValueError, match="binary graph"):
        plot_metrics_summary(gt, scores)


def _node_label_texts(ax):
    """Return node label strings drawn at node Ellipse centers."""
    from matplotlib.patches import Ellipse

    centers = {
        (round(float(p.center[0]), 6), round(float(p.center[1]), 6))
        for p in ax.patches
        if isinstance(p, Ellipse)
    }
    return {
        t.get_text()
        for t in ax.texts
        if (round(float(t.get_position()[0]), 6), round(float(t.get_position()[1]), 6))
        in centers
    }


def _make_sample_results(include_C=False):
    """Build lightweight CdnotsResult and CedarResult instances for plot tests."""
    import pandas as pd

    from causalts.cdnots.result import CdnotsResult
    from causalts.cedar.result import CedarResult

    df = pd.DataFrame(np.zeros((10, 3)), columns=["X0", "X1", "X2"])
    d_graph = 4 if include_C else 3
    g = np.zeros((d_graph, d_graph, 2), dtype=np.int8)
    g[0, 1, 1] = 1
    g[1, 2, 1] = 1
    if include_C:
        g[3, 0, 1] = 1

    cdnots_res = CdnotsResult(
        graph=None,
        cg_tig=g,
        pvalue_matrix=np.ones_like(g, dtype=float),
        pvals=[],
        var_names=list(df.columns),
        num_lags=1,
        lag_list=None,
        include_C=include_C,
        alpha=0.05,
        priority=1,
        stable=False,
        lag_verify_stats=None,
        df=df,
    )
    g_cedar = g[:3, :3, :]
    cedar_res = CedarResult(
        graph=g_cedar,
        val_matrix=g_cedar.astype(float),
        pvalue_matrix=np.ones_like(g_cedar, dtype=float),
        lag_importance=g_cedar.astype(float),
        lag_pvalues=np.ones_like(g_cedar, dtype=float),
        selected_lags=np.ones((3, 3)),
        var_names=list(df.columns),
        max_lag=1,
        alpha_cond1=0.05,
        alpha_cond2=0.05,
        n_tests=0,
        pruned=True,
        df=df,
    )
    return cdnots_res, cedar_res


@pytest.mark.parametrize("kind", ["cdnots", "cdnots_with_c", "cedar"])
@pytest.mark.parametrize("names_kind", ["index", "ndarray"])
def test_result_plot_accepts_index_and_ndarray_var_names(kind, names_kind):
    # Regression test for https://github.com/bloomberg/causal-ts/issues/207.
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.figure import Figure

    include_C = kind == "cdnots_with_c"
    cdnots_res, cedar_res = _make_sample_results(include_C=include_C)
    res = cedar_res if kind == "cedar" else cdnots_res

    override = ["A", "B", "D"]
    var_names = (
        pd.DataFrame(columns=override).columns
        if names_kind == "index"
        else np.array(override)
    )

    fig, ax = res.plot(var_names=var_names)
    try:
        assert isinstance(fig, Figure)
        expected = set(override) | ({"C"} if include_C else set())
        assert _node_label_texts(ax) == expected
    finally:
        plt.close(fig)


@pytest.mark.parametrize("kind", ["cdnots", "cdnots_with_c", "cedar"])
@pytest.mark.parametrize("empty_kind", ["none", "list", "index", "ndarray"])
def test_result_plot_empty_and_none_var_names_fall_back(kind, empty_kind):
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.figure import Figure

    include_C = kind == "cdnots_with_c"
    cdnots_res, cedar_res = _make_sample_results(include_C=include_C)
    res = cedar_res if kind == "cedar" else cdnots_res

    if empty_kind == "none":
        fig, ax = res.plot()
    else:
        empty = (
            []
            if empty_kind == "list"
            else (pd.Index([]) if empty_kind == "index" else np.array([]))
        )
        fig, ax = res.plot(var_names=empty)

    try:
        assert isinstance(fig, Figure)
        expected = set(res.var_names) | ({"C"} if include_C else set())
        assert _node_label_texts(ax) == expected
    finally:
        plt.close(fig)


def test_causal_result_plot_handles_none_and_index_self_var_names():
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.figure import Figure

    from causalts.result import CausalResult

    res = CausalResult.__new__(CausalResult)
    res.cg_tig = _make_graph()
    res.var_names = None

    fig, ax = res.plot()
    try:
        assert isinstance(fig, Figure)
        assert _node_label_texts(ax) == {"0", "1", "2"}
    finally:
        plt.close(fig)

    res.var_names = pd.Index(["P", "Q", "R"])
    fig, ax = res.plot()
    try:
        assert isinstance(fig, Figure)
        assert _node_label_texts(ax) == {"P", "Q", "R"}
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "names_kind",
    ["index", "ndarray", "none", "empty_list", "empty_index", "empty_ndarray"],
)
def test_tigramite_result_plot_marks_accepts_array_like_var_names(names_kind):
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.figure import Figure

    from causalts.tigramite_result import TigramiteResult

    df = pd.DataFrame(np.zeros((10, 3)), columns=["X0", "X1", "X2"])
    g = _make_graph()
    marks = np.full((3, 3, 2), "", dtype="<U3")
    marks[0, 1, 1] = "-->"
    marks[1, 2, 0] = "o-o"
    marks[2, 1, 0] = "o-o"
    res = TigramiteResult(g, df, {"method": "pcmciplus", "graph_raw": marks})

    override = ["A", "B", "D"]
    if names_kind == "index":
        arg, expected = pd.Index(override), set(override)
    elif names_kind == "ndarray":
        arg, expected = np.array(override), set(override)
    elif names_kind == "none":
        arg, expected = None, {"X0", "X1", "X2"}
    elif names_kind == "empty_list":
        arg, expected = [], {"X0", "X1", "X2"}
    elif names_kind == "empty_index":
        arg, expected = pd.Index([]), {"X0", "X1", "X2"}
    else:
        arg, expected = np.array([]), {"X0", "X1", "X2"}

    fig, ax = res.plot(var_names=arg)
    try:
        assert isinstance(fig, Figure)
        assert _node_label_texts(ax) == expected
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "names_kind, expected_x",
    [
        ("index", ["a", "b"]),
        ("ndarray", ["a", "b"]),
        ("none", ["X0", "X1"]),
        ("empty_list", ["X0", "X1"]),
        ("empty_index", ["X0", "X1"]),
        ("empty_ndarray", ["X0", "X1"]),
    ],
)
def test_stability_selector_plot_stability_path_var_names(names_kind, expected_x):
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.figure import Figure

    from causalts.grace.gated_discovery import StabilitySelector

    sel = StabilitySelector(num_vars=2, max_lag=1)
    a = np.zeros((2, 2, 2), dtype=np.int8)
    a[0, 0, 1] = 1
    sel._selections = [a, a, np.zeros_like(a)]
    sel._lambda_values = [0.01, 0.01, 1.0]
    sel._stability_scores = np.mean(sel._selections[:2], axis=0)

    if names_kind == "index":
        arg = pd.Index(["a", "b"])
    elif names_kind == "ndarray":
        arg = np.array(["a", "b"])
    elif names_kind == "none":
        arg = None
    elif names_kind == "empty_list":
        arg = []
    elif names_kind == "empty_index":
        arg = pd.Index([])
    else:
        arg = np.array([])

    fig = sel.plot_stability_path(var_names=arg)
    try:
        assert isinstance(fig, Figure)
        right = fig.axes[1]
        assert [t.get_text() for t in right.get_xticklabels()] == expected_x
        assert [t.get_text() for t in right.get_yticklabels()] == [
            f"{expected_x[1]}_L1"
        ]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("names_kind", ["index", "ndarray"])
@pytest.mark.parametrize("target_node", [0, [0, 2]])
def test_plot_graph_target_node_with_index_or_ndarray_var_names(
    names_kind, target_node
):
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.figure import Figure

    from causalts.plotting import plot_graph

    g = _make_graph()
    names = ["X", "Y", "Z"]
    var_names = pd.Index(names) if names_kind == "index" else np.array(names)

    fig, ax = plot_graph(g, var_names=var_names, target_node=target_node)
    try:
        assert isinstance(fig, Figure)
        assert _node_label_texts(ax) == set(names)
    finally:
        plt.close(fig)
