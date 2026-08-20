# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Smoke tests for plotting (non-interactive Agg backend)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402


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


def test_corrplot_smoke():
    from causalts.plotting import corrplot

    corrplot(_make_frame())


def test_corrplot_colorbar_false_suppresses_for_every_method():
    # Regression: 'color' and 'shade' encode magnitude in the fill alone and
    # used to draw a colorbar even when the caller passed colorbar=False.
    for method in ("circle", "square", "ellipse", "number", "color", "shade", "pie"):
        assert _extra_axes(method=method, colorbar=False) == 0, method


def test_corrplot_colorbar_default_still_draws():
    for method in ("circle", "color", "shade"):
        assert _extra_axes(method=method) == 1, method


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
