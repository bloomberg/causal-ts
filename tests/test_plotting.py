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
