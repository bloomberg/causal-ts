# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Behaviour tests for the network and lag-function plots (causalts.plotting._core)."""

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.markers import MarkerStyle
from matplotlib.patches import ArrowStyle, Ellipse, FancyArrowPatch, FancyBboxPatch

from causalts.plotting import (
    plot_graph,
    plot_lagfuncs,
    plot_time_series_graph,
    setup_matrix,
)
from causalts.plotting._core import (
    _auto_shells,
    _check_matrices,
    _default_kwargs_for_layout,
    _make_nice_axes,
    _merge_layout_kwargs,
    _myround,
    _reverse_patt,
    compute_node_positions,
    kk_positions_packed,
    normalize_positions_to_unit_square,
    spread_duplicates,
)

NAMES = ["X", "Y", "Z"]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Artist helpers
# ---------------------------------------------------------------------------


def _arrows(ax):
    """Visible edge patches (hidden helper patches have alpha 0 or 1e-8)."""
    return [
        p
        for p in ax.get_children()
        if isinstance(p, FancyArrowPatch)
        and p.get_alpha() is not None
        and p.get_alpha() > 1e-6
    ]


def _nodes(ax):
    return [p for p in ax.get_children() if isinstance(p, Ellipse)]


def _markers(ax):
    """Endpoint markers, by shape ('o', 'X' or 'P'); each is drawn once."""
    refs = {}
    for m in "oXP":
        ms = MarkerStyle(m)
        refs[m] = ms.get_path().transformed(ms.get_transform()).vertices
    seen, shapes = set(), []
    for c in ax.collections:
        if id(c) in seen or c.get_alpha() == 0 or not len(c.get_paths()):
            continue
        seen.add(id(c))
        verts = c.get_paths()[0].vertices
        for m, ref in refs.items():
            if verts.shape == ref.shape and np.allclose(verts, ref):
                shapes.append(m)
    return sorted(shapes)


def _texts(ax):
    return [t.get_text() for t in ax.texts]


def _is_line(patch):
    return isinstance(patch.get_arrowstyle(), ArrowStyle.Curve)


def _lagged_graph():
    g = np.zeros((3, 3, 2), dtype=np.int8)
    g[0, 1, 1] = 1  # X(t-1) -> Y(t)
    g[1, 2, 1] = 1  # Y(t-1) -> Z(t)
    return g


def _lag0_string_graph(patt, n=2):
    g = np.full((n, n, 1), "", dtype="<U3")
    g[0, 1, 0] = patt
    g[1, 0, 0] = _reverse_patt(patt)
    return g


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "patt, rev",
    [
        ("-->", "<--"),
        ("<--", "-->"),
        ("o->", "<-o"),
        ("<->", "<->"),
        ("o-o", "o-o"),
        ("x--", "--x"),
        ("+->", "<-+"),
        ("", ""),
    ],
)
def test_reverse_patt(patt, rev):
    assert _reverse_patt(patt) == rev
    assert _reverse_patt(_reverse_patt(patt)) == patt


def test_myround_modes():
    assert _myround(0.33, 0.2) == pytest.approx(0.4)
    assert _myround(0.33, 0.2, "down") == pytest.approx(0.2)
    assert _myround(0.21, 0.2, "up") == pytest.approx(0.4)
    assert _myround(0.33, 0.2, "other") == pytest.approx(0.4)


def test_check_matrices_converts_numeric_graph_to_marks():
    g = np.zeros((3, 3, 2), dtype=int)
    g[0, 1, 0] = 1  # one-way at lag 0: directed
    g[1, 2, 0] = g[2, 1, 0] = 1  # both ways at lag 0: undirected
    g[2, 0, 1] = 1  # lagged
    graph, val, width, attr = _check_matrices(g, None, None, None)
    assert graph.dtype == np.dtype("<U3")
    assert (graph[0, 1, 0], graph[1, 0, 0]) == ("-->", "<--")
    assert (graph[1, 2, 0], graph[2, 1, 0]) == ("o-o", "o-o")
    assert graph[2, 0, 1] == "-->" and graph[0, 2, 1] == ""
    np.testing.assert_array_equal(val, (graph != "").astype(int))
    assert width is None and attr is None


def _sym(shape=(2, 2, 2)):
    return np.zeros(shape)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"graph": "bad"}, "consistent lag-zero"),
        ({"graph": "invalid"}, "Invalid graph entry"),
        ({"val": True}, "val_matrix needs to be symmetric"),
        ({"width": True}, "link_width needs to be symmetric"),
        ({"attr": True}, "link_attribute needs to be symmetric"),
        ({"negative": True}, "link_width must be non-negative"),
    ],
)
def test_check_matrices_validation(kwargs, match):
    g = _lag0_string_graph("-->")
    g = np.concatenate([g, np.full((2, 2, 1), "", dtype="<U3")], axis=2)
    val = width = attr = None
    if kwargs.get("graph") == "bad":
        g[1, 0, 0] = "-->"
    if kwargs.get("graph") == "invalid":
        g[0, 1, 1] = "??>"
    if kwargs.get("val"):
        val = _sym()
        val[0, 1, 0] = 0.5
    if kwargs.get("width"):
        width = _sym()
        width[0, 1, 0] = 1.0
    if kwargs.get("attr"):
        attr = np.full((2, 2, 2), "", dtype=object)
        attr[0, 1, 0] = "spurious"
    if kwargs.get("negative"):
        width = -np.ones((2, 2, 2))
    with pytest.raises(ValueError, match=match):
        _check_matrices(g, val, width, attr)


def test_check_matrices_validates_4d_graphs():
    g = np.full((2, 2, 2, 2), "", dtype="<U3")
    g[0, 1, 1, 0] = "-->"
    g[1, 0, 0, 1] = "<--"
    out, val, _, _ = _check_matrices(g, None, None, None)
    assert out.shape == (2, 2, 2, 2)
    assert val[0, 1, 1, 0] == 1

    bad = g.copy()
    bad[1, 0, 0, 1] = "-->"
    with pytest.raises(ValueError, match="consistent entries"):
        _check_matrices(bad, None, None, None)
    invalid = g.copy()
    invalid[0, 0, 0, 0] = "abc"
    with pytest.raises(ValueError, match="Invalid graph entry"):
        _check_matrices(invalid, None, None, None)
    v = np.zeros((2, 2, 2, 2))
    v[0, 1, 1, 0] = 1.0
    with pytest.raises(ValueError, match="val_matrix needs to have consistent"):
        _check_matrices(g, v, None, None)
    with pytest.raises(ValueError, match="link_width needs to have consistent"):
        _check_matrices(g, None, v, None)
    a = np.full((2, 2, 2, 2), "", dtype=object)
    a[0, 1, 1, 0] = "spurious"
    with pytest.raises(ValueError, match="link_attribute needs to have consistent"):
        _check_matrices(g, None, None, a)


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "layout",
    [
        "circular",
        "spring",
        "kamada_kawai",
        "kk_enhanced",
        "spectral",
        "shell",
        "planar",
        "random",
        "spiral",
        "no-such-layout",
    ],
)
def test_compute_node_positions_layouts_are_normalised(layout):
    G = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    pos = compute_node_positions(G, layout=layout, seed=0)
    assert set(pos) == set(G)
    coords = np.array([pos[n] for n in G], dtype=float)
    assert np.isfinite(coords).all()
    assert np.abs(coords).max() <= 1.0 + 1e-9
    # Distinct nodes get distinct positions.
    assert len({tuple(np.round(c, 6)) for c in coords}) == len(G)


def test_compute_node_positions_unknown_layout_is_circular():
    G = nx.path_graph(4)
    pos = compute_node_positions(G, layout="mystery")
    ref = nx.circular_layout(G)
    for n in G:
        np.testing.assert_allclose(pos[n], ref[n])


def test_compute_node_positions_planar_falls_back_for_k5():
    G = nx.complete_graph(5)  # not planar
    pos = compute_node_positions(G, layout="planar", seed=1)
    assert set(pos) == set(G)


def test_compute_node_positions_callable_node_pos_and_empty():
    G = nx.path_graph(3)

    def line(H, step=1.0):
        return {n: (step * n, 0.0) for n in H}

    raw = compute_node_positions(
        G, layout=line, layout_kwargs={"step": 0.25}, normalize=False
    )
    assert raw == {0: (0.0, 0.0), 1: (0.25, 0.0), 2: (0.5, 0.0)}
    # 'auto' leaves coordinates already inside the unit square alone ...
    assert compute_node_positions(G, layout=line, layout_kwargs={"step": 0.25}) == raw
    # ... and rescales larger ones.
    big = compute_node_positions(G, layout=line, layout_kwargs={"step": 10.0})
    np.testing.assert_allclose([big[n][0] for n in G], [-1.0, 0.0, 1.0])
    forced = compute_node_positions(
        G, layout=line, layout_kwargs={"step": 0.25}, normalize=True
    )
    np.testing.assert_allclose([forced[n][0] for n in G], [-1.0, 0.0, 1.0])

    given = {0: (5.0, 5.0), 1: (6.0, 5.0), 2: (7.0, 5.0)}
    assert compute_node_positions(G, node_pos=given, normalize=False) == given
    assert compute_node_positions(nx.DiGraph(), layout="circular") == {}


def test_normalize_positions_to_unit_square():
    pos = {"a": (0.0, 0.0), "b": (4.0, 1.0), "c": (2.0, 2.0)}
    out = normalize_positions_to_unit_square(pos)
    xy = np.array(list(out.values()))
    assert xy[:, 0].min() == pytest.approx(-1.0) and xy[:, 0].max() == pytest.approx(1)
    assert xy[:, 1].max() == pytest.approx(0.5)  # aspect kept: y span is half

    stretched = normalize_positions_to_unit_square(pos, keep_aspect=False)
    xy = np.array(list(stretched.values()))
    np.testing.assert_allclose(np.abs(xy).max(axis=0), [1.0, 1.0])

    flipped = normalize_positions_to_unit_square(pos, flip_y=True, margin=0.5)
    assert flipped["a"][1] == pytest.approx(-out["a"][1] * 0.5)
    assert flipped["b"][0] == pytest.approx(0.5)

    with pytest.raises(ValueError, match="2D coordinates"):
        normalize_positions_to_unit_square({"a": (0.0, 0.0, 0.0)})


def test_kk_positions_packed_separates_components():
    G = nx.Graph([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
    pos = kk_positions_packed(G, pad=2.5)
    a = np.array([pos[n] for n in (0, 1, 2)])
    b = np.array([pos[n] for n in (3, 4, 5)])
    # Bounding boxes do not overlap along x (components laid out in one row).
    assert a[:, 0].max() < b[:, 0].min() or b[:, 0].max() < a[:, 0].min()


def test_spread_duplicates():
    pos = {"a": (0.0, 0.0), "b": (0.0, 0.0), "c": (1.0, 1.0)}
    out = spread_duplicates(pos, min_sep=0.1)
    assert out["c"] == (1.0, 1.0)
    assert np.hypot(*np.subtract(out["a"], out["b"])) == pytest.approx(0.2)
    assert np.hypot(*out["a"]) == pytest.approx(0.1)


def test_auto_shells():
    assert _auto_shells(nx.Graph()) == [[]]
    path = nx.path_graph(3)
    assert _auto_shells(path) == [[1], [0], [2]]  # by degree, hub first
    # K4 on 0-3 plus a tail 3-4-5: degrees 3, 3, 3, 4, 2, 1.
    G = nx.complete_graph(4)
    G.add_edges_from([(3, 4), (4, 5)])
    shells = _auto_shells(G)
    assert sorted(n for s in shells for n in s) == list(range(6))
    assert 3 in shells[0] and 5 in shells[-1] and len(shells) > 1
    # Degenerate quantiles (a star: five leaves of degree 1) give one shell.
    star = _auto_shells(nx.star_graph(5))
    assert len(star) == 1 and sorted(star[0]) == list(range(6))


def test_layout_kwargs_defaults_and_merge():
    G = nx.path_graph(4)
    spring = _default_kwargs_for_layout(G, "spring", seed=3)
    assert spring == {"k": 0.5, "iterations": 100, "seed": 3}
    assert _default_kwargs_for_layout(G, "dot", None)["prog"] == "dot"
    assert _default_kwargs_for_layout(G, "graphviz", None)["prog"] == "neato"
    assert _default_kwargs_for_layout(G, "unknown", None) == {}

    base = {"k": 0.5, "seed": 3}
    assert _merge_layout_kwargs(base, None, "spring") == base
    assert _merge_layout_kwargs(base, {"k": 1.0}, "spring") == {"k": 1.0, "seed": 3}
    nested = {"spring": {"seed": 9}, "kk": {"weight": None}}
    assert _merge_layout_kwargs(base, nested, "spring") == {"k": 0.5, "seed": 9}


# ---------------------------------------------------------------------------
# plot_graph
# ---------------------------------------------------------------------------


def test_plot_graph_draws_nodes_edges_and_lag_labels():
    fig, ax = plot_graph(_lagged_graph(), var_names=NAMES, node_layout="circular")
    assert len(_nodes(ax)) == 3
    assert {"X", "Y", "Z"} <= set(_texts(ax))
    arrows = _arrows(ax)
    assert len(arrows) == 2
    assert not any(_is_line(a) for a in arrows)
    # Each lagged edge is labelled with its lag.
    assert _texts(ax).count("1") == 2
    # No val_matrix: plain black edges and no colorbars.
    for a in arrows:
        assert a.get_facecolor() == to_rgba("black")
    assert ax.child_axes == []


def test_plot_graph_lag_labels_ordered_by_strength_and_lag_array():
    g = np.zeros((2, 2, 3), dtype=int)
    g[0, 1, 1] = g[0, 1, 2] = 1
    val = np.zeros((2, 2, 3))
    val[0, 1, 1], val[0, 1, 2] = 0.2, -0.7
    _, ax = plot_graph(g, val_matrix=val, node_layout="circular")
    assert "2,1" in _texts(ax)  # strongest lag first
    _, ax = plot_graph(
        g, val_matrix=val, node_layout="circular", lag_array=np.array([0, 5, 10])
    )
    assert "10,5" in _texts(ax)


@pytest.mark.parametrize(
    "patt, n_arrows, line, markers",
    [
        ("-->", 1, False, []),
        ("<--", 1, False, []),
        ("<->", 2, False, []),
        ("o->", 1, False, ["o"]),
        ("o-o", 1, True, ["o", "o"]),
        ("---", 1, True, []),
        ("x-x", 1, True, ["X", "X"]),
        ("o-x", 1, True, ["X", "o"]),
        ("x->", 1, False, ["X"]),
        ("+->", 1, False, ["P"]),
    ],
)
@pytest.mark.parametrize("lag", [0, 1])
def test_plot_graph_edge_marks(patt, n_arrows, line, markers, lag):
    if lag == 0:
        g = _lag0_string_graph(patt)
    else:
        g = np.full((2, 2, 2), "", dtype="<U3")
        g[0, 1, 1] = patt
    _, ax = plot_graph(g, var_names=["A", "B"], node_layout="circular")
    arrows = _arrows(ax)
    assert len(arrows) == n_arrows
    assert all(_is_line(a) == line for a in arrows)
    assert _markers(ax) == markers


def test_plot_graph_lag0_arrow_direction_follows_mark():
    pos = {"x": np.array([-1.0, 1.0]), "y": np.array([0.0, 0.0])}
    _, ax = plot_graph(_lag0_string_graph("-->"), node_pos=pos)
    (fwd,) = _arrows(ax)
    _, ax = plot_graph(_lag0_string_graph("<--"), node_pos=pos)
    (back,) = _arrows(ax)
    # The arrow runs from patchA (tail) to patchB (head).
    assert (fwd.patchA.center[0], fwd.patchB.center[0]) == (-1.0, 1.0)
    assert (back.patchA.center[0], back.patchB.center[0]) == (1.0, -1.0)


def test_plot_graph_val_matrix_colours_links_and_nodes():
    g = np.zeros((2, 2, 2), dtype=int)
    g[0, 1, 1] = 1
    g[0, 0, 1] = 1  # auto-dependency drives the node colour
    val = np.zeros((2, 2, 2))
    val[0, 1, 1] = 0.6
    val[0, 0, 1] = -0.8
    fig, ax = plot_graph(
        g, val_matrix=val, var_names=["A", "B"], node_layout="circular"
    )
    cmap = plt.get_cmap("RdBu_r")
    (arrow,) = _arrows(ax)
    np.testing.assert_allclose(arrow.get_facecolor(), cmap((0.6 + 1) / 2))
    node_colors = [n.get_facecolor() for n in _nodes(ax)]
    assert any(np.allclose(c, cmap((-0.8 + 1) / 2)) for c in node_colors)
    # Link and node colorbars, labelled with the defaults.
    labels = {cax.get_xlabel() for cax in ax.child_axes}
    assert labels == {"MCI", "auto-MCI"}

    fig, ax = plot_graph(g, val_matrix=val, node_layout="circular", show_colorbar=False)
    assert ax.child_axes == []


def test_plot_graph_link_attribute_and_width():
    g = np.zeros((3, 3, 2), dtype=int)
    g[0, 1, 1] = g[1, 2, 1] = 1
    attr = np.full((3, 3, 2), "", dtype=object)
    attr[0, 1, 1] = "spurious"
    width = np.zeros((3, 3, 2))
    width[0, 1, 1], width[1, 2, 1] = 1.0, 2.0
    _, ax = plot_graph(
        g,
        var_names=NAMES,
        node_layout="circular",
        link_attribute=attr,
        link_width=width,
        arrow_linewidth=3.0,
        edge_color="steelblue",
    )
    arrows = _arrows(ax)
    colors = sorted(tuple(a.get_facecolor()) for a in arrows)
    assert colors == sorted([to_rgba("grey"), to_rgba("steelblue")])
    scales = sorted(a.get_mutation_scale() for a in arrows)
    assert scales == pytest.approx([1.5, 3.0])


def test_plot_graph_target_node_colours():
    g = np.zeros((3, 3, 2), dtype=int)
    g[0, 1, 1] = 1  # into Y
    g[1, 2, 1] = 1  # out of Y
    pos = {"x": np.array([0.0, 1.0, 2.0]), "y": np.zeros(3)}
    _, ax = plot_graph(
        g,
        var_names=NAMES,
        node_pos=pos,
        target_node="Y",
        target_in_color="blue",
        target_out_color="red",
        target_node_color="green",
    )
    by_start = {round(a.patchA.center[0]): a.get_facecolor() for a in _arrows(ax)}
    assert by_start[0] == to_rgba("blue")  # X -> Y
    assert by_start[1] == to_rgba("red")  # Y -> Z
    fills = {round(n.center[0]): n.get_facecolor() for n in _nodes(ax)}
    assert fills[1] == to_rgba("green")
    assert fills[0] == to_rgba("lightgrey")

    # Both ends targeted: the "between" colour wins; integer targets work too.
    _, ax = plot_graph(
        g, node_pos=pos, target_node=[0, 1], target_between_color="purple"
    )
    colors = {tuple(a.get_facecolor()) for a in _arrows(ax)}
    assert to_rgba("purple") in colors

    with pytest.raises(ValueError, match="target_node 'W' not found"):
        plot_graph(g, var_names=NAMES, node_pos=pos, target_node="W")


def test_plot_graph_special_nodes_node_pos_and_return_pos():
    pos = {"x": np.array([0.0, 1.0, 2.0]), "y": np.array([0.0, 1.0, 0.0])}
    fig, ax, out = plot_graph(
        _lagged_graph(),
        var_names=NAMES,
        node_pos=pos,
        special_nodes={("Z", 0): "orange"},
        return_pos=True,
    )
    np.testing.assert_allclose(out["x"], pos["x"])
    np.testing.assert_allclose(out["y"], pos["y"])
    centers = {tuple(n.center): n.get_facecolor() for n in _nodes(ax)}
    assert centers[(2.0, 0.0)] == to_rgba("orange")
    assert centers[(0.0, 0.0)] == to_rgba("lightgrey")

    # Computed layout positions round-trip too.
    _, _, auto = plot_graph(_lagged_graph(), node_layout="circular", return_pos=True)
    assert auto["x"].shape == (3,) and np.abs(auto["x"]).max() <= 1.0


def test_plot_graph_autodependency_lags_text():
    g = np.zeros((2, 2, 3), dtype=int)
    g[0, 0, 1] = g[0, 0, 2] = 1
    g[0, 1, 1] = 1
    _, ax = plot_graph(g, node_layout="circular", show_autodependency_lags=True)
    assert "\n\n\n1,2" in _texts(ax)


def test_plot_graph_figure_options(tmp_path):
    fig, ax = plt.subplots()
    out = plot_graph(_lagged_graph(), fig_ax=(fig, ax), node_layout="circular")
    assert out == (fig, ax)

    _, ax = plot_graph(_lagged_graph(), node_layout="circular", title="My graph")
    assert ax.get_title() == "My graph"

    path = tmp_path / "g.png"
    assert plot_graph(_lagged_graph(), node_layout="circular", save_name=path) is None
    assert path.stat().st_size > 0


def test_plot_graph_rejects_link_matrix_and_4d():
    with pytest.raises(ValueError, match="link_matrix is deprecated"):
        plot_graph(_lagged_graph(), link_matrix=_lagged_graph())
    g4 = np.full((2, 2, 2, 2), "", dtype="<U3")
    with pytest.raises(ValueError, match="plot_time_series_graph instead"):
        plot_graph(g4)


@pytest.mark.parametrize("kind", ["empty", "self_loops_only"])
def test_plot_graph_without_cross_links_draws_no_edges(kind):
    g = np.zeros((3, 3, 2), dtype=int)
    if kind == "self_loops_only":
        g[0, 0, 1] = g[2, 2, 1] = 1
    _, ax = plot_graph(g, var_names=NAMES, node_layout="circular")
    assert len(_nodes(ax)) == 3
    assert _arrows(ax) == []


def test_plot_graph_lag0_only_and_2d_graphs():
    g = np.zeros((3, 3, 1), dtype=int)
    g[0, 1, 0] = 1
    _, ax = plot_graph(g, var_names=NAMES, node_layout="circular")
    assert len(_arrows(ax)) == 1
    assert "1" not in _texts(ax)  # contemporaneous links carry no lag label

    # A plain (N, N) adjacency matrix is treated as a lag-0 graph.
    _, ax = plot_graph(g[:, :, 0], var_names=NAMES, node_layout="circular")
    assert len(_arrows(ax)) == 1


# ---------------------------------------------------------------------------
# plot_time_series_graph
# ---------------------------------------------------------------------------


def test_time_series_graph_unrolls_edges_over_lags():
    g = np.full((3, 3, 3), "", dtype="<U3")
    g[0, 1, 1] = "-->"  # lag 1: drawn in 3 - 1 = 2 columns
    g[1, 2, 0] = "-->"  # lag 0: drawn in all 3 columns
    g[2, 1, 0] = "<--"
    g[2, 0, 2] = "-->"  # lag 2: drawn once
    fig, ax = plot_time_series_graph(g, var_names=NAMES)
    assert len(_nodes(ax)) == 3 * 3
    assert len(_arrows(ax)) == 2 + 3 + 1
    texts = _texts(ax)
    assert {"X", "Y", "Z"} <= set(texts)
    assert {"$t$", "$t-1$", "$t-2$"} <= set(texts)
    assert ax.child_axes == []  # no colorbar without val_matrix


def test_time_series_graph_lag_array_title_and_save(tmp_path):
    g = np.zeros((2, 2, 3), dtype=int)
    g[0, 1, 1] = 1
    _, ax = plot_time_series_graph(g, lag_array=[0, 7, 14], title="TSG", title_pad=3.0)
    assert {"$t$", "$t-7$", "$t-14$"} <= set(_texts(ax))
    assert ax.get_title() == "TSG"

    path = tmp_path / "tsg.png"
    assert plot_time_series_graph(g, save_name=path) is None
    assert path.stat().st_size > 0


def test_time_series_graph_order_controls_rows():
    g = np.zeros((3, 3, 2), dtype=int)
    g[0, 1, 1] = 1
    # A swap reads the same whether `order` lists rows or maps variables to
    # rows (see the 3-cycle note in the report).
    _, ax = plot_time_series_graph(g, var_names=NAMES, order=[1, 0, 2])
    y = {t.get_text(): t.get_position()[1] for t in ax.texts if t.get_text() in NAMES}
    assert y["Y"] > y["X"] > y["Z"]
    # Each label sits on its variable's row of nodes.
    rows = sorted({round(n.center[1], 6) for n in _nodes(ax)}, reverse=True)
    assert sorted(round(v, 6) for v in y.values()) == sorted(rows)

    with pytest.raises(ValueError, match="permutation of range"):
        plot_time_series_graph(g, order=[0, 0, 1])
    with pytest.raises(ValueError, match="link_matrix is deprecated"):
        plot_time_series_graph(g, link_matrix=g)


def test_time_series_graph_val_matrix_width_attribute_special_nodes():
    g = np.zeros((2, 2, 2), dtype=int)
    g[0, 1, 1] = 1
    g[1, 0, 1] = 1
    val = np.zeros((2, 2, 2))
    val[0, 1, 1] = 0.5
    val[1, 0, 1] = -0.5
    attr = np.full((2, 2, 2), "", dtype=object)
    attr[1, 0, 1] = "spurious"
    width = np.zeros((2, 2, 2))
    width[0, 1, 1], width[1, 0, 1] = 2.0, 1.0
    fig, ax = plot_time_series_graph(
        g,
        val_matrix=val,
        link_width=width,
        link_attribute=attr,
        arrow_linewidth=4,
        special_nodes={(0, -1): "red", (1, -5): "blue"},  # lag -5 is off-grid
    )
    arrows = _arrows(ax)
    cmap = plt.get_cmap("RdBu_r")
    colors = [tuple(a.get_facecolor()) for a in arrows]
    assert to_rgba("grey") in colors
    assert any(np.allclose(c, cmap(0.75)) for c in colors)
    assert sorted(a.get_mutation_scale() for a in arrows) == pytest.approx([2.0, 4.0])
    assert [cax.get_xlabel() for cax in ax.child_axes] == ["MCI"]
    reds = [n for n in _nodes(ax) if n.get_facecolor() == to_rgba("red")]
    assert len(reds) == 1
    assert not any(n.get_facecolor() == to_rgba("blue") for n in _nodes(ax))


def test_time_series_graph_space_context_nodes_are_merged():
    g = np.zeros((2, 2, 3), dtype=int)
    g[1, 0, 0] = 1
    _, ax = plot_time_series_graph(
        g, node_classification={0: "system", 1: "space_context"}, order=[0, 1]
    )
    wide = [p for p in ax.get_children() if isinstance(p, FancyBboxPatch)]
    assert len(wide) == 1  # one box spanning the context node's row


def test_time_series_graph_empty_draws_no_edges():
    _, ax = plot_time_series_graph(np.zeros((2, 2, 2), dtype=int))
    assert len(_nodes(ax)) == 4
    assert _arrows(ax) == []


def test_time_series_graph_4d_with_explicit_order():
    g = np.full((2, 2, 2, 2), "", dtype="<U3")
    g[0, 1, 1, 0] = "-->"  # X(t-1) -> Y(t)
    g[1, 0, 0, 1] = "<--"
    _, ax = plot_time_series_graph(g, order=[0, 1])
    assert len(_nodes(ax)) == 4
    assert len(_arrows(ax)) == 1


# ---------------------------------------------------------------------------
# Lag-function matrix
# ---------------------------------------------------------------------------


def test_setup_matrix_add_lagfuncs_plots_values_and_thresholds():
    rng = np.random.default_rng(0)
    val = rng.uniform(-0.9, 0.9, size=(2, 2, 4))
    sig = np.full((2, 2, 4), 0.3)
    conf = np.stack([val - 0.1, val + 0.1], axis=-1)
    m = setup_matrix(N=2, tau_max=3, var_names=["a", "b"], plot_gridlines=True)
    assert set(m.axes_dict) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    m.add_lagfuncs(val, sig_thres=sig, conf_matrix=conf, label="MCI", color="red")

    ax = m.axes_dict[(1, 0)]
    ys = [line.get_ydata() for line in ax.lines]
    assert any(np.allclose(y, val[1, 0]) for y in ys)
    assert any(np.allclose(y, conf[1, 0, :, 0]) for y in ys)
    assert any(np.allclose(y, conf[1, 0, :, 1]) for y in ys)
    assert any(np.allclose(y, 0.3) for y in ys)
    assert any(np.allclose(y, -0.3) for y in ys)
    assert ax.get_xlim() == (0.0, 3.0)

    fig, axes = m.savefig()
    assert axes is m.axes_dict
    legend_texts = [
        t.get_text()
        for a in fig.axes
        if a.get_legend()
        for t in a.get_legend().get_texts()
    ]
    assert legend_texts == ["MCI"]
    assert "lag $\\tau$" in [t.get_text() for t in fig.texts]


def test_setup_matrix_one_sided_threshold_and_no_label():
    val = np.zeros((1, 1, 3))
    m = setup_matrix(N=1, tau_max=2)
    m.add_lagfuncs(val, sig_thres=np.full((1, 1, 3), 0.2), two_sided_thres=False)
    ys = [line.get_ydata() for line in m.axes_dict[(0, 0)].lines]
    assert not any(np.allclose(y, -0.2) for y in ys)
    fig, _ = m.savefig()
    assert all(a.get_legend() is None for a in fig.axes)
    assert fig.legends == []


def test_setup_matrix_bottom_legend_titles_and_lag_array(tmp_path):
    m = setup_matrix(
        N=2,
        tau_max=2,
        var_names=["a", "b"],
        row_title="causes",
        col_title="targets",
        legend_loc="bottom",
        lag_units="days",
        lag_label_suffix=" of the cause",
        lag_array=np.array([0, 5, 10]),
        x_base=1,
    )
    m.add_lagfuncs(np.zeros((2, 2, 3)), label="score")
    fig, axes = m.savefig()
    fig_texts = [t.get_text() for t in fig.texts]
    assert "causes" in fig_texts and "targets" in fig_texts
    assert "lag $\\tau$ of the cause [days]" in fig_texts
    assert len(fig.legends) == 1
    assert [t.get_text() for t in fig.legends[0].get_texts()] == ["score"]
    assert [t.get_text() for t in axes[(0, 0)].get_xticklabels()] == ["0", "5", "10"]

    path = tmp_path / "lags.png"
    assert m.savefig(name=path) is None
    assert path.stat().st_size > 0


def test_plot_lagfuncs_wrapper(tmp_path):
    val = np.zeros((2, 2, 3))
    val[0, 1, 1] = 0.7
    m = plot_lagfuncs(val, setup_args={"var_names": ["p", "q"]})
    assert isinstance(m, setup_matrix)
    assert m.tau_max == 2
    ax = m.axes_dict[(0, 1)]
    assert any(np.allclose(line.get_ydata(), val[0, 1]) for line in ax.lines)

    path = tmp_path / "lagfuncs.png"
    plot_lagfuncs(val, name=path)
    assert path.stat().st_size > 0


@pytest.mark.parametrize(
    "where, x_pos, y_pos",
    [
        (["left", "bottom"], "bottom", "left"),
        (["right", "top"], "top", "right"),
    ],
)
def test_make_nice_axes_positions_ticks(where, x_pos, y_pos):
    fig, ax = plt.subplots()
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 1, 2, 3])
    _make_nice_axes(ax, where=where, skip=2)
    assert ax.xaxis.get_ticks_position() == x_pos
    assert ax.yaxis.get_ticks_position() == y_pos
    np.testing.assert_array_equal(ax.get_xticks(), [0, 2])
    np.testing.assert_array_equal(ax.get_yticks(), [0, 2])
    hidden = {"left", "right", "top", "bottom"} - set(where)
    for loc in hidden:
        assert ax.spines[loc].get_edgecolor()[3] == 0.0


def test_make_nice_axes_without_axes():
    fig, ax = plt.subplots()
    _make_nice_axes(ax, where=[])
    assert all(t.get_text() == "" for t in ax.get_xticklabels())
    assert all(t.get_text() == "" for t in ax.get_yticklabels())


def test_time_series_graph_auto_order_handles_lag0_only_and_4d_graphs():
    # Regression: the default order="auto" crashed on (N, N, 1) graphs, which
    # are squeezed to 2-D before ordering, and on 4-D aux graphs.
    g = np.zeros((3, 3, 1), dtype=int)
    g[0, 1, 0] = 1
    _, ax = plot_time_series_graph(g, var_names=NAMES)
    assert len(_nodes(ax)) == 3
    assert len(_arrows(ax)) == 1
    assert "$t$" in _texts(ax)

    g4 = np.full((2, 2, 2, 2), "", dtype="<U3")
    g4[0, 1, 1, 0] = "-->"
    g4[1, 0, 0, 1] = "<--"
    _, ax = plot_time_series_graph(g4)
    assert len(_nodes(ax)) == 4
    assert len(_arrows(ax)) == 1


def test_barycenter_order_accepts_2d_3d_and_4d_graphs():
    from causalts.plotting._core import _barycenter_order

    g = np.zeros((4, 4, 1), dtype=bool)
    g[0, 2, 0] = g[1, 3, 0] = True
    order = _barycenter_order(g)
    assert sorted(order) == [0, 1, 2, 3]
    assert _barycenter_order(g[:, :, 0]) == order

    g4 = np.zeros((4, 4, 1, 1), dtype=bool)
    g4[:, :, 0, 0] = g[:, :, 0]
    assert _barycenter_order(g4) == order


def test_random_layout_is_seeded():
    # Regression: the random layout was always given scale=, which
    # nx.random_layout rejects, and ignored the seed.
    G = nx.path_graph(5)
    a = compute_node_positions(G, layout="random", seed=7)
    b = compute_node_positions(G, layout="random", seed=7)
    c = compute_node_positions(G, layout="random", seed=8)
    assert all(np.allclose(a[n], b[n]) for n in G)
    assert not all(np.allclose(a[n], c[n]) for n in G)
    assert _default_kwargs_for_layout(G, "random", 7) == {"center": None, "seed": 7}


def test_setup_matrix_lag_array_with_default_x_base():
    # Regression: the default x_base=1.0 is a float, and slicing lag_array
    # with it raised TypeError in savefig.
    m = setup_matrix(N=1, tau_max=2, lag_array=np.array([0, 3, 6]))
    m.add_lagfuncs(np.zeros((1, 1, 3)))
    _, axes = m.savefig()
    assert [t.get_text() for t in axes[(0, 0)].get_xticklabels()] == ["0", "3", "6"]


@pytest.mark.parametrize("x_base", [0.5, 1.5, 0])
def test_setup_matrix_lag_array_rejects_fractional_x_base(x_base):
    with pytest.raises(ValueError, match="positive whole number"):
        setup_matrix(N=1, tau_max=2, x_base=x_base, lag_array=np.array([0, 3, 6]))
