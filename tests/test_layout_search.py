# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the edge-aware ``auto`` node layout."""

import matplotlib

matplotlib.use("Agg")

import networkx as nx
import numpy as np
import pytest

from causalts.plotting import _layout_search as ls
from causalts.plotting._layout_search import _Scorer, _to_box, auto_layout, layout_score


def _demo_graph():
    # neato puts node 0 on the line between 2 and 3, so 2 -> 3 brushes past it.
    edges = [
        (0, 1),
        (0, 3),
        (1, 0),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 5),
        (3, 1),
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 5),
        (5, 1),
        (5, 4),
    ]
    G = nx.DiGraph()
    G.add_nodes_from(range(6))
    G.add_edges_from(edges)
    return G


def test_incremental_scoring_matches_a_full_rescore():
    rng = np.random.default_rng(0)
    G = nx.gnm_random_graph(12, 20, seed=3, directed=True)
    arcs = np.array(G.edges())
    scorer = _Scorer(_to_box(rng.normal(size=(12, 2))), arcs, 0.2, 0.2)
    for _ in range(100):
        moved = rng.choice(12, rng.integers(1, 3), replace=False)
        new_xy = np.clip(scorer.P[moved] + rng.normal(0, 0.2, (len(moved), 2)), -1, 1)
        score, state = scorer.propose(moved, new_xy)
        scorer.accept(state)
        assert score == pytest.approx(_Scorer(scorer.P, arcs, 0.2, 0.2).total())


def test_auto_layout_beats_its_starts():
    G = _demo_graph()
    pos = auto_layout(G, seed=0)
    assert set(pos) == set(G.nodes)
    coords = np.array(list(pos.values()))
    assert np.abs(coords).max() == pytest.approx(1.0)
    score = layout_score(pos, G)[0]
    kk = nx.kamada_kawai_layout(G.to_undirected())
    kk = dict(zip(kk, _to_box(np.array(list(kk.values())))))
    assert score <= layout_score(kk, G)[0]
    assert layout_score(pos, G)[2] == 0  # no edge brushes past a node


def test_auto_layout_is_deterministic():
    G = nx.gnm_random_graph(10, 15, seed=1, directed=True)
    assert auto_layout(G, seed=3) == auto_layout(G, seed=3)


def test_auto_layout_without_graphviz(monkeypatch, recwarn):
    monkeypatch.setattr(ls, "_graphviz_available", lambda: False)
    pos = auto_layout(_demo_graph(), seed=0)
    assert len(pos) == 6
    assert not [w for w in recwarn if "Graphviz" in str(w.message)]


@pytest.mark.parametrize("n", [0, 1, 2])
def test_auto_layout_tiny_graphs(n):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    assert len(auto_layout(G)) == n


def test_auto_layout_edgeless_graph():
    G = nx.empty_graph(5, create_using=nx.DiGraph)
    pos = auto_layout(G)
    assert len(pos) == 5


def test_compute_node_positions_auto():
    from causalts.plotting._core import compute_node_positions

    pos = compute_node_positions(_demo_graph(), layout="auto", seed=0)
    assert len(pos) == 6
    assert np.abs(np.array(list(pos.values()))).max() <= 1 + 1e-9


def test_compare_graphs_auto_layout():
    import matplotlib.pyplot as plt

    from causalts.plotting import compare_graphs

    gt = nx.to_numpy_array(_demo_graph(), dtype=int)[:, :, None]
    _, ax, pos = compare_graphs(gt, gt, node_layout="auto", return_pos=True)
    assert len(pos["x"]) == 6
    plt.close("all")


def test_crossing_at_a_sample_point_counts():
    # Straight diagonals of the box meet exactly at their midpoint sample.
    G = nx.DiGraph([(0, 1), (2, 3)])
    pos = {0: (-1, -1), 1: (1, 1), 2: (-1, 1), 3: (1, -1)}
    assert layout_score(pos, G, curved_radius=0.0)[1] == 1


def test_node_between_samples_is_a_near_pass():
    # A long straight edge with a node on it, halfway between two samples.
    G = nx.DiGraph([(0, 1)])
    G.add_node(2)
    pos = {0: (-1.0, 0.0), 1: (1.0, 0.0), 2: (-1.0 + 2.0 / 16, 0.0)}
    assert layout_score(pos, G, curved_radius=0.0, node_size=0.05)[2] == 1


def test_near_tolerance_never_below_the_node_radius():
    scorer = _Scorer(np.zeros((80, 2)), np.zeros((0, 2), dtype=int), 0.2, 0.2)
    assert scorer.near_tol >= 0.1


def test_seed_in_layout_kwargs_is_not_passed_twice():
    from causalts.plotting._core import compute_node_positions

    G = _demo_graph()
    a = compute_node_positions(G, layout="auto", seed=0, layout_kwargs={"seed": 3})
    assert a == compute_node_positions(G, layout="auto", seed=3)


def test_plot_graph_scores_its_own_geometry(monkeypatch):
    import matplotlib.pyplot as plt

    from causalts.plotting import _layout_search, plot_graph

    seen = {}
    real = _layout_search.auto_layout

    def spy(G, **kw):
        seen.update(kw)
        return real(G, **kw)

    monkeypatch.setattr(_layout_search, "auto_layout", spy)
    graph = np.full((3, 3, 2), "", dtype="<U3")
    graph[0, 1, 0], graph[1, 0, 0] = "-->", "<--"  # contemporaneous
    graph[1, 2, 1] = "-->"  # lagged
    plot_graph(graph=graph, node_layout="auto", curved_radius=0.3, node_size=0.15)
    plt.close("all")
    arcs = {(a, b): r for a, b, r in seen["arcs"]}
    assert arcs[(0, 1)] == 0.0  # straight
    assert arcs[(1, 2)] == pytest.approx(-0.3)  # lagged links bend with -radius
    assert seen["node_size"] == 0.15


def test_compare_graphs_scores_every_stroke(monkeypatch):
    import matplotlib.pyplot as plt

    from causalts.plotting import _layout_search, compare_graphs

    seen = {}
    real = _layout_search.auto_layout

    def spy(G, **kw):
        seen.update(kw)
        return real(G, **kw)

    monkeypatch.setattr(_layout_search, "auto_layout", spy)
    gt = np.zeros((3, 3, 3), dtype=np.int8)
    gt[0, 1, 1] = 1  # missed at lag 1 ...
    disc = np.zeros_like(gt)
    disc[0, 1, 2] = 1  # ... found (spuriously) at lag 2: two strokes 0 -> 1
    compare_graphs(gt, disc, node_layout="auto")
    plt.close("all")
    radii = sorted(r for a, b, r in seen["arcs"] if (a, b) == (0, 1))
    assert radii == pytest.approx([-0.2, 0.2])


def test_drawn_arcs_keep_bidirected_marks_in_place():
    from causalts.plotting._core import _drawn_arcs

    G = nx.DiGraph()
    G.add_edge(0, 1, outer_edge=True, outer_edge_type="<->", inner_edge="")
    G.add_edge(1, 2, outer_edge=True, outer_edge_type="<--", inner_edge="")
    arcs = _drawn_arcs(G, 0.2)
    assert (0, 1, -0.2) in arcs  # not reversed: the renderer keeps <-> as is
    assert (2, 1, -0.2) in arcs


def test_wide_nodes_count_as_near_passes():
    # A vertical edge passing 0.3 to the right of node 2: clear of a round
    # node of size 0.2, but inside one drawn four times as wide.
    G = nx.DiGraph([(0, 1)])
    G.add_node(2)
    pos = {0: (0.3, -1.0), 1: (0.3, 1.0), 2: (0.0, 0.0)}
    assert layout_score(pos, G, curved_radius=0.0)[2] == 0
    assert layout_score(pos, G, curved_radius=0.0, node_aspect=4.0)[2] == 1


def test_neato_start_falls_back_to_pydot(monkeypatch):
    from networkx.drawing import nx_agraph, nx_pydot

    calls = []

    def broken(*a, **k):
        raise ImportError("requires pygraphviz")

    def fake_pydot(U, prog):
        calls.append(prog)
        return nx.circular_layout(U)

    monkeypatch.setattr(ls, "_graphviz_available", lambda: True)
    monkeypatch.setattr(nx_agraph, "graphviz_layout", broken)
    monkeypatch.setattr(nx_pydot, "graphviz_layout", fake_pydot)
    auto_layout(_demo_graph(), seed=0)
    assert calls == ["neato"]


@pytest.mark.parametrize("name", ["Auto", " auto "])
def test_auto_name_is_case_insensitive_in_compare_graphs(monkeypatch, name):
    import matplotlib.pyplot as plt

    from causalts.plotting import _layout_search, compare_graphs

    seen = {}
    real = _layout_search.auto_layout

    def spy(G, **kw):
        seen.update(kw)
        return real(G, **kw)

    monkeypatch.setattr(_layout_search, "auto_layout", spy)
    gt = nx.to_numpy_array(_demo_graph(), dtype=int)[:, :, None]
    compare_graphs(gt, gt, node_layout=name)
    plt.close("all")
    assert "arcs" in seen


def test_seed_none_is_allowed():
    assert len(auto_layout(_demo_graph(), seed=None)) == 6


def test_wide_nodes_side_by_side_are_crowded():
    # Centres 0.6 apart: clear for round nodes of size 0.2, overlapping when
    # the nodes are drawn four times as wide.
    G = nx.DiGraph()
    G.add_nodes_from(range(3))
    pos = {0: (-0.3, 0.0), 1: (0.3, 0.0), 2: (0.0, 1.0)}
    round_score = layout_score(pos, G)[0]
    wide_score = layout_score(pos, G, node_aspect=4.0)[0]
    assert wide_score > round_score


def test_neato_start_matches_the_default_neato(monkeypatch):
    from networkx.drawing import nx_agraph

    seen = {}

    def fake(U, prog, **kw):
        seen.update(kw, prog=prog)
        return nx.circular_layout(U)

    monkeypatch.setattr(ls, "_graphviz_available", lambda: True)
    monkeypatch.setattr(nx_agraph, "graphviz_layout", fake)
    auto_layout(_demo_graph(), seed=0)
    assert seen == {"prog": "neato", "args": "-Goverlap=false"}


def test_crossing_rows_match_across_chunks():
    rng = np.random.default_rng(1)
    G = nx.gnm_random_graph(30, 90, seed=2, directed=True)
    scorer = _Scorer(_to_box(rng.normal(size=(30, 2))), np.array(G.edges()), 0.2, 0.2)
    full = scorer._cross_rows(np.arange(len(scorer.arcs)))
    halves = np.vstack(
        [scorer._cross_rows(np.arange(45)), scorer._cross_rows(np.arange(45, 90))]
    )
    np.testing.assert_array_equal(full, halves)


def test_curves_sharing_a_node_can_still_cross():
    # 0 -> 1 bows below the diagonal and cuts through 0 -> 2's path.
    G = nx.DiGraph([(0, 1), (0, 2)])
    pos = {0: (-1.0, -1.0), 1: (1.0, 1.0), 2: (0.0, -0.25)}
    assert layout_score(pos, G, curved_radius=0.4)[1] == 1


def test_curves_sharing_a_node_without_crossing():
    G = nx.DiGraph([(0, 1), (0, 2)])
    pos = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0)}
    assert layout_score(pos, G, curved_radius=0.2)[1] == 0


def test_crossing_just_outside_a_shared_node_counts():
    # 0 -> 1 and 0 -> 2 cross about 0.22 from node 0, outside its 0.075 radius.
    G = nx.DiGraph([(0, 1), (0, 2)])
    pos = {0: (-1.0, 0.0), 1: (1.0, 0.0), 2: (0.2, -0.03)}
    assert layout_score(pos, G, curved_radius=0.2, node_size=0.15)[1] == 1


def test_drawn_arcs_skip_the_invisible_placeholder():
    from causalts.plotting._core import _drawn_arcs

    G = nx.DiGraph()
    G.add_edge(0, 1, inner_edge="xxx", outer_edge=False, no_links=True)
    assert _drawn_arcs(G, 0.2) == []
