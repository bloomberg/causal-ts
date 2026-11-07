# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Contemporaneous background knowledge in CDNOTS and CDNOTS+.

True model ``A(t) -> B(t)`` plus an independent ``D``. Without knowledge the
lag-0 edge is unoriented (``o-o``); CDNOTS reports both directions and CDNOTS+
drops it from ``cg_tig``.
"""

import numpy as np
import pandas as pd
import pytest

from causalts import run_cdnots
from causalts.cdnots.ancestral import AncestralKnowledge
from causalts.cdnots.phase3_utils import make_background_knowledge, run_cdnots_plus
from causalts.ci_tests import ParCorrGPU

RUNNERS = {"cdnots": run_cdnots, "cdnots+": run_cdnots_plus}


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(0)
    T = 600
    a = rng.standard_normal(T)
    b = 0.8 * a + rng.standard_normal(T)
    return pd.DataFrame({"A": a, "B": b, "D": rng.standard_normal(T)})


def _run(df, algo, **kw):
    ci = ParCorrGPU(np.zeros((2, 2)), device="cpu")
    return RUNNERS[algo](df, ci, num_lags=1, include_C=False, **kw)


def _ab(res):
    return res.to_marks()[0, 1, 0], res.cg_tig[0, 1, 0], res.cg_tig[1, 0, 0]


@pytest.mark.parametrize("algo", RUNNERS)
def test_unconstrained_edge_is_unoriented(df, algo):
    assert _ab(_run(df, algo))[0] == "o-o"


@pytest.mark.parametrize("algo", RUNNERS)
@pytest.mark.parametrize("via", ["knowledge", "background_knowledge"])
def test_required_lag0_edge_is_oriented(df, algo, via):
    """A required A -> B stayed o-o; CDNOTS+ then dropped it."""
    if via == "knowledge":
        kw = {"knowledge": AncestralKnowledge().add_required_edge("A", "B", 0, 0)}
    else:
        bk = make_background_knowledge(
            ["A", "B", "D"], 1, include_C=False, required=[("A", "B", 0, 0)]
        )
        kw = {"background_knowledge": bk}
    assert _ab(_run(df, algo, **kw)) == ("-->", 1, 0)


@pytest.mark.parametrize("algo", RUNNERS)
def test_lag0_edge_forbidden_both_ways_is_removed(df, algo):
    """CDNOTS+'s MCI skeleton never checked contemporaneous forbids."""
    k = AncestralKnowledge().add_forbidden_edge("A", "B", 0, 0)
    k.add_forbidden_edge("B", "A", 0, 0)
    res = _run(df, algo, knowledge=k)
    assert _ab(res) == ("", 0, 0)
    assert not res.to_binary(undirected="bidirected")[0, 1, 0]


@pytest.mark.parametrize("algo", RUNNERS)
def test_forbidding_one_lag0_direction_keeps_the_other(df, algo):
    """Forbidding B -> A removed the true A -> B adjacency."""
    k = AncestralKnowledge().add_forbidden_edge("B", "A", 0, 0)
    assert _ab(_run(df, algo, knowledge=k)) == ("-->", 1, 0)


def test_forbidden_lagged_edge_still_removes_the_adjacency():
    """A lagged forbid has no reverse direction: the adjacency still goes."""
    rng = np.random.default_rng(1)
    T = 600
    x = rng.standard_normal((T, 2))
    x[1:, 1] += 0.8 * x[:-1, 0]
    df = pd.DataFrame(x, columns=["A", "B"])
    ci = ParCorrGPU(np.zeros((2, 2)), device="cpu")
    assert run_cdnots(df, ci, num_lags=1, include_C=False).cg_tig[0, 1, 1]
    k = AncestralKnowledge().add_forbidden_edge("A", "B", 1, 0)
    res = run_cdnots(df, ci, num_lags=1, include_C=False, knowledge=k)
    assert not res.cg_tig[0, 1, 1]


def test_mci_forbid_prepass_keeps_a_required_shifted_copy(df):
    """A required copy protects its shift class from the both-ways forbid."""
    k = AncestralKnowledge().add_forbidden_edge("A", "B", 0, 0)
    k.add_forbidden_edge("B", "A", 0, 0).add_required_edge("A", "B", 1, 1)
    res = _run(df, "cdnots+", knowledge=k)
    assert res.to_binary(undirected="bidirected")[0, 1, 0]


def test_knowledge_orientation_refreshes_ancestry():
    """c_lag_cnst's knowledge orientation must leave a consistent dpath cache."""
    from causalts.cdnots.skeleton_discovery import c_lag_cnst, initialize_graph

    data = np.zeros((10, 6))  # 3 variables x (num_lags + 1) slices
    cg = initialize_graph(data, None)
    bk = make_background_knowledge(
        ["A", "B", "D"], 1, include_C=False, required=[("A", "B", 0, 0)]
    )
    out = c_lag_cnst(cg, num_lags=1, include_C=False, background_knowledge=bk)
    a, b = out.G.nodes[0], out.G.nodes[1]
    assert out.G.graph[0, 1] == -1 and out.G.graph[1, 0] == 1
    assert out.G.is_ancestor_of(a, b)


def test_forbid_written_backwards_in_time_still_removes_the_adjacency():
    """("B", "A", 0, 1) names the impossible B(t) -> A(t-1): the pair goes."""
    from causalts.cdnots.skeleton_discovery import initialize_graph

    bk = make_background_knowledge(
        ["A", "B"], 1, include_C=False, forbidden=[("B", "A", 0, 1)]
    )

    cg = initialize_graph(np.zeros((10, 4)), None)
    a1, b0 = cg.G.nodes[2], cg.G.nodes[1]  # A(t-1), B(t)
    assert bk.is_forbidden(a1, b0) and bk.is_forbidden(b0, a1)


def test_knowledge_orientation_respects_forbidden_ancestors():
    """A lag copy that would complete a forbidden-ancestor path stays undirected."""
    from causalts.cdnots.skeleton_discovery import c_lag_cnst, initialize_graph

    # A, B, D x 2 slices: A0=0 B0=1 D0=2 A1=3 B1=4 D1=5.
    cg = initialize_graph(np.zeros((10, 6)), None)
    g = np.zeros((6, 6), dtype=int)
    for x, y in [(0, 1), (3, 4), (4, 2)]:
        g[x, y] = g[y, x] = -1
    cg.G.graph = g
    bk = make_background_knowledge(
        ["A", "B", "D"], 1, include_C=False, forbidden=[("B", "A", 0, 0)]
    )
    out = c_lag_cnst(
        cg,
        num_lags=1,
        include_C=False,
        background_knowledge=bk,
        forbidden_ancestor_pairs=[(3, 2)],  # A(t-1) must not be an ancestor of D(t)
    )
    G = out.G.graph
    assert (G[4, 2], G[2, 4]) == (-1, 1)  # B(t-1) -> D(t), from time order
    assert (G[0, 1], G[1, 0]) == (-1, 1)  # A(t) -> B(t), from the knowledge
    assert G[3, 4] == G[4, 3] == -1  # A(t-1) -> B(t-1) would reach D(t)


def test_forbid_with_both_lags_nonzero_is_anchored_at_lag0():
    """("B", "A", 1, 2) must act on the class the skeleton checks (lag 0)."""
    from causalts.cdnots.skeleton_discovery import initialize_graph

    bk = make_background_knowledge(
        ["A", "B"], 2, include_C=False, forbidden=[("B", "A", 1, 2)]
    )
    cg = initialize_graph(np.zeros((10, 6)), None)
    n = cg.G.nodes  # A0=0 B0=1 A1=2 B1=3 A2=4 B2=5
    assert bk.is_forbidden(n[2], n[1]) and bk.is_forbidden(n[1], n[2])


def test_required_edge_bypasses_the_ancestor_veto():
    """A required orientation is kept even if it completes a forbidden path."""
    from causalts.cdnots.skeleton_discovery import c_lag_cnst, initialize_graph

    cg = initialize_graph(np.zeros((10, 6)), None)
    g = np.zeros((6, 6), dtype=int)
    for x, y in [(0, 1), (0, 5)]:  # A0 - B0, A0 - D1 (oriented D1 -> A0 by time)
        g[x, y] = g[y, x] = -1
    cg.G.graph = g
    bk = make_background_knowledge(
        ["A", "B", "D"], 1, include_C=False, required=[("A", "B", 0, 0)]
    )
    out = c_lag_cnst(
        cg,
        num_lags=1,
        include_C=False,
        background_knowledge=bk,
        forbidden_ancestor_pairs=[(5, 1)],  # D(t-1) must not reach B(t)
    )
    assert (out.G.graph[0, 1], out.G.graph[1, 0]) == (-1, 1)  # still A -> B


def test_required_edge_stated_at_a_lag_orients_the_lag0_pair(df):
    """required ("A", "B", 1, 1) is the lag-0 relation one slice back."""
    k = AncestralKnowledge().add_required_edge("A", "B", 1, 1)
    for algo in RUNNERS:
        assert _ab(_run(df, algo, knowledge=k)) == ("-->", 1, 0)
    # The named copy stays required as well as its lag-0 anchor.
    from causalts.cdnots.skeleton_discovery import initialize_graph

    bk = make_background_knowledge(
        ["A", "B", "D"], 1, include_C=False, required=[("A", "B", 1, 1)]
    )
    n = initialize_graph(np.zeros((10, 6)), None).G.nodes
    assert bk.is_required(n[3], n[4]) and bk.is_required(n[0], n[1])


def test_refused_anchor_orientation_is_not_copied():
    """If the lag-0 orientation would complete a forbidden path, its copies stay
    undirected too, so the window stays time-invariant."""
    from causalts.cdnots.skeleton_discovery import c_lag_cnst, initialize_graph

    cg = initialize_graph(np.zeros((10, 6)), None)
    g = np.zeros((6, 6), dtype=int)
    for x, y in [(0, 1), (3, 4), (0, 5)]:  # A0-B0, A1-B1, A0-D1 (D1 -> A0)
        g[x, y] = g[y, x] = -1
    cg.G.graph = g
    bk = make_background_knowledge(
        ["A", "B", "D"], 1, include_C=False, forbidden=[("B", "A", 0, 0)]
    )
    out = c_lag_cnst(
        cg,
        num_lags=1,
        include_C=False,
        background_knowledge=bk,
        forbidden_ancestor_pairs=[(5, 1)],  # D(t-1) must not reach B(t)
    )
    G = out.G.graph
    assert G[0, 1] == G[1, 0] == -1  # refused at lag 0
    assert G[3, 4] == G[4, 3] == -1  # and not copied to lag 1


def test_refused_anchor_is_not_re_decided_at_a_deeper_slice():
    """Raw knowledge forbidding B -> A in every slice: the refused lag-0 anchor
    decides the whole shift class, so lag 1 is not oriented on its own."""
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

    from causalts.cdnots.skeleton_discovery import c_lag_cnst, initialize_graph

    cg = initialize_graph(np.zeros((10, 6)), None)
    g = np.zeros((6, 6), dtype=int)
    for x, y in [(0, 1), (3, 4), (0, 5)]:  # A0-B0, A1-B1, A0-D1 (D1 -> A0)
        g[x, y] = g[y, x] = -1
    cg.G.graph = g
    bk = BackgroundKnowledge()
    bk.add_forbidden_by_pattern("^X1$", "^X0$")  # B(t) -> A(t)
    bk.add_forbidden_by_pattern("^X4$", "^X3$")  # B(t-1) -> A(t-1)
    out = c_lag_cnst(
        cg,
        num_lags=1,
        include_C=False,
        background_knowledge=bk,
        forbidden_ancestor_pairs=[(5, 1)],  # D(t-1) must not reach B(t)
    )
    G = out.G.graph
    assert G[0, 1] == G[1, 0] == -1  # refused at lag 0
    assert G[3, 4] == G[4, 3] == -1  # and lag 1 follows its anchor


def test_raw_knowledge_on_a_deeper_slice_orients_the_lag0_copy():
    """Raw knowledge naming only A(t-1) -> B(t-1) orients the whole shift class,
    including the lag-0 pair cg_tig reports."""
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

    from causalts.cdnots.skeleton_discovery import c_lag_cnst, initialize_graph

    cg = initialize_graph(np.zeros((10, 6)), None)
    g = np.zeros((6, 6), dtype=int)
    for x, y in [(0, 1), (3, 4)]:  # A0 - B0, A1 - B1
        g[x, y] = g[y, x] = -1
    cg.G.graph = g
    bk = BackgroundKnowledge()
    bk.add_required_by_pattern("^X3$", "^X4$")  # A(t-1) -> B(t-1)
    G = c_lag_cnst(cg, num_lags=1, include_C=False, background_knowledge=bk).G.graph
    assert (G[0, 1], G[1, 0]) == (-1, 1)  # A -> B at lag 0
    assert (G[3, 4], G[4, 3]) == (-1, 1)  # and at lag 1


def test_forbid_implied_orientation_does_not_close_a_cycle():
    """A kept A - D adjacency with A -> B -> D required and A -> D forbidden:
    orienting D -> A would close a directed cycle, so it stays undirected."""
    from causalts.cdnots.skeleton_discovery import c_lag_cnst, initialize_graph

    cg = initialize_graph(np.zeros((10, 6)), None)
    g = np.zeros((6, 6), dtype=int)
    for x, y in [(0, 1), (1, 2), (0, 2)]:  # A0 - B0, B0 - D0, A0 - D0
        g[x, y] = g[y, x] = -1
    cg.G.graph = g
    bk = make_background_knowledge(
        ["A", "B", "D"],
        1,
        include_C=False,
        forbidden=[("A", "D", 0, 0)],
        required=[("A", "B", 0, 0), ("B", "D", 0, 0)],
    )
    G = c_lag_cnst(cg, num_lags=1, include_C=False, background_knowledge=bk).G.graph
    assert (G[0, 1], G[1, 0], G[1, 2], G[2, 1]) == (-1, 1, -1, 1)  # A -> B -> D
    assert G[0, 2] == G[2, 0] == -1  # A - D left for the data


def test_required_ancestor_path_avoids_a_one_sided_forbidden_direction():
    """A -> C -> B with a kept A - B (forbidden A -> B), B - D undirected, and
    required ancestor A => D: the path must go A -> C -> B -> D, never A -> B."""
    from causalts.cdnots.ancestral import apply_ancestral_constraints
    from causalts.cdnots.skeleton_discovery import initialize_graph

    names = ["A", "B", "C", "D"]
    cg = initialize_graph(np.zeros((10, 4)), None)
    g = np.zeros((4, 4), dtype=int)
    for x, y in [(0, 2), (2, 1)]:  # A -> C -> B
        g[x, y], g[y, x] = -1, 1
    for x, y in [(0, 1), (1, 3)]:  # A - B, B - D
        g[x, y] = g[y, x] = -1
    cg.G.graph = g
    ak = (
        AncestralKnowledge()
        .add_forbidden_edge("A", "B", 0, 0)
        .add_required_ancestor("A", "D")
    )
    bk = ak.to_background_knowledge(names, 0, False)
    out, viol = apply_ancestral_constraints(
        cg, ak, names, 0, include_C=False, background_knowledge=bk
    )
    G = out.G.graph
    assert not (G[0, 1] == -1 and G[1, 0] == 1)  # no forbidden A -> B
    assert (G[1, 3], G[3, 1]) == (-1, 1)  # B -> D completes A -> C -> B -> D
    assert viol["required"] == []


def test_required_ancestor_path_honours_the_lag0_anchor_on_a_deeper_slice():
    """Forbidding A -> B is stored at its lag-0 anchor only. A path from A(t-1)
    must still not use A(t-1) -> B(t-1); it goes through C(t-1) instead."""
    from causalts.cdnots.ancestral import apply_ancestral_constraints
    from causalts.cdnots.skeleton_discovery import initialize_graph

    names = ["A", "B", "C", "D"]  # nodes 0-3 at lag 0, 4-7 at lag 1
    cg = initialize_graph(np.zeros((10, 8)), None)
    g = np.zeros((8, 8), dtype=int)
    for x, y in [(4, 5), (4, 6)]:  # A1 - B1, A1 - C1
        g[x, y] = g[y, x] = -1
    for x, y in [(5, 3), (6, 3)]:  # B1 -> D0, C1 -> D0
        g[x, y], g[y, x] = -1, 1
    cg.G.graph = g
    ak = (
        AncestralKnowledge()
        .add_forbidden_edge("A", "B", 0, 0)
        .add_required_ancestor("A", "D")
    )
    bk = ak.to_background_knowledge(names, 1, False)
    out, viol = apply_ancestral_constraints(
        cg, ak, names, 1, include_C=False, background_knowledge=bk
    )
    G = out.G.graph
    assert not (G[4, 5] == -1 and G[5, 4] == 1)  # no A(t-1) -> B(t-1)
    assert (G[4, 6], G[6, 4]) == (-1, 1)  # A(t-1) -> C(t-1) -> D(t)
    assert viol["required"] == []
