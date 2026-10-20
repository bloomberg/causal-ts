# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Orientation steps must respect the skeleton, background knowledge and acyclicity.

- Lag-copy propagation (Meek, collider step) may only orient a shifted copy the
  skeleton has, in a direction background knowledge allows.
- Removing an edge for a forbidden ancestor unshields triples the collider step
  never classified; Meek must not orient through them.
- Phase three must not orient against background knowledge or close a cycle.
"""

import numpy as np
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

import causalts.cdnots.phase3_utils as P
from causalts.cdnots.ancestral import AncestralKnowledge, apply_ancestral_constraints
from causalts.cdnots.meek import meek, reaches, rebuild_dpath
from causalts.cdnots.skeleton_discovery import initialize_graph
from causalts.cdnots.uc_sepset import uc_sepset


def _graph(n, undirected=(), directed=()):
    cg = initialize_graph(np.zeros((20, n)), None)
    keep = {frozenset(e) for e in list(undirected) + list(directed)}
    for i in range(n):
        for j in range(i + 1, n):
            edge = cg.G.get_edge(cg.G.nodes[i], cg.G.nodes[j])
            if edge is not None and frozenset((i, j)) not in keep:
                cg.G.remove_edge(edge)
    for a, b in directed:
        cg.G.remove_edge(cg.G.get_edge(cg.G.nodes[a], cg.G.nodes[b]))
        cg.G.add_directed_edge(cg.G.nodes[a], cg.G.nodes[b])
    return cg


def _directed(cg, a, b):
    return cg.G.graph[a, b] == -1 and cg.G.graph[b, a] == 1


def _undirected(cg, a, b):
    return cg.G.graph[a, b] == -1 and cg.G.graph[b, a] == -1


# --- lag-copy propagation ---------------------------------------------------
# Two variables, one lag: x0, x1 at nodes 0, 1 and x0(t-1), x1(t-1) at 2, 3.
# R1 on x0(t-1) -> x0(t) - x1(t) orients x0 -> x1 and copies it to 2 -> 3.


def test_meek_copies_an_orientation_onto_the_shifted_edge():
    out = meek(_graph(4, undirected=[(0, 1), (2, 3)], directed=[(2, 0)]), num_lags=1)
    assert _directed(out, 0, 1) and _directed(out, 2, 3)


def test_meek_copy_respects_lag_specific_knowledge():
    bk = BackgroundKnowledge().add_forbidden_by_pattern("^X2$", "^X3$")
    cg = _graph(4, undirected=[(0, 1), (2, 3)], directed=[(2, 0)])
    out = meek(cg, background_knowledge=bk, num_lags=1)
    assert _directed(out, 0, 1)
    assert _undirected(out, 2, 3)


def test_meek_copy_does_not_add_an_adjacency():
    out = meek(_graph(4, undirected=[(0, 1)], directed=[(2, 0)]), num_lags=1)
    assert _directed(out, 0, 1)
    assert out.G.get_edge(out.G.nodes[2], out.G.nodes[3]) is None


def test_collider_copy_respects_knowledge_and_adjacency():
    # Three variables, one lag (lagged copies 3-5): collider x0 -> x1 <- x2.
    # The copy 3 -> 4 <- 5 is wanted, except 5 -> 4 is forbidden.
    bk = BackgroundKnowledge().add_forbidden_by_pattern("^X5$", "^X4$")
    cg = _graph(6, undirected=[(0, 1), (1, 2), (3, 4), (4, 5)])
    out = uc_sepset(cg, priority=1, background_knowledge=bk, num_lags=1)
    assert _directed(out, 0, 1) and _directed(out, 2, 1)
    assert _directed(out, 3, 4) and not _directed(out, 5, 4)
    # Without the lagged edges the copies must not be created.
    out = uc_sepset(_graph(6, undirected=[(0, 1), (1, 2)]), priority=1, num_lags=1)
    assert out.G.get_edge(out.G.nodes[3], out.G.nodes[4]) is None


def test_collider_copy_guard_at_the_default_priority():
    # priority=2 copies through add_edge, not orient_or_bidirect.
    bk = BackgroundKnowledge().add_forbidden_by_pattern("^X5$", "^X4$")
    cg = _graph(6, undirected=[(0, 1), (1, 2), (3, 4), (4, 5)])
    out = uc_sepset(cg, priority=2, background_knowledge=bk, num_lags=1)
    assert _directed(out, 3, 4) and not _directed(out, 5, 4)
    out = uc_sepset(_graph(6, undirected=[(0, 1), (1, 2)]), priority=2, num_lags=1)
    assert out.G.get_edge(out.G.nodes[3], out.G.nodes[4]) is None


# --- ancestral edge removal ---------------------------------------------------


def test_forbidden_ancestor_removal_marks_new_triples_ambiguous():
    # A -> D -> B with A - B. Forbidding A as an ancestor of D removes A -> D,
    # which unshields D -> B - A; no collider test classified it, so B - A
    # must stay undirected rather than be oriented B -> A by R1.
    cg = _graph(3, undirected=[(0, 2)], directed=[(0, 1), (1, 2)])
    knowledge = AncestralKnowledge().add_forbidden_ancestor("A", "D")
    out, _ = apply_ancestral_constraints(cg, knowledge, ["A", "D", "B"], 0)
    assert out.G.get_edge(out.G.nodes[0], out.G.nodes[1]) is None
    assert _undirected(out, 0, 2)
    assert (1, 2, 0) in out.ambiguous_triples


def test_last_hop_removal_marks_new_triples_ambiguous():
    # A -> Y -> D -> M with M - Y. Forbidding A as an ancestor of D removes the
    # last hop Y -> D (pass 2), which unshields D -> M - Y.
    cg = _graph(4, undirected=[(3, 1)], directed=[(0, 1), (1, 2), (2, 3)])
    knowledge = AncestralKnowledge().add_forbidden_ancestor("A", "D")
    out, _ = apply_ancestral_constraints(cg, knowledge, ["A", "Y", "D", "M"], 0)
    assert out.G.get_edge(out.G.nodes[1], out.G.nodes[2]) is None
    assert (2, 3, 1) in out.ambiguous_triples


# --- phase three -----------------------------------------------------------------
# x0..x3 plus C (node 4), no lags; C -> x0 and C -> x3, and x0 - x3 undirected.
# The sink score is patched so phase three always picks x0 as the sink, i.e.
# wants x3 -> x0.


def _phase_three(monkeypatch, cg, **kw):
    monkeypatch.setattr(
        P, "infer_nonsta_dir", lambda cau, eff, c, w: 0.0 if eff.name == 0 else 1.0
    )
    data = np.random.default_rng(0).standard_normal((50, 5))
    return P.phase_three(cg, data, num_lags=0, include_C=True, num_c_cols=1, **kw)


def test_phase_three_orients_into_the_chosen_sink(monkeypatch):
    out = _phase_three(
        monkeypatch, _graph(5, undirected=[(0, 3)], directed=[(4, 0), (4, 3)])
    )
    assert _directed(out, 3, 0)


def test_phase_three_does_not_close_a_directed_cycle(monkeypatch):
    cg = _graph(
        5, undirected=[(0, 3)], directed=[(4, 0), (4, 3), (0, 1), (1, 2), (2, 3)]
    )
    out = _phase_three(monkeypatch, cg)
    assert not _directed(out, 3, 0)


def test_phase_three_respects_background_knowledge(monkeypatch):
    bk = BackgroundKnowledge().add_forbidden_by_pattern("^X3$", "^X0$")
    cg = _graph(5, undirected=[(0, 3)], directed=[(4, 0), (4, 3)])
    out = _phase_three(monkeypatch, cg, background_knowledge=bk)
    assert not _directed(out, 3, 0)


def test_phase_three_respects_forbidden_ancestors(monkeypatch):
    cg = _graph(5, undirected=[(0, 3)], directed=[(4, 0), (4, 3)])
    out = _phase_three(monkeypatch, cg, forbidden_ancestor_pairs=[(3, 0)])
    assert not _directed(out, 3, 0)


def _phase_three_by_index(monkeypatch, cg, n_cols, num_lags=0):
    # Sink = the candidate with the lowest index.
    monkeypatch.setattr(P, "infer_nonsta_dir", lambda cau, eff, c, w: float(eff.name))
    data = np.random.default_rng(0).standard_normal((50, n_cols))
    return P.phase_three(cg, data, num_lags=num_lags, include_C=True, num_c_cols=1)


def test_phase_three_meek_sees_the_phase_three_orientations(monkeypatch):
    # x0..x3 + C (node 4). Phase three orients x1 -> x0 and must leave x1 - x3
    # (x1 -> x0 -> x3 is a path); its final Meek call must then see x1 -> x0
    # and orient x1 -> x3, not x3 -> x1, which would close a cycle.
    cg = _graph(
        5,
        undirected=[(0, 1), (1, 3)],
        directed=[(0, 3), (2, 3), (4, 0), (4, 1), (4, 2), (4, 3)],
    )
    out = _phase_three_by_index(monkeypatch, cg, 5)
    assert _directed(out, 1, 0)
    assert not _directed(out, 3, 1)


def test_phase_three_orients_copies_of_an_already_directed_edge(monkeypatch):
    # One lag: sink x0, cause x1 (already x1 -> x0), extra x2, C (node 3);
    # nodes 4, 5 are the lagged x0, x1. The undirected copy x1(t-1) - x0(t-1)
    # still gets the direction of its lag-0 edge.
    cg = _graph(
        8,
        undirected=[(0, 2), (1, 2), (4, 5)],
        directed=[(1, 0), (3, 0), (3, 1), (3, 2)],
    )
    out = _phase_three_by_index(monkeypatch, cg, 8, num_lags=1)
    assert _directed(out, 5, 4)


# --- ancestry cache ---------------------------------------------------------------


def _write_chain(n, chain, undirected=()):
    # Directed edges written straight into the endpoint matrix, as phase three
    # does, so causal-learn's ancestry cache is not updated.
    cg = _graph(n, undirected=list(undirected) + list(zip(chain, chain[1:])))
    for a, b in zip(chain, chain[1:]):
        cg.G.graph[a, b], cg.G.graph[b, a] = -1, 1
    return cg


def test_rebuild_dpath_is_the_full_transitive_closure():
    cg = _write_chain(4, [2, 0, 1, 3])
    rebuild_dpath(cg)
    nodes = cg.G.nodes
    assert cg.G.is_ancestor_of(nodes[2], nodes[3])
    assert not cg.G.is_ancestor_of(nodes[3], nodes[2])


def test_meek_forbidden_ancestor_check_reads_the_live_graph():
    # 2 -> 0 -> 1 -> 3 - 4 with a stale ancestry cache. R1 would orient
    # 3 -> 4, giving the forbidden path 2 -> ... -> 4.
    cg = _write_chain(5, [2, 0, 1, 3], undirected=[(3, 4)])
    out = meek(cg, forbidden_ancestor_pairs=[(2, 4)])
    assert not _directed(out, 3, 4)


def test_meek_cycle_guard_reads_the_live_graph():
    # 0 -> 1 - 2 -> 3 - 0 with a correct cache. R1 orients 1 -> 2; causal-learn's
    # incremental cache update then misses 0 ~> 3, so a cache-based guard lets
    # R1 on 2 -> 3 - 0 orient 3 -> 0 and close 0 -> 1 -> 2 -> 3 -> 0.
    cg = _graph(4, undirected=[(1, 2), (0, 3)], directed=[(0, 1), (2, 3)])
    out = meek(cg)
    assert _directed(out, 1, 2)
    assert not _directed(out, 3, 0)
    # Meek leaves a cache that matches the live graph.
    nodes = out.G.nodes
    assert all(
        out.G.is_ancestor_of(nodes[a], nodes[b]) == reaches(out.G.graph, a, b)
        for a in range(4)
        for b in range(4)
    )


def test_phase_three_meek_gets_the_knowledge():
    # include_C=False: phase three is only its Meek call, which must respect
    # background knowledge (R1 on 2 -> 0 - 1 would orient the forbidden 0 -> 1).
    bk = BackgroundKnowledge().add_forbidden_by_pattern("^X0$", "^X1$")
    cg = _graph(3, undirected=[(0, 1)], directed=[(2, 0)])
    out = P.phase_three(
        cg, np.zeros((20, 3)), num_lags=0, include_C=False, background_knowledge=bk
    )
    assert not _directed(out, 0, 1)


def test_phase_three_copies_into_the_last_candidate(monkeypatch):
    # One lag: x0, x1, x2, C at 0-3, lagged copies at 4-7. Sink x0 is chosen
    # first (x1 -> x0); the loop then stops with x1 left. Its existing
    # x2 -> x1 must still reach the undirected copy x2(t-1) - x1(t-1).
    cg = _graph(
        8,
        undirected=[(0, 1), (4, 5), (5, 6)],
        directed=[(3, 0), (3, 1), (2, 1), (7, 4), (7, 5)],
    )
    out = _phase_three_by_index(monkeypatch, cg, 8, num_lags=1)
    assert _directed(out, 1, 0) and _directed(out, 5, 4)
    assert _directed(out, 6, 5)
