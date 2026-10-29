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
import pytest
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

import causalts.cdnots.ancestral as ancestral
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


def _require(*pairs):
    bk = BackgroundKnowledge()
    for a, b in pairs:
        bk.add_required_by_node(GraphNode(f"X{a}"), GraphNode(f"X{b}"))
    return bk


def test_forbidden_ancestor_keeps_a_required_last_hop():
    # A -> Y -> D with Y -> D required (via the knowledge object): cut A -> Y.
    cg = _graph(3, directed=[(0, 1), (1, 2)])
    knowledge = AncestralKnowledge().add_forbidden_ancestor("A", "D")
    knowledge.add_required_edge("Y", "D")
    names = ["A", "Y", "D"]
    bk = knowledge.to_background_knowledge(names, 0, False)
    out, violations = apply_ancestral_constraints(
        cg, knowledge, names, 0, background_knowledge=bk
    )
    assert _directed(out, 1, 2)
    assert out.G.get_edge(out.G.nodes[0], out.G.nodes[1]) is None
    assert not violations["forbidden"]


def test_forbidden_ancestor_cut_respects_required_ancestors():
    # A -> B -> C -> D, C -> D required, B must stay an ancestor of C: the only
    # admissible cut is A -> B.
    cg = _graph(4, directed=[(0, 1), (1, 2), (2, 3)])
    knowledge = AncestralKnowledge().add_forbidden_ancestor("A", "D")
    knowledge.add_required_ancestor("B", "C")
    out, violations = apply_ancestral_constraints(
        cg, knowledge, ["A", "B", "C", "D"], 0, background_knowledge=_require((2, 3))
    )
    assert out.G.get_edge(out.G.nodes[0], out.G.nodes[1]) is None
    assert _directed(out, 1, 2) and _directed(out, 2, 3)
    assert not violations["forbidden"] and not violations["required"]


def test_forbidden_ancestor_cut_removes_the_lag_copies():
    # One lag: A(t) -> Y(t) -> D(t) with Y -> D required, plus the copy
    # A(t-1) -> Y(t-1). Cutting A -> Y also cuts its copy.
    cg = _graph(6, directed=[(0, 1), (1, 2), (3, 4)])
    knowledge = AncestralKnowledge().add_forbidden_ancestor("A", "D")
    out, _ = apply_ancestral_constraints(
        cg, knowledge, ["A", "Y", "D"], 1, background_knowledge=_require((1, 2))
    )
    assert out.G.get_edge(out.G.nodes[0], out.G.nodes[1]) is None
    assert out.G.get_edge(out.G.nodes[3], out.G.nodes[4]) is None


def test_last_hop_cut_respects_required_ancestors():
    # A -> Y -> D and B -> Y -> D, B must stay an ancestor of D: cutting the
    # last hop Y -> D would break it, so A -> Y goes instead.
    cg = _graph(4, directed=[(0, 2), (1, 2), (2, 3)])
    knowledge = AncestralKnowledge().add_forbidden_ancestor("A", "D")
    knowledge.add_required_ancestor("B", "D")
    out, violations = apply_ancestral_constraints(
        cg, knowledge, ["A", "B", "Y", "D"], 0
    )
    assert out.G.get_edge(out.G.nodes[0], out.G.nodes[2]) is None
    assert _directed(out, 1, 2) and _directed(out, 2, 3)
    assert not violations["forbidden"] and not violations["required"]


def test_direct_cut_with_copies_respects_required_ancestors():
    # One lag; A, B, C, D = 0-3 (t), 4-7 (t-1). A(t) -> D(t) and its copy
    # A(t-1) -> D(t-1) on the required path B(t-1) -> A(t-1) -> D(t-1) -> C(t).
    # Cutting A -> D with its copies would break B => C, so pass 1 undoes that
    # cut and retries the copies one by one: only the lag-0 edge goes.
    cg = _graph(8, directed=[(0, 3), (4, 7), (5, 4), (7, 2)])
    knowledge = AncestralKnowledge().add_forbidden_ancestor("A", "D")
    knowledge.add_required_ancestor("B", "C")
    out, violations = apply_ancestral_constraints(
        cg, knowledge, ["A", "B", "C", "D"], 1
    )
    assert out.G.get_edge(out.G.nodes[0], out.G.nodes[3]) is None
    assert _directed(out, 4, 7) and reaches(out.G.graph, 5, 2)
    assert not violations["forbidden"] and not violations["required"]


def test_last_hop_needed_by_required_ancestor_goes_to_fallback():
    # A -> Y -> D, forbid A => D, require Y => D: pass 2 undoes the cut of the
    # last hop Y -> D, so the fallback cuts A -> Y instead.
    cg = _graph(3, directed=[(0, 1), (1, 2)])
    knowledge = AncestralKnowledge().add_forbidden_ancestor("A", "D")
    knowledge.add_required_ancestor("Y", "D")
    out, violations = apply_ancestral_constraints(cg, knowledge, ["A", "Y", "D"], 0)
    assert out.G.get_edge(out.G.nodes[0], out.G.nodes[1]) is None
    assert _directed(out, 1, 2)
    assert not violations["forbidden"] and not violations["required"]


def test_direct_cut_retries_unneeded_copies_one_by_one():
    # Two lags; A, B, C, D = 0-3 (t), 4-7 (t-1), 8-11 (t-2). Only the copy
    # A(t-1) -> D(t-1) is on the required path B(t-1) -> A(t-1) -> D(t-1) -> C(t);
    # the other copies of A -> D are still cut.
    cg = _graph(12, directed=[(0, 3), (4, 7), (8, 11), (5, 4), (7, 2)])
    knowledge = AncestralKnowledge().add_forbidden_ancestor("A", "D")
    knowledge.add_required_ancestor("B", "C")
    out, violations = apply_ancestral_constraints(
        cg, knowledge, ["A", "B", "C", "D"], 2
    )
    assert out.G.get_edge(out.G.nodes[0], out.G.nodes[3]) is None
    assert out.G.get_edge(out.G.nodes[8], out.G.nodes[11]) is None
    assert _directed(out, 4, 7) and reaches(out.G.graph, 5, 2)
    assert not violations["forbidden"] and not violations["required"]


def test_fallback_cuts_the_edge_alone_when_its_copy_is_needed():
    # One lag; A, Y, D, B, C = 0-4 (t), 5-9 (t-1). A -> Y -> D with Y -> D
    # required; the copy A(t-1) -> Y(t-1) is on the only path
    # B(t-1) -> A(t-1) -> Y(t-1) -> C(t) that keeps B => C. Cut A(t) -> Y(t) only.
    cg = _graph(10, directed=[(0, 1), (1, 2), (5, 6), (8, 5), (6, 4)])
    knowledge = AncestralKnowledge().add_forbidden_ancestor("A", "D")
    knowledge.add_required_ancestor("B", "C")
    out, violations = apply_ancestral_constraints(
        cg,
        knowledge,
        ["A", "Y", "D", "B", "C"],
        1,
        background_knowledge=_require((1, 2)),
    )
    assert out.G.get_edge(out.G.nodes[0], out.G.nodes[1]) is None
    assert _directed(out, 5, 6) and _directed(out, 1, 2)
    assert not violations["forbidden"] and not violations["required"]


def test_verbose_logs_only_kept_removals(capsys):
    # Same graph as above: the cut with copies is undone, the edge alone stays
    # cut, and only that removal is logged, under the forbidden pair.
    cg = _graph(10, directed=[(0, 1), (1, 2), (5, 6), (8, 5), (6, 4)])
    knowledge = AncestralKnowledge().add_forbidden_ancestor("A", "D")
    knowledge.add_required_ancestor("B", "C")
    apply_ancestral_constraints(
        cg,
        knowledge,
        ["A", "Y", "D", "B", "C"],
        1,
        background_knowledge=_require((1, 2)),
        verbose=True,
    )
    removed = [ln for ln in capsys.readouterr().out.splitlines() if "Removed" in ln]
    assert len(removed) == 1
    assert "A → Y (blocked last hop, forbidden_ancestor A→D)" in removed[0]


def test_forbidden_ancestor_keeps_a_required_direct_edge():
    # A -> D itself required: kept, and reported as a violation.
    cg = _graph(2, directed=[(0, 1)])
    knowledge = AncestralKnowledge().add_forbidden_ancestor("A", "D")
    out, violations = apply_ancestral_constraints(
        cg, knowledge, ["A", "D"], 0, background_knowledge=_require((0, 1))
    )
    assert _directed(out, 0, 1)
    assert violations["forbidden"]


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


def test_forbidden_ancestor_cut_set_search_resolves_competing_constraints():
    # A -> B, A -> C, B -> C, B -> D, C -> E, D -> E; require A => E,
    # forbid B => E. Greedy cuts C -> E first and then cannot cut B -> D -> E
    # without breaking A => E; cutting B -> C and B -> D satisfies both.
    cg = _graph(5, directed=[(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)])
    knowledge = AncestralKnowledge().add_forbidden_ancestor("B", "E")
    knowledge.add_required_ancestor("A", "E")
    out, violations = apply_ancestral_constraints(
        cg, knowledge, ["A", "B", "C", "D", "E"], 0
    )
    assert not reaches(out.G.graph, 1, 4) and reaches(out.G.graph, 0, 4)
    assert not violations["forbidden"] and not violations["required"]


def test_cut_set_search_logs_only_its_own_cuts(capsys):
    cg = _graph(5, directed=[(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)])
    knowledge = AncestralKnowledge().add_forbidden_ancestor("B", "E")
    knowledge.add_required_ancestor("A", "E")
    apply_ancestral_constraints(
        cg, knowledge, ["A", "B", "C", "D", "E"], 0, verbose=True
    )
    removed = [ln for ln in capsys.readouterr().out.splitlines() if "Removed" in ln]
    assert len(removed) == 2 and all("cut-set search" in ln for ln in removed)


def test_cut_set_search_result_undone_by_meek_keeps_greedy(monkeypatch, capsys):
    # Meek (patched) re-adds B -> C after the search cuts B -> C and B -> D, so
    # the search result is dropped: the greedy graph and its log are returned.
    import causalts.cdnots.meek as M

    real_meek = M.meek

    def meek_readding(cg, **kw):
        out = real_meek(cg, **kw)
        nodes = out.G.nodes
        if out.G.get_edge(nodes[1], nodes[2]) is None and (
            out.G.get_edge(nodes[1], nodes[3]) is None
        ):
            out.G.add_directed_edge(nodes[1], nodes[2])
        return out

    monkeypatch.setattr(M, "meek", meek_readding)
    cg = _graph(5, directed=[(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)])
    knowledge = AncestralKnowledge().add_forbidden_ancestor("B", "E")
    knowledge.add_required_ancestor("A", "E")
    out, violations = apply_ancestral_constraints(
        cg, knowledge, ["A", "B", "C", "D", "E"], 0, verbose=True
    )
    assert violations["forbidden"]
    assert out.G.get_edge(out.G.nodes[2], out.G.nodes[4]) is None
    assert _directed(out, 1, 2) and _directed(out, 1, 3)
    removed = [ln for ln in capsys.readouterr().out.splitlines() if "Removed" in ln]
    assert removed and not any("cut-set search" in ln for ln in removed)
    assert any("C → E" in ln for ln in removed)


@pytest.mark.parametrize("budget", [None, 5])
def test_cut_set_search_falls_back_to_single_edges(monkeypatch, budget):
    # One lag; A..G = 0-6 (t), 7-13 (t-1). The graph above in both slices, plus
    # F(t-1) -> B(t-1) and C(t-1) -> G(t) with F => G required: every valid cut
    # needs B -> C, but its copy B(t-1) -> C(t-1) carries F => G, so B -> C goes
    # alone (single-edge round). With a budget of 5 the grouped round runs out
    # first; the single-edge round still gets its own tries.
    if budget is not None:
        monkeypatch.setattr(ancestral, "_CUT_SEARCH_BUDGET", budget)
    lag0 = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)]
    directed = lag0 + [(u + 7, v + 7) for u, v in lag0] + [(12, 8), (9, 6)]
    cg = _graph(14, directed=directed)
    knowledge = AncestralKnowledge().add_forbidden_ancestor("B", "E")
    knowledge.add_required_ancestor("A", "E")
    knowledge.add_required_ancestor("F", "G")
    out, violations = apply_ancestral_constraints(
        cg, knowledge, ["A", "B", "C", "D", "E", "F", "G"], 1
    )
    assert out.G.get_edge(out.G.nodes[1], out.G.nodes[2]) is None
    assert _directed(out, 8, 9)
    assert not reaches(out.G.graph, 1, 4) and reaches(out.G.graph, 0, 4)
    assert reaches(out.G.graph, 12, 6)
    assert not violations["forbidden"] and not violations["required"]


def test_cut_set_search_retries_the_cut_that_hit_the_budget(monkeypatch):
    # The graph above with num_lags=1 and one lag copy, D(t-1) -> E(t-1), so the
    # rounds differ. The grouped round stops at its fifth cut, {B -> C, B -> D};
    # the single-edge round must still try it (the first minimal set) rather
    # than treat it as tried and settle for {B -> C, D -> E}.
    monkeypatch.setattr(ancestral, "_CUT_SEARCH_BUDGET", 4)
    directed = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4), (8, 9)]
    knowledge = AncestralKnowledge().add_forbidden_ancestor("B", "E")
    knowledge.add_required_ancestor("A", "E")
    out, violations = apply_ancestral_constraints(
        _graph(10, directed=directed), knowledge, ["A", "B", "C", "D", "E"], 1
    )
    nodes = out.G.nodes
    assert out.G.get_edge(nodes[1], nodes[2]) is None
    assert out.G.get_edge(nodes[1], nodes[3]) is None
    assert _directed(out, 3, 4)
    assert not violations["forbidden"] and not violations["required"]


def test_cut_set_search_cuts_lag_copies():
    # The graph above in both slices of a one-lag window (t: 0-4, t-1: 5-9): the
    # search cuts B -> C and B -> D with their copies, keeping the slices alike.
    lag0 = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)]
    cg = _graph(10, directed=lag0 + [(u + 5, v + 5) for u, v in lag0])
    knowledge = AncestralKnowledge().add_forbidden_ancestor("B", "E")
    knowledge.add_required_ancestor("A", "E")
    out, violations = apply_ancestral_constraints(
        cg, knowledge, ["A", "B", "C", "D", "E"], 1
    )
    for u, v in [(1, 2), (1, 3), (6, 7), (6, 8)]:
        assert out.G.get_edge(out.G.nodes[u], out.G.nodes[v]) is None
    assert _directed(out, 7, 9) and _directed(out, 8, 9)
    assert not violations["forbidden"] and not violations["required"]
