# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""The collider step must not leave a directed lag-0 cycle.

On inconsistent CI results, colliders from different triples can each orient
one edge of a cycle. ``uc_sepset`` re-selects those colliders as whole
triples, keeping the heaviest subset that leaves no cycle.
"""

import networkx as nx
import numpy as np
import pytest
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint

from causalts.cdnots.meek import meek
from causalts.cdnots.skeleton_discovery import initialize_graph
from causalts.cdnots.uc_sepset import _select_colliders, uc_sepset


def _lag0_cyclic(cg_tig):
    a = cg_tig[:, :, 0] != 0
    a = a & ~a.T
    return bool(list(nx.simple_cycles(nx.DiGraph(a.astype(int)))))


# Triangle 0 - 1 - 2 - 0 with one pendant per corner (3 on 1, 4 on 2, 5 on 0).
# Colliders 0 -> 1 <- 3, 1 -> 2 <- 4 and 2 -> 0 <- 5 close 0 -> 1 -> 2 -> 0;
# the triples 2 - 1 - 3, 0 - 2 - 4 and 1 - 0 - 5 are non-colliders.
EDGES = [(0, 1), (1, 2), (2, 0), (3, 1), (4, 2), (5, 0)]
COLLIDER = {frozenset((0, 3)): 1, frozenset((1, 4)): 2, frozenset((2, 5)): 0}
NON_COLLIDER = {frozenset((2, 3)): 1, frozenset((0, 4)): 2, frozenset((1, 5)): 0}
WEAK = frozenset((2, 5))  # also separated by {0} alone: the weakest collider


def _triangle(perm):
    """The graph above with node i relabelled perm[i], and a CI test whose
    p-values make exactly the colliders above (also relabelled)."""
    n = len(perm)
    cg = initialize_graph(np.zeros((20, n)), None)
    keep = {frozenset((perm[a], perm[b])) for a, b in EDGES}
    for i in range(n):
        for j in range(i + 1, n):
            if frozenset((i, j)) not in keep:
                cg.G.remove_edge(cg.G.get_edge(cg.G.nodes[i], cg.G.nodes[j]))
    inv = {perm[i]: i for i in range(n)}

    def ci_test(x, z, Z):
        pair, cond = frozenset((inv[x], inv[z])), {inv[v] for v in Z}
        if pair in COLLIDER:
            y = COLLIDER[pair]
            separated = y not in cond or (pair == WEAK and cond == {y})
        else:
            separated = NON_COLLIDER[pair] in cond
        return (0.9 if separated else 0.01), None

    return cg, ci_test


def _run_triangle(perm, priority=1):
    cg, ci_test = _triangle(perm)
    return uc_sepset(
        cg,
        priority=priority,
        contemp_collider_rule="majority",
        ci_test=ci_test,
        alpha=0.05,
        majority_mode="pc",
    )


def _marks(out, perm):
    """Endpoint pattern of every original edge, in original labels."""
    g = out.G.graph
    return {
        (a, b): (int(g[perm[a], perm[b]]), int(g[perm[b], perm[a]])) for a, b in EDGES
    }


TAIL_ARROW, UNDIRECTED = (-1, 1), (-1, -1)


@pytest.mark.parametrize("priority", [1, 2])
def test_the_weakest_collider_is_dropped(priority):
    perm = list(range(6))
    out = _run_triangle(perm, priority)
    m = _marks(out, perm)
    # 0 -> 1 <- 3 and 1 -> 2 <- 4 stay; 2 -> 0 <- 5 is dropped whole.
    assert m[(0, 1)] == m[(3, 1)] == m[(1, 2)] == m[(4, 2)] == TAIL_ARROW
    assert m[(2, 0)] == m[(5, 0)] == UNDIRECTED
    assert {(2, 0, 5), (5, 0, 2)} <= out.ambiguous_triples


def test_the_selection_does_not_depend_on_node_labels():
    # Visiting order follows node indices; relabelling must not change which
    # collider is dropped.
    expected = _marks(_run_triangle(list(range(6))), list(range(6)))
    rng = np.random.default_rng(0)
    for _ in range(10):
        perm = [int(v) for v in rng.permutation(6)]
        assert _marks(_run_triangle(perm), perm) == expected


def test_equal_evidence_drops_every_collider_on_the_cycle():
    # Without a majority vote every collider weighs the same: any two of the
    # three are acyclic, so no collider is in every heaviest subset.
    cg, _ = _triangle(list(range(6)))
    # The non-colliders, by separating set.
    cg.sepset[2, 3] = cg.sepset[3, 2] = [(1,)]
    cg.sepset[0, 4] = cg.sepset[4, 0] = [(2,)]
    cg.sepset[1, 5] = cg.sepset[5, 1] = [(0,)]
    out = uc_sepset(cg, priority=1)
    g = out.G.graph
    for a, b in EDGES:
        assert (g[a, b], g[b, a]) == UNDIRECTED


def test_select_colliders_unit():
    w = {(0, 1, 3): (0.5, 0.0), (1, 2, 4): (0.5, 0.0), (2, 0, 5): (0.2, 0.0)}
    assert _select_colliders(list(w), w, set(), 6) == {(0, 1, 3), (1, 2, 4)}
    # Ties on the first component fall to the second (smaller p-value sum).
    w2 = {**w, (2, 0, 5): (0.5, -0.5), (1, 2, 4): (0.5, -0.1)}
    assert _select_colliders(list(w2), w2, set(), 6) == {(0, 1, 3), (1, 2, 4)}
    # An arrow in ``others`` against a collider's arm cancels it: no cycle.
    assert _select_colliders(list(w), w, {(0, 2)}, 6) == set(w)
    # Beyond the exact limit the search is greedy, heaviest first.
    assert _select_colliders(list(w), w, set(), 6, exact_limit=1) == {
        (0, 1, 3),
        (1, 2, 4),
    }


def test_a_cycle_oriented_before_the_collider_step_warns():
    cg = initialize_graph(np.zeros((20, 3)), None)
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        cg.G.remove_edge(cg.G.get_edge(cg.G.nodes[a], cg.G.nodes[b]))
        cg.G.add_directed_edge(cg.G.nodes[a], cg.G.nodes[b])
    with pytest.warns(UserWarning, match="directed cycle before the collider"):
        uc_sepset(cg, priority=1)


def test_a_cycle_only_among_lag_copies_is_broken():
    # One lag: nodes 0-5 at lag 0, 6-11 their copies. The triangle's lag-0 edge
    # 0 -> 2 is oriented beforehand, so the collider 2 -> 0 <- 5 meets it and
    # leaves 0 <-> 2: no lag-0 cycle. The copy 8 - 6 is free, so the copies
    # close 6 -> 7 -> 8 -> 6 unless the lagged slice is checked too.
    n = 6
    cg = initialize_graph(np.zeros((20, 2 * n)), None)
    keep = {frozenset((a + s * n, b + s * n)) for a, b in EDGES for s in (0, 1)}
    for i in range(2 * n):
        for j in range(i + 1, 2 * n):
            if frozenset((i, j)) not in keep:
                cg.G.remove_edge(cg.G.get_edge(cg.G.nodes[i], cg.G.nodes[j]))
    cg.G.remove_edge(cg.G.get_edge(cg.G.nodes[0], cg.G.nodes[2]))
    cg.G.add_directed_edge(cg.G.nodes[0], cg.G.nodes[2])
    cg.sepset[2, 3] = cg.sepset[3, 2] = [(1,)]
    cg.sepset[0, 4] = cg.sepset[4, 0] = [(2,)]
    cg.sepset[1, 5] = cg.sepset[5, 1] = [(0,)]
    out = uc_sepset(cg, priority=1, num_lags=1)
    g = out.G.graph
    directed = (g == -1) & (g.T == 1)
    assert not list(nx.simple_cycles(nx.DiGraph(directed.astype(int))))
    # The arrow oriented beforehand is kept.
    assert directed[0, 2] or (g[0, 2] == g[2, 0] == 1)


def _triangle_plus(n_extra, directed, ci_extra, undirected=()):
    """The triangle with nodes 6.. added: ``directed`` edges oriented
    beforehand, ``undirected`` ones left open, and ``ci_extra(pair, cond)``
    deciding the new pairs."""
    cg, ci_base = _triangle(list(range(6 + n_extra)))
    for a, b in directed:
        cg.G.add_directed_edge(cg.G.nodes[a], cg.G.nodes[b])
    for a, b in undirected:
        cg.G.add_edge(Edge(cg.G.nodes[a], cg.G.nodes[b], Endpoint.TAIL, Endpoint.TAIL))

    def ci_test(x, z, Z):
        pair = frozenset((x, z))
        if pair in COLLIDER or pair in NON_COLLIDER:
            return ci_base(x, z, Z)
        return (0.9 if ci_extra(pair, set(Z)) else 0.01), None

    out = uc_sepset(
        cg,
        priority=1,
        contemp_collider_rule="majority",
        ci_test=ci_test,
        alpha=0.05,
        majority_mode="pc",
    )
    return out


def test_a_cycle_oriented_beforehand_does_not_drop_every_collider():
    # A separate cycle 6 -> 7 -> 8 -> 6 oriented before the collider step must
    # not make every implicated collider go: only 2 -> 0 <- 5 should.
    with pytest.warns(UserWarning, match="directed cycle before the collider"):
        out = _triangle_plus(3, [(6, 7), (7, 8), (8, 6)], lambda pair, cond: False)
    m = _marks(out, list(range(9)))
    assert m[(0, 1)] == m[(3, 1)] == m[(1, 2)] == m[(4, 2)] == TAIL_ARROW
    assert m[(2, 0)] == m[(5, 0)] == UNDIRECTED


def test_colliders_reasserting_a_cycle_oriented_beforehand_are_ignored():
    # As above, plus pendants 9, 10, 11 on 7, 8, 6 whose colliders
    # 6 -> 7 <- 9, 7 -> 8 <- 10 and 8 -> 6 <- 11 re-assert every arrow of the
    # earlier cycle. They must not put it back into the selection.
    mid = {
        frozenset((6, 9)): 7,
        frozenset((7, 10)): 8,
        frozenset((8, 11)): 6,
    }
    non = {frozenset((8, 9)): 7, frozenset((6, 10)): 8, frozenset((7, 11)): 6}

    def ci_extra(pair, cond):
        if pair in mid:
            return mid[pair] not in cond
        return non[pair] in cond

    with pytest.warns(UserWarning, match="directed cycle before the collider"):
        out = _triangle_plus(
            6,
            [(6, 7), (7, 8), (8, 6)],
            ci_extra,
            undirected=[(9, 7), (10, 8), (11, 6)],
        )
    m = _marks(out, list(range(12)))
    assert m[(0, 1)] == m[(3, 1)] == m[(1, 2)] == m[(4, 2)] == TAIL_ARROW
    assert m[(2, 0)] == m[(5, 0)] == UNDIRECTED


def test_a_skipped_ambiguous_triple_stays_ambiguous():
    # 6 -> 0 is oriented beforehand. The triple 5 - 0 - 6 is visited after
    # 2 -> 0 <- 5 has oriented 5 -> 0, so the loop skips it; its vote is a tie
    # (ambiguous). Once 2 -> 0 <- 5 is dropped, 6 -> 0 - 5 must not let meek
    # orient 0 -> 5 as if 5 - 0 - 6 were a known non-collider.
    def ci_extra(pair, cond):
        if pair == frozenset((5, 6)):
            return True  # separated with and without 0: fraction 0.5
        return 0 in cond  # 1 - 0 - 6 and 2 - 0 - 6 are non-colliders

    out = _triangle_plus(1, [(6, 0)], ci_extra)
    assert (5, 0, 6) in out.ambiguous_triples
    g = meek(out).G.graph
    assert (g[0, 5], g[5, 0]) == UNDIRECTED


def test_a_skipped_triple_voted_a_collider_is_recorded_as_one():
    # As above, but 5 - 0 - 6 is a collider (5 and 6 separated only without
    # 0). It is classified only when the cycle is resolved, and later steps
    # that read collider_triples must see it.
    def ci_extra(pair, cond):
        if pair == frozenset((5, 6)):
            return 0 not in cond
        return 0 in cond

    out = _triangle_plus(1, [(6, 0)], ci_extra)
    assert (5, 0, 6) in out.collider_triples
    assert (6, 0, 5) in out.collider_triples


def test_a_triple_skipped_without_a_cycle_is_not_read_as_a_non_collider():
    # Star on 0: 1 - 0 - 2 and 1 - 0 - 3 are colliders, so both arms of
    # 2 - 0 - 3 point into 0 before it is visited and it is skipped. No cycle,
    # so it is never voted on: it must not read as a non-collider.
    cg = initialize_graph(np.zeros((20, 4)), None)
    for i, j in [(1, 2), (1, 3), (2, 3)]:
        cg.G.remove_edge(cg.G.get_edge(cg.G.nodes[i], cg.G.nodes[j]))

    def ci_test(x, z, Z):
        if frozenset((x, z)) == frozenset((2, 3)):
            return (0.9 if 0 in Z else 0.01), None
        return (0.9 if 0 not in Z else 0.01), None

    out = uc_sepset(
        cg,
        priority=1,
        contemp_collider_rule="majority",
        ci_test=ci_test,
        alpha=0.05,
        majority_mode="pc",
    )
    g = out.G.graph
    assert all((g[v, 0], g[0, v]) == (-1, 1) for v in (1, 2, 3))
    assert {(2, 0, 3), (3, 0, 2)} <= out.collider_triples


def test_colliders_against_a_cycle_oriented_beforehand_make_conflicts():
    # Pendants 9, 10, 11 on 6, 7, 8 whose colliders 9 -> 6 <- 7, 10 -> 7 <- 8
    # and 11 -> 8 <- 6 point against every arrow of the earlier cycle: each
    # edge becomes a conflict, so the earlier cycle is gone. Seen without the
    # earlier arrows, those reverse arrows would form a cycle of their own.
    mid = {frozenset((9, 7)): 6, frozenset((10, 8)): 7, frozenset((11, 6)): 8}
    non = {frozenset((8, 9)): 6, frozenset((6, 10)): 7, frozenset((7, 11)): 8}

    def ci_extra(pair, cond):
        if pair in mid:
            return mid[pair] not in cond
        return non[pair] in cond

    out = _triangle_plus(
        6,
        [(6, 7), (7, 8), (8, 6)],
        ci_extra,
        undirected=[(9, 6), (10, 7), (11, 8)],
    )
    m = _marks(out, list(range(12)))
    assert m[(0, 1)] == m[(3, 1)] == m[(1, 2)] == m[(4, 2)] == TAIL_ARROW
    assert m[(2, 0)] == m[(5, 0)] == UNDIRECTED


def test_a_collider_whose_re_vote_finds_no_separating_set_weighs_least():
    # A CI test whose p-values change between calls: once the weak pair's vote
    # is re-run from its first subset, nothing separates it any more. With no
    # support it must be the collider dropped, not the strongest.
    cg, base = _triangle(list(range(6)))
    first = []

    def ci_test(x, z, Z):
        if frozenset((x, z)) == WEAK:
            if not first:
                first.append(tuple(Z))
            elif tuple(Z) == first[0]:
                first.append("re-vote")
            if len(first) > 1:
                return 0.01, None
        return base(x, z, Z)

    out = uc_sepset(
        cg,
        priority=1,
        contemp_collider_rule="majority",
        ci_test=ci_test,
        alpha=0.05,
        majority_mode="pc",
    )
    assert len(first) > 1  # the re-vote happened
    m = _marks(out, list(range(6)))
    assert m[(0, 1)] == m[(3, 1)] == m[(1, 2)] == m[(4, 2)] == TAIL_ARROW
    assert m[(2, 0)] == m[(5, 0)] == UNDIRECTED
