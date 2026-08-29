# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Correctness tests for the PDAG/DAG/CPDAG machinery in causalts.lges.

Three bugs lived here and each one silently under- or mis-oriented CPDAGs, which
LGES then emitted as edges in both directions:

* ``order_edges`` sorted the tail ascending, reversing Chickering's inner ordering
* ``label_edges`` omitted Chickering's phase one (propagation through ``x``), so
  edges that Meek's rules force came back reversible
* ``pdag_to_dag`` accepted non-sinks and derived directions from the removal
  order backwards, which could return a cyclic "DAG"

The exhaustive tests below pin the whole pipeline against the definition of
Markov equivalence (same skeleton + same v-structures) rather than against a
reference implementation.
"""

import itertools

import networkx as nx
import numpy as np
import pytest

from causalts.lges import dag_to_cpdag, pdag_to_cpdag, pdag_to_dag


def _all_dags(d):
    pairs = list(itertools.permutations(range(d), 2))
    for mask in range(1 << len(pairs)):
        M = np.zeros((d, d))
        for b, (i, j) in enumerate(pairs):
            if mask >> b & 1:
                M[i, j] = 1
        if any(M[i, j] and M[j, i] for i, j in pairs):
            continue
        g = nx.DiGraph()
        g.add_nodes_from(range(d))
        g.add_edges_from([(i, j) for i in range(d) for j in range(d) if M[i, j]])
        if nx.is_directed_acyclic_graph(g):
            yield M


def _skeleton(M):
    d = len(M)
    return frozenset(frozenset((i, j)) for i in range(d) for j in range(d) if M[i, j])


def _vstructures(M):
    """Unshielded colliders i -> c <- j."""
    d = len(M)
    return frozenset(
        (min(a, b), c, max(a, b))
        for c in range(d)
        for a, b in itertools.combinations([p for p in range(d) if M[p, c]], 2)
        if not M[a, b] and not M[b, a]
    )


def _classes(d):
    """Group every DAG on d nodes into true Markov equivalence classes."""
    out = {}
    for M in _all_dags(d):
        out.setdefault((_skeleton(M), _vstructures(M)), []).append(M)
    return out


def _is_acyclic(M):
    d = len(M)
    g = nx.DiGraph()
    g.add_nodes_from(range(d))
    g.add_edges_from([(i, j) for i in range(d) for j in range(d) if M[i, j]])
    return nx.is_directed_acyclic_graph(g)


def _check_all(d):
    for key, members in _classes(d).items():
        cpdags = [pdag_to_cpdag(M) for M in members]

        # every member of a class must map to the SAME cpdag
        assert len({c.tobytes() for c in cpdags}) == 1, f"class {key} not collapsed"

        # compelled edges are exactly the edges shared by every member
        C = cpdags[0]
        compelled = {
            (i, j) for i in range(d) for j in range(d) if C[i, j] and not C[j, i]
        }
        shared = {
            (i, j) for i in range(d) for j in range(d) if all(M[i, j] for M in members)
        }
        assert compelled == shared, f"class {key}: {compelled} != {shared}"

        # a consistent extension must exist, be acyclic, and be in the same class
        G = pdag_to_dag(C)
        assert _is_acyclic(G), f"class {key}: pdag_to_dag returned a cyclic graph"
        assert (_skeleton(G), _vstructures(G)) == key, f"class {key}: wrong extension"
        assert np.array_equal(dag_to_cpdag(G), C), f"class {key}: round trip failed"


def test_meek_forced_edge_is_oriented():
    """A -> C <- B with C - D: Meek R1 forces C -> D (this used to stay undirected)."""
    M = np.zeros((4, 4))
    M[0, 2] = M[1, 2] = M[2, 3] = 1  # A->C, B->C, C->D
    C = pdag_to_cpdag(M)
    directed = {(i, j) for i in range(4) for j in range(4) if C[i, j] and not C[j, i]}
    assert directed == {(0, 2), (1, 2), (2, 3)}


def test_chain_comes_back_undirected():
    """A -> B -> C is Markov equivalent to two other DAGs, so nothing is compelled."""
    M = np.zeros((3, 3))
    M[0, 1] = M[1, 2] = 1
    C = pdag_to_cpdag(M)
    assert not [(i, j) for i in range(3) for j in range(3) if C[i, j] and not C[j, i]]


def test_triangle_comes_back_undirected():
    """A fully connected triple has no v-structure, so all 6 orientations tie."""
    M = np.zeros((3, 3))
    M[0, 1] = M[0, 2] = M[1, 2] = 1
    C = pdag_to_cpdag(M)
    assert not [(i, j) for i in range(3) for j in range(3) if C[i, j] and not C[j, i]]


def test_exhaustive_4_nodes():
    """All 543 DAGs / 185 equivalence classes on 4 nodes."""
    _check_all(4)


@pytest.mark.slow
def test_exhaustive_5_nodes():
    """All 29281 DAGs / 8782 equivalence classes on 5 nodes."""
    _check_all(5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
