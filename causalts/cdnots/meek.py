"""
This file has been modified by Mohammad Fesanghary on July 28 2025.
Original Package: causal-learn 1.9.0
Original License: MIT License
"""

# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

import numpy as np
from causallearn.graph.Edge import Edge  # type: ignore
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


def _build_adj_dict(cg):
    """Build adjacency dict from CausalGraph for fast neighbor lookup."""
    n = cg.G.num_vars
    G = cg.G.graph
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if G[i, j] != 0 or G[j, i] != 0:
                adj[i].add(j)
                adj[j].add(i)
    return adj


def _find_unshielded_triples_fast(cg, adj):
    """Find unshielded triples i-j-k where i and k are not adjacent. O(V * deg^2)."""
    G = cg.G.graph
    triples = []
    for j, neighbors_j in adj.items():
        nbrs = list(neighbors_j)
        for ii in range(len(nbrs)):
            i = nbrs[ii]
            for ki in range(len(nbrs)):
                k = nbrs[ki]
                if i != k and G[i, k] == 0 and G[k, i] == 0:
                    triples.append((i, j, k))
    return triples


def _find_triangles_fast(cg, adj):
    """Find triangles i-j-k where all three are mutually adjacent. O(V * deg^2)."""
    triples = []
    for j, neighbors_j in adj.items():
        nbrs = list(neighbors_j)
        for ii in range(len(nbrs)):
            i = nbrs[ii]
            adj_i = adj[i]
            for ki in range(len(nbrs)):
                k = nbrs[ki]
                if i != k and k in adj_i:
                    triples.append((i, j, k))
    return triples


def _find_kites_fast(cg, triangles):
    """Find kites from triangles. O(T * max_group) instead of O(T^2)."""
    G = cg.G.graph
    by_anchor_apex = defaultdict(list)
    for i, j, l in triangles:
        by_anchor_apex[(i, l)].append(j)

    kites = []
    for (i, l), middles in by_anchor_apex.items():
        for a in range(len(middles)):
            for b in range(a + 1, len(middles)):
                j, k = middles[a], middles[b]
                if j > k:
                    j, k = k, j
                if G[j, k] == 0 and G[k, j] == 0:
                    kites.append((i, j, k, l))
    return kites


def reaches(graph, src, dst) -> bool:
    """Whether a directed path src -> ... -> dst exists (endpoint matrix).

    Reads the live graph. causal-learn's cached ``dpath`` (behind
    ``is_ancestor_of``) is not a full transitive closure after incremental
    edits or ``reconstitute_dpath``, so it is not used for cycle or constraint
    checks. Only canonical directed edges (tail, arrow) count, which is all
    CDNOTS produces (no compound tail-and-arrow endpoints).
    """
    if src == dst:
        return True
    seen, stack = {src}, [src]
    while stack:
        u = stack.pop()
        for v in np.where((graph[u, :] == -1) & (graph[:, u] == 1))[0]:
            if v == dst:
                return True
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return False


def _closure(adj, start) -> set:
    """Nodes reachable from start along adj (adj[u, v]: u -> v), start included."""
    seen, stack = {start}, [start]
    while stack:
        for v in np.flatnonzero(adj[stack.pop()]):
            if int(v) not in seen:
                seen.add(int(v))
                stack.append(int(v))
    return seen


def completes_forbidden_path(graph, x, y, forbidden_ancestor_pairs) -> bool:
    """Whether orienting x -> y gives some pair (a, d) a path a ->...-> x -> y ->...-> d.

    One search for x's ancestors and one for y's descendants serve every pair.
    """
    if not forbidden_ancestor_pairs:
        return False
    directed = (graph == -1) & (graph.T == 1)
    ancestors_x = _closure(directed.T, x)
    descendants_y = _closure(directed, y)
    return any(
        a in ancestors_x and d in descendants_y for a, d in forbidden_ancestor_pairs
    )


def rebuild_dpath(cg) -> None:
    """Set causal-learn's ancestry cache to the full transitive closure."""
    g = cg.G.graph
    reach = ((g == -1) & (g.T == 1)) | np.eye(g.shape[0], dtype=bool)
    for k in range(g.shape[0]):
        reach |= reach[:, k : k + 1] & reach[k : k + 1, :]
    cg.G.dpath = reach.T.astype(int)  # dpath[j, i] == 1: i is an ancestor of j


def lag_copy_allowed(cg, x, y, background_knowledge=None) -> bool:
    """Whether an orientation x -> y may be copied onto the shifted pair (x, y).

    Lag-copy propagation keeps the window time-invariant, but only where the
    skeleton has the shifted edge and lag-specific background knowledge
    allows that direction: it must never add an adjacency or break a
    constraint the lag-0 pair did not see.
    """
    nodes = cg.G.nodes
    if cg.G.get_edge(nodes[x], nodes[y]) is None:
        return False
    if background_knowledge is not None and (
        background_knowledge.is_forbidden(nodes[x], nodes[y])
        or background_knowledge.is_required(nodes[y], nodes[x])
    ):
        return False
    return True


def meek(
    cg: CausalGraph,
    background_knowledge: BackgroundKnowledge | None = None,
    num_lags=0,
    forbidden_ancestor_pairs: list | None = None,
) -> CausalGraph:
    """Run Meek rules.

    Parameters
    ----------
    cg : CausalGraph
        Where ``graph[j,i]=1, graph[i,j]=-1`` means i --> j,
        ``graph[i,j] = graph[j,i] = -1`` means i --- j,
        ``graph[i,j] = graph[j,i] = 1`` means i <-> j.
    background_knowledge : BackgroundKnowledge or None
        Artificial background knowledge.

    Returns
    -------
    cg_new : CausalGraph
        Oriented graph with the same encoding as *cg*.
    """

    no_of_var = cg.G.num_vars
    assert no_of_var % (num_lags + 1) == 0
    no_of_inst_var = no_of_var // (num_lags + 1)

    cg_new = deepcopy(cg)

    adj = _build_adj_dict(cg_new)
    ut = _find_unshielded_triples_fast(cg_new, adj)
    tri = _find_triangles_fast(cg_new, adj)
    kite = _find_kites_fast(cg_new, tri)

    loop = True

    def _would_complete_forbidden_path(x, y) -> bool:
        return completes_forbidden_path(cg_new.G.graph, x, y, forbidden_ancestor_pairs)

    def naive_add_edge(x, y):
        if _would_complete_forbidden_path(x, y):
            return
        edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
        if edge1 is not None:
            if not reaches(cg_new.G.graph, y, x):
                cg_new.G.remove_edge(edge1)

        cg_new.G.add_edge(
            Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW)
        )

    def add_edge(x, y):
        # Assumption: x < no_of_inst_var
        naive_add_edge(x, y)

        if num_lags > 0:
            y_time_step = y // no_of_inst_var

            for i in range(num_lags - y_time_step):
                cx = x + (i + 1) * no_of_inst_var
                cy = y + (i + 1) * no_of_inst_var
                if lag_copy_allowed(cg_new, cx, cy, background_knowledge):
                    naive_add_edge(cx, cy)

    # Triples the majority rule could not call (uc_sepset). They are not
    # known non-colliders, so R1 and R3 must not orient through them.
    ambiguous = getattr(cg, "ambiguous_triples", ())

    # Edges R1 implied in both directions. They stay undirected: no rule may
    # orient them afterwards (PCMCI+ marks them as conflicts). Kept on the
    # returned graph, so later Meek calls and phase_three respect them too.
    # The graph stores them as plain undirected, so they render as o-o.
    blocked: set[tuple[int, int]] = set(getattr(cg, "conflict_edges", ()))

    counter = -1
    # Every productive pass orients at least one edge, so this is never the
    # binding stop; it only guards against a non-terminating bug.
    MAX_ITER = 1 + int((cg_new.G.graph != 0).sum())
    while loop and counter < MAX_ITER:
        counter += 1
        before = cg_new.G.graph.copy()
        # R1. The source i may be lagged: X(t-tau) -> Y(t) - Z(t) orients
        # Y -> Z just as a lag-0 source does. Implications are collected
        # before any is applied, so an edge implied both ways is caught
        # instead of being oriented by whichever triple is visited first.
        r1 = set()
        for i, j, k in ut:
            if j >= no_of_inst_var or (i, j, k) in ambiguous or (j, k) in blocked:
                continue
            if cg_new.is_fully_directed(i, j) and cg_new.is_undirected(j, k):
                if (background_knowledge is not None) and (
                    background_knowledge.is_forbidden(
                        cg_new.G.nodes[j], cg_new.G.nodes[k]
                    )
                    or background_knowledge.is_required(
                        cg_new.G.nodes[k], cg_new.G.nodes[j]
                    )
                ):
                    pass
                else:
                    r1.add((j, k))
        for j, k in sorted(r1):
            if (k, j) in r1:
                blocked.update({(j, k), (k, j)})
                continue
            if cg_new.is_undirected(j, k):
                add_edge(j, k)

        for i, j, k in tri:
            if i >= no_of_inst_var or k >= no_of_inst_var or (i, k) in blocked:
                continue
            if (
                cg_new.is_fully_directed(i, j)
                and cg_new.is_fully_directed(j, k)
                and cg_new.is_undirected(i, k)
            ):
                if (background_knowledge is not None) and (
                    background_knowledge.is_forbidden(
                        cg_new.G.nodes[i], cg_new.G.nodes[k]
                    )
                    or background_knowledge.is_required(
                        cg_new.G.nodes[k], cg_new.G.nodes[i]
                    )
                ):
                    pass
                else:
                    add_edge(i, k)

        for i, j, k, ll in kite:
            if (
                not (i < no_of_inst_var)
                or not (j < no_of_inst_var)
                or not (k < no_of_inst_var)
                or not (ll < no_of_inst_var)
            ):
                continue
            if (j, i, k) in ambiguous or (i, ll) in blocked:
                continue
            if (
                cg_new.is_undirected(i, j)
                and cg_new.is_undirected(i, k)
                and cg_new.is_fully_directed(j, ll)
                and cg_new.is_fully_directed(k, ll)
                and cg_new.is_undirected(i, ll)
            ):
                if (background_knowledge is not None) and (
                    background_knowledge.is_forbidden(
                        cg_new.G.nodes[i], cg_new.G.nodes[ll]
                    )
                    or background_knowledge.is_required(
                        cg_new.G.nodes[ll], cg_new.G.nodes[i]
                    )
                ):
                    pass
                else:
                    add_edge(i, ll)
        # Iterate to a fixed point: stop once a pass changes nothing.
        loop = not np.array_equal(before, cg_new.G.graph)
    cg_new.conflict_edges = blocked
    # Callers may write endpoints directly (phase three) and causal-learn's
    # incremental update is not a full closure: leave a correct ancestry cache.
    rebuild_dpath(cg_new)
    return cg_new


def definite_meek(
    cg: CausalGraph, background_knowledge: BackgroundKnowledge | None = None
) -> CausalGraph:
    """Apply Meek rules on definite unshielded triples.

    Parameters
    ----------
    cg : CausalGraph
        Where ``graph[j,i]=1, graph[i,j]=-1`` means i --> j,
        ``graph[i,j] = graph[j,i] = -1`` means i --- j,
        ``graph[i,j] = graph[j,i] = 1`` means i <-> j.
    background_knowledge : BackgroundKnowledge or None
        Artificial background knowledge.

    Returns
    -------
    CausalGraph
        Oriented graph with the same encoding as *cg*.
    """

    cg_new = deepcopy(cg)

    tri = cg_new.find_triangles()
    kite = cg_new.find_kites()

    loop = True

    while loop:
        loop = False
        for i, j, k in cg_new.definite_non_UC:
            if (
                cg_new.is_fully_directed(i, j)
                and cg_new.is_undirected(j, k)
                and not (
                    (background_knowledge is not None)
                    and (
                        background_knowledge.is_forbidden(
                            cg_new.G.nodes[j], cg_new.G.nodes[k]
                        )
                        or background_knowledge.is_required(
                            cg_new.G.nodes[k], cg_new.G.nodes[j]
                        )
                    )
                )
            ):
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[j], cg_new.G.nodes[k])
                if edge1 is not None:
                    if reaches(cg_new.G.graph, k, j):
                        continue
                    else:
                        cg_new.G.remove_edge(edge1)
                else:
                    continue
                cg_new.G.add_edge(
                    Edge(
                        cg_new.G.nodes[j],
                        cg_new.G.nodes[k],
                        Endpoint.TAIL,
                        Endpoint.ARROW,
                    )
                )
                loop = True
            elif (
                cg_new.is_fully_directed(k, j)
                and cg_new.is_undirected(j, i)
                and not (
                    (background_knowledge is not None)
                    and (
                        background_knowledge.is_forbidden(
                            cg_new.G.nodes[j], cg_new.G.nodes[i]
                        )
                        or background_knowledge.is_required(
                            cg_new.G.nodes[i], cg_new.G.nodes[j]
                        )
                    )
                )
            ):
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[j], cg_new.G.nodes[i])
                if edge1 is not None:
                    if reaches(cg_new.G.graph, i, j):
                        continue
                    else:
                        cg_new.G.remove_edge(edge1)
                else:
                    continue
                cg_new.G.add_edge(
                    Edge(
                        cg_new.G.nodes[j],
                        cg_new.G.nodes[i],
                        Endpoint.TAIL,
                        Endpoint.ARROW,
                    )
                )
                loop = True

        for i, j, k in tri:
            if (
                cg_new.is_fully_directed(i, j)
                and cg_new.is_fully_directed(j, k)
                and cg_new.is_undirected(i, k)
            ):
                if (background_knowledge is not None) and (
                    background_knowledge.is_forbidden(
                        cg_new.G.nodes[i], cg_new.G.nodes[k]
                    )
                    or background_knowledge.is_required(
                        cg_new.G.nodes[k], cg_new.G.nodes[i]
                    )
                ):
                    pass
                else:
                    edge1 = cg_new.G.get_edge(cg_new.G.nodes[i], cg_new.G.nodes[k])
                    if edge1 is not None:
                        if reaches(cg_new.G.graph, k, i):
                            continue
                        else:
                            cg_new.G.remove_edge(edge1)
                    else:
                        continue
                    cg_new.G.add_edge(
                        Edge(
                            cg_new.G.nodes[i],
                            cg_new.G.nodes[k],
                            Endpoint.TAIL,
                            Endpoint.ARROW,
                        )
                    )
                    loop = True

        for i, j, k, ll in kite:
            if (
                ((j, ll, k) in cg_new.definite_UC or (k, ll, j) in cg_new.definite_UC)
                and (
                    (j, i, k) in cg_new.definite_non_UC
                    or (k, i, j) in cg_new.definite_non_UC
                )
                and cg_new.is_undirected(i, ll)
            ):
                if (background_knowledge is not None) and (
                    background_knowledge.is_forbidden(
                        cg_new.G.nodes[i], cg_new.G.nodes[ll]
                    )
                    or background_knowledge.is_required(
                        cg_new.G.nodes[ll], cg_new.G.nodes[i]
                    )
                ):
                    pass
                else:
                    edge1 = cg_new.G.get_edge(cg_new.G.nodes[i], cg_new.G.nodes[ll])
                    if edge1 is not None:
                        if reaches(cg_new.G.graph, ll, i):
                            continue
                        else:
                            cg_new.G.remove_edge(edge1)
                    else:
                        continue
                    cg_new.G.add_edge(
                        Edge(
                            cg_new.G.nodes[i],
                            cg_new.G.nodes[ll],
                            Endpoint.TAIL,
                            Endpoint.ARROW,
                        )
                    )
                    loop = True

    return cg_new
