"""
This file has been modified by Mohammad Fesanghary on July 28 2025.
Original Package: causal-learn 1.9.0
Original License: MIT License
"""

# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

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
        if not forbidden_ancestor_pairs:
            return False
        node_x = cg_new.G.nodes[x]
        node_y = cg_new.G.nodes[y]
        for a_idx, d_idx in forbidden_ancestor_pairs:
            node_a = cg_new.G.nodes[a_idx]
            node_d = cg_new.G.nodes[d_idx]
            a_reaches_x = (a_idx == x) or cg_new.G.is_ancestor_of(node_a, node_x)
            y_reaches_d = (d_idx == y) or cg_new.G.is_ancestor_of(node_y, node_d)
            if a_reaches_x and y_reaches_d:
                return True
        return False

    def naive_add_edge(x, y):
        if _would_complete_forbidden_path(x, y):
            return
        edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
        if edge1 is not None:
            if not cg_new.G.is_ancestor_of(cg_new.G.nodes[y], cg_new.G.nodes[x]):
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
                naive_add_edge(
                    x + (i + 1) * no_of_inst_var, y + (i + 1) * no_of_inst_var
                )

    added: dict[tuple[int, int], int] = {}  # Prevent infinite loops from buggy logic.
    counter = -1
    MAX_ITER = 500
    while loop and counter < MAX_ITER:
        counter += 1
        loop = False
        for i, j, k in ut:
            if i >= no_of_inst_var or j >= no_of_inst_var:
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
                    add_edge(j, k)
                    # loop = True
                    loop = False if added.get((j, k), 0) == counter - 1 else True
                    added[j, k] = counter

        for i, j, k in tri:
            if i >= no_of_inst_var or k >= no_of_inst_var:
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
                    loop = False if added.get((i, k), 0) == counter - 1 else True
                    added[i, k] = counter

        for i, j, k, ll in kite:
            if (
                not (i < no_of_inst_var)
                or not (j < no_of_inst_var)
                or not (k < no_of_inst_var)
                or not (ll < no_of_inst_var)
            ):
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
                    loop = False if added.get((i, ll), 0) == counter - 1 else True
                    added[i, ll] = counter
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
                    if cg_new.G.is_ancestor_of(cg_new.G.nodes[k], cg_new.G.nodes[j]):
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
                    if cg_new.G.is_ancestor_of(cg_new.G.nodes[i], cg_new.G.nodes[j]):
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
                        if cg_new.G.is_ancestor_of(
                            cg_new.G.nodes[k], cg_new.G.nodes[i]
                        ):
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
                        if cg_new.G.is_ancestor_of(
                            cg_new.G.nodes[ll], cg_new.G.nodes[i]
                        ):
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
