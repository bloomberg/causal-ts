"""
This file has been modified by Mohammad Fesanghary on July 28 2025.
Original Package: causal-learn 1.9.0
Original License: MIT License
"""

# SPDX-License-Identifier: MIT

from __future__ import annotations

from copy import deepcopy

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.Helper import sort_dict_ascending

from causalts.cdnots.meek import _build_adj_dict, _find_unshielded_triples_fast


def uc_sepset(
    cg: CausalGraph,
    priority: int = 2,
    background_knowledge: BackgroundKnowledge | None = None,
    num_lags=0,
    contemp_collider_rule: str | None = None,
    ci_test=None,
    alpha: float = 0.01,
    lagged_parents: dict | None = None,
    max_combinations: int = 1000,
) -> CausalGraph:
    """Run UC_sepset to orient unshielded colliders.

    Parameters
    ----------
    cg : CausalGraph
        A CausalGraph object.
    priority : int
        Rule of resolving conflicts between unshielded colliders
        (default 2). 0: overwrite, 1: orient bi-directed,
        2: prioritize existing colliders, 3: prioritize stronger
        colliders, 4: prioritize stronger* colliders.
    background_knowledge : BackgroundKnowledge, optional
        Artificial background knowledge.
    num_lags : int
        Number of lags in embedding.
    contemp_collider_rule : str or None
        If ``'majority'``, applies PCMCI+-style majority rule for
        contemporaneous unshielded triples: for each triple X-Y-Z
        where all three are at lag 0, test all subsets of
        contemporaneous neighbors as conditioning sets. A collider
        is declared only if < 50% of separating subsets contain Y.
        If = 50%, the triple is ambiguous and left unoriented.
        Requires ``ci_test`` and ``alpha`` to be provided.
    ci_test : object, optional
        CI test instance (required when ``contemp_collider_rule='majority'``).
    alpha : float
        Significance level for the majority rule CI tests.
    lagged_parents : dict, optional
        Lagged parents from Phase 1 (required for majority rule
        MCI-style conditioning).

    Returns
    -------
    cg_new : CausalGraph
        Where ``graph[j,i]=1`` and ``graph[i,j]=-1`` indicates i --> j,
        ``graph[i,j] = graph[j,i] = -1`` indicates i --- j,
        ``graph[i,j] = graph[j,i] = 1`` indicates i <-> j.
    """

    assert priority in [0, 1, 2, 3, 4]

    no_of_var = cg.G.num_vars
    assert no_of_var % (num_lags + 1) == 0
    if num_lags > 0 and priority == 1:
        raise NotImplementedError()

    no_of_inst_var = no_of_var // (num_lags + 1)

    cg_new = deepcopy(cg)

    R0 = []  # Records of possible orientations
    UC_dict = {}
    adj = _build_adj_dict(cg_new)
    UT = [
        (i, j, k)
        for (i, j, k) in _find_unshielded_triples_fast(cg_new, adj)
        if (i < k) and (j < no_of_inst_var)
    ]  # Not considering symmetric triples

    def naive_add_edge(x, y):
        edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
        if edge1 is not None:
            cg_new.G.remove_edge(edge1)

        edge2 = cg_new.G.get_edge(cg_new.G.nodes[y], cg_new.G.nodes[x])
        if edge2 is not None:
            cg_new.G.remove_edge(edge2)
        cg_new.G.add_edge(
            Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW)
        )

    def add_edge(x, y):
        # Assumption: x < no_of_inst_var
        naive_add_edge(x, y)

        if num_lags > 0:
            x_time_step = x // no_of_inst_var
            y_time_step = y // no_of_inst_var

            for i in range(num_lags - max(x_time_step, y_time_step)):
                naive_add_edge(
                    x + (i + 1) * no_of_inst_var, y + (i + 1) * no_of_inst_var
                )

    def get_sepset(x, y):
        sepset = cg.sepset[x, y]
        if sepset is None:
            return ()
        return sepset

    def _is_contemp_triple(x, y, z):
        """True if the middle node y is in the instantaneous (lag-0) slice.
        Matches PCMCI+'s majority rule which applies whenever y is contemporaneous,
        regardless of whether x or z are lagged."""
        return y < no_of_inst_var

    def _majority_check(x, y, z):
        """PCMCI+-style majority rule for triples with contemporaneous middle y.

        Matches PCMCI+'s _pcalg_colliders with contemp_collider_rule='majority':
        - Tests all subsets of contemporaneous neighbors of j (the lag-0 endpoint)
          and, when x is also at lag 0, subsets of x's contemporaneous neighbors.
        - For each subset S, runs CI test conditioned on S + lagged parents (MCI).
        - A collider is declared only if fraction of separating subsets containing
          Y is < 0.5. Fraction = 0.5 or no separating set → ambiguous (not collider).
        """
        from itertools import combinations as _combs

        # j is always the lag-0 endpoint (y is at lag 0 by _is_contemp_triple)
        # x and z may be lagged. Identify the lag-0 side (j) and the other (i).
        # PCMCI+ builds subsets from neighbors of j and (if i is also lag-0) of i.
        j = z if z < no_of_inst_var else x  # the contemporaneous endpoint
        i = x if z < no_of_inst_var else z  # could be lagged

        # Contemporaneous neighbors of j, excluding only the direct test partner i.
        # y is NOT excluded — PCMCI+ includes y in conditioning subsets.
        contemp_neighbors_j = [
            p
            for p in cg_new.neighbors(j)
            if p < no_of_inst_var and p != (i if i < no_of_inst_var else -1)
        ]
        # If i is also lag-0, include its contemp neighbors too (excluding j)
        contemp_neighbors_i = []
        if i < no_of_inst_var:
            contemp_neighbors_i = [
                p for p in cg_new.neighbors(i) if p < no_of_inst_var and p != j
            ]

        # Build all unique subsets (union of both sides)
        all_subsets = []
        for card in range(len(contemp_neighbors_j) + 1):
            for S in _combs(contemp_neighbors_j, card):
                if list(S) not in all_subsets:
                    all_subsets.append(list(S))
        for card in range(len(contemp_neighbors_i) + 1):
            for S in _combs(contemp_neighbors_i, card):
                if list(S) not in all_subsets:
                    all_subsets.append(list(S))

        if not all_subsets:
            all_subsets = [[]]

        # Lagged base conditions from Phase 1 parents (MCI-style)
        base_j = [p for p in (lagged_parents or {}).get(j % no_of_inst_var, [])]
        base_i = [p for p in (lagged_parents or {}).get(i % no_of_inst_var, [])]

        # Test each subset (capped at max_combinations to avoid 2^k blowup)
        separating_subsets = []
        for S in all_subsets[:max_combinations]:
            Z = list(S)
            Z += [p for p in base_j if p not in Z]
            Z += [p for p in base_i if p not in Z]
            p_val, _ = ci_test(x, z, Z)
            if p_val > alpha:
                separating_subsets.append(S)

        if len(separating_subsets) == 0:
            return False  # no separating set found → ambiguous, do NOT orient

        fraction = sum(y in S for S in separating_subsets) / len(separating_subsets)
        # fraction < 0.5: y absent from majority → collider
        # fraction == 0.5: ambiguous → do NOT orient
        # fraction > 0.5: y present in majority → non-collider
        return fraction < 0.5

    for x, y, z in UT:
        if (background_knowledge is not None) and (
            background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y])
            or background_knowledge.is_forbidden(cg_new.G.nodes[z], cg_new.G.nodes[y])
            or background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[x])
            or background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[z])
        ):
            continue
        if cg_new.is_fully_directed(x, y) and cg_new.is_fully_directed(z, y):
            continue

        # For contemporaneous triples with majority rule: use fraction-based check
        if (
            contemp_collider_rule == "majority"
            and _is_contemp_triple(x, y, z)
            and ci_test is not None
        ):
            is_collider = _majority_check(x, y, z)
        else:
            # Standard check: collider if y not in any separating set
            is_collider = all(y not in S for S in get_sepset(x, z))

        if is_collider:
            if priority == 0:  # 0: overwrite
                # Fully orient the edge irrespective of what have been oriented
                add_edge(x, y)
                add_edge(z, y)

            elif priority == 1:  # 1: orient bi-directed
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                if edge1 is not None:
                    if (
                        cg_new.G.graph[x, y] == Endpoint.TAIL.value
                        and cg_new.G.graph[y, x] == Endpoint.TAIL.value
                    ):
                        cg_new.G.remove_edge(edge1)
                        cg_new.G.add_edge(
                            Edge(
                                cg_new.G.nodes[x],
                                cg_new.G.nodes[y],
                                Endpoint.TAIL,
                                Endpoint.ARROW,
                            )
                        )
                    elif (
                        cg_new.G.graph[x, y] == Endpoint.ARROW.value
                        and cg_new.G.graph[y, x] == Endpoint.TAIL.value
                    ):
                        cg_new.G.remove_edge(edge1)
                        cg_new.G.add_edge(
                            Edge(
                                cg_new.G.nodes[x],
                                cg_new.G.nodes[y],
                                Endpoint.ARROW,
                                Endpoint.ARROW,
                            )
                        )
                else:
                    add_edge(x, y)

                edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                if edge2 is not None:
                    if (
                        cg_new.G.graph[z, y] == Endpoint.TAIL.value
                        and cg_new.G.graph[y, z] == Endpoint.TAIL.value
                    ):
                        cg_new.G.remove_edge(edge2)
                        cg_new.G.add_edge(
                            Edge(
                                cg_new.G.nodes[z],
                                cg_new.G.nodes[y],
                                Endpoint.TAIL,
                                Endpoint.ARROW,
                            )
                        )
                    elif (
                        cg_new.G.graph[z, y] == Endpoint.ARROW.value
                        and cg_new.G.graph[y, z] == Endpoint.TAIL.value
                    ):
                        cg_new.G.remove_edge(edge2)
                        cg_new.G.add_edge(
                            Edge(
                                cg_new.G.nodes[z],
                                cg_new.G.nodes[y],
                                Endpoint.ARROW,
                                Endpoint.ARROW,
                            )
                        )
                else:
                    add_edge(z, y)

            elif priority == 2:  # 2: prioritize existing
                if (not cg_new.is_fully_directed(y, x)) and (
                    not cg_new.is_fully_directed(y, z)
                ):
                    edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                    if edge1 is not None:
                        cg_new.G.remove_edge(edge1)
                    # Orient if edges are not already oriented oppositely
                    add_edge(x, y)
                    add_edge(z, y)
            else:
                R0.append((x, y, z))

    if priority in [0, 1, 2]:
        return cg_new

    else:
        if priority == 3:  # 3. Order colliders by p_{xz|y} in ascending order
            for x, y, z in R0:
                cond = cg_new.find_cond_sets_with_mid(x, z, y)
                UC_dict[(x, y, z)] = max([cg_new.ci_test(x, z, S) for S in cond])
            UC_dict = sort_dict_ascending(UC_dict)

        else:  # 4. Order colliders by p_{xy|not y} in descending order
            for x, y, z in R0:
                cond = cg_new.find_cond_sets_without_mid(x, z, y)
                UC_dict[(x, y, z)] = max([cg_new.ci_test(x, z, S) for S in cond])
            UC_dict = sort_dict_ascending(UC_dict, descending=True)

        for x, y, z in UC_dict.keys():
            if (background_knowledge is not None) and (
                background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y])
                or background_knowledge.is_forbidden(
                    cg_new.G.nodes[z], cg_new.G.nodes[y]
                )
                or background_knowledge.is_required(
                    cg_new.G.nodes[y], cg_new.G.nodes[x]
                )
                or background_knowledge.is_required(
                    cg_new.G.nodes[y], cg_new.G.nodes[z]
                )
            ):
                continue
            if (not cg_new.is_fully_directed(y, x)) and (
                not cg_new.is_fully_directed(y, z)
            ):
                # Orient only if the edges have not been oriented the other way around
                add_edge(x, y)
                add_edge(z, y)

        return cg_new


def maxp(
    cg: CausalGraph, priority: int = 3, background_knowledge: BackgroundKnowledge = None
):
    """Run MaxP to orient unshielded colliders.

    Parameters
    ----------
    cg : CausalGraph
        A CausalGraph object.
    priority : int
        Rule of resolving conflicts between unshielded colliders
        (default 3). 0: overwrite, 1: orient bi-directed,
        2: prioritize existing colliders, 3: prioritize stronger
        colliders, 4: prioritize stronger* colliders.
    background_knowledge : BackgroundKnowledge, optional
        Artificial background knowledge.

    Returns
    -------
    cg_new : CausalGraph
        Where ``graph[j,i]=1`` and ``graph[i,j]=-1`` indicates i --> j,
        ``graph[i,j] = graph[j,i] = -1`` indicates i --- j,
        ``graph[i,j] = graph[j,i] = 1`` indicates i <-> j.
    """

    assert priority in [0, 1, 2, 3, 4]

    cg_new = deepcopy(cg)
    UC_dict = {}
    UT = [
        (i, j, k) for (i, j, k) in cg_new.find_unshielded_triples() if i < k
    ]  # Not considering symmetric triples

    for x, y, z in UT:
        if (background_knowledge is not None) and (
            background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y])
            or background_knowledge.is_forbidden(cg_new.G.nodes[z], cg_new.G.nodes[y])
            or background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[x])
            or background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[z])
        ):
            continue

        cond_with_y = cg_new.find_cond_sets_with_mid(x, z, y)
        cond_without_y = cg_new.find_cond_sets_without_mid(x, z, y)

        max_p_contain_y = max([cg_new.ci_test(x, z, S) for S in cond_with_y])
        max_p_not_contain_y = max([cg_new.ci_test(x, z, S) for S in cond_without_y])

        if max_p_not_contain_y > max_p_contain_y:
            if priority == 0:  # 0: overwrite
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                if edge1 is not None:
                    cg_new.G.remove_edge(edge1)
                edge2 = cg_new.G.get_edge(cg_new.G.nodes[y], cg_new.G.nodes[x])
                if edge2 is not None:
                    cg_new.G.remove_edge(edge2)
                # Fully orient the edge irrespective of what have been oriented
                cg_new.G.add_edge(
                    Edge(
                        cg_new.G.nodes[x],
                        cg_new.G.nodes[y],
                        Endpoint.TAIL,
                        Endpoint.ARROW,
                    )
                )

                edge3 = cg_new.G.get_edge(cg_new.G.nodes[y], cg_new.G.nodes[z])
                if edge3 is not None:
                    cg_new.G.remove_edge(edge3)
                edge4 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                if edge4 is not None:
                    cg_new.G.remove_edge(edge4)
                cg_new.G.add_edge(
                    Edge(
                        cg_new.G.nodes[z],
                        cg_new.G.nodes[y],
                        Endpoint.TAIL,
                        Endpoint.ARROW,
                    )
                )

            elif priority == 1:  # 1: orient bi-directed
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                if edge1 is not None:
                    if (
                        cg_new.G.graph[x, y] == Endpoint.TAIL.value
                        and cg_new.G.graph[y, x] == Endpoint.TAIL.value
                    ):
                        cg_new.G.remove_edge(edge1)
                        cg_new.G.add_edge(
                            Edge(
                                cg_new.G.nodes[x],
                                cg_new.G.nodes[y],
                                Endpoint.TAIL,
                                Endpoint.ARROW,
                            )
                        )
                    elif (
                        cg_new.G.graph[x, y] == Endpoint.ARROW.value
                        and cg_new.G.graph[y, x] == Endpoint.TAIL.value
                    ):
                        cg_new.G.remove_edge(edge1)
                        cg_new.G.add_edge(
                            Edge(
                                cg_new.G.nodes[x],
                                cg_new.G.nodes[y],
                                Endpoint.ARROW,
                                Endpoint.ARROW,
                            )
                        )
                else:
                    cg_new.G.add_edge(
                        Edge(
                            cg_new.G.nodes[x],
                            cg_new.G.nodes[y],
                            Endpoint.TAIL,
                            Endpoint.ARROW,
                        )
                    )

                edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                if edge2 is not None:
                    if (
                        cg_new.G.graph[z, y] == Endpoint.TAIL.value
                        and cg_new.G.graph[y, z] == Endpoint.TAIL.value
                    ):
                        cg_new.G.remove_edge(edge2)
                        cg_new.G.add_edge(
                            Edge(
                                cg_new.G.nodes[z],
                                cg_new.G.nodes[y],
                                Endpoint.TAIL,
                                Endpoint.ARROW,
                            )
                        )
                    elif (
                        cg_new.G.graph[z, y] == Endpoint.ARROW.value
                        and cg_new.G.graph[y, z] == Endpoint.TAIL.value
                    ):
                        cg_new.G.remove_edge(edge2)
                        cg_new.G.add_edge(
                            Edge(
                                cg_new.G.nodes[z],
                                cg_new.G.nodes[y],
                                Endpoint.ARROW,
                                Endpoint.ARROW,
                            )
                        )
                else:
                    cg_new.G.add_edge(
                        Edge(
                            cg_new.G.nodes[z],
                            cg_new.G.nodes[y],
                            Endpoint.TAIL,
                            Endpoint.ARROW,
                        )
                    )

            elif priority == 2:  # 2: prioritize existing
                if (not cg_new.is_fully_directed(y, x)) and (
                    not cg_new.is_fully_directed(y, z)
                ):
                    edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                    if edge1 is not None:
                        cg_new.G.remove_edge(edge1)
                    # Ensure edges are not oriented oppositely before proceeding
                    cg_new.G.add_edge(
                        Edge(
                            cg_new.G.nodes[x],
                            cg_new.G.nodes[y],
                            Endpoint.TAIL,
                            Endpoint.ARROW,
                        )
                    )

                    edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                    if edge2 is not None:
                        cg_new.G.remove_edge(edge2)
                    cg_new.G.add_edge(
                        Edge(
                            cg_new.G.nodes[z],
                            cg_new.G.nodes[y],
                            Endpoint.TAIL,
                            Endpoint.ARROW,
                        )
                    )

            elif priority == 3:
                UC_dict[(x, y, z)] = max_p_contain_y

            elif priority == 4:
                UC_dict[(x, y, z)] = max_p_not_contain_y

    if priority in [0, 1, 2]:
        return cg_new

    else:
        if priority == 3:  # 3. Order colliders by p_{xz|y} in ascending order
            UC_dict = sort_dict_ascending(UC_dict)
        else:  # 4. Order colliders by p_{xz|not y} in descending order
            UC_dict = sort_dict_ascending(UC_dict, True)

        for x, y, z in UC_dict.keys():
            if (background_knowledge is not None) and (
                background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y])
                or background_knowledge.is_forbidden(
                    cg_new.G.nodes[z], cg_new.G.nodes[y]
                )
                or background_knowledge.is_required(
                    cg_new.G.nodes[y], cg_new.G.nodes[x]
                )
                or background_knowledge.is_required(
                    cg_new.G.nodes[y], cg_new.G.nodes[z]
                )
            ):
                continue

            if (not cg_new.is_fully_directed(y, x)) and (
                not cg_new.is_fully_directed(y, z)
            ):
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                if edge1 is not None:
                    cg_new.G.remove_edge(edge1)
                # Orient only if the edges have not been oriented the other way around
                cg_new.G.add_edge(
                    Edge(
                        cg_new.G.nodes[x],
                        cg_new.G.nodes[y],
                        Endpoint.TAIL,
                        Endpoint.ARROW,
                    )
                )

                edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                if edge2 is not None:
                    cg_new.G.remove_edge(edge2)
                cg_new.G.add_edge(
                    Edge(
                        cg_new.G.nodes[z],
                        cg_new.G.nodes[y],
                        Endpoint.TAIL,
                        Endpoint.ARROW,
                    )
                )

        return cg_new


def definite_maxp(
    cg: CausalGraph,
    alpha: float,
    priority: int = 4,
    background_knowledge: BackgroundKnowledge = None,
) -> CausalGraph:
    """Run Definite_MaxP to orient unshielded colliders.

    Parameters
    ----------
    cg : CausalGraph
        A CausalGraph object.
    alpha : float
        Significance level for definite collider detection.
    priority : int
        Rule of resolving conflicts between unshielded colliders
        (default 4). 2: prioritize existing colliders, 3: prioritize
        stronger colliders, 4: prioritize stronger* colliders.
    background_knowledge : BackgroundKnowledge, optional
        Artificial background knowledge.

    Returns
    -------
    cg_new : CausalGraph
        Where ``graph[j,i]=1`` and ``graph[i,j]=-1`` indicates i --> j,
        ``graph[i,j] = graph[j,i] = -1`` indicates i --- j,
        ``graph[i,j] = graph[j,i] = 1`` indicates i <-> j.
    """

    assert 1 > alpha >= 0
    assert priority in [2, 3, 4]

    cg_new = deepcopy(cg)
    UC_dict = {}
    UT = [
        (i, j, k) for (i, j, k) in cg_new.find_unshielded_triples() if i < k
    ]  # Not considering symmetric triples

    for x, y, z in UT:
        cond_with_y = cg_new.find_cond_sets_with_mid(x, z, y)
        cond_without_y = cg_new.find_cond_sets_without_mid(x, z, y)
        max_p_contain_y = 0
        max_p_not_contain_y = 0
        uc_bool = True
        nuc_bool = True

        for S in cond_with_y:
            p = cg_new.ci_test(x, z, S)
            if p > alpha:
                uc_bool = False
                break
            elif p > max_p_contain_y:
                max_p_contain_y = p

        for S in cond_without_y:
            p = cg_new.ci_test(x, z, S)
            if p > alpha:
                nuc_bool = False
                if not uc_bool:
                    break  # ambiguous triple
            if p > max_p_not_contain_y:
                max_p_not_contain_y = p

        if uc_bool:
            if nuc_bool:
                if max_p_not_contain_y > max_p_contain_y:
                    if priority in [2, 3]:
                        UC_dict[(x, y, z)] = max_p_contain_y
                    if priority == 4:
                        UC_dict[(x, y, z)] = max_p_not_contain_y
                else:
                    cg_new.definite_non_UC.append((x, y, z))
            else:
                if priority in [2, 3]:
                    UC_dict[(x, y, z)] = max_p_contain_y
                if priority == 4:
                    UC_dict[(x, y, z)] = max_p_not_contain_y

        elif nuc_bool:
            cg_new.definite_non_UC.append((x, y, z))

    if priority == 3:  # 3. Order colliders by p_{xz|y} in ascending order
        UC_dict = sort_dict_ascending(UC_dict)
    elif priority == 4:  # 4. Order colliders by p_{xz|not y} in descending order
        UC_dict = sort_dict_ascending(UC_dict, True)

    for x, y, z in UC_dict.keys():
        if (background_knowledge is not None) and (
            background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y])
            or background_knowledge.is_forbidden(cg_new.G.nodes[z], cg_new.G.nodes[y])
            or background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[x])
            or background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[z])
        ):
            continue

        if (not cg_new.is_fully_directed(y, x)) and (
            not cg_new.is_fully_directed(y, z)
        ):
            edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
            if edge1 is not None:
                cg_new.G.remove_edge(edge1)
            # Orient only if the edges have not been oriented the other way around
            cg_new.G.add_edge(
                Edge(
                    cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW
                )
            )

            edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
            if edge2 is not None:
                cg_new.G.remove_edge(edge2)
            cg_new.G.add_edge(
                Edge(
                    cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW
                )
            )

            cg_new.definite_UC.append((x, y, z))

    return cg_new
