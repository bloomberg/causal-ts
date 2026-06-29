"""
This file has been modified by Mohammad Fesanghary on July 28 2025.
Original Package: causal-learn 1.9.0
Original License: MIT License
"""

# SPDX-License-Identifier: MIT

from __future__ import annotations

import time
from copy import deepcopy
from itertools import combinations, islice
from typing import List

import numpy as np
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.Helper import append_value
from numpy import ndarray
from tqdm.auto import tqdm


def initialize_graph(
    data: ndarray,
    # indep_test: None
    node_names: List[str] | None = None,
) -> CausalGraph:
    """Initialize a CausalGraph object for the given data.

    Parameters
    ----------
    data : numpy.ndarray, shape (n_samples, n_features)
        The lag-embedded input data (includes lagged copies and optional C column).
    node_names : list of str, optional
        Name for each feature (used as node labels in the graph).

    Returns
    -------
    cg : CausalGraph
        Fully connected graph where:
        ``cg.G.graph[j,i]=0`` and ``cg.G.graph[i,j]=1`` indicates i → j,
        ``cg.G.graph[i,j] = cg.G.graph[j,i] = -1`` indicates i — j,
        ``cg.G.graph[i,j] = cg.G.graph[j,i] = 1`` indicates i ↔ j.
    """
    assert isinstance(data, np.ndarray)
    no_of_var = data.shape[1]

    if node_names is None:
        node_names = [f"X{i}" for i in range(no_of_var)]

    cg = CausalGraph(no_of_var, node_names)
    # cg.set_ind_test(indep_test)
    return cg


def skeleton_cnst(cgg, data, num_lags, include_C=True, num_c_cols=1):
    """Modify the causal graph by removing edges based on temporal constraints.

    Parameters
    ----------
    cgg : CausalGraph
        The input causal graph object.
    data : ndarray
        The dataset associated with the causal graph.
    num_lags : int
        The number of lagged variables in the time series.
    include_C : bool
        Whether the time index variable C is included in the data.
    num_c_cols : int
        Number of C columns (nonstationarity indicators). Default 1.

    Returns
    -------
    CausalGraph
        A modified causal graph with updated edges.
    """
    no_of_var = cgg.G.num_vars
    assert no_of_var % (num_lags + 1) == 0
    num_of_inst_vars = no_of_var // (num_lags + 1)

    cg = deepcopy(cgg)

    if include_C:
        # Remove cross-lag C↔X edges (for all C columns, all non-C X)
        for i in range(num_lags + 1):
            for j in range(num_lags + 1):
                if i == j:
                    continue

                for c_col in range(num_c_cols):
                    c_indx = (i + 1) * num_of_inst_vars - num_c_cols + c_col
                    for x_indx in range(
                        j * num_of_inst_vars, (j + 1) * num_of_inst_vars - num_c_cols
                    ):
                        edge1 = cg.G.get_edge(cg.G.nodes[c_indx], cg.G.nodes[x_indx])
                        if edge1 is not None:
                            cg.G.remove_edge(edge1)

        # Remove all C↔C edges (same and cross lag — C cols are deterministic)
        all_c_nodes = [
            (lag + 1) * num_of_inst_vars - num_c_cols + col
            for lag in range(num_lags + 1)
            for col in range(num_c_cols)
        ]
        for a in range(len(all_c_nodes)):
            for b in range(a + 1, len(all_c_nodes)):
                edge1 = cg.G.get_edge(
                    cg.G.nodes[all_c_nodes[a]], cg.G.nodes[all_c_nodes[b]]
                )
                if edge1 is not None:
                    cg.G.remove_edge(edge1)

    return cg


def skeleton_discovery(
    cg: CausalGraph,
    #     data: ndarray,
    alpha: float,
    ci_test: None,
    stable: bool = True,
    background_knowledge: BackgroundKnowledge | None = None,
    verbose: bool = False,
    show_progress: bool = True,
    return_pvals=False,
    max_degree=None,
    num_lags=0,
    max_combinations: int | None = 20,
) -> CausalGraph:
    """
    Perform skeleton discovery.

    Parameters
    ----------
    cg : CausalGraph
        Initial fully connected graph (apply temporal constraints first).
    alpha : float
        Significance level for independence tests, in (0, 1).
    stable : bool
        If True, run stabilised skeleton discovery (default True).
    background_knowledge : BackgroundKnowledge, optional
        Domain constraints on forbidden/required edges.
    verbose : bool
        If True, print each CI test result.
    show_progress : bool
        If True, show a tqdm progress bar.
    return_pvals : bool
        If True, also return all CI test results as a list of
        ``(effect, cause, condition_set, p_value)`` tuples.
    max_degree : int, optional
        Maximum conditioning-set depth.
    num_lags : int
        Number of lags used in the time embedding.
    max_combinations : int or None
        Maximum conditioning-set combinations to test per pair per depth.
        Default 20 (prevents combinatorial explosion at hub nodes).

    Returns
    -------
    cg : CausalGraph
        Skeleton graph where:
        ``cg.G.graph[j,i]=0`` and ``cg.G.graph[i,j]=1`` indicates i → j,
        ``cg.G.graph[i,j] = cg.G.graph[j,i] = -1`` indicates i — j,
        ``cg.G.graph[i,j] = cg.G.graph[j,i] = 1`` indicates i ↔ j.

    """

    #     assert type(data) == np.ndarray
    assert 0 < alpha < 1
    assert ci_test is not None, "ci_test must be provided"

    no_of_var = cg.G.num_vars
    assert no_of_var % (num_lags + 1) == 0
    no_of_inst_var = no_of_var // (num_lags + 1)

    cg = deepcopy(cg)

    depth = -1
    pbar = tqdm(total=no_of_inst_var) if show_progress else None
    use_batch = (
        stable
        and hasattr(ci_test, "batch_test")
        and getattr(ci_test, "enable_batching", True)
    )

    def naive_remove_edge(x, y):
        edge = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
        if edge is not None:
            cg.G.remove_edge(edge)
        edge = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
        if edge is not None:
            cg.G.remove_edge(edge)

    def remove_edge(x, y):
        # Assumption: x < no_of_inst_var
        naive_remove_edge(x, y)
        if num_lags > 0:
            y_time_step = y // no_of_inst_var

            for i in range(num_lags - y_time_step):
                naive_remove_edge(
                    x + (i + 1) * no_of_inst_var, y + (i + 1) * no_of_inst_var
                )

    pvals = []
    parent_strengths = {}  # (x, y) -> min |stat| across depths, for sorting

    def _sort_neighbors(neighbors, node):
        """Sort neighbors by cached test statistic strength (strongest first)."""
        if max_combinations is None or not parent_strengths:
            return neighbors
        return sorted(
            neighbors,
            key=lambda n: parent_strengths.get((min(node, n), max(node, n)), 0),
            reverse=True,
        )

    def _update_strength(x, y, stat):
        """Track minimum |stat| for neighbor sorting."""
        val = abs(stat) if stat is not None and stat != 0 else 0
        key = (min(x, y), max(x, y))
        if key not in parent_strengths or val < parent_strengths[key]:
            parent_strengths[key] = val

    while cg.max_degree() - 1 > depth:
        if max_degree is not None and depth >= max_degree:
            break

        depth += 1
        edge_removal = []
        if show_progress:
            pbar.reset()

        if use_batch:
            # --- Batched path (stable mode + ci_test supports batch_test) ---
            # Process one (x, y) pair at a time to bound peak memory.
            # Accumulating all depth-level tests into one list before calling
            # batch_test creates multi-GB Python data structures at high d
            # (e.g. d=50 hub with degree 23 generates 600K+ tests at depth=5).
            # Per-pair batching caps batch_test input at C(max_deg-1, depth)
            # entries while preserving stable-mode correctness (edge removals
            # are still deferred until all pairs at this depth are tested).
            pair_results = {}
            for x in range(no_of_inst_var):
                neigh_x = cg.neighbors(x)
                if len(neigh_x) < depth - 1:
                    continue
                for y in neigh_x:
                    if background_knowledge is not None and (
                        background_knowledge.is_forbidden(cg.G.nodes[x], cg.G.nodes[y])
                        and background_knowledge.is_forbidden(
                            cg.G.nodes[y], cg.G.nodes[x]
                        )
                    ):
                        edge_removal.append((x, y))
                        continue

                    pair_specs = []
                    pair_inputs = []

                    neigh_x_noy = np.delete(neigh_x, np.where(neigh_x == y))
                    neigh_x_noy = _sort_neighbors(neigh_x_noy, x)
                    comb_iter = combinations(neigh_x_noy, depth)
                    if max_combinations is not None:
                        comb_iter = islice(comb_iter, max_combinations)
                    for S in comb_iter:
                        pair_specs.append((x, y, S, 1))
                        pair_inputs.append((x, y, list(S)))

                    neigh_y = cg.neighbors(y)
                    neigh_y_nox = np.delete(neigh_y, np.where(neigh_y == x))
                    neigh_y_nox = _sort_neighbors(neigh_y_nox, y)
                    comb_iter = combinations(neigh_y_nox, depth)
                    if max_combinations is not None:
                        comb_iter = islice(comb_iter, max_combinations)
                    for S in comb_iter:
                        pair_specs.append((x, y, S, 2))
                        pair_inputs.append((x, y, list(S)))

                    if not pair_inputs:
                        continue

                    pair_batch_results = ci_test.batch_test(pair_inputs)

                    key = (x, y)
                    pair_results[key] = ([], [])
                    for idx, (_, _, S, phase) in enumerate(pair_specs):
                        pair_results[key][phase - 1].append(
                            (S, pair_batch_results[idx])
                        )

            if show_progress:
                pbar.reset()
            for x in range(no_of_inst_var):
                if show_progress:
                    pbar.update()
                    pbar.set_description(f"Depth={depth}, working on node {x}")
                neigh_x = cg.neighbors(x)
                if len(neigh_x) < depth - 1:
                    continue
                for y in neigh_x:
                    key = (x, y)
                    if key not in pair_results:
                        continue
                    phase1_results, phase2_results = pair_results[key]
                    sepsets = set()

                    for S, (p, stat) in phase1_results:
                        if return_pvals:
                            pvals.append((x, y, S, p))
                        _update_strength(x, y, stat)
                        if p > alpha:
                            if verbose:
                                print(f"{x} ind {y} | {S} with p-value {p:.6f}")
                            edge_removal.append((x, y))
                            for s in S:
                                sepsets.add(s)
                        else:
                            if verbose:
                                print(f"{x} dep {y} | {S} with p-value {p:.6f}")
                    append_value(cg.sepset, x, y, tuple(sepsets))
                    append_value(cg.sepset, y, x, tuple(sepsets))

                    for S, (p, stat) in phase2_results:
                        if return_pvals:
                            pvals.append((x, y, S, p))
                        _update_strength(x, y, stat)
                        if p > alpha:
                            if verbose:
                                print(f"{x} ind {y} | {S} with p-value {p:.6f}")
                            edge_removal.append((x, y))
                            for s in S:
                                sepsets.add(s)
                        else:
                            if verbose:
                                print(f"{x} dep {y} | {S} with p-value {p:.6f}")
                    append_value(cg.sepset, x, y, tuple(sepsets))
                    append_value(cg.sepset, y, x, tuple(sepsets))
            pair_results.clear()

        else:
            # --- Sequential path (non-stable or no batch_test) ---
            for x in range(no_of_inst_var):
                if show_progress:
                    pbar.update()
                if show_progress:
                    pbar.set_description(f"Depth={depth}, working on node {x}")
                neigh_x = cg.neighbors(x)
                if len(neigh_x) < depth - 1:
                    continue
                for y in neigh_x:
                    knowledge_ban_edge = False
                    sepsets = set()
                    if background_knowledge is not None and (
                        background_knowledge.is_forbidden(cg.G.nodes[x], cg.G.nodes[y])
                        and background_knowledge.is_forbidden(
                            cg.G.nodes[y], cg.G.nodes[x]
                        )
                    ):
                        knowledge_ban_edge = True
                    if knowledge_ban_edge:
                        if not stable:
                            remove_edge(x, y)
                            append_value(cg.sepset, x, y, ())
                            append_value(cg.sepset, y, x, ())
                            break
                        else:
                            edge_removal.append((x, y))

                    neigh_x_noy = np.delete(neigh_x, np.where(neigh_x == y))
                    neigh_x_noy = _sort_neighbors(neigh_x_noy, x)
                    edge_removed = False
                    for comb_idx, S in enumerate(combinations(neigh_x_noy, depth)):
                        if (
                            max_combinations is not None
                            and comb_idx >= max_combinations
                        ):
                            break
                        p, stat = ci_test(x, y, S)
                        _update_strength(x, y, stat)
                        if return_pvals:
                            pvals.append((x, y, S, p))

                        if p > alpha:
                            if verbose:
                                print(f"{x} ind {y} | {S} with p-value {p:.6f}")
                            if not stable:
                                remove_edge(x, y)
                                append_value(cg.sepset, x, y, S)
                                append_value(cg.sepset, y, x, S)
                                edge_removed = True
                                break
                            else:
                                edge_removal.append((x, y))
                                for s in S:
                                    sepsets.add(s)
                        else:
                            if verbose:
                                print(f"{x} dep {y} | {S} with p-value {p:.6f}")
                    append_value(cg.sepset, x, y, tuple(sepsets))
                    append_value(cg.sepset, y, x, tuple(sepsets))
                    if edge_removed:
                        continue

                    neigh_y = cg.neighbors(y)
                    neigh_y_nox = np.delete(neigh_y, np.where(neigh_y == x))
                    neigh_y_nox = _sort_neighbors(neigh_y_nox, y)
                    for comb_idx, S in enumerate(combinations(neigh_y_nox, depth)):
                        if (
                            max_combinations is not None
                            and comb_idx >= max_combinations
                        ):
                            break
                        p, stat = ci_test(x, y, S)
                        _update_strength(x, y, stat)
                        if return_pvals:
                            pvals.append((x, y, S, p))

                        if p > alpha:
                            if verbose:
                                print(f"{x} ind {y} | {S} with p-value {p:.6f}")
                            if not stable:
                                if background_knowledge is not None and (
                                    background_knowledge.is_required(
                                        cg.G.nodes[x], cg.G.nodes[y]
                                    )
                                    or background_knowledge.is_required(
                                        cg.G.nodes[y], cg.G.nodes[x]
                                    )
                                ):
                                    pass
                                else:
                                    remove_edge(x, y)
                                    append_value(cg.sepset, x, y, S)
                                    append_value(cg.sepset, y, x, S)
                                    break
                            else:
                                edge_removal.append((x, y))
                                for s in S:
                                    sepsets.add(s)
                        else:
                            if verbose:
                                print(f"{x} dep {y} | {S} with p-value {p:.6f}")
                    append_value(cg.sepset, x, y, tuple(sepsets))
                    append_value(cg.sepset, y, x, tuple(sepsets))

        if show_progress:
            pbar.refresh()

        for x, y in list(set(edge_removal)):
            if background_knowledge is not None and (
                background_knowledge.is_required(cg.G.nodes[x], cg.G.nodes[y])
                or background_knowledge.is_required(cg.G.nodes[y], cg.G.nodes[x])
            ):
                continue
            remove_edge(x, y)

    if show_progress:
        pbar.close()

    if return_pvals:
        return cg, pvals
    return cg


def skeleton_discovery_pervar(
    cg: CausalGraph,
    alpha: float,
    ci_test,
    max_combinations: int = 1,
    background_knowledge: BackgroundKnowledge | None = None,
    verbose: bool = False,
    show_progress: bool = True,
    return_pvals=False,
    max_degree=None,
    num_lags=0,
) -> CausalGraph:
    """Per-variable PC skeleton discovery (CDNOTS+ phase 1).

    Like PCMCI+'s run_pc_stable: for each target variable j, tests each
    candidate parent conditioning only on OTHER parents of j (not arbitrary
    neighbors from the full lag-embedded graph). This avoids over-conditioning
    from irrelevant lag-copies.

    Parents are sorted by test statistic strength after each conditioning
    dimension, so max_combinations=1 picks the strongest conditioning set.

    Parameters
    ----------
    cg : CausalGraph
        Initial fully-connected lag-embedded structure.
    alpha : float
        Significance level.
    ci_test : object
        CI test instance.
    max_combinations : int
        Max conditioning sets per edge per depth (default 1).
    background_knowledge : BackgroundKnowledge, optional
        Domain constraints on forbidden/required edges.
    verbose : bool
        If True, print each CI test result.
    show_progress : bool
        If True, show a tqdm progress bar.
    return_pvals : bool
        If True, also return all CI test results.
    max_degree : int, optional
        Max conditioning dimension.
    num_lags : int
        Number of lags in embedding.

    Returns
    -------
    CausalGraph
        Skeleton graph after per-variable discovery. If ``return_pvals``
        is True, returns ``(cg, pvals)`` instead.
    """
    assert 0 < alpha < 1
    assert ci_test is not None

    no_of_var = cg.G.num_vars
    assert no_of_var % (num_lags + 1) == 0
    no_of_inst_var = no_of_var // (num_lags + 1)

    cg = deepcopy(cg)
    pbar = tqdm(total=no_of_inst_var) if show_progress else None
    pvals = []

    def naive_remove_edge(x, y):
        edge = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
        if edge is not None:
            cg.G.remove_edge(edge)
        edge = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
        if edge is not None:
            cg.G.remove_edge(edge)

    def remove_edge(x, y):
        """Remove edge x->y and all its time-shifted copies."""
        naive_remove_edge(x, y)
        if num_lags > 0:
            # Both x and y can be lagged; find the max time step
            x_step = x // no_of_inst_var
            y_step = y // no_of_inst_var
            max_step = max(x_step, y_step)
            for i in range(1, num_lags - max_step + 1):
                nx = x + i * no_of_inst_var
                ny = y + i * no_of_inst_var
                if nx < no_of_var and ny < no_of_var:
                    naive_remove_edge(nx, ny)

    # Per-variable skeleton: each variable j is processed independently
    # with its own parent list, matching PCMCI+'s run_pc_stable. Only
    # LAGGED parents (tau >= 1) are tested here; contemporaneous edges
    # are handled in Phase 2 (MCI). Edge removals are deferred until ALL
    # variables are processed so that each variable starts from the same
    # initial graph.

    # Cache initial LAGGED neighbors for each variable BEFORE any removals.
    initial_parents = {}
    for j in range(no_of_inst_var):
        initial_parents[j] = [p for p in cg.neighbors(j) if p >= no_of_inst_var]

    # Per-variable results: surviving parents after PC testing
    surviving_parents = {}

    for j in range(no_of_inst_var):
        if show_progress:
            pbar.update()
            pbar.set_description(f"Per-var skeleton, node {j}")

        parents = list(initial_parents[j])
        val_min = {p: np.inf for p in parents}
        max_d = max_degree if max_degree is not None else len(parents)

        for conds_dim in range(max_d + 1):
            if len(parents) - 1 < conds_dim:
                break

            nonsig_parents = []

            for parent in list(parents):
                if background_knowledge is not None and (
                    background_knowledge.is_forbidden(cg.G.nodes[parent], cg.G.nodes[j])
                    and background_knowledge.is_forbidden(
                        cg.G.nodes[j], cg.G.nodes[parent]
                    )
                ):
                    nonsig_parents.append(parent)
                    continue
                if background_knowledge is not None and (
                    background_knowledge.is_required(cg.G.nodes[parent], cg.G.nodes[j])
                    or background_knowledge.is_required(
                        cg.G.nodes[j], cg.G.nodes[parent]
                    )
                ):
                    continue  # always keep required edges — skip CI testing

                other_parents = [p for p in parents if p != parent]

                for comb_idx, S in enumerate(combinations(other_parents, conds_dim)):
                    if comb_idx >= max_combinations:
                        break

                    p_val, stat = ci_test(parent, j, list(S))
                    if return_pvals:
                        pvals.append((parent, j, S, p_val))

                    s = abs(stat) if stat is not None and stat != 0 else 0
                    val_min[parent] = min(val_min[parent], s)

                    if p_val > alpha:
                        if verbose:
                            print(
                                f"  {parent} ind {j} | {list(S)} "
                                f"p={p_val:.6f} [dim={conds_dim}]"
                            )
                        nonsig_parents.append(parent)
                        append_value(cg.sepset, parent, j, S)
                        append_value(cg.sepset, j, parent, S)
                        break
                    else:
                        if verbose:
                            print(
                                f"  {parent} dep {j} | {list(S)} "
                                f"p={p_val:.6f} [dim={conds_dim}]"
                            )

            for parent in nonsig_parents:
                if parent in parents:
                    parents.remove(parent)
                    val_min.pop(parent, None)

            parents.sort(key=lambda p: val_min.get(p, 0), reverse=True)

        surviving_parents[j] = set(parents)

    # Apply all lagged edge removals at once (after all variables processed).
    # Contemporaneous edges are untouched here — handled in Phase 2 (MCI).
    for j in range(no_of_inst_var):
        for parent in initial_parents[j]:
            if parent not in surviving_parents[j]:
                remove_edge(parent, j)

    if show_progress:
        pbar.close()

    if return_pvals:
        return cg, pvals
    return cg


def mci_skeleton(
    cg: CausalGraph,
    ci_test,
    alpha: float,
    lagged_parents: dict,
    num_lags: int = 0,
    include_C: bool = True,
    num_c_cols: int = 1,
    max_conds_py: int | None = None,
    max_conds_px: int | None = None,
    verbose: bool = False,
    show_progress: bool = True,
    return_pvals: bool = False,
    no_of_var_ext: int | None = None,
    background_knowledge=None,
) -> CausalGraph:
    """MCI skeleton pruning (phase 2 of CDNOTS+).

    Like PCMCI+ phase 2: for each edge (parent -> j), tests conditioning
    on subsets of OTHER parents of j (from phase 1), iterating through
    all conditioning set sizes with max_combinations=inf. Lagged parents
    of the tested parent are always included as base conditions.

    This is a full per-variable PC on the restricted graph from phase 1.

    Parameters
    ----------
    cg : CausalGraph
        CausalGraph from phase 1 skeleton discovery.
    ci_test : object
        Conditional independence test.
    alpha : float
        Significance threshold.
    lagged_parents : dict
        Mapping ``{j: [list of neighbor indices]}`` from phase 1.
    num_lags : int
        Number of lags in embedding.
    include_C : bool
        Whether C node is present.
    num_c_cols : int
        Number of C columns.
    max_conds_py : int, optional
        Max parents of Y to include in base conditions.
    max_conds_px : int, optional
        Max parents of X to include in base conditions.
    verbose : bool
        If True, print each CI test result.
    show_progress : bool
        If True, show a tqdm progress bar.
    return_pvals : bool
        If True, also return all CI test results.
    no_of_var_ext : int, optional
        Upper bound for shifted-parent column indices. When the CI
        test data was extended by one extra lag (cdnotsplus_discovery does
        this), shifted parents beyond num_lags become accessible, matching
        PCMCI+'s ability to condition on variables beyond tau_max.

    Returns
    -------
    CausalGraph
        Pruned skeleton graph. If ``return_pvals`` is True, returns
        ``(cg, pvals)`` instead.
    """
    no_of_var = cg.G.num_vars
    assert no_of_var % (num_lags + 1) == 0
    no_of_inst_var = no_of_var // (num_lags + 1)
    if no_of_var_ext is None:
        no_of_var_ext = no_of_var

    # c_start: first C column index within a lag slice; nodes >= c_start are C nodes
    c_start = no_of_inst_var - num_c_cols if include_C else no_of_inst_var

    cg = deepcopy(cg)
    pbar = tqdm(total=no_of_inst_var) if show_progress else None
    pvals = []

    def naive_remove_edge(x, y):
        edge = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
        if edge is not None:
            cg.G.remove_edge(edge)
        edge = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
        if edge is not None:
            cg.G.remove_edge(edge)

    def remove_edge(x, y):
        naive_remove_edge(x, y)
        if num_lags > 0:
            x_step = x // no_of_inst_var
            y_step = y // no_of_inst_var
            max_step = max(x_step, y_step)
            for i in range(1, num_lags - max_step + 1):
                nx = x + i * no_of_inst_var
                ny = y + i * no_of_inst_var
                if nx < no_of_var and ny < no_of_var:
                    naive_remove_edge(nx, ny)

    def _is_c_node(idx):
        """Return True if node index idx is a C column (in any lag slice)."""
        return idx % no_of_inst_var >= c_start

    # Phase 2: per-DEPTH (breadth-first) skeleton — matches PCMCI+'s approach.
    # At each conditioning depth, all (parent→j) pairs across ALL variables are
    # tested before any edges are removed. This ensures every test at depth k
    # sees the same graph state, avoiding the per-variable bias where early
    # variables deplete the conditioning pool for later ones.

    # --- Initialise per-variable state ---
    all_parents = {}  # {j: [current parent indices]}
    all_contemp = {}  # {j: [current contemp parent indices]}
    all_val_min = {}  # {j: {p: min |stat|}}
    all_base_j = {}  # {j: base lagged conditions for Y}

    for j in range(no_of_inst_var):
        if _is_c_node(j):
            continue
        p_list = [p for p in cg.neighbors(j) if not _is_c_node(p)]
        all_parents[j] = p_list
        all_val_min[j] = {p: np.inf for p in p_list}
        all_contemp[j] = [p for p in p_list if p < no_of_inst_var]
        base_j = [p for p in lagged_parents.get(j, []) if not _is_c_node(p)]
        if max_conds_py is not None:
            base_j = base_j[:max_conds_py]
        # Include C-node parents discovered in phase 1 in the base conditioning
        # set. Without this, MCI never conditions on C, so C-mediated spurious
        # paths (X ← C → Y) are never blocked in phase 2.
        c_parents_j = [p for p in cg.neighbors(j) if _is_c_node(p)]
        all_base_j[j] = base_j + c_parents_j

    # --- Per-depth loop ---
    conds_dim = 0
    while True:
        if show_progress:
            pbar.set_description(f"MCI phase 2, depth {conds_dim}")

        removals = []  # (parent, j, sepset) — collected globally before any removal
        tested_any = False

        for j in range(no_of_inst_var):
            if _is_c_node(j):
                continue

            parents = all_parents[j]
            contemp = all_contemp[j]
            val_min = all_val_min[j]
            base_j = all_base_j[j]

            for parent in list(parents):
                parent_inst = parent % no_of_inst_var
                parent_lag = parent // no_of_inst_var

                # Use pre-sorted order of `contemp` (sorted at end of previous
                # depth) — do NOT re-sort mid-depth (same fix as phase 1).
                other_contemp = [p for p in contemp if p != parent]

                if conds_dim > len(other_contemp):
                    continue  # not enough contemporaneous conditions for this parent

                tested_any = True

                base_for_this = [p for p in base_j if p % no_of_inst_var != parent_inst]

                base_conds_x = []
                for p in lagged_parents.get(parent_inst, []):
                    p_inst = p % no_of_inst_var
                    p_lag = p // no_of_inst_var
                    shifted_lag = p_lag + parent_lag
                    if p_inst != j and not _is_c_node(p_inst):
                        shifted_idx = p_inst + shifted_lag * no_of_inst_var
                        if shifted_idx < no_of_var_ext:
                            base_conds_x.append(shifted_idx)
                if max_conds_px is not None:
                    base_conds_x = base_conds_x[:max_conds_px]

                for S in combinations(other_contemp, conds_dim):
                    Z = list(S)
                    Z += [p for p in base_for_this if p not in Z]
                    Z += [p for p in base_conds_x if p not in Z]

                    p_val, stat = ci_test(parent, j, Z)
                    if return_pvals:
                        pvals.append((parent, j, tuple(Z), p_val))

                    s = abs(stat) if stat is not None and stat != 0 else 0
                    val_min[parent] = min(val_min[parent], s)

                    if p_val > alpha:
                        if verbose:
                            print(
                                f"  MCI: {parent} ind {j} | {list(Z)} "
                                f"p={p_val:.6f} [dim={conds_dim}]"
                            )
                        removals.append((parent, j, tuple(Z)))
                        append_value(cg.sepset, parent, j, tuple(Z))
                        append_value(cg.sepset, j, parent, tuple(Z))
                        break
                    else:
                        if verbose:
                            print(
                                f"  MCI: {parent} dep {j} | {list(Z)} "
                                f"p={p_val:.6f} [dim={conds_dim}]"
                            )

        # --- Apply all removals at once (stable / breadth-first) ---
        for parent, j, _ in removals:
            if background_knowledge is not None and (
                background_knowledge.is_required(cg.G.nodes[parent], cg.G.nodes[j])
                or background_knowledge.is_required(cg.G.nodes[j], cg.G.nodes[parent])
            ):
                continue
            if parent in all_parents.get(j, []):
                all_parents[j].remove(parent)
                all_val_min[j].pop(parent, None)
                if parent in all_contemp.get(j, []):
                    all_contemp[j].remove(parent)
            # Contemporaneous edges are undirected: also remove j from parent's lists
            if parent < no_of_inst_var:
                if j in all_parents.get(parent, []):
                    all_parents[parent].remove(j)
                    all_val_min[parent].pop(j, None)
                    if j in all_contemp.get(parent, []):
                        all_contemp[parent].remove(j)
            remove_edge(parent, j)

        # Re-sort all variables by updated strength
        for j in range(no_of_inst_var):
            if _is_c_node(j):
                continue
            vm = all_val_min[j]
            all_parents[j].sort(key=lambda p: vm.get(p, 0), reverse=True)
            all_contemp[j].sort(key=lambda p: vm.get(p, 0), reverse=True)

        if not tested_any:
            break
        conds_dim += 1

    if show_progress:
        pbar.close()

    if return_pvals:
        return cg, pvals
    return cg


def lag_verification_pass(
    cg: CausalGraph,
    ci_test,
    alpha: float,
    num_lags: int,
    include_C: bool = True,
    num_c_cols: int = 1,
    verbose: bool = False,
    alpha_lv: float | None = None,
) -> tuple:
    """Multi-lag redundancy test: remove spurious lags between the same variable pair.

    Only activates for variable pairs that have **2 or more surviving lagged
    skeleton edges** (multi-lag pairs). For each such pair and each lag τ, tests
    whether that lag is **rendered redundant** by the other lags of the same
    variable in the pair:

        X_j(t-τ) ⊥? X_i(t) | { X_j at other lags of this pair, other_parents }

    Removal (spurious): p > alpha_lv — lag τ is independent of X_i(t) once
    we condition on the other lags of X_j. It is an autocorrelation artifact
    (e.g. X_j(t-3) → X_j(t-2) → X_i(t) makes lag-3 look causal when lag-2 is
    the true cause; conditioning on X_j(t-2) renders X_j(t-3) redundant).

    Keep (genuine): p ≤ alpha_lv — lag τ retains a direct association with
    X_i(t) even after accounting for all other lags of X_j in the pair.

    This avoids the collider-bias problem of CEDAR's Condition 2 (which conditions on
    X_i(t-1), a descendant of X_j(t-τ-1) for genuine lag-τ edges, opening
    spurious paths). Here we condition only on other lags of the **cause**
    variable, not the effect variable's history.

    Single-lag pairs are left untouched.

    Parameters
    ----------
    cg : CausalGraph
        Skeleton CausalGraph (undirected). Modified on a deep copy.
    ci_test : object
        CI test instance with ``ci_test.data`` already set.
    alpha : float
        Main significance level (fallback when alpha_lv is None).
    num_lags : int
        Number of lags in the causal graph embedding.
    include_C : bool
        Whether the C (time-index) node is present.
    num_c_cols : int
        Number of C columns.
    verbose : bool
        Print each removal decision.
    alpha_lv : float, optional
        Significance level for the redundancy test. A lag is removed
        when p > alpha_lv (independent = redundant). Defaults to ``alpha``.

    Returns
    -------
    cg : CausalGraph
        Updated skeleton with spurious lags removed.
    stats : dict
        Dictionary with keys ``n_multilag_pairs``, ``n_edges_tested``,
        ``n_edges_removed``, ``n_ci_tests``, ``runtime_s``.
    """
    from collections import defaultdict

    if alpha_lv is None:
        alpha_lv = alpha / 5

    t0 = time.perf_counter()
    cg = deepcopy(cg)

    no_of_var = cg.G.num_vars
    no_of_inst_var = no_of_var // (num_lags + 1)
    # c_start: first C column index within a lag slice
    c_start = (no_of_inst_var - num_c_cols) if include_C else no_of_inst_var

    def _is_c_node(idx):
        return idx % no_of_inst_var >= c_start

    n_edges_tested = 0
    n_edges_removed = 0
    n_ci_tests = 0
    edges_to_remove = []

    def naive_remove(x, y):
        e = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
        if e is not None:
            cg.G.remove_edge(e)
        e = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
        if e is not None:
            cg.G.remove_edge(e)

    def remove_edge(x, y):
        naive_remove(x, y)
        if num_lags > 0:
            max_step = max(x // no_of_inst_var, y // no_of_inst_var)
            for k in range(1, num_lags - max_step + 1):
                nx, ny = x + k * no_of_inst_var, y + k * no_of_inst_var
                if nx < no_of_var and ny < no_of_var:
                    naive_remove(nx, ny)

    # Step 1: find variable pairs (i, j) with 2+ lagged skeleton edges.
    pair_to_lags = defaultdict(list)  # (x, j) -> [(tau, y_node_idx), ...]
    for x in range(no_of_inst_var):
        if _is_c_node(x):
            continue
        for y in list(cg.neighbors(x)):
            if y < no_of_inst_var or _is_c_node(y):
                continue
            j = y % no_of_inst_var
            tau = y // no_of_inst_var
            pair_to_lags[(x, j)].append((tau, y))

    multi_lag_pairs = {k: v for k, v in pair_to_lags.items() if len(v) >= 2}
    n_multilag_pairs = len(multi_lag_pairs)

    # Step 2: for each lag in each multi-lag pair, test whether it is redundant
    # given the other lags of the same variable.
    for (x, j), lag_list in multi_lag_pairs.items():
        y_nodes = [y for _, y in lag_list]  # all lag-node indices for X_j → X_i

        # Other lagged parents of X_i(t) from the skeleton (excluding X_j lags):
        # condition on these to block alternative causal paths X_j(t-τ)→X_k→X_i.
        other_parents = [
            n
            for n in cg.neighbors(x)
            if n >= no_of_inst_var and not _is_c_node(n) and n not in y_nodes
        ]

        for tau, y in lag_list:
            # Conditioning set: the OTHER lags of X_j in this pair + other parents.
            other_y = [n for n in y_nodes if n != y]
            cond_set = other_y + other_parents

            n_edges_tested += 1
            n_ci_tests += 1
            p, _ = ci_test(y, x, cond_set)

            if p > alpha_lv:
                # X_j(t-τ) becomes independent of X_i(t) given the other lags
                # of X_j → this lag is redundant (spurious autocorrelation artifact).
                edges_to_remove.append((x, y))
                n_edges_removed += 1
                if verbose:
                    print(
                        f"MultiLagVerify: remove X[{j}](t-{tau}) — X[{x}](t)"
                        f"  [p={p:.4f} > {alpha_lv:.4f}, redundant given"
                        f" other lags {[n // no_of_inst_var for n in other_y]}]"
                    )

    for x, y in edges_to_remove:
        if (
            cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y]) is not None
            or cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x]) is not None
        ):
            remove_edge(x, y)

    stats = {
        "n_multilag_pairs": n_multilag_pairs,
        "n_edges_tested": n_edges_tested,
        "n_edges_removed": n_edges_removed,
        "n_ci_tests": n_ci_tests,
        "runtime_s": time.perf_counter() - t0,
    }
    return cg, stats


def c_lag_cnst(cgg, num_lags=0, include_C=True, num_c_cols=1):
    """Adjust the causal graph by orienting edges based on lagged variables.

    Parameters
    ----------
    cgg : CausalGraph
        The causal graph object to modify.
    num_lags : int
        Number of lagged variables to consider.
    include_C : bool
        Whether the time index variable C is included in the data.
    num_c_cols : int
        Number of C columns (nonstationarity indicators). Default 1.

    Returns
    -------
    CausalGraph
        Modified causal graph with directed edges based on lag constraints.
    """
    no_of_var = cgg.G.num_vars
    assert no_of_var % (num_lags + 1) == 0
    num_of_inst_vars = no_of_var // (num_lags + 1)
    cg = deepcopy(cgg)

    if include_C:
        for time_index in range(num_lags + 1):
            for c_col in range(num_c_cols):
                c_indx_id = (time_index + 1) * num_of_inst_vars - num_c_cols + c_col
                # orient the direction from c_indx to X
                for i in cg.G.get_adjacent_nodes(cg.G.nodes[c_indx_id]):
                    cg.G.add_directed_edge(cg.G.nodes[c_indx_id], i)

    if num_lags > 0:
        for t in range(num_lags):
            for i in range(
                num_of_inst_vars * t,
                num_of_inst_vars * (t + 1),
            ):
                for j in range(
                    num_of_inst_vars * (t + 1), num_of_inst_vars * (num_lags + 1)
                ):
                    if cg.G.is_adjacent_to(
                        cg.G.nodes[i],
                        cg.G.nodes[j],
                    ):
                        cg.G.add_directed_edge(
                            cg.G.nodes[j],
                            cg.G.nodes[i],
                        )
    return cg
