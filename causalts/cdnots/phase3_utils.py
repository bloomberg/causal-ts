"""
This file has been modified by Mohammad Fesanghary on July 28 2025.
Original Package: causal-learn 1.9.0
Original License: MIT License
"""

# SPDX-License-Identifier: MIT

import warnings
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ..ci_tests.kci_gpu import apply_H, rbf_kernel_gpu
from .ancestral import count_c_cols
from .meek import completes_forbidden_path, meek, reaches
from .result import CdnotsResult
from .skeleton_discovery import (
    c_lag_cnst,
    initialize_graph,
    lag_verification_pass,
    mci_skeleton,
    skeleton_cnst,
    skeleton_discovery,
    skeleton_discovery_pervar,
)
from .uc_sepset import uc_sepset, validate_majority_gamma, validate_priority

_C_PRESETS = {
    "linear": lambda T: np.arange(T, dtype=float).reshape(-1, 1),
    "linear+sin": lambda T: np.column_stack(
        [np.arange(T, dtype=float), np.sin(2 * np.pi * np.arange(T) / T)]
    ),
    "linear+exp": lambda T: np.column_stack(
        [np.arange(T, dtype=float), np.exp(np.arange(T, dtype=float) / T) - 1]
    ),
    "linear+quad": lambda T: np.column_stack(
        [np.arange(T, dtype=float), np.arange(T, dtype=float) ** 2 / T]
    ),
    "step": lambda T: (np.arange(T) >= T // 2).astype(float).reshape(-1, 1),
    "step+linear": lambda T: np.column_stack(
        [(np.arange(T) >= T // 2).astype(float), np.arange(T, dtype=float)]
    ),
}

_C_PRESET_LABELS = {
    "linear": ["C_lin"],
    "step": ["C_step"],
    "step+linear": ["C_step", "C_lin"],
    "linear+sin": ["C_lin", "C_sin"],
    "linear+exp": ["C_lin", "C_exp"],
    "linear+quad": ["C_lin", "C_quad"],
}


def make_c_array(T: int, preset: str = "linear") -> np.ndarray:
    """Build a C nonstationarity indicator array for use with run_cdnots.

    Parameters
    ----------
    T : int
        Length of the time series.
    preset : str
        One of ``'linear'``, ``'linear+sin'``, ``'linear+exp'``, ``'linear+quad'``,
        ``'step'``, ``'step+linear'``.

    Returns
    -------
    np.ndarray
        Array of shape ``(T, k)``, where ``k`` depends on the preset.
    """
    if preset not in _C_PRESETS:
        raise ValueError(
            f"Unknown preset '{preset}'. Options: {list(_C_PRESETS.keys())}"
        )
    return _C_PRESETS[preset](T)


def _build_result(
    cg_obj,
    cg_tig,
    pvalue_matrix,
    pvals,
    df_orig,
    var_names_orig,
    lags,
    lag_list,
    include_C,
    lv_stats,
    num_lags,
    alpha,
    priority,
    stable,
    num_c_cols=1,
    c_node_names=None,
    undirected_policy="bidirected",
):
    """Package discovery outputs into a CdnotsResult, replacing method-patching."""
    return CdnotsResult(
        graph=cg_obj,
        cg_tig=cg_tig,
        pvalue_matrix=pvalue_matrix,
        pvals=pvals if pvals is not None else [],
        var_names=var_names_orig,
        num_lags=num_lags,
        lag_list=lags[1:] if lag_list is not None else None,
        include_C=include_C,
        alpha=alpha,
        priority=priority,
        stable=stable,
        lag_verify_stats=lv_stats,
        df=df_orig,
        num_c_cols=num_c_cols,
        c_node_names=c_node_names,
        undirected_policy=undirected_policy,
    )


def infer_nonsta_dir(X, Y, c_indx, width="empirical"):
    """Estimate the test statistic to decide the orientation (equation 16 in cd-nod paper).

    Parameters
    ----------
    X : array-like
        1D or 2D array or DataFrame.
    Y : array-like
        1D or 2D array or DataFrame.
    c_indx : array-like
        Time index.
    width : str
        Kernel width.

    Returns
    -------
    float
        Test statistic.
    """

    T, d = X.shape

    tensor_X = torch.tensor(X.values).to(torch.float)
    tensor_Y = torch.tensor(Y.values).to(torch.float)
    c_indx = c_indx.tail(X.shape[0])
    tensor_C = torch.tensor(c_indx.values).to(torch.float)

    if torch.cuda.is_available():
        tensor_X = tensor_X.cuda()
        tensor_Y = tensor_Y.cuda()
        tensor_C = tensor_C.cuda()

    # normalize
    if X.ndim != 1:
        tensor_X = tensor_X - torch.mean(tensor_X, 0)
        std_x = torch.std(tensor_X, 0)
        tensor_X = tensor_X @ torch.diag(1 / std_x)
    else:
        tensor_X = tensor_X - torch.mean(tensor_X, 0)
        std_x = torch.std(tensor_X, 0)
        tensor_X = tensor_X / std_x
        tensor_X = tensor_X.reshape(-1, 1)

    if Y.ndim != 1:
        tensor_Y = tensor_Y - torch.mean(tensor_Y, 0)
        std_y = torch.std(tensor_Y, 0)
        tensor_Y = tensor_Y @ torch.diag(1 / std_y)
    else:
        tensor_Y = tensor_Y - torch.mean(tensor_Y, 0)
        std_y = torch.std(tensor_Y, 0)
        tensor_Y = tensor_Y / std_y
        tensor_Y = tensor_Y.reshape(-1, 1)

    lambd = 2  # may need training
    Ml = []
    kyy = rbf_kernel_gpu(tensor_Y, width)

    # P(Y|X)
    kxx = rbf_kernel_gpu(tensor_X, width)
    ktt = rbf_kernel_gpu(tensor_C, 1).float()

    invK = torch.linalg.pinv(
        kxx * ktt + lambd * torch.eye(T, device=tensor_X.device), hermitian=True
    )  # symmetric matrix
    kxx3 = kxx**3
    prod_invK = invK @ kyy @ invK
    Ml = (1 / T**2) * ktt * (kxx3 * prod_invK) * ktt  # linear kernel
    D = (
        (torch.diag(torch.diag(Ml)) @ torch.ones(Ml.shape, device=tensor_X.device))
        + (torch.ones(Ml.shape, device=tensor_X.device) @ torch.diag(torch.diag(Ml)))
        - 2 * Ml
    )  # square distance
    dists = torch.tril(D, -1).reshape(-1)
    sigma2_square = torch.median(dists[dists > 0])  # Gaussian kernel
    Mg = torch.exp(-D / sigma2_square / 2)

    # P(X)
    invK2 = torch.linalg.pinv(
        ktt + lambd * torch.eye(T, device=tensor_X.device), hermitian=True
    )
    Ml2 = ktt @ invK2 @ kxx @ invK2 @ ktt  # linear kernel
    D2 = (
        (torch.diag(torch.diag(Ml2)) @ torch.ones(Ml2.shape, device=tensor_X.device))
        + (torch.ones(Ml2.shape, device=tensor_X.device) @ torch.diag(torch.diag(Ml2)))
        - 2 * Ml2
    )
    dists = torch.tril(D2, -1).reshape(-1)
    sigma2_square2 = torch.median(dists[dists > 0])
    Mg2 = torch.exp(-D2 / sigma2_square2 / 2)

    Mg = apply_H(Mg)
    Mg2 = apply_H(Mg2)
    summ = (Mg.T * Mg2).sum().cpu().item()
    testStat = (1 / T**2) * summ

    return testStat


def _orient(graph, x, y):
    """Write x -> y into a causal-learn endpoint matrix."""
    graph[x, y] = -1
    graph[y, x] = 1


def _can_orient(cg, x, y, background_knowledge, forbidden_ancestor_pairs):
    """Whether phase three may orient the undirected edge x - y as x -> y.

    Not if y already reaches x (a directed cycle), background knowledge
    forbids x -> y or requires y -> x, or a forbidden ancestor pair (a, d)
    would get a directed path a ->...-> x -> y ->...-> d.
    """
    if not cg.is_undirected(x, y):
        return False  # already oriented (or bidirected): leave it alone
    graph = cg.G.graph
    if reaches(graph, y, x):
        return False
    if background_knowledge is not None:
        nodes = cg.G.nodes
        if background_knowledge.is_forbidden(
            nodes[x], nodes[y]
        ) or background_knowledge.is_required(nodes[y], nodes[x]):
            return False
    return not completes_forbidden_path(graph, x, y, forbidden_ancestor_pairs)


def phase_three(
    g,
    data,
    c_indx=None,
    num_lags=0,
    width="empirical",
    include_C=True,
    num_c_cols=1,
    orient_margin=0.0,
    background_knowledge=None,
    forbidden_ancestor_pairs=None,
):
    """Estimate the adjacency matrix of phase 3, contemporaneous dependency.

    Parameters
    ----------
    g : CausalGraph
        Graph obtained from phase 2.
    data : array-like
        Time series dataset (with C).
    c_indx : array-like, optional
        Time index for nonstationarity. If None, extracted from data.
    num_lags : int
        Number of max lags.
    width : str
        Kernel width.
    include_C : bool
        Whether time index variable C is included.
    num_c_cols : int
        Number of C columns (nonstationarity indicators). Default 1.
    orient_margin : float
        Confidence gate on the sink search, in [0, 1). The loop below picks
        the next sink as ``argmin`` over candidate scores and commits *all*
        of that node's undirected neighbours to point into it. With the
        default ``0.0`` it always commits, however close the scores are, so
        no contemporaneous edge is ever left unresolved. When positive, the
        search stops as soon as the best score is not separated from the
        runner-up by at least this *relative* margin, leaving the remaining
        contemporaneous edges undirected -- the analogue of PCMCI+ abstaining
        on an ambiguous link. Callers that convert with
        ``undirected="drop"`` then drop them.
    background_knowledge, forbidden_ancestor_pairs : optional
        The constraints the earlier Meek call used. Phase three never orients
        an edge against them, nor one that would close a directed cycle, and
        passes them to its final Meek call.

    Returns
    -------
    CausalGraph
        Adjacency matrix after phase 3.
    """

    def _meek(cg):
        return meek(
            cg=cg,
            background_knowledge=background_knowledge,
            num_lags=num_lags,
            forbidden_ancestor_pairs=forbidden_ancestor_pairs,
        )

    if not include_C:
        # No C variable — skip nonstationarity-based orientation, just apply Meek
        return _meek(deepcopy(g))

    no_of_var = g.G.num_vars
    assert no_of_var % (num_lags + 1) == 0
    no_of_inst_var = no_of_var // (num_lags + 1)

    # c_start: first C column index within a lag slice (lag-0)
    c_start = no_of_inst_var - num_c_cols

    if c_indx is None:
        c_indx = pd.DataFrame(data[:, c_start:no_of_inst_var])

    n = no_of_inst_var  # noqa: F841
    # Find non-C nodes that have an undirected edge with ANY C column at lag 0
    Vns_set = set()
    for c_idx in range(c_start, no_of_inst_var):
        new = np.where(
            (g.G.graph[c_idx, :c_start] == -1) & (g.G.graph[:c_start, c_idx] == 1)
        )[0].tolist()
        Vns_set.update(new)
    Vns = list(Vns_set)

    Vns_un = []  # nodes with nonstationary causal modules and undirected edges

    for i in range(len(Vns)):
        if (
            len(
                np.where(
                    (g.G.graph[Vns[i], :c_start] == -1)
                    & (g.G.graph[:c_start, Vns[i]] == -1)
                )[0].tolist()
            )
            != 0
        ):  # nonstationary undirected nodes
            Vns_un.append(Vns[i])
    Vns = Vns_un
    gns = deepcopy(g)
    conflicts = getattr(g, "conflict_edges", set())
    forb_pairs = forbidden_ancestor_pairs or ()

    def _orient_copies(c, sink):
        # direction stability over diagonal; _can_orient also requires the
        # shifted edge to exist
        for ll in range(1, num_lags + 1):
            cx, cy = c + ll * no_of_inst_var, sink + ll * no_of_inst_var
            if _can_orient(gns, cx, cy, background_knowledge, forb_pairs):
                _orient(gns.G.graph, cx, cy)

    data = pd.DataFrame(data)
    while len(Vns) > 1:
        score = []
        hypo_eff = []
        hypo_cau = []

        for i in range(len(Vns)):
            hypo_eff.append(Vns[i])

            idx1 = np.where(
                (gns.G.graph[Vns[i], :c_start] == -1)
                & (gns.G.graph[:c_start, Vns[i]] == -1)
            )[
                0
            ].tolist()  # undirected (lag-0 only, non-C)
            idx2 = np.where(
                (gns.G.graph[:c_start, Vns[i]] == -1)
                & (gns.G.graph[Vns[i], :c_start] == 1)
            )[
                0
            ].tolist()  # directed (lag-0 only, non-C)
            hypo_cau.append(sorted(idx1 + idx2))

            sc = infer_nonsta_dir(
                data.iloc[:, hypo_cau[i]], data.iloc[:, hypo_eff[i]], c_indx, width
            )
            score.append(sc)

        idd = int(np.argmin(score))

        if orient_margin > 0.0:
            _s = np.asarray(score, dtype=float)
            if not np.isfinite(_s[idd]):
                # Degenerate score: refuse to guess a sink.
                break
            if _s.size > 1:
                _finite = np.sort(_s[np.isfinite(_s)])
                if _finite.size < 2:
                    break
                best, runner_up = _finite[0], _finite[1]
                scale = max(abs(best), abs(runner_up), 1e-300)
                if (runner_up - best) / scale < orient_margin:
                    # The next sink is not identifiable at the requested
                    # confidence. Stop rather than commit: continuing would
                    # compound this guess into every later orientation.
                    break

        sink = Vns[idd]
        # Edges Meek found conflicting (R1 implied both directions) stay
        # undirected; they still count as candidate causes in the score.
        # Causes are taken in index order. Without forbidden ancestor pairs
        # the result does not depend on it (each edge points into the sink,
        # so no orientation can enable or block another); with them, as in
        # Meek, the first admissible orientations win.
        causes = [c for c in hypo_cau[idd] if (c, sink) not in conflicts]
        for c in causes:
            if not gns.is_fully_directed(c, sink):
                if not _can_orient(gns, c, sink, background_knowledge, forb_pairs):
                    continue
                _orient(gns.G.graph, c, sink)
            _orient_copies(c, sink)

        Vns = [i for i in Vns if i != sink]

    # The loop stops before the last candidate (or early, at the margin), so
    # those never get their lag copies above. Copy the directions they
    # already have; nothing new is oriented here.
    for v in Vns:
        for c in range(c_start):
            if gns.is_fully_directed(c, v):
                _orient_copies(c, v)

    return _meek(gns)


def build_pvalue_matrix(pvals, num_vars, max_lag, aggregation="min"):
    """Build a ``(num_vars, num_vars, max_lag+1)`` p-value matrix from skeleton pvals.

    Parameters
    ----------
    pvals : list of tuples
        ``(effect_node, cause_node, condition_set, pval)`` as returned by
        ``cdnots_discovery(..., return_pvals=True)``.
        Node indices follow the CDNOTS expanded layout:
        ``node = var + lag * num_vars``.
    num_vars : int
        Number of original (lag-0) variables.
    max_lag : int
        Maximum lag.
    aggregation : str, optional
        How to aggregate multiple p-values for the same ``(i, j, tau)``.
        ``"min"`` (default) keeps the minimum (strongest evidence of
        dependence).  ``"max"`` keeps the maximum (most conservative —
        the conditioning set that best explains away the relationship).

    Returns
    -------
    M : ndarray, shape (num_vars, num_vars, max_lag+1)
        P-value matrix.  ``M[i, j, tau]`` is the aggregated p-value for
        the test of edge *i(t-tau) → j(t)* (same convention as the
        causal graph arrays).  Untested entries are ``NaN``.
    """
    M = np.full((num_vars, num_vars, max_lag + 1), np.nan, dtype=float)
    agg_fn = np.fmax if aggregation == "max" else np.fmin

    # De-duplicate while preserving order
    pvals = list(dict.fromkeys(pvals))

    for eff_node, cause_node, _cond_set, p in pvals:
        p = float(p)
        e_var = int(eff_node) % num_vars
        e_lag = int(eff_node) // num_vars
        c_var = int(cause_node) % num_vars
        c_lag = int(cause_node) // num_vars

        # Only keep effects at lag 0 (current time)
        if e_lag != 0:
            continue
        if not (0 <= c_lag <= max_lag):
            continue

        # Store as M[cause_var, effect_var, lag] to match graph convention:
        # M[i, j, tau] = p-value for edge i(t-tau) → j(t)
        if np.isnan(M[c_var, e_var, c_lag]):
            M[c_var, e_var, c_lag] = p
        else:
            M[c_var, e_var, c_lag] = agg_fn(M[c_var, e_var, c_lag], p)

    return M


def _detect_discrete_cols(df, T):
    """Heuristically identify discrete columns in df before lag-embedding.

    A column is considered discrete if:
      1. All non-NaN values equal their integer-rounded value (integer-valued), AND
      2. Number of unique values < min(20, sqrt(T))  (low cardinality)

    Returns list of column names that appear discrete.
    """
    threshold = min(20, int(T**0.5))
    discrete = []
    for col in df.columns:
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        if not np.allclose(vals, vals.round(0), atol=1e-9):
            continue
        if vals.nunique() <= threshold:
            discrete.append(col)
    return discrete


def _set_discrete_indices(indep_test, embedded_disc):
    """Point an index-based discrete-aware CI test at embedded columns."""
    indep_test.discrete_cols = embedded_disc
    if hasattr(indep_test, "_disc_set"):
        indep_test._disc_set = set(embedded_disc)
    if hasattr(indep_test, "_disc_global"):
        indep_test._disc_global = set(embedded_disc)


def _c_nodes(cg, num_lags, include_C, num_c_cols):
    """Lag-0 indices of the C columns (the last ``num_c_cols`` of a slice)."""
    if not include_C:
        return ()
    n_inst = cg.G.num_vars // (num_lags + 1)
    return tuple(range(n_inst - num_c_cols, n_inst))


def _deep_lags(lags):
    """Slice lags of the deep array that MCI and the collider step read.

    A lagged node is conditioned on its parents shifted by its lag, so the CI
    test needs every sum of two window lags. The window's own lags come
    first, so the deep array's first columns are exactly the window ``data``
    (and, its deepest lag being ``2 * max(lags)``, the same rows dropped).
    Contiguous lags give ``range(2 * num_lags + 1)``.
    """
    sums = {a + b for a in lags for b in lags}
    return list(lags) + sorted(sums - set(lags))


def _deep_array(df, lags):
    """``(deep_lags, data_ext)``: ``df`` embedded at :func:`_deep_lags`."""
    ext_lags = _deep_lags(lags)
    data_ext = (
        pd.concat([df.shift(i) for i in ext_lags], axis=1).iloc[2 * max(lags) :].values
    )
    return ext_lags, data_ext


def _majority_collider_step(
    cg,
    indep_test,
    df,
    data,
    lags,
    num_lags,
    discrete_cols,
    lagged_parents,
    data_ext=None,
    **uc_kwargs,
):
    """Run the CDNOTS+ (PCMCI+-style) majority-rule collider step.

    A lagged endpoint is conditioned on its parents shifted by its lag, so
    the CI test reads the deep embedding of ``df`` (which already holds any C
    columns) from :func:`_deep_array` for this step, as MCI does.
    ``indep_test.data`` and its discrete-column indices are restored
    afterwards, also on error.
    """
    from ..ci_tests.stratified_cit import compute_discrete_indices

    use_ext = num_lags > 0
    swap_disc = use_ext and bool(discrete_cols) and hasattr(indep_test, "discrete_cols")
    old_disc = list(indep_test.discrete_cols) if swap_disc else None
    has_lags = hasattr(indep_test, "set_lag_structure")
    ext_lags = _deep_lags(lags) if use_ext else lags
    try:
        if use_ext:
            if data_ext is None:
                _, data_ext = _deep_array(df, lags)
            indep_test.data = data_ext
            if has_lags:
                # e.g. residual-score tests size their CV embargo from it.
                indep_test.set_lag_structure(n_vars=df.shape[1], lags=ext_lags)
            if swap_disc:
                # df already includes the C columns, hence include_C=False.
                _set_discrete_indices(
                    indep_test,
                    compute_discrete_indices(
                        discrete_cols, df, ext_lags, include_C=False
                    ),
                )
        return uc_sepset(
            cg,
            num_lags=num_lags,
            contemp_collider_rule="majority",
            ci_test=indep_test,
            lagged_parents=lagged_parents,
            no_of_var_ext=data_ext.shape[1] if use_ext else data.shape[1],
            lags=ext_lags,
            majority_mode="mci",
            **uc_kwargs,
        )
    finally:
        indep_test.data = data
        if use_ext and has_lags:
            indep_test.set_lag_structure(n_vars=df.shape[1], lags=lags)
        if swap_disc:
            _set_discrete_indices(indep_test, old_disc)


#: Refinement / re-orientation passes before giving up on a fixed point.
_REFINE_PASSES = 5


def _lag_refine_step(
    cg,
    indep_test,
    df,
    data,
    lags,
    num_lags,
    discrete_cols,
    include_C,
    num_c_cols,
    background_knowledge,
    forbidden_ancestor_pairs,
    knowledge,
    var_names,
    addback_alpha,
    veto_alpha,
    skip=(),
    tested=None,
):
    """Run :func:`~.lag_refine.refine_lagged` on the deep embedding of ``df``.

    ``cg`` (an oriented graph) is modified in place and the changes returned.
    Swaps ``indep_test.data`` (and discrete indices) like
    :func:`_majority_collider_step` and restores them afterwards, also on
    error.
    """
    from ..ci_tests.stratified_cit import compute_discrete_indices
    from .lag_refine import refine_lagged

    swap_disc = bool(discrete_cols) and hasattr(indep_test, "discrete_cols")
    old_disc = list(indep_test.discrete_cols) if swap_disc else None
    has_lags = hasattr(indep_test, "set_lag_structure")
    ext_lags, data_ext = _deep_array(df, lags)
    complete = ~np.isnan(np.asarray(data_ext, dtype=float))
    min_rows = getattr(indep_test, "_min_samples", 30)
    nodes = cg.G.nodes
    bk = background_knowledge
    n_inst = data.shape[1] // (num_lags + 1)

    def enough_rows(cols):
        # cols = [X, Y] + Z, so len(cols) + 1 == |Z| + 3
        return int(complete[:, cols].all(axis=1).sum()) >= max(min_rows, len(cols) + 1)

    groups = []
    if knowledge is not None:
        for ancestor, desc in getattr(knowledge, "_required_ancestors", ()):
            if ancestor in var_names and desc in var_names:
                d = var_names.index(desc)
                a = var_names.index(ancestor)
                group = [(a + s * n_inst, d) for s in range(num_lags + 1)]
                groups.append([(x, y) for x, y in group if x != y])
    try:
        indep_test.data = data_ext
        if has_lags:
            indep_test.set_lag_structure(n_vars=df.shape[1], lags=ext_lags)
        if swap_disc:
            # df already includes the C columns, hence include_C=False.
            _set_discrete_indices(
                indep_test,
                compute_discrete_indices(discrete_cols, df, ext_lags, include_C=False),
            )
        return refine_lagged(
            cg,
            indep_test,
            num_lags,
            c_start=n_inst - num_c_cols if include_C else n_inst,
            alpha_add=addback_alpha,
            alpha_veto=veto_alpha,
            is_forbidden=lambda x, y: bk is not None
            and bk.is_forbidden(nodes[x], nodes[y]),
            is_required=lambda x, y: bk is not None
            and bk.is_required(nodes[x], nodes[y]),
            forbidden_ancestor_pairs=forbidden_ancestor_pairs or (),
            required_ancestor_groups=groups,
            enough_rows=enough_rows,
            skip=skip,
            tested=tested,
        )
    finally:
        indep_test.data = data
        if has_lags:
            indep_test.set_lag_structure(n_vars=df.shape[1], lags=lags)
        if swap_disc:
            _set_discrete_indices(indep_test, old_disc)


def _is_discrete_aware(indep_test):
    """Return True if the CI test natively handles discrete columns."""
    from ..ci_tests.cmiknn_mixed_gpu import CMIknnMixedGPU
    from ..ci_tests.stratified_cit import StratifiedCIT

    return isinstance(indep_test, (StratifiedCIT, CMIknnMixedGPU))


def _resolve_knowledge(
    knowledge, background_knowledge, var_names, num_lags, include_C, num_c_cols
):
    """AncestralKnowledge -> (BackgroundKnowledge, forbidden-ancestor pairs).

    An explicit ``background_knowledge`` wins over ``knowledge``'s edge
    constraints; the pairs are None when there are no forbidden ancestors.
    """
    forb_ancestor_pairs = None
    if knowledge is not None:
        if background_knowledge is None:
            background_knowledge = knowledge.to_background_knowledge(
                var_names, num_lags, include_C, num_c_cols=num_c_cols
            )
        if knowledge._forbidden_ancestors:
            forb_ancestor_pairs = knowledge.resolve_forbidden_ancestor_pairs(
                var_names, num_lags, include_C, num_c_cols=num_c_cols
            )
    return background_knowledge, forb_ancestor_pairs


def make_background_knowledge(
    var_names,
    num_lags,
    include_C=True,
    forbidden=None,
    required=None,
    num_c_cols=None,
):
    """Build a BackgroundKnowledge object for use with cdnots_discovery.

    Translates human-readable edge constraints (variable names + lags) into
    causal-learn's BackgroundKnowledge format.

    Parameters
    ----------
    var_names : list of str
        Variable names matching the DataFrame columns (e.g. ``["X0", "X1", "X2"]``).
    num_lags : int
        Number of lags (must match the ``num_lags`` passed to ``cdnots_discovery``).
    include_C : bool
        Whether the C node is included (must match ``cdnots_discovery``).
    forbidden : list of tuple, optional
        Edges to forbid. Each tuple is ``(cause_name, effect_name, cause_lag, effect_lag)``
        where lags are 0-indexed (0 = contemporaneous). Example::

            [("X0", "X1", 1, 0)]  # forbid X0(t-1) -> X1(t)

    required : list of tuple, optional
        Edges to require. Same format as ``forbidden``.
    num_c_cols : int, optional
        Number of C columns (e.g. 2 for ``c_preset="linear+sin"``); defaults
        to 1 when ``include_C`` is set.

    Returns
    -------
    BackgroundKnowledge
        Object to pass as ``background_knowledge`` to ``cdnots_discovery``.

    Example
    -------
    >>> bk = make_background_knowledge(
    ...     var_names=["Temp", "Pressure", "Wind"],
    ...     num_lags=2,
    ...     forbidden=[("Wind", "Temp", 1, 0)],   # Wind(t-1) cannot cause Temp(t)
    ...     required=[("Temp", "Pressure", 1, 0)], # Temp(t-1) must cause Pressure(t)
    ... )
    >>> cg, graph, pvals = cdnots_discovery(df, ci_test, num_lags=2, background_knowledge=bk)
    """
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

    d = len(var_names)
    n_per_lag = d + count_c_cols(include_C, num_c_cols)

    def _node_pattern(var_name, lag):
        var_idx = var_names.index(var_name)
        idx = var_idx + lag * n_per_lag
        return f"^X{idx}$"

    bk = BackgroundKnowledge()

    if forbidden:
        for cause_name, effect_name, cause_lag, effect_lag in forbidden:
            src = _node_pattern(cause_name, cause_lag)
            dst = _node_pattern(effect_name, effect_lag)
            bk.add_forbidden_by_pattern(src, dst)
            bk.add_forbidden_by_pattern(dst, src)

    if required:
        for cause_name, effect_name, cause_lag, effect_lag in required:
            src = _node_pattern(cause_name, cause_lag)
            dst = _node_pattern(effect_name, effect_lag)
            bk.add_required_by_pattern(src, dst)

    return bk


def cdnots_discovery(
    df,
    indep_test,
    num_lags=1,
    lag_list=None,
    include_C=True,
    c_array=None,
    c_preset=None,
    alpha=0.05,
    stable=True,
    return_pvals=True,
    max_degree=None,
    max_combinations: int | None = 20,
    priority=2,
    verbose=False,
    show_progress=False,
    impute=None,
    impute_kwargs=None,
    discrete_cols=None,
    background_knowledge=None,
    knowledge=None,
    lag_verify=False,
    alpha_lv=None,
    *,
    contemp_collider_rule: str | None = "majority",
    orient_margin=0.0,
    collider_search="full",
    majority_gamma=0.5,
):
    """Discover a CDNOTS causal graph.

    Most arguments control the skeleton and CI testing. One orientation option
    is exposed here:

    contemp_collider_rule : {"majority", None}, default "majority"
        How unshielded colliders whose middle node is contemporaneous (lag-0)
        are oriented:

        - ``"majority"`` — majority rule: re-test the pair over a family of
          conditioning sets (see ``collider_search``) and declare a collider
          only if the middle node appears in fewer than half of the separating
          sets. An exact 50% tie, or no separating set, leaves the triple
          ambiguous and unoriented. More conservative about declaring
          colliders.
        - ``None`` — classic rule: declare a collider iff the middle node is in
          *no* separating set found by the skeleton search.

    collider_search : {"full", "sepset"}, default "full"
        Conditioning sets of the majority rule. ``"full"``: every subset of
        the window neighbours of either endpoint (lagged ones included), at
        most ``max_combinations`` per size per endpoint and none larger than
        ``max_degree``, the skeleton's budget; neighbours in the skeleton's
        separating sets for the pair are tried first. ``"sepset"``: only the
        subsets of the skeleton's separating set plus the middle node; it
        costs about as much as the old contemporaneous-only rule and is far
        cheaper than ``"full"``, but orients fewer edges. Ignored when
        ``contemp_collider_rule`` is None.

    majority_gamma : float, default 0.5
        Ambiguity band of the majority rule, in ``[0, 0.5]``: a collider needs
        the middle node in at most this share of the separating sets, a
        non-collider in at least ``1 - majority_gamma``; shares in between
        leave the triple ambiguous. ``0.5`` is the plain majority rule.
        Ignored when ``contemp_collider_rule`` is None.
    """
    validate_priority(priority)
    validate_majority_gamma(majority_gamma)
    if contemp_collider_rule not in ("majority", None):
        raise ValueError(
            "contemp_collider_rule must be 'majority' or None, "
            f"got {contemp_collider_rule!r}"
        )
    _COLLIDER_SEARCH = {"full": "pc", "sepset": "pc_sepset"}
    if collider_search not in _COLLIDER_SEARCH:
        raise ValueError(
            "collider_search must be 'full' or 'sepset', " f"got {collider_search!r}"
        )
    # discrete_cols=None  → auto-detect from df
    # discrete_cols=[]    → explicit "no discrete columns" (opt-out)
    # discrete_cols=[...] → user-specified column names
    if discrete_cols is None:
        discrete_cols = _detect_discrete_cols(df, df.shape[0])
        if discrete_cols and not _is_discrete_aware(indep_test):
            warnings.warn(
                f"Detected discrete columns: {discrete_cols}. "
                f"Your CI test ({type(indep_test).__name__}) does not natively handle "
                "discrete data — wrapping it in StratifiedCIT for correct conditioning. "
                "Pass discrete_cols=[] to disable auto-wrapping.",
                UserWarning,
                stacklevel=2,
            )

    if discrete_cols and not _is_discrete_aware(indep_test):
        from ..ci_tests.stratified_cit import StratifiedCIT

        indep_test = StratifiedCIT(
            data=np.zeros((2, 2)),
            discrete_cols=[],
            inner_cit=indep_test,
        )

    _IMPUTE_CHOICES = {None, "pairwise_complete", "var_em", "causal_iterative"}
    impute_kwargs = impute_kwargs or {}
    if impute not in _IMPUTE_CHOICES:
        raise ValueError(f"impute must be one of {_IMPUTE_CHOICES}, got {impute!r}")
    if impute in ("var_em", "causal_iterative") and not df.isna().any().any():
        warnings.warn(
            f"impute='{impute}' was set but the DataFrame has no NaN values; "
            "skipping imputation.",
            UserWarning,
            stacklevel=2,
        )
        impute = None
    if impute == "var_em":
        from ..imputation import var_em_impute

        df = var_em_impute(df, **impute_kwargs)
    elif impute == "causal_iterative":
        from ..imputation import iterative_causal_impute

        df = iterative_causal_impute(
            df,
            indep_test,
            num_lags=num_lags,
            lag_list=lag_list,
            include_C=include_C,
            alpha=alpha,
            **impute_kwargs,
        )

    if lag_list is not None:
        lags = [0] + list(lag_list)
        num_lags = len(lag_list)
    else:
        lags = list(range(num_lags + 1))

    # Capture original df and var_names before C is appended
    _df_orig = df.copy()
    _var_names_orig = list(df.columns)

    if include_C:
        T = df.shape[0]
        if c_array is not None:
            C_data = np.asarray(c_array, dtype=float)
            if C_data.ndim == 1:
                C_data = C_data.reshape(-1, 1)
            _c_node_names = None  # custom array — fall back to C1/C2/... in plot
        elif c_preset is not None:
            C_data = make_c_array(T, c_preset)
            _c_node_names = _C_PRESET_LABELS.get(c_preset)
        else:
            C_data = make_c_array(T, "linear")  # default: np.arange(T)
            _c_node_names = _C_PRESET_LABELS["linear"]
        num_c_cols = C_data.shape[1]
        C = pd.DataFrame(C_data)
        df = pd.concat(
            [df.reset_index(drop=True), C],
            axis=1,
        )
    else:
        num_c_cols = 0
        _c_node_names = None

    # Needs the number of C columns, so it comes after C is built.
    background_knowledge, _forb_ancestor_pairs = _resolve_knowledge(
        knowledge,
        background_knowledge,
        _var_names_orig,
        num_lags,
        include_C,
        num_c_cols,
    )

    data_lagged = pd.concat([df.shift(i) for i in lags], axis=1)
    # Drop 2*tau_max rows to match tigramite's sample alignment.
    n_drop = 2 * max(lags) if lags else 0
    data = data_lagged.iloc[n_drop:].values

    indep_test.data = data

    if hasattr(indep_test, "set_lag_structure"):
        indep_test.set_lag_structure(n_vars=df.shape[1], lags=lags)

    # Propagate embedded discrete column indices to the CI test
    if discrete_cols and hasattr(indep_test, "discrete_cols"):
        from ..ci_tests.stratified_cit import compute_discrete_indices

        # df already includes the C columns, hence include_C=False.
        embedded_disc = compute_discrete_indices(
            discrete_cols, df, lags, include_C=False
        )
        _set_discrete_indices(indep_test, embedded_disc)

    cg_i = initialize_graph(data, None)
    cg_i_cnst = skeleton_cnst(
        cg_i, data, num_lags, include_C=include_C, num_c_cols=num_c_cols
    )
    skel_result = skeleton_discovery(
        cg=cg_i_cnst,
        ci_test=indep_test,
        alpha=alpha,
        stable=stable,
        verbose=verbose,
        show_progress=show_progress,
        return_pvals=return_pvals,
        max_degree=max_degree,
        max_combinations=max_combinations,
        num_lags=num_lags,
        background_knowledge=background_knowledge,
    )
    if return_pvals:
        cg_skel, pvals = skel_result
        num_inst_vars = data.shape[1] // (num_lags + 1)
        pvalue_matrix = build_pvalue_matrix(pvals, num_inst_vars, num_lags)
    else:
        cg_skel = skel_result
        pvals = None
        pvalue_matrix = None

    _lv_stats = None
    if lag_verify:
        cg_skel, _lv_stats = lag_verification_pass(
            cg_skel,
            indep_test,
            alpha=alpha,
            num_lags=num_lags,
            include_C=include_C,
            num_c_cols=num_c_cols,
            verbose=verbose,
            alpha_lv=alpha_lv,
            background_knowledge=background_knowledge,
        )

    cg_skel_cnst = c_lag_cnst(
        cg_skel,
        num_lags,
        include_C=include_C,
        num_c_cols=num_c_cols,
        background_knowledge=background_knowledge,
    )
    # The majority rule searches the window neighbours of both endpoints,
    # lagged ones included, as PC on the window does. Restricting it to
    # contemporaneous neighbours (the old behaviour) means no separating set
    # can contain a lagged node, which in autocorrelated data turns true
    # colliders into "non-colliders".
    cg_sepset = uc_sepset(
        cg_skel_cnst,
        priority=priority,
        num_lags=num_lags,
        background_knowledge=background_knowledge,
        contemp_collider_rule=contemp_collider_rule,
        ci_test=indep_test,
        alpha=alpha,
        lagged_parents=None,
        majority_mode=_COLLIDER_SEARCH[collider_search],
        max_subsets_per_size=max_combinations,
        max_subset_size=max_degree,
        majority_gamma=majority_gamma,
        parentless=_c_nodes(cg_skel_cnst, num_lags, include_C, num_c_cols),
    )
    cg_meek = meek(
        cg_sepset,
        background_knowledge=background_knowledge,
        num_lags=num_lags,
        forbidden_ancestor_pairs=_forb_ancestor_pairs,
    )

    # Build c_indx for phase_three from the actual C columns in data (lag-0 slice)
    if include_C:
        no_of_inst_var_ph3 = data.shape[1] // (num_lags + 1)
        c_start_ph3 = no_of_inst_var_ph3 - num_c_cols
        c_indx_for_phase3 = pd.DataFrame(data[:, c_start_ph3:no_of_inst_var_ph3])
    else:
        c_indx_for_phase3 = None

    cg_obj = phase_three(
        cg_meek,
        data,
        c_indx=c_indx_for_phase3,
        num_lags=num_lags,
        width="median",
        include_C=include_C,
        num_c_cols=num_c_cols,
        orient_margin=orient_margin,
        background_knowledge=background_knowledge,
        forbidden_ancestor_pairs=_forb_ancestor_pairs,
    )
    if knowledge is not None and knowledge.has_path_constraints:
        from .ancestral import apply_ancestral_constraints

        cg_obj, _ = apply_ancestral_constraints(
            cg_obj,
            knowledge,
            _var_names_orig,
            num_lags,
            include_C,
            num_c_cols=num_c_cols,
            background_knowledge=background_knowledge,
            df=_df_orig,
            verbose=verbose,
        )

    # CDNOTS keeps unoriented lag-0 edges (o-o -> symmetric 1s). CDNOTS+
    # drops them; see finish_cdnots_plus. Both defaults are deliberate --
    # cdnots_to_tigramite_graph's docstring explains when each is right.
    cg_tig = cdnots_to_tigramite_graph(
        cg_obj,
        num_lags=num_lags,
        include_C=include_C,
        num_c_cols=num_c_cols,
        undirected="bidirected",
    )
    result = _build_result(
        cg_obj,
        cg_tig,
        pvalue_matrix,
        pvals,
        _df_orig,
        _var_names_orig,
        lags,
        lag_list,
        include_C,
        _lv_stats,
        num_lags=num_lags,
        alpha=alpha,
        priority=priority,
        stable=stable,
        num_c_cols=num_c_cols,
        c_node_names=_c_node_names,
        undirected_policy="bidirected",
    )

    if return_pvals:
        return result, cg_tig, pvals
    return result, cg_tig


from ..algorithms import register_algorithm  # noqa: E402


@register_algorithm("cdnots")
def run_cdnots(
    df,
    indep_test,
    num_lags=1,
    lag_list=None,
    include_C=True,
    c_array=None,
    c_preset=None,
    alpha=0.05,
    stable=True,
    max_degree=None,
    max_combinations=20,
    priority=2,
    verbose=False,
    show_progress=False,
    impute=None,
    impute_kwargs=None,
    discrete_cols=None,
    background_knowledge=None,
    knowledge=None,
    lag_verify=False,
    alpha_lv=None,
    return_pvals=False,
    *,
    orient_margin=0.0,
    collider_search="full",
    majority_gamma=0.5,
):
    """CDNOTS causal discovery — returns a single :class:`CdnotsResult` object.

    Constraint-based discovery with temporal constraints and optional
    nonstationarity-based orientation (C node).

    Parameters
    ----------
    df : DataFrame
        Input time series DataFrame (T x d).
    indep_test : CIT_Base
        CI test instance.
    num_lags : int
        Maximum lag for time-embedding.
    lag_list : list of int or None
        Explicit lag list (overrides num_lags).
    include_C : bool
        Include the time-index nonstationarity node C.
    alpha : float
        Significance level for skeleton CI tests.
    stable : bool
        Use stable (batch) skeleton discovery.
    max_degree : int or None
        Maximum conditioning set size.
    max_combinations : int
        Maximum conditioning sets per edge per depth.
    priority : int
        Collider conflict resolution strategy: 0 (overwrite),
        1 (abstain on conflicts) or 2 (keep the first collider).
    verbose : bool
        Print CI test results.
    show_progress : bool
        Show tqdm progress bar.
    impute : str or None
        Imputation strategy (None, 'pairwise_complete', 'var_em',
        'causal_iterative').
    impute_kwargs : dict or None
        Extra kwargs for imputation.
    discrete_cols : list or None
        Discrete column names (auto-detected if None).
    background_knowledge : BackgroundKnowledge or None
        Domain constraints (forbidden/required edges).
    lag_verify : bool
        Run multi-lag redundancy pass after skeleton.
    alpha_lv : float or None
        Significance threshold for the redundancy test
        (defaults to alpha/5).
    return_pvals : bool
        If True, stores all CI test p-values in the result object,
        enabling result.pvalue_matrix and result.plot_pvalues().
    collider_search : {"full", "sepset"}
        Conditioning sets the lag-0 collider vote searches. ``"full"``
        (default): every subset of the endpoints' window neighbours, within
        the skeleton's budget (``max_combinations`` per size per endpoint,
        ``max_degree``); most accurate, but more CI tests. ``"sepset"``: only subsets of the
        skeleton's separating set plus the middle node; about the cost of the
        old contemporaneous-only rule, with fewer edges oriented.
    majority_gamma : float
        Ambiguity band of the lag-0 collider vote, in ``[0, 0.5]`` (default
        0.5, the plain majority rule). A collider needs the middle node in at
        most this share of the separating sets, a non-collider in at least
        ``1 - majority_gamma``; shares in between leave the triple ambiguous.

    Returns
    -------
    CdnotsResult
        Self-contained result with graph, p-values, plotting, and
        effect-estimation methods.

    Notes
    -----
    Unoriented contemporaneous edges (``o-o``) are **kept**, rendered in
    ``cg_tig`` as symmetric 1s. :func:`run_cdnots_plus` **drops** them
    instead. Both defaults are deliberate; neither is the CPDAG itself,
    which is preserved on ``result.graph``. Re-render either way after the
    fact with ``result.to_binary(undirected=...)``, or losslessly with
    ``result.to_marks()``.
    """
    disc_out = cdnots_discovery(
        df=df,
        indep_test=indep_test,
        num_lags=num_lags,
        lag_list=lag_list,
        include_C=include_C,
        c_array=c_array,
        c_preset=c_preset,
        alpha=alpha,
        stable=stable,
        return_pvals=return_pvals,
        max_degree=max_degree,
        max_combinations=max_combinations,
        priority=priority,
        verbose=verbose,
        show_progress=show_progress,
        impute=impute,
        impute_kwargs=impute_kwargs,
        discrete_cols=discrete_cols,
        background_knowledge=background_knowledge,
        knowledge=knowledge,
        lag_verify=lag_verify,
        alpha_lv=alpha_lv,
        orient_margin=orient_margin,
        collider_search=collider_search,
        majority_gamma=majority_gamma,
    )
    return disc_out[0]


@register_algorithm("cdnots+")
def run_cdnots_plus(
    df,
    indep_test,
    num_lags=1,
    lag_list=None,
    include_C=True,
    c_array=None,
    c_preset=None,
    alpha=0.01,
    max_degree=None,
    max_conds_py=None,
    max_conds_px=None,
    priority=1,
    verbose=False,
    show_progress=False,
    impute=None,
    impute_kwargs=None,
    discrete_cols=None,
    background_knowledge=None,
    knowledge=None,
    lag_verify=False,
    alpha_lv=None,
    return_pvals=False,
    *,
    legacy_mci_conds=False,
    orient_margin=0.1,
    return_prefix=False,
    majority_gamma=0.5,
    lag0_orient=None,
    alpha_mci=None,
    addback_alpha=None,
    veto_alpha=None,
):
    """CDNOTS+ causal discovery (PCMCI+-style two-phase skeleton).

    Phase 1: Per-variable PC-stable skeleton (finds parent superset).
    Phase 2: MCI re-test that conditions only on discovered parents,
    avoiding over-conditioning. Same orientation pipeline as
    :func:`run_cdnots`.

    Parameters
    ----------
    df : DataFrame
        Input time series DataFrame (T x d).
    indep_test : CIT_Base
        CI test instance.
    num_lags : int
        Maximum lag for time-embedding.
    lag_list : list of int or None
        Explicit lag list (overrides num_lags).
    include_C : bool
        Include the time-index nonstationarity node C.
    alpha : float
        Significance level for both skeleton phases. Default 0.01, matching
        PCMCI+'s default and the setting empirically better for CDNOTS+ (plain
        CDNOTS keeps 0.05).
    max_degree : int or None
        Maximum conditioning-set size in phase 1.
    max_conds_py : int or None
        Maximum parents of Y in phase 2 MCI conditioning.
    max_conds_px : int or None
        Maximum parents of X in phase 2 MCI conditioning.
    priority : int
        Collider conflict resolution strategy: 0 (overwrite),
        1 (abstain on conflicts) or 2 (keep the first collider).
    verbose : bool
        Print CI test results.
    show_progress : bool
        Show tqdm progress bar.
    impute : str or None
        Imputation strategy (None, 'pairwise_complete', 'var_em',
        'causal_iterative').
    impute_kwargs : dict or None
        Extra kwargs for imputation.
    discrete_cols : list or None
        Discrete column names (auto-detected if None).
    background_knowledge : BackgroundKnowledge or None
        Domain constraints (forbidden/required edges).
    lag_verify : bool
        Run multi-lag redundancy pass after skeleton.
    alpha_lv : float or None
        Significance threshold for the redundancy test.
    return_pvals : bool
        If True, stores all CI test p-values. Set to False for large
        d to save memory.
    legacy_mci_conds : bool
        Restore the pre-alignment phase-2 conditioning rule. The default
        (False) matches PCMCI+'s ``_run_pcalg_test``.
    return_prefix : bool
        Return the :class:`CdnotsPlusPrefix` computed just before Meek
        orientation instead of a result, so several ``orient_margin`` values
        can share one skeleton + collider phase. Feed it to
        :func:`finish_cdnots_plus`. ``orient_margin`` is stored on the prefix
        as the default ``finish_cdnots_plus`` uses.
    orient_margin : float
        Confidence gate on the ``include_C`` nonstationarity orientation
        (see :func:`phase_three`). ``0.0`` commits every contemporaneous edge
        to a direction. Positive values (default ``0.1``) leave ambiguous
        contemporaneous edges undirected, and CDNOTS+ then drops them. No
        effect when ``include_C=False``.
    majority_gamma : float
        Ambiguity band of the majority collider rule, in ``[0, 0.5]``
        (default 0.5, PCMCI+'s rule). A collider needs the middle node in at
        most this share of the separating sets, a non-collider in at least
        ``1 - majority_gamma``; shares in between leave the triple ambiguous.
    lag0_orient : {None, "lingam"}
        Orient lag-0 edges that conditional independence leaves undirected
        (``o-o``) or conflicting (``x-x``). ``"lingam"`` orders the innovations
        of each such component with DirectLiNGAM and orients an edge only when
        the direction holds in at least 90% of bootstrap re-orderings, one
        endpoint's innovations reject normality (``p < 0.01``), and it makes
        no unshielded collider the CI step ruled out; Meek then propagates
        the new arrowheads. On linear Gaussian data it usually abstains (the
        normality test still rejects about 1% of samples). Needs
        non-Gaussian noise to help, and assumes linear mechanisms: on spline
        mechanisms with Laplace or Gaussian noise most of its orientations
        were wrong (small F1 cost, SHD up to +0.75), though with uniform noise
        it still helped; on CausalRivers it left F1 unchanged (SHD +0.5).
        Discrete columns are never oriented. The decisions are stored on
        ``result.lag0_orientations``. Cost: 101 DirectLiNGAM orderings per
        undirected component, each O(k^3 n) for k nodes and n rows (vectorised;
        about 0.1 s at k=30, n=1000). Default ``None`` (off).
    alpha_mci : float or None
        Significance level of the phase 2 MCI test only; ``None`` (default)
        uses ``alpha``. A loose ``alpha`` with a strict ``alpha_mci`` (e.g.
        0.05 / 0.001) picks conditioning sets permissively but keeps links
        strictly, as PCMCI's ``pc_alpha`` / ``alpha_level`` do. With
        ``addback_alpha`` / ``veto_alpha``, ``alpha=0.01`` without
        ``alpha_mci`` lost less F1 than ``0.05 / 0.001`` at T=100 for d >= 30
        and was within 0.01 F1 of it elsewhere (linear simulations, same
        seeds).
    addback_alpha, veto_alpha : float or None
        Refine the lagged cross links of the finished graph (after
        orientation, so oriented lag-0 parents are conditioned on). Add-back
        re-tests every absent lagged link with the MCI conditioning set
        (parents of both endpoints, ``X``'s shifted by its lag, plus each
        endpoint's next lag) and adds it when ``p < addback_alpha``; the
        veto then re-tests every lagged link of the grown graph the same way
        and removes it when ``p > veto_alpha``. This recovers links that
        phase 1 drops at depth 0, where ``Y``'s own past masks them. Lag-0,
        self and C links are not touched. The graph is then re-oriented from
        the refined skeleton, and refinement repeated on the new orientation
        until a pass changes nothing (at most 5 passes). Each pass that
        changes a link re-runs the majority collider step (the dominant cost
        of CDNOTS+): about 1.5-2.7x a plain run in linear simulations
        (d 10-30). A refined prefix cannot be finished at another
        ``orient_margin`` or ``lag0_orient``. Changes are stored on
        ``result.lag_refinements`` and their p-values written into
        ``pvalue_matrix``. Contiguous lags only; not with ``max_conds_py`` /
        ``max_conds_px``. In linear simulations (d 10-50, T 100-1000),
        ``addback_alpha=1e-4, veto_alpha=0.001`` at ``alpha=0.01`` (no
        ``alpha_mci``), against the plain run at the same alpha, lowered SHD
        in every setting and raised F1 by 0.007-0.05, except at T=100 with
        d=50 (F1 -0.03); against the best single plain alpha it raised F1 in
        9 of 12 settings and lowered SHD in 11 of 12 (SHD +1.5 at T=1000,
        d=50). On nonlinear simulations (spline / MLP, d 10-20, T 300-1000)
        with ``raft`` and the same alphas it was about neutral (F1 within
        0.03). It needs a CI test with power at the sample size, since the
        veto drops links the test cannot detect (by default ``splitkci``
        tests on half the rows, so at T=300 it lost up to 0.12 F1; about
        neutral at T=1000), and p-values that
        track effect size: on long real series (CausalRivers, 5-node graphs)
        the add-back added about four false links per true one (F1
        unchanged, SHD +2). Default ``None`` (off).

    Returns
    -------
    CdnotsResult or CdnotsPlusPrefix
        Self-contained result, or the prefix when ``return_prefix=True``.

    Notes
    -----
    Unoriented contemporaneous edges (``o-o``) are **dropped** from
    ``cg_tig``, where :func:`run_cdnots` keeps them as symmetric 1s. The drop
    suits regimes where lag-0 edges are invalid or confounded; where
    contemporaneous structure is real it costs recall. Both defaults are
    deliberate, and neither discards the CPDAG -- it is preserved on
    ``result.graph``. Re-render the other way with
    ``result.to_binary(undirected="bidirected")``, or losslessly with
    ``result.to_marks()``.
    """
    validate_priority(priority)
    validate_majority_gamma(majority_gamma)
    if lag0_orient not in (None, "lingam"):
        raise ValueError(f"lag0_orient must be None or 'lingam', got {lag0_orient!r}")
    for _name, _a in (
        ("alpha_mci", alpha_mci),
        ("addback_alpha", addback_alpha),
        ("veto_alpha", veto_alpha),
    ):
        if _a is not None and not 0 < _a < 1:
            raise ValueError(f"{_name} must be in (0, 1) or None, got {_a!r}")
    _refine = addback_alpha is not None or veto_alpha is not None
    if _refine:
        if not np.isfinite(orient_margin) or orient_margin < 0:
            raise ValueError(
                f"orient_margin must be finite and >= 0, got {orient_margin!r}"
            )
        if max_conds_py is not None or max_conds_px is not None:
            # A refinement-added parent has no rank among phase 1's: first
            # would evict the strongest, last would fall outside the cap.
            raise ValueError(
                "addback_alpha / veto_alpha cannot be combined with "
                "max_conds_py / max_conds_px"
            )
        if lag_list is not None:
            if not len(lag_list) or list(lag_list) != list(range(1, len(lag_list) + 1)):
                raise ValueError(
                    "addback_alpha / veto_alpha need contiguous lags "
                    f"(lag_list 1..L), got {list(lag_list)!r}"
                )
        elif (
            isinstance(num_lags, bool)
            or not isinstance(num_lags, (int, np.integer))
            or num_lags < 1
        ):
            raise ValueError(
                f"addback_alpha / veto_alpha need an integer num_lags >= 1, got "
                f"{num_lags!r}"
            )
    if discrete_cols is None:
        discrete_cols = _detect_discrete_cols(df, df.shape[0])

    _IMPUTE_CHOICES = {None, "pairwise_complete", "var_em", "causal_iterative"}
    impute_kwargs = impute_kwargs or {}
    if impute not in _IMPUTE_CHOICES:
        raise ValueError(f"impute must be one of {_IMPUTE_CHOICES}, got {impute!r}")
    if impute in ("var_em", "causal_iterative") and not df.isna().any().any():
        warnings.warn(
            f"impute='{impute}' was set but the DataFrame has no NaN values; "
            "skipping imputation.",
            UserWarning,
            stacklevel=2,
        )
        impute = None
    if impute == "var_em":
        from ..imputation import var_em_impute

        df = var_em_impute(df, **impute_kwargs)
    elif impute == "causal_iterative":
        from ..imputation import iterative_causal_impute

        df = iterative_causal_impute(
            df,
            indep_test,
            num_lags=num_lags,
            lag_list=lag_list,
            include_C=include_C,
            alpha=alpha,
            **impute_kwargs,
        )

    if lag_list is not None:
        lags = [0] + list(lag_list)
        num_lags = len(lag_list)
    else:
        lags = list(range(num_lags + 1))

    _df_orig = df.copy()
    _var_names_orig = list(df.columns)

    if include_C:
        T = df.shape[0]
        if c_array is not None:
            C_data = np.asarray(c_array, dtype=float)
            if C_data.ndim == 1:
                C_data = C_data.reshape(-1, 1)
            _c_node_names = None
        elif c_preset is not None:
            C_data = make_c_array(T, c_preset)
            _c_node_names = _C_PRESET_LABELS.get(c_preset)
        else:
            C_data = make_c_array(T, "linear")
            _c_node_names = _C_PRESET_LABELS["linear"]
        num_c_cols = C_data.shape[1]
        C = pd.DataFrame(C_data)
        df = pd.concat(
            [df.reset_index(drop=True), C],
            axis=1,
        )
    else:
        num_c_cols = 0
        _c_node_names = None

    # Needs the number of C columns, so it comes after C is built.
    background_knowledge, _forb_ancestor_pairs = _resolve_knowledge(
        knowledge,
        background_knowledge,
        _var_names_orig,
        num_lags,
        include_C,
        num_c_cols,
    )

    data_lagged = pd.concat([df.shift(i) for i in lags], axis=1)
    # Drop 2*tau_max rows to match tigramite's sample alignment: ensures
    # consistent effective T across Phase 1 and Phase 2 (MCI), and matches
    # PCMCI+'s _get_array which uses T - 2*tau_max samples.
    n_drop = 2 * max(lags) if lags else 0
    data = data_lagged.iloc[n_drop:].values

    indep_test.data = data

    if hasattr(indep_test, "set_lag_structure"):
        indep_test.set_lag_structure(n_vars=df.shape[1], lags=lags)

    if discrete_cols and hasattr(indep_test, "discrete_cols"):
        from ..ci_tests.stratified_cit import compute_discrete_indices

        # df already includes the C columns, hence include_C=False.
        embedded_disc = compute_discrete_indices(
            discrete_cols, df, lags, include_C=False
        )
        _set_discrete_indices(indep_test, embedded_disc)

    cg_i = initialize_graph(data, None)
    cg_i_cnst = skeleton_cnst(
        cg_i, data, num_lags, include_C=include_C, num_c_cols=num_c_cols
    )

    # --- Phase 1: Per-variable skeleton (like PCMCI+'s run_pc_stable) ---
    skel_result = skeleton_discovery_pervar(
        cg=cg_i_cnst,
        ci_test=indep_test,
        alpha=alpha,
        max_combinations=1,
        verbose=verbose,
        show_progress=show_progress,
        return_pvals=return_pvals,
        max_degree=max_degree,
        num_lags=num_lags,
        background_knowledge=background_knowledge,
        include_C=include_C,
        num_c_cols=num_c_cols,
    )
    if return_pvals:
        cg_skel, pvals_phase1 = skel_result
    else:
        cg_skel, pvals_phase1 = skel_result, []

    # --- Extract parent sets from phase 1 skeleton ---
    no_of_inst_var = data.shape[1] // (num_lags + 1)
    # Under a cap, strongest first (phase 1's ranking), so max_conds_py /
    # max_conds_px keep the strongest parents as in PCMCI rather than the
    # lowest node indices. Uncapped lists keep node order, so conditioning-set
    # order (and result.pvals) is unchanged.
    capped = max_conds_py is not None or max_conds_px is not None
    order = getattr(cg_skel, "parent_order", {}) if capped else {}
    lagged_parents = {}
    for j in range(no_of_inst_var):
        nbrs = [p for p in cg_skel.neighbors(j) if p >= no_of_inst_var]
        ranked = [p for p in order.get(j, []) if p in nbrs]
        lagged_parents[j] = ranked + [p for p in nbrs if p not in ranked]

    # --- Extend data to the deep lags for phase 2 conditioning ---
    # MCI reads a deeper array; indep_test.data and its discrete indices are
    # restored afterwards, also on error.
    swap_disc = (
        num_lags > 0 and bool(discrete_cols) and hasattr(indep_test, "discrete_cols")
    )
    ext_lags = lags
    try:
        if num_lags > 0:
            ext_lags, data_ext = _deep_array(df, lags)
            indep_test.data = data_ext
            no_of_var_ext = data_ext.shape[1]
            if hasattr(indep_test, "set_lag_structure"):
                indep_test.set_lag_structure(n_vars=df.shape[1], lags=ext_lags)
            if swap_disc:
                from ..ci_tests.stratified_cit import compute_discrete_indices

                # MCI conditions on columns of the deeper array too; their
                # discrete flags must cover it.
                _set_discrete_indices(
                    indep_test,
                    compute_discrete_indices(
                        discrete_cols, df, ext_lags, include_C=False
                    ),
                )
        else:
            no_of_var_ext = data.shape[1]

        # --- Phase 2: MCI re-test ---
        mci_result = mci_skeleton(
            cg=cg_skel,
            ci_test=indep_test,
            alpha=alpha if alpha_mci is None else alpha_mci,
            lagged_parents=lagged_parents,
            num_lags=num_lags,
            include_C=include_C,
            num_c_cols=num_c_cols,
            max_conds_py=max_conds_py,
            max_conds_px=max_conds_px,
            verbose=verbose,
            show_progress=show_progress,
            return_pvals=return_pvals,
            no_of_var_ext=no_of_var_ext,
            background_knowledge=background_knowledge,
            legacy_mci_conds=legacy_mci_conds,
            lags=ext_lags,
        )
    finally:
        indep_test.data = data
        if num_lags > 0 and hasattr(indep_test, "set_lag_structure"):
            indep_test.set_lag_structure(n_vars=df.shape[1], lags=lags)
        if swap_disc:
            _set_discrete_indices(indep_test, embedded_disc)
    if return_pvals:
        cg_skel_mci, pvals_phase2 = mci_result
    else:
        cg_skel_mci, pvals_phase2 = mci_result, []
    pvals = pvals_phase1 + pvals_phase2
    num_inst_vars = data.shape[1] // (num_lags + 1)
    pvalue_matrix = build_pvalue_matrix(pvals, num_inst_vars, num_lags)

    # --- Lag verification pass (optional) ---
    _lv_stats = None
    if lag_verify:
        cg_skel_mci, _lv_stats = lag_verification_pass(
            cg_skel_mci,
            indep_test,
            alpha=alpha,
            num_lags=num_lags,
            include_C=include_C,
            num_c_cols=num_c_cols,
            verbose=verbose,
            alpha_lv=alpha_lv,
            background_knowledge=background_knowledge,
        )

    # --- Orientation pipeline (same as CDNOTS) ---
    cg_skel_cnst = c_lag_cnst(
        cg_skel_mci,
        num_lags,
        include_C=include_C,
        num_c_cols=num_c_cols,
        background_knowledge=background_knowledge,
    )

    def _collide(cg, parents):
        return _majority_collider_step(
            cg,
            indep_test,
            df,
            data,
            lags,
            num_lags,
            discrete_cols,
            parents,
            data_ext=data_ext if num_lags > 0 else None,
            priority=priority,
            background_knowledge=background_knowledge,
            alpha=alpha,
            max_conds_py=max_conds_py,
            max_conds_px=max_conds_px,
            majority_gamma=majority_gamma,
            parentless=_c_nodes(cg, num_lags, include_C, num_c_cols),
        )

    cg_sepset = _collide(cg_skel_cnst, lagged_parents)
    state = CdnotsPlusPrefix(
        cg_sepset=cg_sepset,
        data=data,
        pvals=pvals,
        pvalue_matrix=pvalue_matrix,
        lv_stats=_lv_stats,
        df_orig=_df_orig,
        var_names_orig=_var_names_orig,
        lags=lags,
        lag_list=lag_list,
        num_lags=num_lags,
        include_C=include_C,
        num_c_cols=num_c_cols,
        c_node_names=_c_node_names,
        alpha=alpha,
        priority=priority,
        background_knowledge=background_knowledge,
        forbidden_ancestor_pairs=_forb_ancestor_pairs,
        knowledge=knowledge,
        verbose=verbose,
        orient_margin=orient_margin,
        lag0_orient=lag0_orient,
        lag0_exclude=[
            _var_names_orig.index(c)
            for c in (discrete_cols or [])
            if c in _var_names_orig
        ],
    )
    if _refine:
        # Refine the lagged links on the oriented graph (its lag-0 parents
        # enter the MCI sets), then re-orient from the refined skeleton: an
        # added or removed lagged link can change colliders and Meek, and a
        # changed lag-0 parent can change a refinement test, so repeat until
        # a pass changes nothing.
        from .lag_refine import apply_lag_changes, class_state, restore_class

        n_inst = data.shape[1] // (num_lags + 1)
        cg_ref = cg_skel_cnst
        parents_ref = {j: list(v) for j, v in lagged_parents.items()}
        net = {}  # (cause, effect, lag) -> net change
        first = {}  # (cause, effect, lag) -> action of its first change
        frozen = set()  # links that flipped back: kept at their original state
        original = {}  # (cause, effect, lag) -> its copies before any change
        parent_pos = {}  # (cause, effect, lag) -> its index in parents_ref, or None
        last_p = {}  # (cause, effect, lag) -> p of its latest test
        M0 = np.array(state.pvalue_matrix, dtype=float, copy=True)
        converged = None  # finished result of the pass that changed nothing
        for _ in range(_REFINE_PASSES):
            tested = {}
            finished = finish_cdnots_plus(state)
            changes = _lag_refine_step(
                # modified in place; finished may be returned as converged
                deepcopy(finished.graph),
                indep_test,
                df,
                data,
                lags,
                num_lags,
                discrete_cols,
                include_C,
                num_c_cols,
                background_knowledge,
                _forb_ancestor_pairs,
                knowledge,
                _var_names_orig,
                addback_alpha,
                veto_alpha,
                skip=frozen,
                tested=tested,
            )
            last_p.update(tested)
            cg_new = deepcopy(cg_ref)
            snap = {}
            for ch in changes:
                key = (ch["cause"], ch["effect"], ch["lag"])
                if key not in original:
                    snap[key] = class_state(cg_new, *key, num_lags)
            changes = apply_lag_changes(cg_new, changes, num_lags)
            if not changes:
                converged = finished
                break
            cg_ref = cg_new
            for ch in changes:
                key = (ch["cause"], ch["effect"], ch["lag"])
                if key in snap:
                    original[key] = snap[key]
            for ch in changes:
                key = (ch["cause"], ch["effect"], ch["lag"])
                node = ch["cause"] + ch["lag"] * n_inst
                lst = parents_ref.setdefault(ch["effect"], [])
                if key not in parent_pos:
                    parent_pos[key] = lst.index(node) if node in lst else None
                if node in lst:
                    lst.remove(node)
                if ch["action"] == "added":
                    lst.append(node)
                if key in first and first[key] != ch["action"]:
                    # Back to its original state after an earlier pass changed
                    # it: an oscillation between orientations. Keep the
                    # original and stop testing it.
                    frozen.add(key)
                    net.pop(key, None)
                    restore_class(cg_ref, original[key])
                    # Its conditioning-parent entry too: phase 1 can list a
                    # parent that MCI dropped from the graph.
                    if node in lst:
                        lst.remove(node)
                    if parent_pos[key] is not None:
                        lst.insert(parent_pos[key], node)
                else:
                    first.setdefault(key, ch["action"])
                    net[key] = {k: v for k, v in ch.items() if k != "conds"}
            state.cg_sepset = _collide(cg_ref, parents_ref)
        else:
            warnings.warn(
                f"lagged-link refinement still changed links after "
                f"{_REFINE_PASSES} passes; keeping the last one",
                RuntimeWarning,
                stacklevel=2,
            )
        # Net changes carry the p-value of their latest test (on the final
        # orientation once a pass changes nothing); links back at their
        # original state keep the discovery p-value.
        M = M0.copy()
        for key, ch in net.items():
            ch["p"] = last_p.get(key, ch["p"])
            M[key] = ch["p"]
        state.pvalue_matrix = M
        state.lag_refinements = [
            dict(
                ch,
                cause_name=_var_names_orig[ch["cause"]],
                effect_name=_var_names_orig[ch["effect"]],
            )
            for ch in net.values()
        ]
        state.refined_orient_margin = state.orient_margin
        state.refined_lag0_orient = state.lag0_orient
        if converged is not None and not return_prefix:
            # Same state as finish_cdnots_plus(state) below except these two
            # fields, so skip the second orientation tail.
            converged.pvalue_matrix = state.pvalue_matrix
            converged.lag_refinements = state.lag_refinements
            return converged
    if return_prefix:
        # Caller may evaluate several orientation settings against this one
        # (expensive) collider phase; finish_cdnots_plus(state) with no
        # overrides reproduces the direct call below.
        return state
    return finish_cdnots_plus(state)


#: Rendering policies for an unoriented (``o-o``) contemporaneous edge.
UNDIRECTED_POLICIES = ("bidirected", "drop")
#: Rendering policies for a conflicting (``x-x``) contemporaneous edge.
CONFLICT_POLICIES = ("drop", "bidirected")


def _resolve_undirected(undirected, keep_undirected):
    """Reconcile the ``undirected=`` string with the legacy boolean.

    ``keep_undirected`` is the original spelling and stays a silent alias --
    ``routed_deconf`` and the experiment harnesses pass it, and they are
    behaving correctly, so there is nothing to warn about.
    """
    if undirected is not None and keep_undirected is not None:
        raise ValueError(
            "pass either undirected= or keep_undirected=, not both "
            f"(got undirected={undirected!r}, keep_undirected={keep_undirected!r})"
        )
    if undirected is None:
        if keep_undirected is None:
            return "bidirected"  # the historical default
        return "bidirected" if keep_undirected else "drop"
    if undirected not in UNDIRECTED_POLICIES:
        raise ValueError(
            f"undirected must be one of {UNDIRECTED_POLICIES}, got {undirected!r}. "
            "For the lossless 'o-o' rendering use cdnots_to_tigramite_marks()."
        )
    return undirected


def _prep_for_conversion(gg, num_lags, include_C, num_c_cols):
    """Shared front half of both converters.

    Returns ``(g, no_of_inst_var, orig)`` where ``g`` is a private copy with
    C edges oriented and ``orig`` is a snapshot of the endpoints *before* any
    policy is applied.
    """
    no_of_var = gg.G.num_vars
    assert no_of_var % (num_lags + 1) == 0
    no_of_inst_var = no_of_var // (num_lags + 1)
    g = deepcopy(gg)

    # c_start: first C column index within a lag slice
    c_start = no_of_inst_var - num_c_cols if include_C else no_of_inst_var

    if include_C:
        # Orient remaining C edges as C -> X (for all C columns)
        for c_col in range(num_c_cols):
            c_idx = c_start + c_col
            for i in range(no_of_inst_var):
                if g.G.graph[c_idx, i] == -1:
                    g.G.graph[c_idx, i] = 0
                    g.G.graph[i, c_idx] = 1

    # Decide from a snapshot of the incoming endpoints. The callers visit
    # both (i, j) and (j, i) and write in place, so reading the live array
    # would let one visit observe what the mirrored visit just wrote -- e.g.
    # the o-o branch writes (1, 1) under undirected="bidirected", which the
    # conflict branch would then mistake for an x-x marker and erase.
    return g, no_of_inst_var, g.G.graph.copy()


def _lagged_marks(orig, no_of_inst_var):
    """Marks of the lagged cells ``[lag-0 node a, lagged node b]``.

    ``marks[a, b]`` describes ``b -> a``. A lagged edge is directed by time
    order, causal-learn ``(1, -1)`` here. The rare other states are decoded,
    not passed through: ``(-1, -1)`` is ``o-o``; ``(1, 1)``, and an arrowhead
    on the earlier node (against time order), are ``x-x``. There is no
    ``<--`` at a lag: the mirrored cell is a different lag relationship.
    """
    o = orig[:no_of_inst_var, no_of_inst_var:]
    r = orig[no_of_inst_var:, :no_of_inst_var].T
    return np.select(
        [(o == 1) & (r != 1), (o == 1) | (r == 1), (o == -1) & (r == -1)],
        ["-->", "x-x", "o-o"],
        default="",
    ).astype("<U3")


def _to_tigramite_layout(arr, no_of_inst_var, num_lags):
    """Reshape a ``(d, d*(num_lags+1))`` matrix to ``(d, d, num_lags+1)``.

    The result is indexed ``[cause, effect, lag]``.
    """
    return np.swapaxes(
        np.swapaxes(
            arr[:no_of_inst_var].reshape(no_of_inst_var, num_lags + 1, no_of_inst_var),
            1,
            2,
        ),
        0,
        1,
    )


def cdnots_to_tigramite_graph(
    gg,
    num_lags=0,
    include_C=True,
    num_c_cols=1,
    keep_undirected=None,
    undirected=None,
    conflict="drop",
):
    """Render a causal-learn CPDAG as a binary ``[cause, effect, lag]`` array.

    Discovery produces a CPDAG; this is a *lossy rendering* of it. A binary
    array cannot express "adjacent but unoriented", so how the CPDAG's
    contemporaneous ``o-o`` and ``x-x`` marks are rendered is a policy choice.
    Use :func:`cdnots_to_tigramite_marks` for the lossless view.

    Parameters
    ----------
    undirected : {"bidirected", "drop"}, optional
        How to render an unoriented contemporaneous edge (causal-learn
        ``(-1, -1)``, tigramite ``o-o``). ``"bidirected"`` writes symmetric
        1s -- adjacency is preserved but the pair scores 1 TP + 1 FP against a
        directed ground truth. ``"drop"`` zeroes both cells, scoring 1 FN.
        Both cost ``SHD = 1``; the choice moves F1 only, in opposite
        directions. Defaults to ``"bidirected"``.
    conflict : {"drop", "bidirected"}, optional
        How to render a conflicting contemporaneous edge (causal-learn
        ``(1, 1)`` from ``priority=1`` conflict marking, tigramite ``x-x``).
        Defaults to ``"drop"``. A C node has no parents, so an arrowhead into
        C (a conflicting C edge, or an ``X -> C``) is a conflict too: under
        ``"drop"`` it renders as ``C -> X``.
    keep_undirected : bool, optional
        Legacy spelling of ``undirected``. ``True`` -> ``"bidirected"``,
        ``False`` -> ``"drop"``. Cannot be combined with ``undirected``.

    Lagged edges are directed by time order and ignore both policies: any
    lagged adjacency, even a (rare) ``o-o`` or ``x-x``, renders as the
    forward link.

    Notes
    -----
    The two CDNOTS entry points render ``o-o`` **differently**, and both
    defaults are deliberate:

    * :func:`run_cdnots` keeps it (``"bidirected"``) -- right when
      contemporaneous structure is real.
    * :func:`run_cdnots_plus` drops it -- right when lag-0 edges are invalid
      or confounded and would otherwise produce a flood of false positives.

    Prefer choosing after the fact via ``CdnotsResult.to_binary()`` rather
    than reaching for this function directly.
    """
    undirected = _resolve_undirected(undirected, keep_undirected)
    if conflict not in CONFLICT_POLICIES:
        raise ValueError(
            f"conflict must be one of {CONFLICT_POLICIES}, got {conflict!r}"
        )

    g, no_of_inst_var, orig = _prep_for_conversion(gg, num_lags, include_C, num_c_cols)

    # Lag-0 block, C columns included: the C edges oriented above are (0, 1)
    # and pass through; a C conflict is handled below.
    for i in range(no_of_inst_var):
        for j in range(no_of_inst_var):
            if i == j:
                continue
            else:
                if orig[i, j] == -1 and orig[j, i] == 1:
                    g.G.graph[i, j] = 0
                    g.G.graph[j, i] = 1
                elif orig[i, j] == 1 and orig[j, i] == -1:
                    g.G.graph[i, j] = 1
                    g.G.graph[j, i] = 0
                elif orig[i, j] == 1 and orig[j, i] == 1:
                    # Conflicting orientation (tigramite 'x-x'), from
                    # priority=1 conflict marking. Dropped by default rather
                    # than emitting two contradictory directed edges.
                    mark = 1 if conflict == "bidirected" else 0
                    g.G.graph[i, j] = mark
                    g.G.graph[j, i] = mark
                elif orig[i, j] == -1 and orig[j, i] == -1:
                    # Unoriented (tigramite 'o-o'): the CPDAG says the two are
                    # adjacent but the orientation is not identified. Note
                    # PCMCI+ *emits and keeps* o-o -- dropping it is this
                    # library's rendering policy, not tigramite's behaviour.
                    mark = 1 if undirected == "bidirected" else 0
                    g.G.graph[i, j] = mark
                    g.G.graph[j, i] = mark

    # The policies are for edges whose direction is not identified. A C edge
    # is C -> X (C has no parents), so an arrowhead into C -- a conflict or an
    # X -> C (e.g. a priority=0 collider overwrite), both x-x in to_marks --
    # under "drop" loses only that arrowhead; "bidirected" shows both.
    if include_C:
        for c in range(no_of_inst_var - num_c_cols, no_of_inst_var):
            for i in range(no_of_inst_var - num_c_cols):
                if orig[c, i] == 1:
                    keep = conflict == "bidirected"
                    g.G.graph[c, i], g.G.graph[i, c] = int(keep), 1

    # A lagged edge's direction is fixed by time order, so any adjacency
    # renders as the forward link, whatever the policies (to_marks keeps a
    # lagged o-o / x-x). Only the cell itself is written: its mirror is a
    # different lag relationship.
    lagged = _lagged_marks(orig, no_of_inst_var)
    g.G.graph[:no_of_inst_var, no_of_inst_var:] = lagged != ""

    # Reshape to (no_of_inst_var, no_of_inst_var, num_lags+1) tigramite format
    # Output includes C columns when include_C=True
    return _to_tigramite_layout(g.G.graph, no_of_inst_var, num_lags)


def cdnots_to_tigramite_marks(gg, num_lags=0, include_C=True, num_c_cols=1):
    """Render a causal-learn CPDAG as a **lossless** tigramite string array.

    Same ``[cause, effect, lag]`` layout as
    :func:`cdnots_to_tigramite_graph`, but dtype ``<U3`` carrying the endpoint
    marks instead of 0/1, so nothing is discarded:

    ==================  ==========  ===================================
    causal-learn        tigramite   meaning
    ==================  ==========  ===================================
    ``(-1, 1)``         ``-->``     directed
    ``(1, -1)``         ``<--``     directed, reversed
    ``(-1, -1)``        ``o-o``     unoriented (Markov equivalent)
    ``(1, 1)``          ``x-x``     conflicting orientation
    ``(0, 0)``          ``""``      no edge
    ==================  ==========  ===================================

    This is the view that makes CDNOTS output comparable to
    ``run_pcmciplus`` mark for mark. It cannot be scored by
    :func:`~causalts.utils.helpers.evaluate_graph` -- use
    :func:`cdnots_to_tigramite_graph` for that -- but it can be plotted
    directly.

    Lags >= 1 are time-ordered, so ``o-o`` and ``x-x`` should not occur
    there; if the underlying graph carries one it is reported rather than
    silently rewritten. An arrowhead that background knowledge rules out --
    on the earlier node of a lagged edge, or into a C node (``include_C``) --
    is reported as ``x-x``; these are the only states whose raw marks are not
    kept.
    """
    _, no_of_inst_var, orig = _prep_for_conversion(gg, num_lags, include_C, num_c_cols)

    # marks[a, b] describes the link b -> a, matching the transpose baked into
    # _to_tigramite_layout (result[cause, effect, lag] == marks[effect, col]).
    # Lag >= 1 is deliberately *not* mirrored -- tigramite only fills
    # graph[i, j, tau] for tau > 0, since the transposed cell means a
    # different lag relationship, not the reverse of this one.
    marks = np.full(orig.shape, "", dtype="<U3")
    marks[:no_of_inst_var, no_of_inst_var:] = _lagged_marks(orig, no_of_inst_var)

    # The whole lag-0 block, C columns included. Contemporaneous marks *are*
    # mirrored in tigramite ("-->" opposite "<--"), and the cases below are
    # transpose-consistent by construction. C edges must be in the block: the
    # C-orientation step in _prep_for_conversion writes (0, 1), not (-1, 1),
    # so a rule keyed only on (-1, 1) would leave them unmirrored.
    blk = (slice(0, no_of_inst_var), slice(0, no_of_inst_var))
    sub = orig[blk]
    sub_t = sub.T
    marks[blk] = np.select(
        [
            (sub == 1) & (sub_t == 1),  # conflicting -- check before "-->"
            (sub == 1) & (sub_t != 1),  # (1, -1) directed, or (1, 0) C edge
            (sub_t == 1) & (sub != 1),  # the mirror of the above
            (sub == -1) & (sub_t == -1),  # unoriented
        ],
        ["x-x", "-->", "<--", "o-o"],
        default="",
    )
    np.fill_diagonal(marks, "")
    if include_C:  # C has no parents: an arrowhead into C is a conflict
        for c in range(no_of_inst_var - num_c_cols, no_of_inst_var):
            for i in range(no_of_inst_var - num_c_cols):
                if orig[c, i] == 1:
                    marks[c, i] = marks[i, c] = "x-x"

    return _to_tigramite_layout(marks, no_of_inst_var, num_lags)


@dataclass
class CdnotsPlusPrefix:
    """Everything :func:`run_cdnots_plus` computes before Meek orientation.

    The expensive part of CDNOTS+ is the skeleton search and, dominantly, the
    majority-rule collider phase in ``uc_sepset`` -- together ~96% of runtime
    at d=40. Neither depends on ``orient_margin``, which only reaches
    :func:`phase_three`. Holding a prefix lets a caller score many margins
    against a single collider phase.

    Get one via ``run_cdnots_plus(..., return_prefix=True)``, then call
    :func:`finish_cdnots_plus` on it repeatedly. Reuse is safe: ``meek`` and
    ``phase_three`` both deepcopy their input, so ``cg_sepset`` is never
    mutated.
    """

    cg_sepset: object
    data: object
    pvals: list
    pvalue_matrix: object
    lv_stats: object
    df_orig: object
    var_names_orig: list
    lags: list
    lag_list: object
    num_lags: int
    include_C: bool
    num_c_cols: int
    c_node_names: object
    alpha: float
    priority: int
    background_knowledge: object = None
    forbidden_ancestor_pairs: object = None
    knowledge: object = None
    verbose: bool = False
    orient_margin: float = 0.1
    lag0_orient: object = None
    lag0_exclude: list = None
    lag_refinements: object = None
    refined_orient_margin: object = None
    refined_lag0_orient: object = None


_UNSET = object()


def finish_cdnots_plus(state, orient_margin=None, lag0_orient=_UNSET):
    """Run the orientation tail of CDNOTS+ on a :class:`CdnotsPlusPrefix`.

    Meek rules -> nonstationarity orientation -> (optional) lag-0 orientation
    from non-Gaussianity -> tigramite conversion. The optional step works on a
    copy, so the prefix is reusable. A lagged-link refinement
    (``addback_alpha`` / ``veto_alpha``) is already applied to the prefix by
    :func:`run_cdnots_plus`; this function only reports it.
    :func:`run_cdnots_plus` calls exactly this function, so reusing a prefix
    is bit-identical to calling ``run_cdnots_plus`` directly with the same
    ``orient_margin`` -- there is no second implementation to drift.

    ``orient_margin`` defaults to the value stored on ``state`` (the one passed
    to ``run_cdnots_plus``), so ``finish_cdnots_plus(state)`` reproduces the
    direct call. Pass it to override. ``lag0_orient`` works the same way
    (``None`` turns the step off), so one prefix can be finished with and
    without it.
    """
    if orient_margin is None:
        orient_margin = state.orient_margin
    if lag0_orient is _UNSET:
        lag0_orient = state.lag0_orient
    if lag0_orient not in (None, "lingam"):
        raise ValueError(f"lag0_orient must be None or 'lingam', got {lag0_orient!r}")
    refined = state.refined_orient_margin
    if refined is not None and (
        orient_margin != refined or lag0_orient != state.refined_lag0_orient
    ):
        # The lagged-link refinement was decided on the orientation at that
        # margin and lag0_orient; other settings would need it recomputed.
        raise ValueError(
            "this prefix was refined (addback_alpha / veto_alpha) at "
            f"orient_margin={refined}, lag0_orient={state.refined_lag0_orient!r}; run "
            "run_cdnots_plus again for other settings"
        )
    num_lags = state.num_lags
    include_C = state.include_C
    num_c_cols = state.num_c_cols
    data = state.data

    cg_meek = meek(
        state.cg_sepset,
        background_knowledge=state.background_knowledge,
        num_lags=num_lags,
        forbidden_ancestor_pairs=state.forbidden_ancestor_pairs,
    )

    if include_C:
        no_of_inst_var_ph3 = data.shape[1] // (num_lags + 1)
        c_start_ph3 = no_of_inst_var_ph3 - num_c_cols
        c_indx_for_phase3 = pd.DataFrame(data[:, c_start_ph3:no_of_inst_var_ph3])
    else:
        c_indx_for_phase3 = None

    cg_obj = phase_three(
        cg_meek,
        data,
        c_indx=c_indx_for_phase3,
        num_lags=num_lags,
        width="median",
        include_C=include_C,
        num_c_cols=num_c_cols,
        orient_margin=orient_margin,
        background_knowledge=state.background_knowledge,
        forbidden_ancestor_pairs=state.forbidden_ancestor_pairs,
    )
    knowledge = state.knowledge
    if knowledge is not None and knowledge.has_path_constraints:
        from .ancestral import apply_ancestral_constraints

        cg_obj, _ = apply_ancestral_constraints(
            cg_obj,
            knowledge,
            state.var_names_orig,
            num_lags,
            include_C,
            num_c_cols=num_c_cols,
            background_knowledge=state.background_knowledge,
            df=state.df_orig,
            verbose=state.verbose,
        )

    lag0_decisions = None
    if lag0_orient == "lingam":
        from .lingam_orient import lingam_orient_lag0

        cg_obj = deepcopy(cg_obj)
        n_inst = data.shape[1] // (num_lags + 1)
        forb = state.forbidden_ancestor_pairs or ()
        lag0_decisions = lingam_orient_lag0(
            cg_obj,
            data,
            num_lags,
            c_start=n_inst - num_c_cols if include_C else n_inst,
            can_orient=lambda cg, x, y: _can_orient(
                cg, x, y, state.background_knowledge, forb
            ),
            orient=_orient,
            exclude_vars=state.lag0_exclude or (),
        )
        if any(d["oriented"] for d in lag0_decisions):
            # Propagate the new arrowheads (phase three ends with Meek too).
            cg_obj = meek(
                cg_obj,
                background_knowledge=state.background_knowledge,
                num_lags=num_lags,
                forbidden_ancestor_pairs=state.forbidden_ancestor_pairs,
            )

    # CDNOTS+ drops unoriented lag-0 edges, where CDNOTS keeps them. This is
    # deliberate, not an oversight: it suits regimes where contemporaneous
    # edges are invalid or confounded. Score the other way with
    # result.to_binary(undirected="bidirected") -- see the converter docstring.
    cg_tig = cdnots_to_tigramite_graph(
        cg_obj,
        num_lags=num_lags,
        include_C=include_C,
        num_c_cols=num_c_cols,
        undirected="drop",
    )
    result = _build_result(
        cg_obj,
        cg_tig,
        state.pvalue_matrix,
        state.pvals,
        state.df_orig,
        state.var_names_orig,
        state.lags,
        state.lag_list,
        include_C,
        state.lv_stats,
        num_lags=num_lags,
        alpha=state.alpha,
        priority=state.priority,
        stable=False,
        num_c_cols=num_c_cols,
        c_node_names=state.c_node_names,
        undirected_policy="drop",
    )
    if lag0_decisions is not None:
        names = state.var_names_orig
        for d in lag0_decisions:
            d["cause_name"] = names[d["cause"]]
            d["effect_name"] = names[d["effect"]]
    result.lag0_orientations = lag0_decisions
    result.lag_refinements = state.lag_refinements
    return result
