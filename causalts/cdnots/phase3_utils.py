"""
This file has been modified by Mohammad Fesanghary on July 28 2025.
Original Package: causal-learn 1.9.0
Original License: MIT License
"""

# SPDX-License-Identifier: MIT

import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from ..ci_tests.kci_gpu import apply_H, rbf_kernel_gpu
from .meek import meek
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
from .uc_sepset import uc_sepset

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
    np.ndarray of shape (T, k) where k depends on the preset.
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


def phase_three(
    g,
    data,
    c_indx=None,
    num_lags=0,
    width="empirical",
    include_C=True,
    num_c_cols=1,
    orient_margin=0.0,
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
        ``keep_undirected=False`` then drop them.

    Returns
    -------
    CausalGraph
        Adjacency matrix after phase 3.
    """

    if not include_C:
        # No C variable — skip nonstationarity-based orientation, just apply Meek
        return meek(cg=deepcopy(g), num_lags=num_lags)

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
        gns.G.graph[hypo_cau[idd], hypo_eff[idd]] = -1
        gns.G.graph[hypo_eff[idd], hypo_cau[idd]] = 1

        if num_lags != 0:  # direction stability over diagonal
            for ll in range(1, num_lags + 1):
                cau = [x + (ll * no_of_inst_var) for x in hypo_cau[idd]]
                gns.G.graph[cau, hypo_eff[idd] + (ll * no_of_inst_var)] = -1
                gns.G.graph[hypo_eff[idd] + (ll * no_of_inst_var), cau] = 1

        Vns = [i for i in Vns if i != sink]

    return meek(cg=gns, num_lags=num_lags)


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


def _is_discrete_aware(indep_test):
    """Return True if the CI test natively handles discrete columns."""
    from ..ci_tests.cmiknn_mixed_gpu import CMIknnMixedGPU
    from ..ci_tests.stratified_cit import StratifiedCIT

    return isinstance(indep_test, (StratifiedCIT, CMIknnMixedGPU))


def make_background_knowledge(
    var_names,
    num_lags,
    include_C=True,
    forbidden=None,
    required=None,
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
    n_per_lag = d + (1 if include_C else 0)

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
    orient_margin=0.0,
):
    # --- Discrete column handling ---
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

    # Resolve AncestralKnowledge → BackgroundKnowledge (edge constraints)
    _forb_ancestor_pairs = None
    if knowledge is not None:
        if background_knowledge is None:
            background_knowledge = knowledge.to_background_knowledge(
                _var_names_orig, num_lags, include_C
            )
        if knowledge._forbidden_ancestors:
            _forb_ancestor_pairs = knowledge.resolve_forbidden_ancestor_pairs(
                _var_names_orig, num_lags, include_C
            )

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

        embedded_disc = compute_discrete_indices(
            discrete_cols, df, lags, include_C=include_C
        )
        indep_test.discrete_cols = embedded_disc
        if hasattr(indep_test, "_disc_set"):
            indep_test._disc_set = set(embedded_disc)
        if hasattr(indep_test, "_disc_global"):
            indep_test._disc_global = set(embedded_disc)

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
        )

    cg_skel_cnst = c_lag_cnst(
        cg_skel, num_lags, include_C=include_C, num_c_cols=num_c_cols
    )
    cg_sepset = uc_sepset(
        cg_skel_cnst,
        priority=priority,
        num_lags=num_lags,
        background_knowledge=background_knowledge,
        contemp_collider_rule="majority",
        ci_test=indep_test,
        alpha=alpha,
        lagged_parents=None,
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
    )
    if knowledge is not None and knowledge.has_path_constraints:
        from .ancestral import apply_ancestral_constraints

        cg_obj, _ = apply_ancestral_constraints(
            cg_obj,
            knowledge,
            _var_names_orig,
            num_lags,
            include_C,
            background_knowledge=background_knowledge,
            df=_df_orig,
            verbose=verbose,
        )

    cg_tig = cdnots_to_tigramite_graph(
        cg_obj, num_lags=num_lags, include_C=include_C, num_c_cols=num_c_cols
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
    orient_margin=0.0,
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
        Collider conflict resolution strategy (0-4).
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
    orient_margin : float
        Confidence gate on the ``include_C`` nonstationarity orientation
        (see :func:`phase_three`). ``0.0`` (default) preserves the existing
        behaviour, which commits every contemporaneous edge to a direction.

    Returns
    -------
    CdnotsResult
        Self-contained result with graph, p-values, plotting, and
        effect-estimation methods.
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
    legacy_mci_conds=False,
    orient_margin=0.1,
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
        PCMCI+'s default and the setting empirically better for CDNOTS+ (the
        opposite is true for plain CDNOTS, which defaults to 0.05).
    max_degree : int or None
        Maximum conditioning-set size in phase 1.
    max_conds_py : int or None
        Maximum parents of Y in phase 2 MCI conditioning.
    max_conds_px : int or None
        Maximum parents of X in phase 2 MCI conditioning.
    priority : int
        Collider conflict resolution strategy (0-4). Default 1 (orient
        bi-directed): contradictory collider orientations are marked
        conflicting and dropped, matching PCMCI+'s ``x-x`` treatment, rather
        than tie-broken into a committed direction.
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
        Restore the pre-alignment phase-2 conditioning rule, which dropped
        every lag-copy of a tested variable from the conditioning set rather
        than only the exact tested node. Default False matches PCMCI+'s
        ``_run_pcalg_test``.
    orient_margin : float
        Confidence gate on the ``include_C`` nonstationarity orientation
        (see :func:`phase_three`). Default 0.1. ``0.0`` restores the
        original behaviour, which commits every contemporaneous edge to a
        direction. No effect when ``include_C=False``.

    Returns
    -------
    CdnotsResult
        Self-contained result.
    """
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

    # Resolve AncestralKnowledge → BackgroundKnowledge (edge constraints)
    _forb_ancestor_pairs = None
    if knowledge is not None:
        if background_knowledge is None:
            background_knowledge = knowledge.to_background_knowledge(
                _var_names_orig, num_lags, include_C
            )
        if knowledge._forbidden_ancestors:
            _forb_ancestor_pairs = knowledge.resolve_forbidden_ancestor_pairs(
                _var_names_orig, num_lags, include_C
            )

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

        embedded_disc = compute_discrete_indices(
            discrete_cols, df, lags, include_C=include_C
        )
        indep_test.discrete_cols = embedded_disc
        if hasattr(indep_test, "_disc_set"):
            indep_test._disc_set = set(embedded_disc)
        if hasattr(indep_test, "_disc_global"):
            indep_test._disc_global = set(embedded_disc)

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
    )
    if return_pvals:
        cg_skel, pvals_phase1 = skel_result
    else:
        cg_skel, pvals_phase1 = skel_result, []

    # --- Extract parent sets from phase 1 skeleton ---
    no_of_inst_var = data.shape[1] // (num_lags + 1)
    lagged_parents = {}
    for j in range(no_of_inst_var):
        lagged_parents[j] = [p for p in cg_skel.neighbors(j) if p >= no_of_inst_var]

    # --- Extend data to 2×num_lags depth for phase 2 conditioning ---
    if num_lags > 0:
        ext_lags = list(range(2 * num_lags + 1))
        data_ext = (
            pd.concat([df.shift(i) for i in ext_lags], axis=1)
            .iloc[2 * num_lags :]
            .values
        )
        indep_test.data = data_ext
        no_of_var_ext = data_ext.shape[1]
    else:
        no_of_var_ext = data.shape[1]

    # --- Phase 2: MCI re-test ---
    mci_result = mci_skeleton(
        cg=cg_skel,
        ci_test=indep_test,
        alpha=alpha,
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
    )
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
        )
        indep_test.data = data

    # --- Orientation pipeline (same as CDNOTS) ---
    cg_skel_cnst = c_lag_cnst(
        cg_skel_mci, num_lags, include_C=include_C, num_c_cols=num_c_cols
    )
    cg_sepset = uc_sepset(
        cg_skel_cnst,
        priority=priority,
        num_lags=num_lags,
        background_knowledge=background_knowledge,
        contemp_collider_rule="majority",
        ci_test=indep_test,
        alpha=alpha,
        lagged_parents=lagged_parents,
    )
    cg_meek = meek(
        cg_sepset,
        background_knowledge=background_knowledge,
        num_lags=num_lags,
        forbidden_ancestor_pairs=_forb_ancestor_pairs,
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
    )
    if knowledge is not None and knowledge.has_path_constraints:
        from .ancestral import apply_ancestral_constraints

        cg_obj, _ = apply_ancestral_constraints(
            cg_obj,
            knowledge,
            _var_names_orig,
            num_lags,
            include_C,
            background_knowledge=background_knowledge,
            df=_df_orig,
            verbose=verbose,
        )

    cg_tig = cdnots_to_tigramite_graph(
        cg_obj,
        num_lags=num_lags,
        include_C=include_C,
        num_c_cols=num_c_cols,
        keep_undirected=False,
    )
    return _build_result(
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
        stable=False,
        num_c_cols=num_c_cols,
        c_node_names=_c_node_names,
    )


def cdnots_to_tigramite_graph(
    gg, num_lags=0, include_C=True, num_c_cols=1, keep_undirected=True
):
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

    # When include_C=True, convert edges for non-C variables only
    # When include_C=False, convert all variables
    edge_vars = (
        c_start  # = no_of_inst_var - num_c_cols when include_C, else no_of_inst_var
    )

    # Decide from a snapshot of the incoming endpoints. The loop below visits
    # both (i, j) and (j, i) and writes in place, so reading the live array
    # would let one visit observe what the mirrored visit just wrote -- e.g.
    # the o-o branch writes (1, 1) under keep_undirected, which the bi-directed
    # branch would then mistake for a conflict marker and erase.
    orig = g.G.graph.copy()

    for i in range(edge_vars):
        for j in range(edge_vars):
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
                    # Bi-directed (i <-> j) from priority=1 conflict marking.
                    # PCMCI+ records these as 'x-x' and excludes them from the
                    # binary graph; drop them rather than emitting two
                    # contradictory directed edges.
                    g.G.graph[i, j] = 0
                    g.G.graph[j, i] = 0
                elif orig[i, j] == -1 and orig[j, i] == -1:
                    if keep_undirected:
                        g.G.graph[i, j] = 1
                        g.G.graph[j, i] = 1
                    else:
                        # Drop unresolved edges — matches PCMCI+'s
                        # treatment of o-o edges.
                        g.G.graph[i, j] = 0
                        g.G.graph[j, i] = 0

    # Reshape to (no_of_inst_var, no_of_inst_var, num_lags+1) tigramite format
    # Output includes C columns when include_C=True
    return np.swapaxes(
        np.swapaxes(
            g.G.graph[:no_of_inst_var].reshape(
                no_of_inst_var, num_lags + 1, no_of_inst_var
            ),
            1,
            2,
        ),
        0,
        1,
    )
