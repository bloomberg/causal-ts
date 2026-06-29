# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Causal influence quantification and distribution change detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .graph_bridge import make_lagged_df
from .root_cause import _parse_node


def arrow_strength(
    graph: np.ndarray,
    data: pd.DataFrame,
    target: str,
    var_names: list[str] | None = None,
    mechanism_type: str = "auto",
    include_c: bool = False,
) -> pd.DataFrame:
    """Quantify how strongly each parent influences a target node.

    Measures causal influence by comparing the target's conditional
    distribution when each parent edge exists vs. when parent is permuted.

    No ground truth required.

    Parameters
    ----------
    graph : np.ndarray
        Shape (d, d, max_lag+1).
    data : pd.DataFrame
        Original (T, d) time series DataFrame.
    target : str
        Target variable (current-time node, e.g. "Mek").
    var_names : list[str] or None
        Variable names.
    mechanism_type : str
        Passed to fit_scm if not cached.
    include_c : bool
        Whether to include the C node.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [parent_node, variable, lag, target,
        strength] sorted by absolute strength descending.
    """
    from ._compat import require_gcm

    require_gcm("arrow strength")
    import dowhy.gcm as gcm

    from .scm import fit_scm

    if var_names is None:
        var_names = list(data.columns)

    scm, dag, lagged_df = fit_scm(
        graph,
        data,
        var_names=var_names,
        mechanism_type=mechanism_type,
        include_c=include_c,
    )

    strengths = gcm.arrow_strength(scm, target)

    rows = []
    for (parent, tgt), val in strengths.items():
        variable, lag = _parse_node(parent)
        rows.append(
            {
                "parent_node": parent,
                "variable": variable,
                "lag": lag,
                "target": tgt,
                "strength": float(val),
            }
        )

    result = pd.DataFrame(rows)
    return result.sort_values("strength", key=abs, ascending=False).reset_index(
        drop=True
    )


def causal_influence(
    graph: np.ndarray,
    data: pd.DataFrame,
    target: str,
    var_names: list[str] | None = None,
    mechanism_type: str = "auto",
    include_c: bool = False,
    num_training_samples: int = 50000,
) -> pd.DataFrame:
    """Shapley decomposition of target's variability by upstream noise sources.

    Answers: "What fraction of target's variance comes from noise in each
    upstream variable?" Uses intrinsic causal influence (Janzing et al. 2021).

    No ground truth required.

    Parameters
    ----------
    graph : np.ndarray
        Shape (d, d, max_lag+1).
    data : pd.DataFrame
        Original (T, d) time series DataFrame.
    target : str
        Target variable (current-time node).
    var_names : list[str] or None
        Variable names.
    mechanism_type : str
        Passed to fit_scm if not cached.
    include_c : bool
        Whether to include the C node.
    num_training_samples : int
        Samples for the attribution model.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [node, variable, lag, influence_score]
        sorted by score descending.
    """
    from ._compat import require_gcm

    require_gcm("intrinsic causal influence")
    import dowhy.gcm as gcm

    from .scm import fit_scm

    if var_names is None:
        var_names = list(data.columns)

    scm, dag, lagged_df = fit_scm(
        graph,
        data,
        var_names=var_names,
        mechanism_type=mechanism_type,
        include_c=include_c,
    )

    influences = gcm.intrinsic_causal_influence(
        scm,
        target,
        num_training_samples=num_training_samples,
    )

    rows = []
    for node, score in influences.items():
        variable, lag = _parse_node(node)
        rows.append(
            {
                "node": node,
                "variable": variable,
                "lag": lag,
                "influence_score": float(score),
            }
        )

    result = pd.DataFrame(rows)
    return result.sort_values("influence_score", ascending=False).reset_index(drop=True)


def parent_relevance(
    graph: np.ndarray,
    data: pd.DataFrame,
    target: str,
    var_names: list[str] | None = None,
    mechanism_type: str = "auto",
    include_c: bool = False,
) -> pd.DataFrame:
    """Rank direct parents by explanatory power (Shapley over parent subsets).

    Answers: "Which direct parents matter most for predicting target?"
    Includes noise relevance as a separate entry.

    No ground truth required.

    Parameters
    ----------
    graph : np.ndarray
        Shape (d, d, max_lag+1).
    data : pd.DataFrame
        Original (T, d) time series DataFrame.
    target : str
        Target variable.
    var_names : list[str] or None
        Variable names.
    mechanism_type : str
        Passed to fit_scm if not cached.
    include_c : bool
        Whether to include the C node.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [node, variable, lag, relevance_score].
        Last row is "noise" with the unexplained fraction.
    """
    from ._compat import require_gcm

    require_gcm("parent relevance")
    import dowhy.gcm as gcm

    from .scm import fit_scm

    if var_names is None:
        var_names = list(data.columns)

    scm, dag, lagged_df = fit_scm(
        graph,
        data,
        var_names=var_names,
        mechanism_type=mechanism_type,
        include_c=include_c,
    )

    relevance_dict, noise_relevance = gcm.parent_relevance(scm, target)

    rows = []
    for key, score in relevance_dict.items():
        # Keys are (parent, target) tuples
        node = key[0] if isinstance(key, tuple) else key
        variable, lag = _parse_node(node)
        rows.append(
            {
                "node": node,
                "variable": variable,
                "lag": lag,
                "relevance_score": float(score),
            }
        )
    noise_val = (
        float(noise_relevance[0])
        if hasattr(noise_relevance, "__len__")
        else float(noise_relevance)
    )
    rows.append(
        {
            "node": "noise",
            "variable": "noise",
            "lag": 0,
            "relevance_score": noise_val,
        }
    )

    result = pd.DataFrame(rows)
    return result.sort_values("relevance_score", ascending=False).reset_index(drop=True)


def distribution_change(
    graph: np.ndarray,
    data_old: pd.DataFrame,
    data_new: pd.DataFrame,
    target: str,
    var_names: list[str] | None = None,
    mechanism_type: str = "auto",
    include_c: bool = False,
    robust: bool = False,
) -> pd.DataFrame:
    """Identify which upstream mechanisms changed between two time periods.

    Given old (reference) and new data, attributes the shift in target's
    distribution to upstream nodes using Shapley decomposition.

    No ground truth required.

    Parameters
    ----------
    graph : np.ndarray
        Shape (d, d, max_lag+1).
    data_old : pd.DataFrame
        Reference period DataFrame.
    data_new : pd.DataFrame
        New/current period DataFrame.
    target : str
        Target variable to explain shift in.
    var_names : list[str] or None
        Variable names.
    mechanism_type : str
        Passed to fit_scm.
    include_c : bool
        Whether to include the C node.
    robust : bool
        If True, use multiply-robust estimator (ICML 2024).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [node, variable, lag, contribution].
        Sorted by absolute contribution descending.
    """
    from ._compat import require_gcm

    require_gcm("distribution change")
    import dowhy.gcm as gcm

    from .scm import fit_scm

    if var_names is None:
        var_names = list(data_old.columns)

    max_lag = graph.shape[2] - 1

    # Fit SCM on old data
    scm, dag, _ = fit_scm(
        graph,
        data_old,
        var_names=var_names,
        mechanism_type=mechanism_type,
        include_c=include_c,
    )

    old_lagged = make_lagged_df(data_old, max_lag, var_names=var_names)
    new_lagged = make_lagged_df(data_new, max_lag, var_names=var_names)

    cols = [c for c in old_lagged.columns if c in dag.nodes]
    old_lagged = old_lagged[cols]
    new_lagged = new_lagged[cols]

    if robust:
        contributions = gcm.distribution_change_robust(
            scm,
            old_lagged,
            new_lagged,
            target,
        )
    else:
        contributions = gcm.distribution_change(
            scm,
            old_lagged,
            new_lagged,
            target,
        )

    rows = []
    for node, val in contributions.items():
        variable, lag = _parse_node(node)
        rows.append(
            {
                "node": node,
                "variable": variable,
                "lag": lag,
                "contribution": float(val),
            }
        )

    result = pd.DataFrame(rows)
    return result.sort_values("contribution", key=abs, ascending=False).reset_index(
        drop=True
    )
