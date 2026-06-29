# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""SCM fitting and counterfactual analysis via DoWhy GCM."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .graph_bridge import graph_to_networkx, make_lagged_df


def fit_scm(
    graph: np.ndarray,
    data: pd.DataFrame,
    var_names: list[str] | None = None,
    mechanism_type: str = "auto",
    include_c: bool = False,
):
    """Fit a Structural Causal Model to the discovered graph.

    Fits functional mechanisms (linear, GP, or auto-selected) to each node
    in the time-expanded causal graph. The fitted SCM unlocks counterfactual
    analysis and root cause attribution.

    Parameters
    ----------
    graph : np.ndarray
        Shape (d, d, max_lag+1) from CDNOTS/CEDAR/GRACE.
    data : pd.DataFrame
        Original (T, d) time series DataFrame.
    var_names : list[str] or None
        Variable names. Defaults to data.columns.
    mechanism_type : str
        "auto" -- DoWhy auto-selects best mechanism per node.
        "linear" -- LinearGaussianCPD for all nodes.
        "gp" -- AdditiveNoiseModel with Gaussian Process.
    include_c : bool
        Whether to include the CDNOTS C (nonstationarity) node.

    Returns
    -------
    tuple
        ``(scm, dag, lagged_df)`` where ``scm`` is the fitted
        InvertibleStructuralCausalModel, ``dag`` is the NetworkX
        DiGraph used, and ``lagged_df`` is the lag-embedded DataFrame
        (pass to counterfactual / attribute_anomaly).
    """
    from ._compat import require_gcm

    require_gcm("SCM fitting")
    import dowhy.gcm as gcm
    from dowhy.gcm import auto

    if var_names is None:
        var_names = list(data.columns)

    max_lag = graph.shape[2] - 1
    dag = graph_to_networkx(
        graph,
        var_names=var_names,
        mode="summary",
        include_c=include_c,
        cycle_resolution="drop",
    )
    lagged_df = make_lagged_df(data, max_lag, var_names=var_names)

    # Only keep columns that exist in the dag
    cols = [c for c in lagged_df.columns if c in dag.nodes]
    lagged_df = lagged_df[cols]

    scm = gcm.InvertibleStructuralCausalModel(dag)

    if mechanism_type == "auto":
        auto.assign_causal_mechanisms(scm, lagged_df, override_models=True)
    elif mechanism_type == "linear":
        _assign_linear(scm, dag)
    elif mechanism_type == "gp":
        _assign_gp(scm, dag)
    else:
        raise ValueError(
            f"mechanism_type must be 'auto', 'linear', or 'gp', got {mechanism_type!r}"
        )

    gcm.fit(scm, lagged_df)
    return scm, dag, lagged_df


def _assign_linear(scm, dag) -> None:
    import dowhy.gcm as gcm
    from dowhy.gcm.ml import create_linear_regressor

    for node in dag.nodes:
        if list(dag.predecessors(node)):
            scm.set_causal_mechanism(
                node, gcm.AdditiveNoiseModel(create_linear_regressor())
            )
        else:
            scm.set_causal_mechanism(node, gcm.EmpiricalDistribution())


def _assign_gp(scm, dag) -> None:
    import dowhy.gcm as gcm
    from dowhy.gcm.ml import create_gaussian_process_regressor

    for node in dag.nodes:
        if list(dag.predecessors(node)):
            scm.set_causal_mechanism(
                node,
                gcm.AdditiveNoiseModel(create_gaussian_process_regressor()),
            )
        else:
            scm.set_causal_mechanism(node, gcm.EmpiricalDistribution())


def counterfactual(
    scm,
    lagged_df: pd.DataFrame,
    intervention: dict,
    target: str,
) -> pd.Series:
    """Estimate counterfactual values for a target variable.

    Parameters
    ----------
    scm : InvertibleStructuralCausalModel
        Fitted InvertibleStructuralCausalModel from fit_scm().
    lagged_df : pd.DataFrame
        Lag-embedded DataFrame returned by fit_scm().
    intervention : dict
        Dict mapping node names to fixed intervened values.
        E.g. {"X0_lag1": 0.5} clamps X0(t-1) to 0.5 (do-operator).
    target : str
        Name of the target variable (current-time node).

    Returns
    -------
    pd.Series
        Series of counterfactual values for the target variable, one
        per row of lagged_df.
    """
    from ._compat import require_gcm

    require_gcm("counterfactual analysis")
    import dowhy.gcm as gcm

    # Each intervention is a function x -> new_value (do-operator style)
    interventions = {k: (lambda x, v=v: v) for k, v in intervention.items()}
    cf_samples = gcm.counterfactual_samples(
        scm,
        interventions,
        observed_data=lagged_df,
    )
    return cf_samples[target]
