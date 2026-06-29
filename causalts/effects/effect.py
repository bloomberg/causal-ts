# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Causal effect estimation for time series graphs via DoWhy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .graph_bridge import _lag_node, graph_to_dowhy_model


def estimate_effect(
    graph: np.ndarray,
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    treatment_lag: int = 1,
    method: str = "backdoor.linear_regression",
    var_names: list[str] | None = None,
    confidence_intervals: bool = True,
    n_bootstraps: int = 100,
    include_c: bool = False,
):
    """Estimate the average causal effect of treatment(t-lag) on outcome(t).

    Uses DoWhy's identification and estimation pipeline with the discovered
    causal graph as the identifying assumption.

    Note: Assumes approximate stationarity. For strongly nonstationary series,
    apply on detrended data or stationary segments.

    Parameters
    ----------
    graph : np.ndarray
        Shape (d, d, max_lag+1) from CDNOTS/CEDAR/GRACE.
    data : pd.DataFrame
        Original (T, d) time series DataFrame.
    treatment : str
        Variable name of the cause (e.g. "X0").
    outcome : str
        Variable name of the effect (e.g. "X2").
    treatment_lag : int
        Which lag of treatment to use (default 1).
    method : str
        DoWhy estimation method. Common choices:
        "backdoor.linear_regression" (default, fast, interpretable),
        "backdoor.propensity_score_matching" (nonlinear).
    var_names : list[str] or None
        Variable names. Defaults to data.columns.
    confidence_intervals : bool
        Whether to compute bootstrap CIs.
    n_bootstraps : int
        Bootstrap replications for CI computation.
    include_c : bool
        Whether to include the CDNOTS C node.

    Returns
    -------
    dowhy.CausalEstimate
        Causal estimate with fields:
        ``.value`` -- point estimate of ATE,
        ``.get_confidence_intervals()`` -- (lower, upper) if requested,
        ``.test_stat_significance()`` -- p-value for H0: effect=0.
    """
    from ._compat import require_dowhy

    require_dowhy("effect estimation")

    if var_names is None:
        var_names = list(data.columns)

    model, lagged_df = graph_to_dowhy_model(
        graph,
        data,
        treatment=treatment,
        outcome=outcome,
        treatment_lag=treatment_lag,
        var_names=var_names,
        include_c=include_c,
    )

    identified = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified,
        method_name=method,
        confidence_intervals=confidence_intervals,
        method_params=(
            {"num_simulations": n_bootstraps} if confidence_intervals else {}
        ),
    )
    return estimate


def print_effect_summary(estimate, treatment: str, outcome: str, lag: int) -> None:
    """Print a human-readable summary of a causal effect estimate."""
    treatment_node = _lag_node(treatment, lag)
    print(f"\nCausal effect: {treatment_node} → {outcome}")
    print(f"  Method:    {estimate.params.get('method_name', 'unknown')}")
    print(f"  ATE:       {estimate.value:.6f}")
    try:
        ci = estimate.get_confidence_intervals()
        if ci is not None:
            print(f"  95% CI:    ({ci[0]:.6f}, {ci[1]:.6f})")
    except Exception:
        pass
    try:
        sig = estimate.test_stat_significance()
        if sig is not None:
            print(f"  p-value:   {sig.get('p_value', 'n/a')}")
    except Exception:
        pass
