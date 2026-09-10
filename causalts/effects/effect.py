# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Causal effect estimation for time series graphs via DoWhy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .graph_bridge import (
    _lag_node,
    force_parent_adjustment_set,
    graph_to_dowhy_model,
    verify_parent_adjustment_set,
)


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
    cycle_resolution: str = "error",
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
    cycle_resolution : str
        What to do with an unoriented contemporaneous edge. ``"error"``
        (default) refuses: silently dropping one changes which backdoor paths
        exist, so it can create or erase confounding and move the estimate.
        ``"drop"`` or ``"first"`` opts in, and also relaxes the adjustment set
        to treat an unoriented lag-0 neighbour of the treatment as a parent --
        which biases the estimate if it is really a child.

    Returns
    -------
    dowhy.CausalEstimate
        Causal estimate with fields:
        ``.value`` -- point estimate of ATE,
        ``.get_confidence_intervals()`` -- (lower, upper) if requested,
        ``.test_stat_significance()`` -- p-value for H0: effect=0.
    """
    model, identified, estimate, _ = _identify_and_estimate(
        graph,
        data,
        treatment=treatment,
        outcome=outcome,
        treatment_lag=treatment_lag,
        method=method,
        var_names=var_names,
        confidence_intervals=confidence_intervals,
        n_bootstraps=n_bootstraps,
        include_c=include_c,
        cycle_resolution=cycle_resolution,
    )
    return estimate


def _identify_and_estimate(
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
    cycle_resolution: str = "error",
):
    """Build the model, force the parent estimand, and estimate.

    Shared by :func:`estimate_effect` and
    :func:`~causalts.effects.validate.refute_effect`. Refutation has to attack
    the *same* estimand the estimate came from -- if it rebuilt the model
    independently it would silently get DoWhy's minimal backdoor set instead of
    the parent set forced below, and refute a quantity nobody reported.

    Returns ``(model, identified_estimand, estimate, lagged_df)``.
    """
    from ._compat import require_identification

    # Not require_dowhy: this path goes through DoWhy's classic identifier,
    # which has a narrower requirement than the rest of the bridge.
    require_identification("effect estimation")

    if not isinstance(treatment, str) or not isinstance(outcome, str):
        # DoWhy's own construct_adjustment_estimand() does `outcome_name =
        # outcome_name[0]` unconditionally -- a multivariate outcome would be
        # silently truncated to its first element rather than raising, and this
        # module's ATE-only scope was never designed to survive that. Refuse
        # rather than build a plausible-looking estimand for a query nobody
        # asked.
        raise TypeError(
            f"treatment and outcome must each be a single variable name, got "
            f"{treatment!r} and {outcome!r}."
        )

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
        cycle_resolution=cycle_resolution,
    )

    identified = model.identify_effect(proceed_when_unidentifiable=True)

    # DoWhy prefers a *minimal* backdoor set, found by searching the unrolled
    # graph as built above -- which only shows a confounder that sits within
    # required_lag_depth()'s horizon. The treatment's parents are a valid
    # adjustment set regardless of whether a confounder further back is
    # visible at all, so adjust for those instead and keep the reported
    # estimand consistent.
    # One knob, not two. Both `cycle_resolution` and `strict_contemporaneous`
    # govern the same ambiguity -- an unoriented lag-0 edge, which could be a
    # parent or a child -- so opting into an arbitrary resolution of the graph
    # has to opt into an arbitrary classification of the adjustment set as well.
    # Otherwise cycle_resolution="drop" is documented but unreachable: the graph
    # builds and the adjustment set then refuses.
    parents = force_parent_adjustment_set(
        identified,
        graph,
        treatment,
        treatment_lag,
        outcome,
        var_names,
        method,
        available=list(lagged_df.columns),
        strict_contemporaneous=cycle_resolution == "error",
    )

    estimate = model.estimate_effect(
        identified,
        method_name=method,
        confidence_intervals=confidence_intervals,
        # DoWhy otherwise reads effect modifiers off the graph -- any ancestor of
        # the outcome that is not one of the treatment -- and fits
        # treatment x modifier interactions. That is a CATE model whose .value is
        # an average, not the ATE this function documents. On a lag embedding the
        # variables it picks are an artefact of which nodes survived the
        # ancestral prune, and they also block DoWhy's linear sensitivity
        # analysis. Ask for the ATE explicitly.
        effect_modifiers=[],
        # Not relying on the default: forcing a specific adjustment set and
        # DoWhy's estimator-reuse cache are at odds unless the cached estimator
        # was fitted for the identical estimand, which nothing here checks. A
        # cached estimator is fetched from `model._estimator_cache` instead of
        # rebuilt when `fit_estimator=False`, so its `_target_estimand` would be
        # whatever a prior call left behind, not `identified`.
        fit_estimator=True,
        method_params=(
            {"num_simulations": n_bootstraps} if confidence_intervals else {}
        ),
    )

    # `estimate.target_estimand` (public) over reaching into
    # `estimate.estimator._target_estimand` (private): both currently point at
    # the same object for the backdoor path, but the public one is the one
    # DoWhy commits to keeping.
    verify_parent_adjustment_set(estimate, parents)

    return model, identified, estimate, lagged_df


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
