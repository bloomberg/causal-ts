# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Probabilistic queries on fitted SCMs.

Supports two query modes:
- Interventional: "What if we SET Mek=5?" (do-operator, causal effect)
- Observational:  "Given we OBSERVE Mek≈5, what's the distribution?" (conditioning)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def interventional_query(
    scm,
    intervention: dict[str, float],
    targets: str | list[str] | None = None,
    n_samples: int = 2000,
) -> pd.DataFrame:
    """Sample from the post-intervention distribution do(X=x).

    Generates samples from the SCM after hard-setting one or more variables
    to fixed values (the do-operator). Unlike observational conditioning,
    this severs incoming edges to the intervened nodes.

    Parameters
    ----------
    scm : InvertibleStructuralCausalModel
        Fitted InvertibleStructuralCausalModel from fit_scm().
    intervention : dict[str, float]
        Dict mapping node names to intervened values.
        E.g. {"Mek": 5.0, "PKA": 2.0}.
    targets : str, list[str], or None
        Variable(s) to return. None returns all non-intervened nodes.
    n_samples : int
        Number of Monte Carlo samples to draw.

    Returns
    -------
    pd.DataFrame
        DataFrame with n_samples rows and one column per target variable.
    """
    from ._compat import require_gcm

    require_gcm("interventional query")
    import dowhy.gcm as gcm

    interventions = {k: (lambda x, v=v: v) for k, v in intervention.items()}

    samples = gcm.interventional_samples(
        scm, interventions, num_samples_to_draw=n_samples
    )

    if targets is not None:
        if isinstance(targets, str):
            targets = [targets]
        samples = samples[targets]
    else:
        samples = samples.drop(columns=list(intervention.keys()), errors="ignore")

    return samples


def observational_query(
    scm,
    lagged_df: pd.DataFrame,
    evidence: dict[str, float],
    targets: str | list[str] | None = None,
    n_samples: int = 5000,
    tolerance: float | None = None,
    kernel_bandwidth: float | None = None,
) -> pd.DataFrame:
    """Sample from the conditional distribution P(targets | evidence).

    Unlike interventional queries, this does NOT sever incoming edges.
    It answers "given that we observe X~=x, what do we expect for Y?"

    Uses importance-weighted sampling: draws from the generative model and
    weights samples by proximity to the evidence. Supports both hard
    conditioning (rejection within tolerance) and soft conditioning
    (Gaussian kernel weighting).

    Parameters
    ----------
    scm : InvertibleStructuralCausalModel
        Fitted InvertibleStructuralCausalModel from fit_scm().
    lagged_df : pd.DataFrame
        Lag-embedded DataFrame from fit_scm() (used for
        empirical scaling of tolerances).
    evidence : dict[str, float]
        Dict mapping node names to observed values.
        E.g. {"Mek": 5.0, "PKA": 2.0}.
    targets : str, list[str], or None
        Variable(s) to return. None returns all non-evidence nodes.
    n_samples : int
        Number of raw samples to draw before filtering/weighting.
        More samples = better approximation. Default 5000.
    tolerance : float or None
        For hard conditioning (rejection sampling). Samples are
        kept if all evidence variables are within +/-tolerance
        (in standardized units). Default None uses soft conditioning.
    kernel_bandwidth : float or None
        Bandwidth for Gaussian kernel weights (in standardized
        units). Default None auto-selects based on data spread
        (0.5 std).

    Returns
    -------
    pd.DataFrame
        DataFrame of (weighted) samples for the target variables.
        For soft conditioning, a '_weight' column is included for
        downstream use.
    """
    from ._compat import require_gcm

    require_gcm("observational query")
    import dowhy.gcm as gcm

    samples = gcm.draw_samples(scm, n_samples)

    # Standardize evidence variables for distance computation
    stds = {}
    for node in evidence:
        if node in lagged_df.columns:
            stds[node] = lagged_df[node].std()
        else:
            stds[node] = samples[node].std()
        if stds[node] == 0 or np.isnan(stds[node]):
            stds[node] = 1.0

    if targets is not None:
        if isinstance(targets, str):
            targets = [targets]
    else:
        targets = [c for c in samples.columns if c not in evidence]

    if tolerance is not None:
        # Hard conditioning: rejection sampling
        mask = np.ones(len(samples), dtype=bool)
        for node, value in evidence.items():
            dist = np.abs(samples[node].values - value) / stds[node]
            mask &= dist <= tolerance
        result = samples.loc[mask, targets].reset_index(drop=True)
    else:
        # Soft conditioning: Gaussian kernel weighting
        if kernel_bandwidth is None:
            kernel_bandwidth = 0.5

        log_weights = np.zeros(len(samples))
        for node, value in evidence.items():
            z = (samples[node].values - value) / stds[node]
            log_weights += -0.5 * (z / kernel_bandwidth) ** 2

        weights = np.exp(log_weights - log_weights.max())
        weights /= weights.sum()

        result = samples[targets].copy()
        result["_weight"] = weights

    return result


def query_summary(
    result: pd.DataFrame,
    percentiles: list[float] | None = None,
) -> pd.DataFrame:
    """Summarize query results into mean, std, and percentiles.

    Handles both weighted (soft conditioning) and unweighted (interventional
    or hard conditioning) results.

    Parameters
    ----------
    result : pd.DataFrame
        DataFrame from interventional_query or observational_query.
    percentiles : list[float] or None
        Quantiles to report. Default [0.05, 0.25, 0.5, 0.75, 0.95].

    Returns
    -------
    pd.DataFrame
        DataFrame with rows = target variables, columns = summary
        statistics.
    """
    if percentiles is None:
        percentiles = [0.05, 0.25, 0.5, 0.75, 0.95]

    weighted = "_weight" in result.columns
    targets = [c for c in result.columns if c != "_weight"]

    rows = []
    for col in targets:
        values = result[col].values
        if weighted:
            weights = result["_weight"].values
            mean = np.average(values, weights=weights)
            var = np.average((values - mean) ** 2, weights=weights)
            std = np.sqrt(var)
            # Weighted percentiles via sorted cumulative weights
            order = np.argsort(values)
            sorted_vals = values[order]
            cum_w = np.cumsum(weights[order])
            cum_w /= cum_w[-1]
            pcts = [sorted_vals[np.searchsorted(cum_w, p)] for p in percentiles]
        else:
            mean = values.mean()
            std = values.std()
            pcts = [np.percentile(values, p * 100) for p in percentiles]

        row = {"variable": col, "mean": mean, "std": std}
        for p, v in zip(percentiles, pcts):
            row[f"p{int(p*100):02d}"] = v  # noqa: E226
        rows.append(row)

    return pd.DataFrame(rows).set_index("variable")
