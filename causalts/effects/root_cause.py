# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Root cause / anomaly attribution via DoWhy GCM."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def attribute_anomaly(
    scm,
    lagged_df: pd.DataFrame,
    anomaly: pd.Series | pd.DataFrame,
    target: str,
    n_samples: int = 2000,
) -> pd.DataFrame:
    """Attribute an anomaly in target to upstream variables.

    Given a fitted SCM and one or more anomalous observations, computes
    how much each upstream variable contributed to the anomaly in the
    target variable using Shapley-based attribution.

    Parameters
    ----------
    scm : InvertibleStructuralCausalModel
        Fitted InvertibleStructuralCausalModel from fit_scm().
    lagged_df : pd.DataFrame
        Normal (training) lag-embedded DataFrame from fit_scm().
    anomaly : pd.Series or pd.DataFrame
        One or more anomalous observations as a Series (single row)
        or DataFrame (multiple rows). Must have the same columns as
        lagged_df.
    target : str
        Current-time variable to explain (e.g. "X2").
    n_samples : int
        Monte Carlo samples for the anomaly scorer.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [node, variable, lag, mean_attribution]
        sorted by absolute mean_attribution descending. Each row is one
        upstream node in the causal graph.
    """
    from ._compat import require_gcm

    require_gcm("root cause analysis")
    import dowhy.gcm as gcm

    if isinstance(anomaly, pd.Series):
        anomaly_df = anomaly.to_frame().T.reset_index(drop=True)
    else:
        anomaly_df = anomaly.reset_index(drop=True)

    # Only pass columns that exist in the SCM
    scm_nodes = set(scm.graph.nodes)
    cols = [c for c in anomaly_df.columns if c in scm_nodes]
    anomaly_df = anomaly_df[cols]

    attributions = gcm.attribute_anomalies(
        scm,
        target_node=target,
        anomaly_samples=anomaly_df,
        num_distribution_samples=n_samples,
    )

    # Build output DataFrame
    rows = []
    for node, scores in attributions.items():
        mean_score = float(np.mean(scores))
        variable, lag = _parse_node(node)
        rows.append(
            {
                "node": node,
                "variable": variable,
                "lag": lag,
                "mean_attribution": mean_score,
            }
        )

    result = pd.DataFrame(rows)
    result = result.sort_values("mean_attribution", key=abs, ascending=False)
    return result.reset_index(drop=True)


def _parse_node(node: str) -> tuple[str, int]:
    """Parse a node name like 'X0_lag2' into ('X0', 2)."""
    m = re.search(r"_lag(\d+)$", node)
    if m:
        lag = int(m.group(1))
        variable = node[: -len(m.group(0))]
    else:
        variable = node
        lag = 0
    return variable, lag
