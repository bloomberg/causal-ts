# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Graph validation and falsification via DoWhy — no ground truth required."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .graph_bridge import graph_to_networkx, make_lagged_df


def falsify_graph(
    graph: np.ndarray,
    data: pd.DataFrame,
    var_names: list[str] | None = None,
    significance_level: float = 0.05,
    n_permutations: int | None = None,
    suggestions: bool = False,
    include_c: bool = False,
) -> dict:
    """Test whether the discovered DAG is consistent with the data.

    Uses permutation-based comparison: tests if the DAG violates fewer
    local Markov conditions than random node permutations of the same graph.

    No ground truth required.

    Args:
        graph: Shape (d, d, max_lag+1) from CDNOTS/CEDAR/GRACE.
        data: Original (T, d) time series DataFrame.
        var_names: Variable names. Defaults to data.columns.
        significance_level: Threshold for falsification.
        n_permutations: Number of random permutations (None = auto).
        suggestions: If True, include fix suggestions for violations.
        include_c: Whether to include the CDNOTS C node.

    Returns:
        Dict with keys:
            falsifiable: bool — is the graph "characteristic enough" to test?
            falsified: bool — does the graph fail the test?
            summary: str — human-readable summary.
            details: object — raw DoWhy EvaluationResult.
    """
    from ._compat import require_gcm

    require_gcm("graph falsification")
    from dowhy.gcm.falsify import falsify_graph as _falsify

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
    cols = [c for c in lagged_df.columns if c in dag.nodes]
    lagged_df = lagged_df[cols]

    result = _falsify(
        dag,
        lagged_df,
        significance_level=significance_level,
        n_permutations=n_permutations,
        suggestions=suggestions,
        show_progress_bar=False,
        plot_histogram=False,
    )

    return {
        "falsifiable": bool(result.falsifiable),
        "falsified": bool(result.falsified),
        "summary": str(result),
        "details": result,
    }


def refute_structure(
    graph: np.ndarray,
    data: pd.DataFrame,
    var_names: list[str] | None = None,
    significance_level: float = 0.05,
    include_c: bool = False,
) -> pd.DataFrame:
    """Test structural assumptions of the discovered graph.

    For each node, tests: (1) edge dependencies — parents are truly
    dependent on the node, (2) local Markov conditions — node is
    independent of non-descendants given parents.

    No ground truth required.

    Args:
        graph: Shape (d, d, max_lag+1).
        data: Original (T, d) time series DataFrame.
        var_names: Variable names.
        significance_level: p-value threshold.
        include_c: Whether to include the CDNOTS C node.

    Returns:
        DataFrame with columns [node, variable, lag, test_type, rejected, p_value].
    """
    from ._compat import require_gcm

    require_gcm("structural refutation")
    import dowhy.gcm as gcm

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
    cols = [c for c in lagged_df.columns if c in dag.nodes]
    lagged_df = lagged_df[cols]

    from .root_cause import _parse_node

    rejection, details = gcm.refute_causal_structure(
        dag,
        lagged_df,
        significance_level=significance_level,
    )

    rows = []
    for node, tests in details.items():
        variable, lag = _parse_node(node)
        for test_type, test_results in tests.items():
            if isinstance(test_results, dict):
                if "p_value" in test_results:
                    # Direct test result: {p_value, fdr_adjusted_p_value, success}
                    rows.append(
                        {
                            "node": node,
                            "variable": variable,
                            "lag": lag,
                            "test_type": test_type,
                            "rejected": not test_results.get("success", True),
                            "p_value": test_results.get(
                                "fdr_adjusted_p_value",
                                test_results.get("p_value", float("nan")),
                            ),
                        }
                    )
                else:
                    # Nested per-edge results: {edge_name: {p_value, ...}, ...}
                    for sub_key, sub_val in test_results.items():
                        if isinstance(sub_val, dict) and "p_value" in sub_val:
                            rows.append(
                                {
                                    "node": node,
                                    "variable": variable,
                                    "lag": lag,
                                    "test_type": f"{test_type}:{sub_key}",
                                    "rejected": not sub_val.get("success", True),
                                    "p_value": sub_val.get(
                                        "fdr_adjusted_p_value",
                                        sub_val.get("p_value", float("nan")),
                                    ),
                                }
                            )

    if not rows:
        rows.append(
            {
                "node": "overall",
                "variable": "",
                "lag": 0,
                "test_type": "overall",
                "rejected": rejection.value == 1,
                "p_value": float("nan"),
            }
        )

    result = pd.DataFrame(rows)
    return result


def evaluate_model(
    graph: np.ndarray,
    data: pd.DataFrame,
    var_names: list[str] | None = None,
    mechanism_type: str = "auto",
    include_c: bool = False,
) -> dict:
    """Comprehensive model evaluation: mechanism fit, invertibility, KL, falsification.

    Fits an SCM (cached) and evaluates it on multiple criteria.

    No ground truth required.

    Args:
        graph: Shape (d, d, max_lag+1).
        data: Original (T, d) time series DataFrame.
        var_names: Variable names.
        mechanism_type: 'auto', 'linear', or 'gp'.
        include_c: Whether to include the CDNOTS C node.

    Returns:
        Dict with keys: mechanism_scores (per-node), kl_divergence,
        invertibility, overall_summary (str).
    """
    from ._compat import require_gcm

    require_gcm("model evaluation")
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

    eval_result = gcm.evaluate_causal_model(
        scm,
        lagged_df,
        evaluate_causal_mechanisms=True,
        evaluate_invertibility_assumptions=True,
        evaluate_overall_kl_divergence=True,
        evaluate_causal_structure=True,
    )

    return {
        "summary": str(eval_result),
        "details": eval_result,
    }
