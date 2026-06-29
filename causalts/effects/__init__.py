# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""DoWhy integration for causal-ts.

Provides effect estimation, SCM fitting, counterfactual analysis, and root
cause attribution on top of causal graphs discovered by CDNOTS/CEDAR/GRACE.

DoWhy must be installed separately:
    pip install dowhy>=0.11

Typical workflow::

    from causalts import run_cdnots
    from causalts.effects import fit_scm, estimate_effect, attribute_anomaly

    res = run_cdnots(df, ci_test, num_lags=3)
    graph = res.cg_tig

    # Effect estimation
    est = estimate_effect(graph, df, treatment="X0", outcome="X2", treatment_lag=1)
    print(est.value)

    # SCM fitting + counterfactuals
    scm, dag, lagged_df = fit_scm(graph, df)
    cf = counterfactual(scm, lagged_df, intervention={"X0_lag1": 0.0}, target="X2")

    # Root cause analysis
    anomaly_row = lagged_df.iloc[-1]
    attrs = attribute_anomaly(scm, lagged_df.iloc[:-1], anomaly_row, target="X2")
    print(attrs.head())
"""

from ._compat import dowhy_available, require_dowhy, require_gcm
from .effect import estimate_effect, print_effect_summary
from .forecast import CausalForecaster, get_causal_parents
from .graph_bridge import graph_to_networkx, make_lagged_df, networkx_to_graph
from .influence import (
    arrow_strength,
    causal_influence,
    distribution_change,
    parent_relevance,
)
from .query import interventional_query, observational_query, query_summary
from .root_cause import attribute_anomaly
from .scm import counterfactual, fit_scm
from .tigramite_effects import TigramiteEffects, pathwise_effects, tigramite_available
from .validate import evaluate_model, falsify_graph, refute_structure
from .wrap import WrappedGraph, wrap_graph

__all__ = [
    "graph_to_networkx",
    "make_lagged_df",
    "networkx_to_graph",
    "fit_scm",
    "counterfactual",
    "estimate_effect",
    "print_effect_summary",
    "attribute_anomaly",
    "falsify_graph",
    "refute_structure",
    "evaluate_model",
    "arrow_strength",
    "causal_influence",
    "parent_relevance",
    "distribution_change",
    "interventional_query",
    "observational_query",
    "query_summary",
    "wrap_graph",
    "WrappedGraph",
    "CausalForecaster",
    "get_causal_parents",
    "dowhy_available",
    "require_dowhy",
    "require_gcm",
    "TigramiteEffects",
    "pathwise_effects",
    "tigramite_available",
]
