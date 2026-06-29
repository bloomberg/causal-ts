# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Wrap a plain 3D graph array with DoWhy bridge methods.

Provides the same cg.estimate_effect / fit_scm / counterfactual /
attribute_anomaly interface for algorithms (like GRACE) that return a
plain NumPy array instead of a causal graph object.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class WrappedGraph:
    """Thin wrapper around a discovered 3D causal graph.

    Exposes the DoWhy bridge as methods, mirroring the interface
    automatically attached to the object returned by cdnots_discovery().

    Attributes:
        cg_tig: The 3D graph array (d, d, max_lag+1).
        df: Original time series DataFrame used for discovery.
        var_names: Variable names (columns of df, C excluded).
    """

    def __init__(
        self,
        graph: np.ndarray,
        df: pd.DataFrame,
        var_names: list[str] | None = None,
    ):
        self.cg_tig = graph
        self._df = df
        self._var_names = list(df.columns) if var_names is None else list(var_names)
        self._scm_cache: dict = {}

    # ------------------------------------------------------------------
    # DoWhy bridge methods — same signatures as on cdnots_discovery output
    # ------------------------------------------------------------------

    def estimate_effect(self, treatment, outcome, treatment_lag=1, **kwargs):
        """Estimate causal effect of treatment(t-lag) on outcome(t).

        Requires DoWhy (pip install dowhy or pip install causalts[dowhy]).
        """
        from .effect import estimate_effect

        return estimate_effect(
            self.cg_tig,
            self._df,
            treatment=treatment,
            outcome=outcome,
            treatment_lag=treatment_lag,
            var_names=self._var_names,
            **kwargs,
        )

    def fit_scm(self, mechanism_type="auto", force=False):
        """Fit a Structural Causal Model to the discovered graph.

        Result is cached — subsequent calls with the same mechanism_type
        return the cached (scm, dag, lagged_df) without refitting.

        Requires DoWhy (pip install dowhy or pip install causalts[dowhy]).
        """
        from .scm import fit_scm

        if not force and mechanism_type in self._scm_cache:
            return self._scm_cache[mechanism_type]
        result = fit_scm(
            self.cg_tig,
            self._df,
            var_names=self._var_names,
            mechanism_type=mechanism_type,
        )
        self._scm_cache[mechanism_type] = result
        return result

    def counterfactual(self, intervention, target, mechanism_type="auto"):
        """Estimate counterfactual values for target under do(intervention).

        Fits the SCM automatically if not already cached.

        Parameters
        ----------
        intervention : dict
            Dict of {node: value}, e.g. {"X0_lag1": 0.0}.
        target : str
            Current-time variable to query (e.g. "X2").
        mechanism_type : str
            Passed to fit_scm if not already cached.

        Notes
        -----
        Requires DoWhy (pip install dowhy or pip install causalts[dowhy]).
        """
        from .scm import counterfactual

        scm, _, lagged_df = self.fit_scm(mechanism_type=mechanism_type)
        return counterfactual(scm, lagged_df, intervention=intervention, target=target)

    def attribute_anomaly(self, anomaly, target, mechanism_type="auto", n_samples=2000):
        """Attribute an anomaly in target to its upstream causes.

        Fits the SCM automatically if not already cached.

        Parameters
        ----------
        anomaly : pd.Series or pd.DataFrame
            Anomalous observation(s).
        target : str
            Variable to explain (e.g. "X2").
        mechanism_type : str
            Passed to fit_scm if not already cached.
        n_samples : int
            Monte Carlo samples for Shapley attribution.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns [node, variable, lag,
            mean_attribution].

        Notes
        -----
        Requires DoWhy (pip install dowhy or pip install causalts[dowhy]).
        """
        from .root_cause import attribute_anomaly

        scm, _, lagged_df = self.fit_scm(mechanism_type=mechanism_type)
        return attribute_anomaly(
            scm, lagged_df, anomaly=anomaly, target=target, n_samples=n_samples
        )

    # ------------------------------------------------------------------
    # Phase 2: Validation, influence, distribution change
    # ------------------------------------------------------------------

    def falsify(self, **kwargs) -> dict:
        """Test if the graph is consistent with the data's independence structure.

        No ground truth required. See validate.falsify_graph for details.
        """
        from .validate import falsify_graph

        return falsify_graph(self.cg_tig, self._df, var_names=self._var_names, **kwargs)

    def refute_structure(self, **kwargs) -> pd.DataFrame:
        """Test structural assumptions (edge deps + local Markov conditions).

        No ground truth required. See validate.refute_structure for details.
        """
        from .validate import refute_structure

        return refute_structure(
            self.cg_tig, self._df, var_names=self._var_names, **kwargs
        )

    def evaluate_model(self, mechanism_type="auto", **kwargs) -> dict:
        """Comprehensive evaluation: mechanism fit, invertibility, KL, falsification.

        No ground truth required. See validate.evaluate_model for details.
        """
        from .validate import evaluate_model

        return evaluate_model(
            self.cg_tig,
            self._df,
            var_names=self._var_names,
            mechanism_type=mechanism_type,
            **kwargs,
        )

    def arrow_strength(self, target, **kwargs) -> pd.DataFrame:
        """Quantify how strongly each parent influences a target.

        See influence.arrow_strength for details.
        """
        from .influence import arrow_strength

        return arrow_strength(
            self.cg_tig, self._df, target, var_names=self._var_names, **kwargs
        )

    def causal_influence(self, target, **kwargs) -> pd.DataFrame:
        """Shapley decomposition of target variability by upstream noise sources.

        See influence.causal_influence for details.
        """
        from .influence import causal_influence

        return causal_influence(
            self.cg_tig, self._df, target, var_names=self._var_names, **kwargs
        )

    def parent_relevance(self, target, **kwargs) -> pd.DataFrame:
        """Rank direct parents by explanatory power.

        See influence.parent_relevance for details.
        """
        from .influence import parent_relevance

        return parent_relevance(
            self.cg_tig, self._df, target, var_names=self._var_names, **kwargs
        )

    def distribution_change(self, data_new, target, **kwargs) -> pd.DataFrame:
        """Identify which mechanisms changed between original and new data.

        See influence.distribution_change for details.
        """
        from .influence import distribution_change

        return distribution_change(
            self.cg_tig,
            self._df,
            data_new,
            target,
            var_names=self._var_names,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Probabilistic queries
    # ------------------------------------------------------------------

    def interventional_query(
        self,
        intervention: dict,
        targets=None,
        n_samples: int = 2000,
        mechanism_type: str = "auto",
    ):
        """Sample from the post-intervention distribution do(X=x).

        Answers "what if we SET variable=value?" by severing incoming
        edges and generating samples from the modified SCM.

        Parameters
        ----------
        intervention : dict
            Dict of {variable: value}, e.g. {"Mek": 5.0}.
            Names should match var_names (current-time nodes).
        targets : str, list[str], or None
            Variable(s) to return. None returns all non-intervened.
        n_samples : int
            Number of Monte Carlo samples to draw.
        mechanism_type : str
            Passed to fit_scm if not already cached.

        Returns
        -------
        pd.DataFrame
            DataFrame of samples from the interventional distribution.
        """
        from .query import interventional_query

        scm, _, _ = self.fit_scm(mechanism_type=mechanism_type)
        return interventional_query(
            scm, intervention, targets=targets, n_samples=n_samples
        )

    def observational_query(
        self,
        evidence: dict,
        targets=None,
        n_samples: int = 5000,
        tolerance: float | None = None,
        kernel_bandwidth: float | None = None,
        mechanism_type: str = "auto",
    ):
        """Sample from the conditional distribution P(targets | evidence).

        Answers "given that we OBSERVE variable~=value, what do we expect?"
        without severing any edges. Uses importance-weighted sampling.

        Parameters
        ----------
        evidence : dict
            Dict of {variable: value}, e.g. {"Mek": 5.0, "PKA": 2.0}.
        targets : str, list[str], or None
            Variable(s) to return. None returns all non-evidence nodes.
        n_samples : int
            Raw samples before filtering/weighting.
        tolerance : float or None
            If set, uses rejection sampling (hard conditioning).
        kernel_bandwidth : float or None
            Bandwidth for soft Gaussian kernel (default auto).
        mechanism_type : str
            Passed to fit_scm if not already cached.

        Returns
        -------
        pd.DataFrame
            DataFrame of (weighted) samples. Includes '_weight' column
            for soft conditioning.
        """
        from .query import observational_query

        scm, _, lagged_df = self.fit_scm(mechanism_type=mechanism_type)
        return observational_query(
            scm,
            lagged_df,
            evidence,
            targets=targets,
            n_samples=n_samples,
            tolerance=tolerance,
            kernel_bandwidth=kernel_bandwidth,
        )

    def query_summary(self, result, percentiles=None):
        """Summarize interventional/observational query results.

        Parameters
        ----------
        result : pd.DataFrame
            DataFrame from interventional_query or observational_query.
        percentiles : list[float] or None
            Quantiles to report. Default [5, 25, 50, 75, 95]%.

        Returns
        -------
        pd.DataFrame
            DataFrame with mean, std, and percentiles per target
            variable.
        """
        from .query import query_summary

        return query_summary(result, percentiles=percentiles)

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def forecast(
        self, model=None, include_contemp=True, targets=None, exog_fallback=True
    ):
        """Fit a graph-informed forecasting model.

        Uses the discovered causal graph to select relevant lagged
        predictors for each target, then fits an sklearn regressor.
        Exogenous variables (no causal parents) get an auto-ETS fallback.

        Parameters
        ----------
        model : object or None
            sklearn-compatible regressor. Defaults to LinearRegression.
        include_contemp : bool
            Include contemporaneous parents (lag=0) as features.
            Set False for multi-step ahead forecasting.
        targets : list[str] or None
            Variables to forecast. Defaults to all with parents.
        exog_fallback : bool
            Fit auto-ETS for parentless variables (default True).

        Returns
        -------
        CausalForecaster
            Fitted CausalForecaster instance.
        """
        from .forecast import CausalForecaster

        forecaster = CausalForecaster(
            self.cg_tig,
            var_names=self._var_names,
            model=model,
            include_contemp=include_contemp,
            exog_fallback=exog_fallback,
        )
        forecaster.fit(self._df, targets=targets)
        return forecaster

    def summary(self, mechanism_type="linear", top_k=5):
        """Print a one-page diagnostic report of the discovered graph.

        Runs falsification, arrow strength for the top-connected nodes,
        and basic graph statistics. Requires DoWhy.

        Parameters
        ----------
        mechanism_type : str
            Passed to fit_scm for arrow strength.
        top_k : int
            Number of strongest edges to show.

        Returns
        -------
        dict
            Dict with all computed metrics (for programmatic access).
        """
        g = self.cg_tig
        d = g.shape[0]
        max_lag = g.shape[2] - 1
        n_edges = int((g == 1).sum())
        n_self = sum(int(g[i, i, :].sum()) for i in range(d))
        n_cross = n_edges - n_self

        print("=" * 60)
        print("CAUSAL GRAPH SUMMARY")
        print("=" * 60)
        print(f"Variables:    {d} ({', '.join(self._var_names)})")
        print(f"Max lag:      {max_lag}")
        print(
            f"Total edges:  {n_edges} ({n_self} self-loops, {n_cross} cross-variable)"
        )
        print(f"Data:         T={len(self._df)}")

        # Edge list
        print("\\nDiscovered edges:")
        for i in range(d):
            for j in range(d):
                for lag in range(max_lag + 1):
                    if g[i, j, lag] == 1:
                        print(
                            f"  {self._var_names[i]}(t-{lag}) -> {self._var_names[j]}"
                        )

        result = {"n_edges": n_edges, "n_self": n_self, "n_cross": n_cross}

        # Falsification
        try:
            fals = self.falsify()
            print("\nGraph falsification:")
            print(f"  Falsifiable: {fals['falsifiable']}")
            print(f"  Falsified:   {fals['falsified']}")
            if fals["falsifiable"] and not fals["falsified"]:
                print("  -> Graph is consistent with data.")
            elif fals["falsified"]:
                print("  -> WARNING: graph may contain errors.")
            result["falsification"] = fals
        except Exception as e:
            print(f"\nGraph falsification: skipped ({e})")

        # Top edges by arrow strength (pick the node with most parents)
        try:
            parent_counts = {}
            for j in range(d):
                n_parents = int(g[:, j, :].sum())
                if n_parents > 0:
                    parent_counts[self._var_names[j]] = n_parents
            if parent_counts:
                top_node = max(parent_counts, key=parent_counts.get)
                strengths = self.arrow_strength(top_node, mechanism_type=mechanism_type)
                print(
                    f"\nArrow strength (-> {top_node}, {parent_counts[top_node]} parents):"
                )
                for _, row in strengths.head(top_k).iterrows():
                    print(f"  {row['parent_node']:<16} {row['strength']:.4f}")
                result["arrow_strength_target"] = top_node
                result["arrow_strength"] = strengths
        except Exception as e:
            print(f"\nArrow strength: skipped ({e})")

        print("=" * 60)
        return result

    def __repr__(self):
        n_edges = int((self.cg_tig == 1).sum())
        return (
            f"WrappedGraph(vars={self._var_names}, "
            f"shape={self.cg_tig.shape}, edges={n_edges})"
        )


def wrap_graph(
    graph: np.ndarray,
    df: pd.DataFrame,
    var_names: list[str] | None = None,
) -> WrappedGraph:
    """Wrap a plain 3D causal graph array with DoWhy bridge methods.

    Use this to attach DoWhy bridge methods to a plain ndarray result.
    Note: ``run_cdnots_gated`` and ``run_stability_selection`` now return
    ``GraceResult`` which already inherits these methods, so ``wrap_graph``
    is only needed for third-party algorithms that return raw arrays::

        res = run_cdnots_gated(df, max_lag=3)
        res.estimate_effect("X0", "X2", treatment_lag=1)  # direct — no wrap needed

        # For raw arrays from custom algorithms:
        cg = wrap_graph(my_graph_array, df)
        cg.estimate_effect("X0", "X2", treatment_lag=1)
        cg.attribute_anomaly(anomaly_row, target="X2")

    Parameters
    ----------
    graph : np.ndarray
        Shape (d, d, max_lag+1) -- the discovered causal graph.
    df : pd.DataFrame
        Original (T, d) time series DataFrame used for discovery.
    var_names : list[str] or None
        Column names. Defaults to df.columns.

    Returns
    -------
    WrappedGraph
        WrappedGraph with estimate_effect / fit_scm / counterfactual /
        attribute_anomaly methods.
    """
    return WrappedGraph(graph, df, var_names=var_names)
