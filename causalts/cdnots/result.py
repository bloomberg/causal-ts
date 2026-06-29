# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Self-contained result object returned by run_cdnots and run_cdnots_plus."""

from __future__ import annotations

import numpy as np

from ..result import CausalResult


class CdnotsResult(CausalResult):
    """Self-contained causal discovery result.

    Returned by :func:`run_cdnots` and :func:`run_cdnots_plus`.  All outputs
    live on a single object — no tuple unpacking required.

    Attributes
    ----------
    graph : CausalGraph
        Raw causal-learn graph (for low-level inspection).
    cg_tig : np.ndarray, shape (d, d, max_lag+1)
        Tigramite-format graph.  ``cg_tig[i, j, tau] == 1`` means
        variable i at lag tau causes variable j at lag 0.
    pvalue_matrix : np.ndarray, shape (d, d, max_lag+1)
        Aggregated p-value per edge from the skeleton phase.
    pvals : list
        Raw ``(effect, cause, condition_set, p_value)`` records.
    var_names : list[str]
        Original column names from the input DataFrame.
    num_lags : int
        Maximum lag used in the time embedding.
    lag_list : list[int] | None
        Explicit lags used (None when num_lags drove the embedding).
    include_C : bool
        Whether the time-index node C was included.
    alpha : float
        Significance level used for skeleton CI tests.
    priority : int
        Collider conflict resolution strategy (0–4).
    stable : bool
        Whether stable skeleton discovery was used.
    lag_verify_stats : dict | None
        Stats from the multi-lag redundancy pass, or None if not run.
    """

    def __init__(
        self,
        graph,
        cg_tig: np.ndarray,
        pvalue_matrix,
        pvals: list,
        var_names: list,
        num_lags: int,
        lag_list,
        include_C: bool,
        alpha: float,
        priority: int,
        stable: bool,
        lag_verify_stats,
        df,
        num_c_cols: int = 1,
        c_node_names=None,
    ):
        self.graph = graph
        self.cg_tig = cg_tig
        self.pvalue_matrix = pvalue_matrix
        self.pvals = pvals
        self.var_names = var_names
        self.num_lags = num_lags
        self.lag_list = lag_list
        self.include_C = include_C
        self.alpha = alpha
        self.priority = priority
        self.stable = stable
        self.lag_verify_stats = lag_verify_stats
        self._df = df
        self.num_c_cols = num_c_cols
        self.c_node_names = c_node_names
        self._scm_cache = {}

    # ------------------------------------------------------------------
    # Background knowledge
    # ------------------------------------------------------------------

    def make_knowledge(
        self,
        forbidden_edges=None,
        required_edges=None,
        required_ancestors=None,
        forbidden_ancestors=None,
    ):
        """Build an :class:`~causalts.cdnots.ancestral.AncestralKnowledge` object.

        Accepts edge constraints (enforced during skeleton discovery) and
        path constraints (enforced during orientation).

        Parameters
        ----------
        forbidden_edges, required_edges : list of (cause, effect, cause_lag, effect_lag)
            Direct edge constraints — same format as :meth:`make_background_knowledge`.
        required_ancestors : list of (ancestor, descendant)
            Pairs where a directed path must exist.
        forbidden_ancestors : list of (ancestor, descendant)
            Pairs where no directed path may exist.

        Returns
        -------
        AncestralKnowledge
        """
        from .ancestral import AncestralKnowledge

        ak = AncestralKnowledge()
        for e in forbidden_edges or []:
            ak.add_forbidden_edge(*e)
        for e in required_edges or []:
            ak.add_required_edge(*e)
        for a, d in required_ancestors or []:
            ak.add_required_ancestor(a, d)
        for a, d in forbidden_ancestors or []:
            ak.add_forbidden_ancestor(a, d)
        return ak

    def make_background_knowledge(self, forbidden=None, required=None):
        """Build a BackgroundKnowledge object using this result's graph settings.

        Equivalent to calling :func:`make_background_knowledge` with
        ``var_names``, ``num_lags``, and ``include_C`` taken from this result.

        Parameters
        ----------
        forbidden : list of (cause, effect, cause_lag, effect_lag), optional
            Edges to forbid.  Each element is a tuple of two variable names
            and two non-negative lag integers.
        required : list of (cause, effect, cause_lag, effect_lag), optional
            Edges to require.

        Returns
        -------
        BackgroundKnowledge
        """
        from .phase3_utils import make_background_knowledge as _mk

        return _mk(
            var_names=self.var_names,
            num_lags=self.num_lags,
            include_C=self.include_C,
            forbidden=forbidden,
            required=required,
        )

    # ------------------------------------------------------------------
    # Plotting (override to handle C-node)
    # ------------------------------------------------------------------

    def plot(self, var_names=None, exclude_C=False, **kwargs):
        """Plot the discovered causal graph.

        Parameters
        ----------
        var_names : list[str] | None
            Override variable names shown on the plot.
        exclude_C : bool
            Strip the C node(s) from the plot.
        """
        from ..plotting._core import plot_graph as _plot_graph

        graph = self.cg_tig
        names = list(var_names or self.var_names)

        if self.include_C:
            if exclude_C:
                graph = graph[: -self.num_c_cols, : -self.num_c_cols, :]
            else:
                if self.c_node_names is not None:
                    names = names + list(self.c_node_names)
                elif self.num_c_cols == 1:
                    names = names + ["C"]
                else:
                    names = names + [f"C{i + 1}" for i in range(self.num_c_cols)]

        if self.lag_list is not None and "lag_array" not in kwargs:
            kwargs["lag_array"] = np.array([0] + list(self.lag_list))
        return _plot_graph(graph=graph, var_names=names, **kwargs)

    def plot_pvalues(self, graph_true, **kwargs):
        """Plot p-value distribution against a ground-truth graph."""
        from ..plotting.compare import plot_pvalue_distribution as _plot_pval

        return _plot_pval(self.pvalue_matrix, graph_true, **kwargs)

    # ------------------------------------------------------------------
    # DoWhy bridge (override to handle C-node exclusion)
    # ------------------------------------------------------------------

    def _bridge(self):
        from ..effects.wrap import wrap_graph

        g = self.cg_tig
        if self.include_C:
            n = len(self.var_names)
            g = g[:n, :n, :]
        return wrap_graph(g, self._df, self.var_names)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self):
        d = len(self.var_names)
        edges = int((self.cg_tig == 1).sum())
        return (
            f"CdnotsResult(vars={d}, num_lags={self.num_lags}, edges={edges},"
            f" alpha={self.alpha}, include_C={self.include_C})"
        )
