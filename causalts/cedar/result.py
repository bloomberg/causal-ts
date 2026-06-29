# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""CEDAR result object."""

from __future__ import annotations

import numpy as np

from ..result import CausalResult


class CedarResult(CausalResult):
    """Result from CEDAR causal discovery.

    Inherits plotting and DoWhy bridge methods from :class:`CausalResult`.

    Attributes
    ----------
    cg_tig : np.ndarray, shape (d, d, max_lag+1)
        Binary adjacency. ``cg_tig[cause, effect, lag] == 1`` means
        variable *cause* at lag *tau* causes variable *effect*.
    val_matrix : np.ndarray, shape (d, d, max_lag+1)
        CI test statistic (strength) per edge.
    pvalue_matrix : np.ndarray, shape (d, d, max_lag+1)
        P-value from Cond1 test for each candidate edge.
    lag_importance : np.ndarray, shape (d, d, max_lag+1)
        Distance correlation scores used for lag selection.
    lag_pvalues : np.ndarray, shape (d, d, max_lag+1)
        Permutation-based p-values for lag significance.
    selected_lags : np.ndarray, shape (d, d)
        Selected minimum lag per (cause, effect) pair. NaN if none.
    var_names : list[str]
        Variable names.
    max_lag : int
        Maximum lag considered.
    alpha_cond1 : float
        Significance threshold for Condition 1 (dependence).
    alpha_cond2 : float
        Significance threshold for Condition 2 (independence).
    n_tests : int
        Total CI tests performed.
    pruned : bool
        Whether pruning was applied.
    """

    def __init__(
        self,
        graph: np.ndarray,
        val_matrix: np.ndarray,
        pvalue_matrix: np.ndarray,
        lag_importance: np.ndarray,
        lag_pvalues: np.ndarray,
        selected_lags: np.ndarray,
        var_names: list[str],
        max_lag: int,
        alpha_cond1: float,
        alpha_cond2: float,
        n_tests: int,
        pruned: bool,
        df,
        lag_method: str = "dcor",
        lag_pvalue_method: str = "t_test",
        forbidden: list | None = None,
        required: list | None = None,
        c_node_names: list | None = None,
        ar_violations: dict | None = None,
        ar_order: dict | None = None,
    ):
        self.cg_tig = graph
        self.val_matrix = val_matrix
        self.pvalue_matrix = pvalue_matrix
        self.lag_importance = lag_importance
        self.lag_pvalues = lag_pvalues
        self.selected_lags = selected_lags
        self.var_names = list(var_names)
        self.max_lag = max_lag
        self.alpha_cond1 = alpha_cond1
        self.alpha_cond2 = alpha_cond2
        self.n_tests = n_tests
        self.pruned = pruned
        self.lag_method = lag_method
        self.lag_pvalue_method = lag_pvalue_method
        self.forbidden = list(forbidden or [])
        self.required = list(required or [])
        self.c_node_names = list(c_node_names or [])
        self.ar_violations = ar_violations or {}
        self.ar_order = ar_order or {}
        self._df = df
        self._scm_cache = {}

    def _bridge(self):
        """DoWhy bridge with implicit AR(1) self-lags restored.

        CEDAR excludes Xi(t-1)->Xi from discovery by design (it conditions on
        them but never tests them). Without these edges DoWhy's backdoor
        criterion omits a key confounder. We restore lag-1 self-edges here.

        C columns are stripped before passing to DoWhy — C has no meaningful
        backdoor path and its deterministic structure breaks DoWhy's graph.
        """
        from ..effects.wrap import wrap_graph

        n_c = len(self.c_node_names)
        d_x = len(self.var_names) - n_c
        g = self.cg_tig[:d_x, :d_x, :].copy()
        if g.shape[2] > 1:
            np.fill_diagonal(g[:, :, 1], 1)
        x_var_names = self.var_names[:d_x]
        return wrap_graph(g, self._df, x_var_names)

    def plot(self, val_matrix="default", **kwargs):
        """Plot the discovered causal graph.

        Parameters
        ----------
        val_matrix : ndarray or None, optional
            Edge-strength matrix passed to the plotting layer.  Pass
            ``None`` to suppress edge-width/color encoding (plain graph).
            Defaults to ``self.val_matrix``.
        """
        kwargs.setdefault("show_colorbar", False)
        kwargs.setdefault("link_colorbar_label", "strength")
        kwargs.setdefault("node_colorbar_label", "auto-dep")
        if val_matrix == "default":
            vm = self.val_matrix
            if self.c_node_names and vm is not None:
                # Lag-0 val_matrix is asymmetric when C is present (C→X
                # values, X→C zeros).  The plotting layer requires lag-0
                # symmetry, so zero it out here.
                vm = vm.copy()
                vm[:, :, 0] = 0.0
        else:
            vm = val_matrix
        return super().plot(val_matrix=vm, **kwargs)

    def plot_lag_importance(
        self,
        figsize=(10, 10),
        maximum=1.0,
        minimum=0.0,
        show_gridlines=True,
        x_base=1,
        y_base=0.2,
        fig_name=None,
        add_lagfunc_args=None,
    ):
        """Plot lag importance (dcor / pearson scores) as a matrix of bar charts.

        Each cell (i, j) shows the importance of variable i at each lag for
        predicting variable j. Wraps :func:`causalts.plotting.plot_lagfuncs`.

        Parameters
        ----------
        figsize : tuple, default (10, 10)
        maximum : float, default 1.0
            Y-axis upper bound.
        minimum : float, default 0.0
            Y-axis lower bound.
        show_gridlines : bool, default True
        x_base : int, default 1
            X-tick spacing (in lags).
        y_base : float, default 0.2
            Y-tick spacing.
        fig_name : str or None
            If given, save figure to this path instead of showing.
        add_lagfunc_args : dict or None
            Extra kwargs forwarded to ``matrix.add_lagfuncs``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from ..plotting._core import plot_lagfuncs

        matrix = plot_lagfuncs(
            val_matrix=self.lag_importance,
            name=fig_name,
            setup_args={
                "var_names": self.var_names,
                "x_base": x_base,
                "y_base": y_base,
                "minimum": minimum,
                "maximum": maximum,
                "plot_gridlines": show_gridlines,
                "figsize": figsize,
            },
            add_lagfunc_args=add_lagfunc_args or {},
        )
        return matrix.fig

    def make_knowledge(
        self,
        forbidden_edges=None,
        required_edges=None,
        required_ancestors=None,
        forbidden_ancestors=None,
    ):
        """Build an :class:`~causalts.cdnots.ancestral.AncestralKnowledge` object.

        Edge constraints are in Cedar's native 3-tuple or 4-tuple format.
        Path constraints are (ancestor, descendant) name pairs.

        Returns
        -------
        AncestralKnowledge
        """
        from ..cdnots.ancestral import AncestralKnowledge

        ak = AncestralKnowledge()
        for e in forbidden_edges or []:
            ak.add_forbidden_edge(*e[:3], effect_lag=e[3] if len(e) > 3 else 0)
        for e in required_edges or []:
            ak.add_required_edge(*e[:3], effect_lag=e[3] if len(e) > 3 else 0)
        for a, d in required_ancestors or []:
            ak.add_required_ancestor(a, d)
        for a, d in forbidden_ancestors or []:
            ak.add_forbidden_ancestor(a, d)
        return ak

    def make_background_knowledge(self, forbidden=None, required=None):
        """Normalize edge constraints for re-use with run_cedar.

        Accepts 3-tuples ``(cause, effect, lag)`` or 4-tuples
        ``(cause, effect, cause_lag, effect_lag)`` — the 4th element is dropped
        so users can pass the same edge lists built for CDNOTS.

        Returns a :class:`types.SimpleNamespace` with ``.forbidden`` and
        ``.required`` attributes (lists of 3-tuples) ready for::

            bk = result.make_background_knowledge(forbidden=fps, required=fns)
            run_cedar(..., forbidden=bk.forbidden, required=bk.required)
        """
        from types import SimpleNamespace

        def _norm(edges):
            return [tuple(e[:3]) for e in (edges or [])]

        return SimpleNamespace(forbidden=_norm(forbidden), required=_norm(required))

    # backward-compat alias
    plot_lag_dependancy = plot_lag_importance

    def __repr__(self):
        d = len(self.var_names)
        edges = int((self.cg_tig == 1).sum())
        extras = []
        if self.c_node_names:
            extras.append(f"C={self.c_node_names}")
        if self.ar_violations:
            extras.append(f"AR(2+)={list(self.ar_violations.keys())}")
        if self.forbidden or self.required:
            extras.append(
                f"forbidden={len(self.forbidden)}, required={len(self.required)}"
            )
        extra_str = (", " + ", ".join(extras)) if extras else ""
        return (
            f"CedarResult(vars={d}, max_lag={self.max_lag}, edges={edges}, "
            f"lag_method={self.lag_method!r}, lag_pvalue_method={self.lag_pvalue_method!r}, "
            f"alpha_cond1={self.alpha_cond1}, alpha_cond2={self.alpha_cond2}, "
            f"pruned={self.pruned}, n_tests={self.n_tests}{extra_str})"
        )
