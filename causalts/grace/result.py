# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Result object returned by GRACE discovery functions."""

from __future__ import annotations

import numpy as np

from ..result import CausalResult


class GraceResult(CausalResult):
    """Result from :func:`run_grace`, :func:`run_reap`, :func:`run_cdnots_gated` or
    :func:`run_stability_selection`.

    Inherits plotting and DoWhy bridge methods from :class:`CausalResult`.

    Attributes
    ----------
    cg_tig : np.ndarray, shape (d, d, max_lag+1)
        Binary adjacency. ``cg_tig[cause, effect, lag] == 1`` means
        cause(t-lag) → effect(t).
    var_names : list[str]
        Variable names.
    gate_values : np.ndarray or None
        Continuous Hard Concrete gate values in [0, 1], same shape as
        ``cg_tig``. Available from :func:`run_grace` (``run_cdnots_gated``).
    stability_scores : np.ndarray or None
        Edge stability frequencies in [0, 1], same shape as ``cg_tig``.
        Available from :func:`run_stability_selection`.
    candidate_graph : np.ndarray or None
        Directed candidate graph the gates (GRACE) or the pruning (REAP) refined:
        the CDNOTS or supplied candidate graph plus the edges grow rounds added,
        shape (d, d, max_lag+1); (d + n_C, d + n_C, max_lag+1) under GRACE with
        ``include_C_in_model=True``, the C columns last. ``skeleton`` is the same
        array (its earlier name).
    lambda_l0 : float or None
        L0 regularisation strength used during training.
    gate_threshold : float or None
        Gate threshold applied to produce the binary graph (the resolved
        number, also when the cutoff was chosen from the data). With
        ``effect_floor > 0`` the graph is then pruned further (``effect_pruned``),
        so a lagged edge can have a gate above the threshold and still be absent.
    gate_threshold_rule : str or None
        How the cutoff was chosen: ``'fixed'``, ``'li'``, or
        ``'li_fallback'`` (Li requested but too few gates; 0.5 used), or
        ``'auto_unimodal'`` (supplied skeleton whose gates form one cluster;
        0.5 used). With ``ar_path=True``: ``'ar_li_floor'`` (max of Li and 0.5),
        ``'ar_li_fallback'`` (too few gates for Li; 0.5 used),
        ``'ar_auto_unimodal'`` (supplied skeleton whose cross gates form one
        cluster; 0.5 used) or ``'ar_fixed'``.
    grow_history : list[dict] or None
        Per grow round (``grow_rounds > 0``): ``n_tests``, ``n_added`` and,
        if the round refined again (gates retrained or shrink re-run),
        ``n_kept`` (grown edges that survived that refinement, counted before
        the final effect-floor pruning; see ``effect_pruned``). With ``grow_variance=True``
        also ``n_tests_variance`` and ``n_added_variance`` (the variance tests,
        a separate Benjamini-Hochberg family; ``n_added`` counts both).
    ar_path : bool or None
        Whether :func:`run_cdnots_gated` used the autoregressive head (after
        resolving its ``None`` default); None for other producers.
    lambda_rule : str or None
        The closed-form lambda rule :func:`run_cdnots_gated` was given (gates
        only; the rule is not used when ``lambda_cv`` or ``lambda_l0`` is set).
    effect_floor : float, str or None
        Minimum effect of an edge (adjusted partial R^2) applied by
        :func:`run_cdnots_gated`: with ``refine='gates'`` the ``effect_floor``
        (0 when off); with ``refine='shrink'`` the ``shrink_effect_floor``
        (``'adaptive'``, a float, or None when off).
    n_effect_pruned : int or None
        Edges removed by the final effect-floor pruning.
    effect_pruned : np.ndarray or None
        Boolean, same shape as ``cg_tig``: edges the gates (or the shrink) kept that the final
        effect-floor pruning removed (their ``gate_values``, or under
        ``refine='shrink'`` their ``edge_pvalue`` and ``edge_strength``, are
        unchanged; zero them to match ``cg_tig``). None
        when nothing was pruned-tested (``effect_floor=0``, ``refine="shrink"``
        without ``shrink_effect_floor``, or data with missing values).
    runtime : float or None
        Total wall-clock time in seconds.
    refine : str or None
        Refinement used: ``'gates'`` (GRACE, neural gated model) or ``'shrink'``
        (REAP, regression-based backward elimination).
    edge_pvalue : np.ndarray or None
        ``refine='shrink'`` only, same shape as ``cg_tig``: for a kept edge, the
        approximate (fixed-basis F) p-value of its parent's feature block given the
        target's other kept
        parents; for a removed candidate, the p-value it was removed with (given
        the candidates still present then; above ``shrink_alpha``); 1.0 where
        nothing was tested or the test was undefined (fewer than 10 residual
        degrees of freedom, a block inside the span of the other regressors, or
        no residual variation left), indistinguishable from each other.
        Backward elimination is path dependent. Lag-0 entries are 1.0 (the
        shrink does not test lag-0 edges; ``contemporaneous_result`` holds the
        CDNOTS+ graph for them, without per-test p-values).
    edge_strength : np.ndarray or None
        ``refine='shrink'`` only, same shape: partial R^2 in [0, 1] from the same
        tests (share of the remaining residual variance the parent explains); a
        continuous edge score in the role of ``gate_values``. 0.0 where nothing
        was tested, including lag 0 under ``contemporaneous``.
        With ``shrink_effect_floor``, edges in ``effect_pruned`` keep their
        ``edge_pvalue`` and ``edge_strength`` from the shrink.
    contemporaneous_result : CdnotsResult or None
        ``contemporaneous=True`` only: the CDNOTS+ run that found the lag-0
        edges with the lagged edges pinned. :meth:`to_marks` and
        :meth:`to_binary` read lag 0 from it, so unoriented lag-0 edges
        (dropped from ``cg_tig``) stay available. A copy with a replaced graph
        (e.g. after a filter) drops it, so pruned lag-0 edges do not reappear.
        The DoWhy bridge reads ``cg_tig``, as for CDNOTS+, so it does not see
        unoriented lag-0 edges. To keep them, wrap the CPDAG rendering yourself,
        ``wrap_graph(res.to_binary(undirected="bidirected"), df)``
        (:func:`~causalts.effects.wrap.wrap_graph`); a symmetric pair is a cycle
        to DoWhy's DAG export, so it then needs an explicit ``cycle_resolution``
        policy.
    """

    def __init__(
        self,
        graph: np.ndarray,
        df,
        var_names: list[str],
        *,
        gate_values: np.ndarray | None = None,
        stability_scores: np.ndarray | None = None,
        skeleton: np.ndarray | None = None,
        lambda_l0: float | None = None,
        gate_threshold: float | None = None,
        gate_threshold_rule: str | None = None,
        grow_history: list | None = None,
        effect_floor: float | str | None = None,
        n_effect_pruned: int | None = None,
        effect_pruned: np.ndarray | None = None,
        ar_path: bool | None = None,
        lambda_rule: str | None = None,
        runtime: float | None = None,
        refine: str | None = None,
        edge_pvalue: np.ndarray | None = None,
        edge_strength: np.ndarray | None = None,
        contemporaneous_result=None,
    ):
        self.cg_tig = graph
        self.var_names = list(var_names)
        self._df = df
        self._scm_cache = {}
        self.gate_values = gate_values
        self.stability_scores = stability_scores
        self.skeleton = skeleton
        self.lambda_l0 = lambda_l0
        self.gate_threshold = gate_threshold
        self.gate_threshold_rule = gate_threshold_rule
        self.grow_history = grow_history
        self.effect_floor = effect_floor
        self.n_effect_pruned = n_effect_pruned
        self.effect_pruned = effect_pruned
        self.ar_path = ar_path
        self.lambda_rule = lambda_rule
        self.runtime = runtime
        self.refine = refine
        self.edge_pvalue = edge_pvalue
        self.edge_strength = edge_strength
        self.contemporaneous_result = contemporaneous_result

    @property
    def candidate_graph(self):
        """Directed candidate graph that was refined (same array as ``skeleton``)."""
        return getattr(self, "skeleton", None)

    def to_binary(self, undirected=None, conflict=None):
        """Binary rendering; lag 0 comes from ``contemporaneous_result`` if set.

        See :meth:`~causalts.cdnots.result.CdnotsResult.to_binary` for the
        ``undirected`` and ``conflict`` policies.
        """
        contemp = getattr(self, "contemporaneous_result", None)  # older pickles
        if contemp is None:
            return super().to_binary(undirected=undirected, conflict=conflict)
        d = len(self.var_names)
        lag0 = contemp.to_binary(undirected=undirected, conflict=conflict)
        out = np.array(self.cg_tig, copy=True)
        out[:, :, 0] = np.asarray(lag0)[:d, :d, 0]
        return out

    def to_marks(self):
        """Edge marks; lag 0 comes from ``contemporaneous_result`` if set.

        Without it the graph is directed and lagged only, so the marks are
        reconstructed from ``cg_tig``
        (:meth:`~causalts.result.CausalResult.to_marks`).
        """
        marks = super().to_marks()
        contemp = getattr(self, "contemporaneous_result", None)  # older pickles
        if contemp is not None:
            d = len(self.var_names)
            marks[:, :, 0] = contemp.to_marks()[:d, :d, 0]
        return marks

    def _with_graph(self, graph):
        new = super()._with_graph(graph)
        # The CDNOTS+ result describes the old graph's lag 0.
        new.contemporaneous_result = None
        return new

    def plot(self, **kwargs):
        """Plot the discovered causal graph."""
        kwargs.setdefault("show_colorbar", False)
        return super().plot(**kwargs)

    def __repr__(self):
        d = len(self.var_names)
        edges = int((self.cg_tig == 1).sum())
        extras = []
        if self.gate_values is not None:
            if self.gate_threshold is not None:
                extras.append(f"gate_threshold={self.gate_threshold:.3g}")
            if self.gate_threshold_rule not in (None, "fixed"):
                extras.append(f"gate_threshold_rule={self.gate_threshold_rule!r}")
        if self.refine == "shrink":
            extras.append("refine='shrink'")
        if getattr(self, "contemporaneous_result", None) is not None:
            extras.append("contemporaneous=True")
        if self.stability_scores is not None:
            extras.append("stability_selection=True")
        if self.lambda_l0 is not None:
            extras.append(f"lambda_l0={self.lambda_l0:.4f}")
        extra_str = (", " + ", ".join(extras)) if extras else ""
        return f"GraceResult(vars={d}, edges={edges}{extra_str})"
