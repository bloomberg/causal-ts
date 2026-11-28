# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import numpy as np


def _resolve_var_names(var_names, fallback):
    """Return ``var_names`` as a list, falling back without truthiness checks.

    ``pd.Index`` and ``np.ndarray`` raise ``ValueError`` on ``bool(...)``, so
    ``var_names or fallback`` fails whenever a caller passes ``df.columns`` or
    an array.
    """
    if var_names is not None:
        names = list(var_names)
        if len(names) > 0:
            return names
    if fallback is None:
        return None
    return list(fallback)


class CausalResult:
    """Base result object for all causal discovery algorithms.

    Subclasses must set at minimum:

    * ``cg_tig`` — np.ndarray of shape ``(d, d, max_lag+1)``
    * ``var_names`` — list of variable name strings
    * ``_df`` — the original input DataFrame (used by the DoWhy bridge)

    All plotting and DoWhy bridge methods are inherited automatically.
    """

    cg_tig: np.ndarray
    var_names: list[str]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Ensure every concrete subclass has _scm_cache as an instance dict,
        # not shared at the class level. Subclasses set self._scm_cache = {}
        # in their own __init__; this guard catches any that forget.
        original_init = cls.__init__

        def _patched_init(self, *args, **kw):
            original_init(self, *args, **kw)
            if not hasattr(self, "_scm_cache"):
                self._scm_cache = {}

        cls.__init__ = _patched_init

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, var_names=None, undirected=None, **kwargs):
        """Draw the discovered graph.

        Parameters
        ----------
        undirected : {"keep", "bidirected", "drop"}, optional
            How to render unoriented contemporaneous (``o-o``) edges. Left
            unset the plot uses ``cg_tig`` exactly as the engine produced it.
            ``"keep"`` draws them as genuine ``o-o`` marks via
            :meth:`~causalts.cdnots.result.CdnotsResult.to_marks`;
            ``"bidirected"`` and ``"drop"`` re-render the binary graph. Only
            available on results that carry a CPDAG (CDNOTS / CDNOTS+).
        """
        from .plotting._core import plot_graph

        if undirected is None:
            graph = self.cg_tig
        elif undirected == "keep":
            graph = self.to_marks()
        else:
            graph = self.to_binary(undirected=undirected)

        return plot_graph(
            graph=graph,
            var_names=_resolve_var_names(var_names, getattr(self, "var_names", None)),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # CPDAG rendering (overridden where a CPDAG is available)
    # ------------------------------------------------------------------

    def to_binary(self, undirected=None, conflict=None):
        """Binary ``[cause, effect, lag]`` rendering of the discovered graph.

        With no arguments this returns :attr:`cg_tig` unchanged. With a
        policy, the base implementation applies it to whatever symmetric lag-0
        pairs are present in :attr:`cg_tig` -- all it can see, since a binary
        array is the only record these engines keep.

        :class:`~causalts.cdnots.result.CdnotsResult` overrides this with a
        re-render off the stored CPDAG, which is strictly richer: it can
        recover ``o-o`` edges that ``cg_tig`` already discarded.

        Parameters
        ----------
        undirected : {"bidirected", "drop"}, optional
            How to render a symmetric lag-0 pair. Defaults to leaving it in
            place.
        conflict : {"drop", "bidirected"}, optional
            Accepted so the signature matches
            :meth:`~causalts.cdnots.result.CdnotsResult.to_binary`, but a
            **no-op here**: ``cg_tig`` records a conflicting ``x-x`` edge and
            an unoriented ``o-o`` edge identically, as symmetric 1s, so there
            is nothing for a non-CPDAG result to act on. The value is still
            validated rather than silently ignored, so a typo surfaces
            instead of quietly doing nothing.
        """
        if undirected is None and conflict is None:
            return self.cg_tig
        import numpy as np

        undirected = undirected or "bidirected"
        if undirected not in ("bidirected", "drop"):
            raise ValueError(
                f"undirected must be 'bidirected' or 'drop', got {undirected!r}"
            )
        if conflict is not None and conflict not in ("drop", "bidirected"):
            raise ValueError(
                f"conflict must be 'drop' or 'bidirected', got {conflict!r}"
            )
        cg = np.array(self.cg_tig, copy=True)
        if undirected == "drop":
            g0 = cg[:, :, 0].astype(bool)
            sym = g0 & g0.T
            cg[:, :, 0][sym] = 0
        return cg

    def to_marks(self):
        """Tigramite edge-mark (``<U3``) rendering of the discovered graph.

        A DAG is a CPDAG with nothing left unoriented, so this is well defined
        for every result type -- it is a *string format* of the graph, not an
        ambiguity report. Reads :attr:`cg_tig`:

        - one-way at lag 0        -> ``"-->"`` / ``"<--"``
        - symmetric at lag 0      -> ``"o-o"``
        - present at lag >= 1     -> ``"-->"`` (time-ordered, never mirrored)

        This is the same reconstruction :func:`~causalts.plotting._core.plot_graph`
        already applies to any binary graph, so plots and marks agree.

        :class:`~causalts.cdnots.result.CdnotsResult` overrides this to read
        the stored CPDAG directly, which is lossless rather than
        reconstructed: it distinguishes ``o-o`` from ``x-x`` and survives
        ``cg_tig`` having dropped the unoriented edges outright. For CEDAR,
        GRACE and LUCID -- all directed by construction -- the reconstruction
        is exact, and a symmetric lag-0 pair would indicate a conflict rather
        than genuine Markov equivalence.
        """
        import numpy as np

        cg = np.asarray(self.cg_tig).astype(bool)
        marks = np.full(cg.shape, "", dtype="<U3")
        marks[cg] = "-->"
        g0 = cg[:, :, 0]
        sym = g0 & g0.T
        rev = g0.T & ~g0
        marks[:, :, 0][sym] = "o-o"
        marks[:, :, 0][rev] = "<--"
        return marks

    # ------------------------------------------------------------------
    # DoWhy bridge
    # ------------------------------------------------------------------

    def _bridge(self):
        from .effects.wrap import wrap_graph

        return wrap_graph(self.cg_tig, self._df, self.var_names)

    def estimate_effect(self, treatment, outcome, treatment_lag=1, **kwargs):
        """Estimate the causal effect of treatment(t - lag) on outcome(t).

        Parameters
        ----------
        treatment : str
            Treatment variable name.
        outcome : str
            Outcome variable name.
        treatment_lag : int, default 1
            Lag of the treatment in time steps.
        **kwargs
            Passed to :func:`causalts.effects.effect.estimate_effect`, e.g.
            ``method``, ``confidence_intervals`` or ``n_bootstraps``.

        Returns
        -------
        DoWhy estimate object
            The estimated effect.

        Notes
        -----
        Requires DoWhy (``pip install causalts[dowhy]``).

        See Also
        --------
        causalts.effects.wrap.WrappedGraph.estimate_effect
        """
        return self._bridge().estimate_effect(
            treatment, outcome, treatment_lag, **kwargs
        )

    def fit_scm(self, mechanism_type="auto", force=False):
        """Fit a structural causal model (SCM) to this result's graph.

        The fitted model is cached on the result per ``mechanism_type``. Later
        calls with the same ``mechanism_type`` return the cached model without
        refitting, unless ``force=True``.

        Parameters
        ----------
        mechanism_type : str, default "auto"
            Mechanism type for the fit.
        force : bool, default False
            Refit even if a model is cached for ``mechanism_type``.

        Returns
        -------
        tuple
            ``(scm, dag, lagged_df)``.

        Notes
        -----
        Requires DoWhy (``pip install causalts[dowhy]``).

        See Also
        --------
        causalts.effects.wrap.WrappedGraph.fit_scm
        """
        if not force and mechanism_type in self._scm_cache:
            return self._scm_cache[mechanism_type]
        result = self._bridge().fit_scm(mechanism_type=mechanism_type)
        self._scm_cache[mechanism_type] = result
        return result

    def counterfactual(self, intervention, target, mechanism_type="auto"):
        """Estimate counterfactual values for target under do(intervention).

        Fits the SCM automatically if not already cached (see :meth:`fit_scm`).

        Parameters
        ----------
        intervention : dict
            Dict of {node: value}, e.g. ``{"X0_lag1": 0.0}``.
        target : str
            Current-time variable to query, e.g. ``"X2"``.
        mechanism_type : str, default "auto"
            Passed to :meth:`fit_scm`.

        Returns
        -------
        pd.Series
            Counterfactual values for ``target``, one per row of the lag-embedded data.

        Notes
        -----
        Requires DoWhy (``pip install causalts[dowhy]``).

        See Also
        --------
        causalts.effects.wrap.WrappedGraph.counterfactual
        """
        from .effects.scm import counterfactual

        scm, _, lagged_df = self.fit_scm(mechanism_type=mechanism_type)
        return counterfactual(scm, lagged_df, intervention=intervention, target=target)

    def attribute_anomaly(self, anomaly, target, mechanism_type="auto", n_samples=2000):
        """Attribute an anomaly in target to its upstream causes.

        Fits the SCM automatically if not already cached (see :meth:`fit_scm`).

        Parameters
        ----------
        anomaly : pd.Series or pd.DataFrame
            Anomalous observation(s).
        target : str
            Variable to explain, e.g. ``"X2"``.
        mechanism_type : str, default "auto"
            Passed to :meth:`fit_scm`.
        n_samples : int, default 2000
            Monte Carlo samples for Shapley attribution.

        Returns
        -------
        pd.DataFrame
            Columns ``[node, variable, lag, mean_attribution]``.

        Notes
        -----
        Requires DoWhy (``pip install causalts[dowhy]``).

        See Also
        --------
        causalts.effects.wrap.WrappedGraph.attribute_anomaly
        """
        from .effects.root_cause import attribute_anomaly

        scm, _, lagged_df = self.fit_scm(mechanism_type=mechanism_type)
        return attribute_anomaly(
            scm, lagged_df, anomaly=anomaly, target=target, n_samples=n_samples
        )

    def falsify(self, **kwargs):
        """Test whether the graph is consistent with the data's independence structure.

        No ground truth required. Deprecated in favour of
        :meth:`validate_transition_graph`, which reads the lag embedding correctly.

        Parameters
        ----------
        **kwargs
            Passed to :func:`causalts.effects.validate.falsify_graph`, e.g.
            ``significance_level``, ``n_permutations`` or ``suggestions``.

        Returns
        -------
        dict
            Falsification results.

        Notes
        -----
        Requires DoWhy (``pip install causalts[dowhy]``).

        See Also
        --------
        causalts.effects.wrap.WrappedGraph.falsify
        """
        return self._bridge().falsify(**kwargs)

    def validate_transition_graph(self, **kwargs):
        """Test the local Markov condition at every current-time node.

        Does not need DoWhy.

        Parameters
        ----------
        **kwargs
            Passed to :func:`causalts.effects.validate.validate_transition_graph`,
            e.g. ``significance_level`` or ``include_c``.

        Returns
        -------
        TransitionValidationResult
            Test results, including ``rejected`` and ``rejected_nodes``.

        See Also
        --------
        causalts.effects.wrap.WrappedGraph.validate_transition_graph
        """
        return self._bridge().validate_transition_graph(**kwargs)

    def history_sufficiency(self, extra_lags=2, **kwargs):
        """Test whether history older than the graph's max lag still informs.

        Does not need DoWhy.

        Parameters
        ----------
        extra_lags : int, default 2
            Number of lags beyond the graph's max lag to test.
        **kwargs
            Passed to :func:`causalts.effects.validate.history_sufficiency`,
            e.g. ``significance_level``.

        Returns
        -------
        HistorySufficiencyResult
            Test results for each variable.

        See Also
        --------
        causalts.effects.wrap.WrappedGraph.history_sufficiency
        """
        return self._bridge().history_sufficiency(extra_lags=extra_lags, **kwargs)

    def to_dowhy_transition(self, **kwargs):
        """Export the one-step transition graph and its lag-embedded data.

        Does not need DoWhy: it only builds the inputs for it.

        Parameters
        ----------
        **kwargs
            Passed to :func:`causalts.effects.export.build_transition_artifacts`,
            e.g. ``include_c`` or ``cycle_resolution``.

        Returns
        -------
        DoWhyArtifacts
            The graph and data to pass to DoWhy's ``gcm.StructuralCausalModel``.

        See Also
        --------
        causalts.effects.wrap.WrappedGraph.to_dowhy_transition
        """
        return self._bridge().to_dowhy_transition(**kwargs)

    def to_dowhy_identification(self, treatment, outcome, treatment_lag=1, **kwargs):
        """Export an unrolled graph and data frame for one effect query.

        Does not need DoWhy: it only builds the inputs for it.

        Parameters
        ----------
        treatment : str
            Treatment variable name.
        outcome : str
            Outcome variable name.
        treatment_lag : int, default 1
            Lag of the treatment in time steps.
        **kwargs
            Passed to :func:`causalts.effects.export.build_identification_artifacts`,
            e.g. ``include_c`` or ``cycle_resolution``.

        Returns
        -------
        DoWhyArtifacts
            The unrolled graph and data for the query.

        See Also
        --------
        causalts.effects.wrap.WrappedGraph.to_dowhy_identification
        """
        return self._bridge().to_dowhy_identification(
            treatment, outcome, treatment_lag=treatment_lag, **kwargs
        )

    def refute_effect(self, treatment, outcome, treatment_lag=1, **kwargs):
        """Stress-test an estimated effect with one of DoWhy's refuters.

        Parameters
        ----------
        treatment : str
            Treatment variable name.
        outcome : str
            Outcome variable name.
        treatment_lag : int, default 1
            Lag of the treatment in time steps.
        **kwargs
            Passed to :func:`causalts.effects.validate.refute_effect`, e.g.
            ``method`` or ``estimation_method``.

        Returns
        -------
        dict
            Refutation results.

        Notes
        -----
        Requires DoWhy (``pip install causalts[dowhy]``).

        See Also
        --------
        causalts.effects.wrap.WrappedGraph.refute_effect
        """
        return self._bridge().refute_effect(
            treatment, outcome, treatment_lag=treatment_lag, **kwargs
        )

    def sensitivity_analysis(self, treatment, outcome, treatment_lag=1, **kwargs):
        """How strong would an omitted confounder have to be to overturn the effect?

        Parameters
        ----------
        treatment : str
            Treatment variable name.
        outcome : str
            Outcome variable name.
        treatment_lag : int, default 1
            Lag of the treatment in time steps.
        **kwargs
            Passed to :func:`causalts.effects.validate.sensitivity_analysis`, e.g.
            ``simulation_method`` or ``estimation_method``.

        Returns
        -------
        dict
            Sensitivity results.

        Notes
        -----
        Requires DoWhy (``pip install causalts[dowhy]``).

        See Also
        --------
        causalts.effects.wrap.WrappedGraph.sensitivity_analysis
        """
        return self._bridge().sensitivity_analysis(
            treatment, outcome, treatment_lag=treatment_lag, **kwargs
        )

    def refute_structure(self, **kwargs):
        """Test structural assumptions: edge dependencies and local Markov conditions.

        No ground truth required.

        Parameters
        ----------
        **kwargs
            Passed to :func:`causalts.effects.validate.refute_structure`, e.g.
            ``significance_level`` or ``include_c``.

        Returns
        -------
        pd.DataFrame
            One row per structural test.

        Notes
        -----
        Requires DoWhy (``pip install causalts[dowhy]``).

        See Also
        --------
        causalts.effects.wrap.WrappedGraph.refute_structure
        """
        return self._bridge().refute_structure(**kwargs)

    def evaluate_model(self, mechanism_type="auto", **kwargs):
        """Evaluate the fitted mechanisms: fit quality and invertibility.

        No ground truth required.

        Parameters
        ----------
        mechanism_type : str, default "auto"
            Mechanism type for the fit.
        **kwargs
            Passed to :func:`causalts.effects.validate.evaluate_model`, e.g.
            ``evaluate_overall_kl_divergence`` or ``evaluate_causal_structure``.

        Returns
        -------
        dict
            Evaluation results.

        Notes
        -----
        Requires DoWhy (``pip install causalts[dowhy]``).

        See Also
        --------
        causalts.effects.wrap.WrappedGraph.evaluate_model
        """
        return self._bridge().evaluate_model(mechanism_type=mechanism_type, **kwargs)

    def arrow_strength(self, target, **kwargs):
        """Quantify how strongly each parent influences a target.

        Parameters
        ----------
        target : str
            Target variable name.
        **kwargs
            Passed to :func:`causalts.effects.influence.arrow_strength`, e.g.
            ``mechanism_type``.

        Returns
        -------
        pd.DataFrame
            One row per parent, with its ``strength``.

        Notes
        -----
        Requires DoWhy (``pip install causalts[dowhy]``).

        See Also
        --------
        causalts.effects.wrap.WrappedGraph.arrow_strength
        """
        return self._bridge().arrow_strength(target, **kwargs)

    def causal_influence(self, target, **kwargs):
        """Shapley decomposition of target's variability by upstream noise sources.

        Parameters
        ----------
        target : str
            Target variable name.
        **kwargs
            Passed to :func:`causalts.effects.influence.causal_influence`, e.g.
            ``mechanism_type`` or ``num_training_samples``.

        Returns
        -------
        pd.DataFrame
            Share of the target's variability from each upstream noise source.

        Notes
        -----
        Requires DoWhy (``pip install causalts[dowhy]``).

        See Also
        --------
        causalts.effects.wrap.WrappedGraph.causal_influence
        """
        return self._bridge().causal_influence(target, **kwargs)

    def parent_relevance(self, target, **kwargs):
        """Rank direct parents of target by explanatory power.

        Parameters
        ----------
        target : str
            Target variable name.
        **kwargs
            Passed to :func:`causalts.effects.influence.parent_relevance`, e.g.
            ``mechanism_type``.

        Returns
        -------
        pd.DataFrame
            One row per parent, plus a noise entry.

        Notes
        -----
        Requires DoWhy (``pip install causalts[dowhy]``).

        See Also
        --------
        causalts.effects.wrap.WrappedGraph.parent_relevance
        """
        return self._bridge().parent_relevance(target, **kwargs)

    def distribution_change(self, data_new, target, **kwargs):
        """Identify which upstream mechanisms changed between two data sets.

        Parameters
        ----------
        data_new : pd.DataFrame
            New data to compare against this result's data.
        target : str
            Target variable name.
        **kwargs
            Passed to :func:`causalts.effects.influence.distribution_change`, e.g.
            ``mechanism_type`` or ``robust``.

        Returns
        -------
        pd.DataFrame
            One row per upstream mechanism.

        Notes
        -----
        Requires DoWhy (``pip install causalts[dowhy]``).

        See Also
        --------
        causalts.effects.wrap.WrappedGraph.distribution_change
        """
        return self._bridge().distribution_change(data_new, target, **kwargs)

    def summary(self, mechanism_type="linear", top_k=5):
        """Print a one-page diagnostic report of the discovered graph.

        Prints edge counts, transition-structure validation and arrow strength
        for the most-connected node.

        Parameters
        ----------
        mechanism_type : str, default "linear"
            Passed to the arrow-strength step.
        top_k : int, default 5
            Number of strongest edges to show.

        Returns
        -------
        dict
            Computed metrics, including ``"transition_validation"`` when that step runs.

        Notes
        -----
        The arrow-strength section requires DoWhy (``pip install causalts[dowhy]``).
        If it fails, the section is skipped and the report says so.

        See Also
        --------
        causalts.effects.wrap.WrappedGraph.summary
        """
        return self._bridge().summary(mechanism_type=mechanism_type, top_k=top_k)

    # ------------------------------------------------------------------
    # Latent-confounding corrections
    # ------------------------------------------------------------------

    def _max_lag(self):
        return int(self.cg_tig.shape[2] - 1)

    def _obs_slice(self):
        """``(graph, d, max_lag)`` restricted to observed variables.

        ``cg_tig`` carries extra C-node rows/columns when discovery ran with
        ``include_C=True``; the deconfounding layer works on observed variables only.
        """
        d = self._df.shape[1]
        max_lag = self._max_lag()
        return self.cg_tig[:d, :d, : max_lag + 1], d, max_lag

    def deconfound(self, **kwargs):
        """Apply LUCID to this already-discovered graph, returning a ``LucidResult``.

        Reuses this result's skeleton instead of re-running discovery: LUCID runs the
        same base engine on both branches, so the routing decision never changes what
        would be discovered, making reuse exact. Equivalent to
        ``run_lucid(df, max_lag, discovery=self)``.

        Compatibility is checked where the information is recorded -- a ``num_lags``
        mismatch raises, differing ``alpha``/``include_C`` warn. See
        :func:`~causalts.confounders.run_lucid`.
        """
        from .confounders.routed_deconf import run_lucid

        graph, _d, max_lag = self._obs_slice()
        return run_lucid(self._df, max_lag, discovery=self, **kwargs)

    def tetrad_filter(self, threshold=0.25):
        """Drop lag-0 edges between variable pairs that share a latent factor.

        A fixed (non-adaptive) deconfounding strategy, in contrast to
        :meth:`deconfound`'s regime-adaptive routing. Returns a **new result of the
        same type** with the pruned graph, so it chains:
        ``res.tetrad_filter().deconfound()``.
        """
        from .utils.tetrad import apply_tetrad_lag0_filter

        pruned = apply_tetrad_lag0_filter(
            self.cg_tig, self._df, list(self._df.columns), threshold=threshold
        )
        return self._with_graph(pruned)

    def pds_filter(self, alpha=1e-10):
        """Prune edges that fail a post-double-selection test against observed controls.

        Returns a **new result of the same type**. Note this conditions on *observed*
        variables only -- it does not make an unobserved common cause observable.
        """
        from .confounders.routed_deconf import pds_filter as _pds

        graph, _d, max_lag = self._obs_slice()
        return self._with_graph(_pds(self._df, graph, max_lag, alpha=alpha))

    def _with_graph(self, graph):
        """Shallow copy of this result carrying a different graph."""
        import copy

        new = copy.copy(self)
        new.cg_tig = np.asarray(graph)
        new._scm_cache = {}
        return new
