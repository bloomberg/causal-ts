# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import numpy as np


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

    def plot(self, var_names=None, **kwargs):
        from .plotting._core import plot_graph

        return plot_graph(
            graph=self.cg_tig,
            var_names=list(var_names or self.var_names),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # DoWhy bridge
    # ------------------------------------------------------------------

    def _bridge(self):
        from .effects.wrap import wrap_graph

        return wrap_graph(self.cg_tig, self._df, self.var_names)

    def estimate_effect(self, treatment, outcome, treatment_lag=1, **kwargs):
        return self._bridge().estimate_effect(
            treatment, outcome, treatment_lag, **kwargs
        )

    def fit_scm(self, mechanism_type="auto", force=False):
        if not force and mechanism_type in self._scm_cache:
            return self._scm_cache[mechanism_type]
        result = self._bridge().fit_scm(mechanism_type=mechanism_type)
        self._scm_cache[mechanism_type] = result
        return result

    def counterfactual(self, intervention, target, mechanism_type="auto"):
        from .effects.scm import counterfactual

        scm, _, lagged_df = self.fit_scm(mechanism_type=mechanism_type)
        return counterfactual(scm, lagged_df, intervention=intervention, target=target)

    def attribute_anomaly(self, anomaly, target, mechanism_type="auto", n_samples=2000):
        from .effects.root_cause import attribute_anomaly

        scm, _, lagged_df = self.fit_scm(mechanism_type=mechanism_type)
        return attribute_anomaly(
            scm, lagged_df, anomaly=anomaly, target=target, n_samples=n_samples
        )

    def falsify(self, **kwargs):
        return self._bridge().falsify(**kwargs)

    def refute_structure(self, **kwargs):
        return self._bridge().refute_structure(**kwargs)

    def evaluate_model(self, mechanism_type="auto", **kwargs):
        return self._bridge().evaluate_model(mechanism_type=mechanism_type, **kwargs)

    def arrow_strength(self, target, **kwargs):
        return self._bridge().arrow_strength(target, **kwargs)

    def causal_influence(self, target, **kwargs):
        return self._bridge().causal_influence(target, **kwargs)

    def parent_relevance(self, target, **kwargs):
        return self._bridge().parent_relevance(target, **kwargs)

    def distribution_change(self, data_new, target, **kwargs):
        return self._bridge().distribution_change(data_new, target, **kwargs)

    def summary(self, mechanism_type="linear", top_k=5):
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
