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
