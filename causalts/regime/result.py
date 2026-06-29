# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Result object for regime-conditional causal discovery."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..result import CausalResult
from .config import RegimeParams


class RegimeResult(CausalResult):
    """Result from regime-conditional causal discovery.

    Stores the aggregated summary graph (in ``cg_tig``) plus per-regime
    graphs, regime assignments, and metadata. Inherits all plotting and
    DoWhy bridge methods from ``CausalResult``.
    """

    def __init__(
        self,
        cg_tig: np.ndarray,
        var_names: list[str],
        regime_graphs: dict[int, np.ndarray],
        regime_labels: np.ndarray,
        regime_names: list[str],
        regime_params: list[RegimeParams],
        aggregation_strategy: str,
        per_regime_results: dict[int, list] | None,
        df: pd.DataFrame,
        metadata: dict | None = None,
    ):
        self.cg_tig = cg_tig
        self.var_names = list(var_names)
        self.regime_graphs = regime_graphs
        self.regime_labels = regime_labels
        self.regime_names = list(regime_names)
        self.regime_params = regime_params
        self.aggregation_strategy = aggregation_strategy
        self.per_regime_results = per_regime_results or {}
        self._df = df
        self.metadata = metadata or {}
        self._scm_cache = {}

    @property
    def n_regimes(self) -> int:
        return len(self.regime_graphs)

    def graph_for_regime(self, regime: int | str) -> np.ndarray:
        """Get the causal graph for a specific regime."""
        if isinstance(regime, str):
            regime = self.regime_names.index(regime)
        return self.regime_graphs[regime]

    def plot_regime(self, regime: int | str, **kwargs):
        """Plot the graph for a single regime."""
        g = self.graph_for_regime(regime)
        from ..plotting._core import plot_graph

        return plot_graph(graph=g, var_names=self.var_names, **kwargs)

    def regime_summary(self) -> dict:
        """Per-regime edge counts and sample statistics."""
        summary = {}
        for r, g in self.regime_graphs.items():
            n_samples = int((self.regime_labels == r).sum())
            n_edges = int((g[:, :, 1:] > 0).any(axis=2).sum())
            summary[self.regime_names[r]] = {
                "n_samples": n_samples,
                "n_edges": n_edges,
                "max_lag": self.regime_params[r].max_lag,
                "alpha": self.regime_params[r].alpha,
            }
        return summary
