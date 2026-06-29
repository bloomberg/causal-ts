# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Result object returned by GRACE discovery functions."""

from __future__ import annotations

import numpy as np

from ..result import CausalResult


class GraceResult(CausalResult):
    """Result from :func:`run_cdnots_gated` or :func:`run_stability_selection`.

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
        ``cg_tig``. Available from :func:`run_cdnots_gated`.
    stability_scores : np.ndarray or None
        Edge stability frequencies in [0, 1], same shape as ``cg_tig``.
        Available from :func:`run_stability_selection`.
    skeleton : np.ndarray or None
        CDNOTS skeleton before gating, shape (d, d, max_lag+1).
    lambda_l0 : float or None
        L0 regularisation strength used during training.
    gate_threshold : float or None
        Gate threshold applied to produce the binary graph.
    runtime : float or None
        Total wall-clock time in seconds.
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
        runtime: float | None = None,
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
        self.runtime = runtime

    def plot(self, **kwargs):
        """Plot the discovered causal graph."""
        kwargs.setdefault("show_colorbar", False)
        return super().plot(**kwargs)

    def __repr__(self):
        d = len(self.var_names)
        edges = int((self.cg_tig == 1).sum())
        extras = []
        if self.gate_values is not None:
            extras.append(f"gate_threshold={self.gate_threshold}")
        if self.stability_scores is not None:
            extras.append("stability_selection=True")
        if self.lambda_l0 is not None:
            extras.append(f"lambda_l0={self.lambda_l0:.4f}")
        extra_str = (", " + ", ".join(extras)) if extras else ""
        return f"GraceResult(vars={d}, edges={edges}{extra_str})"
