# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Result object for LUCID."""

from __future__ import annotations

import numpy as np

from ..result import CausalResult


class LucidResult(CausalResult):
    """Result from :func:`~causalts.confounders.run_lucid`.

    Inherits plotting and DoWhy bridge methods from :class:`CausalResult`.

    Attributes
    ----------
    cg_tig : np.ndarray, shape (d, d, max_lag+1)
        Binary adjacency. ``cg_tig[cause, effect, lag] == 1`` means
        cause(t-lag) -> effect(t).
    var_names : list[str]
        Variable names.
    regime : str
        Confounding regime chosen by the router: ``"sparse"``, ``"pervasive"``, or
        ``"sf"`` (an intermediate hub/scale-free regime).
    spectral_ratio : float or None
        The router statistic ``R`` -- the share of residual correlation mass carried by
        the top ``router_k`` eigenvalues. ``None`` for routers that do not compute it.
    tau : float or None
        Routing threshold ``R`` was compared against. ``R > tau`` selects the pervasive
        branch.
    router : str
        Which router ran (``"auto"``, ``"spectral"`` or ``"mp"``).
    n_factors : int or None
        Number of pervasive latent factors implied by the Marchenko-Pastur edge
        (:func:`~causalts.confounders.mp_factor_count`). ``0`` means no pervasive
        factor was detected.
    factor_loadings : np.ndarray or None, shape (n_factors, d)
        Leading right singular vectors of the VAR residuals, one row per detected
        factor, columns aligned with ``var_names``.

        .. warning::
           These are **descriptive, not identified**. Factor directions are recovered
           only up to rotation, so a large entry means "this variable carries dominant
           shared variation", *not* "this variable has an identified latent parent".
           LUCID does not identify latent variables; it corrects for their footprint.
    info : dict
        Full router/pipeline diagnostics as returned by
        :func:`~causalts.confounders.routed_deconfound` with ``return_info=True``.
    runtime : float or None
        Wall-clock seconds for the LUCID call.
    """

    def __init__(
        self,
        graph: np.ndarray,
        df,
        var_names: list[str],
        *,
        info: dict | None = None,
        n_factors: int | None = None,
        factor_loadings: np.ndarray | None = None,
        runtime: float | None = None,
    ):
        self.cg_tig = graph
        self.var_names = list(var_names)
        self._df = df
        self._scm_cache = {}
        self.info = dict(info or {})
        self.regime = self.info.get("regime")
        self.spectral_ratio = self.info.get("R")
        self.tau = self.info.get("tau")
        self.router = self.info.get("router")
        self.n_factors = n_factors
        self.factor_loadings = factor_loadings
        self.runtime = runtime

    def plot(self, **kwargs):
        """Plot the deconfounded causal graph."""
        kwargs.setdefault("show_colorbar", False)
        return super().plot(**kwargs)

    def top_factor_variables(self, factor: int = 0, n: int = 5):
        """Variables loading most strongly on one factor direction, by ``|loading|``.

        Descriptive only -- see the ``factor_loadings`` warning above.

        Returns a list of ``(var_name, loading)`` pairs, largest ``|loading|`` first.
        """
        if self.factor_loadings is None:
            return []
        if not 0 <= factor < self.factor_loadings.shape[0]:
            raise IndexError(
                f"factor {factor} out of range (n_factors={self.factor_loadings.shape[0]})"
            )
        row = self.factor_loadings[factor]
        order = np.argsort(np.abs(row))[::-1][:n]
        return [(self.var_names[i], float(row[i])) for i in order]

    def __repr__(self):
        d = len(self.var_names)
        edges = int((self.cg_tig == 1).sum())
        bits = [f"regime={self.regime!r}"]
        if self.spectral_ratio is not None and self.tau is not None:
            bits.append(f"R={self.spectral_ratio:.3f} vs tau={self.tau:.3f}")
        if self.n_factors is not None:
            bits.append(f"n_factors={self.n_factors}")
        return f"LucidResult(d={d}, edges={edges}, " + ", ".join(bits) + ")"
