# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tigramite bridge for direct, total, and mediated causal effect estimation.

Wraps ``tigramite.causal_effects.CausalEffects`` with a causal-ts-friendly API:
accepts the standard ``(d, d, max_lag+1)`` binary graph and a pandas DataFrame,
handles the graph-format conversion, and exposes direct effect, total effect, and
mediation analysis in a single :class:`TigramiteEffects` object.

The current DoWhy bridge (``causalts.effects.estimate_effect``) estimates the
*average treatment effect* (ATE) via the backdoor criterion on a lag-embedded DAG.
This module instead uses tigramite's Wright path method, which:

- Works directly on the time-series graph without lag embedding
- Estimates *total* causal effects (sum over all directed paths)
- Separates *direct* effects (only the direct edge, no mediation)
- Identifies mediator variables and quantifies *mediated* effects

Requires tigramite::

    pip install tigramite

The graph input accepts either:

- ``res.cg_tig`` from :func:`run_cdnots` / :func:`run_cdnots_plus`
- ``disc_graph`` — the first return value of ``run_cdnots_gated()``
- Any binary ``(d, d, max_lag+1)`` int array (``graph[cause, effect, lag] = 1``)

Example::

    from causalts import run_cdnots
    from causalts.effects.tigramite_effects import TigramiteEffects

    res = run_cdnots(df, ci, num_lags=3, include_C=False)

    eff = TigramiteEffects(res.cg_tig, df, X_var="X0", X_lag=1, Y_var="X3",
                           var_names=df.columns.tolist())
    eff.fit()

    print("Total effect :", eff.total_effect())
    print("Direct effect:", eff.direct_effect())
    print("Mediators    :", eff.get_mediators())
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def tigramite_available() -> bool:
    """Return True if tigramite is importable."""
    try:
        import tigramite  # noqa: F401

        return True
    except ImportError:
        return False


def _require_tigramite() -> None:
    if not tigramite_available():
        raise ImportError(
            "tigramite is required for this module. Install it with:\n"
            "    pip install tigramite"
        )


# ---------------------------------------------------------------------------
# Graph format conversion
# ---------------------------------------------------------------------------


def _to_tigramite_graph(graph: np.ndarray) -> np.ndarray:
    """Convert causal-ts binary graph to tigramite string-format graph.

    causal-ts convention:  ``graph[cause, effect, lag] = 1``
    tigramite convention:  ``graph[cause, effect, tau] = '-->'``

    Both conventions share the same axis ordering, so conversion is a
    straightforward string substitution.

    Parameters
    ----------
    graph : ndarray (d, d, max_lag+1), dtype int
        Binary causal graph from CDNOTS/CEDAR/GRACE.

    Returns
    -------
    ndarray (d, d, max_lag+1), dtype '<U3'
        Tigramite stationary_dag format.
    """
    tig = np.empty(graph.shape, dtype="<U3")
    tig[:] = ""
    tig[graph.astype(bool)] = "-->"
    # Tigramite requires the diagonal at lag 0 to be empty (no self-loops at t=t)
    np.fill_diagonal(tig[:, :, 0], "")
    return tig


def _resolve_var(var: str | int, var_names: list[str] | None) -> int:
    """Return the integer index for a variable name or index."""
    if isinstance(var, int):
        return var
    if var_names is None:
        raise ValueError(
            f"var_names must be provided when specifying variables by name (got '{var}')."
        )
    try:
        return var_names.index(var)
    except ValueError:
        raise ValueError(f"Variable '{var}' not found in var_names={var_names}.")


def _to_tigramite_df(df: pd.DataFrame) -> Any:
    """Wrap a pandas DataFrame in a tigramite DataFrame object."""
    from tigramite.data_processing import DataFrame as TigramiteDF

    return TigramiteDF(df.to_numpy(dtype=float))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class TigramiteEffects:
    """Tigramite-based causal effect estimator for time-series graphs.

    Estimates direct, total, and mediated causal effects using tigramite's
    Wright path-tracing method, which multiplies regression coefficients along
    directed paths in the time-series graph.

    Parameters
    ----------
    graph : ndarray (d, d, max_lag+1)
        Binary causal graph (causal-ts format: ``graph[cause, effect, lag]``).
    df : DataFrame (T, d)
        Time series used for fitting the path-coefficient regressions.
    X_var : str or int
        Treatment variable (name or index).
    X_lag : int
        Lag of the treatment (``X_lag=1`` means ``X_var(t-1)``).
    Y_var : str or int
        Outcome variable (name or index); always at the current time ``t``.
    var_names : list of str, optional
        Variable names matching ``df`` columns and ``graph`` axes.
        Required when ``X_var`` / ``Y_var`` are strings.

    Examples
    --------
    >>> eff = TigramiteEffects(graph, df, X_var="X0", X_lag=1, Y_var="X3",
    ...                        var_names=df.columns.tolist())
    >>> eff.fit()
    >>> print(eff.total_effect())
    >>> print(eff.direct_effect())
    >>> print(eff.get_mediators())
    """

    def __init__(
        self,
        graph: np.ndarray,
        df: pd.DataFrame,
        X_var: str | int,
        X_lag: int,
        Y_var: str | int,
        var_names: list[str] | None = None,
    ):
        _require_tigramite()

        self._graph_bin = np.asarray(graph)
        self._df = df
        self._var_names = list(var_names) if var_names is not None else list(df.columns)
        self._X_idx = _resolve_var(X_var, self._var_names)
        self._Y_idx = _resolve_var(Y_var, self._var_names)
        self._X_lag = int(X_lag)
        self._X_var = self._var_names[self._X_idx]
        self._Y_var = self._var_names[self._Y_idx]

        # Tigramite uses negative-tau convention: X at lag k → (idx, -k)
        self._X_set = {(self._X_idx, -self._X_lag)}
        self._Y_set = {(self._Y_idx, 0)}

        self._tig_graph = _to_tigramite_graph(self._graph_bin)
        self._tig_df = _to_tigramite_df(df)
        self._ce: Any = None  # tigramite CausalEffects object
        self._fitted_total: Any = None
        self._fitted_direct: Any = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_ce(self) -> Any:
        """Build the tigramite CausalEffects object."""
        from tigramite.causal_effects import CausalEffects

        return CausalEffects(
            self._tig_graph,
            graph_type="stationary_dag",
            X=self._X_set,
            Y=self._Y_set,
            verbosity=0,
        )

    def fit(self, method: str = "parents") -> "TigramiteEffects":
        """Fit path-coefficient models for total and direct effects.

        Parameters
        ----------
        method : {'parents', 'optimal'}
            Adjustment method for Wright's path coefficients.
            ``'parents'`` uses the graph parents (valid for DAGs, fast).
            ``'optimal'`` uses the optimal adjustment set (more robust).

        Returns
        -------
        self
        """
        self._ce = self._build_ce()
        self._ce.fit_wright_effect(self._tig_df, mediation=None, method=method)
        self._fitted_total = True
        self._ce.fit_wright_effect(self._tig_df, mediation="direct", method=method)
        self._fitted_direct = True
        return self

    # ------------------------------------------------------------------
    # Effect queries
    # ------------------------------------------------------------------

    def total_effect(self) -> float:
        """Total causal effect of ``X_var(t-X_lag)`` on ``Y_var(t)``.

        Sum of Wright path products along *all* directed paths from X to Y.

        Returns
        -------
        float
        """
        if not self._fitted_total:
            raise RuntimeError("Call fit() before querying effects.")
        self._ce.fit_wright_effect(self._tig_df, mediation=None, method="parents")
        pred = self._ce.predict_wright_effect(intervention_data=np.ones((1, 1)))
        return float(np.squeeze(pred))

    def direct_effect(self) -> float:
        """Direct causal effect of ``X_var(t-X_lag)`` on ``Y_var(t)``.

        Only the direct edge X→Y; all paths through mediators are excluded.

        Returns
        -------
        float
        """
        if not self._fitted_direct:
            raise RuntimeError("Call fit() before querying effects.")
        self._ce.fit_wright_effect(self._tig_df, mediation="direct", method="parents")
        pred = self._ce.predict_wright_effect(intervention_data=np.ones((1, 1)))
        return float(np.squeeze(pred))

    def mediated_effect(
        self, through: list[tuple[str | int, int]] | None = None
    ) -> float:
        """Causal effect mediated through specific variables.

        Parameters
        ----------
        through : list of (var, lag) tuples, optional
            Mediator variables. Each element is ``(variable, lag)`` where
            lag ≥ 0. If *None*, uses all mediators identified by
            :meth:`get_mediators`.

        Returns
        -------
        float
            Effect that travels through at least one mediator in ``through``.
        """
        if self._ce is None:
            raise RuntimeError("Call fit() before querying effects.")

        if through is None:
            mediators = self.get_mediators(as_names=False)
        else:
            mediators = [
                (_resolve_var(v, self._var_names), -int(lag)) for v, lag in through
            ]

        if not mediators:
            return 0.0

        self._ce.fit_wright_effect(self._tig_df, mediation=mediators, method="parents")
        pred = self._ce.predict_wright_effect(intervention_data=np.ones((1, 1)))
        return float(np.squeeze(pred))

    def get_mediators(self, as_names: bool = True) -> list:
        """Return mediator variables on causal paths from X to Y.

        Parameters
        ----------
        as_names : bool, default True
            If True, return ``[(var_name, lag), ...]``.
            If False, return tigramite's ``[(var_idx, -lag), ...]``.

        Returns
        -------
        list
        """
        if self._ce is None:
            self._ce = self._build_ce()
        raw = self._ce.get_mediators(start=self._X_set, end=self._Y_set)
        if not as_names:
            return list(raw)
        return [(self._var_names[idx], abs(tau)) for idx, tau in sorted(raw)]

    def optimal_adjustment_set(self, as_names: bool = True) -> list:
        """Return the optimal adjustment set for identifying the total effect.

        Parameters
        ----------
        as_names : bool, default True

        Returns
        -------
        list of (var, lag) or (var_idx, -lag)
        """
        if self._ce is None:
            self._ce = self._build_ce()
        raw = self._ce.get_optimal_set()
        if not as_names or raw is None:
            return list(raw) if raw else []
        return [(self._var_names[idx], abs(tau)) for idx, tau in sorted(raw)]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a summary dict with total, direct, and indirect effects.

        Returns
        -------
        dict with keys: treatment, outcome, treatment_lag, total, direct,
            indirect, mediators, optimal_adjustment_set.
        """
        total = self.total_effect()
        direct = self.direct_effect()
        mediators = self.get_mediators()
        return {
            "treatment": self._X_var,
            "treatment_lag": self._X_lag,
            "outcome": self._Y_var,
            "total": total,
            "direct": direct,
            "indirect": total - direct,
            "mediators": mediators,
            "optimal_adjustment_set": self.optimal_adjustment_set(),
        }

    def __repr__(self) -> str:
        fitted = "fitted" if self._fitted_total else "not fitted"
        return (
            f"TigramiteEffects({self._X_var}(t-{self._X_lag}) → {self._Y_var}, "
            f"{fitted})"
        )


# ---------------------------------------------------------------------------
# Pathwise effect decomposition
# ---------------------------------------------------------------------------


def _find_all_paths(
    graph: np.ndarray,
    X_idx: int,
    Y_idx: int,
    max_path_lag: int,
) -> list[dict]:
    """BFS backward from Y to find all directed paths to X.

    Returns a list of dicts, each with:
        "nodes"    : list of variable indices along the path [X, ..., Y]
        "edge_lags": list of edge lags along the path
        "total_lag": sum of edge lags (= X_lag needed)
    """
    d = graph.shape[0]
    max_lag = graph.shape[2] - 1
    paths = []

    # BFS state: (current_var, accumulated_lag, path_nodes_reversed, edge_lags_reversed)
    queue = [(Y_idx, 0, [Y_idx], [])]

    while queue:
        curr, acc_lag, nodes_rev, lags_rev = queue.pop(0)

        # Find all predecessors of curr in the graph
        for pred in range(d):
            if pred == curr and acc_lag == 0:
                continue  # skip self-loop at the start (Y→Y not useful)
            for tau in range(1, max_lag + 1):
                if graph[pred, curr, tau] == 0:
                    continue
                new_lag = acc_lag + tau
                if new_lag > max_path_lag:
                    continue

                new_nodes = [pred] + nodes_rev
                new_lags = [tau] + lags_rev

                if pred == X_idx:
                    paths.append(
                        {
                            "nodes": new_nodes,
                            "edge_lags": new_lags,
                            "total_lag": new_lag,
                        }
                    )
                else:
                    # Avoid revisiting same variable (prevent cycles in variable space)
                    if pred not in nodes_rev:
                        queue.append((pred, new_lag, new_nodes, new_lags))

    return paths


def pathwise_effects(
    graph: np.ndarray,
    df: pd.DataFrame,
    X_var: str | int,
    Y_var: str | int,
    var_names: list[str] | None = None,
    max_path_lag: int | None = None,
    method: str = "parents",
) -> dict:
    """Discover all causal paths from X to Y and decompose effects by pathway.

    Automatically traverses the causal graph to find every directed path from
    ``X_var`` to ``Y_var``, determines the required source lag for each path,
    and uses :class:`TigramiteEffects` to quantify the effect at each lag horizon.

    The result is a complete **pathway decomposition**: what fraction of the
    cumulative causal influence travels along each route.

    Parameters
    ----------
    graph : ndarray (d, d, max_lag+1)
        Binary causal graph.
    df : DataFrame (T, d)
        Time series data for fitting Wright path regressions.
    X_var : str or int
        Treatment (source) variable.
    Y_var : str or int
        Outcome (target) variable.
    var_names : list of str, optional
        Variable names. Defaults to ``df.columns``.
    max_path_lag : int, optional
        Maximum total lag depth to search for paths. Defaults to
        ``graph.shape[2] - 1`` (the graph's max_lag) times 3.
    method : str, default 'parents'
        Wright regression method (passed to ``TigramiteEffects.fit``).

    Returns
    -------
    dict with keys:

    ``paths`` : list of dict
        Each dict has:
        - ``"route"`` — human-readable string like ``"PKC → PKA → Mek"``
        - ``"nodes"`` — list of variable names along the path
        - ``"edge_lags"`` — list of per-edge lags
        - ``"total_lag"`` — sum of edge lags (= X_lag used)

    ``by_lag`` : dict mapping lag → effect info
        For each unique total_lag found, contains:
        - ``"effect"`` — total Wright effect at that lag horizon
        - ``"paths"`` — list of path dicts at that lag

    ``cumulative`` : float
        Sum of effects across all lag horizons.

    ``cumulative_abs`` : float
        Sum of absolute effects (for computing shares).

    ``shares`` : dict mapping lag → float
        ``|effect_at_lag| / cumulative_abs`` — pathway share as fraction.

    Examples
    --------
    >>> from causalts.effects.tigramite_effects import pathwise_effects
    >>> result = pathwise_effects(disc_graph, df, "PKC", "Mek",
    ...                           var_names=var_names)
    >>> for lag, info in sorted(result["by_lag"].items()):
    ...     routes = ", ".join(p["route"] for p in info["paths"])
    ...     print(f"  Lag {lag}: effect={info['effect']:+.4f}  "
    ...           f"share={result['shares'][lag]:.1%}  ({routes})")
    """
    _require_tigramite()

    if var_names is None:
        var_names = list(df.columns)
    X_idx = _resolve_var(X_var, var_names)
    Y_idx = _resolve_var(Y_var, var_names)
    graph = np.asarray(graph)

    if max_path_lag is None:
        max_path_lag = (graph.shape[2] - 1) * 3

    # 1. Discover all directed paths
    raw_paths = _find_all_paths(graph, X_idx, Y_idx, max_path_lag)

    if not raw_paths:
        return {
            "paths": [],
            "by_lag": {},
            "cumulative": 0.0,
            "cumulative_abs": 0.0,
            "shares": {},
        }

    # Enrich with names and route strings
    paths = []
    for p in raw_paths:
        names = [var_names[i] for i in p["nodes"]]
        paths.append(
            {
                "route": " → ".join(names),
                "nodes": names,
                "edge_lags": p["edge_lags"],
                "total_lag": p["total_lag"],
            }
        )

    # 2. Group by total_lag
    from collections import defaultdict

    lag_groups = defaultdict(list)
    for p in paths:
        lag_groups[p["total_lag"]].append(p)

    # 3. Compute effect at each lag horizon
    by_lag = {}
    for lag in sorted(lag_groups):
        eff = TigramiteEffects(
            graph,
            df,
            X_var=X_idx,
            X_lag=lag,
            Y_var=Y_idx,
            var_names=var_names,
        )
        eff.fit(method=method)
        total = eff.total_effect()
        by_lag[lag] = {
            "effect": total,
            "paths": lag_groups[lag],
        }

    # 4. Cumulative and shares
    cumulative = sum(info["effect"] for info in by_lag.values())
    cumulative_abs = sum(abs(info["effect"]) for info in by_lag.values())
    shares = {}
    if cumulative_abs > 0:
        shares = {
            lag: abs(info["effect"]) / cumulative_abs for lag, info in by_lag.items()
        }

    return {
        "paths": paths,
        "by_lag": by_lag,
        "cumulative": cumulative,
        "cumulative_abs": cumulative_abs,
        "shares": shares,
    }
