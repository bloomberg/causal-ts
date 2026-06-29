# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Convert causal-ts 3D graphs to DoWhy-compatible NetworkX DAGs."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def _default_var_names(d: int) -> list[str]:
    return [f"X{i}" for i in range(d)]


def _lag_node(name: str, lag: int, lag_suffix: str = "_lag{k}") -> str:
    if lag == 0:
        return name
    return name + lag_suffix.replace("{k}", str(lag))


def make_lagged_df(
    data: pd.DataFrame,
    max_lag: int,
    lags: list[int] | None = None,
    var_names: list[str] | None = None,
) -> pd.DataFrame:
    """Build a lag-embedded DataFrame matching CDNOTS's internal construction.

    Columns: [X0, X1, ..., X0_lag1, X1_lag1, ..., X0_lagK, X1_lagK, ...]
    First max(lags) rows are dropped (NaN from shift).

    Parameters
    ----------
    data : pd.DataFrame
        Original (T, d) time series DataFrame.
    max_lag : int
        Maximum lag depth.
    lags : list[int] or None
        Explicit lag list (e.g. [0, 1, 3]). Defaults to range(max_lag+1).
    var_names : list[str] or None
        Column names. Defaults to data.columns.

    Returns
    -------
    pd.DataFrame
        Lag-embedded DataFrame with proper column names and no leading
        NaN rows.
    """
    if var_names is None:
        var_names = list(data.columns)

    if lags is None:
        lags = list(range(max_lag + 1))

    frames = []
    for lag in lags:
        shifted = data[var_names].shift(lag)
        if lag == 0:
            shifted.columns = var_names
        else:
            shifted.columns = [_lag_node(n, lag) for n in var_names]
        frames.append(shifted)

    lagged = pd.concat(frames, axis=1)
    n_drop = max(lags) if lags else 0
    return lagged.iloc[n_drop:].reset_index(drop=True)


def graph_to_networkx(
    graph: np.ndarray,
    var_names: list[str] | None = None,
    mode: str = "full",
    include_c: bool = False,
    cycle_resolution: str = "error",
) -> nx.DiGraph:
    """Convert a causal-ts 3D graph array to a NetworkX DiGraph for DoWhy.

    Parameters
    ----------
    graph : np.ndarray
        Shape (d, d, max_lag+1). graph[i, j, lag]=1 means
        variable i at time t-lag causes variable j at time t.
    var_names : list[str] or None
        Variable names. Defaults to ["X0", "X1", ...].
    mode : str
        "full" -- all d*(max_lag+1) nodes (use for effect estimation).
        "summary" -- only current-time nodes; past lags become roots
        (use for GCM fitting and root cause analysis).
    include_c : bool
        If False (default), strips the last variable if it is the
        CDNOTS nonstationarity node C. Auto-detected when
        len(var_names) < graph.shape[0].
    cycle_resolution : str
        What to do when lag-0 edges form a cycle.
        "error" raises ValueError, "drop" silently drops
        the offending edge, "first" keeps the
        lexicographically-first direction.

    Returns
    -------
    nx.DiGraph
        NetworkX DiGraph with node/edge structure compatible with DoWhy.

    Raises
    ------
    ValueError
        If the resulting DAG contains a cycle and
        cycle_resolution="error".
    """
    d_raw = graph.shape[0]
    max_lag = graph.shape[2] - 1

    if var_names is None:
        var_names = _default_var_names(d_raw)

    # Strip C node added by CDNOTS when include_c=False
    if not include_c and len(var_names) < d_raw:
        graph = graph[: len(var_names), : len(var_names), :]
    elif not include_c and len(var_names) == d_raw:
        # Heuristic: if last var is "C", strip it
        if var_names[-1] == "C":
            var_names = var_names[:-1]
            graph = graph[: len(var_names), : len(var_names), :]

    d = len(var_names)

    dag = nx.DiGraph()

    if mode == "full":
        # Add all nodes: current-time + all lag versions
        for name in var_names:
            dag.add_node(name)
        for lag in range(1, max_lag + 1):
            for name in var_names:
                dag.add_node(_lag_node(name, lag))

        # Add edges
        for i in range(d):
            for j in range(d):
                for lag in range(max_lag + 1):
                    if graph[i, j, lag] == 1:
                        src = _lag_node(var_names[i], lag)
                        dst = var_names[j]
                        _add_edge_safe(dag, src, dst, cycle_resolution)

    elif mode == "summary":
        # Only current-time nodes; past variables become roots (exogenous)
        for name in var_names:
            dag.add_node(name)
        for lag in range(1, max_lag + 1):
            for name in var_names:
                dag.add_node(_lag_node(name, lag))

        for i in range(d):
            for j in range(d):
                for lag in range(max_lag + 1):
                    if graph[i, j, lag] == 1:
                        src = _lag_node(var_names[i], lag)
                        dst = var_names[j]
                        _add_edge_safe(dag, src, dst, cycle_resolution)
    else:
        raise ValueError(f"mode must be 'full' or 'summary', got {mode!r}")

    if not nx.is_directed_acyclic_graph(dag):
        cycles = list(nx.simple_cycles(dag))
        raise ValueError(
            f"The resulting graph contains cycles: {cycles}. "
            "Set cycle_resolution='drop' or 'first' to handle automatically."
        )

    return dag


def _add_edge_safe(dag: nx.DiGraph, src: str, dst: str, resolution: str) -> None:
    if dag.has_edge(dst, src):
        if resolution == "error":
            raise ValueError(
                f"Cycle detected: edges {src}->{dst} and {dst}->{src} both exist. "
                "Use cycle_resolution='drop' or 'first'."
            )
        elif resolution == "drop":
            return
        elif resolution == "first":
            if src > dst:
                return  # keep the lexicographically first direction
            dag.remove_edge(dst, src)
    dag.add_edge(src, dst)


def graph_to_dowhy_model(
    graph: np.ndarray,
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    treatment_lag: int = 1,
    var_names: list[str] | None = None,
    include_c: bool = False,
):
    """Build a DoWhy CausalModel from a causal-ts graph for effect estimation.

    Parameters
    ----------
    graph : np.ndarray
        Shape (d, d, max_lag+1).
    data : pd.DataFrame
        Original (T, d) time series DataFrame.
    treatment : str
        Variable name of the treatment (e.g. "X0").
    outcome : str
        Variable name of the outcome (e.g. "X2").
    treatment_lag : int
        Which lag of the treatment to use (default 1).
    var_names : list[str] or None
        Variable names. Defaults to data.columns.
    include_c : bool
        Whether to keep the C nonstationarity node.

    Returns
    -------
    tuple
        ``(dowhy.CausalModel, lagged_df)`` -- the model ready for
        identify+estimate, and the lag-embedded DataFrame used as data.
    """
    from ._compat import require_dowhy

    require_dowhy("effect estimation")
    import dowhy

    if var_names is None:
        var_names = list(data.columns)

    max_lag = graph.shape[2] - 1
    dag = graph_to_networkx(
        graph,
        var_names=var_names,
        mode="full",
        include_c=include_c,
        cycle_resolution="drop",
    )
    lagged_df = make_lagged_df(data, max_lag, var_names=var_names)

    treatment_node = _lag_node(treatment, treatment_lag)
    if treatment_node not in dag.nodes:
        raise ValueError(
            f"Treatment node '{treatment_node}' not found in graph. "
            f"Available nodes: {sorted(dag.nodes)}"
        )
    if outcome not in dag.nodes:
        raise ValueError(
            f"Outcome node '{outcome}' not found in graph. "
            f"Available nodes: {sorted(dag.nodes)}"
        )

    model = dowhy.CausalModel(
        data=lagged_df,
        treatment=treatment_node,
        outcome=outcome,
        graph=dag,
        proceed_when_unidentifiable=True,
    )
    return model, lagged_df


def networkx_to_graph(
    dag: nx.DiGraph,
    var_names: list[str],
    max_lag: int,
) -> np.ndarray:
    """Inverse: reconstruct (d, d, max_lag+1) array from a time-expanded DAG.

    Parameters
    ----------
    dag : nx.DiGraph
        Time-expanded NetworkX DiGraph (nodes named as
        graph_to_networkx produces).
    var_names : list[str]
        Current-time variable names.
    max_lag : int
        Maximum lag depth.

    Returns
    -------
    np.ndarray
        NumPy array of shape (d, d, max_lag+1).
    """
    d = len(var_names)
    out = np.zeros((d, d, max_lag + 1), dtype=int)
    name_to_idx = {n: i for i, n in enumerate(var_names)}

    for src, dst in dag.edges():
        # dst must be a current-time node
        if dst not in name_to_idx:
            continue
        j = name_to_idx[dst]

        # Parse lag from src
        lag = 0
        src_base = src
        for k in range(1, max_lag + 1):
            suffix = f"_lag{k}"
            if src.endswith(suffix):
                src_base = src[: -len(suffix)]
                lag = k
                break

        if src_base in name_to_idx:
            i = name_to_idx[src_base]
            out[i, j, lag] = 1

    return out
