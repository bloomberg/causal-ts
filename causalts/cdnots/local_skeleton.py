# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Local PC-around-the-target skeleton for the CDNOTS family.

Discovers the adjacencies of a *single* target variable in O(d) conditional-
independence tests instead of the O(d²) full CDNOTS skeleton. Used by
:func:`causalts.feature_selection.select_features` with ``algo="fastpc"``.

This is a fast *approximation* of CDNOTS adjacency, NOT full CDNOTS: it runs the
skeleton phase only, with no collider orientation, Meek rules, MCI pruning or
Phase-3 nonstationarity orientation. Use ``algo="cdnots"`` for the real
(O(d²), fully oriented) algorithm.

Correctness rests on the PC theorem: if a candidate ``X`` and the target are
non-adjacent, they are d-separated by some subset of the target's own
adjacencies — so conditioning only on the target's (growing) neighbourhood is
sufficient, never the whole graph. That is what makes the local run both correct
and O(d).

Time-series orientation is *free* for lagged neighbours: a neighbour at lag ≥ 1
precedes the target in time, so ``X(t-τ) -> target(t)`` is necessarily a parent
(a direct cause). Only contemporaneous (lag-0) neighbours are direction-
ambiguous; those are reported as adjacencies (see the ``include_lag0`` note in
``select_features``). Children and spouses are *not* recovered here — this
backend returns the target's direct causes, self-history and the ``C -> target``
nonstationarity flag, matching the validated CDNOTS full-run slice.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .phase3_utils import _C_PRESET_LABELS, make_c_array


def _prepare_embedded_data(df, num_lags, include_C, c_preset, c_array):
    """Append C (if requested) and build the lag-embedded data array, mirroring
    ``cdnots_discovery``'s data prep exactly so CI-test indices line up."""
    var_names = list(df.columns)
    n_c = 0
    c_node_names = None
    if include_C:
        T = df.shape[0]
        if c_array is not None:
            C_data = np.asarray(c_array, dtype=float)
            if C_data.ndim == 1:
                C_data = C_data.reshape(-1, 1)
            c_node_names = None
        elif c_preset is not None:
            C_data = make_c_array(T, c_preset)
            c_node_names = _C_PRESET_LABELS.get(c_preset)
        else:
            C_data = make_c_array(T, "linear")
            c_node_names = _C_PRESET_LABELS["linear"]
        n_c = C_data.shape[1]
        C = pd.DataFrame(C_data)
        df = pd.concat([df.reset_index(drop=True), C], axis=1)

    lags = list(range(num_lags + 1))
    data_lagged = pd.concat([df.shift(i) for i in lags], axis=1)
    n_drop = 2 * max(lags) if lags else 0
    data = data_lagged.iloc[n_drop:].values
    return data, var_names, n_c, c_node_names


def _candidate_indices(d_all, d_x, num_lags, target_idx, include_lag0):
    """Embedded-column indices of every allowed neighbour of ``target(t)``.

    Temporal constraints (mirroring ``skeleton_cnst``):
      * lag ≥ 1: all X variables (incl. the target's own past = self); C is
        NOT allowed cross-lag, so lagged C columns are excluded.
      * lag 0: C columns are allowed (the ``C -> target`` nonstationarity edge);
        cross-variable contemporaneous X only when ``include_lag0`` is set.
    """
    cands = []
    for tau in range(num_lags + 1):
        for v in range(d_all):
            idx = v + tau * d_all
            is_c = v >= d_x
            if tau == 0:
                if v == target_idx:
                    continue
                if is_c:
                    cands.append(idx)  # C(t) -> target(t): nonstationarity
                elif include_lag0:
                    cands.append(idx)  # contemporaneous cross-variable
            else:
                if is_c:
                    continue  # cross-lag C<->X forbidden
                cands.append(idx)  # lagged X (parents + self-history)
    return cands


def local_pc_skeleton(
    df,
    target,
    ci_test,
    *,
    num_lags=1,
    alpha=0.05,
    include_C=True,
    c_preset="linear",
    c_array=None,
    include_lag0=False,
    max_degree=None,
    max_combinations=20,
):
    """Discover the target's adjacencies with a local PC search.

    Returns
    -------
    graph : np.ndarray, shape (d_all, d_all, num_lags+1)
        Incoming edges to the target: ``graph[v, target, tau] == 1``.
    val_matrix : np.ndarray, same shape
        Edge strength (min |stat| across surviving conditioning sets).
    var_names : list[str]
        Variable names including any trailing C columns.
    n_c : int
        Number of C columns appended.
    """
    var_names_orig = list(df.columns)
    if target not in var_names_orig:
        raise ValueError(f"target {target!r} not in df columns")
    target_idx = var_names_orig.index(target)

    data, var_names_orig, n_c, c_node_names = _prepare_embedded_data(
        df, num_lags, include_C, c_preset, c_array
    )
    ci_test.data = data
    if hasattr(ci_test, "set_lag_structure"):
        d_full = len(var_names_orig) + n_c
        ci_test.set_lag_structure(n_vars=d_full, lags=list(range(num_lags + 1)))

    d_x = len(var_names_orig)
    d_all = d_x + n_c

    use_batch = hasattr(ci_test, "batch_test") and getattr(
        ci_test, "enable_batching", True
    )

    adj = _candidate_indices(d_all, d_x, num_lags, target_idx, include_lag0)
    adj = set(adj)
    # min |stat| per surviving candidate, for a strength score + neighbour sort
    strength = {c: np.inf for c in adj}

    def _stat_abs(stat):
        return abs(stat) if stat is not None else 0.0

    depth = -1
    while True:
        depth += 1
        if max_degree is not None and depth > max_degree:
            break
        if len(adj) - 1 < depth:
            break

        # Stable PC: collect removals, apply after the full depth sweep.
        to_remove = []
        # Sort neighbours strongest-first so capped combinations use the most
        # informative conditioners (mirrors full skeleton's _sort_neighbors).
        adj_sorted = sorted(adj, key=lambda c: strength.get(c, np.inf), reverse=True)

        for x in list(adj):
            others = [c for c in adj_sorted if c != x]
            if len(others) < depth:
                continue
            combos = combinations(others, depth)
            specs = []
            for i, S in enumerate(combos):
                if i >= max_combinations:
                    break
                specs.append(list(S))

            removed = False
            if use_batch:
                tests = [(x, target_idx, S) for S in specs]
                results = ci_test.batch_test(tests)
                for S, (p, stat) in zip(specs, results):
                    if p > alpha:
                        to_remove.append(x)
                        removed = True
                        break
                    strength[x] = min(strength[x], _stat_abs(stat))
            else:
                for S in specs:
                    p, stat = ci_test(x, target_idx, S)
                    if p > alpha:
                        to_remove.append(x)
                        removed = True
                        break
                    strength[x] = min(strength[x], _stat_abs(stat))
            if removed:
                continue

        for x in to_remove:
            adj.discard(x)
            strength.pop(x, None)

    graph = np.zeros((d_all, d_all, num_lags + 1), dtype=np.int8)
    val_matrix = np.zeros((d_all, d_all, num_lags + 1), dtype=float)
    for idx in adj:
        v = idx % d_all
        tau = idx // d_all
        graph[v, target_idx, tau] = 1
        s = strength.get(idx, 0.0)
        val_matrix[v, target_idx, tau] = 0.0 if not np.isfinite(s) else s

    var_names = list(var_names_orig)
    if c_node_names:
        var_names = var_names + list(c_node_names)
    elif n_c:
        var_names = var_names + [f"C{i + 1}" for i in range(n_c)]
    return graph, val_matrix, var_names, n_c
