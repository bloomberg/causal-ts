# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Graph utilities for subgraph extraction and manipulation."""

from __future__ import annotations

import numpy as np


def extract_subgraph(
    graph: np.ndarray,
    target: int | str,
    depth: int = 1,
    var_names: list[str] | None = None,
    direction: str = "both",
) -> tuple[np.ndarray, list[int], list[str] | None]:
    """Extract a subgraph around a target variable.

    Parameters
    ----------
    graph : ndarray, shape (d, d, max_lag+1)
        Full causal graph. ``graph[i, j, lag] != 0`` means
        variable *i* at time *t-lag* causes variable *j* at time *t*.
    target : int or str
        Target variable -- index or name (requires *var_names*).
    depth : int, optional
        Number of hops from the target. ``depth=1`` returns direct
        parents and children; ``depth=2`` also includes their
        parents/children. Default 1.
    var_names : list of str, optional
        Variable names. Required when *target* is a string.
    direction : str, optional
        ``"both"`` (default) -- parents and children.
        ``"parents"`` -- ancestors only.
        ``"children"`` -- descendants only.

    Returns
    -------
    sub_graph : ndarray
        Filtered graph of shape ``(len(indices), len(indices), max_lag+1)``
        containing only edges among the included variables.
    indices : list of int
        Variable indices (into the original graph) that are included.
    sub_var_names : list of str or None
        Corresponding variable names, or None if *var_names* was None.
    """
    if direction not in ("both", "parents", "children"):
        raise ValueError(
            f"direction must be 'both', 'parents', or 'children', got {direction!r}"
        )

    d = graph.shape[0]

    if isinstance(target, str):
        if var_names is None:
            raise ValueError("var_names required when target is a string")
        name_to_idx = {n: i for i, n in enumerate(var_names)}
        if target not in name_to_idx:
            raise ValueError(f"target '{target}' not found in var_names")
        target_idx = name_to_idx[target]
    else:
        target_idx = int(target)
        if target_idx < 0 or target_idx >= d:
            raise ValueError(f"target index {target_idx} out of range [0, {d})")

    # Convert to boolean for traversal
    graph_bool = (
        np.asarray(graph).astype(bool)
        if graph.dtype.kind in ("U", "S")
        else (graph != 0)
    )

    included = {target_idx}
    frontier = {target_idx}

    for _ in range(depth):
        new_frontier = set()
        for node in frontier:
            if direction in ("both", "parents"):
                parents = set(np.where(graph_bool[:, node, :].any(axis=1))[0])
                new_frontier |= parents
            if direction in ("both", "children"):
                children = set(np.where(graph_bool[node, :, :].any(axis=1))[0])
                new_frontier |= children
        new_frontier -= included
        included |= new_frontier
        frontier = new_frontier

    indices = sorted(included)
    idx_map = {old: new for new, old in enumerate(indices)}
    n_sub = len(indices)
    max_lag_plus_1 = graph.shape[2]

    sub_graph = np.zeros((n_sub, n_sub, max_lag_plus_1), dtype=graph.dtype)
    for i_old in indices:
        for j_old in indices:
            sub_graph[idx_map[i_old], idx_map[j_old], :] = graph[i_old, j_old, :]

    sub_var_names = [var_names[i] for i in indices] if var_names is not None else None
    return sub_graph, indices, sub_var_names
