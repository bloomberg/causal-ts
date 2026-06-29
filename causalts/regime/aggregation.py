# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Graph aggregation strategies for regime-conditional discovery."""

from __future__ import annotations

import numpy as np


def _pad_to_max_lag(graphs: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Pad all graphs to the same lag dimension (axis 2)."""
    max_lags = max(g.shape[2] for g in graphs.values())
    padded = {}
    for r, g in graphs.items():
        if g.shape[2] < max_lags:
            pad_width = ((0, 0), (0, 0), (0, max_lags - g.shape[2]))
            padded[r] = np.pad(g, pad_width, constant_values=0)
        else:
            padded[r] = g
    return padded


def aggregate_or(graphs: dict[int, np.ndarray]) -> np.ndarray:
    """Edge present if found in ANY regime."""
    padded = _pad_to_max_lag(graphs)
    stacked = np.stack(list(padded.values()), axis=0)
    return (stacked.any(axis=0)).astype(np.int8)


def aggregate_majority(
    graphs: dict[int, np.ndarray], threshold: float = 0.5
) -> np.ndarray:
    """Edge present if found in >= threshold fraction of regimes."""
    padded = _pad_to_max_lag(graphs)
    stacked = np.stack(list(padded.values()), axis=0)
    n = len(graphs)
    return ((stacked > 0).sum(axis=0) >= threshold * n).astype(np.int8)


def aggregate_weighted(
    graphs: dict[int, np.ndarray],
    weights: dict[int, float],
    threshold: float = 0.5,
) -> np.ndarray:
    """Weighted vote (e.g., by sample count per regime)."""
    padded = _pad_to_max_lag(graphs)
    total_weight = sum(weights.values())
    d = next(iter(padded.values())).shape[0]
    max_lags = next(iter(padded.values())).shape[2]
    score = np.zeros((d, d, max_lags), dtype=np.float64)

    for r, g in padded.items():
        w = weights.get(r, 1.0) / total_weight
        score += w * (g > 0).astype(np.float64)

    return (score >= threshold).astype(np.int8)


def aggregate(
    graphs: dict[int, np.ndarray],
    strategy: str = "or",
    weights: dict[int, float] | None = None,
    threshold: float = 0.5,
) -> np.ndarray:
    """Dispatch to the appropriate aggregation function."""
    if not graphs:
        raise ValueError("No regime graphs to aggregate")

    if strategy == "or":
        return aggregate_or(graphs)
    elif strategy == "majority":
        return aggregate_majority(graphs, threshold=threshold)
    elif strategy == "weighted":
        if weights is None:
            raise ValueError("weights required for 'weighted' strategy")
        return aggregate_weighted(graphs, weights, threshold=threshold)
    elif strategy == "intersection":
        padded = _pad_to_max_lag(graphs)
        stacked = np.stack(list(padded.values()), axis=0)
        return (stacked.all(axis=0)).astype(np.int8)
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy!r}")
