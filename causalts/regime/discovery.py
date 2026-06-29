# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Main orchestrator for regime-conditional causal discovery."""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from .aggregation import aggregate
from .config import RegimeConfig, RegimeParams
from .detection import QuantileRegimeDetector, RegimeDetector
from .result import RegimeResult
from .segments import extract_segments

logger = logging.getLogger(__name__)


def run_regime_discovery(
    df: pd.DataFrame,
    discovery_fn: Callable,
    config: RegimeConfig | None = None,
    detector: RegimeDetector | None = None,
    regime_labels: np.ndarray | None = None,
    verbose: bool = False,
) -> RegimeResult:
    """Run causal discovery per regime and aggregate results.

    Parameters
    ----------
    df : DataFrame (T, d)
        Input time series.
    discovery_fn : callable
        Function ``(df, max_lag=..., alpha=..., **kw) -> CausalResult``.
        Called per segment with regime-specific parameters.
    config : RegimeConfig, optional
        Configuration. Defaults to 3 quantile regimes with OR aggregation.
    detector : RegimeDetector, optional
        Custom regime detector. If None, uses QuantileRegimeDetector.
    regime_labels : np.ndarray, optional
        Pre-computed regime labels (T,). Overrides detector.
    verbose : bool
        Print progress.

    Returns
    -------
    RegimeResult
    """
    if config is None:
        config = RegimeConfig()

    n_regimes = config.n_regimes
    var_names = list(df.columns)
    d = df.shape[1]

    # Resolve regime parameters
    if config.regime_params is not None:
        if len(config.regime_params) != n_regimes:
            raise ValueError(
                f"regime_params has {len(config.regime_params)} entries, "
                f"need {n_regimes}"
            )
        regime_params = config.regime_params
    else:
        regime_params = [RegimeParams() for _ in range(n_regimes)]

    # Resolve regime names
    if config.regime_names is not None:
        regime_names = list(config.regime_names)
    else:
        regime_names = [f"regime-{i}" for i in range(n_regimes)]

    # Assign regimes — two paths:
    # 1. Detector with detect_segments() (e.g., PeltRegimeDetector): use directly
    # 2. Standard label-based: assign labels → extract_segments per regime
    use_direct_segments = (
        detector is not None
        and hasattr(detector, "detect_segments")
        and regime_labels is None
    )

    if use_direct_segments:
        all_segments = detector.detect_segments(df)
        labels = np.zeros(len(df), dtype=np.int32)
        for seg in all_segments:
            labels[seg.start_idx : seg.end_idx] = seg.regime
    elif regime_labels is not None:
        labels = np.asarray(regime_labels, dtype=np.int32)
    elif detector is not None:
        labels = detector.assign(df)
    else:
        labels = QuantileRegimeDetector(n_regimes).assign(df)

    if verbose:
        for r in range(n_regimes):
            n_pts = int((labels == r).sum())
            logger.info(
                "%s: %d samples (%.1f%%), max_lag=%d",
                regime_names[r],
                n_pts,
                100 * n_pts / len(df),
                regime_params[r].max_lag,
            )

    # Per-regime discovery
    regime_graphs: dict[int, np.ndarray] = {}
    per_regime_results: dict[int, list] = {}
    regime_weights: dict[int, float] = {}

    # Pre-group direct segments by regime if available
    if use_direct_segments:
        _direct_segments_by_regime: dict[int, list] = {}
        for seg in all_segments:
            _direct_segments_by_regime.setdefault(seg.regime, []).append(seg)
    else:
        _direct_segments_by_regime = None

    for r in range(n_regimes):
        params = regime_params[r]
        min_len = config.min_segment_length or (2 * params.max_lag + 30)

        if _direct_segments_by_regime is not None:
            # Use PELT-provided segments directly (already contiguous)
            segments = [
                s for s in _direct_segments_by_regime.get(r, []) if len(s.df) >= min_len
            ]
        else:
            segments = extract_segments(
                df, labels, r, min_length=min_len, buffer=config.buffer
            )

        if not segments:
            if verbose:
                logger.info("%s: no valid segments, skipping", regime_names[r])
            continue

        total_samples = sum(len(s.df) for s in segments)
        regime_weights[r] = float(total_samples)

        if verbose:
            logger.info(
                "%s: %d segments, %d total samples",
                regime_names[r],
                len(segments),
                total_samples,
            )

        # Run discovery on each segment
        segment_graphs: list[np.ndarray] = []
        segment_results: list = []

        for seg in segments:
            seg_df = seg.df.copy()

            try:
                result = discovery_fn(
                    seg_df,
                    max_lag=params.max_lag,
                    alpha=params.alpha,
                    **params.extra_kwargs,
                )
                g = result.cg_tig[:d, :d, :]
                segment_graphs.append(g)
                segment_results.append(result)
            except Exception as e:
                if verbose:
                    logger.warning("Segment failed for %s: %s", regime_names[r], e)

        if not segment_graphs:
            continue

        per_regime_results[r] = segment_results

        # Within-regime aggregation: majority vote on 2D (any-lag) summary,
        # then populate 3D graph at all lags where any segment had the edge.
        max_lag_r = params.max_lag + 1
        padded = []
        for g in segment_graphs:
            if g.shape[2] < max_lag_r:
                pad = np.zeros((d, d, max_lag_r), dtype=g.dtype)
                pad[:, :, : g.shape[2]] = g
                padded.append(pad)
            else:
                padded.append(g[:, :, :max_lag_r])

        stacked = np.stack(padded, axis=0)
        n_seg = len(padded)
        thresh = max(1, int(config.segment_threshold * n_seg))

        # Vote on 2D: edge (i,j) present if any-lag detected in >= thresh segments
        has_edge_2d = (stacked[:, :, :, 1:] > 0).any(axis=3)  # (n_seg, d, d)
        edge_counts_2d = has_edge_2d.sum(axis=0)  # (d, d)
        accepted_2d = edge_counts_2d >= thresh  # (d, d)

        # Build 3D: for accepted edges, include all lags seen in any segment
        regime_graph = np.zeros((d, d, max_lag_r), dtype=np.int8)
        any_lag_union = (stacked > 0).any(axis=0)  # union of all segments' lags
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                if accepted_2d[i, j]:
                    regime_graph[i, j, :] = any_lag_union[i, j, :]
        # Self-loops: keep if present in majority
        for i in range(d):
            has_self = (stacked[:, i, i, 1:] > 0).any(axis=1)
            if has_self.sum() >= thresh:
                regime_graph[i, i, :] = any_lag_union[i, i, :]

        regime_graphs[r] = regime_graph

        if verbose:
            n_edges = int(regime_graph[:, :, 1:].any(axis=2).sum())
            logger.info(
                "%s: %d/%d segments succeeded, %d edges",
                regime_names[r],
                n_seg,
                len(segments),
                n_edges,
            )

    if not regime_graphs:
        # No regimes produced results — return empty graph
        max_lag_global = max(p.max_lag for p in regime_params) + 1
        return RegimeResult(
            cg_tig=np.zeros((d, d, max_lag_global), dtype=np.int8),
            var_names=var_names,
            regime_graphs={},
            regime_labels=labels,
            regime_names=regime_names,
            regime_params=regime_params,
            aggregation_strategy=config.aggregation,
            per_regime_results={},
            df=df,
        )

    # Cross-regime aggregation
    summary = aggregate(
        regime_graphs,
        strategy=config.aggregation,
        weights=regime_weights if config.aggregation == "weighted" else None,
    )

    return RegimeResult(
        cg_tig=summary,
        var_names=var_names,
        regime_graphs=regime_graphs,
        regime_labels=labels,
        regime_names=regime_names,
        regime_params=regime_params,
        aggregation_strategy=config.aggregation,
        per_regime_results=per_regime_results,
        df=df,
        metadata={"regime_weights": regime_weights},
    )
