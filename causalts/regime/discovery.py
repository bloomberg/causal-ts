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
from .result import RegimeResult, count_edges
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
        Within a regime, a lagged edge ``i -> j`` (at any lag >= 1) is kept,
        at every lag any segment found, when enough segments find it: by
        default ``max(1, int(segment_threshold * n_segments))``, or a share of
        at least ``segment_threshold`` with ``threshold_rounding="ceil"``. A
        lag-0 adjacency needs the same count, either orientation counting; it
        is oriented ``i -> j`` when the segments that orient it that way
        (unoriented ones excluded) reach the count, and is left unoriented
        (symmetric 1s) otherwise, including when both directions reach it
        (possible under the default rounding, e.g. 1 of 3 segments each way).
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

        # Within-regime aggregation: a vote across segments, lagged edges and
        # lag-0 adjacencies voted separately (see below).
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
        if config.threshold_rounding == "ceil":
            # The fewest votes k with k / n_seg >= segment_threshold, i.e. a
            # share of at least segment_threshold (0.5 of 3 is 2; of 4, a tie of
            # 2 passes). Comparing the fraction, not ceil(threshold * n_seg),
            # avoids float artefacts both ways: 0.28 * 25 is 7.000000000000001,
            # and a threshold given as 5 / 6 must accept 5 of 6.
            thresh = next(
                (
                    k
                    for k in range(1, n_seg + 1)
                    if k / n_seg >= config.segment_threshold
                ),
                n_seg + 1,  # a threshold above 1 accepts nothing
            )
        elif config.threshold_rounding == "floor":  # the default: 0.5 of 3 is 1
            thresh = max(1, int(config.segment_threshold * n_seg))
        else:  # set after construction, past RegimeConfig's own check
            raise ValueError(
                "threshold_rounding must be 'floor' or 'ceil', got "
                f"{config.threshold_rounding!r}"
            )

        present = stacked > 0
        regime_graph = np.zeros((d, d, max_lag_r), dtype=np.int8)

        # Lagged edges (self-loops included): time fixes the direction, so vote
        # on i -> j at any lag >= 1, then keep every lag any segment found.
        has_lagged = present[:, :, :, 1:].any(axis=3)  # (n_seg, d, d)
        accepted_lagged = has_lagged.sum(axis=0) >= thresh  # (d, d)
        lagged_union = present[:, :, :, 1:].any(axis=0)  # (d, d, max_lag)
        regime_graph[:, :, 1:] = lagged_union & accepted_lagged[:, :, None]

        # Lag-0 edges: orientation is the least stable thing across segments,
        # so vote on the adjacency (either orientation, or unoriented as
        # symmetric 1s), then orient it by direction votes. An adjacency with
        # no direction reaching the threshold stays unoriented (symmetric 1s).
        lag0 = present[:, :, :, 0].copy()
        lag0[:, np.arange(d), np.arange(d)] = False  # a lag-0 self-loop is no edge
        # Only one-way entries vote for a direction; an unoriented edge votes for
        # the adjacency alone.
        dir_votes = (lag0 & ~lag0.transpose(0, 2, 1)).sum(axis=0)  # (d, d)
        adj_accepted = (lag0 | lag0.transpose(0, 2, 1)).sum(axis=0) >= thresh
        oriented = adj_accepted & (dir_votes >= thresh)
        unoriented = adj_accepted & ~oriented & ~oriented.T
        regime_graph[:, :, 0] = oriented | unoriented

        regime_graphs[r] = regime_graph

        if verbose:
            n_edges = count_edges(regime_graph)
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
