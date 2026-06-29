# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""PELT-based regime detection with inline changepoint algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor

import numpy as np
import pandas as pd

from .segments import Segment


def _pelt_l2(signal: np.ndarray, min_size: int, pen: float, jump: int = 5) -> list[int]:
    """PELT algorithm with L2 (change in mean) cost.

    Finds the optimal set of changepoints that minimizes the sum of
    within-segment variances plus a penalty per changepoint.

    Parameters
    ----------
    signal : 1D array
        Univariate signal to segment.
    min_size : int
        Minimum segment length.
    pen : float
        Penalty per changepoint (higher = fewer changepoints).
    jump : int
        Grid resolution for candidate breakpoints. Only positions that
        are multiples of ``jump`` are considered. Default 5 (matches
        ruptures behaviour for speed and numerical consistency).

    Returns
    -------
    list[int]
        Sorted breakpoint indices (last element is always len(signal)).
    """
    n = len(signal)
    # Precompute cumulative sums for O(1) segment cost
    # Cost(start, end) = sum((x - mean)^2) = sum(x^2) - sum(x)^2 / length
    cumsum = np.zeros(n + 1)
    cumsum2 = np.zeros(n + 1)
    cumsum[1:] = np.cumsum(signal)
    cumsum2[1:] = np.cumsum(signal**2)

    def segment_cost(start: int, end: int) -> float:
        length = end - start
        s = cumsum[end] - cumsum[start]
        s2 = cumsum2[end] - cumsum2[start]
        return s2 - s * s / length

    # Candidate breakpoint positions (multiples of jump >= min_size, plus n)
    candidates = [k for k in range(0, n, jump) if k >= min_size]
    if candidates and candidates[-1] != n:
        candidates.append(n)
    elif not candidates:
        candidates = [n]

    # partitions[t] = dict {(start, end): cost} for optimal partition of [0:t]
    partitions: dict[int, dict] = {0: {(0, 0): 0}}
    admissible: list[int] = []

    for bkp in candidates:
        # Add new admissible point
        new_adm = floor((bkp - min_size) / jump) * jump
        admissible.append(new_adm)

        # Try all admissible start points
        subproblems = []
        for t in admissible:
            if t not in partitions:
                continue
            tmp = partitions[t].copy()
            tmp[(t, bkp)] = segment_cost(t, bkp) + pen
            subproblems.append(tmp)

        if not subproblems:
            continue

        # Best partition for [0:bkp]
        partitions[bkp] = min(subproblems, key=lambda d: sum(d.values()))
        best_val = sum(partitions[bkp].values())

        # Prune admissible set
        admissible = [
            t
            for t, sp in zip(admissible, subproblems)
            if sum(sp.values()) <= best_val + pen
        ]

    # Extract breakpoints from best partition
    best = partitions[n]
    del best[(0, 0)]
    breakpoints = sorted(set(end for _, end in best.keys()))
    return breakpoints


@dataclass
class _PeltSegment:
    start: int
    end: int
    regime: int
    mean_value: float


class PeltRegimeDetector:
    """Detect regime segments via PELT changepoint detection.

    Uses the PELT algorithm (Killick et al. 2012) with L2 cost to find
    optimal changepoint locations, then classifies each segment into a
    regime by its mean level (quantile-based on segment means).

    Unlike other detectors, this returns segment boundaries directly via
    ``detect_segments()`` — no ``extract_segments()`` step needed.

    Parameters
    ----------
    n_regimes : int
        Number of regimes to classify segments into (after PELT finds
        the boundaries).
    min_size : int
        Minimum segment length in samples.
    pen : float
        PELT penalty parameter. Higher = fewer changepoints.
        Recommended: 5000 for 3h-resolution river data (~10k samples).
    signal : str
        What to run PELT on: ``"mean"`` (row-mean of all columns),
        ``"volatility"`` (rolling std of row-mean).
    volatility_window : int
        Window size for rolling std when signal="volatility".
    columns : list or None
        Which columns to use for the signal. None = all columns.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        min_size: int = 500,
        pen: float = 5000,
        signal: str = "mean",
        volatility_window: int = 48,
        columns: list | None = None,
        **kwargs,
    ):
        self.n_regimes = n_regimes
        self.min_size = min_size
        self.pen = pen
        self.signal = signal
        self.volatility_window = volatility_window
        self.columns = columns

    def _get_signal(self, df: pd.DataFrame) -> np.ndarray:
        sub = df[self.columns] if self.columns else df
        indicator = sub.mean(axis=1).interpolate()

        if self.signal == "mean":
            return indicator.values
        elif self.signal == "volatility":
            roll_std = (
                indicator.rolling(
                    self.volatility_window,
                    min_periods=self.volatility_window // 2,
                )
                .std()
                .bfill()
            )
            return roll_std.values
        else:
            raise ValueError(f"Unknown signal: {self.signal!r}")

    def _run_pelt(self, signal: np.ndarray) -> list[int]:
        return _pelt_l2(signal, self.min_size, self.pen)

    def _classify_segments(
        self, signal: np.ndarray, breakpoints: list[int]
    ) -> list[_PeltSegment]:
        """Classify each PELT segment into a regime by quantile of means."""
        starts = [0] + breakpoints[:-1]
        ends = breakpoints

        seg_means = np.array([signal[s:e].mean() for s, e in zip(starts, ends)])

        thresholds = np.quantile(seg_means, np.linspace(0, 1, self.n_regimes + 1)[1:-1])
        regime_labels = np.digitize(seg_means, thresholds)

        return [
            _PeltSegment(start=s, end=e, regime=int(r), mean_value=float(m))
            for s, e, r, m in zip(starts, ends, regime_labels, seg_means)
        ]

    def assign(self, df: pd.DataFrame) -> np.ndarray:
        """Standard interface: per-timestep regime labels."""
        signal = self._get_signal(df)
        breakpoints = self._run_pelt(signal)
        pelt_segments = self._classify_segments(signal, breakpoints)

        labels = np.zeros(len(df), dtype=np.int32)
        for seg in pelt_segments:
            labels[seg.start : seg.end] = seg.regime
        return labels

    def detect_segments(self, df: pd.DataFrame) -> list[Segment]:
        """Direct segment interface — returns Segment objects ready for discovery.

        PELT finds boundaries, classifies each into a regime. Returns
        segments directly (no extract_segments step needed).
        """
        signal = self._get_signal(df)
        breakpoints = self._run_pelt(signal)
        pelt_segments = self._classify_segments(signal, breakpoints)

        return [
            Segment(
                df=df.iloc[seg.start : seg.end].copy(),
                regime=seg.regime,
                start_idx=seg.start,
                end_idx=seg.end,
            )
            for seg in pelt_segments
        ]
