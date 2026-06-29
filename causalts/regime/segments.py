# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Contiguous segment extraction for regime-conditional discovery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Segment:
    """A contiguous time segment belonging to a single regime."""

    df: pd.DataFrame
    regime: int
    start_idx: int
    end_idx: int


def extract_segments(
    df: pd.DataFrame,
    labels: np.ndarray,
    regime: int,
    min_length: int,
    buffer: int = 0,
) -> list[Segment]:
    """Extract contiguous runs where labels == regime.

    Parameters
    ----------
    df : DataFrame
        Full time series.
    labels : array of int, shape (T,)
        Regime label per timestep.
    regime : int
        Which regime to extract.
    min_length : int
        Minimum segment length (excluding buffer).
    buffer : int
        Samples prepended before segment start for lag context.

    Returns
    -------
    list[Segment]
    """
    mask = labels == regime
    segments: list[Segment] = []
    in_segment = False
    start = 0

    for i in range(len(mask)):
        if mask[i] and not in_segment:
            start = i
            in_segment = True
        elif not mask[i] and in_segment:
            buf_start = max(0, start - buffer)
            seg_df = df.iloc[buf_start:i].copy()
            if len(seg_df) >= min_length:
                segments.append(Segment(seg_df, regime, buf_start, i))
            in_segment = False

    if in_segment:
        buf_start = max(0, start - buffer)
        seg_df = df.iloc[buf_start : len(df)].copy()
        if len(seg_df) >= min_length:
            segments.append(Segment(seg_df, regime, buf_start, len(df)))

    return segments
