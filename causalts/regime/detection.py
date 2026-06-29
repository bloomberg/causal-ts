# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regime detection strategies."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class RegimeDetector(Protocol):
    """Protocol for regime assignment strategies."""

    def assign(self, df: pd.DataFrame) -> np.ndarray:
        """Return int array of length T with labels in [0, n_regimes)."""
        ...


class QuantileRegimeDetector:
    """Assign regimes by quantile thresholds of a summary statistic."""

    def __init__(
        self,
        n_regimes: int = 3,
        columns: list[int | str] | None = None,
        stat: str = "mean",
    ):
        self.n_regimes = n_regimes
        self.columns = columns
        self.stat = stat

    def assign(self, df: pd.DataFrame) -> np.ndarray:
        if self.columns is not None:
            sub = df[self.columns]
        else:
            sub = df

        if self.stat == "mean":
            indicator = sub.mean(axis=1)
        elif self.stat == "max":
            indicator = sub.max(axis=1)
        elif self.stat == "sum":
            indicator = sub.sum(axis=1)
        else:
            raise ValueError(f"Unknown stat: {self.stat!r}")

        thresholds = np.nanquantile(
            indicator, np.linspace(0, 1, self.n_regimes + 1)[1:-1]
        )
        return np.digitize(indicator, thresholds).astype(np.int32)


class ColumnRegimeDetector:
    """Assign regimes from an explicit indicator array or column name."""

    def __init__(self, indicator: np.ndarray | str, n_regimes: int | None = None):
        self._indicator = indicator
        self._n_regimes = n_regimes

    def assign(self, df: pd.DataFrame) -> np.ndarray:
        if isinstance(self._indicator, str):
            labels = df[self._indicator].values.astype(np.int32)
        else:
            labels = np.asarray(self._indicator, dtype=np.int32)
        return labels


class SeasonalRegimeDetector:
    """Assign regimes based on calendar season or month grouping."""

    _SEASON_MAP = {
        12: 0,
        1: 0,
        2: 0,
        3: 1,
        4: 1,
        5: 1,
        6: 2,
        7: 2,
        8: 2,
        9: 3,
        10: 3,
        11: 3,
    }

    def __init__(self, grouping: str = "season"):
        self.grouping = grouping

    def assign(self, df: pd.DataFrame) -> np.ndarray:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("SeasonalRegimeDetector requires DatetimeIndex")

        if self.grouping == "season":
            return np.array(
                [self._SEASON_MAP[m] for m in df.index.month], dtype=np.int32
            )
        elif self.grouping == "month":
            return (df.index.month.values - 1).astype(np.int32)
        elif self.grouping == "quarter":
            return (df.index.quarter.values - 1).astype(np.int32)
        else:
            raise ValueError(f"Unknown grouping: {self.grouping!r}")


class ChangePointRegimeDetector:
    """Detect regimes via rolling statistic changepoints."""

    def __init__(
        self,
        n_regimes: int = 3,
        window: int = 50,
        stat: str = "variance",
    ):
        self.n_regimes = n_regimes
        self.window = window
        self.stat = stat

    def assign(self, df: pd.DataFrame) -> np.ndarray:
        if self.stat == "variance":
            rolling = df.rolling(self.window, min_periods=self.window // 2).var()
            indicator = rolling.mean(axis=1).fillna(method="bfill")
        elif self.stat == "mean":
            rolling = df.rolling(self.window, min_periods=self.window // 2).mean()
            indicator = rolling.mean(axis=1).fillna(method="bfill")
        else:
            raise ValueError(f"Unknown stat: {self.stat!r}")

        thresholds = np.nanquantile(
            indicator, np.linspace(0, 1, self.n_regimes + 1)[1:-1]
        )
        return np.digitize(indicator, thresholds).astype(np.int32)
