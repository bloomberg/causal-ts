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
    """Assign regimes by quantile thresholds of a summary statistic.

    Parameters
    ----------
    n_regimes : int
        Number of quantile bins. Labels run from 0 to ``n_regimes - 1``.
    columns : list of int or str or None
        Columns to summarise. None = all columns.
    stat : str
        Row summary to bin: ``"mean"``, ``"max"`` or ``"sum"``.
    """

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
        """Label each row by the quantile bin of its row summary.

        Parameters
        ----------
        df : pd.DataFrame
            Data with one row per time step.

        Returns
        -------
        np.ndarray
            ``int32`` array of length T with labels in ``[0, n_regimes)``.

        Raises
        ------
        ValueError
            If ``stat`` is not one of ``"mean"``, ``"max"`` or ``"sum"``.
        """
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
    """Assign regimes from an explicit indicator array or column name.

    Parameters
    ----------
    indicator : np.ndarray or str
        One integer label per row, or the name of a column in ``df`` that
        holds them.
    n_regimes : int or None
        Accepted for interface parity; ``assign`` does not use it.
    """

    def __init__(self, indicator: np.ndarray | str, n_regimes: int | None = None):
        self._indicator = indicator
        self._n_regimes = n_regimes

    def assign(self, df: pd.DataFrame) -> np.ndarray:
        """Return the indicator as ``int32`` labels, one per row.

        Parameters
        ----------
        df : pd.DataFrame
            Data with one row per time step. Read only when ``indicator`` is a
            column name.

        Returns
        -------
        np.ndarray
            ``int32`` array of length T. Labels are taken as given, so they
            lie in ``[0, n_regimes)`` only if the indicator does.
        """
        if isinstance(self._indicator, str):
            labels = df[self._indicator].values.astype(np.int32)
        else:
            labels = np.asarray(self._indicator, dtype=np.int32)
        return labels


class SeasonalRegimeDetector:
    """Assign regimes based on calendar season or month grouping.

    The index of ``df`` must be a ``pd.DatetimeIndex``.

    Parameters
    ----------
    grouping : str
        ``"season"`` gives 4 labels: 0 = Dec-Feb, 1 = Mar-May, 2 = Jun-Aug,
        3 = Sep-Nov. ``"quarter"`` gives 4 labels: 0-3 for Q1-Q4. ``"month"``
        gives 12 labels: 0 = January, ..., 11 = December.
    """

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
        """Label each row by its calendar group.

        Parameters
        ----------
        df : pd.DataFrame
            Data indexed by ``pd.DatetimeIndex``.

        Returns
        -------
        np.ndarray
            ``int32`` array of length T with labels in ``[0, 4)`` for
            ``"season"`` and ``"quarter"``, or ``[0, 12)`` for ``"month"``.

        Raises
        ------
        ValueError
            If the index is not a ``DatetimeIndex`` or ``grouping`` is unknown.
        """
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
    """Detect regimes via rolling statistic changepoints.

    Parameters
    ----------
    n_regimes : int
        Number of quantile bins for the rolling statistic. Labels run from 0
        to ``n_regimes - 1``.
    window : int
        Rolling window length. Leading rows with fewer than ``window // 2``
        observations take the first defined value.
    stat : str
        Rolling statistic to bin: ``"variance"`` or ``"mean"``.
    """

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
        """Label each row by the quantile bin of its rolling statistic.

        Parameters
        ----------
        df : pd.DataFrame
            Data with one row per time step.

        Returns
        -------
        np.ndarray
            ``int32`` array of length T with labels in ``[0, n_regimes)``.

        Raises
        ------
        ValueError
            If ``stat`` is not ``"variance"`` or ``"mean"``.
        """
        if self.stat == "variance":
            rolling = df.rolling(self.window, min_periods=self.window // 2).var()
            indicator = rolling.mean(axis=1).bfill()
        elif self.stat == "mean":
            rolling = df.rolling(self.window, min_periods=self.window // 2).mean()
            indicator = rolling.mean(axis=1).bfill()
        else:
            raise ValueError(f"Unknown stat: {self.stat!r}")

        thresholds = np.nanquantile(
            indicator, np.linspace(0, 1, self.n_regimes + 1)[1:-1]
        )
        return np.digitize(indicator, thresholds).astype(np.int32)
