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
    n_regimes : int, default 3
        Number of quantile bins. Labels are in ``[0, n_regimes)``.
    columns : list of int or str, optional
        Columns used to build the indicator. ``None`` (default) uses every
        column.
    stat : {"mean", "max", "sum"}, default "mean"
        Row summary that becomes the indicator.

    Notes
    -----
    The thresholds are the ``n_regimes - 1`` interior quantiles of the
    indicator, so the regimes hold roughly the same number of rows.

    Examples
    --------
    >>> import pandas as pd
    >>> from causalts.regime import QuantileRegimeDetector
    >>> df = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
    >>> QuantileRegimeDetector(n_regimes=3).assign(df)
    array([0, 0, 1, 1, 2, 2], dtype=int32)
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
        """Return the quantile-bin label of every row.

        Parameters
        ----------
        df : DataFrame of shape (T, d)
            Time series used to build the indicator.

        Returns
        -------
        ndarray of shape (T,)
            ``int32`` labels in ``[0, n_regimes)``.

        Raises
        ------
        ValueError
            If ``stat`` is not ``"mean"``, ``"max"`` or ``"sum"``.
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
    indicator : ndarray or str
        Per-row labels, or the name of a column that holds them.
    n_regimes : int, optional
        Declared number of regimes. Accepted for API symmetry; the labels are
        returned as given, not rebinned.
    """

    def __init__(self, indicator: np.ndarray | str, n_regimes: int | None = None):
        self._indicator = indicator
        self._n_regimes = n_regimes

    def assign(self, df: pd.DataFrame) -> np.ndarray:
        """Return the indicator labels as an ``int32`` array.

        Parameters
        ----------
        df : DataFrame of shape (T, d)
            Time series; only read when ``indicator`` is a column name.

        Returns
        -------
        ndarray of shape (T,)
            Labels in the range of the supplied indicator.
        """
        if isinstance(self._indicator, str):
            labels = df[self._indicator].values.astype(np.int32)
        else:
            labels = np.asarray(self._indicator, dtype=np.int32)
        return labels


class SeasonalRegimeDetector:
    """Assign regimes based on calendar season or month grouping.

    Parameters
    ----------
    grouping : {"season", "quarter", "month"}, default "season"
        Calendar grouping. ``season`` maps Dec-Feb to 0, Mar-May to 1,
        Jun-Aug to 2 and Sep-Nov to 3; ``quarter`` gives 0-3 and ``month``
        gives 0-11.

    Notes
    -----
    The frame must have a ``DatetimeIndex``; the values are ignored.
    """

    def assign(self, df: pd.DataFrame) -> np.ndarray:
        """Return the calendar label of every row.

        Parameters
        ----------
        df : DataFrame of shape (T, d)
            Time series with a ``DatetimeIndex``.

        Returns
        -------
        ndarray of shape (T,)
            ``int32`` labels in ``[0, 4)`` for ``season``/``quarter`` and
            ``[0, 12)`` for ``month``.

        Raises
        ------
        ValueError
            If ``df`` has no ``DatetimeIndex`` or ``grouping`` is unknown.
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
    n_regimes : int, default 3
        Number of quantile bins. Labels are in ``[0, n_regimes)``.
    window : int, default 50
        Rolling window in rows; ``min_periods`` is half of it.
    stat : {"variance", "mean"}, default "variance"
        Rolling statistic summarised across columns.

    Notes
    -----
    The rolling indicator is back-filled, then binned at its interior
    quantiles, like :class:`QuantileRegimeDetector`.
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
        """Return the changepoint-bin label of every row.

        Parameters
        ----------
        df : DataFrame of shape (T, d)
            Time series used for the rolling statistic.

        Returns
        -------
        ndarray of shape (T,)
            ``int32`` labels in ``[0, n_regimes)``.

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
