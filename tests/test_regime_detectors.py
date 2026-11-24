# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the simple regime detectors in ``causalts.regime.detection``."""

import numpy as np
import pandas as pd
import pytest

from causalts.regime import (
    ColumnRegimeDetector,
    QuantileRegimeDetector,
    SeasonalRegimeDetector,
)


def _assert_labels(labels, length, n_regimes):
    assert isinstance(labels, np.ndarray)
    assert labels.dtype == np.int32
    assert labels.shape == (length,)
    assert labels.min() >= 0
    assert labels.max() < n_regimes


# ── QuantileRegimeDetector ───────────────────────────────────────────────────
def test_quantile_detector_bins_a_known_series():
    df = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})

    labels = QuantileRegimeDetector(n_regimes=3).assign(df)

    _assert_labels(labels, 6, 3)
    np.testing.assert_array_equal(labels, [0, 0, 1, 1, 2, 2])


def test_quantile_detector_uses_only_the_named_columns():
    df = pd.DataFrame({"used": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "ignored": 1e6})

    labels = QuantileRegimeDetector(n_regimes=2, columns=["used"]).assign(df)

    _assert_labels(labels, 6, 2)
    np.testing.assert_array_equal(labels, [0, 0, 0, 1, 1, 1])


@pytest.mark.parametrize("stat", ["mean", "max", "sum"])
def test_quantile_detector_supports_each_statistic(stat):
    df = pd.DataFrame({"x": np.arange(30, dtype=float), "y": np.arange(30, dtype=float)})

    labels = QuantileRegimeDetector(n_regimes=4, stat=stat).assign(df)

    _assert_labels(labels, 30, 4)


def test_quantile_detector_rejects_an_unknown_statistic():
    with pytest.raises(ValueError, match="Unknown stat"):
        QuantileRegimeDetector(stat="median").assign(pd.DataFrame({"x": [1.0, 2.0]}))


# ── ColumnRegimeDetector ─────────────────────────────────────────────────────
def test_column_detector_reads_a_column_name():
    df = pd.DataFrame({"regime": [0, 1, 1, 0]})

    labels = ColumnRegimeDetector("regime").assign(df)

    _assert_labels(labels, 4, 2)
    np.testing.assert_array_equal(labels, [0, 1, 1, 0])


def test_column_detector_reads_an_explicit_array():
    indicator = np.array([2, 0, 1])

    labels = ColumnRegimeDetector(indicator).assign(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))

    _assert_labels(labels, 3, 3)
    np.testing.assert_array_equal(labels, indicator)


# ── SeasonalRegimeDetector ───────────────────────────────────────────────────
def _year_frame(year=2026):
    index = pd.date_range(f"{year}-01-01", periods=12, freq="MS")
    return pd.DataFrame({"x": np.arange(12, dtype=float)}, index=index)


def test_seasonal_detector_maps_seasons():
    labels = SeasonalRegimeDetector(grouping="season").assign(_year_frame())

    _assert_labels(labels, 12, 4)
    np.testing.assert_array_equal(labels, [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0])


def test_seasonal_detector_maps_months():
    labels = SeasonalRegimeDetector(grouping="month").assign(_year_frame())

    _assert_labels(labels, 12, 12)
    np.testing.assert_array_equal(labels, np.arange(12))


def test_seasonal_detector_maps_quarters():
    labels = SeasonalRegimeDetector(grouping="quarter").assign(_year_frame())

    _assert_labels(labels, 12, 4)
    np.testing.assert_array_equal(labels, [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])


def test_seasonal_detector_requires_a_datetime_index():
    with pytest.raises(ValueError, match="DatetimeIndex"):
        SeasonalRegimeDetector().assign(pd.DataFrame({"x": [1.0, 2.0]}))


def test_seasonal_detector_rejects_an_unknown_grouping():
    with pytest.raises(ValueError, match="Unknown grouping"):
        SeasonalRegimeDetector(grouping="week").assign(_year_frame())
