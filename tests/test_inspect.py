# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for inspect_df — the JSON contract and data-health warnings."""

import numpy as np
import pandas as pd
import pytest

from causalts.inspection import inspect_df

_COST_CLASSES = {"cheap", "moderate", "expensive"}


def _linear_var(T=300, d=4, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((T, d))
    for t in range(1, T):
        data[t, 1] += 0.6 * data[t - 1, 0]
        data[t, 2] += 0.5 * data[t - 1, 1]
    return pd.DataFrame(data, columns=[f"X{i}" for i in range(d)])


def test_schema_shape():
    report = inspect_df(_linear_var())
    for key in (
        "schema_version",
        "data",
        "facts",
        "recommendation",
        "cost_class",
        "warnings",
    ):
        assert key in report
    assert report["schema_version"] == 1
    assert report["cost_class"] in _COST_CLASSES
    d = report["data"]
    assert d["n_vars"] == 4 and d["n_rows"] == 300
    assert set(
        ("algorithm", "ci_test", "include_C", "c_preset", "max_lag", "rationale")
    ) <= set(report["recommendation"])
    facts = report["facts"]
    assert "linearity" in facts and "stationarity" in facts
    assert facts["stationarity"]["form"] in {"seasonal", "curved", "trend", "none"}


def test_missing_column_warns():
    df = _linear_var()
    df.loc[df.index[: len(df) // 2], "X3"] = np.nan  # 50% missing
    report = inspect_df(df)
    assert any("X3" in w for w in report["warnings"])
    assert "X3" in report["data"]["missing_by_col"]


def test_constant_column_warns():
    df = _linear_var()
    df["X0"] = 1.0  # constant
    report = inspect_df(df)
    assert "X0" in report["data"]["constant_cols"]
    assert any("Constant" in w or "constant" in w for w in report["warnings"])


def test_max_lag_override():
    report = inspect_df(_linear_var(), max_lag=7)
    assert report["facts"]["suggested_max_lag"] == 7
    assert report["recommendation"]["max_lag"] == 7


def test_json_serialisable():
    import json

    report = inspect_df(_linear_var())
    json.dumps(report, default=str)  # must not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
