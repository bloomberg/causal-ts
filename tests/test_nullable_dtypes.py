# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""pandas nullable numeric dtypes (``Float64``, ``Int64``, ``pd.NA``) behave
like the ordinary ``float64`` frame with the same values."""

import numpy as np
import pandas as pd
import pytest

from causalts.baselines import run_baseline
from causalts.cdnots.phase3_utils import run_cdnots, run_cdnots_plus
from causalts.cedar.discovery import run_cedar
from causalts.ci_tests import create_ci_test
from causalts.inspection import discover_df, inspect_df
from causalts.utils.helpers import normalize_nullable_numeric

DTYPES = ["Float64", "Float32", "Int64", "UInt32"]
T = 200


def _frame(dtype, missing):
    rng = np.random.default_rng(0)
    if dtype.startswith("Float"):
        x = rng.normal(size=T)
        y = np.r_[0.0, 0.8 * x[:-1]] + rng.normal(size=T)
        z = rng.normal(size=T)
    else:
        # Integer valued and non-negative (UInt32); k has three levels.
        x = rng.integers(0, 100, T)
        y = np.r_[0, x[:-1]] + rng.integers(0, 30, T)
        z = rng.integers(0, 3, T)
    df = pd.DataFrame({"x": x, "y": y, "k": z}, index=pd.RangeIndex(10, 10 + T))
    df = df.astype(dtype)
    if missing:
        df.iloc[[3, 50], 0] = pd.NA
        df.iloc[120, 1] = pd.NA
    return df


def _same_graph(res, ref):
    np.testing.assert_array_equal(res.to_binary(), ref.to_binary())
    np.testing.assert_array_equal(res.to_marks(), ref.to_marks())


def _check_unchanged(df, before):
    pd.testing.assert_frame_equal(df, before)
    assert list(df.dtypes) == list(before.dtypes)


@pytest.fixture(params=DTYPES)
def dtype(request):
    return request.param


@pytest.mark.parametrize("missing", [False, True])
def test_inspect_df(dtype, missing):
    df = _frame(dtype, missing)
    before = df.copy()
    report = inspect_df(df)
    ref = inspect_df(df.astype("float64"))
    _check_unchanged(df, before)
    assert report["data"] == ref["data"]
    assert report["facts"] == ref["facts"]
    if missing:
        assert set(report["data"]["missing_by_col"]) == {"x", "y"}


@pytest.mark.parametrize("dtype", ["Int64", "UInt32"])
def test_inspect_df_detects_discrete_nullable_int(dtype):
    for missing in (False, True):
        assert "k" in inspect_df(_frame(dtype, missing))["data"]["discrete_cols"]


@pytest.mark.parametrize("algorithm", ["cdnots", "cedar", "lges"])
def test_discover_df_complete(dtype, algorithm):
    df = _frame(dtype, missing=False)
    before = df.copy()
    kw = {} if algorithm == "lges" else {"ci_test": "parcorr"}
    res = discover_df(df, algorithm=algorithm, max_lag=1, **kw)
    ref = discover_df(df.astype("float64"), algorithm=algorithm, max_lag=1, **kw)
    _check_unchanged(df, before)
    _same_graph(res, ref)


@pytest.mark.parametrize("impute", [None, "var_em"])
@pytest.mark.parametrize("algorithm", ["cdnots", "cedar"])
def test_discover_df_missing(dtype, algorithm, impute):
    df = _frame(dtype, missing=True)
    before = df.copy()
    kw = {"ci_test": "parcorr", "impute": impute}
    res = discover_df(df, algorithm=algorithm, max_lag=1, **kw)
    ref = discover_df(df.astype("float64"), algorithm=algorithm, max_lag=1, **kw)
    _check_unchanged(df, before)
    _same_graph(res, ref)


def test_lges_missing(dtype):
    df = _frame(dtype, missing=True)
    before = df.copy()
    for frame in (df, df.astype("float64")):
        with pytest.raises(ValueError, match="Impute first"):
            discover_df(frame, algorithm="lges", max_lag=1)
        with pytest.raises(ValueError, match="Impute first"):
            run_baseline("lges", frame, max_lag=1)
    res = discover_df(df, algorithm="lges", max_lag=1, impute="var_em")
    ref = discover_df(
        df.astype("float64"), algorithm="lges", max_lag=1, impute="var_em"
    )
    _check_unchanged(df, before)
    _same_graph(res, ref)


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("fn", [run_cdnots, run_cdnots_plus])
def test_run_cdnots_direct(dtype, missing, fn):
    df = _frame(dtype, missing)
    before = df.copy()
    ref_df = df.astype("float64")

    def run(frame):
        test = create_ci_test("parcorr", ref_df.to_numpy())
        return fn(df=frame, indep_test=test, num_lags=1)

    res, ref = run(df), run(ref_df)
    _check_unchanged(df, before)
    _same_graph(res, ref)


def test_run_cdnots_int64_column():
    # The issue's example: float columns plus one nullable integer column.
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(T, 3)), columns=["x", "y", "z"])
    df["k"] = pd.array(rng.integers(0, 3, T), dtype="Int64")
    before = df.copy()
    test = create_ci_test("parcorr", df.to_numpy(dtype=float))
    run_cdnots(df=df, indep_test=test, num_lags=1)
    _check_unchanged(df, before)


@pytest.mark.parametrize("missing", [False, True])
def test_run_cedar_direct(dtype, missing):
    df = _frame(dtype, missing)
    before = df.copy()
    ref_df = df.astype("float64")

    def run(frame):
        return run_cedar(
            df=frame, ci_test=create_ci_test("parcorr", ref_df.to_numpy()), max_lag=1
        )

    res, ref = run(df), run(ref_df)
    _check_unchanged(df, before)
    _same_graph(res, ref)


def test_run_baseline_direct(dtype):
    df = _frame(dtype, missing=False)
    before = df.copy()
    res = run_baseline("lges", df, max_lag=1)
    ref = run_baseline("lges", df.astype("float64"), max_lag=1)
    _check_unchanged(df, before)
    _same_graph(res, ref)


def test_normalize_only_nullable_numeric():
    df = pd.DataFrame(
        {
            "a": pd.array([1, None, 3], dtype="Int64"),
            "b": pd.arrays.SparseArray([0.0, 1.0, 0.0]),
            "c": pd.array([True, None, False], dtype="boolean"),
            "d": ["u", "v", "w"],
        }
    )
    out = normalize_nullable_numeric(df)
    assert out["a"].dtype == np.float64 and np.isnan(out["a"].iloc[1])
    assert list(out.dtypes[1:]) == list(df.dtypes[1:])
    assert df["a"].dtype == "Int64"
