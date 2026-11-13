# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Behaviour tests for graph-informed forecasting (causalts.effects.forecast)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from causalts.effects.forecast import (
    CausalForecaster,
    _ARWrapper,
    _build_feature_matrix,
    _detect_seasonal_period,
    _fit_auto_ets,
    get_causal_parents,
)


def _var_data(T=400, seed=0):
    """x(t) = 0.5 x(t-1) + e;  y(t) = 0.8 x(t-1) - 0.3 y(t-1) + 0.1 e."""
    rng = np.random.default_rng(seed)
    x = np.zeros(T)
    y = np.zeros(T)
    for t in range(1, T):
        x[t] = 0.5 * x[t - 1] + rng.standard_normal()
        y[t] = 0.8 * x[t - 1] - 0.3 * y[t - 1] + 0.1 * rng.standard_normal()
    graph = np.zeros((2, 2, 2), dtype=np.int8)
    graph[0, 0, 1] = 1  # x(t-1) -> x(t)
    graph[0, 1, 1] = 1  # x(t-1) -> y(t)
    graph[1, 1, 1] = 1  # y(t-1) -> y(t)
    return graph, pd.DataFrame({"x": x, "y": y})


# ---------------------------------------------------------------------------
# get_causal_parents
# ---------------------------------------------------------------------------


def test_get_causal_parents_by_index_and_name():
    g = np.zeros((3, 3, 3), dtype=int)
    g[0, 2, 1] = 1
    g[1, 2, 0] = 1
    g[2, 2, 2] = 1  # lagged self-loop is a real parent
    g[2, 2, 0] = 1  # lag-0 self-loop is not
    names = ["a", "b", "c"]

    expected = [("a", 1), ("b", 0), ("c", 2)]
    assert get_causal_parents(g, 2, names) == expected
    assert get_causal_parents(g, "c", names) == expected
    assert get_causal_parents(g, 0, names) == []


def test_get_causal_parents_default_names():
    g = np.zeros((2, 2, 2), dtype=int)
    g[0, 1, 1] = 1
    assert get_causal_parents(g, 1) == [("X0", 1)]
    assert get_causal_parents(g, "X1") == [("X0", 1)]
    with pytest.raises(ValueError):
        get_causal_parents(g, "missing")


def test_get_causal_parents_max_lag_zero():
    g = np.zeros((2, 2, 1), dtype=int)
    g[0, 1, 0] = 1
    assert get_causal_parents(g, 1, ["a", "b"]) == [("a", 0)]


# ---------------------------------------------------------------------------
# _build_feature_matrix
# ---------------------------------------------------------------------------


def test_build_feature_matrix_columns_and_alignment():
    df = pd.DataFrame(
        {"x": np.arange(6.0), "y": np.arange(6.0) * 10, "e": -np.arange(6.0)}
    )
    X, y = _build_feature_matrix(df, [("x", 2), ("y", 0)], "y", ["e"])

    assert list(X.columns) == ["x_lag2", "y", "exog_e"]
    assert len(X) == len(y) == 4  # first two rows lost to the lag-2 shift
    np.testing.assert_array_equal(X["x_lag2"], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(y, [20.0, 30.0, 40.0, 50.0])
    np.testing.assert_array_equal(X["exog_e"], [-2.0, -3.0, -4.0, -5.0])


def test_build_feature_matrix_drops_rows_with_nan_target():
    df = pd.DataFrame({"x": np.arange(5.0), "y": [0.0, 1.0, np.nan, 3.0, 4.0]})
    X, y = _build_feature_matrix(df, [("x", 1)], "y")
    np.testing.assert_array_equal(y, [1.0, 3.0, 4.0])
    np.testing.assert_array_equal(X["x_lag1"], [0.0, 2.0, 3.0])


def test_build_feature_matrix_requires_predictors():
    df = pd.DataFrame({"y": np.arange(5.0)})
    with pytest.raises(ValueError, match="No causal parents found for 'y'"):
        _build_feature_matrix(df, [], "y")


# ---------------------------------------------------------------------------
# Univariate fallback helpers
# ---------------------------------------------------------------------------


def test_detect_seasonal_period_finds_planted_period():
    t = np.arange(240)
    rng = np.random.default_rng(0)
    y = np.sin(2 * np.pi * t / 12) + 0.05 * rng.standard_normal(240)
    assert _detect_seasonal_period(y) == 12


def test_detect_seasonal_period_none_for_noise_and_short_series():
    rng = np.random.default_rng(1)
    assert _detect_seasonal_period(rng.standard_normal(300)) is None
    assert _detect_seasonal_period(np.arange(9.0)) is None


def test_fit_auto_ets_extrapolates_linear_trend():
    rng = np.random.default_rng(2)
    t = np.arange(120.0)
    y = 0.5 * t + 0.1 * rng.standard_normal(120)
    model = _fit_auto_ets(y)
    fc = np.asarray(model.forecast(10))
    truth = 0.5 * np.arange(120.0, 130.0)
    assert fc.shape == (10,)
    np.testing.assert_allclose(fc, truth, atol=1.0)


def test_fit_auto_ets_short_series_falls_back_to_ar1():
    y = np.array([1.0, 2.0, 1.5, 2.5, 2.0])
    model = _fit_auto_ets(y)
    assert isinstance(model, _ARWrapper)
    assert np.isfinite(model.aic)
    fv = model.fittedvalues
    assert fv.shape == (5,)
    assert np.isnan(fv[0]) and np.isfinite(fv[1:]).all()
    assert np.isfinite(model.forecast(3)).all()


# ---------------------------------------------------------------------------
# CausalForecaster: fitting and in-sample behaviour
# ---------------------------------------------------------------------------


def test_fit_recovers_var_coefficients():
    graph, df = _var_data()
    fc = CausalForecaster(graph, var_names=df.columns).fit(df)

    assert set(fc.models_) == {"x", "y"}
    assert fc.parents_["y"] == [("x", 1), ("y", 1)]
    np.testing.assert_allclose(fc.models_["y"].coef_, [0.8, -0.3], atol=0.05)
    np.testing.assert_allclose(fc.models_["x"].coef_, [0.5], atol=0.1)

    imp = fc.feature_importance("y")
    assert list(imp.columns) == ["parent", "lag", "importance"]
    assert list(imp["parent"]) == ["x", "y"]  # sorted by |coef|, descending
    assert list(imp["lag"]) == [1, 1]
    np.testing.assert_allclose(imp["importance"], [0.8, 0.3], atol=0.05)


def test_predict_beats_persistence_out_of_sample():
    graph, df = _var_data(T=500, seed=3)
    train, test = df.iloc[:400], df.iloc[400:]
    fc = CausalForecaster(graph, var_names=df.columns).fit(train)

    pred = fc.predict(test, targets=["y"])
    assert list(pred.columns) == ["y"]
    assert pred.index.equals(test.index)
    assert np.isnan(pred["y"].iloc[0])  # no lag available for the first row

    y_true = test["y"].to_numpy()[1:]
    mse_model = np.mean((pred["y"].to_numpy()[1:] - y_true) ** 2)
    mse_naive = np.mean((test["y"].to_numpy()[:-1] - y_true) ** 2)
    assert mse_model < 0.1 * mse_naive

    scores = fc.score(test)
    assert scores["y"] > 0.9
    assert 0.0 < scores["x"] < 0.6  # AR(1) with phi=0.5: R^2 ~ 0.25


def test_default_targets_and_univariate_fallback():
    rng = np.random.default_rng(4)
    T = 200
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = 0.8 * x[t - 1] + rng.standard_normal()
    y = np.r_[0.0, 1.5 * x[:-1]] + 0.1 * rng.standard_normal(T)
    df = pd.DataFrame({"x": x, "y": y})
    g = np.zeros((2, 2, 2), dtype=int)
    g[0, 1, 1] = 1  # only y has a parent

    fc = CausalForecaster(g, var_names=["x", "y"]).fit(df)
    assert list(fc.models_) == ["y"]
    assert list(fc.exog_models_) == ["x"]  # parentless root gets AR/ETS
    assert repr(fc) == "CausalForecaster(causal=1, exog_ets=1, model=LinearRegression)"

    pred = fc.predict(df)
    assert list(pred.columns) == ["x", "y"]
    assert pred["x"].notna().sum() > T // 2
    assert fc.score(df)["x"] > 0.4  # AR(1) with phi=0.8: R^2 ~ 0.64

    no_fb = CausalForecaster(g, var_names=["x", "y"], exog_fallback=False).fit(df)
    assert no_fb.exog_models_ == {}
    with pytest.raises(ValueError, match="No model fitted for 'x'"):
        no_fb.predict(df, targets=["x"])


def test_fallback_skips_series_shorter_than_four():
    g = np.zeros((2, 2, 2), dtype=int)
    g[0, 1, 1] = 1
    df = pd.DataFrame(
        {"x": [1.0, np.nan, 2.0, np.nan, np.nan, 3.0], "y": np.arange(6.0)}
    )
    fc = CausalForecaster(g, var_names=["x", "y"]).fit(df)
    assert "x" not in fc.exog_models_


def test_include_contemp_false_drops_lag0_parents():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(100)
    df = pd.DataFrame({"x": x, "y": 2.0 * x, "z": np.r_[0.0, x[:-1]]})
    g = np.zeros((3, 3, 2), dtype=int)
    g[0, 1, 0] = 1  # x(t) -> y(t), contemporaneous only
    g[0, 2, 1] = 1  # x(t-1) -> z(t)

    with_c = CausalForecaster(g, var_names=["x", "y", "z"], exog_fallback=False)
    with_c.fit(df)
    assert with_c.parents_["y"] == [("x", 0)]
    np.testing.assert_allclose(with_c.models_["y"].coef_, [2.0])

    no_c = CausalForecaster(
        g, var_names=["x", "y", "z"], include_contemp=False, exog_fallback=False
    ).fit(df)
    assert "y" not in no_c.models_
    assert no_c.parents_["z"] == [("x", 1)]


def test_model_template_is_cloned_not_fitted():
    graph, df = _var_data(T=150)
    template = LinearRegression()
    fc = CausalForecaster(graph, var_names=df.columns, model=template).fit(df)
    assert not hasattr(template, "coef_")
    assert fc.models_["x"] is not fc.models_["y"]
    assert fc.models_["x"] is not template


def test_feature_importance_tree_and_unsupported_models():
    graph, df = _var_data(T=150)
    rf = CausalForecaster(
        graph,
        var_names=df.columns,
        model=RandomForestRegressor(n_estimators=10, random_state=0),
    ).fit(df)
    imp = rf.feature_importance("y")
    assert set(imp["parent"]) == {"x", "y"}
    assert imp["importance"].sum() == pytest.approx(1.0)
    # x(t-1) carries most of the signal for y(t).
    assert imp["parent"].iloc[0] == "x"

    knn = CausalForecaster(
        graph, var_names=df.columns, model=KNeighborsRegressor(n_neighbors=3)
    ).fit(df)
    imp = knn.feature_importance("y")
    assert len(imp) == 2
    assert imp["importance"].isna().all()

    with pytest.raises(ValueError, match="No model fitted for 'nope'"):
        knn.feature_importance("nope")


# ---------------------------------------------------------------------------
# CausalForecaster: multi-step forecasting
# ---------------------------------------------------------------------------


def test_forecast_horizon_requires_lagged_only_models():
    graph, df = _var_data(T=100)
    fc = CausalForecaster(graph, var_names=df.columns).fit(df)
    with pytest.raises(ValueError, match="include_contemp=False"):
        fc.forecast_horizon(df, horizon=3)


def test_forecast_horizon_iterates_fitted_equations():
    graph, df = _var_data(T=300, seed=6)
    fc = CausalForecaster(graph, var_names=df.columns, include_contemp=False)
    fc.fit(df)

    out = fc.forecast_horizon(df, horizon=4)
    assert out.shape == (4, 2)
    assert list(out.columns) == ["x", "y"]

    # Reproduce the recursion by hand from the fitted coefficients.
    mx, my = fc.models_["x"], fc.models_["y"]
    x_prev, y_prev = df["x"].iloc[-1], df["y"].iloc[-1]
    for h in range(4):
        x_new = mx.intercept_ + mx.coef_[0] * x_prev
        y_new = my.intercept_ + my.coef_[0] * x_prev + my.coef_[1] * y_prev
        assert out["x"].iloc[h] == pytest.approx(x_new)
        assert out["y"].iloc[h] == pytest.approx(y_new)
        x_prev, y_prev = x_new, y_new


def test_forecast_horizon_uses_univariate_fallback_for_roots():
    rng = np.random.default_rng(7)
    T = 300
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = 0.7 * x[t - 1] + rng.standard_normal()
    df = pd.DataFrame({"x": x})
    g = np.zeros((1, 1, 2), dtype=int)  # no parents at all
    fc = CausalForecaster(g, var_names=["x"], include_contemp=False).fit(df)
    assert fc.models_ == {}
    assert list(fc.exog_models_) == ["x"]

    out = fc.forecast_horizon(df, horizon=20)
    assert out.shape == (20, 1)
    expected = np.asarray(fc.exog_models_["x"].forecast(20))
    np.testing.assert_allclose(out["x"].to_numpy(), expected)
    assert np.isfinite(expected).all()


def test_forecast_horizon_nan_when_history_shorter_than_lag():
    g = np.zeros((2, 2, 4), dtype=int)
    g[0, 1, 3] = 1
    rng = np.random.default_rng(8)
    df = pd.DataFrame({"x": rng.standard_normal(50), "y": rng.standard_normal(50)})
    fc = CausalForecaster(
        g, var_names=["x", "y"], include_contemp=False, exog_fallback=False
    ).fit(df)
    out = fc.forecast_horizon(df.iloc[:2], horizon=1)
    assert np.isnan(out["y"].iloc[0])


# ---------------------------------------------------------------------------
# Exogenous regressors
# ---------------------------------------------------------------------------


def _exog_data(T=200, seed=9):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(T)
    x = rng.standard_normal(T)
    y = np.r_[0.0, x[:-1]] + 2.0 * e + 0.05 * rng.standard_normal(T)
    g = np.zeros((2, 2, 2), dtype=int)
    g[0, 1, 1] = 1  # x(t-1) -> y(t)
    return g, pd.DataFrame({"x": x, "y": y}), pd.DataFrame({"e": e})


def test_exog_fit_predict_and_score():
    g, df, exog = _exog_data()
    fc = CausalForecaster(g, var_names=["x", "y"]).fit(df, exog=exog)

    # With exog every variable gets a regression model; no AR/ETS fallback.
    assert set(fc.models_) == {"x", "y"}
    assert fc.exog_models_ == {}
    assert fc.exog_cols_ == ["e"]
    assert "exog_features=1" in repr(fc)
    np.testing.assert_allclose(fc.models_["y"].coef_, [1.0, 2.0], atol=0.05)
    imp = fc.feature_importance("y")
    assert imp["parent"].iloc[0] == "exog_e"
    assert imp["lag"].iloc[0] == 0

    assert fc.score(df, exog=exog)["y"] > 0.99
    pred = fc.predict(df, exog=exog)
    np.testing.assert_allclose(pred["y"].iloc[1:], df["y"].iloc[1:], atol=0.25)

    with pytest.raises(ValueError, match="provide exog= to predict"):
        fc.predict(df)
    with pytest.raises(ValueError, match="provide exog= to score"):
        fc.score(df)


def test_exog_forecast_horizon_uses_future_values():
    g, df, exog = _exog_data()
    fc = CausalForecaster(g, var_names=["x", "y"], include_contemp=False)
    fc.fit(df, exog=exog)

    with pytest.raises(ValueError, match="provide exog_future"):
        fc.forecast_horizon(df, horizon=2)
    with pytest.raises(ValueError, match="at least 3 rows"):
        fc.forecast_horizon(df, horizon=3, exog_future=pd.DataFrame({"e": [1.0]}))

    future = pd.DataFrame({"e": [1.0, -1.0, 0.0]})
    out = fc.forecast_horizon(df, horizon=3, exog_future=future)
    # y(T+1) = x(T) + 2 e(T+1); x has no parents so its forecast is its mean ~ 0.
    assert out["y"].iloc[0] == pytest.approx(df["x"].iloc[-1] + 2.0, abs=0.15)
    assert out["y"].iloc[1] == pytest.approx(-2.0, abs=0.3)
    assert out["y"].iloc[2] == pytest.approx(0.0, abs=0.3)


def test_predict_stays_aligned_after_interior_nan():
    # Regression: predictions were written as one contiguous block after
    # dropping NaN rows, shifting every prediction after a gap by one row.
    rng = np.random.default_rng(10)
    x = rng.standard_normal(60)
    df = pd.DataFrame({"x": x, "y": np.r_[0.0, 2.0 * x[:-1]]})
    g = np.zeros((2, 2, 2), dtype=int)
    g[0, 1, 1] = 1
    fc = CausalForecaster(g, var_names=["x", "y"], exog_fallback=False).fit(df)

    gappy = df.copy()
    gappy.loc[10, "x"] = np.nan  # x(10) feeds y(11) only
    gappy.loc[30, "y"] = np.nan  # an unknown target value needs no feature
    pred = fc.predict(gappy, targets=["y"])["y"].to_numpy()

    assert np.flatnonzero(np.isnan(pred)).tolist() == [0, 11]
    ok = ~np.isnan(pred)
    np.testing.assert_allclose(pred[ok], df["y"].to_numpy()[ok], atol=1e-8)
