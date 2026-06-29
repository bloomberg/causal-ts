# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Graph-informed forecasting using discovered causal structure."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression


def get_causal_parents(
    graph: np.ndarray,
    target: int | str,
    var_names: list[str] | None = None,
) -> list[tuple[str, int]]:
    """Extract causal parents of a target variable from the graph.

    Parameters
    ----------
    graph : np.ndarray
        Shape (d, d, max_lag+1). graph[i, j, lag]=1 means
        variable i at time t-lag causes variable j at time t.
    target : int or str
        Target variable index (int) or name (str).
    var_names : list[str] or None
        Variable names. Required if target is a string.

    Returns
    -------
    list[tuple[str, int]]
        List of (variable_name, lag) tuples for each parent edge.
    """
    d = graph.shape[0]
    max_lag = graph.shape[2] - 1

    if var_names is None:
        var_names = [f"X{i}" for i in range(d)]

    if isinstance(target, str):
        j = var_names.index(target)
    else:
        j = target

    parents = []
    for i in range(d):
        for lag in range(max_lag + 1):
            if graph[i, j, lag] == 1:
                if lag == 0 and i == j:
                    continue
                parents.append((var_names[i], lag))
    return parents


class _ARWrapper:
    """Wraps AutoReg result to match the ETS interface (fittedvalues, forecast)."""

    def __init__(self, result, T: int):
        self._result = result
        self._T = T

    @property
    def fittedvalues(self) -> np.ndarray:
        fitted = np.asarray(self._result.fittedvalues)
        out = np.full(self._T, np.nan)
        out[self._T - len(fitted) :] = fitted
        return out

    def forecast(self, steps: int) -> np.ndarray:
        return np.asarray(self._result.forecast(steps))

    @property
    def aic(self) -> float:
        return self._result.aic


def _build_feature_matrix(
    data: pd.DataFrame,
    parents: list[tuple[str, int]],
    target: str,
    exog_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build X, y matrices from parent lags and optional exogenous columns.

    Exogenous columns are included at lag 0 (contemporaneous — known at
    prediction time). Returns (X, y) with leading NaN rows dropped.
    """
    if not parents and not exog_cols:
        raise ValueError(
            f"No causal parents found for '{target}'. "
            "Cannot build a forecasting model without predictors."
        )

    frames = {}
    for var, lag in parents:
        col_name = f"{var}_lag{lag}" if lag > 0 else var
        frames[col_name] = data[var].shift(lag)

    if exog_cols:
        for col in exog_cols:
            frames[f"exog_{col}"] = data[col]

    X = pd.DataFrame(frames)
    y = data[target].copy()

    valid = X.notna().all(axis=1) & y.notna()
    return X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)


def _detect_seasonal_period(
    y: np.ndarray, max_period: int = 200, threshold: float = 0.2
) -> int | None:
    """Detect seasonality period from ACF on differenced series."""
    from statsmodels.tsa.stattools import acf

    T = len(y)
    if T < 10:
        return None

    diff = np.diff(y)
    nlags = min(T // 2, max_period)
    ac = acf(diff, nlags=nlags, fft=True)
    ac[:2] = 0

    peak = int(np.argmax(ac))
    if ac[peak] >= threshold and 4 <= peak <= T // 3:
        return peak
    return None


def _fit_auto_ets(y: np.ndarray, seasonal_period: int | None = None):
    """Fit the best model from AR and ETS candidates.

    Model selection uses held-out multi-step MSE on the last ~15% of the
    series so that AR and ETS candidates are compared on the same metric.
    AIC cannot be used directly because statsmodels AR and ETS report AIC
    on different scales (incomparable log-likelihood normalizations).

    AR(1-3) candidates are included so that stationary autocorrelated
    series (e.g. pure AR processes) are not forced into SES, which would
    forecast a flat line for all multi-step horizons.
    """
    from statsmodels.tsa.ar_model import AutoReg
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    T = len(y)
    val_size = min(30, max(5, T // 6))
    y_train = y[:-val_size]
    y_val = y[-val_size:]

    best_mse = np.inf
    best_model = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for p in [1, 2, 3]:
            if len(y_train) <= p + 5:
                continue
            try:
                res = AutoReg(y_train, lags=p, old_names=False).fit()
                pred = np.asarray(res.forecast(val_size))
                mse = np.mean((y_val - pred) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    res_full = AutoReg(y, lags=p, old_names=False).fit()
                    best_model = _ARWrapper(res_full, T)
            except Exception:
                continue

        configs: list[dict] = [
            {"trend": None, "seasonal": None},
            {"trend": "add", "seasonal": None, "damped_trend": False},
            {"trend": "add", "seasonal": None, "damped_trend": True},
        ]
        if seasonal_period is not None:
            configs.append(
                {
                    "trend": "add",
                    "seasonal": "add",
                    "seasonal_periods": seasonal_period,
                    "damped_trend": False,
                }
            )

        for cfg in configs:
            try:
                m = ExponentialSmoothing(y_train, **cfg).fit(
                    optimized=True, use_brute=False
                )
                pred = np.asarray(m.forecast(val_size))
                mse = np.mean((y_val - pred) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_model = ExponentialSmoothing(y, **cfg).fit(
                        optimized=True, use_brute=False
                    )
            except Exception:
                continue

    if best_model is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = AutoReg(y, lags=1, old_names=False).fit()
            best_model = _ARWrapper(res, T)

    return best_model


class CausalForecaster:
    """Forecast time series using graph-informed feature selection.

    Uses the discovered causal graph to select relevant lagged predictors
    for each target variable, then fits an sklearn-compatible model.

    Exogenous variables (known future values of external regressors) are
    supported via the ``exog`` parameter. They are added as contemporaneous
    (lag-0) features to every causal model and must be supplied at predict /
    forecast time.

    Variables with no causal parents fall back to an auto-selected AR or ETS
    model (chosen by AIC). This fallback is skipped when ``exog`` is provided --
    all variables then receive a regression model with the exog features.

    Parameters
    ----------
    graph : np.ndarray
        Shape (d, d, max_lag+1) discovered causal graph.
    var_names : list[str] or None
        Variable names corresponding to graph dimensions.
    model : object or None
        sklearn-compatible regressor (cloned per target).
        Defaults to LinearRegression.
    include_contemp : bool
        Whether to include contemporaneous parents (lag=0)
        as features. Default True. Set False for pure forecasting
        (no access to other variables at prediction time).
    exog_fallback : bool
        If True (default), fit auto AR/ETS models for
        exogenous variables (no causal parents) so that multi-step
        forecasting does not break on missing root-node predictions.
        Ignored when exog columns are provided.

    Examples
    --------
    ::

        from causalts.effects.forecast import CausalForecaster

        forecaster = CausalForecaster(G_hat, var_names=df.columns)
        forecaster.fit(df_train, exog=exog_train)
        preds = forecaster.predict(df_test, exog=exog_test)
        scores = forecaster.score(df_train, exog=exog_train)
        fcast = forecaster.forecast_horizon(
            df_train, horizon=30, exog_future=exog_future
        )
    """

    def __init__(
        self,
        graph: np.ndarray,
        var_names: list[str] | None = None,
        model=None,
        include_contemp: bool = True,
        exog_fallback: bool = True,
    ):
        d = graph.shape[0]
        self.graph = graph
        self.var_names = (
            list(var_names) if var_names is not None else [f"X{i}" for i in range(d)]
        )
        self.model_template = model if model is not None else LinearRegression()
        self.include_contemp = include_contemp
        self.exog_fallback = exog_fallback
        self.models_: dict[str, object] = {}
        self.parents_: dict[str, list[tuple[str, int]]] = {}
        self.exog_models_: dict[str, object] = {}
        self.exog_cols_: list[str] = []

    def _merge_exog(
        self, data: pd.DataFrame, exog: pd.DataFrame | None
    ) -> pd.DataFrame:
        """Return a copy of data with exog columns appended."""
        if exog is None or not self.exog_cols_:
            return data
        data = data.copy()
        for col in self.exog_cols_:
            data[col] = exog[col].values
        return data

    def fit(
        self,
        data: pd.DataFrame,
        targets: list[str] | None = None,
        exog: pd.DataFrame | None = None,
    ) -> "CausalForecaster":
        """Fit forecasting models for each target variable.

        Parameters
        ----------
        data : pd.DataFrame
            Time series DataFrame (T, d).
        targets : list[str] or None
            Variables to forecast. Defaults to all variables
            that have at least one causal parent. When ``exog`` is
            provided, defaults to all variables (parentless variables
            use exog as their only features).
        exog : pd.DataFrame or None
            Optional DataFrame (T, k) of exogenous variables.
            Columns are added as contemporaneous (lag-0) features to
            all regression models. Future values must be supplied to
            ``predict()`` and ``forecast_horizon()``.

        Returns
        -------
        CausalForecaster
            self (fitted).
        """
        self.exog_cols_ = list(exog.columns) if exog is not None else []
        data_ext = self._merge_exog(data, exog)

        if targets is None:
            targets = []
            for j, name in enumerate(self.var_names):
                parents = get_causal_parents(self.graph, j, self.var_names)
                if parents or self.exog_cols_:
                    targets.append(name)

        for target in targets:
            j = self.var_names.index(target)
            parents = get_causal_parents(self.graph, j, self.var_names)

            if not self.include_contemp:
                parents = [(v, lag) for v, lag in parents if lag > 0]

            if not parents and not self.exog_cols_:
                continue

            self.parents_[target] = parents
            X, y = _build_feature_matrix(data_ext, parents, target, self.exog_cols_)
            m = clone(self.model_template)
            m.fit(X, y)
            self.models_[target] = m

        # AR/ETS fallback for parentless variables — only when no exog provided
        if self.exog_fallback and not self.exog_cols_:
            fitted_vars = set(self.models_.keys())
            for name in self.var_names:
                if name not in fitted_vars:
                    series = data[name].dropna().values
                    if len(series) < 4:
                        continue
                    period = _detect_seasonal_period(series)
                    self.exog_models_[name] = _fit_auto_ets(series, period)

        return self

    def predict(
        self,
        data: pd.DataFrame,
        targets: list[str] | None = None,
        exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate in-sample predictions for target variables.

        Parameters
        ----------
        data : pd.DataFrame
            Time series DataFrame (T, d).
        targets : list[str] or None
            Variables to predict. Defaults to all fitted targets.
        exog : pd.DataFrame or None
            Exogenous variables aligned with data. Required if the
            model was fitted with exog.

        Returns
        -------
        pd.DataFrame
            DataFrame with predictions (NaN where insufficient lag data).
        """
        if self.exog_cols_ and exog is None:
            raise ValueError(
                f"Model fitted with exogenous columns {self.exog_cols_}; "
                "provide exog= to predict()."
            )
        data_ext = self._merge_exog(data, exog)

        if targets is None:
            targets = sorted(set(self.models_) | set(self.exog_models_))

        results = {}
        for target in targets:
            if target in self.models_:
                parents = self.parents_[target]
                X, _ = _build_feature_matrix(data_ext, parents, target, self.exog_cols_)
                max_lag = max((lag for _, lag in parents), default=0)
                preds = np.full(len(data), np.nan)
                preds[max_lag : max_lag + len(X)] = self.models_[target].predict(X)
                results[target] = preds
            elif target in self.exog_models_:
                fitted = np.asarray(self.exog_models_[target].fittedvalues)
                preds = np.full(len(data), np.nan)
                n = min(len(data), len(fitted))
                preds[:n] = fitted[:n]
                results[target] = preds
            else:
                raise ValueError(
                    f"No model fitted for '{target}'. "
                    f"Available: {sorted(set(self.models_) | set(self.exog_models_))}"
                )

        return pd.DataFrame(results, index=data.index)

    def score(
        self,
        data: pd.DataFrame,
        targets: list[str] | None = None,
        exog: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Compute R-squared for each target on given data.

        Parameters
        ----------
        data : pd.DataFrame
            Time series DataFrame (T, d).
        targets : list[str] or None
            Variables to score. Defaults to all fitted targets.
        exog : pd.DataFrame or None
            Exogenous variables aligned with data. Required if the
            model was fitted with exog.

        Returns
        -------
        dict[str, float]
            Dict mapping variable name to R-squared.
        """
        if self.exog_cols_ and exog is None:
            raise ValueError(
                f"Model fitted with exogenous columns {self.exog_cols_}; "
                "provide exog= to score()."
            )
        data_ext = self._merge_exog(data, exog)

        if targets is None:
            targets = sorted(set(self.models_) | set(self.exog_models_))

        scores = {}
        for target in targets:
            if target in self.models_:
                parents = self.parents_[target]
                X, y = _build_feature_matrix(data_ext, parents, target, self.exog_cols_)
                scores[target] = self.models_[target].score(X, y)
            elif target in self.exog_models_:
                m = self.exog_models_[target]
                y = data[target].dropna().values
                fitted = np.asarray(m.fittedvalues)
                n = min(len(y), len(fitted))
                y_a, f_a = y[:n], fitted[:n]
                valid = ~np.isnan(f_a)
                y_v, f_v = y_a[valid], f_a[valid]
                if len(y_v) == 0:
                    scores[target] = 0.0
                    continue
                ss_res = np.sum((y_v - f_v) ** 2)
                ss_tot = np.sum((y_v - y_v.mean()) ** 2)
                scores[target] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return scores

    def feature_importance(self, target: str) -> pd.DataFrame:
        """Get feature importance for a fitted target model.

        For linear models, returns coefficients. For tree-based models,
        returns ``feature_importances_``. Falls back to None if unavailable.

        Parameters
        ----------
        target : str
            Variable name.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns [parent, lag, importance].
        """
        if target not in self.models_:
            raise ValueError(f"No model fitted for '{target}'.")

        model = self.models_[target]
        parents = self.parents_[target]

        feature_names: list[tuple[str, int]] = [(v, l) for v, l in parents]
        for col in self.exog_cols_:
            feature_names.append((f"exog_{col}", 0))

        if hasattr(model, "coef_"):
            importances = np.abs(model.coef_)
        elif hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            importances = [None] * len(feature_names)

        rows = []
        for (var, lag), imp in zip(feature_names, importances):
            rows.append({"parent": var, "lag": lag, "importance": imp})
        return pd.DataFrame(rows).sort_values("importance", ascending=False)

    def forecast_horizon(
        self,
        data: pd.DataFrame,
        horizon: int = 1,
        targets: list[str] | None = None,
        exog_future: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Multi-step ahead forecasting via iterative prediction.

        Uses fitted models to predict forward one step at a time,
        feeding predictions back as inputs for subsequent steps.
        Only works when include_contemp=False (pure lagged models).

        Parameters
        ----------
        data : pd.DataFrame
            Historical time series DataFrame (T, d).
        horizon : int
            Number of steps ahead to forecast.
        targets : list[str] or None
            Variables to forecast. Defaults to all fitted targets.
        exog_future : pd.DataFrame or None
            DataFrame of shape (horizon, k) with future values
            of exogenous variables, in the same column order as the
            ``exog`` passed to ``fit()``. Required when the model was
            fitted with exog.

        Returns
        -------
        pd.DataFrame
            DataFrame of shape (horizon, len(targets)) with forecasted
            values.
        """
        if self.include_contemp:
            raise ValueError(
                "Multi-step forecasting requires include_contemp=False. "
                "Contemporaneous parents are unavailable at forecast time."
            )
        if self.exog_cols_ and exog_future is None:
            raise ValueError(
                f"Model fitted with exogenous columns {self.exog_cols_}; "
                "provide exog_future= (DataFrame of shape (horizon, k))."
            )
        if exog_future is not None and len(exog_future) < horizon:
            raise ValueError(
                f"exog_future must have at least {horizon} rows, "
                f"got {len(exog_future)}."
            )

        if targets is None:
            targets = sorted(set(self.models_) | set(self.exog_models_))

        exog_forecasts = {}
        for var in self.exog_models_:
            if var in targets:
                exog_forecasts[var] = np.asarray(
                    self.exog_models_[var].forecast(horizon)
                )

        extended = data.copy()
        forecasts = []

        for step in range(horizon):
            row = {}
            for target in targets:
                if target in exog_forecasts:
                    row[target] = exog_forecasts[target][step]
                elif target in self.models_:
                    parents = self.parents_[target]
                    feat = {}
                    for var, lag in parents:
                        col_name = f"{var}_lag{lag}" if lag > 0 else var
                        idx = len(extended) - lag
                        if idx < 0:
                            feat[col_name] = np.nan
                        else:
                            feat[col_name] = extended[var].iloc[idx]

                    if exog_future is not None:
                        for col in self.exog_cols_:
                            feat[f"exog_{col}"] = exog_future.iloc[step][col]

                    X_row = pd.DataFrame([feat])
                    if X_row.isna().any(axis=1).iloc[0]:
                        row[target] = np.nan
                    else:
                        row[target] = self.models_[target].predict(X_row)[0]
                else:
                    row[target] = np.nan

            forecasts.append(row)

            new_row = {}
            for col in extended.columns:
                new_row[col] = row.get(col, np.nan)
            extended = pd.concat([extended, pd.DataFrame([new_row])], ignore_index=True)

        return pd.DataFrame(forecasts)

    def __repr__(self):
        n_causal = len(self.models_)
        n_exog = len(self.exog_models_)
        n_ext = len(self.exog_cols_)
        exog_str = f", exog_features={n_ext}" if n_ext else ""
        return (
            f"CausalForecaster(causal={n_causal}, exog_ets={n_exog}, "
            f"model={type(self.model_template).__name__}{exog_str})"
        )
