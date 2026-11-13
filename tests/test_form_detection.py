# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for observed-data (non)stationarity form detection.

These exercise the *detectors* on crafted series (unlike test_recommend, which
feeds hand-built facts). Motivated by a real bug where an ACF/spike heuristic
mislabeled plain trends and AR processes as "seasonal", flipping the C-preset.
"""

import numpy as np
import pandas as pd
import pytest

from causalts.utils.stationarity import check_stationarity_observed, detect_trend_form


def _ar_cols(T=400, d=3, seed=0):
    r = np.random.default_rng(seed)
    a = r.standard_normal((T, d))
    for t in range(1, T):
        for i in range(1, d):
            a[t, i] += 0.6 * a[t - 1, i - 1]
    return a


def _df(x0, seed=0):
    T = len(x0)
    a = _ar_cols(T, 3, seed)
    a[:, 0] = x0
    return pd.DataFrame(a, columns=["X0", "X1", "X2"])


def test_trend_is_trend_not_seasonal():
    T = 400
    t = np.arange(T)
    # test across several noise seeds — the old bug was seed-dependent
    for seed in range(5):
        x0 = 0.05 * t + np.random.default_rng(seed).standard_normal(T)
        assert detect_trend_form(_df(x0, seed)) == "trend"


def test_seasonal_is_seasonal():
    T = 400
    t = np.arange(T)
    x0 = (
        5 * np.sin(2 * np.pi * t / 50)
        + 0.05 * t
        + np.random.default_rng(1).standard_normal(T)
    )
    assert detect_trend_form(_df(x0)) == "seasonal"


def test_curved_is_curved():
    T = 400
    t = np.arange(T)
    x0 = 0.0008 * t**2 + np.random.default_rng(2).standard_normal(T)
    assert detect_trend_form(_df(x0)) == "curved"


def test_stationary_ar_not_flagged_nonstationary():
    # AR(0.6) is stationary — ADF+KPSS should not flag it wholesale
    df = pd.DataFrame(_ar_cols(400, 3, 7), columns=["X0", "X1", "X2"])
    res = check_stationarity_observed(df)
    assert res["fraction_nonstationary"] < 0.5


def test_trend_detected_nonstationary():
    T = 400
    t = np.arange(T)
    x0 = 0.1 * t + np.random.default_rng(0).standard_normal(T)
    df = _df(x0)
    res = check_stationarity_observed(df)
    assert "X0" in res["nonstationary_cols"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_adfuller_asks_for_plain_tuple_when_supported(monkeypatch):
    """statsmodels >= 0.15 warns unless result_object is given; we pass False."""
    import statsmodels.tsa.stattools as st

    real = st.adfuller
    seen = []

    def fake_adfuller(x, maxlag=None, regression="c", autolag="AIC", store=False,
                      regresults=False, result_object=None):  # fmt: skip
        seen.append(result_object)
        return real(x, maxlag=maxlag, regression=regression, autolag=autolag)

    monkeypatch.setattr(st, "adfuller", fake_adfuller)
    rng = np.random.default_rng(0)
    check_stationarity_observed(pd.DataFrame(rng.standard_normal((200, 2))))
    assert seen == [False, False]


def test_check_stationarity_no_future_warning():
    import warnings

    rng = np.random.default_rng(0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        out = check_stationarity_observed(pd.DataFrame(rng.standard_normal((200, 3))))
    # adfuller runs inside try/except, so an escalated warning would show up as NaN
    assert all(np.isfinite(r["adf_pvalue"]) for r in out["results"])
