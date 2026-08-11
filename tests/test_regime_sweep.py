# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""End-to-end integration guard: inspect_df recommendations across regimes.

Complements test_recommend (hand-built facts) and test_form_detection (detectors)
by driving the full inspect_df path on crafted data — the level at which the
regime sweep originally caught the seasonal-misclassification and the
constant-column -> cmiknn-gpu cascade bugs.
"""

import numpy as np
import pandas as pd
import pytest

from causalts import inspect_df


def _ar(T=400, d=4, seed=0):
    r = np.random.default_rng(seed)
    a = r.standard_normal((T, d))
    for t in range(1, T):
        a[t, 1] += 0.6 * a[t - 1, 0]
        a[t, 2] += 0.5 * a[t - 1, 1]
    return pd.DataFrame(a, columns=[f"X{i}" for i in range(d)])


def test_linear_stationary_recommends_parcorr_no_c():
    rec = inspect_df(_ar())["recommendation"]
    assert rec["algorithm"] == "cdnots"
    assert rec["ci_test"] == "parcorr-gpu"
    assert rec["include_C"] is False


def test_trend_recommends_linear_c():
    t = np.arange(400)
    df = _ar()
    df["X0"] = 0.1 * t + np.random.default_rng(0).standard_normal(400)
    rec = inspect_df(df)["recommendation"]
    assert rec["include_C"] is True
    assert rec["c_preset"] == "linear"


def test_seasonal_form_maps_to_linear_sin():
    t = np.arange(400)
    df = _ar()
    df["X0"] = 5 * np.sin(2 * np.pi * t / 50) + 0.05 * t
    df["X0"] += np.random.default_rng(1).standard_normal(400)
    r = inspect_df(df)
    assert r["facts"]["stationarity"]["form"] == "seasonal"


def test_curved_form_maps_to_linear_quad():
    t = np.arange(400)
    df = _ar()
    df["X0"] = 0.0008 * t**2 + np.random.default_rng(2).standard_normal(400)
    r = inspect_df(df)
    assert r["facts"]["stationarity"]["form"] == "curved"


def test_high_dim_recommends_grace():
    r = np.random.default_rng(4)
    T, d = 400, 25
    a = r.standard_normal((T, d))
    for t in range(1, T):
        for i in range(1, d):
            a[t, i] += 0.4 * a[t - 1, i - 1]
    df = pd.DataFrame(a, columns=[f"X{i}" for i in range(d)])
    assert inspect_df(df)["recommendation"]["algorithm"] == "grace"


def test_constant_column_does_not_trigger_cmiknn():
    # regression: a constant column was flagged discrete -> cmiknn-gpu/expensive
    df = _ar()
    df["X0"] = 1.0
    r = inspect_df(df)
    assert "X0" in r["data"]["constant_cols"]
    assert "X0" not in r["data"]["discrete_cols"]
    assert r["recommendation"]["ci_test"] != "cmiknn-gpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
