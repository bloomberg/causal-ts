# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for the deterministic recommend_config decision function."""

import pytest

from causalts.inspection import recommend_config


def _facts(nonlinear=False, nonstationary_cols=None, form="none", max_lag=3):
    return {
        "linearity": {
            "fraction_nonlinear": 0.5 if nonlinear else 0.0,
            "is_nonlinear": nonlinear,
        },
        "stationarity": {
            "fraction_nonstationary": (
                len(nonstationary_cols) / 5 if nonstationary_cols else 0.0
            ),
            "nonstationary_cols": nonstationary_cols or [],
            "form": form,
        },
        "suggested_max_lag": max_lag,
    }


def _data(d=5, T=300, discrete_cols=None):
    return {
        "n_rows": T,
        "n_vars": d,
        "var_names": [f"X{i}" for i in range(d)],
        "missing_by_col": {},
        "constant_cols": [],
        "discrete_cols": discrete_cols or [],
    }


def test_linear_stationary_lowdim():
    rec = recommend_config(_facts(), _data(d=5, T=300))
    assert rec["algorithm"] == "cdnots"
    assert rec["ci_test"] == "parcorr-gpu"
    assert rec["include_C"] is False


def test_nonlinear_small_T():
    rec = recommend_config(_facts(nonlinear=True), _data(d=5, T=200))
    assert rec["ci_test"] == "dfcit"


def test_nonlinear_large_T():
    rec = recommend_config(_facts(nonlinear=True), _data(d=5, T=600))
    assert rec["ci_test"] == "splitkci"


def test_nonstationary_seasonal_uses_linear_sin():
    rec = recommend_config(_facts(nonstationary_cols=["X1"], form="seasonal"), _data())
    assert rec["include_C"] is True
    assert rec["c_preset"] == "linear+sin"


def test_nonstationary_curved_uses_linear_quad():
    rec = recommend_config(_facts(nonstationary_cols=["X1"], form="curved"), _data())
    assert rec["include_C"] is True
    assert rec["c_preset"] == "linear+quad"


def test_high_dim_uses_grace():
    rec = recommend_config(_facts(), _data(d=25, T=400))
    assert rec["algorithm"] == "grace"


def test_discrete_cols_use_discrete_aware_test():
    rec = recommend_config(_facts(), _data(discrete_cols=["X2"]))
    assert rec["ci_test"] == "cmiknn-gpu"


def test_rationale_is_present():
    rec = recommend_config(_facts(), _data())
    assert isinstance(rec["rationale"], str) and rec["rationale"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
