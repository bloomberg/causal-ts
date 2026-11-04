# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for the deterministic recommend_config decision function."""

import pytest

from causalts.inspection import recommend_config


def _facts(
    nonlinear=False, nonstationary_cols=None, form="none", max_lag=3, variance=False
):
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
        "variance_dependence": {
            "detected": variance,
            "n_pairs": 1 if variance else 0,
            "pairs": ["X0 -> X1"] if variance else [],
        },
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
    assert rec["refine"] == "reap"
    assert rec["refine_params"] == {"contemporaneous": True}
    assert rec["include_C"] is False


@pytest.mark.parametrize("T", [150, 600])
def test_nonlinear_uses_raft_at_any_T(T):
    rec = recommend_config(_facts(nonlinear=True), _data(d=5, T=T))
    assert rec["ci_test"] == "raft"
    assert rec["refine"] == "reap"


@pytest.mark.parametrize("nonlinear", [False, True])
def test_variance_parents_switch_to_rcot_and_grace(nonlinear):
    rec = recommend_config(_facts(nonlinear=nonlinear, variance=True), _data())
    assert rec["ci_test"] == "rcot"
    assert rec["refine"] == "grace"
    assert rec["refine_params"] == {}


@pytest.mark.parametrize(
    "reason, says",
    [
        ("too_large", "run causalts.utils.heteroskedasticity"),
        ("unavailable", "could not run"),
        ("not_applicable", None),
    ],
)
def test_variance_check_unavailable_is_not_detection(reason, says):
    facts = _facts()
    facts["variance_dependence"].update(detected=None, reason=reason)
    rec = recommend_config(facts, _data())
    assert rec["ci_test"] == "parcorr-gpu" and rec["refine"] == "reap"
    if says is None:
        assert "variance-dependence check" not in rec["rationale"]
    else:
        assert says in rec["rationale"]


def test_nonstationary_seasonal_uses_linear_sin():
    rec = recommend_config(_facts(nonstationary_cols=["X1"], form="seasonal"), _data())
    assert rec["include_C"] is True
    assert rec["c_preset"] == "linear+sin"


def test_nonstationary_curved_uses_linear_quad():
    rec = recommend_config(_facts(nonstationary_cols=["X1"], form="curved"), _data())
    assert rec["include_C"] is True
    assert rec["c_preset"] == "linear+quad"


@pytest.mark.parametrize("d", [5, 25, 80])
def test_no_dimension_cutoff(d):
    """Discovery is CDNOTS at every d: its cost in d is set by the CI test."""
    rec = recommend_config(_facts(), _data(d=d, T=400))
    assert rec["algorithm"] == "cdnots"
    assert rec["refine"] == "reap"


def test_discrete_cols_use_discrete_aware_test():
    rec = recommend_config(_facts(variance=True), _data(discrete_cols=["X2"]))
    assert rec["ci_test"] == "cmiknn-mixed-gpu"


def test_rationale_is_present():
    rec = recommend_config(_facts(), _data())
    assert isinstance(rec["rationale"], str) and rec["rationale"]


def test_contemporaneous_only_has_no_refinement():
    """Both refinements work on lagged edges, so max_lag=0 gets none."""
    rec = recommend_config(_facts(max_lag=0), _data())
    assert rec["refine"] is None
    assert rec["refine_params"] == {}


@pytest.mark.parametrize("variance", [False, True])
def test_missing_values_impute_in_every_step(variance):
    data = _data()
    data["missing_by_col"] = {"X1": 0.01}
    rec = recommend_config(_facts(variance=variance), data)
    assert rec["params"]["impute"] == "var_em"
    assert rec["refine_params"]["impute"] == "var_em"
    assert "impute" in rec["rationale"]
    rec0 = recommend_config(_facts(max_lag=0), data)
    assert rec0["params"]["impute"] == "var_em" and rec0["refine"] is None


def test_grace_rationale_points_to_merge_lag0():
    rec = recommend_config(_facts(variance=True), _data())
    assert rec["refine"] == "grace" and "merge_lag0" in rec["rationale"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
