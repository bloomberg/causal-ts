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


def test_non_numeric_column_is_skipped_with_a_warning():
    """A date column made discrete detection fail on strings."""
    df = _linear_var()
    df["date"] = pd.date_range("2020-01-01", periods=len(df)).astype(str)
    df["label"] = np.where(np.arange(len(df)) % 2, "up", "down")
    report = inspect_df(df)
    assert report["data"]["n_vars"] == 4
    assert any("date" in w and "label" in w for w in report["warnings"])
    assert report["recommendation"] == inspect_df(_linear_var())["recommendation"]


def test_timedelta_column_is_flagged():
    df = _linear_var()
    df["dur"] = pd.to_timedelta(np.arange(len(df)), unit="s")
    report = inspect_df(df)
    assert report["data"]["n_vars"] == 4
    assert any("Non-numeric" in w and "dur" in w for w in report["warnings"])


def _nonlinear_var(T=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, 3))
    x[1:, 1] += 2.0 * np.sin(2.0 * x[:-1, 0])
    x[1:, 2] += x[:-1, 1] ** 2
    return pd.DataFrame(x, columns=["a", "b", "c"])


def test_missing_values_keep_nonlinearity_detection():
    """One NaN per column made every RESET p-value NaN, read as linear."""
    df = _nonlinear_var()
    full = inspect_df(df)["facts"]["linearity"]["fraction_nonlinear"]
    assert full > 0.2
    rng = np.random.default_rng(1)
    for c in df.columns:
        df.loc[rng.choice(len(df), 4, replace=False), c] = np.nan
    report = inspect_df(df)
    assert report["facts"]["linearity"]["fraction_nonlinear"] == pytest.approx(
        full, abs=0.1
    )
    assert report["facts"]["linearity"]["is_nonlinear"]


def test_max_lag_override():
    report = inspect_df(_linear_var(), max_lag=7)
    assert report["facts"]["suggested_max_lag"] == 7
    assert report["recommendation"]["max_lag"] == 7


def test_json_serialisable():
    import json

    report = inspect_df(_linear_var())
    json.dumps(report, default=str)  # must not raise


# ── facts["latent_factor"] ───────────────────────────────────────────────────
# Positive-evidence-only diagnostic: it can assert that a pervasive factor IS
# present, but "not detected" never means "no confounding" (a confounder touching
# two or three variables leaves no dominant eigenvalue). The tests below pin that
# asymmetry, not just the happy path.
def _pervasive(T=600, d=8, n_factors=2, seed=0):
    """Low-rank innovations: a few latent factors drive every variable."""
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((T, n_factors))
    loadings = rng.standard_normal((n_factors, d))
    data = 2.0 * (factors @ loadings)
    data[1:, 1] += 0.6 * data[:-1, 0]
    data += 0.3 * rng.standard_normal((T, d))
    return pd.DataFrame(data, columns=[f"X{i}" for i in range(d)])


def test_latent_factor_detected_on_pervasive_data():
    lf = inspect_df(_pervasive())["facts"]["latent_factor"]
    assert lf["detected"] is True
    assert lf["spectral_ratio"] > lf["tau"]


def test_latent_factor_not_detected_on_full_rank_data():
    lf = inspect_df(_linear_var(T=600, d=8))["facts"]["latent_factor"]
    assert lf["detected"] is False
    assert lf["spectral_ratio"] <= lf["tau"]


@pytest.mark.parametrize(
    "df",
    [
        pytest.param(
            pd.DataFrame(
                np.where(
                    np.random.default_rng(1).random((300, 6)) < 0.15,
                    np.nan,
                    np.random.default_rng(0).standard_normal((300, 6)),
                ),
                columns=[f"X{i}" for i in range(6)],
            ),
            id="missing_values",
        ),
        pytest.param(
            pd.DataFrame({"X0": np.random.default_rng(0).standard_normal(300)}),
            id="single_column",
        ),
        pytest.param(
            pd.DataFrame(
                np.random.default_rng(0).standard_normal((6, 8)),
                columns=[f"X{i}" for i in range(8)],
            ),
            id="underdetermined_T_lt_d",
        ),
    ],
)
def test_latent_factor_guards_report_none_not_false(df):
    """The VAR(1) least-squares fit behind the statistic cannot run on these.

    ``None`` (not ``False``) is required: "could not check" must never be
    readable as "checked and found nothing".
    """
    import json

    report = inspect_df(df)
    lf = report["facts"]["latent_factor"]
    assert lf == {"detected": None, "spectral_ratio": None, "tau": None}
    json.dumps(report, default=str)  # stays serialisable


def test_deconfound_nudge_only_fires_on_positive_detection():
    assert "deconfound" in inspect_df(_pervasive())["recommendation"]["rationale"]
    rec = inspect_df(_linear_var(T=600, d=8))["recommendation"]
    assert "deconfound" not in rec["rationale"]


def test_latent_factor_does_not_change_algorithm_choice():
    """LUCID is orthogonal post-processing, not a discovery-algorithm alternative."""
    pervasive, plain = _pervasive(d=8), _linear_var(T=600, d=8)
    a = inspect_df(pervasive)["recommendation"]
    b = inspect_df(plain)["recommendation"]
    assert a["algorithm"] == b["algorithm"]
    assert a["include_C"] == b["include_C"]


def test_variance_dependence_fact_and_recommendation():
    rng = np.random.default_rng(0)
    T, d = 600, 4
    x = np.zeros((T, d))
    for t in range(1, T):
        scale = np.ones(d)
        scale[3] = np.exp(0.6 * x[t - 1, 0])
        x[t] = 0.4 * x[t - 1] + scale * rng.normal(size=d)
    report = inspect_df(pd.DataFrame(x, columns=["a", "b", "c", "z"]))
    vd = report["facts"]["variance_dependence"]
    assert vd["detected"] is True and "a -> z" in vd["pairs"]
    assert report["recommendation"]["ci_test"] == "rcot"
    assert report["recommendation"]["refine"] == "grace"


def test_contemporaneous_only_skips_variance_check_and_refinement():
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(200, 3)))
    out = inspect_df(df, max_lag=0)
    assert out["facts"]["variance_dependence"]["detected"] is None
    assert out["recommendation"]["refine"] is None


def test_variance_check_skipped_above_budget(monkeypatch):
    import causalts.inspection as ins

    monkeypatch.setattr(ins, "_VARIANCE_CHECK_BUDGET", 100)
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(200, 3)))
    out = inspect_df(df, max_lag=1)
    vd = out["facts"]["variance_dependence"]
    assert vd["detected"] is None and vd["reason"] == "too_large"
    assert "check skipped" in out["recommendation"]["rationale"]


def test_variance_check_not_applicable_for_one_variable():
    df = pd.DataFrame({"x": np.random.default_rng(0).normal(size=200)})
    out = inspect_df(df, max_lag=1)
    assert out["facts"]["variance_dependence"]["reason"] == "not_applicable"
    assert "variance-dependence check" not in out["recommendation"]["rationale"]


def test_one_numeric_column_does_not_warn_about_untestable_linearity():
    """A date + value frame leaves one column: no pair to test, no failure."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=200).astype(str),
            "x": rng.normal(size=200),
        }
    )
    out = inspect_df(df, max_lag=1)
    assert out["facts"]["linearity"]["n_tested"] == 0
    assert not any("Linearity could not be tested" in w for w in out["warnings"])
    rationale = out["recommendation"]["rationale"]
    assert "fewer than two numeric columns" in rationale
    assert "approximately linear" not in rationale
    assert "linearity could not be tested" not in rationale


def _bool_variance_panel(T=600, seed=0):
    """A Boolean b that only moves the noise scale of y one step later."""
    rng = np.random.default_rng(seed)
    b = rng.random(T) < 0.5
    y = np.r_[0.0, np.where(b[:-1], 3.0, 0.3)] * rng.normal(size=T)
    return pd.DataFrame({"b": b, "y": y, "z": rng.normal(size=T)})


def test_boolean_columns_reach_the_variance_check():
    out = inspect_df(_bool_variance_panel(), max_lag=1)
    vd = out["facts"]["variance_dependence"]
    assert vd["detected"] is True and "b -> y" in vd["pairs"]
    assert out["data"]["discrete_cols"] == ["b"]


def test_discover_df_accepts_boolean_columns():
    from causalts.inspection import discover_df

    df = _bool_variance_panel(T=200)
    res = discover_df(df, ci_test="cmiknn-mixed-gpu", max_lag=1)
    assert np.asarray(res.cg_tig).shape[:2] == (3, 3)


def test_object_true_false_column_with_blank_is_numeric():
    """read_csv leaves True/False with a blank cell as an object column."""
    from causalts.inspection import numeric_frame

    df = _bool_variance_panel(T=200)
    col = df["b"].astype(object)
    col.iloc[3] = None
    df["b"] = col
    out = numeric_frame(df)
    assert out["b"].dtype == np.float64 and np.isnan(out["b"].iloc[3])
    assert inspect_df(df, max_lag=1)["data"]["discrete_cols"] == ["b"]


def test_any_missing_value_warns_and_imputes():
    df = pd.DataFrame(
        np.random.default_rng(0).normal(size=(120, 3)), columns=list("abc")
    )
    df.iloc[5, 1] = np.nan
    out = inspect_df(df, max_lag=1)
    assert any("b has missing values" in w for w in out["warnings"])
    assert out["recommendation"]["refine_params"].get("impute") == "var_em"


def test_merge_lag0_restores_candidate_contemporaneous_edges():
    from causalts.inspection import merge_lag0

    refined = np.zeros((3, 3, 2))
    refined[0, 1, 1] = 1
    candidate = np.zeros((2, 2, 2))
    candidate[0, 1, 0] = 1
    candidate[1, 0, 1] = 1  # lagged edges of the candidate are not copied
    g = merge_lag0(refined, candidate)
    assert g[0, 1, 0] == 1 and g[0, 1, 1] == 1 and g[1, 0, 1] == 0
    assert g.shape == refined.shape and refined[0, 1, 0] == 0


def test_latent_factor_block_unavailable_when_threshold_unreachable():
    # With T_eff = T - d - 2 the auto threshold can reach 1 on narrow-but-short
    # panels; the top-2 trace share cannot exceed 1, so the check cannot fire there.
    from causalts.inspection import _latent_factor_block

    for d, T in [(2, 5000), (3, 300), (4, 60), (10, 13), (10, 22), (20, 23), (20, 28)]:
        out = _latent_factor_block(np.random.default_rng(0).normal(size=(T, d)))
        assert out == {"detected": None, "spectral_ratio": None, "tau": None}
    ok = _latent_factor_block(np.random.default_rng(0).normal(size=(200, 10)))
    assert ok["detected"] is False and ok["tau"] < 1


def test_no_numeric_columns_raises():
    """Numbers parsed as strings left a normal-looking report for d=0."""
    df = pd.DataFrame({"a": ["1,000", "2,000"] * 50, "b": ["3,000", "4,000"] * 50})
    with pytest.raises(ValueError, match="no numeric columns.*'a', 'b'"):
        inspect_df(df)


def test_untestable_linearity_is_reported():
    """Columns observed on disjoint rows: no RESET test can run."""
    df = _linear_var(T=300, d=3)
    rows = np.arange(len(df))
    for i, c in enumerate(df.columns):
        df.loc[rows % 3 != i, c] = np.nan
    report = inspect_df(df)
    lin = report["facts"]["linearity"]
    assert lin["n_tested"] == 0
    assert any("Linearity could not be tested" in w for w in report["warnings"])
    rationale = report["recommendation"]["rationale"]
    assert "approximately linear" not in rationale
    assert "linearity could not be tested" in rationale


def test_cli_inspect_no_numeric_columns_is_a_clean_error(tmp_path):
    from click.testing import CliRunner

    from causalts.cli import main

    path = tmp_path / "d.csv"
    pd.DataFrame({"a": ["1,000", "2,000"] * 50}).to_csv(path, index=False)
    res = CliRunner().invoke(main, ["inspect", str(path)])
    assert res.exit_code != 0
    assert "no numeric columns" in res.output
    assert not isinstance(res.exception, ValueError)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_check_linearity_max_lag_beyond_series_length():
    """A lag longer than the series leaves nothing to test instead of raising."""
    from causalts.utils.linearity import check_linearity

    data = np.random.default_rng(0).standard_normal((20, 2))
    res = check_linearity(data, max_lag=30)
    assert res["n_tested"] == len(res["results"]) > 0
    assert all(r["lag"] <= 10 for r in res["results"])
