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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
