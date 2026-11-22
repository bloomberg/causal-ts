# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for causalts.imputation: var_em_impute and iterative_causal_impute."""

import warnings

import numpy as np
import pandas as pd

from causalts.imputation import (
    _initial_fill,
    _markov_blanket_from_tigramite,
    iterative_causal_impute,
    var_em_impute,
)


def _make_var1_data(T=100, K=3, seed=42):
    rng = np.random.default_rng(seed)
    A = rng.uniform(-0.3, 0.3, (K, K))
    data = np.zeros((T, K))
    data[0] = rng.standard_normal(K)
    for t in range(1, T):
        data[t] = A @ data[t - 1] + 0.5 * rng.standard_normal(K)
    return pd.DataFrame(data, columns=[f"X{i}" for i in range(K)])


def _mask_nan(df, frac=0.10, seed=0):
    rng = np.random.default_rng(seed)
    arr = df.values.copy()
    original = arr.copy()
    mask = rng.random(arr.shape) < frac
    arr[mask] = np.nan
    return pd.DataFrame(arr, columns=df.columns), original, mask


def test_initial_fill_no_nans():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    result = _initial_fill(df)
    pd.testing.assert_frame_equal(result, df)


def test_initial_fill_preserves_non_nan():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 5.0, 6.0]})
    result = _initial_fill(df)
    assert result["a"].iloc[0] == 1.0
    assert result["a"].iloc[2] == 3.0
    assert result["b"].iloc[1] == 5.0
    assert result["b"].iloc[2] == 6.0
    assert not result.isna().any().any()


def test_var_em_no_nans():
    df = _make_var1_data(T=60, K=3)
    result = var_em_impute(df)
    pd.testing.assert_frame_equal(result, df)


def test_var_em_no_nans_in_output():
    df, _, _ = _mask_nan(_make_var1_data(T=100, K=3), frac=0.10)
    result = var_em_impute(df, p=1, max_iter=10)
    assert not result.isna().any().any()


def test_var_em_preserves_non_nan_positions():
    df_clean = _make_var1_data(T=100, K=3)
    df_masked, original, mask = _mask_nan(df_clean, frac=0.10)
    result = var_em_impute(df_masked, p=1, max_iter=10)
    result_arr = result.values
    for t in range(result_arr.shape[0]):
        for k in range(result_arr.shape[1]):
            if not mask[t, k]:
                orig_val = df_masked.values[t, k]
                assert result_arr[t, k] == orig_val


def test_var_em_rmse_reasonable():
    df_clean = _make_var1_data(T=150, K=3, seed=7)
    df_masked, original, mask = _mask_nan(df_clean, frac=0.08, seed=7)
    result = var_em_impute(df_masked, p=1, max_iter=20)
    result_arr = result.values
    rmse = np.sqrt(np.mean((result_arr[mask] - original[mask]) ** 2))
    col_stds = df_clean.std().values.mean()
    assert rmse < 2.0 * col_stds


def test_var_em_warns_on_too_few_obs():
    df, _, _ = _mask_nan(
        pd.DataFrame(np.random.randn(5, 4), columns=list("ABCD")), frac=0.2
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = var_em_impute(df, p=3, max_iter=10)
    assert any("too few observations" in str(warning.message) for warning in w)
    assert not result.isna().any().any()


def test_markov_blanket_basic():
    N, L = 3, 1
    cg_tig = np.zeros((N, N, L + 1), dtype=int)
    cg_tig[0, 1, 1] = 1
    cg_tig[2, 1, 0] = 1
    mb = _markov_blanket_from_tigramite(cg_tig, var_idx=1, num_vars=3)
    assert mb == {(0, -1), (2, 0)}


def test_markov_blanket_includes_parents_of_children():
    N, L = 3, 1
    cg_tig = np.zeros((N, N, L + 1), dtype=int)
    cg_tig[0, 1, 0] = 1
    cg_tig[2, 1, 1] = 1
    mb = _markov_blanket_from_tigramite(cg_tig, var_idx=0, num_vars=3)
    # child X1(t), and X1's other parent X2 at t-1
    assert mb == {(1, 0), (2, -1)}


def test_markov_blanket_excludes_C():
    N_with_C, num_vars, L = 4, 3, 1
    cg_tig = np.zeros((N_with_C, N_with_C, L + 1), dtype=int)
    cg_tig[3, 0, 0] = 1
    mb = _markov_blanket_from_tigramite(cg_tig, var_idx=0, num_vars=num_vars)
    assert all(var != 3 for var, _ in mb)


def test_markov_blanket_empty():
    N, L = 3, 1
    cg_tig = np.zeros((N, N, L + 1), dtype=int)
    mb = _markov_blanket_from_tigramite(cg_tig, var_idx=1, num_vars=3)
    assert mb == set()


def test_markov_blanket_keeps_lags_of_parents_children_and_spouses():
    N, L = 3, 2
    cg_tig = np.zeros((N, N, L + 1), dtype=int)
    cg_tig[1, 1, 1] = 1  # own lag: X1(t-1) -> X1(t)
    cg_tig[0, 1, 2] = 1  # parent: X0(t-2) -> X1(t)
    cg_tig[1, 2, 1] = 1  # child: X1(t-1) -> X2(t), so X2(t+1)
    cg_tig[0, 2, 0] = 1  # spouse: X0(t') -> X2(t'), t' = t+1
    mb = _markov_blanket_from_tigramite(cg_tig, var_idx=1, num_vars=3)
    # the self-lag edge also makes X1(t+1) a child, whose parent X0(t-1) is a spouse
    assert mb == {(1, -1), (0, -2), (1, 1), (2, 1), (0, 1), (0, -1)}
    assert (1, 0) not in mb


def test_markov_blanket_maps_lag_slots_to_lag_list():
    # lag_list=[2, 5]: slot 1 is lag 2, slot 2 is lag 5
    cg_tig = np.zeros((3, 3, 3), dtype=int)
    cg_tig[0, 1, 1] = 1  # X0(t-2) -> X1(t)
    cg_tig[1, 2, 2] = 1  # X1(t-5) -> X2(t), so X2(t+5)
    cg_tig[0, 2, 1] = 1  # X0(t-2) -> X2(t), so X0(t+3) is a spouse
    mb = _markov_blanket_from_tigramite(cg_tig, 1, 3, lags=[0, 2, 5])
    assert mb == {(0, -2), (2, 5), (0, 3)}


def test_iterative_causal_impute_stepped_lag_list():
    from causalts.ci_tests import ParCorrGPU

    rng = np.random.default_rng(7)
    x = rng.normal(size=300)
    y = np.r_[0.0, 0.0, 2 * x[:-2]] + rng.normal(scale=0.01, size=300)
    clean = pd.DataFrame({"X": x, "Y": y})
    df = clean.copy()
    missing = np.arange(20, 280, 9)
    df.loc[missing, "Y"] = np.nan

    ci = ParCorrGPU(np.zeros((2, 2)), device="cpu")
    out = iterative_causal_impute(
        df, ci, lag_list=[2], include_C=False, n_iter=1, var_em_kwargs={"max_iter": 10}
    )
    rmse = np.sqrt(np.mean((out.loc[missing, "Y"] - clean.loc[missing, "Y"]) ** 2))
    assert rmse < 0.05


def test_iterative_causal_impute_uses_lagged_regressors():
    # Y(t) = 2 X(t-1): regressing Y on X(t) instead of X(t-1) wrecked the
    # VAR warm start (RMSE 0.0103 -> 2.37 on this seed).
    from causalts.ci_tests import ParCorrGPU

    rng = np.random.default_rng(2026)
    x = rng.normal(size=200)
    y = np.r_[0.0, 2 * x[:-1]] + rng.normal(scale=0.01, size=200)
    clean = pd.DataFrame({"X": x, "Y": y})
    df = clean.copy()
    missing = np.arange(20, 180, 8)
    df.loc[missing, "Y"] = np.nan

    ci = ParCorrGPU(np.zeros((2, 2)), device="cpu")
    out = iterative_causal_impute(
        df,
        ci,
        num_lags=1,
        include_C=False,
        n_iter=1,
        var_em_kwargs={"max_iter": 10},
    )
    rmse = np.sqrt(np.mean((out.loc[missing, "Y"] - clean.loc[missing, "Y"]) ** 2))
    assert rmse < 0.05
    pd.testing.assert_frame_equal(out.drop(index=missing), clean.drop(index=missing))


def test_cdnots_discovery_impute_var_em():
    from causalts.cdnots.phase3_utils import run_cdnots
    from causalts.ci_tests import ParCorrGPU

    rng = np.random.default_rng(0)
    T, K = 80, 3
    data = np.cumsum(rng.standard_normal((T, K)), axis=0)
    df = pd.DataFrame(data, columns=["A", "B", "C"])
    mask = rng.random(df.shape) < 0.05
    arr_masked = df.values.copy()
    arr_masked[mask] = np.nan
    df_masked = pd.DataFrame(arr_masked, columns=df.columns)

    ci = ParCorrGPU(data=np.zeros((2, 2)), device="cpu")
    res = run_cdnots(
        df=df_masked,
        indep_test=ci,
        num_lags=1,
        include_C=False,
        alpha=0.05,
        impute="var_em",
        impute_kwargs={"p": 1, "max_iter": 5},
    )
    assert res.cg_tig is not None
    assert res.cg_tig.shape[0] == K
    assert not np.isnan(res.cg_tig).any()


def test_cdnots_discovery_warns_no_nans():
    from causalts.cdnots.phase3_utils import run_cdnots
    from causalts.ci_tests import ParCorrGPU

    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.standard_normal((60, 3)), columns=["A", "B", "C"])
    ci = ParCorrGPU(data=np.zeros((2, 2)), device="cpu")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        run_cdnots(df=df, indep_test=ci, num_lags=1, include_C=False, impute="var_em")
    assert any("no NaN" in str(warning.message) for warning in w)


def test_cli_bad_impute_kwargs(tmp_path):
    from click.testing import CliRunner

    from causalts.cli import main

    df = pd.DataFrame(np.random.randn(30, 2), columns=["X", "Y"])
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "discover",
            str(csv_path),
            "--algorithm",
            "cdnots",
            "--impute",
            "var_em",
            "--impute-kwargs",
            "not-valid-json",
        ],
    )
    assert result.exit_code != 0
