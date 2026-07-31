# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for CEDAR causal discovery algorithm."""

import numpy as np
import pandas as pd
import pytest

from causalts.cedar import Cedar, CedarResult, run_cedar
from causalts.cedar.lag_selection import (
    compute_lag_importance,
    compute_lag_pvalues,
    select_lags,
)
from causalts.ci_tests import ParCorrGPU
from causalts.result import CausalResult

DEVICE = "cpu"


@pytest.fixture
def small_df():
    """Small 3-variable DataFrame: X0 -> X1 -> X2 at lag 1."""
    rng = np.random.default_rng(42)
    T = 80
    x0 = rng.standard_normal(T)
    x1 = np.zeros(T)
    x2 = np.zeros(T)
    for t in range(1, T):
        x1[t] = 0.7 * x0[t - 1] + 0.3 * rng.standard_normal()
        x2[t] = 0.5 * x1[t - 1] + 0.3 * rng.standard_normal()
    return pd.DataFrame({"X0": x0, "X1": x1, "X2": x2})


@pytest.fixture
def ci_test():
    return ParCorrGPU(np.zeros((2, 2)), device=DEVICE)


@pytest.fixture
def ar_df():
    """3-variable AR data: X0->X1->X2 at lag 1, all autocorrelated."""
    rng = np.random.default_rng(42)
    T = 200
    x0 = np.zeros(T)
    x1 = np.zeros(T)
    x2 = np.zeros(T)
    for t in range(1, T):
        x0[t] = 0.6 * x0[t - 1] + 0.5 * rng.standard_normal()
        x1[t] = 0.5 * x0[t - 1] + 0.3 * x1[t - 1] + 0.3 * rng.standard_normal()
        x2[t] = 0.5 * x1[t - 1] + 0.3 * x2[t - 1] + 0.3 * rng.standard_normal()
    return pd.DataFrame({"X0": x0, "X1": x1, "X2": x2})


# --- Lag selection tests ---


def test_compute_lag_importance_shape(small_df):
    data = small_df.values.astype(np.float64)
    result = compute_lag_importance(data, max_lag=2, method="dcor")
    assert result.shape == (3, 3, 3)
    assert result.dtype == np.float64
    assert np.all(result >= -1) and np.all(result <= 1)


def test_compute_lag_importance_lag0_excluded_by_default(small_df):
    data = small_df.values.astype(np.float64)
    result = compute_lag_importance(data, max_lag=2, method="dcor", include_lag0=False)
    # Off-diagonal should be 0 when include_lag0=False; diagonal is always 1
    off_diag = result[:, :, 0].copy()
    np.fill_diagonal(off_diag, 0.0)
    np.testing.assert_array_equal(off_diag, 0.0)
    assert np.all(np.diag(result[:, :, 0]) == 1.0)


def test_compute_lag_importance_pearson(small_df):
    data = small_df.values.astype(np.float64)
    result = compute_lag_importance(data, max_lag=2, method="pearson")
    assert result.shape == (3, 3, 3)
    assert np.all(result >= 0)


def test_compute_lag_pvalues_range(small_df):
    data = small_df.values.astype(np.float64)
    imp = compute_lag_importance(data, max_lag=2, method="dcor")
    pvals = compute_lag_pvalues(data, imp, max_lag=2, n_permutations=50)
    assert pvals.shape == imp.shape
    assert np.all(pvals >= 0) and np.all(pvals <= 1)


def test_select_lags_returns_valid(small_df):
    data = small_df.values.astype(np.float64)
    imp = compute_lag_importance(data, max_lag=2, method="dcor")
    pvals = compute_lag_pvalues(data, imp, max_lag=2, n_permutations=50)
    sel = select_lags(imp, pvals, alpha=0.05)
    assert sel.shape == (3, 3)
    assert np.all(np.isnan(np.diag(sel)))


# --- Core algorithm tests ---


def test_run_cedar_returns_result(ar_df, ci_test):
    result = run_cedar(ar_df, ci_test, max_lag=1, n_permutations=50)
    assert isinstance(result, CedarResult)
    assert isinstance(result, CausalResult)
    assert result.cg_tig.shape == (3, 3, 2)
    assert result.var_names == ["X0", "X1", "X2"]
    assert result.max_lag == 1
    assert result.n_tests > 0


def test_cedar_class_api_matches_functional(small_df, ci_test):
    algo = Cedar(df=small_df, ci_test=ci_test, max_lag=1, n_permutations=50)
    result = algo.run()
    assert isinstance(result, CedarResult)
    assert result.cg_tig.shape[0] == small_df.shape[1]


def test_cedar_no_prune(small_df, ci_test):
    result = run_cedar(small_df, ci_test, max_lag=1, prune=False, n_permutations=50)
    assert result.pruned is False


def test_cedar_with_prune(small_df, ci_test):
    result = run_cedar(small_df, ci_test, max_lag=1, prune=True, n_permutations=50)
    assert result.pruned is True


def test_cedar_lag2(small_df, ci_test):
    result = run_cedar(small_df, ci_test, max_lag=2, n_permutations=50)
    assert result.cg_tig.shape == (3, 3, 3)


def test_cedar_result_repr(small_df, ci_test):
    result = run_cedar(small_df, ci_test, max_lag=1, n_permutations=50)
    r = repr(result)
    assert "CedarResult" in r
    assert "vars=3" in r


def test_cedar_result_has_lag_info(small_df, ci_test):
    result = run_cedar(small_df, ci_test, max_lag=1, n_permutations=50)
    assert result.lag_importance.shape == (3, 3, 2)
    assert result.lag_pvalues.shape == (3, 3, 2)
    assert result.selected_lags.shape == (3, 3)


# --- Integration tests ---


def test_old_sypi_still_works(small_df, tmp_path):
    """Old SYPI class is still importable and functional."""
    from causalts.cedar import SYPI

    ci = ParCorrGPU(np.zeros((2, 2)), device=DEVICE)
    sypi = SYPI(
        data=small_df,
        ci_test=ci,
        max_lag=1,
        lag_sel_algo="dcor",
        p_cond1=0.05,
        p_cond2=0.05,
        make_log=False,
        cache_dir=str(tmp_path / "cache"),
    )
    sypi.run_ts_causality()
    assert sypi.graph is not None


# --- partial_dcor regression tests ---
# These guard against silent degradation of the partial_dcor implementation.
# History: ae859ff accidentally dropped _residualize_targets and
# _compute_one_pair_partial_dcor from lag_selection.py, causing partial_dcor
# to silently fall back to Pearson. The tests below would have caught that.


def test_partial_dcor_unknown_method_raises():
    """_compute_one_pair must reject unknown method names, not silently fall back."""
    data = np.random.default_rng(0).standard_normal((50, 2))
    with pytest.raises(ValueError, match="Unknown lag importance method"):
        compute_lag_importance(data, max_lag=2, method="typo_method")


def test_partial_dcor_scores_differ_from_dcor(ar_df):
    """partial_dcor must produce different scores from plain dcor on AR data.

    If they're identical, partial_dcor is silently running dcor (or Pearson),
    meaning the residualization step is not executing.
    """
    data = ar_df.values.astype(np.float64)
    imp_dcor = compute_lag_importance(data, max_lag=2, method="dcor")
    imp_pdcor = compute_lag_importance(data, max_lag=2, method="partial_dcor")
    # Scores should differ because partial_dcor residualises the target first
    assert not np.allclose(imp_dcor, imp_pdcor, atol=1e-6), (
        "partial_dcor scores are identical to dcor — " "residualization is not running"
    )


def test_partial_dcor_regression_f1(ci_test):
    """Frozen regression: partial_dcor default must reproduce known F1.

    This is the canary. If partial_dcor is silently replaced by another
    method, F1 on this fixed benchmark will drop noticeably (from ~0.95 to
    ~0.90 on d=5, or from ~0.82 to ~0.76 on d=10 as observed historically).
    """
    from causalts.synthetic_data.synthetic_datasets import SCPGraphGenerator
    from causalts.utils.helpers import precision_recall_3d

    def _eval(est, gt):
        gt = gt.copy()
        for lag in range(gt.shape[2]):
            np.fill_diagonal(gt[:, :, lag], 0)
        d = gt.shape[0]
        est = est[:d, :d, : gt.shape[2]]
        for lag in range(est.shape[2]):
            np.fill_diagonal(est[:, :, lag], 0)
        prec, rec = precision_recall_3d(gt, est)
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    f1_values = []
    for seed in [42, 56, 45, 123]:
        gen = SCPGraphGenerator(
            n_vars=10,
            max_lag=2,
            density="sparse",
            dependency_funcs=("lin", "lin", "nl1", "nl2"),
        )
        res = gen.sample(seed=seed, T=300, max_tries=10000)
        df, gt = res["df"], res["ground_truth"]
        from causalts.ci_tests import PartialCorr

        ci = PartialCorr(df.values)
        r = run_cedar(df, ci, alpha_cond1=0.01, alpha_cond2=0.01, max_lag=2)
        f1_values.append(_eval(r.cg_tig, gt))

    mean_f1 = np.mean(f1_values)
    # Known good value with partial_dcor: ~0.824 (notebook benchmark, d=10)
    # Threshold set conservatively at 0.75 to allow seed variance while
    # still catching a silent regression to Pearson (which yields ~0.63).
    assert mean_f1 >= 0.75, (
        f"partial_dcor F1={mean_f1:.3f} is below threshold 0.75 — "
        "likely a silent regression in lag_selection.py"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
