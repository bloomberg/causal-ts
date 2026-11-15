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
    select_all_significant_lags,
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


def test_max_prune_path_is_deprecated(small_df, ci_test):
    """max_prune_path was stored but never read."""
    import warnings

    with pytest.warns(DeprecationWarning, match="max_prune_path") as rec:
        Cedar(df=small_df, ci_test=ci_test, max_lag=1, max_prune_path=2)
    hits = [w for w in rec if "max_prune_path" in str(w.message)]
    assert len(hits) == 1 and hits[0].filename == __file__  # points at the caller
    with pytest.warns(DeprecationWarning, match="max_prune_path") as rec:
        run_cedar(small_df, ci_test, max_lag=1, n_permutations=20, max_prune_path=2)
    hits = [w for w in rec if "max_prune_path" in str(w.message)]
    assert len(hits) == 1 and hits[0].filename == __file__
    from causalts.algorithms import run_algorithm

    with pytest.warns(DeprecationWarning, match="max_prune_path") as rec:
        run_algorithm(
            "cedar", small_df, ci_test, max_lag=1, n_permutations=20, max_prune_path=4
        )
    hits = [w for w in rec if "max_prune_path" in str(w.message)]
    assert len(hits) == 1 and hits[0].filename == __file__
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message="max_prune_path", category=DeprecationWarning
        )
        Cedar(df=small_df, ci_test=ci_test, max_lag=1)
        run_cedar(small_df, ci_test, max_lag=1, n_permutations=20)


def test_impute_kwargs_docstring_example_runs(small_df, ci_test):
    """The run_cedar docstring example must use max_iter, which var_em accepts."""
    import ast
    import re

    example = re.search(r"``(\{\"p\".*?\})``", " ".join(run_cedar.__doc__.split()))
    kwargs = ast.literal_eval(example.group(1))
    df = small_df.copy()
    df.iloc[5:8, 1] = np.nan
    result = run_cedar(
        df,
        ci_test,
        max_lag=1,
        n_permutations=20,
        impute="var_em",
        impute_kwargs=kwargs,
    )
    assert result.cg_tig.shape == (3, 3, 2)


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


def _shared_shock_trend(seed, T=400):
    """X(t-1) -> Y plus a shared contemporaneous shock and a linear trend."""
    rng = np.random.default_rng(seed)
    shock = rng.standard_normal(T)
    x = np.zeros(T)
    y = np.zeros(T)
    for t in range(1, T):
        x[t] = 0.5 * x[t - 1] + shock[t] + rng.standard_normal()
        y[t] = 0.8 * x[t - 1] + shock[t] + rng.standard_normal()
    trend = np.linspace(0, 3, T)
    return pd.DataFrame({"X": x + trend, "Y": y + trend})


@pytest.mark.parametrize(
    "multi_lag, sweep_all_lags", [(False, False), (True, False), (True, True)]
)
def test_include_c_keeps_lagged_edge_whose_strongest_lag_is_0(
    multi_lag, sweep_all_lags
):
    """include_C blanked an X-X pair whose top lag was 0 instead of
    falling back to its significant lag 1, dropping the true edge."""
    df = _shared_shock_trend(0)
    ci = ParCorrGPU(np.zeros((2, 2)), device=DEVICE)
    for include_C in (False, True):
        res = run_cedar(
            df,
            ci,
            max_lag=2,
            include_C=include_C,
            multi_lag=multi_lag,
            sweep_all_lags=sweep_all_lags,
            n_permutations=50,
        )
        assert res.cg_tig[0, 1, 1], include_C
        assert not res.cg_tig[:2, :2, 0].any()  # still no X-X lag-0 edge


def test_mci_pruning_never_conditions_on_the_target_itself():
    """Pruning X -> Y at lag 0 conditioned on Y(t) via Y -> X and
    removed the true contemporaneous adjacency."""
    rng = np.random.default_rng(0)
    T = 500
    x = rng.standard_normal(T)
    y = 0.8 * x + rng.standard_normal(T)
    z = rng.standard_normal(T)
    z[1:] += 0.6 * y[:-1]
    df = pd.DataFrame({"X": x, "Y": y, "Z": z})
    ci = ParCorrGPU(np.zeros((2, 2)), device=DEVICE)
    kw = dict(max_lag=1, include_lag0=True, include_C=False, n_permutations=50)
    unpruned = run_cedar(df, ci, prune=False, **kw).cg_tig
    assert unpruned[0, 1, 0] and unpruned[1, 0, 0]  # the setting of the issue
    res = run_cedar(df, ci, prune=True, **kw)
    pruned = res.cg_tig
    assert pruned[0, 1, 0] or pruned[1, 0, 0]
    assert pruned[1, 2, 1]
    # Kept lag-0 pairs carry one statistic per direction; plotting needs
    # lag-0 symmetry and must not raise.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res.plot()
    plt.close("all")


@pytest.mark.parametrize("multi_lag", [False, True])
def test_include_c_lag_selection_uses_the_same_fdr_family(monkeypatch, multi_lag):
    """C cells and the forced X-X lag 0 must not enlarge the BH family.

    Only X0(t-1) -> X1 has a small lag p-value (0.01). Over the four lagged X-X
    hypotheses BH accepts it (0.01 * 4 < 0.05); padding the family with the
    C cells and X-X lag 0 at p = 1 (18 cells) rejected it, so a lag found
    without C was lost with C.
    """
    import causalts.cedar.discovery as cd

    def fake_pvalues(data, lag_importance, max_lag, **kw):
        p = np.ones_like(lag_importance)
        p[0, 1, 1] = 0.01
        return p

    monkeypatch.setattr(cd, "compute_lag_pvalues", fake_pvalues)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((200, 2)), columns=["X0", "X1"])
    ci = ParCorrGPU(np.zeros((2, 2)), device=DEVICE)
    for include_C in (False, True):
        res = run_cedar(
            df,
            ci,
            max_lag=2,
            include_C=include_C,
            multi_lag=multi_lag,
            prune=False,
            n_permutations=50,
        )
        assert res.selected_lags[0, 1] == 1, include_C
        assert np.isnan(res.selected_lags[1, 0]), include_C


def test_select_lags_exclude_mask_leaves_cells_out_of_the_family():
    imp = np.ones((2, 2, 3))
    p = np.ones((2, 2, 3))
    p[0, 1, 1] = 0.01
    p[:, :, 0] = 0.001  # lag 0: strongest, but not a hypothesis here
    excl = np.zeros(p.shape, dtype=bool)
    excl[:, :, 0] = True
    without = select_lags(imp, p, alpha=0.05, include_lag0=False)
    with_excl = select_lags(imp, p, alpha=0.05, include_lag0=True, exclude_mask=excl)
    np.testing.assert_array_equal(without, with_excl)
    assert with_excl[0, 1] == 1
    np.testing.assert_array_equal(
        select_all_significant_lags(imp, p, include_lag0=False),
        select_all_significant_lags(imp, p, include_lag0=True, exclude_mask=excl),
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


def test_autoreg_self_loops_have_no_pvalue(ar_df, ci_test):
    """Self-loops added without a test report p = NaN, not 1.0 ("no evidence")."""
    from causalts.inspection import edges_from_graph

    result = run_cedar(ar_df, ci_test, max_lag=1, n_permutations=50)
    g, pv = result.cg_tig, result.pvalue_matrix
    loops = [(i, j, k) for i, j, k in zip(*np.nonzero(g)) if i == j]
    assert loops
    for i, j, k in loops:
        assert np.isnan(pv[i, j, k])
    # Edges found by the Cond1 test keep their p-value.
    cross = g.astype(bool) & ~np.eye(3, dtype=bool)[:, :, None]
    assert cross.any() and np.isfinite(pv[cross]).all()
    for e in edges_from_graph(g, result.var_names, pv):
        if e["source"] == e["target"]:
            assert e["pvalue"] is None
        else:
            assert e["pvalue"] is not None and e["pvalue"] < 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
