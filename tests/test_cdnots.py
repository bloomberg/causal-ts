# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for CDNOTS causal discovery algorithm."""

import numpy as np

from causalts.cdnots.phase3_utils import (
    build_pvalue_matrix,
    run_cdnots,
    run_cdnots_plus,
)
from causalts.cdnots.result import CdnotsResult
from causalts.ci_tests import ParCorrGPU

DEVICE = "cpu"


def test_cdnots_basic_no_C(small_df):
    ci = ParCorrGPU(np.zeros((2, 2)), device=DEVICE)
    res = run_cdnots(
        df=small_df,
        indep_test=ci,
        num_lags=1,
        include_C=False,
        alpha=0.05,
        show_progress=False,
        verbose=False,
    )
    assert isinstance(res, CdnotsResult)
    K = small_df.shape[1]
    assert res.cg_tig.shape == (K, K, 2)
    assert not np.isnan(res.cg_tig).any()
    assert not np.isinf(res.cg_tig).any()
    assert set(np.unique(res.cg_tig)).issubset({0, 1})


def test_cdnots_with_C(small_df):
    ci = ParCorrGPU(np.zeros((2, 2)), device=DEVICE)
    res = run_cdnots(
        df=small_df,
        indep_test=ci,
        num_lags=1,
        include_C=True,
        alpha=0.05,
        show_progress=False,
        verbose=False,
    )
    K = small_df.shape[1] + 1  # +1 for C
    assert res.cg_tig.shape == (K, K, 2)
    assert not np.isnan(res.cg_tig).any()


def test_cdnots_returns_pvals(small_df):
    ci = ParCorrGPU(np.zeros((2, 2)), device=DEVICE)
    res = run_cdnots(
        df=small_df,
        indep_test=ci,
        num_lags=1,
        include_C=False,
        alpha=0.05,
        show_progress=False,
        verbose=False,
    )
    assert res.pvals is not None


def test_cdnots_lag_2(small_df):
    ci = ParCorrGPU(np.zeros((2, 2)), device=DEVICE)
    res = run_cdnots(
        df=small_df,
        indep_test=ci,
        num_lags=2,
        include_C=False,
        alpha=0.05,
        show_progress=False,
        verbose=False,
    )
    K = small_df.shape[1]
    assert res.cg_tig.shape == (K, K, 3)


def test_cdnots_stable_mode(small_df):
    ci = ParCorrGPU(np.zeros((2, 2)), device=DEVICE)
    res = run_cdnots(
        df=small_df,
        indep_test=ci,
        num_lags=1,
        include_C=False,
        alpha=0.05,
        stable=True,
        show_progress=False,
        verbose=False,
    )
    assert res.cg_tig is not None
    assert not np.isnan(res.cg_tig).any()


def test_build_pvalue_matrix_cause_effect_order():
    """pvals tuples are (effect_node, cause_node, cond_set, p); a candidate parent
    at lag tau (node = var + tau*num_vars) must land at M[cause_var, effect_var,
    tau], not be silently dropped by the e_lag!=0 filter.

    Regression test for a bug where skeleton_discovery_pervar/mci_skeleton
    appended (cause, effect, ...) -- the opposite order -- which caused the
    e_lag check to filter on the CAUSE's lag instead of the effect's, discarding
    every lagged (tau>=1) test and corrupting downstream AUROC scoring.
    """
    num_vars, max_lag = 3, 2
    # effect = var1 at lag0 -> node = 1 + 0*3 = 1
    # cause  = var0 at lag2 -> node = 0 + 2*3 = 6
    pvals = [(1, 6, (), 0.01)]
    M = build_pvalue_matrix(pvals, num_vars, max_lag)
    assert M[0, 1, 2] == 0.01
    assert np.isnan(M[1, 0, 2])  # not written to the transposed cell
    assert np.all(np.isnan(M[:, :, 0])) or M[0, 1, 0] != 0.01


def test_cdnots_plus_pvalue_matrix_has_lagged_entries(small_df):
    """small_df has genuine lag-1 dependencies (X0->X1, X1->X2); the CDNOTS+
    pvalue_matrix must have non-NaN entries at lag>=1, not just lag 0 (the
    symptom of the cause/effect argument-order bug above).
    """
    ci = ParCorrGPU(np.zeros((2, 2)), device=DEVICE)
    res = run_cdnots_plus(
        df=small_df,
        indep_test=ci,
        num_lags=1,
        include_C=False,
        alpha=0.05,
        show_progress=False,
        verbose=False,
        return_pvals=True,
    )
    assert res.pvalue_matrix is not None
    assert not np.all(np.isnan(res.pvalue_matrix[:, :, 1:]))


def test_cdnots_entry_points_keep_released_positional_order():
    """Pin which parameters stay positional; everything after is keyword-only.

    A positional call that reaches the keyword-only options fails loudly
    instead of binding its value to a different option.
    """
    import importlib
    import inspect

    cases = [
        ("phase3_utils", "cdnots_discovery", ["df", "indep_test", "num_lags", "lag_list", "include_C", "c_array", "c_preset", "alpha", "stable", "return_pvals", "max_degree", "max_combinations", "priority", "verbose", "show_progress", "impute", "impute_kwargs", "discrete_cols", "background_knowledge", "knowledge", "lag_verify", "alpha_lv"], ["contemp_collider_rule", "orient_margin", "collider_search"]),
        ("phase3_utils", "run_cdnots", ["df", "indep_test", "num_lags", "lag_list", "include_C", "c_array", "c_preset", "alpha", "stable", "max_degree", "max_combinations", "priority", "verbose", "show_progress", "impute", "impute_kwargs", "discrete_cols", "background_knowledge", "knowledge", "lag_verify", "alpha_lv", "return_pvals"], ["orient_margin", "collider_search"]),
        ("phase3_utils", "run_cdnots_plus", ["df", "indep_test", "num_lags", "lag_list", "include_C", "c_array", "c_preset", "alpha", "max_degree", "max_conds_py", "max_conds_px", "priority", "verbose", "show_progress", "impute", "impute_kwargs", "discrete_cols", "background_knowledge", "knowledge", "lag_verify", "alpha_lv", "return_pvals"], ["legacy_mci_conds", "orient_margin", "return_prefix"]),
        ("skeleton_discovery", "skeleton_discovery_pervar", ["cg", "alpha", "ci_test", "max_combinations", "background_knowledge", "verbose", "show_progress", "return_pvals", "max_degree", "num_lags"], ["include_C", "num_c_cols"]),
        ("skeleton_discovery", "mci_skeleton", ["cg", "ci_test", "alpha", "lagged_parents", "num_lags", "include_C", "num_c_cols", "max_conds_py", "max_conds_px", "verbose", "show_progress", "return_pvals", "no_of_var_ext", "background_knowledge"], ["legacy_mci_conds"]),
    ]  # fmt: skip
    for module, name, positional, keyword_only in cases:
        fn = getattr(importlib.import_module(f"causalts.cdnots.{module}"), name)
        params = inspect.signature(fn).parameters.values()
        assert [p.name for p in params if p.kind is p.POSITIONAL_OR_KEYWORD] == (
            positional
        ), name
        assert [
            p.name for p in params if p.kind is p.KEYWORD_ONLY
        ] == keyword_only, name


def test_cdnots_discovery_rejects_an_unknown_collider_rule():
    import pandas as pd
    import pytest

    from causalts.cdnots.phase3_utils import cdnots_discovery

    df = pd.DataFrame(np.random.default_rng(0).standard_normal((50, 2)))
    with pytest.raises(ValueError, match="contemp_collider_rule"):
        cdnots_discovery(
            df,
            ParCorrGPU(np.zeros((2, 2)), device="cpu"),
            contemp_collider_rule="majorit",
        )
