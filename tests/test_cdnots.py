# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for CDNOTS causal discovery algorithm."""

import numpy as np

from causalts.cdnots.phase3_utils import run_cdnots
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
