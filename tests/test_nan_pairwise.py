# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""
NaN / pairwise-complete tests for all CI tests.

Covers:
  1. _has_nan flag set correctly on data assignment
  2. __call__ returns valid (non-NaN) p-value when data contains NaNs
  3. __call__ returns p=1.0 (not NaN) when fewer than min_samples complete rows
  4. Results match manually pre-filtered (complete-case) data
  5. p=nan never reaches CDNOTS skeleton (integration guard)
"""

import math

import numpy as np
import pandas as pd

DEVICE = "cpu"
T = 80
K = 4


def _make_data(T=T, K=K, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, K)).astype(np.float64)


def _inject_nan(data, col, rows):
    d = data.copy()
    d[rows, col] = np.nan
    return d


def _all_ci_tests(data):
    from causalts.ci_tests import (
        CMIKNN,
        DFCITGPU,
        GCMIGPU,
        KCIGPU,
        RCOTGPU,
        CMIknnGPU,
        ParCorrGPU,
        PartialCorr,
        SigKCIGPU,
        SplitKCIGPU,
    )

    return [
        ("parcorr-gpu", ParCorrGPU(data, device=DEVICE)),
        ("kci", KCIGPU(data, device=DEVICE)),
        ("rcot", RCOTGPU(data=data, device=DEVICE)),
        ("cmiknn", CMIKNN(data=data, k=0.2, sig_samples=20)),
        ("cmiknn-gpu", CMIknnGPU(data, device=DEVICE, sig_samples=20)),
        ("parcorr", PartialCorr(data=data)),
        ("splitkci", SplitKCIGPU(data, device=DEVICE, sig_samples=20)),
        ("dfcit", DFCITGPU(data, device=DEVICE, n_perm=20)),
        ("sigkci", SigKCIGPU(data, device=DEVICE, sig_samples=20)),
        ("gcmi", GCMIGPU(data, device=DEVICE)),
    ]


def test_has_nan_set_on_assignment():
    clean = _make_data()
    dirty = _inject_nan(clean, col=0, rows=[5, 10, 20])

    for name, ci in _all_ci_tests(clean):
        assert not ci._has_nan, f"{name}: _has_nan should be False on clean data"
        ci.data = dirty
        assert ci._has_nan, f"{name}: _has_nan should be True after assigning NaN data"
        ci.data = clean
        assert not ci._has_nan, f"{name}: _has_nan should reset to False on clean data"


def test_call_returns_valid_pval_with_nan():
    clean = _make_data()
    nan_rows = np.arange(0, T, 10)
    dirty = _inject_nan(clean, col=0, rows=nan_rows)

    for name, ci in _all_ci_tests(dirty):
        p, stat = ci(0, 1, [2])
        assert not math.isnan(p), f"{name}: p-value is NaN with NaN data"
        assert not math.isinf(p), f"{name}: p-value is Inf with NaN data"
        assert 0.0 <= p <= 1.0, f"{name}: p-value {p} out of [0,1]"


def test_too_few_complete_rows_returns_p1():
    clean = _make_data(T=100)
    dirty = clean.copy()
    dirty[5:, 0] = np.nan

    for name, ci in _all_ci_tests(dirty):
        p, stat = ci(0, 1, [2])
        assert (
            p == 1.0
        ), f"{name}: expected p=1.0 when <min_samples complete rows, got p={p}"
        assert not math.isnan(p), f"{name}: p is NaN instead of 1.0"


def test_pairwise_matches_manual_filter():
    rng = np.random.default_rng(42)
    z = rng.standard_normal(T)
    x0 = z + 0.3 * rng.standard_normal(T)
    x1 = z + 0.3 * rng.standard_normal(T)
    x2 = z
    clean = np.column_stack([x0, x1, x2, rng.standard_normal(T)])

    nan_rows = np.array([3, 7, 15, 22, 30, 45])
    dirty = _inject_nan(clean, col=0, rows=nan_rows)

    complete_mask = ~np.isnan(dirty[:, [0, 1, 2]]).any(axis=1)
    clean_subset = dirty[complete_mask]

    for name, ci_nan in _all_ci_tests(dirty):
        _, ci_clean = next((n, c) for n, c in _all_ci_tests(clean_subset) if n == name)
        p_nan, _ = ci_nan(0, 1, [2])
        p_clean, _ = ci_clean(0, 1, [2])

        assert not math.isnan(p_nan), f"{name}: NaN path returned p=NaN"
        assert not math.isnan(p_clean), f"{name}: clean path returned p=NaN"
        assert abs(p_nan - p_clean) < 0.4, (
            f"{name}: NaN-path p={p_nan:.3f} vs complete-case p={p_clean:.3f} "
            f"diverge by more than 0.4"
        )


def test_p_nan_never_reaches_skeleton():
    from causalts.cdnots.phase3_utils import run_cdnots
    from causalts.ci_tests import ParCorrGPU

    rng = np.random.default_rng(5)
    data = rng.standard_normal((60, 3))
    arr = data.copy()
    arr[rng.random(arr.shape) < 0.20] = np.nan
    df = pd.DataFrame(arr, columns=["A", "B", "C"])

    ci = ParCorrGPU(data=np.zeros((2, 2)), device=DEVICE)
    res = run_cdnots(
        df=df,
        indep_test=ci,
        num_lags=1,
        include_C=False,
        alpha=0.05,
    )
    assert not np.isnan(res.cg_tig).any()
    assert not np.isinf(res.cg_tig).any()
