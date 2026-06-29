# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for mixed discrete-continuous CI tests and synthetic datasets."""

import warnings

import numpy as np
import pandas as pd

from causalts.ci_tests import ParCorrGPU
from causalts.ci_tests.cmiknn_mixed_gpu import CMIknnMixedGPU
from causalts.ci_tests.stratified_cit import StratifiedCIT, compute_discrete_indices
from causalts.synthetic_data.mixed_datasets import mixed_a, mixed_b, mixed_c

DEVICE = "cpu"


def test_mixed_a_shape_and_types():
    r = mixed_a(seed=42, T=100)
    assert r["df"].shape == (100, 5)
    assert r["discrete_cols"] == ["R"]
    assert set(r["df"]["R"].unique()).issubset({0.0, 1.0})


def test_mixed_b_shape_and_types():
    r = mixed_b(seed=42, T=100)
    assert r["df"].shape == (100, 4)
    assert r["discrete_cols"] == ["C"]
    assert (r["df"]["C"] >= 0).all()
    assert (r["df"]["C"] == r["df"]["C"].astype(int)).all()


def test_mixed_c_shape_and_types():
    r = mixed_c(seed=42, T=100)
    assert r["df"].shape == (100, 6)
    assert r["discrete_cols"] == ["B1", "B2"]
    assert set(r["df"]["B1"].unique()).issubset({0.0, 1.0})


def test_compute_discrete_indices_basic():
    df = pd.DataFrame({"A": [0], "B": [1], "C": [2]})
    lags = [0, 1, 2]
    indices = compute_discrete_indices(["B"], df, lags, include_C=False)
    assert indices == [1, 4, 7]


def test_compute_discrete_indices_with_C():
    df = pd.DataFrame({"A": [0], "B": [1]})
    lags = [0, 1]
    indices = compute_discrete_indices(["B"], df, lags, include_C=True)
    assert indices == [1, 4]


def _make_independent_mixed(T=150, seed=0):
    rng = np.random.default_rng(seed)
    regime = rng.integers(0, 2, T)
    x = np.where(regime == 0, rng.standard_normal(T) * 0.5, rng.standard_normal(T) * 2)
    y = rng.standard_normal(T)
    data = np.column_stack([x, y, regime.astype(float)])
    return data


def _make_dependent_mixed(T=150, seed=0):
    rng = np.random.default_rng(seed)
    regime = rng.integers(0, 2, T)
    x = np.where(regime == 0, rng.standard_normal(T) * 0.5, rng.standard_normal(T) * 2)
    y = 0.8 * x + 0.3 * rng.standard_normal(T)
    data = np.column_stack([x, y, regime.astype(float)])
    return data


def test_stratified_cit_type_i_error():
    alpha, n_trials = 0.05, 100
    rejections = 0
    for seed in range(n_trials):
        data = _make_independent_mixed(T=100, seed=seed)
        inner = ParCorrGPU(data, device=DEVICE)
        cit = StratifiedCIT(data, discrete_cols=[2], inner_cit=inner)
        p, _ = cit(0, 1, [2])
        if p < alpha:
            rejections += 1
    rate = rejections / n_trials
    assert rate < 3 * alpha


def test_stratified_cit_power():
    alpha, n_trials = 0.05, 50
    rejections = 0
    for seed in range(n_trials):
        data = _make_dependent_mixed(T=100, seed=seed)
        inner = ParCorrGPU(data, device=DEVICE)
        cit = StratifiedCIT(data, discrete_cols=[2], inner_cit=inner)
        p, _ = cit(0, 1, [2])
        if p < alpha:
            rejections += 1
    power = rejections / n_trials
    assert power > 0.6


def test_stratified_cit_no_disc_in_z():
    data = _make_independent_mixed(T=100, seed=0)
    inner = ParCorrGPU(data, device=DEVICE)
    cit = StratifiedCIT(data, discrete_cols=[2], inner_cit=inner)
    p_strat, _ = cit(0, 1, [])
    p_inner, _ = inner(0, 1, [])
    assert abs(p_strat - p_inner) < 1e-9


def test_stratified_cit_fisher_combine():
    from scipy.stats import chi2

    pvals = [0.5, 0.5]
    stat = -2 * sum(np.log(p) for p in pvals)
    expected = float(1 - chi2.cdf(stat, 2 * len(pvals)))
    result = StratifiedCIT._fisher_combine(pvals)
    assert abs(result - expected) < 1e-10


def test_cmiknn_mixed_no_nans():
    data = _make_independent_mixed(T=80, seed=1)
    ci = CMIknnMixedGPU(data, discrete_cols=[2], device=DEVICE, sig_samples=20)
    p, stat = ci(0, 1, [2])
    assert 0.0 <= p <= 1.0
    assert not np.isnan(p)


def test_cmiknn_mixed_dep_vs_indep():
    indep = _make_independent_mixed(T=100, seed=5)
    dep = _make_dependent_mixed(T=100, seed=5)
    ci_indep = CMIknnMixedGPU(indep, discrete_cols=[2], device=DEVICE, sig_samples=50)
    ci_dep = CMIknnMixedGPU(dep, discrete_cols=[2], device=DEVICE, sig_samples=50)
    p_indep, _ = ci_indep(0, 1, [2])
    p_dep, _ = ci_dep(0, 1, [2])
    assert p_dep < p_indep


def test_cdnots_discrete_cols_propagation():
    from causalts.cdnots.phase3_utils import run_cdnots

    r = mixed_a(seed=42, T=80)
    df = r["df"]
    placeholder = np.zeros((2, 2))
    inner = ParCorrGPU(placeholder, device=DEVICE)
    cit = StratifiedCIT(placeholder, discrete_cols=[], inner_cit=inner)

    run_cdnots(
        df=df,
        indep_test=cit,
        num_lags=r["max_lag"],
        include_C=False,
        alpha=0.05,
        show_progress=False,
        verbose=False,
        discrete_cols=r["discrete_cols"],
    )
    assert len(cit.discrete_cols) > 0


def test_auto_detect_warns_incompatible_test():
    from causalts.cdnots.phase3_utils import run_cdnots

    r = mixed_a(seed=42, T=100)
    df = r["df"]
    ci = ParCorrGPU(np.zeros((2, 2)), device=DEVICE)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        run_cdnots(
            df=df,
            indep_test=ci,
            num_lags=r["max_lag"],
            include_C=False,
            alpha=0.05,
            show_progress=False,
            verbose=False,
        )

    disc_warnings = [x for x in w if "discrete" in str(x.message).lower()]
    assert len(disc_warnings) == 1
    assert "R" in str(disc_warnings[0].message)
    assert "StratifiedCIT" in str(disc_warnings[0].message)


def test_auto_detect_no_warn_continuous_data():
    from causalts.cdnots.phase3_utils import run_cdnots

    df_cont = pd.DataFrame(np.random.randn(80, 3), columns=["A", "B", "C"])
    ci = ParCorrGPU(np.zeros((2, 2)), device=DEVICE)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        run_cdnots(
            df=df_cont,
            indep_test=ci,
            num_lags=1,
            include_C=False,
            alpha=0.05,
            show_progress=False,
            verbose=False,
        )

    disc_warnings = [x for x in w if "discrete" in str(x.message).lower()]
    assert len(disc_warnings) == 0


def test_auto_detect_opt_out():
    from causalts.cdnots.phase3_utils import run_cdnots

    r = mixed_a(seed=42, T=80)
    df = r["df"]
    ci = ParCorrGPU(np.zeros((2, 2)), device=DEVICE)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        run_cdnots(
            df=df,
            indep_test=ci,
            num_lags=r["max_lag"],
            include_C=False,
            alpha=0.05,
            show_progress=False,
            verbose=False,
            discrete_cols=[],
        )

    disc_warnings = [x for x in w if "discrete" in str(x.message).lower()]
    assert len(disc_warnings) == 0


def test_auto_detect_no_double_wrap():
    from causalts.cdnots.phase3_utils import run_cdnots

    r = mixed_a(seed=42, T=80)
    df = r["df"]
    inner = ParCorrGPU(np.zeros((2, 2)), device=DEVICE)
    cit = StratifiedCIT(np.zeros((2, 2)), discrete_cols=[], inner_cit=inner)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        run_cdnots(
            df=df,
            indep_test=cit,
            num_lags=r["max_lag"],
            include_C=False,
            alpha=0.05,
            show_progress=False,
            verbose=False,
        )

    disc_warnings = [x for x in w if "discrete" in str(x.message).lower()]
    assert len(disc_warnings) == 0
    assert isinstance(cit.inner_cit, ParCorrGPU)


def test_detect_discrete_cols_heuristic():
    from causalts.cdnots.phase3_utils import _detect_discrete_cols

    rng = np.random.default_rng(0)
    T = 100
    df = pd.DataFrame(
        {
            "binary": rng.integers(0, 2, T).astype(float),
            "count": rng.poisson(3, T).astype(float),
            "cont": rng.standard_normal(T),
            "high_card_int": np.arange(T, dtype=float),
        }
    )
    detected = _detect_discrete_cols(df, T)
    assert "binary" in detected
    assert "count" in detected
    assert "cont" not in detected
    assert "high_card_int" not in detected
