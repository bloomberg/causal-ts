# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the temporal-aggregation / subsampling detector.

``detect_subsampling`` asks one question: do the residuals of a VAR fit show
contemporaneous correlation? Under a genuinely fast-sampled process the
innovations are contemporaneously independent, so they should not. Aggregating
or subsampling that process mixes innovations across the discarded steps, which
induces exactly that correlation -- so its presence is evidence of aggregation.

The fixtures below build a stable VAR(1) with *contemporaneously independent*
innovations, then subsample it by a factor ``k``. ``k=1`` is the negative
control and must not flag; ``k>1`` must.
"""

import numpy as np
import pandas as pd
import pytest

from causalts.utils import DetectionResult, detect_subsampling


def _random_stable_A(d, seed, spectral_radius=0.7, density=0.3, self_loop=0.3):
    """Random sparse transition matrix rescaled to a given spectral radius."""
    rng = np.random.default_rng(seed)
    A = np.zeros((d, d))
    mask = rng.random((d, d)) < density
    np.fill_diagonal(mask, False)
    A[mask] = rng.normal(0.0, 1.0, size=mask.sum())
    np.fill_diagonal(A, self_loop)
    eig = np.max(np.abs(np.linalg.eigvals(A)))
    if eig > 0:
        A = A * (spectral_radius / eig)
    return A


def _generate_fast(d=5, T=12000, seed=0, burn_in=500):
    """Fast VAR(1) with i.i.d. (contemporaneously independent) Student-t innovations."""
    rng = np.random.default_rng(seed)
    A = _random_stable_A(d, seed=seed)
    total = T + burn_in
    df = 5.0
    e = rng.standard_t(df=df, size=(total, d)) / np.sqrt(df / (df - 2.0))
    X = np.zeros((total, d))
    for t in range(1, total):
        X[t] = A @ X[t - 1] + e[t]
    return X[burn_in:]


def _subsample(X, k):
    return X[::k].copy()


# ----------------------------------------------------------------------
# Core detection behaviour
# ----------------------------------------------------------------------


def test_no_subsampling_not_flagged():
    """k=1 (genuinely fast-sampled): residuals are the i.i.d. innovations, so the
    detector must NOT flag."""
    X = _subsample(_generate_fast(d=5, T=12000, seed=0), 1)
    res = detect_subsampling(X, alpha=0.05)
    assert isinstance(res, DetectionResult)
    assert not res.subsampling_flagged
    assert res.verdict == "NO AGGREGATION SIGNATURE"


def test_subsampling_flagged():
    """k=2 (subsampled): aggregated innovations are contemporaneously correlated,
    so the detector must flag."""
    X = _subsample(_generate_fast(d=5, T=12000, seed=0), 2)
    res = detect_subsampling(X, alpha=0.05)
    assert res.subsampling_flagged
    assert res.verdict == "SUBSAMPLING/AGGREGATION CANDIDATE"
    assert res.max_abs_offdiag_corr > 0.1
    assert 0.0 <= res.contemp_pvalue <= 1.0


def test_dataframe_input_matches_ndarray():
    """DataFrame input is accepted, and gives the same answer as its .values."""
    X = _subsample(_generate_fast(d=4, T=8000, seed=1), 2)
    from_df = detect_subsampling(pd.DataFrame(X), alpha=0.05)
    from_arr = detect_subsampling(X, alpha=0.05)
    assert from_df.subsampling_flagged
    assert from_df.contemp_pvalue == from_arr.contemp_pvalue
    assert from_df.contemp_stat == from_arr.contemp_stat


def test_power_grows_with_rate():
    """More aggregation (larger k) raises the off-diagonal residual correlation."""
    base = _generate_fast(d=5, T=12000, seed=3)
    r2 = detect_subsampling(_subsample(base, 2)).max_abs_offdiag_corr
    r4 = detect_subsampling(_subsample(base, 4)).max_abs_offdiag_corr
    assert r4 > r2


# ----------------------------------------------------------------------
# Null calibration and the `null=` switch
# ----------------------------------------------------------------------


def test_bootstrap_null_calibrated_short_T():
    """The bootstrap null keeps the k=1 false-positive rate near alpha at short T,
    where the asymptotic chi2 null over-fires."""
    fp = [
        detect_subsampling(
            _subsample(_generate_fast(d=10, T=200, seed=s), 1),
            null="bootstrap",
            n_perm=99,
        ).subsampling_flagged
        for s in range(20)
    ]
    assert np.mean(fp) <= 0.2


@pytest.mark.parametrize(
    "T,d,expected", [(300, 4, "bootstrap"), (500, 4, "asymptotic")]
)
def test_auto_null_picks_by_sample_size(T, d, expected):
    """`null="auto"` documents "bootstrap when T < 100*d, else asymptotic".

    Pinned by agreement with the explicit setting rather than by inspecting
    internals, so the rule can be refactored but not silently changed.
    """
    X = np.random.default_rng(7).standard_normal((T, d))
    auto = detect_subsampling(X, null="auto").contemp_pvalue
    assert auto == detect_subsampling(X, null=expected).contemp_pvalue


def test_bootstrap_is_deterministic():
    """The bootstrap draws from a fixed internal seed, so repeated calls on the
    same data must agree -- otherwise a flag near alpha would be a coin flip."""
    X = np.random.default_rng(11).standard_normal((300, 4))
    first = detect_subsampling(X, null="bootstrap", n_perm=49)
    second = detect_subsampling(X, null="bootstrap", n_perm=49)
    assert first.contemp_pvalue == second.contemp_pvalue


def test_unknown_null_raises():
    X = np.random.default_rng(0).standard_normal((300, 3))
    with pytest.raises(ValueError, match="unknown null"):
        detect_subsampling(X, null="nonsense")


# ----------------------------------------------------------------------
# Result contents
# ----------------------------------------------------------------------


@pytest.mark.parametrize("d", [2, 3, 5])
def test_pair_arithmetic_and_reported_settings(d):
    """dof and n_pairs are both d(d-1)/2, and the settings are echoed back."""
    X = np.random.default_rng(5).standard_normal((400, d))
    res = detect_subsampling(X, lag=1, alpha=0.01)
    expected_pairs = d * (d - 1) // 2
    assert res.contemp_dof == expected_pairs
    assert res.n_pairs == expected_pairs
    assert 0 <= res.n_sig_pairs <= res.n_pairs
    assert res.lag == 1
    assert res.alpha == 0.01
    assert res.notes and all(isinstance(n, str) for n in res.notes)


def test_flag_follows_alpha():
    """`subsampling_flagged` is exactly `contemp_pvalue < alpha`.

    alpha does not change the statistic, only the threshold applied to it, so
    the alphas are chosen to bracket the observed p rather than hard-coded --
    a subsampled series can reach p ~ 1e-85, which no fixed "strict" value
    would sit below.
    """
    X = _subsample(_generate_fast(d=5, T=8000, seed=2), 2)
    p = detect_subsampling(X).contemp_pvalue
    assert p > 0.0, "need a non-zero p to bracket"

    strict = detect_subsampling(X, alpha=p / 2)
    loose = detect_subsampling(X, alpha=min(1.0, p * 2))
    assert strict.contemp_pvalue == loose.contemp_pvalue == p  # same test, same p
    assert not strict.subsampling_flagged
    assert loose.subsampling_flagged


def test_nongaussianity_diagnostics_are_reported():
    """Heavy-tailed innovations should register on the non-Gaussianity diagnostic;
    Gaussian ones should largely not. These are reported, not used to flag."""
    rng = np.random.default_rng(3)
    gauss = detect_subsampling(rng.standard_normal((4000, 4)))
    heavy = detect_subsampling(rng.standard_t(df=3, size=(4000, 4)))
    for res in (gauss, heavy):
        assert 0.0 <= res.nongaussian_fraction <= 1.0
        assert 0.0 <= res.nongaussian_min_p <= 1.0
    assert heavy.nongaussian_fraction >= gauss.nongaussian_fraction


def test_lag_is_honoured():
    """A higher VAR order is fitted and echoed; it changes the residuals, so the
    statistic is not simply reused from lag=1."""
    X = _subsample(_generate_fast(d=4, T=6000, seed=4), 2)
    one, two = detect_subsampling(X, lag=1), detect_subsampling(X, lag=2)
    assert one.lag == 1 and two.lag == 2
    assert one.contemp_stat != two.contemp_stat


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------


def test_series_too_short_raises():
    X = np.random.default_rng(0).standard_normal((5, 3))
    with pytest.raises(ValueError, match="too short"):
        detect_subsampling(X)


def test_univariate_input_raises_clearly():
    """The test is about correlation *between* series, so d<2 has nothing to
    measure. It must say so rather than fail deep inside numpy."""
    X = np.random.default_rng(0).standard_normal((300, 1))
    with pytest.raises(ValueError, match="at least 2 variables"):
        detect_subsampling(X)


def test_non_2d_input_raises_clearly():
    X = np.random.default_rng(0).standard_normal(300)
    with pytest.raises(ValueError, match="2-D"):
        detect_subsampling(X)


if __name__ == "__main__":
    pytest.main([__file__])
