# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the variance-only-parent check behind the inspect recommendation."""

import numpy as np
import pytest

from causalts.utils.heteroskedasticity import check_variance_dependence


def _chain(T, seed, scale_parent=False, mean="linear", noise="gauss"):
    rng = np.random.default_rng(seed)
    d = 5
    x = np.zeros((T + 100, d))
    for t in range(1, T + 100):
        prev = x[t - 1]
        if mean == "linear":
            m = 0.4 * prev + np.r_[0, 0.35 * prev[:-1]]
        else:
            m = 0.4 * prev + np.r_[0, 0.9 * np.sin(2 * prev[:-1])]
        eps = rng.standard_t(3, d) / np.sqrt(3) if noise == "t3" else rng.normal(size=d)
        scale = np.ones(d)
        if scale_parent:
            scale[4] = np.exp(0.6 * prev[0])  # X0 moves only the noise scale of X4
        x[t] = m + scale * eps
    return x[100:]


def test_detects_variance_only_parent():
    res = check_variance_dependence(_chain(600, 0, scale_parent=True), max_lag=2)
    assert res["detected"] is True
    flagged = {(r["x"], r["y"]) for r in res["results"] if r["is_variance_parent"]}
    assert (0, 4) in flagged


@pytest.mark.parametrize(
    "kw", [{}, {"mean": "nonlinear"}, {"noise": "t3"}], ids=["linear", "nlmean", "t3"]
)
def test_null_panels_rarely_flagged(kw):
    """No variance parents: at most 2 of 10 panels flagged (measured rate 0-10%)."""
    flagged = sum(
        bool(check_variance_dependence(_chain(500, s, **kw), max_lag=2)["detected"])
        for s in range(10)
    )
    assert flagged <= 2


def test_nonfinite_or_tiny_input_reports_unavailable():
    x = _chain(300, 1)
    x[5, 2] = np.nan
    assert check_variance_dependence(x)["detected"] is None
    assert (
        check_variance_dependence(np.random.default_rng(0).normal(size=(12, 6)))[
            "detected"
        ]
        is None
    )
    assert check_variance_dependence(np.ones((100, 1)))["detected"] is None


def test_min_effect_floor_suppresses_flags():
    x = _chain(600, 0, scale_parent=True)
    assert check_variance_dependence(x, max_lag=2, min_effect=0.99)["detected"] is False


def test_invariant_to_permutation_scale_offset_and_sign():
    x = _chain(400, 1, scale_parent=True)

    def pvals(a, order=range(5)):
        res = check_variance_dependence(a, max_lag=2)["results"]
        return {(r["x"], r["y"]): r["pvalue"] for r in res}

    base = pvals(x)
    perm = np.array([3, 0, 4, 1, 2])
    inv = np.argsort(perm)
    permuted = pvals(x[:, perm])
    transformed = pvals(
        x * np.array([-1, 2.0, -50, 1e-3, 1]) + np.array([5, 0, 1e3, -2, 0])
    )
    for (i, j), p in base.items():
        assert permuted[(inv[i], inv[j])] == pytest.approx(p, rel=1e-6)
        assert transformed[(i, j)] == pytest.approx(p, rel=1e-6)


@pytest.mark.parametrize("scale", ["band", "lag3"], ids=["non-monotone", "lag-3"])
def test_detects_non_monotone_and_longer_lag_scale_parents(scale):
    rng = np.random.default_rng(0)
    T, d = 600, 4
    x = np.zeros((T, d))
    for t in range(3, T):
        s = np.ones(d)
        if scale == "band":
            s[3] = 0.3 + 1.5 * float(0.5 < abs(x[t - 1, 0]) < 1.2)
        else:
            s[3] = np.exp(0.6 * x[t - 3, 0])
        x[t] = 0.4 * x[t - 1] + s * rng.normal(size=d)
    res = check_variance_dependence(x, max_lag=3)
    flagged = {(r["x"], r["y"]) for r in res["results"] if r["is_variance_parent"]}
    assert (0, 3) in flagged


def _shared_scale_panel(T, seed, kind):
    """Independent AR(1) series whose noise scale moves together: no variance parents."""
    rng = np.random.default_rng(seed)
    d = 5
    if kind == "break":
        scale = np.where(np.arange(T) < T // 2, 0.3, 2.0)
    else:  # common stochastic volatility
        v = np.zeros(T)
        for t in range(1, T):
            v[t] = 0.95 * v[t - 1] + 0.3 * rng.normal()
        scale = np.exp(v)
    x = np.zeros((T, d))
    for t in range(1, T):
        x[t] = 0.4 * x[t - 1] + scale[t] * rng.normal(size=d)
    return x


@pytest.mark.parametrize("kind", ["break", "stochastic"])
def test_shared_volatility_is_not_a_variance_parent(kind):
    """A volatility regime common to all series is not a directed variance effect."""
    flagged = sum(
        bool(check_variance_dependence(_shared_scale_panel(1000, s, kind))["detected"])
        for s in range(10)
    )
    assert flagged <= 1


def test_additive_nonlinear_chain_is_not_flagged():
    """A strongly nonlinear but additive mean (2 sin(3x) on every link)."""

    def panel(seed):
        r = np.random.default_rng(seed)
        x = np.zeros((600, 5))
        for t in range(1, 600):
            p = x[t - 1]
            x[t] = 0.4 * p + np.r_[0, 2 * np.sin(3 * p[:-1])] + r.normal(size=5)
        return x[100:]

    assert (
        sum(bool(check_variance_dependence(panel(s))["detected"]) for s in range(10))
        <= 1
    )


def test_rejects_max_lag_zero():
    with pytest.raises(ValueError, match="max_lag must be >= 1"):
        check_variance_dependence(_chain(300, 0), max_lag=0)


def test_persistent_driver_in_linear_var_is_not_flagged():
    """The api_reference signalling panel (linear Gaussian VAR, an AR(0.9) root).

    A smooth time trend in the scale model once fit a spurious Plcg -> P38 effect
    here (X0 and X5 are in disconnected components).
    """
    np.random.seed(42)
    T, L = 300, 3
    x = np.random.randn(T + L, 11)
    for t in range(L, T + L):
        x[t, 0] += 0.9 * x[t - 1, 0]
        x[t, 1] += 0.6 * x[t - 1, 1] + 0.5 * x[t - 2, 0]
        x[t, 2] += 0.4 * x[t - 1, 2] + 0.4 * x[t - 3, 0] + 0.3 * x[t - 1, 1]
        x[t, 3] += 0.25 * x[t - 1, 3]
        x[t, 4] += 0.25 * x[t - 1, 4] - 0.25 * x[t - 1, 3]
        x[t, 5] += 0.25 * x[t - 1, 5] + 0.50 * x[t - 2, 3] - 0.50 * x[t - 1, 4]
        x[t, 6] += 0.30 * x[t - 1, 6] - 0.25 * x[t - 3, 3] + 0.25 * x[t - 3, 4]
        x[t, 7] += 0.40 * x[t - 1, 7] + 0.35 * x[t - 1, 3] + 0.25 * x[t - 2, 4]
        x[t, 8] += (
            0.50 * x[t - 1, 8]
            - 0.5 * x[t - 1, 3]
            + 0.25 * x[t - 1, 4]
            + 0.6 * x[t - 2, 7]
        )
        x[t, 9] += 0.6 * x[t - 1, 9] + 0.45 * x[t - 1, 4] - 0.35 * x[t - 2, 8]
        x[t, 10] += 0.7 * x[t - 1, 10] - 0.25 * x[t - 1, 4] + 0.25 * x[t - 2, 9]
    res = check_variance_dependence(x[L:], max_lag=3, test_lags=3)
    assert res["detected"] is False


def test_degenerate_targets_are_not_tested():
    """Constant or duplicated columns leave only rounding noise to test."""
    rng = np.random.default_rng(158)
    a = rng.normal(size=100)
    assert check_variance_dependence(np.c_[a, a])["detected"] is not True
    x = np.c_[rng.normal(size=(200, 3)), np.ones(200)]
    res = check_variance_dependence(x, max_lag=2)
    assert all(r["y"] != 3 for r in res["results"])
    dup = rng.normal(size=(300, 3))
    assert (
        check_variance_dependence(np.c_[dup, dup[:, 0]], max_lag=2)["detected"] is False
    )


def test_periodic_and_near_duplicate_columns_are_not_flagged():
    t = np.arange(400)
    periodic = np.c_[t % 4, (t + 1) % 4, (t // 2) % 3, t % 5].astype(float)
    res = check_variance_dependence(periodic, max_lag=3)
    assert res["detected"] is None and res["skipped_targets"] == [0, 1, 2, 3]

    rng = np.random.default_rng(0)
    a = np.zeros(400)
    for k in range(1, 400):
        a[k] = 0.6 * a[k - 1] + rng.normal()
    near = np.c_[a, a + 1e-5 * rng.normal(size=400)]
    assert check_variance_dependence(near, max_lag=1)["detected"] is not True


def test_skipped_targets_reported():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 3))
    res = check_variance_dependence(np.c_[x, np.ones(300)], max_lag=1)
    assert res["skipped_targets"] == [3]
    assert check_variance_dependence(x, max_lag=1)["skipped_targets"] == []


def _self_heteroskedastic(T, rng):
    """AR(1) whose noise scale follows its own lagged level."""
    a = np.zeros(T)
    for t in range(1, T):
        a[t] = 0.5 * a[t - 1] + np.exp(0.5 * np.clip(a[t - 1], -3, 3)) * rng.normal()
    return a


def test_duplicate_of_self_heteroskedastic_series_is_not_a_parent():
    """A duplicate of Y carries Y's own level, not an effect of another variable."""
    for s in range(5):
        rng = np.random.default_rng(s)
        a = _self_heteroskedastic(600, rng)
        other = _chain(600, s)[:, 0]
        for b in (a, a + 1e-5 * rng.normal(size=600)):
            res = check_variance_dependence(np.c_[a, b, other], max_lag=1)
            assert not any(
                r["is_variance_parent"] and {r["x"], r["y"]} == {0, 1}
                for r in res["results"]
            )


def test_driver_of_duplicated_target_is_still_found():
    rng = np.random.default_rng(0)
    drv = rng.normal(size=600)
    lag = np.r_[0, drv[:-1]]
    tgt = 0.3 * lag + np.exp(0.6 * lag) * rng.normal(size=600)
    res = check_variance_dependence(
        np.c_[drv, tgt, tgt, rng.normal(size=600)], max_lag=1
    )
    flagged = {(r["x"], r["y"]) for r in res["results"] if r["is_variance_parent"]}
    assert flagged == {(0, 1), (0, 2)}


def test_target_constant_over_fitted_rows_is_skipped():
    """Only the dropped first row differs; std of the rest is ~1e-17, not 0."""
    y = np.zeros(300)
    y[0] = 1.0
    x = np.c_[np.random.default_rng(0).normal(size=(300, 3)), y]
    res = check_variance_dependence(x)
    assert res["skipped_targets"] == [3] and res["detected"] is False
