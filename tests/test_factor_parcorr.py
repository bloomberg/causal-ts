# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for FactorParCorr and LUCID's ``pervasive_lagged`` option."""

import numpy as np
import pandas as pd
import pytest

from causalts.ci_tests.parcorr_gpu import ParCorrGPU
from causalts.confounders import run_lucid
from causalts.confounders._factor_parcorr import FactorParCorr
from causalts.utils.helpers import evaluate_graph
from tests._factor_panel import factor_panel


def _block_panel(T=1000, seed=0, edge=0.4, n_groups=3, gsz=4, phi=0.0):
    """Block loadings (gsz series per factor); optional edge X0(t-1) -> X1(t)."""
    rng = np.random.default_rng(seed)
    d = n_groups * gsz
    f = np.zeros((T, n_groups))
    for t in range(1, T):
        f[t] = phi * f[t - 1] + rng.standard_normal(n_groups)
    lam = rng.uniform(0.5, 1.5, d)
    X = np.zeros((T, d))
    for t in range(1, T):
        X[t] = lam * f[t, np.arange(d) // gsz] + 0.3 * X[t - 1]
        X[t] += 0.5 * rng.standard_normal(d)
        X[t, 1] += edge * X[t - 1, 0]
    return X


def _embed(X, lags=(0, 1)):
    df = pd.DataFrame(X)
    data = pd.concat([df.shift(lag) for lag in lags], axis=1).iloc[max(lags) :]
    return data.to_numpy()


def _test(X, k=3, lags=(0, 1)):
    t = FactorParCorr(k)
    t.data = _embed(X, lags)
    t.set_lag_structure(n_vars=X.shape[1], lags=list(lags))
    return t


def test_true_edge_survives_groupmate_conditioning():
    # Residuals after removing PCA factors are exactly collinear within a loading
    # block, so conditioning on the cause's groupmates erases a true edge. The
    # leave-out factors keep it.
    X = _block_panel(seed=0)
    d = X.shape[1]
    lag1 = lambda v: d + v  # noqa: E731
    Z = [lag1(1), lag1(2), lag1(3)]
    p, _ = _test(X)(lag1(0), 1, Z)
    assert p < 1e-4

    R = X - X.mean(0)
    _, _, Vt = np.linalg.svd(R / R.std(0), full_matrices=False)
    R = R / R.std(0)
    R = R - R @ Vt[:3].T @ Vt[:3]
    resid = ParCorrGPU(np.zeros((2, 2)))
    resid.data = _embed(R)
    p_resid, _ = resid(lag1(0), 1, Z)
    assert p_resid > 1e-4


def _null_rejections(gsz, phi, n=60, alpha=0.05):
    """One lag-1 null test per independent dataset (tests on one dataset correlate)."""
    fac, plain = [], []
    for seed in range(n):
        X = _block_panel(seed=100 + seed, edge=0.0, phi=phi, gsz=gsz)
        d = X.shape[1]
        i, j = np.random.default_rng(seed).choice(d, 2, replace=False)
        t = _test(X)
        p = ParCorrGPU(np.zeros((2, 2)))
        p.data = t.data
        fac.append(t(d + i, j, [d + j])[0] < alpha)
        plain.append(p(d + i, j, [d + j])[0] < alpha)
    return np.mean(fac), np.mean(plain)


def test_null_calibration():
    # White factors: nothing confounds a lagged pair; the test stays at alpha.
    fac, _ = _null_rejections(gsz=4, phi=0.0)
    assert fac < 0.12
    # Persistent factors confound lagged pairs. Plain partial correlation rejects
    # often; the factor-adjusted test is near alpha once each factor has enough
    # series to be estimated well (residual error shrinks with group size).
    fac, plain = _null_rejections(gsz=16, phi=0.8)
    assert plain > 0.2
    assert fac < 0.15


def test_lagged_test_uses_only_past_factor_scores():
    # Current factor scores are sums of current series (the cause's children), so a
    # lagged test conditions only on scores at the cause's time: the p-value equals
    # an OLS t-test with [1, Z, F(t-1)] and differs from one that adds F(t).
    import statsmodels.api as sm

    X = _block_panel(seed=1, edge=0.0)  # p-values away from 0 so they compare
    d = X.shape[1]
    t = _test(X)
    p, _ = t(d + 0, 1, [d + 1])
    F0, F1 = t._factor_scores({0, 1}, 0)
    D = t.data

    def ols_p(*extra):
        C = sm.add_constant(np.column_stack([D[:, d + 0], D[:, d + 1], *extra]))
        return sm.OLS(D[:, 1], C).fit().pvalues[1]

    assert p == pytest.approx(ols_p(F1), rel=1e-6, abs=0)
    assert p != pytest.approx(ols_p(F1, F0), rel=1e-3, abs=0)


def test_reassigning_data_clears_cache():
    t = _test(_block_panel(seed=2))
    d = 12
    first = t(d + 0, 1, [])
    t.data = _embed(_block_panel(seed=3))
    assert t(d + 0, 1, []) != first


def test_rejects_missing_values_and_column_sets():
    X = _block_panel(seed=0)
    t = _test(X)
    with pytest.raises(TypeError):
        t([0, 1], 2, [])
    data = t.data.copy()
    data[3, 0] = np.nan
    t.data = data
    with pytest.raises(ValueError, match="missing"):
        t(12, 1, [])


def test_time_index_column_is_not_a_factor_input():
    # n_obs excludes trailing per-block columns (CDNOTS's C node) from the factors.
    X = _block_panel(seed=0)
    T, d = X.shape
    Xc = np.column_stack([X, np.arange(T, dtype=float)])
    a = FactorParCorr(3, n_obs=d)
    a.data = _embed(Xc)
    a.set_lag_structure(n_vars=d + 1, lags=[0, 1])
    b = _test(X)
    # Same factors, so the same p-value (rows match: both drop the first row).
    assert a(d + 1 + 0, 1, [])[0] == pytest.approx(b(d + 0, 1, [])[0])


def test_factor_adjusted_lucid_recovers_lagged_edges():
    g = factor_panel(d=20, n_factors=3, T=1000, residual_lag0_density=0.05, seed=0)
    X, G = g["X"], g["residual_graph"]

    def lagpos(res):
        est = np.asarray(res.cg_tig)[:20, :20]
        return evaluate_graph(est, G, exclude_self_loops=True)["F1_lagpos"]

    base = run_lucid(X, 1)
    adj = run_lucid(X, 1, pervasive_lagged="factor_adjusted")
    assert base.regime != "sparse"
    assert adj.info["factor_adjusted"]["n_factors"] == 3
    assert lagpos(adj) > lagpos(base) + 0.2
    # The lag-0 slice still comes from S-L recovery, unchanged.
    np.testing.assert_array_equal(
        np.asarray(adj.cg_tig)[:, :, 0], np.asarray(base.cg_tig)[:, :, 0]
    )


def test_factor_adjusted_keeps_lag0_unchanged_at_higher_max_lag():
    # S-L picks its VAR order from the highest occupied lag, so the adjusted lags
    # must not be visible to lag-0 recovery.
    g = factor_panel(d=12, n_factors=2, T=600, residual_lag0_density=0.1, seed=4)
    base = run_lucid(g["X"], 3)
    adj = run_lucid(g["X"], 3, pervasive_lagged="factor_adjusted")
    assert adj.info["pervasive_lagged"] == "factor_adjusted"
    np.testing.assert_array_equal(
        np.asarray(adj.cg_tig)[:, :, 0], np.asarray(base.cg_tig)[:, :, 0]
    )
    assert not np.array_equal(
        np.asarray(adj.cg_tig)[:, :, 1:], np.asarray(base.cg_tig)[:, :, 1:]
    )


def test_factor_adjusted_is_a_no_op_on_the_sparse_route():
    rng = np.random.default_rng(0)
    T, d = 500, 6
    X = np.zeros((T, d))
    for t in range(1, T):
        X[t] = 0.3 * X[t - 1] + rng.standard_normal(d)
        X[t, 1:] += 0.5 * X[t - 1, :-1]
    df = pd.DataFrame(X, columns=[f"X{i}" for i in range(d)])
    base = run_lucid(df, 1)
    assert base.regime == "sparse"
    adj = run_lucid(df, 1, pervasive_lagged="factor_adjusted")
    np.testing.assert_array_equal(np.asarray(adj.cg_tig), np.asarray(base.cg_tig))


@pytest.mark.parametrize(
    "kw, match",
    [
        (dict(pervasive_lagged="residual"), "unknown pervasive_lagged"),
        (dict(pervasive_filters=True), "pervasive_filters"),
        (dict(profile="unconditional"), "adaptive"),
        (dict(discovery=lambda df, ml: np.zeros((6, 6, 2))), "base discovery"),
    ],
)
def test_pervasive_lagged_rejects_incompatible_options(kw, match):
    df = pd.DataFrame(np.random.default_rng(0).standard_normal((200, 6)))
    kw = {"pervasive_lagged": "factor_adjusted", **kw}
    with pytest.raises(ValueError, match=match):
        run_lucid(df, 1, **kw)


def test_warns_that_ci_is_not_used_for_factor_adjusted_lags():
    df = factor_panel(d=12, n_factors=2, T=300, seed=0)["X"]
    with pytest.warns(UserWarning, match="own factor-adjusted"):
        run_lucid(df, 1, ci=ParCorrGPU(df.values), pervasive_lagged="factor_adjusted")


def test_uses_all_requested_factors_when_few_series_remain():
    # d=3: testing two series leaves one donor, and k=1 must still be used.
    X = _block_panel(n_groups=1, gsz=3, seed=0)
    t = _test(X, k=1)
    F0, F1 = t._factor_scores({0, 1}, 0)
    assert F0.shape[1] == 1 and F1.shape[1] == 1


def test_saturated_design_raises_instead_of_independence():
    X = _block_panel(T=8, n_groups=2, gsz=4, seed=0)
    t = _test(X, k=3)
    with pytest.raises(ValueError, match="cannot support"):
        t(8, 1, list(range(9, 16)))


def test_factor_adjusted_handles_integer_columns():
    # CDNOTS would wrap integer-valued columns in StratifiedCIT, hiding the lag
    # layout; the factor-adjusted path opts out and treats them as continuous.
    g = factor_panel(d=20, n_factors=3, T=600, seed=1)
    df = g["X"].copy()
    df["X0"] = (df["X0"] > 0).astype(int)
    res = run_lucid(df, 1, pervasive_lagged="factor_adjusted")
    assert res.regime == "pervasive"
    assert res.info["pervasive_lagged"] == "factor_adjusted"


def test_factor_adjusted_leaves_the_sf_route_unchanged():
    g = factor_panel(d=20, n_factors=3, T=1000, factor_strength=0.5, seed=0)
    base = run_lucid(g["X"], 1)
    assert base.regime == "sf"
    adj = run_lucid(g["X"], 1, pervasive_lagged="factor_adjusted")
    assert adj.info["pervasive_lagged"] == "observed"
    np.testing.assert_array_equal(np.asarray(adj.cg_tig), np.asarray(base.cg_tig))


def test_sparse_route_reports_observed_and_does_not_warn():
    import warnings

    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 6))
    for t in range(1, 400):
        X[t, 1:] += 0.5 * X[t - 1, :-1]
    df = pd.DataFrame(X, columns=[f"X{i}" for i in range(6)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = run_lucid(
            df, 1, ci=ParCorrGPU(df.values), pervasive_lagged="factor_adjusted"
        )
    assert not [w for w in caught if "factor-adjusted" in str(w.message)]
    assert res.regime == "sparse"
    assert res.info["pervasive_lagged"] == "observed"


def test_factor_adjusted_rejects_tetrad_base():
    # The factor-adjusted lagged search runs plain CDNOTS; with the CDNOTS+ base it
    # would silently switch engines, so the combination is rejected.
    df = pd.DataFrame(np.random.default_rng(0).standard_normal((200, 6)))
    with pytest.raises(ValueError, match="tetrad"):
        run_lucid(df, 1, pervasive_lagged="factor_adjusted", pervasive_base="tetrad")


def test_per_pair_caches_stay_bounded(monkeypatch):
    # Loadings and scores have separate byte budgets, so a wide or long panel
    # cannot accumulate per-pair state and large scores never crowd out loadings;
    # entries past a budget are recomputed, which must not change any p-value.
    from causalts.confounders import _factor_parcorr as fp

    X = _block_panel(seed=4, gsz=8)
    d = X.shape[1]
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    t = _test(X)
    full = [t(d + i, j, [])[0] for i, j in pairs]

    def retained(cache):
        return sum(
            fp._ENTRY_OVERHEAD + sum(a.nbytes for a in (v or []) if a is not None)
            for v, _ in cache.values()
        )

    for name, attr in (
        ("_LOADINGS_BYTES", "_loadings_cache"),
        ("_SCORES_BYTES", "_scores_cache"),
    ):
        budget = getattr(t, attr).nbytes // 20
        with monkeypatch.context() as m:
            m.setattr(fp, name, budget)
            tight = _test(X)
            assert [tight(d + i, j, [])[0] for i, j in pairs] == full
        cache = getattr(tight, attr)
        assert len(cache) < len(pairs)
        assert retained(cache) == cache.nbytes <= budget


def test_full_score_cache_keeps_loadings(monkeypatch):
    # A score budget too small for even one pair must not force an SVD per test.
    from causalts.confounders import _factor_parcorr as fp

    monkeypatch.setattr(fp, "_SCORES_BYTES", 1)
    X = _block_panel(seed=4, gsz=8)
    d = X.shape[1]
    t = _test(X)
    calls = []
    svd = np.linalg.svd
    monkeypatch.setattr(
        fp.np.linalg, "svd", lambda *a, **k: calls.append(1) or svd(*a, **k)
    )
    for _ in range(3):
        t.pvalue_cache = {}
        t(d + 0, 1, [])
    assert len(calls) == 1
    assert len(t._scores_cache) == 0


def test_overflowing_loadings_cache_refits_only_the_overflow(monkeypatch):
    # Pairs are revisited cyclically; a cache just smaller than the working set
    # must still serve the pairs it holds (an LRU would serve none).
    from causalts.confounders import _factor_parcorr as fp

    X = _block_panel(seed=4, gsz=8)
    d = X.shape[1]
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    probe = _test(X)
    for i, j in pairs:
        probe._loadings({i, j})
    monkeypatch.setattr(fp, "_LOADINGS_BYTES", probe._loadings_cache.nbytes - 1)
    monkeypatch.setattr(fp, "_SCORES_BYTES", 0)

    t = _test(X)
    calls = []
    svd = np.linalg.svd
    monkeypatch.setattr(
        fp.np.linalg, "svd", lambda *a, **k: calls.append(1) or svd(*a, **k)
    )
    for i, j in pairs:
        t(d + i, j, [])
    first = len(calls)
    t.pvalue_cache = {}
    for i, j in pairs:
        t(d + i, j, [])
    assert first == len(pairs)
    assert len(calls) - first == len(pairs) - len(t._loadings_cache) == 1
