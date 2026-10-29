# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for GRACE's AR path, self-loop test, second-moment grow and grow nuisance basis."""

import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from causalts.grace.gated_discovery import (
    GatedCausalDiscovery,
    run_cdnots_gated,
    run_stability_selection,
)
from causalts.grace.grow import (
    grow_candidates,
    grow_variance_candidates,
    self_loop_test,
)

FAST = dict(verbose=False, model_seed=0, device="cpu", max_epochs=40)


def _proxy_pair(rng, n, self_loop, phi=0.9):
    """z persistent (AR ~.9), y driven nonlinearly by z(t-1); y's own lag is a proxy."""
    T = n + 200
    z, y = np.zeros(T), np.zeros(T)
    for t in range(1, T):
        z[t] = phi * z[t - 1] + 0.3 * np.tanh(z[t - 1]) + rng.standard_normal()
        y[t] = (
            (0.5 * y[t - 1] if self_loop else 0.0)
            + np.tanh(z[t - 1])
            + 0.5 * np.sin(z[t - 1])
            + rng.standard_normal()
        )
    return np.column_stack([z, y])[200:]


def _ar_only(d=3, T=300, phi=0.8, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((T, d))
    for t in range(1, T):
        X[t] = phi * X[t - 1] + rng.standard_normal(d)
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(d)])


def _cross(d, L, edges):
    G = np.zeros((d, d, L + 1), dtype=bool)
    for c, e, lag in edges:
        G[c, e, lag] = True
    return G


# -- self_loop_test --------------------------------------------------------------


def test_self_loop_test_keeps_own_lag_and_conditions_on_parents():
    X = _proxy_pair(np.random.default_rng(0), 1000, self_loop=True)
    kept = self_loop_test(X, _cross(2, 2, [(0, 1, 1)]), 2)
    assert kept[1, 1, 1] and kept[0, 0, 1]
    assert not kept[:, :, 0].any()
    off = ~np.eye(2, dtype=bool)
    assert not kept[off].any()  # diagonal only


def test_self_loop_test_spline_nuisance_holds_level():
    # Null: y has no self-loop, its parent z is persistent. With the fixed RFF
    # nuisance the misfit of z's effect makes y's own lag look significant.
    cross = _cross(2, 2, [(0, 1, 1)])
    rates = {}
    for nb in ("spline", None):
        rng = np.random.default_rng(7)
        rates[nb] = np.mean(
            [
                self_loop_test(
                    _proxy_pair(rng, 1000, False),
                    cross,
                    2,
                    alpha=0.01,
                    nuisance_basis=nb,
                )[1, 1, 1]
                for _ in range(60)
            ]
        )
    assert rates["spline"] <= 0.05
    assert rates[None] >= 0.1


def test_self_loop_test_pvalues_and_invariances():
    X = _proxy_pair(np.random.default_rng(1), 600, self_loop=True)
    X = np.column_stack([X, np.random.default_rng(2).standard_normal(len(X))])
    cross = _cross(3, 2, [(0, 1, 1)])
    kept, p = self_loop_test(X, cross, 2, return_pvalues=True)
    assert p.shape == kept.shape == (3, 3, 3)
    assert np.isnan(p[:, :, 0]).all() and np.isnan(p[0, 1]).all()
    np.testing.assert_array_equal(kept, np.nan_to_num(p, nan=1.0) < 1e-3)
    # rank transform: invariant to per-column monotone rescaling
    np.testing.assert_array_equal(
        self_loop_test(X * [3.0, 0.01, 7.0] + 5, cross, 2), kept
    )
    perm = [2, 0, 1]
    np.testing.assert_array_equal(
        self_loop_test(X[:, perm], cross[perm][:, perm], 2), kept[perm][:, perm]
    )


def test_self_loop_test_conditions_on_exog():
    # White noise around a trend: the own lag predicts the trend unless the
    # trend is conditioned on.
    rng = np.random.default_rng(3)
    T = 500
    trend = np.linspace(0, 4, T)
    X = rng.standard_normal((T, 2)) + trend[:, None]
    none = np.zeros((2, 2, 3), dtype=bool)
    assert self_loop_test(X, none, 2).any()
    assert not self_loop_test(X, none, 2, exog=trend[:, None]).any()


def _weak_ar(rng, T, kind):
    """Target x with a weak own-lag effect of ``kind`` and a persistent parent z."""
    n = T + 100
    z, x = np.zeros(n), np.zeros(n)
    e, u = rng.standard_normal(n), rng.standard_normal(n)
    own = {
        "linear": lambda v: 0.2 * v,
        "sine": lambda v: 0.35 * np.sin(2 * v),
        "null": lambda v: 0.0,
    }[kind]
    for t in range(1, n):
        z[t] = 0.9 * z[t - 1] + u[t]
        x[t] = own(x[t - 1]) + 0.6 * np.tanh(z[t - 1]) + 0.7 * e[t]
    return np.column_stack([x, z])[100:]


def test_self_loop_test_union_power_and_level():
    # Small T, weak own-lag effects: the union (1-df linear + 4-knot spline) keeps
    # more true self-loops than the 9-column RFF block, linear and nonlinear alike,
    # and holds the level when the target has no self-loop.
    cross = _cross(2, 3, [(1, 0, 1)])
    rate = {}
    for kind in ("linear", "sine", "null"):
        rng = np.random.default_rng(11)
        Xs = [_weak_ar(rng, 250, kind) for _ in range(60)]
        for basis in ("union", "rff"):
            rate[kind, basis] = np.mean(
                [self_loop_test(X, cross, 3, test_basis=basis)[0, 0, 1] for X in Xs]
            )
    assert rate["linear", "union"] > rate["linear", "rff"] + 0.1
    assert rate["sine", "union"] >= rate["sine", "rff"]
    assert rate["null", "union"] <= 0.05


def test_self_loop_test_union_pvalue_is_bonferroni_min():
    X = _proxy_pair(np.random.default_rng(4), 400, self_loop=True)
    cross = _cross(2, 2, [(0, 1, 1)])
    _, p = self_loop_test(X, cross, 2, return_pvalues=True)
    assert np.nanmax(p) <= 1.0 and np.nanmin(p) >= 0.0
    _, p_rff = self_loop_test(X, cross, 2, return_pvalues=True, test_basis="rff")
    assert np.array_equal(np.isnan(p), np.isnan(p_rff))
    # One lag, no parents, no exog: the two nested F-tests on the ranked,
    # standardized design can be written out, and the union is min(1, 2 * min).
    from scipy.stats import f as f_dist
    from scipy.stats import norm, rankdata

    from causalts.grace.grow import spline_basis

    rng = np.random.default_rng(6)
    T = 300
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = 0.15 * x[t - 1] + 0.3 * np.sin(2 * x[t - 1]) + rng.standard_normal()
    Xs = np.column_stack([x, rng.standard_normal(T)])
    _, pu = self_loop_test(Xs, np.zeros((2, 2, 2), bool), 1, return_pvalues=True)
    z = norm.ppf((rankdata(x) - 0.5) / T)
    lagged = np.stack([z[1:], z[:-1]], axis=1)
    y, w = ((lagged - lagged.mean(0)) / lagged.std(0)).T
    ones = np.ones((len(y), 1))

    def ftest(block):
        full = np.column_stack([ones, block])
        rf = y - full @ np.linalg.lstsq(full, y, rcond=None)[0]
        rr = y - y.mean()
        df1 = np.linalg.matrix_rank(full) - 1
        df2 = len(y) - np.linalg.matrix_rank(full)
        return f_dist.sf(((rr @ rr - rf @ rf) / df1) / ((rf @ rf) / df2), df1, df2)

    S = spline_basis(w, n_knots=4)
    assert S.shape[1] > 1  # the spline adds directions: two distinct tests
    want = min(1.0, 2 * min(ftest(w[:, None]), ftest(S)))
    np.testing.assert_allclose(pu[0, 0, 1], want, rtol=1e-6)


def test_self_loop_test_union_discrete_target_is_one_test():
    # A binary own lag: indicators span the linear term, so the union is a single
    # test at alpha (no Bonferroni halving) and matches the RFF block's p-value,
    # which collapses to the same direction.
    rng = np.random.default_rng(5)
    T = 300
    b = np.zeros(T)
    for t in range(1, T):
        b[t] = b[t - 1] if rng.random() < 0.7 else 1 - b[t - 1]
    X = np.column_stack([b, rng.standard_normal(T)])
    none = np.zeros((2, 2, 3), dtype=bool)
    _, p_u = self_loop_test(X, none, 2, return_pvalues=True)
    _, p_r = self_loop_test(X, none, 2, return_pvalues=True, test_basis="rff")
    np.testing.assert_allclose(p_u[0, 0, 1:], p_r[0, 0, 1:], rtol=1e-6)


def test_self_loop_test_union_collapsed_spline_is_one_test():
    # Zero-inflated counts (> 8 levels): the 4-knot spline's quantile knots tie and
    # it collapses to the linear term, so the union is one test, not two at
    # alpha / 2. With no parents and one lag the p-value is the plain linear F-test.
    from scipy.stats import f as f_dist
    from scipy.stats import norm, rankdata

    rng = np.random.default_rng(3)
    T = 400
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = np.floor(0.5 * x[t - 1]) + (
            rng.integers(4, 16) if rng.random() < 0.06 else 0
        )
    assert len(np.unique(x)) > 8 and np.mean(x == 0) > 0.7
    X = np.column_stack([x, rng.standard_normal(T)])
    _, p = self_loop_test(X, np.zeros((2, 2, 2), bool), 1, return_pvalues=True)
    z = norm.ppf((rankdata(x) - 0.5) / T)
    y, w = z[1:] - z[1:].mean(), z[:-1] - z[:-1].mean()
    res = y - w * (w @ y) / (w @ w)
    stat = (y @ y - res @ res) / ((res @ res) / (len(y) - 2))
    np.testing.assert_allclose(p[0, 0, 1], f_dist.sf(stat, 1, len(y) - 2), rtol=1e-6)


def test_self_loop_test_union_discrete_merges_rare_levels():
    # A 3-level own lag with one level seen twice: the rare level is merged into its
    # neighbour, so the test is the 1-df F-test of {0} vs {1, 2}, not an indicator
    # that fits those two rows exactly.
    from scipy.stats import f as f_dist
    from scipy.stats import norm, rankdata

    rng = np.random.default_rng(8)
    T = 300
    b = np.zeros(T)
    for t in range(1, T):
        b[t] = b[t - 1] if rng.random() < 0.7 else 1 - b[t - 1]
    b[[100, 200]] = 2.0
    X = np.column_stack([b, rng.standard_normal(T)])
    _, p = self_loop_test(X, np.zeros((2, 2, 2), bool), 1, return_pvalues=True)
    z = norm.ppf((rankdata(b) - 0.5) / T)
    y = (z[1:] - z[1:].mean()) / z[1:].std()
    full = np.column_stack([np.ones(T - 1), b[:-1] == 0])
    rf = y - full @ np.linalg.lstsq(full, y, rcond=None)[0]
    rr = y - y.mean()
    want = f_dist.sf((rr @ rr - rf @ rf) / ((rf @ rf) / (T - 3)), 1, T - 3)
    np.testing.assert_allclose(p[0, 0, 1], want, rtol=1e-6)


def test_run_cdnots_gated_self_loop_basis_passes_through(monkeypatch):
    import causalts.grace.grow as grow_mod

    seen = []
    real = grow_mod.self_loop_test

    def spy(*args, **kwargs):
        seen.append(kwargs.get("test_basis"))
        return real(*args, **kwargs)

    monkeypatch.setattr(grow_mod, "self_loop_test", spy)
    for basis in ("union", "rff"):
        run_cdnots_gated(
            _ar_only(T=200),
            max_lag=2,
            ar_path=True,
            include_C=False,
            self_loop_basis=basis,
            **FAST,
        )
    assert seen == ["union", "rff"]
    with pytest.raises(ValueError, match="self_loop_basis"):
        run_cdnots_gated(_ar_only(T=200), max_lag=2, ar_path=True, self_loop_basis="x")


def test_self_loop_test_validation():
    X = np.random.default_rng(0).standard_normal((100, 2))
    none = np.zeros((2, 2, 3), dtype=bool)
    Xn = X.copy()
    Xn[5, 0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        self_loop_test(Xn, none, 2)
    with pytest.raises(ValueError, match="shape"):
        self_loop_test(X, none[:, :, :2], 2)
    with pytest.raises(ValueError, match="test_basis"):
        self_loop_test(X, none, 2, test_basis="spline")
    with pytest.raises(ValueError, match="max_lag"):
        self_loop_test(X, none[:, :, :1], 0)
    with pytest.raises(ValueError, match="alpha"):
        self_loop_test(X, none, 2, alpha=1.5)
    with pytest.raises(ValueError, match="nuisance_basis"):
        self_loop_test(X, none, 2, nuisance_basis="poly")


# -- second-moment grow and nuisance basis --------------------------------------


def test_grow_variance_finds_variance_only_parent():
    rng = np.random.default_rng(3)
    n = 800
    x, w = rng.standard_normal(n), rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = np.exp(0.8 * x[t - 1]) * rng.standard_normal()
    X = np.column_stack([x, w, y])
    est = np.zeros((3, 3, 2), dtype=bool)
    assert not grow_candidates(X, est, est, 1, 0.1, "rank")[0].any()
    added, n_tests = grow_variance_candidates(X, est, est, 1)
    assert added[0, 2, 1] and int(added.sum()) == 1
    assert n_tests == 9
    # exog enters both regressions without breaking the test
    added_ex, _ = grow_variance_candidates(X, est, est, 1, exog=np.linspace(0, 1, n))
    assert added_ex[0, 2, 1]


@pytest.mark.parametrize("basis", ["rff", "spline"])
def test_grow_nuisance_same_basis_is_the_old_behaviour(basis):
    X = _proxy_pair(np.random.default_rng(4), 400, self_loop=True)
    X = np.column_stack([X, np.random.default_rng(5).standard_normal(len(X))])
    est = _cross(3, 2, [(0, 1, 1), (1, 1, 1)])
    old = grow_candidates(X, est, est, 2, 0.5, "rank", basis=basis)
    same = grow_candidates(
        X, est, est, 2, 0.5, "rank", basis=basis, nuisance_basis=basis
    )
    np.testing.assert_array_equal(old[0], same[0])
    assert old[1] == same[1]
    with pytest.raises(ValueError, match="nuisance_basis"):
        grow_candidates(X, est, est, 2, nuisance_basis="poly")


def test_shrink_default_nuisance_matches_old_basis(small_df):
    # Under refine="shrink" with the spline basis, the spline nuisance is the
    # tested basis, so on continuous data the new default changes nothing.
    kw = dict(
        max_lag=1, refine="shrink", grow_rounds=1, grow_transform="rank", verbose=False
    )
    a = run_cdnots_gated(small_df, **kw)
    b = run_cdnots_gated(small_df, grow_nuisance_basis=None, **kw)
    np.testing.assert_array_equal(np.asarray(a.cg_tig), np.asarray(b.cg_tig))


# -- run_cdnots_gated(ar_path=True) ---------------------------------------------


def test_ar_path_on_autoregressive_system():
    res = run_cdnots_gated(_ar_only(), max_lag=2, ar_path=True, include_C=False, **FAST)
    G = np.asarray(res.cg_tig).astype(bool)
    for i in range(3):
        assert G[i, i, 1]  # every variable's lag-1 self-loop, from the F-test
    assert res.gate_threshold_rule in ("ar_li_floor", "ar_li_fallback")
    assert res.gate_threshold >= 0.5
    gv = np.asarray(res.gate_values)
    assert np.all(gv[np.arange(3), np.arange(3), :] == 0)  # self-loops are not gated
    assert not G[np.arange(3), np.arange(3), 0].any()


def test_ar_path_fixed_threshold_and_variance_grow():
    res = run_cdnots_gated(
        _ar_only(T=250),
        max_lag=1,
        ar_path=True,
        gate_threshold=0.7,
        grow_rounds=1,
        grow_variance=True,
        **FAST,
    )
    assert res.gate_threshold_rule == "ar_fixed" and res.gate_threshold == 0.7
    assert len(res.grow_history) >= 1


def test_ar_path_and_grow_variance_validation(small_df):
    with pytest.raises(ValueError, match="refine='gates'"):
        run_cdnots_gated(
            small_df, max_lag=1, ar_path=True, refine="shrink", verbose=False
        )
    with pytest.raises(ValueError, match="max_lag >= 1"):
        run_cdnots_gated(small_df, max_lag=0, ar_path=True, verbose=False)
    with pytest.raises(ValueError, match="grow_variance"):
        run_cdnots_gated(
            small_df, max_lag=1, grow_variance=True, grow_rounds=0, verbose=False
        )
    with pytest.raises(ValueError, match="grow_variance"):
        run_cdnots_gated(
            small_df,
            max_lag=1,
            grow_variance=True,
            grow_rounds=1,
            refine="shrink",
            verbose=False,
        )
    with pytest.raises(ValueError, match="grow_nuisance_basis"):
        run_cdnots_gated(small_df, max_lag=1, grow_nuisance_basis="poly", verbose=False)


def test_unknown_model_kwargs_fail_before_the_skeleton(small_df, monkeypatch):
    import causalts.cdnots.phase3_utils as p3

    def boom(*a, **k):
        raise AssertionError("skeleton should not run")

    monkeypatch.setattr(p3, "run_cdnots", boom)
    with pytest.raises(TypeError, match="lambda_override"):
        run_cdnots_gated(small_df, max_lag=1, lambda_override=0.1, verbose=False)
    with pytest.raises(TypeError, match="grow_round"):
        run_cdnots_gated(small_df, max_lag=1, grow_round=2, verbose=False)
    with pytest.raises(TypeError, match="num_vars"):
        run_cdnots_gated(small_df, max_lag=1, num_vars=3, verbose=False)
    with pytest.raises(TypeError, match="foo"):
        run_stability_selection(small_df, max_lag=1, foo=1, verbose=False)
    with pytest.raises(ValueError, match="run_cdnots_gated only"):
        run_stability_selection(small_df, max_lag=1, ar_path=True, verbose=False)
    with pytest.raises(ValueError, match="run_cdnots_gated only"):
        run_stability_selection(small_df, max_lag=1, ar_hidden=8, verbose=False)


def test_gated_model_rejects_unknown_kwargs_and_bad_ar_hidden():
    with pytest.raises(TypeError, match="lambda_override"):
        GatedCausalDiscovery(num_vars=3, max_lag=1, lambda_override=0.1)
    with pytest.raises(ValueError, match="ar_hidden"):
        GatedCausalDiscovery(num_vars=3, max_lag=1, ar_path=True, ar_hidden=0)
    with pytest.raises(ValueError, match="max_lag"):
        GatedCausalDiscovery(num_vars=3, max_lag=0, ar_path=True)
    with pytest.raises(ValueError, match="only applies with ar_path"):
        GatedCausalDiscovery(num_vars=3, max_lag=1, ar_hidden=8)
    m = GatedCausalDiscovery(num_vars=3, max_lag=2, ar_path=True)
    assert not m.gate_mask[np.arange(3), :, np.arange(3)].any()


def _chain(T=300, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((T, 3))
    for t in range(1, T):
        X[t, 0] = 0.6 * X[t - 1, 0] + rng.standard_normal()
        X[t, 1] = 0.4 * X[t - 1, 1] + 1.2 * X[t - 1, 0] + rng.standard_normal()
        X[t, 2] = 0.4 * X[t - 1, 2] + 1.2 * X[t - 1, 1] + rng.standard_normal()
    return pd.DataFrame(X, columns=list("abc"))


def test_ar_path_fixed_threshold_keeps_gates_at_the_cutoff():
    kw = dict(max_lag=1, ar_path=True, include_C=False, **FAST)
    off = ~np.eye(3, dtype=bool)[:, :, None]
    gv = np.asarray(run_cdnots_gated(_chain(), gate_threshold=0.5, **kw).gate_values)
    top = float(np.where(off, gv, -1.0).max())
    assert top > 0
    # same seed, same gates; a cutoff equal to a gate keeps that edge ('fixed' rule)
    res = run_cdnots_gated(_chain(), gate_threshold=top, **kw)
    np.testing.assert_array_equal(np.asarray(res.gate_values), gv)
    G = np.asarray(res.cg_tig).astype(bool)
    np.testing.assert_array_equal(G & off, (gv >= top) & off)
    assert (G & off).any()


def test_ar_path_grow_leaves_self_loops_to_the_f_test():
    sk = np.zeros((3, 3, 2), dtype=np.int8)
    sk[0, 1, 1] = 1  # one cross candidate, no self-loops
    res = run_cdnots_gated(
        _chain(),
        max_lag=1,
        skeleton=sk,
        ar_path=True,
        include_C=False,
        grow_rounds=1,
        grow_variance=True,
        **FAST,
    )
    first = res.grow_history[0]
    # 6 off-diagonal lagged pairs, 1 in the skeleton; no diagonal pair is tested
    assert first["n_tests"] == 5
    assert first["n_tests_variance"] <= 5
    assert 0 <= first["n_added_variance"] <= first["n_added"]


def test_ar_hidden_checked_before_the_skeleton(small_df, monkeypatch):
    import causalts.cdnots.phase3_utils as p3

    monkeypatch.setattr(p3, "run_cdnots", lambda *a, **k: 1 / 0)
    with pytest.raises(ValueError, match="only applies with ar_path"):
        run_cdnots_gated(small_df, max_lag=1, ar_path=False, ar_hidden=8, verbose=False)
    with pytest.raises(ValueError, match="positive integer"):
        run_cdnots_gated(small_df, max_lag=1, ar_path=True, ar_hidden=0, verbose=False)
    with pytest.raises(ValueError, match="positive integer"):
        run_cdnots_gated(
            small_df, max_lag=1, ar_path=True, ar_hidden=2.5, verbose=False
        )


def test_run_algorithm_grace_still_works_and_warns_about_ci_test(small_df):
    from causalts.algorithms import run_algorithm

    with pytest.warns(UserWarning, match="does not use the ci_test"):
        res = run_algorithm(
            "grace", small_df, object(), 1, refine="shrink", verbose=False
        )
    assert np.asarray(res.cg_tig).shape == (3, 3, 2)
    with pytest.warns(UserWarning, match="does not use the ci_test"):
        with pytest.raises(TypeError, match="foo"):
            run_stability_selection(small_df, max_lag=1, ci_test=object(), foo=1)


def test_spline_nuisance_keeps_a_discrete_parent_with_rare_levels():
    import warnings

    from causalts.grace.grow import _nuisance_block

    x = np.zeros(200)
    x[:3] = 1.0  # a rare level: spline_basis merges it away and gives no columns
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        B = _nuisance_block(x, "spline", "rff")
    assert B.shape[1] > 0
    assert _nuisance_block(np.zeros(50), "spline", "rff").shape[1] == 0


def test_ar_path_with_lambda_cv_runs():
    res = run_cdnots_gated(
        _chain(T=200),
        max_lag=1,
        ar_path=True,
        include_C=False,
        lambda_cv=True,
        **dict(FAST, max_epochs=10),
    )
    assert np.asarray(res.cg_tig).shape == (3, 3, 2)


def test_gated_model_hyperparameters_round_trip():
    # load_from_checkpoint rebuilds the model from its saved hyperparameters
    for kw in ({}, {"ar_path": True}, {"ar_path": True, "ar_hidden": 8}):
        m = GatedCausalDiscovery(num_vars=3, max_lag=1, **kw)
        m2 = GatedCausalDiscovery(**dict(m.hparams))
        assert m2.ar_path == m.ar_path
    assert GatedCausalDiscovery(num_vars=3, max_lag=1).hparams["ar_hidden"] is None


def test_self_loop_test_conditions_on_contemporaneous_parents():
    rng = np.random.default_rng(6)
    T = 800
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = 0.9 * x[t - 1] + rng.standard_normal()
    y = np.tanh(x) + 0.5 * x + rng.standard_normal(T)  # y(t) <- x(t) only
    X = np.column_stack([x, y])
    lag0 = _cross(2, 1, [(0, 1, 0)])
    assert self_loop_test(X, np.zeros_like(lag0), 1)[1, 1, 1]  # proxy without x(t)
    assert not self_loop_test(X, lag0, 1)[1, 1, 1]


def test_ar_path_self_loop_test_uses_c_of_the_model_with_supplied_skeleton():
    rng = np.random.default_rng(3)
    T = 500
    df = pd.DataFrame(
        rng.standard_normal((T, 3)) + np.linspace(0, 4, T)[:, None], columns=list("abc")
    )
    res = run_cdnots_gated(
        df,
        max_lag=1,
        skeleton=np.ones((4, 4, 2), np.int8),  # 3 variables + the linear C column
        include_C=True,
        include_C_in_model=True,
        ar_path=True,
        **dict(FAST, max_epochs=20),
    )
    G = np.asarray(res.cg_tig).astype(bool)
    assert not G[np.arange(3), np.arange(3), 1].any()  # trend only, no self-loops


def test_self_loop_test_ignores_unoriented_lag0_edges():
    # e white noise; c(t) = e(t) + e(t-1): c is e's contemporaneous child. With
    # both lag-0 directions kept, conditioning on c(t) would invent e's self-loop.
    rng = np.random.default_rng(9)
    T = 800
    e = rng.standard_normal(T)
    c = e + np.r_[0.0, e[:-1]] + 0.5 * rng.standard_normal(T)
    X = np.column_stack([c, e])
    both = _cross(2, 1, [(0, 1, 0), (1, 0, 0), (1, 0, 1)])
    assert not self_loop_test(X, both, 1)[1, 1, 1]


def test_ar_path_keeps_only_skeleton_pairs_and_ignores_self_loop_bits():
    df = _chain()
    sk = np.zeros((3, 3, 2), dtype=np.int8)
    sk[0, 1, 1] = sk[1, 2, 1] = 1
    kw = dict(max_lag=1, ar_path=True, include_C=False, **FAST)
    res0 = run_cdnots_gated(df, skeleton=sk, gate_threshold=0.0, **kw)
    off = ~np.eye(3, dtype=bool)[:, :, None]
    assert not (np.asarray(res0.cg_tig).astype(bool) & off & (sk == 0)).any()
    # self-loop entries are not gated, so they must not change lambda or the gates
    sk_self = sk.copy()
    sk_self[np.arange(3), np.arange(3), 1] = 1
    a = run_cdnots_gated(df, skeleton=sk, **kw)
    b = run_cdnots_gated(df, skeleton=sk_self, **kw)
    assert a.lambda_l0 == b.lambda_l0
    np.testing.assert_array_equal(np.asarray(a.gate_values), np.asarray(b.gate_values))
    np.testing.assert_array_equal(np.asarray(a.cg_tig), np.asarray(b.cg_tig))


def test_grow_conditions_on_c_of_the_model_with_supplied_skeleton():
    rng = np.random.default_rng(3)
    T = 500
    df = pd.DataFrame(
        rng.standard_normal((T, 2)) + np.linspace(0, 4, T)[:, None], columns=list("ab")
    )
    res = run_cdnots_gated(
        df,
        max_lag=1,
        skeleton=np.zeros((3, 3, 2), np.int8),
        include_C=True,
        include_C_in_model=True,
        grow_rounds=1,
        **dict(FAST, max_epochs=20),
    )
    assert res.grow_history[0]["n_added"] == 0  # the shared trend is C's


def test_supplied_skeleton_shape_is_checked(small_df):
    with pytest.raises(ValueError, match=r"shape \(4, 4, 2\).*C columns"):
        run_cdnots_gated(
            small_df,
            max_lag=1,
            skeleton=np.ones((3, 3, 2), np.int8),
            include_C=True,
            include_C_in_model=True,
            verbose=False,
        )
    with pytest.raises(ValueError, match=r"shape \(3, 3, 2\)"):
        run_cdnots_gated(
            small_df, max_lag=1, skeleton=np.ones((3, 3, 3)), verbose=False
        )


def test_variance_grow_conditions_on_fresh_mean_parents():
    # y(t) = x(t-1) + noise, w(t) = x(t)^2 + noise. Grow adds x -> y; the variance
    # test must not then credit the proxy w(t-1) with x's unremoved mean signal.
    rng = np.random.default_rng(0)
    T = 800
    x = rng.standard_normal(T)
    w = x**2 + 0.3 * rng.standard_normal(T)
    y = np.r_[0.0, x[:-1]] + 0.5 * rng.standard_normal(T)
    df = pd.DataFrame({"x": x, "w": w, "y": y})
    res = run_cdnots_gated(
        df,
        max_lag=1,
        skeleton=np.zeros((3, 3, 2), np.int8),
        include_C=False,
        grow_rounds=1,
        grow_transform="rank",
        grow_variance=True,
        grow_q=0.01,
        **dict(FAST, max_epochs=20),
    )
    first = res.grow_history[0]
    assert first["n_added"] >= 1  # the mean grow found x -> y
    assert first["n_added_variance"] == 0


def test_ar_path_lambda_ignores_masked_lag0_bits():
    df = _chain()
    sk = np.zeros((3, 3, 2), dtype=np.int8)
    sk[0, 1, 1] = sk[1, 2, 1] = 1
    sk_lag0 = sk.copy()
    sk_lag0[:, :, 0] = 1 - np.eye(3, dtype=np.int8)  # masked: include_instantaneous off
    kw = dict(max_lag=1, ar_path=True, include_C=False, **FAST)
    a = run_cdnots_gated(df, skeleton=sk, **kw)
    b = run_cdnots_gated(df, skeleton=sk_lag0, **kw)
    assert a.lambda_l0 == b.lambda_l0
    np.testing.assert_array_equal(np.asarray(a.cg_tig), np.asarray(b.cg_tig))


def test_spline_nuisance_fallback_applies_in_grow_with_spline_basis():
    # z: binary parent whose rare level (4 rows) merges away -> no spline columns.
    # w(t) is a near copy of z(t); without z in the nuisance, w(t-1) looks causal.
    rng = np.random.default_rng(4)
    T = 400
    z = np.zeros(T)
    z[rng.choice(T, 4, replace=False)] = 1.0
    w = z + 0.05 * rng.standard_normal(T)
    y = 3.0 * np.r_[0.0, z[:-1]] + 0.5 * rng.standard_normal(T)
    X = np.column_stack([z, w, y])
    est = _cross(3, 1, [(0, 2, 1)])
    sk = np.ones((3, 3, 2), dtype=bool)
    sk[1, 2, 1] = False  # one test: w(t-1) -> y
    added, _ = grow_candidates(
        X, est, sk, 1, 0.01, None, basis="spline", nuisance_basis="spline"
    )
    assert not added[1, 2, 1]


@pytest.mark.parametrize("basis", ["rff", "spline"])
def test_spline_nuisance_keeps_every_level_of_a_discrete_parent(basis):
    # z has levels with counts [375, 20, 4]; y is driven only by the rarest level,
    # w(t) is a continuous proxy of it. The rare level must stay in the nuisance.
    rng = np.random.default_rng(5)
    T = 399
    z = np.zeros(T)
    idx = rng.permutation(T)
    z[idx[:20]] = 1.0
    z[idx[20:24]] = 2.0
    w = (z == 2.0) + 0.05 * rng.standard_normal(T)
    y = 3.0 * np.r_[0.0, (z == 2.0)[:-1]] + 0.5 * rng.standard_normal(T)
    X = np.column_stack([z, w, y])
    est = _cross(3, 1, [(0, 2, 1)])
    sk = np.ones((3, 3, 2), dtype=bool)
    sk[1, 2, 1] = False  # one test: w(t-1) -> y
    added, _ = grow_candidates(
        X, est, sk, 1, 0.01, None, basis=basis, nuisance_basis="spline"
    )
    assert not added[1, 2, 1]


def test_grow_tests_condition_on_kept_contemporaneous_parents():
    # x AR(1); y(t) depends on x(t) only (mean and scale). x(t-1) proxies x(t).
    rng = np.random.default_rng(0)
    T = 800
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = 0.8 * x[t - 1] + rng.standard_normal()
    y = np.exp(0.45 * x) * rng.standard_normal(T) + 0.8 * x
    X = np.column_stack([x, y])
    est = _cross(2, 1, [(0, 1, 0)])  # x(t) -> y(t) kept
    sk = est | _cross(2, 1, [(0, 0, 1), (1, 1, 1), (1, 0, 1)])  # test x(t-1) -> y
    assert not grow_candidates(X, est, sk, 1, 0.01, "rank", nuisance_basis="spline")[0][
        0, 1, 1
    ]
    assert not grow_variance_candidates(X, est, sk, 1, q=0.01)[0][0, 1, 1]
    # kept both ways, x(t) - y(t) may be a child edge: not conditioned on
    both = est | _cross(2, 1, [(1, 0, 0)])
    assert grow_variance_candidates(X, both, sk | both, 1, q=0.01)[0][0, 1, 1]


def test_gated_tests_take_c_through_the_spline_on_ranked_trends():
    # Two independent series around a quadratic trend: with data_transform="rank"
    # the linear C column cannot remove the (curved) trend; its spline can.
    rng = np.random.default_rng(1)
    T = 500
    tr = (np.arange(T) / (T - 1)) ** 2
    df = pd.DataFrame(
        tr[:, None] + 0.1 * rng.standard_normal((T, 2)), columns=["a", "b"]
    )
    res = run_cdnots_gated(
        df,
        max_lag=1,
        skeleton=np.zeros((3, 3, 2), np.int8),
        include_C=True,
        include_C_in_model=True,
        ar_path=True,
        grow_rounds=1,
        grow_q=0.01,
        **dict(FAST, max_epochs=20),
    )
    G = np.asarray(res.cg_tig).astype(bool)
    assert not G.any()  # no self-loops, no grown edges: the trend is C's


def test_discrete_nuisance_levels_seen_once_or_missing_from_a_lag_window():
    # Level 2 occurs once, level 3 only in the last row (absent at lag >= 1).
    rng = np.random.default_rng(8)
    T = 300
    z = rng.integers(0, 2, T).astype(float)
    z[100] = 2.0
    z[-1] = 3.0
    y = np.r_[0.0, z[:-1]] + rng.standard_normal(T)
    X = np.column_stack([z, rng.standard_normal(T), y])
    est = _cross(3, 2, [(0, 2, 1), (0, 2, 2)])
    for basis in ("rff", "spline"):
        added, n = grow_candidates(
            X, est, est, 2, 0.1, None, basis=basis, nuisance_basis="spline"
        )
        assert n > 0 and added.dtype == bool
    kept, p = self_loop_test(X, est, 2, return_pvalues=True)
    assert np.isfinite(p[2, 2, 1:]).all()


def test_ar_path_self_loop_test_ranks_even_with_raw_data(monkeypatch):
    import causalts.grace.grow as grow_mod

    calls = []
    real = grow_mod.self_loop_test

    def spy(X, *args, **kwargs):
        calls.append(kwargs.get("transform", "rank"))
        return real(X, *args, **kwargs)

    monkeypatch.setattr(grow_mod, "self_loop_test", spy)
    run_cdnots_gated(
        _ar_only(T=200),
        max_lag=2,
        ar_path=True,
        include_C=False,
        data_transform=None,
        **FAST,
    )
    assert calls == ["rank"]
    # why: on raw heavy-tailed nulls the F-test over-rejects (shown with the RFF
    # block, where it is strongest; the default union test must hold the level too)
    null = np.zeros((2, 2, 3), bool)
    counts = {}
    for basis in ("rff", "union"):
        for transform in (None, "rank"):
            rng = np.random.default_rng(0)
            counts[basis, transform] = sum(
                self_loop_test(
                    rng.standard_t(1.5, (300, 2)),
                    null,
                    2,
                    transform=transform,
                    test_basis=basis,
                ).any()
                for _ in range(40)
            )
    assert counts["rff", None] >= 4 and counts["rff", "rank"] <= 2
    assert counts["union", "rank"] <= 2


def test_run_cdnots_gated_positional_binding_is_unchanged():
    import inspect

    params = list(inspect.signature(run_cdnots_gated).parameters.values())
    names = [p.name for p in params]
    i = names.index("grow_transform")
    assert names[i + 1 : i + 9] == [
        "data_transform",
        "noise_model",
        "refine",
        "shrink_alpha",
        "shrink_basis",
        "shrink_robust",
        "shrink_interactions",
        "grow_variance",
    ]
    new = ("grow_variance", "grow_nuisance_basis", "ar_path", "ci_test")
    assert all(
        params[names.index(n)].kind is inspect.Parameter.KEYWORD_ONLY for n in new
    )


def test_lambda_cv_ignores_the_ambient_torch_rng():
    import torch

    lams, gates = [], []
    for ambient in (11, 999):
        torch.manual_seed(ambient)
        res = run_cdnots_gated(
            _chain(T=200),
            max_lag=1,
            ar_path=True,
            include_C=False,
            lambda_cv=True,
            **dict(FAST, max_epochs=10),
        )
        lams.append(res.lambda_l0)
        gates.append(np.asarray(res.gate_values))
    assert lams[0] == lams[1]
    np.testing.assert_array_equal(gates[0], gates[1])


def test_self_loop_test_warns_when_tests_are_undefined():
    rng = np.random.default_rng(0)
    x = np.zeros(30)  # too few rows for six own lags under either test basis
    for t in range(1, 30):
        x[t] = 0.8 * x[t - 1] + rng.standard_normal()
    X = x[:, None]
    with pytest.warns(UserWarning, match="6 of 6 self-loop tests are undefined"):
        kept, pv = self_loop_test(X, np.zeros((1, 1, 7), bool), 6, return_pvalues=True)
    assert not kept.any() and np.isnan(pv[0, 0, 1:]).all()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a defined design does not warn
        assert self_loop_test(X, np.zeros((1, 1, 2), bool), 1)[0, 0, 1]


@pytest.mark.parametrize("fn", [grow_candidates, grow_variance_candidates])
def test_grow_skips_candidates_inside_the_nuisance_span(fn):
    rng = np.random.default_rng(0)
    sk = np.ones((2, 2, 2), bool)
    sk[0, 1, 1] = False  # only constant -> noise is eligible
    X = np.column_stack([np.full(28, 3.0), rng.standard_normal(28)])
    for nb in ("spline", None):
        assert fn(X, np.zeros((2, 2, 2), bool), sk, 1, 0.1, nuisance_basis=nb)[1] == 0
    # an exact copy of a parent, in the parent's own basis, adds no direction
    z = rng.standard_normal(300)
    X = np.column_stack([z, z, rng.standard_normal(300)])
    est = np.zeros((3, 3, 2), bool)
    est[0, 2, 1] = True
    sk = np.ones((3, 3, 2), bool)
    sk[1, 2, 1] = False
    assert fn(X, est, sk, 1, 0.1, nuisance_basis=None)[1] == 0


def test_block_df_drops_only_blocks_inside_the_span():
    from causalts.grace.grow import _block_df, _rff_basis

    rng = np.random.default_rng(0)
    z = rng.standard_normal(500)
    Q, _ = np.linalg.qr(np.column_stack([np.ones(500), _rff_basis(z)]))
    assert _block_df(_rff_basis(z), Q) == 0  # exact copy of a parent
    # a near-copy keeps the degrees of freedom the fit uses (no free directions)
    for delta in (1e-6, 1e-10):
        phi = _rff_basis(z + delta * rng.standard_normal(500))
        assert _block_df(phi, Q) == np.linalg.matrix_rank(phi - Q @ (Q.T @ phi))


def test_ar_path_auto_keeps_the_unimodal_guard_for_supplied_skeletons():
    sk = np.zeros((3, 3, 2), dtype=np.int8)
    sk[0, 1, 1] = sk[1, 2, 1] = 1  # exactly the true cross edges
    res = run_cdnots_gated(
        _chain(), max_lag=1, skeleton=sk, ar_path=True, include_C=False, **FAST
    )
    assert res.gate_threshold_rule == "ar_auto_unimodal" and res.gate_threshold == 0.5
    G = np.asarray(res.cg_tig).astype(bool)
    gv = np.asarray(res.gate_values)
    assert G[0, 1, 1] == (gv[0, 1, 1] >= 0.5) and G[1, 2, 1] == (gv[1, 2, 1] >= 0.5)


def test_ar_head_skips_c_columns_in_the_model():
    with pytest.raises(ValueError, match="ar_exog_cols only applies"):
        GatedCausalDiscovery(num_vars=4, max_lag=1, ar_exog_cols=1)
    for bad in (-1, 4, 1.0, True):
        with pytest.raises(ValueError, match="ar_exog_cols must be"):
            GatedCausalDiscovery(num_vars=4, max_lag=1, ar_path=True, ar_exog_cols=bad)
    m = GatedCausalDiscovery(num_vars=4, max_lag=1, ar_path=True, ar_exog_cols=1)
    assert m.ar_targets.tolist() == [1.0, 1.0, 1.0, 0.0]
    with torch.no_grad():
        m.ar_b2.fill_(5.0)  # an AR contribution the C column must not receive
        x = torch.zeros(2, 4, 2)
        m.set_skeleton_mask(np.zeros((4, 4, 2), dtype=np.int8))
        mu, _ = m(x)
    with torch.no_grad():
        m.ar_targets.fill_(1.0)  # as if C had an AR head
        mu_all, _ = m(x)
    np.testing.assert_allclose((mu_all - mu)[:, 3].numpy(), 5.0, rtol=1e-6)
    np.testing.assert_allclose((mu_all - mu)[:, :3].numpy(), 0.0, atol=1e-6)


def test_run_cdnots_gated_sets_ar_exog_cols_for_c_in_the_model(small_df, monkeypatch):
    import causalts.grace.gated_discovery as gd

    seen = []
    real = gd.GatedCausalDiscovery

    def spy(*args, **kwargs):
        seen.append(kwargs.get("ar_exog_cols", 0))
        return real(*args, **kwargs)

    monkeypatch.setattr(gd, "GatedCausalDiscovery", spy)
    run_cdnots_gated(small_df, max_lag=1, ar_path=True, include_C_in_model=True, **FAST)
    assert seen and all(k == 1 for k in seen)  # one column for c_preset="linear"
    with pytest.raises(TypeError, match="ar_exog_cols"):
        run_cdnots_gated(small_df, max_lag=1, ar_path=True, ar_exog_cols=1, **FAST)


def test_ar_targets_follow_hparams_not_the_state_dict():
    m = GatedCausalDiscovery(num_vars=4, max_lag=1, ar_path=True, ar_exog_cols=1)
    assert "ar_targets" not in m.state_dict()
    m2 = GatedCausalDiscovery(**m.hparams)
    m2.load_state_dict(m.state_dict())  # strict: no missing or unexpected keys
    assert m2.ar_targets.tolist() == [1.0, 1.0, 1.0, 0.0]


def test_compute_lambda_dT_rule():
    from causalts.grace.gated_discovery import LAMBDA_DT_CONSTANT, compute_lambda

    assert compute_lambda(100, 1000, 0.3, rule="dT") == LAMBDA_DT_CONSTANT / 1e5
    assert compute_lambda(20, 500, 0.01, rule="dT") == compute_lambda(
        20, 500, 0.9, rule="dT"
    )  # no density term
    # the default rule is the existing formula
    assert compute_lambda(100, 1000, 0.01) == max(
        0.007 + 0.16 * 0.01, 1 / 100 + 4 / 1000 - 0.4 * 0.01**0.8
    )
    with pytest.raises(ValueError, match="rule"):
        compute_lambda(10, 100, 0.1, rule="dt")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compute_lambda(10, 250, 0.1, rule="dT")  # range edges: no warning
        compute_lambda(100, 1000, 0.1, rule="dT")
    with pytest.warns(UserWarning, match="calibrated"):
        compute_lambda(3, 100, 0.1, rule="dT")


@pytest.mark.filterwarnings("ignore:lambda_rule='xcorr' was calibrated")
def test_compute_lambda_xcorr_rule():
    from causalts.grace.gated_discovery import background_xcorr, compute_lambda

    with pytest.raises(ValueError, match="xcorr"):
        compute_lambda(20, 500, 0.1, rule="xcorr")
    lam = lambda d, T, xc: compute_lambda(d, T, 0.1, rule="xcorr", xcorr=xc)  # noqa
    e = 0.1 - 0.8 / 500**0.5
    assert lam(20, 500, 0.1) == pytest.approx(
        53 * (e / 0.05) ** 0.16 / (20**1.12 * 500**0.58)
    )
    assert lam(20, 500, 0.2) > lam(20, 500, 0.05)  # more background dependence
    assert lam(20, 10**7, 0.2) == pytest.approx(0.3 / 20)  # floor at large T
    assert lam(10**4, 10**6, 0.2) == pytest.approx(0.3 / 10**4)
    assert lam(2, 30, 0.3) == 0.5  # cap
    assert lam(20, 500, 0.0) == lam(20, 500, 0.01)  # excess floored at 0.005
    with pytest.raises(ValueError, match="finite"):
        lam(20, 500, float("nan"))
    with pytest.warns(UserWarning, match="calibrated"):
        lam(3, 500, 0.1)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4000, 5))
    assert background_xcorr(X, 3) < 0.02  # ~0.8/sqrt(T) under independence
    Y = np.cumsum(rng.standard_normal((4000, 1)), 0) + 0.3 * X
    assert background_xcorr(Y, 3) > 0.5  # a shared persistent driver
    X[5, 2] = np.nan
    assert np.isfinite(background_xcorr(X, 3))


def test_run_cdnots_gated_lambda_rule():
    df = _chain()
    sk = np.zeros((3, 3, 2), dtype=np.int8)
    sk[0, 1, 1] = sk[1, 2, 1] = 1
    kw = dict(max_lag=1, ar_path=True, include_C=False, skeleton=sk, **FAST)
    a = run_cdnots_gated(df, lambda_rule="density", **kw)
    with pytest.warns(UserWarning, match=r"300/\(d\*T\)"):  # d=3 is below the range
        b = run_cdnots_gated(df, lambda_rule="dT", **kw)
    assert b.lambda_l0 == pytest.approx(300.0 / (3 * len(df)))
    # the density formula over the trained cross gates
    # (ar_path: 2 of 3*2*1 lagged cross candidates)
    from causalts.grace.gated_discovery import compute_lambda

    assert a.lambda_l0 == pytest.approx(compute_lambda(3, len(df), 2 / 6))
    c = run_cdnots_gated(df, lambda_rule="dT", lambda_l0=0.05, **kw)
    assert c.lambda_l0 == 0.05  # an explicit override still wins
    with pytest.raises(ValueError, match="lambda_rule"):
        run_cdnots_gated(df, lambda_rule="dt", **kw)
    from causalts.grace.gated_discovery import background_xcorr

    x = run_cdnots_gated(df, lambda_rule="xcorr", **kw)
    xc = background_xcorr(df.to_numpy(float), 1)
    assert x.lambda_l0 == pytest.approx(
        compute_lambda(3, len(df), 0.0, rule="xcorr", xcorr=xc)
    )
