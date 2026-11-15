# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the tetrad-based latent-confounder utilities in ``causalts.utils.tetrad``."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from causalts.utils.tetrad import (
    apply_tetrad_lag0_filter,
    check_tetrads,
    detect_confounded_clusters,
    detect_shared_factors,
    factor_consistency_score,
    wishart_tetrad_test,
)


def _one_factor(T, loadings, seed, n_noise=0):
    """Data with one latent factor on ``loadings`` plus ``n_noise`` pure-noise columns."""
    rng = np.random.default_rng(seed)
    F = rng.standard_normal(T)
    cols = [lam * F + rng.standard_normal(T) for lam in loadings]
    cols += [rng.standard_normal(T) for _ in range(n_noise)]
    return np.column_stack(cols)


def _two_factor(T, seed):
    """X0, X1 load on F1; X2, X3 load on F2; corr(F1, F2) = 0.5."""
    rng = np.random.default_rng(seed)
    F1 = rng.standard_normal(T)
    F2 = 0.5 * F1 + np.sqrt(0.75) * rng.standard_normal(T)
    e = rng.standard_normal((T, 4))
    return np.column_stack(
        [2 * F1 + e[:, 0], 2 * F1 + e[:, 1], 2 * F2 + e[:, 2], 2 * F2 + e[:, 3]]
    )


# --------------------------------------------------------------------------
# wishart_tetrad_test
# --------------------------------------------------------------------------


def test_wishart_tetrad_hand_computed():
    # S_ij = S_kl = 0.5, all other off-diagonals 0, unit variances:
    # t = 0.25, Var(t) = 0.25 * (1.25 + 1.25) / n, so z = sqrt(10) at n = 100.
    S = np.eye(4)
    S[0, 1] = S[1, 0] = 0.5
    S[2, 3] = S[3, 2] = 0.5
    t, z, p = wishart_tetrad_test(S, 100, 0, 1, 2, 3)
    assert t == pytest.approx(0.25)
    assert z == pytest.approx(np.sqrt(10))
    assert p == pytest.approx(2 * stats.norm.sf(np.sqrt(10)))


def test_wishart_tetrad_vanishes_for_equicorrelated():
    S = np.full((4, 4), 0.4) + 0.6 * np.eye(4)
    t, z, p = wishart_tetrad_test(S, 500, 0, 1, 2, 3)
    assert t == pytest.approx(0.0)
    assert z == pytest.approx(0.0)
    assert p == pytest.approx(1.0)


def test_wishart_tetrad_sign_flips_with_index_order():
    S = np.eye(4)
    S[0, 1] = S[1, 0] = 0.5
    S[2, 3] = S[3, 2] = 0.5
    t1, z1, p1 = wishart_tetrad_test(S, 100, 0, 1, 2, 3)
    # Swapping (j, k) swaps the two products, so the tetrad changes sign.
    t2, z2, p2 = wishart_tetrad_test(S, 100, 0, 2, 1, 3)
    assert t2 == pytest.approx(-t1)
    assert z2 == pytest.approx(-z1)
    assert p2 == pytest.approx(p1)


# --------------------------------------------------------------------------
# detect_confounded_clusters / check_tetrads
# --------------------------------------------------------------------------


def test_detect_confounded_clusters_one_factor_vanishes():
    X = _one_factor(3000, [1.0, 1.5, 2.0, 0.8], seed=0)
    res = detect_confounded_clusters(X, alpha=0.01, var_names=list("abcd"))
    assert res["n_tested"] == 1
    assert res["n_vanishing"] == 1
    assert res["vanishing_quartets"] == [(0, 1, 2, 3)]
    assert res["vanishing_names"] == [("a", "b", "c", "d")]
    assert res["flagged_pairs"] == {
        ("a", "b"),
        ("a", "c"),
        ("a", "d"),
        ("b", "c"),
        ("b", "d"),
        ("c", "d"),
    }


def test_detect_confounded_clusters_two_factor_rejects():
    res = detect_confounded_clusters(_two_factor(3000, seed=1), alpha=0.01)
    assert res["n_tested"] == 1
    assert res["n_vanishing"] == 0
    assert res["vanishing_quartets"] == []
    assert res["flagged_pairs"] == set()


def test_detect_confounded_clusters_counts_quartets_and_default_names():
    X = _one_factor(2000, [1.0, 1.2, 1.4, 1.6, 1.8], seed=2)
    res = detect_confounded_clusters(X, alpha=0.01)
    assert res["n_tested"] == 5  # C(5, 4)
    assert res["n_vanishing"] == 5
    assert all(n.startswith("X") for q in res["vanishing_names"] for n in q)
    assert len(res["flagged_pairs"]) == 10  # every pair of 5


def test_detect_confounded_clusters_fewer_than_four_vars():
    X = _one_factor(200, [1.0, 1.0, 1.0], seed=3)
    res = detect_confounded_clusters(X)
    assert res["n_tested"] == 0
    assert res["n_vanishing"] == 0


def test_check_tetrads_var_order_from_graph_and_prewhiten_flag():
    X = _one_factor(1000, [1.0, 1.5, 2.0, 0.8], seed=4)
    g = np.zeros((4, 4, 3), dtype=int)
    res = check_tetrads(X, graph=g, alpha=0.01)
    assert res["prewhitened"] is True
    assert res["var_order"] == 2
    assert res["n_vanishing"] == 1

    raw = check_tetrads(X, prewhiten=False, var_names=list("abcd"))
    assert raw["prewhitened"] is False
    assert raw["var_order"] == 1
    # Without pre-whitening the result equals a direct scan of the raw data.
    direct = detect_confounded_clusters(X, alpha=0.01, var_names=list("abcd"))
    assert raw["vanishing_names"] == direct["vanishing_names"]


def test_check_tetrads_scans_var_residuals():
    from statsmodels.tsa.api import VAR

    rng = np.random.default_rng(5)
    T = 600
    X = np.zeros((T, 4))
    for t in range(1, T):
        X[t] = 0.9 * X[t - 1] + rng.standard_normal(4)
    resid = VAR(X).fit(maxlags=1, ic=None, trend="c").resid
    res = check_tetrads(X, alpha=0.2)
    direct = detect_confounded_clusters(resid, alpha=0.2)
    assert res["vanishing_quartets"] == direct["vanishing_quartets"]
    assert res["var_order"] == 1


def test_check_tetrads_falls_back_to_raw_data_when_var_fit_fails(monkeypatch):
    import statsmodels.tsa.api as tsa_api

    class _Failing:
        def __init__(self, data):
            pass

        def fit(self, **kwargs):
            raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(tsa_api, "VAR", _Failing)
    X = _one_factor(800, [1.0, 1.5, 2.0, 0.8], seed=6)
    res = check_tetrads(X, alpha=0.01)
    direct = detect_confounded_clusters(X, alpha=0.01)
    assert res["prewhitened"] is True
    assert res["vanishing_quartets"] == direct["vanishing_quartets"]


# --------------------------------------------------------------------------
# factor_consistency_score / detect_shared_factors
# --------------------------------------------------------------------------


def test_factor_consistency_score_exact_one_factor():
    # Population correlations of a one-factor model: C_ik = l_i l_k (i != k).
    lam = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    C = np.outer(lam, lam)
    np.fill_diagonal(C, 1.0)
    # Ratios C_0k / C_1k = 0.9 / 0.8 for every k -> zero spread.
    assert factor_consistency_score(C, 0, 1) == pytest.approx(0.0, abs=1e-12)


def test_factor_consistency_score_hand_computed_cv():
    C = np.eye(5)
    # Ratios C[0, k] / C[1, k] for k = 2, 3, 4 are 1, 2, 3.
    C[0, 2:] = [0.2, 0.4, 0.6]
    C[1, 2:] = [0.2, 0.2, 0.2]
    expected = np.std([1.0, 2.0, 3.0]) / 2.0
    assert factor_consistency_score(C, 0, 1) == pytest.approx(expected, rel=1e-6)


def test_factor_consistency_score_inf_with_few_references():
    C = np.full((4, 4), 0.5) + 0.5 * np.eye(4)
    # d = 4 leaves only two reference variables.
    assert factor_consistency_score(C, 0, 1) == float("inf")
    # min_corr filter drops small references.
    C5 = np.eye(5)
    C5[1, 2:] = [0.01, 0.02, 0.5]
    assert factor_consistency_score(C5, 0, 1) == float("inf")


def test_detect_shared_factors_flags_factor_pairs_only():
    X = _one_factor(5000, [1.0, 1.2, 1.4, 1.6, 1.8], seed=7, n_noise=1)
    names = ["a", "b", "c", "d", "e", "noise"]
    res = detect_shared_factors(X, threshold=0.25, var_names=names)
    S = res["score_matrix"]
    assert S.shape == (6, 6)
    assert np.allclose(S, S.T)
    assert np.all(np.isinf(np.diag(S)))
    pairs = {(a, b) for a, b, _ in res["flagged_pairs"]}
    assert res["n_flagged"] == len(pairs) == 10
    assert all("noise" not in p for p in pairs)
    scores = [s for _, _, s in res["flagged_pairs"]]
    assert scores == sorted(scores)
    assert all(s < 0.25 for s in scores)


def test_detect_shared_factors_default_names_and_threshold_zero():
    X = _one_factor(2000, [1.0, 1.2, 1.4, 1.6, 1.8], seed=8)
    res = detect_shared_factors(X, threshold=0.0)
    assert res["n_flagged"] == 0
    loose = detect_shared_factors(X, threshold=np.inf)
    assert {a for a, _, _ in loose["flagged_pairs"]} <= {"X0", "X1", "X2", "X3"}


# --------------------------------------------------------------------------
# apply_tetrad_lag0_filter
# --------------------------------------------------------------------------


def test_apply_tetrad_lag0_filter_zeroes_only_flagged_lag0_edges():
    X = _one_factor(5000, [1.0, 1.2, 1.4, 1.6, 1.8], seed=9, n_noise=1)
    names = ["a", "b", "c", "d", "e", "noise"]
    df = pd.DataFrame(X, columns=names)
    # Graph has one extra (C-node) row/column that the filter must not touch.
    g = np.ones((7, 7, 2), dtype=int)
    out = apply_tetrad_lag0_filter(g, df, names, threshold=0.25)

    assert out is not g
    assert np.all(g == 1)  # input unchanged
    assert np.all(out[:, :, 1] == 1)  # lagged edges untouched
    factor = range(5)
    for i in factor:
        for j in factor:
            assert out[i, j, 0] == (1 if i == j else 0)
    assert np.all(out[5, :, 0] == 1) and np.all(out[:, 5, 0] == 1)
    assert np.all(out[6, :, 0] == 1) and np.all(out[:, 6, 0] == 1)


def test_apply_tetrad_lag0_filter_no_flags_returns_equal_copy():
    rng = np.random.default_rng(10)
    df = pd.DataFrame(rng.standard_normal((500, 4)), columns=list("abcd"))
    g = rng.integers(0, 2, size=(4, 4, 2))
    out = apply_tetrad_lag0_filter(g, df, list("abcd"))
    assert np.array_equal(out, g)
    assert out is not g
