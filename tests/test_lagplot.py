# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from causalts.cedar.lag_selection import (  # noqa: E402
    compute_lag_importance,
    compute_lag_pvalues,
)
from causalts.plotting import (  # noqa: E402
    compute_lag_dependence,
    plot_lag_dependence,
)

SIG_COLOR = "#D55E00"
SELF_COLOR = "#999999"


@pytest.fixture
def lagged_df():
    """x -> y at lag 2 (nonlinear), z independent noise."""
    rng = np.random.default_rng(0)
    T = 400
    x = rng.standard_normal(T)
    y = np.zeros(T)
    y[2:] = np.tanh(2 * x[:-2]) + 0.3 * rng.standard_normal(T - 2)
    z = rng.standard_normal(T)
    return pd.DataFrame({"x": x, "y": y, "z": z})


def _points(ax, color):
    """Plotted (lag, value) pairs of the marker lines drawn in ``color``."""
    out = []
    for line in ax.get_lines():
        if line.get_marker() in ("", "None") or line.get_color() != color:
            continue
        for k, v in zip(line.get_xdata(), line.get_ydata()):
            if np.isfinite(v):
                out.append((int(k), float(v)))
    return out


def test_compute_matches_cedar_scores(lagged_df):
    ours = compute_lag_dependence(lagged_df, max_lag=3)
    ref = compute_lag_importance(
        lagged_df.to_numpy(), 3, method="dcor", include_lag0=True
    )
    np.testing.assert_array_equal(ours, ref)
    np.testing.assert_array_equal(
        compute_lag_dependence(lagged_df.to_numpy(), max_lag=3), ref
    )


@pytest.mark.parametrize("method", ["dcor", "dcor_biased", "pearson", "partial_dcor"])
def test_true_lag_has_peak_score(lagged_df, method):
    scores = compute_lag_dependence(lagged_df, max_lag=4, method=method)
    assert scores.shape == (3, 3, 5)
    assert np.argmax(scores[0, 1, 1:]) + 1 == 2


def test_callable_metric_matches_builtin(lagged_df):
    def abs_corr(a, b):
        return abs(np.corrcoef(a, b)[0, 1])

    np.testing.assert_allclose(
        compute_lag_dependence(lagged_df, max_lag=3, method=abs_corr),
        compute_lag_dependence(lagged_df, max_lag=3, method="pearson"),
    )


def test_input_validation(lagged_df):
    with pytest.raises(ValueError, match="Unknown method"):
        compute_lag_dependence(lagged_df, max_lag=2, method="mic")
    with pytest.raises(ValueError, match="Non-numeric"):
        compute_lag_dependence(lagged_df.assign(day="mon"), max_lag=2)
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="max_lag must be"):
            compute_lag_dependence(lagged_df, max_lag=bad)
    with pytest.raises(ValueError, match="2-D"):
        compute_lag_dependence(np.zeros(50), max_lag=2)
    with pytest.raises(ValueError, match="var_names"):
        plot_lag_dependence(lagged_df, max_lag=2, var_names=["a", "b"])
    with pytest.raises(ValueError, match="supported only"):
        plot_lag_dependence(lagged_df, max_lag=2, method="pearson", alpha=0.05)
    with pytest.raises(ValueError, match="does not test"):
        plot_lag_dependence(
            lagged_df,
            max_lag=2,
            method="partial_dcor",
            alpha=0.05,
            pvalue_method="circular_shift",
        )
    with pytest.raises(ValueError, match="must be greater than minimum"):
        plot_lag_dependence(lagged_df, max_lag=2, minimum=0.5, maximum=0.2)
    with pytest.raises(ValueError, match="set alpha too"):
        plot_lag_dependence(lagged_df, max_lag=2, pvalue_method="t_test")
    with pytest.raises(ValueError, match="alpha must be"):
        plot_lag_dependence(lagged_df, max_lag=2, alpha=5)
    with pytest.raises(ValueError, match="not attainable"):
        plot_lag_dependence(lagged_df, max_lag=2, alpha=0.01, n_permutations=99)


def test_plot_draws_scores_and_skips_lag0(lagged_df):
    fig = plot_lag_dependence(lagged_df, max_lag=3, include_lag0=False)
    assert len(fig.axes) == 10  # 9 panels + the legend axis
    # Row-major grid: axes[i * d + j] is cell (cause i, effect j).
    pts = _points(fig.axes[1], "black")
    assert sorted(k for k, _ in pts) == [1, 2, 3]
    assert max(pts, key=lambda p: p[1])[0] == 2
    # Lag-0 self-dependence is never drawn, even with include_lag0.
    fig0 = plot_lag_dependence(lagged_df, max_lag=3, include_lag0=True)
    assert 0 not in [k for k, _ in _points(fig0.axes[0], "black")]
    assert 0 in [k for k, _ in _points(fig0.axes[1], "black")]
    plt.close("all")


@pytest.mark.parametrize(
    "method,test",
    [("dcor", None), ("dcor", "t_test"), ("partial_dcor", None)],
)
def test_plot_marks_significant_lags(lagged_df, method, test):
    fig = plot_lag_dependence(
        lagged_df,
        max_lag=3,
        method=method,
        alpha=0.01,
        pvalue_method=test,
        n_permutations=199,
    )
    sig_xy = [k for k, _ in _points(fig.axes[1], SIG_COLOR)]
    assert sig_xy == [2]
    # Independent noise z: nothing significant into or out of it.
    for cell in (2, 5, 6, 7):
        assert _points(fig.axes[cell], SIG_COLOR) == []
    # Diagonal (own lags) is untested: drawn grey, never as significant.
    for cell in (0, 4, 8):
        assert _points(fig.axes[cell], SIG_COLOR) == []
        assert _points(fig.axes[cell], "black") == []
        assert len(_points(fig.axes[cell], "#999999")) == 3
    plt.close("all")


def test_plot_saves_and_accepts_callable(lagged_df, tmp_path):
    def abs_corr(a, b):
        return abs(np.corrcoef(a, b)[0, 1])

    out = tmp_path / "lags.png"
    fig = plot_lag_dependence(lagged_df, max_lag=2, method=abs_corr, fig_name=out)
    assert out.exists() and out.stat().st_size > 0
    assert fig.axes[0].texts[0].get_text() == "x"
    assert "[" not in fig.texts[-1].get_text()  # no empty lag-unit brackets
    plt.close("all")


def test_significance_with_lag0_and_random_permutation(lagged_df):
    fig = plot_lag_dependence(
        lagged_df,
        max_lag=3,
        include_lag0=True,
        alpha=0.01,
        pvalue_method="permutation",
        n_permutations=199,
    )
    assert [k for k, _ in _points(fig.axes[1], SIG_COLOR)] == [2]
    # Lag 0 is drawn for cross pairs, never for the (untested) diagonal.
    assert 0 in [k for k, _ in _points(fig.axes[1], "black")]
    assert sorted(k for k, _ in _points(fig.axes[0], "#999999")) == [1, 2, 3]
    plt.close("all")


def test_significance_rejects_nan_but_scores_accept_it(lagged_df):
    df = lagged_df.copy()
    df.iloc[30, 0] = np.nan
    for test in ("circular_shift", "t_test", "permutation"):
        with pytest.raises(ValueError, match="without NaNs"):
            plot_lag_dependence(df, max_lag=2, alpha=0.05, pvalue_method=test)
    with pytest.raises(ValueError, match="without NaNs"):
        plot_lag_dependence(df, max_lag=2, method="partial_dcor", alpha=0.05)
    scores = compute_lag_dependence(df, max_lag=2)
    assert np.isfinite(scores).all()
    plot_lag_dependence(df, max_lag=2)
    plt.close("all")


def test_repeated_column_labels_are_numeric(lagged_df):
    df = lagged_df.copy()
    df.columns = ["a", "a", "b"]
    np.testing.assert_array_equal(
        compute_lag_dependence(df, max_lag=2),
        compute_lag_dependence(lagged_df, max_lag=2),
    )


def test_default_range_includes_zero():
    data = np.random.default_rng(2).normal(size=(30, 1))
    fig = plot_lag_dependence(data, max_lag=2)
    lo, hi = fig.axes[0].get_ylim()
    assert lo <= 0 <= hi
    plt.close("all")


def test_user_range_drops_out_of_range_scores(lagged_df):
    scores = compute_lag_dependence(lagged_df, max_lag=3, method="pearson")
    fig = plot_lag_dependence(
        lagged_df, max_lag=3, method="pearson", minimum=0, maximum=0.1, y_base=0.05
    )
    lo, hi = fig.axes[1].get_ylim()
    drawn = _points(fig.axes[1], "black")
    assert all(lo <= v <= hi for _, v in drawn)
    expected = [k for k in (0, 1, 2, 3) if lo <= scores[0, 1, k] <= hi]
    assert sorted(k for k, _ in drawn) == expected and 2 not in expected
    # In-range markers keep clip_on=False (no half-cut dots at the edges).
    markers = [
        ln for ln in fig.axes[1].get_lines() if ln.get_marker() not in ("", "None")
    ]
    assert markers and not any(ln.get_clip_on() for ln in markers)
    plt.close("all")


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_significance_rejects_infinite_values(lagged_df, bad):
    df = lagged_df.copy()
    df.iloc[30, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        plot_lag_dependence(df, max_lag=2, alpha=0.05)


@pytest.mark.parametrize(
    "method,test",
    [
        ("dcor", "circular_shift"),
        ("dcor", "permutation"),
        ("dcor", "t_test"),
        ("partial_dcor", "t_test"),
    ],
)
def test_lag0_dependence_is_marked(method, test):
    rng = np.random.default_rng(43)
    x = rng.standard_normal(200)
    df = pd.DataFrame({"x": x, "y": np.tanh(2 * x) + 0.1 * rng.standard_normal(200)})
    fig = plot_lag_dependence(
        df,
        max_lag=2,
        method=method,
        include_lag0=True,
        alpha=0.01,
        pvalue_method=test,
        n_permutations=199,
    )
    assert 0 in [k for k, _ in _points(fig.axes[1], SIG_COLOR)]
    plt.close("all")


@pytest.mark.parametrize("test", ["circular_shift", "permutation", "t_test"])
@pytest.mark.parametrize("lag0", [False, True])
def test_pvalue_helper_skips_diagonal(lagged_df, test, lag0):
    # The grey "self (not tested)" markers rely on this.
    X = lagged_df.to_numpy()
    scores = compute_lag_importance(X, 2, include_lag0=lag0)
    p = compute_lag_pvalues(
        X, scores, 2, method=test, n_permutations=19, include_lag0=lag0
    )
    assert (p[np.arange(3), np.arange(3), :] == 1).all()


# --- adjust="parents" -------------------------------------------------------


@pytest.fixture
def chain_df():
    """x -> y -> z, each at lag 1 (nonlinear), so x reaches z only via y."""
    rng = np.random.default_rng(1)
    T = 500
    x = rng.standard_normal(T)
    y = np.zeros(T)
    z = np.zeros(T)
    for t in range(1, T):
        y[t] = np.tanh(2 * x[t - 1]) + 0.3 * rng.standard_normal()
        z[t] = np.tanh(2 * y[t - 1]) + 0.3 * rng.standard_normal()
    return pd.DataFrame({"x": x, "y": y, "z": z})


def test_adjust_parents_removes_indirect_lag(chain_df):
    pair = compute_lag_dependence(chain_df, max_lag=3)
    adj, parents = compute_lag_dependence(
        chain_df, max_lag=3, adjust="parents", return_parents=True
    )
    # x -> z at lag 2 exists only through y: large pairwise, ~0 after adjusting.
    assert pair[0, 2, 2] > 0.2
    assert abs(adj[0, 2, 2]) < 0.02
    # The direct links stay and peak at their lag.
    assert np.argmax(adj[0, 1, 1:]) + 1 == 1 and adj[0, 1, 1] > 0.2
    assert np.argmax(adj[1, 2, 1:]) + 1 == 1 and adj[1, 2, 1] > 0.2
    assert parents["y"] == [("x", 1)] and parents["z"] == [("y", 1)]
    # Lag 0: one symmetric score per pair; self at lag 0 is not scored.
    np.testing.assert_array_equal(adj[:, :, 0], adj[:, :, 0].T)
    assert (adj[np.arange(3), np.arange(3), 0] == 0).all()
    assert np.abs(adj[np.arange(3), np.arange(3), 1:]).max() < 0.05


def test_adjust_parents_markers(chain_df):
    fig = plot_lag_dependence(chain_df, max_lag=3, adjust="parents", alpha=0.01)
    d = 3
    sig = {
        (i, j): [k for k, _ in _points(fig.axes[i * d + j], SIG_COLOR)]
        for i in range(d)
        for j in range(d)
    }
    assert sig[(0, 1)] == [1] and sig[(1, 2)] == [1]
    # Indirect x -> z (lag 2 via y) is not flagged once y is adjusted for.
    assert 2 not in sig[(0, 2)]
    # Own lags are drawn as scores but never tested, so never marked.
    for c in (0, 4, 8):
        assert _points(fig.axes[c], SIG_COLOR) == []
        assert len(_points(fig.axes[c], "black")) == 3
    plt.close("all")


def test_adjust_parents_is_deterministic(lagged_df):
    a = compute_lag_dependence(lagged_df, max_lag=3, adjust="parents", seed=3)
    b = compute_lag_dependence(lagged_df, max_lag=3, adjust="parents", seed=3)
    np.testing.assert_array_equal(a, b)
    assert np.argmax(a[0, 1, 1:]) + 1 == 2


def test_adjust_parents_validation(lagged_df):
    with pytest.raises(ValueError, match="adjust must be"):
        compute_lag_dependence(lagged_df, max_lag=2, adjust="all")
    with pytest.raises(ValueError, match='method="dcor"'):
        compute_lag_dependence(lagged_df, max_lag=2, adjust="parents", method="pearson")
    with pytest.raises(ValueError, match="at least 100"):
        compute_lag_dependence(
            lagged_df, max_lag=2, adjust="parents", n_permutations=50
        )
    df = lagged_df.copy()
    df.iloc[5, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compute_lag_dependence(df, max_lag=2, adjust="parents")
    with pytest.raises(ValueError, match="rows after lagging"):
        compute_lag_dependence(lagged_df.iloc[:60], max_lag=6, adjust="parents")
    with pytest.raises(ValueError, match="return_parents needs"):
        compute_lag_dependence(lagged_df, max_lag=2, return_parents=True)
    with pytest.raises(ValueError, match="does not test"):
        plot_lag_dependence(
            lagged_df, max_lag=2, adjust="parents", alpha=0.05, pvalue_method="t_test"
        )


def test_mci_conditioning_set():
    from causalts.plotting._lag_adjust import _mci_conditioning

    parents = {0: [(0, 1), (2, 2)], 1: [(0, 1), (1, 1)], 2: []}
    # X0(t-1) -> X1(t): X1's other parent, then X0's parents shifted by 1.
    assert _mci_conditioning(0, 1, 1, parents) == [(1, 1), (0, 2), (2, 3)]
    # X0(t-2) -> X1(t): (0, 1) is X1's parent, not the tested source.
    assert _mci_conditioning(0, 1, 2, parents) == [(0, 1), (1, 1), (0, 3), (2, 4)]
    # A shifted source parent already among the target's parents appears once.
    shared = {0: [(2, 1)], 1: [(2, 2)], 2: []}
    assert _mci_conditioning(0, 1, 1, shared) == [(2, 2)]
    assert _mci_conditioning(2, 0, 1, parents) == [(0, 1), (2, 2)]
    # Ranking set: the source's own lags leave the shifted group only.
    assert _mci_conditioning(0, 1, 1, parents, self_history=False) == [(1, 1), (2, 3)]
    assert _mci_conditioning(0, 1, 2, parents, self_history=False) == [
        (0, 1),
        (1, 1),
        (2, 4),
    ]
    assert _mci_conditioning(2, 0, 1, parents, self_history=False) == [(0, 1), (2, 2)]


def test_rank_scores_keep_source_history_markers_do_not(monkeypatch):
    import causalts.plotting._lag_adjust as la

    # Persistent driver x -> y at lag 1, nonlinear: x's own lag is a parent.
    rng = np.random.default_rng(0)
    T = 400
    x, y = np.zeros(T), np.zeros(T)
    for t in range(1, T):
        x[t] = 0.95 * x[t - 1] + 0.3 * rng.standard_normal()
        y[t] = np.sin(2 * x[t - 1]) + 0.5 * rng.standard_normal()
    v = np.column_stack([x, y])
    scores, pvals, parents = la.parent_adjusted_scores(v, 2, 100, 0, pvalues=True)
    assert (0, 1) in parents[0]
    # The pair test finds x -> y; the marker goes to its peak lag only.
    assert pvals[0, 1, 1] < 0.01 <= pvals[0, 1, 2]
    # Frozen marker thresholds (count / 101) of this fixture.
    np.testing.assert_array_equal(np.round(pvals[0, 1, 1:] * 101), [1, 73])
    full = la._mci_conditioning
    monkeypatch.setattr(
        la, "_mci_conditioning", lambda i, j, k, p, self_history=True: full(i, j, k, p)
    )
    f_scores, _, f_parents = la.parent_adjusted_scores(v, 2, 100, 0)
    assert f_parents == parents
    # Scores skip the source's own lags, which lifts the true lag.
    assert scores[0, 1, 1] > f_scores[0, 1, 1]


def test_alpha_and_n_jobs_do_not_change_markers():
    import causalts.plotting._lag_adjust as la

    rng = np.random.default_rng(2)
    T = 300
    v = rng.standard_normal((T, 3))
    for t in range(2, T):
        v[t, 0] += 0.8 * v[t - 1, 0]
        v[t, 1] += 0.7 * np.tanh(v[t - 2, 0])
        v[t, 2] += 0.5 * v[t - 1, 1]
    S, P, par = la.parent_adjusted_scores(v, 3, 100, 1, pvalues=True)
    Sa, Pa, para = la.parent_adjusted_scores(v, 3, 100, 1, pvalues=True, alpha=0.05)
    Sj, Pj, parj = la.parent_adjusted_scores(
        v, 3, 100, 1, pvalues=True, alpha=0.05, n_jobs=2
    )
    np.testing.assert_array_equal(S, Sa)
    # Worker processes can sum in a different BLAS order (1 ulp on Windows).
    np.testing.assert_allclose(Sa, Sj, rtol=1e-12, atol=1e-15)
    assert par == para == parj
    np.testing.assert_array_equal(P < 0.05, Pa < 0.05)
    np.testing.assert_allclose(Pa, Pj, rtol=1e-12, atol=1e-15)
    # Below alpha the values are exact; at or above they are lower bounds.
    np.testing.assert_array_equal(np.where(Pa < 0.05, Pa, 0), np.where(P < 0.05, P, 0))
    assert (Pa[P >= 0.05] >= 0.05).all()


def test_marker_pvalues_rule():
    from causalts.plotting._lag_adjust import _pair_marker_pvalues

    sc = np.array([0.1, 0.5, 0.2])
    lag_p = np.array([0.002, 0.3, 0.004])
    # Significant pair: peak lag 2 takes the pair p; lags 1 and 3 need both tests.
    np.testing.assert_allclose(_pair_marker_pvalues(sc, lag_p, 0.005), [0.005] * 3)
    # Pair not significant: no lag can be starred, whatever its own test says.
    assert (_pair_marker_pvalues(sc, lag_p, 0.2) >= 0.2).all()


def test_score_pair_uses_the_marker_rule(monkeypatch):
    import causalts.plotting._lag_adjust as la

    seen = []
    rule = la._pair_marker_pvalues
    monkeypatch.setattr(
        la,
        "_pair_marker_pvalues",
        lambda sc, lag_p, pair_p: seen.append(1) or rule(sc, lag_p, pair_p),
    )
    v = np.random.default_rng(0).standard_normal((120, 2))
    la.parent_adjusted_scores(v, 2, 100, 0, pvalues=True)
    assert len(seen) == 2  # once per cross pair


def test_pair_pvalue_fast_route_matches_fallback(monkeypatch):
    from causalts.plotting import _lag_adjust as la

    rng = np.random.default_rng(0)
    X = rng.standard_normal((150, 3))
    y = np.tanh(2 * X[:, 1]) + rng.standard_normal(150)
    fast = la._pair_pvalue(X, y, 99, np.random.default_rng(1))
    monkeypatch.setattr(la, "_MAX_STACK_ENTRIES", 0)
    slow = la._pair_pvalue(X, y, 99, np.random.default_rng(1))
    assert fast == pytest.approx(slow) and fast < 0.05


def test_pair_pvalue_matches_independent_oracle():
    import dcor

    from causalts.plotting import _lag_adjust as la

    rng = np.random.default_rng(4)
    X = rng.standard_normal((120, 3))
    y = 0.25 * X[:, 2] + rng.standard_normal(120)
    n, b = 120, int(np.ceil(np.sqrt(120)))
    starts = np.arange(0, n, b)

    def stat(yv):
        return max(dcor.u_distance_correlation_sqr(X[:, c], yv) for c in range(3))

    obs, gen, count = stat(y), np.random.default_rng(2), 0
    for _ in range(99):
        yp = np.concatenate([y[s : s + b] for s in gen.permutation(starts)])
        count += stat(yp) >= obs - 1e-12 * abs(obs)
    expected = (1 + count) / 100
    assert 0.02 < expected < 0.9  # a weak case, not the minimum p
    assert la._pair_pvalue(X, y, 99, np.random.default_rng(2)) == pytest.approx(
        expected
    )


def test_pair_pvalue_constant_target_draws_nothing(monkeypatch):
    from causalts.plotting import _lag_adjust as la

    X = np.random.default_rng(0).standard_normal((60, 2))
    for limit in (la._MAX_STACK_ENTRIES, 0):
        monkeypatch.setattr(la, "_MAX_STACK_ENTRIES", limit)
        gen = np.random.default_rng(9)
        assert la._pair_pvalue(X, np.ones(60), 29, gen) == 1.0
        assert gen.integers(1 << 30) == np.random.default_rng(9).integers(1 << 30)


def test_adjust_parents_stars_a_cause_at_max_lag():
    rng = np.random.default_rng(5)
    T = 400
    x, y = rng.standard_normal(T), np.zeros(T)
    y[3:] = 0.8 * x[:-3] + 0.5 * rng.standard_normal(T - 3)
    fig = plot_lag_dependence(
        pd.DataFrame({"x": x, "y": y}), max_lag=3, adjust="parents", alpha=0.01
    )
    assert [k for k, _ in _points(fig.axes[1], SIG_COLOR)] == [3]
    plt.close("all")


def test_adjust_parents_stars_both_lags_of_a_two_lag_cause():
    rng = np.random.default_rng(3)
    T = 500
    x, y = rng.standard_normal(T), np.zeros(T)
    for t in range(2, T):
        y[t] = 0.8 * x[t - 1] + 0.6 * x[t - 2] + 0.5 * rng.standard_normal()
    fig = plot_lag_dependence(
        pd.DataFrame({"x": x, "y": y}), max_lag=3, adjust="parents", alpha=0.01
    )
    assert [k for k, _ in _points(fig.axes[1], SIG_COLOR)] == [1, 2]
    plt.close("all")


@pytest.mark.parametrize("beta", [0.0, 0.1, 0.4])
def test_max_test_fast_route_matches_fallback(beta):
    from causalts.plotting import _lag_adjust as la

    rng = np.random.default_rng(0)
    cands = [rng.standard_normal(200) for _ in range(6)]
    y = beta * np.tanh(2 * cands[2]) + rng.standard_normal(200)
    best = cands[2]
    fast = la._block_max_rejects(best, cands, y, 199, np.random.default_rng(1), 0.05)
    old = la._MAX_STACK_ENTRIES
    la._MAX_STACK_ENTRIES = 0
    try:
        slow = la._block_max_rejects(
            best, cands, y, 199, np.random.default_rng(1), 0.05
        )
    finally:
        la._MAX_STACK_ENTRIES = old
    assert fast == slow
    assert fast == (beta == 0.4)


def test_max_test_rejects_only_below_alpha():
    from causalts.plotting import _lag_adjust as la

    x = np.random.default_rng(0).standard_normal(100)
    # With 99 permutations the smallest p-value is exactly 1/100 = alpha, which
    # is not below it: even a perfectly dependent pair is not significant.
    assert not la._block_max_rejects(x, [x], x, 99, np.random.default_rng(0), 0.01)
    assert la._block_max_rejects(x, [x], x, 100, np.random.default_rng(0), 0.01)

    class OneTieRng:
        """First permutation is the identity (ties the observed value exactly);
        the rest reverse the block order (far below it)."""

        def __init__(self):
            self.first = True

        def permutation(self, starts):
            if self.first:
                self.first = False
                return starts
            return starts[::-1]

    # 199 permutations, one tie: p = 2/200 = 0.01, not below alpha = 0.01.
    assert not la._block_max_rejects(x, [x], x, 199, OneTieRng(), 0.01)
    assert la._block_max_rejects(x, [x], x, 199, OneTieRng(), 0.0101)


def test_adjust_parents_small_candidate_sets():
    rng = np.random.default_rng(1)
    v = np.zeros(150)
    for t in range(1, 150):
        v[t] = 0.95 * v[t - 1] + 0.1 * rng.standard_normal()
    scores, parents = compute_lag_dependence(
        v[:, None], 1, adjust="parents", return_parents=True, n_permutations=100
    )
    assert scores.shape == (1, 1, 2) and parents == {"X0": [("X0", 1)]}


def test_adjust_parents_null_on_persistent_series():
    # Independent AR(0.95) series: the flexible adjustment alone used to
    # extrapolate badly across blocks and flag most of these lags.
    rng = np.random.default_rng(0)
    v = rng.standard_normal((800, 3))
    for t in range(1, 800):
        v[t] += 0.95 * v[t - 1]
    fig = plot_lag_dependence(v[500:], max_lag=2, adjust="parents", alpha=0.01)
    flagged = sum(len(_points(ax, SIG_COLOR)) for ax in fig.axes[:9])
    assert flagged == 0
    plt.close("all")


def test_adjust_parents_short_series_boundary():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(56, 2))
    for t in range(1, 56):
        v[t] = 0.95 * v[t - 1] + 0.1 * v[t]
    # T=56, max_lag=3: too few rows per fold once the embargo is applied.
    with pytest.raises(ValueError, match="training rows in every"):
        compute_lag_dependence(v, 3, adjust="parents", n_permutations=100)
    # The smallest accepted length runs and gives finite scores.
    from causalts.plotting._lag_adjust import MIN_TRAIN, smallest_outer_train

    T = next(
        T
        for T in range(56, 200)
        if T - 6 >= 50 and smallest_outer_train(T - 6, 3) >= MIN_TRAIN
    )
    w = rng.normal(size=(T, 2))
    for t in range(1, T):
        w[t] = 0.95 * w[t - 1] + 0.1 * w[t]
    scores = compute_lag_dependence(w, 3, adjust="parents", n_permutations=100)
    assert np.isfinite(scores).all()


def test_joint_residuals_match_separate_fits():
    from causalts.plotting import _lag_adjust as la

    rng = np.random.default_rng(0)
    z = rng.standard_normal(300)
    # One response linear in z, one strongly nonlinear: each column picks its
    # own design, so the joint fit must equal the two single-column fits.
    Y = np.column_stack(
        [
            2 * z + 0.3 * rng.standard_normal(300),
            np.sin(3 * z) + 0.1 * rng.standard_normal(300),
        ]
    )
    joint = la._residuals(Y, [z], gap=4)
    sep = np.column_stack([la._residual(Y[:, c], [z], gap=4) for c in range(2)])
    np.testing.assert_allclose(joint, sep)
    # The nonlinear response is adjusted with the dictionary (a linear fit
    # would leave most of sin(3z) in the residual).
    assert joint[:, 1].var() < 0.2 * Y[:, 1].var()


@pytest.mark.parametrize("fast", [True, False])
@pytest.mark.parametrize(
    "n_perm,ties,alpha", [(199, 1, 0.01), (299, 20, 0.07), (99, 6, 0.07)]
)
def test_max_test_equality_boundary_both_routes(fast, n_perm, ties, alpha):
    from causalts.plotting import _lag_adjust as la

    class TieRng:
        def __init__(self):
            self.left = ties

        def permutation(self, starts):
            if self.left:
                self.left -= 1
                return starts
            return starts[::-1]

    x = np.random.default_rng(0).standard_normal(100)
    old = la._MAX_STACK_ENTRIES
    if not fast:
        la._MAX_STACK_ENTRIES = 0
    try:
        # p = (1 + ties) / (n_perm + 1) == alpha: not significant.
        assert not la._block_max_rejects(x, [x], x, n_perm, TieRng(), alpha)
    finally:
        la._MAX_STACK_ENTRIES = old


def _ar_pair(T=300, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((T, 2))
    for t in range(1, T):
        v[t, 0] += 0.9 * v[t - 1, 0]
        v[t, 1] += 0.3 * v[t - 1, 0]
    return pd.DataFrame(v, columns=["a", "b"])


def test_adjust_parents_scores_own_lags_and_caps_them():
    df = _ar_pair()
    adj = compute_lag_dependence(df, max_lag=2, adjust="parents")
    # a is strongly persistent: its own lag 1 dominates the diagonal.
    assert adj[0, 0, 1] > 0.3 and adj[0, 0, 1] > adj[0, 1, 1]
    fig = plot_lag_dependence(df, max_lag=2, adjust="parents")
    top = fig.axes[0].get_ylim()[1]
    # The axis is scaled to the cross pairs; the own lag above it is a
    # triangle at the top edge with its value written next to it.
    assert top < adj[0, 0, 1]
    tri = [ln for ln in fig.axes[0].get_lines() if ln.get_marker() == "^"]
    assert len(tri) == 1 and tri[0].get_xdata()[0] == 1
    assert f"{adj[0, 0, 1]:.2f}" in [t.get_text() for t in fig.axes[0].texts]
    plt.close("all")


def test_axis_starts_at_zero_despite_tiny_negative_scores(lagged_df):
    fig = plot_lag_dependence(lagged_df, max_lag=3, adjust="parents")
    assert fig.axes[0].get_ylim()[0] == 0.0
    plt.close("all")


def test_layout_titles_and_long_names(lagged_df):
    fig = plot_lag_dependence(lagged_df, max_lag=2)
    texts = [t.get_text() for t in fig.texts]
    assert "causes" in texts and "targets at time t" in texts
    assert any("of the cause" in t for t in texts)
    df = lagged_df.copy()
    df.columns = ["first_long_name", "second_long_name", "third_long_name"]
    fig = plot_lag_dependence(df, max_lag=2, wrap_labels=8)
    row = [t for t in fig.axes[0].texts if "first" in t.get_text()]
    assert row and row[0].get_rotation() == 90
    assert row[0].get_text() == "first_\nlong_\nname"
    plt.close("all")


@pytest.mark.parametrize(
    "name,width,expected",
    [
        ("sp500_return", 8, "sp500_\nreturn"),
        ("us_10y_treasury_yield", 8, "us_10y_\ntreasury_\nyield"),
        ("oil", 8, "oil"),
        ("fed funds rate", 8, "fed\nfunds\nrate"),
        ("abcdefghijk", 4, "abcd\nefgh\nijk"),
    ],
)
def test_wrap_names(name, width, expected):
    from causalts.plotting._core import _wrap

    assert _wrap(name, width) == expected


def test_true_graph_marks_missed_and_false_lags(lagged_df):
    d = lagged_df.shape[1]
    truth = np.zeros((d, d, 4), dtype=int)
    truth[0, 1, 2] = 1  # the real lag: found
    truth[1, 0, 3] = 1  # not in the data: missed
    fig = plot_lag_dependence(lagged_df, max_lag=3, alpha=0.01, true_graph=truth)
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert "true lag (1 found)" in labels and "true lag, missed (1)" in labels
    n_false = sum(
        1
        for ax in fig.axes
        for ln in ax.get_lines()
        if ln.get_marker() == "x" and ln.get_color() == "black" and len(ln.get_xdata())
    )
    assert f"false positive ({n_false})" in labels
    with pytest.raises(ValueError, match="true_graph must have shape"):
        plot_lag_dependence(lagged_df, max_lag=3, true_graph=np.zeros((2, 2, 4)))
    plt.close("all")


def test_min_score_filters_markers(chain_df):
    kw = dict(max_lag=3, alpha=0.05)
    base = plot_lag_dependence(chain_df, **kw)
    marked = [p for ax in base.axes for p in _points(ax, SIG_COLOR)]
    cut = float(np.median([v for _, v in marked]))
    strict = plot_lag_dependence(chain_df, min_score=cut, **kw)
    kept = [p for ax in strict.axes for p in _points(ax, SIG_COLOR)]
    assert 0 < len(kept) < len(marked) and all(v >= cut for _, v in kept)
    with pytest.raises(ValueError, match="set alpha too"):
        plot_lag_dependence(chain_df, max_lag=3, min_score=0.1)
    plt.close("all")


def test_cedar_lag_importance_uses_readable_layout(lagged_df):
    from causalts import run_cedar
    from causalts.ci_tests import ParCorrGPU

    res = run_cedar(lagged_df, ParCorrGPU(lagged_df.values), max_lag=3)
    fig = res.plot_lag_importance(figsize=(5, 5), wrap_labels=6)
    texts = [t.get_text() for t in fig.texts]
    assert "causes" in texts and "targets at time t" in texts
    plt.close("all")


def test_own_lags_outside_axis_stay_in_their_panel():
    # IID data: a negative own-lag score below the cross-pair axis.
    v = np.random.default_rng(39).normal(size=(80, 2))
    scores = compute_lag_dependence(v, 1, adjust="parents", n_permutations=100)
    fig = plot_lag_dependence(v, 1, adjust="parents", n_permutations=100)
    lo, hi = fig.axes[0].get_ylim()
    assert scores[0, 0, 1] < lo
    for c in (0, 3):
        for ln in fig.axes[c].get_lines():
            y = np.asarray(ln.get_ydata(), float)
            y = y[np.isfinite(y)]
            assert ((y >= lo - 1e-12) & (y <= hi + 1e-12)).all()
    assert [ln.get_marker() for ln in fig.axes[0].get_lines()].count("v") == 1
    plt.close("all")


def test_legend_fits_the_figure():
    v = np.random.default_rng(0).normal(size=(200, 2))
    fig = plot_lag_dependence(
        v, 2, alpha=0.05, pvalue_method="t_test", true_graph=np.zeros((2, 2, 3))
    )
    fig.canvas.draw()
    box = fig.legends[0].get_window_extent()
    assert box.x0 >= 0 and box.x1 <= fig.bbox.width
    plt.close("all")


def test_true_graph_counts_include_lag0_and_visible_markers(lagged_df):
    d = lagged_df.shape[1]
    truth = np.zeros((d, d, 4), dtype=int)
    truth[0, 1, 2] = 1
    truth[0, 1, 0] = 1  # lag 0: counted only with include_lag0
    fig = plot_lag_dependence(
        lagged_df, max_lag=3, alpha=0.01, true_graph=truth, include_lag0=True
    )
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert "true lag (1 found)" in labels and "true lag, missed (1)" in labels
    # Counts follow the tests, not the y range: a marker hidden by the range
    # still counts as found (and the truth shading stays blue).
    fig = plot_lag_dependence(
        lagged_df, max_lag=3, alpha=0.01, true_graph=truth, minimum=0, maximum=0.05
    )
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert "true lag (1 found)" in labels
    plt.close("all")


def test_layout_input_validation(lagged_df):
    with pytest.raises(ValueError, match="positive integer"):
        plot_lag_dependence(lagged_df, max_lag=2, wrap_labels=-1)
    from causalts import run_cedar
    from causalts.ci_tests import ParCorrGPU

    res = run_cedar(lagged_df, ParCorrGPU(lagged_df.values), max_lag=2)
    fig = res.plot_lag_importance(figsize=None)
    assert tuple(fig.get_size_inches()) == tuple(plt.rcParams["figure.figsize"])
    plt.close("all")


def test_adjust_parents_lag0_is_untested_and_grey():
    rng = np.random.default_rng(4)
    T = 300
    common = rng.standard_normal(T)
    v = np.column_stack(
        [common + 0.5 * rng.standard_normal(T), common + 0.5 * rng.standard_normal(T)]
    )
    adj = compute_lag_dependence(v, max_lag=2, adjust="parents")
    assert adj[0, 1, 0] == adj[1, 0, 0] > 0.3  # strong shared shock
    off = compute_lag_dependence(v, max_lag=2, adjust="parents", include_lag0=False)
    assert (off[:, :, 0] == 0).all()
    fig = plot_lag_dependence(v, max_lag=2, adjust="parents", alpha=0.01)
    # Drawn in the "not tested" grey, never as a significance marker.
    assert 0 in [k for k, _ in _points(fig.axes[1], SELF_COLOR)]
    assert 0 not in [k for k, _ in _points(fig.axes[1], SIG_COLOR)]
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert "lag 0 (not tested)" in labels
    plt.close("all")


def test_true_graph_lag0_symmetric_in_pairwise_mode():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(300)
    y = 0.9 * x + 0.3 * rng.standard_normal(300)  # x -> y at lag 0
    truth = np.zeros((2, 2, 3), dtype=int)
    truth[0, 1, 0] = 1
    fig = plot_lag_dependence(
        pd.DataFrame({"x": x, "y": y}), max_lag=2, alpha=0.01, true_graph=truth
    )
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert "true lag (1 found)" in labels
    # The reverse lag-0 marker (y -> x) is drawn but is not a false positive.
    assert 0 in [k for k, _ in _points(fig.axes[2], SIG_COLOR)]
    crosses = [
        ln.get_xdata()[0]
        for ln in fig.axes[2].get_lines()
        if ln.get_marker() == "x" and len(ln.get_xdata())
    ]
    assert 0 not in crosses
    plt.close("all")


def test_own_lag_triangles_survive_a_user_range():
    df = _ar_pair()
    fig = plot_lag_dependence(df, max_lag=2, adjust="parents", minimum=0, maximum=0.1)
    tri = [ln for ln in fig.axes[0].get_lines() if ln.get_marker() == "^"]
    assert len(tri) == 1
    plt.close("all")


def test_private_metric_name_and_seed_none(lagged_df):
    def _my_metric(x, y):
        return abs(np.corrcoef(x, y)[0, 1])

    fig = plot_lag_dependence(lagged_df, max_lag=2, method=_my_metric)
    assert [t.get_text() for t in fig.legends[0].get_texts()] == ["my_metric"]
    fig = plot_lag_dependence(
        lagged_df, max_lag=2, adjust="parents", alpha=0.05, seed=None
    )
    plt.close("all")
