# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for GRACE (gated causal discovery) algorithm."""

import numpy as np
import pandas as pd
import pytest
import torch

from causalts.grace.gated_discovery import (
    GatedCausalDiscovery,
    evaluate_graph,
    gates_bimodal,
    li_threshold,
    prepare_data,
    run_cdnots_gated,
)

DEVICE = "cpu"


def test_prepare_data(small_df):
    ds = prepare_data(small_df, max_lag=2, normalize=True)
    T, K = small_df.shape
    assert len(ds) == T - 2
    x = ds[0][0]
    assert x.shape == (K, 3)


def test_prepare_data_no_normalize(small_df):
    ds = prepare_data(small_df, max_lag=1, normalize=False)
    T, K = small_df.shape
    assert len(ds) == T - 1


def test_evaluate_graph_perfect():
    G = np.zeros((3, 3, 2), dtype=np.int8)
    G[0, 1, 1] = 1
    G[1, 2, 1] = 1
    metrics = evaluate_graph(G, G)
    assert metrics["F1"] == 1.0
    assert metrics["SHD"] == 0
    assert metrics["Precision"] == 1.0


def test_evaluate_graph_with_errors():
    G_true = np.zeros((3, 3, 2), dtype=np.int8)
    G_true[0, 1, 1] = 1
    G_true[1, 2, 1] = 1

    G_est = np.zeros((3, 3, 2), dtype=np.int8)
    G_est[0, 1, 1] = 1  # TP
    G_est[2, 0, 1] = 1  # FP

    metrics = evaluate_graph(G_est, G_true)
    assert metrics["Precision"] == 0.5
    assert metrics["SHD"] > 0


def test_gated_model_forward():
    num_vars, max_lag = 3, 2
    model = GatedCausalDiscovery(num_vars=num_vars, max_lag=max_lag)
    x = torch.randn(4, num_vars, max_lag + 1)
    x_hat, x_hat_l0 = model(x)
    assert x_hat.shape == (4, num_vars)
    assert x_hat_l0.shape == (4, num_vars)
    gates = model.get_gate_values()
    assert gates.shape[0] == num_vars


def test_run_cdnots_gated(small_df):
    res = run_cdnots_gated(
        df=small_df,
        max_lag=1,
        alpha=0.1,
        max_epochs=5,
        patience=3,
        device="cpu",
        verbose=False,
        model_seed=42,
    )
    K = small_df.shape[1]
    assert res.cg_tig.shape == (K, K, 2)
    assert res.gate_values.shape == (K, K, 2)
    assert not np.isnan(res.cg_tig).any()
    assert set(np.unique(res.cg_tig)).issubset({0, 1})


def _bimodal_gates(seed=0):
    rng = np.random.default_rng(seed)
    open_ = rng.beta(8, 2, 40)
    weak = 0.6 * rng.beta(2, 8, 200)
    return np.r_[np.zeros(300), weak, open_]


def test_li_threshold_hand_computed():
    v = np.array([0.0, 0.0, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
    # Zeros dropped; shifted values are 0.001 and 0.801, one fixed-point step.
    t = (0.801 - 0.001) / (np.log(0.801) - np.log(0.001))
    assert li_threshold(v) == pytest.approx(t + 0.1 - 1e-3)


def test_li_threshold_matches_skimage():
    filters = pytest.importorskip("skimage.filters")
    v = _bimodal_gates()
    # The 1e-3 positivity shift puts us slightly above skimage's cutoff.
    assert li_threshold(v) == pytest.approx(filters.threshold_li(v[v > 0.05]), abs=2e-3)


def test_li_threshold_separates_modes():
    v = _bimodal_gates()
    cut = li_threshold(v)
    assert 0.3 < cut < 0.6
    assert (v > cut).sum() == pytest.approx(40, abs=5)


def test_li_threshold_falls_back_on_too_few_values():
    assert li_threshold(np.zeros(50)) is None
    assert li_threshold(np.r_[np.zeros(50), 0.2, 0.9]) is None
    # Enough values overall, but only two above the cutoff.
    assert li_threshold(np.r_[np.full(20, 0.1), 0.95, 0.96]) is None
    assert li_threshold(np.r_[np.full(20, 0.1), 0.95, 0.96, 0.97]) is not None


def test_get_estimated_graph_li():
    torch.manual_seed(0)
    model = GatedCausalDiscovery(num_vars=4, max_lag=2)
    _, W = model.get_estimated_graph(threshold=None)
    G, _ = model.get_estimated_graph(threshold="li")
    cut = li_threshold(W)
    expected = W > (0.5 if cut is None else cut)
    np.testing.assert_array_equal(G, expected.astype(np.int8))


def test_run_cdnots_gated_li_records_cutoff(small_df):
    res = run_cdnots_gated(
        df=small_df,
        max_lag=1,
        alpha=0.1,
        gate_threshold="li",
        ar_path=False,  # the gated self-loop cut (the AR head prefixes the rule)
        max_epochs=5,
        patience=3,
        device="cpu",
        verbose=False,
        model_seed=42,
    )
    assert res.gate_threshold_rule in ("li", "li_fallback")
    cut = li_threshold(res.gate_values)
    if res.gate_threshold_rule == "li":
        assert res.gate_threshold == pytest.approx(cut)
    else:
        assert cut is None and res.gate_threshold == 0.5
    expected = (res.gate_values > res.gate_threshold).astype(np.int8)
    np.fill_diagonal(expected[:, :, 0], 0)
    np.testing.assert_array_equal(res.cg_tig, expected)


def test_run_cdnots_gated_fixed_records_rule(small_df):
    res = run_cdnots_gated(
        df=small_df,
        max_lag=1,
        alpha=0.1,
        gate_threshold=0.4,
        ar_path=False,
        max_epochs=5,
        patience=3,
        device="cpu",
        verbose=False,
        model_seed=42,
    )
    assert res.gate_threshold == 0.4
    assert res.gate_threshold_rule == "fixed"


def test_run_cdnots_gated_rejects_unknown_rule(small_df):
    with pytest.raises(ValueError, match="'li'"):
        run_cdnots_gated(df=small_df, max_lag=1, gate_threshold="otsu", verbose=False)


def _nonlinear_chain(T=600, seed=0):
    # x1(t) = sin(2 x0(t-1)) has almost no linear signal; x2 is independent noise.
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(T)
    x1 = np.zeros(T)
    x1[1:] = np.sin(2 * x0[:-1]) + 0.3 * rng.standard_normal(T - 1)
    return np.column_stack([x0, x1, rng.standard_normal(T)])


def test_grow_candidates_finds_missed_nonlinear_edge():
    from causalts.grace.grow import grow_candidates

    X = _nonlinear_chain()
    empty = np.zeros((3, 3, 3), dtype=bool)
    added, n_tests = grow_candidates(X, empty, empty, max_lag=2)
    assert n_tests == 3 * 3 * 2  # every pair at lags 1..2
    assert added[0, 1, 1]
    assert added.sum() == 1


def test_grow_candidates_skips_candidates_and_conditions_on_parents():
    from causalts.grace.grow import grow_candidates

    X = _nonlinear_chain()
    sk = np.zeros((3, 3, 3), dtype=bool)
    sk[0, 1, 1] = True
    added, n_tests = grow_candidates(X, sk, sk, max_lag=2)
    assert n_tests == 3 * 3 * 2 - 1
    # x0(t-2) only looks related to x1 through x0(t-1); given that parent it is not.
    assert not added.any()


def test_run_cdnots_gated_grow_records_history(small_df):
    res = run_cdnots_gated(
        df=small_df,
        max_lag=1,
        alpha=0.1,
        grow_rounds=2,
        max_epochs=5,
        patience=3,
        device="cpu",
        verbose=False,
        model_seed=42,
    )
    assert 1 <= len(res.grow_history) <= 2
    for h in res.grow_history:
        assert {"round", "n_tests", "n_added"} <= set(h)
    assert res.grow_history[-1]["n_added"] == 0 or len(res.grow_history) == 2


def test_run_cdnots_gated_grow_rounds_zero_is_off(small_df):
    res = run_cdnots_gated(
        df=small_df, max_lag=1, max_epochs=2, device="cpu", verbose=False, grow_rounds=0
    )
    assert res.grow_history is None


def test_run_cdnots_gated_defaults_are_the_paper_configuration(small_df):
    # gates: AR head, one grow round with scale candidates, the bounded xcorr lambda
    res = run_cdnots_gated(
        df=small_df, max_lag=1, max_epochs=2, device="cpu", verbose=False
    )
    assert res.gate_threshold_rule.startswith("ar_")
    assert res.ar_path is True and res.lambda_rule == "xcorr"
    assert len(res.grow_history) == 1
    assert "n_tests_variance" in res.grow_history[0]
    kw = dict(df=small_df, max_lag=1, max_epochs=2, device="cpu", verbose=False)
    assert res.lambda_l0 == pytest.approx(
        run_cdnots_gated(lambda_rule="xcorr", **kw).lambda_l0
    )
    assert res.lambda_l0 != pytest.approx(
        run_cdnots_gated(lambda_rule="density", **kw).lambda_l0
    )
    # shrink: the REAP configuration (three grow rounds on rank-transformed data),
    # no AR head
    sh = run_cdnots_gated(df=small_df, max_lag=1, refine="shrink", verbose=False)
    assert 1 <= len(sh.grow_history) <= 3  # stops early once a round adds nothing
    assert sh.ar_path is False and sh.lambda_rule is None
    explicit = run_cdnots_gated(
        df=small_df,
        max_lag=1,
        refine="shrink",
        grow_rounds=3,
        grow_transform="rank",
        verbose=False,
    )
    np.testing.assert_array_equal(sh.cg_tig, explicit.cg_tig)
    assert sh.grow_history == explicit.grow_history


def test_default_resolution_edges(small_df):
    kw = dict(df=small_df, max_epochs=2, device="cpu", verbose=False)
    # max_lag=0: no AR head (self-loops are lagged); the grow round has nothing to test
    r0 = run_cdnots_gated(max_lag=0, **kw)
    assert r0.ar_path is False and r0.grow_history[0]["n_tests"] == 0
    # an explicit grow_variance=False keeps the default grow round, mean tests only
    rv = run_cdnots_gated(max_lag=1, grow_variance=False, **kw)
    assert len(rv.grow_history) == 1 and "n_tests_variance" not in rv.grow_history[0]


def test_supplied_skeleton_containment_needs_grow_and_ar_off():
    rng = np.random.default_rng(0)
    T = 800
    X = rng.standard_normal((T, 3))
    for t in range(1, T):
        X[t, 0] += 0.8 * X[t - 1, 0]  # a strong self-loop the skeleton lacks
        X[t, 1] += 0.7 * X[t - 1, 0]
        X[t, 2] += 0.7 * X[t - 1, 1]  # 1 -> 2, which the skeleton lacks
    df = pd.DataFrame(X)
    sk = np.zeros((3, 3, 2), dtype=np.int8)
    sk[0, 1, 1] = 1
    kw = dict(
        max_lag=1, skeleton=sk, verbose=False, device="cpu", max_epochs=5, model_seed=0
    )
    grown = np.asarray(run_cdnots_gated(df, **kw).cg_tig) != 0
    assert grown[1, 2, 1] and grown[0, 0, 1]  # grow round and self-loop test
    strict = np.asarray(run_cdnots_gated(df, grow_rounds=0, ar_path=False, **kw).cg_tig)
    assert not ((strict != 0) & (sk == 0)).any()
    assert strict[0, 1, 1]  # the strong skeleton edge survives


def test_grow_candidates_rank_transform():
    from causalts.grace.grow import grow_candidates

    X = _nonlinear_chain()
    empty = np.zeros((3, 3, 3), dtype=bool)
    added, _ = grow_candidates(X, empty, empty, max_lag=2, transform="rank")
    assert added[0, 1, 1] and added.sum() == 1
    with pytest.raises(ValueError, match="rank"):
        grow_candidates(X, empty, empty, max_lag=2, transform="log")


def test_gates_bimodal_two_clusters_vs_one():
    rng = np.random.default_rng(0)
    two = np.r_[rng.uniform(0.1, 0.3, 60), rng.uniform(0.7, 0.9, 40)]
    one = rng.uniform(0.6, 0.8, 100)  # precise skeleton: every candidate stays open
    assert gates_bimodal(two)
    assert not gates_bimodal(one)
    assert not gates_bimodal(np.array([0.7, 0.8]))  # too few values


def test_auto_cutoff_keeps_precise_supplied_skeleton(small_df):
    # True lagged edges of the conftest chain: X0(t-1)->X1, X1(t-1)->X2.
    sk = np.zeros((3, 3, 2), dtype=np.int8)
    sk[0, 1, 1] = sk[1, 2, 1] = 1
    res = run_cdnots_gated(
        df=small_df,
        max_lag=1,
        skeleton=sk,
        ar_path=False,
        max_epochs=5,
        patience=3,
        device="cpu",
        verbose=False,
        model_seed=42,
    )
    # Two gates cannot show two clusters, so auto must not run Li here.
    assert res.gate_threshold_rule == "auto_unimodal"
    assert res.gate_threshold == 0.5


def test_auto_cutoff_uses_li_on_own_skeleton(small_df):
    res = run_cdnots_gated(
        df=small_df,
        max_lag=1,
        alpha=0.1,
        ar_path=False,
        max_epochs=5,
        patience=3,
        device="cpu",
        verbose=False,
        model_seed=42,
    )
    assert res.gate_threshold_rule in ("li", "li_fallback")


def test_student_t_noise_model_and_rank_transform(small_df):
    res = run_cdnots_gated(
        df=small_df,
        max_lag=1,
        alpha=0.1,
        max_epochs=3,
        device="cpu",
        verbose=False,
        model_seed=0,
        noise_model="student_t",
        data_transform="rank",
    )
    assert np.isfinite(res.gate_values).all()
    assert res._df is small_df  # result keeps the caller's data
    with pytest.raises(ValueError, match="noise_model"):
        GatedCausalDiscovery(num_vars=3, max_lag=1, noise_model="laplace")
    with pytest.raises(ValueError, match="data_transform"):
        run_cdnots_gated(df=small_df, max_lag=1, data_transform="log", verbose=False)


def test_rank_transform_keeps_missing_values(small_df):
    df = small_df.copy()
    df.iloc[5, 1] = np.nan
    res = run_cdnots_gated(
        df=df,
        max_lag=1,
        alpha=0.1,
        max_epochs=2,
        device="cpu",
        verbose=False,
        impute="var_em",
        model_seed=0,
    )
    assert np.isfinite(res.gate_values).all()


def test_grow_conditions_on_exogenous_trend():
    # Two independent series share only a linear trend; lagged x0 predicts x1
    # unless the trend (the C basis) is conditioned on.
    from causalts.grace.grow import grow_candidates

    rng = np.random.default_rng(1)
    T = 800
    trend = np.linspace(-2, 2, T)
    X = np.column_stack(
        [3 * trend + rng.standard_normal(T), 3 * trend + rng.standard_normal(T)]
    )
    empty = np.zeros((2, 2, 2), dtype=bool)
    sk = empty.copy()
    sk[0, 0, 1] = sk[1, 1, 1] = True  # self-lags already candidates
    added, _ = grow_candidates(X, sk, sk, max_lag=1)
    assert added[0, 1, 1]
    added, _ = grow_candidates(X, sk, sk, max_lag=1, exog=trend[:, None])
    assert not added.any()


def test_rank_transform_keeps_discrete_columns(small_df, monkeypatch):
    import types

    import causalts.cdnots.phase3_utils as p3

    df = small_df.copy()
    df["D"] = (df["X0"] > 0).astype(int)  # low-cardinality integer column
    seen = {}

    def fake_run_cdnots(df, discrete_cols=None, **kw):
        seen["df"], seen["discrete_cols"] = df, discrete_cols
        d = df.shape[1] + 1  # + C node
        return types.SimpleNamespace(cg_tig=np.zeros((d, d, 2), dtype=int))

    monkeypatch.setattr(p3, "run_cdnots", fake_run_cdnots)
    run_cdnots_gated(df=df, max_lag=1, max_epochs=1, device="cpu", verbose=False)
    assert seen["discrete_cols"] == ["D"]
    np.testing.assert_array_equal(seen["df"]["D"].to_numpy(), df["D"].to_numpy())
    assert not np.allclose(seen["df"]["X1"].to_numpy(), df["X1"].to_numpy())


def test_lambda_cv_uses_configured_noise_model(small_df, monkeypatch):
    import causalts.grace.gated_discovery as gd

    seen = {}

    def fake_cv(*args, model_kwargs=None, **kw):
        seen.update(model_kwargs)
        return 0.05

    monkeypatch.setattr(gd, "compute_lambda_cv", fake_cv)
    run_cdnots_gated(
        df=small_df,
        max_lag=1,
        alpha=0.1,
        lambda_cv=True,
        max_epochs=1,
        device="cpu",
        verbose=False,
    )
    assert seen["noise_model"] == "student_t"


def test_grace_result_repr_without_threshold():
    from causalts.grace.result import GraceResult

    res = GraceResult(
        graph=np.zeros((2, 2, 2), dtype=int),
        df=None,
        var_names=["a", "b"],
        gate_values=np.zeros((2, 2, 2)),
    )
    assert "GraceResult" in repr(res)


def test_grow_ignores_duplicated_exog():
    # Duplicated regressors add no information; a rank-blind projector would
    # remove one arbitrary extra direction per copy and change the result.
    from causalts.grace.grow import grow_candidates

    rng = np.random.default_rng(0)
    T = 300
    trend = np.linspace(-1, 1, T)
    X = np.column_stack([trend + rng.standard_normal(T), rng.standard_normal(T)])
    X[1:, 1] += 0.2 * np.sin(2 * X[:-1, 0])
    sk = np.zeros((2, 2, 2), dtype=bool)
    a1, _ = grow_candidates(X, sk, sk, max_lag=1, exog=trend[:, None])
    a30, _ = grow_candidates(X, sk, sk, max_lag=1, exog=np.tile(trend[:, None], 30))
    np.testing.assert_array_equal(a1, a30)


def test_discrete_cols_integer_labels(small_df, monkeypatch):
    import types

    import causalts.cdnots.phase3_utils as p3

    df = small_df.copy()
    df.columns = [10, 20, 30]
    df[40] = (df[10] > 0).astype(int)
    seen = {}

    def fake_run_cdnots(df, discrete_cols=None, **kw):
        seen["discrete_cols"] = discrete_cols
        d = df.shape[1] + 1
        return types.SimpleNamespace(cg_tig=np.zeros((d, d, 2), dtype=int))

    monkeypatch.setattr(p3, "run_cdnots", fake_run_cdnots)
    run_cdnots_gated(
        df=df, max_lag=1, discrete_cols=[40], max_epochs=1, device="cpu", verbose=False
    )
    assert seen["discrete_cols"] == [40]


def test_supplied_skeleton_grow_does_not_use_c(small_df, monkeypatch):
    import causalts.grace.grow as gr

    seen = []
    real = gr.grow_candidates

    def spy(*a, exog=None, **kw):
        seen.append(exog)
        return real(*a, exog=exog, **kw)

    monkeypatch.setattr(gr, "grow_candidates", spy)
    sk = np.zeros((3, 3, 2), dtype=np.int8)
    sk[0, 1, 1] = 1
    run_cdnots_gated(
        df=small_df,
        max_lag=1,
        skeleton=sk,
        grow_rounds=1,
        max_epochs=1,
        device="cpu",
        verbose=False,
    )
    assert seen and all(e is None for e in seen)


def test_grow_skips_tests_without_residual_dof():
    import warnings as _w

    from causalts.grace.grow import grow_candidates

    rng = np.random.default_rng(3)
    T, d = 40, 6
    X = rng.standard_normal((T, d))
    full = np.zeros((d, d, 4), dtype=bool)
    full[:, :, 1:] = True
    full[1, 0, 1:] = False  # leave a few candidates untested-into for target 0
    est = full.copy()
    # 15 parents x 9 features exceed n=37: every remaining test is undefined
    # and must be left out of the BH family, not counted with p = 1 or nan.
    with _w.catch_warnings():
        _w.simplefilter("error")  # no RuntimeWarning from nan p-values
        added, n_tests = grow_candidates(X, est, full, max_lag=3)
    assert n_tests == 0
    assert not added.any()


def test_nu_excluded_from_weight_decay():
    model = GatedCausalDiscovery(num_vars=3, max_lag=1, noise_model="student_t")
    opt = model.configure_optimizers()
    decay = {id(p): g["weight_decay"] for g in opt.param_groups for p in g["params"]}
    assert decay[id(model.nu_raw)] == 0.0
    assert all(v == model.wd for k, v in decay.items() if k != id(model.nu_raw))


def test_rank_transform_warns_on_detected_discrete(small_df, monkeypatch):
    import types

    import causalts.cdnots.phase3_utils as p3

    df = small_df.copy()
    df["D"] = (df["X0"] > 0).astype(int)
    monkeypatch.setattr(
        p3,
        "run_cdnots",
        lambda df, **kw: types.SimpleNamespace(
            cg_tig=np.zeros((df.shape[1] + 1,) * 2 + (2,), dtype=int)
        ),
    )
    with pytest.warns(UserWarning, match="Detected discrete columns"):
        run_cdnots_gated(df=df, max_lag=1, max_epochs=1, device="cpu", verbose=False)
