# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fast tests for GRACE's less-travelled paths.

Stability selection, the CI skeleton, lambda cross-validation, model options
(nonlinear encoder, lag-group penalty, plain linear encoder), argument checks
and the verbose reporting of ``run_cdnots_gated``. Data are tiny (d=3,
T<=200, max_lag=1) and training runs a handful of epochs on the CPU.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from causalts.grace.gated_discovery import (
    GatedCausalDiscovery,
    StabilitySelector,
    _cdnots_graph_to_binary,
    _detect_accelerator,
    background_xcorr,
    build_smart_lambda_grid,
    compute_lambda_cv,
    prepare_data,
    run_cdnots_gated,
    run_ci_skeleton,
    run_stability_selection,
)

# Small networks keep each training run to a fraction of a second.
TINY = dict(hidden_size=8, decoder_hidden=8)


def _planted(T=200, seed=0, nan_at=None):
    """X0(t-1) -> X1(t) strong (0.9); X2 independent noise."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(T)
    x1 = np.zeros(T)
    for t in range(1, T):
        x1[t] = 0.9 * x0[t - 1] + 0.3 * rng.standard_normal()
    x2 = rng.standard_normal(T)
    df = pd.DataFrame({"X0": x0, "X1": x1, "X2": x2})
    if nan_at is not None:
        df.iloc[nan_at, 2] = np.nan
    return df


def _truth():
    g = np.zeros((3, 3, 2), dtype=np.int8)
    g[0, 1, 1] = 1
    return g


# ----------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------


def test_cdnots_graph_to_binary_strings():
    g = np.full((4, 4, 2), "", dtype="<U3")  # 4th variable plays the C node
    g[0, 1, 1] = "-->"
    g[2, 0, 0] = "<--"  # X0 -> X2 at lag 0, written from X2's side
    g[0, 3, 1] = "-->"  # into C: trimmed away
    out = _cdnots_graph_to_binary(g, n_vars=3, max_lag=1)
    assert out.dtype == np.int8 and out.shape == (3, 3, 2)
    expected = np.zeros((3, 3, 2), dtype=np.int8)
    expected[0, 1, 1] = 1
    expected[0, 2, 0] = 1
    np.testing.assert_array_equal(out, expected)


def test_cdnots_graph_to_binary_aligns_lag_axis():
    g = np.zeros((3, 3, 2), dtype=int)
    g[0, 1, 1] = 1
    padded = _cdnots_graph_to_binary(g, n_vars=3, max_lag=3)
    assert padded.shape == (3, 3, 4)
    assert padded[0, 1, 1] == 1 and padded[:, :, 2:].sum() == 0
    trimmed = _cdnots_graph_to_binary(g, n_vars=3, max_lag=0)
    assert trimmed.shape == (3, 3, 1) and trimmed.sum() == 0


def test_build_smart_lambda_grid():
    np.testing.assert_allclose(build_smart_lambda_grid(3), np.logspace(-3, 0, 3))
    grid = build_smart_lambda_grid(10)
    assert len(grid) == 10
    assert np.all(np.diff(grid) > 0)
    assert np.isclose(grid[0], 1e-3) and np.isclose(grid[-1], 1.0)
    # Most points sit in the transition region [1e-2, 10^-0.5].
    in_transition = ((grid >= 1e-2) & (grid <= 10**-0.5)).sum()
    assert in_transition >= 6
    assert len(build_smart_lambda_grid(4)) == 4


def test_background_xcorr_degenerate_inputs_and_signal():
    rng = np.random.default_rng(0)
    assert background_xcorr(rng.standard_normal((50, 1)), 1) == 0.0  # d < 2
    assert background_xcorr(rng.standard_normal((3, 3)), 1) == 0.0  # T too short
    X = np.full((20, 2), np.nan)
    assert background_xcorr(X, 1) == 0.0  # no complete rows
    coupled = _planted(T=300).to_numpy()
    noise = rng.standard_normal((300, 3))
    assert background_xcorr(coupled, 1) > background_xcorr(noise, 1)


def test_prepare_data_rejects_nan():
    with pytest.raises(ValueError, match="NaN found"):
        prepare_data(_planted(T=30, nan_at=5), max_lag=1)


def test_detect_accelerator(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert _detect_accelerator() == "cpu"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert _detect_accelerator() == "mps"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _detect_accelerator() == "gpu"


# ----------------------------------------------------------------------
# GatedCausalDiscovery options
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, err, msg",
    [
        ({"lambda_override": 1.0}, TypeError, "unexpected keyword"),
        ({"max_lag": 0, "ar_path": True}, ValueError, "max_lag >= 1"),
        ({"ar_hidden": 4}, ValueError, "ar_hidden only applies"),
        ({"ar_path": True, "ar_exog_cols": 3}, ValueError, r"\[0, num_vars\)"),
        ({"ar_path": True, "ar_exog_cols": True}, ValueError, "ar_exog_cols must"),
        ({"ar_exog_cols": 1}, ValueError, "ar_exog_cols only applies"),
        ({"noise_model": "laplace"}, ValueError, "noise_model must"),
    ],
)
def test_model_rejects_bad_options(kwargs, err, msg):
    base = dict(num_vars=3, max_lag=1)
    base.update(kwargs)
    with pytest.raises(err, match=msg):
        GatedCausalDiscovery(**base)


@pytest.mark.parametrize(
    "kwargs", [{"nonlinear_encoder": True}, {"basis_expansion": False}]
)
def test_model_encoder_variants_forward(kwargs):
    torch.manual_seed(0)
    model = GatedCausalDiscovery(num_vars=3, max_lag=1, **TINY, **kwargs)
    assert hasattr(model, "encoder") == bool(kwargs.get("nonlinear_encoder"))
    model.eval()
    mu, sig = model(torch.randn(5, 3, 2))
    assert mu.shape == sig.shape == (5, 3)
    assert torch.all(sig > 0)


def test_get_estimated_graph_threshold_none_and_signed():
    model = GatedCausalDiscovery(num_vars=3, max_lag=1, **TINY)
    sk = _truth()
    model.init_from_skeleton(sk)  # planted edge opens, the rest stay shut
    G, W = model.get_estimated_graph(threshold=None)
    assert G is None and W.shape == (3, 3, 2)
    G2, _ = model.get_estimated_graph(threshold=0.5, absval=False)
    np.testing.assert_array_equal(G2, sk)


def test_lag_group_penalty_charges_extra_lags():
    torch.manual_seed(0)
    x = torch.randn(16, 3, 3)

    def loss(**kw):
        torch.manual_seed(1)
        m = GatedCausalDiscovery(num_vars=3, max_lag=2, lambda_l0=0.5, **TINY, **kw)
        with torch.no_grad():
            m.log_alpha.fill_(3.0)  # every gate open: two lags per lagged pair
        m.eval()  # deterministic forward ...
        mu, sig = m(x)
        m.train()  # ... but the training-time penalty
        return float(m.get_loss(mu, sig, x)[0].detach())

    base = loss()
    scalar = loss(lambda_lag_group=1.0, max_lags_per_pair=1)
    loose = loss(lambda_lag_group=1.0, max_lags_per_pair=3)
    array = loss(lambda_lag_group=1.0, max_lags_per_pair=np.ones((3, 3)))
    assert scalar > base
    assert np.isclose(loose, base)  # budget above the lags used: no charge
    assert np.isclose(array, scalar)


# ----------------------------------------------------------------------
# StabilitySelector without training
# ----------------------------------------------------------------------


def _selector(**kw):
    return StabilitySelector(num_vars=2, max_lag=1, **kw)


def test_prune_lags_per_pair_scalar_array_and_none():
    active = np.ones((2, 2, 2), dtype=np.int8)  # [cause, lag, effect]
    values = np.zeros((2, 2, 2))
    values[:, 1, :] = 1.0  # lag index 1 ranks first everywhere
    out = _selector()._prune_lags_per_pair(active.copy(), values)
    assert out.sum(axis=1).max() == 1
    assert (out[:, 1, :] == 1).all() and (out[:, 0, :] == 0).all()

    budget = np.array([[2, 1], [1, 2]])
    sel = _selector(model_kwargs={"max_lags_per_pair": budget})
    out = sel._prune_lags_per_pair(active.copy(), values)
    np.testing.assert_array_equal(out.sum(axis=1), budget)

    sel = _selector()
    sel.max_lags_per_pair = None
    np.testing.assert_array_equal(sel._prune_lags_per_pair(active.copy(), values), 1)


def test_get_graph_orients_and_prunes_scores():
    sel = _selector()
    scores = np.zeros((2, 2, 2))  # internal [cause, lag(L..0), effect]
    scores[0, 0, 1] = 0.9  # X0 -> X1 at lag 1
    scores[0, 1, 1] = 0.7  # X0 -> X1 at lag 0 (weaker; pruned by the 1-lag budget)
    scores[1, 0, 0] = 0.3  # below threshold
    sel._stability_scores = scores
    s = sel.get_stability_scores()  # [cause, effect, lag 0..L]
    assert s[0, 1, 1] == 0.9 and s[0, 1, 0] == 0.7 and s[1, 0, 1] == 0.3
    G, sc = sel.get_graph(threshold=0.6)
    np.testing.assert_array_equal(sc, s)
    assert G.dtype == np.int8
    assert G[0, 1, 1] == 1 and G.sum() == 1


def test_expected_false_positives_bound():
    sel = _selector()
    sel._selections = [np.zeros((2, 2, 2), dtype=np.int8)]
    assert sel.expected_false_positives(0.8) == 0.0  # no informative run
    a = np.zeros((2, 2, 2), dtype=np.int8)
    a[0, 0, 1] = a[1, 0, 0] = 1  # 2 edges
    b = np.zeros_like(a)
    b[0, 0, 1] = 1  # 1 edge
    sel._selections = [a, b, np.zeros_like(a)]
    # q = 1.5 edges per informative run, 2 informative runs
    assert np.isclose(sel.expected_false_positives(0.75), 1.5**2 / (0.5 * 2))
    assert sel.expected_false_positives(0.5) == float("inf")


def test_plot_stability_path():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = _selector()
    a = np.zeros((2, 2, 2), dtype=np.int8)
    a[0, 0, 1] = 1
    sel._selections = [a, a, np.zeros_like(a)]
    sel._lambda_values = [0.01, 0.01, 1.0]
    sel._stability_scores = np.mean(sel._selections[:2], axis=0)
    fig = sel.plot_stability_path(var_names=["a", "b"])
    left, right = fig.axes[:2]
    x, y = left.lines[0].get_data()
    np.testing.assert_allclose(x, [0.01, 1.0])
    np.testing.assert_allclose(y, [1.0, 0.0])
    assert [t.get_text() for t in right.get_yticklabels()] == ["b_L1"]
    plt.close(fig)


# ----------------------------------------------------------------------
# CI skeleton
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "include_C, c_preset", [(True, "linear"), (False, "linear"), (True, "linear+sin")]
)
def test_run_ci_skeleton_finds_planted_edge(include_C, c_preset, capsys):
    df = _planted()
    sk, pv = run_ci_skeleton(
        df,
        max_lag=1,
        include_C=include_C,
        c_preset=c_preset,
        verbose=True,
        return_pvalues=True,
    )
    assert "CI skeleton:" in capsys.readouterr().out
    assert sk.shape == pv.shape == (3, 3, 2)
    assert sk[0, 1, 1] == 1 and pv[0, 1, 1] < 1e-6
    assert ((pv <= 0.05) == (sk == 1)).all()
    assert sk[:, 2, :].sum() == 0  # nothing explains the independent series
    plain = run_ci_skeleton(df, max_lag=1, include_C=include_C, c_preset=c_preset)
    np.testing.assert_array_equal(plain, sk)


# ----------------------------------------------------------------------
# Stability selection end to end
# ----------------------------------------------------------------------


def test_run_stability_selection_with_ci_skeleton(capsys):
    torch.manual_seed(0)
    res = run_stability_selection(
        _planted(),
        max_lag=1,
        lambda_grid=np.array([0.01, 0.05, 0.2]),
        n_subsamples=2,
        use_ci_skeleton=True,
        max_epochs=6,
        warm_start_epochs=3,
        patience=3,
        device="cpu",
        verbose=True,
        ground_truth=_truth(),
        model_seed=0,
        lambda_lag_group=1.0,
        **TINY,
    )
    out = capsys.readouterr().out
    for s in (
        "Auto batch_size",
        "Running CI-based skeleton discovery",
        "Per-pair lag budget",
        "3 lambdas x 2 subsamples = 6 max runs",
        "Expected false positives",
        "Evaluation vs Ground Truth",
    ):
        assert s in out
    assert res.cg_tig.shape == res.stability_scores.shape == (3, 3, 2)
    assert set(np.unique(res.cg_tig)) <= {0, 1}
    assert ((res.stability_scores >= 0) & (res.stability_scores <= 1)).all()
    # The planted edge starts open (skeleton init) and stays selected.
    assert res.stability_scores[0, 1, 1] == 1.0
    assert res.cg_tig[0, 1, 1] == 1


def test_run_stability_selection_impute_underdetermined_and_early_stop(capsys):
    df = _planted(T=30, nan_at=[4, 11])
    with pytest.warns(UserWarning, match="causal_iterative"):
        res = run_stability_selection(
            df,
            max_lag=1,
            lambda_grid=np.array([50.0, 100.0, 200.0, 400.0]),
            impute="causal_iterative",
            normalize_l0=False,
            subsample_frac=1.0,
            max_epochs=2,
            warm_start_epochs=None,
            patience=1,
            device="cpu",
            verbose=True,
            model_seed=0,
            **TINY,
        )
    out = capsys.readouterr().out
    assert "Underdetermined problem" in out
    assert "Auto batch_size=16" in out
    # Every gate starts shut and the huge penalties keep it shut, so the
    # selector stops after three empty lambdas and reports no edges.
    assert "Early stop: 3 consecutive lambdas" in out
    assert "0 informative" in out
    assert res.cg_tig.sum() == 0 and res.stability_scores.sum() == 0
    assert not res._df.isna().any().any()  # the imputed frame is kept


# ----------------------------------------------------------------------
# Lambda cross-validation
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_kwargs", [None, {"nonlinear_encoder": True, "ar_path": True, **TINY}]
)
def test_compute_lambda_cv_returns_a_grid_value(model_kwargs, capsys):
    ds = prepare_data(_planted(T=120), max_lag=1)
    sk = _truth()
    lam = compute_lambda_cv(
        ds,
        num_vars=3,
        max_lag=1,
        skeleton=sk,
        model_kwargs=model_kwargs,
        n_lambdas=2,
        max_epochs=2,
        warm_start_epochs=1,
        patience=1,
        batch_size=32,
        model_seed=0,
        device="cpu",
        verbose=True,
    )
    out = capsys.readouterr().out
    assert "CV lambda selection: 2 lambdas" in out and "CV selected lambda" in out
    assert any(np.isclose(lam, g) for g in np.logspace(-2.5, -0.3, 2))


# ----------------------------------------------------------------------
# run_cdnots_gated
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, msg",
    [
        ({"lambda_rule": "foo"}, "lambda_rule must"),
        ({"refine": "foo"}, "refine must"),
        ({"refine": "shrink", "include_instantaneous": True}, "lagged edges only"),
        ({"refine": "shrink", "max_lag": 0}, "max_lag >= 1"),
        ({"refine": "shrink", "shrink_basis": "poly"}, "shrink_basis must"),
        ({"refine": "shrink", "shrink_alpha": 1.5}, "shrink_alpha must"),
        ({"contemporaneous": True}, "needs refine='shrink'"),
        (
            {"refine": "shrink", "contemporaneous": True, "contemporaneous_alpha": 0},
            "contemporaneous_alpha must",
        ),
        ({"effect_floor": 1.0}, "effect_floor must"),
        ({"shrink_effect_floor": 0.1}, "applies with refine='shrink' only"),
        (
            {"refine": "shrink", "shrink_effect_floor": 0.1, "shrink_interactions": 1},
            "does not support shrink_interactions",
        ),
        ({"refine": "shrink", "shrink_effect_floor": "fixed"}, "'adaptive'"),
        ({"refine": "shrink", "shrink_effect_floor": True}, "'adaptive'"),
        ({"refine": "shrink", "shrink_effect_floor": 1.2}, "'adaptive'"),
        ({"grow_rounds": 1, "grow_q": 0.0}, "grow_q must"),
        ({"grow_rounds": 1, "grow_transform": "log"}, "grow_transform must"),
        ({"grow_nuisance_basis": "poly"}, "grow_nuisance_basis must"),
    ],
)
def test_run_cdnots_gated_rejects_bad_options(kwargs, msg):
    kw = dict(max_lag=1, verbose=False, device="cpu")
    kw.update(kwargs)
    with pytest.raises(ValueError, match=msg):
        run_cdnots_gated(_planted(T=40), **kw)


def test_run_cdnots_gated_shrink_rejects_missing_values():
    with pytest.raises(ValueError, match="without missing values"):
        run_cdnots_gated(
            _planted(T=40, nan_at=3), max_lag=1, refine="shrink", verbose=False
        )


def test_run_cdnots_gated_verbose_gates_impute_and_evaluation(capsys):
    df = _planted(nan_at=[7, 50])
    with pytest.warns(UserWarning, match="causal_iterative"):
        res = run_cdnots_gated(
            df,
            max_lag=1,
            impute="causal_iterative",
            max_epochs=8,
            patience=3,
            device="cpu",
            verbose=True,
            ground_truth=_truth(),
            model_seed=0,
            **TINY,
        )
    out = capsys.readouterr().out
    for s in (
        "Imputed 2 NaN values via VAR-EM",
        "Step 1: CDNOTS skeleton discovery",
        "lambda_L0=",
        "Step 2: Single gated training",
        "Training done:",
        "Grow round 1:",
        "Effect floor",
        "Evaluation vs Ground Truth",
    ):
        assert s in out
    assert res.cg_tig[0, 1, 1] == 1  # the planted edge survives
    assert res.refine == "gates" and res.grow_history[0]["round"] == 1
    assert not res._df.isna().any().any()


def test_run_cdnots_gated_verbose_supplied_skeleton(capsys):
    sk = np.ones((3, 3, 2), dtype=np.int8)
    sk[np.arange(3), np.arange(3), 0] = 0
    res = run_cdnots_gated(
        _planted(),
        max_lag=1,
        skeleton=sk,
        skeleton_runtime=1.5,
        grow_rounds=0,
        max_epochs=8,
        patience=3,
        device="cpu",
        verbose=True,
        model_seed=0,
        **TINY,
    )
    out = capsys.readouterr().out
    assert "Step 1: Using pre-computed skeleton" in out
    assert "Skeleton: 1.5s" in out
    assert res.cg_tig[0, 1, 1] == 1
    np.testing.assert_array_equal(res.skeleton, sk)


def test_run_cdnots_gated_verbose_shrink_with_floor_and_lag0(capsys):
    df = _planted(T=150)
    df["X2"] = df["X2"] + 0.8 * df["X0"]  # a lag-0 dependence for the lag-0 step
    res = run_cdnots_gated(
        df,
        max_lag=1,
        refine="shrink",
        shrink_effect_floor=0.02,
        contemporaneous=True,
        device="cpu",
        verbose=True,
    )
    out = capsys.readouterr().out
    for s in (
        "Step 2: Regression-based shrink",
        "Shrink done:",
        "Effect floor (0.02):",
        "Step 3: CDNOTS+ contemporaneous edges",
        "Lag-0:",
        "edges after shrink and the lag-0 step",
    ):
        assert s in out
    assert res.refine == "shrink" and res.contemporaneous_result is not None
    assert res.cg_tig[0, 1, 1] == 1
    # The lag-0 X0 - X2 link is found but cannot be oriented, so it shows in
    # the marks (o-o), not in the binary cg_tig.
    marks = res.to_marks()
    assert marks[0, 2, 0] != "" and marks[2, 0, 0] != ""
    assert marks[0, 1, 1] == "-->"
