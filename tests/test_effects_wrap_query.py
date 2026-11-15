# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""WrappedGraph bridge, probabilistic queries and influence measures.

Uses a d=3 linear-Gaussian chain X0(t-1) -> X1(t) -> X2(t+1) with known
coefficients, so effects and query means can be checked against the truth.
Skipped when DoWhy is not installed.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("dowhy")

import dowhy.gcm as gcm  # noqa: E402

from causalts.effects.query import (  # noqa: E402
    interventional_query,
    observational_query,
    query_summary,
)
from causalts.effects.wrap import WrappedGraph, wrap_graph  # noqa: E402

gcm.config.disable_progress_bars()

B01 = 0.8  # X0(t-1) -> X1(t)
B12 = 0.6  # X1(t-1) -> X2(t)


def _chain(T=300, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(T)
    x1 = np.zeros(T)
    x2 = np.zeros(T)
    for t in range(1, T):
        x1[t] = B01 * x0[t - 1] + shift + 0.3 * rng.standard_normal()
        x2[t] = B12 * x1[t - 1] + 0.3 * rng.standard_normal()
    return pd.DataFrame({"X0": x0, "X1": x1, "X2": x2})


def _graph():
    g = np.zeros((3, 3, 2), dtype=np.int8)
    g[0, 1, 1] = 1
    g[1, 2, 1] = 1
    return g


@pytest.fixture(scope="module")
def wrapped():
    np.random.seed(0)
    return wrap_graph(_graph(), _chain())


# ----------------------------------------------------------------------
# WrappedGraph basics
# ----------------------------------------------------------------------


def test_wrap_graph_defaults_and_repr():
    df = _chain(T=50)
    w = wrap_graph(_graph(), df)
    assert isinstance(w, WrappedGraph)
    assert w._var_names == ["X0", "X1", "X2"]
    assert w.cg_tig is not None and w.cg_tig.shape == (3, 3, 2)
    assert repr(w) == "WrappedGraph(vars=['X0', 'X1', 'X2'], shape=(3, 3, 2), edges=2)"

    w2 = wrap_graph(_graph(), df, var_names=["a", "b", "c"])
    assert w2._var_names == ["a", "b", "c"]


def test_fit_scm_is_cached_per_mechanism_and_force_refits(wrapped):
    first = wrapped.fit_scm(mechanism_type="linear")
    assert wrapped.fit_scm(mechanism_type="linear") is first
    scm, dag, lagged_df = first
    assert {"X0_lag1", "X1", "X2", "X1_lag1"} <= set(dag.nodes)
    assert ("X0_lag1", "X1") in dag.edges
    assert ("X1_lag1", "X2") in dag.edges
    assert len(lagged_df) == 299

    refit = wrapped.fit_scm(mechanism_type="linear", force=True)
    assert refit is not first
    assert wrapped.fit_scm(mechanism_type="linear") is refit


def test_estimate_effect_recovers_lagged_coefficient(wrapped):
    res = wrapped.estimate_effect("X0", "X1", treatment_lag=1, n_bootstraps=5)
    assert abs(res.value - B01) < 0.1


def test_counterfactual_sets_lagged_parent(wrapped):
    _, _, lagged = wrapped.fit_scm(mechanism_type="linear")
    cf = wrapped.counterfactual({"X0_lag1": 0.0}, "X1", mechanism_type="linear")
    assert len(cf) == len(lagged)
    # With the parent set to 0, X1 keeps only its noise (sd 0.3), well below
    # the observed sd of ~0.85.
    assert cf.std() < 0.5 < lagged["X1"].std()


def test_attribute_anomaly_blames_the_perturbed_parent(wrapped):
    _, _, lagged = wrapped.fit_scm(mechanism_type="linear")
    anomaly = lagged.iloc[[10]].copy()
    anomaly["X0_lag1"] += 5.0
    anomaly["X1"] += B01 * 5.0
    out = wrapped.attribute_anomaly(
        anomaly, "X1", mechanism_type="linear", n_samples=200
    )
    assert list(out.columns) == ["node", "variable", "lag", "mean_attribution"]
    assert out.iloc[0]["node"] == "X0_lag1"
    assert out.iloc[0]["lag"] == 1


def test_validation_wrappers_accept_the_true_graph(wrapped):
    v = wrapped.validate_transition_graph()
    assert not v.rejected
    hs = wrapped.history_sufficiency(extra_lags=1)
    assert hs.tested and not hs.rejected


@pytest.mark.parametrize(
    "method, func, kwargs",
    [
        ("falsify", "falsify_graph", {"n_permutations": 5}),
        ("refute_structure", "refute_structure", {"significance_level": 0.01}),
        ("evaluate_model", "evaluate_model", {"mechanism_type": "linear"}),
    ],
)
def test_validation_wrappers_forward_graph_data_and_kwargs(
    monkeypatch, method, func, kwargs
):
    # The real validators are slow (permutation / mechanism-fit loops) and are
    # exercised in test_effects_refutation.py; here we check the forwarding.
    import causalts.effects.validate as validate

    calls = []

    def fake(graph, data, **kw):
        calls.append((graph, data, kw))
        return "sentinel"

    monkeypatch.setattr(validate, func, fake)
    df = _chain(T=30)
    g = _graph()
    w = wrap_graph(g, df, var_names=["a", "b", "c"])
    assert getattr(w, method)(**kwargs) == "sentinel"
    graph, data, kw = calls[0]
    assert graph is g and data is df
    assert kw["var_names"] == ["a", "b", "c"]
    for k, v in kwargs.items():
        assert kw[k] == v


@pytest.mark.parametrize(
    "method, module, func",
    [
        ("refute_effect", "validate", "refute_effect"),
        ("sensitivity_analysis", "validate", "sensitivity_analysis"),
        ("to_dowhy_identification", "export", "build_identification_artifacts"),
    ],
)
def test_effect_query_wrappers_forward_treatment_and_lag(
    monkeypatch, method, module, func
):
    import importlib

    mod = importlib.import_module(f"causalts.effects.{module}")
    calls = []

    def fake(graph, data, **kw):
        calls.append((graph, data, kw))
        return "sentinel"

    monkeypatch.setattr(mod, func, fake)
    df = _chain(T=30)
    w = wrap_graph(_graph(), df)
    assert getattr(w, method)("X0", "X2", treatment_lag=2, extra=1) == "sentinel"
    graph, data, kw = calls[0]
    assert data is df and graph is w.cg_tig
    assert (kw["treatment"], kw["outcome"], kw["treatment_lag"]) == ("X0", "X2", 2)
    assert kw["var_names"] == ["X0", "X1", "X2"] and kw["extra"] == 1


def test_to_dowhy_transition_forwards(monkeypatch):
    import causalts.effects.export as export

    seen = {}

    def fake(graph, data, **kw):
        seen.update(kw)
        return "sentinel"

    monkeypatch.setattr(export, "build_transition_artifacts", fake)
    w = wrap_graph(_graph(), _chain(T=30), var_names=["a", "b", "c"])
    assert w.to_dowhy_transition(include_c=False) == "sentinel"
    assert seen == {"var_names": ["a", "b", "c"], "include_c": False}


def test_validate_transition_rejects_empty_graph():
    w = wrap_graph(np.zeros((3, 3, 2), dtype=np.int8), _chain())
    v = w.validate_transition_graph()
    assert v.rejected
    assert set(v.rejected_nodes) == {"X1", "X2"}


# ----------------------------------------------------------------------
# Probabilistic queries
# ----------------------------------------------------------------------


def test_interventional_query_mean_matches_do_effect(wrapped):
    np.random.seed(1)
    samples = wrapped.interventional_query(
        {"X0_lag1": 2.0}, targets="X1", n_samples=1000, mechanism_type="linear"
    )
    assert list(samples.columns) == ["X1"]
    assert len(samples) == 1000
    assert abs(samples["X1"].mean() - B01 * 2.0) < 0.1


def test_interventional_query_default_targets_drop_intervened(wrapped):
    scm, _, _ = wrapped.fit_scm(mechanism_type="linear")
    np.random.seed(2)
    samples = interventional_query(scm, {"X0_lag1": 1.0}, n_samples=50)
    assert "X0_lag1" not in samples.columns
    assert {"X1", "X2"} <= set(samples.columns)


def test_observational_query_soft_weights_shift_toward_evidence(wrapped):
    np.random.seed(3)
    res = wrapped.observational_query(
        {"X0_lag1": 1.5}, targets=["X1"], n_samples=3000, mechanism_type="linear"
    )
    assert list(res.columns) == ["X1", "_weight"]
    assert np.isclose(res["_weight"].sum(), 1.0)
    assert (res["_weight"] >= 0).all()
    summ = query_summary(res)
    # Kernel smoothing (bandwidth 0.5 sd) shrinks toward the prior mean 0, so
    # the weighted mean sits between 0 and the exact conditional 1.2.
    assert 0.5 < summ.loc["X1", "mean"] < B01 * 1.5 + 0.1


def test_observational_query_hard_and_narrow_kernel(wrapped):
    scm, _, lagged = wrapped.fit_scm(mechanism_type="linear")
    np.random.seed(4)
    hard = observational_query(
        scm, lagged, {"X0_lag1": 1.5}, n_samples=4000, tolerance=0.2
    )
    assert "_weight" not in hard.columns
    assert "X0_lag1" not in hard.columns  # default targets exclude evidence
    assert 0 < len(hard) < 4000
    assert abs(hard["X1"].mean() - B01 * 1.5) < 0.2

    # Evidence on a node absent from lagged_df falls back to the sample std.
    narrow = observational_query(
        scm,
        lagged.drop(columns=["X0_lag1"]),
        {"X0_lag1": 1.5},
        targets="X1",
        n_samples=4000,
        kernel_bandwidth=0.1,
    )
    m = query_summary(narrow).loc["X1", "mean"]
    assert abs(m - B01 * 1.5) < 0.2


def test_observational_query_zero_std_evidence_uses_unit_scale(wrapped):
    scm, _, lagged = wrapped.fit_scm(mechanism_type="linear")
    const = lagged.copy()
    const["X0_lag1"] = 0.0
    np.random.seed(5)
    res = observational_query(
        scm, const, {"X0_lag1": 0.0}, targets="X1", n_samples=500, tolerance=0.5
    )
    assert 0 < len(res) < 500


def test_query_summary_unweighted_and_custom_percentiles():
    df = pd.DataFrame({"a": np.arange(101, dtype=float)})
    s = query_summary(df, percentiles=[0.1, 0.5])
    assert list(s.columns) == ["mean", "std", "p10", "p50"]
    assert s.loc["a", "mean"] == 50.0
    assert s.loc["a", "p10"] == 10.0
    assert s.loc["a", "p50"] == 50.0


def test_query_summary_weighted_uses_weights():
    df = pd.DataFrame({"a": [0.0, 10.0], "_weight": [0.25, 0.75]})
    s = query_summary(df)
    assert list(s.index) == ["a"]
    assert np.isclose(s.loc["a", "mean"], 7.5)
    assert np.isclose(s.loc["a", "std"], np.sqrt(0.25 * 7.5**2 + 0.75 * 2.5**2))
    assert s.loc["a", "p05"] == 0.0
    assert s.loc["a", "p50"] == 10.0
    # The wrapper delegates to the same function.
    w = wrap_graph(_graph(), _chain(T=20))
    pd.testing.assert_frame_equal(w.query_summary(df), s)


# ----------------------------------------------------------------------
# Influence measures
# ----------------------------------------------------------------------


def test_arrow_strength_single_parent(wrapped):
    np.random.seed(6)
    out = wrapped.arrow_strength("X1", mechanism_type="linear")
    assert list(out.columns) == ["parent_node", "variable", "lag", "target", "strength"]
    assert out["parent_node"].tolist() == ["X0_lag1"]
    row = out.iloc[0]
    assert row["variable"] == "X0" and row["lag"] == 1 and row["target"] == "X1"
    # Removing the arrow costs about B01^2 * Var(X0) = 0.64 of variance.
    assert 0.4 < row["strength"] < 0.9


def test_causal_influence_splits_between_parent_and_noise(wrapped):
    np.random.seed(7)
    out = wrapped.causal_influence(
        "X2", mechanism_type="linear", num_training_samples=500
    )
    assert list(out.columns) == ["node", "variable", "lag", "influence_score"]
    scores = dict(zip(out["node"], out["influence_score"]))
    assert set(scores) == {"X1_lag1", "X2"}
    # Var from the lagged parent (0.36 * 0.73) dominates own noise (0.09).
    assert scores["X1_lag1"] > scores["X2"] > 0
    assert out["influence_score"].is_monotonic_decreasing


def test_parent_relevance_has_noise_row(wrapped):
    np.random.seed(8)
    out = wrapped.parent_relevance("X1", mechanism_type="linear")
    assert list(out.columns) == ["node", "variable", "lag", "relevance_score"]
    assert out.iloc[0]["node"] == "X0_lag1"
    noise = out.set_index("node").loc["noise"]
    assert noise["variable"] == "noise" and noise["lag"] == 0
    assert 0 < noise["relevance_score"] < out.iloc[0]["relevance_score"]


@pytest.mark.parametrize("robust", [False, True])
def test_distribution_change_blames_shifted_mechanism(wrapped, robust):
    np.random.seed(9)
    new = _chain(seed=1, shift=2.0)
    out = wrapped.distribution_change(new, "X1", mechanism_type="linear", robust=robust)
    assert list(out.columns) == ["node", "variable", "lag", "contribution"]
    contrib = out.set_index("node")["contribution"]
    assert out.iloc[0]["node"] == "X1"
    # The mean of X1 moved by +2 through its own mechanism, not through X0.
    assert contrib["X1"] > 1.0
    assert abs(contrib["X0_lag1"]) < 0.2 * contrib["X1"]


# ----------------------------------------------------------------------
# Forecast and summary
# ----------------------------------------------------------------------


def test_forecast_builds_graph_informed_model(wrapped):
    fc = wrapped.forecast()
    r = repr(fc)
    assert "causal=2" in r and "LinearRegression" in r


def test_summary_reports_edges_validation_and_strength(wrapped, capsys):
    out = wrapped.summary(top_k=2)
    text = capsys.readouterr().out
    assert out["n_edges"] == 2 and out["n_self"] == 0 and out["n_cross"] == 2
    assert not out["transition_validation"].rejected
    assert out["arrow_strength_target"] == "X1"
    assert "X0(t-1) -> X1" in text and "X1(t-1) -> X2" in text
    assert "\nDiscovered edges:" in text
    assert "\\n" not in text  # no literal backslash-n in the report


def test_summary_empty_graph_warns_and_skips_strength(capsys):
    w = wrap_graph(np.zeros((3, 3, 2), dtype=np.int8), _chain(T=200))
    out = w.summary()
    text = capsys.readouterr().out
    assert out["n_edges"] == 0
    assert out["transition_validation"].rejected
    assert "WARNING: local Markov condition violated at X1, X2" in text
    assert "arrow_strength" not in out


def test_summary_survives_failing_steps(monkeypatch, capsys):
    w = wrap_graph(_graph(), _chain(T=200))

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(w, "validate_transition_graph", boom)
    monkeypatch.setattr(w, "arrow_strength", boom)
    out = w.summary()
    text = capsys.readouterr().out
    assert "Transition-structure validation: skipped (boom)" in text
    assert "Arrow strength: skipped (boom)" in text
    assert set(out) == {"n_edges", "n_self", "n_cross"}


def test_influence_functions_default_var_names_to_columns():
    from causalts.effects import influence

    df = _chain(T=200)
    g = _graph()
    np.random.seed(10)
    a = influence.arrow_strength(g, df, "X1", mechanism_type="linear")
    assert a["parent_node"].tolist() == ["X0_lag1"]
    c = influence.causal_influence(
        g, df, "X1", mechanism_type="linear", num_training_samples=200
    )
    assert set(c["node"]) == {"X0_lag1", "X1"}
    p = influence.parent_relevance(g, df, "X1", mechanism_type="linear")
    assert set(p["node"]) == {"X0_lag1", "noise"}
    d = influence.distribution_change(g, df, df, "X1", mechanism_type="linear")
    assert np.allclose(d["contribution"], 0.0, atol=0.1)
