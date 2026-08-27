# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Contract / smoke tests for LUCID regime-adaptive deconfounding.

Covers the public API of :mod:`causalts.confounders` (``run_lucid``, ``LucidResult``,
``routed_deconfound``, the post-hoc ``deconfound`` layer and the standalone filters),
the result-object methods on :class:`causalts.result.CausalResult`, plus the
``apply_confounding`` synthetic-data helper.
Kept CPU-fast and seed-fixed; numeric performance claims live in
``experiments/confounders`` (the paper's reproducibility harness), not here.
"""

import numpy as np
import pandas as pd
import pytest

from causalts.cdnots.phase3_utils import run_cdnots
from causalts.cedar.discovery import run_cedar
from causalts.cedar.result import CedarResult
from causalts.ci_tests.parcorr_gpu import ParCorrGPU
from causalts.confounders import (
    LucidResult,
    deconfound,
    routed_deconfound,
    run_lucid,
    tetrad_filter,
)
from causalts.confounders.routed_deconf import routed_deconfound_lucid
from causalts.grace.gated_discovery import run_cdnots_gated
from causalts.grace.result import GraceResult
from causalts.synthetic_data.confounding import apply_confounding
from causalts.synthetic_data.synthetic_datasets import SCPGraphGenerator

D, T, MAX_LAG = 6, 500, 1
VALID_REGIMES = {"sparse", "sf", "pervasive"}


def _sparse_data(d=D, T=T, seed=0):
    """Full-rank innovations: a sparse lag-1 chain, no pervasive factor."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, d))
    for t in range(1, T):
        X[t, 1] += 0.6 * X[t - 1, 0]
        X[t, 3] += 0.6 * X[t - 1, 2]
    return pd.DataFrame(X, columns=[f"X{i}" for i in range(d)])


def _pervasive_data(d=D, T=T, n_factors=2, seed=0):
    """Low-rank innovations: a few dominant latent factors drive every variable,
    on top of two genuine lag-1 edges."""
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((T, n_factors))
    loadings = rng.standard_normal((n_factors, d))
    X = 2.0 * (factors @ loadings)
    X[1:, 1] += 0.6 * X[:-1, 0]
    X[1:, 3] += 0.6 * X[:-1, 2]
    X += 0.3 * rng.standard_normal((T, d))
    return pd.DataFrame(X, columns=[f"X{i}" for i in range(d)])


def _assert_valid_graph(g, d=D, max_lag=MAX_LAG):
    assert isinstance(g, np.ndarray)
    assert g.shape == (d, d, max_lag + 1)
    assert np.issubdtype(g.dtype, np.integer)
    assert set(np.unique(g)).issubset({0, 1})


def test_routed_deconfound_sparse_branch():
    """Full-rank data routes to the sparse regime and returns a valid graph."""
    df = _sparse_data()
    g, info = routed_deconfound(df, MAX_LAG, return_info=True)
    _assert_valid_graph(g)
    assert info["regime"] in VALID_REGIMES
    assert info["regime"] == "sparse"


def test_routed_deconfound_pervasive_branch():
    """Dominant-factor data routes off the sparse branch (pervasive family)."""
    df = _pervasive_data()
    g, info = routed_deconfound(df, MAX_LAG, return_info=True)
    _assert_valid_graph(g)
    assert info["regime"] in VALID_REGIMES
    assert info["regime"] != "sparse"
    # pervasive branch records the gate outcome
    assert "pervasive_filters" in info


def test_routed_deconfound_default_return_is_bare_graph():
    """Without return_info the call returns just the graph (no tuple)."""
    g = routed_deconfound(_sparse_data(), MAX_LAG)
    _assert_valid_graph(g)


@pytest.mark.parametrize("profile", ["adaptive", "unconditional"])
def test_profiles_run(profile):
    g = routed_deconfound(_pervasive_data(), MAX_LAG, profile=profile)
    _assert_valid_graph(g)


@pytest.mark.parametrize("router", ["auto", "spectral", "mp"])
def test_routers_run(router):
    g, info = routed_deconfound(
        _pervasive_data(), MAX_LAG, router=router, return_info=True
    )
    _assert_valid_graph(g)
    assert info["router"] == router


@pytest.mark.parametrize("regime", sorted(VALID_REGIMES))
def test_deconfound_posthoc_layer_all_regimes(regime):
    """The post-hoc filter layer runs for every regime on a supplied graph and
    never adds edges (a filter can only remove)."""
    df = _pervasive_data()
    g_in = np.ones((D, D, MAX_LAG + 1), dtype=np.int8)
    for i in range(D):  # self-loops are not meaningful inputs to the filters
        g_in[i, i, :] = 0
    g_out = deconfound(g_in, df, MAX_LAG, regime)
    _assert_valid_graph(g_out)
    assert g_out.sum() <= g_in.sum()


def test_invalid_profile_raises():
    with pytest.raises(ValueError):
        routed_deconfound(_sparse_data(), MAX_LAG, profile="nope")


def test_invalid_router_raises():
    with pytest.raises(ValueError):
        routed_deconfound(_sparse_data(), MAX_LAG, router="nope")


def test_invalid_regime_raises():
    g = np.zeros((D, D, MAX_LAG + 1), dtype=np.int8)
    with pytest.raises(ValueError):
        deconfound(g, _sparse_data(), MAX_LAG, regime="nope")


def test_apply_confounding_removes_nodes():
    """apply_confounding drops eligible nodes (>=2 distinct children) as latent
    confounders, shrinking the observed panel and its ground-truth graph
    consistently."""
    rng = np.random.default_rng(0)
    d = 6
    gt = np.zeros((d, d, 2), dtype=np.int8)
    gt[0, 1, 1] = gt[0, 2, 1] = 1  # node 0 is a fork -> eligible confounder
    gt[3, 4, 1] = 1
    var_names = [f"X{i}" for i in range(d)]
    sample = {
        "df": pd.DataFrame(rng.standard_normal((300, d)), columns=var_names),
        "ground_truth": gt,
        "var_names": var_names,
        "max_lag": 1,
    }
    out = apply_confounding(sample, confound_fraction=1.0, seed=0)
    assert out["n_eligible"] >= 1
    assert out["n_confounders"] >= 1
    assert out["df"].shape[1] == d - out["n_confounders"]
    assert out["ground_truth"].shape[0] == out["df"].shape[1]
    assert len(out["confounder_nodes"]) == out["n_confounders"]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))


# ── run_lucid / LucidResult ──────────────────────────────────────────────────
def _cdnots_result(df, max_lag=MAX_LAG, alpha=0.05):
    ci = ParCorrGPU(df.values.copy())
    return run_cdnots(
        df,
        ci,
        num_lags=max_lag,
        include_C=True,
        c_preset="linear",
        alpha=alpha,
        verbose=False,
    )


def test_run_lucid_returns_populated_result():
    res = run_lucid(_pervasive_data(), MAX_LAG)
    assert isinstance(res, LucidResult)
    _assert_valid_graph(np.asarray(res.cg_tig))
    assert res.regime in VALID_REGIMES
    assert res.var_names == [f"X{i}" for i in range(D)]
    assert 0.0 <= res.spectral_ratio and res.tau > 0
    assert res.n_factors is not None and res.n_factors >= 0
    assert res.runtime >= 0
    assert "regime" in res.info


def test_lucid_result_factor_loadings_shape_matches_n_factors():
    res = run_lucid(_pervasive_data(), MAX_LAG)
    if res.n_factors:
        assert res.factor_loadings.shape == (res.n_factors, D)
        top = res.top_factor_variables(0, 3)
        assert len(top) == 3 and all(v in res.var_names for v, _ in top)
    else:
        assert res.factor_loadings is None


def test_routed_deconfound_defaults_are_lucid():
    """The low-level entry point's defaults must BE the shipped method."""
    df = _pervasive_data()
    assert np.array_equal(
        np.asarray(routed_deconfound(df, MAX_LAG)),
        np.asarray(routed_deconfound_lucid(df, MAX_LAG)),
    )


# ── graph reuse: .deconfound() ───────────────────────────────────────────────
def test_deconfound_matches_run_lucid():
    """Reusing a discovered skeleton must be EXACT, not merely similar.

    Regression test: an earlier cut passed ``cg_tig`` unsliced, so the C-node
    rows/columns leaked into the reused graph and the results diverged.
    """
    df = _pervasive_data()
    res = run_lucid(df, MAX_LAG)
    reused = _cdnots_result(df).deconfound()
    assert isinstance(reused, LucidResult)
    assert np.array_equal(np.asarray(reused.cg_tig), np.asarray(res.cg_tig))


def test_run_lucid_accepts_raw_array_discovery():
    df = _sparse_data()
    graph = np.asarray(_cdnots_result(df).cg_tig)[:D, :D, : MAX_LAG + 1]
    res = run_lucid(df, MAX_LAG, discovery=graph)
    _assert_valid_graph(np.asarray(res.cg_tig))


def test_deconfound_num_lags_mismatch_raises():
    df = _sparse_data()
    res2 = _cdnots_result(df, max_lag=2)
    with pytest.raises(ValueError, match="num_lags"):
        run_lucid(df, MAX_LAG, discovery=res2)


def test_deconfound_alpha_mismatch_warns():
    df = _sparse_data()
    odd = _cdnots_result(df, alpha=0.2)
    with pytest.warns(UserWarning, match="alpha"):
        run_lucid(df, MAX_LAG, discovery=odd)


# ── result-object filter methods ─────────────────────────────────────────────
def test_tetrad_filter_method_returns_same_type_and_only_removes():
    cd = _cdnots_result(_pervasive_data())
    out = cd.tetrad_filter()
    assert type(out) is type(cd)
    assert out.cg_tig.sum() <= cd.cg_tig.sum()
    assert cd.cg_tig.sum() == cd.cg_tig.sum()  # original untouched


def test_tetrad_filter_method_matches_function():
    df = _pervasive_data()
    cd = _cdnots_result(df)
    method = np.asarray(cd.tetrad_filter(threshold=0.25).cg_tig)
    func = np.asarray(tetrad_filter(df, cd.cg_tig, MAX_LAG, threshold=0.25))
    assert np.array_equal(method, func)


def test_filters_chain_into_deconfound():
    cd = _cdnots_result(_pervasive_data())
    out = cd.tetrad_filter().deconfound()
    assert isinstance(out, LucidResult)


def test_pds_filter_method_only_removes():
    cd = _cdnots_result(_sparse_data())
    before = np.asarray(cd.cg_tig)[:D, :D, : MAX_LAG + 1].sum()
    out = cd.pds_filter(alpha=1e-10)
    assert type(out) is type(cd)
    assert out.cg_tig.sum() <= before


# ── exposed constants ────────────────────────────────────────────────────────
def test_exposed_constants_at_defaults_reproduce_shipped_output():
    """Passing each newly exposed knob at its default must change nothing."""
    df = _pervasive_data()
    base = np.asarray(routed_deconfound(df, MAX_LAG))
    for kw in (
        {"router_k": 2},
        {"gamma": 1.3},
        {"factor_count_margin": 1.02},
        {"tetrad_threshold": 0.25},
    ):
        assert np.array_equal(
            np.asarray(routed_deconfound(df, MAX_LAG, **kw)), base
        ), kw


def test_router_k_is_recorded_in_info():
    _, info = routed_deconfound(
        _pervasive_data(), MAX_LAG, router_k=3, return_info=True
    )
    assert info["router_k"] == 3 and info["gamma"] == 1.3


# ── unsupported combinations fail loudly ─────────────────────────────────────
def test_invalid_lag0_engine_raises():
    with pytest.raises(ValueError, match="lag0_engine"):
        routed_deconfound(_pervasive_data(), MAX_LAG, lag0_engine="adjudicate")


def test_keep_undirected_with_tetrad_base_raises():
    with pytest.raises(NotImplementedError, match="keep_undirected"):
        routed_deconfound(
            _pervasive_data(), MAX_LAG, pervasive_base="tetrad", keep_undirected=True
        )


# ── cross-type: .deconfound()/.tetrad_filter()/.pds_filter() on non-CDNOTS results ──
# _with_graph() shallow-copies `self` rather than reconstructing via __init__, so it
# should generalize to any CausalResult subclass without knowing that subclass's own
# fields (GraceResult.gate_values, CedarResult's internals, ...). Assert that directly
# instead of only ever exercising it against CdnotsResult.
#
# module-scoped: deconfound()/tetrad_filter()/pds_filter() copy before mutating (see
# _with_graph, apply_tetrad_lag0_filter, pds_filter), so the same discovery result can
# be reused across every test below instead of re-running GRACE/CEDAR discovery per test.
@pytest.fixture(scope="module")
def grace_result():
    gen = SCPGraphGenerator(n_vars=6, max_lag=1)
    data = gen.sample(seed=1, T=400)
    return run_cdnots_gated(
        df=data["df"],
        max_lag=data["max_lag"],
        verbose=False,
        device="cpu",
        model_seed=1,
    )


@pytest.fixture(scope="module")
def cedar_result():
    gen = SCPGraphGenerator(n_vars=6, max_lag=1)
    data = gen.sample(seed=2, T=400)
    ci = ParCorrGPU(data["df"].values.copy())
    return run_cedar(data["df"], ci, data["max_lag"])


@pytest.fixture(params=["grace_result", "cedar_result"])
def non_cdnots_result(request):
    return request.getfixturevalue(request.param)


def test_deconfound_generalizes_to_non_cdnots_results(non_cdnots_result):
    res = non_cdnots_result
    assert isinstance(res, (GraceResult, CedarResult))
    out = res.deconfound()
    assert isinstance(out, LucidResult)
    assert out.regime in VALID_REGIMES


def test_filters_generalize_to_non_cdnots_results(non_cdnots_result):
    res = non_cdnots_result
    same_type = type(res)
    assert same_type in (GraceResult, CedarResult)

    tet = res.tetrad_filter()
    assert type(tet) is same_type
    assert tet.cg_tig.sum() <= res.cg_tig.sum()

    pds = res.pds_filter()
    assert type(pds) is same_type
    assert pds.cg_tig.sum() <= res.cg_tig.sum()

    assert isinstance(res.tetrad_filter().deconfound(), LucidResult)


def test_grace_specific_fields_survive_the_shallow_copy(grace_result):
    """A subclass's own fields (not on the CausalResult base) must not be dropped."""
    filtered = grace_result.tetrad_filter()
    assert filtered.gate_values is grace_result.gate_values
