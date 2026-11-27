# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the tigramite wrappers in ``causalts.tigramite_discovery``.

Covers the string-graph converter, the CI-test factory, ``CITestAdapter``
(a causal-ts CI test driving PCMCI) and the ``run_*`` wrappers on a small
series with a planted lag-1 chain X0 -> X1 -> X2.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tigramite")

from causalts import tigramite_discovery as td  # noqa: E402
from causalts.ci_tests import ParCorrGPU  # noqa: E402


def _chain(T=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, 3))
    for t in range(1, T):
        x[t, 1] += 0.8 * x[t - 1, 0]
        x[t, 2] += 0.8 * x[t - 1, 1]
    return pd.DataFrame(x, columns=["X0", "X1", "X2"])


@pytest.fixture(scope="module")
def chain():
    return _chain()


def _cross_edges(G):
    return {(int(i), int(j), int(k)) for i, j, k in zip(*np.nonzero(G)) if i != j}


# ---------------------------------------------------------------------------
# tigramite_graph_to_binary
# ---------------------------------------------------------------------------


def _marks(entries, d=3, tau=2):
    g = np.full((d, d, tau + 1), "", dtype="<U3")
    for (i, j, k), m in entries.items():
        g[i, j, k] = m
    return g


def test_graph_to_binary_directed_and_partial_marks():
    g = _marks(
        {
            (0, 1, 1): "-->",
            (1, 2, 0): "o->",
            (2, 0, 0): "<--",  # lag 0: renders 0 -> 2
            (2, 1, 2): "<-o",  # lag 2: time order fixes 2(t-2) -> 1(t)
            (0, 2, 1): "<->",  # latent confounder: never an edge
        }
    )
    G = td.tigramite_graph_to_binary(g)
    assert G.dtype == np.int8 and G.shape == (3, 3, 3)
    assert {(int(i), int(j), int(k)) for i, j, k in zip(*np.nonzero(G))} == {
        (0, 1, 1),
        (1, 2, 0),
        (0, 2, 0),
        (2, 1, 2),
    }


def test_graph_to_binary_undirected_and_conflict_policies():
    g = _marks({(0, 1, 0): "o-o", (1, 0, 0): "o-o", (1, 2, 0): "x-x", (0, 2, 1): "o-o"})
    # Default: lag-0 o-o / x-x dropped; a lagged o-o is time-oriented.
    assert _cross_edges(td.tigramite_graph_to_binary(g)) == {(0, 2, 1)}
    G = td.tigramite_graph_to_binary(g, undirected="bidirected")
    assert _cross_edges(G) == {(0, 1, 0), (1, 0, 0), (0, 2, 1)}
    G = td.tigramite_graph_to_binary(g, conflict="bidirected")
    assert _cross_edges(G) == {(1, 2, 0), (2, 1, 0), (0, 2, 1)}


def test_graph_to_binary_accepts_bytes_marks():
    g = _marks({(0, 1, 1): "-->"}).astype("S3")
    assert _cross_edges(td.tigramite_graph_to_binary(g)) == {(0, 1, 1)}


def test_graph_to_binary_thresholds_on_p_matrix():
    g = _marks({(0, 1, 1): "-->", (1, 2, 1): "<--", (2, 0, 1): "o-o"})
    p = np.full(g.shape, 0.5)
    p[0, 1, 1] = 0.001
    G = td.tigramite_graph_to_binary(g, p_matrix=p, alpha_level=0.05)
    assert _cross_edges(G) == {(0, 1, 1)}
    # Without alpha_level the p_matrix is ignored.
    assert len(_cross_edges(td.tigramite_graph_to_binary(g, p_matrix=p))) == 3


@pytest.mark.parametrize("kw", [{"undirected": "keep"}, {"conflict": "keep"}])
def test_graph_to_binary_rejects_unknown_policies(kw):
    with pytest.raises(ValueError, match="must be 'drop' or 'bidirected'"):
        td.tigramite_graph_to_binary(_marks({}), **kw)


# ---------------------------------------------------------------------------
# CI-test factory and availability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, cls, module",
    [
        ("parcorr", "ParCorr", "parcorr"),
        ("CMIknn", "CMIknn", "cmiknn"),
        ("gpdc", "GPDC", "gpdc"),
        ("robust-parcorr", "RobustParCorr", "robust_parcorr"),
        ("parcorr_mult", "ParCorrMult", "parcorr_mult"),
        ("cmiknn_mixed", "CMIknnMixed", "cmiknn_mixed"),
        ("gsquared", "Gsquared", "gsquared"),
    ],
)
def test_make_tigramite_ci_test_builds_named_tests(name, cls, module):
    pytest.importorskip(f"tigramite.independence_tests.{module}")
    test = td._make_tigramite_ci_test(name)
    assert type(test).__name__ == cls


def test_make_tigramite_ci_test_forwards_kwargs_and_passes_objects_through():
    t = td._make_tigramite_ci_test("parcorr", significance="shuffle_test")
    assert t.significance == "shuffle_test"
    assert td._make_tigramite_ci_test(t) is t
    wrapped = td._make_tigramite_ci_test(ParCorrGPU(np.zeros((2, 2))))
    assert isinstance(wrapped, td.CITestAdapter)
    assert wrapped.measure == "gpu_ParCorrGPU"
    with pytest.raises(ValueError, match="Unknown CI test"):
        td._make_tigramite_ci_test("hsic")


def test_wrappers_require_tigramite(monkeypatch, chain):
    assert td.tigramite_available()
    monkeypatch.setattr(td, "tigramite_available", lambda: False)
    with pytest.raises(ImportError, match="PCMCI\\+ requires tigramite"):
        td.run_pcmciplus(chain, tau_max=1)


# ---------------------------------------------------------------------------
# CITestAdapter
# ---------------------------------------------------------------------------


def _adapter(df):
    from tigramite.data_processing import DataFrame as TigDF

    a = td.CITestAdapter(ParCorrGPU(np.zeros((2, 2))))
    a.set_dataframe(TigDF(df.to_numpy(), var_names=list(df.columns)))
    return a


def test_adapter_pvalues_match_tigramite_parcorr(chain):
    from tigramite.data_processing import DataFrame as TigDF
    from tigramite.independence_tests.parcorr import ParCorr

    a = _adapter(chain)
    ref = ParCorr()
    ref.set_dataframe(TigDF(chain.to_numpy()))
    for X, Y, Z in [
        ([(0, -1)], [(1, 0)], None),  # dependent
        ([(0, -2)], [(2, 0)], [(1, -1)]),  # independent given the mediator
        ([(2, -1)], [(0, 0)], [(0, -1)]),
    ]:
        val, pval = a.run_test(X, Y, Z, tau_max=2)
        rval, rp = ref.run_test(X, Y, Z, tau_max=2)
        assert val == pytest.approx(rval, abs=1e-6)
        assert pval == pytest.approx(rp, rel=1e-3, abs=1e-8)
    assert a.n_tests == 3
    val, pval, dep = a.run_test([(0, -1)], [(1, 0)], alpha_or_thres=0.05)
    assert dep and pval < 1e-10


def test_adapter_get_array_cutoffs(chain):
    a = _adapter(chain)
    T = len(chain)
    arr, xyz, XYZ, *_ = a._get_array([(0, -1)], [(1, 0)], [(2, -2)], tau_max=3)
    assert arr.shape == (3, T - 6) and xyz.tolist() == [0, 1, 2]
    assert XYZ == ([0], [1], [2])
    # Row 0 is X0 lagged by one relative to row 1 (X1 at t).
    np.testing.assert_array_equal(arr[0], chain["X0"].to_numpy()[5 : T - 1])
    arr, *_ = a._get_array([(0, -1)], [(1, 0)], cut_off="max_lag")
    assert arr.shape == (2, T - 1)
    arr, xyz, XYZ, *_ = a._get_array([(0, -1)], [(1, 0)], tau_max=2, cut_off="other")
    assert arr.shape == (2, T - 2) and XYZ[2] is None


def test_adapter_run_test_raw_and_stubs(chain):
    a = _adapter(chain)
    x = chain[["X0"]].to_numpy()[:-1]
    y = chain[["X1"]].to_numpy()[1:]
    z = chain[["X2"]].to_numpy()[:-1]
    v1, p1 = a.run_test_raw(x, y)
    v2, p2 = a.run_test_raw(x, y, z)
    assert p1 < 1e-10 and p2 < 1e-10 and v1 > 0.4
    assert a.get_analytic_significance(v2, 10, 3) == p2
    assert a.get_shuffle_significance(None, None, v2) == p2
    assert a.get_confidence([(0, -1)], [(1, 0)]) is None
    assert a.get_model_selection_criterion(1, [(0, -1)]) == 0.0
    arr = np.ones((2, 3))
    assert a._get_array_hash(arr, np.array([0, 1]), None) == a._get_array_hash(
        arr.copy(), np.array([0, 1]), None
    )
    assert a.save_incremental_cache("k", "h", 0.1, 0.2) is None


def test_adapter_requires_a_dataframe_and_defaults_before_any_test():
    a = td.CITestAdapter(ParCorrGPU(np.zeros((2, 2))))
    assert a.get_analytic_significance(0.0, 10, 2) == 0.0
    assert a.get_shuffle_significance(None, None, 0.0) == 0.0
    with pytest.raises(RuntimeError, match="dataframe not set"):
        a._get_array([(0, -1)], [(1, 0)])


def test_adapter_extract_data_handles_arrays_and_unknown_values():
    class _DF:
        def __init__(self, values):
            self.values = values

    a = td.CITestAdapter(ParCorrGPU(np.zeros((2, 2))))
    arr = np.arange(6.0).reshape(3, 2)
    assert a._extract_data(_DF({0: arr})) is arr
    assert a._extract_data(_DF(arr)) is arr
    assert a._extract_data(_DF([1, 2])) is None


# ---------------------------------------------------------------------------
# run_* wrappers
# ---------------------------------------------------------------------------


def test_run_pcmciplus_recovers_the_chain(chain):
    G, info = td.run_pcmciplus(chain, tau_max=2, pc_alpha=0.01)
    assert G.shape == (3, 3, 3) and G.dtype == np.int8
    assert _cross_edges(G) == {(0, 1, 1), (1, 2, 1)}
    assert info["method"] == "pcmciplus"
    assert info["p_matrix"].shape == info["val_matrix"].shape == (3, 3, 3)
    assert info["graph_raw"][0, 1, 1] == "-->"
    assert isinstance(info["ambiguous_triples"], list)


def test_run_pcmciplus_orients_a_contemporaneous_collider():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((400, 3))
    x[:, 2] += 0.8 * x[:, 0] + 0.8 * x[:, 1]  # X0 -> X2 <- X1 at lag 0
    df = pd.DataFrame(x, columns=["a", "b", "c"])
    G, info = td.run_pcmciplus(df, tau_max=1, pc_alpha=0.01)
    assert _cross_edges(G) == {(0, 2, 0), (1, 2, 0)}
    assert info["graph_raw"][2, 0, 0] == "<--"


def test_run_pcmci_with_the_adapter_matches_tigramite_parcorr(chain):
    kw = dict(tau_max=2, alpha_level=0.01)
    G_ref, info_ref = td.run_pcmci(chain, ci_test="parcorr", **kw)
    G_gpu, info_gpu = td.run_pcmci(chain, ci_test=ParCorrGPU(np.zeros((2, 2))), **kw)
    assert _cross_edges(G_ref) == _cross_edges(G_gpu) == {(0, 1, 1), (1, 2, 1)}
    # Same statistic; conditioning sets can differ slightly between the runs.
    np.testing.assert_allclose(
        info_gpu["val_matrix"], info_ref["val_matrix"], atol=0.02
    )
    assert info_ref["method"] == "pcmci" and info_ref["alpha_level"] == 0.01


def test_run_pcmci_alpha_level_thresholds_the_graph(chain):
    G, info = td.run_pcmci(chain, tau_max=2, tau_min=1, alpha_level=1e-300)
    assert G.sum() == 0
    assert info["p_matrix"][0, 1, 1] < 1e-10


def test_run_lpcmci_recovers_the_chain_and_counts_marks(chain):
    G, info = td.run_lpcmci(chain, tau_max=1, pc_alpha=0.01)
    assert {(0, 1, 1), (1, 2, 1)} <= _cross_edges(G)
    assert info["method"] == "lpcmci"
    raw = info["graph_raw"]
    assert sum(info["edge_types"].values()) == int((raw != "").sum())


def test_run_rpcmci_finds_regime_specific_links():
    pytest.importorskip("ortools")  # tigramite.rpcmci imports it
    rng = np.random.default_rng(0)
    T = 300
    x = rng.standard_normal((T, 2))
    regime = (np.arange(T) // 75) % 2  # four blocks, alternating
    for t in range(1, T):
        coef = 0.9 if regime[t] == 0 else 0.0
        x[t, 1] += coef * x[t - 1, 0]
    df = pd.DataFrame(x, columns=["a", "b"])
    G, info = td.run_rpcmci(
        df,
        tau_max=1,
        num_regimes=2,
        max_transitions=4,
        num_iterations=10,
        max_anneal=3,
        n_jobs=1,
        seed=0,
    )
    assert G.shape == (2, 2, 2)
    assert info["regime_labels"].shape == (T,)
    assert set(info["regime_graphs"]) == {0, 1}
    # The union graph has the switching edge, found in one regime only.
    assert G[0, 1, 1] == 1
    assert sum(g[0, 1, 1] for g in info["regime_graphs"].values()) == 1
    labels = info["regime_labels"]
    # Regime ids are arbitrary; the labelling matches the truth up to a swap.
    assert max(np.mean(labels == regime), np.mean(labels != regime)) > 0.9


def test_run_rpcmci_raises_when_no_annealing_converges(monkeypatch, chain):
    pytest.importorskip("ortools")  # tigramite.rpcmci imports it
    import tigramite.rpcmci

    monkeypatch.setattr(tigramite.rpcmci.RPCMCI, "run_rpcmci", lambda self, **kw: None)
    with pytest.raises(RuntimeError, match="no annealings converged"):
        td.run_rpcmci(chain, tau_max=1)


@pytest.mark.parametrize("method", ["pcmciplus", "PCMCI", "lpcmci"])
def test_run_tigramite_delegates_to_the_wrappers(chain, method):
    G, info = td.run_tigramite(chain, method=method, tau_max=1)
    assert info["method"] == method.lower()
    assert {(0, 1, 1), (1, 2, 1)} <= _cross_edges(G)


@pytest.mark.parametrize("method", ["pcalg", "fullci", "bivci"])
def test_run_tigramite_generic_methods(chain, method):
    G, info = td.run_tigramite(chain, method=method, tau_max=1)
    assert G.shape == (3, 3, 2) and info["method"] == method
    assert {(0, 1, 1), (1, 2, 1)} <= _cross_edges(G)


def test_bivci_keeps_the_indirect_link_that_fullci_conditions_away(chain):
    # X0(t-2) -> X1(t-1) -> X2(t): the bivariate test sees X0 -> X2 at lag 2,
    # full CI conditions on X1(t-1) and drops it.
    G_biv, _ = td.run_tigramite(chain, method="bivci", tau_max=2)
    G_full, _ = td.run_tigramite(chain, method="fullci", tau_max=2)
    assert (0, 2, 2) in _cross_edges(G_biv)
    assert (0, 2, 2) not in _cross_edges(G_full)


def test_run_tigramite_without_a_graph_thresholds_the_p_matrix(monkeypatch, chain):
    from tigramite.pcmci import PCMCI

    p = np.full((3, 3, 2), 0.9)
    p[0, 1, 1] = p[1, 1, 0] = 0.001  # the lag-0 self entry must be cleared
    monkeypatch.setattr(
        PCMCI,
        "run_fullci",
        lambda self, **kw: {"p_matrix": p, "val_matrix": np.zeros_like(p)},
    )
    G, info = td.run_tigramite(chain, method="fullci", tau_max=1)
    assert {(int(i), int(j), int(k)) for i, j, k in zip(*np.nonzero(G))} == {(0, 1, 1)}
    assert info["graph_raw"] is None


def test_run_tigramite_rejects_unknown_methods(chain):
    with pytest.raises(ValueError, match="Unknown method"):
        td.run_tigramite(chain, method="granger")


# ---------------------------------------------------------------------------
# selected_links -> link_assumptions (#146)
# ---------------------------------------------------------------------------


def test_run_pcmciplus_accepts_link_assumptions():
    # Issue #146 example: the "o?o at lag 0" assumptions run cleanly.
    df = pd.DataFrame(
        np.random.default_rng(0).normal(size=(300, 2)), columns=["a", "b"]
    )
    old = {0: [(1, 0), (0, -1)], 1: [(0, 0), (1, -1)]}
    link_assumptions = {
        j: {k: ("o?o" if k[1] == 0 else "-?>") for k in links}
        for j, links in old.items()
    }
    G, _ = td.run_pcmciplus(df, tau_max=1, link_assumptions=link_assumptions)
    assert G.shape == (2, 2, 2)


def test_run_pcmciplus_link_assumptions_restricts_links(chain):
    # Links missing from link_assumptions are assumed absent by tigramite,
    # so only the allowed self-lags can appear and no cross edge survives.
    link_assumptions = {j: {(j, -1): "-?>"} for j in range(3)}
    G, _ = td.run_pcmciplus(chain, tau_max=1, link_assumptions=link_assumptions)
    assert _cross_edges(G) == set()


def test_run_pcmci_link_assumptions_restricts_links(chain):
    link_assumptions = {j: {(j, -1): "-?>"} for j in range(3)}
    G, _ = td.run_pcmci(chain, tau_max=1, link_assumptions=link_assumptions)
    assert _cross_edges(G) == set()


@pytest.mark.parametrize(
    "fn", [td.run_pcmciplus, td.run_pcmci], ids=["pcmciplus", "pcmci"]
)
def test_selected_links_raises_and_points_to_link_assumptions(fn, chain):
    selected = {0: [(0, -1)], 1: [(0, -1), (1, -1)]}
    with pytest.raises(ValueError, match="link_assumptions"):
        fn(chain, tau_max=1, selected_links=selected)
    # Passing both arguments still raises for selected_links.
    with pytest.raises(ValueError, match="link_assumptions"):
        fn(
            chain,
            tau_max=1,
            selected_links=selected,
            link_assumptions={0: {(0, -1): "-?>"}},
        )


# ---------------------------------------------------------------------------
# TigramiteResult (return_result=True) and the named runners
# ---------------------------------------------------------------------------

_WRAPPERS = {
    "pcmci": td.run_pcmci,
    "pcmciplus": td.run_pcmciplus,
    "lpcmci": td.run_lpcmci,
}


@pytest.mark.parametrize("method", list(_WRAPPERS))
def test_return_result_matches_the_tuple(chain, method):
    import matplotlib

    matplotlib.use("Agg")
    from causalts.result import CausalResult
    from causalts.tigramite_result import TigramiteResult

    fn = _WRAPPERS[method]
    out = fn(chain, tau_max=1, return_result=False)
    assert isinstance(out, tuple) and len(out) == 2
    G, info = out
    res = fn(chain, tau_max=1, return_result=True)
    assert isinstance(res, TigramiteResult) and isinstance(res, CausalResult)
    np.testing.assert_array_equal(res.cg_tig, G)
    np.testing.assert_array_equal(res.to_marks(), info["graph_raw"])
    np.testing.assert_array_equal(res.p_matrix, info["p_matrix"])
    np.testing.assert_array_equal(res.val_matrix, info["val_matrix"])
    assert res.var_names == ["X0", "X1", "X2"] and res.method == method
    fig = res.plot()
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close("all")


def test_result_keeps_unoriented_marks_out_of_cg_tig():
    # Two copies of one noise series: an unorientable lag-0 link (o-o) that
    # the binary graph leaves out and the marks keep.
    rng = np.random.default_rng(0)
    x = rng.standard_normal(300)
    df = pd.DataFrame({"a": x, "b": x + 0.1 * rng.standard_normal(300)})
    res = td.run_pcmciplus(df, tau_max=1, return_result=True)
    assert res.to_marks()[0, 1, 0] == "o-o"
    assert res.cg_tig[0, 1, 0] == res.cg_tig[1, 0, 0] == 0
    assert res.to_binary(undirected="bidirected")[0, 1, 0] == 1
    # The bridge must not read cg_tig (o-o dropped) as a DAG.
    assert res.bridge_policy == "refuse"
    with pytest.raises(ValueError, match="unresolved"):
        res.estimate_effect("a", "b", treatment_lag=1)
    res.bridge_policy = "drop_unresolved"
    np.testing.assert_array_equal(res._bridge_graph(res.cg_tig), res.cg_tig)


def test_lpcmci_result_refuses_to_bridge_a_possible_cause():
    # o-> is a directed 1 in cg_tig, but only a possible cause: unresolved.
    from causalts.tigramite_result import TigramiteResult

    df = _chain(T=50)
    g = _marks({(0, 1, 1): "o->", (1, 2, 1): "-->"})
    info = {"method": "lpcmci", "graph_raw": g}
    res = TigramiteResult(td.tigramite_graph_to_binary(g), df, info)
    assert res.cg_tig[0, 1, 1] == 1
    with pytest.raises(ValueError, match="unresolved"):
        res._bridge_graph(res.cg_tig)
    res.bridge_policy = "drop_unresolved"
    bridged = res._bridge_graph(res.cg_tig)
    assert bridged[0, 1, 1] == 0 and bridged[1, 2, 1] == 1


def test_changing_bridge_policy_clears_the_scm_cache():
    # A cached SCM was fitted under the old policy; reusing it would skip the
    # refusal (or keep edges the new policy drops).
    from causalts.tigramite_result import TigramiteResult

    df = _chain(T=50)
    g = _marks({(0, 1, 1): "o->", (1, 2, 1): "-->"})
    res = TigramiteResult(td.tigramite_graph_to_binary(g), df, {"graph_raw": g})
    res.bridge_policy = "drop_unresolved"
    res._scm_cache["linear"] = object()
    res.bridge_policy = "refuse"
    assert res._scm_cache == {}
    with pytest.raises(ValueError, match="unresolved"):
        res._bridge_graph(res.cg_tig)


def test_run_tigramite_algorithm_by_name(chain):
    from causalts.algorithms import list_algorithms, run_algorithm
    from causalts.tigramite_result import TIGRAMITE_ALGORITHMS

    assert set(TIGRAMITE_ALGORITHMS) <= set(list_algorithms())
    res = run_algorithm("pcmciplus", chain, ci_test=None, max_lag=1)
    G, _ = td.run_pcmciplus(chain, tau_max=1)
    np.testing.assert_array_equal(res.cg_tig, G)


def test_run_tigramite_algorithm_alpha_maps_to_each_wrapper(chain):
    from causalts.tigramite_result import run_tigramite_algorithm

    res = run_tigramite_algorithm("pcmci", chain, 1, alpha=0.01)
    assert res.info["alpha_level"] == 0.01
    G, _ = td.run_pcmciplus(chain, tau_max=1, pc_alpha=0.2)
    res = run_tigramite_algorithm("pcmciplus", chain, 1, alpha=0.2)
    np.testing.assert_array_equal(res.cg_tig, G)
    with pytest.raises(ValueError, match="not both"):
        run_tigramite_algorithm("pcmciplus", chain, 1, alpha=0.2, pc_alpha=0.1)
    with pytest.raises(ValueError, match="unknown algorithm"):
        run_tigramite_algorithm("pcalg", chain, 1)


def test_run_tigramite_algorithm_causalts_ci_test_and_nan(chain):
    from causalts.tigramite_result import run_tigramite_algorithm

    res = run_tigramite_algorithm("pcmciplus", chain, 1, ci_test="parcorr-gpu")
    assert {(0, 1, 1), (1, 2, 1)} <= _cross_edges(res.cg_tig)
    bad = chain.copy()
    bad.iloc[3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        run_tigramite_algorithm("pcmciplus", bad, 1)


def test_discover_df_runs_tigramite(chain):
    from causalts.inspection import discover_df
    from causalts.tigramite_result import TigramiteResult

    res = discover_df(chain, algorithm="lpcmci", max_lag=1)
    assert isinstance(res, TigramiteResult)
    G, _ = td.run_lpcmci(chain, tau_max=1)
    np.testing.assert_array_equal(res.cg_tig, G)
    with pytest.raises(ValueError, match="include_C"):
        discover_df(chain, algorithm="pcmci", max_lag=1, include_C=True)


def test_graph_raw_follows_filters():
    from causalts.tigramite_result import TigramiteResult

    df = _chain(T=50)
    g = _marks({(0, 1, 1): "-->", (1, 2, 1): "-->"})
    res = TigramiteResult(td.tigramite_graph_to_binary(g), df, {"graph_raw": g})
    assert res.graph_raw is res.graph_marks
    cg = res.cg_tig.copy()
    cg[0, 1, 1] = 0
    new = res._with_graph(cg)
    assert new.graph_raw[0, 1, 1] == "" and new.to_marks()[0, 1, 1] == ""
    assert res.info["graph_raw"][0, 1, 1] == "-->"  # engine output untouched


def test_discrete_column_ci_tests_are_rejected():
    from causalts.ci_tests import create_ci_test
    from causalts.inspection import discover_df
    from causalts.tigramite_result import run_tigramite_algorithm

    df = _chain(T=200)
    df["D"] = np.random.default_rng(1).integers(0, 3, len(df)).astype(float)
    ci = create_ci_test("cmiknn-mixed-gpu", df.to_numpy(), device="cpu")
    assert ci.discrete_cols
    with pytest.raises(ValueError, match="cmiknn-mixed-gpu with cdnots"):
        run_tigramite_algorithm("pcmciplus", df, 1, ci_test=ci)
    with pytest.raises(ValueError, match="cmiknn-mixed-gpu with cdnots"):
        discover_df(df, algorithm="pcmci", max_lag=1, ci_test="cmiknn-mixed-gpu")


@pytest.mark.parametrize(
    "algorithm, kw, expected",
    [
        ("pcmci", {"alpha": 0.001}, 0.001),  # alpha_level; pc_alpha 0.2
        ("pcmci", {}, 0.05),  # defaults: min(pc_alpha 0.2, alpha_level 0.05)
        ("pcmci", {"alpha": 0.1, "pc_alpha": 0.02}, 0.02),
        ("pcmciplus", {}, 0.01),
        ("pcmciplus", {"pc_alpha": [0.05, 0.002]}, 0.002),
        ("lpcmci", {"alpha": 0.001}, 0.001),
    ],
)
def test_dfcit_gets_the_runs_smallest_alpha(
    monkeypatch, chain, algorithm, kw, expected
):
    import causalts.ci_tests as ci_tests
    from causalts.tigramite_result import run_tigramite_algorithm

    seen = {}

    class _Stop(Exception):
        pass

    def fake_create(name, data, **ci_kw):
        seen.update(name=name, **ci_kw)
        raise _Stop

    monkeypatch.setattr(ci_tests, "create_ci_test", fake_create)
    with pytest.raises(_Stop):
        run_tigramite_algorithm(algorithm, chain, 1, ci_test="dfcit", **kw)
    assert seen["name"] == "dfcit" and seen["alpha"] == pytest.approx(expected)
    # Other causal-ts tests are built without an alpha.
    seen.clear()
    with pytest.raises(_Stop):
        run_tigramite_algorithm(algorithm, chain, 1, ci_test="parcorr-gpu", **kw)
    assert "alpha" not in seen


def test_causalts_ci_test_gets_ci_kwargs(monkeypatch, chain):
    import causalts.ci_tests as ci_tests
    from causalts.tigramite_result import run_tigramite_algorithm

    seen = {}

    class _Stop(Exception):
        pass

    def fake_create(name, data, **ci_kw):
        seen.clear()
        seen.update(ci_kw)
        raise _Stop

    monkeypatch.setattr(ci_tests, "create_ci_test", fake_create)
    with pytest.raises(_Stop):
        run_tigramite_algorithm(
            "pcmciplus", chain, 1, ci_test="dfcit", ci_kwargs={"seed": 7, "n_perm": 9}
        )
    assert seen["seed"] == 7 and seen["n_perm"] == 9 and seen["alpha"] == 0.01
    with pytest.raises(_Stop):  # an explicit ci_kwargs alpha wins
        run_tigramite_algorithm(
            "pcmciplus", chain, 1, ci_test="dfcit", ci_kwargs={"alpha": 0.02}
        )
    assert seen["alpha"] == 0.02


@pytest.mark.parametrize("algorithm, expected", [("pcmciplus", 0.001), ("pcmci", 0.05)])
def test_min_threshold_counts_the_automatic_pc_alpha_grid(algorithm, expected):
    from causalts.tigramite_result import _TIGRAMITE_METHODS, _min_threshold

    fn, key = _TIGRAMITE_METHODS[algorithm]
    got = _min_threshold(td, fn, algorithm, key, None, {"pc_alpha": None})
    assert got == pytest.approx(expected)


def test_cmiknn_mixed_is_refused(chain):
    from causalts.tigramite_result import run_tigramite_algorithm

    with pytest.raises(ValueError, match="data-type mask"):
        run_tigramite_algorithm("pcmciplus", chain, 1, ci_test="cmiknn_mixed")


def test_filters_refuse_unresolved_marks_and_run_on_definite_ones():
    from causalts.tigramite_result import TigramiteResult

    df = _chain(T=200)
    g = _marks({(0, 1, 1): "o->", (1, 2, 1): "-->"})
    res = TigramiteResult(td.tigramite_graph_to_binary(g), df, {"graph_raw": g})
    for call in (res.tetrad_filter, res.pds_filter, res.deconfound):
        with pytest.raises(ValueError, match=r"such marks \(o->"):
            call()
    g = _marks({(0, 1, 1): "-->", (1, 2, 1): "-->"})
    res = TigramiteResult(td.tigramite_graph_to_binary(g), df, {"graph_raw": g})
    out = res.pds_filter()
    assert isinstance(out, TigramiteResult)
    assert out.cg_tig.shape == res.cg_tig.shape


@pytest.mark.parametrize(
    "algorithm, kw, expected",
    [
        ("lpcmci", {"alpha": np.float32(0.001)}, 0.001),
        ("pcmciplus", {"pc_alpha": np.array([0.05, 0.002], dtype=np.float32)}, 0.002),
        ("pcmciplus", {"pc_alpha": [float("nan"), 0.003]}, 0.003),
    ],
)
def test_min_threshold_takes_numpy_floats_and_skips_nan(algorithm, kw, expected):
    from causalts.tigramite_result import _TIGRAMITE_METHODS, _min_threshold

    fn, key = _TIGRAMITE_METHODS[algorithm]
    alpha = kw.pop("alpha", None)
    got = _min_threshold(td, fn, algorithm, key, alpha, kw)
    assert got == pytest.approx(expected, rel=1e-6)


def test_gpdc_torch_without_gpytorch_names_it(monkeypatch, chain):
    import importlib.util

    from causalts.tigramite_result import run_tigramite_algorithm

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name == "gpytorch" else real(name, *a),
    )
    with pytest.raises(ImportError, match="pip install gpytorch") as exc:
        run_tigramite_algorithm("pcmciplus", chain, 1, ci_test="gpdc_torch")
    assert exc.value.name == "gpytorch"


def test_discover_df_warns_when_a_name_means_tigramites_test(chain):
    import warnings

    from causalts.inspection import discover_df

    with pytest.warns(UserWarning, match="tigramite's own parcorr"):
        discover_df(chain, algorithm="pcmciplus", max_lag=1, ci_test="parcorr")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # tigramite-only name: nothing to confuse it with.
        discover_df(chain, algorithm="pcmciplus", max_lag=1, ci_test="robust_parcorr")


def test_filters_refuse_endpoint_conflict_marks():
    # LPCMCI can emit x-> (an endpoint conflict); cg_tig drops it, so the
    # post-processors must refuse it like o-o / o-> / <->.
    from causalts.tigramite_result import TigramiteResult

    df = _chain(T=200)
    g = _marks({(0, 1, 1): "x->", (1, 2, 1): "-->"})
    res = TigramiteResult(td.tigramite_graph_to_binary(g), df, {"graph_raw": g})
    for call in (res.tetrad_filter, res.pds_filter, res.deconfound):
        with pytest.raises(ValueError, match=r"such marks \(x->"):
            call()


def test_filters_let_lag0_o_o_through_with_a_warning():
    # PCMCI+-style result: a lag-0 o-o pair plus definite lagged arrows. The
    # filters run, warn the pair is not tested, and keep it.
    from causalts.tigramite_result import TigramiteResult

    df = _chain(T=200)
    g = _marks({(0, 1, 1): "-->", (1, 2, 0): "o-o", (2, 1, 0): "o-o"})
    res = TigramiteResult(td.tigramite_graph_to_binary(g), df, {"graph_raw": g})
    assert res.cg_tig[1, 2, 0] == res.cg_tig[2, 1, 0] == 0
    for name in ("tetrad_filter", "pds_filter"):
        with pytest.warns(UserWarning, match="unresolved adjacencies"):
            out = getattr(res, name)()
        assert isinstance(out, TigramiteResult)
        assert out.to_marks()[1, 2, 0] == out.to_marks()[2, 1, 0] == "o-o"
    with pytest.warns(UserWarning, match="unresolved adjacencies"):
        res.deconfound()
    # A lagged o-o is PAG-specific: still refused.
    g = _marks({(0, 1, 1): "o-o"})
    res = TigramiteResult(td.tigramite_graph_to_binary(g), df, {"graph_raw": g})
    with pytest.raises(ValueError, match=r"such marks \(o-o"):
        res.pds_filter()


def _chain_with_gaps(T=300, frac=0.05, seed=2):
    df = _chain(T=T)
    rng = np.random.default_rng(seed)
    mask = rng.random(df.shape) < frac
    mask[:5] = False
    return df.mask(mask)


@pytest.mark.parametrize("algorithm", ["pcmci", "pcmciplus", "lpcmci"])
def test_discover_df_var_em_matches_filling_first(algorithm):
    from causalts.imputation import var_em_impute
    from causalts.inspection import discover_df
    from causalts.tigramite_result import TigramiteResult

    df = _chain_with_gaps()
    res = discover_df(df, algorithm=algorithm, max_lag=1, impute="var_em")
    assert isinstance(res, TigramiteResult)
    assert not res._df.isna().any().any()  # the data discovery ran on
    assert res.info["impute"] == "var_em"
    G, _ = getattr(td, f"run_{algorithm}")(var_em_impute(df), tau_max=1)
    np.testing.assert_array_equal(res.cg_tig, G)


@pytest.mark.parametrize("algorithm", ["pcmci", "pcmciplus", "lpcmci"])
def test_tigramite_algorithms_refuse_missing_values_and_other_imputers(algorithm):
    from causalts.inspection import discover_df

    df = _chain_with_gaps()
    with pytest.raises(ValueError, match="NaN values. Pass impute='var_em'"):
        discover_df(df, algorithm=algorithm, max_lag=1)
    with pytest.raises(ValueError, match="impute='var_em' only"):
        discover_df(df, algorithm=algorithm, max_lag=1, impute="pairwise_complete")
    with pytest.raises(ValueError, match="impute_kwargs given without"):
        discover_df(df, algorithm=algorithm, max_lag=1, impute_kwargs={"p": 2})


@pytest.mark.parametrize("algorithm", ["pcmci", "pcmciplus", "lpcmci"])
def test_var_em_impute_kwargs_are_checked(algorithm):
    from causalts.inspection import discover_df

    df = _chain_with_gaps()
    with pytest.raises(ValueError, match=r"takes impute_kwargs .*unknown \['m'\]"):
        discover_df(
            df, algorithm=algorithm, max_lag=1, impute="var_em", impute_kwargs={"m": 5}
        )


def test_gaps_in_the_results_data_block_the_bridge_and_filters():
    from causalts.tigramite_result import TigramiteResult

    df = _chain_with_gaps()
    g = _marks({(0, 1, 1): "-->", (1, 2, 1): "-->"})
    res = TigramiteResult(td.tigramite_graph_to_binary(g), df, {"graph_raw": g})
    with pytest.raises(ValueError, match="needs complete data"):
        res.estimate_effect("X0", "X1", treatment_lag=1)
    with pytest.raises(ValueError, match="needs complete data"):
        res.pds_filter()
