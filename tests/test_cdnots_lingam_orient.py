import numpy as np
import pandas as pd
import pytest

from causalts.cdnots.lingam_orient import direct_lingam_order, lingam_orient_lag0
from causalts.cdnots.phase3_utils import (
    _can_orient,
    _orient,
    finish_cdnots_plus,
    run_cdnots_plus,
)
from causalts.cdnots.skeleton_discovery import initialize_graph
from causalts.ci_tests.parcorr_gpu import ParCorrGPU


def _pair(noise, T=1000, seed=0):
    """X0 -> X1 at lag 0 only, no autocorrelation: CI cannot orient it."""
    rng = np.random.default_rng(seed)
    draw = {
        "uniform": lambda: rng.uniform(-1, 1, T),
        "gaussian": lambda: rng.normal(size=T),
    }[noise]
    x0 = draw()
    x1 = 0.8 * x0 / x0.std() + draw()
    return pd.DataFrame({"X0": x0, "X1": x1})


def _run(df, **kw):
    return run_cdnots_plus(
        df,
        ParCorrGPU(df.values, device="cpu"),
        num_lags=1,
        include_C=False,
        alpha=0.01,
        **kw,
    )


def test_direct_lingam_order_finds_the_exogenous_variable():
    rng = np.random.default_rng(1)
    e = rng.uniform(-1, 1, (2000, 3))
    x0 = e[:, 0]
    x1 = 0.9 * x0 + e[:, 1]
    x2 = 0.7 * x1 + e[:, 2]
    assert direct_lingam_order(np.column_stack([x2, x0, x1])) == [1, 2, 0]


def test_default_is_unchanged_and_records_nothing():
    df = _pair("uniform")
    base = _run(df)
    off = _run(df, lag0_orient=None)
    np.testing.assert_array_equal(np.asarray(base.cg_tig), np.asarray(off.cg_tig))
    assert off.lag0_orientations is None
    # without the step the lag-0 edge is unoriented, so CDNOTS+ drops it
    assert base.to_marks()[0, 1, 0] == "o-o"
    assert np.asarray(base.cg_tig)[:2, :2, 0].sum() == 0


def test_non_gaussian_lag0_edge_is_oriented():
    res = _run(_pair("uniform"), lag0_orient="lingam")
    G = np.asarray(res.cg_tig)
    assert G[0, 1, 0] == 1 and G[1, 0, 0] == 0
    assert res.to_marks()[0, 1, 0] == "-->"
    (dec,) = res.lag0_orientations
    assert (dec["cause"], dec["effect"], dec["oriented"]) == (0, 1, True)


def test_gaussian_lag0_edge_is_left_alone():
    df = _pair("gaussian")
    res = _run(df, lag0_orient="lingam")
    np.testing.assert_array_equal(np.asarray(res.cg_tig), np.asarray(_run(df).cg_tig))
    assert not any(d["oriented"] for d in res.lag0_orientations)


def test_discrete_columns_are_never_oriented():
    df = _pair("uniform")
    df["X1"] = (df["X1"] > 0).astype(int)
    res = _run(df, lag0_orient="lingam", discrete_cols=["X1"])
    assert res.lag0_orientations == []


def test_rejects_unknown_method():
    with pytest.raises(ValueError, match="lag0_orient"):
        _run(_pair("uniform"), lag0_orient="ica")


def test_one_prefix_finishes_with_and_without_the_step():
    df = _pair("uniform")
    prefix = _run(df, return_prefix=True)
    on = finish_cdnots_plus(prefix, lag0_orient="lingam")
    direct = _run(df, lag0_orient="lingam")
    np.testing.assert_array_equal(np.asarray(on.cg_tig), np.asarray(direct.cg_tig))
    assert on.lag0_orientations == direct.lag0_orientations
    # the stored value (None) is the default, and the prefix is not mutated
    off = finish_cdnots_plus(prefix)
    assert off.lag0_orientations is None
    np.testing.assert_array_equal(np.asarray(off.cg_tig), np.asarray(_run(df).cg_tig))
    with pytest.raises(ValueError, match="lag0_orient"):
        finish_cdnots_plus(prefix, lag0_orient="LiNGAM")


def test_gaussian_data_with_a_trend_is_left_alone_under_include_c():
    """Do-no-harm with a C node: Gaussian noise plus a shared trend stays unoriented."""
    rng = np.random.default_rng(3)
    T = 1000
    trend = np.linspace(-3, 3, T)
    x0 = trend + rng.normal(size=T)
    x1 = 0.8 * x0 + trend + rng.normal(size=T)
    df = pd.DataFrame({"X0": x0, "X1": x1})
    res = run_cdnots_plus(
        df,
        ParCorrGPU(df.values, device="cpu"),
        num_lags=1,
        include_C=True,
        alpha=0.01,
        lag0_orient="lingam",
    )
    assert not any(d["oriented"] for d in res.lag0_orientations)


def _graph(n, undirected=(), directed=()):
    cg = initialize_graph(np.zeros((20, n)), None)
    keep = {frozenset(e) for e in list(undirected) + list(directed)}
    for i in range(n):
        for j in range(i + 1, n):
            edge = cg.G.get_edge(cg.G.nodes[i], cg.G.nodes[j])
            if edge is not None and frozenset((i, j)) not in keep:
                cg.G.remove_edge(edge)
    for a, b in directed:
        cg.G.remove_edge(cg.G.get_edge(cg.G.nodes[a], cg.G.nodes[b]))
        cg.G.add_directed_edge(cg.G.nodes[a], cg.G.nodes[b])
    return cg


def _orient_step(cg, data, num_lags=0, c_start=None):
    return lingam_orient_lag0(
        cg,
        data,
        num_lags,
        c_start=cg.G.num_vars // (num_lags + 1) if c_start is None else c_start,
        can_orient=lambda g, x, y: _can_orient(g, x, y, None, None),
        orient=_orient,
    )


def _uniform_pair(T=1000, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1, 1, T)
    return np.column_stack([x0, 0.8 * x0 / x0.std() + rng.uniform(-1, 1, T)])


def test_orientation_refreshes_the_ancestry_cache():
    res = _run(_pair("uniform"), lag0_orient="lingam")
    nodes = res.graph.G.nodes
    assert res.graph.G.is_ancestor_of(nodes[0], nodes[1])


def test_conflicting_lag_copies_are_resolved_with_the_edge():
    # Two variables, one lag: nodes 0,1 (lag 0) and 2,3 (lag 1); x-x on both copies.
    cg = _graph(4, undirected=[(0, 1), (2, 3)])
    g = cg.G.graph
    g[0, 1] = g[1, 0] = g[2, 3] = g[3, 2] = 1
    x = _uniform_pair()
    data = np.column_stack(
        [x[1:], x[:-1]]
    )  # lag-embedded: [X0(t), X1(t), X0(t-1), X1(t-1)]
    (dec,) = _orient_step(cg, data, num_lags=1)
    assert dec["oriented"] and (dec["cause"], dec["effect"]) == (0, 1)
    assert (g[0, 1], g[1, 0]) == (-1, 1) and (g[2, 3], g[3, 2]) == (-1, 1)


def test_exactly_explained_endpoint_abstains():
    # X1 is an exact copy of its directed parent X2: no innovation to order.
    x = _uniform_pair()
    data = np.column_stack([x[:, 0], x[:, 1], x[:, 1]])
    cg = _graph(3, undirected=[(0, 1)], directed=[(2, 1)])
    (dec,) = _orient_step(cg, data)
    assert not dec["oriented"] and dec["agreement"] is None
    assert cg.G.graph[0, 1] == cg.G.graph[1, 0] == -1


def test_too_few_complete_rows_still_records_the_pair():
    data = _uniform_pair(T=20)
    cg = _graph(2, undirected=[(0, 1)])
    (dec,) = _orient_step(cg, data)
    assert not dec["oriented"] and dec["normality_p"] is None


def test_constant_endpoint_abstains():
    # X1 never varies; its least-squares residual on a varying parent is
    # rounding noise, which must not read as a strongly non-Gaussian innovation.
    x = _uniform_pair()
    data = np.column_stack([x[:, 0], np.full(len(x), 3.0), x[:, 1]])
    cg = _graph(3, undirected=[(0, 1)], directed=[(2, 1)])
    (dec,) = _orient_step(cg, data)
    assert not dec["oriented"] and dec["agreement"] is None


def _collider_case(seed=0, T=1000):
    """X0 -> X1 <- X2 with uniform noise; X2 -> X1 already directed."""
    rng = np.random.default_rng(seed)
    x0, x2 = rng.uniform(-1, 1, T), rng.uniform(-1, 1, T)
    x1 = 0.8 * x0 + 0.8 * x2 + rng.uniform(-1, 1, T)
    return np.column_stack([x0, x1, x2])


def test_no_collider_the_ci_step_ruled_out():
    cg = _graph(3, undirected=[(0, 1)], directed=[(2, 1)])
    (dec,) = _orient_step(cg, _collider_case())
    assert not dec["oriented"]  # 0 -> 1 <- 2 would contradict a non-collider


def test_ambiguous_triple_may_be_decided_by_the_data():
    cg = _graph(3, undirected=[(0, 1)], directed=[(2, 1)])
    cg.ambiguous_triples = {(0, 1, 2), (2, 1, 0)}
    (dec,) = _orient_step(cg, _collider_case())
    assert dec["oriented"] and (dec["cause"], dec["effect"]) == (0, 1)


def test_component_touching_an_excluded_variable_abstains():
    x = _uniform_pair()
    data = np.column_stack([x, (x[:, 1] > 0).astype(float)])
    cg = _graph(3, undirected=[(0, 1), (1, 2)])
    decs = lingam_orient_lag0(
        cg,
        data,
        0,
        c_start=3,
        can_orient=lambda g, a, b: _can_orient(g, a, b, None, None),
        orient=_orient,
        exclude_vars=[2],
    )
    assert [(d["oriented"], d["agreement"]) for d in decs] == [(False, None)]


def test_meek_runs_again_after_new_orientations(monkeypatch):
    import causalts.cdnots.phase3_utils as p3

    calls = []
    real = p3.meek

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(p3, "meek", spy)
    _run(_pair("uniform"))
    base = len(calls)
    _run(_pair("uniform"), lag0_orient="lingam")
    assert len(calls) - base == base + 1


def _two_parent_case(seed=0, T=1000):
    """X0 -> X1 <- X2, X3 -> X1, uniform noise."""
    rng = np.random.default_rng(seed)
    x0, x2, x3 = (rng.uniform(-1, 1, T) for _ in range(3))
    x1 = 0.8 * x0 + 0.8 * x2 + 0.8 * x3 + rng.uniform(-1, 1, T)
    return np.column_stack([x0, x1, x2, x3])


@pytest.mark.parametrize(
    "colliders, oriented", [(set(), False), ({2}, False), ({2, 3}, True)]
)
def test_conflicting_edge_respects_each_non_collider(colliders, oriented):
    # 0 x-x 1 with 2 -> 1 and 3 -> 1: only the triples the vote declared
    # colliders let 0 -> 1 through.
    cg = _graph(4, undirected=[(0, 1)], directed=[(2, 1), (3, 1)])
    cg.G.graph[0, 1] = cg.G.graph[1, 0] = 1
    cg.collider_triples = {t for z in colliders for t in ((z, 1, 0), (0, 1, z))}
    (dec,) = _orient_step(cg, _two_parent_case())
    assert dec["oriented"] is oriented
    if not oriented:  # put back as it was
        assert cg.G.graph[0, 1] == cg.G.graph[1, 0] == 1


def test_graph_without_a_collider_record_skips_the_check_for_conflicting_edges():
    # E.g. a prefix cached before collider_triples existed: an empty record
    # would block 0 -> 1 above; a missing one falls back to skipping, loudly.
    cg = _graph(4, undirected=[(0, 1)], directed=[(2, 1), (3, 1)])
    cg.G.graph[0, 1] = cg.G.graph[1, 0] = 1
    assert not hasattr(cg, "collider_triples")
    with pytest.warns(RuntimeWarning, match="collider_triples"):
        (dec,) = _orient_step(cg, _two_parent_case())
    assert dec["oriented"]


def test_uc_sepset_records_the_triples_it_declared_colliders():
    df = pd.DataFrame(_collider_case(), columns=["X0", "X1", "X2"])
    state = _run(df, return_prefix=True)
    assert {(0, 1, 2), (2, 1, 0)} <= state.cg_sepset.collider_triples


def test_two_undirected_edges_into_one_node_do_not_both_orient():
    # 0 - 1 - 2, both o-o, 0 and 2 not adjacent: a non-collider for the CI
    # step. Data make 1 last in the order, so each edge alone would point at 1.
    cg = _graph(3, undirected=[(0, 1), (1, 2)])
    decs = _orient_step(cg, _collider_case())
    g = cg.G.graph
    into_1 = [z for z in (0, 2) if g[z, 1] == -1 and g[1, z] == 1]
    assert len(into_1) <= 1 and sum(d["oriented"] for d in decs) <= 1


def test_decisions_name_their_variables_and_the_field_is_declared():
    res = _run(_pair("uniform"), lag0_orient="lingam")
    (dec,) = res.lag0_orientations
    assert (dec["cause_name"], dec["effect_name"]) == ("X0", "X1")
    from causalts.cdnots.phase3_utils import run_cdnots

    df = _pair("uniform")
    plain = run_cdnots(df, ParCorrGPU(df.values, device="cpu"), num_lags=1, alpha=0.01)
    assert plain.lag0_orientations is None


def test_lag_copies_follow_the_lag0_decision():
    # One lag: X0..X2 at lag 0 (0-2), lag 1 (3-5). 2 -> 1 and its copy 5 -> 4;
    # 0 - 1 and its copy 3 - 4 undirected; the lag-0 triple is ambiguous.
    cg = _graph(6, undirected=[(0, 1), (3, 4)], directed=[(2, 1), (5, 4)])
    cg.ambiguous_triples = {(0, 1, 2), (2, 1, 0)}
    x = _collider_case()
    data = np.column_stack([x[1:], x[:-1]])
    (dec,) = _orient_step(cg, data, num_lags=1)
    g = cg.G.graph
    assert dec["oriented"]
    assert (g[0, 1], g[1, 0]) == (-1, 1) and (g[3, 4], g[4, 3]) == (-1, 1)


def test_meek_conflict_edge_is_exempt_and_cleared_once_oriented():
    # 0 - 1 left undirected as an R1 conflict; 2 -> 1 would otherwise block it.
    cg = _graph(3, undirected=[(0, 1)], directed=[(2, 1)])
    cg.conflict_edges = {(0, 1), (1, 0)}
    (dec,) = _orient_step(cg, _collider_case())
    assert dec["oriented"] and (dec["cause"], dec["effect"]) == (0, 1)
    assert not cg.conflict_edges


def test_gaussian_component_skips_the_bootstrap(monkeypatch):
    import causalts.cdnots.lingam_orient as lo

    calls = []
    real = lo.direct_lingam_order
    monkeypatch.setattr(lo, "direct_lingam_order", lambda X: calls.append(1) or real(X))
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=2000)
    data = np.column_stack([x0, 0.8 * x0 + rng.normal(size=2000)])
    cg = _graph(2, undirected=[(0, 1)])
    (dec,) = _orient_step(cg, data)
    assert calls == [] and dec["agreement"] is None and dec["normality_p"] >= 0.01


def test_more_confident_edge_wins_the_collider_check(monkeypatch):
    # 0 - 1 - 2 non-collider. Full order [0, 2, 1] points both edges at 1; 5 of
    # 100 bootstraps say [2, 1, 0], so 0 -> 1 has agreement 0.95, 2 -> 1 has 1.0.
    import causalts.cdnots.lingam_orient as lo

    orders = iter([[0, 2, 1]] + [[0, 2, 1]] * 95 + [[2, 1, 0]] * 5)
    monkeypatch.setattr(lo, "direct_lingam_order", lambda X: next(orders))
    cg = _graph(3, undirected=[(0, 1), (1, 2)])
    decs = _orient_step(cg, _collider_case())
    by_pair = {(d["cause"], d["effect"]): d for d in decs}
    assert by_pair[(2, 1)]["oriented"] and not by_pair[(0, 1)]["oriented"]
    assert [(d["cause"], d["effect"]) for d in decs] == [(0, 1), (2, 1)]  # pair order


def test_meek_conflict_edge_is_checked_against_arrowheads_written_later(monkeypatch):
    # 0 - 1 is an R1 conflict, 1 - 2 plain o-o, 0 and 2 not adjacent. 2 -> 1
    # is more confident and goes first; 0 -> 1 would then make a collider
    # the vote ruled out, and 2 -> 1 played no part in the R1 conflict.
    import causalts.cdnots.lingam_orient as lo

    orders = iter([[0, 2, 1]] + [[0, 2, 1]] * 95 + [[2, 1, 0]] * 5)
    monkeypatch.setattr(lo, "direct_lingam_order", lambda X: next(orders))
    cg = _graph(3, undirected=[(0, 1), (1, 2)])
    cg.conflict_edges = {(0, 1), (1, 0)}
    decs = _orient_step(cg, _collider_case())
    by_pair = {(d["cause"], d["effect"]): d for d in decs}
    assert by_pair[(2, 1)]["oriented"] and not by_pair[(0, 1)]["oriented"]


def test_collinear_innovations_abstain():
    rng = np.random.default_rng(0)
    x0, x1 = rng.uniform(-1, 1, 1000), rng.uniform(-1, 1, 1000)
    data = np.column_stack([x0, x1, x0 + x1])  # an exact identity
    cg = _graph(3, undirected=[(0, 2), (1, 2)])
    decs = _orient_step(cg, data)
    assert all(not d["oriented"] and d["agreement"] is None for d in decs)
