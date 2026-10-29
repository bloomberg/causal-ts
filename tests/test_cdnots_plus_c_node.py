# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""CDNOTS+ must test the lag-0 edges of the time-index node C.

Phase 1 of CDNOTS+ screened only lagged candidate parents and phase 2 (MCI)
skipped C nodes, so no lag-0 C - X edge was ever tested: with
``include_C=True`` every variable came out C-adjacent, i.e. flagged as
nonstationary, even on stationary data.

Also covers other C-node handling: background knowledge with several C
columns, a forbidden C -> X edge, lag verification and feature selection.
"""

import numpy as np
import pandas as pd
import pytest
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

from causalts.cdnots.ancestral import AncestralKnowledge
from causalts.cdnots.phase3_utils import (
    cdnots_discovery,
    make_background_knowledge,
    run_cdnots,
    run_cdnots_plus,
)
from causalts.cdnots.result import CdnotsResult
from causalts.cdnots.skeleton_discovery import (
    c_lag_cnst,
    initialize_graph,
    lag_verification_pass,
    mci_skeleton,
    skeleton_cnst,
    skeleton_discovery_pervar,
)
from causalts.cdnots.uc_sepset import uc_sepset
from causalts.ci_tests.parcorr_gpu import ParCorrGPU
from causalts.feature_selection import select_features


def _stationary_scp(d, T, seed, max_lag=1, contemp_fraction=0.0):
    """Stationary linear-Gaussian SCP: random lagged (and lag-0) edges, no trend."""
    rng = np.random.default_rng(seed)
    lagged = np.zeros((d, d))
    if max_lag >= 1:
        mask = rng.random((d, d)) < 1.5 / d
        lagged[mask] = rng.choice([-1, 1], mask.sum()) * rng.uniform(
            0.2, 0.4, mask.sum()
        )
        np.fill_diagonal(lagged, rng.uniform(0.2, 0.5, d))
        rho = np.abs(np.linalg.eigvals(lagged)).max()
        if rho >= 0.9:
            lagged *= 0.9 / rho
    contemp = np.zeros((d, d))  # contemp[i, j]: X_i(t) -> X_j(t), i < j
    upper = np.triu(rng.random((d, d)) < contemp_fraction, k=1)
    contemp[upper] = rng.choice([-1, 1], upper.sum()) * rng.uniform(
        0.4, 0.8, upper.sum()
    )
    x = np.zeros((T + 100, d))
    for t in range(1, T + 100):
        x[t] = x[t - 1] @ lagged + rng.standard_normal(d)
        for j in range(d):
            x[t, j] += x[t, :j] @ contemp[:j, j]
    return {"df": pd.DataFrame(x[100:], columns=[f"X{i}" for i in range(d)])}


def _c_adjacent(res):
    """Boolean per variable: adjacent to (any) C node at any lag in cg_tig."""
    cg = np.asarray(res.cg_tig)
    d = len(res.var_names)
    c = slice(d, cg.shape[0])
    return ((cg[c, :d, :] != 0).any(axis=(0, 2))) | (
        (cg[:d, c, :] != 0).any(axis=(1, 2))
    )


def _run(fn, df, num_lags):
    return fn(
        df,
        ParCorrGPU(np.zeros((2, 2)), device="cpu"),
        num_lags=num_lags,
        include_C=True,
        alpha=0.05,
        show_progress=False,
    )


@pytest.mark.parametrize("num_lags", [0, 1])
@pytest.mark.parametrize("fn", [run_cdnots, run_cdnots_plus])
def test_c_is_rarely_adjacent_on_stationary_data(fn, num_lags):
    flagged = 0
    for seed in range(5):
        if num_lags == 0:
            b = _stationary_scp(5, 500, seed, max_lag=0, contemp_fraction=0.5)
        else:
            b = _stationary_scp(5, 500, seed, max_lag=1)
        flagged += int(_c_adjacent(_run(fn, b["df"], num_lags)).sum())
    # 5 variables x 5 seeds at alpha = 0.05: a handful of false positives at
    # most. CDNOTS+ used to flag all 25.
    assert flagged <= 3, flagged


def _trend_root(kind, T=600, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, 4))
    trend = np.linspace(0, 3, T)
    for t in range(1, T):
        x[t, 0] += 0.4 * x[t - 1, 0] + trend[t]
        x[t, 2] += 0.5 * x[t - 1, 2]
        x[t, 3] += 0.5 * x[t, 2]
        parent = x[t, 0] if kind == "lag0" else x[t - 1, 0]
        x[t, 1] += 0.4 * x[t - 1, 1] + 0.8 * parent
    return pd.DataFrame(x, columns=[f"X{i}" for i in range(4)])


@pytest.mark.parametrize("kind", ["lag0", "lag1"])
@pytest.mark.parametrize("fn", [run_cdnots, run_cdnots_plus])
def test_c_finds_the_nonstationary_root_and_not_its_descendant(fn, kind):
    # X0 carries a trend; X1 inherits it only through X0 (at lag 0 or 1);
    # X2 -> X3 is stationary. C must be adjacent to X0, not to the others.
    hits = np.zeros(4, int)
    for seed in range(3):
        hits += _c_adjacent(_run(fn, _trend_root(kind, seed=seed), num_lags=1))
    assert hits[0] == 3
    assert hits[1] <= 1 and hits[2] == 0 and hits[3] == 0, hits.tolist()


def test_phase_one_tests_the_lag0_c_node_as_a_candidate_parent():
    # Window for 2 variables + 1 C column, 1 lag: x0, x1, C at 0-2; lagged
    # copies at 3-5. C is never a child, so it is screened with the past.
    data = np.random.default_rng(0).standard_normal((200, 6))
    cg = skeleton_cnst(
        initialize_graph(data, None), data, 1, include_C=True, num_c_cols=1
    )
    calls = []

    def oracle(x, y, S):
        calls.append((int(x), int(y)))
        return (0.9, 0.0) if 2 in (x, y) else (0.0, 9.0)

    out = skeleton_discovery_pervar(
        cg,
        alpha=0.05,
        ci_test=oracle,
        num_lags=1,
        show_progress=False,
        include_C=True,
        num_c_cols=1,
    )
    assert (2, 0) in calls and (2, 1) in calls
    assert 2 not in out.neighbors(0) and 2 not in out.neighbors(1)


def test_phase_one_defaults_leave_c_untested():
    # Direct callers that do not pass include_C keep the old behaviour.
    data = np.random.default_rng(0).standard_normal((200, 6))
    cg = skeleton_cnst(
        initialize_graph(data, None), data, 1, include_C=True, num_c_cols=1
    )
    calls = []

    def oracle(x, y, S):
        calls.append((int(x), int(y)))
        return 0.0, 9.0

    skeleton_discovery_pervar(
        cg, alpha=0.05, ci_test=oracle, num_lags=1, show_progress=False
    )
    assert not any(2 in pair for pair in calls)


def test_multi_column_c_preset_is_tested_too():
    # Two C columns (linear + sine): both lag-0 C nodes must be screened.
    flagged = 0
    for seed in range(3):
        b = _stationary_scp(5, 500, seed, max_lag=1)
        res = run_cdnots_plus(
            b["df"],
            ParCorrGPU(np.zeros((2, 2)), device="cpu"),
            num_lags=1,
            include_C=True,
            c_preset="linear+sin",
            alpha=0.05,
            show_progress=False,
        )
        flagged += int(_c_adjacent(res).sum())
    assert flagged <= 3, flagged


def _keep_only(cg, wanted):
    wanted = {frozenset(e) for e in wanted}
    n = cg.G.num_vars
    for i in range(n):
        for j in range(i + 1, n):
            edge = cg.G.get_edge(cg.G.nodes[i], cg.G.nodes[j])
            if edge is not None and frozenset((i, j)) not in wanted:
                cg.G.remove_edge(edge)


def test_mci_keeps_a_pruned_lagged_parent_in_the_base_set():
    # 3 variables, 1 lag (lagged copies 3-5). Lagged parent 3 of target 0 is
    # pruned at depth 0; parent 4 is then tested at depth 1 and, as in PCMCI+,
    # the base set fixed by phase 1 must still contain 3.
    cg = initialize_graph(np.zeros((20, 6)), None)
    _keep_only(cg, [(0, 1), (0, 3), (0, 4)])
    calls = []

    def oracle(x, y, Z):
        calls.append((int(x), int(y), tuple(Z)))
        return (0.9, 0.0) if (x, y) == (3, 0) else (0.0, 1.0)

    mci_skeleton(
        cg,
        oracle,
        alpha=0.05,
        lagged_parents={0: [3, 4], 1: [], 2: []},
        num_lags=1,
        include_C=False,
        show_progress=False,
    )
    depth1 = [Z for x, y, Z in calls if (x, y) == (4, 0) and 1 in Z]
    assert depth1 and all(3 in Z for Z in depth1), calls


def test_mci_drops_a_pruned_c_parent_from_the_base_set():
    # 3 variables + 1 C column, 1 lag: x0-x2, C at 0-3. C -> x0 is pruned at
    # depth 0, so the depth-1 tests of x0's other parents must no longer
    # condition on C.
    cg = initialize_graph(np.zeros((20, 8)), None)
    _keep_only(cg, [(0, 1), (0, 2), (0, 3)])
    calls = []

    def oracle(x, y, Z):
        calls.append((int(x), int(y), tuple(Z)))
        return (0.9, 0.0) if (x, y) == (3, 0) else (0.0, 1.0)

    out = mci_skeleton(
        cg,
        oracle,
        alpha=0.05,
        lagged_parents={0: [], 1: [], 2: []},
        num_lags=1,
        include_C=True,
        num_c_cols=1,
        show_progress=False,
    )
    assert 3 not in out.neighbors(0)
    assert (3, 0, ()) in calls
    # Depth 0 conditions x1 -> x0 and x2 -> x0 on C alone; at depth 1 the
    # contemporaneous subset is the only condition left.
    depth1 = [Z for x, y, Z in calls if y == 0 and set(Z) & {1, 2}]
    assert sorted(depth1) == [(1,), (2,)], calls


def test_ancestor_pairs_count_every_c_column():
    # A, B + 2 C columns per slice: A(t-1) is node 4.
    knowledge = AncestralKnowledge().add_forbidden_ancestor("A", "B")
    knowledge.add_required_ancestor("A", "B")
    for resolve in (
        knowledge.resolve_forbidden_ancestor_pairs,
        knowledge.resolve_required_ancestor_pairs,
    ):
        assert resolve(["A", "B"], 1, True, num_c_cols=2) == [(0, 1), (4, 1)]
        assert resolve(["A", "B"], 1, True) == [(0, 1), (3, 1)]  # one C column


def test_background_knowledge_counts_every_c_column():
    # A, B + 2 C columns per slice: A(t-1) is node 4, not node 2 + 1 = 3.
    bk = make_background_knowledge(
        ["A", "B"], 1, include_C=True, required=[("A", "B", 1, 0)], num_c_cols=2
    )
    cg = initialize_graph(np.zeros((10, 8)), None)
    nodes = cg.G.nodes
    assert bk.is_required(nodes[4], nodes[1])
    assert not bk.is_required(nodes[3], nodes[1])


@pytest.mark.parametrize("fn", [run_cdnots, run_cdnots_plus])
def test_required_edge_lands_on_the_right_node_with_two_c_columns(fn):
    df = pd.DataFrame(
        np.random.default_rng(7).standard_normal((400, 2)), columns=["A", "B"]
    )
    knowledge = AncestralKnowledge().add_required_edge(
        "A", "B", cause_lag=1, effect_lag=0
    )
    res = fn(
        df,
        ParCorrGPU(np.zeros((2, 2)), device="cpu"),
        num_lags=1,
        include_C=True,
        c_preset="linear+sin",
        alpha=0.01,
        knowledge=knowledge,
        show_progress=False,
    )
    assert res.cg_tig[0, 1, 1] == 1  # A(t-1) -> B(t), as required
    assert res.cg_tig[3, 1, 0] == 0  # not C_sin -> B


@pytest.mark.parametrize("fn", [run_cdnots, run_cdnots_plus])
def test_forbidden_c_to_x_removes_the_edge(fn):
    # C can only be a cause, so forbidding C -> X leaves no admissible edge.
    rng = np.random.default_rng(4)
    t = np.arange(500, dtype=float)
    df = pd.DataFrame({"X": 0.04 * t + rng.normal(size=500), "Y": rng.normal(size=500)})
    bk = BackgroundKnowledge().add_forbidden_by_pattern("^X2$", "^X0$")
    res = fn(
        df,
        ParCorrGPU(np.zeros((2, 2)), device="cpu"),
        num_lags=0,
        include_C=True,
        alpha=0.01,
        background_knowledge=bk,
        show_progress=False,
    )
    assert res.cg_tig[2, 0, 0] == 0 and res.cg_tig[0, 2, 0] == 0


def test_lag_verification_conditions_on_c():
    # X and Y are noisy copies of a seasonal C; the X(t-1), X(t-2) - Y(t)
    # links exist only through C and must be removed.
    T = 1000
    rng = np.random.default_rng(0)
    c = np.sin(2 * np.pi * np.arange(T) / 30)
    df = pd.DataFrame(
        {
            "X": c + 0.8 * rng.normal(size=T),
            "Y": c + 0.8 * rng.normal(size=T),
            "C": c,
        }
    )
    data = pd.concat([df.shift(i) for i in range(3)], axis=1).iloc[4:].values
    cg = initialize_graph(data, None)
    _keep_only(cg, [(1, 3), (1, 6), (1, 2), (0, 2)])
    out, stats = lag_verification_pass(
        cg,
        ParCorrGPU(data, device="cpu"),
        alpha=0.05,
        num_lags=2,
        include_C=True,
        num_c_cols=1,
        alpha_lv=0.05,
    )
    assert stats["n_edges_removed"] == 2
    assert sorted(int(n) for n in out.neighbors(1)) == [2]


@pytest.mark.parametrize(
    "c_kw, expected",
    [
        ({}, "C_lin"),
        ({"c_array": np.c_[np.arange(500.0), np.sin(np.arange(500) / 20)]}, "C1"),
    ],
)
def test_select_features_reads_c_edges_from_a_cdnots_result(c_kw, expected):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((500, 3))
    x[:, 0] += 0.01 * np.arange(500)
    df = pd.DataFrame(x, columns=["A", "B", "Z"])
    res = run_cdnots_plus(
        df,
        ParCorrGPU(np.zeros((2, 2)), device="cpu"),
        num_lags=1,
        include_C=True,
        show_progress=False,
        **c_kw,
    )
    fs = select_features(df, "A", max_lag=1, from_result=res)
    assert [(f.variable, f.role) for f in fs.features] == [
        (expected, "nonstationarity")
    ]


@pytest.mark.parametrize("fn", [run_cdnots, run_cdnots_plus])
def test_forbidden_c_edge_keeps_the_ci_separation(fn):
    # Chain C -> Y -> X: the CI tests separate C and X given Y, so knowledge
    # forbidding C -> X must not replace that sepset; Meek orients Y -> X.
    rng = np.random.default_rng(0)
    y = 0.02 * np.arange(600.0) + rng.normal(size=600)
    df = pd.DataFrame({"X": 1.5 * y + rng.normal(size=600), "Y": y})
    bk = BackgroundKnowledge().add_forbidden_by_pattern("^X2$", "^X0$")
    res = fn(
        df,
        ParCorrGPU(np.zeros((2, 2)), device="cpu"),
        num_lags=0,
        include_C=True,
        alpha=0.01,
        background_knowledge=bk,
        show_progress=False,
    )
    assert res.cg_tig[1, 0, 0] == 1 and res.cg_tig[0, 1, 0] == 0  # Y -> X


def test_forbidden_c_edges_keep_the_ci_separation_for_the_sepset_rule():
    # Two chains C -> Y -> X and C -> Z -> W with C -> X and C -> W forbidden.
    # Dropping those edges before the CI tests would lose the sepsets {Y} and
    # {Z} that let Meek orient Y -> X and Z -> W; phase three then guesses.
    rng = np.random.default_rng(0)
    t = np.arange(800.0)
    y = 0.018 * t + rng.normal(size=800)
    z = 0.014 * t + rng.normal(scale=1.3, size=800)
    df = pd.DataFrame(
        {
            "X": 1.3 * y + rng.normal(size=800),
            "Y": y,
            "W": 1.1 * z + rng.normal(size=800),
            "Z": z,
        }
    )
    bk = BackgroundKnowledge()
    bk.add_forbidden_by_pattern("^X4$", "^X0$")
    bk.add_forbidden_by_pattern("^X4$", "^X2$")
    g = cdnots_discovery(
        df,
        ParCorrGPU(np.zeros((2, 2)), device="cpu"),
        num_lags=0,
        include_C=True,
        alpha=0.001,
        background_knowledge=bk,
        contemp_collider_rule=None,
        show_progress=False,
    )[0].cg_tig
    assert g[1, 0, 0] == 1 and g[0, 1, 0] == 0  # Y -> X
    assert g[3, 2, 0] == 1 and g[2, 3, 0] == 0  # Z -> W


def test_knowledge_removed_c_edge_is_no_collider_evidence():
    # C also drives X directly, so the C - X edge survives the CI tests and is
    # removed by knowledge alone. The empty sepset recording that is not a CI
    # result, so the sepset collider rule must not orient X -> Y <- C.
    rng = np.random.default_rng(0)
    t = np.arange(600.0)
    y = 0.02 * t + rng.normal(size=600)
    df = pd.DataFrame({"X": 1.5 * y + 0.02 * t + rng.normal(size=600), "Y": y})
    bk = BackgroundKnowledge().add_forbidden_by_pattern("^X2$", "^X0$")
    res = cdnots_discovery(
        df,
        ParCorrGPU(np.zeros((2, 2)), device="cpu"),
        num_lags=0,
        include_C=True,
        alpha=0.01,
        background_knowledge=bk,
        contemp_collider_rule=None,
        show_progress=False,
    )[0]
    assert res.cg_tig[2, 0, 0] == 0
    assert res.cg_tig[0, 1, 0] == 1 and res.cg_tig[1, 0, 0] == 1  # X - Y


def test_select_features_spouses_use_the_padded_names():
    # Custom C (no c_node_names) and an observed column named like a padded C
    # node: the spouse in T -> C3 <- Spouse must still be found.
    names = ["T", "C3", "Spouse"]
    g = np.zeros((4, 4, 1), dtype=np.int8)
    g[0, 1, 0] = g[2, 1, 0] = 1  # T -> C3 <- Spouse
    g[3, 0, 0] = 1  # C -> T
    df = pd.DataFrame(np.zeros((30, 3)), columns=names)
    res = CdnotsResult(
        graph=initialize_graph(np.zeros((30, 4)), None),
        cg_tig=g,
        pvalue_matrix=np.ones_like(g, dtype=float),
        pvals=[],
        var_names=names,
        num_lags=0,
        lag_list=None,
        include_C=True,
        alpha=0.05,
        priority=1,
        stable=False,
        lag_verify_stats=None,
        df=df,
        num_c_cols=1,
        c_node_names=None,
    )
    fs = select_features(df, "T", max_lag=0, from_result=res, markov_blanket=True)
    assert sorted((f.variable, f.role) for f in fs.features) == [
        ("C1", "nonstationarity"),
        ("C3", "child"),
        ("Spouse", "spouse"),
    ]


def test_select_features_c_label_does_not_shadow_an_observed_c1():
    # Custom C (no c_node_names) and an observed column named exactly like the
    # fallback C label: the child C1 must resolve to the observed node, so the
    # spouse in T -> C1 <- Spouse is found, and the C node gets its own label.
    names = ["T", "C1", "Spouse"]
    g = np.zeros((4, 4, 1), dtype=np.int8)
    g[0, 1, 0] = g[2, 1, 0] = 1  # T -> C1 <- Spouse
    g[3, 0, 0] = 1  # C -> T
    df = pd.DataFrame(np.zeros((30, 3)), columns=names)
    res = CdnotsResult(
        graph=initialize_graph(np.zeros((30, 4)), None),
        cg_tig=g,
        pvalue_matrix=np.ones_like(g, dtype=float),
        pvals=[],
        var_names=names,
        num_lags=0,
        lag_list=None,
        include_C=True,
        alpha=0.05,
        priority=1,
        stable=False,
        lag_verify_stats=None,
        df=df,
        num_c_cols=1,
        c_node_names=None,
    )
    fs = select_features(df, "T", max_lag=0, from_result=res, markov_blanket=True)
    assert sorted((f.variable, f.role) for f in fs.features) == [
        ("C1", "child"),
        ("Spouse", "spouse"),
        ("_C1", "nonstationarity"),
    ]


def test_select_features_cdnots_keeps_a_spouse_named_like_the_c_node():
    # Fresh algo="cdnots" with a custom C: its fallback label is C{d} = "C3",
    # the name of an observed spouse in T(t-1) -> Child(t) <- C3(t-1). The C
    # node must get its own label instead of shadowing that spouse.
    rng = np.random.default_rng(0)
    n = 600
    t = rng.standard_normal(n) + 0.01 * np.arange(n)
    c3 = rng.standard_normal(n)
    child = 0.5 * rng.standard_normal(n)
    child[1:] += 0.8 * t[:-1] + 0.8 * c3[:-1]
    df = pd.DataFrame({"T": t, "Child": child, "C3": c3})
    fs = select_features(
        df,
        "T",
        max_lag=1,
        algo="cdnots",
        markov_blanket=True,
        c_array=np.arange(n, dtype=float)[:, None],
    )
    assert sorted((f.variable, f.lag, f.role) for f in fs.features) == [
        ("C3", 0, "spouse"),
        ("Child", 1, "child"),
        ("_C3", 0, "nonstationarity"),
    ]


def test_collider_lag_copies_do_not_recreate_a_forbidden_c_edge():
    # x, y, z, C at lags 0 and 1 (lagged copies 4-7); only C(t-1) -> x(t-1)
    # is forbidden. Orienting the collider C -> x <- y copies it to lag 1,
    # which must not bring the removed C(t-1) - x(t-1) edge back.
    g = initialize_graph(np.zeros((20, 8)), None)
    _keep_only(g, [(3, 0), (0, 1), (7, 4), (4, 5)])
    bk = BackgroundKnowledge().add_forbidden_by_pattern("^X7$", "^X4$")
    g = c_lag_cnst(g, 1, True, 1, background_knowledge=bk)
    assert g.G.get_edge(g.G.nodes[7], g.G.nodes[4]) is None
    out = uc_sepset(
        g,
        priority=1,
        background_knowledge=bk,
        num_lags=1,
        contemp_collider_rule=None,
    )
    assert out.G.get_edge(out.G.nodes[7], out.G.nodes[4]) is None
    assert out.is_fully_directed(3, 0) and out.is_fully_directed(1, 0)


@pytest.mark.parametrize("seed", [1, 2, 5])
def test_collider_vote_conditions_on_c_parents(seed):
    """X0 -> X1 <- X2 at lag 0, X1's mean shifts with C (C -> X1).

    The majority vote for X0(t-1) -> X0(t) - X1(t) tests X1(t) vs X0(t-1). With C
    only an optional subset member, the subsets without it test an X1 swamped by
    its regime shift, find spurious separating sets without X0(t), call a collider
    and orient X1 -> X0: a conflict (x-x) with the true X0 -> X1. C is a parent,
    so the vote conditions on it in every test, as MCI does.
    """
    T = 1000
    rng = np.random.default_rng(seed)
    c = np.r_[np.zeros(T // 2), np.ones(T // 2)]
    x = np.zeros((T, 3))
    e = rng.normal(size=(T, 3))
    for t in range(1, T):
        x[t, 0] = 0.5 * x[t - 1, 0] + e[t, 0]
        x[t, 2] = 0.5 * x[t - 1, 2] + e[t, 2]
        x[t, 1] = 0.5 * x[t - 1, 1] + 0.6 * x[t, 0] + 0.6 * x[t, 2] + 2 * c[t] + e[t, 1]
    df = pd.DataFrame(x, columns=["X0", "X1", "X2"])
    res = run_cdnots_plus(
        df,
        ParCorrGPU(df.values, device="cpu"),
        num_lags=1,
        alpha=0.01,
        include_C=True,
        c_array=c,
        show_progress=False,
    )
    marks = np.asarray(res.to_marks())
    assert marks[0, 1, 0] == "-->" and marks[2, 1, 0] == "-->"
