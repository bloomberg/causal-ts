# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for lag-0 orientation from lagged parents.

Defects that kept CDNOTS / CDNOTS+ from orienting contemporaneous edges:

1. Meek R1 skipped every triple whose source was lagged, so
   ``X(t-tau) -> Y(t) - Z(t)`` never oriented ``Y -> Z``.
2. The majority collider rule conditioned a lagged endpoint on its
   *unshifted* phase-1 parents. With an autocorrelated endpoint that put the
   endpoint into its own conditioning set, every test returned p ~ 1 and the
   vote came out exactly 0.5.
3. An ambiguous majority vote was passed to Meek as a known non-collider.
4. Plain CDNOTS ran the majority rule with no lagged conditioning at all, so
   a separating set could never contain a lagged node.

Node layout used below (two variables, one lag):
``0 = x0(t)``, ``1 = x1(t)``, ``2 = x0(t-1)``, ``3 = x1(t-1)``.
"""

import numpy as np
import pandas as pd
import pytest

from causalts.cdnots.meek import meek
from causalts.cdnots.phase3_utils import run_cdnots, run_cdnots_plus
from causalts.cdnots.skeleton_discovery import c_lag_cnst, initialize_graph
from causalts.cdnots.uc_sepset import uc_sepset
from causalts.ci_tests.parcorr_gpu import ParCorrGPU
from causalts.ci_tests.stratified_cit import StratifiedCIT


def _graph(edges, n_nodes=4, num_lags=1):
    """Window graph with only ``edges`` kept, lagged edges directed by time."""
    cg = initialize_graph(np.zeros((5, n_nodes)), None)
    keep = {frozenset(e) for e in edges}
    for a in range(n_nodes):
        for b in range(a + 1, n_nodes):
            if frozenset((a, b)) not in keep:
                cg.G.remove_edge(cg.G.get_edge(cg.G.nodes[a], cg.G.nodes[b]))
    return c_lag_cnst(cg, num_lags, include_C=False)


def _directed(cg, a, b):
    g = cg.G.graph
    return g[a, b] == -1 and g[b, a] == 1


def _undirected(cg, a, b):
    g = cg.G.graph
    return g[a, b] == -1 and g[b, a] == -1


class _Oracle:
    """Records each call; answers like x0(t) -> x1(t) with AR(1) on both.

    A test that conditions on one of its own arguments returns p = 1, which is
    what a real test does when X is in Z.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, x, z, Z):
        self.calls.append((x, z, list(Z)))
        if x in Z or z in Z:
            return 1.0, 0.0
        pair = {x, z}
        if pair == {0, 3}:  # x0(t) -> x1(t) <- x1(t-1): collider
            return (0.0, 9.0) if 1 in Z else (0.9, 0.1)
        if pair == {1, 2}:  # x0(t-1) -> x0(t) -> x1(t): chain
            return (0.9, 0.1) if 0 in Z else (0.0, 9.0)
        return 0.0, 9.0


def _uc(oracle, lagged_parents, no_of_var_ext=None):
    cg = _graph([(0, 1), (2, 0), (3, 1), (2, 3)])
    return uc_sepset(
        cg,
        priority=1,
        num_lags=1,
        contemp_collider_rule="majority",
        ci_test=oracle,
        alpha=0.05,
        lagged_parents=lagged_parents,
        no_of_var_ext=no_of_var_ext,
    )


def test_majority_rule_never_conditions_on_the_tested_pair():
    # x1(t-1) (node 3) is also listed as a phase-1 parent of x0: phase 1
    # predates MCI pruning, so a tested node can still sit there.
    oracle = _Oracle()
    _uc(oracle, {0: [2, 3], 1: [3]})
    assert oracle.calls
    bad = [c for c in oracle.calls if c[0] in c[2] or c[1] in c[2]]
    assert not bad, f"tested node inside its own conditioning set: {bad}"


def test_majority_rule_finds_collider_into_autocorrelated_child():
    cg = _uc(_Oracle(), {0: [2], 1: [3]})
    assert _directed(cg, 0, 1)
    assert not cg.ambiguous_triples


def test_majority_rule_shifts_lagged_endpoint_parents():
    # With a 3-slice array, x1(t-1)'s parent x1(t-1) shifts to x1(t-2) = node 5.
    oracle = _Oracle()
    _uc(oracle, {0: [2], 1: [3]}, no_of_var_ext=6)
    lagged_end = [Z for x, z, Z in oracle.calls if {x, z} == {0, 3}]
    assert lagged_end and all(5 in Z for Z in lagged_end)


def test_meek_r1_orients_from_lagged_parent():
    # x0(t-1) -> x0(t) - x1(t), x0(t-1) not adjacent to x1(t).
    cg = meek(_graph([(2, 0), (0, 1), (2, 3)]), num_lags=1)
    assert _directed(cg, 0, 1)
    assert _directed(cg, 2, 3)  # propagated to the lagged copy


def test_meek_r1_skips_ambiguous_triple():
    cg = _graph([(2, 0), (0, 1), (2, 3)])
    cg.ambiguous_triples = {(2, 0, 1), (1, 0, 2)}
    cg = meek(cg, num_lags=1)
    assert _undirected(cg, 0, 1)


def test_meek_r1_skips_ambiguous_triple_with_lag0_source():
    # No lags: 0 -> 1 - 2, 0 not adjacent to 2. R1 would orient 1 -> 2.
    cg = _graph([(0, 1), (1, 2)], n_nodes=3, num_lags=0)
    cg.G.graph[0, 1], cg.G.graph[1, 0] = -1, 1
    cg.ambiguous_triples = {(0, 1, 2), (2, 1, 0)}
    assert _undirected(meek(cg, num_lags=0), 1, 2)
    del cg.ambiguous_triples
    assert _directed(meek(cg, num_lags=0), 1, 2)


def test_cdnots_plus_orients_contemporaneous_edge_between_ar_series():
    rng = np.random.default_rng(0)
    T = 2000
    x = rng.standard_normal((T, 2))
    for t in range(1, T):
        x[t, 0] += 0.5 * x[t - 1, 0]
        x[t, 1] += 0.5 * x[t - 1, 1] + 0.6 * x[t, 0]
    df = pd.DataFrame(x, columns=["X0", "X1"])
    res = run_cdnots_plus(
        df,
        ParCorrGPU(np.zeros((2, 2)), device="cpu"),
        num_lags=2,
        include_C=False,
        alpha=0.01,
        show_progress=False,
    )
    marks = np.asarray(res.to_marks())
    assert marks[0, 1, 0] == "-->"
    assert res.cg_tig[0, 1, 0] == 1 and res.cg_tig[1, 0, 0] == 0


class _Fixed:
    """Oracle with fixed verdicts per unordered pair; records every call."""

    def __init__(self, verdicts, default=0.0):
        self.verdicts, self.default, self.calls = verdicts, default, []

    def __call__(self, x, z, Z):
        self.calls.append((x, z, list(Z)))
        p = self.verdicts.get(frozenset((x, z)), self.default)
        return p, 0.0


def test_majority_rule_records_ambiguous_triples_and_meek_respects_them():
    # x0(t) / x1(t-1): never separated -> ambiguous (no separating set).
    # x1(t) / x0(t-1): separated by every subset -> vote exactly 0.5.
    oracle = _Fixed({frozenset((0, 3)): 0.0, frozenset((1, 2)): 0.9})
    cg = _uc(oracle, {0: [2], 1: [3]})
    assert cg.ambiguous_triples == {(0, 1, 3), (3, 1, 0), (1, 0, 2), (2, 0, 1)}
    assert _undirected(meek(cg, num_lags=1), 0, 1)


def test_meek_r3_skips_ambiguous_triple():
    # Kite: 0 - 1, 0 - 2, 0 - 3, 1 -> 3 <- 2, 1 and 2 nonadjacent.
    cg = _graph([(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)], n_nodes=4, num_lags=0)
    for a in (1, 2):
        cg.G.graph[a, 3], cg.G.graph[3, a] = -1, 1
    assert _directed(meek(cg, num_lags=0), 0, 3)
    cg.ambiguous_triples = {(1, 0, 2), (2, 0, 1)}
    assert _undirected(meek(cg, num_lags=0), 0, 3)


def test_meek_r1_conflict_leaves_edge_undirected():
    # x0(t-1) -> x0(t) - x1(t) <- x1(t-1): R1 implies both 0 -> 1 and 1 -> 0.
    # The result used to depend on which triple was visited first.
    cg = meek(_graph([(2, 0), (3, 1), (0, 1), (2, 3)]), num_lags=1)
    assert _undirected(cg, 0, 1)


def test_majority_rule_shifts_by_real_lag_with_gapped_lag_list():
    # lags [0, 1, 2, 4]: slice 3 holds lag 4. Two variables, so node
    # x1(t-2) = 5, and its lag-2 parent x1(t-2) shifts to x1(t-4) = 7.
    cg = _graph([(0, 1), (5, 1)], n_nodes=8, num_lags=3)
    oracle = _Fixed({}, default=0.9)
    uc_sepset(
        cg,
        num_lags=3,
        contemp_collider_rule="majority",
        ci_test=oracle,
        alpha=0.05,
        lagged_parents={0: [], 1: [5]},
        lags=[0, 1, 2, 4],
    )
    zs = [Z for x, z, Z in oracle.calls if {x, z} == {0, 5}]
    assert zs and all(7 in Z for Z in zs)


def test_majority_rule_applies_max_conds_caps():
    oracle = _Oracle()
    cg = _graph([(0, 1), (2, 0), (3, 1), (2, 3)])
    uc_sepset(
        cg,
        num_lags=1,
        contemp_collider_rule="majority",
        ci_test=oracle,
        alpha=0.05,
        lagged_parents={0: [2], 1: [3]},
        no_of_var_ext=6,
        max_conds_py=0,
        max_conds_px=0,
    )
    assert oracle.calls and all(not (set(Z) - {0, 1}) for _, _, Z in oracle.calls)


class _Recording(StratifiedCIT):
    """Discrete-aware test that records data width, conditioning set and the
    discrete indices in force at each call."""

    def __init__(self):
        super().__init__(
            data=np.zeros((2, 2)),
            discrete_cols=[],
            inner_cit=ParCorrGPU(np.zeros((2, 2)), device="cpu"),
        )
        self.seen = []

    def __call__(self, X, Y, condition_set=None):
        z = [int(c) for c in (condition_set or [])]
        self.seen.append((self.data.shape[1], z, sorted(self.discrete_cols)))
        return super().__call__(X, Y, condition_set)


def _ar_pair(T=1500, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, 2))
    x[:, 0] = np.round(x[:, 0])  # few-level, treated as discrete below
    for t in range(1, T):
        x[t, 1] += 0.5 * x[t - 1, 1] + 0.6 * x[t, 0]
    return pd.DataFrame(x, columns=["X0", "X1"])


def _run_recorded(fn, include_C):
    t = _Recording()
    fn(
        _ar_pair(),
        t,
        num_lags=1,
        include_C=include_C,
        alpha=0.01,
        discrete_cols=["X0"],
        show_progress=False,
    )
    return t


@pytest.mark.parametrize("include_C", [False, True])
@pytest.mark.parametrize("fn", [run_cdnots, run_cdnots_plus])
def test_discrete_indices_skip_the_c_columns_once(fn, include_C):
    # Slice width is 2 without C and 3 with it; X0 is column 0 of each slice.
    # compute_discrete_indices used to add a second C to a df that already
    # had one, giving X0 at [0, 4] instead of [0, 3].
    width = 3 if include_C else 2
    t = _run_recorded(fn, include_C)
    assert sorted(t.discrete_cols) == [0, width]
    assert t.data.shape[1] == 2 * width


@pytest.mark.parametrize("include_C", [False, True])
def test_cdnots_plus_collider_step_reads_deeper_array_and_restores_it(include_C):
    width = 3 if include_C else 2
    t = _run_recorded(run_cdnots_plus, include_C)
    deep = [s for s in t.seen if s[1] and max(s[1]) >= 2 * width]
    assert deep, "no shifted parent past lag 1 was ever conditioned on"
    assert all(max(z, default=-1) < w for w, z, _ in t.seen)
    # Every call on the deep array (MCI and collider step) sees the deep
    # discrete indices; every call on the window sees the window ones.
    assert all(d == [0, width, 2 * width] for w, _, d in t.seen if w == 3 * width)
    assert all(d == [0, width] for w, _, d in t.seen if w == 2 * width)
    assert t.data.shape[1] == 2 * width and sorted(t.discrete_cols) == [0, width]


@pytest.mark.parametrize("fail_at", [2, 4], ids=["mci_swap", "collider_swap"])
def test_cdnots_plus_restores_state_when_a_deep_swap_fails(monkeypatch, fail_at):
    # _set_discrete_indices calls: 1 window indices, 2 deep (MCI), 3 restore,
    # 4 deep (collider step), 5 restore.
    import causalts.cdnots.phase3_utils as P

    real = P._set_discrete_indices
    calls = {"n": 0}

    def flaky(test, idx):
        calls["n"] += 1
        if calls["n"] == fail_at:
            raise RuntimeError("boom")
        real(test, idx)

    monkeypatch.setattr(P, "_set_discrete_indices", flaky)
    t = _Recording()
    with pytest.raises(RuntimeError, match="boom"):
        run_cdnots_plus(
            _ar_pair(),
            t,
            num_lags=1,
            include_C=False,
            alpha=0.01,
            discrete_cols=["X0"],
            show_progress=False,
        )
    assert t.data.shape[1] == 4 and sorted(t.discrete_cols) == [0, 2]


def test_cdnots_majority_rule_searches_lagged_neighbours_without_forcing_them(
    monkeypatch,
):
    # Plain CDNOTS is PC on the lag window: lagged neighbours are optional
    # members of the conditioning subsets, never forced in, and nothing
    # outside the window is used. Only calls from the collider step count.
    import causalts.cdnots.phase3_utils as P

    t = _Recording()
    real = P.uc_sepset

    def wrapped(*a, **k):
        t.seen.clear()
        out = real(*a, **k)
        t.collider_calls = list(t.seen)
        return out

    monkeypatch.setattr(P, "uc_sepset", wrapped)
    run_cdnots(
        _ar_pair(),
        t,
        num_lags=1,
        include_C=False,
        alpha=0.01,
        discrete_cols=["X0"],
        show_progress=False,
    )
    calls = t.collider_calls
    assert calls, "collider step ran no tests"
    assert all(w == 4 and max(z, default=-1) < 4 for w, z, _ in calls)
    with_lagged = [z for _, z, _ in calls if any(c >= 2 for c in z)]
    without_lagged = [z for _, z, _ in calls if not any(c >= 2 for c in z)]
    assert with_lagged and without_lagged


def test_pc_majority_finds_separator_that_a_forced_set_would_miss():
    # Window nodes: 0 = x0(t), 1 = x1(t), 2 = x0(t-1), 3 = x1(t-1).
    # Triple x0(t) - x1(t) - x1(t-1) with neighbours of x0(t): {1, 2}.
    # x0(t) and x1(t-1) are separated by {2} alone; adding x1(t) (the
    # collider) re-opens the path. So the triple is a collider.
    def oracle(x, z, Z):
        if {x, z} == {0, 3}:
            return (0.9, 0.0) if 2 in Z and 1 not in Z else (0.0, 9.0)
        return 0.0, 9.0

    cg = _graph([(0, 1), (2, 0), (3, 1), (2, 3)])
    out = uc_sepset(
        cg,
        num_lags=1,
        contemp_collider_rule="majority",
        ci_test=oracle,
        alpha=0.05,
        majority_mode="pc",
    )
    assert _directed(out, 0, 1)
    assert (0, 1, 3) not in out.ambiguous_triples


@pytest.mark.parametrize(
    "chain",
    [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]],
    ids=["forward", "reverse"],
)
def test_meek_reaches_closure_on_a_chain(chain):
    # a -> b - c - d - e: R1 must propagate all the way along. The loop's
    # bookkeeping used to end the search early; the reverse chain (sources
    # with high indices) stopped before the last edge.
    edges = list(zip(chain, chain[1:]))
    cg = _graph(edges, n_nodes=5, num_lags=0)
    a, b = edges[0]
    cg.G.graph[a, b], cg.G.graph[b, a] = -1, 1
    out = meek(cg, num_lags=0)
    for a, b in edges:
        assert _directed(out, a, b), (a, b)


def test_meek_r1_conflict_is_not_overridden_by_r2():
    # 2 -> 0 - 1 <- 3 (R1 implies both 0 -> 1 and 1 -> 0) plus 0 -> 4 -> 1,
    # which would let R2 orient 0 -> 1 in the same pass.
    cg = _graph([(2, 0), (3, 1), (0, 1), (0, 4), (4, 1)], n_nodes=5, num_lags=0)
    for a, b in [(2, 0), (3, 1), (0, 4), (4, 1)]:
        cg.G.graph[a, b], cg.G.graph[b, a] = -1, 1
    assert _undirected(meek(cg, num_lags=0), 0, 1)


def test_majority_vote_stops_early_with_the_same_decision():
    # No lags. x = 0 and z = 1 are nonadjacent; y = 2 is adjacent to both and
    # 3, 4, 5 are further neighbours of x: 16 subsets of adj(x) - {z}.
    # Separated exactly when y is left out -> a collider, which is certain
    # before all 16 subsets have been tested.
    edges = [(0, 2), (1, 2), (0, 3), (0, 4), (0, 5)]
    calls = []

    def oracle(x, z, Z):
        if {x, z} != {0, 1}:
            return 0.9, 0.0
        calls.append(list(Z))
        return (0.9, 0.0) if 2 not in Z else (0.0, 9.0)

    out = uc_sepset(
        _graph(edges, n_nodes=6, num_lags=0),
        num_lags=0,
        contemp_collider_rule="majority",
        ci_test=oracle,
        alpha=0.05,
        majority_mode="pc",
    )
    assert _directed(out, 0, 2) and _directed(out, 1, 2)
    assert 0 < len(calls) < 16


def test_collider_search_rejects_unknown_value_before_any_work():
    calls = []

    def ci(*a, **k):  # pragma: no cover - must never run
        calls.append(a)
        return 0.0, 0.0

    with pytest.raises(ValueError, match="collider_search"):
        run_cdnots(_ar_pair(T=50), ci, num_lags=1, collider_search="fast")
    assert not calls


def test_sepset_search_votes_only_over_the_separating_set_and_middle():
    # Window: 0 = x0(t), 1 = x1(t), 2 = x0(t-1), 3 = x1(t-1); the skeleton
    # separated x0(t) and x1(t-1) by {x0(t-1)}. The vote may use only
    # subsets of {x0(t-1), x1(t)}.
    # The skeleton records one entry per depth tested: () placeholders at
    # depth 0 where the pair was still dependent, then the separating set.
    cg = _graph([(0, 1), (2, 0), (3, 1), (2, 3)])
    cg.sepset[0, 3] = [(), (), (2,)]
    cg.sepset[3, 0] = [(), (), (2,)]
    calls = []

    def oracle(x, z, Z):
        if {x, z} == {0, 3}:
            calls.append(sorted(Z))
            return (0.9, 0.0) if 1 not in Z else (0.0, 9.0)
        return 0.9, 0.0

    out = uc_sepset(
        cg,
        num_lags=1,
        contemp_collider_rule="majority",
        ci_test=oracle,
        alpha=0.05,
        majority_mode="pc_sepset",
    )
    assert calls and all(Z in ([], [1], [2], [1, 2]) for Z in calls)
    assert any(2 in Z for Z in calls), "the recorded separator was ignored"
    assert _directed(out, 0, 1)


def test_sepset_search_falls_back_to_full_search_without_a_sepset():
    cg = _graph([(0, 1), (2, 0), (3, 1), (2, 3)])
    seen = {"pc": [], "pc_sepset": []}
    for mode in seen:

        def oracle(x, z, Z, mode=mode):
            if {x, z} == {0, 3}:
                seen[mode].append(sorted(Z))
            return 0.9, 0.0

        uc_sepset(
            cg,
            num_lags=1,
            contemp_collider_rule="majority",
            ci_test=oracle,
            alpha=0.05,
            majority_mode=mode,
        )
    assert seen["pc_sepset"] == seen["pc"] and seen["pc"]


def test_run_cdnots_collider_search_sepset_runs_and_differs_only_in_lag0():
    full = run_cdnots(
        _ar_pair(),
        ParCorrGPU(np.zeros((2, 2)), device="cpu"),
        num_lags=1,
        include_C=False,
        alpha=0.01,
        discrete_cols=[],
        show_progress=False,
    )
    cheap = run_cdnots(
        _ar_pair(),
        ParCorrGPU(np.zeros((2, 2)), device="cpu"),
        num_lags=1,
        include_C=False,
        alpha=0.01,
        discrete_cols=[],
        show_progress=False,
        collider_search="sepset",
    )
    np.testing.assert_array_equal(full.cg_tig[:, :, 1:], cheap.cg_tig[:, :, 1:])


def test_conflict_edges_survive_a_second_meek_call():
    cg = _graph([(2, 0), (3, 1), (0, 1), (0, 4), (4, 1)], n_nodes=5, num_lags=0)
    for a, b in [(2, 0), (3, 1)]:
        cg.G.graph[a, b], cg.G.graph[b, a] = -1, 1
    first = meek(cg, num_lags=0)
    assert first.conflict_edges == {(0, 1), (1, 0)}
    # Now give R2 a reason to orient 0 -> 1: it must still abstain.
    for a, b in [(0, 4), (4, 1)]:
        first.G.graph[a, b], first.G.graph[b, a] = -1, 1
    assert _undirected(meek(first, num_lags=0), 0, 1)


def test_phase_three_does_not_orient_a_conflict_edge(monkeypatch):
    import causalts.cdnots.phase3_utils as P

    # Nodes 0, 1 and C (= 2), no lags: C -> 0, C -> 1, 0 - 1 undirected.
    cg = _graph([(0, 1), (2, 0), (2, 1)], n_nodes=3, num_lags=0)
    for c_end in (0, 1):
        cg.G.graph[2, c_end], cg.G.graph[c_end, 2] = -1, 1
    monkeypatch.setattr(P, "infer_nonsta_dir", lambda cau, eff, c, w: float(eff.name))
    data = np.random.default_rng(0).standard_normal((50, 3))
    oriented = P.phase_three(cg, data, num_lags=0, include_C=True, num_c_cols=1)
    assert not _undirected(oriented, 0, 1)  # control: the C step does orient it
    cg.conflict_edges = {(0, 1), (1, 0)}
    kept = P.phase_three(cg, data, num_lags=0, include_C=True, num_c_cols=1)
    assert _undirected(kept, 0, 1)


def test_meek_has_no_fixed_pass_ceiling():
    # Closure of a 520-edge chain takes more passes than the old cap of 500.
    # The chain runs from high to low node index, which also defeats the old
    # loop bookkeeping.
    n = 521
    cg = _graph([(a, a - 1) for a in range(n - 1, 0, -1)], n_nodes=n, num_lags=0)
    cg.G.graph[n - 1, n - 2], cg.G.graph[n - 2, n - 1] = -1, 1
    out = meek(cg, num_lags=0)
    assert _directed(out, 1, 0)


def test_run_cdnots_passes_its_skeleton_budget_to_the_vote(monkeypatch):
    import causalts.cdnots.phase3_utils as P

    seen = {}
    real = P.uc_sepset

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(P, "uc_sepset", spy)
    run_cdnots(
        _ar_pair(T=300),
        ParCorrGPU(np.zeros((2, 2)), device="cpu"),
        num_lags=1,
        include_C=False,
        discrete_cols=[],
        max_combinations=7,
        max_degree=3,
        show_progress=False,
    )
    assert seen["max_subsets_per_size"] == 7 and seen["max_subset_size"] == 3
    assert seen["majority_mode"] == "pc"


def test_required_ancestor_knowledge_may_orient_a_conflict_edge():
    from causalts.cdnots.ancestral import (
        AncestralKnowledge,
        apply_ancestral_constraints,
    )

    cg = _graph([(0, 1)], n_nodes=2, num_lags=0)
    cg.conflict_edges = {(0, 1), (1, 0)}
    knowledge = AncestralKnowledge().add_required_ancestor("A", "B")
    out, _ = apply_ancestral_constraints(
        meek(cg, num_lags=0), knowledge, ["A", "B"], num_lags=0, include_C=False
    )
    assert _directed(out, 0, 1)
    assert not out.conflict_edges  # metadata must not claim a conflict


def _pc_calls_for_pair(cg, pair, **kw):
    calls = []

    def oracle(x, z, Z):
        if {x, z} == set(pair):
            calls.append(sorted(Z))
        return 0.0, 9.0  # never separated: every subset is tested

    uc_sepset(
        cg,
        num_lags=0,
        contemp_collider_rule="majority",
        ci_test=oracle,
        alpha=0.05,
        majority_mode="pc",
        **kw,
    )
    return calls


def test_pc_vote_budget_is_per_endpoint_like_the_skeleton():
    # x = 0 has neighbours {2, 3, 6}; z = 1 has neighbours {4, 5, 6}; the
    # middle node 6 is shared.
    edges = [(0, 2), (0, 3), (0, 6), (1, 4), (1, 5), (1, 6)]
    cg = _graph(edges, n_nodes=7, num_lags=0)
    calls = _pc_calls_for_pair(cg, (0, 1), max_subsets_per_size=1)
    # One size-1 subset from each endpoint's side, not one shared.
    assert sorted(Z for Z in calls if len(Z) == 1) == [[2], [4]]


def test_pc_vote_tries_the_recorded_separator_first():
    # Under a cap of one subset per size and endpoint, the skeleton's
    # separating node (4, a high index) must still be in the vote.
    edges = [(0, 2), (0, 3), (0, 4), (1, 2)]
    cg = _graph(edges, n_nodes=5, num_lags=0)
    cg.sepset[0, 1] = [(), (4,)]
    cg.sepset[1, 0] = [(), (4,)]
    calls = _pc_calls_for_pair(cg, (0, 1), max_subsets_per_size=1)
    assert [4] in calls


class _LagRecording(_Recording):
    """Also records the lag structure registered at each call."""

    def set_lag_structure(self, n_vars, lags):
        self._lags = list(lags)

    def __call__(self, X, Y, condition_set=None):
        self.lag_seen.append((self.data.shape[1], list(getattr(self, "_lags", []))))
        return super().__call__(X, Y, condition_set)


def test_cdnots_plus_deep_array_steps_register_the_deeper_lags():
    t = _LagRecording()
    t.lag_seen = []
    run_cdnots_plus(
        _ar_pair(),
        t,
        num_lags=1,
        include_C=False,
        alpha=0.01,
        discrete_cols=["X0"],
        show_progress=False,
    )
    deep = [lags for w, lags in t.lag_seen if w == 6]
    window = [lags for w, lags in t.lag_seen if w == 4]
    assert deep and all(lags == [0, 1, 2] for lags in deep)
    assert window and all(lags == [0, 1] for lags in window)
    assert t._lags == [0, 1]
