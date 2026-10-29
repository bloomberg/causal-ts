# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Required (background-knowledge) edges are kept without CI tests in CDNOTS+
phase 2 (MCI) and in the lag-verification pass, as in phase 1."""

import numpy as np
import pandas as pd
import pytest
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

from causalts.cdnots.phase3_utils import run_cdnots_plus
from causalts.cdnots.skeleton_discovery import (
    initialize_graph,
    lag_verification_pass,
    mci_skeleton,
)
from causalts.ci_tests.parcorr_gpu import ParCorrGPU


class _Independent:
    """Every test says independent; records the tested pairs and sets."""

    def __init__(self):
        self.pairs, self.calls = [], []

    def __call__(self, X, Y, condition_set=None):
        self.pairs.append(frozenset((X, Y)))
        self.calls.append((frozenset((X, Y)), list(condition_set or [])))
        return 0.9, 0.0


class _PinnedLagKnowledge(BackgroundKnowledge):
    """Set lookups over CDNOTS node names (``X{idx}``)."""

    def __init__(self, forbidden, required):
        super().__init__()
        self._forbidden = forbidden
        self._required = required

    def is_forbidden(self, node1, node2):
        return (node1.get_name(), node2.get_name()) in self._forbidden

    def is_required(self, node1, node2):
        return (node1.get_name(), node2.get_name()) in self._required


def _required(*pairs):
    req = set()
    for a, b in pairs:
        req |= {(f"X{a}", f"X{b}")}
    return _PinnedLagKnowledge(set(), req)


def test_mci_keeps_required_parents_without_testing_them():
    # Two variables, one lag: 0 = x0(t), 1 = x1(t), 2 = x0(t-1), 3 = x1(t-1).
    cg = initialize_graph(np.zeros((5, 4)), None)
    oracle = _Independent()
    out = mci_skeleton(
        cg,
        oracle,
        alpha=0.05,
        lagged_parents={0: [2, 3], 1: [2, 3]},
        num_lags=1,
        include_C=False,
        num_c_cols=0,
        show_progress=False,
        background_knowledge=_required((2, 1)),
    )
    assert frozenset((2, 1)) not in oracle.pairs
    assert frozenset((3, 1)) in oracle.pairs
    assert out.G.get_edge(out.G.nodes[2], out.G.nodes[1]) is not None
    assert out.G.get_edge(out.G.nodes[3], out.G.nodes[1]) is None


def test_lag_verification_keeps_required_lags():
    # Two lags: x0(t-1) = 2 and x0(t-2) = 4 both adjacent to x1(t) = 1.
    cg = initialize_graph(np.zeros((5, 6)), None)
    for a, b in [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 3), (1, 5)]:
        cg.G.remove_edge(cg.G.get_edge(cg.G.nodes[a], cg.G.nodes[b]))
    oracle = _Independent()
    out, _ = lag_verification_pass(
        cg,
        oracle,
        alpha=0.05,
        num_lags=2,
        include_C=False,
        num_c_cols=0,
        background_knowledge=_required((4, 1)),
    )
    assert frozenset((4, 1)) not in oracle.pairs
    assert out.G.get_edge(out.G.nodes[4], out.G.nodes[1]) is not None
    assert out.G.get_edge(out.G.nodes[2], out.G.nodes[1]) is None
    # The required lag still conditions the other one.
    assert all(4 in Z for pair, Z in oracle.calls if pair == frozenset((2, 1)))


def test_run_cdnots_plus_never_tests_a_required_edge():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 3))
    for t in range(1, 300):
        x[t, 1] += 0.5 * x[t - 1, 0]
        x[t, 2] += 0.5 * x[t - 1, 1]
    seen = []

    class _Rec(ParCorrGPU):
        def __call__(self, X, Y, condition_set=None):
            seen.append(frozenset((X, Y)))
            return super().__call__(X, Y, condition_set)

    # x0(t-1) = 3 -> x1(t) = 1, and its copy one slice back is in no window.
    bk = _required((3, 1))
    res = run_cdnots_plus(
        pd.DataFrame(x),
        _Rec(np.zeros((2, 2)), device="cpu"),
        include_C=False,
        background_knowledge=bk,
        show_progress=False,
        lag_verify=True,
    )
    assert seen and frozenset((3, 1)) not in seen
    assert res.cg_tig[0, 1, 1] == 1


def test_a_required_shifted_copy_protects_its_edge():
    # Two lags, two variables: x0(t-1) = 2 - x1(t) = 1, and its copy
    # x0(t-2) = 4 - x1(t-1) = 3. Requiring only the copy keeps both: removing
    # an edge removes its copies.
    from causalts.cdnots.phase3_utils import make_background_knowledge

    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((300, 2)), columns=["A", "B"])
    bk = make_background_knowledge(
        ["A", "B"], num_lags=2, required=[("A", "B", 2, 1)], include_C=False
    )
    seen = []

    class _Rec(ParCorrGPU):
        def __call__(self, X, Y, condition_set=None):
            seen.append(frozenset((X, Y)))
            return super().__call__(X, Y, condition_set)

    res = run_cdnots_plus(
        df,
        _Rec(np.zeros((2, 2)), device="cpu"),
        num_lags=2,
        include_C=False,
        background_knowledge=bk,
        show_progress=False,
        lag_verify=True,
    )
    assert seen and frozenset((2, 1)) not in seen
    assert res.cg_tig[0, 1, 1] == 1


@pytest.mark.parametrize("stable", [True, False])
def test_plain_skeleton_keeps_required_edges(stable):
    from causalts.cdnots.skeleton_discovery import skeleton_discovery

    cg = initialize_graph(np.zeros((5, 4)), None)
    oracle = _Independent()
    out = skeleton_discovery(
        cg,
        0.05,
        oracle,
        stable=stable,
        background_knowledge=_required((2, 1)),
        show_progress=False,
        num_lags=1,
    )
    assert out.G.get_edge(out.G.nodes[2], out.G.nodes[1]) is not None
    assert out.G.get_edge(out.G.nodes[3], out.G.nodes[1]) is None
    assert frozenset((2, 1)) not in oracle.pairs


def test_run_cdnots_batched_skeleton_never_tests_a_required_edge():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 3))
    for t in range(1, 300):
        x[t, 1] += 0.5 * x[t - 1, 0]
    seen = []

    class _Rec(ParCorrGPU):
        def __call__(self, X, Y, condition_set=None):
            seen.append(frozenset((X, Y)))
            return super().__call__(X, Y, condition_set)

        def batch_test(self, tests):
            seen.extend(frozenset((X, Y)) for X, Y, _ in tests)
            return super().batch_test(tests)

    # x2(t-1) = 5 -> x1(t) = 1: no dependence, so only knowledge keeps it.
    from causalts.cdnots.phase3_utils import run_cdnots

    res = run_cdnots(
        pd.DataFrame(x),
        _Rec(np.zeros((2, 2)), device="cpu"),
        include_C=False,
        background_knowledge=_required((5, 1)),
        show_progress=False,
    )
    assert seen and frozenset((5, 1)) not in seen
    assert res.cg_tig[2, 1, 1] == 1


class _SeparatedByZero:
    """1 and 2 are independent only given 0; every other pair is dependent."""

    def __init__(self):
        self.sets = []

    def __call__(self, X, Y, condition_set=None):
        S = tuple(sorted(int(v) for v in (condition_set or [])))
        if frozenset((X, Y)) == frozenset((1, 2)):
            self.sets.append(S)
            return (0.9 if 0 in S else 0.001), 0.0
        return 0.001, 5.0


class _BatchedSeparatedByZero(_SeparatedByZero):
    """Same answers through ``batch_test``, the path GPU tests take."""

    def batch_test(self, tests):
        return [self(X, Y, S) for X, Y, S in tests]


@pytest.mark.parametrize(
    "stable, oracle_cls",
    [
        (True, _SeparatedByZero),
        (False, _SeparatedByZero),
        (True, _BatchedSeparatedByZero),  # batching only runs when stable
    ],
)
def test_untested_required_neighbours_rank_first_as_separators(stable, oracle_cls):
    # 0 -> 1 and 0 -> 2 required (so never tested); 1 and 2 are separated by
    # {0} alone and 3 is a distractor. With one set per size, {0} must be the
    # depth-1 set tried for 1 - 2, as in CDNOTS+.
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

    from causalts.cdnots.skeleton_discovery import skeleton_discovery

    cg = initialize_graph(np.zeros((20, 4)), None)
    bk = BackgroundKnowledge()
    for b in (1, 2):
        bk.add_required_by_node(cg.G.nodes[0], cg.G.nodes[b])
    oracle = oracle_cls()
    out = skeleton_discovery(
        cg,
        0.05,
        oracle,
        stable=stable,
        background_knowledge=bk,
        show_progress=False,
        max_combinations=1,
    )
    assert out.G.get_edge(out.G.nodes[1], out.G.nodes[2]) is None
    depth1 = {S for S in oracle.sets if len(S) == 1}
    assert depth1 == {(0,)}
    assert not any(3 in S for S in oracle.sets)
