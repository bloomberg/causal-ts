# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""CDNOTS+ with a gapped ``lag_list``: MCI and the collider step must read
columns at their real lags, aligned with the phase-1 window."""

import numpy as np
import pandas as pd

from causalts.cdnots.phase3_utils import (
    _deep_lags,
    _majority_collider_step,
    run_cdnots_plus,
)
from causalts.cdnots.skeleton_discovery import (
    c_lag_cnst,
    initialize_graph,
    mci_skeleton,
)
from causalts.ci_tests.parcorr_gpu import ParCorrGPU


def test_deep_lags():
    assert _deep_lags([0, 1, 2]) == [0, 1, 2, 3, 4]
    assert _deep_lags([0, 2, 4]) == [0, 2, 4, 6, 8]
    assert _deep_lags([0, 1, 3]) == [0, 1, 3, 2, 4, 6]


class _Fixed:
    """Records every call; every test is independent."""

    def __init__(self):
        self.calls = []

    def __call__(self, X, Y, condition_set=None):
        self.calls.append((X, Y, list(condition_set or [])))
        return 0.9, 0.0


def test_mci_shifts_parents_by_real_lag():
    # Window lags [0, 1, 3], two variables: x0(t-3) = 4, x0(t-1) = 2.
    # Testing x0(t-3) -> x1(t) conditions on x0's lag-1 parent shifted by 3,
    # i.e. x0(t-4): slice 4 of the deep lags [0, 1, 3, 2, 4, 6], column 8.
    # Slice arithmetic (1 + 2 = slice 3) would pick x0(t-2), column 6.
    cg = initialize_graph(np.zeros((5, 6)), None)
    oracle = _Fixed()
    mci_skeleton(
        cg,
        oracle,
        alpha=0.05,
        lagged_parents={0: [2], 1: [4]},
        num_lags=2,
        include_C=False,
        num_c_cols=0,
        show_progress=False,
        no_of_var_ext=12,
        lags=[0, 1, 3, 2, 4, 6],
    )
    zs = [Z for x, y, Z in oracle.calls if (x, y) == (4, 1)]
    assert zs and all(8 in Z and 6 not in Z for Z in zs)


def test_collider_step_reads_a_deep_lag():
    # Window lags [0, 1, 3], two variables. Triple x1(t-3) = 5 - x0(t) = 0 -
    # x1(t) = 1. x1(t-3)'s lag-1 parent x1(t-1) shifts to x1(t-4), which only
    # the deep array holds: lag 4 is slice 4 of [0, 1, 3, 2, 4, 6], column 9.
    cg = initialize_graph(np.zeros((5, 6)), None)
    keep = [{0, 1}, {0, 5}]
    for a in range(6):
        for b in range(a + 1, 6):
            if {a, b} not in keep:
                cg.G.remove_edge(cg.G.get_edge(cg.G.nodes[a], cg.G.nodes[b]))
    cg = c_lag_cnst(cg, 2, include_C=False)
    oracle = _Fixed()
    oracle.data = None
    df = pd.DataFrame(np.random.default_rng(0).standard_normal((40, 2)))
    _majority_collider_step(
        cg,
        oracle,
        df,
        np.zeros((34, 6)),
        [0, 1, 3],
        2,
        [],
        {0: [], 1: [3]},
        alpha=0.05,
    )
    zs = [Z for x, y, Z in oracle.calls if {x, y} == {1, 5}]
    assert zs and all(9 in Z for Z in zs)


class _Checked(ParCorrGPU):
    """ParCorr that checks every column it reads is ``df[var]`` at the lag
    its registered lag structure claims, on the window's rows."""

    def __init__(self, df, n_drop):
        super().__init__(np.zeros((2, 2)), device="cpu")
        self.df, self.n_drop = df, n_drop
        self.widths = set()

    def set_lag_structure(self, n_vars, lags):
        self._n, self._lags = n_vars, list(lags)

    def __call__(self, X, Y, condition_set=None):
        assert self.data.shape[1] == self._n * len(self._lags)
        self.widths.add(self.data.shape[1])
        for c in [X, Y, *(condition_set or [])]:
            lag = self._lags[c // self._n]
            want = self.df.iloc[:, c % self._n].shift(lag).values[self.n_drop :]
            np.testing.assert_array_equal(self.data[:, c], want)
        return super().__call__(X, Y, condition_set)


def _gapped_series(T=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, 3))
    for t in range(3, T):
        x[t, 0] += 0.5 * x[t - 1, 0]
        x[t, 1] += 0.4 * x[t - 1, 1] + 0.6 * x[t - 3, 0]
        x[t, 2] += 0.5 * x[t, 1]
    return pd.DataFrame(x, columns=["a", "b", "c"])


def test_run_cdnots_plus_gapped_lags_read_real_lags():
    df = _gapped_series()
    test = _Checked(df, n_drop=6)
    res = run_cdnots_plus(
        df,
        test,
        lag_list=[1, 3],
        include_C=False,
        alpha=0.01,
        show_progress=False,
    )
    # Window (3 lags) and the deep array (6 lags) were both exercised.
    assert test.widths == {3 * 3, 3 * 6}
    # a(t-3) -> b(t) is found at its real lag (slice 2 of [0, 1, 3]).
    assert res.cg_tig[0, 1, 2] == 1
