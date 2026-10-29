# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for GRACE's effect floor: grow admission and final pruning."""

import numpy as np
import pandas as pd
import pytest

from causalts.grace.gated_discovery import EFFECT_FLOOR, run_cdnots_gated
from causalts.grace.grow import grow_candidates, prune_effect_floor


def _weak_strong(rng, T, strong=0.6, weak=0.06):
    """x1(t) = strong * x0(t-1) + weak * x2(t-1) + noise; x0, x2 white."""
    X = rng.standard_normal((T, 3))
    X[1:, 1] += strong * X[:-1, 0] + weak * X[:-1, 2]
    return X


def test_default_is_point_01_nats():
    assert EFFECT_FLOOR == pytest.approx(1 - np.exp(-0.02))
    assert 0.0197 < EFFECT_FLOOR < 0.0199


def test_grow_floor_drops_significant_but_negligible_edge():
    # T large: the weak edge (partial R^2 ~ 0.004) is highly significant
    X = _weak_strong(np.random.default_rng(0), 20000)
    empty = np.zeros((3, 3, 2), dtype=bool)
    loose, _ = grow_candidates(X, empty, empty, max_lag=1, transform="rank")
    floored, _ = grow_candidates(
        X, empty, empty, max_lag=1, transform="rank", effect_floor=EFFECT_FLOOR
    )
    assert loose[0, 1, 1] and loose[2, 1, 1]
    assert floored[0, 1, 1] and not floored[2, 1, 1]


def test_grow_floor_zero_is_unchanged():
    X = _weak_strong(np.random.default_rng(1), 2000)
    empty = np.zeros((3, 3, 2), dtype=bool)
    a, na = grow_candidates(X, empty, empty, max_lag=1, transform="rank")
    b, nb = grow_candidates(
        X, empty, empty, max_lag=1, transform="rank", effect_floor=0.0
    )
    assert na == nb and np.array_equal(a, b)


def test_prune_drops_weak_keeps_strong_and_self_loops():
    X = _weak_strong(np.random.default_rng(2), 20000)
    G = np.zeros((3, 3, 2), dtype=np.int8)
    G[0, 1, 1] = G[2, 1, 1] = 1
    G[1, 1, 1] = 1  # self-loop: never pruned
    out = prune_effect_floor(X, G, 1, EFFECT_FLOOR)
    assert out[0, 1, 1] == 1 and out[2, 1, 1] == 0 and out[1, 1, 1] == 1
    assert np.array_equal(prune_effect_floor(X, G, 1, 0.0), G)


def test_prune_keeps_variance_only_parent():
    # x1's noise scale is set by x0(t-1); its mean does not depend on x0
    rng = np.random.default_rng(3)
    T = 5000
    x0 = rng.standard_normal(T)
    x1 = np.r_[0.0, np.exp(0.8 * x0[:-1]) * rng.standard_normal(T - 1)]
    G = np.zeros((2, 2, 2), dtype=np.int8)
    G[0, 1, 1] = 1
    out = prune_effect_floor(np.column_stack([x0, x1]), G, 1, EFFECT_FLOOR)
    assert out[0, 1, 1] == 1


def test_prune_removes_redundant_proxy_not_the_parent():
    # x2 is a noisy copy of x0: given x0, it adds nothing for x1
    rng = np.random.default_rng(4)
    T = 10000
    x0 = rng.standard_normal(T)
    x2 = x0 + 0.5 * rng.standard_normal(T)
    x1 = np.r_[0.0, 0.7 * x0[:-1] + rng.standard_normal(T - 1)]
    G = np.zeros((3, 3, 2), dtype=np.int8)
    G[0, 1, 1] = G[2, 1, 1] = 1
    out = prune_effect_floor(np.column_stack([x0, x1, x2]), G, 1, EFFECT_FLOOR)
    assert out[0, 1, 1] == 1 and out[2, 1, 1] == 0


@pytest.mark.parametrize("bad", [-0.1, 1.0])
def test_run_rejects_bad_floor(bad):
    df = pd.DataFrame(np.random.default_rng(5).standard_normal((100, 3)))
    with pytest.raises(ValueError, match="effect_floor"):
        run_cdnots_gated(df, max_lag=1, effect_floor=bad, verbose=False)


def test_run_reports_floor_and_prunes_negligible_edge():
    X = _weak_strong(np.random.default_rng(6), 6000, weak=0.1)  # R^2 ~ 0.007
    df = pd.DataFrame(X)
    sk = np.zeros((3, 3, 2), dtype=np.int8)
    sk[0, 1, 1] = sk[2, 1, 1] = 1
    kw = dict(
        max_lag=1,
        skeleton=sk,
        verbose=False,
        model_seed=0,
        device="cpu",
        max_epochs=40,
        ar_path=True,
    )
    res = run_cdnots_gated(df, **kw)
    off = run_cdnots_gated(df, effect_floor=0.0, **kw)
    assert res.effect_floor == pytest.approx(EFFECT_FLOOR)
    assert off.n_effect_pruned == 0
    assert off.cg_tig[2, 1, 1]  # the gates alone keep the negligible edge
    assert off.effect_pruned is None
    assert res.n_effect_pruned >= 1 and res.effect_pruned[2, 1, 1]
    assert res.cg_tig[0, 1, 1] and not res.cg_tig[2, 1, 1]
    assert not (res.effect_pruned & (np.asarray(res.cg_tig) != 0)).any()
