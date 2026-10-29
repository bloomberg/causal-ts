# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""The majority rule's ambiguity band (``majority_gamma``)."""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from causalts.cdnots.phase3_utils import run_cdnots, run_cdnots_plus
from causalts.cdnots.skeleton_discovery import c_lag_cnst, initialize_graph
from causalts.cdnots.uc_sepset import uc_sepset
from causalts.ci_tests.parcorr_gpu import ParCorrGPU

# x = 0, y = 1, z = 2; the skeleton separated x and z by {3, 4} (+ more), so
# the vote runs over the subsets of {y, 3, 4, ...}.


def _triple(n_extra=2):
    n = 3 + n_extra
    cg = initialize_graph(np.zeros((5, n)), None)
    for a in range(n):
        for b in range(a + 1, n):
            if {a, b} not in ({0, 1}, {1, 2}):
                cg.G.remove_edge(cg.G.get_edge(cg.G.nodes[a], cg.G.nodes[b]))
    cg = c_lag_cnst(cg, 0, include_C=False)
    sep = [tuple(range(3, n))]
    cg.sepset[0, 2], cg.sepset[2, 0] = sep, sep
    return cg


class _BySet:
    """Separates x and z exactly by the conditioning sets in ``separating``."""

    def __init__(self, separating):
        self.separating = {frozenset(S) for S in separating}

    def __call__(self, x, z, Z):
        return (0.9 if frozenset(Z) in self.separating else 0.0), 0.0


def _verdict(cg):
    g = cg.G.graph
    if g[0, 1] == -1 and g[1, 0] == 1 and g[2, 1] == -1 and g[1, 2] == 1:
        return "collider"
    if (0, 1, 2) in cg.ambiguous_triples:
        return "ambiguous"
    return "non-collider"


def _run(separating, gamma, n_extra=2):
    return uc_sepset(
        _triple(n_extra),
        contemp_collider_rule="majority",
        ci_test=_BySet(separating),
        alpha=0.05,
        majority_mode="pc_sepset",
        majority_gamma=gamma,
    )


# 2 of 5 separating sets hold y (share 0.4); 3 of 5 (0.6); 2 of 4 (0.5).
SHARE_04 = [(1,), (1, 3), (), (3,), (4,)]
SHARE_06 = [(1,), (1, 3), (1, 4), (), (3,)]
SHARE_05 = [(1,), (1, 3), (), (3,)]


@pytest.mark.parametrize(
    "separating, gamma, expected",
    [
        (SHARE_04, 0.5, "collider"),
        (SHARE_04, 0.4, "collider"),
        (SHARE_04, 0.3, "ambiguous"),
        (SHARE_06, 0.5, "non-collider"),
        (SHARE_06, 0.4, "non-collider"),
        (SHARE_06, 0.3, "ambiguous"),
        (SHARE_05, 0.5, "ambiguous"),
        (SHARE_05, 0.25, "ambiguous"),
        ([(), (3,), (4,)], 0.0, "collider"),
        ([(1,), (1, 3)], 0.0, "non-collider"),
        (SHARE_04, 0.0, "ambiguous"),
    ],
)
def test_band(separating, gamma, expected):
    assert _verdict(_run(separating, gamma)) == expected


@pytest.mark.parametrize("gamma", [0.5, 0.45, 0.4, 1 / 3, 0.25, 0.1, 0.0])
def test_early_stop_matches_the_full_vote(gamma):
    # The vote stops once the rest cannot change the verdict; it must agree
    # with classifying the full vote.
    rng = np.random.default_rng(int(gamma * 1000))
    cand = [1, 3, 4, 5]
    subsets = [S for k in range(5) for S in combinations(cand, k)]
    for _ in range(150):
        separating = [S for S in subsets if rng.random() < rng.random()]
        n = len(separating)
        n_y = sum(1 in S for S in separating)
        if n == 0:
            expected = "ambiguous"
        elif n_y / n < 0.5 and n_y / n <= gamma + 1e-12:
            expected = "collider"
        elif n_y / n > 0.5 and n_y / n >= 1 - gamma - 1e-12:
            expected = "non-collider"
        else:
            expected = "ambiguous"
        assert _verdict(_run(separating, gamma, n_extra=3)) == expected


@pytest.mark.parametrize("bad", [-0.1, 0.51, 1.0])
def test_entry_points_reject_a_bad_gamma_before_the_search(bad):
    df = pd.DataFrame(np.random.default_rng(0).standard_normal((60, 2)))
    for fn in (run_cdnots, run_cdnots_plus):
        with pytest.raises(ValueError, match="majority_gamma"):
            fn(df, None, majority_gamma=bad, show_progress=False)


def test_vote_stops_early_inside_the_band():
    # Separated by every subset without y and none with it: share 0 after
    # four of the eight subsets settles a collider for any gamma.
    calls = []

    class _Rec(_BySet):
        def __call__(self, x, z, Z):
            calls.append(Z)
            return super().__call__(x, z, Z)

    uc_sepset(
        _triple(),
        contemp_collider_rule="majority",
        ci_test=_Rec([(), (3,), (4,), (3, 4)]),
        alpha=0.05,
        majority_mode="pc_sepset",
        majority_gamma=0.25,
    )
    assert 0 < len(calls) < 8


@pytest.mark.parametrize("fn", [run_cdnots, run_cdnots_plus])
def test_entry_points_forward_gamma(monkeypatch, fn):
    import causalts.cdnots.phase3_utils as p3

    seen = []
    real = p3.uc_sepset

    def spy(*a, **k):
        seen.append(k.get("majority_gamma"))
        return real(*a, **k)

    monkeypatch.setattr(p3, "uc_sepset", spy)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((120, 3)))
    test = ParCorrGPU(np.zeros((2, 2)), device="cpu")
    fn(df, test, majority_gamma=0.3, include_C=False, show_progress=False)
    assert seen == [0.3]
