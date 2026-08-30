# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Search-quality tests for the vendored GES/LGES implementation.

Two bugs in the Insert operator made the forward phase stall far short of the
optimum, so LGES returned a much lower-scoring graph than GES and never improved
with more data:

* ``_creates_cycle`` rejected an insert whenever *any* semi-directed path y -> x
  existed, instead of only when one is *unblocked* by ``NA_yx | T``
* ``_score_valid_insert_operators`` drew ``T`` from ``NA_yx`` rather than
  ``Ne(y) \\ Adj(x)``, and omitted ``NA_yx`` from both sides of the score delta

GES is a consistent estimator, so the sharpest regression test is convergence:
given enough data from a linear-Gaussian model, the search must recover the true
equivalence class exactly.
"""

import numpy as np
import pytest

from causalts.baselines import ges_discovery, lges_discovery, tges_discovery
from causalts.lges import GaussObsL0Pen, fit, pdag_to_dag
from causalts.synthetic_data.synthetic_datasets import load_dataset
from causalts.utils import evaluate_graph


def _collider_data(n=20000, seed=7):
    """A -> C <- B, C -> D. Fully identifiable: the collider forces C -> D."""
    r = np.random.default_rng(seed)
    a = r.standard_normal(n)
    b = r.standard_normal(n)
    c = a + b + r.standard_normal(n)
    d = 1.2 * c + r.standard_normal(n)
    return np.column_stack([a, b, c, d])


def test_recovers_fully_identifiable_cpdag():
    """The search used to stall here and return every edge undirected."""
    X = _collider_data()
    A, _ = fit(GaussObsL0Pen(X), score_based=False, prune=False, forbidden=None)
    directed = {(i, j) for i in range(4) for j in range(4) if A[i, j] and not A[j, i]}
    undirected = {
        (i, j) for i in range(4) for j in range(i + 1, 4) if A[i, j] and A[j, i]
    }
    assert directed == {(0, 2), (1, 2), (2, 3)}
    assert undirected == set()


def test_search_is_not_beaten_by_causal_learn():
    """LGES must not return a lower-scoring graph than causal-learn's GES."""
    causallearn_ges = pytest.importorskip("causallearn.search.ScoreBased.GES").ges
    from causallearn.graph.Endpoint import Endpoint

    from causalts.baselines import _patch_ges_numpy2

    _patch_ges_numpy2()

    X = _collider_data()
    score = GaussObsL0Pen(X)

    A_lges, _ = fit(GaussObsL0Pen(X), score_based=False, prune=False, forbidden=None)

    G = causallearn_ges(X, score_func="local_score_BIC")["G"]
    nodes = G.get_nodes()
    A_cl = np.zeros((4, 4))
    for e in G.get_graph_edges():
        u, v = nodes.index(e.get_node1()), nodes.index(e.get_node2())
        e1, e2 = e.get_endpoint1(), e.get_endpoint2()
        if e1 == Endpoint.TAIL and e2 == Endpoint.ARROW:
            A_cl[u, v] = 1
        elif e1 == Endpoint.ARROW and e2 == Endpoint.TAIL:
            A_cl[v, u] = 1
        else:
            A_cl[u, v] = A_cl[v, u] = 1

    s_lges = score.score_dag(pdag_to_dag(A_lges))
    s_cl = score.score_dag(pdag_to_dag(A_cl))
    assert s_lges >= s_cl - 1e-6, f"LGES {s_lges:.2f} < causal-learn {s_cl:.2f}"


@pytest.mark.parametrize("mode", ["ges", "lges"])
def test_converges_on_large_sample(mode):
    """With 5000 samples of ex2 the search must recover the graph exactly.

    Before the fix this plateaued around F1 0.35-0.52 at every sample size.
    """
    r = load_dataset("ex2", seed=42, T=5000)
    df, gt = r["df"], r["ground_truth"]
    G_hat, _ = lges_discovery(df, max_lag=gt.shape[2] - 1, mode=mode)
    assert evaluate_graph(G_hat, gt)["F1"] == pytest.approx(1.0)


def test_tges_matches_lges_on_large_sample():
    r = load_dataset("ex2", seed=42, T=5000)
    df, gt = r["df"], r["ground_truth"]
    ml = gt.shape[2] - 1
    f_tges = evaluate_graph(tges_discovery(df, max_lag=ml, mode="lges")[0], gt)["F1"]
    f_lges = evaluate_graph(lges_discovery(df, max_lag=ml, mode="lges")[0], gt)["F1"]
    assert f_tges == pytest.approx(f_lges)


def test_ges_wrapper_also_converges():
    """Sanity anchor: the causal-learn-backed wrapper converges on the same data."""
    r = load_dataset("ex2", seed=42, T=5000)
    df, gt = r["df"], r["ground_truth"]
    G_hat, _ = ges_discovery(df, max_lag=gt.shape[2] - 1)
    assert evaluate_graph(G_hat, gt)["F1"] == pytest.approx(1.0)


def _random_lagged_search(seed, d, max_lag, n_parents, T=600):
    """Simulate a random legal SEM under temporal_forbidden and run fit()."""
    from causalts.lges import temporal_forbidden

    r = np.random.default_rng(seed)
    n = d * (max_lag + 1)
    forbidden = temporal_forbidden(d, max_lag)
    B = np.zeros((n, n))
    for j in range(n):
        legal = [i for i in range(n) if forbidden[i, j] == 0 and i != j]
        if not legal:
            continue
        k = min(n_parents, len(legal))
        for i in r.choice(legal, size=k, replace=False):
            B[i, j] = r.uniform(0.3, 0.9) * r.choice([-1, 1])
    X = r.standard_normal((T, n))
    for j in sorted(range(n), key=lambda x: -(x // d)):
        parents = np.where(B[:, j] != 0)[0]
        if len(parents):
            X[:, j] = X[:, parents] @ B[parents, j] + r.standard_normal(T)
    A, metrics = fit(
        GaussObsL0Pen(X), score_based=False, prune=False, forbidden=forbidden
    )
    return X, forbidden, A, d


@pytest.mark.parametrize("seed", range(30))
def test_forbidden_search_terminates_without_backward_edges(seed):
    """Regression cover for two defects found together:

    * ``pdag_to_dag``'s completion step ignored ``forbidden`` entirely, so it
      could resolve an edge the search legally created into the direction
      background knowledge rules out.
    * An early, overly-strict fix for that (reject any sink with a forbidden
      neighbour) was *incomplete*: 39 of 44 "No consistent extension exists"
      cases on a random sweep were false negatives -- a valid extension
      existed, the rule just couldn't find it. Fixed by forcing every edge
      with a uniquely-legal direction before running plain Dor-Tarsi, which
      preserves its classical completeness guarantee.
    * With the turning phase included, one specific case (d=5, max_lag=2,
      2 parents/node, seed=10) revealed a genuine 2-cycle: turn A applied,
      then turn B claims a gain that undoes A, forever. `fit()`'s turning
      loop now stops on a repeated graph state instead of looping forever.
    """
    r = np.random.default_rng(seed)
    d = int(r.integers(3, 8))
    max_lag = int(r.integers(0, 4))
    n_parents = int(r.integers(1, 4))
    X, forbidden, A, d = _random_lagged_search(seed, d, max_lag, n_parents)
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if i // d < j // d:  # i more recent than j
                assert not (
                    A[i, j] != 0 and A[j, i] == 0
                ), f"seed={seed}: directed backward-in-time edge col{i}->col{j}"


def test_turning_phase_terminates_on_known_cycle():
    """The specific (d, max_lag, seed) that used to hang forever must finish.

    No timeout plugin is installed, so without the SIGALRM guard a regression
    here hangs the whole suite instead of failing this one test.
    """
    import signal

    class _Hung(Exception):
        pass

    def _alarm(signum, frame):
        raise _Hung()

    previous = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(20)
    try:
        X, forbidden, A, d = _random_lagged_search(seed=10, d=5, max_lag=2, n_parents=2)
        A2, metrics = fit(
            GaussObsL0Pen(X),
            phases=["forward", "backward", "turning"],
            score_based=False,
            prune=False,
            forbidden=forbidden,
        )
    except _Hung:
        pytest.fail(
            "turning phase did not terminate within 20s -- cycle guard regressed"
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    assert A2 is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
