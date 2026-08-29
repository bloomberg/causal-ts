# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the GES baseline wrapper (causalts.baselines.ges_discovery).

Regression cover for two bugs:

* An edge-orientation bug in the extraction loop, which read causal-learn's
  adjacency matrix with the endpoints transposed -- reversing every directed
  lag-0 edge and silently dropping every directed lagged edge.
* A missing-background-knowledge bug: with no temporal constraint, the search
  could orient edges backward in time between lag blocks. The fast engine now
  builds a ``forbidden`` matrix (``causalts.lges.temporal_forbidden``) and
  passes it all the way through the search, including into PDAG completion
  (``pdag_to_dag``/``pdag_to_cpdag``), which previously had no notion of it and
  could resolve an ambiguous edge in the illegal direction after the search had
  already respected the constraint everywhere else.
"""

import numpy as np
import pandas as pd
import pytest

from causalts.baselines import _ges_fast, _lag_embed, ges_discovery
from causalts.lges import lges_discovery, temporal_forbidden

T = 3000


def _edges(G_hat, names):
    """{(cause, lag, effect)} from a (d, d, max_lag+1) graph."""
    return {
        (names[i], lag, names[j])
        for lag in range(G_hat.shape[2])
        for i in range(G_hat.shape[0])
        for j in range(G_hat.shape[1])
        if G_hat[i, j, lag]
    }


@pytest.fixture
def lagged_df():
    """A(t-1) -> C(t) <- B(t-1),  C(t-1) -> D(t)."""
    r = np.random.default_rng(7)
    a = r.standard_normal(T)
    b = r.standard_normal(T)
    c = np.zeros(T)
    d = np.zeros(T)
    for t in range(1, T):
        c[t] = a[t - 1] + b[t - 1] + r.standard_normal()
        d[t] = 1.2 * c[t - 1] + r.standard_normal()
    return pd.DataFrame({"A": a, "B": b, "C": c, "D": d})


@pytest.fixture
def collider_df():
    """Contemporaneous collider X -> Z <- Y (direction is identifiable)."""
    r = np.random.default_rng(3)
    x = r.standard_normal(T)
    y = r.standard_normal(T)
    return pd.DataFrame({"X": x, "Y": y, "Z": x + y + r.standard_normal(T)})


def test_recovers_lagged_edges(lagged_df):
    G_hat, _ = ges_discovery(lagged_df, max_lag=1)
    assert _edges(G_hat, list("ABCD")) == {
        ("A", 1, "C"),
        ("B", 1, "C"),
        ("C", 1, "D"),
    }


def test_agrees_with_lges_ges_mode(lagged_df):
    """Same algorithm, independent implementation and adjacency convention."""
    names = list("ABCD")
    ges_edges = _edges(ges_discovery(lagged_df, max_lag=1)[0], names)
    lges_edges = _edges(lges_discovery(lagged_df, max_lag=1, mode="ges")[0], names)
    assert ges_edges == lges_edges


def test_lag0_collider_not_reversed(collider_df):
    """The bug returned Z -> X and Z -> Y here — every arrow flipped."""
    G_hat, _ = ges_discovery(collider_df, max_lag=0)
    assert _edges(G_hat, list("XYZ")) == {("X", 0, "Z"), ("Y", 0, "Z")}


@pytest.mark.parametrize("lam", [None, 0.5, 2.0])
def test_engines_agree_at_max_lag_zero(collider_df, lam):
    """At max_lag=0 there is no temporal ordering to protect, so the two
    engines run the literal same algorithm and must return the same graph.
    """
    fast, info_f = ges_discovery(
        collider_df, max_lag=0, lambda_value=lam, engine="fast"
    )
    slow, info_s = ges_discovery(
        collider_df, max_lag=0, lambda_value=lam, engine="causal-learn"
    )
    assert info_f["engine"] == "fast"
    assert info_s["engine"] == "causal-learn"
    assert np.array_equal(fast, slow)


@pytest.mark.parametrize("max_lag", [1, 2])
def test_causal_learn_engine_rejects_lagged_data(lagged_df, max_lag):
    """causal-learn's ges() has no forbidden-edges parameter, so running it on
    lagged data would search unconstrained and could orient edges backward in
    time -- refuse rather than silently return a weaker/different graph.
    """
    with pytest.raises(ValueError, match="temporal background knowledge"):
        ges_discovery(lagged_df, max_lag=max_lag, engine="causal-learn")


def test_fast_is_the_default(lagged_df):
    assert ges_discovery(lagged_df, max_lag=1)[1]["engine"] == "fast"


def test_unknown_engine_rejected(lagged_df):
    with pytest.raises(ValueError, match="engine must be"):
        ges_discovery(lagged_df, max_lag=1, engine="nope")


def test_non_bic_score_falls_back_to_causal_learn(collider_df):
    """The fast engine only implements BIC, so other scores must still work
    (at max_lag=0, where the causal-learn engine is actually usable).
    """
    _, info = ges_discovery(
        collider_df, max_lag=0, score_func="local_score_BDeu", engine="fast"
    )
    assert info["engine"] == "causal-learn"


def test_non_bic_score_rejected_with_lags(lagged_df):
    with pytest.raises(ValueError, match="temporal background knowledge"):
        ges_discovery(lagged_df, max_lag=1, score_func="local_score_BDeu")


def test_no_backward_in_time_directed_edges(lagged_df):
    """The search must never leave a directed edge pointing from a more recent
    lag block into an older one -- this is what 'forbidden' exists to prevent,
    and PDAG completion (pdag_to_dag) used to ignore it after the fact.
    """
    embedded, d = _lag_embed(lagged_df, max_lag=2)
    forbidden = temporal_forbidden(d, 2)
    _, info = _ges_fast(embedded, d, 2, None, forbidden)
    A = info["cpdag"]
    p = A.shape[0]
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if i // d < j // d:  # i is more recent than j
                assert not (
                    A[i, j] != 0 and A[j, i] == 0
                ), f"directed backward-in-time edge: col{i} -> col{j}"


def test_lag0_chain_stays_undirected():
    """X -> Y -> Z is Markov equivalent to two other DAGs, so GES cannot orient it.

    An unoriented edge must be reported in both directions, not dropped.
    """
    r = np.random.default_rng(11)
    x = r.standard_normal(T)
    y = 1.5 * x + r.standard_normal(T)
    z = 1.5 * y + r.standard_normal(T)
    G_hat, _ = ges_discovery(pd.DataFrame({"X": x, "Y": y, "Z": z}), max_lag=0)
    assert _edges(G_hat, list("XYZ")) == {
        ("X", 0, "Y"),
        ("Y", 0, "X"),
        ("Y", 0, "Z"),
        ("Z", 0, "Y"),
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
