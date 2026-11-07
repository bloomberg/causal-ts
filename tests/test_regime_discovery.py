# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the within-regime segment vote of ``run_regime_discovery``."""

import types

import numpy as np
import pandas as pd
import pytest

from causalts.regime import RegimeConfig
from causalts.regime.discovery import run_regime_discovery

A, B, C = 0, 1, 2


def _run(segment_graphs, segment_threshold=0.5, threshold_rounding="ceil"):
    """Three or more 100-row segments of one regime, each given a stub graph.

    ``threshold_rounding=None`` leaves the RegimeConfig default.
    """
    n = len(segment_graphs)
    T = 100 * (2 * n - 1)
    labels = np.ones(T, dtype=np.int32)  # 1 = gap rows outside the regime
    starts = [200 * k for k in range(n)]
    for s in starts:
        labels[s : s + 100] = 0
    by_start = dict(zip(starts, segment_graphs))

    def discovery_fn(seg_df, max_lag, alpha, **_):
        return types.SimpleNamespace(cg_tig=by_start[seg_df.index[0]])

    df = pd.DataFrame(np.zeros((T, 3)), columns=["A", "B", "C"])
    kw = (
        {} if threshold_rounding is None else {"threshold_rounding": threshold_rounding}
    )
    config = RegimeConfig(n_regimes=1, segment_threshold=segment_threshold, **kw)
    res = run_regime_discovery(df, discovery_fn, config=config, regime_labels=labels)
    # Failed segments are dropped silently; a rejection must not pass that way.
    assert len(res.per_regime_results[0]) == n, "a stub segment failed silently"
    return res.regime_graphs[0]


def _graph(*edges):
    g = np.zeros((3, 3, 2), dtype=np.int8)
    for i, j, lag in edges:
        g[i, j, lag] = 1
    return g


def test_lag0_edge_found_in_every_segment_is_kept():
    """The vote looked at lags >= 1 only, so lag-0 edges never passed."""
    g = _run([_graph((A, B, 0))] * 3)
    assert g[A, B, 0] == 1


def test_lag0_self_loop_is_not_an_edge():
    g = _run([_graph((A, A, 0))] * 3)
    assert not g.any()


def test_ceil_threshold_rounds_up():
    """int(0.5 * 3) = 1 accepted an edge found in one segment of three."""
    one_of_three = [_graph((C, A, 1)), _graph(), _graph()]
    assert _run(one_of_three)[C, A, 1] == 0
    two_of_three = [_graph((C, A, 1)), _graph((C, A, 1)), _graph()]
    assert _run(two_of_three)[C, A, 1] == 1


@pytest.mark.parametrize("found, kept", [(6, False), (7, True)])
def test_threshold_is_not_pushed_up_by_float_rounding(found, kept):
    """0.28 * 25 is 7.000000000000001 in floating point; seven of 25 suffice."""
    graphs = [_graph((C, A, 1))] * found + [_graph()] * (25 - found)
    assert bool(_run(graphs, segment_threshold=0.28)[C, A, 1]) is kept


def test_threshold_just_above_an_integer_rounds_up():
    """0.70000000001 of 10 segments needs 8 votes, not 7."""
    seven = [_graph((C, A, 1))] * 7 + [_graph()] * 3
    assert _run(seven, segment_threshold=0.70000000001)[C, A, 1] == 0
    assert _run(seven, segment_threshold=0.7)[C, A, 1] == 1


def test_threshold_given_as_a_ratio_accepts_that_ratio():
    """5 / 6 as a threshold must accept an edge found in five of six segments."""
    five = [_graph((C, A, 1))] * 5 + [_graph()]
    assert _run(five, segment_threshold=5 / 6)[C, A, 1] == 1
    four = [_graph((C, A, 1))] * 4 + [_graph()] * 2
    assert _run(four, segment_threshold=5 / 6)[C, A, 1] == 0


def test_regime_summary_counts_lag0_edges():
    """regime_summary counted lags >= 1 only, missing lag-0 edges the vote keeps."""
    from causalts.regime.result import RegimeResult

    g = _graph((A, B, 0), (A, A, 0), (B, C, 1))
    res = RegimeResult(
        cg_tig=g,
        var_names=["A", "B", "C"],
        regime_graphs={0: g},
        regime_labels=np.zeros(10, dtype=np.int32),
        regime_names=["r0"],
        regime_params=[types.SimpleNamespace(max_lag=1, alpha=0.05)],
        aggregation_strategy="or",
        per_regime_results={},
        df=pd.DataFrame(np.zeros((10, 3)), columns=["A", "B", "C"]),
    )
    assert res.regime_summary()["r0"]["n_edges"] == 2


def test_lag0_vote_is_on_the_adjacency_not_the_orientation():
    """A->B, B->A and nothing: two of three segments found an A-B adjacency."""
    g = _run([_graph((A, B, 0)), _graph((B, A, 0)), _graph()])
    assert g[A, B, 0] == 1 and g[B, A, 0] == 1  # kept, left unoriented


def test_lag0_adjacency_is_oriented_by_a_direction_majority():
    unoriented = _graph((A, B, 0), (B, A, 0))
    g = _run([_graph((A, B, 0)), _graph((A, B, 0)), unoriented])
    assert g[A, B, 0] == 1 and g[B, A, 0] == 0


def test_unoriented_lag0_edge_does_not_vote_for_a_direction():
    """A->B, A-B (unoriented) and nothing: only one segment chose A->B."""
    unoriented = _graph((A, B, 0), (B, A, 0))
    g = _run([_graph((A, B, 0)), unoriented, _graph()])
    assert g[A, B, 0] == 1 and g[B, A, 0] == 1


def test_lag0_adjacency_below_the_threshold_is_dropped():
    g = _run([_graph((A, B, 0)), _graph(), _graph()])
    assert not g.any()


def test_lagged_vote_ignores_lag0():
    """A lag-0 edge does not vote for the lagged edge of the same pair."""
    g = _run([_graph((A, B, 1)), _graph((A, B, 0)), _graph()])
    assert not g.any()


def test_count_edges_counts_an_unoriented_lag0_edge_once():
    from causalts.regime.result import count_edges

    assert count_edges(_graph((A, B, 0), (B, A, 0))) == 1
    assert count_edges(_graph((A, B, 0), (B, A, 0), (A, B, 1))) == 2
    assert count_edges(_graph((A, B, 0), (B, C, 1), (C, C, 1))) == 3


def test_default_rounding_is_floor():
    """The historical rule stays the default: one segment of three suffices at 0.5."""
    assert RegimeConfig().threshold_rounding == "floor"
    one_of_three = [_graph((C, A, 1)), _graph(), _graph()]
    assert _run(one_of_three, threshold_rounding=None)[C, A, 1] == 1


@pytest.mark.parametrize(
    "n, found, threshold, kept",
    [
        (3, 1, 0.5, True),  # max(1, int(1.5)) = 1
        (5, 1, 0.5, False),  # int(2.5) = 2
        (5, 2, 0.5, True),
        (4, 1, 0.25, True),  # int(1.0) = 1
        (10, 2, 0.1, True),  # max(1, int(1.0)) = 1
        (3, 1, 0.1, True),  # max(1, int(0.3)) = 1
    ],
)
def test_floor_rounding_matches_the_historical_formula(n, found, threshold, kept):
    graphs = [_graph((C, A, 1))] * found + [_graph()] * (n - found)
    g = _run(graphs, segment_threshold=threshold, threshold_rounding="floor")
    assert bool(g[C, A, 1]) is kept


def test_floor_rounding_applies_to_the_lag0_vote():
    g = _run([_graph((A, B, 0)), _graph(), _graph()], threshold_rounding="floor")
    assert g[A, B, 0] == 1 and g[B, A, 0] == 0


def test_floor_rounding_leaves_lag0_unoriented_when_both_directions_reach_the_count():
    """A->B, A->B, B->A with a count of 1: both directions reach it."""
    g = _run(
        [_graph((A, B, 0)), _graph((A, B, 0)), _graph((B, A, 0))],
        threshold_rounding="floor",
    )
    assert g[A, B, 0] == 1 and g[B, A, 0] == 1


def test_floor_rounding_orients_lag0_from_one_one_way_segment():
    """A->B, A-B, A-B with a count of 1: one segment orients it, none the other way."""
    unoriented = _graph((A, B, 0), (B, A, 0))
    g = _run([_graph((A, B, 0)), unoriented, unoriented], threshold_rounding="floor")
    assert g[A, B, 0] == 1 and g[B, A, 0] == 0


def test_unknown_threshold_rounding_is_rejected():
    with pytest.raises(ValueError, match="threshold_rounding"):
        RegimeConfig(threshold_rounding="round")


def test_threshold_rounding_set_after_construction_is_still_checked():
    """A typo set on the mutable config must not fall back to floor rounding."""
    labels = np.zeros(300, dtype=np.int32)
    config = RegimeConfig(n_regimes=1)
    config.threshold_rounding = "Ceil"
    df = pd.DataFrame(np.zeros((300, 3)), columns=["A", "B", "C"])

    def discovery_fn(seg_df, max_lag, alpha, **_):
        return types.SimpleNamespace(cg_tig=_graph())

    with pytest.raises(ValueError, match="threshold_rounding"):
        run_regime_discovery(df, discovery_fn, config=config, regime_labels=labels)


def test_regime_config_positional_order_is_unchanged():
    cfg = RegimeConfig(1, None, None, 0, "or", 0.5, ["r0"])
    assert cfg.regime_names == ["r0"] and cfg.threshold_rounding == "floor"
