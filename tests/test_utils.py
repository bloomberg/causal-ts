# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for utility functions."""

import numpy as np
import pandas as pd
import pytest

from causalts.utils.helpers import evaluate_graph
from causalts.utils.linearity import check_linearity


def test_evaluate_graph_perfect_match():
    G = np.zeros((3, 3, 2), dtype=np.int8)
    G[0, 1, 1] = 1
    G[1, 2, 1] = 1
    metrics = evaluate_graph(G, G)
    assert metrics["F1"] == 1.0
    assert metrics["SHD"] == 0
    assert metrics["Precision"] == 1.0
    assert metrics["TPR"] == 1.0


def test_evaluate_graph_empty_vs_nonempty():
    G_true = np.zeros((3, 3, 2), dtype=np.int8)
    G_true[0, 1, 1] = 1
    G_est = np.zeros((3, 3, 2), dtype=np.int8)
    metrics = evaluate_graph(G_est, G_true)
    assert metrics["TPR"] == 0.0
    assert metrics["SHD"] > 0


def test_evaluate_graph_all_false_positives():
    G_true = np.zeros((3, 3, 2), dtype=np.int8)
    G_est = np.ones((3, 3, 2), dtype=np.int8)
    metrics = evaluate_graph(G_est, G_true)
    assert metrics["Precision"] < 0.5
    assert metrics["SHD"] > 0


_LEGACY_KEYS = (
    "TPR",
    "FPR",
    "Precision",
    "F1",
    "SHD",
    "SHD_pair",
    "F1_pair",
    "TPR_pair",
    "FPR_pair",
    "Precision_pair",
    "TP",
    "FP",
    "FN",
    "TN",
)


def _cases():
    """(name, G_est, G_true, exclude_self_loops) covering the interesting shapes."""
    d = 3
    perfect = np.zeros((d, d, 2), dtype=np.int8)
    perfect[0, 1, 1] = 1
    perfect[1, 2, 0] = 1

    empty = np.zeros((d, d, 2), dtype=np.int8)
    full = np.ones((d, d, 2), dtype=np.int8)

    selfloops = np.zeros((d, d, 2), dtype=np.int8)
    selfloops[0, 0, 1] = 1
    selfloops[0, 1, 0] = 1

    return [
        ("perfect", perfect, perfect, False),
        ("miss_all", empty, perfect, False),
        ("all_fp", full, empty, False),
        ("selfloops_kept", selfloops, perfect, False),
        ("selfloops_excluded", selfloops, perfect, True),
        ("no_lagged_slice", perfect[:, :, :1], perfect[:, :, :1], False),
    ]


def test_new_keys_do_not_disturb_existing_ones():
    """Additive only: every pre-existing key keeps its exact value."""
    for name, est, true, excl in _cases():
        m = evaluate_graph(est, true, exclude_self_loops=excl)
        # Recompute the legacy block the way it was computed before the
        # lag-resolved groups were added, and require an exact match.
        d = est.shape[0]
        mask = np.ones(est.shape, dtype=bool)
        for i in range(d):
            if excl:
                mask[i, i, :] = False
            else:
                mask[i, i, 0] = False
        e, t = est[mask].astype(bool), true[mask].astype(bool)
        tp = int(np.sum(e & t))
        fp = int(np.sum(e & ~t))
        fn = int(np.sum(~e & t))
        assert (m["TP"], m["FP"], m["FN"]) == (tp, fp, fn), name
        assert m["SHD"] == fp + fn, name
        for k in _LEGACY_KEYS:
            assert k in m, f"{name}: legacy key {k} disappeared"


def test_lag_groups_partition_the_pooled_counts():
    """lag0 + lagpos must sum to the pooled directed counts."""
    for name, est, true, excl in _cases():
        m = evaluate_graph(est, true, exclude_self_loops=excl)
        for c in ("TP", "FP", "FN"):
            assert m[f"{c}_lag0"] + m[f"{c}_lagpos"] == m[c], f"{name}: {c}"


def test_lag0_adjacency_hand_computed():
    """3-node graph small enough to verify by eye.

    Truth is X0 -> X1 at lag 0. The estimate returns it unoriented, as a
    symmetric pair -- the exact case an o-o rendering produces.
    """
    true = np.zeros((3, 3, 1), dtype=np.int8)
    true[0, 1, 0] = 1
    est = np.zeros((3, 3, 1), dtype=np.int8)
    est[0, 1, 0] = 1
    est[1, 0, 0] = 1

    m = evaluate_graph(est, true)

    # Directed: the mirrored cell is a false positive.
    assert (m["TP_lag0"], m["FP_lag0"], m["FN_lag0"]) == (1, 1, 0)
    # Adjacency: one pair, counted once, correct.
    assert (m["TP_lag0_adj"], m["FP_lag0_adj"], m["FN_lag0_adj"]) == (1, 0, 0)
    assert m["F1_lag0_adj"] == 1.0

    # Dropping the edge instead costs the same SHD but the opposite F1 error.
    dropped = np.zeros((3, 3, 1), dtype=np.int8)
    m2 = evaluate_graph(dropped, true)
    assert m["SHD_lag0"] == m2["SHD_lag0"] == 1
    assert m2["F1_lag0_adj"] == 0.0


def test_lagpos_is_zero_when_there_is_no_lagged_slice():
    g = np.zeros((3, 3, 1), dtype=np.int8)
    g[0, 1, 0] = 1
    m = evaluate_graph(g, g)
    assert m["TP_lagpos"] == m["FP_lagpos"] == m["FN_lagpos"] == 0
    assert m["SHD_lagpos"] == 0


def test_evaluate_graph_rejects_string_arrays():
    marks = np.full((3, 3, 2), "", dtype="<U3")
    marks[0, 1, 0] = "o-o"
    gt = np.zeros((3, 3, 2), dtype=np.int8)
    with pytest.raises(TypeError, match="to_binary"):
        evaluate_graph(marks, gt)


def test_check_linearity_on_linear_data():
    rng = np.random.default_rng(42)
    T = 300
    x = rng.standard_normal(T)
    y = 0.8 * x + 0.2 * rng.standard_normal(T)
    df = pd.DataFrame({"X": x, "Y": y})
    result = check_linearity(df, alpha=0.05)
    assert isinstance(result, dict)
    assert "fraction_nonlinear" in result
    assert result["fraction_nonlinear"] == 0.0


def test_check_linearity_on_nonlinear_data():
    rng = np.random.default_rng(42)
    T = 200
    x = rng.standard_normal(T)
    y = np.sin(3 * x) + 0.1 * rng.standard_normal(T)
    df = pd.DataFrame({"X": x, "Y": y})
    result = check_linearity(df, alpha=0.05)
    assert isinstance(result, dict)
    assert "fraction_nonlinear" in result
    assert result["fraction_nonlinear"] > 0
