# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for TigramiteEffects (skipped if tigramite not installed)."""

import numpy as np
import pandas as pd
import pytest

tigramite = pytest.importorskip("tigramite")

from causalts.effects.tigramite_effects import (
    TigramiteEffects,
    _to_tigramite_graph,
    tigramite_available,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _chain_graph_and_data():
    """3-variable chain: X0(t-1)->X1(t), X1(t-1)->X2(t), self-AR on all."""
    rng = np.random.default_rng(42)
    T = 500
    data = np.zeros((T, 3))
    for t in range(2, T):
        data[t, 0] = 0.50 * data[t - 1, 0] + rng.normal()
        data[t, 1] = 0.60 * data[t - 1, 0] + 0.40 * data[t - 1, 1] + rng.normal()
        data[t, 2] = 0.70 * data[t - 1, 1] + 0.30 * data[t - 1, 2] + rng.normal()
    df = pd.DataFrame(data, columns=["X0", "X1", "X2"])

    G = np.zeros((3, 3, 3), dtype=int)  # max_lag=2
    G[0, 1, 1] = 1  # X0(t-1) -> X1(t)
    G[1, 2, 1] = 1  # X1(t-1) -> X2(t)
    for i in range(3):
        G[i, i, 1] = 1  # self-AR
    return G, df


# ---------------------------------------------------------------------------
# Graph conversion (no tigramite dependency — pure numpy)
# ---------------------------------------------------------------------------


def test_to_tigramite_graph_shape():
    G, _ = _chain_graph_and_data()
    tig = _to_tigramite_graph(G)
    assert tig.shape == G.shape
    assert tig.dtype.kind == "U"  # string dtype


def test_to_tigramite_graph_edges():
    G, _ = _chain_graph_and_data()
    tig = _to_tigramite_graph(G)
    assert tig[0, 1, 1] == "-->"
    assert tig[1, 2, 1] == "-->"
    assert tig[0, 2, 1] == ""  # no direct X0->X2 edge


def test_to_tigramite_graph_lag0_diagonal_empty():
    G, _ = _chain_graph_and_data()
    # Even if lag-0 self-loops were set, they must be empty in tigramite format
    G[0, 0, 0] = 1
    tig = _to_tigramite_graph(G)
    assert tig[0, 0, 0] == ""


# ---------------------------------------------------------------------------
# TigramiteEffects (requires tigramite)
# ---------------------------------------------------------------------------


def test_tigramite_available():
    assert tigramite_available() is True


def test_repr():
    G, df = _chain_graph_and_data()
    eff = TigramiteEffects(
        G, df, X_var="X0", X_lag=2, Y_var="X2", var_names=["X0", "X1", "X2"]
    )
    assert "X0" in repr(eff)
    assert "X2" in repr(eff)


def test_var_by_index():
    G, df = _chain_graph_and_data()
    eff = TigramiteEffects(G, df, X_var=0, X_lag=2, Y_var=2)
    assert eff._X_var == "X0"
    assert eff._Y_var == "X2"


def test_total_effect_chain():
    """X0(t-2) -> X1(t-1) -> X2(t): total ≈ 0.6 * 0.7 = 0.42."""
    G, df = _chain_graph_and_data()
    eff = TigramiteEffects(
        G, df, X_var="X0", X_lag=2, Y_var="X2", var_names=["X0", "X1", "X2"]
    )
    eff.fit()
    total = eff.total_effect()
    assert abs(total - 0.42) < 0.08  # within 8% of expected


def test_direct_effect_no_direct_edge():
    """X0->X2 is fully mediated via X1, so direct effect ≈ 0."""
    G, df = _chain_graph_and_data()
    eff = TigramiteEffects(
        G, df, X_var="X0", X_lag=2, Y_var="X2", var_names=["X0", "X1", "X2"]
    )
    eff.fit()
    assert abs(eff.direct_effect()) < 0.05


def test_direct_effect_direct_edge():
    """X0(t-1) -> X1(t) is a direct edge, so direct ≈ total ≈ 0.6."""
    G, df = _chain_graph_and_data()
    eff = TigramiteEffects(
        G, df, X_var="X0", X_lag=1, Y_var="X1", var_names=["X0", "X1", "X2"]
    )
    eff.fit()
    assert abs(eff.direct_effect() - 0.6) < 0.08


def test_get_mediators():
    """X1 is on the path X0(t-2) -> X1(t-1) -> X2(t)."""
    G, df = _chain_graph_and_data()
    eff = TigramiteEffects(
        G, df, X_var="X0", X_lag=2, Y_var="X2", var_names=["X0", "X1", "X2"]
    )
    eff.fit()
    meds = eff.get_mediators()
    assert any(name == "X1" for name, _ in meds)


def test_summary_keys():
    G, df = _chain_graph_and_data()
    eff = TigramiteEffects(
        G, df, X_var="X0", X_lag=2, Y_var="X2", var_names=["X0", "X1", "X2"]
    )
    eff.fit()
    s = eff.summary()
    for key in (
        "treatment",
        "treatment_lag",
        "outcome",
        "total",
        "direct",
        "indirect",
        "mediators",
        "optimal_adjustment_set",
    ):
        assert key in s


def test_indirect_equals_total_minus_direct():
    G, df = _chain_graph_and_data()
    eff = TigramiteEffects(
        G, df, X_var="X0", X_lag=2, Y_var="X2", var_names=["X0", "X1", "X2"]
    )
    eff.fit()
    s = eff.summary()
    assert abs(s["indirect"] - (s["total"] - s["direct"])) < 1e-10


def test_fit_required_before_effects():
    G, df = _chain_graph_and_data()
    eff = TigramiteEffects(
        G, df, X_var="X0", X_lag=2, Y_var="X2", var_names=["X0", "X1", "X2"]
    )
    with pytest.raises(RuntimeError, match="fit"):
        eff.total_effect()


# ---------------------------------------------------------------------------
# pathwise_effects
# ---------------------------------------------------------------------------

from causalts.effects.tigramite_effects import _find_all_paths, pathwise_effects


def test_find_all_paths_chain():
    """X0->X1->X2 chain: two paths (direct at lag 1 impossible, mediated at lag 2)."""
    G, _ = _chain_graph_and_data()
    # Remove self-loops for clarity in path-finding
    G_noself = G.copy()
    for i in range(3):
        G_noself[i, i, :] = 0
    paths = _find_all_paths(G_noself, X_idx=0, Y_idx=2, max_path_lag=6)
    # Only path: X0 -> X1 -> X2 at total_lag=2 (two lag-1 edges)
    assert len(paths) == 1
    assert paths[0]["total_lag"] == 2
    assert paths[0]["nodes"] == [0, 1, 2]


def test_find_all_paths_direct():
    """Direct edge: single path at total_lag = edge lag."""
    G, _ = _chain_graph_and_data()
    paths = _find_all_paths(G, X_idx=0, Y_idx=1, max_path_lag=6)
    # X0->X1 at lag 1 is direct
    assert any(p["total_lag"] == 1 and p["nodes"] == [0, 1] for p in paths)


def test_find_all_paths_no_connection():
    """Disconnected variables yield no paths."""
    G = np.zeros((3, 3, 3), dtype=int)
    G[0, 0, 1] = 1  # self-loop only
    paths = _find_all_paths(G, X_idx=0, Y_idx=2, max_path_lag=6)
    assert paths == []


def test_pathwise_effects_chain():
    """Chain X0->X1->X2: all effect is mediated at lag 2, nothing at lag 1."""
    G, df = _chain_graph_and_data()
    result = pathwise_effects(G, df, "X0", "X2", var_names=["X0", "X1", "X2"])
    assert "paths" in result
    assert "by_lag" in result
    assert "cumulative" in result
    assert "shares" in result
    # Should find at least the X0->X1->X2 path at lag 2
    assert 2 in result["by_lag"]
    assert abs(result["by_lag"][2]["effect"] - 0.42) < 0.1


def test_pathwise_effects_shares_sum_to_one():
    G, df = _chain_graph_and_data()
    result = pathwise_effects(G, df, "X0", "X2", var_names=["X0", "X1", "X2"])
    if result["shares"]:
        assert abs(sum(result["shares"].values()) - 1.0) < 1e-10


def test_pathwise_effects_no_connection():
    """Disconnected X,Y: empty result."""
    G = np.zeros((3, 3, 3), dtype=int)
    df = pd.DataFrame(np.random.randn(100, 3), columns=["X0", "X1", "X2"])
    result = pathwise_effects(G, df, "X0", "X2", var_names=["X0", "X1", "X2"])
    assert result["paths"] == []
    assert result["cumulative"] == 0.0
