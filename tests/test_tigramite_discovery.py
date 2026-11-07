# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the tigramite method dispatcher ``run_tigramite``."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tigramite")

from causalts.tigramite_discovery import run_tigramite  # noqa: E402


def _lagged_chain(T=500, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, 3))
    for t in range(1, T):
        x[t, 1] += 0.7 * x[t - 1, 0]
        x[t, 2] += 0.7 * x[t - 1, 1]
    return pd.DataFrame(x, columns=["A", "B", "C"])


def test_pc_stable_returns_the_discovered_parents():
    """run_pc_stable returns a parents dict, which was read as no edges."""
    G, info = run_tigramite(
        _lagged_chain(), method="pc_stable", tau_max=2, ci_test="parcorr"
    )
    assert G.shape == (3, 3, 3)
    assert G[0, 1, 1] and G[1, 2, 1]
    assert not G[:, :, 0].any()  # PC-stable finds lagged parents only
    assert info["p_matrix"].shape == (3, 3, 3)
    assert info["val_matrix"].shape == (3, 3, 3)
    # Each reported parent carries its PC-stable p-value, selected at pc_alpha.
    assert info["pc_alpha"] == 0.2
    assert (info["p_matrix"][G.astype(bool)] <= 0.2).all()


def test_pc_stable_reports_per_target_alpha_when_optimised():
    G, info = run_tigramite(
        _lagged_chain(), method="pc_stable", tau_max=2, pc_alpha=[0.05, 0.2]
    )
    assert set(info["pc_alpha"]) == {0, 1, 2}
    assert set(info["pc_alpha"].values()) <= {0.05, 0.2}
    for j, alpha in info["pc_alpha"].items():
        assert (info["p_matrix"][:, j][G[:, j].astype(bool)] <= alpha).all()


def test_pc_stable_rejects_zero_lag_parents():
    with pytest.raises(ValueError, match="zero-lag"):
        run_tigramite(_lagged_chain(T=100), method="pc_stable", tau_max=1, tau_min=0)


def test_pc_stable_rejects_a_returned_zero_lag_parent(monkeypatch):
    """Lag-0 parents reached by a path other than tau_min are not written as
    directed edges."""
    from tigramite.pcmci import PCMCI

    monkeypatch.setattr(
        PCMCI, "run_pc_stable", lambda self, **kw: {0: [(1, 0)], 1: [], 2: []}
    )
    with pytest.raises(ValueError, match="zero-lag parent"):
        run_tigramite(_lagged_chain(T=100), method="pc_stable", tau_max=1)
