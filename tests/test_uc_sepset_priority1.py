# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for the PCMCI+-style conflict-abstention path (priority=1).

`uc_sepset(priority=1)` marks contradictory collider orientations as
``i <-> j`` -- the analogue of PCMCI+'s ``x-x`` conflict marker. It used to
raise ``NotImplementedError`` for any ``num_lags > 0``, because the
bi-directed marks were applied to a single node pair without being
propagated to their time-shifted copies. Two things are checked here:

1. the lagged path runs at all, and
2. the bi-directed marks are *dropped* on the way out rather than being
   emitted as two contradictory directed edges. Before
   `cdnots_to_tigramite_graph` learned about the ``(1, 1)`` case, a conflict
   marker fell through untouched and both entries were scored as edges.
"""

import numpy as np
import pandas as pd
import pytest

from causalts.cdnots.phase3_utils import run_cdnots, run_cdnots_plus
from causalts.ci_tests.parcorr_gpu import ParCorrGPU


def _data(T=400, d=6, seed=0):
    """Dense-ish contemporaneous + lagged mix, to provoke collider conflicts."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, d))
    for t in range(2, T):
        x[t, 1] += 0.7 * x[t - 1, 0]
        x[t, 2] += 0.6 * x[t - 1, 1] + 0.5 * x[t, 0]
        x[t, 3] += 0.5 * x[t - 1, 2] + 0.4 * x[t, 1]
        x[t, 4] += 0.6 * x[t - 2, 0] + 0.4 * x[t, 3]
        x[t, 5] += 0.5 * x[t - 1, 4]
    return pd.DataFrame(x, columns=[f"X{i}" for i in range(d)])


@pytest.mark.parametrize("num_lags", [1, 2, 3])
def test_priority1_runs_with_lags(num_lags):
    """priority=1 used to raise NotImplementedError whenever num_lags > 0."""
    df = _data()
    ci = ParCorrGPU(np.zeros((2, 2)), device="cpu")
    res = run_cdnots_plus(
        df,
        ci,
        num_lags=num_lags,
        include_C=False,
        alpha=0.05,
        priority=1,
        show_progress=False,
    )
    assert res.cg_tig.shape == (df.shape[1], df.shape[1], num_lags + 1)


def test_priority1_emits_no_contradictory_edge_pairs():
    """A conflict must be dropped, not rendered as i->j AND j->i."""
    df = _data()
    ci = ParCorrGPU(np.zeros((2, 2)), device="cpu")
    g = run_cdnots_plus(
        df,
        ci,
        num_lags=2,
        include_C=False,
        alpha=0.05,
        priority=1,
        show_progress=False,
    ).cg_tig
    contemp = g[:, :, 0]
    both = contemp & contemp.T
    assert not both.any(), (
        "bi-directed conflict markers leaked into the output as two "
        f"contradictory lag-0 edges: {np.argwhere(both).tolist()}"
    )


def test_priority1_only_affects_lag0():
    """Conflicts are a contemporaneous phenomenon; lagged edges must be
    identical to the default priority=2 run."""
    df = _data()
    out = {}
    for p in (1, 2):
        ci = ParCorrGPU(np.zeros((2, 2)), device="cpu")
        out[p] = run_cdnots_plus(
            df,
            ci,
            num_lags=2,
            include_C=False,
            alpha=0.05,
            priority=p,
            show_progress=False,
        ).cg_tig
    np.testing.assert_array_equal(out[1][:, :, 1:], out[2][:, :, 1:])


def test_default_priority_unchanged_by_bidirected_branch():
    """The new (1,1) branch must be inert for the default priority=2.

    Only ``run_cdnots_plus`` is checked for the absence of symmetric lag-0
    pairs: it converts with ``keep_undirected=False``. ``run_cdnots`` uses the
    ``keep_undirected=True`` default, where an o-o edge is *meant* to render as
    a symmetric pair -- see
    :func:`test_run_cdnots_still_renders_undirected_edges`.
    """
    df = _data()
    ci = ParCorrGPU(np.zeros((2, 2)), device="cpu")
    g = run_cdnots_plus(
        df, ci, num_lags=2, include_C=False, alpha=0.05, show_progress=False
    ).cg_tig
    contemp = g[:, :, 0]
    assert not (contemp & contemp.T).any()


def test_run_cdnots_still_renders_undirected_edges():
    """End-to-end guard for the snapshot fix.

    ``run_cdnots`` keeps undirected edges, so they arrive at the converter as
    (-1, -1) and leave as a symmetric (1, 1) pair -- byte-identical to a
    priority=1 conflict marker. A bi-directed branch reading the live array
    would erase them on the mirrored visit, silently dropping every o-o edge
    from plain CDNOTS's output.
    """
    df = _data()
    ci = ParCorrGPU(np.zeros((2, 2)), device="cpu")
    g = run_cdnots(
        df, ci, num_lags=2, include_C=False, alpha=0.05, show_progress=False
    ).cg_tig
    contemp = g[:, :, 0]
    assert (contemp & contemp.T).any(), (
        "plain CDNOTS emitted no undirected lag-0 edges at all; the "
        "bi-directed branch has most likely eaten them"
    )


def test_keep_undirected_survives_bidirected_branch():
    """``keep_undirected=True`` (the function default) must still render o-o
    edges as symmetric pairs.

    The conversion loop visits both (i, j) and (j, i) and writes in place. The
    o-o branch emits (1, 1), which is byte-identical to a priority=1 conflict
    marker -- so a bi-directed branch that reads the live array erases every
    undirected edge on the mirrored visit. Decisions must come from a snapshot.
    """
    from causalts.cdnots.phase3_utils import cdnots_to_tigramite_graph
    from causalts.cdnots.skeleton_discovery import initialize_graph

    d, num_lags = 4, 1
    data = np.zeros((10, d * (num_lags + 1)))
    cg = initialize_graph(data, None)
    n = cg.G.num_vars
    # Wipe to an empty graph, then plant one o-o edge at lag 0.
    cg.G.graph[:, :] = 0
    cg.G.graph[0, 1] = -1
    cg.G.graph[1, 0] = -1

    kept = cdnots_to_tigramite_graph(
        cg, num_lags=num_lags, include_C=False, keep_undirected=True
    )
    assert kept[0, 1, 0] == 1 and kept[1, 0, 0] == 1, (
        "o-o edge was erased under keep_undirected=True -- the bi-directed "
        f"branch consumed it (got {kept[0, 1, 0]}, {kept[1, 0, 0]})"
    )
    assert n == d * (num_lags + 1)

    dropped = cdnots_to_tigramite_graph(
        cg, num_lags=num_lags, include_C=False, keep_undirected=False
    )
    assert dropped[0, 1, 0] == 0 and dropped[1, 0, 0] == 0


if __name__ == "__main__":
    pytest.main([__file__])
