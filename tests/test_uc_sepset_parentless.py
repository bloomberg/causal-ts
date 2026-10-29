# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""A C node has no parents, so the collider step never orients into it."""

import numpy as np
import pytest

from causalts.cdnots.skeleton_discovery import c_lag_cnst, initialize_graph
from causalts.cdnots.uc_sepset import uc_sepset


def _x_c_y():
    # X0 = 0, X1 = 1, C = 2 (lag 0 only): X0 - C - X1, X0 and X1 non-adjacent,
    # with C -> X oriented by c_lag_cnst. The empty sepset makes C a collider.
    cg = initialize_graph(np.zeros((5, 3)), None)
    cg.G.remove_edge(cg.G.get_edge(cg.G.nodes[0], cg.G.nodes[1]))
    cg = c_lag_cnst(cg, 0, include_C=True, num_c_cols=1)
    cg.sepset[0, 1] = cg.sepset[1, 0] = [()]
    return cg


def _c_to_x(g, x):
    return g[2, x] == -1 and g[x, 2] == 1


@pytest.mark.parametrize("priority", [0, 1, 2])
def test_no_arrowhead_into_c(priority):
    out = uc_sepset(_x_c_y(), priority=priority, parentless=(2,))
    g = out.G.graph
    assert _c_to_x(g, 0) and _c_to_x(g, 1)


def test_without_parentless_c_is_a_collider_middle():
    # The old behaviour, kept for nodes not declared parentless.
    g = uc_sepset(_x_c_y(), priority=1).G.graph
    assert not (_c_to_x(g, 0) and _c_to_x(g, 1))


@pytest.mark.parametrize("priority", [0, 1])
@pytest.mark.parametrize("fn_name", ["run_cdnots", "run_cdnots_plus"])
@pytest.mark.parametrize("seed", range(6))
def test_cdnots_never_orients_into_c(seed, fn_name, priority):
    import pandas as pd

    from causalts.cdnots import phase3_utils
    from causalts.ci_tests.parcorr_gpu import ParCorrGPU

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((400, 8))
    for t in range(1, 400):
        for j in range(8):
            x[t, j] += 0.4 * x[t - 1, j] + (0.5 * x[t - 1, j - 1] if j else 0)
    # A trend on every other series: C adjacent to non-adjacent pairs.
    x += np.linspace(0, 3, 400)[:, None] * (np.arange(8) % 2 == 0)
    res = getattr(phase3_utils, fn_name)(
        pd.DataFrame(x),
        ParCorrGPU(np.zeros((2, 2)), device="cpu"),
        include_C=True,
        priority=priority,
        alpha=0.05,
        show_progress=False,
    )
    # An arrowhead into C would show as x-x (conflict or X -> C) or <--.
    assert not np.isin(res.to_marks()[-1, :, 0], ["x-x", "<--"]).any()
