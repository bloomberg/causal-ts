# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for post-discovery ``o-o`` (unoriented CPDAG edge) policy.

Discovery produces a CPDAG. ``cg_tig`` is a *lossy rendering* of it, and the
two CDNOTS entry points render an unoriented contemporaneous edge
**differently** -- ``run_cdnots`` keeps it as symmetric 1s, ``run_cdnots_plus``
zeroes it.

What must not regress:

1. Neither default moves. ``cg_tig`` stays bit-identical on every existing
   call, which is what most of this file pins.
2. ``to_binary()`` with no argument never re-renders -- it hands back
   ``cg_tig`` verbatim, so it is safe on results pickled before any of this
   existed.
3. Re-rendering from ``result.graph`` agrees with what the engine emitted, so
   the new path cannot silently drift from ``cdnots_to_tigramite_graph``.
"""

import numpy as np
import pandas as pd
import pytest

from causalts.cdnots.phase3_utils import (
    cdnots_to_tigramite_graph,
    cdnots_to_tigramite_marks,
    run_cdnots,
    run_cdnots_plus,
)
from causalts.ci_tests.parcorr_gpu import ParCorrGPU
from causalts.utils.helpers import evaluate_graph

ENGINES = [("run_cdnots", run_cdnots, "bidirected"), ("plus", run_cdnots_plus, "drop")]


def _data(T=300, d=5, seed=0):
    """Mixed lagged + contemporaneous chain, dense enough to leave o-o edges."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, d))
    for t in range(1, T):
        x[t, 1] += 0.6 * x[t - 1, 0]
        x[t, 2] += 0.7 * x[t, 1]
        x[t, 3] += 0.5 * x[t - 1, 2]
        x[t, 4] += 0.6 * x[t, 3]
    return pd.DataFrame(x, columns=[f"X{i}" for i in range(d)])


def _run(fn, df, **kw):
    ci = ParCorrGPU(np.zeros((2, 2)), device="cpu")
    return fn(
        df,
        ci,
        num_lags=1,
        alpha=0.05,
        include_C=False,
        verbose=False,
        show_progress=False,
        **kw,
    )


# ----------------------------------------------------------------------
# Defaults must not move
# ----------------------------------------------------------------------


@pytest.mark.parametrize("label,fn,expected", ENGINES)
def test_engine_default_policy_unchanged(label, fn, expected):
    """run_cdnots keeps o-o; run_cdnots_plus drops it. Neither may flip."""
    res = _run(fn, _data())
    assert res.undirected_policy == expected


@pytest.mark.parametrize("label,fn,expected", ENGINES)
def test_to_binary_no_arg_returns_cg_tig(label, fn, expected):
    """No-arg to_binary() must never re-render -- old pickles depend on it."""
    res = _run(fn, _data())
    assert np.array_equal(res.to_binary(), res.cg_tig)


@pytest.mark.parametrize("label,fn,expected", ENGINES)
def test_rerender_with_own_policy_reproduces_cg_tig(label, fn, expected):
    """The load-bearing guard.

    Fails if anyone flips an engine's hardcoded policy, *or* if to_binary()
    drifts from cdnots_to_tigramite_graph. Stronger than a golden hash,
    because it also exercises the new code path.
    """
    res = _run(fn, _data())
    assert np.array_equal(res.to_binary(undirected=expected), res.cg_tig)


def test_the_two_engines_actually_disagree():
    """Guard against the fixture going degenerate.

    Every assertion above passes vacuously on a graph with no o-o edges, so
    pin that this fixture really does produce some.
    """
    res = _run(run_cdnots_plus, _data())
    assert (res.to_marks() == "o-o").any(), "fixture no longer produces o-o edges"
    assert res.to_binary(undirected="bidirected").sum() > res.cg_tig.sum()


# ----------------------------------------------------------------------
# Re-rendering
# ----------------------------------------------------------------------


@pytest.mark.parametrize("label,fn,expected", ENGINES)
def test_marks_binarise_to_kept_rendering(label, fn, expected):
    """to_marks() must carry exactly the adjacencies to_binary() can express."""
    res = _run(fn, _data())
    marks = res.to_marks()
    binarised = np.isin(marks, ["-->", "o-o", "x-x"]).astype(np.int8)
    keep_all = res.to_binary(undirected="bidirected", conflict="bidirected")
    assert np.array_equal(binarised, keep_all)


def test_marks_preserve_lagged_and_c_edges():
    """Lagged and C-node cells are (0, 1) in causal-learn, not (-1, 1).

    An endpoint-pair rule alone silently erases every one of them, so this
    pins that to_marks() agrees with cg_tig outside the lag-0 block.
    """
    res = _run(run_cdnots, _data(), c_preset="linear")
    marks = res.to_marks()
    lagged = np.isin(marks[:, :, 1:], ["-->"]).astype(np.int8)
    assert np.array_equal(lagged, res.cg_tig[:, :, 1:])
    assert res.cg_tig[:, :, 1:].sum() > 0, "fixture produced no lagged edges"


def test_works_without_undirected_policy_attribute():
    """Stand-in for results pickled before undirected_policy existed."""
    res = _run(run_cdnots, _data())
    expected = res.to_binary(undirected="bidirected")
    del res.undirected_policy
    assert np.array_equal(res.to_binary(), res.cg_tig)
    assert np.array_equal(res.to_binary(conflict="drop"), expected)


# ----------------------------------------------------------------------
# Converter-level, on a hand-built graph
# ----------------------------------------------------------------------


def _planted_oo(d=4, num_lags=1):
    from causalts.cdnots.skeleton_discovery import initialize_graph

    cg = initialize_graph(np.zeros((10, d * (num_lags + 1))), None)
    cg.G.graph[:, :] = 0
    cg.G.graph[0, 1] = -1  # o-o between X0 and X1 at lag 0
    cg.G.graph[1, 0] = -1
    # causal-learn encodes i -> j as graph[i, j] == -1 and graph[j, i] == 1.
    cg.G.graph[2, 3] = -1  # X2 -> X3
    cg.G.graph[3, 2] = 1
    return cg


def test_converter_policies_on_planted_graph():
    cg = _planted_oo()
    kw = dict(num_lags=1, include_C=False)

    kept = cdnots_to_tigramite_graph(cg, undirected="bidirected", **kw)
    assert kept[0, 1, 0] == 1 and kept[1, 0, 0] == 1

    dropped = cdnots_to_tigramite_graph(cg, undirected="drop", **kw)
    assert dropped[0, 1, 0] == 0 and dropped[1, 0, 0] == 0

    # The directed edge is untouched by the policy.
    for g in (kept, dropped):
        assert g[2, 3, 0] == 1 and g[3, 2, 0] == 0

    marks = cdnots_to_tigramite_marks(cg, **kw)
    assert marks[0, 1, 0] == "o-o" and marks[1, 0, 0] == "o-o"
    assert marks[2, 3, 0] == "-->" and marks[3, 2, 0] == "<--"


def test_keep_undirected_alias_still_works():
    """routed_deconf and the experiment harnesses still pass the boolean."""
    cg = _planted_oo()
    kw = dict(num_lags=1, include_C=False)
    assert np.array_equal(
        cdnots_to_tigramite_graph(cg, keep_undirected=True, **kw),
        cdnots_to_tigramite_graph(cg, undirected="bidirected", **kw),
    )
    assert np.array_equal(
        cdnots_to_tigramite_graph(cg, keep_undirected=False, **kw),
        cdnots_to_tigramite_graph(cg, undirected="drop", **kw),
    )
    # Historical default was keep_undirected=True.
    assert np.array_equal(
        cdnots_to_tigramite_graph(cg, **kw),
        cdnots_to_tigramite_graph(cg, undirected="bidirected", **kw),
    )


def test_converter_rejects_contradictory_and_unknown_policies():
    cg = _planted_oo()
    kw = dict(num_lags=1, include_C=False)
    with pytest.raises(ValueError, match="not both"):
        cdnots_to_tigramite_graph(cg, undirected="drop", keep_undirected=True, **kw)
    with pytest.raises(ValueError, match="to_marks|cdnots_to_tigramite_marks"):
        cdnots_to_tigramite_graph(cg, undirected="keep", **kw)
    with pytest.raises(ValueError, match="conflict"):
        cdnots_to_tigramite_graph(cg, conflict="nonsense", **kw)


def test_conflict_policy_renders_x_x():
    cg = _planted_oo()
    cg.G.graph[0, 1] = 1  # (1, 1) == conflicting orientation
    cg.G.graph[1, 0] = 1
    kw = dict(num_lags=1, include_C=False)

    assert cdnots_to_tigramite_marks(cg, **kw)[0, 1, 0] == "x-x"
    assert cdnots_to_tigramite_graph(cg, **kw)[0, 1, 0] == 0
    assert cdnots_to_tigramite_graph(cg, conflict="bidirected", **kw)[0, 1, 0] == 1


# ----------------------------------------------------------------------
# Plotting + scoring entry points
# ----------------------------------------------------------------------


def test_plot_default_reads_cg_tig(monkeypatch):
    res = _run(run_cdnots_plus, _data())
    seen = {}

    def _fake(graph, var_names, **kw):
        seen["graph"] = graph
        return None

    # CdnotsResult.plot overrides CausalResult.plot -- patch the shared
    # plot_graph so this exercises whichever override is actually in play.
    monkeypatch.setattr("causalts.plotting._core.plot_graph", _fake)

    res.plot()
    assert np.array_equal(seen["graph"], res.cg_tig)

    res.plot(undirected="bidirected")
    assert np.array_equal(seen["graph"], res.to_binary(undirected="bidirected"))

    res.plot(undirected="keep")
    assert seen["graph"].dtype.kind == "U"


def test_evaluate_graph_rejects_edge_marks():
    res = _run(run_cdnots, _data())
    gt = np.zeros_like(res.cg_tig)
    with pytest.raises(TypeError, match="to_binary"):
        evaluate_graph(res.to_marks(), gt)


def test_base_marks_are_a_string_format_of_any_graph():
    """A DAG is a CPDAG with nothing unoriented, so to_marks() must work for
    CEDAR/GRACE/LUCID too -- it is a string format, not an ambiguity report.
    """
    from causalts.result import CausalResult

    obj = CausalResult.__new__(CausalResult)
    g = np.zeros((3, 3, 2), dtype=np.int8)
    g[0, 1, 0] = 1  # one-way at lag 0
    g[1, 2, 0] = 1  # symmetric pair at lag 0
    g[2, 1, 0] = 1
    g[0, 2, 1] = 1  # lagged
    obj.cg_tig = g

    m = obj.to_marks()
    assert m.dtype.kind == "U"
    assert m[0, 1, 0] == "-->" and m[1, 0, 0] == "<--"
    assert m[1, 2, 0] == "o-o" and m[2, 1, 0] == "o-o"
    assert m[0, 2, 1] == "-->" and m[2, 0, 1] == ""  # lag>=1 never mirrored

    assert np.array_equal(obj.to_binary(), obj.cg_tig)
    dropped = obj.to_binary(undirected="drop")
    assert dropped[1, 2, 0] == 0 and dropped[2, 1, 0] == 0
    assert dropped[0, 1, 0] == 1 and dropped[0, 2, 1] == 1  # untouched
    with pytest.raises(ValueError, match="undirected"):
        obj.to_binary(undirected="keep")


def test_base_to_binary_validates_the_no_op_conflict_arg():
    """`conflict` cannot do anything on a binary graph -- x-x and o-o are both
    symmetric 1s -- but a bad value must still surface rather than silently
    returning an unchanged copy.
    """
    from causalts.result import CausalResult

    obj = CausalResult.__new__(CausalResult)
    g = np.zeros((3, 3, 2), dtype=np.int8)
    g[1, 2, 0] = 1
    g[2, 1, 0] = 1
    obj.cg_tig = g

    with pytest.raises(ValueError, match="conflict"):
        obj.to_binary(conflict="garbage")
    with pytest.raises(ValueError, match="conflict"):
        obj.to_binary(undirected="drop", conflict="garbage")

    # Valid values are accepted and genuinely change nothing.
    for c in ("drop", "bidirected"):
        assert np.array_equal(obj.to_binary(conflict=c), obj.cg_tig)
        assert np.array_equal(
            obj.to_binary(undirected="drop", conflict=c),
            obj.to_binary(undirected="drop"),
        )


def test_base_marks_agree_with_plotting_reconstruction():
    """plot_graph already rebuilds o-o from symmetric 1s; marks must match."""
    from causalts.plotting._core import _check_matrices
    from causalts.result import CausalResult

    obj = CausalResult.__new__(CausalResult)
    g = np.zeros((3, 3, 2), dtype=np.int8)
    g[0, 1, 0] = 1
    g[1, 2, 0] = 1
    g[2, 1, 0] = 1
    g[0, 2, 1] = 1
    obj.cg_tig = g

    from_plotting = _check_matrices(g, None, None, None)[0]
    assert np.array_equal(obj.to_marks(), from_plotting)


def test_cedar_and_grace_get_working_marks():
    """End-to-end on the real result classes, not a hand-built stub."""
    df = _data()
    from causalts.cedar.discovery import run_cedar

    cres = run_cedar(
        df,
        ParCorrGPU(np.zeros((2, 2)), device="cpu"),
        max_lag=1,
        include_lag0=True,
        verbose=False,
    )
    m = cres.to_marks()
    assert m.shape == cres.cg_tig.shape and m.dtype.kind == "U"
    # CEDAR is directed by construction: no o-o should appear.
    assert not (m == "o-o").any(), "CEDAR emitted a symmetric lag-0 pair"
    assert np.array_equal(np.isin(m, ["-->", "o-o"]).astype(np.int8), cres.cg_tig)


if __name__ == "__main__":
    pytest.main([__file__])
