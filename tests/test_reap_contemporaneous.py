# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for REAP's contemporaneous step (``run_cdnots_gated(contemporaneous=True)``)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from causalts.grace.contemporaneous import _SplineBlockCIT, pinned_lag_knowledge
from causalts.grace.gated_discovery import run_cdnots_gated

KW = dict(
    max_lag=1, refine="shrink", grow_rounds=1, grow_transform="rank", verbose=False
)


def _lag0_chain(T=600, seed=0):
    """a AR(1); b(t) <- tanh(2 a(t)) (the one lag-0 edge); c(t) <- b(t-1)^2 / 2; d AR(1)."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((T, 4))
    X = np.zeros((T, 4))
    for t in range(1, T):
        X[t, 0] = 0.6 * X[t - 1, 0] + e[t, 0]
        X[t, 1] = np.tanh(2 * X[t, 0]) + 0.4 * X[t - 1, 1] + 0.5 * e[t, 1]
        X[t, 2] = 0.5 * X[t - 1, 1] ** 2 + 0.3 * X[t - 1, 2] + 0.5 * e[t, 2]
        X[t, 3] = 0.5 * X[t - 1, 3] + e[t, 3]
    return pd.DataFrame(X, columns=["a", "b", "c", "d"])


def _run(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_cdnots_gated(df, **{**KW, **kw})


def test_off_by_default_leaves_lag0_empty():
    r = _run(_lag0_chain())
    assert r.contemporaneous_result is None
    assert not np.asarray(r.cg_tig)[:, :, 0].any()


@pytest.mark.parametrize("test", ["spline", "parcorr"])
def test_finds_lag0_adjacency_and_keeps_lagged_edges(test):
    df = _lag0_chain()
    r0 = _run(df)
    r1 = _run(df, contemporaneous=True, contemporaneous_test=test)
    assert np.array_equal(
        np.asarray(r0.cg_tig)[:, :, 1:], np.asarray(r1.cg_tig)[:, :, 1:]
    )
    marks = r1.to_marks()
    assert marks[0, 1, 0] != "" and marks[1, 0, 0] != ""  # a - b found
    # every other lag-0 pair is absent
    off = [(i, j) for i in range(4) for j in range(4) if i != j and {i, j} != {0, 1}]
    assert all(marks[i, j, 0] == "" for i, j in off)
    # the CDNOTS+ run's lagged slice is REAP's (the knowledge pins it)
    cr = np.asarray(r1.contemporaneous_result.cg_tig)[:4, :4, 1:].astype(bool)
    assert np.array_equal(cr, np.asarray(r1.cg_tig)[:, :, 1:].astype(bool))


def test_unoriented_lag0_edges_recoverable_from_marks_and_binary():
    # Two lag-0-coupled variables with no lagged structure to orient them: o-o.
    rng = np.random.default_rng(1)
    T = 600
    a = rng.standard_normal(T)
    b = np.tanh(2 * a) + 0.5 * rng.standard_normal(T)
    df = pd.DataFrame({"a": a, "b": b, "c": rng.standard_normal(T)})
    r = _run(df, contemporaneous=True, contemporaneous_test="spline")
    m = r.to_marks()
    assert m[0, 1, 0] == "o-o" and m[1, 0, 0] == "o-o"
    assert not np.asarray(r.cg_tig)[:, :, 0].any()  # dropped from cg_tig, as CDNOTS+
    G = r.to_binary(undirected="bidirected")
    assert G[0, 1, 0] and G[1, 0, 0]
    assert np.array_equal(np.asarray(G)[:, :, 1:], np.asarray(r.cg_tig)[:, :, 1:])


def test_ci_test_instance_accepted():
    df = _lag0_chain()
    inst = _SplineBlockCIT(np.ascontiguousarray(df.to_numpy(float)))
    by_name = _run(df, contemporaneous=True, contemporaneous_test="spline")
    by_inst = _run(df, contemporaneous=True, contemporaneous_test=inst)
    assert np.array_equal(by_name.to_marks(), by_inst.to_marks())


@pytest.mark.parametrize(
    "kw, err, match",
    [
        (dict(refine="gates"), ValueError, "needs refine='shrink'"),
        (dict(contemporaneous_alpha=0.0), ValueError, "contemporaneous_alpha"),
        (dict(contemporaneous_test="nope"), ValueError, "Unknown contemporaneous_test"),
        (dict(contemporaneous_test=None), TypeError, "test name or a CI test"),
        (dict(contemporaneous_test=_SplineBlockCIT), TypeError, "not a class"),
    ],
)
def test_options_validated_before_skeleton(monkeypatch, kw, err, match):
    import causalts.cdnots.phase3_utils as p3

    def skeleton_ran(*a, **k):
        raise AssertionError("skeleton ran before validation")

    monkeypatch.setattr(p3, "run_cdnots", skeleton_ran)
    with pytest.raises(err, match=match):
        run_cdnots_gated(_lag0_chain(T=100), **{**KW, "contemporaneous": True, **kw})


def test_pinned_knowledge_rules():
    lagged = np.zeros((2, 2, 3), dtype=bool)
    lagged[0, 1, 1] = True  # a(t-1) -> b(t)
    bk = pinned_lag_knowledge(lagged, max_lag=2, n_c_cols=1)  # 3 nodes per lag slice
    req, forb = bk._required, bk._forbidden
    assert ("X3", "X1") in req  # a(t-1) -> b(t)
    assert ("X6", "X4") in req  # shifted copy a(t-2) -> b(t-1)
    assert ("X4", "X0") in forb and ("X0", "X4") in forb  # b(t-1) -> a(t), both ways
    assert ("X6", "X1") in forb  # a(t-2) -> b(t): not selected
    names = {n for pair in req | forb for n in pair}
    assert not names & {"X2", "X5", "X8"}  # C columns stay open
    assert not any(int(n[1:]) < 3 and int(m[1:]) < 3 for n, m in forb)  # lag 0 open


def test_spline_cit_symmetric_and_detects_dependence():
    rng = np.random.default_rng(0)
    n = 800
    z = rng.standard_normal(n)
    # Gentle curvature: on wiggly effects (e.g. sin(2z)) the fixed spline basis
    # leaves misfit and the test over-rejects, as documented for the shrink.
    x = np.tanh(2 * z) + 0.3 * rng.standard_normal(n)
    y = z**2 + 0.3 * rng.standard_normal(n)
    w = np.tanh(2 * x) + 0.3 * rng.standard_normal(n)
    t = _SplineBlockCIT(np.column_stack([x, y, z, w]))
    assert t(0, 1, [2])[0] == t(1, 0, [2])[0]
    assert t(0, 1, [2])[0] > 0.01  # x _||_ y | z
    assert t(0, 1, [])[0] < 1e-3  # x, y dependent through z
    assert t(0, 3, [2])[0] < 1e-6  # x -> w


def test_cli_contemporaneous_writes_marks(tmp_path):
    import json

    from click.testing import CliRunner

    from causalts.cli import main

    csv = tmp_path / "data.csv"
    _lag0_chain(T=400).to_csv(csv, index=False)
    out = tmp_path / "out"
    result = CliRunner().invoke(
        main,
        ["-o", str(out), "-q", "discover", str(csv), "--algorithm", "grace"]
        + ["--max-lag", "1", "--refine", "shrink", "--contemporaneous"],
    )
    assert result.exit_code == 0, result.output
    run_dir = next(out.glob("*/"))
    marks = np.load(run_dir / "edge_marks.npy")
    assert marks.shape == (4, 4, 2) and marks[0, 1, 0] != ""
    assert json.loads((run_dir / "summary.json").read_text())["contemporaneous"] is True


@pytest.mark.parametrize("args", [["--algorithm", "cdnots"], ["--algorithm", "grace"]])
def test_cli_contemporaneous_needs_grace_shrink(tmp_path, args):
    from click.testing import CliRunner

    from causalts.cli import main

    csv = tmp_path / "data.csv"
    _lag0_chain(T=100).to_csv(csv, index=False)
    result = CliRunner().invoke(
        main,
        ["-o", str(tmp_path / "o"), "-q", "discover", str(csv), "--max-lag", "1"]
        + args
        + ["--contemporaneous"],
    )
    assert result.exit_code != 0 and "--refine shrink" in result.output


def test_graph_replacement_drops_stale_lag0():
    r = _run(_lag0_chain(), contemporaneous=True)
    G = np.array(r.cg_tig, copy=True)
    G[:, :, 0] = 0
    f = r._with_graph(G)  # what tetrad_filter / pds_filter return
    assert f.contemporaneous_result is None
    assert (f.to_marks()[:, :, 0] == "").all()
    assert r.contemporaneous_result is not None  # the original is untouched


def test_result_pickled_before_contemporaneous_still_renders():
    r = _run(_lag0_chain())
    del r.__dict__["contemporaneous_result"]  # as unpickled from an older version
    assert r.to_marks().shape == r.cg_tig.shape
    assert np.array_equal(r.to_binary(undirected="drop"), r.cg_tig)
    assert "contemporaneous" not in repr(r)


@pytest.mark.parametrize(
    "kw, want_c",
    [
        (dict(), True),  # own skeleton, include_C=True: the shrink uses C
        (dict(skeleton="ones"), False),  # supplied skeleton: the shrink never sees C
        (dict(skeleton="ones", include_C=False, include_C_in_model=True), True),
    ],
)
def test_lag0_step_conditions_on_c_exactly_when_the_shrink_does(
    monkeypatch, kw, want_c
):
    import causalts.grace.contemporaneous as cm

    seen = {}
    real = cm.contemporaneous_edges

    def spy(df, lagged, max_lag, **k):
        seen.update(k, df=df)
        return real(df, lagged, max_lag, **k)

    monkeypatch.setattr(cm, "contemporaneous_edges", spy)
    df = _lag0_chain(T=300)
    if kw.get("skeleton") == "ones":
        kw = {**kw, "skeleton": np.ones((4, 4, 2), dtype=bool)}
    _run(df, contemporaneous=True, **kw)
    assert seen["include_C"] is want_c


def test_lag0_step_sees_discrete_columns_rank_transformed(monkeypatch):
    import causalts.grace.contemporaneous as cm

    seen = {}
    real = cm.contemporaneous_edges

    def spy(df, lagged, max_lag, **k):
        seen["df"] = df
        return real(df, lagged, max_lag, **k)

    monkeypatch.setattr(cm, "contemporaneous_edges", spy)
    df = _lag0_chain(T=300)
    df["k"] = np.random.default_rng(0).integers(0, 4, len(df)).astype(float)
    _run(df, contemporaneous=True)
    k = seen["df"]["k"].to_numpy()
    assert not set(np.unique(k)) <= {0.0, 1.0, 2.0, 3.0}  # ranked, not raw codes


# Full edge_pvalue / edge_strength (shape (4, 4, 3), C order) of the default run below,
# computed before the contemporaneous option existed.
_FROZEN_PVALUE = [
    1.0,
    9.306410801383835e-21,
    1.0,
    1.0,
    3.998393995755494e-05,
    1.0,
    1.0,
    0.25213347568195327,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.00022804873175051872,
    0.2013250517653856,
    1.0,
    4.856595267399401e-53,
    0.10873645656463923,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    4.2833112650410925e-20,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    8.303552027577752e-17,
    1.0,
]

_FROZEN_STRENGTH = [
    0.0,
    0.30649902241341714,
    0.0,
    0.0,
    0.09979679887815006,
    0.0,
    0.0,
    0.028419291222884786,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.08701708998493361,
    0.030243764839447733,
    0.0,
    0.5971987713656898,
    0.036725055272686706,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.30501392206348515,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.25996149291956167,
    0.0,
]


def test_default_output_matches_pre_contemporaneous_code():
    # With the default contemporaneous=False every output array is unchanged.
    r = _run(_lag0_chain(T=300), max_lag=2, grow_rounds=2)
    G = np.asarray(r.cg_tig)
    assert G.shape == (4, 4, 3)
    assert np.argwhere(G.astype(bool)).tolist() == [
        [0, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
        [1, 2, 1],
        [2, 2, 1],
        [3, 3, 1],
    ]
    for got, want in (
        (r.edge_pvalue, _FROZEN_PVALUE),
        (r.edge_strength, _FROZEN_STRENGTH),
    ):
        got = np.asarray(got)
        assert got.shape == (4, 4, 3) and got.dtype == np.float64
        np.testing.assert_allclose(got.ravel(), want, rtol=1e-9, atol=1e-15)


def test_spline_cit_ignores_an_undefined_direction():
    # x has 4 events, so its levels merge and it gets no spline columns: the
    # x-block test is undefined (1.0) and must not swamp the y-block test.
    rng = np.random.default_rng(0)
    n = 600
    x = np.zeros(n)
    x[[50, 200, 350, 500]] = 1.0
    y = 4.0 * x + rng.standard_normal(n)
    t = _SplineBlockCIT(np.column_stack([x, y, np.ones(n)]))
    assert t(0, 1, [])[0] < 1e-10
    assert t(0, 2, [])[0] == 1.0  # both directions undefined (constant column)


def test_lag0_spline_test_gets_ranked_data_under_rff_shrink(monkeypatch):
    import causalts.grace.contemporaneous as cm

    seen = {}
    real = cm.contemporaneous_edges

    def spy(df, lagged, max_lag, **k):
        seen["df"] = df
        return real(df, lagged, max_lag, **k)

    monkeypatch.setattr(cm, "contemporaneous_edges", spy)
    df = _lag0_chain(T=300)
    df["k"] = np.random.default_rng(0).integers(0, 4, len(df)).astype(float)
    _run(df, contemporaneous=True, shrink_basis="rff")
    assert not set(np.unique(seen["df"]["k"].to_numpy())) <= {0.0, 1.0, 2.0, 3.0}
    seen.clear()
    _run(df, contemporaneous=True, shrink_basis="rff", contemporaneous_test="parcorr")
    k = seen["df"]["k"].to_numpy()  # other tests keep the codes, as CDNOTS does
    assert set(np.unique(k)) <= {0.0, 1.0, 2.0, 3.0}


def test_lag0_spline_on_raw_data_warns():
    with pytest.warns(UserWarning, match="contemporaneous_test='spline' with data_tr"):
        run_cdnots_gated(
            _lag0_chain(T=300),
            **{**KW, "shrink_basis": "rff", "data_transform": None},
            contemporaneous=True,
        )


def test_spline_cit_keeps_c_columns_linear():
    # Two lag slices of (x, y, C): C columns enter as one linear column, as in the
    # shrink; the variables get spline blocks.
    rng = np.random.default_rng(0)
    n = 400
    t = np.linspace(-1, 1, n)
    D = np.column_stack([rng.standard_normal(n), rng.standard_normal(n), t] * 2)
    cit = _SplineBlockCIT(D, n_c_cols=1)
    cit.set_lag_structure(n_vars=3, lags=[0, 1])
    assert cit._block(D, 2).shape[1] == 1 and cit._block(D, 5).shape[1] == 1
    assert cit._block(D, 0).shape[1] > 1 and cit._block(D, 4).shape[1] > 1


def test_spline_cit_conditions_on_every_level_of_a_rare_discrete_column():
    # A 4-event regime column: as a tested block its levels merge (no columns),
    # as a conditioning block it keeps its indicator, so it can separate x and y.
    rng = np.random.default_rng(0)
    n = 600
    z = np.zeros(n)
    z[[50, 200, 350, 500]] = 1.0
    D = np.column_stack([rng.standard_normal(n), rng.standard_normal(n), z])
    cit = _SplineBlockCIT(D)
    assert cit._block(D, 2).shape[1] == 0
    assert cit._block(D, 2, nuisance=True).shape[1] == 1


def test_spline_cit_cache_matches_fresh_computation():
    rng = np.random.default_rng(1)
    n = 500
    z = rng.standard_normal(n)
    D = np.column_stack(
        [np.tanh(z) + rng.standard_normal(n), z**2 + rng.standard_normal(n), z]
    )
    cached = _SplineBlockCIT(D)
    calls = [(0, 1, [2]), (0, 1, []), (1, 2, [0]), (0, 1, [2])]
    got = [cached(*c)[0] for c in calls]
    want = [_SplineBlockCIT(D)(*c)[0] for c in calls]  # a new instance per call
    assert got == want
    # Replacing the data (as CDNOTS does) must invalidate the cache.
    D2 = D[::-1].copy()
    cached.data = D2
    assert cached(0, 1, [2])[0] == _SplineBlockCIT(D2)(0, 1, [2])[0]
