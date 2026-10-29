# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for REAP's effect floor (``run_cdnots_gated(refine="shrink", shrink_effect_floor=...)``)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from causalts.cdnots.phase3_utils import make_c_array
from causalts.grace.gated_discovery import (
    SHRINK_EFFECT_FLOOR_ADAPTIVE,
    run_cdnots_gated,
)
from causalts.grace.grow import prune_effect_floor

KW = dict(
    max_lag=1, refine="shrink", grow_rounds=1, grow_transform="rank", verbose=False
)


def _strong_weak(T=6000, strong=0.6, weak=0.15, seed=0):
    """b(t) = strong * a(t-1) + weak * c(t-1) + noise; a, c white.

    The weak edge's partial R^2 is about weak^2 = 0.022: above 0.01 (and highly
    significant at this T) but below 0.2 times the strong edge's (about 0.26)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, 3))
    X[1:, 1] += strong * X[:-1, 0] + weak * X[:-1, 2]
    return pd.DataFrame(X, columns=["a", "b", "c"])


def _run(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_cdnots_gated(df, **{**KW, **kw})


def test_adaptive_constants():
    assert SHRINK_EFFECT_FLOOR_ADAPTIVE == dict(floor=0.01, relative=0.2)


def test_prune_relative_threshold_scales_with_the_strongest_edge():
    X = _strong_weak().to_numpy()
    G = np.zeros((3, 3, 2), dtype=np.int8)
    G[0, 1, 1] = G[2, 1, 1] = 1
    G[1, 1, 1] = 1  # self-loop: never pruned
    fixed = prune_effect_floor(X, G, 1, 0.01, basis="spline")
    rel = prune_effect_floor(X, G, 1, 0.01, basis="spline", relative=0.2)
    assert fixed[2, 1, 1] == 1  # 0.022 >= 0.01
    assert rel[0, 1, 1] == 1 and rel[2, 1, 1] == 0 and rel[1, 1, 1] == 1
    # floor 0 with a relative part still prunes; both 0 is the identity
    assert prune_effect_floor(X, G, 1, 0.0, basis="spline", relative=0.2)[2, 1, 1] == 0
    assert np.array_equal(prune_effect_floor(X, G, 1, 0.0), G)


def test_prune_single_parent_only_meets_the_floor():
    # one cross parent: relative * strongest is never above its own effect
    X = _strong_weak(weak=0.0).to_numpy()
    G = np.zeros((3, 3, 2), dtype=np.int8)
    G[0, 1, 1] = 1
    assert prune_effect_floor(X, G, 1, 0.01, relative=0.9)[0, 1, 1] == 1


@pytest.mark.parametrize(
    "kw",
    [
        dict(effect_floor=-0.1),
        dict(effect_floor=1.0),
        dict(relative=-0.1),
        dict(relative=1.0),
    ],
)
def test_prune_rejects_out_of_range(kw):
    G = np.zeros((2, 2, 2), dtype=np.int8)
    args = dict(effect_floor=0.01, relative=0.0) | kw
    with pytest.raises(ValueError, match="effect_floor|relative"):
        prune_effect_floor(np.zeros((50, 2)), G, 1, args.pop("effect_floor"), **args)


def test_off_by_default():
    df = _strong_weak()
    r = _run(df)
    assert (
        r.effect_floor is None and r.n_effect_pruned is None and r.effect_pruned is None
    )
    assert np.array_equal(r.cg_tig, _run(df, shrink_effect_floor=None).cg_tig)


def test_adaptive_drops_the_negligible_edge_and_matches_prune():
    df = _strong_weak()
    plain = _run(df)
    assert plain.cg_tig[0, 1, 1] and plain.cg_tig[2, 1, 1]
    r = _run(df, shrink_effect_floor="adaptive")
    assert r.cg_tig[0, 1, 1] and not r.cg_tig[2, 1, 1]
    assert r.effect_floor == "adaptive" and r.n_effect_pruned == int(
        r.effect_pruned.sum()
    )
    assert r.effect_pruned[2, 1, 1] and not r.effect_pruned[0, 1, 1]
    # the same as pruning the plain result with the shrink's blocks and linear C
    expected = prune_effect_floor(
        df.to_numpy(float),
        plain.cg_tig,
        1,
        0.01,
        exog=make_c_array(len(df), "linear"),
        basis="spline",
        nuisance_basis="spline",
        relative=0.2,
    )
    assert np.array_equal(r.cg_tig, expected)
    # the shrink's own scores are untouched
    assert np.array_equal(r.edge_pvalue, plain.edge_pvalue)


def test_fixed_float_floor():
    df = _strong_weak()
    assert _run(df, shrink_effect_floor=0.01).cg_tig[2, 1, 1]  # 0.022 >= 0.01
    assert not _run(df, shrink_effect_floor=0.05).cg_tig[2, 1, 1]


def test_floor_runs_before_the_lag0_step():
    df = _strong_weak(T=1500)
    plain = _run(df, contemporaneous=True)
    r = _run(df, shrink_effect_floor=0.05, contemporaneous=True)
    assert plain.cg_tig[2, 1, 1] and not r.cg_tig[2, 1, 1]
    # the lagged graph CDNOTS+ pinned is the pruned one (lagged edges are required
    # or forbidden in its background knowledge, so they come back unchanged)
    assert plain.contemporaneous_result.cg_tig[2, 1, 1]
    assert not r.contemporaneous_result.cg_tig[2, 1, 1]
    assert r.contemporaneous_result.cg_tig[0, 1, 1]


@pytest.mark.parametrize("bad", ["fixed", -0.1, 1.0, True, np.bool_(True), [0.1]])
def test_rejects_bad_value(bad):
    df = _strong_weak(T=100)
    with pytest.raises(ValueError, match="shrink_effect_floor"):
        run_cdnots_gated(df, **{**KW, "shrink_effect_floor": bad})


@pytest.mark.parametrize("value", [np.float32(0.05), np.int64(0)])
def test_accepts_numpy_scalars(value):
    r = _run(_strong_weak(T=600), shrink_effect_floor=value)
    assert r.effect_floor == value


def test_rejects_interactions():
    df = _strong_weak(T=100)
    with pytest.raises(ValueError, match="shrink_interactions"):
        run_cdnots_gated(
            df, **{**KW, "shrink_effect_floor": "adaptive", "shrink_interactions": True}
        )


def test_rejects_gates():
    df = _strong_weak(T=100)
    with pytest.raises(ValueError, match="refine='shrink'"):
        run_cdnots_gated(df, max_lag=1, shrink_effect_floor="adaptive", verbose=False)


def test_cli_shrink_effect_floor(tmp_path):
    import json

    from click.testing import CliRunner

    from causalts.cli import main

    csv = tmp_path / "data.csv"
    _strong_weak(T=3000).to_csv(csv, index=False)
    out = tmp_path / "out"
    result = CliRunner().invoke(
        main,
        ["-o", str(out), "-q", "discover", str(csv), "--algorithm", "grace"]
        + ["--max-lag", "1", "--refine", "shrink", "--shrink-effect-floor", "adaptive"],
    )
    assert result.exit_code == 0, result.output
    run_dir = next(out.glob("*/"))
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["shrink_effect_floor"] == "adaptive"
    assert summary["n_effect_pruned"] == int(
        np.load(run_dir / "effect_pruned.npy").sum()
    )


@pytest.mark.parametrize(
    "args,msg",
    [
        (["--refine", "gates", "--shrink-effect-floor", "adaptive"], "--refine shrink"),
        (
            ["--refine", "shrink", "--shrink-effect-floor", "big"],
            "'adaptive' or a number",
        ),
        (
            ["--refine", "shrink", "--shrink-effect-floor", "1.5"],
            "'adaptive' or a number",
        ),
    ],
)
def test_cli_rejects_bad_shrink_effect_floor(tmp_path, args, msg):
    from click.testing import CliRunner

    from causalts.cli import main

    csv = tmp_path / "data.csv"
    _strong_weak(T=100).to_csv(csv, index=False)
    result = CliRunner().invoke(
        main,
        [
            "-o",
            str(tmp_path / "o"),
            "-q",
            "discover",
            str(csv),
            "--max-lag",
            "1",
            "--algorithm",
            "grace",
        ]
        + args,
    )
    assert result.exit_code != 0 and msg in result.output
