# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""run_grace / run_reap: the GRACE and REAP entry points over run_cdnots_gated."""

import json

import numpy as np
import pytest


def _cli(tmp_path, small_df, *args):
    from click.testing import CliRunner

    from causalts.cli import main

    csv = tmp_path / "data.csv"
    small_df.to_csv(csv, index=False)
    out = tmp_path / f"out{len(list(tmp_path.iterdir()))}"
    result = CliRunner().invoke(
        main, ["-o", str(out), "-q", "discover", str(csv), "--max-lag", "1", *args]
    )
    return result, out


def test_run_grace_matches_gates(small_df):
    from causalts.grace import run_cdnots_gated, run_grace

    kw = dict(max_lag=1, verbose=False, device="cpu", max_epochs=5, model_seed=0)
    a = run_grace(small_df, **kw)
    b = run_cdnots_gated(small_df, refine="gates", **kw)
    assert a.refine == "gates"
    np.testing.assert_array_equal(a.cg_tig, b.cg_tig)
    np.testing.assert_array_equal(a.gate_values, b.gate_values)
    np.testing.assert_array_equal(a.candidate_graph, b.skeleton)


def test_run_reap_matches_shrink_with_renamed_options(small_df):
    from causalts.grace import run_cdnots_gated, run_reap

    a = run_reap(
        small_df,
        max_lag=1,
        prune_alpha=0.01,
        basis="rff",
        robust=False,
        effect_floor=0.0,
        grow_rounds=1,
        verbose=False,
    )
    b = run_cdnots_gated(
        small_df,
        max_lag=1,
        refine="shrink",
        shrink_alpha=0.01,
        shrink_basis="rff",
        shrink_robust=False,
        shrink_effect_floor=0.0,
        grow_rounds=1,
        verbose=False,
    )
    assert a.refine == "shrink" and a.gate_values is None
    np.testing.assert_array_equal(a.cg_tig, b.cg_tig)
    np.testing.assert_array_equal(a.edge_pvalue, b.edge_pvalue)
    assert a.effect_floor == 0.0


def test_candidate_graph_is_used_and_exposed(small_df):
    from causalts.grace import run_reap

    cand = np.zeros((3, 3, 2), dtype=np.int8)
    cand[0, 1, 1] = 1  # 1 -> 2 is left out
    res = run_reap(
        small_df, max_lag=1, candidate_graph=cand, grow_rounds=0, verbose=False
    )
    np.testing.assert_array_equal(res.candidate_graph, cand)
    assert res.candidate_graph is res.skeleton
    assert not ((np.asarray(res.cg_tig) != 0) & (cand == 0)).any()
    # the default grow rounds add the missing 1 -> 2 to the candidates
    grown = run_reap(small_df, max_lag=1, candidate_graph=cand, verbose=False)
    assert grown.candidate_graph[1, 2, 1] and grown.cg_tig[1, 2, 1]


def test_run_grace_rejects_skeleton_and_reap_options(small_df):
    from causalts.grace import run_grace

    sk = np.zeros((3, 3, 2), dtype=np.int8)
    with pytest.raises(TypeError, match="candidate_graph"):
        run_grace(small_df, max_lag=1, skeleton=sk)
    with pytest.raises(TypeError, match="run_reap"):
        run_grace(small_df, max_lag=1, refine="shrink")
    with pytest.raises(TypeError, match="run_reap"):
        run_grace(small_df, max_lag=1, shrink_alpha=0.01)
    with pytest.raises(TypeError, match="use run_reap"):
        run_grace(small_df, max_lag=1, prune_alpha=0.01)
    with pytest.raises(TypeError, match="drop the argument"):
        run_grace(small_df, max_lag=1, refine="gates")


def test_run_reap_rejects_gate_options(small_df):
    from causalts.grace import run_reap

    with pytest.raises(TypeError, match="use run_grace"):
        run_reap(small_df, max_lag=1, gate_threshold=0.5)
    with pytest.raises(TypeError, match="candidate_graph"):
        run_reap(small_df, max_lag=1, skeleton=np.zeros((3, 3, 2)))
    with pytest.raises(TypeError, match="shrink_alpha -> prune_alpha"):
        run_reap(small_df, max_lag=1, shrink_alpha=0.01)
    with pytest.raises(TypeError, match="drop the argument"):
        run_reap(small_df, max_lag=1, refine="shrink")
    with pytest.raises(TypeError, match="use run_grace"):
        run_reap(small_df, max_lag=1, refine="gates")
    with pytest.raises(TypeError, match="use run_grace"):
        run_reap(small_df, max_lag=1, hidden_size=64)
    with pytest.raises(TypeError, match="not_an_option"):
        run_reap(small_df, max_lag=1, not_an_option=1)


def test_registry_and_top_level_exports():
    import causalts
    from causalts.algorithms import _ALGO_REGISTRY, _resolve_lazy, list_algorithms
    from causalts.grace import run_reap

    assert "reap" in list_algorithms()
    assert (_ALGO_REGISTRY.get("reap") or _resolve_lazy("reap")) is run_reap
    assert causalts.run_reap is run_reap
    assert callable(causalts.run_grace)


def test_cli_reap_matches_grace_refine_shrink(small_df, tmp_path):
    r1, out1 = _cli(tmp_path, small_df, "--algorithm", "reap")
    r2, out2 = _cli(tmp_path, small_df, "--algorithm", "grace", "--refine", "shrink")
    assert r1.exit_code == 0, r1.output
    assert r2.exit_code == 0, r2.output
    d1, d2 = next(out1.glob("*/")), next(out2.glob("*/"))
    np.testing.assert_array_equal(
        np.load(d1 / "estimated_graph.npy"), np.load(d2 / "estimated_graph.npy")
    )
    s1 = json.loads((d1 / "summary.json").read_text())
    assert s1["algorithm"] == "reap" and s1["refine"] == "shrink"
    assert (d1 / "edge_pvalue.npy").exists()


def test_cli_ci_test_reaches_the_candidate_graph(small_df, tmp_path, monkeypatch):
    import causalts.grace as grace

    seen = []
    real = grace.run_cdnots_gated

    def spy(*a, **kw):
        seen.append(kw.get("ci_test_class"))
        return real(*a, **kw)

    monkeypatch.setattr(grace, "run_cdnots_gated", spy)
    r1, out1 = _cli(tmp_path, small_df, "--algorithm", "reap")
    r2, out2 = _cli(tmp_path, small_df, "--algorithm", "reap", "--ci-test", "parcorr")
    assert r1.exit_code == 0, r1.output
    assert r2.exit_code == 0, r2.output
    assert seen[0] is None  # default: GRACE picks its own test
    assert type(seen[1](small_df.values)).__name__ == "PartialCorr"
    s1 = json.loads((next(out1.glob("*/")) / "summary.json").read_text())
    s2 = json.loads((next(out2.glob("*/")) / "summary.json").read_text())
    assert s1["ci_test"].startswith("auto") and s2["ci_test"] == "parcorr"


def test_cli_ci_test_reaches_grace_ss(small_df, tmp_path, monkeypatch):
    import causalts.grace as grace

    seen = []

    def spy(*a, **kw):
        seen.append(kw.get("ci_test_class"))
        raise RuntimeError("stop")

    monkeypatch.setattr(grace, "run_stability_selection", spy)
    _cli(tmp_path, small_df, "--algorithm", "grace-ss", "--ci-test", "parcorr")
    assert type(seen[0](small_df.values)).__name__ == "PartialCorr"


def test_cli_reap_rejects_refine_gates(small_df, tmp_path):
    result, _ = _cli(tmp_path, small_df, "--algorithm", "reap", "--refine", "gates")
    assert result.exit_code != 0 and "contradicts" in result.output


def test_cli_reap_only_flags_accept_reap(small_df, tmp_path):
    result, out = _cli(
        tmp_path, small_df, "--algorithm", "reap", "--shrink-effect-floor", "adaptive"
    )
    assert result.exit_code == 0, result.output
    summary = json.loads((next(out.glob("*/")) / "summary.json").read_text())
    assert summary["shrink_effect_floor"] == "adaptive"


def test_cli_reap_warns_on_gate_flags(small_df, tmp_path):
    from click.testing import CliRunner

    from causalts.cli import main

    csv = tmp_path / "data.csv"
    small_df.to_csv(csv, index=False)
    result = CliRunner().invoke(
        main,
        ["-o", str(tmp_path / "o"), "discover", str(csv), "--algorithm", "reap"]
        + ["--max-lag", "1", "--ar-path", "--max-epochs", "5", "--no-plot"],
    )
    assert result.exit_code == 0, result.output
    assert "--max-epochs, --ar-path have no effect with REAP" in result.output


def test_cli_reap_effect_floor_points_to_shrink_effect_floor(small_df, tmp_path):
    result, _ = _cli(
        tmp_path, small_df, "--algorithm", "reap", "--effect-floor", "0.05"
    )
    assert result.exit_code != 0 and "--shrink-effect-floor" in result.output


def test_run_reap_defaults_are_the_experiment_configuration(small_df):
    # REAP's experiments: three grow rounds on rank-transformed data, the rest at
    # the defaults (prune_alpha 0.001, spline basis, robust, no floor, lagged only)
    from causalts.grace import run_cdnots_gated, run_reap

    a = run_reap(small_df, max_lag=1, verbose=False)
    b = run_cdnots_gated(
        small_df,
        max_lag=1,
        refine="shrink",
        grow_rounds=3,
        grow_transform="rank",
        shrink_alpha=0.001,
        verbose=False,
    )
    np.testing.assert_array_equal(a.cg_tig, b.cg_tig)
    np.testing.assert_array_equal(a.edge_pvalue, b.edge_pvalue)
    assert a.grow_history == b.grow_history


@pytest.mark.parametrize(
    "kw,msg",
    [
        ({"prune_alpha": 0}, "prune_alpha must be in"),
        ({"basis": "poly"}, "basis must be 'spline' or 'rff'"),
        ({"effect_floor": "auto"}, "effect_floor must be None, 'adaptive'"),
        ({"interactions": True, "effect_floor": "adaptive"}, "interactions=True"),
    ],
)
def test_run_reap_errors_use_its_names(small_df, kw, msg):
    from causalts.grace import run_reap

    with pytest.raises(ValueError, match=msg) as exc:
        run_reap(small_df, max_lag=1, verbose=False, **kw)
    assert "shrink_" not in str(exc.value)


def test_candidate_graph_is_keyword_only_and_checked(small_df):
    from causalts.grace import run_grace, run_reap

    for fn in (run_grace, run_reap):
        with pytest.raises(TypeError):
            fn(small_df, 1, np.zeros((3, 3, 2)))  # a ported (df, max_lag, alpha) call
        with pytest.raises(ValueError, match="candidate_graph must have shape"):
            fn(small_df, max_lag=1, candidate_graph=np.zeros((3, 3, 3)))
        with pytest.raises(ValueError, match="candidate_graph must have shape"):
            fn(small_df, max_lag=1, candidate_graph=0.01)


def test_discover_df_reap(small_df):
    import causalts
    from causalts.grace import run_reap

    a = causalts.discover_df(small_df, algorithm="reap", max_lag=1, verbose=False)
    b = run_reap(small_df, max_lag=1, include_C=False, verbose=False)
    assert a.refine == "shrink"
    np.testing.assert_array_equal(a.cg_tig, b.cg_tig)


def test_candidate_graph_with_c_columns(small_df):
    from causalts.grace import run_grace

    kw = dict(max_lag=1, verbose=False, device="cpu", max_epochs=2, model_seed=0)
    G = np.ones((4, 4, 2), dtype=np.int8)  # 3 variables + the linear C column
    res = run_grace(small_df, candidate_graph=G, include_C_in_model=True, **kw)
    assert res.cg_tig.shape == (3, 3, 2)
    with pytest.raises(ValueError, match=r"\(4, 4, 2\)"):
        run_grace(small_df, candidate_graph=G[:3, :3], include_C_in_model=True, **kw)
    # without include_C the engine's configuration error comes first
    with pytest.raises(ValueError, match="requires include_C=True"):
        run_grace(
            small_df,
            candidate_graph=G[:3, :3],
            include_C=False,
            include_C_in_model=True,
            **kw,
        )


def test_weighted_candidate_graph_is_binarised(small_df):
    from causalts.grace import run_grace, run_reap

    w = np.zeros((3, 3, 2))
    w[0, 1, 1], w[1, 2, 1] = -0.3, 0.2  # signed / small weights are edges
    b = (w != 0).astype(np.int8)
    for fn, kw in ((run_reap, {}), (run_grace, dict(device="cpu", max_epochs=2))):
        kw = dict(max_lag=1, grow_rounds=0, verbose=False, model_seed=0, **kw)
        if fn is run_reap:
            kw.pop("model_seed")
        else:
            kw["ar_path"] = False
        a = fn(small_df, candidate_graph=w, **kw)
        c = fn(small_df, candidate_graph=b, **kw)
        np.testing.assert_array_equal(a.cg_tig, c.cg_tig)
        np.testing.assert_array_equal(a.candidate_graph, b)
    w[0, 0, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        run_reap(small_df, max_lag=1, candidate_graph=w, verbose=False)
