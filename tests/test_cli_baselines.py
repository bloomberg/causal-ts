# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Score-based methods through ``causal-ts discover`` and discover_df.

Also covers passing a result with C-node rows/columns as a GRACE/REAP candidate graph.
"""

import json

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from causalts.baselines import (
    BASELINE_ALGORITHMS,
    BaselineResult,
    granger_discovery,
    lasso_var_discovery,
    run_baseline,
    varlingam_discovery,
)
from causalts.cli import main
from causalts.inspection import discover_df, edges_from_graph
from causalts.lges import lges_discovery, tges_discovery
from causalts.synthetic_data.synthetic_datasets import load_dataset

ALGOS = ["lges", "tges", "ges", "lasso-var", "granger", "varlingam"]
LAG0_ALGOS = {"lges", "tges", "ges", "varlingam"}


@pytest.fixture(scope="module")
def ex2():
    out = load_dataset("ex2", seed=0, T=300)
    return out["df"], out["ground_truth"]


@pytest.fixture(scope="module")
def ex2_csv(tmp_path_factory, ex2):
    path = tmp_path_factory.mktemp("data") / "ex2.csv"
    ex2[0].to_csv(path, index=False)
    np.save(path.parent / "gt.npy", ex2[1])
    return path


@pytest.fixture(scope="module")
def lag0_df():
    """A(t) -> B(t) at lag 0, B(t-1) -> C(t), uniform noise (non-Gaussian)."""
    rng = np.random.default_rng(0)
    T = 400
    e = rng.uniform(-1, 1, (T, 3))
    X = np.zeros((T, 3))
    for t in range(1, T):
        X[t, 0] = 0.5 * X[t - 1, 0] + e[t, 0]
        X[t, 1] = 0.9 * X[t, 0] + e[t, 1]
        X[t, 2] = 0.6 * X[t - 1, 1] + e[t, 2]
    return pd.DataFrame(X, columns=["A", "B", "C"])


def _discover(tmp_path, csv, *args):
    return CliRunner().invoke(
        main,
        [
            "-q",
            "-s",
            "0",
            "-o",
            str(tmp_path),
            "discover",
            str(csv),
            "--no-plot",
            *args,
        ],
    )


def _summary(res):
    assert res.exit_code == 0, res.output
    return json.loads(res.output)


def test_registered_as_builtins():
    from causalts.algorithms import list_algorithms
    from causalts.cli import _BUILTIN_ALGORITHMS

    assert set(ALGOS) == set(BASELINE_ALGORITHMS)
    assert set(ALGOS) <= set(list_algorithms())
    assert set(ALGOS) <= set(_BUILTIN_ALGORITHMS)


@pytest.mark.parametrize("algo", ALGOS)
def test_cli_json_edges(tmp_path, ex2_csv, algo):
    s = _summary(_discover(tmp_path, ex2_csv, "-a", algo, "--max-lag", "2", "--json"))
    assert s["algorithm"] == algo
    assert s["ci_test"] is None and s["alpha"] is None
    assert s["include_C"] is False
    assert s["contemporaneous"] is (algo in LAG0_ALGOS)
    (outdir,) = tmp_path.glob(f"discover_{algo}_*")
    g = np.load(outdir / "estimated_graph.npy")
    assert g.shape == (6, 6, 3)
    assert s["edges"] and s["n_edges"] == len(s["edges"]) == int(g.sum())
    names = [f"X{i}" for i in range(6)]
    assert s["edges"] == edges_from_graph(g, names)
    # The CLI graph is the library function's graph (same seed for VARLiNGAM's ICA).
    np.random.seed(0)
    assert np.array_equal(g, run_baseline(algo, pd.read_csv(ex2_csv), 2).cg_tig)
    if algo in ("lasso-var", "granger", "varlingam"):
        assert (outdir / "coefficients.npy").exists()
    if algo in LAG0_ALGOS:
        assert (outdir / "edge_marks.npy").exists()


@pytest.mark.parametrize("algo", ALGOS)
def test_cli_lag0_edges(tmp_path, lag0_df, algo):
    csv = tmp_path / "lag0.csv"
    lag0_df.to_csv(csv, index=False)
    s = _summary(_discover(tmp_path, csv, "-a", algo, "--max-lag", "1", "--json"))
    lag0 = {(e["source"], e["target"]) for e in s["edges"] if e["lag"] == 0}
    if algo in LAG0_ALGOS:
        assert ("A", "B") in lag0
        assert s["diagnostics"]["contemporaneous"] == len(lag0)
    else:
        assert not lag0 and s["diagnostics"]["contemporaneous"] == 0
        assert ("B", "C", 1) in {
            (e["source"], e["target"], e["lag"]) for e in s["edges"]
        }


@pytest.mark.parametrize(
    "algo,args,hint",
    [
        ("lges", ["--ci-test", "parcorr"], "--ci-test"),
        ("granger", ["--include-c"], "--include-c"),
        ("ges", ["--c-preset", "step"], "--c-preset"),
        ("varlingam", ["--alpha", "0.1"], "--alpha"),
        ("tges", ["--lag-list", "1,2"], "--lag-list"),
        ("lasso-var", ["--pvalues"], "--pvalues"),
    ],
)
def test_cli_rejects_inapplicable_flags(tmp_path, ex2_csv, algo, args, hint):
    res = _discover(tmp_path, ex2_csv, "-a", algo, "--max-lag", "1", *args)
    assert res.exit_code == 2
    assert f"{hint} does not apply to --algorithm {algo}" in res.output


def test_cli_accepts_no_c(tmp_path, ex2_csv):
    s = _summary(
        _discover(tmp_path, ex2_csv, "-a", "lges", "--max-lag", "1", "--no-c", "--json")
    )
    assert s["include_C"] is False


@pytest.mark.parametrize(
    "algo,args",
    [
        ("cdnots", ["--lambda-value", "2"]),
        ("ges", ["--lges-mode", "lges-safe"]),
        ("granger", ["--cv-folds", "3"]),
        ("lasso-var", ["--coef-threshold", "0.1"]),
    ],
)
def test_cli_rejects_method_flags_elsewhere(tmp_path, ex2_csv, algo, args):
    res = _discover(tmp_path, ex2_csv, "-a", algo, "--max-lag", "1", *args)
    assert res.exit_code == 2
    assert f"{args[0]} applies only to --algorithm" in res.output


def test_cli_lagged_only_needs_max_lag(tmp_path, ex2_csv):
    res = _discover(tmp_path, ex2_csv, "-a", "granger", "--max-lag", "0")
    assert res.exit_code == 2 and "lagged edges only" in res.output


def test_varlingam_needs_a_lag(tmp_path, ex2_csv, ex2):
    res = _discover(tmp_path, ex2_csv, "-a", "varlingam", "--max-lag", "0")
    assert res.exit_code == 2 and "fits a VAR" in res.output
    with pytest.raises(ValueError, match="varlingam fits a VAR"):
        discover_df(ex2[0], algorithm="varlingam", max_lag=0)


@pytest.mark.parametrize("algo", ["lges", "tges", "ges"])
def test_score_methods_run_at_max_lag_0(tmp_path, ex2_csv, algo):
    s = _summary(_discover(tmp_path, ex2_csv, "-a", algo, "--max-lag", "0", "--json"))
    assert s["edges"] and all(e["lag"] == 0 for e in s["edges"])
    (outdir,) = tmp_path.glob(f"discover_{algo}_*")
    assert np.load(outdir / "estimated_graph.npy").shape == (6, 6, 1)


@pytest.mark.parametrize(
    "algo,args,kw",
    [
        ("lges", ["--lambda-value", "20", "--lges-mode", "lges-safe"],
         {"lambda_value": 20.0, "mode": "lges-safe"}),
        ("ges", ["--lambda-value", "20"], {"lambda_value": 20.0}),
        ("lasso-var", ["--cv-folds", "3"], {"cv_folds": 3}),
        ("varlingam", ["--coef-threshold", "0.3"], {"threshold": 0.3}),
        ("granger", ["--coef-threshold", "0.3"], {"threshold": 0.3}),
    ],
)  # fmt: skip
def test_cli_method_options_pass_through(tmp_path, ex2_csv, ex2, algo, args, kw):
    s = _summary(
        _discover(tmp_path, ex2_csv, "-a", algo, "--max-lag", "2", "--json", *args)
    )
    for k, v in kw.items():
        assert s["method_params"][k] == v
    assert "mode" not in s and "threshold" not in s
    np.random.seed(0)
    want = run_baseline(algo, ex2[0], 2, **kw).cg_tig
    assert s["n_edges"] == int(want.sum())
    assert s["n_edges"] < int(run_baseline(algo, ex2[0], 2).cg_tig.sum()) or (
        algo == "lasso-var"
    )


def test_cli_validate_ground_truth_and_evaluate(tmp_path, ex2_csv):
    gt = ex2_csv.parent / "gt.npy"
    res = CliRunner().invoke(
        main,
        [
            "-q",
            "-o",
            str(tmp_path),
            "discover",
            str(ex2_csv),
            "-a",
            "lasso-var",
            "--max-lag",
            "2",
            "--validate",
            "--n-bootstrap",
            "2",
            "--ground-truth",
            str(gt),
        ],
    )
    assert res.exit_code == 0, res.output
    (outdir,) = tmp_path.glob("discover_lasso-var_*")
    s = json.loads((outdir / "summary.json").read_text())
    assert all(0 <= e["persistence"] <= 1 for e in s["edges"])
    ev = CliRunner().invoke(
        main,
        ["-q", "-o", str(tmp_path), "evaluate", str(gt), str(outdir / "estimated_graph.npy"), "--no-plot"],
    )  # fmt: skip
    assert ev.exit_code == 0, ev.output


def test_cli_saves_plot(tmp_path, ex2_csv):
    res = CliRunner().invoke(
        main,
        ["-q", "-o", str(tmp_path), "discover", str(ex2_csv), "-a", "varlingam", "--max-lag", "1"],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    (outdir,) = tmp_path.glob("discover_varlingam_*")
    assert (outdir / "network.png").exists()


@pytest.mark.parametrize(
    "algo,fn,n_out",
    [
        ("lges", lges_discovery, 2),
        ("tges", tges_discovery, 2),
        ("lasso-var", lasso_var_discovery, 3),
        ("granger", granger_discovery, 3),
        ("varlingam", varlingam_discovery, 3),
    ],
)
def test_discover_df_matches_library(ex2, algo, fn, n_out):
    df = ex2[0]
    np.random.seed(0)  # VARLiNGAM's ICA draws from the global state
    res = discover_df(df, algorithm=algo, max_lag=2)
    assert isinstance(res, BaselineResult) and res.method == algo
    np.random.seed(0)
    out = fn(df, max_lag=2)
    assert np.array_equal(res.cg_tig, out[0])
    if n_out == 3:
        assert res.coefficients.shape == res.cg_tig.shape
    else:
        want = out[1].get("tiered_cpdag", out[1]["cpdag"])  # TGES: tier-oriented
        assert np.array_equal(res.cpdag, want)
    assert res.var_names == list(df.columns)
    assert res.info["method"]


def test_discover_df_ges_and_kwargs(ex2):
    df = ex2[0]
    res = discover_df(df, algorithm="ges", max_lag=1, lambda_value=50.0)
    assert res.cpdag is not None and res.coefficients is None
    assert res.cg_tig.sum() <= discover_df(df, algorithm="ges", max_lag=1).cg_tig.sum()


@pytest.mark.parametrize(
    "kw",
    [
        {"ci_test": "parcorr"},
        {"include_C": True},
        {"c_preset": "step"},
        {"alpha": 0.1},
    ],
)
def test_discover_df_rejects_inapplicable(ex2, kw):
    with pytest.raises(ValueError, match="does not apply to algorithm='lges'"):
        discover_df(ex2[0], algorithm="lges", max_lag=1, **kw)


def test_run_algorithm_rejects_ci_test(ex2):
    from causalts.algorithms import run_algorithm
    from causalts.ci_tests import create_ci_test

    ci = create_ci_test("parcorr", ex2[0].values)
    with pytest.raises(ValueError, match="takes no CI test"):
        run_algorithm("granger", ex2[0], ci, 1)
    assert isinstance(run_algorithm("granger", ex2[0], None, 1), BaselineResult)


def test_result_surface(lag0_df):
    res = run_baseline("lges", lag0_df, 1)
    marks = res.to_marks()
    assert marks[0, 1, 0] == marks[1, 0, 0] == "o-o"
    fig = res.plot()
    fig = fig[0] if isinstance(fig, tuple) else fig
    assert fig is not None
    wrapped = res._bridge()
    assert np.array_equal(wrapped.cg_tig, res.cg_tig)
    vl = run_baseline("varlingam", lag0_df, 1)
    assert vl.to_marks()[0, 1, 0] == "-->" and vl.info["causal_order"]


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ex2_nan(ex2):
    df = ex2[0].copy()
    df.iloc[::17, 1] = np.nan
    return df


@pytest.mark.parametrize("algo", ALGOS)
def test_nan_input_raises(ex2_nan, algo):
    with pytest.raises(ValueError, match=f"{algo} needs complete.*var_em"):
        run_baseline(algo, ex2_nan, 2)
    with pytest.raises(ValueError, match="needs complete"):
        discover_df(ex2_nan, algorithm=algo, max_lag=2)
    inf = ex2_nan.fillna(0.0)
    inf.iloc[3, 0] = np.inf
    with pytest.raises(ValueError, match="NaN/inf"):
        run_baseline(algo, inf, 2)


@pytest.mark.parametrize("algo", ["lges", "lasso-var"])
def test_cli_impute_var_em(tmp_path, ex2_nan, algo):
    from causalts.imputation import var_em_impute

    csv = tmp_path / "nan.csv"
    ex2_nan.to_csv(csv, index=False)
    s = _summary(
        _discover(
            tmp_path, csv, "-a", algo, "--max-lag", "2", "--impute", "var_em",
            "--json", "--validate", "--n-bootstrap", "2",
        )
    )  # fmt: skip
    assert s["impute"] == "var_em" and s["n_edges"] > 0
    assert all(e["persistence"] is not None for e in s["edges"])
    (outdir,) = tmp_path.glob(f"discover_{algo}_*")
    g = np.load(outdir / "estimated_graph.npy")
    filled = var_em_impute(pd.read_csv(csv))
    assert np.array_equal(g, run_baseline(algo, filled, 2).cg_tig)


def test_cli_nan_without_impute_errors(tmp_path, ex2_nan):
    csv = tmp_path / "nan.csv"
    ex2_nan.to_csv(csv, index=False)
    res = _discover(tmp_path, csv, "-a", "lges", "--max-lag", "2")
    assert res.exit_code == 1
    assert "needs complete" in res.output
    assert isinstance(res.exception, SystemExit)  # clean CLI error, no traceback


@pytest.mark.parametrize("strategy", ["pairwise_complete", "causal_iterative"])
def test_cli_rejects_other_impute(tmp_path, ex2_csv, strategy):
    res = _discover(tmp_path, ex2_csv, "-a", "lges", "--impute", strategy)
    assert res.exit_code == 2
    assert f"--impute {strategy} does not apply to --algorithm lges" in res.output


# ---------------------------------------------------------------------------
# Candidate graph with C-node rows/columns (run_reap / run_grace)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cdnots_with_c(ex2):
    return discover_df(
        ex2[0], algorithm="cdnots", ci_test="parcorr", max_lag=2, include_C=True
    )


def test_reap_accepts_result_with_c(ex2, cdnots_with_c):
    from causalts.grace.runners import run_reap

    df = ex2[0]
    assert cdnots_with_c.cg_tig.shape[0] == df.shape[1] + 1
    kw = dict(max_lag=2, grow_rounds=0, verbose=False)
    a = run_reap(df, candidate_graph=cdnots_with_c, **kw)
    b = run_reap(df, candidate_graph=cdnots_with_c.cg_tig[:6, :6], **kw)
    assert np.array_equal(a.cg_tig, b.cg_tig)
    assert np.array_equal(a.candidate_graph, b.candidate_graph)


def test_candidate_bare_array_with_c_names_the_slice(ex2, cdnots_with_c):
    from causalts.grace.runners import run_grace, run_reap

    for fn in (run_reap, run_grace):
        with pytest.raises(ValueError, match=r"res\.cg_tig\[:6, :6\]"):
            fn(ex2[0], max_lag=2, candidate_graph=cdnots_with_c.cg_tig)


def test_candidate_result_checks_variables(ex2, cdnots_with_c):
    from causalts.grace.runners import _check_candidate_graph

    df = ex2[0]
    G = _check_candidate_graph(cdnots_with_c, df, 2)
    assert G.shape == (6, 6, 3)
    assert np.array_equal(G, (cdnots_with_c.cg_tig[:6, :6] != 0).astype(np.int8))
    renamed = df.rename(columns={"X0": "Z"})
    with pytest.raises(ValueError, match="result over variables"):
        _check_candidate_graph(cdnots_with_c, renamed, 2)
    # Custom c_array: no C names recorded, but var_names lists exactly the
    # observed columns, so the trailing rows are still dropped as C nodes.
    import copy

    unrecorded = copy.copy(cdnots_with_c)
    unrecorded.c_node_names = None
    G2 = _check_candidate_graph(unrecorded, df, 2)
    assert np.array_equal(G2, G)


class _FakeResult:
    def __init__(self, cg_tig, var_names, c_node_names):
        self.cg_tig = cg_tig
        self.var_names = var_names
        self.c_node_names = c_node_names


def test_candidate_result_observed_variable_not_taken_for_c():
    from causalts.grace.runners import _check_candidate_graph

    g = np.ones((3, 3, 2))
    df = pd.DataFrame(np.zeros((10, 2)), columns=["A", "B"])
    res = _FakeResult(g, ["A", "B", "C"], [])
    for n_extra in (0, 1):
        with pytest.raises(ValueError, match="result over variables"):
            _check_candidate_graph(res, df, 1, n_extra)
    # No recorded C nodes and no variable names: extra rows still raise.
    with pytest.raises(ValueError, match="cannot tell which"):
        _check_candidate_graph(_FakeResult(g, [], None), df, 1, 1)
    # Recorded C node: dropped, or kept when the model expects one C column.
    rec = _FakeResult(g, ["A", "B"], ["C_lin"])
    assert _check_candidate_graph(rec, df, 1).shape == (2, 2, 2)
    assert _check_candidate_graph(rec, df, 1, 1).shape == (3, 3, 2)


def test_candidate_cedar_result_with_c(ex2):
    from causalts.grace.runners import _check_candidate_graph

    df = ex2[0]
    res = discover_df(
        df, algorithm="cedar", ci_test="parcorr", max_lag=2, include_C=True,
        c_preset="linear+sin",
    )  # fmt: skip
    assert res.cg_tig.shape[0] == 8 and res.var_names[-2:] == res.c_node_names
    dropped = _check_candidate_graph(res, df, 2)
    assert np.array_equal(dropped, (res.cg_tig[:6, :6] != 0).astype(np.int8))
    kept = _check_candidate_graph(res, df, 2, 2)
    assert np.array_equal(kept, (res.cg_tig != 0).astype(np.int8))
    with pytest.raises(ValueError, match="C basis has 1 columns"):
        _check_candidate_graph(res, df, 2, 1)
    # Keeping C rows checks their names, not only their count.
    kept2 = _check_candidate_graph(res, df, 2, 2, ["C_lin", "C_sin"])
    assert np.array_equal(kept2, kept)
    with pytest.raises(ValueError, match="do not match the model's C basis"):
        _check_candidate_graph(res, df, 2, 2, ["C_lin", "C_exp"])


def test_candidate_cdnots_result_keep_path(ex2, cdnots_with_c):
    from causalts.grace.runners import _check_candidate_graph

    kept = _check_candidate_graph(cdnots_with_c, ex2[0], 2, 1)
    assert kept.shape == (7, 7, 3)


def test_candidate_cdnots_result_with_observed_column_named_like_c():
    """CDNOTS lists observed columns only; one may be named like a C node."""
    from causalts.grace.runners import _check_candidate_graph, run_grace, run_reap

    rng = np.random.default_rng(0)
    T = 300
    a = rng.normal(size=T)
    b = np.zeros(T)
    for t in range(1, T):
        b[t] = 0.8 * a[t - 1] + 0.3 * rng.normal()
    df = pd.DataFrame({"A": a, "C_lin": b})
    res = discover_df(
        df, algorithm="cdnots", ci_test="parcorr", max_lag=1, include_C=True
    )
    assert res.var_names == ["A", "C_lin"] and res.c_node_names == ["C_lin"]
    G = _check_candidate_graph(res, df, 1)
    assert np.array_equal(G, (res.cg_tig[:2, :2] != 0).astype(np.int8))
    kw = dict(max_lag=1, candidate_graph=res, grow_rounds=0, verbose=False)
    assert run_reap(df, **kw).cg_tig.shape == (2, 2, 2)
    assert run_grace(df, max_epochs=5, **kw).cg_tig.shape == (2, 2, 2)


def test_discover_df_baseline_impute_var_em(ex2_nan):
    from causalts.baselines import BaselineResult
    from causalts.inspection import discover_df

    res = discover_df(ex2_nan, algorithm="lges", max_lag=2, impute="var_em")
    assert isinstance(res, BaselineResult)
    # no NaN: impute="var_em" is accepted and does nothing
    clean = ex2_nan.interpolate().bfill()
    discover_df(clean, algorithm="lasso-var", max_lag=2, impute="var_em")
    with pytest.raises(ValueError, match="does not apply"):
        discover_df(ex2_nan, algorithm="lges", max_lag=2, impute="pairwise_complete")


def test_tges_result_cpdag_is_tiered(ex2):
    from causalts.baselines import run_baseline

    res = run_baseline("tges", ex2[0], 2)
    assert res.cpdag is res.info["tiered_cpdag"]


def test_candidate_result_with_custom_c_array(ex2):
    """A CDNOTS result built from a custom c_array records no C names."""
    import pandas as pd

    from causalts.grace.runners import _check_candidate_graph

    df = ex2[0]
    d = df.shape[1]

    class _Res:
        pass

    res = _Res()
    res.cg_tig = np.zeros((d + 2, d + 2, 3))
    res.cg_tig[0, 1, 1] = 1
    res.var_names = list(df.columns)
    res.c_node_names = None
    G = _check_candidate_graph(res, df, 2)
    assert G.shape == (d, d, 3) and G[0, 1, 1] == 1
    # names over more variables than df, no C names recorded: still ambiguous
    res2 = _Res()
    res2.cg_tig = np.zeros((3, 3, 2))
    res2.var_names = ["A", "B", "C"]
    res2.c_node_names = []
    with pytest.raises(ValueError):
        _check_candidate_graph(
            res2, pd.DataFrame(np.zeros((5, 2)), columns=["A", "B"]), 1
        )


def test_run_grace_rejects_result_with_other_c_preset(ex2):
    """A step-basis result must not be reused as the model's linear C column."""
    from causalts.grace.runners import run_grace

    df = ex2[0]
    res = discover_df(
        df, algorithm="cdnots", ci_test="parcorr", max_lag=2, include_C=True,
        c_preset="step",
    )  # fmt: skip
    with pytest.raises(ValueError, match="do not match the model's C basis"):
        run_grace(
            df, 2, candidate_graph=res, include_C_in_model=True, c_preset="linear"
        )


def test_baseline_names_in_sync():
    """One list of baseline names; every registry agrees with it."""
    from causalts import cli
    from causalts.algorithms import _LAZY_ALGO_REGISTRY
    from causalts.baselines import BASELINE_ALGORITHMS

    in_registry = {
        k for k, (mod, _) in _LAZY_ALGO_REGISTRY.items() if mod == "causalts.baselines"
    }
    assert in_registry == set(BASELINE_ALGORITHMS)
    assert set(cli._BASELINE_ALGORITHMS) == set(BASELINE_ALGORITHMS)
    assert set(BASELINE_ALGORITHMS) <= set(cli._BUILTIN_ALGORITHMS)
