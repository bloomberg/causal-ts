# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later
"""With ``--json``, stdout carries only the JSON document (progress goes to stderr)."""

import json
import sys

import click
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from causalts.cli import _json_only_stdout, main
from causalts.synthetic_data.synthetic_datasets import load_dataset


def _runner():
    # click < 8.2 mixes stderr into stdout unless asked not to; 8.2 removed the argument.
    try:
        return CliRunner(mix_stderr=False)
    except TypeError:
        return CliRunner()


@pytest.fixture(scope="module")
def data_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("json") / "data.csv"
    load_dataset("ex2", seed=0, T=200)["df"].to_csv(path, index=False)
    return path


def _discover(tmp_path, csv, *args):
    runner = _runner()
    return runner.invoke(
        main, ["-s", "0", "-o", str(tmp_path), "discover", str(csv), *args]
    )


@pytest.mark.parametrize("algorithm", ["cdnots", "reap", "lges"])
def test_discover_json_stdout_is_pure_json(tmp_path, data_csv, algorithm):
    # reap prints its own progress from library code; lges takes the baseline path.
    res = _discover(tmp_path, data_csv, "-a", algorithm, "--max-lag", "2", "--json")
    assert res.exit_code == 0, res.output
    summary = json.loads(res.stdout)  # the whole of stdout, not a slice
    assert summary["algorithm"] == algorithm
    assert "Reading data from" in res.stderr


def test_discover_without_json_keeps_progress_on_stdout(tmp_path, data_csv):
    res = _discover(tmp_path, data_csv, "-a", "cdnots", "--max-lag", "2")
    assert res.exit_code == 0, res.output
    assert "Reading data from" in res.stdout


def test_ground_truth_metrics_go_to_stderr_with_json(tmp_path, data_csv):
    gt = tmp_path / "gt.npy"
    np.save(gt, load_dataset("ex2", seed=0, T=200)["ground_truth"])
    res = _discover(
        tmp_path, data_csv, "-a", "cdnots", "--max-lag", "2", "--json",
        "--ground-truth", str(gt),
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    assert "metrics" in json.loads(res.stdout)
    assert "Precision=" in res.stderr


def test_deconfound_json_stdout_is_pure_json(tmp_path, data_csv):
    res = _discover(tmp_path, data_csv, "-a", "cdnots", "--max-lag", "2", "--json")
    assert res.exit_code == 0, res.output
    (graph,) = tmp_path.glob("discover_cdnots_*/estimated_graph.npy")
    runner = _runner()
    out = runner.invoke(
        main,
        [
            "-o",
            str(tmp_path),
            "deconfound",
            str(graph),
            "--data",
            str(data_csv),
            "--json",
        ],
    )
    assert out.exit_code == 0, out.output
    json.loads(out.stdout)


def test_json_only_stdout_restores_stdout_on_close():
    before = sys.stdout
    ctx = click.Context(main, obj={})
    try:
        real = _json_only_stdout(ctx)
        assert real is before and sys.stdout is sys.stderr
    finally:
        ctx.close()
    assert sys.stdout is before


@pytest.fixture(scope="module")
def trend_csv(tmp_path_factory):
    # Two trending series; one column happens to be called "C2".
    rng = np.random.default_rng(0)
    T = 400
    a = rng.normal(size=T) + 0.01 * np.arange(T)
    b = np.zeros(T)
    for i in range(1, T):
        b[i] = 0.8 * a[i - 1] + 0.3 * rng.normal() + 0.01 * i
    path = tmp_path_factory.mktemp("trend") / "data.csv"
    pd.DataFrame({"A": a, "C2": b}).to_csv(path, index=False)
    return path


def test_discover_json_names_and_flags_the_c_node(tmp_path, trend_csv):
    res = _discover(
        tmp_path, trend_csv, "-a", "cdnots", "--include-c", "--max-lag", "1",
        "--json", "--no-plot", "--pvalues", "--validate", "--n-bootstrap", "3",
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    out = json.loads(res.stdout)
    (run,) = tmp_path.glob("discover_cdnots_*")
    g = np.load(run / "estimated_graph.npy")
    pv = np.load(run / "pvalues.npy")
    P = np.load(run / "persistence.npy")
    assert g.shape[0] == 3
    edges = out["edges"]
    assert len(edges) == int(g.astype(bool).sum()) == out["n_edges"]
    for e, (i, j, k) in zip(edges, zip(*np.where(g))):
        assert e["lag"] == k
        assert e.get("c_node", False) == (i == 2 or j == 2)
        if e.get("c_node"):
            assert "C_lin" in (e["source"], e["target"])
        else:
            assert {e["source"], e["target"]} <= {"A", "C2"}
        assert e["pvalue"] == (float(pv[i, j, k]) if np.isfinite(pv[i, j, k]) else None)
        assert e["persistence"] == round(float(P[i, j, k]), 3)
    assert {"source": "A", "target": "C2", "lag": 1} in [
        {k: e[k] for k in ("source", "target", "lag")} for e in edges
    ]
    assert any(e.get("c_node") for e in edges)
    diag = out["diagnostics"]
    assert diag["contemporaneous"] == 0 and diag["self_loops"] == 0
    assert diag["n_edges"] == int(g[:2, :2].astype(bool).sum())
    assert diag["nonstationary_vars"]


def test_discover_json_without_c_is_unchanged(tmp_path, trend_csv):
    from causalts.inspection import diagnostics_from_graph, edges_from_graph

    res = _discover(
        tmp_path, trend_csv, "-a", "cdnots", "--no-c", "--max-lag", "1",
        "--json", "--no-plot",
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    out = json.loads(res.stdout)
    (graph,) = tmp_path.glob("discover_cdnots_*/estimated_graph.npy")
    g = np.load(graph)
    assert out["edges"] == edges_from_graph(g, ["A", "C2"])
    assert out["diagnostics"] == diagnostics_from_graph(g, ["A", "C2"])
    assert not any("c_node" in e for e in out["edges"])


def test_discover_json_grace_with_c_exports_observed_edges(
    tmp_path, trend_csv, monkeypatch
):
    # GRACE uses C during discovery but its graph holds the observed variables only.
    from types import SimpleNamespace

    import causalts.grace as grace

    g = np.zeros((2, 2, 2), dtype=int)
    g[0, 1, 1] = g[1, 0, 0] = g[1, 1, 1] = 1
    fake = SimpleNamespace(
        cg_tig=g,
        gate_values=None,
        gate_threshold=0.5,
        gate_threshold_rule="fixed",
        c_node_names=["C_lin", "C_sin"],
    )
    monkeypatch.setattr(grace, "run_cdnots_gated", lambda **kw: fake)
    res = _discover(
        tmp_path, trend_csv, "-a", "grace", "--include-c", "--c-preset",
        "linear+sin", "--max-lag", "1", "--json", "--no-plot",
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    out = json.loads(res.stdout)
    assert out["edges"] == [
        {"source": "A", "target": "C2", "lag": 1, "pvalue": None},
        {"source": "C2", "target": "A", "lag": 0, "pvalue": None},
        {"source": "C2", "target": "C2", "lag": 1, "pvalue": None},
    ]
    assert out["diagnostics"] == {
        "n_edges": 3,
        "density": 0.375,
        "self_loops": 1,
        "contemporaneous": 1,
        "lagged": 2,
        "max_in_degree": 2,
        "hub": "C2",
        "empty": False,
        "saturated": False,
    }


def test_discover_json_cedar_c_columns_without_names(tmp_path, trend_csv, monkeypatch):
    # CEDAR result with two C columns but no recorded names: fallback labels,
    # disambiguated against the real "C2" column.
    from types import SimpleNamespace

    import causalts.cedar.discovery as cedar

    g = np.zeros((4, 4, 2), dtype=int)
    g[0, 1, 1] = g[2, 0, 0] = g[3, 1, 0] = 1
    fake = SimpleNamespace(
        cg_tig=g,
        val_matrix=None,
        pvalue_matrix=None,
        ar_violations=None,
        ar_order=None,
        var_names=["A", "C2"],
        n_tests=0,
        c_node_names=None,
    )
    monkeypatch.setattr(cedar, "run_cedar", lambda **kw: fake)
    res = _discover(
        tmp_path, trend_csv, "-a", "cedar", "--include-c", "--c-preset",
        "linear+sin", "--max-lag", "1", "--json", "--no-plot",
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    out = json.loads(res.stdout)
    assert out["edges"] == [
        {"source": "A", "target": "C2", "lag": 1, "pvalue": None},
        {"source": "C1", "target": "A", "lag": 0, "pvalue": None, "c_node": True},
        {"source": "C2_1", "target": "C2", "lag": 0, "pvalue": None, "c_node": True},
    ]
    assert out["diagnostics"]["n_edges"] == 1
    assert out["diagnostics"]["contemporaneous"] == 0
    assert out["diagnostics"]["nonstationary_vars"] == ["A", "C2"]
