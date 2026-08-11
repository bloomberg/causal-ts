# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for Phase-2 additions: diagnostics block and discover --validate."""

import json
import os

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from causalts import discover_df, inspect_df
from causalts.cli import main
from causalts.inspection import diagnostics_from_graph


def test_diagnostics_from_graph_fields():
    g = np.zeros((3, 3, 3), dtype=int)
    g[0, 0, 1] = 1  # self-loop, lag 1
    g[0, 1, 2] = 1  # X0 -> X1 @2
    g[1, 2, 1] = 1  # X1 -> X2 @1
    d = diagnostics_from_graph(g, ["X0", "X1", "X2"])
    assert d["n_edges"] == 3
    assert d["self_loops"] == 1
    assert d["contemporaneous"] == 0 and d["lagged"] == 3
    assert d["max_in_degree"] == 1
    assert d["empty"] is False and d["saturated"] is False


def test_diagnostics_empty_and_saturated_flags():
    assert diagnostics_from_graph(np.zeros((3, 3, 2), dtype=int), ["a", "b", "c"])[
        "empty"
    ]
    full = np.ones((3, 3, 2), dtype=int)
    assert diagnostics_from_graph(full, ["a", "b", "c"])["saturated"]


def _write_csv(path, T=80, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(T)
    x1 = np.zeros(T)
    x2 = np.zeros(T)
    for t in range(1, T):
        x1[t] = 0.7 * x0[t - 1] + 0.3 * rng.standard_normal()
        x2[t] = 0.6 * x1[t - 1] + 0.3 * rng.standard_normal()
    pd.DataFrame({"X0": x0, "X1": x1, "X2": x2}).to_csv(path, index=False)


def _extract_json(output):
    return json.loads(output[output.index("{") :])


def test_discover_validate_annotates_persistence():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_csv("data.csv")
        result = runner.invoke(
            main,
            [
                "--quiet",
                "-o",
                "out",
                "discover",
                "data.csv",
                "--json",
                "--algorithm",
                "cdnots",
                "--ci-test",
                "parcorr-gpu",
                "--max-lag",
                "2",
                "--no-plot",
                "--validate",
                "--n-bootstrap",
                "5",
            ],
        )
        assert result.exit_code == 0, result.output
        s = _extract_json(result.output)
        assert "diagnostics" in s and "hub" in s["diagnostics"]
        assert s["stability"]["n_bootstrap"] == 5
        assert s["edges"], "expected some edges"
        for e in s["edges"]:
            assert "persistence" in e and 0.0 <= e["persistence"] <= 1.0


def test_validate_rejected_for_grace():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_csv("data.csv")
        result = runner.invoke(
            main,
            ["-o", "out", "discover", "data.csv", "--algorithm", "grace", "--validate"],
        )
        assert result.exit_code != 0
        assert "not supported for GRACE" in result.output


def test_pvalues_rejected_for_grace():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_csv("data.csv")
        result = runner.invoke(
            main,
            ["-o", "out", "discover", "data.csv", "--algorithm", "grace", "--pvalues"],
        )
        assert result.exit_code != 0
        assert "not supported for GRACE" in result.output


def test_cdnots_forwards_c_preset():
    # regression: c_preset was computed by recommend_config and documented in
    # SKILL.md/--c-preset help as applying to CDNOTS, but the cdnots branch of
    # `discover` never forwarded it to run_cdnots -- silently falling back to
    # the function's default preset regardless of --c-preset.
    runner = CliRunner()
    with runner.isolated_filesystem():
        T = 200
        t = np.arange(T)
        rng = np.random.default_rng(1)
        x0 = 5 * np.sin(2 * np.pi * t / 50) + 0.05 * t + rng.standard_normal(T)
        x1 = np.zeros(T)
        for k in range(1, T):
            x1[k] = 0.6 * x0[k - 1] + 0.3 * rng.standard_normal()
        pd.DataFrame({"X0": x0, "X1": x1}).to_csv("data.csv", index=False)

        result = runner.invoke(
            main,
            [
                "--quiet",
                "-o",
                "out",
                "discover",
                "data.csv",
                "--algorithm",
                "cdnots",
                "--ci-test",
                "parcorr-gpu",
                "--max-lag",
                "2",
                "--no-plot",
                "--include-c",
                "--c-preset",
                "linear+sin",
            ],
        )
        assert result.exit_code == 0, result.output
        d = os.path.join("out", os.listdir("out")[0])
        graph = np.load(os.path.join(d, "estimated_graph.npy"))
        # linear+sin appends 2 C columns; linear (the old silent fallback) is 1.
        assert graph.shape[0] == 2 + 2  # 2 data vars + 2 C(linear+sin) columns


def test_pacf_max_lag_not_inflated_on_ar_chain():
    # AR(0.4) chain: PACF should suggest a small lag, not pin to the cap (5)
    T, d = 400, 4
    r = np.random.default_rng(0)
    a = r.standard_normal((T, d))
    for t in range(1, T):
        for i in range(1, d):
            a[t, i] += 0.4 * a[t - 1, i - 1]
    df = pd.DataFrame(a, columns=[f"X{i}" for i in range(d)])
    assert inspect_df(df)["facts"]["suggested_max_lag"] <= 3


def test_discover_df_returns_graph():
    T = 80
    r = np.random.default_rng(0)
    x0 = r.standard_normal(T)
    x1 = np.zeros(T)
    for t in range(1, T):
        x1[t] = 0.7 * x0[t - 1] + 0.3 * r.standard_normal()
    df = pd.DataFrame({"X0": x0, "X1": x1})
    res = discover_df(df, algorithm="cdnots", ci_test="parcorr-gpu", max_lag=2)
    assert res.cg_tig.shape[:2] == (2, 2)
    assert res.cg_tig.shape[2] == 3  # max_lag + 1


def test_pvalues_flag_populates_edges():
    runner = CliRunner()
    base = [
        "discover",
        "data.csv",
        "--json",
        "--algorithm",
        "cdnots",
        "--ci-test",
        "parcorr-gpu",
        "--max-lag",
        "2",
        "--no-plot",
    ]
    with runner.isolated_filesystem():
        _write_csv("data.csv")
        # default: p-values off (high-dim safeguard) -> all null
        s0 = _extract_json(runner.invoke(main, ["--quiet", "-o", "o0"] + base).output)
        assert all(e["pvalue"] is None for e in s0["edges"])
        # --pvalues: opt in -> at least one real p-value
        s1 = _extract_json(
            runner.invoke(main, ["--quiet", "-o", "o1"] + base + ["--pvalues"]).output
        )
        assert any(e["pvalue"] is not None for e in s1["edges"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
