# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression test: `discover --algorithm cdnots+` was a silent no-op.

`cdnots+` is a valid ALGORITHM_CHOICES entry, but the `discover` command's
dispatch chain only matched the literal string "cdnots" -- so a cdnots+ run
exited 0, logged "Results saved to ...", and wrote a summary.json claiming
algorithm=cdnots+, but never wrote estimated_graph.npy or computed anything.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from causalts.cli import main


def _write_csv(path, T=100, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(T)
    x1 = np.zeros(T)
    x2 = np.zeros(T)
    for t in range(1, T):
        x1[t] = 0.7 * x0[t - 1] + 0.3 * rng.standard_normal()
        x2[t] = 0.6 * x1[t - 1] + 0.3 * rng.standard_normal()
    pd.DataFrame({"X0": x0, "X1": x1, "X2": x2}).to_csv(path, index=False)


def test_cdnots_plus_actually_runs():
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
                "--algorithm",
                "cdnots+",
                "--ci-test",
                "parcorr-gpu",
                "--max-lag",
                "2",
                "--no-plot",
            ],
        )
        assert result.exit_code == 0, result.output
        outdir = os.path.join("out", os.listdir("out")[0])
        graph_path = os.path.join(outdir, "estimated_graph.npy")
        assert os.path.exists(graph_path), "cdnots+ produced no estimated_graph.npy"
        graph = np.load(graph_path)
        # 3 data vars + 1 C column (include_C defaults to True)
        assert graph.shape == (4, 4, 3)
        with open(os.path.join(outdir, "summary.json")) as f:
            summary = json.load(f)
        assert "n_edges" in summary, "summary missing n_edges -- discovery never ran"


def test_cdnots_plus_supports_pvalues():
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
                "--algorithm",
                "cdnots+",
                "--ci-test",
                "parcorr-gpu",
                "--max-lag",
                "2",
                "--no-plot",
                "--pvalues",
            ],
        )
        assert result.exit_code == 0, result.output
        outdir = os.path.join("out", os.listdir("out")[0])
        assert os.path.exists(os.path.join(outdir, "pvalues.npy"))


def test_cdnots_plus_supports_validate():
    # regression: the --validate bootstrap closure's _disc() passed
    # stable=stable unconditionally; run_cdnots_plus has no `stable` param
    # (and no **kwargs), so `discover --algorithm cdnots+ --validate` raised
    # TypeError: run_cdnots_plus() got an unexpected keyword argument 'stable'.
    # --validate is only rejected for GRACE, so this path was reachable.
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
                "cdnots+",
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
        summary = json.loads(result.output[result.output.index("{") :])
        assert "stability" in summary
        assert summary["edges"], "expected some edges"
        for e in summary["edges"]:
            assert "persistence" in e and 0.0 <= e["persistence"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
