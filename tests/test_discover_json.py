# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Test that `discover --json` emits a parseable summary with a named edge list."""

import json
import os

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from causalts.cli import main


def _write_csv(path, T=60, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(T)
    x1 = np.zeros(T)
    x2 = np.zeros(T)
    for t in range(1, T):
        x1[t] = 0.7 * x0[t - 1] + 0.3 * rng.standard_normal()
        x2[t] = 0.6 * x1[t - 1] + 0.3 * rng.standard_normal()
    pd.DataFrame({"X0": x0, "X1": x1, "X2": x2}).to_csv(path, index=False)


def _extract_json(output):
    """Pull the JSON object out of mixed CLI output."""
    start = output.index("{")
    return json.loads(output[start:])


def test_discover_json_has_edges():
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
            ],
        )
        assert result.exit_code == 0, result.output
        summary = _extract_json(result.output)
        assert summary["algorithm"] == "cdnots"
        assert "edges" in summary and isinstance(summary["edges"], list)
        for e in summary["edges"]:
            assert set(("source", "target", "lag", "pvalue")) <= set(e)
        # artifacts written to -o dir
        assert any(
            os.path.exists(os.path.join(root, "estimated_graph.npy"))
            for root, _, _ in os.walk("out")
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
