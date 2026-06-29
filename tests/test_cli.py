# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""CLI smoke tests."""

import numpy as np
import pandas as pd
from click.testing import CliRunner

from causalts.cli import main


def test_ci_test_info():
    runner = CliRunner()
    result = runner.invoke(main, ["ci-test-info"])
    assert result.exit_code == 0
    assert "parcorr" in result.output.lower()


def test_generate_ex1(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["-o", str(tmp_path), "-s", "42", "generate", "--dataset", "ex1", "-T", "50"],
    )
    assert result.exit_code == 0


def test_discover_cdnots(tmp_path):
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.standard_normal((60, 3)), columns=["A", "B", "C"])
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "-o",
            str(tmp_path),
            "-s",
            "42",
            "discover",
            str(csv_path),
            "--algorithm",
            "cdnots",
            "--ci-test",
            "parcorr-gpu",
            "--max-lag",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output


def test_evaluate(tmp_path):
    g = np.zeros((3, 3, 2), dtype=int)
    g[0, 1, 1] = 1
    true_path = tmp_path / "true.npy"
    est_path = tmp_path / "est.npy"
    np.save(true_path, g)
    np.save(est_path, g)

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["evaluate", str(true_path), str(est_path)],
    )
    assert result.exit_code == 0
    assert "F1" in result.output or "f1" in result.output.lower()
