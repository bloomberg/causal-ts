# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""create_ci_test: device handling, mixed-data defaults and package exports."""

import numpy as np
import pandas as pd
from click.testing import CliRunner

from causalts.ci_tests import create_ci_test


def test_device_is_dropped_for_cpu_only_tests():
    test = create_ci_test(
        "parcorr", np.random.default_rng(0).normal(size=(50, 3)), device="cpu"
    )
    assert type(test).__name__ == "PartialCorr"


def test_cmiknn_mixed_detects_discrete_cols_by_default():
    rng = np.random.default_rng(0)
    data = np.c_[rng.normal(size=(400, 2)), rng.integers(0, 3, 400)]
    assert create_ci_test("cmiknn-mixed-gpu", data, device="cpu").discrete_cols == [2]
    explicit = create_ci_test("cmiknn-mixed-gpu", data, device="cpu", discrete_cols=[])
    assert explicit.discrete_cols == []


def test_cmiknn_mixed_assumes_continuous_on_tiny_placeholders():
    test = create_ci_test("cmiknn-mixed-gpu", np.zeros((2, 2)), device="cpu")
    assert test.discrete_cols == []


def test_cli_rejects_cmiknn_mixed_with_cedar():
    from causalts.cli import main

    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(100, 2)), columns=list("ab"))
    runner = CliRunner()
    with runner.isolated_filesystem():
        df.to_csv("data.csv", index=False)
        res = runner.invoke(
            main,
            [
                "-o",
                "out",
                "discover",
                "data.csv",
                "-a",
                "cedar",
                "--ci-test",
                "cmiknn-mixed-gpu",
            ],
        )
    assert res.exit_code != 0 and "not supported" in res.output


def test_stratified_cit_is_exported():
    from causalts.ci_tests import StratifiedCIT
    from causalts.ci_tests.stratified_cit import StratifiedCIT as direct

    assert StratifiedCIT is direct


def test_cli_discover_accepts_cmiknn_mixed():
    from causalts.cli import main

    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        np.c_[rng.normal(size=(150, 2)), rng.integers(0, 3, 150)], columns=list("abc")
    )
    runner = CliRunner()
    with runner.isolated_filesystem():
        df.to_csv("data.csv", index=False)
        res = runner.invoke(
            main,
            [
                "-o",
                "out",
                "discover",
                "data.csv",
                "--ci-test",
                "cmiknn-mixed-gpu",
                "--max-lag",
                "1",
            ],
        )
    assert res.exit_code == 0, res.output
