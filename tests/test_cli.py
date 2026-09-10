# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""CLI smoke tests."""

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from causalts.cli import CI_TEST_CHOICES, CI_TEST_GUIDE, main


@pytest.mark.parametrize("options", [[], ["--test", "all"]])
def test_ci_test_info(options):
    runner = CliRunner()
    result = runner.invoke(main, ["ci-test-info", *options])
    assert result.exit_code == 0
    assert result.output == CI_TEST_GUIDE + "\n"


@pytest.mark.parametrize(
    "test_name, last_line",
    [
        ("parcorr-gpu", "Works at any T and any d. Misses nonlinear edges."),
        ("gcmi", "(monotone) nonlinearity only; misses non-monotone dependencies."),
        ("kci", "Most general kernel test. O(T^3) — slow at large T."),
        ("splitkci", "data. Performance improves with larger T."),
        ("dfcit", "columns natively. Results vary by graph structure."),
        ("rcot", "Fast. At T<=300, auto-averages 5 RFF draws to reduce variance."),
        (
            "cmiknn-gpu",
            "The permutation null stops early once the p-value is decided.",
        ),
        ("sigkci", "Captures temporal structure that pointwise tests ignore."),
    ],
)
def test_ci_test_info_filters_summary(test_name, last_line):
    result = CliRunner().invoke(main, ["ci-test-info", "--test", test_name])

    assert result.exit_code == 0
    assert result.output.rstrip().endswith(last_line)
    headings = [
        name
        for name in CI_TEST_CHOICES
        if any(line.startswith(f"  {name} ") for line in result.output.splitlines())
    ]
    assert headings == [test_name]
    assert "When to use what" not in result.output
    assert "Key tradeoffs" not in result.output


def test_every_registered_ci_test_has_a_guide_section():
    """A CI test registered without a guide section makes --test <name> fail."""
    sections = CI_TEST_GUIDE.split("\n\n")
    missing = [
        name
        for name in CI_TEST_CHOICES
        if not any(section.startswith(f"  {name} ") for section in sections)
    ]
    assert missing == []


@pytest.mark.parametrize("test_name", CI_TEST_CHOICES)
def test_ci_test_info_selects_every_registered_test(test_name):
    result = CliRunner().invoke(main, ["ci-test-info", "--test", test_name])

    assert result.exit_code == 0
    headings = [
        name
        for name in CI_TEST_CHOICES
        if any(line.startswith(f"  {name} ") for line in result.output.splitlines())
    ]
    assert headings == [test_name]
    assert "When to use what" not in result.output


def test_ci_test_info_invalid_choice():
    result = CliRunner().invoke(main, ["ci-test-info", "--test", "unknown"])

    assert result.exit_code == 2
    assert "Invalid value for '--test'" in result.output


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


def _validate_fixture(tmp_path, max_lag):
    """A small AR chain plus a graph with the requested lag depth."""
    rng = np.random.default_rng(0)
    n = 600
    x = np.zeros((n, 3))
    for t in range(1, n):
        x[t, 0] = 0.5 * x[t - 1, 0] + rng.standard_normal()
        x[t, 1] = 0.6 * x[t - 1, 0] + rng.standard_normal()
        x[t, 2] = 0.4 * x[t - 1, 1] + 0.7 * x[t - 1, 0] + rng.standard_normal()
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(x, columns=["V0", "V1", "V2"]).to_csv(csv_path, index=False)

    g = np.zeros((3, 3, max_lag + 1), dtype=int)
    if max_lag >= 1:
        g[0, 0, 1] = g[0, 1, 1] = g[1, 2, 1] = g[0, 2, 1] = 1
    graph_path = tmp_path / "graph.npy"
    np.save(graph_path, g)
    return graph_path, csv_path


def test_dowhy_validate_transition(tmp_path):
    graph_path, csv_path = _validate_fixture(tmp_path, max_lag=1)
    result = CliRunner().invoke(
        main,
        [
            "dowhy",
            "validate",
            str(graph_path),
            "--data",
            str(csv_path),
            "--test",
            "transition",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Transition-Structure Validation" in result.output


def test_dowhy_validate_history_skips_a_contemporaneous_only_graph(tmp_path):
    """max_lag=0 has no history window; it must skip, not abort --test all."""
    graph_path, csv_path = _validate_fixture(tmp_path, max_lag=0)
    result = CliRunner().invoke(
        main,
        [
            "dowhy",
            "validate",
            str(graph_path),
            "--data",
            str(csv_path),
            "--test",
            "history",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "contemporaneous-only" in result.output
