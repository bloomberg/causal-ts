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
        ("splitkci", "loses power at small T; performance improves with larger T."),
        ("dfcit", "found). Results vary by graph structure."),
        ("rcot", "sized to the sample; deterministic (seed=42)."),
        (
            "cmiknn-gpu",
            "The permutation null stops early once the p-value is decided.",
        ),
        ("linsig", "Captures temporal structure that pointwise tests ignore."),
        ("sigkci", "Deprecated alias for linsig; will be removed in a future release."),
        ("raft", "channel for power at small T."),
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


def test_generate_seed_zero_is_kept(tmp_path):
    """-s 0 was replaced by the default seed 42."""
    import json

    res = CliRunner().invoke(
        main,
        ["-o", str(tmp_path), "-s", "0", "generate", "--dataset", "ex1", "-T", "50"],
    )
    assert res.exit_code == 0, res.output
    (meta,) = tmp_path.glob("*/meta.json")
    assert json.loads(meta.read_text())["seed"] == 0


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


def test_discover_cedar_json_pvalues_are_valid_json(tmp_path):
    """Untested Cedar self-loops emit pvalue null, not 1.0 (or bare NaN)."""
    import json

    rng = np.random.default_rng(0)
    x = np.zeros((300, 3))
    for t in range(1, 300):
        x[t] = 0.6 * x[t - 1] + rng.standard_normal(3)
        x[t, 1] += 0.8 * x[t - 1, 0]
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(x, columns=["A", "B", "D"]).to_csv(csv_path, index=False)
    res = CliRunner().invoke(
        main,
        [
            "-o",
            str(tmp_path),
            "discover",
            str(csv_path),
            "--algorithm",
            "cedar",
            "--ci-test",
            "parcorr-gpu",
            "--max-lag",
            "1",
            "--no-c",
            "--pvalues",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.output
    summary = json.loads(res.output[res.output.index("{") :])
    loops = [e for e in summary["edges"] if e["source"] == e["target"]]
    assert loops and all(e["pvalue"] is None for e in loops)


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


def _evaluate_metrics(tmp_path, gt, pred):
    import json

    np.save(tmp_path / "true.npy", gt)
    np.save(tmp_path / "est.npy", pred)
    result = CliRunner().invoke(
        main,
        [
            "-o",
            str(tmp_path),
            "evaluate",
            str(tmp_path / "true.npy"),
            str(tmp_path / "est.npy"),
            "--no-plot",
        ],
    )
    assert result.exit_code == 0, result.output
    (path,) = tmp_path.glob("evaluate_*/metrics.json")
    return json.loads(path.read_text())


def test_evaluate_counts_false_positives_beyond_true_max_lag(tmp_path):
    """Extra lags in the prediction were trimmed away, hiding their FPs."""
    gt = np.zeros((3, 3, 2), dtype=int)
    gt[0, 1, 1] = 1
    pred = np.zeros((3, 3, 6), dtype=int)
    pred[0, 1, 1] = 1
    pred[1, 2, 3] = pred[2, 0, 4] = pred[0, 2, 5] = 1
    m = _evaluate_metrics(tmp_path, gt, pred)
    assert (m["TP"], m["FP"], m["FN"]) == (1, 3, 0)
    assert m["Precision"] == 0.25


def test_evaluate_counts_missed_edges_beyond_predicted_max_lag(tmp_path):
    gt = np.zeros((3, 3, 4), dtype=int)
    gt[0, 1, 1] = gt[1, 2, 3] = 1
    pred = np.zeros((3, 3, 2), dtype=int)
    pred[0, 1, 1] = 1
    m = _evaluate_metrics(tmp_path, gt, pred)
    assert (m["TP"], m["FP"], m["FN"]) == (1, 0, 1)


def test_evaluate_drops_extra_c_variable(tmp_path):
    gt = np.zeros((3, 3, 2), dtype=int)
    gt[0, 1, 1] = 1
    pred = np.zeros((4, 4, 2), dtype=int)
    pred[0, 1, 1] = pred[3, 0, 0] = 1
    m = _evaluate_metrics(tmp_path, gt, pred)
    assert (m["TP"], m["FP"], m["FN"]) == (1, 0, 0)


def test_evaluate_counts_true_edges_on_variables_the_prediction_lacks(tmp_path):
    """Cutting the ground truth to the prediction's variables hid these FNs."""
    gt = np.zeros((4, 4, 2), dtype=int)
    gt[0, 1, 1] = gt[1, 3, 1] = gt[3, 2, 0] = 1
    pred = np.zeros((3, 3, 2), dtype=int)
    pred[0, 1, 1] = 1
    m = _evaluate_metrics(tmp_path, gt, pred)
    assert (m["TP"], m["FP"], m["FN"]) == (1, 0, 2)


def test_discover_ground_truth_pads_lags(tmp_path):
    """discover --ground-truth must not trim the prediction's extra lags."""
    import json

    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 3))
    x[1:, 1] += 0.9 * x[:-1, 0]
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(x, columns=["A", "B", "D"]).to_csv(csv_path, index=False)
    gt = np.zeros((3, 3, 1), dtype=int)
    np.save(tmp_path / "gt.npy", gt)

    result = CliRunner().invoke(
        main,
        [
            "-o",
            str(tmp_path),
            "discover",
            str(csv_path),
            "--ci-test",
            "parcorr-gpu",
            "--max-lag",
            "1",
            "--no-c",
            "--ground-truth",
            str(tmp_path / "gt.npy"),
        ],
    )
    assert result.exit_code == 0, result.output
    (outdir,) = (p.parent for p in tmp_path.glob("*/metrics.json"))
    pred = np.load(outdir / "estimated_graph.npy")
    assert pred[0, 1, 1]
    m = json.loads((outdir / "metrics.json").read_text())
    # GT is empty, so every scored edge (all but lag-0 self-loops) is an FP;
    # the old lag-0-only trimming could never reach this count.
    p = pred.astype(bool)
    assert m["FP"] == int(p.sum() - np.trace(p[:, :, 0]))


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


def test_generate_henon_bounded_start_keeps_a_diverging_seed_finite(tmp_path):
    """Seed 3 diverges with the default start; --bounded-start keeps it finite."""
    import json

    runner = CliRunner()

    def generate(*flags):
        out = tmp_path / ("bounded" if flags else "default")
        result = runner.invoke(
            main,
            ["-o", str(out), "-s", "3", "generate", "--dataset", "henon", *flags],
        )
        assert result.exit_code == 0, result.output
        (csv,) = out.glob("generate_*/data.csv")
        meta = json.loads((csv.parent / "meta.json").read_text())
        assert meta.get("bounded_start", False) == bool(flags)
        return result.output, pd.read_csv(csv).to_numpy(dtype=float)

    output, data = generate()
    assert not np.isfinite(data).all()  # default unchanged
    assert "non-finite" in output and "--bounded-start" in output
    output, data = generate("--bounded-start")
    assert np.isfinite(data).all()
    assert "non-finite" not in output


def test_generate_bounded_start_is_ignored_with_a_warning_without_henon(tmp_path):
    result = CliRunner().invoke(
        main, ["-o", str(tmp_path), "generate", "-n", "3", "--bounded-start"]
    )
    assert result.exit_code == 0, result.output
    assert "--bounded-start has no effect with random generation" in result.output


def test_discover_alpha_defaults_per_algorithm(tmp_path):
    """CDNOTS+ runs at 0.01 by default (like run_cdnots_plus), CDNOTS at 0.05,
    and an explicit --alpha always wins."""
    import inspect
    import json

    from causalts.cdnots.phase3_utils import run_cdnots, run_cdnots_plus

    assert inspect.signature(run_cdnots_plus).parameters["alpha"].default == 0.01
    assert inspect.signature(run_cdnots).parameters["alpha"].default == 0.05

    rng = np.random.default_rng(0)
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(rng.standard_normal((80, 3)), columns=list("ABC")).to_csv(
        csv_path, index=False
    )
    for algorithm, extra, expected in (
        ("cdnots+", [], 0.01),
        ("cdnots", [], 0.05),
        ("cdnots+", ["--alpha", "0.2"], 0.2),
    ):
        result = CliRunner().invoke(
            main,
            [
                "--quiet",
                "-o",
                str(tmp_path),
                "discover",
                str(csv_path),
                "--algorithm",
                algorithm,
                "--ci-test",
                "parcorr-gpu",
                "--max-lag",
                "1",
                "--no-plot",
                "--json",
                *extra,
            ],
        )
        assert result.exit_code == 0, result.output
        summary = json.loads(result.output[result.output.index("{") :])
        assert summary["alpha"] == expected, (algorithm, extra)
