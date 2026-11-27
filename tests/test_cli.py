# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""CLI smoke tests."""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

import causalts.cli as cli_module
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


# ---------------------------------------------------------------------------
# Non-numeric columns and --index-col (#130)
# ---------------------------------------------------------------------------


def _dated_csv(tmp_path, periods=200):
    """A CSV with a date column next to two numeric series, as in #130."""
    rng = np.random.default_rng(0)
    path = tmp_path / "dated.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=periods).strftime("%Y-%m-%d"),
            "x": rng.normal(size=periods),
            "y": rng.normal(size=periods),
        }
    ).to_csv(path, index=False)
    return path


def test_discover_names_non_numeric_columns_before_running(tmp_path):
    out = tmp_path / "out"
    result = CliRunner().invoke(
        main,
        ["-o", str(out), "discover", str(_dated_csv(tmp_path)), "--max-lag", "1"],
    )
    assert result.exit_code == 2, result.output
    assert "column(s) ['date'] are not numeric; drop it or pass" in result.output
    assert "--index-col date" in result.output
    assert not out.exists()  # rejected before the output directory is made


def test_non_numeric_hint_for_several_columns(tmp_path):
    path = tmp_path / "two.csv"
    pd.DataFrame(
        {"date": ["d1", "d2", "d3"], "region": ["a", "b", "c"], "x": [1.0, 2.0, 3.0]}
    ).to_csv(path, index=False)
    result = CliRunner().invoke(
        main, ["-o", str(tmp_path / "out"), "discover", str(path)]
    )
    assert result.exit_code == 2, result.output
    assert "drop them; one of them can be --index-col instead" in result.output


def test_non_numeric_column_hint_is_shell_quoted(tmp_path):
    path = tmp_path / "spaced.csv"
    pd.DataFrame({"Trade Date": ["a", "b", "c"], "x": [1.0, 2.0, 3.0]}).to_csv(
        path, index=False
    )
    result = CliRunner().invoke(
        main, ["-o", str(tmp_path / "out"), "discover", str(path)]
    )
    assert result.exit_code == 2, result.output
    assert "--index-col 'Trade Date'" in result.output


def test_non_numeric_hint_drops_index_col_advice_when_one_is_set(tmp_path):
    path = tmp_path / "labelled.csv"
    pd.DataFrame(
        {"date": ["d1", "d2", "d3"], "label": ["a", "b", "c"], "x": [1.0, 2.0, 3.0]}
    ).to_csv(path, index=False)
    result = CliRunner().invoke(
        main,
        ["-o", str(tmp_path / "out"), "discover", str(path), "--index-col", "date"],
    )
    assert result.exit_code == 2, result.output
    assert "column(s) ['label'] are not numeric; drop them" in result.output
    assert "--index-col label" not in result.output


@pytest.mark.parametrize("index_col", ["date", "0"])
def test_discover_index_col_is_left_out_of_the_variables(tmp_path, index_col):
    import json

    result = CliRunner().invoke(
        main,
        [
            "--quiet",
            "-o",
            str(tmp_path),
            "discover",
            str(_dated_csv(tmp_path)),
            "--ci-test",
            "parcorr-gpu",
            "--max-lag",
            "1",
            "--no-plot",
            "--json",
            "--index-col",
            index_col,
        ],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output[result.output.index("{") :])
    assert summary["var_names"] == ["x", "y"]
    assert summary["data_shape"] == [200, 2]
    assert summary["index_col"] == index_col


def test_inspect_index_col_is_left_out_of_the_report(tmp_path):
    import json

    result = CliRunner().invoke(
        main, ["inspect", str(_dated_csv(tmp_path)), "--index-col", "date"]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["data"]["var_names"] == ["x", "y"]
    assert not any("Non-numeric" in w for w in report["warnings"])


def test_index_col_the_file_lacks_is_a_usage_error(tmp_path):
    result = CliRunner().invoke(
        main, ["inspect", str(_dated_csv(tmp_path)), "--index-col", "time"]
    )
    assert result.exit_code == 2
    assert "--index-col" in result.output
    assert "['date', 'x', 'y']" in result.output


def test_discover_accepts_a_numeric_categorical_column(tmp_path):
    pytest.importorskip("pyarrow")
    rng = np.random.default_rng(0)
    path = tmp_path / "d.feather"
    pd.DataFrame(
        {
            "x": rng.normal(size=200),
            "c": pd.Categorical(rng.integers(0, 2, size=200)),
        }
    ).to_feather(path)
    result = CliRunner().invoke(
        main,
        [
            "--quiet",
            "-o",
            str(tmp_path),
            "discover",
            str(path),
            "--ci-test",
            "parcorr-gpu",
            "--max-lag",
            "1",
            "--no-plot",
        ],
    )
    assert result.exit_code == 0, result.output


def test_deconfound_names_non_numeric_columns_and_takes_index_col(tmp_path):
    import json

    graph = np.zeros((2, 2, 2), dtype=int)
    graph[0, 1, 1] = 1
    np.save(tmp_path / "graph.npy", graph)
    out = tmp_path / "out"
    args = [
        "-o",
        str(out),
        "deconfound",
        str(tmp_path / "graph.npy"),
        "--data",
        str(_dated_csv(tmp_path)),
        "--strategy",
        "pds",
    ]

    rejected = CliRunner().invoke(main, args)
    assert rejected.exit_code == 2, rejected.output
    assert "--index-col date" in rejected.output
    assert not out.exists()

    # A graph with fewer variables than the data columns is a usage error.
    rng = np.random.default_rng(1)
    pd.DataFrame(
        {"t": range(200), "x": rng.normal(size=200), "y": rng.normal(size=200)}
    ).to_csv(tmp_path / "numeric_index.csv", index=False)
    mismatch = CliRunner().invoke(
        main, [*args[:5], str(tmp_path / "numeric_index.csv"), *args[6:]]
    )
    assert mismatch.exit_code == 2, mismatch.output
    assert "pass the --index-col you discovered with" in mismatch.output
    assert not out.exists()

    accepted = CliRunner().invoke(main, [*args, "--index-col", "date"])
    assert accepted.exit_code == 0, accepted.output
    (summary_path,) = out.glob("deconfound_pds_*/summary.json")
    assert json.loads(summary_path.read_text())["n_vars"] == 2


def test_deconfound_rejects_an_index_col_discovery_did_not_use(tmp_path):
    rng = np.random.default_rng(0)
    data = tmp_path / "t.csv"
    pd.DataFrame(
        {"t": np.arange(200.0), "x": rng.normal(size=200), "y": rng.normal(size=200)}
    ).to_csv(data, index=False)
    disc = CliRunner().invoke(
        main,
        [
            "--quiet",
            "-o",
            str(tmp_path / "disc"),
            "discover",
            str(data),
            "--ci-test",
            "parcorr-gpu",
            "--max-lag",
            "1",
            "--no-plot",
        ],
    )
    assert disc.exit_code == 0, disc.output
    (graph,) = (tmp_path / "disc").glob("discover_*/estimated_graph.npy")
    args = ["-o", str(tmp_path / "out"), "deconfound", str(graph), "--data", str(data)]

    # Discovered on (t, x, y); with --index-col t the data has only (x, y).
    rejected = CliRunner().invoke(main, [*args, "--index-col", "t"])
    assert rejected.exit_code == 2, rejected.output
    assert "discovered on 3 variables" in rejected.output
    assert "discover used no --index-col; drop it" in rejected.output
    assert not (tmp_path / "out").exists()
    # --var-names that fit the discovered graph still get the --index-col hint.
    named = CliRunner().invoke(
        main, [*args, "--index-col", "t", "--var-names", "t,x,y"]
    )
    assert named.exit_code == 2, named.output
    assert "discover used no --index-col; drop it" in named.output
    assert not (tmp_path / "out").exists()

    assert CliRunner().invoke(main, args).exit_code == 0


def _write_four_column_csv(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.mark.parametrize("names", ["a,b", "a,a,b,c"])
def test_inspect_var_names_wrong_count_or_duplicate(tmp_path, names):
    # A wrong count or a duplicate name is a CLI usage error (#133),
    # not a raw pandas traceback.
    path = _write_four_column_csv(tmp_path)
    result = CliRunner().invoke(main, ["inspect", path, "--var-names", names])
    assert result.exit_code == 2
    assert "--var-names" in result.output


def test_evaluate_var_names_wrong_count(tmp_path):
    # evaluate/plot check --var-names against the graph size (#133).
    gt = np.zeros((2, 2, 2), dtype=np.int8)
    pred = np.zeros((2, 2, 2), dtype=np.int8)
    gt_path = tmp_path / "gt.npy"
    pred_path = tmp_path / "pred.npy"
    np.save(gt_path, gt)
    np.save(pred_path, pred)
    out = tmp_path / "out"
    for cmd in (
        ["evaluate", str(gt_path), str(pred_path)],
        ["plot", str(gt_path)],
    ):
        result = CliRunner().invoke(
            main, ["-o", str(out), *cmd, "--var-names", "a,b,c"]
        )
        assert result.exit_code == 2, result.output
        assert "--var-names" in result.output
        # A usage error leaves no output directory behind.
        assert not out.exists()


class _FrozenClock:
    @staticmethod
    def now():
        return datetime(2026, 1, 2, 3, 4, 5)


def test_make_output_dir_same_second_gets_new_dir(tmp_path, monkeypatch):
    # Runs that start in the same second must not share a directory (#166).
    monkeypatch.setattr(cli_module, "datetime", _FrozenClock)
    base = tmp_path / "out"
    first = cli_module._make_output_dir(str(base), "x")
    second = cli_module._make_output_dir(str(base), "x")
    third = cli_module._make_output_dir(str(base), "x")
    assert first == str(base / "x_20260102_030405")
    assert second == str(base / "x_20260102_030405_1")
    assert third == str(base / "x_20260102_030405_2")
    assert all(os.path.isdir(p) for p in (first, second, third))


def test_generate_twice_in_same_second_keeps_both_runs(tmp_path, monkeypatch):
    # Two `generate` runs under a frozen clock write to separate directories
    # (#166); the second used to overwrite the first.
    monkeypatch.setattr(cli_module, "datetime", _FrozenClock)
    for seed in ("1", "2"):
        result = CliRunner().invoke(
            main,
            [
                "-o",
                str(tmp_path),
                "-s",
                seed,
                "generate",
                "--dataset",
                "ex2",
                "-T",
                "50",
            ],
        )
        assert result.exit_code == 0, result.output
    dirs = sorted(p.name for p in tmp_path.iterdir())
    assert dirs == ["generate_20260102_030405", "generate_20260102_030405_1"]
    seeds = [json.loads((tmp_path / d / "meta.json").read_text())["seed"] for d in dirs]
    assert seeds == [1, 2]


def test_discover_cdnots_default_saves_network_plot(tmp_path):
    # With the C node (the default), the plot used to get "C" twice and fail
    # the var_names check, so network.png was silently not written.
    import matplotlib

    matplotlib.use("Agg")
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    path = tmp_path / "d.csv"
    pd.DataFrame({"x": x, "y": np.r_[0, x[:-1]] + rng.normal(size=300)}).to_csv(
        path, index=False
    )
    result = CliRunner().invoke(
        main,
        [
            "-o",
            str(tmp_path),
            "discover",
            str(path),
            "--max-lag",
            "1",
            "--ci-test",
            "parcorr-gpu",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "could not save network plot" not in result.output
    assert list(tmp_path.glob("discover_*/network.png"))


def _tigramite_csv(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 3))
    for t in range(1, 300):
        x[t, 1] += 0.8 * x[t - 1, 0]
        x[t, 2] += 0.8 * x[t - 1, 1]
    path = tmp_path / "chain.csv"
    pd.DataFrame(x, columns=["X0", "X1", "X2"]).to_csv(path, index=False)
    return path


def _discover_json(tmp_path, *args):
    return CliRunner().invoke(
        main,
        ["-q", "-o", str(tmp_path), "discover", *args, "--no-plot", "--json"],
    )


def test_discover_pcmciplus_json_edges(tmp_path):
    pytest.importorskip("tigramite")
    path = _tigramite_csv(tmp_path)
    res = _discover_json(tmp_path, str(path), "--algorithm", "pcmciplus")
    assert res.exit_code == 0, res.output
    s = json.loads(res.output)
    assert s["algorithm"] == "pcmciplus"
    assert s["ci_test"] == "parcorr (tigramite)"
    assert s["alpha"] == 0.01 and s["alpha_parameter"] == "pc_alpha"
    edges = {(e["source"], e["target"], e["lag"]) for e in s["edges"]}
    assert {("X0", "X1", 1), ("X1", "X2", 1)} <= edges
    assert s["n_edges"] == len(s["edges"])
    (outdir,) = tmp_path.glob("discover_pcmciplus_*")
    assert np.load(outdir / "edge_marks.npy").dtype.kind == "U"


@pytest.mark.parametrize("algo", ["pcmci", "lpcmci"])
def test_discover_tigramite_alpha_and_pvalues(tmp_path, algo):
    pytest.importorskip("tigramite")
    path = _tigramite_csv(tmp_path)
    res = _discover_json(
        tmp_path,
        str(path),
        "-a",
        algo,
        "--max-lag",
        "1",
        "--alpha",
        "0.02",
        "--pvalues",
    )
    assert res.exit_code == 0, res.output
    s = json.loads(res.output)
    assert s["alpha"] == 0.02
    assert s["edges"] and all(e["pvalue"] is not None for e in s["edges"])
    (outdir,) = tmp_path.glob(f"discover_{algo}_*")
    assert np.load(outdir / "pvalues.npy").shape == (3, 3, 2)


def test_discover_tigramite_rejects_c_node_flags(tmp_path):
    path = _tigramite_csv(tmp_path)
    res = _discover_json(tmp_path, str(path), "-a", "pcmci", "--include-c")
    assert res.exit_code == 2
    assert "--include-c does not apply to --algorithm pcmci" in res.output


def test_discover_tigramite_without_tigramite_errors(tmp_path, monkeypatch):
    import causalts.tigramite_discovery as td

    monkeypatch.setattr(td, "tigramite_available", lambda: False)
    path = _tigramite_csv(tmp_path)
    res = _discover_json(tmp_path, str(path), "-a", "pcmciplus")
    assert res.exit_code == 1
    assert "requires tigramite" in res.output


def test_discover_tigramite_rejects_discrete_column_ci_test(tmp_path):
    pytest.importorskip("tigramite")
    path = _tigramite_csv(tmp_path)
    df = pd.read_csv(path)
    df["D"] = np.random.default_rng(1).integers(0, 3, len(df))
    df.to_csv(path, index=False)
    res = _discover_json(
        tmp_path, str(path), "-a", "pcmciplus", "--ci-test", "cmiknn-mixed-gpu"
    )
    assert res.exit_code == 1
    assert "cmiknn-mixed-gpu with cdnots" in res.output


def test_discover_tigramite_native_ci_test_choice(tmp_path):
    pytest.importorskip("tigramite")
    path = _tigramite_csv(tmp_path)
    res = _discover_json(
        tmp_path, str(path), "-a", "pcmciplus", "--ci-test", "robust_parcorr"
    )
    assert res.exit_code == 0, res.output
    edges = {
        (e["source"], e["target"], e["lag"]) for e in json.loads(res.output)["edges"]
    }
    assert {("X0", "X1", 1), ("X1", "X2", 1)} <= edges
    res = _discover_json(tmp_path, str(path), "-a", "cdnots", "--ci-test", "gpdc")
    assert res.exit_code == 2
    assert "gpdc is a tigramite CI test" in res.output


def test_discover_tigramite_overlapping_ci_test_is_noted_and_seeded(
    tmp_path, monkeypatch
):
    pytest.importorskip("tigramite")
    import causalts.tigramite_result as tr

    seen = {}
    real = tr.run_tigramite_algorithm

    def spy(*args, **kw):
        seen.update(kw)
        return real(*args, **kw)

    monkeypatch.setattr(tr, "run_tigramite_algorithm", spy)
    path = _tigramite_csv(tmp_path)
    res = CliRunner().invoke(
        main,
        ["-s", "7", "-d", "cpu", "-o", str(tmp_path), "discover", str(path)]
        + ["-a", "pcmciplus", "--ci-test", "parcorr", "--max-lag", "1", "--no-plot"],
    )
    assert res.exit_code == 0, res.output
    assert "runs tigramite's own parcorr, not causal-ts's" in res.output
    assert "--device cpu does not apply" in res.output
    assert seen["ci_test"] == "parcorr" and seen["ci_kwargs"] == {"seed": 7}


@pytest.mark.parametrize(
    "algo, extra, expected",
    [
        ("pcmciplus", [], 0.01),  # wrapper's pc_alpha, not the generic 0.05
        ("pcmciplus", ["--alpha", "0.002"], 0.002),
        ("pcmci", [], 0.05),  # min(pc_alpha 0.2, alpha_level 0.05)
    ],
)
def test_discover_tigramite_sizes_dfcit_for_the_wrappers_alpha(
    tmp_path, monkeypatch, algo, extra, expected
):
    pytest.importorskip("tigramite")
    import causalts.cli as cli_mod

    seen = []
    real = cli_mod._make_ci_test

    def spy(name, data, device, **kw):
        seen.append((name, kw.get("alpha")))
        # A fast stand-in: only the alpha dfcit would be built with matters.
        return real("parcorr-gpu", data, device, seed=kw.get("seed"))

    monkeypatch.setattr(cli_mod, "_make_ci_test", spy)
    path = _tigramite_csv(tmp_path)
    res = _discover_json(
        tmp_path, str(path), "-a", algo, "--ci-test", "dfcit", "--max-lag", "1", *extra
    )
    assert res.exit_code == 0, res.output
    assert seen and all(n == "dfcit" for n, _ in seen)
    assert all(a == pytest.approx(expected) for _, a in seen)


def test_discover_tigramite_summary_carries_marks(tmp_path):
    # Two copies of one noise series: PCMCI+ leaves the lag-0 link o-o, which
    # cg_tig (and so `edges`) leaves out; summary lists it as unresolved.
    pytest.importorskip("tigramite")
    rng = np.random.default_rng(0)
    x = rng.standard_normal(300)
    path = tmp_path / "pair.csv"
    pd.DataFrame({"a": x, "b": x + 0.1 * rng.standard_normal(300)}).to_csv(
        path, index=False
    )
    res = _discover_json(tmp_path, str(path), "-a", "pcmciplus", "--max-lag", "1")
    assert res.exit_code == 0, res.output
    s = json.loads(res.output)
    assert all(e["mark"] == "-->" for e in s["edges"])
    assert {"source": "a", "target": "b", "lag": 0, "mark": "o-o"} in s[
        "unresolved_links"
    ]
    assert not any(e["lag"] == 0 for e in s["edges"])


def test_discover_tigramite_validate_builds_a_ci_test_per_window(tmp_path, monkeypatch):
    pytest.importorskip("tigramite")
    import causalts.cli as cli_mod

    built = []
    real = cli_mod._make_ci_test

    def spy(name, data, device, **kw):
        obj = real("parcorr-gpu", data, device, seed=kw.get("seed"))
        built.append((obj, len(data), kw.get("alpha")))
        return obj

    monkeypatch.setattr(cli_mod, "_make_ci_test", spy)
    path = _tigramite_csv(tmp_path)
    res = _discover_json(
        tmp_path,
        str(path),
        "-a",
        "pcmciplus",
        "--ci-test",
        "dfcit",
        "--max-lag",
        "1",
        "--validate",
        "--n-bootstrap",
        "3",
        "--window-frac",
        "0.5",
    )
    assert res.exit_code == 0, res.output
    # Main-run test (built on a placeholder), then one per window on its rows.
    windows = built[1:]
    assert len(windows) == 3
    assert len({id(obj) for obj, _, _ in built}) == len(built)
    assert all(n == 150 for _, n, _ in windows)
    assert all(a == pytest.approx(0.01) for _, _, a in built)  # PCMCI+ pc_alpha
    s = json.loads(res.output)
    assert "binary graph" in s["stability"]["note"]


def _synthetic_tigramite_result(marks):
    import causalts.tigramite_discovery as td
    from causalts.tigramite_result import TigramiteResult

    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((100, 3)), columns=["X0", "X1", "X2"])
    g = np.full((3, 3, 2), "", dtype="<U3")
    for (i, j, k), m in marks.items():
        g[i, j, k] = m
    info = {"method": "lpcmci", "graph_raw": g, "p_matrix": np.full(g.shape, 0.5)}
    return TigramiteResult(td.tigramite_graph_to_binary(g), df, info)


@pytest.mark.parametrize(
    "marks, n_edges, n_unresolved",
    [
        # lag-0 o-o (stored both ways), lagged o->, lagged <->
        (
            {(0, 1, 0): "o-o", (1, 0, 0): "o-o", (1, 2, 1): "o->", (2, 0, 1): "<->"},
            1,
            2,
        ),
        ({(0, 1, 0): "o-o", (1, 0, 0): "o-o"}, 0, 1),  # only unresolved links
    ],
)
def test_discover_tigramite_diagnostics_count_unresolved_links(
    tmp_path, monkeypatch, marks, n_edges, n_unresolved
):
    pytest.importorskip("tigramite")
    import causalts.tigramite_result as tr

    res_obj = _synthetic_tigramite_result(marks)
    monkeypatch.setattr(tr, "run_tigramite_algorithm", lambda *a, **k: res_obj)
    path = _tigramite_csv(tmp_path)
    res = _discover_json(tmp_path, str(path), "-a", "lpcmci", "--max-lag", "1")
    assert res.exit_code == 0, res.output
    s = json.loads(res.output)
    d = s["diagnostics"]
    assert d["n_edges"] == n_edges and d["n_unresolved"] == n_unresolved
    assert d["empty"] is False
    assert len(s["unresolved_links"]) == n_unresolved
    if n_edges:
        assert s["edges"][0]["mark"] == "o->"
        assert {e["mark"] for e in s["unresolved_links"]} == {"o-o", "<->"}


def test_discover_gpdc_torch_without_gpytorch_names_it(tmp_path, monkeypatch):
    pytest.importorskip("tigramite")
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name == "gpytorch" else real(name, *a),
    )
    path = _tigramite_csv(tmp_path)
    res = _discover_json(
        tmp_path, str(path), "-a", "pcmciplus", "--ci-test", "gpdc_torch"
    )
    assert res.exit_code == 1
    assert "pip install gpytorch" in res.output
    assert "causal-ts[tigramite]" not in res.output


def test_discover_tigramite_lists_a_dropped_lag0_partial_arrow(tmp_path, monkeypatch):
    # A lag-0 X1 o-> X0 that cg_tig does not hold (e.g. thresholded away) is
    # in neither `edges` nor the binary graph: unresolved_links must list it.
    pytest.importorskip("tigramite")
    import causalts.tigramite_result as tr

    res_obj = _synthetic_tigramite_result({(1, 0, 0): "o->", (0, 1, 0): "<-o"})
    res_obj.cg_tig = np.zeros_like(res_obj.cg_tig)
    monkeypatch.setattr(tr, "run_tigramite_algorithm", lambda *a, **k: res_obj)
    path = _tigramite_csv(tmp_path)
    res = _discover_json(tmp_path, str(path), "-a", "lpcmci", "--max-lag", "1")
    assert res.exit_code == 0, res.output
    s = json.loads(res.output)
    assert s["edges"] == []
    assert len(s["unresolved_links"]) == 1
    assert s["unresolved_links"][0]["lag"] == 0
    assert s["diagnostics"]["empty"] is False


@pytest.mark.parametrize("algo", ["pcmciplus", "lpcmci"])
def test_discover_tigramite_impute_var_em_and_refusals(tmp_path, algo):
    pytest.importorskip("tigramite")
    path = _tigramite_csv(tmp_path)
    df = pd.read_csv(path)
    mask = np.random.default_rng(3).random(df.shape) < 0.05
    mask[:5] = False
    df.mask(mask).to_csv(path, index=False)
    res = _discover_json(
        tmp_path, str(path), "-a", algo, "--max-lag", "1", "--impute", "var_em"
    )
    assert res.exit_code == 0, res.output
    s = json.loads(res.output)
    assert s["impute"] == "var_em"
    edges = {(e["source"], e["target"], e["lag"]) for e in s["edges"]}
    assert {("X0", "X1", 1), ("X1", "X2", 1)} <= edges
    # pairwise_complete does not apply; no --impute on gaps is a clean error.
    res = _discover_json(
        tmp_path, str(path), "-a", algo, "--impute", "pairwise_complete"
    )
    assert res.exit_code == 2
    assert f"--impute pairwise_complete does not apply to --algorithm {algo}" in (
        res.output
    )
    res = _discover_json(tmp_path, str(path), "-a", algo, "--max-lag", "1")
    assert res.exit_code == 1
    assert "Traceback" not in res.output and "--impute var_em" in res.output


def test_discover_tigramite_explicit_parcorr_gpu_is_noted(tmp_path):
    pytest.importorskip("tigramite")
    path = _tigramite_csv(tmp_path)
    args = ["-o", str(tmp_path), "discover", str(path), "-a", "pcmciplus"]
    args += ["--max-lag", "1", "--no-plot"]
    res = CliRunner().invoke(main, args + ["--ci-test", "parcorr-gpu"])
    assert res.exit_code == 0, res.output
    assert "runs causal-ts ParCorrGPU through an adapter" in res.output
    assert '"ci_test": "parcorr-gpu"' in "".join(
        p.read_text() for p in tmp_path.glob("discover_pcmciplus_*/summary.json")
    )
    res = CliRunner().invoke(main, args)  # default: tigramite's ParCorr, no note
    assert res.exit_code == 0, res.output
    assert "through an adapter" not in res.output


def test_generate_help_lists_all_registered_datasets():
    from causalts.synthetic_data.synthetic_datasets import DATASET_FUNCS

    result = CliRunner().invoke(main, ["generate", "--help"])
    assert result.exit_code == 0, result.output
    normalized = " ".join(result.output.split())
    for name in sorted(DATASET_FUNCS):
        assert name in normalized


def test_generate_lorenz96(tmp_path):
    result = CliRunner().invoke(
        main,
        [
            "-o",
            str(tmp_path),
            "-s",
            "0",
            "generate",
            "--dataset",
            "lorenz96",
            "-T",
            "100",
        ],
    )
    assert result.exit_code == 0, result.output
    (out,) = tmp_path.glob("generate_*")
    df = pd.read_csv(out / "data.csv")
    gt = np.load(out / "ground_truth.npy")
    meta = json.loads((out / "meta.json").read_text())
    assert df.shape == (100, 10)
    assert np.isfinite(df.to_numpy(dtype=float)).all()
    assert gt.shape == (10, 10, 2)
    assert meta["dataset"] == "lorenz96"
    assert meta["seed"] == 0
    assert meta["T"] == 100
    assert meta["n_vars"] == 10
    assert meta["max_lag"] == 1
    assert meta["name"] == "Lorenz-96"


def test_generate_ex1_nonstationary_preserves_c_node_metadata(tmp_path):
    result = CliRunner().invoke(
        main,
        [
            "-o",
            str(tmp_path),
            "-s",
            "0",
            "generate",
            "--dataset",
            "ex1_nonstationary",
            "-T",
            "100",
        ],
    )
    assert result.exit_code == 0, result.output
    (out,) = tmp_path.glob("generate_*")
    df = pd.read_csv(out / "data.csv")
    gt = np.load(out / "ground_truth.npy")
    meta = json.loads((out / "meta.json").read_text())
    assert df.shape == (100, 5)
    assert gt.shape == (6, 6, 3)
    assert meta["dataset"] == "ex1_nonstationary"
    assert meta["n_vars"] == 5
    assert meta["max_lag"] == 2
    assert meta["include_C"] is True
    assert meta["var_names"] == ["X0", "X1", "X2", "X3", "X4", "C"]
    assert meta["name"] == "5-Node Nonstationary System"


@pytest.mark.parametrize(
    "name, expected_discrete",
    [
        ("mixed_a", ["R"]),
        ("mixed_b", ["C"]),
        ("mixed_c", ["B1", "B2"]),
    ],
)
def test_generate_mixed_datasets_without_name_or_weights(
    tmp_path, name, expected_discrete
):
    result = CliRunner().invoke(
        main,
        ["-o", str(tmp_path), "-s", "0", "generate", "--dataset", name, "-T", "100"],
    )
    assert result.exit_code == 0, result.output
    (out,) = tmp_path.glob("generate_*")
    df = pd.read_csv(out / "data.csv")
    gt = np.load(out / "ground_truth.npy")
    meta = json.loads((out / "meta.json").read_text())
    assert df.shape[0] == 100
    assert gt.shape[0] == gt.shape[1] == df.shape[1]
    assert meta["dataset"] == name
    assert meta["discrete_cols"] == expected_discrete
    assert meta["var_names"] == list(df.columns)
    assert "name" not in meta


def test_generate_other_registered_datasets(tmp_path):
    for name in ("confounded_trends", "ercot_weather", "scale_free"):
        sub = tmp_path / name
        result = CliRunner().invoke(
            main,
            ["-o", str(sub), "-s", "0", "generate", "--dataset", name, "-T", "100"],
        )
        assert result.exit_code == 0, (name, result.output)
        (out,) = sub.glob("generate_*")
        df = pd.read_csv(out / "data.csv")
        gt = np.load(out / "ground_truth.npy")
        meta = json.loads((out / "meta.json").read_text())
        assert df.shape[0] == 100
        assert gt.shape == (df.shape[1], df.shape[1], meta["max_lag"] + 1)
        assert meta["dataset"] == name
