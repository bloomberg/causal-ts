# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""More CLI tests: discover option validation, inspect, evaluate, plot,
deconfound, the dowhy group, install-skill and the small parsing helpers.

All runs use a tiny CSV with a planted chain X0 -> X1 -> X2 at lag 1.
"""

import importlib.util
import json
import os
import sys
import warnings

import click
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from causalts import cli
from causalts.cli import main


def _chain(T=200, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((T, 3))
    for t in range(1, T):
        x[t] = 0.4 * x[t - 1] + rng.standard_normal(3)
        x[t, 1] += 0.9 * x[t - 1, 0]
        x[t, 2] += 0.8 * x[t - 1, 1]
    return pd.DataFrame(x, columns=["X0", "X1", "X2"])


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli_extra")
    df = _chain()
    df.to_csv(d / "data.csv", index=False)
    df.iloc[100:].to_csv(d / "data_late.csv", index=False)
    g = np.zeros((3, 3, 2), dtype=int)
    g[0, 1, 1] = g[1, 2, 1] = 1
    g[[0, 1, 2], [0, 1, 2], 1] = 1
    np.save(d / "graph.npy", g)
    return {"dir": d, "csv": str(d / "data.csv"), "graph": str(d / "graph.npy")}


def _invoke(tmp_path, args, runner=None, **kw):
    runner = runner or CliRunner()
    return runner.invoke(main, ["-o", str(tmp_path), "-s", "0", *args], **kw)


def _stdout_only_runner():
    # click < 8.2 mixes stderr into stdout unless asked not to; 8.2 removed the argument.
    try:
        return CliRunner(mix_stderr=False)
    except TypeError:
        return CliRunner()


def _outdir(tmp_path, prefix):
    (out,) = [p for p in tmp_path.iterdir() if p.name.startswith(prefix)]
    return out


# ---------------------------------------------------------------------------
# discover: option validation (rejected before any computation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args, message",
    [
        (["--algorithm", "reap", "--refine", "gates"], "contradicts --algorithm reap"),
        (["--algorithm", "grace", "--validate"], "--validate is not supported"),
        (["--algorithm", "reap", "--pvalues"], "--pvalues is not supported"),
        (["--algorithm", "cdnots", "--refine", "shrink"], "applies only to"),
        (["--algorithm", "cedar", "--contemporaneous"], "--contemporaneous applies"),
        (["--algorithm", "cedar", "--grow-rounds", "2"], "--grow-rounds applies"),
        (["--algorithm", "cdnots", "--lambda-rule", "xcorr"], "--lambda-rule applies"),
        (["--algorithm", "cdnots", "--no-ar-path"], "--ar-path applies"),
        (["--algorithm", "cedar", "--effect-floor", "0.1"], "--effect-floor applies"),
        (
            ["--algorithm", "grace", "--shrink-effect-floor", "0.1"],
            "--shrink-effect-floor applies only",
        ),
        (
            ["--algorithm", "reap", "--shrink-effect-floor", "1.5"],
            "must be 'adaptive' or a number",
        ),
        (
            ["--algorithm", "reap", "--shrink-effect-floor", "abc"],
            "must be 'adaptive' or a number",
        ),
        (["--algorithm", "reap", "--effect-floor", "0.1"], "is GRACE's floor"),
        (["--algorithm", "cdnots", "--lambda-value", "1.0"], "--lambda-value applies"),
        (["--algorithm", "cedar", "--cv-folds", "3"], "--cv-folds applies"),
        (["--algorithm", "reap", "--impute", "pairwise_complete"], "complete data"),
        (
            ["--algorithm", "cedar", "--ci-test", "cmiknn-mixed-gpu"],
            "not supported with --algorithm cedar",
        ),
        (["--algorithm", "cdnots", "--impute-kwargs", "{bad"], "must be valid JSON"),
        (["--algorithm", "cdnots", "--gate-threshold", "high"], "'auto' or 'li'"),
    ],
)
def test_discover_rejects_inconsistent_options(tmp_path, files, args, message):
    res = _invoke(tmp_path, ["discover", files["csv"], "--max-lag", "1", *args])
    assert res.exit_code == 2, res.output
    assert message in res.output


# ---------------------------------------------------------------------------
# discover: successful runs
# ---------------------------------------------------------------------------


def test_discover_cdnots_validate_annotates_edges_with_persistence(tmp_path, files):
    res = _invoke(
        tmp_path,
        [
            "discover",
            files["csv"],
            "--algorithm",
            "cdnots",
            "--max-lag",
            "1",
            "--no-c",
            "--no-plot",
            "--validate",
            "--n-bootstrap",
            "2",
            "--json",
        ],
        runner=_stdout_only_runner(),
    )
    assert res.exit_code == 0, res.output
    summary = json.loads(res.stdout)
    edges = {(e["source"], e["target"], e["lag"]): e for e in summary["edges"]}
    # The planted chain is found and recurs in both windows.
    for key in [("X0", "X1", 1), ("X1", "X2", 1)]:
        assert edges[key]["persistence"] == 1.0
    assert summary["stability"]["n_bootstrap"] == 2
    out = _outdir(tmp_path, "discover_cdnots")
    assert np.load(out / "persistence.npy").shape == (3, 3, 2)


def test_discover_cedar_validate_and_ground_truth(tmp_path, files):
    res = _invoke(
        tmp_path,
        [
            "discover",
            files["csv"],
            "--algorithm",
            "cedar",
            "--ci-test",
            "parcorr-gpu",
            "--max-lag",
            "1",
            "--no-c",
            "--no-plot",
            "--validate",
            "--n-bootstrap",
            "2",
            "--ground-truth",
            files["graph"],
            "--var-names",
            "a,b,c",
            # run_cedar's library defaults; the CLI defaults (0.05, AR-order
            # estimation) miss X0 -> X1 on this short series.
            "--assume-ar1",
            "--alpha-cond1",
            "0.01",
            "--alpha-cond2",
            "0.01",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "Precision=1.000  Recall=1.000" in res.output
    out = _outdir(tmp_path, "discover_cedar")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["var_names"] == ["a", "b", "c"]
    metrics = json.loads((out / "metrics.json").read_text())
    assert (metrics["TP"], metrics["FP"], metrics["FN"]) == (5, 0, 0)
    assert {("a", "b"), ("b", "c")} <= {
        (e["source"], e["target"]) for e in summary["edges"]
    }
    assert all("persistence" in e for e in summary["edges"])


def test_discover_reap_contemporaneous_writes_marks_and_floor(tmp_path, files):
    res = _invoke(
        tmp_path,
        [
            "discover",
            files["csv"],
            "--algorithm",
            "reap",
            "--ci-test",
            "parcorr",
            "--max-lag",
            "1",
            "--no-c",
            "--no-plot",
            "--contemporaneous",
            "--shrink-effect-floor",
            "0.05",
            "--max-epochs",
            "5",
        ],
    )
    assert res.exit_code == 0, res.output
    # Training options are reported as ignored by REAP.
    assert "--max-epochs has no effect with REAP" in res.output
    out = _outdir(tmp_path, "discover_reap")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["refine"] == "shrink" and summary["contemporaneous"]
    assert summary["shrink_effect_floor"] == 0.05
    files_out = summary["output_files"]
    assert {"edge_marks", "edge_strength", "edge_pvalue"} <= set(files_out)
    g = np.load(out / "estimated_graph.npy")
    assert g[0, 1, 1] and g[1, 2, 1]


# ---------------------------------------------------------------------------
# inspect / evaluate / plot / generate
# ---------------------------------------------------------------------------


def test_inspect_reports_shape_and_renamed_columns(files):
    res = CliRunner().invoke(
        main, ["inspect", files["csv"], "--var-names", "a,b,c", "--max-lag", "2"]
    )
    assert res.exit_code == 0, res.output
    report = json.loads(res.output)
    assert {"facts", "recommendation", "cost_class"} <= set(report)
    assert report["data"]["var_names"] == ["a", "b", "c"]


def test_inspect_rejects_a_file_without_numeric_columns(tmp_path):
    p = tmp_path / "text.csv"
    pd.DataFrame({"a": ["x", "y", "z"], "b": ["u", "v", "w"]}).to_csv(p, index=False)
    res = CliRunner().invoke(main, ["inspect", str(p)])
    assert res.exit_code == 1
    assert "numeric" in res.output.lower()


def test_evaluate_aligns_shapes_and_excludes_self_loops(tmp_path, files):
    pred = np.zeros((3, 3, 3), dtype=int)  # one more lag than the truth
    pred[0, 1, 1] = 1
    pred[0, 0, 1] = 1
    np.save(tmp_path / "pred.npy", pred)
    res = _invoke(
        tmp_path,
        [
            "evaluate",
            files["graph"],
            str(tmp_path / "pred.npy"),
            "--exclude-self-loops",
            "--var-names",
            "a,b,c",
            "--plot-format",
            "svg",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "Shape mismatch" in res.output
    out = _outdir(tmp_path, "evaluate")
    m = json.loads((out / "metrics.json").read_text())
    # Without self-loops: truth has two edges, the prediction finds one.
    assert (m["TP"], m["FP"], m["FN"]) == (1, 0, 1)
    assert (out / "comparison.svg").exists() and (out / "metrics_bar.svg").exists()


@pytest.mark.parametrize("plot_type", ["network", "time-series"])
def test_plot_writes_the_named_figure(tmp_path, files, plot_type):
    vm = np.full((3, 3, 2), 0.5)
    np.save(tmp_path / "vm.npy", vm)
    res = _invoke(
        tmp_path,
        [
            "plot",
            files["graph"],
            "--type",
            plot_type,
            "--val-matrix",
            str(tmp_path / "vm.npy"),
            "--var-names",
            "a,b,c",
            "--figsize",
            "5,4",
            "--save-name",
            "fig",
        ],
    )
    assert res.exit_code == 0, res.output
    assert (_outdir(tmp_path, "plot") / "fig.png").stat().st_size > 0


def test_plot_rejects_a_bad_figsize(tmp_path, files):
    res = _invoke(tmp_path, ["plot", files["graph"], "--figsize", "5"])
    assert res.exit_code != 0
    assert "figsize must be two numbers" in res.output


def test_generate_random_scp_writes_matching_files(tmp_path):
    res = _invoke(
        tmp_path,
        [
            "generate",
            "-n",
            "4",
            "--n-links",
            "3",
            "--max-lag",
            "2",
            "-T",
            "80",
            "--density",
            "dense",
            "--dep-funcs",
            "lin",
        ],
    )
    assert res.exit_code == 0, res.output
    out = _outdir(tmp_path, "generate")
    df = pd.read_csv(out / "data.csv")
    gt = np.load(out / "ground_truth.npy")
    meta = json.loads((out / "meta.json").read_text())
    assert df.shape == (80, 4) and gt.shape == (4, 4, 3)
    assert meta["n_links"] == 3 and meta["density"] == "dense"
    assert meta["funcs_count"] == {"lin": int(gt.sum())}


# ---------------------------------------------------------------------------
# deconfound
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy, extra, key, value",
    [
        ("tetrad", ["--threshold", "0.3"], "threshold", 0.3),
        ("pds", [], "alpha", 1e-10),
        ("adaptive", [], "regime", None),
    ],
)
def test_deconfound_strategies(tmp_path, files, strategy, extra, key, value):
    res = _invoke(
        tmp_path,
        ["deconfound", files["graph"], "--data", files["csv"], "--strategy", strategy]
        + extra,
    )
    assert res.exit_code == 0, res.output
    out = _outdir(tmp_path, f"deconfound_{strategy}")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["edges_before"] == 5
    assert summary["edges_after"] + summary["edges_removed"] == 5
    assert key in summary and (value is None or summary[key] == value)
    g = np.load(out / "deconfounded_graph.npy")
    assert g.shape == (3, 3, 2) and int(g.sum()) == summary["edges_after"]


def test_deconfound_notes_options_for_other_strategies(tmp_path, files):
    res = _invoke(
        tmp_path,
        [
            "deconfound",
            files["graph"],
            "--data",
            files["csv"],
            "--strategy",
            "pds",
            "--threshold",
            "0.3",
            "--var-names",
            "a,b,c",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "--threshold only applies to --strategy tetrad" in res.output
    summary = json.loads(
        (_outdir(tmp_path, "deconfound_pds") / "summary.json").read_text()
    )
    assert {e["source"] for e in summary["edges"]} <= {"a", "b", "c"}


# ---------------------------------------------------------------------------
# dowhy group (linear mechanisms keep these fast)
# ---------------------------------------------------------------------------

needs_dowhy = pytest.mark.skipif(
    importlib.util.find_spec("dowhy") is None, reason="dowhy not installed"
)


@needs_dowhy
def test_dowhy_effect_recovers_the_planted_coefficient(files):
    res = CliRunner().invoke(
        main,
        [
            "dowhy",
            "effect",
            files["graph"],
            "--data",
            files["csv"],
            "--treatment",
            "X0",
            "--outcome",
            "X1",
            "--no-ci",
        ],
    )
    assert res.exit_code == 0, res.output
    ate = float(res.output.split("ATE:")[1].split()[0])
    assert ate == pytest.approx(0.9, abs=0.15)


@needs_dowhy
def test_dowhy_fit_scm_saves_the_model(tmp_path, files):
    res = CliRunner().invoke(
        main,
        [
            "dowhy",
            "fit-scm",
            files["graph"],
            "--data",
            files["csv"],
            "--mechanism",
            "linear",
            "--outdir",
            str(tmp_path / "scm"),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "SCM fitted with linear mechanisms" in res.output
    assert (tmp_path / "scm" / "scm.pkl").exists()
    lagged = pd.read_csv(tmp_path / "scm" / "lagged_data.csv")
    assert {"X0", "X1", "X2", "X0_lag1"} <= set(lagged.columns)


@needs_dowhy
def test_dowhy_root_cause_attributes_to_target_parents(files):
    res = CliRunner().invoke(
        main,
        [
            "dowhy",
            "root-cause",
            files["graph"],
            "--data",
            files["csv"],
            "--target",
            "X2",
            "--mechanism",
            "linear",
            "--n-samples",
            "200",
            "--anomaly-row",
            '{"X2": 8.0}',
        ],
    )
    assert res.exit_code == 0, res.output
    table = res.output.split("Anomaly attribution for 'X2':")[1]
    # Only X2's own noise and its lag-1 parents can be blamed.
    assert "X1_lag1" in table and "X2_lag1" in table
    assert "X0_lag1" not in table


@needs_dowhy
def test_dowhy_strength_and_drift(tmp_path, files):
    base = ["--data", files["csv"], "--target", "X2", "--mechanism", "linear"]
    res = CliRunner().invoke(main, ["dowhy", "strength", files["graph"], *base])
    assert res.exit_code == 0, res.output
    assert "Arrow Strength" in res.output and "Parent Relevance" in res.output
    assert "X1_lag1" in res.output
    res = CliRunner().invoke(
        main,
        [
            "dowhy",
            "drift",
            files["graph"],
            "--data-old",
            files["csv"],
            "--data-new",
            str(files["dir"] / "data_late.csv"),
            "--target",
            "X2",
            "--mechanism",
            "linear",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "Distribution Change: X2" in res.output


@needs_dowhy
def test_dowhy_validate_history(files):
    res = CliRunner().invoke(
        main,
        [
            "dowhy",
            "validate",
            files["graph"],
            "--data",
            files["csv"],
            "--test",
            "history",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "History Sufficiency" in res.output
    assert "max_lag=1 sufficient" in res.output
    assert "Transition-Structure" not in res.output


# ---------------------------------------------------------------------------
# install-skill (home resolves to tmp_path)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    # Patch home resolution itself: on Windows expanduser ignores HOME.
    real = os.path.expanduser
    monkeypatch.setattr(
        os.path,
        "expanduser",
        lambda p: str(tmp_path) + p[1:] if p.startswith("~") else real(p),
    )
    return tmp_path


def _skill_dests(home):
    return [
        home / ".claude" / "skills" / "causal-ts-discovery",
        home / ".agents" / "skills" / "causal-ts-discovery",
    ]


@pytest.mark.skipif(
    sys.platform == "win32", reason="directory symlinks need privileges on Windows"
)
def test_install_skill_dry_run_and_symlink(fake_home):
    runner = CliRunner()
    dests = _skill_dests(fake_home)

    res = runner.invoke(main, ["install-skill", "--dry-run"])
    assert res.exit_code == 0 and res.output.count("[dry-run] would symlink") == 2
    assert not any(d.exists() for d in dests)

    res = runner.invoke(main, ["install-skill"])
    assert res.exit_code == 0, res.output
    assert all(d.is_symlink() and (d / "SKILL.md").exists() for d in dests)
    # Re-running replaces the symlinks idempotently.
    assert runner.invoke(main, ["install-skill"]).exit_code == 0


def test_install_skill_copy_needs_force_over_a_directory(fake_home):
    runner = CliRunner()
    dests = _skill_dests(fake_home)
    for d in dests:
        d.mkdir(parents=True)
    res = runner.invoke(main, ["install-skill", "--copy"])
    assert res.exit_code == 1 and "re-run with --force" in res.output

    res = runner.invoke(main, ["install-skill", "--copy", "--force"])
    assert res.exit_code == 0 and res.output.count("Copied") == 2
    assert all(not d.is_symlink() and (d / "SKILL.md").exists() for d in dests)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_parsing_helpers():
    assert cli._parse_comma_list(None) is None
    assert cli._parse_comma_list("1, 3,,12", int) == [1, 3, 12]
    assert cli._parse_figsize(None) is None
    assert cli._parse_figsize("10,8") == (10.0, 8.0)
    with pytest.raises(click.BadParameter):
        cli._parse_figsize("1,2,3")
    assert cli._parse_gate_threshold(None, None, " LI ") == "li"
    assert cli._parse_gate_threshold(None, None, "0.4") == 0.4


def test_align_graphs_cuts_extra_variables_and_pads_lags():
    gt = np.zeros((2, 2, 2))
    gt[0, 1, 1] = 1
    pred = np.zeros((3, 3, 3))  # an extra C variable and an extra lag
    pred[0, 1, 2] = pred[2, 0, 0] = 1
    a, b = cli._align_graphs(gt, pred)
    assert a.shape == b.shape == (2, 2, 3)
    assert a[0, 1, 1] == 1 and b[0, 1, 2] == 1 and b.sum() == 1
    # A prediction with fewer variables is zero-padded.
    a, b = cli._align_graphs(gt, np.zeros((1, 1, 1)))
    assert b.shape == (2, 2, 2) and b.sum() == 0


def test_make_ci_test_drops_an_unsupported_seed_with_a_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        test = cli._make_ci_test("parcorr", np.zeros((2, 2)), "cpu", seed=3)
    assert test is not None
    assert any("does not take a seed" in str(x.message) for x in w)
    # A test that accepts the seed is built silently.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cli._make_ci_test("parcorr-gpu", np.zeros((2, 2)), "cpu", seed=3)
    assert not any("seed" in str(x.message) for x in w)
