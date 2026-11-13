# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later
"""With ``--json``, stdout carries only the JSON document (progress goes to stderr)."""

import json
import sys

import click
import numpy as np
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
