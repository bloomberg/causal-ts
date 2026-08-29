# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Third-party algorithm plugins: registry, entry points, and CLI dispatch.

The plugin path exists so someone can add an algorithm without editing the
package. Two halves have to work: registering it (in-process decorator, or an
entry point for the installed ``causal-ts`` command), and actually *running* it
from ``discover`` -- which previously fell through the built-in ``if/elif``
chain and silently wrote no graph.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

import causalts.algorithms as algorithms
from causalts import CausalResult
from causalts.algorithms import list_algorithms, register_algorithm, run_algorithm
from causalts.cli import main

BUILTINS = ("cdnots", "cdnots+", "cedar", "grace", "grace-ss")


class _StubResult(CausalResult):
    def __init__(self, graph, df, var_names):
        self.cg_tig = graph
        self.var_names = list(var_names)
        self._df = df
        self._scm_cache = {}


def _stub_algorithm(df, ci_test=None, max_lag=2, **kwargs):
    """Minimal plugin: one edge at lag 1, so the graph is unmistakably ours."""
    d = df.shape[1]
    graph = np.zeros((d, d, max_lag + 1), dtype=np.int8)
    graph[0, 1, 1] = 1
    return _StubResult(graph, df, list(df.columns))


class _FakeEntryPoint:
    def __init__(self, name, loader):
        self.name = name
        self.value = "stub:stub"
        self._loader = loader

    def load(self):
        return self._loader()


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((120, 4))
    for t in range(1, 120):
        arr[t, 1] += 0.6 * arr[t - 1, 0]
    return pd.DataFrame(arr, columns=[f"X{i}" for i in range(4)])


@pytest.fixture
def csv(tmp_path, data):
    path = tmp_path / "d.csv"
    data.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def registered():
    """Register a plugin via the decorator, then remove it again."""
    register_algorithm("stubalgo")(_stub_algorithm)
    yield "stubalgo"
    algorithms._ALGO_REGISTRY.pop("stubalgo", None)


@pytest.fixture
def entry_points(monkeypatch):
    """Install fake entry points and reset the one-shot load flag."""

    def _install(eps):
        monkeypatch.setattr(
            "importlib.metadata.entry_points", lambda group=None: list(eps)
        )
        monkeypatch.setattr(algorithms, "_entry_points_loaded", False)

    yield _install
    algorithms._entry_points_loaded = False
    for name in ("stubalgo", "ep_algo", "boom"):
        algorithms._ALGO_REGISTRY.pop(name, None)


# ── decorator registration ───────────────────────────────────────────────────
def test_decorator_registers_and_runs(registered, data):
    assert registered in list_algorithms()
    res = run_algorithm(registered, df=data, ci_test=None, max_lag=2)
    assert isinstance(res, CausalResult)
    assert res.cg_tig[0, 1, 1] == 1


def test_unknown_algorithm_raises():
    with pytest.raises(ValueError, match="Unknown algorithm"):
        run_algorithm("does-not-exist", df=None, ci_test=None, max_lag=1)


# ── entry-point discovery ────────────────────────────────────────────────────
def test_entry_point_plugin_is_discovered(entry_points, data):
    """An installed distribution reaches the registry with no import by the caller."""
    entry_points([_FakeEntryPoint("ep_algo", lambda: _stub_algorithm)])
    assert "ep_algo" in list_algorithms()
    assert (
        run_algorithm("ep_algo", df=data, ci_test=None, max_lag=2).cg_tig[0, 1, 1] == 1
    )


def test_broken_plugin_warns_and_is_skipped(entry_points):
    """One bad third-party package must not make causal-ts unusable."""

    def explode():
        raise ImportError("simulated broken plugin")

    entry_points(
        [
            _FakeEntryPoint("boom", explode),
            _FakeEntryPoint("ep_algo", lambda: _stub_algorithm),
        ]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        names = list_algorithms()
    assert any("could not load" in str(w.message) for w in caught)
    assert "boom" not in names
    assert "ep_algo" in names  # the healthy plugin still loaded
    assert all(b in names for b in BUILTINS)


def test_plugin_cannot_shadow_a_builtin(entry_points):
    from causalts.cedar.discovery import run_cedar

    entry_points([_FakeEntryPoint("cedar", lambda: _stub_algorithm)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        list_algorithms()
    assert any("clashes with a built-in" in str(w.message) for w in caught)
    assert algorithms._ALGO_REGISTRY.get("cedar") is run_cedar


def test_entry_points_scanned_once(entry_points, monkeypatch):
    """Scanning is cached: plain lookups must not re-walk installed metadata."""
    calls = []

    def counting(group=None):
        calls.append(group)
        return []

    monkeypatch.setattr("importlib.metadata.entry_points", counting)
    monkeypatch.setattr(algorithms, "_entry_points_loaded", False)
    list_algorithms()
    list_algorithms()
    list_algorithms()
    assert len(calls) == 1


# ── CLI dispatch (the regression this file exists for) ───────────────────────
@pytest.fixture
def cli_with_plugin(entry_points, monkeypatch):
    """A CLI whose ``--algorithm`` choices include an entry-point plugin.

    ``causalts.cli`` snapshots ``ALGORITHM_CHOICES = list_algorithms()`` at import
    time, so an installed plugin is picked up only because entry points are read
    during that first call. Reloading the module here reproduces a fresh
    interpreter -- which is what the ``causal-ts`` command actually is -- rather
    than pretending a late in-process registration would be visible.
    """
    import importlib

    import causalts.cli as cli_module

    entry_points([_FakeEntryPoint("ep_algo", lambda: _stub_algorithm)])
    reloaded = importlib.reload(cli_module)
    yield reloaded.main, "ep_algo"
    algorithms._entry_points_loaded = False
    algorithms._ALGO_REGISTRY.pop("ep_algo", None)
    importlib.reload(cli_module)  # restore the un-patched choice list


def test_cli_offers_installed_plugin(cli_with_plugin):
    cli_main, name = cli_with_plugin
    result = CliRunner().invoke(cli_main, ["discover", "--help"])
    assert name in result.output


def test_cli_runs_plugin_and_writes_a_graph(cli_with_plugin, csv, tmp_path):
    """Regression: discover used to fall through the built-in chain, report
    success, and write summary.json with no graph at all."""
    cli_main, name = cli_with_plugin
    out = tmp_path / "out"
    result = CliRunner().invoke(
        cli_main,
        ["-o", str(out), "-q", "discover", csv, "--algorithm", name, "--max-lag", "2"],
    )
    assert result.exit_code == 0, result.output
    graphs = list(out.glob("*/estimated_graph.npy"))
    assert graphs, "plugin produced no graph (the silent no-op regression)"
    g = np.load(graphs[0])
    assert g[0, 1, 1] == 1
    assert int(g.astype(bool).sum()) == 1


def test_cli_plugin_json_has_edges_and_diagnostics(cli_with_plugin, csv, tmp_path):
    """The generic post-dispatch block must treat a plugin like any built-in."""
    import json

    cli_main, name = cli_with_plugin
    result = CliRunner().invoke(
        cli_main,
        [
            "-o",
            str(tmp_path / "o"),
            "-q",
            "discover",
            csv,
            "--algorithm",
            name,
            "--max-lag",
            "2",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{") :])
    assert payload["n_edges"] == 1
    assert payload["output_files"]["graph"] == "estimated_graph.npy"
    assert payload["edges"] == [
        {"source": "X0", "target": "X1", "lag": 1, "pvalue": None}
    ]
    assert payload["diagnostics"]["n_edges"] == 1


def test_cli_still_rejects_unknown_algorithm(csv, tmp_path):
    result = CliRunner().invoke(
        main,
        ["-o", str(tmp_path), "-q", "discover", csv, "--algorithm", "not-an-algo"],
    )
    assert result.exit_code != 0
    assert "not-an-algo" in result.output


def test_cli_validate_rejects_plugin(cli_with_plugin, csv, tmp_path):
    """--validate must not silently annotate a plugin with CEDAR's persistence.

    Regression for a review finding: the bootstrap re-discovery closure only
    special-cases cdnots/cdnots+ and otherwise falls through to CEDAR -- fine
    for built-ins (nothing else used to reach it), but wrong for a plugin now
    that the else-branch makes it reachable there too.
    """
    cli_main, name = cli_with_plugin
    result = CliRunner().invoke(
        cli_main,
        [
            "-o",
            str(tmp_path),
            "-q",
            "discover",
            csv,
            "--algorithm",
            name,
            "--max-lag",
            "2",
            "--validate",
        ],
    )
    assert result.exit_code != 0
    assert "--validate" in result.output
    assert name in result.output
    assert not list(tmp_path.glob("*/estimated_graph.npy"))


@pytest.mark.parametrize("algorithm", ["cdnots", "cedar"])
def test_cli_builtins_unaffected(algorithm, csv, tmp_path):
    """The new else-branch must not capture any built-in."""
    out = tmp_path / algorithm.replace("+", "p")
    result = CliRunner().invoke(
        main,
        [
            "-o",
            str(out),
            "-q",
            "discover",
            csv,
            "--algorithm",
            algorithm,
            "--max-lag",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert list(out.glob("*/estimated_graph.npy"))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
