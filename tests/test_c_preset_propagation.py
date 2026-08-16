# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests that configuration reaches the algorithm instead of being dropped.

Three silent-drop bugs motivated these: ``--c-preset`` never reached
``run_cdnots``, GRACE had no way to pick a C basis at all, and ``discover_df``
forwarded almost nothing to GRACE. Each test asserts on the kwargs the
underlying ``run_*`` function actually received.
"""

import json
import types

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

import causalts.cdnots.phase3_utils as p3
import causalts.grace.gated_discovery as gd
from causalts import discover_df
from causalts.cli import main


def _df(T=60, d=3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, d))
    for t in range(1, T):
        x[t, 1] += 0.7 * x[t - 1, 0]
        x[t, 2] += 0.6 * x[t - 1, 1]
    return pd.DataFrame(x, columns=[f"X{i}" for i in range(d)])


def _fake_result(d, max_lag):
    return types.SimpleNamespace(
        cg_tig=np.zeros((d, d, max_lag + 1), dtype=np.int8),
        pvalue_matrix=None,
    )


class _Spy:
    """Record the kwargs of every call, return a minimal result object."""

    def __init__(self, d, max_lag, n_c_cols=0):
        self.calls = []
        self._d = d + n_c_cols
        self._max_lag = max_lag

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        return _fake_result(self._d, self._max_lag)


# ---------------------------------------------------------------------------
# GRACE: c_preset reaches the CDNOTS skeleton
# ---------------------------------------------------------------------------
def test_grace_forwards_c_preset_to_skeleton(monkeypatch):
    df = _df()
    spy = _Spy(d=df.shape[1], max_lag=2, n_c_cols=2)
    monkeypatch.setattr(p3, "run_cdnots", spy)
    # Stop after the skeleton — training is irrelevant to this assertion.
    monkeypatch.setattr(
        gd, "prepare_data", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop"))
    )

    with pytest.raises(RuntimeError, match="stop"):
        gd.run_cdnots_gated(df, max_lag=2, c_preset="linear+sin", verbose=False)

    assert len(spy.calls) == 1
    assert spy.calls[0]["c_preset"] == "linear+sin"
    assert spy.calls[0]["include_C"] is True


def test_grace_default_c_preset_is_linear(monkeypatch):
    df = _df()
    spy = _Spy(d=df.shape[1], max_lag=2, n_c_cols=1)
    monkeypatch.setattr(p3, "run_cdnots", spy)
    monkeypatch.setattr(
        gd, "prepare_data", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop"))
    )

    with pytest.raises(RuntimeError, match="stop"):
        gd.run_cdnots_gated(df, max_lag=2, verbose=False)

    assert spy.calls[0]["c_preset"] == "linear"


def test_grace_rejects_unknown_c_preset_before_running_skeleton(monkeypatch):
    df = _df()
    spy = _Spy(d=df.shape[1], max_lag=2)
    monkeypatch.setattr(p3, "run_cdnots", spy)

    with pytest.raises(ValueError, match="Unknown preset"):
        gd.run_cdnots_gated(df, max_lag=2, c_preset="not-a-preset", verbose=False)

    assert spy.calls == [], "preset must be validated before the expensive skeleton"


def test_grace_multicolumn_preset_widens_the_model(monkeypatch):
    """A 2-column preset must add 2 model variables, not 1."""
    df = _df()
    d = df.shape[1]
    monkeypatch.setattr(p3, "run_cdnots", _Spy(d=d, max_lag=2, n_c_cols=2))

    seen = {}

    def _capture(df_model, max_lag, normalize=True):
        seen["cols"] = list(df_model.columns)
        raise RuntimeError("stop")

    monkeypatch.setattr(gd, "prepare_data", _capture)

    with pytest.raises(RuntimeError, match="stop"):
        gd.run_cdnots_gated(
            df,
            max_lag=2,
            c_preset="linear+sin",
            include_C_in_model=True,
            verbose=False,
        )

    assert len(seen["cols"]) == d + 2
    assert seen["cols"][d:] == ["C_lin", "C_sin"]


def test_grace_result_shape_unaffected_by_multicolumn_preset():
    """End-to-end: C columns are stripped, whatever the preset's width."""
    df = _df(T=60, d=3)
    shapes = {}
    for preset in ("linear", "linear+sin"):
        res = gd.run_cdnots_gated(
            df,
            max_lag=2,
            c_preset=preset,
            include_C_in_model=True,
            max_epochs=2,
            patience=1,
            model_seed=0,
            verbose=False,
        )
        shapes[preset] = res.cg_tig.shape
    assert shapes["linear"] == (3, 3, 3)
    assert shapes["linear+sin"] == (3, 3, 3)


def test_grace_include_c_defaults_true_for_backward_compatibility(monkeypatch):
    df = _df()
    spy = _Spy(d=df.shape[1], max_lag=2, n_c_cols=1)
    monkeypatch.setattr(p3, "run_cdnots", spy)
    monkeypatch.setattr(
        gd, "prepare_data", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop"))
    )

    with pytest.raises(RuntimeError, match="stop"):
        gd.run_cdnots_gated(df, max_lag=2, verbose=False)

    assert spy.calls[0]["include_C"] is True


def test_grace_include_c_false_reaches_the_skeleton(monkeypatch):
    df = _df()
    spy = _Spy(d=df.shape[1], max_lag=2)
    monkeypatch.setattr(p3, "run_cdnots", spy)
    monkeypatch.setattr(
        gd, "prepare_data", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop"))
    )

    with pytest.raises(RuntimeError, match="stop"):
        gd.run_cdnots_gated(df, max_lag=2, include_C=False, verbose=False)

    assert spy.calls[0]["include_C"] is False


def test_grace_rejects_model_c_without_skeleton_c():
    with pytest.raises(ValueError, match="requires include_C=True"):
        gd.run_cdnots_gated(
            _df(), max_lag=2, include_C=False, include_C_in_model=True, verbose=False
        )


def test_grace_ss_forwards_include_c_to_ci_skeleton(monkeypatch):
    calls = []

    def _fake_ci_skeleton(df, **kwargs):
        # prepare_data runs before the skeleton here, so stop from inside.
        calls.append(kwargs)
        raise RuntimeError("stop")

    monkeypatch.setattr(gd, "run_ci_skeleton", _fake_ci_skeleton)

    with pytest.raises(RuntimeError, match="stop"):
        gd.run_stability_selection(
            _df(),
            max_lag=2,
            use_ci_skeleton=True,
            include_C=False,
            c_preset="linear+sin",
            verbose=False,
        )

    assert calls[0]["include_C"] is False
    assert calls[0]["c_preset"] == "linear+sin"


def test_cli_no_c_reaches_grace(monkeypatch):
    """`--no-c` used to be silently ignored by GRACE."""
    calls = []

    def _fake_grace(**kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(
            cg_tig=np.zeros((3, 3, 3), dtype=np.int8),
            gate_values=np.zeros((3, 3, 3)),
        )

    # The CLI imports from the package namespace, not the module.
    import causalts.grace as grace_pkg

    monkeypatch.setattr(grace_pkg, "run_cdnots_gated", _fake_grace)
    runner = CliRunner()
    with runner.isolated_filesystem():
        _df().to_csv("data.csv", index=False)
        res = runner.invoke(
            main,
            [
                "--quiet",
                "-o",
                "out",
                "discover",
                "data.csv",
                "--algorithm",
                "grace",
                "--max-lag",
                "2",
                "--no-c",
                "--no-plot",
                "--json",
            ],
        )
    assert res.exit_code == 0, res.output
    assert calls[0]["include_C"] is False
    summary = json.loads(res.output[res.output.index("{") :])
    assert summary["include_C"] is False


def test_run_ci_skeleton_honors_c_preset():
    df = _df(T=80, d=3)
    skel = gd.run_ci_skeleton(
        df, max_lag=2, c_preset="linear+sin", verbose=False, return_pvalues=False
    )
    # C columns are trimmed regardless of how many the preset produced.
    assert skel.shape == (3, 3, 3)


# ---------------------------------------------------------------------------
# discover_df: nothing silently dropped
# ---------------------------------------------------------------------------
def test_discover_df_alpha_none_leaves_algorithm_defaults(monkeypatch):
    import causalts.cedar.discovery as cd

    spy = _Spy(d=3, max_lag=2)
    monkeypatch.setattr(cd, "run_cedar", spy)
    discover_df(_df(), algorithm="cedar", max_lag=2)
    assert "alpha_cond1" not in spy.calls[0]
    assert "alpha_cond2" not in spy.calls[0]


def test_discover_df_alpha_reaches_cedar(monkeypatch):
    import causalts.cedar.discovery as cd

    spy = _Spy(d=3, max_lag=2)
    monkeypatch.setattr(cd, "run_cedar", spy)
    discover_df(_df(), algorithm="cedar", max_lag=2, alpha=0.02)
    assert spy.calls[0]["alpha_cond1"] == 0.02
    assert spy.calls[0]["alpha_cond2"] == 0.02


def test_discover_df_alpha_reaches_cdnots(monkeypatch):
    spy = _Spy(d=3, max_lag=2)
    monkeypatch.setattr(p3, "run_cdnots", spy)
    discover_df(_df(), algorithm="cdnots", max_lag=2, alpha=0.02)
    assert spy.calls[0]["alpha"] == 0.02


def test_discover_df_forwards_c_preset_and_include_c_to_grace(monkeypatch):
    spy = _Spy(d=3, max_lag=2)
    monkeypatch.setattr(gd, "run_cdnots_gated", spy)
    discover_df(
        _df(),
        algorithm="grace",
        max_lag=2,
        include_C=True,
        c_preset="linear+quad",
        alpha=0.02,
    )
    call = spy.calls[0]
    assert call["c_preset"] == "linear+quad"
    assert call["include_C"] is True
    assert call["alpha"] == 0.02
    # The gated model's own C stays opt-in, matching the CLI.
    assert "include_C_in_model" not in call


def test_discover_df_include_c_false_reaches_grace(monkeypatch):
    spy = _Spy(d=3, max_lag=2)
    monkeypatch.setattr(gd, "run_cdnots_gated", spy)
    discover_df(_df(), algorithm="grace", max_lag=2)
    assert spy.calls[0]["include_C"] is False


def test_discover_df_rejects_ci_test_for_grace():
    with pytest.raises(ValueError, match="not supported for 'grace'"):
        discover_df(_df(), algorithm="grace", ci_test="kci")


def test_discover_df_rejects_include_c_for_grace_ss_without_ci_skeleton():
    with pytest.raises(ValueError, match="use_ci_skeleton=True"):
        discover_df(_df(), algorithm="grace-ss", include_C=True)


def test_discover_df_unknown_algorithm():
    with pytest.raises(ValueError, match="unknown algorithm"):
        discover_df(_df(), algorithm="nope")


# ---------------------------------------------------------------------------
# CLI: --c-preset and --impute reach every call, including bootstrap windows
# ---------------------------------------------------------------------------
def test_cli_forwards_c_preset_to_cdnots(monkeypatch):
    spy = _Spy(d=3, max_lag=2)
    monkeypatch.setattr(p3, "run_cdnots", spy)
    runner = CliRunner()
    with runner.isolated_filesystem():
        _df().to_csv("data.csv", index=False)
        res = runner.invoke(
            main,
            [
                "--quiet",
                "-o",
                "out",
                "discover",
                "data.csv",
                "--algorithm",
                "cdnots",
                "--max-lag",
                "2",
                "--include-c",
                "--c-preset",
                "linear+sin",
                "--no-plot",
            ],
        )
    assert res.exit_code == 0, res.output
    assert spy.calls[0]["c_preset"] == "linear+sin"


def test_cli_cdnots_plus_actually_dispatches(monkeypatch):
    """`-a cdnots+` used to be an accepted choice that ran nothing."""
    spy = _Spy(d=3, max_lag=2)
    monkeypatch.setattr(p3, "run_cdnots_plus", spy)
    runner = CliRunner()
    with runner.isolated_filesystem():
        _df().to_csv("data.csv", index=False)
        res = runner.invoke(
            main,
            [
                "--quiet",
                "-o",
                "out",
                "discover",
                "data.csv",
                "--algorithm",
                "cdnots+",
                "--max-lag",
                "2",
                "--no-plot",
                "--json",
            ],
        )
    assert res.exit_code == 0, res.output
    assert len(spy.calls) == 1
    summary = json.loads(res.output[res.output.index("{") :])
    assert summary["output_files"]["graph"] == "estimated_graph.npy"
    # `stable` is CDNOTS-only; cdnots+ must not claim it.
    assert "stable" not in summary


def test_cli_validate_windows_use_the_same_impute(monkeypatch):
    spy = _Spy(d=3, max_lag=2)
    monkeypatch.setattr(p3, "run_cdnots", spy)
    runner = CliRunner()
    with runner.isolated_filesystem():
        df = _df()
        df.iloc[5, 1] = np.nan
        df.to_csv("data.csv", index=False)
        res = runner.invoke(
            main,
            [
                "--quiet",
                "-o",
                "out",
                "discover",
                "data.csv",
                "--algorithm",
                "cdnots",
                "--max-lag",
                "2",
                "--no-plot",
                "--impute",
                "var_em",
                "--validate",
                "--n-bootstrap",
                "3",
            ],
        )
    assert res.exit_code == 0, res.output
    assert len(spy.calls) > 1, "expected the main run plus bootstrap windows"
    for call in spy.calls:
        assert call["impute"] == "var_em"


if __name__ == "__main__":
    pytest.main([__file__])
