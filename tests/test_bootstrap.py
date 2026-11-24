# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for ``temporal_bootstrap``."""

import numpy as np
import pandas as pd
import pytest

from causalts import temporal_bootstrap


def _df(T=100, cols=("a", "b")):
    return pd.DataFrame(
        np.random.default_rng(0).normal(size=(T, len(cols))), columns=list(cols)
    )


def _const_fn(shape=(2, 2, 2)):
    def fn(window):
        return np.zeros(shape)

    return fn


def test_each_run_keeps_its_own_output_when_discovery_fn_reuses_a_buffer():
    buffer = np.zeros((1, 1, 2))
    count = [0]

    def reuse_output(window):
        count[0] += 1
        buffer[0, 0, 1] = count[0]
        return buffer

    r = temporal_bootstrap(
        pd.DataFrame({"X": range(30)}),
        reuse_output,
        n_bootstrap=3,
        window_frac=0.5,
        show_progress=False,
    )

    assert [float(g[0, 0, 1]) for g in r["graphs"]] == [1.0, 2.0, 3.0]
    assert float(r["persistence"][0, 0, 1]) == 2.0


def test_same_seed_reproduces_persistence():
    def window_stat(window):
        m = float(window["a"].mean())
        return np.array([[[m, 0.0], [0.0, m]]])

    r1 = temporal_bootstrap(
        _df(),
        window_stat,
        n_bootstrap=8,
        window_frac=0.5,
        seed=123,
        show_progress=False,
    )
    r2 = temporal_bootstrap(
        _df(),
        window_stat,
        n_bootstrap=8,
        window_frac=0.5,
        seed=123,
        show_progress=False,
    )

    np.testing.assert_allclose(r1["persistence"], r2["persistence"])
    assert len(r1["graphs"]) == len(r2["graphs"]) == 8
    for g1, g2 in zip(r1["graphs"], r2["graphs"]):
        np.testing.assert_allclose(g1, g2)


def test_persistence_in_unit_interval_for_binary_outputs():
    def binary_fn(window):
        c = np.corrcoef(window.to_numpy().T)
        g = (np.abs(c) > 0.2).astype(float)
        return np.stack([g, g], axis=-1)  # shape (2, 2, 2)

    r = temporal_bootstrap(
        _df(T=120),
        binary_fn,
        n_bootstrap=10,
        window_frac=0.5,
        seed=1,
        show_progress=False,
    )

    p = r["persistence"]
    assert p.shape == (2, 2, 2)
    assert np.all(p >= 0.0)
    assert np.all(p <= 1.0)


def test_partly_failing_discovery_fn_records_errors():
    calls = {"n": 0}

    def flaky(window):
        calls["n"] += 1
        if calls["n"] % 3 == 0:
            raise RuntimeError("boom")
        return np.zeros((2, 2, 2))

    r = temporal_bootstrap(
        _df(), flaky, n_bootstrap=9, window_frac=0.5, seed=3, show_progress=False
    )

    assert r["n_success"] == 6
    assert r["n_errors"] == 3
    assert [i for i, _ in r["errors"]] == [2, 5, 8]
    assert all("boom" in msg for _, msg in r["errors"])
    assert r["persistence"].shape == (2, 2, 2)


def test_all_runs_failing_raises_runtime_error():
    def always_fail(window):
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="All 2 bootstrap runs failed"):
        temporal_bootstrap(
            _df(), always_fail, n_bootstrap=2, window_frac=0.5, show_progress=False
        )


@pytest.mark.parametrize(
    "bad_frac", [0.0, 1.0, 1.5, -0.5, float("nan"), float("inf"), "0.5"]
)
def test_window_frac_outside_open_unit_interval_raises(bad_frac):
    with pytest.raises(ValueError, match="window_frac"):
        temporal_bootstrap(
            _df(), _const_fn(), window_frac=bad_frac, show_progress=False
        )


def test_too_small_window_message_says_to_increase_window_frac():
    with pytest.raises(ValueError, match="Increase window_frac"):
        temporal_bootstrap(_df(), _const_fn(), window_frac=0.05, show_progress=False)


@pytest.mark.parametrize("bad_n", [0, -3, 2.5, "20", True])
def test_invalid_n_bootstrap_raises(bad_n):
    with pytest.raises(ValueError, match="n_bootstrap"):
        temporal_bootstrap(_df(), _const_fn(), n_bootstrap=bad_n, show_progress=False)


def test_mismatched_output_shapes_raise_clear_error():
    calls = {"n": 0}

    def growing(window):
        calls["n"] += 1
        return np.zeros((2, 2, 2)) if calls["n"] == 1 else np.zeros((3, 3, 3))

    with pytest.raises(ValueError) as excinfo:
        temporal_bootstrap(
            _df(), growing, n_bootstrap=3, window_frac=0.5, show_progress=False
        )

    msg = str(excinfo.value)
    assert "run 1" in msg
    assert "(3, 3, 3)" in msg
    assert "(2, 2, 2)" in msg


def _window_fn_failing_on_run(fail_on):
    calls = {"n": 0}

    def fn(window):
        calls["n"] += 1
        if calls["n"] == fail_on:
            raise RuntimeError("boom")
        a = window["a"].to_numpy()
        return np.stack([np.outer(a[:2], a[1:3]), np.outer(a[2:4], a[3:5])], axis=-1)

    return fn


def test_retain_graphs_false_matches_default_persistence():
    kwargs = dict(n_bootstrap=7, window_frac=0.5, seed=5, show_progress=False)
    full = temporal_bootstrap(_df(), _window_fn_failing_on_run(3), **kwargs)
    lean = temporal_bootstrap(
        _df(), _window_fn_failing_on_run(3), retain_graphs=False, **kwargs
    )

    assert lean["graphs"] is None
    assert len(full["graphs"]) == 6
    assert np.allclose(lean["persistence"], full["persistence"])
    assert lean["n_success"] == full["n_success"] == 6
    assert lean["n_errors"] == full["n_errors"] == 1
    assert [i for i, _ in lean["errors"]] == [2]
    assert lean["window_size"] == full["window_size"]


def test_retain_graphs_false_does_not_alias_a_reused_buffer():
    # Runs write 1, 10, 100 into one buffer, so the mean is 37.  A total that
    # aliased the buffer would instead end at 2 * 100, giving 200 / 3.
    buffer = np.zeros((1, 1, 2))
    values = iter([1.0, 10.0, 100.0])

    def reuse_output(window):
        buffer[0, 0, 1] = next(values)
        return buffer

    r = temporal_bootstrap(
        pd.DataFrame({"X": range(30)}),
        reuse_output,
        n_bootstrap=3,
        window_frac=0.5,
        show_progress=False,
        retain_graphs=False,
    )

    assert float(r["persistence"][0, 0, 1]) == 37.0


def test_retain_graphs_false_all_runs_failing_raises_runtime_error():
    def always_fail(window):
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="All 2 bootstrap runs failed"):
        temporal_bootstrap(
            _df(),
            always_fail,
            n_bootstrap=2,
            window_frac=0.5,
            show_progress=False,
            retain_graphs=False,
        )


def test_retain_graphs_false_mismatched_output_shapes_raise_clear_error():
    calls = {"n": 0}

    def growing(window):
        calls["n"] += 1
        return np.zeros((2, 2, 2)) if calls["n"] == 1 else np.zeros((3, 3, 3))

    with pytest.raises(ValueError, match=r"run 1.*\(3, 3, 3\).*\(2, 2, 2\)"):
        temporal_bootstrap(
            _df(),
            growing,
            n_bootstrap=3,
            window_frac=0.5,
            show_progress=False,
            retain_graphs=False,
        )
