# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for ``temporal_bootstrap``."""

import numpy as np
import pandas as pd
import pytest

from causalts import temporal_bootstrap


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


def _fixed_graph(value=1.0):
    def discovery(window):
        return np.full((2, 2, 2), value)

    return discovery


def _df(T=100):
    return pd.DataFrame({"a": np.arange(T, dtype=float), "b": np.arange(T, dtype=float)})


def test_same_seed_draws_the_same_windows():
    r1 = temporal_bootstrap(_df(), _fixed_graph(), n_bootstrap=4, seed=7, show_progress=False)
    r2 = temporal_bootstrap(_df(), _fixed_graph(), n_bootstrap=4, seed=7, show_progress=False)

    assert r1["window_size"] == r2["window_size"]
    assert all(np.array_equal(a, b) for a, b in zip(r1["graphs"], r2["graphs"]))


def test_binary_graphs_give_persistence_in_unit_interval():
    def discovery(window):
        graph = np.zeros((2, 2, 2))
        graph[0, 1, 1] = float(window["a"].iloc[0] % 2 == 0)
        return graph

    r = temporal_bootstrap(_df(), discovery, n_bootstrap=6, seed=0, show_progress=False)

    assert 0.0 <= float(r["persistence"].min()) <= float(r["persistence"].max()) <= 1.0


def test_partly_failing_discovery_is_recorded_in_errors():
    calls = [0]

    def flaky(window):
        calls[0] += 1
        if calls[0] % 2 == 0:
            raise RuntimeError("boom")
        return np.ones((2, 2, 2))

    r = temporal_bootstrap(_df(), flaky, n_bootstrap=6, seed=0, show_progress=False)

    assert r["n_success"] == 3
    assert r["n_errors"] == 3
    assert [i for i, _msg in r["errors"]] == [1, 3, 5]
    assert "boom" in r["errors"][0][1]


def test_too_small_window_says_to_increase_window_frac():
    with pytest.raises(ValueError, match="Increase window_frac"):
        temporal_bootstrap(_df(), _fixed_graph(), window_frac=0.05, show_progress=False)


@pytest.mark.parametrize("window_frac", [0, -0.5, 1, 1.5, float("nan"), float("inf")])
def test_window_frac_outside_the_unit_interval_raises(window_frac):
    with pytest.raises(ValueError, match="window_frac"):
        temporal_bootstrap(_df(), _fixed_graph(), window_frac=window_frac, show_progress=False)


@pytest.mark.parametrize("n_bootstrap", [0, -1, 2.5])
def test_n_bootstrap_must_be_a_positive_integer(n_bootstrap):
    with pytest.raises(ValueError, match="n_bootstrap"):
        temporal_bootstrap(_df(), _fixed_graph(), n_bootstrap=n_bootstrap, show_progress=False)


def test_mismatched_shapes_name_the_run_and_both_shapes():
    shapes = [(2, 2, 2), (2, 2, 3)]
    calls = [0]

    def changing(window):
        shape = shapes[min(calls[0], 1)]
        calls[0] += 1
        return np.zeros(shape)

    with pytest.raises(ValueError) as excinfo:
        temporal_bootstrap(_df(), changing, n_bootstrap=3, show_progress=False)

    message = str(excinfo.value)
    assert "run 1" in message
    assert "(2, 2, 2)" in message and "(2, 2, 3)" in message
