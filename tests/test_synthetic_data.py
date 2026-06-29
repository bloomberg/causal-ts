# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for synthetic data generation."""

import numpy as np
import pandas as pd

from causalts.synthetic_data.synthetic_datasets import (
    SCPGraphGenerator,
    ex1_nl,
    ex2,
    ex3,
    henon_chain,
    load_dataset,
)


def test_load_dataset_ex1():
    r = load_dataset("ex1", seed=42, T=60)
    assert "df" in r
    assert "ground_truth" in r
    assert isinstance(r["df"], pd.DataFrame)
    assert r["df"].shape[0] == 60
    assert not r["df"].isna().any().any()


def test_load_dataset_ex2():
    r = load_dataset("ex2", seed=42, T=60)
    assert r["df"].shape == (60, 6)
    assert r["ground_truth"].ndim == 3


def test_load_dataset_ex3():
    r = load_dataset("ex3", seed=42, T=60)
    assert r["df"].shape == (60, 11)


def test_load_dataset_henon():
    r = load_dataset("henon", seed=42, T=60)
    assert "df" in r
    assert r["df"].shape[0] == 60


def test_ex1_nl_returns_expected_shape():
    r = ex1_nl(seed=42, T=50)
    assert r["df"].shape[1] == 5
    assert r["ground_truth"].shape[0] == 5
    assert r["ground_truth"].shape[1] == 5


def test_ex2_ground_truth_shape():
    r = ex2(seed=42, T=50)
    K = r["df"].shape[1]
    gt = r["ground_truth"]
    assert gt.shape[0] == K
    assert gt.shape[1] == K


def test_henon_chain_params():
    r = henon_chain(seed=0, T=50, Q=0.3, M=3)
    assert r["df"].shape[1] == 3


def test_scp_generator():
    gen = SCPGraphGenerator(n_vars=4, max_lag=2, density="sparse")
    r = gen.sample(seed=42, T=60)
    assert "df" in r
    assert "ground_truth" in r
    assert r["df"].shape == (60, 4)
    gt = r["ground_truth"]
    assert gt.shape == (4, 4, 3)
    assert set(np.unique(gt)).issubset({0, 1})


def test_scp_generator_dense():
    gen = SCPGraphGenerator(n_vars=3, max_lag=1, density="dense")
    r = gen.sample(seed=7, T=50)
    assert r["df"].shape == (50, 3)
    assert not r["df"].isna().any().any()
