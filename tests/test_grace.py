# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for GRACE (gated causal discovery) algorithm."""

import numpy as np
import pandas as pd
import torch

from causalts.grace.gated_discovery import (
    GatedCausalDiscovery,
    evaluate_graph,
    prepare_data,
    run_cdnots_gated,
)

DEVICE = "cpu"


def test_prepare_data(small_df):
    ds = prepare_data(small_df, max_lag=2, normalize=True)
    T, K = small_df.shape
    assert len(ds) == T - 2
    x = ds[0][0]
    assert x.shape == (K, 3)


def test_prepare_data_no_normalize(small_df):
    ds = prepare_data(small_df, max_lag=1, normalize=False)
    T, K = small_df.shape
    assert len(ds) == T - 1


def test_evaluate_graph_perfect():
    G = np.zeros((3, 3, 2), dtype=np.int8)
    G[0, 1, 1] = 1
    G[1, 2, 1] = 1
    metrics = evaluate_graph(G, G)
    assert metrics["F1"] == 1.0
    assert metrics["SHD"] == 0
    assert metrics["Precision"] == 1.0


def test_evaluate_graph_with_errors():
    G_true = np.zeros((3, 3, 2), dtype=np.int8)
    G_true[0, 1, 1] = 1
    G_true[1, 2, 1] = 1

    G_est = np.zeros((3, 3, 2), dtype=np.int8)
    G_est[0, 1, 1] = 1  # TP
    G_est[2, 0, 1] = 1  # FP

    metrics = evaluate_graph(G_est, G_true)
    assert metrics["Precision"] == 0.5
    assert metrics["SHD"] > 0


def test_gated_model_forward():
    num_vars, max_lag = 3, 2
    model = GatedCausalDiscovery(num_vars=num_vars, max_lag=max_lag)
    x = torch.randn(4, num_vars, max_lag + 1)
    x_hat, x_hat_l0 = model(x)
    assert x_hat.shape == (4, num_vars)
    assert x_hat_l0.shape == (4, num_vars)
    gates = model.get_gate_values()
    assert gates.shape[0] == num_vars


def test_run_cdnots_gated(small_df):
    res = run_cdnots_gated(
        df=small_df,
        max_lag=1,
        alpha=0.1,
        max_epochs=5,
        patience=3,
        device="cpu",
        verbose=False,
        model_seed=42,
    )
    K = small_df.shape[1]
    assert res.cg_tig.shape == (K, K, 2)
    assert res.gate_values.shape == (K, K, 2)
    assert not np.isnan(res.cg_tig).any()
    assert set(np.unique(res.cg_tig)).issubset({0, 1})
