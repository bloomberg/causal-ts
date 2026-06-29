# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for SigKCI GPU (signature kernel) conditional independence test."""

import numpy as np
import torch

from causalts.ci_tests.sigkci_gpu import (
    SigKCIGPU,
    _compute_sig_gram_batch,
    _compute_sig_gram_vectorized,
)

DEVICE = "cpu"


def test_unconditional_independent():
    rng = np.random.default_rng(42)
    n = 60
    x = rng.standard_normal(n).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)
    data = np.column_stack([x, y])
    ci = SigKCIGPU(data, device=DEVICE, approx="gamma", time_delay_dim=2)
    p, stat = ci(0, 1)
    assert isinstance(p, float)
    assert isinstance(stat, float)
    assert 0 <= p <= 1


def test_conditional_independent(independent_data):
    data = independent_data(n=60)
    ci = SigKCIGPU(data, device=DEVICE, approx="gamma", time_delay_dim=2)
    p, stat = ci(0, 1, [2])
    assert 0 <= p <= 1


def test_conditional_dependent(dependent_data):
    data = dependent_data(n=60)
    ci = SigKCIGPU(data, device=DEVICE, approx="gamma", time_delay_dim=2)
    p, stat = ci(0, 1, [2])
    assert 0 <= p <= 1


def test_cache(independent_data):
    data = independent_data(n=50)
    ci = SigKCIGPU(data, device=DEVICE, time_delay_dim=2)
    p1, s1 = ci(0, 1, [2])
    n_after = ci.n_actual_tests
    p2, s2 = ci(0, 1, [2])
    assert p1 == p2 and s1 == s2
    assert ci.n_actual_tests == n_after


def test_pde_solver():
    """Verify loop and vectorized PDE solvers produce matching Gram matrices."""
    torch.manual_seed(42)
    paths = torch.randn(8, 3, 2)
    K_loop = _compute_sig_gram_batch(paths, sigma=1.0, dyadic_order=0)
    K_vec = _compute_sig_gram_vectorized(paths, sigma=1.0, dyadic_order=0)
    diff = (K_loop - K_vec).abs().max().item()
    assert diff < 1e-4
    assert K_loop.diag().min().item() > 0
