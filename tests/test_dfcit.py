# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for DFCIT GPU (distribution-free) conditional independence test."""

import numpy as np

from causalts.ci_tests import DFCITGPU

DEVICE = "cpu"


def test_unconditional_independent():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100).astype(np.float32)
    y = rng.standard_normal(100).astype(np.float32)
    data = np.column_stack([x, y])
    ci = DFCITGPU(data, device=DEVICE, n_perm=100)
    p, stat = ci(0, 1)
    assert isinstance(p, float)
    assert isinstance(stat, float)
    assert 0 <= p <= 1


def test_unconditional_dependent():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100).astype(np.float32)
    y = (0.8 * x + 0.2 * rng.standard_normal(100)).astype(np.float32)
    data = np.column_stack([x, y])
    ci = DFCITGPU(data, device=DEVICE, n_perm=100)
    p, stat = ci(0, 1)
    assert 0 <= p <= 1


def test_conditional_independent(independent_data):
    data = independent_data(n=100)
    ci = DFCITGPU(data, device=DEVICE, n_perm=100)
    p, stat = ci(0, 1, [2])
    assert 0 <= p <= 1


def test_conditional_dependent(dependent_data):
    data = dependent_data(n=100)
    ci = DFCITGPU(data, device=DEVICE, n_perm=100)
    p, stat = ci(0, 1, [2])
    assert 0 <= p <= 1


def test_cache(independent_data):
    data = independent_data(n=80)
    ci = DFCITGPU(data, device=DEVICE, n_perm=50)
    p1, s1 = ci(0, 1, [2])
    n_after = ci.n_actual_tests
    p2, s2 = ci(0, 1, [2])
    assert p1 == p2 and s1 == s2
    assert ci.n_actual_tests == n_after


def test_multivariate_z():
    rng = np.random.default_rng(42)
    n = 100
    z1 = rng.standard_normal(n).astype(np.float32)
    z2 = rng.standard_normal(n).astype(np.float32)
    x = (z1 + 0.3 * rng.standard_normal(n)).astype(np.float32)
    y = (z2 + 0.3 * rng.standard_normal(n)).astype(np.float32)
    data = np.column_stack([x, y, z1, z2])
    ci = DFCITGPU(data, device=DEVICE, n_perm=50)
    p, stat = ci(0, 1, [2, 3])
    assert 0 <= p <= 1
