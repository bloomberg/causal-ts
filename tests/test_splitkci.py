# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for SplitKCI GPU conditional independence test."""

import numpy as np

from causalts.ci_tests import SplitKCIGPU

DEVICE = "cpu"


def test_unconditional_independent():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100).astype(np.float32)
    y = rng.standard_normal(100).astype(np.float32)
    data = np.column_stack([x, y])
    ci = SplitKCIGPU(data, device=DEVICE)
    p, stat = ci(0, 1)
    assert isinstance(p, float)
    assert isinstance(stat, float)
    assert 0 <= p <= 1
    assert p > 0.01


def test_conditional_independent(independent_data):
    data = independent_data(n=150)
    ci = SplitKCIGPU(data, device=DEVICE, pval_method="gamma")
    p, stat = ci(0, 1, [2])
    assert 0 <= p <= 1


def test_conditional_dependent(dependent_data):
    data = dependent_data(n=150)
    ci = SplitKCIGPU(data, device=DEVICE, pval_method="gamma")
    p, stat = ci(0, 1, [2])
    assert 0 <= p <= 1


def test_cache(independent_data):
    data = independent_data(n=80)
    ci = SplitKCIGPU(data, device=DEVICE)
    p1, s1 = ci(0, 1, [2])
    n_tests_after_first = ci.n_actual_tests
    p2, s2 = ci(0, 1, [2])
    assert p1 == p2 and s1 == s2
    assert ci.n_actual_tests == n_tests_after_first


def test_wild_bootstrap(independent_data):
    data = independent_data(n=100)
    ci = SplitKCIGPU(data, device=DEVICE, pval_method="wild_bootstrap", n_bootstrap=50)
    p, stat = ci(0, 1, [2])
    assert 0 <= p <= 1
