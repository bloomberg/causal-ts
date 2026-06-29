# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for CMIknnGPU (k-NN CMI) conditional independence test."""

import numpy as np

from causalts.ci_tests import CMIknnGPU

DEVICE = "cpu"


def test_unconditional_independent():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(80).astype(np.float32)
    y = rng.standard_normal(80).astype(np.float32)
    data = np.column_stack([x, y])
    ci = CMIknnGPU(data, device=DEVICE, sig_samples=20)
    p, stat = ci(0, 1)
    assert 0 <= p <= 1


def test_conditional_independent(independent_data):
    data = independent_data(n=80)
    ci = CMIknnGPU(data, device=DEVICE, sig_samples=20)
    p, stat = ci(0, 1, [2])
    assert 0 <= p <= 1


def test_conditional_dependent(dependent_data):
    data = dependent_data(n=80)
    ci = CMIknnGPU(data, device=DEVICE, sig_samples=20)
    p, stat = ci(0, 1, [2])
    assert 0 <= p <= 1


def test_cache(independent_data):
    data = independent_data(n=60)
    ci = CMIknnGPU(data, device=DEVICE, sig_samples=20)
    p1, s1 = ci(0, 1, [2])
    n_after = ci.n_actual_tests
    p2, s2 = ci(0, 1, [2])
    assert p1 == p2 and s1 == s2
    assert ci.n_actual_tests == n_after
