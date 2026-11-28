# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""KCI and k-NN CMI tests on a constant tested column: p = 1, never a crash."""

import numpy as np
import pytest
import torch

from causalts.ci_tests import create_ci_test
from causalts.ci_tests.kci_gpu import kci_cind

NAMES = ["kci", "cmiknn", "cmiknn-gpu", "cmiknn-mixed-gpu"]
C = 0.1  # not representable in binary: centring it leaves round-off


def _make(name, data):
    return create_ci_test(name, data, device="cpu")


def _data(seed=0, T=200):
    # 0, 1: dependent pair; 2: constant; 3: noise; 4: another constant.
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(T)
    y = x + 0.3 * rng.standard_normal(T)
    return np.column_stack(
        [x, y, np.full(T, C), rng.standard_normal(T), np.full(T, 1 / 3)]
    )


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("Z", [[], [3]])
@pytest.mark.parametrize("seed", range(3))
def test_constant_tested_column_is_independent(name, Z, seed):
    test = _make(name, _data(seed))
    for x, y in ((0, 2), (2, 0), (2, 4)):
        assert test(x, y, Z) == (1.0, 0.0)


@pytest.mark.parametrize("name", NAMES)
def test_constant_conditioning_column_does_not_short_circuit(name):
    test = _make(name, _data())
    for Z in ([2], [2, 4]):
        p, stat = test(0, 3, Z)
        assert 0 <= p <= 1 and np.isfinite(stat)
    p, _ = test(0, 1, [2])
    assert p < 0.05


@pytest.mark.parametrize("name", NAMES)
def test_dependent_pair_unchanged(name):
    p, _ = _make(name, _data())(0, 1, [3])
    assert p < 0.05


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("Z", [[], [3]])
def test_constant_after_nan_masking(name, Z):
    # Column 2 varies only on rows where column 0 is missing.
    data = _data()
    data[:50, 0] = np.nan
    data[:50, 2] = np.arange(50.0)
    assert _make(name, data)(0, 2, Z) == (1.0, 0.0)


@pytest.mark.parametrize("name", NAMES)
def test_repeated_queries_and_data_replacement(name):
    data = _data()
    test = _make(name, data)
    n = test.n_tests
    assert test(0, 2, [3]) == (1.0, 0.0)
    assert test(0, 2, [3]) == (1.0, 0.0)
    assert test.n_tests > n
    # Column 2 now follows column 0: the cached constant answer must not stick.
    varying = data.copy()
    varying[:, 2] = data[:, 0] + 0.3 * np.random.default_rng(1).standard_normal(200)
    test.data = varying
    p, _ = test(0, 2, [3])
    assert p < 0.05
    test.data = data
    assert test(0, 2, [3]) == (1.0, 0.0)


@pytest.mark.parametrize("name", NAMES)
def test_multivariate_tested_vector(name):
    test = _make(name, _data())
    # A constant component does not make a varying vector constant.
    p, _ = test([1, 2], 0, [3])
    assert p < 0.05
    assert test([2, 4], 0, [3]) == (1.0, 0.0)


@pytest.mark.parametrize("approx", ["hbe", "sw", "bootstrap"])
def test_kci_cind_without_positive_eigenvalue(approx):
    # Defence in depth: a constant y reaching kci_cind directly leaves no
    # eigenvalue above the threshold (max() of an empty tensor raised).
    data = torch.tensor(_data(), dtype=torch.float32)
    p, stat = kci_cind(data[:, 0], data[:, 2], data[:, 3:4], approx=approx, seed=0)
    assert p == 1.0 and np.isfinite(stat)
