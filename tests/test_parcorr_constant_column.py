# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""CPU ParCorr on a constant column (#111): p = 1, never NaN."""

import warnings

import numpy as np
import pytest

from causalts.ci_tests.parcorr import ParCorr
from causalts.cli import create_ci_test


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    z = rng.standard_normal(200)
    return np.column_stack(
        [x, np.full(200, 3.0), z, 0.8 * z + rng.standard_normal(200)]
    )


@pytest.mark.parametrize("name", ["parcorr", "parcorr-gpu"])
@pytest.mark.parametrize("cond", [[], [2]])
def test_constant_column_is_independent(data, name, cond):
    # A constant column is independent of everything: p must be 1, never NaN
    # (NaN > alpha is False, so the skeleton search would keep the edge).
    test = create_ci_test(name, data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for x, y in ((0, 1), (1, 3)):
            pval, stat = test(x, y, cond)
            assert pval == 1.0 and stat == 0.0


def test_constant_column_does_not_change_other_tests(data):
    pval, stat = create_ci_test("parcorr", data)(2, 3, [])
    assert np.isfinite(pval) and pval < 1e-6 and 0.4 < stat < 0.9


def test_parcorr_run_test_raw_constant_column():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((200, 1))
    stat, pval = ParCorr().run_test_raw(x, np.ones((200, 1)), z=np.empty((200, 0)))
    assert (stat, pval) == (0.0, 1.0)


@pytest.mark.parametrize("T", [37, 200, 999])
@pytest.mark.parametrize("c", [0.1, 1 / 3, 123.456])
def test_non_representable_constant_conditional(T, c):
    # Centring such a constant leaves ~1e-17, and regressing it on Z leaves
    # round-off that varies (not bit-for-bit constant): still independent.
    rng = np.random.default_rng(T)
    x = rng.standard_normal((T, 1))
    z = rng.standard_normal((T, 2))
    stat, pval = ParCorr().run_test_raw(x, np.full((T, 1), c), z=z)
    assert (stat, pval) == (0.0, 1.0)


def test_small_genuine_residual_keeps_its_dependence():
    # A variable almost explained by Z can still depend on Y through a tiny
    # residual; only a constant input column means independence.
    rng = np.random.default_rng(0)
    z = rng.standard_normal((200, 1))
    y = rng.standard_normal((200, 1))
    stat, pval = ParCorr().run_test_raw(z + 1e-11 * y, y, z=z)
    assert stat > 0.99 and pval < 1e-10
