# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for GCMIGPU (Gaussian Copula MI) conditional independence test."""

import csv

import numpy as np
import pytest
import torch

from causalts.ci_tests import GCMIGPU

DEVICE = "cpu"


def test_unconditional_independent():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100).astype(np.float32)
    y = rng.standard_normal(100).astype(np.float32)
    data = np.column_stack([x, y])
    ci = GCMIGPU(data, device=DEVICE)
    p, stat = ci(0, 1)
    assert 0 <= p <= 1


def test_unconditional_dependent():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100).astype(np.float32)
    y = (0.9 * x + 0.1 * rng.standard_normal(100)).astype(np.float32)
    data = np.column_stack([x, y])
    ci = GCMIGPU(data, device=DEVICE)
    p, stat = ci(0, 1)
    assert 0 <= p <= 1
    assert p < 0.05


def test_conditional_independent(independent_data):
    data = independent_data(n=100)
    ci = GCMIGPU(data, device=DEVICE)
    p, stat = ci(0, 1, [2])
    assert 0 <= p <= 1


def test_conditional_dependent(dependent_data):
    data = dependent_data(n=100)
    ci = GCMIGPU(data, device=DEVICE)
    p, stat = ci(0, 1, [2])
    assert 0 <= p <= 1


def test_cache(independent_data):
    data = independent_data(n=80)
    ci = GCMIGPU(data, device=DEVICE)
    p1, s1 = ci(0, 1, [2])
    n_after = ci.n_actual_tests
    p2, s2 = ci(0, 1, [2])
    assert p1 == p2 and s1 == s2
    assert ci.n_actual_tests == n_after


def test_multivariate_x_tests_x_against_y():
    """It used to take the partial correlation of columns 0 and 1 of [X, Y, Z],
    so with two X columns it tested X[0] against X[1]."""
    rng = np.random.default_rng(0)
    n = 500
    x0 = rng.standard_normal(n)
    data = np.column_stack(
        [
            x0,
            rng.standard_normal(n),
            rng.standard_normal(n),
            x0 + 0.01 * rng.standard_normal(n),
        ]
    )
    p_multi, _ = GCMIGPU(data, device=DEVICE)([0, 3], 1)
    p_single, _ = GCMIGPU(data, device=DEVICE)(0, 1)
    assert p_single > 0.05 and p_multi > 0.05
    # Y dependent on the second X column only is detected.
    data[:, 1] = data[:, 3] + 0.5 * rng.standard_normal(n)
    assert GCMIGPU(data, device=DEVICE)([0, 3], 1, [2])[0] < 1e-6


def test_multivariate_null_is_calibrated():
    rng = np.random.default_rng(1)
    pvals = []
    for _ in range(300):
        data = rng.standard_normal((200, 5))
        pvals.append(GCMIGPU(data, device=DEVICE)([0, 1], [2, 3], [4])[0])
    pvals = np.array(pvals)
    assert 0.02 <= (pvals < 0.05).mean() <= 0.09


def test_univariate_result_unchanged_by_list_wrapping():
    rng = np.random.default_rng(2)
    data = rng.standard_normal((300, 3))
    data[:, 1] += 0.3 * data[:, 0]
    a = GCMIGPU(data, device=DEVICE, jitter=0)(0, 1, [2])
    b = GCMIGPU(data, device=DEVICE, jitter=0)([0], [1], [2])
    assert np.isclose(a[0], b[0])


def test_multivariate_redundant_columns_keep_the_dependence():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(300)
    data = np.column_stack([x, x, x + 0.3 * rng.standard_normal(300)])
    assert GCMIGPU(data, device=DEVICE)([0, 1], 2)[0] < 1e-6


def test_multivariate_too_few_rows_is_undefined():
    rng = np.random.default_rng(4)
    data = rng.standard_normal((30, 30))
    p, _ = GCMIGPU(data, device=DEVICE)(list(range(15)), list(range(15, 30)))
    assert np.isnan(p)


def test_multivariate_cache_key_differs_from_old_builds():
    rng = np.random.default_rng(5)
    ci = GCMIGPU(rng.standard_normal((100, 4)), device=DEVICE)
    ci([0, 3], 1)
    assert all(ph != "gcmi_gpu" for _, ph in ci.pvalue_cache)


def test_multivariate_conditioning_that_exhausts_the_sample_is_undefined():
    rng = np.random.default_rng(6)
    data = rng.standard_normal((30, 33))
    p, _ = GCMIGPU(data, device=DEVICE)([0, 1], [2, 3], list(range(4, 33)))
    assert np.isnan(p)


def test_multivariate_redundant_sign_flips_match_duplicates():
    """[x, -x] and [x, x] have one direction: same rank, same decision."""
    rng = np.random.default_rng(7)
    x, y, z = rng.standard_normal((3, 200))
    y = y + 0.15 * x
    dup = GCMIGPU(np.column_stack([x, x, y, z, z]), device=DEVICE, jitter=0)
    flip = GCMIGPU(np.column_stack([x, -x, y, z, -z]), device=DEVICE, jitter=0)
    p_dup, _ = dup([0, 1], 2, [3, 4])
    p_flip, _ = flip([0, 1], 2, [3, 4])
    assert np.isclose(p_dup, p_flip, rtol=1e-3)


def test_multivariate_full_rank_boundary_is_defined():
    rng = np.random.default_rng(8)
    p, _ = GCMIGPU(rng.standard_normal((4, 3)), device=DEVICE, jitter=0)([0, 1], 2)
    assert np.isfinite(p)


def test_multivariate_orderings_give_one_answer():
    rng = np.random.default_rng(9)
    data = rng.standard_normal((150, 6))
    data[:, 5] = data[:, 4] + 1e-4 * rng.standard_normal(150)
    want = GCMIGPU(data, device=DEVICE, jitter=0)([0, 1], [2, 3], [4, 5])
    for X, Y, Z in [([1, 0], [3, 2], [5, 4]), ([2, 3], [0, 1], [4, 5])]:
        assert GCMIGPU(data, device=DEVICE, jitter=0)(X, Y, Z) == want


def test_multivariate_single_row_is_undefined():
    p, _ = GCMIGPU(np.zeros((1, 6)), device=DEVICE)([0, 1], 2, [3, 4, 5])
    assert np.isnan(p)


def _redundant_z_data(z2_of, T=500, seed=0):
    """y depends on x; Z = [z, z2] with z2 redundant with z after the copula."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=T)
    x = rng.normal(size=T)
    y = x + rng.normal(size=T)
    return np.c_[x, y, z, z2_of(z, rng)]


@pytest.mark.parametrize(
    "z2_of",
    [
        lambda z, rng: z,
        lambda z, rng: 2 * z,
        lambda z, rng: z + 1e-6 * rng.normal(size=z.size),
    ],
    ids=["z", "2*z", "z+1e-6*noise"],
)
def test_redundant_conditioning_column_keeps_the_dependence(z2_of):
    """It returned p = 1.0: the singular covariance failed to invert (#142)."""
    data = _redundant_z_data(z2_of)
    p, _ = GCMIGPU(data, device=DEVICE)(0, 1, [2, 3])
    assert p < 1e-10


def test_redundant_conditioning_column_does_not_change_the_answer():
    data = _redundant_z_data(lambda z, rng: np.exp(z))
    single = GCMIGPU(data, device=DEVICE, jitter=0)(0, 1, [2])
    both = GCMIGPU(data, device=DEVICE, jitter=0)(0, 1, [2, 3])
    assert both == pytest.approx(single, rel=1e-6)


def test_independent_given_a_duplicated_conditioning_column():
    rng = np.random.default_rng(1)
    z = rng.normal(size=500)
    x = z + rng.normal(size=500)
    y = -z + rng.normal(size=500)
    p, _ = GCMIGPU(np.c_[x, y, z, 3 * z], device=DEVICE)(0, 1, [2, 3])
    assert p > 0.05


def test_variable_against_itself_is_dependent():
    rng = np.random.default_rng(2)
    p, _ = GCMIGPU(rng.normal(size=(200, 2)), device=DEVICE)(0, 0)
    assert p < 1e-10


def test_variable_determined_by_the_conditioning_set_has_nothing_left_to_test():
    rng = np.random.default_rng(3)
    z = rng.normal(size=300)
    y = z + rng.normal(size=300)
    data = np.c_[np.exp(z), y, z]  # X is an increasing function of Z
    assert GCMIGPU(data, device=DEVICE)(0, 1, [2]) == (1.0, 0.0)


def test_non_finite_covariance_raises():
    cov = torch.full((3, 3), float("nan"), dtype=torch.float64)
    with pytest.raises(ValueError, match="not finite"):
        GCMIGPU._gcmi_univariate(cov, T=50, n_z=1)


def test_univariate_does_not_serve_a_result_cached_before_the_fix(tmp_path):
    data = _redundant_z_data(lambda z, rng: z)
    want = GCMIGPU(data, device=DEVICE)(0, 1, [2, 3])
    old = GCMIGPU(data, device=DEVICE, cache_dir=str(tmp_path))
    # The only row in the cache: the p = 1 earlier builds wrote for this query,
    # under the key they used.
    old_key = f"gcmi_gpu-jit{old.jitter}-seed{old._base_seed}-ms{old._min_samples}"
    with open(old.cache_file, "a", newline="") as fout:
        csv.writer(fout).writerow(
            [old._get_array_hash(0, 1, [2, 3]), old_key, 1.0, 0.0]
        )
    got = GCMIGPU(data, device=DEVICE, cache_dir=str(tmp_path))(0, 1, [2, 3])
    assert got == pytest.approx(want)
