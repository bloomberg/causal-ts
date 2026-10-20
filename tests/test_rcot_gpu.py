# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for RCOTGPU (Random Fourier Features) conditional independence test."""

import numpy as np
import pytest

from causalts.ci_tests import RCOTGPU

DEVICE = "cpu"


def test_unconditional_independent():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100).astype(np.float32)
    y = rng.standard_normal(100).astype(np.float32)
    data = np.column_stack([x, y])
    ci = RCOTGPU(data=data, device=DEVICE)
    p, stat = ci(0, 1)
    assert 0 <= p <= 1


def test_conditional_independent(independent_data):
    data = independent_data(n=100)
    ci = RCOTGPU(data=data, device=DEVICE)
    p, stat = ci(0, 1, [2])
    assert 0 <= p <= 1


def test_conditional_dependent(dependent_data):
    data = dependent_data(n=100)
    ci = RCOTGPU(data=data, device=DEVICE)
    p, stat = ci(0, 1, [2])
    assert 0 <= p <= 1


def test_cache(independent_data):
    data = independent_data(n=80)
    ci = RCOTGPU(data=data, device=DEVICE)
    p1, s1 = ci(0, 1, [2])
    n_after = ci.n_actual_tests
    p2, s2 = ci(0, 1, [2])
    assert p1 == p2 and s1 == s2
    assert ci.n_actual_tests == n_after


def test_num_f2_default_capped_by_sample_size():
    rng = np.random.default_rng(0)
    ci = RCOTGPU(data=rng.standard_normal((100, 3)), device=DEVICE)
    for T, want in [(50, 7), (100, 10), (400, 20), (624, 24), (625, 25), (800, 25)]:
        assert ci._num_f2_for(T) == want
    ci = RCOTGPU(data=rng.standard_normal((100, 3)), num_f2=25, device=DEVICE)
    assert ci._num_f2_for(50) == 25


def test_num_f2_follows_data_assigned_after_construction():
    # The documented placeholder pattern must not freeze num_f2 at the
    # placeholder's 2 rows.
    rng = np.random.default_rng(1)
    data = rng.standard_normal((400, 3))
    late = RCOTGPU(np.zeros((2, 2)), device=DEVICE)
    late.data = data
    assert late(0, 1, [2]) == RCOTGPU(data=data, device=DEVICE)(0, 1, [2])


@pytest.mark.parametrize("approx", ["hbe", "perm"])
@pytest.mark.parametrize("z", [[], [2]])
def test_default_is_reproducible(dependent_data, approx, z):
    data = dependent_data(n=100)
    p1, s1 = RCOTGPU(data=data, approx=approx, device=DEVICE)(0, 1, z)
    p2, s2 = RCOTGPU(data=data, approx=approx, device=DEVICE)(0, 1, z)
    assert p1 == p2 and s1 == s2


def test_seeded_call_leaves_global_rng_alone(dependent_data):
    import torch

    data = dependent_data(n=100)
    torch.manual_seed(7)
    want = torch.rand(3)
    torch.manual_seed(7)
    RCOTGPU(data=data, device=DEVICE)(0, 1, [2])
    assert torch.equal(torch.rand(3), want)


def test_seed_is_part_of_cache_key(independent_data):
    data = independent_data(n=80)
    seeded = RCOTGPU(data=data, device=DEVICE).param_hash
    assert seeded != RCOTGPU(data=data, seed=None, device=DEVICE).param_hash
    assert seeded != RCOTGPU(data=data, seed=0, device=DEVICE).param_hash


def test_each_test_gets_its_own_reproducible_draw(dependent_data, monkeypatch):
    import causalts.ci_tests.pyRcot_gpu as mod

    seeds = []
    real = mod.rcot

    def spy(*args, **kwargs):
        seeds.append(kwargs["seed"])
        return real(*args, **kwargs)

    monkeypatch.setattr(mod, "rcot", spy)
    data = dependent_data(n=100)
    RCOTGPU(data=data, device=DEVICE)(0, 1, [2])
    RCOTGPU(data=data, device=DEVICE)(0, 2, [1])
    RCOTGPU(data=data, device=DEVICE)(0, 1, [2])
    RCOTGPU(data=data, seed=None, device=DEVICE)(0, 1, [2])
    assert seeds[0] != seeds[1]  # different tests, different draws
    assert seeds[0] == seeds[2]  # same test, same draw on every run
    assert seeds[3] is None  # seed=None stays unseeded


def test_conditioning_order_does_not_matter():
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((200, 2))
    x = np.sin(Z[:, 0]) + 0.5 * rng.standard_normal(200)
    y = Z[:, 1] ** 2 + 0.3 * x + 0.5 * rng.standard_normal(200)
    data = np.column_stack([x, y, Z])
    a = RCOTGPU(data, device="cpu")(0, 1, [2, 3])
    b = RCOTGPU(data, device="cpu")(0, 1, [3, 2])
    assert a == b


def test_bandwidth_sample_size_is_part_of_the_cache_key():
    data = np.random.default_rng(0).standard_normal((60, 3))
    a = RCOTGPU(data, device="cpu", max_sample_4_std_es=50).param_hash
    b = RCOTGPU(data, device="cpu", max_sample_4_std_es=500).param_hash
    assert a != b
