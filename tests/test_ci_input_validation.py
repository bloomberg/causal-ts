# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""CI-test input handling: +/-inf is rejected, NaN stays pairwise."""

import numpy as np
import pytest

from causalts.ci_tests import create_ci_test

# Tests that returned p = 1 / NaN (or failed inside linear algebra) on an inf.
# fisherz, parcorr-gpu and kci asserted on an inf at construction, but not when
# the data had NaN (parcorr-gpu, kci) or was assigned later, as the CLI does.
INF_TESTS = [
    "gcm",
    "wgcm",
    "pcm",
    "linsig",
    "splitkci",
    "parcorr",
    "rcot",
    "fisherz",
    "parcorr-gpu",
    "kci",
    "raft",
    "rift",
]


def _make(name, data):
    """Build the test on finite data, then assign ``data`` (the CLI's order)."""
    try:
        ci = create_ci_test(name, np.zeros_like(data), device="cpu")
    except TypeError:  # CPU-only tests take no device
        ci = create_ci_test(name, np.zeros_like(data))
    ci.data = data
    return ci


def _dependent(T=300, seed=0):
    rng = np.random.default_rng(seed)
    x, z = rng.standard_normal((2, T))
    y = x + 0.3 * rng.standard_normal(T)
    return np.column_stack([x, y, z])


@pytest.mark.parametrize("name", INF_TESTS)
@pytest.mark.parametrize("Z", [[], [2]])
@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_infinite_value_in_a_tested_column_raises(name, Z, value):
    data = _dependent()
    data[50, 1] = value
    with pytest.raises(ValueError, match="infinite"):
        _make(name, data)(0, 1, Z)


# causal-learn's FisherZ rejects NaN or inf anywhere in the data by itself.
PAIRWISE_TESTS = [n for n in INF_TESTS if n != "fisherz"]


@pytest.mark.parametrize("name", PAIRWISE_TESTS)
def test_infinite_value_in_an_untested_column_is_ignored(name):
    data = np.column_stack([_dependent(), np.full(300, np.inf)])
    p, _ = _make(name, data)(0, 1, [2])
    assert 0 <= p <= 1


@pytest.mark.parametrize("name", PAIRWISE_TESTS)
def test_missing_values_still_use_complete_rows(name):
    data = _dependent()
    data[:20, 1] = np.nan
    p, _ = _make(name, data)(0, 1, [2])
    assert np.isfinite(p) and 0 <= p <= 1


@pytest.mark.parametrize(
    "Z", [np.array([2, 3]), np.array([], dtype=int), np.array([])], ids=str
)
def test_fisherz_accepts_numpy_conditioning_sets(Z):
    """``if condition_set:`` failed on a numpy array."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((300, 4))
    data[:, 1] += data[:, 0]
    ci = _make("fisherz", data)
    assert ci(0, 1, Z) == _make("fisherz", data)(0, 1, [int(v) for v in Z])


class _AlwaysHit(dict):
    """A p-value cache that holds an entry for every key."""

    def __contains__(self, key):
        return True

    def __getitem__(self, key):
        return [0.5, 0.1]

    def get(self, key, default=None):
        return [0.5, 0.1]


# parcorr (CPU) has no p-value cache.
@pytest.mark.parametrize("name", [n for n in INF_TESTS if n != "parcorr"])
def test_cached_result_does_not_answer_for_an_infinite_input(name):
    """A cache entry (e.g. one persisted by an older build) must not bypass it."""
    data = _dependent()
    data[50, 1] = np.inf
    ci = _make(name, data)
    ci.pvalue_cache = _AlwaysHit()
    with pytest.raises(ValueError, match="infinite"):
        ci(0, 1, [2])


def test_stratified_wrapper_rejects_inf_in_a_discrete_conditioning_column():
    from causalts.ci_tests.stratified_cit import StratifiedCIT

    data = _dependent()
    data[:, 2] = np.arange(300) % 2
    data[7, 2] = np.inf
    ci = StratifiedCIT(data, [2], _make("gcm", data))
    with pytest.raises(ValueError, match="infinite"):
        ci(0, 1, [2])


def test_stratified_wrapper_rejects_inf_in_a_row_without_a_stratum():
    """A row dropped for a missing stratum value must not hide an inf in Y."""
    from causalts.ci_tests.stratified_cit import StratifiedCIT

    data = _dependent()
    data[:, 2] = np.arange(300) % 2
    data[7, 2] = np.nan
    data[7, 1] = np.inf
    ci = StratifiedCIT(data, [2], _make("gcm", data))
    with pytest.raises(ValueError, match="infinite"):
        ci(0, 1, [2])


@pytest.mark.parametrize("enable_caching", [True, False])
def test_parcorr_gpu_batch_test_rejects_inf(enable_caching):
    """batch_test (CDNOTS's default path) bypasses __call__."""
    from causalts.ci_tests import ParCorrGPU

    data = _dependent()
    data[:10, 2] = np.nan  # skips the constructor's check
    data[50, 1] = np.inf
    ci = ParCorrGPU(data, device="cpu", enable_caching=enable_caching)
    with pytest.raises(ValueError, match="infinite"):
        ci.batch_test([(0, 2, []), (0, 1, [2])])


# Rank- and CDF-based tests that accept an infinite value.
INF_TOLERANT_TESTS = ["gcmi", "cmiknn", "dfcit"]


def test_rejects_inf_flag_matches_behaviour():
    for name in INF_TESTS:
        assert _make(name, _dependent()).rejects_inf, name
    for name in INF_TOLERANT_TESTS:
        assert not getattr(_make(name, _dependent()), "rejects_inf", False), name


@pytest.mark.parametrize("name", INF_TOLERANT_TESTS)
@pytest.mark.parametrize("Z", [[2], [3]], ids=["discrete-Z", "continuous-Z"])
def test_stratified_wrapper_leaves_inf_to_a_tolerant_inner_test(name, Z):
    """Wrapping (auto-detected discrete columns) must not change inf handling."""
    from causalts.ci_tests.stratified_cit import StratifiedCIT

    rng = np.random.default_rng(3)
    data = np.column_stack(
        [_dependent(), (np.arange(300) % 2).astype(float), rng.standard_normal(300)]
    )[:, [0, 1, 3, 4]]
    data[7, 2] = np.nan  # a row the strata drop
    data[7, 1] = np.inf
    data[50, 1] = np.inf
    inner = _make(name, data)
    ci = StratifiedCIT(data, [2], inner)
    p, _ = ci(0, 1, Z)
    assert 0 <= p <= 1


def test_stratified_wrapper_uses_the_inner_tests_inf_check(monkeypatch):
    """Wrapped parcorr-gpu keeps its O(k) check from cached flags."""
    from causalts.ci_tests import ParCorrGPU
    from causalts.ci_tests.stratified_cit import StratifiedCIT

    data = _dependent()
    data[:, 2] = np.arange(300) % 2
    inner = ParCorrGPU(data, device="cpu")
    ci = StratifiedCIT(data, [2], inner)
    calls = []
    real = inner._reject_inf
    monkeypatch.setattr(inner, "_reject_inf", lambda *a: calls.append(a) or real(*a))
    ci(0, 1, [2])
    assert calls and calls[0] == (0, 1, [2])
    bad = data.copy()
    bad[7, 2] = np.inf  # in the stratified column the inner test never sees
    ci.data = bad
    with pytest.raises(ValueError, match="infinite"):
        ci(0, 1, [2])


def test_unset_data_keeps_the_data_not_set_error():
    """The inf check must not preempt the clear error for missing data."""
    from causalts.ci_tests.causallearn_tests import CausalLearnFisherZ

    with pytest.raises(ValueError, match="Data not set"):
        CausalLearnFisherZ()(0, 1, [])


@pytest.mark.parametrize("caching", [True, False])
def test_parcorr_gpu_with_data_unset_raises_instead_of_reusing_old_data(caching):
    """Setting .data to None left the old covariance cached, so tests kept answering."""
    from causalts.ci_tests import ParCorrGPU

    ci = ParCorrGPU(np.ones((50, 3)).cumsum(0), device="cpu", enable_caching=caching)
    ci.data = None
    with pytest.raises(ValueError, match="Data not set"):
        ci(0, 1, [2])
    with pytest.raises(ValueError, match="Data not set"):
        ci.batch_test([(0, 1, [2])])


@pytest.mark.parametrize("name", ["parcorr-gpu", "fisherz", "gcm"])
def test_numpy_unsigned_indices_still_work(name):
    """np.uint64 mixed with int must not become a float index in the inf check."""
    ci = _make(name, _dependent())
    p, _ = ci(np.uint64(0), 1, [])
    assert 0 <= p <= 1
    p, _ = ci(0, 1, np.array([2], dtype=np.uint64))
    assert 0 <= p <= 1


@pytest.mark.parametrize("name", ["parcorr", "cmiknn"])
def test_run_test_raw_without_z_matches_an_empty_conditioning_set(name):
    data = _dependent()
    x, y = data[:, :1], data[:, 1:2]
    ci = create_ci_test(name, data)
    empty = ci.run_test_raw(x, y, np.empty((len(x), 0)))
    assert create_ci_test(name, data).run_test_raw(x, y) == empty
    # z=None and an empty z are the same test, so they share one cache entry
    assert ci.run_test_raw(x, y) == empty
    assert len(ci.pvalue_cache) == 1
