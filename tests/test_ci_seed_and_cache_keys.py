# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Seeded randomised CI tests and complete disk-cache keys."""

import numpy as np
import pytest
import torch

from causalts.ci_tests import DFCITGPU, GCMIGPU, KCIGPU, LinSigCIGPU, SplitKCIGPU
from causalts.ci_tests.cmiknn_gpu import CMIknnGPU

DEVICE = "cpu"


def _data(T=120, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, 3))
    x[:, 1] += 0.3 * x[:, 0]
    return x


def _tied(T=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 3, size=(T, 3)).astype(float)
    x[:, 1] = (x[:, 1] + x[:, 0]) % 3
    return x


# --- fresh instances with one seed agree ---------------------------
FRESH = {
    "kci-perm-uncond": (
        lambda: KCIGPU(_data(), device=DEVICE, approx="perm", nperms=100, seed=0),
        (0, 1, []),
    ),
    "kci-perm-cond": (
        lambda: KCIGPU(_data(), device=DEVICE, approx="perm", nperms=50, seed=0),
        (0, 1, [2]),
    ),
    "kci-bootstrap": (
        lambda: KCIGPU(_data(), device=DEVICE, approx="bootstrap", nperms=500, seed=0),
        (0, 1, [2]),
    ),
    "linsig-perm-uncond": (
        lambda: LinSigCIGPU(_data(), device=DEVICE, approx="perm", nperms=100, seed=42),
        (0, 1, []),
    ),
    "linsig-perm-cond": (
        lambda: LinSigCIGPU(_data(), device=DEVICE, approx="perm", nperms=50, seed=42),
        (0, 2, [1]),  # independent pair: a dependent one gives p = 0 every time
    ),
    "splitkci-wild-bootstrap-uncond": (
        lambda: SplitKCIGPU(
            _data(), device=DEVICE, pval_method="bootstrap", n_bootstrap=200, seed=3
        ),
        (0, 2, []),
    ),
    "splitkci-wild-bootstrap-cond": (
        lambda: SplitKCIGPU(
            _data(), device=DEVICE, pval_method="bootstrap", n_bootstrap=200, seed=3
        ),
        (0, 2, [1]),
    ),
    "cmiknn-gpu-cond": (
        lambda: CMIknnGPU(_data(), device=DEVICE, sig_samples=40, seed=5),
        (0, 2, [1]),
    ),
    "cmiknn-gpu-uncond": (
        lambda: CMIknnGPU(_data(), device=DEVICE, sig_samples=40, seed=5),
        (0, 2, []),
    ),
    "gcmi-ties": (
        lambda: GCMIGPU(_tied(), device=DEVICE, jitter=1e-3, seed=0),
        (0, 1, [2]),
    ),
}


@pytest.mark.parametrize("name", FRESH)
def test_fresh_instances_with_one_seed_agree(name):
    make, (x, y, z) = FRESH[name]
    got = {make()(x, y, z)[0] for _ in range(4)}
    assert len(got) == 1, got


def test_seeded_pvalue_does_not_depend_on_call_order():
    a = KCIGPU(_data(), device=DEVICE, approx="perm", nperms=100, seed=0)
    b = KCIGPU(_data(), device=DEVICE, approx="perm", nperms=100, seed=0)
    a(0, 2, [])
    assert a(0, 1, [])[0] == b(0, 1, [])[0]


def test_seed_none_stays_unseeded():
    def p():
        ci = KCIGPU(_data(), device=DEVICE, approx="perm", nperms=200, seed=None)
        return ci(0, 2, [])[0]

    assert len({p() for _ in range(6)}) > 1


# --- KCIGPU forwards its parameters ----------------------------------
def test_kci_lamda_and_thresh_change_the_conditional_pvalue():
    base = KCIGPU(_data(), device=DEVICE, approx="hbe")(0, 1, [2])[0]
    assert KCIGPU(_data(), device=DEVICE, approx="hbe", lamda=10)(0, 1, [2])[0] != base
    assert (
        KCIGPU(_data(), device=DEVICE, approx="hbe", thresh=0.5)(0, 1, [2])[0] != base
    )


def test_kci_uind_uses_nperms():
    # p-values from n permutations are multiples of 1/n.
    p = KCIGPU(_data(), device=DEVICE, approx="perm", nperms=7, seed=1)(0, 2, [])[0]
    assert np.isclose(p * 7, round(p * 7))


# --- settings sharing a cache_dir do not reuse each other ------------
SHARED = {
    "splitkci-seed": (
        lambda **k: SplitKCIGPU(_data(), device=DEVICE, seed=42, **k),
        lambda **k: SplitKCIGPU(_data(), device=DEVICE, seed=7, **k),
    ),
    "dfcit-early-stop": (
        lambda **k: DFCITGPU(_data(), device=DEVICE, n_perm=200, **k),
        lambda **k: DFCITGPU(_data(), device=DEVICE, n_perm=200, early_stop=False, **k),
    ),
    "dfcit-seed": (
        lambda **k: DFCITGPU(_data(), device=DEVICE, n_perm=50, seed=1, **k),
        lambda **k: DFCITGPU(_data(), device=DEVICE, n_perm=50, seed=2, **k),
    ),
    "gcmi-jitter": (
        lambda **k: GCMIGPU(_tied(), device=DEVICE, jitter=1e-10, **k),
        lambda **k: GCMIGPU(_tied(), device=DEVICE, jitter=0.5, **k),
    ),
    "kci-lamda": (
        lambda **k: KCIGPU(_data(), device=DEVICE, approx="hbe", **k),
        lambda **k: KCIGPU(_data(), device=DEVICE, approx="hbe", lamda=10, **k),
    ),
    "cmiknn-gpu-sig-samples": (
        lambda **k: CMIknnGPU(_data(), device=DEVICE, sig_samples=50, **k),
        lambda **k: CMIknnGPU(_data(), device=DEVICE, sig_samples=61, **k),
    ),
}


@pytest.mark.parametrize("name", SHARED)
def test_settings_sharing_a_cache_dir_get_their_own_entries(name, tmp_path):
    first, second = SHARED[name]
    # An independent pair for DFCIT: on a dependent one both settings give 0.
    x, y, z = (0, 2, [1]) if name.startswith("dfcit") else (0, 1, [2])
    a = first(cache_dir=str(tmp_path))
    p_first = a(x, y, z)[0]
    a.flush_cache()
    want = second()(x, y, z)[0]
    assert want != p_first  # the settings really differ on this test
    assert second(cache_dir=str(tmp_path))(x, y, z)[0] == want


# --- one cache key, one answer: X/Y swaps and Z order (opt-in) -------------
ORIENT = {
    "kci-hbe-cond": lambda **k: KCIGPU(_data(), device=DEVICE, approx="hbe", **k),
    "kci-perm": lambda **k: KCIGPU(
        _data(), device=DEVICE, approx="perm", nperms=50, **k
    ),
    "linsig-perm": lambda **k: LinSigCIGPU(
        _data(), device=DEVICE, approx="perm", nperms=50, **k
    ),
    "gcmi-ties": lambda **k: GCMIGPU(_tied(), device=DEVICE, jitter=1e-3, **k),
    "cmiknn-gpu": lambda **k: CMIknnGPU(_data(), device=DEVICE, sig_samples=40, **k),
}


@pytest.mark.parametrize("name", ORIENT)
def test_swapped_queries_get_the_answer_of_the_canonical_one(name):
    make = lambda: ORIENT[name](order_independent=True)  # noqa: E731
    want = make()(0, 2, [1])[0]
    assert make()(2, 0, [1])[0] == want
    # Same through the cache: whichever orientation runs first.
    ci = make()
    assert ci(2, 0, [1])[0] == want and ci(0, 2, [1])[0] == want


@pytest.mark.parametrize("name", ORIENT)
def test_order_independent_is_in_the_cache_key(name):
    key = lambda ci: ci.param_hash if hasattr(ci, "param_hash") else None  # noqa
    off, on = ORIENT[name](), ORIENT[name](order_independent=True)
    if key(off) is not None:
        assert key(off) != key(on)
    assert off._oi_tag == "" and on._oi_tag == "-oi"


def test_kci_default_computes_in_the_callers_order():
    """Default (as in earlier releases): no swap; the first orientation is cached."""
    from causalts.ci_tests.kci_gpu import kci_cind

    data = _data()
    ci = KCIGPU(data, device=DEVICE, approx="hbe")
    t = ci.data_torch
    direct = {
        (a, b): kci_cind(t[:, a], t[:, b], t[:, [1]], approx="hbe")[0]
        for a, b in ((0, 2), (2, 0))
    }
    assert direct[(0, 2)] != direct[(2, 0)]  # the statistic is not symmetric
    assert ci(2, 0, [1])[0] == direct[(2, 0)]
    assert ci(0, 2, [1])[0] == direct[(2, 0)]  # cache: first orientation wins
    assert KCIGPU(data, device=DEVICE, approx="hbe")(0, 2, [1])[0] == direct[(0, 2)]


def test_kci_cache_key_uses_the_effective_unconditional_bandwidths():
    default = KCIGPU(_data(), device=DEVICE)
    explicit = KCIGPU(_data(), device=DEVICE, width_x="median", width_y="median")
    assert default.param_hash != explicit.param_hash


@pytest.mark.parametrize(
    "make",
    [
        lambda **k: KCIGPU(_nan_data(), device=DEVICE, **k),
        lambda **k: GCMIGPU(_nan_data(), device=DEVICE, **k),
        lambda **k: CMIknnGPU(_nan_data(), device=DEVICE, sig_samples=20, **k),
        lambda **k: SplitKCIGPU(_nan_data(), device=DEVICE, **k),
        lambda **k: DFCITGPU(_nan_data(), device=DEVICE, n_perm=20, **k),
        lambda **k: LinSigCIGPU(_nan_data(), device=DEVICE, **k),
    ],
)
def test_min_samples_is_part_of_the_cache_key(make, tmp_path):
    a = make(min_samples=30, cache_dir=str(tmp_path))
    assert a(0, 1, [])[0] != 1.0  # 40 complete rows: tested
    a.flush_cache()
    b = make(min_samples=50, cache_dir=str(tmp_path))
    assert b(0, 1, []) == (1.0, 0.0)  # too few rows for this setting


def _nan_data():
    x = _data(T=200)
    x[40:, 1] = np.nan
    return x


def test_cmiknn_gpu_pvalue_does_not_depend_on_call_order():
    kw = dict(device=DEVICE, sig_samples=40, seed=0, order_independent=True)
    a, b = CMIknnGPU(_data(), **kw), CMIknnGPU(_data(), **kw)
    a(0, 1, [])  # consumes draws from the instance generator by default
    assert a(0, 2, [])[0] == b(0, 2, [])[0]


def test_cmiknn_gpu_default_draws_from_the_instance_generator():
    """Default (as in earlier releases): one generator per instance, seeded once."""
    kw = dict(device=DEVICE, sig_samples=40, seed=0)
    a, b = CMIknnGPU(_data(), **kw), CMIknnGPU(_data(), **kw)
    assert [a(0, 1, [])[0], a(0, 2, [])[0]] == [b(0, 1, [])[0], b(0, 2, [])[0]]
    c = CMIknnGPU(_data(), **kw)
    before = c.random_state
    c(0, 1, [])
    assert c.random_state is before  # not reseeded per test


@pytest.mark.parametrize(
    "make, random_approx, analytic_approx",
    [
        (lambda a, s: KCIGPU(_data(), device=DEVICE, approx=a, seed=s), "perm", "hbe"),
        (
            lambda a, s: LinSigCIGPU(_data(), device=DEVICE, approx=a, seed=s),
            "perm",
            "hbe",
        ),
    ],
)
def test_seed_is_in_the_cache_key_only_for_random_nulls(
    make, random_approx, analytic_approx
):
    assert make(analytic_approx, 1).param_hash == make(analytic_approx, 2).param_hash
    assert make(random_approx, 1).param_hash != make(random_approx, 2).param_hash


def test_cli_seed_reaches_the_ci_test():
    """``causal-ts -s N`` must seed the CI test, not only NumPy/Torch globals."""
    from causalts.cli import CI_TEST_CHOICES, _make_ci_test

    placeholder = np.zeros((2, 2))
    for name in CI_TEST_CHOICES:
        try:
            _make_ci_test(name, placeholder, "cpu")
        except TypeError:
            continue  # needs more than a placeholder (cmiknn-mixed-gpu)
        _make_ci_test(name, placeholder, "cpu", seed=7)  # a seed never breaks it
    assert _make_ci_test("gcmi", placeholder, "cpu", seed=7)._base_seed == 7
    assert _make_ci_test("kci", placeholder, "cpu", seed=7)._base_seed == 7
    assert _make_ci_test("gcmi", placeholder, "cpu")._base_seed == 42  # default


def test_cli_warns_when_the_seed_is_not_applied():
    """A test built without ``-s`` says so instead of silently dropping it."""
    import warnings

    from causalts.cli import _make_ci_test

    placeholder = np.zeros((2, 2))
    with pytest.warns(UserWarning, match="-s 7 is not applied"):
        _make_ci_test("parcorr", placeholder, "cpu", seed=7)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _make_ci_test("parcorr", placeholder, "cpu")
        _make_ci_test("gcmi", placeholder, "cpu", seed=7)


def test_numpy_integer_seed_draws_like_the_python_int():
    """``seed=np.int64(3)`` (e.g. from ``np.arange``) gives the int-3 draws."""
    from causalts.ci_tests.kci_gpu import KCIGPU

    data = np.random.default_rng(0).standard_normal((60, 3))
    key = "k"
    a = KCIGPU(data, device="cpu", seed=3)._test_seed(key)
    b = KCIGPU(data, device="cpu", seed=np.int64(3))._test_seed(key)
    assert a == b


_ACCEL = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else None
)


@pytest.mark.skipif(_ACCEL is None, reason="needs a CUDA or MPS device")
def test_seeded_nulls_run_under_an_accelerator_default_device():
    """The seeded CPU generators draw on the CPU even when torch's default
    device is an accelerator (``torch.set_default_device`` / ``torch.device``)."""
    from causalts.ci_tests.gcmi_gpu import GCMIGPU
    from causalts.ci_tests.linsig_gpu import LinSigCIGPU
    from causalts.ci_tests.splitkci_gpu import SplitKCIGPU

    tied = np.random.default_rng(0).integers(0, 3, (80, 3)).astype(float)
    cont = tied + np.random.default_rng(1).standard_normal(tied.shape)
    with torch.device(_ACCEL):
        GCMIGPU(tied, device="cpu", seed=1)(0, 1, [2])
        LinSigCIGPU(cont, device="cpu", approx="perm", seed=1)(0, 1, [2])
        SplitKCIGPU(cont, device="cpu", seed=1)(0, 1, [2])


# --- layout-aware tests key their caches on the lag layout -------------------


def test_new_lag_layout_misses_the_old_cached_result():
    """CDNOTS+ re-embeds with byte-identical leading columns; a result cached
    under the old layout must not be returned under the new one."""
    data = np.random.default_rng(0).standard_normal((200, 4)).astype(np.float32)
    t = LinSigCIGPU(data, device=DEVICE)
    t.set_lag_structure(n_vars=2, lags=[0, 1])
    first = t.param_hash
    t(0, 1, [])
    t.set_lag_structure(n_vars=4, lags=[0])
    assert t.param_hash != first
    assert not any(k[1] == t.param_hash for k in t.pvalue_cache if isinstance(k, tuple))
    t.set_lag_structure(n_vars=2, lags=[0, 1])
    assert t.param_hash == first  # re-setting a layout does not stack suffixes


def test_layout_keyed_tests_are_marked():
    from causalts.ci_tests._residual_score import ResidualScoreCIT

    assert LinSigCIGPU._cache_keyed_on_layout
    assert ResidualScoreCIT._cache_keyed_on_layout
