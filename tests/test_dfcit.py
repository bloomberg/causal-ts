# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for DFCIT GPU (distribution-free) conditional independence test."""

import warnings

import numpy as np
import pytest
import torch

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


def _independent_ar1(T=400, phi=0.7, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((T, 2))
    x = np.zeros((T, 2))
    for t in range(1, T):
        x[t] = phi * x[t - 1] + e[t]
    return x.astype(np.float32)


def test_auto_block_length_follows_data_assigned_later():
    """The CLI assigns .data after building on a placeholder."""
    data = _independent_ar1()
    late = DFCITGPU(np.zeros((2, 2)), device=DEVICE, n_perm=200)
    late.data = data
    fresh = DFCITGPU(data, device=DEVICE, n_perm=200)
    assert late.sig_blocklength == fresh.sig_blocklength == 20
    assert late.param_hash == fresh.param_hash
    assert late(0, 1) == fresh(0, 1)


def test_explicit_block_length_survives_data_assignment():
    ci = DFCITGPU(np.zeros((2, 2)), device=DEVICE, sig_blocklength=7)
    ci.data = _independent_ar1()
    assert ci.sig_blocklength == 7
    # Auto and an explicit length equal to auto's full-T value must not share a
    # cache key: auto sizes blocks from each test's complete rows.
    explicit = DFCITGPU(_independent_ar1(), device=DEVICE, sig_blocklength=20)
    auto = DFCITGPU(_independent_ar1(), device=DEVICE)
    assert auto.sig_blocklength == 20 and auto.param_hash != explicit.param_hash
    ci = DFCITGPU(np.zeros((2, 2)), device=DEVICE, sig_blocklength=None)
    ci.data = _independent_ar1()
    assert ci.sig_blocklength is None


def test_auto_block_length_uses_complete_rows():
    """With few complete rows, a full-T block length left nothing to shuffle."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((10_000, 2)).astype(np.float32)
    data[40:, 1] = np.nan  # 40 complete rows; sqrt(T) = 100 > 40
    for late in (False, True):
        if late:
            ci = DFCITGPU(np.zeros((2, 2)), device=DEVICE, n_perm=50)
            ci.data = data
        else:
            ci = DFCITGPU(data, device=DEVICE, n_perm=50)
        p, _ = ci(0, 1)
        assert 0 <= p <= 1


def test_explicit_block_length_longer_than_complete_rows_falls_back():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((400, 2)).astype(np.float32)
    data[40:, 1] = np.nan  # 40 complete rows
    ci = DFCITGPU(data, device=DEVICE, n_perm=50, sig_blocklength=50)
    with pytest.warns(UserWarning, match="fewer than 5 blocks"):
        p, _ = ci(0, 1)
    assert 0 <= p <= 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # warned once per instance
        ci.pvalue_cache.clear()  # recompute, so the fallback runs again
        ci(0, 1)
    # Fewer than 5 blocks keeps the longest length that leaves 5 (90 rows,
    # length 30: 3 blocks -> 18); five blocks (length 18) are left alone.
    rows = rng.standard_normal((90, 2)).astype(np.float32)
    with pytest.warns(UserWarning, match="fewer than 5 blocks"):
        DFCITGPU(rows, device=DEVICE, n_perm=50, sig_blocklength=30)(0, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DFCITGPU(rows, device=DEVICE, n_perm=50, sig_blocklength=18)(0, 1)
    # Lengths <= 1 cannot fall back, so their cache key is unchanged.
    assert "-blNone-" in DFCITGPU(data, device=DEVICE, sig_blocklength=None).param_hash
    # Lengths <= 1 mean a plain permutation, as before; 0 must not divide.
    for bl in (0, 1):
        p, _ = DFCITGPU(data, device=DEVICE, n_perm=50, sig_blocklength=bl)(0, 1)
        assert 0 <= p <= 1


def test_auto_block_length_is_sqrt_of_tested_rows(monkeypatch):
    """Auto is sqrt(n) of the rows a test uses, with row shuffling when that
    leaves fewer than 5 blocks (it is shortened to keep 5). A 10-block cap was dropped: it over-rejected
    on short AR series."""
    import causalts.ci_tests.dfcit_gpu as mod

    seen = []
    real = mod.citest_uind
    monkeypatch.setattr(
        mod,
        "citest_uind",
        lambda *a, **kw: seen.append(kw["block_length"]) or real(*a, **kw),
    )
    rng = np.random.default_rng(0)
    for n, want in [(5, None), (12, 2), (16, 3), (20, 4), (25, 5), (40, 6), (400, 20)]:
        data = rng.standard_normal((n, 2)).astype(np.float32)
        ci = DFCITGPU(data, device=DEVICE, n_perm=20)
        ci(0, 1)
        assert seen[-1] == want
        # With no missing values the attribute is the length actually used.
        assert ci.sig_blocklength == want


def test_block_floor_follows_alpha(monkeypatch):
    """The fewest blocks b satisfy b! >= 1/alpha, so a test can reach p < alpha
    (4 blocks give p >= ~1/24, too coarse for alpha 0.01). A test built directly
    for a run_* call never sees discovery's alpha, so the floor is sized for at
    most 0.01."""
    import causalts.ci_tests.dfcit_gpu as mod

    assert [mod._min_blocks(a) for a in (0.05, 0.01, 0.001)] == [4, 5, 7]
    seen = []
    real = mod.citest_uind
    monkeypatch.setattr(
        mod,
        "citest_uind",
        lambda *a, **kw: seen.append(kw["block_length"]) or real(*a, **kw),
    )
    rng = np.random.default_rng(0)
    for n, alpha in [(16, 0.05), (25, 0.05), (49, 0.001)]:
        data = rng.standard_normal((n, 2)).astype(np.float32)
        DFCITGPU(data, device=DEVICE, n_perm=20, alpha=alpha)(0, 1)
    # 16 rows: sqrt gives 4 blocks, too few for 0.01, so 5 blocks of 3 even at
    # the default alpha; 25 rows: 5 blocks of 5; 49 rows at 0.001: 7 of 7.
    assert seen == [3, 5, 7]
    # alpha <= 0 (early stopping off) must not hang the floor computation.
    assert mod._min_blocks(0) == mod._min_blocks(None) == 5
    DFCITGPU(data, device=DEVICE, n_perm=20, alpha=0)(0, 1)


@pytest.mark.slow
def test_auto_block_length_calibrated_on_short_ar_series():
    """Two independent AR(0.9) series, T=36: sqrt(n) blocks reject about 12% at
    alpha 0.05; the dropped 10-block cap (3-row blocks) rejected about 21%."""
    rng = np.random.default_rng(3609)
    reps, rejections = 200, 0
    for r in range(reps):
        e = rng.standard_normal((36, 2))
        x = np.zeros((36, 2))
        for t in range(1, 36):
            x[t] = 0.9 * x[t - 1] + e[t]
        ci = DFCITGPU(x.astype(np.float32), device=DEVICE, n_perm=200, seed=r)
        rejections += ci(0, 1)[0] < 0.05
    assert rejections / reps <= 0.16


def test_multivariate_bandwidth_rule():
    """The automatic bandwidth must grow with the dimension of Z."""
    from causalts.ci_tests.dfcit_gpu import _multivariate_silverman_bandwidth as bw

    # d = 1 is the familiar 1-D rule, 1.06 n^(-1/5).
    assert bw(200, 1) == pytest.approx(1.06 * 200**-0.2, rel=0.01)
    hs = [bw(200, d) for d in (1, 2, 3, 5)]
    assert hs == sorted(hs) and hs[-1] > 1.3 * hs[0]  # 0.37 -> 0.52


def test_leave_one_out_cdf_handles_an_isolated_point():
    from causalts.ci_tests.dfcit_gpu import _estimate_conditional_cdfs

    rng = np.random.default_rng(0)
    z = torch.tensor(rng.standard_normal((50, 2)), dtype=torch.float32)
    z[0] = 1e4  # no neighbour weight left once its own is dropped
    x = torch.tensor(rng.standard_normal(50), dtype=torch.float32)
    u = _estimate_conditional_cdfs(x, z, 0.3, leave_one_out=True)
    assert torch.isfinite(u).all() and ((u >= 0) & (u <= 1)).all()
    assert u[0] == pytest.approx((x <= x[0]).float().mean().item())


@pytest.mark.slow
def test_conditional_calibration_with_three_conditioning_variables():
    """With the 1-D bandwidth rule FPR was 0.76 at |Z| = 3, T = 200."""
    rng = np.random.default_rng(0)
    reps, rejections = 60, 0
    for r in range(reps):
        data = rng.standard_normal((200, 5)).astype(np.float32)
        ci = DFCITGPU(data, device=DEVICE, n_perm=200, seed=r)
        rejections += ci(0, 1, [2, 3, 4])[0] < 0.05
    assert rejections / reps <= 0.2


@pytest.mark.slow
def test_numeric_bandwidth_also_leaves_one_out():
    """Leave-one-out is not tied to the automatic rule. At bandwidth 0.5
    and |Z| = 3 the self-included estimator gave FPR 0.41 (0.085 without)."""
    rng = np.random.default_rng(0)
    reps, rejections = 60, {True: 0, False: 0}
    for r in range(reps):
        data = rng.standard_normal((200, 5)).astype(np.float32)
        for loo in rejections:
            ci = DFCITGPU(
                data,
                device=DEVICE,
                n_perm=200,
                seed=r,
                bandwidth=0.5,
                leave_one_out=loo,
            )
            rejections[loo] += ci(0, 1, [2, 3, 4])[0] < 0.05
    assert rejections[True] / reps <= 0.2 < rejections[False] / reps


def test_warns_once_when_rows_have_no_kernel_neighbour():
    """A bandwidth far below Z's scale leaves rows with no neighbour weight;
    they fall back to the marginal CDF and ignore Z, so the test must say so."""
    from causalts.ci_tests.dfcit_gpu import citest_cind

    rng = np.random.default_rng(0)
    data = rng.standard_normal((200, 6)).astype(np.float32)
    ci = DFCITGPU(data, device=DEVICE, n_perm=50, bandwidth=0.05)
    with pytest.warns(UserWarning, match="ignore Z") as record:
        ci(0, 1, [2, 3, 4])
        ci(0, 1, [3, 4, 5])  # a second sparse test: no second warning
    sparse = [w for w in record if "ignore Z" in str(w.message)]
    assert len(sparse) == 1
    assert "bandwidth=0.05" in str(sparse[0].message)
    assert sparse[0].filename == __file__  # attributed to the caller, not torch
    # The function itself warns too when no callback is given.
    t = torch.tensor(data)
    with pytest.warns(UserWarning, match="ignore Z") as record:
        citest_cind(t[:, 0], t[:, 1], t[:, 2:5], h=0.05, n_perm=50)
    assert record[0].filename == __file__


def test_self_included_estimator_also_warns_for_isolated_rows():
    """With leave_one_out=False an isolated row keeps only its own weight, which
    pins U_i to 1 whatever Z is; the warning must fire on that path too."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((200, 6)).astype(np.float32)
    ci = DFCITGPU(data, device=DEVICE, n_perm=50, bandwidth=0.05, leave_one_out=False)
    with pytest.warns(UserWarning, match="ignore Z"):
        ci(0, 1, [2, 3, 4])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DFCITGPU(data, device=DEVICE, n_perm=50, leave_one_out=False)(0, 1, [2, 3])


def test_sparse_results_stay_off_the_disk_cache(tmp_path):
    """A later run must recompute (and warn about) a result that ignored Z."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((200, 6)).astype(np.float32)
    for _ in range(2):
        ci = DFCITGPU(
            data, device=DEVICE, n_perm=50, bandwidth=0.05, cache_dir=str(tmp_path)
        )
        with pytest.warns(UserWarning, match="ignore Z"):
            ci(0, 1, [2, 3, 4])
        ci.flush_cache()
    # A well-conditioned test on the same cache does reach the disk.
    ci = DFCITGPU(data, device=DEVICE, n_perm=50, cache_dir=str(tmp_path))
    ci(0, 1, [2, 3, 4])
    ci.flush_cache()
    assert ci.cache_file and open(ci.cache_file).read().count("\n") == 2


def test_automatic_bandwidth_does_not_warn():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((200, 7)).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DFCITGPU(data, device=DEVICE, n_perm=50)(0, 1, [2, 3, 4, 5, 6])


def test_cache_key_tags_the_estimator():
    data = np.zeros((10, 3), dtype=np.float32)

    def key(**kw):
        return DFCITGPU(data, device=DEVICE, **kw).param_hash

    assert "-bw_silverman-mv2-loo2-" in key()
    assert "-bw_silverman-mv2-whiten" in key(leave_one_out=False)
    assert "-bw_0.5-loo2-" in key(bandwidth=0.5)
    # Opting out of leave-one-out with a numeric bandwidth is the old estimator.
    assert "-bw_0.5-whiten" in key(bandwidth=0.5, leave_one_out=False)


def test_cli_and_discover_df_pass_alpha_to_dfcit(monkeypatch):
    """Discovery's alpha must reach DFCIT so its block floor allows p < alpha."""
    from causalts.cli import _make_ci_test
    from causalts.inspection import discover_df

    data = np.random.default_rng(0).standard_normal((60, 3)).astype(np.float32)
    assert _make_ci_test("dfcit", data, "cpu", alpha=0.001).alpha == 0.001
    assert _make_ci_test("parcorr", data, "cpu", alpha=0.001) is not None  # ignored
    seen = []
    real = DFCITGPU.__init__

    def spy(self, *a, **kw):
        seen.append(kw.get("alpha"))
        real(self, *a, **kw)

    monkeypatch.setattr(DFCITGPU, "__init__", spy)
    import pandas as pd

    df = pd.DataFrame(data, columns=["a", "b", "c"])
    discover_df(df, algorithm="cdnots", ci_test="dfcit", alpha=0.001, max_lag=1)
    assert seen and seen[0] == 0.001
    # Cedar's unconditional tests are Cond1: alpha_cond1 sizes the floor.
    seen.clear()
    discover_df(df, algorithm="cedar", ci_test="dfcit", max_lag=1, alpha_cond1=0.001)
    assert seen and seen[0] == 0.001
    # Unset thresholds follow the engine defaults (0.01 for CDNOTS+ and Cedar).
    for algorithm in ("cdnots+", "cedar"):
        seen.clear()
        discover_df(df, algorithm=algorithm, ci_test="dfcit", max_lag=1)
        assert seen and seen[0] == 0.01
    # Explicit alpha wins over Cedar's defaults; native overrides are honoured.
    cases = [
        ("cedar", {"alpha": 0.05}, 0.05),
        ("cedar", {"alpha": 0.05, "alpha_cond2": 1e-5, "alpha_mci": 0.002}, 0.05),
        ("cdnots+", {"alpha_mci": 0.002}, 0.002),
        ("grace-ss", {"ci_alpha": 0.002}, 0.002),
    ]
    # Stub the engines: only the constructed test's alpha matters here.
    import causalts.cdnots.phase3_utils as cdnots_mod
    import causalts.cedar.discovery as cedar_mod
    import causalts.grace.gated_discovery as grace_mod

    monkeypatch.setattr(cedar_mod, "run_cedar", lambda **kw: None)
    monkeypatch.setattr(cdnots_mod, "run_cdnots_plus", lambda **kw: None)
    monkeypatch.setattr(
        grace_mod,
        "run_stability_selection",
        lambda df, **kw: kw["ci_test_class"](df.values),
    )
    for algorithm, kw, expected in cases:
        seen.clear()
        discover_df(df, algorithm=algorithm, ci_test="dfcit", max_lag=1, **kw)
        assert seen and seen[0] == expected, (algorithm, kw, seen)
