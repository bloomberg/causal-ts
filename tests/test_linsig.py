# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for LinSig GPU (linear signature kernel) conditional independence test."""

import numpy as np
import pandas as pd
import pytest
import torch

from causalts.ci_tests.linsig_gpu import (
    LinSigCIGPU,
    _cond_pval,
    _kci_residual_maker,
    _linear_sig_gram_batch,
    _linear_sig_gram_vectorized,
)

DEVICE = "cpu"


def test_unconditional_independent():
    rng = np.random.default_rng(42)
    n = 60
    x = rng.standard_normal(n).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)
    data = np.column_stack([x, y])
    ci = LinSigCIGPU(data, device=DEVICE, approx="gamma", time_delay_dim=2)
    p, stat = ci(0, 1)
    assert isinstance(p, float)
    assert isinstance(stat, float)
    assert 0 <= p <= 1


def test_conditional_independent(independent_data):
    data = independent_data(n=60)
    ci = LinSigCIGPU(data, device=DEVICE, approx="gamma", time_delay_dim=2)
    p, stat = ci(0, 1, [2])
    assert 0 <= p <= 1


def test_conditional_dependent(dependent_data):
    data = dependent_data(n=60)
    ci = LinSigCIGPU(data, device=DEVICE, approx="gamma", time_delay_dim=2)
    p, stat = ci(0, 1, [2])
    assert 0 <= p <= 1


def test_cache(independent_data):
    data = independent_data(n=50)
    ci = LinSigCIGPU(data, device=DEVICE, time_delay_dim=2)
    p1, s1 = ci(0, 1, [2])
    n_after = ci.n_actual_tests
    p2, s2 = ci(0, 1, [2])
    assert p1 == p2 and s1 == s2
    assert ci.n_actual_tests == n_after


def test_selfloop_not_fabricated():
    """A same-variable self-loop test must not be a spurious detection.

    LinSig's time-delay embedding of two lags of the same raw variable produces
    aligned/overlapping paths whose linear-signature kernel saturates (worst at
    gap == time_delay_dim - 1: default td=3 fabricates lag-2 self-loops). With
    lag structure registered, such tests route to the value-based fallback,
    which correctly finds no dependence on i.i.d. noise shifted by `gap`.
    """
    T, n_rep = 400, 20
    for gap in (1, 2):  # gap=2 is the td=3 resonance that previously rejected ~1.0
        n_reject = 0
        for seed in range(n_rep):
            r = np.random.default_rng(seed)
            s = r.standard_normal(T + gap + 4)
            data = np.column_stack([s[gap:][:T], s[:-gap][:T]]).astype(float)
            ci = LinSigCIGPU(data, device=DEVICE, time_delay_dim=3)
            ci.set_lag_structure(n_vars=1, lags=[0, gap])
            p, _ = ci(0, 1, [])
            n_reject += int(p < 0.05)
        assert n_reject / n_rep <= 0.2, f"self-loop FP rate too high at gap={gap}"


def test_selfloop_true_ar_detected():
    """The value-based fallback must still detect a genuine self-loop
    (random-walk persistence), not merely suppress false ones."""
    T, n_rep = 400, 20
    n_reject = 0
    for seed in range(n_rep):
        r = np.random.default_rng(seed)
        x = np.cumsum(r.standard_normal(T + 1))
        data = np.column_stack([x[1:][:T], x[:-1][:T]]).astype(float)
        ci = LinSigCIGPU(data, device=DEVICE, time_delay_dim=3)
        ci.set_lag_structure(n_vars=1, lags=[0, 1])
        p, _ = ci(0, 1, [])
        n_reject += int(p < 0.05)
    assert n_reject / n_rep >= 0.8, "true self-loop detection power too low"


def test_pde_solver():
    """Verify loop and vectorized PDE solvers produce matching Gram matrices."""
    torch.manual_seed(42)
    paths = torch.randn(8, 3, 2)
    K_loop = _linear_sig_gram_batch(paths, sigma=1.0, dyadic_order=0)
    K_vec = _linear_sig_gram_vectorized(paths, sigma=1.0, dyadic_order=0)
    diff = (K_loop - K_vec).abs().max().item()
    assert diff < 1e-4
    assert K_loop.diag().min().item() > 0


def test_cond_pval_degenerate_returns_one_not_nan():
    """Regression guard for the curse-of-dimensionality NaN bug.

    At higher conditioning dimensionality, the projected Kxz/Ky can collapse to
    ~0: every eigenvalue gets trimmed by
    the threshold, or the surviving ones are so tiny that the HBE moment-matching
    formula (nu = 8*K2**3/K3**2) divides ~0 by ~0. Either way `_cond_pval` must
    fail safe with p=1.0 ("cannot reject"), never a silent NaN -- a NaN reads as
    "never reject" downstream (NaN < alpha is always False) with no error raised,
    which looks indistinguishable from perfect calibration in a benchmark table.
    """
    T = 50
    zero = torch.zeros(T, T, dtype=torch.float64)
    stat = torch.tensor(0.0, dtype=torch.float64)

    p_empty, _ = _cond_pval(stat, zero, zero, T, "hbe")
    assert p_empty == 1.0

    # A single, tiny-but-nonzero eigenvalue survives the threshold -- degenerate
    # enough that K3 underflows to 0 while K2 doesn't, still a 0/0 risk in nu.
    tiny = torch.eye(T, dtype=torch.float64) * 1e-20
    p_tiny, _ = _cond_pval(stat, tiny, tiny, T, "hbe")
    assert np.isfinite(p_tiny)
    assert 0.0 <= p_tiny <= 1.0


def _spectrum(R):
    return torch.linalg.eigvalsh((R.double() + R.double().T) / 2)


def test_residual_maker_is_identity_for_a_numerically_zero_kernel():
    """Regression guard: conditioning on a linear trend must be a no-op.

    The linear signature Gram of a linear trend is constant (equal increments),
    so its centered eigenvalues are rounding noise (~1e-12 in float32). The old
    relative-threshold pseudo-inverse inverted that noise into a "projector"
    with Frobenius norm in the thousands, every test conditioned on the trend
    returned p=1, and CDNOTS with its time-index node C gave an empty graph.
    """
    T = 294
    eye = torch.eye(T)
    assert torch.allclose(_kci_residual_maker(torch.zeros(T, T)), eye)
    assert torch.allclose(_kci_residual_maker(torch.full((T, T), 3.7)), eye)
    # Constant up to float32 rounding (as left by float32 kernel arithmetic or
    # centering): still no information about Z.
    rng = np.random.default_rng(0)
    E = torch.tensor(rng.standard_normal((T, T)), dtype=torch.float32)
    jitter = 3.7 + 1e-7 * (E + E.T)
    assert torch.allclose(_kci_residual_maker(jitter), eye)


def test_conditioning_on_a_trend_changes_nothing():
    """Through the class, as CDNOTS calls it: a trend column is invisible to the
    linear signature kernel, so conditioning on it alone must give the
    unconditional p-value (the old projector returned p=1 here). Checked on
    mid-range p-values, in path and flat modes."""
    rng = np.random.default_rng(3)
    T = 200
    z = rng.standard_normal(T)
    x = 0.3 * z + rng.standard_normal(T)
    y = 0.3 * z + 0.05 * x + rng.standard_normal(T)
    trend = np.linspace(0.0, 1.0, T)
    data = np.column_stack([x, y, z, trend])
    for path in (True, False):
        ci = LinSigCIGPU(data, device=DEVICE, time_delay_dim=3 if path else 1)
        if path:
            ci.set_lag_structure(4, [0])
        p_plain, _ = ci(0, 1, [])
        p_z, _ = ci(0, 1, [2])
        if path:
            assert 0.001 < p_plain < 0.9 and 0.001 < p_z < 0.9
            assert ci(0, 1, [3])[0] == p_plain
            p_mixed = ci(0, 1, [2, 3])[0]
            assert 0.001 < p_mixed < 0.9  # a time coordinate on z, not p=1
        else:
            # RBF on levels: a trend is information there and must be kept.
            assert ci(0, 1, [3])[0] != p_plain


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_residual_maker_spectrum_and_symmetry(dtype):
    T = 60
    rng = np.random.default_rng(0)
    Q, _ = torch.linalg.qr(torch.tensor(rng.standard_normal((T, T))))
    grams = {
        "ill_conditioned": Q @ torch.diag(torch.logspace(-10, 2, T).double()) @ Q.T,
        "low_rank": torch.tensor(
            rng.standard_normal((T, 2)) @ rng.standard_normal((2, T))
        ),
        "slightly_indefinite": Q
        @ torch.diag(torch.linspace(-1e-6, 5, T).double())
        @ Q.T,
    }
    for name, G in grams.items():
        R = _kci_residual_maker(G.to(dtype))
        assert R.dtype == dtype, name
        assert torch.allclose(R, R.T, atol=1e-6), name
        ev = _spectrum(R)
        assert ev.min() >= -1e-5 and ev.max() <= 1 + 1e-5, name


def test_residual_maker_is_scale_invariant():
    T = 80
    rng = np.random.default_rng(1)
    A = torch.tensor(rng.standard_normal((T, 3)))
    G = A @ A.T + torch.tensor(rng.standard_normal((T, 1))) ** 2
    R1 = _kci_residual_maker(G)
    for scale in (1e-6, 1e6):
        assert torch.allclose(_kci_residual_maker(G * scale), R1, atol=1e-8)


def test_ridge_is_part_of_the_cache_key():
    data = np.random.default_rng(0).standard_normal((50, 3))
    a = LinSigCIGPU(data, device=DEVICE)
    b = LinSigCIGPU(data, device=DEVICE, ridge=1e-2)
    assert a.param_hash != b.param_hash


def test_cdnots_with_time_index_node_recovers_a_chain():
    """End to end: x(t-1) -> z(t) -> y(t+1), CDNOTS default include_C=True."""
    from causalts import run_cdnots

    rng = np.random.default_rng(0)
    T = 300
    x = rng.standard_normal(T)
    z = np.zeros(T)
    y = np.zeros(T)
    for t in range(1, T):
        z[t] = 0.6 * x[t - 1] + 0.3 * rng.standard_normal()
        y[t] = np.tanh(z[t - 1]) + 0.3 * rng.standard_normal()
    df = pd.DataFrame({"x": x, "z": z, "y": y})
    ci = LinSigCIGPU(np.zeros((2, 2)), device=DEVICE)
    g = run_cdnots(df=df, indep_test=ci, num_lags=2, alpha=0.05).cg_tig
    edges = {(int(i), int(j), int(lag)) for i, j, lag in zip(*np.nonzero(g))}
    assert edges == {(0, 1, 1), (1, 2, 1)}


def test_invalid_ridge_is_rejected():
    data = np.random.default_rng(0).standard_normal((40, 3))
    for bad in (0.0, -1e-3, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="ridge"):
            LinSigCIGPU(data, device=DEVICE, ridge=bad)
        with pytest.raises(ValueError, match="ridge"):
            _kci_residual_maker(torch.eye(5), ridge=bad)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_residual_maker_on_mps_matches_cpu():
    rng = np.random.default_rng(0)
    A = torch.tensor(rng.standard_normal((50, 3)), dtype=torch.float32)
    G = A @ A.T
    R_cpu = _kci_residual_maker(G)
    R_mps = _kci_residual_maker(G.to("mps"))
    assert R_mps.device.type == "mps" and R_mps.dtype == torch.float32
    assert torch.allclose(R_mps.cpu(), R_cpu, atol=1e-6)


def test_trend_fallback_after_new_data():
    rng = np.random.default_rng(5)
    T = 120
    x, y = rng.standard_normal(T), rng.standard_normal(T)
    trend = np.column_stack([x, y, np.linspace(0, 1, T)])
    noise = np.column_stack([x, y, rng.standard_normal(T)])
    ci = LinSigCIGPU(trend, device=DEVICE)
    ci.set_lag_structure(3, [0])
    assert ci(0, 1, [2])[0] == ci(0, 1, [])[0]
    ci.data = noise
    fresh = LinSigCIGPU(noise, device=DEVICE)
    fresh.set_lag_structure(3, [0])
    assert ci(0, 1, [2])[0] == fresh(0, 1, [2])[0]
    assert fresh(0, 1, [2])[0] != fresh(0, 1, [])[0]


def test_trend_fallback_with_missing_values_uses_the_kernels_rows():
    """Affinity is judged on the rows the kernel actually uses, in order."""
    rng = np.random.default_rng(7)
    T = 200
    x = rng.standard_normal(T)
    y = 0.3 * x + rng.standard_normal(T)
    trend = np.linspace(0, 1, T)
    trend[40:] = np.nan  # complete rows 0..39: still a trend -> unconditional there
    ci = LinSigCIGPU(np.column_stack([x, y, trend]), device=DEVICE)
    ci.set_lag_structure(3, [0])
    head = LinSigCIGPU(np.column_stack([x, y, trend])[:40], device=DEVICE)
    head.set_lag_structure(3, [0])
    assert ci(0, 1, [2])[0] == head(0, 1, [])[0]


def test_trend_is_kept_when_missing_x_rows_break_its_spacing():
    """Dropping scattered X rows compresses time: the surviving trend values are
    no longer evenly spaced, so the kernel can see them and Z is conditioned on
    (judging the full, evenly spaced column would wrongly drop it)."""
    rng = np.random.default_rng(1)
    T = 200
    t = np.linspace(0, 1, T)
    x = t + 0.02 * rng.standard_normal(T)
    y = t + 0.02 * rng.standard_normal(T)
    x[rng.choice(T, 60, replace=False)] = np.nan
    ci = LinSigCIGPU(np.column_stack([x, y, t]), device=DEVICE)
    ci.set_lag_structure(3, [0])
    assert ci(0, 1, [2])[0] != ci(0, 1, [])[0]


@pytest.mark.parametrize("sigma", ["median", 10**-0.5, 3.0])
def test_trend_fallback_does_not_depend_on_bandwidth(sigma):
    rng = np.random.default_rng(1)
    T = 500
    x = rng.standard_normal(T)
    y = 0.3 * x + rng.standard_normal(T)
    data = np.column_stack([x, y, np.linspace(-1, 1, T)])
    ci = LinSigCIGPU(data, device=DEVICE, sigma=sigma)
    ci.set_lag_structure(3, [0])
    assert ci(0, 1, [2])[0] == ci(0, 1, [])[0]


def test_affine_in_time():
    from causalts.ci_tests.linsig_gpu import _affine_in_time

    t = np.arange(300, dtype=float)
    assert _affine_in_time(3.0 - 0.2 * t)
    assert _affine_in_time(np.full(300, 4.0))
    assert _affine_in_time(np.linspace(0, 1, 300).astype(np.float32))
    assert not _affine_in_time(t + 1e-3 * t**2 / 300)
    assert not _affine_in_time(np.random.default_rng(0).standard_normal(300))
    assert not _affine_in_time(np.array([1.0, np.nan, 2.0]))


def test_smooth_trend_in_z_is_conditioned_on():
    """A quadratic trend is weak but real structure for the linear signature
    kernel; the noise floor must not mistake it for an uninformative Z."""
    rng = np.random.default_rng(0)
    T = 500
    t = np.linspace(-1.0, 1.0, T)
    z = t + 0.1 * t**2  # top eigenvalue ~1.4e-4, between sqrt(T) and T floors
    x = z + 0.001 * rng.standard_normal(T)
    y = z + 0.001 * rng.standard_normal(T)
    ci = LinSigCIGPU(np.column_stack([x, y, z]), device=DEVICE)
    ci.set_lag_structure(3, [0])
    assert ci(0, 1, [])[0] < 0.01
    assert ci(0, 1, [2])[0] > 0.05
    # The p-value alone would also pass with no residualization (p=1), so check
    # the residual-maker the test builds for this Z directly.
    from causalts.ci_tests.kci_gpu import normalize_gpu
    from causalts.ci_tests.linsig_gpu import _compute_linear_gram, _median_bandwidth

    paths = ci._build_row_paths(normalize_gpu(ci.data_torch), [2])
    Gz = _compute_linear_gram(paths, _median_bandwidth(paths))
    assert not torch.allclose(_kci_residual_maker(Gz), torch.eye(Gz.shape[0]))


def test_sigkci_is_a_deprecated_alias_for_linsig():
    from causalts.ci_tests import create_ci_test
    from causalts.ci_tests.sigkci_gpu import SigKCIGPU

    data = np.random.default_rng(0).standard_normal((60, 3))
    with pytest.warns(DeprecationWarning, match="linsig"):
        old = create_ci_test("sigkci", data, device="cpu", lamda=0.001)
    assert isinstance(old, SigKCIGPU) and isinstance(old, LinSigCIGPU)
    new = LinSigCIGPU(data, device="cpu")
    assert old(0, 1, [2]) == new(0, 1, [2])


def test_sigkci_pickled_by_0_27_0_still_works():
    """A 0.27.0 SigKCIGPU pickle has ``lamda`` and no ``ridge``; after the upgrade
    it must unpickle into a working test that matches a fresh LinSigCIGPU and
    does not serve p-values cached by the old conditioning."""
    import pickle
    import warnings

    from causalts.ci_tests.sigkci_gpu import SigKCIGPU

    data = np.random.default_rng(0).standard_normal((80, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = SigKCIGPU(data, device="cpu")
    state = old.__dict__
    del state["ridge"]
    state["lamda"] = 0.001
    state["method"] = "sigkci"
    state["param_hash"] = state["param_hash"].rsplit("-rr", 1)[0]
    stale_key = f"sigkci-{old._get_array_hash(0, 1, [2])}", state["param_hash"]
    state["pvalue_cache"] = {stale_key: (0.123, 9.0)}

    loaded = pickle.loads(pickle.dumps(old))
    fresh = LinSigCIGPU(data, device="cpu")
    assert loaded.ridge == fresh.ridge and not hasattr(loaded, "lamda")
    assert loaded.param_hash == fresh.param_hash
    assert loaded(0, 2, [1]) == fresh(0, 2, [1])
    assert loaded(0, 1, [2])[0] != 0.123

    state["lamda"] = 0.25  # a tuned 0.27.0 object: migrating it must say so
    with pytest.warns(UserWarning, match="lamda=0.25 is ignored"):
        pickle.loads(pickle.dumps(old))


def test_sigkci_functions_and_lamda_still_work():
    import warnings

    from causalts.ci_tests.linsig_gpu import linsig_cind, linsig_uind
    from causalts.ci_tests.sigkci_gpu import SigKCIGPU, sigkci_cind, sigkci_uind

    rng = np.random.default_rng(0)
    x, y, z = (torch.tensor(rng.standard_normal((60, 1))) for _ in range(3))
    with pytest.warns(DeprecationWarning):
        assert sigkci_uind(x, y) == linsig_uind(x, y)
    with pytest.warns(DeprecationWarning):
        assert sigkci_cind(x, y, z) == linsig_cind(x, y, z)
    with pytest.warns(UserWarning, match="lamda=1e-06 is ignored"):
        sigkci_cind(x, y, z, lamda=1e-6)
    data = rng.standard_normal((60, 3))
    with pytest.warns(UserWarning, match="ignored"):
        SigKCIGPU(data, device="cpu", lamda=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        SigKCIGPU(data, device="cpu", lamda=0.001)  # the old default: no warning


def test_sigkci_positional_arguments_keep_their_0_27_0_meaning():
    from causalts.ci_tests.sigkci_gpu import SigKCIGPU

    data = np.random.default_rng(0).standard_normal((60, 3))
    # 0.27.0 order: data, device, sigma, dyadic_order, path_transform,
    # time_delay_dim, approx, nperms, lamda, normalize, seed
    with pytest.warns(UserWarning, match="lamda=1.0 is ignored"):
        old = SigKCIGPU(
            data, "cpu", 2.0, 0, "time_delay", 2, "gamma", 50, 1.0, False, 7
        )
    assert old.ridge == 1e-3 and old.sigma == 2.0 and old.time_delay_dim == 2
    assert old.approx == "gamma" and old.nperms == 50 and old.normalize is False
    new = LinSigCIGPU(
        data,
        device="cpu",
        sigma=2.0,
        time_delay_dim=2,
        approx="gamma",
        nperms=50,
        normalize=False,
        seed=7,
    )
    assert old(0, 1, [2]) == new(0, 1, [2])
