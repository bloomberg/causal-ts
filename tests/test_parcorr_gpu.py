# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for ParCorrGPU conditional independence test."""

import itertools

import numpy as np
import pytest
import torch

from causalts.ci_tests import ParCorrGPU

DEVICE = "cpu"


def test_unconditional_independent():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100).astype(np.float32)
    y = rng.standard_normal(100).astype(np.float32)
    data = np.column_stack([x, y])
    ci = ParCorrGPU(data, device=DEVICE)
    p, stat = ci(0, 1)
    assert 0 <= p <= 1
    assert p > 0.05


def test_unconditional_dependent():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100).astype(np.float32)
    y = (0.9 * x + 0.1 * rng.standard_normal(100)).astype(np.float32)
    data = np.column_stack([x, y])
    ci = ParCorrGPU(data, device=DEVICE)
    p, stat = ci(0, 1)
    assert 0 <= p <= 1
    assert p < 0.05


def test_conditional_independent(independent_data):
    data = independent_data(n=100)
    ci = ParCorrGPU(data, device=DEVICE)
    p, stat = ci(0, 1, [2])
    assert 0 <= p <= 1


def test_conditional_dependent(dependent_data):
    data = dependent_data(n=100)
    ci = ParCorrGPU(data, device=DEVICE)
    p, stat = ci(0, 1, [2])
    assert 0 <= p <= 1
    assert p < 0.05


def test_cache(independent_data):
    data = independent_data(n=80)
    ci = ParCorrGPU(data, device=DEVICE)
    p1, s1 = ci(0, 1, [2])
    n_after = ci.n_actual_tests
    p2, s2 = ci(0, 1, [2])
    assert p1 == p2 and s1 == s2
    assert ci.n_actual_tests == n_after


def test_scalar_matches_batch():
    """The scalar and batched precision paths return the same numbers.

    Both read the same cached covariance matrix, so a result must not depend on
    which entry point the caller used.
    """
    rng = np.random.default_rng(0)
    data = rng.standard_normal((300, 12))
    tests = [(i, j, [8, 9, 10]) for i in range(4) for j in range(4, 8)]

    scalar = ParCorrGPU(data, device=DEVICE)
    batched = ParCorrGPU(data, device=DEVICE).batch_test(list(tests))

    for (X, Y, Z), (p_b, s_b) in zip(tests, batched):
        p_s, s_s = scalar(X, Y, Z)
        assert np.isclose(p_s, p_b, rtol=0, atol=1e-12)
        assert np.isclose(s_s, s_b, rtol=0, atol=1e-12)


def test_scalar_agrees_with_ols_method():
    """The precision default and ``method="ols"`` estimate the same quantity."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal((300, 10))
    precision = ParCorrGPU(data, device=DEVICE)
    ols = ParCorrGPU(data, device=DEVICE, method="ols")

    for X, Y, Z in [(0, 1, []), (0, 1, [2]), (3, 4, [0, 1, 2, 5])]:
        p_p, s_p = precision(X, Y, Z)
        p_o, s_o = ols(X, Y, Z)
        # OLS residualizes in float32, hence the loose tolerance.
        assert np.isclose(s_p, s_o, rtol=0, atol=1e-5)
        assert np.isclose(p_p, p_o, rtol=0, atol=1e-4)


def test_degenerate_subcovariance_falls_back():
    """Singular sub-covariances must return a result, not raise."""
    rng = np.random.default_rng(2)
    base = rng.standard_normal((200, 4))

    # Constant column, duplicated column, and Z overlapping X.
    constant = np.column_stack([base, np.ones(200)])
    duplicate = np.column_stack([base, base[:, 0]])

    for data, (X, Y, Z) in [
        (constant, (0, 1, [4])),
        (constant, (0, 4, [1])),
        (duplicate, (1, 2, [0, 4])),
        (base, (0, 1, [0, 2])),
    ]:
        p, stat = ParCorrGPU(data, device=DEVICE)(X, Y, Z)
        assert 0 <= p <= 1
        assert np.isfinite(stat)


def _near_singular_case(eps=1e-8, n=300, seed=0):
    """Data whose (X, Y, Z) sub-covariance is rank-deficient to working precision."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n)
    z = rng.standard_normal(n)
    x = w + 0.3 * rng.standard_normal(n)
    y = w + 0.3 * rng.standard_normal(n)
    return np.column_stack([x, y, z, z + eps * w])


def _reciprocal_condition_case(seed=0, k=14, T=40, a=0.05):
    """Data whose (X, Y, Z) sub-covariance has reciprocal condition ~4e-18.

    Every column after the first is ``sqrt(1 - a**2)`` times a unit vector over
    the previous columns plus an ``a``-scaled noise term, so conditioning
    collapses as ``k`` grows. The construction from #73's permutation test,
    reused as issue #74's reproduction.
    """
    rng = np.random.default_rng(seed)
    L = np.zeros((k, k))
    L[0, 0] = 1.0
    for i in range(1, k):
        v = rng.standard_normal(i)
        v /= np.linalg.norm(v)
        L[i, :i] = np.sqrt(1 - a * a) * v
        L[i, i] = a
    return rng.standard_normal((T, k)) @ L.T


def test_declined_precision_falls_back_to_the_existing_ols_path():
    """A declined precision test must reproduce ``method="ols"`` exactly.

    A custom float64 residualization was tried for the case the precision path
    declines, gated by a residual-reliability threshold. It is not viable: a
    fixed threshold on the residual norm either discards real signal or
    accepts noise depending on the condition number of ``Z``, and the
    threshold's correct form needs an SVD of ``Z`` to get right — strictly
    more machinery than a residual-norm comparison can provide.

    Falling back to ``_compute_parcorr`` (the ``method="ols"`` computation,
    unchanged since before this class supported a precision path at all)
    sidesteps that entire design problem: for the tests the precision path
    declines, scalar ``__call__`` returns exactly what every scalar test
    already returned in production before this class had a precision path,
    which cannot be a regression by construction. It is not necessarily an
    accurate answer on some of these inputs (float32 OLS has its own numerical
    limits near collinearity) — only an unchanged one.
    """
    data = _near_singular_case()
    ci = ParCorrGPU(data, device=DEVICE)
    ols = ParCorrGPU(data, device=DEVICE, method="ols")

    assert ci._compute_parcorr_precision(0, 1, [2, 3]) is None
    assert ci(0, 1, [2, 3]) == ols(0, 1, [2, 3])


def test_statistic_never_leaves_the_correlation_range():
    """No input may produce |rho| > 1, across a sweep of conditioning levels."""
    for k in range(12):
        data = _near_singular_case(eps=10.0 ** (-4 - k), seed=k)
        _, stat = ParCorrGPU(data, device=DEVICE)(0, 1, [2, 3])
        assert abs(stat) <= 1.0, f"|rho| = {abs(stat)} at eps=1e-{4 + k}"


def test_well_conditioned_data_uses_the_precision_path():
    """The conditioning guard must not divert ordinary data to the slow path."""
    rng = np.random.default_rng(3)
    data = rng.standard_normal((500, 12))
    ci = ParCorrGPU(data, device=DEVICE)
    for X in range(4):
        for Y in range(4, 8):
            assert ci._compute_parcorr_precision(X, Y, [8, 9, 10]) is not None


def test_precision_and_ols_do_not_share_cache_entries():
    """The two estimators differ numerically, so they must key separately."""
    rng = np.random.default_rng(4)
    data = rng.standard_normal((300, 6))
    precision = ParCorrGPU(data, device=DEVICE)
    ols = ParCorrGPU(data, device=DEVICE, method="ols")
    assert precision.param_hash != ols.param_hash


def test_non_integer_indices_are_rejected():
    """Floats and bools are not column indices and must not be coerced to one.

    Validation must happen before the cache lookup: ``_get_array_hash`` calls
    ``int(X)`` unconditionally, so a non-integer index that collides with an
    already-cached integer one would otherwise silently return that cached
    result instead of raising.
    """
    rng = np.random.default_rng(5)
    data = rng.standard_normal((200, 5))
    ci = ParCorrGPU(data, device=DEVICE)
    ci(1, 2, [])  # populate the cache entry that a coerced 1.9 or True would hit

    with pytest.raises(TypeError):
        ci(1.9, 2, [])
    with pytest.raises(TypeError):
        ci(True, 2, [])
    with pytest.raises(TypeError):
        ci(0, 3, [True])  # non-integer element inside condition_set

    # numpy integers remain valid indices
    assert ci(np.int64(1), 2, [])[1] == ci(1, 2, [])[1]


def test_condition_set_overlapping_x_or_y_declines_precision():
    """X or Y appearing in its own condition_set must not use the inverse.

    The sub-covariance is singular by construction (X's row/column is
    duplicated), so the precision path must decline; the fallback then
    reproduces whatever ``method="ols"`` already gives, which is not
    necessarily exactly zero (float32 residualizing X against Z that includes
    X leaves a small but nonzero rounding residual) — that is unchanged
    legacy behavior, not a new guarantee this path makes.
    """
    for seed in range(20):
        data = np.random.default_rng(seed).standard_normal((300, 4))
        ci = ParCorrGPU(data, device=DEVICE)
        assert ci._compute_parcorr_precision(0, 1, [0]) is None
        ols = ParCorrGPU(data, device=DEVICE, method="ols")
        assert ci(0, 1, [0]) == ols(0, 1, [0])


def test_conditioning_guard_is_permutation_invariant():
    """The precision-path decline decision must not depend on Z's order.

    ``_get_array_hash`` sorts the conditioning set for the cache key, so two
    callers passing the same set in a different order share one cache entry —
    the guard deciding whether to trust the inverse must agree regardless of
    order, or the result becomes call-order dependent. A Cholesky-pivot
    criterion failed this: reciprocal condition 4e-18 (should decline) passed
    the pivot test for one ordering of a 12-element Z while a permutation of
    the same set correctly declined, disagreeing by rho = -0.17 vs -0.97.
    """
    data = _reciprocal_condition_case()
    ci = ParCorrGPU(data, device=DEVICE)

    z_a = list(range(1, 13))
    z_b = [8, 2, 3, 5, 11, 9, 4, 12, 10, 6, 1, 7]
    assert sorted(z_a) == sorted(z_b)

    assert ci._compute_parcorr_precision(0, 13, z_a) is None
    assert ci._compute_parcorr_precision(0, 13, z_b) is None

    ols = ParCorrGPU(data, device=DEVICE, method="ols")
    assert ci(0, 13, z_a) == ols(0, 13, z_a)


def test_duplicate_column_under_a_different_index_declines():
    """X exactly duplicated as a Z column, under a different column index.

    The explicit ``X in condition_set`` check (in the precision path) only
    catches a duplicate at the same column index; a duplicate under a
    *different* index still reaches the precision path's conditioning guard,
    which must also decline it.
    """
    rng = np.random.default_rng(3975)
    n = 30
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    data = np.column_stack([x, y, x.copy()])
    ci = ParCorrGPU(data, device=DEVICE)

    assert ci._compute_parcorr_precision(0, 1, [2]) is None
    ols = ParCorrGPU(data, device=DEVICE, method="ols")
    assert ci(0, 1, [2]) == ols(0, 1, [2])


def test_residual_at_the_rounding_floor_declines():
    """A variable shared with Z up to O(1e-15) noise must decline, not guess.

    The residual's *direction* here is dominated by rounding, not signal — an
    earlier fallback design computed a specific (wrong) rho from it rather than
    declining. The precision path's conditioning guard must decline this case.
    """
    rng = np.random.default_rng(12)
    n = 100
    z = rng.standard_normal(n)
    u = rng.standard_normal(n)
    v = rng.standard_normal(n)
    x = z + 1e-15 * u
    y = z + 1e-15 * v
    data = np.column_stack([x, y, z])
    ci = ParCorrGPU(data, device=DEVICE)

    assert ci._compute_parcorr_precision(0, 1, [2]) is None
    ols = ParCorrGPU(data, device=DEVICE, method="ols")
    assert ci(0, 1, [2]) == ols(0, 1, [2])


def test_precision_decision_is_invariant_to_x_y_z_ordering():
    """Cache-equivalent (X, Y, Z) orderings must reach the same decision.

    ``eigvalsh`` is permutation-invariant mathematically but not bit-for-bit in
    floating point, while ``_get_array_hash`` sorts X/Y and Z into one cache
    key regardless of caller order. On a matrix constructed to sit within
    ~1e-5 relative of ``_RCOND``, the four orderings of one semantic test
    computed reciprocal conditions differing in the sixth significant digit —
    close enough that a different BLAS build could land only one ordering on
    the other side of the cutoff, letting a cache-equivalent query accept on
    one call and decline (to a very different ``method="ols"`` value) on
    another. The conditioning check now runs on indices canonicalized the same
    way the cache key is, making the four orderings compute on one identical
    array.
    """
    rng = np.random.default_rng(2)
    n = 300
    w = rng.standard_normal(n)
    z = rng.standard_normal(n)
    u = rng.standard_normal(n)
    v = rng.standard_normal(n)
    eps = 3.409376130181348e-05
    data = np.column_stack([w + u, w + v, z, z + eps * w])
    ci = ParCorrGPU(data, device=DEVICE)

    orderings = [(0, 1, [2, 3]), (1, 0, [2, 3]), (0, 1, [3, 2]), (1, 0, [3, 2])]
    decisions = {ci._compute_parcorr_precision(X, Y, Z) for X, Y, Z in orderings}
    assert len(decisions) == 1


def test_precision_decision_invariant_near_boundary_property_sweep():
    """Broader property check for the same bug: not just one pinned construction.

    Sweeps constructions whose reciprocal condition lands within two orders of
    magnitude of ``_RCOND`` on either side — the band where floating-point,
    order-dependent noise in ``eigvalsh`` could plausibly flip an accept/decline
    decision between cache-equivalent orderings — and asserts all four
    orderings of (X, Y) and Z agree for every case landing in that band.
    """
    orderings = [(0, 1, [2, 3]), (1, 0, [2, 3]), (0, 1, [3, 2]), (1, 0, [3, 2])]
    n_checked = 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        n = 100
        w = rng.standard_normal(n)
        z = rng.standard_normal(n)
        u = rng.standard_normal(n)
        v = rng.standard_normal(n)
        eps = 10 ** rng.uniform(-6, -3)
        data = np.column_stack([w + u, w + v, z, z + eps * w])
        ci = ParCorrGPU(data, device=DEVICE)

        sub = ci._cov_np[np.ix_([0, 1, 2, 3], [0, 1, 2, 3])]
        diag = np.diag(sub)
        if not (diag > 0).all():
            continue
        scale = np.sqrt(diag)
        eig = np.linalg.eigvalsh(sub / scale / scale[:, None])
        rcond = eig[0] / eig[-1]
        if not (1e-11 < rcond < 1e-9):
            continue  # only the boundary band is at risk of a split decision

        n_checked += 1
        decisions = {ci._compute_parcorr_precision(X, Y, Z) for X, Y, Z in orderings}
        assert (
            len(decisions) == 1
        ), f"seed {seed}: orderings disagree, rcond={rcond:.3e}"

    assert n_checked >= 20, "sweep should reliably land several cases in the band"


def _fresh_scalar(data, X, Y, Z):
    """``__call__`` on a new instance, so no earlier result is served from cache."""
    return ParCorrGPU(data, device=DEVICE)(X, Y, Z)


def test_batch_paths_decline_where_the_scalar_guard_declines():
    """Issue #74: both batched entry points must fall back where #73 declines.

    On the ~4e-18 construction, ``_batch_test_precision`` and
    ``_batch_test_fast`` inverted unconditionally and returned rho ~ -0.138
    / p ~ 0.485 while the guarded scalar path gives rho ~ -0.023 / p ~ 0.91 —
    the same test answering differently depending only on which entry point
    reached it. The batched fallback must return exactly the scalar answer.
    """
    data = _reciprocal_condition_case()
    z = list(range(1, 13))
    assert ParCorrGPU(data, device=DEVICE)._compute_parcorr_precision(0, 13, z) is None

    want = _fresh_scalar(data, 0, 13, z)
    for kwargs in ({}, {"enable_caching": False}):
        got = ParCorrGPU(data, device=DEVICE, **kwargs).batch_test([(0, 13, z)])[0]
        assert got == want


def test_batch_declined_and_accepted_tests_share_one_chunk():
    """A mixed chunk routes declined tests to OLS and keeps accepted ones exact.

    Covers the two hazards issue #74 flagged around the fallback: it must not
    break dedup (cache-equivalent orderings of a declined test share one
    result) or caching (a rerun is served from ``pvalue_cache``), and the
    well-conditioned members of the same |Z| chunk must still match the
    scalar precision path as ``test_scalar_matches_batch`` demands.
    """
    rng = np.random.default_rng(1)
    data = np.column_stack([_near_singular_case(), rng.standard_normal((300, 2))])
    tests = [
        (0, 1, [2, 3]),  # rank-deficient → OLS fallback
        (1, 0, [3, 2]),  # same test, other order → dedup
        (0, 4, [2, 3]),  # another rank-deficient test, distinct key
        (0, 1, [4, 5]),  # well-conditioned → precision, same chunk
        (0, 1, [2]),  # well-conditioned, other chunk
    ]
    ci = ParCorrGPU(data, device=DEVICE)
    for i, t in enumerate(tests):
        assert (ci._compute_parcorr_precision(*t) is None) == (i < 3)
    results = ci.batch_test(list(tests))

    assert results[0] == results[1]
    for i in (0, 2):
        assert results[i] == _fresh_scalar(data, *tests[i])
    for i in (3, 4):
        p_s, s_s = _fresh_scalar(data, *tests[i])
        assert np.isclose(results[i][1], s_s, rtol=0, atol=1e-12)
        assert np.isclose(results[i][0], p_s, rtol=0, atol=1e-12)

    # Cache: a rerun is served, not recomputed.
    assert ci.batch_test(list(tests)) == results

    # Without dedup each ordering is evaluated on its own; same answers.
    fast = ParCorrGPU(data, device=DEVICE, enable_caching=False)
    assert fast.batch_test(list(tests)) == results


@pytest.mark.parametrize("eps", [1.000043771885943e-4, 1.000043971985993e-4])
def test_batch_matches_scalar_at_the_rcond_cutoff(eps):
    """Rows near ``_RCOND`` take the scalar guard's decision, not torch's.

    NumPy and batched torch ``eigvalsh`` differ by ~1e-6 relative in rcond.
    On the macOS CPU BLAS this was measured on, these two inputs fall on
    opposite sides of the cutoff in each direction (NumPy 1.0000005e-10 vs
    torch 0.9999994e-10, and 0.99999999e-10 vs 1.0000010e-10). The
    assertion holds on any BLAS, but only exercises the fix where it flips.
    """
    data = _near_singular_case(eps=eps)
    t = (0, 1, [2, 3])
    want = _fresh_scalar(data, *t)
    for kwargs in ({}, {"enable_caching": False}):
        assert ParCorrGPU(data, device=DEVICE, **kwargs).batch_test([t]) == [want]


def test_batch_exact_correlation_gets_zero_pvalue():
    """A raw rho of 1 + 2e-16 is clipped before the exact-one check."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal(31)
    data = np.column_stack([x, x + 1e-12 * rng.standard_normal(31)])
    want = _fresh_scalar(data, 0, 1, [])
    assert want == (0.0, 1.0)
    for kwargs in ({}, {"enable_caching": False}):
        got = ParCorrGPU(data, device=DEVICE, **kwargs).batch_test([(0, 1, [])])
        assert got == [want]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_declined_fallback_runs_on_mps():
    """MPS has no ``lstsq``; the declined fallback runs on CPU instead."""
    data = _near_singular_case()
    t = (0, 1, [2, 3])
    want = ParCorrGPU(data, device="cpu")(*t)
    assert ParCorrGPU(data, device="mps")(*t) == want
    for kwargs in ({}, {"enable_caching": False}):
        assert ParCorrGPU(data, device="mps", **kwargs).batch_test([t]) == [want]


def test_batch_unconditional_constant_column_falls_back():
    """The k == 2 variance gate must match the scalar path (issue #74 scope).

    A constant column makes the (X, Y) sub-covariance singular. The scalar
    path has always declined it and answered through OLS (rho = 0, p = 1);
    the batched paths computed ``cov / sqrt(var * 0)`` and returned
    ``(nan, nan)`` — an invalid p-value.
    """
    rng = np.random.default_rng(2)
    base = rng.standard_normal((200, 4))
    data = np.column_stack([base, np.ones(200)])
    tests = [(0, 4, []), (1, 4, [])]

    want = [_fresh_scalar(data, *t) for t in tests]
    assert want == [(1.0, 0.0)] * 2
    for kwargs in ({}, {"enable_caching": False}):
        got = ParCorrGPU(data, device=DEVICE, **kwargs).batch_test(list(tests))
        assert got == want


def test_declined_fallback_is_order_invariant_and_in_range():
    """Cache-equivalent orderings of a declined test must agree, in [-1, 1].

    Float32 ``lstsq`` on these inputs moved rho by up to 0.06 between
    orderings of one test, and could return 1 + 1.2e-7 (e.g. seed 2,
    a = 0.05, X, Y = 0, 1 on this BLAS). Both entry points canonicalize and
    clip the fallback, so every ordering and path gives one in-range answer.
    """
    n_declined = 0
    for seed in range(4):
        for a in (0.02, 0.05):
            data = _reciprocal_condition_case(seed=seed, k=8, T=60, a=a)
            ci = ParCorrGPU(data, device=DEVICE)
            for X, Y in itertools.combinations(range(8), 2):
                z = [c for c in range(8) if c not in (X, Y)][:5]
                if ci._compute_parcorr_precision(X, Y, z) is not None:
                    continue
                n_declined += 1
                orderings = [(X, Y, z), (Y, X, z[::-1])]
                want = _fresh_scalar(data, *orderings[0])
                assert abs(want[1]) <= 1.0
                for t in orderings:
                    assert _fresh_scalar(data, *t) == want
                    for kwargs in ({}, {"enable_caching": False}):
                        batch = ParCorrGPU(data, device=DEVICE, **kwargs)
                        assert batch.batch_test([t]) == [want]
    assert n_declined >= 50
