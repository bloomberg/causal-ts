# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for ParCorrGPU conditional independence test."""

import numpy as np
import pytest

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


def test_batch_paths_decline_where_the_scalar_guard_declines():
    """Issue #74: both batched entry points must fall back where #73 declines.

    On the ~4e-18 construction, ``_batch_test_precision`` and
    ``_batch_test_fast`` inverted unconditionally and returned rho ~ -0.138
    / p ~ 0.485 while the guarded scalar path (and ``method="ols"``) gives
    rho ~ -0.02 / p ~ 0.91 — the same test answering differently depending
    only on which entry point reached it.

    The batched fallback must be the OLS estimator: bitwise-identical in the
    statistic to ``method="ols"``'s own batch (both run ``_ols_rho``), and
    within float32 batched-vs-vector lstsq noise of the scalar OLS value.
    ``cond(Z)`` as stored in float32 is ~1.5e4 here, so those two
    formulations legitimately differ at the ~1e-3 level with no precision
    path involved (measured on main: ``method="ols"`` scalar -0.0227 vs
    ``method="ols"`` batch -0.0181).
    """
    data = _reciprocal_condition_case()
    z = list(range(1, 13))
    scalar = ParCorrGPU(data, device=DEVICE)
    ols_scalar = ParCorrGPU(data, device=DEVICE, method="ols")
    ols_batch = ParCorrGPU(data, device=DEVICE, method="ols")

    assert scalar._compute_parcorr_precision(0, 13, z) is None
    assert scalar(0, 13, z) == ols_scalar(0, 13, z)

    # Reference first, on a fresh instance: a prior scalar call on this
    # instance would populate `pvalue_cache` and make `batch_test` return it.
    p_ols_b, s_ols_b = ols_batch.batch_test([(0, 13, z)])[0]
    for kwargs in ({}, {"enable_caching": False}):
        p, s = ParCorrGPU(data, device=DEVICE, **kwargs).batch_test([(0, 13, z)])[0]
        assert s == s_ols_b
        assert np.isclose(p, p_ols_b, rtol=0, atol=1e-8)

    p_scalar, s_scalar = ols_scalar(0, 13, z)
    assert np.isclose(s, s_scalar, rtol=0, atol=2e-2)
    assert np.isclose(p, p_scalar, rtol=0, atol=5e-2)


def test_batch_declined_and_accepted_tests_share_one_chunk():
    """A mixed chunk routes declined tests to OLS and keeps accepted ones exact.

    Covers the two hazards issue #74 flagged around the fallback: it must not
    break dedup (cache-equivalent orderings of a declined test share one OLS
    result) or caching (a rerun is served from ``pvalue_cache``), and the
    well-conditioned members of the same chunk must still match the scalar
    precision path as ``test_scalar_matches_batch`` demands.
    """
    data = _near_singular_case()
    tests = [
        (0, 1, [2, 3]),  # rank-deficient → OLS fallback
        (1, 0, [3, 2]),  # same test, other order → dedup / canonical row
        (0, 1, [2]),  # well-conditioned → precision
        (0, 1, []),  # well-conditioned → precision
    ]
    ci = ParCorrGPU(data, device=DEVICE)
    results = ci.batch_test(list(tests))

    # The declined pair shares one result (dedup), equal to the legacy OLS
    # batch computation — same `_ols_rho` code, bitwise statistic.
    assert results[0][1] == results[1][1]
    p_o, s_o = ParCorrGPU(data, device=DEVICE, method="ols").batch_test([tests[0]])[0]
    assert results[0][1] == s_o
    assert np.isclose(results[0][0], p_o, rtol=0, atol=1e-8)

    # Accepted members stay on the precision path, equal to the scalar call.
    scalar = ParCorrGPU(data, device=DEVICE)
    for i in (2, 3):
        X, Y, Z = tests[i]
        assert scalar._compute_parcorr_precision(X, Y, Z) is not None
        p_s, s_s = scalar(X, Y, Z)
        assert np.isclose(results[i][1], s_s, rtol=0, atol=1e-12)
        assert np.isclose(results[i][0], p_s, rtol=0, atol=1e-12)

    # Cache: a rerun is served, not recomputed.
    assert ci.batch_test(list(tests)) == results

    # The no-cache path reaches the same decisions. Its declined rows may
    # move in the last digits: dedup shrinks the OLS batch (B=1 vs B=2) and
    # float32 batched lstsq is not bit-stable across batch shapes — the
    # legacy `_batch_test_ols` has the same property. Tolerances are ~100x
    # the measured drift (1.2e-7) and ~300x below what the unguarded inverse
    # produced (0.971 vs the OLS 0.929, and 1.0 for the permuted duplicate).
    fast = ParCorrGPU(data, device=DEVICE, enable_caching=False)
    r_fast = fast.batch_test(list(tests))
    for i in (2, 3):
        assert r_fast[i] == results[i]
    for j in (0, 1):
        assert abs(r_fast[j][1] - s_o) <= 1e-4
        assert np.isclose(r_fast[j][0], p_o, rtol=0.01, atol=0)
    assert abs(r_fast[0][1] - r_fast[1][1]) <= 1e-4


def test_batch_unconditional_constant_column_falls_back():
    """The k == 2 variance gate must match the scalar path (issue #74 scope).

    A constant column makes the (X, Y) sub-covariance singular. The scalar
    path has always declined it and answered through OLS (rho = 0, p = 1);
    the batched paths computed ``cov / sqrt(var * 0)`` and returned
    ``(nan, nan)`` — an invalid p-value. The guard's diagonal check must
    decline it to the same OLS answer.
    """
    rng = np.random.default_rng(2)
    base = rng.standard_normal((200, 4))
    data = np.column_stack([base, np.ones(200)])
    tests = [(0, 4, []), (1, 4, [])]

    want = ParCorrGPU(data, device=DEVICE, method="ols").batch_test(list(tests))
    assert ParCorrGPU(data, device=DEVICE)(0, 4, []) == want[0]
    for kwargs in ({}, {"enable_caching": False}):
        got = ParCorrGPU(data, device=DEVICE, **kwargs).batch_test(list(tests))
        for (p, s), (p_w, s_w) in zip(got, want):
            assert s == s_w
            assert np.isclose(p, p_w, rtol=0, atol=1e-8)
            assert 0 <= p <= 1
