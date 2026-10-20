# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

import warnings
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from causalts.ci_tests import create_ci_test, list_ci_tests
from causalts.ci_tests.raft import RAFTCI
from causalts.cli import main

# raft's defaults before the ridge-penalized Z adjustment, HC2 and the linear
# channel (the robust raft); and its original settings, before the Huber residual,
# tested-side feature count and Cauchy combination (rift's earlier defaults too).
ROBUST = {"z_ridge": False, "hc2": False, "linear_channel": False}
LEGACY = {"robust": False, "n_rff_tested": 8, "combine": "bonferroni", **ROBUST}


def _null_data(n, rng):
    z = rng.standard_normal(n)
    x = np.sin(z) + 0.5 * rng.standard_normal(n)
    y = np.tanh(2 * z) + 0.5 * rng.standard_normal(n)
    return np.column_stack([x, y, z])


def test_registered():
    assert "raft" in list_ci_tests()


@pytest.mark.parametrize("variance", ["f", "dwb"])
def test_detects_nonlinear_dependence(variance):
    rng = np.random.default_rng(0)
    n = 400
    z = rng.standard_normal(n)
    x = rng.standard_normal(n)
    y = np.sin(2 * x) + 0.3 * z + 0.3 * rng.standard_normal(n)
    p, _ = create_ci_test("raft", np.column_stack([x, y, z]), variance=variance)(
        0, 1, [2]
    )
    assert p < 1e-6  # moment-matched null: no 1/(n_boot+1) floor


@pytest.mark.parametrize("variance", ["f", "dwb"])
def test_null_rejection_near_alpha(variance):
    rng = np.random.default_rng(1)
    rej = 0
    for _ in range(100):
        ci = create_ci_test("raft", _null_data(300, rng), variance=variance)
        rej += ci(0, 1, [2])[0] <= 0.05
    assert 1 <= rej <= 12  # nominal 5; generous bounds for 100 draws


def test_bootstrap_fixes_heteroscedastic_null():
    """Plain F over-rejects when the error scale depends on Z; the bootstrap does not."""
    rng = np.random.default_rng(2)
    rej = {"f": 0, "dwb": 0}
    for _ in range(100):
        n = 500
        z = rng.standard_normal(n)
        s = np.exp(0.8 * z)
        x = np.sin(z) + 0.5 * s * rng.standard_normal(n)
        y = np.tanh(z) + 0.5 * s * rng.standard_normal(n)
        d = np.column_stack([x, y, z])
        for v in rej:
            rej[v] += create_ci_test("raft", d, variance=v)(0, 1, [2])[0] <= 0.05
    assert rej["f"] >= 20
    assert rej["dwb"] <= 12


def test_symmetric_and_cached():
    d = _null_data(300, np.random.default_rng(3))
    ci = create_ci_test("raft", d)
    assert ci(0, 1, [2]) == ci(1, 0, [2])
    n_before = ci.n_actual_tests
    ci(0, 1, [2])
    assert ci.n_actual_tests == n_before


def test_unconditional_and_constant_columns():
    rng = np.random.default_rng(4)
    x = rng.standard_normal(200)
    d = np.column_stack([x, x**2 + 0.1 * rng.standard_normal(200), np.ones(200)])
    ci = create_ci_test("raft", d)
    assert ci(0, 1, [])[0] < 1e-6
    p, _ = ci(0, 1, [2])  # constant conditioning column is ignored, not an error
    assert p < 1e-6


def test_saturated_conditioning_set_returns_one():
    rng = np.random.default_rng(5)
    d = rng.standard_normal((40, 8))
    ci = create_ci_test("raft", d, **ROBUST)
    with pytest.warns(RuntimeWarning, match="reads as independence"):
        p, _ = ci(0, 1, list(range(2, 8)))  # 6 * 9 features > 40 rows
    assert p == 1.0
    assert ci.n_saturated >= 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # warned once per instance only
        assert ci(2, 3, list(range(4, 8)) + [0, 1])[0] == 1.0
    assert ci.n_saturated >= 2


def test_bad_variance_raises():
    with pytest.raises(ValueError):
        create_ci_test("raft", np.zeros((50, 3)), variance="nope")


def test_data_swap_invalidates_feature_cache():
    """Adapters assign a new array per test; results must match a fresh instance."""
    rng = np.random.default_rng(6)
    a = _null_data(300, rng)
    b = rng.standard_normal((300, 3))
    b[:, 1] = np.sin(2 * b[:, 0]) + 0.3 * rng.standard_normal(300)
    ci = create_ci_test("raft", a)
    ci(0, 1, [2])
    ci.data = b
    assert ci(0, 1, [2]) == create_ci_test("raft", b)(0, 1, [2])


def test_bootstrap_null_with_large_conditioning_set():
    """The bootstrap residual sum of squares must exclude span(Z); otherwise the test
    turns liberal as the conditioning features grow relative to n."""
    rng = np.random.default_rng(7)
    rej = 0
    for _ in range(100):
        n = 150
        z = rng.standard_normal((n, 10))
        x = np.sin(z[:, 0]) + 0.5 * rng.standard_normal(n)
        y = np.tanh(z[:, 1]) + 0.5 * rng.standard_normal(n)
        d = np.column_stack([x, y, z])
        rej += create_ci_test("raft", d)(0, 1, list(range(2, 12)))[0] <= 0.05
    assert rej <= 12  # the uncorrected denominator rejects ~40%


# ----- rift: interaction-adjusted ------------------------------------------------


def _shared_interaction_null(n, rng):
    """X and Y both depend on z1*z2 (no edge between them): an additive model of Z
    cannot absorb the shared interaction."""
    z = rng.standard_normal((n, 3))
    g = z[:, 0] * z[:, 1]
    return np.column_stack(
        [g + 0.5 * rng.standard_normal(n), np.tanh(g) + 0.5 * rng.standard_normal(n), z]
    )


def test_rift_registered_and_same_class():
    assert "rift" in list_ci_tests()
    d = _null_data(100, np.random.default_rng(20))
    assert create_ci_test("rift", d).adjust == "interaction"
    assert create_ci_test("raft", d).adjust == "additive"


def test_rift_reduces_to_raft_without_its_dictionary():
    d = _null_data(300, np.random.default_rng(21))
    raft = create_ci_test("raft", d, **LEGACY)
    assert create_ci_test("rift", d, **LEGACY)(0, 1, []) == raft(0, 1, [])
    # One conditioning variable and no frequency grid: no dictionary terms at all.
    rift = create_ci_test("rift", d, freqs=(), **LEGACY)
    assert rift(0, 1, [2]) == raft(0, 1, [2])
    # With the defaults rift's dictionary is raft's plus its own terms, so without
    # them the two are the same test.
    new = create_ci_test("raft", d)
    assert create_ci_test("rift", d)(0, 1, []) == new(0, 1, [])
    assert create_ci_test("rift", d, freqs=())(0, 1, [2]) == new(0, 1, [2])


def test_rift_calibrated_where_raft_fails():
    """Shared z1*z2 null: raft rejects essentially always, rift holds its level."""
    rng = np.random.default_rng(10)
    rej = {"raft": 0, "rift": 0}
    for _ in range(100):
        d = _shared_interaction_null(300, rng)
        for t in rej:
            rej[t] += create_ci_test(t, d)(0, 1, [2, 3, 4])[0] <= 0.05
    assert rej["raft"] >= 90
    assert 1 <= rej["rift"] <= 12


def test_rift_null_rejection_near_alpha_additive():
    rng = np.random.default_rng(11)
    ps = []
    for _ in range(100):
        n = 300
        z = rng.standard_normal((n, 2))
        x = np.sin(z[:, 0]) + 0.5 * rng.standard_normal(n)
        y = np.tanh(z[:, 1]) + np.cos(z[:, 0]) + 0.5 * rng.standard_normal(n)
        ps.append(create_ci_test("rift", np.column_stack([x, y, z]))(0, 1, [2, 3])[0])
    ps = np.array(ps)
    assert np.sum(ps <= 0.05) <= 12
    assert np.sum(ps <= 0.10) >= 1  # not conservative to the point of p = 1


def test_rift_detects_nonlinear_dependence():
    rng = np.random.default_rng(12)
    n = 400
    z = rng.standard_normal((n, 2))
    x = rng.standard_normal(n)
    y = np.sin(2 * x) + z[:, 0] * z[:, 1] + 0.3 * rng.standard_normal(n)
    p, _ = create_ci_test("rift", np.column_stack([x, y, z]))(0, 1, [2, 3])
    assert p < 1e-6


def test_rift_conditioning_order_invariant_and_symmetric():
    d = _shared_interaction_null(300, np.random.default_rng(13))
    ci = create_ci_test("rift", d)
    ref = ci(0, 1, [2, 3, 4])
    assert create_ci_test("rift", d)(0, 1, [4, 2, 3]) == ref
    assert create_ci_test("rift", d)(1, 0, [3, 4, 2]) == ref


def test_rift_data_swap_and_nan_rows():
    rng = np.random.default_rng(14)
    a = _shared_interaction_null(300, rng)
    b = _shared_interaction_null(300, rng)
    ci = create_ci_test("rift", a)
    ci(0, 1, [2, 3, 4])
    ci.data = b
    assert ci(0, 1, [2, 3, 4]) == create_ci_test("rift", b)(0, 1, [2, 3, 4])
    b_nan = b.copy()
    b_nan[5, 3] = np.nan
    p, _ = create_ci_test("rift", b_nan)(0, 1, [2, 3, 4])
    assert np.isfinite(p) and 0.0 <= p <= 1.0


@pytest.mark.parametrize("value", [0.0, 1.0, 1e7])
@pytest.mark.parametrize("name", ["raft", "rift"])
def test_constant_tested_variable_gives_p_one(name, value):
    """A constant (including all-zero) tested column must not read as a perfect fit."""
    d = _shared_interaction_null(200, np.random.default_rng(15))
    d[:, 0] = value
    assert create_ci_test(name, d)(0, 1, [2, 3, 4])[0] == 1.0
    assert create_ci_test(name, d)(0, 1, [])[0] == 1.0


@pytest.mark.parametrize("name", ["raft", "rift"])
def test_scale_and_offset_invariant(name):
    """A tiny signal, or one on a large offset, is a signal, not a constant."""
    rng = np.random.default_rng(33)
    n = 300
    u, z, w = rng.standard_normal((3, n))
    y = np.sin(2 * u) + 0.3 * z + 0.3 * rng.standard_normal(n)
    ref = create_ci_test(name, np.column_stack([u, y, z, w]))(0, 1, [2, 3])[0]
    assert ref < 1e-6
    for x in (1e-13 * u, 1e12 + 0.5 * u):
        p = create_ci_test(name, np.column_stack([x, y, z, w]))(0, 1, [2, 3])[0]
        assert p == pytest.approx(ref, rel=1e-2, abs=0)


@pytest.mark.parametrize("name", ["raft", "rift"])
def test_vector_with_constant_member_and_response_explained_by_z(name):
    d = _shared_interaction_null(200, np.random.default_rng(16))
    d = np.column_stack([d, np.zeros(200)])  # column 5: all zero
    ref = create_ci_test(name, d)(0, 1, [2, 3, 4])
    assert create_ci_test(name, d)([0, 5], 1, [2, 3, 4]) == ref
    d[:, 1] = d[:, 2]  # Y is a conditioning variable: nothing left to explain
    assert create_ci_test(name, d)(0, 1, [2, 3, 4])[0] == 1.0


@pytest.mark.parametrize("name", ["raft", "rift"])
def test_tested_component_inside_z_is_dropped(name):
    d = _shared_interaction_null(300, np.random.default_rng(34))
    ref = create_ci_test(name, d)(0, 1, [2, 3])
    assert create_ci_test(name, d)([0, 2], 1, [2, 3]) == ref
    assert create_ci_test(name, d)(2, 1, [2, 3])[0] == 1.0


def test_in_place_mutation_does_not_reuse_stale_features():
    rng = np.random.default_rng(35)
    d = _null_data(300, rng)
    for name in ("raft", "rift"):
        a = d.copy()
        ci = create_ci_test(name, a)
        ci(0, 1, [2])
        a[:, 2] = rng.standard_normal(300)
        assert ci(0, 1, [2]) == create_ci_test(name, a.copy())(0, 1, [2])


def test_set_data_flushes_buffer_to_old_cache_file(tmp_path):
    rng = np.random.default_rng(36)
    a, b = _null_data(200, rng), _null_data(200, rng)
    ci = create_ci_test("raft", a, cache_dir=str(tmp_path))
    ci(0, 1, [2])
    old_file = ci.cache_file
    ci.set_data(b)
    assert ci.cache_file != old_file
    assert len(Path(old_file).read_text().splitlines()) == 2  # header + the a result
    assert len(Path(ci.cache_file).read_text().splitlines()) == 1


@pytest.mark.parametrize("name", ["raft", "rift"])
def test_tested_component_copying_z_is_dropped(name):
    """Adapters copy columns, so a tested component equal to a Z column has its own
    index; it must still be recognised. An affine copy is not removed (a loose match
    could erase a genuine variable) but carries nothing beyond Z: its direction has
    p = 1 and it only adds a Bonferroni share."""
    d = _shared_interaction_null(300, np.random.default_rng(37))
    d = np.column_stack([d, d[:, 2], 5.0 - 2.0 * d[:, 3]])
    kw = LEGACY if name == "rift" else {}
    ref = create_ci_test(name, d, **kw)(0, 1, [2, 3])
    assert create_ci_test(name, d, **kw)([0, 5], 1, [2, 3]) == ref
    p_aff = create_ci_test(name, d, **kw)([0, 6], 1, [2, 3])[0]
    assert p_aff == pytest.approx(min(1.0, 1.5 * ref[0]), rel=1e-4)
    assert create_ci_test(name, d, **kw)(5, 1, [2, 3])[0] == 1.0
    if name == "rift":
        # default rift: the copy is dropped exactly; the affine copy's untested
        # direction sends the combination to Bonferroni, which is conservative
        ref = create_ci_test(name, d)(0, 1, [2, 3])
        assert create_ci_test(name, d)([0, 5], 1, [2, 3]) == ref
        assert create_ci_test(name, d)([0, 6], 1, [2, 3])[0] >= ref[0]


def test_rift_order_invariant_with_tied_prefixes():
    """Distinct conditioning columns that agree on their first values and moments
    must still be ordered by content, not by position."""
    rng = np.random.default_rng(38)
    n = 300
    head, tail = rng.standard_normal(40), rng.standard_normal(n - 40)
    # Same first values and the same multiset of values (hence equal moments).
    zs = [np.r_[head, rng.permutation(tail)] for _ in range(4)]
    g = zs[0] * zs[1] + np.sin(zs[2])
    d = np.column_stack(
        [g + 0.5 * rng.standard_normal(n), np.tanh(g) + 0.5 * rng.standard_normal(n)]
        + zs
    )
    ref = create_ci_test("rift", d)(0, 1, [2, 3, 4, 5])[0]
    for perm in ([5, 4, 3, 2], [3, 5, 2, 4]):
        e = d[:, [0, 1] + perm]
        assert create_ci_test("rift", e)(0, 1, [2, 3, 4, 5])[0] == pytest.approx(
            ref, rel=1e-6
        )


def test_rift_keeps_near_copies_distinct():
    rng = np.random.default_rng(39)
    z = rng.standard_normal(400)
    d = np.column_stack(
        [rng.standard_normal((400, 2)), z, z + 3e-5 * rng.standard_normal(400)]
    )
    ci = create_ci_test("rift", d)
    _, U, s = ci._z_parts([2, 3], d)
    assert s.size  # two distinct variables: the dictionary is built


@pytest.mark.parametrize("name", ["raft", "rift"])
def test_nan_introduced_in_place_is_masked(name):
    d = _shared_interaction_null(300, np.random.default_rng(40))
    ci = create_ci_test(name, d)
    ci(0, 1, [2, 3, 4])
    d[7, 2] = np.nan  # edited after assignment
    p, _ = ci(0, 1, [2])  # a fresh (uncached) test on the mutated array
    assert p == create_ci_test(name, d.copy())(0, 1, [2])[0]
    assert np.isfinite(ci(0, 1, [2, 3, 4])[0])


@pytest.mark.parametrize("name", ["raft", "rift"])
def test_integer_and_extreme_magnitude_data(name):
    rng = np.random.default_rng(41)
    n = 300
    u, z, w = rng.standard_normal((3, n))
    y = np.sin(2 * u) + 0.3 * z + 0.3 * rng.standard_normal(n)
    ref = create_ci_test(name, np.column_stack([u, y, z, w]))(0, 1, [2, 3])[0]
    for scale in (1e160, 1e170, 1e-300):
        p = create_ci_test(name, np.column_stack([scale * u, y, z, w]))(0, 1, [2, 3])
        assert p[0] == pytest.approx(ref, rel=1e-9, abs=0)
    big = np.column_stack([np.round(u * 1e6), np.round(y * 1e6), np.round(z * 1e6)])
    p = create_ci_test(name, big.astype(np.int64) + 2**62)(0, 1, [2])[0]
    assert 0.0 <= p < 1e-6


@pytest.mark.parametrize("name", ["raft", "rift"])
def test_large_offset_column_is_not_mistaken_for_a_copy(name):
    """A conditioning column whose offset has eaten most of its precision must not
    be merged with (and so erase) a genuine tested variable."""
    rng = np.random.default_rng(43)
    n = 300
    x = rng.standard_normal(n)
    z = 1e14 + rng.integers(0, 2, n).astype(float)
    d = np.column_stack([x, x + 0.1 * rng.standard_normal(n), z])
    assert create_ci_test(name, d)(0, 1, [2])[0] < 1e-10


def test_rift_orientation_stable_for_balanced_discrete_copies():
    rng = np.random.default_rng(41)  # a draw where rounding flips the old argmax
    n = 300
    zb = rng.permutation(np.r_[np.ones(n // 3), -np.ones(n // 3), np.zeros(n // 3)])
    w = rng.standard_normal(n)
    g = zb * w
    base = [g + 0.5 * rng.standard_normal(n), np.tanh(g) + 0.5 * rng.standard_normal(n)]
    d1 = np.column_stack(base + [1e6 + zb, w])
    d2 = np.column_stack(base + [1e5 + 0.1 * zb, w])
    assert create_ci_test("rift", d1)(0, 1, [2, 3])[0] == pytest.approx(
        create_ci_test("rift", d2)(0, 1, [2, 3])[0], rel=1e-6
    )


@pytest.mark.parametrize("name", ["raft", "rift"])
def test_subnormal_scale_column(name):
    rng = np.random.default_rng(45)
    n = 300
    u, z = rng.standard_normal((2, n))
    y = np.sin(2 * u) + 0.3 * z + 0.3 * rng.standard_normal(n)
    ref = create_ci_test(name, np.column_stack([u, y, z]))(0, 1, [2])[0]
    tiny = np.round(u * 30) * 5e-324  # a discrete column of deep-subnormal values
    p = create_ci_test(name, np.column_stack([tiny, y, z]))(0, 1, [2])[0]
    assert np.isfinite(p) and p < 1e-6 and ref < 1e-6


def test_infinite_value_in_a_nan_row_still_raises():
    d = _null_data(100, np.random.default_rng(46))
    d[3, 1], d[3, 2] = np.inf, np.nan
    with pytest.raises(ValueError):
        create_ci_test("raft", d)(0, 1, [2])


@pytest.mark.parametrize("name", ["raft", "rift"])
def test_offset_binary_confounder_is_not_constant(name):
    """1e15 + {0, 1} is a represented binary variable, not a constant: conditioning
    on it must remove the dependence it induces."""
    rng = np.random.default_rng(47)
    n = 400
    b = rng.integers(0, 2, n).astype(float)
    d = np.column_stack(
        [b + 0.3 * rng.standard_normal(n), b + 0.3 * rng.standard_normal(n), 1e15 + b]
    )
    assert create_ci_test(name, d)(0, 1, [])[0] < 1e-10
    p = create_ci_test(name, d)(0, 1, [2])[0]
    assert p == pytest.approx(
        create_ci_test(name, np.column_stack([d[:, :2], b]))(0, 1, [2])[0], rel=1e-6
    )


@pytest.mark.parametrize("name", ["raft", "rift"])
def test_near_copy_of_z_is_not_erased(name):
    rng = np.random.default_rng(48)
    n = 300
    u, v = rng.standard_normal((2, n))
    z = 1e8 + u
    x = z + 2e-7 * v  # resolvably different from z
    y = v + 0.1 * rng.standard_normal(n)  # depends on x only through x - z
    assert create_ci_test(name, np.column_stack([x, y, z]))(0, 1, [2])[0] < 1e-10


def test_rift_orientation_survives_offset_with_near_tied_extremes():
    rng = np.random.default_rng(49)
    n = 300
    u = np.clip(rng.standard_normal(n // 2), -2.5, 2.5)
    z = np.r_[u, -u]  # mean zero, so standardizing keeps the two extremes tied
    z[0], z[n // 2] = 3.0, -3.0 - 1e-8  # tied to within a few offset ulps
    w, v = rng.standard_normal((2, n))
    g = z * w * v
    base = [g + 0.5 * rng.standard_normal(n), np.tanh(g) + 0.5 * rng.standard_normal(n)]
    d1 = np.column_stack(base + [z, w, v])
    d2 = np.column_stack(base + [1e7 + z, w, v])
    assert create_ci_test("rift", d1)(0, 1, [2, 3, 4])[0] == pytest.approx(
        create_ci_test("rift", d2)(0, 1, [2, 3, 4])[0], rel=1e-4
    )


@pytest.mark.parametrize("name", ["raft", "rift"])
@pytest.mark.parametrize("a, b", [(-1.0, 0.0), (-2.0, 3.0), (0.5, 7.0)])
def test_affine_copy_of_z_as_x_gives_p_one(name, a, b):
    """X = a Z + b is trivially independent of Y given Z: its features must add
    nothing beyond Z's, whatever the sign of a."""
    rng = np.random.default_rng(51)
    n = 400
    z = rng.standard_normal(n)
    y = np.sin(4 * z) + 0.3 * rng.standard_normal(n)
    assert create_ci_test(name, np.column_stack([a * z + b, y, z]))(0, 1, [2])[0] == 1.0


@pytest.mark.parametrize("f", [4.0, 6.5])
def test_rift_absorbs_wiggly_marginal_effect(f):
    """A sin(f z) effect shared by X and Y is beyond raft's additive features (with
    z_ridge off it rejects the true null almost always); rift's frequency grid
    absorbs it, also off the grid (6.5)."""
    rng = np.random.default_rng(52)
    rej = {"raft": 0, "rift": 0}
    for _ in range(100):
        n = 500
        z = rng.standard_normal(n)
        g = np.sin(f * z)
        d = np.column_stack(
            [
                g + 0.5 * rng.standard_normal(n),
                np.tanh(2 * g) + 0.5 * rng.standard_normal(n),
                z,
            ]
        )
        for t in rej:
            kw = ROBUST if t == "raft" else {}
            rej[t] += create_ci_test(t, d, **kw)(0, 1, [2])[0] <= 0.05
    assert rej["raft"] >= 80
    assert rej["rift"] <= 15


def test_statistic_comes_from_the_deciding_direction():
    from causalts.ci_tests.raft import RAFTCI

    class Fixed(RAFTCI):
        def _one_direction(self, y, FX, Q, mult, key=None):
            return self._answers.pop(0)

    ci = Fixed(
        _null_data(200, np.random.default_rng(53)), combine="bonferroni", **ROBUST
    )
    ci._answers = [(0.5, 10.0), (0.01, 3.0)]
    p, stat = ci(0, 1, [2])
    assert p == pytest.approx(0.02) and stat == 3.0
    # Cauchy combination (raft's default): same deciding statistic
    ci = Fixed(_null_data(200, np.random.default_rng(53)), **ROBUST)
    ci._answers = [(0.5, 10.0), (0.01, 3.0)]
    p, stat = ci(0, 1, [2])
    t = (np.tan(0.0) + np.tan((0.5 - 0.01) * np.pi)) / 2
    assert p == pytest.approx(0.5 - np.arctan(t) / np.pi) and stat == 3.0
    # A degenerate p = 1 direction must not veto a significant one
    ci = Fixed(_null_data(200, np.random.default_rng(53)), **ROBUST)
    ci._answers = [(1.0, 3.0), (1e-8, 40.0)]
    p, stat = ci(0, 1, [2])
    assert p == pytest.approx(1e-8, rel=1e-9) and stat == 40.0


def test_exact_copy_of_z_as_x_at_large_t():
    """At large T the QR residue of an exact copy must stay below the precision
    floor (variance="f" has no bootstrap to hide it)."""
    rng = np.random.default_rng(3)
    n = 100_000
    for zgen in (
        lambda: 3 * rng.standard_normal(n) + 5,
        lambda: rng.uniform(-1, 1, n),
        lambda: rng.standard_t(3, n),
    ):
        z = zgen()
        for a, b in ((-1.0, 0.0), (-3.0, 2.0)):
            d = np.column_stack([a * z + b, rng.standard_normal(n), z])
            assert create_ci_test("raft", d, variance="f")(0, 1, [2])[0] == 1.0


def test_rift_dedupes_exact_copies_of_coarse_columns():
    """Exact duplicates are copies even when an offset leaves the column too coarse
    for the rounding-based check."""
    rng = np.random.default_rng(55)
    n = 400
    zi = rng.integers(0, 101, n).astype(float)
    g = np.sin(4 * zi / 30.0)
    z = 1e15 + zi
    d = np.column_stack(
        [
            g + 0.5 * rng.standard_normal(n),
            np.tanh(2 * g) + 0.5 * rng.standard_normal(n),
            z,
            z.copy(),
        ]
    )
    assert create_ci_test("rift", d)(0, 1, [2, 3])[0] == pytest.approx(
        create_ci_test("rift", d)(0, 1, [2])[0]
    )


def test_infinite_values_raise():
    d = _null_data(100, np.random.default_rng(42))
    d[3, 1] = np.inf
    with pytest.raises(ValueError):
        create_ci_test("raft", d)(0, 1, [2])


def test_vector_order_invariant():
    d = _shared_interaction_null(300, np.random.default_rng(17))
    for name in ("raft", "rift"):
        ref = create_ci_test(name, d)([0, 2], 1, [3, 4])
        assert create_ci_test(name, d)([2, 0], 1, [4, 3]) == ref


def test_rift_ignores_redundant_conditioning_columns():
    """Constant columns and affine copies leave the conditioning sigma-field unchanged,
    so they must not switch the interaction dictionary on, and which copy is kept
    must not matter."""
    rng = np.random.default_rng(18)
    n = 400
    z = rng.standard_normal(n)
    w = rng.standard_normal(n)
    g = np.sin(12 * z)
    d = np.column_stack(
        [
            g + 0.65 * rng.standard_normal(n),
            np.tanh(2 * g) + 0.3 * w + 0.65 * rng.standard_normal(n),
            z,
            z.copy(),
            2.0 + 3.0 * z,
            np.ones(n),
            2.0 - 3.0 * z,
            w,
            1e6 + 3.0 * z,
        ]
    )
    ref = create_ci_test("rift", d)(0, 1, [2])
    for extra in ([3], [4], [5], [6], [8], [3, 4, 5, 6]):
        assert create_ci_test("rift", d)(0, 1, [2] + extra) == pytest.approx(ref)
    for extra in ([3], [6]):  # raft too: sine/cosine features are sign-symmetric
        assert create_ci_test("raft", d)(0, 1, [2] + extra) == pytest.approx(
            create_ci_test("raft", d)(0, 1, [2])
        )
    # With a genuine second variable, the dictionary's representative of {z, -z}
    # does not change the result.
    swapped = d[:, [0, 1, 6, 3, 4, 5, 2, 7, 8]]
    ref = create_ci_test("rift", d)(0, 1, [2, 6, 7])[0]
    assert create_ci_test("rift", swapped)(0, 1, [2, 6, 7])[0] == pytest.approx(
        ref, rel=1e-6
    )
    # The kept copy and the column order do not change the dictionary.
    reordered = d[:, [0, 1, 7, 3, 4, 5, 6, 2, 8]]  # w now precedes z and -z
    assert create_ci_test("rift", reordered)(0, 1, [2, 6, 7])[0] == pytest.approx(
        ref, rel=1e-6
    )
    assert create_ci_test("rift", d)(0, 1, [7, 8])[0] == pytest.approx(
        create_ci_test("rift", d)(0, 1, [2, 7])[0], rel=1e-3
    )


def test_rift_discrete_conditioning_variables():
    rng = np.random.default_rng(19)
    n = 300
    z1 = rng.integers(0, 2, n).astype(float)
    z2 = rng.integers(0, 3, n).astype(float)
    z3 = rng.standard_normal(n)
    g = z1 * z3 + z2
    d = np.column_stack(
        [g + 0.5 * rng.standard_normal(n), g + 0.5 * rng.standard_normal(n), z1, z2, z3]
    )
    p, stat = create_ci_test("rift", d)(0, 1, [2, 3, 4])
    assert np.isfinite(p) and np.isfinite(stat) and 0.0 <= p <= 1.0


@pytest.mark.parametrize("name", ["raft", "rift"])
def test_too_few_rows_returns_one(name):
    for n in (0, 5):
        with pytest.warns(RuntimeWarning):
            assert create_ci_test(name, np.ones((n, 5)))(0, 1, [2, 3])[0] == 1.0


def test_rift_saturated_conditioning_set_returns_one():
    d = np.random.default_rng(22).standard_normal((40, 8))
    ci = create_ci_test("rift", d, **LEGACY)  # z_ridge adds one column per variable
    with pytest.warns(RuntimeWarning, match="use up the sample"):
        assert ci(0, 1, list(range(2, 8)))[0] == 1.0
    assert ci.n_saturated >= 1


def test_rift_partial_saturation_keeps_every_bonferroni_share():
    """The ridge skip depends on the response, so it must not shrink Bonferroni."""
    d = _shared_interaction_null(300, np.random.default_rng(26))
    d[:, 0] += 0.25 * np.tanh(d[:, 1])  # weak dependence, so p is not capped at 1
    ci = create_ci_test("rift", d, combine="bonferroni", linear_channel=False)
    seen = []
    ridge = ci._ridge_direction

    def record(y, *args):
        if not seen:  # skip the first direction as if its GCV fit used up the rows
            ci._saturated(len(y))
            out = (1.0, 0.0)
        else:
            out = ridge(y, *args)
        seen.append(out[0])
        return out

    ci._ridge_direction = record
    with pytest.warns(RuntimeWarning, match="remaining"):
        p, _ = ci(0, [1, 2], [3, 4])
    assert len(seen) == 3 and ci.n_saturated == 1
    assert 3 * min(seen) < 1.0
    assert p == pytest.approx(3 * min(seen), rel=1e-9, abs=0)


def test_nan_in_unused_column_matches_clean_data():
    d = _shared_interaction_null(300, np.random.default_rng(23))
    d_nan = np.column_stack([d, np.full(300, np.nan)])
    for name in ("raft", "rift"):
        assert create_ci_test(name, d_nan)(0, 1, [2, 3, 4]) == create_ci_test(name, d)(
            0, 1, [2, 3, 4]
        )


def test_nan_rows_use_original_timeline_multipliers():
    d = _shared_interaction_null(300, np.random.default_rng(24))
    d[::2, 0] = np.nan
    ci = create_ci_test("raft", d)
    p, _ = ci(0, 1, [2, 3, 4])
    assert 0.0 <= p <= 1.0
    assert list(ci._xi_cache._d) == [300]  # drawn for the full series, then subset


def test_data_swap_across_shapes_does_not_reuse_stale_pvalue():
    """Equal column bytes in a different shape must not hit the p-value cache."""
    rng = np.random.default_rng(25)
    a = rng.standard_normal((30, 4))
    b = np.column_stack([a[:, [0, 1]].ravel(), a[:, [2, 3]].ravel()])
    b[:, 1] += 2.0 * np.sin(2 * b[:, 0])
    a = np.column_stack([b[:, 0].reshape(30, 2), b[:, 1].reshape(30, 2)])
    for name in ("raft", "rift"):
        ci = create_ci_test(name, a)
        ci([0, 1], [2, 3], [])
        ci.data = b
        assert ci(0, 1, []) == create_ci_test(name, b)(0, 1, [])


def test_set_data_clears_pvalue_cache():
    rng = np.random.default_rng(26)
    ci = create_ci_test("raft", _null_data(200, rng))
    ci(0, 1, [2])
    ci.set_data(_null_data(200, rng), data_hash="new")
    assert ci.data_hash == "new" and not ci.pvalue_cache


def test_param_hash_includes_min_samples():
    d = _null_data(100, np.random.default_rng(31))
    assert (
        create_ci_test("raft", d, min_samples=30)._param_hash()
        != create_ci_test("raft", d, min_samples=50)._param_hash()
    )


def test_cached_basis_owns_its_memory():
    rng = np.random.default_rng(32)
    z = rng.standard_normal(200)
    d = np.column_stack([rng.standard_normal((200, 2)), z, z, 2 * z])
    Qa, _, _ = create_ci_test("raft", d)._z_parts([2, 3, 4], d)
    assert Qa.base is None or Qa.base.nbytes == Qa.nbytes


def test_param_hash_keeps_float_precision():
    d = _null_data(100, np.random.default_rng(27))
    a = create_ci_test("rift", d, w_tol=0.9090908636363636)._param_hash()
    b = create_ci_test("rift", d, w_tol=0.9090909545454545)._param_hash()
    assert a != b


@pytest.mark.parametrize("name", ["raft", "rift"])
def test_disk_cache_round_trip(name, tmp_path):
    d = _shared_interaction_null(200, np.random.default_rng(28))
    ci = create_ci_test(name, d, cache_dir=str(tmp_path))
    ref = ci(0, 1, [2, 3, 4])
    ci.flush_cache()
    ci2 = create_ci_test(name, d, cache_dir=str(tmp_path))
    assert ci2(0, 1, [2, 3, 4]) == pytest.approx(ref)
    assert ci2.n_actual_tests == 0


def test_lru_keeps_latest_oversized_entry():
    from causalts.ci_tests.raft import _LRU

    c = _LRU(max_items=4, max_bytes=100)
    c.put("a", 1, 10)
    c.put("big", 2, 1000)  # larger than the whole bound: kept, older entries evicted
    assert c.get("big") == (True, 2) and c.get("a") == (False, None)
    c.put("b", 3, 10)
    assert c.get("b") == (True, 3) and c.get("big") == (False, None)


def test_multipliers_cached_beyond_byte_bound():
    d = _null_data(300, np.random.default_rng(50))
    ci = create_ci_test("raft", d)
    ci._xi_cache.max_bytes = 1024  # far below one 300 x 199 matrix
    ci(0, 1, [2])
    first = ci._multipliers(300)
    assert ci._multipliers(300) is first  # reused, not regenerated per test


def test_seed_keyword_sets_the_draws():
    d = _shared_interaction_null(300, np.random.default_rng(56))
    for name in ("raft", "rift"):
        default = create_ci_test(name, d)(0, 1, [2, 3, 4])
        assert create_ci_test(name, d, rff_seed=12345)(0, 1, [2, 3, 4]) == default
        p7 = create_ci_test(name, d, seed=7)(0, 1, [2, 3, 4])
        assert p7 == create_ci_test(name, d, rff_seed=7)(0, 1, [2, 3, 4])
        assert p7 != default
        # an explicit rff_seed wins over seed
        assert (
            create_ci_test(name, d, seed=7, rff_seed=12345)(0, 1, [2, 3, 4]) == default
        )


def test_seed_none_means_default():
    d = _null_data(200, np.random.default_rng(57))
    assert create_ci_test("raft", d, seed=None)(0, 1, [2]) == create_ci_test("raft", d)(
        0, 1, [2]
    )


@pytest.mark.parametrize("name", ["raft", "rift"])
def test_duplicated_tested_component_is_counted_once(name):
    d = _null_data(300, np.random.default_rng(58))
    d[:, 1] += 0.25 * np.sin(2 * d[:, 0])  # weak dependence: p well below the cap
    d = np.column_stack([d, d[:, 0]])
    ref = create_ci_test(name, d)(0, 1, [2])
    assert ref[0] < 0.3
    assert create_ci_test(name, d)([0, 3], 1, [2]) == ref


def test_partial_saturation_is_counted_per_call_and_worded():
    """n = 26, |Z| = 1, vector X of two: Y's regression on X's two feature blocks
    has too few residual degrees of freedom and is skipped, while each X component's
    regression on Y's block fits. The call counts once, and the warning says the
    p-value comes from the remaining directions."""
    rng = np.random.default_rng(59)
    n = 26
    z = rng.standard_normal(n)
    d = np.column_stack([rng.standard_normal((n, 2)), z + rng.standard_normal(n), z])
    for combine in ("bonferroni", "cauchy"):
        # n_rff_tested=8: X's two full feature blocks saturate Y's regression
        ci = create_ci_test("raft", d, combine=combine, n_rff_tested=8, **ROBUST)
        seen = []
        one = ci._one_direction

        def record(*args):
            before = ci._sat_dirs
            out = one(*args)
            seen.append((out[0], ci._sat_dirs > before))
            return out

        ci._one_direction = record
        with pytest.warns(RuntimeWarning, match="remaining") as rec:
            p, _ = ci([0, 1], 2, [3])
        assert len(rec) == 1 and ci.n_saturated == 1 and p < 1.0
        # Combined over the directions actually tested; the skipped one is left out
        tested = np.array([q for q, skipped in seen if not skipped])
        assert len(tested) == len(seen) - 1 == 2
        if combine == "bonferroni":
            want = min(1.0, len(tested) * tested.min())
        else:
            want = 0.5 - np.arctan(np.mean(np.tan((0.5 - tested) * np.pi))) / np.pi
        assert p == pytest.approx(want, rel=1e-9, abs=0)


@pytest.mark.parametrize("name", ["raft", "rift"])
def test_offset_tested_variable_keeps_its_dependence(name):
    rng = np.random.default_rng(60)
    n = 300
    b = rng.integers(0, 2, n).astype(float)
    z = rng.standard_normal(n)
    y = b + 0.3 * rng.standard_normal(n)
    assert create_ci_test(name, np.column_stack([1e15 + b, y, z]))(0, 1, [2])[0] < 1e-10


def test_signed_zero_duplicates_are_one_component():
    d = _null_data(300, np.random.default_rng(61))
    d[:, 1] += 0.15 * np.sin(2 * d[:, 0])
    x = d[:, 0].copy()
    x[:5] = 0.0
    neg = x.copy()
    neg[:5] = -0.0
    d = np.column_stack([x, d[:, 1], d[:, 2], neg])
    assert create_ci_test("raft", d)([0, 3], 1, [2]) == create_ci_test("raft", d)(
        0, 1, [2]
    )


def test_saturated_results_are_not_cached():
    d = np.random.default_rng(62).standard_normal((40, 8))
    ci = create_ci_test("raft", d, **ROBUST)
    with pytest.warns(RuntimeWarning):
        ci(0, 1, list(range(2, 8)))
    ci(0, 1, list(range(2, 8)))  # recomputed, so counted again (warned once)
    assert ci.n_saturated == 2 and not ci.pvalue_cache


def test_too_few_rows_has_its_own_warning():
    ci = create_ci_test(
        "raft", np.random.default_rng(63).standard_normal((40, 8)), **ROBUST
    )
    ci.data = ci.data[:8]
    with pytest.warns(RuntimeWarning, match="usable rows"):
        ci(0, 1, [2])
    ci.data = np.random.default_rng(63).standard_normal((40, 8))
    with pytest.warns(RuntimeWarning, match="reads as independence"):  # not suppressed
        ci(0, 1, list(range(2, 8)))


@pytest.mark.parametrize("bad", [0.9, "0", False, [0.0]])
def test_column_indices_must_be_integers(bad):
    ci = create_ci_test("raft", _null_data(100, np.random.default_rng(64)))
    with pytest.raises(ValueError):
        ci(bad, 1, [2])


def test_null_pvalue_filtering_and_fallback():
    ci = create_ci_test("raft", _null_data(100, np.random.default_rng(29)))
    assert ci._null_pvalue(1.0, np.array([np.nan, np.inf, 1.0])) == 1.0
    assert ci._null_pvalue(2.0, np.ones(10)) == pytest.approx(1.0 / 11.0)


def test_gcv_can_select_no_interaction():
    d = _shared_interaction_null(300, np.random.default_rng(30))
    ci = create_ci_test("rift", d)
    Qa, U, s = ci._z_parts([2, 3, 4], d)
    assert s.size and np.allclose(U.T @ U, np.eye(s.size), atol=1e-10)
    # A response with nothing in the interaction directions: any ridge fit only
    # spends degrees of freedom, so GCV must pick lam = inf (all weights zero).
    e = np.random.default_rng(0).standard_normal(len(d))
    v = e - Qa @ (Qa.T @ e) - U @ (U.T @ e)
    assert not np.any(ci._ridge_weights(v, Qa, U, s))


# Regression locks: values of this release, on canonical (sorted, unique) conditioning
# sets.
_LOCKED = {
    "raft": [1.697308966278816e-05, 0.00036770041851908196, 0.0030895338146459143, 0.0022920523743366387, 0.0034698455898419618],
    "f": [1.2531535166324153e-09, 2.649090580006296e-06, 2.096200347739123e-05, 0.0002037208095403574, 0.0002542037943727564],
    "rift": [5.206776233446231e-06, 2.3658273072268976e-06, 1.0894586058296518e-06, 5.704029067965521e-07, 4.656142712278211e-06],
    "rift_legacy": [1.697308966278816e-05, 0.00036770041851908743, 4.161142155733533e-05, 1.278569902582873e-06, 2.6561759884062526e-05],
    "raft_default": [5.206776233446231e-06, 2.0612085332438745e-06, 1.0997232905984866e-05, 0.00011303124703077039, 1.7150406634081072e-05],
    "raft_robust": [2.1504603271771394e-05, 0.00020273239530721465, 0.0033596026329216766, 0.0024687080166104214, 0.005068508440093766],
}  # fmt: skip


@pytest.mark.parametrize("key", list(_LOCKED))
def test_locked_reference_values(key):
    rng = np.random.default_rng(0)
    n = 300
    Z = rng.standard_normal((n, 4))
    x = np.sin(Z[:, 0]) + Z[:, 1] * Z[:, 2] + 0.5 * rng.standard_normal(n)
    y = np.tanh(Z[:, 0] * Z[:, 1]) + 0.3 * x + 0.5 * rng.standard_normal(n)
    d = np.column_stack([x, y, Z])
    name, kw = {
        "raft": ("raft", LEGACY),
        "f": ("raft", {"variance": "f"}),
        "rift": ("rift", {}),
        "rift_legacy": ("rift", LEGACY),
        "raft_default": ("raft", {}),
        "raft_robust": ("raft", ROBUST),
    }[key]
    got = [
        create_ci_test(name, d, **kw)(0, 1, list(range(2, 2 + k)))[0] for k in range(5)
    ]
    assert got == pytest.approx(_LOCKED[key], rel=1e-6)


def test_cli_guide_lists_both_names():
    out = CliRunner().invoke(main, ["ci-test-info"]).output
    assert "raft" in out and "rift" in out


def test_bad_adjust_and_incompatible_options_raise():
    d = np.zeros((50, 3))
    with pytest.raises(ValueError):
        create_ci_test("raft", d, adjust="nope")
    with pytest.raises(ValueError):
        create_ci_test("rift", d, variance="f")
    with pytest.raises(ValueError):
        create_ci_test("raft", d, n_boot=1)
    bad = [
        {"n_knots": 1},
        {"n_triple": -1},
        {"w_tol": float("nan")},
        {"w_tol": 1.0},
        {"bandwidth": 0.0},
        {"n_rff": -1},
        {"max_pair": 0},
        {"n_rff": 7},
        {"n_rff": 2.5},
        {"n_rff": "2"},
        {"n_boot": 10.0},
        {"n_knots": True},
        {"freqs": (1.0, -2.0)},
        {"freqs": (float("nan"),)},
    ]
    for kw in bad:
        with pytest.raises(ValueError):
            create_ci_test("rift", d, **kw)
    create_ci_test("raft", d, variance="f", n_boot=1)  # n_boot unused by "f"


# ----- Huber residual, tested-side feature count, Cauchy combination ---------------


def test_robust_options_resolve_per_method_and_validate():
    d = _null_data(200, np.random.default_rng(70))
    ci = create_ci_test("raft", d)
    assert (ci.robust, ci.n_rff_tested, ci.combine) == ("huber", "auto", "cauchy")
    assert (ci.z_ridge, ci.hc2, ci.linear_channel) == (True, True, 512)
    ci = create_ci_test("rift", d)
    assert (ci.robust, ci.n_rff_tested, ci.combine) == ("huber", "auto", "cauchy")
    assert (ci.z_ridge, ci.hc2, ci.linear_channel) == (True, True, 512)
    ci = create_ci_test("raft", d, variance="f")
    assert (ci.robust, ci.n_rff_tested, ci.combine) == (False, 8, "bonferroni")
    assert (ci.z_ridge, ci.hc2, ci.linear_channel) == (False, False, False)
    # earlier settings keep their old cache keys, so old disk caches stay valid
    legacy = create_ci_test("raft", d, **LEGACY)._param_hash()
    assert legacy == "raft-m8-bw1.0-vdwb-s12345-b199-ms30-v4"
    robust = create_ci_test("raft", d, **ROBUST)._param_hash()
    assert robust == "raft-m8-bw1.0-vdwb-s12345-b199-rhuber-mtauto-ccauchy-ms30-v4"
    assert create_ci_test("raft", d)._param_hash() != legacy
    for kw in (
        {"robust": "tukey"},
        {"robust": True},
        {"n_rff_tested": 3},
        {"n_rff_tested": 10},
        {"n_rff_tested": True},
        {"n_rff_tested": "half"},
        {"combine": "fisher"},
    ):
        with pytest.raises(ValueError):
            create_ci_test("raft", d, **kw)
    with pytest.raises(ValueError):
        create_ci_test("raft", d, variance="f", robust="huber")


def test_tested_feature_count_auto_and_conditioning_side_unchanged():
    d = _null_data(300, np.random.default_rng(71))
    ci = create_ci_test("raft", d)
    assert [ci._tested_rff(n) for n in (30, 50, 100, 149, 200, 300, 400, 5000)] == [
        2,
        2,
        2,
        2,
        4,
        6,
        8,
        8,
    ]
    assert create_ci_test("raft", d, n_rff=4)._tested_rff(1000) == 4
    assert create_ci_test("raft", d, n_rff_tested=2)._tested_rff(1000) == 2
    # the conditioning variables keep n_rff whatever the tested side uses
    a = create_ci_test("raft", d, n_rff_tested=2, **ROBUST)._z_parts([2], d)[0]
    b = create_ci_test("raft", d, **LEGACY)._z_parts([2], d)[0]
    assert a.shape == b.shape == (300, 1 + 9)  # intercept + one full block
    # n_rff_tested=n_rff is the plain test
    kw = {"robust": False, "combine": "bonferroni", **ROBUST}
    assert create_ci_test("raft", d, n_rff_tested=8, **kw)(0, 1, [2]) == create_ci_test(
        "raft", d, **LEGACY
    )(0, 1, [2])


def test_feature_cache_keeps_tested_and_conditioning_blocks_apart():
    """The same column as a tested variable (fewer features) and as a conditioning
    variable (n_rff features) must not share a cache entry."""
    d = _shared_interaction_null(300, np.random.default_rng(72))
    ci = create_ci_test("raft", d)
    got = [ci(0, 1, [2]), ci(2, 1, [0]), ci(0, 2, [1]), ci(1, 0, [2])]
    want = [
        create_ci_test("raft", d)(0, 1, [2]),
        create_ci_test("raft", d)(2, 1, [0]),
        create_ci_test("raft", d)(0, 2, [1]),
        create_ci_test("raft", d)(1, 0, [2]),
    ]
    assert got == want


def test_huber_residual_is_clipped_and_scale_free():
    from causalts.ci_tests.raft import _huber

    r = np.array([-100.0, -1.0, 0.0, 0.5, 1.0, 2.0, 50.0])
    s = np.median(np.abs(r - np.median(r))) / 0.6745
    assert np.array_equal(_huber(r), np.clip(r, -1.345 * s, 1.345 * s))
    assert np.array_equal(_huber(np.ones(5)), np.ones(5))  # zero MAD: unchanged
    # Clipping at a MAD multiple is scale-equivariant, so rescaling or shifting a
    # column leaves the test unchanged.
    d = _null_data(300, np.random.default_rng(73))
    d[:, 1] += 0.3 * np.sin(2 * d[:, 0])
    e = d.copy()
    e[:, 1] = 1e3 * e[:, 1] + 7.0
    e[:, 0] = -0.01 * e[:, 0]
    p, q = create_ci_test("raft", d)(0, 1, [2]), create_ci_test("raft", e)(0, 1, [2])
    assert p[0] == pytest.approx(q[0], rel=1e-6)


def test_cauchy_uses_only_directions_actually_tested():
    d = _null_data(300, np.random.default_rng(74))
    d[:, 1] += 0.3 * np.sin(2 * d[:, 0])
    # the additive path (with z_ridge every direction keeps a Bonferroni share,
    # as rift's do; see the rift tests below)
    ref = create_ci_test("raft", d, **ROBUST)(0, 1, [2])
    assert ref[0] < 0.1
    # An affine copy of Z as an extra X component has nothing to test: it must not
    # pull the Cauchy combination towards 1 (nor cost a share).
    e = np.column_stack([d, -3.0 * d[:, 2] + 5.0])
    assert create_ci_test("raft", e, **ROBUST)([0, 3], 1, [2])[0] == pytest.approx(
        ref[0], rel=1e-9
    )
    # Nothing tested at all: exactly 1.
    z = np.random.default_rng(75).standard_normal(300)
    f = np.column_stack([2.0 * z + 1.0, np.sin(4 * z), z])
    assert create_ci_test("raft", f, **ROBUST)(0, 1, [2])[0] == 1.0


def test_rift_cauchy_falls_back_to_bonferroni_on_a_skip():
    """With the ridge dictionary a skip depends on the response, so the Cauchy
    combination must not drop it: every direction keeps a Bonferroni share."""
    d = _shared_interaction_null(300, np.random.default_rng(26))
    d[:, 0] += 0.25 * np.tanh(d[:, 1])
    ci = create_ci_test("rift", d, combine="cauchy", linear_channel=False)
    seen = []
    ridge = ci._ridge_direction

    def record(y, *args):
        if not seen:
            ci._saturated(len(y))
            out = (1.0, 0.0)
        else:
            out = ridge(y, *args)
        seen.append(out[0])
        return out

    ci._ridge_direction = record
    with pytest.warns(RuntimeWarning, match="remaining"):
        p, _ = ci(0, [1, 2], [3, 4])
    assert len(seen) == 3 and 3 * min(seen) < 1.0
    assert p == pytest.approx(3 * min(seen), rel=1e-9, abs=0)


def test_rift_cauchy_falls_back_when_a_direction_is_explained():
    """With the ridge dictionary an explained response is judged after its own GCV
    fit, so leaving it out must send Cauchy back to Bonferroni over every
    direction, exactly as combine="bonferroni" does."""
    d = _shared_interaction_null(300, np.random.default_rng(76))
    d[:, 1] += 0.3 * np.tanh(d[:, 0])
    e = np.column_stack([d, -2.0 * d[:, 3] + 1.0])  # X component = affine copy of z
    a = create_ci_test("rift", e, combine="cauchy")([0, 5], 1, [2, 3, 4])
    b = create_ci_test("rift", e, combine="bonferroni")([0, 5], 1, [2, 3, 4])
    assert a == b


# ----- z_ridge, hc2, linear_channel ---------------------------------------------

# an earlier candidate (linear channel up to 250 rows); the default uses 512
R1 = {"z_ridge": True, "hc2": True, "linear_channel": 250}


def test_ridge_options_validate_and_hash():
    d = _null_data(200, np.random.default_rng(80))
    base = create_ci_test("raft", d)._param_hash()
    assert (
        create_ci_test(
            "raft", d, z_ridge=True, hc2=True, linear_channel=512
        )._param_hash()
        == base
    )
    hashes = {
        create_ci_test("raft", d, **{**ROBUST, **kw})._param_hash()
        for kw in (
            {},
            {"z_ridge": True},
            {"z_ridge": True, "z_ridge_under": 0.1},
            {"z_ridge": True, "z_ridge_rff": 12},
            {"hc2": True},
            {"linear_channel": 250},
            {"linear_channel": True},
        )
    }
    assert len(hashes) == 7
    for kw in (
        {"z_ridge": 1},
        {"z_ridge_rff": 7},
        {"z_ridge_rff": 0},
        {"z_ridge_under": 0.0},
        {"z_ridge_under": 2.0},
        {"hc2": "yes"},
        {"linear_channel": "always"},
        {"linear_channel": "auto"},
        {"cache_mb": 0},
        {"cache_mb": True},
        {"variance": "f", "hc2": True},
        {"variance": "f", "z_ridge": True},
    ):
        with pytest.raises(ValueError):
            create_ci_test("raft", d, **kw)


def test_z_ridge_parts_shape_and_cache():
    d = _shared_interaction_null(300, np.random.default_rng(81))
    ci = create_ci_test("raft", d, z_ridge=True)
    Qa, U, s = ci._z_parts([2, 3, 4], d)
    assert Qa.shape == (300, 4)  # intercept + three linear terms
    assert U.shape[1] == s.size > 0 and np.allclose(Qa.T @ U, 0, atol=1e-8)
    assert ci._z_parts([2, 3, 4], d)[0] is Qa  # cached by content
    assert create_ci_test("raft", d, z_ridge=True)(0, 1, [2, 3, 4]) == ci(
        0, 1, [2, 3, 4]
    )


def test_hc2_scales_bootstrap_by_leverage_only():
    """hc2 changes only the bootstrap: the statistic is unchanged."""
    d = _null_data(300, np.random.default_rng(82))
    a = create_ci_test("raft", d, **ROBUST)(0, 1, [2])
    b = create_ci_test("raft", d, **{**ROBUST, "hc2": True})(0, 1, [2])
    assert a[1] == b[1] and a[0] != b[0]


def test_linear_channel_is_off_above_its_cutoff():
    d = _null_data(400, np.random.default_rng(83))
    d[:, 1] += 0.1 * d[:, 0]
    plain = create_ci_test("raft", d, linear_channel=False)(0, 1, [2])
    assert create_ci_test("raft", d, linear_channel=250)(0, 1, [2]) == plain
    assert create_ci_test("raft", d)(0, 1, [2]) != plain  # default: up to 512 rows


def test_linear_channel_rescues_a_saturated_full_block():
    """When the full tested block saturates a direction, the linear-only block can
    still be tested; the direction is then not reported as skipped."""
    rng = np.random.default_rng(59)
    n = 26
    z = rng.standard_normal(n)
    d = np.column_stack([rng.standard_normal((n, 2)), z + rng.standard_normal(n), z])
    kw = {"n_rff_tested": 8, "combine": "bonferroni", "z_ridge": False, "hc2": False}
    with pytest.warns(RuntimeWarning):
        create_ci_test("raft", d, linear_channel=False, **kw)([0, 1], 2, [3])
    ci = create_ci_test("raft", d, linear_channel=True, **kw)
    p, _ = ci([0, 1], 2, [3])
    assert ci.n_saturated == 0 and 0.0 <= p < 1.0


def test_r1_null_rejection_near_alpha():
    rng = np.random.default_rng(84)
    ps = []
    for _ in range(100):
        d = _null_data(150, rng)
        ps.append(create_ci_test("raft", d, **R1)(0, 1, [2])[0])
    assert np.mean(np.array(ps) <= 0.10) <= 0.2


def test_ridge_hash_covers_w_tol_and_numpy_bool_linear_channel():
    d = _null_data(300, np.random.default_rng(85))
    a = create_ci_test("raft", d, z_ridge=True)._param_hash()
    assert create_ci_test("raft", d, z_ridge=True, w_tol=1e-3)._param_hash() != a
    ci = create_ci_test("raft", d, linear_channel=np.bool_(True))
    assert ci.linear_channel is True
    assert ci(0, 1, [2]) == create_ci_test("raft", d, linear_channel=True)(0, 1, [2])


def test_response_caches_match_fresh_fits():
    # A response fit or basis shared across tests (and the linear channel) must give
    # exactly what a fresh instance computes for each test on its own.
    rng = np.random.default_rng(85)
    z = rng.standard_normal((120, 2))
    d = np.column_stack(
        [
            np.sin(z[:, 0]) + 0.5 * rng.standard_normal(120),
            np.tanh(z[:, 1]) + 0.5 * rng.standard_normal(120),
            z[:, 0] * z[:, 1] + 0.5 * rng.standard_normal(120),
            z,
        ]
    )
    queries = [(0, 1, [3, 4]), (0, 2, [3, 4]), (1, 2, [3, 4]), (2, 0, [3]), (1, 0, [3])]
    for kw in (R1, {}, dict(hc2=True)):  # ridge path, and the additive path
        shared = create_ci_test("raft", d, **kw)
        for x, y, zs in queries + [(0, 1, []), (2, 1, [])]:
            assert shared(x, y, zs) == create_ci_test("raft", d, **kw)(x, y, zs)


def test_svd_falls_back_when_gesdd_does_not_converge(monkeypatch):
    # Some BLAS builds fail to converge in gesdd on ill-conditioned input; the
    # gesvd retry must give the same test on both the additive and ridge paths.
    import causalts.ci_tests.raft as raft_mod

    rng = np.random.default_rng(88)
    d = _null_data(200, rng)
    want = [create_ci_test("raft", d, **kw)(0, 1, [2])[0] for kw in ({}, R1)]

    def fail(*a, **k):
        raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(raft_mod.np.linalg, "svd", fail)
    got = [create_ci_test("raft", d, **kw)(0, 1, [2])[0] for kw in ({}, R1)]
    assert got == pytest.approx(want, rel=1e-9, abs=0)


def test_linear_channel_int_cutoff():
    d = _null_data(300, np.random.default_rng(89))
    for bad in (1.0, 0, -5, "on", "auto"):
        with pytest.raises(ValueError):
            RAFTCI(d, linear_channel=bad)
    assert RAFTCI(d, linear_channel=np.int64(400)).linear_channel == 400
    assert RAFTCI(d, linear_channel=np.True_).linear_channel is True
    h = {RAFTCI(d, linear_channel=v)._param_hash() for v in (False, 400, 300, True)}
    assert len(h) == 4
    # inside the cutoff the channel runs, so 300 rows with cutoff 400 equals True;
    # outside it equals no channel
    p = {
        v: RAFTCI(d, n_boot=99, linear_channel=v)(0, 1, [2])
        for v in (True, False, 400, 299)
    }
    assert p[400] == p[True] and p[299] == p[False]
