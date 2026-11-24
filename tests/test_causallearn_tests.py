# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""causal-learn adapters: the reported statistic matches the p-value."""

import math
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2, norm

from causalts.ci_tests import create_ci_test


def _offset_data(T=2000):
    # Z has a large nonzero mean, and x and y have nonzero intercepts, so a
    # regression on Z without a constant term leaves Z in the residuals.
    rng = np.random.default_rng(0)
    z = rng.normal(size=T) + 10
    x = -3 + 0.3 * z + 0.5 * rng.normal(size=T)
    y = 20 - 0.5 * z + 0.5 * rng.normal(size=T)
    return np.c_[x, y, z]


def test_fisherz_statistic_matches_parcorr_when_z_has_nonzero_mean():
    data = _offset_data()
    pval, stat = create_ci_test("fisherz", data)(0, 1, [2])
    _, parcorr_stat = create_ci_test("parcorr", data)(0, 1, [2])
    assert abs(stat - parcorr_stat) < 1e-6
    # The p-value comes straight from causal-learn and is unchanged.
    assert abs(pval - 0.3924609094314029) < 1e-9


def test_fisherz_statistic_is_the_partial_correlation_behind_its_pvalue():
    data = _offset_data()
    n = data.shape[0]
    pval, stat = create_ci_test("fisherz", data)(0, 1, [2])

    # Partial correlation from the precision matrix of the correlation
    # matrix, which is what causal-learn's Fisher-Z p-value is built on.
    prec = np.linalg.inv(np.corrcoef(data.T))
    r = -prec[0, 1] / np.sqrt(prec[0, 0] * prec[1, 1])
    assert abs(stat - r) < 1e-9

    # Fisher's z-transform of the reported statistic gives back the p-value.
    z = 0.5 * np.log((1 + stat) / (1 - stat)) * np.sqrt(n - 1 - 3)
    assert abs(pval - 2 * (1 - norm.cdf(abs(z)))) < 1e-9


def test_fisherz_statistic_unconditional_is_pearson_correlation():
    data = _offset_data()
    _, stat = create_ci_test("fisherz", data)(0, 1, [])
    assert abs(stat - np.corrcoef(data[:, 0], data[:, 1])[0, 1]) < 1e-12


def test_fisherz_ignores_a_statistic_cached_before_the_intercept_fix(tmp_path):
    from causalts.ci_tests.causallearn_tests import CausalLearnFisherZ

    data = _offset_data()
    fresh = CausalLearnFisherZ(data, cache_dir=str(tmp_path))
    key = fresh._get_array_hash(0, 1, [2])
    pd.DataFrame(
        [[key, "fisherz", 0.5, -0.49]], columns=["key", "param_hash", "pval", "stats"]
    ).to_csv(fresh.cache_file, index=False)

    _, stat = CausalLearnFisherZ(data, cache_dir=str(tmp_path))(0, 1, [2])
    _, parcorr_stat = create_ci_test("parcorr", data)(0, 1, [2])
    assert abs(stat - parcorr_stat) < 1e-6


def test_fisherz_statistic_does_not_depend_on_the_units_of_z():
    data = _offset_data()
    _, stat = create_ci_test("fisherz", data)(0, 1, [2])
    tiny = data.copy()
    tiny[:, 2] *= 1e-13
    _, tiny_stat = create_ci_test("fisherz", tiny)(0, 1, [2])
    assert abs(stat - tiny_stat) < 1e-6


def test_fisherz_statistic_with_large_scale_float32_z():
    data = _offset_data()
    _, stat = create_ci_test("fisherz", data)(0, 1, [2])
    big = data.astype(np.float32)
    big[:, 2] *= np.float32(1e18)
    _, big_stat = create_ci_test("fisherz", big)(0, 1, [2])
    assert abs(stat - big_stat) < 1e-3


# --- chisq / gsquared: the statistic is the plug-in CMI in nats -------------

CATEGORICAL = [("chisq", "chisq"), ("gsquared", "gsq")]


def _oracle_cmi(data, X, Y, Z):
    """Plug-in CMI and df from explicit per-stratum count tables."""
    strata = {}
    for row in data:
        strata.setdefault(tuple(row[list(Z)]), []).append((row[X], row[Y]))
    stat, df = 0.0, 0
    for pairs in strata.values():
        cells = Counter(pairs)
        nx = Counter(a for a, _ in pairs)
        ny = Counter(b for _, b in pairs)
        if len(nx) < 2 or len(ny) < 2:
            continue
        df += (len(nx) - 1) * (len(ny) - 1)
        for (a, b), c in cells.items():
            stat += c * math.log(c * len(pairs) / (nx[a] * ny[b]))
    return stat / len(data), df


def _rows(table, z=None):
    """Expand a count table into rows of (x, y[, z])."""
    rows = [[i, j] for (i, j), c in np.ndenumerate(np.asarray(table)) for _ in range(c)]
    return np.array([r + ([z] if z is not None else []) for r in rows])


def _direct_pvalue(cl_method, data, X, Y, Z):
    from causallearn.utils.cit import Chisq_or_Gsq

    return float(Chisq_or_Gsq(data, cl_method)(X, Y, list(Z)))


def _dependent(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 3, n)
    y = np.where(rng.random(n) < 0.6, x, rng.integers(0, 3, n))
    return np.c_[x, y, rng.integers(0, 2, n), rng.integers(0, 3, n)]


@pytest.mark.parametrize("method, cl_method", CATEGORICAL)
def test_categorical_statistic_exact_unconditional_value(method, cl_method):
    data = _rows([[30, 20], [20, 30]])
    pval, stat = create_ci_test(method, data)(0, 1, [])
    assert abs(stat - 0.0201355135506889) < 1e-12
    assert pval == _direct_pvalue(cl_method, data, 0, 1, [])


@pytest.mark.parametrize("method, cl_method", CATEGORICAL)
def test_categorical_statistic_pools_strata_over_all_rows(method, cl_method):
    # Stratum B (X constant) is degenerate but its 10 rows count in n = 30.
    data = np.r_[_rows([[10, 0], [0, 10]], z=0), _rows([[5, 5]], z=1)]
    pval, stat = create_ci_test(method, data)(0, 1, [2])
    assert abs(stat - 20 / 30 * math.log(2)) < 1e-12
    assert abs(stat - 0.462098120373297) < 1e-12
    assert pval == _direct_pvalue(cl_method, data, 0, 1, [2])


@pytest.mark.parametrize("method, cl_method", CATEGORICAL)
def test_categorical_statistic_matches_oracle_on_random_data(method, cl_method):
    labels = np.array([3.1, 3.5, 3.9])  # int() would merge all three
    mid = 0
    for seed in range(12):
        rng = np.random.default_rng(seed)
        data = labels[rng.integers(0, 3, (70, 4))]
        for Z in ([], [2], [2, 3]):
            pval, stat = create_ci_test(method, data)(0, 1, Z)
            expected, df = _oracle_cmi(data, 0, 1, Z)
            assert abs(stat - expected) < 1e-12
            assert pval == _direct_pvalue(cl_method, data, 0, 1, Z)
            if method == "gsquared":
                ref = chi2.sf(2 * len(data) * stat, df) if df > 0 else 1.0
                assert abs(pval - ref) < 1e-10
            mid += 0.05 < pval < 0.95
    assert mid >= 10  # p-values away from 0 and 1


@pytest.mark.parametrize("method, cl_method", CATEGORICAL)
@pytest.mark.parametrize(
    "case",
    ["single_row", "x_constant", "y_constant", "zero_cell", "all_degenerate"],
)
def test_categorical_statistic_degenerate_strata(method, cl_method, case):
    informative = _rows([[8, 2], [3, 7]], z=0)
    other = {
        "single_row": np.array([[1, 0, 1]]),
        "x_constant": _rows([[4, 6]], z=1),
        "y_constant": _rows([[4], [6]], z=1),
        # The empty (1, 1) cell has positive row and column margins: kept.
        "zero_cell": _rows([[3, 4], [5, 0]], z=1),
        "all_degenerate": _rows([[4, 6]], z=1),
    }[case]
    if case == "all_degenerate":
        informative = _rows([[4], [6]], z=0)
    data = np.r_[informative, other]
    pval, stat = create_ci_test(method, data)(0, 1, [2])
    expected, df = _oracle_cmi(data, 0, 1, [2])
    assert abs(stat - expected) < 1e-12
    assert pval == _direct_pvalue(cl_method, data, 0, 1, [2])
    if case == "zero_cell":
        assert df == 2
    elif case == "all_degenerate":
        assert (pval, stat) == (1.0, 0.0)
    else:
        assert df == 1
        # Only the informative stratum contributes, divided by all rows.
        alone, _ = _oracle_cmi(informative, 0, 1, [2])
        assert abs(stat - alone * len(informative) / len(data)) < 1e-12


@pytest.mark.parametrize("method, cl_method", CATEGORICAL)
def test_categorical_statistic_ignores_constant_conditioning_column(method, cl_method):
    data = _dependent()
    data[:, 2] = 7
    assert create_ci_test(method, data)(0, 1, [2]) == pytest.approx(
        create_ci_test(method, data)(0, 1, []), abs=1e-12
    )
    assert create_ci_test(method, data)(0, 1, [2, 3]) == pytest.approx(
        create_ci_test(method, data)(0, 1, [3]), abs=1e-12
    )


@pytest.mark.parametrize("method, cl_method", CATEGORICAL)
def test_categorical_statistic_is_monotone_in_dependence(method, cl_method):
    tables = [[[25, 25], [25, 25]], [[30, 20], [20, 30]], [[40, 10], [10, 40]]]
    results = [create_ci_test(method, _rows(t))(0, 1, []) for t in tables]
    pvals, stats = zip(*results)
    assert stats[0] == 0.0
    assert stats[0] < stats[1] < stats[2]
    assert pvals[0] > pvals[1] > pvals[2]


@pytest.mark.parametrize("method, cl_method", CATEGORICAL)
def test_categorical_statistic_null_scale(method, cl_method):
    ratios = []
    for seed in range(50):
        rng = np.random.default_rng(seed)
        x = rng.integers(0, 3, 6000)
        y = rng.integers(0, 3, 6000)
        z = rng.integers(0, 4, 6000)
        data = np.c_[x, y, z]
        _, stat = create_ci_test(method, data)(0, 1, [2])
        _, df = _oracle_cmi(data, 0, 1, [2])
        assert df == 16
        ratios.append(2 * len(data) * stat / df)
    assert 0.8 < np.mean(ratios) < 1.2


@pytest.mark.parametrize("method, cl_method", CATEGORICAL)
def test_categorical_statistic_invariances(method, cl_method):
    data = _dependent()

    def run(d, X=0, Y=1, Z=(2, 3)):
        return create_ci_test(method, d)(X, Y, Z)

    base = run(data)
    assert base[1] > 0
    assert run(data, 1, 0) == pytest.approx(base, abs=1e-12)
    perm = np.random.default_rng(1).permutation(len(data))
    assert run(data[perm]) == pytest.approx(base, abs=1e-12)
    assert run(data, Z=[3, 2]) == pytest.approx(base, abs=1e-12)
    assert run(data, Z=np.array([2, 3])) == pytest.approx(base, abs=1e-12)

    # Fractional labels that collide under int(), and integers that collide
    # as float64.
    frac = np.array([2.9, 2.1, 2.5])[data]
    assert run(frac) == pytest.approx(base, abs=1e-12)
    big = np.array([2**53 + 1, 2**53, 2**53 + 2], dtype=np.int64)[data]
    assert len(np.unique(big[:, 0].astype(np.float64))) < 3
    assert run(big) == pytest.approx(base, abs=1e-12)

    uncond = run(data, Z=[])
    assert uncond[1] > 0
    assert run(data, Z=None) == uncond
    assert run(data, Z=np.array([], dtype=int)) == uncond


@pytest.mark.parametrize("method, cl_method", CATEGORICAL)
def test_categorical_statistic_follows_data_reassignment(method, cl_method):
    a, b = _dependent(300, seed=0), _dependent(120, seed=1)
    fresh_a = create_ci_test(method, a)(0, 1, [2])
    fresh_b = create_ci_test(method, b)(0, 1, [2])
    assert fresh_a != fresh_b

    delayed = create_ci_test(method, None)
    delayed.set_data(b)
    assert delayed(0, 1, [2]) == pytest.approx(fresh_b, abs=1e-12)

    test = create_ci_test(method, a)
    assert test(0, 1, [2]) == pytest.approx(fresh_a, abs=1e-12)
    test.set_data(b)
    assert test(0, 1, [2]) == pytest.approx(fresh_b, abs=1e-12)
    test.data = a
    assert test(0, 1, [2]) == pytest.approx(fresh_a, abs=1e-12)


@pytest.mark.parametrize("method, cl_method", CATEGORICAL)
def test_categorical_statistic_cache(method, cl_method, tmp_path, monkeypatch):
    from causalts.ci_tests import causallearn_tests

    cls = {
        "chisq": causallearn_tests.CausalLearnChisq,
        "gsquared": causallearn_tests.CausalLearnGsq,
    }[method]
    data = _dependent()
    expected, _ = _oracle_cmi(data, 0, 1, [2])
    assert expected > 0.1

    first = cls(data, cache_dir=str(tmp_path))
    key = first._get_array_hash(0, 1, [2])
    old = [key, method, 0.25, 0.0]
    pd.DataFrame([old], columns=["key", "param_hash", "pval", "stats"]).to_csv(
        first.cache_file, index=False
    )

    test = cls(data, cache_dir=str(tmp_path))  # loads the pre-fix row
    pval, stat = test(0, 1, [2])
    assert abs(stat - expected) < 1e-12
    assert pval == _direct_pvalue(cl_method, data, 0, 1, [2])

    def fail(*args, **kwargs):
        raise AssertionError("recomputed a cached result")

    monkeypatch.setattr(causallearn_tests, "_plugin_cmi", fail)
    test._cl_test = fail
    assert test(0, 1, [2]) == (pval, stat)

    test.flush_cache()
    rows = pd.read_csv(test.cache_file)
    assert list(rows.columns) == ["key", "param_hash", "pval", "stats"]
    assert rows.iloc[0].tolist() == old
    new = rows[rows["param_hash"] == f"{method}-cmi"]
    assert len(new) == 1 and new["key"].iloc[0] == key
    assert new["pval"].iloc[0] == pytest.approx(pval, rel=1e-12, abs=1e-300)
    assert new["stats"].iloc[0] == pytest.approx(stat, rel=1e-12)

    reloaded = cls(data, cache_dir=str(tmp_path))
    monkeypatch.setattr(reloaded, "_ensure_test", fail)
    assert reloaded(0, 1, [2]) == pytest.approx((pval, stat), rel=1e-12, abs=1e-300)
