# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""CausalLearnFisherZ: the reported statistic matches the p-value."""

import numpy as np
import pandas as pd
from scipy.stats import norm

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
