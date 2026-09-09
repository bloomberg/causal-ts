# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest
from causallearn.utils.cit import CIT

import causalts.ci_tests.causal_learn_bridge as bridge

# Excluded from the registry: both require constructor arguments with no
# defaults (discrete_cols, and for StratifiedCIT also a pre-built inner_cit),
# so they can't be selected by bare name the way the registered tests are.
EXCLUDED_FROM_REGISTRY = {"cmiknn_mixed_gpu", "stratified_cit"}


def _make_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    z = rng.standard_normal(n)
    y = 0.5 * x + 0.3 * z + 0.1 * rng.standard_normal(n)
    return np.column_stack([x, y, z])


@pytest.mark.parametrize("name", sorted(bridge._REGISTRY))
def test_registered_test_returns_pvalue(name):
    data = _make_data()
    cit = CIT(data, method=name, device="cpu")
    pval = cit(0, 1, [2])
    assert isinstance(pval, float)
    assert 0.0 <= pval <= 1.0
    assert cit.method == name


@pytest.mark.parametrize("name", sorted(EXCLUDED_FROM_REGISTRY))
def test_tests_requiring_extra_kwargs_are_not_registered(name):
    assert name not in bridge._REGISTRY
    data = _make_data()
    with pytest.raises(ValueError):
        CIT(data, method=name)


def test_unknown_method_still_raises():
    data = _make_data()
    with pytest.raises(ValueError):
        CIT(data, method="not_a_real_method")


def test_pc_algorithm_runs_with_registered_test():
    from causallearn.search.ConstraintBased.PC import pc

    rng = np.random.default_rng(0)
    n = 300
    x = rng.standard_normal(n)
    z = rng.standard_normal(n)
    y = 0.5 * x + 0.3 * z + 0.1 * rng.standard_normal(n)
    w = 0.4 * y + 0.1 * rng.standard_normal(n)
    data = np.column_stack([x, y, z, w])

    cg = pc(data, indep_test="parcorr_gpu", device="cpu", show_progress=False)
    assert cg.G.graph.shape == (4, 4)
