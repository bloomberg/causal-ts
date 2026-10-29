# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""run_cdnots_plus wraps a discrete-unaware CI test in StratifiedCIT, as
run_cdnots does."""

import warnings

import numpy as np
import pandas as pd
import pytest

from causalts.cdnots.phase3_utils import run_cdnots_plus
from causalts.ci_tests.parcorr_gpu import ParCorrGPU
from causalts.ci_tests.stratified_cit import StratifiedCIT


def _mixed(T=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, 3))
    x[:, 0] = (x[:, 0] > 0).astype(float)
    for t in range(1, T):
        x[t, 1] += 0.5 * x[t - 1, 1] + 1.0 * x[t - 1, 0]
        x[t, 2] += 0.5 * x[t, 1]
    return pd.DataFrame(x, columns=["d", "a", "b"])


def _parcorr():
    return ParCorrGPU(np.zeros((2, 2)), device="cpu")


KW = dict(num_lags=1, include_C=False, alpha=0.01, show_progress=False)


def test_auto_detected_discrete_column_wraps_the_test():
    df = _mixed()
    with pytest.warns(UserWarning, match="wrapping it in StratifiedCIT"):
        auto = run_cdnots_plus(df, _parcorr(), **KW)
    wrapped = StratifiedCIT(
        data=np.zeros((2, 2)), discrete_cols=[], inner_cit=_parcorr()
    )
    ref = run_cdnots_plus(df, wrapped, discrete_cols=["d"], **KW)
    np.testing.assert_array_equal(auto.cg_tig, ref.cg_tig)


class _Rows(ParCorrGPU):
    """ParCorr that records the row counts it is called on."""

    def __init__(self, handles_discrete=False):
        super().__init__(np.zeros((2, 2)), device="cpu")
        self.handles_discrete = handles_discrete
        self.rows = set()

    def __call__(self, X, Y, condition_set=None):
        self.rows.add(self.data.shape[0])
        return super().__call__(X, Y, condition_set)


def test_explicit_discrete_cols_wrap_without_warning():
    df = _mixed()
    test = _Rows()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run_cdnots_plus(df, test, discrete_cols=["d"], **KW)
    # Wrapped: conditioning on the binary column splits the rows into strata.
    assert any(r < len(df) - 2 for r in test.rows)


def test_opt_out_and_self_declared_tests_are_not_wrapped():
    df = _mixed()
    for test, disc in ((_Rows(handles_discrete=True), None), (_Rows(), [])):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            run_cdnots_plus(df, test, discrete_cols=disc, **KW)
        # Called directly on full-length data, never on a stratum.
        assert test.rows == {len(df) - 2}


def test_wrapped_test_still_gets_the_lag_structure():
    class _Lagged(ParCorrGPU):
        def set_lag_structure(self, n_vars, lags):
            self.layouts.append((n_vars, list(lags)))

    inner = _Lagged(np.zeros((2, 2)), device="cpu")
    inner.layouts = []
    run_cdnots_plus(_mixed(), inner, discrete_cols=["d"], **KW)
    # The window, then the deeper MCI / collider-step array.
    assert (3, [0, 1]) in inner.layouts and (3, [0, 1, 2]) in inner.layouts


class _Counting(ParCorrGPU):
    """ParCorr with a lag hook that counts its own calls."""

    def __init__(self):
        super().__init__(np.zeros((2, 2)), device="cpu")
        self.n_calls = 0

    def set_lag_structure(self, n_vars, lags):
        pass

    def __call__(self, X, Y, condition_set=None):
        self.n_calls += 1
        return super().__call__(X, Y, condition_set)


def _strat(inner, data, disc):
    return StratifiedCIT(data=data, discrete_cols=disc, inner_cit=inner)


def test_stratified_cache_is_reset_by_a_new_lag_layout():
    data = _mixed().to_numpy()
    inner = _Counting()
    inner.pvalue_cache = {}
    t = _strat(inner, data, [0])
    t(1, 2, [])
    t(1, 2, [])
    n = inner.n_calls
    inner.pvalue_cache = {}
    t.set_lag_structure(n_vars=3, lags=[0, 1])
    t(1, 2, [])
    assert inner.n_calls == n + 1


def test_stratified_cache_key_holds_the_discrete_columns():
    data = _mixed().to_numpy()
    t = _strat(_parcorr(), data, [])
    unstratified = t(1, 2, [0])
    t.discrete_cols, t._disc_set = [0], {0}
    assert t(1, 2, [0]) != unstratified


def test_stratified_ignores_rows_missing_a_discrete_value():
    # x and y correlate with opposite signs in the two strata of d, so only
    # the stratified test sees the dependence.
    rng = np.random.default_rng(0)
    d = (rng.random(400) > 0.5).astype(float)
    s = rng.standard_normal(400)
    x = (2 * d - 1) * s + 0.5 * rng.standard_normal(400)
    y = s + 0.5 * rng.standard_normal(400)
    data = np.column_stack([d, x, y])
    assert ParCorrGPU(data, device="cpu")(1, 2, [0])[0] > 0.01
    holed = data.copy()
    holed[5, 0] = np.nan
    p_holed = _strat(_parcorr(), holed, [0])(1, 2, [0])[0]
    p_complete = _strat(_parcorr(), np.delete(data, 5, axis=0), [0])(1, 2, [0])[0]
    assert p_holed < 1e-6
    assert p_holed == pytest.approx(p_complete)


def test_stratum_failure_restores_the_inner_data():
    class _Boom(ParCorrGPU):
        def __call__(self, X, Y, condition_set=None):
            raise RuntimeError("boom")

    data = _mixed().to_numpy()
    inner = _Boom(np.zeros((2, 2)), device="cpu")
    t = _strat(inner, data, [0])
    with pytest.raises(RuntimeError):
        t(1, 2, [0])
    assert inner.data is data


def test_categorical_tests_are_not_wrapped():
    from causalts.cdnots.phase3_utils import _is_discrete_aware
    from causalts.ci_tests.causallearn_tests import CausalLearnChisq, CausalLearnGsq

    assert _is_discrete_aware(CausalLearnChisq())
    assert _is_discrete_aware(CausalLearnGsq())


def test_min_stratum_counts_complete_rows():
    # One stratum has 35 rows but only 25 with x observed: too few, so the
    # test falls back to the full data rather than combine an empty stratum.
    rng = np.random.default_rng(1)
    d = np.r_[np.zeros(35), np.ones(200)]
    x = rng.standard_normal(235)
    y = x + 0.5 * rng.standard_normal(235)
    x[:10] = np.nan
    data = np.column_stack([d, x, y])
    inner = _Rows()
    StratifiedCIT(data=data, discrete_cols=[0], inner_cit=inner, min_stratum=30)(
        1, 2, [0]
    )
    assert inner.rows == {235}
