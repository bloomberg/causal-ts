# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Adapters exposing causal-learn CI tests through the causal-ts registry.

These wrap causal-learn's FisherZ, Chi-squared, and G-squared tests to
conform to the ``(p_value, statistic)`` return convention used by all
causal-ts CI tests.
"""

import numpy as np

from .cit_test import CIT_Base


def _codes(*cols):
    """Integer codes of the distinct rows of ``cols`` (one category per row)."""
    stacked = np.column_stack(cols)
    return np.unique(stacked, axis=0, return_inverse=True)[1].ravel()


def _plugin_cmi(data, X, Y, condition_set):
    """Plug-in conditional mutual information of categorical X and Y given Z.

    Returns ``(stat, df)``: ``stat`` is, in nats,
    ``(1/n) * sum n_xyz * log(n_xyz * n_z / (n_xz * n_yz))`` over the cells
    with a positive count, with ``n`` the number of rows of ``data``, and
    ``df = sum_z (r_z - 1)(c_z - 1)`` over the strata, where ``r_z`` and
    ``c_z`` count the X and Y categories present in stratum ``z``.

    A stratum with a single X or Y category gets ``r_z - 1 = 0`` or
    ``c_z - 1 = 0`` and each of its terms is ``log 1 = 0``, so degenerate
    strata contribute nothing to either value, as in causal-learn.
    Categories absent from a stratum have no positive cells, so they are
    dropped from its table.
    """
    n = data.shape[0]
    # Recode in the original dtype: casting labels to int or float64 first
    # can merge distinct categories (3.1 and 3.5; integers above 2**53).
    x = _codes(data[:, X])
    y = _codes(data[:, Y])
    if condition_set is not None and len(condition_set) > 0:
        z = _codes(*(_codes(data[:, c]) for c in condition_set))
    else:
        z = np.zeros(n, dtype=np.intp)

    def per_row_count(*codes):
        _, inv, counts = np.unique(
            np.column_stack(codes), axis=0, return_inverse=True, return_counts=True
        )
        return counts[inv.ravel()].astype(np.float64)

    # Summing over rows weights each cell's term by its count n_xyz.
    ratio = per_row_count(z, x, y) * per_row_count(z)
    ratio /= per_row_count(z, x) * per_row_count(z, y)
    stat = float(np.log(ratio).sum() / n) if n > 0 else 0.0

    n_strata = int(z.max()) + 1 if n > 0 else 0
    r = np.bincount(np.unique(np.c_[z, x], axis=0)[:, 0], minlength=n_strata)
    c = np.bincount(np.unique(np.c_[z, y], axis=0)[:, 0], minlength=n_strata)
    df = int(((r - 1) * (c - 1)).sum())
    return stat, df


class CausalLearnFisherZ(CIT_Base):
    """Fisher-Z CI test from causal-learn (Gaussian, analytic p-value)."""

    rejects_inf = True  # an infinite tested value raises ValueError

    def __init__(self, data=None, **kwargs):
        kwargs.pop("device", None)
        super().__init__(data=data, method="fisherz", **kwargs)
        self._cl_test = None

    def _ensure_test(self):
        if self._cl_test is None or self._cl_data_id != id(self.data):
            from causallearn.utils.cit import FisherZ

            self._cl_test = FisherZ(self.data)
            self._cl_data_id = id(self.data)

    def set_data(self, data, data_hash=None):
        super().set_data(data, data_hash)
        self._cl_test = None

    def __call__(self, X, Y, condition_set=None):
        self.n_tests += 1
        if condition_set is None:
            condition_set = []
        # Before the cache, and as a ValueError: the constructor's
        # assert_input_data_is_valid is an assert, which python -O drops.
        self._reject_inf(X, Y, condition_set)

        # A constant column is independent of everything: p = 1, never NaN
        # (NaN > alpha is False, so the skeleton search would keep the edge).
        # A constant conditioning column carries no information, so drop it.
        # Without data, fall through to _get_array_hash's 'Data not set' error.
        # min == max rather than np.ptp, which raises on boolean data.
        if self.data is not None:

            def constant(c):
                col = self.data[:, c]
                return col.min() == col.max()

            if constant(X) or constant(Y):
                return 1.0, 0.0
            condition_set = [c for c in condition_set if not constant(c)]

        cache_key = self._get_array_hash(X, Y, condition_set)
        # Versioned: rows cached before the intercept fix carry the old statistic.
        param_hash = "fisherz-intercept"
        cached = self.pvalue_cache.get((cache_key, param_hash))
        if cached is not None:
            return cached[0], cached[1]

        self._ensure_test()
        pval = float(self._cl_test(X, Y, condition_set))

        # float64: centring and scaling float32 data can overflow.
        x = self.data[:, X].astype(np.float64)
        y = self.data[:, Y].astype(np.float64)
        if len(condition_set) > 0:
            from numpy.linalg import lstsq

            # Centre (the same fit as an intercept): otherwise the residuals
            # still depend on Z whenever Z has a nonzero mean, and the statistic
            # no longer matches the partial correlation behind causal-learn's
            # p-value. Scaling to unit variance keeps lstsq's rank cutoff from
            # dropping a column only because of its units.
            Z = self.data[:, condition_set].astype(np.float64)
            Z = Z - Z.mean(axis=0)
            sd = Z.std(axis=0)
            Z = Z / np.where(sd > 0, sd, 1.0)
            x = x - x.mean()
            y = y - y.mean()
            x = x - Z @ lstsq(Z, x, rcond=None)[0]
            y = y - Z @ lstsq(Z, y, rcond=None)[0]
        stat = float(np.corrcoef(x, y)[0, 1])

        self.pvalue_cache[(cache_key, param_hash)] = [pval, stat]
        self.save_incremental_cache(cache_key, param_hash, pval, stat)
        return pval, stat


class CausalLearnChisq(CIT_Base):
    """Chi-squared CI test from causal-learn (categorical data).

    The statistic is the plug-in conditional mutual information
    ``I(X;Y|Z)`` in nats (see ``_plugin_cmi``), not ``chi^2 / (2n)``, which
    only approximates it near independence. It is the same for ``chisq`` and
    ``gsquared``: the two tests differ only in their p-value. Strata in
    which X or Y takes a single value contribute 0 to it and to the degrees
    of freedom; if every stratum does, the result is ``(1.0, 0.0)``. No bias
    correction is applied: for a fixed, finite category support and
    adequately populated tables, its first-order upward bias under
    conditional independence is ``df / (2n)``. This is an asymptotic
    approximation, not a finite-sample identity, and does not hold for
    sparse strata.
    """

    handles_discrete = True  # CDNOTS does not wrap it in StratifiedCIT

    def __init__(self, data=None, **kwargs):
        kwargs.pop("device", None)
        super().__init__(data=data, method="chisq", **kwargs)
        self._cl_test = None

    def _ensure_test(self):
        if self._cl_test is None or self._cl_data_id != id(self.data):
            from causallearn.utils.cit import Chisq_or_Gsq

            self._cl_test = Chisq_or_Gsq(self.data, "chisq")
            self._cl_data_id = id(self.data)

    def set_data(self, data, data_hash=None):
        super().set_data(data, data_hash)
        self._cl_test = None

    def __call__(self, X, Y, condition_set=None):
        self.n_tests += 1
        if condition_set is None:
            condition_set = []

        cache_key = self._get_array_hash(X, Y, condition_set)
        # Versioned: rows cached before the statistic was added carry 0.0.
        param_hash = "chisq-cmi"
        cached = self.pvalue_cache.get((cache_key, param_hash))
        if cached is not None:
            return cached[0], cached[1]

        self._ensure_test()
        pval = float(self._cl_test(X, Y, condition_set))
        stat = _plugin_cmi(self.data, X, Y, condition_set)[0]

        self.pvalue_cache[(cache_key, param_hash)] = [pval, stat]
        self.save_incremental_cache(cache_key, param_hash, pval, stat)
        return pval, stat


class CausalLearnGsq(CIT_Base):
    """G-squared (likelihood ratio) CI test from causal-learn (categorical data).

    The statistic, ``G^2 / (2n)``, is the plug-in conditional mutual
    information ``I(X;Y|Z)`` in nats (see ``_plugin_cmi``), the same for ``chisq`` and
    ``gsquared``: the two tests differ only in their p-value. Strata in
    which X or Y takes a single value contribute 0 to it and to the degrees
    of freedom; if every stratum does, the result is ``(1.0, 0.0)``. No bias
    correction is applied: for a fixed, finite category support and
    adequately populated tables, its first-order upward bias under
    conditional independence is ``df / (2n)``. This is an asymptotic
    approximation, not a finite-sample identity, and does not hold for
    sparse strata.
    """

    handles_discrete = True  # CDNOTS does not wrap it in StratifiedCIT

    def __init__(self, data=None, **kwargs):
        kwargs.pop("device", None)
        super().__init__(data=data, method="gsquared", **kwargs)
        self._cl_test = None

    def _ensure_test(self):
        if self._cl_test is None or self._cl_data_id != id(self.data):
            from causallearn.utils.cit import Chisq_or_Gsq

            self._cl_test = Chisq_or_Gsq(self.data, "gsq")
            self._cl_data_id = id(self.data)

    def set_data(self, data, data_hash=None):
        super().set_data(data, data_hash)
        self._cl_test = None

    def __call__(self, X, Y, condition_set=None):
        self.n_tests += 1
        if condition_set is None:
            condition_set = []

        cache_key = self._get_array_hash(X, Y, condition_set)
        # Versioned: rows cached before the statistic was added carry 0.0.
        param_hash = "gsquared-cmi"
        cached = self.pvalue_cache.get((cache_key, param_hash))
        if cached is not None:
            return cached[0], cached[1]

        self._ensure_test()
        pval = float(self._cl_test(X, Y, condition_set))
        stat = _plugin_cmi(self.data, X, Y, condition_set)[0]

        self.pvalue_cache[(cache_key, param_hash)] = [pval, stat]
        self.save_incremental_cache(cache_key, param_hash, pval, stat)
        return pval, stat
