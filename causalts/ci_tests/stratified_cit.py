# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""StratifiedCIT: wrapper that handles discrete conditioning variables by
partitioning the sample into strata and combining per-stratum p-values via
Fisher's method. Works with any inner CIT_Base-compatible test."""

import warnings

import numpy as np
from scipy.stats import chi2

from .cit_test import CIT_Base


def compute_discrete_indices(discrete_col_names, df, lags, include_C=True):
    """Map original DataFrame column names to indices in the lag-embedded matrix.

    Parameters
    ----------
    discrete_col_names : list[str]
        Column names in df that are discrete.
    df : pd.DataFrame
        Original (pre-embedding) time series DataFrame.
    lags : list[int]
        Lag values used in cdnots_discovery (e.g. [0, 1, 2, 3]).
    include_C : bool
        Whether the time node C is appended as the last column.

    Returns
    -------
    list[int]
        Column indices in the lag-embedded (T - max_lag, K * (num_lags+1)) matrix
        corresponding to all lags of each discrete column.
    """
    cols = list(df.columns)
    if include_C:
        cols = cols + ["C"]
    K = len(cols)
    indices = []
    for name in discrete_col_names:
        if name not in cols:
            warnings.warn(
                f"compute_discrete_indices: '{name}' not in lag-embedded columns; skipped.",
                UserWarning,
                stacklevel=2,
            )
            continue
        base = cols.index(name)
        for lag_i, _ in enumerate(lags):
            indices.append(lag_i * K + base)
    return sorted(set(indices))


class StratifiedCIT(CIT_Base):
    """Stratified Conditional Independence Test for mixed discrete-continuous data.

    .. note::
        This class is an **internal wrapper** — it is not registered in the
        CI test factory (``create_ci_test``) and cannot be selected via
        ``--ci-test`` on the CLI. It is applied automatically by
        ``run_cdnots`` / ``run_cdnots_plus`` when discrete columns are
        detected in the input DataFrame. Pass ``discrete_cols=[]`` to opt out.

    For CI tests where the conditioning set Z contains discrete columns:
    partitions the sample into strata defined by the joint discrete values,
    runs the inner CI test on continuous Z within each stratum, then combines
    stratum p-values via Fisher's method (-2·Σlog(p) ~ χ²(2k)).

    Falls back to the inner CI test directly when:
    - No discrete columns appear in the conditioning set
    - The number of strata exceeds max_strata
    - Any stratum has fewer than min_stratum observations

    Parameters
    ----------
    data : np.ndarray, shape (T, d)
        Full lag-embedded data matrix.
    discrete_cols : list[int]
        Column indices (into data) that are discrete.
        Use compute_discrete_indices() to map from original column names.
    inner_cit : CIT_Base instance
        The inner CI test to use within each stratum (or for continuous-only tests).
        Must already have data set (data will be overridden per-stratum).
    max_strata : int, default 8
        Maximum number of strata before falling back to inner_cit on full data.
    min_stratum : int, default 30
        Minimum rows per stratum before falling back.
    """

    def __init__(
        self,
        data,
        discrete_cols,
        inner_cit,
        max_strata=8,
        min_stratum=30,
        **kwargs,
    ):
        # Set inner_cit before super().__init__ because CIT_Base.__init__
        # calls self.data = data which triggers set_data -> inner_cit.data = data
        self.inner_cit = inner_cit
        self.discrete_cols = list(discrete_cols)
        self._disc_set = set(discrete_cols)
        self.max_strata = max_strata
        self.min_stratum = min_stratum
        super().__init__(data=data, method="stratified_cit", **kwargs)

    def get_data(self):
        return self._data

    def set_data(self, d):
        self._data = d
        self._has_nan = bool(np.isnan(d).any()) if d is not None else False
        self.inner_cit.data = d

    data = property(get_data, set_data)

    def __call__(self, X, Y, condition_set=None):
        self.n_tests += 1
        if condition_set is None:
            condition_set = []

        cache_key = self._get_array_hash(X, Y, condition_set)
        cache_key = f"strat-{cache_key}"
        combined_hash = (cache_key, "stratified")
        if combined_hash in self.pvalue_cache:
            return self.pvalue_cache[combined_hash]

        disc_in_z = [c for c in condition_set if c in self._disc_set]
        cont_in_z = [c for c in condition_set if c not in self._disc_set]

        if not disc_in_z:
            p, stat = self.inner_cit(X, Y, condition_set)
            self.pvalue_cache[combined_hash] = (p, stat)
            return p, stat

        # Partition rows by joint value of discrete conditioning columns
        data = self._data
        disc_vals = data[:, disc_in_z]
        unique_combos = np.unique(disc_vals, axis=0)

        if len(unique_combos) > self.max_strata:
            p, stat = self.inner_cit(X, Y, condition_set)
            self.pvalue_cache[combined_hash] = (p, stat)
            return p, stat

        pvals = []
        for combo in unique_combos:
            mask = np.all(disc_vals == combo, axis=1)
            if mask.sum() < self.min_stratum:
                # Fall back: too few observations in this stratum
                p, stat = self.inner_cit(X, Y, condition_set)
                self.pvalue_cache[combined_hash] = (p, stat)
                return p, stat
            p_s, _ = self._test_stratum(X, Y, cont_in_z, mask)
            pvals.append(p_s)

        p = self._fisher_combine(pvals)
        self.pvalue_cache[combined_hash] = (p, 0.0)
        return p, 0.0

    def _test_stratum(self, X, Y, cont_in_z, mask):
        """Run inner_cit on the subset of rows defined by mask."""
        stratum_data = self._data[mask]
        self.inner_cit.data = stratum_data
        p, stat = self.inner_cit(X, Y, cont_in_z)
        self.inner_cit.data = self._data  # restore
        return p, stat

    @staticmethod
    def _fisher_combine(pvals):
        """Combine p-values via Fisher's method: -2·Σlog(p) ~ χ²(2k)."""
        pvals = [max(p, 1e-300) for p in pvals]
        stat = -2.0 * sum(np.log(p) for p in pvals)
        df = 2 * len(pvals)
        return float(1.0 - chi2.cdf(stat, df))
