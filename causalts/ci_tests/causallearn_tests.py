# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Adapters exposing causal-learn CI tests through the causal-ts registry.

These wrap causal-learn's FisherZ, Chi-squared, and G-squared tests to
conform to the ``(p_value, statistic)`` return convention used by all
causal-ts CI tests.
"""

import numpy as np

from .cit_test import CIT_Base


class CausalLearnFisherZ(CIT_Base):
    """Fisher-Z CI test from causal-learn (Gaussian, analytic p-value)."""

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

        cache_key = self._get_array_hash(X, Y, condition_set)
        param_hash = "fisherz"
        cached = self.pvalue_cache.get((cache_key, param_hash))
        if cached is not None:
            return cached[0], cached[1]

        self._ensure_test()
        pval = float(self._cl_test(X, Y, condition_set))

        x = self.data[:, X]
        y = self.data[:, Y]
        if condition_set:
            from numpy.linalg import lstsq

            Z = self.data[:, condition_set]
            x = x - Z @ lstsq(Z, x, rcond=None)[0]
            y = y - Z @ lstsq(Z, y, rcond=None)[0]
        stat = float(np.corrcoef(x, y)[0, 1])

        self.pvalue_cache[(cache_key, param_hash)] = [pval, stat]
        self.save_incremental_cache(cache_key, param_hash, pval, stat)
        return pval, stat


class CausalLearnChisq(CIT_Base):
    """Chi-squared CI test from causal-learn (categorical data)."""

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
        param_hash = "chisq"
        cached = self.pvalue_cache.get((cache_key, param_hash))
        if cached is not None:
            return cached[0], cached[1]

        self._ensure_test()
        pval = float(self._cl_test(X, Y, condition_set))

        self.pvalue_cache[(cache_key, param_hash)] = [pval, 0.0]
        self.save_incremental_cache(cache_key, param_hash, pval, 0.0)
        return pval, 0.0


class CausalLearnGsq(CIT_Base):
    """G-squared (likelihood ratio) CI test from causal-learn (categorical data)."""

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
        param_hash = "gsquared"
        cached = self.pvalue_cache.get((cache_key, param_hash))
        if cached is not None:
            return cached[0], cached[1]

        self._ensure_test()
        pval = float(self._cl_test(X, Y, condition_set))

        self.pvalue_cache[(cache_key, param_hash)] = [pval, 0.0]
        self.save_incremental_cache(cache_key, param_hash, pval, 0.0)
        return pval, 0.0
