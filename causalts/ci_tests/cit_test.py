# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

import csv
import hashlib
import operator
import os
import time
from collections.abc import Iterable
from hashlib import sha1

import numpy as np
import pandas as pd


def stable_seed(*parts):
    """Deterministic 32-bit seed from arbitrary values.

    ``hash()`` must not be used for this. Python randomises string hashing per
    process (PYTHONHASHSEED), so a seed derived from it changes between runs:
    a bootstrap or permutation test reseeded that way silently returns a
    different p-value in every process, and any harness pairing methods on a
    shared draw stops being paired. blake2b is stable across processes,
    platforms and versions.
    """
    h = hashlib.blake2b(repr(parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % (2**32)


def _test_columns(X, Y, condition_set):
    """Column indices a CI test (X, Y, condition_set) reads, as Python ints.

    Plain ints matter: a list mixing ``np.uint64`` with ``int`` would become a
    float64 index array, which numpy refuses for indexing.
    """
    cols = [X] if isinstance(X, (int, np.integer)) else list(X)
    cols += [Y] if isinstance(Y, (int, np.integer)) else list(Y)
    if isinstance(condition_set, (int, np.integer)):
        cols.append(condition_set)
    elif condition_set is not None and len(condition_set) > 0:
        cols += list(condition_set)
    return [operator.index(c) for c in cols]


def reject_inf(data, X, Y, condition_set, name):
    """Raise ``ValueError`` if a tested column holds +/-inf.

    Missing values (NaN) are left to each test's pairwise-complete handling.
    An infinite value is not missing: without this check moment- and
    kernel-based tests returned p = 1 or NaN for it, which a discovery
    algorithm reads as independence.
    """
    if data is None:
        return  # _get_array_hash raises the 'Data not set' ValueError
    if np.isinf(data[:, _test_columns(X, Y, condition_set)]).any():
        raise ValueError(f"{name}: the tested columns contain infinite values")


class CIT_Base(object):
    """Unified CIT_Base (consolidated from former cit_test and cit_base).

    Backward compatibility:

    - data is now optional; call ``set_data(data)`` later if needed.
    - seed and random_state added (from legacy cit_base).
    - ``cit_base.py`` now re-exports this class and issues DeprecationWarning.

    ``order_independent`` (keyword, default False): the p-value cache, which is
    always on, treats ``(X, Y, Z)`` and ``(Y, X, Z)`` as one test, so by default
    the orientation queried first decides the value for both, and a test that
    draws from the instance generator depends on the tests run before it. With
    ``order_independent=True``, ``KCIGPU``, ``LinSigCIGPU``, ``GCMIGPU`` and
    ``CMIknnGPU`` / ``CMIknnMixedGPU`` compute every query in one canonical
    order, and ``CMIknnGPU`` draws each test's randomness from ``seed`` and the
    test itself, so a p-value depends on neither the call order nor the
    orientation. Other tests ignore it. Off by default so p-values match
    earlier releases.

    Attributes
    ----------
    data : np.ndarray
        Array of shape (n_samples, n_features) or None until set.
    method : str
        Name of the CI test method.
    cache_dir : str or None
        Directory for on-disk p-value cache.
    data_hash : str
        Hash identifying the current dataset.
    pvalue_cache : dict
        In-memory cache of computed p-values.
    pvalue_local : dict
        Session-local cache of computed p-values.
    random_state : np.random.Generator
        NumPy random number generator instance.
    """

    # Whether the test raises ValueError for an infinite value in a tested
    # column. Rank- and CDF-based tests accept one and leave this False;
    # StratifiedCIT follows its inner test.
    rejects_inf = False

    def __init__(
        self,
        data=None,
        method="kcigpu",
        cache_dir=None,
        data_hash=None,
        seed=42,
        **kwargs,
    ):
        # allow delayed data assignment
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("Input data must be a numpy array.")
        self.data = data
        self.method = method
        self.random_state = np.random.default_rng(seed)
        self._base_seed = seed
        # hash: if data absent create placeholder
        if data_hash is None:
            data_hash = (
                hashlib.md5(str(data).encode("utf-8")).hexdigest()
                if data is not None
                else f"nodata-{method}"
            )
        self.data_hash = data_hash
        if data is not None:
            self.sample_size, self.num_features = data.shape
        else:
            self.sample_size, self.num_features = (0, 0)

        self._has_nan = bool(np.isnan(data).any()) if data is not None else False
        self._min_samples = kwargs.pop("min_samples", 30)
        # Opt-in: one answer per test whatever the call order or orientation
        # (see _canonical_query / _test_seed). Off by default so p-values stay
        # those of earlier releases.
        self._order_independent = bool(kwargs.pop("order_independent", False))

        self.cache_dir = cache_dir
        self.pvalue_cache = {}
        self.pvalue_local = {}
        self.n_actual_tests = 0
        self.n_tests = 0
        self._cache_buffer = []
        self._last_flush_time = time.time()
        self._flush_interval = 20  # seconds between disk writes

        if cache_dir is not None:
            self.cache_file = os.path.join(cache_dir, f"{self.data_hash}.csv")
            if os.path.exists(self.cache_file):
                self.pvalue_cache = (
                    pd.read_csv(self.cache_file)
                    .drop_duplicates()
                    .set_index(["key", "param_hash"])
                    .apply(lambda row: row.values.tolist(), axis=1)
                    .filter(like=method, axis=0)
                    .to_dict()
                )
            else:
                os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
                if not os.path.exists(self.cache_file):
                    pd.DataFrame(
                        [], columns=["key", "param_hash", "pval", "stats"]
                    ).to_csv(self.cache_file, index=False)
        else:
            self.cache_file = None

    @property
    def _oi_tag(self):
        """``param_hash`` suffix for ``order_independent=True`` (empty when off)."""
        return "-oi" if getattr(self, "_order_independent", False) else ""

    def _canonical_query(self, X, Y, Z):
        """(X, Y, Z) in the order :meth:`_get_array_hash` treats as one test.

        Only with ``order_independent=True``; otherwise the query is returned
        unchanged. The cache key ignores an X/Y swap and the order within X, Y
        and Z, so for a test whose computation is not symmetric (a permuted X, a
        kernel on (X, Z) but not (Y, Z), jitter assigned by column position)
        whichever orientation runs first decides the cached answer for both.
        That is the default, kept so p-values match earlier releases; the
        opt-in runs every query in one canonical order instead.
        """
        if not getattr(self, "_order_independent", False):
            return X, Y, Z
        x = sorted(X) if isinstance(X, Iterable) else int(X)
        y = sorted(Y) if isinstance(Y, Iterable) else int(Y)
        z = sorted(Z) if Z is not None else []
        x_hash = sha1(np.ascontiguousarray(self.data[:, x])).hexdigest()
        y_hash = sha1(np.ascontiguousarray(self.data[:, y])).hexdigest()
        if y_hash < x_hash:
            x, y = y, x
        return x, y, z

    def _test_seed(self, cache_key):
        """A 32-bit seed for one test, from ``seed`` and the test's cache key.

        Randomised tests (permutations, bootstrap draws, tie-breaking jitter)
        draw from a generator seeded with this, so a test's p-value depends on
        ``seed`` and the test alone: not on the call order, and the same in a
        fresh instance. ``seed=None`` keeps the draws unseeded.
        """
        base = getattr(self, "_base_seed", None)
        if base is None:
            return None
        if isinstance(base, (int, np.integer)):
            base = int(base)  # repr(np.int64(3)) is 'np.int64(3)' on NumPy 2
        return stable_seed(base, cache_key)

    def refresh_cache(self):
        """
        Refreshes the p-value cache by loading data from the specified cache file.
        """
        if self.cache_file is not None:
            self.pvalue_cache = (
                pd.read_csv(self.cache_file)
                .set_index(["key", "param_hash"])
                .apply(lambda row: row.values.tolist(), axis=1)
                .to_dict()
            )

    def check_cache_method_consistent(self, method_name, parameters_hash):
        """
        Ensures consistency between the cached method name and parameters hash
        with the provided values. Updates the cache if it is newly created.

        Args:
            method_name (str): The name of the CI test method.
            parameters_hash (str): The hash of the method's parameters.

        Raises:
            AssertionError: If the cached method name or parameters hash
                            does not match the provided values.
        """
        self.method = method_name
        if method_name not in self.pvalue_cache:
            self.pvalue_cache["method_name"] = method_name  # a newly created cache
            self.pvalue_cache["parameters_hash"] = parameters_hash
        else:
            assert (
                self.pvalue_cache["method_name"] == method_name
            ), "CI test method name mismatch."  # a loaded cache
            assert (
                self.pvalue_cache["parameters_hash"] == parameters_hash
            ), "CI test method parameters mismatch."

    def assert_input_data_is_valid(self, allow_nan=False, allow_inf=False):
        """
        Validates the input data by checking for NaN and Inf values.

        Parameters:
            allow_nan (bool): If True, allows NaN values in the data. Default is False.
            allow_inf (bool): If True, allows Inf values in the data. Default is False.

        Raises:
            AssertionError: If NaN or Inf values are found and not allowed.
        """
        assert (
            allow_nan or not np.isnan(self.data).any()
        ), "Input data contains NaN. Please check."
        assert (
            allow_inf or not np.isinf(self.data).any()
        ), "Input data contains Inf. Please check."

    def save_incremental_cache(self, key, param_hash, pval, stat):
        """
        Buffers cache data and flushes to disk periodically.

        Rows are accumulated in memory and written to the CSV file every
        ``_flush_interval`` seconds (default 20s) to avoid the overhead of
        opening/closing the file after every single CI test.

        Args:
            key (str): The unique identifier for the cache entry.
            param_hash (str): The hash of the parameters associated with the entry.
            pval (float): The p-value to be cached.
            stat (float): The statistical value to be cached.
        """
        if self.cache_file is not None:
            self._cache_buffer.append([key, param_hash, pval, stat])
            if time.time() - self._last_flush_time >= self._flush_interval:
                self.flush_cache()

    def flush_cache(self):
        """Flush any buffered cache rows to disk."""
        if self.cache_file is not None and self._cache_buffer:
            with open(self.cache_file, "a") as fout:
                csv_writer = csv.writer(fout)
                csv_writer.writerows(self._cache_buffer)
            self._cache_buffer.clear()
            self._last_flush_time = time.time()

    def __del__(self):
        """Flush remaining buffered cache entries on garbage collection."""
        try:
            self.flush_cache()
        except Exception:
            pass

    def set_data(self, data, data_hash=None):
        """Assign data after construction (for backward compatibility).

        Invalidates the in-memory cache and updates the data hash so that
        stale results from previous data are never returned.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy array.")
        # Buffered rows belong to the old data's cache file; write them there first.
        self.flush_cache()
        self.data = data
        self.sample_size, self.num_features = data.shape
        self._has_nan = bool(np.isnan(data).any())
        if data_hash is not None:
            self.data_hash = data_hash
        else:
            self.data_hash = hashlib.md5(str(data).encode("utf-8")).hexdigest()
        # Invalidate in-memory cache — it was keyed on the old data
        self.pvalue_cache = {}
        self.pvalue_local = {}
        # Re-point cache file to the new data hash
        if self.cache_dir is not None:
            self.cache_file = os.path.join(self.cache_dir, f"{self.data_hash}.csv")
            if os.path.exists(self.cache_file):
                self.pvalue_cache = (
                    pd.read_csv(self.cache_file)
                    .drop_duplicates()
                    .set_index(["key", "param_hash"])
                    .apply(lambda row: row.values.tolist(), axis=1)
                    .filter(like=self.method, axis=0)
                    .to_dict()
                )
            else:
                os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
                pd.DataFrame([], columns=["key", "param_hash", "pval", "stats"]).to_csv(
                    self.cache_file, index=False
                )

    def _pairwise_complete(self, X, Y, condition_set):
        """Return boolean row mask where all relevant columns are non-NaN."""
        cols = _test_columns(X, Y, condition_set)
        return ~np.isnan(self.data[:, cols]).any(axis=1)

    def _reject_inf(self, X, Y, condition_set):
        """Raise ``ValueError`` if a tested column holds +/-inf (see reject_inf)."""
        reject_inf(self.data, X, Y, condition_set, type(self).__name__)

    def _get_array_hash(self, X, Y, Z):
        """Helper function to get hash of array.

        For a CI test X _|_ Y | Z the order of variables within X or Y or Z
        does not matter and also the order X and Y can be swapped.
        Hence, to compare hashes of the whole array, we order accordingly
        to create a unique, order-independent hash."""

        if self.data is None:
            raise ValueError("Data not set. Call set_data() before hashing.")
        if Z is None:
            Z = []

        x_ordered = sorted(X) if isinstance(X, Iterable) else int(X)
        y_ordered = sorted(Y) if isinstance(Y, Iterable) else int(Y)
        z_ordered = sorted(Z) if isinstance(Z, Iterable) else int(Z)

        x_hash = sha1(np.ascontiguousarray(self.data[:, x_ordered])).hexdigest()
        y_hash = sha1(np.ascontiguousarray(self.data[:, y_ordered])).hexdigest()
        z_hash = sha1(np.ascontiguousarray(self.data[:, z_ordered])).hexdigest()

        sorted_xy = sorted([x_hash, y_hash])
        combined_hash = "_".join((sorted_xy[0], sorted_xy[1], z_hash))
        return combined_hash
