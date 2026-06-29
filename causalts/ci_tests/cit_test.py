# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

import csv
import hashlib
import os
import time
from collections.abc import Iterable
from hashlib import sha1

import numpy as np
import pandas as pd


class CIT_Base(object):
    """Unified CIT_Base (consolidated from former cit_test and cit_base).

    Backward compatibility:

    - data is now optional; call ``set_data(data)`` later if needed.
    - seed and random_state added (from legacy cit_base).
    - ``cit_base.py`` now re-exports this class and issues DeprecationWarning.

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
        # hash: if data absent create placeholder
        if data_hash is not None:
            self.data_hash = data_hash
        else:
            self.data_hash = (
                hashlib.md5(str(data).encode("utf-8")).hexdigest()
                if data is not None
                else f"nodata-{method}"
            )
        if data is not None:
            self.sample_size, self.num_features = data.shape
        else:
            self.sample_size, self.num_features = (0, 0)

        self._has_nan = bool(np.isnan(data).any()) if data is not None else False
        self._min_samples = kwargs.pop("min_samples", 30)

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
        cols = [X] if isinstance(X, (int, np.integer)) else list(X)
        cols += [Y] if isinstance(Y, (int, np.integer)) else list(Y)
        if condition_set is not None and len(condition_set) > 0:
            cols += list(condition_set)
        return ~np.isnan(self.data[:, cols]).any(axis=1)

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

        x_orderd = sorted(X) if isinstance(X, Iterable) else int(X)
        y_orderd = sorted(Y) if isinstance(Y, Iterable) else int(Y)
        z_orderd = sorted(Z) if isinstance(Z, Iterable) else int(Z)

        x_hash = sha1(np.ascontiguousarray(self.data[:, x_orderd])).hexdigest()
        y_hash = sha1(np.ascontiguousarray(self.data[:, y_orderd])).hexdigest()
        z_hash = sha1(np.ascontiguousarray(self.data[:, z_orderd])).hexdigest()

        sorted_xy = sorted([x_hash, y_hash])
        combined_hash = "_".join((sorted_xy[0], sorted_xy[1], z_hash))
        return combined_hash
