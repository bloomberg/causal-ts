# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

import csv
import os
import warnings
from collections.abc import Iterable
from hashlib import sha1

# Deprecated shim: use causalts.ci_tests.cit_test.CIT_Base instead.
from .cit_test import CIT_Base  # noqa: F401

warnings.warn(
    "cit_base.CIT_Base is deprecated and consolidated into cit_test.CIT_Base. "
    "Update imports to: from causalts.ci_tests.cit_test import CIT_Base",
    DeprecationWarning,
    stacklevel=2,
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


class CIT_Base(object):  # noqa: F811
    """Base class of conditional independence tests"""

    def __init__(
        self,
        seed=42,
        cache_dir=None,
        data_hash=None,
        method="parcorr",
        **kwargs,
    ):
        # Set the dataframe to None for now, will be reset during cdnots call
        self.dataframe = None
        # Set the options
        self.random_state = np.random.default_rng(seed)
        self.pvalue_cache = {}
        self.cache_dir = cache_dir
        self.data_hash = data_hash
        self.n_actual_tests = 0
        self.n_tests = 0
        self.method = method
        self.cache_file = None
        self.pvalue_cache = {}
        # self.pvalue_local = {}
        # self.sample_size, self.num_features = data.shape

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
                )  # .filter(like=method,axis=0)
            else:
                os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
                if not os.path.exists(self.cache_file):
                    pd.DataFrame(
                        [], columns=["key", "param_hash", "pval", "stats"]
                    ).to_csv(self.cache_file, index=False)
        else:
            self.cache_file = None

    def refresh_cache(self):
        """Refresh the p-value cache from the on-disk cache file.

        Reads the CSV into a dict keyed by ``(key, param_hash)``.

        Raises
        ------
        FileNotFoundError
            If the specified cache file does not exist.
        ValueError
            If the cache file does not have the required columns.
        """
        if self.cache_file is not None:
            self.pvalue_cache = (
                pd.read_csv(self.cache_file)
                .set_index(["key", "param_hash"])
                .apply(lambda row: row.values.tolist(), axis=1)
                .to_dict()
            )

    def check_cache_method_consistent(self, method_name, parameters_hash):
        """Ensure the cached method name and parameters hash are consistent.

        Parameters
        ----------
        method_name : str
            The name of the CI test method.
        parameters_hash : str
            A hash representing the parameters of the method.

        Raises
        ------
        AssertionError
            If the cached method name or parameters hash do not match.
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
        """Check that the input data contains no NaN or Inf values.

        Parameters
        ----------
        allow_nan : bool
            If True, allows NaN values in the data.
        allow_inf : bool
            If True, allows Inf values in the data.

        Raises
        ------
        AssertionError
            If the data contains NaN or Inf when not allowed.
        """
        assert (
            allow_nan or not np.isnan(self.data).any()
        ), "Input data contains NaN. Please check."
        assert (
            allow_inf or not np.isinf(self.data).any()
        ), "Input data contains Inf. Please check."

    def save_incremental_cache(self, key, param_hash, pval, stat):
        if self.cache_file is not None:
            with open(self.cache_file, "a") as fout:
                csv_writer = csv.writer(fout)
                csv_writer.writerow([key, param_hash, pval, stat])

    def _get_array_hash(self, X, Y, Z):
        """Helper function to get hash of array.

        For a CI test X _|_ Y | Z the order of variables within X or Y or Z
        does not matter and also the order X and Y can be swapped.
        Hence, to compare hashes of the whole array, we order accordingly
        to create a unique, order-independent hash."""

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
