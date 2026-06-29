# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""GPU-accelerated partial correlation CI test with batched evaluation."""

import hashlib
import sys
from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import torch
from scipy import stats

from .cit_test import CIT_Base


def _data_fingerprint(data):
    """Fast dataset identity hash — sampled MD5, not a per-column hash.

    Samples ~50K values so cost stays < 1ms even at d=75 T=2000 (2.4 MB).
    Used in param_hash so pvalue_cache keys automatically miss when data changes.
    """
    step = max(1, data.size // 50_000)
    return hashlib.md5(data.flat[::step].tobytes()).hexdigest()[:12]


class ParCorrGPU(CIT_Base):
    """GPU-accelerated partial correlation test with batched evaluation.

    Supports two methods for computing partial correlation:

    - ``"precision"`` (default): inverts the sub-covariance matrix for each
      test to obtain the partial correlation from the precision matrix.
      Much faster and more memory-efficient than OLS.
    - ``"ols"``: regresses out conditioning variables via OLS
      (torch.linalg.lstsq) and tests the Pearson correlation of residuals.

    Parameters
    ----------
    data : numpy.ndarray
        Input data matrix of shape (n_samples, n_features).
    device : str, optional
        Device to use ('cuda', 'mps', or 'cpu'). Auto-detects if None.
    enable_batching : bool, optional
        Whether to enable batched evaluation. Default is True.
    method : str, optional
        ``"precision"`` (default) or ``"ols"``.
    **kwargs : dict
        Additional arguments passed to CIT_Base.
    """

    def __init__(
        self,
        data,
        device=None,
        enable_batching=True,
        method="precision",
        enable_caching=True,
        **kwargs,
    ):
        self.device = device
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

        self._method = method
        # Precision path always runs on CPU — small matrix inversions (k×k, k=2-10)
        # are faster on CPU than GPU due to kernel launch overhead
        self._precision_device = "cpu"
        self._data = data
        if data is not None:
            self.data_torch = torch.Tensor(data).to(self.device)
            # Pre-compute covariance matrix (skip if NaN present)
            if not np.isnan(data).any():
                data_t = torch.Tensor(data).to(self._precision_device)
                data_centered = data_t - data_t.mean(dim=0, keepdim=True)
                self._cov_matrix = (data_centered.T @ data_centered) / (
                    data_centered.shape[0] - 1
                )
            else:
                self._cov_matrix = None

        super().__init__(data, method="parcorr_gpu", **kwargs)
        fp = _data_fingerprint(data)
        self.param_hash = f"T{data.shape[0]}-analytic-{fp}"
        self.enable_batching = enable_batching and not self._has_nan
        self.enable_caching = enable_caching
        if not self._has_nan:
            self.assert_input_data_is_valid()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d):
        self._data = d
        self._has_nan = bool(np.isnan(d).any()) if d is not None else False
        if self._has_nan:
            self.enable_batching = False
        if d is not None:
            self.data_torch = torch.Tensor(d).to(self.device)
            if not self._has_nan:
                data_t = torch.Tensor(d).to(self._precision_device)
                data_centered = data_t - data_t.mean(dim=0, keepdim=True)
                self._cov_matrix = (data_centered.T @ data_centered) / (
                    data_centered.shape[0] - 1
                )
            else:
                self._cov_matrix = None
            self.param_hash = f"T{d.shape[0]}-analytic-{_data_fingerprint(d)}"

    def _get_array_hash(self, X, Y, Z):
        """Hash a (X, Y, Z) CI test triple using column indices only.

        Since the dataset is fixed for the lifetime of this CI object, column
        index i always refers to the same data — no need to SHA1-hash the
        actual array contents. Plain integer formatting is ~10-25x faster
        than SHA1 and produces collision-free keys for our index range.
        """
        if Z is None:
            Z = []
        x = int(X) if not isinstance(X, Iterable) else tuple(sorted(int(i) for i in X))
        y = int(Y) if not isinstance(Y, Iterable) else tuple(sorted(int(i) for i in Y))
        z = tuple(sorted(int(i) for i in Z))
        lo, hi = (min(x, y), max(x, y)) if isinstance(x, int) else tuple(sorted([x, y]))
        return f"{lo}_{hi}_{z}"

    def __call__(self, X, Y, condition_set=None):
        """Perform partial correlation test for given variables.

        Parameters
        ----------
        X : int
            Index of variable X.
        Y : int
            Index of variable Y.
        condition_set : list, optional
            Indices of conditioning variables.

        Returns
        -------
        tuple
            (p_value, statistic)
        """
        self.n_tests += 1
        if condition_set is None:
            condition_set = []

        cache_key = self._get_array_hash(X, Y, condition_set)
        cache_key = f"{self.method}-{cache_key}"
        combined_hash = cache_key, self.param_hash
        if combined_hash in self.pvalue_cache:
            return self.pvalue_cache[combined_hash]
        self.n_actual_tests += 1

        if self._has_nan:
            mask = self._pairwise_complete(X, Y, condition_set)
            n_valid = int(mask.sum())
            if n_valid < self._min_samples:
                self.pvalue_cache[combined_hash] = 1.0, 0.0
                return 1.0, 0.0
            data_clean = self.data_torch[torch.from_numpy(mask).to(self.device)]
            val = self._compute_parcorr_on(data_clean, X, Y, condition_set)
            pval = self._analytic_pvalue(val, condition_set, n_override=n_valid)
        else:
            val = self._compute_parcorr(X, Y, condition_set)
            pval = self._analytic_pvalue(val, condition_set)

        self.pvalue_cache[combined_hash] = pval, val
        self.save_incremental_cache(cache_key, self.param_hash, pval, val)
        return pval, val

    def _compute_parcorr(self, X, Y, condition_set):
        """Compute partial correlation between X and Y given condition_set."""
        x = self.data_torch[:, X].unsqueeze(1)  # (T, 1)
        y = self.data_torch[:, Y].unsqueeze(1)  # (T, 1)

        # Standardize
        x = self._standardize(x)
        y = self._standardize(y)

        if len(condition_set) > 0:
            z = self.data_torch[:, condition_set]  # (T, d)
            z = self._standardize(z)
            # Residualize X and Y w.r.t. Z
            x_resid = self._residualize(z, x)
            y_resid = self._residualize(z, y)
        else:
            x_resid = x
            y_resid = y

        # Pearson correlation of residuals
        val = self._pearson_corr(x_resid.squeeze(), y_resid.squeeze())
        return val.item()

    def _standardize(self, v):
        """Standardize tensor along sample dimension (dim=0)."""
        mean = v.mean(dim=0, keepdim=True)
        std = v.std(dim=0, keepdim=True)
        # Avoid division by zero for constant columns
        std = torch.where(std == 0, torch.ones_like(std), std)
        return (v - mean) / std

    def _residualize(self, z, target):
        """Compute residuals of regressing target on z via OLS."""
        beta = torch.linalg.lstsq(z, target).solution
        return target - z @ beta

    def _pearson_corr(self, x, y):
        """Compute Pearson correlation between two 1-D tensors."""
        x_c = x - x.mean()
        y_c = y - y.mean()
        num = (x_c * y_c).sum()
        denom = torch.sqrt((x_c * x_c).sum() * (y_c * y_c).sum())
        if denom == 0:
            return torch.tensor(0.0, device=self.device)
        return num / denom

    def _compute_parcorr_on(self, data_clean, X, Y, condition_set):
        """Compute partial correlation on a subset of rows (for NaN handling)."""
        x = self._standardize(data_clean[:, X].unsqueeze(1))
        y = self._standardize(data_clean[:, Y].unsqueeze(1))
        if condition_set and len(condition_set) > 0:
            z = self._standardize(data_clean[:, condition_set])
            x = self._residualize(z, x)
            y = self._residualize(z, y)
        return self._pearson_corr(x.squeeze(), y.squeeze()).item()

    def _analytic_pvalue(self, val, condition_set, n_override=None):
        """Compute analytic p-value using Student's t-distribution."""
        T = n_override if n_override is not None else self.data_torch.shape[0]
        dim_z = (
            len(condition_set)
            if condition_set is not None and len(condition_set) > 0
            else 0
        )
        deg_f = T - dim_z - 2

        if deg_f < 1:
            return np.nan
        if abs(abs(val) - 1.0) <= sys.float_info.min:
            return 0.0

        # Clamp to avoid sqrt of negative
        val_clamped = np.clip(val, -1 + 1e-15, 1 - 1e-15)
        trafo_val = val_clamped * np.sqrt(deg_f / (1.0 - val_clamped**2))
        pval = stats.t.sf(np.abs(trafo_val), deg_f) * 2
        return pval

    def batch_test(self, tests, max_batch_size=2048):
        """Evaluate many CI tests in batched GPU operations.

        Parameters
        ----------
        tests : list of tuples
            Each element is (X, Y, condition_set) where X, Y are ints
            and condition_set is a list of ints (or None).
        max_batch_size : int, optional
            Maximum batch size per GPU kernel launch. Default 2048.

        Returns
        -------
        list of tuples
            Each element is (p_value, statistic) in the same order as input.
        """
        if not self.enable_caching:
            return self._batch_test_fast(tests)

        results = [None] * len(tests)

        # Normalize condition sets, compute hashes once, deduplicate
        hashes = [None] * len(tests)
        uncached_hashes = {}  # combined_hash -> first index
        uncached_indices = []
        dedup_map = {}  # combined_hash -> list of indices sharing this hash

        for i, (X, Y, cond) in enumerate(tests):
            if cond is None:
                cond = []
                tests[i] = (X, Y, cond)

            self.n_tests += 1
            raw_hash = self._get_array_hash(X, Y, cond)
            cache_key = f"{self.method}-{raw_hash}"
            combined_hash = (cache_key, self.param_hash)
            hashes[i] = combined_hash

            if combined_hash in self.pvalue_cache:
                results[i] = self.pvalue_cache[combined_hash]
            elif combined_hash in uncached_hashes:
                # Duplicate of another uncached test — will reuse its result
                dedup_map[combined_hash].append(i)
            else:
                uncached_hashes[combined_hash] = i
                uncached_indices.append(i)
                dedup_map[combined_hash] = [i]

        if not uncached_indices:
            return results

        # Group unique uncached tests by conditioning set size
        groups = defaultdict(list)
        for i in uncached_indices:
            X, Y, cond = tests[i]
            groups[len(cond)].append(i)

        T = self.data_torch.shape[0]

        if self._method == "precision":
            self._batch_test_precision(tests, groups, hashes, results, dedup_map, T)
        else:
            self._batch_test_ols(
                tests, groups, hashes, results, dedup_map, T, max_batch_size
            )

        return results

    def _batch_test_fast(self, tests, max_batch_size=50_000):
        """Fast path: skip all SHA1 hashing, caching, and deduplication.

        For CI tests that are cheap to compute (e.g., the ParCorrGPU
        "precision" method), SHA1 hashing costs ~5 µs/test while the actual
        computation costs ~17 µs/test. With a typical cache hit rate of <10%,
        caching is net-negative — hashing overhead exceeds savings. This path
        groups tests by conditioning-set size and calls
        ``_batch_test_precision`` directly.
        """
        results = [None] * len(tests)
        groups = defaultdict(list)
        for i, (X, Y, cond) in enumerate(tests):
            if cond is None:
                cond = []
                tests[i] = (X, Y, cond)
            self.n_tests += 1
            self.n_actual_tests += 1
            groups[len(cond)].append(i)

        if not groups:
            return results

        T = self.data_torch.shape[0]
        dev = self._precision_device

        for d, indices in groups.items():
            k = d + 2
            for chunk_start in range(0, len(indices), max_batch_size):
                chunk_idx = indices[chunk_start : chunk_start + max_batch_size]
                B = len(chunk_idx)

                idx_array = torch.zeros(B, k, dtype=torch.long, device=dev)
                for b, idx in enumerate(chunk_idx):
                    X, Y, cond = tests[idx]
                    idx_array[b, 0] = X
                    idx_array[b, 1] = Y
                    for j, zj in enumerate(cond):
                        idx_array[b, 2 + j] = zj

                sub_cov = self._cov_matrix[idx_array[:, :, None], idx_array[:, None, :]]

                if k == 2:
                    rho = sub_cov[:, 0, 1] / torch.sqrt(
                        sub_cov[:, 0, 0] * sub_cov[:, 1, 1]
                    )
                else:
                    prec = torch.linalg.inv(sub_cov)
                    rho = -prec[:, 0, 1] / torch.sqrt(prec[:, 0, 0] * prec[:, 1, 1])

                vals_np = rho.cpu().numpy()
                vals_clamped = np.clip(vals_np, -1 + 1e-15, 1 - 1e-15)

                deg_f = T - d - 2
                if deg_f < 1:
                    pvals = np.full(B, np.nan)
                else:
                    trafo_vals = vals_clamped * np.sqrt(deg_f / (1.0 - vals_clamped**2))
                    pvals = stats.t.sf(np.abs(trafo_vals), deg_f) * 2
                    exact_one = np.abs(np.abs(vals_np) - 1.0) <= sys.float_info.min
                    pvals[exact_one] = 0.0

                for j, idx in enumerate(chunk_idx):
                    results[idx] = (float(pvals[j]), float(vals_clamped[j]))

        return results

    def _batch_test_precision(
        self, tests, groups, hashes, results, dedup_map, T, max_batch_size=50_000
    ):
        """Batch CI tests via precision matrix (covariance inversion).

        Uses CUDA when available (fast batched inv), CPU otherwise.
        MPS falls back to CPU due to kernel launch overhead on small matrices.

        Tests are processed in chunks of ``max_batch_size`` to bound memory
        usage (at d=50 scale-free, a single depth can produce 7M+ tests;
        processing them all at once would allocate multi-GB tensors).
        """
        dev = self._precision_device
        for d, indices in groups.items():
            k = d + 2  # X, Y, + conditioning set

            for chunk_start in range(0, len(indices), max_batch_size):
                chunk_idx = indices[chunk_start : chunk_start + max_batch_size]
                B = len(chunk_idx)

                idx_array = torch.zeros(B, k, dtype=torch.long, device=dev)
                for b, idx in enumerate(chunk_idx):
                    X, Y, cond = tests[idx]
                    idx_array[b, 0] = X
                    idx_array[b, 1] = Y
                    for j, zj in enumerate(cond):
                        idx_array[b, 2 + j] = zj

                # Extract sub-covariance matrices: (B, k, k)
                sub_cov = self._cov_matrix[idx_array[:, :, None], idx_array[:, None, :]]

                if k == 2:
                    rho = sub_cov[:, 0, 1] / torch.sqrt(
                        sub_cov[:, 0, 0] * sub_cov[:, 1, 1]
                    )
                else:
                    prec = torch.linalg.inv(sub_cov)
                    rho = -prec[:, 0, 1] / torch.sqrt(prec[:, 0, 0] * prec[:, 1, 1])

                vals_np = rho.cpu().numpy()
                vals_clamped = np.clip(vals_np, -1 + 1e-15, 1 - 1e-15)

                deg_f = T - d - 2
                if deg_f < 1:
                    pvals = np.full(B, np.nan)
                else:
                    trafo_vals = vals_clamped * np.sqrt(deg_f / (1.0 - vals_clamped**2))
                    pvals = stats.t.sf(np.abs(trafo_vals), deg_f) * 2
                    exact_one = np.abs(np.abs(vals_np) - 1.0) <= sys.float_info.min
                    pvals[exact_one] = 0.0

                for j, idx in enumerate(chunk_idx):
                    p, v = float(pvals[j]), float(vals_clamped[j])
                    result = (p, v)
                    combined_hash = hashes[idx]
                    self.pvalue_cache[combined_hash] = result
                    cache_key = combined_hash[0]
                    self.save_incremental_cache(cache_key, self.param_hash, p, v)

                    for dup_idx in dedup_map[combined_hash]:
                        results[dup_idx] = result
                        self.n_actual_tests += 1

    def _batch_test_ols(
        self, tests, groups, hashes, results, dedup_map, T, max_batch_size
    ):
        """Batch CI tests via OLS residualization (original approach)."""
        for d, indices in groups.items():
            for start in range(0, len(indices), max_batch_size):
                batch_idx = indices[start : start + max_batch_size]
                B = len(batch_idx)

                # Vectorized tensor construction via advanced indexing
                x_cols = torch.tensor(
                    [tests[idx][0] for idx in batch_idx], dtype=torch.long
                )
                y_cols = torch.tensor(
                    [tests[idx][1] for idx in batch_idx], dtype=torch.long
                )
                # (T, B) -> (B, T, 1)
                x_batch = self.data_torch[:, x_cols].T.unsqueeze(-1)
                y_batch = self.data_torch[:, y_cols].T.unsqueeze(-1)

                if d > 0:
                    # Flatten all conditioning columns, gather, reshape
                    z_flat = torch.tensor(
                        [c for idx in batch_idx for c in tests[idx][2]],
                        dtype=torch.long,
                    )
                    # (T, B*d) -> (B, d, T) -> (B, T, d)
                    z_batch = (
                        self.data_torch[:, z_flat].T.reshape(B, d, T).permute(0, 2, 1)
                    )

                # Standardize along T dimension (dim=1)
                x_batch = self._batch_standardize(x_batch)
                y_batch = self._batch_standardize(y_batch)

                if d > 0:
                    z_batch = self._batch_standardize(z_batch)
                    x_resid = self._batch_residualize(z_batch, x_batch)
                    y_resid = self._batch_residualize(z_batch, y_batch)
                else:
                    x_resid = x_batch
                    y_resid = y_batch

                # Batched Pearson correlation
                vals = self._batch_pearson_corr(
                    x_resid.squeeze(-1), y_resid.squeeze(-1)
                )

                # Vectorized p-value computation
                vals_np = vals.cpu().numpy()
                deg_f = T - d - 2
                if deg_f < 1:
                    pvals = np.full(B, np.nan)
                else:
                    vals_clamped = np.clip(vals_np, -1 + 1e-15, 1 - 1e-15)
                    trafo_vals = vals_clamped * np.sqrt(deg_f / (1.0 - vals_clamped**2))
                    pvals = stats.t.sf(np.abs(trafo_vals), deg_f) * 2
                    exact_one = np.abs(np.abs(vals_np) - 1.0) <= sys.float_info.min
                    pvals[exact_one] = 0.0

                # Store results, cache, and propagate to duplicates
                for j, idx in enumerate(batch_idx):
                    p, v = float(pvals[j]), float(vals_np[j])
                    result = (p, v)
                    combined_hash = hashes[idx]
                    self.pvalue_cache[combined_hash] = result
                    cache_key = combined_hash[0]
                    self.save_incremental_cache(cache_key, self.param_hash, p, v)

                    # Assign to this index and all its duplicates
                    for dup_idx in dedup_map[combined_hash]:
                        results[dup_idx] = result
                        self.n_actual_tests += 1

        return results

    def _batch_standardize(self, v):
        """Standardize batched tensor along sample dimension (dim=1).

        Parameters
        ----------
        v : torch.Tensor
            Shape (B, T, d)
        """
        mean = v.mean(dim=1, keepdim=True)
        std = v.std(dim=1, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std), std)
        return (v - mean) / std

    def _batch_residualize(self, z, target):
        """Batched OLS residualization.

        Parameters
        ----------
        z : torch.Tensor
            Shape (B, T, d)
        target : torch.Tensor
            Shape (B, T, 1)

        Returns
        -------
        torch.Tensor
            Residuals, shape (B, T, 1)
        """
        beta = torch.linalg.lstsq(z, target).solution  # (B, d, 1)
        return target - torch.bmm(z, beta)

    def _batch_pearson_corr(self, x, y):
        """Batched Pearson correlation.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, T)
        y : torch.Tensor
            Shape (B, T)

        Returns
        -------
        torch.Tensor
            Correlation values, shape (B,)
        """
        x_c = x - x.mean(dim=1, keepdim=True)
        y_c = y - y.mean(dim=1, keepdim=True)
        num = (x_c * y_c).sum(dim=1)
        denom = torch.sqrt((x_c**2).sum(dim=1) * (y_c**2).sum(dim=1))
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        return num / denom
