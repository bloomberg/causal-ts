# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import torch

from .cit_test import CIT_Base


class CMIknnGPU(CIT_Base):
    """GPU-accelerated Conditional Mutual Information test via k-NN estimator.

    Uses torch.cdist with Chebyshev (L-infinity) distance on GPU to replace
    scipy.spatial.cKDTree. Implements the Frenzel-Pompe (2007) estimator with
    neighborhood-preserving shuffle significance test (Runge 2018).

    Parameters
    ----------
    data : np.ndarray, shape (T, n_features)
        Input data array.
    device : str or None
        Torch device ('cuda' or 'cpu'). Auto-detected if None.
        MPS is not supported (no float64 or cdist p=inf on Apple Silicon).
    k : int or float, optional (default: 0.2)
        Number of nearest neighbors. If < 1, fraction of T.
    shuffle_neighbors : int, optional (default: 5)
        Neighbors in Z-space for shuffle surrogates.
    sig_samples : int, optional (default: 500)
        Number of shuffle surrogates for significance test.
    transform : str, optional (default: 'ranks')
        Data transform: 'ranks', 'standardize', 'uniform', or None.
    early_stop : bool, optional (default: True)
        Stop shuffle test early when significance is clear.
    target_quantile : float, optional (default: 0.05)
        Target significance level for early stopping.
    chunk_size : int, optional (default: 4096)
        Max samples before chunking distance computation.
    """

    def __init__(
        self,
        data,
        device=None,
        k=0.2,
        shuffle_neighbors=5,
        sig_samples=500,
        transform="ranks",
        early_stop=True,
        target_quantile=0.05,
        chunk_size=4096,
        sig_blocklength=3,
        **kwargs,
    ):
        super().__init__(data=data, method="cmiknn_gpu", **kwargs)

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
            # MPS excluded: no float64 support and cdist(p=inf) unsupported
        self.device = torch.device(device)

        self.k = k
        self.shuffle_neighbors = shuffle_neighbors
        self.sig_samples = sig_samples
        self.transform = transform
        self.early_stop = early_stop
        self.target_quantile = target_quantile
        self.chunk_size = chunk_size
        self.sig_blocklength = sig_blocklength

    def get_data(self):
        return self._data

    def set_data(self, d):
        self._data = d
        self._has_nan = bool(np.isnan(d).any()) if d is not None else False

    data = property(get_data, set_data)

    def __call__(self, X, Y, condition_set=None):
        self.n_tests += 1
        if condition_set is None:
            condition_set = []

        cache_key = self._get_array_hash(X, Y, condition_set)
        param_hash = f"cmiknn_gpu-k{self.k}"
        cached = self.pvalue_cache.get((cache_key, param_hash))
        if cached is not None:
            return cached[0], cached[1]

        data = self.data
        if self._has_nan:
            mask = self._pairwise_complete(X, Y, condition_set)
            if mask.sum() < self._min_samples:
                self.pvalue_cache[(cache_key, param_hash)] = [1.0, 0.0]
                return 1.0, 0.0
            data = self._data[mask]

        x = data[:, X]
        y = data[:, Y]
        z = data[:, condition_set] if len(condition_set) > 0 else None

        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if z is not None and z.ndim == 1:
            z = z.reshape(-1, 1)

        if z is not None:
            array = np.hstack([x, y, z]).T  # (dim, T)
            xyz = np.array(
                [0] * x.shape[1] + [1] * y.shape[1] + [2] * z.shape[1],
                dtype=np.int32,
            )
        else:
            array = np.hstack([x, y]).T  # (dim, T)
            xyz = np.array([0] * x.shape[1] + [1] * y.shape[1], dtype=np.int32)

        self.n_actual_tests += 1
        val = self._cmi_estimate(array, xyz)
        pval = self._shuffle_significance(array, xyz, val)

        self.pvalue_cache[(cache_key, param_hash)] = [pval, val]
        self.save_incremental_cache(cache_key, param_hash, pval, val)

        return pval, val

    def _resolve_knn(self, T):
        if self.k < 1:
            return max(1, int(self.k * T))
        return max(1, int(self.k))

    def _apply_transform(self, array):
        """Transform array in-place. array shape: (dim, T)."""
        dim, T = array.shape
        array = array.astype(np.float64)
        array += (
            1e-6
            * array.std(axis=1).reshape(dim, 1)
            * self.random_state.random(array.shape)
        )

        if self.transform == "standardize":
            mean = array.mean(axis=1, keepdims=True)
            std = array.std(axis=1, keepdims=True)
            std[std == 0] = 1.0
            array = (array - mean) / std
        elif self.transform == "uniform":
            for i in range(dim):
                row = array[i]
                sorted_row = np.sort(row)
                uniform = np.linspace(1.0 / T, 1.0, T)
                array[i] = np.interp(row, sorted_row, uniform)
        elif self.transform == "ranks":
            array = array.argsort(axis=1).argsort(axis=1).astype(np.float64)

        return array

    def _chebyshev_knn(self, data_t, k):
        """Find k-th nearest neighbor Chebyshev distance per point.

        Parameters
        ----------
        data_t : torch.Tensor, shape (T, D)
        k : int

        Returns
        -------
        eps : torch.Tensor, shape (T,)
        """
        T = data_t.shape[0]
        if T <= self.chunk_size:
            dists = torch.cdist(data_t, data_t, p=float("inf"))
            dists.fill_diagonal_(float("inf"))
            eps, _ = dists.kthvalue(k, dim=1)
        else:
            eps = torch.empty(T, device=data_t.device)
            for i in range(0, T, self.chunk_size):
                end = min(i + self.chunk_size, T)
                chunk = data_t[i:end]
                d = torch.cdist(chunk, data_t, p=float("inf"))
                d[:, i:end].fill_diagonal_(float("inf"))
                eps[i:end], _ = d.kthvalue(k, dim=1)
        return eps

    def _count_neighbors_in_radius(self, data_t, eps):
        """Count neighbors with Chebyshev distance <= eps[i] for each point.

        Uses <= to match scipy cKDTree.query_ball_point which includes the
        point itself (self-distance = 0 <= eps always), so counts include self.

        Parameters
        ----------
        data_t : torch.Tensor, shape (T, D)
        eps : torch.Tensor, shape (T,)

        Returns
        -------
        counts : torch.Tensor, shape (T,), dtype float
        """
        T = data_t.shape[0]
        if T <= self.chunk_size:
            dists = torch.cdist(data_t, data_t, p=float("inf"))
            counts = (dists <= eps.unsqueeze(1)).sum(dim=1).float()
        else:
            counts = torch.zeros(T, device=data_t.device)
            for i in range(0, T, self.chunk_size):
                end = min(i + self.chunk_size, T)
                chunk = data_t[i:end]
                d = torch.cdist(chunk, data_t, p=float("inf"))
                counts[i:end] = (d <= eps[i:end].unsqueeze(1)).sum(dim=1).float()
        return counts

    def _cmi_estimate(self, array, xyz):
        """Compute CMI using Frenzel-Pompe k-NN estimator on GPU.

        Parameters
        ----------
        array : np.ndarray, shape (dim, T)
        xyz : np.ndarray, shape (dim,)

        Returns
        -------
        val : float
        """
        _, T = array.shape
        knn = self._resolve_knn(T)

        array = self._apply_transform(array.copy())
        # float32 for GPU distance ops (CUDA float64 throughput is ~1/32 of float32)
        dtype = torch.float32 if self.device.type == "cuda" else torch.float64
        data = torch.tensor(array.T, dtype=dtype, device=self.device)

        x_idx = np.where(xyz == 0)[0]
        y_idx = np.where(xyz == 1)[0]
        z_idx = np.where(xyz == 2)[0]

        xz_idx = np.concatenate([x_idx, z_idx])
        yz_idx = np.concatenate([y_idx, z_idx])

        eps = self._chebyshev_knn(data, knn)
        eps = eps * (1 - 1e-9)

        k_xz = self._count_neighbors_in_radius(data[:, xz_idx], eps)
        k_yz = self._count_neighbors_in_radius(data[:, yz_idx], eps)

        if len(z_idx) > 0:
            k_z = self._count_neighbors_in_radius(data[:, z_idx], eps)
        else:
            k_z = torch.full((T,), float(T), device=self.device)

        val = torch.digamma(torch.tensor(float(knn), device=self.device))
        val = (
            val
            + (torch.digamma(k_z) - torch.digamma(k_xz) - torch.digamma(k_yz)).mean()
        )

        return val.item()

    def _shuffle_significance(self, array, xyz, value):
        """Neighborhood-preserving shuffle test with early stopping.

        Parameters
        ----------
        array : np.ndarray, shape (dim, T)
        xyz : np.ndarray, shape (dim,)
        value : float

        Returns
        -------
        pval : float
        """
        _, T = array.shape
        x_indices = np.where(xyz == 0)[0]
        z_indices = np.where(xyz == 2)[0]

        if len(z_indices) > 0 and self.shuffle_neighbors < T:
            return self._nn_shuffle(array, xyz, value, x_indices, z_indices)
        else:
            return self._random_shuffle(array, xyz, value, x_indices)

    def _cmi_from_gpu_tensor(self, data_gpu, xyz, knn, knn_t=None):
        """Compute CMI directly from a GPU tensor (no CPU round-trip)."""
        T = data_gpu.shape[0]
        x_idx = np.where(xyz == 0)[0]
        y_idx = np.where(xyz == 1)[0]
        z_idx = np.where(xyz == 2)[0]
        xz_idx = np.concatenate([x_idx, z_idx])
        yz_idx = np.concatenate([y_idx, z_idx])

        eps = self._chebyshev_knn(data_gpu, knn)
        eps = eps * (1 - 1e-9)

        k_xz = self._count_neighbors_in_radius(data_gpu[:, xz_idx], eps)
        k_yz = self._count_neighbors_in_radius(data_gpu[:, yz_idx], eps)

        if len(z_idx) > 0:
            k_z = self._count_neighbors_in_radius(data_gpu[:, z_idx], eps)
        else:
            k_z = torch.full((T,), float(T), device=self.device)

        if knn_t is None:
            knn_t = torch.tensor(float(knn), device=self.device)
        val = (
            torch.digamma(knn_t)
            + (torch.digamma(k_z) - torch.digamma(k_xz) - torch.digamma(k_yz)).mean()
        )
        return val.item()

    def _nn_shuffle(self, array, xyz, value, x_indices, z_indices):
        """Neighborhood-preserving shuffle (Runge 2018)."""
        _, T = array.shape
        knn = self._resolve_knn(T)
        dtype = torch.float32 if self.device.type == "cuda" else torch.float64

        # Pre-transform and upload ONCE — reuse across all shuffle iterations
        array_t = self._apply_transform(array.copy())
        data_gpu = torch.tensor(array_t.T, dtype=dtype, device=self.device)  # (T, D)

        # Build Z-space neighbors ONCE on GPU
        z_array = array[z_indices, :].copy()
        if self.transform == "standardize":
            mean = z_array.mean(axis=1, keepdims=True)
            std = z_array.std(axis=1, keepdims=True)
            std[std == 0] = 1.0
            z_array = (z_array - mean) / std
        elif self.transform == "uniform":
            for i in range(len(z_indices)):
                row = z_array[i]
                sorted_row = np.sort(row)
                uniform = np.linspace(1.0 / T, 1.0, T)
                z_array[i] = np.interp(row, sorted_row, uniform)
        elif self.transform == "ranks":
            z_array = z_array.argsort(axis=1).argsort(axis=1).astype(np.float64)

        z_data = torch.tensor(z_array.T, dtype=dtype, device=self.device)
        z_dists = torch.cdist(z_data, z_data, p=float("inf"))
        # Include self to match cKDTree.query behavior
        _, nn_idx = z_dists.topk(self.shuffle_neighbors, dim=1, largest=False)
        neighbors = nn_idx.cpu().numpy().astype(np.int32)

        null_dist = np.zeros(self.sig_samples)
        running_pvals = np.zeros(self.sig_samples)
        knn_t = torch.tensor(float(knn), device=self.device)

        sam = 0
        for sam in range(self.sig_samples):
            order = self.random_state.permutation(T).astype(np.int32)
            for i in range(T):
                self.random_state.shuffle(neighbors[i])
            perm = self._restricted_permutation(T, neighbors, order)
            perm_t = torch.tensor(perm, dtype=torch.long, device=self.device)
            data_shuf = data_gpu.clone()
            for i in x_indices:
                data_shuf[:, i] = data_gpu[perm_t, i]
            null_dist[sam] = self._cmi_from_gpu_tensor(data_shuf, xyz, knn, knn_t)

            if self.early_stop:
                running_pvals[sam] = (null_dist[: sam + 1] >= value).mean()
            if (
                self.early_stop
                and sam > 20
                and sam % 20 == 0
                and (
                    running_pvals[sam] > (self.target_quantile + 0.1)
                    or np.std(running_pvals[:sam]) == 0
                )
            ):
                std_ = np.std(null_dist[:sam])
                if std_ > 0:
                    q_ = np.quantile(null_dist[:sam], self.target_quantile)
                    z_ = np.abs(value - q_) / std_
                    if z_ > 5:
                        break
                std_pvals = np.std(running_pvals[:sam])
                z_pval = (np.median(running_pvals[:sam]) - self.target_quantile) / max(
                    std_pvals, 0.0001
                )
                if z_pval >= 3:
                    break

        null_dist = null_dist[: sam + 1]
        pval = (null_dist >= value).mean()
        return float(pval)

    def _block_permutation(self, T):
        """Block-shuffle permutation of length T using sig_blocklength.

        Shuffles contiguous blocks rather than individual time steps, preserving
        short-range temporal autocorrelation in the null distribution. Matches
        the tigramite sig_blocklength=3 block-shuffle used by CMIknn.
        """
        bl = max(1, self.sig_blocklength)
        n_blocks = T // bl
        remainder = T - n_blocks * bl
        block_order = self.random_state.permutation(n_blocks)
        perm = np.concatenate([np.arange(b * bl, (b + 1) * bl) for b in block_order])
        if remainder > 0:
            perm = np.concatenate([perm, np.arange(n_blocks * bl, T)])
        return perm.astype(np.int32)

    def _random_shuffle(self, array, xyz, value, x_indices):
        """Block shuffle when Z is empty, preserving temporal autocorrelation.

        Matches tigramite CMIknn: shuffles raw X in block-permuted order then
        re-applies the rank transform per iteration. This gives the correct null
        distribution for autocorrelated time series, unlike shuffling pre-computed
        ranks which under-estimates null variance.
        """
        _, T = array.shape
        knn = self._resolve_knn(T)

        null_dist = np.zeros(self.sig_samples)
        running_pvals = np.zeros(self.sig_samples)

        sam = 0
        for sam in range(self.sig_samples):
            perm = self._block_permutation(T)
            # Shuffle raw X values, then re-rank the full array (matches tigramite)
            array_shuf = array.copy()
            for i in x_indices:
                array_shuf[i] = array[i, perm]
            array_t = self._apply_transform(array_shuf)
            dtype = torch.float32 if self.device.type == "cuda" else torch.float64
            data_gpu = torch.tensor(array_t.T, dtype=dtype, device=self.device)
            null_dist[sam] = self._cmi_from_gpu_tensor(data_gpu, xyz, knn)

            if self.early_stop:
                running_pvals[sam] = (null_dist[: sam + 1] >= value).mean()
            if (
                self.early_stop
                and sam > 20
                and sam % 20 == 0
                and running_pvals[sam] > (self.target_quantile + 0.1)
            ):
                std_ = np.std(null_dist[:sam])
                if std_ > 0:
                    q_ = np.quantile(null_dist[:sam], self.target_quantile)
                    z_ = np.abs(value - q_) / std_
                    if z_ > 5:
                        break

        null_dist = null_dist[: sam + 1]
        pval = (null_dist >= value).mean()
        return float(pval)

    @staticmethod
    def _restricted_permutation(T, neighbors, order):
        """Create permutation avoiding duplicate indices."""
        perm = np.zeros(T, dtype=np.int32)
        used = set()
        sn = neighbors.shape[1]
        for sample_index in order:
            m = 0
            use = neighbors[sample_index, m]
            while use in used and m < sn - 1:
                m += 1
                use = neighbors[sample_index, m]
            perm[sample_index] = use
            used.add(int(use))
        return perm
