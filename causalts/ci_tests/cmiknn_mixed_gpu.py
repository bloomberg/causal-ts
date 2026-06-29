# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""CMIknnMixedGPU: GPU-accelerated CMI k-NN for mixed discrete-continuous data.

Implements the MSinf distance strategy (Popescu, Gerhardus, Runge 2023,
arXiv:2310.11132): pairs where any discrete dimension disagrees get infinite
distance, excluding them from all k-NN balls. Within the set of observations
sharing the same discrete values, standard Chebyshev (L-inf) distance over
continuous dimensions is used. This is exact stratification at the k-NN level.
"""

import numpy as np
import torch

from .cmiknn_gpu import CMIknnGPU


def _msinf_knn(data_t, k, disc_local):
    """k-th nearest Chebyshev neighbor with MSinf masking for discrete columns.

    Parameters
    ----------
    data_t : torch.Tensor (T, D)
    k : int
    disc_local : list[int]  — local column indices that are discrete
    """
    T, D = data_t.shape
    cont_local = [i for i in range(D) if i not in set(disc_local)]

    dists = (
        torch.cdist(data_t[:, cont_local], data_t[:, cont_local], p=float("inf"))
        if cont_local
        else torch.zeros(T, T, dtype=data_t.dtype, device=data_t.device)
    )
    for col in disc_local:
        vals = data_t[:, col]
        dists[vals.unsqueeze(0) != vals.unsqueeze(1)] = float("inf")

    dists.fill_diagonal_(float("inf"))
    eps, _ = dists.kthvalue(k, dim=1)
    return eps


def _msinf_count(data_t, eps, disc_local):
    """Count neighbors within radius eps with MSinf masking.

    Parameters
    ----------
    data_t : torch.Tensor (T, D)
    eps : torch.Tensor (T,)
    disc_local : list[int]
    """
    T, D = data_t.shape
    cont_local = [i for i in range(D) if i not in set(disc_local)]

    dists = (
        torch.cdist(data_t[:, cont_local], data_t[:, cont_local], p=float("inf"))
        if cont_local
        else torch.zeros(T, T, dtype=data_t.dtype, device=data_t.device)
    )
    for col in disc_local:
        vals = data_t[:, col]
        dists[vals.unsqueeze(0) != vals.unsqueeze(1)] = float("inf")

    return (dists <= eps.unsqueeze(1)).sum(dim=1).float()


def _subspace_disc(global_sequence, disc_global_set):
    """Local positions in global_sequence that are discrete.

    Parameters
    ----------
    global_sequence : list[int]
        Ordered global column indices forming a subspace (e.g. xz_global).
    disc_global_set : set[int]
        Set of global column indices that are discrete.

    Returns list[int] of local (0-based) positions within global_sequence.
    """
    return [i for i, g in enumerate(global_sequence) if g in disc_global_set]


class CMIknnMixedGPU(CMIknnGPU):
    """GPU CMI k-NN with MSinf distance for mixed discrete-continuous data.

    Inherits all parameters and behaviour from CMIknnGPU. Overrides
    `_cmi_estimate` and `_cmi_from_gpu_tensor` to apply the MSinf rule using
    the global→local mapping established in `__call__`.

    Parameters
    ----------
    data : np.ndarray, shape (T, d)
        Lag-embedded data matrix.
    discrete_cols : list[int]
        Column indices (into data) that are discrete. Use
        `stratified_cit.compute_discrete_indices()` to convert from
        original DataFrame column names.
    device : str or None
        'cuda' or 'cpu'. MPS not supported. Auto-detects CUDA.
    (all other CMIknnGPU parameters apply)
    """

    def __init__(self, data, discrete_cols, device=None, **kwargs):
        super().__init__(data=data, device=device, **kwargs)
        self.discrete_cols = list(discrete_cols)
        self._disc_global = set(discrete_cols)
        # Stores the global column sequence [X_cols, Y_cols, Z_cols] for the
        # current __call__ invocation; used by _cmi_estimate to remap subspaces
        self._current_global_cols = []

    def __call__(self, X, Y, condition_set=None):
        """Override to record the global column ordering before CMI estimation."""
        if condition_set is None:
            condition_set = []
        x_cols = [X] if isinstance(X, (int, np.integer)) else list(X)
        y_cols = [Y] if isinstance(Y, (int, np.integer)) else list(Y)
        z_cols = list(condition_set)
        self._current_global_cols = x_cols + y_cols + z_cols
        return super().__call__(X, Y, condition_set)

    # ------------------------------------------------------------------
    # MSinf-aware CMI estimation
    # ------------------------------------------------------------------

    def _transform_preserve_discrete(self, array, gcols):
        """Apply transform but restore original values for discrete columns.

        After rank-transforming, all values in a discrete column become unique
        floats — breaking MSinf equality checks. We restore the original integer
        values for discrete columns so that equality comparisons work correctly.
        """
        array_orig = array.copy()
        array = self._apply_transform(array.copy())
        for local_i, global_i in enumerate(gcols):
            if global_i in self._disc_global:
                array[local_i] = array_orig[local_i]
        return array

    def _cmi_estimate(self, array, xyz):
        """Frenzel-Pompe CMI with MSinf distance for discrete columns."""
        _, T = array.shape
        knn = self._resolve_knn(T)

        gcols = self._current_global_cols
        array = self._transform_preserve_discrete(array, gcols)

        dtype = torch.float32 if self.device.type == "cuda" else torch.float64
        data = torch.tensor(array.T, dtype=dtype, device=self.device)

        x_local = list(np.where(xyz == 0)[0])
        y_local = list(np.where(xyz == 1)[0])
        z_local = list(np.where(xyz == 2)[0])
        xz_local = x_local + z_local
        yz_local = y_local + z_local
        full_local = list(range(len(xyz)))

        # Map local positions to global column indices using __call__'s record
        gcols = self._current_global_cols
        disc = self._disc_global

        disc_full = _subspace_disc([gcols[i] for i in full_local], disc)
        disc_xz = _subspace_disc([gcols[i] for i in xz_local], disc)
        disc_yz = _subspace_disc([gcols[i] for i in yz_local], disc)
        disc_z = _subspace_disc([gcols[i] for i in z_local], disc)

        eps = _msinf_knn(data, knn, disc_full) * (1 - 1e-9)

        k_xz = _msinf_count(data[:, xz_local], eps, disc_xz)
        k_yz = _msinf_count(data[:, yz_local], eps, disc_yz)
        k_z = (
            _msinf_count(data[:, z_local], eps, disc_z)
            if z_local
            else torch.full((T,), float(T), device=self.device)
        )

        val = torch.digamma(torch.tensor(float(knn), device=self.device))
        val = (
            val
            + (torch.digamma(k_z) - torch.digamma(k_xz) - torch.digamma(k_yz)).mean()
        )
        return val.item()

    def _cmi_from_gpu_tensor(self, data_gpu, xyz, knn, knn_t=None):
        """CMI from pre-uploaded tensor with MSinf (used by shuffle tests)."""
        T = data_gpu.shape[0]

        x_local = list(np.where(xyz == 0)[0])
        y_local = list(np.where(xyz == 1)[0])
        z_local = list(np.where(xyz == 2)[0])
        xz_local = x_local + z_local
        yz_local = y_local + z_local
        full_local = list(range(len(xyz)))

        gcols = self._current_global_cols
        disc = self._disc_global

        disc_full = _subspace_disc([gcols[i] for i in full_local], disc)
        disc_xz = _subspace_disc([gcols[i] for i in xz_local], disc)
        disc_yz = _subspace_disc([gcols[i] for i in yz_local], disc)
        disc_z = _subspace_disc([gcols[i] for i in z_local], disc)

        eps = _msinf_knn(data_gpu, knn, disc_full) * (1 - 1e-9)

        k_xz = _msinf_count(data_gpu[:, xz_local], eps, disc_xz)
        k_yz = _msinf_count(data_gpu[:, yz_local], eps, disc_yz)
        k_z = (
            _msinf_count(data_gpu[:, z_local], eps, disc_z)
            if z_local
            else torch.full((T,), float(T), device=self.device)
        )

        if knn_t is None:
            knn_t = torch.tensor(float(knn), device=self.device)
        val = (
            torch.digamma(knn_t)
            + (torch.digamma(k_z) - torch.digamma(k_xz) - torch.digamma(k_yz)).mean()
        )
        return val.item()

    # ------------------------------------------------------------------
    # Shuffle overrides: preserve original discrete values after transform
    # ------------------------------------------------------------------

    def _build_gpu_tensor(self, array):
        """Transform array but keep original values for discrete columns, then upload."""
        array = self._transform_preserve_discrete(array, self._current_global_cols)
        dtype = torch.float32 if self.device.type == "cuda" else torch.float64
        return torch.tensor(array.T, dtype=dtype, device=self.device)

    def _nn_shuffle(self, array, xyz, value, x_indices, z_indices):
        """Override: build data_gpu with discrete columns un-transformed."""
        _, T = array.shape
        knn = self._resolve_knn(T)
        dtype = torch.float32 if self.device.type == "cuda" else torch.float64

        data_gpu = self._build_gpu_tensor(array)

        # Build Z-space neighbors: use ranks for continuous Z, original for discrete Z
        z_array = array[z_indices, :].copy()
        gcols = self._current_global_cols
        for local_i_in_full, z_global in zip(z_indices, [gcols[i] for i in z_indices]):
            row_idx = list(z_indices).index(local_i_in_full)
            if z_global not in self._disc_global:
                if self.transform == "ranks":
                    z_array[row_idx] = (
                        z_array[row_idx].argsort().argsort().astype(np.float64)
                    )
                # else keep as-is (standardize/uniform handled below)

        z_data = torch.tensor(z_array.T, dtype=dtype, device=self.device)

        # For MSinf: Z neighbor finding uses MSinf on discrete Z dims
        disc_z_local = [
            i for i, zi in enumerate(z_indices) if gcols[zi] in self._disc_global
        ]
        if disc_z_local:
            z_dists = torch.zeros(T, T, dtype=dtype, device=self.device)
            cont_z_local = [
                i for i in range(len(z_indices)) if i not in set(disc_z_local)
            ]
            if cont_z_local:
                z_dists = torch.cdist(
                    z_data[:, cont_z_local], z_data[:, cont_z_local], p=float("inf")
                )
            for col in disc_z_local:
                vals = z_data[:, col]
                z_dists[vals.unsqueeze(0) != vals.unsqueeze(1)] = float("inf")
        else:
            z_dists = torch.cdist(z_data, z_data, p=float("inf"))

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
                and running_pvals[sam] > (self.target_quantile + 0.1)
            ):
                std_ = np.std(null_dist[:sam])
                if std_ > 0:
                    q_ = np.quantile(null_dist[:sam], self.target_quantile)
                    z_ = np.abs(value - q_) / std_
                    if z_ > 5:
                        break

        null_dist = null_dist[: sam + 1]
        return float((null_dist >= value).mean())

    def _random_shuffle(self, array, xyz, value, x_indices):
        """Override: build data_gpu with discrete columns un-transformed."""
        _, T = array.shape
        knn = self._resolve_knn(T)

        null_dist = np.zeros(self.sig_samples)
        running_pvals = np.zeros(self.sig_samples)

        sam = 0
        for sam in range(self.sig_samples):
            perm = self._block_permutation(T)
            array_shuf = array.copy()
            for i in x_indices:
                array_shuf[i] = array[i, perm]
            data_gpu = self._build_gpu_tensor(array_shuf)
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
        return float((null_dist >= value).mean())
