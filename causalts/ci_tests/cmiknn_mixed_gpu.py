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
    """Per-point k-th nearest Chebyshev neighbor with MSinf masking.

    Neighbors can only come from the point's own discrete stratum of size
    ``n_i``. A fractional ``k`` (< 1) is a fraction of that stratum,
    ``k_i = max(1, int(k * n_i))``, as it is a fraction of T without discrete
    columns; an integer ``k`` is used as is. Either way ``k_i`` is capped at the
    ``n_i - 1`` other members, and the estimator uses ``digamma(k_i)`` for that
    point. Points alone in their stratum get ``k_i = 0`` and are left out. Where
    the ``k_i``-th distance is 0 (ties), ``k_i`` becomes the number of tied
    points, as in the mixed estimator of Gao et al. (2017).

    Parameters
    ----------
    data_t : torch.Tensor (T, D)
    k : int or float
    disc_local : list[int]  — local column indices that are discrete

    Returns
    -------
    eps : torch.Tensor (T,)
        Distance to the ``k_i``-th neighbor (0 where ``k_i == 0``).
    k_i : torch.Tensor (T,)
        Neighbors used per point.
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
    others = torch.isfinite(dists).sum(dim=1)
    if k < 1:
        k_i = torch.minimum(
            (k * (others + 1).double()).floor().long().clamp(min=1), others
        )
    else:
        k_i = others.clamp(max=int(k))
    k_max = max(int(k_i.max()), 1)
    nearest, _ = dists.topk(k_max, dim=1, largest=False)  # ascending
    eps = nearest.gather(1, (k_i - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
    eps = torch.where(k_i > 0, eps, torch.zeros_like(eps))
    # Zero radius (ties, e.g. all columns discrete): the ball counts every tied
    # point, so k_i is the number of ties (Gao et al. 2017).
    zero = (k_i > 0) & (eps == 0)
    k_i = torch.where(zero, (dists == 0).sum(dim=1), k_i)
    return eps, k_i


def _msinf_cmi(data, eps_k, xz_local, yz_local, z_local, disc_xz, disc_yz, disc_z):
    """Frenzel-Pompe CMI from per-point radii and neighbor counts."""
    eps, k_i = eps_k
    # Strictly inside the k-th neighbor (Frenzel-Pompe); a relative shrink like
    # eps * (1 - 1e-9) is a no-op in float32. A zero radius stays 0 and counts
    # the ties, matching k_i from _msinf_knn.
    eps = torch.nextafter(eps, torch.zeros_like(eps))
    T = data.shape[0]
    k_xz = _msinf_count(data[:, xz_local], eps, disc_xz)
    k_yz = _msinf_count(data[:, yz_local], eps, disc_yz)
    k_z = (
        _msinf_count(data[:, z_local], eps, disc_z)
        if z_local
        else torch.full((T,), float(T), dtype=eps.dtype, device=data.device)
    )
    ok = k_i > 0
    if not bool(ok.any()):
        return 0.0
    terms = (
        torch.digamma(k_i[ok].to(eps.dtype))
        + torch.digamma(k_z[ok])
        - torch.digamma(k_xz[ok])
        - torch.digamma(k_yz[ok])
    )
    return terms.mean().item()


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
        self.discrete_cols = discrete_cols
        # Stores the global column sequence [X_cols, Y_cols, Z_cols] for the
        # current __call__ invocation; used by _cmi_estimate to remap subspaces
        self._current_global_cols = []
        self._query_disc = True

    @property
    def discrete_cols(self):
        return sorted(self._disc_global)

    @discrete_cols.setter
    def discrete_cols(self, cols):
        self._disc_global = set(int(c) for c in cols)

    def _get_array_hash(self, X, Y, Z):
        """Cache key: the columns plus which of them are discrete.

        The estimate depends on the discrete flags of the tested columns, so a
        result computed under another configuration (``discrete_cols`` or
        ``_disc_global`` changed on a live object) must not be reused. With no
        discrete column in the test the key is ``CMIknnGPU``'s: the estimate then
        is ``CMIknnGPU``'s too.
        """
        key = super()._get_array_hash(X, Y, Z)
        cols = [X] if isinstance(X, (int, np.integer)) else list(X)
        cols += [Y] if isinstance(Y, (int, np.integer)) else list(Y)
        cols += [] if Z is None else list(Z)
        disc = sorted({int(c) for c in cols if int(c) in self._disc_global})
        return f"{key}|disc{disc}" if disc else key

    def __call__(self, X, Y, condition_set=None):
        """Override to record the global column ordering before CMI estimation."""
        if condition_set is None:
            condition_set = []
        # With order_independent=True, canonical order first, so the columns
        # recorded here match the order CMIknnGPU.__call__ computes in (it
        # canonicalises the same way); a no-op otherwise.
        X, Y, condition_set = self._canonical_query(X, Y, condition_set)
        x_cols = [X] if isinstance(X, (int, np.integer)) else list(X)
        y_cols = [Y] if isinstance(Y, (int, np.integer)) else list(Y)
        z_cols = list(condition_set)
        self._current_global_cols = x_cols + y_cols + z_cols
        self._query_disc = any(
            c in self._disc_global for c in self._current_global_cols
        )
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
        if not self._query_disc:
            return super()._cmi_estimate(array, xyz)
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

        return _msinf_cmi(
            data,
            _msinf_knn(data, self.k, disc_full),
            xz_local,
            yz_local,
            z_local,
            disc_xz,
            disc_yz,
            disc_z,
        )

    def _cmi_from_gpu_tensor(self, data_gpu, xyz, knn, knn_t=None):
        """CMI from pre-uploaded tensor with MSinf (used by shuffle tests).

        ``knn`` and ``knn_t`` are unused: the estimator uses each point's own
        ``k_i`` (see ``_msinf_knn``).
        """
        if not self._query_disc:
            return super()._cmi_from_gpu_tensor(data_gpu, xyz, knn, knn_t)
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

        return _msinf_cmi(
            data_gpu,
            _msinf_knn(data_gpu, self.k, disc_full),
            xz_local,
            yz_local,
            z_local,
            disc_xz,
            disc_yz,
            disc_z,
        )

    # ------------------------------------------------------------------
    # Shuffle overrides: preserve original discrete values after transform
    # ------------------------------------------------------------------

    def _build_gpu_tensor(self, array):
        """Transform array but keep original values for discrete columns, then upload."""
        array = self._transform_preserve_discrete(array, self._current_global_cols)
        dtype = torch.float32 if self.device.type == "cuda" else torch.float64
        return torch.tensor(array.T, dtype=dtype, device=self.device)

    def _stratum_block_permutation(self, members, bl):
        """Shuffle a stratum's rows (in time order) by blocks of length bl.

        Blocks are shortened so the stratum has at least 4 of them; with fewer,
        the permutation barely moves the rows and the p-value goes to 1.
        """
        bl = max(1, min(bl, len(members) // 4))
        blocks = [members[k : k + bl] for k in range(0, len(members), bl)]
        order = self.random_state.permutation(len(blocks))
        return np.concatenate([blocks[b] for b in order])

    def _transform_z_rows(self, z_array):
        """Transform continuous Z rows for neighbor search, as CMIknnGPU does."""
        z_array = z_array.astype(np.float64)
        T = z_array.shape[1]
        if self.transform == "standardize":
            mean = z_array.mean(axis=1, keepdims=True)
            std = z_array.std(axis=1, keepdims=True)
            std[std == 0] = 1.0
            z_array = (z_array - mean) / std
        elif self.transform == "uniform":
            uniform = np.linspace(1.0 / T, 1.0, T)
            z_array = np.stack([np.interp(r, np.sort(r), uniform) for r in z_array])
        elif self.transform == "ranks":
            z_array = z_array.argsort(axis=1).argsort(axis=1).astype(np.float64)
        return z_array

    def _nn_shuffle(self, array, xyz, value, x_indices, z_indices):
        """Neighborhood-preserving shuffle that never moves X across Z strata.

        X is permuted only within the stratum of the discrete Z columns: by
        blocks of consecutive members of each stratum when every Z column is
        discrete (at least ``sig_blocklength`` long, and longer for a persistent
        X, so its serial dependence survives in the null), and among each point's ``shuffle_neighbors``
        nearest same-stratum neighbors in the continuous Z columns otherwise.
        """
        if not self._query_disc:
            return super()._nn_shuffle(array, xyz, value, x_indices, z_indices)
        _, T = array.shape
        knn = self._resolve_knn(T)
        dtype = torch.float32 if self.device.type == "cuda" else torch.float64

        data_gpu = self._build_gpu_tensor(array)

        gcols = self._current_global_cols
        disc_z = [r for r, zi in enumerate(z_indices) if gcols[zi] in self._disc_global]
        cont_z = [r for r in range(len(z_indices)) if r not in set(disc_z)]
        if disc_z:
            _, strata = np.unique(
                array[np.asarray(z_indices)[disc_z], :].T, axis=0, return_inverse=True
            )
            strata = np.asarray(strata).ravel()
        else:  # the discrete columns are in X or Y only
            strata = np.zeros(T, dtype=int)

        neighbors = None
        if cont_z:
            # Continuous Z transformed as in CMIknnGPU; other strata are infinitely
            # far away.
            z_array = self._transform_z_rows(array[np.asarray(z_indices)[cont_z], :])
            z_data = torch.tensor(z_array.T, dtype=dtype, device=self.device)
            z_dists = torch.cdist(z_data, z_data, p=float("inf"))
            same = torch.tensor(strata, device=self.device)
            z_dists[same.unsqueeze(0) != same.unsqueeze(1)] = float("inf")
            n_nb = min(self.shuffle_neighbors, T)
            nn_d, nn_idx = z_dists.topk(n_nb, dim=1, largest=False)
            neighbors = nn_idx.cpu().numpy().astype(np.int32)
            finite = torch.isfinite(nn_d).cpu().numpy()
            # A stratum smaller than shuffle_neighbors: cycle through its own
            # members (self is always among them) instead of crossing strata.
            for i in np.nonzero(~finite.all(axis=1))[0]:
                own = neighbors[i, finite[i]]
                neighbors[i] = np.resize(own, n_nb)
        groups = [np.nonzero(strata == g)[0] for g in np.unique(strata)]
        # Block length for the within-stratum shuffle: at least sig_blocklength,
        # longer for persistent X ((1 + r) / (1 - r) for lag-1 autocorrelation
        # r, the variance inflation of an AR(1) mean). Short blocks leave a
        # persistent X's serial dependence in the data but not in the null.
        r1 = 0.0
        for xi in x_indices:
            xs = array[xi]
            if xs.std() > 0:
                r1 = max(r1, abs(float(np.corrcoef(xs[:-1], xs[1:])[0, 1])))
        r1 = min(r1, 0.95)
        block_len = max(1, int(self.sig_blocklength), int(np.ceil((1 + r1) / (1 - r1))))

        null_dist = np.zeros(self.sig_samples)
        running_pvals = np.zeros(self.sig_samples)
        knn_t = torch.tensor(float(knn), device=self.device)

        sam = 0
        for sam in range(self.sig_samples):
            if neighbors is None:
                perm = np.arange(T)
                for g in groups:
                    perm[g] = self._stratum_block_permutation(g, block_len)
            else:
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
        if not self._query_disc:
            return super()._random_shuffle(array, xyz, value, x_indices)
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
