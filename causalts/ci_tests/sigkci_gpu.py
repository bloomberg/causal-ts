# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# SigKCI: Signature Kernel Conditional Independence Test
# Based on: Manten, Casolo, Ferrucci, Mogensen, Salvi, Kilbertus,
# "Signature Kernel Conditional Independence Tests in Causal Discovery
# for Stochastic Processes", arXiv:2402.18477, 2024.
#
# Uses the truncated signature kernel via PDE discretization
# (Salvi et al., "The Signature Kernel is the solution of a Goursat PDE",
# SIAM J. Math. Data Sci., 2021).

import numpy as np
import torch
from scipy.stats import gamma as gamma_distr

from .cit_test import CIT_Base
from .kci_gpu import apply_H, normalize_gpu

try:
    import sigkernel as _sigkernel

    SIGKERNEL_AVAILABLE = True
except ImportError:
    SIGKERNEL_AVAILABLE = False


def _time_delay_embedding(x, dim=3, lag=1):
    """Embed a 1D signal into a path via time-delay embedding.

    Args:
        x: (T,) or (T, d) tensor
        dim: embedding dimension
        lag: delay between coordinates

    Returns:
        (T - (dim-1)*lag, dim*d) tensor representing the path
    """
    if x.dim() == 1:
        x = x.unsqueeze(1)
    T, d = x.shape
    n_points = T - (dim - 1) * lag
    if n_points < 2:
        return x[:2].unsqueeze(0)
    cols = []
    for i in range(dim):
        cols.append(x[i * lag : i * lag + n_points])
    return torch.cat(cols, dim=1)


def _lead_lag_transform(x):
    """Lead-lag transform: embed 1D signal into 2D path.

    Maps (x_0, ..., x_T) to path in R^2:
    (x_0, x_0), (x_1, x_0), (x_1, x_1), (x_2, x_1), ...
    """
    if x.dim() == 1:
        x = x.unsqueeze(1)
    T, d = x.shape
    path = torch.zeros(2 * T - 1, 2 * d, device=x.device)
    for i in range(T):
        path[2 * i, :d] = x[i]
        path[2 * i, d:] = x[i] if i == 0 else x[i - 1]
        if i < T - 1:
            path[2 * i + 1, :d] = x[i]
            path[2 * i + 1, d:] = x[i]
    return path


def _truncated_signature_kernel_gram(paths_x, paths_y, sigma=1.0, depth=4):
    """Compute signature kernel Gram matrix via PDE discretization.

    Uses the Goursat PDE approach to compute the untruncated signature
    kernel k_sig(x, y) = <phi(x), phi(y)> where phi is the signature
    feature map.

    Args:
        paths_x: (n, L_x, d) batch of paths
        paths_y: (m, L_y, d) batch of paths
        sigma: RBF bandwidth for the static kernel
        depth: dyadic order for PDE solver (higher = more accurate)

    Returns:
        (n, m) Gram matrix
    """
    n = paths_x.shape[0]
    m = paths_y.shape[0]
    device = paths_x.device

    K = torch.zeros(n, m, device=device)

    for i in range(n):
        for j in range(m):
            K[i, j] = _sig_kernel_pde(paths_x[i], paths_y[j], sigma, depth)

    return K


def _sig_kernel_pde(x, y, sigma=1.0, dyadic_order=0):
    """Compute signature kernel between two paths via PDE solver.

    Solves the Goursat PDE:
        d²K/ds dt = <dx_s, dy_t> * K(s,t)
    with boundary conditions K(0,t) = K(s,0) = 1.

    Args:
        x: (L_x, d) path
        y: (L_y, d) path
        sigma: bandwidth for inner product kernel
        dyadic_order: refinement level (0 = no refinement)

    Returns:
        scalar kernel value
    """
    L_x, d = x.shape
    L_y = y.shape[0]

    M = 2**dyadic_order * (L_x - 1)
    N = 2**dyadic_order * (L_y - 1)

    if dyadic_order > 0:
        inc_x = _refine_increments(x, dyadic_order)
        inc_y = _refine_increments(y, dyadic_order)
    else:
        inc_x = x[1:] - x[:-1]
        inc_y = y[1:] - y[:-1]

    K_pde = torch.ones(M + 1, N + 1, device=x.device)

    for i in range(1, M + 1):
        for j in range(1, N + 1):
            inner = (inc_x[i - 1] * inc_y[j - 1]).sum() / (sigma**2)
            K_pde[i, j] = (
                K_pde[i - 1, j]
                + K_pde[i, j - 1]
                - K_pde[i - 1, j - 1]
                + K_pde[i - 1, j - 1] * inner
            )

    return K_pde[M, N]


def _refine_increments(path, dyadic_order):
    """Refine path increments by linear interpolation."""
    inc = path[1:] - path[:-1]
    for _ in range(dyadic_order):
        inc = inc.repeat_interleave(2, dim=0) / 2.0
    return inc


def _compute_sig_gram_batch(paths, sigma=1.0, dyadic_order=0):
    """Compute symmetric Gram matrix for a batch of paths (loop version)."""
    n = paths.shape[0]
    device = paths.device
    K = torch.ones(n, n, device=device)

    for i in range(n):
        for j in range(i, n):
            val = _sig_kernel_pde(paths[i], paths[j], sigma, dyadic_order)
            K[i, j] = val
            K[j, i] = val

    return K


def _compute_sig_gram_vectorized(paths, sigma=1.0, dyadic_order=0):
    """Compute symmetric Gram matrix via vectorized PDE solver.

    Solves the Goursat PDE for all n² path pairs simultaneously by
    batching inner products and propagating the PDE grid in parallel.
    """
    n, L, d = paths.shape
    device = paths.device

    inc = paths[:, 1:] - paths[:, :-1]
    if dyadic_order > 0:
        for _ in range(dyadic_order):
            inc = inc.repeat_interleave(2, dim=1) / 2.0
    M = inc.shape[1]

    # Precompute all pairwise inner products: (n, n, M, M)
    # inner[a, b, i, j] = <inc_a[i], inc_b[j]> / sigma²
    inner = torch.einsum("aid,bjd->abij", inc, inc) / (sigma**2)

    # Solve PDE for all pairs simultaneously
    # K_pde[a, b, i, j] with boundary K[:,:,0,:] = K[:,:,:,0] = 1
    K_pde = torch.ones(n, n, M + 1, M + 1, device=device)

    for i in range(1, M + 1):
        for j in range(1, M + 1):
            K_pde[:, :, i, j] = (
                K_pde[:, :, i - 1, j]
                + K_pde[:, :, i, j - 1]
                - K_pde[:, :, i - 1, j - 1]
                + K_pde[:, :, i - 1, j - 1] * inner[:, :, i - 1, j - 1]
            )

    return K_pde[:, :, M, M]


def _gamma_or_hbe_pval(stat, Kx, Ky, T, approx):
    """Compute p-value for unconditional HSIC statistic."""
    if approx == "perm":
        raise ValueError("Use sigkci_uind directly for permutation tests")
    mean_appr = torch.trace(Kx) * torch.trace(Ky) / T
    var_appr = (
        2 * torch.einsum("ij,ji->", Kx, Kx) * torch.einsum("ij,ji->", Ky, Ky) / (T**2)
    )
    var_appr = torch.clamp(var_appr, min=1e-20)
    mean_appr = torch.where(
        mean_appr.abs() < 1e-20,
        torch.tensor(1e-20, device=stat.device),
        mean_appr,
    )
    k_appr = (mean_appr**2 / var_appr).cpu()
    theta_appr = (var_appr / mean_appr).cpu()
    p_val = 1 - gamma_distr.cdf(stat.cpu(), a=k_appr, scale=theta_appr)
    return p_val, stat.cpu().item()


def _cond_pval(stat, Kxz_r, Ky_r, T, approx):
    """Compute p-value for conditional HSIC statistic via eigendecomposition."""
    thresh = 1e-5
    eig_Kxz, eivx = torch.linalg.eigh((Kxz_r + Kxz_r.T) / 2)
    eig_Ky, eivy = torch.linalg.eigh((Ky_r + Ky_r.T) / 2)
    x_ix = (eig_Kxz > eig_Kxz.max() * thresh).nonzero().ravel()
    y_ix = (eig_Ky > eig_Ky.max() * thresh).nonzero().ravel()
    eig_Kxz = eig_Kxz[x_ix]
    eivx = eivx[:, x_ix]
    eig_Ky = eig_Ky[y_ix]
    eivy = eivy[:, y_ix]
    eiv_prodx = eivx * torch.sqrt(eig_Kxz)
    eiv_prody = eivy * torch.sqrt(eig_Ky)
    uu = eiv_prodx.unsqueeze(-1) * eiv_prody.unsqueeze(-2)
    uu = uu.reshape(T, -1)
    if uu.shape[1] > T:
        uu_prod = uu @ uu.T
    else:
        uu_prod = uu.T @ uu
    eig_uu = torch.linalg.eigvalsh(uu_prod)
    eig_uu = eig_uu[eig_uu > eig_uu.max() * thresh]

    if approx == "gamma":
        mean_appr = eig_uu.sum()
        var_appr = 2 * (eig_uu**2).sum()
        k_appr = (mean_appr**2 / var_appr).cpu()
        theta_appr = (var_appr / mean_appr).cpu()
        p_val = 1 - gamma_distr.cdf(stat.cpu(), a=k_appr, scale=theta_appr)
    else:
        K1 = eig_uu.sum()
        K2 = 2 * (eig_uu**2).sum()
        K3 = 8 * (eig_uu**3).sum()
        nu = 8 * (K2**3) / (K3**2)
        k = nu / 2
        x_stat = ((2 * nu / K2) ** 0.5) * (stat - K1) + nu
        p_val = 1 - gamma_distr.cdf(x_stat.cpu(), a=k.cpu(), scale=2)
    return p_val, stat.cpu().item()


def _rbf_gram(x, sigma=1.0):
    """RBF kernel Gram matrix for flat (non-path) data."""
    if x.dim() == 3:
        x = x.squeeze(1)
    w = 1.0 / (sigma**2)
    return torch.exp(-w * torch.cdist(x, x, p=2) ** 2)


def _compute_sig_gram_fast(paths, sigma=1.0, dyadic_order=0):
    """Compute signature kernel Gram matrix.

    For 1-point paths (time_delay_dim=1), falls back to RBF kernel
    since the signature kernel degenerates to a constant.

    Uses the sigkernel library if available, otherwise falls back to
    a vectorized PDE solver that batches all path pairs.
    """
    if paths.shape[1] <= 1:
        return _rbf_gram(paths, sigma)

    if SIGKERNEL_AVAILABLE:
        static_kernel = _sigkernel.RBFKernel(sigma=sigma)
        sk = _sigkernel.SigKernel(static_kernel, dyadic_order=dyadic_order)
        return sk.compute_Gram(paths, paths, sym=True)

    return _compute_sig_gram_vectorized(paths, sigma, dyadic_order)


def _median_bandwidth(paths, max_samples=500):
    """Compute adaptive sigma via median heuristic on path increments."""
    n, L, d = paths.shape
    if L < 2:
        flat = paths.squeeze(1)
    else:
        flat = paths.reshape(n, -1)
    subset = flat[:max_samples] if n > max_samples else flat
    dists = torch.nn.functional.pdist(subset, p=2)
    positive = dists[dists > 0]
    if len(positive) == 0:
        return 1.0
    return (
        torch.sqrt(torch.tensor(flat.shape[1], dtype=torch.float32)) * positive.median()
    ).item()


def _embed_as_paths(
    data, indices, path_transform="time_delay", time_delay_dim=3, time_delay_lag=1
):
    """Embed column data as a batch of T paths for signature kernel.

    Creates one path segment per observation using a local window,
    always returning exactly T paths. When time_delay_dim=1, returns
    trivial 1-point paths (no temporal embedding).
    """
    x = data[:, indices] if hasattr(indices, "__len__") else data[:, [indices]]
    T, d = x.shape
    window = time_delay_dim

    if window <= 1:
        return x.unsqueeze(1)

    paths = torch.zeros(T, window, d, device=x.device)
    for i in range(T):
        start = max(0, i - window + 1)
        end = start + window
        if end > T:
            end = T
            start = max(0, end - window)
        seg = x[start:end]
        if seg.shape[0] < window:
            pad = window - seg.shape[0]
            seg = torch.cat([seg[:1].expand(pad, -1), seg])
        paths[i] = seg

    return paths


@torch.no_grad()
def sigkci_uind(
    x,
    y,
    sigma="median",
    dyadic_order=0,
    path_transform="time_delay",
    time_delay_dim=3,
    approx="gamma",
    nperms=200,
):
    """Unconditional independence test using signature kernels."""
    device = x.device
    T = x.shape[0]

    x_in = x.unsqueeze(1) if x.dim() == 1 else x
    y_in = y.unsqueeze(1) if y.dim() == 1 else y

    paths_x = _embed_as_paths(
        x_in, list(range(x_in.shape[1])), path_transform, time_delay_dim
    )
    paths_y = _embed_as_paths(
        y_in, list(range(y_in.shape[1])), path_transform, time_delay_dim
    )

    if sigma == "median":
        sigma_x = _median_bandwidth(paths_x)
        sigma_y = _median_bandwidth(paths_y)
    else:
        sigma_x = sigma_y = float(sigma)

    Kx = apply_H(_compute_sig_gram_fast(paths_x, sigma_x, dyadic_order))
    Ky = apply_H(_compute_sig_gram_fast(paths_y, sigma_y, dyadic_order))
    stat = torch.einsum("ij,ji->", Kx, Ky)

    if approx == "perm":
        stats = []
        for _ in range(nperms):
            perm = torch.randperm(T, device=device)
            Kx2 = Kx[perm][:, perm]
            stats.append(torch.einsum("ij,ji->", Kx2, Ky).item())
        p_val = (np.array(stats) >= stat.item()).mean()
        return p_val, stat.cpu().item()
    else:
        mean_appr = torch.trace(Kx) * torch.trace(Ky) / T
        var_appr = (
            2
            * torch.einsum("ij,ji->", Kx, Kx)
            * torch.einsum("ij,ji->", Ky, Ky)
            / (T**2)
        )
        var_appr = torch.clamp(var_appr, min=1e-20)
        mean_appr = torch.where(
            mean_appr.abs() < 1e-20,
            torch.tensor(1e-20, device=device),
            mean_appr,
        )
        k_appr = (mean_appr**2 / var_appr).cpu()
        theta_appr = (var_appr / mean_appr).cpu()
        p_val = 1 - gamma_distr.cdf(stat.cpu(), a=k_appr, scale=theta_appr)
        return p_val, stat.cpu().item()


@torch.no_grad()
def sigkci_cind(
    x,
    y,
    z,
    sigma="median",
    dyadic_order=0,
    path_transform="time_delay",
    time_delay_dim=3,
    approx="gamma",
    nperms=200,
    lamda=0.001,
):
    """Conditional independence test using signature kernels.

    Uses the KCI framework with signature-based kernel matrices.
    """
    device = x.device
    T = x.shape[0]

    if x.dim() == 1:
        x = x.unsqueeze(1)
    if y.dim() == 1:
        y = y.unsqueeze(1)
    if z.dim() == 1:
        z = z.unsqueeze(1)

    xz = torch.cat([x, z], dim=1)
    paths_xz = _embed_as_paths(
        xz, list(range(xz.shape[1])), path_transform, time_delay_dim
    )
    paths_y = _embed_as_paths(
        y, list(range(y.shape[1])), path_transform, time_delay_dim
    )
    paths_z = _embed_as_paths(
        z, list(range(z.shape[1])), path_transform, time_delay_dim
    )

    if sigma == "median":
        sigma_xz = _median_bandwidth(paths_xz)
        sigma_y = _median_bandwidth(paths_y)
        sigma_z = _median_bandwidth(paths_z)
    else:
        sigma_xz = sigma_y = sigma_z = float(sigma)

    Kx = apply_H(_compute_sig_gram_fast(paths_xz, sigma_xz, dyadic_order))
    Ky = apply_H(_compute_sig_gram_fast(paths_y, sigma_y, dyadic_order))
    Kz = apply_H(_compute_sig_gram_fast(paths_z, sigma_z, dyadic_order))

    eye_T = torch.eye(T, device=device)
    reg = lamda
    for _ in range(5):
        try:
            P1 = eye_T - Kz @ torch.linalg.solve(Kz + reg * eye_T, eye_T)
            break
        except torch._C._LinAlgError:
            reg *= 10
    else:
        P1 = eye_T
    Kxz = P1 @ Kx @ P1.T
    Kyz = P1 @ Ky @ P1.T
    stat = torch.einsum("ij,ji->", Kxz, Kyz)

    if approx == "perm":
        stats = []
        for _ in range(nperms):
            perm = torch.randperm(T, device=device)
            Kxz2 = P1 @ Kx[perm][:, perm] @ P1.T
            stats.append(torch.einsum("ij,ji->", Kxz2, Kyz).item())
        p_val = (np.array(stats) >= stat.item()).mean()
        return p_val, stat.cpu().item()

    eig_Kxz, eivx = torch.linalg.eigh((Kxz + Kxz.T) / 2)
    eig_Kyz, eivy = torch.linalg.eigh((Kyz + Kyz.T) / 2)

    thresh = 1e-5
    x_ix = (eig_Kxz > eig_Kxz.max() * thresh).nonzero().ravel()
    y_ix = (eig_Kyz > eig_Kyz.max() * thresh).nonzero().ravel()

    eig_Kxz = eig_Kxz[x_ix]
    eivx = eivx[:, x_ix]
    eig_Kyz = eig_Kyz[y_ix]
    eivy = eivy[:, y_ix]

    eiv_prodx = eivx * torch.sqrt(eig_Kxz)
    eiv_prody = eivy * torch.sqrt(eig_Kyz)

    uu = eiv_prodx.unsqueeze(-1) * eiv_prody.unsqueeze(-2)
    uu = uu.reshape(T, -1)
    size_u = uu.shape[1]

    if size_u > T:
        uu_prod = uu @ uu.T
    else:
        uu_prod = uu.T @ uu
    eig_uu = torch.linalg.eigvalsh(uu_prod)
    eig_uu = eig_uu[eig_uu > eig_uu.max() * thresh]

    if approx == "gamma":
        mean_appr = eig_uu.sum()
        var_appr = 2 * (eig_uu**2).sum()
        k_appr = (mean_appr**2 / var_appr).cpu()
        theta_appr = (var_appr / mean_appr).cpu()
        p_val = 1 - gamma_distr.cdf(stat.cpu(), a=k_appr, scale=theta_appr)
    else:
        K1 = eig_uu.sum()
        K2 = 2 * (eig_uu**2).sum()
        K3 = 8 * (eig_uu**3).sum()
        nu = 8 * (K2**3) / (K3**2)
        k = nu / 2
        x_stat = ((2 * nu / K2) ** 0.5) * (stat - K1) + nu
        p_val = 1 - gamma_distr.cdf(x_stat.cpu(), a=k.cpu(), scale=2)

    return p_val, stat.cpu().item()


class SigKCIGPU(CIT_Base):
    """GPU-accelerated Signature Kernel Conditional Independence Test.

    Uses signature kernels on path-space embeddings combined with the
    KCI framework for conditional independence testing in time series.

    When called from CDNOTS (which provides lag structure metadata via
    set_lag_structure), builds proper temporal paths from consecutive
    rows — making the signature kernel genuinely different from KCI.
    Without lag structure, falls back to RBF kernel (time_delay_dim=1
    equivalent) for backward compatibility.
    """

    def __init__(
        self,
        data,
        device=None,
        sigma="median",
        dyadic_order=0,
        path_transform="time_delay",
        time_delay_dim=3,
        approx="hbe",
        nperms=200,
        lamda=0.001,
        normalize=True,
        seed=42,
        **kwargs,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._data = data
        self.data_torch = torch.tensor(np.asarray(data).copy(), dtype=torch.float32).to(
            self.device
        )
        self.sigma = sigma
        self.dyadic_order = dyadic_order
        self.path_transform = path_transform
        self.time_delay_dim = time_delay_dim
        self.approx = approx
        self.nperms = nperms
        self.lamda = lamda
        self.normalize = normalize
        self.param_hash = (
            f"T{data.shape[0]}-sig{sigma}-do{dyadic_order}"
            f"-{path_transform}-td{time_delay_dim}-{approx}-{nperms}-n{normalize}"
        )
        self._has_lag_structure = False
        self._n_vars = None
        self._lags = None
        super().__init__(data, method="sigkci", seed=seed, **kwargs)
        self.n_actual_tests = 0
        self.n_tests = 0

    def get_data(self):
        return self._data

    def set_data(self, d):
        self._data = d
        self._has_nan = bool(np.isnan(d).any())
        self.data_torch = torch.tensor(np.asarray(d).copy(), dtype=torch.float32).to(
            self.device
        )

    data = property(get_data, set_data)

    def set_lag_structure(self, n_vars, lags):
        """Register lag-embedding structure from CDNOTS.

        This enables SigKCI to build proper temporal paths from consecutive
        rows of the same original variable, instead of treating the lag-embedded
        matrix as flat tabular data.

        Args:
            n_vars: Number of original variables (including C if present).
            lags: List of lag indices used in embedding (e.g., [0, 1, 2]).
        """
        self._n_vars = n_vars
        self._lags = lags
        self._has_lag_structure = True

    def _col_to_var(self, col):
        """Map lag-embedded column index to original variable index."""
        return col % self._n_vars

    def _build_row_paths(self, data_t, col_indices):
        """Build temporal paths from consecutive rows for given columns.

        For each column, reads the same column across consecutive rows
        to form a genuine temporal path of that variable.

        Returns:
            (T_eff, time_delay_dim, n_cols) tensor of paths
        """
        T = data_t.shape[0]
        td = self.time_delay_dim
        n_cols = len(col_indices) if hasattr(col_indices, "__len__") else 1
        if not hasattr(col_indices, "__len__"):
            col_indices = [col_indices]

        x = data_t[:, col_indices]
        T_eff = T - td + 1
        if T_eff < 2:
            return x.unsqueeze(1)

        paths = torch.zeros(T_eff, td, n_cols, device=data_t.device)
        for i in range(T_eff):
            paths[i] = x[i : i + td]

        return paths

    def __call__(self, X, Y, condition_set=None):
        self.n_tests += 1
        self.n_actual_tests += 1

        cache_key = self._get_array_hash(X, Y, condition_set)
        cache_key = f"{self.method}-{cache_key}"
        combined_hash = cache_key, self.param_hash

        if combined_hash in self.pvalue_cache:
            p, stat = self.pvalue_cache[combined_hash]
            if combined_hash not in self.pvalue_local:
                self.pvalue_local[combined_hash] = (p, stat)
            else:
                self.n_actual_tests -= 1
            return p, stat

        data_t = self.data_torch
        if self._has_nan:
            mask = self._pairwise_complete(X, Y, condition_set)
            if mask.sum() < self._min_samples:
                p, stat = 1.0, 0.0
                self.pvalue_cache[combined_hash] = (p, stat)
                return p, stat
            data_t = self.data_torch[torch.from_numpy(mask).to(self.device)]

        if self.normalize:
            data_t = normalize_gpu(data_t)

        use_row_paths = (
            getattr(self, "_has_lag_structure", False) and self.time_delay_dim > 1
        )

        if use_row_paths:
            p, stat = self._call_with_row_paths(data_t, X, Y, condition_set)
        else:
            x = data_t[:, X]
            y = data_t[:, Y]
            if condition_set is None or len(condition_set) == 0:
                p, stat = sigkci_uind(
                    x,
                    y,
                    sigma=self.sigma,
                    dyadic_order=self.dyadic_order,
                    path_transform=self.path_transform,
                    time_delay_dim=self.time_delay_dim,
                    approx=self.approx,
                    nperms=self.nperms,
                )
            else:
                z = data_t[:, condition_set]
                p, stat = sigkci_cind(
                    x,
                    y,
                    z,
                    sigma=self.sigma,
                    dyadic_order=self.dyadic_order,
                    path_transform=self.path_transform,
                    time_delay_dim=self.time_delay_dim,
                    approx=self.approx,
                    nperms=self.nperms,
                    lamda=self.lamda,
                )

        self.pvalue_cache[combined_hash] = (p, stat)
        self.pvalue_local[combined_hash] = (p, stat)
        self.save_incremental_cache(cache_key, self.param_hash, p, stat)
        return p, stat

    @torch.no_grad()
    def _call_with_row_paths(self, data_t, X, Y, condition_set):
        """CI test using row-based temporal paths from lag-embedded data.

        Builds genuine temporal paths by reading consecutive rows of
        the same column, producing paths of shape (T_eff, td, d).
        """
        x_cols = [X] if not hasattr(X, "__len__") else list(X)
        y_cols = [Y] if not hasattr(Y, "__len__") else list(Y)

        paths_x = self._build_row_paths(data_t, x_cols)
        paths_y = self._build_row_paths(data_t, y_cols)
        T_eff = paths_x.shape[0]

        sigma_x = (
            _median_bandwidth(paths_x) if self.sigma == "median" else float(self.sigma)
        )
        sigma_y = (
            _median_bandwidth(paths_y) if self.sigma == "median" else float(self.sigma)
        )

        Kx = apply_H(_compute_sig_gram_fast(paths_x, sigma_x, self.dyadic_order))
        Ky = apply_H(_compute_sig_gram_fast(paths_y, sigma_y, self.dyadic_order))

        if condition_set is None or len(condition_set) == 0:
            stat = torch.einsum("ij,ji->", Kx, Ky)
            return _gamma_or_hbe_pval(stat, Kx, Ky, T_eff, self.approx)

        z_cols = list(condition_set)
        xz_cols = x_cols + z_cols
        paths_xz = self._build_row_paths(data_t, xz_cols)
        paths_z = self._build_row_paths(data_t, z_cols)

        sigma_xz = (
            _median_bandwidth(paths_xz) if self.sigma == "median" else float(self.sigma)
        )
        sigma_z = (
            _median_bandwidth(paths_z) if self.sigma == "median" else float(self.sigma)
        )

        Kxz = apply_H(_compute_sig_gram_fast(paths_xz, sigma_xz, self.dyadic_order))
        Kz = apply_H(_compute_sig_gram_fast(paths_z, sigma_z, self.dyadic_order))

        eye_T = torch.eye(T_eff, device=data_t.device)
        reg = self.lamda
        for _ in range(5):
            try:
                P1 = eye_T - Kz @ torch.linalg.solve(Kz + reg * eye_T, eye_T)
                break
            except torch._C._LinAlgError:
                reg *= 10
        else:
            P1 = eye_T

        Kxz_r = P1 @ Kxz @ P1.T
        Ky_r = P1 @ Ky @ P1.T
        stat = torch.einsum("ij,ji->", Kxz_r, Ky_r)
        return _cond_pval(stat, Kxz_r, Ky_r, T_eff, self.approx)
