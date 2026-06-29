# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# CItest: A Distribution Free Conditional Independence Test
# Based on: Cai, Li, Zhang, "A Distribution Free Conditional Independence
# Test with Applications to Causal Discovery", JMLR 23(152):1-34, 2022.
# Reference implementation: https://github.com/zhanruicai/CItest (MIT license)

import numpy as np
import torch

from .cit_test import CIT_Base


def _silverman_bandwidth(z):
    """Silverman's rule of thumb for kernel bandwidth."""
    n = z.shape[0]
    if z.dim() == 1:
        std = z.std().item()
    else:
        std = z.std(dim=0).mean().item()
    if std < 1e-10:
        std = 1.0
    return 1.06 * std * (4.0 / (3.0 * n)) ** 0.2


def _pca_whiten(z, var_threshold=0.999):
    """Project z onto its principal components and standardize.

    Removes near-collinear directions (keeps components explaining
    var_threshold of total variance). Returns whitened z with
    uncorrelated, unit-variance columns.
    """
    if z.dim() == 1 or z.shape[1] == 1:
        return z

    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = z_centered.T @ z_centered / (z.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)

    # Keep components in descending eigenvalue order
    eigvals = eigvals.flip(0)
    eigvecs = eigvecs.flip(1)

    # Retain dims explaining var_threshold of variance
    total = eigvals.sum()
    cumvar = eigvals.cumsum(0) / total.clamp(min=1e-10)
    n_keep = int((cumvar < var_threshold).sum().item()) + 1
    n_keep = max(1, min(n_keep, z.shape[1]))

    eigvals_k = eigvals[:n_keep].clamp(min=1e-10)
    eigvecs_k = eigvecs[:, :n_keep]

    # Project and scale to unit variance
    z_proj = z_centered @ eigvecs_k
    return z_proj / eigvals_k.sqrt().unsqueeze(0)


def _gaussian_kernel_weights(z, h):
    """Compute Gaussian kernel weight matrix K_h(z_i - z_j) for all pairs.

    Returns (n, n) matrix where W[i, j] = K_h(z_i - z_j).
    """
    if z.dim() == 1:
        z = z.unsqueeze(1)
    sq_dist = torch.cdist(z, z, p=2) ** 2
    d = z.shape[1]
    return torch.exp(-sq_dist / (2 * h**2)) / ((2 * torch.pi) ** (d / 2.0))


def _estimate_conditional_cdfs(x, z, h):
    """Estimate F(x_i | z_i) for each observation i using NW kernel smoothing.

    Returns vector of length n with CDF values.
    """
    if z.dim() == 1:
        z = z.unsqueeze(1)
    K = _gaussian_kernel_weights(z, h)
    indicator = (x.unsqueeze(0) <= x.unsqueeze(1)).float()
    weighted_ind = K * indicator
    K_sum = K.sum(dim=1).clamp(min=1e-10)
    U = weighted_ind.sum(dim=1) / K_sum
    return U


def _estimate_marginal_cdf(z):
    """Estimate marginal CDF: W_i = P(Z <= z_i) = mean(Z <= z_i)."""
    if z.dim() == 1:
        return (z.unsqueeze(0) <= z.unsqueeze(1)).float().mean(dim=1)
    else:
        cdfs = []
        for col in range(z.shape[1]):
            cdfs.append(
                (z[:, col].unsqueeze(0) <= z[:, col].unsqueeze(1)).float().mean(dim=1)
            )
        return torch.stack(cdfs, dim=1)


def _build_uu_matrix(u):
    """Build the uu kernel matrix from CDF values.

    uu[i,j] = exp(-|u_i - u_j|) + (exp(-u_i) + exp(u_i-1)) + (exp(-u_j) + exp(u_j-1)) + 2*exp(-1) - 4
    """
    u_dist = torch.cdist(u.unsqueeze(1), u.unsqueeze(1), p=1).squeeze()
    if u_dist.dim() == 0:
        u_dist = u_dist.unsqueeze(0).unsqueeze(0)
    ui = torch.exp(-u) + torch.exp(u - 1)
    uu = torch.exp(-u_dist) + ui.unsqueeze(1) + ui.unsqueeze(0) + 2 * np.exp(-1) - 4
    return uu


def _build_ww_matrix(w):
    """Build the ww kernel matrix from marginal CDF values."""
    if w.dim() == 1:
        w_dist = torch.cdist(w.unsqueeze(1), w.unsqueeze(1), p=1).squeeze()
        if w_dist.dim() == 0:
            w_dist = w_dist.unsqueeze(0).unsqueeze(0)
    else:
        w_dist = torch.zeros(w.shape[0], w.shape[0], device=w.device)
        for col in range(w.shape[1]):
            w_dist += torch.cdist(
                w[:, col].unsqueeze(1), w[:, col].unsqueeze(1), p=1
            ).squeeze()
    return torch.exp(-w_dist)


def _permutation_test(
    uu,
    fixed,
    stat,
    n_perm,
    n,
    device,
    seed=None,
    alpha=0.05,
    early_stop=True,
    block_length=None,
):
    """Batched permutation test with early stopping.

    Permutes uu rows/cols against a fixed matrix (vv*ww or just vv).
    Uses batched tensor operations and stops early when the p-value
    is clearly resolved. Two-level early stopping (following CMIknn):

    1. Running p-value check: if p >> alpha or p-value std is 0, enter
       second-level check.
    2. Null distribution z-score: if |stat - quantile(null, alpha)| / std(null)
       exceeds 5, or if the running p-value median is >3 std from alpha, stop.

    block_length : int or None
        If > 1, use block shuffle (permute contiguous blocks of rows) instead of
        fully random permutation. Preserves short-range temporal autocorrelation
        in the null distribution. Recommended for unconditional tests (Z empty)
        on autocorrelated time series. Set adaptively as max(3, int(T**0.5)).
    """
    gen = torch.Generator(device="cpu")
    rng_block = None
    if seed is not None:
        gen.manual_seed(seed)
    if block_length is not None and block_length > 1:
        rng_block = np.random.default_rng(seed if seed is not None else 0)

    fixed_flat = fixed.reshape(-1)
    n_sq = n * n

    count_ge = 0
    total = 0
    batch_size = min(50, n_perm)
    min_perms = 20
    check_interval = 20

    null_dist = []
    running_pvals = []

    for start in range(0, n_perm, batch_size):
        end = min(start + batch_size, n_perm)
        batch = end - start

        if rng_block is not None:
            bl = block_length
            n_blocks = n // bl
            raw_perms = []
            for _ in range(batch):
                block_order = rng_block.permutation(n_blocks)
                p = np.concatenate(
                    [np.arange(b * bl, (b + 1) * bl) for b in block_order]
                )
                if n > n_blocks * bl:
                    p = np.concatenate([p, np.arange(n_blocks * bl, n)])
                raw_perms.append(torch.tensor(p, dtype=torch.long))
            perms = torch.stack(raw_perms).to(device)
        else:
            perms = torch.stack(
                [torch.randperm(n, generator=gen) for _ in range(batch)]
            ).to(device)
        uu_rows = uu[perms]
        uu_perm = torch.gather(uu_rows, 2, perms.unsqueeze(1).expand(-1, n, -1))
        perm_stats = (uu_perm.reshape(batch, n_sq) * fixed_flat).mean(dim=1)

        count_ge += (perm_stats >= stat).sum().item()
        total += batch
        null_dist.extend(perm_stats.cpu().tolist())
        running_pvals.append(count_ge / total)

        if not early_stop or total < min_perms:
            continue
        if total % check_interval != 0:
            continue

        running_p = count_ge / total

        # Level 1: quick check — is p clearly away from alpha?
        if running_p <= alpha + 0.1 and np.std(running_pvals) > 0:
            continue

        # Level 2: null distribution z-score check
        null_arr = np.array(null_dist)
        null_std = np.std(null_arr)
        if null_std > 0:
            q = np.quantile(null_arr, alpha)
            z = np.abs(stat.item() - q) / null_std
            if z > 5:
                break

        # Level 3: running p-value stability
        pval_std = np.std(running_pvals)
        z_pval = (np.median(running_pvals) - alpha) / max(pval_std, 0.0001)
        if z_pval >= 3:
            break

    return count_ge / total


def _regress_out(target, z):
    """OLS-residualize target against z. Returns residuals."""
    if z.dim() == 1:
        z = z.unsqueeze(1)
    z_with_intercept = torch.cat([z, torch.ones(z.shape[0], 1, device=z.device)], dim=1)
    # beta = (Z'Z)^{-1} Z' target
    beta = torch.linalg.lstsq(z_with_intercept, target.unsqueeze(1)).solution
    resid = target - (z_with_intercept @ beta).squeeze()
    return resid


@torch.no_grad()
def citest_cind(
    x,
    y,
    z,
    h=None,
    n_perm=500,
    seed=None,
    alpha=0.05,
    early_stop=True,
    whiten_z=False,
    residualize=False,
):
    """Distribution-free conditional independence test.

    Tests H0: X _|_ Y | Z using empirical conditional CDFs
    and a permutation-based p-value.

    residualize : bool (default False)
        Regress X and Y on Z first, then run the unconditional CDF test
        on the residuals. Only effective for linear Z effects; destroys
        nonlinear signal so disabled by default.

    whiten_z : bool (default False)
        PCA-whiten Z before NW bandwidth computation (used only when
        residualize=False). Removes collinearity in the conditioning set.
        Disabled by default as it can hurt recall on nonlinear data.
    """
    device = x.device
    n = x.shape[0]

    if residualize:
        x_res = _regress_out(x, z)
        y_res = _regress_out(y, z)
        return citest_uind(
            x_res, y_res, n_perm=n_perm, seed=seed, alpha=alpha, early_stop=early_stop
        )

    if whiten_z and z.dim() > 1 and z.shape[1] > 1:
        z = _pca_whiten(z)

    if h is None:
        h = _silverman_bandwidth(z)

    U = _estimate_conditional_cdfs(x, z, h)
    V = _estimate_conditional_cdfs(y, z, h)
    W = _estimate_marginal_cdf(z)

    uu = _build_uu_matrix(U)
    vv = _build_uu_matrix(V)
    ww = _build_ww_matrix(W)

    vv_ww = vv * ww
    stat = (uu * vv_ww).mean()

    p_val = _permutation_test(
        uu, vv_ww, stat, n_perm, n, device, seed, alpha, early_stop
    )
    return p_val, stat.cpu().item()


@torch.no_grad()
def citest_uind(
    x, y, n_perm=500, seed=None, alpha=0.05, early_stop=True, block_length=None
):
    """Distribution-free unconditional independence test.

    Tests H0: X _|_ Y using empirical marginal CDFs.

    block_length : int or None
        Block length for block-shuffle null distribution. Recommended for
        autocorrelated time series. If None, uses fully random permutation.
    """
    device = x.device
    n = x.shape[0]

    U = _estimate_marginal_cdf(x)
    V = _estimate_marginal_cdf(y)

    uu = _build_uu_matrix(U)
    vv = _build_uu_matrix(V)

    stat = (uu * vv).mean()

    p_val = _permutation_test(
        uu,
        vv,
        stat,
        n_perm,
        n,
        device,
        seed,
        alpha,
        early_stop,
        block_length=block_length,
    )
    return p_val, stat.cpu().item()


class DFCITGPU(CIT_Base):
    """GPU-accelerated Distribution-Free Conditional Independence Test.

    Implements Cai, Li, Zhang (JMLR 2022) using empirical conditional CDFs
    with Nadaraya-Watson kernel smoothing and permutation-based p-values.
    """

    def __init__(
        self,
        data,
        device=None,
        n_perm=500,
        bandwidth="silverman",
        alpha=0.05,
        early_stop=True,
        whiten_z=False,
        residualize=False,
        seed=42,
        sig_blocklength="auto",
        **kwargs,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._data = data
        self.data_torch = torch.tensor(np.asarray(data).copy(), dtype=torch.float32).to(
            self.device
        )
        self.n_perm = n_perm
        self.bandwidth = bandwidth
        self.alpha = alpha
        self.early_stop = early_stop
        self.whiten_z = whiten_z
        self.residualize = residualize
        self._seed = seed
        T = data.shape[0]
        if sig_blocklength == "auto":
            self.sig_blocklength = max(3, int(T**0.5))
        else:
            self.sig_blocklength = sig_blocklength  # int or None
        self.param_hash = (
            f"T{T}-nperm{n_perm}-bw_{bandwidth}"
            f"-whiten{whiten_z}-resid{residualize}-bl{self.sig_blocklength}"
        )
        super().__init__(data, method="dfcit", seed=seed, **kwargs)
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

        x = data_t[:, X]
        y = data_t[:, Y]

        if condition_set is None or len(condition_set) == 0:
            p, stat = citest_uind(
                x,
                y,
                n_perm=self.n_perm,
                seed=self._seed,
                alpha=self.alpha,
                early_stop=self.early_stop,
                block_length=self.sig_blocklength,
            )
        else:
            z = data_t[:, condition_set]
            h = None
            if self.bandwidth != "silverman":
                h = float(self.bandwidth)
            p, stat = citest_cind(
                x,
                y,
                z,
                h=h,
                n_perm=self.n_perm,
                seed=self._seed,
                alpha=self.alpha,
                early_stop=self.early_stop,
                whiten_z=self.whiten_z,
                residualize=self.residualize,
            )

        self.pvalue_cache[combined_hash] = (p, stat)
        self.pvalue_local[combined_hash] = (p, stat)
        self.save_incremental_cache(cache_key, self.param_hash, p, stat)
        return p, stat
