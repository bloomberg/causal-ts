# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# SplitKCI: Practical Kernel Tests of Conditional Independence
# Based on: Pogodin, Gismalla, Oates, "Practical Kernel Tests of
# Conditional Independence", arXiv:2402.13196, 2024.
# Reference implementation: https://github.com/romanpogodin/kernel-ci-testing

import numpy as np
import torch
from scipy.stats import gamma as gamma_distr

from .cit_test import CIT_Base
from .kci_gpu import normalize_gpu


def _median_bandwidth(x):
    if x.shape[0] > 500:
        dist = torch.nn.functional.pdist(x[:500], p=2)
    else:
        dist = torch.nn.functional.pdist(x, p=2)
    positive = dist[dist > 0]
    if len(positive) == 0:
        return 1.0
    return (
        torch.sqrt(torch.tensor(x.shape[1], dtype=torch.float32)) * positive.median()
    ).item()


def _gaussian_kernel(x, sigma2):
    sq_dist = torch.cdist(x, x, p=2) ** 2
    return torch.exp(-sq_dist / sigma2)


def _center_kernel(K):
    K = K - K.mean(dim=1, keepdim=True)
    return K - K.mean(dim=0, keepdim=True)


def _compute_hsic(K, L):
    return torch.einsum("ij,ij->", K, L) / (K.shape[0] ** 2)


def _loo_cv_ridge(Kxx, Kyy, lambda_values, cpu_solver=False):
    """Select ridge lambda via leave-one-out cross-validation."""
    n = Kxx.shape[0]
    best_lambda = lambda_values[0]
    best_error = float("inf")

    for lam in lambda_values:
        A = Kxx + lam * torch.eye(n, device=Kxx.device)
        A_inv = torch.linalg.solve(A, torch.eye(n, device=Kxx.device))
        H = Kxx @ A_inv
        residuals = Kyy - H @ Kyy
        leverage = 1.0 - torch.diag(H)
        leverage = torch.clamp(leverage, min=1e-10)
        loo_residuals = residuals / leverage.unsqueeze(1)
        error = (loo_residuals**2).mean().item()
        if error < best_error:
            best_error = error
            best_lambda = lam

    return best_lambda


def _fit_krr(x_train, y_train, sigma2_x, sigma2_y, ridge_lambda, lambda_values):
    """Fit kernel ridge regression and return precomputed matrices."""
    Kxx = _gaussian_kernel(x_train, sigma2_x)
    Kyy = _gaussian_kernel(y_train, sigma2_y)

    if lambda_values is not None:
        ridge_lambda = _loo_cv_ridge(Kxx, Kyy, lambda_values)

    n = Kxx.shape[0]
    A = Kxx + ridge_lambda * torch.eye(n, device=Kxx.device)
    Kxx_inv = torch.linalg.solve(A, torch.eye(n, device=Kxx.device))
    Kxx_inv_Kyy = Kxx_inv @ Kyy

    return Kxx_inv, Kxx_inv_Kyy, Kxx, Kyy, ridge_lambda


def _cme_centered_kernel(
    x_test, y_test, x_train, y_train, Kxx_inv, Kxx_inv_Kyy, sigma2_x, sigma2_y
):
    """Compute the CME-centered kernel matrix on test data."""
    Kx_train_test = _gaussian_kernel_cross(x_train, x_test, sigma2_x)
    Ky_train_test = _gaussian_kernel_cross(y_train, y_test, sigma2_y)
    Ky_test = _gaussian_kernel(y_test, sigma2_y)

    Kxx_inv_Kx = Kxx_inv @ Kx_train_test
    cme_correction = -Ky_train_test.T @ Kxx_inv_Kx
    regression_term = Kx_train_test.T @ Kxx_inv_Kyy @ Kx_train_test

    result = Ky_test + cme_correction + cme_correction.T + regression_term
    return result


def _gaussian_kernel_cross(x, y, sigma2):
    sq_dist = torch.cdist(x, y, p=2) ** 2
    return torch.exp(-sq_dist / sigma2)


def _gamma_pvalue(stat, K, L):
    """Satterthwaite-Welch (2-moment) gamma approximation."""
    K = _center_kernel(K)
    L = _center_kernel(L)
    KL = K * L
    mean = torch.diagonal(KL).mean()
    var = 2 * (KL**2).mean()
    if var < 1e-20 or mean.abs() < 1e-20:
        return 1.0
    k = (mean**2 / var).item()
    theta = (var / mean / K.shape[0]).item()
    return gamma_distr.sf(stat.item(), a=k, loc=0, scale=theta)


def _hbe_pvalue(stat, Ka, Kb):
    """Hall-Buckley-Eagleson (3-moment) gamma approximation.

    Unlike KCI where Kxz = P1 @ Kx @ P1.T is PSD, SplitKCI's CME-centered
    kernels may not be PSD. We clamp negative eigenvalues to zero before
    computing the product eigenvalues for the null distribution.
    """
    Ka = _center_kernel(Ka)
    Kb = _center_kernel(Kb)
    T = Ka.shape[0]
    thresh = 1e-5

    eig_Ka, eivx = torch.linalg.eigh((Ka + Ka.T) / 2)
    eig_Kb, eivy = torch.linalg.eigh((Kb + Kb.T) / 2)

    # Clamp negative eigenvalues to zero (project onto PSD cone)
    eig_Ka = eig_Ka.clamp(min=0)
    eig_Kb = eig_Kb.clamp(min=0)

    x_ix = (eig_Ka > eig_Ka.max() * thresh).nonzero().ravel()
    y_ix = (eig_Kb > eig_Kb.max() * thresh).nonzero().ravel()

    if len(x_ix) == 0 or len(y_ix) == 0:
        return 1.0

    eig_Ka = eig_Ka[x_ix]
    eivx = eivx[:, x_ix]
    eig_Kb = eig_Kb[y_ix]
    eivy = eivy[:, y_ix]

    eiv_prodx = eivx * torch.sqrt(eig_Ka)
    eiv_prody = eivy * torch.sqrt(eig_Kb)

    uu = eiv_prodx.unsqueeze(-1) * eiv_prody.unsqueeze(-2)
    uu = uu.reshape(T, -1)

    if uu.shape[1] > T:
        uu_prod = uu @ uu.T
    else:
        uu_prod = uu.T @ uu
    eig_uu = torch.linalg.eigvalsh(uu_prod)
    eig_uu = eig_uu[eig_uu > eig_uu.max() * thresh]

    if len(eig_uu) == 0:
        return 1.0

    # The stat uses HSIC = sum(Ka*Kb)/n^2, but eigenvalue moments
    # correspond to trace(Ka @ Kb). Convert stat to trace scale.
    stat_trace = stat * T * T

    K1 = eig_uu.sum()
    K2 = 2 * (eig_uu**2).sum()
    K3 = 8 * (eig_uu**3).sum()

    if K3.abs() < 1e-20 or K2.abs() < 1e-20:
        return 1.0

    nu = 8 * (K2**3) / (K3**2)
    k = nu / 2
    x_stat = ((2 * nu / K2) ** 0.5) * (stat_trace - K1) + nu

    return 1 - gamma_distr.cdf(x_stat.cpu().item(), a=k.cpu().item(), scale=2)


def _wild_bootstrap_pvalue(stat, K, L, n_samples=500, chunk_size=500):
    n = K.shape[0]
    device = K.device
    pval_count = 0
    total = 0

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        batch = end - start
        Q = 2 * torch.randint(0, 2, (batch, n), device=device).float() - 1
        QQ = Q.unsqueeze(2) * Q.unsqueeze(1)
        KQ = QQ * K.unsqueeze(0)
        boot_stats = torch.einsum("bij,ij->b", KQ, L) / (n**2)
        pval_count += (boot_stats > stat).sum().item()
        total += batch

    return pval_count / total


def _compute_pvalue(stat, Ka, Kb, pval_method, n_bootstrap=500):
    if pval_method == "hbe":
        return _hbe_pvalue(stat, Ka, Kb)
    elif pval_method == "gamma":
        return _gamma_pvalue(stat, Ka, Kb)
    else:
        return _wild_bootstrap_pvalue(stat, Ka, Kb, n_samples=n_bootstrap)


@torch.no_grad()
def splitkci_uind(
    x, y, bandwidth="median", pval_method="gamma", n_bootstrap=500, device=None
):
    """Unconditional independence test using HSIC."""
    x = normalize_gpu(x.reshape(-1, 1) if x.dim() == 1 else x)
    y = normalize_gpu(y.reshape(-1, 1) if y.dim() == 1 else y)

    sigma2_x = _median_bandwidth(x) ** 2 if bandwidth == "median" else bandwidth**2
    sigma2_y = _median_bandwidth(y) ** 2 if bandwidth == "median" else bandwidth**2

    Kx = _center_kernel(_gaussian_kernel(x, sigma2_x))
    Ky = _center_kernel(_gaussian_kernel(y, sigma2_y))
    stat = _compute_hsic(Kx, Ky)

    p_val = _compute_pvalue(stat, Kx, Ky, pval_method, n_bootstrap)
    return p_val, stat.cpu().item()


def _cme_one_direction(
    target_train,
    target_test,
    z_train,
    z_test,
    sigma2_target,
    sigma2_z,
    Kzz_train,
    lambda_values,
    device,
):
    """Compute CME-centered kernel for one variable (target|Z)."""
    n = Kzz_train.shape[0]
    Ktarget_train = _gaussian_kernel(target_train, sigma2_target)
    ridge_lam = _loo_cv_ridge(Kzz_train, Ktarget_train, lambda_values)
    A_inv = torch.linalg.solve(
        Kzz_train + ridge_lam * torch.eye(n, device=device),
        torch.eye(n, device=device),
    )
    Kz_cross = _gaussian_kernel_cross(z_train, z_test, sigma2_z)
    Kt_cross = _gaussian_kernel_cross(target_train, target_test, sigma2_target)
    Kt_test = _gaussian_kernel(target_test, sigma2_target)

    Ainv_Kz = A_inv @ Kz_cross
    cme = -Kt_cross.T @ Ainv_Kz
    reg = Kz_cross.T @ A_inv @ Ktarget_train @ A_inv @ Kz_cross
    return Kt_test + cme + cme.T + reg


@torch.no_grad()
def splitkci_cind(
    x,
    y,
    z,
    split_ratio=0.5,
    ridge_lambda=1e-3,
    lambda_values=None,
    bandwidth="median",
    pval_method="gamma",
    n_bootstrap=500,
    seed=None,
    use_all_data=False,
):
    """Conditional independence test with train-test splitting.

    When use_all_data=True, fits two KRR models on each half and tests on
    all data using cross-predictions (no double-dipping). More powerful
    but ~4x slower due to full n×n kernel matrices.
    """
    device = x.device
    T = x.shape[0]
    x = normalize_gpu(x.reshape(-1, 1) if x.dim() == 1 else x)
    y = normalize_gpu(y.reshape(-1, 1) if y.dim() == 1 else y)
    z = normalize_gpu(z.reshape(-1, 1) if z.dim() == 1 else z)

    sigma2_z = _median_bandwidth(z) ** 2 if bandwidth == "median" else bandwidth**2
    sigma2_x = _median_bandwidth(x) ** 2 if bandwidth == "median" else bandwidth**2
    sigma2_y = _median_bandwidth(y) ** 2 if bandwidth == "median" else bandwidth**2

    if lambda_values is None:
        lambda_values = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    perm = torch.randperm(T, generator=gen)
    n_half = T // 2
    idx_1 = perm[:n_half]
    idx_2 = perm[n_half : 2 * n_half]

    if use_all_data:
        # Split KRR: fit on each half, test on ALL data
        x1, x2 = x[idx_1], x[idx_2]
        y1, y2 = y[idx_1], y[idx_2]
        z1, z2 = z[idx_1], z[idx_2]

        Kzz_1 = _gaussian_kernel(z1, sigma2_z)
        Kzz_2 = _gaussian_kernel(z2, sigma2_z)

        # KRR 1 predicts on all data, KRR 2 predicts on all data
        # CME-centered kernel = K_test + cross terms from both halves
        # For variable a (e.g. Y):
        #   Ka = Ky_all - Ky1^T @ Kzz1_inv @ Kz1_all
        #                - Ky2^T @ Kzz2_inv @ Kz2_all
        #                + Kz1_all^T @ Kzz1_inv @ Ky12 @ Kzz2_inv @ Kz2_all
        n1 = Kzz_1.shape[0]
        n2 = Kzz_2.shape[0]

        # Y|Z
        Kyy_1 = _gaussian_kernel(y1, sigma2_y)
        lam_y1 = _loo_cv_ridge(Kzz_1, Kyy_1, lambda_values)
        A_inv_y1 = torch.linalg.solve(
            Kzz_1 + lam_y1 * torch.eye(n1, device=device),
            torch.eye(n1, device=device),
        )
        Kyy_2 = _gaussian_kernel(y2, sigma2_y)
        lam_y2 = _loo_cv_ridge(Kzz_2, Kyy_2, lambda_values)
        A_inv_y2 = torch.linalg.solve(
            Kzz_2 + lam_y2 * torch.eye(n2, device=device),
            torch.eye(n2, device=device),
        )

        Kz1_all = _gaussian_kernel_cross(z1, z, sigma2_z)
        Kz2_all = _gaussian_kernel_cross(z2, z, sigma2_z)
        Ky1_all = _gaussian_kernel_cross(y1, y, sigma2_y)
        Ky2_all = _gaussian_kernel_cross(y2, y, sigma2_y)
        Ky_all = _gaussian_kernel(y, sigma2_y)
        Ky12 = _gaussian_kernel_cross(y1, y2, sigma2_y)

        cme_y = -Ky1_all.T @ (A_inv_y1 @ Kz1_all) - Ky2_all.T @ (A_inv_y2 @ Kz2_all)
        reg_y = Kz1_all.T @ A_inv_y1 @ Ky12 @ A_inv_y2 @ Kz2_all
        Ka = Ky_all + (cme_y + cme_y.T) / 2 + reg_y

        # X|Z
        Kxx_1 = _gaussian_kernel(x1, sigma2_x)
        lam_x1 = _loo_cv_ridge(Kzz_1, Kxx_1, lambda_values)
        A_inv_x1 = torch.linalg.solve(
            Kzz_1 + lam_x1 * torch.eye(n1, device=device),
            torch.eye(n1, device=device),
        )
        Kxx_2 = _gaussian_kernel(x2, sigma2_x)
        lam_x2 = _loo_cv_ridge(Kzz_2, Kxx_2, lambda_values)
        A_inv_x2 = torch.linalg.solve(
            Kzz_2 + lam_x2 * torch.eye(n2, device=device),
            torch.eye(n2, device=device),
        )

        Kx1_all = _gaussian_kernel_cross(x1, x, sigma2_x)
        Kx2_all = _gaussian_kernel_cross(x2, x, sigma2_x)
        Kx_all = _gaussian_kernel(x, sigma2_x)
        Kx12 = _gaussian_kernel_cross(x1, x2, sigma2_x)

        cme_x = -Kx1_all.T @ (A_inv_x1 @ Kz1_all) - Kx2_all.T @ (A_inv_x2 @ Kz2_all)
        reg_x = Kz1_all.T @ A_inv_x1 @ Kx12 @ A_inv_x2 @ Kz2_all
        Kb = Kx_all + (cme_x + cme_x.T) / 2 + reg_x

    else:
        # Standard split: train on one half, test on the other
        idx_train = idx_2
        idx_test = idx_1
        x_train, x_test = x[idx_train], x[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]
        z_train, z_test = z[idx_train], z[idx_test]

        Kzz_train = _gaussian_kernel(z_train, sigma2_z)
        Ka = _cme_one_direction(
            y_train,
            y_test,
            z_train,
            z_test,
            sigma2_y,
            sigma2_z,
            Kzz_train,
            lambda_values,
            device,
        )
        Kb = _cme_one_direction(
            x_train,
            x_test,
            z_train,
            z_test,
            sigma2_x,
            sigma2_z,
            Kzz_train,
            lambda_values,
            device,
        )

    Ka_centered = _center_kernel(Ka)
    Kb_centered = _center_kernel(Kb)
    stat = _compute_hsic(Ka_centered, Kb_centered)

    p_val = _compute_pvalue(stat, Ka_centered, Kb_centered, pval_method, n_bootstrap)
    return p_val, stat.cpu().item()


class SplitKCIGPU(CIT_Base):
    """GPU-accelerated Split Kernel Conditional Independence Test.

    Uses train-test data splitting to control bias from kernel ridge
    regression, following Pogodin et al. (2024).
    """

    def __init__(
        self,
        data,
        device=None,
        split_ratio=0.5,
        ridge_lambda=1e-3,
        pval_method="gamma",
        n_bootstrap=500,
        bandwidth="median",
        use_all_data=False,
        seed=42,
        **kwargs,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._data = data
        self.data_torch = torch.tensor(np.asarray(data).copy(), dtype=torch.float32).to(
            self.device
        )
        self.split_ratio = split_ratio
        self.ridge_lambda = ridge_lambda
        self.pval_method = pval_method
        self.n_bootstrap = n_bootstrap
        self.bandwidth = bandwidth
        self.use_all_data = use_all_data
        self._seed = seed
        self.param_hash = (
            f"T{data.shape[0]}-split{split_ratio}-lam{ridge_lambda}"
            f"-{pval_method}-{n_bootstrap}-bw_{bandwidth}"
            f"-alldata{use_all_data}"
        )
        super().__init__(data, method="splitkci", seed=seed, **kwargs)
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

        if condition_set is None or len(condition_set) == 0:
            p, stat = splitkci_uind(
                data_t[:, X],
                data_t[:, Y],
                bandwidth=self.bandwidth,
                pval_method=self.pval_method,
                n_bootstrap=self.n_bootstrap,
                device=self.device,
            )
        else:
            p, stat = splitkci_cind(
                data_t[:, X],
                data_t[:, Y],
                data_t[:, condition_set],
                split_ratio=self.split_ratio,
                ridge_lambda=self.ridge_lambda,
                bandwidth=self.bandwidth,
                pval_method=self.pval_method,
                n_bootstrap=self.n_bootstrap,
                seed=self._seed,
                use_all_data=self.use_all_data,
            )

        self.pvalue_cache[combined_hash] = (p, stat)
        self.pvalue_local[combined_hash] = (p, stat)
        self.save_incremental_cache(cache_key, self.param_hash, p, stat)
        return p, stat
