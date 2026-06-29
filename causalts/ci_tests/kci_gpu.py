# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import torch
from scipy.stats import gamma

from .cit_test import CIT_Base


def normalize_gpu(a):
    """
    Normalize a tensor by subtracting the mean and dividing by the standard deviation.

    Args:
        a (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    m = torch.mean(a, axis=0)
    std = torch.std(a, axis=0)
    std = torch.clamp(std, min=1e-10)
    return (a - m) / std


def rbf_kernel_gpu(x, width="empirical"):
    """
    Compute the RBF kernel for a single variable.

    Args:
        x (torch.Tensor): Input tensor.
        width (str or float, optional): Kernel bandwidth ('empirical', 'median', or a float). Defaults to "empirical".

    Returns:
        torch.Tensor: RBF kernel matrix.
    """
    T = x.size(0)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    if width == "empirical":
        if T < 200:
            width = 1.2
        elif T < 1200:
            width = 0.7
        else:
            width = 0.4
    elif width == "median":
        if T > 500:
            dist = torch.nn.functional.pdist(x[:500], p=2)
        else:
            dist = torch.nn.functional.pdist(x, p=2)
        positive_dists = dist[dist > 0]
        if len(positive_dists) == 0:
            width = 1.0
        else:
            width = (
                torch.sqrt(torch.tensor(x.size(1), dtype=torch.float32))
                * positive_dists.median()
            )

    w = 1.0 / (width**2)
    return torch.exp(-w * torch.cdist(x, x, p=2) ** 2)


def apply_H(K):
    """
    Center a kernel matrix using the centering matrix H = I - 1/n.

    Args:
        K (torch.Tensor): Kernel matrix.

    Returns:
        torch.Tensor: Centered kernel matrix.
    """
    T = K.shape[0]
    K_colsums = K.sum(axis=0)
    K_allsum = K_colsums.sum()
    return K - (K_colsums[None, :] + K_colsums[:, None]) / T + (K_allsum / T**2)


def ukci_gpu_no_gamma(
    x, y, approx="perm", nperms=200, width_x="empirical", width_y="empirical"
):
    """
    Perform an unconditional kernel-based independence test (UKCI) without gamma approximation.

    Args:
        x (torch.Tensor): Input variable X.
        y (torch.Tensor): Input variable Y.
        approx (str, optional): Approximation method ('perm' or 'sw'). Defaults to 'perm'.
        nperms (int, optional): Number of permutations for permutation testing. Defaults to 200.
        width_x (str or float, optional): Kernel bandwidth for X. Defaults to "empirical".
        width_y (str or float, optional): Kernel bandwidth for Y. Defaults to "empirical".

    Returns:
        tuple: P-value and test statistic (or additional parameters for non-permutation methods).
    """
    x = normalize_gpu(x)
    y = normalize_gpu(y)
    T = len(x)

    Kx = apply_H(rbf_kernel_gpu(x, width_x))
    Ky = apply_H(rbf_kernel_gpu(y, width_y))
    stat = torch.einsum("ij,ji->", Kx, Ky)  # Trace(Kx @ Ky)
    stats = []
    if approx == "perm":
        for i in range(nperms):
            perm = np.random.permutation(T)
            Kx2 = Kx[perm][:, perm]
            stats.append(torch.einsum("ij,ji->", Kx2, Ky).item())  # Trace(Kxz @ Kyz)
        stats = np.array(stats)
        p = (stats >= stat.item()).mean()
        return p, stat
    else:
        mean_appr = torch.trace(Kx) * torch.trace(Ky) / T
        Kx_stat = torch.einsum("ij,ji->", Kx, Kx)  # Trace(Kx @ Kx)
        Ky_stat = torch.einsum("ij,ji->", Ky, Ky)  # Trace(Ky @ Ky)
        var_appr = 2 * Kx_stat * Ky_stat / (T**2)
        var_appr = torch.clamp(var_appr, min=1e-20)
        mean_appr = torch.where(
            mean_appr.abs() < 1e-20,
            torch.tensor(1e-20, device=mean_appr.device),
            mean_appr,
        )
        k_appr = (mean_appr**2) / var_appr
        theta_appr = var_appr / mean_appr

        return stat, k_appr, theta_appr


@torch.no_grad()
def kci_uind(x, y, approx="perm", nperms=200, width_x="empirical", width_y="empirical"):
    """
    Perform an unconditional kernel-based independence test (KCI).

    Args:
        x (torch.Tensor): Input variable X.
        y (torch.Tensor): Input variable Y.
        approx (str, optional): Approximation method ('perm' or 'sw'). Defaults to 'perm'.
        nperms (int, optional): Number of permutations for permutation testing. Defaults to 200.
        width_x (str or float, optional): Kernel bandwidth for X. Defaults to "empirical".
        width_y (str or float, optional): Kernel bandwidth for Y. Defaults to "empirical".

    Returns:
        tuple: P-value and test statistic.
    """
    if approx == "perm":
        p_val, stat = ukci_gpu_no_gamma(
            x, y, approx="perm", width_x=width_x, width_y=width_y
        )
        return p_val, stat.cpu().item()
    else:
        stat, k_appr, theta_appr = ukci_gpu_no_gamma(
            x, y, approx="sw", width_x=width_x, width_y=width_y
        )
        stat, k_appr, theta_appr = stat.cpu(), k_appr.cpu(), theta_appr.cpu()
        p_val = 1 - gamma.cdf(stat, a=k_appr, scale=theta_appr)
        return p_val, stat.item()


@torch.no_grad()
def kci_cind(
    x,
    y,
    z,
    approx="perm",
    nperms=200,
    width_xz="median",
    width_y="median",
    width_z="median",
    thresh=1e-5,
    lamda=0.001,
):
    """
    Perform a kernel-based conditional independence test (KCI).

    Args:
        x (torch.Tensor): Input variable X.
        y (torch.Tensor): Input variable Y.
        z (torch.Tensor): Conditioning set Z.
        approx (str, optional): Approximation method ('perm', 'sw', or 'hbe'). Defaults to 'perm'.
        nperms (int, optional): Number of permutations for permutation testing. Defaults to 200.
        width_xz (str or float, optional): Kernel bandwidth for X and Z. Defaults to "median".
        width_y (str or float, optional): Kernel bandwidth for Y. Defaults to "median".
        width_z (str or float, optional): Kernel bandwidth for Z. Defaults to "median".
        thresh (float, optional): Threshold for eigenvalue filtering. Defaults to 1e-5.
        lamda (float, optional): Regularization parameter. Defaults to 0.001.

    Returns:
        tuple: P-value and test statistic.
    """
    T = len(x)
    x = x.reshape(T, -1)
    y = y.reshape(T, -1)
    z = z.reshape(T, -1)
    device = x.device

    x = normalize_gpu(x)
    y = normalize_gpu(y)
    z = normalize_gpu(z)
    lamda = torch.tensor(lamda, device=x.device)
    thresh = torch.tensor(thresh, device=x.device)

    Kx = apply_H(rbf_kernel_gpu(torch.column_stack((x, z / 2.0)), width_xz))
    Ky = apply_H(rbf_kernel_gpu(y, width_y))
    Kz = apply_H(rbf_kernel_gpu(z, width_z))

    eye_T = torch.eye(T, device=device)
    P1 = eye_T - torch.matmul(Kz, torch.linalg.solve(Kz + lamda * eye_T, eye_T))
    Kxz = torch.matmul(torch.matmul(P1, Kx), P1.T)
    Kyz = torch.matmul(torch.matmul(P1, Ky), P1.T)
    stat = torch.einsum("ij,ji->", Kxz, Kyz)  # Trace(Kxz @ Kyz)

    if approx == "perm":
        stats = []
        for i in range(nperms):
            perm = np.random.permutation(T)
            Kxz2 = torch.matmul(torch.matmul(P1, Kx[perm][:, perm]), P1.T)
            stats.append(torch.einsum("ij,ji->", Kxz2, Kyz).item())  # Trace(Kxz @ Kyz)
        stats = np.array(stats)
        p = (stats >= stat.item()).mean()
        return p, stat.cpu().item()

    eig_Kxz, eivx = torch.linalg.eigh((Kxz + Kxz.T) / 2)
    eig_Kyz, eivy = torch.linalg.eigh((Kyz + Kyz.T) / 2)

    x_ix = torch.argwhere(eig_Kxz > torch.max(eig_Kxz) * thresh).ravel()
    y_ix = torch.argwhere(eig_Kyz > torch.max(eig_Kyz) * thresh).ravel()

    eig_Kxz = eig_Kxz[x_ix]
    eivx = eivx[:, x_ix]
    eig_Kyz = eig_Kyz[y_ix]
    eivy = eivy[:, y_ix]

    eiv_prodx = eivx * torch.sqrt(eig_Kxz)
    eiv_prody = eivy * torch.sqrt(eig_Kyz)

    n_eigx = eiv_prodx.shape[1]
    n_eigy = eiv_prody.shape[1]
    size_u = n_eigx * n_eigy

    uu = eiv_prodx.unsqueeze(-1) * eiv_prody.unsqueeze(-2)
    uu = uu.reshape(T, -1)

    if size_u > T:
        uu_prod = torch.mm(
            uu, uu.T
        )  # torch.einsum("ij,ji->i", uu, uu.T)  # diagonal(uu @ uu.T)  # np.dot(uu, uu.T)
    else:
        uu_prod = torch.mm(
            uu.T, uu
        )  # torch.einsum("ij,ji->i", uu.T, uu)  # diagonal(uu.T @ uu)
    eig_uu = torch.linalg.eigvalsh(uu_prod)
    eig_uu = eig_uu[torch.argwhere(eig_uu > torch.max(eig_uu) * thresh).ravel()]

    if approx == "bootstrap":
        if len(eig_uu) * T < 1e6:
            f_rand1 = torch.randn((len(eig_uu), nperms), device=device) ** 2
            null_dstr = eig_uu @ f_rand1
        else:  # for Memory
            length = int(np.max([np.floor(1e6 // T), 100]))
            max_iter = int(np.floor(len(eig_uu) / length))
            null_dstr = 0
            for i in range(max_iter):
                sub_eig_uu = eig_uu[i * length : (i + 1) * length]
                f_rand1 = torch.randn((len(sub_eig_uu), nperms), device=device) ** 2
                null_dstr += sub_eig_uu @ f_rand1
        null_dstr = np.array(torch.tensor(null_dstr).cpu())
        p_val = np.array(null_dstr > stat.item()).mean()
        return p_val, stat.item()

    elif approx == "sw":
        mean_appr = eig_uu.sum()
        var_appr = 2 * torch.sum(eig_uu**2)
        k_appr = mean_appr**2 / var_appr
        theta_appr = var_appr / mean_appr
        stat, k_appr, theta_appr = stat.cpu(), k_appr.cpu(), theta_appr.cpu()
        p_val = 1 - gamma.cdf(stat, a=k_appr, scale=theta_appr)
        return p_val, stat.item()

    elif approx == "hbe":  # approx == "hbe"
        K1 = eig_uu.sum()
        K2 = 2 * (eig_uu**2).sum()
        K3 = 8 * (eig_uu**3).sum()
        nu = 8 * (K2**3) / (K3**2)
        k = nu / 2
        x = ((2 * nu / K2) ** (0.5)) * (stat - K1) + nu
        x, k = x.cpu(), k.cpu()
        p_val = 1 - gamma.cdf(x, a=k, scale=2)
        stat = stat.cpu()
        return p_val, stat.item()
    else:
        raise Exception(f"{approx} for KCIT Has not bben implemented")


class KCIGPU(CIT_Base):
    """
    GPU-accelerated implementation of the Kernel-based Conditional Independence Test (KCI).
    """

    def __init__(self, data, device=None, **kwargs):
        """
        Initialize the KCIGPU class.

        Args:
            data (numpy.ndarray): Input data matrix.
            device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to None.
            kwargs: Additional parameters for KCI.
        """
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(data, method="kcigpu", **kwargs)

        self.kci_ui_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ["width_x", "width_y", "approx", "nperms"]
        }
        self.kci_ci_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ["width_xz", "width_y", "width_z", "approx", "nperms"]
        }
        self.kci_ui_kwargs.setdefault("approx", kwargs.get("approx", "hbe"))
        self.kci_ci_kwargs.setdefault("approx", kwargs.get("approx", "hbe"))

        if not self._has_nan:
            self.assert_input_data_is_valid()
        self.method = "kcigpu"
        self._data = data
        self.n_actual_tests = 0
        self.n_tests = 0
        self.nperms = kwargs.get("nperms", 200)
        self.approx = kwargs.get("approx", "hbe")
        self.width_xz = kwargs.get("width_xz", "median")
        self.width_z = kwargs.get("width_z", "median")
        self.width_x = kwargs.get("width_x", "median")
        self.width_y = kwargs.get("width_y", "median")
        self.param_hash = (
            f"T{self._data.shape[0]}-{self.approx}-{self.nperms}-"
            f"{self.width_xz}-{self.width_y}-{self.width_z}"
        )

    def get_data(self):
        return self._data

    def set_data(self, d):
        self._data = d
        self._has_nan = bool(np.isnan(d).any())
        self.data_torch = torch.Tensor(np.asarray(d).copy()).to(self.device)

    data = property(get_data, set_data)

    def __call__(self, X, Y, condition_set=None):
        """
        Perform the KCI test for given variables.

        Args:
            X (int): Index of variable X.
            Y (int): Index of variable Y.
            condition_set (list, optional): Indices of conditioning set Z. Defaults to None.

        Returns:
            tuple: P-value and test statistic.
        """
        self.n_tests += 1
        self.n_actual_tests += 1

        cache_key = self._get_array_hash(X, Y, condition_set)
        cache_key = f"{self.method}-{cache_key}"
        combined_hash = cache_key, self.param_hash

        if combined_hash in self.pvalue_cache:
            p, stat = self.pvalue_cache[combined_hash]
            if combined_hash not in self.pvalue_local.keys():
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

        p, stat = (
            kci_uind(data_t[:, X], data_t[:, Y], **self.kci_ui_kwargs)
            if condition_set is None or len(condition_set) == 0
            else kci_cind(
                data_t[:, X],
                data_t[:, Y],
                data_t[:, condition_set],
                **self.kci_ci_kwargs,
            )
        )
        self.pvalue_cache[combined_hash] = (
            p,
            stat,
        )
        self.pvalue_local[combined_hash] = (
            p,
            stat,
        )
        self.save_incremental_cache(cache_key, self.param_hash, p, stat)
        return p, stat
