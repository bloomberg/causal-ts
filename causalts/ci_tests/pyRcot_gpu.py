# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import torch
from scipy.stats import gamma

from .cit_test import CIT_Base


def normalize_gpu(a):
    """Normalize a tensor by subtracting the mean and dividing by the std.

    Parameters
    ----------
    a : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Normalized tensor.
    """
    m = torch.mean(a, axis=0)
    std = torch.std(a, axis=0)
    std = torch.clamp(std, min=1e-10)
    return (a - m) / std


def random_fourier_features(
    x, w=None, b=None, num_f=25, sigma=None, seed=None, device=None
):
    """Generate random Fourier features for kernel approximation.

    Parameters
    ----------
    x : torch.Tensor
        Input data.
    w : torch.Tensor, optional
        Weight matrix. Default is None.
    b : torch.Tensor, optional
        Bias vector. Default is None.
    num_f : int, optional
        Number of features. Default is 25.
    sigma : float, optional
        Kernel bandwidth. Default is None.
    seed : int, optional
        Random seed. Default is None.
    device : str, optional
        Device to use ('cuda' or 'cpu'). Default is None.

    Returns
    -------
    torch.Tensor
        Random Fourier features.
    """

    T, c = x.shape
    pi = np.pi

    if (sigma is None) or (sigma == 0):
        sigma = 1

    if w is None:
        if seed is not None:
            torch.manual_seed(seed)

        w = torch.randn((num_f, c), dtype=torch.double).to(device) / sigma

        b = torch.rand(num_f, dtype=torch.double).to(device)
        b = 2 * pi * b.reshape(-1, 1)
        # b = 2 * pi * b.repeat_interleave(T,dim=0).reshape(-1,T)
    feat = np.sqrt(2) * (torch.cos(w @ x.T + b)).T

    return feat


def Sta_perm(f_x, f_y, r, num_f2):
    """Compute the test statistic for permutation testing.

    Parameters
    ----------
    f_x : torch.Tensor
        Fourier features of X.
    f_y : torch.Tensor
        Fourier features of Y.
    r : int
        Sample size.
    num_f2 : int
        Number of Fourier features.

    Returns
    -------
    torch.Tensor
        Test statistic.
    """
    Cxy = torch.cov(torch.cat((f_x.T, f_y.T), axis=0))[:num_f2, num_f2:]
    Sta = r * torch.sum(Cxy**2)
    return Sta


def RIT(
    x,
    y,
    num_f2=10,
    seed=None,
    device=None,
    approx="hbe",
    nperm=200,
    max_sample_4_std_est=500,
):
    """Perform the Randomized Independence Test (RIT).

    Parameters
    ----------
    x : torch.Tensor
        Input variable X.
    y : torch.Tensor
        Input variable Y.
    num_f2 : int, optional
        Number of Fourier features. Default is 10.
    seed : int, optional
        Random seed. Default is None.
    device : str, optional
        Device to use ('cuda' or 'cpu'). Default is None.
    approx : str, optional
        Approximation method ('hbe', 'sw', or 'perm'). Default is 'hbe'.
    nperm : int, optional
        Number of permutations for permutation testing. Default is 200.
    max_sample_4_std_est : int, optional
        Maximum sample size for standard deviation estimation.
        Default is 500.

    Returns
    -------
    p : float
        P-value.
    Sta : float
        Test statistic.
    """

    # device = "cuda" if torch.cuda.is_available() else "cpu"

    if not isinstance(x, torch.Tensor):
        x = torch.Tensor(x)
    x = x.double().to(device)
    if not isinstance(y, torch.Tensor):
        y = torch.Tensor(y)
    y = y.double().to(device)

    if (torch.std(x) == 0) or (torch.std(y) == 0):
        return 1, 0  # if sd of x or sd of y == 0 then x and y are independent

    T = len(x)
    x = x.reshape(T, -1)
    y = y.reshape(T, -1)

    r1 = max_sample_4_std_est if T > max_sample_4_std_est else T

    x = normalize_gpu(x)
    y = normalize_gpu(y)

    sigma_x = torch.median(torch.nn.functional.pdist(x[:r1,], p=2))
    sigma_y = torch.median(torch.nn.functional.pdist(y[:r1,], p=2))

    four_x = random_fourier_features(
        x, w=None, b=None, num_f=num_f2, sigma=sigma_x, seed=seed, device=device
    )
    four_y = random_fourier_features(
        y, w=None, b=None, num_f=num_f2, sigma=sigma_y, seed=seed, device=device
    )
    f_x = normalize_gpu(four_x)
    f_y = normalize_gpu(four_y)

    Sta = Sta_perm(f_x, f_y, T, num_f2).item()
    if approx == "perm":
        Stas = []
        for i in range(nperm):
            perm = torch.randperm(T)
            Sta_p = Sta_perm(f_x[perm, :], f_y, T, num_f2)
            Stas.append(Sta_p)
        Stas = np.array(torch.tensor(Stas).cpu())
        p = 1 - len(Stas[Stas < Sta]) / len(Stas)

    else:
        res_x = f_x - f_x.mean(axis=0).reshape(-1)
        res_y = f_y - f_y.mean(axis=0).reshape(-1)
        res = (res_x.unsqueeze(-2) * res_y.unsqueeze(-1)).reshape(res_x.size(0), -1)

        Cov = 1 / T * ((res.T) @ res)
        w = torch.linalg.eigvalsh(Cov)
        eig_thresh = max(1e-10, 1e-10 * w.max().item()) if w.numel() > 0 else 1e-10
        eig_ = w[w > eig_thresh]

        if approx == "sw":  # Satterthwaite-Welch
            w = eig_.sum()
            u = (eig_**2).sum() / (w**2)
            k = 0.5 / u
            theta = 2 * u * w
            a, scale = k.item(), theta.item()
            p = 1 - gamma.cdf(Sta, a=a, scale=scale)
        if approx == "hbe":  # Hall-Buckley-Eagleson
            K_1 = eig_.sum()
            K_2 = 2 * (eig_**2).sum()
            K_3 = 8 * (eig_**3).sum()
            nu = 8 * (K_2**3) / (K_3**2)
            k = nu / 2
            xx = ((2 * nu / K_2).sqrt()) * (Sta - K_1) + nu
            xx, a = xx.item(), k.item()
            p = 1 - gamma.cdf(xx, a=a, scale=2)
    return max(0, p), Sta


def rcot(
    x,
    y,
    z=None,
    num_f=25,
    num_f2=10,
    seed=None,
    device=None,
    approx="hbe",
    nperm=200,
    max_sample_4_std_est=500,
):
    """Perform the Randomized Conditional Independence Test (RCOT).

    Parameters
    ----------
    x : torch.Tensor
        Input variable X.
    y : torch.Tensor
        Input variable Y.
    z : torch.Tensor, optional
        Conditioning set Z. Default is None.
    num_f : int, optional
        Number of Fourier features for Z. Default is 25.
    num_f2 : int, optional
        Number of Fourier features for X and Y. Default is 10.
    seed : int, optional
        Random seed. Default is None.
    device : str, optional
        Device to use ('cuda' or 'cpu'). Default is None.
    approx : str, optional
        Approximation method ('hbe', 'sw', or 'perm'). Default is 'hbe'.
    nperm : int, optional
        Number of permutations for permutation testing. Default is 200.
    max_sample_4_std_est : int, optional
        Maximum sample size for standard deviation estimation.
        Default is 500.

    Returns
    -------
    p : float
        P-value.
    Sta : float
        Test statistic.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # Unconditional Testing
    if z is None or len(z) == 0:
        return RIT(x, y, num_f2=num_f2, approx=approx, seed=seed, device=device)

    if not isinstance(x, torch.Tensor):
        x = torch.Tensor(x)
    x = x.double().to(device)
    if not isinstance(y, torch.Tensor):
        y = torch.Tensor(y)
    y = y.double().to(device)

    if torch.std(x) == 0 or torch.std(y) == 0:
        return 1, 0  # if sd of x or sd of y == 0 then x and y are independent

    T = len(x)
    x = x.reshape(T, -1)
    y = y.reshape(T, -1)

    if not isinstance(z, torch.Tensor):
        z = torch.Tensor(z)
    z = z.double().to(device)
    z = z.reshape(T, -1)
    r1 = max_sample_4_std_est if T > max_sample_4_std_est else T

    x = normalize_gpu(x)
    y = normalize_gpu(y)
    z = normalize_gpu(z)

    sigma_x = torch.median(torch.nn.functional.pdist(x[:r1,], p=2))
    sigma_y = torch.median(torch.nn.functional.pdist(y[:r1,], p=2))
    sigma_z = torch.median(torch.nn.functional.pdist(z[:r1,], p=2))

    four_x = random_fourier_features(
        x, w=None, b=None, num_f=num_f2, sigma=sigma_x, seed=seed, device=device
    )
    four_y = random_fourier_features(
        y, w=None, b=None, num_f=num_f2, sigma=sigma_y, seed=seed, device=device
    )
    four_z = random_fourier_features(
        z, w=None, b=None, num_f=num_f, sigma=sigma_z, seed=seed, device=device
    )

    f_x = normalize_gpu(four_x)
    f_y = normalize_gpu(four_y)
    f_z = normalize_gpu(four_z)

    Cxy = torch.cov(torch.cat((f_x.T, f_y.T), axis=0))[:num_f2, num_f2:]
    Czz = torch.cov(torch.cat((f_z.T, f_z.T), axis=0))[:num_f, :num_f]

    Imat = torch.eye(num_f, device=x.device)
    L = torch.linalg.cholesky(Czz + Imat * 1e-10)
    L_inv = torch.linalg.solve_triangular(L, Imat, upper=False)

    i_Czz = torch.matmul(L_inv.T, L_inv)  # numf,numf

    Cxz = torch.cov(torch.cat((f_x.T, f_z.T), axis=0))[:num_f2, num_f2:]  # numf2,numf
    Czy = torch.cov(torch.cat((f_z.T, f_y.T), axis=0))[:num_f, num_f:]  # numf,numf2

    z_i_Czz = f_z @ i_Czz
    e_x_z = z_i_Czz @ Cxz.T
    e_y_z = z_i_Czz @ Czy
    res_x = f_x - e_x_z
    res_y = f_y - e_y_z

    if approx == "perm":
        Sta = Sta_perm(res_x, res_y, T, num_f2).item()
        Stas = []
        for i in range(nperm):
            perm = torch.randperm(T)
            Sta_p = Sta_perm(res_x[perm], res_y, T, num_f2)
            Stas.append(Sta_p)
        # Stas = np.array(Stas)
        Stas = np.array(torch.tensor(Stas).cpu())
        p = np.array(1 - len(Stas[Stas < Sta]) / len(Stas))

    else:
        matmul = Cxz @ (i_Czz @ Czy)
        Cxy_z = Cxy - matmul  # less accurate for permutation testing
        Sta = T * torch.sum(Cxy_z**2).item()

        res = (res_x.unsqueeze(-2) * res_y.unsqueeze(-1)).reshape(res_x.size(0), -1)
        Cov = 1 / T * ((res.T) @ res)
        w = torch.linalg.eigvalsh(Cov)
        eig_thresh = max(1e-10, 1e-10 * w.max().item()) if w.numel() > 0 else 1e-10
        eig_ = w[w > eig_thresh]

        if approx == "sw":  # Satterthwaite-Welch
            w = (eig_).sum()
            u = (eig_**2).sum() / (w**2)
            k = 0.5 / u
            theta = 2 * u * w
            a, scale = k.item(), theta.item()
            p = 1 - gamma.cdf(Sta, a=a, scale=scale)
        if approx == "hbe":  # Hall-Buckley-Eagleson
            K_1 = eig_.sum()
            K_2 = 2 * (eig_**2).sum()
            K_3 = 8 * (eig_**3).sum()
            nu = 8 * (K_2**3) / (K_3**2)
            k = nu / 2
            xx = ((2 * nu / K_2).sqrt()) * (Sta - K_1) + nu
            xx, a = xx.item(), k.item()
            p = 1 - gamma.cdf(xx, a=a, scale=2)
    return max(0, p), Sta


class RCOTGPU(CIT_Base):
    """
    GPU-accelerated implementation of the RCOT.
    """

    def __init__(
        self,
        data=None,
        num_f=25,
        num_f2=25,
        approx="hbe",
        nperm=200,
        max_sample_4_std_es=500,
        seed=None,
        n_avg=5,
        device=None,
        **kwargs,
    ):
        """Initialize the RCOTGPU class.

        Parameters
        ----------
        data : numpy.ndarray
            Input data matrix.
        num_f : int, optional
            Number of Fourier features for Z. Default is 25.
        num_f2 : int, optional
            Number of Fourier features for X and Y. Default is 25.
        approx : str, optional
            Approximation method ('hbe', 'sw', or 'perm'). Default is 'hbe'.
        nperm : int, optional
            Number of permutations for permutation testing. Default is 200.
        max_sample_4_std_es : int, optional
            Sample size for std estimation. Default is 500.
        seed : int, optional
            Base random seed for ensemble draws. Default is None (fully random
            draws each call, which is appropriate for n_avg>1).
        n_avg : int, optional
            Number of independent RFF draws whose p-values are combined via
            harmonic mean. Reduces variance caused by RFF randomness at small
            T. Default is 5. Set to 1 to match original single-draw behaviour.
        device : str, optional
            Device to use ('cuda' or 'cpu'). Default is None.
        """
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._data = data
        if data is not None:
            self.data_torch = torch.Tensor(np.asarray(data).copy()).to(self.device)

        super().__init__(data, method="rcotgpu", **kwargs)
        self.num_f = num_f
        self.num_f2 = num_f2
        self.seed = seed
        self.n_avg = max(1, int(n_avg))
        self.approx = approx
        self.nperm = nperm  # for perm method only
        self.max_sample_4_std_es = max_sample_4_std_es
        rcot_backend = kwargs.get("rcot_backend", "python")
        self.param_hash = (
            f"T{data.shape[0]}-z{num_f}-xy{num_f2}-{approx}-{nperm}-"
            f"navg{self.n_avg}-bknd_{rcot_backend}"
        )
        self._data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d):
        self._data = d
        if d is not None:
            self._has_nan = bool(np.isnan(d).any())
            self.data_torch = torch.Tensor(np.asarray(d).copy()).to(self.device)
        else:
            self._has_nan = False

    def __call__(self, X, Y, condition_set=None):
        """Perform the RCOT test for given variables.

        Parameters
        ----------
        X : int
            Index of variable X.
        Y : int
            Index of variable Y.
        condition_set : list, optional
            Indices of conditioning set.

        Returns
        -------
        p : float
            P-value.
        stat : float
            Test statistic.
        """

        self.n_tests += 1
        cache_key = self._get_array_hash(X, Y, condition_set)
        cache_key = f"{self.method}-{cache_key}"
        combined_hash = cache_key, self.param_hash
        if combined_hash in self.pvalue_cache:
            return self.pvalue_cache[combined_hash]
        self.n_actual_tests += 1  # actualy need to compute now

        data_t = self.data_torch
        if self._has_nan:
            mask = self._pairwise_complete(X, Y, condition_set)
            if mask.sum() < self._min_samples:
                self.pvalue_cache[combined_hash] = (1.0, 0.0)
                return 1.0, 0.0
            data_t = torch.tensor(
                self._data[mask], dtype=data_t.dtype, device=self.device
            )

        if condition_set is None or len(condition_set) == 0:
            z = []
        else:
            z = data_t[:, condition_set]

        rcot_kwargs = dict(
            x=data_t[:, X],
            y=data_t[:, Y],
            z=z,
            num_f=self.num_f,
            num_f2=self.num_f2,
            device=self.device,
            approx=self.approx,
            nperm=self.nperm,
            max_sample_4_std_est=self.max_sample_4_std_es,
        )

        # Harmonic-mean averaging over multiple RFF draws stabilises p-values when
        # T is small (RFF randomness is high). For large T the approximation is
        # already stable, so a single draw is sufficient and much faster.
        T = data_t.shape[0]
        effective_n_avg = self.n_avg if T <= 300 else 1

        if effective_n_avg == 1:
            p, stat = rcot(seed=self.seed, **rcot_kwargs)
        else:
            # Combine p-values from effective_n_avg independent RFF draws via the
            # harmonic mean (Wilson 2019, PNAS). Unlike arithmetic mean, the harmonic
            # mean is dominated by small p-values, so one noisy high-p draw cannot
            # rescue a strong signal. Valid under arbitrary dependence between draws
            # sharing the same data.
            ps, stat = [], None
            for i in range(effective_n_avg):
                seed_i = None if self.seed is None else self.seed + i
                p_i, stat_i = rcot(seed=seed_i, **rcot_kwargs)
                ps.append(max(p_i, 1e-300))  # guard against exact zero
                if stat is None:
                    stat = stat_i
            p = float(effective_n_avg / np.sum(1.0 / np.array(ps)))
            p = min(p, 1.0)

        self.pvalue_cache[combined_hash] = p, stat
        self.save_incremental_cache(cache_key, self.param_hash, p, stat)
        return p, stat
