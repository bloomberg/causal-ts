# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

import sys

import numpy as np
import torch
from scipy import stats

from .cit_test import CIT_Base


class GCMIGPU(CIT_Base):
    """Fast nonlinear CI test via Gaussian Copula Mutual Information (GCMI).

    Transforms variables to Gaussian marginals via the empirical copula, then
    computes CMI analytically as the partial correlation of the normalized data.
    Detects linear and all monotone nonlinear dependencies. No shuffle test —
    uses an analytic Student's t p-value instead.

    CMI = -0.5 * log(1 - r²) where r is the partial correlation of
    copula-normalized (X, Y) conditioned on Z.

    Parameters
    ----------
    data : np.ndarray, shape (T, n_features)
        Input data array.
    device : str or None
        Torch device ('cuda', 'mps', 'cpu'). Auto-detected if None.
    jitter : float, optional (default: 1e-10)
        Tiny noise added to break rank ties in degenerate data.
    """

    def __init__(self, data, device=None, jitter=1e-10, **kwargs):
        super().__init__(data=data, method="gcmi_gpu", **kwargs)

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.jitter = jitter

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
        param_hash = "gcmi_gpu"
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

        parts = [x, y] + ([z] if z is not None else [])
        raw = np.hstack(parts)  # (T, d)

        self.n_actual_tests += 1
        val, pval = self._gcmi_pval(raw, n_z=z.shape[1] if z is not None else 0)

        self.pvalue_cache[(cache_key, param_hash)] = [pval, val]
        self.save_incremental_cache(cache_key, param_hash, pval, val)
        return pval, val

    def _copula_normalize(self, data_t):
        """Rank → uniform → Gaussian (probit) transform.

        Parameters
        ----------
        data_t : torch.Tensor, shape (T, d)

        Returns
        -------
        normalized : torch.Tensor, shape (T, d)
        """
        T = data_t.shape[0]
        if self.jitter > 0:
            data_t = data_t + self.jitter * torch.randn_like(data_t)

        # Double argsort → ranks in [0, T-1]
        ranks = data_t.argsort(dim=0).argsort(dim=0).float()
        # Scale to open interval (0, 1)
        uniform = (ranks + 0.5) / T
        # Probit: Φ⁻¹(u) = √2 · erfinv(2u − 1)
        return torch.erfinv(2.0 * uniform - 1.0) * (2.0**0.5)

    def _gcmi_pval(self, raw, n_z):
        """Compute GCMI statistic and analytic p-value.

        Parameters
        ----------
        raw : np.ndarray, shape (T, d)  — columns: [X, Y, Z...]
        n_z : int — number of Z columns

        Returns
        -------
        cmi : float
        pval : float
        """
        T = raw.shape[0]
        dtype = torch.float32 if str(self.device).startswith("mps") else torch.float64
        data_t = torch.tensor(raw, dtype=dtype, device=self.device)
        normed = self._copula_normalize(data_t)  # (T, d)

        # Compute precision matrix (inverse covariance) on the normalized data
        cov = torch.cov(normed.T)  # (d, d)
        # Regularize for numerical stability
        cov = cov + 1e-10 * torch.eye(cov.shape[0], device=self.device, dtype=cov.dtype)

        try:
            prec = torch.linalg.inv(cov)
        except Exception:
            return 0.0, 1.0

        # Partial correlation of X (col 0) and Y (col 1) given Z (cols 2..)
        p_xx = prec[0, 0]
        p_yy = prec[1, 1]
        p_xy = prec[0, 1]

        denom = torch.sqrt(p_xx * p_yy)
        if denom == 0:
            return 0.0, 1.0

        r = (-p_xy / denom).item()
        r = float(np.clip(r, -1 + 1e-15, 1 - 1e-15))

        # CMI from partial correlation (Gaussian formula)
        cmi = -0.5 * np.log(1.0 - r**2)

        # Analytic p-value: t-distribution with df = T - n_z - 2
        df = T - n_z - 2
        if df < 1:
            return cmi, np.nan
        if abs(abs(r) - 1.0) <= sys.float_info.min:
            return cmi, 0.0

        t_stat = r * np.sqrt(df / (1.0 - r**2))
        pval = stats.t.sf(abs(t_stat), df) * 2
        return cmi, float(pval)
