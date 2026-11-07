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
    copula-normalized (X, Y) conditioned on Z. For multivariate X or Y it is
    the multivariate Gaussian CMI, with a chi-square p-value (Bartlett's
    approximation to Wilks' lambda).

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
        param_hash = (
            f"gcmi_gpu-jit{self.jitter}-seed{self._base_seed}-ms{self._min_samples}"
            f"{self._oi_tag}"
        )
        if np.ndim(X) > 0 and len(X) > 1 or np.ndim(Y) > 0 and len(Y) > 1:
            # Builds before the multivariate statistic tested X[0] vs X[1]:
            # a separate key keeps their cached rows out.
            param_hash += "-mv2"
            # Same column order for every cache-equivalent call (the key sorts
            # X, Y and Z and lets X and Y swap): the rank cutoff can otherwise
            # keep slightly different directions for another ordering.
            X, Y = sorted([sorted(np.atleast_1d(X)), sorted(np.atleast_1d(Y))])
            condition_set = sorted(condition_set)
        cached = self.pvalue_cache.get((cache_key, param_hash))
        if cached is not None:
            return cached[0], cached[1]

        X, Y, condition_set = self._canonical_query(X, Y, condition_set)
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
        val, pval = self._gcmi_pval(
            raw,
            n_z=z.shape[1] if z is not None else 0,
            n_x=x.shape[1],
            n_y=y.shape[1],
            seed=self._test_seed(cache_key),
        )

        self.pvalue_cache[(cache_key, param_hash)] = [pval, val]
        self.save_incremental_cache(cache_key, param_hash, pval, val)
        return pval, val

    def _copula_normalize(self, data_t, seed=None):
        """Rank → uniform → Gaussian (probit) transform.

        Parameters
        ----------
        data_t : torch.Tensor, shape (T, d)
        seed : int or None
            Seed for the tie-breaking jitter; None leaves it unseeded.

        Returns
        -------
        normalized : torch.Tensor, shape (T, d)
        """
        T = data_t.shape[0]
        if self.jitter > 0:
            # CPU generator, then moved: device generators differ by backend.
            gen = torch.Generator()
            if seed is None:
                gen.seed()
            else:
                gen.manual_seed(seed)
            noise = torch.randn(
                data_t.shape, generator=gen, device="cpu", dtype=data_t.dtype
            )
            data_t = data_t + self.jitter * noise.to(data_t.device)

        # Double argsort → ranks in [0, T-1]
        ranks = data_t.argsort(dim=0).argsort(dim=0).float()
        # Scale to open interval (0, 1)
        uniform = (ranks + 0.5) / T
        # Probit: Φ⁻¹(u) = √2 · erfinv(2u − 1)
        return torch.erfinv(2.0 * uniform - 1.0) * (2.0**0.5)

    def _gcmi_pval(self, raw, n_z, n_x=1, n_y=1, seed=None):
        """Compute GCMI statistic and analytic p-value.

        Parameters
        ----------
        raw : np.ndarray, shape (T, d)  — columns: [X..., Y..., Z...]
        n_z : int — number of Z columns
        n_x, n_y : int — number of X and Y columns. With more than one, the
            CMI is the multivariate Gaussian one and the p-value comes from
            Bartlett's chi-square approximation (see ``_gcmi_multivariate``).

        Returns
        -------
        cmi : float
        pval : float
        """
        T = raw.shape[0]
        dtype = torch.float32 if str(self.device).startswith("mps") else torch.float64
        data_t = torch.tensor(raw, dtype=dtype, device=self.device)
        normed = self._copula_normalize(data_t, seed=seed)  # (T, d)

        # Compute precision matrix (inverse covariance) on the normalized data
        cov = torch.cov(normed.T)  # (d, d)
        # Regularize for numerical stability
        cov = cov + 1e-10 * torch.eye(cov.shape[0], device=self.device, dtype=cov.dtype)

        if n_x > 1 or n_y > 1:
            return self._gcmi_multivariate(cov, T, n_x, n_y, n_z)

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

    @staticmethod
    def _gcmi_multivariate(cov, T, n_x, n_y, n_z, tol=1e-6):
        """Gaussian CMI of multivariate X and Y given Z, with its p-value.

        Works on the copula-normalized covariance: the (X, Y) covariance given Z
        (Schur complement, pseudo-inverse of the Z block), with X and Y each
        reduced to their directions of non-negligible residual variance. The
        canonical correlations rho_i between those give
        ``CMI = -0.5 * sum log(1 - rho_i^2)``, which is minus half the log of
        Wilks' lambda for X _|_ Y | Z; Bartlett's statistic
        ``(T - r_z - (r_x + r_y + 3) / 2) * 2 * CMI`` is approximately
        chi-square with ``r_x * r_y`` degrees of freedom under the null (r_*
        the ranks). Redundant columns therefore do not hide a dependence, and
        too few rows for the ranks give an undefined p-value (NaN), as the
        univariate path does when it has no degrees of freedom. Callers must
        handle NaN themselves: ``p > alpha`` reads it as dependent (CDNOTS
        keeps the edge) and ``p <= alpha`` as independent (``CITestAdapter``
        drops it).
        """
        # The normal scores are float32 (ranks via .float()), so a direction
        # with relative variance below ~1e-6 is rounding noise: ``tol`` sets
        # every rank and the Z pseudo-inverse with that one cutoff.
        cov = cov.cpu().double()  # MPS has no float64
        if T < 2 or not torch.isfinite(cov).all():
            return 0.0, np.nan  # no sample covariance to decompose
        ix = list(range(n_x))
        iy = list(range(n_x, n_x + n_y))
        ixy = ix + iy
        iz = list(range(n_x + n_y, n_x + n_y + n_z))
        scale = float(torch.diagonal(cov).max())

        def _basis(m):
            """Whitening basis of the non-negligible directions of PSD m."""
            vals, vecs = torch.linalg.eigh(m)
            keep = vals > tol * scale
            return vecs[:, keep] / vals[keep].sqrt()

        resid = cov[ixy][:, ixy]
        r_z = 0
        if iz:
            szz = cov[iz][:, iz]
            vals = torch.linalg.eigvalsh(szz)
            r_z = int((vals > tol * scale).sum())
            s_xz = cov[ixy][:, iz]
            szz_pinv = torch.linalg.pinv(szz, atol=tol * scale, hermitian=True)
            resid = resid - s_xz @ szz_pinv @ s_xz.T
        if T - 1 <= r_z + 1:
            # Z leaves no residual degrees of freedom: a zero residual rank
            # below would reflect the sample, not X or Y being set by Z.
            return 0.0, np.nan
        wx = _basis(resid[:n_x, :n_x])
        wy = _basis(resid[n_x:, n_x:])
        r_x, r_y = wx.shape[1], wy.shape[1]
        if r_x == 0 or r_y == 0:
            return 0.0, 1.0  # X or Y is determined by Z: nothing left to test
        rho = torch.linalg.svdvals(wx.T @ resid[:n_x, n_x:] @ wy).numpy()
        rho = np.clip(rho, 0.0, 1.0 - 1e-15)
        cmi = float(-0.5 * np.log1p(-(rho**2)).sum())
        bartlett = T - r_z - (r_x + r_y + 3) / 2.0
        if T - 1 < r_x + r_y + r_z or bartlett <= 0:
            return cmi, np.nan
        pval = stats.chi2.sf(2.0 * bartlett * cmi, r_x * r_y)
        return cmi, float(pval)
