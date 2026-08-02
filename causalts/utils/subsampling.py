# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Detect temporal aggregation / subsampling in a multivariate time series.

Core idea (Gong et al. 2015, ICML "Discovering Temporal Causal Relations from
Subsampled Data", Thm 1): if an observed series y is a subsampling of a faster
VAR(1) with contemporaneously-INDEPENDENT innovations, then the residuals of a
VAR(1) fit to y are an aggregate over the skipped innovations and are therefore
CONTEMPORANEOUSLY CORRELATED across variables. So:

    significant contemporaneous correlation in VAR residuals  ==>
        temporal aggregation / subsampling is a candidate explanation.

IMPORTANT HONESTY CAVEAT (Danks & Plis 2013/2014; Hyttinen et al. 2016):
    contemporaneous residual correlation is NECESSARY but NOT SUFFICIENT for
    subsampling. The same signature is produced by (a) genuine instantaneous
    causation and (b) true latent confounding. These three are NOT separable
    from a single slow-sampled series by any local per-edge test. This detector
    therefore reports a *flag + evidence*, not a proof.

Depends only on numpy/scipy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class DetectionResult:
    contemp_stat: float  # Bartlett test statistic on residual corr matrix
    contemp_dof: int  # chi-square degrees of freedom, d(d-1)/2
    contemp_pvalue: float  # p-value: H0 = residuals contemporaneously uncorrelated
    max_abs_offdiag_corr: float  # largest |residual correlation| off the diagonal
    n_sig_pairs: int  # # variable pairs with individually significant corr
    n_pairs: int  # total pairs = d(d-1)/2
    nongaussian_fraction: float  # fraction of residual dims failing a normality test
    nongaussian_min_p: float  # smallest per-dim normality p-value
    lag: int  # VAR order used
    alpha: float
    verdict: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def subsampling_flagged(self) -> bool:
        return self.contemp_pvalue < self.alpha


def _fit_var(X: np.ndarray, lag: int):
    """OLS VAR(lag) with intercept. Returns (beta, resid).

    beta has shape (d*lag+1, d); rows are [intercept; lag-1 block; ...; lag-`lag`
    block] so that  y_t = [1, y_{t-1}, ..., y_{t-lag}] @ beta.
    """
    T, d = X.shape
    if T <= lag + d * lag + 1:
        raise ValueError("series too short for the requested VAR lag / dimension")
    rows = T - lag
    Z = np.ones((rows, d * lag + 1))
    for i in range(1, lag + 1):
        Z[:, 1 + (i - 1) * d : 1 + i * d] = X[lag - i : T - i]
    Y = X[lag:]
    beta, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    return beta, Y - Z @ beta


def _simulate_var(beta, lag, resid, X0, rng):
    """Simulate a VAR(lag) series under H0 (contemporaneously INDEPENDENT
    innovations), by resampling each residual column independently (so the null has
    the right marginals but zero cross-correlation). X0 seeds the first `lag` rows.
    """
    n, d = resid.shape
    T = n + lag
    E = np.empty((n, d))
    for j in range(d):
        E[:, j] = resid[rng.integers(0, n, size=n), j]
    Y = np.empty((T, d))
    Y[:lag] = X0[:lag]
    for t in range(lag, T):
        z = np.empty(d * lag + 1)
        z[0] = 1.0
        for i in range(1, lag + 1):
            z[1 + (i - 1) * d : 1 + i * d] = Y[t - i]
        Y[t] = z @ beta + E[t - lag]
    return Y


def _bartlett_stat(resid: np.ndarray) -> tuple[float, float]:
    """Bartlett statistic -(n-1-(2d+5)/6)*log(det(R)) and max|off-diag corr|."""
    n, d = resid.shape
    R = np.corrcoef(resid, rowvar=False)
    eig = np.clip(np.linalg.eigvalsh(R), 1e-8, None)  # stable log-det
    logdet = float(np.sum(np.log(eig)))
    stat = -(n - 1 - (2 * d + 5) / 6.0) * logdet
    max_abs = float(np.max(np.abs(R - np.eye(d))))
    return stat, max_abs


def _corr_test(X, lag, beta, resid, null, n_boot, seed):
    """Test that the VAR residual correlation matrix is the identity.

    null="asymptotic": Bartlett chi2 (d(d-1)/2 dof). Fast, but LIBERAL when T is
        small relative to d -- VAR OLS estimation error injects real residual
        cross-correlation, so the chi2 (and even a residual permutation) over-fire.
    null="bootstrap": parametric bootstrap. Simulate under H0 (the fitted VAR with
        contemporaneously INDEPENDENT innovations), refit the VAR, and rebuild the
        Bartlett-statistic null -- this reproduces the estimation-error inflation, so
        the p-value is calibrated at any d/T. Use for short series (financial T~100s).
    Returns (stat, dof, pvalue, max|offdiag|).
    """
    n, d = resid.shape
    stat, max_abs = _bartlett_stat(resid)
    dof = d * (d - 1) // 2
    if null == "asymptotic":
        pvalue = float(stats.chi2.sf(stat, dof))
    elif null == "bootstrap":
        rng = np.random.default_rng(seed)
        ge = 1  # +1 for the observed (standard bootstrap correction)
        for _ in range(n_boot):
            Yb = _simulate_var(beta, lag, resid, X[:lag], rng)
            if _bartlett_stat(_fit_var(Yb, lag)[1])[0] >= stat:
                ge += 1
        pvalue = ge / (n_boot + 1)
    else:
        raise ValueError(f"unknown null: {null!r}")
    return stat, dof, pvalue, max_abs


def _pairwise_sig_count(resid: np.ndarray, alpha: float) -> int:
    """Count variable pairs whose residual correlation is individually significant
    (Fisher z-test, Bonferroni-corrected across pairs)."""
    n, d = resid.shape
    R = np.corrcoef(resid, rowvar=False)
    n_pairs = d * (d - 1) // 2
    if n_pairs == 0:
        return 0
    iu = np.triu_indices(d, k=1)
    r = np.clip(R[iu], -0.999999, 0.999999)
    z = np.arctanh(r) * np.sqrt(n - 3)
    p = 2 * stats.norm.sf(np.abs(z))
    return int(np.sum(p < alpha / n_pairs))


def _nongaussianity(resid: np.ndarray, alpha: float) -> tuple[float, float]:
    """Per-dimension D'Agostino-Pearson normality test on residuals.

    Returns (fraction of dims that REJECT normality, min p-value). Non-Gaussian
    residuals are the regime where Gong Thm 2 makes the fast transition matrix
    identifiable.
    """
    d = resid.shape[1]
    pvals = []
    for j in range(d):
        try:
            _, p = stats.normaltest(resid[:, j])
        except ValueError:
            p = 1.0
        pvals.append(p)
    pvals = np.asarray(pvals)
    frac = float(np.mean(pvals < alpha))
    return frac, float(np.min(pvals))


def detect_subsampling(
    X: np.ndarray,
    lag: int = 1,
    alpha: float = 0.05,
    null: str = "auto",
    n_perm: int = 299,
) -> DetectionResult:
    """Flag whether ``X`` shows the contemporaneous-correlation signature of
    temporal aggregation / subsampling.

    Parameters
    ----------
    X : numpy.ndarray or pandas.DataFrame
        Observed (possibly subsampled) multivariate series of shape (T, d).
    lag : int, optional
        VAR order to fit. Gong's model implies the subsampled process is still
        VAR(1), so ``lag=1`` (default) is the principled choice.
    alpha : float, optional
        Significance level for the contemporaneous-correlation test (default 0.05).
    null : {"auto", "asymptotic", "bootstrap"}, optional
        Null distribution for the Bartlett statistic. "asymptotic" uses the chi2
        approximation (fast, but liberal at small T relative to d). "bootstrap"
        uses a parametric bootstrap (calibrated at small T). "auto" (default)
        picks bootstrap when T < 100*d, else asymptotic.
    n_perm : int, optional
        Bootstrap replications for the bootstrap null (default 299).

    Returns
    -------
    DetectionResult
        Dataclass with the test statistic, p-value, residual-correlation and
        non-Gaussianity diagnostics, a ``verdict`` string, ``notes`` (including the
        non-separability caveat), and the ``subsampling_flagged`` property.
    """
    if hasattr(X, "values"):
        X = X.values
    X = np.asarray(X, dtype=float)
    beta, resid = _fit_var(X, lag)
    n, d = resid.shape
    if null == "auto":
        null = "bootstrap" if n < 100 * d else "asymptotic"
    stat, dof, pvalue, max_abs = _corr_test(X, lag, beta, resid, null, n_perm, seed=0)
    n_sig = _pairwise_sig_count(resid, alpha)
    d = X.shape[1]
    n_pairs = d * (d - 1) // 2
    ng_frac, ng_min_p = _nongaussianity(resid, alpha)

    res = DetectionResult(
        contemp_stat=stat,
        contemp_dof=dof,
        contemp_pvalue=pvalue,
        max_abs_offdiag_corr=max_abs,
        n_sig_pairs=n_sig,
        n_pairs=n_pairs,
        nongaussian_fraction=ng_frac,
        nongaussian_min_p=ng_min_p,
        lag=lag,
        alpha=alpha,
    )

    if pvalue < alpha:
        res.verdict = "SUBSAMPLING/AGGREGATION CANDIDATE"
        res.notes.append(
            f"VAR({lag}) residuals are contemporaneously correlated "
            f"(Bartlett chi2={stat:.1f}, dof={dof}, p={pvalue:.2e}; "
            f"max|offdiag r|={max_abs:.3f}, {n_sig}/{n_pairs} pairs individually "
            f"significant). Consistent with temporal aggregation of a faster process."
        )
        res.notes.append(
            "NOT proof: instantaneous causation and true latent confounding produce "
            "the same signature and are not locally separable (Danks & Plis 2013/2014)."
        )
        if ng_frac >= 0.5:
            res.notes.append(
                f"Residuals are largely non-Gaussian ({ng_frac:.0%} of dims reject "
                f"normality) -- the regime where Gong (2015) Thm 2 makes the fast "
                f"transition matrix identifiable in principle."
            )
        else:
            res.notes.append(
                f"Residuals look mostly Gaussian ({ng_frac:.0%} of dims reject "
                f"normality) -- the regime where the fast transition matrix is NOT "
                f"identifiable from 2nd-order statistics alone (Palm-Nijman)."
            )
    else:
        res.verdict = "NO AGGREGATION SIGNATURE"
        res.notes.append(
            f"VAR({lag}) residuals show no significant contemporaneous correlation "
            f"(Bartlett p={pvalue:.2e} >= alpha). No evidence of subsampling from this "
            f"test (a genuinely fast-sampled process, or aggregation too weak to detect)."
        )
    return res
