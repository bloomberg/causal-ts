# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Vectorized lag selection for CEDAR.

Three methods for lag significance:
1. dcor t-test (fast, no permutation, ~200x faster than permutation)
2. Circular shift permutation (proper for AR data, preserves autocorrelation)
3. BH-FDR correction across all hypotheses

All functions operate on numpy arrays.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t as t_dist

try:
    from joblib import Parallel, delayed

    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False


# ---------------------------------------------------------------------------
# Importance scores
# ---------------------------------------------------------------------------


def _dcor_pair_unbiased(source_col: np.ndarray, target_col: np.ndarray) -> float:
    """Unbiased squared distance correlation (U-centering, Székely & Rizzo 2014).

    Returns the unbiased R_U^2, which can be negative under independence
    (centered at 0 regardless of sample size).
    """
    import dcor

    return float(dcor.u_distance_correlation_sqr(source_col, target_col))


def _dcor_pair_biased(source_col: np.ndarray, target_col: np.ndarray) -> float:
    """Standard (biased) distance correlation using AVL method."""
    import dcor

    return dcor.distance_correlation(source_col, target_col, method="avl")


def _pearson_pair(source_col: np.ndarray, target_col: np.ndarray) -> float:
    """Absolute Pearson correlation between two 1-D arrays."""
    r = np.corrcoef(source_col, target_col)[0, 1]
    return abs(r) if np.isfinite(r) else 0.0


def _lasso_pair(
    sources: np.ndarray,
    target_col: np.ndarray,
    n_lags: int,
) -> np.ndarray:
    """Lag importance via LassoCV: fit all lags simultaneously as design matrix.

    Matches SyPI (Mastakouri et al., ICML 2021): the lag with the maximum
    absolute Lasso coefficient is taken as the selected lag. LassoCV with
    cv=5 chooses lambda automatically.

    Parameters
    ----------
    sources : ndarray, shape (n_lags, effective_T)
        Pre-sliced source variable at each lag (lag indices 0..n_lags-1).
    target_col : ndarray, shape (effective_T,)
        Target variable aligned to the lag window.
    n_lags : int
        Number of lags.

    Returns
    -------
    ndarray, shape (n_lags,)
        Absolute Lasso coefficients as importance scores.
    """
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler

    X = sources.T  # (effective_T, n_lags)
    y = target_col

    valid = ~np.isnan(y)
    for lag_idx in range(n_lags):
        valid &= ~np.isnan(sources[lag_idx])

    if valid.sum() < max(10, n_lags + 1):
        return np.zeros(n_lags)

    X_v = StandardScaler().fit_transform(X[valid])
    y_v = y[valid]

    try:
        reg = LassoCV(cv=5, max_iter=5000, random_state=42).fit(X_v, y_v)
        return np.abs(reg.coef_)
    except Exception:
        return np.zeros(n_lags)


def _compute_one_pair(
    sources: np.ndarray,
    target_col: np.ndarray,
    n_lags: int,
    method: str,
) -> np.ndarray:
    """Compute lag importance for one (cause, effect) pair across all lags.

    Parameters
    ----------
    sources : ndarray, shape (n_lags, effective_T)
        Pre-sliced source variable at each lag.
    target_col : ndarray, shape (effective_T,)
        Target variable (aligned to lag window).
    n_lags : int
        Number of lags.
    method : str
        "dcor" (unbiased), "dcor_biased", "pearson", or "lasso".

    Returns
    -------
    ndarray, shape (n_lags,)
    """
    if method == "lasso":
        return _lasso_pair(sources, target_col, n_lags)
    elif method == "dcor":
        func = _dcor_pair_unbiased
    elif method == "dcor_biased":
        func = _dcor_pair_biased
    elif method == "pearson":
        func = _pearson_pair
    else:
        raise ValueError(
            f"Unknown lag importance method: {method!r}. "
            "Choose 'dcor', 'dcor_biased', 'pearson', 'lasso', or 'partial_dcor'."
        )

    scores = np.zeros(n_lags)
    for lag in range(n_lags):
        src = sources[lag]
        valid = ~(np.isnan(src) | np.isnan(target_col))
        if valid.sum() < 10:
            continue
        scores[lag] = func(src[valid], target_col[valid])
    return scores


def _residualize_targets(data: np.ndarray, max_lag: int) -> np.ndarray:
    """Residualize each variable X_j(t) on X_j(t-1) via OLS.

    Returns array of shape (T, d) with residuals for t = max_lag, ..., T-1.
    Used by compute_lag_importance with method="partial_dcor".
    """
    T, d = data.shape
    eff_T = T - max_lag
    resids = np.full((T, d), np.nan)
    target_t = data[max_lag:, :]
    target_tm1 = data[max_lag - 1 : T - 1, :]
    for j in range(d):
        y = target_t[:, j]
        yp = target_tm1[:, j]
        valid = ~np.isnan(y) & ~np.isnan(yp)
        if valid.sum() < 10:
            continue
        A = np.column_stack([np.ones(valid.sum()), yp[valid]])
        coef, *_ = np.linalg.lstsq(A, y[valid], rcond=None)
        r = np.full(eff_T, np.nan)
        r[valid] = y[valid] - A @ coef
        resids[max_lag:, j] = r
    return resids


def _compute_one_pair_partial_dcor(
    source_lags: np.ndarray,
    resid_target: np.ndarray,
    n_lags: int,
) -> np.ndarray:
    """Partial dcor lag importance: u_dcor(X_i(t-k), X_j(t) - β·X_j(t-1)).

    Removes X_j's own AR(1) contribution by OLS residualization, then
    measures nonlinear dependence via unbiased distance correlation.
    Unlike Granger, does NOT condition on shorter lags of X_i, avoiding
    over-suppression of true lag-k signals when the source has strong AR.
    """
    import dcor as _dcor

    scores = np.zeros(n_lags)
    for lag_idx in range(n_lags):
        x_test = source_lags[lag_idx]
        valid = ~np.isnan(x_test) & ~np.isnan(resid_target)
        if valid.sum() < 10:
            continue
        score = float(
            _dcor.u_distance_correlation_sqr(x_test[valid], resid_target[valid])
        )
        scores[lag_idx] = score if np.isfinite(score) else 0.0
    return scores


def compute_lag_importance(
    data: np.ndarray,
    max_lag: int,
    method: str = "dcor",
    include_lag0: bool = False,
    n_jobs: int = 1,
    target_col: int | None = None,
) -> np.ndarray:
    """Compute lag importance matrix for all variable pairs.

    Uses the unbiased U-centered distance correlation by default,
    which is properly centered at 0 under independence regardless of
    sample size (no positive bias at small n).

    Parameters
    ----------
    data : ndarray, shape (T, d)
        Normalized time series data.
    max_lag : int
        Maximum lag to consider.
    method : str
        "dcor" (default, unbiased U-centered), "dcor_biased", "pearson", or
        "lasso" (LassoCV per pair, matching SyPI Mastakouri et al. 2021).
    include_lag0 : bool
        Whether to include contemporaneous effects (lag 0).
    n_jobs : int
        Number of parallel workers (requires joblib).
    target_col : int or None
        If set, compute only the 2d-1 pairs that involve this column as
        cause or effect. All other entries remain 0. Use with ``target_var``
        mode in :class:`Cedar` to cut Phase-0 cost from O(d²) to O(d).

    Returns
    -------
    ndarray, shape (d, d, max_lag+1)
        ``result[cause, effect, lag]`` = importance score.
        Lag 0 slot is zero unless ``include_lag0=True``.
    """
    T, d = data.shape
    start_lag = 0 if include_lag0 else 1
    n_lags = max_lag + 1 - start_lag

    effective_T = T - max_lag
    if effective_T < 10:
        raise ValueError(
            f"Not enough samples after lagging: T={T}, max_lag={max_lag}, "
            f"effective_T={effective_T}. Need at least 10."
        )

    targets = data[max_lag:, :]  # (effective_T, d)
    sources = np.stack(
        [
            data[max_lag - (start_lag + i) : T - (start_lag + i), :]
            for i in range(n_lags)
        ],
        axis=0,
    )  # (n_lags, effective_T, d)

    result = np.zeros((d, d, max_lag + 1))

    if target_col is not None:
        # Only compute pairs where target_col is the effect (all c → target)
        # or where target_col is the cause (target → all e). The child-filter
        # in Cedar needs both directions to determine asymmetry.
        pairs = [(c, target_col) for c in range(d)] + [
            (target_col, e) for e in range(d) if e != target_col
        ]
    else:
        pairs = [(c, e) for c in range(d) for e in range(d)]

    if method == "partial_dcor":
        resid_matrix = _residualize_targets(data, max_lag)
        resid_eff = resid_matrix[max_lag:]  # (eff_T, d) aligned with targets
        for c_idx, e_idx in pairs:
            result[c_idx, e_idx, start_lag:] = _compute_one_pair_partial_dcor(
                sources[:, :, c_idx],
                resid_eff[:, e_idx],
                n_lags,
            )
    elif n_jobs != 1 and _HAS_JOBLIB:
        scores = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_compute_one_pair)(
                sources[:, :, c_idx], targets[:, e_idx], n_lags, method
            )
            for c_idx, e_idx in pairs
        )
        for idx, (c_idx, e_idx) in enumerate(pairs):
            result[c_idx, e_idx, start_lag:] = scores[idx]
    else:
        for c_idx, e_idx in pairs:
            result[c_idx, e_idx, start_lag:] = _compute_one_pair(
                sources[:, :, c_idx], targets[:, e_idx], n_lags, method
            )

    # Diagonal at lag 0 is trivially 1.0 for any correlation measure.
    np.fill_diagonal(result[:, :, 0], 1.0)

    return result


# ---------------------------------------------------------------------------
# Significance testing
# ---------------------------------------------------------------------------


def compute_lag_pvalues(
    data: np.ndarray,
    lag_importance: np.ndarray,
    max_lag: int,
    method: str = "t_test",
    n_permutations: int = 200,
    seed: int = 42,
    include_lag0: bool = False,
    n_jobs: int = 1,
    target_col: int | None = None,
) -> np.ndarray:
    """Compute p-values for lag importance scores.

    Parameters
    ----------
    data : ndarray, shape (T, d)
        Normalized data.
    lag_importance : ndarray, shape (d, d, max_lag+1)
        Observed importance scores.
    max_lag : int
        Maximum lag.
    method : str
        "t_test" — dcor t-test (Székely & Rizzo 2013), asymptotic,
            no permutation, ~200x faster. Uses Student-t null distribution
            based on unbiased U-centered estimator. Well-calibrated for
            n >= 20. Slightly inflated FPR (~15%) for autocorrelated data.
        "circular_shift" — circular shift permutation. Preserves
            autocorrelation within the target by shifting rather than
            shuffling. Properly calibrated FPR (~5%) for AR data.
        "permutation" — standard random permutation (legacy).
    n_permutations : int
        Number of permutations (for "circular_shift" and "permutation").
    seed : int
        Random seed for reproducibility.
    include_lag0 : bool
        Whether lag 0 was included.
    n_jobs : int
        Parallelism (future use).
    target_col : int or None
        If set, only compute p-values for the 2d-1 pairs that involve this
        column. Must match the ``target_col`` passed to
        :func:`compute_lag_importance`.

    Returns
    -------
    ndarray, shape (d, d, max_lag+1)
        P-values. Entries where lag_importance <= 0 get p-value = 1.0.
    """
    if method == "t_test":
        return _pvalues_t_test(data, lag_importance, max_lag, include_lag0, target_col)
    elif method == "circular_shift":
        return _pvalues_circular_shift(
            data,
            lag_importance,
            max_lag,
            n_permutations,
            seed,
            include_lag0,
            target_col,
        )
    else:
        return _pvalues_random_permutation(
            data,
            lag_importance,
            max_lag,
            n_permutations,
            seed,
            include_lag0,
            target_col,
        )


def _pvalues_t_test(
    data: np.ndarray,
    lag_importance: np.ndarray,
    max_lag: int,
    include_lag0: bool,
    target_col: int | None = None,
) -> np.ndarray:
    """P-values via dcor t-test (Székely & Rizzo 2013).

    T_n = sqrt(v-1) * R_U^2 / sqrt(1 - R_U^4)
    where v = n(n-3)/2, df = v-1.
    Under H0: T_n ~ Student-t(df).
    """
    T, d = data.shape
    start_lag = 0 if include_lag0 else 1
    n_lags = max_lag + 1 - start_lag
    pvalues = np.ones_like(lag_importance)

    effective_T = T - max_lag

    if target_col is not None:
        pairs = [(c, target_col) for c in range(d) if c != target_col] + [
            (target_col, e) for e in range(d) if e != target_col
        ]
    else:
        pairs = [(c, e) for c in range(d) for e in range(d) if c != e]

    for c_idx, e_idx in pairs:
        for lag_idx in range(n_lags):
            lag = start_lag + lag_idx
            r_u_sq = lag_importance[c_idx, e_idx, lag]
            if r_u_sq <= 0:
                pvalues[c_idx, e_idx, lag] = 1.0
                continue

            n = effective_T
            v = n * (n - 3) / 2
            if v <= 1:
                pvalues[c_idx, e_idx, lag] = 1.0
                continue

            denom = 1 - r_u_sq**2
            if denom <= 0:
                t_stat = np.inf
            else:
                t_stat = np.sqrt(v - 1) * r_u_sq / np.sqrt(denom)

            pvalues[c_idx, e_idx, lag] = 1 - t_dist.cdf(t_stat, df=v - 1)

    return pvalues


def _pvalues_circular_shift(
    data: np.ndarray,
    lag_importance: np.ndarray,
    max_lag: int,
    n_permutations: int,
    seed: int,
    include_lag0: bool,
    target_col: int | None = None,
) -> np.ndarray:
    """P-values via circular shift permutation.

    Shifts the target by a random offset (1..n-1), preserving its
    autocorrelation structure while breaking the cross-dependence with source.
    """
    import dcor

    T, d = data.shape
    start_lag = 0 if include_lag0 else 1
    n_lags = max_lag + 1 - start_lag
    effective_T = T - max_lag

    rng = np.random.default_rng(seed)
    pvalues = np.ones_like(lag_importance)

    targets = data[max_lag:, :]
    sources = np.stack(
        [
            data[max_lag - (start_lag + i) : T - (start_lag + i), :]
            for i in range(n_lags)
        ],
        axis=0,
    )

    if target_col is not None:
        pairs = [(c, target_col) for c in range(d) if c != target_col] + [
            (target_col, e) for e in range(d) if e != target_col
        ]
    else:
        pairs = [(c, e) for c in range(d) for e in range(d) if c != e]

    for c_idx, e_idx in pairs:
        for lag_idx in range(n_lags):
            lag = start_lag + lag_idx
            observed = lag_importance[c_idx, e_idx, lag]
            if observed <= 0:
                continue

            source_col = sources[lag_idx, :, c_idx]
            target_col_data = targets[:, e_idx]

            # Biased dcor for comparison (observed was computed as unbiased)
            observed_biased = dcor.distance_correlation(
                source_col, target_col_data, method="avl"
            )

            count_ge = 0
            for _ in range(n_permutations):
                shift = rng.integers(1, effective_T)
                target_shifted = np.roll(target_col_data, shift)
                perm_score = dcor.distance_correlation(
                    source_col, target_shifted, method="avl"
                )
                if perm_score >= observed_biased:
                    count_ge += 1

            pvalues[c_idx, e_idx, lag] = (count_ge + 1) / (n_permutations + 1)

    return pvalues


def _pvalues_random_permutation(
    data: np.ndarray,
    lag_importance: np.ndarray,
    max_lag: int,
    n_permutations: int,
    seed: int,
    include_lag0: bool,
    target_col: int | None = None,
) -> np.ndarray:
    """P-values via standard random permutation (legacy method)."""
    import dcor

    T, d = data.shape
    start_lag = 0 if include_lag0 else 1
    n_lags = max_lag + 1 - start_lag
    effective_T = T - max_lag

    rng = np.random.default_rng(seed)
    pvalues = np.ones_like(lag_importance)

    targets = data[max_lag:, :]
    sources = np.stack(
        [
            data[max_lag - (start_lag + i) : T - (start_lag + i), :]
            for i in range(n_lags)
        ],
        axis=0,
    )

    perm_indices = np.stack(
        [rng.permutation(effective_T) for _ in range(n_permutations)]
    )

    if target_col is not None:
        pairs = [(c, target_col) for c in range(d) if c != target_col] + [
            (target_col, e) for e in range(d) if e != target_col
        ]
    else:
        pairs = [(c, e) for c in range(d) for e in range(d) if c != e]

    for c_idx, e_idx in pairs:
        for lag_idx in range(n_lags):
            lag = start_lag + lag_idx
            observed = lag_importance[c_idx, e_idx, lag]
            if observed <= 0:
                continue

            source_col = sources[lag_idx, :, c_idx]
            observed_biased = dcor.distance_correlation(
                source_col, targets[:, e_idx], method="avl"
            )

            count_ge = 0
            for perm_i in range(n_permutations):
                target_perm = targets[perm_indices[perm_i], e_idx]
                perm_score = dcor.distance_correlation(
                    source_col, target_perm, method="avl"
                )
                if perm_score >= observed_biased:
                    count_ge += 1

            pvalues[c_idx, e_idx, lag] = (count_ge + 1) / (n_permutations + 1)

    return pvalues


# ---------------------------------------------------------------------------
# Lag selection with FDR correction
# ---------------------------------------------------------------------------


def _bh_fdr(pvalues_flat: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Returns adjusted p-values (monotonized).
    """
    n = len(pvalues_flat)
    if n == 0:
        return pvalues_flat.copy()

    sorted_idx = np.argsort(pvalues_flat)
    sorted_pvals = pvalues_flat[sorted_idx]

    # BH adjustment: p_adj[i] = p[i] * n / rank[i]
    ranks = np.arange(1, n + 1)
    adjusted = sorted_pvals * n / ranks

    # Enforce monotonicity (cumulative minimum from the right)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)

    # Unsort
    result = np.empty_like(adjusted)
    result[sorted_idx] = adjusted
    return result


def select_lags(
    lag_importance: np.ndarray,
    lag_pvalues: np.ndarray,
    alpha: float = 0.05,
    strategy: str = "max_importance",
    include_lag0: bool = False,
    fdr_correction: bool = True,
) -> np.ndarray:
    """Select one lag per (cause, effect) pair.

    Parameters
    ----------
    lag_importance : ndarray, shape (d, d, max_lag+1)
        Importance scores (unbiased dcor²).
    lag_pvalues : ndarray, shape (d, d, max_lag+1)
        P-values from significance testing.
    alpha : float
        Significance threshold (applied after FDR correction if enabled).
    strategy : str
        "min_significant" — smallest significant lag (paper's approach).
        "max_importance" — among significant lags, pick highest importance.
    include_lag0 : bool
        Whether lag 0 is valid.
    fdr_correction : bool
        Apply Benjamini-Hochberg FDR correction across all hypotheses.

    Returns
    -------
    ndarray, shape (d, d)
        Selected lag for each pair. NaN where no significant lag exists.
    """
    d = lag_importance.shape[0]
    start_lag = 0 if include_lag0 else 1
    selected = np.full((d, d), np.nan)

    # Apply BH-FDR correction across all (cause, effect, lag) hypotheses
    if fdr_correction:
        mask = np.zeros_like(lag_pvalues, dtype=bool)
        mask[:, :, start_lag:] = True
        np.fill_diagonal(mask[:, :, 0], False)
        for lag in range(start_lag, lag_pvalues.shape[2]):
            np.fill_diagonal(mask[:, :, lag], False)

        flat_pvals = lag_pvalues[mask]
        adjusted = _bh_fdr(flat_pvals, alpha)
        adj_pvalues = lag_pvalues.copy()
        adj_pvalues[mask] = adjusted
    else:
        adj_pvalues = lag_pvalues

    for c_idx in range(d):
        for e_idx in range(d):
            if c_idx == e_idx:
                continue
            sig_mask = adj_pvalues[c_idx, e_idx, start_lag:] < alpha
            if not sig_mask.any():
                continue

            sig_lags = np.where(sig_mask)[0] + start_lag

            if strategy == "min_significant":
                selected[c_idx, e_idx] = sig_lags[0]
            elif strategy == "max_importance":
                importances = lag_importance[c_idx, e_idx, sig_lags]
                selected[c_idx, e_idx] = sig_lags[np.argmax(importances)]
            else:
                selected[c_idx, e_idx] = sig_lags[0]

    return selected


def select_all_significant_lags(
    lag_importance: np.ndarray,
    lag_pvalues: np.ndarray,
    alpha: float = 0.05,
    include_lag0: bool = False,
    fdr_correction: bool = True,
) -> np.ndarray:
    """Return boolean mask of all significant lags per (cause, effect) pair.

    Parameters
    ----------
    lag_importance : ndarray, shape (d, d, max_lag+1)
        Importance scores.
    lag_pvalues : ndarray, shape (d, d, max_lag+1)
        P-values from significance testing.
    alpha : float
        Significance threshold.
    include_lag0 : bool
        Whether lag 0 is valid.
    fdr_correction : bool
        Apply BH-FDR correction.

    Returns
    -------
    ndarray, shape (d, d, max_lag+1), dtype=bool
        True where the (cause, effect, lag) triple is significant.
    """
    d = lag_importance.shape[0]
    start_lag = 0 if include_lag0 else 1

    if fdr_correction:
        mask = np.zeros_like(lag_pvalues, dtype=bool)
        mask[:, :, start_lag:] = True
        for lag in range(start_lag, lag_pvalues.shape[2]):
            np.fill_diagonal(mask[:, :, lag], False)

        flat_pvals = lag_pvalues[mask]
        adjusted = _bh_fdr(flat_pvals, alpha)
        adj_pvalues = lag_pvalues.copy()
        adj_pvalues[mask] = adjusted
    else:
        adj_pvalues = lag_pvalues

    significant = np.zeros_like(lag_pvalues, dtype=bool)
    for c_idx in range(d):
        for e_idx in range(d):
            if c_idx == e_idx:
                continue
            significant[c_idx, e_idx, start_lag:] = (
                adj_pvalues[c_idx, e_idx, start_lag:] < alpha
            )

    return significant
