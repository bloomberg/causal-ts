# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Baseline discovery methods wrapping causal-learn algorithms.

These are thin wrappers around causal-learn's VARLiNGAM, Granger, and GES
that produce the standard causal-ts graph format ``(d, d, max_lag+1)``.
"""

import numpy as np
import pandas as pd


def varlingam_discovery(df, max_lag=1, threshold=0.05):
    """Run VARLiNGAM (Vector Autoregressive LiNGAM) discovery.

    Assumes linear, non-Gaussian noise. Returns instantaneous and lagged
    causal effects via ICA decomposition.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data, shape ``(T, d)``.
    max_lag : int
        Number of lags to include in the VAR model.
    threshold : float
        Minimum absolute coefficient to count as an edge.

    Returns
    -------
    G_hat : ndarray, shape ``(d, d, max_lag+1)``
        Estimated graph in ``[cause, effect, lag]`` format.
    coefficients : ndarray, shape ``(d, d, max_lag+1)``
        Raw coefficient matrices (instantaneous at lag 0, lagged at lag 1+).
    info : dict
        ``causal_order``, ``p_values`` (from error independence test).
    """
    from causallearn.search.FCMBased.lingam.var_lingam import VARLiNGAM

    data = df.values if isinstance(df, pd.DataFrame) else np.asarray(df)
    d = data.shape[1]

    model = VARLiNGAM(lags=max_lag)
    model.fit(data)

    matrices = model.adjacency_matrices_
    n_matrices = len(matrices)

    coefficients = np.zeros((d, d, max_lag + 1))
    G_hat = np.zeros((d, d, max_lag + 1), dtype=int)

    for lag_idx in range(min(n_matrices, max_lag + 1)):
        mat = matrices[lag_idx]
        coefficients[:, :, lag_idx] = mat.T
        G_hat[:, :, lag_idx] = (np.abs(mat.T) > threshold).astype(int)

    info = {
        "method": "varlingam",
        "causal_order": list(model.causal_order_),
        "p_values": model.get_error_independence_p_values(),
        "threshold": threshold,
    }

    return G_hat, coefficients, info


def granger_discovery(df, max_lag=1, threshold=None):
    """Run Granger causality via LASSO-VAR.

    Linear method that tests whether past values of one variable improve
    prediction of another beyond its own past.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data, shape ``(T, d)``.
    max_lag : int
        Maximum lag order for the VAR model.
    threshold : float or None
        Minimum absolute coefficient to count as an edge. If *None*,
        any non-zero LASSO coefficient is treated as an edge.

    Returns
    -------
    G_hat : ndarray, shape ``(d, d, max_lag+1)``
        Estimated graph. Lag 0 is always empty (Granger is purely lagged).
    coefficients : ndarray, shape ``(d, d, max_lag+1)``
        Raw LASSO coefficient matrices per lag.
    info : dict
        Metadata including method name.
    """
    from causallearn.search.Granger.Granger import Granger

    data = df.values if isinstance(df, pd.DataFrame) else np.asarray(df)
    d = data.shape[1]

    model = Granger(maxlag=max_lag)
    coef_matrix = model.granger_lasso(data)

    coefficients = np.zeros((d, d, max_lag + 1))
    G_hat = np.zeros((d, d, max_lag + 1), dtype=int)

    for lag in range(1, max_lag + 1):
        start_col = (lag - 1) * d
        end_col = lag * d
        if end_col <= coef_matrix.shape[1]:
            lag_coefs = coef_matrix[:, start_col:end_col]
            coefficients[:, :, lag] = lag_coefs.T
            if threshold is None:
                G_hat[:, :, lag] = (np.abs(lag_coefs.T) > 0).astype(int)
            else:
                G_hat[:, :, lag] = (np.abs(lag_coefs.T) > threshold).astype(int)

    info = {"method": "granger", "threshold": threshold}

    return G_hat, coefficients, info


def lasso_var_discovery(df, max_lag=1, cv_folds=5):
    """Run LASSO-VAR (LASSO VAR) for lagged causal discovery.

    Fits an L1-penalized vector autoregression per target variable using
    LassoCV. Only discovers lagged effects (no lag-0 / instantaneous edges).
    The LASSO penalty is selected automatically via cross-validation, so
    no manual thresholding is needed — any non-zero coefficient is an edge.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data, shape ``(T, d)``.
    max_lag : int
        Number of lags in the VAR model.
    cv_folds : int
        Number of cross-validation folds for LassoCV.

    Returns
    -------
    G_hat : ndarray, shape ``(d, d, max_lag+1)``
        Estimated graph. Lag 0 is always empty.
    coefficients : ndarray, shape ``(d, d, max_lag+1)``
        Raw LASSO coefficient matrices per lag.
    info : dict
        ``alphas`` (per-target CV-selected penalties), ``mean_alpha``.
    """
    from sklearn.linear_model import LassoCV

    data = df.values if isinstance(df, pd.DataFrame) else np.asarray(df)
    T, d = data.shape

    X_lag = np.column_stack(
        [data[max_lag - lag : T - lag, :] for lag in range(1, max_lag + 1)]
    )
    Y = data[max_lag:, :]

    G_hat = np.zeros((d, d, max_lag + 1), dtype=int)
    coefficients = np.zeros((d, d, max_lag + 1))
    alphas_used = []

    for e in range(d):
        lasso = LassoCV(cv=cv_folds, max_iter=10000, n_jobs=-1)
        lasso.fit(X_lag, Y[:, e])
        alphas_used.append(float(lasso.alpha_))

        coefs = lasso.coef_.reshape(max_lag, d)
        for lag_idx in range(max_lag):
            for c in range(d):
                coefficients[c, e, lag_idx + 1] = coefs[lag_idx, c]
                if abs(coefs[lag_idx, c]) > 0:
                    G_hat[c, e, lag_idx + 1] = 1

    info = {
        "method": "lasso_var",
        "alphas": alphas_used,
        "mean_alpha": float(np.mean(alphas_used)),
    }

    return G_hat, coefficients, info


_ges_patched = False


def _patch_ges_numpy2():
    """Fix causal-learn GES for numpy >= 2.0 (float() on 1x1 matrix)."""
    global _ges_patched
    if _ges_patched:
        return
    _ges_patched = True

    import causallearn.score.LocalScoreFunction as _mod

    _original = _mod.local_score_BIC_from_cov

    def _patched(Data, i, PAi, parameters=None):
        cov, n = Data
        if parameters is None:
            lambda_value = 0.5
        else:
            lambda_value = parameters["lambda_value"]
        sigma = cov[i, i]
        if len(PAi) > 0:
            yX = cov[np.ix_([i], PAi)]
            XX = cov[np.ix_(PAi, PAi)]
            try:
                XX_inv = np.linalg.inv(XX)
            except np.linalg.LinAlgError:
                XX_inv = np.linalg.pinv(XX)
            sigma = (cov[i, i] - yX @ XX_inv @ yX.T).item()
        if sigma <= 0:
            sigma = np.finfo(float).eps
        likelihood = -0.5 * n * (1 + np.log(sigma))
        penalty = lambda_value * (len(PAi) + 1) * np.log(n)
        return likelihood - penalty

    _patched.__name__ = _original.__name__
    _patched.__qualname__ = _original.__qualname__
    _mod.local_score_BIC_from_cov = _patched
    # Also patch the reference in GES module (imported at module level)
    import causallearn.search.ScoreBased.GES as _ges_mod

    _ges_mod.local_score_BIC_from_cov = _patched


def ges_discovery(df, max_lag=1, score_func="local_score_BIC", lambda_value=None):
    """Run GES (Greedy Equivalence Search) on lag-embedded data.

    Score-based method that searches over equivalence classes of DAGs.
    The time series is first embedded with lagged copies, then GES is
    applied to the augmented variable set.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data, shape ``(T, d)``.
    max_lag : int
        Number of lags to embed.
    score_func : str
        Scoring function for GES (e.g. ``"local_score_BIC"``).
    lambda_value : float or None
        BIC penalty hyperparameter. Larger values produce sparser graphs.

    Returns
    -------
    G_hat : ndarray, shape ``(d, d, max_lag+1)``
        Estimated graph.
    info : dict
        ``score``, ``ges_graph`` (raw GES output).

    """
    from causallearn.search.ScoreBased.GES import ges

    _patch_ges_numpy2()

    data = df.values if isinstance(df, pd.DataFrame) else np.asarray(df)
    T_orig, d = data.shape

    embedded_cols = []
    for lag in range(max_lag + 1):
        start = max_lag - lag
        end = T_orig - lag
        embedded_cols.append(data[start:end, :])

    embedded = np.hstack(embedded_cols)
    n_vars = embedded.shape[1]  # noqa: F841

    result = ges(embedded, score_func=score_func, lambda_value=lambda_value)
    ges_graph = result["G"].graph

    G_hat = np.zeros((d, d, max_lag + 1), dtype=int)

    for lag in range(max_lag + 1):
        cause_start = lag * d
        cause_end = (lag + 1) * d
        effect_start = 0
        effect_end = d

        block = ges_graph[effect_start:effect_end, cause_start:cause_end]
        for i in range(d):
            for j in range(d):
                if lag == 0 and i == j:
                    continue
                if (
                    block[j, i] == -1
                    and ges_graph[cause_start + i, effect_start + j] == 1
                ):
                    G_hat[i, j, lag] = 1
                elif (
                    block[j, i] == -1
                    and ges_graph[cause_start + i, effect_start + j] == -1
                ):
                    G_hat[i, j, lag] = 1

    info = {
        "method": "ges",
        "score": result.get("score"),
        "score_func": score_func,
        "ges_graph": ges_graph,
    }

    return G_hat, info


# Re-export LGES for convenience
from .lges import lges_discovery, tges_discovery  # noqa: E402, F401
