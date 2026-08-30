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


def _lag_embed(df, max_lag):
    """Stack the present block beside one block per lag: [X_t, X_{t-1}, ...]."""
    data = df.values if isinstance(df, pd.DataFrame) else np.asarray(df)
    T_orig, d = data.shape
    blocks = [data[max_lag - lag : T_orig - lag, :] for lag in range(max_lag + 1)]
    return np.hstack(blocks), d


def _ges_fast(embedded, d, max_lag, lambda_value, forbidden):
    """GES via the vendored search in :mod:`causalts.lges`.

    Same algorithm as causal-learn's ``ges`` -- forward Insert phase then backward
    Delete phase, no turning -- plus temporal background knowledge forbidding
    edges from the present into the past (see ``temporal_forbidden``). Without
    it, the search treats every lag-embedded column as an ordinary variable and
    happily inserts and scores backward-in-time edges -- e.g. present X2 into
    past-lag-1 X5 -- which are never reported (extraction only reads out
    lag-to-present cells) but still consume search budget and can change which
    forward edges win. Verified to return the same graph as causal-learn's
    ``ges`` when there is no lag structure to protect (``max_lag == 0``, where
    ``forbidden`` is all zeros). It is also much faster on large samples
    because ``GaussObsL0Pen`` caches the scatter matrix, so each local score is
    O(1) in the sample size instead of a fresh pass over the data.
    """
    from .lges import GaussObsL0Pen, fit

    n = embedded.shape[0]
    # causal-learn penalises lambda_value * (|pa| + 1) * log(n); GaussObsL0Pen
    # penalises lmbda * (|pa| + 1), so fold the log(n) in to match.
    lmbda = None if lambda_value is None else lambda_value * np.log(n)
    A, metrics = fit(
        GaussObsL0Pen(embedded, lmbda=lmbda),
        phases=["forward", "backward"],
        score_based=False,
        prune=False,
        forbidden=forbidden,
    )

    G_hat = np.zeros((d, d, max_lag + 1), dtype=int)
    for lag in range(max_lag + 1):
        for i in range(d):
            for j in range(d):
                if lag == 0 and i == j:
                    continue
                # ges-package convention: A[u, v] != 0 means u -> v (both
                # directions set means the edge is undirected).
                if A[lag * d + i, j] != 0:
                    G_hat[i, j, lag] = 1
    return G_hat, {"method": "ges", "engine": "fast", "cpdag": A, "metrics": metrics}


def ges_discovery(
    df, max_lag=1, score_func="local_score_BIC", lambda_value=None, engine="fast"
):
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
        Scoring function for GES (e.g. ``"local_score_BIC"``). Only
        ``"local_score_BIC"`` is available under ``engine="fast"``; any other
        value transparently selects the causal-learn engine.
    lambda_value : float or None
        BIC penalty hyperparameter. Larger values produce sparser graphs.
    engine : {"fast", "causal-learn"}
        Which implementation of GES to run. ``"fast"`` (the default) uses the
        vendored search in :mod:`causalts.lges`, which forbids edges from the
        present into the past during the search itself -- background knowledge
        that a lagged time series always licenses -- and is dramatically
        quicker on large samples, since ``GaussObsL0Pen`` caches the scatter
        matrix (a 24-column embedding at T=20,000 takes 1.2s against
        causal-learn's 131s). ``"causal-learn"`` runs the original
        implementation, which has no parameter for background knowledge; it is
        offered for parity/validation and matches the fast engine exactly at
        ``max_lag=0``, where there is no temporal ordering to protect. At
        ``max_lag > 0`` it raises, rather than silently return a graph from an
        unconstrained search that can orient edges backward in time.

    Returns
    -------
    G_hat : ndarray, shape ``(d, d, max_lag+1)``
        Estimated graph.
    info : dict
        ``engine`` plus, for the causal-learn engine, ``score`` and ``ges_graph``
        (raw GES output); for the fast engine, ``cpdag`` and ``metrics``.

    """
    if engine not in ("fast", "causal-learn"):
        raise ValueError(f"engine must be 'fast' or 'causal-learn', got {engine!r}")

    embedded, d = _lag_embed(df, max_lag)

    from .lges import temporal_forbidden

    forbidden = temporal_forbidden(d, max_lag)

    if engine == "fast" and score_func == "local_score_BIC":
        return _ges_fast(embedded, d, max_lag, lambda_value, forbidden)

    if max_lag > 0:
        raise ValueError(
            "the causal-learn engine cannot honor temporal background "
            "knowledge: causal-learn's ges() has no forbidden-edges "
            "parameter, so running it on lagged data (max_lag > 0) would "
            "search unconstrained and could orient edges backward in time. "
            f"This was reached because {'engine=' + repr(engine) if engine == 'causal-learn' else 'score_func=' + repr(score_func) + ' is not local_score_BIC'} "  # noqa: E501
            "-- use engine='fast' with score_func='local_score_BIC' (the "
            "default) instead, or call with max_lag=0."
        )

    from causallearn.search.ScoreBased.GES import ges

    _patch_ges_numpy2()

    result = ges(embedded, score_func=score_func, lambda_value=lambda_value)
    ges_graph = result["G"].graph

    G_hat = np.zeros((d, d, max_lag + 1), dtype=int)

    # causal-learn adjacency convention (verified against GeneralGraph.add_edge):
    #   u -> v   is  graph[u, v] == -1 (tail at u)  and  graph[v, u] == 1 (arrow at v)
    #   u -- v   is  graph[u, v] == graph[v, u] == -1
    # Undirected edges are recorded in the temporal direction, which is the only
    # one consistent with the lag embedding.
    for lag in range(max_lag + 1):
        for i in range(d):
            for j in range(d):
                if lag == 0 and i == j:
                    continue
                cause = lag * d + i  # variable i at time t - lag
                effect = j  # variable j at time t
                tail, head = ges_graph[cause, effect], ges_graph[effect, cause]
                directed = tail == -1 and head == 1
                undirected = tail == -1 and head == -1
                if directed or undirected:
                    G_hat[i, j, lag] = 1

    info = {
        "method": "ges",
        "engine": "causal-learn",
        "score": result.get("score"),
        "score_func": score_func,
        "ges_graph": ges_graph,
    }

    return G_hat, info


# Re-export LGES for convenience
from .lges import lges_discovery, tges_discovery  # noqa: E402, F401
