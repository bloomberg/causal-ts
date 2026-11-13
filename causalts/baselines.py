# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Baseline discovery methods wrapping causal-learn algorithms.

These are thin wrappers around causal-learn's VARLiNGAM, Granger, and GES
that produce the standard causal-ts graph format ``(d, d, max_lag+1)``.
"""

import numpy as np
import pandas as pd

from .result import CausalResult


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

# ---------------------------------------------------------------------------
# Result objects and named runners (CLI ``discover --algorithm`` / discover_df)
# ---------------------------------------------------------------------------

#: Name -> (function path, whether the method can return lag-0 edges).
_BASELINE_METHODS = {
    "lges": ("causalts.lges.lges_discovery", True),
    "tges": ("causalts.lges.tges_discovery", True),
    "ges": ("causalts.baselines.ges_discovery", True),
    "lasso-var": ("causalts.baselines.lasso_var_discovery", False),
    "granger": ("causalts.baselines.granger_discovery", False),
    "varlingam": ("causalts.baselines.varlingam_discovery", True),
}

#: Names of the score- and regression-based methods :func:`run_baseline` runs.
BASELINE_ALGORITHMS = tuple(_BASELINE_METHODS)


class BaselineResult(CausalResult):
    """Result of a score- or regression-based method run by :func:`run_baseline`.

    Inherits plotting and the DoWhy bridge from
    :class:`~causalts.result.CausalResult`.

    Attributes
    ----------
    cg_tig : np.ndarray, shape (d, d, max_lag+1)
        Binary graph, ``cg_tig[cause, effect, lag] == 1`` meaning
        ``cause(t-lag) -> effect(t)``. LGES, TGES and GES record an unoriented
        CPDAG edge at lag 0 in both directions (:meth:`to_marks` shows it as
        ``o-o``); LASSO-VAR and Granger never return lag-0 edges.
    var_names : list[str]
        Variable names.
    method : str
        The algorithm name (``"lges"``, ``"tges"``, ``"ges"``, ``"lasso-var"``,
        ``"granger"`` or ``"varlingam"``).
    coefficients : np.ndarray or None
        Coefficients, same shape as ``cg_tig`` (LASSO-VAR, Granger, VARLiNGAM).
    cpdag : np.ndarray or None
        CPDAG over the lag-embedded variables (LGES, TGES, GES).
    info : dict
        The method's info dict (score, CV penalties, causal order, ...).
    contemporaneous : bool
        Whether the method can return lag-0 edges.
    """

    def __init__(self, cg_tig, df, method, coefficients=None, info=None):
        self.cg_tig = np.asarray(cg_tig)
        self.var_names = list(df.columns)
        self._df = df
        self.method = method
        self.coefficients = coefficients
        self.info = dict(info or {})
        # TGES orients by tiers after the search; report the graph cg_tig uses.
        self.cpdag = self.info.get("tiered_cpdag", self.info.get("cpdag"))
        self.contemporaneous = _BASELINE_METHODS[method][1]
        self._scm_cache = {}


def run_baseline(algorithm, df, max_lag=1, **kwargs):
    """Run a score- or regression-based method by name and wrap its output.

    Parameters
    ----------
    algorithm : str
        One of :data:`BASELINE_ALGORITHMS`: ``"lges"``
        (:func:`~causalts.lges.lges_discovery`), ``"tges"``
        (:func:`~causalts.lges.tges_discovery`), ``"ges"``
        (:func:`ges_discovery`), ``"lasso-var"`` (:func:`lasso_var_discovery`),
        ``"granger"`` (:func:`granger_discovery`) or ``"varlingam"``
        (:func:`varlingam_discovery`).
    df : pd.DataFrame
        Time series data, shape ``(T, d)``.
    max_lag : int
        Maximum lag. LASSO-VAR and Granger are lagged-only, and VARLiNGAM fits
        a VAR, so all three need ``max_lag >= 1``.
    **kwargs
        Passed to the method: ``mode`` and ``lambda_value`` (LGES, TGES; the
        absolute BIC penalty per parameter), ``lambda_value`` (GES; a multiplier
        on ``log(n)``, as in causal-learn), ``score_func`` and ``engine`` (GES), ``cv_folds``
        (LASSO-VAR), ``threshold`` (Granger, VARLiNGAM).

    Returns
    -------
    BaselineResult

    Raises
    ------
    ValueError
        If ``df`` has NaN or infinite values (impute first, e.g. with
        :func:`~causalts.imputation.var_em_impute`), or ``max_lag < 1`` for
        LASSO-VAR, Granger or VARLiNGAM.
    """
    import importlib

    if algorithm not in _BASELINE_METHODS:
        raise ValueError(
            f"unknown algorithm {algorithm!r}; expected one of {BASELINE_ALGORITHMS}"
        )
    path, contemporaneous = _BASELINE_METHODS[algorithm]
    if not contemporaneous and max_lag < 1:
        raise ValueError(
            f"{algorithm} finds lagged edges only and needs max_lag >= 1, "
            f"got {max_lag}"
        )
    if algorithm == "varlingam" and max_lag < 1:
        raise ValueError(
            f"varlingam fits a VAR and needs at least one lag (max_lag >= 1), "
            f"got {max_lag}"
        )
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(np.asarray(df))
    values = df.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(
            f"{algorithm} needs complete, finite data, but df has "
            f"{int((~np.isfinite(values)).sum())} NaN/inf values. Impute first, "
            "e.g. causalts.imputation.var_em_impute(df) (CLI: --impute var_em)."
        )
    module, name = path.rsplit(".", 1)
    fn = getattr(importlib.import_module(module), name)
    out = fn(df, max_lag=max_lag, **kwargs)
    if len(out) == 3:
        graph, coefficients, info = out
    else:
        (graph, info), coefficients = out, None
    return BaselineResult(graph, df, algorithm, coefficients=coefficients, info=info)


def _check_no_ci_test(algorithm, ci_test):
    if ci_test is not None:
        raise ValueError(f"{algorithm} takes no CI test; got ci_test={ci_test!r}")


# Runners under the plugin registry's ``fn(df=, ci_test=, max_lag=, **kwargs)``
# convention (see causalts.algorithms); ``ci_test`` must be None.
def run_lges(df, ci_test=None, max_lag=1, mode="lges", lambda_value=None):
    """:func:`run_baseline` for ``"lges"``."""
    _check_no_ci_test("lges", ci_test)
    return run_baseline("lges", df, max_lag, mode=mode, lambda_value=lambda_value)


def run_tges(df, ci_test=None, max_lag=1, mode="lges", lambda_value=None):
    """:func:`run_baseline` for ``"tges"``."""
    _check_no_ci_test("tges", ci_test)
    return run_baseline("tges", df, max_lag, mode=mode, lambda_value=lambda_value)


def run_ges(
    df,
    ci_test=None,
    max_lag=1,
    score_func="local_score_BIC",
    lambda_value=None,
    engine="fast",
):
    """:func:`run_baseline` for ``"ges"``."""
    _check_no_ci_test("ges", ci_test)
    return run_baseline(
        "ges",
        df,
        max_lag,
        score_func=score_func,
        lambda_value=lambda_value,
        engine=engine,
    )


def run_lasso_var(df, ci_test=None, max_lag=1, cv_folds=5):
    """:func:`run_baseline` for ``"lasso-var"``."""
    _check_no_ci_test("lasso-var", ci_test)
    return run_baseline("lasso-var", df, max_lag, cv_folds=cv_folds)


def run_granger(df, ci_test=None, max_lag=1, threshold=None):
    """:func:`run_baseline` for ``"granger"``."""
    _check_no_ci_test("granger", ci_test)
    return run_baseline("granger", df, max_lag, threshold=threshold)


def run_varlingam(df, ci_test=None, max_lag=1, threshold=0.05):
    """:func:`run_baseline` for ``"varlingam"``."""
    _check_no_ci_test("varlingam", ci_test)
    return run_baseline("varlingam", df, max_lag, threshold=threshold)
