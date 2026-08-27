# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later
"""LUCID: regime-adaptive deconfounding for time-series causal discovery.

LUCID (Learning Under Confounding for Inference and Discovery). Latent confounders
leave different statistical fingerprints depending on their structure, and no single
correction handles all of them: rank/tetrad methods exploit the low-rank covariance of
*pervasive* factor confounding but destroy true edges under *sparse* (local fork)
confounding, and vice-versa. Applying the wrong correction can be as damaging as
applying none. This module infers the regime from the data and applies the matching
correction.

Pipeline (per dataset):

  1. Route by a data-only statistic into a regime:
       - "sparse"     : full-rank residual covariance (local latent forks)
       - "pervasive"  : one or more dominant latent factors (e.g. GARCH factors)
       ("sf" is an intermediate hub/scale-free regime; see routing below.)
  2. Run skeleton discovery. Both branches use the *same* base engine (plain
     ``run_cdnots``), so the regime affects only the correction that follows -- which
     isolates the contribution of routing from any change in the search itself, and
     means an already-discovered graph can be reused (``run_lucid(discovery=...)``).
  3. Apply the regime-matched correction:
       - sparse    : a post-double-selection (PDS) edge filter over observed controls.
                     Granger-style at lags >= 1; at lag 0 the corresponding
                     contemporaneous partial regression with lagged controls.
       - pervasive : lagged edges are kept as discovered and the lag-0 slice is
                     *reconstructed* by S-L -- attenuate the factor-dominated spectral
                     directions, then recover sparse conditional-dependence candidates
                     from the residual innovations against an edge-free null, with a
                     persistence gate to limit moralization.

Routers:
  - router="auto" (default): parameter-free; the sparse<->pervasive boundary is derived
    from the Marchenko-Pastur no-factor null (adapts to d/T), with a robust sub-regime
    split.
  - router="spectral": spectral-gap statistic with calibrated thresholds (ablation;
    expert users may override the thresholds).
  - router="mp": Marchenko-Pastur factor count (ablation).

Public API:
  run_lucid(df_obs, max_lag, ...)      -> LucidResult  [graph + router diagnostics]
  routed_deconfound(df_obs, max_lag, ...) -> (d, d, max_lag+1) graph  [low-level]
  deconfound(graph, df_obs, max_lag, regime, ...)     -> graph  [post-hoc layer]
  tetrad_filter(df_obs, graph, max_lag, threshold)    -> graph  [fixed comparator]
  pds_filter(df_obs, graph, max_lag, alpha)           -> graph
  spectral_gap(X, k) / mp_factor_count(X)             -> router statistics

Every result object also exposes ``.deconfound()``, ``.tetrad_filter()`` and
``.pds_filter()``; see :mod:`causalts.result`.

Non-default research knobs are retained for the paper's ablations but are not the
shipped configuration: ``pervasive_base="tetrad"`` swaps in the CD-NOTS+ engine plus
the tetrad lag-0 filter, and ``pervasive_filters=True`` re-enables the LLCCA,
volatility-invariance and factor-PDS filters, which the filter ablation found
net-harmful on the unified base.
"""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.linear_model import LassoLarsIC

from ..cdnots.phase3_utils import run_cdnots, run_cdnots_plus
from ..ci_tests.parcorr_gpu import ParCorrGPU
from ..utils.tetrad import apply_tetrad_lag0_filter
from .result import LucidResult

# ── Defaults ─────────────────────────────────────────────────────────────────
# Spectral-router thresholds — calibrated on the synthetic benchmark (ER ~0.16-0.23,
# SF ~0.24-0.34, GARCH ~0.49-0.51). Only used by router="spectral".
_SPECTRAL_TAU = 0.215
_SPECTRAL_TAU2 = 0.40
# router="auto": the sparse<->pervasive boundary is derived from the Marchenko-Pastur
# no-factor null (adapts to d/T, no user knob). _TAU_MARGIN is a mild safety factor
# above the null edge; at d=15/T=1000 it recovers the calibrated 0.215.
_TAU_MARGIN = 1.3
# PDS significance (edge kept iff p < alpha). Plateau values from the alpha sweep.
_ALPHA_PDS_SPARSE = 1e-10
_ALPHA_PDS_PERVASIVE = 1e-8
_ALPHA_VOL = 0.05
_LLCCA_THRESHOLD = 0.05
# Factor-augmented PDS level (homoskedastic-pervasive branch) + ARCH heteroskedasticity
# gate level. Both conventional defaults; no per-problem tuning required.
_ALPHA_FACTOR_PDS = 0.01
_ARCH_ALPHA = 0.05
_HAC_LAGS = 2
_MAX_SEL = 20
_MIN_ENV_SIZE = 30
_TETRAD_THRESHOLD = 0.25
# Number of leading eigenvalues in the router statistic R and its threshold.
_ROUTER_K = 2
# Multiplicative slack on the MP edge in mp_factor_count.
_FACTOR_COUNT_MARGIN = 1.02
# Double-persistence lag-0 gate (research-loop ideas 001c/002/003/007). Soft gate on
# the full-conditioning |pcorr| score by the S=empty marginal (Spearman rank)
# correlation of the deconfounded residuals: gate(m) = 1 - exp(-(m/tau)^2). tau=0.10
# is the VAL-tuned optimum for the rank-correlation gate.
_GATE_TAU = (
    0.15  # validated (30-seed Wilcoxon): 0.15 > 0.10 (tau=0.10 sig. hurts garch/t3_d40)
)
# Winsorization constant (idea 007) for the full-conditioning covariance arm only.
_PC_COV_ESTIMATOR = (
    "sample"  # gate-only is the default method. winsor_4.5 is an OPTIONAL
)
# heavy-tail refinement (30-seed Wilcoxon: significant only on t3_d24 +0.033; neutral on the
# other 6 cells; the apparent t3_d40 dip is not significant). Off by default for parsimony.


# ── Router statistics (data-only) ────────────────────────────────────────────
def _var1_residuals(X):
    """Residuals of a VAR(1) least-squares fit (with intercept)."""
    T = X.shape[0]
    Y = X[1:]
    Z = np.column_stack([np.ones(T - 1), X[:-1]])
    beta, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    return Y - Z @ beta


def _varp_residuals(X, p):
    """Residuals of a VAR(p) least-squares fit (with intercept). p>=1."""
    X = np.asarray(X)
    T = X.shape[0]
    if p <= 1:
        return _var1_residuals(X)
    Y = X[p:]
    Z = np.column_stack([np.ones(T - p)] + [X[p - 1 - k : T - 1 - k] for k in range(p)])
    beta, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    return Y - Z @ beta


def _select_var_order_whiteness(X, max_p=6, alpha=0.05):
    """Lag order by RESIDUAL WHITENESS: raise p until the VAR(p) residuals have no significant
    lagged cross-correlation at lags > p (max |cross-corr| test, Bonferroni over the d^2 pairs).

    Why not AIC/BIC: at d~20 a full VAR(p) adds a dense d x d coefficient block (~d^2 params),
    so an information criterion over the full likelihood over-penalizes it and stays at p=1 even
    when a FEW sparse higher-lag edges are present (validated: BIC no-op; AIC recovers only when
    higher-lag structure is dense/strong). A whiteness test targets exactly "is there temporal
    structure LEFT in the residuals", so it fires on sparse higher-lag too. Bonferroni over d^2
    (not d^2 x horizon) is the calibrated sweet spot: PERFECT do-no-harm on genuinely lag-1 data
    (returns 1, residualization unchanged) with 2-4x more higher-lag recovery than AIC. A
    portmanteau (chi^2 aggregate) variant over-selects and harms lag-1 -- rejected. Recovery is
    partial on the weakest sparse lag>=2 (picks p~1.5-2, not the full order); that residual gap
    is a documented limitation."""
    from scipy.stats import norm

    X = np.asarray(X)
    T, d = X.shape
    max_p = max(1, min(max_p, (T - 1) // (2 * (d + 1))))
    for p in range(1, max_p + 1):
        R = _varp_residuals(X, p)
        n = R.shape[0]
        if n < 8:
            return p
        Rs = (R - R.mean(0)) / (R.std(0) + 1e-12)
        thr = norm.ppf(1 - alpha / (2 * d * d)) / np.sqrt(n)
        white = True
        for lag in range(p + 1, max_p + 1):
            if n - lag < 5:
                continue
            C = (Rs[:-lag].T @ Rs[lag:]) / (n - lag)
            if np.max(np.abs(C)) > thr:
                white = False
                break
        if white:
            return p
    return max_p


def _var_residuals(X, order=1):
    """VAR residuals; order=int for a fixed lag, or "auto" for whiteness-selected VAR(p)
    (`_select_var_order_whiteness`; do-no-harm on lag-1, partially recovers higher-lag).
    An AIC/BIC order selector was evaluated and dropped: it under-detects sparse
    higher-lag structure, because a full VAR(p) block costs ~d^2 parameters."""
    p = _select_var_order_whiteness(X) if order == "auto" else int(order)
    return _varp_residuals(X, p)


def spectral_gap(X, k=2):
    """Top-k eigenvalue mass of the VAR(1) innovation *correlation* matrix."""
    d = X.shape[1]
    C = np.corrcoef(_var1_residuals(X), rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    eigvals = np.sort(np.linalg.eigvalsh(C))[::-1]
    return float(eigvals[:k].sum() / d)


def _mp_null_top_mass(d, T_eff, k=2):
    """Expected top-k eigenvalue mass (/d) of a correlation matrix under the
    Marchenko-Pastur no-factor null. Used to set the sparse<->pervasive boundary
    parameter-free, adapting to the d/T aspect ratio."""
    lam_plus = (1.0 + np.sqrt(d / max(T_eff, 1))) ** 2
    return k * lam_plus / d


def mp_factor_count(X, var_order=1, factor_count_margin=_FACTOR_COUNT_MARGIN):
    """Parameter-free count of pervasive latent factors via the Marchenko-Pastur law.

    Eigenvalues of the VAR(1) innovation *correlation* matrix that exceed the MP upper
    edge lambda_+ = (1 + sqrt(d/T_eff))^2 cannot come from a no-factor (bulk) null, so
    they signal pervasive latent factors. Uses the correlation matrix (unit-variance,
    so the MP null scale sigma^2 = 1), which also makes it robust to per-series
    heteroskedasticity (e.g. GARCH) that would distort a covariance-based edge.

    ``factor_count_margin`` is a small multiplicative slack on the edge so borderline
    bulk eigenvalues are not counted as factors (paper Table 4).

    Returns the number of eigenvalues above the edge (0 = sparse/no pervasive factor).
    """
    resid = _var_residuals(X, var_order)
    T_eff, d = resid.shape
    C = np.corrcoef(resid, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    eigvals = np.linalg.eigvalsh(C)
    q = d / max(T_eff, 1)
    lambda_plus = (1.0 + np.sqrt(q)) ** 2
    return int(np.sum(eigvals > lambda_plus * factor_count_margin))


def _vol_clustering_pvalue(X, lags=10):
    """Ljung-Box p-value for autocorrelation in the shared-volatility series (mean
    squared VAR(1) innovation across variables). Small p => ARCH-type volatility
    clustering (GARCH-like); large p => no clustering (e.g. a scale-free hub factor).
    Returns None on failure."""
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        resid = _var1_residuals(X)
        totvol = np.mean(resid**2, axis=1)
        totvol = totvol - totvol.mean()
        lb = acorr_ljungbox(totvol, lags=[min(lags, len(totvol) // 5)], return_df=True)
        return float(lb["lb_pvalue"].iloc[-1])
    except Exception:
        return None


def _route(
    X,
    router="mp",
    thresholds=None,
    router_k=_ROUTER_K,
    gamma=_TAU_MARGIN,
    factor_count_margin=_FACTOR_COUNT_MARGIN,
):
    """Return regime label in {"sparse", "sf", "pervasive"} plus a diagnostics dict.

    ``router_k`` is the number of leading eigenvalues entering the statistic and its
    threshold; because both scale with k they largely cancel under the no-factor null,
    so the routing decision is insensitive to it (paper Appendix C). ``gamma`` is the
    fixed margin over the Marchenko-Pastur edge.
    """
    if router in ("auto", "spectral"):
        R = spectral_gap(X, k=router_k)
        if router == "auto":
            # sparse<->pervasive boundary from the MP null (adapts to d/T); SF<->GARCH
            # boundary is a robust default in a wide gap (0.31->0.49), expert-overridable.
            T_eff, d = X.shape[0] - 1, X.shape[1]
            tau = gamma * _mp_null_top_mass(d, T_eff, k=router_k)
            tau2 = thresholds[1] if thresholds else _SPECTRAL_TAU2
        else:  # "spectral": fixed calibrated thresholds
            tau, tau2 = thresholds or (_SPECTRAL_TAU, _SPECTRAL_TAU2)
        if R <= tau:
            regime = "sparse"
        elif R <= tau2:
            regime = "sf"
        else:
            regime = "pervasive"
        return regime, {
            "router": router,
            "regime": regime,
            "R": R,
            "tau": tau,
            "tau2": tau2,
            "router_k": router_k,
            "gamma": gamma,
        }
    elif router == "mp":
        # Stage 1 (Marchenko-Pastur edge): any pervasive latent factor present?
        # This cleanly separates sparse (0 factors) from the pervasive family, but
        # NOT scale-free from GARCH -- a scale-free hub is also low-rank, so both
        # show factors. Stage 2 splits the pervasive family by a volatility-
        # clustering test: GARCH has ARCH-type clustering, a hub confounder does not.
        n = mp_factor_count(X, factor_count_margin=factor_count_margin)
        if n == 0:
            regime = "sparse"
            return regime, {
                "router": "mp",
                "regime": regime,
                "n_factors": n,
                "vol_clustering_p": None,
            }
        p_arch = _vol_clustering_pvalue(X)
        regime = "pervasive" if (p_arch is not None and p_arch < 0.05) else "sf"
        return regime, {
            "router": "mp",
            "regime": regime,
            "n_factors": n,
            "vol_clustering_p": p_arch,
        }
    else:
        raise ValueError(f"unknown router {router!r} (use 'mp' or 'spectral')")


# ── Edge filters ─────────────────────────────────────────────────────────────
def _lagged_design(X, max_lag):
    """Stack lag blocks 1..max_lag into one design matrix aligned to Y = X[max_lag:]."""
    T, d = X.shape
    rows = T - max_lag
    cols, names = [], []
    for lag in range(1, max_lag + 1):
        start = max_lag - lag
        cols.append(X[start : start + rows])
        names += [(j, lag) for j in range(d)]
    Z = np.column_stack(cols) if cols else np.zeros((rows, 0))
    return Z, names, rows


def _select_controls(Zc, target):
    if Zc.shape[1] == 0:
        return []
    mu, sd = Zc.mean(0), Zc.std(0) + 1e-8
    try:
        m = LassoLarsIC(criterion="bic", max_iter=200).fit((Zc - mu) / sd, target)
        return np.where(np.abs(m.coef_) > 1e-8)[0].tolist()
    except Exception:
        return []


def _pds_test(y, x, Zc, alpha):
    sel = list(set(_select_controls(Zc, y)) | set(_select_controls(Zc, x)))[:_MAX_SEL]
    Xd = np.column_stack([x, Zc[:, sel]]) if sel else x.reshape(-1, 1)
    try:
        model = sm.OLS(y, sm.add_constant(Xd)).fit(
            cov_type="HAC", cov_kwds={"maxlags": _HAC_LAGS}
        )
        pval = model.pvalues[1]
    except Exception:
        return True  # abstain (keep) on numerical failure
    return True if np.isnan(pval) else pval < alpha


def tetrad_filter(df_obs, g_est, max_lag, threshold=_TETRAD_THRESHOLD):
    """Drop lag-$0$ edges between variable pairs that share a latent factor.

    A fixed (non-adaptive) deconfounding strategy, and the natural comparator for
    LUCID's regime-adaptive routing: it always assumes pervasive factor structure. Uses
    the tetrad vanishing condition (Spearman 1928) via
    :func:`causalts.utils.tetrad.detect_shared_factors`.

    Post-hoc like the other filters in this module -- it consumes an already-discovered
    graph and never re-runs discovery, so it composes with any base engine.

    Parameters
    ----------
    df_obs : DataFrame (T, d)
        Observed data.
    g_est : ndarray (d, d, max_lag+1)
        Discovered graph to filter.
    max_lag : int
        Maximum lag (present for signature parity with the other filters; the tetrad
        test only touches the lag-0 slice).
    threshold : float, default 0.25
        Factor-consistency threshold; lower is more conservative.

    Returns
    -------
    ndarray
        Copy of ``g_est`` with confounded lag-0 edges removed.
    """
    return apply_tetrad_lag0_filter(
        g_est, df_obs, list(df_obs.columns), threshold=threshold
    )


def pds_filter(df_obs, g_est, max_lag, alpha):
    """Post-double-selection Granger edge filter: drop edges whose direct effect is
    insignificant after Lasso-selecting controls from all lagged variables."""
    X = df_obs.values
    d = X.shape[1]
    Z_full, names_full, rows = _lagged_design(X, max_lag)
    Y = X[max_lag:]
    idx_map = {key: c for c, key in enumerate(names_full)}
    g_out = g_est.copy()
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            for lag in range(max_lag + 1):
                if g_est[i, j, lag] == 0:
                    continue
                y = Y[:, j]
                if lag == 0:
                    x_edge, ctrl_keys = X[max_lag:, i], names_full
                else:
                    key = (i, lag)
                    if key not in idx_map:
                        continue
                    x_edge = Z_full[:, idx_map[key]]
                    ctrl_keys = [k for k in names_full if k != key]
                ctrl_idx = [idx_map[k] for k in ctrl_keys]
                Zc = Z_full[:, ctrl_idx] if ctrl_idx else np.zeros((rows, 0))
                if not _pds_test(y, x_edge, Zc, alpha):
                    g_out[i, j, lag] = 0
    return g_out


def _llcca_asymmetry(X, i, j, max_lag):
    fwd = bwd = 0.0
    T = X.shape[0]
    for lag in range(1, max_lag + 1):
        if T - lag < 2:
            continue
        c_fwd = np.corrcoef(X[: T - lag, i], X[lag:, j])[0, 1]
        c_bwd = np.corrcoef(X[: T - lag, j], X[lag:, i])[0, 1]
        if not np.isnan(c_fwd):
            fwd += abs(c_fwd)
        if not np.isnan(c_bwd):
            bwd += abs(c_bwd)
    return (fwd - bwd) / (fwd + bwd + 1e-8)


def llcca_filter(df_obs, g_est, max_lag, threshold=_LLCCA_THRESHOLD):
    """Lead-lag cross-correlation asymmetry: drop edges whose forward/backward
    cross-correlation is near-symmetric (the footprint of a shared latent cause
    rather than a directed effect)."""
    X = df_obs.values
    d = X.shape[1]
    g_out = g_est.copy()
    for i in range(d):
        for j in range(d):
            if i == j or not np.any(g_out[i, j, :]):
                continue
            if _llcca_asymmetry(X, i, j, max_lag) < threshold:
                g_out[i, j, :] = 0
    return g_out


def _volatility_environments(X, max_lag):
    """0/1 low/high shared-volatility label per row of Y=X[max_lag:], from the mean
    squared VAR(1) innovation across all variables (a common latent-vol proxy)."""
    T, d = X.shape
    resid = _var1_residuals(X)
    logvol = np.log(np.mean(resid**2, axis=1) + 1e-12)
    try:
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(logvol.reshape(-1, 1))
        env_full = km.labels_
        if logvol[env_full == 0].mean() > logvol[env_full == 1].mean():
            env_full = 1 - env_full
    except Exception:
        return None
    rows = T - max_lag
    start = max_lag - 1
    if start < 0 or start + rows > env_full.shape[0]:
        return None
    return env_full[start : start + rows]


def _vol_invariance_test(y, x_edge, env, alpha):
    if int(np.sum(env == 0)) < _MIN_ENV_SIZE or int(np.sum(env == 1)) < _MIN_ENV_SIZE:
        return True  # abstain
    design = sm.add_constant(np.column_stack([x_edge, env, x_edge * env]))
    try:
        model = sm.OLS(y, design).fit(cov_type="HAC", cov_kwds={"maxlags": _HAC_LAGS})
        pval = model.pvalues[-1]
    except Exception:
        return True
    return True if np.isnan(pval) else pval >= alpha


def vol_invariance_filter(df_obs, g_est, max_lag, alpha=_ALPHA_VOL):
    """Drop edges whose regression coefficient depends on a shared volatility regime
    (a by-product of a latent common volatility factor)."""
    X = df_obs.values
    d = X.shape[1]
    env = _volatility_environments(X, max_lag)
    if env is None:
        return g_est
    Z_full, names_full, rows = _lagged_design(X, max_lag)
    Y = X[max_lag:]
    idx_map = {key: c for c, key in enumerate(names_full)}
    g_out = g_est.copy()
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            for lag in range(max_lag + 1):
                if g_out[i, j, lag] == 0:
                    continue
                y = Y[:, j]
                if lag == 0:
                    x_edge = X[max_lag:, i]
                else:
                    key = (i, lag)
                    if key not in idx_map:
                        continue
                    x_edge = Z_full[:, idx_map[key]]
                if not _vol_invariance_test(y, x_edge, env, alpha):
                    g_out[i, j, lag] = 0
    return g_out


# ── Discovery + regime-specific pipelines ────────────────────────────────────
def _pca_factor_scores(X, k):
    """Top-k PCA scores of the VAR(1) innovation covariance (aligned to X[1:]).
    A cheap, parameter-free proxy for the MP-detected latent pervasive factor(s):
    unlike the lasso-selected controls in `pds_filter`, these scores are an explicit
    estimate of the shared confounder itself rather than other (equally confounded)
    observed variables."""
    resid = _var1_residuals(X)  # (T-1, d), aligned to X[1:]
    if k <= 0 or resid.shape[0] < 5:
        return None
    Xc = resid - resid.mean(axis=0)
    try:
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    k = min(k, Vt.shape[0])
    return Xc @ Vt[:k].T  # (T-1, k), row r aligns to X[r + 1]


def _drop_instantaneous(g_est):
    """Zero out lag-0 (contemporaneous) edges. Under a diagnosed pervasive/hub
    latent-factor regime an instantaneous cross-sectional correlation cannot be
    assigned a direction without extra assumptions (no time ordering) -- it is the
    footprint of a shared same-period confounder, not a lagged direct effect.
    `llcca_filter` cannot catch these (its asymmetry statistic uses lag>=1 cross-
    correlations only)."""
    g_out = g_est.copy()
    g_out[:, :, 0] = 0
    return g_out


def _kendall_sine_corr(U):
    """Tail-robust correlation of the columns of ``U`` via Kendall's tau -> sin(pi/2 * tau)
    (the transelliptical/nonparanormal SKEPTIC relation), PSD-repaired by clipping negative
    eigenvalues. Rank-based, so robust to heavy-tailed (Student-t / GARCH) innovations.
    """
    from scipy.stats import kendalltau

    d = U.shape[1]
    corr = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            tau, _ = kendalltau(U[:, i], U[:, j])
            corr[i, j] = corr[j, i] = np.sin(
                np.pi / 2.0 * (0.0 if np.isnan(tau) else tau)
            )
    w, V = np.linalg.eigh((corr + corr.T) / 2)
    corr = (V * np.clip(w, 1e-4, None)) @ V.T
    dd = np.sqrt(np.clip(np.diag(corr), 1e-12, None))
    return corr / np.outer(dd, dd)


def _rank_corr(U):
    """Pairwise Spearman (rank) correlation matrix of the columns of ``U``."""
    from scipy.stats import rankdata

    R = np.apply_along_axis(rankdata, 0, U)
    C = np.corrcoef(R, rowvar=False)
    return np.nan_to_num(C, nan=0.0)


def _double_persistence_score(pc, marg, tau=_GATE_TAU):
    """Soft moralization gate: suppress pairs whose full-conditioning score ``pc``
    lacks any marginal (S=empty) corroboration ``marg``.

    ``tau <= 0`` disables the gate (returns ``pc`` unchanged) -- the ablation arm. Without
    this guard tau=0 would give 0/0 -> nan for any exactly-zero marginal correlation.
    """
    if tau <= 0:
        return pc
    gate = 1.0 - np.exp(-np.square(marg / tau))
    return pc * gate


def _winsorize_cols(U, c):
    """Per-column MAD-based winsorization -- clip each column at median +/- c*MAD."""
    med = np.median(U, axis=0, keepdims=True)
    mad = np.median(np.abs(U - med), axis=0, keepdims=True) * 1.4826
    mad = np.maximum(mad, 1e-8)
    return med + np.clip(U - med, -c * mad, c * mad)


def _sl_cov(U, cov_estimator="sample"):
    """Covariance of the deconfounded residuals ``U`` for the S-L contemporaneous step.

    ``cov_estimator``:
      "sample"         : np.cov -- the shipped default; best under Gaussian noise (do-no-harm
                         to the existing benchmark).
      "kendall_shrink" : transelliptical Kendall sine correlation (`_kendall_sine_corr`) plus
                         identity shrinkage delta=clip(d/n, 0.05, 0.9), in CORRELATION units
                         (partial correlation is scale-invariant, so no marginal std/MAD
                         scaling). Validated (2026-08-12 covariance ablation, 20 seeds,
                         d in {10,24,40}) to STRICTLY DOMINATE "sample" for lag-0 recovery in
                         the heavy-tailed, moderate-to-high-dimensional regime (t3 noise,
                         d>=24) -- the financial regime; it pays a modest efficiency cost
                         under Gaussian noise, so it is OPTIONAL, not the default. The
                         identity shrinkage is what rescues the unregularized Kendall
                         estimator's high-d/low-T collapse.
    """
    if cov_estimator == "sample":
        return np.cov(U, rowvar=False)
    if cov_estimator == "kendall_shrink":
        n, d = U.shape
        R = _kendall_sine_corr(U)
        delta = float(np.clip(d / max(n, 1), 0.05, 0.9))
        return (1 - delta) * R + delta * np.eye(d)
    if cov_estimator.startswith("winsor_"):
        c = float(cov_estimator.split("_", 1)[1])
        Uw = _winsorize_cols(U, c)
        return np.cov(Uw, rowvar=False)
    raise ValueError(
        f"unknown cov_estimator {cov_estimator!r} (use 'sample', 'kendall_shrink', or 'winsor_<c>')"
    )


def _sl_deconf_pcorr(
    X,
    k,
    ktrim_margin=0,
    cov_estimator=_PC_COV_ESTIMATOR,
    gate_tau=_GATE_TAU,
    presvd_c=None,
    var_order=1,
):
    """|partial correlation| of the low-rank-deconfounded contemporaneous precision (S-L).

    Spectral-trim the top-k singular values of the VAR(1) innovations (removes the pervasive-factor
    footprint), then take partial correlations of the deconfounded precision. Returns
    (pcorr[d,d], U[n,d] innovations, (Uu,s,Vt) SVD, Udec[n,d] deconfounded residuals).

    ``ktrim_margin`` (research, ported from the profile="sl" full-replacement research loop's
    idea-020/`_sl_multilag_skeleton`, see SL_VALIDATION.md): an extra integer safety margin
    added on top of the nominal k trim count. mp_factor_count's k can under-trim under
    heteroskedasticity (a volatility-clustering factor's smaller singular values blend into
    the bulk edge more than a homoskedastic factor's would), leaving a stable (non-noise)
    residual footprint that no downstream threshold can separate from a genuine edge.
    Default 0 preserves the original (pre-research) behavior exactly.
    """
    U = _var_residuals(
        X, var_order
    )  # var_order="auto" -> BIC-selected VAR(p) (higher-lag safe)
    U = U - U.mean(0, keepdims=True)
    d = U.shape[1]
    Usvd_in = _winsorize_cols(U, presvd_c) if presvd_c is not None else U
    Uu, s, Vt = np.linalg.svd(Usvd_in, full_matrices=False)
    ktrim = min(k + max(int(ktrim_margin), 0), len(s) - 1)
    s2 = s.copy()
    if 0 < ktrim < len(s):
        s2[:ktrim] = s[ktrim]
    Udec = (Uu * s2) @ Vt
    S = _sl_cov(Udec, cov_estimator=cov_estimator)
    P = np.linalg.pinv(S + 1e-3 * np.trace(S) / d * np.eye(d))
    dP = np.sqrt(np.clip(np.diag(P), 1e-12, None))
    pc = np.abs(-P / np.outer(dP, dP))
    np.fill_diagonal(pc, 0.0)
    marg = np.abs(_rank_corr(Udec))
    pc = _double_persistence_score(pc, marg, tau=gate_tau)
    np.fill_diagonal(pc, 0.0)
    return pc, U, (Uu, s, Vt), Udec


def _sl_maxt_threshold(
    U,
    svd,
    k,
    alpha=0.05,
    n_boot=200,
    seed=0,
    ktrim_margin=0,
    cov_estimator=_PC_COV_ESTIMATOR,
    gate_tau=_GATE_TAU,
    presvd_c=None,
):
    """Edge-free-null max-T threshold: keep the factor part of the innovations, circular-shift each
    idiosyncratic coordinate independently (destroys contemporaneous idiosyncratic EDGES, preserves
    the factor FOOTPRINT + serial structure), rerun the S-L score, and return the (1-alpha) quantile
    of the null max. Do-no-harm by construction: P(any edge | edge-free) <= alpha.

    The persistence gate is applied to EVERY null draw, so the quantile is taken over the same
    gated statistic `_sl_deconf_pcorr` returns for the observed data -- not over a raw |pcorr|.
    Calibrating a gated observed score against an ungated null would compare two different
    statistics and misplace the threshold.

    ``ktrim_margin``: MUST match the value passed to `_sl_deconf_pcorr` for the observed
    statistic -- both the null and the observed score need the same effective trim amount,
    or the threshold is calibrated against a different-bias statistic than the one it's
    applied to."""
    Uu, s, Vt = svd
    n, d = U.shape
    Uc = _winsorize_cols(U, presvd_c) if presvd_c is not None else U
    ktrim = min(k + max(int(ktrim_margin), 0), len(s) - 1)
    # factor part -- ELEVATED ktrim (matches the multilag original's convention: the
    # extra margin components are treated as factor for BOTH the null split and the
    # trim-squash step)
    Uk = (Uu[:, :ktrim] * s[:ktrim]) @ Vt[:ktrim, :]
    Ures = Uc - Uk  # idiosyncratic (holds genuine edges)
    iu = np.triu_indices(d, 1)
    rng = np.random.default_rng(seed)
    null_max = np.empty(n_boot)
    for b in range(n_boot):
        sh = rng.integers(1, n, size=d)
        Uperm = np.column_stack([np.roll(Ures[:, j], sh[j]) for j in range(d)])
        Ustar = Uk + Uperm
        uu, ss, vt = np.linalg.svd(Ustar, full_matrices=False)
        ss2 = ss.copy()
        if 0 < ktrim < len(ss):
            ss2[:ktrim] = ss[ktrim]
        Udec_null = (uu * ss2) @ vt
        Sd = _sl_cov(Udec_null, cov_estimator=cov_estimator)
        Pn = np.linalg.pinv(Sd + 1e-3 * np.trace(Sd) / d * np.eye(d))
        dPn = np.sqrt(np.clip(np.diag(Pn), 1e-12, None))
        pcn = np.abs(-Pn / np.outer(dPn, dPn))
        np.fill_diagonal(pcn, 0.0)
        marg_null = np.abs(_rank_corr(Udec_null))
        pcn = _double_persistence_score(pcn, marg_null, tau=gate_tau)
        null_max[b] = pcn[iu].max()
    return float(np.quantile(null_max, 1 - alpha))


def _sl_block_bootstrap_column(col, block_len, rng):
    """Stationary block bootstrap of a single 1-D array (ported from the profile="sl"
    full-replacement research loop's idea-004): concatenate randomly-placed (wrap-around)
    contiguous blocks of length ``block_len`` until reaching the original length, then
    truncate. See `_sl_multilag_blockboot_threshold`'s docstring for the rationale (fixes
    a circular-shift null's burst-alignment inflation under GARCH without stripping real
    variance-driven signal)."""
    n = len(col)
    if block_len >= n:
        start = rng.integers(0, n)
        return np.roll(col, -start)[:n]
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n, size=n_blocks)
    out = np.empty(n_blocks * block_len)
    for bi, s in enumerate(starts):
        idx = (s + np.arange(block_len)) % n
        out[bi * block_len : (bi + 1) * block_len] = col[idx]
    return out[:n]


def _sl_blockboot_threshold(
    U,
    svd,
    k,
    alpha=0.05,
    n_boot=1000,
    seed=0,
    ktrim_margin=0,
    block_len=20,
    cov_estimator=_PC_COV_ESTIMATOR,
    gate_tau=_GATE_TAU,
    presvd_c=None,
):
    """Single-lag-0 analogue of `_sl_maxt_threshold`, using the stationary block bootstrap
    (`_sl_block_bootstrap_column`) instead of a single circular shift per idiosyncratic
    column. Research (2026-08-10): tested as an alternative to `_sl_maxt_threshold` inside
    `sl_lag0_recover`/Workstream A; see SL_VALIDATION.md for the result."""
    Uu, s, Vt = svd
    n, d = U.shape
    Uc = _winsorize_cols(U, presvd_c) if presvd_c is not None else U
    ktrim = min(k + max(int(ktrim_margin), 0), len(s) - 1)
    Uk = (Uu[:, :ktrim] * s[:ktrim]) @ Vt[:ktrim, :]
    Ures = Uc - Uk
    iu = np.triu_indices(d, 1)
    rng = np.random.default_rng(seed)
    null_max = np.empty(n_boot)
    for b in range(n_boot):
        Uperm = np.column_stack(
            [_sl_block_bootstrap_column(Ures[:, j], block_len, rng) for j in range(d)]
        )
        Ustar = Uk + Uperm
        uu, ss, vt = np.linalg.svd(Ustar, full_matrices=False)
        ss2 = ss.copy()
        if 0 < ktrim < len(ss):
            ss2[:ktrim] = ss[ktrim]
        Udec_null = (uu * ss2) @ vt
        Sd = _sl_cov(Udec_null, cov_estimator=cov_estimator)
        Pn = np.linalg.pinv(Sd + 1e-3 * np.trace(Sd) / d * np.eye(d))
        dPn = np.sqrt(np.clip(np.diag(Pn), 1e-12, None))
        pcn = np.abs(-Pn / np.outer(dPn, dPn))
        np.fill_diagonal(pcn, 0.0)
        marg_null = np.abs(_rank_corr(Udec_null))
        pcn = _double_persistence_score(pcn, marg_null, tau=gate_tau)
        null_max[b] = pcn[iu].max()
    return float(np.quantile(null_max, 1 - alpha))


def _pairwise_lingam_lr(x, y):
    """Hyvarinen-Smith pairwise LiNGAM likelihood ratio. >0 => x->y, <0 => y->x, ~0 => Gaussian
    (unorientable). Reliable only under non-Gaussianity."""
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    rho = np.mean(x * y)
    return float(rho * (np.mean(x**3 * y) - np.mean(x * y**3)))


# Sample excess kurtosis of a true Gaussian at T=300 has sampling std ~= sqrt(24/300) ~= 0.28;
# t3 innovations run excess kurtosis into the several-to-many range. 1.0 (~3.5 sigma) cleanly
# separates the two regimes without false-triggering on Gaussian sampling noise.
_KURT_GATE = 0.6


def _cross_lag_score(xa, xb):
    """Cedar-style lag-1 lead-lag asymmetry (Gaussian-friendly -- uses temporal asymmetry, not
    higher moments): a_ab = |corr(xa[:-1], xb[1:])|, a_ba = |corr(xb[:-1], xa[1:])|; >0 favors
    a->b, <0 favors b->a. (A confound-robust "partial" variant -- residualizing each lag-1
    predictor on the other series' own lag-1 -- was tried and made this WORSE: on gaussAR_d20 it
    collapsed a plain-cross_lag 0.75 pairwise accuracy to 0.50/chance, because when a->b and a is
    autocorrelated, b's own lag-1 already carries a's lagged influence via the edge, so
    partialling it out strips the very signal the asymmetry needs. Kept plain.)"""
    a_ab = abs(float(np.corrcoef(xa[:-1], xb[1:])[0, 1]))
    a_ba = abs(float(np.corrcoef(xb[:-1], xa[1:])[0, 1]))
    return a_ab - a_ba


def _bivar_var1_asymmetry(xa, xb):
    """Joint bivariate VAR(1) Granger-direction asymmetry (research-loop idea 008).

    Fits [xa_t, xb_t] ~ const + xa_{t-1} + xb_{t-1} jointly (one 2-variable OLS, not two
    separate univariate correlations) and compares the cross-lag coefficient magnitudes:
    |xa_{t-1} -> xb_t| vs |xb_{t-1} -> xa_t|. >0 favors a->b.

    Why this beats plain pairwise cross-correlation (`_llcca_asymmetry`) on the COLLIDER
    regime specifically: when a target has multiple contemporaneous parents (or any other
    lag-1 predictor), a plain corr(xa_{t-1}, xb_t) mixes in xb's own persistence (xb_{t-1}
    is correlated with xa_{t-1} through the stationary cross-covariance), diluting the
    edge-specific signal. The joint regression's coefficient on xa_{t-1} already controls
    for xb_{t-1} (and vice versa) in the SAME fit, so each cross-lag coefficient is the
    edge's marginal contribution net of the target's own AR and of the correlated regressor
    -- a sharper, jointly-estimated Granger-direction statistic instead of two independent
    correlations."""
    n = min(len(xa), len(xb))
    xa = np.asarray(xa)[:n]
    xb = np.asarray(xb)[:n]
    if n < 8:
        return 0.0
    Y = np.column_stack([xa[1:], xb[1:]])
    Z = np.column_stack([np.ones(n - 1), xa[:-1], xb[:-1]])
    try:
        beta, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0
    b_b_from_a = abs(beta[1, 1])  # xa_{t-1} -> xb_t  (a leads b)
    b_a_from_b = abs(beta[2, 0])  # xb_{t-1} -> xa_t  (b leads a)
    return (b_b_from_a - b_a_from_b) / (b_b_from_a + b_a_from_b + 1e-8)


def _bivar_var1_blockboot_median(xa, xb, n_boot=25, block_len=25, seed=0):
    """Block-bootstrap MEDIAN of `_bivar_var1_asymmetry` (research-loop idea 014).

    Motivation: ideas 011a/011b/013 each tried reformulating the SAME single joint-OLS fit
    (t-stat/SE normalization, extra lags, rank transform) and all made collider WORSE --
    the single-fit point estimate itself is not mis-specified, it is just noisy on collider
    (an unobserved co-parent inflates the target equation's residual variance without biasing
    the coefficient). Rather than another single-fit reformulation, resample overlapping
    contiguous blocks (preserves the VAR(1) lag structure, unlike i.i.d. resampling) and take
    the MEDIAN of the resulting asymmetry statistic across resamples -- a variance-reduction
    (bagging) move, not a different estimator of the same one-shot quantity."""
    n = min(len(xa), len(xb))
    xa = np.asarray(xa)[:n]
    xb = np.asarray(xb)[:n]
    if n < 2 * block_len:
        return _bivar_var1_asymmetry(xa, xb)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_len))
    vals = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block_len, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:n]
        vals.append(_bivar_var1_asymmetry(xa[idx], xb[idx]))
    return float(np.median(vals))


# Shipped default. Switched twice before landing here (2026-08-17 to "llcca", 2026-08-18
# back to "leadlag", now to "llcca" again on 2026-08-19) -- see "WHICH RULE SHIPS" in
# orient_lag0_pair for why each prior measurement was wrong and what the corrected,
# code_version-stamped comparison actually shows. Read that docstring before touching this.
_ORIENT_RULE = "llcca"


def orient_lag0_pair(xa, xb, ua, ub, rule=None):
    """Orientation score for a contemporaneous (lag-0) edge between two variables.

    ``rule`` selects the statistic (default ``_ORIENT_RULE`` = "llcca"):
      "llcca"   : pairwise lead-lag correlation asymmetry on the raw series. THE DEFAULT --
                  see "WHICH RULE SHIPS" below.
      "leadlag" : block-bootstrap-median joint bivariate VAR(1) asymmetry + 0.5 * pairwise
                  cross-lag score (research-loop ideas 008/014/015). A FORMER default;
                  kept as an ablation arm, and described at length below because the
                  per-edge validation history is still the clearest statement of what each
                  term was meant to buy.
      "bivar"   : idea-008 alone (one joint VAR(1) fit, no bootstrap, no fusion). ABLATION.
      "boot"    : idea-014 (blockboot median of "bivar", no fusion). ABLATION.
      "lingam"  : pairwise LiNGAM likelihood ratio on the deconfounded residuals; needs
                  non-Gaussianity, at chance without it. ABLATION ARM (Appendix app:orient).

    WHICH RULE SHIPS: "llcca", as of 2026-08-19, confirmed on the shipped pipeline with
    every unit's producing commit stamped in the saved record (``s3_utils.code_version``).
    Restricted to pervasive_dense -- the ONLY family where S-L recovers any lag-0
    candidates; the other nine tie exactly, verified unit-for-unit identical graphs, because
    the orientation function is never called there:

        method        leadlag    llcca     delta (llcca-leadlag)
        lucid_nofilt   0.3198   0.3460    +0.0262
        lucid_pcmci    0.3645   0.3910    +0.0265
        lucid_nts      0.2894   0.3180    +0.0286

    llcca wins on all three base engines, consistently rather than noisily. Overall
    (family-weighted across all ten families) moves by only ~+0.003, because the effect is
    confined to one family out of ten -- do not read a small Overall delta as "the choice
    doesn't matter"; read the pervasive_dense row.

    Why llcca over lingam, given lingam was competitive in per-edge validation: lingam's
    identifiability requires non-Gaussian residuals and it is at chance without them, and
    6 of the 8 pervasive_dense generators draw Gaussian innovations (`rng.normal()` in
    dense_dgps.py) -- lingam's advantage concentrates on the 2 explicitly non-Gaussian
    generators (heavy-tailed contamination, GARCH) and is a liability elsewhere. llcca's
    lead-lag mechanism has no such dependency.

    TWO PRIOR DEFAULTS, BOTH WRONG FOR DIFFERENT REASONS -- do not repeat either mistake:

    (1) A 20-seed ablation (`run_orient_lag0only.py`) reported llcca beating "leadlag" by
    +0.032 on the lag-0 slice, and the default was switched on it. That ablation called
    `routed_deconfound_lucid(df, ml, ci=ParCorrGPU(...))` with NO `base_discovery`, building
    its own skeleton instead of the shipped `base_discovery=_reuse_base("naive_cdnots",
    cdnots@alpha=0.05)`. A different skeleton yields a different lag-0 candidate set, and the
    rules ranked differently on it. Fix: any orientation comparison MUST run the shipped
    base engine, or it measures a different method.

    (2) The default was then reverted to "leadlag" using an end-to-end comparison that
    DID use the correct base engine, but read a stale `results/` store: the sweep ran in
    phases, and pervasive_dense's units for lucid_nofilt/pcmci/nts had not been recomputed
    since 2026-08-13 -- before `orient_lag0_pair` existed at all (only plain pairwise-LiNGAM
    was available then), before the persistence gate, and before VAR-order selection. The
    "leadlag" numbers being compared were therefore a mislabeled historical LiNGAM run, not
    the shipped rule. Confirmed by checking out that exact commit and reproducing its
    number bit-for-bit. Fix: `code_version` is now stamped into every saved result
    specifically so a future version of this mistake fails loudly (a mismatch across a
    method's files) instead of silently.

    NOTE an all-lag F1 compresses any orientation difference to ~0.002: 158 of the suite's
    167 lag-0 ground-truth cross edges live in pervasive_dense. Read the dense row.

    >0 => a->b, <0 => b->a, ~0 (|.|<1e-6) => unorientable (kept undirected).

    ``xa, xb``: the RAW observed series (retain AR / temporal structure -- needed for any
    lead-lag / cross-lag statistic). ``ua, ub``: the deconfounded VAR(1) innovation residuals
    (temporally white by construction -- used for the non-Gaussian pairwise-LiNGAM statistic).

    DEFAULT = joint bivariate VAR(1) Granger-direction asymmetry on the RAW series
    (`_bivar_var1_asymmetry`, research-loop idea 008). In a FAITHFUL structural SVAR a
    contemporaneous edge a->b makes the target b accumulate a's influence and persist it
    through b's own AR, so the lead-lag becomes asymmetric even for GAUSSIAN edges -- which
    pairwise LiNGAM cannot orient. The joint-regression form additionally controls each
    cross-lag coefficient for the OTHER lag-1 regressor in the same fit, which sharpens the
    asymmetry when the target has extra structure (e.g. a second parent) diluting a plain
    pairwise correlation -- see `_bivar_var1_asymmetry` docstring.

    VALIDATION (faithful (I-B)^{-1} DGPs, orient_loop_bench.py, VAL seeds 0-19 / FINAL seeds
    40-59): bivariate-VAR1 vs the prior LLCCA-pairwise-correlation default --
    block .89/.85->.84/.85, dense .85/.87->.81/.87, weak .94/.88->.86/.88, hetero .88/.87->.
    88/.87, t3 .89/.81->.84/.81, garch .90/.87->.82/.87, hub .883/.883->.892/.883, COLLIDER
    (the prior worst cell) .75/.?? -> .7875 (VAL) / .7958 (FINAL) -- consistent +0.04-0.05 lift
    on collider on BOTH disjoint splits, all other cells stay within noise of the prior
    default (no regression on any cell in either split). Worst-case 0.7500 -> 0.7875 (VAL).
    Mechanism: collider targets have TWO contemporaneous parents, so a plain pairwise
    cross-correlation for either parent is diluted by the correlated presence of the other
    parent's lag-1 value; the joint bivariate-VAR1 fit controls for the co-regressor within
    the same regression, recovering more of the edge-specific signal. `_llcca_asymmetry`
    (multi-lag pairwise correlation) and `_pairwise_lingam_lr` (residuals) are kept available
    as fallbacks/ablations. ``ua, ub`` retained in the signature for that ablation.

    Idea 014 wraps this in a block-bootstrap MEDIAN (`_bivar_var1_blockboot_median`, n_boot=25,
    block_len=25) for variance reduction. VALIDATION (orient_loop_bench.py, VAL seeds 0-19 /
    FINAL seeds 40-59): block .84/.85->.90/.89, dense .81/.87->.86/.87, weak .86/.88->.88/.88,
    hetero .88/.87->.90/.87, t3 .84/.81->.87/.85, garch .82/.87->.84/.87, hub .8917/.8833->.
    9333/.9167, COLLIDER (worst cell) .7875/.7958->.8417/.8208. Every cell improved or held on
    BOTH disjoint splits (no regressions) -- worst-case 0.7875->0.8400 (VAL) / 0.7958->0.8208
    (FINAL). Mechanism: a single-shot joint-VAR1 OLS point estimate on T=250 is noisy
    (amplified on collider by the unobserved co-parent's residual variance); the block
    bootstrap resamples overlapping length-25 blocks (preserving lag-1 structure, unlike i.i.d.
    resampling) and takes the MEDIAN of the resulting asymmetry statistic, denoising the
    estimate without changing which fit is used -- a variance-reduction (bagging) move, not a
    reformulation of the statistic itself (unlike ideas 011a/011b/013, which all reformulated
    the single-fit estimator and failed).

    Idea 015 adds a small ADDITIVE fusion term: `bivar_var1_blockboot_median(xa,xb) +
    0.5*cross_lag_score(xa,xb)`. `_cross_lag_score` (Cedar-style raw pairwise lag-1 lead-lag
    asymmetry) is an independent estimate of the same lead-lag direction, computed differently
    (two separate raw correlations, no joint co-regressor control) -- summing (not gating) the
    two denoises further on cells where either alone is marginal. VALIDATION (orient_loop_bench.py,
    VAL seeds 0-19 / FINAL seeds 40-59): block .90/.89, dense .86/.87, weak .88/.88, hetero
    .90/.87, t3 .87/.85, garch .84/.87, hub .933/.917, COLLIDER (worst cell) .8417/.8208->
    .8625/.8375. Every cell improved or held on both splits. Worst-case 0.8400->0.8625 (VAL) /
    0.8208->0.8375 (FINAL). Note (research-loop idea 017): this fusion was already confirmed by
    idea 015 but got accidentally wiped by the idea-016 revert (both were bundled in one commit,
    and reverting idea-016's failed 3rd-term extensions reverted idea-015's win along with it) --
    restored here as a standalone commit. Idea 016 (a 3rd additive term: retuned bootstrap
    hyperparameters, +gamma*lingam, bootstrapped cross_lag) tried and failed to beat this on
    BOTH splits simultaneously -- do not repeat."""
    rule = _ORIENT_RULE if rule is None else rule
    if rule == "lingam":
        return _pairwise_lingam_lr(ua, ub)
    if rule == "llcca":
        # multi-lag pairwise lead-lag asymmetry on the raw series (max_lag=1 window)
        X = np.column_stack([np.asarray(xa), np.asarray(xb)])
        return _llcca_asymmetry(X, 0, 1, 1)
    if rule == "bivar":
        # idea-008 ALONE: one joint VAR(1) fit, no bootstrap, no fusion -- the rung the
        # orientation audit recommended headlining (no n_boot/block_len/fuse constants).
        return _bivar_var1_asymmetry(xa, xb)
    if rule == "boot":
        # idea-014: blockboot median of idea-008, still WITHOUT the 0.5*cross_lag fusion.
        # Isolates what the fusion term (idea-015) is worth end-to-end.
        return _bivar_var1_blockboot_median(xa, xb)
    if rule != "leadlag":
        raise ValueError(f"unknown orientation rule {rule!r}")
    return _bivar_var1_blockboot_median(xa, xb) + 0.5 * _cross_lag_score(xa, xb)


def sl_lag0_recover(
    df_obs,
    g_est,
    max_lag,
    k=None,
    alpha=0.05,
    n_boot=200,
    seed=0,
    orient=True,
    ktrim_margin=0,
    threshold_rule="maxt",
    block_len=20,
    cov_estimator=_PC_COV_ESTIMATOR,
    gate_tau=_GATE_TAU,
    presvd_c=None,
    var_order=None,
    orient_rule=None,
):
    """S-L lag-0 recovery: replace the lag-0 slice of ``g_est`` with the low-rank-deconfounded
    precision skeleton, thresholded by the edge-free-null max-T rule (do-no-harm by construction,
    self-calibrating via ``alpha`` -- no magic amplitude constant). lag>=1 is returned untouched.

    Orientation: where the deconfounded residuals are non-Gaussian, orient each edge by pairwise
    LiNGAM; where ~Gaussian (unorientable), keep it undirected (both directions). Skeleton recovery
    is the principled part; lag-0 orientation is identifiability-limited to the non-Gaussian case.

    ``ktrim_margin`` (research, default 0 = original behavior): see `_sl_deconf_pcorr`'s
    docstring. Ported from the profile="sl" full-replacement research loop's idea-020. TESTED
    (2026-08-10) on this lag-0-only engine and found to HURT monotonically (margin=1/2
    degrade lag-0 F1, margin=2 collapses it) -- the fix is specific to the full multilag
    joint-stack construction, does NOT transfer here. Left at default 0; do not change.
    ``threshold_rule`` : "maxt" (default, original behavior) or "blockboot" (research,
    ported from idea-004: stationary block bootstrap instead of a single circular shift per
    idiosyncratic column -- see `_sl_blockboot_threshold`). See SL_VALIDATION.md for results.
    """
    X = df_obs.values
    # VAR order for residualization = the EFFECTIVE lag the base discovery actually found
    # (highest occupied lag slice of g_est), capped by max_lag, floored at 1 -- NOT the nominal
    # max_lag. Trusting the discovery avoids over-residualization when the user sets max_lag much
    # higher than the true order: a too-large VAR(p) regresses on many useless d x d lag blocks,
    # overfits at finite T, and DEGRADES S-L (measured: true-lag-1 data, p=6 -> AUPR 0.93->0.82).
    # Residualizing to the effective order removes the real lag 1..p structure and fully recovers
    # (true-lag-3 -> 0.91) while staying do-no-harm when the effective order is 1. If discovery
    # slightly over-finds a lag, we over-residualize by ~1 (small, gradual cost). var_order="auto"
    # (whiteness) / an integer override remain available.
    if var_order is None:
        occ = [lag for lag in range(1, g_est.shape[2]) if np.any(g_est[:, :, lag])]
        vo = min(max(occ) if occ else 1, max(int(max_lag), 1))
    else:
        vo = var_order
    if k is None:
        k = max(mp_factor_count(X, var_order=vo), 1)
    pc, U, svd, Udec = _sl_deconf_pcorr(
        X,
        k,
        ktrim_margin=ktrim_margin,
        cov_estimator=cov_estimator,
        gate_tau=gate_tau,
        presvd_c=presvd_c,
        var_order=vo,
    )
    if threshold_rule == "blockboot":
        thr = _sl_blockboot_threshold(
            U,
            svd,
            k,
            alpha=alpha,
            n_boot=n_boot,
            seed=seed,
            ktrim_margin=ktrim_margin,
            block_len=block_len,
            cov_estimator=cov_estimator,
            gate_tau=gate_tau,
            presvd_c=presvd_c,
        )
    elif threshold_rule == "maxt":
        thr = _sl_maxt_threshold(
            U,
            svd,
            k,
            alpha=alpha,
            n_boot=n_boot,
            seed=seed,
            ktrim_margin=ktrim_margin,
            cov_estimator=cov_estimator,
            gate_tau=gate_tau,
            presvd_c=presvd_c,
        )
    else:
        raise ValueError(
            f"unknown threshold_rule {threshold_rule!r} (use 'maxt' or 'blockboot')"
        )
    g_out = g_est.copy()
    g_out[:, :, 0] = 0
    for a, b in zip(*np.where(pc > thr)):
        if a >= b:
            continue  # each undirected pair once
        if orient:
            lr = orient_lag0_pair(
                X[:, a], X[:, b], Udec[:, a], Udec[:, b], rule=orient_rule
            )
            if abs(lr) < 1e-6:  # Gaussian / unorientable -> keep undirected
                g_out[a, b, 0] = 1
                g_out[b, a, 0] = 1
            elif lr > 0:
                g_out[a, b, 0] = 1
            else:
                g_out[b, a, 0] = 1
        else:
            g_out[a, b, 0] = 1
            g_out[b, a, 0] = 1
    return g_out


def _sl_rel50_threshold(pc0, lag_pc, max_lag, floor=0.12):
    """Relative-to-max threshold, generalized per lag block: keep |pcorr| > 0.5*max(|pcorr|)
    within that block, with an absolute floor (do-no-harm when the block's max is below the
    floor -> no edges). Same rule validated in the lag-0-only research (F16/F18) to beat
    max-T on raw F1 at the cost of being empirically (not by-construction) do-no-harm;
    applied here per lag block for the same reason `_sl_multilag_maxt_threshold` is
    per-block, not global."""

    def _thr(block_vals):
        m = block_vals.max() if block_vals.size else 0.0
        return 0.5 * m if m > floor else float("inf")

    d = pc0.shape[0]
    iu = np.triu_indices(d, 1)
    thr = {"lag0": _thr(pc0[iu])}
    off_diag = ~np.eye(d, dtype=bool)
    for ell in range(1, max_lag + 1):
        thr[ell] = _thr(lag_pc[ell][off_diag])
    return thr


def factor_pds_filter(df_obs, g_est, max_lag, alpha=0.01, k=None):
    """Post-double-selection Granger filter that forces the MP-estimated pervasive
    factor score(s) into the control set, instead of lasso-selecting other lagged
    observed variables (as `pds_filter` does). The lasso-selected controls are
    themselves correlated with the target only through the SAME shared latent factor,
    so including them is redundant with -- and numerically unstable relative to --
    conditioning on the factor directly; that is the documented cause of PDS over-
    pruning true edges on homoskedastic pervasive data. Conditioning on an explicit
    factor proxy isolates genuine lagged direct effects without that collinearity."""
    X = df_obs.values
    d = X.shape[1]
    if k is None:
        k = max(mp_factor_count(X), 1)
    scores = _pca_factor_scores(X, k)
    if scores is None:
        return g_est
    Z_full, names_full, rows = _lagged_design(X, max_lag)
    Y = X[max_lag:]
    idx_map = {key: c for c, key in enumerate(names_full)}
    start = max_lag - 1  # scores row r aligns to X[r+1]; Y row m aligns to X[max_lag+m]
    if start < 0 or start + rows > scores.shape[0]:
        return g_est
    S = scores[start : start + rows]
    g_out = g_est.copy()
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            for lag in range(max_lag + 1):
                if g_out[i, j, lag] == 0:
                    continue
                y = Y[:, j]
                if lag == 0:
                    x_edge = X[max_lag:, i]
                else:
                    key = (i, lag)
                    if key not in idx_map:
                        continue
                    x_edge = Z_full[:, idx_map[key]]
                Xd = np.column_stack([x_edge, S])
                try:
                    model = sm.OLS(y, sm.add_constant(Xd)).fit(
                        cov_type="HAC", cov_kwds={"maxlags": _HAC_LAGS}
                    )
                    pval = model.pvalues[1]
                except Exception:
                    continue  # abstain (keep) on numerical failure
                if not np.isnan(pval) and pval >= alpha:
                    g_out[i, j, lag] = 0
    return g_out


def _discover(
    df_obs,
    ci,
    max_lag,
    tetrad,
    keep_undirected=False,
    alpha=0.05,
    tetrad_threshold=_TETRAD_THRESHOLD,
):
    """Run the base skeleton engine, optionally followed by the tetrad lag-0 filter.

    The tetrad step is applied *after* discovery rather than inside the engine. That is
    bit-identical to the old ``run_cdnots_plus(deconf_threshold=...)`` path -- which
    itself filtered the finished graph -- and it keeps discovery independent of the
    routed regime, so an already-discovered graph can be reused (see ``run_lucid``'s
    ``discovery=`` argument).
    """
    if tetrad and keep_undirected:
        # `keep_undirected` is only meaningful for the CD-NOTS+ orientation tail, and
        # this build's `run_cdnots_plus` does not expose it. Nothing combines the two
        # (the lag-0 ablation's keep_all arm runs on pervasive_base="naive", where the
        # flag is a no-op), so fail loudly rather than silently ignore the request.
        raise NotImplementedError(
            "keep_undirected=True is not supported with pervasive_base='tetrad'; "
            "use the default pervasive_base='naive'."
        )
    fn = run_cdnots_plus if tetrad else run_cdnots
    kw = dict(
        num_lags=max_lag, include_C=True, c_preset="linear", alpha=alpha, verbose=False
    )
    result = fn(df_obs, ci, **kw)
    cg = result.cg_tig
    if tetrad:
        cg = apply_tetrad_lag0_filter(
            cg, df_obs, list(df_obs.columns), threshold=tetrad_threshold
        )
    d = df_obs.shape[1]
    return np.asarray(cg[:d, :d, : max_lag + 1])


def deconfound(
    graph,
    df_obs,
    max_lag,
    regime,
    alpha_pds_sparse=_ALPHA_PDS_SPARSE,
    alpha_pds_pervasive=_ALPHA_PDS_PERVASIVE,
    alpha_vol=_ALPHA_VOL,
    llcca_threshold=_LLCCA_THRESHOLD,
):
    """Post-hoc layer: apply the regime-appropriate edge filters to an already
    discovered `graph` (d,d,max_lag+1). Works with any base discovery method."""
    if regime == "sparse":
        return pds_filter(df_obs, graph, max_lag, alpha=alpha_pds_sparse)
    g = pds_filter(df_obs, graph, max_lag, alpha=alpha_pds_pervasive)
    if regime == "sf":
        return llcca_filter(df_obs, g, max_lag, threshold=llcca_threshold)
    if regime == "pervasive":
        return vol_invariance_filter(df_obs, g, max_lag, alpha=alpha_vol)
    raise ValueError(f"unknown regime {regime!r}")


def routed_deconfound(
    df_obs,
    max_lag,
    ci=None,
    router="auto",
    thresholds=None,
    profile="adaptive",
    alpha_factor_pds=_ALPHA_FACTOR_PDS,
    arch_alpha=_ARCH_ALPHA,
    drop_lag0=True,
    keep_undirected=False,
    recover_lag0=True,
    lag0_engine="sl",
    alpha_sl=0.05,
    n_boot_sl=200,
    orient_lag0=True,
    seed_sl=0,
    pervasive_base="naive",
    pervasive_filters=False,
    alpha_ci=0.05,
    base_discovery=None,
    router_k=_ROUTER_K,
    gamma=_TAU_MARGIN,
    factor_count_margin=_FACTOR_COUNT_MARGIN,
    tetrad_threshold=_TETRAD_THRESHOLD,
    return_info=False,
    **filter_kw,
):
    """Regime-adaptive causal discovery under latent confounders.

    Routes the dataset to a regime (data-only via the MP-null spectral router). By
    default (the shipped LUCID configuration) both regimes run the SAME base skeleton
    engine (plain CD-NOTS); the regime affects only the correction applied afterward
    -- the sparse branch gets a PDS edge filter, the pervasive branch gets S-L lag-0
    recovery. Passing ``pervasive_base="tetrad"`` switches the pervasive branch to a
    different base engine (CD-NOTS+ with a tetrad deconfounding filter) instead; this
    is a research configuration, not the shipped default.

    profile="adaptive" (DEFAULT):
      sparse                          -> CD-NOTS + PDS-Granger.
      pervasive, volatility clustering present (heteroskedastic latent factor):
        scale-free sub-regime -> PDS + lead-lag-asymmetry (LLCCA);
        pervasive  sub-regime -> volatility-invariance.
      pervasive, NO volatility clustering (homoskedastic latent factor):
        lead-lag-asymmetry + factor-augmented PDS (conditions on the PCA-estimated
        factor itself, not on other equally-confounded observed vars); then drop
        undirectable lag-0 edges (a shared same-period confounder's footprint).
      Gating the volatility-based filters on actual heteroskedasticity keeps the
      strong heteroskedastic-factor performance without over-pruning true edges on
      homoskedastic factors. Fully parameter-free (conventional 0.05 levels).

    profile="unconditional" applies the full deconfounding stack on every pervasive
      route without the heteroskedasticity gate — stronger on heteroskedastic
      (volatility-clustering) confounding, weaker on homoskedastic; provided for
      users who know their confounding is volatility-clustering, and for ablations.

    Parameters
    ----------
    df_obs : DataFrame (T, d) of observed variables only.
    max_lag : int.
    ci : optional CI test; defaults to ParCorrGPU(df_obs.values).
    router : "auto" (default; MP-null sparse boundary + robust sub-regime split),
        "spectral" (fixed calibrated thresholds), or "mp" (factor count; ablation).
    thresholds : optional (tau, tau2); with router="auto" only tau2 is used (expert).
    profile : "adaptive" (default) or "unconditional".
    alpha_factor_pds, arch_alpha : factor-PDS significance level and ARCH-gate level.
    drop_lag0 : if True (default, shipped behavior), zero out lag-0 edges on the
        pervasive route (Section on undirectable contemporaneous confounder
        footprint); set False for the T0.2 lag-0-contamination ablation.
    keep_undirected : if True, the pervasive-route CDNOTS+ base engine keeps
        unresolved (o-o) edges as bidirected instead of dropping them (default
        False, matching CDNOTS+'s and PCMCI+'s shipped convention). Only
        relevant with `drop_lag0=False`, since `drop_lag0=True` zeroes lag-0
        either way.
    recover_lag0 : if True (default, shipped behavior), reconstruct the pervasive
        branch's lag-0 slice via S-L (`sl_lag0_recover`) instead of blanket-dropping
        it. If False, `drop_lag0` decides whether the lag-0 slice is dropped or kept
        as discovered.
    pervasive_base : "naive" (default, shipped) uses plain CD-NOTS -- the SAME base
        engine as the sparse branch -- so both branches share one discovery method,
        differing only in which correction runs downstream. "tetrad" (research
        configuration) switches the pervasive branch to CD-NOTS+ with a tetrad
        deconfounding filter as its base skeleton engine instead.
    return_info : if True, also return the routing/gate diagnostics dict.
    **filter_kw : optional alpha/threshold overrides passed to `deconfound`.

    Returns
    -------
    graph : ndarray (d, d, max_lag+1) in [cause, effect, lag]. (graph, info) if return_info.
    """
    X = df_obs.values
    if ci is None:
        ci = ParCorrGPU(X)
    regime, info = _route(
        X,
        router=router,
        thresholds=thresholds,
        router_k=router_k,
        gamma=gamma,
        factor_count_margin=factor_count_margin,
    )
    is_sparse = regime == "sparse"

    use_tetrad = (not is_sparse) and pervasive_base == "tetrad"
    graph = (
        base_discovery(df_obs, max_lag)
        if base_discovery is not None
        else _discover(
            df_obs,
            ci,
            max_lag,
            tetrad=use_tetrad,
            keep_undirected=keep_undirected,
            alpha=alpha_ci,
            tetrad_threshold=tetrad_threshold,
        )
    )

    if profile == "unconditional":
        out = deconfound(graph, df_obs, max_lag, regime, **filter_kw)
        return (out, {**info, "profile": "unconditional"}) if return_info else out
    if profile != "adaptive":
        raise ValueError(
            f"unknown profile {profile!r} (use 'adaptive' or 'unconditional')"
        )

    if is_sparse:
        out = pds_filter(
            df_obs,
            graph,
            max_lag,
            alpha=filter_kw.get("alpha_pds_sparse", _ALPHA_PDS_SPARSE),
        )
        info = {**info, "profile": "adaptive", "pervasive_filters": None}
        return (out, info) if return_info else out

    if not pervasive_filters:
        # No pervasive lag>=1 filtering (ARCH gate is moot with nothing to gate). The
        # benchmark ablation (`run_filter_ablation.py`) showed the vol-invariance / LLCCA /
        # factor-PDS filters are net-harmful on the unified naive base -- they over-prune,
        # increasingly so at higher lags -- so the shipped pipeline omits them and relies on
        # the router + S-L lag-0 recovery. The base skeleton passes through unfiltered.
        out = graph
        p_arch, applied = None, "none"
    else:
        p_arch = _vol_clustering_pvalue(X)
        clustered = p_arch is not None and p_arch < arch_alpha
        if clustered and regime == "sf":
            g = pds_filter(df_obs, graph, max_lag, alpha=_ALPHA_PDS_PERVASIVE)
            out = llcca_filter(df_obs, g, max_lag, threshold=_LLCCA_THRESHOLD)
            applied = "pds_llcca"
        elif clustered:  # pervasive sub-regime
            out = vol_invariance_filter(df_obs, graph, max_lag, alpha=_ALPHA_VOL)
            applied = "vol_invariance"
        else:  # homoskedastic pervasive factor
            g = llcca_filter(df_obs, graph, max_lag, threshold=_LLCCA_THRESHOLD)
            out = factor_pds_filter(df_obs, g, max_lag, alpha=alpha_factor_pds)
            applied = "llcca_factorpds"
    if recover_lag0:
        # Lag-0 recovery: low-rank-deconfounded precision (S-L) skeleton plus an
        # edge-free-null max-T threshold (do-no-harm by construction; generalises to
        # dense pervasive confounding). This is the only supported engine.
        if lag0_engine != "sl":
            raise ValueError(
                f"unknown lag0_engine {lag0_engine!r} (only 'sl' is supported)"
            )
        out = sl_lag0_recover(
            df_obs,
            out,
            max_lag,
            alpha=alpha_sl,
            n_boot=n_boot_sl,
            seed=seed_sl,
            orient=orient_lag0,
            ktrim_margin=filter_kw.get("sl_ktrim_margin", 0),
            threshold_rule=filter_kw.get("sl_lag0_threshold_rule", "maxt"),
            block_len=filter_kw.get("sl_lag0_block_len", 20),
            cov_estimator=filter_kw.get("sl_cov_estimator", "sample"),
            # persistence (moralization) gate width; default _GATE_TAU
            # preserves shipped behaviour exactly. Exposed so the
            # constant can be swept (run_constant_sensitivity.py).
            gate_tau=filter_kw.get("sl_gate_tau", _GATE_TAU),
            # lag-0 orientation statistic; None = shipped _ORIENT_RULE ("llcca")
            orient_rule=filter_kw.get("sl_orient_rule"),
        )
    elif drop_lag0:
        out = _drop_instantaneous(out)
    info = {
        **info,
        "profile": "adaptive",
        "vol_clustering_p": p_arch,
        "pervasive_filters": applied,
    }
    return (out, info) if return_info else out


def routed_deconfound_naive_sl(df_obs, max_lag, **kw):
    """Unified-base-engine variant: pervasive branch uses PLAIN CD-NOTS (no tetrad) --
    the SAME discovery method as the sparse branch -- keeps the shipped ARCH-gate/
    LLCCA/vol-invariance/factor-PDS filter stack for lag>=1, and replaces the blanket
    lag-0 drop with the already-validated lag-0-only S-L recovery (`sl_lag0_recover`,
    F13/F14: max-T, alpha=0.05, n_boot=200 -- stable at this scope since lag-0-only has
    far fewer tested pairs than the failed full-multilag-replacement attempt).

    Research question (2026-08-09, `experiments/confounders/SL_VALIDATION.md`): with a
    single shared base engine across both branches, does S-L lag-0 recovery alone (not a
    full pervasive-branch replacement) match/beat the shipped CDNOTS+/tetrad pervasive
    base engine on the paper's OOD suite?

    Reproduce::

        python run_ood_benchmark.py \\
            --method causalts.confounders.routed_deconf:routed_deconfound_naive_sl \\
            --worktree /Users/mfesanghary1/workspace/causal-ts-lag0 --seeds 0 1 2
    """
    return routed_deconfound(
        df_obs,
        max_lag,
        profile="adaptive",
        pervasive_base="naive",
        recover_lag0=True,
        lag0_engine="sl",
        **kw,
    )


def routed_deconfound_lucid(df_obs, max_lag, **kw):
    """The shipped LUCID pipeline (2026-08-12): unified plain-CD-NOTS base on both branches,
    S-L lag-0 recovery, and NO pervasive lag>=1 filters. The benchmark filter ablation
    (`experiments/confounders/run_filter_ablation.py`, 20 seeds + higher-lag check) showed the
    vol-invariance/LLCCA/factor-PDS filters are net-harmful on the unified base -- they
    over-prune, increasingly at higher lags -- so LUCID omits them (and the ARCH gate they
    were gated on). This is `routed_deconfound_naive_sl` with `pervasive_filters=False`.
    """
    return routed_deconfound(
        df_obs,
        max_lag,
        profile="adaptive",
        pervasive_base="naive",
        recover_lag0=True,
        lag0_engine="sl",
        pervasive_filters=False,
        **kw,
    )


def _factor_diagnostics(X, factor_count_margin=_FACTOR_COUNT_MARGIN):
    """(n_factors, loadings) from the VAR residual spectrum.

    ``loadings`` are the leading right singular vectors of the centred residuals, one
    row per detected factor. Descriptive only -- factors are identified up to rotation
    (see :class:`~causalts.confounders.result.LucidResult`).
    """
    try:
        k = mp_factor_count(X, factor_count_margin=factor_count_margin)
        if k <= 0:
            return 0, None
        U = _var_residuals(X, 1)
        U = U - U.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(U, full_matrices=False)
        return int(k), np.asarray(Vt[:k])
    except Exception:  # diagnostics must never break a discovery run
        return None, None


def _resolve_discovery(discovery, max_lag, d):
    """Turn ``discovery=`` into a ``base_discovery`` callable, validating what we can.

    Accepts a :class:`~causalts.result.CausalResult` (any subclass), a raw
    ``(d, d, L+1)`` array, a callable ``(df, max_lag) -> graph``, or ``None``.

    A result's ``cg_tig`` carries extra C-node rows/columns when discovery ran with
    ``include_C=True``, so it is sliced to the observed block -- exactly what
    ``_discover`` returns.
    """
    import warnings as _warnings

    if discovery is None or callable(discovery):
        return discovery

    graph = discovery
    if hasattr(discovery, "cg_tig"):  # a CausalResult subclass
        graph = discovery.cg_tig
        rec_lags = getattr(discovery, "num_lags", None)
        if rec_lags is not None and int(rec_lags) != int(max_lag):
            raise ValueError(
                f"discovery result was built with num_lags={rec_lags}, but max_lag="
                f"{max_lag} was requested; re-run discovery at the same lag horizon."
            )
        rec_alpha = getattr(discovery, "alpha", None)
        if rec_alpha is not None and not np.isclose(rec_alpha, 0.05):
            _warnings.warn(
                f"reusing a discovery result built with alpha={rec_alpha}; LUCID's own "
                "skeleton uses alpha=0.05, so results may differ from run_lucid(df, ...)",
                stacklevel=3,
            )
        if getattr(discovery, "include_C", True) is False:
            _warnings.warn(
                "reusing a discovery result built with include_C=False; LUCID's own "
                "skeleton sets include_C=True, so results may differ from "
                "run_lucid(df, ...)",
                stacklevel=3,
            )
    graph = np.asarray(graph)[:d, :d, : max_lag + 1]
    return lambda _df, _ml, _g=graph: _g


def run_lucid(df_obs, max_lag, ci=None, discovery=None, **kwargs):
    """Run LUCID and return a :class:`~causalts.confounders.result.LucidResult`.

    LUCID infers the latent-confounding regime from the data with a Marchenko-Pastur
    spectral router, then applies the deconfounding strategy matched to that regime:
    a post-double-selection edge filter on the sparse branch, and spectral (S-L)
    recovery of the contemporaneous slice on the pervasive branch.

    Parameters
    ----------
    df_obs : DataFrame (T, d)
        Observed time series.
    max_lag : int
        Maximum lag to consider.
    ci : CIT_Base, optional
        Conditional-independence test for the base skeleton search. Defaults to
        :class:`ParCorrGPU` on ``df_obs``.
    discovery : CausalResult | ndarray | callable, optional
        Skip LUCID's own skeleton search and reuse an existing discovery. Because both
        branches run the *same* base engine, reuse is exact -- the routing decision does
        not change what is discovered. Pass a result object (``CdnotsResult``,
        ``CedarResult``, ``GraceResult``, ...), a raw ``(d, d, max_lag+1)`` array, or a
        ``(df, max_lag) -> graph`` callable.

        Reuse assumes the graph was discovered with LUCID-compatible settings. A
        ``num_lags`` mismatch raises; differing ``alpha``/``include_C`` warn. ``c_preset``
        is not recorded on result objects and therefore cannot be checked.
    **kwargs
        Forwarded to :func:`routed_deconfound` (``router``, ``router_k``, ``gamma``,
        ``alpha_ci``, ``tetrad_threshold``, ...).

    Returns
    -------
    LucidResult

    Examples
    --------
    >>> res = run_lucid(df, max_lag=2)              # doctest: +SKIP
    >>> res.regime, res.n_factors                   # doctest: +SKIP
    ('pervasive', 3)
    >>> res.plot()                                  # doctest: +SKIP
    """
    import time as _time

    t0 = _time.time()
    base_discovery = _resolve_discovery(discovery, max_lag, df_obs.shape[1])
    graph, info = routed_deconfound(
        df_obs,
        max_lag,
        ci=ci,
        base_discovery=base_discovery,
        return_info=True,
        **kwargs,
    )
    X = df_obs.values if hasattr(df_obs, "values") else np.asarray(df_obs)
    n_factors, loadings = _factor_diagnostics(
        X, factor_count_margin=kwargs.get("factor_count_margin", _FACTOR_COUNT_MARGIN)
    )
    return LucidResult(
        np.asarray(graph),
        df_obs,
        list(df_obs.columns),
        info=info,
        n_factors=n_factors,
        factor_loadings=loadings,
        runtime=_time.time() - t0,
    )
