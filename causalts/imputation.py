# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

import warnings

import numpy as np
import pandas as pd


def _initial_fill(df: pd.DataFrame) -> pd.DataFrame:
    filled = df.copy()
    col_means = df.mean()
    for col in df.columns:
        mask = filled[col].isna()
        if mask.any():
            filled.loc[mask, col] = col_means[col]
    return filled


def var_em_impute(
    df: pd.DataFrame,
    p: int = 1,
    max_iter: int = 50,
    tol: float = 1e-4,
    verbose: bool = False,
) -> pd.DataFrame:
    """Impute missing values via EM with a VAR(p) model.

    Uses the multivariate temporal structure symmetrically (no causal direction assumed).
    Non-NaN positions are always restored exactly after each EM step.
    """
    nan_mask = df.isna()
    if not nan_mask.any().any():
        return df.copy()

    T, K = df.shape
    if T - p < 2 * K:
        warnings.warn(
            f"var_em_impute: too few observations (T={T}, p={p}, K={K}) for VAR fitting; "
            "falling back to column-mean fill.",
            UserWarning,
            stacklevel=2,
        )
        return _initial_fill(df)

    from statsmodels.tsa.vector_ar.var_model import VAR  # lazy import

    nan_mask_arr = nan_mask.values
    original_vals = df.values.copy()

    df_filled = _initial_fill(df)
    filled_arr = df_filled.values.copy()

    prev_imputed = filled_arr[nan_mask_arr].copy()

    for iteration in range(max_iter):
        try:
            results = VAR(df_filled).fit(maxlags=p, ic=None, trend="c", verbose=False)
        except np.linalg.LinAlgError:
            warnings.warn(
                "var_em_impute: VAR fitting failed (singular matrix); stopping early.",
                UserWarning,
                stacklevel=2,
            )
            break

        fv = results.fittedvalues.values  # shape (T-p, K)
        # fv[i] corresponds to original row i+p
        for t_fv in range(fv.shape[0]):
            t_orig = t_fv + p
            for k in range(K):
                if nan_mask_arr[t_orig, k]:
                    filled_arr[t_orig, k] = fv[t_fv, k]

        # restore non-NaN positions
        filled_arr[~nan_mask_arr] = original_vals[~nan_mask_arr]
        df_filled = pd.DataFrame(filled_arr, columns=df.columns, index=df.index)

        new_imputed = filled_arr[nan_mask_arr]
        delta = np.max(np.abs(new_imputed - prev_imputed))
        if verbose:
            print(f"  var_em iter {iteration + 1}: delta={delta:.6f}")
        if delta < tol:
            break
        prev_imputed = new_imputed.copy()

    return df_filled


def _markov_blanket_from_tigramite(
    cg_tig: np.ndarray,
    var_idx: int,
    num_vars: int,
    lags=None,
) -> set[tuple[int, int]]:
    """Extract the time-unrolled Markov blanket from a tigramite graph array.

    cg_tig[i, j, tau] == 1 means variable i at lag tau causes variable j at lag 0.
    For the target ``k`` at time ``t``, returns ``(variable, shift)`` pairs where
    the regressor is ``variable`` at time ``t + shift``:

    - parents ``i(t - tau)``, including the target's own lags;
    - children ``j(t + tau)`` for every edge ``k(t - tau) -> j(t)``;
    - the children's other parents ``i(t + tau_child - tau_parent)``.

    Only the unshifted target ``(k, 0)`` is left out. Indices are restricted to
    range(num_vars) (excludes C if present). ``lags`` maps each lag slot of
    ``cg_tig`` to its lag (``[0] + lag_list``); the default reads slot ``tau`` as
    lag ``tau``.
    """
    L = cg_tig.shape[2] - 1
    k = var_idx
    lags = list(range(L + 1)) if lags is None else list(lags)

    def _parents(j):
        return [
            (i, lags[tau])
            for i in range(num_vars)
            for tau in range(L + 1)
            if cg_tig[i, j, tau] == 1 and not (i == j and tau == 0)
        ]

    blanket = {(i, -tau) for i, tau in _parents(k)}
    for j in range(num_vars):
        for tau_child in range(L + 1):
            if cg_tig[k, j, tau_child] != 1 or (j == k and tau_child == 0):
                continue
            lag_child = lags[tau_child]
            blanket.add((j, lag_child))
            blanket.update((i, lag_child - lag_parent) for i, lag_parent in _parents(j))

    blanket.discard((k, 0))
    return blanket


def _shifted_design(filled_arr: np.ndarray, blanket) -> np.ndarray:
    """Column per ``(variable, shift)``: value at ``t + shift``, NaN past the ends."""
    T = filled_arr.shape[0]
    X = np.full((T, len(blanket)), np.nan)
    for col, (var, shift) in enumerate(blanket):
        if shift >= 0:
            X[: T - shift, col] = filled_arr[shift:, var]
        else:
            X[-shift:, col] = filled_arr[: T + shift, var]
    return X


def iterative_causal_impute(
    df: pd.DataFrame,
    indep_test,
    num_lags: int = 1,
    lag_list=None,
    include_C: bool = True,
    alpha: float = 0.05,
    n_iter: int = 3,
    tol: float = 1e-4,
    var_em_kwargs: dict = None,
    cdnots_kwargs: dict = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Iterative causal imputation using the CDNOTS-discovered Markov blanket.

    Warm-starts with VAR-EM, then alternates between discovering the causal graph
    and re-imputing each variable using only its Markov blanket as predictors.
    Each blanket member enters at its own time offset (a parent ``X(t-1)`` is
    regressed on as ``X`` shifted by one row), the regression is fitted on rows
    where the target is observed, and a variable with an empty blanket (or a row
    whose regressors fall past the series ends) keeps its current values, the VAR-EM
    warm start on the first pass.
    This restricts imputation to causally relevant variables, reducing the risk of
    injecting spurious cross-variable dependencies.
    """
    nan_mask = df.isna()
    if not nan_mask.any().any():
        return df.copy()

    from sklearn.linear_model import Ridge  # lazy import

    nan_mask_arr = nan_mask.values
    original_vals = df.values.copy()
    num_vars = df.shape[1]

    df_filled = var_em_impute(df, **(var_em_kwargs or {}))
    filled_arr = df_filled.values.copy()
    prev_imputed = filled_arr[nan_mask_arr].copy()

    # cg_tig lag slots are lags only for lag_list=[1..L]
    lags = None if lag_list is None else [0] + list(lag_list)

    for iteration in range(n_iter):
        if verbose:
            print(f"Causal imputation iteration {iteration + 1}/{n_iter}")

        indep_test.pvalue_cache.clear()
        indep_test.pvalue_local.clear()

        # lazy import to avoid circular dependency
        from causalts.cdnots.phase3_utils import run_cdnots

        cdnots_res = run_cdnots(
            df=df_filled,
            indep_test=indep_test,
            num_lags=num_lags,
            lag_list=lag_list,
            include_C=include_C,
            alpha=alpha,
            show_progress=False,
            verbose=False,
            **(cdnots_kwargs or {}),
        )

        for k in range(num_vars):
            if not nan_mask_arr[:, k].any():
                continue
            mb = _markov_blanket_from_tigramite(cdnots_res.cg_tig, k, num_vars, lags)
            if not mb:
                # Nothing causally relevant: keep the current values.
                continue
            X_all = _shifted_design(filled_arr, sorted(mb))
            defined = ~np.isnan(X_all).any(axis=1)
            fit_rows = defined & ~nan_mask_arr[:, k]
            if not fit_rows.any():
                continue
            model = Ridge(alpha=1e-3).fit(X_all[fit_rows], filled_arr[fit_rows, k])
            nan_rows = nan_mask_arr[:, k] & defined
            if nan_rows.any():
                filled_arr[nan_rows, k] = model.predict(X_all[nan_rows])

        filled_arr[~nan_mask_arr] = original_vals[~nan_mask_arr]
        df_filled = pd.DataFrame(filled_arr, columns=df.columns, index=df.index)

        new_imputed = filled_arr[nan_mask_arr]
        delta = np.max(np.abs(new_imputed - prev_imputed))
        if verbose:
            print(f"  delta={delta:.6f}")
        if delta < tol:
            break
        prev_imputed = new_imputed.copy()

    return df_filled
