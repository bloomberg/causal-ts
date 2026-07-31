# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_reset


def check_linearity(data, alpha=0.05, max_lag=1, pairs=None):
    """Test pairwise linearity using the Ramsey RESET test.

    Guides CI test selection by checking whether relationships are linear.

    The regression always conditions on Y_j(t-1) to absorb the autocorrelation
    that is inherent in any time series.  Without this control, simple bivariate
    regressions on VAR data have structured (autocorrelated) residuals that the
    RESET test misinterprets as nonlinearity — inflating the false positive rate
    well above the nominal alpha.

    Parameters
    ----------
    data : numpy.ndarray or pandas.DataFrame
        Array of shape (T, N) with T observations and N variables.
    alpha : float, optional
        Significance level for rejecting linearity (default 0.05).
    max_lag : int, optional
        Maximum lag to test for X_i(t-lag) → Y_j(t) relationships.
        Default is 1. Set to 0 to test only contemporaneous pairs.
    pairs : list of tuple, optional
        List of (i, j) tuples specifying which directed pairs to test.
        Default: all ordered pairs (i, j) where i != j.

    Returns
    -------
    dict
        Dict with keys:

        - results: list of dicts with x, y, lag, reset_pvalue, is_nonlinear
        - fraction_nonlinear: fraction of tests rejecting linearity
        - summary: human-readable recommendation string
    """
    if hasattr(data, "values"):
        data = data.values
    data = np.asarray(data, dtype=np.float64)
    T, N = data.shape

    if pairs is None:
        pairs = [(i, j) for i in range(N) for j in range(N) if i != j]

    lags = list(range(0, max_lag + 1))
    results = []

    for i, j in pairs:
        for lag in lags:
            # Align X_i(t-lag) with Y_j(t), always keeping Y_j(t-1) as a control.
            # Using t=1..T-lag so that Y_j(t-1) is always available.
            start = max(lag, 1)
            x = data[start - lag : T - lag, i]  # X_i(t-lag), t = start..T-1
            y = data[start:, j]  # Y_j(t),     t = start..T-1
            y_lag1 = data[start - 1 : T - 1, j]  # Y_j(t-1),   t = start..T-1

            if len(x) < 10:
                continue

            X = sm.add_constant(np.column_stack([x, y_lag1]))
            try:
                model = sm.OLS(y, X).fit()
                reset_result = linear_reset(model, power=3, use_f=True)
                pval = reset_result.pvalue
            except Exception:
                pval = np.nan

            results.append(
                {
                    "x": i,
                    "y": j,
                    "lag": lag,
                    "reset_pvalue": pval,
                    "is_nonlinear": pval < alpha if np.isfinite(pval) else False,
                }
            )

    n_nonlinear = sum(r["is_nonlinear"] for r in results)
    n_total = len(results)
    fraction = n_nonlinear / n_total if n_total > 0 else 0.0

    if fraction > 0.2:
        summary = (
            f"{n_nonlinear}/{n_total} pairs show significant nonlinearity "
            f"(alpha={alpha}). Consider a nonlinear CI test (splitkci, dfcit)."
        )
    elif n_nonlinear > 0:
        summary = (
            f"{n_nonlinear}/{n_total} pairs show significant nonlinearity "
            f"(alpha={alpha}). Mostly linear — parcorr-gpu likely sufficient, "
            f"but check flagged pairs."
        )
    else:
        summary = (
            "No significant nonlinearity detected. " "parcorr-gpu should work well."
        )

    return {
        "results": results,
        "fraction_nonlinear": fraction,
        "summary": summary,
    }
