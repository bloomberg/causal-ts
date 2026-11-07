# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np


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
        - fraction_nonlinear: fraction of tests rejecting linearity, among those
          that gave a p-value (rows with a missing value are dropped per test);
          0.0 when none did, so check n_tested
        - n_tested: number of tests that gave a p-value
        - summary: human-readable recommendation string
    """
    # Imported here rather than at module scope: statsmodels costs ~0.25s to
    # import and this is the only thing in causalts that needs it eagerly.
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import linear_reset

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
            if T - start < 10:  # too few rows; also avoids misaligned slices
                continue
            x = data[start - lag : T - lag, i]  # X_i(t-lag), t = start..T-1
            y = data[start:, j]  # Y_j(t),     t = start..T-1
            y_lag1 = data[start - 1 : T - 1, j]  # Y_j(t-1),   t = start..T-1

            # Drop rows with a missing value; OLS on them gives a NaN p-value.
            ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(y_lag1)
            x, y, y_lag1 = x[ok], y[ok], y_lag1[ok]
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

    # A test that still failed (p is NaN) is untested, not linear: leave it out
    # of the denominator so it cannot dilute the fraction towards zero.
    n_nonlinear = sum(r["is_nonlinear"] for r in results)
    n_total = sum(np.isfinite(r["reset_pvalue"]) for r in results)
    fraction = n_nonlinear / n_total if n_total > 0 else 0.0

    if n_total == 0:
        summary = (
            "Linearity could not be tested: no RESET test gave a p-value (too "
            "few columns or complete rows, or singular regressions)."
        )
    elif fraction > 0.2:
        summary = (
            f"{n_nonlinear}/{n_total} pairs show significant nonlinearity "
            f"(alpha={alpha}). Consider a nonlinear CI test (raft; rcot if the "
            f"dependence may act through the variance)."
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
        "n_tested": int(n_total),
        "summary": summary,
    }
