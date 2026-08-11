# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Observed-data (non)stationarity detection for the ``inspect`` command.

Mirrors the shape of :func:`causalts.utils.linearity.check_linearity`.  Two
functions:

- :func:`check_stationarity_observed` — per-column ADF + KPSS, the *presence*
  of nonstationarity.  (Distinct from
  ``synthetic_data.structural_causal_processes.check_stationarity``, which tests
  the spectral radius of a synthetic-DGP link specification, not observed data.)
- :func:`detect_trend_form` — the *form* of the trend (seasonal / curved /
  trend / none), used to pick a ``c_preset`` for the C node.
"""

from __future__ import annotations

import warnings

import numpy as np


def _to_2d(data):
    if hasattr(data, "values"):
        cols = list(data.columns)
        arr = data.values
    else:
        arr = np.asarray(data)
        cols = [f"X{i}" for i in range(arr.shape[1])]
    return np.asarray(arr, dtype=np.float64), cols


def check_stationarity_observed(data, alpha=0.05):
    """Test each column for a unit root with ADF and KPSS.

    A column is flagged nonstationary when ADF **fails to reject** its unit-root
    null *and* KPSS **rejects** its stationarity null — the agreement of the two
    complementary tests.  Ambiguous columns (tests disagree) are not flagged.

    Parameters
    ----------
    data : numpy.ndarray or pandas.DataFrame
        Array of shape (T, N).
    alpha : float, optional
        Significance level (default 0.05).

    Returns
    -------
    dict with keys ``results`` (list of per-column dicts), ``fraction_nonstationary``,
    ``nonstationary_cols`` (list of column names), and ``summary``.
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    arr, cols = _to_2d(data)
    T, N = arr.shape
    results = []

    for i, col in enumerate(cols):
        x = arr[:, i]
        x = x[np.isfinite(x)]
        adf_p = np.nan
        kpss_p = np.nan
        if len(x) >= 12 and np.std(x) > 0:
            try:
                adf_p = adfuller(x, autolag="AIC")[1]
            except Exception:
                adf_p = np.nan
            try:
                with warnings.catch_warnings():
                    # KPSS emits InterpolationWarning when the stat is outside
                    # the lookup table; the p-value is still usable (clipped).
                    warnings.simplefilter("ignore")
                    kpss_p = kpss(x, regression="c", nlags="auto")[1]
            except Exception:
                kpss_p = np.nan

        adf_unit_root = np.isfinite(adf_p) and adf_p >= alpha
        kpss_nonstat = np.isfinite(kpss_p) and kpss_p < alpha
        is_nonstationary = bool(adf_unit_root and kpss_nonstat)

        results.append(
            {
                "col": col,
                "adf_pvalue": adf_p,
                "kpss_pvalue": kpss_p,
                "is_nonstationary": is_nonstationary,
            }
        )

    nonstationary_cols = [r["col"] for r in results if r["is_nonstationary"]]
    fraction = len(nonstationary_cols) / N if N > 0 else 0.0

    if fraction > 0.3:
        summary = (
            f"{len(nonstationary_cols)}/{N} columns appear nonstationary "
            f"(ADF+KPSS, alpha={alpha}). Use include_C=True (a C node) or "
            f"difference the data."
        )
    elif nonstationary_cols:
        summary = (
            f"{len(nonstationary_cols)}/{N} columns appear nonstationary "
            f"(ADF+KPSS, alpha={alpha}): {nonstationary_cols}. "
            f"Consider include_C=True."
        )
    else:
        summary = "No clear nonstationarity detected (ADF+KPSS)."

    return {
        "results": results,
        "fraction_nonstationary": fraction,
        "nonstationary_cols": nonstationary_cols,
        "summary": summary,
    }


def detect_trend_form(data, seasonal_band_frac=0.25):
    """Classify the dominant nonstationarity *form* across columns.

    Returns one of ``"seasonal"``, ``"curved"``, ``"trend"``, ``"none"`` — a
    single label used to choose a C-node basis: ``seasonal`` → ``linear+sin``,
    ``curved`` → ``linear+quad``, otherwise ``linear``.

    Heuristics (cheap, per-column, majority vote), tested in priority order:

    - **seasonal**: after linear detrending, a single frequency in the
      *seasonal band* (period between 4 samples and T/3, i.e. at least ~3 full
      cycles) carries more than ``seasonal_band_frac`` of the detrended power.
      Restricting to that band is what separates genuine periodicity from two
      look-alikes: white trend-residual spreads power thinly across all bins
      (no bin dominates), and AR / red-noise concentrates power at the *lowest*
      frequencies (period ≈ T, below the band). A plain spike-height threshold
      is unreliable — the max of ~T/2 noise periodogram ordinates routinely sits
      ~8× the median by chance, flipping the verdict with the noise seed.
    - **curved**: a quadratic time fit cuts residual variance by >20% over a
      linear fit (the trend bends).
    - **trend**: a nonzero linear time slope but neither of the above.
    - **none**: no column shows any of the above.
    """
    arr, cols = _to_2d(data)
    T, N = arr.shape
    votes = {"seasonal": 0, "curved": 0, "trend": 0}
    t = np.arange(T, dtype=np.float64)

    for i in range(N):
        x = arr[:, i].astype(np.float64)
        mask = np.isfinite(x)
        if mask.sum() < 24 or np.std(x[mask]) == 0:
            continue
        xi = x[mask]
        ti = t[mask]

        try:
            lin = np.polyfit(ti, xi, 1)
            quad = np.polyfit(ti, xi, 2)
            res_lin = np.var(xi - np.polyval(lin, ti))
            res_quad = np.var(xi - np.polyval(quad, ti))
            slope = lin[0]
            scale = np.std(xi) + 1e-12
        except Exception:
            continue

        # --- seasonality: a dominant peak within the seasonal band ---
        detr = xi - np.polyval(lin, ti)
        seasonal = False
        m = len(detr)
        if np.std(detr) > 0 and m >= 24:
            ps = np.abs(np.fft.rfft(detr - detr.mean())) ** 2
            total = ps[1:].sum()  # exclude DC
            # band: 4 <= period <= T/3  =>  bin k in [3, m//4]  (period = m/k)
            k_lo, k_hi = 3, m // 4
            if total > 0 and k_hi >= k_lo:
                band_peak = ps[k_lo : k_hi + 1].max()
                if band_peak / total > seasonal_band_frac:
                    seasonal = True

        if seasonal:
            votes["seasonal"] += 1
        elif res_lin > 0 and (res_lin - res_quad) / res_lin > 0.2:
            votes["curved"] += 1
        elif abs(slope) * T > 0.5 * scale:
            # linear trend spans a meaningful fraction of the series scale
            votes["trend"] += 1

    if all(v == 0 for v in votes.values()):
        return "none"
    return max(votes, key=votes.get)
