# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

import random
from contextlib import contextmanager
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .structural_causal_processes import (
    check_stationarity,
    generate_structural_causal_process,
    structural_causal_process,
)


def ex1_nl(seed=42, T=300):
    """Nonlinear 5-variable dataset with linear AR and squared terms.

    Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4196918/ (2014).

    Return shapes:

    - ``ground_truth``: [cause, effect, lag] with lag 0..max_lag
    - ``weights_lin``: same shape, linear coefficients
    - ``weights_sq``: same shape, coefficient for x^2 terms (power=2)
    """
    max_lag = 3
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T + max_lag, 5))
    c = np.sqrt(2.0)

    for t in range(max_lag, T + max_lag):
        # X0 AR(2)
        x[t, 0] += 0.95 * c * x[t - 1, 0] - 0.9025 * x[t - 2, 0]

        # Nonlinear: X0^2 -> X1 (lag 2)
        x[t, 1] += 0.5 * x[t - 2, 0] ** 2.0

        # Linear: X0 -> X2 (lag 3)
        x[t, 2] += -0.4 * x[t - 3, 0]

        # Mixed: X3 AR(1), X0^2 (lag 2), X4 -> X3 (lag 1)
        x[t, 3] += (
            0.25 * c * x[t - 1, 3] - 0.50 * x[t - 2, 0] ** 2 + 0.25 * c * x[t - 1, 4]
        )

        # Linear: X4 AR(1), X3 -> X4 (lag 1)
        x[t, 4] += 0.25 * c * x[t - 1, 4] - 0.25 * c * x[t - 1, 3]

    var_names = [f"X{i}" for i in range(x.shape[1])]
    df_nl = pd.DataFrame(x[max_lag:], columns=var_names)

    d = len(var_names)
    L = max_lag

    # --- Adjacency and weights ---
    G = np.zeros(
        (d, d, L + 1), dtype=np.int8
    )  # 1 if a link exists (linear or nonlinear)
    W_lin = np.zeros((d, d, L + 1), dtype=float)  # linear coefficients & nonlinear!
    W_sq = np.zeros((d, d, L + 1), dtype=float)  # squared-term coefficients (power=2)

    # X0 -> X0 (lag 1, 2) linear
    G[0, 0, 1] = 1
    W_lin[0, 0, 1] = 0.95 * c
    G[0, 0, 2] = 1
    W_lin[0, 0, 2] = -0.9025

    # X0^2 -> X1 (lag 2)
    G[0, 1, 2] = 1
    W_sq[0, 1, 2] = 0.5
    W_lin[0, 1, 2] = 0.5

    # X0 -> X2 (lag 3) linear
    G[0, 2, 3] = 1
    W_lin[0, 2, 3] = -0.4

    # X3 -> X3 (lag 1) linear
    G[3, 3, 1] = 1
    W_lin[3, 3, 1] = 0.25 * c

    # X0^2 -> X3 (lag 2)
    G[0, 3, 2] = 1
    W_sq[0, 3, 2] = -0.50
    W_lin[0, 3, 2] = -0.50

    # X4 -> X3 (lag 1) linear
    G[4, 3, 1] = 1
    W_lin[4, 3, 1] = 0.25 * c

    # X4 -> X4 (lag 1) linear
    G[4, 4, 1] = 1
    W_lin[4, 4, 1] = 0.25 * c

    # X3 -> X4 (lag 1) linear
    G[3, 4, 1] = 1
    W_lin[3, 4, 1] = -0.25 * c

    return {
        "df": df_nl,
        "max_lag": max_lag,
        "ground_truth": G,
        "weights_lin": W_lin,
        "weights_sq": W_sq,
        "name": "5-Node Nonlinear System",
    }


def ex1_nonstationary(seed=32, T=300, trend_strength=1.0):
    """Nonstationary time series with explicit time dependence (C node).

    5 variables with contemporaneous and lagged dependence, plus
    nonstationarity via t-squared and sin(t) terms. Designed to demonstrate
    CDNOTS's ability to discover the time node C and orient edges
    using nonstationarity information.

    Parameters
    ----------
    seed : int
        Random seed.
    T : int
        Number of time steps.
    trend_strength : float, optional
        Multiplier for nonstationarity terms. Default 1.0 reproduces the
        original system. Higher values (e.g. 3.0) make the trend dominate,
        causing more spurious edges without C.

    Notes
    -----
    Equations::

        X0 = 0.6*X0_{t-1} + 0.3*X1_{t-2} + s*3*(t^2/100000) + eps
        X1 = 0.8*X1_{t-1} + s*1.5*(t^2/100000) + eps
        X2 = 0.6*X2_{t-1} + s*sin(t/20) + eps
        X3 = 0.5*X3_{t-1} + 0.7*X2_{t-2} + eps
        X4 = 0.5*X4_{t-1} + 0.3*X3_{t-1} + eps

    where s = trend_strength.
    """
    s = trend_strength
    max_lag = 2
    rng = np.random.default_rng(seed)
    ts = rng.standard_normal((T + max_lag, 5))
    for i in range(max_lag, T + max_lag):
        ts[i, 0] += (
            0.6 * ts[i - 1, 0]
            + 0.3 * ts[i - 2, 1]
            + s * 3 * ((i - max_lag) ** 2 / 100000)
        )
        ts[i, 1] += 0.8 * ts[i - 1, 1] + s * 1.5 * ((i - max_lag) ** 2 / 100000)
        ts[i, 2] += 0.6 * ts[i - 1, 2] + s * np.sin((i - max_lag) / 20)
        ts[i, 3] += 0.5 * ts[i - 1, 3] + 0.7 * ts[i - 2, 2]
        ts[i, 4] += 0.5 * ts[i - 1, 4] + 0.3 * ts[i - 1, 3]

    df = pd.DataFrame(ts[max_lag:], columns=[f"X{i}" for i in range(5)])

    # Ground truth includes C node (index 5)
    d = 6  # 5 vars + C
    gt = np.zeros((d, d, max_lag + 1))
    gt[0, 0, 1] = 1  # X0,t-1 -> X0,t
    gt[1, 0, 2] = 1  # X1,t-2 -> X0,t
    gt[5, 0, 0] = 1  # C,t -> X0,t
    gt[1, 1, 1] = 1  # X1,t-1 -> X1,t
    gt[5, 1, 0] = 1  # C,t -> X1,t
    gt[2, 2, 1] = 1  # X2,t-1 -> X2,t
    gt[5, 2, 0] = 1  # C,t -> X2,t
    gt[3, 3, 1] = 1  # X3,t-1 -> X3,t
    gt[2, 3, 2] = 1  # X2,t-2 -> X3,t
    gt[4, 4, 1] = 1  # X4,t-1 -> X4,t
    gt[3, 4, 1] = 1  # X3,t-1 -> X4,t

    return {
        "df": df,
        "max_lag": max_lag,
        "ground_truth": gt,
        "var_names": df.columns.tolist() + ["C"],
        "include_C": True,
        "name": "5-Node Nonstationary System",
    }


def confounded_trends(seed=42, T=500, trend_strength=10.0):
    """7-variable system with shared quadratic trend causing spurious confounding.

    Five variables share a quadratic time trend driven by population growth.
    None of them cause each other -- the trend is an omitted confounder.
    Without conditioning on C, the algorithm links them via spurious edges.
    With ``linear+quad`` C, all spurious edges vanish and only Temp->Energy
    (the sole genuine causal link) remains.

    Parameters
    ----------
    seed : int
        Random seed.
    T : int
        Number of time steps.
    trend_strength : float, optional
        Amplitude of the shared quadratic trend.

    Notes
    -----
    Variables:

    - Chocolate : trend only (root) -- chocolate sales grow with population
    - IceCream  : trend only (root) -- ice cream sales grow with population
    - Sunburn   : trend + self-lag (root) -- sunburn cases grow with population
    - Airline   : trend only (root) -- air travel grows with population
    - Fuel      : trend + self-lag (root) -- fuel consumption grows with population
    - Temp      : self-lag only (root, no trend)
    - Energy    : Temp(t-2) + self-lag -- the only true causal link
    """
    max_lag = 2
    rng = np.random.default_rng(seed)
    d = 7
    ts = rng.standard_normal((T + max_lag, d)) * 0.3

    for i in range(max_lag, T + max_lag):
        t_norm = (i - max_lag) / T
        trend = trend_strength * t_norm**2

        ts[i, 0] += trend  # Chocolate
        ts[i, 1] += 0.9 * trend  # IceCream
        ts[i, 2] += 0.5 * ts[i - 1, 2] + 0.85 * trend  # Sunburn (trend + self-lag)

        ts[i, 3] += 0.8 * trend  # Airline
        ts[i, 4] += 0.6 * ts[i - 1, 4] + 0.75 * trend  # Fuel (trend + self-lag)

        ts[i, 5] += 0.5 * ts[i - 1, 5]  # Temp
        ts[i, 6] += 0.4 * ts[i - 1, 6] + 0.6 * ts[i - 2, 5]  # Energy

    names = ["Chocolate", "IceCream", "Sunburn", "Airline", "Fuel", "Temp", "Energy"]
    df = pd.DataFrame(ts[max_lag:], columns=names)

    gt = np.zeros((d, d, max_lag + 1))
    gt[2, 2, 1] = 1  # Sunburn self-lag
    gt[4, 4, 1] = 1  # Fuel self-lag
    gt[5, 5, 1] = 1  # Temp self-lag
    gt[5, 6, 2] = 1  # Temp(t-2) -> Energy  (only true causal link)
    gt[6, 6, 1] = 1  # Energy self-lag

    return {
        "df": df,
        "max_lag": max_lag,
        "ground_truth": gt,
        "var_names": names,
        "include_C": True,
        "name": "Confounded Trends (7-Node)",
    }


def ex2(seed=42, T=300):
    max_lag = 3
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T + max_lag, 6))
    c = np.sqrt(2.0)
    for t in range(max_lag, T + max_lag):
        x[t, 0] += 0.5 * c * x[t - 1, 0]  # X0,t-1 -> X0
        x[t, 1] += 0.5 * x[t - 2, 0]  # X0,t-2 -> X1
        x[t, 2] += -0.4 * x[t - 3, 0]  # X0,t-3 -> X2
        x[t, 3] += (
            0.25 * c * x[t - 1, 3] - 0.50 * x[t - 2, 0] + 0.45 * c * x[t - 1, 4]
        )  # X3,t-1 -> X3     X0,t-2 -> X3   X4,t-1 -> X3
        x[t, 4] += (
            0.25 * c * x[t - 1, 4] - 0.5 * c * x[t - 1, 3]
        )  # X4,t-1 -> X4    X3,t-1 -> X4
        x[t, 5] += (
            -0.1 * x[t - 1, 2] - 0.55 * x[t - 2, 4] + 0.65 * x[t - 1, 5]
        )  # X2,t-1 -> X5    X4,t-2 -> X5    X5,t-1 -> X5
    df_ex2 = pd.DataFrame(
        x[max_lag:], columns=["X" + str(i) for i in range(x.shape[1])]
    )

    gt = np.zeros((df_ex2.shape[1], df_ex2.shape[1], max_lag + 1))
    gt[0, 0, 1] = 1  # X0,t-1 -> X0,t
    gt[0, 1, 2] = 1  # X0,t-2 -> X1,t
    gt[0, 2, 3] = 1  # X0,t-3 -> X2,t
    gt[3, 3, 1] = 1  # X3,t-1 -> X3,t
    gt[0, 3, 2] = 1  # X0,t-2 -> X3,t
    gt[4, 3, 1] = 1  # X4,t-1 -> X3,t
    gt[4, 4, 1] = 1  # X4,t-1 -> X4,t
    gt[3, 4, 1] = 1  # X3,t-1 -> X4,t
    gt[5, 5, 1] = 1  # X5,t-1 -> X5,t
    gt[2, 5, 1] = 1  # X2,t-1 -> X5,t
    gt[4, 5, 2] = 1  # X4,t-2 -> X5,t

    d = 6
    c = np.sqrt(2.0)
    W = np.zeros((d, d, max_lag + 1), dtype=float)

    # From the equations:
    W[0, 0, 1] = 0.5 * c  # X0,t-1 -> X0,t
    W[0, 1, 2] = 0.5  # X0,t-2 -> X1,t
    W[0, 2, 3] = -0.4  # X0,t-3 -> X2,t

    W[3, 3, 1] = 0.25 * c  # X3,t-1 -> X3,t
    W[0, 3, 2] = -0.50  # X0,t-2 -> X3,t
    W[4, 3, 1] = 0.45 * c  # X4,t-1 -> X3,t

    W[4, 4, 1] = 0.25 * c  # X4,t-1 -> X4,t
    W[3, 4, 1] = -0.50 * c  # X3,t-1 -> X4,t

    W[5, 5, 1] = 0.65  # X5,t-1 -> X5,t
    W[2, 5, 1] = -0.10  # X2,t-1 -> X5,t
    W[4, 5, 2] = -0.55  # X4,t-2 -> X5,t

    return {
        "df": df_ex2,
        "max_lag": max_lag,
        "ground_truth": gt,
        "weights": W,
        "name": "6-Node Linear VAR",
    }


def ex3(seed=42, T=300):
    rng = np.random.default_rng(seed)
    maxlag = 3
    x = rng.standard_normal((T + maxlag, 11))
    for t in range(maxlag, T + maxlag):
        x[t, 0] += 0.9 * x[t - 1, 0]  # Plcg
        x[t, 1] += 0.6 * x[t - 1, 1] + 0.5 * x[t - 2, 0]  # PIP3
        x[t, 2] += 0.4 * x[t - 1, 2] + 0.4 * x[t - 3, 0] + 0.3 * x[t - 1, 1]  # PIP2
        x[t, 3] += 0.25 * x[t - 1, 3]  # PKC
        x[t, 4] += 0.25 * x[t - 1, 4] - 0.25 * x[t - 1, 3]  # PKA
        x[t, 5] += 0.25 * x[t - 1, 5] + 0.50 * x[t - 2, 3] - 0.50 * x[t - 1, 4]  # P38
        x[t, 6] += 0.30 * x[t - 1, 6] - 0.25 * x[t - 3, 3] + 0.25 * x[t - 3, 4]  # Jnk
        x[t, 7] += 0.40 * x[t - 1, 7] + 0.35 * x[t - 1, 3] + 0.25 * x[t - 2, 4]  # Raf
        x[t, 8] += (
            0.50 * x[t - 1, 8]
            - 0.5 * x[t - 1, 3]
            + 0.25 * x[t - 1, 4]
            + 0.6 * x[t - 2, 7]
        )  # Mek
        x[t, 9] += 0.6 * x[t - 1, 9] + 0.45 * x[t - 1, 4] - 0.35 * x[t - 2, 8]  # Erk
        x[t, 10] += 0.7 * x[t - 1, 10] - 0.25 * x[t - 1, 4] + 0.25 * x[t - 2, 9]  # Akt
    var_names = [
        "Plcg",
        "Pip3",
        "Pip2",
        "PKC",
        "PKA",
        "P38",
        "Jnk",
        "Raf",
        "Mek",
        "Erk",
        "Akt",
    ]
    df_ex3 = pd.DataFrame(x[maxlag:], columns=var_names)

    d = len(var_names)

    # adjacency and weights: [cause, effect, lag], lag 0..maxlag
    G = np.zeros((d, d, maxlag + 1), dtype=np.int8)
    W = np.zeros((d, d, maxlag + 1), dtype=float)

    def link(cause, effect, lag, w):
        G[cause, effect, lag] = 1
        W[cause, effect, lag] = w

    # ---- edges from the equations ----
    # x[t,0]  += 0.9 * x[t-1,0]
    link(0, 0, 1, 0.9)

    # x[t,1]  += 0.6 * x[t-1,1] + 0.5 * x[t-2,0]
    link(1, 1, 1, 0.6)
    link(0, 1, 2, 0.5)

    # x[t,2]  += 0.4 * x[t-1,2] + 0.4 * x[t-3,0] + 0.3 * x[t-1,1]
    link(2, 2, 1, 0.4)
    link(0, 2, 3, 0.4)
    link(1, 2, 1, 0.3)

    # x[t,3]  += 0.25 * x[t-1,3]
    link(3, 3, 1, 0.25)

    # x[t,4]  += 0.25 * x[t-1,4] - 0.25 * x[t-1,3]
    link(4, 4, 1, 0.25)
    link(3, 4, 1, -0.25)

    # x[t,5]  += 0.25 * x[t-1,5] + 0.50 * x[t-2,3] - 0.50 * x[t-1,4]
    link(5, 5, 1, 0.25)
    link(3, 5, 2, 0.50)
    link(4, 5, 1, -0.50)

    # x[t,6]  += 0.30 * x[t-1,6] - 0.25 * x[t-3,3] + 0.25 * x[t-3,4]
    link(6, 6, 1, 0.30)
    link(3, 6, 3, -0.25)
    link(4, 6, 3, 0.25)

    # x[t,7]  += 0.40 * x[t-1,7] + 0.35 * x[t-1,3] + 0.25 * x[t-2,4]
    link(7, 7, 1, 0.40)
    link(3, 7, 1, 0.35)
    link(4, 7, 2, 0.25)

    # x[t,8]  += 0.50 * x[t-1,8] - 0.5 * x[t-1,3] + 0.25 * x[t-1,4] + 0.6 * x[t-2,7]
    link(8, 8, 1, 0.50)
    link(3, 8, 1, -0.50)
    link(4, 8, 1, 0.25)
    link(7, 8, 2, 0.60)

    # x[t,9]  += 0.6 * x[t-1,9] + 0.45 * x[t-1,4] - 0.35 * x[t-2,8]
    link(9, 9, 1, 0.60)
    link(4, 9, 1, 0.45)
    link(8, 9, 2, -0.35)

    # x[t,10] += 0.7 * x[t-1,10] - 0.25 * x[t-1,4] + 0.25 * x[t-2,9]
    link(10, 10, 1, 0.70)
    link(4, 10, 1, -0.25)
    link(9, 10, 2, 0.25)

    return {
        "df": df_ex3,
        "max_lag": maxlag,
        "ground_truth": G,
        "weights": W,
        "name": "Cascade Network (11-Node)",
    }


def henon_chain(seed=42, T=200, Q=0.5, M=5):
    """Coupled (Henon-style) chain with nearest-neighbor nonlinear coupling.

    Parameters
    ----------
    seed : int
        RNG seed.
    T : int
        Number of samples returned (after burn-in of max_lag).
    Q : float
        Coupling strength (as in the provided equations).
    M : int
        Number of variables in the chain.

    Returns
    -------
    dict
        Dict with keys:

        - 'df': DataFrame of shape [T, M], columns 'X0'..'X{M-1}'
        - 'max_lag': int (=2)
        - 'ground_truth': ndarray [cause, effect, lag] with lag in [0..2];
          (1 marks an edge; includes nonlinear parents for lag 1,
          and self linear AR term at lag 2)
        - 'weights': None (nonlinear mixture, no single linear weight
          per edge)
    """
    max_lag = 2
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T + max_lag, M))

    for t in range(max_lag, T + max_lag):
        for m in range(M):
            if m in (0, M - 1):
                # boundaries: self-only nonlinear at lag 1 + linear AR at lag 2
                x[t, m] = 1.4 - x[t - 1, m] ** 2 + 0.3 * x[t - 2, m]
            else:
                # interior: nonlinear mix of neighbors + self at lag 1, plus linear AR at lag 2
                mix = (
                    0.5
                    * Q
                    * (x[t - 1, m - 1] + x[t - 1, m + 1] + (1 - Q) * x[t - 1, m])
                )
                x[t, m] = 1.4 - mix**2 + 0.3 * x[t - 2, m]

    var_names = [f"X{i}" for i in range(M)]
    df = pd.DataFrame(x[max_lag:], columns=var_names)

    # ---- Ground-truth adjacency: [cause, effect, lag], lag=0 is instantaneous (unused here) ----
    G = np.zeros((M, M, max_lag + 1), dtype=np.int8)

    # All nodes have linear self AR term at lag=2 (coefficient 0.3)
    for m in range(M):
        G[m, m, 2] = 1

    # Nonlinear parents at lag=1:
    # - boundaries: self at lag=1 (since x_{t-1,m}^2 term)
    # - interior: (m-1), m, (m+1) at lag=1 (since squared mix of those three)
    for m in range(M):
        if m == 0 or m == M - 1:
            G[m, m, 1] = 1
        else:
            G[m - 1, m, 1] = 1
            G[m, m, 1] = 1
            G[m + 1, m, 1] = 1

    return {
        "df": df,
        "max_lag": max_lag,
        "ground_truth": G,  # [cause, effect, lag]
        "weights": None,  # nonlinear mixture => no single linear weights
        "name": "Hénon Chain",
    }


from causalts.synthetic_data.weather_datasets import ercot_weather  # noqa: E402

DATASET_FUNCS = {
    "ex2": ex2,
    "ex3": ex3,
    "ex1": ex1_nl,
    "henon": henon_chain,
    "ercot_weather": ercot_weather,
}


def load_dataset(name, **kwargs):
    return DATASET_FUNCS[name](**kwargs)


# ---- safe nonlinearities (avoid overflow/NaNs) ----
def lin(x):
    return x


def nl1(x):
    # x + 0.5 x^2 with clipping so large magnitudes don't blow up
    z = np.clip(x, -50.0, 50.0)
    return z + 0.5 * (z * z)


def nl2(x):
    # x + sin(0.1 x) with wide clip for safety
    z = np.clip(x, -1e6, 1e6)
    return z + np.sin(0.1 * z)


def nl_sq(x):
    """Pure quadratic (no linear component)."""
    return np.clip(x**2, -50, 50)


def nl_abs(x):
    """Absolute value."""
    return np.abs(x)


def nl_sin(x):
    """Pure sinusoidal."""
    return np.sin(x)


DEFAULT_FUNC_MAP = {
    "lin": lin,
    "nl1": nl1,
    "nl2": nl2,
    "nl_sq": nl_sq,
    "nl_abs": nl_abs,
    "nl_sin": nl_sin,
}
DEFAULT_FUNC_NAMES = ("lin", "lin", "nl1", "nl2")  # names passed to the generator
NL_FUNC_NAMES = (
    "nl_sq",
    "nl_abs",
    "nl_sin",
    "nl_sq",
)  # purely nonlinear (no linear component)


class SCPGraphGenerator:
    """
    Generate ONE structural causal process (SCP): data, links, and ground-truth adjacency.

    Returns a dict:
      - df            : DataFrame [T, n_vars]
      - max_lag       : int
      - links         : raw links dict (from generator)
      - noises        : noise spec (from generator)
      - ground_truth  : ndarray [cause, effect, lag], lag in [0..max_lag]
      - var_names     : list of 'X0'..'X{n_vars-1}'
      - meta          : dict (n_links, funcs_count, attempt, seed_used)
      - weights       : None  (kept for interface compatibility; edges can be nonlinear)
    """

    # Default coefficient ranges (used to detect caller overrides in dense mode)
    _DEFAULT_AUTO_RANGE = (0.30, 0.79)
    _DEFAULT_DEP_POS_RANGE = (0.40, 0.89)
    _DEFAULT_DEP_NEG_RANGE = (-0.40, -0.89)
    _DEFAULT_MAX_SPECTRAL_RADIUS = 0.85

    def __init__(
        self,
        n_vars: int,
        max_lag: int = 5,
        # coefficient pools (defaults mirror your script)
        auto_range: Tuple[float, float] = _DEFAULT_AUTO_RANGE,
        dep_pos_range: Tuple[float, float] = _DEFAULT_DEP_POS_RANGE,
        dep_neg_range: Tuple[float, float] = _DEFAULT_DEP_NEG_RANGE,
        step: float = 0.01,
        dependency_funcs: Tuple[
            str, ...
        ] = DEFAULT_FUNC_NAMES,  # names the generator expects
        func_map: Optional[Dict[str, Callable]] = None,  # callables for simulation
        enforce_stationary: bool = True,
        max_spectral_radius: float = _DEFAULT_MAX_SPECTRAL_RADIUS,
        max_std_ratio: float = 20.0,  # reject if max(std)/min(std) exceeds this
        density: str = "sparse",  # "sparse" (default) or "dense"
    ):
        if density not in ("sparse", "dense"):
            raise ValueError(f"density must be 'sparse' or 'dense', got {density!r}")

        self.n_vars = int(n_vars)
        self.max_lag = int(max_lag)
        self.density = density
        self.step = float(step)
        self.dependency_funcs = tuple(dependency_funcs)
        self.func_map = func_map if func_map is not None else DEFAULT_FUNC_MAP
        self.enforce_stationary = bool(enforce_stationary)
        self.max_std_ratio = float(max_std_ratio)

        # In dense mode, override defaults only if caller hasn't explicitly set them
        if density == "dense":
            self.auto_range = (
                auto_range if auto_range != self._DEFAULT_AUTO_RANGE else (0.15, 0.50)
            )
            self.dep_pos_range = (
                dep_pos_range
                if dep_pos_range != self._DEFAULT_DEP_POS_RANGE
                else (0.15, 0.45)
            )
            self.dep_neg_range = (
                dep_neg_range
                if dep_neg_range != self._DEFAULT_DEP_NEG_RANGE
                else (-0.45, -0.15)
            )
            self.max_spectral_radius = (
                max_spectral_radius
                if max_spectral_radius != self._DEFAULT_MAX_SPECTRAL_RADIUS
                else 0.70
            )
        else:
            self.auto_range = auto_range
            self.dep_pos_range = dep_pos_range
            self.dep_neg_range = dep_neg_range
            self.max_spectral_radius = float(max_spectral_radius)

    # ---------- public API ----------
    def sample(
        self,
        seed: int,
        T: int = 3000,
        n_links: Optional[int] = None,
        max_tries: int = 200,
    ):
        rng = np.random.default_rng(seed)

        auto_coeffs = self._arange_inclusive(*self.auto_range, self.step).tolist()
        pos = self._arange_inclusive(*self.dep_pos_range, self.step).tolist()
        neg = self._arange_inclusive(*self.dep_neg_range, self.step).tolist()
        dependency_coeffs = pos + neg
        rng.shuffle(dependency_coeffs)

        if n_links is None:
            if self.density == "dense":
                lb, ub = self._default_links_bounds_dense(self.n_vars)
            else:
                lb, ub = self._default_links_bounds(self.n_vars)
            n_links = int(rng.integers(lb, ub + 1))

        attempt = 0
        target_sr = self.max_spectral_radius
        while attempt < max_tries:
            gen_seed = int((seed * 1009 + attempt) & 0xFFFFFFFF)
            noise_seed = int((seed * 9176 + attempt) & 0xFFFFFFFF)
            sim_seed = int((seed * 9176 + attempt) & 0xFFFFFFFF)

            # deterministic generator
            with self._seed_scope(gen_seed):
                links, noises = generate_structural_causal_process(
                    N=self.n_vars,
                    L=n_links,
                    max_lag=self.max_lag,
                    dependency_funcs=list(self.dependency_funcs),
                    auto_coeffs=auto_coeffs,
                    dependency_coeffs=dependency_coeffs,
                    seed=gen_seed,
                    noise_seed=noise_seed,
                )

            if self.enforce_stationary:
                sr = self._spectral_radius(links)
                if sr >= 1.0:
                    if self.density == "dense":
                        # Rescue rescaling: pull sr down instead of rejecting
                        links = self._rescale_links(links, 0.6 / sr)
                    else:
                        attempt += 1
                        continue
                # Rescale coefficients to target spectral radius
                if sr > target_sr:
                    links = self._rescale_links(links, target_sr / sr)

            # deterministic simulator (seed passed + global RNG scoped)
            with self._seed_scope(sim_seed):
                data, nonstat = structural_causal_process(
                    links, T=T, noises=noises, func_map=self.func_map, seed=sim_seed
                )
            if not nonstat:
                # Check std ratio to reject wild variance from nonlinear amplification
                stds = np.std(data, axis=0)
                std_ratio = stds.max() / (stds.min() + 1e-10)
                if std_ratio <= self.max_std_ratio:
                    break
                # Progressively tighten spectral radius to dampen nonlinear cascades
                target_sr = max(0.3, target_sr * 0.97)
            attempt += 1

        if attempt >= max_tries:
            raise RuntimeError(
                "Failed to generate a stationary process in max_tries attempts."
            )

        var_names = [f"X{i}" for i in range(self.n_vars)]
        df = pd.DataFrame(data, columns=var_names)
        gt = self._links_to_gt(links, self.n_vars, self.max_lag)
        funcs_count = self._func_usage(links)

        return {
            "df": df,
            "max_lag": self.max_lag,
            "links": links,
            "noises": noises,
            "ground_truth": gt,
            "var_names": var_names,
            "name": f"SCP Random Graph (d={self.n_vars})",
            "meta": {
                "n_links": int(n_links),
                "funcs_count": funcs_count,
                "attempt": attempt,
                "seed_used": int(seed),
                "deterministic": True,
                "density": self.density,
            },
            "weights": None,
        }

    # ---------- helpers ----------
    @contextmanager
    def _seed_scope(self, seed: int):
        """Temporarily set NumPy and Python RNG states; restore afterward."""
        # Save current states
        py_state = random.getstate()
        try:
            np_state = np.random.get_state()  # works for NumPy ≤/≥ 2 via legacy API
        except Exception:
            np_state = None

        # Set seeds (mask to 32-bit for safety)
        s = int(seed) & 0xFFFFFFFF
        random.seed(s)
        try:
            np.random.seed(s)
        except Exception:
            pass  # very old/odd NumPy—safe to ignore

        try:
            yield  # run the seeded code block
        finally:
            # Restore previous states
            random.setstate(py_state)
            if np_state is not None:
                try:
                    np.random.set_state(np_state)
                except Exception:
                    pass

    @staticmethod
    def _arange_inclusive(a: float, b: float, step: float) -> np.ndarray:
        if step <= 0:
            raise ValueError("step must be > 0")
        if a <= b:
            n = int(round((b - a) / step)) + 1
            vals = a + step * np.arange(n)
        else:
            n = int(round((a - b) / step)) + 1
            vals = a - step * np.arange(n)
        return np.round(vals, 6)

    @staticmethod
    def _rescale_links(links, scale: float):
        """Multiply all coefficients by scale, preserving graph structure."""
        new_links = {}
        for j, edges in links.items():
            new_links[j] = [
                (link_props[0], link_props[1] * scale, link_props[2])
                for link_props in edges
            ]
        return new_links

    @staticmethod
    def _spectral_radius(links) -> float:
        """Compute spectral radius of the companion VAR matrix."""
        N = len(links)
        max_lag = 0
        for j in range(N):
            for link_props in links[j]:
                _, lag = link_props[0]
                max_lag = max(max_lag, abs(lag))
        if max_lag == 0:
            return 0.0
        graph = np.zeros((N, N, max_lag))
        for j in range(N):
            for link_props in links[j]:
                var, lag = link_props[0]
                coeff = link_props[1]
                if abs(lag) > 0:
                    graph[j, var, abs(lag) - 1] = coeff
        stabmat = np.zeros((N * max_lag, N * max_lag))
        for idx in range(max_lag):
            i = idx * N
            stabmat[:N, i : i + N] = graph[:, :, idx]
            if idx < max_lag - 1:
                stabmat[i + N : i + 2 * N, i : i + N] = np.identity(N)
        return float(np.max(np.abs(np.linalg.eig(stabmat)[0])))

    @staticmethod
    def _default_links_bounds(n_vars: int) -> Tuple[int, int]:
        # Original heuristic, capped to keep large networks in sparse regime
        lb = int(n_vars - 1.9 - 0.009 * (n_vars**2))
        ub = int(np.round(2.25 * n_vars - 4))
        lb = max(lb, 1)
        # For d>=20, cap at d + 5*sqrt(d) to avoid nonlinear cascade blowup
        sparse_cap = int(n_vars + 4 * np.sqrt(n_vars))
        ub = min(ub, sparse_cap)
        ub = max(ub, lb)
        return lb, ub

    @staticmethod
    def _default_links_bounds_dense(n_vars: int) -> Tuple[int, int]:
        """Link bounds for dense mode: ~3-5 edges per variable."""
        lb = int(2.0 * n_vars)
        ub = int(3.5 * n_vars)
        # Can't exceed total directed pairs (excluding self-loops which are AR)
        max_pairs = n_vars * (n_vars - 1) // 2
        ub = min(ub, max_pairs)
        lb = min(lb, ub)
        lb = max(lb, 1)
        return lb, ub

    # ---- robust parsing for your link shape: ((cause, -lag), coef, func) ----
    @staticmethod
    def _num_from_any(x):
        """int from int/float/np types or strings like 'X3'/'3'."""
        if isinstance(x, (int, np.integer, float, np.floating)):
            return int(x)
        if isinstance(x, str):
            import re

            m = re.search(r"-?\d+", x)
            if m:
                return int(m.group(0))
        return None

    def _links_to_gt(self, links: dict, n_vars: int, max_lag: int) -> np.ndarray:
        """
        Convert 'links' to binary adjacency G[cause, effect, lag] with lag in [0..max_lag].
        Accepts entries like: [ ((cause, -lag), coef, func), ... ].
        (No unconditional skipping of index 0; add only when parsing succeeds.)
        """
        G = np.zeros((n_vars, n_vars, max_lag + 1), dtype=np.int8)

        for e_key, lst in links.items():
            e = self._num_from_any(e_key)
            if e is None or not (0 <= e < n_vars):
                continue

            for entry in lst:
                # Expect tuple/list with ((cause, -lag), coef, func)
                if not (isinstance(entry, (list, tuple)) and len(entry) >= 3):
                    continue
                pair, coef, func = entry[0], entry[1], entry[2]  # noqa: F841
                if not (isinstance(pair, (list, tuple)) and len(pair) >= 2):
                    continue

                c = self._num_from_any(pair[0])
                neg_lag = self._num_from_any(pair[1])
                if c is None or neg_lag is None:
                    continue

                lag = -int(neg_lag)  # convert negative lag to positive index
                if 0 <= c < n_vars and 0 <= e < n_vars and 0 <= lag <= max_lag:
                    G[c, e, lag] = 1

        return G

    def _func_usage(self, links: dict) -> Dict[str, int]:
        from collections import Counter

        names = []
        for _, lst in links.items():
            for entry in lst:
                if isinstance(entry, (list, tuple)) and len(entry) >= 3:
                    fn = entry[2]
                    if isinstance(fn, (str, np.str_)):
                        names.append(str(fn))
        return dict(Counter(names))


def lorenz96(seed=42, T=1000, N=10, F=10.0, dt=0.1, obs_noise=0.1, burn_in=1000):
    """Lorenz-96 model with multiplicative interactions.

    ODE: dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F

    Each variable x_i has 4 causal parents at lag 1: x_{i-2}, x_{i-1}, x_i, x_{i+1}
    (cyclic boundary). Mechanism is multiplicative — violates additive assumptions.
    Used as a robustness benchmark for causal discovery methods.

    Parameters match the CUTS+ implementation (Cheng et al., 2024):
    F=10, dt=0.1, obs_noise=0.1, burn_in=1000, scipy odeint integrator.

    Parameters
    ----------
    seed : int
    T : int
        Number of time steps returned (after burn-in).
    N : int
        Number of variables.
    F : float
        Forcing constant (F=10 is standard chaotic regime).
    dt : float
        Integration step.
    obs_noise : float
        Std of additive Gaussian observation noise.
    burn_in : int
        Integration steps to discard as transient.

    Returns
    -------
    dict with 'df' [T, N], 'ground_truth' [N, N, 2], 'max_lag' (=1), 'weights' (None)
    """
    from scipy.integrate import odeint

    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    def _rhs(x_state, t, F_val):
        p = len(x_state)
        dxdt = np.zeros(p)
        for i in range(p):
            dxdt[i] = (
                (x_state[(i + 1) % p] - x_state[(i - 2) % p]) * x_state[(i - 1) % p]
                - x_state[i]
                + F_val
            )
        return dxdt

    x0 = rng.standard_normal(N) * 0.01
    t_span = (
        np.arange(T + burn_in) * dt
    )  # exact dt steps so data is reproducible across different T values
    X = odeint(_rhs, x0, t_span, args=(F,))
    X += rng.normal(scale=obs_noise, size=X.shape)
    data = X[burn_in:]

    var_names = [f"X{i}" for i in range(N)]
    df = pd.DataFrame(data, columns=var_names)

    G = np.zeros((N, N, 2), dtype=np.int8)
    for i in range(N):
        G[(i - 2) % N, i, 1] = 1
        G[(i - 1) % N, i, 1] = 1
        G[i, i, 1] = 1
        G[(i + 1) % N, i, 1] = 1

    return {
        "df": df,
        "max_lag": 1,
        "ground_truth": G,
        "weights": None,
        "name": "Lorenz-96",
    }


def topology_scp(
    topology: str,
    seed: int,
    n_vars: int = 15,
    max_lag: int = 3,
    T: int = 500,
    dependency_funcs=DEFAULT_FUNC_NAMES,
    auto_range=(0.30, 0.79),
    dep_pos_range=(0.40, 0.89),
    dep_neg_range=(-0.40, -0.89),
    max_spectral_radius: float = 0.85,
    max_std_ratio: float = 20.0,
    max_tries: int = 200,
    ba_m: int = 2,
    er_p: float = 0.08,
    ws_k: int = 4,
    ws_p: float = 0.3,
) -> dict:
    """Generate SCP time-series data with a structured graph topology.

    Topologies:

    - ``'scale_free'``  -- Barabasi-Albert hub-and-spoke (ba_m edges per new node)
    - ``'erdos_renyi'`` -- random directed graph (er_p edge probability)
    - ``'small_world'`` -- Watts-Strogatz clustered ring (ws_k neighbours,
      ws_p rewiring)

    Uses the same SCP coefficient pools, nonlinear functions, and stability
    checks as SCPGraphGenerator. Returns the same bundle format.

    Parameters
    ----------
    topology : str
        One of 'scale_free', 'erdos_renyi', 'small_world'.
    seed : int
        Random seed.
    n_vars : int
        Number of variables.
    max_lag : int
        Maximum lag depth.
    T : int
        Number of time steps.
    ba_m : int
        BA edges added per new node (scale_free density control).
    er_p : float
        ER edge probability (erdos_renyi density control).
    ws_k : int
        WS nearest neighbours (small_world density control).
    ws_p : float
        WS rewiring probability.
    """
    import networkx as nx

    rng = np.random.default_rng(seed)  # noqa: F841

    auto_coeffs = np.arange(auto_range[0], auto_range[1] + 0.005, 0.01).tolist()
    pos_coeffs = np.arange(dep_pos_range[0], dep_pos_range[1] + 0.005, 0.01).tolist()
    neg_coeffs = np.arange(dep_neg_range[0], dep_neg_range[1] + 0.005, 0.01).tolist()
    dep_coeffs = pos_coeffs + neg_coeffs
    func_list = list(dependency_funcs)

    def _make_directed_edges(topo, n, attempt_rng):
        nx_seed = int(attempt_rng.integers(2**31))
        if topo == "scale_free":
            G_nx = nx.barabasi_albert_graph(n, m=ba_m, seed=nx_seed)
            edges = [(min(u, v), max(u, v)) for u, v in G_nx.edges()]
        elif topo == "erdos_renyi":
            G_nx = nx.erdos_renyi_graph(n, p=er_p, directed=True, seed=nx_seed)
            edges = [(u, v) for u, v in G_nx.edges() if u != v]
        elif topo == "small_world":
            G_nx = nx.watts_strogatz_graph(n, k=ws_k, p=ws_p, seed=nx_seed)
            edges = [(min(u, v), max(u, v)) for u, v in G_nx.edges()]
        else:
            raise ValueError(
                f"Unknown topology: {topo!r}. Choose from 'scale_free', 'erdos_renyi', 'small_world'."
            )
        return list(set(edges))

    attempt = 0
    target_sr = max_spectral_radius

    while attempt < max_tries:
        attempt_rng = np.random.default_rng(seed * 1009 + attempt)
        directed_edges = _make_directed_edges(topology, n_vars, attempt_rng)

        links = {j: [] for j in range(n_vars)}
        for j in range(n_vars):
            a = float(attempt_rng.choice(auto_coeffs))
            if a != 0.0:
                links[j].append(((int(j), -1), a, "lin"))
        for cause, effect in directed_edges:
            lag = int(attempt_rng.integers(1, max_lag + 1))
            coeff = float(attempt_rng.choice(dep_coeffs))
            func = str(attempt_rng.choice(func_list))
            if coeff != 0.0:
                links[effect].append(((int(cause), -lag), coeff, func))

        sr = SCPGraphGenerator._spectral_radius(links)
        if sr >= 1.0:
            links = SCPGraphGenerator._rescale_links(links, 0.6 / sr)
            sr = SCPGraphGenerator._spectral_radius(links)
        if sr > target_sr:
            links = SCPGraphGenerator._rescale_links(links, target_sr / sr)

        sim_seed = int((seed * 9176 + attempt) & 0xFFFFFFFF)
        sim_rng = np.random.RandomState(sim_seed)
        noises = []
        for _ in range(n_vars):
            sigma = float(attempt_rng.uniform(0.5, 2.0))

            def _make_noise(s, rng_ref=sim_rng):
                def noise(T):
                    return s * rng_ref.randn(T)

                return noise

            noises.append(_make_noise(sigma))

        from .structural_causal_processes import structural_causal_process

        data, nonstat = structural_causal_process(
            links,
            T=T,
            noises=noises,
            func_map=DEFAULT_FUNC_MAP,
            seed=sim_seed,
        )

        if not nonstat:
            stds = np.std(data, axis=0)
            if stds.min() > 1e-10 and stds.max() / stds.min() <= max_std_ratio:
                break
            target_sr = max(0.3, target_sr * 0.97)
        attempt += 1

    if attempt >= max_tries:
        raise RuntimeError(
            f"Failed to generate stationary {topology!r} process in {max_tries} attempts."
        )

    var_names = [f"X{i}" for i in range(n_vars)]
    df = pd.DataFrame(data, columns=var_names)
    gen = SCPGraphGenerator(n_vars=n_vars, max_lag=max_lag)
    gt = gen._links_to_gt(links, n_vars, max_lag)
    n_cross = len(directed_edges)
    possible = n_vars * (n_vars - 1) * max_lag
    edge_density = n_cross / possible if possible > 0 else 0.0

    # Degree stats from the undirected topology (cause+effect count per node)
    from collections import Counter

    deg_count = Counter()
    for u, v in directed_edges:
        deg_count[u] += 1
        deg_count[v] += 1
    degrees = [deg_count.get(i, 0) for i in range(n_vars)]
    mean_degree = sum(degrees) / n_vars if n_vars > 0 else 0.0
    max_degree = max(degrees) if degrees else 0

    return {
        "df": df,
        "max_lag": max_lag,
        "ground_truth": gt,
        "var_names": var_names,
        "weights": None,
        "meta": {
            "topology": topology,
            "n_cross_edges": n_cross,
            "edge_density": round(edge_density, 4),
            "mean_degree": round(mean_degree, 2),
            "max_degree": int(max_degree),
            "spectral_radius": round(
                float(SCPGraphGenerator._spectral_radius(links)), 4
            ),
            "seed_used": int(seed),
            "attempt": attempt,
        },
    }


def scale_free(seed=42, n_vars=15, T=500, **kw):
    """Scale-free (Barabasi-Albert) sparse topology, ba_m=2."""
    return topology_scp("scale_free", seed=seed, n_vars=n_vars, T=T, ba_m=2, **kw)


def scale_free_dense(seed=42, n_vars=15, T=500, **kw):
    """Scale-free (Barabasi-Albert) dense topology, ba_m=4."""
    return topology_scp("scale_free", seed=seed, n_vars=n_vars, T=T, ba_m=4, **kw)


def erdos_renyi(seed=42, n_vars=15, T=500, **kw):
    """Erdos-Renyi sparse topology, er_p=0.06."""
    return topology_scp("erdos_renyi", seed=seed, n_vars=n_vars, T=T, er_p=0.06, **kw)


def erdos_renyi_dense(seed=42, n_vars=15, T=500, **kw):
    """Erdos-Renyi dense topology, er_p=0.18."""
    return topology_scp("erdos_renyi", seed=seed, n_vars=n_vars, T=T, er_p=0.18, **kw)


def small_world(seed=42, n_vars=15, T=500, **kw):
    """Small-world (Watts-Strogatz) sparse topology, ws_k=3."""
    return topology_scp("small_world", seed=seed, n_vars=n_vars, T=T, ws_k=3, **kw)


def small_world_dense(seed=42, n_vars=15, T=500, **kw):
    """Small-world (Watts-Strogatz) dense topology, ws_k=6."""
    return topology_scp("small_world", seed=seed, n_vars=n_vars, T=T, ws_k=6, **kw)


DATASET_FUNCS.update(
    {
        "scale_free": scale_free,
        "scale_free_dense": scale_free_dense,
        "erdos_renyi": erdos_renyi,
        "erdos_renyi_dense": erdos_renyi_dense,
        "small_world": small_world,
        "small_world_dense": small_world_dense,
    }
)


# ---------- Example usage ----------
if __name__ == "__main__":
    gen = SCPGraphGenerator(
        n_vars=6,
        max_lag=5,
        # You can customize ranges; defaults match your script
        # auto_range=(0.30, 0.79),
        # dep_pos_range=(0.40, 0.89),
        # dep_neg_range=(-0.40, -0.89),
        # dependency_funcs=("lin", "lin", "nl1", "nl2"),
    )

    sample = gen.sample(seed=123, T=1000, n_links=None, max_tries=200)
    print("Data shape:", sample["df"].shape)
    print("Adj nonzeros:", int(sample["ground_truth"].sum()))
    print("Func counts:", sample["meta"]["funcs_count"])
