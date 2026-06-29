# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Synthetic benchmark datasets with mixed discrete and continuous variables."""

import numpy as np
import pandas as pd


def mixed_a(seed=42, T=300):
    """Binary regime + continuous variables (5 vars, 1 binary).

    Variables:
        R  : binary regime indicator, AR(1) Markov chain (transition prob 0.1)
        X0 : continuous, driven by R_{t-1} and X0_{t-1}
        X1 : continuous, driven by X0_{t-2}
        X2 : continuous, driven by R_t (contemporaneous) and X2_{t-1}
        X3 : continuous, independent AR(1)

    True edges (cause → effect, lag):
        R_{t-1}  → R_t       (lag 1, self)
        R_{t-1}  → X0_t      (lag 1)
        X0_{t-1} → X0_t      (lag 1, self)
        X0_{t-2} → X1_t      (lag 2)
        R_t      → X2_t      (lag 0, contemporaneous)
        X2_{t-1} → X2_t      (lag 1, self)
        X3_{t-1} → X3_t      (lag 1, self)

    Returns dict with keys: df, ground_truth, discrete_cols, max_lag.
    """
    rng = np.random.default_rng(seed)
    max_lag = 2

    R = np.zeros(T + max_lag, dtype=np.int32)
    R[0] = rng.integers(0, 2)
    trans = np.array([[0.9, 0.1], [0.1, 0.9]])
    for t in range(1, T + max_lag):
        R[t] = rng.choice(2, p=trans[R[t - 1]])

    X0 = np.zeros(T + max_lag)
    X1 = np.zeros(T + max_lag)
    X2 = np.zeros(T + max_lag)
    X3 = np.zeros(T + max_lag)

    eps = rng.standard_normal((T + max_lag, 4))
    for t in range(max_lag, T + max_lag):
        X0[t] = 0.6 * X0[t - 1] + 0.5 * R[t - 1] + eps[t, 0]
        X1[t] = -0.4 * X0[t - 2] + eps[t, 1]
        X2[t] = 0.3 * R[t] + 0.7 * X2[t - 1] + eps[t, 2]
        X3[t] = 0.6 * X3[t - 1] + eps[t, 3]

    df = pd.DataFrame(
        {
            "R": R[max_lag:].astype(float),
            "X0": X0[max_lag:],
            "X1": X1[max_lag:],
            "X2": X2[max_lag:],
            "X3": X3[max_lag:],
        }
    )

    N = df.shape[1]
    L = max_lag
    G = np.zeros((N, N, L + 1), dtype=np.int8)
    idx = {c: i for i, c in enumerate(df.columns)}

    G[idx["R"], idx["R"], 1] = 1  # R_{t-1} -> R_t
    G[idx["R"], idx["X0"], 1] = 1  # R_{t-1} -> X0_t
    G[idx["X0"], idx["X0"], 1] = 1  # X0_{t-1} -> X0_t
    G[idx["X0"], idx["X1"], 2] = 1  # X0_{t-2} -> X1_t
    G[idx["R"], idx["X2"], 0] = 1  # R_t -> X2_t (contemporaneous)
    G[idx["X2"], idx["X2"], 1] = 1  # X2_{t-1} -> X2_t
    G[idx["X3"], idx["X3"], 1] = 1  # X3_{t-1} -> X3_t

    return {
        "df": df,
        "ground_truth": G,
        "discrete_cols": ["R"],
        "max_lag": max_lag,
    }


def mixed_b(seed=42, T=300):
    """Count variable + continuous variables (4 vars, 1 Poisson count).

    Variables:
        X  : continuous AR(1)
        C  : Poisson count driven by X_t (contemporaneous cause X→C)
        Y  : continuous driven by C_{t-1}
        Z  : continuous independent AR(1)

    True edges:
        X_t      → C_t       (lag 0, contemporaneous)
        X_{t-1}  → X_t       (lag 1, self)
        C_{t-1}  → Y_t       (lag 1)
        Z_{t-1}  → Z_t       (lag 1, self)

    Returns dict with keys: df, ground_truth, discrete_cols, max_lag.
    """
    rng = np.random.default_rng(seed)
    max_lag = 1

    X = np.zeros(T + max_lag)
    Y = np.zeros(T + max_lag)
    Z = np.zeros(T + max_lag)
    C = np.zeros(T + max_lag, dtype=np.int32)

    eps = rng.standard_normal((T + max_lag, 3))
    for t in range(max_lag, T + max_lag):
        X[t] = 0.8 * X[t - 1] + eps[t, 0]
        lam = np.exp(0.3 * X[t])
        C[t] = rng.poisson(lam)
        Y[t] = 0.5 * C[t - 1] + eps[t, 1]
        Z[t] = 0.5 * Z[t - 1] + eps[t, 2]

    df = pd.DataFrame(
        {
            "X": X[max_lag:],
            "C": C[max_lag:].astype(float),
            "Y": Y[max_lag:],
            "Z": Z[max_lag:],
        }
    )

    N = df.shape[1]
    L = max_lag
    G = np.zeros((N, N, L + 1), dtype=np.int8)
    idx = {c: i for i, c in enumerate(df.columns)}

    G[idx["X"], idx["X"], 1] = 1  # X_{t-1} -> X_t
    G[idx["X"], idx["C"], 0] = 1  # X_t -> C_t  (contemporaneous)
    G[idx["C"], idx["Y"], 1] = 1  # C_{t-1} -> Y_t
    G[idx["Z"], idx["Z"], 1] = 1  # Z_{t-1} -> Z_t

    return {
        "df": df,
        "ground_truth": G,
        "discrete_cols": ["C"],
        "max_lag": max_lag,
    }


def mixed_c(seed=42, T=300):
    """Multiple binary variables + continuous variables (6 vars, 2 binary).

    Variables:
        B1 : binary Markov chain (transition prob 0.15)
        B2 : binary Markov chain (transition prob 0.15), independent of B1
        X0 : continuous, driven by B1_{t-1}
        X1 : continuous, driven by B1_{t-1} and B2_{t-1}
        X2 : continuous, driven by X0_{t-2}
        X3 : continuous, driven by X1_{t-1}

    True edges:
        B1_{t-1} → B1_t      (lag 1, self)
        B2_{t-1} → B2_t      (lag 1, self)
        B1_{t-1} → X0_t      (lag 1)
        X0_{t-1} → X0_t      (lag 1, self)
        B1_{t-1} → X1_t      (lag 1)
        B2_{t-1} → X1_t      (lag 1)
        X1_{t-1} → X1_t      (lag 1, self)
        X0_{t-2} → X2_t      (lag 2)
        X2_{t-1} → X2_t      (lag 1, self)
        X1_{t-1} → X3_t      (lag 1)
        X3_{t-1} → X3_t      (lag 1, self)

    Returns dict with keys: df, ground_truth, discrete_cols, max_lag.
    """
    rng = np.random.default_rng(seed)
    max_lag = 2

    trans = np.array([[0.85, 0.15], [0.15, 0.85]])
    B1 = np.zeros(T + max_lag, dtype=np.int32)
    B2 = np.zeros(T + max_lag, dtype=np.int32)
    B1[0] = rng.integers(0, 2)
    B2[0] = rng.integers(0, 2)
    for t in range(1, T + max_lag):
        B1[t] = rng.choice(2, p=trans[B1[t - 1]])
        B2[t] = rng.choice(2, p=trans[B2[t - 1]])

    X0 = np.zeros(T + max_lag)
    X1 = np.zeros(T + max_lag)
    X2 = np.zeros(T + max_lag)
    X3 = np.zeros(T + max_lag)
    eps = rng.standard_normal((T + max_lag, 4))

    for t in range(max_lag, T + max_lag):
        X0[t] = 0.5 * X0[t - 1] + 1.5 * B1[t - 1] + eps[t, 0]
        X1[t] = 0.6 * X1[t - 1] + B2[t - 1] - B1[t - 1] + eps[t, 1]
        X2[t] = 0.4 * X2[t - 1] - 0.8 * X0[t - 2] + eps[t, 2]
        X3[t] = 0.3 * X3[t - 1] + 0.5 * X1[t - 1] + eps[t, 3]

    df = pd.DataFrame(
        {
            "B1": B1[max_lag:].astype(float),
            "B2": B2[max_lag:].astype(float),
            "X0": X0[max_lag:],
            "X1": X1[max_lag:],
            "X2": X2[max_lag:],
            "X3": X3[max_lag:],
        }
    )

    N = df.shape[1]
    L = max_lag
    G = np.zeros((N, N, L + 1), dtype=np.int8)
    idx = {c: i for i, c in enumerate(df.columns)}

    G[idx["B1"], idx["B1"], 1] = 1
    G[idx["B2"], idx["B2"], 1] = 1
    G[idx["B1"], idx["X0"], 1] = 1
    G[idx["X0"], idx["X0"], 1] = 1
    G[idx["B1"], idx["X1"], 1] = 1
    G[idx["B2"], idx["X1"], 1] = 1
    G[idx["X1"], idx["X1"], 1] = 1
    G[idx["X0"], idx["X2"], 2] = 1
    G[idx["X2"], idx["X2"], 1] = 1
    G[idx["X1"], idx["X3"], 1] = 1
    G[idx["X3"], idx["X3"], 1] = 1

    return {
        "df": df,
        "ground_truth": G,
        "discrete_cols": ["B1", "B2"],
        "max_lag": max_lag,
    }


MIXED_DATASETS = {
    "mixed_a": mixed_a,
    "mixed_b": mixed_b,
    "mixed_c": mixed_c,
}
