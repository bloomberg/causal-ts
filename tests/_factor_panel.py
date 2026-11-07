# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test helper: a factor panel with known residual network.

``X_t = Lambda F_t + E_t`` with ``k`` AR(1) Laplace factors loading densely on ``d``
series, and residuals ``E`` following a sparse VAR(1) (Student-t innovations) plus
optional lag-0 edges drawn as a DAG. ``residual_graph`` is ``(d, d, 2)`` in
``[cause, effect, lag]`` form (self-lags included in slice 1).
"""

import numpy as np
import pandas as pd

_BURN_IN = 200


def factor_panel(
    d=30,
    n_factors=3,
    T=1000,
    *,
    factor_strength=1.0,
    factor_persistence=0.5,
    residual_density=0.05,
    residual_strength=0.4,
    residual_persistence=0.3,
    residual_lag0_density=0.0,
    residual_lag0_strength=0.4,
    seed=0,
):
    rng = np.random.default_rng(seed)
    k, n = n_factors, T + _BURN_IN

    L = rng.standard_normal((d, k)) * factor_strength
    A = np.eye(k) * factor_persistence
    innov_sd = np.sqrt(max(1.0 - factor_persistence**2, 1e-3))
    F = np.zeros((n, k))
    eta = rng.laplace(size=(n, k)) / np.sqrt(2.0) * innov_sd
    for t in range(1, n):
        F[t] = A @ F[t - 1] + eta[t]

    def offdiag(M):
        return M - np.diag(np.diag(M))

    B = np.eye(d) * residual_persistence
    mask = rng.random((d, d)) < residual_density
    np.fill_diagonal(mask, False)
    B[mask] = residual_strength * rng.choice([-1.0, 1.0], size=mask.sum())
    while np.max(np.abs(np.linalg.eigvals(B))) >= 0.95:
        B = np.diag(np.diag(B)) + 0.9 * offdiag(B)
    eps = rng.standard_t(5, size=(n, d)) / np.sqrt(5.0 / 3.0)

    B0 = np.zeros((d, d))
    if residual_lag0_density > 0:
        rng0 = np.random.default_rng([int(seed), 1])
        order = rng0.permutation(d)
        pos = np.empty(d, dtype=int)
        pos[order] = np.arange(d)
        edges = (rng0.random((d, d)) < residual_lag0_density) & (
            pos[None, :] < pos[:, None]
        )
        B0[edges] = residual_lag0_strength * rng0.choice([-1.0, 1.0], size=edges.sum())
    M = np.linalg.inv(np.eye(d) - B0)
    while np.max(np.abs(np.linalg.eigvals(M @ B))) >= 0.95:
        B = np.diag(np.diag(B)) + 0.9 * offdiag(B)
    E = np.zeros((n, d))
    for t in range(1, n):
        E[t] = M @ (B @ E[t - 1] + eps[t])

    F, E = F[_BURN_IN:], E[_BURN_IN:]
    G = np.zeros((d, d, 2), dtype=int)
    G[:, :, 0] = (B0.T != 0).astype(int)
    G[:, :, 1] = (B.T != 0).astype(int)
    names = [f"X{i}" for i in range(d)]
    return {"X": pd.DataFrame(F @ L.T + E, columns=names), "residual_graph": G}
