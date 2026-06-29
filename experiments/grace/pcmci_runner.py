# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Thin adapter: wraps causalts.run_pcmci with the (data_array, max_lag, alpha, ci_test) interface."""

import pandas as pd


def run_pcmci(data, max_lag, alpha=0.05, ci_test="parcorr", **ci_kwargs):
    """Run PCMCI and return a binary causal graph.

    Parameters
    ----------
    data : array [T, n_vars]
    max_lag : int
    alpha : float
        Significance level used for both PC skeleton and MCI thresholding.
    ci_test : str
        CI test name ('parcorr', 'gpdc', 'cmiknn', 'kcit', 'rcot').
    **ci_kwargs
        Extra keyword arguments forwarded to the CI test constructor
        (e.g. num_f, num_f2 for RCOT).

    Returns
    -------
    G : np.ndarray [cause, effect, lag], int8
    info : dict with p_matrix etc.
    """
    from causalts import run_pcmci as _run_pcmci

    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
    G_hat, info = _run_pcmci(
        df,
        tau_max=max_lag,
        ci_test=ci_test,
        pc_alpha=alpha,
        alpha_level=alpha,
        ci_kwargs=ci_kwargs if ci_kwargs else None,
    )
    return G_hat, info
