# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Partial correlation adjusted for leave-out common factors (private to LUCID).

Used only by ``run_lucid(pervasive_lagged="factor_adjusted")``; not a general
conditional-independence test and not registered with ``create_ci_test``.

``FactorParCorr`` tests ``X _|_ Y | Z, F`` where ``F`` are principal-component factor
scores. The factors for a test are estimated from all observed series except the ones
``X`` and ``Y`` belong to, and only scores at the older of the two time points and
earlier enter ``F``.

Why leave them out: residuals after removing ``k`` estimated factors satisfy ``k``
exact linear constraints. When loadings are block-sparse, a series' residual is then an
exact combination of a few groupmates' residuals. Conditioning on those groupmates
makes any true edge from it look like independence. Factors built without ``X`` and
``Y`` are not combinations of them, so no exact collinearity can arise.

Why only past scores: a factor score at time ``t`` is a weighted sum of every series
at ``t``, including the children of a lagged cause, so conditioning on it opens
collider paths and creates spurious lagged edges. Scores at the cause's own time and
earlier still block confounding through a persistent factor.

Limitations: the factor proxies are estimated, so confounding by a persistent factor is
removed only approximately; the leftover grows when a factor loads on few series and the
more persistent the factor is. Under the null, rejection at alpha 0.05 reaches about 0.2
with 4 series per factor and a persistent factor (T = 1000), and grows with T. Treat
alpha here as a graph-tuning parameter rather than a calibrated error rate; at graph level
this test still beats the calibrated alternatives tried (two-proxy IV and
errors-in-variables corrections), which lose too much power.
Scores at the cause's own time can still contain its lag-0 children, a small collider
effect when the factors are pervasive (each series has weight of order 1/d).
"""

from __future__ import annotations

import numpy as np
from scipy import special

from ..ci_tests.cit_test import CIT_Base

# Separate memory bounds for the per-pair caches. Loadings are small (d x k) and
# costly to refit (an SVD); scores are large (T x k per lag) and cheap to rebuild,
# so large scores must never crowd loadings out.
_LOADINGS_BYTES = 512 * 2**20
_SCORES_BYTES = 256 * 2**20
# Charged per cache entry for the Python objects around its arrays.
_ENTRY_OVERHEAD = 1024


class _BudgetCache(dict):
    """``key -> (value, size)`` that keeps entries only while they fit a byte budget.

    Nothing is evicted: once full, new entries are simply not stored. CDNOTS
    revisits pairs cyclically, and an LRU over a cyclic pattern larger than
    itself gets no hits; keeping a fixed subset instead makes the cost of
    overflow proportional to how far the working set exceeds the budget.
    """

    nbytes = 0

    def get_value(self, key):
        hit = self.get(key)
        return (False, None) if hit is None else (True, hit[0])

    def put(self, key, value, arrays, budget):
        """Store (or replace) ``value`` if it fits; otherwise leave the cache as is."""
        size = _ENTRY_OVERHEAD + sum(a.nbytes for a in arrays if a is not None)
        old = self[key][1] if key in self else 0
        if self.nbytes - old + size <= budget:
            self[key] = (value, size)
            self.nbytes += size - old


class FactorParCorr(CIT_Base):
    """Partial-correlation CI test that partials out leave-out PCA factors.

    Parameters
    ----------
    n_factors : int
        Number of common factors to partial out at each conditioning lag.
    n_obs : int, optional
        Number of observed series per lag block; columns past them (e.g. CDNOTS's
        time-index node) are never used to estimate factors. Default: every column
        of the lag block.
    data : ndarray, optional
        ``(T, n_vars * n_lags)`` lag-embedded data; CDNOTS sets it.
    """

    def __init__(self, n_factors, n_obs=None, data=None, **kwargs):
        if int(n_factors) < 1:
            raise ValueError("n_factors must be >= 1")
        self.n_factors = int(n_factors)
        self.n_obs = None if n_obs is None else int(n_obs)
        self._n_vars = None
        self._lags = [0]
        self._reset_cache()
        super().__init__(data=data, method="factor_parcorr", **kwargs)
        self.param_hash = f"k={self.n_factors},past"

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        # CDNOTS swaps arrays on one instance; cached p-values and factors are keyed
        # by column index, so they must not survive a data change.
        self._data = value
        self._reset_cache()
        self._has_nan = value is not None and bool(np.isnan(value).any())
        if hasattr(self, "pvalue_cache"):
            self.pvalue_cache = {}

    def set_lag_structure(self, n_vars, lags):
        self._n_vars = int(n_vars)
        self._lags = list(lags)
        self._reset_cache()
        self.pvalue_cache = {}

    def _reset_cache(self):
        self._loadings_cache = _BudgetCache()
        self._scores_cache = _BudgetCache()

    def _layout(self):
        n_vars = self._n_vars if self._n_vars is not None else self.data.shape[1]
        if self.data.shape[1] != n_vars * len(self._lags):
            raise ValueError(
                f"data has {self.data.shape[1]} columns, expected n_vars * n_lags = "
                f"{n_vars} * {len(self._lags)}"
            )
        n_obs = n_vars if self.n_obs is None else min(self.n_obs, n_vars)
        return n_vars, n_obs

    def _loadings(self, exclude):
        """``(keep, mu, sd, W)`` for observed series not in ``exclude``, or None.

        Memoised per left-out pair up to ``_LOADINGS_BYTES`` (about 54 MB at
        d = 100, k = 12, so realistic panels never refit); pairs past the budget
        are refit on each use, never answered differently.
        """
        key = tuple(sorted(exclude))
        hit, fit = self._loadings_cache.get_value(key)
        if hit:
            return fit
        _, n_obs = self._layout()
        keep = np.array([v for v in range(n_obs) if v not in exclude], dtype=np.intp)
        k = min(self.n_factors, len(keep))
        if k < 1:
            fit = None
            self._loadings_cache.put(key, fit, [], _LOADINGS_BYTES)
        else:
            B0 = self.data[:, keep]
            mu, sd = B0.mean(0), B0.std(0)
            sd[sd == 0] = 1.0
            _, _, Vt = np.linalg.svd((B0 - mu) / sd, full_matrices=False)
            fit = (keep, mu, sd, Vt[:k].T.copy())
            self._loadings_cache.put(key, fit, fit, _LOADINGS_BYTES)
        return fit

    def _factor_scores(self, exclude, min_lag):
        """Factor scores for each lag block with lag >= ``min_lag``.

        A left-out pair's lag blocks are scored on first use and kept, up to
        ``_SCORES_BYTES``, in their own cache. CDNOTS returns to a series pair once
        per lagged column pair, so stored scores are reused; for pairs past the
        budget a test costs what an uncached one does (one projection per needed
        block), never another SVD.
        """
        key = tuple(sorted(exclude))
        hit, blocks = self._scores_cache.get_value(key)
        fit = None
        if not hit:
            fit = self._loadings(exclude)
            blocks = [] if fit is None else [None] * len(self._lags)
        if not blocks:
            if not hit:
                self._scores_cache.put(key, blocks, [], _SCORES_BYTES)
            return []
        need = [b for b, lag in enumerate(self._lags) if lag >= min_lag]
        missing = [b for b in need if blocks[b] is None]
        if missing:
            blocks = list(blocks)  # the stored list changes only if put() accepts it
            keep, mu, sd, W = fit if fit is not None else self._loadings(exclude)
            n_vars, _ = self._layout()
            for b in missing:
                blocks[b] = ((self.data[:, b * n_vars + keep] - mu) / sd) @ W
        if missing or not hit:
            self._scores_cache.put(key, blocks, blocks, _SCORES_BYTES)
        return [blocks[b] for b in need]

    def __call__(self, X, Y, condition_set=None):
        self.n_tests += 1
        Z = [] if condition_set is None else [int(z) for z in condition_set]
        if not (np.isscalar(X) and np.isscalar(Y)):
            raise TypeError("FactorParCorr tests single columns X and Y")
        X, Y = int(X), int(Y)
        if self._has_nan:
            raise ValueError("FactorParCorr does not support missing values")
        key = (min(X, Y), max(X, Y), tuple(sorted(Z)), self.param_hash)
        if key in self.pvalue_cache:
            return self.pvalue_cache[key]
        self.n_actual_tests += 1
        n_vars, _ = self._layout()
        oldest = max(self._lags[X // n_vars], self._lags[Y // n_vars])
        F = self._factor_scores({X % n_vars, Y % n_vars}, oldest)
        n = self.data.shape[0]
        C = np.column_stack([np.ones(n), self.data[:, Z], *F])
        xy = self.data[:, [X, Y]]
        beta, _, rank, _ = np.linalg.lstsq(C, xy, rcond=None)
        r = xy - C @ beta
        dof = n - rank - 1
        denom = np.sqrt((r[:, 0] @ r[:, 0]) * (r[:, 1] @ r[:, 1]))
        if dof < 1:
            raise ValueError(
                f"FactorParCorr: {n} rows cannot support {rank} conditioning columns "
                "(conditioning set, factors and intercept); use fewer factors or lags"
            )
        if denom <= 0:  # X or Y is a linear function of the conditioning set
            out = (1.0, 0.0)
        else:
            rho = float(np.clip(r[:, 0] @ r[:, 1] / denom, -1.0, 1.0))
            t = rho * np.sqrt(dof / max(1.0 - rho**2, 1e-300))
            # stdtr is what scipy's t.sf evaluates, without the distribution overhead.
            out = (float(2 * special.stdtr(dof, -abs(t))), rho)
        self.pvalue_cache[key] = out
        return out
