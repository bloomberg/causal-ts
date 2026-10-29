# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression-based refinement for GRACE: prune candidate parents by backward elimination.

``run_cdnots_gated(refine="shrink")`` replaces the neural gate stage with this. For
each target ``x_e(t)``, its lagged candidate parents (from the skeleton) enter an
additive least-squares regression, one feature block per parent. The weakest parent
is dropped while its block does not significantly improve the fit given the others
(nested partial F-test), and the fit is redone. Nothing is trained and no cutoff is
chosen; every step is a QR decomposition and an F-test.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.linalg import qr
from scipy.stats import f as f_dist

from .grow import _parent_block

_HUBER_C = 1.345


def _huber(r: np.ndarray) -> np.ndarray:
    """Clip residuals at ``1.345`` robust standard deviations (MAD scale).

    When more than half the residuals are tied (zero MAD, e.g. zero-inflated
    data), the scale falls back to the mean absolute deviation from the median
    (times 1.2533, its normal-consistency factor), so a few spikes are still
    clipped rather than deciding the test."""
    dev = np.abs(r - np.median(r))
    s = np.median(dev) / 0.6745
    if not s > 0:
        s = 1.2533 * dev.mean()
    if not s > 0:
        return r
    return np.clip(r, -_HUBER_C * s, _HUBER_C * s)


def _orthonormal(n: int, blocks: list) -> np.ndarray:
    """Orthonormal basis of span(1, blocks) by rank-revealing (pivoted) QR."""
    A = np.column_stack([np.ones(n)] + blocks)
    Q, R, _ = qr(A, mode="economic", pivoting=True, check_finite=False)
    d = np.abs(np.diag(R))
    return Q[:, : int(np.sum(d > d[0] * 1e-8))]


def block_test(
    y: np.ndarray, block: np.ndarray, Q: np.ndarray, robust: bool = True
) -> Tuple[float, float]:
    """p-value and partial R^2 of adding ``block`` to a regression of ``y`` on span(Q).

    With ``robust``, the restricted residual is Huber-clipped and projected off
    span(Q) again before the F statistic is formed, so a few extreme residuals
    cannot decide the test. Directions of the block that add less than 1e-3 of its
    strongest direction beyond span(Q) are dropped: they carry no information but
    sit on a few high-leverage rows. Returns ``(1.0, 0.0)`` when the test is
    undefined (nothing left to test, or fewer than 10 residual degrees of freedom).
    """
    r = y - Q @ (Q.T @ y)
    if robust:
        r = _huber(r)
        r = r - Q @ (Q.T @ r)
    Xp = block - Q @ (Q.T @ block)
    U, sv, _ = np.linalg.svd(Xp, full_matrices=False)
    if not sv.size or sv[0] <= 1e-12:
        return 1.0, 0.0
    return _f_test(r, U[:, sv > sv[0] * 1e-3], Q.shape[1])


def _f_test(r, U, n_nuisance):
    """F p-value and partial R^2 of the orthonormal block ``U`` for residual ``r``.

    ``r`` is the restricted residual (already Huber-clipped and re-projected when
    robust).
    """
    q = U.shape[1]
    df2 = len(r) - n_nuisance - q
    rss0 = float(r @ r)
    if q == 0 or df2 < 10 or rss0 <= 0:  # too few residual degrees of freedom
        return 1.0, 0.0
    beta = U.T @ r
    num = float(beta @ beta)
    rss1 = rss0 - num
    if rss1 <= 1e-12 * rss0:
        return 0.0, 1.0
    stat = (num / q) / (rss1 / df2)
    return float(f_dist.sf(stat, q, df2)), num / rss0


def _product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Standardized product column of two lagged variables (pairwise interaction)."""
    v = a * b
    return ((v - v.mean()) / (v.std() + 1e-12))[:, None]


def _lagged(X: np.ndarray, max_lag: int) -> np.ndarray:
    """``Z[t, c, lag] = x_c(t + max_lag - lag)``, standardized per (variable, lag)."""
    T = X.shape[0]
    Z = np.stack([X[max_lag - lag : T - lag] for lag in range(max_lag + 1)], axis=2)
    return (Z - Z.mean(0)) / (Z.std(0) + 1e-12)


def shrink_parents(
    X: np.ndarray,
    candidates: np.ndarray,
    max_lag: int,
    alpha: float = 0.001,
    robust: bool = True,
    basis: str = "spline",
    exog: Optional[np.ndarray] = None,
    interactions: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward elimination of each target's lagged candidate parents.

    For each target ``e``, ``x_e(t)`` is regressed on one feature block
    (:func:`~causalts.grace.grow.feature_block`) per candidate parent ``x_c(t-l)``
    with ``candidates[c, e, l]`` (lags >= 1), plus ``exog``. While the parent
    whose block is least significant given all the others has p-value above
    ``alpha``, it is removed and the regression refitted.

    Parameters
    ----------
    X : numpy.ndarray
        Data, shape (T, d), no missing values. GRACE passes its
        rank-Gaussianized data.
    candidates : numpy.ndarray
        Candidate graph, boolean-like (d, d, max_lag+1) as ``[cause, effect, lag]``.
        Lag 0 is ignored.
    max_lag : int
        Maximum lag.
    alpha : float
        Significance level for keeping a parent. The p-values come from the F
        distribution with a fixed basis, so they are approximate: when effects are strongly curved they reject true conditional
        independence more often than ``alpha`` says, increasingly so as T grows
        (bias of the fixed-capacity fit, which a bootstrap would not remove). Use
        them to rank edges and for selection; for calibrated tests use the ``raft``
        CI test (``rift`` when conditioning effects interact or are wiggly).
    robust : bool
        Huber-clip the residual before each test (see :func:`block_test`).
        Limits the influence of heavy-tailed and ARCH-type noise at no cost
        under Gaussian noise.
    basis : {"spline", "rff"}
        Feature block per parent. ``"spline"`` (natural cubic spline, knots from
        the sample minimum to maximum), on rank-transformed data, gives extreme rows much lower leverage than ``"rff"`` (the grow
        step's original random-Fourier basis).
    exog : numpy.ndarray or None
        Extra regressors, shape (T, k), aligned with ``X`` (e.g. the C
        nonstationarity basis), kept in every regression.
    interactions : bool
        Add pairwise products of the current parents to every regression: when
        parent j is tested, the products among the other parents stay in the
        nuisance and j's products with them are tested together with its block.
        Removes the false positives an additive fit invites when parents interact
        (a non-parent correlated with them absorbs the leftover interaction, more
        so as T grows), at the cost of degrees of freedom when a target has many
        candidate parents. The number of product columns grows as k(k-1)/2 in the
        number of candidates k, and so does the cost; tests left with fewer than
        10 residual degrees of freedom are undefined, and ties among them are
        broken by the strongest pairwise association.

    Returns
    -------
    graph : numpy.ndarray
        Boolean (d, d, max_lag+1): the surviving parents.
    pvalue : numpy.ndarray
        Same shape. For a surviving parent, the p-value of its block (with its
        products under ``interactions``) given the other survivors; for a removed
        candidate, the p-value it was removed with (given the candidates still in
        the model then; above ``alpha`` by construction); 1.0 where nothing was
        tested (not a candidate, or lag 0). Backward elimination is path
        dependent: a removed candidate may have become significant after later
        removals.
    strength : numpy.ndarray
        Same shape, partial R^2 in [0, 1] from the same tests: the share of the
        residual variance left by the other parents in that model (the final
        survivors for a kept parent, the candidates present at its removal for a
        removed one) that the block explains (on the Huber-clipped residual when
        ``robust``); 0.0 where nothing was tested. Path dependent like the
        p-values. An undefined test gives p-value 1.0 and strength 0.0, a sentinel
        indistinguishable from an untested entry: fewer than 10 residual degrees
        of freedom, a block entirely inside the span of the other regressors
        (e.g. a duplicated candidate), or no residual variation left.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    X = np.asarray(X, dtype=float)
    if np.isnan(X).any():
        raise ValueError("shrink_parents needs data without missing values")
    T, d = X.shape
    L = max_lag
    if not 0 < L < T - 1:
        raise ValueError(f"need 0 < max_lag < T - 1, got max_lag={L}, T={T}")
    cand = np.asarray(candidates).astype(bool)
    if cand.shape != (d, d, L + 1):
        raise ValueError(
            f"candidates must have shape {(d, d, L + 1)}, got {cand.shape}"
        )
    Z = _lagged(X, L)
    n = Z.shape[0]
    base = []
    if exog is not None:
        E = np.asarray(exog, dtype=float).reshape(T, -1)[L:]
        sd = E.std(0)
        base = [
            (E[:, k] - E[:, k].mean()) / sd[k] for k in range(E.shape[1]) if sd[k] > 0
        ]

    graph = np.zeros((d, d, L + 1), dtype=bool)
    pvalue = np.ones((d, d, L + 1))
    strength = np.zeros((d, d, L + 1))
    feats, prods = {}, {}

    def block(p):
        if p not in feats:
            feats[p] = _parent_block(Z, p[0], p[1], basis)
        return feats[p]

    def prod(a, b):
        key = (a, b) if a < b else (b, a)
        if key not in prods:
            prods[key] = _product(
                Z[:, key[0][0], key[0][1]], Z[:, key[1][0], key[1][1]]
            )
        return prods[key]

    def design(kept):
        """Items (blocks, then products) and, per parent, the items tested for it."""
        items = [block(p) for p in kept]
        groups = [[i] for i in range(len(kept))]
        if interactions:
            for a in range(len(kept)):
                for b in range(a + 1, len(kept)):
                    groups[a].append(len(items))
                    groups[b].append(len(items))
                    items.append(prod(kept[a], kept[b]))
        return items, groups

    def direct(items, group):
        """block_test of ``group`` given the other items, refitting directly."""
        rest = [it for i, it in enumerate(items) if i not in group]
        Q = _orthonormal(n, base + rest)
        return block_test(y, np.column_stack([items[i] for i in group]), Q, robust)

    for e in range(d):
        y = Z[:, e, 0]
        # A parent with no feature columns (all its levels merged) is not tested:
        # with interactions it would otherwise enter through its product columns,
        # which are nonzero on only its few rare rows. Its edge_pvalue stays 1.0.
        tested = [
            (c, lag)
            for c in range(d)
            for lag in range(1, L + 1)
            if cand[c, e, lag] and block((c, lag)).shape[1] > 0
        ]
        kept = list(tested)
        # Tie-break for equal p-values (all 1.0 when too many candidates leave no
        # residual degrees of freedom): drop the weakest marginal association
        # first, so the order is data-driven rather than the column order. With
        # interactions, a parent's marginal association is the best pairwise model
        # [p, o, p*o] over the other candidates, so a parent that acts only
        # through an interaction is not dropped first.
        Q0 = _orthonormal(n, base)
        marginal_cache, pair_cache = {}, {}

        def marginal(p):
            # Only needed when p-values tie. With interactions the score is the best
            # pairwise model [p, o, p*o] over the other candidates (each unordered
            # pair evaluated once; O(k^2) tests on first use).
            if p not in marginal_cache:
                m = block_test(y, block(p), Q0, robust)[1]
                if interactions:
                    for o in tested:
                        if o != p:
                            key = (p, o) if p < o else (o, p)
                            if key not in pair_cache:
                                pair = np.column_stack([block(p), block(o), prod(p, o)])
                                pair_cache[key] = block_test(y, pair, Q0, robust)[1]
                            m = max(m, pair_cache[key])
                marginal_cache[p] = m
            return marginal_cache[p]

        scores, removed_scores = {}, {}
        while kept:
            items, groups = design(kept)
            scores = {p: direct(items, g) for p, g in zip(kept, groups)}
            top = max(scores[p][0] for p in kept)
            tied = [p for p in kept if scores[p][0] == top]
            weakest = (
                tied[0] if len(tied) == 1 else max(tied, key=lambda p: -marginal(p))
            )
            if scores[weakest][0] <= alpha:
                break
            kept.remove(weakest)
            removed_scores[weakest] = scores[weakest]
        if not kept:
            scores = {}
        # Removed candidates keep the scores they were removed with (p > alpha by
        # construction), so edge_pvalue never contradicts the selection.
        scores.update(removed_scores)
        for (c, lag), (pv, r2) in scores.items():
            pvalue[c, e, lag], strength[c, e, lag] = pv, r2
        for c, lag in kept:
            graph[c, e, lag] = True
    return graph, pvalue, strength
