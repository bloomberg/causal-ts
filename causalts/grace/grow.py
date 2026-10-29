# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Grow step for GRACE: propose lagged edges the candidate skeleton missed."""

import warnings
from typing import Optional, Tuple

import numpy as np
from scipy.stats import f as f_dist
from scipy.stats import norm, rankdata

_N_RFF = 8
_RFF_SEED = 12345


def _rff_basis(x: np.ndarray) -> np.ndarray:
    """``x`` plus 8 random Fourier features of an RBF kernel (bandwidth 1).

    The draws are fixed, so the basis is the same on every call. It is generic
    on purpose rather than matched to any data-generating process.
    """
    rng = np.random.default_rng(_RFF_SEED)
    w = rng.normal(size=_N_RFF)
    b = rng.uniform(0, 2 * np.pi, size=_N_RFF)
    return np.column_stack([x, np.sqrt(2 / _N_RFF) * np.cos(np.outer(x, w) + b)])


def spline_basis(
    x: np.ndarray, n_knots: int = 8, lo: float = 0.0, hi: float = 1.0
) -> np.ndarray:
    """Natural cubic spline basis of ``x`` without intercept, columns standardized.

    ``x`` plus ``K - 2`` truncated-power terms (Hastie, Tibshirani and Friedman,
    *The Elements of Statistical Learning*, eqs. 5.4-5.5), with ``K`` distinct knots
    at the ``lo`` .. ``hi`` quantiles of ``x`` (``K <= n_knots``; tied quantiles are
    merged). The default boundary knots are the sample minimum and maximum, so the
    fit is cubic across the whole observed range. With inner boundary knots (e.g.
    5%/95%) the fit is linear beyond them, which misses curvature in the tails and
    makes the partial F-tests that use this basis reject true conditional
    independence far too often at large T. On rank-transformed data (GRACE's
    default) extreme rows get much lower leverage than with the random-Fourier
    basis, whose near-collinear columns let single rows be fitted almost exactly.
    On raw data an outlier can dominate: with the last knot at the maximum, every
    truncated column loads on the most extreme row, whose leverage then approaches
    1, so rank-transform inputs with outliers first. Variables with at most
    ``n_knots`` distinct values (binary or few-level) get one indicator column per
    level but the first, so every level's effect is representable whatever the
    knot grid would have hit; levels seen in fewer than 5 rows are first merged
    into the smaller adjacent group, since an indicator for a level seen in
    m rows gives each of them leverage 1/m (exactly 1 for a level seen once). A
    variable whose levels all merge into one group (e.g. a binary variable whose
    rare level has fewer than 5 rows) therefore gets no columns and cannot be
    selected; :func:`grow_candidates` and
    :func:`~causalts.grace.shrink.shrink_parents` warn, naming it. A continuous variable with
    fewer than three distinct knots (heavily concentrated) gets its linear term
    only.
    """
    levels, counts = np.unique(x, return_counts=True)
    if len(levels) < 2:
        return np.zeros((len(x), 0))
    if len(levels) <= n_knots:  # discrete: level indicators, sparse levels merged
        groups = _merge_sparse_levels(counts, 5)
        if groups.max() == 0:
            return np.zeros((len(x), 0))
        g = groups[np.searchsorted(levels, x)]
        A = (g[:, None] == np.arange(1, groups.max() + 1)[None, :]).astype(float)
        return (A - A.mean(0)) / A.std(0)
    k = np.unique(np.quantile(x, np.linspace(lo, hi, n_knots)))
    if len(k) < 3:  # continuous but concentrated: linear term only
        A = x[:, None]
        return (A - A.mean(0)) / A.std(0)
    n_knots = len(k)

    def d(j):
        return (np.clip(x - k[j], 0, None) ** 3 - np.clip(x - k[-1], 0, None) ** 3) / (
            k[-1] - k[j]
        )

    A = np.column_stack([x] + [d(j) - d(n_knots - 2) for j in range(n_knots - 2)])
    sd = A.std(0)
    keep = sd > 1e-12 * max(sd.max(), 1e-300)
    return (A[:, keep] - A[:, keep].mean(0)) / sd[keep]


def _merge_sparse_levels(counts: np.ndarray, min_count: int) -> np.ndarray:
    """Group index per ordered level: repeatedly merge the sparsest group with fewer
    than ``min_count`` rows into its smaller neighbour, until none is left (or one
    group remains). Returns group ids 0..G-1 in level order."""
    groups = [[i] for i in range(len(counts))]
    size = [int(c) for c in counts]
    while len(groups) > 1 and min(size) < min_count:
        i = int(np.argmin(size))
        if i == 0:
            j = 1
        elif i == len(groups) - 1:
            j = i - 1
        else:
            j = i - 1 if size[i - 1] <= size[i + 1] else i + 1
        lo, hi = min(i, j), max(i, j)
        groups[lo] = groups[lo] + groups[hi]
        size[lo] += size[hi]
        del groups[hi], size[hi]
    out = np.empty(len(counts), dtype=int)
    for g, members in enumerate(groups):
        out[members] = g
    return out


def feature_block(x: np.ndarray, basis: str = "rff") -> np.ndarray:
    """Additive feature block of one (standardized) lagged variable.

    ``"rff"``: :func:`_rff_basis`. ``"spline"``: :func:`spline_basis`.
    """
    if basis == "rff":
        return _rff_basis(x)
    if basis == "spline":
        return spline_basis(x)
    raise ValueError(f"basis must be 'rff' or 'spline', got {basis!r}")


def _parent_block(Z: np.ndarray, c: int, lag: int, basis: str) -> np.ndarray:
    """:func:`feature_block` of ``x_c(t-lag)``, warning if a varying one gets none."""
    x = Z[:, c, lag]
    phi = feature_block(x, basis)
    if phi.shape[1] == 0 and len(np.unique(x[np.isfinite(x)])) > 1:
        levels, counts = np.unique(x[np.isfinite(x)], return_counts=True)
        warnings.warn(
            f"variable {c} at lag {lag} gets no {basis} feature columns and cannot be "
            f"selected as a parent: its {len(levels)} levels (row counts "
            f"{counts.tolist()}) merge into one group, since levels seen in fewer "
            "than 5 rows are merged into a neighbour.",
            UserWarning,
            stacklevel=3,
        )
    return phi


_NUISANCE_BASES = (None, "rff", "spline")


def _check_nuisance(nuisance_basis):
    if nuisance_basis not in _NUISANCE_BASES:
        raise ValueError(
            f"nuisance_basis must be None, 'rff' or 'spline', got {nuisance_basis!r}"
        )


def _nuisance_block(x: np.ndarray, nuisance_basis: Optional[str], basis: str):
    """Feature block of a conditioning (parent) variable in the grow-type tests.

    ``None``: the tested block's basis (``feature_block(x, basis)``). ``"rff"``: the
    RFF block. ``"spline"``: natural cubic spline with boundary knots at the sample
    min/max (:func:`spline_basis` defaults). With a fixed random-Fourier block, the
    part of a parent's effect it cannot fit becomes significant at large T, and a
    candidate that proxies the same driver absorbs it: on null configurations
    (T=1000, driver autocorrelation 0.9) the tests rejected up to 32% of true nulls
    at level 0.001, with the full-range spline about 2% or less.

    Under ``"spline"`` a discrete variable (at most 8 levels) gets one indicator
    per level but the first, without :func:`spline_basis`'s merging of rare
    levels: as a tested block a rare level's leverage matters, as nuisance every
    level's effect has to be removed.
    """
    if nuisance_basis is None:
        return feature_block(x, basis)
    if nuisance_basis == "rff":
        return _rff_basis(x)
    if _is_discrete(x):
        return _level_indicators(x)
    return spline_basis(x)


def _is_discrete(x: np.ndarray, max_levels: int = 8) -> bool:
    return len(np.unique(x)) <= max_levels


def _level_indicators(x: np.ndarray) -> np.ndarray:
    """Standardized indicators of every level of ``x`` but the first."""
    levels = np.unique(x)
    A = (x[:, None] == levels[None, 1:]).astype(float)
    return (A - A.mean(0)) / A.std(0) if A.shape[1] else A


def _nuisance_features(Z, feats, nuisance_basis, basis):
    """Parent (nuisance) blocks for the grow-type tests, keyed like ``feats``.

    ``None``: ``feats`` itself (the old behaviour). The tested basis named
    explicitly: ``feats`` too (same features, no duplicate warnings), except that
    under ``"spline"`` discrete variables get :func:`_nuisance_block`'s unmerged
    level indicators. Another basis: :func:`_nuisance_block` for every variable.
    """
    if nuisance_basis is None:
        return feats
    if nuisance_basis == basis:
        return {
            cl: (
                _nuisance_block(Z[:, cl[0], cl[1]], nuisance_basis, basis)
                if basis == "spline" and _is_discrete(Z[:, cl[0], cl[1]])
                else phi
            )
            for cl, phi in feats.items()
        }
    return {
        cl: _nuisance_block(Z[:, cl[0], cl[1]], nuisance_basis, basis) for cl in feats
    }


def _lag0_parent_blocks(Z, est, e, nuisance_basis, basis) -> list:
    """Blocks of ``e``'s kept contemporaneous parents: ``c(t)`` with ``est[c, e, 0]``
    and not ``est[e, c, 0]`` (an edge kept both ways may point to a child, and
    conditioning on a child would create dependence)."""
    d = est.shape[0]
    return [
        _nuisance_block(Z[:, c, 0], nuisance_basis, basis)
        for c in range(d)
        if c != e and est[c, e, 0] and not est[e, c, 0]
    ]


def _exog_columns(exog: Optional[np.ndarray], T: int, L: int) -> list:
    """Standardized columns of ``exog`` (T, k) aligned with the lagged design (rows
    ``L..T-1``); constant columns are dropped."""
    if exog is None:
        return []
    E = np.asarray(exog, dtype=float).reshape(T, -1)[L:]
    sd = E.std(0)
    return [(E[:, k] - E[:, k].mean()) / sd[k] for k in range(E.shape[1]) if sd[k] > 0]


def _bh_reject(p: np.ndarray, q: float) -> np.ndarray:
    """Boolean mask of Benjamini-Hochberg rejections at level ``q``."""
    m = len(p)
    order = np.argsort(p)
    below = np.nonzero(p[order] <= q * np.arange(1, m + 1) / m)[0]
    reject = np.zeros(m, dtype=bool)
    if len(below):
        reject[order[: below[-1] + 1]] = True
    return reject


def _block_df(phi: np.ndarray, Q: np.ndarray) -> int:
    """Numerator degrees of freedom of block ``phi`` given orthonormal ``Q``.

    The rank of the projected block, as ``lstsq`` fits it; 0 when the block lies
    inside the span of ``Q`` (e.g. a constant variable's RFF block, within the
    intercept, or a copy of a parent whose nuisance block spans the tested one;
    a continuous parent's spline block does not span a copy's RFF block). That
    check is relative to ``phi`` itself: projection leaves
    rounding residue, which a rank relative to the residue counts as directions.
    """
    s0 = np.linalg.norm(phi, 2)
    s = np.linalg.svd(phi - Q @ (Q.T @ phi), compute_uv=False)
    tol = max(phi.shape) * np.finfo(float).eps
    if s0 == 0 or not np.any(s > s0 * tol):
        return 0
    return int(np.sum(s > s.max() * tol))  # matrix_rank's default tolerance


def _adjusted_r2(rss0: float, rss1: float, df1: int, df2: int) -> float:
    """Adjusted partial R^2 of a block: ``1 - (rss1/df2) / (rss0/(df1+df2))``.

    The share of the residual variance (after the nuisance) the block explains, corrected
    for the fit a ``df1``-column block gets by chance (raw partial R^2 is biased up by
    about ``df1/n``), so the same floor means the same effect at any series length."""
    return 1.0 - (rss1 / df2) / (rss0 / (df1 + df2))


def _added(tests, pvals, effects, q, effect_floor, d, L):
    """BH rejections at ``q`` on the original p-values, intersected with
    ``effect >= effect_floor`` when the floor is positive (adjusted R^2 can be
    negative, so a 0 floor must not filter at all)."""
    added = np.zeros((d, d, L + 1), dtype=bool)
    if tests:
        reject = _bh_reject(np.array(pvals), q)
        if effect_floor > 0:
            reject &= np.array(effects) >= effect_floor
        for (c, e, lag), hit in zip(tests, reject):
            added[c, e, lag] = hit
    return added


def grow_candidates(
    X: np.ndarray,
    est: np.ndarray,
    skeleton: np.ndarray,
    max_lag: int,
    q: float = 0.1,
    transform: Optional[str] = None,
    exog: Optional[np.ndarray] = None,
    basis: str = "rff",
    interactions: bool = False,
    nuisance_basis: Optional[str] = None,
    effect_floor: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """Test every lagged pair outside ``skeleton`` against the current model.

    For each target ``e``, ``x_e(t)`` is regressed on an additive basis
    (:func:`feature_block`) of its current lagged parents in ``est``. Each lagged
    ``x_c(t-l)`` not in ``skeleton`` is then tested with a nested partial
    F-test of adding its basis block, projected off the parents' features.
    Rejections at Benjamini-Hochberg level ``q`` across all tests are returned.

    Parameters
    ----------
    X : numpy.ndarray
        Data, shape (T, d).
    est : numpy.ndarray
        Current graph, boolean (d, d, max_lag+1) as ``[cause, effect, lag]``.
        Lags >= 1 are parents, and lag-0 entries ``c -> e`` whose reverse
        ``e -> c`` is not also set (contemporaneous parents).
    skeleton : numpy.ndarray
        Candidate set, same shape. Pairs in it are not tested.
    max_lag : int
        Maximum lag.
    q : float
        Benjamini-Hochberg level.
    transform : {None, "rank"}
        ``"rank"`` maps each variable to normal scores of its ranks before
        testing. Heavy-tailed or ARCH-type noise otherwise makes the F-test
        propose many false edges, because a few extreme values dominate the fit.
    exog : numpy.ndarray or None
        Extra regressors, shape (T, k), aligned with ``X`` (e.g. the C
        nonstationarity basis). Every regression and test conditions on them.
    basis : {"rff", "spline"}
        Feature block per lagged variable (:func:`feature_block`). ``"rff"``
        (default) is the original basis; ``"spline"`` is used by
        ``run_cdnots_gated(refine="shrink")``.
    interactions : bool
        Also condition on the pairwise products of the target's current parents,
        so a candidate correlated with interacting parents is not added for the
        interaction an additive fit leaves behind. This suppresses false
        additions only: a candidate is still tested by its own block, so a
        missing parent that acts purely through an interaction is not recovered.
        Default False (unchanged).
    nuisance_basis : {None, "rff", "spline"}
        Basis of the current parents (the nuisance). ``None`` (default): the same
        as ``basis`` (kept lag-0 parents enter the nuisance in any case).
        ``"spline"``: full-range natural cubic spline
        (:func:`_nuisance_block`), which keeps the test near its level at large T
        when parents are persistent. The tested block always uses ``basis``.
    effect_floor : float
        Minimum adjusted partial R^2 of the tested block (:func:`_adjusted_r2`): a
        candidate is added only if BH rejects it AND its effect reaches the floor, so at
        large T a significant edge that explains almost nothing is not added. 0
        (default) adds every BH rejection.

    Returns
    -------
    added : numpy.ndarray
        Boolean (d, d, max_lag+1) of edges to add (lags >= 1 only).
    n_tests : int
        Number of pairs tested.
    """
    if basis not in ("rff", "spline"):
        raise ValueError(f"basis must be 'rff' or 'spline', got {basis!r}")
    _check_nuisance(nuisance_basis)
    if transform == "rank":
        X = norm.ppf((np.apply_along_axis(rankdata, 0, X) - 0.5) / len(X))
    elif transform is not None:
        raise ValueError(f"transform must be None or 'rank', got {transform!r}")
    T, d = X.shape
    L = max_lag
    # lagged[t, c, l] = x_c(t + L - l), standardized per (variable, lag)
    Z = np.stack([X[L - lag : T - lag] for lag in range(L + 1)], axis=2)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    n = Z.shape[0]
    base = [np.ones(n)] + _exog_columns(exog, T, L)
    feats = {
        (c, lag): _parent_block(Z, c, lag, basis)
        for c in range(d)
        for lag in range(1, L + 1)
    }

    nfeats = _nuisance_features(Z, feats, nuisance_basis, basis)

    tests, pvals, effects = [], [], []
    for e in range(d):
        parents = [(c, lag) for c, lag in feats if est[c, e, lag]]
        prods = []
        if interactions:
            for i, a in enumerate(parents):
                for b in parents[i + 1 :]:
                    v = Z[:, a[0], a[1]] * Z[:, b[0], b[1]]
                    prods.append(((v - v.mean()) / (v.std() + 1e-12))[:, None])
        # Rank-revealing basis of the nuisance span: exactly dependent columns
        # (duplicated or collinear regressors) must not remove extra directions.
        lag0 = _lag0_parent_blocks(Z, est, e, nuisance_basis, basis)
        A = np.column_stack(base + [nfeats[p] for p in parents] + lag0 + prods)
        U, s, _ = np.linalg.svd(A, full_matrices=False)
        Q = U[:, s > s[0] * max(A.shape) * np.finfo(float).eps]
        y = Z[:, e, 0]
        r = y - Q @ (Q.T @ y)
        rss0 = r @ r
        for (c, lag), phi in feats.items():
            if skeleton[c, e, lag] or phi.shape[1] == 0:  # 0 cols: no levels
                continue
            df1 = _block_df(phi, Q)
            phi = phi - Q @ (Q.T @ phi)
            coef, *_ = np.linalg.lstsq(phi, r, rcond=None)
            rss1 = np.sum((r - phi @ coef) ** 2)
            df2 = n - Q.shape[1] - df1
            # No residual degrees of freedom (short series, many parents): the
            # test is undefined, so leave it out of the BH family entirely.
            if df1 == 0 or df2 <= 0 or rss1 <= 0:
                continue
            stat = ((rss0 - rss1) / df1) / (rss1 / df2)
            tests.append((c, e, lag))
            pvals.append(f_dist.sf(stat, df1, df2))
            effects.append(_adjusted_r2(rss0, rss1, df1, df2))
    return _added(tests, pvals, effects, q, effect_floor, d, L), len(tests)


def grow_variance_candidates(
    X: np.ndarray,
    est: np.ndarray,
    skeleton: np.ndarray,
    max_lag: int,
    q: float = 0.1,
    transform: Optional[str] = "rank",
    *,
    basis: str = "rff",
    nuisance_basis: Optional[str] = "spline",
    exog: Optional[np.ndarray] = None,
    effect_floor: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """Propose lagged edges that act on a target's noise scale rather than its mean.

    Experimental; the p-values are approximate (the response is built from fitted
    residuals, so the F reference is not exact; in null checks with persistent
    proxies, heavy tails and a variance parent among the parents, it rejected
    2.5-7.5% at level 0.05). For each target ``e``, ``x_e(t)`` is regressed on an additive
    basis of its current parents in ``est``, and the size of the residual
    (normal scores of its absolute value) becomes the response. Each lagged
    ``x_c(t-l)`` outside ``skeleton`` is then tested with the same nested
    partial F-test as :func:`grow_candidates`, with the parents' features as
    nuisance in both regressions, so a parent that already drives the target's
    variance does not make its proxies look like new edges. Rejections at
    Benjamini-Hochberg level ``q`` are returned. Mean-based candidate steps
    (CDNOTS, PCMCI+, :func:`grow_candidates`) cannot see a parent that changes
    only the noise scale; a likelihood-based refinement can use one.

    Parameters
    ----------
    X : numpy.ndarray
        Data, shape (T, d), without missing values.
    est : numpy.ndarray
        Current graph, boolean (d, d, max_lag+1) as ``[cause, effect, lag]``;
        parents as in :func:`grow_candidates` (lags >= 1, and lag-0 entries
        whose reverse is not also set).
    skeleton : numpy.ndarray
        Candidate set, same shape. Pairs in it are not tested.
    max_lag : int
        Maximum lag.
    q : float
        Benjamini-Hochberg level across all tests.
    transform : {"rank", None}
        ``"rank"`` (default) maps each variable to normal scores of its ranks first.
    basis : {"rff", "spline"}
        Feature block of the tested candidate (:func:`feature_block`).
    nuisance_basis : {"spline", "rff", None}
        Basis of the parents, as in :func:`grow_candidates` (default ``"spline"``).
    exog : numpy.ndarray or None
        Extra regressors, shape (T, k), aligned with ``X`` (e.g. the C
        nonstationarity basis); they enter both regressions.
    effect_floor : float
        Minimum adjusted partial R^2 of the tested block on the scale response, as in
        :func:`grow_candidates`. 0 (default) adds every BH rejection.

    Returns
    -------
    added : numpy.ndarray
        Boolean (d, d, max_lag+1) of edges to add (lags >= 1 only).
    n_tests : int
        Number of pairs tested.
    """
    _check_nuisance(nuisance_basis)
    if transform == "rank":
        X = norm.ppf((np.apply_along_axis(rankdata, 0, X) - 0.5) / len(X))
    elif transform is not None:
        raise ValueError(f"transform must be None or 'rank', got {transform!r}")
    T, d = X.shape
    L = max_lag
    Z = np.stack([X[L - lag : T - lag] for lag in range(L + 1)], axis=2)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    n = Z.shape[0]
    feats = {
        (c, lag): feature_block(Z[:, c, lag], basis)
        for c in range(d)
        for lag in range(1, L + 1)
    }
    nfeats = _nuisance_features(Z, feats, nuisance_basis, basis)
    ex = _exog_columns(exog, T, L)
    tests, pvals, effects = [], [], []
    for e in range(d):
        parents = [(c, lag) for c, lag in feats if est[c, e, lag]]
        lag0 = _lag0_parent_blocks(Z, est, e, nuisance_basis, basis)
        A = np.column_stack([np.ones(n)] + ex + [nfeats[p] for p in parents] + lag0)
        U, s, _ = np.linalg.svd(A, full_matrices=False)
        Q = U[:, s > s[0] * max(A.shape) * np.finfo(float).eps]
        y = Z[:, e, 0]
        r = y - Q @ (Q.T @ y)
        v = norm.ppf((rankdata(np.abs(r)) - 0.5) / n)
        rv = v - Q @ (Q.T @ v)
        rss0 = rv @ rv
        for (c, lag), phi in feats.items():
            if skeleton[c, e, lag] or phi.shape[1] == 0:
                continue
            df1 = _block_df(phi, Q)
            phi = phi - Q @ (Q.T @ phi)
            coef, *_ = np.linalg.lstsq(phi, rv, rcond=None)
            rss1 = np.sum((rv - phi @ coef) ** 2)
            df2 = n - Q.shape[1] - df1
            if df1 == 0 or df2 <= 0 or rss1 <= 0:
                continue
            stat = ((rss0 - rss1) / df1) / (rss1 / df2)
            tests.append((c, e, lag))
            pvals.append(f_dist.sf(stat, df1, df2))
            effects.append(_adjusted_r2(rss0, rss1, df1, df2))
    return _added(tests, pvals, effects, q, effect_floor, d, L), len(tests)


def self_loop_test(
    X: np.ndarray,
    cross: np.ndarray,
    max_lag: int,
    alpha: float = 1e-3,
    transform: Optional[str] = "rank",
    nuisance_basis: Optional[str] = "spline",
    exog: Optional[np.ndarray] = None,
    return_pvalues: bool = False,
    test_basis: str = "union",
):
    """Decide self-loops with a per-lag partial F-test (used by ``ar_path``).

    For each target ``e`` and lag ``l >= 1``, own lag ``x_e(t-l)`` is kept if
    adding its tested block (``test_basis``) to an additive regression of ``x_e(t)`` on the kept
    cross parents in ``cross`` (lags >= 1, and lag-0 parents whose reverse lag-0
    edge is not also in ``cross``; diagonal ignored), the other own lags and
    ``exog`` gives p < ``alpha``. A mis-oriented lag-0 edge (a child kept as a
    parent) can still create a spurious self-loop.
    Conditioning on the cross parents keeps an own lag that only stands in for an
    autocorrelated parent from being kept. It tests the conditional mean: an own
    lag that acts only on the noise scale (ARCH-type) is not detected. The
    p-values come from a fixed-basis F-test and are approximate.

    Parameters
    ----------
    X : numpy.ndarray
        Data, shape (T, d), without missing values.
    cross : numpy.ndarray
        Boolean (d, d, max_lag+1) cross-parent graph ``[cause, effect, lag]``.
    max_lag : int
        Maximum lag (>= 1).
    alpha : float
        Nominal level of each per-lag test, not of the family: with every
        self-loop absent and the null well specified, about
        ``alpha * d * max_lag`` false self-loops are expected (0.2 at d=200,
        max_lag=1). The p-values are approximate, so actual rates can be higher
        (4.4% per series on ranked trend-only data with the 4-knot C basis). A family-wise correction would cost power in
        autoregressive systems, where most self-loops are real.
    transform : {"rank", None}
        ``"rank"`` maps each variable to normal scores of its ranks first.
    nuisance_basis : {"spline", "rff", None}
        Basis of the conditioning terms (cross parents and the other own lags);
        ``None`` and ``"rff"`` use the RFF block. See :func:`grow_candidates` for
        why ``"spline"`` is the default.
    exog : numpy.ndarray or None
        Extra regressors, shape (T, k), aligned with ``X``.
    return_pvalues : bool
        Also return the p-values (NaN where a test was undefined; a
        ``UserWarning`` reports how many were).
    test_basis : {"union", "rff"}, default "union"
        Block of the tested own lag. ``"union"``: two F-tests, the own lag alone
        (1 df) and a natural cubic spline of it (4 knots), Bonferroni over the
        tests that add degrees of freedom: the reported p-value is
        ``min(1, k * min(p))`` with ``k`` the number of tests run, i.e.
        ``2 * min(p_linear, p_spline)`` normally, and the single p-value when the
        spline collapses to the linear term (tied quantiles) or one test is
        undefined; a discrete variable (at most 8 levels) gets one test of its
        level indicators at ``alpha`` (levels seen in fewer than 5 rows merged
        into a neighbour, as in :func:`spline_basis`). ``"rff"``: the
        9-column RFF block, the behaviour before this option. The RFF block's
        degrees of freedom cost power at small T: on simulated systems at T=250
        it kept about half the true self-loops, the union about three quarters,
        with about as many false self-loops, and the union was at least as
        powerful on nonlinear own-lag effects (sine, kink, absolute value).

    Returns
    -------
    numpy.ndarray or (numpy.ndarray, numpy.ndarray)
        Boolean (d, d, max_lag+1) array with entries only on the diagonal; with
        ``return_pvalues``, also the float array of p-values in the same layout.
    """
    _check_nuisance(nuisance_basis)
    if test_basis not in ("union", "rff"):
        raise ValueError(f"test_basis must be 'union' or 'rff', got {test_basis!r}")
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (T, d), got shape {X.shape}")
    if np.isnan(X).any():
        raise ValueError("self_loop_test needs data without missing values")
    T, d = X.shape
    L = int(max_lag)
    if L < 1:
        raise ValueError(f"max_lag must be >= 1, got {max_lag!r}")
    if np.shape(cross) != (d, d, L + 1):
        raise ValueError(
            f"cross must have shape {(d, d, L + 1)}, got {np.shape(cross)}"
        )
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    if transform == "rank":
        X = norm.ppf((np.apply_along_axis(rankdata, 0, X) - 0.5) / len(X))
    elif transform is not None:
        raise ValueError(f"transform must be None or 'rank', got {transform!r}")
    cross = np.asarray(cross, dtype=bool)
    Z = np.stack([X[L - lag : T - lag] for lag in range(L + 1)], axis=2)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    n = Z.shape[0]
    nb = "rff" if nuisance_basis is None else nuisance_basis
    ex = [c[:, None] for c in _exog_columns(exog, T, L)]
    out = np.zeros((d, d, L + 1), dtype=bool)
    pv = np.full((d, d, L + 1), np.nan)
    for e in range(d):
        y = Z[:, e, 0]
        base = (
            [np.ones((n, 1))]
            + ex
            + [
                _nuisance_block(Z[:, c, lag], nb, "rff")
                for c in range(d)
                for lag in range(L + 1)
                # lag 0: only parents oriented into e; with the reverse edge
                # also kept, c(t) may be e's child, and conditioning on a child
                # that depends on e(t-1) would make e's own past look causal
                if c != e and cross[c, e, lag] and (lag > 0 or not cross[e, c, 0])
            ]
        )
        if test_basis == "rff":
            tested = {lag: [_rff_basis(Z[:, e, lag])] for lag in range(1, L + 1)}
        else:
            tested = {}
            for lag in range(1, L + 1):
                z = Z[:, e, lag]
                if _is_discrete(z):
                    # Level indicators span every function of a discrete lag (the
                    # linear term included): one test, not two copies at alpha / 2.
                    # spline_basis merges levels seen in fewer than 5 rows, so a
                    # rare level has no leverage near 1 (all levels merged: no
                    # columns, the test is undefined).
                    tested[lag] = [spline_basis(z)]
                else:  # the own lag alone and its spline, each at alpha / 2
                    tested[lag] = [z[:, None], spline_basis(z, n_knots=4)]
        own_n = {
            lag: (
                _rff_basis(Z[:, e, lag])
                if nb == "rff"
                else _nuisance_block(Z[:, e, lag], nb, "rff")
            )
            for lag in tested
        }
        for lag in tested:
            # Tested own lag, the other own lags as nuisance.
            others = [own_n[k] for k in tested if k != lag]
            red = np.column_stack(base + others)
            rr = y - red @ np.linalg.lstsq(red, y, rcond=None)[0]
            rk_r = np.linalg.matrix_rank(red)
            ps, last_df1 = [], 0
            for block in tested[lag]:  # nested blocks: the spline contains the lag
                full = np.column_stack(base + [block] + others)
                rf = y - full @ np.linalg.lstsq(full, y, rcond=None)[0]
                rk_f = np.linalg.matrix_rank(full)
                df1, df2 = rk_f - rk_r, n - rk_f
                rss_f, rss_r = rf @ rf, rr @ rr
                # Undefined test (no new direction, no residual df, exact fit), or
                # the same span as the previous block (a spline collapsed to the
                # linear term, e.g. on tied quantiles): not a separate test.
                if df1 <= last_df1 or df2 <= 0 or rss_f <= 0:
                    continue
                last_df1 = df1
                ps.append(
                    float(f_dist.sf(((rss_r - rss_f) / df1) / (rss_f / df2), df1, df2))
                )
            if not ps:
                continue
            # Bonferroni over the distinct tests actually run
            p = min(1.0, len(ps) * min(ps))
            pv[e, e, lag] = p
            out[e, e, lag] = p < alpha
    n_undefined = int(sum(np.isnan(pv[e, e, 1:]).sum() for e in range(d)))
    if n_undefined:
        warnings.warn(
            f"self_loop_test: {n_undefined} of {d * L} self-loop tests are undefined "
            "(no new direction, no residual degrees of freedom or an exact fit, e.g. "
            "too few rows for the kept parents and lags, or a constant or duplicated "
            "variable); those self-loops are not kept.",
            UserWarning,
            stacklevel=2,
        )
    return (out, pv) if return_pvalues else out


def _block_effect(resp, Q, phi):
    """Adjusted partial R^2 of ``phi`` for ``resp`` given orthonormal ``Q``; None when
    undefined (block inside the nuisance span, or no residual degrees of freedom)."""
    df1 = _block_df(phi, Q)
    df2 = len(resp) - Q.shape[1] - df1
    if df1 == 0 or df2 <= 0:
        return None
    r = resp - Q @ (Q.T @ resp)
    ph = phi - Q @ (Q.T @ phi)
    coef, *_ = np.linalg.lstsq(ph, r, rcond=None)
    rss0, rss1 = r @ r, np.sum((r - ph @ coef) ** 2)
    if rss0 <= 0 or rss1 <= 0:
        return None
    return _adjusted_r2(rss0, rss1, df1, df2)


def prune_effect_floor(
    X: np.ndarray,
    graph: np.ndarray,
    max_lag: int,
    effect_floor: float,
    *,
    exog: Optional[np.ndarray] = None,
    basis: str = "rff",
    nuisance_basis: Optional[str] = "spline",
    relative: float = 0.0,
) -> np.ndarray:
    """Drop lagged cross edges whose unique effect on their target is below a floor.

    Per target, backward elimination: each lagged cross edge ``x_c(t-l) -> x_e(t)`` gets
    the adjusted partial R^2 (:func:`_adjusted_r2`) of its ``basis`` block given the
    ``nuisance_basis`` blocks of the target's other lagged parents (self-lags included),
    its oriented lag-0 parents and ``exog``; on the mean, and on the scale (normal scores
    of the absolute residual of the model with the edge), whichever is larger, so a parent
    that acts only on the noise scale is kept. The weakest edge below the target's
    threshold, ``max(effect_floor, relative * top)`` with ``top`` the largest effect among
    the target's cross edges in the first pass, is removed and the rest re-scored, until
    every remaining edge reaches it (the threshold stays fixed meanwhile). An edge
    whose effect is undefined (no residual degrees of freedom) is kept. Self-loops and
    lag-0 edges are never removed.

    Significance tests and the L0 gates keep any real dependence once T is large; real
    data are only approximately sparse, so their graphs fill with edges that are real but
    explain almost nothing (e.g. on river networks, every reverse-neighbour and two-hop
    edge). The floor is a fraction of the target's unexplained variance, so it is free of
    units and of T.

    Parameters
    ----------
    X : numpy.ndarray
        Data, shape (T, d), without missing values; rank-transformed internally.
    graph : numpy.ndarray
        Graph (d, d, max_lag+1) as ``[cause, effect, lag]``.
    max_lag : int
        Maximum lag.
    effect_floor : float
        Minimum adjusted partial R^2 in [0, 1). With ``relative`` 0, 0 returns
        ``graph`` unchanged.
    exog : numpy.ndarray or None
        Extra regressors (T, k), conditioned on in every fit (e.g. the C basis).
    basis, nuisance_basis : str
        Bases of the scored edge and of the conditioning parents, as in
        :func:`grow_candidates`.
    relative : float
        In [0, 1): a target's threshold is also at least this fraction of its strongest
        cross edge's effect, so it scales with the target's signal (a target with one
        cross parent only meets ``effect_floor``).

    Returns
    -------
    numpy.ndarray
        Copy of ``graph`` with the pruned edges set to 0.
    """
    if not 0 <= effect_floor < 1:
        raise ValueError(f"effect_floor must be in [0, 1), got {effect_floor!r}")
    if not 0 <= relative < 1:
        raise ValueError(f"relative must be in [0, 1), got {relative!r}")
    G = np.array(graph, copy=True)
    if effect_floor <= 0 and relative <= 0:
        return G
    X = norm.ppf(
        (np.apply_along_axis(rankdata, 0, np.asarray(X, float)) - 0.5) / len(X)
    )
    T, d = X.shape
    L = max_lag
    Z = np.stack([X[L - lag : T - lag] for lag in range(L + 1)], axis=2)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    n = Z.shape[0]
    base = [np.ones(n)] + _exog_columns(exog, T, L)
    feats = {
        (c, lag): _parent_block(Z, c, lag, basis)
        for c in range(d)
        for lag in range(1, L + 1)
    }
    nfeats = _nuisance_features(Z, feats, nuisance_basis, basis)
    est = G[:d, :d].astype(bool)

    def effect(e, keep, cand, lag0):
        A = np.column_stack(base + [nfeats[p] for p in keep if p != cand] + lag0)
        U, s, _ = np.linalg.svd(A, full_matrices=False)
        Q = U[:, s > s[0] * max(A.shape) * np.finfo(float).eps]
        y = Z[:, e, 0]
        phi = feats[cand]
        mean = _block_effect(y, Q, phi)
        # scale channel: absolute residual of the model WITH the edge, so mean structure
        # the edge explains does not reappear as a variance effect
        full = np.column_stack([Q, phi])
        coef, *_ = np.linalg.lstsq(full, y, rcond=None)
        v = norm.ppf((rankdata(np.abs(y - full @ coef)) - 0.5) / n)
        scale = _block_effect(v, Q, phi)
        vals = [x for x in (mean, scale) if x is not None]
        return max(vals) if vals else None

    for e in range(d):
        keep = [(c, lag) for c in range(d) for lag in range(1, L + 1) if est[c, e, lag]]
        lag0 = _lag0_parent_blocks(Z, est, e, nuisance_basis, basis)
        tau = None
        while True:
            scored = {p: effect(e, keep, p, lag0) for p in keep if p[0] != e}
            scored = {p: v for p, v in scored.items() if v is not None}
            if not scored:
                break
            if tau is None:
                tau = max(effect_floor, relative * max(scored.values()))
            worst = min(scored, key=lambda p: (scored[p], p))
            if scored[worst] >= tau:
                break
            keep.remove(worst)
            G[worst[0], e, worst[1]] = 0
    return G
