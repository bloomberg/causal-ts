# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Parent-adjusted lag dependence (``adjust="parents"`` in the lag plot).

Three steps, the first two as in PCMCI:

1. For each target, select lagged parents greedily by distance correlation with
   the current residual, and keep the best candidate while a block-permutation
   test of the *maximum* over all candidates rejects independence (testing the
   winner alone would let false parents through at about ``m * alpha`` for
   ``m`` candidates). Then prune parents that add nothing given the others.
2. Score each ``X_i(t-k) -> X_j(t)`` by the unbiased squared distance
   correlation of the two residuals after adjusting both for ``X_j``'s other
   parents and ``X_i``'s parents shifted by ``k``, leaving out ``X_i``'s own
   lags from the shifted group.
3. Markers: a block-permutation test per source-target pair of the maximum
   dcor over all lags, with every source lag and the target residualized on
   the target's selected parents minus all lags of the source. In a pair that rejects, the peak-score lag is marked, and so is
   any other lag whose own full-MCI test (own lags included, as in PCMCI)
   rejects. A test per lag alone loses many true lags of persistent sources.

Adjustment is a ridge regression on either the conditioning columns
themselves (linear) or the dictionary of the ``rift`` CI test: cubic splines
and multi-frequency sines/cosines per column, plus pairwise spline tensor
products and a few triple products. The two are chosen per fit by blocked,
embargoed cross-validation on the training rows. The dictionary approximates
nonlinear and interacting conditional means without a specified form, but
extrapolates badly on strongly persistent series, where the held-out blocks
cover values the training rows do not; without the linear option the
leftover structure made independent AR(0.95) series look dependent (25% of
null lags flagged at 0.01, against 1% with the selection). The ridge penalty
is chosen by leave-one-out; residuals are out-of-fold on blocked folds whose
embargo covers the lag embedding.

The scores measure dependence left in the residuals, which is the
conditional dependence only under additive noise: a common driver that
changes both variables' noise variance leaves dependent residuals even with
the true conditional means removed.
"""

import dcor
import numpy as np

from ..ci_tests._residual_score import _loo_ridge, blocked_folds

#: Significance level of the parent-selection test.
PARENT_ALPHA = 0.01
#: Most parents kept per target.
MAX_PARENTS = 4
_N_SPLITS = 5
#: Fewest rows left after lagging (T - 2 * max_lag) for the cross-fitted
#: adjustment to be meaningful.
MIN_ROWS = 50
#: Fewest training rows in any outer fold (after the embargo); below this in an
#: inner fold the design choice is skipped and the linear design is used.
MIN_TRAIN = 30


def embargo(max_lag):
    """Rows dropped on each side of a held-out block: covers the lag
    embedding, so a training row's lagged predictors never reach held-out
    values."""
    return 2 * max_lag + 2


def smallest_outer_train(n, max_lag):
    """Rows in the smallest outer training fold for ``n`` rows after lagging."""
    folds = blocked_folds(n, n_splits=_N_SPLITS, gap=embargo(max_lag))
    return min((len(tr) for tr, _ in folds), default=0)


_rift = None


def _rift_dictionary(Zs):
    """rift's adjustment dictionary for the standardized columns of ``Zs``,
    plus the columns themselves. One column gets its marginal splines here;
    from two columns they come with rift's interaction dictionary."""
    global _rift
    from ..ci_tests.raft import RIFTCI, _standardize_columns

    if _rift is None:
        _rift = RIFTCI(np.zeros((50, 3)))
    cols = [Zs[:, c] for c in range(Zs.shape[1])]
    parts = [Zs]
    for c in cols:
        a = np.outer(c, _rift.freqs)
        parts.append(_standardize_columns(np.hstack([np.cos(a), np.sin(a)])))
    if len(cols) >= 2:
        parts.append(_rift._interaction_dictionary(cols))
    elif np.ptp(cols[0]) > 0:
        parts.append(_rift._spline_block(cols[0]))
    return np.hstack(parts)


def _fit_predict(D, Y, train, held):
    _, coef, x_mean, y_mean = _loo_ridge(D[train], Y[train])
    return (D[held] - x_mean) @ coef + y_mean


def _residuals(Y, Z, gap):
    """Out-of-fold residuals of the columns of ``Y`` on the columns in ``Z``.

    Per outer fold and per column of ``Y``, the design (linear or dictionary)
    is chosen by blocked cross-validation inside the training rows, so the
    held-out rows never inform the choice. The designs depend on ``Z`` only,
    never on ``Y``, so they are built once on all rows.
    """
    Y = np.asarray(Y, dtype=float)
    if not Z:
        return Y - Y.mean(0)
    Zc = np.column_stack(Z)
    sd = Zc.std(0)
    Zs = (Zc - Zc.mean(0)) / np.where(sd > 1e-12, sd, 1.0)
    designs = (Zs, _rift_dictionary(Zs))
    R = np.empty_like(Y)
    for train, held in blocked_folds(len(Y), n_splits=_N_SPLITS, gap=gap):
        if len(train) < MIN_TRAIN:
            raise ValueError(
                f"Too few rows for the cross-fitted adjustment: {len(train)} "
                f"training rows after the embargo (need {MIN_TRAIN})."
            )
        err = np.zeros((len(designs), Y.shape[1]))
        inner = list(blocked_folds(len(train), n_splits=3, gap=gap))
        # Too short to choose reliably: keep the linear design (err stays 0
        # and ties keep the first design).
        if all(len(i_tr) >= MIN_TRAIN and len(i_va) for i_tr, i_va in inner):
            for inner_tr, inner_va in inner:
                tr, va = train[inner_tr], train[inner_va]
                for m, D in enumerate(designs):
                    err[m] += ((Y[va] - _fit_predict(D, Y, tr, va)) ** 2).sum(0)
        choice = np.argmin(err, axis=0)  # ties keep the linear design
        for m, D in enumerate(designs):
            cols = np.flatnonzero(choice == m)
            if cols.size:
                R[np.ix_(held, cols)] = Y[np.ix_(held, cols)] - _fit_predict(
                    D, Y[:, cols], train, held
                )
    return R


def _residual(y, Z, gap):
    """Out-of-fold residual of one series ``y`` on the columns in ``Z``."""
    return _residuals(y[:, None], Z, gap)[:, 0]


def _mci_conditioning(i, j, k, parents, self_history=True):
    """MCI conditioning set of ``X_i(t-k) -> X_j(t)`` as ``(variable, lag)``
    pairs relative to ``t``: ``X_j``'s other parents, then ``X_i``'s parents
    shifted by ``k`` (those not already in the first group).

    ``self_history=False`` leaves out ``X_i``'s own lags from the shifted
    group (the ranking score): for a persistent source they remove most of
    the variation that drives the target."""
    Z = [p for p in parents[j] if p != (i, k)]
    Z += [
        (a, b + k)
        for a, b in parents[i]
        if (a, b + k) not in parents[j] and (self_history or a != i)
    ]
    return Z


def _block_pvalue(x, y, n_permutations, rng):
    """Permutation p-value of dcor(x, y), shuffling contiguous blocks of ``y``.

    Blocks of about sqrt(n) rows keep ``y``'s short-range autocorrelation. A
    circular shift would also keep it, but realigns a periodic ``x`` with ``y``
    at multiples of its period and then cannot reject.
    """
    n = len(y)
    b = int(np.ceil(np.sqrt(n)))
    starts = np.arange(0, n, b)
    observed = dcor.distance_correlation(x, y)
    count = 0
    for _ in range(n_permutations):
        yp = np.concatenate([y[s : s + b] for s in rng.permutation(starts)])
        count += dcor.distance_correlation(x, yp) >= observed
    return (1 + count) / (n_permutations + 1)


#: Above this many float64 entries (candidates x n x n) the max test falls
#: back to per-candidate dcor calls instead of stacked distance matrices.
_MAX_STACK_ENTRIES = 1e7


def _centered_distances(v):
    """Double-centred distance matrix of a 1-D sample (as in distance covariance)."""
    D = np.abs(v[:, None] - v[None, :])
    return D - D.mean(0) - D.mean(1)[:, None] + D.mean()


def _block_max_rejects(best, cands, y, n_permutations, rng, alpha):
    """Whether the best candidate's dcor with ``y`` is significant at ``alpha``
    against the null distribution of the *maximum* dcor over all candidates
    (block permutation of ``y``), which accounts for having picked the winner.

    Rejects when ``p = (1 + count) / (n_permutations + 1) < alpha``. Only the
    decision is needed, so the loop ends once the count of permutations
    reaching the observed value already implies ``p >= alpha``.
    A block permutation of ``y`` only reorders the rows and columns of its
    distance matrix, so with the candidates' centred distance matrices stacked,
    every candidate's (squared, biased) dcor with a permuted ``y`` is one
    matrix-vector product.
    """
    n = len(y)
    b = int(np.ceil(np.sqrt(n)))
    starts = np.arange(0, n, b)

    def settled(count):  # p >= alpha already, whatever the rest gives
        return (1 + count) / (n_permutations + 1) >= alpha

    if settled(0):
        return False
    count = 0
    if len(cands) * n * n > _MAX_STACK_ENTRIES:
        observed = dcor.distance_correlation(best, y)
        for _ in range(n_permutations):
            yp = np.concatenate([y[s : s + b] for s in rng.permutation(starts)])
            if any(dcor.distance_correlation(c, yp) >= observed for c in cands):
                count += 1
                if settled(count):
                    return False
        return True
    A = np.stack([_centered_distances(c).ravel() for c in cands])
    var_x = np.einsum("ij,ij->i", A, A)
    B = _centered_distances(y)
    var_y = np.einsum("ij,ij->", B, B)
    if var_y <= 0:
        return False
    scale = np.sqrt(np.where(var_x > 0, var_x, np.inf) * var_y)
    Ab = _centered_distances(best).ravel()
    var_b = Ab @ Ab
    if var_b <= 0:
        return False
    observed = (Ab @ B.ravel()) / np.sqrt(var_b * var_y)
    idx = np.arange(n)
    for _ in range(n_permutations):
        perm = np.concatenate([idx[s : s + b] for s in rng.permutation(starts)])
        r2 = (A @ B[np.ix_(perm, perm)].ravel()) / scale
        if r2.max() >= observed * (1 - 1e-12):
            count += 1
            if settled(count):
                return False
    return True


def _u_centered(v):
    """U-centred distance matrix of a 1-D sample (as in unbiased dcor)."""
    n = len(v)
    D = np.abs(v[:, None] - v[None, :])
    r = D.sum(1)
    U = D - r[:, None] / (n - 2) - r[None, :] / (n - 2) + r.sum() / ((n - 1) * (n - 2))
    np.fill_diagonal(U, 0.0)
    return U


def _pair_pvalue(X, y, n_permutations, rng, alpha=None):
    """Block-permutation p-value of the maximum unbiased squared dcor between
    the columns of ``X`` (one per lag) and ``y``, shuffling blocks of ``y``.

    Testing all lags of a source at once keeps the evidence that a persistent
    source spreads over neighbouring lags, which a test per lag splits up.
    Needs ``n >= 4`` (unbiased dcor). A constant ``y`` returns 1 without
    drawing permutations on either route, so later tests see the same stream.
    With ``alpha``, stops once ``p >= alpha`` is certain and returns that lower
    bound (exact below ``alpha``).
    """
    n, m = X.shape
    if n < 4:
        raise ValueError("_pair_pvalue needs at least 4 rows.")
    if np.ptp(y) == 0:
        return 1.0
    b = int(np.ceil(np.sqrt(n)))
    starts = np.arange(0, n, b)

    def peak(yv):
        return max(dcor.u_distance_correlation_sqr(X[:, c], yv) for c in range(m))

    if m * n * n > _MAX_STACK_ENTRIES:
        observed = peak(y)
        count = 0
        for _ in range(n_permutations):
            yp = np.concatenate([y[s : s + b] for s in rng.permutation(starts)])
            count += peak(yp) >= observed - 1e-12 * abs(observed)
            if _settled(count, n_permutations, alpha):
                break
        return (1 + count) / (n_permutations + 1)
    A = np.stack([_u_centered(X[:, c]).ravel() for c in range(m)])
    var_x = np.einsum("ij,ij->i", A, A)
    U = _u_centered(y)
    var_y = np.einsum("ij,ij->", U, U)
    if var_y <= 0:
        return 1.0
    scale = np.sqrt(np.where(var_x > 0, var_x, np.inf) * var_y)
    observed = ((A @ U.ravel()) / scale).max()
    idx = np.arange(n)
    count = 0
    for _ in range(n_permutations):
        perm = np.concatenate([idx[s : s + b] for s in rng.permutation(starts)])
        r2 = (A @ U[np.ix_(perm, perm)].ravel()) / scale
        count += r2.max() >= observed - 1e-12 * abs(observed)
        if _settled(count, n_permutations, alpha):
            break
    return (1 + count) / (n_permutations + 1)


def _settled(count, n_permutations, alpha):
    """Whether ``p = (1 + count) / (n_permutations + 1) >= alpha`` already."""
    return alpha is not None and (1 + count) / (n_permutations + 1) >= alpha


def _lag_pvalue(x, y, n_permutations, rng, alpha=None):
    """:func:`_block_pvalue` with the early stop of :func:`_pair_pvalue`."""
    n = len(y)
    b = int(np.ceil(np.sqrt(n)))
    starts = np.arange(0, n, b)
    observed = dcor.distance_correlation(x, y)
    count = 0
    for _ in range(n_permutations):
        yp = np.concatenate([y[s : s + b] for s in rng.permutation(starts)])
        count += dcor.distance_correlation(x, yp) >= observed
        if _settled(count, n_permutations, alpha):
            break
    return (1 + count) / (n_permutations + 1)


def _pair_marker_pvalues(sc, lag_p, pair_p):
    """Marker thresholds for one source-target pair, lags 1..max_lag, such
    that ``value < alpha`` is the star rule at any ``alpha``: the pair counts
    when its joint test rejects (null: no dependence of the target on any lag
    of the source); its highest-scoring lag then gets a star, and so does any
    other lag whose own full-MCI test also rejects. The peak's value is the
    pair p-value: the peak lag has no test of its own, so for a persistent
    source the star can land on a neighbour of the true lag. These are not
    lag-level p-values (no FDR on them)."""
    pv = np.maximum(pair_p, lag_p)
    pv[int(np.argmax(sc))] = pair_p
    return pv


def parent_adjusted_scores(
    values,
    max_lag,
    n_permutations,
    seed,
    pvalues=False,
    alpha=None,
    n_jobs=1,
    include_lag0=False,
):
    """Parent-adjusted lag scores.

    ``scores`` rank lags by the residual dependence after adjusting for the
    target's other parents and the source's parents other than its own lags;
    ``pvals`` are marker thresholds combining a joint test per pair with the
    full-MCI test per lag (see :func:`_pair_marker_pvalues`); ``pvals < alpha`` is
    the marker rule. They are not lag-level p-values.

    With ``alpha``, only the tests that can change a marker at ``alpha`` run
    (per-lag tests only for non-peak lags of significant pairs) and each stops
    once its p-value is certain to be at least ``alpha``: the markers are the
    same as without ``alpha``, but values at or above ``alpha`` are then
    lower bounds or 1. ``n_jobs`` parallelizes over source-target pairs
    (joblib); every test has its own seeded generator, so results do not
    depend on it.

    ``include_lag0`` adds a symmetric, untested lag-0 score per pair (both
    series adjusted for their lagged parents).

    Returns ``(scores, pvals, parents)``: ``scores`` and ``pvals`` are
    ``(d, d, max_lag+1)``. Lag 0 is 0 unless ``include_lag0`` (p 1 either way); the diagonal holds
    each variable's own-lag scores, untested (p 1). ``pvals`` is None unless
    ``pvalues``; ``parents[j]`` lists the selected ``(source, lag)`` pairs of
    variable ``j``.
    """
    if seed is None:  # a fresh run, as elsewhere; one concrete seed for all tests
        seed = int(np.random.SeedSequence().entropy % 2**32)
    rng = np.random.default_rng(seed)
    T, d = values.shape
    # Rows with room for every lag of a parent of a source shifted by its lag.
    t = np.arange(2 * max_lag, T)
    gap = embargo(max_lag)

    def col(i, k):
        return values[t - k, i]

    candidates = [(i, k) for i in range(d) for k in range(1, max_lag + 1)]
    parents = {}
    for j in range(d):
        target = values[t, j]
        sel = []
        for _ in range(MAX_PARENTS):
            remaining = [c for c in candidates if c not in sel]
            if not remaining:
                break
            r = _residual(target, [col(*p) for p in sel], gap)
            best = max(
                remaining, key=lambda c: dcor.u_distance_correlation_sqr(col(*c), r)
            )
            if not _block_max_rejects(
                col(*best),
                [col(*c) for c in remaining],
                r,
                n_permutations,
                rng,
                PARENT_ALPHA,
            ):
                break
            sel.append(best)
        # Prune: a proxy picked before the real parent adds nothing once the
        # real parent is in.
        while len(sel) > 1:
            pv = [
                _block_pvalue(
                    col(*c),
                    _residual(target, [col(*o) for o in sel if o != c], gap),
                    n_permutations,
                    rng,
                )
                for c in sel
            ]
            worst = int(np.argmax(pv))
            if pv[worst] < PARENT_ALPHA:
                break
            sel.pop(worst)
        parents[j] = sel

    jobs = [(i, j) for i in range(d) for j in range(d)]
    args = (values, t, gap, parents, max_lag, n_permutations, seed, pvalues, alpha)
    if n_jobs == 1:
        out = [_score_pair(i, j, *args) for i, j in jobs]
    else:
        from joblib import Parallel, delayed

        out = Parallel(n_jobs=n_jobs)(
            delayed(_score_pair)(i, j, *args) for i, j in jobs
        )
    scores = np.zeros((d, d, max_lag + 1))
    pvals = np.ones((d, d, max_lag + 1)) if pvalues else None
    for (i, j), (sc, pv) in zip(jobs, out):
        scores[i, j, 1:] = sc
        if pvalues and pv is not None:
            pvals[i, j, 1:] = pv
    if include_lag0:
        # Contemporaneous dependence has no direction: one symmetric score per
        # pair, both series adjusted for the union of their lagged parents.
        # Not tested (p stays 1).
        for i in range(d):
            for j in range(i + 1, d):
                Z = list(dict.fromkeys(parents[i] + parents[j]))
                R = _residuals(
                    np.column_stack([values[t, i], values[t, j]]),
                    [col(a, b) for a, b in Z],
                    gap,
                )
                scores[i, j, 0] = scores[j, i, 0] = dcor.u_distance_correlation_sqr(
                    R[:, 0], R[:, 1]
                )
    return scores, pvals, parents


def _grouped_residuals(values, t, gap, i, j, lags, conditioning):
    """Residuals of ``X_i(t-k)`` for each lag in ``lags`` and of ``X_j(t)``,
    fitting each distinct conditioning set once. ``conditioning(k)`` gives the
    set for lag ``k``. Columns are fitted independently, so grouping does not
    change any residual. Returns ``{k: (source residual, target residual)}``."""
    groups = {}
    for k in lags:
        groups.setdefault(tuple(conditioning(k)), []).append(k)
    out = {}
    for Zset, ks in groups.items():
        Z = [values[t - b, a] for a, b in Zset]
        Y = np.column_stack([values[t - k, i] for k in ks] + [values[t, j]])
        R = _residuals(Y, Z, gap)
        for c, k in enumerate(ks):
            out[k] = (R[:, c], R[:, -1])
    return out


def _score_pair(
    i, j, values, t, gap, parents, max_lag, n_permutations, seed, pvalues, alpha
):
    """Scores (and marker thresholds) of ``X_i(t-k) -> X_j(t)`` for k >= 1.

    The diagonal (``i == j``) gets scores only: a variable's own lags are
    judged by the parent selection. Each test draws from its own generator,
    seeded by ``(seed, i, j, test)``, so results do not depend on ``n_jobs``
    or on which tests run.
    """
    lags = range(1, max_lag + 1)
    rank = _grouped_residuals(
        values,
        t,
        gap,
        i,
        j,
        lags,
        lambda k: _mci_conditioning(i, j, k, parents, self_history=False),
    )
    sc = np.array([dcor.u_distance_correlation_sqr(*rank[k]) for k in lags])
    if not pvalues or i == j:
        return sc, None
    streams = np.random.SeedSequence([seed, i, j]).spawn(max_lag + 1)
    # Pair test: all lags of the source against the target, both adjusted for
    # the target's parents from other variables.
    Z = [values[t - b, a] for a, b in parents[j] if a != i]
    pair = np.column_stack([values[t - k, i] for k in lags] + [values[t, j]])
    R = _residuals(pair, Z, gap)
    pair_p = _pair_pvalue(
        R[:, :-1], R[:, -1], n_permutations, np.random.default_rng(streams[0]), alpha
    )
    peak = 1 + int(np.argmax(sc))
    if alpha is None:
        need = list(lags)
    elif pair_p < alpha:
        need = [k for k in lags if k != peak]
    else:
        need = []
    lag_p = np.ones(max_lag)
    if need:
        full = _grouped_residuals(
            values, t, gap, i, j, need, lambda k: _mci_conditioning(i, j, k, parents)
        )
        for k in need:
            lag_p[k - 1] = _lag_pvalue(
                *full[k], n_permutations, np.random.default_rng(streams[k]), alpha
            )
    return sc, _pair_marker_pvalues(sc, lag_p, pair_p)
