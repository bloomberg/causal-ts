# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared machinery for residual-score conditional independence tests.

GCM, WGCM and PCM differ only in the statistic they compute; everything up to
that point is common. They all extract ``(x, y, Z)``, regress ``x`` and ``y`` on
``Z``, and test a function of the residual product. This module holds that shared
path so each concrete test in :mod:`causalts.ci_tests.gcm`,
:mod:`causalts.ci_tests.wgcm` and :mod:`causalts.ci_tests.pcm` is just its
statistic.

The three concerns collected here are all consequences of the rows being a time
series rather than an i.i.d. sample:

Blocked cross-fitting
    Rows reaching a CI test are strictly time-ordered (see
    ``causalts.cdnots.phase3_utils``, which lag-embeds with ``df.shift`` and then
    slices with ``.iloc``, never shuffling). Random K-fold cross-fitting leaks
    neighbouring observations into nuisance predictions: with an AR(1)
    coefficient of 0.9, the row immediately before an evaluation point carries
    most of its information, and a random split will usually place it in the
    training set. :class:`BlockCrossFitter` uses contiguous folds with an embargo
    gap so a nuisance model never sees observations adjacent to the rows it
    predicts. The nuisance learners (see :func:`make_regressor`) are deliberately
    lightweight rather than Gaussian processes or deep models, being far cheaper
    inside a search issuing thousands of queries. The default,
    :class:`SplineSelectRidge`, is an additive cubic-spline ridge with a
    leave-one-out penalty that falls back to linear ridge when that fits better:
    a linear nuisance leaves a nonlinear ``E[X | Z]`` in the residuals and makes
    the tests over-reject, while a fixed low-flexibility basis under-fits rough
    mechanisms. On known-CI triples it matches an exact Gaussian process on
    size-adjusted power at a small fraction of the cost. The construction
    follows double/debiased-ML (Chernozhukov et al., 2018); the embargo adapts
    purged/embargoed cross-validation to the lag-embedded setting.

Long-run-variance calibration
    For a residual-product series ``q_t`` the variance of its mean is
    ``(1/n) * sum_{|h|<n} (1 - |h|/n) * gamma(h)``, not ``var(q)/n``. Using the
    i.i.d. formula on an autocorrelated series understates the standard error and
    the test over-rejects -- which, inside a constraint-based search issuing
    thousands of queries, turns into spurious edges that then contaminate later
    conditioning sets. :func:`hac_se` gives the Newey--West long-run standard
    error; :func:`multiplier_bootstrap` calibrates by resampling, either i.i.d.
    (valid for independent rows) or with Shao's dependent wild bootstrap (which
    preserves serial correlation) and also handles the maximum over a weight bank
    in one pass.

Fixed weight banks
    Averaging the residual product over all rows is blind to any alternative
    where the conditional covariance changes sign across the conditioning space:
    the positive and negative regions cancel. :class:`FeatureBank` multiplies the
    product by fixed functions of ``Z`` before averaging, recovering those
    alternatives. The weights are pre-specified -- choosing them from the residual
    product would let the same noise drive both the weight choice and the
    statistic -- and the maximum over the bank is calibrated jointly by
    :func:`multiplier_bootstrap`, so the search itself is paid for.

References
----------
Newey & West (1987) for the HAC long-run standard error; Shao (2010), "The
dependent wild bootstrap", for the Bartlett-correlated multipliers.
"""

from functools import lru_cache

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from .cit_test import CIT_Base, stable_seed

__all__ = [
    "ResidualScoreCIT",
    "BlockCrossFitter",
    "blocked_folds",
    "make_regressor",
    "NUISANCE_LEARNERS",
    "SplineSelectRidge",
    "hac_se",
    "bartlett_bandwidth",
    "multiplier_bootstrap",
    "moving_block_bootstrap",
    "FeatureBank",
    "DEFAULT_WEIGHTS",
]


# ---------------------------------------------------------------------------
# Blocked cross-fitting
# ---------------------------------------------------------------------------


def blocked_folds(n, n_splits=5, gap=0):
    """Yield ``(train_idx, eval_idx)`` pairs of contiguous blocks with embargo.

    The evaluation blocks partition ``range(n)``. The training set for a block
    is everything outside it, minus ``gap`` rows on each side.

    Parameters
    ----------
    n : int
        Number of (time-ordered) rows.
    n_splits : int
        Number of contiguous evaluation blocks.
    gap : int
        Embargo width. Rows within ``gap`` positions of an evaluation block are
        withheld from that block's training set.

    Yields
    ------
    (np.ndarray, np.ndarray)
        Training and evaluation index arrays.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if gap < 0:
        raise ValueError("gap must be non-negative.")

    bounds = np.linspace(0, n, n_splits + 1).astype(int)
    all_idx = np.arange(n)
    for k in range(n_splits):
        lo, hi = bounds[k], bounds[k + 1]
        if lo == hi:
            continue
        eval_idx = all_idx[lo:hi]
        embargo_lo = max(0, lo - gap)
        embargo_hi = min(n, hi + gap)
        train_mask = np.ones(n, dtype=bool)
        train_mask[embargo_lo:embargo_hi] = False
        yield all_idx[train_mask], eval_idx


def _ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def _spline():
    # Low, fixed degrees of freedom. At T=40 a flexible basis overfits the
    # nuisance and inflates the residual product's variance.
    return make_pipeline(
        StandardScaler(),
        SplineTransformer(n_knots=4, degree=3, include_bias=False),
        Ridge(alpha=1.0),
    )


def _tree():
    return ExtraTreesRegressor(
        n_estimators=100,
        max_depth=3,
        min_samples_leaf=5,
        random_state=0,
        n_jobs=1,
    )


#: Ridge penalties searched by :class:`SplineSelectRidge` (efficient leave-one-out).
_LOO_ALPHAS = np.logspace(-3, 3, 13)


def _cubic_bspline_basis(x, lo, hi, n_knots):
    """Uniform-knot cubic B-spline basis of ``x`` on ``[lo, hi]``.

    Same basis as ``SplineTransformer(n_knots, degree=3, include_bias=False)``
    fitted on data spanning ``[lo, hi]``: knots extended by three spacings on each
    side, constant extrapolation (``x`` clipped to the range), last column dropped.
    Evaluated in closed form (the cardinal cubic B-spline), which is what makes it
    cheap enough to rebuild inside every fold of every CI query.
    """
    step = (hi - lo) / (n_knots - 1)
    starts = lo + step * (np.arange(n_knots + 2) - 3)
    u = (np.clip(x, lo, hi)[:, None] - starts[None, :]) / step
    B = np.zeros_like(u)
    m = (u >= 0) & (u < 1)
    B[m] = u[m] ** 3 / 6.0
    m = (u >= 1) & (u < 2)
    v = u[m]
    B[m] = (-3 * v**3 + 12 * v**2 - 12 * v + 4) / 6.0
    m = (u >= 2) & (u < 3)
    v = u[m]
    B[m] = (3 * v**3 - 24 * v**2 + 60 * v - 44) / 6.0
    m = (u >= 3) & (u < 4)
    B[m] = (4 - u[m]) ** 3 / 6.0
    return B[:, :-1]


def _loo_ridge(X, Y):
    """Ridge with an unpenalized intercept, one penalty per column of ``Y`` by LOO.

    One eigendecomposition of the centred design's Gram matrix gives the exact
    leave-one-out error of every penalty in :data:`_LOO_ALPHAS` in closed form
    (hat-matrix diagonal), without dividing by small singular values. ``Y`` is
    ``(n, k)``; each column picks its own penalty. Returns
    ``(loo_mse (k,), coef (p, k), x_mean, y_mean)``.
    """
    n, p = X.shape
    k = Y.shape[1]
    x_mean, y_mean = X.mean(0), Y.mean(0)
    Xc, Yc = X - x_mean, Y - y_mean
    if p > n:
        # Wide design (a spline basis on several columns at small T): the n x n
        # dual Gram gives the same hat matrix and coefficients more cheaply.
        eig, U = np.linalg.eigh(Xc @ Xc.T)
        eig = np.clip(eig, 0.0, None)
        Uty = U.T @ Yc  # (n, k)
        inv = 1.0 / (eig[None, :] + _LOO_ALPHAS[:, None])  # (A, n)
        shrink = eig[None, :] * inv  # (A, n)
        lev = (U**2) @ shrink.T + 1.0 / n  # (n, A)
        W = (shrink[:, :, None] * Uty[None]).transpose(1, 0, 2).reshape(n, -1)
        fit = (U @ W).reshape(n, len(_LOO_ALPHAS), k)
        loo = np.mean(((Yc[:, None, :] - fit) / (1.0 - lev)[:, :, None]) ** 2, axis=0)
        best = np.argmin(loo, axis=0)
        coef = Xc.T @ (U @ (inv[best].T * Uty))  # (p, k)
        return loo[best, np.arange(k)], coef, x_mean, y_mean
    eig, V = np.linalg.eigh(Xc.T @ Xc)
    eig = np.clip(eig, 0.0, None)
    P = Xc @ V  # (n, p): principal-axis scores
    Pty = P.T @ Yc  # (p, k)
    inv = 1.0 / (eig[None, :] + _LOO_ALPHAS[:, None])  # (A, p)
    lev = (P**2) @ inv.T + 1.0 / n  # (n, A)
    W = (inv[:, :, None] * Pty[None]).transpose(1, 0, 2).reshape(p, -1)  # (p, A*k)
    fit = (P @ W).reshape(n, len(_LOO_ALPHAS), k)
    resid = Yc[:, None, :] - fit
    loo = np.mean((resid / (1.0 - lev)[:, :, None]) ** 2, axis=0)  # (A, k)
    best = np.argmin(loo, axis=0)  # first minimum, as a strict-< scan would pick
    coef = V @ (inv[best].T * Pty)  # (p, k)
    return loo[best, np.arange(k)], coef, x_mean, y_mean


class SplineSelectRidge:
    """Additive cubic-spline ridge that falls back to linear ridge when it fits better.

    Each conditioning column is standardized and expanded into ``n_knots`` uniform
    cubic B-splines; ridge with a leave-one-out penalty is fitted on that basis and,
    separately, on the standardized columns themselves. Whichever has the lower
    leave-one-out error is kept. The spline basis gives the nuisance model the
    flexibility the published GCM/WGCM/PCM assume (a linear nuisance leaves a
    nonlinear ``E[X | Z]`` in the residuals and the test over-rejects), while the
    linear candidate keeps the linear case as sharp as plain ridge. The basis is
    fixed by the knots, so the learner is deterministic; only the coefficients and
    penalty are estimated. ``Y`` may have several columns sharing one design (one
    decomposition), each selecting its own model.

    Penalty and model are chosen by leave-one-out, which treats the training rows
    as exchangeable. On autocorrelated, lag-embedded rows a held-out row's
    neighbours carry most of its information, so leave-one-out is optimistic and
    leans towards the more flexible fit. The outer cross-fit stays embargoed, so
    this affects the nuisance fit's variance, not the evaluation rows. Measured in
    CDNOTS discovery (d=20, T=100, spline mechanism), choosing by contiguous-block
    CV with an embargo instead recovered only a small part of ``wgcm``'s recall
    loss against linear ridge and changed nothing at T=400, so leave-one-out is
    kept for its closed form. Outside the training range the basis is flat
    (constant extrapolation). Under blocked cross-fitting on a strongly drifting
    ``Z`` each held-out block covers values no training row has, so a nonlinear
    ``E[X | Z]`` cannot be learned there by any nuisance model and the tests
    over-reject; linear ridge does far worse in that setting, and linear
    extrapolation of the basis was measured to be worse than flat.
    """

    multi_output = True

    def __init__(self, n_knots=12, select=True):
        self.n_knots = int(n_knots)
        self.select = bool(select)

    def _design(self, Z, spline):
        Zs = (Z - self.mu_) / self.sd_
        if not spline:
            return Zs
        cols = [
            _cubic_bspline_basis(Zs[:, j], self.lo_[j], self.hi_[j], self.n_knots)
            for j in range(Zs.shape[1])
            if self.hi_[j] - self.lo_[j] > 1e-12
        ]
        return np.hstack(cols) if cols else np.zeros((len(Z), 0))

    def fit(self, Z, Y):
        Y = np.asarray(Y, dtype=float)
        self.vector_ = Y.ndim == 1
        Y = Y.reshape(len(Y), -1)
        Z = np.asarray(Z, dtype=float).reshape(len(Y), -1)
        self.mu_ = Z.mean(0)
        sd = Z.std(0)
        # Unit scale for a constant column, as StandardScaler does: a held-out row
        # with a different value must not be multiplied by ~1e12.
        self.sd_ = np.where(sd > 1e-12, sd, 1.0)
        Zs = (Z - self.mu_) / self.sd_
        self.lo_, self.hi_ = Zs.min(0), Zs.max(0)
        k = Y.shape[1]
        self.models_ = [None] * k
        best = np.full(k, np.inf)
        for spline in (True, False) if self.select else (True,):
            X = self._design(Z, spline)
            if X.shape[1] == 0:
                continue
            loo, coef, x_mean, y_mean = _loo_ridge(X, Y)
            for c in np.flatnonzero(loo < best):
                best[c] = loo[c]
                self.models_[c] = (spline, coef[:, c], x_mean, y_mean[c])
        self.y_mean_ = Y.mean(0)
        return self

    def predict(self, Z):
        Z = np.asarray(Z, dtype=float).reshape(len(Z), -1)
        designs = {}
        out = np.empty((len(Z), len(self.models_)))
        for c, model in enumerate(self.models_):
            if model is None:  # no usable column: predict the training mean
                out[:, c] = self.y_mean_[c]
                continue
            spline, coef, x_mean, y_mean = model
            if spline not in designs:
                designs[spline] = self._design(Z, spline)
            out[:, c] = (designs[spline] - x_mean) @ coef + y_mean
        return out[:, 0] if self.vector_ else out


NUISANCE_LEARNERS = {
    "ridge": _ridge,
    "spline": _spline,
    "tree": _tree,
    "spline_select": SplineSelectRidge,
}


def make_regressor(name):
    """Return a fresh unfitted regressor for a registered learner name."""
    try:
        return NUISANCE_LEARNERS[name]()
    except KeyError:
        raise ValueError(
            f"Unknown nuisance learner {name!r}. "
            f"Available: {sorted(NUISANCE_LEARNERS)}"
        ) from None


class BlockCrossFitter:
    """Produce out-of-fold residuals using contiguous, embargoed folds.

    Unlike a single train/test split (as used by ``splitkci``), every row
    contributes to the final score: each is predicted by a model that was fitted
    without it or its temporal neighbours. At T=40 that difference is the whole
    ballgame, since a 50/50 split leaves 20 effective rows.

    Parameters
    ----------
    n_splits : int
        Number of contiguous folds.
    gap : int or None
        Embargo width. ``None`` means "use the value supplied by
        :meth:`set_gap`", falling back to 1.
    regressors : tuple of str
        Candidate nuisance learners. When more than one is given, the learner is
        chosen per fold by blocked validation *within the training rows only*.
    """

    def __init__(self, n_splits=5, gap=None, regressors=("spline_select",)):
        self.n_splits = n_splits
        self.gap = gap
        self.regressors = tuple(regressors)
        for name in self.regressors:
            if name not in NUISANCE_LEARNERS:
                raise ValueError(
                    f"Unknown nuisance learner {name!r}. "
                    f"Available: {sorted(NUISANCE_LEARNERS)}"
                )
        self._selected = []

    def set_gap(self, gap):
        """Set the embargo width (used when ``gap`` was left as ``None``)."""
        self.gap = int(gap)

    @property
    def effective_gap(self):
        return 1 if self.gap is None else int(self.gap)

    def _select_learner(self, Z_train, y_train):
        """Pick a learner by blocked validation inside the training rows.

        Selection never touches evaluation rows, so it cannot leak into the
        final score.
        """
        if len(self.regressors) == 1:
            return self.regressors[0]

        n = len(y_train)
        if n < 4 * len(self.regressors):
            return self.regressors[0]

        best_name, best_mse = self.regressors[0], np.inf
        for name in self.regressors:
            errs = []
            for tr, va in blocked_folds(n, n_splits=3, gap=self.effective_gap):
                if len(tr) < 3 or len(va) == 0:
                    continue
                model = make_regressor(name)
                try:
                    model.fit(Z_train[tr], y_train[tr])
                    pred = model.predict(Z_train[va])
                except Exception:
                    errs = [np.inf]
                    break
                errs.append(np.mean((y_train[va] - pred) ** 2))
            if errs:
                mse = float(np.mean(errs))
                if mse < best_mse:
                    best_name, best_mse = name, mse
        return best_name

    def residuals(self, y, Z):
        """Return out-of-fold residuals of ``y`` on ``Z``.

        Parameters
        ----------
        y : np.ndarray, shape (n,)
            Response.
        Z : np.ndarray, shape (n, p) or None
            Conditioning matrix. ``None`` or zero columns means the residual is
            simply ``y`` centred.

        Returns
        -------
        np.ndarray, shape (n,)
            Out-of-fold residuals.
        """
        y = np.asarray(y, dtype=float).ravel()
        n = y.shape[0]

        if Z is None or (np.ndim(Z) == 2 and Z.shape[1] == 0):
            return y - y.mean()

        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        n_splits = min(self.n_splits, max(2, n // 10))
        resid = np.empty(n, dtype=float)
        filled = np.zeros(n, dtype=bool)
        self._selected = []

        for train_idx, eval_idx in blocked_folds(
            n, n_splits=n_splits, gap=self.effective_gap
        ):
            if len(train_idx) < 3:
                # Embargo consumed the training set; fall back to the mean so
                # the fold still yields a defined (if uninformative) residual.
                resid[eval_idx] = y[eval_idx] - y.mean()
                filled[eval_idx] = True
                continue
            name = self._select_learner(Z[train_idx], y[train_idx])
            self._selected.append(name)
            model = make_regressor(name)
            try:
                model.fit(Z[train_idx], y[train_idx])
                pred = model.predict(Z[eval_idx])
            except Exception:
                pred = np.full(len(eval_idx), y[train_idx].mean())
            resid[eval_idx] = y[eval_idx] - np.asarray(pred).ravel()
            filled[eval_idx] = True

        if not filled.all():
            resid[~filled] = y[~filled] - y.mean()
        return resid


# ---------------------------------------------------------------------------
# Long-run-variance and bootstrap calibration
# ---------------------------------------------------------------------------


def bartlett_bandwidth(n):
    """Newey--West automatic bandwidth, ``floor(4 * (n/100)^(2/9))``.

    Kept deliberately small and data-independent: selecting a bandwidth by
    inspecting the statistic being tested would invalidate the p-value.
    """
    return max(1, int(np.floor(4.0 * (max(n, 1) / 100.0) ** (2.0 / 9.0))))


def hac_se(q, bandwidth=None):
    """Newey--West standard error of ``mean(q)``.

    Parameters
    ----------
    q : np.ndarray, shape (n,)
        Score series, in time order.
    bandwidth : int or None
        Bartlett lag truncation. ``None`` uses :func:`bartlett_bandwidth`.

    Returns
    -------
    float
        Standard error of the sample mean. Never returns exactly zero.
    """
    q = np.asarray(q, dtype=float).ravel()
    n = q.shape[0]
    if n < 2:
        return np.inf

    if bandwidth is None:
        bandwidth = bartlett_bandwidth(n)
    bandwidth = int(min(bandwidth, n - 1))

    qc = q - q.mean()
    gamma0 = float(np.dot(qc, qc) / n)
    lrv = gamma0
    for h in range(1, bandwidth + 1):
        gamma_h = float(np.dot(qc[h:], qc[:-h]) / n)
        lrv += 2.0 * (1.0 - h / (bandwidth + 1.0)) * gamma_h

    # A negative truncated estimate is possible; fall back to the i.i.d. value
    # rather than emitting a NaN standard error.
    if not np.isfinite(lrv) or lrv <= 0:
        lrv = gamma0
    return float(np.sqrt(max(lrv, 1e-300) / n))


@lru_cache(maxsize=32)
def _dwb_factor(n, block_length):
    """Cholesky factor of a Bartlett correlation matrix.

    The Bartlett (triangular) kernel is positive definite, so the matrix
    ``C[i, j] = max(0, 1 - |i - j| / l)`` is a valid correlation matrix and
    admits a Cholesky factor. Cached because a graph search reuses the same
    ``(n, l)`` thousands of times.
    """
    idx = np.arange(n)
    dist = np.abs(idx[:, None] - idx[None, :])
    corr = np.clip(1.0 - dist / float(block_length), 0.0, None)
    corr.flat[:: n + 1] = 1.0
    jitter = 1e-8
    for _ in range(6):
        try:
            return np.linalg.cholesky(corr + jitter * np.eye(n))
        except np.linalg.LinAlgError:
            jitter *= 10.0
    # Fall back to independent multipliers rather than failing the test.
    return np.eye(n)


def _draw_multipliers(n, n_boot, kind, block_length, rng):
    """Return an ``(n_boot, n)`` array of mean-zero, unit-variance weights."""
    z = rng.standard_normal((n_boot, n))
    if kind == "iid":
        return z
    if kind == "dependent":
        block_length = max(2, int(block_length))
        if block_length >= n:
            block_length = max(2, n // 2)
        factor = _dwb_factor(n, block_length)
        return z @ factor.T
    raise ValueError(f"Unknown multiplier kind {kind!r}; use 'iid' or 'dependent'.")


def multiplier_bootstrap(
    Q,
    n_boot=499,
    kind="dependent",
    block_length=None,
    rng=None,
):
    """Calibrate ``max_j |sqrt(n) * mean(Q[:, j]) / sd_j|`` by multiplier bootstrap.

    Parameters
    ----------
    Q : np.ndarray, shape (n,) or (n, J)
        Score contributions per row, one column per weight in the bank.
    n_boot : int
        Number of bootstrap replicates.
    kind : {"dependent", "iid"}
        ``"iid"`` reproduces the published WGCM calibration and is valid for
        independent rows. ``"dependent"`` uses Shao's dependent wild bootstrap
        and is the correct choice for serially correlated scores.
    block_length : int or None
        Dependence horizon for the dependent bootstrap. ``None`` uses
        ``ceil(n ** (1/3))``.
    rng : np.random.Generator or None
        Random source.

    Returns
    -------
    (float, float)
        ``(p_value, observed_statistic)``. The p-value uses the plus-one
        correction, so it is never exactly zero.
    """
    Q = np.asarray(Q, dtype=float)
    if Q.ndim == 1:
        Q = Q.reshape(-1, 1)
    n, n_weights = Q.shape
    if n < 3:
        return 1.0, 0.0
    if rng is None:
        rng = np.random.default_rng(0)
    if block_length is None:
        block_length = max(2, int(np.ceil(n ** (1.0 / 3.0))))

    centred = Q - Q.mean(axis=0, keepdims=True)
    sd = centred.std(axis=0, ddof=1)
    sd = np.where(sd < 1e-12, np.inf, sd)

    observed = float(np.max(np.abs(np.sqrt(n) * Q.mean(axis=0) / sd)))
    if not np.isfinite(observed):
        return 1.0, 0.0

    xi = _draw_multipliers(n, n_boot, kind, block_length, rng)
    # (n_boot, n) @ (n, J) -> (n_boot, J)
    boot = (xi @ (centred / sd)) / np.sqrt(n)
    null = np.max(np.abs(boot), axis=1)

    p = (1.0 + np.sum(null >= observed)) / (n_boot + 1.0)
    return float(min(max(p, 0.0), 1.0)), observed


def moving_block_bootstrap(q, statistic, n_boot=499, block_length=None, rng=None):
    """Moving-block bootstrap of a scalar ``statistic(q)``.

    Provided for ablation against :func:`multiplier_bootstrap`; the multiplier
    route is preferred for smooth score statistics because it does not distort
    the series length. Blocks are drawn from the mean-centred series, so the
    null distribution is that of a zero-mean location statistic.
    """
    q = np.asarray(q, dtype=float).ravel()
    n = q.shape[0]
    if n < 3:
        return 1.0, 0.0
    if rng is None:
        rng = np.random.default_rng(0)
    if block_length is None:
        block_length = max(2, int(np.ceil(n ** (1.0 / 3.0))))
    block_length = int(min(block_length, n))

    observed = float(statistic(q))
    n_blocks = int(np.ceil(n / block_length))
    max_start = n - block_length + 1
    q0 = q - q.mean()

    null = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        starts = rng.integers(0, max_start, size=n_blocks)
        null[b] = statistic(
            np.concatenate([q0[s : s + block_length] for s in starts])[:n]
        )

    p = (1.0 + np.sum(np.abs(null) >= abs(observed))) / (n_boot + 1.0)
    return float(min(max(p, 0.0), 1.0)), observed


# ---------------------------------------------------------------------------
# Fixed weight banks
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = ("constant", "sign", "rank", "quantile_bins")


def _centred_ranks(v):
    # Average ranks, so tied values (a discrete or constant column) share a
    # weight instead of being ordered by row position. Without ties this equals
    # the plain ordinal rank.
    n = len(v)
    if n < 2:
        return np.zeros(n)
    order = rankdata(v, method="average") - 1.0
    return 2.0 * (order / (n - 1.0)) - 1.0


def _leading_directions(Z, max_dirs):
    """Return up to ``max_dirs`` standardised directions summarising ``Z``.

    With a single conditioning variable this is just that variable. With many,
    the leading principal components stand in for them, which keeps the bank
    size independent of ``|Z|`` -- important because low sample size is usually
    caused by conditioning dimension rather than by ``T`` itself.
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    n, p = Z.shape
    Zc = Z - Z.mean(axis=0, keepdims=True)
    scale = Zc.std(axis=0, ddof=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    Zs = Zc / scale

    if p <= max_dirs:
        return [Zs[:, j] for j in range(p)]

    try:
        _, _, vt = np.linalg.svd(Zs, full_matrices=False)
        comps = Zs @ vt[:max_dirs].T
    except np.linalg.LinAlgError:
        comps = Zs[:, :max_dirs]
    return [comps[:, j] for j in range(comps.shape[1])]


class FeatureBank:
    """Build a fixed ``(n, J)`` matrix of weights from the conditioning set.

    Parameters
    ----------
    weights : tuple of str
        Any of ``"constant"``, ``"sign"``, ``"rank"``, ``"spline"``,
        ``"quantile_bins"``.
    max_dirs : int
        How many directions of ``Z`` each non-constant weight family expands
        over. Keeps the bank small when ``|Z|`` is large.
    max_weights : int
        Hard cap on bank size. More weights means a wider joint null and less
        power per weight, which is the wrong trade at T=40.
    """

    def __init__(self, weights=DEFAULT_WEIGHTS, max_dirs=2, max_weights=8):
        unknown = set(weights) - {
            "constant",
            "sign",
            "rank",
            "spline",
            "quantile_bins",
        }
        if unknown:
            raise ValueError(f"Unknown weight(s): {sorted(unknown)}")
        self.weights = tuple(weights)
        self.max_dirs = int(max_dirs)
        self.max_weights = int(max_weights)
        self.names_ = []

    def build(self, Z, n=None):
        """Return an ``(n, J)`` weight matrix, always including the constant."""
        if Z is None or (np.ndim(Z) == 2 and np.asarray(Z).shape[1] == 0):
            if n is None:
                raise ValueError("n must be given when Z is empty.")
            self.names_ = ["constant"]
            return np.ones((n, 1))

        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        n = Z.shape[0]
        dirs = _leading_directions(Z, self.max_dirs)

        cols, names = [], []
        for kind in self.weights:
            if kind == "constant":
                cols.append(np.ones(n))
                names.append("constant")
                continue
            for j, v in enumerate(dirs):
                if kind == "sign":
                    w = np.sign(v - np.median(v))
                elif kind == "rank":
                    w = _centred_ranks(v)
                elif kind == "spline":
                    # One smooth, mean-zero contrast. Not a full basis: at low T
                    # extra basis functions cost more in null width than they
                    # buy in power.
                    w = v**2 - np.mean(v**2)
                elif kind == "quantile_bins":
                    lo, hi = np.quantile(v, [1.0 / 3.0, 2.0 / 3.0])
                    w = np.where(v <= lo, -1.0, np.where(v >= hi, 1.0, 0.0))
                else:  # pragma: no cover - guarded in __init__
                    continue

                sd = w.std(ddof=0)
                if sd < 1e-12:
                    continue
                cols.append(w / sd)
                names.append(f"{kind}[{j}]")

        if not cols:
            cols, names = [np.ones(n)], ["constant"]
        if "constant" not in names:
            cols.insert(0, np.ones(n))
            names.insert(0, "constant")

        if len(cols) > self.max_weights:
            cols = cols[: self.max_weights]
            names = names[: self.max_weights]

        self.names_ = names
        return np.column_stack(cols)


# ---------------------------------------------------------------------------
# Residual-score base class
# ---------------------------------------------------------------------------


class ResidualScoreCIT(CIT_Base):
    """Base for tests built on out-of-fold residuals of X and Y given Z.

    Parameters
    ----------
    data : np.ndarray, shape (T, n_features)
        Time-ordered observations.
    n_splits : int
        Cross-fitting folds. Ignored when ``crossfit="full"``.
    gap : int or None
        Embargo width. ``None`` lets :meth:`set_lag_structure` supply the
        maximum embedded lag, falling back to 1.
    regressors : tuple of str
        Candidate nuisance learners; see :data:`NUISANCE_LEARNERS`. The default
        ``("spline_select",)`` is :class:`SplineSelectRidge`.
    crossfit : {"blocked", "random", "full"}
        ``"blocked"`` uses contiguous embargoed folds (correct for time series).
        ``"random"`` shuffles rows before folding, reproducing the i.i.d.
        assumption made by ``splitkci``; kept so the difference can be measured
        rather than asserted. ``"full"`` fits the nuisance on all rows, which is
        the original GCM prescription.
    seed : int
        Accepted for interface compatibility. Random draws (bootstrap
        multipliers, ``crossfit="random"`` folds) are seeded from the query and
        the test's settings, so a test's answer does not depend on call order
        and is reproducible; changing ``seed`` does not change them.
    """

    #: What the test's null hypothesis actually is. Surfaced so callers can
    #: tell a full-CI test from a conditional-mean one without reading source.
    inferential_target = "residual_moment"

    def __init__(
        self,
        data,
        method="residual_score",
        n_splits=5,
        gap=None,
        regressors=("spline_select",),
        crossfit="blocked",
        seed=42,
        **kwargs,
    ):
        super().__init__(data=data, method=method, seed=seed, **kwargs)
        if crossfit not in ("blocked", "random", "full"):
            raise ValueError(
                f"crossfit must be 'blocked', 'random' or 'full', got {crossfit!r}"
            )
        self.n_splits = int(n_splits)
        self.gap = gap
        self.regressors = tuple(regressors)
        self.crossfit = crossfit
        self._fitter = BlockCrossFitter(
            n_splits=self.n_splits, gap=gap, regressors=self.regressors
        )
        self._lags = None
        self._n_vars = None

    def get_data(self):
        return self._data

    def _assign_data(self, d):
        # Named apart from set_data so the inherited set_data(data, data_hash),
        # which also invalidates the cache, stays reachable.
        self._data = d
        self._has_nan = bool(np.isnan(d).any()) if d is not None else False

    data = property(get_data, _assign_data)

    def set_lag_structure(self, n_vars, lags):
        """Register the lag embedding so the embargo can size itself.

        Called by ``causalts.cdnots.phase3_utils``. The shortest defensible
        embargo is the maximum embedded lag: below that, a training row shares
        raw observations with the evaluation row it predicts.
        """
        self._n_vars = n_vars
        self._lags = list(lags) if lags is not None else None
        if self.gap is None and self._lags:
            self._fitter.set_gap(max(self._lags))

    def _residuals(self, v, Z, rng, perm=None):
        """Out-of-fold (or full-sample) residuals of ``v`` on ``Z``.

        ``perm`` is the fold permutation for ``crossfit="random"``; pass the same
        one for X and Y so the two fits share folds.
        """
        if self.crossfit == "full":
            v = np.asarray(v, dtype=float).ravel()
            if Z is None or Z.shape[1] == 0:
                return v - v.mean()
            model = make_regressor(self.regressors[0])
            try:
                model.fit(Z, v)
                return v - np.asarray(model.predict(Z)).ravel()
            except Exception:
                return v - v.mean()

        if self.crossfit == "random":
            n = len(v)
            if perm is None or len(perm) != n:
                perm = rng.permutation(n)
            inv = np.argsort(perm)
            res = self._fitter.residuals(
                np.asarray(v).ravel()[perm], None if Z is None else Z[perm]
            )
            return res[inv]

        return self._fitter.residuals(v, Z)

    def _prepare(self, X, Y, condition_set):
        """Return ``(x, y, Z)`` as float arrays, or ``None`` if unusable."""
        if condition_set is None:
            condition_set = []
        condition_set = list(condition_set)

        data = self.data
        if self._has_nan:
            mask = self._pairwise_complete(X, Y, condition_set)
            if mask.sum() < self._min_samples:
                return None
            data = data[mask]

        x = np.asarray(data[:, X], dtype=float).ravel()
        y = np.asarray(data[:, Y], dtype=float).ravel()
        if len(condition_set) > 0:
            Z = np.asarray(data[:, condition_set], dtype=float)
            if Z.ndim == 1:
                Z = Z.reshape(-1, 1)
        else:
            Z = np.zeros((len(x), 0))

        if len(x) < max(self._min_samples, 8):
            return None
        return x, y, Z

    def _statistic(self, ex, ey, Z, rng):  # pragma: no cover - abstract
        """Return ``(p_value, statistic)`` from residuals. Subclasses override."""
        raise NotImplementedError

    def _param_hash(self):
        return (
            f"{self.method}-cf{self.crossfit}-k{self.n_splits}"
            f"-g{self._fitter.effective_gap}-r{'+'.join(self.regressors)}"
        )

    #: Bumped when results change for a fixed ``_param_hash``, so persistent
    #: caches written by older code miss instead of returning stale p-values.
    _CACHE_VERSION = "c2"

    def _cache_hash(self):
        """Cache identity: the seed material plus everything else that shapes
        the result. Kept separate from :meth:`_param_hash`, which also seeds the
        random draws, so adding a field here never changes a default result."""
        return f"{self._param_hash()}-ms{self._min_samples}-{self._CACHE_VERSION}"

    def __call__(self, X, Y, condition_set=None):
        self.n_tests += 1
        if condition_set is None:
            condition_set = []
        # Same order the cache key uses, so one cache entry has one answer.
        condition_set = sorted(condition_set)

        cache_key = self._get_array_hash(X, Y, condition_set)
        param_hash = self._param_hash()
        cache_param = self._cache_hash()
        cached = self.pvalue_cache.get((cache_key, cache_param))
        if cached is not None:
            return cached[0], cached[1]

        prepared = self._prepare(X, Y, condition_set)
        if prepared is None:
            self.pvalue_cache[(cache_key, cache_param)] = [1.0, 0.0]
            return 1.0, 0.0
        x, y, Z = prepared

        self.n_actual_tests += 1
        # Seed from the query so repeated calls are deterministic and a graph
        # search cannot change its answer by reordering its queries.
        rng = np.random.default_rng(stable_seed(cache_key, param_hash))
        # One fold permutation per query, shared by both residual fits, so
        # crossfit="random" does not depend on which of X and Y is fitted first.
        perm = (
            np.random.default_rng(
                stable_seed(cache_key, param_hash, "folds")
            ).permutation(len(x))
            if self.crossfit == "random"
            else None
        )

        ex = self._residuals(x, Z, rng, perm)
        ey = self._residuals(y, Z, rng, perm)
        p, stat = self._statistic(ex, ey, Z, rng)

        if not np.isfinite(p):
            p, stat = 1.0, 0.0
        p = float(min(max(p, 0.0), 1.0))
        stat = float(stat)

        self.pvalue_cache[(cache_key, cache_param)] = [p, stat]
        self.save_incremental_cache(cache_key, cache_param, p, stat)
        return p, stat
