# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

_OWN_LAGS = 5
# Columns of the cubic spline basis per lag: x, x^2, x^3 and n knots. The mean
# model uses the richest basis the sample size allows.
_SPLINE_COLS = 7
_MEAN_SPLINE_SIZES = (31, 23, 15, 11, 7, 4)


def _standardize(a):
    sd = a.std(axis=0)
    sd[sd == 0] = 1.0
    return (a - a.mean(axis=0)) / sd


def _normal_scores(x):
    from scipy import stats

    return stats.norm.ppf((stats.rankdata(x) - 0.5) / len(x))


def _spline_basis(x, n_knots=_SPLINE_COLS - 3):
    """Cubic truncated-power basis (x, x^2, x^3, (x-k)^3_+) on normal scores.

    With the linear term it spans the full cubic spline space for its knots, and
    the knots (quantiles of symmetric normal scores) are symmetric, so the basis
    does not change when ``x`` changes sign.

    Built on the normal scores of ``x``: cubes of a heavy-tailed variable are a
    few extreme leverage points that the mean fit chases, leaving small residuals
    exactly where ``|x|`` is large -- a spurious "variance effect".
    """
    x = _normal_scores(x)
    knots = np.quantile(x, np.linspace(0, 1, n_knots + 2)[1:-1])
    cols = [x, x**2, x**3] + [np.clip(x - k, 0, None) ** 3 for k in knots]
    return _standardize(np.column_stack(cols))


def _scale_block(xi):
    """Scale-model terms of X_i: per lag, a cubic spline in its normal scores.

    Normal scores give heavy-tailed X no leverage; the spline lets the test see
    non-monotone scale effects (e.g. noise that grows only in a band of X).
    """
    cols = []
    for k in range(xi.shape[1]):
        cols.append(_spline_basis(xi[:, k]))
    return _standardize(np.column_stack(cols))


def _orth(M, rtol=1e-8, scale=None):
    """Orthonormal basis of the column space of ``M``, truncated to its numerical rank.

    Plain QR keeps a column per input column even when ``M`` is rank-deficient,
    and the extra directions are arbitrary (they change, e.g., when a variable
    changes sign). Singular values below ``rtol * scale`` are dropped; ``scale``
    defaults to the largest singular value of ``M`` (pass the norm of the matrix
    ``M`` was residualised from, so that columns it no longer spans drop out).
    """
    U, sv, _ = np.linalg.svd(M, full_matrices=False)
    if not sv.size:
        return U
    return U[:, sv > rtol * (sv[0] if scale is None else scale)]


def check_variance_dependence(
    data, alpha=0.01, max_lag=1, min_effect=0.05, test_lags=None
):
    """Test for variance-only parents: lagged X_i that change the noise scale of Y_j.

    Guides the choice between mean-dependence tools (REAP, ``raft``, ``parcorr``)
    and tools that also see dependence through the conditional variance (GRACE with
    ``grow_variance``, ``rcot``, ``kci``).

    For every ordered pair (i, j), i != j:

    1. **Mean model.** Y_j(t) is regressed on all variables at lags 1..``max_lag``
       (a linear VAR) plus an additive cubic spline in every variable's lags (on
       their normal scores), so that a nonlinear *mean* effect is not mistaken
       for a variance effect. When T is too small, the VAR lag and then the
       spline basis are reduced, and in the last resort only the tested X_i gets
       a spline.
    2. **Scale model.** The ranks of the absolute residuals ``|e_j(t)|`` are
       regressed on their own lags (at least 5, absorbing ARCH-type volatility
       clustering) with and without, per tested lag, the normal scores of
       ``X_i(t-l)`` and a cubic spline in them (so non-monotone effects, e.g.
       noise that grows only in a band of X_i, are seen). The partial F-test of
       the X_i block, with degrees of freedom from matrix ranks, is a
       Glejser-type heteroskedasticity test; ranks keep heavy tails from
       dominating it. The scale model also controls for scale changes shared by
       all series: the mean ranked ``|e|`` of the other series at times t,
       t-1, ... (a common volatility factor), so a volatility regime that hits
       every series is not read as X_i driving Y_j.

    The p-values are corrected with Benjamini-Hochberg across all pairs, and a
    flagged pair must also have an effect of at least ``min_effect``: the share of
    the variance of the ranked ``|e_j|`` the X_i block explains beyond what its
    degrees of freedom would explain by chance. The floor stops small misfits of
    the mean model from switching the recommendation.

    In simulations (VAR chains, d = 5 and 25, T = 200-1000, ``max_lag=3``) the
    check flagged 0-5% of panels without variance parents under Gaussian, t(3) and
    own-GARCH noise and nonlinear and threshold mean effects (one cell at 20%:
    t(3) noise, d = 25, T = 500), and up to 15-30% at T <= 500 for a strongly
    nonlinear mean on every link (``2 sin(3 X)``) or a volatility level shared by
    all series (a regime break or common stochastic volatility); 0-20% at
    T = 1000. It flagged 95-100% of panels with a moderate variance parent at
    T >= 500 (noise scale ``exp(0.6 X_i)`` at lag 1 or 3, or rising 6-fold in a
    band of ``|X_i|``); weak effects (``exp(0.3 X_i)``) are often missed. A
    variable that drives the noise scale of nearly every other series looks like
    a common volatility factor and is mostly missed at d = 25. On the built-in
    ``ex2`` and ``ex3`` datasets it flagged none of 20 seeds; on ``ex1`` (a
    squared effect of a near-unit-root driver) 0-20%. Interactions in the mean
    (e.g. ``(X_i + X_k)^2``, as in ``henon``) are outside the additive mean model
    and are usually flagged. So is a series that tracks the level of a
    self-heteroskedastic series (one whose noise scale follows its own level),
    e.g. a child of it: controlling for the target's own level would remove that,
    but would also hide genuine variance parents in feedback systems. Treat the result as a recommendation input, not a
    test of record.

    Parameters
    ----------
    data : numpy.ndarray or pandas.DataFrame
        Array of shape (T, N) with T observations and N variables. Must be finite.
    alpha : float, optional
        False discovery rate for the Benjamini-Hochberg correction (default 0.01).
    max_lag : int, optional
        Lags of the mean model (default 1; must be >= 1). Set it to the process's
        lag order: a mean effect at an omitted lag stays in the residual and can
        look like a variance effect. Reduced when T is too small for the full
        model.
    test_lags : int, optional
        Lags at which scale effects of X_i are tested (default ``max_lag``).
    min_effect : float, optional
        Minimum effect of the X_i block in the scale model (default 0.05): its
        partial R^2 net of the chance gain of its degrees of freedom.

    Returns
    -------
    dict
        Dict with keys:

        - results: list of dicts with x, y, pvalue, partial_r2 (the net effect
          above), is_variance_parent
        - n_significant: number of pairs flagged after BH correction
        - detected: True if any pair is flagged, False if none, None if the test
          could not run (non-finite data, too few rows, N < 2, or no testable
          pair)
        - skipped_targets: column indices with no null scale model, so never
          tested as a target: their residual size does not vary beyond rounding
          (constant columns, or periodic ones the mean model fits exactly), or
          the other series' noise scale already explains it (e.g. an exact
          duplicate in a two-column panel). Near-duplicate columns
          (absolute correlation above 0.999) are not tested against each other.
        - summary: human-readable recommendation string
    """
    from scipy import stats

    if hasattr(data, "values"):
        data = data.values
    data = np.asarray(data, dtype=np.float64)
    T, N = data.shape
    if int(max_lag) < 1:
        raise ValueError(f"max_lag must be >= 1, got {max_lag}")
    L = int(max_lag)
    p = L if test_lags is None else min(max(int(test_lags), 1), L)

    unavailable = {
        "results": [],
        "n_significant": 0,
        "detected": None,
        "skipped_targets": list(range(N)),
        "summary": (
            "Variance dependence not checked (non-finite data, too few rows or "
            "degenerate columns)."
        ),
    }
    if N < 2 or not np.isfinite(data).all():
        return unavailable

    # Mean model = linear VAR + an additive spline in every variable's lags, so a
    # nonlinear mean effect of ANY variable (not just the tested one) stays out
    # of the residual. Keep at least ~3 rows per parameter; with too few rows,
    # shrink the VAR lag, then the additive basis, and finally fall back to a
    # spline in the tested variable only (added per pair).
    budget = (T - L) / 3 - 1
    p_mean = L
    while p_mean >= 1 and N * p_mean + _SPLINE_COLS * p > budget:
        p_mean -= 1
    if p_mean < 1:
        return unavailable
    # Shared additive basis: (spline lags, basis size), richest that fits. If it
    # does not cover the tested lags with a full basis, the tested variable gets
    # its own spline per pair.
    # Covering every lag of the mean model comes first (an omitted nonlinear
    # lag leaks into the residual of every pair); then the richest basis.
    additive = None
    for lags in sorted({p_mean, p, 1}, reverse=True):
        for k in _MEAN_SPLINE_SIZES:
            covers_tested = k >= _SPLINE_COLS and lags >= p
            extra = 0 if covers_tested else _SPLINE_COLS * p
            if lags <= p_mean and N * p_mean + N * lags * k + extra <= budget:
                additive = (lags, k)
                break
        if additive is not None:
            break
    per_pair_spline = not (
        additive is not None and additive[1] >= _SPLINE_COLS and additive[0] >= p
    )

    z = _standardize(data)
    rows = np.arange(L, T)
    n = len(rows)
    base = [np.ones((n, 1))] + [z[rows - lag] for lag in range(1, p_mean + 1)]
    if additive is not None:
        lags, k = additive
        base += [
            _spline_basis(z[rows - lag, v], n_knots=k - 3)
            for lag in range(1, lags + 1)
            for v in range(N)
        ]
    # The shared part of the mean model is partialled out once (Frisch-Waugh-
    # Lovell); with a per-pair spline each pair then only regresses the target's
    # residual on X_i's residualised spline terms.
    Q = _orth(np.column_stack(base))

    def _partial_out(M):
        return M - Q @ (Q.T @ M)

    y_res = _partial_out(z[rows])  # (n, N): every target's residual
    q = max(p, _OWN_LAGS)
    # A target whose noise scale does not vary beyond rounding (a constant
    # column, one the mean model reproduces exactly such as a duplicate, or an
    # exactly periodic series with residuals of constant size) has nothing to
    # test; ranking |e| would turn rounding noise into a signal.
    y_sd = z[rows].std(axis=0)
    # std of identical values can come out as ~1e-17, not 0: test the range.
    y_varies = np.ptp(z[rows], axis=0) > 0

    def _ranked_abs(resid):
        # Ranks of |e|: heavy-tailed noise puts a few huge residuals in the
        # scale regression, which a raw |e| (or e^2) fit chases.
        alive = y_varies & (np.abs(resid).std(axis=0) > 1e-6 * y_sd)
        E = np.column_stack([stats.rankdata(np.abs(resid[:, j])) / n for j in range(N)])
        return E, alive

    def _scale_null(E, alive, j):
        """Null scale model of target j: orthonormal basis, residual, RSS."""
        if not alive[j]:
            return None
        e = E[:, j]
        # Scale model on rows where the own |e| lags exist. Volatility
        # clustering is persistent, so absorb at least _OWN_LAGS own lags.
        target = e[q:]
        own = np.column_stack([e[q - lag : n - lag] for lag in range(1, q + 1)])
        # Y_j's own lagged levels are deliberately not a control: in a feedback
        # system (X_i driven by Y_j's level) they absorb a genuine X_i -> Y_j
        # variance effect. The price is that a series tracking the level of a
        # self-heteroskedastic Y_j can be flagged (see the docstring).
        controls = [np.ones((len(target), 1)), own]
        # Common volatility: the mean ranked |e| of the other series at t, t-1,
        # ..., t-q. A volatility regime shared by all series makes a large |X_i|
        # predict a large |e_j| without any link from X_i to Y_j. (A smooth time
        # trend was tried as well; with a persistent X_i it fit spurious effects
        # and added nothing.)
        others = [k for k in range(N) if k != j and alive[k]]
        if others:
            c = E[:, others].mean(axis=1)
            controls.append(
                np.column_stack([c[q - lag : n - lag] for lag in range(q + 1)])
            )
        Q0 = _orth(np.column_stack(controls))
        t_res = target - Q0 @ (Q0.T @ target)
        rss0 = float(t_res @ t_res)
        # (Almost) nothing left to explain: a duplicate or near-duplicate series
        # makes the common factor (nearly) equal the target, and an F-test on
        # the few ranks that differ is meaningless.
        if rss0 <= 1e-3 * float(((target - target.mean()) ** 2).sum()):
            return None
        return Q0, t_res, rss0

    # Scale-model terms of every variable's tested lags.
    lagged = [
        np.column_stack([z[rows - lag, v] for lag in range(1, p + 1)]) for v in range(N)
    ]
    scale_blocks = [_scale_block(xv[q:]) for xv in lagged]

    # Without a per-pair spline the residuals, and so each target's null scale
    # model, are the same for every tested X_i: build them once.
    shared = None
    if not per_pair_spline:
        E_shared, alive_shared = _ranked_abs(y_res)
        shared = [_scale_null(E_shared, alive_shared, j) for j in range(N)]

    # Near-duplicate columns are not tested against each other: X_i would carry
    # Y_j's own level, not an effect of another variable.
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.nan_to_num(np.corrcoef(z[rows].T))
    duplicate = np.abs(corr) > 0.999

    results = []
    testable = set()
    for i in range(N):
        xi = lagged[i]
        if per_pair_spline:
            spl = _partial_out(
                np.column_stack([_spline_basis(xi[:, k]) for k in range(p)])
            )
            Qs = _orth(spl)
            # residuals for (i, j)
            E, alive = _ranked_abs(y_res - Qs @ (Qs.T @ y_res))
        x_block = scale_blocks[i]
        x_scale = np.linalg.norm(x_block, 2)
        for j in range(N):
            if i == j:
                continue
            null = shared[j] if shared is not None else _scale_null(E, alive, j)
            if null is None:
                continue
            testable.add(j)
            if duplicate[i, j]:
                continue
            Q0, t_res, rss0 = null
            # The X_i block beyond the null model; its rank is the test's df1 (a
            # discrete X_i makes some columns collinear).
            xb = x_block - Q0 @ (Q0.T @ x_block)
            Qx = _orth(xb, scale=x_scale)
            df1 = Qx.shape[1]
            df2 = len(t_res) - Q0.shape[1] - df1
            if df1 <= 0 or df2 < 10:
                continue
            proj = Qx.T @ t_res
            rss1 = rss0 - float(proj @ proj)
            if rss1 <= 0:
                continue
            F = ((rss0 - rss1) / df1) / (rss1 / df2)
            pval = float(stats.f.sf(F, df1, df2))
            # Effect size net of the gain df1 noise columns would give by chance.
            r2 = max(rss0 - rss1 - df1 * rss1 / df2, 0.0) / rss0
            results.append({"x": i, "y": j, "pvalue": pval, "partial_r2": r2})

    if not results:
        skipped = [j for j in range(N) if j not in testable]
        return {
            **unavailable,
            "summary": (
                "Variance dependence not checked: no testable pair (targets "
                f"without a noise scale to test: columns {skipped}; others are "
                "duplicates of each other, or the sample is too short)."
            ),
            "skipped_targets": skipped,
        }

    # Benjamini-Hochberg step-up across all tested pairs.
    pvals = np.array([r["pvalue"] for r in results])
    m = len(pvals)
    order = np.argsort(pvals)
    passed = pvals[order] <= alpha * np.arange(1, m + 1) / m
    k = int(np.max(np.nonzero(passed)[0]) + 1) if passed.any() else 0
    flagged = np.zeros(m, dtype=bool)
    flagged[order[:k]] = True
    for r, f in zip(results, flagged):
        r["is_variance_parent"] = bool(f) and r["partial_r2"] >= min_effect
    k = sum(r["is_variance_parent"] for r in results)

    if k > 0:
        summary = (
            f"{k}/{m} pairs show a lagged effect on the noise scale (BH q={alpha}, "
            f"partial R^2 >= {min_effect}). "
            "Mean-only tools (REAP, raft, parcorr) cannot see these edges; "
            "prefer GRACE (grow_variance) and rcot."
        )
    else:
        summary = (
            "No variance-only dependence detected. "
            "Mean-dependence tools (REAP, raft, parcorr) should suffice."
        )
    skipped = [j for j in range(N) if j not in testable]
    if skipped:
        summary += (
            f" Not tested as targets (no noise scale to test): columns {skipped}."
        )
    return {
        "results": results,
        "n_significant": k,
        "detected": k > 0,
        "skipped_targets": skipped,
        "summary": summary,
    }
