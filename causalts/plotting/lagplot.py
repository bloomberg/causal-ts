# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Lag-dependence plot: dependence of each variable on each other at each lag.

A standalone version of the Cedar lag-importance plot that works on raw data,
without running a discovery algorithm. Scores are pairwise by default, or
adjusted for each variable's selected parents (``adjust="parents"``).
"""

import numpy as np
import pandas as pd

from ..cedar.lag_selection import compute_lag_importance, compute_lag_pvalues
from ._core import lag_grid_layout, setup_matrix
from ._lag_adjust import (
    MIN_ROWS,
    MIN_TRAIN,
    PARENT_ALPHA,
    parent_adjusted_scores,
    smallest_outer_train,
)

_METHODS = ("dcor", "dcor_biased", "pearson", "partial_dcor", "lasso")
# compute_lag_pvalues tests dependence with distance correlation: the t-test
# reads the score as the unbiased squared dcor; the permutation tests rank the
# biased dcor of the raw columns (a valid independence test, but of a
# different statistic). So only the dcor scores get markers, and partial_dcor
# (residualised target) only the t-test. The first entry is the default test.
_PVALUE_METHODS = {
    "dcor": ("circular_shift", "t_test", "permutation"),
    "partial_dcor": ("t_test",),
}


def _as_array(data):
    """Return ``(values, var_names)`` for a DataFrame or 2-D array."""
    if isinstance(data, pd.DataFrame):
        non_numeric = [
            c
            for c, dtype in zip(data.columns, data.dtypes)
            if not pd.api.types.is_numeric_dtype(dtype)
        ]
        if non_numeric:
            raise ValueError(
                f"Non-numeric columns {non_numeric}; drop them or set them as "
                "the index before computing lag dependence."
            )
        return data.to_numpy(dtype=np.float64), [str(c) for c in data.columns]
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"data must be 2-D (T, d), got shape {values.shape}.")
    return values, [f"X{i}" for i in range(values.shape[1])]


_ADJUST = (None, "parents")


def _check_args(method, max_lag, adjust=None, include_lag0=False):
    if isinstance(max_lag, bool) or not isinstance(max_lag, (int, np.integer)):
        raise ValueError(f"max_lag must be an integer >= 1, got {max_lag!r}.")
    if max_lag < 1:
        raise ValueError(f"max_lag must be >= 1, got {max_lag}.")
    if not callable(method) and method not in _METHODS:
        raise ValueError(
            f"Unknown method {method!r}. Choose one of {_METHODS} or pass a "
            "callable metric(source, target) -> float."
        )
    if adjust not in _ADJUST:
        raise ValueError(f"adjust must be one of {_ADJUST}, got {adjust!r}.")
    if adjust == "parents":
        if method != "dcor":
            raise ValueError(
                'adjust="parents" scores by distance correlation; use '
                f'method="dcor", got {method!r}.'
            )


def _check_adjust_data(values, n_permutations, max_lag):
    n = values.shape[0] - 2 * max_lag
    if n < MIN_ROWS or smallest_outer_train(n, max_lag) < MIN_TRAIN:
        raise ValueError(
            f'adjust="parents" needs at least {MIN_ROWS} rows after lagging '
            f"(T - 2 * max_lag) and at least {MIN_TRAIN} training rows in every "
            f"cross-fitting fold after its {2 * max_lag + 2}-row embargo; got "
            f"T={values.shape[0]} for max_lag={max_lag}. Use a longer series or "
            "a smaller max_lag."
        )
    if not np.isfinite(values).all():
        raise ValueError(
            'adjust="parents" needs finite data without NaNs; drop or impute '
            "missing values first."
        )
    if 1 / (n_permutations + 1) >= PARENT_ALPHA:
        raise ValueError(
            f'adjust="parents" selects parents at p < {PARENT_ALPHA}; '
            f"n_permutations must be at least {int(1 / PARENT_ALPHA)}."
        )


def compute_lag_dependence(
    data,
    max_lag,
    method="dcor",
    include_lag0=True,
    n_jobs=1,
    adjust=None,
    n_permutations=200,
    seed=42,
    return_parents=False,
):
    """Compute the dependence of every variable on every variable at each lag.

    Entry ``[i, j, k]`` scores the dependence between ``X_i(t-k)`` and
    ``X_j(t)``. By default one pair and one lag at a time (no conditioning,
    except that ``"partial_dcor"`` removes the target's own AR(1) part first):
    the Phase-0 lag score used by Cedar, exposed for exploratory use. Pairwise
    scores also pick up indirect paths, common causes and autocorrelation.

    With ``adjust="parents"`` each variable's lagged parents are selected
    first, and every score is computed after removing, from both the lagged
    source and the target, the target's other parents and the source's
    (shifted) parents, as in the MCI step of PCMCI, except the source's own
    lags among the shifted source parents (a source lag that is itself one of
    the target's other parents is still removed). For a persistent source
    those lags would remove most of the variation that drives the target and
    bury real lags, so the scores keep them. The cost: a persistent source
    with no effect that is correlated with a true parent the selection
    missed can rank high. Scores are an exploratory ranking; see
    :func:`plot_lag_dependence` for the significance markers. Each adjustment is a ridge
    regression, either linear or on the dictionary of the ``rift`` CI test
    (splines, sines/cosines and pairwise interactions), chosen per fit by
    blocked cross-validation; the dictionary approximates nonlinear parent
    effects without a specified form. Indirect and common-cause dependence
    then drops towards zero, and a weak direct effect hidden by a dominant
    co-parent can show up. The score is the dependence left in the residuals:
    it equals conditional dependence under additive noise, but a common driver
    of both variables' noise *variance* leaves dependent residuals, so a
    nonzero score or a marker is evidence of residual dependence, not proof
    of a direct effect.

    Known weak spots, from simulations with significance markers at
    ``alpha=0.01`` (T=300 unless noted):

    * Chaotic deterministic systems (Henon maps, Lorenz-96): the adjustment
      cannot remove the dynamics; markers were false on 8-23% of non-causal
      lags and the ranking was worse than the pairwise ``"partial_dcor"``.
    * A common driver of noise variance: about 1% false markers at 0.01 but
      7% at 0.05.
    * For a persistent source the marker can sit one lag off the true lag
      (3.5% of neighbouring lags marked).
    * A persistent non-cause correlated with a true parent: marked in 2.5%
      of datasets.
    * Few rows: on spline and threshold mechanisms markers found 63-77% of
      true lags at T=300, 72-86% at T=500 (d=20 and d=10) and nearly all at
      T=1000.

    Parameters
    ----------
    data : pd.DataFrame or ndarray, shape (T, d)
        Numeric time series, one column per variable. NaN rows are dropped
        per (pair, lag); ``"lasso"`` drops them per pair (a NaN in any lagged
        source or the target), and ``"partial_dcor"`` also drops rows where
        the target's previous value is NaN.
    max_lag : int
        Largest lag to score (>= 1).
    method : str or callable, default "dcor"
        ``"dcor"`` (unbiased squared distance correlation, can be slightly
        negative under independence), ``"dcor_biased"``, ``"pearson"``
        (absolute correlation), ``"partial_dcor"``, ``"lasso"`` (absolute
        LassoCV coefficients over all lags jointly), or a callable
        ``metric(source, target) -> float``.
    include_lag0 : bool, default True
        Also score contemporaneous (lag-0) dependence.
    n_jobs : int, default 1
        Parallel workers (needs joblib; not used by ``"partial_dcor"``). With
        ``adjust="parents"`` they run the source-target pairs after the
        parent selection; results do not depend on it.
    adjust : {None, "parents"}, default None
        ``None``: pairwise scores. ``"parents"``: parent-adjusted (MCI-style)
        scores, see above. Requires ``method="dcor"``, finite data and enough
        rows: at least 50 after lagging
        (``T - 2 * max_lag``), and at least 30 training rows per
        cross-fitting fold after an embargo of ``2 * max_lag + 2`` rows on
        each side (roughly ``T >= 50 + 7 * max_lag``). Parents are selected greedily, at most 4 per
        variable: the best candidate is kept while a block-permutation test of
        the maximum over all candidates gives p < 0.01, then parents that add
        nothing given the others are pruned. The selected sets are
        approximate: a picked proxy adjusts well only when it tracks the true
        parent closely relative to the noise. Much slower than pairwise
        scores (about 10-25 s on one core for d=5, max_lag=5, T=500). The
        diagonal holds each variable's own-lag scores, adjusted for its other
        parents.
    n_permutations : int, default 200
        Permutations of the parent-selection test (``adjust="parents"``
        only; at least 100).
    seed : int, default 42
        Seed of the parent-selection test (``adjust="parents"`` only).
    return_parents : bool, default False
        With ``adjust="parents"``, also return the selected parents as
        ``{variable: [(parent, lag), ...]}``.

    Returns
    -------
    ndarray, shape (d, d, max_lag+1)
        Scores indexed ``[cause, effect, lag]``. Pairwise: the lag-0 slice is
        zero when ``include_lag0=False`` and its diagonal is always 1.
        Parent-adjusted: lag 0 holds a symmetric score per pair (both series
        adjusted for their lagged parents; not tested), or 0 when
        ``include_lag0=False``.
    dict
        Only with ``return_parents=True`` and ``adjust="parents"``.
    """
    _check_args(method, max_lag, adjust, include_lag0)
    if return_parents and adjust is None:
        raise ValueError('return_parents needs adjust="parents".')
    values, names = _as_array(data)
    if adjust is None:
        return compute_lag_importance(
            values, max_lag, method=method, include_lag0=include_lag0, n_jobs=n_jobs
        )
    _check_adjust_data(values, n_permutations, max_lag)
    scores, _, parents = parent_adjusted_scores(
        values, max_lag, n_permutations, seed, n_jobs=n_jobs, include_lag0=include_lag0
    )
    if return_parents:
        return scores, _named_parents(parents, names)
    return scores


def _named_parents(parents, names):
    return {
        names[j]: [(names[i], int(k)) for i, k in sel] for j, sel in parents.items()
    }


def _draw_truth(
    matrix, truth, d, max_lag, include_lag0, sig, shown, true_c, miss_c, lag0_tested
):
    """Shade true lags; with markers, flag missed true lags and false markers."""
    G = np.zeros((d, d, max_lag + 1), dtype=bool)
    L = min(truth.shape[2], max_lag + 1)
    G[:, :, :L] = truth[:, :, :L]
    if not include_lag0:
        G[:, :, 0] = False
    G[np.arange(d), np.arange(d), 0] = False
    cross = ~np.eye(d, dtype=bool)[:, :, None] & np.ones(max_lag + 1, dtype=bool)
    if not lag0_tested:
        cross[:, :, 0] = False  # untested lag 0: shaded, not counted
    # Lag-0 scores have no direction: a marker either way of a true lag-0
    # link counts as finding it, and neither direction is a false positive.
    G_sym = G.copy()
    G_sym[:, :, 0] |= G[:, :, 0].T
    sig_sym = None if sig is None else sig.copy()
    if sig is not None:
        sig_sym[:, :, 0] |= sig[:, :, 0].T
    missed = np.zeros_like(G) if sig is None else G & cross & ~sig_sym
    false = np.zeros_like(G) if sig is None else sig & ~G_sym & cross
    for i, j, k in zip(*np.nonzero(G)):
        matrix.axes_dict[(i, j)].axvspan(
            k - 0.35,
            k + 0.35,
            color=miss_c if missed[i, j, k] else true_c,
            alpha=0.18,
            lw=0,
            zorder=0,
        )
    # Crosses only where the point is drawn; counts include hidden ones.
    for i, j, k in zip(*np.nonzero(false & np.isfinite(shown))):
        matrix.axes_dict[(i, j)].plot(
            k, shown[i, j, k], marker="x", color="black", markersize=6, zorder=5
        )
    n_found = int((G & cross).sum() - missed.sum())
    entries = [
        (f"true lag ({n_found} found)" if sig is not None else "true lag", true_c)
    ]
    if sig is not None:
        entries.append((f"true lag, missed ({int(missed.sum())})", miss_c))
    for label, c in entries:
        matrix.labels.append((label, c, "s", 8, 0.35))
    if sig is not None:
        matrix.labels.append(
            (f"false positive ({int(false.sum())})", "black", "x", 6, 1)
        )


def _nice_step(span):
    """A 1/2/5 x 10^k tick step giving about four ticks over ``span``."""
    raw = span / 4.0
    exp = np.floor(np.log10(raw))
    for m in (1.0, 2.0, 5.0, 10.0):
        if m * 10**exp >= raw:
            return float(m * 10**exp)


def plot_lag_dependence(
    data,
    max_lag,
    method="dcor",
    include_lag0=True,
    var_names=None,
    alpha=None,
    pvalue_method=None,
    n_permutations=200,
    seed=42,
    n_jobs=1,
    figsize=None,
    minimum=None,
    maximum=None,
    x_base=1,
    y_base=None,
    show_gridlines=True,
    color="black",
    sig_color="#D55E00",
    self_color="#999999",
    fig_name=None,
    adjust=None,
    wrap_labels=None,
    true_graph=None,
    min_score=None,
    true_color="#0072B2",
    missed_color="#CC3311",
):
    """Plot the lag dependence of every variable pair as a grid of lag functions.

    Cell ``(i, j)`` shows the score of ``X_i(t-k)`` against ``X_j(t)`` for each
    lag ``k`` (see :func:`compute_lag_dependence`). It answers "at which lags
    does X_i carry information about X_j?" before choosing ``max_lag`` or
    running discovery. By default scores are pairwise and unconditional, so
    indirect and autocorrelation-driven dependence shows up too;
    ``adjust="parents"`` removes each variable's selected parents first, which
    removes most indirect and common-cause dependence (see
    :func:`compute_lag_dependence` for what it does and does not guarantee). The
    trivial lag-0 self-dependence on the diagonal is not drawn.

    Parameters
    ----------
    data : pd.DataFrame or ndarray, shape (T, d)
        Numeric time series, one column per variable.
    max_lag : int
        Largest lag to plot (>= 1).
    method : str or callable, default "dcor"
        Dependence score; see :func:`compute_lag_dependence`.
    include_lag0 : bool, default True
        Also plot lag-0 dependence. With ``adjust="parents"`` it is a
        symmetric score (no direction at lag 0), drawn in ``self_color`` and
        not tested.
    var_names : list of str, optional
        Variable labels. Defaults to the DataFrame columns, else ``X0, X1, ...``.
    alpha : float, optional
        If given, highlight lags with p-value below ``alpha`` in ``sig_color``.
        Supported for ``method="dcor"`` and ``"partial_dcor"`` (the tests are
        distance-correlation tests; the permutation tests rank the biased dcor
        rather than the plotted unbiased score). Requires finite data (no NaN).
        P-values are not multiplicity-corrected, and the diagonal
        (autodependence) is not tested.
    pvalue_method : {"circular_shift", "t_test", "permutation", "block"}, optional
        Test used when ``alpha`` is set (see
        :func:`causalts.cedar.lag_selection.compute_lag_pvalues`). Defaults to
        ``"circular_shift"`` for ``"dcor"``: it keeps the target's
        autocorrelation, whereas the asymptotic ``"t_test"`` is faster but
        flags many more lags on autocorrelated series. ``"partial_dcor"``
        supports only ``"t_test"`` (its default). The permutation tests cost
        about ``d**2 * max_lag * n_permutations`` dcor evaluations (roughly 20 s
        for d=10, max_lag=5, T=1000); use ``"t_test"`` for a quick look. With
        ``adjust="parents"`` the only test, and the default, is ``"block"``,
        block permutations of the adjusted residuals in two steps. First each
        source-target pair is tested over all its lags at once (maximum dcor
        over lags; every source lag and the target residualized on the
        target's selected parents other than lags of this source). Only in a
        pair that rejects is the highest-scoring lag marked, plus any other
        lag whose own full-MCI test (also adjusted for the source's own past)
        rejects. A test per lag alone splits a persistent source's evidence
        over neighbouring lags and misses many true lags at a few hundred
        rows. The peak lag has no test of its own, so for a persistent source
        its marker can sit on a neighbour of the true lag. In our stationary
        simulations false markers stayed near ``alpha`` or below; that is
        empirical, not a guarantee.
    n_permutations : int, default 200
        Permutations for the permutation tests. The smallest attainable
        p-value is ``1 / (n_permutations + 1)``, which must be below ``alpha``.
    seed : int, default 42
        Seed for the permutation tests.
    n_jobs : int, default 1
        Parallel workers for the scores.
    figsize : tuple, optional
        Figure size; defaults to ``(1.5 d + 0.6, 1.4 d + 0.9)``.
    minimum, maximum : float, optional
        Y-axis range; defaults to the range of the plotted scores (including 0).
        Scores outside a user-supplied range are not drawn.
    x_base : int, default 1
        X-tick spacing in lags.
    y_base : float, optional
        Y-tick spacing; chosen automatically by default.
    show_gridlines : bool, default True
    color : str, default "black"
        Marker colour for the scores.
    sig_color : str, default "#D55E00"
        Marker colour for significant lags (when ``alpha`` is set).
    self_color : str, default "#999999"
        Marker colour for the diagonal (a variable's own lags) when ``alpha``
        is set; the tests cover cross pairs only, so these are not tested.
    fig_name : str, optional
        If given, save the figure to this path.
    true_color, missed_color : str
        Shading of true lags, and of true lags without a marker.
    min_score : float, optional
        With ``alpha``, mark only lags whose score is also at least this
        value: a practical-relevance filter on top of significance (at large
        T a negligible dependence can still test significant). Off by
        default.
    true_graph : array-like, shape (d, d, L+1), optional
        Known graph indexed ``[cause, effect, lag]`` (e.g. a simulation's
        ground truth or a result's ``cg_tig``); nonzero entries are true lags.
        They are shaded; with ``alpha``, true cross lags without a marker are
        shaded in ``missed_color`` and markers on lags that are not true get a
        cross. The legend counts follow the tests, so markers hidden by a
        user ``minimum``/``maximum`` still count (no cross is drawn for them).
        Lags beyond ``max_lag`` are ignored.
    wrap_labels : int, optional
        Wrap variable names longer than this many characters onto several
        lines, breaking after ``_``, ``-``, ``.`` or spaces where possible.
        Row names longer than 4 characters are always drawn vertically.
    adjust : {None, "parents"}, default None
        ``"parents"`` plots parent-adjusted (MCI-style) scores instead of
        pairwise ones; see :func:`compute_lag_dependence` for the method,
        requirements and cost. The diagonal shows each variable's own-lag
        scores (untested; strong ones above the axis as triangles with their
        value). Markers: see
        ``pvalue_method``. A high score without a marker can mean the lag adds
        little beyond the pair's peak lag or the source's own history, or that
        the test lacks power; it is not evidence of a negligible effect.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_args(method, max_lag, adjust, include_lag0)
    if alpha is None and min_score is not None:
        raise ValueError("min_score filters significance markers; set alpha too.")
    if alpha is None and pvalue_method is not None:
        raise ValueError(
            "pvalue_method is only used for significance markers; set alpha too."
        )
    if alpha is not None:
        if adjust == "parents":
            allowed = ("block",)
        else:
            allowed = _PVALUE_METHODS.get(method if isinstance(method, str) else None)
        if allowed is None:
            raise ValueError(
                "alpha (significance markers) is supported only for "
                f"method in {tuple(_PVALUE_METHODS)}, got {method!r}."
            )
        if pvalue_method is None:
            pvalue_method = allowed[0]
        if pvalue_method not in allowed:
            raise ValueError(
                f"pvalue_method={pvalue_method!r} does not test method="
                f"{method!r}; use one of {allowed}."
            )
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if pvalue_method != "t_test" and alpha <= 1 / (n_permutations + 1):
            raise ValueError(
                f"alpha={alpha:g} is not attainable with n_permutations="
                f"{n_permutations} (smallest p-value 1/{n_permutations + 1}); "
                "raise n_permutations or alpha."
            )

    values, default_names = _as_array(data)
    if alpha is not None and not np.isfinite(values).all():
        # The p-value helpers do not apply the scores' per-pair NaN masks
        # (wrong sample size in the t-test, NaN null statistics in the
        # permutation tests), so the markers would be wrong.
        raise ValueError(
            "alpha (significance markers) needs finite data without NaNs; "
            "drop or impute missing values first. The scores alone "
            "(alpha=None) handle NaNs."
        )
    if adjust == "parents":
        _check_adjust_data(values, n_permutations, max_lag)
    d = values.shape[1]
    if var_names is None:
        var_names = default_names
    elif len(var_names) != d:
        raise ValueError(f"var_names has {len(var_names)} entries but data has {d}.")
    if true_graph is not None:
        tg = np.asarray(true_graph)
        if tg.ndim != 3 or tg.shape[:2] != (d, d):
            raise ValueError(
                f"true_graph must have shape (d, d, L+1) with d={d}, got {tg.shape}."
            )

    pvalues = None
    if adjust == "parents":
        scores, pvalues, _ = parent_adjusted_scores(
            values,
            max_lag,
            n_permutations,
            seed,
            pvalues=alpha is not None,
            alpha=alpha,
            n_jobs=n_jobs,
            include_lag0=include_lag0,
        )
    else:
        scores = compute_lag_importance(
            values, max_lag, method=method, include_lag0=include_lag0, n_jobs=n_jobs
        )
    shown = scores.astype(float)
    if not include_lag0:
        shown[:, :, 0] = np.nan
    np.fill_diagonal(shown[:, :, 0], np.nan)
    if adjust == "parents":
        shown[np.arange(d), np.arange(d), 0] = np.nan

    in_range = shown
    if adjust == "parents":
        # Scale to the cross pairs: strong own lags would squash them. Own lags
        # above the axis are drawn at the top edge with their value.
        in_range = shown.copy()
        in_range[np.arange(d), np.arange(d), :] = np.nan
    finite = in_range[np.isfinite(in_range)]
    user_range = minimum is not None or maximum is not None
    if minimum is not None and maximum is not None and maximum <= minimum:
        raise ValueError(
            f"maximum ({maximum}) must be greater than minimum ({minimum})."
        )
    if maximum is None:
        maximum = max(0.0, float(finite.max())) if finite.size else 1.0
    if minimum is None:
        minimum = min(0.0, float(finite.min())) if finite.size else 0.0
        # The unbiased dcor dips slightly below zero under independence; do
        # not let that round the axis down a whole tick step.
        if minimum > -0.05 * (maximum - minimum):
            minimum = 0.0
    if maximum <= minimum:
        maximum = minimum + 1.0
    if y_base is None:
        y_base = _nice_step(maximum - minimum)
    if figsize is None:
        figsize = (1.5 * d + 0.6, 1.4 * d + 0.9)

    matrix = setup_matrix(
        N=d,
        tau_max=max_lag,
        minimum=minimum,
        maximum=maximum,
        x_base=x_base,
        y_base=y_base,
        plot_gridlines=show_gridlines,
        **lag_grid_layout(var_names, figsize, wrap_labels),
    )
    # Before any axis clipping: the true_graph counts follow the tests.
    scored = np.isfinite(shown)
    pre_range = shown
    if user_range:
        # add_lagfuncs draws with clip_on=False, so out-of-range scores would
        # land in neighbouring panels; drop them instead of clipping, which
        # would also cut in-range markers at the panel edges.
        lo, hi = next(iter(matrix.axes_dict.values())).get_ylim()
        shown = np.where((shown < lo) | (shown > hi), np.nan, shown)
    label = method if isinstance(method, str) else getattr(method, "__name__", None)
    if isinstance(label, str):
        label = label.lstrip("_") or "metric"  # matplotlib hides "_..." labels
    if adjust == "parents":
        label = "score, parent-adjusted"
    self_pair = np.zeros(shown.shape, dtype=bool)
    self_pair[np.arange(d), np.arange(d), :] = True
    # Own lags outside the axis get an edge triangle, so take them from before
    # the user-range drop.
    own_values = pre_range
    if adjust == "parents":
        shown = np.where(self_pair, np.nan, shown)
    lag0_values = None
    if adjust == "parents" and include_lag0:
        # Adjusted lag 0 is a symmetric, untested score: grey, never marked.
        lag0 = np.zeros_like(self_pair)
        lag0[:, :, 0] = True
        lag0 &= ~self_pair
        lag0_values = np.where(lag0, shown, np.nan)
        shown = np.where(lag0, np.nan, shown)
    if alpha is None:
        matrix.add_lagfuncs(val_matrix=shown, color=color, label=label)
    else:
        if pvalues is None:
            pvalues = compute_lag_pvalues(
                values,
                scores,
                max_lag,
                method=pvalue_method,
                n_permutations=n_permutations,
                seed=seed,
                include_lag0=include_lag0,
            )
        sig = pvalues < alpha
        if min_score is not None:
            sig &= np.nan_to_num(scores, nan=-np.inf) >= min_score
        sig_cross = sig & ~self_pair
        # Only cross pairs are tested: the diagonal (each variable's own lags)
        # keeps p = 1, so draw it as untested rather than as not significant.
        matrix.add_lagfuncs(
            val_matrix=np.where(sig | self_pair, np.nan, shown),
            color=color,
            label=label,
        )
        matrix.add_lagfuncs(
            val_matrix=np.where(sig & ~self_pair, shown, np.nan),
            color=sig_color,
            label=f"significant, p < {alpha:g}"
            + ("" if min_score is None else f", score >= {min_score:g}"),
        )
        if adjust is None:
            matrix.add_lagfuncs(
                val_matrix=np.where(self_pair, shown, np.nan),
                color=self_color,
                label="own lag (not tested)",
            )
    if lag0_values is not None:
        matrix.add_lagfuncs(
            val_matrix=lag0_values, color=self_color, label="lag 0 (not tested)"
        )
    if adjust == "parents":
        bottom, top = next(iter(matrix.axes_dict.values())).get_ylim()
        # A point slightly past an edge stays in its panel's padding; only
        # clear overflow gets the edge triangle.
        slack = 0.05 * (top - bottom)
        above = self_pair & (own_values > top + slack)
        below = self_pair & (own_values < bottom - slack)
        own = np.where(above | below, np.nan, own_values)
        # Own lags: plain scores, untested (no markers, no legend entry).
        matrix.add_lagfuncs(val_matrix=np.where(self_pair, own, np.nan), color=color)
        # Own lags outside the axis (scaled to the cross pairs): a triangle at
        # the edge with the value, never the point itself, which would land
        # in a neighbouring panel.
        for mask, edge, marker, dy, va in (
            (above, top, "^", -5, "top"),
            (below, bottom, "v", 5, "bottom"),
        ):
            for i, _, k in zip(*np.nonzero(mask)):
                ax = matrix.axes_dict[(i, i)]
                ax.plot(
                    k, edge, marker=marker, color=color, markersize=6, clip_on=False
                )
                ax.annotate(
                    f"{own_values[i, i, k]:.2f}",
                    (k, edge),
                    xytext=(0, dy),
                    textcoords="offset points",
                    ha="center",
                    va=va,
                    rotation=90,
                    fontsize=6,
                    color=color,
                    annotation_clip=False,
                )
    if true_graph is not None:
        _draw_truth(
            matrix,
            np.asarray(true_graph) != 0,
            d,
            max_lag,
            include_lag0,
            None if alpha is None else sig_cross & scored,
            shown,
            true_color,
            missed_color,
            adjust is None,
        )
    matrix.savefig(name=fig_name)
    return matrix.fig
