# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Contemporaneous (lag-0) edges for refine="shrink": CDNOTS+ with the lagged graph pinned."""

import numpy as np
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

from ..ci_tests.cit_test import CIT_Base
from ..ci_tests.raft import _cauchy
from .grow import _nuisance_block, spline_basis
from .shrink import _orthonormal, block_test


def _std(v):
    return (v - v.mean()) / (v.std() + 1e-12)


class _SplineBlockCIT(CIT_Base):
    """The shrink's test as a CI test: X _||_ Y | Z by the partial F of X's spline
    block for Y given the spline blocks of Z (Huber-clipped residual,
    :func:`~causalts.grace.shrink.block_test`), in both directions, Cauchy-combined
    (a direction whose test is undefined is left out).

    CDNOTS tests unordered pairs, so neither side is the effect. Reads ``self.data``
    at call time, since CDNOTS replaces it with its lag embedding. The last
    ``n_c_cols`` columns of each lag slice (the C basis, located through
    :meth:`set_lag_structure`) enter linearly, as in the shrink. Its p-values
    share the shrink's approximations (see
    :func:`~causalts.grace.shrink.shrink_parents`).
    """

    handles_discrete = True  # discrete columns get level-indicator blocks

    def __init__(self, data, n_c_cols=0, **kwargs):
        super().__init__(data, method="reap_spline", **kwargs)
        self.n_c_cols = n_c_cols
        self._n_vars = None  # columns per lag slice, from set_lag_structure
        self._cached, self._bases = None, {}

    def set_lag_structure(self, n_vars, lags):
        self._n_vars = n_vars

    def _block(self, D, c, nuisance=False):
        """Feature block of column ``c`` of ``D``, cached until ``D`` changes.

        A conditioning (``nuisance``) block of a discrete column gets one
        indicator per level, unmerged, so every level's effect is removed; a
        tested block merges rare levels (:func:`spline_basis`), as in the grow.
        """
        if self._cached is not D:  # holding D keeps its identity from being reused
            self._cached, self._bases = D, {}
        key = (c, nuisance)
        if key not in self._bases:
            v = _std(D[:, c])
            is_c = self._n_vars is not None and self.n_c_cols > 0
            is_c = is_c and c % self._n_vars >= self._n_vars - self.n_c_cols
            if is_c:
                self._bases[key] = v[:, None]
            elif nuisance:
                self._bases[key] = _nuisance_block(v, "spline", "spline")
            else:
                self._bases[key] = spline_basis(v)
        return self._bases[key]

    def __call__(self, X, Y, condition_set=None):
        D = np.asarray(self.data, dtype=float)  # the same object for float data
        x, y = int(np.atleast_1d(X)[0]), int(np.atleast_1d(Y)[0])
        z = [
            int(c)
            for c in np.atleast_1d([] if condition_set is None else condition_set)
        ]
        Q = _orthonormal(D.shape[0], [self._block(D, c, nuisance=True) for c in z])
        p1 = block_test(_std(D[:, y]), self._block(D, x), Q)[0]
        p2 = block_test(_std(D[:, x]), self._block(D, y), Q)[0]
        # Exactly 1.0 is block_test's "undefined" result (e.g. a variable with no
        # spline columns); combined, it would swamp the other direction.
        ps = [q for q in (p1, p2) if q < 1.0]
        p = _cauchy(ps) if ps else 1.0
        return p, -np.log10(max(p, 1e-300))


class _PinnedLagKnowledge(BackgroundKnowledge):
    """Background knowledge with set lookups over CDNOTS node names (``X{idx}``).

    causal-learn's lookups scan every rule (with a regex per pattern rule), which
    is too slow for the d^2 * max_lag rules that pin a whole lagged graph. CDNOTS
    only calls :meth:`is_forbidden` and :meth:`is_required`.
    """

    def __init__(self, forbidden, required):
        super().__init__()
        self._forbidden = forbidden
        self._required = required

    def is_forbidden(self, node1, node2):
        return (node1.get_name(), node2.get_name()) in self._forbidden

    def is_required(self, node1, node2):
        return (node1.get_name(), node2.get_name()) in self._required


def pinned_lag_knowledge(lagged, max_lag, n_c_cols):
    """Knowledge that fixes every lagged edge to ``lagged`` and leaves lag 0 open.

    ``lagged[c, e, lag]`` (lag >= 1) becomes a required ``c(t-lag) -> e(t)``;
    every other cross-time pair of the d variables is forbidden in both
    directions. Shifted copies (``c(t-lag-s) -> e(t-s)``) get the same rule.
    Pairs involving C columns are left open.
    """
    d = lagged.shape[0]
    per_lag = d + n_c_cols
    forbidden, required = set(), set()
    for lag in range(1, max_lag + 1):
        for s in range(max_lag - lag + 1):
            for c in range(d):
                for e in range(d):
                    src = f"X{c + (lag + s) * per_lag}"
                    dst = f"X{e + s * per_lag}"
                    if lagged[c, e, lag]:
                        required.add((src, dst))
                    else:
                        forbidden.add((src, dst))
                        forbidden.add((dst, src))
    return _PinnedLagKnowledge(forbidden, required)


def check_ci_test(ci_test):
    """Raise ``ValueError`` if ``ci_test`` is a name that is neither ``"spline"``
    nor a registered CI test, and ``TypeError`` if it is neither a name nor a
    callable CI-test instance (e.g. ``None``, or a CI-test class)."""
    from ..ci_tests import list_ci_tests

    names = ["spline", *list_ci_tests()]
    if isinstance(ci_test, str):
        if ci_test not in names:
            raise ValueError(
                f"Unknown contemporaneous_test {ci_test!r}. Available: {names}"
            )
    elif isinstance(ci_test, type) or not callable(ci_test):
        raise TypeError(
            "contemporaneous_test must be a test name or a CI test instance "
            f"(not a class), got {ci_test!r}"
        )


def contemporaneous_edges(
    df,
    lagged,
    max_lag,
    *,
    ci_test,
    alpha,
    include_C,
    c_preset,
    discrete_cols,
    show_progress=False,
):
    """Run CDNOTS+ on ``df`` with lagged edges pinned to ``lagged``.

    Returns the :class:`~causalts.cdnots.result.CdnotsResult`; its lag-0 slice
    holds the contemporaneous adjacencies and orientations.
    """
    from ..cdnots.phase3_utils import (
        _detect_discrete_cols,
        make_c_array,
        run_cdnots_plus,
    )
    from ..ci_tests import create_ci_test

    check_ci_test(ci_test)
    n_c_cols = make_c_array(len(df), c_preset).shape[1] if include_C else 0
    if isinstance(ci_test, str):
        # CDNOTS+ replaces the data with its lag embedding before testing.
        data = np.ascontiguousarray(df.to_numpy(float))
        if ci_test == "spline":
            ci_test = _SplineBlockCIT(data, n_c_cols=n_c_cols)
        else:
            ci_test = create_ci_test(ci_test, data)
    if discrete_cols is None:
        # Detected here, so run_cdnots_plus does not warn when it wraps a
        # discrete-unaware test in StratifiedCIT.
        discrete_cols = _detect_discrete_cols(df, len(df))
    bk = pinned_lag_knowledge(np.asarray(lagged, dtype=bool), max_lag, n_c_cols)
    return run_cdnots_plus(
        df,
        ci_test,
        num_lags=max_lag,
        include_C=include_C,
        c_preset=c_preset,
        alpha=alpha,
        discrete_cols=discrete_cols,
        background_knowledge=bk,
        show_progress=show_progress,
    )
