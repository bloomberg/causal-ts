# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Projected Covariance Measure
# Based on: Lundborg, Kim, Shah, Samworth, "The Projected Covariance Measure for
# Assumption-Lean Variable Significance Testing", Ann. Statist. 52(6):2851-2878,
# 2024 (arXiv:2211.02039).
# Reference implementation: comets R package.

"""Projected covariance measure.

PCM asks a strictly easier question than GCM: does ``X`` contribute to the
*conditional mean* of ``Y`` beyond ``Z``? It learns a projection
``f(X, Z) ~ E[Y | X, Z]`` on one part of the sample, then on held-out rows tests
whether the residual of that projection covaries with the residual of ``Y``.
Because the direction of the alternative is learned rather than guessed, PCM
does not need a weight bank, and the spline version is minimax-optimal for the
conditional-mean problem.

The easier question is the point and also the catch. An edge that changes only
the conditional variance, the tail, or any other distributional feature of ``Y``
leaves its conditional mean untouched, and PCM will not see it. It is reported
here as ``inferential_target = "conditional_mean"`` so that a caller can tell it
apart from a full-CI test without reading the source.

In this implementation the projection ``f(X, Z)`` and both regressions on ``Z``
default to :class:`~causalts.ci_tests._residual_score.SplineSelectRidge`, an
additive cubic-spline ridge, rather than a Gaussian process, for speed and
stability at small ``T``.
"""

from collections.abc import Iterable
from hashlib import sha1

import numpy as np
from scipy import stats

from ._residual_score import (
    ResidualScoreCIT,
    blocked_folds,
    hac_se,
    make_regressor,
)

__all__ = ["PCM"]


class PCM(ResidualScoreCIT):
    """Projected covariance measure CI test.

    Parameters
    ----------
    data : np.ndarray, shape (T, n_features)
        Time-ordered observations.
    projection : str
        Nuisance learner used for ``E[Y | X, Z]``; see
        :data:`~causalts.ci_tests._residual_score.NUISANCE_LEARNERS`. The
        default ``"spline_select"`` is an additive cubic-spline ridge with a
        leave-one-out penalty (falling back to linear ridge when that fits
        better), the flexible regressor the paper assumes; ``"tree"`` is more
        robust to non-smooth mean functions.
    variance : {"hac", "iid"}
        Long-run or i.i.d. standard error for the score.
    crossfit : {"blocked"}
        Only contiguous embargoed folds; other modes raise ``ValueError``.
    one_sided : bool
        The published test is one-sided: the projection is constructed so the
        score is positive under the alternative, and a two-sided test throws
        away half the power for nothing.

    Notes
    -----
    Tests conditional-mean independence, **not** full conditional independence.

    Examples
    --------
    >>> import numpy as np
    >>> from causalts.ci_tests import create_ci_test
    >>> rng = np.random.default_rng(0)
    >>> z = rng.standard_normal(300)
    >>> x = z + rng.standard_normal(300)
    >>> y = z + rng.standard_normal(300)
    >>> ci = create_ci_test("pcm", np.column_stack([x, y, z]))
    >>> p, stat = ci(0, 1, [2])
    >>> 0.0 <= p <= 1.0
    True
    """

    inferential_target = "conditional_mean"

    def __init__(
        self,
        data,
        projection="spline_select",
        variance="hac",
        one_sided=True,
        **kwargs,
    ):
        kwargs.setdefault("method", "pcm")
        if kwargs.get("crossfit", "blocked") != "blocked":
            raise ValueError(
                "PCM supports only crossfit='blocked': the projection must be "
                f"learned out of fold, got {kwargs['crossfit']!r}"
            )
        super().__init__(data=data, **kwargs)
        make_regressor(projection)  # unknown name: fail here, not as p=1 per test
        if variance not in ("hac", "iid"):
            raise ValueError(f"variance must be 'hac' or 'iid', got {variance!r}")
        self.projection = projection
        self.variance = variance
        self.one_sided = bool(one_sided)

    def _param_hash(self):
        return (
            f"{super()._param_hash()}-proj{self.projection}"
            f"-v{self.variance}-os{int(self.one_sided)}"
        )

    def _score_series(self, x, y, Z):
        """Cross-fitted ``resid(f(X,Z) | Z) * resid(Y | Z)`` and a scored-row mask.

        Rows whose fold is skipped or fails are left out of the mask rather than
        scored as zero, which would pull the statistic towards the null.
        """
        n = len(x)
        XZ = np.column_stack([x, Z]) if Z.shape[1] > 0 else x.reshape(-1, 1)
        q = np.zeros(n, dtype=float)
        scored = np.zeros(n, dtype=bool)

        n_splits = min(self.n_splits, max(2, n // 10))
        gap = self._fitter.effective_gap

        for train_idx, eval_idx in blocked_folds(n, n_splits=n_splits, gap=gap):
            if len(train_idx) < 6 or len(eval_idx) == 0:
                continue
            try:
                # Projection f(X, Z) ~ E[Y | X, Z], learned on training rows.
                f_model = make_regressor(self.projection)
                f_model.fit(XZ[train_idx], y[train_idx])
                f_all = np.asarray(f_model.predict(XZ)).ravel()

                if Z.shape[1] > 0:
                    # Both f(X,Z) and Y are then residualised on Z alone, using
                    # models that never saw the evaluation rows. A multi-output
                    # learner fits both on one shared design (identical result,
                    # half the work).
                    mz = make_regressor(self.regressors[0])
                    if getattr(mz, "multi_output", False):
                        both = np.column_stack([f_all[train_idx], y[train_idx]])
                        pred = np.asarray(
                            mz.fit(Z[train_idx], both).predict(Z[eval_idx])
                        )
                        f_hat, y_hat = pred[:, 0], pred[:, 1]
                    else:
                        mz.fit(Z[train_idx], f_all[train_idx])
                        f_hat = np.asarray(mz.predict(Z[eval_idx])).ravel()

                        my = make_regressor(self.regressors[0])
                        my.fit(Z[train_idx], y[train_idx])
                        y_hat = np.asarray(my.predict(Z[eval_idx])).ravel()
                else:
                    f_hat = np.full(len(eval_idx), f_all[train_idx].mean())
                    y_hat = np.full(len(eval_idx), y[train_idx].mean())

                q[eval_idx] = (f_all[eval_idx] - f_hat) * (y[eval_idx] - y_hat)
                scored[eval_idx] = True
            except Exception:
                continue
        return q, scored

    def __call__(self, X, Y, condition_set=None):
        self.n_tests += 1
        if condition_set is None:
            condition_set = []
        # Same order the cache key uses, so one cache entry has one answer.
        condition_set = sorted(condition_set)

        # The base hash is symmetric in X and Y, but PCM's statistic is not:
        # it projects E[Y | X, Z]. Tag the key with the X side so the two
        # orientations get separate cache entries.
        x_cols = sorted(X) if isinstance(X, Iterable) else int(X)
        x_tag = sha1(np.ascontiguousarray(self.data[:, x_cols])).hexdigest()
        cache_key = f"{self._get_array_hash(X, Y, condition_set)}_x{x_tag}"
        param_hash = self._cache_hash()
        cached = self.pvalue_cache.get((cache_key, param_hash))
        if cached is not None:
            return cached[0], cached[1]

        prepared = self._prepare(X, Y, condition_set)
        if prepared is None:
            self.pvalue_cache[(cache_key, param_hash)] = [1.0, 0.0]
            return 1.0, 0.0
        x, y, Z = prepared

        self.n_actual_tests += 1
        q, scored = self._score_series(x, y, Z)
        q = q[scored]
        n = len(q)

        if n < 2:
            se = np.inf
        elif self.variance == "hac":
            se = hac_se(q)
        else:
            sd = float(q.std(ddof=1))
            se = sd / np.sqrt(n) if sd > 1e-12 else np.inf

        if not np.isfinite(se) or se <= 0:
            p, stat = 1.0, 0.0
        else:
            stat = float(q.mean() / se)
            if self.one_sided:
                p = float(stats.norm.sf(stat))
            else:
                p = float(2.0 * stats.norm.sf(abs(stat)))

        p = float(min(max(p, 0.0), 1.0))
        self.pvalue_cache[(cache_key, param_hash)] = [p, stat]
        self.save_incremental_cache(cache_key, param_hash, p, stat)
        return p, stat
