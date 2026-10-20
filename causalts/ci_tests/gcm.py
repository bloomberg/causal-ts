# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Generalised Covariance Measure
# Based on: Shah, Peters, "The Hardness of Conditional Independence Testing and
# the Generalised Covariance Measure", Ann. Statist. 48(3):1514-1538, 2020
# (arXiv:1804.07203).
# Reference implementation: GeneralisedCovarianceMeasure R package.

"""Generalised covariance measure.

Regress ``X`` on ``Z`` and ``Y`` on ``Z``, then ask whether the residual product
has non-zero mean. Under conditional independence the product is mean-zero, and
the normalised sum is asymptotically standard normal -- provided the two
regressions are accurate enough that the *product* of their mean-squared errors
vanishes faster than ``1/n``.

That proviso is the whole method. Shah and Peters also prove that no test can be
valid against all continuous alternatives and still have power, so GCM's
assumption is not a weakness to be engineered away; it is the price of having
any power at all.

This is the cheapest member of the residual-score family and the baseline that
:mod:`causalts.ci_tests.wgcm` and everything downstream of it must beat.

This implementation departs from the published i.i.d. test in two ways: nuisance
regressions default to the additive cubic-spline ridge
:class:`~causalts.ci_tests._residual_score.SplineSelectRidge` rather than a
Gaussian process, and the
shipped defaults (``variance="hac"``, ``crossfit="blocked"``) add
serial-dependence robustness. ``variance="iid"``, ``crossfit="full"`` recovers
the original form.
"""

import numpy as np
from scipy import stats

from ._residual_score import ResidualScoreCIT, hac_se

__all__ = ["GCM"]


class GCM(ResidualScoreCIT):
    """Generalised covariance measure CI test.

    Parameters
    ----------
    data : np.ndarray, shape (T, n_features)
        Time-ordered observations.
    variance : {"hac", "iid"}
        ``"hac"`` uses a Newey--West long-run standard error, appropriate when
        the residual product is serially correlated. ``"iid"`` reproduces the
        published normalisation and will over-reject on autocorrelated data.
    crossfit : {"blocked", "random", "full"}
        See :class:`~causalts.ci_tests._residual_score.ResidualScoreCIT`.
        ``"full"`` matches the original prescription; ``"blocked"`` is the
        default here because the rows are a time series.

    Notes
    -----
    Tests a residual moment, not full conditional independence: an edge that
    changes only the conditional variance of ``Y`` leaves the residual product
    mean-zero and will be missed. Pair with a distributional test when that
    matters.

    Examples
    --------
    >>> import numpy as np
    >>> from causalts.ci_tests import create_ci_test
    >>> rng = np.random.default_rng(0)
    >>> z = rng.standard_normal(300)
    >>> data = np.column_stack([z + rng.standard_normal(300),
    ...                         z + rng.standard_normal(300), z])
    >>> ci = create_ci_test("gcm", data)
    >>> p, stat = ci(0, 1, [2])
    >>> 0.0 <= p <= 1.0
    True
    """

    inferential_target = "residual_moment"

    def __init__(self, data, variance="hac", **kwargs):
        kwargs.setdefault("method", "gcm")
        super().__init__(data=data, **kwargs)
        if variance not in ("hac", "iid"):
            raise ValueError(f"variance must be 'hac' or 'iid', got {variance!r}")
        self.variance = variance

    def _param_hash(self):
        return f"{super()._param_hash()}-v{self.variance}"

    def _statistic(self, ex, ey, Z, rng):
        q = ex * ey
        n = q.shape[0]
        if n < 3:
            return 1.0, 0.0

        mean_q = float(q.mean())
        if self.variance == "hac":
            se = hac_se(q)
        else:
            sd = float(q.std(ddof=1))
            se = sd / np.sqrt(n) if sd > 1e-12 else np.inf

        if not np.isfinite(se) or se <= 0:
            return 1.0, 0.0

        stat = mean_q / se
        p = 2.0 * stats.norm.sf(abs(stat))
        return p, stat
