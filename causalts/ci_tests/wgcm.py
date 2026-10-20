# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Weighted Generalised Covariance Measure
# Based on: Scheidegger, Hörrmann, Bühlmann, "The Weighted Generalised
# Covariance Measure", JMLR 23(273):1-68, 2022 (arXiv:2111.04361).
# Reference implementation: weightedGCM R package.

"""Weighted generalised covariance measure.

GCM averages the residual product over all rows, so it is blind to any
alternative where the conditional covariance changes sign across the
conditioning space: the positive and negative regions cancel and the mean is
near zero. WGCM multiplies the product by fixed functions of ``Z`` before
averaging and takes the maximum over that bank, recovering exactly those
alternatives.

The weights must be pre-specified. The maximum over the bank is calibrated by
multiplier bootstrap, so the search across weights is paid for rather than
ignored.

This implementation draws its weight functions from the fixed
:class:`causalts.ci_tests._residual_score.FeatureBank` rather than the paper's
weight construction, and nuisance fits default to the additive cubic-spline
ridge :class:`~causalts.ci_tests._residual_score.SplineSelectRidge` rather than a
Gaussian process. The
shipped defaults (``crossfit="full"``, ``calibration="iid"``) reproduce the
published i.i.d. test; ``crossfit="blocked"`` with ``calibration="dependent"``
adds serial-dependence robustness for a lag-embedded time series.
"""

from ._residual_score import (
    DEFAULT_WEIGHTS,
    FeatureBank,
    ResidualScoreCIT,
    multiplier_bootstrap,
)

__all__ = ["WGCM"]


class WGCM(ResidualScoreCIT):
    """Weighted generalised covariance measure CI test.

    Parameters
    ----------
    data : np.ndarray, shape (T, n_features)
        Time-ordered observations.
    weights : tuple of str
        Weight families; see
        :class:`~causalts.ci_tests._residual_score.FeatureBank`.
    calibration : {"iid", "dependent"}
        Multiplier bootstrap flavour. ``"iid"`` is the published choice and the
        default, valid for independent rows; ``"dependent"`` preserves serial
        correlation and is the correct choice for a lag-embedded time series.
    n_boot : int
        Bootstrap replicates.
    max_dirs, max_weights : int
        Bank size controls. Larger banks widen the joint null, which costs
        power -- the wrong trade when ``T`` is small.

    Notes
    -----
    Like GCM this targets a residual moment, not full conditional independence.
    A wider bank covers more alternatives but never makes the test omnibus.

    Examples
    --------
    >>> import numpy as np
    >>> from causalts.ci_tests import create_ci_test
    >>> rng = np.random.default_rng(0)
    >>> z = rng.standard_normal(200)
    >>> # conditional covariance flips sign with z, so plain GCM is blind
    >>> e = rng.standard_normal(200)
    >>> data = np.column_stack([e, np.sign(z) * e + rng.standard_normal(200), z])
    >>> ci = create_ci_test("wgcm", data, n_boot=199)
    >>> p, stat = ci(0, 1, [2])
    >>> 0.0 <= p <= 1.0
    True
    """

    inferential_target = "residual_moment"

    def __init__(
        self,
        data,
        weights=DEFAULT_WEIGHTS,
        calibration="iid",
        n_boot=499,
        max_dirs=2,
        max_weights=8,
        block_length=None,
        **kwargs,
    ):
        kwargs.setdefault("method", "wgcm")
        # Published defaults: full-sample nuisance fits and i.i.d. multipliers.
        kwargs.setdefault("crossfit", "full")
        super().__init__(data=data, **kwargs)
        if calibration not in ("dependent", "iid"):
            raise ValueError(
                f"calibration must be 'dependent' or 'iid', got {calibration!r}"
            )
        self.calibration = calibration
        self.n_boot = int(n_boot)
        self.block_length = block_length
        self.bank = FeatureBank(
            weights=weights, max_dirs=max_dirs, max_weights=max_weights
        )

    def _param_hash(self):
        return (
            f"{super()._param_hash()}-cal{self.calibration}"
            f"-b{self.n_boot}-w{'+'.join(self.bank.weights)}"
        )

    def _cache_hash(self):
        return (
            f"{super()._cache_hash()}-md{self.bank.max_dirs}"
            f"-mw{self.bank.max_weights}-bl{self.block_length}"
        )

    def _statistic(self, ex, ey, Z, rng):
        product = ex * ey
        W = self.bank.build(Z, n=len(product))
        Q = W * product[:, None]
        return multiplier_bootstrap(
            Q,
            n_boot=self.n_boot,
            kind=self.calibration,
            block_length=self.block_length,
            rng=rng,
        )
