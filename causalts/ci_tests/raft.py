# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Random-feature F-tests for conditional independence (RAFT / RIFT).
# A series-regression significance test (nested least squares, partial F) on a
# random-Fourier-feature basis. Related: residual nR^2 tests against random
# nonlinear features (Lee, White, Granger 1993, J. Econometrics 56:269-290) and
# random Fourier features (Rahimi, Recht 2007).

"""Random-feature F-tests: ``raft`` (additive) and ``rift`` (adaptive adjustment).

Each tested variable is expanded on its own into ``[x, cos(w_1 x), sin(w_1 x),
..., cos(w_k x), sin(w_k x)]`` (standardized ``x``, fixed random frequencies
``w``; an RBF-kernel approximation whose span does not change when ``x`` changes
sign). ``X _||_ Y | Z`` is tested by asking whether ``X``'s features
improve a least-squares fit of ``Y`` on ``Z`` (nested regression, partial F
statistic), and symmetrically whether ``Y``'s features improve ``X`` on ``Z``.
The p-values (one per response component; two for scalar ``X`` and ``Y``) are
combined (by default the Cauchy combination, which falls back to Bonferroni over
every direction when one is left untested), so the test (and its cache key) is
symmetric in ``X`` and ``Y``. By default the p-value comes from a dependent wild
bootstrap of the statistic under the null, and both tests also use a
Huber-clipped residual, fewer random features for the tested variables at small
T, leverage-rescaled bootstrap residuals and, up to 512 rows, a linear-only
channel (``robust``, ``n_rff_tested``, ``combine``, ``hc2``, ``linear_channel``).

The two tests differ only in how ``Z`` is adjusted for:

* ``raft`` (Random-feature Additive F-Test, ``adjust="additive"``): additive in
  ``Z``. By default (``z_ridge``) each conditioning variable's linear term,
  unpenalized, plus a dictionary of random sine/cosine features per variable
  whose ridge penalty is chosen by generalized cross-validation for each
  response, so the adjustment grows with the data without spending a degree of
  freedom per column. With ``z_ridge=False``, one random feature block per
  conditioning variable fitted by OLS, which underfits smooth effects of ``Z``
  once T is large. Effects of ``Z`` much wigglier than the features reach (e.g.
  ``sin(6 z)`` on the standardized scale) are not absorbed, and neither are
  interactions among conditioning variables; the test then rejects true
  conditional independence.
* ``rift`` (Random-feature Interaction-adjusted F-Test,
  ``adjust="interaction"``): ``raft``'s adjustment with more terms in the same
  penalized dictionary: per conditioning variable, sines and cosines on a fixed
  grid of frequencies (``freqs``); with two or more, also cubic B-splines, their
  pairwise tensor products and a few triple products. With smooth data the
  extra terms contribute almost nothing. It absorbs smooth interactions among
  conditioning variables (e.g. ``z1 * z2``) and wiggly marginal effects that
  make ``raft`` reject true conditional independence; rough interactions, and
  effects beyond the highest frequency, are only partly absorbed. A discovery
  run takes about twice as long as with ``raft``.

Both target the **conditional mean**. Neither has power against dependence that
lives only in an interaction of ``X`` with ``Z`` that has no main effect, or only
in the conditional variance; use ``kci``/``rcot`` there.
"""

import hashlib
import warnings
from collections import OrderedDict
from itertools import combinations

import numpy as np
from scipy import special, stats
from scipy.interpolate import BSpline
from scipy.linalg import qr
from scipy.linalg import svd as _scipy_svd

from .cit_test import CIT_Base, stable_seed

__all__ = ["RAFTCI", "RIFTCI"]

_LAM_GRID = np.logspace(-6, 2, 17)


def _standardize_columns(A):
    """Centre and scale columns; drop numerically constant ones."""
    A = np.asarray(A, dtype=np.float64)
    if A.ndim == 1:
        A = A[:, None]
    if A.shape[1] == 0:
        return np.zeros((A.shape[0], 0))
    sd = A.std(axis=0)
    keep = sd > 1e-10
    if not np.any(keep):
        return np.zeros((A.shape[0], 0))
    return (A[:, keep] - A[:, keep].mean(axis=0)) / sd[keep]


def _cauchy(ps):
    """Cauchy combination (Liu and Xie 2020) of p-values."""
    q = np.clip(np.asarray(ps, dtype=float), 1e-15, 1.0 - 1e-15)
    return float(0.5 - np.arctan(np.mean(np.tan((0.5 - q) * np.pi))) / np.pi)


def _huber(r, c=1.345):
    """Residuals clipped at ``c`` robust standard deviations (MAD scale); returned
    unchanged when the MAD is zero."""
    s = np.median(np.abs(r - np.median(r))) / 0.6745
    if not s > 0:
        return r
    return np.clip(r, -c * s, c * s)


def _is_constant(x):
    """True when a column takes a single value. Any represented variation counts,
    however small or far out on an offset."""
    return bool(np.ptp(np.asarray(x, dtype=np.float64)) == 0.0)


def _rounding_level(x):
    """One ulp of a column's largest value, in standard deviations: how finely the
    column's variation is represented (large when an offset eats the precision)."""
    return np.finfo(float).eps * float(np.max(np.abs(x))) / float(x.std())


def _explain_tol(level):
    """Relative residual size below which Z counts as explaining a column: four
    rounding levels (an exact affine copy leaves about 0.3), never below 1e3 eps.
    When the column is too coarse for that to be small (an offset has eaten its
    precision), fall back to the floor rather than call real variation rounding."""
    floor = 1e3 * np.finfo(float).eps
    return 4.0 * level if floor < 4.0 * level <= 1e-3 else floor


def _canonical(x):
    """A column standardized and oriented by the sign of its first clearly nonzero
    value, with its rounding level (one ulp of its largest value, in standard
    deviations): affine copies of one variable share this form up to that level."""
    sd = float(x.std())
    zs = (x - x.mean()) / sd
    level = _rounding_level(x)
    # The first value with |z| > 0.5 (away from that cutoff) is one that affine
    # copies agree on up to rounding; ties among the extremes, common in discrete
    # columns, are not.
    if zs[int(np.argmax(np.abs(zs) > 0.5 + 1e-6))] < 0:
        zs = -zs
    return zs, level


def _same_variable(a, b):
    """True when two canonical columns are affine copies, up to their rounding.

    Used only to deduplicate the interaction dictionary, where a wrong merge just
    drops a few dictionary terms. Only resolvable when both rounding levels are
    tiny; otherwise columns are treated as distinct."""
    (za, ea), (zb, eb) = a, b
    if np.array_equal(za, zb) or np.array_equal(za, -zb):
        return True  # exact copies, however coarse their rounding
    tol = 16.0 * (ea + eb)
    if tol > 1e-5:
        return False
    rms = min(np.sqrt(np.mean((za - zb) ** 2)), np.sqrt(np.mean((za + zb) ** 2)))
    return float(rms) <= tol


def _spline_basis(z, n_knots, degree=3):
    """Cubic B-spline basis on quantile knots, last spline dropped (the same basis
    as scikit-learn's ``SplineTransformer(knots="quantile", include_bias=False)``
    on in-sample data)."""
    base = np.percentile(z, np.linspace(0, 100, n_knots))
    d0, d1 = base[1] - base[0], base[-1] - base[-2]
    t = np.r_[
        base[0] - d0 * np.arange(degree, 0, -1),
        base,
        base[-1] + d1 * np.arange(1, degree + 1),
    ]
    return BSpline.design_matrix(z, t, degree, extrapolate=False).toarray()[:, :-1]


def _svd(A):
    """Thin SVD. LAPACK's default divide-and-conquer driver (gesdd) can fail to
    converge on some BLAS builds for ill-conditioned input; retry with the slower
    QR-iteration driver (gesvd), which is robust. Converging cases are unchanged."""
    try:
        return np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        return _scipy_svd(A, full_matrices=False, lapack_driver="gesvd")


def _left_singular(J, rtol=1e-6):
    """Left singular vectors and values of J above ``rtol * s_max``, via QR and an
    SVD of the small triangular factor. The dictionary is ill-conditioned, and the
    cheaper Gram-matrix route squares its condition number (U then drifts from
    orthonormal by up to ~1e-5)."""
    Q, R = qr(J, mode="economic", check_finite=False)
    Ur, s, _ = _svd(R)
    if not s.size or s[0] <= 1e-12:
        return np.zeros((J.shape[0], 0)), np.zeros(0)
    keep = s > s[0] * rtol
    return Q @ Ur[:, keep], s[keep]


def _content_key(A):
    """Hashable key for an array's shape, dtype and bytes."""
    A = np.ascontiguousarray(A)
    return A.shape, A.dtype.str, hashlib.sha1(A.tobytes()).hexdigest()


class _LRU:
    """Least-recently-used cache bounded by entry count and total bytes. The latest
    entry is always kept, even when it alone exceeds the byte bound (a discovery
    run reuses one large item, e.g. the multipliers of its single sample size)."""

    def __init__(self, max_items, max_bytes):
        self.max_items, self.max_bytes = max_items, max_bytes
        self._d = OrderedDict()
        self._bytes = 0

    def get(self, key):
        """``(True, value)`` on a hit (refreshing its recency), else ``(False, None)``."""
        if key not in self._d:
            return False, None
        self._d.move_to_end(key)
        return True, self._d[key][0]

    def put(self, key, value, nbytes):
        if key in self._d:
            return
        while self._d and (
            len(self._d) >= self.max_items or self._bytes + nbytes > self.max_bytes
        ):
            self._bytes -= self._d.popitem(last=False)[1][1]
        self._d[key] = (value, nbytes)
        self._bytes += nbytes


class RAFTCI(CIT_Base):
    """Random-feature F-test of conditional independence (``raft`` / ``rift``).

    Parameters
    ----------
    data : np.ndarray, shape (T, n_features)
        Observations; rows in time order when ``variance="dwb"``.
    adjust : {"additive", "interaction"}
        How ``Z`` is adjusted for: additive random-feature blocks (``raft``,
        default), or additionally a ridge-penalized dictionary that absorbs
        smooth interactions among conditioning variables and wiggly marginal
        effects (``rift``). See the module docstring.
    n_rff : int
        Random Fourier features per variable, as ``n_rff / 2`` sine/cosine
        pairs (so it must be even); the raw standardized value is always
        included as well.
    bandwidth : float
        RBF bandwidth on the standardized scale (``w ~ N(0, 1/bandwidth^2)``).
    variance : {"dwb", "f"}
        ``"dwb"`` (default): dependent wild bootstrap of the partial F statistic
        under the null (block-correlated Gaussian multipliers); holds its level
        under heteroscedastic, heavy-tailed and serially correlated errors.
        ``"f"``: analytic partial F p-value (additive adjustment only); about
        twice as fast, but exact only for i.i.d. homoscedastic Gaussian errors
        and badly liberal when the error scale depends on ``Z``.
    rff_seed : int, optional
        Seed for the feature draws, the tensor sketch, the triple products and
        the bootstrap multipliers (fixed so results are reproducible). When not
        given, the usual ``seed`` keyword is used, else 12345.
    n_boot : int
        Bootstrap draws for ``"dwb"``. They estimate the null's scale and shape
        (moment-matched scaled chi-square), so p-values are not floored at
        ``1 / (n_boot + 1)``.
    n_knots : int
        Quantile knots per conditioning variable in the interaction dictionary.
    max_pair : int
        Above this many pairwise tensor columns (large conditioning sets), the
        pairwise block is compressed by a fixed Gaussian sketch to this size.
    n_triple : int
        Random triple products of spline columns added to the dictionary.
    w_tol : float
        Ridge directions kept with weight below this are dropped (they change
        nothing but cost time).
    freqs : sequence of float
        Frequencies (on the standardized scale) of the sine/cosine terms each
        conditioning variable contributes to the ``rift`` dictionary; empty
        disables them. The default is the unit-spaced frequencies 1-8.
    robust : {None, False, "huber"}
        ``"huber"``: test the Huber-clipped residual of each response given ``Z``
        (clipped at 1.345 robust standard deviations), so heavy tails and
        volatility bursts do not dominate the statistic. Needs
        ``variance="dwb"``. With skewed noise whose scale depends
        on the conditioning variables non-additively (e.g. on ``z1 * z2``), the
        clipped residual's conditional mean is no longer additive in ``Z`` and the
        test turns liberal (rejection about 0.17 at alpha=0.05 for T=1000 in
        pre-release runs, against 0.06 with ``robust=False``). Separately, when
        one conditioning variable drives a skewed noise's scale extremely
        (exp(z^2 / 2), infinite variance) the additive adjustment cannot track
        the shift and rejection reached 0.65 (``robust=False`` 0.26; prefer
        ``rcot`` there). ``None`` (default): ``"huber"`` with the bootstrap
        null, else off.
    n_rff_tested : {None, "auto"} or int
        Random features per tested variable (the conditioning variables keep
        ``n_rff``). ``"auto"``: one sine/cosine pair per about 100 rows, between
        one and four pairs, which keeps power at small T without weakening the
        adjustment for ``Z``. ``None`` (default): ``"auto"`` with the bootstrap
        null, else ``n_rff``.
    combine : {None, "bonferroni", "cauchy"}
        How the per-direction p-values are combined: Bonferroni, or the Cauchy
        combination (Liu and Xie 2020), which stays valid under dependence between
        the directions and loses less when several carry the signal. Only
        directions actually tested enter it (on the ridge path, any untested
        direction sends it back to Bonferroni over every direction). ``None``
        (default): ``"cauchy"`` with the bootstrap null, else ``"bonferroni"``.
        The original behaviour of both methods is ``robust=False,
        n_rff_tested=n_rff, combine="bonferroni", z_ridge=False, hc2=False,
        linear_channel=False``.
    z_ridge : {None, True, False}
        Adjust for ``Z`` by its linear terms (unpenalized) plus a ridge-penalized
        dictionary of ``z_ridge_rff`` random sine/cosine features per distinct
        conditioning variable (two thirds at ``bandwidth``, one third at half of
        it), the penalty chosen by generalized cross-validation for each response.
        A fixed feature basis underfits smooth effects of ``Z`` once T is large, and
        its misfit then reads as dependence; the penalized dictionary grows with the
        data and costs few degrees of freedom at small T. With ``rift`` its
        frequency and spline-interaction terms join the same dictionary. Needs
        ``variance="dwb"``. ``None`` (default): on with the bootstrap null, else
        off.
    z_ridge_rff : int
        Size of the ``z_ridge`` dictionary per conditioning variable (even).
    z_ridge_under : float
        Scale on the GCV penalty, in (0, 1]; below 1 undersmooths.
    hc2 : {None, True, False}
        Rescale the bootstrap residuals by ``1 / sqrt(1 - h)``, ``h`` the leverage
        of the ``Z`` adjustment (the exact diagonal of the ridge smoother with
        ``z_ridge``), so rows the adjustment fits closely still carry their share of
        noise. Changes only the null, not the statistic. Needs ``variance="dwb"``.
        ``None`` (default): on with the bootstrap null, else off.
    linear_channel : {None, False, True} or int
        Also test each direction with only the tested variables' linear terms and
        Cauchy-combine the two p-values (either one alone when the other is
        skipped): a nearly linear dependence is then tested on one degree of freedom
        per variable, which recovers power at small T. ``True``: always; an int: at
        most that many usable rows. ``None`` (default): 512 with the bootstrap
        null, else off.
    cache_mb : float
        Memory budget, in MB, for this instance's caches (features, conditioning-set
        fits, response fits, bootstrap multipliers). Results do not depend on it;
        a smaller budget only recomputes more.

    Notes
    -----
    Tests conditional-mean dependence; see the module docstring for what each
    adjustment can and cannot detect.

    Examples
    --------
    >>> import numpy as np
    >>> from causalts.ci_tests import create_ci_test
    >>> rng = np.random.default_rng(0)
    >>> z = rng.standard_normal(500)
    >>> data = np.column_stack([np.sin(z) + 0.3 * rng.standard_normal(500),
    ...                         np.cos(z) + 0.3 * rng.standard_normal(500), z])
    >>> p, stat = create_ci_test("raft", data)(0, 1, [2])
    >>> 0.0 <= p <= 1.0
    True
    """

    inferential_target = "conditional_mean"

    def __init__(
        self,
        data,
        adjust="additive",
        n_rff=8,
        bandwidth=1.0,
        variance="dwb",
        rff_seed=None,
        n_boot=199,
        n_knots=5,
        max_pair=176,
        n_triple=8,
        w_tol=1e-4,
        freqs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
        robust=None,
        n_rff_tested=None,
        combine=None,
        z_ridge=None,
        z_ridge_rff=24,
        z_ridge_under=1.0,
        hc2=None,
        linear_channel=None,
        cache_mb=256,
        **kwargs,
    ):
        if adjust not in ("additive", "interaction"):
            raise ValueError(
                f"adjust must be 'additive' or 'interaction', got {adjust!r}"
            )
        if variance not in ("f", "dwb"):
            raise ValueError(f"variance must be 'f' or 'dwb', got {variance!r}")
        if adjust == "interaction" and variance != "dwb":
            raise ValueError("adjust='interaction' requires variance='dwb'")
        if rff_seed is None:
            # The usual ``seed`` keyword sets the draws when ``rff_seed`` is not given.
            rff_seed = kwargs.get("seed")
            rff_seed = 12345 if rff_seed is None else rff_seed
        for name, v in (
            ("n_rff", n_rff),
            ("n_boot", n_boot),
            ("n_knots", n_knots),
            ("max_pair", max_pair),
            ("n_triple", n_triple),
            ("rff_seed", rff_seed),
        ):
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
                raise ValueError(f"{name} must be an integer, got {v!r}")
        if variance == "dwb" and int(n_boot) < 2:
            raise ValueError(f"n_boot must be at least 2, got {n_boot!r}")
        if int(n_rff) < 0 or int(n_rff) % 2:
            raise ValueError(f"n_rff must be a non-negative even number, got {n_rff!r}")
        freqs = tuple(float(f) for f in freqs)
        if not all(np.isfinite(f) and f > 0 for f in freqs):
            raise ValueError(f"freqs must be finite and > 0, got {freqs!r}")
        if not (np.isfinite(bandwidth) and bandwidth > 0):
            raise ValueError(f"bandwidth must be finite and > 0, got {bandwidth!r}")
        if int(n_knots) < 2:
            raise ValueError(f"n_knots must be at least 2, got {n_knots!r}")
        if int(max_pair) < 1:
            raise ValueError(f"max_pair must be at least 1, got {max_pair!r}")
        if int(n_triple) < 0:
            raise ValueError(f"n_triple must be non-negative, got {n_triple!r}")
        if not (np.isfinite(w_tol) and 0.0 <= w_tol < 1.0):
            raise ValueError(f"w_tol must be finite and in [0, 1), got {w_tol!r}")
        # None means the method's default: with the bootstrap null (raft and rift)
        # the Huber residual, fewer features for the tested variables, the Cauchy
        # combination, the ridge-penalized Z dictionary, HC2 and the linear channel;
        # with variance="f" none of these.
        default_on = variance == "dwb"
        if robust is None:
            robust = "huber" if default_on else False
        if n_rff_tested is None:
            n_rff_tested = "auto" if default_on else n_rff
        if combine is None:
            combine = "cauchy" if default_on else "bonferroni"
        if z_ridge is None:
            z_ridge = default_on
        if hc2 is None:
            hc2 = default_on
        if linear_channel is None:
            linear_channel = 512 if default_on else False
        if robust is not False and robust != "huber":
            raise ValueError(f"robust must be None, False or 'huber', got {robust!r}")
        if robust and variance != "dwb":
            raise ValueError("robust='huber' requires variance='dwb'")
        if n_rff_tested != "auto":
            if isinstance(n_rff_tested, bool) or not isinstance(
                n_rff_tested, (int, np.integer)
            ):
                raise ValueError(
                    f"n_rff_tested must be None, 'auto' or an integer, "
                    f"got {n_rff_tested!r}"
                )
            if not 0 <= int(n_rff_tested) <= int(n_rff) or int(n_rff_tested) % 2:
                raise ValueError(
                    f"n_rff_tested must be an even number in [0, n_rff], "
                    f"got {n_rff_tested!r}"
                )
            n_rff_tested = int(n_rff_tested)
        if combine not in ("bonferroni", "cauchy"):
            raise ValueError(
                f"combine must be 'bonferroni' or 'cauchy', got {combine!r}"
            )
        if not isinstance(z_ridge, (bool, np.bool_)):
            raise ValueError(f"z_ridge must be True or False, got {z_ridge!r}")
        if z_ridge and variance != "dwb":
            raise ValueError("z_ridge=True requires variance='dwb'")
        if (
            isinstance(z_ridge_rff, bool)
            or not isinstance(z_ridge_rff, (int, np.integer))
            or int(z_ridge_rff) < 2
            or int(z_ridge_rff) % 2
        ):
            raise ValueError(
                f"z_ridge_rff must be an even integer >= 2, got {z_ridge_rff!r}"
            )
        if not (np.isfinite(z_ridge_under) and 0.0 < z_ridge_under <= 1.0):
            raise ValueError(f"z_ridge_under must be in (0, 1], got {z_ridge_under!r}")
        if not isinstance(hc2, (bool, np.bool_)):
            raise ValueError(f"hc2 must be True or False, got {hc2!r}")
        if hc2 and variance != "dwb":
            raise ValueError("hc2=True requires variance='dwb'")
        if isinstance(cache_mb, bool) or not (
            isinstance(cache_mb, (int, float, np.integer, np.floating))
            and np.isfinite(cache_mb)
            and cache_mb > 0
        ):
            raise ValueError(f"cache_mb must be a positive number, got {cache_mb!r}")
        # linear_channel: False / True / an int row cutoff; checked by type, so 1.0
        # cannot pass
        if isinstance(linear_channel, (bool, np.bool_)):
            linear_channel = bool(linear_channel)
        elif isinstance(linear_channel, (int, np.integer)) and linear_channel >= 1:
            linear_channel = int(linear_channel)
        else:
            raise ValueError(
                "linear_channel must be None, False, True or a positive int row "
                f"cutoff, got {linear_channel!r}"
            )
        kwargs.setdefault("method", "raft" if adjust == "additive" else "rift")
        super().__init__(data=data, **kwargs)
        self.adjust = adjust
        # One memory budget (cache_mb, per instance) split over the caches; the
        # conditioning-set and response caches are the ones that fill. Results do
        # not depend on it, only the time spent recomputing evicted entries.
        mb = float(cache_mb) * 2**20
        self._xi_cache = _LRU(max_items=16, max_bytes=mb / 8)
        self._feat_cache = _LRU(max_items=4096, max_bytes=mb / 8)
        # Content-keyed, so they stay valid when an adapter swaps the data array.
        self._spl_cache = _LRU(max_items=1024, max_bytes=mb / 16)
        self._z_cache = _LRU(max_items=1024, max_bytes=mb / 4)
        # Per (response, Z) fit and bootstrap residuals, shared by every test with
        # that response and conditioning set; per (tested variable, Z) basis.
        self._resp_cache = _LRU(max_items=4096, max_bytes=mb / 4)
        self._basis_cache = _LRU(max_items=4096, max_bytes=mb / 16)
        self.n_boot = int(n_boot)
        self.n_rff = int(n_rff)
        self.bandwidth = float(bandwidth)
        self.variance = variance
        self.rff_seed = int(rff_seed)
        self.n_knots = int(n_knots)
        self.max_pair = int(max_pair)
        self.n_triple = int(n_triple)
        self.w_tol = float(w_tol)
        self.freqs = freqs
        self.robust = robust
        self.n_rff_tested = n_rff_tested
        self.combine = combine
        self.z_ridge = bool(z_ridge)
        self.z_ridge_rff = int(z_ridge_rff)
        self.z_ridge_under = float(z_ridge_under)
        self.hc2 = bool(hc2)
        self.linear_channel = linear_channel
        if self.z_ridge:
            # Wide conditioning dictionary: z_ridge_rff / 2 sine/cosine pairs, two
            # thirds at the bandwidth and one third at half of it (wigglier).
            g = np.random.default_rng(
                stable_seed("raft-zridge", self.rff_seed, self.z_ridge_rff)
            )
            k = self.z_ridge_rff // 2
            k1 = (2 * k) // 3
            self._wz = np.r_[
                g.standard_normal(k1) / self.bandwidth,
                2.0 * g.standard_normal(k - k1) / self.bandwidth,
            ]
        rng = np.random.default_rng(stable_seed("raft", self.rff_seed, self.n_rff))
        self._w = rng.standard_normal(self.n_rff // 2) / self.bandwidth
        self.n_saturated = 0
        self._sat_dirs = 0
        self._call_saturated = False
        self._warned = set()

    def get_data(self):
        return self._data

    def _assign_data(self, d):
        # Adapters (e.g. tigramite's CITestAdapter) assign a new array per test and
        # renumber its columns; every per-column cache is keyed by content, so it
        # stays valid (and is safe under in-place mutation). ``set_data`` (base
        # class) also clears the p-value caches.
        self._data = d
        if d is not None:
            self.sample_size, self.num_features = d.shape

    data = property(get_data, _assign_data)

    def _get_array_hash(self, X, Y, Z):
        # The base hash covers column contents only; add the row count and dtype so
        # equal bytes laid out in a different shape cannot share a key.
        return f"{super()._get_array_hash(X, Y, Z)}_{self._data.shape[0]}{self._data.dtype.str}"

    def _param_hash(self):
        # Floats in exact (repr) form. The method name leads because the base
        # class's disk-cache loader keeps only rows that contain it.
        h = f"{self.method}-m{self.n_rff}-bw{self.bandwidth!r}-v{self.variance}"
        h += f"-s{self.rff_seed}"
        if self.variance == "dwb":
            h += f"-b{self.n_boot}"
        if self.adjust == "interaction":
            h += f"-k{self.n_knots}-p{self.max_pair}-t{self.n_triple}-w{self.w_tol!r}"
            h += "-f" + ",".join(repr(f) for f in self.freqs)
        if self.robust:
            h += f"-r{self.robust}"
        if self.n_rff_tested != self.n_rff:
            h += f"-mt{self.n_rff_tested}"
        if self.combine != "bonferroni":
            h += f"-c{self.combine}"
        if self.z_ridge:
            h += f"-zr{self.z_ridge_rff}-u{self.z_ridge_under!r}-w{self.w_tol!r}"
        if self.hc2:
            h += "-hc2"
        if self.linear_channel:
            h += f"-lin{self.linear_channel}"
        return h + f"-ms{self._min_samples}-v4"

    # ----- feature blocks -------------------------------------------------------

    def _column_features(self, x, m=None):
        """Feature block of one column, shape (n, 1+m); None if the column is constant.
        Sine/cosine pairs (rather than phase-shifted cosines) keep the span the same
        when the column changes sign, so an affine copy of a conditioning variable
        adds nothing beyond it. ``m`` < ``n_rff`` uses the first ``m / 2`` of the
        same frequency draws."""
        if _is_constant(x):
            return None
        m = self.n_rff if m is None else m
        x = (x - x.mean()) / x.std()
        if not m:
            return x[:, None]
        a = np.outer(x, self._w[: m // 2])
        c = np.sqrt(2.0 / m)
        return np.column_stack([x, c * np.cos(a), c * np.sin(a)])

    def _tested_rff(self, n):
        """Random features per tested variable for an ``n``-row test:
        ``n_rff_tested``, where ``"auto"`` gives one sine/cosine pair per about 100
        rows, between 1 and 4 pairs (capped at ``n_rff``). The conditioning
        variables always keep ``n_rff``."""
        if self.n_rff_tested == "auto":
            return min(self.n_rff, 2 * int(np.clip(round(n / 100), 1, 4)))
        return self.n_rff_tested

    def _features(self, idx, data, m=None):
        """Additive feature block for data columns ``idx``. Column blocks are cached
        by content: discovery algorithms test the same columns many times."""
        m = self.n_rff if m is None else m
        blocks = []
        for j in idx:
            x = data[:, j]
            key = _content_key(x) if m == self.n_rff else (m, _content_key(x))
            hit, f = self._feat_cache.get(key)
            if not hit:
                f = self._column_features(x, m)
                self._feat_cache.put(key, f, 0 if f is None else f.nbytes)
            if f is not None:
                blocks.append(f)
        return np.hstack(blocks) if blocks else np.zeros((data.shape[0], 0))

    @staticmethod
    def _distinct_z(zi, data):
        """Canonical forms of the distinct conditioning variables, for the
        interaction dictionary: constant columns and affine copies of another
        column dropped (their spline spans coincide; copies would add spurious
        self-products). Candidates are ordered by their full canonical content
        before copies are dropped, so neither the column order nor which copy is
        listed first changes the dictionary or its random draws."""
        cand = [_canonical(data[:, j]) for j in zi if not _is_constant(data[:, j])]
        cand.sort(key=lambda c: (tuple(np.round(c[0], 8)), c[0].tobytes()))
        kept = []
        for c in cand:
            if not any(_same_variable(c, k) for k in kept):
                kept.append(c)
        return [c[0] for c in kept]

    def _z_basis(self, FZ):
        """Orthonormal basis of span(1, FZ) by rank-revealing (pivoted) QR."""
        n = FZ.shape[0]
        A = np.column_stack([np.ones(n), FZ])
        Q, R, _ = qr(A, mode="economic", pivoting=True, check_finite=False)
        d = np.abs(np.diag(R))
        r = int(np.sum(d > d[0] * 1e-8))
        # Copy a rank-deficient slice so a cached basis does not pin the full Q.
        return Q if r == Q.shape[1] else Q[:, :r].copy(order="K")

    def _spline_block(self, z):
        """Standardized marginal spline block of a (non-constant) column."""
        key = _content_key(z)
        hit, b = self._spl_cache.get(key)
        if not hit:
            b = _standardize_columns(_spline_basis(z, self.n_knots))
            self._spl_cache.put(key, b, b.nbytes)
        return b

    def _interaction_dictionary(self, cols):
        """Marginal splines of the columns ``cols``, all pairwise spline tensor
        products (sketched to ``max_pair`` columns when larger) and ``n_triple``
        random triple products."""
        n = cols[0].shape[0]
        blocks = [self._spline_block(c) for c in cols]
        blocks = [b for b in blocks if b.shape[1]]
        if len(blocks) < 2:
            return np.zeros((n, 0))
        main = np.hstack(blocks)
        pairs = list(combinations(range(len(blocks)), 2))
        sketch = sum(blocks[a].shape[1] * blocks[b].shape[1] for a, b in pairs)
        sketch = sketch > self.max_pair
        parts, acc = [], np.zeros((n, self.max_pair)) if sketch else None
        for a, b in pairs:
            P = (blocks[a][:, :, None] * blocks[b][:, None, :]).reshape(n, -1)
            P = _standardize_columns(P)
            if sketch:
                # Gaussian sketch drawn block by block (the same distribution as one
                # full draw), so the raw pair columns are never all held at once.
                rng = np.random.default_rng(
                    stable_seed(
                        "rift-pair-sketch",
                        self.rff_seed,
                        a,
                        b,
                        P.shape[1],
                        self.max_pair,
                    )
                )
                acc += P @ rng.standard_normal((P.shape[1], self.max_pair))
            else:
                parts.append(P)
        pair = _standardize_columns(acc) if sketch else np.hstack(parts)
        triple = np.zeros((n, 0))
        if len(blocks) >= 3 and self.n_triple > 0:
            rng = np.random.default_rng(
                stable_seed("rift-triple", self.rff_seed, len(blocks), self.n_triple)
            )
            cols = []
            for _ in range(self.n_triple):
                prod = np.ones(n)
                for k in rng.choice(len(blocks), size=3, replace=False):
                    prod = prod * blocks[k][:, rng.integers(blocks[k].shape[1])]
                cols.append(prod)
            triple = _standardize_columns(np.column_stack(cols))
        return np.hstack([main, pair, triple])

    def _z_parts(self, zi, data):
        """(Qa, U, s): the unpenalized basis, and the left singular vectors / values
        of the ridge-penalized dictionary after projecting off Qa: the ``z_ridge``
        dictionary (plus ``rift``'s terms), or with ``z_ridge=False`` ``rift``'s
        dictionary alone. U and s are empty for ``raft`` with ``z_ridge=False``,
        without informative conditioning variables, or when Qa already uses up the
        sample. Cached by the content of the Z columns."""
        key = _content_key(data[:, zi])
        hit, parts = self._z_cache.get(key)
        if hit:
            return parts
        n = data.shape[0]
        if self.z_ridge:
            parts = self._z_ridge_parts(zi, data)
            self._z_cache.put(key, parts, sum(a.nbytes for a in parts))
            return parts
        Qa = self._z_basis(self._features(zi, data))
        U, s = np.zeros((n, 0)), np.zeros(0)
        # The additive basis uses every conditioning column (pivoted QR drops exact
        # duplicates); the dictionary uses the distinct ones in canonical form.
        zd = self._distinct_z(zi, data) if self.adjust == "interaction" else []
        if zd and n - Qa.shape[1] >= 11:
            # Multi-scale sines/cosines per variable (wiggly marginal effects), plus
            # the spline interaction dictionary when there are two or more.
            parts = []
            if self.freqs:
                for c in zd:
                    a = np.outer(c, self.freqs)
                    parts.append(
                        _standardize_columns(np.hstack([np.cos(a), np.sin(a)]))
                    )
            if len(zd) >= 2:
                parts.append(self._interaction_dictionary(zd))
            J = np.hstack(parts) if parts else np.zeros((n, 0))
            J = J - Qa @ (Qa.T @ J)
            if J.shape[1]:
                U, s = _left_singular(J)
        parts = (Qa, U, s)
        self._z_cache.put(key, parts, Qa.nbytes + U.nbytes + s.nbytes)
        return parts

    def _z_ridge_parts(self, zi, data):
        """``z_ridge``: the conditioning variables' linear terms unpenalized (Qa),
        and their wide random-feature dictionary, projected off Qa, as (U, s) for
        the GCV ridge smoother. A fixed small basis underfits smooth effects once T
        is large (its misfit then reads as dependence); the penalized dictionary
        grows the adjustment with the data and costs few degrees of freedom at
        small T."""
        n = data.shape[0]
        lin = [
            ((x - x.mean()) / x.std())[:, None]
            for x in (data[:, j] for j in zi)
            if not _is_constant(x)
        ]
        # The dictionary uses the distinct variables in canonical form (as rift's),
        # so a copied or affinely duplicated conditioning column neither doubles its
        # dictionary nor shifts the GCV penalty grid (scaled by the top singular
        # value); the pivoted QR of Qa drops duplicate linear terms.
        dic = []
        zd = self._distinct_z(zi, data)
        for c in zd:
            a = np.outer(c, self._wz)
            dic.append(np.column_stack([np.cos(a), np.sin(a)]))
        if self.adjust == "interaction":
            # rift on the z_ridge base: its multi-scale sines/cosines and spline
            # interaction dictionary join the same penalized dictionary. Its blocks
            # are standardized while the wide features above are not (variance at
            # most 1/2), so one GCV penalty shrinks the wide features somewhat more
            # under rift than under raft; this is the weighting that was tuned.
            if self.freqs:
                for c in zd:
                    a = np.outer(c, self.freqs)
                    dic.append(_standardize_columns(np.hstack([np.cos(a), np.sin(a)])))
            if len(zd) >= 2:
                dic.append(self._interaction_dictionary(zd))
        Qa = self._z_basis(np.hstack(lin) if lin else np.zeros((n, 0)))
        U, s = np.zeros((n, 0)), np.zeros(0)
        if dic and n - Qa.shape[1] >= 11:
            J = np.hstack(dic)
            J = J - Qa @ (Qa.T @ J)
            U, s = _left_singular(J)
        return Qa, U, s

    # ----- inference ------------------------------------------------------------

    def _saturated(self, n):
        """Note a regression direction the sample cannot support (it is skipped)."""
        self._sat_dirs += 1

    def _report_saturation(self, n, full, rows=False):
        """Count a test the sample cannot (fully) support and warn once per kind:
        when every direction is skipped, or there are too few rows at all, the test
        returns p = 1, which a constraint-based search reads as independence."""
        self.n_saturated += 1
        self._call_saturated = True
        kind = "rows" if rows else "full" if full else "partial"
        if kind in self._warned:
            return
        self._warned.add(kind)
        if rows:
            msg = (
                f"{self.method}: only n={n} usable rows; returning p=1, which reads "
                "as independence."
            )
        else:
            what = (
                "returning p=1, which reads as independence"
                if full
                else "some regression directions were skipped; the p-value comes "
                "from the remaining ones"
            )
            msg = (
                f"{self.method}: the conditioning features use up the sample "
                f"(n={n}); {what}. Lower n_rff or the conditioning-set size (each "
                f"conditioning variable adds {1 + self.n_rff} columns)."
            )
        warnings.warn(
            msg + " Such tests are counted in n_saturated and not cached.",
            RuntimeWarning,
            stacklevel=4,
        )

    def _multipliers(self, n):
        """Bootstrap multipliers (n, n_boot), drawn once per sample size and reused
        (each test stays valid)."""
        hit, xi = self._xi_cache.get(n)
        if not hit:
            rng = np.random.default_rng(stable_seed("raft-boot", self.rff_seed, n))
            # Dependent wild bootstrap: Gaussian multipliers with Bartlett
            # correlation over blocks of ~n^(1/3) rows. A length-bl moving sum of
            # i.i.d. normals, scaled by 1/sqrt(bl), has exactly that correlation,
            # max(0, 1 - |i - j| / bl), in O(n * n_boot) time and memory.
            bl = min(max(2, int(np.ceil(n ** (1.0 / 3.0)))), max(2, n // 2))
            # Built in place: about twice the matrix size at peak.
            e = rng.standard_normal((n + bl - 1, self.n_boot))
            np.cumsum(e, axis=0, out=e)  # e[k] = sum of draws 0..k
            xi = e[bl - 1 :].copy()
            xi[1:] -= e[: n - 1]
            xi /= np.sqrt(bl)
            del e
            self._xi_cache.put(n, xi, xi.nbytes)
        return xi

    def _null_pvalue(self, stat, stat_b):
        """Moment-matched scaled chi-square p-value from bootstrap draws.

        The bootstrap calibrates the null's scale and shape, not its tail count:
        match mean and variance to a scaled chi2_nu / nu (Satterthwaite) and read
        the p-value off that curve. Under i.i.d. homoscedastic errors this is the
        usual chi2_q / q, and unlike an exceedance count it resolves p-values far
        below 1 / n_boot (discovery algorithms test at alpha = 0.001).
        """
        stat_b = stat_b[np.isfinite(stat_b)]
        if stat_b.size < 2:
            return 1.0
        m, v = float(stat_b.mean()), float(stat_b.var(ddof=1))
        if not (m > 0 and v > 0):
            return float((1.0 + np.sum(stat_b >= stat)) / (stat_b.size + 1.0))
        nu = 2.0 * m * m / v
        # chdtrc is what chi2.sf evaluates, without its per-call overhead
        return float(special.chdtrc(nu, stat * nu / m))

    @staticmethod
    def _explained(r, y):
        """True when Z leaves no residual variation in y beyond y's own rounding:
        there is nothing for the added block to explain, which must not be read as
        a perfect fit."""
        yc = y - y.mean()
        tol = _explain_tol(_rounding_level(y))
        return float(r @ r) <= tol**2 * float(yc @ yc)

    @staticmethod
    def _added_basis(blocks):
        """Orthonormal basis of the added feature blocks after partialling out Z.

        ``blocks`` holds ``(partialled, original, rounding level)`` per tested
        variable. A block whose variable (its first, linear column) is explained by
        Z to within its rounding is dropped: an affine copy of Z adds nothing. Within a kept block, directions
        weaker than 1e-3 of its strongest carry no information but sit on a few
        high-leverage rows, so they are dropped too. Blocks are judged separately,
        so one variable's scale cannot hide another's."""
        parts = []
        for Bp, Bo, level in blocks:
            tol = _explain_tol(level)
            if np.linalg.norm(Bp[:, 0]) <= tol * np.linalg.norm(Bo[:, 0]):
                continue
            U, sv, _ = _svd(Bp)
            if sv.size and sv[0] > 1e-12:
                parts.append(U[:, sv > sv[0] * 1e-3])
        if len(parts) <= 1:
            return parts[0] if parts else np.zeros((blocks[0][0].shape[0], 0))
        U, sv, _ = _svd(np.hstack(parts))
        return U[:, sv > sv[0] * 1e-8]

    def _additive_response(self, y, Q, mult, key):
        """Response side of ``_one_direction``, which depends on (y, Z) only: the
        (robust) residual, and with the bootstrap the multiplied residuals R with
        their sums of squares in span(Z) and in total; ``None`` when Z leaves
        nothing to explain in y. Cached under ``key`` when given."""
        if key is not None:
            hit, fit = self._resp_cache.get(key)
            if hit:
                return fit
        r = y - Q @ (Q.T @ y)
        fit = None
        if not self._explained(r, y):
            if self.robust:
                # Test the Huber-clipped residual: a function of (y, Z), so under
                # the null it stays mean-independent of the tested features, while
                # a few very large residuals (heavy tails, volatility bursts) no
                # longer dominate the statistic.
                r = _huber(r)
                r = r - Q @ (Q.T @ r)
            R = ss_z = ss = None
            if self.variance == "dwb":
                if self.hc2:
                    # HC2: residuals rescaled by 1 / sqrt(1 - h), h the leverage of
                    # span(Q), so rows the conditioning basis fits almost exactly
                    # still carry their share of noise into the bootstrap.
                    h = np.einsum("ij,ij->i", Q, Q)
                    R = (r / np.sqrt(np.clip(1.0 - h, 1e-6, None)))[:, None] * mult
                else:
                    R = r[:, None] * mult
                # Multiplied residuals leave the residual space: project out
                # span(1, Z) as well as the added block before taking the residual
                # sum of squares, or the denominator is inflated and the test
                # turns liberal.
                ss_z = ((Q.T @ R) ** 2).sum(0)
                ss = np.einsum("ij,ij->j", R, R)
            fit = (r, R, ss_z, ss)
        if key is not None:
            size = 0 if fit is None else sum(a.nbytes for a in fit if a is not None)
            self._resp_cache.put(key, fit, size)
        return fit

    def _one_direction(self, y, FX, Q, mult, key=None):
        """p-value and statistic for 'FX adds to a regression of y on span(Q)'."""
        n, k = y.shape[0], Q.shape[1]
        if n - k < 11:  # conditioning features use up the sample
            self._saturated(n)
            return 1.0, 0.0
        if not FX:
            return 1.0, 0.0
        fit = self._additive_response(y, Q, mult, key)
        if fit is None:  # Z leaves nothing to explain in y
            return 1.0, 0.0
        r, R, ss_z, ss = fit
        U = None
        if key is not None:
            # The added basis depends on the tested blocks and Z only (not on y);
            # a block is fixed by its variable (its linear column) and its width.
            ukey = (
                ("basis",)
                + key[2:]
                + tuple((_content_key(B[:, 0]), B.shape[1], lv) for B, lv in FX)
            )
            hit, U = self._basis_cache.get(ukey)
        if U is None:
            U = self._added_basis([(B - Q @ (Q.T @ B), B, lv) for B, lv in FX])
            if key is not None:
                self._basis_cache.put(ukey, U, U.nbytes)
        q = U.shape[1]
        df2 = n - k - q
        if q == 0:
            return 1.0, 0.0
        if df2 < 10:
            self._saturated(n)
            return 1.0, 0.0
        beta = U.T @ r
        rss0 = float(r @ r)
        rss1 = rss0 - float(beta @ beta)
        if rss1 <= 1e-12 * max(rss0, 1e-300):
            return 0.0, np.inf
        stat = ((rss0 - rss1) / q) / (rss1 / df2)
        if self.variance == "f":
            return float(stats.f.sf(stat, q, df2)), float(stat)
        # Dependent wild bootstrap under the null: restricted residuals r
        # times the multipliers; U is fixed, so each draw is one projection.
        ss_add = ((U.T @ R) ** 2).sum(0)
        stat_b = (ss_add / q) / ((ss - ss_z - ss_add) / df2)
        return self._null_pvalue(stat, stat_b), float(stat)

    def _ridge_weights(self, v, Qa, U, s):
        w = self._gcv_weights(v, Qa, U, s)
        if not self.z_ridge or self.z_ridge_under == 1.0:
            return w
        # Undersmooth: lambda = z_ridge_under * lambda_GCV (recovered from any
        # 0 < w < 1; when GCV drops the dictionary, w = 0 stays).
        mid = (w > 0) & (w < 1)
        if not mid.any():
            return w
        j = int(np.argmax(mid))
        lam = s[j] ** 2 * (1.0 - w[j]) / w[j]
        return s**2 / (s**2 + self.z_ridge_under * lam)

    def _gcv_weights(self, v, Qa, U, s):
        """Ridge weights ``w = s^2 / (s^2 + lam)`` for response v, lam by generalized
        cross-validation over a grid that includes lam = inf (no interaction term)."""
        n, ka = v.shape[0], Qa.shape[1]
        r0 = v - Qa @ (Qa.T @ v)
        c = U.T @ r0
        outside = float(((r0 - U @ c) ** 2).sum())  # untouched by the ridge part
        best = n * float(r0 @ r0) / (n - ka) ** 2  # lam = inf
        lams = _LAM_GRID * s[0] ** 2
        W = s**2 / (s**2 + lams[:, None])
        df = n - ka - W.sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = n * (outside + (((1.0 - W) * c) ** 2).sum(1)) / df**2
        g[df <= 5] = np.inf
        j = int(np.argmin(g))  # the first minimum, as a scan would keep it
        if not g[j] < best:
            return np.zeros(s.size)
        return s**2 / (s**2 + lams[j])

    @staticmethod
    def _residualize(V, Qa, U, w):
        """Apply M = I - Qa Qa' - U diag(w) U' (the additive projection plus the
        ridge smoother) to the columns of V."""
        V = V - Qa @ (Qa.T @ V)
        if w.size:
            V = V - U @ ((w[:, None] if V.ndim == 2 else w) * (U.T @ V))
        return V

    def _ridge_response(self, y, Qa, U, s, mult, key):
        """Response side of ``_ridge_direction``, which depends on (y, Z) only: the
        smoother weights, the (robust) residual and its sum of squares, and the
        smoothed bootstrap residuals with their sums of squares; ``None`` when Z
        leaves nothing to explain in y. Cached under ``key`` when given."""
        if key is not None:
            hit, fit = self._resp_cache.get(key)
            if hit:
                return fit
        w = self._ridge_weights(y, Qa, U, s)
        big = w > self.w_tol
        Uw, ww = U[:, big], w[big]
        r = self._residualize(y, Qa, Uw, ww)
        fit = None
        if not self._explained(r, y):
            if self.robust:
                # Huber step on the smoother's residual, smoothed again.
                r = self._residualize(_huber(r), Qa, Uw, ww)
            if self.hc2:
                # HC2 for the ridge smoother M = I - S: Var[(M e)_i] / sigma^2 =
                # 1 - |Qa_i|^2 - sum_j (2 w_j - w_j^2) U_ij^2.
                h = np.einsum("ij,ij->i", Qa, Qa) + (Uw**2) @ (2.0 * ww - ww**2)
                r_b = r / np.sqrt(np.clip(1.0 - h, 1e-6, None))
            else:
                r_b = r
            Rb = self._residualize(r_b[:, None] * mult, Qa, Uw, ww)
            ss = np.einsum("ij,ij->j", Rb, Rb)
            fit = (Uw, ww, r, float(r @ r), Rb, ss)
        if key is not None:
            size = 0 if fit is None else sum(a.nbytes for a in (Uw, ww, r, Rb, ss))
            self._resp_cache.put(key, fit, size)
        return fit

    def _ridge_direction(self, y, FX, Qa, U, s, mult, key=None):
        """As ``_one_direction``, with the ridge-smoothed interaction adjustment.

        The bootstrap pushes the multiplied residuals through the same smoother
        before recomputing the statistic: the counterpart of projecting span(Z)
        out of the bootstrap residuals in the additive test. The ratio statistic
        is scale-free, so the smoother's shrinkage of the residuals cancels.
        """
        n, ka = y.shape[0], Qa.shape[1]
        if n - ka < 11 or not FX:
            if n - ka < 11:
                self._saturated(n)
            return 1.0, 0.0
        fit = self._ridge_response(y, Qa, U, s, mult, key)
        if fit is None:  # Z leaves nothing to explain in y
            return 1.0, 0.0
        Uw, ww, r, rss0, Rb, ss = fit
        Ux = self._added_basis(
            [(self._residualize(B, Qa, Uw, ww), B, lv) for B, lv in FX]
        )
        q = Ux.shape[1]
        df2 = n - ka - float(ww.sum()) - q
        if q == 0:
            return 1.0, 0.0
        if df2 < 10:
            self._saturated(n)
            return 1.0, 0.0
        b = Ux.T @ r
        ssa = float(b @ b)
        if rss0 - ssa <= 1e-12 * max(rss0, 1e-300):
            return 0.0, np.inf
        stat = (ssa / q) / ((rss0 - ssa) / df2)
        ssa_b = ((Ux.T @ Rb) ** 2).sum(0)
        stat_b = (ssa_b / q) / ((ss - ssa_b) / df2)
        return self._null_pvalue(stat, stat_b), float(stat)

    def __call__(self, X, Y, condition_set=None):
        self.n_tests += 1

        # Canonical (sorted, unique) X, Y and Z: results and random draws must not
        # depend on the order the caller lists variables in.
        def canon(v):
            arr = np.atleast_1d(v if v is not None else [])
            if arr.size and (
                arr.dtype == bool or not np.issubdtype(arr.dtype, np.integer)
            ):
                raise ValueError(f"column indices must be integers, got {v!r}")
            return sorted({int(c) for c in arr})

        xi, yi, zi = canon(X), canon(Y), canon(condition_set)
        cache_key = self._get_array_hash(xi, yi, zi)
        param_hash = self._param_hash()
        cached = self.pvalue_cache.get((cache_key, param_hash))
        if cached is not None:
            return cached[0], cached[1]
        self._call_saturated = False
        pval, stat = self._test(xi, yi, zi)
        # Saturated results are not cached, so a repeat (or a reload from disk)
        # still counts and warns.
        if not self._call_saturated:
            self.pvalue_cache[(cache_key, param_hash)] = [pval, stat]
            self.save_incremental_cache(cache_key, param_hash, pval, stat)
        return pval, stat

    def _test(self, xi, yi, zi):
        self._sat_dirs = 0
        # Work on a copy of the columns in use: missing values are checked on every
        # uncached call (the array may have been edited in place), and each column
        # is scaled by a power of two, which is exact (results are unchanged) and
        # keeps sums of squares finite at extreme magnitudes.
        cols = sorted(set(xi) | set(yi) | set(zi))
        raw = self._data[:, cols]
        if np.issubdtype(raw.dtype, np.integer):
            # Centre exactly in integer arithmetic (unsigned wrap-around gives the
            # true offset): float64 cannot hold variation above 2**53.
            raw = raw.astype(np.uint64) - raw.min(axis=0).astype(np.uint64)
        data = np.asarray(raw, dtype=np.float64)
        if np.isinf(data).any():
            raise ValueError("raft/rift: the tested columns contain infinite values")
        mask = None
        ok = ~np.isnan(data).any(axis=1)
        if not ok.all():
            if ok.sum() < self._min_samples:
                return 1.0, 0.0
            data, mask = data[ok], ok
        if data.shape[0] < 12:  # too few rows for any regression with an intercept
            self._report_saturation(data.shape[0], full=True, rows=True)
            return 1.0, 0.0
        data = np.ldexp(data, -np.frexp(np.abs(data).max(axis=0))[1])
        pos = {c: k for k, c in enumerate(cols)}
        xi, yi, zi = ([pos[c] for c in v] for v in (xi, yi, zi))
        self.n_actual_tests += 1
        # Tested components that are constant, or identical to a conditioning column
        # (adapters copy columns, so compare content, not indices), carry no
        # information beyond Z and as responses would be fitted perfectly; leave
        # them out, so they cost no Bonferroni share either. Only exact copies are
        # removed: a looser match could erase a genuine variable.
        zbytes = {data[:, j].tobytes() for j in zi}

        def informative(j):
            x = data[:, j]
            return not _is_constant(x) and x.tobytes() not in zbytes

        def unique(idx):  # a duplicated component adds nothing but a Bonferroni share
            seen, out = set(), []
            for j in idx:
                key = (data[:, j] + 0.0).tobytes()  # -0.0 and 0.0 alike
                if key not in seen:
                    seen.add(key)
                    out.append(j)
            return out

        xr = unique([j for j in xi if informative(j)])
        yr = unique([j for j in yi if informative(j)])
        if not xr or not yr:
            return 1.0, 0.0

        # One feature block per tested component (constant ones have none).
        m = self._tested_rff(data.shape[0])

        def blocks(idx):
            out = [
                (self._features([j], data, m), _rounding_level(data[:, j])) for j in idx
            ]
            return [(b, lv) for b, lv in out if b.shape[1]]

        FX, FY = blocks(xr), blocks(yr)
        Qa, U, s = self._z_parts(zi, data)
        mult = None
        if self.variance == "dwb":
            # Multipliers live on the original timeline: rows dropped for NaNs
            # leave gaps, and the retained rows keep their true time distances.
            mult = self._multipliers(self._data.shape[0])
            if mask is not None:
                mult = mult[mask]
        # Both directions share the Z adjustment; each regresses every response
        # column on the other block.
        res, tested = [], []
        lc = self.linear_channel
        lin = lc is True or (lc is not False and data.shape[0] <= lc)

        # The response fit depends on the data, Z, the rows kept and the settings
        # that shape it; the content keys keep it valid if the array is swapped.
        base = (
            _content_key(data[:, zi]),
            None if mask is None else _content_key(mask),
            self._data.shape[0],
            self._param_hash(),
        )

        def direction(y, F):
            if s.size:
                key = (_content_key(y),) + base
                return self._ridge_direction(y, F, Qa, U, s, mult, key)
            return self._one_direction(y, F, Qa, mult, ("add", _content_key(y)) + base)

        for j, F in [(j, FX) for j in yr] + [(j, FY) for j in xr]:
            before = self._sat_dirs
            out = direction(data[:, j], F)
            if lin and any(B.shape[1] > 1 for B, _ in F):
                # Linear channel: the same direction with only the tested variables'
                # linear features, Cauchy-combined with the full block. At small T
                # a nearly linear dependence is then tested on one degree of freedom
                # per variable; either part alone is used if the other was skipped.
                mid = self._sat_dirs
                out_l = direction(data[:, j], [(B[:, :1], lv) for B, lv in F])
                full_ok = mid == before and out != (1.0, 0.0)
                lin_ok = self._sat_dirs == mid and out_l != (1.0, 0.0)
                self._sat_dirs = before if (full_ok or lin_ok) else mid
                if full_ok and lin_ok:
                    # report the statistic of the channel that carried the evidence
                    st = out[1] if out[0] <= out_l[0] else out_l[1]
                    out = (_cauchy([out[0], out_l[0]]), st)
                elif lin_ok:
                    out = out_l
            res.append(out)
            # A skipped (saturated) direction, or one with nothing to test (the
            # response is explained by Z, or the added block is empty), returns
            # (1, 0); the Cauchy combination uses only directions actually tested.
            # An exact p = 1 (degenerate bootstrap draws) is left out too: in the
            # Cauchy mean it would outweigh any other direction.
            if self._sat_dirs == before and res[-1][0] < 1.0:
                tested.append(res[-1][0])
        # Report the statistic of the direction that set the p-value, so tests
        # sorted by statistic (e.g. tigramite's parent ordering) follow p-values.
        if self._sat_dirs:
            self._report_saturation(data.shape[0], full=self._sat_dirs == len(res))
        best = min(res, key=lambda r: (r[0], -r[1]))
        # Without the ridge dictionary skipping is decided by the design (Z
        # features, added-block rank), so Bonferroni counts only the directions
        # actually tested. With it, the GCV penalty and hence the skip depend on
        # the response, so every direction keeps its share; any direction left
        # untested there (skipped, or explained by Z or with an empty added block,
        # both judged after the response's own GCV fit) also sends the Cauchy
        # combination back to Bonferroni over every direction.
        cauchy = self.combine == "cauchy" and not (s.size and len(tested) < len(res))
        if cauchy and len(tested) == 1:
            pval = float(tested[0])
        elif cauchy and len(tested) >= 2:
            # Cauchy combination (Liu and Xie 2020): valid under arbitrary
            # dependence between the directions at small alpha, and it loses less
            # than Bonferroni when several directions carry the signal.
            q = np.clip(np.asarray(tested), 1e-15, 1.0 - 1e-15)
            pval = float(0.5 - np.arctan(np.mean(np.tan((0.5 - q) * np.pi))) / np.pi)
        else:
            m = len(res) if s.size else max(1, len(res) - self._sat_dirs)
            pval = float(min(1.0, m * best[0]))
        stat = float(best[1])
        return pval, stat


class RIFTCI(RAFTCI):
    """``rift``: :class:`RAFTCI` with ``adjust="interaction"`` (see the module
    docstring)."""

    def __init__(self, data, adjust="interaction", **kwargs):
        super().__init__(data=data, adjust=adjust, **kwargs)
