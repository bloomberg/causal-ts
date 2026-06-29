# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Changepoint-based regime detectors using statistical two-sample tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import bartlett, ks_2samp, levene

from .segments import Segment


@dataclass
class _CPSegment:
    start: int
    end: int
    regime: int
    mean_value: float


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _extract_univariate_signal(
    df: pd.DataFrame,
    columns: list | None,
    signal: str = "mean",
    volatility_window: int = 48,
) -> np.ndarray:
    """Reduce multivariate DataFrame to a 1-D signal for changepoint detection."""
    sub = df[columns] if columns else df
    indicator = sub.mean(axis=1).interpolate()

    if signal == "mean":
        return indicator.values
    elif signal == "volatility":
        roll_std = (
            indicator.rolling(volatility_window, min_periods=volatility_window // 2)
            .std()
            .bfill()
        )
        return roll_std.values
    else:
        raise ValueError(f"Unknown signal: {signal!r}")


def _classify_breakpoints(
    signal: np.ndarray,
    breakpoints: list[int],
    n_regimes: int,
) -> list[_CPSegment]:
    """Quantile-classify segments by their mean values."""
    if not breakpoints:
        return [
            _CPSegment(
                start=0,
                end=len(signal),
                regime=0,
                mean_value=float(signal.mean()),
            )
        ]

    bkps = sorted(breakpoints)
    if bkps[-1] != len(signal):
        bkps.append(len(signal))

    starts = [0] + bkps[:-1]
    ends = bkps
    seg_means = np.array([signal[s:e].mean() for s, e in zip(starts, ends)])

    if n_regimes >= len(seg_means):
        regime_labels = np.arange(len(seg_means))
    else:
        thresholds = np.quantile(seg_means, np.linspace(0, 1, n_regimes + 1)[1:-1])
        regime_labels = np.digitize(seg_means, thresholds)

    return [
        _CPSegment(start=s, end=e, regime=int(r), mean_value=float(m))
        for s, e, r, m in zip(starts, ends, regime_labels, seg_means)
    ]


def _binary_segmentation(
    signal: np.ndarray,
    test_fn,
    alpha: float,
    min_segment: int,
    step: int,
    max_changepoints: int,
) -> list[int]:
    """Recursive binary segmentation with an arbitrary two-sample test.

    Parameters
    ----------
    signal : 1-D array
    test_fn : callable(left, right) -> p_value
    alpha : significance threshold
    min_segment : minimum samples on each side of a candidate split
    step : stride between candidate split points
    max_changepoints : hard cap on number of breakpoints returned
    """
    breakpoints: list[int] = []

    def _recurse(lo: int, hi: int) -> None:
        if len(breakpoints) >= max_changepoints:
            return
        if hi - lo < 2 * min_segment:
            return

        best_p = 1.0
        best_pos = -1
        for pos in range(lo + min_segment, hi - min_segment + 1, step):
            left = signal[lo:pos]
            right = signal[pos:hi]
            try:
                p = test_fn(left, right)
            except Exception:
                continue
            if p < best_p:
                best_p = p
                best_pos = pos

        if best_p < alpha and best_pos > 0:
            breakpoints.append(best_pos)
            _recurse(lo, best_pos)
            _recurse(best_pos, hi)

    _recurse(0, len(signal))
    return sorted(breakpoints)


def _segments_from_cp(
    df: pd.DataFrame,
    signal: np.ndarray,
    breakpoints: list[int],
    n_regimes: int,
) -> list[Segment]:
    """Build Segment objects from breakpoints."""
    cp_segs = _classify_breakpoints(signal, breakpoints, n_regimes)
    return [
        Segment(
            df=df.iloc[seg.start : seg.end].copy(),
            regime=seg.regime,
            start_idx=seg.start,
            end_idx=seg.end,
        )
        for seg in cp_segs
    ]


def _labels_from_cp(
    n: int,
    signal: np.ndarray,
    breakpoints: list[int],
    n_regimes: int,
) -> np.ndarray:
    """Build per-timestep regime label array from breakpoints."""
    cp_segs = _classify_breakpoints(signal, breakpoints, n_regimes)
    labels = np.zeros(n, dtype=np.int32)
    for seg in cp_segs:
        labels[seg.start : seg.end] = seg.regime
    return labels


# ---------------------------------------------------------------------------
# LeveneRegimeDetector
# ---------------------------------------------------------------------------


class LeveneRegimeDetector:
    """Detect variance/volatility regime shifts via Levene's (or Bartlett's) test.

    Uses recursive binary segmentation: at each step the segment is scanned
    for the split point with the most significant variance difference, and
    the split is accepted if p < ``alpha``.

    Parameters
    ----------
    n_regimes : int
        Number of regimes to classify segments into.
    alpha : float
        Significance threshold for accepting a split.
    min_segment : int
        Minimum number of samples on each side of a candidate split.
    step : int
        Stride between candidate split positions.
    method : str
        ``"levene"`` (robust, uses median) or ``"bartlett"`` (assumes normality).
    max_changepoints : int
        Maximum number of breakpoints to find.
    signal : str
        ``"mean"`` or ``"volatility"`` — how to reduce multivariate data.
    volatility_window : int
        Rolling-std window when ``signal="volatility"``.
    columns : list or None
        Which DataFrame columns to use. None = all.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        alpha: float = 0.05,
        min_segment: int = 30,
        step: int = 5,
        method: str = "levene",
        max_changepoints: int = 10,
        signal: str = "mean",
        volatility_window: int = 48,
        columns: list | None = None,
    ):
        self.n_regimes = n_regimes
        self.alpha = alpha
        self.min_segment = min_segment
        self.step = step
        self.method = method
        self.max_changepoints = max_changepoints
        self.signal = signal
        self.volatility_window = volatility_window
        self.columns = columns

    def _test_fn(self, left: np.ndarray, right: np.ndarray) -> float:
        if self.method == "levene":
            _, p = levene(left, right, center="median")
        elif self.method == "bartlett":
            _, p = bartlett(left, right)
        else:
            raise ValueError(f"Unknown method: {self.method!r}")
        return float(p)

    def _get_signal(self, df: pd.DataFrame) -> np.ndarray:
        return _extract_univariate_signal(
            df, self.columns, self.signal, self.volatility_window
        )

    def _find_breakpoints(self, signal: np.ndarray) -> list[int]:
        return _binary_segmentation(
            signal,
            self._test_fn,
            self.alpha,
            self.min_segment,
            self.step,
            self.max_changepoints,
        )

    def assign(self, df: pd.DataFrame) -> np.ndarray:
        signal = self._get_signal(df)
        bkps = self._find_breakpoints(signal)
        return _labels_from_cp(len(df), signal, bkps, self.n_regimes)

    def detect_segments(self, df: pd.DataFrame) -> list[Segment]:
        signal = self._get_signal(df)
        bkps = self._find_breakpoints(signal)
        return _segments_from_cp(df, signal, bkps, self.n_regimes)


# ---------------------------------------------------------------------------
# KSRegimeDetector
# ---------------------------------------------------------------------------


class KSRegimeDetector:
    """Detect distributional regime shifts via the Kolmogorov-Smirnov test.

    Uses recursive binary segmentation with ``scipy.stats.ks_2samp`` as the
    splitting criterion.  Catches any distributional change (mean, variance,
    shape, skewness) — the most general-purpose detector.

    Parameters
    ----------
    n_regimes : int
        Number of regimes to classify segments into.
    alpha : float
        Significance threshold for accepting a split.
    min_segment : int
        Minimum samples on each side of a candidate split.
    step : int
        Stride between candidate split positions.
    max_changepoints : int
        Maximum breakpoints to find.
    signal : str
        ``"mean"`` or ``"volatility"`` — how to reduce multivariate data.
    volatility_window : int
        Rolling-std window when ``signal="volatility"``.
    columns : list or None
        Which DataFrame columns to use. None = all.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        alpha: float = 0.05,
        min_segment: int = 30,
        step: int = 5,
        max_changepoints: int = 10,
        signal: str = "mean",
        volatility_window: int = 48,
        columns: list | None = None,
    ):
        self.n_regimes = n_regimes
        self.alpha = alpha
        self.min_segment = min_segment
        self.step = step
        self.max_changepoints = max_changepoints
        self.signal = signal
        self.volatility_window = volatility_window
        self.columns = columns

    @staticmethod
    def _test_fn(left: np.ndarray, right: np.ndarray) -> float:
        _, p = ks_2samp(left, right)
        return float(p)

    def _get_signal(self, df: pd.DataFrame) -> np.ndarray:
        return _extract_univariate_signal(
            df, self.columns, self.signal, self.volatility_window
        )

    def _find_breakpoints(self, signal: np.ndarray) -> list[int]:
        return _binary_segmentation(
            signal,
            self._test_fn,
            self.alpha,
            self.min_segment,
            self.step,
            self.max_changepoints,
        )

    def assign(self, df: pd.DataFrame) -> np.ndarray:
        signal = self._get_signal(df)
        bkps = self._find_breakpoints(signal)
        return _labels_from_cp(len(df), signal, bkps, self.n_regimes)

    def detect_segments(self, df: pd.DataFrame) -> list[Segment]:
        signal = self._get_signal(df)
        bkps = self._find_breakpoints(signal)
        return _segments_from_cp(df, signal, bkps, self.n_regimes)


# ---------------------------------------------------------------------------
# CusumRegimeDetector
# ---------------------------------------------------------------------------


class CusumRegimeDetector:
    """Detect regime shifts via CUSUM (Cumulative Sum) control charts.

    Computes the CUSUM statistic for mean and/or variance shifts, then
    finds changepoints where the CUSUM slope changes direction.

    Parameters
    ----------
    n_regimes : int
        Number of regimes to classify segments into.
    target : str
        What type of shift to detect: ``"mean"``, ``"variance"``, or
        ``"both"`` (union of changepoints from both).
    threshold : float or None
        CUSUM magnitude threshold for declaring a changepoint.  If None,
        auto-calibrated as ``3 * std(cusum)``.
    min_segment : int
        Minimum spacing between changepoints.
    smooth_window : int
        Window size for smoothing the CUSUM derivative.
    signal : str
        ``"mean"`` or ``"volatility"`` — how to reduce multivariate data.
    volatility_window : int
        Rolling-std window when ``signal="volatility"``.
    columns : list or None
        Which DataFrame columns to use. None = all.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        target: str = "variance",
        threshold: float | None = None,
        min_segment: int = 30,
        smooth_window: int = 50,
        signal: str = "mean",
        volatility_window: int = 48,
        columns: list | None = None,
    ):
        self.n_regimes = n_regimes
        self.target = target
        self.threshold = threshold
        self.min_segment = min_segment
        self.smooth_window = smooth_window
        self.signal = signal
        self.volatility_window = volatility_window
        self.columns = columns

    def _get_signal(self, df: pd.DataFrame) -> np.ndarray:
        return _extract_univariate_signal(
            df, self.columns, self.signal, self.volatility_window
        )

    @staticmethod
    def _cusum_mean(signal: np.ndarray) -> np.ndarray:
        mu = signal.mean()
        return np.cumsum(signal - mu)

    @staticmethod
    def _cusum_variance(signal: np.ndarray) -> np.ndarray:
        mu = signal.mean()
        sigma2 = signal.var()
        return np.cumsum((signal - mu) ** 2 - sigma2)

    def _find_cusum_changepoints(self, cusum: np.ndarray) -> list[int]:
        """Find changepoints from a CUSUM curve via slope-change detection."""
        n = len(cusum)
        if n < 2 * self.min_segment:
            return []

        w = min(self.smooth_window, n // 4)
        if w < 3:
            return []

        kernel = np.ones(w) / w
        deriv = np.diff(cusum)
        if len(deriv) < w:
            return []

        smoothed = np.convolve(deriv, kernel, mode="same")

        thresh = self.threshold
        if thresh is None:
            cusum_std = np.std(cusum)
            if cusum_std < 1e-10:
                return []
            thresh = 3.0 * cusum_std

        sign_changes = np.diff(np.sign(smoothed))
        candidates = np.where(sign_changes != 0)[0] + 1

        abs_cusum = np.abs(cusum)
        filtered = [int(c) for c in candidates if c < n and abs_cusum[c] > thresh * 0.3]

        if not filtered:
            return []
        result = [filtered[0]]
        for c in filtered[1:]:
            if c - result[-1] >= self.min_segment:
                result.append(c)
        return result

    def _find_breakpoints(self, signal: np.ndarray) -> list[int]:
        if self.target == "mean":
            cusum = self._cusum_mean(signal)
            return self._find_cusum_changepoints(cusum)
        elif self.target == "variance":
            cusum = self._cusum_variance(signal)
            return self._find_cusum_changepoints(cusum)
        elif self.target == "both":
            bkps_mean = self._find_cusum_changepoints(self._cusum_mean(signal))
            bkps_var = self._find_cusum_changepoints(self._cusum_variance(signal))
            merged = sorted(set(bkps_mean + bkps_var))
            if not merged:
                return []
            result = [merged[0]]
            for c in merged[1:]:
                if c - result[-1] >= self.min_segment:
                    result.append(c)
            return result
        else:
            raise ValueError(f"Unknown target: {self.target!r}")

    def assign(self, df: pd.DataFrame) -> np.ndarray:
        signal = self._get_signal(df)
        bkps = self._find_breakpoints(signal)
        return _labels_from_cp(len(df), signal, bkps, self.n_regimes)

    def detect_segments(self, df: pd.DataFrame) -> list[Segment]:
        signal = self._get_signal(df)
        bkps = self._find_breakpoints(signal)
        return _segments_from_cp(df, signal, bkps, self.n_regimes)


# ---------------------------------------------------------------------------
# EnsembleRegimeDetector
# ---------------------------------------------------------------------------


def _rolling_profiles(
    signal: np.ndarray,
    window: int,
    features: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Compute rolling statistics over a 1-D signal."""
    from scipy.stats import kurtosis as kurt_fn

    n = len(signal)
    if n < window:
        raise ValueError(f"Signal length {n} < window {window}")

    profiles: dict[str, np.ndarray] = {}
    n_windows = n - window + 1

    if "mean" in features:
        cumsum = np.cumsum(np.insert(signal, 0, 0))
        profiles["mean"] = (cumsum[window:] - cumsum[:n_windows]) / window

    if "variance" in features:
        cs = np.cumsum(np.insert(signal, 0, 0))
        cs2 = np.cumsum(np.insert(signal**2, 0, 0))
        s = cs[window:] - cs[:n_windows]
        s2 = cs2[window:] - cs2[:n_windows]
        profiles["variance"] = s2 / window - (s / window) ** 2

    if "slope" in features:
        t = np.arange(window, dtype=float)
        t_mean = t.mean()
        t_var = ((t - t_mean) ** 2).sum()
        slopes = np.empty(n_windows)
        for i in range(n_windows):
            seg = signal[i : i + window]
            slopes[i] = ((t - t_mean) * (seg - seg.mean())).sum() / t_var
        profiles["slope"] = slopes

    if "kurtosis" in features:
        kurts = np.empty(n_windows)
        for i in range(n_windows):
            kurts[i] = float(kurt_fn(signal[i : i + window]))
        profiles["kurtosis"] = kurts

    return profiles


def _merge_changepoints(
    all_cps: list[list[int]],
    tolerance: int,
    min_segment: int,
) -> list[int]:
    """Merge changepoints from multiple features.

    CPs within ``tolerance`` of each other are clustered; each cluster
    is represented by its median.  The final list is filtered so that
    consecutive CPs are at least ``min_segment`` apart.
    """
    flat = sorted(c for cps in all_cps for c in cps)
    if not flat:
        return []

    clusters: list[list[int]] = [[flat[0]]]
    for c in flat[1:]:
        if c - clusters[-1][-1] <= tolerance:
            clusters[-1].append(c)
        else:
            clusters.append([c])

    merged = [int(np.median(cl)) for cl in clusters]

    result = [merged[0]]
    for c in merged[1:]:
        if c - result[-1] >= min_segment:
            result.append(c)
    return result


class EnsembleRegimeDetector:
    """Generic multi-feature changepoint detector.

    Computes rolling statistics (mean, variance, slope, kurtosis) in
    sliding windows, runs PELT independently on each normalised
    feature profile, and merges the detected changepoints.

    This detects **any** type of regime change — mean shifts, variance
    shifts, trend changes, and distributional shape changes — because
    at least one feature profile will exhibit an abrupt level shift at
    the true changepoint.

    Parameters
    ----------
    n_regimes : int
        Number of regimes to classify segments into.
    features : tuple of str
        Rolling statistics to compute.  Options: ``"mean"``,
        ``"variance"``, ``"slope"``, ``"kurtosis"``.
    window : int
        Rolling window size for computing statistics.
    pen : float
        PELT penalty applied to each feature profile.
    min_segment : int
        Minimum spacing between final changepoints.
    merge_tolerance : int
        CPs from different features within this many samples of each
        other are merged into a single changepoint.
    signal : str
        ``"mean"`` or ``"volatility"`` — how to reduce multivariate data.
    volatility_window : int
        Rolling-std window when ``signal="volatility"``.
    columns : list or None
        Which DataFrame columns to use. None = all.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        features: tuple[str, ...] = ("mean", "variance", "slope"),
        window: int = 100,
        pen: float = 200,
        min_segment: int = 50,
        merge_tolerance: int = 50,
        signal: str = "mean",
        volatility_window: int = 48,
        columns: list | None = None,
    ):
        self.n_regimes = n_regimes
        self.features = tuple(features)
        self.window = window
        self.pen = pen
        self.min_segment = min_segment
        self.merge_tolerance = merge_tolerance
        self.signal = signal
        self.volatility_window = volatility_window
        self.columns = columns

    def _get_signal(self, df: pd.DataFrame) -> np.ndarray:
        return _extract_univariate_signal(
            df, self.columns, self.signal, self.volatility_window
        )

    def _find_breakpoints(self, signal: np.ndarray) -> list[int]:
        from ..regime.pelt import _pelt_l2

        profiles = _rolling_profiles(signal, self.window, self.features)
        half_w = self.window // 2

        all_cps: list[list[int]] = []
        for _feat_name, profile in profiles.items():
            std = profile.std()
            if std < 1e-10:
                continue
            normed = (profile - profile.mean()) / std
            pelt_min = max(10, self.min_segment // 2)
            bkps = _pelt_l2(normed, min_size=pelt_min, pen=self.pen)
            cps = [b + half_w for b in bkps[:-1] if b + half_w < len(signal)]
            all_cps.append(cps)

        return _merge_changepoints(all_cps, self.merge_tolerance, self.min_segment)

    def assign(self, df: pd.DataFrame) -> np.ndarray:
        signal = self._get_signal(df)
        bkps = self._find_breakpoints(signal)
        return _labels_from_cp(len(df), signal, bkps, self.n_regimes)

    def detect_segments(self, df: pd.DataFrame) -> list[Segment]:
        signal = self._get_signal(df)
        bkps = self._find_breakpoints(signal)
        return _segments_from_cp(df, signal, bkps, self.n_regimes)


# ---------------------------------------------------------------------------
# HMMRegimeDetector
# ---------------------------------------------------------------------------


def _sticky_viterbi(
    log_emissionprob: np.ndarray,
    log_transmat: np.ndarray,
    log_startprob: np.ndarray,
    switch_cost: float = 0.0,
) -> np.ndarray:
    """Viterbi decoding with an additive penalty for state transitions.

    Each transition from state *i* to a different state *j* incurs an
    extra cost of ``switch_cost`` in negative-log-likelihood space.
    Higher values produce longer (stickier) segments.
    """
    T, K = log_emissionprob.shape
    V = np.zeros((T, K))
    B = np.zeros((T, K), dtype=np.int32)
    V[0] = log_startprob + log_emissionprob[0]

    for t in range(1, T):
        for j in range(K):
            scores = V[t - 1] + log_transmat[:, j]
            if switch_cost > 0:
                for i in range(K):
                    if i != j:
                        scores[i] -= switch_cost
            B[t, j] = np.argmax(scores)
            V[t, j] = scores[B[t, j]] + log_emissionprob[t, j]

    path = np.zeros(T, dtype=np.int32)
    path[-1] = np.argmax(V[-1])
    for t in range(T - 2, -1, -1):
        path[t] = B[t + 1, path[t + 1]]
    return path


def _auto_switch_cost(
    log_emissionprob: np.ndarray,
    log_transmat: np.ndarray,
    log_startprob: np.ndarray,
    target_cps: int,
) -> float:
    """Binary-search for the switch_cost that yields ~target_cps transitions."""
    lo, hi = 0.0, 100_000.0
    for _ in range(30):
        mid = (lo + hi) / 2
        path = _sticky_viterbi(log_emissionprob, log_transmat, log_startprob, mid)
        cps = int((np.diff(path) != 0).sum())
        if cps > target_cps:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


class HMMRegimeDetector:
    """Regime detection via Gaussian Hidden Markov Model.

    Fits a multivariate Gaussian HMM to the data and decodes the most
    likely state sequence using **sticky Viterbi** — standard Viterbi
    with an additive penalty for state transitions.  The penalty is
    auto-calibrated from ``min_segment`` so that segments are at least
    that long on average (no manual tuning needed).

    Unlike the other detectors which reduce data to a 1-D signal, this
    uses **all columns** jointly so it can detect changes in
    cross-variable structure.

    When ``n_regimes="auto"``, fits models for k=2..``max_regimes`` and
    selects the k with the lowest BIC.

    Requires ``hmmlearn`` (``pip install hmmlearn``).

    Parameters
    ----------
    n_regimes : int or ``"auto"``
        Number of hidden states.  ``"auto"`` uses BIC model selection.
    max_regimes : int
        Upper bound for BIC search when ``n_regimes="auto"``.
    covariance_type : str
        HMM covariance type: ``"full"``, ``"diag"``, ``"spherical"``,
        ``"tied"``.
    n_iter : int
        Maximum EM iterations.
    min_segment : int
        Target minimum segment length.  The sticky-Viterbi switch cost
        is auto-calibrated so that ``T / min_segment`` transitions are
        produced.
    columns : list or None
        Which DataFrame columns to use. None = all.
    random_state : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_regimes: int | str = 3,
        max_regimes: int = 6,
        covariance_type: str = "full",
        n_iter: int = 200,
        min_segment: int = 30,
        columns: list | None = None,
        random_state: int | None = 42,
    ):
        self.n_regimes = n_regimes
        self.max_regimes = max_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.min_segment = min_segment
        self.columns = columns
        self.random_state = random_state

    @staticmethod
    def _require_hmmlearn():
        try:
            from hmmlearn.hmm import GaussianHMM

            return GaussianHMM
        except ImportError:
            raise ImportError(
                "HMMRegimeDetector requires hmmlearn. "
                "Install it with: pip install hmmlearn"
            )

    def _get_data(self, df: pd.DataFrame) -> np.ndarray:
        sub = df[self.columns] if self.columns else df
        data = sub.interpolate().ffill().bfill().values.astype(np.float64)
        return np.ascontiguousarray(data)

    def _fit_hmm(self, data: np.ndarray, k: int):
        GaussianHMM = self._require_hmmlearn()
        hmm = GaussianHMM(
            n_components=k,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        hmm.fit(data)
        return hmm

    def _select_n_regimes(self, data: np.ndarray) -> int:
        n, d = data.shape
        best_bic = np.inf
        best_k = 2
        for k in range(2, self.max_regimes + 1):
            try:
                hmm = self._fit_hmm(data, k)
                log_l = hmm.score(data)
            except Exception:
                continue
            if self.covariance_type == "full":
                n_cov = k * d * (d + 1) // 2
            elif self.covariance_type == "diag":
                n_cov = k * d
            elif self.covariance_type == "spherical":
                n_cov = k
            else:
                n_cov = d * (d + 1) // 2
            n_params = k * d + n_cov + k * (k - 1)
            bic = -2 * log_l + n_params * np.log(n)
            if bic < best_bic:
                best_bic = bic
                best_k = k
        return best_k

    def _decode_sticky(self, hmm, data: np.ndarray) -> np.ndarray:
        """Decode with sticky Viterbi, auto-calibrating the switch cost."""
        log_ep = hmm._compute_log_likelihood(data)
        log_tm = np.log(hmm.transmat_)
        log_sp = np.log(hmm.startprob_)

        target_cps = max(1, len(data) // self.min_segment)

        path_plain = _sticky_viterbi(log_ep, log_tm, log_sp, 0.0)
        actual_cps = int((np.diff(path_plain) != 0).sum())

        if actual_cps <= target_cps:
            return path_plain

        cost = _auto_switch_cost(log_ep, log_tm, log_sp, target_cps)
        return _sticky_viterbi(log_ep, log_tm, log_sp, cost)

    def assign(self, df: pd.DataFrame) -> np.ndarray:
        data = self._get_data(df)
        if self.n_regimes == "auto":
            k = self._select_n_regimes(data)
        else:
            k = int(self.n_regimes)
        hmm = self._fit_hmm(data, k)
        return self._decode_sticky(hmm, data)

    def detect_segments(self, df: pd.DataFrame) -> list[Segment]:
        labels = self.assign(df)
        segments: list[Segment] = []
        start = 0
        for i in range(1, len(labels)):
            if labels[i] != labels[start]:
                segments.append(
                    Segment(
                        df=df.iloc[start:i].copy(),
                        regime=int(labels[start]),
                        start_idx=start,
                        end_idx=i,
                    )
                )
                start = i
        segments.append(
            Segment(
                df=df.iloc[start:].copy(),
                regime=int(labels[start]),
                start_idx=start,
                end_idx=len(labels),
            )
        )
        return segments
