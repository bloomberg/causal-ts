# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Behaviour tests for the changepoint regime detectors (causalts.regime.changepoint)."""

import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_ind

from causalts.regime import (
    CusumRegimeDetector,
    EnsembleRegimeDetector,
    HMMRegimeDetector,
    KSRegimeDetector,
    LeveneRegimeDetector,
)
from causalts.regime.changepoint import (
    _auto_switch_cost,
    _binary_segmentation,
    _classify_breakpoints,
    _extract_univariate_signal,
    _merge_changepoints,
    _rolling_profiles,
    _sticky_viterbi,
)

T = 300
CP = 150


def _mean_shift(seed=0, shift=3.0, n=T, cp=CP):
    rng = np.random.default_rng(seed)
    a = np.r_[rng.standard_normal(cp), shift + rng.standard_normal(n - cp)]
    b = a + 0.3 * rng.standard_normal(n)
    return pd.DataFrame({"a": a, "b": b})


def _var_shift(seed=0, scale=4.0, n=T, cp=CP):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"a": np.r_[rng.standard_normal(cp), scale * rng.standard_normal(n - cp)]}
    )


def _assert_segments_tile(segments, df, labels):
    """Segments are contiguous, cover the series and agree with assign()."""
    assert segments[0].start_idx == 0
    assert segments[-1].end_idx == len(df)
    for prev, nxt in zip(segments, segments[1:]):
        assert prev.end_idx == nxt.start_idx
    for seg in segments:
        assert len(seg.df) == seg.end_idx - seg.start_idx
        assert seg.df.index[0] == df.index[seg.start_idx]
        assert (labels[seg.start_idx : seg.end_idx] == seg.regime).all()


def _near(bkps, target, tol):
    return any(abs(b - target) <= tol for b in bkps)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def test_extract_signal_mean_columns_and_interpolation():
    df = pd.DataFrame({"a": [0.0, 2.0, np.nan, 6.0], "b": [2.0, 4.0, np.nan, 8.0]})
    np.testing.assert_allclose(
        _extract_univariate_signal(df, None), [1.0, 3.0, 5.0, 7.0]
    )
    np.testing.assert_allclose(
        _extract_univariate_signal(df, ["b"]), [2.0, 4.0, 6.0, 8.0]
    )


def test_extract_signal_volatility_is_backfilled_rolling_std():
    df = _var_shift()
    sig = _extract_univariate_signal(df, None, "volatility", volatility_window=20)
    assert sig.shape == (T,)
    assert np.isfinite(sig).all()
    expected = df["a"].rolling(20, min_periods=10).std()
    np.testing.assert_allclose(sig[9:], expected[9:])
    assert (sig[:9] == sig[9]).all()
    # Rolling std tracks the planted variance jump.
    assert sig[200:].mean() > 3 * sig[:100].mean()


def test_extract_signal_rejects_unknown_signal():
    with pytest.raises(ValueError, match="Unknown signal"):
        _extract_univariate_signal(_var_shift(), None, "median")


def test_classify_breakpoints_no_breaks():
    sig = np.arange(10.0)
    (seg,) = _classify_breakpoints(sig, [], n_regimes=3)
    assert (seg.start, seg.end, seg.regime) == (0, 10, 0)
    assert seg.mean_value == pytest.approx(4.5)


def test_classify_breakpoints_quantile_labels_follow_level():
    sig = np.r_[np.zeros(10), 5 * np.ones(10), np.zeros(10), 5 * np.ones(10)]
    segs = _classify_breakpoints(sig, [30, 10, 20, 40], n_regimes=2)
    assert [(s.start, s.end) for s in segs] == [(0, 10), (10, 20), (20, 30), (30, 40)]
    assert [s.regime for s in segs] == [0, 1, 0, 1]
    assert [s.mean_value for s in segs] == [0.0, 5.0, 0.0, 5.0]


def test_classify_breakpoints_one_regime_per_segment_when_regimes_suffice():
    sig = np.r_[5 * np.ones(10), np.zeros(10)]
    segs = _classify_breakpoints(sig, [10], n_regimes=3)
    assert [s.regime for s in segs] == [0, 1]  # segment order, not level


def test_binary_segmentation_finds_planted_mean_shifts():
    rng = np.random.default_rng(0)
    sig = np.r_[np.zeros(100), 4 * np.ones(100), np.zeros(100)]
    sig = sig + 0.5 * rng.standard_normal(300)

    def t_test_p(left, right):
        return ttest_ind(left, right).pvalue

    bkps = _binary_segmentation(sig, t_test_p, 0.01, 20, 5, 10)
    assert bkps == sorted(bkps)
    assert _near(bkps, 100, 5) and _near(bkps, 200, 5)

    capped = _binary_segmentation(sig, t_test_p, 0.01, 20, 5, max_changepoints=1)
    assert len(capped) == 1
    assert _near(capped, 100, 5) or _near(capped, 200, 5)

    assert _binary_segmentation(sig, t_test_p, 0.01, 151, 5, 10) == []


def test_binary_segmentation_skips_failing_test_fn():
    def boom(left, right):
        raise RuntimeError("test failed")

    assert _binary_segmentation(np.arange(100.0), boom, 0.05, 10, 5, 10) == []


def test_merge_changepoints_clusters_and_spaces():
    assert _merge_changepoints([[], []], tolerance=5, min_segment=10) == []
    merged = _merge_changepoints([[100, 300], [104, 110], [305]], 10, 50)
    # {100, 104, 110} -> median 104; {300, 305} -> median 302.
    assert merged == [104, 302]
    # A merged CP closer than min_segment to the previous one is dropped.
    assert _merge_changepoints([[100], [130]], tolerance=5, min_segment=50) == [100]


def test_rolling_profiles_match_direct_computation():
    rng = np.random.default_rng(1)
    sig = rng.standard_normal(80)
    w = 10
    prof = _rolling_profiles(sig, w, ("mean", "variance", "slope", "kurtosis"))
    s = pd.Series(sig)
    np.testing.assert_allclose(prof["mean"], s.rolling(w).mean()[w - 1 :])
    np.testing.assert_allclose(
        prof["variance"], s.rolling(w).var(ddof=0)[w - 1 :], atol=1e-12
    )
    assert prof["kurtosis"].shape == (80 - w + 1,)

    ramp = 3.0 * np.arange(30.0)
    np.testing.assert_allclose(_rolling_profiles(ramp, 5, ("slope",))["slope"], 3.0)


def test_rolling_profiles_rejects_short_signal():
    with pytest.raises(ValueError, match="Signal length 5 < window 10"):
        _rolling_profiles(np.zeros(5), 10, ("mean",))


def test_sticky_viterbi_switch_cost_controls_switching():
    # Emissions alternate their preferred state every 5 steps.
    pref = np.repeat(np.arange(8) % 2, 5)
    log_ep = np.where(np.eye(2)[pref] == 1, 0.0, -2.0)
    log_tm = np.log(np.full((2, 2), 0.5))
    log_sp = np.log(np.full(2, 0.5))

    free = _sticky_viterbi(log_ep, log_tm, log_sp, 0.0)
    np.testing.assert_array_equal(free, pref)
    sticky = _sticky_viterbi(log_ep, log_tm, log_sp, switch_cost=100.0)
    assert np.unique(sticky).size == 1

    cost = _auto_switch_cost(log_ep, log_tm, log_sp, target_cps=2)
    path = _sticky_viterbi(log_ep, log_tm, log_sp, cost)
    assert 0 < int((np.diff(path) != 0).sum()) <= 2


# ---------------------------------------------------------------------------
# Levene / Bartlett
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["levene", "bartlett"])
def test_levene_detector_finds_variance_shift(method):
    df = _var_shift()
    det = LeveneRegimeDetector(n_regimes=2, method=method)
    bkps = det._find_breakpoints(det._get_signal(df))
    assert _near(bkps, CP, 10)

    labels = det.assign(df)
    assert labels.shape == (T,) and labels.dtype == np.int32
    segs = det.detect_segments(df)
    _assert_segments_tile(segs, df, labels)
    assert any(abs(s.start_idx - CP) <= 10 for s in segs)


def test_levene_detector_no_split_on_homogeneous_noise():
    df = pd.DataFrame({"a": np.random.default_rng(1).standard_normal(T)})
    det = LeveneRegimeDetector(alpha=0.01)
    assert det._find_breakpoints(det._get_signal(df)) == []
    assert (det.assign(df) == 0).all()


def test_levene_detector_respects_max_changepoints_and_columns():
    rng = np.random.default_rng(2)
    scales = np.repeat([1.0, 5.0, 1.0, 5.0], 100)
    df = pd.DataFrame(
        {"loud": scales * rng.standard_normal(400), "calm": rng.standard_normal(400)}
    )
    full = LeveneRegimeDetector(columns=["loud"])
    assert len(full._find_breakpoints(full._get_signal(df))) >= 3
    one = LeveneRegimeDetector(columns=["loud"], max_changepoints=1)
    assert len(one._find_breakpoints(one._get_signal(df))) == 1
    calm = LeveneRegimeDetector(columns=["calm"], alpha=0.01)
    assert calm._find_breakpoints(calm._get_signal(df)) == []


# ---------------------------------------------------------------------------
# Kolmogorov-Smirnov
# ---------------------------------------------------------------------------


def test_ks_detector_finds_mean_shift_and_labels_by_level():
    df = _mean_shift()
    det = KSRegimeDetector(n_regimes=2)
    bkps = det._find_breakpoints(det._get_signal(df))
    assert _near(bkps, CP, 5)

    labels = det.assign(df)
    assert labels[: CP - 10].max() == 0
    assert labels[CP + 10 :].min() == 1  # the high-mean segment
    _assert_segments_tile(det.detect_segments(df), df, labels)


def test_ks_detector_no_split_on_noise_and_volatility_signal():
    df = pd.DataFrame({"a": np.random.default_rng(1).standard_normal(T)})
    det = KSRegimeDetector(alpha=0.01)
    assert det._find_breakpoints(det._get_signal(df)) == []

    vol = KSRegimeDetector(n_regimes=2, signal="volatility", volatility_window=30)
    bkps = vol._find_breakpoints(vol._get_signal(_var_shift()))
    assert _near(bkps, CP, 30)


# ---------------------------------------------------------------------------
# CUSUM
# ---------------------------------------------------------------------------


def test_cusum_statistics():
    sig = np.array([1.0, 3.0, 1.0, 3.0])
    np.testing.assert_allclose(
        CusumRegimeDetector._cusum_mean(sig), [-1.0, 0.0, -1.0, 0.0]
    )
    np.testing.assert_allclose(CusumRegimeDetector._cusum_variance(sig), 0.0)


def test_cusum_mean_target_finds_mean_shift():
    df = _mean_shift()
    det = CusumRegimeDetector(n_regimes=2, target="mean")
    bkps = det._find_breakpoints(det._get_signal(df))
    assert _near(bkps, CP, 10)
    labels = det.assign(df)
    assert labels[: CP - 10].max() == 0 and labels[CP + 10 :].min() == 1
    _assert_segments_tile(det.detect_segments(df), df, labels)


def test_cusum_variance_target_finds_variance_shift():
    df = _var_shift()
    det = CusumRegimeDetector(n_regimes=2, target="variance")
    assert _near(det._find_breakpoints(det._get_signal(df)), CP, 10)


def test_cusum_both_target_merges_and_spaces():
    df = _mean_shift()
    det = CusumRegimeDetector(target="both", min_segment=30)
    sig = det._get_signal(df)
    bkps = det._find_breakpoints(sig)
    assert bkps == sorted(bkps)
    assert np.all(np.diff(bkps) >= 30)
    assert _near(bkps, CP, 10)
    # Every mean-CUSUM changepoint survives the merge, or was dropped only for
    # sitting within min_segment of a kept one.
    mean_only = CusumRegimeDetector(target="mean", min_segment=30)
    for b in mean_only._find_breakpoints(sig):
        assert _near(bkps, b, 29)


def test_cusum_degenerate_inputs_give_one_segment():
    det = CusumRegimeDetector(min_segment=30)
    # Shorter than 2 * min_segment.
    assert det._find_breakpoints(np.random.default_rng(0).standard_normal(50)) == []
    # Constant signal: the CUSUM is identically zero.
    assert det._find_breakpoints(np.ones(200)) == []
    # Smoothing window below 3 (n // 4 < 3) with a tiny min_segment.
    assert CusumRegimeDetector(min_segment=2)._find_breakpoints(np.arange(8.0)) == []
    # An explicit huge threshold rejects every candidate.
    big = CusumRegimeDetector(target="mean", threshold=1e9)
    df = _mean_shift()
    assert big._find_breakpoints(big._get_signal(df)) == []
    assert (big.assign(df) == 0).all()
    assert len(big.detect_segments(df)) == 1


def test_cusum_rejects_unknown_target():
    with pytest.raises(ValueError, match="Unknown target"):
        CusumRegimeDetector(target="skew").assign(_mean_shift())


# ---------------------------------------------------------------------------
# Ensemble (rolling profiles + PELT)
# ---------------------------------------------------------------------------


def test_ensemble_detector_finds_mean_shift():
    df = _mean_shift()
    det = EnsembleRegimeDetector(
        n_regimes=2, window=40, pen=20, min_segment=30, merge_tolerance=20
    )
    bkps = det._find_breakpoints(det._get_signal(df))
    assert _near(bkps, CP, 15)
    labels = det.assign(df)
    assert labels[: CP - 20].max() == 0 and labels[CP + 20 :].min() == 1
    _assert_segments_tile(det.detect_segments(df), df, labels)


def test_ensemble_detector_variance_feature_only():
    det = EnsembleRegimeDetector(
        n_regimes=2, features=("variance",), window=40, pen=20, min_segment=30
    )
    df = _var_shift()
    assert _near(det._find_breakpoints(det._get_signal(df)), CP, 15)


def test_ensemble_detector_constant_signal_and_short_series():
    det = EnsembleRegimeDetector(window=40)
    # Every rolling profile is flat, so no feature contributes changepoints.
    assert det._find_breakpoints(np.ones(200)) == []
    with pytest.raises(ValueError, match="< window"):
        det.assign(pd.DataFrame({"a": np.zeros(30)}))


# ---------------------------------------------------------------------------
# HMM
# ---------------------------------------------------------------------------


def test_hmm_detector_recovers_two_states():
    pytest.importorskip("hmmlearn")
    df = _mean_shift()
    df.iloc[20, 0] = np.nan  # interpolated, not propagated
    det = HMMRegimeDetector(n_regimes=2, min_segment=50, n_iter=50)
    labels = det.assign(df)
    assert labels.shape == (T,)
    assert set(np.unique(labels)) == {0, 1}
    truth = (np.arange(T) >= CP).astype(int)
    acc = max((labels == truth).mean(), (labels != truth).mean())
    assert acc > 0.97

    segs = det.detect_segments(df)
    _assert_segments_tile(segs, df, labels)
    assert any(abs(s.start_idx - CP) <= 5 for s in segs[1:])


@pytest.mark.parametrize("cov", ["full", "diag", "spherical", "tied"])
def test_hmm_detector_auto_selects_number_of_states(cov):
    pytest.importorskip("hmmlearn")
    det = HMMRegimeDetector(
        n_regimes="auto",
        max_regimes=3,
        covariance_type=cov,
        n_iter=30,
        min_segment=50,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k = det._select_n_regimes(det._get_data(_mean_shift()))
        labels = det.assign(_mean_shift())
    assert k in (2, 3)
    assert labels.shape == (T,)
    assert np.unique(labels).size <= k


def test_hmm_detector_requires_hmmlearn(monkeypatch):
    monkeypatch.setitem(sys.modules, "hmmlearn", None)
    monkeypatch.setitem(sys.modules, "hmmlearn.hmm", None)
    with pytest.raises(ImportError, match="pip install hmmlearn"):
        HMMRegimeDetector(n_regimes=2).assign(_mean_shift())


def test_levene_detector_rejects_unknown_method():
    # Regression: the ValueError was swallowed by binary segmentation, so a
    # typo silently returned a single regime.
    det = LeveneRegimeDetector(method="brown-forsythe")
    with pytest.raises(ValueError, match="Unknown method"):
        det.assign(_var_shift())
    with pytest.raises(ValueError, match="Unknown method"):
        det.detect_segments(_var_shift())


def test_hmm_detector_sticky_decoding_limits_switches():
    # Regression: _auto_switch_cost returned the midpoint of its final bracket,
    # which can sit on the too-many-switches side (5 switches for a target of 3
    # here).
    pytest.importorskip("hmmlearn")
    rng = np.random.default_rng(3)
    a = np.where(np.arange(T) % 20 < 10, 0.0, 3.0) + 0.5 * rng.standard_normal(T)
    df = pd.DataFrame({"a": a, "b": a + 0.3 * rng.standard_normal(T)})
    det = HMMRegimeDetector(n_regimes=2, min_segment=100, n_iter=50)
    labels = det.assign(df)
    # Without the penalty the 10-step blocks give ~29 switches.
    assert 0 < int((np.diff(labels) != 0).sum()) <= T // 100
