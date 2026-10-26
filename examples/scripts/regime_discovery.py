# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: .venv-nbrun
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Regime-Conditional Causal Discovery
#
# Many real-world systems exhibit **variable-lag causal relationships** — the propagation delay between variables changes depending on the system's operating regime. Examples:
#
# - **River networks**: flood season has fast propagation (short lags), dry season has slow propagation (long lags)
# - **Financial markets**: volatility regimes change lead-lag relationships between assets
# - **Climate systems**: teleconnection delays vary by season
#
# Standard causal discovery algorithms assume a fixed lag structure and struggle with these systems. The `causalts.regime` module addresses this by:
#
# 1. **Detecting regimes** via changepoint detection (PELT, Levene, KS, CUSUM, Ensemble, or HMM)
# 2. **Running discovery per regime** with regime-appropriate `max_lag`
# 3. **Aggregating** per-regime graphs into a summary
#
# This notebook demonstrates the approach on synthetic data with known regime-dependent lags, comparing 8 different regime detectors across 4 types of regime change (mean shift, variance shift, shape shift, gradual drift).

# %%
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reproducibility ------------------------------------------------------------
# Every synthetic dataset below is generated from an explicit, isolated
# ``np.random.default_rng(seed)`` so results do NOT depend on cell execution
# order or cumulative global RNG state. SEED is the single dataset shown in the
# plots; SEEDS drives the multi-seed detector comparisons (reported as
# mean ± std) so the rankings are statistically robust rather than a single draw.
SEED = 42
N_SEEDS = 10
SEEDS = list(range(N_SEEDS))

# Optional dependency: hmmlearn powers HMMRegimeDetector. The notebook runs on a
# standard install without it — HMM detectors are simply skipped where absent.
try:
    import hmmlearn  # noqa: F401
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    print("hmmlearn not installed — HMM detectors will be skipped. "
          "Install with: pip install hmmlearn")


# %%
# Seeded synthetic-data generators ------------------------------------------
# Each returns (df, ground_truth) for a 4-variable chain A -> B -> C -> D whose
# causal lag switches between regimes. Passing a seed makes every dataset fully
# reproducible and independent of execution order. A burn-in is generated and
# discarded so the series reaches stationarity before the regime change.

_CHAIN_NAMES = ['A', 'B', 'C', 'D']


def _chain_ground_truth(d=4):
    gt = np.zeros((d, d), dtype=np.int8)
    gt[0, 1] = gt[1, 2] = gt[2, 3] = 1  # A->B, B->C, C->D
    return gt


def generate_mean_shift_data(seed, T_total=1000, burnin=200):
    """Regime 1 (t<500): mean~50, lag=1.  Regime 2: mean~10, lag=3."""
    rng = np.random.default_rng(seed)
    T_raw = T_total + burnin
    d = 4
    x = np.zeros((T_raw, d))
    for t in range(3, T_raw):
        if t < 500 + burnin:
            x[t, 0] = 50 + 0.4 * x[t-1, 0] + 5 * rng.standard_normal()
            x[t, 1] = 50 + 0.6 * x[t-1, 0] + 3 * rng.standard_normal()
            x[t, 2] = 50 + 0.6 * x[t-1, 1] + 3 * rng.standard_normal()
            x[t, 3] = 50 + 0.6 * x[t-1, 2] + 3 * rng.standard_normal()
        else:
            x[t, 0] = 10 + 0.4 * x[t-1, 0] + 2 * rng.standard_normal()
            x[t, 1] = 10 + 0.5 * x[t-3, 0] + 2 * rng.standard_normal()
            x[t, 2] = 10 + 0.5 * x[t-3, 1] + 2 * rng.standard_normal()
            x[t, 3] = 10 + 0.5 * x[t-3, 2] + 2 * rng.standard_normal()
    df = pd.DataFrame(x[burnin:], columns=_CHAIN_NAMES)
    return df, _chain_ground_truth(d)


def generate_variance_shift_data(seed, T_total=1000, burnin=200):
    """Same mean (30) throughout; noise amplitude 2 -> 8.  Lag 1 -> 3."""
    rng = np.random.default_rng(seed)
    T_raw = T_total + burnin
    d = 4
    x = np.zeros((T_raw, d))
    for t in range(3, T_raw):
        if t < 500 + burnin:
            x[t, 0] = 30 + 0.4 * x[t-1, 0] + 2 * rng.standard_normal()
            x[t, 1] = 30 + 0.6 * x[t-1, 0] + 2 * rng.standard_normal()
            x[t, 2] = 30 + 0.6 * x[t-1, 1] + 2 * rng.standard_normal()
            x[t, 3] = 30 + 0.6 * x[t-1, 2] + 2 * rng.standard_normal()
        else:
            x[t, 0] = 30 + 0.4 * x[t-1, 0] + 8 * rng.standard_normal()
            x[t, 1] = 30 + 0.5 * x[t-3, 0] + 8 * rng.standard_normal()
            x[t, 2] = 30 + 0.5 * x[t-3, 1] + 8 * rng.standard_normal()
            x[t, 3] = 30 + 0.5 * x[t-3, 2] + 8 * rng.standard_normal()
    df = pd.DataFrame(x[burnin:], columns=_CHAIN_NAMES)
    return df, _chain_ground_truth(d)


def generate_shape_shift_data(seed, T_total=1000, burnin=200):
    """Same mean/scale; noise switches Gaussian -> Laplace (heavier tails)."""
    rng = np.random.default_rng(seed)
    T_raw = T_total + burnin
    d = 4
    x = np.zeros((T_raw, d))
    for t in range(3, T_raw):
        if t < 500 + burnin:
            noise = rng.standard_normal(d) * 3
            x[t, 0] = 30 + 0.4 * x[t-1, 0] + noise[0]
            x[t, 1] = 30 + 0.6 * x[t-1, 0] + noise[1]
            x[t, 2] = 30 + 0.6 * x[t-1, 1] + noise[2]
            x[t, 3] = 30 + 0.6 * x[t-1, 2] + noise[3]
        else:
            noise = rng.laplace(0, 2.1, d)
            x[t, 0] = 30 + 0.4 * x[t-1, 0] + noise[0]
            x[t, 1] = 30 + 0.5 * x[t-3, 0] + noise[1]
            x[t, 2] = 30 + 0.5 * x[t-3, 1] + noise[2]
            x[t, 3] = 30 + 0.5 * x[t-3, 2] + noise[3]
    df = pd.DataFrame(x[burnin:], columns=_CHAIN_NAMES)
    return df, _chain_ground_truth(d)


def generate_drift_data(seed, T_total=1000, burnin=200):
    """Mean drifts 10 -> 50 over [0, 700), then stable.  Lag 1 -> 3."""
    rng = np.random.default_rng(seed)
    T_raw = T_total + burnin
    d = 4
    x = np.zeros((T_raw, d))
    for t in range(3, T_raw):
        if t < burnin:
            base, lag, coef = 10, 1, 0.6
        elif t < 700 + burnin:
            base, lag, coef = 10 + 40 * ((t - burnin) / 700), 1, 0.6
        else:
            base, lag, coef = 50, 3, 0.5
        x[t, 0] = base + 0.3 * x[t-1, 0] + 3 * rng.standard_normal()
        x[t, 1] = base + coef * x[t-lag, 0] + 3 * rng.standard_normal()
        x[t, 2] = base + coef * x[t-lag, 1] + 3 * rng.standard_normal()
        x[t, 3] = base + coef * x[t-lag, 2] + 3 * rng.standard_normal()
    df = pd.DataFrame(x[burnin:], columns=_CHAIN_NAMES)
    return df, _chain_ground_truth(d)



# %% [markdown]
# ## 1. Synthetic Data: Chain Graph with Regime-Dependent Lags
#
# We generate a 4-variable chain: **A → B → C → D**, where the causal lag depends on the regime:
#
# - **Regime 1** (t < 500): high activity, fast propagation — lag = 1
# - **Regime 2** (t ≥ 500): low activity, slow propagation — lag = 3
#
# The mean level also shifts between regimes (50 vs 10), mimicking flood/dry conditions.

# %%
names = _CHAIN_NAMES
d = len(names)

# Reproducible regardless of execution order: the generator seeds its own RNG.
df, gt = generate_mean_shift_data(SEED)
data = df.values
T = len(df)
BURNIN_ORIG = 200

print(f"T={T} (after discarding {BURNIN_ORIG} burn-in samples), d={d}, true edges={int(gt.sum())}")
print(f"Regime 1 (t<500): A->B->C->D at lag 1, mean~50")
print(f"Regime 2 (t>=500): A->B->C->D at lag 3, mean~10")


# %%
fig, ax = plt.subplots(figsize=(12, 3))
for i, name in enumerate(names):
    ax.plot(data[:, i], alpha=0.7, label=name)
ax.axvline(500, color='red', linestyle='--', label='Regime change')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend(loc='upper right')
ax.set_title('Synthetic regime-switching time series')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Baseline: CEDAR with Fixed max_lag
#
# Running CEDAR on the full data with a single `max_lag` — it cannot capture both regimes simultaneously.

# %%
from causalts.cedar import run_cedar
from causalts.ci_tests.parcorr_gpu import ParCorrGPU

def evaluate(est, gt):
    est = est.copy()
    np.fill_diagonal(est, 0)
    tp = int(((est == 1) & (gt == 1)).sum())
    fp = int(((est == 1) & (gt == 0)).sum())
    fn = int(((est == 0) & (gt == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "Prec": prec, "Rec": rec, "F1": f1}

print("Baseline CEDAR (fixed max_lag, full data):")
print(f"{'max_lag':>8} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>6} {'Rec':>6} {'F1':>6}")
for ml in [1, 2, 3, 4, 5]:
    ci = ParCorrGPU(np.ascontiguousarray(df.values))
    result = run_cedar(df=df, ci_test=ci, max_lag=ml,
                       alpha_cond1=0.01, alpha_cond2=0.01,
                       lag_method='dcor', prune=True, verbose=False)
    est = result.cg_tig[:d, :d, 1:].any(axis=2).astype(np.int8)
    np.fill_diagonal(est, 0)
    m = evaluate(est, gt)
    print(f"{ml:>8} {m['TP']:>4} {m['FP']:>4} {m['FN']:>4} {m['Prec']:>6.3f} {m['Rec']:>6.3f} {m['F1']:>6.3f}")

# %% [markdown]
# The baseline fails regardless of `max_lag` — the mixed-regime data confuses lag selection and CI testing. Precision drops as `max_lag` grows, because the extra lags pick up spurious edges, and F1 stays well below 1.
#
# ## 3. PELT Regime Detection
#
# PELT finds changepoints in the signal's mean level, directly giving us regime boundaries.

# %%
from causalts.regime import PeltRegimeDetector

detector = PeltRegimeDetector(n_regimes=2, min_size=100, pen=5000, signal='mean')
segments = detector.detect_segments(df)

print(f"PELT found {len(segments)} segments:")
for seg in segments:
    seg_mean = data[seg.start_idx:seg.end_idx].mean()
    print(f"  t=[{seg.start_idx}, {seg.end_idx}) — regime {seg.regime}, "
          f"mean={seg_mean:.1f}, length={seg.end_idx - seg.start_idx}")

# %% [markdown]
# PELT correctly identifies the regime boundary at t=500.
#
# ## 4. Regime-Conditional Discovery with CEDAR
#
# Now we run CEDAR **per regime** with appropriate `max_lag` for each:
# - High-flow regime: `max_lag=2` (short lags)
# - Low-flow regime: `max_lag=4` (longer lags)

# %%
from causalts.regime import run_regime_discovery, RegimeConfig, RegimeParams

def cedar_fn(sub_df, max_lag=3, alpha=0.01, **kw):
    dat = np.ascontiguousarray(sub_df.values)
    ci = ParCorrGPU(dat)
    return run_cedar(df=sub_df, ci_test=ci, max_lag=max_lag,
                     alpha_cond1=alpha, alpha_cond2=alpha,
                     lag_method='dcor', prune=True, verbose=False)

config = RegimeConfig(
    n_regimes=2,
    regime_params=[
        RegimeParams(max_lag=4, alpha=0.01),  # low-flow (slow, long lags)
        RegimeParams(max_lag=2, alpha=0.01),  # high-flow (fast, short lags)
    ],
    aggregation='or',
    segment_threshold=0.3,
    regime_names=['low-flow', 'high-flow'],
)

result_cedar = run_regime_discovery(df, discovery_fn=cedar_fn, config=config, detector=detector)

est = result_cedar.cg_tig[:d, :d, 1:].any(axis=2).astype(np.int8)
np.fill_diagonal(est, 0)
m = evaluate(est, gt)
print(f"PELT + CEDAR: TP={m['TP']} FP={m['FP']} FN={m['FN']} "
      f"Prec={m['Prec']:.3f} Rec={m['Rec']:.3f} F1={m['F1']:.3f}")
print()
print("Per-regime edges:")
for r, g in result_cedar.regime_graphs.items():
    edges = [(names[i], names[j], tau)
             for i in range(d) for j in range(d) for tau in range(g.shape[2])
             if g[i, j, tau] and i != j]
    print(f"  {result_cedar.regime_names[r]}: {edges}")

# %% [markdown]
# ## 5. Regime-Conditional Discovery with CDNOTS+
#
# We can plug in any discovery algorithm. Let's also try CDNOTS+ (constraint-based with MCI pruning).

# %%
from causalts.cdnots.phase3_utils import run_cdnots_plus

def cdnots_plus_fn(sub_df, max_lag=3, alpha=0.01, **kw):
    dat = np.ascontiguousarray(sub_df.values)
    ci = ParCorrGPU(dat)
    return run_cdnots_plus(df=sub_df, indep_test=ci, num_lags=max_lag,
                           alpha=alpha, include_C=False, verbose=False)

result_cdnots = run_regime_discovery(df, discovery_fn=cdnots_plus_fn, config=config, detector=detector)

est2 = result_cdnots.cg_tig[:d, :d, 1:].any(axis=2).astype(np.int8)
np.fill_diagonal(est2, 0)
m2 = evaluate(est2, gt)
print(f"PELT + CDNOTS+: TP={m2['TP']} FP={m2['FP']} FN={m2['FN']} "
      f"Prec={m2['Prec']:.3f} Rec={m2['Rec']:.3f} F1={m2['F1']:.3f}")
print()
print("Per-regime edges:")
for r, g in result_cdnots.regime_graphs.items():
    edges = [(names[i], names[j], tau)
             for i in range(d) for j in range(d) for tau in range(g.shape[2])
             if g[i, j, tau] and i != j]
    print(f"  {result_cdnots.regime_names[r]}: {edges}")

# %% [markdown]
# ## 6. Comparison with RPCMCI (tigramite)
#
# RPCMCI (Saggioro et al., Chaos 2020) is an iterative method that **learns** regimes from prediction residuals. Unlike our PELT approach which detects regimes from the signal level, RPCMCI can discover structural regime changes even when marginal distributions don't shift. However, it requires `tigramite` + `ortools` and is significantly slower.
#
# Let's compare on this data where the mean shift makes regimes observable.

# %%
from causalts.tigramite_discovery import run_rpcmci

try:
    G_rpcmci, info_rpcmci = run_rpcmci(
        df, tau_max=4, num_regimes=2,
        max_transitions=50,
        pc_alpha=0.2, alpha_level=0.01,
        tau_min=1, n_jobs=1, max_anneal=10,
        seed=42, verbosity=0,
    )

    est3 = G_rpcmci[:d, :d, 1:].any(axis=2).astype(np.int8)
    np.fill_diagonal(est3, 0)
    m3 = evaluate(est3, gt)
    print(f"RPCMCI: TP={m3['TP']} FP={m3['FP']} FN={m3['FN']} "
          f"Prec={m3['Prec']:.3f} Rec={m3['Rec']:.3f} F1={m3['F1']:.3f}")
    print(f"  Error-free annealings: {info_rpcmci['error_free_annealings']}")

    labels = info_rpcmci['regime_labels']
    match_a = ((labels[:500] == 0).sum() + (labels[500:] == 1).sum()) / T
    match_b = ((labels[:500] == 1).sum() + (labels[500:] == 0).sum()) / T
    print(f"  Regime assignment accuracy: {max(match_a, match_b)*100:.1f}%")
    print()
    print("Per-regime edges:")
    for r, g in info_rpcmci['regime_graphs'].items():
        edges = [(names[i], names[j], tau)
                 for i in range(d) for j in range(d) for tau in range(g.shape[2])
                 if g[i, j, tau] and i != j]
        n_samples = (labels == r).sum()
        print(f"  Regime {r} ({n_samples} samples): {edges}")
except Exception as e:
    print(f"RPCMCI failed: {e}")
    m3 = {"TP": 0, "FP": 0, "FN": 3, "Prec": 0, "Rec": 0, "F1": 0}

# %% [markdown]
# ## 7. Results Comparison

# %%
print(f"{'Method':<25} {'TP':>3} {'FP':>3} {'FN':>3} {'Prec':>6} {'Rec':>6} {'F1':>6}")
print("-" * 60)

# Best baseline
ci = ParCorrGPU(np.ascontiguousarray(df.values))
res_bl = run_cedar(df=df, ci_test=ci, max_lag=4,
                   alpha_cond1=0.01, alpha_cond2=0.01,
                   lag_method='dcor', prune=True, verbose=False)
est_bl = res_bl.cg_tig[:d, :d, 1:].any(axis=2).astype(np.int8)
np.fill_diagonal(est_bl, 0)
m_bl = evaluate(est_bl, gt)
print(f"{'Baseline CEDAR (lag=4)':<25} {m_bl['TP']:>3} {m_bl['FP']:>3} {m_bl['FN']:>3} "
      f"{m_bl['Prec']:>6.3f} {m_bl['Rec']:>6.3f} {m_bl['F1']:>6.3f}")
print(f"{'PELT + CEDAR':<25} {m['TP']:>3} {m['FP']:>3} {m['FN']:>3} "
      f"{m['Prec']:>6.3f} {m['Rec']:>6.3f} {m['F1']:>6.3f}")
print(f"{'PELT + CDNOTS+':<25} {m2['TP']:>3} {m2['FP']:>3} {m2['FN']:>3} "
      f"{m2['Prec']:>6.3f} {m2['Rec']:>6.3f} {m2['F1']:>6.3f}")
print(f"{'RPCMCI (tigramite)':<25} {m3['TP']:>3} {m3['FP']:>3} {m3['FN']:>3} "
      f"{m3['Prec']:>6.3f} {m3['Rec']:>6.3f} {m3['F1']:>6.3f}")

# %% [markdown]
# The PELT approach works perfectly for this mean-shift case — both CEDAR and CDNOTS+ recover the true graph with F1=1.0 per regime. RPCMCI struggles with regime assignment accuracy. Next, we explore how different types of regime changes require different detection methods.

# %% [markdown]
# ## 8. Changepoint Detector Comparison
#
# PELT with L2 cost on the **mean** signal detects mean-level shifts — perfect for the example above. Pointed at a different signal (`signal="volatility"`) it segments the rolling standard deviation instead, so it is not limited to mean shifts. Real-world regime changes can also manifest as:
#
# - **Variance/volatility shifts** — mean stays constant, noise amplitude changes
# - **Distribution shape shifts** — same mean and variance, but different tail behavior
# - **Gradual drift** — slow trend rather than abrupt jump
#
# The `causalts.regime` module provides several changepoint detectors, each sensitive to different types of change:
#
# | Detector | Approach | Detects |
# |----------|----------|---------|
# | `PeltRegimeDetector` | L2 cost on mean or rolling-std signal | Mean shifts (`signal="mean"`); variance shifts (`signal="volatility"`) |
# | `LeveneRegimeDetector` | Levene's test (binary segmentation) | Variance shifts |
# | `KSRegimeDetector` | Kolmogorov-Smirnov (binary segmentation) | Any distributional change |
# | `CusumRegimeDetector` | CUSUM control chart | Mean and/or variance drift |
# | `EnsembleRegimeDetector` | PELT on rolling stats (mean, variance, slope) | Multiple change types |
# | `HMMRegimeDetector` | Gaussian HMM (multivariate) | All types; learns regimes jointly (requires `hmmlearn`) |
#
# We test all detectors on three synthetic datasets, each with a different type of regime change. To avoid drawing conclusions from a single lucky (or unlucky) seed, each comparison reports F1 as **mean ± standard deviation across `N_SEEDS` independently generated datasets**.

# %%
# Multi-seed evaluation harness ---------------------------------------------
# A single seed gives a misleading ranking (e.g. a detector can look near-worst
# on one draw yet best on average). We therefore evaluate every detector on
# N_SEEDS independent datasets and report F1 as mean ± std.
from collections import OrderedDict

from causalts.regime import (
    PeltRegimeDetector, LeveneRegimeDetector, KSRegimeDetector,
    CusumRegimeDetector, EnsembleRegimeDetector, HMMRegimeDetector,
)

regime_colors_2 = {0: '#2196F3', 1: '#F44336', 2: '#4CAF50', 3: '#FF9800'}


def cedar_fn(sub_df, max_lag=3, alpha=0.01, **kw):
    """Discovery callback used for every regime-conditional run below."""
    dat = np.ascontiguousarray(sub_df.values)
    ci = ParCorrGPU(dat)
    return run_cedar(df=sub_df, ci_test=ci, max_lag=max_lag,
                     alpha_cond1=alpha, alpha_cond2=alpha,
                     lag_method='dcor', prune=True, verbose=False)


def _edges_to_binary(result, d):
    est = result.cg_tig[:d, :d, 1:].any(axis=2).astype(np.int8)
    np.fill_diagonal(est, 0)
    return est


def compare_detectors_multiseed(generate_fn, detector_factory, config, seeds,
                                baseline_max_lag=4):
    """Evaluate baseline CEDAR + each detector over multiple seeds.

    ``detector_factory`` must return a *fresh* ``{name: detector}`` mapping on
    each call — detectors can be stateful after fitting, and each seed needs a
    clean instance. Returns ``{method_name: np.ndarray of F1 per seed}``.
    """
    f1_by_method = OrderedDict()
    f1_by_method['Baseline CEDAR'] = []
    for name in detector_factory():
        f1_by_method[name + ' + CEDAR'] = []

    for seed in seeds:
        df_s, gt_s = generate_fn(seed)
        d_s = gt_s.shape[0]

        ci = ParCorrGPU(np.ascontiguousarray(df_s.values))
        res_bl = run_cedar(df=df_s, ci_test=ci, max_lag=baseline_max_lag,
                           alpha_cond1=0.01, alpha_cond2=0.01,
                           lag_method='dcor', prune=True, verbose=False)
        f1_by_method['Baseline CEDAR'].append(
            evaluate(_edges_to_binary(res_bl, d_s), gt_s)['F1'])

        for name, det in detector_factory().items():
            try:
                res = run_regime_discovery(df_s, discovery_fn=cedar_fn,
                                           config=config, detector=det, verbose=False)
                f1 = evaluate(_edges_to_binary(res, d_s), gt_s)['F1']
            except Exception as e:
                print(f"seed {seed}: {name} failed — {type(e).__name__}: {e}")
                f1 = np.nan
            f1_by_method[name + ' + CEDAR'].append(f1)

    return {k: np.asarray(v, dtype=float) for k, v in f1_by_method.items()}


def print_multiseed_table(f1_by_method, n_seeds, title=None):
    if title:
        print(title)
    print(f"{'Method':<25} {'F1 mean':>8} {'±std':>7} {'n':>4}   (of {n_seeds} seeds)")
    print("-" * 52)
    for name, arr in f1_by_method.items():
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            print(f"{name:<25} {'skipped':>8}")
        else:
            print(f"{name:<25} {valid.mean():>8.3f} {valid.std():>7.3f} {valid.size:>4}")



# %% [markdown]
# ### 8.1 Variance Shift
#
# Same chain graph **A → B → C → D**, but now the regime change is a **variance shift** — the mean stays at 30 throughout, but noise amplitude quadruples in regime 2. This is invisible to PELT with the default `signal="mean"` (its L2 cost tracks the mean, which is unchanged), but detectable by Levene, HMM, `PeltRegimeDetector(signal="volatility")`, and others.

# %%
# Variance-shift data: same mean, different noise amplitude per regime.
# Shown for a single representative seed; the comparison table below averages
# over N_SEEDS independent draws.
names_v = _CHAIN_NAMES
d_v = len(names_v)
df_v, gt_v = generate_variance_shift_data(SEED)
data_v = df_v.values
T_v = len(df_v)

fig, ax = plt.subplots(figsize=(12, 3))
for i, name in enumerate(names_v):
    ax.plot(data_v[:, i], alpha=0.7, label=name, linewidth=0.5)
ax.axvline(500, color='red', linestyle='--', label='Regime change')
ax.set_xlabel('Time'); ax.set_ylabel('Value')
ax.legend(loc='upper right')
ax.set_title('Variance-shift regime change (same mean=30, noise 2→8, burn-in discarded)')
plt.tight_layout()
plt.show()

print(f"T={T_v} (seed={SEED}, burn-in discarded)")
print(f"Regime 1 std: {data_v[:500].std(axis=0).mean():.1f}")
print(f"Regime 2 std: {data_v[500:].std(axis=0).mean():.1f}")


# %%
# Detector factory for the variance-shift case. HMM detectors are included only
# when hmmlearn is installed, so the notebook runs on a standard install.
def make_variance_detectors():
    dets = OrderedDict([
        ("PELT", PeltRegimeDetector(n_regimes=2, min_size=50, pen=500, signal='mean')),
        ("Levene", LeveneRegimeDetector(n_regimes=2, alpha=0.01, min_segment=30, step=5)),
        ("KS", KSRegimeDetector(n_regimes=2, alpha=0.01, min_segment=30, step=5)),
        ("CUSUM (var)", CusumRegimeDetector(n_regimes=2, target='variance', min_segment=30)),
        ("CUSUM (mean)", CusumRegimeDetector(n_regimes=2, target='mean', min_segment=30)),
        ("Ensemble", EnsembleRegimeDetector(n_regimes=2, window=100, pen=200, min_segment=50)),
    ])
    if HMMLEARN_AVAILABLE:
        dets["HMM"] = HMMRegimeDetector(n_regimes=2, min_segment=30)
        dets["HMM (auto)"] = HMMRegimeDetector(n_regimes='auto', max_regimes=5, min_segment=30)
    return dets


# Visualise the segmentation each detector produces on the representative seed.
detectors_v = make_variance_detectors()
signal_v = df_v.mean(axis=1)

n_det = len(detectors_v)
n_cols = 2
n_rows = (n_det + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 2.5 * n_rows), sharex=True)
axes_flat = axes.flatten()

for idx, (det_name, det) in enumerate(detectors_v.items()):
    ax = axes_flat[idx]
    labels = det.assign(df_v)
    cps = np.where(np.diff(labels) != 0)[0] + 1
    for regime in sorted(np.unique(labels)):
        mask = labels == regime
        ax.scatter(np.arange(T_v)[mask], signal_v.values[mask],
                   c=regime_colors_2.get(regime, '#999'), s=0.5, alpha=0.6,
                   label=f'R{regime} (n={mask.sum()})')
    for cp in cps:
        ax.axvline(cp, color='black', linewidth=2, alpha=0.7)
    ax.axvline(500, color='red', linestyle='--', linewidth=2, alpha=0.8, label='True CP')
    ax.set_title(f'{det_name}  ({len(cps)} CPs)', fontweight='bold', fontsize=10)
    ax.legend(loc='upper right', fontsize=7, markerscale=8, ncol=2)

for idx in range(n_det, len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig.suptitle(f'Variance-Shift Data — Detector Comparison (seed={SEED})', fontsize=13, y=1.01)
fig.tight_layout()
plt.show()


# %%
# Regime-conditional CEDAR with each detector, averaged over N_SEEDS datasets.
config_v = RegimeConfig(
    n_regimes=2,
    regime_params=[
        RegimeParams(max_lag=4, alpha=0.01),
        RegimeParams(max_lag=2, alpha=0.01),
    ],
    aggregation='or',
    segment_threshold=0.3,
    regime_names=['low-noise', 'high-noise'],
)

f1_v = compare_detectors_multiseed(
    generate_variance_shift_data, make_variance_detectors, config_v, SEEDS)
print_multiseed_table(f1_v, N_SEEDS, title="Variance-shift data — F1 over seeds")


# %% [markdown]
# ### 8.2 Distribution Shape Shift
#
# The regime change is a **distribution shape shift** — mean and variance stay similar, but the noise switches from Gaussian to Laplace (heavier tails). KS (which compares full CDFs), HMM (which learns full covariance), and Ensemble (via its kurtosis feature) are designed to pick this up; the multi-seed table below shows how each detector actually fares.

# %%
# Shape-shift data: Gaussian -> Laplace noise (same mean, similar scale).
# Shown for a single representative seed; the table below averages over N_SEEDS.
names_s = _CHAIN_NAMES
d_s = len(names_s)
df_s, gt_s = generate_shape_shift_data(SEED)
data_s = df_s.values
T_s = len(df_s)

fig, axes = plt.subplots(1, 2, figsize=(14, 3))
axes[0].hist(data_s[:500].ravel(), bins=50, alpha=0.6, density=True, label='Regime 1 (Gaussian)')
axes[0].hist(data_s[500:].ravel(), bins=50, alpha=0.6, density=True, label='Regime 2 (Laplace)')
axes[0].legend()
axes[0].set_title('Distribution comparison')

axes[1].plot(data_s[:, 0], alpha=0.7, linewidth=0.5)
axes[1].axvline(500, color='red', linestyle='--')
axes[1].set_title('Variable A (hard to see the change visually)')
axes[1].set_xlabel('Time')

fig.suptitle('Shape-shift: Gaussian → Laplace (burn-in discarded)', fontsize=12)
fig.tight_layout()
plt.show()

from scipy.stats import kurtosis
print(f"T={T_s} (seed={SEED}, burn-in discarded)")
print(f"Regime 1 kurtosis: {kurtosis(data_s[:500].ravel()):.2f} (Gaussian ≈ 0)")
print(f"Regime 2 kurtosis: {kurtosis(data_s[500:].ravel()):.2f} (Laplace ≈ 3)")


# %%
# Detector factory for the shape-shift case (Levene/KS at alpha=0.05; Ensemble
# gains a kurtosis feature). HMM detectors included only when hmmlearn is present.
def make_shape_detectors():
    dets = OrderedDict([
        ("PELT", PeltRegimeDetector(n_regimes=2, min_size=50, pen=500, signal='mean')),
        ("Levene", LeveneRegimeDetector(n_regimes=2, alpha=0.05, min_segment=30, step=5)),
        ("KS", KSRegimeDetector(n_regimes=2, alpha=0.05, min_segment=30, step=5)),
        ("CUSUM (var)", CusumRegimeDetector(n_regimes=2, target='variance', min_segment=30)),
        ("CUSUM (mean)", CusumRegimeDetector(n_regimes=2, target='mean', min_segment=30)),
        ("Ensemble", EnsembleRegimeDetector(
            n_regimes=2, features=('mean', 'variance', 'slope', 'kurtosis'),
            window=100, pen=200, min_segment=50)),
    ])
    if HMMLEARN_AVAILABLE:
        dets["HMM"] = HMMRegimeDetector(n_regimes=2, min_segment=30)
        dets["HMM (auto)"] = HMMRegimeDetector(n_regimes='auto', max_regimes=5, min_segment=30)
    return dets


# Visualise the segmentation each detector produces on the representative seed.
detectors_s = make_shape_detectors()
signal_s = df_s.mean(axis=1)
n_det = len(detectors_s)
n_cols = 2
n_rows = (n_det + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 2.5 * n_rows), sharex=True)
axes_flat = axes.flatten()

for idx, (det_name, det) in enumerate(detectors_s.items()):
    ax = axes_flat[idx]
    labels = det.assign(df_s)
    cps = np.where(np.diff(labels) != 0)[0] + 1
    for regime in sorted(np.unique(labels)):
        mask = labels == regime
        ax.scatter(np.arange(T_s)[mask], signal_s.values[mask],
                   c=regime_colors_2.get(regime, '#999'), s=0.5, alpha=0.6,
                   label=f'R{regime} (n={mask.sum()})')
    for cp in cps:
        ax.axvline(cp, color='black', linewidth=2, alpha=0.7)
    ax.axvline(500, color='red', linestyle='--', linewidth=2, alpha=0.8, label='True CP')
    ax.set_title(f'{det_name}  ({len(cps)} CPs)', fontweight='bold', fontsize=10)
    ax.legend(loc='upper right', fontsize=7, markerscale=8, ncol=2)

for idx in range(n_det, len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig.suptitle(f'Shape-Shift Data — Detector Comparison (seed={SEED})', fontsize=13, y=1.01)
fig.tight_layout()
plt.show()

# Regime-conditional CEDAR, averaged over N_SEEDS datasets.
config_s = RegimeConfig(
    n_regimes=2,
    regime_params=[
        RegimeParams(max_lag=4, alpha=0.01),
        RegimeParams(max_lag=2, alpha=0.01),
    ],
    aggregation='or', segment_threshold=0.3,
    regime_names=['gaussian', 'laplace'],
)

f1_s = compare_detectors_multiseed(
    generate_shape_shift_data, make_shape_detectors, config_s, SEEDS)
print()
print_multiseed_table(f1_s, N_SEEDS, title="Shape-shift data — F1 over seeds")


# %% [markdown]
# ### 8.3 Gradual Drift
#
# A **linear trend** in the mean, then stabilization. This is challenging because there's no abrupt changepoint — the system drifts continuously. A detector can split in the **middle** of the drift rather than where it **ends**. HMM (multivariate) and Ensemble (via rolling slope) are designed with this in mind; the multi-seed table below shows how each detector actually fares.

# %%
# Gradual drift data: linear trend then stabilization.
# Shown for a single representative seed; the table below averages over N_SEEDS.
names_d = _CHAIN_NAMES
d_d = len(names_d)
df_d, gt_d = generate_drift_data(SEED)
data_d = df_d.values
T_d = len(df_d)

fig, ax = plt.subplots(figsize=(12, 3))
for i, name in enumerate(names_d):
    ax.plot(data_d[:, i], alpha=0.7, label=name, linewidth=0.5)
ax.axvline(700, color='red', linestyle='--', label='Drift ends (t=700)')
ax.set_xlabel('Time'); ax.set_ylabel('Value')
ax.legend(loc='upper left')
ax.set_title('Gradual drift: mean 10→50 over [0,700), then stable (burn-in discarded)')
plt.tight_layout()
plt.show()

print(f"T={T_d} (seed={SEED}, burn-in discarded)")


# %%
# Detector factory for the gradual-drift case (CUSUM "both" instead of "var").
# HMM detectors included only when hmmlearn is present.
def make_drift_detectors():
    dets = OrderedDict([
        ("PELT", PeltRegimeDetector(n_regimes=2, min_size=50, pen=500, signal='mean')),
        ("Levene", LeveneRegimeDetector(n_regimes=2, alpha=0.01, min_segment=30, step=5)),
        ("KS", KSRegimeDetector(n_regimes=2, alpha=0.01, min_segment=30, step=5)),
        ("CUSUM (mean)", CusumRegimeDetector(n_regimes=2, target='mean', min_segment=30)),
        ("CUSUM (both)", CusumRegimeDetector(n_regimes=2, target='both', min_segment=30)),
        ("Ensemble", EnsembleRegimeDetector(n_regimes=2, window=100, pen=200, min_segment=50)),
    ])
    if HMMLEARN_AVAILABLE:
        dets["HMM"] = HMMRegimeDetector(n_regimes=2, min_segment=30)
        dets["HMM (auto)"] = HMMRegimeDetector(n_regimes='auto', max_regimes=5, min_segment=30)
    return dets


# Visualise the segmentation each detector produces on the representative seed.
detectors_d = make_drift_detectors()
signal_d = df_d.mean(axis=1)
n_det = len(detectors_d)
n_cols = 2
n_rows = (n_det + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 2.5 * n_rows), sharex=True)
axes_flat = axes.flatten()

for idx, (det_name, det) in enumerate(detectors_d.items()):
    ax = axes_flat[idx]
    labels = det.assign(df_d)
    cps = np.where(np.diff(labels) != 0)[0] + 1
    for regime in sorted(np.unique(labels)):
        mask = labels == regime
        ax.scatter(np.arange(T_d)[mask], signal_d.values[mask],
                   c=regime_colors_2.get(regime, '#999'), s=0.5, alpha=0.6,
                   label=f'R{regime} (n={mask.sum()})')
    for cp in cps:
        ax.axvline(cp, color='black', linewidth=2, alpha=0.7)
    ax.axvline(700, color='red', linestyle='--', linewidth=2, alpha=0.8, label='True CP')
    ax.set_title(f'{det_name}  ({len(cps)} CPs)', fontweight='bold', fontsize=10)
    ax.legend(loc='upper left', fontsize=7, markerscale=8, ncol=2)

for idx in range(n_det, len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig.suptitle(f'Gradual Drift Data — Detector Comparison (seed={SEED})', fontsize=13, y=1.01)
fig.tight_layout()
plt.show()

# Regime-conditional CEDAR, averaged over N_SEEDS datasets.
config_d = RegimeConfig(
    n_regimes=2,
    regime_params=[
        RegimeParams(max_lag=4, alpha=0.01),
        RegimeParams(max_lag=2, alpha=0.01),
    ],
    aggregation='or', segment_threshold=0.3,
    regime_names=['drifting', 'stable'],
)

f1_d = compare_detectors_multiseed(
    generate_drift_data, make_drift_detectors, config_d, SEEDS)
print()
print_multiseed_table(f1_d, N_SEEDS, title="Gradual-drift data — F1 over seeds")


# %% [markdown]
# ## 9. Summary: Which Detector for Which Regime Change?
#
# Recommendations below follow the multi-seed tables in §8.1–8.3 (F1 mean over 10
# seeds). Detectors with the same mean are listed together; check the ± column in §8
# before reading much into small gaps.
#
# | Regime change type | Top detectors (mean F1) | Notes |
# |----|----|----|
# | **Mean shift** | `PeltRegimeDetector` | Optimal L2 segmentation on the level signal (§1–5 example; not part of the multi-seed comparison) |
# | **Variance/volatility shift** | `EnsembleRegimeDetector` (1.000); `LeveneRegimeDetector`, `KSRegimeDetector`, `HMMRegimeDetector` (0.986) | Every detector except the baseline is near ceiling; PELT with the default `signal="mean"` still reaches 0.967 |
# | **Distribution shape change** | `PeltRegimeDetector`, `CusumRegimeDetector(target="mean")`, `HMMRegimeDetector` (0.986) | KS (0.907) and Ensemble (0.927) trail |
# | **Gradual drift** | `PeltRegimeDetector` (0.854) | Highest mean; next are `HMMRegimeDetector(n_regimes="auto")` (0.790, but ±0.29) and KS (0.776) |
# | **Unknown / mixed** | `PeltRegimeDetector`, then `KSRegimeDetector` | Best worst case across the three tables (PELT ≥ 0.854, KS ≥ 0.776); HMM drops to 0.724 on drift, and `n_regimes="auto"` is unstable (std ≈ 0.3 on shape shift and drift) |
#
# | Method | Approach | Strengths | Limitations |
# |--------|----------|-----------|-------------|
# | **PELT** | L2 optimal segmentation on a chosen signal | Fast, exact; `signal="mean"` for level shifts, `signal="volatility"` for variance shifts | Each run targets one signal (mean *or* volatility), not both at once |
# | **Levene** | Variance binary segmentation | Targets volatility directly | Misses mean/shape changes |
# | **KS** | Distribution binary segmentation | Non-parametric, general | Can over-segment |
# | **CUSUM** | Cumulative sum control chart | Detects sustained drifts | Threshold-sensitive |
# | **Ensemble** | PELT on rolling stats | Multi-type | Window size tuning |
# | **HMM** | Gaussian Hidden Markov Model | Multivariate, can auto-select k | Requires `hmmlearn`; auto k is unstable across seeds |
# | **RPCMCI** | Iterative regime learning | Latent structural changes | Very slow, `ortools` |
#
# **Practical guidance:**
# - **Start with PELT** — best or tied-best on shape shift and gradual drift, and 0.967 on variance shift even with the default `signal="mean"`. For volatility regimes, run it with `signal="volatility"` so it segments the rolling standard deviation instead of the mean.
# - **HMM with a fixed `n_regimes`** — tied-best on variance and shape shifts, weaker on gradual drift (0.724). Requires `hmmlearn` (`pip install hmmlearn`). Prefer a fixed `n_regimes` over `"auto"`, whose BIC selection varies widely from seed to seed.
# - **Levene** for financial/volatility data where mean stays constant
# - **KS** as a non-parametric fallback when you don't know the change type — never the outright best here, but never far behind
# - **Ensemble** when you want a dependency-free detector — top on variance shift, mid-pack on shape shift and drift

# %% [markdown]
# ## 10. Real-World: Elbe River Network
#
# The Elbe River benchmark (12 stations, chain graph with 11 true edges) exhibits exactly the variable-lag problem: flood season propagation is fast (lag 1-2 at 3h resolution), dry season is slow (lag 5+).
#
# Here we apply regime-conditional discovery with PELT (the best detector for this mean-shift-dominated dataset). Data must be downloaded first (see `examples/elbe_river.ipynb`).

# %%
import pickle
from pathlib import Path
from causalts.imputation import var_em_impute

DATA_DIR = Path("/tmp/causalrivers/product")
if not DATA_DIR.exists():
    print("CausalRivers data not found. Run examples/elbe_river.ipynb first to download.")
else:
    # Load Elbe main branch (12 stations)
    ELBE_STATIONS = [166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177]
    full_G = pickle.load(open(DATA_DIR / "rivers_east_germany.p", "rb"))
    ts_raw = pd.read_csv(DATA_DIR / "rivers_ts_east_germany.csv", index_col=0, parse_dates=True)
    ts_res = ts_raw.resample("3h").mean()
    avail = [n for n in ELBE_STATIONS if str(n) in ts_res.columns]
    df_elbe = ts_res[ts_res.index < "2023-01-01"][[str(n) for n in avail]]

    # Ground truth
    basin = sorted(avail)
    d_elbe = len(basin)
    G_basin = full_G.subgraph(basin).copy()
    node_idx = {n: i for i, n in enumerate(basin)}
    gt_elbe = np.zeros((d_elbe, d_elbe), dtype=np.int8)
    for u, v in G_basin.edges():
        if u in node_idx and v in node_idx:
            gt_elbe[node_idx[u], node_idx[v]] = 1
    df_elbe.columns = list(range(d_elbe))
    var_names_elbe = [f"S{n}" for n in basin]

    print(f"Elbe: d={d_elbe} stations, T={len(df_elbe)}, true edges={int(gt_elbe.sum())}")
    print(f"Missing: {df_elbe.isna().sum().sum()}/{df_elbe.size} ({100*df_elbe.isna().sum().sum()/df_elbe.size:.1f}%)")

# %%
import warnings
warnings.filterwarnings("ignore")

if DATA_DIR.exists():
    # Interpolate for discovery (CI tests need complete data)
    df_elbe_clean = df_elbe.interpolate().ffill().bfill()

    def cedar_elbe(sub_df, max_lag=5, alpha=0.01, **kw):
        dat = np.ascontiguousarray(sub_df.values)
        ci = ParCorrGPU(dat)
        return run_cedar(df=sub_df, ci_test=ci, max_lag=max_lag,
                         alpha_cond1=alpha, alpha_cond2=alpha,
                         lag_method='dcor', prune=True,
                         lag_filter='ar_relative', lag_filter_frac=0.97,
                         verbose=False)

    def cdnots_plus_elbe(sub_df, max_lag=5, alpha=0.01, **kw):
        dat = np.ascontiguousarray(sub_df.values)
        ci = ParCorrGPU(dat)
        return run_cdnots_plus(df=sub_df, indep_test=ci, num_lags=max_lag,
                               alpha=alpha, include_C=False, verbose=False)

    # PELT on raw data (flood peaks define regimes — don't smooth them)
    det_elbe = PeltRegimeDetector(n_regimes=3, min_size=300, pen=5000, signal='mean')
    config_elbe = RegimeConfig(
        n_regimes=3,
        regime_params=[
            RegimeParams(max_lag=8, alpha=0.01),   # low-flow
            RegimeParams(max_lag=5, alpha=0.01),   # normal
            RegimeParams(max_lag=3, alpha=0.01),   # high-flow
        ],
        aggregation='or',
        segment_threshold=0.25,
        regime_names=['low-flow', 'normal', 'high-flow'],
    )

    # Use raw df for PELT segmentation (preserves regime signal)
    segments_elbe = det_elbe.detect_segments(df_elbe_clean)
    regime_colors = {0: '#2196F3', 1: '#4CAF50', 2: '#F44336'}
    regime_labels_map = {0: 'low-flow', 1: 'normal', 2: 'high-flow'}

    # Visualize with dates on x-axis
    dates = ts_res[ts_res.index < "2023-01-01"].index[:len(df_elbe_clean)]
    fig, ax = plt.subplots(figsize=(14, 3))
    station_col = 0
    ax.plot(dates, df_elbe_clean.iloc[:, station_col].values,
            color='black', linewidth=0.5, alpha=0.8)
    for seg in segments_elbe:
        ax.axvspan(dates[seg.start_idx], dates[min(seg.end_idx - 1, len(dates) - 1)],
                   alpha=0.15, color=regime_colors[seg.regime])
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=regime_colors[r], alpha=0.3, label=regime_labels_map[r])
                      for r in sorted(regime_colors)]
    ax.legend(handles=legend_patches, loc='upper right')
    ax.set_xlabel('Date')
    ax.set_ylabel('Discharge')
    ax.set_title(f'Station {var_names_elbe[station_col]} — PELT regime segmentation')
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

    # Baseline CEDAR (single run on full interpolated data)
    ci_bl = ParCorrGPU(np.ascontiguousarray(df_elbe_clean.values))
    res_bl = run_cedar(df=df_elbe_clean, ci_test=ci_bl, max_lag=5,
                       alpha_cond1=0.01, alpha_cond2=0.01,
                       lag_method='dcor', prune=True, verbose=False)
    est_bl = res_bl.cg_tig[:d_elbe, :d_elbe, 1:].any(axis=2).astype(np.int8)
    np.fill_diagonal(est_bl, 0)
    m_bl = evaluate(est_bl, gt_elbe)

    # PELT + CEDAR
    res_cedar_elbe = run_regime_discovery(df_elbe_clean, discovery_fn=cedar_elbe,
                                          config=config_elbe, detector=det_elbe)
    est_ce = res_cedar_elbe.cg_tig[:d_elbe, :d_elbe, 1:].any(axis=2).astype(np.int8)
    np.fill_diagonal(est_ce, 0)
    m_ce = evaluate(est_ce, gt_elbe)

    # PELT + CDNOTS+
    res_cdnots_elbe = run_regime_discovery(df_elbe_clean, discovery_fn=cdnots_plus_elbe,
                                           config=config_elbe, detector=det_elbe)
    est_cd = res_cdnots_elbe.cg_tig[:d_elbe, :d_elbe, 1:].any(axis=2).astype(np.int8)
    np.fill_diagonal(est_cd, 0)
    m_cd = evaluate(est_cd, gt_elbe)

    print(f"{'Method':<25} {'TP':>3} {'FP':>3} {'FN':>3} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("-" * 60)
    print(f"{'Baseline CEDAR':<25} {m_bl['TP']:>3} {m_bl['FP']:>3} {m_bl['FN']:>3} "
          f"{m_bl['Prec']:>6.3f} {m_bl['Rec']:>6.3f} {m_bl['F1']:>6.3f}")
    print(f"{'PELT + CEDAR':<25} {m_ce['TP']:>3} {m_ce['FP']:>3} {m_ce['FN']:>3} "
          f"{m_ce['Prec']:>6.3f} {m_ce['Rec']:>6.3f} {m_ce['F1']:>6.3f}")
    print(f"{'PELT + CDNOTS+':<25} {m_cd['TP']:>3} {m_cd['FP']:>3} {m_cd['FN']:>3} "
          f"{m_cd['Prec']:>6.3f} {m_cd['Rec']:>6.3f} {m_cd['F1']:>6.3f}")
    print()
    print("Per-regime (CEDAR):")
    for r, g in res_cedar_elbe.regime_graphs.items():
        g2d = g[:d_elbe, :d_elbe, 1:].any(axis=2).astype(np.int8)
        np.fill_diagonal(g2d, 0)
        mr = evaluate(g2d, gt_elbe)
        print(f"  {res_cedar_elbe.regime_names[r]}: TP={mr['TP']} FP={mr['FP']} F1={mr['F1']:.3f}")
