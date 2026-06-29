# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Stress tests for CEDAR under assumption violations.

Tests CEDAR's robustness when its key assumptions are violated, using
graphs large enough to expose genuine failure modes:

  1. Multiple A7 bypass paths (d=15): 5 simultaneous lag-mismatched mediators
  2. Dense AR(2) network (d=15): all variables AR(2), chain+hub structure
  3. Mixed violations (d=20): AR(2) + multi-lag + A7 bypasses combined

All tests use full CEDAR (partial_dcor lag selection) vs CDNOTS+.

Usage:
    python experiments/cedar/run_stress_tests.py
"""

import warnings

import numpy as np
import pandas as pd  # noqa: F401

warnings.filterwarnings("ignore")

N_SEEDS = 10
T_DEFAULT = 300
ALPHA = 0.01


def generate_scp(edges, d, T, ar_coeffs=None, seed=42):
    """Generate time series from a structural causal process.

    Parameters
    ----------
    edges : list of (cause, effect, lag, weight)
    d : int, number of variables
    T : int, number of time steps
    ar_coeffs : dict mapping var -> list of AR coefficients, default AR(1)=0.5
    seed : int
    """
    rng = np.random.default_rng(seed)
    if ar_coeffs is None:
        ar_coeffs = {i: [0.5] for i in range(d)}

    max_lag = max(
        max(lag for _, _, lag, _ in edges), max(len(c) for c in ar_coeffs.values())
    )
    data = np.zeros((T + max_lag, d))

    for t in range(max_lag, T + max_lag):
        for j in range(d):
            val = rng.standard_normal()
            for p, coeff in enumerate(ar_coeffs.get(j, [0.5])):
                val += coeff * data[t - p - 1, j]
            for c, e, lag, w in edges:
                if e == j:
                    val += w * data[t - lag, c]
            data[t, j] = val

    data = data[max_lag:]
    gt = np.zeros((d, d, max_lag + 1), dtype=np.int8)
    for c, e, lag, _ in edges:
        gt[c, e, lag] = 1

    df = pd.DataFrame(data, columns=list(range(d)))
    return df, gt


def evaluate(est_3d, gt_3d, d):
    """Lag-specific F1 on cross-variable edges."""
    est = est_3d[:d, :d, :].copy()
    gt = gt_3d[:d, :d, :].copy()
    for i in range(d):
        est[i, i, :] = 0
        gt[i, i, :] = 0
    tp = int(((est == 1) & (gt == 1)).sum())
    fp = int(((est == 1) & (gt == 0)).sum())
    fn = int(((est == 0) & (gt == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return dict(
        TP=tp, FP=fp, FN=fn, Prec=round(prec, 3), Rec=round(rec, 3), F1=round(f1, 3)
    )


def run_test(label, edges, d, max_lag, T=T_DEFAULT, ar_coeffs=None):
    """Run a single stress test across seeds."""
    global _last_edges, _last_ar

    from causalts.cdnots.phase3_utils import run_cdnots_plus
    from causalts.cedar import run_cedar
    from causalts.ci_tests.parcorr_gpu import ParCorrGPU

    if ar_coeffs is None:
        ar_coeffs = {i: [0.5] for i in range(d)}

    _last_edges = edges
    _last_ar = ar_coeffs

    gt = np.zeros((d, d, max_lag + 1), dtype=np.int8)
    for c, e, lag, _ in edges:
        gt[c, e, lag] = 1

    n_true = int(gt.sum())
    print(f"\n{'=' * 65}")
    print(f"{label} (d={d}, T={T}, max_lag={max_lag}, true_edges={n_true})")
    print(f"{'=' * 65}")
    print(
        f"{'Method':<12} {'TP':>4} {'FP':>4} {'FN':>4} "
        f"{'Prec':>6} {'Rec':>6} {'F1':>6}"
    )
    print("-" * 50)

    methods = ["CEDAR", "CDNOTS+"]

    for name in methods:
        all_metrics = []
        for seed in range(N_SEEDS):
            df, _ = generate_scp(edges, d, T, ar_coeffs, seed=seed)
            ci = ParCorrGPU(np.ascontiguousarray(df.values))
            try:
                if name == "CEDAR":
                    res = run_cedar(
                        df=df,
                        ci_test=ci,
                        max_lag=max_lag,
                        alpha_cond1=ALPHA,
                        alpha_cond2=ALPHA,
                        lag_method="partial_dcor",
                        prune=True,
                        include_C=False,
                        verbose=False,
                    )
                else:
                    res = run_cdnots_plus(
                        df=df,
                        indep_test=ci,
                        num_lags=max_lag,
                        alpha=ALPHA,
                        include_C=False,
                        verbose=False,
                    )
                cg = res.cg_tig if hasattr(res, "cg_tig") else res["cg_tig"]
                m = evaluate(cg, gt, d)
                all_metrics.append(m)
            except Exception:
                pass

        if all_metrics:
            avg = {
                k: np.mean([m[k] for m in all_metrics])
                for k in ["TP", "FP", "FN", "Prec", "Rec", "F1"]
            }
            print(
                f"{name:<12} {avg['TP']:>4.1f} {avg['FP']:>4.1f} {avg['FN']:>4.1f} "
                f"{avg['Prec']:>6.3f} {avg['Rec']:>6.3f} {avg['F1']:>6.3f}"
            )


def main():
    # --- Test 1: Multiple simultaneous A7 bypass paths (d=15) ---
    # 5 direct edges each with a competing lag-mismatched mediator.
    # X_i -> X_{i+5} at lag 2 (direct) and X_i -> X_{i+10} at lag 3,
    # X_{i+10} -> X_{i+5} at lag 1 (bypass).
    # The bypass path arrives at lag 3+1=4 but CEDAR's S places X_{i+10}
    # at t-w_{i+10}-1 = t-1-1 = t-2, which does not block the lag-4 bypass.
    a7_edges = []
    for i in range(5):
        src, tgt, med = i, i + 5, i + 10
        a7_edges += [
            (src, tgt, 2, 0.5),  # direct: lag 2
            (src, med, 3, 0.45),  # mediator source
            (med, tgt, 1, 0.45),  # mediator leg (bypass at lag 4)
        ]
    run_test(
        label="A7 bypass x5 (d=15, violates A7)",
        edges=a7_edges,
        d=15,
        max_lag=4,
    )

    # --- Test 2: Dense AR(2) network (d=15) ---
    # All variables have AR(2) dynamics. Chain plus hub with 20+ edges.
    ar2_edges = [(i, i + 1, 1, 0.45) for i in range(9)]  # chain 0..9
    ar2_edges += [(0, i, 2, 0.35) for i in range(10, 15)]  # hub -> tail
    ar2_edges += [
        (10, 11, 1, 0.3),
        (11, 12, 1, 0.3),
        (12, 13, 2, 0.3),
        (13, 14, 1, 0.3),
    ]
    ar2_ar = {i: [0.55, -0.25] for i in range(15)}  # all AR(2)
    run_test(
        label="Dense AR(2) network (d=15, violates A3)",
        edges=ar2_edges,
        d=15,
        max_lag=3,
        ar_coeffs=ar2_ar,
    )

    # --- Test 3: Mixed violations (d=20) ---
    # Combines: AR(2) on 8 variables, 4 multi-lag pairs, 3 A7 bypasses.
    mixed_edges = []
    # Chain backbone
    for i in range(14):
        mixed_edges.append((i, i + 1, 1, 0.4))
    # Multi-lag pairs (violates A5): extra lag on 4 edges
    for i in [0, 3, 6, 9]:
        mixed_edges.append((i, i + 1, 3, 0.3))
    # A7 bypasses: 3 patterns
    for i, (src, tgt, med) in enumerate([(15, 16, 17), (17, 18, 19), (0, 5, 15)]):
        mixed_edges += [
            (src, tgt, 2, 0.45),
            (src, med, 3, 0.4),
            (med, tgt, 1, 0.4),
        ]
    mixed_ar = {i: ([0.5, -0.2] if i % 3 == 0 else [0.5]) for i in range(20)}
    run_test(
        label="Mixed violations (d=20, A3+A5+A7)",
        edges=mixed_edges,
        d=20,
        max_lag=4,
        ar_coeffs=mixed_ar,
    )


if __name__ == "__main__":
    main()
