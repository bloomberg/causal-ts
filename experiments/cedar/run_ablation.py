# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""CEDAR ablation study across graph topologies.

Runs the incremental ablation (baseline SyPI-style -> full CEDAR) on
ER, scale-free, and small-world random graphs, extending the single-
benchmark ablation in the paper.

Usage:
    python experiments/cedar/run_ablation.py
    python experiments/cedar/run_ablation.py --d 30 --T 300 --seeds 20
"""

import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(description="CEDAR ablation across topologies")
    p.add_argument("--d", type=int, default=20)
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--max-lag", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.01)
    return p.parse_args()


def generate_random_scp(d, topology, max_lag, seed=42):
    """Generate a random sparse SCP with the given topology."""
    from causalts.synthetic_data.synthetic_datasets import topology_scp

    return topology_scp(
        topology=topology,
        seed=seed,
        n_vars=d,
        max_lag=max_lag,
        T=1000,
    )


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
    return dict(F1=f1, Prec=prec, Rec=rec)


def run_cedar_config(
    df,
    max_lag,
    alpha,
    lag_method="pearson",
    lag_pvalue_method="t_test",
    lag_alpha=0.05,
    multi_lag=False,
    prune=False,
    partial_dcor=False,
):
    """Run CEDAR with a specific ablation configuration."""
    from causalts.cedar import run_cedar
    from causalts.ci_tests.parcorr_gpu import ParCorrGPU

    ci = ParCorrGPU(np.ascontiguousarray(df.values))

    actual_lag_method = "partial_dcor" if partial_dcor else lag_method

    return run_cedar(
        df=df,
        ci_test=ci,
        max_lag=max_lag,
        alpha_cond1=alpha,
        alpha_cond2=alpha,
        lag_method=actual_lag_method,
        lag_pvalue_method=lag_pvalue_method,
        lag_alpha=lag_alpha,
        prune=prune,
        multi_lag=multi_lag,
        include_C=False,
        verbose=False,
    )


def main():
    args = parse_args()

    configs = [
        (
            "Baseline (SyPI-style)",
            dict(
                lag_method="pearson",
                lag_pvalue_method="t_test",
                lag_alpha=1.0,  # select top-1 lag unconditionally (SyPI uses Lasso coeff)
                multi_lag=False,
                prune=False,
                partial_dcor=False,
            ),
        ),
        (
            "+ U-centered dcor",
            dict(
                lag_method="dcor",
                lag_pvalue_method="t_test",
                lag_alpha=1.0,
                multi_lag=False,
                prune=False,
                partial_dcor=False,
            ),
        ),
        (
            "+ Analytic t-test",
            dict(
                lag_method="dcor",
                lag_pvalue_method="t_test",
                lag_alpha=0.05,
                multi_lag=False,
                prune=False,
                partial_dcor=False,
            ),
        ),
        (
            "+ Multi-lag",
            dict(
                lag_method="dcor",
                lag_pvalue_method="t_test",
                multi_lag=True,
                prune=False,
                partial_dcor=False,
            ),
        ),
        (
            "+ MCI pruning",
            dict(
                lag_method="dcor",
                lag_pvalue_method="t_test",
                multi_lag=True,
                prune=True,
                partial_dcor=False,
            ),
        ),
        (
            "+ AR(1)-res. dcor (CEDAR)",
            dict(
                lag_method="dcor",
                lag_pvalue_method="t_test",
                multi_lag=True,
                prune=True,
                partial_dcor=True,
            ),
        ),
    ]

    topologies = ["erdos_renyi", "scale_free", "small_world"]
    topo_short = {"erdos_renyi": "ER", "scale_free": "SF", "small_world": "SW"}

    print(
        f"CEDAR Ablation: d={args.d}, T={args.T}, "
        f"max_lag={args.max_lag}, seeds={args.seeds}"
    )

    # Header
    header = f"{'Configuration':<30}"
    for topo in topologies:
        header += f"  {'F1':>5} {'Prec':>5} {'Rec':>5}"
    print(f"\n{header}")
    topo_header = f"{'':30}"
    for topo in topologies:
        topo_header += f"  {topo_short[topo]:>17}"
    print(topo_header)
    print("-" * (30 + 19 * len(topologies)))

    for config_name, config_kwargs in configs:
        row = f"{config_name:<30}"
        for topo in topologies:
            f1s, precs, recs = [], [], []
            for seed in range(args.seeds):
                try:
                    scp = generate_random_scp(args.d, topo, args.max_lag, seed=seed)
                    df = scp["df"].iloc[: args.T]
                    gt = scp["ground_truth"]

                    res = run_cedar_config(
                        df, args.max_lag, args.alpha, **config_kwargs
                    )
                    m = evaluate(res.cg_tig, gt, args.d)
                    f1s.append(m["F1"])
                    precs.append(m["Prec"])
                    recs.append(m["Rec"])
                except Exception:
                    pass

            if f1s:
                row += (
                    f"  {np.mean(f1s):>5.3f}"
                    f" {np.mean(precs):>5.3f}"
                    f" {np.mean(recs):>5.3f}"
                )
            else:
                row += f"  {'---':>5} {'---':>5} {'---':>5}"

        print(row)

    print("\nDone.")


if __name__ == "__main__":
    main()
