# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""SYPI-all-targets baseline comparison.

Compares CEDAR against a faithful SYPI-all-targets adaptation:
run CEDAR with Lasso lag selection (matching Mastakouri et al. ICML 2021),
no MCI pruning, single lag per pair, and equal alpha_cond1=alpha_cond2=0.05.

This establishes the novelty of CEDAR's components over the closest ancestor.

Table rows:
  SYPI-all-targets : Lasso lag selection, no pruning, single lag, alpha=0.05
  CEDAR-no-MCI     : partial_dcor lags, no pruning, multi-lag, alpha=0.01
  Full CEDAR       : partial_dcor lags, MCI pruning, multi-lag, alpha=0.01

Run on:
  - Random SCP ER/SF, d=20, T=100/200/500
  - Scale-free d=40, T=200
  - 11-node cascade

Usage:
    python experiments/cedar/run_sypi_all_targets.py
    python experiments/cedar/run_sypi_all_targets.py --d 30 --seeds 20
"""

import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(description="SYPI-all-targets vs CEDAR comparison")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--max-lag", type=int, default=3)
    return p.parse_args()


def evaluate(est_3d, gt_3d):
    """Lag-specific F1 on cross-variable edges."""
    d = min(est_3d.shape[0], gt_3d.shape[0])
    L = min(est_3d.shape[2], gt_3d.shape[2])
    est = est_3d[:d, :d, :L].copy()
    gt = gt_3d[:d, :d, :L].copy()
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


def run_config(df, max_lag, cfg):
    """Run CEDAR with a specific configuration."""
    from causalts.cedar import run_cedar
    from causalts.ci_tests.parcorr_gpu import ParCorrGPU

    ci = ParCorrGPU(np.ascontiguousarray(df.values))
    return run_cedar(
        df=df,
        ci_test=ci,
        max_lag=max_lag,
        alpha_cond1=cfg["alpha"],
        alpha_cond2=cfg["alpha"],
        lag_method=cfg["lag_method"],
        lag_pvalue_method="t_test",
        lag_alpha=cfg.get("lag_alpha", 0.05),
        prune=cfg["prune"],
        multi_lag=cfg["multi_lag"],
        include_C=False,
        verbose=False,
    )


CONFIGS = [
    (
        "SYPI-all-targets",
        dict(
            lag_method="lasso",
            lag_alpha=1.0,  # always select max-coef lag (SyPI: argmax of Lasso coefs)
            multi_lag=False,
            prune=False,
            alpha=0.05,  # SyPI paper: threshold1=threshold2, swept 0.01-0.12, ROC uses equal ratio
        ),
    ),
    (
        "CEDAR-no-MCI",
        dict(
            lag_method="partial_dcor",
            lag_alpha=0.05,
            multi_lag=True,
            prune=False,
            alpha=0.01,
        ),
    ),
    (
        "Full CEDAR",
        dict(
            lag_method="partial_dcor",
            lag_alpha=0.05,
            multi_lag=True,
            prune=True,
            alpha=0.01,
        ),
    ),
]


def run_benchmark(label, scp_fn, seeds, max_lag):
    """Run all configs on a benchmark, return results dict."""
    print(f"\n{'=' * 65}")
    print(label)
    print(f"{'=' * 65}")
    header = f"{'Method':<22}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}"
    print(header)
    print("-" * 44)

    results = {}
    for name, cfg in CONFIGS:
        f1s, precs, recs = [], [], []
        for seed in range(seeds):
            try:
                scp = scp_fn(seed)
                T = scp.get("T_eval", len(scp["df"]))
                df = scp["df"].iloc[:T]
                gt = scp["ground_truth"]
                res = run_config(df, max_lag, cfg)
                m = evaluate(res.cg_tig, gt)
                f1s.append(m["F1"])
                precs.append(m["Prec"])
                recs.append(m["Rec"])
            except Exception:
                pass
        if f1s:
            r = dict(F1=np.mean(f1s), Prec=np.mean(precs), Rec=np.mean(recs))
        else:
            r = dict(F1=float("nan"), Prec=float("nan"), Rec=float("nan"))
        results[name] = r
        print(f"{name:<22}  {r['F1']:>6.3f}  {r['Prec']:>6.3f}  {r['Rec']:>6.3f}")
    return results


def main():
    args = parse_args()

    from causalts.synthetic_data.synthetic_datasets import ex3, topology_scp

    # -----------------------------------------------------------------------
    # 1. Random SCP: ER and SF, d=20, T=100/200/500
    # -----------------------------------------------------------------------
    for topo in ["erdos_renyi", "scale_free"]:
        topo_short = "ER" if topo == "erdos_renyi" else "SF"
        for T in [100, 200, 500]:

            def make_scp(seed, _topo=topo, _T=T):
                scp = topology_scp(
                    topology=_topo, seed=seed, n_vars=20, max_lag=args.max_lag, T=1000
                )
                scp["T_eval"] = _T
                return scp

            run_benchmark(
                f"Random SCP {topo_short}, d=20, T={T}",
                make_scp,
                args.seeds,
                args.max_lag,
            )

    # -----------------------------------------------------------------------
    # 2. Scale-free d=40, T=200
    # -----------------------------------------------------------------------
    def make_sf40(seed):
        scp = topology_scp(
            topology="scale_free", seed=seed, n_vars=40, max_lag=args.max_lag, T=1000
        )
        scp["T_eval"] = 200
        return scp

    run_benchmark("Scale-free SF, d=40, T=200", make_sf40, args.seeds, args.max_lag)

    # -----------------------------------------------------------------------
    # 3. 11-node cascade (ex3), T=200
    # -----------------------------------------------------------------------
    def make_cascade(seed):
        scp = ex3(seed=seed)
        scp["T_eval"] = 200
        return scp

    run_benchmark(
        "11-node cascade (ex3), T=200", make_cascade, args.seeds, args.max_lag
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
