# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

#!/usr/bin/env python3
"""CEDAR Paper — Reproducible Baseline Results.

Runs cedar, pcmci+, cdnots, and cdnots+ on the scaling benchmark
(ER/SF/SW topologies, d=10-50, T=100-1000) and saves results to CSV.

Usage:
    python run_paper_baselines.py                          # scaling experiment (full)
    python run_paper_baselines.py --algos cedar --topos sf # cedar only, scale-free
    python run_paper_baselines.py --seeds 42 --d-values 30 --t-values 300  # sanity check
    python run_paper_baselines.py --clean                  # wipe CSV and re-run
    python run_paper_baselines.py ablation                 # ablation on cascade network
    python run_paper_baselines.py named                    # named benchmarks (appendix)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from causalts import run_cdnots, run_cdnots_plus
from causalts.cedar import run_cedar
from causalts.ci_tests import ParCorrGPU
from causalts.synthetic_data.synthetic_datasets import (
    SCPGraphGenerator,
    confounded_trends,
    load_dataset,
    scale_free,
    small_world,
)
from causalts.tigramite_discovery import run_pcmciplus
from causalts.utils import evaluate_graph

CSV_PATH = Path(__file__).parent / "paper_results.csv"
CSV_COLS = [
    "algo",
    "topo",
    "d",
    "T",
    "seed",
    "F1",
    "Precision",
    "TPR",
    "SHD",
    "F1_pair",
    "Precision_pair",
    "TPR_pair",
    "time",
]
MIXED_FUNCS = ("lin", "lin", "nl1", "nl2")
MAX_LAG = 3
ALPHA = 0.01


def generate_data(topo, seed, d, T_max):
    if topo == "sf":
        return scale_free(
            seed=seed,
            n_vars=d,
            T=T_max,
            max_lag=MAX_LAG,
            dependency_funcs=MIXED_FUNCS,
            max_tries=10000,
        )
    elif topo == "sw":
        return small_world(
            seed=seed,
            n_vars=d,
            T=T_max,
            max_lag=MAX_LAG,
            dependency_funcs=MIXED_FUNCS,
            max_tries=10000,
        )
    elif topo == "er":
        gen = SCPGraphGenerator(
            n_vars=d, max_lag=MAX_LAG, density="sparse", dependency_funcs=MIXED_FUNCS
        )
        return gen.sample(seed=seed, T=T_max, max_tries=10000)
    raise ValueError(f"Unknown topology: {topo}")


def run_one(algo, df_data, gt, d, ci):
    t0 = time.perf_counter()

    if algo == "cedar":
        res = run_cedar(
            df_data,
            ci,
            MAX_LAG,
            alpha_cond1=ALPHA,
            alpha_cond2=ALPHA,
            filter_children=False,
            prune=True,
            multi_lag=True,
            multi_lag_keep="first",
            include_autoreg=True,
            lag_method="partial_dcor",
        )
        G_hat = res.cg_tig[:d, :d, :]
    elif algo == "pcmci+":
        G_hat, _ = run_pcmciplus(df_data, tau_max=MAX_LAG, ci_test=ci, pc_alpha=ALPHA)
    elif algo == "cdnots":
        res = run_cdnots(
            df_data,
            indep_test=ci,
            num_lags=MAX_LAG,
            alpha=ALPHA,
            include_C=False,
            stable=True,
            verbose=False,
        )
        G_hat = res.cg_tig[:d, :d, :]
    elif algo == "cdnots+":
        res = run_cdnots_plus(
            df_data,
            indep_test=ci,
            num_lags=MAX_LAG,
            alpha=ALPHA,
            include_C=False,
            verbose=False,
        )
        G_hat = res.cg_tig[:d, :d, :]
    else:
        raise ValueError(f"Unknown algo: {algo}")

    elapsed = time.perf_counter() - t0
    m = evaluate_graph(G_hat, gt, exclude_self_loops=True)
    m["time"] = elapsed
    return m


def main():
    p = argparse.ArgumentParser(description="CEDAR paper baseline results")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 62)))
    p.add_argument("--topos", nargs="+", default=["er", "sf", "sw"])
    p.add_argument("--d-values", nargs="+", type=int, default=[10, 20, 30, 40, 50])
    p.add_argument(
        "--t-values", nargs="+", type=int, default=[100, 200, 300, 500, 1000]
    )
    p.add_argument(
        "--algos", nargs="+", default=["cedar", "pcmci+", "cdnots", "cdnots+"]
    )
    p.add_argument("--clean", action="store_true", help="Wipe CSV before running")
    args = p.parse_args()

    if args.clean and CSV_PATH.exists():
        CSV_PATH.unlink()

    done = set()
    if CSV_PATH.exists():
        for _, r in pd.read_csv(CSV_PATH).iterrows():
            done.add((r["algo"], r["topo"], int(r["d"]), int(r["T"]), int(r["seed"])))

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy = np.zeros((2, 2))

    total = (
        len(args.topos)
        * len(args.d_values)
        * len(args.t_values)
        * len(args.seeds)
        * len(args.algos)
    )
    completed = 0

    for topo in args.topos:
        for d in args.d_values:
            print(f"\n{'='*50}")
            print(f"topo={topo}  d={d}")
            print(f"{'='*50}")

            T_max = max(args.t_values)
            cache = {}
            for seed in args.seeds:
                try:
                    cache[seed] = generate_data(topo, seed, d, T_max)
                except Exception as e:
                    print(f"  gen seed={seed} FAIL: {e}")

            for T in args.t_values:
                for seed in args.seeds:
                    b = cache.get(seed)
                    if b is None:
                        completed += len(args.algos)
                        continue

                    df_data = b["df"].iloc[:T]
                    gt = b["ground_truth"]

                    for algo in args.algos:
                        completed += 1
                        key = (algo, topo, d, T, seed)
                        if key in done:
                            continue

                        try:
                            ci = ParCorrGPU(dummy, device=device)
                            m = run_one(algo, df_data, gt, d, ci)

                            print(
                                f"  [{completed}/{total}] {algo:8s} T={T:4d} "
                                f"s={seed} F1={m['F1']:.3f} "
                                f"P={m['Precision']:.3f} "
                                f"R={m['TPR']:.3f} t={m['time']:.1f}s"
                            )

                            hdr = not CSV_PATH.exists()
                            with open(CSV_PATH, "a", newline="") as f:
                                w = csv.DictWriter(f, fieldnames=CSV_COLS)
                                if hdr:
                                    w.writeheader()
                                w.writerow(
                                    {
                                        "algo": algo,
                                        "topo": topo,
                                        "d": d,
                                        "T": T,
                                        "seed": seed,
                                        "F1": m["F1"],
                                        "Precision": m["Precision"],
                                        "TPR": m["TPR"],
                                        "SHD": m["SHD"],
                                        "F1_pair": m["F1_pair"],
                                        "Precision_pair": m["Precision_pair"],
                                        "TPR_pair": m["TPR_pair"],
                                        "time": m["time"],
                                    }
                                )
                            done.add(key)

                        except Exception as e:
                            print(
                                f"  [{completed}/{total}] {algo:8s} T={T:4d} "
                                f"s={seed} FAIL: {e}"
                            )

    print(f"\nDone. Results in {CSV_PATH}")
    _print_summary(args)


def _print_summary(args):
    if not CSV_PATH.exists():
        return
    df = pd.read_csv(CSV_PATH)

    print(f"\n{'='*70}")
    print("SUMMARY — mean F1 by algo, topo, d, T")
    print(f"{'='*70}")

    for topo in sorted(df["topo"].unique()):
        print(f"\n  === {topo.upper()} ===")
        tdf = df[df["topo"] == topo]

        for d in sorted(tdf["d"].unique()):
            ddf = tdf[tdf["d"] == d]
            T_vals = sorted(ddf["T"].unique())
            print(f"\n  d={int(d)}")
            hdr = f"  {'algo':10s}"
            for T in T_vals:
                hdr += f"  {'T=' + str(int(T)):>8s}"
            print(hdr)
            print(f"  {'-' * 10}" + "-" * 10 * len(T_vals))

            for algo in ["cedar", "pcmci+", "cdnots+", "cdnots"]:
                adf = ddf[ddf["algo"] == algo]
                if len(adf) == 0:
                    continue
                line = f"  {algo:10s}"
                for T in T_vals:
                    v = adf[adf["T"] == T]["F1"].mean()
                    line += f"  {v:8.3f}" if not np.isnan(v) else f"  {'n/a':>8s}"
                print(line)


def run_ablation(seeds=None, T_values=None):
    """Incremental ablation on the 11-node cascade network (ex3)."""
    if seeds is None:
        seeds = list(range(42, 52))
    if T_values is None:
        T_values = [100, 200, 500]

    csv_path = Path(__file__).parent / "ablation_results.csv"
    cols = ["variant", "T", "seed", "F1", "Precision", "TPR", "SHD"]
    max_lag = 3

    configs = [
        (
            "baseline",
            dict(
                lag_method="pearson",
                lag_pvalue_method="permutation",
                multi_lag=False,
                filter_children=False,
                prune=False,
            ),
        ),
        (
            "+unbiased_dcor",
            dict(
                lag_method="dcor",
                lag_pvalue_method="permutation",
                multi_lag=False,
                filter_children=False,
                prune=False,
            ),
        ),
        (
            "+analytic_ttest",
            dict(
                lag_method="dcor",
                lag_pvalue_method="t_test",
                multi_lag=False,
                filter_children=False,
                prune=False,
            ),
        ),
        (
            "+multi_lag",
            dict(
                lag_method="dcor",
                lag_pvalue_method="t_test",
                multi_lag=True,
                multi_lag_keep="first",
                filter_children=False,
                prune=False,
            ),
        ),
        (
            "+mci_pruning",
            dict(
                lag_method="dcor",
                lag_pvalue_method="t_test",
                multi_lag=True,
                multi_lag_keep="first",
                filter_children=False,
                prune=True,
            ),
        ),
        (
            "+residualized_dcor",
            dict(
                lag_method="partial_dcor",
                lag_pvalue_method="t_test",
                multi_lag=True,
                multi_lag_keep="first",
                filter_children=False,
                prune=True,
            ),
        ),
    ]

    done = set()
    if csv_path.exists():
        for _, r in pd.read_csv(csv_path).iterrows():
            done.add((r["variant"], int(r["T"]), int(r["seed"])))

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy = np.zeros((2, 2))

    for T in T_values:
        print(f"\n{'='*60}\nABLATION — 11-node cascade, T={T}\n{'='*60}")

        for seed in seeds:
            data = load_dataset("ex3", seed=seed, T=T)
            df, gt = data["df"], data["ground_truth"]
            d = df.shape[1]

            for name, kw in configs:
                if (name, T, seed) in done:
                    continue
                try:
                    ci = ParCorrGPU(dummy, device=device)
                    res = run_cedar(
                        df,
                        ci,
                        max_lag,
                        alpha_cond1=ALPHA,
                        alpha_cond2=ALPHA,
                        include_autoreg=True,
                        **kw,
                    )
                    m = evaluate_graph(
                        res.cg_tig[:d, :d, :], gt, exclude_self_loops=True
                    )
                    print(f"  {name:25s} T={T} s={seed} F1={m['F1']:.3f}")

                    hdr = not csv_path.exists()
                    with open(csv_path, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=cols)
                        if hdr:
                            w.writeheader()
                        w.writerow(
                            {
                                "variant": name,
                                "T": T,
                                "seed": seed,
                                "F1": m["F1"],
                                "Precision": m["Precision"],
                                "TPR": m["TPR"],
                                "SHD": m["SHD"],
                            }
                        )
                    done.add((name, T, seed))
                except Exception as e:
                    print(f"  {name:25s} T={T} s={seed} FAIL: {e}")

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"\nAblation summary:")
        for T in T_values:
            print(f"\n  T={T}:")
            for name, _ in configs:
                sub = df[(df["variant"] == name) & (df["T"] == T)]
                if len(sub) > 0:
                    print(
                        f"    {name:25s}: F1={sub.F1.mean():.3f}±{sub.F1.std():.2f}  "
                        f"Prec={sub.Precision.mean():.3f}  TPR={sub.TPR.mean():.3f}"
                    )


def run_named(seeds=None):
    """Named benchmarks: cascade, linear, nonlinear, confounded trends."""
    if seeds is None:
        seeds = list(range(42, 47))  # 5 seeds

    csv_path = Path(__file__).parent / "named_results.csv"
    cols = ["variant", "algo", "seed", "F1", "Precision", "TPR", "SHD"]

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy = np.zeros((2, 2))
    from causalts.ci_tests import SplitKCIGPU

    benchmarks = [
        ("cascade_11node", dict(name="ex3", T=300, max_lag=3, ci="parcorr")),
        ("linear_6node", dict(name="ex2", T=300, max_lag=3, ci="parcorr")),
        ("nonlinear_5node", dict(name="ex1", T=300, max_lag=3, ci="splitkci")),
        (
            "confounded_trends",
            dict(
                name=None,
                T=500,
                max_lag=3,
                ci="parcorr",
                cedar_kw={"include_C": True, "c_preset": "linear+quad"},
                cdnots_kw={"include_C": True, "c_preset": "linear+quad"},
            ),
        ),
    ]

    done = set()
    if csv_path.exists():
        for _, r in pd.read_csv(csv_path).iterrows():
            done.add((r["variant"], r["algo"], int(r["seed"])))

    print(f"\n{'='*60}\nNAMED BENCHMARKS\n{'='*60}")

    for variant, cfg in benchmarks:
        T, max_lag = cfg["T"], cfg["max_lag"]
        print(f"\n  {variant} (T={T})")

        for seed in seeds:
            if cfg["name"]:
                data = load_dataset(cfg["name"], seed=seed, T=T)
            else:
                data = confounded_trends(seed=seed, T=T)
            df_data, gt = data["df"], data["ground_truth"]
            d = df_data.shape[1]

            for algo in ["cedar", "pcmci+", "cdnots", "cdnots+"]:
                if (variant, algo, seed) in done:
                    continue
                try:
                    if cfg["ci"] == "splitkci":
                        ci = SplitKCIGPU(dummy, device=device)
                    else:
                        ci = ParCorrGPU(dummy, device=device)

                    cedar_kw = cfg.get("cedar_kw", {})
                    cdnots_kw = cfg.get("cdnots_kw", {})

                    t0 = time.perf_counter()
                    if algo == "cedar":
                        res = run_cedar(
                            df_data,
                            ci,
                            max_lag,
                            alpha_cond1=ALPHA,
                            alpha_cond2=ALPHA,
                            filter_children=False,
                            prune=True,
                            multi_lag=True,
                            multi_lag_keep="first",
                            include_autoreg=True,
                            lag_method="partial_dcor",
                            **cedar_kw,
                        )
                        G_hat = res.cg_tig[:d, :d, :]
                    elif algo == "pcmci+":
                        G_hat, _ = run_pcmciplus(
                            df_data, tau_max=max_lag, ci_test=ci, pc_alpha=ALPHA
                        )
                    elif algo == "cdnots":
                        res = run_cdnots(
                            df_data,
                            indep_test=ci,
                            num_lags=max_lag,
                            alpha=ALPHA,
                            stable=True,
                            verbose=False,
                            **cdnots_kw,
                        )
                        G_hat = res.cg_tig[:d, :d, :]
                    elif algo == "cdnots+":
                        res = run_cdnots_plus(
                            df_data,
                            indep_test=ci,
                            num_lags=max_lag,
                            alpha=ALPHA,
                            verbose=False,
                            **cdnots_kw,
                        )
                        G_hat = res.cg_tig[:d, :d, :]

                    m = evaluate_graph(G_hat, gt, exclude_self_loops=True)
                    print(f"    {algo:8s} s={seed} F1={m['F1']:.3f}")

                    hdr = not csv_path.exists()
                    with open(csv_path, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=cols)
                        if hdr:
                            w.writeheader()
                        w.writerow(
                            {
                                "variant": variant,
                                "algo": algo,
                                "seed": seed,
                                "F1": m["F1"],
                                "Precision": m["Precision"],
                                "TPR": m["TPR"],
                                "SHD": m["SHD"],
                            }
                        )
                    done.add((variant, algo, seed))
                except Exception as e:
                    print(f"    {algo:8s} s={seed} FAIL: {e}")


def run_calibration(seeds=None, T_values=None, ar_coeffs=None, d=5, alpha=0.05):
    """Type-I error calibration of the analytic dcor t-test under AR(1) null."""
    from causalts.cedar.lag_selection import compute_lag_importance, compute_lag_pvalues

    if seeds is None:
        seeds = list(range(200))
    if T_values is None:
        T_values = [100, 200, 500]
    if ar_coeffs is None:
        ar_coeffs = [0.0, 0.2, 0.5, 0.8, 0.95]

    csv_path = Path(__file__).parent / "calibration_results.csv"
    cols = ["ar_coeff", "T", "empirical_type1", "n_false_pos", "n_tests"]

    print(f"\n{'='*60}\nCALIBRATION — dcor t-test type-I error (nominal alpha={alpha})")
    print(f"d={d}, max_lag={MAX_LAG}, {len(seeds)} null simulations per setting")
    print(f"{'='*60}")

    rows = []
    for ar_coeff in ar_coeffs:
        for T in T_values:
            false_positives = 0
            total_tests = 0
            for sim_seed in seeds:
                rng = np.random.default_rng(sim_seed)
                data = np.zeros((T, d))
                noise = rng.standard_normal((T, d))
                for t in range(1, T):
                    data[t] = ar_coeff * data[t - 1] + noise[t]

                lag_imp = compute_lag_importance(
                    data, MAX_LAG, method="partial_dcor", include_lag0=False
                )
                lag_pvals = compute_lag_pvalues(
                    data, lag_imp, MAX_LAG, method="t_test", include_lag0=False
                )

                for i in range(d):
                    for j in range(d):
                        if i == j:
                            continue
                        for lag in range(1, MAX_LAG + 1):
                            total_tests += 1
                            if lag_pvals[i, j, lag] < alpha:
                                false_positives += 1

            rate = false_positives / total_tests
            print(
                f"  AR={ar_coeff:.2f}, T={T:4d}: type-I = {rate:.4f} "
                f"({false_positives}/{total_tests})"
            )
            rows.append(
                {
                    "ar_coeff": ar_coeff,
                    "T": T,
                    "empirical_type1": rate,
                    "n_false_pos": false_positives,
                    "n_tests": total_tests,
                }
            )

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")


def run_cnode_misspec(seeds=None):
    """C-node misspecification: all methods on confounded-trends with correct/wrong/no basis."""
    if seeds is None:
        seeds = list(range(42, 52))

    csv_path = Path(__file__).parent / "cnode_misspec_results.csv"
    cols = ["algo", "c_config", "seed", "F1", "Precision", "TPR"]

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy = np.zeros((2, 2))

    c_configs = [
        ("linear+quad", dict(include_C=True, c_preset="linear+quad")),
        ("linear_only", dict(include_C=True, c_preset="linear")),
        ("no_cnode", dict(include_C=False)),
    ]

    print(f"\n{'='*60}\nC-NODE MISSPECIFICATION — confounded-trends, T=500")
    print(f"{'='*60}")

    rows = []
    for c_label, c_kw in c_configs:
        print(f"\n  --- {c_label} ---")
        for algo in ["cedar", "pcmci+", "cdnots", "cdnots+"]:
            f1s = []
            for seed in seeds:
                data = confounded_trends(seed=seed, T=500)
                df, gt = data["df"], data["ground_truth"]
                d = df.shape[1]
                max_lag = gt.shape[2] - 1
                ci = ParCorrGPU(dummy, device=device)
                try:
                    if algo == "cedar":
                        res = run_cedar(
                            df,
                            ci,
                            max_lag,
                            alpha_cond1=ALPHA,
                            alpha_cond2=ALPHA,
                            filter_children=False,
                            prune=True,
                            multi_lag=True,
                            multi_lag_keep="first",
                            include_autoreg=True,
                            lag_method="partial_dcor",
                            **c_kw,
                        )
                        G_hat = res.cg_tig[:d, :d, :]
                    elif algo == "pcmci+":
                        G_hat, _ = run_pcmciplus(
                            df, tau_max=max_lag, ci_test=ci, pc_alpha=ALPHA
                        )
                    elif algo == "cdnots":
                        res = run_cdnots(
                            df,
                            indep_test=ci,
                            num_lags=max_lag,
                            alpha=ALPHA,
                            stable=True,
                            verbose=False,
                            **c_kw,
                        )
                        G_hat = res.cg_tig[:d, :d, :]
                    elif algo == "cdnots+":
                        res = run_cdnots_plus(
                            df,
                            indep_test=ci,
                            num_lags=max_lag,
                            alpha=ALPHA,
                            verbose=False,
                            **c_kw,
                        )
                        G_hat = res.cg_tig[:d, :d, :]
                    m = evaluate_graph(G_hat, gt, exclude_self_loops=True)
                    f1s.append(m["F1"])
                    rows.append(
                        {
                            "algo": algo,
                            "c_config": c_label,
                            "seed": seed,
                            "F1": m["F1"],
                            "Precision": m["Precision"],
                            "TPR": m["TPR"],
                        }
                    )
                except Exception as e:
                    f1s.append(0.0)
                    rows.append(
                        {
                            "algo": algo,
                            "c_config": c_label,
                            "seed": seed,
                            "F1": 0.0,
                            "Precision": 0.0,
                            "TPR": 0.0,
                        }
                    )
            print(f"    {algo:8s}: F1={np.mean(f1s):.3f} +/- {np.std(f1s):.2f}")

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "ablation":
        sys.argv.pop(1)
        ap = argparse.ArgumentParser()
        ap.add_argument("--t-values", nargs="+", type=int, default=[100, 200, 500])
        ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
        abl_args = ap.parse_args()
        run_ablation(seeds=abl_args.seeds, T_values=abl_args.t_values)
    elif len(sys.argv) > 1 and sys.argv[1] == "named":
        sys.argv.pop(1)
        run_named()
    elif len(sys.argv) > 1 and sys.argv[1] == "calibration":
        sys.argv.pop(1)
        run_calibration()
    elif len(sys.argv) > 1 and sys.argv[1] == "cnode_misspec":
        sys.argv.pop(1)
        run_cnode_misspec()
    else:
        main()
