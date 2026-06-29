# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run CUTS+ comparison: pair-level F1 across methods.

Usage:
    # Run all methods at d=20,50 T=1000 (10 seeds)
    python run_cuts_plus_comparison.py

    # Run only CUTS+ at d=20 (seeds 0-4) on MPS
    python run_cuts_plus_comparison.py --methods cuts_plus --dims 20 --seeds 0 1 2 3 4 --device mps

    # Skip already-computed results (safe for multi-machine runs)
    python run_cuts_plus_comparison.py --skip-existing
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from result_storage import (
    load_data_bundle,
    result_exists,
    save_result,
)

DEFAULT_RESULTS_DIR = "results/comparison"

# Methods that have no lag resolution — only pair-level metrics are meaningful
NO_LAG_METHODS = {"cuts_plus"}


def pair_metrics(G_est, G_true):
    """Compute pair-level precision, recall, F1 (collapse lag dimension)."""
    est_pairs = G_est.any(axis=2).astype(int)
    true_pairs = G_true.any(axis=2).astype(int)
    np.fill_diagonal(est_pairs, 0)
    np.fill_diagonal(true_pairs, 0)

    tp = (est_pairs & true_pairs).sum()
    fp = (est_pairs & ~true_pairs).sum()
    fn = (~est_pairs & true_pairs).sum()

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "pair_prec": prec,
        "pair_rec": rec,
        "pair_f1": f1,
        "pair_tp": int(tp),
        "pair_fp": int(fp),
        "pair_fn": int(fn),
    }


def load_bundle(n_vars, seed, T, data_dir="results/scaling/data"):
    """Load a pre-generated SCP bundle and slice to T."""
    bundle = load_data_bundle(n_vars, seed, data_dir=data_dir)
    if bundle is None:
        raise FileNotFoundError(
            f"No data bundle for d={n_vars} seed={seed}. "
            f"Generate with: python run_scaling.py generate --n_vars {n_vars} --n_graphs {seed+1}"
        )
    df = bundle["df"].iloc[:T].copy()
    bundle["df"] = df
    return bundle


def run_method(method_name, bundle, **kwargs):
    """Run a single method via METHOD_MAP."""
    from run_baselines import METHOD_MAP

    return METHOD_MAP[method_name](bundle, **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Run CUTS+ comparison experiments with pair-level F1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["pcmci", "cdnots", "tcdf", "cdnots_gated", "cuts_plus"],
        help="Methods to run",
    )
    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=[20, 50, 100],
        help="Dimensionalities to test",
    )
    parser.add_argument(
        "--T",
        nargs="+",
        type=int,
        default=[300, 500, 1000, 2000],
        help="Time series lengths to test",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(range(10)), help="Seeds to run"
    )
    parser.add_argument(
        "--cuts-plus-epochs", type=int, default=64, help="CUTS+ training epochs"
    )
    parser.add_argument(
        "--device", type=str, default="mps", help="Device for CUTS+ (cuda, mps, cpu)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help="Local results directory",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip if result exists locally"
    )
    parser.add_argument(
        "--model-seed", type=int, default=None, help="Model seed for gated methods"
    )
    parser.add_argument(
        "--reuse-skeleton",
        action="store_true",
        help="Reuse pre-computed skeleton from existing results",
    )
    parser.add_argument(
        "--skeleton-dir",
        type=str,
        default="results",
        help="Directory with cdnots results for skeleton reuse",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="results/scaling/data",
        help="Directory with pre-generated data bundles",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    all_results = []

    for d in args.dims:
        for T_val in args.T:
            print(f"\n{'='*60}")
            print(f"d={d}, T={T_val}")
            print(f"{'='*60}")

            for method in args.methods:
                pair_f1_list, runtime_list = [], []
                pair_prec_list, pair_rec_list = [], []

                for seed in args.seeds:
                    ds_name = f"scp_d{d}"

                    if args.skip_existing and result_exists(
                        ds_name,
                        method,
                        seed,
                        T_val,
                        results_dir=results_dir,
                    ):
                        print(f"  {method} seed={seed}... exists, skipping")
                        continue

                    print(f"  {method} seed={seed}...", end=" ", flush=True)

                    try:
                        bundle = load_bundle(d, seed, T_val, data_dir=args.data_dir)
                    except FileNotFoundError as e:
                        print(f"SKIPPED (no data: {e})")
                        continue
                    gt = bundle["ground_truth"]

                    kwargs = {}
                    if method == "cuts_plus":
                        kwargs["total_epoch"] = args.cuts_plus_epochs
                        kwargs["seed"] = seed
                        kwargs["device"] = args.device
                    if (
                        method in ("cdnots_gated", "cdnots_rcot_gated")
                        and args.model_seed is not None
                    ):
                        kwargs["model_seed"] = args.model_seed
                    if args.reuse_skeleton and method in (
                        "cdnots_gated",
                        "cdnots_rcot_gated",
                    ):
                        skel_method = "cdnots_rcot" if "rcot" in method else "cdnots"
                        from result_storage import load_skeleton

                        skel, skel_rt = load_skeleton(
                            ds_name,
                            seed,
                            T_val,
                            skeleton_method=skel_method,
                            results_dir=args.skeleton_dir,
                        )
                        if skel is not None:
                            kwargs["skeleton"] = skel
                            kwargs["skeleton_runtime"] = skel_rt
                            print(
                                f"(reusing {skel_method} skeleton) ", end="", flush=True
                            )

                    try:
                        G_est, method_params, method_info, runtime = run_method(
                            method, bundle, **kwargs
                        )
                    except Exception as e:
                        print(f"FAILED: {e}")
                        import traceback

                        traceback.print_exc()
                        continue

                    if G_est.shape != gt.shape:
                        from run_scaling import _align_graph_lags

                        G_est = _align_graph_lags(G_est, bundle["max_lag"])

                    path = save_result(
                        dataset=ds_name,
                        seed=seed,
                        T=T_val,
                        n_vars=d,
                        max_lag=bundle["max_lag"],
                        method=method,
                        method_params=method_params,
                        G_est=G_est,
                        G_true=gt,
                        runtime_seconds=runtime,
                        method_info=method_info,
                        results_dir=results_dir,
                    )

                    pm = pair_metrics(G_est, gt)

                    pair_f1_list.append(pm["pair_f1"])
                    pair_prec_list.append(pm["pair_prec"])
                    pair_rec_list.append(pm["pair_rec"])
                    runtime_list.append(runtime)

                    print(
                        f"pairF1={pm['pair_f1']:.3f} P={pm['pair_prec']:.3f} R={pm['pair_rec']:.3f} {runtime:.1f}s"
                    )

                if pair_f1_list:
                    row = {
                        "d": d,
                        "T": T_val,
                        "method": method,
                        "pair_f1_mean": np.mean(pair_f1_list),
                        "pair_f1_std": np.std(pair_f1_list),
                        "pair_prec_mean": np.mean(pair_prec_list),
                        "pair_rec_mean": np.mean(pair_rec_list),
                        "runtime_mean": np.mean(runtime_list),
                        "n_seeds": len(pair_f1_list),
                    }
                    all_results.append(row)
                    print(
                        f"  >> {method}: pairF1={row['pair_f1_mean']:.3f}+/-{row['pair_f1_std']:.3f}  "
                        f"P={row['pair_prec_mean']:.3f}  R={row['pair_rec_mean']:.3f}  "
                        f"runtime={row['runtime_mean']:.1f}s  ({row['n_seeds']} seeds)"
                    )

    if all_results:
        out_dir = Path(results_dir)
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(out_dir / "pair_f1_summary.csv", index=False)
        print(f"\nSummary saved to {out_dir / 'pair_f1_summary.csv'}")

        print("\n" + "=" * 80)
        print("PAIR-LEVEL F1 COMPARISON")
        print("=" * 80)
        for d in args.dims:
            for T_val in args.T:
                print(f"\nd={d}, T={T_val}:")
                print(
                    f"  {'Method':<20} {'Pair F1':>16} {'Pair Prec':>10} {'Pair Rec':>10} {'Runtime':>10}"
                )
                print(f"  {'-'*68}")
                for row in all_results:
                    if row["d"] == d and row["T"] == T_val:
                        print(
                            f"  {row['method']:<20} "
                            f"{row['pair_f1_mean']:.3f}+/-{row['pair_f1_std']:.3f} "
                            f"{row['pair_prec_mean']:>10.3f} "
                            f"{row['pair_rec_mean']:>10.3f} "
                            f"{row['runtime_mean']:>8.1f}s"
                        )


if __name__ == "__main__":
    main()
