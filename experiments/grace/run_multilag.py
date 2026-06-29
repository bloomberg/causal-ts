# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Multi-lag ablation: evaluate GRACE with k=1 vs k=2 on multi-lag SCP graphs.

Pre-generated data is expected in results/scaling/data/scp_multilag_d{d}_seed{s}.npz
(use SCPGraphGenerator.sample(multi_lag_fraction=0.3) to create).

Usage:
    # Run all configs
    python run_multilag.py --n_vars 20 50 --T 500 1000 --n_graphs 10

    # Quick test
    python run_multilag.py --n_vars 20 --T 500 --n_graphs 3
"""

import argparse
import json
import os
import signal
import time
import traceback

import numpy as np
import pandas as pd
from result_storage import (
    result_exists,
    save_result,
)
from run_baselines import run_cdnots, run_dynotears, run_dynotears_gated

from causalts.grace import run_cdnots_gated
from causalts.grace.gated_discovery import evaluate_graph

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_multilag_bundle(n_vars, seed, data_dir="results/scaling/data"):
    """Load pre-generated multi-lag SCP data bundle."""
    fname = f"scp_multilag_d{n_vars}_seed{seed}.npz"
    fpath = os.path.join(data_dir, fname)
    if not os.path.exists(fpath):
        return None
    npz = np.load(fpath, allow_pickle=True)
    meta = json.loads(str(npz["meta_json"][0]))
    return {
        "df": pd.DataFrame(npz["data"], columns=npz["var_names"]),
        "ground_truth": npz["ground_truth"],
        "max_lag": int(npz["max_lag"][0]),
        "var_names": list(npz["var_names"]),
        "meta": meta,
    }


def slice_bundle(bundle, T):
    """Slice data to first T rows."""
    sliced = dict(bundle)
    sliced["df"] = bundle["df"].iloc[:T].copy()
    return sliced


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------


def run_cdnots_baseline(bundle):
    """Run CDNOTS (no gating) and return (G_est, metrics, runtime)."""
    G_est, _, _, runtime = run_cdnots(bundle)
    metrics = evaluate_graph(G_est, bundle["ground_truth"])
    return G_est, metrics, runtime


def run_grace(bundle, max_lags_per_pair=None, model_seed=42):
    """Run CDNOTS-Gated with specified k and return (G_est, metrics, runtime, info)."""
    df = bundle["df"]
    gt = bundle["ground_truth"]
    max_lag = bundle["max_lag"]

    lag_group = 5.0 if max_lags_per_pair is not None else 0.0

    t0 = time.time()
    G_hat, gate_values, info = run_cdnots_gated(
        df,
        max_lag=max_lag,
        ground_truth=gt,
        verbose=False,
        lambda_lag_group=lag_group,
        max_lags_per_pair=max_lags_per_pair if max_lags_per_pair is not None else 1,
        model_seed=model_seed,
    )
    runtime = time.time() - t0
    metrics = evaluate_graph(G_hat, gt)
    return G_hat, metrics, runtime, info


# ---------------------------------------------------------------------------
# Timeout support
# ---------------------------------------------------------------------------


class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _Timeout("Method timed out")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

METHODS = ["cdnots", "grace", "grace_k1", "grace_k2", "dynotears", "dynotears_gated"]


def main():
    parser = argparse.ArgumentParser(
        description="Multi-lag ablation: GRACE k=1 vs k=2 on multi-lag SCP graphs"
    )
    parser.add_argument("--n_vars", nargs="+", type=int, default=[20, 50])
    parser.add_argument("--T", nargs="+", type=int, default=[500, 1000])
    parser.add_argument("--n_graphs", type=int, default=10)
    parser.add_argument(
        "--methods", nargs="+", default=METHODS, choices=METHODS, help="Methods to run"
    )
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--data-dir", type=str, default="results/scaling/data")
    parser.add_argument("--results-dir", type=str, default="results/multilag")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip", action="store_true")

    args = parser.parse_args()

    if args.no_skip:
        args.skip_existing = False

    seeds = list(range(args.n_graphs))
    T_values = sorted(args.T)
    os.makedirs(args.results_dir, exist_ok=True)

    print("Multi-Lag Ablation")
    print(f"  n_vars:     {args.n_vars}")
    print(f"  T:          {T_values}")
    print(f"  methods:    {args.methods}")
    print(f"  n_graphs:   {args.n_graphs}")
    print(f"  model_seed: {args.model_seed}")
    print(f"  timeout:    {args.timeout}s")
    print(f"  data_dir:   {args.data_dir}")
    print(f"  results ->  {args.results_dir}/")
    print(f"  skip:       {args.skip_existing}")
    print()

    summary_rows = []

    for n_vars in args.n_vars:
        for seed in seeds:
            bundle_full = load_multilag_bundle(n_vars, seed, data_dir=args.data_dir)
            if bundle_full is None:
                print(f"d={n_vars} seed={seed}: data not found, skipping")
                continue

            gt = bundle_full["ground_truth"]
            n_edges = int(gt[:, :, 1:].sum())
            n_pairs = int((gt[:, :, 1:].sum(axis=2) > 0).sum())
            n_multi = int(((gt[:, :, 1:].sum(axis=2)) > 1).sum())

            print(f"{'='*70}")
            print(
                f"d={n_vars}, seed={seed}: {n_edges} edges, "
                f"{n_pairs} pairs, {n_multi} multi-lag pairs"
            )
            print(f"{'='*70}")

            for T_val in T_values:
                bundle = slice_bundle(bundle_full, T_val)
                ds_name = f"scp_multilag_d{n_vars}"

                print(f"\n  --- T={T_val} ---")

                for method_name in args.methods:
                    if args.skip_existing and result_exists(
                        ds_name,
                        method_name,
                        seed,
                        T_val,
                        results_dir=args.results_dir,
                    ):
                        print(f"    [{method_name}] exists, skipping")
                        continue

                    print(f"    [{method_name}] ", end="", flush=True)

                    try:
                        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                        signal.alarm(args.timeout)

                        try:
                            if method_name == "cdnots":
                                G_est, metrics, runtime = run_cdnots_baseline(bundle)
                                method_params = {"method": "cdnots"}
                                method_info = {"n_edges": int(G_est.sum())}
                            elif method_name == "grace":
                                G_est, metrics, runtime, info = run_grace(
                                    bundle,
                                    max_lags_per_pair=None,
                                    model_seed=args.model_seed,
                                )
                                method_params = {
                                    "max_lags_per_pair": None,
                                    "lambda_lag_group": 0.0,
                                    "lambda_l0": info["lambda_l0"],
                                    "skeleton_density": info["skeleton_density"],
                                }
                                method_info = {
                                    "n_skeleton_edges": info["n_skeleton_edges"],
                                    "runtime_skeleton": info["runtime_skeleton"],
                                    "runtime_training": info["runtime_training"],
                                }
                            elif method_name == "grace_k1":
                                G_est, metrics, runtime, info = run_grace(
                                    bundle,
                                    max_lags_per_pair=1,
                                    model_seed=args.model_seed,
                                )
                                method_params = {
                                    "max_lags_per_pair": 1,
                                    "lambda_l0": info["lambda_l0"],
                                    "skeleton_density": info["skeleton_density"],
                                }
                                method_info = {
                                    "n_skeleton_edges": info["n_skeleton_edges"],
                                    "runtime_skeleton": info["runtime_skeleton"],
                                    "runtime_training": info["runtime_training"],
                                }
                            elif method_name == "grace_k2":
                                G_est, metrics, runtime, info = run_grace(
                                    bundle,
                                    max_lags_per_pair=2,
                                    model_seed=args.model_seed,
                                )
                                method_params = {
                                    "max_lags_per_pair": 2,
                                    "lambda_l0": info["lambda_l0"],
                                    "skeleton_density": info["skeleton_density"],
                                }
                                method_info = {
                                    "n_skeleton_edges": info["n_skeleton_edges"],
                                    "runtime_skeleton": info["runtime_skeleton"],
                                    "runtime_training": info["runtime_training"],
                                }
                            elif method_name == "dynotears":
                                G_est, dyno_params, dyno_info, runtime = run_dynotears(
                                    bundle
                                )
                                metrics = evaluate_graph(G_est, gt)
                                method_params = dyno_params
                                method_info = {"n_edges": int(G_est.sum())}
                            elif method_name == "dynotears_gated":
                                G_est, dg_params, dg_info, runtime = (
                                    run_dynotears_gated(
                                        bundle,
                                        verbose=False,
                                        model_seed=args.model_seed,
                                    )
                                )
                                metrics = evaluate_graph(G_est, gt)
                                method_params = dg_params
                                method_info = dg_info
                            else:
                                raise ValueError(f"Unknown method: {method_name}")
                        finally:
                            signal.alarm(0)
                            signal.signal(signal.SIGALRM, old_handler)

                        path = save_result(
                            dataset=ds_name,
                            seed=seed,
                            T=T_val,
                            n_vars=n_vars,
                            max_lag=bundle["max_lag"],
                            method=method_name,
                            method_params=method_params,
                            G_est=G_est,
                            G_true=gt,
                            runtime_seconds=runtime,
                            method_info={
                                **method_info,
                                "n_multi_lag_pairs": n_multi,
                            },
                            results_dir=args.results_dir,
                        )

                        row = {
                            "d": n_vars,
                            "T": T_val,
                            "seed": seed,
                            "method": method_name,
                            "runtime": runtime,
                            "n_multi_lag": n_multi,
                            **metrics,
                        }
                        summary_rows.append(row)

                        print(
                            f"F1={metrics['F1']:.3f}  SHD={metrics['SHD']}  "
                            f"TPR={metrics['TPR']:.3f}  Prec={metrics['Precision']:.3f}  "
                            f"({runtime:.1f}s)"
                        )

                    except _Timeout:
                        print(f"TIMEOUT (>{args.timeout}s)")
                        summary_rows.append(
                            {
                                "d": n_vars,
                                "T": T_val,
                                "seed": seed,
                                "method": method_name,
                                "runtime": args.timeout,
                                "n_multi_lag": n_multi,
                                "F1": np.nan,
                                "SHD": np.nan,
                            }
                        )
                    except Exception as e:
                        print(f"FAILED: {e}")
                        traceback.print_exc()

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)

        print(f"\n{'='*70}")
        print("PER-SEED RESULTS")
        print(f"{'='*70}")
        cols = [
            "d",
            "seed",
            "T",
            "method",
            "F1",
            "SHD",
            "TPR",
            "Precision",
            "n_multi_lag",
            "runtime",
        ]
        detail = df_summary[[c for c in cols if c in df_summary.columns]]
        detail = detail.sort_values(["d", "seed", "T", "method"])
        print(detail.to_string(index=False, float_format="%.3f"))

        print(f"\n{'='*70}")
        print("AGGREGATED SUMMARY (over seeds)")
        print(f"{'='*70}")
        agg = (
            df_summary.groupby(["d", "T", "method"])
            .agg(
                {
                    "F1": ["mean", "std", "count"],
                    "SHD": ["mean", "std"],
                    "TPR": "mean",
                    "Precision": "mean",
                    "runtime": "mean",
                }
            )
            .round(3)
        )
        print(agg.to_string())


if __name__ == "__main__":
    main()
