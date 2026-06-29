# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run GRACE and baselines on Lorenz-96 benchmark (multiplicative interactions).

This benchmark directly tests whether GRACE handles non-additive causal mechanisms,
since x_{i-1} * (x_{i+1} - x_{i-2}) creates multiplicative parent interactions.

Usage:
    # Quick local test
    python run_lorenz96.py --N 10 --T 500 1000 --n_seeds 3 --device mps

    # Full experiment
    python run_lorenz96.py --N 10 20 --T 500 1000 2000 --n_seeds 10 --device mps
"""

import argparse
import json
import math
import os
import tempfile
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from causalts.synthetic_data.synthetic_datasets import lorenz96

RESULTS_DIR = "results/lorenz96"


def run_grace_on_lorenz(
    bundle, device="mps", model_seed=42, alpha=0.05, skeleton=None, skeleton_runtime=0.0
):
    """Run CDNOTS-Gated on a Lorenz-96 bundle."""
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader

    from causalts.cdnots.phase3_utils import cdnots_discovery
    from causalts.grace.gated_discovery import (
        GatedCausalDiscovery,
        _cdnots_graph_to_binary,
        _detect_accelerator,
        _get_parcorr_class,
        _silence_lightning,
        compute_lambda,
        prepare_data,
    )

    df = bundle["df"]
    max_lag = bundle["max_lag"]
    num_vars = df.shape[1]

    t0 = time.time()

    if skeleton is not None:
        t_skeleton = skeleton_runtime
    else:
        ci_cls = _get_parcorr_class()
        ci_test = ci_cls(np.ascontiguousarray(df.values))
        cdnots_result = cdnots_discovery(
            df=df,
            indep_test=ci_test,
            num_lags=max_lag,
            include_C=True,
            alpha=alpha,
            stable=True,
        )
        if len(cdnots_result) == 3:
            cg, cg_tig, _pvals = cdnots_result
        else:
            cg, cg_tig = cdnots_result

        skeleton = _cdnots_graph_to_binary(cg_tig, num_vars, max_lag)
        t_skeleton = time.time() - t0

    n_skel = int(skeleton.sum())
    n_possible = num_vars * num_vars * (max_lag + 1) - num_vars
    rho_S = n_skel / n_possible if n_possible > 0 else 0.0

    lambda_l0 = compute_lambda(num_vars, len(df), rho_S)

    dataset = prepare_data(df, max_lag, normalize=True)
    N = len(dataset)
    batch_size = max(32, min(N // 8, 256))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    steps_per_epoch = math.ceil(N / batch_size)

    if model_seed is not None:
        pl.seed_everything(model_seed, workers=True)

    model = GatedCausalDiscovery(
        num_vars=num_vars,
        max_lag=max_lag,
        lambda_l0=float(lambda_l0),
        normalize_l0=True,
        steps_per_epoch=steps_per_epoch,
        lambda_lag_group=0.0,
    )
    model.init_from_skeleton(skeleton)
    model.set_skeleton_mask(skeleton)

    accelerator = (
        _detect_accelerator() if device is None or device == "auto" else device
    )

    _silence_lightning()
    trainer = pl.Trainer(
        max_epochs=150,
        accelerator=accelerator,
        deterministic="warn" if model_seed is not None else False,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="train_loss", patience=20, mode="min")
        ],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=loader)

    G_hat, gate_values = model.get_estimated_graph(threshold=0.5)
    runtime = time.time() - t0

    return G_hat, {
        "method": "CDNOTS-Gated",
        "alpha": alpha,
        "lambda_l0": lambda_l0,
        "skeleton_density": rho_S,
        "n_skeleton_edges": n_skel,
        "runtime_skeleton": round(t_skeleton, 2),
        "runtime_total": round(runtime, 2),
    }


def run_cdnots_on_lorenz(bundle, alpha=0.05):
    """Run standalone CDNOTS."""
    from causalts.cdnots.phase3_utils import cdnots_discovery
    from causalts.grace.gated_discovery import (
        _cdnots_graph_to_binary,
        _get_parcorr_class,
    )

    df = bundle["df"]
    max_lag = bundle["max_lag"]
    num_vars = df.shape[1]

    t0 = time.time()
    ci_cls = _get_parcorr_class()
    ci_test = ci_cls(np.ascontiguousarray(df.values))
    cdnots_result = cdnots_discovery(
        df=df,
        indep_test=ci_test,
        num_lags=max_lag,
        include_C=True,
        alpha=alpha,
        stable=True,
    )
    if len(cdnots_result) == 3:
        cg, cg_tig, _ = cdnots_result
    else:
        cg, cg_tig = cdnots_result

    G_hat = _cdnots_graph_to_binary(cg_tig, num_vars, max_lag)
    runtime = time.time() - t0
    return G_hat, {"method": "CDNOTS", "alpha": alpha, "runtime": round(runtime, 2)}


def run_pcmci_on_lorenz(bundle):
    """Run PCMCI with ParCorr."""
    from pcmci_runner import run_pcmci as _run_pcmci

    df = bundle["df"]
    max_lag = bundle["max_lag"]
    df.shape[1]

    t0 = time.time()
    G_hat, _ = _run_pcmci(df.values, max_lag, alpha=0.05, ci_test="parcorr")
    runtime = time.time() - t0
    return G_hat, {"method": "PCMCI", "runtime": round(runtime, 2)}


def run_cuts_plus_on_lorenz(bundle, device="mps"):
    """Run CUTS+ on a Lorenz-96 bundle. Requires CUTS+ to be installed separately."""
    from cuts_plus.cuts_plus import execute_cutplus

    df = bundle["df"]
    max_lag = bundle["max_lag"]
    gt = bundle["ground_truth"]
    data = df.values.astype(np.float32)
    d = data.shape[1]

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        Graph = execute_cutplus(
            data,
            device=device,
            log_dir=tmpdir,
            seed=42,
            options={"total_epoch": 64, "input_step": max_lag},
        )
    runtime = time.time() - t0

    gt_pairs = gt.max(axis=2).astype(np.int8)
    np.fill_diagonal(gt_pairs, 0)

    probs = Graph.copy()
    np.fill_diagonal(probs, 0)

    mask = ~np.eye(d, dtype=bool)
    y_true = gt_pairs[mask].ravel()
    y_score = probs[mask].ravel()

    auc = float(roc_auc_score(y_true, y_score))

    y_pred = (y_score > 0.5).astype(int)
    f1_p = float(f1_score(y_true, y_pred, zero_division=0))
    prec_p = float(precision_score(y_true, y_pred, zero_division=0))
    tpr_p = float(recall_score(y_true, y_pred, zero_division=0))

    G_pairs = (Graph > 0.5).astype(np.int8)
    np.fill_diagonal(G_pairs, 0)
    G_est = np.zeros((d, d, max_lag + 1), dtype=np.int8)
    for lag in range(1, max_lag + 1):
        G_est[:, :, lag] = G_pairs

    return G_est, {
        "method": "CUTS+",
        "runtime": round(runtime, 2),
        "AUC_pair": round(auc, 4),
        "F1_pair": round(f1_p, 4),
        "Prec_pair": round(prec_p, 4),
        "TPR_pair": round(tpr_p, 4),
        "graph_probs_mean": round(float(probs[mask].mean()), 4),
        "graph_probs_min": round(float(probs[mask].min()), 4),
        "graph_probs_max": round(float(probs[mask].max()), 4),
    }


def evaluate_and_print(G_est, G_true, method_name, info):
    """Evaluate and print results."""
    from causalts.grace.gated_discovery import evaluate_graph

    metrics = evaluate_graph(G_est, G_true)
    line = (
        f"    {method_name:20s} F1={metrics['F1']:.3f}  "
        f"Prec={metrics['Precision']:.3f}  TPR={metrics['TPR']:.3f}  "
        f"SHD={metrics['SHD']}  ({info.get('runtime', info.get('runtime_total', 0)):.1f}s)"
    )
    if "AUC_pair" in info:
        line += f"  AUC={info['AUC_pair']:.3f}  F1_pair={info['F1_pair']:.3f}"
    print(line)
    for k in ("AUC_pair", "F1_pair", "Prec_pair", "TPR_pair"):
        if k in info:
            metrics[k] = info[k]
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Lorenz-96 benchmark")
    parser.add_argument(
        "--N",
        nargs="+",
        type=int,
        default=[10, 20],
        help="Number of Lorenz-96 variables",
    )
    parser.add_argument("--T", nargs="+", type=int, default=[500, 1000, 2000])
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["cdnots", "grace", "pcmci"],
        choices=["cdnots", "grace", "pcmci", "cuts_plus"],
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--F", type=float, default=10.0, help="Lorenz forcing")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="CI test significance level (default 0.05)",
    )
    parser.add_argument(
        "--reuse-skeleton",
        action="store_true",
        help="Reuse CDNOTS skeleton from existing results",
    )
    parser.add_argument(
        "--no-skip", action="store_true", help="Re-run even if result file exists"
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Lorenz-96 Benchmark (multiplicative interactions)")
    print(f"  N: {args.N}, T: {args.T}, Seeds: 0-{args.n_seeds-1}")
    print(f"  Methods: {args.methods}")
    alpha_tag = "" if args.alpha == 0.05 else f"_a{args.alpha}"
    print(f"  F={args.F}, alpha={args.alpha}")
    print()

    summary_rows = []

    for N_vars in args.N:
        for seed in range(args.n_seeds):
            T_max = max(args.T) + 500
            bundle_full = lorenz96(seed=seed, T=T_max, N=N_vars, F=args.F)
            gt = bundle_full["ground_truth"]

            print(f"\n{'='*60}")
            print(f"N={N_vars}, seed={seed}")

            for T_val in args.T:
                print(f"\n  --- T={T_val} ---")
                bundle = dict(bundle_full)
                bundle["df"] = bundle_full["df"].iloc[:T_val].copy()

                for method in args.methods:
                    result_path = os.path.join(
                        RESULTS_DIR,
                        f"lorenz96_N{N_vars}_{method}{alpha_tag}_seed{seed}_T{T_val}.json",
                    )
                    if os.path.exists(result_path) and not args.no_skip:
                        print(f"    [{method}] exists, skipping")
                        continue

                    try:
                        if method == "grace":
                            grace_kwargs = dict(device=args.device, alpha=args.alpha)
                            if args.reuse_skeleton:
                                skel_fname = f"lorenz96_N{N_vars}_cdnots{alpha_tag}_seed{seed}_T{T_val}.json"
                                skel_path = os.path.join(RESULTS_DIR, skel_fname)
                                if os.path.exists(skel_path):
                                    import json as _json

                                    with open(skel_path) as _f:
                                        skel_rec = _json.load(_f)
                                    g_key = (
                                        "graph_estimated"
                                        if "graph_estimated" in skel_rec
                                        else "G_est"
                                    )
                                    grace_kwargs["skeleton"] = np.array(
                                        skel_rec[g_key], dtype=np.int8
                                    )
                                    rt_key = (
                                        "runtime_seconds"
                                        if "runtime_seconds" in skel_rec
                                        else "info"
                                    )
                                    if rt_key == "info":
                                        grace_kwargs["skeleton_runtime"] = skel_rec.get(
                                            "info", {}
                                        ).get("runtime", 0.0)
                                    else:
                                        grace_kwargs["skeleton_runtime"] = skel_rec.get(
                                            "runtime_seconds", 0.0
                                        )
                                    print(
                                        "(reusing cdnots skeleton) ", end="", flush=True
                                    )
                            G_est, info = run_grace_on_lorenz(bundle, **grace_kwargs)
                        elif method == "cdnots":
                            G_est, info = run_cdnots_on_lorenz(bundle, alpha=args.alpha)
                        elif method == "pcmci":
                            G_est, info = run_pcmci_on_lorenz(bundle)
                        elif method == "cuts_plus":
                            G_est, info = run_cuts_plus_on_lorenz(
                                bundle, device=args.device
                            )
                        else:
                            continue

                        metrics = evaluate_and_print(G_est, gt, method, info)

                        row = {
                            "N": N_vars,
                            "T": T_val,
                            "seed": seed,
                            "method": method,
                            **metrics,
                            **info,
                        }
                        summary_rows.append(row)

                        result = {
                            "dataset": f"lorenz96_N{N_vars}",
                            "method": method,
                            "seed": seed,
                            "T": T_val,
                            "N": N_vars,
                            "metrics": metrics,
                            "info": info,
                            "G_est": G_est.tolist(),
                        }
                        with open(result_path, "w") as f:
                            json.dump(result, f, indent=2)

                    except Exception as e:
                        print(f"    [{method}] FAILED: {e}")
                        traceback.print_exc()

    if summary_rows:
        print(f"\n{'='*60}")
        print("Summary")
        df_summary = pd.DataFrame(summary_rows)
        pivot = (
            df_summary.groupby(["N", "T", "method"])["F1"]
            .agg(["mean", "std", "count"])
            .round(3)
        )
        print(pivot.to_string())


if __name__ == "__main__":
    main()
