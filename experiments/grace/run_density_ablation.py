# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Density ablation: sweep ER edge probability × λ to understand GRACE on dense graphs.

Tests how GRACE gate values and F1 degrade as graph density increases,
and whether λ tuning can recover performance.

Usage:
    # Quick test
    python run_density_ablation.py --er_p 0.08 --n_seeds 1 --device mps

    # Full experiment
    python run_density_ablation.py --device mps
"""

import argparse
import json
import math
import os
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from causalts.synthetic_data.synthetic_datasets import topology_scp

RESULTS_DIR = "results/density_ablation"

DEFAULT_DENSITIES = [0.04, 0.08, 0.12, 0.16]
DEFAULT_LAMBDAS = [0.001, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02]


def compute_pair_metrics(gate_values, gt, threshold=0.5):
    """Compute pair-level metrics from continuous gate values."""
    d = gt.shape[0]
    gate_pairs = gate_values.max(axis=2)
    gt_pairs = gt.max(axis=2).astype(np.int8)
    np.fill_diagonal(gate_pairs, 0)
    np.fill_diagonal(gt_pairs, 0)

    mask = ~np.eye(d, dtype=bool)
    y_true = gt_pairs[mask].ravel()
    y_score = gate_pairs[mask].ravel()

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return None

    auc = float(roc_auc_score(y_true, y_score))

    y_pred = (y_score > threshold).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    tpr = float(recall_score(y_true, y_pred, zero_division=0))

    return {"AUC": auc, "F1": f1, "Prec": prec, "TPR": tpr}


def find_optimal_threshold(gate_values, gt):
    """Sweep thresholds to find the one that maximizes pair-level F1."""
    d = gt.shape[0]
    gate_pairs = gate_values.max(axis=2)
    gt_pairs = gt.max(axis=2).astype(np.int8)
    np.fill_diagonal(gate_pairs, 0)
    np.fill_diagonal(gt_pairs, 0)

    mask = ~np.eye(d, dtype=bool)
    y_true = gt_pairs[mask].ravel()
    y_score = gate_pairs[mask].ravel()

    best_f1, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.05, 0.95, 0.05):
        y_pred = (y_score > thresh).astype(int)
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)

    return best_thresh, best_f1


def gate_stats(gate_values):
    """Compute summary statistics of gate values."""
    nonzero = gate_values[gate_values > 0]
    return {
        "n_nonzero": int(len(nonzero)),
        "median": float(np.median(nonzero)) if len(nonzero) > 0 else 0.0,
        "mean": float(np.mean(nonzero)) if len(nonzero) > 0 else 0.0,
        "frac_above_05": float((nonzero > 0.5).sum() / max(len(nonzero), 1)),
        "frac_below_01": float((nonzero < 0.1).sum() / max(len(nonzero), 1)),
    }


def run_cdnots_skeleton(bundle, alpha=0.05, no_batch=False, max_degree=None):
    """Run CDNOTS and return both the full result and the binary skeleton."""
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
    ci_kwargs = {}
    if no_batch and ci_cls.__name__ == "ParCorrGPU":
        ci_kwargs["enable_batching"] = False
    ci_test = ci_cls(np.ascontiguousarray(df.values), **ci_kwargs)
    cdnots_kwargs = dict(
        df=df,
        indep_test=ci_test,
        num_lags=max_lag,
        include_C=True,
        alpha=alpha,
        stable=True,
    )
    if max_degree is not None:
        cdnots_kwargs["max_degree"] = max_degree
    cdnots_result = cdnots_discovery(**cdnots_kwargs)
    if len(cdnots_result) == 3:
        cg, cg_tig, _ = cdnots_result
    else:
        cg, cg_tig = cdnots_result

    G_cdnots = _cdnots_graph_to_binary(cg_tig, num_vars, max_lag)
    runtime = time.time() - t0

    return G_cdnots, runtime


def run_grace_with_lambda(bundle, skeleton, lambda_l0, device="mps", model_seed=42):
    """Run GRACE gated model with a specific λ, reusing a pre-computed skeleton."""
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader

    from causalts.grace.gated_discovery import (
        GatedCausalDiscovery,
        _detect_accelerator,
        _silence_lightning,
        prepare_data,
    )

    df = bundle["df"]
    max_lag = bundle["max_lag"]
    num_vars = df.shape[1]

    t0 = time.time()

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

    accelerator = device if device not in (None, "auto") else _detect_accelerator()
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

    return G_hat, gate_values, runtime


def evaluate_graph(G_est, G_true):
    """Compute lag-level metrics."""
    from causalts.grace.gated_discovery import evaluate_graph as _eval

    return _eval(G_est, G_true)


def main():
    parser = argparse.ArgumentParser(description="Density ablation for GRACE")
    parser.add_argument(
        "--er_p",
        nargs="+",
        type=float,
        default=DEFAULT_DENSITIES,
        help="ER edge probabilities to sweep",
    )
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=DEFAULT_LAMBDAS,
        help="Lambda values to sweep",
    )
    parser.add_argument("--n_vars", type=int, default=50)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--no-batch", action="store_true")
    parser.add_argument(
        "--max-degree",
        type=int,
        default=None,
        help="Max conditioning set size for CDNOTS (limits memory on dense graphs)",
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    from causalts.grace.gated_discovery import compute_lambda

    lambda_list = args.lambdas + ["adaptive", "density_corrected"]

    print("Density Ablation")
    print(f"  ER densities: {args.er_p}")
    print(f"  Lambdas: {lambda_list}")
    print(f"  d={args.n_vars}, T={args.T}, Seeds: 0-{args.n_seeds-1}")
    print()

    for er_p in args.er_p:
        for seed in range(args.n_seeds):
            T_gen = args.T + 500
            try:
                bundle_full = topology_scp(
                    "erdos_renyi",
                    seed=seed,
                    n_vars=args.n_vars,
                    T=T_gen,
                    max_lag=5,
                    er_p=er_p,
                )
            except RuntimeError as e:
                print(f"  [p={er_p}] seed={seed}: generation failed: {e}")
                continue

            bundle = dict(bundle_full)
            bundle["df"] = bundle_full["df"].iloc[: args.T].copy()
            gt = bundle_full["ground_truth"]
            meta = bundle_full["meta"]
            parents_per_var = meta["n_cross_edges"] / args.n_vars

            print(f"\n{'='*60}")
            print(
                f"ER p={er_p} | seed={seed} | {meta['n_cross_edges']} edges "
                f"| {parents_per_var:.1f} parents/var | sr={meta['spectral_radius']:.3f}"
            )

            cdnots_path = os.path.join(
                RESULTS_DIR,
                f"er_d{args.n_vars}_T{args.T}_p{er_p}_cdnots_seed{seed}.json",
            )
            if os.path.exists(cdnots_path):
                print("    [cdnots] exists, loading skeleton")
                with open(cdnots_path) as f:
                    cdnots_result = json.load(f)
                skeleton = np.array(cdnots_result["G_est"])
            else:
                try:
                    skeleton, skel_time = run_cdnots_skeleton(
                        bundle,
                        alpha=args.alpha,
                        no_batch=args.no_batch,
                        max_degree=args.max_degree,
                    )
                    cdnots_metrics = evaluate_graph(skeleton, gt)
                    print(
                        f"    cdnots    F1={cdnots_metrics['F1']:.3f}  "
                        f"Prec={cdnots_metrics['Precision']:.3f}  "
                        f"TPR={cdnots_metrics['TPR']:.3f}  ({skel_time:.1f}s)"
                    )

                    result = {
                        "method": "cdnots",
                        "er_p": er_p,
                        "seed": seed,
                        "n_vars": args.n_vars,
                        "T": args.T,
                        "metrics": cdnots_metrics,
                        "info": {"runtime": round(skel_time, 2)},
                        "G_est": skeleton.tolist(),
                        "meta": meta,
                    }
                    with open(cdnots_path, "w") as f:
                        json.dump(result, f, indent=2)
                except Exception as e:
                    print(f"    [cdnots] FAILED: {e}")
                    traceback.print_exc()
                    continue

            n_skel = int(skeleton.sum())
            n_possible = args.n_vars * args.n_vars * 6 - args.n_vars
            rho_S = n_skel / n_possible if n_possible > 0 else 0.0
            avg_skel_parents = n_skel / args.n_vars

            for lam in lambda_list:
                if lam == "adaptive":
                    lambda_val = compute_lambda(args.n_vars, args.T, rho_S)
                    lam_str = "adaptive"
                elif lam == "density_corrected":
                    lambda_base = compute_lambda(args.n_vars, args.T, rho_S)
                    lambda_val = lambda_base / max(avg_skel_parents / 2, 1.0)
                    lam_str = "density_corrected"
                else:
                    lambda_val = float(lam)
                    lam_str = f"{lam}"

                result_path = os.path.join(
                    RESULTS_DIR,
                    f"er_d{args.n_vars}_T{args.T}_p{er_p}_lam{lam_str}_seed{seed}.json",
                )
                if os.path.exists(result_path):
                    print(f"    [λ={lam_str}] exists, skipping")
                    continue

                try:
                    G_est, gate_values, runtime = run_grace_with_lambda(
                        bundle, skeleton, lambda_val, device=args.device
                    )

                    metrics = evaluate_graph(G_est, gt)
                    pair_metrics = compute_pair_metrics(gate_values, gt, threshold=0.5)
                    opt_thresh, opt_f1 = find_optimal_threshold(gate_values, gt)
                    gs = gate_stats(gate_values)

                    line = (
                        f"    λ={lam_str:>8s}  F1={metrics['F1']:.3f}  "
                        f"Prec={metrics['Precision']:.3f}  TPR={metrics['TPR']:.3f}"
                    )
                    if pair_metrics:
                        line += f"  AUC={pair_metrics['AUC']:.3f}"
                    line += f"  opt_F1={opt_f1:.3f}@{opt_thresh:.2f}"
                    line += f"  gate_med={gs['median']:.3f}  ({runtime:.0f}s)"
                    print(line)

                    result = {
                        "method": "grace",
                        "er_p": er_p,
                        "seed": seed,
                        "n_vars": args.n_vars,
                        "T": args.T,
                        "lambda": lam_str,
                        "lambda_val": lambda_val,
                        "metrics": metrics,
                        "pair_metrics": pair_metrics,
                        "optimal_threshold": opt_thresh,
                        "f1_at_optimal": opt_f1,
                        "gate_stats": gs,
                        "info": {
                            "runtime": round(runtime, 2),
                            "lambda_l0": lambda_val,
                            "skeleton_density": rho_S,
                            "n_skeleton_edges": n_skel,
                            "n_true_edges": meta["n_cross_edges"],
                            "parents_per_var": parents_per_var,
                        },
                        "G_est": G_est.tolist(),
                        "gate_values": gate_values.tolist(),
                        "meta": meta,
                    }
                    with open(result_path, "w") as f:
                        json.dump(result, f, indent=2)

                except Exception as e:
                    print(f"    [λ={lam_str}] FAILED: {e}")
                    traceback.print_exc()

    print(f"\n{'='*60}")
    print("Summary")
    import glob

    files = glob.glob(os.path.join(RESULTS_DIR, "er_d*_lam*.json"))
    if files:
        rows = []
        for f in files:
            r = json.load(open(f))
            if r["method"] != "grace":
                continue
            pm = r.get("pair_metrics") or {}
            gs = r.get("gate_stats") or {}
            rows.append(
                {
                    "er_p": r["er_p"],
                    "lambda": r["lambda"],
                    "F1": r["metrics"]["F1"],
                    "AUC": pm.get("AUC"),
                    "opt_F1": r.get("f1_at_optimal"),
                    "gate_med": gs.get("median"),
                }
            )
        if rows:
            df_summary = pd.DataFrame(rows)
            pivot = (
                df_summary.groupby(["er_p", "lambda"])
                .agg(
                    {"F1": "mean", "AUC": "mean", "opt_F1": "mean", "gate_med": "mean"}
                )
                .round(3)
            )
            print(pivot.to_string())


if __name__ == "__main__":
    main()
