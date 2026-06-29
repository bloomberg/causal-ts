# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run GRACE and baselines on diverse graph topology benchmarks.

Tests GRACE on three fundamentally different graph families:
  - Scale-free (Barabasi-Albert): hub-and-spoke structure
  - Erdos-Renyi: random directed graph with controlled density
  - Small-world (Watts-Strogatz): clustered ring with rewiring

Usage:
    # Quick test
    python run_topology.py --topologies scale_free --n_vars 50 --T 1000 --n_seeds 3 --device mps

    # Full experiment
    python run_topology.py --n_vars 50 --T 500 1000 --n_seeds 10 --device mps
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

RESULTS_DIR = "results/topology"


def compute_pair_auroc(gate_values, gt):
    """Compute pair-level AUROC from continuous gate values."""
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
    return float(roc_auc_score(y_true, y_score))


def run_grace_on_topology(
    bundle,
    device="mps",
    model_seed=42,
    alpha=0.05,
    no_batch=False,
    skeleton=None,
    skeleton_runtime=0.0,
):
    """Run CDNOTS-Gated on a topology bundle."""
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
        ci_kwargs = {}
        if no_batch and ci_cls.__name__ == "ParCorrGPU":
            ci_kwargs["enable_batching"] = False
        ci_test = ci_cls(np.ascontiguousarray(df.values), **ci_kwargs)
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

    gt = bundle["ground_truth"]
    auroc = compute_pair_auroc(gate_values, gt)

    return (
        G_hat,
        {
            "method": "grace",
            "alpha": alpha,
            "lambda_l0": lambda_l0,
            "skeleton_density": rho_S,
            "n_skeleton_edges": n_skel,
            "runtime_skeleton": round(t_skeleton, 2),
            "runtime_total": round(runtime, 2),
            "AUC_pair": round(auroc, 4) if auroc is not None else None,
        },
        gate_values,
    )


def run_cdnots_on_topology(bundle, alpha=0.05, no_batch=False):
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
    ci_kwargs = {}
    if no_batch and ci_cls.__name__ == "ParCorrGPU":
        ci_kwargs["enable_batching"] = False
    ci_test = ci_cls(np.ascontiguousarray(df.values), **ci_kwargs)
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
    return (
        G_hat,
        {"method": "cdnots", "alpha": alpha, "runtime": round(runtime, 2)},
        None,
    )


def run_pcmci_on_topology(bundle):
    """Run PCMCI with ParCorr."""
    from pcmci_runner import run_pcmci as _run_pcmci

    df = bundle["df"]
    max_lag = bundle["max_lag"]
    df.shape[1]

    t0 = time.time()
    G_hat, _ = _run_pcmci(df.values, max_lag, alpha=0.05, ci_test="parcorr")
    runtime = time.time() - t0
    return G_hat, {"method": "pcmci", "runtime": round(runtime, 2)}, None


def run_cuts_plus_on_topology(bundle, device="mps"):
    """Run CUTS+ on a topology bundle. Requires CUTS+ to be installed separately."""
    import tempfile

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

    auc = float(roc_auc_score(y_true, y_score)) if y_true.sum() > 0 else None

    y_pred = (y_score > 0.5).astype(int)
    f1_p = float(f1_score(y_true, y_pred, zero_division=0))
    prec_p = float(precision_score(y_true, y_pred, zero_division=0))
    tpr_p = float(recall_score(y_true, y_pred, zero_division=0))

    G_pairs = (Graph > 0.5).astype(np.int8)
    np.fill_diagonal(G_pairs, 0)
    G_est = np.zeros((d, d, max_lag + 1), dtype=np.int8)
    for lag in range(1, max_lag + 1):
        G_est[:, :, lag] = G_pairs

    return (
        G_est,
        {
            "method": "cuts_plus",
            "runtime": round(runtime, 2),
            "AUC_pair": round(auc, 4) if auc is not None else None,
            "F1_pair": round(f1_p, 4),
            "Prec_pair": round(prec_p, 4),
            "TPR_pair": round(tpr_p, 4),
        },
        None,
    )


def evaluate_and_print(G_est, G_true, method_name, info):
    """Evaluate and print results."""
    from causalts.grace.gated_discovery import evaluate_graph

    metrics = evaluate_graph(G_est, G_true)
    line = (
        f"    {method_name:15s} F1={metrics['F1']:.3f}  "
        f"Prec={metrics['Precision']:.3f}  TPR={metrics['TPR']:.3f}  "
        f"SHD={metrics['SHD']}  ({info.get('runtime', info.get('runtime_total', 0)):.1f}s)"
    )
    if info.get("AUC_pair") is not None:
        line += f"  AUC={info['AUC_pair']:.3f}"
    print(line)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Topology benchmark")
    parser.add_argument(
        "--topologies",
        nargs="+",
        default=["scale_free", "erdos_renyi", "small_world"],
        choices=["scale_free", "erdos_renyi", "small_world"],
    )
    parser.add_argument("--n_vars", type=int, default=50)
    parser.add_argument("--T", nargs="+", type=int, default=[1000])
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["cdnots", "grace", "pcmci"],
        choices=["cdnots", "grace", "pcmci", "cuts_plus"],
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Disable ParCorrGPU batching (avoids OOM on dense graphs)",
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

    print("Topology Benchmark")
    print(f"  Topologies: {args.topologies}")
    print(f"  d={args.n_vars}, T={args.T}, Seeds: 0-{args.n_seeds-1}")
    print(f"  Methods: {args.methods}")
    print()

    summary_rows = []

    for topo in args.topologies:
        for seed in range(args.n_seeds):
            T_max = max(args.T) + 500
            try:
                bundle_full = topology_scp(
                    topo, seed=seed, n_vars=args.n_vars, T=T_max, max_lag=5
                )
            except RuntimeError as e:
                print(f"  [{topo}] seed={seed}: generation failed: {e}")
                continue

            gt = bundle_full["ground_truth"]
            meta = bundle_full["meta"]

            print(f"\n{'='*60}")
            print(
                f"{topo} | seed={seed} | {meta['n_cross_edges']} cross-edges | sr={meta['spectral_radius']:.3f}"
            )

            for T_val in args.T:
                print(f"\n  --- T={T_val} ---")
                bundle = dict(bundle_full)
                bundle["df"] = bundle_full["df"].iloc[:T_val].copy()

                for method in args.methods:
                    result_path = os.path.join(
                        RESULTS_DIR,
                        f"{topo}_d{args.n_vars}_{method}_seed{seed}_T{T_val}.json",
                    )
                    if os.path.exists(result_path) and not args.no_skip:
                        print(f"    [{method}] exists, skipping")
                        continue

                    try:
                        if method == "grace":
                            grace_kwargs = dict(
                                device=args.device,
                                alpha=args.alpha,
                                no_batch=args.no_batch,
                            )
                            if args.reuse_skeleton:
                                skel_fname = f"{topo}_d{args.n_vars}_cdnots_seed{seed}_T{T_val}.json"
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
                                else:
                                    print(
                                        f"    [grace] SKIPPED (no cdnots skeleton for {topo} seed={seed})"
                                    )
                                    continue
                            G_est, info, gate_vals = run_grace_on_topology(
                                bundle, **grace_kwargs
                            )
                        elif method == "cdnots":
                            G_est, info, gate_vals = run_cdnots_on_topology(
                                bundle, alpha=args.alpha, no_batch=args.no_batch
                            )
                        elif method == "pcmci":
                            G_est, info, gate_vals = run_pcmci_on_topology(bundle)
                        elif method == "cuts_plus":
                            G_est, info, gate_vals = run_cuts_plus_on_topology(
                                bundle, device=args.device
                            )
                        else:
                            continue

                        metrics = evaluate_and_print(G_est, gt, method, info)

                        row = {
                            "topology": topo,
                            "d": args.n_vars,
                            "T": T_val,
                            "seed": seed,
                            "method": method,
                            **metrics,
                            **info,
                        }
                        summary_rows.append(row)

                        result = {
                            "dataset": f"{topo}_d{args.n_vars}",
                            "topology": topo,
                            "method": method,
                            "seed": seed,
                            "T": T_val,
                            "n_vars": args.n_vars,
                            "metrics": metrics,
                            "info": info,
                            "G_est": G_est.tolist(),
                            "meta": meta,
                        }
                        if gate_vals is not None:
                            result["gate_values"] = gate_vals.tolist()

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
            df_summary.groupby(["topology", "d", "T", "method"])["F1"]
            .agg(["mean", "std", "count"])
            .round(3)
        )
        print(pivot.to_string())


if __name__ == "__main__":
    main()
