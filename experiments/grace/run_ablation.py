# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Ablation study for GRACE: skeleton quality, basis expansion, lambda formula.

Reuses data bundles from scaling experiments. Results saved to results/ablation/.

Usage:
    # Run all ablation variants at d=50,100 T=500,1000
    python run_ablation.py \\
        --variants skel_maxdeg1 skel_maxdeg0 no_basis fixed_lambda_0.01 fixed_lambda_0.1 \\
        --n_vars 50 100 --T 500 1000 --n_graphs 10 --device mps

    # Quick local test
    python run_ablation.py --variants no_basis --n_vars 20 --T 300 --n_graphs 1 --device mps
"""

import argparse
import math
import time
import traceback

import numpy as np
import pandas as pd
from result_storage import (
    load_data_bundle,
    load_skeleton,
    result_exists,
    save_result,
)

RESULTS_DIR = "results/ablation"

# ---------------------------------------------------------------------------
# Ablation variant definitions
# ---------------------------------------------------------------------------

ABLATION_VARIANTS = {
    "no_skeleton": {"_no_skeleton": True},
    "skel_maxdeg1": {"_max_degree": 1},
    "no_basis": {"basis_expansion": False},
    "fixed_lambda_0.01": {"_fixed_lambda": 0.01},
    "fixed_lambda_0.05": {"_fixed_lambda": 0.05},
    "fixed_lambda_0.1": {"_fixed_lambda": 0.1},
    "fixed_lambda_0.5": {"_fixed_lambda": 0.5},
}


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------


def run_ablation_variant(
    variant_name,
    bundle,
    device="mps",
    model_seed=42,
    skeleton=None,
    skeleton_runtime=0.0,
):
    """Run a single ablation variant on a data bundle."""
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

    config = dict(ABLATION_VARIANTS[variant_name])
    df = bundle["df"]
    max_lag = bundle["max_lag"]
    num_vars = df.shape[1]

    no_skeleton = config.pop("_no_skeleton", False)
    max_degree = config.pop("_max_degree", None)
    fixed_lambda = config.pop("_fixed_lambda", None)
    model_kwargs = config

    t0 = time.time()

    if no_skeleton:
        skeleton = np.ones((num_vars, num_vars, max_lag + 1), dtype=np.int8)
        for i in range(num_vars):
            skeleton[i, i, 0] = 0
        t_skeleton = 0.0
    elif skeleton is not None and max_degree is None:
        t_skeleton = skeleton_runtime
    else:
        ci_cls = _get_parcorr_class()
        ci_test = ci_cls(np.ascontiguousarray(df.values))

        cdnots_kwargs = dict(
            df=df,
            indep_test=ci_test,
            num_lags=max_lag,
            include_C=True,
            alpha=0.05,
            stable=True,
        )
        if max_degree is not None:
            cdnots_kwargs["max_degree"] = max_degree

        cdnots_result = cdnots_discovery(**cdnots_kwargs)
        if len(cdnots_result) == 3:
            cg, cg_tig, _pvals = cdnots_result
        else:
            cg, cg_tig = cdnots_result

        skeleton = _cdnots_graph_to_binary(cg_tig, num_vars, max_lag)
        t_skeleton = time.time() - t0

    n_skeleton_edges = int(skeleton.sum())
    n_possible = num_vars * num_vars * (max_lag + 1) - num_vars
    skeleton_density = n_skeleton_edges / n_possible if n_possible > 0 else 0.0

    if fixed_lambda is not None:
        lambda_l0 = fixed_lambda
    else:
        lambda_l0 = compute_lambda(num_vars, len(df), skeleton_density)

    dataset = prepare_data(df, max_lag, normalize=True)
    N = len(dataset)
    batch_size = max(32, min(N // 8, 256))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    steps_per_epoch = math.ceil(N / batch_size)

    if model_seed is not None:
        pl.seed_everything(model_seed, workers=True)

    model_kwargs.setdefault("lambda_lag_group", 0.0)

    model = GatedCausalDiscovery(
        num_vars=num_vars,
        max_lag=max_lag,
        lambda_l0=float(lambda_l0),
        normalize_l0=True,
        steps_per_epoch=steps_per_epoch,
        **model_kwargs,
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
    t_training = runtime - t_skeleton

    method_params = {
        "variant": variant_name,
        "lambda_l0": lambda_l0,
        "skeleton_density": skeleton_density,
        "no_skeleton": no_skeleton,
        "max_degree": max_degree,
        "fixed_lambda": fixed_lambda,
        "basis_expansion": model_kwargs.get("basis_expansion", True),
        "gate_threshold": 0.5,
    }
    method_info = {
        "n_skeleton_edges": n_skeleton_edges,
        "runtime_skeleton": round(t_skeleton, 2),
        "runtime_training": round(t_training, 2),
    }
    return G_hat, method_params, method_info, runtime


# ---------------------------------------------------------------------------
# Bundle utilities
# ---------------------------------------------------------------------------


def slice_bundle(bundle, T):
    sliced = dict(bundle)
    sliced["df"] = bundle["df"].iloc[:T].copy()
    return sliced


def _ds_name(n_vars):
    return f"scp_d{n_vars}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="GRACE ablation study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(ABLATION_VARIANTS.keys()),
        choices=list(ABLATION_VARIANTS.keys()),
        help="Ablation variants to run",
    )
    parser.add_argument("--n_vars", nargs="+", type=int, default=[50, 100])
    parser.add_argument("--T", nargs="+", type=int, default=[500, 1000])
    parser.add_argument("--n_graphs", type=int, default=10)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--data-dir", default="results/scaling/data")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument(
        "--reuse-skeleton",
        action="store_true",
        help="Reuse CDNOTS skeleton from scaling results",
    )
    parser.add_argument(
        "--skeleton-dir",
        default="results",
        help="Directory with cdnots results for skeleton reuse",
    )
    args = parser.parse_args()

    summary_rows = []

    print("GRACE Ablation Study")
    print(f"  Variants: {args.variants}")
    print(f"  n_vars:   {args.n_vars}")
    print(f"  T:        {args.T}")
    print(f"  Seeds:    0-{args.n_graphs - 1}")
    print(f"  Device:   {args.device}")
    print()

    from causalts.grace.gated_discovery import evaluate_graph

    for n_vars in args.n_vars:
        for seed in range(args.n_graphs):
            bundle_full = load_data_bundle(n_vars, seed, data_dir=args.data_dir)
            if bundle_full is None:
                print(f"\n[d={n_vars}, seed={seed}] No data bundle found, skipping")
                continue

            print(f"\n{'='*60}")
            print(f"d={n_vars}, seed={seed}")

            gt = bundle_full["ground_truth"]

            for T_val in args.T:
                print(f"\n  --- T={T_val} ---")
                bundle = slice_bundle(bundle_full, T_val)
                ds_name = _ds_name(n_vars)

                cached_skel, cached_skel_rt = None, 0.0
                if args.reuse_skeleton:
                    cached_skel, cached_skel_rt = load_skeleton(
                        ds_name,
                        seed,
                        T_val,
                        skeleton_method="cdnots",
                        results_dir=args.skeleton_dir,
                    )
                    if cached_skel is not None:
                        print(f"    (loaded cdnots skeleton from {args.skeleton_dir})")
                    else:
                        print("    (no cdnots skeleton found, will compute)")

                for variant_name in args.variants:
                    method_name = f"abl_{variant_name}"

                    if result_exists(
                        ds_name, method_name, seed, T_val, results_dir=args.results_dir
                    ):
                        print(f"    [{variant_name}] exists, skipping")
                        continue

                    print(f"    [{variant_name}] ", end="", flush=True)

                    try:
                        abl_kwargs = dict(
                            device=args.device, model_seed=args.model_seed
                        )
                        if cached_skel is not None and variant_name not in (
                            "no_skeleton",
                            "skel_maxdeg1",
                        ):
                            abl_kwargs["skeleton"] = cached_skel
                            abl_kwargs["skeleton_runtime"] = cached_skel_rt
                        G_est, method_params, method_info, runtime = (
                            run_ablation_variant(variant_name, bundle, **abl_kwargs)
                        )

                        metrics = evaluate_graph(G_est, gt)

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
                            method_info=method_info,
                            results_dir=args.results_dir,
                        )

                        row = {
                            "d": n_vars,
                            "T": T_val,
                            "seed": seed,
                            "variant": variant_name,
                            "runtime": runtime,
                            **metrics,
                        }
                        summary_rows.append(row)

                        print(
                            f"F1={metrics['F1']:.3f}  SHD={metrics['SHD']}  "
                            f"Prec={metrics['Precision']:.3f}  "
                            f"TPR={metrics['TPR']:.3f}  ({runtime:.1f}s)"
                        )

                    except Exception as e:
                        print(f"FAILED: {e}")
                        traceback.print_exc()

    if summary_rows:
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        df_summary = pd.DataFrame(summary_rows)
        pivot = (
            df_summary.groupby(["d", "T", "variant"])["F1"]
            .agg(["mean", "std", "count"])
            .round(3)
        )
        print(pivot.to_string())


if __name__ == "__main__":
    main()
