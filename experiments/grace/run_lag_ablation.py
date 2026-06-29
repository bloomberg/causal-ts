# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Lag misspecification ablation.

Generates SCP data with true max_lag=5, then runs CDNOTS-Gated, CDNOTS, and
PCMCI with a range of assumed max_lag values.  Results go to results/lag_ablation/.

Individual files are saved per (seed, method, lag) so runs can be resumed.
Already-completed runs are skipped unless --no-skip is passed.

Usage:
    python run_lag_ablation.py
    python run_lag_ablation.py --d 50 --T 1000 --n_graphs 10
    python run_lag_ablation.py --methods pcmci   # add pcmci only
    python run_lag_ablation.py --no-skip         # rerun everything
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from causalts.synthetic_data.synthetic_datasets import SCPGraphGenerator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_parcorr_gpu_warned = False


def _make_parcorr(data):
    global _parcorr_gpu_warned
    try:
        import torch

        if torch.cuda.is_available():
            from causalts.ci_tests.parcorr_gpu import ParCorrGPU

            return ParCorrGPU(data)
    except ImportError:
        pass
    if not _parcorr_gpu_warned:
        import warnings

        warnings.warn("ParCorrGPU requires CUDA; falling back to CPU PartialCorr")
        _parcorr_gpu_warned = True
    from causalts.ci_tests.parcorr import PartialCorr

    return PartialCorr(data)


def _align_graph_lags(G: np.ndarray, target_max_lag: int) -> np.ndarray:
    """Pad or truncate lag dimension to match target_max_lag+1."""
    d1, d2, L = G.shape
    target_L = target_max_lag + 1
    if L == target_L:
        return G
    if L < target_L:
        pad = np.zeros((d1, d2, target_L - L), dtype=G.dtype)
        return np.concatenate([G, pad], axis=2)
    return G[:, :, :target_L]


def _expand_gt(gt: np.ndarray, max_lag_search: int) -> np.ndarray:
    return _align_graph_lags(gt, max_lag_search)


# ---------------------------------------------------------------------------
# Per-method runners
# ---------------------------------------------------------------------------


def run_cdnots_gated(bundle: dict, max_lag_search: int, **kwargs) -> tuple:
    from causalts.grace import run_cdnots_gated as _run

    df = bundle["df"]
    t0 = time.time()
    G_hat, gate_values, info = _run(df, max_lag=max_lag_search, verbose=False, **kwargs)
    rt = time.time() - t0
    params = {"max_lag_search": max_lag_search, "method": "cdnots_gated"}
    return G_hat, params, rt


def run_cdnots(bundle: dict, max_lag_search: int, **kwargs) -> tuple:
    from causalts.cdnots.phase3_utils import cdnots_discovery

    df = bundle["df"]
    n_vars = df.shape[1]
    ci_test = _make_parcorr(np.ascontiguousarray(df.values))

    t0 = time.time()
    cdnots_result = cdnots_discovery(
        df=df,
        indep_test=ci_test,
        num_lags=max_lag_search,
        include_C=True,
        alpha=0.05,
        stable=True,
    )
    rt = time.time() - t0
    if len(cdnots_result) == 3:
        cg, cg_tig, _ = cdnots_result
    else:
        cg, cg_tig = cdnots_result

    cg_trimmed = cg_tig[:n_vars, :n_vars, :]
    if cg_trimmed.dtype.kind in ("U", "S", "O"):
        from run_baselines import _string_graph_to_binary

        G_est = _string_graph_to_binary(cg_trimmed, n_vars, max_lag_search)
    else:
        G_est = _align_graph_lags(cg_trimmed.astype(np.int8), max_lag_search)

    params = {"max_lag_search": max_lag_search, "method": "cdnots"}
    return G_est, params, rt


def run_pcmci(bundle: dict, max_lag_search: int, **kwargs) -> tuple:
    from pcmci_runner import run_pcmci as _run_pcmci

    df = bundle["df"]
    t0 = time.time()
    G_est, _ = _run_pcmci(df.values, max_lag=max_lag_search, alpha=0.05)
    rt = time.time() - t0
    G_est = _align_graph_lags(G_est.astype(np.int8), max_lag_search)
    params = {"max_lag_search": max_lag_search, "method": "pcmci"}
    return G_est, params, rt


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _f1(G_est: np.ndarray, G_true_search: np.ndarray) -> dict:
    tp = int((G_est & G_true_search).sum())
    fp = int((G_est & ~G_true_search).sum())
    fn = int((~G_est & G_true_search).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"f1": f1, "precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# Individual file helpers
# ---------------------------------------------------------------------------


def _record_path(
    results_dir: str, d: int, T: int, method: str, seed: int, lag: int
) -> str:
    fname = f"lag_abl_{method}_seed{seed}_lag{lag}_d{d}_T{T}.json"
    return os.path.join(results_dir, fname)


def _record_exists(
    results_dir: str, d: int, T: int, method: str, seed: int, lag: int
) -> bool:
    return os.path.exists(_record_path(results_dir, d, T, method, seed, lag))


def _save_record(record: dict, results_dir: str) -> str:
    path = _record_path(
        results_dir,
        record["d"],
        record["T"],
        record["method"],
        record["seed"],
        record["max_lag_search"],
    )
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path


def _load_all_records(results_dir: str, d: int, T: int) -> list:
    """Load all individual lag-ablation records for this (d, T) config."""
    records = []
    for fname in os.listdir(results_dir):
        if fname.startswith("lag_abl_") and f"_d{d}_T{T}.json" in fname:
            with open(os.path.join(results_dir, fname)) as f:
                records.append(json.load(f))
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

METHODS = {
    "cdnots_gated": run_cdnots_gated,
    "cdnots": run_cdnots,
    "pcmci": run_pcmci,
}

TRUE_MAX_LAG = 5
SEARCH_LAGS = [2, 3, 5, 7, 10]


def main():
    parser = argparse.ArgumentParser(description="Lag misspecification ablation")
    parser.add_argument("--d", type=int, default=50)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--n_graphs", type=int, default=10)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS.keys()),
        choices=list(METHODS.keys()),
    )
    parser.add_argument("--search_lags", nargs="+", type=int, default=SEARCH_LAGS)
    parser.add_argument("--results_dir", default="results/lag_ablation")
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Rerun and overwrite already-completed results",
    )
    args = parser.parse_args()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    gen = SCPGraphGenerator(n_vars=args.d, max_lag=TRUE_MAX_LAG)

    for seed in range(args.n_graphs):
        print(f"\n=== Seed {seed} | d={args.d} T={args.T} ===")
        bundle = None

        for method_name in args.methods:
            runner = METHODS[method_name]
            for lag_search in args.search_lags:
                if not args.no_skip and _record_exists(
                    args.results_dir, args.d, args.T, method_name, seed, lag_search
                ):
                    print(f"  {method_name:18s} lag={lag_search:2d}  [skip]")
                    continue

                if bundle is None:
                    bundle = gen.sample(seed=seed, T=args.T, max_tries=500)

                try:
                    G_est, params, rt = runner(
                        bundle, max_lag_search=lag_search, model_seed=seed
                    )
                    gt = bundle["ground_truth"]
                    gt_search = _expand_gt(gt, lag_search).astype(bool)
                    metrics = _f1(G_est.astype(bool), gt_search)
                    record = {
                        "seed": seed,
                        "d": args.d,
                        "T": args.T,
                        "true_max_lag": TRUE_MAX_LAG,
                        "max_lag_search": lag_search,
                        "method": method_name,
                        "runtime": round(rt, 2),
                        **metrics,
                    }
                    path = _save_record(record, args.results_dir)
                    print(
                        f"  {method_name:18s} lag={lag_search:2d}  "
                        f"F1={metrics['f1']:.3f}  P={metrics['precision']:.3f}  "
                        f"R={metrics['recall']:.3f}  ({rt:.1f}s)  → {os.path.basename(path)}"
                    )
                except Exception as e:
                    print(f"  {method_name} lag={lag_search} FAILED: {e}")

    all_records = _load_all_records(args.results_dir, args.d, args.T)
    out_path = Path(args.results_dir) / f"lag_ablation_d{args.d}_T{args.T}.json"
    with open(out_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\nAggregated {len(all_records)} records → {out_path}")

    import pandas as pd

    df_res = pd.DataFrame(all_records)
    summary = (
        df_res.groupby(["method", "max_lag_search"])["f1"].agg(["mean", "std"]).round(3)
    )
    print("\nF1 summary (mean ± std):")
    print(summary.to_string())


if __name__ == "__main__":
    main()
