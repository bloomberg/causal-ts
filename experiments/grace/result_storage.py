# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Experiment result storage as local JSON files."""

import glob as globmod
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from causalts.grace.gated_discovery import evaluate_graph


def _ndarray_to_list(obj):
    """Recursively convert numpy arrays/scalars to JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _ndarray_to_list(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_ndarray_to_list(v) for v in obj]
    return obj


def _edges_from_graph(G: np.ndarray) -> List[List[int]]:
    """Extract [cause, effect, lag] triples from binary graph."""
    edges = []
    it = np.nditer(G, flags=["multi_index"])
    while not it.finished:
        if int(it[0]) == 1:
            edges.append(list(it.multi_index))
        it.iternext()
    return edges


def save_result(
    dataset: str,
    seed: int,
    T: int,
    n_vars: int,
    max_lag: int,
    method: str,
    method_params: Dict[str, Any],
    G_est: np.ndarray,
    G_true: np.ndarray,
    runtime_seconds: float,
    method_info: Optional[Dict[str, Any]] = None,
    results_dir: str = "results",
) -> str:
    """Save experiment result as JSON. Returns the path to the saved file."""
    os.makedirs(results_dir, exist_ok=True)

    metrics = evaluate_graph(G_est, G_true)

    record = {
        "dataset": dataset,
        "seed": seed,
        "T": T,
        "n_vars": n_vars,
        "max_lag": max_lag,
        "method": method,
        "method_params": _ndarray_to_list(method_params),
        "metrics": _ndarray_to_list(metrics),
        "discovered_edges": _edges_from_graph(G_est),
        "ground_truth_edges": _edges_from_graph(G_true),
        "graph_estimated": G_est.tolist(),
        "graph_true": G_true.tolist(),
        "timestamp": datetime.now().isoformat(),
        "runtime_seconds": round(runtime_seconds, 3),
        "method_info": _ndarray_to_list(method_info or {}),
    }

    params_key = json.dumps(_ndarray_to_list(method_params), sort_keys=True)
    params_hash = hashlib.md5(params_key.encode()).hexdigest()[:6]
    fname = f"{dataset}_{method}_seed{seed}_T{T}_{params_hash}.json"
    path = os.path.join(results_dir, fname)
    with open(path, "w") as f:
        json.dump(record, f, indent=2)

    return path


def load_result(path: str) -> dict:
    """Load a single result JSON."""
    with open(path) as f:
        return json.load(f)


def load_all_results(
    results_dir: str = "results",
    dataset: Optional[str] = None,
    method: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[dict]:
    """Load all result JSONs, optionally filtered by dataset, method, and/or seed."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return []

    results = []
    for p in sorted(results_path.glob("*.json")):
        rec = load_result(str(p))
        if dataset and rec.get("dataset") != dataset:
            continue
        if method and rec.get("method") != method:
            continue
        if seed is not None and rec.get("seed") != seed:
            continue
        results.append(rec)
    return results


def load_skeleton(
    dataset: str,
    seed: int,
    T: int,
    skeleton_method: str = "cdnots",
    results_dir: str = "results",
) -> tuple:
    """Load a pre-computed skeleton from an existing local result file.

    Returns (skeleton, runtime) or (None, None) if not found.
    """
    results = load_all_results(
        results_dir, dataset=dataset, method=skeleton_method, seed=seed
    )
    for r in results:
        if r.get("T") == T:
            skeleton = np.array(r["graph_estimated"], dtype=np.int8)
            runtime = r.get("runtime_seconds", 0.0)
            return skeleton, runtime

    return None, None


def load_coef_matrix(
    dataset: str,
    seed: int,
    T: int,
    results_dir: str = "results",
) -> tuple:
    """Load the DYNOTEARS coefficient matrix from an existing dynotears result.

    Returns (coef_matrix shape (d,d,max_lag), runtime_seconds) or (None, None).
    """

    def _extract(r):
        coef = r.get("method_info", {}).get("coef_matrix")
        if coef is None:
            return None, None
        return np.array(coef, dtype=np.float64), r.get("runtime_seconds", 0.0)

    results = load_all_results(
        results_dir, dataset=dataset, method="dynotears", seed=seed
    )
    for r in results:
        if r.get("T") == T:
            return _extract(r)

    return None, None


def _method_detail(rec: dict) -> str:
    """Build a short descriptor string from method_params/method_info."""
    params = rec.get("method_params", {})
    method = rec.get("method", "")

    parts = []
    if method == "gated":
        ci = params.get("use_ci_skeleton", False)
        mlp = params.get("max_lags_per_pair")
        if ci or ci == "True" or (isinstance(mlp, str) and "CI" in mlp):
            parts.append("ci=yes")
        else:
            parts.append("ci=no")
        if mlp is not None:
            mlp_str = str(mlp).replace("auto (from CI skeleton)", "auto")
            parts.append(f"mlp={mlp_str}")
    elif method in (
        "cdnots",
        "cdnots_kcit",
        "cdnots_rcot",
        "ci_skeleton",
        "pcmci",
        "pcmci_gpdc",
        "pcmci_cmiknn",
        "pcmci_kcit",
        "pcmci_rcot",
    ):
        ci_test = params.get("ci_test", "")
        if ci_test:
            parts.append(ci_test)
        alpha = params.get("alpha")
        if alpha is not None:
            parts.append(f"α={alpha}")
        num_f = params.get("num_f")
        if num_f is not None:
            parts.append(f"f={num_f}")
        num_f2 = params.get("num_f2")
        if num_f2 is not None:
            parts.append(f"f2={num_f2}")
    elif method == "tcdf":
        sig = params.get("significance")
        if sig is not None:
            parts.append(f"sig={sig}")

    return ", ".join(parts) if parts else ""


def results_to_dataframe(results: List[dict]) -> pd.DataFrame:
    """Convert list of result dicts to a comparison DataFrame."""
    rows = []
    for rec in results:
        T = rec["T"]
        d = rec["n_vars"]
        s = rec["seed"]
        row = {
            "dataset": rec["dataset"],
            "data_desc": f"n={T},d={d},seed={s}",
            "method": rec["method"],
            "detail": _method_detail(rec),
            "seed": s,
            "T": T,
            "n_vars": d,
            "max_lag": rec["max_lag"],
            "runtime_s": rec.get("runtime_seconds", None),
            "runtime_skeleton": rec.get("method_info", {}).get(
                "runtime_skeleton", None
            ),
            "runtime_training": rec.get("method_info", {}).get(
                "runtime_training", None
            ),
        }
        metrics = rec.get("metrics", {})
        if "graph_estimated" in rec and "graph_true" in rec:
            G_est = np.asarray(rec["graph_estimated"])
            G_true = np.asarray(rec["graph_true"])
            fresh = evaluate_graph(G_est, G_true)
            for k in ("F1_pair", "TPR_pair", "FPR_pair", "Precision_pair", "SHD_pair"):
                metrics[k] = fresh[k]
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def result_exists(dataset, method, seed, T, results_dir="results"):
    """Check if a result file already exists locally."""
    pattern = os.path.join(results_dir, f"{dataset}_{method}_seed{seed}_T{T}_*.json")
    return bool(globmod.glob(pattern))


def save_data_bundle(
    bundle, n_vars, seed, nonlinear=False, dense=False, data_dir="results/scaling/data"
):
    """Save a data bundle as .npz. Returns the local path."""
    os.makedirs(data_dir, exist_ok=True)
    prefix = "scp_nl" if nonlinear else "scp"
    if dense:
        prefix += "_dense"
    fname = f"{prefix}_d{n_vars}_seed{seed}.npz"
    path = os.path.join(data_dir, fname)

    df = bundle["df"]
    save_dict = {
        "data": df.values.astype(np.float64),
        "ground_truth": bundle["ground_truth"].astype(np.int8),
        "max_lag": np.array([bundle["max_lag"]]),
    }
    var_names = bundle.get("var_names")
    if var_names is not None:
        save_dict["var_names"] = np.array(var_names, dtype=object)
    meta = bundle.get("meta", {})
    if meta:
        save_dict["meta_json"] = np.array([json.dumps(meta)])

    np.savez_compressed(path, **save_dict)
    return path


def load_data_bundle(
    n_vars, seed, nonlinear=False, dense=False, data_dir="results/scaling/data"
):
    """Load a data bundle from local storage. Returns dict or None."""
    prefix = "scp_nl" if nonlinear else "scp"
    if dense:
        prefix += "_dense"
    fname = f"{prefix}_d{n_vars}_seed{seed}.npz"
    local_path = os.path.join(data_dir, fname)

    if not os.path.exists(local_path):
        return None

    npz = np.load(local_path, allow_pickle=True)
    data = npz["data"]
    gt = npz["ground_truth"]
    max_lag = int(npz["max_lag"][0])

    if "var_names" in npz:
        var_names = list(npz["var_names"])
    else:
        var_names = [f"X{i}" for i in range(data.shape[1])]

    meta = {}
    if "meta_json" in npz:
        meta = json.loads(str(npz["meta_json"][0]))

    return {
        "df": pd.DataFrame(data, columns=var_names),
        "max_lag": max_lag,
        "ground_truth": gt,
        "var_names": var_names,
        "meta": meta,
        "links": None,
        "noises": None,
        "weights": None,
    }
