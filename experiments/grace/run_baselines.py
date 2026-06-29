# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run baseline causal discovery methods on synthetic datasets and save results.

Usage:
    python run_baselines.py --methods gated tcdf cdnots pcmci --datasets ex1 ex2 ex3 scp
    python run_baselines.py --datasets ex2 --methods gated   # single run
    python run_baselines.py --datasets ex2 --methods gated --seed 0 --T 500
"""

import argparse
import math
import time
from typing import Optional, Tuple

import numpy as np
from result_storage import save_result

from causalts.synthetic_data.synthetic_datasets import (
    NL_FUNC_NAMES,
    SCPGraphGenerator,
    load_dataset,
)

_parcorr_gpu_warned = False


def _make_parcorr(data):
    """Create ParCorrGPU if CUDA available, else CPU PartialCorr with one-time warning."""
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


# ---------------------------------------------------------------------------
# Dataset configs: (loader_fn, kwargs)
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    "ex1": {"seed": 42, "T": 500},
    "ex2": {"seed": 42, "T": 300},
    "ex3": {"seed": 42, "T": 300},
    "scp": {"seed": 130, "T": 300},
    "scp_nl": {"seed": 130, "T": 300},
    "ex_contemp": {"seed": 42, "T": 500},
    "nonstationary_ex1": {"seed": 32, "T": 500},
}

# Per-dataset overrides for the gated method.
GATED_KWARGS = {
    "ex1": {"max_lags_per_pair": 2},
    "ex_contemp": {"include_instantaneous": True},
}


def _load_bundle(
    name: str, seed: Optional[int] = None, T: Optional[int] = None
) -> dict:
    """Load a dataset bundle, applying optional overrides."""
    cfg = DATASET_CONFIGS[name].copy()
    if seed is not None:
        cfg["seed"] = seed
    if T is not None:
        cfg["T"] = T

    if name == "scp":
        scp = SCPGraphGenerator(n_vars=8, max_lag=5)
        bundle = scp.sample(seed=cfg["seed"], T=cfg["T"], n_links=10, max_tries=200)
    elif name == "scp_nl":
        scp = SCPGraphGenerator(n_vars=8, max_lag=5, dependency_funcs=NL_FUNC_NAMES)
        bundle = scp.sample(seed=cfg["seed"], T=cfg["T"], n_links=10, max_tries=200)
    else:
        bundle = load_dataset(name, **cfg)
    return bundle


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


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


def _string_graph_to_binary(graph_str, n_vars: int, max_lag: int) -> np.ndarray:
    """Convert tigramite/cdnots string graph array to binary [cause, effect, lag]."""
    G = np.zeros((n_vars, n_vars, max_lag + 1), dtype=np.int8)
    for i in range(graph_str.shape[0]):
        for j in range(graph_str.shape[1]):
            for lag in range(min(graph_str.shape[2], max_lag + 1)):
                s = str(graph_str[i, j, lag]).strip()
                if s == "-->":
                    G[i, j, lag] = 1
                elif s == "<--":
                    G[j, i, lag] = 1
    return G


# ---------------------------------------------------------------------------
# Method wrappers — each returns (G_est, method_params, method_info, runtime)
# ---------------------------------------------------------------------------


def run_ci_skeleton_baseline(
    bundle: dict, **kwargs
) -> Tuple[np.ndarray, dict, dict, float]:
    """Run CI skeleton discovery (PartialCorr, max_degree=1) as a standalone baseline."""
    from causalts.grace.gated_discovery import run_ci_skeleton

    df = bundle["df"]
    max_lag = bundle["max_lag"]

    t0 = time.time()
    skeleton = run_ci_skeleton(
        df,
        max_lag=max_lag,
        alpha=0.05,
        normalize=True,
        max_degree=1,
        include_C=True,
        verbose=True,
    )
    runtime = time.time() - t0

    method_params = {
        "alpha": 0.05,
        "max_degree": 1,
        "ci_test": "PartialCorr",
        "include_C": True,
    }
    method_info = {"n_edges": int(skeleton.sum())}
    return skeleton.astype(np.int8), method_params, method_info, runtime


def run_gated(
    bundle: dict, dataset_name: str = "", **kwargs
) -> Tuple[np.ndarray, dict, dict, float]:
    """Run gated causal discovery with stability selection."""
    from causalts.grace import run_stability_selection

    df = bundle["df"]
    max_lag = bundle["max_lag"]
    gt = bundle["ground_truth"]

    defaults = dict(
        normalize_l0=True,
        ground_truth=gt,
        verbose=True,
        lambda_lag_group=0.0,
    )
    defaults.update(GATED_KWARGS.get(dataset_name, {}))
    defaults.update(kwargs)

    t0 = time.time()
    G_hat, scores, selector = run_stability_selection(
        df,
        max_lag=max_lag,
        **defaults,
    )
    runtime = time.time() - t0

    run_gated._last_skeleton = selector.skeleton

    _PARAM_EXCLUDE = {"ground_truth", "ci_test_class"}
    method_params = {
        k: str(v) if isinstance(v, np.ndarray) else v
        for k, v in defaults.items()
        if k not in _PARAM_EXCLUDE
    }
    method_params["use_ci_skeleton"] = selector.skeleton is not None
    method_info = {
        "n_runs": len(selector._selections),
        "expected_fp": selector.expected_false_positives(0.6),
    }
    return G_hat, method_params, method_info, runtime


def run_tcdf(bundle: dict, **kwargs) -> Tuple[np.ndarray, dict, dict, float]:
    """Run TCDF baseline.

    Requires TCDF to be installed: https://github.com/M-Nauta/TCDF
    """
    from TCDF import runTCDF

    df = bundle["df"]
    max_lag = bundle["max_lag"]
    kernel_size = max_lag

    t0 = time.time()
    cg, allcauses, alldelays, allreallosses, allscores, varnames = runTCDF(
        df, kernel_size=kernel_size, significance=0.8, verbose=False
    )
    runtime = time.time() - t0

    G_est = _align_graph_lags(cg.astype(np.int8), max_lag)

    method_params = {"kernel_size": kernel_size, "significance": 0.8}
    method_info = {}
    return G_est, method_params, method_info, runtime


def run_cdnots(bundle: dict, **kwargs) -> Tuple[np.ndarray, dict, dict, float]:
    """Run CD-NOTS baseline with ParCorrGPU."""
    from causalts.cdnots.phase3_utils import cdnots_discovery

    alpha = kwargs.pop("alpha", 0.05)
    df = bundle["df"]
    max_lag = bundle["max_lag"]
    n_vars = df.shape[1]

    ci_test = _make_parcorr(np.ascontiguousarray(df.values))

    t0 = time.time()
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
    runtime = time.time() - t0

    cg_tig_trimmed = cg_tig[:n_vars, :n_vars, :]
    if cg_tig_trimmed.dtype.kind in ("U", "S", "O"):
        G_est = _string_graph_to_binary(cg_tig_trimmed, n_vars, max_lag)
    else:
        G_est = _align_graph_lags(cg_tig_trimmed.astype(np.int8), max_lag)

    method_params = {"alpha": alpha, "ci_test": "PartialCorr", "include_C": True}
    method_info = {}
    return G_est, method_params, method_info, runtime


def run_cdnots_cmiknn(bundle: dict, **kwargs) -> Tuple[np.ndarray, dict, dict, float]:
    """Run CD-NOTS with CMIKNN."""
    from causalts.cdnots.phase3_utils import cdnots_discovery
    from causalts.ci_tests.cmiknn import CMIKNN

    df = bundle["df"]
    max_lag = bundle["max_lag"]
    n_vars = df.shape[1]

    ci_test = CMIKNN(df.values, k=0.2, early_stop=True, sig_samples=200)

    t0 = time.time()
    cdnots_result = cdnots_discovery(
        df=df,
        indep_test=ci_test,
        num_lags=max_lag,
        include_C=True,
        alpha=0.05,
        stable=True,
    )
    if len(cdnots_result) == 3:
        cg, cg_tig, _ = cdnots_result
    else:
        cg, cg_tig = cdnots_result
    runtime = time.time() - t0

    cg_tig_trimmed = cg_tig[:n_vars, :n_vars, :]
    if cg_tig_trimmed.dtype.kind in ("U", "S", "O"):
        G_est = _string_graph_to_binary(cg_tig_trimmed, n_vars, max_lag)
    else:
        G_est = _align_graph_lags(cg_tig_trimmed.astype(np.int8), max_lag)

    method_params = {"alpha": 0.05, "ci_test": "CMIKNN", "include_C": True, "k": 0.2}
    method_info = {}
    return G_est, method_params, method_info, runtime


def run_cdnots_kcit(bundle: dict, **kwargs) -> Tuple[np.ndarray, dict, dict, float]:
    """Run CD-NOTS with KCIGPU."""
    from causalts.cdnots.phase3_utils import cdnots_discovery
    from causalts.ci_tests.kci_gpu import KCIGPU

    df = bundle["df"]
    max_lag = bundle["max_lag"]
    n_vars = df.shape[1]

    ci_test = KCIGPU(df.values, approx="sw")

    t0 = time.time()
    cdnots_result = cdnots_discovery(
        df=df,
        indep_test=ci_test,
        num_lags=max_lag,
        include_C=True,
        alpha=0.05,
        stable=True,
    )
    if len(cdnots_result) == 3:
        cg, cg_tig, _ = cdnots_result
    else:
        cg, cg_tig = cdnots_result
    runtime = time.time() - t0

    cg_tig_trimmed = cg_tig[:n_vars, :n_vars, :]
    if cg_tig_trimmed.dtype.kind in ("U", "S", "O"):
        G_est = _string_graph_to_binary(cg_tig_trimmed, n_vars, max_lag)
    else:
        G_est = _align_graph_lags(cg_tig_trimmed.astype(np.int8), max_lag)

    method_params = {
        "alpha": 0.05,
        "ci_test": "KCIGPU(sw)",
        "include_C": True,
        "approx": "sw",
    }
    method_info = {}
    return G_est, method_params, method_info, runtime


def run_cdnots_rcot(bundle: dict, **kwargs) -> Tuple[np.ndarray, dict, dict, float]:
    """Run CD-NOTS with RCOTGPU."""
    from causalts.cdnots.phase3_utils import cdnots_discovery
    from causalts.ci_tests.pyRcot_gpu import RCOTGPU

    df = bundle["df"]
    max_lag = bundle["max_lag"]
    n_vars = df.shape[1]

    num_f = 100
    num_f2 = 10
    ci_test = RCOTGPU(df.values, num_f=num_f, num_f2=num_f2)

    t0 = time.time()
    cdnots_result = cdnots_discovery(
        df=df,
        indep_test=ci_test,
        num_lags=max_lag,
        include_C=True,
        alpha=0.05,
        stable=True,
    )
    if len(cdnots_result) == 3:
        cg, cg_tig, _ = cdnots_result
    else:
        cg, cg_tig = cdnots_result
    runtime = time.time() - t0

    cg_tig_trimmed = cg_tig[:n_vars, :n_vars, :]
    if cg_tig_trimmed.dtype.kind in ("U", "S", "O"):
        G_est = _string_graph_to_binary(cg_tig_trimmed, n_vars, max_lag)
    else:
        G_est = _align_graph_lags(cg_tig_trimmed.astype(np.int8), max_lag)

    method_params = {
        "alpha": 0.05,
        "ci_test": "RCOTGPU",
        "include_C": True,
        "num_f": num_f,
        "num_f2": num_f2,
    }
    method_info = {}
    return G_est, method_params, method_info, runtime


def _run_pcmci_direct(
    bundle: dict,
    ci_test: str,
    label: str,
    ci_kwargs: dict | None = None,
    extra_params: dict | None = None,
    **kwargs,
) -> Tuple[np.ndarray, dict, dict, float]:
    """Run PCMCI via causalts."""
    from pcmci_runner import run_pcmci as _run_pcmci

    df = bundle["df"]
    max_lag = bundle["max_lag"]

    t0 = time.time()
    G_est, _ = _run_pcmci(
        df.values, max_lag, alpha=0.05, ci_test=ci_test, **(ci_kwargs or {})
    )
    runtime = time.time() - t0

    G_est = _align_graph_lags(G_est, max_lag)

    method_params = {"alpha": 0.05, "ci_test": label}
    if extra_params:
        method_params.update(extra_params)
    method_info = {}
    return G_est, method_params, method_info, runtime


def run_pcmci(bundle: dict, **kwargs) -> Tuple[np.ndarray, dict, dict, float]:
    """Run PCMCI with ParCorr."""
    return _run_pcmci_direct(bundle, ci_test="parcorr", label="parcorr")


def run_pcmci_gpdc(bundle: dict, **kwargs) -> Tuple[np.ndarray, dict, dict, float]:
    """Run PCMCI with GPDC."""
    return _run_pcmci_direct(bundle, ci_test="gpdc", label="GPDC")


def run_pcmci_cmiknn(bundle: dict, **kwargs) -> Tuple[np.ndarray, dict, dict, float]:
    """Run PCMCI with CMIknn."""
    return _run_pcmci_direct(bundle, ci_test="cmiknn", label="CMIknn")


def run_pcmci_kcit(bundle: dict, **kwargs) -> Tuple[np.ndarray, dict, dict, float]:
    """Run PCMCI with KCIT."""
    return _run_pcmci_direct(bundle, ci_test="kcit", label="KCIT(hbe)")


def run_pcmci_rcot(bundle: dict, **kwargs) -> Tuple[np.ndarray, dict, dict, float]:
    """Run PCMCI with RCOT."""
    num_f, num_f2 = 100, 10
    return _run_pcmci_direct(
        bundle,
        ci_test="rcot",
        label="RCOT(hbe)",
        ci_kwargs={"num_f": num_f, "num_f2": num_f2},
        extra_params={"num_f": num_f, "num_f2": num_f2},
    )


def _make_gated_ci(ci_test_name):
    """Create a gated+CI baseline runner for a specific CI test."""

    def runner(bundle, dataset_name="", **kwargs):
        if ci_test_name == "parcorr":
            from causalts.ci_tests.parcorr import PartialCorr

            ci_cls = PartialCorr
        elif ci_test_name == "rcot":
            from causalts.ci_tests.pyRcot_gpu import RCOTGPU

            ci_cls = RCOTGPU
        elif ci_test_name == "kcit":
            from causalts.ci_tests.kci_gpu import KCIGPU

            ci_cls = KCIGPU
        else:
            raise ValueError(f"Unknown CI test: {ci_test_name}")

        G_est, params, info, rt = run_gated(
            bundle,
            dataset_name=dataset_name,
            use_ci_skeleton=True,
            ci_test_class=ci_cls,
            **kwargs,
        )
        params["ci_test"] = ci_test_name
        return G_est, params, info, rt

    return runner


def run_cdnots_gated_baseline(
    bundle: dict, dataset_name: str = "", **kwargs
) -> Tuple[np.ndarray, dict, dict, float]:
    """Run CDNOTS skeleton + single gated refinement."""
    from causalts.grace import run_cdnots_gated

    df = bundle["df"]
    max_lag = bundle["max_lag"]
    gt = bundle["ground_truth"]

    model_seed = kwargs.pop("model_seed", None)

    run_kwargs = dict(
        ground_truth=gt,
        verbose=kwargs.pop("verbose", True),
    )
    run_kwargs.update(kwargs)
    if model_seed is not None:
        run_kwargs["model_seed"] = model_seed

    t0 = time.time()
    G_hat, gate_values, info = run_cdnots_gated(df, max_lag=max_lag, **run_kwargs)
    runtime = time.time() - t0

    method_params = {
        "gate_threshold": info["gate_threshold"],
        "lambda_l0": info["lambda_l0"],
        "skeleton_density": info["skeleton_density"],
    }
    method_info = {
        "n_skeleton_edges": info["n_skeleton_edges"],
        "runtime_skeleton": info["runtime_skeleton"],
        "runtime_training": info["runtime_training"],
    }
    return G_hat, method_params, method_info, runtime


def _make_cdnots_gated(ci_test_name):
    """Create a CDNOTS+gated runner for a specific CI test."""

    def runner(bundle, dataset_name="", **kwargs):
        from causalts.ci_tests.parcorr import PartialCorr
        from causalts.ci_tests.pyRcot_gpu import RCOTGPU

        ci_map = {"parcorr": PartialCorr, "rcot": RCOTGPU}
        ci_cls = ci_map[ci_test_name]

        G, params, info, rt = run_cdnots_gated_baseline(
            bundle,
            dataset_name=dataset_name,
            ci_test_class=ci_cls,
            **kwargs,
        )
        params["ci_test"] = ci_test_name
        return G, params, info, rt

    return runner


def run_pcmci_gated(
    bundle: dict, dataset_name: str = "", **kwargs
) -> Tuple[np.ndarray, dict, dict, float]:
    """Run PCMCI skeleton + GRACE gated refinement."""
    import pytorch_lightning as pl
    from pcmci_runner import run_pcmci as _run_pcmci
    from torch.utils.data import DataLoader

    from causalts.grace.gated_discovery import (
        GatedCausalDiscovery,
        _detect_accelerator,
        _silence_lightning,
        compute_lambda,
        prepare_data,
    )

    df = bundle["df"]
    max_lag = bundle["max_lag"]
    num_vars = df.shape[1]
    model_seed = kwargs.pop("model_seed", None)
    precomputed_skeleton = kwargs.pop("skeleton", None)
    precomputed_skeleton_runtime = kwargs.pop("skeleton_runtime", 0.0)

    t0 = time.time()

    if precomputed_skeleton is not None:
        skeleton = precomputed_skeleton
        t_skeleton = precomputed_skeleton_runtime
    else:
        G_pcmci, _ = _run_pcmci(df.values, max_lag, alpha=0.05, ci_test="parcorr")
        skeleton = _align_graph_lags(G_pcmci, max_lag)
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

    device = kwargs.pop("device", None)
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

    method_params = {
        "gate_threshold": 0.5,
        "lambda_l0": lambda_l0,
        "skeleton_density": rho_S,
    }
    method_info = {
        "n_skeleton_edges": n_skel,
        "runtime_skeleton": round(t_skeleton, 2),
        "runtime_total": round(runtime, 2),
    }
    return G_hat, method_params, method_info, runtime


def run_cuts_plus(bundle: dict, **kwargs) -> Tuple[np.ndarray, dict, dict, float]:
    """Run CUTS+ baseline. Requires CUTS+ to be installed separately."""
    import tempfile

    from cuts_plus.cuts_plus import execute_cutplus

    df = bundle["df"]
    max_lag = bundle["max_lag"]
    data = df.values.astype(np.float32)
    d = data.shape[1]

    total_epoch = kwargs.pop("total_epoch", 64)
    seed = kwargs.pop("seed", 42)
    threshold = kwargs.pop("threshold", 0.5)
    device = kwargs.pop("device", "cuda")

    options = {"total_epoch": total_epoch, "input_step": max_lag}

    with tempfile.TemporaryDirectory() as tmpdir:
        t0 = time.time()
        Graph = execute_cutplus(
            data, device=device, log_dir=tmpdir, seed=seed, options=options
        )
        runtime = time.time() - t0

    G_pairs = (Graph > threshold).astype(np.int8)
    np.fill_diagonal(G_pairs, 0)

    G_est = np.zeros((d, d, max_lag + 1), dtype=np.int8)
    for lag in range(1, max_lag + 1):
        G_est[:, :, lag] = G_pairs

    method_params = {
        "total_epoch": total_epoch,
        "threshold": threshold,
        "seed": seed,
        "input_step": max_lag,
        "device": device,
    }
    method_info = {
        "n_pairs": int(G_pairs.sum()),
        "graph_probs_mean": float(Graph.mean()),
        "graph_probs_min": float(Graph.min()),
        "graph_probs_max": float(Graph.max()),
    }
    return G_est, method_params, method_info, runtime


def run_dynotears(bundle: dict, **kwargs) -> Tuple[np.ndarray, dict, dict, float]:
    """Run DYNOTEARS baseline (L1-penalized VAR via LassoCV)."""
    from sklearn.linear_model import LassoCV

    df = bundle["df"]
    max_lag = bundle["max_lag"]
    data = df.values.astype(np.float64)
    T, d = data.shape

    X_lag = np.column_stack(
        [data[max_lag - lag : T - lag, :] for lag in range(1, max_lag + 1)]
    )
    Y = data[max_lag:, :]

    G_est = np.zeros((d, d, max_lag + 1), dtype=np.int8)
    all_coefs = np.zeros((d, d, max_lag))
    alphas_used = []

    t0 = time.time()
    for e in range(d):
        lasso = LassoCV(cv=5, max_iter=10000, n_jobs=-1)
        lasso.fit(X_lag, Y[:, e])
        alphas_used.append(float(lasso.alpha_))

        coefs = lasso.coef_.reshape(max_lag, d)
        for lag_idx in range(max_lag):
            for c in range(d):
                all_coefs[c, e, lag_idx] = coefs[lag_idx, c]
                if abs(coefs[lag_idx, c]) > 0:
                    G_est[c, e, lag_idx + 1] = 1
    runtime = time.time() - t0

    method_params = {
        "cv_folds": 5,
        "max_iter": 10000,
        "mean_alpha": float(np.mean(alphas_used)),
    }
    method_info = {
        "n_edges": int(G_est.sum()),
        "alphas": alphas_used,
        "coef_matrix": all_coefs.tolist(),
    }
    return G_est, method_params, method_info, runtime


def run_dynotears_gated(
    bundle: dict,
    dataset_name: str = "",
    skeleton_threshold: float = 0.0,
    preloaded_coef_matrix: np.ndarray = None,
    **kwargs,
) -> Tuple[np.ndarray, dict, dict, float]:
    """Run DYNOTEARS skeleton + GRACE gated refinement."""
    from causalts.grace import run_cdnots_gated

    df = bundle["df"]
    max_lag = bundle["max_lag"]
    gt = bundle["ground_truth"]
    d = df.shape[1]

    model_seed = kwargs.pop("model_seed", None)

    if preloaded_coef_matrix is not None:
        coef = preloaded_coef_matrix
        dyno_runtime = 0.0
        dyno_mean_alpha = float("nan")
    else:
        _, dyno_params, dyno_info, dyno_runtime = run_dynotears(bundle)
        coef = np.array(dyno_info["coef_matrix"])
        dyno_mean_alpha = dyno_params["mean_alpha"]

    G_dyno = np.zeros((d, d, max_lag + 1), dtype=np.int8)
    G_dyno[:, :, 1 : 1 + coef.shape[2]] = (np.abs(coef) > skeleton_threshold).astype(
        np.int8
    )
    skeleton = G_dyno

    run_kwargs = dict(
        ground_truth=gt,
        verbose=kwargs.pop("verbose", True),
        skeleton=skeleton,
        skeleton_runtime=dyno_runtime,
    )
    run_kwargs.update(kwargs)
    if model_seed is not None:
        run_kwargs["model_seed"] = model_seed

    G_hat, gate_values, info = run_cdnots_gated(df, max_lag=max_lag, **run_kwargs)

    method_params = {
        "gate_threshold": info["gate_threshold"],
        "lambda_l0": info["lambda_l0"],
        "skeleton_density": info["skeleton_density"],
        "skeleton_source": "dynotears",
        "skeleton_threshold": skeleton_threshold,
        "dynotears_mean_alpha": dyno_mean_alpha,
    }
    method_info = {
        "n_skeleton_edges": info["n_skeleton_edges"],
        "runtime_skeleton": info["runtime_skeleton"],
        "runtime_training": info["runtime_training"],
    }
    return G_hat, method_params, method_info, info["runtime"]


def _make_dynotears_gated(skeleton_threshold: float):
    """Factory for dynotears_gated with a fixed skeleton threshold."""

    def _run(bundle, **kwargs):
        return run_dynotears_gated(
            bundle, skeleton_threshold=skeleton_threshold, **kwargs
        )

    _run.__name__ = f"run_dynotears_gated_t{str(skeleton_threshold).replace('.','')}"
    return _run


METHOD_MAP = {
    "ci_skeleton": run_ci_skeleton_baseline,
    "gated": run_gated,
    "gated_ci": _make_gated_ci("parcorr"),
    "gated_ci_rcot": _make_gated_ci("rcot"),
    "gated_ci_kcit": _make_gated_ci("kcit"),
    "cdnots_gated": run_cdnots_gated_baseline,
    "cdnots_rcot_gated": _make_cdnots_gated("rcot"),
    "tcdf": run_tcdf,
    "cdnots": run_cdnots,
    "cdnots_kcit": run_cdnots_kcit,
    "cdnots_rcot": run_cdnots_rcot,
    "pcmci": run_pcmci,
    "pcmci_gated": run_pcmci_gated,
    "pcmci_gpdc": run_pcmci_gpdc,
    "pcmci_cmiknn": run_pcmci_cmiknn,
    "pcmci_kcit": run_pcmci_kcit,
    "pcmci_rcot": run_pcmci_rcot,
    "cuts_plus": run_cuts_plus,
    "dynotears": run_dynotears,
    "dynotears_gated": run_dynotears_gated,
}


def run_cdnots_gated_cv(bundle, **kwargs):
    """CDNOTS-Gated with cross-validation lambda selection."""
    return run_cdnots_gated_baseline(bundle, lambda_cv=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Run causal discovery baselines")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHOD_MAP.keys()),
        choices=list(METHOD_MAP.keys()),
        help="Methods to run",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_CONFIGS.keys()),
        choices=list(DATASET_CONFIGS.keys()),
        help="Datasets to run on",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument(
        "--T", type=int, default=None, help="Override time series length"
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=None,
        help="Seed for model weight init & training",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Output directory"
    )
    args = parser.parse_args()

    print(f"Datasets: {args.datasets}")
    print(f"Methods:  {args.methods}")
    if args.model_seed is not None:
        print(f"Model seed: {args.model_seed}")
    print(f"Results → {args.results_dir}/")
    print()

    for ds_name in args.datasets:
        print(f"{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        bundle = _load_bundle(ds_name, seed=args.seed, T=args.T)
        df = bundle["df"]
        gt = bundle["ground_truth"]
        max_lag = bundle["max_lag"]
        seed = args.seed or DATASET_CONFIGS[ds_name]["seed"]
        T_actual = len(df)

        n_vars = df.shape[1]
        print(f"  n={T_actual}, d={n_vars}, seed={seed}, max_lag={max_lag}")
        print(f"  Ground truth edges: {int(gt.sum())}")
        print()

        for method_name in args.methods:
            print(f"  --- {method_name} ---")
            fn = METHOD_MAP[method_name]

            try:
                extra_kwargs = {}
                if args.model_seed is not None and (
                    method_name.startswith("gated")
                    or method_name in ("cdnots_gated", "cdnots_rcot_gated")
                ):
                    extra_kwargs["model_seed"] = args.model_seed
                G_est, method_params, method_info, runtime = fn(
                    bundle, dataset_name=ds_name, **extra_kwargs
                )

                path = save_result(
                    dataset=ds_name,
                    seed=seed,
                    T=T_actual,
                    n_vars=n_vars,
                    max_lag=max_lag,
                    method=method_name,
                    method_params=method_params,
                    G_est=G_est,
                    G_true=gt,
                    runtime_seconds=runtime,
                    method_info=method_info,
                    results_dir=args.results_dir,
                )
                print(f"  Saved → {path}  ({runtime:.1f}s)")
            except Exception as e:
                print(f"  FAILED: {e}")
            print()


if __name__ == "__main__":
    main()
