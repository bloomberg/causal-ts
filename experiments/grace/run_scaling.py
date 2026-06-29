# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Scaling experiments for causal discovery methods.

Generates SCP problems at specified (d, T) and runs selected methods.
Uses prefix slicing: for each (d, graph_seed), data is generated once at T_max
and sliced to each target T, isolating the pure effect of sample size.

Subcommands:
    generate  -- Phase 1: generate and save data bundles only
    run       -- Phase 2: run methods on existing (or freshly generated) data

Legacy (no subcommand) works as before but with try-catch + skip-existing.

Usage:
    # Generate data bundles
    python run_scaling.py generate --n_vars 5 10 20 30 50 --n_graphs 10

    # Run methods on existing data
    python run_scaling.py run --n_vars 5 10 20 --T 300 500 1000 --methods gated pcmci --n_graphs 3

    # Legacy (backward-compatible)
    python run_scaling.py --n_vars 5 10 20 --T 300 500 1000 --methods gated pcmci --n_graphs 3
"""

import argparse
import os
import signal
import time
import traceback

import numpy as np
import pandas as pd
from result_storage import (
    load_coef_matrix,
    load_data_bundle,
    load_skeleton,
    result_exists,
    save_data_bundle,
    save_result,
)
from run_baselines import (
    _align_graph_lags,
    _make_dynotears_gated,
    run_cdnots,
    run_cdnots_gated_baseline,
    run_cdnots_gated_cv,
    run_cdnots_kcit,
    run_cdnots_rcot,
    run_dynotears,
    run_dynotears_gated,
    run_pcmci,
    run_pcmci_gated,
    run_pcmci_kcit,
    run_pcmci_rcot,
    run_tcdf,
)

from causalts.synthetic_data.synthetic_datasets import (
    DEFAULT_FUNC_NAMES,
    NL_FUNC_NAMES,
    SCPGraphGenerator,
)

# Margin above max requested T for generation
T_MAX_MARGIN = 500

# Map gated methods to their skeleton source method
SKELETON_SOURCE = {
    "cdnots_gated": "cdnots",
    "cdnots_rcot_gated": "cdnots_rcot",
    "pcmci_gated": "pcmci",
    "dynotears_gated": "dynotears",
    "dynotears_gated_t005": "dynotears",
    "cdnots_gated_cv": "cdnots",
}


def generate_scp(n_vars, T_max, seed, nonlinear=False, dense=False, max_tries=500):
    """Generate an SCP problem at T_max. Caller slices prefixes for each T."""
    funcs = NL_FUNC_NAMES if nonlinear else DEFAULT_FUNC_NAMES
    density = "dense" if dense else "sparse"
    gen = SCPGraphGenerator(
        n_vars=n_vars, max_lag=5, dependency_funcs=funcs, density=density
    )
    bundle = gen.sample(seed=seed, T=T_max, n_links=None, max_tries=max_tries)
    return bundle


def slice_bundle(bundle, T):
    """Create a new bundle with data sliced to first T rows."""
    sliced = dict(bundle)
    sliced["df"] = bundle["df"].iloc[:T].copy()
    return sliced


def run_gated_method(bundle, model_seed=42, **kwargs):
    from causalts.grace import run_stability_selection

    df = bundle["df"]
    max_lag = bundle["max_lag"]
    gt = bundle["ground_truth"]

    defaults = dict(
        normalize_l0=True,
        ground_truth=gt,
        verbose=True,
        lambda_lag_group=0.0,
        model_seed=model_seed,
    )
    defaults.update(kwargs)

    t0 = time.time()
    G_hat, scores, selector = run_stability_selection(df, max_lag=max_lag, **defaults)
    runtime = time.time() - t0

    _PARAM_EXCLUDE = {"ground_truth", "ci_test_class"}
    method_params = {
        k: str(v) if isinstance(v, np.ndarray) else v
        for k, v in defaults.items()
        if k not in _PARAM_EXCLUDE
    }
    method_params["use_ci_skeleton"] = defaults.get("use_ci_skeleton", False)
    method_info = {
        "n_runs": len(selector._selections),
        "expected_fp": selector.expected_false_positives(0.6),
    }
    return G_hat, method_params, method_info, runtime


def _make_gated_ci_runner(ci_test_name):
    """Create a gated+CI method runner for a specific CI test."""

    def runner(bundle, model_seed=42, **kwargs):
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

        G_hat, method_params, method_info, runtime = run_gated_method(
            bundle,
            model_seed=model_seed,
            use_ci_skeleton=True,
            ci_test_class=ci_cls,
            **kwargs,
        )
        method_params["ci_test"] = ci_test_name
        return G_hat, method_params, method_info, runtime

    return runner


def run_cdnots_gated_method(bundle, model_seed=42, **kwargs):
    return run_cdnots_gated_baseline(
        bundle, model_seed=model_seed, verbose=False, **kwargs
    )


def run_cdnots_rcot_gated_method(bundle, model_seed=42, **kwargs):
    from causalts.ci_tests.pyRcot_gpu import RCOTGPU

    return run_cdnots_gated_baseline(
        bundle, model_seed=model_seed, verbose=False, ci_test_class=RCOTGPU, **kwargs
    )


SCALING_METHODS = {
    "gated": run_gated_method,
    "gated_ci": _make_gated_ci_runner("parcorr"),
    "gated_ci_rcot": _make_gated_ci_runner("rcot"),
    "gated_ci_kcit": _make_gated_ci_runner("kcit"),
    "pcmci": run_pcmci,
    "pcmci_gated": run_pcmci_gated,
    "pcmci_kcit": run_pcmci_kcit,
    "pcmci_rcot": run_pcmci_rcot,
    "cdnots": run_cdnots,
    "cdnots_kcit": run_cdnots_kcit,
    "cdnots_gated": run_cdnots_gated_method,
    "cdnots_rcot_gated": run_cdnots_rcot_gated_method,
    "cdnots_rcot": run_cdnots_rcot,
    "tcdf": run_tcdf,
    "dynotears": run_dynotears,
    "dynotears_gated": run_dynotears_gated,
    "dynotears_gated_t005": _make_dynotears_gated(0.05),
    "cdnots_gated_cv": run_cdnots_gated_cv,
}


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Method timed out")


def evaluate_graph(G_est, G_true):
    from causalts.grace.gated_discovery import evaluate_graph as _eval

    return _eval(G_est, G_true)


def _ds_name(n_vars, nonlinear, dense=False):
    prefix = "scp_nl" if nonlinear else "scp"
    if dense:
        prefix += "_dense"
    return f"{prefix}_d{n_vars}"


# ---------------------------------------------------------------------------
# Subcommand: generate
# ---------------------------------------------------------------------------


def cmd_generate(args):
    """Generate and save data bundles only."""
    seeds = list(range(args.n_graphs))

    print("Data Generation")
    print(f"  n_vars:    {args.n_vars}")
    print(f"  n_graphs:  {args.n_graphs} (seeds {seeds[0]}-{seeds[-1]})")
    print(f"  nonlinear: {args.nonlinear}")
    print(f"  dense:     {args.dense}")
    print(f"  max_tries: {args.max_tries}")
    print(f"  data_dir:  {args.data_dir}")
    print()

    successes, failures = 0, 0

    for n_vars in args.n_vars:
        for seed in seeds:
            prefix = "scp_nl" if args.nonlinear else "scp"
            if args.dense:
                prefix += "_dense"
            fname = f"{prefix}_d{n_vars}_seed{seed}.npz"
            if os.path.exists(os.path.join(args.data_dir, fname)):
                print(f"  d={n_vars} seed={seed}: exists (local), skipping")
                successes += 1
                continue

            try:
                T_max = 2500
                t0 = time.time()
                bundle = generate_scp(
                    n_vars,
                    T_max,
                    seed,
                    nonlinear=args.nonlinear,
                    dense=args.dense,
                    max_tries=args.max_tries,
                )
                gen_time = time.time() - t0
                n_edges = int(bundle["ground_truth"].sum())

                path = save_data_bundle(
                    bundle,
                    n_vars,
                    seed,
                    nonlinear=args.nonlinear,
                    dense=args.dense,
                    data_dir=args.data_dir,
                )
                print(
                    f"  d={n_vars} seed={seed}: {n_edges} edges, T_max={T_max} "
                    f"({gen_time:.1f}s) -> {path}"
                )
                successes += 1

            except RuntimeError as e:
                print(f"  d={n_vars} seed={seed}: FAILED ({e})")
                failures += 1
            except Exception as e:
                print(f"  d={n_vars} seed={seed}: FAILED ({e})")
                traceback.print_exc()
                failures += 1

    print(f"\nDone: {successes} succeeded, {failures} failed")


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------


def _get_or_generate_bundle(n_vars, seed, T_max, args):
    """Load data bundle from local storage, or generate fresh."""
    bundle = load_data_bundle(
        n_vars,
        seed,
        nonlinear=args.nonlinear,
        dense=args.dense,
        data_dir=args.data_dir,
    )
    if bundle is not None and len(bundle["df"]) >= T_max:
        return bundle, "loaded"
    if bundle is not None:
        print(f"    (bundle has T={len(bundle['df'])}, need {T_max}; regenerating)")

    bundle = generate_scp(
        n_vars,
        T_max,
        seed,
        nonlinear=args.nonlinear,
        dense=args.dense,
        max_tries=args.max_tries,
    )
    save_data_bundle(
        bundle,
        n_vars,
        seed,
        nonlinear=args.nonlinear,
        dense=args.dense,
        data_dir=args.data_dir,
    )
    return bundle, "generated"


def _run_methods_on_bundle(bundle_full, n_vars, seed, T_values, args, summary_rows):
    """Run all requested methods on a bundle at each T value."""
    gt = bundle_full["ground_truth"]
    skip = args.skip_existing

    for T_val in T_values:
        print(f"\n  --- T={T_val} ---")
        bundle = slice_bundle(bundle_full, T_val)
        ds_name_str = _ds_name(n_vars, args.nonlinear, dense=args.dense)

        for method_name in args.methods:
            if skip and result_exists(
                ds_name_str,
                method_name,
                seed,
                T_val,
                results_dir=args.results_dir,
            ):
                print(f"    [{method_name}] exists, skipping")
                continue

            print(f"    [{method_name}] ", end="", flush=True)
            fn = SCALING_METHODS[method_name]

            try:
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(args.timeout)

                try:
                    extra_kwargs = {}
                    if "gated" in method_name:
                        extra_kwargs["model_seed"] = args.model_seed
                    if (
                        getattr(args, "reuse_skeleton", False)
                        and method_name in SKELETON_SOURCE
                    ):
                        skel_method = SKELETON_SOURCE[method_name]
                        skel, skel_rt = load_skeleton(
                            ds_name_str,
                            seed,
                            T_val,
                            skeleton_method=skel_method,
                            results_dir=args.results_dir,
                        )
                        if skel is not None:
                            extra_kwargs["skeleton"] = skel
                            extra_kwargs["skeleton_runtime"] = skel_rt
                            print(
                                f"(reusing {skel_method} skeleton) ", end="", flush=True
                            )
                        else:
                            print(
                                f"(no {skel_method} result, computing skeleton) ",
                                end="",
                                flush=True,
                            )
                    if method_name.startswith("dynotears_gated"):
                        coef, dyno_rt = load_coef_matrix(
                            ds_name_str,
                            seed,
                            T_val,
                            results_dir=args.results_dir,
                        )
                        if coef is not None:
                            extra_kwargs["preloaded_coef_matrix"] = coef
                            print("(reusing dynotears coefs) ", end="", flush=True)
                        else:
                            print(
                                "(no dynotears result, running LASSO VAR) ",
                                end="",
                                flush=True,
                            )
                    G_est, method_params, method_info, runtime = fn(
                        bundle, **extra_kwargs
                    )
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

                if G_est.shape != gt.shape:
                    G_est = _align_graph_lags(G_est, bundle["max_lag"])

                metrics = evaluate_graph(G_est, gt)

                path = save_result(
                    dataset=ds_name_str,
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
                    "method": method_name,
                    "runtime": runtime,
                    **metrics,
                }
                summary_rows.append(row)

                print(
                    f"F1={metrics['F1']:.3f}  SHD={metrics['SHD']}  "
                    f"TPR={metrics['TPR']:.3f}  Prec={metrics['Precision']:.3f}  "
                    f"({runtime:.1f}s)"
                )

            except TimeoutError:
                print(f"TIMEOUT (>{args.timeout}s)")
                row = {
                    "d": n_vars,
                    "T": T_val,
                    "seed": seed,
                    "method": method_name,
                    "runtime": args.timeout,
                    "F1": np.nan,
                    "SHD": np.nan,
                    "TPR": np.nan,
                    "Precision": np.nan,
                    "FPR": np.nan,
                    "TP": np.nan,
                    "FP": np.nan,
                    "FN": np.nan,
                }
                summary_rows.append(row)
            except Exception as e:
                print(f"FAILED: {e}")
                traceback.print_exc()


def _print_summary(summary_rows):
    """Print per-seed and aggregated summary tables."""
    if not summary_rows:
        return
    df_summary = pd.DataFrame(summary_rows)

    print(f"\n{'='*70}")
    print("PER-SEED RESULTS")
    print(f"{'='*70}")
    detail_cols = [
        "d",
        "seed",
        "T",
        "method",
        "F1",
        "SHD",
        "TPR",
        "Precision",
        "runtime",
    ]
    detail = df_summary[[c for c in detail_cols if c in df_summary.columns]]
    detail = detail.sort_values(["d", "seed", "T", "method"])
    print(detail.to_string(index=False, float_format="%.3f"))

    print(f"\n{'='*70}")
    print("AGGREGATED SUMMARY (over seeds)")
    print(f"{'='*70}")
    group_cols = ["d", "T", "method"]
    agg = (
        df_summary.groupby(group_cols)
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


def cmd_run(args):
    """Run methods on existing (or freshly generated) data."""
    seeds = list(range(args.n_graphs))
    T_values = sorted(args.T)
    T_max = max(T_values) + T_MAX_MARGIN

    print("Scaling Experiment (run)")
    print(f"  n_vars:     {args.n_vars}")
    print(f"  T:          {T_values}")
    print(f"  methods:    {args.methods}")
    print(f"  n_graphs:   {args.n_graphs} (seeds {seeds[0]}-{seeds[-1]})")
    print(f"  model_seed: {args.model_seed}")
    print(f"  nonlinear:  {args.nonlinear}")
    print(f"  dense:      {args.dense}")
    print(f"  timeout:    {args.timeout}s")
    print(f"  T_max:      {T_max} (generate once, slice prefixes)")
    print(f"  results ->  {args.results_dir}/")
    print(f"  data_dir:   {args.data_dir}")
    print(f"  skip:       {args.skip_existing}")
    print()

    summary_rows = []

    for n_vars in args.n_vars:
        for seed in seeds:
            print(f"{'='*70}")
            print(
                f"d={n_vars}, graph_seed={seed}, {'nonlinear' if args.nonlinear else 'mixed'}"
            )
            print(f"{'='*70}")

            try:
                t0 = time.time()
                bundle_full, source = _get_or_generate_bundle(n_vars, seed, T_max, args)
                gen_time = time.time() - t0
                gt = bundle_full["ground_truth"]
                n_true_edges = int(gt.sum())
                print(
                    f"  Data {source} (T={len(bundle_full['df'])}): "
                    f"{n_true_edges} true edges ({gen_time:.1f}s)"
                )
            except RuntimeError as e:
                print(f"  Data generation FAILED for d={n_vars} seed={seed}: {e}")
                print("  Skipping this graph.")
                continue
            except Exception as e:
                print(f"  Data generation FAILED for d={n_vars} seed={seed}: {e}")
                traceback.print_exc()
                print("  Skipping this graph.")
                continue

            _run_methods_on_bundle(
                bundle_full, n_vars, seed, T_values, args, summary_rows
            )

    _print_summary(summary_rows)


# ---------------------------------------------------------------------------
# Legacy (no subcommand) — backward compatible
# ---------------------------------------------------------------------------


def cmd_legacy(args):
    """Original behavior with try-catch + skip-existing added."""
    seeds = list(range(args.n_graphs))
    T_values = sorted(args.T)
    T_max = max(T_values) + T_MAX_MARGIN

    print("Scaling Experiment")
    print(f"  n_vars:    {args.n_vars}")
    print(f"  T:         {T_values}")
    print(f"  methods:   {args.methods}")
    print(f"  n_graphs:  {args.n_graphs} (seeds {seeds[0]}-{seeds[-1]})")
    print(f"  model_seed: {args.model_seed}")
    print(f"  nonlinear: {args.nonlinear}")
    print(f"  dense:     {args.dense}")
    print(f"  timeout:   {args.timeout}s")
    print(f"  T_max:     {T_max} (generate once, slice prefixes)")
    print(f"  results ->  {args.results_dir}/")
    print(f"  skip:       {args.skip_existing}")
    print()

    summary_rows = []

    for n_vars in args.n_vars:
        for seed in seeds:
            print(f"{'='*70}")
            print(
                f"d={n_vars}, graph_seed={seed}, {'nonlinear' if args.nonlinear else 'mixed'}"
            )
            print(f"{'='*70}")

            try:
                t0 = time.time()
                bundle_full = generate_scp(
                    n_vars,
                    T_max,
                    seed,
                    nonlinear=args.nonlinear,
                    dense=args.dense,
                    max_tries=args.max_tries,
                )
                gen_time = time.time() - t0
                gt = bundle_full["ground_truth"]
                n_true_edges = int(gt.sum())
                print(
                    f"  Generated T_max={T_max}: {n_true_edges} true edges ({gen_time:.1f}s)"
                )
            except RuntimeError as e:
                print(f"  Data generation FAILED for d={n_vars} seed={seed}: {e}")
                print("  Skipping this graph.")
                continue
            except Exception as e:
                print(f"  Data generation FAILED for d={n_vars} seed={seed}: {e}")
                traceback.print_exc()
                print("  Skipping this graph.")
                continue

            _run_methods_on_bundle(
                bundle_full, n_vars, seed, T_values, args, summary_rows
            )

    _print_summary(summary_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common_args(parser):
    """Add arguments shared across subcommands."""
    parser.add_argument(
        "--n_vars", nargs="+", type=int, default=[20], help="Number of variables"
    )
    parser.add_argument(
        "--n_graphs",
        type=int,
        default=10,
        help="Number of graphs per d (seeds 0..n_graphs-1)",
    )
    parser.add_argument(
        "--nonlinear", action="store_true", help="Use purely nonlinear SCP functions"
    )
    parser.add_argument(
        "--max-tries",
        type=int,
        default=500,
        help="Max tries for stationary SCP generation (default 500)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="results/scaling/data",
        help="Directory for data bundles",
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Use dense SCP graphs (~3-5 edges/var instead of ~1-2)",
    )


def _add_run_args(parser):
    """Add arguments specific to running methods."""
    parser.add_argument(
        "--T", nargs="+", type=int, default=[1000], help="Time series lengths"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(SCALING_METHODS.keys()),
        choices=list(SCALING_METHODS.keys()),
        help="Methods to run",
    )
    parser.add_argument(
        "--model-seed", type=int, default=42, help="Model seed for gated method"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Per-method timeout in seconds (default 1800)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Output directory for result JSONs",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip methods whose results already exist (default)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Disable skip-existing (re-run everything)",
    )
    parser.add_argument(
        "--reuse-skeleton",
        action="store_true",
        help="Reuse pre-computed skeleton from existing cdnots/pcmci results",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Scaling experiments for causal discovery",
    )
    subparsers = parser.add_subparsers(dest="command")

    gen_parser = subparsers.add_parser(
        "generate", help="Generate and save data bundles"
    )
    _add_common_args(gen_parser)

    run_parser = subparsers.add_parser("run", help="Run methods on existing data")
    _add_common_args(run_parser)
    _add_run_args(run_parser)

    _add_common_args(parser)
    _add_run_args(parser)

    args = parser.parse_args()

    if hasattr(args, "no_skip") and args.no_skip:
        args.skip_existing = False
    if not hasattr(args, "skip_existing"):
        args.skip_existing = True

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "run":
        cmd_run(args)
    else:
        cmd_legacy(args)


if __name__ == "__main__":
    main()
