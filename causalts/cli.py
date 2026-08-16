# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

import matplotlib

matplotlib.use("Agg")

import json  # noqa: E402
import os  # noqa: E402
import time as _time  # noqa: E402
from datetime import datetime  # noqa: E402

import click  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .ci_tests import create_ci_test, list_ci_tests  # noqa: E402

CI_TEST_CHOICES = list_ci_tests()

CI_TEST_GUIDE = """
Conditional Independence Test Selection Guide
=============================================

  parcorr-gpu  Linear partial correlation. Near-instant (analytic p-value).
               Works at any T and any d. Misses nonlinear edges.

  gcmi         Gaussian Copula + Partial Correlation. Near-instant (analytic).
               Rank-normalises then runs partial correlation. Handles mild
               (monotone) nonlinearity only; misses non-monotone dependencies.

  kci          Kernel CI (RBF, HBE p-value). Uses all T points, no split bias.
               Most general kernel test. O(T^3) — slow at large T.

  splitkci     Bias-controlled kernel CI (train/test split, gamma p-value).
               10-20x faster than kci. Good speed/F1 tradeoff for nonlinear
               data. Performance improves with larger T.

  dfcit        Distribution-free CI (empirical CDFs, permutation p-value).
               Strong recall on nonlinear data. Handles mixed discrete-continuous
               columns natively. Results vary by graph structure.

  rcot         Randomised conditional correlation (Random Fourier Features).
               Fast. At T<=300, auto-averages 5 RFF draws to reduce variance.

  cmiknn-gpu   k-NN conditional mutual information. Nonparametric, no bandwidth.
               Sensitive to many dependency types but slow (O(T^2) k-NN search).

  sigkci       Signature kernel CI for path-valued data (SDEs, diffusions).
               Captures temporal structure that pointwise tests ignore.

When to use what
----------------
  Linear data, any T               -> parcorr-gpu (recommended), then gcmi
  Nonlinear, smaller samples       -> dfcit (high recall)
  Nonlinear, sufficient data       -> splitkci (best speed/F1 tradeoff)
  Nonlinear, high collinearity     -> kci (uses all data, most robust)
  Monotone nonlinear, fast needed  -> gcmi (analytic, near-instant)
  Stochastic processes / SDEs      -> sigkci
  Runtime-constrained              -> parcorr-gpu or gcmi

Key tradeoffs
-------------
  - Large variable sets: kernel tests lose power as conditioning sets grow;
    parcorr-gpu and gcmi scale better.
  - Weak edges: nonlinear tests need more data than linear tests to detect
    low-coefficient edges.
  - Near-collinear conditioning: kci is most robust; dfcit can over-discover;
    splitkci can be noisy.
    Use seed averaging if results look unstable.
  - gcmi is NOT a true MI estimator: cannot detect non-monotone relationships.
"""
from .algorithms import list_algorithms as _list_algorithms  # noqa: E402

ALGORITHM_CHOICES = _list_algorithms()
PLOT_FORMAT_CHOICES = ["png", "pdf", "svg"]
DATASET_CHOICES = ["ex1", "ex2", "ex3", "henon"]
LAG_SEL_CHOICES = ["partial_dcor", "dcor", "dcor_biased", "pearson", "lasso"]
LAG_PVALUE_CHOICES = ["t_test", "circular_shift"]
DENSITY_CHOICES = ["sparse", "dense"]


def _make_ci_test(name, data, device):
    return create_ci_test(name, data, device=device)


def _make_output_dir(base, prefix):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, f"{prefix}_{ts}")
    os.makedirs(path, exist_ok=True)
    return path


def _save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _parse_comma_list(value, cast=str):
    if value is None:
        return None
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def _parse_figsize(value):
    if value is None:
        return None
    parts = value.split(",")
    if len(parts) != 2:
        raise click.BadParameter("figsize must be two numbers: e.g. 10,8")
    return (float(parts[0]), float(parts[1]))


def _log(ctx, msg):
    if not ctx.obj.get("quiet"):
        click.echo(msg)


def _seed_all(seed):
    if seed is not None:
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
        except ImportError:
            pass


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-o",
    "--output-dir",
    default="./causal_ts_output",
    type=click.Path(),
    help="Base output directory.",
)
@click.option("-s", "--seed", type=int, default=None, help="Random seed.")
@click.option(
    "-d",
    "--device",
    type=str,
    default=None,
    help="Torch device: cpu, cuda, cuda:0, mps. Auto-detected if omitted.",
)
@click.option(
    "-v", "--verbose", count=True, help="Increase verbosity (repeat for more)."
)
@click.option("-q", "--quiet", is_flag=True, help="Suppress non-error output.")
@click.version_option(package_name="causalts")
@click.pass_context
def main(ctx, output_dir, seed, device, verbose, quiet):
    """Causal discovery for nonstationary time series."""
    ctx.ensure_object(dict)
    ctx.obj["output_dir"] = output_dir
    ctx.obj["seed"] = seed
    ctx.obj["device"] = device
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    os.makedirs(output_dir, exist_ok=True)
    _seed_all(seed)


# ---------------------------------------------------------------------------
# ci-test-info
# ---------------------------------------------------------------------------
@main.command("ci-test-info")
@click.option(
    "--test",
    type=click.Choice(CI_TEST_CHOICES + ["all"]),
    default="all",
    help="Show info for a specific test or all (default).",
)
def ci_test_info(test):
    """Print the CI test selection guide and per-test summaries."""
    click.echo(CI_TEST_GUIDE)


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------
@main.command("inspect")
@click.argument("data", type=click.Path(exists=True))
@click.option(
    "--var-names", type=str, default=None, help="Comma-separated variable names."
)
@click.option(
    "--max-lag", type=int, default=None, help="Override the suggested max lag."
)
def inspect(data, var_names, max_lag):
    """Inspect a time series (CSV/parquet/feather): data health + recommended config.

    Emits a JSON report {schema_version, data, facts, recommendation, cost_class,
    warnings} to stdout — the pre-flight for `discover`.
    """
    import json as _json

    from .inspection import inspect_df
    from .utils.io import read_dataframe

    df = read_dataframe(data)
    var_name_list = _parse_comma_list(var_names)
    if var_name_list:
        df.columns = var_name_list
    report = inspect_df(df, max_lag=max_lag)
    click.echo(_json.dumps(report, indent=2, default=str))


# ---------------------------------------------------------------------------
# discover
# ---------------------------------------------------------------------------
@main.command()
@click.argument("data", type=click.Path(exists=True))
@click.option(
    "--algorithm",
    "-a",
    type=click.Choice(ALGORITHM_CHOICES),
    default="cdnots",
    help="Discovery algorithm (default: cdnots).",
)
@click.option(
    "--ci-test",
    type=click.Choice(CI_TEST_CHOICES),
    default="parcorr-gpu",
    help="Conditional independence test. Run 'ci-test-info' for selection guidance.",
)
@click.option("--max-lag", type=int, default=3, help="Maximum time lag.")
@click.option("--alpha", type=float, default=0.05, help="Significance level.")
@click.option(
    "--lag-list", type=str, default=None, help="Comma-separated lag list (e.g. 1,3,12)."
)
@click.option(
    "--include-c/--no-c",
    default=True,
    help="Include nonstationarity variable C (CDNOTS and CEDAR).",
)
@click.option(
    "--c-preset",
    "c_preset",
    type=click.Choice(
        ["linear", "step", "step+linear", "linear+sin", "linear+exp", "linear+quad"]
    ),
    default="linear",
    help="C node basis preset for nonstationarity (CDNOTS and CEDAR, default: linear).",
)
@click.option("--stable", is_flag=True, help="Stable skeleton discovery (CDNOTS).")
@click.option(
    "--max-degree", type=int, default=None, help="Max degree constraint (CDNOTS)."
)
@click.option(
    "--lag-method",
    "--lag-sel-algo",
    "lag_method",
    type=click.Choice(LAG_SEL_CHOICES),
    default="partial_dcor",
    show_default=True,
    help="CEDAR lag importance method. --lag-sel-algo accepted for legacy.",
)
@click.option(
    "--lag-pvalue-method",
    type=click.Choice(LAG_PVALUE_CHOICES),
    default="t_test",
    help="CEDAR lag significance test: t_test (default, ~200x faster) or circular_shift (permutation, more accurate for AR data).",
)
@click.option(
    "--lag-alpha",
    type=float,
    default=0.05,
    help="CEDAR p-value threshold for lag significance (default: 0.05).",
)
@click.option(
    "--alpha-cond1",
    "--p-cond1",
    "alpha_cond1",
    type=float,
    default=0.05,
    help="CEDAR significance threshold for dependence test / Condition 1 (default: 0.05). --p-cond1 accepted for legacy.",
)
@click.option(
    "--alpha-cond2",
    "--p-cond2",
    "alpha_cond2",
    type=float,
    default=0.05,
    help="CEDAR significance threshold for independence test / Condition 2 (default: 0.05). --p-cond2 accepted for legacy.",
)
@click.option(
    "--include-lag0",
    is_flag=True,
    help="CEDAR include contemporaneous (lag-0) effects.",
)
@click.option(
    "--normalize",
    "--normalization",
    "normalize",
    type=click.Choice(["minmax", "zscore", "none"]),
    default="minmax",
    help="CEDAR data normalization before lag importance (default: minmax). --normalization accepted for legacy.",
)
@click.option(
    "--filter-children/--no-filter-children",
    "filter_children",
    default=True,
    help="CEDAR skip candidates whose dcor asymmetry indicates they are effects, not causes (default: on).",
)
@click.option(
    "--multi-lag/--no-multi-lag",
    "multi_lag",
    default=True,
    help="CEDAR test all significant lags per pair in importance order (default: on). --no-multi-lag restricts to top lag only.",
)
@click.option(
    "--multi-lag-keep",
    type=click.Choice(["first", "all"]),
    default="first",
    help="CEDAR multi-lag policy: first (stop at first accepted lag, default) or all (report every accepted lag, max recall).",
)
@click.option(
    "--target-var",
    type=str,
    default=None,
    help="CEDAR discover causes of a single target variable only (O(d) tests instead of O(d²)).",
)
@click.option(
    "--no-prune", is_flag=True, help="CEDAR skip MCI pruning pass after discovery."
)
@click.option(
    "--include-autoreg/--no-autoreg",
    "include_autoreg",
    default=True,
    help="CEDAR add autoregressive self-loops Xi(t-k)→Xi(t) for the detected AR order of each variable (default: on).",
)
@click.option(
    "--assume-ar1",
    "assume_ar1",
    is_flag=True,
    help="CEDAR skip AR order estimation and use standard AR(1) Cond2 for all variables — backward-compatible with original SyPI (Mastakouri et al. 2021) behavior.",
)
@click.option(
    "--gate-threshold",
    type=float,
    default=0.5,
    help="Gate threshold for GRACE.",
)
@click.option(
    "--stability-threshold",
    type=float,
    default=0.6,
    help="Stability threshold for GRACE-SS.",
)
@click.option("--max-epochs", type=int, default=150, help="Training epochs (GRACE).")
@click.option(
    "--batch-size", type=int, default=None, help="Batch size (GRACE, auto if omitted)."
)
@click.option(
    "--patience", type=int, default=20, help="Early stopping patience (GRACE)."
)
@click.option(
    "--ground-truth",
    type=click.Path(exists=True),
    default=None,
    help="Path to ground truth .npy for instant evaluation.",
)
@click.option("--save-plot/--no-plot", default=True, help="Save plot images.")
@click.option(
    "--plot-format",
    type=click.Choice(PLOT_FORMAT_CHOICES),
    default="png",
    help="Plot format.",
)
@click.option(
    "--var-names", type=str, default=None, help="Comma-separated variable names."
)
@click.option(
    "--impute",
    type=click.Choice(["pairwise_complete", "var_em", "causal_iterative"]),
    default=None,
    help=(
        "Missing-value strategy. Works with all algorithms (cdnots, cedar, grace, "
        "grace-ss). 'pairwise_complete': CI tests drop NaN rows per test. "
        "'var_em': fill via VAR-EM before discovery. "
        "'causal_iterative': VAR-EM + Markov-blanket Ridge (cdnots only)."
    ),
)
@click.option(
    "--impute-kwargs",
    type=str,
    default=None,
    help='JSON dict of kwargs for the imputer, e.g. \'{"p": 2, "n_iter": 5}\'.',
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=False,
    help="Echo the run summary (incl. an edge list) as JSON to stdout.",
)
@click.option(
    "--pvalues/--no-pvalues",
    "want_pvalues",
    default=False,
    help=(
        "Return per-edge p-values (CDNOTS family and CEDAR). Off by default "
        "as a memory safeguard for very high-dimensional data; safe to enable "
        "on low/moderate d."
    ),
)
@click.option(
    "--validate",
    "do_validate",
    is_flag=True,
    default=False,
    help=(
        "After discovery, run a contiguous-window bootstrap-stability check and "
        "annotate each edge with its persistence (fraction of windows it recurs). "
        "Not supported for GRACE. Measures sampling robustness, not correctness."
    ),
)
@click.option(
    "--n-bootstrap",
    type=int,
    default=20,
    help="Number of bootstrap windows for --validate (default 20).",
)
@click.option(
    "--window-frac",
    type=float,
    default=0.6,
    help="Window size as a fraction of T for --validate (default 0.6).",
)
@click.pass_context
def discover(
    ctx,
    data,
    algorithm,
    ci_test,
    max_lag,
    alpha,
    lag_list,
    include_c,
    c_preset,
    stable,
    max_degree,
    lag_method,
    lag_pvalue_method,
    lag_alpha,
    alpha_cond1,
    alpha_cond2,
    include_lag0,
    normalize,
    filter_children,
    multi_lag,
    multi_lag_keep,
    target_var,
    no_prune,
    include_autoreg,
    assume_ar1,
    gate_threshold,
    stability_threshold,
    max_epochs,
    batch_size,
    patience,
    ground_truth,
    save_plot,
    plot_format,
    var_names,
    impute,
    impute_kwargs,
    output_json,
    want_pvalues,
    do_validate,
    n_bootstrap,
    window_frac,
):
    """Run causal discovery on a time series file (CSV, parquet, or feather)."""
    if do_validate and algorithm in ("grace", "grace-ss"):
        raise click.BadParameter(
            "--validate is not supported for GRACE (bootstrapping neural-gate "
            "training over many windows is impractical). Re-run the stability "
            "check with --algorithm cdnots or cedar.",
            param_hint="--validate",
        )
    if want_pvalues and algorithm in ("grace", "grace-ss"):
        raise click.BadParameter(
            "--pvalues is not supported for GRACE (it emits continuous gate "
            "values, not a per-test p-value). Re-run with --algorithm cdnots "
            "or cedar for p-values.",
            param_hint="--pvalues",
        )
    from matplotlib import pyplot as plt

    t0 = _time.time()
    outdir = _make_output_dir(ctx.obj["output_dir"], f"discover_{algorithm}")
    device = ctx.obj["device"]
    var_name_list = _parse_comma_list(var_names)
    lag_list_parsed = _parse_comma_list(lag_list, int)

    _log(ctx, f"Reading data from {data}")
    from .utils.io import read_dataframe

    df = read_dataframe(data)
    if var_name_list:
        df.columns = var_name_list
    var_name_list = list(df.columns)

    _log(ctx, f"Data shape: {df.shape}, variables: {var_name_list}")
    _log(ctx, f"Algorithm: {algorithm}, CI test: {ci_test}, max_lag: {max_lag}")

    placeholder = np.zeros((2, 2))
    ci = _make_ci_test(ci_test, placeholder, device)

    import json as _json

    _impute_kwargs = {}
    if impute_kwargs:
        try:
            _impute_kwargs = _json.loads(impute_kwargs)
        except _json.JSONDecodeError as e:
            raise click.BadParameter(
                f"--impute-kwargs must be valid JSON: {e}", param_hint="--impute-kwargs"
            )

    summary = {
        "algorithm": algorithm,
        "ci_test": ci_test,
        "data_file": os.path.abspath(data),
        "data_shape": list(df.shape),
        "var_names": var_name_list,
        "max_lag": max_lag,
        "alpha": alpha,
        "seed": ctx.obj["seed"],
        "device": str(device),
        "timestamp": datetime.now().isoformat(),
        "output_files": {},
    }

    if impute is not None and algorithm not in (
        "cdnots",
        "cdnots+",
        "cedar",
        "grace",
        "grace-ss",
    ):
        _log(
            ctx,
            f"Warning: --impute is not supported for {algorithm}; ignoring.",
        )
        impute = None

    if algorithm in ("cdnots", "cdnots+"):
        label = "CDNOTS+" if algorithm == "cdnots+" else "CDNOTS"
        _log(ctx, f"Running {label}...")
        if impute is not None and impute != "pairwise_complete":
            _log(ctx, f"Imputation strategy: {impute}")
        from .cdnots.phase3_utils import run_cdnots, run_cdnots_plus

        fn = run_cdnots if algorithm == "cdnots" else run_cdnots_plus
        extra_kwargs = {"stable": stable} if algorithm == "cdnots" else {}
        res = fn(
            df=df,
            indep_test=ci,
            num_lags=max_lag,
            lag_list=lag_list_parsed,
            include_C=include_c,
            c_preset=c_preset,
            alpha=alpha,
            max_degree=max_degree,
            return_pvals=want_pvalues,
            verbose=ctx.obj["verbose"] > 0,
            show_progress=not ctx.obj["quiet"],
            impute=impute,
            impute_kwargs=_impute_kwargs,
            **extra_kwargs,
        )

        np.save(os.path.join(outdir, "estimated_graph.npy"), res.cg_tig)
        summary["output_files"]["graph"] = "estimated_graph.npy"
        summary["include_C"] = include_c
        summary["lag_list"] = lag_list_parsed
        if algorithm == "cdnots":
            summary["stable"] = stable
        summary["impute"] = impute
        summary["impute_kwargs"] = _impute_kwargs

        if res.pvalue_matrix is not None:
            np.save(os.path.join(outdir, "pvalues.npy"), res.pvalue_matrix)
            summary["output_files"]["pvalues"] = "pvalues.npy"

        if save_plot:
            try:
                plot_names = list(var_name_list)
                if include_c and res.cg_tig.shape[0] > len(plot_names):
                    plot_names.append("C")
                plot_result = res.plot(var_names=plot_names, exclude_C=not include_c)
                fig = plot_result[0] if isinstance(plot_result, tuple) else plot_result
                if fig is not None:
                    fname = f"network.{plot_format}"
                    fig.savefig(
                        os.path.join(outdir, fname), dpi=150, bbox_inches="tight"
                    )
                    plt.close(fig)
                    summary["output_files"]["network_plot"] = fname
            except Exception as e:
                _log(ctx, f"Warning: could not save network plot: {e}")

        n_edges = int(res.cg_tig.astype(bool).sum())
        elapsed = _time.time() - t0
        summary["elapsed_seconds"] = round(elapsed, 1)
        summary["n_edges"] = n_edges
        _log(
            ctx,
            f"{label} complete. Graph shape: {res.cg_tig.shape}, {n_edges} edges, {elapsed:.1f}s",
        )

    elif algorithm == "cedar":
        _log(
            ctx,
            f"Running CEDAR (lag_method={lag_method}, alpha_cond1={alpha_cond1}, "
            f"alpha_cond2={alpha_cond2}, lag_pvalue_method={lag_pvalue_method})...",
        )
        from .cedar.discovery import run_cedar

        cedar_result = run_cedar(
            df=df,
            ci_test=ci,
            max_lag=max_lag,
            alpha_cond1=alpha_cond1,
            alpha_cond2=alpha_cond2,
            lag_method=lag_method,
            lag_pvalue_method=lag_pvalue_method,
            lag_alpha=lag_alpha,
            normalize=None if normalize == "none" else normalize,
            filter_children=filter_children,
            include_lag0=include_lag0,
            prune=not no_prune,
            multi_lag=multi_lag,
            multi_lag_keep=multi_lag_keep,
            target_var=target_var,
            verbose=ctx.obj["verbose"] > 0,
            impute=impute,
            impute_kwargs=_impute_kwargs,
            include_C=include_c,
            c_preset=c_preset,
            include_autoreg=include_autoreg,
            assume_ar1=assume_ar1,
        )

        graph = cedar_result.cg_tig
        np.save(os.path.join(outdir, "estimated_graph.npy"), graph)
        summary["output_files"]["graph"] = "estimated_graph.npy"

        if cedar_result.val_matrix is not None:
            np.save(os.path.join(outdir, "val_matrix.npy"), cedar_result.val_matrix)
            summary["output_files"]["val_matrix"] = "val_matrix.npy"

        if want_pvalues and cedar_result.pvalue_matrix is not None:
            np.save(os.path.join(outdir, "pvalues.npy"), cedar_result.pvalue_matrix)
            summary["output_files"]["pvalues"] = "pvalues.npy"

        summary["lag_method"] = lag_method
        summary["lag_pvalue_method"] = lag_pvalue_method
        summary["lag_alpha"] = lag_alpha
        summary["alpha_cond1"] = alpha_cond1
        summary["alpha_cond2"] = alpha_cond2
        summary["include_lag0"] = include_lag0
        summary["normalize"] = normalize
        summary["filter_children"] = filter_children
        summary["multi_lag"] = multi_lag
        summary["multi_lag_keep"] = multi_lag_keep
        summary["pruned"] = not no_prune
        summary["impute"] = impute
        summary["impute_kwargs"] = _impute_kwargs
        summary["include_C"] = include_c
        summary["c_preset"] = c_preset
        summary["include_autoreg"] = include_autoreg
        summary["assume_ar1"] = assume_ar1
        if cedar_result.ar_violations:
            summary["ar_violations"] = cedar_result.ar_violations
        if cedar_result.ar_order:
            summary["ar_order"] = {
                cedar_result.var_names[k]: v for k, v in cedar_result.ar_order.items()
            }

        if save_plot:
            try:
                result = cedar_result.plot()
                fig = result[0] if isinstance(result, tuple) else result
                if fig is not None:
                    fname = f"network.{plot_format}"
                    fig.savefig(
                        os.path.join(outdir, fname), dpi=150, bbox_inches="tight"
                    )
                    plt.close(fig)
                    summary["output_files"]["network_plot"] = fname
            except Exception as e:
                _log(ctx, f"Warning: could not save network plot: {e}")

        n_edges = int(graph.astype(bool).sum())
        elapsed = _time.time() - t0
        summary["elapsed_seconds"] = round(elapsed, 1)
        summary["n_edges"] = n_edges
        _log(
            ctx,
            f"CEDAR complete. Graph: {graph.shape}, {n_edges} edges, "
            f"n_tests={cedar_result.n_tests}, {elapsed:.1f}s",
        )

    elif algorithm == "grace":
        _log(ctx, "Running GRACE (CDNOTS + gated refinement)...")
        from .grace import run_cdnots_gated

        grace_res = run_cdnots_gated(
            df=df,
            max_lag=max_lag,
            alpha=alpha,
            include_C=include_c,
            c_preset=c_preset,
            gate_threshold=gate_threshold,
            max_epochs=max_epochs,
            batch_size=batch_size,
            patience=patience,
            device=device,
            verbose=not ctx.obj["quiet"],
            model_seed=ctx.obj["seed"],
            impute=impute,
            impute_kwargs=_impute_kwargs,
        )

        np.save(os.path.join(outdir, "estimated_graph.npy"), grace_res.cg_tig)
        np.save(os.path.join(outdir, "gate_values.npy"), grace_res.gate_values)
        summary["output_files"]["graph"] = "estimated_graph.npy"
        summary["output_files"]["gate_values"] = "gate_values.npy"
        summary["gate_threshold"] = gate_threshold
        summary["include_C"] = include_c
        summary["max_epochs"] = max_epochs

        if save_plot:
            try:
                result = grace_res.plot(var_names=var_name_list)
                fig = result[0] if isinstance(result, tuple) else result
                fname = f"network.{plot_format}"
                fig.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
                plt.close(fig)
                summary["output_files"]["network_plot"] = fname
            except Exception as e:
                _log(ctx, f"Warning: could not save network plot: {e}")

        n_edges = int(grace_res.cg_tig.astype(bool).sum())
        elapsed = _time.time() - t0
        summary["elapsed_seconds"] = round(elapsed, 1)
        summary["n_edges"] = n_edges
        _log(
            ctx,
            f"GRACE complete. Graph shape: {grace_res.cg_tig.shape}, {n_edges} edges, {elapsed:.1f}s",
        )

    elif algorithm == "grace-ss":
        _log(ctx, "Running GRACE with stability selection...")
        from .grace import run_stability_selection

        grace_ss_res = run_stability_selection(
            df=df,
            max_lag=max_lag,
            use_ci_skeleton=True,
            ci_alpha=alpha,
            include_C=include_c,
            c_preset=c_preset,
            stability_threshold=stability_threshold,
            max_epochs=max_epochs,
            batch_size=batch_size,
            patience=patience,
            device=device,
            verbose=not ctx.obj["quiet"],
            model_seed=ctx.obj["seed"],
            impute=impute,
            impute_kwargs=_impute_kwargs,
        )

        np.save(os.path.join(outdir, "estimated_graph.npy"), grace_ss_res.cg_tig)
        np.save(
            os.path.join(outdir, "stability_scores.npy"), grace_ss_res.stability_scores
        )
        summary["output_files"]["graph"] = "estimated_graph.npy"
        summary["output_files"]["stability_scores"] = "stability_scores.npy"
        summary["stability_threshold"] = stability_threshold
        summary["include_C"] = include_c
        summary["max_epochs"] = max_epochs

        if save_plot:
            try:
                result = grace_ss_res.plot(var_names=var_name_list)
                fig = result[0] if isinstance(result, tuple) else result
                fname = f"network.{plot_format}"
                fig.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
                plt.close(fig)
                summary["output_files"]["network_plot"] = fname
            except Exception as e:
                _log(ctx, f"Warning: could not save network plot: {e}")

        n_edges = int(grace_ss_res.cg_tig.astype(bool).sum())
        elapsed = _time.time() - t0
        summary["elapsed_seconds"] = round(elapsed, 1)
        summary["n_edges"] = n_edges
        _log(
            ctx,
            f"GRACE-SS complete. Graph shape: {grace_ss_res.cg_tig.shape}, {n_edges} edges, {elapsed:.1f}s",
        )

    # Named edge list for interpretation (agent-legible). Load the graph back
    # from disk so this works regardless of which algorithm branch ran.
    _graph_path = os.path.join(outdir, "estimated_graph.npy")
    if os.path.exists(_graph_path):
        from .inspection import diagnostics_from_graph, edges_from_graph

        _g = np.load(_graph_path)
        _pv_path = os.path.join(outdir, "pvalues.npy")
        _pv = np.load(_pv_path) if os.path.exists(_pv_path) else None
        summary["edges"] = edges_from_graph(_g, var_name_list, _pv)
        summary["diagnostics"] = diagnostics_from_graph(_g, var_name_list)

        if do_validate:
            _log(ctx, f"Validating: bootstrap stability over {n_bootstrap} windows...")
            from .bootstrap import temporal_bootstrap

            def _disc(sub):
                # Fresh CI test per window; mirror the main run's exact config
                # by closing over this command's parameters.
                _ci = _make_ci_test(ci_test, sub.values, device)
                if algorithm in ("cdnots", "cdnots+"):
                    from .cdnots.phase3_utils import run_cdnots, run_cdnots_plus

                    fn = run_cdnots if algorithm == "cdnots" else run_cdnots_plus
                    _extra = {"stable": stable} if algorithm == "cdnots" else {}
                    return fn(
                        df=sub,
                        indep_test=_ci,
                        num_lags=max_lag,
                        lag_list=lag_list_parsed,
                        include_C=include_c,
                        c_preset=c_preset,
                        alpha=alpha,
                        max_degree=max_degree,
                        verbose=False,
                        show_progress=False,
                        impute=impute,
                        impute_kwargs=_impute_kwargs,
                        **_extra,
                    ).cg_tig
                # cedar
                from .cedar.discovery import run_cedar

                return run_cedar(
                    df=sub,
                    ci_test=_ci,
                    max_lag=max_lag,
                    alpha_cond1=alpha_cond1,
                    alpha_cond2=alpha_cond2,
                    lag_method=lag_method,
                    lag_pvalue_method=lag_pvalue_method,
                    lag_alpha=lag_alpha,
                    normalize=None if normalize == "none" else normalize,
                    filter_children=filter_children,
                    include_lag0=include_lag0,
                    prune=not no_prune,
                    multi_lag=multi_lag,
                    multi_lag_keep=multi_lag_keep,
                    target_var=target_var,
                    verbose=False,
                    include_C=include_c,
                    c_preset=c_preset,
                    include_autoreg=include_autoreg,
                    assume_ar1=assume_ar1,
                    impute=impute,
                    impute_kwargs=_impute_kwargs,
                ).cg_tig

            boot = temporal_bootstrap(
                df,
                _disc,
                n_bootstrap=n_bootstrap,
                window_frac=window_frac,
                seed=ctx.obj["seed"] or 0,
                show_progress=not ctx.obj["quiet"],
            )
            P = boot["persistence"]
            np.save(os.path.join(outdir, "persistence.npy"), P)
            summary["output_files"]["persistence"] = "persistence.npy"
            # annotate each edge (same np.where order as edges_from_graph)
            for e, (i, j, k) in zip(summary["edges"], zip(*np.where(_g))):
                try:
                    e["persistence"] = round(float(P[i, j, k]), 3)
                except (IndexError, ValueError):
                    e["persistence"] = None
            summary["stability"] = {
                "n_bootstrap": n_bootstrap,
                "window_frac": window_frac,
                "stable_threshold": 0.6,
                "note": (
                    "persistence = fraction of windows an edge recurs; measures "
                    "robustness to sampling, NOT correctness (a systematic "
                    "artifact can be stable yet wrong)."
                ),
            }

    _save_json(summary, os.path.join(outdir, "summary.json"))
    _log(ctx, f"Results saved to {outdir}")

    # Instant evaluation against ground truth if provided
    if ground_truth is not None:
        from .utils.helpers import evaluate_graph

        gt = np.load(ground_truth)
        graph_path = os.path.join(outdir, "estimated_graph.npy")
        pred = np.load(graph_path)
        if gt.shape != pred.shape:
            n = min(gt.shape[0], pred.shape[0])
            tau = min(gt.shape[2], pred.shape[2])
            gt = gt[:n, :n, :tau]
            pred = pred[:n, :n, :tau]
        metrics = evaluate_graph(pred, gt)
        _save_json(
            {k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
            os.path.join(outdir, "metrics.json"),
        )
        summary["output_files"]["metrics"] = "metrics.json"
        summary["metrics"] = {
            k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()
        }
        click.echo(
            f"  Precision={metrics['Precision']:.3f}  Recall={metrics['TPR']:.3f}"
            f"  F1={metrics['F1']:.3f}  SHD={metrics['SHD']}"
        )

    if output_json:
        import json as _json

        click.echo(_json.dumps(summary, indent=2, default=str))


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------
@main.command()
@click.option(
    "--dataset",
    type=click.Choice(DATASET_CHOICES),
    default=None,
    help="Built-in dataset: ex1 (5-Node Nonlinear System), ex2 (6-Node Linear VAR), ex3 (Cascade Network 11-Node), henon (Hénon Chain).",
)
@click.option(
    "-n",
    "--n-vars",
    type=int,
    default=5,
    help="Number of variables (random generation).",
)
@click.option(
    "--n-links", type=int, default=None, help="Number of cross-links (auto if omitted)."
)
@click.option("--max-lag", type=int, default=3, help="Maximum lag.")
@click.option("-T", "--samples", type=int, default=500, help="Time series length.")
@click.option(
    "--density",
    type=click.Choice(DENSITY_CHOICES),
    default="sparse",
    help="Link density: sparse or dense.",
)
@click.option(
    "--dep-funcs",
    type=str,
    default="lin,lin,nl1,nl2",
    help="Comma-separated dependency functions.",
)
@click.pass_context
def generate(ctx, dataset, n_vars, n_links, max_lag, samples, density, dep_funcs):
    """Generate synthetic time series data with known causal structure."""
    outdir = _make_output_dir(ctx.obj["output_dir"], "generate")
    seed = ctx.obj["seed"] or 42

    if dataset is not None:
        _log(ctx, f"Loading built-in dataset: {dataset}")
        from .synthetic_data.synthetic_datasets import load_dataset

        result = load_dataset(dataset, seed=seed, T=samples)
        df = result["df"]
        gt = result["ground_truth"]
        meta = {
            "dataset": dataset,
            "seed": seed,
            "T": samples,
            "n_vars": df.shape[1],
            "max_lag": result["max_lag"],
            "var_names": list(df.columns),
        }
    else:
        _log(
            ctx,
            f"Generating random SCP: {n_vars} vars, max_lag={max_lag}, density={density}",
        )
        from .synthetic_data.synthetic_datasets import SCPGraphGenerator

        funcs = tuple(dep_funcs.split(","))
        gen = SCPGraphGenerator(
            n_vars=n_vars,
            max_lag=max_lag,
            density=density,
            dependency_funcs=funcs,
        )
        result = gen.sample(seed=seed, T=samples, n_links=n_links)
        df = result["df"]
        gt = result["ground_truth"]
        meta = {
            "seed": seed,
            "T": samples,
            "n_vars": n_vars,
            "n_links": result["meta"]["n_links"],
            "max_lag": max_lag,
            "density": density,
            "dep_funcs": dep_funcs,
            "var_names": result["var_names"],
            "funcs_count": result["meta"]["funcs_count"],
            "attempts": result["meta"]["attempt"],
        }

    df.to_csv(os.path.join(outdir, "data.csv"), index=False)
    np.save(os.path.join(outdir, "ground_truth.npy"), gt)
    _save_json(meta, os.path.join(outdir, "meta.json"))

    n_edges = int(gt.sum())
    _log(
        ctx,
        f"Generated {df.shape[0]} samples, {df.shape[1]} variables, {n_edges} edges",
    )
    _log(ctx, f"Saved to {outdir}")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
@main.command()
@click.argument("graph_true", type=click.Path(exists=True))
@click.argument("graph_discovered", type=click.Path(exists=True))
@click.option(
    "--var-names", type=str, default=None, help="Comma-separated variable names."
)
@click.option(
    "--exclude-self-loops",
    is_flag=True,
    help="Exclude autodependencies from metrics.",
)
@click.option("--save-plot/--no-plot", default=True, help="Save comparison plot.")
@click.option(
    "--plot-format",
    type=click.Choice(PLOT_FORMAT_CHOICES),
    default="png",
    help="Plot format.",
)
@click.pass_context
def evaluate(
    ctx,
    graph_true,
    graph_discovered,
    var_names,
    exclude_self_loops,
    save_plot,
    plot_format,
):
    """Compare discovered graph against ground truth.

    Computes TPR, FPR, Precision, Recall, F1, SHD, and pair-level metrics.
    """
    from matplotlib import pyplot as plt

    from .plotting.compare import compare_graphs, plot_metrics_summary
    from .utils.helpers import evaluate_graph

    outdir = _make_output_dir(ctx.obj["output_dir"], "evaluate")
    var_name_list = _parse_comma_list(var_names)

    gt = np.load(graph_true)
    pred = np.load(graph_discovered)

    # If shapes differ (e.g. discovered includes C variable), trim to match
    if gt.shape != pred.shape:
        n = min(gt.shape[0], pred.shape[0])
        tau = min(gt.shape[2], pred.shape[2])
        _log(
            ctx,
            f"Shape mismatch (gt={gt.shape}, pred={pred.shape}), trimming to ({n},{n},{tau})",
        )
        gt = gt[:n, :n, :tau]
        pred = pred[:n, :n, :tau]

    _log(ctx, f"Ground truth shape: {gt.shape}, Discovered shape: {pred.shape}")

    metrics = evaluate_graph(pred, gt, exclude_self_loops=exclude_self_loops)

    display_keys = [
        "Precision",
        "TPR",
        "F1",
        "SHD",
        "FPR",
        "TP",
        "FP",
        "FN",
        "Precision_pair",
        "TPR_pair",
        "F1_pair",
        "SHD_pair",
    ]
    click.echo("\n  Evaluation Metrics")
    click.echo("  " + "-" * 36)
    for k in display_keys:
        if k in metrics:
            v = metrics[k]
            if isinstance(v, float):
                click.echo(f"  {k:<18s}  {v:.4f}")
            else:
                click.echo(f"  {k:<18s}  {v}")
    click.echo()

    serializable = {
        k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()
    }
    _save_json(serializable, os.path.join(outdir, "metrics.json"))

    if save_plot:
        try:
            fig, ax = compare_graphs(gt, pred, var_names=var_name_list)
            fname = f"comparison.{plot_format}"
            fig.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
            plt.close(fig)
            _log(ctx, f"Saved comparison plot: {fname}")
        except Exception as e:
            _log(ctx, f"Warning: could not save comparison plot: {e}")

        try:
            fig, ax = plot_metrics_summary(gt, pred)
            fname = f"metrics_bar.{plot_format}"
            fig.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
            plt.close(fig)
            _log(ctx, f"Saved metrics bar chart: {fname}")
        except Exception as e:
            _log(ctx, f"Warning: could not save metrics bar chart: {e}")

    _log(ctx, f"Results saved to {outdir}")


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------
@main.command()
@click.argument("graph", type=click.Path(exists=True))
@click.option(
    "--type",
    "plot_type",
    type=click.Choice(["network", "time-series"]),
    default="network",
    help="Plot type.",
)
@click.option(
    "--val-matrix",
    type=click.Path(exists=True),
    default=None,
    help="Path to val_matrix.npy.",
)
@click.option(
    "--var-names", type=str, default=None, help="Comma-separated variable names."
)
@click.option(
    "--figsize", type=str, default=None, help="Figure size as W,H (e.g. 10,8)."
)
@click.option(
    "--plot-format",
    type=click.Choice(PLOT_FORMAT_CHOICES),
    default="png",
    help="Plot format.",
)
@click.option(
    "--save-name",
    type=str,
    default=None,
    help="Custom output filename (without extension).",
)
@click.pass_context
def plot(ctx, graph, plot_type, val_matrix, var_names, figsize, plot_format, save_name):
    """Plot a saved causal graph."""
    from matplotlib import pyplot as plt

    from .plotting._core import plot_graph, plot_time_series_graph

    outdir = _make_output_dir(ctx.obj["output_dir"], "plot")
    var_name_list = _parse_comma_list(var_names)
    figsize_parsed = _parse_figsize(figsize)

    g = np.load(graph, allow_pickle=True)
    vm = np.load(val_matrix) if val_matrix else None

    _log(ctx, f"Graph shape: {g.shape}")

    if save_name is None:
        save_name = plot_type

    kwargs = {}
    if var_name_list:
        kwargs["var_names"] = var_name_list
    if figsize_parsed:
        kwargs["figsize"] = figsize_parsed
    if vm is not None:
        kwargs["val_matrix"] = vm

    if plot_type == "network":
        result = plot_graph(g, **kwargs)
    else:
        result = plot_time_series_graph(g, **kwargs)

    fig = result[0] if isinstance(result, tuple) else result

    fname = f"{save_name}.{plot_format}"
    fig.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)

    _log(ctx, f"Saved {fname} to {outdir}")


# ---------------------------------------------------------------------------
# dowhy — effect estimation, SCM fitting, root cause analysis
# ---------------------------------------------------------------------------
@main.group("dowhy")
def dowhy_group():
    """DoWhy-powered effect estimation, SCM fitting, and root cause analysis.

    Requires DoWhy >=0.11 (install with: pip install dowhy).
    """
    pass


@dowhy_group.command("effect")
@click.argument("graph_path", metavar="GRAPH", type=click.Path(exists=True))
@click.option(
    "--data",
    "data_path",
    required=True,
    type=click.Path(exists=True),
    help="CSV data file.",
)
@click.option("--treatment", required=True, help="Treatment variable name (e.g. X0).")
@click.option("--outcome", required=True, help="Outcome variable name (e.g. X2).")
@click.option(
    "--lag",
    "treatment_lag",
    type=int,
    default=1,
    show_default=True,
    help="Lag of treatment.",
)
@click.option(
    "--method",
    default="backdoor.linear_regression",
    show_default=True,
    help="DoWhy estimation method.",
)
@click.option("--no-ci", is_flag=True, help="Skip confidence interval computation.")
@click.option(
    "--bootstraps",
    type=int,
    default=100,
    show_default=True,
    help="Bootstrap replications for CIs.",
)
@click.pass_context
def dowhy_effect(
    ctx,
    graph_path,
    data_path,
    treatment,
    outcome,
    treatment_lag,
    method,
    no_ci,
    bootstraps,
):
    """Estimate the causal effect of TREATMENT(t-LAG) on OUTCOME(t).

    GRAPH is a .npy file produced by the discover command.
    """
    try:
        from .effects.effect import estimate_effect, print_effect_summary
    except ImportError as e:
        raise click.ClickException(str(e))

    graph = np.load(graph_path)
    data = pd.read_csv(data_path)
    estimate = estimate_effect(
        graph,
        data,
        treatment=treatment,
        outcome=outcome,
        treatment_lag=treatment_lag,
        method=method,
        confidence_intervals=not no_ci,
        n_bootstraps=bootstraps,
    )
    print_effect_summary(estimate, treatment, outcome, treatment_lag)


@dowhy_group.command("fit-scm")
@click.argument("graph_path", metavar="GRAPH", type=click.Path(exists=True))
@click.option(
    "--data",
    "data_path",
    required=True,
    type=click.Path(exists=True),
    help="CSV data file.",
)
@click.option(
    "--mechanism",
    "mechanism_type",
    type=click.Choice(["auto", "linear", "gp"]),
    default="auto",
    show_default=True,
    help="Causal mechanism type.",
)
@click.option(
    "--outdir",
    default=".",
    show_default=True,
    type=click.Path(),
    help="Directory to save fitted SCM.",
)
@click.pass_context
def dowhy_fit_scm(ctx, graph_path, data_path, mechanism_type, outdir):
    """Fit a Structural Causal Model to the discovered graph.

    GRAPH is a .npy file produced by the discover command.
    Saves fitted SCM info and prints mechanism summaries.
    """
    import pickle

    try:
        from .effects.scm import fit_scm
    except ImportError as e:
        raise click.ClickException(str(e))

    graph = np.load(graph_path)
    data = pd.read_csv(data_path)
    scm, dag, lagged_df = fit_scm(graph, data, mechanism_type=mechanism_type)

    os.makedirs(outdir, exist_ok=True)
    scm_path = os.path.join(outdir, "scm.pkl")
    with open(scm_path, "wb") as f:
        pickle.dump({"scm": scm, "dag": dag}, f)
    lagged_df.to_csv(os.path.join(outdir, "lagged_data.csv"), index=False)

    click.echo(f"SCM fitted with {mechanism_type} mechanisms.")
    click.echo(f"Nodes: {list(dag.nodes)}")
    click.echo(f"Saved SCM to {scm_path}")


@dowhy_group.command("root-cause")
@click.argument("graph_path", metavar="GRAPH", type=click.Path(exists=True))
@click.option(
    "--data",
    "data_path",
    required=True,
    type=click.Path(exists=True),
    help="CSV data file (normal/training data).",
)
@click.option(
    "--target", required=True, help="Target variable to explain anomaly in (e.g. X2)."
)
@click.option(
    "--anomaly-row",
    "anomaly_json",
    default=None,
    help='JSON dict of anomalous observation, e.g. \'{"X0": 2.1, "X1": -0.5}\'.',
)
@click.option(
    "--anomaly-index",
    type=int,
    default=-1,
    show_default=True,
    help="Row index in data to use as anomaly (used if --anomaly-row not given).",
)
@click.option(
    "--mechanism",
    "mechanism_type",
    type=click.Choice(["auto", "linear", "gp"]),
    default="auto",
    show_default=True,
)
@click.option(
    "--n-samples",
    type=int,
    default=2000,
    show_default=True,
    help="Monte Carlo samples for attribution.",
)
@click.pass_context
def dowhy_root_cause(
    ctx,
    graph_path,
    data_path,
    target,
    anomaly_json,
    anomaly_index,
    mechanism_type,
    n_samples,
):
    """Attribute an anomaly in TARGET to its upstream causes.

    GRAPH is a .npy file produced by the discover command.
    """
    try:
        from .effects.root_cause import attribute_anomaly
        from .effects.scm import fit_scm
    except ImportError as e:
        raise click.ClickException(str(e))

    graph = np.load(graph_path)
    data = pd.read_csv(data_path)
    scm, dag, lagged_df = fit_scm(graph, data, mechanism_type=mechanism_type)

    if anomaly_json:
        anomaly_dict = json.loads(anomaly_json)
        anomaly = pd.Series(anomaly_dict)
        # Fill missing cols from last row of lagged_df
        for col in lagged_df.columns:
            if col not in anomaly.index:
                anomaly[col] = lagged_df[col].iloc[-1]
        anomaly = anomaly[lagged_df.columns]
    else:
        anomaly = lagged_df.iloc[anomaly_index]

    attrs = attribute_anomaly(
        scm, lagged_df, anomaly, target=target, n_samples=n_samples
    )
    click.echo(f"\nAnomaly attribution for '{target}':")
    click.echo(attrs.to_string(index=False))


@dowhy_group.command("validate")
@click.argument("graph_path", metavar="GRAPH", type=click.Path(exists=True))
@click.option(
    "--data", "data_path", required=True, type=click.Path(exists=True), help="CSV data."
)
@click.option(
    "--test",
    "test_type",
    type=click.Choice(["falsify", "refute", "evaluate", "all"]),
    default="all",
    show_default=True,
    help="Validation test to run.",
)
@click.option(
    "--mechanism",
    "mechanism_type",
    type=click.Choice(["auto", "linear", "gp"]),
    default="auto",
    show_default=True,
)
@click.pass_context
def dowhy_validate(ctx, graph_path, data_path, test_type, mechanism_type):
    """Validate the discovered graph against data (no ground truth needed).

    GRAPH is a .npy file produced by the discover command.
    """
    try:
        from .effects.validate import (
            evaluate_model,
            falsify_graph,
            refute_structure,
        )
    except ImportError as e:
        raise click.ClickException(str(e))

    graph = np.load(graph_path)
    data = pd.read_csv(data_path)

    if test_type in ("falsify", "all"):
        click.echo("\n--- Graph Falsification ---")
        result = falsify_graph(graph, data)
        click.echo(f"Falsifiable: {result['falsifiable']}")
        click.echo(f"Falsified:   {result['falsified']}")

    if test_type in ("refute", "all"):
        click.echo("\n--- Structural Refutation ---")
        df_result = refute_structure(graph, data)
        click.echo(df_result.to_string(index=False))

    if test_type in ("evaluate", "all"):
        click.echo("\n--- Model Evaluation ---")
        result = evaluate_model(graph, data, mechanism_type=mechanism_type)
        click.echo(result["summary"])


@dowhy_group.command("strength")
@click.argument("graph_path", metavar="GRAPH", type=click.Path(exists=True))
@click.option(
    "--data", "data_path", required=True, type=click.Path(exists=True), help="CSV data."
)
@click.option("--target", required=True, help="Target variable (e.g. Mek).")
@click.option(
    "--mechanism",
    "mechanism_type",
    type=click.Choice(["auto", "linear", "gp"]),
    default="auto",
    show_default=True,
)
@click.pass_context
def dowhy_strength(ctx, graph_path, data_path, target, mechanism_type):
    """Compute arrow strength and parent relevance for a target variable.

    GRAPH is a .npy file produced by the discover command.
    """
    try:
        from .effects.influence import arrow_strength, parent_relevance
    except ImportError as e:
        raise click.ClickException(str(e))

    graph = np.load(graph_path)
    data = pd.read_csv(data_path)

    click.echo(f"\n--- Arrow Strength: → {target} ---")
    strengths = arrow_strength(graph, data, target, mechanism_type=mechanism_type)
    click.echo(strengths.to_string(index=False))

    click.echo(f"\n--- Parent Relevance: {target} ---")
    relevance = parent_relevance(graph, data, target, mechanism_type=mechanism_type)
    click.echo(relevance.to_string(index=False))


@dowhy_group.command("drift")
@click.argument("graph_path", metavar="GRAPH", type=click.Path(exists=True))
@click.option(
    "--data-old",
    "data_old_path",
    required=True,
    type=click.Path(exists=True),
    help="Reference period CSV.",
)
@click.option(
    "--data-new",
    "data_new_path",
    required=True,
    type=click.Path(exists=True),
    help="New/current period CSV.",
)
@click.option("--target", required=True, help="Target variable to explain shift in.")
@click.option(
    "--mechanism",
    "mechanism_type",
    type=click.Choice(["auto", "linear", "gp"]),
    default="auto",
    show_default=True,
)
@click.option(
    "--robust", is_flag=True, help="Use multiply-robust estimator (ICML 2024)."
)
@click.pass_context
def dowhy_drift(
    ctx, graph_path, data_old_path, data_new_path, target, mechanism_type, robust
):
    """Attribute a distributional shift in TARGET to upstream mechanism changes.

    GRAPH is a .npy file produced by the discover command.
    """
    try:
        from .effects.influence import distribution_change
    except ImportError as e:
        raise click.ClickException(str(e))

    graph = np.load(graph_path)
    data_old = pd.read_csv(data_old_path)
    data_new = pd.read_csv(data_new_path)

    click.echo(f"\n--- Distribution Change: {target} ---")
    result = distribution_change(
        graph,
        data_old,
        data_new,
        target,
        mechanism_type=mechanism_type,
        robust=robust,
    )
    click.echo(result.to_string(index=False))


# ---------------------------------------------------------------------------
# install-skill
# ---------------------------------------------------------------------------
@main.command("install-skill")
@click.option(
    "--copy",
    "do_copy",
    is_flag=True,
    default=False,
    help="Copy files instead of symlinking (for environments that reject symlinks).",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Replace an existing non-symlink target.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print what would be done without changing anything.",
)
def install_skill(do_copy, force, dry_run):
    """Install the causal-ts-discovery agent skill into the local harness dirs.

    Symlinks (default) the packaged skill into ``~/.claude/skills/`` (Claude Code)
    and ``~/.agents/skills/`` (Codex and other Agent-Skills harnesses) so an agent
    can drive causal-ts.  Symlinks stay current on ``pip install -U``; pass
    ``--copy`` for a static copy instead.
    """
    import shutil
    from importlib.resources import files

    skill_name = "causal-ts-discovery"
    src_path = os.fspath(files("causalts") / "skills" / skill_name)
    if not os.path.isdir(src_path):
        raise click.ClickException(f"Packaged skill not found at {src_path}")

    dests = [
        os.path.expanduser(f"~/.claude/skills/{skill_name}"),
        os.path.expanduser(f"~/.agents/skills/{skill_name}"),
    ]

    for dest in dests:
        if dry_run:
            verb = "copy" if do_copy else "symlink"
            click.echo(f"[dry-run] would {verb} {src_path} -> {dest}")
            continue

        os.makedirs(os.path.dirname(dest), exist_ok=True)

        if os.path.islink(dest):
            os.unlink(dest)  # replace our own (or any) symlink idempotently
        elif os.path.exists(dest):
            if not force:
                raise click.ClickException(
                    f"{dest} exists and is not a symlink; re-run with --force to replace."
                )
            shutil.rmtree(dest) if os.path.isdir(dest) else os.remove(dest)

        if do_copy:
            shutil.copytree(src_path, dest)
            click.echo(f"Copied {skill_name} -> {dest}")
        else:
            os.symlink(src_path, dest)
            click.echo(f"Symlinked {skill_name} -> {dest}")


if __name__ == "__main__":
    main()
