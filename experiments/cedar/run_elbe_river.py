# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Elbe River benchmark for CEDAR.

Runs CEDAR on the 12-station Elbe main branch from CausalRivers.
Ground truth: chain graph (11 directed edges, upstream → downstream).
Evaluation: 2D summary graph (any-lag match) since propagation speed varies.

Data source: CausalRivers (https://github.com/causalrivers/causalrivers)
Download product.zip to DATA_DIR before running.

Usage:
    # Reproduce paper results (F1=0.833, TP=10, FP=3):
    python run_elbe_river.py --regime-module --pelt --pelt-min-size 300 \
        --alpha 0.01 --lag-filter ar_relative --lag-filter-frac 0.97 \
        --no-mad-clip --segment-threshold 0.25

    # Regenerate comparison figure from saved graphs (fast):
    python run_elbe_river.py --plot-only --regime-module --pelt \
        --pelt-min-size 300 --no-mad-clip

    # Run single + PELT and generate figure in one step:
    python run_elbe_river.py --plot --regime-module --pelt --pelt-min-size 300 \
        --alpha 0.01 --lag-filter ar_relative --lag-filter-frac 0.97 \
        --no-mad-clip --segment-threshold 0.25
"""

import argparse
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DATA_DIR = "/tmp/causalrivers/product"
ELBE_STATIONS = [166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177]
RESAMPLE_H = 3
MAX_LAG = 5
ALPHA = 0.01
IMPUTE = None
CUTOFF_DATE = "2023-01-01"
MAD_FACTOR = 5
OUT_DIR = Path(__file__).parent / "elbe_results"


def parse_args():
    p = argparse.ArgumentParser(description="CEDAR on Elbe River")
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--resample-hours", type=int, default=RESAMPLE_H)
    p.add_argument("--max-lag", type=int, default=MAX_LAG)
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument(
        "--impute",
        default=IMPUTE,
        choices=[None, "pairwise_complete", "var_em"],
        type=lambda x: None if x == "None" else x,
    )
    p.add_argument("--no-c-node", action="store_true")
    p.add_argument("--no-prune", action="store_true")
    p.add_argument(
        "--lag-filter",
        default=None,
        choices=[None, "ar_relative"],
        type=lambda x: None if x == "None" else x,
    )
    p.add_argument("--lag-filter-frac", type=float, default=0.97)
    p.add_argument(
        "--segment-threshold",
        type=float,
        default=0.25,
        help="Within-regime segment majority threshold (default 0.25)",
    )
    p.add_argument(
        "--regime-module",
        action="store_true",
        help="Use causalts.regime module for regime-conditional discovery",
    )
    p.add_argument(
        "--pelt",
        action="store_true",
        help="Use PELT changepoint detection for regime segmentation (requires ruptures)",
    )
    p.add_argument(
        "--pelt-pen",
        type=float,
        default=5000,
        help="PELT penalty (higher=fewer segments, default 5000)",
    )
    p.add_argument(
        "--pelt-min-size",
        type=int,
        default=250,
        help="PELT minimum segment size in samples (default 250)",
    )
    p.add_argument(
        "--pelt-signal",
        default="mean",
        choices=["mean", "volatility"],
        help="Signal for PELT: 'mean' (discharge) or 'volatility' (rolling std)",
    )
    p.add_argument(
        "--no-mad-clip",
        action="store_true",
        help="Disable MAD outlier clipping (better for PELT regime detection)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument(
        "--plot",
        action="store_true",
        help="Run single + PELT, save graphs, then generate comparison figure",
    )
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Load saved graphs (from elbe_results/ and elbe_results_single/) and regenerate figure only",
    )
    p.add_argument(
        "--figure-out",
        type=Path,
        default=Path(__file__).parents[2]
        / "papers"
        / "cedar"
        / "figures"
        / "elbe_compare.pdf",
        help="Output path for the comparison figure",
    )
    p.add_argument(
        "--cr-repo",
        type=Path,
        default=Path("/tmp/causalrivers"),
        help="Path to CausalRivers repo (for shapefile map backgrounds)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading (mirrors examples/elbe_river.ipynb)
# ---------------------------------------------------------------------------


def load_elbe_data(data_dir, resample_h, cutoff_date, mad_factor, mad_clip=True):
    data_dir = Path(data_dir)
    full_G = pickle.load(open(data_dir / "rivers_east_germany.p", "rb"))
    ts_raw = pd.read_csv(
        data_dir / "rivers_ts_east_germany.csv", index_col=0, parse_dates=True
    )

    ts_res = ts_raw.resample(f"{resample_h}h").mean()
    avail = [n for n in ELBE_STATIONS if str(n) in ts_res.columns]
    ts_basin = ts_res[ts_res.index < cutoff_date][[str(n) for n in avail]]

    if mad_clip:
        for col in ts_basin.columns:
            s = ts_basin[col].dropna()
            med = s.median()
            mad = np.median(np.abs(s - med))
            if mad > 1e-10:
                limit = mad_factor * mad * 1.4826
                ts_basin[col] = ts_basin[col].clip(med - limit, med + limit)

    basin = sorted(avail)
    G_basin = full_G.subgraph(basin).copy()
    d = len(basin)
    node_idx = {n: i for i, n in enumerate(basin)}

    # label_mat[effect, cause] = 1 (GRACE notebook convention)
    label_mat = np.zeros((d, d), dtype=np.float32)
    for u, v in G_basin.edges():
        if u in node_idx and v in node_idx:
            label_mat[node_idx[v], node_idx[u]] = 1.0

    # gt_ce[cause, effect] = 1 (cg_tig convention)
    gt_ce = (label_mat.T > 0.5).astype(np.int8)

    ts_basin.columns = list(range(d))
    var_names = [f"S{n}" for n in basin]

    n_missing = ts_basin.isna().sum().sum()
    n_total = ts_basin.shape[0] * ts_basin.shape[1]
    print(f"Loaded: d={d} stations, T={len(ts_basin)} samples ({resample_h}h)")
    print(f"Date range: {ts_basin.index[0].date()} – {ts_basin.index[-1].date()}")
    print(
        f"True edges: {int(gt_ce.sum())}  Missing: {n_missing}/{n_total} ({100*n_missing/n_total:.1f}%)"
    )

    return ts_basin, gt_ce, var_names


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_2d(est_ce, gt_ce):
    d = gt_ce.shape[0]
    est = est_ce[:d, :d].copy()
    np.fill_diagonal(est, 0)
    gt = gt_ce.copy()
    np.fill_diagonal(gt, 0)
    tp = int(((est == 1) & (gt == 1)).sum())
    fp = int(((est == 1) & (gt == 0)).sum())
    fn = int(((est == 0) & (gt == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "Precision": prec, "Recall": rec, "F1": f1}


def graph_to_2d(graph_3d, d):
    est = graph_3d[:d, :d, 1:].any(axis=2).astype(np.int8)
    np.fill_diagonal(est, 0)
    return est


# ---------------------------------------------------------------------------
# Single CEDAR run on a dataframe
# ---------------------------------------------------------------------------


def run_cedar_single(
    df,
    max_lag,
    alpha,
    impute,
    include_C,
    prune,
    lag_method="dcor",
    lag_filter=None,
    lag_filter_frac=0.97,
):
    from causalts.cedar import run_cedar
    from causalts.ci_tests.parcorr_gpu import ParCorrGPU

    data = np.ascontiguousarray(df.values)
    ci = ParCorrGPU(data)
    result = run_cedar(
        df=df,
        ci_test=ci,
        max_lag=max_lag,
        alpha_cond1=alpha,
        alpha_cond2=alpha,
        lag_method=lag_method,
        impute=impute,
        include_C=include_C,
        prune=prune,
        lag_filter=lag_filter,
        lag_filter_frac=lag_filter_frac,
        verbose=False,
    )
    return result


# ---------------------------------------------------------------------------
# Regime-Stratified Discovery
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Compare-graphs figure
# ---------------------------------------------------------------------------


def make_compare_figure(data_dir, single_dir, pelt_dir, out_path, cr_repo=None):
    """Load saved graphs and create 3-panel comparison figure.

    Panel 1: geographic map (ground truth chain).
    Panel 2: CEDAR single run (compare_graphs vs ground truth).
    Panel 3: CEDAR + PELT regime-conditional.

    Edges are compared in 2D (any-lag match).
    """
    import pickle

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from causalts.plotting import compare_graphs

    data_dir = Path(data_dir)
    single_dir = Path(single_dir)
    pelt_dir = Path(pelt_dir)

    # --- Load data and ground truth ---
    full_G = pickle.load(open(data_dir / "rivers_east_germany.p", "rb"))
    basin = sorted(ELBE_STATIONS)
    G_basin = full_G.subgraph(basin).copy()
    d = len(basin)
    node_idx = {n: i for i, n in enumerate(basin)}
    var_names = [f"S{n}" for n in basin]

    gt_ce = np.zeros((d, d), dtype=np.int8)
    for u, v in G_basin.edges():
        if u in node_idx and v in node_idx:
            gt_ce[node_idx[u], node_idx[v]] = 1

    # --- Load saved graphs ---
    g_single = np.load(single_dir / "cedar_graph.npy")
    g_pelt = np.load(pelt_dir / "cedar_graph.npy")
    single_2d = graph_to_2d(g_single, d)
    pelt_2d = graph_to_2d(g_pelt, d)

    def metrics(g2d):
        tp = int(((g2d == 1) & (gt_ce == 1)).sum())
        fp = int(((g2d == 1) & (gt_ce == 0)).sum())
        fn = int(((g2d == 0) & (gt_ce == 1)).sum())
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        return tp, fp, fn, f1

    tp_s, fp_s, fn_s, f1_s = metrics(single_2d)
    tp_p, fp_p, fn_p, f1_p = metrics(pelt_2d)
    print(f"  Single:  TP={tp_s} FP={fp_s} FN={fn_s} F1={f1_s:.3f}")
    print(f"  PELT:    TP={tp_p} FP={fp_p} FN={fn_p} F1={f1_p:.3f}")

    # Geographic coordinates (used for geomap panel only)
    lons = np.array([full_G.nodes[n].get("p", [0, 0])[1] for n in basin], dtype=float)
    lats = np.array([full_G.nodes[n].get("p", [0, 0])[0] for n in basin], dtype=float)

    # Wrap 2D graphs at lag slot 1 (no lag labels printed)
    def to_3d(g2d):
        g3d = np.zeros((d, d, 2), dtype=np.int8)
        g3d[:, :, 1] = g2d
        return g3d

    gt_3d = to_3d(gt_ce)
    single_3d = to_3d(single_2d)
    pelt_3d = to_3d(pelt_2d)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Panel 1: Geographic map ---
    ax0 = axes[0]
    cr_repo = Path(cr_repo) if cr_repo else None
    map_shp = (
        cr_repo / "product/visualization/east_germany/east_german_map.shp"
        if cr_repo
        else None
    )
    river_shp = (
        cr_repo / "product/visualization/east_germany/river_east_german_map.shp"
        if cr_repo
        else None
    )

    try:
        import geopandas as gpd

        if map_shp and map_shp.exists():
            gpd.read_file(map_shp).plot(
                color="#e8f5e9", ax=ax0, alpha=0.5, linewidth=1.5, edgecolor="#333"
            )
        if river_shp and river_shp.exists():
            gpd.read_file(river_shp).plot(
                color="#90caf9", ax=ax0, alpha=0.4, linewidth=1.5, edgecolor="#42a5f5"
            )
    except Exception:
        ax0.set_facecolor("#e8f5e9")

    heights = [full_G.nodes[n].get("H", 0) for n in basin]
    sc = ax0.scatter(
        lons,
        lats,
        c=heights,
        cmap="RdYlBu_r",
        s=120,
        edgecolors="#333",
        linewidth=1,
        zorder=5,
    )
    plt.colorbar(sc, ax=ax0, label="Elevation (m)", shrink=0.7)
    for i, n in enumerate(basin):
        ax0.annotate(
            f"S{n}",
            (lons[i], lats[i]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
            zorder=6,
        )
    for u, v in G_basin.edges():
        if u in node_idx and v in node_idx:
            i_u, i_v = node_idx[u], node_idx[v]
            ax0.annotate(
                "",
                xy=(lons[i_v], lats[i_v]),
                xytext=(lons[i_u], lats[i_u]),
                arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.5, alpha=0.8),
                zorder=4,
            )
    pad = 0.3
    ax0.set_xlim(lons.min() - pad, lons.max() + pad)
    ax0.set_ylim(lats.min() - pad, lats.max() + pad)
    ax0.set_xlabel("Longitude")
    ax0.set_ylabel("Latitude")
    ax0.set_title(
        "Ground Truth\n(11 directed edges, upstream → downstream)", fontsize=10
    )
    compare_graphs(
        graph_true=gt_3d,
        graph_discovered=single_3d,
        var_names=var_names,
        node_layout="circular",
        node_size=0.15,
        arrow_linewidth=2,
        link_label_fontsize=0,
        tp_color="black",
        show_legend=False,
        legend_loc=None,
        node_label_size=8,
        title=f"CEDAR single ($L{{=}}5$)\nTP={tp_s}  FP={fp_s}  FN={fn_s}  F1={f1_s:.3f}",
        fig_ax=(fig, axes[1]),
    )
    compare_graphs(
        graph_true=gt_3d,
        graph_discovered=pelt_3d,
        var_names=var_names,
        node_layout="circular",
        node_size=0.15,
        arrow_linewidth=2,
        link_label_fontsize=0,
        tp_color="black",
        legend_loc=None,
        node_label_size=8,
        title=f"CEDAR + PELT (regime-conditional)\nTP={tp_p}  FP={fp_p}  FN={fn_p}  F1={f1_p:.3f}",
        fig_ax=(fig, axes[2]),
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nSaved figure: {out_path}")


def run(args):
    df, gt_ce, var_names = load_elbe_data(
        args.data_dir,
        args.resample_hours,
        CUTOFF_DATE,
        MAD_FACTOR,
        mad_clip=not args.no_mad_clip,
    )
    if args.plot_only:
        make_compare_figure(
            args.data_dir,
            args.out_dir.parent / "elbe_results_single",
            args.out_dir,
            args.figure_out,
            cr_repo=args.cr_repo,
        )
        return

    if args.plot:
        _run_and_plot(args, df, gt_ce, var_names)
        return

    if args.regime_module:
        run_regime_module(args, df, gt_ce, var_names)
    else:
        run_single(args, df, gt_ce, var_names)


def _run_and_plot(args, df, gt_ce, var_names):
    """Run single + PELT (saves graphs), then produce the compare_graphs figure."""
    # Run single and PELT (saves graphs via save_results)
    print("\n[1/2] CEDAR single run...")
    single_args = argparse.Namespace(**vars(args))
    single_args.out_dir = args.out_dir.parent / "elbe_results_single"
    single_args.lag_filter = (
        None  # no lag filter for single run — shows nonstationarity impact
    )
    run_single(single_args, df, gt_ce, var_names)

    print("\n[2/2] CEDAR + PELT (regime-conditional)...")
    run_regime_module(args, df, gt_ce, var_names)

    make_compare_figure(
        args.data_dir,
        args.out_dir.parent / "elbe_results_single",
        args.out_dir,
        args.figure_out,
        cr_repo=args.cr_repo,
    )


def run_regime_module(args, df, gt_ce, var_names):
    """Run regime-conditional discovery using causalts.regime module."""
    from causalts.regime import (
        RegimeConfig,
        RegimeParams,
        run_regime_discovery,
    )

    d = len(var_names)
    n_regimes = 3
    regime_max_lags = [8, 5, 3]

    config = RegimeConfig(
        n_regimes=n_regimes,
        regime_params=[
            RegimeParams(max_lag=ml, alpha=args.alpha) for ml in regime_max_lags
        ],
        aggregation="or",
        buffer=max(regime_max_lags),
        segment_threshold=args.segment_threshold,
        regime_names=["low-flow", "normal", "high-flow"],
    )

    def cedar_fn(sub_df, max_lag=5, alpha=0.01, **kw):
        from causalts.cedar import run_cedar
        from causalts.ci_tests.parcorr_gpu import ParCorrGPU
        from causalts.imputation import var_em_impute

        # Impute missing values before discovery
        if sub_df.isna().any().any():
            try:
                sub_df = var_em_impute(sub_df)
            except Exception:
                sub_df = sub_df.ffill().bfill()
        if sub_df.isna().any().any():
            sub_df = sub_df.ffill().bfill()

        data = np.ascontiguousarray(sub_df.values)
        ci = ParCorrGPU(data)
        return run_cedar(
            df=sub_df,
            ci_test=ci,
            max_lag=max_lag,
            alpha_cond1=alpha,
            alpha_cond2=alpha,
            lag_method="dcor",
            include_C=not args.no_c_node,
            prune=not args.no_prune,
            lag_filter=args.lag_filter,
            lag_filter_frac=args.lag_filter_frac,
            verbose=False,
        )

    # Set up detector
    detector = None
    if args.pelt:
        from causalts.regime import PeltRegimeDetector

        detector = PeltRegimeDetector(
            n_regimes=n_regimes,
            model="l2",
            min_size=args.pelt_min_size,
            pen=args.pelt_pen,
            signal=args.pelt_signal,
        )
        print(
            f"\nRegime-Conditional Discovery (PELT, pen={args.pelt_pen}, "
            f"signal={args.pelt_signal})"
        )
    else:
        print("\nRegime-Conditional Discovery (quantile regimes)")

    print(f"{'='*60}")
    for i, ml in enumerate(regime_max_lags):
        name = config.regime_names[i] if config.regime_names else f"regime-{i}"
        print(f"  {name}: max_lag={ml}, alpha={args.alpha}")

    t0 = time.time()
    result = run_regime_discovery(
        df,
        discovery_fn=cedar_fn,
        config=config,
        detector=detector,
        verbose=True,
    )
    elapsed = time.time() - t0

    # Evaluate
    est_2d = graph_to_2d(result.cg_tig, d)
    m = evaluate_2d(est_2d, gt_ce)

    print(f"\n{'='*60}")
    print(
        f"Aggregated ({result.aggregation_strategy}): "
        f"TP={m['TP']} FP={m['FP']} FN={m['FN']} "
        f"Prec={m['Precision']:.3f} Rec={m['Recall']:.3f} F1={m['F1']:.3f}"
    )
    print_edges(est_2d, gt_ce, var_names)

    # Per-regime results
    for r, g in result.regime_graphs.items():
        g_2d = g[:d, :d, 1:].any(axis=2).astype(np.int8)
        np.fill_diagonal(g_2d, 0)
        m_r = evaluate_2d(g_2d, gt_ce)
        rname = result.regime_names[r]
        print(
            f"  {rname}: TP={m_r['TP']} FP={m_r['FP']} FN={m_r['FN']} F1={m_r['F1']:.3f}"
        )

    # Regime summary
    summary = result.regime_summary()
    print(f"\nRegime summary: {summary}")
    print(f"Elapsed: {elapsed:.1f}s")

    save_results(
        args,
        m,
        result.cg_tig,
        elapsed,
        mode="regime_module",
        extra={
            "n_regimes": n_regimes,
            "regime_max_lags": regime_max_lags,
            "aggregation": config.aggregation,
        },
    )


def run_single(args, df, gt_ce, var_names):
    d = len(var_names)
    print(
        f"\nRunning CEDAR (impute={args.impute}, alpha={args.alpha}, "
        f"max_lag={args.max_lag}, C={not args.no_c_node}, prune={not args.no_prune}, "
        f"lag_filter={args.lag_filter}, frac={args.lag_filter_frac})"
    )

    t0 = time.time()
    result = run_cedar_single(
        df,
        args.max_lag,
        args.alpha,
        args.impute,
        not args.no_c_node,
        not args.no_prune,
        lag_filter=args.lag_filter,
        lag_filter_frac=args.lag_filter_frac,
    )
    elapsed = time.time() - t0

    graph = result.cg_tig[:d, :d, :]
    est_2d = graph_to_2d(graph, d)
    m = evaluate_2d(est_2d, gt_ce)

    print_results("CEDAR (single)", m, est_2d, gt_ce, graph, var_names, elapsed)
    save_results(args, m, graph, elapsed, mode="single")


def print_results(title, m, est_2d, gt_ce, graph_3d, var_names, elapsed):
    d = len(var_names)
    print(f"\n{'='*60}")
    print(f"{title} ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(
        f"  TP={m['TP']}  FP={m['FP']}  FN={m['FN']}  "
        f"Prec={m['Precision']:.3f}  Rec={m['Recall']:.3f}  F1={m['F1']:.3f}"
    )
    print_edges(est_2d, gt_ce, var_names)

    if graph_3d is not None:
        print("\nDetailed edges (with lags):")
        for c in range(d):
            for e in range(d):
                if c == e:
                    continue
                for lag in range(1, graph_3d.shape[2]):
                    if graph_3d[c, e, lag]:
                        true = "+" if gt_ce[c, e] else "-"
                        print(
                            f"  {var_names[c]}(t-{lag}) -> {var_names[e]}(t)  [{true}]"
                        )


def print_edges(est_2d, gt_ce, var_names):
    d = len(var_names)
    missed = []
    for c in range(d):
        for e in range(d):
            if gt_ce[c, e] and not est_2d[c, e]:
                missed.append(f"  {var_names[c]} -> {var_names[e]}")
    if missed:
        print(f"\nMissed ({len(missed)}):")
        for m in missed:
            print(m)


def save_results(args, m, graph, elapsed, mode="single", extra=None):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if graph is not None:
        np.save(args.out_dir / "cedar_graph.npy", graph)
    results = {
        "method": f"cedar_{mode}",
        "resample_hours": args.resample_hours,
        "max_lag": args.max_lag,
        "alpha": args.alpha,
        "impute": args.impute,
        "include_C": not args.no_c_node,
        "prune": not args.no_prune,
        "elapsed_s": round(elapsed, 1),
        **m,
    }
    if extra:
        results.update(extra)
    with open(args.out_dir / "cedar_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {args.out_dir}/")


if __name__ == "__main__":
    run(parse_args())
