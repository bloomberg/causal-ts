# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

#!/usr/bin/env python3
"""Generate all paper figures.

Usage:
    python generate_figures.py              # all figures
    python generate_figures.py scaling      # only F1/SHD vs d
    python generate_figures.py gt           # only ground truth graphs
    python generate_figures.py compare      # only CEDAR vs GT comparison
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from causalts.cedar import run_cedar
from causalts.ci_tests import ParCorrGPU, SplitKCIGPU
from causalts.plotting import compare_graphs, plot_graph
from causalts.synthetic_data.synthetic_datasets import confounded_trends, load_dataset
from causalts.utils import evaluate_graph

FIG_DIR = Path(__file__).parent / "figures"
PAPER_FIG_DIR = Path(__file__).resolve().parents[2] / "papers" / "cedar" / "figures"
FIG_DIR.mkdir(exist_ok=True)
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = Path(__file__).parent / "paper_results.csv"
ALPHA = 0.01

CASCADE_NAMES = [
    "Plcg",
    "PIP3",
    "PIP2",
    "PKC",
    "PKA",
    "P38",
    "Jnk",
    "Raf",
    "Mek",
    "Erk",
    "Akt",
]

# Publication style
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

ALGO_STYLE = {
    "cedar": {
        "color": "#2196F3",
        "marker": "o",
        "label": "CEDAR",
        "zorder": 10,
        "linewidth": 2.0,
        "markersize": 7,
    },
    "pcmci+": {
        "color": "#FF9800",
        "marker": "s",
        "label": "PCMCI+",
        "zorder": 5,
        "linewidth": 1.6,
        "markersize": 5,
    },
    "cdnots+": {
        "color": "#4CAF50",
        "marker": "^",
        "label": "CDNOTS+",
        "zorder": 5,
        "linewidth": 1.6,
        "markersize": 5,
    },
    "cdnots": {
        "color": "#9E9E9E",
        "marker": "D",
        "label": "CDNOTS",
        "zorder": 3,
        "linewidth": 1.4,
        "markersize": 5,
    },
}

TOPO_NAMES = {"er": "Erdős–Rényi", "sf": "Scale-free", "sw": "Small-world"}


# =============================================================================
# Scaling figures (F1 and SHD vs d)
# =============================================================================


def _plot_method(ax, x, y, yerr, style):
    is_ours = style.get("zorder", 5) >= 10
    c = style["color"]
    ax.plot(
        x,
        y,
        color=c,
        marker=style["marker"],
        label=style["label"],
        linewidth=style["linewidth"],
        markersize=style["markersize"],
        zorder=style["zorder"],
        markeredgecolor="white" if is_ours else c,
        markeredgewidth=1.2 if is_ours else 0.5,
    )
    if yerr is not None and len(x) > 1:
        ax.fill_between(
            x,
            y - yerr,
            y + yerr,
            color=c,
            alpha=0.15 if is_ours else 0.08,
            zorder=style["zorder"] - 1,
        )


def fig_scaling(metric="F1", T_values=None, ylabel=None, filename=None, ylim=None):
    import pandas as pd

    df = pd.read_csv(CSV_PATH)
    df = df[df["d"] <= 75]
    if T_values is None:
        T_values = [100, 200, 500]
    if ylabel is None:
        ylabel = metric
    if filename is None:
        filename = f"{metric.lower()}_vs_d.pdf"

    n_T = len(T_values)
    fig, axes = plt.subplots(n_T, 3, figsize=(10, 3.2 * n_T), squeeze=False)

    for row, T in enumerate(T_values):
        for col, topo in enumerate(["er", "sf", "sw"]):
            ax = axes[row, col]
            sub = df[(df["topo"] == topo) & (df["T"] == T)]
            for algo, style in ALGO_STYLE.items():
                asub = sub[sub["algo"] == algo]
                if len(asub) == 0:
                    continue
                agg = asub.groupby("d")[metric].agg(["mean", "std"]).reset_index()
                _plot_method(
                    ax, agg["d"].values, agg["mean"].values, agg["std"].values, style
                )
            if row == n_T - 1:
                ax.set_xlabel("Number of variables $d$")
            if col == 0:
                ax.set_ylabel(f"{ylabel}\n($T={T}$)")
            if row == 0:
                ax.set_title(TOPO_NAMES[topo], fontweight="medium", pad=8)
            d_vals = sorted(sub["d"].unique()) if len(sub) > 0 else [10, 20, 30, 40, 50]
            ax.set_xticks(d_vals)
            if ylim:
                ax.set_ylim(ylim)
            elif metric == "SHD":
                ax.set_ylim(bottom=0)
            ax.tick_params(axis="both", which="major", length=4)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.01),
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
        framealpha=0.95,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, filename)


# =============================================================================
# Ground truth graphs
# =============================================================================


def fig_ground_truth(name, data, var_names, figsize=(5, 4.5), **kwargs):
    defaults = dict(
        show_colorbar=False,
        show_autodependency_lags=True,
        node_size=0.22,
        arrow_linewidth=2.5,
        node_label_size=9,
        link_label_fontsize=9,
    )
    defaults.update(kwargs)

    gt = data["ground_truth"]
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_graph(gt, val_matrix=None, var_names=var_names, fig_ax=(fig, ax), **defaults)
    fig.tight_layout()
    _save(fig, f"gt_{name}.pdf")


# =============================================================================
# Compare graphs (CEDAR discovered vs ground truth)
# =============================================================================


def fig_compare(
    name,
    data,
    ci_name="parcorr",
    cedar_kw=None,
    var_names=None,
    figsize=(5.5, 4.5),
    seed=42,
):
    df = data["df"]
    gt = data["ground_truth"]
    max_lag = data.get("max_lag", data.get("maxlag", 3))
    if var_names is None:
        var_names = data.get("var_names", [str(i) for i in range(df.shape[1])])

    dummy = np.zeros((2, 2))
    ci = (
        SplitKCIGPU(dummy, device="cpu")
        if ci_name == "splitkci"
        else ParCorrGPU(dummy, device="cpu")
    )

    kw = dict(
        alpha_cond1=ALPHA,
        alpha_cond2=ALPHA,
        filter_children=False,
        prune=True,
        multi_lag=True,
        multi_lag_keep="first",
        include_autoreg=True,
        lag_method="partial_dcor",
    )
    if cedar_kw:
        kw.update(cedar_kw)

    res = run_cedar(df, ci, max_lag, **kw)
    d_orig = gt.shape[0]
    G_hat = res.cg_tig[:d_orig, :d_orig, :]

    m = evaluate_graph(G_hat, gt, exclude_self_loops=True)
    title = (
        f"F1={m['F1']:.2f}, Prec={m['Precision']:.2f}, "
        f"TPR={m['TPR']:.2f}, SHD={int(m['SHD'])}"
    )

    if hasattr(res, "c_node_names") and res.c_node_names:
        d_full = res.cg_tig.shape[0]
        G_display = res.cg_tig[:d_full, :d_full, :]
        gt_display = np.zeros_like(G_display)
        gt_display[:d_orig, :d_orig, :] = gt
        display_names = var_names + list(res.c_node_names)
    else:
        G_display = G_hat
        gt_display = gt
        display_names = var_names[:d_orig]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    compare_graphs(
        gt_display,
        G_display,
        var_names=display_names,
        fig_ax=(fig, ax),
        title=title,
        node_size=0.22,
        arrow_linewidth=2.5,
        node_label_size=9,
        link_label_fontsize=7,
        show_legend=False,
    )
    fig.tight_layout()
    _save(fig, f"compare_{name}.pdf")


def fig_compare_confounded(data, seed=42):
    """Special compare for confounded trends using res.plot() with C-node highlighting."""
    df = data["df"]
    gt = data["ground_truth"]
    max_lag = data.get("max_lag", 2)
    var_names = data.get("var_names", [str(i) for i in range(df.shape[1])])

    dummy = np.zeros((2, 2))
    ci = ParCorrGPU(dummy, device="cpu")

    res = run_cedar(
        df,
        ci,
        max_lag,
        alpha_cond1=ALPHA,
        alpha_cond2=ALPHA,
        filter_children=False,
        prune=True,
        multi_lag=True,
        multi_lag_keep="first",
        include_autoreg=True,
        lag_method="partial_dcor",
        include_C=True,
        c_preset="linear",
    )

    d_orig = gt.shape[0]
    G_hat = res.cg_tig[:d_orig, :d_orig, :]
    m = evaluate_graph(G_hat, gt, exclude_self_loops=True)
    title = (
        f"F1={m['F1']:.2f}, Prec={m['Precision']:.2f}, "
        f"TPR={m['TPR']:.2f}, SHD={int(m['SHD'])}"
    )

    c_name = (
        res.c_node_names[0]
        if hasattr(res, "c_node_names") and res.c_node_names
        else None
    )

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
    res.plot(
        fig_ax=(fig, ax),
        val_matrix=None,
        node_size=0.22,
        arrow_linewidth=2.5,
        node_label_size=9,
        link_label_fontsize=7,
        show_autodependency_lags=True,
        show_colorbar=False,
        target_node=c_name,
        target_node_color="orange",
        target_out_color="orange",
        target_in_color="orange",
    )
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    _save(fig, "compare_confounded_trends.pdf")


# =============================================================================
# Helpers
# =============================================================================


def _save(fig, filename):
    fig.savefig(FIG_DIR / filename)
    # Also copy to paper figures
    fig.savefig(PAPER_FIG_DIR / filename)
    print(f"  Saved {filename}")
    plt.close()


def _x_names(n):
    return [f"$X_{{{i}}}$" for i in range(n)]


# =============================================================================
# Main
# =============================================================================


def gen_scaling():
    print("\n--- Scaling figures ---")
    fig_scaling("F1", [100, 200, 500], "F1 score", "f1_vs_d.pdf", ylim=(0.4, 1.0))
    fig_scaling("SHD", [100, 200, 500], "SHD (lower is better)", "shd_vs_d.pdf")


def gen_ground_truth():
    print("\n--- Ground truth graphs ---")
    data = load_dataset("ex3", seed=42, T=300)
    fig_ground_truth("cascade_11node", data, CASCADE_NAMES, figsize=(5.5, 5))

    data = load_dataset("ex2", seed=42, T=300)
    fig_ground_truth(
        "linear_6node", data, _x_names(6), figsize=(5, 5), node_layout="sfdp"
    )

    data = load_dataset("ex1", seed=42, T=300)
    fig_ground_truth(
        "nonlinear_5node", data, _x_names(5), figsize=(5, 5), node_layout="sfdp"
    )

    # Confounded trends: augment GT with a τ node showing the unobserved trend
    data = confounded_trends(seed=42, T=500)
    gt = data["ground_truth"]  # (7, 7, max_lag+1)
    d, _, lags = gt.shape
    # Extend to d+1 variables, add τ as the last node
    gt_aug = np.zeros((d + 1, d + 1, lags))
    gt_aug[:d, :d, :] = gt
    # τ → X0, X1, X2, X3, X4 at lag 0 (trend-driven variables)
    for i in [0, 1, 2, 3, 4]:
        gt_aug[d, i, 0] = 1
    names = data.get("var_names", _x_names(7)) + ["τ"]
    fig_ground_truth(
        "confounded_trends",
        {"ground_truth": gt_aug},
        names,
        figsize=(5, 5),
        node_layout="sfdp",
        target_node="τ",
        target_node_color="orange",
        target_out_color="orange",
        target_in_color="orange",
    )


def gen_compare():
    print("\n--- Comparison graphs ---")
    data = load_dataset("ex3", seed=42, T=300)
    fig_compare("cascade_11node", data, var_names=CASCADE_NAMES)

    data = load_dataset("ex2", seed=42, T=300)
    fig_compare("linear_6node", data, var_names=_x_names(6))

    data = load_dataset("ex1", seed=42, T=300)
    fig_compare("nonlinear_5node", data, ci_name="splitkci", var_names=_x_names(5))

    # Confounded trends: use res.plot() with special_nodes for C nodes
    data = confounded_trends(seed=42, T=500)
    fig_compare_confounded(data)


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target in ("all", "scaling"):
        gen_scaling()
    if target in ("all", "gt"):
        gen_ground_truth()
    if target in ("all", "compare"):
        gen_compare()

    print("\nDone.")


if __name__ == "__main__":
    main()
