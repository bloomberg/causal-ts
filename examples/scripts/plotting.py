# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: .venv-nbrun
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Visualization Guide
#
# Comprehensive guide to causal-ts visualization: graph plots, time-series graphs, comparison plots, p-value diagnostics, lag metrics, custom edge colors, target node highlighting, subgraph extraction, and CEDAR-specific plots.

# %%
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

import causalts.plotting as tp
from causalts.plotting import extract_subgraph
from causalts.synthetic_data.synthetic_datasets import ex1_nl, ex3

# %% [markdown]
# ## 1. Setup: Generate Data & Run Discovery
#
# We use `ex1_nl` (5-variable nonlinear) and `ex3` (11-variable cascade) from the built-in datasets, then run CDNOTS to get a discovered graph with p-values.

# %%
data1 = ex1_nl(T=500, seed=42)
df1, gt1, names1 = data1["df"], data1["ground_truth"], list(data1["df"].columns)
print(f"ex1_nl: {data1['name']}, d={len(names1)}, T={len(df1)}, max_lag={data1['max_lag']}")

data3 = ex3(T=500, seed=42)
df3, gt3, names3 = data3["df"], data3["ground_truth"], list(data3["df"].columns)
print(f"ex3: {data3['name']}, d={len(names3)}, T={len(df3)}, max_lag={data3['max_lag']}")

# %%
from causalts import run_cdnots_plus
from causalts.ci_tests import RCOTGPU

# ex1_nl has nonlinear dependencies (x^2 terms) — use RCOTGPU which
# handles nonlinearity well and is fast (Random Fourier Features).
ci_test = RCOTGPU(df1.values, num_f=100, num_f2=10)
result = run_cdnots_plus(
    df1, ci_test, num_lags=data1["max_lag"],
    alpha=0.05, include_C=False, verbose=0,
    return_pvals=True,
)
discovered = result.cg_tig
print(f"Discovered graph shape: {discovered.shape}")
print(f"P-value matrix shape: {result.pvalue_matrix.shape}")

# %% [markdown]
# ## 2. Basic `plot_graph`
#
# The core network visualization. Variables are nodes; straight edges are contemporaneous, curved edges are lagged.

# %%
tp.plot_graph(gt1, var_names=names1, figsize=(4, 4), title="Ground Truth");

# %%
# Built-in Networkx layout algorithms
layouts = ["circular", "spring", "kamada_kawai", "shell", "spectral"]
fig, axes = plt.subplots(1, 5, figsize=(12, 4))
for ax, layout in zip(axes.flat, layouts):
    tp.plot_graph(
        gt1, var_names=names1, fig_ax=(fig, ax),
        node_layout=layout, node_size=0.2,
        arrow_linewidth=3, show_colorbar=False,
    )
    ax.set_title(layout)
plt.tight_layout()
plt.show()

# %%
# Graphviz layouts (requires: pip install pygraphviz; brew install graphviz)
# These produce higher-quality layouts but need system-level graphviz.
# Falls back to 'circular' if pygraphviz is not installed.
import shutil

graphviz_layouts = ["neato", "fdp", "sfdp"]

if shutil.which("neato"):
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, layout in zip(axes, graphviz_layouts):
        tp.plot_graph(
            gt1, var_names=names1, fig_ax=(fig, ax),
            node_layout=layout, node_size=0.2,
            arrow_linewidth=3, show_colorbar=False,
        )
        ax.set_title(layout)
    plt.tight_layout()
    plt.show()
else:
    print("Graphviz not found — skipping neato/fdp/sfdp layouts.")
    print("Install with: brew install graphviz && pip install pygraphviz")

# %% [markdown]
# ### Position reuse with `return_pos`
#
# Compute positions once and reuse across plots for consistent layout.

# %%
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
_, _, pos = tp.plot_graph(
    gt1, var_names=names1, node_layout="neato", fig_ax=(fig, axes[0]),
    show_colorbar=False, return_pos=True, title="Ground truth (spring)"
)

tp.plot_graph(
    discovered, var_names=names1, fig_ax=(fig, axes[1]),
    node_pos=pos, show_colorbar=False, title="Discovered (spring)"
);


# %% [markdown]
# ### Convenience method: `result.plot()`
#
# The `CdnotsResult` object has a bound `.plot()` method that wraps `plot_graph` and auto-handles C node labels.

# %%
result.plot(
    var_names=names1, node_layout="neato",
    node_size=0.2, show_colorbar=False,
    arrow_linewidth=3, link_label_fontsize=8,
    title="result.plot() convenience method",
);

# %% [markdown]
# ## 3. `plot_time_series_graph`
#
# Time-unrolled DAG: variables on rows, time on columns. Lagged edges are drawn as curved arrows across time steps.

# %%
tp.plot_time_series_graph(
    gt1, var_names=names1,
    arrow_linewidth=3, node_size=0.06, figsize=(10, 4),
    title="Time-series graph",
    edge_color='steelblue', node_label_size=8,
    order=[1,2,0,4,3] # custom node order (optional)
)
plt.show()

# %%
# order='auto' uses barycenter heuristic to reduce edge crossings
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
tp.plot_time_series_graph(
    gt1, var_names=names1,
    arrow_linewidth=3, node_size=0.06,
    fig_ax=(fig, axes[0]),
)
axes[0].set_title("Original order")
tp.plot_time_series_graph(
    gt1, var_names=names1,
    arrow_linewidth=3, node_size=0.06,
    order="auto", fig_ax=(fig, axes[1]),
)
axes[1].set_title("order='auto' (barycenter)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Custom Edge Color
#
# Use `edge_color` to change the default edge color (previously hardcoded to black). Works when `val_matrix` is not provided.

# %%
fig, axes = plt.subplots(1, 3, figsize=(10, 4))

colors = ["black", "orange", "#8B0000"]
labels = ["Default (black)", "orange", "Dark red (#8B0000)"]

for ax, color, label in zip(axes, colors, labels):
    tp.plot_graph(
        gt1, var_names=names1, fig_ax=(fig, ax),
        edge_color=color, show_colorbar=False,
    )
    ax.set_title(label)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Target Node Highlighting
#
# Highlight one or more variables to distinguish **incoming** (causes) and **outgoing** (effects) edges. Useful for understanding causal neighborhoods in large graphs.
#
# Parameters:
# - `target_node` — variable name, index, **or list** of names/indices for multi-target
# - `target_in_color` — edges INTO target(s) (default: blue `#2196F3`). Pass a list for per-target colors.
# - `target_out_color` — edges OUT OF target(s) (default: orange `#FF9800`). Pass a list for per-target colors.
# - `target_node_color` — fill color for target node(s) (default: green `#4CAF50`). Pass a list for per-target colors.
# - `target_between_color` — edges connecting two target nodes (default: purple `#9C27B0`)
# - `edge_color` — non-target edges (default: black)

# %%
tp.plot_graph(
    gt1, var_names=names1,
    target_node=names1[3],
    target_in_color="#2196F3",
    target_out_color="#FF9800",
    target_node_color="#4CAF50",
    edge_color="#AAAAAA",
    figsize=(4, 4),
    title=f"Target: {names1[3]} — Blue=causes, Orange=effects",
);


# %%
# Compare different target nodes
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
for ax, target in zip(axes, names1[:3]):
    tp.plot_graph(
        gt1, var_names=names1, fig_ax=(fig, ax),
        target_node=target, edge_color="#CCCCCC",
    )
    ax.set_title(f"Target: {target}")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Multi-target highlighting
#
# Pass a list to `target_node` to highlight multiple variables simultaneously. Edges between targets use `target_between_color`.

# %%
# Target highlighting overrides val_matrix colormap
rng = np.random.default_rng(0)
val = rng.standard_normal(gt1.shape) * gt1

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

tp.plot_graph(gt1, val_matrix=val, var_names=names1, fig_ax=(fig, axes[0]))
axes[0].set_title("With val_matrix (colormap)")

tp.plot_graph(
    gt1, val_matrix=val, var_names=names1,
    fig_ax=(fig, axes[1]),
    target_node=names1[2], edge_color="#999999",
)
axes[1].set_title(f"target_node={names1[2]} overrides colormap")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Subgraph Extraction
#
# `extract_subgraph()` extracts the causal neighborhood around a target variable at configurable depth:
# - `depth=1` — direct parents and children
# - `depth=2` — parents/children of parents/children
# - `direction="parents"` / `"children"` — restrict to ancestors or descendants

# %%
# Full 11-variable cascade network
tp.plot_graph(gt3, var_names=names3, figsize=(6, 6), show_colorbar=False)
plt.title(f"ex3: Full {len(names3)}-variable graph")
plt.show()

# %%
# We pick a variable with both parents and children for best demonstration
target_var = "Mek"

sub1, idx1, sub_names1 = extract_subgraph(gt3, target=target_var, depth=1, var_names=names3)
sub2, idx2, sub_names2 = extract_subgraph(gt3, target=target_var, depth=2, var_names=names3)
print(f"Target: {target_var}")
print(f"Depth-1: {sub_names1} ({len(sub_names1)} vars)")
print(f"Depth-2: {sub_names2} ({len(sub_names2)} vars)")

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
tp.plot_graph(
    sub1, var_names=sub_names1, fig_ax=(fig, axes[0]),
    target_node=target_var, edge_color="#AAAAAA",
)
axes[0].set_title(f"Depth 1 ({len(sub_names1)} vars)")
tp.plot_graph(
    sub2, var_names=sub_names2, fig_ax=(fig, axes[1]),
    target_node=target_var, edge_color="#AAAAAA",
)
axes[1].set_title(f"Depth 2 ({len(sub_names2)} vars)")
plt.tight_layout()
plt.show()

# %%
# Direction filtering: ancestors vs descendants
sub_p, _, names_p = extract_subgraph(
    gt3, target=target_var, depth=2, var_names=names3, direction="parents",
)
sub_c, _, names_c = extract_subgraph(
    gt3, target=target_var, depth=2, var_names=names3, direction="children",
)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
tp.plot_graph(
    sub_p, var_names=names_p, fig_ax=(fig, axes[0]),
    target_node=target_var, edge_color="#AAAAAA",
)
axes[0].set_title(f"Ancestors (depth=2): {names_p}")
tp.plot_graph(
    sub_c, var_names=names_c, fig_ax=(fig, axes[1]),
    target_node=target_var, edge_color="#AAAAAA",
)
axes[1].set_title(f"Descendants (depth=2): {names_c}")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. `compare_graphs` — TP / FP / FN Overlay
#
# Overlays discovered vs ground truth with color-coded edges: TP (green), FP (orange), FN (red dashed).

# %%
tp.compare_graphs(
    gt1, discovered, var_names=names1,
    figsize=(5, 5), show_legend=True, legend_loc="lower right", #set to None for auto placement of legend box
    arrow_linewidth=4, link_label_fontsize=10,
)
plt.show()

# %%
# Custom comparison colors
tp.compare_graphs(
    gt1, discovered, var_names=names1,
    figsize=(5, 5),
    tp_color="#1B5E20", fp_color="#E65100", fn_color="#B71C1C",
    fn_style="dotted",
    show_legend=True, legend_loc="lower right",
)
plt.title("Custom comparison colors")
plt.show()

# %% [markdown]
# ## 8. `plot_metrics_summary` — Evaluation Bar Chart
#
# Horizontal bar chart of Precision, Recall, F1, and SHD. Supports `level='lag'` (default) and `level='pair'`.

# %%
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
tp.plot_metrics_summary(
    gt1, discovered, title="Per-Lag", fig_ax=(fig, axes[0]),
)
tp.plot_metrics_summary(
    gt1, discovered, level="pair", title="Pair-Level", fig_ax=(fig, axes[1]),
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. `plot_lag_metrics` — Per-Lag Heatmap
#
# Heatmap of precision / recall / F1 broken down by lag.

# %%
tp.plot_lag_metrics(
    gt1, discovered,
    title="Per-Lag Breakdown", figsize=(4,3),
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. `plot_pvalue_distribution` — P-Value Diagnostics
#
# Histogram of p-values split by edge type. Two modes:
# - **All tested pairs**: true edges (green) vs non-edges (grey)
# - **Discovered edges only**: TP (green) vs FP (grey) — pass `graph_discovered`

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# log_scale=True spreads out the near-zero p-values so the separation
# between true edges and non-edges is visible
tp.plot_pvalue_distribution(
    result.pvalue_matrix, gt1, alpha_line=0.05,
    log_scale=True,
    title="All tested pairs (log scale)", fig_ax=(fig, axes[0]),
)

tp.plot_pvalue_distribution(
    result.pvalue_matrix, gt1, graph_discovered=discovered,
    alpha_line=0.05, log_scale=True,
    title="Discovered edges — TP vs FP (log scale)",
    fig_ax=(fig, axes[1]),
)
plt.tight_layout()
plt.show()

# %%
# Convenience method on the result object (also supports log_scale)
result.plot_pvalues(gt1, alpha_line=0.05, log_scale=True, title="result.plot_pvalues()")
plt.show()

# %% [markdown]
# ## 11. Combining Features: Discovery + Extract + Highlight + Compare
#
# Full pipeline: extract subgraph around a variable of interest, then compare discovered vs ground truth with target highlighting.

# %%
focus_var = names1[1]
sub_disc, idx_disc, names_disc = extract_subgraph(
    discovered, target=focus_var, depth=1, var_names=names1,
)
sub_gt = gt1[np.ix_(idx_disc, idx_disc, range(gt1.shape[2]))]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

tp.plot_graph(
    sub_gt, var_names=names_disc, fig_ax=(fig, axes[0]),
    target_node=focus_var, edge_color="#CCCCCC",
)
axes[0].set_title(f"Ground truth around {focus_var}")

tp.plot_graph(
    sub_disc, var_names=names_disc, fig_ax=(fig, axes[1]),
    target_node=focus_var, edge_color="#CCCCCC",
)
axes[1].set_title(f"Discovered around {focus_var}")

tp.compare_graphs(
    sub_gt, sub_disc, var_names=names_disc,
    fig_ax=(fig, axes[2]), show_legend=True,
    legend_loc="lower left",
    arrow_linewidth=3,
)
axes[2].set_title(f"Comparison around {focus_var}")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 12. CEDAR Plots
#
# `CedarResult` (returned by `run_cedar`) has built-in visualization methods:
# - `.plot()` — causal graph (wraps `plot_graph`)
# - `.plot_lag_importance()` — heatmap of lag-wise dcor/pearson scores per variable pair

# %%
from causalts.cedar import run_cedar

res_cedar = run_cedar(
    df=df1,
    ci_test=ci_test,
    max_lag=data1["max_lag"],
    lag_method="dcor",
    alpha_cond1=0.01,
    alpha_cond2=0.01,
    normalize="minmax",
    verbose=False,
)
res_cedar.plot_lag_importance(
    x_base=1, show_gridlines=True,
    maximum=1, minimum=0,
    figsize=(8, 5),
)
plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
pos = tp.plot_graph(
    gt1, var_names=names1,
    arrow_linewidth=3, show_colorbar=False,
    fig_ax=(fig, axes[0]),
    return_pos=True,
)
axes[0].set_title("Ground Truth")
res_cedar.plot(
    node_pos=pos[2],
    fig_ax=(fig, axes[1]),
    node_size=0.25, arrow_linewidth=3,
)
axes[1].set_title("CEDAR Discovered")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 13. Customization Reference
#
# Fine-tune the visual appearance.

# %%
# special_nodes for manual node coloring
tp.plot_graph(
    gt1, var_names=names1,
    special_nodes={
        (names1[0], 0): "#E91E63",
        (names1[3], 0): "#9C27B0",
    },
    edge_color="#22EA2C",
    node_size=0.25,
    arrow_linewidth=4,
    curved_radius=0.3,
    show_colorbar=False,
    figsize=(4, 4),
)
plt.title("special_nodes + edge_color + custom size/curvature")
plt.show()

# %%
# Saving figures
fig, ax = tp.plot_graph(
    gt1, var_names=names1,
    edge_color="steelblue",
    target_node=names1[2],
    figsize=(4, 4),
)
# fig.savefig("my_graph.png", dpi=300, bbox_inches="tight")
# fig.savefig("my_graph.pdf", bbox_inches="tight")
print("Uncomment savefig lines above to export")
plt.show()

# %%

# %% [markdown]
# ## `corrplot` — Correlation & Association Matrix
#
# Matrix visualization inspired by R's `corrplot` package. Supports 7 glyph methods, arbitrary pairwise metrics (Pearson, Spearman, dcor, custom callables), significance overlays, confidence intervals, hierarchical ordering, and upper/lower triangle splits.
#
# Two public functions:
# - **`corrplot(data, ...)`** — the main visualization, returns `(fig, ax, matrix)`
# - **`compute_association_matrix(data, metric)`** — computes pairwise associations separately
#

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from causalts.plotting import corrplot, compute_association_matrix

# Self-contained example data: a few correlated variables
rng = np.random.default_rng(0)
n = 300
z = rng.standard_normal(n)
df1 = pd.DataFrame({
    "A": z + 0.2 * rng.standard_normal(n),
    "B": 0.85 * z + 0.5 * rng.standard_normal(n),
    "C": -0.6 * z + 0.7 * rng.standard_normal(n),
    "D": rng.standard_normal(n),
    "E": rng.standard_normal(n),
})
df1["F"] = 0.7 * df1["D"] + 0.5 * rng.standard_normal(n)
corr = df1.corr()
corr.round(2)


# %% [markdown]
# ### All 7 Glyph Methods

# %%
methods = ['circle', 'square', 'ellipse', 'number', 'color', 'shade', 'pie']
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, method in zip(axes.flat, methods):
    corrplot(df1, metric='pearson', method=method, fig_ax=(fig, ax), colorbar=False, tl_cex=0.8)
    ax.set_title(method)
axes.flat[-1].set_visible(False)  # hide unused subplot
fig.suptitle('All 7 Glyph Methods', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Upper / Lower Triangle Split
#
# Different glyph methods per triangle — the signature feature of R's corrplot.

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
corrplot(df1, metric='pearson', upper='ellipse', lower='number', fig_ax=(fig, axes[0]),
         tl_cex=0.8, title='ellipse / number')
corrplot(df1, metric='pearson', upper='circle', lower='shade', fig_ax=(fig, axes[1]),
         tl_cex=0.8, title='circle / shade')
corrplot(df1, metric='pearson', upper='pie', lower='color', fig_ax=(fig, axes[2]),
         tl_cex=0.8, title='pie / color')
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Triangle Type & Hierarchical Ordering

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
corrplot(df1, metric='pearson', type='full', fig_ax=(fig, axes[0]), tl_cex=0.8, title='type=full')
corrplot(df1, metric='pearson', type='upper', fig_ax=(fig, axes[1]), tl_cex=0.8, title='type=upper')
corrplot(df1, metric='pearson', type='lower', fig_ax=(fig, axes[2]), tl_cex=0.8, title='type=lower')
plt.tight_layout()
plt.show()


# %%
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, ord_method in zip(axes, ['original', 'hclust', 'AOE', 'FPC']):
    kw = {'addrect': 2, 'rect_col': 'red'} if ord_method == 'hclust' else {}
    corrplot(corr, order=ord_method, method='ellipse',
             fig_ax=(fig, ax), colorbar=False, tl_cex=0.8, **kw)
    ax.set_title(f'order={ord_method}')
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Significance Overlays
#
# Mark insignificant cells given a p-value matrix.

# %%
from scipy import stats

n_vars = len(df1.columns)
pval_matrix = np.zeros((n_vars, n_vars))
for i in range(n_vars):
    for j in range(n_vars):
        if i != j:
            _, p = stats.pearsonr(df1.iloc[:, i], df1.iloc[:, j])
            pval_matrix[i, j] = p

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
corrplot(df1, metric='pearson', pvalues=pval_matrix,
         insig='pch', pch='X', pch_cex=2,
         fig_ax=(fig, axes[0]), tl_cex=0.8, title="insig='pch' (X marks)")
corrplot(df1, metric='pearson', pvalues=pval_matrix,
         insig='blank', method='number', tl_pos='lb', tl_srt=0,
         fig_ax=(fig, axes[1]), tl_cex=0.8, title="insig='blank', number")
corrplot(df1, metric='pearson', pvalues=pval_matrix,
         insig='p-value', method='color', number_fmt='.2f',
         fig_ax=(fig, axes[2]), tl_cex=0.8, title="insig='p-value'")
corrplot(df1, metric='pearson', pvalues=pval_matrix,
         insig='label_sig', sig_level=[0.001, 0.01, 0.05],
         method='color', pch_cex=2.5,
         fig_ax=(fig, axes[3]), tl_cex=0.8, title="multi-level stars")
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Coefficients & Diagonal Options

# %%
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
corrplot(corr, method='color', add_coef=True, add_coef_col='white',
         number_cex=0.9, fig_ax=(fig, axes[0]), tl_cex=0.8, title='add_coef on color')
corrplot(corr, diag='names', fig_ax=(fig, axes[1]), tl_cex=0.8, title="diag='names'")
corrplot(corr, diag='values', fig_ax=(fig, axes[2]), tl_cex=0.8, title="diag='values'")
corrplot(df1, metric='pearson', diag='hist', fig_ax=(fig, axes[3]), tl_cex=0.8, title="diag='hist'")
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Different Association Metrics

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
corrplot(df1, metric='spearman', fig_ax=(fig, axes[0]), tl_cex=0.8, title='Spearman')
corrplot(df1, metric='dcor', fig_ax=(fig, axes[1]), tl_cex=0.8, title='dcor (sequential)')
corrplot(df1, metric=lambda x, y: np.abs(np.corrcoef(x, y)[0, 1]),
         is_corr=False, fig_ax=(fig, axes[2]), tl_cex=0.8, title='|Pearson| (custom)')
plt.tight_layout()
plt.show()


# %% [markdown]
# ### `compute_association_matrix` — Reusable Metric Computation
#
# Pre-compute once, plot many times with different visual settings.

# %%
dcor_mat = compute_association_matrix(df1, metric='dcor')
print(f'dcor matrix shape: {dcor_mat.shape}')
print(dcor_mat.round(3))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
corrplot(dcor_mat, is_corr=False, method='circle',
         fig_ax=(fig, axes[0]), tl_cex=0.8, title='dcor — circles')
corrplot(dcor_mat, is_corr=False, method='number',
         fig_ax=(fig, axes[1]), tl_cex=0.8, title='dcor — numbers')
plt.tight_layout()
plt.show()


# %% [markdown]
# ## `corr_table` — APA-Style Correlation Table
#
# Inspired by R's `apaTables::apa.cor.table`. Returns a `CorrTableResult` that
# prints nicely in a notebook (`_repr_html_`) or a terminal (`__repr__`):
# lower-triangle r values with significance stars, Mean/SD columns, and a
# footnote — each toggleable (`full_matrix=`, `show_n=`, `sig_stars=`, `show_ci=`).

# %%
from causalts.plotting import corr_table

corr_table(df1)

# %% [markdown]
# ## `pairs_panel` — Scatterplot Matrix
#
# Inspired by R's `psych::pairs.panels`. Diagonal shows a histogram (with a
# Gaussian KDE overlay) per variable, the lower triangle shows scatterplots
# with a linear fit line, and the upper triangle shows the pairwise
# correlation as text, sized/colored by `|r|` (colorblind-safe palette).

# %%
from causalts.plotting import pairs_panel

fig, axes, matrix = pairs_panel(df1)
plt.show()
