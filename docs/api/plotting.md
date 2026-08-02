# Plotting API

Causal-TS provides network, time-series, comparison, and diagnostic plots — all built on Matplotlib.

```python
from causalts.plotting import (
    plot_graph,
    plot_time_series_graph,
    compare_graphs,
    plot_metrics_summary,
    plot_lag_metrics,
    plot_pvalue_distribution,
    plot_lagfuncs,
    setup_matrix,
    corrplot,
    compute_association_matrix,
)
```

---

## Graph Plots

### `plot_graph`

Network layout — nodes are variables, edges are contemporaneous or lagged causal links.

```{eval-rst}
.. autofunction:: causalts.plotting._core.plot_graph
```

### `plot_time_series_graph`

Time-unrolled DAG — variables are columns, time is the horizontal axis, lagged edges are arrows.

```{eval-rst}
.. autofunction:: causalts.plotting._core.plot_time_series_graph
```

### `plot_lagfuncs`

Lag-function matrix — auto- and cross-correlation as a function of lag.

```{eval-rst}
.. autofunction:: causalts.plotting._core.plot_lagfuncs
```

### `setup_matrix`

Configurable subplot matrix for custom multi-panel displays.

```{eval-rst}
.. autoclass:: causalts.plotting._core.setup_matrix
   :members:
   :show-inheritance:
```

---

## Comparison & Evaluation Plots

### `compare_graphs`

Overlay TP (green) / FP (red) / FN (orange) edges between two graphs.

```{eval-rst}
.. autofunction:: causalts.plotting.compare.compare_graphs
```

### `plot_metrics_summary`

Bar chart of Precision, Recall, F1, and SHD for one or more results.

```{eval-rst}
.. autofunction:: causalts.plotting.compare.plot_metrics_summary
```

### `plot_lag_metrics`

Heatmap of per-lag Precision, Recall, and F1 — useful for spotting which lags are hardest.

```{eval-rst}
.. autofunction:: causalts.plotting.compare.plot_lag_metrics
```

### `plot_pvalue_distribution`

Histogram of CI test p-values split by true edges vs. non-edges.

```{eval-rst}
.. autofunction:: causalts.plotting.compare.plot_pvalue_distribution
```

---

## Association Matrix (Corrplot)

### `corrplot`

R-corrplot-style association matrix — 7 glyph methods (circle, square, ellipse,
number, color, shade, pie), significance overlays, confidence intervals, and
hierarchical variable ordering.

```{eval-rst}
.. autofunction:: causalts.plotting.corrplot.corrplot
```

### `compute_association_matrix`

Pairwise association matrix from raw data (Pearson, Spearman, Kendall, dcor,
or a custom callable) — the usual input to `corrplot`.

```{eval-rst}
.. autofunction:: causalts.plotting.corrplot.compute_association_matrix
```

---

## Quick Examples

**Network plot from CDNOTS output:**
```python
from causalts.plotting import plot_graph

fig, ax = plot_graph(
    graph=graph,                      # shape (d, d, max_lag+1)
    val_matrix=pvals,                 # optional: p-values → edge width
    var_names=list(df.columns) + ["C"],
)
fig.savefig("causal_graph.pdf", bbox_inches="tight")
```

**TP/FP/FN comparison:**
```python
from causalts.plotting import compare_graphs, plot_metrics_summary

fig, ax = compare_graphs(ground_truth, discovered, var_names=list(df.columns))

metrics = evaluate_graph(discovered, ground_truth)
fig2, ax2 = plot_metrics_summary({"CDNOTS": metrics, "Baseline": baseline_metrics})
```

**P-value diagnostic:**
```python
from causalts.plotting import plot_pvalue_distribution

fig, ax = plot_pvalue_distribution(
    pval_matrix=pvals,
    ground_truth=ground_truth,
    alpha=0.05,
)
```
