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
    corr_table,
    pairs_panel,
    plot_lag_dependence,
    compute_lag_dependence,
    extract_subgraph,
    build_pvalue_matrix,
)
```

## Graph Plots

### `plot_graph`

Network layout — nodes are variables, edges are contemporaneous or lagged causal links.

Graphviz-family layouts (`graphviz`, `neato`, `dot`, `fdp`, `sfdp`, `twopi`, and
`circo`) use the optional `pygraphviz` backend first and then the optional
`pydot` backend. Both backends require a Graphviz executable (such as `neato`)
to be installed and available on `PATH`; neither Python package is installed by
`causalts`. If both backends are unavailable or fail, `causalts` falls back to
a circular layout and emits a `UserWarning`.

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

## Comparison & Evaluation Plots

### `compare_graphs`

Overlay TP (green) / FP (orange) / FN (red, dashed) edges between two graphs.

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

### `corr_table`

APA-style correlation table (inspired by R's apaTables::apa.cor.table) —
lower-triangle r values with significance stars, Mean/SD columns, and a
footnote, all individually toggleable. Prints nicely in notebooks
(`_repr_html_`) and terminals (`__repr__`).

```{eval-rst}
.. autofunction:: causalts.plotting.corrplot.corr_table
```

## Pairs Panel

### `pairs_panel`

Scatterplot matrix (inspired by R's psych::pairs.panels) — histogram
diagonal, scatter + linear fit below the diagonal, and correlation text
above.

```{eval-rst}
.. autofunction:: causalts.plotting.pairs.pairs_panel
```

## Lag Dependence

### `plot_lag_dependence`

Grid of lag functions from raw data: cell (i, j) shows the dependence of
`X_i(t-k)` on `X_j(t)` at each lag `k` (dcor, partial dcor, Pearson, Lasso or a
custom callable), with optional significance markers. Useful for choosing
`max_lag` before discovery; the same scores Cedar uses for lag selection.
`adjust="parents"` removes each variable's selected parents first (MCI-style),
which removes most indirect and common-cause dependence when the parents are
found. Its scores are an exploratory ranking: they skip the source's own lags
among its shifted parents, which helps persistent sources but can lift a
persistent non-cause correlated with a missed parent. Its markers test each
source-target pair over all lags; only in a significant pair is the peak lag
marked, plus any other lag that passes its own full-MCI test. With a known graph,
`true_graph=` shades the true lags and flags missed lags and false markers;
`min_score=` hides markers on negligible scores.

```{eval-rst}
.. autofunction:: causalts.plotting.lagplot.plot_lag_dependence
```

### `compute_lag_dependence`

The `(d, d, max_lag+1)` score array behind `plot_lag_dependence`.

```{eval-rst}
.. autofunction:: causalts.plotting.lagplot.compute_lag_dependence
```

## Quick Examples

**Network plot from CDNOTS output:**
```python
from causalts.plotting import plot_graph

fig, ax = plot_graph(
    graph=graph,                      # shape (d, d, max_lag+1)
    val_matrix=val_matrix,            # optional: signed strengths in [-1, 1] → edge colour
    var_names=list(df.columns) + ["C"],
)
fig.savefig("causal_graph.pdf", bbox_inches="tight")
```

**TP/FP/FN comparison:**
```python
from causalts.plotting import compare_graphs, plot_metrics_summary

fig, ax = compare_graphs(ground_truth, discovered, var_names=list(df.columns))

fig2, ax2 = plot_metrics_summary(ground_truth, discovered)   # F1, precision, recall, …
```

**P-value diagnostic:**
```python
from causalts.plotting import plot_pvalue_distribution

fig, ax = plot_pvalue_distribution(
    pvalue_matrix=res.pvalue_matrix,   # run_cdnots(..., return_pvals=True)
    graph_true=ground_truth,
    alpha_line=0.05,
)
```

## Graph Utilities

```{eval-rst}
.. autofunction:: causalts.plotting.extract_subgraph

.. autofunction:: causalts.plotting.build_pvalue_matrix
```
