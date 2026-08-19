# Getting Started

---

## Installation

::::{tab-set}

:::{tab-item} Standard
```bash
git clone https://github.com/bloomberg/causal-ts.git
cd causal-ts
pip install -e .
```
:::

:::{tab-item} With Effect Estimation
```bash
git clone https://github.com/bloomberg/causal-ts.git
cd causal-ts
pip install -e ".[dowhy]"
```
:::

::::

:::{note}
PyTorch is installed automatically. **CUDA** and **Apple MPS** are auto-detected at runtime — CPU is the fallback.
:::

---

## Quick Start

Follow these four steps to run your first causal discovery:

::::{grid} 2
:gutter: 3

:::{grid-item-card} {fas}`database;1em;sd-text-primary` &nbsp; Step 1 — Load Data
:shadow: sm

```python
from causalts.synthetic_data.synthetic_datasets import load_dataset

data = load_dataset("ex2", seed=42, T=500)
df, ground_truth = data["df"], data["ground_truth"]
```
:::

:::{grid-item-card} {fas}`vial;1em;sd-text-primary` &nbsp; Step 2 — Choose a CI Test
:shadow: sm

```python
from causalts.ci_tests import ParCorrGPU

ci_test = ParCorrGPU(df.values, device="cpu")
```
:::

:::{grid-item-card} {fas}`circle-nodes;1em;sd-text-primary` &nbsp; Step 3 — Run Discovery
:shadow: sm

```python
from causalts import run_cdnots

res = run_cdnots(
    df=df, indep_test=ci_test,
    num_lags=data["max_lag"],
    include_C=True, alpha=0.05, stable=True,
)
```
:::

:::{grid-item-card} {fas}`chart-bar;1em;sd-text-primary` &nbsp; Step 4 — Evaluate & Plot
:shadow: sm

```python
from causalts.utils import evaluate_graph
from causalts.plotting import compare_graphs

d = ground_truth.shape[0]
metrics = evaluate_graph(
    res.cg_tig[:d, :d, :], ground_truth,
)
print(f"F1={metrics['F1']:.3f}, SHD={metrics['SHD']}")
res.plot()
compare_graphs(ground_truth, res.cg_tig[:d, :d, :],
               var_names=list(df.columns))
```
:::

::::

---

## Choosing a CI Test

::::{tab-set}

:::{tab-item} {fas}`bolt` Linear data
Use `ParCorrGPU` — analytic p-value, near-instant, works at any sample size.

```python
from causalts.ci_tests import ParCorrGPU
ci_test = ParCorrGPU(df.values, device="cpu")
```
:::

:::{tab-item} {fas}`wave-square` Nonlinear data
Use `SplitKCIGPU` — best speed/F1 tradeoff on nonlinear data.

```python
from causalts.ci_tests import SplitKCIGPU
ci_test = SplitKCIGPU(df.values, device="cpu")
```
:::

:::{tab-item} {fas}`question` Not sure?
Run the Ramsey RESET test first to detect nonlinearity.

```python
from causalts.utils.linearity import check_linearity
report = check_linearity(df)
```

If no significant nonlinearity is detected, `parcorr-gpu` is optimal.
:::

::::

See the full [CI Test Selection Guide](../ci_tests.md) for benchmarks across all 8 tests.

---

## CLI Usage

```bash
# Generate synthetic data
causal-ts generate --dataset ex1 -T 500

# Run discovery
causal-ts discover data.csv --algorithm cdnots --ci-test parcorr-gpu --max-lag 3

# Evaluate results
causal-ts evaluate ground_truth.npy estimated_graph.npy

# Get CI test recommendations
causal-ts ci-test-info
```

---

## Next Steps

::::{grid} 3
:gutter: 2

:::{grid-item-card} {fas}`circle-nodes` Algorithms
:link: ../algorithms
:link-type: doc
:shadow: sm
:text-align: center

Understand CDNOTS, CDNOTS+, CEDAR, and GRACE in depth — when to use each and how they work.
:::

:::{grid-item-card} {fas}`vials` CI Test Guide
:link: ../ci_tests
:link-type: doc
:shadow: sm
:text-align: center

Strengths, limitations, and tradeoff analysis across all GPU-accelerated CI tests, linear and nonlinear.
:::

:::{grid-item-card} {fas}`book-open` Examples
:link: ../examples/index
:link-type: doc
:shadow: sm
:text-align: center

Hands-on notebooks from basic discovery to counterfactual analysis.
:::

::::
