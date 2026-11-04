# Getting Started

## Installation

Causal-TS needs Python 3.10 or newer.

::::{tab-set}

:::{tab-item} PyPI
```bash
pip install causalts
# or, with uv
uv pip install causalts
```
:::

:::{tab-item} With extras
```bash
pip install "causalts[dowhy]"      # effect estimation (DoWhy)
pip install "causalts[tigramite]"  # PCMCI / PCMCI+ / LPCMCI wrappers
pip install "causalts[parquet]"    # read Parquet files
```
:::

:::{tab-item} From source
```bash
git clone https://github.com/bloomberg/causal-ts.git
cd causal-ts
pip install -e ".[dev,dowhy,tigramite]"
```
:::

::::

:::{note}
PyTorch is installed automatically. **CUDA** and **Apple MPS** are auto-detected at runtime — CPU is the fallback.
The [stable](https://causal-ts.readthedocs.io/en/stable/) docs match the latest PyPI release; the
[latest](https://causal-ts.readthedocs.io/en/latest/) docs follow the `main` branch and can describe features not yet released.
:::

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

## Choosing a Configuration

Let the library measure the data and recommend a configuration:

```python
from causalts import inspect_df, discover_df

report = inspect_df(df)
rec = report["recommendation"]
print(rec["rationale"])

res = discover_df(df, algorithm=rec["algorithm"], ci_test=rec["ci_test"],
                  max_lag=rec["max_lag"], include_C=rec["include_C"],
                  c_preset=rec["c_preset"])
# Optional refinement of that CDNOTS graph's lagged edges (REAP or GRACE;
# rec["refine"] is None when max_lag is 0):
if rec["refine"]:
    refined = discover_df(df, algorithm=rec["refine"], ci_test=rec["ci_test"],
                          max_lag=rec["max_lag"], include_C=rec["include_C"],
                          c_preset=rec["c_preset"], **rec["refine_params"])
```

The same report is available from the terminal with `causal-ts inspect data.csv`. The rule it applies:

| Data | CI test | Refinement |
|---|---|---|
| Discrete or mixed discrete-continuous columns | `cmiknn-mixed-gpu` | REAP |
| Some lagged variable changes another's noise scale only (`facts["variance_dependence"]`) | `rcot` (full conditional independence) | GRACE (keeps variance-only parents) |
| Nonlinear mean effects | `raft` (fast, conditional mean only) | REAP |
| Approximately linear | `parcorr-gpu` | REAP |

Both refinements refine **lagged** edges: REAP re-adds lag-0 edges with `contemporaneous=True` (in
`rec["refine_params"]`); with GRACE, keep the CDNOTS graph's lag-0 edges. Discovery itself is CDNOTS at any dimension: its cost in the number of variables is set by the CI test,
not by the algorithm. Nonstationary series get the C node with a preset matched to the detected trend
form. See the [CI Test Selection Guide](../ci_tests.md) for every test and the
[Algorithms](../algorithms.md) page for CDNOTS+, CEDAR, GRACE and REAP.

## CLI Usage

```bash
# Generate synthetic data -> out/generate_<timestamp>/{data.csv, ground_truth.npy}
causal-ts -o out generate --dataset ex1 -T 500

# Measure the data and get a recommended configuration (JSON)
causal-ts inspect out/generate_*/data.csv

# Run discovery -> out/discover_cdnots_<timestamp>/estimated_graph.npy
causal-ts -o out discover out/generate_*/data.csv --algorithm cdnots --ci-test parcorr-gpu --max-lag 3

# Evaluate against the ground truth
causal-ts -o out evaluate out/generate_*/ground_truth.npy out/discover_cdnots_*/estimated_graph.npy

# CI test selection guide
causal-ts ci-test-info
```

`-o/--output-dir` is a global option and goes before the subcommand. Each command writes a new
timestamped folder under it (default `./causal_ts_output`). See the [CLI reference](../cli.md).

## Next Steps

::::{grid} 3
:gutter: 2

:::{grid-item-card} {fas}`circle-nodes` Algorithms
:link: ../algorithms
:link-type: doc
:shadow: sm
:text-align: center

CDNOTS, CDNOTS+, CEDAR, and the GRACE / REAP refinements — how they work and when to use each.
:::

:::{grid-item-card} {fas}`vials` CI Test Guide
:link: ../ci_tests
:link-type: doc
:shadow: sm
:text-align: center

Strengths, limitations, and tradeoffs of every CI test, linear and nonlinear.
:::

:::{grid-item-card} {fas}`book-open` Examples
:link: ../examples/index
:link-type: doc
:shadow: sm
:text-align: center

Hands-on notebooks from basic discovery to counterfactual analysis.
:::

::::
