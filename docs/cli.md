# CLI Reference

The `causal-ts` command provides a full pipeline from the terminal: generate data, run discovery, evaluate results, and plot graphs.

```bash
causal-ts [OPTIONS] COMMAND [ARGS]...
```

**Global options** (pass before the subcommand):

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output-dir PATH` | `./causal_ts_output` | Directory for all saved files |
| `-s, --seed INT` | random | Global random seed |
| `--device TEXT` | auto | Compute device (`cpu`, `cuda`, `mps`) |
| `-v, --verbose` | off | Verbose logging |
| `-q, --quiet` | off | Suppress non-error output |

---

## `discover`

Run causal discovery on a CSV file.

```bash
causal-ts discover DATA.csv [OPTIONS]
```

### Algorithm options

| Flag | Default | Description |
|------|---------|-------------|
| `-a, --algorithm` | `cdnots` | `cdnots`, `cdnots+`, `cedar`, `grace`, `grace-ss` |
| `--ci-test` | `parcorr-gpu` | CI test — see `ci-test-info` |
| `--max-lag INT` | `3` | Maximum time lag |
| `--alpha FLOAT` | `0.05` | Significance level (`0.01` recommended for better results, required for CDNOTS+) |
| `--lag-list TEXT` | — | Comma-separated explicit lags, e.g. `1,3,12` |
| `--var-names TEXT` | from CSV | Comma-separated variable names |

### CDNOTS-specific

| Flag | Default | Description |
|------|---------|-------------|
| `--include-c / --no-c` | on | Include nonstationarity node C (also applies to CEDAR) |
| `--c-preset` | `linear` | C node basis: `linear`, `step`, `step+linear`, `linear+sin`, `linear+exp`, `linear+quad` (also applies to CEDAR) |
| `--stable` | off | Stable skeleton discovery (slower, more robust) |
| `--max-degree INT` | — | Maximum node degree constraint |

### CEDAR-specific

| Flag | Default | Description |
|------|---------|-------------|
| `--lag-method` | `dcor` | Lag importance metric: `dcor` (default), `dcor_biased`, `pearson`. `--lag-sel-algo` also accepted for legacy. |
| `--lag-pvalue-method` | `t_test` | Lag significance: `t_test` (~200× faster) or `circular_shift` (permutation) |
| `--lag-alpha FLOAT` | `0.05` | P-value threshold for lag significance |
| `--alpha-cond1 FLOAT` | `0.05` | Significance threshold for dependence test (Condition 1). `--p-cond1` also accepted. |
| `--alpha-cond2 FLOAT` | `0.05` | Significance threshold for independence test (Condition 2). `--p-cond2` also accepted. |
| `--normalize` | `minmax` | `minmax`, `zscore`, or `none`. `--normalization` also accepted. |
| `--filter-children / --no-filter-children` | on | Skip candidates whose dcor asymmetry suggests they are effects |
| `--multi-lag / --no-multi-lag` | on | Test all significant lags per pair (vs only the top lag) |
| `--multi-lag-keep` | `first` | `first` (stop at first accepted lag) or `all` (max recall) |
| `--target-var TEXT` | — | Discover causes of one variable only — O(d) tests instead of O(d²) |
| `--no-prune` | off | Skip MCI pruning pass after discovery |
| `--impute` | — | Missing-value strategy: `pairwise_complete` (default) or `var_em` |
| `--include-autoreg / --no-autoreg` | on | Add autoregressive self-loops `Xi(t-k)→Xi(t)` for the detected AR order of each variable |
| `--assume-ar1` | off | Skip AR order estimation; use standard AR(1) Cond2 for all variables (backward-compatible with original SyPI) |
| `--include-c / --no-c` | on | Append a nonstationarity time-index variable C — forces `include_lag0=True` so C(t) → X(t) edges are detectable |
| `--c-preset` | `linear` | C node basis preset: `linear`, `step`, `step+linear`, `linear+sin`, `linear+exp`, `linear+quad` |

### GRACE-specific

| Flag | Default | Description |
|------|---------|-------------|
| `--gate-threshold FLOAT` | `0.5` | Minimum gate value to keep an edge |
| `--stability-threshold FLOAT` | `0.6` | Minimum selection frequency (GRACE-SS) |
| `--max-epochs INT` | `150` | Training epochs |
| `--batch-size INT` | auto | Mini-batch size |
| `--patience INT` | `20` | Early-stopping patience |

### Missing data

| Flag | Default | Description |
|------|---------|-------------|
| `--impute` | — | `pairwise_complete`, `var_em`, or `causal_iterative` |
| `--impute-kwargs JSON` | — | Extra imputer kwargs, e.g. `'{"p": 2, "n_iter": 5}'` |

### Output & evaluation

| Flag | Default | Description |
|------|---------|-------------|
| `--ground-truth PATH` | — | `.npy` ground truth for instant evaluation |
| `--save-plot / --no-plot` | on | Save graph plot images |
| `--plot-format` | `png` | `png`, `pdf`, or `svg` |

**Example — full pipeline in one command:**
```bash
causal-ts -s 42 discover data.csv \
  --algorithm cdnots \
  --ci-test splitkci \
  --max-lag 3 \
  --alpha 0.05 \
  --ground-truth ground_truth.npy \
  --plot-format pdf
```

---

## `generate`

Generate synthetic time series with a known causal structure.

```bash
causal-ts generate [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | — | Built-in dataset: `ex1` (5-Node Nonlinear System), `ex2` (6-Node Linear VAR), `ex3` (Cascade Network 11-Node), `henon` (Hénon Chain) — see Python API for `lorenz96`, `scale_free`, `erdos_renyi`, `small_world`, `ex1_nonstationary`, weather benchmark |
| `-n, --n-vars INT` | `5` | Number of variables (random generation) |
| `--n-links INT` | auto | Number of cross-variable links |
| `--max-lag INT` | `3` | Maximum lag |
| `-T, --samples INT` | `500` | Time series length |
| `--density` | `sparse` | `sparse` or `dense` |
| `--dep-funcs TEXT` | `lin,lin,nl1,nl2` | Comma-separated dependency functions |

**Outputs:** `data.csv`, `ground_truth.npy`, `meta.json`

```bash
# Load a built-in dataset
causal-ts generate --dataset ex1 -T 1000

# Generate a random 10-variable sparse network
causal-ts generate -n 10 --max-lag 2 -T 500 --density sparse
```

---

## `evaluate`

Compare a discovered graph against ground truth and compute metrics.

```bash
causal-ts evaluate GRAPH_TRUE.npy GRAPH_DISCOVERED.npy [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--var-names TEXT` | — | Comma-separated variable names |
| `--exclude-self-loops` | off | Exclude auto-dependencies from metrics |
| `--save-plot / --no-plot` | on | Save TP/FP/FN comparison plot |
| `--plot-format` | `png` | `png`, `pdf`, or `svg` |

**Metrics reported:** TPR, FPR, Precision, Recall, F1, SHD — both edge-level and pair-level.

```bash
causal-ts evaluate ground_truth.npy discovered.npy --var-names X0,X1,X2,X3
```

---

## `ci-test-info`

Print a selection guide for CI tests with indicative performance notes.

```bash
causal-ts ci-test-info [--test TEST_NAME]
```

Run without arguments for a full table; pass `--test splitkci` for details on a single test.

---

## `plot`

Visualise a saved causal graph (`.npy` file produced by `discover`).

```bash
causal-ts plot GRAPH [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--type` | `network` | Plot type: `network` or `time-series` |
| `--val-matrix` | — | Path to `val_matrix.npy` for edge-width/color encoding |
| `--var-names` | — | Comma-separated variable names |
| `--figsize` | — | Figure size as `W,H` (e.g. `10,8`) |
| `--plot-format` | `png` | Output format: `png`, `pdf`, or `svg` |
| `--save-name` | — | Custom output filename (without extension) |

---

## `dowhy` — Effect Estimation

Requires `pip install causalts[dowhy]`. All subcommands take a saved `graph.npy` and a CSV data file.

### `dowhy effect`

Estimate the average treatment effect (ATE) via the backdoor criterion.

```bash
causal-ts dowhy effect GRAPH --data data.csv \
  --treatment X0 --outcome X2 --lag 1 \
  --method backdoor.linear_regression
```

### `dowhy fit-scm`

Fit a Structural Causal Model and optionally run a counterfactual query.

```bash
causal-ts dowhy fit-scm GRAPH --data data.csv \
  --mechanism linear \
  --counterfactual-target X2 \
  --intervention "X0=0.0"
```

### `dowhy root-cause`

Attribute an observed anomaly to root-cause variables using Shapley-based intrinsic causal contribution.

```bash
causal-ts dowhy root-cause GRAPH --data data.csv --target X2
```

### `dowhy validate`

Falsify the graph structure against data using DoWhy's falsification tests.

```bash
causal-ts dowhy validate GRAPH --data data.csv
```

### `dowhy strength`

Compute arrow strength and causal influence for each edge.

```bash
causal-ts dowhy strength GRAPH --data data.csv
```

### `dowhy drift`

Detect distribution change / mechanism drift over time.

```bash
causal-ts dowhy drift GRAPH --data data.csv
```

---

## Typical Full Workflow

```bash
# 1. Generate data
causal-ts -s 42 -o ./run1 generate --dataset ex1 -T 500

# 2. Run discovery
causal-ts -s 42 -o ./run1 discover run1/generate/data.csv \
  --ci-test splitkci --max-lag 3 \
  --ground-truth run1/generate/ground_truth.npy

# 3. Evaluate separately (optional)
causal-ts evaluate run1/generate/ground_truth.npy \
  run1/discover/graph.npy --var-names X0,X1,X2,X3,X4
```
