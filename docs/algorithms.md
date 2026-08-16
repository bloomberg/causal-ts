# Algorithms

Causal-TS implements four discovery algorithms for time series. Use the table below to pick the right one, then read each section for details.

---

::::{grid} 4
:gutter: 3

:::{grid-item-card}
:shadow: sm
:text-align: center

{fas}`circle-nodes;2em;sd-text-primary`

**CDNOTS**

Constraint-based · PC extension

{bdg-primary-line}`full orientation`
:::

:::{grid-item-card}
:shadow: sm
:text-align: center

{fas}`circle-nodes;2em;sd-text-primary`

**CDNOTS+**

Constraint-based · PCMCI+ skeleton

{bdg-primary-line}`full orientation`
:::

:::{grid-item-card}
:shadow: sm
:text-align: center

{fas}`shuffle;2em;sd-text-primary`

**CEDAR**

Pairwise · Scalable

{bdg-primary-line}`O(d²)`
:::

:::{grid-item-card}
:shadow: sm
:text-align: center

{fas}`brain;2em;sd-text-primary`

**GRACE**

Hybrid neural · High-dim

{bdg-primary-line}`gate values`
:::

::::

| Algorithm | Complexity | Nonstationarity | Orientation |
|-----------|-----------|-----------------|-------------|
| CDNOTS | O(d^p) | {fas}`check;sd-text-success` C node | Full (standard PC skeleton) |
| CDNOTS+ | O(d^p) | {fas}`check;sd-text-success` C node | Full (PCMCI+ two-step skeleton) |
| CEDAR | O(d²) | Via lag selection | Pairwise |
| GRACE | O(d²) + NN | {fas}`check;sd-text-success` via skeleton | CDNOTS/CDNOTS+ skeleton + orientation → gates prune edges |

:::{note}
**O(d^p) complexity:** `d` is the number of variables and `p` is the maximum conditioning set size tested during skeleton discovery (bounded by the maximum degree of the true graph). The PC skeleton grows conditioning sets from size 0 up to size `p`, giving O(d^p) CI tests in total. For sparse graphs `p` is small (typically 2–4) and the algorithm is tractable; for dense graphs `p` can grow with `d`.

Scalability in d is determined by the **CI test**, not the algorithm. Nonlinear kernel tests (KCI, SplitKCI) become expensive at large d due to growing conditioning sets. For high-dimensional problems, use linear tests (`parcorr-gpu`, `gcmi`) which scale to any d.
:::

---

## {fas}`circle-nodes;1em;sd-text-primary` CDNOTS

**Constraint-based Discovery for Nonstationary Time Series** {bdg-primary}`constraint-based` {bdg-secondary}`nonstationary`

Extends the PC algorithm to time series by embedding lagged variables and adding a time-index node **C** to handle nonstationarity. Operates in three phases:

1. **Skeleton discovery** — conditional independence tests with temporal constraints
2. **Orientation** — unshielded colliders + Meek rules with lag constraints
3. **Nonstationarity-based orientation** — uses C to infer remaining edge directions

::::{tab-set}

:::{tab-item} CDNOTS
```python
from causalts import run_cdnots
from causalts.ci_tests import ParCorrGPU

ci_test = ParCorrGPU(df.values, device="cpu")
res = run_cdnots(
    df=df, indep_test=ci_test, num_lags=3,
    include_C=True, alpha=0.05, stable=True,
)
res.plot()  # C node labels auto-appended
```
:::

:::{tab-item} CDNOTS+ (dense graphs)
```python
from causalts import run_cdnots_plus

# PCMCI+-style two-phase skeleton — restricts conditioning sets
# to discovered parents. Prefer over CDNOTS when the graph is
# dense or variables are highly coupled (hub-structured graphs).
res = run_cdnots_plus(
    df=df, indep_test=ci_test, num_lags=3,
    include_C=True, alpha=0.05,
)
```
:::

:::{tab-item} Custom C basis
```python
from causalts import run_cdnots
from causalts.cdnots.phase3_utils import make_c_array

# Built-in presets: 'linear' (default), 'linear+quad',
# 'linear+sin', 'linear+exp', 'step', 'step+linear'
res = run_cdnots(df=df, indep_test=ci_test, num_lags=3,
                 include_C=True, c_preset='linear+quad')

# Or supply any (T, k) array for full control
C = make_c_array(len(df), 'linear+sin')
res = run_cdnots(df=df, indep_test=ci_test, num_lags=3,
                 include_C=True, c_array=C)
```
:::

::::

:::{tip}
**CDNOTS vs CDNOTS+:** Use `run_cdnots` by default. Switch to `run_cdnots_plus` when the graph is likely dense (hub-structured or highly coupled) — it avoids the over-conditioning problem in standard PC by restricting conditioning sets to discovered parents. See `cdnots_or_cdnots_plus.ipynb` for a systematic comparison.

**C node basis:** The default `c_preset='linear'` (scalar time index) handles smooth monotone drift. For quadratic trends use `'linear+quad'`, for seasonal effects `'linear+sin'`. Supply `c_array` for any custom nonstationarity pattern. See `multi_c_nonstationarity.ipynb`.
:::

---

## {fas}`shuffle;1em;sd-text-primary` CEDAR

**Causal Edge Discovery for Autoregressive processes** {bdg-primary}`pairwise` {bdg-secondary}`scalable`

Pairwise algorithm with O(d²) complexity. Uses `partial_dcor` lag selection (unbiased distance correlation on Y(t) residualised on Y(t-1), suppressing AR-mediated lag inflation) followed by two CI tests per candidate edge.

```python
from causalts.cedar import run_cedar
from causalts.ci_tests import ParCorrGPU

ci_test = ParCorrGPU(df.values, device="cpu")
result = run_cedar(
    df=df, ci_test=ci_test, max_lag=3,
    lag_method="dcor", alpha_cond1=0.05, alpha_cond2=0.05,
)
result.plot()
```

**Lag selection methods** (`lag_method`):

| Value | Method | Best when |
|-------|--------|-----------|
| `dcor` | Unbiased U-centered distance correlation | General nonlinear; **default** |
| `dcor_biased` | Biased distance correlation | Marginal speed advantage |
| `pearson` | Pearson correlation | Confirmed linear data; fastest |
| `lasso` | LassoCV per pair (SyPI, Mastakouri et al. 2021) | Sparse linear; simultaneous lag fitting |

:::{tip}
**Best for:** Fast pairwise results at any dimensionality. O(d²) pairwise testing and `target_var` mode (O(d)) keep runtime manageable.

**Contemporaneous effects:** Pass `include_lag0=True` to test for same-time (lag-0) dependencies. Off by default.

**Background knowledge:** Pass `forbidden=[("X0","X1",1)]` and `required=[("X2","X3",2)]` to constrain the discovered graph.

**Missing values:** Pass `impute="var_em"` for VAR-EM pre-filling, or leave unset for per-test pairwise-complete handling.

**DoWhy integration:** `result.estimate_effect(...)`, `result.fit_scm()`, `result.counterfactual(...)` are available directly on the returned `CedarResult`.
:::

---

## {fas}`brain;1em;sd-text-primary` GRACE

**Gated Refinement After Constraint-based Estimation** {bdg-primary}`neural` {bdg-secondary}`high-dim`

Two-stage hybrid: constraint-based skeleton (CDNOTS or CDNOTS+, high recall) + neural gated refinement with L0 (Hard Concrete) regularization to prune false positives. Any PC-family algorithm that produces a skeleton can serve as the first stage.

:::::{tab-set}

::::{tab-item} Standard
```python
from causalts.grace import run_cdnots_gated

res = run_cdnots_gated(
    df=df, max_lag=3, alpha=0.05,
    gate_threshold=0.5, max_epochs=150,
    device="cpu", verbose=False,
)
G_hat, gate_values = res.cg_tig, res.gate_values
```
::::

::::{tab-item} Stability Selection *(experimental)*
```python
from causalts.grace import run_stability_selection

res = run_stability_selection(
    df=df, max_lag=3,
    n_subsamples=20, stability_threshold=0.6,
)
G_hat, scores = res.cg_tig, res.stability_scores
```

:::{warning}
Stability selection runs GRACE on random subsamples without a pre-filtered skeleton. This is only practical for small, low-dimensional problems — at high d it becomes prohibitively slow and the subsampling instability dominates. For high-d use cases, provide a precomputed skeleton to `run_cdnots_gated` instead of using stability selection.
:::
::::

:::::

**Key hyperparameters:**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `gate_threshold` | `0.5` | Edges with gate value below this are pruned. Lower → sparser graph. |
| `stability_threshold` | `0.6` | (SS only) Minimum selection frequency across subsamples. |
| `max_epochs` | `150` | Training budget. Increase for complex data. |
| `patience` | `20` | Early-stopping patience in epochs. |
| `batch_size` | auto | Auto-set based on T. Set manually if OOM. |
| `include_C` | `True` | Include the C nonstationarity node in the skeleton. Set `False` for already-detrended data. |
| `c_preset` | `linear` | C basis for that skeleton — same presets as CDNOTS. |
| `include_C_in_model` | `False` | Also feed C to the gated model (not just the skeleton). Requires `include_C=True`. |

**Interpreting `gate_values`:** Values near 1.0 indicate high confidence edges; near 0.0 are pruned. Plot `gate_values` as a heatmap to visualize edge confidence.

:::{tip}
**Best for:** When false positive control is critical. For high-dimensional problems, provide a precomputed skeleton (e.g. CDNOTS with `max_degree=1` and a fast CI test) to `run_cdnots_gated` to pre-filter edges before gated training.

**Lambda selection:** The regularization strength λ is set automatically via an adaptive formula. For data-driven selection, use `compute_lambda_cv()` from `causalts.grace` to run cross-validation before calling `run_cdnots_gated`.

**DoWhy integration:** Use `wrap_graph()` from `causalts.effects` to attach effect estimation and counterfactual methods to GRACE's output graph.
:::
