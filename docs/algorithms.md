# Algorithms

Causal-TS implements five discovery algorithms for time series in three families: the constraint-based **CDNOTS** and **CDNOTS+**, the pairwise **CEDAR** for autoregressive processes, and the grow-shrink refinements **GRACE** and **REAP**, which refine a candidate graph from any method (CDNOTS by default). Latent-confounder correction ([LUCID](api/confounders.md)) and [regime discovery](examples/regime_discovery.ipynb) work on top of any of them.

For a recommended configuration for your data, run `inspect_df(df)` or `causal-ts inspect data.csv` (see [Choosing a configuration](getting_started/index.md#choosing-a-configuration)).

::::{grid} 4
:gutter: 3

:::{grid-item-card}
:shadow: sm
:text-align: center

{fas}`circle-nodes;2em;sd-text-primary`

**CDNOTS**

Constraint-based · PC extension

{bdg-primary-line}`CPDAG + C orientation`
:::

:::{grid-item-card}
:shadow: sm
:text-align: center

{fas}`circle-nodes;2em;sd-text-primary`

**CDNOTS+**

Constraint-based · PCMCI+ skeleton

{bdg-primary-line}`CPDAG + C orientation`
:::

:::{grid-item-card}
:shadow: sm
:text-align: center

{fas}`shuffle;2em;sd-text-primary`

**CEDAR**

Pairwise · Autoregressive

{bdg-primary-line}`pair-wise`
:::

:::{grid-item-card}
:shadow: sm
:text-align: center

{fas}`code-merge;2em;sd-text-primary`

**GRACE · REAP**

Grow-shrink refinement

{bdg-primary-line}`any candidate graph`
:::

::::

| Algorithm | CI tests | Nonstationarity | Orientation |
|-----------|-----------|-----------------|-------------|
| CDNOTS | Exponential in the conditioning-set size `p` | {fas}`check;sd-text-success` C node | Colliders, Meek rules and C; unresolved lag-0 edges stay undirected (`o-o`) |
| CDNOTS+ | As CDNOTS, conditioning on discovered parents | {fas}`check;sd-text-success` C node | As CDNOTS; unoriented lag-0 edges are dropped from `cg_tig` (kept in `to_marks()`); opt-in LiNGAM orientation |
| CEDAR | O(d²) pairs, plus MCI pruning | {fas}`check;sd-text-success` C node (`include_C`) | Lagged edges oriented by time |
| GRACE | Candidate graph (CDNOTS by default) + O(d²) grow tests + neural training | {fas}`check;sd-text-success` via the candidate graph | Lagged edges (oriented by time): gates, then grow rounds and retraining; lag-0 edges are not refined |
| REAP | Candidate graph + O(d²) grow tests + regressions | {fas}`check;sd-text-success` via the candidate graph | Lagged edges: backward elimination by partial F-tests, then three grow → re-shrink rounds (no training); optional lag-0 step |

:::{note}
**Cost of the PC skeleton:** `d` is the number of variables and `p` the largest conditioning set tested (bounded by the maximum degree of the true graph). Each of the roughly d²·L candidate lagged pairs is tested against conditioning sets of size 0..`p` drawn from its neighbours, so the number of tests grows polynomially in `d` with exponent set by `p`. For sparse graphs `p` is small (typically 2–4) and discovery is tractable; for dense graphs `p` grows with `d`.

In practice the **CI test** sets how far CDNOTS scales in `d`: kernel tests (`kci`, `splitkci`) become expensive as conditioning sets grow, while `parcorr-gpu`, `gcmi` and `raft` stay fast. There is no dimension above which another algorithm is required.
:::

## {fas}`circle-nodes;1em;sd-text-primary` CDNOTS

**Constraint-based Discovery for Nonstationary Time Series** {bdg-primary}`constraint-based` {bdg-secondary}`nonstationary`

Extends the PC algorithm to time series by embedding lagged variables and adding a time-index node **C** to handle nonstationarity. Operates in three phases:

1. **Skeleton discovery** — conditional independence tests with temporal constraints
2. **Orientation** — unshielded colliders + Meek rules with lag constraints
3. **Nonstationarity-based orientation** — uses C to infer remaining edge directions

The result is a CPDAG: contemporaneous edges that none of these rules can orient stay undirected (`o-o`) or conflicting (`x-x`); `result.to_marks()` shows them and `result.cg_tig` stores them as symmetric entries. `run_cdnots` defaults to `stable=True` (order-independent skeleton); the CLI's `--stable` flag is off by default. Other orientation options (`orient_margin`, `collider_search`, `majority_gamma`, the ambiguity band of the majority collider rule) are described in the `run_cdnots` docstring.

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
    include_C=True, alpha=0.01,
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
**CDNOTS vs CDNOTS+:** Use `run_cdnots` by default. Switch to `run_cdnots_plus` when the graph is likely dense (hub-structured or highly coupled) — it avoids the over-conditioning problem in standard PC by restricting conditioning sets to discovered parents, and `alpha_mci` sets its MCI level apart from `alpha`. See [CDNOTS or CDNOTS+?](examples/cdnots_or_cdnots_plus.ipynb) for a systematic comparison.

**C node basis:** The default `c_preset='linear'` (scalar time index) handles smooth monotone drift. For quadratic trends use `'linear+quad'`. `'linear+sin'` adds a single slow oscillation over the whole sample (`sin(2πt/T)`), not a seasonal cycle; for seasonality at a known period, supply `c_array` with the matching sine/cosine columns. See [Multi-C nonstationarity](examples/multi_c_nonstationarity.ipynb).
:::

### Optional CDNOTS+ steps

`run_cdnots_plus` has two extra steps, both off by default. Each helps when its assumptions hold, and can cost accuracy when they do not, so turn them on deliberately.

```python
res = run_cdnots_plus(
    df=df, indep_test=ci_test, num_lags=3, include_C=False, alpha=0.01,
    lag0_orient="lingam",                     # orient undirected lag-0 edges
    addback_alpha=1e-4, veto_alpha=0.001,     # re-test lagged links
)
res.lag0_orientations   # one record per o-o / x-x lag-0 pair it considered
res.lag_refinements     # links added or removed, with p-values
```

**`lag0_orient="lingam"`** orients lag-0 edges that conditional independence leaves undirected (`o-o`) or conflicting (`x-x`). These are the edges PCMCI+-style search cannot orient. It orders the innovations of each such component with DirectLiNGAM, and orients an edge only when:

- the direction holds in 90% of bootstrap re-orderings,
- an endpoint's innovations are non-Gaussian,
- and the edge creates no collider the CI step ruled out.

**`addback_alpha` / `veto_alpha`** re-test lagged cross links after orientation, using the full MCI conditioning set (lag-0, self and C links are not touched). Absent links with `p < addback_alpha` are added, then present links with `p > veto_alpha` are removed, and the graph is re-oriented until nothing changes (at most 5 passes). This recovers weak links that the first skeleton pass drops before the target's own past is conditioned on. It needs contiguous lags (`num_lags`, or a `lag_list` of `1..L`) and cannot be combined with `max_conds_py` / `max_conds_px`.

| | `lag0_orient="lingam"` | `addback_alpha` / `veto_alpha` |
|---|---|---|
| Helps | Linear mechanisms with non-Gaussian noise: F1 +0.02 to +0.04 when most edges are contemporaneous. | Linear simulations at `alpha=0.01`: SHD lower in every setting tried, F1 up to +0.05 (but -0.03 at T=100 with d=50). |
| Neutral | Linear mechanisms with Gaussian noise: the normality gate usually keeps it from acting (the test rejects about 1% of Gaussian samples). | Nonlinear data with the `raft` CI test. |
| Hurts | Spline mechanisms with Laplace or Gaussian noise: most of its orientations are wrong. | A CI test without power at the sample size (e.g. `splitkci` at T=300): the veto removes true links it cannot detect. |
| Real data (CausalRivers) | F1 unchanged, SHD +0.5. | F1 unchanged, SHD +2: on long series it adds about four false links per true one. |
| Cost | 101 DirectLiNGAM orderings per undirected component. | About 1.5–2.7× a plain run. |

Neither has improved results on the real data tested so far. Use `lag0_orient` when relationships are close to linear with non-Gaussian noise, and the refinement when the CI test is well powered and you can afford more false links in exchange for weak true ones. The `run_cdnots_plus` docstring has the full settings and numbers. Both are Python-only options (not exposed by `causal-ts discover`).

## {fas}`shuffle;1em;sd-text-primary` CEDAR

**Causal Edge Discovery for Autoregressive processes** {bdg-primary}`pairwise` {bdg-secondary}`autoregressive`

Pairwise algorithm over the O(d²) ordered pairs, designed for autoregressive processes (each variable driven by its own past plus lagged parents). For each pair it selects the minimum lag (by default `partial_dcor`: unbiased distance correlation on Y(t) residualised on Y(t-1), suppressing AR-mediated lag inflation), runs two CI tests for that candidate edge, and then prunes the retained edges with an MCI test (`prune=True`). Self-loops come from the estimated or assumed AR order (`include_autoreg`, `assume_ar1`). The C node is supported (`include_C`, `c_preset`).

```python
from causalts.cedar import run_cedar
from causalts.ci_tests import ParCorrGPU

ci_test = ParCorrGPU(df.values, device="cpu")
result = run_cedar(
    df=df, ci_test=ci_test, max_lag=3,
    alpha_cond1=0.01, alpha_cond2=0.01,   # the Python defaults
)
result.plot()
```

:::{note}
`causal-ts discover --algorithm cedar` uses different defaults from `run_cedar`: `alpha_cond1`/`alpha_cond2` 0.05 (Python 0.01), `filter_children` and the C node on (Python off), and the AR order estimated by BIC unless `--assume-ar1` is given (Python assumes AR(1)).
:::

**Lag selection methods** (`lag_method`):

| Value | Method | Best when |
|-------|--------|-----------|
| `partial_dcor` | Unbiased distance correlation on Y(t) residualised on Y(t-1) | General; suppresses lags inflated by autocorrelation; **default** |
| `dcor` | Unbiased U-centered distance correlation | General nonlinear |
| `dcor_biased` | Biased distance correlation | Marginal speed advantage |
| `pearson` | Pearson correlation | Confirmed linear data; fastest |
| `lasso` | LassoCV per pair (SyPI, Mastakouri et al. 2021) | Sparse linear; simultaneous lag fitting |

:::{tip}
**Best for:** Autoregressive systems, and single-target questions ("what drives Y?") via `target_var` mode (O(d)). It is not a generic replacement for CDNOTS: its pairwise tests assume the autoregressive design.

**Contemporaneous effects:** Pass `include_lag0=True` to test for same-time (lag-0) dependencies. Off by default.

**Background knowledge:** Pass `forbidden=[("X0","X1",1)]` and `required=[("X2","X3",2)]` to constrain the discovered graph.

**Missing values:** Pass `impute="var_em"` for VAR-EM pre-filling, or leave unset for per-test pairwise-complete handling.

**DoWhy integration:** `result.estimate_effect(...)`, `result.fit_scm()`, `result.counterfactual(...)` are available directly on the returned `CedarResult`.
:::

## {fas}`code-merge;1em;sd-text-primary` GRACE · REAP

**Grow-shrink refinement of a candidate graph** {bdg-primary}`any candidate graph` {bdg-secondary}`gates | regression`

Both take a directed, lag-resolved candidate graph — CDNOTS by default (built with the CI test you pass as `ci_test_class`, or by name through `discover_df` / `causal-ts discover --ci-test`), or any method's graph as `candidate_graph` — and refine its **lagged** edges: a shrink step prunes the candidate parents, then grow rounds test the lagged pairs outside the graph, add the hits and shrink again. Lag-0 edges of the candidate graph are not carried over. They differ in the shrink step and in what they can find:

| | GRACE (`run_grace`) | REAP (`run_reap`) |
|---|---|---|
| Shrink | Neural model with one Hard Concrete (L0) gate per candidate edge | Backward elimination by partial F-tests on additive spline regressions |
| Training | Yes (slower; GPU helps) | None |
| Parents that change only a target's noise scale | Proposed by the grow step (`grow_variance`) and kept by the gated model's likelihood | Not detected (mean effects only) |
| Edge scores | `gate_values` | `edge_pvalue`, `edge_strength` (partial R²) |
| Contemporaneous edges | Not refined (keep the candidate graph's lag-0 edges) | Optional, `contemporaneous=True` (CDNOTS+ with the lagged edges pinned) |

Use REAP by default; use GRACE when some variables may act only on another's variance (`inspect_df` reports this as `facts["variance_dependence"]` and then recommends GRACE).

### GRACE

**Gated Refinement for Accurate Causal Edge discovery.** Two-stage hybrid: a high-recall directed candidate graph (CDNOTS by default, or any `candidate_graph` you supply, e.g. a loose PCMCI+ run) + neural gated refinement with L0 (Hard Concrete) regularization (`run_grace`). The gates prune false positives; the grow round (`grow_rounds`, one by default) also adds edges the candidate graph missed, so GRACE is a refinement stage rather than a pure pruner. It helps most on loose, high-recall candidate graphs and can only lose on one that is already precise.

### REAP

**REAP** (`run_reap`) swaps the neural stage for a regression-based one: each target's candidate parents enter an additive least-squares regression (one natural-spline block per lagged parent, plus the C basis on the CDNOTS candidate graph when `include_C=True`, or on a supplied one with `include_C_in_model=True`), and the least significant parent is dropped while its block's partial F-test p-value exceeds `prune_alpha` (Huber-clipped residuals, so heavy tails cannot decide a test). These are fixed-basis F-tests: their p-values are approximate and become liberal at large T when effects are strongly curved, so treat `edge_pvalue` as a ranking score rather than a calibrated test (use the `raft` CI test for that, or `rift` when conditioning effects interact or are wiggly). Nothing is trained and no cutoff is chosen; with `grow_rounds` (three by default) it alternates grow and shrink. It returns `edge_pvalue` and `edge_strength` (partial R², a continuous edge score) instead of `gate_values`. The shrink handles lagged edges; `contemporaneous=True` adds lag-0 edges afterwards by running CDNOTS+ with every lagged edge pinned to the shrink's selection, so CDNOTS+ only tests lag-0 adjacencies (conditioning on the selected lagged parents) and orients them by its collider and Meek rules, which use the selected lagged arrows, and by its nonstationarity rule when C is included. The lagged edges are unchanged; unoriented lag-0 edges are dropped from `cg_tig`, as in CDNOTS+, and kept in `to_marks()`. On long series (T in the thousands) the shrink's tests keep real but negligible dependencies; `effect_floor="adaptive"` prunes lagged edges whose unique effect is below 0.01 or below a fifth of the target's strongest edge.

:::::{tab-set}

::::{tab-item} Standard
```python
from causalts import run_grace

res = run_grace(df, max_lag=3, verbose=False)  # CDNOTS candidate graph
G_hat, gate_values = res.cg_tig, res.gate_values

# or refine another method's directed graph, shape (d, d, max_lag+1)
res = run_grace(df, max_lag=3, candidate_graph=G_pcmci, verbose=False)
```
::::

::::{tab-item} REAP (regression-based)
```python
from causalts import run_reap

res = run_reap(df, max_lag=3, verbose=False)  # 3 grow rounds by default
G_hat, strength, pvals = res.cg_tig, res.edge_strength, res.edge_pvalue

# lag-0 edges too (CDNOTS+ with the lagged edges pinned)
res0 = run_reap(df, max_lag=3, contemporaneous=True, verbose=False)
marks = res0.to_marks()  # "-->", "<--", "o-o" at lag 0
```
::::

::::{tab-item} Stability Selection *(experimental)*
```python
from causalts.grace import run_stability_selection

res = run_stability_selection(
    df=df, max_lag=3,
    n_subsamples=20, stability_threshold=0.6,   # n_subsamples defaults to 1
)
G_hat, scores = res.cg_tig, res.stability_scores
```

`causal-ts discover --algorithm grace-ss` always uses a CI skeleton (`use_ci_skeleton=True`); in Python it is off by default, and `include_C` only has an effect with it.

:::{warning}
Stability selection runs GRACE on random subsamples without a pre-filtered skeleton. This is only practical for small, low-dimensional problems — at high d it becomes prohibitively slow and the subsampling instability dominates. For high-d use cases, pass a candidate graph to `run_grace` instead of using stability selection.
:::
::::

:::::

**Key hyperparameters** (as named in `run_cdnots_gated`, which `run_grace` and `run_reap` call with `refine="gates"` and `refine="shrink"`; both take `candidate_graph` for `skeleton`, and `run_reap` drops the `shrink_` prefix, with `shrink_alpha` as `prune_alpha` and `shrink_effect_floor` as `effect_floor`):

| Parameter | Default | Effect |
|-----------|---------|--------|
| `gate_threshold` | `'auto'` | Cutoff on gate values. `'li'` picks it per run from the candidate gates (Li minimum cross-entropy). `'auto'` uses Li on GRACE's own skeleton, and on a supplied `skeleton` only when the gates form two clusters (else 0.5, so a precise skeleton is not split). A number fixes it (`0.5` was the old default); lower → denser graph. The applied value is in `res.gate_threshold`. |
| `data_transform` | `'rank'` | Rank-Gaussianize every continuous column before skeleton, training and grow (missing values stay missing). Discrete columns keep their codes, except that REAP's shrink and grow (spline basis) see them rank-transformed too. `None` uses raw data. |
| `noise_model` | `'student_t'` | Likelihood of the gated model's errors; Student-t keeps heavy-tailed noise from dominating the fit. `'gaussian'` was the previous default. |
| `grow_rounds` | `1` (`'gates'`), `3` (`'shrink'`) | Grow-shrink rounds. Each one tests the lagged pairs outside the skeleton (additive F-test, BH at `grow_q=0.1`; random-Fourier features under `'gates'`, `shrink_basis` features plus optional interaction nuisance under `'shrink'`), adds the hits and refines again (retrains the gates, or re-runs the shrink). Recovers edges the skeleton missed (not reliably a parent that acts only through an interaction); under `refine='gates'` costs one extra training per round, under `'shrink'` one more shrink with the `shrink_basis` features. |
| `grow_transform` | `'auto'` | Run the grow tests on rank-Gaussianized data (`'rank'`) or as given (`None`); `'auto'` is `None` under `'gates'` and `'rank'` under `'shrink'` (REAP). |
| `grow_nuisance_basis` | `'spline'` | Basis of the current parents in the grow tests (and the AR path's self-loop test): a full-range natural spline, which keeps those tests near their level at large T when parents are persistent. `'rff'` uses the RFF block; `None` uses the tested block's basis (RFF under `'gates'`). |
| `grow_variance` | `True` with `'gates'` and `grow_rounds >= 1`, else `False` | *(`'gates'`, needs `grow_rounds`)* Each grow round also proposes parents that change only a target's noise scale (test on the size of the residual); the gated model's likelihood can keep them. |
| `ar_path` | `True` with `'gates'` and `max_lag >= 1`, else `False` | *(`'gates'`)* Own lags get an ungated autoregressive head and self-loops leave the gates; cross edges are cut at max(Li, 0.5), and self-loops are kept by a partial F-test given the kept cross parents and the other own lags (the own lag alone and a 4-knot spline of it, Bonferroni over the tests run: each at p < 0.0005 when both run). Use it for strongly autoregressive systems, where gated self-loops absorb signal. |
| `self_loop_basis` | `'union'` | (`ar_path=True`) Test for the AR path's self-loops: `'union'` is the linear-plus-spline test above; `'rff'` is the 9-column RFF block used before, which reproduces the earlier self-loops (fewer kept at small T). |
| `refine` | `'gates'` | `'gates'`: neural gated refinement. `'shrink'`: regression-based backward elimination of each target's candidate parents (no training; gate, likelihood and training options do not apply; lagged edges unless `contemporaneous=True`). |
| `shrink_alpha` | `0.001` | (`refine='shrink'`) Significance level for keeping a parent. |
| `shrink_basis` | `'spline'` | (`refine='shrink'`) Feature block per lagged parent in the shrink and grow tests: natural cubic spline (for continuous rank-transformed inputs, much lower extreme-row leverage than RFF) or the grow step's original `'rff'`. |
| `shrink_robust` | `True` | (`refine='shrink'`) Huber-clip residuals before each shrink test. |
| `shrink_interactions` | `False` | (`refine='shrink'`) Add pairwise products of each target's parents to the shrink and grow regressions. Keeps precision on non-additive mechanisms (parents that interact), where an additive fit admits correlated non-parents as T grows; costs some power on dense graphs. |
| `shrink_effect_floor` | `None` | (`refine='shrink'`) After the shrink and grow rounds, prune lagged cross edges with a small unique effect (adjusted partial R²): `'adaptive'` drops, per target, edges below `max(0.01, 0.2 × strongest)` (backward elimination; self-loops stay), so the threshold scales with each target's signal (real parents weaker than that are dropped too); a float is a fixed floor; not with `shrink_interactions=True`. For long series (T in the thousands), where significance tests keep real but negligible dependencies. |
| `contemporaneous` | `False` | (`refine='shrink'`) Add lag-0 edges with CDNOTS+, lagged edges pinned to the shrink's selection (see above). Off: the shrink is lagged-only and its output is unchanged. |
| `contemporaneous_test` | `'spline'` | (`contemporaneous=True`) CI test of the lag-0 step: `'spline'` is the shrink's own test (liberal when effects are wiggly, like the shrink's p-values); any registered name, e.g. `'raft'` (calibrated, less power at small T), or an instance. |
| `contemporaneous_alpha` | `0.001` | (`contemporaneous=True`) Significance level of the lag-0 step. |
| `include_C_in_model` | `False` | Under `refine='shrink'`: condition every shrink and grow regression on the C basis also for a supplied `skeleton`, independently of `include_C` (GRACE's own skeleton is conditioned on C when `include_C=True`). Under `'gates'` it requires `include_C=True`. |
| `stability_threshold` | `0.6` | (SS only) Minimum selection frequency across subsamples. |
| `max_epochs` | `150` | Training budget. Increase for complex data. |
| `patience` | `20` | Early-stopping patience in epochs. |
| `batch_size` | auto | Auto-set based on T. Set manually if OOM. |
| `include_C` | `True` | Include the C nonstationarity node in the skeleton. Set `False` for already-detrended data. |
| `c_preset` | `linear` | C basis for that skeleton — same presets as CDNOTS. |
| `effect_floor` | `1 - exp(-0.02)` ≈ 0.0198 (GRACE) | Cross edges whose adjusted partial R² is below this are removed after the gates. REAP's `effect_floor` is the `shrink_effect_floor` above (default `None`). |
| `lambda_rule` | `'xcorr'` | Closed-form rule for the L0 strength λ (see the tip below); `lambda_cv=True` selects λ by cross-validation instead. |
| `grow_q` | `0.1` | Benjamini-Hochberg level of the grow tests. |

**Interpreting `gate_values`:** Values near 1.0 indicate high confidence edges; near 0.0 are pruned. Plot `gate_values` as a heatmap to visualize edge confidence. With `ar_path=True` (the default under `refine="gates"`) self-loops are not gated: their entries are 0 and the F-test decides them.

:::{tip}
**Candidate graphs:** GRACE and REAP help most on a loose, high-recall candidate graph and can only lose on one that is already precise. Any method's directed graph can be passed as `candidate_graph`.

**Lambda selection:** The regularization strength λ is set automatically by `lambda_rule="xcorr"` (default; described below). `lambda_rule="density"` is the earlier adaptive formula; `lambda_rule="dT"` uses λ = 300/(d·T), which keeps falling with d where the density formula levels off. Its constant was fitted on calibration sweeps (`ar_path=True`, rank data, Student-t likelihood; sparse and dense graphs; d 5–100, T 250–1000), where at d ≥ 50 it kept true nonlinear edges the density formula dropped and the skeleton density did not predict the optimal λ. It warns outside d 10–100, T 250–1000 (at d=5 it lost to the density formula); other settings are untested. When refining a loose supplied skeleton on a sparse graph, the optimal λ was about 1.3× larger. `lambda_rule="xcorr"` bounds λ to [0.3/d, 0.5] and scales it with the data's background lagged cross-correlation, so it does not vanish on long series and prunes more under strong autoregression or persistent latent drivers; it was fitted on d 5–100, T 250–10000 (it warns outside) and matches `"dT"` inside that rule's range. For data-driven selection, pass `lambda_cv=True` to `run_grace` (or call `compute_lambda_cv()` from `causalts.grace` yourself).

**DoWhy integration:** `run_grace` and `run_reap` return a `GraceResult`, so `res.estimate_effect(...)`, `res.fit_scm()` and `res.counterfactual(...)` are available directly.
:::
