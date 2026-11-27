# Discovery API

## Inspection & recommendation

Measure a dataset's health and turn those facts into a discovery
configuration — the shared core behind `causal-ts inspect`. `discover_df` is
the in-memory twin of `causal-ts discover`, and the two `*_from_graph` helpers
turn a result array into the named edge list and diagnostics that
`discover --json` emits.

```{eval-rst}
.. autofunction:: causalts.inspection.inspect_df

.. autofunction:: causalts.inspection.recommend_config

.. autofunction:: causalts.inspection.discover_df

.. autofunction:: causalts.inspection.edges_from_graph

.. autofunction:: causalts.inspection.diagnostics_from_graph

.. autofunction:: causalts.utils.heteroskedasticity.check_variance_dependence
```

## Feature selection

Single-target discovery: recover the causal predictors of one variable. The
parent-only `cedar` and `fastpc` backends need O(d) tests instead of the O(d²)
of a full graph; the `cdnots` backend and Markov-blanket mode cost O(d²).

```{eval-rst}
.. autofunction:: causalts.feature_selection.select_features

.. autoclass:: causalts.feature_selection.FeatureSelectionResult
   :members:

.. autoclass:: causalts.feature_selection.Feature
   :members:
```

## Evaluation & stability

```{eval-rst}
.. autofunction:: causalts.utils.helpers.evaluate_graph

.. autofunction:: causalts.bootstrap.temporal_bootstrap
```

## Results

Every discovery result subclasses `CausalResult`: `.cg_tig` is a
`(d, d, max_lag + 1)` array (`(d + k, d + k, …)` with `k` C-node columns
appended last) where `cg_tig[cause, effect, lag]` marks
`cause(t - lag) → effect(t)`, and the DoWhy bridge methods (`estimate_effect`,
`fit_scm`, `counterfactual`, …) are available on every result.

```{eval-rst}
.. autoclass:: causalts.result.CausalResult
   :members:
   :show-inheritance:
```

## CDNOTS

```{eval-rst}
.. autofunction:: causalts.cdnots.phase3_utils.run_cdnots
```

## CDNOTS+

```{eval-rst}
.. autofunction:: causalts.cdnots.phase3_utils.run_cdnots_plus
```

## C basis & background knowledge

```{eval-rst}
.. autofunction:: causalts.cdnots.phase3_utils.make_c_array

.. autoclass:: causalts.cdnots.ancestral.AncestralKnowledge
   :members:

.. autofunction:: causalts.cdnots.phase3_utils.make_background_knowledge

.. autofunction:: causalts.cdnots.phase3_utils.build_pvalue_matrix
```

## CdnotsResult

```{eval-rst}
.. autoclass:: causalts.cdnots.result.CdnotsResult
   :members:
   :undoc-members:
   :show-inheritance:
```

## CEDAR

```{eval-rst}
.. autofunction:: causalts.cedar.discovery.run_cedar
```

```{eval-rst}
.. autoclass:: causalts.cedar.discovery.Cedar
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: causalts.cedar.result.CedarResult
   :members:
   :undoc-members:
   :show-inheritance:
```

## GRACE · REAP

```{eval-rst}
.. autofunction:: causalts.grace.runners.run_grace

.. autofunction:: causalts.grace.runners.run_reap

.. autoclass:: causalts.grace.result.GraceResult
   :members:
   :show-inheritance:

.. autofunction:: causalts.grace.gated_discovery.run_cdnots_gated

.. autofunction:: causalts.grace.gated_discovery.run_stability_selection

.. autofunction:: causalts.grace.gated_discovery.prepare_data

.. autofunction:: causalts.grace.shrink.shrink_parents

.. autofunction:: causalts.grace.grow.grow_candidates
```

## Score-based

LGES, TGES and GES search for the best-scoring (BIC) equivalence class on the lag-embedded
variables, with edges into the past forbidden. Lasso-VAR, VARLiNGAM and Granger causality
read edges off a fitted vector autoregression. All six are also reachable as
`causal-ts discover --algorithm lges` (and so on) and through `discover_df(algorithm=...)`.

```{eval-rst}
.. autofunction:: causalts.baselines.run_baseline

.. autoclass:: causalts.baselines.BaselineResult
   :members:
   :show-inheritance:

.. autofunction:: causalts.lges.lges_discovery

.. autofunction:: causalts.lges.tges_discovery

.. autofunction:: causalts.baselines.ges_discovery

.. autofunction:: causalts.baselines.lasso_var_discovery

.. autofunction:: causalts.baselines.varlingam_discovery

.. autofunction:: causalts.baselines.granger_discovery
```

## Regime discovery

Detect regimes (change points, hidden states, distribution shifts) and discover
a causal graph per regime. See the
[regime discovery notebook](../examples/regime_discovery.ipynb).

```{eval-rst}
.. autofunction:: causalts.regime.run_regime_discovery

.. autoclass:: causalts.regime.RegimeResult
   :members:

.. autoclass:: causalts.regime.RegimeConfig

.. autoclass:: causalts.regime.RegimeParams
```

Detectors (all subclass `causalts.regime.RegimeDetector`):
`PeltRegimeDetector`, `ChangePointRegimeDetector`, `CusumRegimeDetector`,
`HMMRegimeDetector`, `KSRegimeDetector`, `LeveneRegimeDetector`,
`QuantileRegimeDetector`, `SeasonalRegimeDetector`, `ColumnRegimeDetector`,
`EnsembleRegimeDetector`.

## Tigramite wrappers

Run PCMCI, PCMCI+ and LPCMCI with causal-ts CI tests and get the same result
objects. Requires `pip install causalts[tigramite]`.

```{eval-rst}
.. autofunction:: causalts.tigramite_discovery.run_pcmci

.. autofunction:: causalts.tigramite_discovery.run_pcmciplus

.. autofunction:: causalts.tigramite_discovery.run_lpcmci

.. autofunction:: causalts.tigramite_discovery.run_tigramite
```

`run_pcmci`, `run_pcmciplus` and `run_lpcmci` return a `(G_hat, info)` tuple
by default. With `return_result=True` they return a `TigramiteResult`
instead: a {class}`~causalts.result.CausalResult` with `.cg_tig`,
`.p_matrix`, `.val_matrix`, `.plot()` and `.to_marks()` (tigramite's own
marks, `o-o`, `x-x`, `o->` and `<->` included). It also has the DoWhy bridge
(`estimate_effect`, `fit_scm`, ...). Because `cg_tig` drops lag-0
`o-o`/`x-x` and `<->` links and renders `o->` as directed, `bridge_policy` defaults to `"refuse"`:
the bridge raises while unresolved marks remain. Set
`res.bridge_policy = "drop_unresolved"` to bridge without them.
`tetrad_filter()`, `pds_filter()` and `deconfound()` work on `cg_tig`; they
leave a lag-0 `o-o` (PCMCI+'s unorientable pair) untested and in the marks,
with a warning. PAG-specific marks (`o->`, `<-o`, `<->`, `x-x`, endpoint
conflicts such as `x->`) make them raise.

`run_tigramite_algorithm` runs them by name (`"pcmci"`, `"pcmciplus"`,
`"lpcmci"`) and returns a `TigramiteResult`. It is what
`discover_df(df, algorithm="pcmciplus")`, `run_algorithm` and
`causal-ts discover --algorithm` call (see the [CLI reference](../cli)). The
data must be complete, or pass `impute="var_em"` to fill it first (the result
then holds the filled data).

```{eval-rst}
.. autoclass:: causalts.tigramite_result.TigramiteResult
   :members: graph_raw
   :show-inheritance:

.. autofunction:: causalts.tigramite_result.run_tigramite_algorithm
```

## Algorithm registry

Register your own algorithm so `run_algorithm` (and, through the
`causalts.algorithms` entry point, `causal-ts discover --algorithm`) can run it.
See the [custom algorithm notebook](../examples/custom_algorithm.ipynb).

```{eval-rst}
.. autofunction:: causalts.algorithms.register_algorithm

.. autofunction:: causalts.algorithms.list_algorithms

.. autofunction:: causalts.algorithms.run_algorithm
```

## Legacy

`cdnots_discovery` is kept for backward compatibility; prefer
:func:`~causalts.cdnots.phase3_utils.run_cdnots`, which returns a `CdnotsResult`.

```{eval-rst}
.. autofunction:: causalts.cdnots.phase3_utils.cdnots_discovery
```
