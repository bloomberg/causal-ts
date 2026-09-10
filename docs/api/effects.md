# Effects API

:::{note}
Requires `pip install causalts[dowhy]`.
:::

All effect estimation functions accept the 3D graph array (`d × d × (max_lag+1)`) returned by any Causal-TS discovery algorithm. For algorithms that return a plain NumPy array (e.g. GRACE), use {py:func}`wrap_graph` first.

---

## Workflow: Discovery → Effects

```python
# 1. Discover
from causalts import run_cdnots
from causalts.ci_tests import ParCorrGPU

ci = ParCorrGPU(df.values, device="cpu")
res = run_cdnots(df=df, indep_test=ci, num_lags=3)   # returns CdnotsResult

# 2. Estimate effect directly from result (bridge methods attached)
ate = res.estimate_effect("X1", "X3", treatment_lag=1)

# 3. Or wrap a plain graph (GRACE, CEDAR)
from causalts.effects import wrap_graph
wrapped = wrap_graph(graph, df)
ate = wrapped.estimate_effect("X1", "X3", treatment_lag=1)
```

---

## Convenience Wrapper

### `wrap_graph`

Wrap any 3D graph array with bound effect-estimation methods (same interface as `CdnotsResult` / `CedarResult`).

```{eval-rst}
.. autofunction:: causalts.effects.wrap.wrap_graph
```

### `WrappedGraph`

```{eval-rst}
.. autoclass:: causalts.effects.wrap.WrappedGraph
   :members:
   :show-inheritance:
```

---

## Effect Estimation

```{eval-rst}
.. autofunction:: causalts.effects.effect.estimate_effect
```

---

## Structural Causal Models

```{eval-rst}
.. autofunction:: causalts.effects.scm.fit_scm

.. autofunction:: causalts.effects.scm.counterfactual
```

---

## Root Cause Analysis

```{eval-rst}
.. autofunction:: causalts.effects.root_cause.attribute_anomaly
```

---

## Influence & Feature Relevance

```{eval-rst}
.. autofunction:: causalts.effects.influence.arrow_strength

.. autofunction:: causalts.effects.influence.causal_influence

.. autofunction:: causalts.effects.influence.parent_relevance

.. autofunction:: causalts.effects.influence.distribution_change
```

---

## Validation

A lag embedding is a *conditional transition graph*, not a joint DAG over the
embedded row. It asserts `p(H_t, X_t) = p(H_t) · Π_i p(X_i,t | Pa(X_i,t))` with
the history slice `p(H_t)` **unrestricted** — nothing claims that `X_0(t-1)` and
`X_1(t-1)` are independent, and in any autoregressive process they are not. So
the local Markov condition is validated only at current-time nodes, and the
question "is `max_lag` deep enough?" is a separate diagnostic rather than part of
the same verdict.

```{eval-rst}
.. autofunction:: causalts.effects.validate.validate_transition_graph

.. autoclass:: causalts.effects.validate.TransitionValidationResult
   :members:

.. autofunction:: causalts.effects.validate.history_sufficiency

.. autoclass:: causalts.effects.validate.HistorySufficiencyResult
   :members:

.. autofunction:: causalts.effects.validate.linear_ci_test

.. autofunction:: causalts.effects.refute_effect

.. autofunction:: causalts.effects.sensitivity_analysis

.. autofunction:: causalts.effects.validate.falsify_graph

.. autofunction:: causalts.effects.validate.refute_structure

.. autofunction:: causalts.effects.validate.evaluate_model
```

---

## Exporting to DoWhy

For driving DoWhy directly rather than through the wrappers above. There is no
`to_dowhy()` with a default flavour: identification needs the *unrolled* graph so
the lagged treatment has parents, and exporting the transition graph instead
silently empties the adjustment set.

```{eval-rst}
.. autoclass:: causalts.effects.export.DoWhyArtifacts

.. autofunction:: causalts.effects.export.build_identification_artifacts

.. autofunction:: causalts.effects.export.build_transition_artifacts
```

### Keeping the identification guarantee with your own estimator

`model.identify_effect()` on an exported identification graph returns **DoWhy's
own** minimal backdoor set, not the treatment's parents `estimate_effect()`
guarantees — those are not always the same, and DoWhy's own choice depends on
the graph's lag horizon being deep enough to expose every confounding path,
which the treatment's parents do not. If you want that guarantee with an estimator this package
does not wrap (`backdoor.generalized_linear_model`, an econml estimator, ...),
call `force_parent_adjustment_set()` between `identify_effect()` and
`estimate_effect()`, then `verify_parent_adjustment_set()` on the result:

```python
from causalts.effects import force_parent_adjustment_set, verify_parent_adjustment_set

identified = model.identify_effect(proceed_when_unidentifiable=True)
parents = force_parent_adjustment_set(
    identified, graph, treatment, treatment_lag, outcome, var_names,
    method="backdoor.generalized_linear_model", available=list(artifacts.data.columns),
)
estimate = model.estimate_effect(identified, method_name="backdoor.generalized_linear_model", ...)
verify_parent_adjustment_set(estimate, parents)  # raises if it silently didn't take
```

```{eval-rst}
.. autofunction:: causalts.effects.graph_bridge.force_parent_adjustment_set

.. autofunction:: causalts.effects.graph_bridge.verify_parent_adjustment_set
```

---

## Graph Utilities

```{eval-rst}
.. autofunction:: causalts.effects.graph_bridge.graph_to_networkx

.. autofunction:: causalts.effects.graph_bridge.make_lagged_df
```
