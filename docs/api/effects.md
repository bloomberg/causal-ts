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

```{eval-rst}
.. autofunction:: causalts.effects.validate.falsify_graph

.. autofunction:: causalts.effects.validate.refute_structure

.. autofunction:: causalts.effects.validate.evaluate_model
```

---

## Graph Utilities

```{eval-rst}
.. autofunction:: causalts.effects.graph_bridge.graph_to_networkx

.. autofunction:: causalts.effects.graph_bridge.make_lagged_df
```
