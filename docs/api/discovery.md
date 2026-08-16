# Discovery API

---

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
```

---

## Feature selection

Single-target discovery: recover the causal predictors of one variable in
O(d) tests instead of the O(d²) of a full graph.

```{eval-rst}
.. autofunction:: causalts.feature_selection.select_features
```

---

## Stability

```{eval-rst}
.. autofunction:: causalts.bootstrap.temporal_bootstrap
```

---

## CDNOTS

```{eval-rst}
.. autofunction:: causalts.cdnots.phase3_utils.run_cdnots
```

---

## CDNOTS+

```{eval-rst}
.. autofunction:: causalts.cdnots.phase3_utils.run_cdnots_plus
```

---

## CdnotsResult

```{eval-rst}
.. autoclass:: causalts.cdnots.result.CdnotsResult
   :members:
   :undoc-members:
   :show-inheritance:
```

---

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

---

## GRACE

```{eval-rst}
.. autofunction:: causalts.grace.gated_discovery.run_cdnots_gated

.. autofunction:: causalts.grace.gated_discovery.run_stability_selection

.. autofunction:: causalts.grace.gated_discovery.evaluate_graph

.. autofunction:: causalts.grace.gated_discovery.prepare_data
```

---

## Legacy

:::{deprecated} 0.3.0
Use :func:`~causalts.cdnots.phase3_utils.run_cdnots` instead.
:::

```{eval-rst}
.. autofunction:: causalts.cdnots.phase3_utils.cdnots_discovery
```
