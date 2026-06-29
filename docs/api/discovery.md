# Discovery API

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
