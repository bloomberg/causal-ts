# Confounders API

**LUCID** — regime-adaptive deconfounding for causal discovery under latent
confounders. Latent confounding does not have a single statistical signature, and
applying the wrong correction can be as damaging as applying none: a low-rank
correction on sparsely confounded data deletes real edges, while conditioning on
observed controls cannot block a factor that loads on everything.

LUCID infers the regime from the residual spectrum — against the Marchenko–Pastur
no-factor null — and applies the matching correction. Both branches run the *same*
base discovery engine, so the regime affects only the correction that follows.

```python
from causalts.confounders import (
    run_lucid,          # main entry point -> LucidResult
    LucidResult,
    routed_deconfound,  # low-level, returns a bare array
    deconfound,         # post-hoc filter layer
    pds_filter,
    tetrad_filter,
    spectral_gap,
    mp_factor_count,
)
```

See the [Unobserved Confounders (LUCID)](../examples/latent_confounder_detection)
tutorial for a worked example.

---

## Running LUCID

```{eval-rst}
.. autofunction:: causalts.confounders.routed_deconf.run_lucid
```

### `LucidResult`

```{eval-rst}
.. autoclass:: causalts.confounders.result.LucidResult
   :members: top_factor_variables
```

---

## From an existing discovery

Every result object exposes the deconfounding layer, so LUCID can be applied to a
graph you already discovered — with any engine — without re-running the skeleton
search:

```python
res = run_cdnots(df, ci, num_lags=2)
res.deconfound()                      # -> LucidResult
res.tetrad_filter()                   # -> same result type, pruned
res.tetrad_filter().deconfound()      # filters chain
```

See {meth}`causalts.result.CausalResult.deconfound`.

---

## Low-level entry point

```{eval-rst}
.. autofunction:: causalts.confounders.routed_deconf.routed_deconfound
```

## Post-hoc deconfounding layer

Applies the regime-appropriate edge filter to an already-discovered graph.

```{eval-rst}
.. autofunction:: causalts.confounders.routed_deconf.deconfound
```

---

## Filters

### `pds_filter`

Post-double-selection filter over observed controls — the sparse-regime default.
Granger-style at lags ≥ 1; at lag 0 the corresponding contemporaneous partial
regression with lagged controls. Conditions on *observed* variables only, so it does
not make an unobserved common cause observable.

```{eval-rst}
.. autofunction:: causalts.confounders.routed_deconf.pds_filter
```

### `tetrad_filter`

Fixed (non-adaptive) deconfounder: always assumes pervasive factor structure and drops
lag-0 edges consistent with a shared latent cause, via the tetrad vanishing condition.
The natural comparator for LUCID's routing.

```{eval-rst}
.. autofunction:: causalts.confounders.routed_deconf.tetrad_filter

.. autofunction:: causalts.utils.tetrad.apply_tetrad_lag0_filter
```

---

## Router diagnostics

```{eval-rst}
.. autofunction:: causalts.confounders.routed_deconf.spectral_gap

.. autofunction:: causalts.confounders.routed_deconf.mp_factor_count
```
