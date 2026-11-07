# Confounders API

**LUCID** — regime-adaptive deconfounding for causal discovery under latent
confounders. Latent confounding does not have a single statistical signature, and
applying the wrong correction can be as damaging as applying none: a low-rank
correction on sparsely confounded data deletes real edges, while conditioning on
observed controls cannot block a factor that loads on everything.

LUCID infers the regime from the residual spectrum — against the Marchenko–Pastur
no-factor null — and applies the matching correction. Both branches run the *same*
base discovery engine, so the regime affects only the correction that follows. See
[arXiv:2609.31315](https://arxiv.org/abs/2609.31315).

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

## Factor-adjusted lagged edges

By default LUCID deconfounds only the contemporaneous (lag-0) slice. Its lagged edges
come from the base skeleton on the observed data. There, factor variance swamps the
idiosyncratic signal, so true lagged edges are missed, and a persistent
(autocorrelated) factor can also link `X_i(t-1)` and `X_j(t)` spuriously. In our
benchmarks the gain is recall: adjusting recovers many true lagged edges but also adds
some false ones (false positives rose in almost every benchmark cell), so expect more
lagged edges, not fewer.
`pervasive_lagged="factor_adjusted"` re-runs the lagged search with
LUCID's private `FactorParCorr`, a partial correlation that
also partials out principal-component factor scores:

```python
res = run_lucid(df, max_lag=2, pervasive_lagged="factor_adjusted")
res.info["pervasive_lagged"]    # "factor_adjusted" if it ran, else "observed"
```

Two details make the test valid when factors are estimated from the same panel:

- **The factors for a test of `X` and `Y` are estimated without `X` and `Y`.**
  Residuals after removing `k` PCA factors satisfy `k` exact linear constraints. With
  block-sparse loadings, a series is then an exact combination of a few groupmates, and
  conditioning on them makes a true edge look like independence.
- **Only scores at the older time point of the pair and before are used.** A current
  factor score is a sum of current series, including the children of a lagged cause.
  Conditioning on it opens collider paths.

The option applies only on the `pervasive` route. On `sf` the benchmark results were
mixed, and `sparse` has no factors to adjust for. The lag-0 slice is still LUCID's S-L
recovery. The default stays `"observed"`, so results from earlier versions are unchanged.

Paired change versus default LUCID, over pervasive-routed units only (10 seeds per problem
unless noted; `sf` and `sparse` units are identical by construction). The route restriction
was chosen after seeing seeds 0–9 of the held-out families; seeds 10–19 alone give
ΔF1 +0.074 and ΔSHD −0.99. All Wilcoxon p < 1e-9:

| Suite | Units | ΔF1 | ΔSHD | Better / worse (F1) |
|---|---|---|---|---|
| Synthetic factor panel, network among the idiosyncratic parts | 53 | +0.233 | −7.5 | 53 / 0 |
| LUCID out-of-distribution suite | 110 | +0.114 | −0.9 | 83 / 8 |
| Expanded LUCID benchmark, 8 held-out families (20 seeds) | 754 | +0.073 | −1.0 | 400 / 123 |

Three held-out families lose even on the pervasive route, though only on a few units:
mixed lags up to 4 (15 units, ΔSHD +2.1), the benchmark's near-unit-root family (18 units,
ΔSHD +0.8; at later seeds the clearer loss, on both F1 and SHD, came from the cell with
weakly persistent factors and own AR 0.3, while the near-unit-root cell gained F1 but was
worse on SHD, +0.5) and heterogeneous factors (27 units, ΔSHD +0.2).

**When to use it.** Turn it on for factor-driven panels (LUCID reports the `pervasive`
route), except where it lost: series driven by weakly persistent factors, especially
near-unit-root ones (in our own stress panels, with persistent factors it still gained),
several lag lengths mixed up to lag 4, and very heterogeneous factors. Results were also
mixed with GARCH factor volatility combined with lag-0 mixing or nonlinear links. On a
separate pre-registered suite of eight new panel families, seven gained in every cell
that LUCID routed to `pervasive`: +0.11 to +0.66 F1 on the lagged edges and SHD lower by
1.4 to 24, with heavy tails, nonlinear links, structural breaks, lagged feedback,
measurement noise (noise variance 20-30 % of the signal's), wider panels (d up to 40) and
many factors. The eighth, near-unit-root series driven by weakly persistent factors, was
neutral on F1 and slightly worse on SHD (+0.3 to +0.85); in development panels of that
kind some single panels lost a lot. That family is why the default stays `"observed"`.
None of the gates we tested on statistics of the observed panel beat always using
factor_adjusted, so there is no automatic switch.

Limitations. The factor proxies are estimated, so confounding through a persistent
factor is removed only approximately. The leftover is largest when a factor loads on few
series and is persistent, and it grows with `T`: at α = 0.05 with an AR(0.8) factor, the
null rejection rate is 0.25 / 0.13 / 0.07 at 4 / 8 / 16 series per factor (plain partial
correlation: 0.33–0.36); with white factors it is 0.05. Read α here as a graph-tuning
parameter rather than a calibrated error rate. Calibrated alternatives (a two-proxy IV
test, an errors-in-variables correction) were tried and lost too much power: the graph
F1 gain over default LUCID disappeared. The test is linear, treats every column as
continuous, and rejects missing values. It makes LUCID about 1.5–3× slower on pervasive
panels: only the skeleton is re-run, since the lag-0 slice is not used.

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
