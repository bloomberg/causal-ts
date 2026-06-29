# CI Test Selection Guide

All CI tests share a common interface and run on CUDA, MPS, or CPU. Run `causal-ts ci-test-info` for a terminal summary.

---

## Quick Pick

::::{grid} 3
:gutter: 2

:::{grid-item-card} {fas}`bolt;1em;sd-text-warning` Linear data
:shadow: sm
:class-card: ct-quickpick

**`parcorr-gpu`**

Analytic p-value, near-instant. Works at any T and any d.

{bdg-success}`recommended`
:::

:::{grid-item-card} {fas}`wave-square;1em;sd-text-primary` Nonlinear · sufficient T
:shadow: sm
:class-card: ct-quickpick

**`splitkci`**

Good speed/F1 tradeoff. 10–20× faster than KCI. Performance improves with larger T.

{bdg-primary}`balanced`
:::

:::{grid-item-card} {fas}`magnifying-glass-chart;1em;sd-text-primary` Nonlinear · small T
:shadow: sm
:class-card: ct-quickpick

**`dfcit`**

Strong recall on nonlinear data at limited sample sizes. Results vary by graph structure.

{bdg-primary}`high recall`
:::

:::{grid-item-card} {fas}`network-wired;1em;sd-text-danger` General nonlinear
:shadow: sm
:class-card: ct-quickpick

**`kci`**

Uses all T points — no split bias. Most general kernel test.

{bdg-warning}`robust`
:::

:::{grid-item-card} {fas}`timeline;1em;sd-text-primary` Stochastic processes
:shadow: sm
:class-card: ct-quickpick

**`sigkci`**

Signature kernel on path space. Built for SDEs and diffusions.

{bdg-secondary}`path-space`
:::

:::{grid-item-card} {fas}`gauge-high;1em;sd-text-success` Runtime-constrained
:shadow: sm
:class-card: ct-quickpick

**`parcorr-gpu`** or **`gcmi`**

Fastest options. `parcorr-gpu` is instant (analytic); `gcmi` adds rank-normalisation overhead but handles mild nonlinearity.

{bdg-success}`fast`
:::

::::

:::{tip} Not sure what your data looks like?
Use `check_linearity(df)` from `causalts.utils.linearity` to run the Ramsey RESET test across all variable pairs. If no significant nonlinearity is found, `parcorr-gpu` is optimal.
:::

---

## All Tests Overview

| Test | Type | Speed | Best for |
|------|------|:-----:|----------|
| `parcorr-gpu` | Linear | {fas}`bolt;sd-text-warning` instant | Linear data, any T |
| `gcmi` | Monotone nonlinear | {fas}`forward;sd-text-success` fast | Mild nonlinearity, skewed data |
| `splitkci` | Kernel | {fas}`gauge;sd-text-primary` moderate | Nonlinear, larger T |
| `rcot` | RFF | {fas}`gauge;sd-text-primary` moderate | Any T; fast nonlinear option |
| `sigkci` | Signature | {fas}`gauge;sd-text-primary` moderate | SDEs / stochastic processes |
| `dfcit` | Distribution-free | {fas}`gauge;sd-text-primary` moderate | High recall on nonlinear data; handles mixed discrete-continuous |
| `kci` | Kernel | {fas}`snail;sd-text-danger` slow | General nonlinear; uses all samples (no split bias) |
| `cmiknn-gpu` | k-NN | {fas}`snail;sd-text-danger` slow | Nonparametric baseline |

---

## Test Details

:::{note}
For benchmark results across datasets, sample sizes, and algorithms (CDNOTS vs CDNOTS+), see the [CI Test Comparison notebook](examples/ci_test_comparison). F1 varies substantially with algorithm choice and graph structure.
:::

### `parcorr-gpu`
Linear partial correlation with analytic p-value.

**Strengths:**
- Near-instant: analytic p-value, no permutations or kernel evaluations
- Scales to any T and any d with no degradation

**Limitations:**
- Assumes linear relationships — misses nonlinear edges
- Run `check_linearity(df)` first to verify the assumption

### `gcmi`
Gaussian Copula + Partial Correlation: rank-normalises variables then runs partial correlation.

**Strengths:**
- Fast analytic p-value (slower than `parcorr-gpu` due to rank-normalisation overhead)
- Handles mild nonlinearity and skewed marginals via rank normalisation

**Limitations:**
- Not a true MI estimator — only detects **monotone nonlinear** dependencies
- Misses non-monotone relationships (e.g. U-shaped, oscillatory)

### `kci`
Kernel CI test with RBF kernel and HBE p-value, using all T points.

**Strengths:**
- Uses all T points (no train/test split) — avoids split bias
- No assumption about functional form

**Limitations:**
- O(T³) — becomes prohibitively slow at large T
- Bandwidth selection via median heuristic can fail at very small T

### `splitkci`
Bias-controlled kernel CI: splits data into train/test halves, uses gamma p-value.

**Strengths:**
- 10–20× faster than `kci` with similar power on nonlinear data
- Bias-controlled: train/test split prevents distributional leakage

**Limitations:**
- Each test only sees n/2 samples — power drops at small T
- Noisy under near-collinear conditioning because the split amplifies instability

### `dfcit`
Distribution-free CI using empirical CDFs and permutation p-values with early stopping.

**Strengths:**
- Only test that handles mixed discrete-continuous columns natively
- Strong recall on nonlinear data, especially at small T where kernel tests lose power

**Limitations:**
- Permutation-based — slower than analytic tests
- False positive rate can rise under near-collinear conditioning at large T

### `rcot`
Randomized Conditional Correlation via Random Fourier Features.

**Strengths:**
- Near-instant regardless of T — runtime does not scale with sample size
- At T ≤ 300, automatically averages 5 independent RFF draws via harmonic mean (Wilson 2019) to reduce variance

**Limitations:**
- RFF approximation quality depends on bandwidth; can be inconsistent on strongly nonlinear data
- Performance degrades at very large conditioning sets

### `cmiknn-gpu`
GPU-accelerated k-NN conditional mutual information. Nonparametric, no bandwidth selection.

**Strengths:**
- Fully nonparametric: no kernel bandwidth to tune
- Sensitive to a wide range of dependency structures

**Limitations:**
- O(T²) k-NN search — very slow at large T
- k-NN estimator can be biased in high-dimensional conditioning sets

### `sigkci`
Signature kernel CI for path-valued data via PDE-based computation.

**Strengths:**
- Principled test for stochastic processes (SDEs, diffusions) — operates on path space
- Captures temporal structure that pointwise tests ignore

**Limitations:**
- Not designed for standard tabular time-series observations — underperforms vs kernel tests on VAR data
- Computational cost scales with path length and truncation order

---

## Key Tradeoffs

::::{grid} 2
:gutter: 2

:::{grid-item-card} {fas}`expand;1em;sd-text-warning` &nbsp; Large variable sets
:shadow: sm

Large d: kernel tests (`kci`, `splitkci`) slow down as conditioning sets grow.

**Use:** `parcorr-gpu` or `gcmi` — analytic p-values scale well.
:::

:::{grid-item-card} {fas}`feather;1em;sd-text-primary` &nbsp; Weak edges
:shadow: sm

Nonlinear tests may need substantially more data than linear tests to detect low-coefficient edges.

**Use:** `parcorr-gpu` at any T if `check_linearity` shows no nonlinearity.
:::

::::

---

## Result Caching

All CI tests support persistent result caching via `cache_dir`. This can dramatically speed up repeated runs on the same data (e.g. re-running with a different `alpha`, or resuming after a crash).

```python
from causalts.ci_tests import SplitKCIGPU

ci_test = SplitKCIGPU(df.values, device="cpu", cache_dir="/tmp/myrun_splitkci")

# Subsequent runs with the same data reuse cached p-values
ci_test2 = SplitKCIGPU(df.values, device="cpu", cache_dir="/tmp/myrun_splitkci")
```

The cache key is derived from the data hash and test parameters. Caches are safe to share across runs on the same dataset. Delete the directory to invalidate.

---

## Custom CI Tests

You can register your own CI test and use it with all discovery algorithms and the CLI. Subclass `CIT_Base`, implement `__call__`, and decorate with `@register_ci_test`:

```python
import numpy as np
from causalts.ci_tests import register_ci_test
from causalts.ci_tests.cit_test import CIT_Base


@register_ci_test("my_test")
class MyCITest(CIT_Base):
    """Minimal example of a custom CI test."""

    def __init__(self, data=None, **kwargs):
        kwargs.pop("device", None)  # absorb device if not needed
        super().__init__(data=data, method="my_test", **kwargs)

    def __call__(self, X, Y, condition_set=None):
        """Test X _|_ Y | condition_set.

        Parameters
        ----------
        X, Y : int
            Column indices of the two variables.
        condition_set : list of int or None
            Column indices of conditioning variables.

        Returns
        -------
        p_value : float
        test_statistic : float
        """
        if condition_set is None:
            condition_set = []

        x = self.data[:, X]
        y = self.data[:, Y]
        z = self.data[:, condition_set] if condition_set else None

        # --- your test logic here ---
        p_value, statistic = 0.5, 0.0
        return p_value, statistic
```

Once registered, the test is available everywhere:

```python
from causalts.ci_tests import create_ci_test, list_ci_tests
from causalts import run_cdnots

print(list_ci_tests())        # [..., "my_test", ...]
ci = create_ci_test("my_test", data)
p, stat = ci(0, 1, [2, 3])    # use directly

# Or pass to any algorithm
res = run_cdnots(df=df, indep_test=ci, num_lags=3)
```

See `causalts/ci_tests/causallearn_tests.py` for real-world examples wrapping causal-learn CI tests.
