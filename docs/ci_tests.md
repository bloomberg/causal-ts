# CI Test Selection Guide

All CI tests share a common interface; the GPU tests run on CUDA, MPS, or CPU (`raft`
and `rift` are CPU-only NumPy implementations). Run `causal-ts ci-test-info` for a terminal summary.

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

**`linsig`**

Linear-signature kernel on path space. Built for SDEs and diffusions.

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
| `linsig` | Signature (linear) | {fas}`gauge;sd-text-primary` moderate | SDEs / stochastic processes, fast |
| `dfcit` | Distribution-free | {fas}`gauge;sd-text-primary` moderate | High recall on nonlinear data; handles mixed discrete-continuous |
| `kci` | Kernel | {fas}`snail;sd-text-danger` slow | General nonlinear; uses all samples (no split bias) |
| `cmiknn-gpu` | k-NN | {fas}`snail;sd-text-danger` slow | Nonparametric baseline |
| `gcm` | Residual score | {fas}`forward;sd-text-success` fast | Nonlinear via a regression you choose; analytic p-value |
| `wgcm` | Residual score | {fas}`gauge;sd-text-primary` moderate | Conditional covariance that changes sign across Z |
| `pcm` | Residual score | {fas}`gauge;sd-text-primary` moderate | Conditional-**mean** effects; no weight bank needed |
| `raft` | Random-feature F-test, additive | {fas}`forward;sd-text-success` fast | Additive nonlinear mean effects; cheap nonlinear counterpart of `parcorr` |
| `rift` | Random-feature F-test, interaction-adjusted | {fas}`forward;sd-text-success` fast | As `raft`, but also adjusts for interactions among, and wiggly effects of, conditioning variables |

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
- Sizes its X/Y Fourier features to the sample, `min(25, floor(sqrt(T)))`, so short series do not pay for features they cannot estimate
- Deterministic by default (`seed=42`); `n_avg>1` averages independent RFF draws via harmonic mean (Wilson 2019) at T ≤ 300

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

### `linsig`

Linear-signature-kernel CI for path-valued data: a closed-form linear signature
kernel (Goursat PDE recursion) + the classic KCI conditioning framework (Zhang et
al. 2011) + a gamma/HBE null. Fast and dependency-free; its kernel is always linear.
Formerly `sigkci` / `SigKCIGPU`: the old names still work but are deprecated, since
the test never solved the RBF signature-kernel PDE the name implied.

**Strengths:**
- Principled test for stochastic processes (SDEs, diffusions) — operates on path space
- Captures temporal structure that pointwise tests ignore
- No extra dependencies; deterministic kernel choice

**Limitations:**
- Not designed for standard tabular time-series observations — underperforms vs kernel tests on VAR data
- Computational cost scales with path length
- Cannot adjust for a conditioning variable with constant increments, such as a linear
  trend or CDNOTS's time-index node C: the linear signature kernel does not see it, so a
  test conditioned only on it is the unconditional test and C-adjacency
  (nonstationarity) is not detected

### `gcm`

Generalised covariance measure (Shah & Peters 2020). Regresses X and Y on Z and
tests whether the residual product has non-zero mean.

**Strengths:**

- Very fast; one-dimensional score and an analytic p-value
- Nonlinearity is handled by the nuisance regression, which you choose and can inspect; the default
  (`regressors=("spline_select",)`, shared by `wgcm` and `pcm`) is an additive cubic-spline ridge that
  falls back to linear ridge when that fits better
- Long-run (HAC) variance by default, so serial correlation in the residual product is accounted for

**Limitations:**

- Validity rests on the regressions being accurate — this is an assumption, not a technicality
- Tests a residual moment, not full CI: an edge that changes only the conditional variance of Y is invisible
- Blind to conditional covariance that changes sign across Z — use `wgcm` for that

---

### `wgcm`

Weighted GCM (Scheidegger, Hörrmann & Bühlmann 2022). Multiplies the residual
product by a fixed bank of weights `w(Z)` before averaging, and takes the
calibrated maximum.

**Strengths:**

- Recovers alternatives where GCM cancels to zero because the conditional covariance flips sign
- The weight search is paid for by the multiplier bootstrap rather than ignored

**Limitations:**

- Still a residual-moment test; a wider bank never makes it omnibus
- A larger bank widens the joint null and costs power, which matters most at small T

---

### `pcm`

Projected covariance measure (Lundborg, Kim, Shah & Samworth 2024). Learns
`E[Y|X,Z]` on held-out blocks and tests the projected residual.

**Strengths:**

- Learns the direction of the alternative instead of guessing it, so no weight bank is needed
- Minimax-optimal for the conditional-mean problem (Lundborg et al. 2024)

**Limitations:**

- Tests conditional-**mean** independence only — reported as `inferential_target = "conditional_mean"`. It will not see variance-only or tail-only edges, by construction
- Graph-level consistency needs the extra assumption that edges act through the mean

---

### `raft` and `rift`

Random-feature F-tests: one method with two ways of adjusting for Z. Each tested
variable is expanded on its own into its standardized value plus up to 8 random
Fourier features (up to 4 sine/cosine pairs at random frequencies; an RBF-kernel
approximation; `raft` uses fewer for the tested variables at small T, see below). `X ⫫ Y | Z` is tested by whether X's
features improve a least-squares fit of Y after adjusting for Z, and vice versa
(partial F statistic, one p-value per direction, or per retained response component
when X or Y is a vector; with the bootstrap null both combine them by the Cauchy combination, falling back
to Bonferroni over every direction when one is left untested). The p-value comes from
a dependent wild bootstrap of that statistic under the null (`variance="dwb"`,
default); `raft` also offers the F distribution (`variance="f"`).

- **`raft`** (Random-feature Additive F-Test, `adjust="additive"`): Z enters
  additively. By default (`z_ridge=True`) that is each conditioning variable's
  linear term, unpenalized, plus a ridge-penalized dictionary of random sine/cosine
  features per variable, the penalty chosen by generalized cross-validation for each
  response; with `z_ridge=False`, one random-feature block per conditioning variable
  fitted by least squares.
- **`rift`** (Random-feature Interaction-adjusted F-Test, `adjust="interaction"`):
  `raft`'s adjustment with more terms in its ridge-penalized dictionary: sines and
  cosines of each conditioning variable at frequencies 1, 2, ..., 8 on the
  standardized scale and, with two or more conditioning variables, cubic B-splines
  and their pairwise tensor products (with `z_ridge=False`, as before, these sit on
  top of `raft`'s unpenalized random-feature blocks). The penalty is chosen by generalized cross-validation
  for each response, so strong directions are kept almost unshrunk and weak ones
  shrink towards zero: the adjustment does not spend a degree of freedom per column,
  and with smooth data the extra terms contribute almost nothing. The bootstrap
  pushes its draws through the same ridge smoother.

`raft`'s defaults also make it robust to non-Gaussian noise and keep its power at
small T. It tests the Huber-clipped residual of each response given Z
(`robust="huber"`), so heavy tails and volatility bursts do not dominate the
statistic; it expands each tested variable into one sine/cosine pair per about 100
rows (between one and four; `n_rff_tested="auto"`) while the conditioning variables
keep all `n_rff` features; and it combines the directions actually tested by the
Cauchy combination (`combine="cauchy"`). In pre-release runs on freshly generated
graphs (CDNOTS, d=10, sparse, linear and spline mechanisms, T=50-800) these raised
`raft`'s F1 from 0.50-0.54 to 0.64-0.65 under heteroscedastic and ARCH noise and
from 0.62 to 0.73 under Student-t noise, about level with `rcot` overall (F1
within 0.01, SHD within 0.1) at about 1/60 of its run time; the remaining gap is
heteroscedastic and ARCH noise at T=200 (F1 about 0.05 lower). Type-I on known-CI
triples stayed at most 0.09 at alpha=0.05 (mean 0.05) in those runs. A larger
confirmation on separate graphs (d=5-30, T=50-800, alpha 0.05 and 0.01)
reproduced the gains (F1 +0.11 under heteroscedastic, ARCH and Student-t noise, +0.03
under Gaussian noise, SHD lower) with precision higher in every non-Gaussian
setting, but its known-CI grid found Type-I at alpha=0.05 above 0.10 in 12 of 288
settings (the earlier `raft`: 4), all at T=400 with spline mechanisms and Student-t or ARCH noise
(worst 0.21, against 0.14 for the earlier `raft`; mean 0.050 against 0.034): clipping
heavy-tailed noise makes `raft`'s additive misfit of a spline mechanism easier to
detect as T grows. The earlier `raft` is
`robust=False, n_rff_tested=8, combine="bonferroni"`. The Huber residual has one
cost: with skewed noise whose scale depends on the conditioning variables
non-additively (e.g. on `z1 * z2`), clipping shifts the residual's conditional mean
by an amount that is not additive in Z, and the test turns mildly liberal (rejection
at alpha=0.05 up to about 0.12 at T=300 and 0.17 at T=1000 in those runs, against
0.06 with `robust=False`; `rcot` rejected 0.89-1.00 in the same settings). A
related failure appears when one conditioning variable drives a skewed noise's
scale extremely: with scale exp(z^2 / 2), which gives the noise infinite variance,
the additive adjustment cannot track the resulting steep shift, and clipping makes
it worse: rejection reached 0.41 at T=300 and 0.65 at T=1000 (`robust=False` 0.18
and 0.26, `rcot` at most 0.07). Use `robust=False` when skewed noise whose scale
depends on Z non-additively is likely, and prefer `rcot` when the scale depends on
Z extremely strongly. Do not rank-transform the
data first: with two or more conditioning variables the normal-score map makes an
additive conditional mean non-additive, and the test then rejects true
conditional independence (up to about half the time at T=1000 in those runs).

Three further defaults followed (the settings above, without them, are
`z_ridge=False, hc2=False, linear_channel=False`, with the same results and cache
keys as before). A fixed feature basis for Z underfits smooth effects of Z once T
is large, and the misfit reads as dependence: this caused the T=400 failures above.
`z_ridge=True` replaces it by the penalized dictionary, which grows with the data
and costs few degrees of freedom at small T. `hc2=True` rescales the bootstrap
residuals by their leverage under that adjustment. `linear_channel=512` also tests
each direction with only the tested variables' linear terms, at up to 512 usable
rows, and Cauchy-combines the two p-values: a nearly linear dependence is then
tested on one degree of freedom per variable, which recovers power at small T. The
settings, including the 512-row cutoff, were chosen on graphs and known-CI
generators separate from the confirmation. In the confirmation (CDNOTS, d=5-30,
T=50-800, four noise regimes) F1 rose from 0.62 to 0.68 at alpha=0.05 and from
0.55 to 0.61 at alpha=0.01, and SHD fell from 23.2 to 22.3 and from 25.4 to 23.1; the gain is concentrated at T <= 400
(T=50: 0.10 to 0.21; at T=300 0.80, against 0.79 for `rcot`), and overall
`raft` now edges `rcot` at alpha=0.05 (0.679 against 0.670; `rcot` stays ahead at
alpha=0.01, 0.62 against 0.61, from T >= 600). Mean Type-I on known-CI triples stayed
about the same (about 0.049 against 0.050) but no setting exceeded 0.10 (the T=400
spline settings with heavy-tailed or ARCH noise from 0.13 to about 0.06); under strong autocorrelation
(phi=0.9) all tests stay liberal, `raft` about 0.19-0.21 as before. The costs:
precision up to 0.04 lower at T=300-400 at alpha=0.05 (negligible at alpha=0.01),
and about 1.3-1.5 times the run time per graph.

`rift` now shares all of `raft`'s defaults (Huber residual, tested-side feature
count, Cauchy combination, `z_ridge`, `hc2`, `linear_channel=512`); its earlier
defaults are `robust=False, n_rff_tested=8, combine="bonferroni", z_ridge=False,
hc2=False, linear_channel=False`, with the same results and cache keys as before.
On tuning graphs (CDNOTS, d=10, linear and spline mechanisms, sparse and dense, with
and without interaction terms, T=100-800; no separate confirmation yet) this raised
`rift`'s F1 from 0.60 to 0.73 at alpha=0.05 and from 0.51 to 0.65 at alpha=0.01,
SHD from 18.5 to 14.6, level with the new `raft`, at about twice `raft`'s run time
per graph. On interaction-null triples its calibration stayed where it was:
rejection at alpha=0.05 with shared smooth interactions 0.07-0.08 at T=100-400
(earlier 0.06-0.10; `raft` 0.96-1.00), shared rough interactions 0.17-0.20 (earlier
0.16-0.21), with power at T=100 up from 0.58 to 0.67. In those discovery benchmarks
the new `rift` was no better than the new `raft`, even on graphs with interaction
terms: its value is calibration when X and Y share interactions of conditioning
variables.

**Strengths:**

- Cheap: a single cold `raft` call costs about 2-10 ms from T=250 to T=1000 with
  3-6 conditioning variables (1-5 ms with `z_ridge=False`); within a discovery run
  most of the conditioning-set work is cached and reused. `rift` costs about
  6-28 ms per cold call there (4.7-21 ms with its earlier defaults, measured on
  the same machine; the earlier 3-15 ms figure came from another), and is
  several times cheaper than `kci` at T=1000. Cost grows
  about linearly in T
- Both hold their level under heteroscedastic, heavy-tailed and serially correlated
  errors on known-CI triples with additive mechanisms (rejection at alpha=0.05: at
  most about 0.06 across 108 null settings in pre-release runs, at alpha=0.001 at
  most 0.004; the bootstrap multipliers and features are fixed per seed, so a
  single run's per-setting rates move by about 0.02 with `rff_seed`)
- `rift` also holds its level when X and Y depend on a wiggly function of a
  conditioning variable, which defeats fixed-basis tests: for X = sin(4Z) + e,
  Y = tanh(2 sin(4Z)) + e (T=500, i.i.d.) rejection at alpha=0.05 was about 0.05
  (0.07-0.09 with its earlier defaults),
  against about 0.95 for `rcot` and for `raft` with `z_ridge=False` (`raft`'s
  default about 0.37), 0.41 for `kci` and 0.16 for `gcm`, with
  no loss of power at T=50-200 in pre-release runs
- `rift` holds its level far better when X and Y depend on smooth interactions of
  conditioning variables (figures in this item are for its earlier defaults; see
  above for the current ones), where `raft` (and other additive-nuisance tests) reject true
  conditional independence almost always: on autocorrelated triples with pure
  smooth interaction terms (gated and tanh products) its mean rejection at
  alpha=0.05 was about 0.04 at T=100 and 0.07 at T=400, against 0.46 and 0.77 for
  `raft`, with the same power as `raft` at T=400

**Limitations:**

- `raft`'s model of Z is **additive and smooth**. When X and Y both depend on an
  interaction of conditioning variables (e.g. on `z1 * z2`), it rejects true
  conditional independence almost always, and on non-additive systems its false
  positives grow with T. Wiggly marginal effects are only partly absorbed: for a
  shared `sin(4z)` rejection was about 0.37 at T=500 (0.98 with `z_ridge=False`),
  for `sin(6.5z)` essentially always. Use `rift` there
- `rift` absorbs smooth interactions well, but rough interactions (products of
  wiggly functions shared by X and Y) remain partly unexplained: on autocorrelated
  triples with shared spline products its mean rejection at alpha=0.05 was about
  0.09 at T=100 and 0.12 at T=400 (worst settings 0.21-0.25; `raft` 0.38-0.59), and
  in the hardest stress settings 0.19-0.23 (about 0.03 at alpha=0.001). The
  frequency terms cause part of this: with `freqs=()` the hardest settings drop to
  about 0.12, but the sin(4Z) case above returns to about 0.95. For shared effects
  sin(fz) with f from 2.5 to 7.5 its rejection at alpha=0.05 was 0.05-0.07 up to
  f=6.5 and 0.14 at f=7.5 (T=500; earlier defaults 0.03-0.11); effects beyond
  frequency 8 are only partly absorbed. With many conditioning
  variables the pairwise dictionary is compressed by a random sketch
- With its earlier defaults, `rift`'s extra adjustment cost recall on strongly
  coupled systems (not yet re-measured with the current ones): on Lorenz-96
  (d=20, T=500, PCMCI+, 3 seeds) its F1 was 0.80 (recall 0.68, SHD 26.7) against
  0.89 for `raft` (recall 0.81, SHD 15.3); on spline and Lorenz-96-map systems the
  two were within 0.01 F1
- Both test conditional-**mean** dependence (`inferential_target =
  "conditional_mean"`); variance-only edges, and interactions of X with Z that have
  no main effect, are outside their target
- `raft`'s `variance="f"` is about twice as fast but badly liberal when the error
  scale depends on Z
- The test needs about 10 residual degrees of freedom beyond the adjustment for Z.
  With `z_ridge` (default) each conditioning variable adds one unpenalized column
  and the penalized dictionary costs only its effective degrees of freedom, so it
  saturates late (from about |Z| = 37 at T=100, beyond 57 at T=200); with
  `z_ridge=False` each adds `n_rff + 1` columns (9 by default) and it saturates
  from about |Z| = 10 at T=100 and |Z| = 21 at T=200. The exact cutoff depends on
  the data. When every regression direction is
  skipped it returns p = 1, which a constraint-based search reads as independence;
  when only some are, the p-value comes from the remaining ones. Each case raises a
  `RuntimeWarning` once per test object; such tests are counted in `n_saturated`
  and not cached. Lower `n_rff` for such runs (with `z_ridge=False`)
- The bootstrap keeps a T x `n_boot` multiplier matrix (about 1.6 kB per row at the
  default `n_boot=199`, so about 16 MB at T=10,000, and about twice that while
  it is built; each test adds temporaries of the same size); lower `n_boot` for very
  long series

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
