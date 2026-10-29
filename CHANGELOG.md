# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- textlint-disable -->

## [Unreleased]

### Added

* `corr_table()` — APA-style correlation table (r, stars, M/SD, CI, footnote), prints via `_repr_html_`/`__repr__`.
* `pairs_panel()` — scatterplot matrix (histogram+KDE diagonal, scatter+fit, correlation text), colorblind-safe palette.
* CI tests `gcm`, `wgcm`, `pcm` (residual-score) and `raft`, `rift` (random-feature F-tests).
* How an unoriented (`o-o`) contemporaneous edge is rendered is now a post-discovery choice, not a hardcoded one: `result.to_binary(undirected=, conflict=)`, the lossless `result.to_marks()`, `result.plot(undirected=)`, `result.undirected_policy`, and the same knobs on `cdnots_to_tigramite_graph()` / `tigramite_graph_to_binary()`. Engine defaults are unchanged — `cg_tig` is bit-identical. `evaluate_graph()` gains `*_lag0`, `*_lag0_adj` and `*_lagpos` metric groups.
* `run_cdnots(collider_search="full"|"sepset")`: which conditioning sets the lag-0 collider
  vote searches. `run_cdnots_plus(return_prefix=True)` + `finish_cdnots_plus()` reuse one
  skeleton for several `orient_margin` values.
* `run_cdnots_plus(lag0_orient="lingam")`: orients contemporaneous edges that conditional
  independence leaves undirected (`o-o`) or conflicting (`x-x`), which PCMCI+-style search
  cannot do. Each such component's innovations are ordered by DirectLiNGAM, and an edge is
  oriented only when the direction holds in at least 90% of bootstrap re-orderings, one
  endpoint's innovations reject normality (p < 0.01), and it creates no unshielded
  collider the CI step ruled out; Meek then propagates the new arrowheads. Decisions are
  on `result.lag0_orientations`. Off by default. On held-out linear simulations (d 10–30,
  ParCorr) it added +0.02 to +0.04 F1 under non-Gaussian noise when 60% of the true edges
  are contemporaneous (+0.002 to +0.017 with a 0.3 share), and changed F1 on Gaussian
  data by at most 0.0003 on average (0.0008 in any single cell). It assumes linear
  mechanisms: on spline mechanisms with Laplace or Gaussian noise most of its orientations
  were wrong (small F1 cost, SHD up to +0.75), though with uniform noise it still helped,
  and on CausalRivers it left F1 unchanged while raising SHD by about 0.5. Discrete
  columns are never oriented.
* `run_cdnots_plus(addback_alpha=..., veto_alpha=..., alpha_mci=...)`: grow-then-prune
  refinement of lagged links. After orientation, every absent lagged cross link is
  re-tested with the MCI conditioning set (parents of both endpoints, the cause's shifted
  by its lag, plus each endpoint's next lag) and added when `p < addback_alpha`; every
  lagged link of the grown graph is then re-tested and removed when `p > veto_alpha`.
  Phase 1 drops most missed lagged links at depth 0, where the target's own past masks
  them. The graph is then re-oriented from the refined skeleton, repeating until a pass
  changes nothing; a link changes together with its lag copies, within background
  knowledge and ancestral constraints, and a test with too few complete rows changes
  nothing. `alpha_mci` sets the phase 2 MCI level apart from `alpha`. Changes are on
  `result.lag_refinements`, their p-values in `pvalue_matrix`. On held-out linear
  simulations (ParCorr, d 10–50, T 100–1000, Gaussian and Laplace noise),
  `addback_alpha=1e-4, veto_alpha=0.001` at `alpha=0.01`, against the plain run at the
  same alpha, lowered SHD (by 0.1–5) in every setting and raised F1 by 0.007–0.05, except
  at T=100 with d=50 (F1 -0.03). Against CDNOTS+ at its best single alpha it raised F1 in
  9 of 12 settings (not at T=100, d=50: -0.03; nor at T=1000 with d=10 or 50: -0.002
  and -0.006) and lowered SHD in 11 of 12 (+1.5 at T=1000, d=50). A strict `alpha_mci` is
  not needed at `alpha=0.01`. On nonlinear simulations (spline / MLP, d 10–20, T 300–1000)
  with `raft` and the same alphas it was about neutral (F1 within 0.03). It needs a CI
  test with power at the sample size, since the veto drops links the test cannot detect
  (`splitkci` tests on half the rows, so at T=300 it lost up to 0.12 F1; about neutral at
  T=1000), and p-values that track effect size: on CausalRivers the add-back left F1
  unchanged while raising SHD by 2. All default `None`: results unchanged.
* `majority_gamma` on `run_cdnots`, `run_cdnots_plus`, `cdnots_discovery` and `uc_sepset`:
  an ambiguity band for the majority collider rule (Zhalama, Zhang & Mayer 2017). With
  share `f` of separating sets containing the middle node, a triple is a collider if
  `f < 0.5` and `f <= majority_gamma`, a non-collider if `f > 0.5` and
  `f >= 1 - majority_gamma`, and ambiguous (left for Meek not to orient through)
  otherwise. The default 0.5 is the plain majority rule, so results are unchanged; 0
  orients only unanimous votes, as conservative PC does. On a CDNOTS+ / ParCorr
  simulation grid `0.4` gave F1 +0.003 and SHD −0.35 on held-out seeds; it is untested on
  other CI tests and on plain CDNOTS.

### Changed

* `sigkci` / `SigKCIGPU` renamed to `linsig` / `LinSigCIGPU` (old names deprecated). Results
  change: kernel-scaled ridge (`ridge`, replaces `lamda`), no spurious lag-2 self-loops.
* `RCOTGPU` defaults: `num_f2=min(25, floor(sqrt(T)))`, `n_avg=1`, `seed=42`
  (was `25, 5, None`).
* `ParCorrGPU.__call__` now reads partial correlation off the cached covariance
  matrix under `method="precision"` (default) instead of solving OLS every call —
  5.8x faster. Only benefits scalar callers (Cedar, `run_cdnots(stable=False)`);
  default `run_cdnots` gets no speedup (it already used the same cache via
  `batch_test`), but its statistics shift by ~1e-8 from the float32->float64
  covariance, and any persisted `pvalue_cache` misses across the upgrade.
  Ill-conditioned tests fall back to the unchanged `method="ols"` path.
* `discover -a cdnots+` defaults `--alpha` to 0.01, matching `run_cdnots_plus`.
* Keyword-only: `orient_margin` in `cdnots_discovery` / `run_cdnots`, `legacy_mci_conds` and
  `orient_margin` in `run_cdnots_plus`, and `legacy_mci_conds` in `mci_skeleton`, along with
  the new options. Passing them positionally now raises `TypeError`.
* Results change (fixes below): CDNOTS and CDNOTS+ lag-0 orientations; with
  `include_C=True`, CDNOTS+ C-node edges and lagged edges whose MCI tests conditioned on C;
  lagged edges under `lag_verify=True`.

### Fixed

* `CIT_Base.set_data` wrote buffered cache rows into the new data's cache file.
* `RCOTGPU` p-values depended on the order of the conditioning set.
* `ParCorrGPU.batch_test` (precision method) now runs the same
  reciprocal-condition guard the scalar path got in #73, instead of inverting
  every sub-covariance blindly and returning a near-random statistic for
  ill-conditioned tests (issue #74). Declined tests fall back to the same OLS
  computation `__call__` uses, so both entry points return identical results.
  That fallback now uses the cache key's column order and is clipped to
  [-1, 1]; previously, reorderings of one declined test could differ by up to
  0.06 in the statistic, which could also exceed 1 by float32 rounding. On
  MPS, which lacks `lstsq`, the fallback runs on CPU instead of raising.
  Well-conditioned results are unchanged up to rounding. The persisted p-value
  cache key changes, so caches written by earlier builds are recomputed.
* `examples/regime_discovery.ipynb` now seeds each synthetic dataset independently
  (`np.random.default_rng`), reports detector comparisons as mean ± std across
  multiple seeds instead of a single misleading run, degrades gracefully when the
  optional `hmmlearn` dependency is absent, and documents PELT's `signal="volatility"`
  option (it is not limited to mean shifts).
* `sensitivity_analysis()` reports why it cannot run instead of raising (`reason`, `stats`).
* `routed_deconfound(..., keep_undirected=True, pervasive_base="tetrad")` works instead of raising `NotImplementedError`.
* CDNOTS / CDNOTS+ lag-0 orientation: Meek R1 orients from lagged parents and runs to
  closure; collider votes condition on correctly shifted lagged parents (plain CDNOTS also
  on lagged neighbours); ambiguous triples and conflicting orientations stay undirected;
  colliders that would close a lag-0 cycle are re-selected.
* `run_cdnots_plus(include_C=True)` never tested C's lag-0 edges, so every variable came
  out C-adjacent.
* C-node handling: `knowledge=` with several C columns, forbidden `C -> X`, `lag_verify`
  ignoring C, `select_features(from_result=...)` `IndexError` on results with a C edge.
* Orientation guards: lag-copied orientations respect the skeleton and knowledge; the
  C-node phase respects knowledge and acyclicity; cycle checks use the live graph.
* Discrete-column indices were misaligned with `include_C=True` (`StratifiedCIT`,
  `CMIknnMixedGPU`).
* CDNOTS+ `pvalue_matrix` (`return_pvals=True`) dropped every lagged test and transposed the
  lag-0 ones: its raw `pvals` records were `(cause, effect, ...)`. They are now
  `(effect, cause, ...)`, as in CDNOTS.
* `select_features(algo="cdnots", markov_blanket=True)` with a custom `c_array` silently
  dropped spouses whose column name equalled the C node's label (e.g. `C3`). Colliding C
  labels now get a `_` prefix (`_C3`).
* `run_cdnots_plus(include_C=True)`: the majority-rule collider step now conditions every
  vote on both endpoints' C parents (phase 2 MCI does so for the effect), instead of
  offering C as an optional subset member. Votes without C tested an endpoint whose
  regime shift swamps the dependence, found spurious separating sets and declared false
  colliders, which showed up as conflicting (`x-x`) lag-0 edges at nonstationary
  variables. With a C-driven mean shift on 20% of the variables (linear Gaussian, d 10–20,
  alpha 0.01) correctly oriented lag-0 edges rose by 2–7% and F1 by +0.004 to +0.016 for a
  strong shift, ~0 for a weak one. Outputs with `include_C=False`, or where no variable is
  C-adjacent, are unchanged.
* `run_cdnots_plus(max_conds_py=..., max_conds_px=...)` now keeps the *strongest* lagged
  parents, as PCMCI does. Phase 1 ranks each variable's parents by test strength, but
  phase 2 rebuilt the lists from graph neighbours in node order, so the caps kept the
  lowest-indexed parents instead. The same lists condition the majority-rule collider
  votes, so capped runs can change orientations as well as the skeleton. Uncapped runs
  (the default) are unchanged.
* `run_cdnots_plus` with a gapped `lag_list` (e.g. `[2, 4]`): the MCI phase read a dense
  `0..2*len(lag_list)` embedding with different rows from the phase-1 window, so tested
  and conditioning columns sat at the wrong lags, and parents were shifted by slice
  position instead of lag. MCI and the collider step now read every sum of two window
  lags, aligned with the window. On 20 simulated series with true lags 2 and 4, lagged F1
  rose from 0.46 to 0.96. Contiguous lags are unchanged.
* CDNOTS+ phase 2 (MCI) and the plain CDNOTS skeleton no longer CI-test edges that
  background knowledge requires; they ran the tests and discarded the result, as CDNOTS+
  phase 1 already avoided. A required neighbour now ranks first as a conditioning
  candidate, as in phase 1, so outputs with required edges can change slightly; without
  background knowledge they are unchanged. Required edges get no MCI p-value
  (`pvalue_matrix` NaN, as from phase 1). `lag_verify=True` no longer removes a required
  lagged edge. In every skeleton phase, an edge whose time-shifted copy is required (e.g.
  `make_background_knowledge(required=[("A", "B", 2, 1)])`) now counts as required:
  removing an edge also removes its copies, so the required copy used to be lost.
* The CDNOTS / CDNOTS+ collider step no longer takes a C column as the middle of a
  collider. C has no parents, so `X -> C <- Y` was always wrong: `priority=1` turned the C
  edges into conflicts (`x-x` in `to_marks()`) and `priority=0` reversed them. `cg_tig`
  was already rendered as C -> X; the CPDAG and its marks now are too. `uc_sepset` gains
  `parentless=` for such nodes.
* CDNOTS graph rendering (`cg_tig`, `to_binary`, `to_marks`) covers every endpoint state.
  Lagged cells read only a bare "1": a lagged `o-o` leaked into the binary graph as -1, and
  `to_marks()` showed a lagged `x-x` as `-->`. `to_marks()` now reports lagged `o-o` /
  `x-x` (and an arrowhead against time order as `x-x`), and the binary graph renders any
  lagged adjacency as the forward link, since time order fixes its direction. A
  conflicting C edge rendered as symmetric 1s (including the impossible X -> C) even with
  `conflict="drop"`. C has no parents, so an arrowhead into C (that conflict, or an
  X -> C from e.g. a `priority=0` collider overwrite) is `x-x` in `to_marks()` and
  C -> X under `"drop"`. With `priority=1` this can change `cg_tig` on normal runs with C.
  `tigramite_graph_to_binary` renders a lagged `o-o` / `x-x` as the forward link and no
  longer writes its mirrored cell, a different link.
* Forbidden-ancestor cleanup (`AncestralKnowledge.add_forbidden_ancestor`) no longer
  removes an edge that background knowledge requires, or one a satisfied required ancestor
  (`add_required_ancestor`) depends on. With `A -> Y -> D`, `Y -> D` required and `A`
  forbidden as an ancestor of `D`, it removed `Y -> D` and reported no violation; it now
  cuts the path at its last edge that is neither required nor needed by a satisfied
  required ancestor (here `A -> Y`, with its lag copies), or reports the violation when
  there is no such edge. When the cleanup still leaves a forbidden ancestor violated
  because of competing required edges or ancestors, it now searches for the fewest edges
  to cut (up to 4, each with its lag copies if possible, from the graph before its own
  cuts) that satisfy every forbidden and satisfied required ancestor, and uses them if
  found. With `A -> B, A -> C, B -> C, B -> D, C -> E, D -> E`, `A` required and `B`
  forbidden as ancestors of `E`, it cut `C -> E` and reported `B => E` violated; it now
  cuts `B -> C` and `B -> D`. Results where the cleanup succeeds are unchanged.

## [0.27.0]

### Added

* **LUCID** (`causalts.confounders`) — regime-adaptive deconfounding for causal discovery
  under latent confounders. `run_lucid(df, max_lag)` diagnoses whether latent confounding
  is sparse or pervasive from the residual spectrum (against a Marchenko–Pastur no-factor
  null) and applies the matching correction, returning a `LucidResult`. Every
  `CausalResult` now also exposes `.deconfound()`, `.tetrad_filter()`, and `.pds_filter()`
  so LUCID (or a fixed-strategy comparator) can be applied to a graph already discovered
  with any algorithm, without re-running the skeleton search. See the new
  [Unobserved Confounders (LUCID)](examples/latent_confounder_detection) tutorial and the
  `causalts.confounders` API page.
* **DoWhy effects bridge.** `validate_transition_graph()` and `history_sufficiency()`
  validate a lag-embedded graph (replacing the deprecated `falsify_graph()`);
  `refute_effect()` and `sensitivity_analysis()` stress-test an estimate;
  `build_identification_artifacts()` / `build_transition_artifacts()` hand back a graph
  and the frame it belongs with, for driving DoWhy directly. All are on the result
  objects, and `causal-ts dowhy validate --test` exposes the validators from the CLI.
* `corrplot(..., diag="glyph")` — renders the diagonal as an ordinary cell,
  using the same `method` and colormap as the rest of the matrix. Intended for
  *directed* matrices (a cause→effect adjacency or an edge-stability matrix),
  where the diagonal is real data such as a self-loop rather than the trivial
  1.0 of a correlation matrix. Significance markers and confidence-interval
  overlays are still skipped on the diagonal.

### Fixed

* **Breaking.** `estimate_effect()` reported an unadjusted regression as an identified
  causal effect — a lagged treatment had no parents, so no backdoor path was found. The
  graph is now unrolled and the adjustment set verified against the fitted estimator.
  ATEs change.
* **The GES, LGES and TGES baselines were substantially understated.** `ges_discovery`
  read causal-learn's adjacency matrix with the endpoints transposed, and the vendored
  GES search behind `lges_discovery` / `tges_discovery` had a broken CPDAG construction
  that stopped LGES far short of convergence. Both are fixed and now apply temporal
  background knowledge; F1 on `baseline_comparison` moves from 0.125–0.522 to 0.636–0.949.
  Any prior comparison against these baselines understates them.
* `refute_effect()` no longer reports an uncomputable refutation as a failure, and
  `refute_structure()` returns `p_value` and `adjusted_p_value` separately. DoWhy's
  renamed identification APIs are bound by capability, not version.
* `causal-ts ci-test-info --test <name>` now shows only the selected test's
  summary instead of the full guide. Registered tests without a guide section
  report a clear error; the default and `--test all` output are unchanged.
* The CI test selection guide now covers all registered tests. `parcorr`,
  `cmiknn`, `cmiknn-mixed-gpu`, `fisherz`, `chisq` and `gsquared` had no entry,
  so `ci-test-info --test <name>` failed for them. The `cmiknn-gpu` entry also
  now notes that its permutation null stops early.
* `priority=3` and `priority=4` (collider strength ordering) now raise `ValueError`
  at the entry point, instead of failing with an `AttributeError` from inside
  causal-learn after the skeleton search. They score each conflict over the full
  powerset of the endpoints' neighbours — exponential in node degree — and were
  never wired to a CI test. Use `priority=1` (abstain) or `2` (keep first).
* `detect_subsampling` raises `ValueError` on a 1-D or single-variable input, instead of
  failing inside NumPy with `LinAlgError: 0-dimensional array given`.
* The generator in the *Unobserved Confounders (LUCID)* tutorial injected its true lag-1
  edges after the recursion instead of inside it, so the simulated data did not match the
  ground truth the notebook scored against.
* `corrplot` dropped the right and bottom edges of its grid border. All axes
  spines are hidden, and the border was drawn with `axhline`/`axvline` at
  exactly the axis limits, so half of each boundary line fell outside the clip
  box. The border is now an unclipped rectangle and all four edges render.
* `corrplot(..., colorbar=False)` was ignored for the colour-only glyph methods
  (`"color"`, `"shade"`), which silently overrode an explicit argument and
  forced callers to use `cl_pos="n"` instead. `colorbar=False` now suppresses
  the colorbar for every method.
* Documentation: the "New in v0.26" banner linked to the example notebook with a
  path relative to the site root, but the banner renders on every page — from
  anything below the root (`examples/`, `api/`, `getting_started/`) it resolved to
  a nonexistent nested path and 404'd. The link is now resolved per page.

### Changed

* **`run_cdnots_plus` defaults changed — this can change results for existing
  callers that don't pin these parameters explicitly.** `run_cdnots_plus` now
  follows PCMCI+'s conventions more closely: colliders that conflict during
  orientation are abstained on rather than tie-broken (`priority=1`, was `2`),
  MCI conditioning excludes only the exact tested lag rather than every lag of
  a variable (`legacy_mci_conds=False`), the nonstationarity sink search stops
  when candidates aren't clearly separated rather than always committing
  (`orient_margin=0.1`, was unconditional), and the default significance level
  is `alpha=0.01` (was `0.05`), matching PCMCI+ and empirically better for
  CDNOTS+ (plain `run_cdnots` is unaffected and keeps `alpha=0.05`). In
  aggregate these changes track PCMCI+ much more closely on stationary and
  latently-confounded data while widening CDNOTS+'s advantage over plain
  CDNOTS where nonstationarity is real. Pin the old values explicitly
  (`priority=2, legacy_mci_conds=True, orient_margin=0.0, alpha=0.05`) to
  reproduce prior behavior.
* `causalts.ci_tests.SigKCIGPU` no longer uses the optional `sigkernel`
  package: it crashed with a buffer dtype mismatch whenever `sigkernel` was
  installed, and measured slower than the existing pure-torch fallback at the
  path lengths this test uses. `SigKCIGPU` now always uses that fallback.

* The `guided_discovery` example notebook is now `agentic_discovery` ("Agentic
  Causal Discovery"), naming it after the `causal-ts-discovery` agent skill whose
  workflow it walks through. The old `examples/guided_discovery.html` URL is gone;
  the page is at `examples/agentic_discovery.html`.

## [0.26.0]

### Added

* `causal-ts-discovery` agent skill (Claude Code / OpenAI Codex) shipped in the package for model-driven causal discovery
* `causal-ts inspect <file>` — data-health facts plus an algorithm / CI-test / C-preset recommendation and a cost class (JSON)
* `causal-ts discover --json` — echoes the run summary, a named edge list, and a `diagnostics` block (density, self-loops, contemporaneous/lagged counts, hub in-degree, `empty` / `saturated` flags) to stdout
* `causal-ts discover --validate` (with `--n-bootstrap` / `--window-frac`) — temporal-bootstrap edge-stability check
* `causal-ts discover --pvalues` — opt-in p-value matrix in the output (off by default as a high-dimensional memory safeguard)
* `causal-ts install-skill` — install the skill into `~/.claude/skills` and `~/.agents/skills`
* `.claude-plugin/` manifests — the skill can also be added as a Claude Code plugin from a checkout
* `include_C` and `c_preset` for GRACE — `run_cdnots_gated`, `run_stability_selection`, `run_ci_skeleton`, and `causal-ts discover --algorithm grace/grace-ss` now let you configure the C node of the skeleton GRACE refines, rather than always building one with the `linear` basis. `include_C` defaults to `True` (the previous behaviour); multi-column presets such as `linear+sin` are supported end to end.
* Public Python entry points `inspect_df`, `recommend_config`, `discover_df`
* Observed-data (non)stationarity detection (ADF + KPSS) and trend-form detection for C-preset selection
* Multi-format data reader (csv / parquet / feather) with an optional `parquet` extra
* `guided_discovery` example notebook — the inspect → recommend → discover → interpret workflow end to end (renamed to `agentic_discovery` after release; see Unreleased)
* Causal feature selection — `select_features` for O(d) single-target discovery (`causalts.feature_selection`)
* Correlation / association plots — `corrplot` and `compute_association_matrix` (`causalts.plotting`)
* Temporal-subsampling detection — `detect_subsampling` and `DetectionResult` (`causalts.utils`)

### Fixed

* `causal-ts discover --algorithm cdnots+` was an accepted choice with no dispatch branch — it produced no graph and exited successfully. CDNOTS+ now runs (and honours `--impute`).
* `--c-preset` was never forwarded to `run_cdnots`, so every CDNOTS run silently used the default `linear` basis regardless of the flag.
* `--no-c` had no effect on GRACE, whose skeleton always included a C node.
* `discover_df` silently dropped `alpha` for CEDAR and dropped `alpha`, `include_C`, `c_preset`, and `ci_test` for GRACE. Everything it can honour is now forwarded, and anything it cannot raises instead of being ignored. `alpha` now defaults to `None`, meaning "use the algorithm's own default", so CEDAR keeps its 0.01 thresholds.
* `discover --validate` ran its bootstrap windows without the main run's `--impute` settings, so persistence was measured under a different configuration than the graph it annotated.
* GRACE high-dimensional memory usage in `run_cdnots_gated`
* Documentation: the README CI-test table listed `linsig`, which the package does not ship — the path-space test is `sigkci`
* Documentation: the GRACE examples in the algorithms guide unpacked a tuple, but `run_cdnots_gated` / `run_stability_selection` return a `GraceResult`; copying them raised `TypeError`
* Documentation: absolute-value notation in two docstrings was parsed as an RST substitution reference, producing errors in the docs build

## [0.25.2]

### Fixed

* Documentation build: aliased the `importlib.metadata.version` import in
  `docs/conf.py` so it no longer shadows Sphinx's `version` config (had failed
  the Read the Docs build with a `TypeError` in the inventory dump)
* Restored the API Reference example notebook link (moved the symlink to
  `docs/examples/api_reference.ipynb`)
* Version switcher now reports v0.25.2

### Added

* CDNOTS algorithm — constraint-based causal discovery with nonstationarity-based orientation
* CDNOTS+ algorithm — PCMCI+-style two-phase skeleton for improved precision on dense graphs
* CEDAR algorithm — scalable pairwise discovery with automatic lag selection
* GRACE algorithm — hybrid neural gates with L0 regularization for high-dimensional settings
* 8 GPU-accelerated conditional independence tests (KCI, SplitKCI, DFCIT, SigKCI, RCOT, ParCorr, CMIknn, GCMI) plus StratifiedCIT and CMIknnMixedGPU for mixed data
* CLI (`causal-ts`) with `discover`, `generate`, `evaluate`, and `plot` commands
* DoWhy integration for causal effect estimation, counterfactuals, and root-cause analysis
* Comprehensive documentation and example notebooks
* Synthetic data generators (nonstationary, mixed discrete-continuous, Lorenz-96)


[unreleased]: https://github.com/bloomberg/causal-ts/compare/v0.27.0...HEAD
[0.27.0]: https://github.com/bloomberg/causal-ts/compare/v0.26.0...v0.27.0
[0.26.0]: https://github.com/bloomberg/causal-ts/compare/v0.25.2...v0.26.0
[0.25.2]: https://github.com/bloomberg/causal-ts/compare/v0.25.1...v0.25.2

<!-- textlint-enable -->
