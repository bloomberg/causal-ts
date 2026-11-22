# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- textlint-disable -->

## [Unreleased]

### Added

* `compare_graphs`, `plot_lag_metrics` and `plot_pvalue_distribution` accept
  `undirected=` and `conflict=` for discovered `o-o` and `x-x` marks:
  `"drop"`, `"bidirected"` (the default, so existing results are unchanged) or
  `"adjacency"`, which draws the edge undirected and scores it as a correct
  adjacency with unknown orientation. `plot_lag_metrics` reports separate
  adjacency precision / recall / F1, TP / FP / FN and SHD rows.
* `node_layout="auto"` in `compare_graphs`, `plot_graph` and `compute_node_positions`:
  starts from neato (if Graphviz is installed) and Kamada-Kawai, then moves nodes to cut
  edge crossings and edges passing close to other nodes; works without Graphviz.
* `compare_graphs(show_self_loops=True)` and `plot_graph(show_self_loops=True)` draw
  lagged self-links as loops in the widest gap between each node's edges, as thin as
  missed edges (`self_loop_linewidth`); off by default. `compare_graphs(legend_loc="below")`
  puts the legend under the graph so it never covers an edge.
* `--index-col NAME_OR_POSITION` on `causal-ts inspect`, `discover` and `deconfound`
  reads a column (e.g. a date) as the index instead of as a variable;
  `read_dataframe(path, index_col=...)` does the same in Python (#130).
* `read_dataframe` and the CLI read compressed CSV (`.csv.gz`, `.csv.bz2`, `.csv.zip`,
  `.csv.xz`, `.csv.zst`) and tab-separated `.tsv` files (#128).

### Changed

* More tests: forecasting, plotting, the DoWhy bridge, GRACE, the CLI, the Tigramite
  wrappers and the data generators.

### Fixed

* `causal-ts discover` and `deconfound` given a non-numeric column (e.g. a date) exit
  with a usage error that names it, before any engine runs, instead of a `TypeError`
  from deep inside the engine; `deconfound` reads Booleans as 0/1, as `discover` does
  (#130).
  A `category` column with numeric categories (as parquet and feather keep them) is
  read as numbers; `inspect` used to leave it out as non-numeric.
  `deconfound` stops with a usage error when the data's columns do not match the
  variables the graph was discovered on (e.g. a different `--index-col`).
* `causal-ts inspect`, `discover`, `deconfound`, `evaluate` and `plot` reject a
  `--var-names` list with the wrong count or a duplicate name as a usage error, before
  creating any output, instead of a pandas or plotting traceback (#133).
* `gcmi` no longer returns p = 1 (independent) when the conditioning set has a redundant column, such as `z` and `2*z` or any increasing function of `z`, which are exact copies after the copula transform (#142). The univariate test now takes the partial correlation from the covariance given Z with a pseudo-inverse of the Z block, as the multivariate test does, and its degrees of freedom count the rank of Z. A variable compared with itself is dependent; X or Y determined by Z still gives p = 1; a covariance that is not finite raises `ValueError`. Results cached by earlier builds are not reused.
* `compare_graphs`: `curved_radius=0` no longer hangs when a node pair has two
  strokes, and a pair with three or more strokes no longer draws two on the same arc.
* `CausalForecaster.forecast_horizon(targets=...)` with a subset of variables no longer
  returns NaN from the second step on.
* `CausalForecaster.forecast_horizon` only refuses to run when a fitted model uses a
  lag-0 parent, so the default `include_contemp=True` works with lagged-only graphs
  (exogenous columns are allowed); the forecast index continues the input's
  `DatetimeIndex` (using its frequency, or the inferred one), `PeriodIndex` or
  `RangeIndex` and keeps its name (#144).
* `fisherz` returns p = 1 for a constant column instead of NaN, and ignores a constant
  conditioning column, as `parcorr` does (#134).
* `extract_segments` applies `min_length` to the regime run itself, not to the run plus
  its lag buffer, so a short regime run no longer passes on rows from the previous
  regime (#163).
* `plot_time_series_graph(order=...)` lists variables top to bottom, as documented (also
  `order="auto"`); single-variable graphs plot; `plot_lagfuncs(dpi=...)` is honoured.
* The deprecated `causalts.ci_tests.cit_base` import gives the unified `CIT_Base`; pass its
  arguments by keyword.
* `run_regime_discovery` raises when every segment's `discovery_fn` fails, instead of
  returning an empty graph, and warns when some fail; failures are listed in
  `metadata["failed_segments"]`. With `aggregation="weighted"`, each regime is weighted
  by the samples of its successful segments, so a failed regime does not dilute the vote
  (#136).
* `rcot` / `RCOTGPU` with `approx="perm"` no longer emits a NumPy 2
  `DeprecationWarning`; p-values are unchanged (#137).
* `WrappedGraph.summary()` and `run_cdnots_gated(verbose=True)` print correctly;
  `set_up_logging` cleans up after a failed setup.
* `CausalForecaster` checks the graph shape and the number of `var_names`, with a hint for
  a time-index (C) node; `exclude_C=True` drops it, and `WrappedGraph.forecast()` forecasts
  the observed variables of a graph discovered with `include_C=True` (#145).
* `import causalts.plotting` no longer loads `dcor` and numba; parallel docs builds crashed
  on numba's cache.
* `lorenz96` no longer reseeds NumPy's global random state (it drew only from its own
  `default_rng`); its output for a given seed is unchanged (#129).
* `CondIndTest.run_test_raw(x, y)` with the default `z=None` runs the unconditional test
  instead of raising `TypeError` in the cache-key helper.
* `run_pcmci` and `run_pcmciplus` take tigramite's `link_assumptions`; `selected_links`,
  which tigramite 5.2 removed, raises a clear `ValueError`. The `tigramite` extra needs
  `tigramite>=5.2` (#146).
* `precision_recall_3d` counts the true positives of bool graphs (it returned 0 for
  both) and accepts a graph with only lag 0 instead of raising `IndexError`.
* The `fisherz` CI test's statistic is the partial correlation behind its p-value when
  the conditioning variables have a nonzero mean (it regressed on `Z` without an
  intercept, so value matrices and edge strengths could be far off).
* `iterative_causal_impute` regresses each variable on its Markov blanket at the
  right time offsets (parents and own lags, children as leads, spouses shifted
  accordingly) instead of the same-time values of those variables, fits only on rows
  where the target is observed, and leaves a variable with an empty blanket unchanged
  (it regressed the target on itself). Lags from a stepped `lag_list` are mapped
  correctly (#141).

## [0.28.1]

### Added

* Score-based methods in `causal-ts discover` and `discover_df`: `lges`, `tges`, `ges`,
  `lasso-var`, `granger`, `varlingam` (returns a `BaselineResult`). `run_grace` and
  `run_reap` accept a result as `candidate_graph`.
* `plot_lag_dependence` / `compute_lag_dependence`: per-lag dependence of every pair from
  raw data, optionally parent-adjusted (`adjust="parents"`) and marked against a
  `true_graph`.

### Changed

* `compare_graphs` places its legend at `"best"`; lag-function grids get clearer axis labels
  and `wrap_labels=`.
* CI also runs the core tests on macOS and Windows and reports coverage.

### Fixed

* With statsmodels 0.15, `CausalForecaster`'s fallback model fits its AR candidates again; a removed
  `AutoReg` argument made them fail silently, so only exponential smoothing was chosen.
  `CausalForecaster.fit` now emits a `RuntimeWarning` when every AR candidate fails to fit.
* `CausalForecaster.predict` keeps each prediction on its own row after a missing value
  (later predictions were shifted one row).
* `HMMRegimeDetector` stays within its switch budget; `LeveneRegimeDetector` raises on an
  unknown `method`.
* Plotting: `plot_time_series_graph(order="auto")` handles lag-0-only and 4-D graphs,
  `node_layout="random"` works, and `setup_matrix(lag_array=...)` works with the default
  `x_base`.
* Cedar with `include_C=True` keeps lagged edges whose strongest lag is 0, and MCI pruning
  no longer conditions on the edge's own target (lag-0 pairs can now be `o-o`).
* CPU `parcorr` gives p = 1 for a constant column instead of NaN (#111).
* The CPU `cmiknn` cache key includes every setting that changes the p-value.
* Clear CI-test caches (`cache_dir`) written before this release: they may hold the NaN
  `parcorr` results or `cmiknn` results under the wrong settings.
* `discover --json` / `deconfound --json` print only the JSON to stdout.
* `extract_subgraph` reads lag-0 `<--` / `<-o` marks as reversed edges;
  `tigramite_graph_to_binary` reads `bytes` mark graphs.
* `compare_graphs`, `plot_lag_metrics` and `plot_pvalue_distribution` no longer count `<->`
  as an edge and reject a ground truth with any mark other than `-->` / `<--`.
* No statsmodels `adfuller` `FutureWarning` from `inspect`; no blank lines in CI-test caches
  on Windows; `__version__` is `0+unknown` (not `0.14.0-dev`) in an uninstalled checkout.

## [0.28.0]

### Added

* `run_grace` and `run_reap` refine any directed candidate graph (`candidate_graph=`) or a
  CDNOTS graph: GRACE with L0 Hard Concrete gates, REAP with a spline partial-F
  grow-shrink (`contemporaneous=True`, `effect_floor="adaptive"`); CLI
  `discover --algorithm reap`. New GRACE options: grow rounds (`grow_rounds`,
  `grow_variance`, ...), an F-tested autoregressive head (`ar_path`),
  `gate_threshold="auto" | "li"`, `lambda_rule`, `effect_floor`, `data_transform`,
  `noise_model`, with matching `GraceResult` fields.
* CI tests `gcm`, `wgcm`, `pcm` (residual score) and `raft`, `rift` (random-feature
  F-tests); `order_independent=True` on `KCIGPU`, `LinSigCIGPU`, `GCMIGPU`, `CMIknnGPU`
  and `CMIknnMixedGPU`.
* CDNOTS+ options, off by default: `lag0_orient="lingam"` orients undirected lag-0 edges
  (bootstrap-checked DirectLiNGAM, linear mechanisms); `addback_alpha` / `veto_alpha` /
  `alpha_mci` grow and prune lagged links (`result.lag_refinements`);
  `return_prefix=True` with `finish_cdnots_plus()` reuses one skeleton.
* CDNOTS / CDNOTS+: `majority_gamma` (ambiguity band for the majority collider rule) and
  `run_cdnots(collider_search="full" | "sepset")`.
* Rendering of undirected lag-0 edges is chosen after discovery: `to_binary(undirected=,
  conflict=)`, lossless `to_marks()`, `plot(undirected=)`, and the same on the tigramite
  converters (`cg_tig` unchanged). `evaluate_graph()` adds `*_lag0`, `*_lag0_adj` and
  `*_lagpos` metrics.
* `run_lucid(pervasive_lagged="factor_adjusted")`: opt-in factor-adjusted lagged edges on
  the pervasive route.
* `check_variance_dependence` (variance-only lagged parents), feeding new `inspect` facts
  and `recommendation["refine"]`; `merge_lag0`.
* `corr_table()` and `pairs_panel()`.
* Opt-in generator fixes: `henon_chain(bounded_start=True)`,
  `topology_scp(coef_pools="fixed")`, `RegimeConfig(threshold_rounding="ceil")`.
* Example notebooks `ci_test_selection` and `dowhy_latent_confounders`.

### Changed

Results can change for:

* CDNOTS / CDNOTS+ lag-0 orientation: Meek R1 runs to closure, collider votes condition on
  correctly shifted parents, ambiguous triples and NaN scores stay undirected, collider
  cycles keep their true edges. Also CDNOTS+ with `include_C=True` and `lag_verify=True`.
* `run_cdnots_plus` wraps discrete-unaware CI tests in `StratifiedCIT`, like `run_cdnots`;
  neither wraps `chisq` / `gsquared` (`discrete_cols=[]` restores the old behaviour).
* A one-sided lag-0 forbid keeps the adjacency; forbid both directions to remove it.
* GRACE defaults to its paper configuration (rank transform, Student-t likelihood,
  `lambda_rule="xcorr"`, one grow round, autoregressive head, `gate_threshold="auto"`,
  `effect_floor=1-exp(-0.02)`). A supplied `skeleton=` is a starting point;
  `grow_rounds=0, ar_path=False` keeps the result inside it.
* `recommend_config`: always CDNOTS; the CI test follows the data (`cmiknn-mixed-gpu`,
  `rcot`, `raft`, `parcorr-gpu`), with a GRACE or REAP refinement.
* LUCID and `inspect` place the Marchenko-Pastur edge from residual degrees of freedom, so
  routing and `n_factors` change on wide panels.
* `DFCITGPU`: the bandwidth scales with |Z| and uses leave-one-out smoothing; the automatic
  block length is sqrt(n) of the tested rows; the CLI and `discover_df` pass it their alpha.
* `sigkci` / `SigKCIGPU` are renamed `linsig` / `LinSigCIGPU` (old names deprecated) with a
  kernel-scaled `ridge`; `RCOTGPU` defaults to `num_f2=min(25, floor(sqrt(T)))`,
  `n_avg=1`, `seed=42`; `ParCorrGPU.__call__` reads the cached covariance (5.8x faster for
  scalar callers).
* `causal-ts -s` seeds the CI test, and disk-cache keys include every result-shaping
  setting, so old caches miss. `discover -a cdnots+` defaults `--alpha` to 0.01.
* `run_regime_discovery` counts lag-0 edges in the segment vote; `to_design_matrix` drops
  lagged children (with a warning); `inspect` skips non-numeric columns.
* Cedar `pvalue_matrix` is NaN (JSON `null`) for untested self-loops and required edges.

API:

* Keyword-only: `orient_margin`, `legacy_mci_conds`, `routed_deconfound` options after
  `arch_alpha`, and `run_cdnots_gated` arguments after `lambda_cv`.
* `discover_df(ci_test=)` reaches GRACE, GRACE-SS and REAP; `create_ci_test` ignores
  `device` for CPU-only tests; `causalts.__all__` drops import helpers; Cedar's
  `max_prune_path` is deprecated.

### Fixed

* CDNOTS / CDNOTS+ background knowledge: required lag-0 edges are oriented and never
  tested away (time-shifted copies included); edges forbidden both ways are removed; the
  forbidden-ancestor cleanup keeps required edges; `ancestral_violations` reports
  unsatisfied constraints.
* CDNOTS+: `pvalue_matrix` dropped lagged tests and transposed lag-0 ones; `max_conds_*`
  kept the lowest-indexed parents instead of the strongest; a gapped `lag_list` read the
  wrong lags; a C column was taken as a collider middle; lagged `o-o` / `x-x` and C edges
  rendered wrongly.
* Discrete data: `StratifiedCIT` / `CMIknnMixedGPU` lag structure, missing values, cache
  keys and permutation null; misaligned discrete columns with `include_C=True`; a crash
  with `--ci-test cmiknn-mixed-gpu` (now rejected for CEDAR and GRACE-SS); boolean columns.
* CI tests: infinite input raises `ValueError`; `ParCorrGPU` stale cache after replaced
  data and ill-conditioned batch tests (#74); `GCMIGPU` with multivariate X or Y;
  randomised nulls honour `seed`; `RCOTGPU` order dependence; `LinSigCIGPU` reuse across
  lag layouts; `DFCITGPU` over-rejection with |Z| >= 2 and the CLI's fixed block length.
* GRACE `run_ci_skeleton(include_C=False)` treated the last variable as C; a core-only
  install failed to import (undeclared `six`).
* `causal-ts evaluate` / `--ground-truth` trimmed extra lags; `shd_2d` / `shd_3d`
  counting; graph comparison plots with lag-0 mirror marks or one variable;
  `run_tigramite(method="pc_stable")` returned an empty graph; `select_features` C-label
  collisions.
* `sensitivity_analysis()` and `routed_deconfound(keep_undirected=True,
  pervasive_base="tetrad")` no longer raise; `ChangePointRegimeDetector.assign` on
  pandas 3; `generate -s 0`; `check_linearity` on long lags; read-only input warnings;
  docstring and CLI help errors.

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

[unreleased]: https://github.com/bloomberg/causal-ts/compare/v0.28.1...HEAD
[0.28.1]: https://github.com/bloomberg/causal-ts/compare/v0.28.0...v0.28.1
[0.28.0]: https://github.com/bloomberg/causal-ts/compare/v0.27.0...v0.28.0
[0.27.0]: https://github.com/bloomberg/causal-ts/compare/v0.26.0...v0.27.0
[0.26.0]: https://github.com/bloomberg/causal-ts/compare/v0.25.2...v0.26.0
[0.25.2]: https://github.com/bloomberg/causal-ts/compare/v0.25.1...v0.25.2

<!-- textlint-enable -->
