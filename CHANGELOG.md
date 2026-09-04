# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- textlint-disable -->

## [Unreleased]

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
* `corrplot(..., diag="glyph")` — renders the diagonal as an ordinary cell,
  using the same `method` and colormap as the rest of the matrix. Intended for
  *directed* matrices (a cause→effect adjacency or an edge-stability matrix),
  where the diagonal is real data such as a self-loop rather than the trivial
  1.0 of a correlation matrix. Significance markers and confidence-interval
  overlays are still skipped on the diagonal.

### Fixed

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

* **The GES, LGES and TGES baselines were substantially understated.** Two independent
  problems, both now fixed and covered by regression tests:
  * `ges_discovery` read causal-learn's adjacency matrix with the endpoints transposed,
    so every directed contemporaneous edge came back reversed and every directed lagged
    edge was silently dropped — only undirected edges survived.
  * The vendored GES search behind `lges_discovery` / `tges_discovery`
    (`causalts/lges.py`) is a hand-extracted condensation of upstream `ges`, and the
    extraction broke the CPDAG construction and the Insert, Delete and Turn operators.
    The forward phase stopped far short of the optimum, so LGES never converged — its
    F1 sat at 0.35–0.52 on `ex2` no matter how much data it was given.

  All three now also apply temporal background knowledge (a variable can only cause
  another at an equal or later time step), which the lag-embedded search previously
  ignored. On the `baseline_comparison` datasets, F1 moves from 0.125/0.154/0.522/0.333
  (GES) and 0.400/0.417/0.526/0.333 (LGES/TGES) to 0.636/0.917/0.949/0.435 for all
  three; LGES now reaches F1 1.000 on `ex2` by T=5,000. Every correction was verified
  against upstream `ges` 1.1.1 — no defect originated with the upstream authors — and
  the corrected search returns CPDAGs identical to upstream's on random DAGs while
  running faster than it. `ges_discovery` gains an `engine=` argument selecting the
  vendored search (default) or causal-learn.

  Any prior comparison against these baselines understates them.
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


[unreleased]: https://github.com/bloomberg/causal-ts/compare/v0.26.0...HEAD
[0.26.0]: https://github.com/bloomberg/causal-ts/compare/v0.25.2...v0.26.0
[0.25.2]: https://github.com/bloomberg/causal-ts/compare/v0.25.1...v0.25.2

<!-- textlint-enable -->
