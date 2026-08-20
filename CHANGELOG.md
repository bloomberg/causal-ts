# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- textlint-disable -->

## [Unreleased]

### Added

* `corrplot(..., diag="glyph")` — renders the diagonal as an ordinary cell,
  using the same `method` and colormap as the rest of the matrix. Intended for
  *directed* matrices (a cause→effect adjacency or an edge-stability matrix),
  where the diagonal is real data such as a self-loop rather than the trivial
  1.0 of a correlation matrix. Significance markers and confidence-interval
  overlays are still skipped on the diagonal.

### Fixed

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
