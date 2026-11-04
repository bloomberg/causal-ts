---
name: causal-ts-discovery
description: >-
  Discover causal structure in time-series data with the causal-ts library.
  Use when the user wants a causal graph, or lead-lag / temporal dependencies,
  from time-series, sensor, or sequential data. NOT for effect size, ATE, or
  counterfactuals — that is effect estimation (see the handoff at the end).
---

# Causal discovery for time series (causal-ts)

Recover the causal graph — who-causes-whom, at which lags — from a multivariate
time series. CLI-first; drop to Python only where noted.

## 0. Prerequisite

The `causal-ts` CLI must be importable. If `causal-ts --help` fails, install with
`pip install causalts` (add `pip install causalts[parquet]` if the data is
parquet/feather).

## 1. Locate the data and do a health check

Identify what the user handed you:

- **A file** (`.csv`, `.parquet`, `.feather`) → use the CLI path (steps 2–4).
- **An in-memory pandas DataFrame** (notebook/session) → use the Python path:
  `from causalts import inspect_df; report = inspect_df(df)`, then jump to step 3
  using `report`. For discovery, `discover_df(df, algorithm=..., ci_test=...,
  max_lag=..., include_C=..., c_preset=...)` is the in-memory twin of the CLI —
  it returns a result object whose `.cg_tig` you can pass to
  `causalts.inspection.edges_from_graph` / `diagnostics_from_graph` to get the
  same edge list and diagnostics §5 expects. It raises rather than silently
  ignoring an argument the chosen algorithm can't honour. Otherwise write the
  frame to a temp file (`df.to_parquet("/tmp/ts.parquet")`) and use the CLI.

## 2. Inspect (facts + recommendation, one call)

```bash
causal-ts inspect <data-file>
```

This prints JSON: `{schema_version, data, facts, recommendation, cost_class,
warnings}`. Read it — do **not** re-derive these facts yourself.

- **`warnings`** — surface every one to the user (missing data, constant columns,
  too-few rows). If a column is heavily missing or constant, recommend fixing it
  (impute/drop) before trusting results.
- **`facts`** — linearity, per-column (non)stationarity + its `form`, suggested
  max lag, `latent_factor`, and `variance_dependence`.
  - **`variance_dependence`** — `{detected, n_pairs, pairs, reason}`. `detected: true`
    means some lagged variable changes another's *noise scale* without moving its
    mean (e.g. `"X0 -> X5"`). Mean-only tools (`raft`, `parcorr-gpu`, REAP) cannot
    see those edges, which is why the recommendation then switches to `rcot` +
    GRACE. `null` means the check did not run, and `reason` says why:
    `"not_applicable"` (`max_lag` 0 or a single variable), `"too_large"` (skipped to
    keep `inspect` cheap; mention to the user that variance-only parents were not
    ruled out) or `"unavailable"` (missing values, too few rows, degenerate
    columns). Never report `null` as "none found".
  - **`latent_factor`** — `{detected, spectral_ratio, tau}`. `detected: true` means
    one or more latent factors load broadly across the panel; plan on deconfounding
    (§5). Read the negatives narrowly: `false` only rules out factor structure this
    test can see — a latent cause touching just two or three variables produces no
    dominant eigenvalue and stays invisible — and `null` means the check could not run
    (missing values, `d < 2`, too few rows). **Never report either as "no
    confounding".**
- **`recommendation`** — `{algorithm, ci_test, refine, refine_params, include_C,
  c_preset, max_lag, params, rationale}`. `algorithm` is always `cdnots`; `refine`
  (`reap`, or `grace` when variance-only parents were detected) is an optional
  second step that refines that CDNOTS graph's **lagged** edges, with keyword
  arguments `refine_params` (`{"contemporaneous": true}` for REAP, which re-adds
  lag-0 edges; GRACE returns none, see §4; plus `"impute": "var_em"` when values
  are missing). This is a deterministic default; you may **override it** with
  context the tool can't see (e.g. the user says "these are already
  differenced/log-returns" → drop the C node; "I only care about causes of Y" →
  see `--target-var` in `causal-ts discover --help`).
- **`cost_class`** — `cheap | moderate | expensive`.

If unsure which CI test fits, run `causal-ts ci-test-info` and read the live guide
rather than guessing.

## 3. Cost gate

- `cost_class == "cheap"` → proceed to discovery without asking.
- `cost_class == "moderate"` → proceed, but tell the user what you're running.
- `cost_class == "expensive"` (large d, `kci`/`cmiknn-gpu`/`cmiknn-mixed-gpu`) →
  **checkpoint first.** Tell the user the config and that it may be slow, then
  offer a concrete cheaper alternative — stay **inside causal-ts** (do NOT
  suggest external tools like PCMCI/tigramite; they are optional deps and may be
  absent). Pick the alternative by what makes it expensive:
  - expensive from **high `d`** → offer `parcorr-gpu` (or `raft` for nonlinear
    mean effects), which scale best with the size of the conditioning sets;
  - expensive from the **CI test** (`kci` / `cmiknn-gpu`) → downshift to
    `rcot` (still full conditional independence) or `raft` (mean only);
  - or lower `--max-lag`.
  Then wait for the go-ahead.

## 4. Run discovery

Use the recommended config. Example:

```bash
causal-ts -o <out-dir> discover <data-file> --json \
  --algorithm <rec.algorithm> --ci-test <rec.ci_test> \
  --max-lag <rec.max_lag> <C flags> [--impute var_em]
```

Add `--impute var_em` when `rec.params` has `impute` (the data has missing values).

`<C flags>` is `--include-c --c-preset <rec.c_preset>` when `rec.include_C` is
true and `--no-c` when it is false. Always pass one of them: the CLI's own
default is `--include-c`.

Then, as the optional refinement (skip it when `rec.refine` is null), run the
same command with `--algorithm <rec.refine>` (plus `--contemporaneous` for
REAP): REAP/GRACE build
that CDNOTS graph with the given `--ci-test` and refine its lagged edges in one run (GRACE trains a neural model: tell the user
it is slower, and checkpoint first when `d` is large); keep `--impute var_em` when
`rec.refine_params` has `impute`. GRACE returns **no lag-0 edges**: report
the lagged edges from the GRACE run and the lag-0 edges from the CDNOTS run
(in Python, `causalts.inspection.merge_lag0(refined, cdnots_result)`).

`-o` is a **global** option and must come *before* `discover` (same for other
`causal-ts` commands). `--json` echoes the summary (including a named **edge
list**) to stdout — parse that directly. Artifacts (`estimated_graph.npy`,
`summary.json`) are written to the `-o` directory.

## 5. Interpret the graph

Read the `edges` list (`source → target @ lag`, with `pvalue` when present) and
report it in plain English. Apply this diagnostic playbook.

Note on `pvalue`: for the CDNOTS family and CEDAR it is **`null` unless you asked
for it**.
P-values are off by default as a memory safeguard for very high-dimensional data;
pass **`--pvalues`** to `discover` to get them (safe on low/moderate `d` — i.e.
when `cost_class` is `cheap`/`moderate`; skip it on very high `d`). So if the user
wants per-edge p-values and `d` isn't huge, re-run with `--pvalues`. `null` means
"not requested," not "no confidence" and not that the algorithm gates instead of
testing — don't speculate about internals. For a robustness measure (as opposed
to a per-test p-value), use `--validate` (§6).

`discover --json` includes a computed `diagnostics` block (`empty`, `saturated`,
`density`, `self_loops`, `contemporaneous`/`lagged`, `max_in_degree` + `hub`) —
read it rather than eyeballing, then apply:

- **`empty` (no edges)** → alpha may be too strict, max_lag too small, or the
  CI test underpowered for the mechanism. Suggest a looser alpha or a
  more general test (`rcot`, or `dfcit` for high recall).
- **`saturated` (near-complete graph)** → likely over-connection. Common causes:
  unhandled nonstationarity (was `include_C` set when the data is nonstationary?)
  or near-unit-root persistence. Suggest a stricter alpha (e.g. 0.01),
  differencing, or verifying the C-node preset matches the trend `form`.
- **High `max_in_degree` at `hub`** → inspect whether that variable is a common
  effect or an artifact of a confounder/persistence.
- **Many `contemporaneous` edges, especially with step-2 `latent_factor.detected`**
  → a broadly-loading latent factor induces exactly this signature (lag-0 edges
  among variables with no direct causal link). Offer to correct it:
  `causal-ts deconfound <graph.npy> --data <data-file>` (or `res.deconfound()` in
  Python), which infers whether the confounding is sparse or pervasive and applies
  the matching correction. Report edges-before/after and be explicit that the
  removed edges were **judged confounded, not disproven**. Which edges are eligible
  depends on the inferred regime: the pervasive branch rewrites the lag-0 slice only,
  while the sparse branch also tests lagged edges against observed controls.
- **`lagged` == 0 (all edges contemporaneous)** → check that `max_lag` is
  adequate and the sampling rate isn't washing out dynamics.
- **Self-loops** are autoregressive terms (a variable's own past), expected for
  persistent series — not spurious.
- Lag 0 = contemporaneous; lag k = `source(t-k) → target(t)`.

If the data was flagged nonstationary/short in step 2, caveat the result
accordingly.

**Done when** every edge is stated as `source → target @ lag`, all step-2
warnings are surfaced, and a next step (robustness in §6, or the effects handoff
in §7) is offered.

## 6. (Optional) Robustness — only with consent

If the user wants confidence in the edges and accepts extra runtime, use the
built-in stability check: add **`--validate`** to `discover` (tune with
`--n-bootstrap`, default 20, and `--window-frac`, default 0.6). It re-runs
discovery on contiguous windows and annotates every edge in the JSON with
`persistence` (fraction of windows it recurs) plus a `stability` block:

```bash
causal-ts -o <dir> discover <data> --json --algorithm <rec.algorithm> \
  --ci-test <rec.ci_test> --max-lag <rec.max_lag> --validate
```

Report edges with `persistence >= ~0.6` as stable and the rest as fragile.
`--validate` is **not supported for GRACE, GRACE-SS or REAP** (re-running the
refinement over many windows is impractical) — validate the CDNOTS graph they
refine with `--algorithm cdnots` instead,
or in Python drive `causalts.bootstrap.temporal_bootstrap` with a discovery
function returning `run_grace(sub, max_lag=<...>)`'s final graph
(`cg_tig != 0`), not its `gate_values`: self-loops are decided by an F-test and have
no gate value, and edges the effect floor removed keep theirs.

**Caveat: persistence measures robustness to *sampling*, not correctness.** A
*systematic* artifact — e.g. a lag-k edge that is just an echo of a true
lag-(k−1) edge through a feedback loop, or shared autocorrelation — recurs in
every window and so scores high persistence while still being wrong. Read high
persistence as "not a sampling fluke", not "definitely causal"; for a suspected
echo, check whether the lag-k edge survives conditioning on the lag-(k−1) edge
(stricter alpha, or inspect the p-value).
This is more expensive; get consent first, especially when `cost_class` was
`expensive`. Quick alternative: re-run discovery at a stricter alpha and report
which edges survive.

## 7. Effects handoff (out of scope here)

If the user asks **how much** X affects Y, for an ATE, or for counterfactuals —
that is effect *estimation*, not discovery. Don't do it in this skill. Point them
to causal-ts's DoWhy integration:

> causal-ts can estimate that from the discovered graph via its DoWhy bridge —
> see `causal-ts dowhy --help` or `result.estimate_effect(...)` (needs
> `pip install causalts[dowhy]`). Want me to set that up?
