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
  using `report`. For discovery, write it to a temp file
  (`df.to_parquet("/tmp/ts.parquet")`) and use the CLI, or ask the user how
  they'd like to proceed.

## 2. Inspect (facts + recommendation, one call)

```
causal-ts inspect <data-file>
```

This prints JSON: `{schema_version, data, facts, recommendation, cost_class,
warnings}`. Read it — do **not** re-derive these facts yourself.

- **`warnings`** — surface every one to the user (missing data, constant columns,
  too-few rows). If a column is heavily missing or constant, recommend fixing it
  (impute/drop) before trusting results.
- **`facts`** — linearity, per-column (non)stationarity + its `form`, suggested
  max lag.
- **`recommendation`** — `{algorithm, ci_test, include_C, c_preset, max_lag,
  rationale}`. This is a deterministic default; you may **override it** with
  context the tool can't see (e.g. the user says "these are already
  differenced/log-returns" → drop the C node; "I only care about causes of Y" →
  see `--target-var` in `causal-ts discover --help`).
- **`cost_class`** — `cheap | moderate | expensive`.

If unsure which CI test fits, run `causal-ts ci-test-info` and read the live guide
rather than guessing.

## 3. Cost gate

- `cost_class == "cheap"` → proceed to discovery without asking.
- `cost_class == "moderate"` → proceed, but tell the user what you're running.
- `cost_class == "expensive"` (large d, `kci`/`cmiknn-gpu`, or GRACE) →
  **checkpoint first.** Tell the user the config and that it may be slow, then
  offer a concrete cheaper alternative — stay **inside causal-ts** (do NOT
  suggest external tools like PCMCI/tigramite; they are optional deps and may be
  absent). Pick the alternative by what makes it expensive:
    - expensive from **GRACE / high `d`** → offer `cdnots` or `cedar` with
      `parcorr-gpu` (same skeleton, no neural-gate training);
    - expensive from the **CI test** (`kci` / `cmiknn-gpu`) → downshift to
      `splitkci`;
    - or lower `--max-lag`.
  Then wait for the go-ahead.

## 4. Run discovery

Use the recommended config. Example:

```
causal-ts -o <out-dir> discover <data-file> --json \
  --algorithm <rec.algorithm> --ci-test <rec.ci_test> \
  --max-lag <rec.max_lag> [--include-c --c-preset <rec.c_preset>]
```

`-o` is a **global** option and must come *before* `discover` (same for other
`causal-ts` commands). `--json` echoes the summary (including a named **edge
list**) to stdout — parse that directly. Artifacts (`estimated_graph.npy`,
`summary.json`) are written to the `-o` directory.

## 5. Interpret the graph

Read the `edges` list (`source → target @ lag`, with `pvalue` when present) and
report it in plain English. Apply this diagnostic playbook.

Note on `pvalue`: for the CDNOTS family it is **`null` unless you asked for it**.
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
  higher-recall test (`dfcit`).
- **`saturated` (near-complete graph)** → likely over-connection. Common causes:
  unhandled nonstationarity (was `include_C` set when the data is nonstationary?)
  or near-unit-root persistence. Suggest a stricter alpha (e.g. 0.01),
  differencing, or verifying the C-node preset matches the trend `form`.
- **High `max_in_degree` at `hub`** → inspect whether that variable is a common
  effect or an artifact of a confounder/persistence.
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

```
causal-ts -o <dir> discover <data> --json --algorithm <rec.algorithm> \
  --ci-test <rec.ci_test> --max-lag <rec.max_lag> --validate
```

Report edges with `persistence >= ~0.6` as stable and the rest as fragile.
`--validate` is **not supported for GRACE** (bootstrapping neural training is
impractical) — validate a GRACE result with `--algorithm cdnots`/`cedar` instead,
or in Python drive `causalts.bootstrap.temporal_bootstrap` with a
`run_cdnots_gated(sub, max_lag=<...>).gate_values` discovery function.

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
