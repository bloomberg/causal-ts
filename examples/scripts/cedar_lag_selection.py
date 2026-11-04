# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CEDAR : Choosing a Lag Selection Method
#
# CEDAR Phase 0 selects the most predictive lag per variable pair before running any CI test.
# Two parameters control this:
#
# - **`lag_method`** — how importance scores are computed  
# - **`lag_pvalue_method`** — how significance is tested
#
# This notebook compares options on **ex3** (d=11, T=300) and gives practical guidance on
# when to deviate from the defaults.
#
# | `lag_method` | Type | Best when |
# |---|---|---|
# | `partial_dcor` | Unbiased dcor on Y(t) residualised on Y(t-1) | **Default** — removes AR-mediated lag inflation; best for hub-heavy graphs |
# | `dcor` | Unbiased U-centered dcor | General nonlinear |
# | `dcor_biased` | Biased distance correlation | Comparison with the unbiased estimator; no speed advantage |
# | `pearson` | Pearson correlation | Confirmed linear data; fastest |
#
# | `lag_pvalue_method` | Type | Best when |
# |---|---|---|
# | `t_test` | Asymptotic t-test, no permutation | **Default** — far faster than permutation (ratio printed in section 2), n ≥ 20 |
# | `circular_shift` | Circular-shift permutation | Strict FPR control on short AR series |

# %%
import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from causalts.synthetic_data.synthetic_datasets import load_dataset
from causalts.ci_tests import PartialCorr
from causalts.cedar import run_cedar
from causalts.utils import evaluate_graph


# %% [markdown]
# ## Dataset: Cascade Network (ex3, d=11)
#
# d=11 variables, T=300, max_lag=3 — **linear VAR** signaling cascade (all edges are linear weighted sums of lagged values, Gaussian noise). Stationary, no nonstationarity node C.

# %%
result = load_dataset("ex3", seed=42, T=300)
df = result["df"]
ground_truth = result["ground_truth"]

print(f"Dataset: {df.shape[0]} × {df.shape[1]} variables")
print(f"Variables: {list(df.columns)}")


# %% [markdown]
# ## 1. Compare `lag_method`
#
# Timings are wall-clock for a single run. The first dcor-based method in the loop
# (`dcor_biased`) also pays a one-off setup cost, so its time is inflated; run warm,
# `dcor_biased` is still slower than `dcor` (about 0.18 s vs 0.09 s on this data).

# %%
lag_methods = ["pearson", "dcor_biased", "dcor", "partial_dcor"]
records = []
results_by_method = {}

for lag_method in lag_methods:
    ci = PartialCorr(np.ascontiguousarray(df.values))
    t0 = time.time()
    res = run_cedar(
        df=df, ci_test=ci, max_lag=3,
        lag_method=lag_method,
        alpha_cond1=0.05, alpha_cond2=0.05,
    )
    elapsed = time.time() - t0
    results_by_method[lag_method] = res

    # AR(1) self-loops are included by default (include_autoreg=True)
    # and match the autoregressive ground truth in ex3.
    m = evaluate_graph(res.cg_tig, ground_truth)
    records.append(dict(
        lag_method=lag_method,
        F1=m["F1"], Precision=m["Precision"], Recall=m["TPR"],
        SHD=m["SHD"], TP=m["TP"], FP=m["FP"], FN=m["FN"],
        n_tests=res.n_tests, Time=elapsed,
    ))
    default = " ← default" if lag_method == "partial_dcor" else ""
    print(f"{lag_method:<14} F1={m['F1']:.3f}  Prec={m['Precision']:.3f}  "
          f"Recall={m['TPR']:.3f}  SHD={m['SHD']}  n_tests={res.n_tests}  ({elapsed:.2f}s){default}")

# %%
# Lag importance matrix for the default (partial_dcor) result
results_by_method["partial_dcor"].plot_lag_importance(figsize=(10, 8))
plt.show()


# %%
df_res = pd.DataFrame(records)
colors = ["#6b7280", "#00B5AD", "#4B44B8", "#B8860B"]
x = np.arange(len(lag_methods))

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, col, title in zip(axes, ["F1", "Precision", "SHD"],
                           ["F1", "Precision", "SHD (lower=better)"]):
    ax.bar(x, df_res[col], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(lag_methods, rotation=30, ha="right")
    ax.set_title(title)
    if col != "SHD":
        ax.set_ylim(0, 1)

plt.tight_layout()
plt.show()

print(df_res[["lag_method", "F1", "Precision", "Recall", "SHD", "n_tests", "Time"]].to_string(index=False))


# %% [markdown]
# ## 2. Compare `lag_pvalue_method`
#
# `t_test` is analytic. `circular_shift` runs permutations that preserve the target's
# autocorrelation, which gives better-calibrated p-values for a single test, but it is
# slow and interacts poorly with the Benjamini-Hochberg (BH) FDR correction that
# Phase 0 applies across all lag hypotheses. This section uses `lag_method="dcor"`
# because the permutation test compares raw (not residualised) distance correlations.
#
# > **Why `circular_shift` can find no cross-variable edges here:** With 200
# > permutations the smallest achievable p-value is 1/201 ≈ 0.005. Phase 0 tests
# > every ordered pair of distinct variables at lags 1-3 (lag 0 and self-pairs are
# > excluded), so m = 11 × 10 × 3 = 330 hypotheses. BH rejects the k smallest p-values
# > for the largest k with p₍ₖ₎ ≤ k·α/m. At α = 0.05 the first cutoff is
# > 0.05/330 ≈ 1.5e-4, far below 0.005, so a p-value at the floor survives only if at
# > least 33 hypotheses share it (33 × 0.05/330 ≥ 0.005). The cell below counts how many
# > reach the floor in this run. The edge count in the table below is not zero
# > because Cedar adds AR self-loops in a separate phase that the lag test does not
# > gate. Those are real edges, so precision reads 1.000 even though no cross-variable
# > edge was discovered. This is a resolution limit of permutation p-values under
# > multiple-testing correction, not a bug.
# >
# > **Workaround:** use `t_test` (analytic, not limited to discrete p-values), or raise
# > `n_permutations` so that 1/(n_permutations+1) is below α/m (more than ≈6,600 here;
# > the cost grows linearly with `n_permutations`).

# %%
pvalue_methods = ["t_test", "circular_shift"]
records_pv = []
edge_split = {}   # method -> (cross-variable edges, AR self-loops)

for pv_method in pvalue_methods:
    ci = PartialCorr(np.ascontiguousarray(df.values))
    t0 = time.time()
    res = run_cedar(
        df=df, ci_test=ci, max_lag=3,
        lag_method="dcor",
        lag_pvalue_method=pv_method,
        alpha_cond1=0.05, alpha_cond2=0.05,
    )
    elapsed = time.time() - t0

    m = evaluate_graph(res.cg_tig, ground_truth)  # includes AR self-loops
    g = res.cg_tig
    n_self = int(sum(g[k, k, :].sum() for k in range(g.shape[0])))
    edge_split[pv_method] = (int(g.sum()) - n_self, n_self)
    if pv_method == "circular_shift":
        # Phase 0 p-values for the hypotheses BH corrects over (lags 1-3, no self-pairs)
        hyp = np.zeros_like(res.lag_pvalues, dtype=bool)
        hyp[:, :, 1:] = True
        for lag in range(1, hyp.shape[2]):
            np.fill_diagonal(hyp[:, :, lag], False)
        cs_pvals = res.lag_pvalues[hyp]
    records_pv.append(dict(
        lag_pvalue_method=pv_method,
        F1=m["F1"], Precision=m["Precision"], Recall=m["TPR"],
        SHD=m["SHD"], edges=int(res.cg_tig.sum()),
        n_tests=res.n_tests, Time=elapsed,
    ))
    print(f"{pv_method:<16} F1={m['F1']:.3f}  Prec={m['Precision']:.3f}  "
          f"Recall={m['TPR']:.3f}  SHD={m['SHD']}  edges={int(res.cg_tig.sum())}  ({elapsed:.2f}s)")

print()
print(pd.DataFrame(records_pv)[
    ["lag_pvalue_method","F1","Precision","Recall","SHD","edges","n_tests","Time"]
].to_string(index=False))
print()
n_cross, n_self = edge_split["circular_shift"]
m_hyp = cs_pvals.size
p_floor = 1 / 201
print(f"circular_shift: {n_cross} cross-variable edges, {n_self} AR self-loops.")
print(f"BH over m={m_hyp} hypotheses: first cutoff 0.05/m = {0.05 / m_hyp:.1e}; "
      f"smallest p-value = {cs_pvals.min():.4f}")
print(f"Hypotheses at the permutation floor 1/201: {int(np.isclose(cs_pvals, p_floor).sum())} "
      f"(at least {int(np.ceil(p_floor * m_hyp / 0.05))} needed for the floor to survive BH)")
t_ratio = records_pv[1]["Time"] / records_pv[0]["Time"]
print(f"circular_shift / t_test runtime: {t_ratio:.0f}x")


# %% [markdown]
# ## Recommendations
#
# **`lag_method`:**
#
# | Method | Use when |
# |---|---|
# | `partial_dcor` | **Default** — removes AR-mediated lag inflation; best overall, especially hub-heavy graphs |
# | `dcor` | Alternative for general nonlinear data |
# | `dcor_biased` | Comparison only; slower and less accurate than `dcor` on this dataset |
# | `pearson` | Data is confirmed linear (see Ramsey RESET below); fastest |
#
# **`lag_pvalue_method`:**
#
# | Method | Use when |
# |---|---|
# | `t_test` | **Default** — analytic and much faster; use for standard settings |
# | `circular_shift` | Short, strongly autocorrelated series at small d, where the BH cutoff α/m is not far below 1/(n_permutations+1); otherwise raise `n_permutations` above m/α |
#
# The linearity check below uses the same rule as `inspect_df`: data count as
# nonlinear when more than 20% of the Ramsey RESET tests reject linearity.

# %%
from causalts.utils.linearity import check_linearity

# RESET test conditions on Y_j(t-1) to absorb VAR autocorrelation,
# avoiding the inflated false positive rate of simple bivariate regressions.
lin_report = check_linearity(df)
frac = lin_report["fraction_nonlinear"]
print(f"Ramsey RESET (AR-controlled): {frac:.1%} of tests reject linearity")
if frac > 0.2:
    print("→ nonlinear: keep the default 'partial_dcor' (or 'dcor')")
else:
    print("→ appears linear: 'pearson' is a faster option; 'partial_dcor' remains valid")
