# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: .venv-nbrun
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Baseline Comparison: Constraint-Based vs. Standard Methods
#
# This notebook compares causal-ts discovery algorithms (CDNOTS, CDNOTS+, CEDAR) against
# standard baselines available from `causalts.baselines`:
#
# | Method | Type | Assumptions | Thresholding |
# |--------|------|-------------|--------------|
# | **CDNOTS** | Constraint-based (PC) | Any CI test, handles nonstationarity | CI test p-value |
# | **CDNOTS+** | Constraint-based (PCMCI+) | Any CI test, handles nonstationarity | CI test p-value |
# | **CEDAR (linear)** | Autoregressive | Lag selection + ParCorr CI testing | CI test p-value |
# | **CEDAR (nonlinear)** | Autoregressive | Lag selection + SplitKCI CI testing | CI test p-value |
# | **GES** | Score-based (BIC) | Linear Gaussian, lag-embedded with temporal background knowledge (no edge from present into past) | BIC penalty (automatic) |
# | **LGES** | Score-based (BIC) | Linear Gaussian, less greedy (SafeInsert + ConservativeInsert) | BIC penalty (automatic) |
# | **TGES** | Score-based (BIC) | Linear Gaussian, LGES + tier-based orientation of undirected edges | BIC penalty (automatic) |
# | **VARLiNGAM** | ICA-based | Linear, non-Gaussian noise | Coefficient threshold (default 0.05) |
# | **LASSO-VAR** | L1-penalized VAR | Linear, autoregressive | LassoCV penalty (automatic) |
# | **Granger** | LASSO-VAR | Linear, autoregressive | Any non-zero LASSO coef |
#
# > **CDNOTS vs. CDNOTS+:** CDNOTS uses the standard PC skeleton. CDNOTS+ replaces it
# > with a two-phase PCMCI+-style skeleton: phase 1 discovers a parent superset per variable,
# > phase 2 re-tests each edge conditioning only on discovered parents — avoiding the
# > over-conditioning problem in high-dimensional settings. Both share the same orientation
# > pipeline (colliders → Meek → nonstationarity-based).
#
# > **CEDAR (linear) vs. CEDAR (nonlinear):** Both use the same two-phase pipeline:
# > Phase 0 selects lags via `partial_dcor` (unbiased distance correlation on
# > Y(t) residualised on Y(t−1)); Phases 1–2 run conditional independence tests
# > using the provided CI test object. The linear variant passes `ParCorrGPU`
# > (partial correlation, fast, assumes linear Gaussian relationships); the nonlinear
# > variant passes `SplitKCIGPU` (kernel CI test, handles arbitrary nonlinear
# > dependencies at the cost of speed).
#
# > **GES vs. LGES vs. TGES:** all three run on the lag-embedded matrix under the same temporal
# > background knowledge — a variable can only cause another at an equal or later time step, so the
# > search never considers an edge from the present into the past. GES is vanilla greedy equivalence
# > search under that constraint. LGES (Less Greedy ES) adds SafeInsert (score check before
# > inserting) and ConservativeInsert (early stopping), making it more conservative. TGES (Temporal
# > GES, Larsen et al. 2025) adds a post-CPDAG orientation step that uses the temporal tier structure
# > to direct remaining undirected edges — producing a "tiered maximally oriented PDAG" with strictly
# > more directed edges than the raw CPDAG.
#
# > **Note on evaluation fairness:** All methods are evaluated on the full ground truth including
# > autoregressive self-edges (X_{t-k}→X_t). CEDAR adds AR(1) self-loops for all variables
# > unconditionally via `assume_ar1=True` — not through CI testing. When the ground truth is
# > missing AR self-loops (e.g. ex2), these become false positives and adversely affect
# > CEDAR's precision.
#
# > **Note:** VARLiNGAM uses a coefficient magnitude threshold (default `threshold=0.05`)
# > to determine edges. LASSO-VAR and Granger use LASSO sparsity — non-zero coefficients
# > are edges. GES/LGES/TGES and constraint-based methods produce edges directly.

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from causalts.synthetic_data.synthetic_datasets import load_dataset
from causalts.ci_tests import ParCorrGPU
from causalts.ci_tests import SplitKCIGPU
from causalts import run_cdnots, run_cdnots_plus
from causalts.cedar import run_cedar
from causalts.utils import evaluate_graph
from causalts.baselines import (
    ges_discovery, lges_discovery, tges_discovery,
    varlingam_discovery, granger_discovery, lasso_var_discovery
)


# %%
def run_comparison(dataset_name, seed=42, nonlinear=False):
    """Run all methods on a dataset and return metrics."""
    result = load_dataset(dataset_name, seed=seed)
    df, gt = result["df"], result["ground_truth"]
    max_lag = gt.shape[2] - 1
    d = df.shape[1]
    print(f"Dataset: {dataset_name} | d={d}, T={df.shape[0]}, max_lag={max_lag}")
    print("-" * 65)

    rows = []

    # --- CDNOTS (linear) ---
    ci_test = ParCorrGPU(np.zeros((2, 2)), device="cpu")
    res = run_cdnots(
        df=df, indep_test=ci_test, num_lags=max_lag,
        include_C=True, alpha=0.05, stable=True
    )
    m = evaluate_graph(res.cg_tig[:d, :d, :], gt)
    rows.append({"Method": "CDNOTS (linear)", **m})

    # --- CDNOTS (nonlinear) ---
    if nonlinear:
        ci_nl = SplitKCIGPU(np.zeros((2, 2)), device="cpu")
        res_nl = run_cdnots(
            df=df, indep_test=ci_nl, num_lags=max_lag,
            include_C=True, alpha=0.05, stable=True
        )
        m_nl = evaluate_graph(res_nl.cg_tig[:d, :d, :], gt)
        rows.append({"Method": "CDNOTS (nonlinear)", **m_nl})

    # --- CDNOTS+ (linear) ---
    ci_plus = ParCorrGPU(np.zeros((2, 2)), device="cpu")
    res_plus = run_cdnots_plus(
        df=df, indep_test=ci_plus, num_lags=max_lag,
        include_C=True, alpha=0.05
    )
    m_plus = evaluate_graph(res_plus.cg_tig[:d, :d, :], gt)
    rows.append({"Method": "CDNOTS+ (linear)", **m_plus})

    # --- CDNOTS+ (nonlinear) ---
    if nonlinear:
        ci_plus_nl = SplitKCIGPU(np.zeros((2, 2)), device="cpu")
        res_plus_nl = run_cdnots_plus(
            df=df, indep_test=ci_plus_nl, num_lags=max_lag,
            include_C=True, alpha=0.05
        )
        m_plus_nl = evaluate_graph(res_plus_nl.cg_tig[:d, :d, :], gt)
        rows.append({"Method": "CDNOTS+ (nonlinear)", **m_plus_nl})

    # --- CEDAR (linear) ---
    # exclude_self_loops=True: CEDAR tests cross-variable pairs only and never
    # discovers autoregressive self-edges (X_{t-k}→X_t); excluding them gives
    # a fair evaluation of what CEDAR is designed to find.
    try:
        ci_cedar = ParCorrGPU(np.zeros((2, 2)), device="cpu")
        res_cedar = run_cedar(df, ci_cedar, alpha_cond1=0.01, alpha_cond2=0.01, max_lag=max_lag)
        m = evaluate_graph(res_cedar.cg_tig, gt) #, exclude_self_loops=True)
        rows.append({"Method": "CEDAR (linear)", **m})
    except Exception as e:
        print(f"  CEDAR (linear) failed: {e}")
        rows.append({"Method": "CEDAR (linear)", "F1": np.nan, "Precision": np.nan, "TPR": np.nan, "SHD": np.nan})

    # --- CEDAR (nonlinear) ---
    if nonlinear:
        try:
            ci_cedar_nl = SplitKCIGPU(np.zeros((2, 2)), device="cpu")
            res_cedar_nl = run_cedar(df, ci_cedar_nl, alpha_cond1=0.01, alpha_cond2=0.01, max_lag=max_lag)
            m_nl = evaluate_graph(res_cedar_nl.cg_tig, gt) #, exclude_self_loops=True)
            rows.append({"Method": "CEDAR (nonlinear)", **m_nl})
        except Exception as e:
            print(f"  CEDAR (nonlinear) failed: {e}")
            rows.append({"Method": "CEDAR (nonlinear)", "F1": np.nan, "Precision": np.nan, "TPR": np.nan, "SHD": np.nan})

    # --- GES ---
    try:
        G_ges, _ = ges_discovery(df, max_lag=max_lag)
        m = evaluate_graph(G_ges, gt)
        rows.append({"Method": "GES", **m})
    except Exception as e:
        print(f"  GES failed: {e}")
        rows.append({"Method": "GES", "F1": np.nan, "Precision": np.nan, "TPR": np.nan, "SHD": np.nan})

    # --- LGES ---
    try:
        G_lges, _ = lges_discovery(df, max_lag=max_lag, mode="lges")
        m = evaluate_graph(G_lges, gt)
        rows.append({"Method": "LGES", **m})
    except Exception as e:
        print(f"  LGES failed: {e}")
        rows.append({"Method": "LGES", "F1": np.nan, "Precision": np.nan, "TPR": np.nan, "SHD": np.nan})

    # --- TGES ---
    try:
        G_tges, _ = tges_discovery(df, max_lag=max_lag, mode="lges")
        m = evaluate_graph(G_tges, gt)
        rows.append({"Method": "TGES", **m})
    except Exception as e:
        print(f"  TGES failed: {e}")
        rows.append({"Method": "TGES", "F1": np.nan, "Precision": np.nan, "TPR": np.nan, "SHD": np.nan})

    # --- VARLiNGAM (threshold=0.05) ---
    G_vl, _, _ = varlingam_discovery(df, max_lag=max_lag, threshold=0.05)
    m = evaluate_graph(G_vl, gt)
    rows.append({"Method": "VARLiNGAM", **m})

    # --- LASSO-VAR ---
    G_lv, _, _ = lasso_var_discovery(df, max_lag=max_lag)
    m = evaluate_graph(G_lv, gt)
    rows.append({"Method": "LASSO-VAR", **m})

    # --- Granger ---
    G_gr, _, _ = granger_discovery(df, max_lag=max_lag)
    m = evaluate_graph(G_gr, gt)
    rows.append({"Method": "Granger", **m})

    df_results = pd.DataFrame(rows)[["Method", "F1", "Precision", "TPR", "SHD"]]
    for col in ["F1", "Precision", "TPR"]:
        df_results[col] = df_results[col].round(3)
    return df_results


# %% [markdown]
# ## Example 1: 5-Node Nonlinear System (ex1, d=5, T=300)
#
# A 5-variable nonlinear system. We run CDNOTS with both a linear CI test (`ParCorrGPU`)
# and a nonlinear one (`SplitKCIGPU`) to show the importance of matching the test to the data.
# Baselines (GES, VARLiNGAM, Granger) all assume linearity.

# %%
results_ex1 = run_comparison("ex1", nonlinear=True)
results_ex1

# %% [markdown]
# ## Example 2: 6-Node Linear VAR (ex2, d=6, T=300)
#
# A 6-variable linear VAR process. All methods should handle this reasonably well,
# but constraint-based methods leverage conditional independence structure more effectively.

# %%
results_ex2 = run_comparison("ex2")
results_ex2

# %% [markdown]
# ## Example 3: Cascade Network (ex3, d=11, T=300)
#
# An 11-variable linear system — the increased dimensionality challenges all methods
# but particularly score-based and LASSO approaches.

# %%
results_ex3 = run_comparison("ex3")
results_ex3

# %% [markdown]
# ## Example 4: Henon Chain (d=5, T=200, highly coupled nonlinear)
#
# Each node depends nonlinearly on its nearest neighbors — a dense coupling
# pattern that creates high cross-correlation across all variables. This tests
# the hypothesis that **CDNOTS+ helps when features are correlated**: standard
# PC builds large conditioning sets on dense graphs, making CI tests noisy,
# while CDNOTS+ restricts conditioning to discovered parents.
#

# %%
from causalts.synthetic_data.synthetic_datasets import henon_chain

results_henon = run_comparison("henon", nonlinear=True)
results_henon

# %% [markdown]
# ## Do GES, LGES and TGES ever actually differ?
#
# On the four datasets above they land on identical graphs, which raises a fair question:
# what is LGES's extra machinery (SafeInsert, ConservativeInsert) or TGES's tier-orientation
# pass buying you, if the output never changes?
#
# That is an artifact of testing on small (d <= 11), clean systems where plain GES's own
# search already reaches the same fixed point. At larger `d`, lower `T`, or with
# contemporaneous edges on top of the lagged structure, they diverge.
#

# %%
import numpy as np
from causalts.baselines import ges_discovery, lges_discovery, tges_discovery
from causalts.utils import evaluate_graph


def stable_var(d, T, n_parents, seed, radius=0.7, contemp_frac=0.3):
    """A VAR(1) with some contemporaneous (lag-0) edges mixed in."""
    r = np.random.default_rng(seed)
    B_lag = np.zeros((d, d))
    for j in range(d):
        for i in r.choice([k for k in range(d) if k != j], size=n_parents, replace=False):
            B_lag[i, j] = r.uniform(0.3, 0.8) * r.choice([-1, 1])
    ev = np.max(np.abs(np.linalg.eigvals(B_lag)))
    if ev > 0:
        B_lag *= radius / ev
    order = r.permutation(d)
    B_c = np.zeros((d, d))
    for idx, j in enumerate(order):
        if idx > 0 and r.random() < contemp_frac:
            i = order[r.integers(0, idx)]
            B_c[i, j] = r.uniform(0.3, 0.7) * r.choice([-1, 1])
    X = np.zeros((T, d))
    I_minus_Bc_inv = np.linalg.inv(np.eye(d) - B_c.T)
    for t in range(1, T):
        eps = r.standard_normal(d) + X[t - 1] @ B_lag
        X[t] = eps @ I_minus_Bc_inv.T
    gt = np.zeros((d, d, 2), dtype=int)
    gt[:, :, 1] = (B_lag != 0).astype(int)
    gt[:, :, 0] = (B_c != 0).astype(int)
    return pd.DataFrame(X, columns=[f"X{i}" for i in range(d)]), gt


df_big, gt_big = stable_var(d=15, T=300, n_parents=3, seed=7)

G_ges, _ = ges_discovery(df_big, max_lag=1)
G_lges, _ = lges_discovery(df_big, max_lag=1, mode="lges")
G_tges, _ = tges_discovery(df_big, max_lag=1, mode="lges")

for name, G in [("GES", G_ges), ("LGES", G_lges), ("TGES", G_tges)]:
    m = evaluate_graph(G, gt_big)
    print(f"{name:<5} F1={m['F1']:.3f}  Precision={m['Precision']:.3f}  "
          f"TPR={m['TPR']:.3f}  SHD={m['SHD']}  edges={int(G.sum())}")

print(f"\nGES vs LGES identical? {np.array_equal(G_ges, G_lges)}")
print(f"LGES vs TGES identical? {np.array_equal(G_lges, G_tges)}")


# %% [markdown]
# ## Summary
#

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 7))
axes = axes.flatten()

datasets = [
    ("ex1 (d=5, nonlinear)",       results_ex1),
    ("ex2 (d=6, linear)",           results_ex2),
    ("ex3 (d=11, linear, sparse)",  results_ex3),
    ("Henon (d=5, nonlinear, dense)", results_henon),
]

colors = {
    "CDNOTS (linear)":      "#90CAF9",
    "CDNOTS (nonlinear)":   "#1565C0",
    "CDNOTS+ (linear)":     "#A5D6A7",
    "CDNOTS+ (nonlinear)":  "#2E7D32",
    "CEDAR (linear)":       "#CE93D8",
    "CEDAR (nonlinear)":    "#6A1B9A",
    "GES":                  "#FFCC80",
    "LGES":                 "#FF9800",
    "TGES":                 "#E65100",
    "VARLiNGAM":            "#EF9A9A",
    "LASSO-VAR":            "#795548",
    "Granger":              "#F44336",
}

for ax, (title, df_r) in zip(axes, datasets):
    bars = ax.bar(
        df_r["Method"], df_r["F1"],
        color=[colors.get(m, "#999") for m in df_r["Method"]],
    )
    for bar, v in zip(bars, df_r["F1"]):
        if v == v:  # skip NaN
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=7)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
    ax.tick_params(axis="x", rotation=45, labelsize=7)

plt.suptitle("Causal Discovery: Constraint-Based vs. Baselines", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Key Takeaways
#
# 1. **CEDAR leads on ex1 (nonlinear) and ex3; CDNOTS+ ties CEDAR on ex2; CDNOTS leads on Henon.**
#    - **ex1** (nonlinear, d=5): CEDAR (nonlinear) achieves the highest F1 at 0.778 vs CDNOTS+ (nonlinear) 0.571. Partial distance correlation lag selection captures nonlinear associations that ParCorr misses, giving CEDAR an edge before the CI test even runs.
#    - **ex2** (linear VAR, d=6): GES, LGES and TGES tie for the lead at 0.917, ahead of CDNOTS+ and CEDAR (both 0.870) and CDNOTS (0.800).
#    - **ex3** (cascade, d=11): CEDAR achieves perfect F1=1.000 (GES/LGES/TGES 0.949, CDNOTS 0.945, CDNOTS+ 0.964). The chain structure and AR self-loops in the ground truth align exactly with CEDAR's pairwise design and `assume_ar1=True`.
#    - **Henon** (nonlinear, dense, d=5): CDNOTS (nonlinear) dominates at 0.828. Coupled-map dynamics violate the AR(1) assumption; lag selection misidentifies the dominant lag and Cond2 is unreliable (CEDAR 0.560 linear, 0.667 nonlinear).
#
# 2. **CDNOTS+ advantage is dataset-dependent.**
#    Helps on moderately dense linear graphs (ex2: +0.070 over CDNOTS) via MCI conditioning, and now also helps on the sparse cascade (ex3: +0.019). Hurts on dense nonlinear graphs (Henon: −0.207 vs CDNOTS nonlinear) where parent estimation errors compound.
#
# 3. **Nonlinear CI tests strongly help on coupled systems.**
#    CDNOTS gains +0.196 on ex1 and +0.309 on Henon from switching to SplitKCI. CEDAR gains +0.190 on ex1 and +0.107 on Henon.
#
# 4. **CEDAR lag selection uses partial distance correlation.**
#    `partial_dcor` residualises each target on its own lag-1 via OLS, then computes unbiased dcor against the residual. This removes the target's AR contribution before scoring cross-variable dependence. It assumes AR(1) for the target; results may degrade when the target has AR(p>1) dynamics.
#
# 5. **CEDAR self-loops are assumed, not discovered.** `assume_ar1=True` unconditionally adds one self-loop per variable. When the ground truth has self-loops (ex3), these are true positives. When it lacks them (ex2, Henon), they become false positives. All methods here are evaluated on the full ground truth including self-loops.
#
# 6. **GES, LGES and TGES are identical on all four datasets here — but not in general.** All three run on the lag-embedded matrix under the same temporal background knowledge (a variable can only cause another at an equal or later time step). LGES adds SafeInsert and ConservativeInsert on top; TGES adds a tier-based orientation pass on top of LGES. On these four small, clean systems all three land on the same F1/Precision/TPR/SHD, because plain GES's own search already reaches the fixed point the extra machinery is there to reach for it. That is a property of these datasets, not of the methods: at `d=15`, `T=300`, with contemporaneous edges mixed into the lagged structure (see the demonstration above), GES adds 7 spurious edges LGES's ConservativeInsert withholds (F1 0.870 → 0.936, SHD 15 → 7).
#
# 7. **VARLiNGAM, LASSO-VAR, and Granger lag far behind.** VARLiNGAM achieves high recall but poor precision; non-Gaussian identifiability rarely holds. LASSO-VAR and Granger produce nearly identical results and are dominated by constraint-based and CEDAR methods on every dataset.
#
# > **Note on reproducibility:** CDNOTS+'s numbers above reflect its current defaults (see the CDNOTS vs. CDNOTS+ notebook). CEDAR's own numbers can also shift slightly between runs of this notebook: CEDAR's `SplitKCIGPU`/permutation-based lag selection draws from PyTorch's global RNG, so anything that runs before it in the same kernel and consumes random draws (including CDNOTS+, whose behaviour depends on its own internal test count) can leave that RNG in a different state. This is a pre-existing property of the shared global generator, not a change to CEDAR's code or parameters.
#
# ---
#
# All baselines are available via `causalts.baselines`.
#
