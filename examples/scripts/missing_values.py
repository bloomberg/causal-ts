# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: py311
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Missing Values
#
# `causal-ts` handles NaN entries in the input DataFrame through the `impute` parameter. There are three strategies:
#
# | Strategy | `impute=` | When to use |
# |---|---|---|
# | Pairwise-complete | `None` (default) or `"pairwise_complete"` | Light, random missingness: each CI test only loses the rows it touches |
# | VAR-EM | `"var_em"` | Heavier missingness, or algorithms that need complete data |
# | Causal iterative | `"causal_iterative"` | As VAR-EM, then refines the fill using the discovered graph (CDNOTS / CDNOTS+ only) |
#
# **Pairwise-complete**: each CI test drops the rows where any of X, Y or Z is NaN for that test. No values are invented, but every lagged copy of a missing entry costs a row, so the effective sample size shrinks quickly as missingness and `max_lag` grow.
#
# **VAR-EM**: fits a VAR(p) model by EM, fills the NaN positions with fitted values and repeats until convergence. It assumes no causal direction.
#
# **Causal iterative**: starts from the VAR-EM fill, then alternates between running CDNOTS to get a graph and re-imputing each variable from its Markov blanket (parents, children and parents of children) with Ridge regression.
#
# Not every algorithm accepts every strategy:
#
# | Algorithm | `None` / `"pairwise_complete"` | `"var_em"` | `"causal_iterative"` |
# |---|---|---|---|
# | CDNOTS, CDNOTS+ | yes | yes | yes |
# | CEDAR | yes | yes | raises `ValueError` |
# | GRACE | raises `ValueError` if NaNs are present (the neural gates need complete data) | yes | falls back to VAR-EM with a warning |
# | REAP | raises `ValueError` if NaNs are present (the regression refinement needs complete data) | yes | falls back to VAR-EM with a warning |
#
# The examples below compare the three strategies for CDNOTS and CDNOTS+ after removing each entry independently with probability 0.10 (missing completely at random), so about 10% of the values are missing.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import causalts.plotting as tp
from causalts import run_cdnots, run_cdnots_plus
from causalts.ci_tests import ParCorrGPU, SplitKCIGPU
from causalts.synthetic_data.synthetic_datasets import ex2, henon_chain
from causalts.utils import evaluate_graph

T_SMALL, T_LARGE = 200, 500

res_ex2_small  = ex2(seed=42, T=T_SMALL)
res_ex2_large  = ex2(seed=42, T=T_LARGE)
res_hc_small   = henon_chain(seed=42, T=T_SMALL)
res_hc_large   = henon_chain(seed=42, T=T_LARGE)

for res, t in [
    (res_ex2_small, "T=200"), (res_ex2_large, "T=500"),
    (res_hc_small,  "T=200"), (res_hc_large,  "T=500"),
]:
    print(f"{res['name']} {t}: d={res['df'].shape[1]}, T={len(res['df'])}, max_lag={res['max_lag']}")


# %%
def introduce_nan(df, frac=0.10, seed=99):
    rng = np.random.default_rng(seed)
    arr = df.values.copy()
    mask = rng.random(arr.shape) < frac
    arr[mask] = np.nan
    return pd.DataFrame(arr, columns=df.columns), mask.sum()

_NAN_STRATEGIES = [
    ("Pairwise",  None,               {}),
    ("VAR-EM",    "var_em",           {"p": 1}),
    ("Causal It", "causal_iterative", {"n_iter": 2}),
]
STRAT_COLS   = ["clean", "Pairwise", "VAR-EM", "Causal It"]
STRAT_LABELS = STRAT_COLS
METHODS      = ["cdnots+", "cdnots"]


def run_scenario(df_clean, df_nan, max_lag, make_ci, alpha=0.01):
    """Run both cdnots+ and cdnots for all strategies. make_ci() returns a fresh CI test."""
    out = {}
    for method, runner in [("cdnots+", run_cdnots_plus), ("cdnots", run_cdnots)]:
        out[method] = {}
        out[method]["clean"] = runner(
            df=df_clean, indep_test=make_ci(), num_lags=max_lag,
            include_C=False, alpha=alpha,
        ).cg_tig
        for label, impute, ikw in _NAN_STRATEGIES:
            out[method][label] = runner(
                df=df_nan, indep_test=make_ci(), num_lags=max_lag,
                include_C=False, alpha=alpha,
                impute=impute, impute_kwargs=ikw,
            ).cg_tig
    return out


def plot_scenario(key, scenarios, figsize=(22, 9)):
    sc        = scenarios[key]
    true_g    = sc["true_graph"]
    var_names = sc["var_names"]
    n_cols    = 1 + len(STRAT_LABELS)

    fig, axes = plt.subplots(len(METHODS), n_cols, figsize=figsize)
    fig.suptitle(key, fontsize=15, fontweight="bold")

    _, _, pos = tp.plot_graph(
        graph=true_g, var_names=var_names,
        arrow_linewidth=3, show_colorbar=False,
        fig_ax=(fig, axes[0, 0]), return_pos=True,
    )
    axes[0, 0].set_title("True Graph", fontsize=11)

    for row_idx, method in enumerate(METHODS):
        if row_idx > 0:
            tp.plot_graph(
                graph=true_g, var_names=var_names, node_pos=pos,
                arrow_linewidth=3, show_colorbar=False,
                fig_ax=(fig, axes[row_idx, 0]),
            )
            axes[row_idx, 0].set_title("True Graph", fontsize=11)

        for col_idx, label in enumerate(STRAT_LABELS, start=1):
            g = sc["results"][method][label]
            m = evaluate_graph(g, true_g)
            tp.compare_graphs(
                graph_true=true_g, graph_discovered=g,
                var_names=var_names, node_pos=pos,
                arrow_linewidth=3, legend_loc=None,
                show_legend=(row_idx == 0 and col_idx == 1),
                fig_ax=(fig, axes[row_idx, col_idx]),
            )
            axes[row_idx, col_idx].set_title(
                f"{label}\nF1={m['F1']:.3f}, SHD={m['SHD']}", fontsize=10
            )

        axes[row_idx, 0].set_ylabel(method, fontsize=13, fontweight="bold", labelpad=10)

    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## Example 1: VAR — Linear, d=6, max_lag=3
#
# Linear VAR model with 6 variables. CI test: **PartialCorr** (correctly specified for linear data).

# %%
make_parcorr = lambda: ParCorrGPU(data=np.zeros((2, 2)))

scenarios_ex2 = {}
for t_label, res in [("T=200", res_ex2_small), ("T=500", res_ex2_large)]:
    df_clean = res["df"]
    df_nan, n_missing = introduce_nan(df_clean)
    key = f"{res['name']} {t_label}"
    print(f"Running {key}  ({n_missing}/{df_clean.size} NaN = {100*n_missing/df_clean.size:.1f}%)")
    scenarios_ex2[key] = dict(
        results    = run_scenario(df_clean, df_nan, res["max_lag"], make_parcorr),
        true_graph = res["ground_truth"],
        var_names  = res["df"].columns.tolist(),
    )

print()
for key, sc in scenarios_ex2.items():
    true_g = sc["true_graph"]
    print(f"  {key}")
    print(f"  {'Strategy':<12} {'cdnots+ F1':>11} {'cdnots+ SHD':>12} {'cdnots F1':>10} {'cdnots SHD':>11}")
    print("  " + "-" * 60)
    for col in STRAT_COLS:
        mp = evaluate_graph(sc["results"]["cdnots+"][col], true_g)
        mb = evaluate_graph(sc["results"]["cdnots"][col],  true_g)
        print(f"  {col:<12} {mp['F1']:>11.3f} {mp['SHD']:>12} {mb['F1']:>10.3f} {mb['SHD']:>11}")
    print()

# %% [markdown]
# On the linear VAR, pairwise deletion reproduces the clean-data result at both sample sizes. VAR-EM and causal-iterative imputation do no better and sometimes worse: at T=500 CDNOTS+ drops from F1 0.952 to 0.870 (VAR-EM) and 0.833 (causal iterative). Imputed entries are fitted values without noise, which slightly distorts the partial correlations, while at this level of random missingness pairwise deletion loses little information.

# %%
for key in scenarios_ex2:
    plot_scenario(key, scenarios_ex2)

# %% [markdown]
# ## Example 2: Hénon Chain — Nonlinear, d=5, max_lag=2
#
# Coupled Hénon maps with quadratic dynamics (`x[t] = 1.4 - mix²+ 0.3·x[t-2]`). PartialCorr is misspecified here; comparing against **SplitKCIGPU** (kernel nonlinear test, 10–20× faster than KCI, no functional form assumption).

# %%
make_splitkci = lambda: SplitKCIGPU(data=np.zeros((2, 2)))
hc_name = res_hc_small["name"]  # "Hénon Chain"

scenarios_hc = {}
for t_label, res in [("T=200", res_hc_small), ("T=500", res_hc_large)]:
    df_clean   = res["df"]
    df_nan, _  = introduce_nan(df_clean)
    max_lag    = res["max_lag"]
    true_graph = res["ground_truth"]
    var_names  = res["df"].columns.tolist()

    for ci_name, make_ci in [("ParCorr", make_parcorr), ("SplitKCI", make_splitkci)]:
        key = f"{hc_name} {t_label} {ci_name}"
        print(f"Running {key} ...")
        scenarios_hc[key] = dict(
            results    = run_scenario(df_clean, df_nan, max_lag, make_ci),
            true_graph = true_graph,
            var_names  = var_names,
        )

print("\nDone.")

# %%
for t_label in ["T=200", "T=500"]:
    key_pc = f"{hc_name} {t_label} ParCorr"
    key_sk = f"{hc_name} {t_label} SplitKCI"
    true_g = scenarios_hc[key_pc]["true_graph"]

    print(f"\n  {hc_name} {t_label}")
    print(f"  {'Strategy':<12} {'method':<9} {'ParCorr F1':>11} {'ParCorr SHD':>12} {'SplitKCI F1':>12} {'SplitKCI SHD':>13}")
    print("  " + "-" * 73)
    for method in METHODS:
        for col in STRAT_COLS:
            mp = evaluate_graph(scenarios_hc[key_pc]["results"][method][col], true_g)
            ms = evaluate_graph(scenarios_hc[key_sk]["results"][method][col], true_g)
            print(f"  {col:<12} {method:<9} {mp['F1']:>11.3f} {mp['SHD']:>12} {ms['F1']:>12.3f} {ms['SHD']:>13}")
        print()

# %% [markdown]
# On the Hénon chain no strategy is consistently best. The ranking changes with T, method and CI test: causal-iterative imputation gives the best ParCorr result for CDNOTS+ at both sample sizes but the worst SplitKCI result for CDNOTS+ at T=200 (0.300), and VAR-EM is the best SplitKCI strategy for CDNOTS+ at T=200 but the worst at T=500. With 16 true edges and a single seed, one or two edges more or less move F1 by 0.05–0.10, so read these as single runs rather than a ranking. The CI test matters more than the strategy: CDNOTS with SplitKCI is the strongest combination on clean data at both sample sizes (F1 0.720 and 0.769), and stays at or above 0.52 under every strategy.

# %%
for t_label in ["T=200", "T=500"]:
    plot_scenario(f"{hc_name} {t_label} SplitKCI", scenarios_hc)
