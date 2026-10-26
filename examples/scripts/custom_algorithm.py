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
# # Plugging a Custom Algorithm into causal-ts
#
# causal-ts is designed so that any causal discovery algorithm can be registered in three steps:
#
# 1. **Subclass `CausalResult`** — set `cg_tig`, `var_names`, `_df`, `_scm_cache`.
# 2. **Decorate with `@register_algorithm("name")`** — makes the algorithm available to
#    `list_algorithms()` and `run_algorithm(...)` in-process. Add an entry point (Step 6)
#    to reach the `causal-ts` command as well.
# 3. **Return your result object** — you automatically inherit `.plot()`, `.estimate_effect()`,
#    `.fit_scm()`, `.counterfactual()`, `.attribute_anomaly()`, and all other DoWhy methods.
#
# This notebook walks through a complete worked example: a **Granger causality** algorithm
# wrapped using this interface.

# %%
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from causalts import register_algorithm, list_algorithms, CausalResult
from causalts.utils import evaluate_graph
from causalts.synthetic_data.synthetic_datasets import load_dataset
import causalts.plotting as tp

# Datasets used throughout
data_ex2 = load_dataset('ex2', seed=42, T=300)
df_ex2, gt_ex2 = data_ex2['df'], data_ex2['ground_truth']

print("Before registration:", list_algorithms())


# %% [markdown]
# ## Step 1 — Define a Result class
#
# Subclass `CausalResult` and set the four required attributes.
# Everything else — plotting, DoWhy, CLI saving — is inherited.

# %%
class GrangerResult(CausalResult):
    """Result from Naive Granger discovery."""

    def __init__(self, graph, var_names, df, *, aic=None, max_lag_fit=None):
        self.cg_tig     = graph               # (d, d, max_lag+1) — REQUIRED
        self.var_names  = list(var_names)     # list[str]         — REQUIRED
        self._df        = df                  # pd.DataFrame      — REQUIRED
        self._scm_cache = {}                  # empty dict        — REQUIRED
        # algorithm-specific extras (optional)
        self.aic         = aic
        self.max_lag_fit = max_lag_fit

    def __repr__(self):
        d = len(self.var_names)
        edges = int((self.cg_tig == 1).sum())
        return (f"GrangerResult(vars={d}, edges={edges}, "
                f"max_lag_fit={self.max_lag_fit}, aic={self.aic:.1f})"
                if self.aic else f"GrangerResult(vars={d}, edges={edges})")



# %% [markdown]
# ## Step 2 — Implement and register the algorithm
#
# The function signature must accept `df`, `ci_test`, and `max_lag` as positional
# arguments (so the registry can call it uniformly). Extra parameters go in `**kwargs`
# or as keyword-only args.

# %%
@register_algorithm("granger")
def run_granger(df, ci_test=None, max_lag=3, *, alpha=0.05, **kwargs):
    """Naive Granger causality via OLS VAR regression.

    Fits a VAR(max_lag) model with statsmodels and keeps edges where the
    coefficient p-value is below *alpha*. This is a deliberately simple
    algorithm — the point is to show the integration, not the statistics.

    Parameters
    ----------
    df : pd.DataFrame
    ci_test : ignored — Granger uses OLS internally
    max_lag : int
    alpha : float
        P-value threshold for coefficient significance.
    """
    from statsmodels.tsa.api import VAR

    d = df.shape[1]
    mdl   = VAR(df.values).fit(maxlags=max_lag, trend='n')
    n_fit = mdl.k_ar                               # lags actually fitted

    graph = np.zeros((d, d, max_lag + 1), dtype=np.int8)
    for lag in range(1, min(n_fit, max_lag) + 1):
        # pvalues block for this lag: rows = causes, cols = effects (statsmodels convention)
        block = mdl.pvalues[(lag - 1) * d : lag * d, :]   # (d_causes, d_effects)
        for cause in range(d):
            for effect in range(d):
                if cause != effect and block[cause, effect] <= alpha:
                    graph[cause, effect, lag] = 1

    return GrangerResult(
        graph=graph,
        var_names=list(df.columns),
        df=df,
        aic=mdl.aic,
        max_lag_fit=n_fit,
    )

print("After registration:", list_algorithms())

# %% [markdown]
# ## Step 3 — Run and evaluate

# %%
res = run_granger(df_ex2, max_lag=3, alpha=0.05)
print(res)

m = evaluate_graph(res.cg_tig, gt_ex2)
print(f"F1={m['F1']:.3f}  Precision={m['Precision']:.3f}  Recall={m['TPR']:.3f}  SHD={m['SHD']}")

# %% [markdown]
# ## Step 4 — Plotting and comparison (inherited for free)
#
# Because `GrangerResult` subclasses `CausalResult`, `.plot()` and `compare_graphs()`
# work without any extra code.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

_, _, pos = tp.plot_graph(
    gt_ex2, var_names=list(df_ex2.columns),
    arrow_linewidth=3, link_label_fontsize=8, node_size=0.2,
    show_colorbar=False, fig_ax=(fig, axes[0]), return_pos=True,
)
axes[0].set_title("True Graph", fontsize=12)

tp.compare_graphs(
    gt_ex2, res.cg_tig,
    var_names=list(df_ex2.columns), node_pos=pos,
    arrow_linewidth=3, link_label_fontsize=8, node_size=0.2,
    fig_ax=(fig, axes[1]), legend_loc=None,
)
axes[1].set_title(
    f"Granger (F1={m['F1']:.2f}  SHD={m['SHD']})", fontsize=12
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 5 — DoWhy bridge (inherited for free)
#
# Effect estimation, counterfactuals, and root cause analysis work on any
# `CausalResult` subclass. Install DoWhy with `pip install causalts[dowhy]`.

# %%
try:
    effect = res.estimate_effect(
        treatment='X3', outcome='X4', treatment_lag=1,
        method='backdoor.linear_regression',
        confidence_intervals=False,
    )
    print(f"ATE  X3(t-1) → X4 : {effect.value:+.4f}")
except ImportError:
    print("DoWhy not installed. Run:  pip install 'causalts[dowhy]'")

# %% [markdown]
# ## Step 6 — CLI integration
#
# `@register_algorithm` registers within the **running process**, so it covers the Python
# API and any script that imports your module:
#
# ```python
# from causalts.algorithms import run_algorithm
# import my_granger_plugin  # triggers registration
#
# res = run_algorithm("granger", df=df, ci_test=None, max_lag=3)
# ```
#
# The installed `causal-ts` command is a **separate process** that never imports your code,
# so the decorator alone cannot reach it. To make your algorithm available there, package
# it and advertise the `causalts.algorithms` entry point in your own `pyproject.toml`:
#
# ```toml
# # my_granger_plugin/pyproject.toml
# [project.entry-points."causalts.algorithms"]
# granger = "my_granger_plugin.plugin:run_granger"
# ```
#
# ```bash
# pip install -e .          # your plugin package
# causal-ts discover data.csv --algorithm granger --max-lag 3
# ```
#
# causal-ts discovers installed entry points on first use, so no import is needed on the
# caller's side — `granger` simply appears in `--algorithm`, and `list_algorithms()`
# reports it. A plugin that fails to import is warned about and skipped rather than
# breaking the CLI, and it can never shadow a built-in name.
#
# > **Signature contract.** The CLI calls your function as
# > `fn(df=..., ci_test=..., max_lag=...)`, so it must accept those keyword names (extra
# > options via `**kwargs`). Ignore `ci_test` if your method doesn't use one, as Granger
# > does above.
#

# %% [markdown]
# ## Summary — Minimal interface checklist
#
# To integrate any causal discovery algorithm:
#
# ```python
# from causalts import register_algorithm, CausalResult
# import numpy as np
#
# class MyResult(CausalResult):
#     def __init__(self, graph, df, var_names):
#         self.cg_tig     = graph           # np.ndarray (d, d, max_lag+1), dtype int8
#         self.var_names  = list(var_names) # list[str]
#         self._df        = df              # pd.DataFrame — needed for DoWhy bridge
#         self._scm_cache = {}              # empty dict
#
# @register_algorithm("my_algo")
# def run_my_algo(df, ci_test, max_lag, **kwargs):
#     d = df.shape[1]
#     graph = np.zeros((d, d, max_lag + 1), dtype=np.int8)
#     # ... your discovery logic ...
#     return MyResult(graph, df, list(df.columns))
# ```
#
# **Inherited for free:**
# - `result.plot(**kwargs)` — network graph visualization
# - `result.estimate_effect(treatment, outcome, ...)` — DoWhy ATE
# - `result.fit_scm()` / `result.counterfactual(...)` — structural causal model
# - `result.attribute_anomaly(...)` — Shapley root cause attribution
# - `causal-ts discover --algorithm my_algo data.csv` — CLI (once the entry point in Step 6 is declared)
#
# **Optional overrides:**
# - `__repr__` — custom summary string
# - `plot()` — custom plot defaults (e.g., `show_colorbar=False`)
# - `_bridge()` — custom DoWhy graph preparation (see `CedarResult` for an example
#   that restores implicit AR(1) self-lags before DoWhy sees the graph)
