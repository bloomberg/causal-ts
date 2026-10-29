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
# # GRACE for High-Dimensional Time Series (d=30)
#
# This notebook runs GRACE (Gated Refinement for Accurate Causal Edge discovery) on a 30-variable causal system, where constraint-based methods such as CDNOTS keep many false edges, and compares it with CDNOTS on the same data.

# %%
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from causalts.synthetic_data.synthetic_datasets import SCPGraphGenerator
from causalts.ci_tests import PartialCorr
from causalts import run_cdnots, run_cdnots_plus
from causalts import run_grace
from causalts.utils import evaluate_graph

# %% [markdown]
# ## 1. Generate Data
#
# Sparse mixed linear/nonlinear SCP: d=30 variables, T=1000 time points, max_lag=5.
# - `density="sparse"` — ~1.3–2.2 edges per variable
# - `dependency_funcs=("lin", "lin", "nl1", "nl2")` — mixed linear/nonlinear
# - `seed=0`

# %%
gen = SCPGraphGenerator(
    n_vars=30,
    max_lag=5,
    density="sparse",
    dependency_funcs=("lin", "lin", "nl1", "nl2"),
)
result = gen.sample(seed=0, T=1000, max_tries=500)

df = result["df"]
ground_truth = result["ground_truth"]  # shape (30, 30, 6)

n_edges = int(ground_truth.sum())
print(f"Dataset: {df.shape[0]} time points, {df.shape[1]} variables")
print(f"True causal edges: {n_edges}")

# %% [markdown]
# ## 2. CDNOTS Baseline
#
# Constraint-based discovery with partial correlation. At d=30, conditioning sets become large relative to T=1000, causing some precision loss.

# %%
ci_test = PartialCorr(np.ascontiguousarray(df.values))

t0 = time.time()
_r = run_cdnots(
    df=df, indep_test=ci_test, num_lags=5,
    include_C=True, alpha=0.05,
)
graph_cdnots = _r.cg_tig
t_cdnots = time.time() - t0

graph_cdnots = graph_cdnots[:30, :30, :]
metrics_cdnots = evaluate_graph(graph_cdnots, ground_truth)

print(f"CDNOTS ({t_cdnots:.0f}s):")
print(f"  F1={metrics_cdnots['F1']:.3f}  Precision={metrics_cdnots['Precision']:.3f}"
      f"  Recall={metrics_cdnots['TPR']:.3f}  SHD={metrics_cdnots['SHD']}")
print(f"  TP={metrics_cdnots['TP']}  FP={metrics_cdnots['FP']}  FN={metrics_cdnots['FN']}")

# %% [markdown]
# ## 3. GRACE
#
# GRACE with its defaults:
#
# 1. **Candidate graph.** A CDNOTS run (partial correlation, α=0.05) on the rank-transformed data, unless you pass your own `candidate_graph`.
# 2. **Gated model.** One Hard Concrete gate per candidate cross edge, trained with an $L_0$ penalty. Its strength λ comes from a bounded rule in d, T and the data's background lagged cross-correlation (`lambda_rule="xcorr"`).
# 3. **Grow round.** Lagged pairs outside the candidates are tested, for an effect on the target's mean and on its noise scale; hits join the candidates and the gates are retrained.
# 4. **Final graph.**
#    - Cross edges are cut from the gates (the `auto` cutoff).
#    - Self-loops come from an ungated autoregressive head and are kept by a partial F-test.
#    - A minimum effect per edge (`effect_floor`) removes cross edges that explain a negligible share of their target's variance.

# %%
t0 = time.time()
_gr = run_grace(
    df,
    max_lag=5,
    alpha=0.05,
    # candidate_graph=graph_cdnots,  # or start from another method's graph
    model_seed=42,
    verbose=False,
)
G_grace     = _gr.cg_tig
gate_values = _gr.gate_values
info        = {"lambda_l0": _gr.lambda_l0, "cutoff": _gr.gate_threshold,
               "grown": sum(h["n_added"] for h in _gr.grow_history or []),
               "effect_pruned": _gr.n_effect_pruned}
t_grace = time.time() - t0

metrics_grace = evaluate_graph(G_grace, ground_truth)
print(f"GRACE ({t_grace:.0f}s):")
print(f"  F1={metrics_grace['F1']:.3f}  Precision={metrics_grace['Precision']:.3f}"
      f"  Recall={metrics_grace['TPR']:.3f}  SHD={metrics_grace['SHD']}")
print(f"  lambda={info['lambda_l0']:.4f}  gate cutoff={info['cutoff']:.2f}  "
      f"candidates added by the grow round={info['grown']}  "
      f"edges removed by the effect floor={info['effect_pruned']}")

# %% [markdown]
# ## 4. Gate Value Heatmap
#
# Each gate value is the model's confidence in a candidate cross edge. Edges with a gate above the cutoff enter the graph unless the effect floor removes them; self-loops are not gated (their entries are 0) and are decided by the F-test.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

gate_sum = gate_values.sum(axis=2)  # (d, d)
im = axes[0].imshow(gate_sum, cmap="RdYlGn", vmin=0, vmax=gate_values.shape[2])
axes[0].set_title("Gate values (summed over lags)\nGreen = confident causal edges; "
                  "self-loops are F-tested, not gated (0 here)", fontsize=12)
axes[0].set_xlabel("Effect variable")
axes[0].set_ylabel("Cause variable")
plt.colorbar(im, ax=axes[0], label="Sum of gate values")

gt_sum = ground_truth.sum(axis=2)
axes[1].imshow(gt_sum, cmap="Blues", vmin=0, vmax=1)
axes[1].set_title("Ground truth edges", fontsize=12)
axes[1].set_xlabel("Effect variable")
axes[1].set_ylabel("Cause variable")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Gate Value Distribution
#
# The histogram shows the final gate value of every cross edge, with the cutoff GRACE chose from the data (the `auto` rule). Pairs that are not candidates have no gate (value 0).
#
# In this run, 38 of the 40 true cross edges are candidates and 26 of them clear the cutoff. Three non-causal edges also clear it; after the effect floor, the final graph keeps one false edge. The gates do not separate the two classes perfectly: most of the recall GRACE gives up comes from true cross edges whose gates stay below the cutoff.

# %%
# cross edges only: self-loops are decided by an F-test and have no gate value
cross = np.ones(gate_values.shape, dtype=bool)
cross[np.arange(gate_values.shape[0]), np.arange(gate_values.shape[0]), :] = False
flat_gates = gate_values[cross]
flat_truth = ground_truth[cross]

true_gates  = flat_gates[flat_truth == 1]
false_gates = flat_gates[flat_truth == 0]

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(false_gates, bins=40, alpha=0.75, color="#ef4444",
        label=f"Non-causal cross edges (n={len(false_gates)})")
ax.hist(true_gates,  bins=40, alpha=0.75, color="#22c55e",
        label=f"True causal cross edges (n={len(true_gates)})")
cut = info["cutoff"]
ax.axvline(cut, color="black", linestyle="--", linewidth=2, label=f"Gate cutoff ({cut:.2f})")
ax.set_xlabel("Gate value", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_yscale("log")
ax.set_title("Hard Concrete gate distribution at d=30", fontsize=13)
ax.legend(fontsize=11)
plt.tight_layout()
plt.show()

cand = _gr.candidate_graph[cross] != 0  # the refined candidate graph, incl. grow additions
print(f"True cross edges in the candidate set: {int((flat_truth[cand] == 1).sum())} of {len(true_gates)}")
print(f"  with gate above the cutoff: {int((true_gates > cut).sum())}")
print(f"Non-causal cross edges with gate above the cutoff: {int((false_gates > cut).sum())}")

# %% [markdown]
# ## 6. Summary

# %%
summary = pd.DataFrame({
    "Method": ["CDNOTS", "GRACE"],
    "F1": [f"{metrics_cdnots['F1']:.3f}", f"{metrics_grace['F1']:.3f}"],
    "Precision": [f"{metrics_cdnots['Precision']:.3f}", f"{metrics_grace['Precision']:.3f}"],
    "Recall": [f"{metrics_cdnots['TPR']:.3f}", f"{metrics_grace['TPR']:.3f}"],
    "SHD": [metrics_cdnots['SHD'], metrics_grace['SHD']],
    "TP/FP/FN": [
        f"{metrics_cdnots['TP']}/{metrics_cdnots['FP']}/{metrics_cdnots['FN']}",
        f"{metrics_grace['TP']}/{metrics_grace['FP']}/{metrics_grace['FN']}",
    ],
    "Runtime (s)": [f"{t_cdnots:.0f}", f"{t_grace:.0f}"],
})
print(summary.to_string(index=False))
print(f"\nTrue edges: {n_edges}")

# %% [markdown]
# ## Key Takeaways
#
# - **Precision.** On this d=30 system, GRACE keeps 1 false edge against CDNOTS's 26 (precision 0.705 → 0.982) and halves the SHD (34 → 16).
# - **Recall.** GRACE finds 55 of the 70 true edges against CDNOTS's 62 (recall 0.886 → 0.786); most of its misses are weak cross edges whose gates stay below the cutoff.
# - **No tuning.** λ (`lambda_rule="xcorr"`) and the gate cutoff are set from the data; nothing was tuned here.
# - **Gate values are not probabilities.** For a confidence measure, use selection frequency across subsamples or windows: `causalts.grace.run_stability_selection` or `causalts.temporal_bootstrap`.
# - **Runtime.** GRACE runs much longer than CDNOTS. It uses a GPU automatically when one is available (see the `device` argument).
