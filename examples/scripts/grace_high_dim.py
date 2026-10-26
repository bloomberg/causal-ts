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
# This notebook demonstrates GRACE (Gated Refinement After Constraint-based Estimation) on a 30-variable causal system — a regime where constraint-based methods like CDNOTS start to degrade while GRACE maintains high precision.
#
# Benchmark results across increasing dimensionality (sparse SCP, T=1000, averaged over 10 seeds):
#
# | d | CDNOTS F1 | GRACE F1 | GRACE Precision |
# |---|-----------|----------|-----------------|
# | 20 | 0.844 | 0.868 | 95.0% |
# | 30 | 0.816 | 0.893 | — |
# | 50 | 0.742 | 0.915 | 96.7% |
# | 70 | 0.665 | 0.927 | 97.7% |
# | 100 | 0.531 | 0.920 | 97.4% |
#
# **Key insight:** At d=100, GRACE nearly doubles F1 over CDNOTS (0.92 vs 0.53) with 97.4% precision and SHD reduced by 90% (281 → 27).

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
from causalts.grace import run_cdnots_gated
from causalts.utils import evaluate_graph

# %% [markdown]
# ## 1. Generate Data
#
# Sparse mixed linear/nonlinear SCP: d=30 variables, T=1000 time points, max_lag=5.
# - `density="sparse"` — ~1.3–2.2 edges per variable
# - `dependency_funcs=("lin", "lin", "nl1", "nl2")` — mixed linear/nonlinear
# - `seed=0` — first of 10 seeds used in benchmarks

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
# GRACE uses the CDNOTS skeleton as a high-recall candidate set, then trains independent Hard Concrete gates to prune false positives.
#
# Lambda is selected via an adaptive formula: `max(0.007 + 0.16·ρ_S, 1/d + 4/T - 0.4·ρ_S^0.8)` where ρ_S is skeleton density.

# %%
t0 = time.time()
_gr = run_cdnots_gated(
    df=df,
    max_lag=5,
    alpha=0.05,
    # skeleton=graph_cdnots, # we can provide the skeleton from cdnots
    gate_threshold=0.5,
    max_epochs=150,
    model_seed=42,
    verbose=True,
    ground_truth=ground_truth,
)
G_grace     = _gr.cg_tig
gate_values = _gr.gate_values
info        = {"lambda_l0": _gr.lambda_l0,
               "skeleton_density": float(_gr.skeleton.astype(bool).mean()) if _gr.skeleton is not None else None}
t_grace = time.time() - t0

metrics_grace = evaluate_graph(G_grace, ground_truth)
print(f"\nGRACE ({t_grace:.0f}s):")
print(f"  F1={metrics_grace['F1']:.3f}  Precision={metrics_grace['Precision']:.3f}"
      f"  Recall={metrics_grace['TPR']:.3f}  SHD={metrics_grace['SHD']}")
print(f"  Lambda used: {info['lambda_l0']:.4f}  Skeleton density: {info['skeleton_density']:.3f}")

# %% [markdown]
# ## 4. Gate Value Heatmap
#
# Each gate value indicates confidence in an edge's existence. Values > 0.5 are included in the final graph.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

gate_sum = gate_values.sum(axis=2)  # (d, d)
im = axes[0].imshow(gate_sum, cmap="RdYlGn", vmin=0, vmax=gate_values.shape[2])
axes[0].set_title("Gate values (summed over lags)\nGreen = confident causal edges", fontsize=12)
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
# Hard Concrete gates produce a bimodal distribution: true causal edges concentrate near 1, false edges near 0, with almost no overlap at the 0.5 decision boundary. This makes the threshold robust without tuning.

# %%
flat_gates = gate_values.flatten()
flat_truth = ground_truth.flatten()

true_gates  = flat_gates[flat_truth == 1]
false_gates = flat_gates[flat_truth == 0]

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(false_gates, bins=40, alpha=0.75, color="#ef4444",
        label=f"Non-causal edges (n={len(false_gates)})")
ax.hist(true_gates,  bins=40, alpha=0.75, color="#22c55e",
        label=f"True causal edges (n={len(true_gates)})")
ax.axvline(0.5, color="black", linestyle="--", linewidth=2, label="Decision boundary (0.5)")
ax.set_xlabel("Gate value", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_yscale("log")
ax.set_title("Hard Concrete gate distribution at d=30", fontsize=13)
ax.legend(fontsize=11)
plt.tight_layout()
plt.show()

print(f"True edges with gate > 0.5: {(true_gates > 0.5).mean()*100:.1f}%")
print(f"False edges with gate < 0.5: {(false_gates < 0.5).mean()*100:.1f}%")

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
# - **Use GRACE when d > 20** — gating consistently improves precision with minimal recall loss
# - **Lambda adapts automatically** to d, T, and skeleton density — no manual tuning
# - **Gate values are interpretable**: > 0.8 = high-confidence; 0.5–0.8 = borderline; < 0.5 = pruned
# - **Bimodal separation** confirms the 0.5 threshold is reliable without calibration
#
# For d ≥ 30, run on GPU (`device='cuda'` or `device='mps'`) to reduce training time.
