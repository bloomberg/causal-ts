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
# # Real-World Discovery: Elbe River Network
#
# This notebook applies GRACE to real river flow data from the Elbe River main branch — a benchmark from the [CausalRivers](https://github.com/causalrivers/causalrivers) dataset.
#
# We use the **12 gauging stations** of the Elbe main branch, whose causal structure is known: water flows downstream. Flow is aggregated to 6-hour volumes and differenced; gaps are filled by interpolating each station's own series in time.
#
# We also demonstrate a **bootstrap approach**: running GRACE on several 1-year windows and keeping the edges found in most of them, to reduce sensitivity to individual seasons.
#
# ---
#
# ## Setup
#
# The next cell downloads the CausalRivers data (`product.zip`) into `CAUSALRIVERS_DIR` (default `/tmp/causalrivers`) if it is not already there.

# %%
import os
import subprocess
from pathlib import Path

DATA_DIR = Path(os.environ.get("CAUSALRIVERS_DIR", "/tmp/causalrivers"))

if not (DATA_DIR / "product" / "rivers_ts_east_germany.csv").exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading benchmark data...")
    subprocess.run([
        "curl", "-L", "-o", str(DATA_DIR / "product.zip"),
        "https://github.com/CausalRivers/benchmark/releases/download/First_release/product.zip"
    ], check=True)
    subprocess.run(["unzip", "-q", str(DATA_DIR / "product.zip"), "-d", str(DATA_DIR)],
                   check=True)
    print("Setup complete.")
else:
    print("CausalRivers data already downloaded")

# %%
import warnings
warnings.filterwarnings('ignore')

import pickle
import time
import numpy as np
import pandas as pd

from causalts import run_grace
from causalts.grace.gated_discovery import run_ci_skeleton

# %% [markdown]
# ## 1. Load the Elbe River Data
#
# We use the Elbe main branch (d=12 stations). Flow is averaged over 6-hour intervals (proportional to the 6-hour volume) and first-differenced. Gaps are filled by time interpolation within each station's series and flood spikes are clipped with a MAD rule. The river graph is the ground truth, so it is not used to fill gaps.

# %%
ELBE_STATIONS = [166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177]
RESAMPLE_H = 6

full_G = pickle.load(open(DATA_DIR / "product" / "rivers_east_germany.p", "rb"))
ts_raw = pd.read_csv(DATA_DIR / "product" / "rivers_ts_east_germany.csv",
                     index_col=0, parse_dates=True)
stations = [n for n in ELBE_STATIONS if str(n) in ts_raw.columns]
ts = ts_raw[[str(n) for n in stations]].resample(f"{RESAMPLE_H}h").mean()

# Fill gaps by time interpolation within each station's series
n_nan = int(ts.isna().sum().sum())
ts = ts.interpolate(method="time", limit_direction="both")
print(f"Interpolated {n_nan} missing values")

# MAD clipping of flood spikes
for col in ts.columns:
    s = ts[col]
    med, mad = s.median(), np.median(np.abs(s - s.median()))
    if mad > 1e-10:
        ts[col] = s.clip(med - 5 * mad * 1.4826, med + 5 * mad * 1.4826)

# First difference of the 6-hour volumes
df_river = ts.diff().dropna()
d = len(stations)
df_river.columns = list(range(d))
var_names = [f"S{n}" for n in stations]

# Ground truth: truth[cause, effect] = 1 for each river reach
node_idx = {n: i for i, n in enumerate(stations)}
truth = np.zeros((d, d), dtype=np.int8)
for u, v in full_G.subgraph(stations).edges():
    truth[node_idx[u], node_idx[v]] = 1

print(f"Loaded: d={d} stations, T={len(df_river)} samples ({RESAMPLE_H}h resolution)")
print(f"Date range: {df_river.index[0].date()} – {df_river.index[-1].date()}")
print(f"True edges: {int(truth.sum())}")
print(f"Stations: {var_names}")

# %% [markdown]
# ## 2. Scoring
#
# Water takes a varying number of 6-hour steps to travel between stations, so the true lag of each edge is not known. We score the **summary graph**: a pair counts as an edge if the method finds it at any lag 1..`MAX_LAG`, and self-loops are not scored.

# %%
MAX_LAG = 3
off_diag = ~np.eye(d, dtype=bool)

def summary(G):
    """(d', d', L+1) graph -> (d, d) summary over lags 1..L, self-loops removed."""
    P = (np.asarray(G)[:d, :d, 1:] != 0).any(axis=2)
    return P & off_diag

def score(P):
    tp = int((P & (truth == 1)).sum())
    fp = int((P & (truth == 0) & off_diag).sum())
    fn = int((~P & (truth == 1)).sum())
    return {"TP": tp, "FP": fp, "FN": fn,
            "Precision": tp / max(tp + fp, 1), "Recall": tp / max(tp + fn, 1),
            "F1": 2 * tp / max(2 * tp + fp + fn, 1)}

def report(name, m):
    print(f"{name}: TP={m['TP']}  FP={m['FP']}  FN={m['FN']}  "
          f"Precision={m['Precision']:.3f}  Recall={m['Recall']:.3f}  F1={m['F1']:.3f}")


# %% [markdown]
# ## 3. Candidate Graph
#
# As in the GRACE paper's river experiment, GRACE starts from a shallow, high-recall CDNOTS graph: partial correlation, α=0.05, conditioning sets of size at most one (`run_ci_skeleton`). Without a `candidate_graph`, `run_grace` would build a full-depth CDNOTS graph instead.

# %%
t0 = time.time()
cand = run_ci_skeleton(df_river, max_lag=MAX_LAG, verbose=False)
m_cand = score(summary(cand))
print(f"({time.time() - t0:.0f}s)")
report("Candidate graph", m_cand)

# %% [markdown]
# ## 4. GRACE
#
# GRACE refines the candidate graph with its defaults: one grow round that can add lagged edges, Hard Concrete gates on the cross edges, an F-test for self-loops, and an effect floor that drops edges with a negligible partial $R^2$.

# %%
t0 = time.time()
res_grace = run_grace(
    df_river, max_lag=MAX_LAG, candidate_graph=cand, model_seed=0, verbose=False
)
t_grace = time.time() - t0

m_grace = score(summary(res_grace.cg_tig))
print(f"({t_grace:.0f}s)")
report("GRACE", m_grace)

# %% [markdown]
# ## 5. Bootstrap GRACE (Temporal Robustness)
#
# River data is nonstationary: different seasons have different flow patterns. We run the same pipeline (candidate graph, then GRACE) on several random 1-year windows and record how often each station pair is selected (at any lag, as in the scoring). Pairs selected in at least half of the windows are kept.

# %%
SAMPLES_PER_YEAR = int(365.25 * 24 / RESAMPLE_H)
N_BOOTSTRAP = 5
BOOTSTRAP_SEED = 123
STABILITY_THRESHOLD = 0.5   # keep edges selected in at least half of the windows

from causalts import temporal_bootstrap

def _run_grace(sub_df):
    sk = run_ci_skeleton(sub_df, max_lag=MAX_LAG, verbose=False)
    res = run_grace(
        sub_df, max_lag=MAX_LAG, candidate_graph=sk, model_seed=0, verbose=False
    )
    return (np.asarray(res.cg_tig) != 0).astype(float)

boot = temporal_bootstrap(
    df_river, _run_grace,
    n_bootstrap=N_BOOTSTRAP,
    window_frac=SAMPLES_PER_YEAR / len(df_river),
    seed=BOOTSTRAP_SEED,
    show_progress=False,
)
# share of windows selecting each station pair (at any lag, as in the scoring above)
sel_freq = np.mean([summary(g) for g in boot["graphs"]], axis=0)
print(f"Completed {boot['n_success']}/{N_BOOTSTRAP} runs  (window {boot['window_size']} rows)")

m_boot = score((sel_freq >= STABILITY_THRESHOLD) & off_diag)
report(f"GRACE-Bootstrap (threshold={STABILITY_THRESHOLD})", m_boot)

# %% [markdown]
# ## 6. Summary

# %%
rows = [("Candidate graph", m_cand), ("GRACE (single run)", m_grace),
        ("GRACE-Bootstrap", m_boot)]
summary_table = pd.DataFrame(
    [{"Method": name, **{k: m[k] for k in ("Precision", "Recall", "F1", "TP", "FP", "FN")}}
     for name, m in rows]
).set_index("Method")
summary_table.round(3)

# %% [markdown]
# ## Key Takeaways
#
# - **The candidate graph has full recall but low precision** on this river: it contains all 11 true edges and 44 false ones (precision 0.20).
# - **GRACE keeps all 11 true edges and only 6 false ones**, against 44 in the candidate graph (precision 0.65, F1 0.79).
# - **The bootstrap** over five 1-year windows, keeping station pairs selected in at least half of them, keeps all 11 true edges and 8 false ones (F1 0.73), so here it does not improve on the single full-series run. What the bootstrap adds is the per-pair selection frequency (`sel_freq`), which shows how stable each edge is across seasons.
# - **`STABILITY_THRESHOLD`** sets the precision/recall trade-off: a higher threshold keeps fewer edges.
