# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: swneural (3.12.12)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils.plt_setup_utils import setup_matplotlib_for_paper

setup_matplotlib_for_paper()

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
RESULTS_DIR = BASE_DIR / "results"
FIG3_OUTPUT_DIR = BASE_DIR.parent / "Fig3" / "outputs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# %%
filepath = OUTPUT_DIR / "expressivity_sweep.npz"
data = np.load(filepath)

beta = data["beta"]
eirat = data["eirat"]
le_mean = data["le_mean"]
lg_mean = data["lg_mean"]
# %%
fig, ax = plt.subplots(figsize=(4, 4))
m0 = ax.pcolormesh(beta, eirat, lg_mean.T)
ax.set_xscale("log")
fig.colorbar(m0, ax=ax, orientation="horizontal", location="top", pad=0.08)
plt.savefig(RESULTS_DIR / "FigS4a.tiff", dpi=300, bbox_inches="tight")

# %%
data_filepath = FIG3_OUTPUT_DIR / "2D_perturb_sweep.npz"
data = np.load(data_filepath)

beta = data["beta"]
eirat = data["eirat"]
tau_c = data["tau_c"]
tau_p = data["tau_p"]
clst = data["clst"]
avpl = data["avpl"]

tau_c_mean = np.mean(tau_c, axis=(2, 3))
tau_p_mean = np.mean(tau_p, axis=(2, 3))

tau_c_mean_norm = tau_c_mean / np.max(tau_c_mean)
tau_p_mean_norm = tau_p_mean / np.max(tau_p_mean)

# Pool (2, 2) blocks: (60, 60) -> (30, 30)
pool_h = tau_c_mean_norm.shape[0] // le_mean.shape[0]
pool_w = tau_c_mean_norm.shape[1] // le_mean.shape[1]

if (tau_c_mean_norm.shape[0] % le_mean.shape[0] != 0) or (
    tau_c_mean_norm.shape[1] % le_mean.shape[1] != 0
):
    raise ValueError(
        f"Cannot pool tau_c_mean_norm shape {tau_c_mean_norm.shape} to {le_mean.shape}"
    )

tau_c_mean_norm_pooled = tau_c_mean_norm.reshape(
    le_mean.shape[0], pool_h, le_mean.shape[1], pool_w
).mean(axis=(1, 3))

# %%
lg_mean_flat = np.reshape(lg_mean, (-1,))
tau_c_mean_norm_flat = np.reshape(tau_c_mean_norm_pooled, (-1,))
fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(
    tau_c_mean_norm_flat,
    lg_mean_flat,
    s=20,
    facecolors="none",
    marker="o",
    linewidths=0.4,
    edgecolors="black",
)
plt.savefig(RESULTS_DIR / "FigS4b.tiff", dpi=300, bbox_inches="tight")
plt.close("all")
