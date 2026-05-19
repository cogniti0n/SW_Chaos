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

setup_matplotlib_for_paper(figsize=(4.0, 3.5))

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# %%
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

Ns = [100, 200, 300, 400, 500]
results = []

for N in Ns:
    filepath = OUTPUT_DIR / f"2D_perturb_sweep_n{N}.npz"
    data = np.load(filepath)

    beta = data["beta"]
    eirat = data["eirat"]
    tau_c = data["tau_c"]
    tau_p = data["tau_p"]

    tau_c_mean = np.mean(tau_c, axis=(2, 3))
    tau_p_mean = np.mean(tau_p, axis=(2, 3))

    tau_c_mean_norm = tau_c_mean / np.max(tau_c_mean)
    tau_p_mean_norm = tau_p_mean / np.max(tau_p_mean)

    results.append((N, beta, eirat, tau_c_mean_norm, tau_p_mean_norm))

nrows = len(Ns)
fig, axes = plt.subplots(
    nrows=nrows, ncols=2, figsize=(12, 6 * nrows), sharex="col", sharey="row"
)

cmap = "Spectral"
norm = Normalize(vmin=0.0, vmax=1.0)
for i, (N, beta, eirat, tau_c_mean_norm, tau_p_mean_norm) in enumerate(results):
    ax_c = axes[i, 0]
    ax_p = axes[i, 1]

    ax_c.pcolormesh(
        beta, eirat, tau_c_mean_norm.T, cmap=cmap, norm=norm, shading="auto"
    )
    ax_p.pcolormesh(
        beta, eirat, tau_p_mean_norm.T, cmap=cmap, norm=norm, shading="auto"
    )

    ax_c.set_xscale("log")
    ax_p.set_xscale("log")

    ax_c.set_ylabel("Excitation Ratio")
    ax_c.set_title(rf"$\tau_c$ (N={N})")
    ax_p.set_title(rf"$\tau_p$ (N={N})")

    if i == nrows - 1:
        ax_c.set_xlabel(r"$\beta$")
        ax_p.set_xlabel(r"$\beta$")

    for tau_name, tau_values in [
        ("tau_c", tau_c_mean_norm),
        ("tau_p", tau_p_mean_norm),
    ]:
        panel_fig, panel_ax = plt.subplots()
        panel_ax.pcolormesh(
            beta, eirat, tau_values.T, cmap=cmap, norm=norm, shading="auto"
        )
        panel_ax.set_xscale("log")
        panel_fig.tight_layout()
        panel_fig.savefig(
            RESULTS_DIR / f"FigS2S3_{tau_name}_N{N}.tiff",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(panel_fig)
