# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: swneural (3.12.3)
#     language: python
#     name: python3
# ---

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils.plt_setup_utils import setup_matplotlib_for_paper

# %%
setup_matplotlib_for_paper(figsize=(4.0, 2.8))

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

epses = [1e-3, 0.03, 0.05, 0.1, 0.2]

for eps in epses:
    filepath = OUTPUT_DIR / f"beta_perturb_sweep_eps{eps}.npz"
    data = np.load(filepath)

    beta = data["beta"]
    tau_c = data["tau_c"]
    tau_p = data["tau_p"]

    tau_c_mean_batch = np.mean(tau_c, axis=(2,))
    tau_p_mean_batch = np.mean(tau_p, axis=(2,))

    tau_c_mean = np.mean(tau_c_mean_batch, axis=1)
    tau_p_mean = np.mean(tau_p_mean_batch, axis=1)

    x_tau_c = np.repeat(beta, tau_c_mean_batch.shape[-1])
    y_tau_c = tau_c_mean_batch.flatten() / np.max(tau_c_mean_batch)

    x_tau_p = np.repeat(beta, tau_p_mean_batch.shape[-1])
    y_tau_p = tau_p_mean_batch.flatten() / np.max(tau_p_mean_batch)

    fig, ax = plt.subplots()

    ax.scatter(x_tau_c, y_tau_c, c="tab:red", alpha=0.2, s=20, label=r"$\tilde{\tau}$")
    ax.scatter(
        x_tau_p,
        y_tau_p,
        c="tab:blue",
        alpha=0.2,
        s=20,
        label=r"$\tilde{\tau}_{thresh}$",
    )

    ax.plot(
        beta,
        tau_c_mean / np.max(tau_c_mean_batch),
        ".-",
        color="tab:red",
        label=r"$\langle\tilde{\tau}\rangle$",
    )
    ax.plot(
        beta,
        tau_p_mean / np.max(tau_p_mean_batch),
        ".-",
        color="tab:blue",
        label=r"$\langle\tilde{\tau}_{thresh}\rangle$",
    )

    ax.set_xscale("log")
    ax.set_ylim(0, 1.2)

    fig.savefig(
        RESULTS_DIR / f"FigS1_eps{eps}.tiff",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)
