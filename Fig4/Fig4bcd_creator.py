# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: swneural (3.11.9)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils.plt_setup_utils import setup_matplotlib_for_paper

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

setup_matplotlib_for_paper(figsize=(5, 3))

# %%
r2_hor1 = []
r2_hor5 = []
for seed in range(1, 11):
    lorenz_path = OUTPUT_DIR / (f"lorenz_sweep_seed{seed}.npz")
    data = np.load(lorenz_path)
    r2_hor1.append(data["hor1"])
    r2_hor5.append(data["hor2"])
r2_hor1 = np.asarray(r2_hor1)
r2_hor5 = np.asarray(r2_hor5)
logbeta = np.linspace(-4.0, 0.0, 30)
beta = 10**logbeta

# %%
plt.figure()
for i in range(10):
    plt.plot(beta, r2_hor1[i], ".", color="tab:red", alpha=0.3)
plt.plot(
    beta,
    np.nanmean(r2_hor1, axis=0),
    marker="o",
    linestyle="--",
    color="tab:red",
    label="1s",
)
for i in range(10):
    plt.plot(beta, r2_hor5[i], ".", c="tab:blue", alpha=0.3)
plt.plot(
    beta,
    np.nanmean(r2_hor5, axis=0),
    marker="o",
    linestyle="--",
    c="tab:blue",
    label="5s",
)
plt.xscale("log")
plt.ylim(1.8, 3.3)
plt.yticks([2.0, 2.5, 3.0])
plt.savefig(RESULTS_DIR / "Fig4d.tiff", dpi=300, bbox_inches="tight")

# %%
acc_store = []
for seed in range(1, 11):
    mnist_path = OUTPUT_DIR / (f"mnist_sweep_seed{seed}.npz")
    data = np.load(mnist_path)
    acc = data["acc"]
    acc_store.append(acc)
acc = np.asarray(acc_store)

# %%
plt.figure()
for i in range(10):
    plt.plot(beta, 1 - acc[i], ".", c="tab:green", alpha=0.3)
plt.plot(beta, 1 - np.nanmean(acc, axis=0), marker="o", linestyle="--", c="tab:green")
print(np.max(acc, axis=0)[0])
plt.ylim(0.13, 0.53)
plt.yticks([0.2, 0.3, 0.4, 0.5])
plt.xscale("log")
plt.savefig(RESULTS_DIR / "Fig4b.tiff", dpi=300, bbox_inches="tight")

# %%

rb = []
for seed in range(1, 11):
    rb_path = OUTPUT_DIR / f"remote_bandwidth_sweep_seed{seed}.npz"
    data = np.load(rb_path)
    beta = data["beta"]
    rmse_test = data["rmse_test"]

    rb.append(rmse_test)
rb = np.asarray(rb)

plt.figure()
for i in range(10):
    plt.plot(beta, rb[i], ".", c="tab:pink", alpha=0.3)
plt.plot(beta, np.nanmean(rb, axis=0), marker="o", linestyle="--", c="tab:pink")
plt.ylim(0.0, 0.75)
plt.yticks([0.1, 0.3, 0.5, 0.7])
plt.xscale("log")
plt.savefig(RESULTS_DIR / "Fig4c.tiff", dpi=300, bbox_inches="tight")
