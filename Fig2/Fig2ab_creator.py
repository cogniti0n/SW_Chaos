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

# %%
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

data_filepath = OUTPUT_DIR / "beta_perturb_sweep.npz"
data = np.load(data_filepath)

beta = data["beta"]
tau_c = data["tau_c"]
tau_p = data["tau_p"]
clst = data["clst"]
avpl = data["avpl"]

print(tau_c.shape)
print(tau_p.shape)
print(clst.shape)
print(avpl.shape)

# %%
from utils.plt_setup_utils import setup_matplotlib_for_paper

setup_matplotlib_for_paper(figsize=(4.0, 2.8))

# %%
clst_mean = np.mean(clst, axis=1)
avpl_mean = np.mean(avpl, axis=1)

clst_norm = clst_mean / np.max(clst_mean)
avpl_norm = avpl_mean / np.max(avpl_mean)

x_c = np.repeat(beta, clst.shape[1])
y_c = clst.flatten() / np.max(clst)

x_l = np.repeat(beta, avpl.shape[1])
y_l = avpl.flatten() / np.max(avpl)

# %%
fig1, ax1 = plt.subplots()

sc_c = ax1.scatter(x_c, y_c, c="tab:red", alpha=0.2, s=20, label=r"$C$")
sc_l = ax1.scatter(x_l, y_l, c="tab:blue", alpha=0.2, s=20, label=r"$L$")

ax1.plot(
    beta, clst_mean / np.max(clst), ".--", color="tab:red", label=r"$\langle C \rangle$"
)
ax1.plot(
    beta,
    avpl_mean / np.max(avpl),
    ".--",
    color="tab:blue",
    label=r"$\langle L \rangle$",
)

ax1.set_xscale("log")
ax1.set_ylim(0, 1.2)

fig1.savefig(RESULTS_DIR / "Fig2a.tiff", dpi=200, bbox_inches="tight")

# %%
tau_c_mean_batch = np.mean(tau_c, axis=(2,))
tau_p_mean_batch = np.mean(tau_p, axis=(2,))

tau_c_mean = np.mean(tau_c_mean_batch, axis=1)
tau_p_mean = np.mean(tau_p_mean_batch, axis=1)

tau_c_norm = tau_c_mean / np.max(tau_c_mean)
tau_p_norm = tau_p_mean / np.max(tau_p_mean)

x_tau_c = np.repeat(beta, tau_c_mean_batch.shape[-1])
y_tau_c = tau_c_mean_batch.flatten() / np.max(tau_c_mean_batch)

x_tau_p = np.repeat(beta, tau_p_mean_batch.shape[-1])
y_tau_p = tau_p_mean_batch.flatten() / np.max(tau_p_mean_batch)

# %%
fig, ax = plt.subplots()

sc1 = ax.scatter(
    x_tau_c, y_tau_c, c="tab:red", alpha=0.2, s=20, label=r"$\tilde{\tau}$"
)
sc2 = ax.scatter(
    x_tau_p, y_tau_p, c="tab:blue", alpha=0.2, s=20, label=r"$\tilde{\tau}_{thresh}$"
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

fig.savefig(RESULTS_DIR / "Fig2b.tiff", dpi=200, bbox_inches="tight")

plt.close()
