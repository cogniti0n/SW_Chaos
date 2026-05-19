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
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# %%
from utils.plt_setup_utils import setup_matplotlib_for_paper

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

setup_matplotlib_for_paper(figsize=(5.0, 5.0))

# %%
data_filepath = OUTPUT_DIR / "2D_perturb_sweep.npz"
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


# %%

# --- Figure 3a: tau_c ---
_hex_3a = [
    "#e9871b",
    "#ea9834",
    "#eba84c",
    "#edb763",
    "#efc57b",
    "#f2d394",
    "#f5e0ad",
    "#f9edc7",
    "#fff9e1",
]
reverse_cmap_3a = True  # True: reverse colormap (custom only)
use_custom_cmap_3a = True  # False: use builtin_cmap_3a instead
builtin_cmap_3a = "gist_heat"
cmap_3a = (
    LinearSegmentedColormap.from_list(
        "cmap_3a", _hex_3a if not reverse_cmap_3a else _hex_3a[::-1]
    )
    if use_custom_cmap_3a
    else plt.get_cmap(builtin_cmap_3a)
)

vmin, vmax = 0, 1

fig, ax = plt.subplots()
m0 = ax.pcolormesh(
    beta,
    eirat,
    tau_c_mean_norm.T,
    cmap=cmap_3a,
    shading="auto",
    vmin=vmin,
    vmax=vmax,
)
ax.set_xscale("log")
cax = ax.inset_axes((1.02, 0.0, 0.04, 1.0))
cb = fig.colorbar(m0, cax=cax, orientation="vertical")
cb.set_ticks([0, 0.5, 1.0])
plt.savefig(RESULTS_DIR / "Fig3a.tiff", dpi=600, bbox_inches="tight")
plt.close()


# --- Figure 3b: Lyapunov ---
_hex_3b = [
    "#e36d6d",
    "#f58d84",
    "#ffc6c0",
    "#ffffff",
    "#c3d5eb",
    "#85acd7",
    "#3685c2",
]
reverse_cmap_3b = False  # True: reverse colormap (custom only)
use_custom_cmap_3b = True  # False: use builtin_cmap_3b instead
builtin_cmap_3b = "PRGn"
cmap_3b = (
    LinearSegmentedColormap.from_list(
        "cmap_3b", _hex_3b if not reverse_cmap_3b else _hex_3b[::-1]
    )
    if use_custom_cmap_3b
    else plt.get_cmap(builtin_cmap_3b)
)

lyapunov_filepath = OUTPUT_DIR / "lyapunov_sweep.npz"
ld = np.load(lyapunov_filepath)

beta = ld["beta"]
eirat = ld["eirat"]
lle_mean = ld["lle_mean"]

C = lle_mean.T
vmax = 0.30
vmin = -0.3

norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

fig, ax = plt.subplots()

m2 = ax.pcolormesh(beta, eirat, C, cmap=cmap_3b, norm=norm, shading="auto")

ax.set_xscale("log")

# Colorbar on the right
cax = ax.inset_axes((1.02, 0.0, 0.04, 1.0))
cb = fig.colorbar(m2, cax=cax, orientation="vertical")
cb.set_ticks([vmin, 0.0, vmax])
plt.savefig(RESULTS_DIR / "Fig3c.tiff", dpi=600, bbox_inches="tight")
plt.close()


# --- Figure 3c: tau_p ---
_hex_3c = [
    "#6559a4",
    "#7d6aae",
    "#937db8",
    "#a890c2",
    "#bca4cd",
    "#ceb9d8",
    "#dfcee4",
    "#f0e4f1",
    "#fffbff",
]
reverse_cmap_3c = True  # True: reverse colormap (custom only)
use_custom_cmap_3c = True  # False: use builtin_cmap_3c instead
builtin_cmap_3c = "Purples"
cmap_3c = (
    LinearSegmentedColormap.from_list(
        "cmap_3c", _hex_3c if not reverse_cmap_3c else _hex_3c[::-1]
    )
    if use_custom_cmap_3c
    else plt.get_cmap(builtin_cmap_3c)
)
vmax = 1.0
vmin = 0.0

fig, ax = plt.subplots()
m1 = ax.pcolormesh(
    beta,
    eirat,
    tau_p_mean_norm.T,
    cmap=cmap_3c,
    shading="auto",
    vmin=vmin,
    vmax=vmax,
)
ax.set_xscale("log")
cax = ax.inset_axes((1.02, 0.0, 0.04, 1.0))
cb = fig.colorbar(m1, cax=cax, orientation="vertical")
cb.set_ticks([vmin, 0.5, vmax])
plt.savefig(RESULTS_DIR / "Fig3b.tiff", dpi=600, bbox_inches="tight")
plt.close()
