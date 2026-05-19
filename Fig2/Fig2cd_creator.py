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
import os
from pathlib import Path

from utils.swneural import *
from utils.quantities import *

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax

# %%
from utils.plt_setup_utils import setup_matplotlib_for_paper

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

setup_matplotlib_for_paper(
    figsize=(4.0, 3.5),
)

# %%
# correlation plot
n = 300
p = 0.07
eirat = 0.2
tmax = 300
dt = 0.1
perturbation = 10
perturbation_idx_list = [jnp.arange(10)]
numsamples = 1000

beta_reg = 1e-4
beta_sw = 0.1
beta_rand = 0.8

run_tmp = lambda b: run_perturbed(
    jax.random.key(42),
    n,
    b,
    eirat,
    10.0,
    perturbation_idx_list,
    tmax,
    numsamples,
    dt,
)

x_reg = run_tmp(beta_reg)
x_sw = run_tmp(beta_sw)
x_rand = run_tmp(beta_rand)

# %%
c_reg = corr_keepaxis(x_reg[0], 100)
c_sw = corr_keepaxis(x_sw[0], 100)
c_rand = corr_keepaxis(x_rand[0], 100)

# %%
print(c_reg.shape)
c_reg_mean = np.mean(c_reg, axis=0).squeeze()
c_sw_mean = np.mean(c_sw, axis=0).squeeze()
c_rand_mean = np.mean(c_rand, axis=0).squeeze()

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
x_axis = np.arange(0, c_reg_mean.shape[0]) * dt

axes[0].plot(x_axis[:2000], c_reg_mean[:2000], color="tab:grey", label="Regular")
axes[1].plot(x_axis[:2000], c_sw_mean[:2000], color="tab:grey", label="Small-world")
axes[2].plot(x_axis[:2000], c_rand_mean[:2000], color="tab:grey", label="Random")

axes[0].tick_params(axis="x", labelbottom=False)
axes[1].tick_params(axis="x", labelbottom=False)

# plt.legend(loc="upper right", fontsize=20, markerscale=0.9)

plt.savefig(RESULTS_DIR / "Fig2c.tiff", dpi=300, bbox_inches="tight")

# %%
import matplotlib.colors as mcolors

n = 300
p = 0.07
eirat = 0.2
tmax = 10
dt = 0.1
perturbation = 10
perturbation_idx = jnp.arange(10)
timeaxis = np.arange(0, tmax, dt)

beta = [1e-4, 0.1, 0.8]

fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)

norm_abs = mcolors.SymLogNorm(linthresh=0.02, linscale=1.0, vmin=0, vmax=10, base=10)

single_cmap = mcolors.ListedColormap(["C0"])

base_rgba = mcolors.to_rgba("C0")
alpha_levels = np.linspace(0.0, 1.0, 256)
alpha_colors = np.column_stack(
    [
        np.full_like(alpha_levels, base_rgba[0]),
        np.full_like(alpha_levels, base_rgba[1]),
        np.full_like(alpha_levels, base_rgba[2]),
        alpha_levels,
    ]
)
alpha_cmap = mcolors.ListedColormap(alpha_colors)
alpha_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

ims = []

for i, (ax, b) in enumerate(zip(axes, beta)):
    seed = 42
    key = jax.random.key(seed)
    rng = np.random.default_rng(seed)

    adj = sw_adj(rng, n, p, b)
    adj, graph = sw_gen(rng, adj, eirat)
    adj_jax = jnp.array(adj)

    x = solve_single_dynamics(
        adj_jax, perturbation, perturbation_idx, relaxation_time=10.0, tmax=tmax, dt=dt
    )

    node_distances = []
    source_indices = [i for i in range(10)]

    for node in graph.nodes():
        if node in source_indices:
            dist = 0
        else:
            try:
                dists_to_sources = [
                    nx.shortest_path_length(graph, source=s, target=node)
                    for s in source_indices
                ]
                dist = min(dists_to_sources)
            except nx.NetworkXNoPath:
                dist = float("inf")

        node_distances.append((node, dist))

    sorted_nodes = sorted(node_distances, key=lambda x: x[1])
    sorted_indices = [x[0] for x in sorted_nodes]
    sorted_data = x[:, sorted_indices]
    plot_data = np.abs(sorted_data[1:40, :].T)

    alpha_map = np.clip(norm_abs(plot_data), 0, 1)

    im = ax.imshow(
        plot_data,
        norm=norm_abs,
        aspect="auto",
        cmap=single_cmap,
        alpha=alpha_map,
        origin="lower",
        extent=(timeaxis[1], timeaxis[40], 0, plot_data.shape[0]),
    )
    ims.append(im)

    if i < len(axes) - 1:
        ax.tick_params(axis="x", labelbottom=False)

for ax in axes:
    ax.set_yticks([0, 150])
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlim(0.1, 3.3)

pos_top = axes[0].get_position()
pos_bottom = axes[-1].get_position()

cbar_ax = fig.add_axes(
    (pos_top.x1 + 0.02, pos_bottom.y0, 0.02, pos_top.y1 - pos_bottom.y0)
)

sm = plt.cm.ScalarMappable(norm=alpha_norm, cmap=alpha_cmap)
sm.set_array([])

cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_ticks([0.0, 0.5, 1.0])

plt.savefig(RESULTS_DIR / "Fig2d.tiff", dpi=300, bbox_inches="tight")

plt.close()
