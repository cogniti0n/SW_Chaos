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
from utils.swneural import *
from pathlib import Path

from utils.rewire import *
from utils.plt_setup_utils import setup_matplotlib_for_paper
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

setup_matplotlib_for_paper(figsize=(6.0, 2.0))


# %%
n = 300
p = 0.07
eirat = 0.2
tmax = 600.0
dt = 0.1
time_axis = np.arange(0, tmax, dt)

perturbation = 10.0
perturbation_idx = np.arange(0, 10)


def build_base_adj(rng, beta, eirat=eirat):
    adj = sw_adj(rng, n, p, beta)
    adj, g = sw_gen(rng, adj, eirat)
    print(g.number_of_edges())
    return adj


def run_rewire(key, adj_list, t_cuts):
    adj_sequence = jnp.asarray(adj_list)
    t_cuts_array = jnp.asarray(t_cuts, dtype=float)
    return rundynnp_rewire(
        key,
        adj_sequence,
        t_cuts_array,
        perturbation,
        jnp.asarray(perturbation_idx),
        tmax,
        dt,
    )


_line_colors = [
    "#bb4e99",
    "#003f5c",
    "#ffa600",
]


def plot_traces(state_traces, node_indices, outpath, t_cuts, colors=_line_colors):
    plt.figure()
    state_traces = state_traces.squeeze()
    for i, node_idx in enumerate(node_indices):
        plt.plot(time_axis, state_traces[:, node_idx], color=colors[i % len(colors)])
    for cut_time in t_cuts:
        plt.axvline(cut_time, ymin=-10.0, ymax=10.0, color="#ff5f68", linestyle="--")
    plt.ylim(-10, 10)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")


# %%
# dynamics control using rewiring
beta_rewire = 10 ** (-1.15)
rewire_ratio = 0.03

rewire_seed = 42
rewire_base_rng = np.random.default_rng(rewire_seed)
rewire_rng = np.random.default_rng(rewire_seed + 1)

rewire_base_adj = build_base_adj(rewire_base_rng, beta_rewire)
rewire_adj = sw_rewire_random_edgeswap(rewire_rng, rewire_base_adj, rewire_ratio)

rewire_t_cuts = [200.0, 400.0]

rewire_state_traces = run_rewire(
    jax.random.key(rewire_seed),
    [rewire_base_adj, rewire_adj, rewire_base_adj],
    t_cuts=rewire_t_cuts,
)

plot_traces(
    rewire_state_traces,
    [0, 149, 299],
    RESULTS_DIR / "Fig3e.tiff",
    rewire_t_cuts,
)
