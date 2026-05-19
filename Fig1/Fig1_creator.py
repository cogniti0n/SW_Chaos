from utils.swneural import *
from utils.quantities import *
from pathlib import Path

import matplotlib.pyplot as plt

from utils.plt_setup_utils import setup_matplotlib_for_paper

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
setup_matplotlib_for_paper(
    figsize=(4.0, 2.8),
)

n = 300
p = 0.07
eirat = 0.2
tmax = 300
dt = 0.1

seed = 142
key = jax.random.key(seed)
rng = np.random.default_rng(seed)

timeaxis = np.linspace(0.0, tmax, int(tmax / dt))
timeaxis2 = timeaxis[DEFAULT_RELXATION_IDX:]

beta = [1e-4, 0.1, 0.8]

adj1 = sw_adj(rng, n, p, beta[0])
adj1 = sw_nogen(rng, adj1, eirat=eirat)
adj1_jax = jnp.array(adj1)

adj2 = sw_adj(rng, n, p, beta[1])
adj2 = sw_nogen(rng, adj2, eirat=eirat)
adj2_jax = jnp.array(adj2)

adj3 = sw_adj(rng, n, p, beta[2])
adj3 = sw_nogen(rng, adj3, eirat=eirat)
adj3_jax = jnp.array(adj3)


def rundynnp_same_init(adj, y0, dt, tmax):
    def func(t, y, args):
        return -y + adj @ jnp.tanh(y)

    term = dfx.ODETerm(func)
    solver = dfx.Dopri5()
    saveat = dfx.SaveAt(ts=jnp.linspace(0, tmax, int(tmax / dt)))
    stepsize_controller = dfx.PIDController(rtol=1e-5, atol=1e-5)

    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=tmax,
        dt0=dt,
        y0=y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
    )

    return jnp.asarray(sol.ys)


y0 = jax.random.uniform(key, (n,)) * 10.0 - 5.0

x1 = rundynnp_same_init(adj1_jax, y0, dt, tmax)
x2 = rundynnp_same_init(adj2_jax, y0, dt, tmax)
x3 = rundynnp_same_init(adj3_jax, y0, dt, tmax)

x1 = np.asarray(x1)
x2 = np.asarray(x2)
x3 = np.asarray(x3)

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


class NarrowCenterNorm(Normalize):
    """Narrow the white zone: keep only values near zero (±center_width) at the colormap center."""

    def __init__(self, vmin=-1, vmax=1, center_width=0.2, **kwargs):
        super().__init__(vmin, vmax, **kwargs)
        self.center_width = (
            center_width  # White-ish inside this range, stronger color outside.
        )

    def __call__(self, value, clip=None):
        v = np.ma.asarray(value).astype(float)
        if clip:
            v = np.ma.clip(v, self.vmin, self.vmax)
        cw = self.center_width
        lo, hi = self.vmin, self.vmax
        # [-1, -cw] -> [0, 0.47], [-cw, cw] -> [0.47, 0.53], [cw, 1] -> [0.53, 1]
        x = np.ma.where(
            v < -cw,
            0.47 * (v - lo) / (-cw - lo),
            np.ma.where(
                v <= cw,
                0.47 + (0.06 / (2 * cw)) * (v + cw),
                0.53 + 0.47 * (v - cw) / (hi - cw),
            ),
        )
        return np.ma.clip(x, 0, 1)


def plot_lines_y_gradient(
    timeaxis, data_columns, plot_idx, vmin=-10, vmax=10, cmap="RdBu_r"
):
    """Apply a y-value gradient per line segment with a narrow white band near zero."""
    ax = plt.gca()
    t = np.asarray(timeaxis)
    norm = NarrowCenterNorm(vmin=vmin, vmax=vmax, center_width=1.0)
    for it in plot_idx:
        y = np.asarray(data_columns[:, it])
        points = np.column_stack([t, y])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=3.5)
        lc.set_array(
            (y[:-1] + y[1:]) / 2
        )  # Color is determined by each segment midpoint y-value.
        ax.add_collection(lc)
    ax.set_ylim(vmin, vmax)
    ax.set_xlim(t.min(), t.max())


plot_idx = [49, 149, 249]
figsize = (6, 2.5)
tick_fontsize = 24

# Use only the 0-200 time range.
t_end = 200
n_time = int(t_end / dt) + 1
time_slice = timeaxis[:n_time]

d1 = np.asarray(x1[:n_time, :])[:, plot_idx]
d2 = np.asarray(x2[:n_time, :])[:, plot_idx]
d3 = np.asarray(x3[:n_time, :])[:, plot_idx]
vmin_val, vmax_val = -10.5, 10.5

plt.figure(figsize=figsize)
plot_lines_y_gradient(time_slice, d1, [0, 1, 2], vmin=vmin_val, vmax=vmax_val)
plt.ylim(vmin_val, vmax_val)
plt.xticks([], [])
plt.yticks([], [])
plt.savefig(RESULTS_DIR / "Fig1b.tiff", dpi=300, bbox_inches="tight", pad_inches=0)

plt.figure(figsize=figsize)
plot_lines_y_gradient(time_slice, d2, [0, 1, 2], vmin=vmin_val, vmax=vmax_val)
plt.ylim(vmin_val, vmax_val)
plt.xticks([], [])
plt.yticks([], [])
plt.savefig(RESULTS_DIR / "Fig1c.tiff", dpi=300, bbox_inches="tight", pad_inches=0)

plt.figure(figsize=figsize)
plot_lines_y_gradient(time_slice, d3, [0, 1, 2], vmin=vmin_val, vmax=vmax_val)
plt.ylim(vmin_val, vmax_val)
plt.xticks([], [])
plt.yticks([], [])
plt.savefig(RESULTS_DIR / "Fig1d.tiff", dpi=300, bbox_inches="tight", pad_inches=0)

plt.show()
