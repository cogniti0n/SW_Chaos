from Fig4.scripts.train_lorenz import (
    generate_lorenz96,
    predict_ridge as predict_lorenz_ridge,
    ridge_readout,
    simulate_reservoir_multi,
)
from Fig4.scripts.train_remote_bandwidth import (
    block_indices,
    fit_ridge,
    make_symbol_stream,
    predict_ridge,
    simulate_reservoir_saveat_zoh,
)
from utils.swneural import sw_adj, sw_nogen
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import os


import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_FIGSIZE = (5.0, 3.0)
DEFAULT_FIG_DIR = RESULTS_DIR


def save_single_line_plot(signal, save_path):
    signal = np.asarray(signal)
    plt.figure(figsize=(3.5, 2.5))
    plt.plot(signal, color="#59a89c", linewidth=4.0)
    plt.axis("off")
    plt.gca().set_facecolor("none")
    plt.savefig(save_path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0)


def fit_future_readout_and_predict(x, s, dt, horizon, washout_time, alpha=1e-4):
    washout_steps = int(washout_time / dt)
    xw = x[washout_steps:]  # (Tw, N)
    sw = s[washout_steps:]  # (Tw,)

    k = int(jnp.round(horizon / dt))
    Xtr = xw[:-k]
    ztr = sw[k:]
    w, mu, sig = ridge_readout(Xtr, ztr, alpha=alpha)
    pred = predict_lorenz_ridge(Xtr, w, mu, sig)

    return ztr, pred


def fit_signal_readout_and_predict(x, s, washout_time, dt, alpha=1e-4):
    washout_steps = int(washout_time / dt)
    xw = x[washout_steps:]
    sw = s[washout_steps:]

    w, mu, sig = ridge_readout(xw, sw, alpha=alpha)
    pred = predict_lorenz_ridge(xw, w, mu, sig)
    return sw, pred


def make_default_reservoir(n_res=300, p=0.07, beta=0.10, eirat=0.2, seed=0):
    rng = np.random.default_rng(seed)
    adj = sw_adj(rng, n_res, p, beta)
    adj = sw_nogen(rng, adj, eirat)
    return rng, jnp.asarray(adj)


def extend_step_series(t, y, t_end):
    t = np.asarray(t)
    y = np.asarray(y)
    return np.concatenate([t, [t_end]]), np.concatenate([y, [y[-1]]])


def style_axis_frame(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(length=0, labelbottom=False, labelleft=False)


def run_and_save_lorenz_signals(
    adj,
    u,
    beta_key_seed=0,
    lorenz_key_seed=1,
    ts_end=200.0,
    dt=0.05,
    n_dim=40,
    F=8.0,
    horizon=1.0,
    washout_time=50.0,
    alpha=1e-4,
    out_dir=DEFAULT_FIG_DIR,
):
    os.makedirs(out_dir, exist_ok=True)

    ts = jnp.arange(0.0, ts_end + 1e-12, dt)

    # generate lorenz96
    lorenz_key = jax.random.key(lorenz_key_seed)
    S_ts = generate_lorenz96(ts, n_dim=n_dim, F=F, key=lorenz_key)  # (T, n_dim)

    # original signal
    s0 = S_ts[:, 0]

    save_single_line_plot(
        s0,
        os.path.join(out_dir, "Fig4a_lorenz.tiff"),
    )

    # reservoir simulation
    res_key = jax.random.key(beta_key_seed)
    x = simulate_reservoir_multi(
        adj=adj,
        key=res_key,
        u=u,
        S_ts=S_ts,
        ts=ts,
        x0=None,
    )  # (T, n_res)

    # prediction
    target_true, pred = fit_future_readout_and_predict(
        x=x,
        s=s0,
        dt=dt,
        horizon=horizon,
        washout_time=washout_time,
        alpha=alpha,
    )

    save_single_line_plot(
        pred,
        os.path.join(out_dir, "Fig4a_lorenz_predicted_2.tiff"),
    )


def plot_lorenz(figsize=DEFAULT_FIGSIZE, out_dir=DEFAULT_FIG_DIR):
    os.makedirs(out_dir, exist_ok=True)

    dt = 0.01
    ts = jnp.arange(0.0, 20.0 + 1e-12, dt)

    n_lorenz = 10
    lorenz_key = jax.random.key(1)
    lorenz_dynamics = generate_lorenz96(ts, n_dim=n_lorenz, key=lorenz_key)
    s0 = lorenz_dynamics[:, 0]

    n_res = 300
    p = 0.07
    beta = 0.10
    eirat = 0.2
    seed = 0

    _, adj = make_default_reservoir(
        n_res=n_res,
        p=p,
        beta=beta,
        eirat=eirat,
        seed=seed,
    )

    key = jax.random.key(seed)
    key_res, key_u = jax.random.split(key)
    u = jax.random.normal(key_u, (n_res, n_lorenz)) * 0.1

    x = simulate_reservoir_multi(adj=adj, key=key_res, u=u, S_ts=lorenz_dynamics, ts=ts)
    s_true, s_pred = fit_signal_readout_and_predict(
        x=x,
        s=s0,
        washout_time=10.0,
        dt=dt,
    )

    t_plot = np.asarray(ts[int(10.0 / dt) :])
    s_true = np.asarray(s_true)
    s_pred = np.asarray(s_pred)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t_plot, s_true, color="#59a89c", linewidth=4.0)
    style_axis_frame(ax)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(os.path.join(out_dir, "Fig4a_lorenz.tiff"), dpi=300, transparent=True)

    fig, ax = plt.subplots(figsize=figsize)
    pred_mask = t_plot >= 15.0
    ax.plot(t_plot, s_true, color="#59a89c", linewidth=4.0)
    ax.plot(t_plot[pred_mask], s_pred[pred_mask], color="#000000", linewidth=4.0)
    style_axis_frame(ax)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(os.path.join(out_dir, "Fig4a_lorenz_2.tiff"), dpi=300, transparent=True)


def fit_remote_bandwidth_readout_and_predict(
    X,
    y,
    washout_syms,
    alpha=1e-2,
    bias=True,
    standardize=True,
):
    X = jnp.asarray(X)
    y = jnp.asarray(y)

    Xw = X[washout_syms:]
    yw = y[washout_syms:]

    w, mu, sig = fit_ridge(
        Xw,
        yw,
        alpha=alpha,
        bias=bias,
        standardize=standardize,
    )
    yhat = predict_ridge(
        Xw,
        w,
        mu,
        sig,
        bias=bias,
        standardize=standardize,
    )
    return np.asarray(yw), np.asarray(yhat)


def plot_remote_bandwidth(
    T_total=50.0,
    T_sym=2.5,
    dt=0.01,
    seed=0,
    figsize=DEFAULT_FIGSIZE,
    out_dir=DEFAULT_FIG_DIR,
):
    os.makedirs(out_dir, exist_ok=True)

    n_res = 300
    p = 0.07
    beta = 0.1
    eirat = 0.2
    k_out = 30
    washout_time = 10.0
    washout_syms = int(washout_time / T_sym)

    rng, adj = make_default_reservoir(
        n_res=n_res,
        p=p,
        beta=beta,
        eirat=eirat,
        seed=seed,
    )

    in_idx = block_indices(n_res, 0, 1)
    out_idx = block_indices(n_res, n_res // 2, k_out)
    B_in = np.zeros((n_res, 1), dtype=np.float32)
    B_in[in_idx[0], 0] = 0.5
    B_in = jnp.asarray(B_in)

    ts_base, s_ts, s_sym = make_symbol_stream(
        T_total=T_total,
        dt=dt,
        T_sym=T_sym,
        rng=rng,
        sigma_symbol_noise=0.0,
    )
    u_ts = jnp.asarray(s_ts[:, None], dtype=jnp.float32)

    symbol_times = np.arange(s_sym.shape[0], dtype=np.float32) * T_sym
    save_ts = (np.arange(s_sym.shape[0], dtype=np.float32) + 1.0) * T_sym - 1e-9
    x_save = simulate_reservoir_saveat_zoh(
        key=jax.random.key(seed),
        adj=adj,
        B_in=B_in,
        s_sym=s_sym,
        T_sym=T_sym,
        save_ts=save_ts,
        dt0=dt,
        rtol=1e-5,
        atol=1e-5,
    )

    X = jnp.tanh(x_save[:, jnp.asarray(out_idx)])
    y_true, y_pred = fit_remote_bandwidth_readout_and_predict(
        X=X,
        y=s_sym,
        washout_syms=washout_syms,
    )

    t_plot = symbol_times[washout_syms:]
    t_plot_step, y_true_step = extend_step_series(t_plot, y_true, T_total)
    _, y_pred_step = extend_step_series(t_plot, y_pred, T_total)

    fig, ax = plt.subplots(figsize=figsize)
    ax.step(t_plot_step, y_true_step, where="post", color="#e15759", linewidth=4.0)
    style_axis_frame(ax)

    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(
        os.path.join(out_dir, "Fig4a_remote_bandwidth.tiff"),
        dpi=300,
        transparent=True,
    )

    fig, ax = plt.subplots(figsize=figsize)
    pred_mask = t_plot_step >= 30.0
    ax.step(
        t_plot_step,
        y_true_step,
        where="post",
        color="#e15759",
        linewidth=4.0,
    )
    ax.step(
        t_plot_step[pred_mask],
        y_pred_step[pred_mask],
        where="post",
        color="#000000",
        linewidth=4.0,
    )
    style_axis_frame(ax)

    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(
        os.path.join(out_dir, "Fig4a_remote_bandwidth_2.tiff"),
        dpi=300,
        transparent=True,
    )


def plot_symbolstream(out_dir=DEFAULT_FIG_DIR):
    os.makedirs(out_dir, exist_ok=True)

    _, _, symbol_stream = make_symbol_stream(
        10.0,
        0.1,
        2.0,
        np.random.default_rng(0),
        0.0,
    )
    symbol_stream = np.asarray(symbol_stream)

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot(symbol_stream, color="#36864a", linewidth=4.0)
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(
        os.path.join(out_dir, "Fig4a_symbolstream.tiff"),
        dpi=300,
        transparent=True,
    )


def plot_mnist(out_dir=DEFAULT_FIG_DIR):
    from torchvision.datasets import MNIST

    os.makedirs(out_dir, exist_ok=True)
    ds = MNIST(root="./data", train=True, download=True)

    for i in range(4):
        img, _ = ds[i]
        img_arr = np.asarray(img, dtype=np.float32) / 255.0
        plt.imsave(
            os.path.join(out_dir, f"Fig4a_mnist_{i + 1}.tiff"),
            img_arr,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            format="tiff",
        )


if __name__ == "__main__":
    figsize = DEFAULT_FIGSIZE
    out_dir = DEFAULT_FIG_DIR

    plot_lorenz(figsize=figsize, out_dir=out_dir)
    plot_remote_bandwidth(figsize=figsize, out_dir=out_dir)
    plot_mnist(out_dir=out_dir)
    plt.close("all")
