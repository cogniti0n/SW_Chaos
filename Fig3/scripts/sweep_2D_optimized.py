import numpy as np
import jax.numpy as jnp
import time
import os
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

FIG_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = FIG_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from utils.swneural import run_perturbed_multi_gpu
from utils.quantities import corr_keepaxis, tau_keepaxis, threshold_pass_time_keepaxis

from tqdm import tqdm


def sweep_beta_eirat_2D_optimized(
    key,
    n: int,
    perturbation: float,
    perturbation_batchs: int,
    tmax: float,
    numsamples: int,
    numpoints_beta: int,
    numpoints_eirat: int,
    *,
    dt: float = 0.1,
    logmin: float = -4.0,
    logmax: float = 0.0,
    eirat_min: float = 0.0,
    eirat_max: float = 1.0,
    p: float = 0.07,
):

    num_devices = jax.local_device_count()
    print(f"Running on {num_devices} devices.")

    logbeta = np.linspace(logmin, logmax, numpoints_beta)
    beta = 10**logbeta
    eirat = np.linspace(eirat_min, eirat_max, numpoints_eirat)

    if n % perturbation_batchs != 0:
        raise ValueError(f"n must be divisible by perturbation_batchs")

    perturbation_idx = jnp.split(jnp.arange(n), perturbation_batchs)

    tau_c_store = np.zeros(
        (numpoints_beta, numpoints_eirat, numsamples, len(perturbation_idx)),
        dtype=np.float32,
    )
    tau_p_store = np.zeros(
        (numpoints_beta, numpoints_eirat, numsamples, len(perturbation_idx)),
        dtype=np.float32,
    )

    clst_store = np.zeros(
        (
            numpoints_beta,
            numpoints_eirat,
            numsamples,
        ),
        dtype=np.float32,
    )
    avpl_store = np.zeros(
        (
            numpoints_beta,
            numpoints_eirat,
            numsamples,
        ),
        dtype=np.float32,
    )

    start_time = time.time()

    for i, b in tqdm(
        enumerate(beta), total=numpoints_beta, desc="sweeping 2D", position=0
    ):
        for j, e in tqdm(
            enumerate(eirat),
            desc=f"beta #{i+1}",
            total=numpoints_eirat,
            position=1,
            leave=False,
        ):
            try:
                X, C, L = run_perturbed_multi_gpu(
                    key,
                    n=n,
                    beta=b,
                    eirat=e,
                    perturbation=perturbation,
                    perturbation_idx_list=perturbation_idx,
                    tmax=tmax,
                    numsamples=numsamples,
                    p=p,
                )

                X, C, L = (
                    X.astype(np.float32),
                    C.astype(np.float32),
                    L.astype(np.float32),
                )

                X_corr = corr_keepaxis(X, relaxation_time=100)
                tau_c_store[i, j] = tau_keepaxis(X_corr, dt)

                tau_p_store[i, j] = threshold_pass_time_keepaxis(
                    X, dt, perturbation_batchs=perturbation_batchs
                )

                clst_store[i, j] = C.astype(np.float32)
                avpl_store[i, j] = L.astype(np.float32)

            except Exception as exep:
                print(f"Error processing beta={b}, eirat={e}: {exep}")

    end_time = time.time()
    print(f"Sweep completed in {end_time - start_time:.2f} seconds.")

    np.savez_compressed(
        OUTPUT_DIR / f"2D_perturb_sweep.npz",
        beta=beta,
        eirat=eirat,
        tau_c=tau_c_store,
        tau_p=tau_p_store,
        clst=clst_store,
        avpl=avpl_store,
    )

    return beta, eirat, tau_c_store, tau_p_store, clst_store, avpl_store


import jax
from argparse import ArgumentParser

if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--n", type=int, default=300, help="Number of nodes")
    argparser.add_argument("--p", type=float, default=0.07, help="Network sparsity")
    argparser.add_argument(
        "--perturbation", type=float, default=10, help="Perturbation magnitude"
    )
    argparser.add_argument(
        "--perturbation_batchs",
        type=int,
        default=30,
        help="Number of perturbation batches",
    )
    argparser.add_argument(
        "--tmax", type=float, default=300, help="Maximum simulation time"
    )
    argparser.add_argument(
        "--numsamples", type=int, default=24, help="Number of samples"
    )
    argparser.add_argument(
        "--numpoints_beta", type=int, default=60, help="Number of beta points"
    )
    argparser.add_argument(
        "--numpoints_eirat", type=int, default=60, help="Number of eirat points"
    )
    argparser.add_argument(
        "--logmin", type=float, default=-4.0, help="Logarithmic minimum beta value"
    )
    argparser.add_argument(
        "--logmax", type=float, default=0.0, help="Logarithmic maximum beta value"
    )
    argparser.add_argument("--dt", type=float, default=0.1, help="Time step")
    argparser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = argparser.parse_args()

    key = jax.random.key(args.seed)

    beta, eirat, tau_c, tau_p, _, _ = sweep_beta_eirat_2D_optimized(
        key,
        n=args.n,
        perturbation=args.perturbation,
        perturbation_batchs=args.perturbation_batchs,
        tmax=args.tmax,
        numsamples=args.numsamples,
        numpoints_beta=args.numpoints_beta,
        numpoints_eirat=args.numpoints_eirat,
        dt=args.dt,
        logmin=args.logmin,
        logmax=args.logmax,
        p=args.p,
    )

    import matplotlib.pyplot as plt

    tau_c_mean = np.mean(tau_c, axis=(2, 3))
    tau_p_mean = np.mean(tau_p, axis=(2, 3))

    tau_c_mean_norm = tau_c_mean / np.max(tau_c_mean)
    tau_p_mean_norm = tau_p_mean / np.max(tau_p_mean)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.pcolor(beta, eirat, tau_c_mean_norm.T, cmap="Spectral", shading="auto")
    ax.set_xscale("log")
    ax.set_ylabel("Excitation Ratio")
    ax.set_xlabel(r"$\beta$")
    cbar = plt.colorbar(
        ax.pcolormesh(beta, eirat, tau_c_mean_norm.T, cmap="Spectral"),
        ax=ax,
        location="top",
    )
    cbar.set_label(r"$\tau_c$")

    plt.savefig(OUTPUT_DIR / "tmpfig_2D_c.png")

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.pcolor(beta, eirat, tau_p_mean_norm.T, cmap="Spectral", shading="auto")
    ax.set_xscale("log")
    ax.set_ylabel("Excitation Ratio")
    ax.set_xlabel(r"$\beta$")
    cbar = plt.colorbar(
        ax.pcolormesh(beta, eirat, tau_p_mean_norm.T, cmap="Spectral"),
        ax=ax,
        location="top",
    )
    cbar.set_label(r"$\tau_p$")

    plt.savefig(OUTPUT_DIR / "tmpfig_2D_p.png")
