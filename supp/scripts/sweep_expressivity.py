import os
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils.expressivity import run_circle_expressivity

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

FIG_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = FIG_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sweep_beta_eirat_expressivity(
    n: int,
    numsamples: int,
    numpoints_beta: int,
    numpoints_eirat: int,
    *,
    logmin: float = -4.0,
    logmax: float = 0.0,
    eirat_min: float = 0.0,
    eirat_max: float = 1.0,
    p: float = 0.07,
    seed: int = 42,
    n_theta: int = 256,
    alpha: float = 1.0,
    t_on: float = 10.0,
    t_off: float = 0.0,
    dt: float = 0.1,
):
    logbeta = np.linspace(logmin, logmax, numpoints_beta)
    beta = 10**logbeta
    eirat = np.linspace(eirat_min, eirat_max, numpoints_eirat)

    le_store = np.zeros((numpoints_beta, numpoints_eirat, numsamples), dtype=np.float32)
    lg_store = np.zeros((numpoints_beta, numpoints_eirat, numsamples), dtype=np.float32)
    kappa_store = np.zeros(
        (numpoints_beta, numpoints_eirat, numsamples), dtype=np.float32
    )

    le_mean = np.zeros((numpoints_beta, numpoints_eirat), dtype=np.float32)
    lg_mean = np.zeros((numpoints_beta, numpoints_eirat), dtype=np.float32)
    kappa_mean = np.zeros((numpoints_beta, numpoints_eirat), dtype=np.float32)

    start_time = time.time()
    sample_offsets = np.arange(numsamples, dtype=np.int64)
    for i, b in tqdm(
        enumerate(beta), total=numpoints_beta, desc="sweeping 2D", position=0
    ):
        for j, e in tqdm(
            enumerate(eirat),
            desc=f"beta #{i + 1}",
            total=numpoints_eirat,
            position=1,
            leave=False,
        ):
            for s in range(numsamples):
                run_seed = int(seed + 10_000 * i + 100 * j + sample_offsets[s])
                try:
                    le, lg, kappa = run_circle_expressivity(
                        seed=run_seed,
                        n=n,
                        p=p,
                        beta=b,
                        eirat=e,
                        n_theta=n_theta,
                        alpha=alpha,
                        t_on=t_on,
                        t_off=t_off,
                        dt=dt,
                    )
                except Exception as exep:
                    print(f"Error processing beta={b}, eirat={e}, sample={s}: {exep}")
                    le = np.nan
                    lg = np.nan
                    kappa = np.nan

                le_store[i, j, s] = le
                lg_store[i, j, s] = lg
                kappa_store[i, j, s] = kappa

            le_mean[i, j] = np.nanmean(le_store[i, j])
            lg_mean[i, j] = np.nanmean(lg_store[i, j])
            kappa_mean[i, j] = np.nanmean(kappa_store[i, j])

    end_time = time.time()
    print(f"Sweep completed in {end_time - start_time:.2f} seconds.")

    out_path = OUTPUT_DIR / f"expressivity_sweep.npz"

    np.savez_compressed(
        out_path,
        beta=beta,
        eirat=eirat,
        le=le_store,
        lg=lg_store,
        kappa=kappa_store,
        le_mean=le_mean,
        lg_mean=lg_mean,
        kappa_mean=kappa_mean,
        n=n,
        numsamples=numsamples,
        n_theta=n_theta,
        alpha=alpha,
        t_on=t_on,
        t_off=t_off,
        dt=dt,
        seed=seed,
        logmin=logmin,
        logmax=logmax,
        eirat_min=eirat_min,
        eirat_max=eirat_max,
        p=p,
    )

    return beta, eirat, le_store, lg_store, kappa_store, le_mean, lg_mean, kappa_mean


if __name__ == "__main__":
    from argparse import ArgumentParser
    import matplotlib.pyplot as plt

    argparser = ArgumentParser()
    argparser.add_argument("--n", type=int, default=300, help="Number of nodes")
    argparser.add_argument(
        "--numsamples", type=int, default=24, help="Number of samples"
    )
    argparser.add_argument(
        "--numpoints_beta", type=int, default=30, help="Number of beta points"
    )
    argparser.add_argument(
        "--numpoints_eirat", type=int, default=30, help="Number of eirat points"
    )
    argparser.add_argument(
        "--logmin", type=float, default=-4.0, help="Logarithmic minimum beta value"
    )
    argparser.add_argument(
        "--logmax", type=float, default=0.0, help="Logarithmic maximum beta value"
    )
    argparser.add_argument(
        "--eirat_min", type=float, default=0.0, help="Minimum edge-to-node ratio"
    )
    argparser.add_argument(
        "--eirat_max", type=float, default=1.0, help="Maximum edge-to-node ratio"
    )
    argparser.add_argument("--p", type=float, default=0.07, help="Network sparsity")
    argparser.add_argument("--seed", type=int, default=42, help="Random seed")
    argparser.add_argument(
        "--n_theta", type=int, default=256, help="Number of circle samples"
    )
    argparser.add_argument("--alpha", type=float, default=1.0, help="Input gain")
    argparser.add_argument("--t_on", type=float, default=10.0, help="On duration")
    argparser.add_argument("--t_off", type=float, default=0.0, help="Off duration")
    argparser.add_argument("--dt", type=float, default=0.1, help="Time step")

    args = argparser.parse_args()

    beta, eirat, le, lg, kappa, le_mean, lg_mean, kappa_mean = (
        sweep_beta_eirat_expressivity(
            n=args.n,
            numsamples=args.numsamples,
            numpoints_beta=args.numpoints_beta,
            numpoints_eirat=args.numpoints_eirat,
            logmin=args.logmin,
            logmax=args.logmax,
            eirat_min=args.eirat_min,
            eirat_max=args.eirat_max,
            p=args.p,
            seed=args.seed,
            n_theta=args.n_theta,
            alpha=args.alpha,
            t_on=args.t_on,
            t_off=args.t_off,
            dt=args.dt,
        )
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.pcolor(beta, eirat, le_mean.T, cmap="Spectral", shading="auto")
    ax.set_xscale("log")
    ax.set_ylabel("Excitation Ratio")
    ax.set_xlabel(r"$\beta$")
    cbar = plt.colorbar(
        ax.pcolormesh(beta, eirat, le_mean.T, cmap="Spectral"),
        ax=ax,
        location="top",
    )
    cbar.set_label(r"$L_E$")

    plt.savefig(OUTPUT_DIR / "tmpfig_expressivity_le.png")

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.pcolor(beta, eirat, lg_mean.T, cmap="Spectral", shading="auto")
    ax.set_xscale("log")
    ax.set_ylabel("Excitation Ratio")
    ax.set_xlabel(r"$\beta$")
    cbar = plt.colorbar(
        ax.pcolormesh(beta, eirat, lg_mean.T, cmap="Spectral"),
        ax=ax,
        location="top",
    )
    cbar.set_label(r"$L_G$")

    plt.savefig(OUTPUT_DIR / "tmpfig_expressivity_lg.png")

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.pcolor(beta, eirat, kappa_mean.T, cmap="Spectral", shading="auto")
    ax.set_xscale("log")
    ax.set_ylabel("Excitation Ratio")
    ax.set_xlabel(r"$\beta$")
    cbar = plt.colorbar(
        ax.pcolormesh(beta, eirat, kappa_mean.T, cmap="Spectral"),
        ax=ax,
        location="top",
    )
    cbar.set_label(r"$\kappa$")

    plt.savefig(OUTPUT_DIR / "tmpfig_expressivity_kappa.png")
