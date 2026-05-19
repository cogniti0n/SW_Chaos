import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from utils.swneural import sw_adj, sw_nogen

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = FIG_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import Fig4.scripts.train_lorenz as train_mod


def sweep_beta_lorenz(
    numpoints: int,
    *,
    n: int = 300,
    p: float = 0.07,
    seed: int = 0,
    eirat: float = 0.2,
    n_lorenz: int = 10,
    F: float = 8.0,
    tmax: float = 200.0,
    dt: float = 0.1,
    horizon1: float = 1.0,
    horizon2: float = 5.0,
    washout_time: float = 20.0,
    alpha: float = 1e-4,
    logmin: float = -4.0,
    logmax: float = 0.0,
):
    ts = jnp.arange(0.0, tmax, dt)
    horizons = [horizon1, horizon2]

    key = jax.random.key(seed)
    rng = np.random.default_rng(seed)
    key_res, key_u, lorenz_key = jax.random.split(key, 3)

    lorenz_traj = train_mod.generate_lorenz96(ts, n_dim=n_lorenz, F=F, key=lorenz_key)
    u_matrix = jax.random.normal(key_u, (n, n_lorenz)) * 0.1
    target_signal = lorenz_traj[:, 0]

    logbeta = np.linspace(logmin, logmax, numpoints)
    betas = 10**logbeta

    hor1_store = np.full((numpoints,), np.nan, dtype=np.float32)
    hor2_store = np.full((numpoints,), np.nan, dtype=np.float32)

    start_time = time.time()
    print(f"Starting sweep over {numpoints} beta values...")
    progress = tqdm(
        betas,
        total=numpoints,
        desc="sweeping beta",
        position=0,
        leave=True,
        dynamic_ncols=True,
    )
    for i, beta in enumerate(progress):
        try:
            adj = sw_adj(rng, n, p, beta)
            adj = sw_nogen(rng, adj, eirat)
            adj = jnp.asarray(adj)

            x_states = train_mod.simulate_reservoir_multi(
                adj, key_res, u_matrix, lorenz_traj, ts
            )
            rmse_results = train_mod.chaotic_prediction_rmse(
                x_states,
                target_signal,
                dt,
                horizons,
                washout_time=washout_time,
                alpha=alpha,
            )

            hor1_store[i] = rmse_results[0]
            hor2_store[i] = rmse_results[1]
            progress.set_postfix(
                beta=f"{beta:.5g}",
                h1=f"{float(rmse_results[0]):.4f}",
                h2=f"{float(rmse_results[1]):.4f}",
            )
            tqdm.write(
                f"beta={beta:.5g} "
                f"Prediction Horizon {horizon1}: RMSE = {rmse_results[0]:.4f} "
                f"Prediction Horizon {horizon2}: RMSE = {rmse_results[1]:.4f}"
            )
        except Exception as exep:
            tqdm.write(f"Error processing beta={beta}: {exep}")

    end_time = time.time()
    print(f"Sweep completed in {end_time - start_time:.2f} seconds.")
    print(f"Final horizon arrays (shape {hor1_store.shape}):")
    print("hor1:", hor1_store)
    print("hor2:", hor2_store)

    out_path = OUTPUT_DIR / (f"lorenz_sweep_seed{seed}.npz")
    np.savez_compressed(
        out_path,
        beta=betas,
        hor1=hor1_store,
        hor2=hor2_store,
        seed=seed,
        logmin=logmin,
        logmax=logmax,
        n=n,
        p=p,
        eirat=eirat,
        n_lorenz=n_lorenz,
        F=F,
        tmax=tmax,
        dt=dt,
        horizon1=horizon1,
        horizon2=horizon2,
        washout_time=washout_time,
        alpha=alpha,
    )

    return betas, hor1_store, hor2_store


if __name__ == "__main__":
    from argparse import ArgumentParser

    argparser = ArgumentParser()
    argparser.add_argument("--n", type=int, default=300)
    argparser.add_argument("--p", type=float, default=0.07)
    argparser.add_argument("--eirat", type=float, default=0.2)
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--n-lorenz", type=int, default=10)
    argparser.add_argument("--F", type=float, default=8.0)
    argparser.add_argument("--tmax", type=float, default=200.0)
    argparser.add_argument("--dt", type=float, default=0.1)
    argparser.add_argument("--horizon1", type=float, default=1.0)
    argparser.add_argument("--horizon2", type=float, default=5.0)
    argparser.add_argument("--washout-time", type=float, default=20.0)
    argparser.add_argument("--alpha", type=float, default=1e-4)
    argparser.add_argument("--numpoints", type=int, default=30)
    argparser.add_argument("--logmin", type=float, default=-4.0)
    argparser.add_argument("--logmax", type=float, default=0.0)

    args = argparser.parse_args()

    sweep_beta_lorenz(
        numpoints=args.numpoints,
        n=args.n,
        p=args.p,
        seed=args.seed,
        eirat=args.eirat,
        n_lorenz=args.n_lorenz,
        F=args.F,
        tmax=args.tmax,
        dt=args.dt,
        horizon1=args.horizon1,
        horizon2=args.horizon2,
        washout_time=args.washout_time,
        alpha=args.alpha,
        logmin=args.logmin,
        logmax=args.logmax,
    )
