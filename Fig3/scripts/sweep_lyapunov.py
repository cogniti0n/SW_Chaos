import numpy as np
import time
import os
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

FIG_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = FIG_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import jax
from utils.lyapunov import run_lle_multi_gpu

from tqdm import tqdm


def sweep_beta_eirat_lyapunov_optimized(
    key,
    n: int,
    numsamples: int,
    numpoints_beta: int,
    numpoints_eirat: int,
    *,
    t_transient: float = 10.0,
    t_total: float = 300.0,
    dt0: float = 0.1,
    seg_len: float = 2.0,
    rtol: float = 1e-5,
    atol: float = 1e-5,
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

    lle_store = np.zeros(
        (numpoints_beta, numpoints_eirat, numsamples), dtype=np.float32
    )
    lle_mean_store = np.zeros((numpoints_beta, numpoints_eirat), dtype=np.float32)

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
                lle = run_lle_multi_gpu(
                    key,
                    n=n,
                    beta=b,
                    eirat=e,
                    numsamples=numsamples,
                    p=p,
                    t_transient=t_transient,
                    t_total=t_total,
                    dt0=dt0,
                    seg_len=seg_len,
                    rtol=rtol,
                    atol=atol,
                )

                lle_store[i, j] = lle.astype(np.float32)
                lle_mean_store[i, j] = np.mean(lle_store[i, j])

            except Exception as exep:
                print(f"Error processing beta={b}, eirat={e}: {exep}")

    end_time = time.time()
    print(f"Sweep completed in {end_time - start_time:.2f} seconds.")

    np.savez_compressed(
        OUTPUT_DIR / f"lyapunov_sweep.npz",
        beta=beta,
        eirat=eirat,
        lle=lle_store,
        lle_mean=lle_mean_store,
    )

    return beta, eirat, lle_store, lle_mean_store


from argparse import ArgumentParser

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--n", type=int, default=300, help="Number of nodes")
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
    argparser.add_argument(
        "--eirat_min",
        type=float,
        default=0.0,
        help="Minimum excitatory-inhibitory ratio",
    )
    argparser.add_argument(
        "--eirat_max",
        type=float,
        default=1.0,
        help="Maximum excitatory-inhibitory ratio",
    )
    argparser.add_argument(
        "--t_transient", type=float, default=20.0, help="Transient time"
    )
    argparser.add_argument("--t_total", type=float, default=2000.0, help="Total time")
    argparser.add_argument("--dt0", type=float, default=0.1, help="Initial time step")
    argparser.add_argument("--seg_len", type=float, default=2.0, help="Segment length")
    argparser.add_argument("--rtol", type=float, default=1e-5, help="Relative tol")
    argparser.add_argument("--atol", type=float, default=1e-5, help="Absolute tol")
    argparser.add_argument("--p", type=float, default=0.07, help="Network sparsity")
    argparser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = argparser.parse_args()

    key = jax.random.key(args.seed)

    beta, eirat, lle, lle_mean = sweep_beta_eirat_lyapunov_optimized(
        key,
        n=args.n,
        numsamples=args.numsamples,
        numpoints_beta=args.numpoints_beta,
        numpoints_eirat=args.numpoints_eirat,
        t_transient=args.t_transient,
        t_total=args.t_total,
        dt0=args.dt0,
        seg_len=args.seg_len,
        rtol=args.rtol,
        atol=args.atol,
        logmin=args.logmin,
        logmax=args.logmax,
        eirat_min=args.eirat_min,
        eirat_max=args.eirat_max,
        p=args.p,
    )
