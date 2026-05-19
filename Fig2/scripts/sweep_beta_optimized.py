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


def sweep_beta_perturb_optimized(
    key,
    n: int,
    eirat: float,
    perturbation: float,
    perturbation_batchs: int,
    tmax: float,
    numsamples: int,
    numpoints: int,
    *,
    eps: float = 0.01,
    dt: float = 0.1,
    logmin: float = -4.0,
    logmax: float = 0.0,
    p: float = 0.07,
):

    logbeta = np.linspace(logmin, logmax, numpoints)
    beta = 10**logbeta

    if n % perturbation_batchs != 0:
        raise ValueError(f"n must be divisible by perturbation_batchs")

    perturbation_idx = jnp.split(jnp.arange(n), perturbation_batchs)

    tau_c_store = np.zeros(
        (numpoints, numsamples, len(perturbation_idx)), dtype=np.float32
    )
    tau_p_store = np.zeros(
        (numpoints, numsamples, len(perturbation_idx)), dtype=np.float32
    )

    clst_store = np.zeros(
        (
            numpoints,
            numsamples,
        ),
        dtype=np.float32,
    )
    avpl_store = np.zeros(
        (
            numpoints,
            numsamples,
        ),
        dtype=np.float32,
    )

    start_time = time.time()
    print(f"Starting sweep over {numpoints} beta values...")
    for i, b in tqdm(
        enumerate(beta), total=numpoints, desc="sweeping beta", position=0
    ):
        try:
            X, C, L = run_perturbed_multi_gpu(
                key,
                n=n,
                beta=b,
                eirat=eirat,
                perturbation=perturbation,
                perturbation_idx_list=perturbation_idx,
                tmax=tmax,
                numsamples=numsamples,
                p=p,
            )

            X, C, L = X.astype(np.float32), C.astype(np.float32), L.astype(np.float32)

            X_corr = corr_keepaxis(X, relaxation_time=100)
            tau_c_store[i] = tau_keepaxis(X_corr, dt)

            tau_p_store[i] = threshold_pass_time_keepaxis(
                X, dt, perturbation_batchs=perturbation_batchs, eps=eps
            )

            clst_store[i] = C.astype(np.float32)
            avpl_store[i] = L.astype(np.float32)

        except Exception as e:
            print(f"Error processing beta={b}: {e}")

    end_time = time.time()
    print(f"Sweep completed in {end_time - start_time:.2f} seconds.")

    np.savez_compressed(
        OUTPUT_DIR / f"beta_perturb_sweep.npz",
        beta=beta,
        tau_c=tau_c_store,
        tau_p=tau_p_store,
        clst=clst_store,
        avpl=avpl_store,
    )

    return beta, tau_c_store, tau_p_store, clst_store, avpl_store


from argparse import ArgumentParser
import jax

if __name__ == "__main__":
    import jax

    argparser = ArgumentParser()
    argparser.add_argument("--n", type=int, default=300, help="Number of nodes")
    argparser.add_argument(
        "--eirat", type=float, default=0.2, help="excitatory-inhibitory ratio"
    )
    argparser.add_argument(
        "--perturbation", type=float, default=10, help="Perturbation magnitude"
    )
    argparser.add_argument(
        "--perturbation_batchs",
        type=int,
        default=30,
        help="Number of perturbation batches",
    )
    argparser.add_argument("--eps", type=float, default=0.01)
    argparser.add_argument(
        "--tmax", type=float, default=300, help="Maximum simulation time"
    )
    argparser.add_argument(
        "--numsamples", type=int, default=24, help="Number of samples"
    )
    argparser.add_argument("--numpoints", type=int, default=30, help="Number of points")
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

    beta, tau_c, tau_p, _, _ = sweep_beta_perturb_optimized(
        key,
        n=args.n,
        eirat=args.eirat,
        perturbation=args.perturbation,
        perturbation_batchs=args.perturbation_batchs,
        tmax=args.tmax,
        numsamples=args.numsamples,
        eps=args.eps,
        numpoints=args.numpoints,
        dt=args.dt,
        logmin=args.logmin,
        logmax=args.logmax,
    )

    import matplotlib.pyplot as plt

    tau_c_mean = np.mean(tau_c, axis=(1, 2))
    tau_p_mean = np.mean(tau_p, axis=(1, 2))

    tau_c_mean_norm = tau_c_mean / np.max(tau_c_mean)
    tau_p_mean_norm = tau_p_mean / np.max(tau_p_mean)

    plt.figure()
    plt.semilogx(beta, tau_c_mean_norm, label="Correalation Time")
    plt.semilogx(beta, tau_p_mean_norm, label="Threshold Pass Time")
    plt.xlabel("Beta")
    plt.ylabel("Time")
    plt.title(
        f"Sweep Beta (n={args.n}, eirat={args.eirat}, perturbation={args.perturbation})"
    )
    plt.legend()

    plt.savefig(OUTPUT_DIR / "tmpfig_beta.png")
