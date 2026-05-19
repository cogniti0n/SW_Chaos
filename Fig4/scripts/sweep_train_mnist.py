import os
import sys
import time
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from tqdm import tqdm

from utils.swneural import sw_adj, sw_nogen

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = FIG_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import Fig4.scripts.train_mnist as train_mod


def sweep_beta_mnist(
    numpoints: int,
    *,
    seed: int = 0,
    n_side: int = 20,
    p: float = 0.07,
    eirat: float = 0.2,
    t_on: float = 10.0,
    t_off: float = 0.0,
    dt: float = 0.1,
    batch_size: int = 256,
    alpha: float = 1e-4,
    logmin: float = -4.0,
    logmax: float = 0.0,
    max_batches_train=234,
    max_batches_test=39,
):
    logbeta = np.linspace(logmin, logmax, numpoints)
    beta = 10**logbeta

    n = n_side * n_side
    u = jnp.ones((n,), dtype=jnp.float32) / jnp.sqrt(n)

    train_loader, test_loader = train_mod.make_mnist_loaders(
        n_side=n_side, batch_size=batch_size
    )

    acc_store = np.full((numpoints,), np.nan, dtype=np.float32)

    start_time = time.time()
    print(f"Starting sweep over {numpoints} beta values...")
    progress = tqdm(
        beta,
        total=numpoints,
        desc="sweeping beta",
        position=0,
        leave=True,
        dynamic_ncols=True,
    )
    for i, b in enumerate(progress):
        try:
            rng = np.random.default_rng(seed)
            adj_bool = sw_adj(rng, n, p, b)
            adj_signed = sw_nogen(rng, adj_bool, eirat)
            adj = jnp.asarray(adj_signed, dtype=jnp.float32)

            Xtr, ytr = train_mod.extract_features(
                train_loader,
                adj,
                u,
                t_on,
                t_off,
                dt,
                max_batches=max_batches_train,
            )
            Xte, yte = train_mod.extract_features(
                test_loader,
                adj,
                u,
                t_on,
                t_off,
                dt,
                max_batches=max_batches_test,
            )

            acc = train_mod.train_ridge(
                Xtr,
                ytr,
                Xte,
                yte,
                l2=alpha,
            )
            acc_store[i] = acc
            progress.set_postfix(beta=f"{b:.5g}", acc=f"{float(acc):.4f}")
            tqdm.write(f"beta={b:.5g} acc={acc:.4f}")
        except Exception as exep:
            tqdm.write(f"Error processing beta={b}: {exep}")

    end_time = time.time()
    print(f"Sweep completed in {end_time - start_time:.2f} seconds.")
    print(f"Final accuracy vector (shape {acc_store.shape}):")
    print(acc_store)

    out_path = OUTPUT_DIR / (f"mnist_sweep_seed{seed}.npz")
    np.savez_compressed(
        out_path,
        beta=beta,
        acc=acc_store,
        seed=seed,
        logmin=logmin,
        logmax=logmax,
        n_side=n_side,
        p=p,
        eirat=eirat,
        t_on=t_on,
        t_off=t_off,
        dt=dt,
        batch_size=batch_size,
        alpha=alpha,
    )

    return beta, acc_store


if __name__ == "__main__":
    from argparse import ArgumentParser

    argparser = ArgumentParser()
    argparser.add_argument("--n-side", type=int, default=20)
    argparser.add_argument("--p", type=float, default=0.07)
    argparser.add_argument("--eirat", type=float, default=0.2)
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--t-on", type=float, default=10.0)
    argparser.add_argument("--t-off", type=float, default=0.0)
    argparser.add_argument("--dt", type=float, default=0.1)
    argparser.add_argument("--batch-size", type=int, default=256)
    argparser.add_argument("--alpha", type=float, default=1e-4)
    argparser.add_argument("--numpoints", type=int, default=30)
    argparser.add_argument("--logmin", type=float, default=-4.0)
    argparser.add_argument("--logmax", type=float, default=0.0)
    argparser.add_argument("--max-batches-train", type=int, default=None)
    argparser.add_argument("--max-batches-test", type=int, default=None)

    args = argparser.parse_args()

    sweep_beta_mnist(
        numpoints=args.numpoints,
        seed=args.seed,
        n_side=args.n_side,
        p=args.p,
        eirat=args.eirat,
        t_on=args.t_on,
        t_off=args.t_off,
        dt=args.dt,
        batch_size=args.batch_size,
        alpha=args.alpha,
        logmin=args.logmin,
        logmax=args.logmax,
        max_batches_train=args.max_batches_train,
        max_batches_test=args.max_batches_test,
    )
